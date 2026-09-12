# -*- coding: utf-8 -*-
"""辅助动力学分析模块（设计大纲第 20 章）。

均方位移 MSD：

.. math::

    MSD(t) = \\langle |\\mathbf r(t) - \\mathbf r(0)|^2 \\rangle

MSD 在本项目中作为辅助分析，可用于比较原子运动、分子运动、链段运动、
界面区域运动差异以及结晶前后的运动变化。

因为要统计粒子的真实位移，必须跨越周期性边界追踪粒子，所以本模块使用
``NoJump`` 变换消除帧间的坐标跳变，而不是只做帧内的分子展开。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..core import AnalysisResult, Panel
from ..preprocess import FrameSelection
from .base import register

__all__ = ["compute_msd", "analyze_msd", "diffusion_coefficient"]

#: Å²/ps -> m²/s
A2_PER_PS_TO_M2_PER_S = 1e-8


def unwrap_over_frames(mdt, ag, frame_indices, *, verbose: bool = False):
    """跨帧追踪原子，返回连续位移坐标 ``(n_frames, n_atoms, 3)``。

    做法：对相邻两帧计算**最小镜像位移**并累加，从而跨越周期性边界追踪粒子。
    这比 ``NoJump`` 更可控：本函数会统计"位移超过最小盒高度一半"的原子比例，
    该比例就是最小镜像近似可能失效的比例，作为结果可靠性诊断量返回。

    返回 ``(positions, diagnostics)``。
    """
    from MDAnalysis.lib.distances import minimize_vectors

    u = mdt.universe
    idx = np.asarray(frame_indices, dtype=int)
    n = idx.size
    pos = np.full((n, ag.n_atoms, 3), np.nan, dtype=float)
    n_big = 0
    n_tot = 0
    prev = None
    unw = None
    for k, i in enumerate(idx):
        u.trajectory[int(i)]
        box = u.dimensions
        p = np.asarray(ag.positions, dtype=float).copy()
        if prev is None:
            unw = p.copy()
        else:
            d = p - prev
            if box is not None:
                d = minimize_vectors(d, box)
                # 诊断：位移是否超过最小盒高度的一半
                h = float(np.min(_box_heights_from_dims(box)))
                mag = np.linalg.norm(d, axis=1)
                n_big += int(np.sum(mag > 0.5 * h))
                n_tot += mag.size
            unw = unw + d
        pos[k] = unw
        prev = p
    diag = {
        "最小镜像近似失效原子数": int(n_big),
        "检查的原子-帧数": int(n_tot),
        "失效比例": float(n_big / n_tot) if n_tot else 0.0,
    }
    return pos, diag


def _box_heights_from_dims(dims) -> np.ndarray:
    from MDAnalysis.lib.mdamath import triclinic_vectors

    m = np.asarray(triclinic_vectors(np.asarray(dims, dtype=float)), dtype=float)
    v = float(abs(np.linalg.det(m)))
    h = np.empty(3, dtype=float)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        area = float(np.linalg.norm(np.cross(m[j], m[k])))
        h[i] = v / area if area > 0 else np.inf
    return h


def compute_msd(mdt, ag, selection: FrameSelection, *, max_lag: int | None = None,
                verbose: bool = False):
    """计算原子组的 MSD。

    返回 ``(lag_times_ps, msd_Å², diagnostics)``。

    算法：先用最小镜像位移把轨迹累加成连续轨迹（跨越周期性边界追踪粒子），
    再对所有 ``(t₀, t₀+τ)`` 组合与所有原子求平均。
    """
    idx = np.asarray(selection.indices, dtype=int)
    if idx.size < 2:
        raise ValueError("MSD 至少需要 2 帧")

    pos, diag = unwrap_over_frames(mdt, ag, idx, verbose=verbose)
    times = np.asarray(selection.times_ps, dtype=float)
    n = idx.size
    n_lag = n if max_lag is None else min(n, int(max_lag) + 1)
    msd = np.full(n_lag, np.nan, dtype=float)
    for m in range(n_lag):
        if m == 0:
            msd[m] = 0.0
            continue
        d = pos[m:] - pos[:n - m]
        msd[m] = float(np.nanmean(np.sum(d * d, axis=2)))
    lag_t = times[:n_lag] - times[0]
    return lag_t, msd, diag


def diffusion_coefficient(lag_t: np.ndarray, msd: np.ndarray,
                          fit_fraction: tuple[float, float] = (0.3, 1.0),
                          dim: int = 3) -> dict:
    """由 MSD 的线性区拟合扩散系数。

    ``MSD = 2·dim·D·t``（三维时 ``MSD = 6Dt``）。返回 ``D``（Å²/ps 与 m²/s）、
    拟合区间、对数-对数斜率 ``α``（MSD ∝ t^α）与一条可靠性判断。

    物理约束
    --------
    MSD 在真实体系里是**单调不减**的；扩散系数绝不能为负。如果拟合斜率 ≤ 0，
    说明该组分的 MSD 是一条平坦的曲线——粒子基本不动（玻璃态/被约束/结晶相），
    或者帧间隔太大导致曲线不具扩散性。这时返回 ``Nan`` 并说明原因，
    而不是给出一个看似精确、实则无意义的负数。
    """
    t = np.asarray(lag_t, dtype=float)
    y = np.asarray(msd, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & (t > 0)
    out = {"D (Å²/ps)": float("nan"), "D (m²/s)": float("nan"),
           "拟合区间 (ps)": "", "R²": float("nan"), "α (log-log 斜率)": float("nan"),
           "拟合可靠性": ""}
    if ok.sum() < 3:
        out["拟合可靠性"] = "点太少，无法拟合"
        return out
    t_ok, y_ok = t[ok], y[ok]
    lo = fit_fraction[0] * t_ok[-1]
    hi = fit_fraction[1] * t_ok[-1]
    m = (t_ok >= lo) & (t_ok <= hi)
    if m.sum() < 3:
        m = np.ones_like(t_ok, dtype=bool)
    tt, yy = t_ok[m], y_ok[m]
    slope, intercept = np.polyfit(tt, yy, 1)
    pred = slope * tt + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out["拟合区间 (ps)"] = f"{tt[0]:.1f} – {tt[-1]:.1f}"
    out["R²"] = r2
    pos = (t_ok > 0) & (y_ok > 0)
    if pos.sum() >= 3:
        out["α (log-log 斜率)"] = float(np.polyfit(np.log(t_ok[pos]),
                                                   np.log(y_ok[pos]), 1)[0])

    if slope <= 0:
        out["拟合可靠性"] = ("MSD 曲线平坦或下降，粒子基本没有扩散"
                             "（玻璃态/被约束/结晶相，或帧间隔过大）——"
                             "不给出扩散系数")
        return out

    d_a2ps = float(slope) / (2.0 * dim)
    out["D (Å²/ps)"] = d_a2ps
    out["D (m²/s)"] = d_a2ps * A2_PER_PS_TO_M2_PER_S
    alpha = out["α (log-log 斜率)"]
    if np.isfinite(r2) and r2 < 0.5:
        out["拟合可靠性"] = "拟合优度低（R²<0.5），D 只能作定性参考"
    elif np.isfinite(alpha) and not (0.8 <= alpha <= 1.2):
        out["拟合可靠性"] = (f"MSD∝t^{alpha:.2f} 偏离扩散标度（α≈1），"
                             f"D 只能作定性参考")
    else:
        out["拟合可靠性"] = "良好（线性区拟合正常）"
    return out


@register("msd", "均方位移 MSD")
def analyze_msd(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                fit_fraction: tuple[float, float] = (0.3, 1.0),
                dim: int = 3, verbose: bool = False) -> AnalysisResult:
    """MSD 辅助分析。

    参数
    ----
    groups
        ``{组分名: AtomGroup}``，可一次比较多个组分（例如界面区 vs 体相区）。
    fit_fraction
        线性区拟合区间（相对最大 lag 时间的比例）。
    dim
        维度，三维取 3（``MSD = 6Dt``）。
    """
    if not groups:
        raise ValueError("必须提供至少一个组分")

    res = AnalysisResult(
        name="msd",
        title="均方位移 MSD",
        meta={"n_frames": int(len(selection.indices)),
              "fit_fraction": list(map(float, fit_fraction)), "dim": int(dim)},
    )
    res.panels = [
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)", title="MSD 随时间间隔变化",
              xscale="log", yscale="log"),
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)", title="MSD 线性区拟合"),
    ]

    fail_by_group: dict[str, float] = {}
    for name, ag in groups.items():
        if ag is None or ag.n_atoms == 0:
            continue
        lag_t, msd, diag = compute_msd(mdt, ag, selection, max_lag=None, verbose=verbose)
        fit = diffusion_coefficient(lag_t, msd, fit_fraction=fit_fraction, dim=dim)
        fail_by_group[name] = float(diag.get("失效比例", 0.0) or 0.0)
        res.add_curve(f"{name} ({ag.n_atoms} 原子)", lag_t, msd, panel=0)

        # 线性区的拟合直线
        t = np.asarray(lag_t, dtype=float)
        ok = np.isfinite(t) & np.isfinite(msd) & (t > 0)
        if ok.sum() >= 3:
            t_ok, y_ok = t[ok], msd[ok]
            m = ((t_ok >= fit_fraction[0] * t_ok[-1])
                 & (t_ok <= fit_fraction[1] * t_ok[-1]))
            if m.sum() < 3:
                m = np.ones_like(t_ok, dtype=bool)
            slope = np.polyfit(t_ok[m], y_ok[m], 1)
            res.add_curve(f"{name} 线性拟合", t_ok[m], np.polyval(slope, t_ok[m]),
                          panel=1)
        res.add_curve(f"{name}", lag_t, msd, panel=1)

        for k, v in fit.items():
            res.summary[f"{name} {k}"] = v
        res.summary[f"{name} 原子数"] = int(ag.n_atoms)
        res.summary[f"{name} MSD(最大 τ) (Å²)"] = float(msd[-1]) if msd.size else float("nan")
        fail = float(diag.get("失效比例", 0.0) or 0.0)
        res.summary[f"{name} 最小镜像失效比例"] = fail
        # 最小镜像追踪的可靠性必须并入总判定：失效比例高时，被错误回绕的原子贡献的
        # 是**随机**大位移，会让 MSD 看起来依然线性（R² 甚至接近 1）却整体失真。
        # 因此单看 R²/α 会给出过于乐观的结论，这里显式降级。
        if fail > 0.05:
            base = str(res.summary.get(f"{name} 拟合可靠性", "") or "")
            if fail > 0.20:
                tag = (f"最小镜像失效比例 {fail * 100:.0f}%（严重）——"
                       f"帧间隔内多数原子的位移已超过最小盒高的一半，"
                       f"跨周期追踪不可靠，D 很可能被污染，**不建议引用**；"
                       f"请用更密的帧间隔（该体系建议 ≤ 数百 ps）重算")
            else:
                tag = (f"最小镜像失效比例 {fail * 100:.1f}%，D 可能偏离真值，"
                       f"建议缩小帧间隔复核")
            res.summary[f"{name} 拟合可靠性"] = f"{base}；{tag}" if base else tag
        if ag.n_atoms < 20:
            res.add_notes(
                f"{name} 只有 {ag.n_atoms} 个原子，其扩散系数与 MSD 曲线的统计误差"
                f"很大，只能作为定性参考。"
            )

    bad_fail = [n for n, f in fail_by_group.items() if f > 0.20]
    res.add_notes("MSD = <|r(t+τ) − r(t)|²>，对全部时间起点与全部原子求平均。")
    res.add_notes("跨周期性边界通过「最小镜像位移累加」追踪粒子，相当于把分子在时间上解开。")
    res.add_notes("若「最小镜像失效比例」明显大于 0，说明帧间隔过大，"
                  "粒子真实位移超过最小盒高度的一半，MSD 会被低估。")
    res.add_notes("注意：最小镜像失效带来的误差是**随机**的，它会让 MSD 曲线依然"
                  "看起来平滑、线性（R² 甚至接近 1），但绝对数值已被污染。"
                  "因此失效比例高时不能只看 R² 与 α 判断 D 是否可信。")
    if bad_fail:
        res.add_notes(
            "以下组分的「最小镜像失效比例」超过 20%，其 D 不应直接引用："
            + "、".join(bad_fail)
            + "。建议用更密的输出间隔重新计算 MSD。")
    res.add_notes(f"扩散系数由 MSD = 2·dim·D·t（dim={dim}）的线性区斜率得到。")

    # 明确提示哪些组分的 D 不可用（MSD 平坦 = 粒子没扩散）
    bad = [n for n in groups
           if not np.isfinite(res.summary.get(f"{n} D (Å²/ps)", float("nan")))]
    if bad:
        res.add_notes("以下组分的 MSD 曲线平坦或下降，粒子基本没有扩散，"
                      "因此不给扩散系数（D = nan）：" + "、".join(bad)
                      + "。这通常意味着该组分处于玻璃态/被约束/结晶相，"
                        "也可能是帧间隔太大导致曲线不具扩散性。")
    return res
