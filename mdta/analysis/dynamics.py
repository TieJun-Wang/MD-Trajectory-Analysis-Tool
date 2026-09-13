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

__all__ = ["compute_msd", "compute_msd_axes", "particle_series", "msd_fft",
           "analyze_msd", "diffusion_coefficient"]

#: Å²/ps -> m²/s
A2_PER_PS_TO_M2_PER_S = 1e-8

#: 粒子坐标序列的内存上限（字节）。FFT 求 MSD 需要把整条序列放进内存，
#: 粒子数超过预算时按等间隔抽稀（会在结果说明里写明抽稀比例）。
DEFAULT_MEMORY_BUDGET = 1.2e9


def _molecule_ids(ag) -> np.ndarray | None:
    """原子级分子编号；无 ``molnums`` 时退回键图碎片，都没有则返回 ``None``。"""
    from .interface import _atom_molnums

    m = _atom_molnums(ag)
    if m is not None:
        return m
    try:
        if len(ag.bonds) == 0:
            return None
        allidx = np.asarray(ag.indices, dtype=np.int64)
        m = np.full(ag.n_atoms, -1, dtype=np.int64)
        for k, frag in enumerate(ag.fragments):
            m[np.searchsorted(allidx, np.asarray(frag.indices, dtype=np.int64))] = k
        return m if bool((m >= 0).all()) else None
    except Exception:  # noqa: BLE001
        return None


def _safe_masses(ag) -> np.ndarray:
    """质量数组；缺失或非正时退回单位质量（几何中心），并在调用方注明。"""
    try:
        w = np.asarray(ag.masses, dtype=float)
        if w.size == ag.n_atoms and bool(np.all(np.isfinite(w))) and bool(np.all(w > 0)):
            return w
    except Exception:  # noqa: BLE001
        pass
    return np.ones(ag.n_atoms, dtype=float)


def _frame_com(ag, mol, n_mol, box, weights) -> np.ndarray:
    """当前帧的**分子质心**坐标 ``(n_mol, 3)``。

    先按最小镜像把分子内的原子聚拢到该分子的第 1 个原子上，再加权平均。
    必须先聚拢：跨越周期边界的分子，其原子坐标分散在盒子的两侧，直接求质心
    会得到分子外部的一个假位置。
    """
    pos = np.asarray(ag.positions, dtype=float)
    first = np.full(n_mol, -1, dtype=np.int64)
    uniq, idx0 = np.unique(mol, return_index=True)
    first[uniq] = idx0
    ref = pos[first[mol]]
    if box is not None:
        try:
            from MDAnalysis.lib.distances import minimize_vectors

            pos = ref + minimize_vectors(pos - ref, np.asarray(box, dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass
    num = np.zeros((n_mol, 3), dtype=float)
    for d in range(3):
        num[:, d] = np.bincount(mol, weights=weights * pos[:, d], minlength=n_mol)
    den = np.bincount(mol, weights=weights, minlength=n_mol)
    den[den <= 0] = 1.0
    return num / den[:, None]


def particle_series(mdt, ag, frame_indices, *, object="molecule",
                    max_particles: int | None = None,
                    memory_budget: float = DEFAULT_MEMORY_BUDGET,
                    verbose: bool = False):
    """把原子组变成逐帧的"被追踪粒子"坐标序列 ``(n_frames, n_particles, 3)``。

    ``object="molecule"``（默认）→ 每个**分子质心**是一个粒子（质量加权）；
    ``object="atom"`` → 每个原子是一个粒子。

    返回 ``(series, boxes, meta)``；``boxes`` 是逐帧的 ``dimensions``。
    """
    u = mdt.universe
    idx = np.asarray(frame_indices, dtype=int)
    n_frame = int(idx.size)
    obj = str(object).lower()
    use_com = obj.startswith("mol") or obj in ("com", "分子", "分子质心")
    notes: list[str] = []

    # 内存预算：FFT 需要整条序列驻留内存，据此反推粒子数上限
    if memory_budget and n_frame > 0:
        cap = max(1, int(memory_budget / (n_frame * 3 * 8)))
        max_particles = cap if not max_particles else min(int(max_particles), cap)

    mol = _molecule_ids(ag) if use_com else None
    if use_com and mol is None:
        use_com = False
        notes.append("该组分没有分子编号（molnums）也无法从键图分组，"
                     "本轮退化为**按原子**追踪。")

    if use_com:
        uniq = np.unique(mol)
        n_part_all = int(uniq.size)
        keep = uniq
        if max_particles and n_part_all > max_particles:
            step = int(np.ceil(n_part_all / float(max_particles)))
            keep = uniq[::step]
            notes.append(f"分子数 {n_part_all:,} 超过上限 {int(max_particles):,}，"
                         f"每 {step} 个分子取 1 个，抽稀到 {keep.size:,} 个。")
        mask = np.isin(mol, keep)
        ag_use = ag[mask]
        _, inv = np.unique(mol[mask], return_inverse=True)
        mol_use = np.asarray(inv, dtype=np.int64)
        n_part = int(keep.size)
        weights = _safe_masses(ag_use)
        try:
            w_chk = np.asarray(ag_use.masses, dtype=float)
            has_mass = (w_chk.size == ag_use.n_atoms
                        and bool(np.all(np.isfinite(w_chk))) and bool(np.all(w_chk > 0)))
        except Exception:  # noqa: BLE001
            has_mass = False
        if not has_mass:
            notes.append("该组分没有质量信息，质心退化为几何中心。")
    else:
        n_part_all = int(ag.n_atoms)
        if max_particles and n_part_all > max_particles:
            step = int(np.ceil(n_part_all / float(max_particles)))
            ag_use = ag[::step]
            notes.append(f"原子数 {n_part_all:,} 超过上限 {int(max_particles):,}，"
                         f"每 {step} 个取 1 个，抽稀到 {ag_use.n_atoms:,} 个。")
        else:
            ag_use = ag
        mol_use = None
        n_part = int(ag_use.n_atoms)
        weights = None

    series = np.empty((n_frame, n_part, 3), dtype=float)
    boxes: list = []
    for k, i in enumerate(idx):
        u.trajectory[int(i)]
        box = u.dimensions
        boxes.append(None if box is None else np.asarray(box, dtype=float))
        if use_com:
            series[k] = _frame_com(ag_use, mol_use, n_part, box, weights)
        else:
            series[k] = np.asarray(ag_use.positions, dtype=float)

    meta = {"object": "molecule" if use_com else "atom",
            "n_particles": n_part, "n_particles_total": n_part_all,
            "n_frames": n_frame, "notes": notes}
    if verbose:
        print(f"  [MSD] 追踪对象={meta['object']}，粒子 {n_part:,}，帧 {n_frame:,}")
    return series, boxes, meta


def _unwrap_series(series: np.ndarray, boxes, *, verbose: bool = False):
    """跨帧追踪粒子：对相邻帧做**最小镜像位移**累加，返回连续轨迹与诊断。

    比 ``NoJump`` 更可控的地方是它会统计"位移超过最小盒高度一半"的粒子比例，
    该比例即最小镜像近似可能失效的比例——这种误差是**随机**的，会让 MSD 曲线
    依然平滑线性（R² 甚至接近 1）却整体失真，所以必须显式报出来。
    """
    from MDAnalysis.lib.distances import minimize_vectors

    n = int(series.shape[0])
    out = np.empty_like(series)
    if n == 0:
        return out, {"最小镜像近似失效粒子数": 0, "检查的粒子-帧数": 0, "失效比例": 0.0}
    out[0] = series[0]
    n_big = 0
    n_tot = 0
    for k in range(1, n):
        d = series[k] - series[k - 1]
        box = boxes[k] if k < len(boxes) else None
        if box is not None and bool(np.all(np.asarray(box[:3], dtype=float) > 0)):
            d = minimize_vectors(d, np.asarray(box, dtype=np.float32))
            h = float(np.min(_box_heights_from_dims(box)))
            mag = np.linalg.norm(d, axis=1)
            n_big += int(np.sum(mag > 0.5 * h))
            n_tot += mag.size
        out[k] = out[k - 1] + d
    diag = {
        "最小镜像近似失效粒子数": int(n_big),
        "检查的粒子-帧数": int(n_tot),
        "失效比例": float(n_big / n_tot) if n_tot else 0.0,
    }
    return out, diag


def msd_fft(series: np.ndarray) -> np.ndarray:
    """``(n_frames, n_particles)`` → MSD(τ)，对全部时间起点与全部粒子求平均。

    用快速相关算法（FFT）算，复杂度 ``O(N log N)``，**保留完整时间分辨率**
    （不做分块、不牺牲长 τ）。

    为什么不用 ``tidynamics.msd``：它把输入的第二轴当作**空间分量**求和
    （源码里逐列 ``autocorrelation_1d`` 后相加，``rsq = Σ_d pos²``），因此只
    能算"单个粒子、多分量"的 MSD，无法一次对几百上千个分子求平均；逐粒子
    调用又需要 P 次 FFT。这里用同一套快速相关算法，但把 FFT 沿**时间轴批量**
    做一次算完所有粒子。数值上与暴力 ``(t₀, τ)`` 双重求和逐点一致（selftest
    与 tidynamics、暴力法双向校验）。

    估计量与暴力法相同（无偏，按实际样本数 ``N−τ`` 平均）：

    .. math:: \\mathrm{MSD}(\\tau) = \\frac{1}{N-\\tau}\\sum_{t=0}^{N-\\tau-1}
              |\\mathbf r_{t+\\tau}-\\mathbf r_t|^2
    """
    A = np.ascontiguousarray(series, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    n = int(A.shape[0])
    if n < 2:
        return np.zeros(n, dtype=float)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(A, n=n_fft, axis=0)
    corr = np.fft.irfft(F * np.conjugate(F), n=n_fft, axis=0)[:n]   # Σ_t A_t A_{t+τ}
    sq = A * A
    cs = np.cumsum(sq, axis=0)
    tot = cs[-1]
    tau = np.arange(n)
    p2 = cs[n - 1 - tau]                                            # Σ_{t≤N-τ-1} A_t²
    p1 = tot - np.concatenate([np.zeros((1, A.shape[1])), cs[:-1]], axis=0)
    num = p1 + p2 - 2.0 * corr
    den = (n - tau).astype(float) * float(A.shape[1])
    return num.sum(axis=1) / den


def unwrap_over_frames(mdt, ag, frame_indices, *, verbose: bool = False):
    """兼容入口：跨帧追踪原子，返回 ``(连续坐标, 诊断)``。

    现在内部统一走 :func:`particle_series` + :func:`_unwrap_series`（按原子）。
    """
    series, boxes, _ = particle_series(mdt, ag, frame_indices, object="atom",
                                       verbose=verbose)
    return _unwrap_series(series, boxes, verbose=verbose)


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


def compute_msd_axes(mdt, ag, selection: FrameSelection, *, object: str = "molecule",
                     remove_drift: bool = True, max_particles: int | None = None,
                     max_lag: int | None = None, n_error_blocks: int = 5,
                     verbose: bool = False):
    """逐轴 MSD：返回 ``(lag_t, msd_dict, diag)``。

    ``msd_dict`` 的键：``total``（x+y+z）、``x`` / ``y`` / ``z``、
    ``para``（=x+y，平行于 z 轴的面内分量）、``perp``（=z）。

    为什么要把各轴分开：只有分子在某个方向上**取向有序**（界面吸附、拉伸、
    受限孔道）时，MSD 才会各向异性；``D∥/D⊥`` 是这类体系最直接的可观测量，
    而单一的 MSD 总值会把它平均掉。

    ``remove_drift``：减去全部粒子的平均位移轨迹（消除整体漂移，例如 NPT
    下整盒的缓慢平移）。粒子数 < 4 时不做（少数粒子时"平均位移"本身就是
    被测对象，减掉会把信号一起减没）。
    """
    idx = np.asarray(selection.indices, dtype=int)
    if idx.size < 2:
        raise ValueError("MSD 至少需要 2 帧")

    series, boxes, meta = particle_series(mdt, ag, idx, object=object,
                                          max_particles=max_particles,
                                          verbose=verbose)
    unwrapped, diag = _unwrap_series(series, boxes, verbose=verbose)
    drift_removed = False
    if remove_drift and unwrapped.shape[1] >= 4:
        unwrapped = unwrapped - unwrapped.mean(axis=1, keepdims=True)
        drift_removed = True

    msd = {k: msd_fft(unwrapped[:, :, i]) for i, k in enumerate(("x", "y", "z"))}
    msd["total"] = msd["x"] + msd["y"] + msd["z"]
    msd["para"] = msd["x"] + msd["y"]
    msd["perp"] = msd["z"]

    times = np.asarray(selection.times_ps, dtype=float)
    n_lag = times.size if max_lag is None else min(times.size, int(max_lag) + 1)
    lag_t = times[:n_lag] - times[0]
    for k in list(msd):
        msd[k] = np.asarray(msd[k][:n_lag], dtype=float)

    # ---- 误差分块：把时间轴切块，每块**独立**算一条 MSD。
    # 扩散系数的标准误由这些块的 D 散布给出（经验值），比"假设一个残差相关矩阵"
    # 更可靠。块要足够长才落在同一个拟合窗口里，所以块数按帧数自适应。
    block_curves: dict[str, list] = {k: [] for k in ("total", "para", "perp")}
    n_blk = int(n_error_blocks or 0)
    if n_blk >= 2 and unwrapped.shape[0] >= 8:
        edges = np.linspace(0, unwrapped.shape[0], n_blk + 1).astype(int)
        for b in range(n_blk):
            lo_b, hi_b = int(edges[b]), int(edges[b + 1])
            if hi_b - lo_b < 4:
                continue
            sub = unwrapped[lo_b:hi_b]
            mb = {k: msd_fft(sub[:, :, i]) for i, k in enumerate(("x", "y", "z"))}
            mb["total"] = mb["x"] + mb["y"] + mb["z"]
            mb["para"] = mb["x"] + mb["y"]
            mb["perp"] = mb["z"]
            n_b = int(mb["total"].size)
            bt = lag_t[:n_b]
            for k in block_curves:
                block_curves[k].append((bt, np.asarray(mb[k][:n_b], dtype=float)))

    diag = dict(diag)
    diag["误差分块_curves"] = block_curves
    diag["误差分块数"] = len(block_curves.get("total", []))
    diag.update({"对象": meta["object"], "粒子数": int(meta["n_particles"]),
                 "粒子数_总": int(meta["n_particles_total"]),
                 "去漂移": bool(drift_removed), "算法": "FFT 快速相关",
                 "说明": list(meta["notes"])})
    return lag_t, msd, diag


def compute_msd(mdt, ag, selection: FrameSelection, *, max_lag: int | None = None,
                object: str = "atom", remove_drift: bool = False,
                max_particles: int | None = None, verbose: bool = False):
    """计算原子组的总 MSD，返回 ``(lag_times_ps, msd_Å², diagnostics)``。

    默认仍按**原子**追踪（保持既有调用方的语义）；``analyze_msd`` 默认按
    **分子质心**追踪，那才是"分子扩散"的正确对象。
    """
    lag_t, msd, diag = compute_msd_axes(mdt, ag, selection, object=object,
                                        remove_drift=remove_drift,
                                        max_particles=max_particles,
                                        max_lag=max_lag, verbose=verbose)
    return lag_t, msd["total"], diag


def _fit_slope_with_error(t, y, method: str = "gls"):
    """线性拟合斜率并给出**标准误**，返回 ``(slope, intercept, se, r2, rho)``。

    为什么不能只用 ``np.polyfit(cov=True)``：那个误差假设各点残差独立，而 MSD 在
    不同 τ 上是**同一段轨迹算出来的**，强相关，因此会系统性**低估**误差。

    ``method="gls"`` 时用**本曲线自己估出来的**残差滞后 1 自相关系数 ρ 做有效样本量
    修正：``se_eff = se_ols·sqrt((1+ρ)/(1−ρ))``（ρ=0 时退化为 OLS），并把 ρ 一起报出来。
    强调"估出来"是因为 MDTransport 那类实现是**假设**一个相关矩阵（如
    ``2^{-|i-j|/2}``）；从数据估计至少是自洽的，而且 ρ 可被检查。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    dof = max(n - 2, 1)
    sxx = float(np.sum((t - t.mean()) ** 2))
    se = float(np.sqrt(float(np.sum(resid ** 2)) / dof / sxx)) if sxx > 0 else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    r0 = resid - resid.mean()
    den = float(np.sum(r0 * r0))
    rho = float(np.clip(np.sum(r0[1:] * r0[:-1]) / den, -0.99, 0.99)) if den > 0 else 0.0
    if str(method).lower() == "gls" and rho > 0:
        se *= float(np.sqrt((1.0 + rho) / (1.0 - rho)))
    return float(slope), float(intercept), float(se), float(r2), rho


def diffusion_coefficient(lag_t: np.ndarray, msd: np.ndarray,
                          fit_fraction: tuple[float, float] = (0.3, 1.0),
                          dim: int = 3, fit_method: str = "gls",
                          block_curves=None) -> dict:
    """由 MSD 的线性区拟合扩散系数（含**标准误**）。

    ``MSD = 2·dim·D·t``（三维时 ``MSD = 6Dt``）。返回 ``D``（Å²/ps 与 m²/s）、
    标准误、拟合区间、对数-对数斜率 ``α``（MSD ∝ t^α）与可靠性判断。

    标准误怎么来
    ------------
    1. **分块平均（首选）**：把轨迹切成若干时间块，每块**独立**算一条 MSD 并各自
       拟合，取各块 D 的标准误 ``std(D_i)/√n``。这是**经验散布**，不依赖任何模型假设。
    2. 退路（块数 < 3，例如轨迹太短）：拟合协方差（``fit_method="gls"`` 时再乘实测
       AR(1) 修正）。此时误差偏乐观，结果里会写明来源。

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
           "D 标准误 (m²/s)": float("nan"), "D 相对标准误": float("nan"),
           "标准误来源": "", "残差 AR(1) ρ": float("nan"),
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
    slope, _intercept, se_slope, r2, rho = _fit_slope_with_error(tt, yy, fit_method)
    out["拟合区间 (ps)"] = f"{tt[0]:.1f} – {tt[-1]:.1f}"
    out["R²"] = r2
    out["残差 AR(1) ρ"] = rho
    pos = (t_ok > 0) & (y_ok > 0)
    if pos.sum() >= 3:
        out["α (log-log 斜率)"] = float(np.polyfit(np.log(t_ok[pos]),
                                                   np.log(y_ok[pos]), 1)[0])

    if slope <= 0:
        out["拟合可靠性"] = ("MSD 曲线平坦或下降，粒子基本没有扩散"
                             "（玻璃态/被约束/结晶相，或帧间隔过大）——"
                             "不给出扩散系数")
        out["标准误来源"] = "未拟合（斜率 ≤ 0）"
        return out

    d_a2ps = float(slope) / (2.0 * dim)
    out["D (Å²/ps)"] = d_a2ps
    out["D (m²/s)"] = d_a2ps * A2_PER_PS_TO_M2_PER_S

    # ---- 标准误：优先用分块的经验散布
    d_blocks: list[float] = []
    for item in (block_curves or []):
        try:
            bt, by = np.asarray(item[0], dtype=float), np.asarray(item[1], dtype=float)
        except Exception:  # noqa: BLE001
            continue
        m2 = np.isfinite(bt) & np.isfinite(by) & (bt >= lo) & (bt <= hi)
        if m2.sum() < 3:
            continue
        s_i = _fit_slope_with_error(bt[m2], by[m2], method="ols")[0]
        if np.isfinite(s_i):
            d_blocks.append(float(s_i) / (2.0 * dim))
    se_D = float("nan")
    if len(d_blocks) >= 3:
        arr = np.asarray(d_blocks, dtype=float)
        se_D = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        out["分块 D 个数"] = int(arr.size)
        out["标准误来源"] = f"分块平均（{arr.size} 块各自独立拟合）"
    elif np.isfinite(se_slope):
        se_D = float(se_slope) / (2.0 * dim)
        out["标准误来源"] = ("拟合协方差 × 实测 AR(1) 修正（分块不足 3 块，误差偏乐观）"
                        if (str(fit_method).lower() == "gls" and rho > 0)
                        else "拟合协方差（残差独立假设，会低估；分块不足 3 块）")
    if np.isfinite(se_D):
        out["D 标准误 (m²/s)"] = se_D * A2_PER_PS_TO_M2_PER_S
        out["D 相对标准误"] = float(se_D / d_a2ps) if d_a2ps > 0 else float("nan")

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
                dim: int = 3, object: str = "molecule",
                remove_drift: bool = True,
                max_particles: int | None = None,
                fit_method: str = "gls",
                n_error_blocks: int = 5,
                verbose: bool = False) -> AnalysisResult:
    """均方位移 MSD（默认追踪对象：**分子质心**）。

    参数
    ----
    groups
        ``{组分名: AtomGroup}``，可一次比较多个组分（例如界面区 vs 体相区）。
    object
        ``"molecule"``（默认）→ 每个分子取其**质心**作为一个被追踪粒子，
        得到的是"分子的平动扩散"（质量加权，跨周期边界先聚拢再求质心）；
        ``"atom"`` → 每个原子各算一个粒子，MSD 里会混进分子内部振动/转动。
        组分没有分子信息时自动退化为原子并注明。
    remove_drift
        是否减去全体粒子的平均位移轨迹（消除整体漂移）。粒子数 < 4 时不做。
    fit_fraction
        线性区拟合区间（相对最大 lag 时间的比例）。
    dim
        总 MSD 的拟合维度，三维取 3（``MSD = 6Dt``）；``D∥`` 用 2、``D⊥`` 用 1。
    """
    if not groups:
        raise ValueError("必须提供至少一个组分")

    res = AnalysisResult(
        name="msd",
        title="均方位移 MSD",
        meta={"n_frames": int(len(selection.indices)),
              "fit_fraction": list(map(float, fit_fraction)), "dim": int(dim),
              "object": str(object), "remove_drift": bool(remove_drift)},
    )
    res.panels = [
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)", title="MSD 随时间间隔变化",
              xscale="log", yscale="log"),
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)", title="MSD 线性区拟合"),
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)",
              title="各轴分解与各向异性（x/y/z 及 ∥、⊥）"),
    ]

    fail_by_group: dict[str, float] = {}
    for name, ag in groups.items():
        if ag is None or ag.n_atoms == 0:
            continue
        lag_t, msd, diag = compute_msd_axes(mdt, ag, selection, object=object,
                                            remove_drift=remove_drift,
                                            max_particles=max_particles,
                                            n_error_blocks=n_error_blocks,
                                            verbose=verbose)
        bl = diag.get("误差分块_curves") or {}
        if diag.get("粒子数", 0) < 2:
            res.add_notes(
                f"{name} 只追踪到 {diag.get('粒子数')} 个粒子，MSD 没有集合平均，"
                f"结果只是单条轨迹的位移平方，统计意义有限。")
        fit = diffusion_coefficient(lag_t, msd["total"], fit_fraction=fit_fraction,
                                    dim=dim, fit_method=fit_method,
                                    block_curves=bl.get("total"))
        fit_para = diffusion_coefficient(lag_t, msd["para"], fit_fraction=fit_fraction,
                                         dim=2, fit_method=fit_method,
                                         block_curves=bl.get("para"))
        fit_perp = diffusion_coefficient(lag_t, msd["perp"], fit_fraction=fit_fraction,
                                         dim=1, fit_method=fit_method,
                                         block_curves=bl.get("perp"))
        fail_by_group[name] = float(diag.get("失效比例", 0.0) or 0.0)

        obj_label = "分子质心" if diag.get("对象") == "molecule" else "原子"
        res.add_curve(f"{name} ({obj_label}, {diag.get('粒子数', 0):,} 个)",
                      lag_t, msd["total"], panel=0)

        # 线性区的拟合直线（沿用旧口径，画在同一张图上便于目视检查）
        t = np.asarray(lag_t, dtype=float)
        y_tot = np.asarray(msd["total"], dtype=float)
        ok = np.isfinite(t) & np.isfinite(y_tot) & (t > 0)
        if ok.sum() >= 3:
            t_ok, y_ok = t[ok], y_tot[ok]
            m = ((t_ok >= fit_fraction[0] * t_ok[-1])
                 & (t_ok <= fit_fraction[1] * t_ok[-1]))
            if m.sum() < 3:
                m = np.ones_like(t_ok, dtype=bool)
            slope = np.polyfit(t_ok[m], y_ok[m], 1)
            res.add_curve(f"{name} 线性拟合", t_ok[m], np.polyval(slope, t_ok[m]),
                          panel=1)
        res.add_curve(f"{name}", lag_t, y_tot, panel=1)

        # 各轴分解：各向异性是界面/受限体系最直接的可观测量
        for key, clabel in (("x", f"{name} MSD_x"), ("y", f"{name} MSD_y"),
                            ("z", f"{name} MSD_z"), ("para", f"{name} MSD∥ (x+y)"),
                            ("perp", f"{name} MSD⊥ (z)")):
            res.add_curve(clabel, lag_t, msd[key], panel=2)

        for k, v in fit.items():
            res.summary[f"{name} {k}"] = v
        if np.isfinite(fit.get("D 标准误 (m²/s)", np.nan)):
            res.add_notes(
                f"{name}：D = {fit['D (m²/s)']:.3g} ± {fit['D 标准误 (m²/s)']:.3g} m²/s"
                f"（相对 {fit['D 相对标准误'] * 100:.1f}%）。标准误来源："
                f"{fit.get('标准误来源') or '未说明'}。")
        res.summary[f"{name} 原子数"] = int(ag.n_atoms)
        res.summary[f"{name} 追踪对象"] = obj_label
        res.summary[f"{name} 追踪粒子数"] = int(diag.get("粒子数", 0))
        res.summary[f"{name} 去漂移"] = "是" if diag.get("去漂移") else "否"
        res.summary[f"{name} 算法"] = str(diag.get("算法", ""))
        res.summary[f"{name} MSD(最大 τ) (Å²)"] = float(y_tot[-1]) if y_tot.size else float("nan")
        res.summary[f"{name} MSD∥(最大 τ) (Å²)"] = float(msd["para"][-1]) if lag_t.size else float("nan")
        res.summary[f"{name} MSD⊥(最大 τ) (Å²)"] = float(msd["perp"][-1]) if lag_t.size else float("nan")
        res.summary[f"{name} D∥ (m²/s)"] = fit_para["D (m²/s)"]
        res.summary[f"{name} D⊥ (m²/s)"] = fit_perp["D (m²/s)"]
        d_para, d_perp = fit_para["D (m²/s)"], fit_perp["D (m²/s)"]
        ratio = (d_para / d_perp if (np.isfinite(d_para) and np.isfinite(d_perp)
                                    and d_perp > 0) else float("nan"))
        res.summary[f"{name} 各向异性 D∥/D⊥"] = ratio
        res.summary[f"{name} α∥ (log-log 斜率)"] = fit_para["α (log-log 斜率)"]
        res.summary[f"{name} α⊥ (log-log 斜率)"] = fit_perp["α (log-log 斜率)"]

        fail = float(diag.get("失效比例", 0.0) or 0.0)
        res.summary[f"{name} 最小镜像失效比例"] = fail
        # 最小镜像追踪的可靠性必须并入总判定：失效比例高时，被错误回绕的粒子贡献的
        # 是**随机**大位移，会让 MSD 看起来依然线性（R² 甚至接近 1）却整体失真。
        if fail > 0.05:
            base = str(res.summary.get(f"{name} 拟合可靠性", "") or "")
            if fail > 0.20:
                tag = (f"最小镜像失效比例 {fail * 100:.0f}%（严重）——"
                       f"帧间隔内多数粒子的位移已超过最小盒高的一半，"
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
        for extra in diag.get("说明", []) or []:
            res.add_notes(f"{name}：{extra}")
        if np.isfinite(ratio):
            if 0.8 <= ratio <= 1.25:
                res.add_notes(
                    f"{name} 的扩散是**近似各向同性**的（D∥/D⊥ = {ratio:.2f}）。")
            else:
                res.add_notes(
                    f"{name} 的扩散**各向异性**明显：D∥/D⊥ = {ratio:.2f}"
                    f"（D∥={d_para:.3g}、D⊥={d_perp:.3g} m²/s）。"
                    f"这通常意味着取向有序（界面吸附、拉伸或受限孔道），"
                    f"只报总 D 会把这个信息平均掉。")

    bad_fail = [n for n, f in fail_by_group.items() if f > 0.20]
    res.add_notes("MSD = <|r(t+τ) − r(t)|²>，对全部时间起点与全部粒子求平均。")
    res.add_notes("默认追踪对象是**分子质心**：得到的是分子的平动扩散；"
                  "若选「原子」，MSD 会额外包含分子内部的振动与转动"
                  "（长 τ 下斜率仍趋于同一个 D，但截距更大）。")
    res.add_notes("MSD 用 FFT 快速相关算法一次算完，保留完整时间分辨率"
                  "（不做分块、不牺牲长 τ）。")
    res.add_notes("跨周期性边界通过「最小镜像位移累加」追踪粒子，相当于把分子在时间上解开。")
    res.add_notes("MSD∥ = MSD_x + MSD_y（垂直于 z 轴的面内分量），MSD⊥ = MSD_z；"
                  "D∥ 按 2 维、D⊥ 按 1 维拟合（MSD∥ = 4D∥t，MSD⊥ = 2D⊥t）。")
    if bad_fail:
        res.add_notes(
            "以下组分的「最小镜像失效比例」超过 20%，其 D 不应直接引用："
            + "、".join(bad_fail)
            + "。建议用更密的输出间隔重新计算 MSD。")
    res.add_notes(f"总扩散系数由 MSD = 2·dim·D·t（dim={dim}）的线性区斜率得到。")

    bad = [n for n in groups
           if not np.isfinite(res.summary.get(f"{n} D (Å²/ps)", float("nan")))]
    if bad:
        res.add_notes("以下组分的 MSD 曲线平坦或下降，粒子基本没有扩散，"
                      "因此不给扩散系数（D = nan）：" + "、".join(bad)
                      + "。这通常意味着该组分处于玻璃态/被约束/结晶相，"
                        "也可能是帧间隔太大导致曲线不具扩散性。")
    return res
