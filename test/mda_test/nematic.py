# -*- coding: utf-8 -*-
"""取向参数 S（二阶序参数）—— 用 **freud ``Nematic``** 独立复现并与引擎对拍。

我们的口径（``mdta/analysis/crystallinity.py``）
------------------------------------------------
「链段取向分析」把链切成**链段矢量** ``u``（化学重复单元 / 几何骨架 + stride，
可用"相邻键中点矢量"法），再算取向张量的最大本征值

    Q = <(3 u uᵀ − I)/2>        S = λ_max(Q)
    P₂ = 3cos²θ − 1 的二阶形式：S 也就是"相对自身取向张量"的取向参数

「结构有序度分析」里的**取向有序度 S** 是同一个量，并且 1.0.9 起会
**自动沿用链段取向分析的链段定义与统计权重**，两者逐点相同。

本文件的独立复现
----------------
1. **链段矢量**（:func:`segment_vectors`）：用 ``orientation.py`` 里那套
   键中点矢量（独立于引擎的 ``resolve_segments``）按分子拆链、可选 ``make_whole``
   解包、只保留**线性链**、剔除非线性碎片；
2. **S 的计算**：交给 **freud** 的 ``freud.order.Nematic``（第三方实现：
   ``order`` 就是 3/2·λmax − 1/2，已用"完全对齐=1 / 随机≈0 / 30°锥=0.625"
   三个解析用例钉死）；
3. **对拍**：同一批帧、同一批链段下，S(t) 逐点应与引擎一致。

⚠️ 权重口径：freud 的 ``Nematic`` **不支持逐链权重**，所以本脚本验证的是
**链段加权（segment，默认）** 那一档。要看"每条链等权"，用
``orientation.py`` 导出的 ``P2``（逐链等权）列，或引擎里把统计权重改成
「每条链等权」再比 —— 两边都是同一个张量，只是权重不同。

单独跑：``python test/mda_test/nematic.py``（菜单）
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, save_csv
    from .orientation import (chain_coordinates, chain_size_histogram,
                              midpoint_vectors)
except ImportError:
    from _common import frame_indices, save_csv
    from orientation import (chain_coordinates, chain_size_histogram,
                             midpoint_vectors)

#: 张量 → S 的换算：S = (3 λmax − 1) / 2（freud 的 order 就是这个）
def s_from_eigenvalues(lam: np.ndarray) -> float:
    return float((3.0 * float(np.max(lam)) - 1.0) / 2.0)


def _freud_nematic():
    """freud 的 ``Nematic``；没装就给一条清楚的提示（不静默降级）。"""
    try:
        import freud
    except Exception as exc:                                  # noqa: BLE001
        raise SystemExit(
            "需要 freud-analysis 才能用这个脚本：\n"
            "  python -m pip install -i https://pypi.org/simple --only-binary=:all: "
            "freud-analysis\n"
            f"（导入 freud 失败：{type(exc).__name__}: {exc}）") from exc
    return freud.order.Nematic()


def segment_vectors(u, ag, *, frames=None, unwrap: bool = True,
                    verbose: bool = True):
    """逐帧的链段**单位矢量**（形状 ``(n_frames, n_segments, 3)``）+ 链信息。

    链段 = 相邻两个**键中点**连成的矢量（``v_i = m_{i+1} − m_i = (r_{i+2} − r_i)/2``），
    与 ``orientation.py`` 的 allfast 口径相同；只保留线性链（石墨片/环/支化不算链）。
    """
    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    per_frame = []
    n_seg = None
    hist = None
    for fi in use:
        u.trajectory[fi]
        chains = chain_coordinates(ag, unwrap=unwrap, verbose=False)
        if not chains:
            raise SystemExit("该原子组里没有 ≥3 个重原子的线性链，无法算取向参数")
        v = np.concatenate([midpoint_vectors(c) for c in chains])
        n = np.linalg.norm(v, axis=1)
        good = n > 1e-12
        v = v[good] / n[good, None]
        per_frame.append(v)
        n_seg = v.shape[0]
        if hist is None:
            hist = chain_size_histogram(chains)
    if verbose:
        print(f"  链段矢量：{len(use)} 帧 × {n_seg} 个链段（线性链，"
              f"链长分布 {hist}）")
    return np.asarray(per_frame, dtype=float), {"frames": use, "n_segments": n_seg,
                                                "chain_sizes": hist}


def nematic_reference(u, ag, *, frames=None, unwrap: bool = True,
                      verbose: bool = True):
    """用 freud ``Nematic`` 算 S(t)（链段加权口径）。"""
    nem = _freud_nematic()
    vecs, info = segment_vectors(u, ag, frames=frames, unwrap=unwrap,
                                 verbose=verbose)
    s_t = np.full(vecs.shape[0], np.nan)
    for k in range(vecs.shape[0]):
        nem.compute(vecs[k])
        s_t[k] = float(nem.order)
    info["S"] = s_t
    if verbose:
        print(f"  freud Nematic：S = {np.round(s_t, 6).tolist()}")
        print(f"    平均 S = {np.nanmean(s_t):.6f}（0 = 完全无序，1 = 完全取向）")
    return info


def compare_engine(u, ag, *, frames=None, mode: str = "backbone",
                   stride: int = 2, unwrap: bool = True):
    """与引擎 ``analyze_orientation`` 的 S(t) 对拍（面板 1 = 取向参数 S）。"""
    from types import SimpleNamespace

    from mdta.analysis.crystallinity import analyze_orientation
    from mdta.io import MDTrajectory

    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    ref = nematic_reference(u, ag, frames=use, unwrap=unwrap, verbose=False)
    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(use, dtype=int),
                          times_ps=times[np.asarray(use, dtype=int)],
                          n_frames=len(use), describe=f"{len(use)} 帧", notes=[],
                          head=lambda self, k: self)
    res = analyze_orientation(mdt, ag, sel, mode=str(mode), stride=int(stride),
                              p2_axis="z")
    s_eng = None
    for c in res.curves:
        if c.panel == 0:
            s_eng = np.asarray(c.y, dtype=float)
            break
    out = {"S_ref": ref["S"], "S_engine": s_eng,
           "n_segments_ref": ref["n_segments"],
           "n_segments_engine": int(res.summary.get("链段数", 0))}
    print(f"\n=== 与引擎对拍（{len(use)} 帧，mode={mode}，stride={stride}）===")
    print(f"  链段数：参考 {out['n_segments_ref']} vs 引擎 {out['n_segments_engine']}"
          f"{'  ✅' if out['n_segments_ref'] == out['n_segments_engine'] else '  ★ 不一致'}"
          "（链段定义不同时两边就不是同一个量）")
    if s_eng is not None and s_eng.size == ref["S"].size:
        d = np.abs(s_eng - ref["S"])
        out["dmax"] = float(d.max())
        print(f"  S(t)：参考 {np.round(ref['S'], 6).tolist()}")
        print(f"        引擎 {np.round(s_eng, 6).tolist()}")
        # freud 内部用 float32 做张量运算（本征分解也在 float32/float64 之间），
        # 实测差在 1e-7 量级；1e-6 以内就认为是同一个数。
        print(f"  最大绝对差 {d.max():.3e}"
              f"{'  ✅ 一致（freud 内部 float32，~1e-7）' if d.max() < 1e-6 else '  ★ 偏差较大'}")
    return out


def test_nematic(u, ag, *, frames=None, mode: str = "backbone", stride: int = 2,
                 prefix: str = "nematic_mda", compare: bool = True):
    """freud Nematic 算 S(t) + 导出 CSV + 与引擎对拍。"""
    ref = nematic_reference(u, ag, frames=frames)
    use = ref["frames"]
    times = []
    for fi in use:
        try:
            times.append(float(u.trajectory[int(fi)].time))
        except Exception:                                # noqa: BLE001
            times.append(float(fi))
    save_csv(f"{prefix}.csv",
             [np.asarray(use, dtype=np.int64), np.asarray(times, dtype=float),
              ref["S"]],
             header=("frame,time(ps),S_nematic  # S = λmax of <(3uu^T-I)/2> by "
                     "freud.order.Nematic; segment-weighted"),
             fmt="%d,%.4f,%.8f")
    if compare:
        compare_engine(u, ag, frames=frames, mode=mode, stride=stride)
    return ref


def menu():
    """菜单 / 命令行入口：``python test/mda_test/nematic.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("mode", "链段定义（与引擎对拍时用）", "choice", "backbone",
          choices=[("backbone", "几何骨架（几何法）"),
                   ("repeat", "化学重复单元（引擎默认）")]),
        Q("stride", "几何骨架 stride", "int", 2, lo=1,
          help="2 = 相邻键中点矢量（烷烃锯齿链上才是链的走向）"),
        Q("unwrap", "先 make_whole 解包", "bool", True),
        Q("compare", "同时与引擎对拍？", "bool", True),
    ]

    def _run(cfg):
        return test_nematic(cfg["u"], cfg["ag"], frames=cfg["frames"],
                            mode=cfg["mode"], stride=int(cfg["stride"]),
                            prefix=cfg["prefix"],
                            compare=bool(cfg.get("compare", True)))

    return run_module("取向参数 S（freud Nematic）", _run, specs,
                      sub="键中点矢量 → freud.order.Nematic 求 S(t)；"
                          "与引擎链段取向分析的 S 逐点对拍",
                      default_frames=10, default_prefix="nematic_mda")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
