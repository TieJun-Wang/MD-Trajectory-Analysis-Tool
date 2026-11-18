# -*- coding: utf-8 -*-
"""键取向序参数 BOO（q4 / q6 / q̄4 / q̄6）—— 用 **freud ``Steinhardt``** 独立复现。

我们的口径（``mdta/analysis/boo.py::compute_ql``，默认 ``backend="numpy"``）
---------------------------------------------------------------------
对每个原子的邻居（``r ≤ cutoff``、不含自身）求球谐平均

    q_lm(i) = (1/N_i) Σ_j Y_lm(r̂_ij)          q_l(i) = √(4π/(2l+1) Σ_m |q_lm(i)|²)

再给 Lechner–Dellago **平均版** ``q̄_l``：先把邻居的 ``q_lm(j)`` 平均、再求模
（``q̄_lm(i) = (1/N_i) Σ_j q_lm(j)``）。近邻半径的自动推定是
「跳过成键对（< 2.2 Å）后的最近邻距离中位数 × 1.35」。

本文件的独立复现
----------------
1. **近邻半径**（:func:`estimate_cutoff_freud`）：用 freud 的
   ``locality.NearestNeighbors`` 取 k 近邻距离，跳过 < 2.2 Å 的成键对，
   取中位数 × 1.35 —— 与引擎同一**约定**，但邻居搜索是第三方的；
2. **q_l / q̄_l**：交给 freud ``order.Steinhardt(l=4 / 6)``
   （``average=False`` → q_l；``average=True`` → Lechner–Dellago 的 q̄_l），
   邻居表用 ``{"r_max": cutoff, "exclude_ii": True}``，盒子用
   ``freud.box.Box.from_box``（自带最小镜像）；
3. **对拍**：逐原子 q4/q6（及平均版）与引擎 numpy 实现比较，
   再比帧平均与引擎 ``analyze_boo`` 摘要里的平均值。

单独跑：``python test/mda_test/boo.py``（菜单）
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, save_csv
except ImportError:
    from _common import frame_indices, save_csv

#: 与引擎 ``_BOND_EXCLUDE`` 同一个约定：推定"第一壳层"时跳过成键对
BOND_EXCLUDE = 2.2
#: 自动 cutoff = 最近邻距离中位数 × 该系数（与引擎一致）
CUT_FACTOR = 1.35


def _freud():
    try:
        import freud
    except Exception as exc:                                  # noqa: BLE001
        raise SystemExit(
            "需要 freud-analysis 才能用这个脚本：\n"
            "  python -m pip install -i https://pypi.org/simple --only-binary=:all: "
            "freud-analysis\n"
            f"（导入 freud 失败：{type(exc).__name__}: {exc}）") from exc
    return freud


def _box_of(box):
    """把 ``u.dimensions``（MDAnalysis 的 ``[Lx,Ly,Lz,α,β,γ]``）转成 freud Box。

    ⚠️ freud 的 ``from_box([Lx,Ly,Lz,xy,xz,yz])`` 后三位是**倾斜因子**、不是角度；
    直接把 MDAnalysis 的 6 个数丢进去会得到畸形的盒子（实测报
    "AABBQuery r_max is too large for this box"）。所以先用 MDAnalysis 的
    ``triclinic_vectors`` 转成 3×3 盒矢量，再 ``Box.from_matrix``。
    """
    freud = _freud()
    if box is None:
        return freud.box.Box.cube(1e6)       # 无周期盒：给个足够大的盒子
    dims = np.asarray(box, dtype=float)
    if dims.size < 3 or not np.all(np.isfinite(dims[:3])) or np.any(dims[:3] <= 0):
        return freud.box.Box.cube(1e6)
    if dims.size >= 6:
        from MDAnalysis.lib.mdamath import triclinic_vectors

        return freud.box.Box.from_matrix(triclinic_vectors(dims[:6]))
    return freud.box.Box.from_box(np.asarray([*dims[:3]], dtype=float))


def estimate_cutoff_freud(pos, box, probe: float = 8.0,
                          verbose: bool = True) -> float:
    """用 freud 的邻居查询推定第一壳层半径（跳过成键对后的最近邻中位数 × 1.35）。

    freud 3.x 没有 ``locality.NearestNeighbors``，用 ``AABBQuery`` 取 ``r ≤ probe``
    的邻居，再对每个原子取"第一个距离 > ``BOND_EXCLUDE`` 的邻居"。
    """
    freud = _freud()
    pts = np.asarray(pos, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return 5.0
    bb = _box_of(box)
    aq = freud.locality.AABBQuery(bb, pts)
    nlist = aq.query(pts, {"r_max": float(probe), "exclude_ii": True}
                     ).toNeighborList()
    dist = np.asarray(nlist.distances, dtype=float)
    qi = np.asarray(nlist.query_point_indices, dtype=int)
    first = np.full(n, np.inf)
    m = dist > BOND_EXCLUDE
    np.minimum.at(first, qi[m], dist[m])
    fin = first[np.isfinite(first)]
    if fin.size == 0:
        return 5.0
    cut = float(np.median(fin) * CUT_FACTOR)
    if verbose:
        print(f"  自动近邻半径：最近邻中位数 {np.median(fin):.4f} Å × "
              f"{CUT_FACTOR} = {cut:.4f} Å")
    return cut


def boo_freud_frame(pos, box, cutoff: float, l_list=(4, 6), average: bool = True,
                    verbose: bool = False) -> dict:
    """一帧：freud Steinhardt 给出的逐原子 q_l / q̄_l 与全帧平均。"""
    freud = _freud()
    pts = np.asarray(pos, dtype=np.float64)
    bb = _box_of(box)
    args = {"r_max": float(cutoff), "exclude_ii": True}
    out = {"n_neighbors": None}
    for l in l_list:
        st = freud.order.Steinhardt(l=int(l), average=bool(average))
        st.compute((bb, pts), args)
        key = f"q{l}" if not average else f"q{l}bar"
        arr = np.asarray(st.particle_order, dtype=float)
        out[key] = arr
        # ⚠️ 帧平均要用**逐原子值的算术平均**，不能用 freud 的 ``st.order``：
        #    ``order`` 是"先对全体系平均 q_lm、再求模"的**全局**序参数
        #    （完全无序时它趋于 0，而逐原子平均仍是非零的局部值），
        #    与引擎摘要里的"q_l 平均"不是同一个量。
        out[f"{key}_mean"] = float(np.nanmean(arr))
        out[f"{key}_global"] = float(st.order)
        if verbose:
            print(f"    freud l={l} {'平均版' if average else ''}："
                  f"逐原子平均 {out[f'{key}_mean']:.6f}"
                  f"（全局 order={out[f'{key}_global']:.6f}）")
    return out


def boo_reference(u, ag, *, frames=None, cutoff: float | None = None,
                  l_list=(4, 6), verbose: bool = True):
    """逐帧用 freud 算 BOO，返回逐帧平均 + 第一帧的逐原子值。"""
    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    keys = []
    for l in l_list:
        keys += [f"q{l}", f"q{l}bar"]
    means = {k: np.full(len(use), np.nan) for k in keys}
    atom_first = None
    cut_used = float(cutoff) if cutoff is not None else None
    n_atoms = int(ag.n_atoms)
    for k, fi in enumerate(use):
        u.trajectory[fi]
        pos = np.asarray(ag.positions, dtype=float)
        box = np.asarray(u.dimensions, dtype=float)
        cut = cut_used if cut_used is not None else estimate_cutoff_freud(
            pos, box, verbose=False)
        cut_used = cut
        plain = boo_freud_frame(pos, box, cut, l_list=l_list, average=False)
        avg = boo_freud_frame(pos, box, cut, l_list=l_list, average=True)
        for l in l_list:
            means[f"q{l}"][k] = plain[f"q{l}_mean"]
            means[f"q{l}bar"][k] = avg[f"q{l}bar_mean"]
        if k == 0:
            atom_first = {key: val for key, val in {**plain, **avg}.items()
                          if isinstance(val, np.ndarray)}
        if verbose:
            print(f"  帧 {fi}: cutoff = {cut:.3f} Å，"
                  + "，".join(f"⟨q{l}⟩ {means[f'q{l}'][k]:.4f} / "
                              f"⟨q̄{l}⟩ {means[f'q{l}bar'][k]:.4f}" for l in l_list))
    if verbose:
        print(f"  全轨迹平均：" + "，".join(
            f"q{l} {np.nanmean(means[f'q{l}']):.5f} / "
            f"q̄{l} {np.nanmean(means[f'q{l}bar']):.5f}" for l in l_list))
    return {"frames": use, "cutoff": cut_used, "means": means,
            "atom_first": atom_first, "n_atoms": n_atoms,
            "l_list": list(l_list)}


def compare_engine(u, ag, *, frames=None, cutoff: float | None = None,
                   l_list=(4, 6)):
    """与引擎 ``compute_ql``（numpy 后端）逐原子对拍 + 与 ``analyze_boo`` 平均值对拍。"""
    from types import SimpleNamespace

    from mdta.analysis.boo import analyze_boo, compute_ql
    from mdta.io import MDTrajectory

    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    ref = boo_reference(u, ag, frames=use, cutoff=cutoff, l_list=l_list,
                        verbose=False)
    cut = float(ref["cutoff"])

    print(f"\n=== 逐原子对拍（freud vs 引擎 numpy，cutoff={cut:.3f} Å）===")
    worst = {f"q{l}": None for l in l_list}
    for k, fi in enumerate(use):
        u.trajectory[fi]
        pos = np.asarray(ag.positions, dtype=float)
        box = np.asarray(u.dimensions, dtype=float)
        plain = boo_freud_frame(pos, box, cut, l_list=l_list, average=False)
        avg = boo_freud_frame(pos, box, cut, l_list=l_list, average=True)
        eng_plain = compute_ql(pos, box, cut, l_list=l_list, averaged=False,
                               backend="numpy")
        eng_avg = compute_ql(pos, box, cut, l_list=l_list, averaged=True,
                             backend="numpy")
        for l in l_list:
            d1 = float(np.nanmax(np.abs(plain[f"q{l}"] - eng_plain[f"q{l}"])))
            d2 = float(np.nanmax(np.abs(avg[f"q{l}bar"] - eng_avg[f"q{l}bar"])))
            prev = worst[f"q{l}"] or 0.0
            worst[f"q{l}"] = max(prev, d1, d2)
            print(f"  帧 {fi}  l={l}: q{l} 最大差 {d1:.3e}，"
                  f"q̄{l} 最大差 {d2:.3e}")
    for key, val in worst.items():
        # freud 用 float32 算球谐，逐原子差在 1e-6 量级；1e-5 以内视为一致
        print(f"  {key} 逐原子最大差（全部帧）= {val:.3e}"
              f"{'  ✅ 一致（freud 内部 float32，~1e-6）' if val < 1e-5 else '  ★ 偏差较大'}")

    # 与 analyze_boo 的摘要平均值对拍
    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(use, dtype=int),
                          times_ps=times[np.asarray(use, dtype=int)],
                          n_frames=len(use), describe=f"{len(use)} 帧", notes=[],
                          head=lambda self, k: self)
    res = analyze_boo(mdt, ag, sel, cutoff=cut, l_list=l_list, averaged=True,
                      backend="numpy")
    print(f"\n=== 与引擎 analyze_boo 的帧平均对拍 ===")
    out = {"worst": worst, "cutoff": cut}
    for l in l_list:
        e_bar = res.summary.get(f"q{l}bar 帧平均（时间序列均值）")
        r_bar = float(np.nanmean(ref["means"][f"q{l}bar"]))
        if e_bar is not None:
            d = abs(float(e_bar) - r_bar)
            out[f"d{l}bar"] = d
            print(f"  q̄{l} 帧平均：freud {r_bar:.8f} vs 引擎 {float(e_bar):.8f}"
                  f"（差 {d:.2e}）{'  ✅' if d < 1e-6 else '  ★'}")
        e_q = res.summary.get(f"q{l} 帧平均（时间序列均值）")
        r_q = float(np.nanmean(ref["means"][f"q{l}"]))
        if e_q is not None:
            d = abs(float(e_q) - r_q)
            out[f"d{l}"] = d
            print(f"  q{l} 帧平均：freud {r_q:.8f} vs 引擎 {float(e_q):.8f}"
                  f"（差 {d:.2e}）{'  ✅' if d < 1e-6 else '  ★'}")
    return out


def test_boo(u, ag, *, frames=None, cutoff: float | None = None,
             l_list=(4, 6), prefix: str = "boo_mda", compare: bool = True):
    """freud Steinhardt 算 BOO + 导出 CSV + 与引擎对拍。"""
    ref = boo_reference(u, ag, frames=frames, cutoff=cutoff, l_list=l_list)
    use = ref["frames"]
    cols = [np.asarray(use, dtype=np.int64)]
    head = ["frame"]
    for l in ref["l_list"]:
        cols.append(ref["means"][f"q{l}"])
        cols.append(ref["means"][f"q{l}bar"])
        head += [f"q{l}_mean", f"q{l}bar_mean"]
    fmt = "%d" + ",%.8f" * (len(cols) - 1)
    save_csv(f"{prefix}.csv", cols,
             header=",".join(head) + f"  # cutoff={ref['cutoff']:.4f}A; "
             "source=freud.order.Steinhardt(l=4/6, average=False/True)",
             fmt=fmt)
    if ref["atom_first"] is not None:
        cols2 = [np.arange(ref["n_atoms"], dtype=np.int64)]
        head2 = ["atom_index"]
        for key, val in ref["atom_first"].items():
            if val.ndim == 1 and val.size == ref["n_atoms"]:
                cols2.append(val)
                head2.append(key)
        if len(cols2) > 2:
            save_csv(f"{prefix}__atoms.csv", cols2,
                     header=",".join(head2) + "  # 第 1 帧的逐原子值（freud）",
                     fmt="%d" + ",%.8f" * (len(cols2) - 1))
    if compare:
        compare_engine(u, ag, frames=frames, cutoff=ref["cutoff"], l_list=l_list)
    return ref


def menu():
    """菜单 / 命令行入口：``python test/mda_test/boo.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("l", "角量子数 l", "choice", "46",
          choices=[("46", "4 与 6（默认）"), ("4", "只要 l=4"), ("6", "只要 l=6")]),
        Q("cutoff", "近邻半径 (Å，0 = 自动)", "float", 0.0, lo=0.0,
          help="自动 = 跳过成键对后的最近邻距离中位数 × 1.35"),
        Q("compare", "同时与引擎对拍？", "bool", True),
    ]

    def _run(cfg):
        l_sel = str(cfg.get("l") or "46")
        l_list = {"4": (4,), "6": (6,)}.get(l_sel, (4, 6))
        cut = float(cfg.get("cutoff") or 0.0) or None
        return test_boo(cfg["u"], cfg["ag"], frames=cfg["frames"], cutoff=cut,
                        l_list=l_list, prefix=cfg["prefix"],
                        compare=bool(cfg.get("compare", True)))

    return run_module("键取向序参数 BOO（freud Steinhardt）", _run, specs,
                      sub="freud.order.Steinhardt(l=4/6, average=False/True) "
                          "求 q4/q6 与平均版 q̄4/q̄6；与引擎逐原子对拍",
                      default_frames=5, default_prefix="boo_mda")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
