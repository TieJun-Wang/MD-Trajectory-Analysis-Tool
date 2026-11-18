# -*- coding: utf-8 -*-
"""二面角分布（dihedral）—— 用 **MDAnalysis ``Dihedral``** 独立复现并与引擎对拍。

我们的口径（``mdta/analysis/conformation.py::analyze_dihedrals``）
----------------------------------------------------------------
``mode="chain"``（默认，非蛋白体系）：**逐分子**在键图上取**最长路径**，
沿路径每连续 4 个重原子取一个二面角（``chain_dihedrals``）；角度在**解包坐标**
上算；分布按 Materials Studio Forcite 的「Bin width」取网格
（端点 ``±(180+bin)``、步长 ``bin``），y 是**概率密度** ``计数/(样本数·bin)``；
trans 判据 ``|φ| > gauche_edge``（默认 120°，与「二面角分析」同一阈值）。

本文件的独立复现
----------------
1. **四元组**（:func:`chain_quadruplets`）：自己按键表做 BFS 两遍取最长路径
   （不调用引擎的 ``chain_dihedrals``），与引擎的口径一致但实现独立；
2. **角度**：交给 ``MDAnalysis.analysis.dihedrals.Dihedral`` 逐帧算
   —— 它是第三方实现，内部用 ``calc_dihedrals`` 且**自己处理周期性**
   （实测与"解包后再算"差 1.7e-6°）；
3. **对拍**：同一批四元组、同一帧、同一 bin 宽度下，分布（概率密度）与
   trans 比例应与引擎一致；差异只应来自浮点求和顺序。

⚠️ 与引擎的**已知口径差别**：MDAnalysis ``Dihedral`` 用**拓扑里的原子顺序**
（即我们传进去的四元组顺序），所以必须把 BFS 路径按顺序喂给它 —— 我们已经这样做。

单独跑：``python test/mda_test/dihedral.py``（菜单）
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, save_csv
except ImportError:
    from _common import frame_indices, save_csv


def _frames(u, frames):
    return frame_indices(u) if frames is None else [int(f) for f in frames]


# ------------------------------------------------------------------ 四元组
def _heavy_mask(mol):
    """碎片里的重原子掩码（元素缺失时按原子名首字母判断）。"""
    try:
        el = np.asarray([str(e) for e in mol.elements], dtype=str)
        if el.size == mol.n_atoms:
            return (el != "H") & (el != "")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        nm = np.asarray([str(n).strip().upper() for n in mol.names], dtype=str)
        return ~np.char.startswith(nm, "H")
    except Exception:                                    # noqa: BLE001
        return np.ones(int(mol.n_atoms), dtype=bool)


def _bfs_far(adj, start):
    """从 start 出发 BFS，返回 (最远点, 距离, 前驱数组)。"""
    n = len(adj)
    dist = np.full(n, -1, dtype=int)
    prev = np.full(n, -1, dtype=int)
    dist[start] = 0
    queue = [start]
    k = 0
    while k < len(queue):
        v = queue[k]
        k += 1
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                prev[w] = v
                queue.append(w)
    far = int(np.argmax(dist))
    return far, int(dist[far]), prev


def chain_quadruplets(ag, *, heavy_only: bool = True, max_molecules: int = 200):
    """**独立实现**：逐分子在键图上取最长路径，返回连续四原子二面角。

    返回 ``(idx, notes)``：``idx`` 形状 ``(n, 4)`` 的**全局原子索引**。
    与引擎 ``chain_dihedrals`` 同一口径（每个分子一条最长路径、重原子、
    不足 4 个重原子的分子跳过、最多 ``max_molecules`` 个分子），但代码独立。
    """
    notes: list[str] = []
    idx = np.zeros((0, 4), dtype=int)
    if ag is None or ag.n_atoms == 0:
        return idx, ["选择为空"]
    try:
        bonds = np.asarray(ag.bonds.indices, dtype=int)
    except Exception:                                    # noqa: BLE001
        return idx, ["当前拓扑不含键连接信息，无法取几何骨架二面角"]
    if bonds.size == 0:
        return idx, ["当前拓扑不含键连接信息，无法取几何骨架二面角"]

    order = np.asarray(ag.indices, dtype=int)
    pos_of = {int(a): i for i, a in enumerate(order)}

    try:
        frags = list(ag.fragments)
    except Exception:                                    # noqa: BLE001
        frags = [ag]
    usable = []
    for f in frags:
        keep = _heavy_mask(f) if heavy_only else np.ones(int(f.n_atoms), dtype=bool)
        n_heavy = int(keep.sum())
        if n_heavy >= 4:
            usable.append(f)
    notes.append(f"选择内 {len(usable)} 个分子（每个至少 4 个重原子）")
    truncated = len(usable) > int(max_molecules)
    if truncated:
        notes.append(f"分子多于上限 {max_molecules}，只取前 {max_molecules} 个")
        usable = usable[: int(max_molecules)]

    rows = []
    n_quad = 0
    for f in usable:
        gi = [int(x) for x in np.asarray(f.indices, dtype=int)]
        keep = _heavy_mask(f) if heavy_only else np.ones(len(gi), dtype=bool)
        h_idx = [g for g, k in zip(gi, keep) if k]
        if len(h_idx) < 4:
            continue
        local = {int(g): i for i, g in enumerate(h_idx)}
        adj = [[] for _ in h_idx]
        for a, b in bonds:
            ia, ib = local.get(int(a)), local.get(int(b))
            if ia is None or ib is None:
                continue
            adj[ia].append(ib)
            adj[ib].append(ia)
        if not any(adj):                     # 没有重原子之间的键
            continue
        # 两遍 BFS：任一 → 最远点 u；u → 最远点 v，前驱链即最长路径
        u0 = next(k for k, a in enumerate(adj) if a)
        u, _d, _p = _bfs_far(adj, u0)
        v, _d2, prev = _bfs_far(adj, u)
        path = []
        cur = v
        while cur >= 0:
            path.append(h_idx[cur])
            cur = int(prev[cur])
        path.reverse()
        for k in range(len(path) - 3):
            rows.append([path[k], path[k + 1], path[k + 2], path[k + 3]])
            n_quad += 1
    if rows:
        idx = np.asarray(rows, dtype=int)
    notes.append(f"共提取连续四原子二面角 {n_quad} 个")
    _ = pos_of
    return idx, notes


# ------------------------------------------------------------------ 参考值
def dihedral_reference(u, ag, *, frames=None, bin_width: float = 1.0,
                       heavy_only: bool = True, max_molecules: int = 200,
                       verbose: bool = True):
    """用 MDAnalysis ``Dihedral`` 算二面角序列 → 分布（概率密度）+ trans 比例。"""
    from MDAnalysis.analysis.dihedrals import Dihedral

    use = _frames(u, frames)
    idx, notes = chain_quadruplets(ag, heavy_only=heavy_only,
                                   max_molecules=max_molecules)
    out = {"angles": np.zeros((0, 0)), "idx": idx, "notes": notes,
           "centers": np.zeros(0), "density": np.zeros(0),
           "trans": float("nan"), "gauche_plus": float("nan"),
           "gauche_minus": float("nan"), "n_samples": 0}
    if idx.shape[0] == 0:
        if verbose:
            print("  ⚠️ 没有可用的二面角：" + "；".join(notes))
        return out

    aggs = [u.atoms[[int(a), int(b), int(c), int(d)]] for a, b, c, d in idx]
    d = Dihedral(aggs)
    step = max(1, int(round((use[-1] - use[0]) / max(len(use) - 1, 1))))
    d.run(start=use[0], stop=use[-1] + 1, step=step)
    ang = np.asarray(d.results.angles, dtype=float)

    bw = float(bin_width)
    # 分箱网格与 Forcite / 引擎一致：最外侧**格心** = ±(180+bin)，步长 = bin，
    # 桶边界 = 格心 ± bin/2（bin=1 → 363 个格心 −181…181，边界 ±181.5）
    limit = 180.0 + bw
    n_bins = max(int(round(2.0 * limit / bw)) + 1, 1)
    centers = -limit + bw * np.arange(n_bins)
    edges = np.append(centers - bw / 2.0, centers[-1] + bw / 2.0)
    finite = ang[np.isfinite(ang)]
    counts, _ = np.histogram(finite, bins=edges)
    n_samp = int(counts.sum())
    dens = counts / max(float(n_samp) * bw, 1e-300)
    tr = float(np.mean(np.abs(finite) > 120.0)) if n_samp else float("nan")
    gp = float(np.mean((finite > 60.0) & (finite <= 120.0))) if n_samp else float("nan")
    gm = float(np.mean((finite < -60.0) & (finite >= -120.0))) if n_samp else float("nan")
    out.update(angles=ang, centers=centers, density=dens, trans=tr,
               gauche_plus=gp, gauche_minus=gm, n_samples=n_samp,
               bin_width=bw)
    if verbose:
        print(f"  MDAnalysis Dihedral：{idx.shape[0]} 个二面角 × {ang.shape[0]} 帧，"
              f"样本 {n_samp:,}")
        print(f"    分布峰值（概率密度）{dens.max():.6f} 1/°，"
              f"平均角 {finite.mean():+.2f}°")
        print(f"    trans（|φ|>120°）{tr:.4f} / gauche+ {gp:.4f} / gauche− {gm:.4f}")
    return out


def compare_engine(u, ag, *, frames=None, bin_width: float = 1.0,
                   gauche_edge: float = 120.0, heavy_only: bool = True,
                   max_molecules: int = 200):
    """与工具引擎 ``analyze_dihedrals`` 对拍：分布概率密度 + trans 比例。"""
    from types import SimpleNamespace

    from mdta.analysis.conformation import analyze_dihedrals
    from mdta.io import MDTrajectory

    use = _frames(u, frames)
    ref = dihedral_reference(u, ag, frames=use, bin_width=bin_width,
                             heavy_only=heavy_only,
                             max_molecules=max_molecules, verbose=False)
    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(use, dtype=int),
                          times_ps=times[np.asarray(use, dtype=int)],
                          n_frames=len(use), describe=f"{len(use)} 帧", notes=[],
                          head=lambda self, k: self)
    res = analyze_dihedrals(mdt, ag, sel, mode="chain", bin_width=float(bin_width),
                            gauche_edge=float(gauche_edge),
                            heavy_only=bool(heavy_only),
                            max_molecules=int(max_molecules))
    dens = None
    for c in res.curves:
        if c.panel == 0:
            dens = np.asarray(c.y, dtype=float)
            break
    out = {"n_engine": int(res.summary.get("dihedral 二面角个数", 0)),
           "trans_engine": float(res.summary.get("dihedral trans 比例", float("nan"))),
           "dmax": None}
    print(f"\n=== 与引擎对拍（{len(use)} 帧，bin={bin_width}°）===")
    print(f"  二面角个数：参考 {ref['idx'].shape[0]} vs 引擎 {out['n_engine']}"
          f"{'  ✅' if ref['idx'].shape[0] == out['n_engine'] else '  ★ 不一致'}")
    if dens is not None and ref["density"].size == dens.size:
        d = np.abs(dens - ref["density"])
        out["dmax"] = float(d.max())
        print(f"  分布（概率密度）：{dens.size} 个 bin，最大绝对差 {d.max():.3e}"
              f"{'  ✅' if d.max() < 1e-9 else '  ★ 偏差较大'}")
    if np.isfinite(out["trans_engine"]):
        dt = abs(out["trans_engine"] - ref["trans"])
        print(f"  trans 比例：参考 {ref['trans']:.6f} vs 引擎 "
              f"{out['trans_engine']:.6f}（差 {dt:.2e}）"
              f"{'  ✅' if dt < 1e-6 else '  ★'}")
        out["dtrans"] = float(dt)
    return out


def test_dihedral(u, ag, *, frames=None, bin_width: float = 1.0,
                  gauche_edge: float = 120.0, heavy_only: bool = True,
                  max_molecules: int = 200, prefix: str = "dihedral_mda",
                  compare: bool = True):
    """算二面角分布（MDAnalysis Dihedral）+ 导出 CSV + 与引擎对拍。"""
    ref = dihedral_reference(u, ag, frames=frames, bin_width=bin_width,
                             heavy_only=heavy_only,
                             max_molecules=max_molecules)
    if ref["centers"].size:
        save_csv(f"{prefix}.csv", [ref["centers"], ref["density"]],
                 header=("phi_deg,P_phi  # probability density (1/deg); "
                         f"bin={bin_width}; n_samples={ref['n_samples']}; "
                         "source=MDAnalysis.analysis.dihedrals.Dihedral"),
                 fmt="%.4f,%.8f")
    if compare:
        compare_engine(u, ag, frames=frames, bin_width=bin_width,
                       gauche_edge=gauche_edge, heavy_only=heavy_only,
                       max_molecules=max_molecules)
    return ref


def menu():
    """菜单 / 命令行入口：``python test/mda_test/dihedral.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("bin_width", "Bin width 柱宽 (°)", "float", 1.0, lo=0.05,
          help="与 Forcite Torsion Distribution 的 Bin width 一致"),
        Q("gauche_edge", "trans 判据 |φ| > (°)", "float", 120.0, lo=60.0, hi=180.0,
          help="与「二面角分析」共用同一阈值"),
        Q("heavy_only", "只用重原子", "bool", True,
          help="几何骨架只走重原子（氢会把路径拐进侧基）"),
        Q("max_molecules", "最多取多少个分子", "int", 200, lo=1),
        Q("compare", "同时与引擎对拍？", "bool", True),
    ]

    def _run(cfg):
        return test_dihedral(cfg["u"], cfg["ag"], frames=cfg["frames"],
                             bin_width=float(cfg["bin_width"]),
                             gauche_edge=float(cfg["gauche_edge"]),
                             heavy_only=bool(cfg.get("heavy_only", True)),
                             max_molecules=int(cfg.get("max_molecules") or 200),
                             prefix=cfg["prefix"],
                             compare=bool(cfg.get("compare", True)))

    return run_module("二面角分布（MDAnalysis Dihedral）", _run, specs,
                      sub="BFS 最长路径取四元组 → MDAnalysis Dihedral 算角度 → "
                          "概率密度分布 + trans 比例，与引擎对拍",
                      default_frames=10, default_prefix="dihedral_mda")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
