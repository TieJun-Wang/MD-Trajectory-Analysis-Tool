# -*- coding: utf-8 -*-
"""链构象分析模块（设计大纲第 9 章）。

提供三个核心分析：

- :func:`analyze_rg`         回转半径 Rg
- :func:`analyze_end_to_end` 端到端距离 R_ee
- :func:`analyze_dihedrals`  二面角（含 trans / gauche 构象比例）

所有几何量都在"按分子连通性展开（unwrap）"后的坐标上计算，
以保证周期性边界条件不会把分子切断而使结果失真。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core import AnalysisResult, Panel, block_average, describe_array
from ..preprocess import FrameSelection
from ..units import auto_bins, bin_edges_to_centers, time_axis
from .base import frame_iterator, positions_for, register

__all__ = [
    "compute_rg",
    "analyze_rg",
    "analyze_end_to_end",
    "analyze_dihedrals",
    "circular_stats",
    "classify_dihedrals",
    "backbone_path",
    "topology_dihedrals",
]


# ------------------------------------------------------------------ 基本算法
def compute_rg(pos: np.ndarray, masses: np.ndarray | None = None) -> float:
    """回转半径。

    ``Rg = sqrt( Σ m_i |r_i - r_cm|² / Σ m_i )``

    ``masses`` 为 ``None`` 时按质心（等权重）计算。
    """
    pos = np.asarray(pos, dtype=float)
    if pos.shape[0] == 0:
        return float("nan")
    if masses is None:
        com = pos.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((pos - com) ** 2, axis=1))))
    m = np.asarray(masses, dtype=float)
    mtot = m.sum()
    if mtot <= 0:
        return compute_rg(pos, None)
    com = (m[:, None] * pos).sum(axis=0) / mtot
    d2 = np.sum((pos - com) ** 2, axis=1)
    return float(np.sqrt((m * d2).sum() / mtot))


def compute_end_to_end(p_first: np.ndarray, p_last: np.ndarray, box=None) -> float:
    """端到端距离；给出 ``box`` 时按最小镜像修正。"""
    v = np.asarray(p_last, dtype=float) - np.asarray(p_first, dtype=float)
    if box is not None:
        from MDAnalysis.lib.distances import minimize_vectors

        v = minimize_vectors(v, box)
    return float(np.linalg.norm(v))


def circular_stats(angles_deg: np.ndarray) -> dict:
    """角度数据的圆统计量（角度不能直接取算术平均）。"""
    a = np.radians(np.asarray(angles_deg, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "R": float("nan"), "std": float("nan"), "n": 0}
    c, s = np.cos(a).mean(), np.sin(a).mean()
    r = float(np.hypot(c, s))
    mean = float(np.degrees(np.arctan2(s, c)))
    std = float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-12))))) if r > 1e-12 else float("nan")
    return {"mean": mean, "R": r, "std": std, "n": int(a.size)}


def classify_dihedrals(angles_deg: np.ndarray, gauche_edge: float = 120.0) -> dict:
    """按二面角划分 trans / gauche+ / gauche-。

    约定：``|φ| > gauche_edge`` 为 trans；``0 < φ < gauche_edge`` 为 gauche+；
    ``-gauche_edge < φ < 0`` 为 gauche-。
    """
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return {"trans": 0.0, "gauche+": 0.0, "gauche-": 0.0, "n": 0}
    tr = np.abs(a) > gauche_edge
    gp = (a > 0) & ~tr
    gm = (a < 0) & ~tr
    return {
        "trans": float(tr.sum()) / n,
        "gauche+": float(gp.sum()) / n,
        "gauche-": float(gm.sum()) / n,
        "n": int(n),
        "trans_count": int(tr.sum()),
    }


# ------------------------------------------------------------------ 二面角集合
def _bonds_of(ag):
    """安全地取键表：缺键信息时返回 ``None``（MDAnalysis 会抛 NoDataError）。"""
    try:
        b = ag.bonds
    except Exception:  # noqa: BLE001 - 该 Universe 根本没有 bonds 属性
        return None
    return b if (b is not None and len(b) > 0) else None


#: 拓扑缺键信息时给出的统一提示（各分析模块复用）
NO_BOND_HINT = (
    "当前拓扑不含键连接信息。可以：① 换用自带键表的拓扑（.tpr / .psf / .prmtop）；"
    "② 用 load_trajectory(..., guess_bonds=True) 按坐标猜键"
    "（.gro / 无 CONECT 的 .pdb 默认会自动猜）；"
    "③ 对二面角分析改用 mode='phi_psi' 或 mode='custom'。"
)


def topology_dihedrals(ag, heavy_only: bool = False):
    """取拓扑中定义、且完全落在原子组 ``ag`` 内的二面角。

    返回形状 ``(n, 4)`` 的**绝对原子索引**数组；缺键/二面角信息时返回空数组。
    """
    if _bonds_of(ag) is None:
        return np.zeros((0, 4), dtype=int)
    idx = np.asarray(ag.indices)
    lookup = {int(a): i for i, a in enumerate(idx)}
    heavy = None
    if heavy_only:
        try:
            el = np.asarray(ag.elements, dtype=str)
            heavy = {int(a): (e != "H" and e != "") for a, e in zip(idx, el)}
        except Exception:  # noqa: BLE001
            heavy = None
    rows: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    try:
        dihedrals = ag.dihedrals
    except Exception:  # noqa: BLE001
        return np.zeros((0, 4), dtype=int)
    if dihedrals is None:
        return np.zeros((0, 4), dtype=int)
    for d in dihedrals:
        at = [int(a.index) for a in d]
        if any(a not in lookup for a in at):
            continue
        if heavy is not None and not all(heavy.get(a, True) for a in at):
            continue
        key = tuple(at)
        if key in seen or key[::-1] in seen:
            continue
        seen.add(key)
        rows.append(key)
    return np.asarray(rows, dtype=int).reshape(-1, 4)


def backbone_path(ag, heavy_only: bool = True) -> np.ndarray:
    """在键连接图中找一条最长简单路径，作为链骨架（适用于线形高分子）。

    做法：从任一原子做 BFS 找到最远原子 u，再从 u 做 BFS 找到最远原子 v，
    记录路径；对链状分子该路径即为骨架。返回原子索引序列（绝对索引）。
    """
    from collections import deque

    idx = np.asarray(ag.indices)
    n = idx.size
    if n == 0:
        return np.array([], dtype=int)
    bonds = _bonds_of(ag)
    if bonds is None:
        return np.array([], dtype=int)          # 没有键 → 无法提骨架
    pos_of = {int(a): i for i, a in enumerate(idx)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for b in bonds:
        i, j = int(b[0].index), int(b[1].index)
        if i in pos_of and j in pos_of:
            adj[pos_of[i]].append(pos_of[j])
            adj[pos_of[j]].append(pos_of[i])

    if heavy_only:
        try:
            el = np.asarray(ag.elements, dtype=str)
            keep = [i for i in range(n) if el[i] not in ("H", "")]
        except Exception:  # noqa: BLE001
            keep = list(range(n))
        if 0 < len(keep) < n:
            allowed = set(keep)
            adj = [[x for x in row if x in allowed] for row in adj]
        else:
            allowed = None
    else:
        allowed = None

    def bfs(src: int):
        dist = {src: 0}
        prev = {src: None}
        q = deque([src])
        far = src
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
                    if dist[v] > dist[far]:
                        far = v
        return far, prev, dist

    start = next((i for i in range(n) if adj[i]), 0)
    u, _, dist = bfs(start)
    v, prev, _ = bfs(u)
    path: list[int] = []
    cur: int | None = v
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return idx[np.asarray(path, dtype=int)]


def make_dihedral_indices(path_atoms: np.ndarray) -> np.ndarray:
    """由原子路径（长度 L）生成 ``L-3`` 个连续二面角的四原子索引。"""
    p = np.asarray(path_atoms, dtype=int)
    if p.size < 4:
        return np.zeros((0, 4), dtype=int)
    return np.stack([p[i:i + 4] for i in range(p.size - 3)], axis=0)


def compute_dihedral_series(mdt, indices: np.ndarray, selection: FrameSelection, *,
                            unwrap_group=None, verbose: bool = False,
                            progress=None) -> np.ndarray:
    """逐帧计算二面角，返回 ``(n_frames, n_dihedrals)`` 的角度数组（度）。

    二面角是**分子内**几何量，必须用 unwrap 后的坐标，否则跨越模拟盒边界的
    二面角会得到完全错误的角度。

    参数
    ----
    indices
        形状 ``(n, 4)`` 的**绝对原子索引**数组。
    unwrap_group
        用于 PBC 展开的原子组（通常是整条链）。``None`` 时只用涉及到的原子。
    """
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("二面角索引必须是 (n, 4) 的整数数组")
    u = mdt.universe
    n_d = indices.shape[0]
    n_f = len(selection.indices)
    out = np.full((n_f, n_d), np.nan, dtype=float)
    if n_d == 0:
        return out

    from MDAnalysis.lib.distances import calc_dihedrals

    group = unwrap_group if unwrap_group is not None else u.atoms[np.unique(indices)]
    abs_to_local = {int(a): i for i, a in enumerate(np.asarray(group.indices))}
    try:
        sel = np.array([[abs_to_local[int(x)] for x in row] for row in indices], dtype=int)
    except KeyError as exc:  # 索引不在展开组内
        raise ValueError(
            f"二面角原子 {exc} 不在用于 PBC 展开的原子组内；"
            f"请把 unwrap_group 设为包含这些原子的链。"
        ) from exc

    i0, i1, i2, i3 = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pos = positions_for(group, unwrap=True)
        out[k] = np.degrees(calc_dihedrals(pos[i0], pos[i1], pos[i2], pos[i3]))
        if progress is not None:
            progress(k + 1, n_f)
    return out


# ------------------------------------------------------------------ 分析接口
@register("rg", "回转半径 Rg")
def analyze_rg(mdt, ag, selection: FrameSelection, *,
               mass_weighted: bool = True, unwrap: bool = True,
               nbins: int | None = None, verbose: bool = False,
               label: str = "链") -> AnalysisResult:
    """回转半径分析。

    输出：Rg–Time 曲线、Rg 分布、Rg 平均值/标准差/块平均误差。
    """
    u = mdt.universe
    times = np.asarray(selection.times_ps, dtype=float)
    rg = np.full(times.size, np.nan)
    masses = np.asarray(ag.masses, dtype=float) if mass_weighted else None

    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pos = positions_for(ag, unwrap=unwrap)
        rg[k] = compute_rg(pos, masses)

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="rg",
        title=f"回转半径 Rg —— {label}",
        meta={"selection": label, "n_atoms": int(ag.n_atoms),
              "mass_weighted": bool(mass_weighted), "unwrap": bool(unwrap),
              "n_frames": times.size},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="回转半径 Rg (Å)", title="Rg 随时间变化"),
        Panel(xlabel="回转半径 Rg (Å)", ylabel="概率密度 P(Rg)", title="Rg 分布"),
    ]
    res.add_curve(label, tx, rg, kind="line", panel=0)

    stat = describe_array(rg)
    res.add_curve("平均 Rg", tx, np.full_like(tx, stat["mean"]), kind="line", panel=0)
    _bmeans, bsem = block_average(rg, n_blocks=min(5, max(2, rg.size // 2)))

    finite = rg[np.isfinite(rg)]
    if finite.size:
        edges = auto_bins(finite, nbins)
        hist, _ = np.histogram(finite, bins=edges, density=True)
        res.add_curve("Rg 分布", bin_edges_to_centers(edges), hist, kind="bar", panel=1)

    res.summary.update(describe_array(rg, prefix="Rg "))
    res.summary["Rg 块平均标准误"] = float(bsem)
    res.add_notes(f"质量加权: {mass_weighted}；PBC 展开: {unwrap}")
    res.add_notes(f"参与统计原子数: {ag.n_atoms}")
    return res


@register("ree", "端到端距离 R_ee")
def analyze_end_to_end(mdt, ag, selection: FrameSelection, *,
                       atom_indices: Sequence[int] | None = None,
                       unwrap: bool = True, nbins: int | None = None,
                       verbose: bool = False, label: str = "链") -> AnalysisResult:
    """端到端距离分析。

    ``atom_indices`` 可用绝对原子编号显式指定链两端；默认取所选原子组的
    首尾两个原子。
    """
    u = mdt.universe
    times = np.asarray(selection.times_ps, dtype=float)
    ree = np.full(times.size, np.nan)
    if atom_indices is not None:
        pair = ag[np.asarray([int(atom_indices[0]), int(atom_indices[1])])]
    else:
        pair = ag[[0, ag.n_atoms - 1]] if ag.n_atoms >= 2 else ag

    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        p = positions_for(pair, unwrap=unwrap)
        if p.shape[0] < 2:
            ree[k] = np.nan
        else:
            ree[k] = compute_end_to_end(p[0], p[-1], box=None if unwrap else u.dimensions)

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="ree",
        title=f"端到端距离 R_ee —— {label}",
        meta={"selection": label, "n_atoms": int(ag.n_atoms), "n_frames": times.size},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="端到端距离 R_ee (Å)", title="端到端距离随时间变化"),
        Panel(xlabel="端到端距离 R_ee (Å)", ylabel="概率密度 P(R_ee)", title="端到端距离分布"),
    ]
    res.add_curve(label, tx, ree, kind="line", panel=0)

    stat = describe_array(ree)
    res.add_curve("平均 R_ee", tx, np.full_like(tx, stat["mean"]), kind="line", panel=0)
    finite = ree[np.isfinite(ree)]
    if finite.size:
        edges = auto_bins(finite, nbins)
        hist, _ = np.histogram(finite, bins=edges, density=True)
        res.add_curve("R_ee 分布", bin_edges_to_centers(edges), hist, kind="bar", panel=1)

    res.summary.update(describe_array(ree, prefix="R_ee "))
    _bm, bsem = block_average(ree, n_blocks=min(5, max(2, ree.size // 2)))
    res.summary["R_ee 块平均标准误"] = float(bsem)
    res.add_notes("默认取所选原子组的第一个与最后一个原子作为链两端；"
                  "如不符合实际链端，请用 atom_indices 指定。")
    return res


@register("dihedral", "二面角分析")
def analyze_dihedrals(mdt, ag, selection: FrameSelection, *,
                      mode: str = "auto",
                      custom_indices: np.ndarray | None = None,
                      heavy_only: bool = True,
                      max_series: int = 5,
                      gauche_edge: float = 120.0,
                      nbins: int = 72,
                      verbose: bool = False,
                      label: str = "链") -> AnalysisResult:
    """二面角分析。

    参数
    ----
    mode
        - ``"auto"``：蛋白质用 phi/psi，其它体系用链骨架二面角
        - ``"phi_psi"``：蛋白质主链 φ/ψ
        - ``"chain"``：从键连接图提取链骨架，取连续四原子二面角
        - ``"topology"``：使用拓扑文件中已定义的二面角
        - ``"custom"``：使用 ``custom_indices``
    custom_indices
        形状 ``(n, 4)`` 的绝对原子索引数组。
    heavy_only
        ``chain`` / ``topology`` 模式下是否忽略含氢二面角。
    max_series
        在"二面角随时间变化"面板中最多画几条。
    gauche_edge
        trans / gauche 的分界角（度）。
    """
    u = mdt.universe
    times = np.asarray(selection.times_ps, dtype=float)
    note: list[str] = []

    # ---- 1. 确定二面角集合
    if mode == "auto":
        try:
            is_prot = ag.select_atoms("protein").n_atoms > 0
        except Exception:  # noqa: BLE001
            is_prot = False
        mode = "phi_psi" if is_prot else "chain"

    phi_idx = psi_idx = None
    idx = np.zeros((0, 4), dtype=int)
    if mode == "phi_psi":
        prot = ag.select_atoms("protein") if ag.select_atoms("protein").n_atoms else ag
        phi = []
        psi = []
        for res in prot.residues:
            try:
                a = res.phi_selection()
                if a is not None and a.n_atoms == 4:
                    phi.append([int(x.index) for x in a])
                b = res.psi_selection()
                if b is not None and b.n_atoms == 4:
                    psi.append([int(x.index) for x in b])
            except Exception:  # noqa: BLE001
                continue
        phi_idx = np.asarray(phi, dtype=int).reshape(-1, 4)
        psi_idx = np.asarray(psi, dtype=int).reshape(-1, 4)
        note.append(f"蛋白质主链 φ: {phi_idx.shape[0]} 个，ψ: {psi_idx.shape[0]} 个")
    elif mode == "chain":
        path = backbone_path(ag, heavy_only=heavy_only)
        idx = make_dihedral_indices(path)
        note.append(f"链骨架长度 {path.size} 原子，提取骨架二面角 {idx.shape[0]} 个")
    elif mode == "topology":
        idx = topology_dihedrals(ag, heavy_only=heavy_only)
        note.append(f"拓扑中属于所选原子的二面角 {idx.shape[0]} 个")
    elif mode == "custom":
        if custom_indices is None:
            raise ValueError("mode='custom' 时必须提供 custom_indices")
        idx = np.asarray(custom_indices, dtype=int)
        note.append(f"用户指定二面角 {idx.shape[0]} 个")
    else:
        raise ValueError(f"未知的二面角模式: {mode!r}")

    if mode == "phi_psi":
        series_phi = compute_dihedral_series(mdt, phi_idx, selection,
                                             unwrap_group=ag, verbose=verbose) \
            if phi_idx.size else np.zeros((times.size, 0))
        series_psi = compute_dihedral_series(mdt, psi_idx, selection,
                                             unwrap_group=ag, verbose=verbose) \
            if psi_idx.size else np.zeros((times.size, 0))
        series = np.concatenate([series_phi, series_psi], axis=1) if (
            series_phi.size or series_psi.size) else np.zeros((times.size, 0))
        groups = {"phi": series_phi, "psi": series_psi}
    else:
        series = compute_dihedral_series(mdt, idx, selection,
                                         unwrap_group=ag, verbose=verbose)
        groups = {"dihedral": series}

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="dihedral",
        title=f"二面角分析 —— {label}",
        meta={"selection": label, "mode": mode, "n_dihedrals": int(series.shape[1]),
              "n_frames": times.size, "gauche_edge": gauche_edge},
    )
    res.panels = [
        Panel(xlabel="二面角 φ (°)", ylabel="概率密度 P(φ)", title="二面角分布"),
        Panel(xlabel=tlabel, ylabel="构象比例", title="trans / gauche 构象比例随时间变化"),
        Panel(xlabel=tlabel, ylabel="二面角 φ (°)", title="部分二面角随时间变化"),
    ]

    # ---- 2. 分布
    for gname, arr in groups.items():
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        edges = np.linspace(-180.0, 180.0, int(nbins) + 1)
        hist, _ = np.histogram(arr[np.isfinite(arr)], bins=edges, density=True)
        res.add_curve(f"{gname} 分布", bin_edges_to_centers(edges), hist,
                      kind="bar", panel=0)
        st = circular_stats(arr)
        res.summary[f"{gname} 平均角 (°)"] = st["mean"]
        res.summary[f"{gname} 圆统计量 R"] = st["R"]
        cl = classify_dihedrals(arr, gauche_edge=gauche_edge)
        res.summary[f"{gname} trans 比例"] = cl["trans"]
        res.summary[f"{gname} gauche+ 比例"] = cl["gauche+"]
        res.summary[f"{gname} gauche- 比例"] = cl["gauche-"]
        res.summary[f"{gname} 二面角个数"] = int(arr.shape[1])

    # ---- 3. 构象比例随时间
    trans_t = np.full(times.size, np.nan)
    gp_t = np.full(times.size, np.nan)
    gm_t = np.full(times.size, np.nan)
    if series.shape[1]:
        for k in range(times.size):
            cl = classify_dihedrals(series[k], gauche_edge=gauche_edge)
            trans_t[k] = cl["trans"]
            gp_t[k] = cl["gauche+"]
            gm_t[k] = cl["gauche-"]
        res.add_curve("trans", tx, trans_t, panel=1)
        res.add_curve("gauche+", tx, gp_t, panel=1)
        res.add_curve("gauche-", tx, gm_t, panel=1)
        res.summary["trans 比例（时间平均）"] = float(np.nanmean(trans_t))
        res.summary["trans 比例标准差"] = float(np.nanstd(trans_t, ddof=1)) if times.size > 1 else 0.0

    # ---- 4. 部分二面角随时间
    if series.shape[1]:
        nshow = int(min(max_series, series.shape[1]))
        show = np.linspace(0, series.shape[1] - 1, nshow).round().astype(int)
        for j in show:
            res.add_curve(f"二面角 #{j + 1}", tx, series[:, j], panel=2)
        res.summary["二面角总数"] = int(series.shape[1])
        res.summary["二面角帧数"] = int(times.size)

    res.add_notes(*note)
    res.add_notes(f"trans 判据: |φ| > {gauche_edge:g}°")
    return res
