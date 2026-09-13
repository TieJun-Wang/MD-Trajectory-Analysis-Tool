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
                            notes: list | None = None,
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
    notes
        传一个 list 时，会把"展开组被自动扩充"这类情况写成说明追加进去
        （界面会显示），而不是静默处理。
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
    have = {int(a) for a in np.asarray(group.indices)}
    missing = sorted({int(a) for a in np.unique(indices)} - have)
    if missing:
        # 二面角跨出了分析对象：典型例子是**糖基化/带帽残基的 ψ** ——
        # 它需要"下一个 residue 的 N"，而那个 residue 属于另一条分子
        # （实测 md_biopolymer_nowater：protein 组分 3035 个原子，
        #  ψ 要用到糖基的原子 3042）。此时两种选择：
        #   (a) 丢掉这些二面角 —— 静默少算数据，二面角分布在链端就错了；
        #   (b) 把缺失原子所在的**整个共价分子**并入展开组 —— 展开本来就是
        #       按分子做的，共价相连的另一半并进来才是物理上正确的。
        # 这里选 (b)。若体系没有键信息，fragments 退化为单原子，等价于
        # 把这些原子自己加进来，不会引入跨分子的错误展开。
        try:
            extra = u.atoms[missing].fragments
            add = np.concatenate([np.asarray(f.indices, dtype=int) for f in extra])
        except Exception:  # noqa: BLE001
            add = np.asarray(missing, dtype=int)
        merged = np.union1d(np.asarray(group.indices, dtype=int), add)
        group = u.atoms[merged]
        if notes is not None:
            notes.append(
                f"二面角有 {len(missing)} 个原子落在分析对象之外（如糖基化残基的 ψ "
                f"需要下一个 residue 的 N），已把其所在的整个共价分子并入 PBC 展开组"
                f"（展开原子 {len(have)} → {group.n_atoms} 个）")
        if verbose:
            print(f"[二面角] 展开组已扩充：+{len(missing)} 个越界原子 → "
                  f"{group.n_atoms} 个原子")

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
# ------------------------------------------------------- 分子对象 / 链端识别
def molecule_slices(ag) -> tuple[np.ndarray, list[np.ndarray], list[str]] | None:
    """把原子组按**分子**（``molnums``）拆开。

    返回 ``(原子级分子编号, [每个分子的组内下标, ...], [标签, ...])``；
    体系没有分子信息时返回 ``None``。

    为什么需要：Rg / R_ee 这类"链尺寸"量必须**按分子**算。对一个含 541 个分子的
    组分（如 46 体系的 polymer）直接算整组的 Rg，得到的是**整团物质**的回转半径
    （溶液里它接近盒子尺度），而不是"分子有多大"——这就是"这个数到底是谁的数"。
    """
    from .interface import _atom_molnums

    m = _atom_molnums(ag)
    if m is None:
        return None
    uniq = np.unique(m)
    slices: list[np.ndarray] = []
    labels: list[str] = []
    for mol in uniq:
        sl = np.flatnonzero(m == mol)
        slices.append(sl)
        try:
            r = ag[int(sl[0])].residue
            labels.append(f"{r.resname}{r.resid}")
        except Exception:  # noqa: BLE001
            labels.append(f"分子{int(mol)}")
    return m, slices, labels


def _atom_label(u, index: int) -> str:
    """原子的可读标签（残基名+编号+原子名），便于用户核对链端选得对不对。"""
    try:
        a = u.atoms[int(index)]
        return f"{a.resname}{a.resid}({a.name})"
    except Exception:  # noqa: BLE001
        return f"原子{int(index)}"


def _protein_termini(ag) -> tuple[int, int] | None:
    """蛋白质主链端基：首个残基的 N 与末个残基的 C（缺失时退到 CA）。

    为什么蛋白质不能按键图找端：蛋白质里"度为 1 的原子"极多（每个末端 H、
    每个侧基的末端原子都是），图直径往往穿过最长的那条侧链——实测 AdK 会选到
    MET1 的侧链氢，那不是链端。折叠蛋白的"端到端距离"标准定义是**主链两端**。
    """
    try:
        prot = ag.select_atoms("protein")
    except Exception:  # noqa: BLE001
        return None
    if prot.n_atoms == 0 or prot.n_atoms < 0.9 * ag.n_atoms or len(prot.residues) < 2:
        return None

    def pick(res, names: tuple[str, ...]):
        for nm in names:
            try:
                sel = res.atoms.select_atoms(f"name {nm}")
            except Exception:  # noqa: BLE001
                continue
            if sel.n_atoms:
                return int(sel.indices[0])
        return None

    a = pick(prot.residues[0], ("N", "CA"))
    b = pick(prot.residues[-1], ("C", "CA", "O"))
    if a is None or b is None or a == b:
        return None
    return a, b


def bond_graph_ends(ag, heavy_only: bool = False) -> dict:
    """找链两端（端到端距离的正确对象）。

    判定顺序：

    1. 没有键信息 → ``ok=False``（给出 :data:`NO_BOND_HINT`）；
    2. 图不连通（组里有多个分子/链）→ ``ok=False``，报告连通分量数，
       由调用方改成"逐分子算 R_ee"；
    3. **蛋白质** → 主链端基（首个残基 N / 末个残基 C）：蛋白质的"度为 1
       原子"包含全部末端氢与侧基末端原子，按键图找端会选到侧链，没意义；
    4. 恰好 2 个度为 1 的原子 → 它们就是链端（线形高分子的正确情形）；
    5. 多于 2 个度为 1（支化链/带侧基）→ 用两次 BFS（图的直径）取**最远的一对**
       作端，并标记 ``branched=True``；
    6. 没有度为 1 的原子（环状/交联网络）→ ``ok=False``：
       **端到端距离对环状分子没有定义**，不能硬算一个数出来。

    返回 ``{ok, ends, reason, method, n_components, n_degree1, branched}``。
    """
    from collections import deque

    idx = np.asarray(ag.indices)
    n = idx.size
    out = {"ok": False, "ends": None, "reason": "", "n_components": 0,
           "n_degree1": 0, "branched": False}
    if n < 2:
        out["reason"] = "原子数少于 2，谈不上端到端距离"
        return out
    bonds = _bonds_of(ag)
    if bonds is None:
        out["reason"] = "拓扑没有键连接信息，无法判断链端。" + NO_BOND_HINT
        return out

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
            keep = {i for i in range(n) if el[i] not in ("H", "")}
        except Exception:  # noqa: BLE001
            keep = set(range(n))
        if 0 < len(keep) < n:
            adj = [[x for x in row if x in keep] for row in adj]
            for i in range(n):
                if i not in keep:
                    adj[i] = []

    # 连通分量
    seen = np.zeros(n, dtype=bool)
    n_comp = 0
    for s in range(n):
        if seen[s] or not adj[s]:
            continue
        n_comp += 1
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
    lonely = int(np.sum(~seen))
    out["n_components"] = n_comp
    if n_comp > 1:
        out["reason"] = (f"该组包含 {n_comp} 个互不连接的分子/链"
                         + (f"（另有 {lonely} 个孤立原子）" if lonely else "")
                         + "，整组的端到端距离没有定义，已改为逐分子计算。")
        return out

    deg1 = [i for i in range(n) if len(adj[i]) == 1]
    out["n_degree1"] = len(deg1)

    # 蛋白质优先走主链端基（见 _protein_termini 的说明）
    pt = _protein_termini(ag)
    if pt is not None:
        out["ok"] = True
        out["ends"] = np.asarray(pt, dtype=int)
        out["method"] = "蛋白质主链端基（首个残基 N / 末个残基 C）"
        return out

    if len(deg1) >= 2:
        if len(deg1) > 2:
            out["branched"] = True
            # 支化链：取图直径的两端（两次 BFS），比任取两个端基更接近"最长跨度"
            def bfs(src: int):
                dist = {src: 0}
                prev = {src: None}
                q = deque([src])
                far = src
                while q:
                    uu = q.popleft()
                    for vv in adj[uu]:
                        if vv not in dist:
                            dist[vv] = dist[uu] + 1
                            prev[vv] = uu
                            q.append(vv)
                            if dist[vv] > dist[far]:
                                far = vv
                return far, prev
            u, _ = bfs(deg1[0])
            v, _ = bfs(u)
            pair = (u, v)
        else:
            pair = (deg1[0], deg1[1])
        out["ok"] = True
        out["ends"] = np.asarray([int(idx[pair[0]]), int(idx[pair[1]])], dtype=int)
        out["method"] = ("键图度 1 的两个端原子" if not out["branched"]
                         else f"键图直径两端（该组有 {len(deg1)} 个度为 1 的原子，"
                              f"属支化链/带侧基，取最远的一对）")
        return out

    # 度为 1 的原子为 0：环状或全连接网络
    out["reason"] = ("所有原子的键连度都 ≥ 2，说明该组是**环状分子或交联网络**，"
                     "没有端基——端到端距离对这类对象没有定义，因此不给出该数。"
                     "若确实要看链尺寸，请改用回转半径 Rg，或用 atom_indices 显式"
                     "指定你想量的两个原子。")
    return out


@register("rg", "回转半径 Rg")
def analyze_rg(mdt, ag, selection: FrameSelection, *,
               mass_weighted: bool = True, unwrap: bool = True,
               nbins: int | None = None, verbose: bool = False,
               label: str = "链", per_molecule: bool = True,
               max_molecules_drawn: int = 400) -> AnalysisResult:
    """回转半径分析（**按分子**统计）。

    统计口径（关键）
    ----------------
    * 组分里只有 1 个分子（如一条蛋白链）→ 整组 Rg **就是**该分子的 Rg；
    * 组分里有 N 个分子（如 46 体系的 polymer = 541 个分子）→ 除整组 Rg 外，
      **逐个分子算 Rg**，给出分子间的分布与平均。

    为什么必须按分子：分子数 > 1 时，"整组 Rg"是**整团物质**的回转半径
    （溶液中它接近盒子尺度、随浓度变化），而"分子有多大"是完全不同的量。
    1.0.0 只给整组那一个数，用户会把后者当成前者读。
    """
    u = mdt.universe
    times = np.asarray(selection.times_ps, dtype=float)
    n_frame = times.size
    rg = np.full(n_frame, np.nan)
    mass_note = ""
    masses = None
    if mass_weighted:
        try:
            m_all = np.asarray(ag.masses, dtype=float)
            if (m_all.size == ag.n_atoms and bool(np.all(np.isfinite(m_all)))
                    and float(m_all.sum()) > 0):
                masses = m_all
        except Exception:  # noqa: BLE001 - 无 masses 属性
            masses = None
        if masses is None:
            mass_note = ("该组分没有可用的质量信息，已退化为**等权重（几何）**"
                         "回转半径。若拓扑带 masses 或 elements，本工具会自动由元素"
                         "补出质量并改用质量加权。")

    ms = molecule_slices(ag) if per_molecule else None
    if ms is not None:
        _m, slices, mol_labels = ms
        n_mol = len(slices)
    else:
        slices, mol_labels, n_mol = [], [], 1
    multi = n_mol > 1
    rg_mol = np.full((n_frame, n_mol), np.nan)

    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        # 整组只展开一次：positions_for 内部按连通分量（即按分子）聚拢，
        # 因此逐分子 Rg 可以直接在这份坐标上切片，无需重复展开。
        pos = positions_for(ag, unwrap=unwrap)
        rg[k] = compute_rg(pos, masses)
        if multi:
            for j, sl in enumerate(slices):
                sub_m = None if masses is None else masses[sl]
                rg_mol[k, j] = compute_rg(pos[sl], sub_m)

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="rg",
        title=f"回转半径 Rg —— {label}",
        meta={"selection": label, "n_atoms": int(ag.n_atoms),
              "mass_weighted": bool(mass_weighted), "unwrap": bool(unwrap),
              "n_frames": times.size, "n_molecules": int(n_mol),
              "per_molecule": bool(multi)},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="回转半径 Rg (Å)", title="Rg 随时间变化"),
        Panel(xlabel="回转半径 Rg (Å)", ylabel="概率密度 P(Rg)", title="Rg 分布"),
    ]
    if multi:
        res.panels.append(
            Panel(xlabel="分子序号（按平均 Rg 升序）", ylabel="平均 Rg (Å)",
                  title="各分子的平均回转半径"))

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
    res.summary["分子数"] = int(n_mol)

    if multi:
        mol_mean = np.nanmean(rg_mol, axis=0)                  # 每个分子的时间平均 Rg
        ok = np.isfinite(mol_mean)
        res.summary["统计口径"] = "整组 Rg + 逐分子 Rg"
        res.summary["单分子 Rg 平均 (Å)"] = float(np.mean(mol_mean[ok])) if ok.any() else float("nan")
        res.summary["单分子 Rg 标准差 (Å)"] = float(np.std(mol_mean[ok])) if ok.any() else float("nan")
        res.summary["单分子 Rg 最小 (Å)"] = float(np.min(mol_mean[ok])) if ok.any() else float("nan")
        res.summary["单分子 Rg 最大 (Å)"] = float(np.max(mol_mean[ok])) if ok.any() else float("nan")
        # 分子间平均值的分布（时间与分子两个维度的变化分开呈现）
        if ok.any():
            e2 = auto_bins(mol_mean[ok], nbins)
            h2, _ = np.histogram(mol_mean[ok], bins=e2, density=True)
            res.add_curve("单分子平均 Rg 分布", bin_edges_to_centers(e2), h2,
                          kind="bar", panel=1)
        order = np.argsort(np.where(ok, mol_mean, np.inf))
        n_draw = min(int(max_molecules_drawn), n_mol)
        res.add_curve("各分子平均 Rg", np.arange(1, n_draw + 1),
                      np.where(ok, mol_mean, np.nan)[order[:n_draw]],
                      kind="bar", panel=2)
        res.add_notes(
            f"该组分含 **{n_mol} 个分子**：上面的「{label}」与「Rg 」统计量是"
            f"**整组**（{ag.n_atoms:,} 个原子一起）的回转半径——溶液中它接近盒子"
            f"尺度、并随浓度变化，**不等于「分子有多大」**；"
            f"「单分子 Rg 平均」= {res.summary['单分子 Rg 平均 (Å)']:.3f} Å "
            f"（±{res.summary['单分子 Rg 标准差 (Å)']:.3f} Å，分子间）才是逐分子算出的"
            f"链尺寸。两者不可混用。")
        res.add_notes(
            f"各分子平均 Rg 的分布见「Rg 分布」面板中的另一条曲线；"
            f"逐分子数值见「各分子的平均回转半径」面板（最多画 "
            f"{n_draw} 个，按升序排列）。")
        if n_mol > 1 and res.summary["单分子 Rg 标准差 (Å)"] > 0.3 * res.summary["单分子 Rg 平均 (Å)"]:
            res.add_notes(
                "分子间 Rg 的相对离散度超过 30%，说明体系里各分子的链尺寸差别很大"
                "（存在不同聚合度/构象亚群），只报一个平均值会掩盖这种多分散性。")
    else:
        res.summary["统计口径"] = "整组（该组即 1 个分子）"
        if per_molecule:
            res.add_notes("该组分只含 1 个分子，因此整组 Rg 就是该分子的回转半径。")
        else:
            res.add_notes("已按 per_molecule=False 关闭逐分子统计。")

    res.add_notes(f"质量加权: {mass_weighted}；PBC 展开: {unwrap}")
    res.add_notes(f"参与统计原子数: {ag.n_atoms}")
    if mass_note:
        res.add_notes(mass_note)
    return res


@register("ree", "端到端距离 R_ee")
def analyze_end_to_end(mdt, ag, selection: FrameSelection, *,
                       atom_indices: Sequence[int] | None = None,
                       ends: str = "bond_graph",
                       unwrap: bool = True, nbins: int | None = None,
                       verbose: bool = False, label: str = "链",
                       per_molecule: bool = True,
                       max_molecules_drawn: int = 400) -> AnalysisResult:
    """端到端距离分析（链端**由键连接图确定**）。

    参数
    ----
    atom_indices
        显式指定两个端原子的绝对编号，优先级最高（用于复现旧口径或特殊对象）。
    ends
        ``"bond_graph"``（默认）→ 由键图找端原子；``"selection"`` → 沿用 1.0.0 的
        旧口径（所选原子组的第一个与最后一个原子），仅用于对照。
    per_molecule
        组内含多个分子时逐分子各算自己的 R_ee（各自找端原子）。

    为什么必须改：1.0.0 取的是"所选原子组里索引最小与最大的两个原子"
    （``ag[[0, -1]]``）。对一条蛋白链，这两个原子只是**文件里的第一个和最后一个
    原子**，通常不是主链两端；对含多个分子的组分更完全没有意义。实测 AdK 蛋白：
    旧口径 9.911 Å（首尾原子），键图端原子给出的是真正的主链 N 端/C 端间距。
    """
    u = mdt.universe
    times = np.asarray(selection.times_ps, dtype=float)
    n_frame = times.size
    notes: list[str] = []

    # ------------------------------------------------ 端原子集合的确定
    ms = molecule_slices(ag) if (per_molecule and atom_indices is None) else None
    pairs: list[tuple[str, int, int]] = []          # (标签, 端原子1, 端原子2)
    end_method = "键连接图端原子"
    if atom_indices is not None:
        if len(atom_indices) < 2:
            raise ValueError("atom_indices 需要两个绝对原子编号")
        pairs = [("用户指定", int(atom_indices[0]), int(atom_indices[1]))]
        end_method = "用户指定"
        notes.append(f"使用 atom_indices 显式指定的两个原子作为链端："
                     f"{_atom_label(u, pairs[0][1])} – {_atom_label(u, pairs[0][2])}。")
    elif str(ends) == "selection":
        pairs = [(label, int(ag.indices[0]), int(ag.indices[-1]))]
        end_method = "旧口径（所选原子组首尾原子）"
        notes.append("⚠️ 使用旧口径（所选原子组的首尾原子）作为链端——"
                     "这通常**不是**真实链端，仅用于与 1.0.0 对照。")
    elif ms is not None and len(ms[1]) > 1:
        _m, slices, mol_labels = ms
        skipped = 0
        for sl, name in zip(slices, mol_labels):
            sub = ag[sl]
            info = bond_graph_ends(sub)
            if not info["ok"] or info["ends"] is None:
                skipped += 1
                continue
            pairs.append((name, int(info["ends"][0]), int(info["ends"][1])))
        if skipped:
            notes.append(f"有 {skipped} 个分子（环状/无端基/缺键信息）无法定义端到端"
                         f"距离，已跳过，未计入统计。")
        if pairs:
            notes.append(f"该组分含 {len(ms[1])} 个分子，已**逐个分子**用自己的"
                         f"键图端原子计算 R_ee（有效 {len(pairs)} 个分子）。")
    else:
        info = bond_graph_ends(ag)
        if not info["ok"] or info["ends"] is None:
            res = AnalysisResult(
                name="ree",
                title=f"端到端距离 R_ee —— {label}",
                meta={"selection": label, "n_atoms": int(ag.n_atoms),
                      "n_frames": n_frame, "defined": False},
            )
            res.panels = [
                Panel(xlabel=time_axis(times)[1], ylabel="端到端距离 R_ee (Å)",
                      title="端到端距离随时间变化"),
            ]
            res.summary["R_ee 是否可定义"] = "否"
            res.summary["原因"] = str(info["reason"])
            res.add_notes(f"**未给出 R_ee**：{info['reason']}")
            res.add_notes(f"参与统计原子数: {ag.n_atoms}")
            return res
        pairs = [(label, int(info["ends"][0]), int(info["ends"][1]))]
        end_method = str(info.get("method", "键连接图端原子"))
        notes.append(f"链端来源：{end_method}。选中的两个原子是 "
                     f"{_atom_label(u, pairs[0][1])} – {_atom_label(u, pairs[0][2])}"
                     f"（1.0.0 取的是「所选原子组首尾原子」，通常并不是链端）。")
        if info.get("n_degree1", 0) > 2 and "直径" in end_method:
            notes.append(f"该组有 {info['n_degree1']} 个度为 1 的原子（支化链或带侧基），"
                         f"取的是键图直径两端；若要量特定两个原子，请用 atom_indices "
                         f"显式指定。")

    # ------------------------------------------------ 逐帧计算
    n_pair = len(pairs)
    ree = np.full((n_frame, n_pair), np.nan)
    for p, (name, i1, i2) in enumerate(pairs):
        pair_ag = u.atoms[[i1, i2]]
        for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
            pp = positions_for(pair_ag, unwrap=unwrap)
            if pp.shape[0] < 2:
                continue
            ree[k, p] = compute_end_to_end(pp[0], pp[-1],
                                           box=None if unwrap else u.dimensions)

    main = np.nanmean(ree, axis=1) if n_pair > 1 else ree[:, 0]
    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="ree",
        title=f"端到端距离 R_ee —— {label}",
        meta={"selection": label, "n_atoms": int(ag.n_atoms), "n_frames": n_frame,
              "n_pairs": int(n_pair), "ends": str(ends),
              "end_atoms": [[int(i1), int(i2)] for _n, i1, i2 in pairs]},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="端到端距离 R_ee (Å)", title="端到端距离随时间变化"),
        Panel(xlabel="端到端距离 R_ee (Å)", ylabel="概率密度 P(R_ee)",
              title="端到端距离分布"),
    ]
    curve_label = label if n_pair == 1 else f"{label}（{n_pair} 个分子平均）"
    res.add_curve(curve_label, tx, main, kind="line", panel=0)
    stat = describe_array(main)
    res.add_curve("平均 R_ee", tx, np.full_like(tx, stat["mean"]), kind="line", panel=0)

    finite = main[np.isfinite(main)]
    if finite.size:
        edges = auto_bins(finite, nbins)
        hist, _ = np.histogram(finite, bins=edges, density=True)
        res.add_curve("R_ee 分布", bin_edges_to_centers(edges), hist, kind="bar", panel=1)

    res.summary.update(describe_array(main, prefix="R_ee "))
    _bm, bsem = block_average(main, n_blocks=min(5, max(2, main.size // 2)))
    res.summary["R_ee 块平均标准误"] = float(bsem)
    res.summary["R_ee 是否可定义"] = "是"
    res.summary["链端来源"] = end_method
    res.summary["参与统计的链数"] = int(n_pair)
    if n_pair == 1:
        res.summary["链端原子"] = (f"{_atom_label(u, pairs[0][1])} – "
                                f"{_atom_label(u, pairs[0][2])}")
    else:
        per_mol = np.nanmean(ree, axis=0)
        okm = np.isfinite(per_mol)
        res.summary["单链 R_ee 平均 (Å)"] = float(np.mean(per_mol[okm])) if okm.any() else float("nan")
        res.summary["单链 R_ee 标准差 (Å)"] = float(np.std(per_mol[okm])) if okm.any() else float("nan")
        order = np.argsort(np.where(okm, per_mol, np.inf))
        n_draw = min(int(max_molecules_drawn), n_pair)
        res.panels.append(Panel(xlabel="链序号（按平均 R_ee 升序）", ylabel="平均 R_ee (Å)",
                                title="各链的平均端到端距离"))
        res.add_curve("各链平均 R_ee", np.arange(1, n_draw + 1),
                      np.where(okm, per_mol, np.nan)[order[:n_draw]],
                      kind="bar", panel=2)

    for nt in notes:
        res.add_notes(nt)
    res.add_notes(f"参与统计原子数: {ag.n_atoms}；PBC 展开: {unwrap}")
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
        - ``"auto"``：蛋白质用 phi/psi，其它体系用几何连续原子模式
        - ``"phi_psi"``：蛋白质主链 φ/ψ（对蛋白质这是**唯一**有明确化学含义的二面角）
        - ``"chain"``：**几何连续原子模式**——沿键连接图的最长路径取连续四原子
          二面角。注意它给的是"几何上连成一串的原子"之间的二面角，**不保证**
          等于化学意义上的主链/重复单元二面角（例如支化或带长侧基的分子，
          最长路径可能拐进侧基）。名称由 1.0.0 的"链骨架"更正为现在的说法。
        - ``"topology"``：使用拓扑文件中已定义的二面角
        - ``"custom"``：使用 ``custom_indices``
    custom_indices
        形状 ``(n, 4)`` 的绝对原子索引数组。
    heavy_only
        ``chain`` / ``topology`` 模式下是否忽略含氢二面角。
    max_series
        在"二面角随时间变化"面板中最多画几条。
    gauche_edge
        trans / gauche 的分界角（**度**，默认 120）：``|φ| > gauche_edge`` 记为
        trans，其余按符号分 gauche+ / gauche−。不同力场/文献取 110～120 不等，
        因此做成参数而不是写死；结果里会写明本次用的阈值。
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
        note.append(f"**几何连续原子模式**：键图最长路径 {path.size} 个原子，"
                    f"提取连续四原子二面角 {idx.shape[0]} 个"
                    f"（这是几何上连成一串的原子，不保证等于化学主链/重复单元）")
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
                                             unwrap_group=ag, notes=note,
                                             verbose=verbose) \
            if phi_idx.size else np.zeros((times.size, 0))
        series_psi = compute_dihedral_series(mdt, psi_idx, selection,
                                             unwrap_group=ag, notes=note,
                                             verbose=verbose) \
            if psi_idx.size else np.zeros((times.size, 0))
        series = np.concatenate([series_phi, series_psi], axis=1) if (
            series_phi.size or series_psi.size) else np.zeros((times.size, 0))
        groups = {"phi": series_phi, "psi": series_psi}
    else:
        series = compute_dihedral_series(mdt, idx, selection,
                                         unwrap_group=ag, notes=note,
                                         verbose=verbose)
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
