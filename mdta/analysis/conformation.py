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

from typing import NamedTuple, Sequence

import numpy as np

from ..core import AnalysisResult, Panel, block_average, describe_array
from ..preprocess import FrameSelection
from ..units import auto_bins, bin_edges_to_centers, time_axis
from .base import frame_iterator, positions_for, register

__all__ = [
    "compute_rg",
    "analyze_rg",
    "normalize_rg_object",
    "RG_OBJECT_LABELS",
    "analyze_end_to_end",
    "analyze_dihedrals",
    "circular_stats",
    "classify_dihedrals",
    "backbone_path",
    "topology_dihedrals",
]


# ------------------------------------------------------------------ 基本算法

class MoleculeGroups(NamedTuple):
    """一条轨迹上**不变**的分子划分（算一次、逐帧复用）。"""

    inv: np.ndarray          #: 原子级分子下标，取值 ``0..n_mol-1``
    n_mol: int               #: 分子数
    order: np.ndarray        #: 原子按分子排好序的次序（供 ``np.maximum.reduceat`` 用）
    starts: np.ndarray       #: 每个分子在 ``order`` 中的起点


def prepare_molecule_groups(mol_ids) -> MoleculeGroups:
    """把分子划分**算一次**供整条轨迹复用。

    为什么要缓存：``np.unique`` + ``argsort`` 在 4636 原子上各要几十微秒，逐帧做
    就是白花几秒；而分子划分（哪些原子属于哪个分子）在一条轨迹上是不变的。
    """
    ids = np.asarray(mol_ids).ravel()
    uniq, inv = np.unique(ids, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(int(uniq.size)))
    return MoleculeGroups(inv, int(uniq.size), order, np.asarray(starts, dtype=np.intp))


def rg_fast(pos, masses, box, mol_ids=None, *, groups=None, ag=None, stats=None):
    """**向量化的 PBC 展开** + 质量加权 Rg（多分子对象走这条路，快一两个数量级）。

    做法（与 MDAnalysis ``unwrap(compound='fragments', reference='com')`` 同一约定）：

    1. 用**包裹坐标**算每个分子的质量加权质心 ``com_i``（``bincount``，全程向量化）；
    2. 每个原子相对自己分子质心取**最小镜像位移** ``s = MIN(p − com_i)``；
    3. 展开后坐标 ``d = com_i + s`` —— 这就是 MDAnalysis 逐碎片 make_whole 做的事，
       只是没有 Python 层的逐碎片循环；
    4. 再把每个分子的质心按 ``apply_PBC`` 搬回主盒（MDAnalysis ``reference='com'``
       的收尾步骤）—— 少了这一步，整组 Rg 会落在另一套镜像约定上。

    返回 ``(整组 Rg, 逐分子 Rg, safe, 展开后的坐标 d)``。

    * **逐分子 Rg** 与 MDAnalysis 逐碎片展开逐位同源：对 200×30 的 PEG 体系与
      纯键连通重建（``mdamath.make_whole``）比，最大偏差 1.1e-09 Å；
    * **整组 Rg** 与 ``positions_for(ag, unwrap=True)``（精确路径）同一约定，
      差别只剩 MDAnalysis 内部把碎片质心转成 float32 的舍入（实测 ≤1.2e-07 Å）。

    ``safe=False`` 的两种情况
    -------------------------
    1. 有原子在**自己分子的质心**外超过 ``0.4×最短盒边``（分子尺度接近半盒，
       最小镜像不再等价）——用户 122C12 体系里 960 原子的那个大分子（跨度 50.8 Å，
       盒边 49.2 Å）会踩到；
    2. 上面这种分子``mdamath.make_whole`` 也修不了（没键、碎片不连通）。

    为此 ``rg_fast`` 支持**逐分子**判定：只把踩线的那几个分子交给 MDAnalysis 的
    键连通重建（``ag`` 给出来时才做），其余分子照走向量化路径 —— 于是"全部原子"
    这种混着大分子的选择也不必整组退回慢路径（实测 29.3 ms/帧 → 1.0 ms/帧）。
    修掉的分子序号会写进 ``stats["repaired_molecules"]``（若给了 ``stats``）。

    ⚡ 性能（1.0.3 重写，实测用户体系 4636 原子 / 122 分子）：**0.37 ms/帧**
    （6001 帧 2.2 s；精确路径 18 ms/帧）。慢在三个"看着像向量化其实不是"的地方，
    都已去掉：

    * ``np.maximum.at``（求逐分子最大位移）是**无缓冲** ufunc.at，逐元素开销 ~1 µs
      → 4636 原子 4.6 ms → 换成 ``np.maximum.reduceat``（按分子排好序的次序）；
    * 逐分子 Rg 用了 Python 的 ``for j in range(n_mol)`` + 布尔掩码 → 换成两次
      ``np.bincount``（分子内 Σw|s|² 与 Σw）；
    * 每帧都 ``np.unique(mol_ids)`` → 分子划分在整条轨迹上不变，调用方用
      :func:`prepare_molecule_groups` 算一次传进来（``groups=``）。
    """
    from MDAnalysis.lib.distances import minimize_vectors

    pos = np.asarray(pos, dtype=float)
    if pos.size == 0:
        return float("nan"), None, False, pos
    w = np.ones(pos.shape[0], dtype=float) if masses is None else np.asarray(masses, float)
    lim = np.inf
    if box is not None:
        L = np.asarray(box, dtype=float)[:3]
        lim = 0.4 * float(np.min(L[L > 0])) if np.any(L > 0) else np.inf

    if mol_ids is None and groups is None:
        d = pos - pos[0]
        if box is not None:
            d = minimize_vectors(d, box=box)
        if float(np.max(np.linalg.norm(d - d.mean(axis=0), axis=1))) >= lim:
            return float("nan"), None, False, pos
        return _rg_of(d, w), None, True, d

    if groups is None:
        groups = prepare_molecule_groups(mol_ids)
    inv, n_mol, order, starts = groups
    n_mol = int(n_mol)
    if n_mol <= 0 or starts.size != n_mol:
        return float("nan"), None, False, pos
    msum = np.bincount(inv, weights=w, minlength=n_mol)
    safe_den = np.where(msum > 0, msum, 1.0)

    com_w = np.stack([np.bincount(inv, weights=w * pos[:, k], minlength=n_mol)
                      / safe_den for k in range(3)], axis=1)
    shift = pos - com_w[inv]
    _min_image_inplace(shift, box)
    s2 = np.einsum("ij,ij->i", shift, shift)          # |shift|²，逐原子

    # ---- 安全判据（**逐分子**，用 |s|² 比 lim²，省一次逐原子 sqrt）----------
    if lim != np.inf:
        mx2 = np.maximum.reduceat(s2[order], starts)  # 每个分子的 max|s|²
        bad = np.flatnonzero(mx2 >= lim * lim)
        if bad.size:
            if ag is None or not _repair_large_molecules(shift, s2, com_w, ag, inv,
                                                         bad, w):
                return float("nan"), None, False, pos
            if stats is not None:
                stats.setdefault("repaired_molecules", set()).update(int(j) for j in bad)

    d = com_w[inv] + shift
    # 逐分子 Rg：Σ w|d − com_own|² = Σ w|shift|²，两次 bincount 搞定
    num = np.bincount(inv, weights=w * s2, minlength=n_mol)
    rg_mol = np.sqrt(np.maximum(num, 0.0) / safe_den)
    # ---- 与 MDAnalysis `unwrap(compound='fragments', reference='com')` 对齐 ----
    # MDAnalysis 在 make_whole 之后还会把**每个碎片的质量加权质心**按周期性边界
    # 搬回主盒（`target = apply_PBC(com); positions += target - com`）。只重排分子
    # 内部坐标、不搬质心，会让"整组 Rg"落在另一套镜像约定上 —— 实测用户 122C12
    # 体系同一帧整组 Rg 会差 26.8 Å vs 88.9 Å（差 3.3 倍），而两条路径（快/精确）
    # 谁被选中取决于分子尺度，于是**同一个选择在不同路径下给出不同的整组 Rg**。
    # 这里把这一步补上：实测整组 Rg 与精确路径差 8.3e-08 Å（此前 5.7e-03）。
    if box is not None:
        com_d = np.stack([np.bincount(inv, weights=w * d[:, k], minlength=n_mol)
                          / safe_den for k in range(3)], axis=1)
        target = _apply_pbc(com_d, box)
        d = d + (target - com_d)[inv]
    return _rg_of(d, w), rg_mol, True, d


def _repair_large_molecules(shift, s2, com_w, ag, inv, bad, w) -> bool:
    """对"尺度接近半盒"的少数分子改用 MDAnalysis 的**键连通重建**。

    ``bad`` 里的分子用 ``mdamath.make_whole`` 重排内部坐标，写回 ``shift``/``s2``，
    并把该分子的**锚点** ``com_w[j]`` 换成重建后的质量加权质心 —— 这一步不能省：
    跨盒分子的"包裹坐标平均"（``com_w``）并不等于真实质心的任何镜像（它是盒子中间
    那个假位置），锚在它上面会让该分子被放到错误的镜像里，整组 Rg 差 0.98 Å（实测）。
    锚到 ``make_whole`` 的质心后，展开坐标 = ``wh`` 本身，收尾的 apply_PBC 再把它
    搬到与 MDAnalysis 完全相同的位置。

    修不了（没键信息/碎片不连通）返回 ``False``，调用方退回整组精确路径。
    """
    from MDAnalysis.lib import mdamath

    for j in bad:
        j = int(j)
        sel = np.flatnonzero(inv == j)
        if sel.size == 0:
            continue
        try:
            wh = np.asarray(mdamath.make_whole(ag[sel], inplace=False), dtype=float)
        except Exception:  # noqa: BLE001 - 无键 / 非单一连通碎片
            return False
        wj = w[sel]
        cj = (wj @ wh) / wj.sum() if float(wj.sum()) > 0 else wh.mean(axis=0)
        shift[sel] = wh - cj
        s2[sel] = np.einsum("ij,ij->i", shift[sel], shift[sel])
        com_w[j] = cj
    return True


def _min_image_inplace(shift: np.ndarray, box) -> None:
    """把位移就地取最小镜像（正交盒走 numpy，三斜盒交给 MDAnalysis）。

    为什么要自己写：``minimize_vectors`` 是 Cython 逐矢量循环，4636 个原子要
    0.33 ms（占 ``rg_fast`` 四成）；正交盒的最小镜像就是
    ``x − round(x/L)·L``，一次向量化调用几十微秒，结果与 MDAnalysis 逐位相同
    （5007 个位移含 ±L/2 边界情形，最大差 4.0e-15，见 ``_probe_rg_fast4.py``）。
    三斜盒仍需按盒矢量分解，退回 MDAnalysis 的实现。
    """
    if box is None:
        return
    b = np.asarray(box, dtype=float)
    L = b[:3]
    if not bool(np.all(L > 0)):
        return
    if not bool(np.all(np.abs(b[3:] - 90.0) < 1e-4)):
        from MDAnalysis.lib.distances import minimize_vectors

        shift[:] = minimize_vectors(shift, box=box)
        return
    # 与 MDAnalysis `_minimize_vectors_ortho` **逐位同规则**：``x − round(x/L)·L``，
    # 其中 round 是 C 的 `round()`（**半数远离零**），不是 numpy 默认的半数取偶
    # （``np.rint(0.5) == 0``）。差别只在"原子恰好落在质心 ±L/2"这种测度为零的
    # 情形（结果差一个盒长），但能对齐就不留这个不一致。
    # 半数远离零 = ``sign(x)·floor(|x| + 0.5)``。
    q = np.abs(shift) / L
    q += 0.5
    np.floor(q, out=q)
    q *= np.sign(shift)
    shift -= q * L


def _apply_pbc(com: np.ndarray, box) -> np.ndarray:
    """把每个分子的质心搬回主盒（与 MDAnalysis ``apply_PBC`` 同一约定）。"""
    try:
        from MDAnalysis.lib.distances import apply_PBC

        return apply_PBC(com, box)
    except Exception:  # noqa: BLE001 - 盒信息退化时按正交盒手工回卷
        L = np.asarray(box, dtype=float)[:3]
        good = L > 0
        out = np.array(com, dtype=float)
        out[:, good] -= np.floor(out[:, good] / L[good]) * L[good]
        return out


def _rg_of(pos, w):
    """给定坐标与权重算质量加权 Rg（与 :func:`compute_rg` 同一公式）。"""
    wsum = float(w.sum())
    if wsum <= 0 or pos.shape[0] == 0:
        return float("nan")
    com = (w @ pos) / wsum                      # matvec 比 (w[:,None]*pos).sum(0) 快
    dd = pos - com
    return float(np.sqrt(float(w @ np.einsum("ij,ij->i", dd, dd)) / wsum))


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


#: 「分析对象」的两种模式（**互斥**）：整体体系 / 分子·链
RG_OBJECT_LABELS = {"molecule": "分子 / 链", "system": "整体体系"}


def normalize_rg_object(value) -> str:
    """把「分析对象」的取值归一化成 ``"molecule"`` / ``"system"``。

    接受界面传来的中文（``"整体体系"`` / ``"分子 / 链"``）与英文
    （``"system"`` / ``"molecule"``、``"chain"``…），其余一律报错 —— 与其
    猜一个,不如让调用方看到可用的取值（静默兜底会让人以为开关生效了）。
    """
    s = str("molecule" if value is None else value).strip().lower().replace(" ", "")
    if s in {"system", "whole", "overall", "all", "cluster", "整体", "整体体系",
             "整组", "全体系", "整个体系"}:
        return "system"
    if s in {"molecule", "molecules", "mol", "chain", "chains", "per_molecule",
             "分子", "链", "分子/链", "分子·链", "分子链"}:
        return "molecule"
    raise ValueError(f"未知的「分析对象」取值 {value!r}：可用 "
                     f"'molecule'（分子 / 链）或 'system'（整体体系）")


@register("rg", "回转半径 Rg")
def analyze_rg(mdt, ag, selection: FrameSelection, *,
               mass_weighted: bool = True, unwrap: bool = True,
               nbins: int | None = None, verbose: bool = False,
               label: str = "链", object: str = "molecule",
               per_molecule: bool | None = None,
               max_molecules_drawn: int = 400) -> AnalysisResult:
    """回转半径分析。**「分析对象」二选一（互斥），默认分子 / 链。**

    ``object``（界面上的「分析对象」）
    -----------------------------
    * ``"molecule"``（默认，界面「分子 / 链」）—— **逐分子**算 Rg；
      多分子组分的主曲线是「逐帧各分子 Rg 平均」（每个分子各算再对分子平均）。
      这是 Materials Studio Forcite 的 "Radius of gyration" 口径
      （实测逐帧相关系数 1.00000、均值差 0.0000%），也是"分子有多大"的正确回答。
    * ``"system"``（界面「整体体系」）—— 把所选对象**当成一个整体**算 Rg。
      单分子/单链时它就是该链的尺寸（与 GROMACS ``gyrate`` 同口径）；
      多分子组分上它是"整团物质"的回转半径 —— 溶液中接近盒子尺度、随浓度变化，
      **不是**"分子有多大"。

    为什么必须互斥：多分子组分上这是两个**完全不同的量**（实测同一体系
    4.39 Å vs 21.18 Å，差 4.8 倍），放在一张图里会互相干扰；而 1.0.3 及更早
    一次运行会把两套都算一遍 —— 同一条轨迹扫两遍用途、同一批数据出两组数，
    用户点名要去掉（本版按"选哪个算哪个"实现：一次运行只产出一个量）。

    ``per_molecule``（兼容 1.0.3 的旧布尔开关，已废弃）
    -------------------------------------------------
    给了就以它为准：``True`` → ``object="molecule"``、``False`` → ``object="system"``。
    """
    mode = normalize_rg_object(object)
    if per_molecule is not None:                 # 老界面/老脚本的布尔开关
        mode = "molecule" if bool(per_molecule) else "system"
    want_mol = mode == "molecule"

    u = mdt.universe
    times = np.asarray(selection.times_ps, dtype=float)
    n_frame = times.size
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

    #: 分子划分（一次算好，逐帧复用）。**分子数必须如实报出**：早先这里在关掉
    #: 逐分子统计时直接写 n_mol = 1，于是一个含 648 个癸醇分子的 polymer 组分
    #: 会在结果里显示成「该组即 1 个分子」—— 与事实相反。
    ms = molecule_slices(ag)
    if ms is not None:
        mol_ids, slices, mol_labels = ms
        n_mol = len(slices)
    else:
        from .interface import _atom_molnums

        mol_ids = _atom_molnums(ag)
        n_mol = int(np.unique(mol_ids).size) if mol_ids is not None else 1
        slices, mol_labels = [], []
    multi = n_mol > 1
    #: 「分子 / 链」模式下**拿不到分子划分**时的兜底：该组本身就是**一个**连通对象
    #: （键图只有 1 个碎片、拓扑又没有 molnums，例如 Material Studio 导出的单个
    #: 大分子）。此时把它当成"一个分子"算整组 Rg —— 早先这里因为 `slices` 为空而
    #: 一帧都不算，结果整条曲线是 NaN（实测用户体系里 960 原子的那个分子
    #: 在「分子 / 链」档直接空白）。整组 Rg 就是它的 Rg，本来就是正确答案。
    whole_as_one = bool(want_mol and not slices)

    n_fast = 0
    #: 整体体系模式的主序列（整组 Rg）
    rg_system = np.full(n_frame, np.nan)
    #: 分子/链模式的逐分子序列
    rg_mol = np.full((n_frame, n_mol), np.nan)
    #: 快路径条件：**多分子**对象 + 有原子级分子编号。单分子对象保持精确路径
    #: （那是与 GROMACS gyrate / PLUMED 对照过的 0.000% 口径，不动它）。
    fast_ok = bool(unwrap and multi and mol_ids is not None
                   and mol_ids.size == ag.n_atoms)
    #: 分子划分在整条轨迹上不变 → **只算一次**（旧版每帧都 np.unique，白花时间）
    groups = prepare_molecule_groups(mol_ids) if fast_ok else None
    #: 快路径里被"分子尺度接近半盒"判据挑出来、改用键连通重建的分子（供说明用）
    fast_stats: dict = {}

    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        # ---------------------------------------------------------- 快路径
        # 多分子对象用向量化展开（`rg_fast`）：它与 MDAnalysis 的逐碎片展开在
        # **逐分子** Rg 上逐位一致（对 200×30 的 PEG 与纯键连通重建比，最大偏差
        # 1.1e-09 Å），整组 Rg 也与之同约定（差 ≤1.2e-07 Å），而速度快 1~2 个
        # 数量级 —— 精确路径 `positions_for(unwrap=True)` 是 Python 层逐碎片循环，
        # 122 个分子 × 6001 帧 = 73 万次调用（实测 18.1 ms/帧 ≈ 108 s），
        # 快路径 0.37 ms/帧（≈2.2 s）。
        #
        # 两种模式**共用这一次扫描**，但各自只落自己那个量：分子/链模式只取逐分子
        # 值、整体体系模式只取整组值 —— 不会为另一个口径多扫一遍轨迹。
        if fast_ok:
            pos_raw = np.asarray(ag.positions, dtype=float)
            rg_total, rg_each, safe, _d = rg_fast(
                pos_raw, masses if mass_weighted else None, u.dimensions,
                groups=groups, ag=ag, stats=fast_stats)
            if safe:
                n_fast += 1
                if want_mol:
                    rg_mol[k, :] = rg_each
                else:
                    rg_system[k] = rg_total
                continue
        # ---- 精确路径（单分子 / 快路径也修不了时）----------------------------
        pos = positions_for(ag, unwrap=unwrap)
        if want_mol:
            if slices:
                for j, sl in enumerate(slices):
                    sub_m = None if masses is None else masses[sl]
                    rg_mol[k, j] = compute_rg(pos[sl], sub_m)
            else:
                # 拿不到分子划分：该组就是一个连通对象，整组 Rg 即它的 Rg
                rg_mol[k, 0] = compute_rg(pos, masses)
        else:
            rg_system[k] = compute_rg(pos, masses)

    # ---------------------------------------------------------------- 主序列
    if want_mol:
        mol_mean = np.nanmean(rg_mol, axis=0)          # 每个分子的时间平均 Rg
        ok = np.isfinite(mol_mean)
        # ⭐「逐帧各分子 Rg 的平均」= **每个分子各自算 Rg，再对分子取平均**。
        # 这正是 Materials Studio Forcite 的 "Radius of gyration" 口径，也是
        # "分子有多大"的正确回答。单分子组分时它就等于该分子的 Rg（曲线用组分名）。
        main = np.nanmean(rg_mol, axis=1)
        main_label = "逐帧各分子 Rg 平均" if multi else label
    else:
        mol_mean, ok = None, None
        main = rg_system
        main_label = label

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="rg",
        title=f"回转半径 Rg —— {label}",
        meta={"selection": label, "n_atoms": int(ag.n_atoms),
              "mass_weighted": bool(mass_weighted), "unwrap": bool(unwrap),
              "n_frames": times.size, "n_molecules": int(n_mol),
              "object": mode, "per_molecule": bool(want_mol)},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="回转半径 Rg (Å)", title="Rg 随时间变化"),
        Panel(xlabel="回转半径 Rg (Å)", ylabel="概率密度 P(Rg)", title="Rg 分布"),
    ]
    if want_mol and multi:
        res.panels.append(
            Panel(xlabel="分子序号（按平均 Rg 升序）", ylabel="平均 Rg (Å)",
                  title="各分子的平均回转半径"))

    res.add_curve(main_label, tx, main, kind="line", panel=0)

    stat = describe_array(main)
    res.add_curve("平均 Rg", tx, np.full_like(tx, stat["mean"]), kind="line", panel=0)
    _bmeans, bsem = block_average(main, n_blocks=min(5, max(2, main.size // 2)))

    finite = main[np.isfinite(main)]
    if finite.size:
        edges = auto_bins(finite, nbins)
        hist, _ = np.histogram(finite, bins=edges, density=True)
        res.add_curve("Rg 分布", bin_edges_to_centers(edges), hist, kind="bar", panel=1)

    res.summary.update(describe_array(main, prefix="Rg "))
    res.summary["Rg 块平均标准误"] = float(bsem)
    res.summary["分子数"] = int(n_mol)
    #: 每个分子的原子数（中位）：一眼看出分子划分有没有被切碎
    #: （C12 = 38 原子；若显示 19 说明每条链被切成了两半 → Rg 会偏小 ~30%）。
    if want_mol:
        _sizes = (np.array([len(s) for s in slices]) if slices
                  else (np.bincount(groups.inv) if groups is not None else None))
        if _sizes is not None and _sizes.size:
            res.summary["每分子原子数（中位）"] = int(np.median(_sizes))
    res.summary["分析对象"] = RG_OBJECT_LABELS[mode]
    #: 交代这些数是哪条路径算的（多分子组分通常 = 全部帧；单分子对象恒为 0 帧）
    res.summary["向量化展开帧数"] = f"{n_fast}/{n_frame}"
    _repaired = fast_stats.get("repaired_molecules")
    if _repaired:
        # 少数分子尺度接近半盒 → 它们的内部坐标改由**键连通重建**给出（MDAnalysis
        # make_whole），其余分子仍走向量化路径。如实报出，免得用户以为整组都换了算法。
        res.summary["键连通重建的分子数"] = int(len(_repaired))
        res.add_notes(
            f"该组分里有 **{len(_repaired)} 个分子**的尺度接近盒子的一半"
            f"（最大原子–质心距离 ≥ 0.4×最短盒边），逐原子最小镜像对它们不再等价："
            f"这些分子的坐标改由**键连通重建**给出（与 MDAnalysis "
            f"`unwrap(compound='fragments')` 同算法），其余分子走向量化展开。"
            f"两类分子的口径一致，结果里不再有「整组退回慢路径」的隐含切换。")

    if want_mol and multi:
        res.summary["统计口径"] = "分子 / 链（逐分子算 Rg，再对分子取平均）"
        res.summary["各分子 Rg 平均（逐帧对分子平均）"] = (
            float(np.nanmean(main)) if np.isfinite(main).any() else float("nan"))
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
            f"分析对象 = **分子 / 链**：该组分含 **{n_mol} 个分子**，"
            f"每个分子各自算 Rg，曲线「{main_label}」= 每帧对分子取平均；"
            f"Rg 统计量（{res.summary['Rg mean']:.3f} Å）也是这条曲线的统计量。")
        res.add_notes(
            f"「单分子 Rg 平均」= {res.summary['单分子 Rg 平均 (Å)']:.3f} Å "
            f"（±{res.summary['单分子 Rg 标准差 (Å)']:.3f} Å，分子间）是每个分子"
            f"时间平均后再对分子平均；各分子数值见「各分子的平均回转半径」面板"
            f"（最多画 {n_draw} 个，按升序排列）。")
        res.add_notes(
            "本口径与 Materials Studio Forcite 的 \"Radius of gyration\" 一致"
            "（用户体系实测逐帧相关系数 1.00000）。若要看「整团物质有多大」，"
            "把「分析对象」切成「整体体系」——那个数是整组（含分子间间隔）的尺寸，"
            "**不能**当作单链尺寸引用。")
        if res.summary["单分子 Rg 标准差 (Å)"] > 0.3 * res.summary["单分子 Rg 平均 (Å)"]:
            res.add_notes(
                "分子间 Rg 的相对离散度超过 30%，说明体系里各分子的链尺寸差别很大"
                "（存在不同聚合度/构象亚群），只报一个平均值会掩盖这种多分散性。")
        # ---- **谁把逐分子平均拉上去了**（用户实测踩过）：集合里只要混进少数
        #      超大分子（石墨片、整块材料、交联网络…），逐分子平均就会被它们主导 ——
        #      例如"元素 C"集合同时选中 50 条 C10H22（Rg≈2.94 Å）与 2 个 800 原子
        #      石墨片（Rg≈20 Å）时，平均值是 3.60 Å 而不是 2.94 Å。Forcite 那边如果
        #      没选石墨，比出来的差就是**选择不同**，不是算错。这里直接把元凶点名。
        try:
            _msizes = np.asarray([len(s) for s in slices], dtype=int) if slices \
                else (np.bincount(groups.inv) if groups is not None else None)
            _med = float(np.nanmedian(mol_mean[ok])) if ok.any() else float("nan")
            res.summary["单分子 Rg 中位 (Å)"] = _med
            if ok.any() and _med > 0:
                _big = np.flatnonzero(ok & (mol_mean > 2.0 * _med))
                if _big.size:
                    res.summary["比单分子 Rg 中位大 2 倍以上的分子数"] = int(_big.size)
                    _names = []
                    for _k in _big[:6]:
                        _nm = ""
                        try:
                            _ia = int(np.flatnonzero(groups.inv == _k)[0])
                            _ri = int(np.asarray(u.atoms.resindices)[_ia])
                            _nm = str(np.asarray(u.residues.resnames)[_ri])
                        except Exception:                        # noqa: BLE001
                            _nm = ""
                        _na = int(_msizes[_k]) if _msizes is not None and _k < _msizes.size else 0
                        _names.append(f"{_nm or '分子'} #{int(_k) + 1}"
                                      f"（{_na:,} 原子，Rg {mol_mean[_k]:.2f} Å）")
                    res.add_notes(
                        f"⚠️ 这个集合里有 **{_big.size} 个分子明显比其它大**"
                        f"（Rg 超过单分子中位数 {_med:.2f} Å 的 2 倍）："
                        + "；".join(_names)
                        + "。逐分子平均是**等权平均**，这些大分子会把结果拉高"
                        "（用户实测：C 元素集合里混进 2 片 800 原子石墨时，"
                        "逐分子平均从 2.94 Å 变成 3.60 Å）。要和 Materials Studio "
                        "Forcite 的结果对比，请确认**两边选中的分子一致** —— "
                        "例如把石墨/刚性片层排除，只留聚合物链。")
        except Exception:                                    # noqa: BLE001
            pass          # 诊断信息失败不影响 Rg 结果本身
    elif want_mol:
        res.summary["统计口径"] = "分子 / 链（该组即 1 个分子，逐分子 Rg = 该分子的 Rg）"
        res.add_notes(
            "该组分只含 1 个分子：「分子 / 链」与「整体体系」两种分析对象给出的是"
            "同一个数（整组 Rg 就是该分子的回转半径），所以这里不分两套。")
        if whole_as_one:
            res.add_notes(
                "拓扑里没有可用的分子划分（既没有 molnums、键图也只有 1 个连通碎片，"
                "常见于 Materials Studio 导出的单个分子），因此按**一条链/一个分子**"
                "处理：这里的 Rg 就是该对象的回转半径。")
    elif multi:
        # 整体体系模式 + 多分子组分：如实说明这个数是"整团物质"的尺寸
        res.summary["统计口径"] = f"整体体系（整组 Rg；该组分含 {n_mol} 个分子）"
        res.add_notes(
            f"分析对象 = **整体体系**：该组分含 **{n_mol} 个分子**，这里的 Rg 是"
            f"**整团物质**（{ag.n_atoms:,} 个原子一起）的回转半径 —— 溶液中它接近"
            f"盒子尺度、随浓度变化，**不是单个分子有多大**。"
            f"要看链尺寸请把「分析对象」切成「分子 / 链」，程序会逐分子算 Rg。")
    else:
        res.summary["统计口径"] = "整体体系（该组即 1 个分子）"
        res.add_notes(
            "该组分只含 1 个分子：「整体体系」与「分子 / 链」两种分析对象给出的是"
            "同一个数（整组 Rg 就是该分子的回转半径）。")

    res.add_notes(f"质量加权: {mass_weighted}；PBC 展开: {unwrap}")

    res.add_notes(f"参与统计原子数: {ag.n_atoms}")
    if mass_note:
        res.add_notes(mass_note)
    return res


def _ends_of_element(ag, el: str) -> dict:
    """只用**某一种元素**的原子重做一次链端判定。

    键图端偶尔会给出「一端 C、一端 H」这种元素不同的组合（末基氢成了度为 1 的原子），
    这时按同元素子图（碳骨架）重找一次，得到化学意义上的 C…C 链端。
    """
    from collections import deque

    idx = np.asarray(ag.indices)
    n = idx.size
    out = {"ok": False, "ends": None, "reason": "", "n_components": 0,
           "n_degree1": 0, "branched": False, "element": el}
    bonds = _bonds_of(ag)
    if bonds is None or n < 3:
        out["reason"] = "无法按单元素键图判定链端"
        return out
    try:
        els = np.asarray(ag.elements, dtype=str)
    except Exception:  # noqa: BLE001
        return out
    pos_of = {int(a): i for i, a in enumerate(idx)}
    keep = {i for i in range(n) if els[i] == el}
    if len(keep) < 3:
        out["reason"] = f"{el} 原子不足 3 个"
        return out
    adj: list[list[int]] = [[] for _ in range(n)]
    for b in bonds:
        i, j = int(b[0].index), int(b[1].index)
        if i in pos_of and j in pos_of:
            a, c = pos_of[i], pos_of[j]
            if a in keep and c in keep:
                adj[a].append(c)
                adj[c].append(a)
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for s in sorted(keep):
        if seen[s] or not adj[s]:
            continue
        q = deque([s])
        seen[s] = True
        comp = [s]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    comp.append(v)
                    q.append(v)
        comps.append(comp)
    if len(comps) != 1:
        out["reason"] = f"{el} 键图有 {len(comps)} 个连通分量"
        return out
    comp = comps[0]
    deg1 = [v for v in comp if len(adj[v]) == 1]
    if not deg1:
        out["reason"] = f"{el} 键图没有端原子（环状/网络）"
        return out
    if len(deg1) == 2:
        i1, i2 = deg1[0], deg1[1]
    else:
        def bfs(src):
            d = {src: 0}
            qq = deque([src])
            while qq:
                u = qq.popleft()
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        qq.append(v)
            far = max(d, key=lambda x: d[x])
            return far, d[far]
        a0, _d0 = bfs(deg1[0])
        a1, _d1 = bfs(a0)
        i1, i2 = a0, a1
    out.update(ok=True, ends=(int(idx[i1]), int(idx[i2])),
               n_components=1, n_degree1=len(deg1), branched=len(deg1) > 2,
               method=f"同元素键图端点（{el}…{el}）")
    return out


def _finish_ends(ag, info: dict) -> dict:
    """键图端给出后：若两端**不是同一种元素**，改用同元素子图重新配对。"""
    if not info.get("ok") or info.get("ends") is None:
        return info
    try:
        els = np.asarray(ag.elements, dtype=str)
    except Exception:  # noqa: BLE001
        return info
    idx = np.asarray(ag.indices)
    pos_of = {int(a): i for i, a in enumerate(idx)}
    try:
        e1 = str(els[pos_of[int(info["ends"][0])]])
        e2 = str(els[pos_of[int(info["ends"][1])]])
    except Exception:  # noqa: BLE001
        return info
    info["elements"] = f"{e1}…{e2}"
    if not e1 or not e2 or e1 == e2:
        return info
    cand = [e for e in (e1, e2) if e != "H"] or [e1, e2]
    for el in cand:
        alt = _ends_of_element(ag, el)
        if alt.get("ok"):
            alt["elements"] = f"{el}…{el}"
            alt["method"] = str(alt.get("method")) +                 f"（原图两端元素不同：{e1}/{e2}）"
            return alt
    return info


@register("ree", "端到端距离 R_ee")

def _bond_graph_ends_prefer_heavy(ag) -> dict:
    """优先用**重原子键图**定链端，重原子图给不出端点时回退到全图。

    为什么改默认：全图（含 H）时"度为 1 的原子"包含**端基氢**，得到的并不是化学
    意义上的链端。实测 20mer 全原子体系：全图给 end20(HxaA)-beg1(HxaA) → 16.24 Å，
    而作者（PLUMED ``e2e.dat``：``ATOMS=5,423``，即首尾 residue 的 **O1 氧**）
    给 11.928 Å。改用重原子图后默认就落在端基杂原子上，与文献口径一致。
    """
    info = bond_graph_ends(ag, heavy_only=True)
    if info.get("ok"):
        info["method"] = (info.get("method") or "") + "（重原子键图）"
        return _finish_ends(ag, info)
    return _finish_ends(ag, bond_graph_ends(ag, heavy_only=False))


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
        旧口径（所选原子组的第一个与最后一个原子），仅用于对照；
        ``"residue"`` → 逐**残基**取 ``res.atoms[0]`` / ``res.atoms[-1]``，并用
        **原始坐标**（不解包、不做最小镜像）—— 与 ``test/mda_test.py`` 的
        ``test_ree`` 完全同口径，用于逐帧对账。
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
    raw_mode = False          # True = 用原始坐标（不解包、不最小镜像）
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
    elif str(ends) == "residue":
        # 逐残基的首/末原子（= mda_test.py 的 test_ree 口径）。注意它量的是
        # **残基**的首末原子，不是化学意义上的链端；且用原始坐标，所以跨盒子时
        # 数值会跳（这正是那份导出的特征）。
        raw_mode = True
        for ri in range(ag.residues.n_residues):
            atoms = ag.residues[ri].atoms
            if atoms.n_atoms < 2:
                continue
            nm = f"{atoms.resnames[0]}{atoms.resids[0]}"
            pairs.append((nm, int(atoms.indices[0]), int(atoms.indices[-1])))
        end_method = "逐残基首/末原子（原始坐标，mda_test.py 口径）"
        notes.append(f"该组分含 {ag.residues.n_residues} 个残基，逐残基用"
                     f"`res.atoms[0]` 与 `res.atoms[-1]` 计算，帧内对残基平均。")
        notes.append("⚠️ 这是**残基**首末原子、且**不做**跨周期解包/最小镜像，"
                     "所以数值会随周期性跳变；要看化学链端请用默认的「键图端原子」。")
    elif ms is not None and len(ms[1]) > 1:
        _m, slices, mol_labels = ms
        skipped = 0
        for sl, name in zip(slices, mol_labels):
            sub = ag[sl]
            info = _bond_graph_ends_prefer_heavy(sub)
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
        info = _bond_graph_ends_prefer_heavy(ag)
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
        if info.get("elements"):
            notes.append(f"链端元素：{info['elements']}")
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
            pp = positions_for(pair_ag, unwrap=bool(unwrap and not raw_mode))
            if pp.shape[0] < 2:
                continue
            ree[k, p] = compute_end_to_end(
                pp[0], pp[-1],
                box=None if (unwrap or raw_mode) else u.dimensions)

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
    try:
        if info.get("elements"):
            res.summary["链端元素"] = str(info["elements"])
    except NameError:
        pass

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
        # 「各链的平均端到端距离」逐链柱状图**已删**（用户要求，1.0.4）：
        # 单条链的 R_ee 涨落远大于差别，按序排开的柱图只是噪声，冷门且占版面；
        # 有意义的"单链 R_ee 平均/标准差"仍在摘要里。

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

    # ---- 4. 部分二面角随时间：**已删**（用户要求，1.0.4）
    # 原来画 max_series 根"按序号均匀挑出来的"二面角曲线 —— 挑哪几根没有物理含义，
    # 曲线又挤在一起，属于冷门图。二面角总数/帧数仍进摘要。
    if series.shape[1]:
        res.summary["二面角总数"] = int(series.shape[1])
        res.summary["二面角帧数"] = int(times.size)

    res.add_notes(*note)
    res.add_notes(f"trans 判据: |φ| > {gauche_edge:g}°")
    return res
