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
    "chain_dihedrals",
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



class _WholeCheck:
    """缓存"每根键在组内位置"的检查器（ag.bonds 每次访问要重建键表，约 3 ms）。"""

    def __init__(self, ag):
        self.ok = False
        self.i = self.j = None
        try:
            b = np.asarray(ag.bonds.indices, dtype=np.int64).reshape(-1, 2)
            if b.size == 0:
                return
            order = np.asarray(ag.indices, dtype=np.int64)
            i = np.searchsorted(order, b[:, 0])
            j = np.searchsorted(order, b[:, 1])
            if not (np.all(order[np.clip(i, 0, order.size - 1)] == b[:, 0])
                    and np.all(order[np.clip(j, 0, order.size - 1)] == b[:, 1])):
                return
            self.i, self.j = i, j
            self.ok = True
        except Exception:                                     # noqa: BLE001
            self.ok = False

    def check(self, ag, box, max_bond: float = 2.2) -> bool:
        if not self.ok or self.i is None:
            return False
        try:
            pos = np.asarray(ag.positions, dtype=float)
            d = pos[self.i] - pos[self.j]
            if box is not None:
                L = np.asarray(box[:3], dtype=float)
                d -= L * np.round(d / np.where(L > 0, L, 1.0))
            return bool(np.max(np.linalg.norm(d, axis=1)) < max_bond)
        except Exception:                                     # noqa: BLE001
            return False


def coords_already_whole(ag, box, max_bond: float = 2.2) -> bool:
    """当前帧的坐标是否**已经是完整分子**（不需要 make_whole 展开）。

    判据：组内每一根键的长度都 < ``max_bond``（默认 2.2 Å，> 任何共价键）。
    MS 的 .trj 坐标通常是展开的，``positions_for(unwrap=True)`` 里逐碎片 make_whole
    在 1700 原子的单分子上要 24 ms/帧（实测 graphite GRA），而"坐标本来就完整"
    时 make_whole 是恒等变换 —— 先花 ~0.1 ms 判定，能省掉这 24 ms。
    """
    try:
        b = ag.bonds
        if b is None or len(b) == 0:
            return False
        idx = np.asarray(b.indices, dtype=np.int64).reshape(-1, 2)
        if not idx.size:
            return False
        # 向量化映射：ag.indices 升序 → searchsorted 一次搞定，避免逐键 dict 查询
        order = np.asarray(ag.indices, dtype=np.int64)
        i = np.searchsorted(order, idx[:, 0])
        j = np.searchsorted(order, idx[:, 1])
        if not (np.all(order[np.clip(i, 0, order.size - 1)] == idx[:, 0])
                and np.all(order[np.clip(j, 0, order.size - 1)] == idx[:, 1])):
            return False
        pos = np.asarray(ag.positions, dtype=float)
        d = pos[i] - pos[j]
        if box is not None:
            L = np.asarray(box[:3], dtype=float)
            d -= L * np.round(d / np.where(L > 0, L, 1.0))
        return bool(np.max(np.linalg.norm(d, axis=1)) < max_bond)
    except Exception:                                         # noqa: BLE001
        return False

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


def dihedral_bin_grid(bin_width: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """二面角分布的分箱网格（**与 Forcite 的 Torsion distribution 逐点对齐**）。

    返回 ``(centers, edges)``：

    - ``centers``：格子**中心**，最外侧 = ±(180 + bin)，步长 = bin；
    - ``edges``：格子边界 = ``centers ± bin/2``。

    规则来自 Forcite 实测（20PE30，bin=0.5）：它的 Torsion 列是
    ``−180.5, −180, −179.5, …, 180, 180.5``（723 个），把这些值当作**中心**、
    桶边界放在 ``±0.25``（即 −180.75 … 180.75）时，我们的计数与它 723 格**逐格相同**
    （L1 = 0）；而把同样的列当"桶左边界"用（边界 −180.5 … 180.5）时，45% 的格子
    计数不同 —— 差的正是**半个 bin**。

    ``bin_width=1``  → 363 个中心 −181 … 181，边界 ±181.5
    ``bin_width=0.5``→ 723 个中心 −180.5 … 180.5，边界 ±180.75
    """
    try:
        bw = float(bin_width)
    except (TypeError, ValueError):
        bw = 1.0
    if not np.isfinite(bw) or bw <= 0:
        bw = 1.0
    bw = float(min(max(bw, 0.05), 180.0))         # 兜底约束：0.05° … 180°
    limit = 180.0 + bw                            # 最外侧的**格子中心**
    n_bins = max(int(round(2.0 * limit / bw)) + 1, 1)
    centers = -limit + bw * np.arange(n_bins)
    edges = np.append(centers - bw / 2.0, centers[-1] + bw / 2.0)
    return centers, edges


def _topology_empty_hint(ag, heavy_only: bool) -> str:
    """``topology`` 模式一个二面角都没取到时，说清到底是哪种原因。"""
    if _bonds_of(ag) is None:
        return "拓扑模式无可分析对象：当前拓扑不含键连接信息。" + NO_BOND_HINT
    try:
        n_topo = len(ag.dihedrals)
    except Exception:  # noqa: BLE001
        return ("拓扑模式无可分析对象：当前拓扑只提供了**键**，没有显式的二面角表"
                "（.gro、按坐标猜键的 .pdb 都是这种情况）。请改用 "
                "mode='chain'（逐分子沿键图骨架取二面角），"
                "或换成自带二面角的拓扑（.tpr / .psf / .prmtop）")
    if n_topo == 0:
        return "拓扑里没有任何二面角定义，拓扑模式无可分析对象；请改用 mode='chain'"
    return (f"拓扑里有 {n_topo} 个二面角，但没有一个完全落在所选原子上"
            + ("（heavy_only 打开时含氢二面角会被剔除，可关闭后重试）"
               if heavy_only else ""))


def chain_dihedrals(ag, *, heavy_only: bool = True, max_molecules: int = 200):
    """**逐分子**取"键图最长路径"上的连续四原子二面角。

    返回 ``(indices, notes)``：``indices`` 是形状 ``(n, 4)`` 的绝对原子索引，
    ``notes`` 是要写进结果的说明（含覆盖率、截断、无键等情形）。

    为什么要逐分子（1.0.5 修正）
    ---------------------------
    1.0.4 及以前是对**整个选择**做两次 BFS 取一条最长路径。BFS 跳不出连通分量，
    于是多分子体系里**只有第一个分子被分析**，其余分子被静默丢弃。实测 20 条
    DP=30 的 PE（C60H122 × 20，3640 原子）：

    - 旧做法：最长路径 60 个原子 → **57** 个二面角，全部落在分子 0 内，
      trans 比例 0.5263（n=57）；
    - 逐分子：20 × 57 = **1140** 个二面角，trans 比例 0.4851（n=1140）——
      不只是样本量差 20 倍，trans/gauche 比例本身也偏了 0.04～0.10
      （gauche− 从 0.271 掉到 0.175）。

    支化/环状分子里还有第二重失真：即使只有一个分子，"最长路径"也可能拐进
    侧链而漏掉主链（实测 40 碳主链 + 30 碳侧基挂第 20 号碳：路径走侧链 30 个
    碳 + 主链 21 个碳，主链 C21–C39 一个二面角都不算）。这种情况会写进 notes。

    重原子数 < 4 的分子（水、离子等）直接跳过：凑不出四元组。
    """
    notes: list[str] = []
    empty = np.zeros((0, 4), dtype=int)
    if ag is None or ag.n_atoms == 0:
        return empty, notes
    if _bonds_of(ag) is None:
        notes.append("当前拓扑不含键连接信息，无法提取几何骨架二面角。" + NO_BOND_HINT)
        return empty, notes

    try:
        frags = list(ag.fragments)
    except Exception:  # noqa: BLE001
        frags = [ag]

    usable: list[tuple[object, int]] = []
    for f in frags:
        n_heavy = int(f.n_atoms)
        if heavy_only:
            try:
                el = np.asarray(f.elements, dtype=str)
                n_heavy = int(np.sum((el != "H") & (el != "")))
            except Exception:  # noqa: BLE001
                n_heavy = int(f.n_atoms)
        if n_heavy < 4:
            continue
        usable.append((f, n_heavy))
    if not usable:
        notes.append("所选对象里没有含 ≥4 个重原子的分子（少于 4 个原子凑不出"
                     "四元组），无法用几何骨架模式提取二面角")
        return empty, notes

    cap = max(int(max_molecules), 1)
    truncated = len(usable) > cap
    total_mol = len(usable)
    if truncated:
        usable = usable[:cap]

    rows: list[np.ndarray] = []
    sizes: list[int] = []
    missed = missed_atoms = 0
    for f, n_heavy in usable:
        path = backbone_path(f, heavy_only=heavy_only)
        d = make_dihedral_indices(path)
        if d.shape[0] == 0:
            continue
        rows.append(d)
        sizes.append(int(path.size))
        if path.size < n_heavy:
            missed += 1
            missed_atoms += int(n_heavy - path.size)
    if not rows:
        notes.append("所选对象里的分子都没能在键图上串成 ≥4 个重原子的路径，"
                     "无法提取二面角")
        return empty, notes

    idx = np.vstack(rows)
    uniq = sorted(set(sizes))
    span = f"{uniq[0]}" if len(uniq) == 1 else f"{uniq[0]}–{uniq[-1]}"
    n_mol_used = len(rows)
    notes.append(
        f"**几何连续原子模式（逐分子）**：选择内 {total_mol} 个分子"
        f"（每个至少 4 个重原子），对**每个分子**单独在键图上取最长路径"
        f"（每条链 {span} 个重原子），共提取连续四原子二面角 {idx.shape[0]} 个"
        f"（{n_mol_used} 个分子 × 约 {idx.shape[0] // max(n_mol_used, 1)} 个）。"
        f"这些是几何上连成一串的原子，不保证等于化学主链/重复单元")
    if truncated:
        notes.append(f"选择内分子数超过上限 max_molecules={cap}，只分析了前 {cap} 个"
                     f"分子，其余 {total_mol - cap} 个未参与统计")
    if missed:
        notes.append(f"其中 {missed} 个分子的最长路径未覆盖该分子全部重原子"
                     f"（共 {missed_atoms} 个重原子不在路径上，源自支化/环状结构），"
                     f"这些分子的侧基扭转与环内扭转不会出现在结果里；"
                     f"要按拓扑定义的全部二面角请用 mode='topology'"
                     f"（需 .tpr / .psf / .prmtop）")
    return idx, notes


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

    #: "坐标是否已完整"的键表检查器（一次性缓存，循环里只做键长比较）
    whole_chk = _WholeCheck(ag)

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
        # 坐标本来就是完整分子时，positions_for(unwrap=True) 的 make_whole 是
        # 恒等变换，却要逐碎片写 Python 循环（实测 1700 原子单分子 24 ms/帧）。
        # 先用一次向量化的键长检查判定，能省就省（结果完全一致）。
        if unwrap and whole_chk.check(ag, u.dimensions):
            pos = np.asarray(ag.positions, dtype=float)
        else:
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
            raise ValueError("atom_indices 需要两个原子编号")
        n_ag = int(len(ag.indices))
        sel_abs = [int(x) for x in np.asarray(ag.indices)]
        pos = []
        for x in atom_indices[:2]:
            xi = int(x)
            if 1 <= xi <= n_ag:
                pos.append(sel_abs[xi - 1])            # 1 起：本组（主链）第 xi 个原子
            elif 0 <= xi < n_ag:
                pos.append(sel_abs[xi])                # 兼容 0 起写法
            else:
                pos.append(xi)                         # 兜底：当作绝对编号
        pairs = [("用户指定", pos[0], pos[1])]
        end_method = "用户指定"
        notes.append(f"按「本组（主链）原子顺序」取第 {int(atom_indices[0])} / "
                     f"第 {int(atom_indices[1])} 个原子作为链端："
                     f"{_atom_label(u, pairs[0][1])} – {_atom_label(u, pairs[0][2])}"
                     f"（本组共 {n_ag} 个原子）。")
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

    # ------------------------------------------------ 逐帧计算（**向量化**）
    # 旧写法是「每个分子 × 每一帧」都单独取一次坐标（MDAnalysis 子集构造很贵），
    # 分子多、帧数多时是平方级开销。现在只遍历**帧**一次：
    #   · 把全部端原子压成一个去重索引表 need，逐帧只取这一小批坐标；
    #   · 所有分子的端距用一次 numpy 减法 + norm 批量算出来（NPT 下逐帧按该帧的
    #     box 做最小镜像，与 compute_end_to_end 的 minimize_vectors 口径一致）。
    n_pair = len(pairs)
    ree = np.full((n_frame, n_pair), np.nan)
    if pairs:
        from ..cancel import check as _cancel_check

        i1s = np.fromiter((int(p[1]) for p in pairs), dtype=np.int64, count=n_pair)
        i2s = np.fromiter((int(p[2]) for p in pairs), dtype=np.int64, count=n_pair)
        need = np.unique(np.concatenate([i1s, i2s]))
        ia = np.searchsorted(need, i1s)
        ib = np.searchsorted(need, i2s)
        pos = np.empty((n_frame, need.size, 3), dtype=float)
        boxes: list = []
        for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
            _cancel_check()
            pos[k] = u.atoms[need].positions
            boxes.append(None if u.dimensions is None
                         else np.asarray(u.dimensions, dtype=float))
        # (n_frame, n_pair, 3)：一次减完所有分子
        diff = pos[:, ia, :] - pos[:, ib, :]
        use_min_image = bool(not (unwrap or raw_mode))
        if not use_min_image:
            ree = np.linalg.norm(diff, axis=2)
        else:
            from MDAnalysis.lib.distances import minimize_vectors

            for k in range(n_frame):
                box = boxes[k]
                if box is None:
                    ree[k] = np.linalg.norm(diff[k], axis=2)
                else:
                    ree[k] = np.linalg.norm(minimize_vectors(diff[k], box), axis=2)

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
                      max_molecules: int = 200,
                      max_series: int = 5,
                      gauche_edge: float = 120.0,
                      bin_width: float = 1.0,
                      verbose: bool = False,
                      label: str = "链") -> AnalysisResult:
    """二面角分析。

    参数
    ----
    mode
        - ``"auto"``：蛋白质用 phi/psi，其它体系用几何连续原子模式
        - ``"phi_psi"``：蛋白质主链 φ/ψ（对蛋白质这是**唯一**有明确化学含义的二面角）
        - ``"chain"``：**几何连续原子模式**——对选择里的**每个分子**分别沿键连接图
          取最长路径，再生成连续四原子二面角（见 :func:`chain_dihedrals`）。
          注意它给的是"几何上连成一串的原子"之间的二面角，**不保证**等于化学意义
          上的主链/重复单元二面角（支化或带长侧基的分子，最长路径可能拐进侧基，
          这种覆盖率不足的情形会写进结果说明）。名称由 1.0.0 的"链骨架"更正为
          现在的说法；1.0.5 起由"整组取一条最长路径"改为"逐分子取"，多分子体系
          不再只分析第一个分子。
        - ``"topology"``：使用拓扑文件中已定义的二面角（需要 .tpr / .psf / .prmtop
          这类自带二面角表的拓扑；只有键的 .gro 会得到 0 个并在说明里写明原因）
        - ``"custom"``：使用 ``custom_indices``
    custom_indices
        形状 ``(n, 4)`` 的绝对原子索引数组。
    heavy_only
        ``chain`` / ``topology`` 模式下是否忽略含氢二面角。
    max_molecules
        ``chain`` 模式最多分析多少个分子（默认 200）。选择里的分子数超过这个上限时
        只取前若干个并在结果说明里写明，避免上千个分子的体系把时间全花在建路径上。
    max_series
        在"二面角随时间变化"面板中最多画几条。
    gauche_edge
        trans / gauche 的分界角（**度**，默认 120）：``|φ| > gauche_edge`` 记为
        trans，其余按符号分 gauche+ / gauche−。不同力场/文献取 110～120 不等，
        因此做成参数而不是写死；结果里会写明本次用的阈值。
    bin_width
        分布直方图的**每格宽度（度）**，默认 **1.0**。网格范围固定为
        ``±(180 + bin_width)``，两端各多出一格的周期余量：

        - ``bin_width=1``   → −181, −180, …, 180, 181（362 格，中心 ±180.5）
        - ``bin_width=0.5`` → −180.5, −180, …, 180, 180.5（722 格，中心 ±180.25）
        - ``bin_width=5``   → −185, −180, …, 185（74 格）

        为什么要两端余量：二面角是周期量，±180° 是同一个角，峰的尾巴分别落在
        +180 与 −180 附近；留一格后两端都画得出来，不会贴在坐标轴边界上被截掉。
        取值会被夹到 0.05° … 180°。
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
        idx, cnote = chain_dihedrals(ag, heavy_only=heavy_only,
                                     max_molecules=max_molecules)
        note.extend(cnote)
    elif mode == "topology":
        idx = topology_dihedrals(ag, heavy_only=heavy_only)
        if idx.shape[0]:
            note.append(f"拓扑中属于所选原子的二面角 {idx.shape[0]} 个")
        else:
            note.append(_topology_empty_hint(ag, heavy_only))
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
              "n_frames": times.size, "gauche_edge": gauche_edge,
              "bin_width": float(bin_width)},
    )
    res.panels = [
        Panel(xlabel="二面角 φ (°)", ylabel="概率密度 P(φ)（1/°）",
              title="二面角分布（probability density）"),
        Panel(xlabel=tlabel, ylabel="构象比例", title="trans / gauche 构象比例随时间变化"),
    ]

    # ---- 2. 分布
    # 分箱网格与 **Materials Studio Forcite 的 "Bin width" 完全对齐**
    # （Forcite 帮助原文：Bin width: Specify the bin width, in degrees, for torsion
    #  distribution calculations… 每个 bin 的宽度就是这里的 bin_width）：
    #   网格端点 = ±(180 + bin_width)，步长 = bin_width，即
    #   bin=1   → −181, −180, −179, …, 179, 180, 181      （362 格）
    #   bin=0.5 → −180.5, −180, −179.5, …, 179.5, 180, 180.5（722 格）
    #   bin=2   → −182, −180, …, 180, 182                  （182 格）
    # 为什么两端各多出**一格**：二面角是周期量，±180° 是同一个角，峰的尾巴分别
    # 落在 +180 / −180 附近；Forcite 也是这么画的（两端各留一格），所以
    # **不做**首末格折叠，否则反而与 Forcite 对不上。
    # ---- 分箱对齐（**已用 Forcite 实测数据逐点核对**，见 dihedral_bin_grid）
    # Forcite 的 Torsion 列是每个桶的**中心**，桶边界在中心 ± bin/2。
    # 之前把 x 当"桶的左边界"用，整条分布就差**半个 bin**（0.25°），
    # 导致 45% 的桶计数与 Forcite 不同 —— 这就是"数据对不上"的原因。
    centers, edges = dihedral_bin_grid(bin_width)
    bw = float(edges[1] - edges[0])
    limit = float(centers[-1])
    n_bins = int(centers.size)

    # ---- 分布 = **概率密度**（小数，不是计数）
    # 为什么是概率密度：分布曲线要能跨体系/跨 bin 比较，y 必须是"每度多少概率"
    # 这种与样本总数、每格宽度都无关的量（× 每格宽度求和 = 1）。
    # 与 Forcite / 其它工具导出的列对比时：
    #   · bin=1 时"概率密度"与"每格概率"数值完全相同（乘 1°）；
    #   · bin≠1 时两者只差一个常数 1/每格宽度；
    #   · 概率密度 = 每格概率 ÷ 每格宽度 = 计数 ÷ (样本总数 × 每格宽度)，
    #     样本总数见摘要「分布样本总数（帧 × 二面角）」。
    for gname, arr in groups.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        # x 用**格子中心**（与 Forcite 的 Torsion 列一致）：
        #   bin=1   → −181, −180, …, 180, 181
        #   bin=0.5 → −180.5, −180, …, 180, 180.5
        counts, _ = np.histogram(finite, bins=edges)
        n_samp = int(counts.sum())
        dens = counts / max(float(n_samp) * bw, 1e-300)      # 概率密度 P(φ)
        res.add_curve(f"{gname} 分布", centers, dens, kind="bar", panel=0)
        res.summary[f"{gname} 分布样本总数（帧 × 二面角）"] = n_samp
        res.summary[f"{gname} 分布峰值（概率密度）"] = (
            float(dens.max()) if dens.size else float("nan"))
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
    res.add_notes(f"分布网格: 格子中心 {-limit:g}° … {limit:g}°，每格 {bw:g}°，"
                  f"共 {n_bins} 格（对齐 Forcite：Torsion 列是格子**中心**，"
                  f"桶边界 = 中心 ± {bw / 2:g}°，"
                  f"最外侧边界 {edges[0]:g}° … {edges[-1]:g}°）")
    res.add_notes("分布 y 是**概率密度**（probability density，单位 **1/°**，"
                  "与 Forcite 的 Torsion distribution 一致）："
                  "每格计数 ÷ (样本总数 × 每格宽度)，∫P(φ)dφ = 1。"
                  "它是小数、与样本总数无关，可跨体系/跨 bin 直接比较；"
                  "每格概率 = 概率密度 × 每格宽度（bin=1° 时两者数值相同）")
    return res
