# -*- coding: utf-8 -*-
"""拓扑补全：为不含键连接信息的拓扑文件（``.gro`` / 无 CONECT 的 ``.pdb`` / ``.xyz``）
按坐标猜出键连接。

为什么需要
----------
``.tpr`` / ``.psf`` / ``.prmtop`` 这些拓扑文件本身带键表，MDAnalysis 能直接得到
分子与链的划分；而 ``.gro`` 格式**根本无法存储键连接**，``.pdb`` 只有写了
``CONECT`` 记录才有键。没有键会连带影响：

- "分子/链"划分（退化成按 residue，见 :func:`mdta.selection._fragments`）；
- 链骨架二面角（``dihedral mode='chain'``）、链段取向（``orientation``）；
- PBC 展开（``unwrap`` 依赖键图把分子拼完整）。

所以对这些格式，本模块按**共价半径判据**从坐标猜键，把上面的能力补回来。

判据
----
``|r_i − r_j| < fudge × (R_i + R_j)``

这是 VMD / MDAnalysis / OpenBabel 都在用的经典判据。``fudge`` 默认 1.2，
既能覆盖正常的共价键（C–C 1.53 Å vs 1.82 Å 阈值），又不会把氢键
（O···H ≈ 1.8 Å ≥ 1.16 Å 阈值）或非键接触误判成键。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

#: 共价半径（Å，Pyykkö 单键共价半径），键名用大写元素符号。
#: 另加常见力场里的虚拟位点类型（半径 0 → 永不成键）。
COVALENT_RADII: dict[str, float] = {
    "H": 0.32, "HE": 0.46,
    "LI": 1.33, "BE": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63,
    "F": 0.64, "NE": 0.67,
    "NA": 1.55, "MG": 1.39, "AL": 1.26, "SI": 1.16, "P": 1.11, "S": 1.03,
    "CL": 0.99, "AR": 0.96,
    "K": 1.96, "CA": 1.71, "SC": 1.48, "TI": 1.36, "V": 1.34, "CR": 1.22,
    "MN": 1.19, "FE": 1.16, "CO": 1.11, "NI": 1.10, "CU": 1.12, "ZN": 1.18,
    "GA": 1.24, "GE": 1.21, "AS": 1.21, "SE": 1.16, "BR": 1.14, "KR": 1.17,
    "RB": 2.10, "SR": 1.85, "Y": 1.63, "ZR": 1.54, "NB": 1.47, "MO": 1.38,
    "TC": 1.28, "RU": 1.25, "RH": 1.25, "PD": 1.20, "AG": 1.28, "CD": 1.36,
    "IN": 1.42, "SN": 1.40, "SB": 1.40, "TE": 1.36, "I": 1.33, "XE": 1.31,
    "CS": 2.32, "BA": 1.96, "LA": 1.80, "CE": 1.63, "PR": 1.76, "ND": 1.74,
    "PM": 1.73, "SM": 1.72, "EU": 1.68, "GD": 1.69, "TB": 1.68, "DY": 1.67,
    "HO": 1.66, "ER": 1.65, "TM": 1.64, "YB": 1.70, "LU": 1.62,
    "HF": 1.52, "TA": 1.46, "W": 1.37, "RE": 1.31, "OS": 1.29, "IR": 1.22,
    "PT": 1.23, "AU": 1.24, "HG": 1.33,
    "TL": 1.44, "PB": 1.44, "BI": 1.51, "PO": 1.45, "AT": 1.47,
    "FR": 2.23, "RA": 2.01, "AC": 1.86, "TH": 1.75, "PA": 1.69, "U": 1.70,
    # 力场里的虚拟位点 / 未知类型：半径 0 → 永远不成键
    "DUMMY": 0.0, "M": 0.0, "MW": 0.0, "VS": 0.0, "LP": 0.0, "EP": 0.0,
    "X": 0.0, "": 0.0,
}

#: 默认 fudge factor（键长判据的放大系数）
DEFAULT_FUDGE = 1.2

#: 单原子离子的类型/元素名。这些原子**不参与猜键**：
#: 否则 Na⁺ 会被判成与 2.4 Å 处的水氧成键（R_Na+R_O=2.32 Å，×1.2=2.78 Å > 2.4 Å），
#: 把离子和水粘成一个大碎片。GROMACS 本身也把单原子离子视为独立分子。
ION_TYPES = {
    "NA", "K", "CL", "BR", "IOD", "I", "F", "LI", "RB", "CS", "MG", "CA",
    "ZN", "FE", "MN", "CU", "CO", "NI", "SR", "BA", "AL", "CD", "HG", "PB",
    "CR", "AG", "PT", "AU",
}


def _atom_types(u) -> np.ndarray | None:
    """取原子类型：优先 ``types``，其次 ``elements``；都没有则从原子名推。"""
    for attr in ("types", "elements"):
        try:
            v = np.asarray(getattr(u.atoms, attr), dtype=str)
            if v.size == u.atoms.n_atoms and any(str(x).strip() for x in v):
                return np.char.upper(np.char.strip(v))
        except Exception:  # noqa: BLE001 - MDAnalysis 缺该属性时抛 NoDataError
            continue
    # 从原子名猜元素（取开头的字母部分）
    try:
        names = np.asarray(u.atoms.names, dtype=str)
    except Exception:  # noqa: BLE001
        return None
    out = []
    for nm in names:
        s = "".join(c for c in str(nm) if c.isalpha())
        # 常见命名：C1/H1/OW/HW1/NA/CL → 元素；MW 之类无元素
        if not s:
            out.append("")
            continue
        two = s[:2].upper()
        out.append(two if two in COVALENT_RADII and len(s) > 1
                   and s[1].islower() else s[0].upper())
    return np.asarray(out, dtype=str)


def guess_bonds(universe, fudge_factor: float = DEFAULT_FUDGE,
                max_neighbors: int = 8, verbose: bool = False) -> dict:
    """按共价半径从坐标猜键，并把结果写回 ``universe``。

    返回统计信息 ``{"bonds": n, "n_types": ..., "unknown_types": [...],
    "fragments": n, "elapsed_sec": t}``。
    ``max_neighbors`` 给每个原子设一个成键上限，避免极端构型下键数爆炸。
    """
    import time

    from MDAnalysis.lib.distances import capped_distance

    t0 = time.time()
    u = universe
    n = u.atoms.n_atoms
    info: dict = {"bonds": 0, "unknown_types": [], "fragments": 0, "elapsed_sec": 0.0}

    types = _atom_types(u)
    if types is None or types.size != n:
        info["skipped"] = "无法确定原子类型"
        return info

    radii = np.array([COVALENT_RADII.get(str(t), 0.0) for t in types], dtype=float)
    unknown = sorted({str(t) for t, r in zip(types, radii) if r <= 0.0 and str(t)})
    info["unknown_types"] = unknown
    info["n_types"] = int(len(set(types.tolist())))
    good = radii > 0.0

    # 单原子离子不参与猜键（避免把 Na⁺/Cl⁻ 粘到邻近的水或残基上）
    try:
        res_counts = np.bincount(np.asarray(u.atoms.resindices, dtype=int),
                                 minlength=u.residues.n_residues)
        free_ion = np.isin(types, list(ION_TYPES)) & (res_counts[
            np.asarray(u.atoms.resindices, dtype=int)] == 1)
        info["n_free_ions"] = int(free_ion.sum())
        good = good & ~free_ion
    except Exception:  # noqa: BLE001
        pass

    if good.sum() < 2:
        info["skipped"] = "所有原子的共价半径都未知"
        return info

    pos = np.asarray(u.atoms.positions, dtype=float)
    box = u.dimensions
    cutoff = float(2.0 * fudge_factor * radii.max())
    if box is not None:
        # 截断距离不能超过盒条件允许的范围
        from .analysis.interface import safe_max_cutoff

        cutoff = min(cutoff, safe_max_cutoff(u))

    pairs, dist = capped_distance(pos, pos, max_cutoff=cutoff, box=box,
                                  return_distances=True)
    if not pairs.size:
        info["elapsed_sec"] = round(time.time() - t0, 2)
        return info

    i, j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    keep = (i < j) & good[i] & good[j] & (dist < fudge_factor * (radii[i] + radii[j]))
    i, j, dist = i[keep], j[keep], dist[keep]
    if not i.size:
        info["elapsed_sec"] = round(time.time() - t0, 2)
        return info

    # 每个原子只保留最近的 max_neighbors 个键
    if max_neighbors and max_neighbors > 0:
        order = np.lexsort((dist, i))
        i, j = i[order], j[order]
        cnt = np.zeros(n, dtype=np.int64)
        np.add.at(cnt, i, 1)
        rank = np.empty(i.size, dtype=np.int64)
        seen = np.zeros(n, dtype=np.int64)
        for k in range(i.size):
            a = i[k]
            rank[k] = seen[a]
            seen[a] += 1
        sel = rank < max_neighbors
        i, j = i[sel], j[sel]

    try:
        u.add_TopologyAttr("bonds", list(zip(i.tolist(), j.tolist())))
    except Exception as exc:  # noqa: BLE001
        info["skipped"] = f"写入键失败: {exc}"
        return info

    info["bonds"] = int(i.size)
    try:
        info["fragments"] = int(len(u.atoms.fragments))
    except Exception:  # noqa: BLE001
        pass
    info["elapsed_sec"] = round(time.time() - t0, 2)
    if verbose:
        print(f"  [拓扑补全] 按共价半径猜出 {info['bonds']:,} 个键，"
              f"{info['fragments']:,} 个分子/链，用时 {info['elapsed_sec']}s")
        if unknown:
            print(f"  [拓扑补全] 无共价半径的原子类型（不会成键）: {unknown}")
    return info


def needs_bond_guessing(universe) -> bool:
    """当前 Universe 是否缺少键信息。"""
    try:
        return len(universe.bonds) == 0
    except Exception:  # noqa: BLE001 - NoDataError 表示完全没有键属性
        return True
