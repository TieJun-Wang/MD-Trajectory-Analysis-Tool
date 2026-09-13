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


# --------------------------------------------------------------- 对象层属性
#: 标准原子质量 (u)。``.gro`` / 多数 ``.pdb`` 都不带质量，
#: 而质量加权 Rg、质量密度、质心 MSD 都要用 —— 这里按元素补齐。
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "HE": 4.003, "LI": 6.94, "BE": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180, "NA": 22.990,
    "MG": 24.305, "AL": 26.982, "SI": 28.085, "P": 30.974, "S": 32.06,
    "CL": 35.45, "AR": 39.948, "K": 39.098, "CA": 40.078, "SC": 44.956,
    "TI": 47.867, "V": 50.942, "CR": 51.996, "MN": 54.938, "FE": 55.845,
    "CO": 58.933, "NI": 58.693, "CU": 63.546, "ZN": 65.38, "GA": 69.723,
    "GE": 72.630, "AS": 74.922, "SE": 78.971, "BR": 79.904, "KR": 83.798,
    "RB": 85.468, "SR": 87.62, "Y": 88.906, "ZR": 91.224, "NB": 92.906,
    "MO": 95.95, "RU": 101.07, "RH": 102.91, "PD": 106.42, "AG": 107.87,
    "CD": 112.41, "IN": 114.82, "SN": 118.71, "SB": 121.76, "TE": 127.60,
    "I": 126.90, "XE": 131.29, "CS": 132.91, "BA": 137.33, "LA": 138.91,
    "CE": 140.12, "PT": 195.08, "AU": 196.97, "HG": 200.59, "TL": 204.38,
    "PB": 207.2, "BI": 208.98, "PO": 208.98, "U": 238.03,
}

#: 明确不是元素的"类型名"（虚拟位点 / 孤对电子 / 力场占位符）
NON_ELEMENT_TYPES = {"DUMMY", "M", "MW", "VS", "LP", "LP1", "LP2", "EP", "X", ""}


def ensure_object_attrs(universe, *, verbose: bool = False) -> dict:
    """补齐**对象模型必需、但 ``.gro`` 常常不提供**的三个属性。

    ==========  ======================================================
    属性        缺失时的来源
    ==========  ======================================================
    elements    由原子名/类型推断（复用猜键用的同一套推断）
    masses      由 elements 查 :data:`ATOMIC_MASSES`
    molnums     由**键图连通分量**（``atoms.fragments``）编号
    ==========  ======================================================

    为什么必须补：``molnums`` 是按分子排除配对的**唯一依据**
    （MDAnalysis 的 ``InterRDF.exclude_same`` 不支持 molecule，而本类体系
    segment 恒为 SYSTEM，不能用来区分分子）；``masses`` 是质量加权 Rg /
    质量密度 / 质心 MSD 的前提。

    返回诊断信息，便于把"这个数是谁的数"写进结果 metadata。
    """
    import numpy as np

    u = universe
    n = u.atoms.n_atoms
    info: dict = {"added": [], "n_molecules": 0, "n_missing_mass": 0,
                  "n_missing_element": 0, "sources": {}}

    # ---------------------------------------------------------- elements
    elements = None
    try:
        cur = np.asarray(u.atoms.elements, dtype=str)
        if cur.size == n and any(str(x).strip() for x in cur):
            elements = np.char.upper(np.char.strip(cur))
            info["sources"]["elements"] = "拓扑自带"
    except Exception:  # noqa: BLE001 - NoDataError
        pass
    if elements is None:
        types = _atom_types(u)
        if types is not None and types.size == n:
            elements = np.array(
                ["" if str(t).upper() in NON_ELEMENT_TYPES else str(t).upper()
                 for t in types], dtype=object)
            info["n_missing_element"] = int(
                sum(1 for e in elements if e and str(e).upper() not in ATOMIC_MASSES))
            u.add_TopologyAttr("elements", [str(e) for e in elements])
            info["added"].append("elements")
            info["sources"]["elements"] = "由原子名推断"

    # ------------------------------------------------------------ masses
    try:
        cur = np.asarray(u.atoms.masses, dtype=float)
        has_mass = cur.size == n and bool(np.all(np.isfinite(cur))) and bool(np.any(cur > 0))
    except Exception:  # noqa: BLE001
        has_mass = False
    if not has_mass and elements is not None:
        masses = np.array(
            [ATOMIC_MASSES.get(str(e).upper(), 0.0) if str(e) else 1.0
             for e in elements], dtype=float)
        # 未知元素给 1.0（而非 0）：零质量会把质心与质量加权拉偏；
        # 同时报出个数，调用方可据此决定是否退化成非质量加权。
        n_missing = int(np.sum(masses == 0.0))
        masses[masses == 0.0] = 1.0
        info["n_missing_mass"] = n_missing
        u.add_TopologyAttr("masses", masses)
        info["added"].append("masses")
        info["sources"]["masses"] = "由元素查表" + (
            f"（{n_missing} 个未知元素按 1.0 处理）" if n_missing else "")

    # ----------------------------------------------------------- molnums
    # ⚠️ MDAnalysis 的 ``molnums`` 是**残基级**属性（和 segids/resids 一样），
    #    长度必须是残基数，不是原子数；写成原子级会直接报错。
    has_mol = False
    try:
        cur = np.asarray(u.residues.molnums, dtype=int)
        has_mol = cur.size == u.residues.n_residues and int(cur.min()) >= 0
    except Exception:  # noqa: BLE001
        pass
    if has_mol:
        info["n_molecules"] = int(len(set(
            np.asarray(u.residues.molnums, dtype=int).tolist())))
        info["sources"]["molnums"] = "拓扑自带"
    elif needs_bond_guessing(u):
        info["sources"]["molnums"] = "无键表，无法划分分子（已跳过）"
    else:
        n_res = u.residues.n_residues
        res_mol = np.full(n_res, -1, dtype=int)
        frags = u.atoms.fragments
        for i, f in enumerate(frags):
            ridx = np.unique(np.asarray(f.resindices, dtype=int))
            res_mol[ridx] = i
        # 一个残基被拆到多个碎片（键断裂）时上面是"后写覆盖"；报出来
        n_unassigned = int(np.sum(res_mol < 0))
        # 压缩成连续编号 0..k-1
        uniq = np.unique(res_mol[res_mol >= 0])
        remap = np.full(int(res_mol.max()) + 1 if res_mol.size and res_mol.max() >= 0
                        else 0, -1, dtype=int)
        remap[uniq] = np.arange(uniq.size, dtype=int)
        res_mol = np.where(res_mol >= 0, remap[np.clip(res_mol, 0, None)], -1)
        u.add_TopologyAttr("molnums", res_mol)
        info["added"].append("molnums")
        info["n_molecules"] = int(uniq.size)
        info["n_residues_unassigned"] = n_unassigned
        info["sources"]["molnums"] = (
            f"由键图连通分量按**残基**编号（{len(frags):,} 个碎片 → "
            f"{int(uniq.size):,} 个分子编号）")
        if n_unassigned:
            info["sources"]["molnums"] += f"；{n_unassigned} 个残基未归属"

    if verbose:
        print(f"[对象层属性] 补齐 {info['added'] or '无'}；"
              f"分子 {info['n_molecules']:,}；来源 {info['sources']}")
    return info
