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
    """取原子类型：优先 ``types``，其次 ``elements``；都没有则从原子名推。

    ⚠️ ``types`` 里装的**未必是元素符号**：GROMACS/CHARMM 是 `CT`/`HC`，Materials
    Studio 是 `c3a`/`h1`。这类名字在 ``COVALENT_RADII`` 里查不到半径 → 猜键一根
    都猜不出来（实测 `.car` 读进来后 0 个键、`fragments` 直接不可用）。
    所以这里加一道"像不像元素"的校验：**至少 80% 的类型能在共价半径表里查到**才用它，
    否则退到 ``elements``，再退到原子名。
    """
    def _element_like(v: np.ndarray) -> bool:
        if v.size == 0:
            return False
        known = sum(1 for x in v if str(x).strip().upper() in COVALENT_RADII)
        return known >= 0.8 * v.size

    for attr in ("types", "elements"):
        try:
            v = np.char.upper(np.char.strip(
                np.asarray(getattr(u.atoms, attr), dtype=str)))
            if (v.size == u.atoms.n_atoms and any(str(x).strip() for x in v)
                    and _element_like(v)):
                return v
        except Exception:  # noqa: BLE001 - MDAnalysis 缺该属性时抛 NoDataError
            continue
    # 从原子名猜元素（取开头的字母部分）
    try:
        names = np.asarray(u.atoms.names, dtype=str)
    except Exception:  # noqa: BLE001
        # 连原子名都没有（LAMMPS data 就是这种）：按质量反推元素
        return _elements_from_masses(u)
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
    arr = np.asarray(out, dtype=str)
    # 原子名推不出元素（如 LAMMPS 的数字 type 被当成名字）→ 退到按质量反推
    filled = sum(1 for x in arr if str(x).strip() and str(x).upper() in COVALENT_RADII)
    if filled < 0.8 * arr.size:
        by_mass = _elements_from_masses(u)
        if by_mass is not None:
            return by_mass
    return arr


def residue_names(universe) -> np.ndarray:
    """残基名数组；**拓扑没带 `resnames` 时也不抛异常**。

    真实故障（LAMMPS 补充后缀后暴露）：`write_data` 导出的 `.lmp` 里
    `Atoms # full` 只有 mol-id，没有残基名 → 打开体系时 `auto_setup` 走到
    `u.residues.resnames` 直接抛
    ``NoDataError: This Universe does not contain resname information``，
    整个体系连打开都过不去（和之前 MS 的 PDB 缺 molnums 是同一类坑）。

    降级顺序：``resnames`` → 每个残基第一个原子的 ``types``/``names`` → ``"RES"``。
    只用于**分类与显示**（water / ion / polymer…），不影响任何数值分析。
    """
    import numpy as _np

    u = universe
    n_res = u.residues.n_residues
    for attr in ("resnames", "names"):
        try:
            v = _np.asarray(getattr(u.residues, attr), dtype=str)
            if v.size == n_res and any(str(x).strip() for x in v):
                return v
        except Exception:                                    # noqa: BLE001
            pass
    # 逐残基取"第一个原子"的类型/名字（LAMMPS data 没有残基名，但有原子类型）
    try:
        res_idx = _np.asarray(u.atoms.resindices, dtype=int)
        first = _np.zeros(n_res, dtype=int) - 1
        for i in range(res_idx.size - 1, -1, -1):
            first[res_idx[i]] = i                    # 倒序写 → 最后留下最小的原子号
        for attr in ("types", "names"):
            try:
                vals = _np.asarray(getattr(u.atoms, attr), dtype=str)
            except Exception:                                # noqa: BLE001
                continue
            out = _np.array(["RES"] * n_res, dtype=str)
            for r in range(n_res):
                i = first[r]
                if 0 <= i < vals.size and str(vals[i]).strip():
                    out[r] = str(vals[i])
            if all(str(x).isdigit() for x in out):
                # LAMMPS 的 atom type 是纯数字（1/2/3…），当残基名不好读；
                # 换成元素符号（C/H/O…）更直观：`polymer[C]` 比 `polymer[1]` 有用
                elem = _atom_types(u)
                if elem is not None and elem.size == u.atoms.n_atoms:
                    for r in range(n_res):
                        i = first[r]
                        if 0 <= i < elem.size and str(elem[i]).strip():
                            out[r] = str(elem[i])
            if any(str(x) != "RES" for x in out):
                return out
    except Exception:                                        # noqa: BLE001
        pass
    return _np.array(["RES"] * n_res, dtype=str)



def repair_fragmented_bonds(universe, fudge_factor: float | None = None,
                            max_add: int = 20000) -> dict:
    """修**拓扑键表不全**的体系：只在不同连通分量之间按距离补键。

    为什么要有（真实数据）：Materials Studio 导出的 .xsd 里，个别
    「链–链 / 单元–单元」之间的键会漏写，于是一条长链被切成 1 个原子、4 个原子
    这种碎片 —— 逐分子 Rg/R_ee、分子数、芳香环识别全错。
    实测 20PE30-10PS20：5547 原子本该是 20 条 DP30 聚乙烯 + 10 条聚苯乙烯，
    结果被切成 48 个碎片（还有 8 个孤立原子），苯环一个也认不出来。

    算法：先按现有键做并查集得到连通分量；用共价半径筛出"可能是键"的近距离原子对，
    只在**不同分量之间**合并（分量内部一根都不新增，避免凭空造出环），
    按距离从近到远合并，直到没有可合并的对。
    """
    from MDAnalysis.lib.distances import capped_distance

    u = universe
    n = int(u.atoms.n_atoms)
    info = {"added": 0, "components_before": 0, "components_after": 0,
            "skipped": ""}
    if n < 2:
        return info
    try:
        bonds = np.asarray([(int(b[0].index), int(b[1].index)) for b in u.bonds],
                           dtype=np.int64)
    except Exception:                                         # noqa: BLE001
        info["skipped"] = "没有键表"
        return info
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for a, b in bonds:
        union(int(a), int(b))
    comps0 = len({find(i) for i in range(n)})
    info["components_before"] = int(comps0)

    types = _atom_types(u)
    if types is None or len(types) != n:
        info["skipped"] = "无法确定原子类型"
        return info
    radii = np.array([COVALENT_RADII.get(str(x), 0.0) for x in types], dtype=float)
    pos = np.asarray(u.atoms.positions, dtype=float)
    fudge = DEFAULT_FUDGE if fudge_factor is None else float(fudge_factor)
    max_r = float(radii[radii > 0].max()) if np.any(radii > 0) else 1.8
    cutoff = max(0.9, 2.0 * max_r * fudge)
    # ⚠️ 没有周期盒时 MDAnalysis 的 capped_distance 会退化成**暴力 N² 两两比较**
    # （实测 47681 原子的 .tpr 在这里跑了 98 s 还没完，内存也暴涨 —— 用户报告的
    # "单独读取 adk_oplsaa.tpr 卡死 + OOM" 就是这一处）。
    # 补键只对小体系的"碎片链"有意义，所以：
    #   · 原子数 > 30,000 → 一律跳过；
    #   · 没有周期盒 → 只对 ≤8,000 原子做（8k 的暴力搜索 ~数千万对，仍在秒级），
    #     更大的无盒体系直接跳过（宁可少补键，也不能卡死/OOM）。
    box = u.dimensions
    if int(n) > 30_000:
        info["skipped"] = (f"原子数 {n:,} 超过补键上限 30,000，"
                           f"不做按距离补键（避免 O(N²) 卡死）")
        info["components_after"] = int(comps0)
        return info
    box_ok = box is not None and bool(np.all(np.asarray(box[:3], dtype=float) > 0))
    if not box_ok and int(n) > 8_000:
        info["skipped"] = (f"没有周期盒且原子数 {n:,} > 8,000："
                           f"距离搜索会退化成 O(N²)，跳过按距离补键")
        info["components_after"] = int(comps0)
        return info
    pairs = capped_distance(pos, pos, max_cutoff=cutoff,
                            box=box if box_ok else None, return_distances=True)
    pairs_idx = np.asarray(pairs[0]) if isinstance(pairs, tuple) else np.asarray(pairs)
    dists = np.asarray(pairs[1]) if isinstance(pairs, tuple) else None
    if pairs_idx.size == 0:
        info["components_after"] = int(comps0)
        return info
    a = pairs_idx[:, 0].astype(np.int64)
    b = pairs_idx[:, 1].astype(np.int64)
    keep = a != b
    a, b = a[keep], b[keep]
    need = (radii[a] + radii[b]) * fudge
    ok = (radii[a] > 0) & (radii[b] > 0)
    ok &= dists[keep] <= need if dists is not None else True
    a, b = a[ok], b[ok]
    if dists is not None:
        order = np.argsort(dists[keep][ok])
        a, b = a[order], b[order]
    added = []
    for i in range(a.size):
        if len(added) >= max_add:
            break
        if union(int(a[i]), int(b[i])):
            added.append((int(a[i]), int(b[i])))
    if added:
        try:
            u.add_bonds(added)
        except Exception:                                     # noqa: BLE001
            info["skipped"] = "写回键表失败"
            return info
    info["added"] = len(added)
    info["components_after"] = int(len({find(i) for i in range(n)}))
    return info

def guess_bonds(universe, fudge_factor: float = DEFAULT_FUDGE,
                max_neighbors: int = 8, verbose: bool = False,
                replace: bool = False) -> dict:
    """按共价半径从坐标猜键，并把结果写回 ``universe``。

    返回统计信息 ``{"bonds": n, "n_types": ..., "unknown_types": [...],
    "fragments": n, "elapsed_sec": t}``。
    ``max_neighbors`` 给每个原子设一个成键上限，避免极端构型下键数爆炸。

    ``replace=True``：先**清掉拓扑里已有的键**再猜。用于 ``CONECT`` 不完整的
    PDB（Material Studio 导出常见）：实测同一套原子，MS-PDB 只带 4393 个键，
    按坐标猜键得到 5954 个（与作者 PDB 的键数一致）。键表不全时"逐分子 Rg/R_ee、
    分子质心 MSD、RDF 的同分子剔除"都会**静默**变差，所以给一个显式开关。
    """
    import time

    from MDAnalysis.lib.distances import capped_distance

    if replace:
        try:
            universe.delete_bonds(universe.bonds)
        except Exception:                                     # noqa: BLE001
            try:
                universe.delete_TopologyAttr("bonds")
            except Exception:                                 # noqa: BLE001
                pass

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
        # 数碎片用并查集（不建 MDAnalysis 的 FragDict）：几十万原子时
        # `u.atoms.fragments` 要吃掉 GB 级内存
        comp = _bond_component_ids(u)
        info["fragments"] = int(comp.max()) + 1 if comp.size else 0
    except Exception:  # noqa: BLE001
        pass

    # ---------------------------------------------------- 跨碎片补键（修"只差一根键"）
    # 判据 |r_i−r_j| < fudge·(R_i+R_j)（fudge=1.2）在**边界情形**会漏键：
    # 实测用户体系里有一根 C–H 键长 1.298 Å，而阈值是 1.2×(0.76+0.31)=1.284 Å
    # —— 只差 0.014 Å，于是那个 C12 分子被切成「37 原子 + 1 个游离 H」两块，
    # 逐分子 Rg 平均里混进两个假分子，均值被拉低 0.8%（与 Forcite 的对照就卡在这）。
    # 修法：用**稍宽**的阈值再找一次配对，但**只保留"两端本来就在不同碎片里"的**，
    # 这样不会给已经成键的分子乱加键，同时把"只差一根键"的碎片接回去。
    try:
        fid = _bond_component_ids(u)              # 原子 → 碎片编号（并查集，低内存）
        n_frag_before = int(fid.max()) + 1 if fid.size else 0
        if n_frag_before > 1:
            rep_factor = max(float(fudge_factor) * 1.2, 1.35)
            rep_cut = float(2.0 * rep_factor * radii.max())
            if box is not None:
                from .analysis.interface import safe_max_cutoff

                rep_cut = min(rep_cut, safe_max_cutoff(u))
            p2, d2 = capped_distance(pos, pos, max_cutoff=rep_cut, box=box,
                                     return_distances=True)
            if p2.size:
                a2, b2 = p2[:, 0].astype(np.int64), p2[:, 1].astype(np.int64)
                keep2 = ((a2 < b2) & good[a2] & good[b2]
                         & (d2 < rep_factor * (radii[a2] + radii[b2]))
                         & (fid[a2] >= 0) & (fid[b2] >= 0)
                         & (fid[a2] != fid[b2]))
                if keep2.any() and (fid[a2[keep2]] != fid[b2[keep2]]).any():
                    u.add_bonds(list(zip(a2[keep2].tolist(), b2[keep2].tolist())))
                    info["repair_bonds"] = int(keep2.sum())
                    info["fragments_before_repair"] = n_frag_before
                    comp2 = _bond_component_ids(u)
                    info["fragments"] = (int(comp2.max()) + 1
                                         if comp2.size else 0)
                    info["repair_note"] = (
                        f"跨碎片补键 {info['repair_bonds']} 根："
                        f"碎片 {n_frag_before:,} → {info['fragments']:,}"
                        f"（判据放宽到 {rep_factor:.2f}×(R_i+R_j)，"
                        f"只补「两端在不同碎片」的配对）")
    except Exception as exc:  # noqa: BLE001 - 补键失败不应让整项分析崩
        info["repair_skipped"] = f"{type(exc).__name__}: {exc}"

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


def _elements_from_masses(universe, tol: float = 0.05) -> np.ndarray | None:
    """按**质量**反推元素符号（最近的标准原子质量，相对误差 ≤ ``tol`` 才算）。

    为什么需要：LAMMPS `write_data` 的 data 文件里**没有元素列**，只有
    `Masses` 段 + 数字 atom type（`1 2 3 …`）；`.pdb`/`.gro` 的原子名有时也是
    `C1`/`H1` 之类推不出元素的写法。没有元素 → 猜不出键 → 分子被切成一个个孤立
    原子（Rg/MSD 全错）。
    实测：`Masses 12.011 / 1.008` → ``C`` / ``H``，与真实体系一致。
    """
    try:
        m = np.asarray(universe.atoms.masses, dtype=float)
    except Exception:                                        # noqa: BLE001
        return None
    if m.size != universe.atoms.n_atoms or not np.any(m > 0):
        return None
    table = sorted(ATOMIC_MASSES.items(), key=lambda kv: kv[1])
    cache: dict[float, str] = {}
    out = np.full(m.size, "", dtype=object)
    for i, x in enumerate(m):
        if not np.isfinite(x) or x <= 0:
            continue
        key = round(float(x), 3)
        if key not in cache:
            best, best_d = "", None
            for el, ref in table:
                d = abs(ref - x) / ref
                if best_d is None or d < best_d:
                    best, best_d = el, d
            cache[key] = best if best_d is not None and best_d <= tol else ""
        out[i] = cache[key]
    if not any(out):
        return None
    return out


def _drop_molnums(universe) -> str:
    """把 ``molnums`` 从拓扑里去掉（失败则退化为"写上哨兵值 -1"）。

    ⚠️ MDAnalysis 2.10 里方法名是 **`del_TopologyAttr`**，没有 `delete_TopologyAttr`
    —— 早先这里写了后者并被 `except Exception: pass` 吞掉，于是"删掉退化 molnums"
    这条逻辑**根本没生效**（只是看起来像生效了，实际是下游 `_atom_molnums` 自己按
    "分子数远小于碎片数"又回退了一次）。现在把两种名字都试，再不行就写 -1 哨兵，
    让下游的"分子数 ≤1 就用键图碎片"判据接管。
    """
    for name in ("del_TopologyAttr", "delete_TopologyAttr"):
        fn = getattr(universe, name, None)
        if callable(fn):
            try:
                fn("molnums")
                try:
                    universe.residues.molnums
                except Exception:                              # noqa: BLE001
                    return name
            except Exception:                                  # noqa: BLE001
                continue
    try:                       # 退路：写哨兵（全 -1），下游按"只有 1 个分子"回退
        universe.add_TopologyAttr("molnums", np.full(universe.residues.n_residues, -1))
        return "哨兵 -1"
    except Exception:                                          # noqa: BLE001
        return "未能移除（下游按碎片回退）"


def bond_components(n_atoms: int, edges) -> np.ndarray:
    """按键表做**并查集**，返回每个原子所属的分量编号（0..C-1）。

    为什么不用 ``u.atoms.fragments``：MDAnalysis 为了同时给出
    ``atom->fragment`` 与 ``fragment->atoms`` 两套映射，第一次调用会建整张
    ``_fragdict`` —— AdK（47 681 原子 / 25 533 键）实测要 **~2 s**，
    而我们只要"分成几块、每块多大"。纯 Python 并查集在同一体系上 **0.29 s**。
    对照：两者结果一致（AdK 都是 22 173 个分量）。
    """
    import numpy as np

    n = int(n_atoms)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    try:
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    except Exception:                                        # noqa: BLE001
        e = np.zeros((0, 2), dtype=np.int64)
    if not e.size:
        return np.arange(n, dtype=np.int64)

    parent = list(range(n))
    for a, b in zip(e[:, 0].tolist(), e[:, 1].tolist()):
        if not (0 <= a < n and 0 <= b < n):
            continue
        ra = a
        while parent[ra] != ra:
            ra = parent[ra]
        rb = b
        while parent[rb] != rb:
            rb = parent[rb]
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    _uniq, comp = np.unique(np.asarray(parent, dtype=np.int64),
                            return_inverse=True)
    return comp.astype(np.int64)



def _bond_component_count(ag) -> int:
    """键图连通分量个数（**不建** MDAnalysis 的 FragDict，大体系上也很快）。

    ``atoms.fragments`` 会为每个原子造 Fragment 对象：AdK 47 681 原子实测约 2 s、
    几十万原子时内存可达 GB 级。这里只需要"分成几块"这一个数，用并查集即可。
    """
    comp = _bond_component_ids(ag)
    return int(np.asarray(comp).max()) + 1 if np.asarray(comp).size else 0


def _bond_component_ids(ag) -> np.ndarray:
    """原子级键图分量编号（0..C-1）；没有键表时每个原子自成一团。

    与 :func:`bond_components` 的区别：这里接受 **AtomGroup**，自己从
    ``universe.bonds`` 取边（``bond_components`` 只吃裸数组）。
    """
    try:
        u = ag.universe
        n = int(u.atoms.n_atoms)
        bonds = getattr(u, "bonds", None)
        edges = (np.zeros((0, 2), dtype=np.int64) if bonds is None
                 else np.asarray(bonds.indices, dtype=np.int64).reshape(-1, 2))
    except Exception:                                            # noqa: BLE001
        return np.zeros(0, dtype=np.int64)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(bond_components(n, edges), dtype=np.int64)


def molecule_ids(ag) -> np.ndarray | None:
    """**原子级**分子编号：能用 ``molnums`` 就用，否则退回**键图连通分量**。

    为什么要有这个函数（真实故障）：退化拓扑（Material Studio 的 PDB 把 5596 个原子
    塞进 2 个残基）下，`ensure_object_attrs` 会**删除** molnums，让下游改用键连通
    分量。但只要有**任何**一处代码直接写 `ag.molnums`，就会抛
    ``NoDataError: This Universe does not contain molnum information`` ——
    整个体系直接跑不起来（实测 `auto_setup` 就踩了）。

    所以：库内部一律用这个函数拿分子编号，**不要**直接访问 ``ag.molnums``。
    返回 ``None`` 只在"既没有 molnums、又没有键（连碎片都分不出）"时出现。
    """
    m = None
    try:
        cand = np.asarray(ag.molnums, dtype=np.int64)
        if cand.size == getattr(ag, "n_atoms", cand.size):
            m = cand
    except Exception:                                          # noqa: BLE001
        m = None
    if m is not None and np.unique(m).size > 1:
        return m
    # 退化（全体一个编号）或缺失 → 用键图碎片
    try:
        frags = ag.fragments
        if len(frags) > 1:
            ids = np.empty(ag.n_atoms, dtype=np.int64)
            base = np.asarray(ag.indices, dtype=np.int64)
            for k, f in enumerate(frags):
                ids[np.searchsorted(base, np.asarray(f.indices, dtype=np.int64))] = k
            if m is None or np.unique(ids).size > np.unique(m).size:
                return ids
    except Exception:                                          # noqa: BLE001
        pass
    return m


def topology_degeneracy(ag) -> dict:
    """判断拓扑**自带的键表**是否明显残缺（分子被拆成碎片）。

    为什么需要：Material Studio 导出的 PDB 里 ``CONECT`` 只写了十二烷的一部分键，
    于是 ``polymer[P]``（4636 原子 = 122 × C12H26）被切成 **243 个碎片**
    —— 122 个近乎完整的链（37–38 原子）+ **121 个孤立 H**。
    逐分子 Rg 的平均值于是被这些"0 Å 的单原子"拉低一半
    （实测 2.19456 Å，而正确值 4.3929 Å，与 Materials Studio Forcite 一致）。

    判据（任一成立即认为明显残缺）：

    * 碎片**中位原子数 ≤ 2**（正常分子至少 3 个原子）；
    * **孤立原子占比 ≥ 30%**（正常体系里孤立原子只可能是离子/溶剂单原子）。

    返回 ``{"degenerate", "n_frag", "median_size", "isolated_frac", "note"}``。

    ⚠️ **不调 ``ag.fragments``**：MDAnalysis 为了建 ``_fragdict`` 要给每个原子
    造 Fragment 对象，AdK（47 681 原子）实测 ~2 s，几十万原子的体系会吃掉
    GB 级内存（用户实测就卡在「读取」阶段）。这里改成同样结果的
    :func:`bond_components`（纯 Python 并查集，只分配一个 int 数组）。
    """
    out = {"degenerate": False, "n_frag": 0, "median_size": 0,
           "isolated_frac": 0.0, "note": ""}
    try:
        u = ag.universe
        n_atoms = int(u.atoms.n_atoms)
        bonds = getattr(u, "bonds", None)
        edges = (np.zeros((0, 2), dtype=np.int64) if bonds is None
                 else np.asarray(bonds.indices, dtype=np.int64).reshape(-1, 2))
    except Exception:                                            # noqa: BLE001
        return out
    if n_atoms == 0:
        return out
    comp = bond_components(n_atoms, edges)
    sizes = np.bincount(np.asarray(comp, dtype=np.int64))
    n_frag = int(sizes.size)
    if n_frag <= 1:
        return out
    med = int(np.median(sizes))
    iso = float(np.mean(sizes == 1))
    out.update(n_frag=n_frag, median_size=med, isolated_frac=iso)
    if med <= 2 or iso >= 0.3:
        out["degenerate"] = True
        out["note"] = (f"键图碎片 {n_frag:,} 个（中位 {med} 原子，孤立原子 "
                       f"{iso:.0%}）")
    return out


def molecule_count(ag) -> int:
    """分子数（容错版）：没有 molnums 就按键图碎片数算；都没有则返回 1。"""
    ids = molecule_ids(ag)
    if ids is None:
        return 1
    return int(np.unique(ids).size)


def bond_scale_report(ag, n_sample: int = 800) -> dict:
    """**坐标尺度体检**：用共价键长当"尺子"判断轨迹坐标是不是 Å。

    为什么需要（真实故障）：用户把一条 6001 帧的轨迹读进来，Rg 恰好是 Materials
    Studio Forcite 的 **一半**（2.19456 vs 4.3929 Å），而逐帧比值稳定在 2.0000 ——
    这不是算法差异，而是**坐标整体被缩放过**（键长也会同比例变短）：Rg 这类长度量
    全部按同一比例偏小，而且不会有任何报错。

    做法：抽 ``n_sample`` 根键，按两端原子的共价半径算出"应有的键长"
    （``R_i + R_j``），与坐标里实际量到的键长比较。正常体系比值应在 0.8–1.3
    （C–C 单键 1.54 Å，共价半径和 1.50 Å）。

    返回 ``{"n_bonds", "median", "expected", "ratio", "ok", "note"}``；
    ``n_bonds == 0``（没有键表）时 ``ratio``/``ok`` 为 ``None``。
    """
    out = {"n_bonds": 0, "median": float("nan"), "expected": float("nan"),
           "ratio": None, "ok": None, "note": ""}
    try:
        bonds = getattr(ag, "bonds", None)
        n = 0 if bonds is None else len(bonds)
    except Exception:                                            # noqa: BLE001
        n = 0
    if not n:
        out["note"] = "拓扑没有键表，无法用键长标定坐标尺度"
        return out
    out["n_bonds"] = int(n)
    try:
        idx = np.asarray(bonds.indices, dtype=np.int64)
    except Exception:                                            # noqa: BLE001
        out["note"] = "键索引读不出来"
        return out
    if idx.ndim != 2 or idx.shape[0] == 0:
        return out
    step = max(1, idx.shape[0] // max(int(n_sample), 1))
    pair = idx[::step][: int(n_sample)]
    pos = np.asarray(ag.positions, dtype=float)
    base = np.asarray(ag.indices, dtype=np.int64)
    ia = np.searchsorted(base, pair[:, 0])
    ib = np.searchsorted(base, pair[:, 1])
    d = np.linalg.norm(pos[ia] - pos[ib], axis=1)
    types = _atom_types(ag)
    if types is None:
        out["note"] = "读不到元素/原子名，无法估算应有键长"
        return out
    ta = types[ia]
    tb = types[ib]
    exp = np.array([COVALENT_RADII.get(str(a).upper(), 0.0)
                    + COVALENT_RADII.get(str(b).upper(), 0.0) for a, b in zip(ta, tb)])
    good = np.isfinite(d) & (exp > 0.2) & (d > 1e-6)
    if good.sum() < 3:            # 小分子（例如 6 原子链只有 5 根键）也要能标定
        out["note"] = "可用来标定的键太少"
        return out
    med, ex = float(np.median(d[good])), float(np.median(exp[good]))
    ratio = med / ex if ex > 0 else float("nan")
    out["median"], out["expected"], out["ratio"] = med, ex, ratio
    out["ok"] = bool(0.8 <= ratio <= 1.3)
    if out["ok"]:
        out["note"] = (f"键长中位数 {med:.3f} Å（按共价半径应为 {ex:.3f} Å，"
                       f"比值 {ratio:.2f}）—— 坐标尺度正常")
    else:
        hint = ("坐标单位可能不是 Å（例如轨迹存的是 nm）"
                if ratio < 0.3 else
                "坐标可能被整体缩放（或盒/单位换算错了一档）")
        out["note"] = (f"键长中位数 {med:.3f} Å，而按共价半径应为 {ex:.3f} Å"
                       f"（比值 {ratio:.2f}，正常 0.8–1.3）—— {hint}；"
                       f"Rg / R_ee / 密度厚度等**长度量会按同一比例（约 "
                       f"{1.0 / ratio:.2f}×）偏小或偏大**，请看输入文件的坐标单位")
    return out


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
        n_mol = int(len(set(np.asarray(u.residues.molnums, dtype=int).tolist())))
        mean_per_mol = n / max(n_mol, 1)
        # 「退化」判据需要**同时**满足"碎片数 ≥ 3×分子数"和"平均每分子 ≥ 8 原子"。
        # 后一条只看数字就能判，所以**先算它**：平均每分子只有几个原子时
        # （正常力场拓扑都这样，例如 AdK 47681 原子 / 11089 分子 = 4.3），
        # 根本不可能退化，就不必去算 `u.atoms.fragments` ——
        # 那一步在 tpr 上要 1 s 左右（实测），纯属白算。
        n_frag = n_mol
        if mean_per_mol >= 8.0:
            # 数碎片用并查集（不建 MDAnalysis 的 FragDict）：大体系上
            # `u.atoms.fragments` 又慢又占内存，而这里只要一个"几块"的数。
            n_frag = _bond_component_count(u)
        # 自带 molnums 也可能是**退化**的（Material Studio 的 PDB 会把整个体系塞进
        # 1–2 个残基，于是"分子数=2"而键图有上百个碎片）。这种情况不覆盖、而是
        # **不提供 molnums**，让下游自动回退到"键连通分量"（见
        # interface._atom_molnums）—— 那才是真实的分子划分。
        if n_frag >= 3 * max(n_mol, 1) and mean_per_mol >= 8.0:
            how = _drop_molnums(u)
            info["molnums_degenerate"] = {
                "自带分子数": n_mol, "键图碎片数": n_frag,
                "平均每分子原子数": round(mean_per_mol, 1),
                "移除方式": how,
            }
            info["n_molecules"] = n_frag
            info["sources"]["molnums"] = (
                f"拓扑自带的分子划分**过于粗糙**（{n_mol} 个分子 / 平均 "
                f"{mean_per_mol:.0f} 原子每分子，而键图有 {n_frag:,} 个碎片）；"
                f"已弃用该划分（{how}），下游按键连通分量分分子。")
        else:
            info["n_molecules"] = n_mol
            info["sources"]["molnums"] = "拓扑自带"
    elif needs_bond_guessing(u):
        info["sources"]["molnums"] = "无键表，无法划分分子（已跳过）"
    else:
        n_res = u.residues.n_residues
        comp = _bond_component_ids(u)          # 原子级 0..C-1，低内存
        n_frag = int(comp.max()) + 1 if comp.size else 0
        # 残基信息过于粗糙（Material Studio 导出的 PDB：5596 原子只有 2 个残基）时，
        # "按残基编号分子"会把整个体系压成 1–2 个分子。这时**不写 molnums**，
        # 让下游回退到键连通分量（interface._atom_molnums）。
        if n_frag and n_res * 4 < n_frag:
            info["molnums_degenerate"] = {"残基数": n_res, "键图碎片数": n_frag}
            info["n_molecules"] = n_frag
            info["sources"]["molnums"] = (
                f"拓扑的残基信息**过于粗糙**（{n_res} 个残基 vs {n_frag:,} 个键图碎片，"
                f"典型是 Material Studio 导出的 PDB）；已跳过按残基编号，"
                f"下游按键连通分量分分子。")
            if verbose:
                print(f"[对象层属性] {info['sources']['molnums']}")
            return info
        # 每个残基取"**第一个原子**所属的键图分量"（原来遍历 fragments 属后写覆盖；
        # 正常体系里一个残基就是一个分子，两种写法结果相同，但这里不建 FragDict）
        res_of_atom = np.asarray(u.atoms.resindices, dtype=int)
        res_mol = np.full(n_res, -1, dtype=int)
        if n_frag and res_of_atom.size:
            _uni, first = np.unique(res_of_atom, return_index=True)
            res_mol[res_of_atom[first]] = comp[first]
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
            f"由键图连通分量按**残基**编号（{n_frag:,} 个碎片 → "
            f"{int(uniq.size):,} 个分子编号）")
        if n_unassigned:
            info["sources"]["molnums"] += f"；{n_unassigned} 个残基未归属"

    # ------------------------------------------------------------ segids
    # LAMMPS data（以及部分 `.xyz`）没有 segid；链清单/主链标签要读
    # `f.segids[0]`，缺了就是 `NoDataError`（和 resnames 同一类坑）。
    # 补一个常量段名 SYSTEM —— 与 `systeminfo` 里"segment 恒为 SYSTEM"的约定一致。
    try:
        cur = np.asarray(u.residues.segids, dtype=str)
        has_seg = (cur.size == u.residues.n_residues
                   and any(str(x).strip() for x in cur))
    except Exception:                                        # noqa: BLE001
        has_seg = False
    if not has_seg:
        try:
            u.add_TopologyAttr(
                "segids", np.array(["SYSTEM"] * u.residues.n_residues, dtype=object))
            info["added"].append("segids")
            info["sources"]["segids"] = "拓扑没有段名，统一记为 SYSTEM"
        except Exception:                                    # noqa: BLE001
            pass

    if verbose:
        print(f"[对象层属性] 补齐 {info['added'] or '无'}；"
              f"分子 {info['n_molecules']:,}；来源 {info['sources']}")
    return info