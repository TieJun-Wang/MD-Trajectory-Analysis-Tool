# -*- coding: utf-8 -*-
"""分子与聚合物链自动识别（结构层级的深层识别，对应 recognize.md）。

设计要点
--------
· **分子归属只用化学键连接关系**：拓扑里的键表（MS 的 ``<Bond Connects>``、
  GROMACS 的 tpr、PDB 的 CONECT…）建无向图，连通分量 = 一个共价连接组分。
  **绝不**用空间距离阈值判键，也不靠元素种类/原子数/残基名去猜。
· 连通分量 ≠ 独立聚合物链：在连通分量内部再用**重原子骨架**的拓扑特征
  （环秩 = E − V + 1、支化点数、重原子数）判定它到底是小分子、线形链、
  支化链、环状链还是交联网络。判不准就标 ``unknown``，并留下警告。
· 识别结果一次算完并缓存（:meth:`Recognition`），原子归属可全量校验：
  ``已归属原子数 == 有效原子数``，同一层级内不重复、不丢失。
· 只做"识别 + 归属 + 选择"，不参与任何分析计算（Rg/RDF/MSD/Contact 都不动）。

对外只用两个东西：:func:`recognize` 和 :class:`Recognition`。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------- 判定阈值
#: 重原子数达到这个数才可能算"聚合物组分"。
#: 10 是有意的：C10 以上才谈得上寡聚/聚合；苯（6 个重原子）必须留在小分子里。
POLY_MIN_HEAVY = 10
#: 环秩（E − V + 1）达到 max(NETWORK_MIN_RINGS, NETWORK_RING_RATIO × 重原子数)
#: 且支化点 ≥ 3 时判为交联/网络（石墨烯片、交联树脂都在这里）。
NETWORK_MIN_RINGS = 3
NETWORK_RING_RATIO = 0.02

#: 组分（sections）顺序与中文名 —— 结构树就按这个顺序排
SECTIONS = [
    ("polymer", "聚合物"),
    ("small_molecule", "小分子"),
    ("network", "网络 / 交联"),
    ("unknown", "未识别"),
]
SECTION_LABEL = dict(SECTIONS)

#: 对象类型（链类型）中文名
CHAIN_TYPE_LABEL = {
    "linear": "线形链",
    "branched": "支化链",
    "cyclic": "环状链",
    "network": "交联网络",
    "single": "单原子",
}


def _elements(u) -> np.ndarray:
    """元素符号（大写）；拓扑没给元素时按原子名/类型兜底，仍未知记 ``X``。"""
    n = int(u.atoms.n_atoms)
    try:
        el = np.array([str(x).strip().upper() or "X" for x in u.atoms.elements],
                      dtype=object)
        if el.size == n:
            return el
    except Exception:                                        # noqa: BLE001
        pass
    out = np.array(["X"] * n, dtype=object)
    try:
        names = [str(x).strip().upper() for x in u.atoms.names]
        for i, nm in enumerate(names[:n]):
            # 原子名常写成 C1/H12/Cl3 这种：取开头的字母就是元素
            k = 0
            while k < len(nm) and nm[k].isalpha():
                k += 1
            if k:
                out[i] = nm[:k][:2]
    except Exception:                                        # noqa: BLE001
        pass
    return out


@dataclass
class Recognition:
    """一次识别的全部结果（紧凑数组 + 按需展开的对象清单）。"""

    n_atoms: int
    #: 每个原子所属的**分子**下标（0..n_molecules-1）
    mol_of_atom: np.ndarray
    #: 每个分子所属的**链**下标（不是聚合物则为 -1）
    chain_of_mol: np.ndarray
    #: 每个分子的分组下标（同结构同成分的分子归一组）
    group_of_mol: np.ndarray
    #: 每个分子的类别：polymer / small_molecule / network / unknown
    type_of_mol: list[str]
    #: 每个分子的链类型：linear / branched / cyclic / network / single / -
    chain_type_of_mol: list[str]
    #: 每个分子的原子数 / 键数 / 重原子数 / 环秩 / 支化点数
    atoms_of_mol: np.ndarray
    bonds_of_mol: np.ndarray
    heavy_of_mol: np.ndarray
    rings_of_mol: np.ndarray
    branch_of_mol: np.ndarray
    #: 6 元环数 / 苯环数（6 元碳环、环内键长 ≤1.50 Å）/ 环大小分布
    ring6_of_mol: np.ndarray
    benz_of_mol: np.ndarray
    ring_sizes_of_mol: list
    #: 每个分子的元素组成 ``{元素: 个数}``（只在需要时用，分子级别很小）
    comp_of_mol: list[dict]
    #: 每个分子的代表残基名（拓扑里的 resname，可能为空串）
    resname_of_mol: list[str]
    #: 每个分子的识别置信度与状态
    conf_of_mol: np.ndarray
    status_of_mol: list[str]
    #: 分组：``{key, label, formula, type, chain_type, mol_ids, n_objects, n_atoms}``
    groups: list[dict]
    #: 键表来源：``topology`` / ``molnums`` / ``none``
    bond_source: str
    #: 警告（面向用户，可直接显示）
    warnings: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    #: 人工确认覆盖：``{对象 id: {name?, type?, chain_type?, status?}}``
    overrides: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 基本量
    @property
    def n_molecules(self) -> int:
        return int(self.type_of_mol.__len__())

    def mol_indices(self, molid: int) -> np.ndarray:
        """某个分子的**全部原子索引**（原始 Universe 编号）。"""
        return np.flatnonzero(self.mol_of_atom == int(molid))

    # ------------------------------------------------------------ 对象命名
    def mol_id(self, molid: int) -> str:
        return f"M{int(molid):06d}"

    def chain_id(self, molid: int) -> str:
        return f"C{int(molid):06d}"

    def mol_name(self, molid: int) -> str:
        """人类可读名字：``ETH_032``（分组标签 + 组内序号）。"""
        m = int(molid)
        ov = self.overrides.get(self.mol_id(m)) or {}
        if ov.get("name"):
            return str(ov["name"])
        g = int(self.group_of_mol[m])
        label = self._group_label(g)
        # 组内序号（1 基）：该分子在本组里的位置
        first = int(self.groups[g]["mol_ids"][0])
        return f"{label}_{m - first + 1:03d}"

    def _group_label(self, g: int) -> str:
        grp = self.groups[g]
        return str(grp.get("label") or grp.get("formula") or "UNK")

    # ------------------------------------------------------------ 摘要
    def summary(self) -> dict:
        counts = Counter(self.type_of_mol)
        n_unknown = int(counts.get("unknown", 0))
        n_assigned = int(self.atoms_of_mol.sum())
        return {
            "total_atoms": int(self.n_atoms),
            "total_molecules": int(self.n_molecules),
            "total_polymer_chains": int(counts.get("polymer", 0)),
            "total_network_components": int(counts.get("network", 0)),
            "total_small_molecules": int(counts.get("small_molecule", 0)),
            "total_unknown": n_unknown,
            "total_groups": int(len(self.groups)),
            "assigned_atoms": n_assigned,
            "unassigned_atoms": int(self.n_atoms - n_assigned),
            "bond_source": self.bond_source,
            "elapsed_sec": round(float(self.elapsed_sec), 3),
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------ 结构树
    def tree(self) -> dict:
        """结构树数据：System → 类别 → 分组（每个分组再按需展开成对象）。

        统计数字**全部来自真实识别结果**，不在前端推算。
        """
        sections: dict[str, list] = {}
        for g_i, grp in enumerate(self.groups):
            sect = str(grp["type"])
            sections.setdefault(sect, []).append({
                "group_id": f"G{g_i:04d}",
                "group_index": int(g_i),
                "label": self._group_label(g_i),
                "formula": grp.get("formula", ""),
                "chem": grp.get("chem", {}),
                "type": grp["type"],
                "type_label": SECTION_LABEL.get(grp["type"], grp["type"]),
                "chain_type": grp.get("chain_type", ""),
                "chain_type_label": CHAIN_TYPE_LABEL.get(
                    grp.get("chain_type", ""), grp.get("chain_type", "")),
                "n_objects": int(grp["n_objects"]),
                "n_atoms": int(grp["n_atoms"]),
                "n_residues": int(grp.get("n_residues", 0)),
                "status": grp.get("status", "recognized"),
                "status_label": _STATUS_LABEL.get(
                    grp.get("status", "recognized"), ""),
            })
        out_sections = []
        for key, label in SECTIONS:
            kids = sections.get(key) or []
            if not kids:
                continue
            kids.sort(key=lambda x: -x["n_objects"])
            out_sections.append({
                "key": key,
                "label": label,
                "n_groups": len(kids),
                "n_objects": int(sum(x["n_objects"] for x in kids)),
                "n_atoms": int(sum(x["n_atoms"] for x in kids)),
                "groups": kids,
            })
        return {
            "sections": out_sections,
            "summary": self.summary(),
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------ 分页对象
    def objects(self, group_index: int | None = None, *, section: str = "",
                offset: int = 0, limit: int = 100) -> dict:
        """某一组（或某一类别）下的**对象清单**（分页）。

        大体系（上万个分子）不可能一次给完，所以按组 + 分页取。
        """
        if group_index is None:
            ids = [i for i in range(self.n_molecules) if not section
                   or self.type_of_mol[i] == section]
        else:
            g = int(group_index)
            if not 0 <= g < len(self.groups):
                raise ValueError(f"不存在的分组下标 {group_index}")
            ids = list(self.groups[g]["mol_ids"])
        total = len(ids)
        page = ids[int(offset):int(offset) + int(limit)]
        rows = [self._object_row(i) for i in page]
        return {"total": int(total), "offset": int(offset),
                "rows": rows, "returned": len(rows)}

    def _object_row(self, molid: int) -> dict:
        m = int(molid)
        ov = self.overrides.get(self.mol_id(m)) or {}
        typ = str(ov.get("type") or self.type_of_mol[m])
        ctype = str(ov.get("chain_type") or self.chain_type_of_mol[m])
        label = self._group_label(int(self.group_of_mol[m]))
        item = {
            "object_type": "molecule" if typ == "small_molecule" else "chain",
            "object_id": self.mol_id(m) if typ == "small_molecule" else self.chain_id(m),
            "molecule_id": self.mol_id(m),
            "chain_id": (self.chain_id(m) if ctype not in ("-", "single") else ""),
            "name": self.mol_name(m),
            "label": label,
            "type": typ,
            "type_label": SECTION_LABEL.get(typ, typ),
            "chain_type": ctype,
            "chain_type_label": CHAIN_TYPE_LABEL.get(ctype, ctype),
            "n_atoms": int(self.atoms_of_mol[m]),
            "n_bonds": int(self.bonds_of_mol[m]),
            "n_heavy": int(self.heavy_of_mol[m]),
            "n_rings": int(self.rings_of_mol[m]),
            # 6 元环 / 苯环（石墨烯那种带虚线圈的芳香六元环）
            "n_ring6": (int(self.ring6_of_mol[m])
                        if self.ring6_of_mol is not None else 0),
            "n_benzene": (int(self.benz_of_mol[m])
                          if self.benz_of_mol is not None else 0),
            "ring_sizes": (dict(self.ring_sizes_of_mol[m])
                           if self.ring_sizes_of_mol else {}),
            "n_branch": int(self.branch_of_mol[m]),
            "composition": dict(self.comp_of_mol[m]),
            "resname": self.resname_of_mol[m],
            "recognition_confidence": float(round(float(self.conf_of_mol[m]), 2)),
            "status": str(ov.get("status") or self.status_of_mol[m]),
        }
        item["status_label"] = _STATUS_LABEL.get(item["status"], item["status"])
        return item

    # ------------------------------------------------------------ 展开选择
    def expand(self, object_type: str, object_id: str) -> dict:
        """把结构树上的任意节点展开成**完整原子索引**（原始 Universe 编号）。

        ``object_type``：``section`` / ``group`` / ``molecule`` / ``chain`` / ``all``
        """
        ot = str(object_type or "all").lower()
        oid = str(object_id or "")
        if ot in ("all", "system"):
            idx = np.arange(self.n_atoms, dtype=np.int64)
            return {"label": "全部原子", "object_type": ot, "n_atoms": int(idx.size),
                    "indices": idx.tolist()}
        if ot == "section":
            keep = np.isin(self.mol_of_atom,
                           np.array([i for i in range(self.n_molecules)
                                     if self.type_of_mol[i] == oid], dtype=np.int64))
            idx = np.flatnonzero(keep)
            return {"label": f"{SECTION_LABEL.get(oid, oid)}（全部）",
                    "object_type": ot, "n_atoms": int(idx.size),
                    "indices": idx.tolist()}
        if ot == "group":
            # 允许 G0007 或纯数字
            g = int(oid[1:]) if oid.upper().startswith("G") else int(oid)
            if not 0 <= g < len(self.groups):
                raise ValueError(f"不存在的分组 {object_id!r}")
            ids = np.asarray(self.groups[g]["mol_ids"], dtype=np.int64)
            idx = np.flatnonzero(np.isin(self.mol_of_atom, ids))
            return {"label": f"{self._group_label(g)}（{len(ids)} 个对象）",
                    "object_type": ot, "n_atoms": int(idx.size),
                    "indices": idx.tolist()}
        if ot in ("molecule", "chain"):
            m = _parse_mol_id(oid)
            if m is None or not 0 <= m < self.n_molecules:
                raise ValueError(f"不存在的对象 {object_id!r}")
            idx = self.mol_indices(m)
            return {"label": self.mol_name(m), "object_type": ot,
                    "molecule_id": self.mol_id(m),
                    "n_atoms": int(idx.size), "indices": idx.tolist()}
        raise ValueError(f"不认识的 object_type {object_type!r}"
                         "（section / group / molecule / chain / all）")

    def detail(self, molid: int) -> dict:
        """单个对象的详情（右侧属性区用）。"""
        row = self._object_row(molid)
        row["atom_indices"] = self.mol_indices(molid).tolist()
        return row


_STATUS_LABEL = {
    "recognized": "已识别",
    "partially_recognized": "部分识别",
    "unrecognized": "未识别",
    "manually_confirmed": "人工确认",
}


def _parse_mol_id(s: str) -> int | None:
    t = str(s or "").strip().upper()
    if not t:
        return None
    if t[0] in "MC" and t[1:].isdigit():
        return int(t[1:])
    if t.isdigit():
        return int(t)
    return None


# ==================================================================== 识别主体
def recognize(universe, *, timeout_hint: float | None = None) -> Recognition:
    """对当前 Universe 做一次分子 / 聚合物链识别（结果自带缓存价值，别反复调）。"""
    import time

    t0 = time.time()
    u = universe
    n = int(u.atoms.n_atoms)
    warnings: list[str] = []

    if n == 0:
        raise ValueError("体系里没有原子，无法识别")

    el = _elements(u)

    # ---- 键图：以拓扑自带的键为准
    edges = np.zeros((0, 2), dtype=np.int64)
    bond_source = "none"
    try:
        nb = len(u.bonds)
    except Exception:                                        # noqa: BLE001
        nb = 0
    if nb:
        try:
            edges = np.asarray(u.bonds.indices, dtype=np.int64).reshape(-1, 2)
            edges = edges[(edges[:, 0] < n) & (edges[:, 1] < n)]
            bond_source = "topology"
        except Exception:                                    # noqa: BLE001
            edges = np.zeros((0, 2), dtype=np.int64)

    if edges.size:
        comp = _components(n, edges)
        # ---- 用拓扑自带的**分子编号**再并一次：
        # TIP4P 水的虚位点 M 在 tpr 里没有出现在键表里（GROMACS 用 settle 约束），
        # 于是它成了"孤立单原子分子"，11084 个水就被拆成 11084 + 11084 个组分。
        # 分子编号是拓扑给的信息（不是距离推断），同一编号的组分本来就该是一个分子。
        comp = _merge_by_molnums(u, comp)
    else:
        # 没有键表：退回到拓扑里的分子编号（它本身也多由键推出），仍没有就全体系一个
        mol = None
        try:
            from .topology import molecule_ids

            mol = molecule_ids(u.atoms)
        except Exception:                                    # noqa: BLE001
            mol = None
        if mol is not None and np.unique(mol).size > 1:
            _uniq, comp = np.unique(np.asarray(mol, dtype=np.int64),
                                    return_inverse=True)
            comp = comp.astype(np.int64)
            bond_source = "molnums"
            warnings.append(
                "拓扑里没有键表：分子归属回退到拓扑自带的分子编号，"
                "无法验证键连接（按分子/链选择的结果可能不准）")
        else:
            comp = np.zeros(n, dtype=np.int64)
            bond_source = "none"
            warnings.append(
                "拓扑里既没有化学键也没有分子编号：整个体系只能作为 1 个未识别组分。"
                "勾选「按坐标猜键」重新打开体系可得到键表")

    n_c = int(comp.max()) + 1 if n else 0

    # ---- 逐分子统计（向量化，不用循环 5 万次）
    v = np.bincount(comp, minlength=n_c).astype(np.int64)
    if edges.size:
        e = np.bincount(comp[edges[:, 0]], minlength=n_c).astype(np.int64)
        deg = np.bincount(edges.ravel(), minlength=n)
    else:
        e = np.zeros(n_c, dtype=np.int64)
        deg = np.zeros(n, dtype=np.int64)
    heavy = np.array([str(x).upper() != "H" for x in el], dtype=bool)
    hv = np.bincount(comp[heavy], minlength=n_c).astype(np.int64)
    if edges.size:
        he_mask = heavy[edges[:, 0]] & heavy[edges[:, 1]]
        he_edges = edges[he_mask]
    else:
        he_edges = np.zeros((0, 2), dtype=np.int64)
    if he_edges.size:
        he = np.bincount(comp[he_edges[:, 0]], minlength=n_c).astype(np.int64)
        hdeg = np.zeros(n, dtype=np.int64)
        np.add.at(hdeg, he_edges.ravel(), 1)
        branch = np.bincount(comp[hdeg >= 3], minlength=n_c).astype(np.int64)
    else:
        he = np.zeros(n_c, dtype=np.int64)
        branch = np.zeros(n_c, dtype=np.int64)
    # 环秩（连通分量内）：E − V + 1，重原子骨架同理
    rings = np.maximum(0, he - hv + 1)

    # 6 元环 / 苯环：只给**有环**的分子算（无环分子一次都不跑，不影响大体系速度）
    ring6 = np.zeros(n_c, dtype=np.int64)
    benz = np.zeros(n_c, dtype=np.int64)
    ring_sizes: list[dict] = [dict() for _ in range(n_c)]
    try:
        from .rings import analyze_rings
        cand = np.flatnonzero(rings > 0)
        if 0 < cand.size <= 20000:                    # 防御：超多分子不逐个枚举
            for c in cand:
                idxs = np.flatnonzero(comp == c)
                st = analyze_rings(u, idxs)
                ring6[c] = int(st["n_ring6"])
                benz[c] = int(st["n_benzene"])
                ring_sizes[c] = dict(st["sizes"])
    except Exception:                                 # noqa: BLE001
        pass

    # ---- 元素组成 / 残基名（每个分子一份，分子通常很小）
    comp_of = [dict() for _ in range(n_c)]
    for i in range(n):
        c = int(comp[i])
        d = comp_of[c]
        k = str(el[i])
        d[k] = d.get(k, 0) + 1
    try:
        atom_resname = [str(x) for x in u.atoms.resnames]
    except Exception:                                        # noqa: BLE001
        atom_resname = [""] * n
    resname_of: list[str] = []
    n_res_of: np.ndarray = np.ones(n_c, dtype=np.int64)
    protein_of = [False] * n_c
    try:
        atom_res = np.asarray(u.atoms.resindices, dtype=np.int64)
        if atom_res.size != n:
            atom_res = None
    except Exception:                                        # noqa: BLE001
        atom_res = None
    for c in range(n_c):
        idx = np.flatnonzero(comp == c)
        cnt = Counter(atom_resname[i] for i in idx)
        resname_of.append(str(cnt.most_common(1)[0][0]) if cnt else "")
        if atom_res is not None:
            n_res_of[c] = int(np.unique(atom_res[idx]).size)
        # 蛋白质判定只用来**命名**（不影响任何拓扑分类）：
        # 多残基 + 六成以上残基是标准氨基酸名
        if n_res_of[c] >= 10 and cnt:
            hit = sum(1 for k in cnt if k.upper() in _AMINO)
            if hit / len(cnt) >= 0.6:
                protein_of[c] = True

    # ---- 分类
    from .chem import summarize

    type_of: list[str] = []
    ctype_of: list[str] = []
    for c in range(n_c):
        typ = _classify(int(v[c]), int(hv[c]), int(rings[c]), int(branch[c]))
        type_of.append(typ)
        ctype_of.append(_chain_type(int(v[c]), int(hv[c]), int(rings[c]),
                                    int(branch[c]), typ))
    if bond_source == "none":
        # 一点键信息都没有时，任何"分子边界"都是假的 —— 整体系只能算 1 个未识别组分。
        # 宁可标 unknown 也不要把 5 个不相连的原子说成"一个小分子"。
        type_of = ["unknown"] * n_c
        ctype_of = ["-"] * n_c

    # ---- 分组：同类别 + 同链类型 + 同成分 + 同重原子数 → 同一组
    #      （这就是"122 条乙烯链"能被识别成一组 122 个对象的依据，纯拓扑，不靠名字）
    keys: list[tuple] = []
    for c in range(n_c):
        formula = _formula(comp_of[c])
        keys.append((type_of[c], ctype_of[c], formula, int(hv[c])))
    group_index: dict[tuple, int] = {}
    group_of = np.zeros(n_c, dtype=np.int64)
    groups: list[dict] = []
    for c in range(n_c):
        k = keys[c]
        g = group_index.get(k)
        if g is None:
            g = len(groups)
            group_index[k] = g
            n_rings = 1 if rings[c] > 0 else 0
            info = summarize(comp_of[c], resname=resname_of[c], n_rings=n_rings,
                             is_chain=(ctype_of[c] in ("linear", "branched", "cyclic")),
                             n_atoms=int(v[c]))
            label = _group_label_from(
                resname_of[c], info, n_residues=int(n_res_of[c]),
                protein=bool(protein_of[c]))
            groups.append({
                "key": list(k), "formula": k[2], "type": type_of[c],
                "chain_type": ctype_of[c], "label": label, "chem": info,
                "mol_ids": [], "n_objects": 0, "n_atoms": 0,
                "n_residues": int(n_res_of[c]),
                "status": "recognized",
            })
        group_of[c] = g
        grp = groups[g]
        grp["mol_ids"].append(c)
        grp["n_objects"] += 1
        grp["n_atoms"] += int(v[c])

    # ---- 分组排序：按类别顺序、组内对象数从多到少，让 G0000 就是最显眼的那个
    sec_order = {k: i for i, (k, _l) in enumerate(SECTIONS)}
    order = sorted(range(len(groups)),
                   key=lambda g: (sec_order.get(groups[g]["type"], 99),
                                  -groups[g]["n_objects"],
                                  -groups[g]["n_atoms"]))
    if order != list(range(len(groups))):
        remap = np.empty(len(groups), dtype=np.int64)
        for new, old in enumerate(order):
            remap[old] = new
        groups = [groups[old] for old in order]
        group_of = remap[group_of]

    # ---- 链编号：只在"聚合物/网络"上给（小分子标签为 -1）
    chain_of = np.full(n_c, -1, dtype=np.int64)
    k = 0
    for c in range(n_c):
        if type_of[c] in ("polymer", "network"):
            chain_of[c] = k
            k += 1

    # ---- 置信度 / 状态
    conf = np.ones(n_c, dtype=float)
    if bond_source != "topology":
        conf *= 0.6
    lonely = (deg == 0) & (v[comp] > 1)          # 分子里"没有键"的原子（键表缺行）
    n_lonely = int(lonely.sum())
    if n_lonely:
        conf *= 0.85
    unknown_el = int(np.sum(el == "X"))
    if unknown_el:
        conf *= 0.9
    conf = np.clip(conf, 0.0, 1.0)
    status_of = []
    for c in range(n_c):
        if type_of[c] == "unknown":
            status_of.append("unrecognized")
        elif conf[c] >= 0.9:
            status_of.append("recognized")
        elif conf[c] >= 0.5:
            status_of.append("partially_recognized")
        else:
            status_of.append("unrecognized")

    if n_lonely:
        warnings.append(
            f"有 {n_lonely} 个原子没有任何键（键表可能缺行，或存在周期镜像/游离原子）："
            "它们会跟着所在分子一起归属，但分子边界可能不完整")
    if unknown_el:
        warnings.append(f"有 {unknown_el} 个原子识别不出元素（记为 X）：按重原子处理，"
                        "环/支化判据可能偏保守")
    if any(t == "unknown" for t in type_of):
        n_unk = int(sum(1 for t in type_of if t == "unknown"))
        warnings.append(f"{n_unk} 个组分无法可靠判断类型，已标为「未识别」"
                        "（宁可标未识别，也不猜错归属）")

    # 分组状态：整组都高置信才算 recognized（组内残基数已在建组时算好）
    for grp in groups:
        cs = [float(conf[c]) for c in grp["mol_ids"]]
        grp["min_conf"] = min(cs) if cs else 0.0
        grp["status"] = ("recognized" if min(cs) >= 0.9
                         else "partially_recognized" if min(cs) >= 0.5
                         else "unrecognized")

    # ---- 一致性校验：归属必须不重不漏
    n_assigned = int(v.sum())
    if n_assigned != n:
        warnings.append(f"归属校验失败：已归属 {n_assigned} / 有效 {n} 个原子")

    return Recognition(
        n_atoms=n,
        mol_of_atom=comp.astype(np.int64),
        chain_of_mol=chain_of,
        group_of_mol=group_of,
        type_of_mol=type_of,
        chain_type_of_mol=ctype_of,
        atoms_of_mol=v,
        bonds_of_mol=e,
        heavy_of_mol=hv,
        rings_of_mol=rings,
        ring6_of_mol=ring6,
        benz_of_mol=benz,
        ring_sizes_of_mol=ring_sizes,
        branch_of_mol=branch,
        comp_of_mol=comp_of,
        resname_of_mol=resname_of,
        conf_of_mol=conf,
        status_of_mol=status_of,
        groups=groups,
        bond_source=bond_source,
        warnings=warnings,
        elapsed_sec=time.time() - t0,
    )


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    """键图的连通分量（并查集；不依赖 scipy，也不用任何距离判据）。"""
    pl = list(range(n))
    for x, y in zip(edges[:, 0].tolist(), edges[:, 1].tolist()):
        rx = pl[x]
        while pl[rx] != rx:
            rx = pl[rx]
        ry = pl[y]
        while pl[ry] != ry:
            ry = pl[ry]
        if rx != ry:
            pl[rx] = ry
    # 最终一遍压缩 + 重新编号成 0..C-1
    for i in range(n):
        r = i
        while pl[r] != r:
            r = pl[r]
        pl[i] = r
    _uniq, comp = np.unique(np.asarray(pl, dtype=np.int64), return_inverse=True)
    return comp.astype(np.int64)


def _merge_by_molnums(u, comp: np.ndarray) -> np.ndarray:
    """把**同一个分子编号**下的多个键连通分量并成一个分子。

    为什么需要（真实数据）：TIP4P/OPLS 的水在 tpr 里有 4 个原子（O、H、H、M），
    M 是虚位点，GROMACS 用 settles 约束、不写成键，于是 M 成了孤立原子 ——
    11084 个水被拆成 11084 个"3 原子水" + 11084 个"单原子水"。

    分子编号（``molnums``）本身就是拓扑信息，同一编号的组分必定是同一个分子，
    用它合并**不算**距离推断；没有 molnums 或退化成一个编号时原样返回。
    """
    n = int(comp.size)
    try:
        mol = np.asarray(u.atoms.molnums, dtype=np.int64)
        if mol.size != n:
            return comp
    except Exception:                                        # noqa: BLE001
        return comp
    if np.unique(mol).size <= 1:
        return comp
    # 同一个 molnum 下出现过哪些分量 → 把它们并起来（并查集作用在"分量号"上）
    n_c = int(comp.max()) + 1
    order = np.argsort(mol, kind="stable")
    mol_sorted = mol[order]
    comp_sorted = comp[order]
    parent = list(range(n_c))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 同一 molnum 的连续区间里，把所有分量都并到区间第一个分量
    starts = np.flatnonzero(np.r_[True, mol_sorted[1:] != mol_sorted[:-1]])
    ends = np.r_[starts[1:], mol_sorted.size]
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s <= 1:
            continue
        base = int(comp_sorted[s])
        for k in range(s + 1, e):
            ra, rb = root(base), root(int(comp_sorted[k]))
            if ra != rb:
                parent[rb] = ra
    roots = np.array([root(i) for i in range(n_c)], dtype=np.int64)
    _uniq, inv = np.unique(roots, return_inverse=True)
    return inv.astype(np.int64)[comp]


def _formula(counts: dict) -> str:
    """Hill 式分子式（C、H 在前，其余按字母序）——分组键用，必须稳定。"""
    if not counts:
        return ""
    order = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    for k in sorted(counts):
        if k not in order:
            order.append(k)
    out = []
    for k in order:
        c = int(counts[k])
        out.append(k if c == 1 else f"{k}{c}")
    return "".join(out)


def _classify(n_atoms: int, hv: int, rings: int, branch: int) -> str:
    """把连通分量分成 polymer / small_molecule / network / unknown。

    只用**拓扑量**：重原子数、环秩、支化点数。不看元素种类，也不看残基名。
    """
    if n_atoms <= 0:
        return "unknown"
    if hv <= 1:
        # 单重原子（离子、甲烷这类）——小分子
        return "small_molecule"
    ring_gate = max(NETWORK_MIN_RINGS, NETWORK_RING_RATIO * hv)
    if rings >= ring_gate and branch >= 3:
        return "network"
    if hv >= POLY_MIN_HEAVY:
        return "polymer"
    return "small_molecule"


def _chain_type(n_atoms: int, hv: int, rings: int, branch: int, typ: str) -> str:
    """链类型：只用拓扑量判定。

    顺序有意：**先排除网络，再判支化，最后才谈环**。
    反例就是蛋白质：它有芳香环（环秩 > 0）但主体是侧链支化，
    先判环会把它说成"环状链"，与事实不符。
    """
    if typ == "unknown":
        return "-"
    if n_atoms == 1:
        return "single"
    ring_gate = max(NETWORK_MIN_RINGS, NETWORK_RING_RATIO * hv)
    if rings >= ring_gate and branch >= 3:
        return "network"
    if typ != "polymer":
        return "cyclic" if rings > 0 else "-"
    if branch >= 1:
        return "branched"
    if rings > 0:
        return "cyclic"
    return "linear"


#: 标准氨基酸（GROMACS / CHARMM / AMBER 常见写法），只用于**显示命名**
_AMINO = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "LYN", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX",
    "CYM", "ACE", "NME", "NH2", "HYP", "SEP", "TPO", "PTR",
}


def _group_label_from(resname: str, chem: dict, *, n_residues: int = 1,
                      protein: bool = False) -> str:
    """分组标签：优先用拓扑里的残基名（ETH/SOL/NA+），否则用化学名/分子式。

    只有**残基构成单一**的组分才用残基名做标签：像蛋白质那种一条链里几十种
    残基的，拿"最常见那个残基"当名字会误导（实测会显示成 LYSH）。
    """
    if protein:
        return "蛋白质"
    rn = str(resname or "").strip()
    if rn and rn.upper() not in ("UNK", "RES", "MOL", "SYSTEM") and n_residues <= 3:
        return rn
    name = str((chem or {}).get("name") or "").strip()
    if name:
        return name.split("（")[0][:12]
    return str((chem or {}).get("formula") or "UNK")[:12]
