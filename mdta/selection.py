# -*- coding: utf-8 -*-
"""体系信息与原子/分子/链选择模块。

对应设计大纲第 7 章。提供两类功能：

1. **体系信息** —— 列出模拟盒、原子、residue、分子/链、组分类型。
2. **选择器** —— 支持按原子名称、residue、分子、链、原子编号选择，
   也支持直接写 MDAnalysis 选择语句（高级用法）。
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .topology import residue_names

# 常见水分子 / 离子 residue 名（GROMACS、AMBER、CHARMM 命名习惯）
WATER_RESNAMES = {
    "SOL", "HOH", "WAT", "H2O", "OH2", "TIP", "TIP2", "TIP3", "TIP3P", "TIP4P",
    "TIP4PEW", "TIP5P", "T3P", "T4P", "T5P", "SPC", "SPCE", "SPC/E", "DOD",
}
ION_RESNAMES = {
    "NA", "NA+", "SOD", "CL", "CL-", "CLA", "CHL", "K", "K+", "POT", "MG", "MG2",
    "CA", "CA2", "CAL", "ZN", "ZN2", "ZINC", "FE", "FE2", "FE3", "MN", "MN2",
    "CU", "CU1", "CU2", "CUM", "CO", "CO2", "NI", "NI2", "CD", "CD2", "HG",
    "HG2", "PB", "PB2", "SR", "SR2", "BA", "BA2",
    "LI", "LI+", "LIT", "RB", "RB+", "CS", "CS+", "BR", "BR-", "IOD", "I-",
    "F", "F-", "AL", "SO4", "PO4",
}

#: 力场里"改了名字的"标准氨基酸（CHARMM36m 的 M 前缀质子化变体等）。
#: MDAnalysis 的 ``protein`` 选择器只认标准名，这些得自己补上，
#: 否则被糖化的/加质子的残基会被当成"待研究组分"（polymer），蛋白序列就断了一截。
PROTEIN_RESNAME_EXTRA = {
    # CHARMM36m / CHARMM 的 M 前缀变体
    "MALA", "MARG", "MASN", "MASP", "MCYS", "MCYX", "MGLN", "MGLU", "MGLY",
    "MGLH", "MHIS", "MHSE", "MHSD", "MHSP", "MILE", "MLEU", "MLYS", "MLYN",
    "MMET", "MPHE", "MPRO", "MSER", "MTHR", "MTRP", "MTYR", "MVAL", "MASH",
    # 常见非标准/修饰残基
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "LYN", "ASH",
    "GLH", "ARN", "TYM", "SEP", "TPO", "PTR", "MLZ", "M3L", "KCX", "LLP",
    "CSO", "CME", "OCS", "SMC", "ALY", "FME", "PCA",
}
#: 常见糖/糖基 residue 名（糖蛋白的糖链、多糖等）。
#: MDAnalysis 的 ``protein`` 选择器不认这些残基，若不单独归类会被当成"聚合物"。
SUGAR_RESNAMES = {
    "BGLCNA", "BGLCA", "NAG", "NGA", "NDG", "BMA", "MAN", "GAL", "GLA", "GLC",
    "BGC", "GCS", "FUC", "FUL", "FCA", "SIA", "NAN", "SLB", "XYS", "XYP", "XYL",
    "ARA", "RIB", "RIP", "IDR", "GTR", "GCU", "GL0", "GLS", "TRE", "SUC", "MAL",
    "LAC", "AMG", "BXYL", "PSE", "RHA", "6LX", "UAP", "GLCNAC", "GALNAC",
    "NEU", "NEU5AC", "KDN", "MAN3", "BMA3", "A2G",
}

#: 糖的三字母核心码。GLYCAM-06 之类的命名会在前面加异头构型前缀
#: （``A`` = α、``B`` = β），例如 ``AFUC``（α-L-岩藻糖）、``BMAN``（β-D-甘露糖）、
#: ``AMAN``。因此匹配时先把前缀去掉再比对核心码。
_SUGAR_CORE = {
    "FUC", "MAN", "GAL", "GLC", "NAG", "NDG", "BMA", "BGC", "GLA", "SIA",
    "NAN", "XYL", "XYS", "ARA", "RIB", "RHA", "GUL", "IDO", "TAL", "ALL",
    "ALT", "TRE", "SUC", "MAL", "LAC", "AMG", "GCU", "GL0", "GLS", "PSE",
    "KDN", "NEU", "GTR", "FUL", "FCA", "SLB", "UAP", "API", "ACI", "BXY",
}

#: GROMACS ``.gro`` 的 residue 名字段只有 **5 个字符**，6 字符的 GLYCAM 名会被截断
#: （实测 ``BGLCNA`` → ``BGLCN``）。若只做精确匹配，被截断的糖残基会掉进
#: ``polymer`` 兜底类别，导致同一体系用 ``.gro`` 与 ``.tpr`` 得到不同的组分。
#: 因此核心码比对允许"前缀命中"，长度上限取 5（原始名最长为 6，
#: 剥掉异头前缀后最长 5）。
_MAX_CORE_LEN = 5


def is_sugar_resname(resname: str) -> bool:
    """判断 residue 名是否属于糖/糖基。

    匹配顺序：

    1. 精确命中 :data:`SUGAR_RESNAMES`；
    2. 剥掉 GLYCAM 风格的 ``A``/``B``（α/β）前缀后命中 :data:`_SUGAR_CORE`；
    3. 剥掉前缀后**以**某个核心码开头且长度不超过 5 —— 用于兼容 ``.gro``
       对 6 字符残基名的截断（``BGLCNA`` → ``BGLCN``）。

    标准氨基酸名不会误判：``ALA`` 剥掉前缀后是 ``LA``（长度不足）；
    CHARMM 变体 ``MARG``/``MHSE`` 等不以任何糖核心码开头。
    """
    rn = str(resname).strip().upper()
    if not rn:
        return False
    if rn in SUGAR_RESNAMES:
        return True
    core = rn
    while len(core) > 3 and core[0] in "AB":
        core = core[1:]
    if core in _SUGAR_CORE:
        return True
    # 截断兼容：GLYCAM 名最长 6 字符，剥掉异头前缀后 <= 5
    if 4 <= len(core) <= _MAX_CORE_LEN and core[:3] in _SUGAR_CORE:
        return True
    return False

_CHARGE_HINTS = ("+", "-")

#: 单原子残基里**元素**属于这些符号时按离子处理（resname 常常是 ``MOL`` / ``1`` /
#: 空串之类，靠名字认不出来）。用户实测：MS 导出的 Na⁺/Cl⁻ 在拓扑里叫 ``MOL``，
#: 结果被归成 ``other``，界面上找不到"离子"这一组分。
ION_ELEMENTS = {
    "NA", "K", "LI", "RB", "CS", "MG", "CA", "SR", "BA", "ZN", "FE", "MN",
    "CU", "CO", "NI", "CD", "HG", "PB", "AL", "CL", "BR", "I", "F",
}


class SelectionError(ValueError):
    """选择语句错误。"""


# ---------------------------------------------------------------- 组分分类
def _residue_composition(u) -> list[dict]:
    """每个 residue 的原子组成 ``[{"O":1,"H":2}, ...]``（元素符号大写）。

    读不到元素属性时退回原子名前缀；再不行记 ``X``。用于"按组成兜底识别"——
    resname 各式各样（``MOL`` / ``1`` / 空），但水的组成永远是 H₂O。
    """
    n_res = u.residues.n_residues
    out: list[dict] = [{} for _ in range(n_res)]
    try:
        resind = np.asarray(u.atoms.resindices, dtype=int)
    except Exception:                                    # noqa: BLE001
        return out
    try:
        els = [str(x).strip().upper() for x in u.atoms.elements]
    except Exception:                                    # noqa: BLE001
        els = []
        try:
            for nm in u.atoms.names:
                s = str(nm).strip().upper()
                k = 0
                while k < len(s) and s[k].isalpha():
                    k += 1
                els.append(s[:k][:2] or "X")
        except Exception:                                # noqa: BLE001
            els = ["X"] * int(u.atoms.n_atoms)
    for el, ri in zip(els, resind):
        if 0 <= ri < n_res:
            d = out[int(ri)]
            d[el] = d.get(el, 0) + 1
    return out


def is_water_composition(comp: dict) -> bool:
    """按**元素组成**判水：1 个 O、1–3 个 H、无其它元素（≤5 原子）。

    兼容 TIP4P/TIP5P 的虚拟位点（会多一个 ``M`` 或空元素，故只看"非 H 非 O"的
    原子数 ≤ 2）。这样 resname 是 ``MOL`` 的 MS 水分子也能认出来。
    """
    if not comp:
        return False
    n_o = int(comp.get("O", 0))
    n_h = int(comp.get("H", 0))
    if n_o != 1 or not (1 <= n_h <= 3):
        return False
    other = sum(v for k, v in comp.items() if k not in ("H", "O", "M", "X", ""))
    return other == 0 and sum(comp.values()) <= 6


def is_ion_composition(comp: dict) -> bool:
    """按组成判离子：**只有一个原子**，且元素属于常见离子元素。"""
    if not comp or sum(comp.values()) != 1:
        return False
    el = next(iter(comp))
    return str(el).upper() in ION_ELEMENTS


def classify_residues(universe) -> "OrderedDict[str, np.ndarray]":
    """把 residue 按组分类型分类，返回 ``{类别: residue 索引数组}``。

    类别包括：``protein`` / ``nucleic`` / ``sugar`` / ``polymer`` / ``water`` /
    ``ion`` / ``other``。

    ``polymer`` 是兜底类别，含义是"未被识别为蛋白/核酸/糖/水/离子的待研究组分"
    ——高分子体系里的链、以及不认识的有机小分子都会落到这里。

    识别顺序：**先名字，再组成**。名字（resname）是首选，但 MS/LAMMPS 导出的
    体系里名字常常没意义（水的 resname 是 ``MOL``、离子的也是 ``MOL``），
    只靠名字会把水并进 polymer（用户实测："水都被识别成聚合物了"）。
    因此这里再用元素组成兜底：H₂O 组成 → water，单原子 + 常见离子元素 → ion。
    """
    u = universe
    n_res = u.residues.n_residues
    labels = np.array(["other"] * n_res, dtype=object)

    resnames = residue_names(u)
    res_counts = np.bincount(np.asarray(u.atoms.resindices, dtype=int), minlength=n_res)

    is_water = np.isin(resnames, list(WATER_RESNAMES))
    is_ion = np.isin(resnames, list(ION_RESNAMES))
    is_sugar = np.array([is_sugar_resname(rn) for rn in resnames], dtype=bool)
    # 带电荷后缀、且原子数很少的也算离子
    is_ion = is_ion | ((res_counts <= 4)
                       & np.array([any(c in rn for c in _CHARGE_HINTS) for rn in resnames]))

    # ---- 按**组成**兜底（名字认不出来时）：水 / 单原子离子
    comps = _residue_composition(u)
    for i, c in enumerate(comps):
        if is_water[i] or is_ion[i] or is_sugar[i]:
            continue
        if is_water_composition(c):
            is_water[i] = True
        elif is_ion_composition(c):
            is_ion[i] = True

    labels[is_water] = "water"
    labels[is_ion] = "ion"
    labels[is_sugar] = "sugar"

    for kw, lab in (("protein", "protein"), ("nucleic", "nucleic")):
        try:
            idx = u.select_atoms(kw).residues.ix
            labels[idx] = lab
        except Exception:  # noqa: BLE001
            pass

    # 力场里的非标准氨基酸名（CHARMM36m 的 MARG/MHSE 等）MDAnalysis 不认，
    # 补进 protein，否则蛋白序列会被截断
    extra_prot = np.isin(resnames, list(PROTEIN_RESNAME_EXTRA))
    if extra_prot.any():
        labels[extra_prot] = "protein"

    # 未被识别的中大分子 residue 视为高分子/待研究组分
    unknown = labels == "other"
    if unknown.any():
        labels[unknown & (res_counts >= 3)] = "polymer"

    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for lab in ("protein", "nucleic", "sugar", "polymer", "water", "ion", "other"):
        idx = np.nonzero(labels == lab)[0]
        if idx.size:
            out[lab] = idx
    return out


def residue_category(universe) -> np.ndarray:
    """每个 residue 的类别标签（长度为 ``n_residues`` 的字符串数组）。"""
    labels = np.array(["other"] * universe.residues.n_residues, dtype=object)
    for lab, idx in classify_residues(universe).items():
        labels[idx] = lab
    return labels


# ---------------------------------------------------------------- 组分原子组
def component_groups(universe, min_atoms: int = 1) -> "OrderedDict[str, object]":
    """自动识别体系中可用于界面分析的组分（protein / water / ion / ...）。

    返回 ``{组分名: AtomGroup}``，键名形如 ``protein``、``water``、``ion``、
    ``sugar``、``polymer``。当存在多个 segid 时，会按 segid 进一步细分，
    例如 ``polymer[seg_0_PE]``。
    """
    u = universe
    cats = classify_residues(u)
    out: "OrderedDict[str, object]" = OrderedDict()
    for lab, ridx in cats.items():
        ag = u.residues[ridx].atoms
        if ag.n_atoms < min_atoms:
            continue
        segs = sorted(set(str(s) for s in np.asarray(ag.segids, dtype=str)))
        if len(segs) > 1 and lab in ("polymer", "protein", "nucleic", "sugar"):
            for s in segs:
                sub = ag[np.asarray(ag.segids, dtype=str) == s]
                if sub.n_atoms >= min_atoms:
                    out[f"{lab}[{s}]"] = sub
        else:
            out[lab] = ag
    return out


# ---------------------------------------------------- 初步识别"这是什么分子"
#: 常见离子元素 → 中文名（suggested label 用）
_ION_ZH = {
    "NA": "钠离子 Na⁺", "K": "钾离子 K⁺", "LI": "锂离子 Li⁺", "RB": "铷离子 Rb⁺",
    "CS": "铯离子 Cs⁺", "MG": "镁离子 Mg²⁺", "CA": "钙离子 Ca²⁺",
    "SR": "锶离子 Sr²⁺", "BA": "钡离子 Ba²⁺", "ZN": "锌离子 Zn²⁺",
    "FE": "铁离子", "MN": "锰离子", "CU": "铜离子", "CO": "钴离子",
    "NI": "镍离子", "CD": "镉离子", "HG": "汞离子", "PB": "铅离子",
    "AL": "铝离子", "CL": "氯离子 Cl⁻", "BR": "溴离子 Br⁻", "I": "碘离子 I⁻",
    "F": "氟离子 F⁻",
}


def _element_array(ag) -> list[str]:
    """组内每个原子的元素符号（大写）；缺元素属性时用原子名前缀兜底。"""
    n = int(ag.n_atoms)
    try:
        el = [str(x).strip().upper() or "X" for x in ag.elements]
        if len(el) == n:
            return el
    except Exception:                                    # noqa: BLE001
        pass
    out: list[str] = []
    try:
        names = [str(x).strip().upper() for x in ag.names]
        for nm in names[:n]:
            k = 0
            while k < len(nm) and nm[k].isalpha():
                k += 1
            out.append(nm[:k][:2] or "X")
    except Exception:                                    # noqa: BLE001
        out = ["X"] * n
    return out


def _hill_formula(counts: dict) -> str:
    """Hill 式分子式（C、H 在前，其余按字母序）。"""
    if not counts:
        return ""
    order: list[str] = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    for k in sorted(counts):
        if k not in order:
            order.append(k)
    parts = []
    for k in order:
        c = int(counts[k])
        parts.append(k if c == 1 else f"{k}{c}")
    return "".join(parts)


#: 标准氨基酸名（含常见质子化/修饰变体）——只用于"这是不是蛋白"的初步判断
_AMINO_NAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ACE", "NME", "NH2", "HYP", "SEP", "TPO", "PTR", "MSE", "KCX", "MLZ",
} | set(PROTEIN_RESNAME_EXTRA)
#: 20 种标准氨基酸的**三字母核心**：CHARMM 会用 ``CYSH`` / ``HISB`` / ``LYSH``
#: 这类"核心码 + 质子化状态"的写法，gromacs 也有 ``CYS2`` 之类 → 允许 3+1 位。
_AMINO_CORE = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _is_amino_resname(rn: str) -> bool:
    """residue 名是不是（可能带质子化状态后缀的）氨基酸。"""
    rn = str(rn).strip().upper()
    if not rn:
        return False
    if rn in _AMINO_NAMES:
        return True
    return len(rn) == 4 and rn[:3] in _AMINO_CORE


#: 核酸残基名（DNA/RNA 的碱基与糖-磷酸残基）
_NUCLEIC_NAMES = {
    "DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "T", "RA", "RC", "RG",
    "RU", "ADE", "CYT", "GUA", "THY", "URA", "PSU", "5MC", "OMG", "OMC",
}


def _residue_set(ag) -> set[str]:
    """该组的 residue 名集合（大写）；没有 resnames 属性时返回空集。"""
    try:
        return {str(r).strip().upper() for r in ag.residues.resnames if str(r).strip()}
    except Exception:                                    # noqa: BLE001
        return set()


def _classify_formula(counts: dict) -> tuple[str, str]:
    """由**单个分子**的元素计数给出 (kind, 中文名)。只做保守推断，宁可说"有机分子"。"""
    if not counts:
        return ("unknown", "未知")
    n_atoms = sum(int(v) for v in counts.values())
    if is_water_composition(counts):
        return ("water", "水 H₂O")
    if is_ion_composition(counts):
        el = next(iter(counts))
        return ("ion", _ION_ZH.get(str(el).upper(), f"{el} 离子"))
    c = int(counts.get("C", 0))
    h = int(counts.get("H", 0))
    o = int(counts.get("O", 0))
    n = int(counts.get("N", 0))
    s = int(counts.get("S", 0))
    p = int(counts.get("P", 0))
    others = {k: int(v) for k, v in counts.items()
              if k not in ("C", "H", "O", "N", "S", "P")}
    if c > 0 and h == 0 and o == 0 and n == 0 and s == 0 and p == 0 and not others:
        if c >= 100:
            return ("carbon", "碳材料（石墨烯 / 碳片）")
        return ("carbon", f"碳分子 C{c}")
    if c > 0 and o == 0 and n == 0 and s == 0 and p == 0 and not others:
        if h == 2 * c + 2:
            return ("alkane", f"正构烷烃 C{c}H{h}")
        if h == 2 * c:
            return ("alkene", f"烯烃（含一个不饱和键）C{c}H{h}")
        if h == 2 * c - 2:
            return ("alkyne", f"炔烃 / 二烯 C{c}H{h}")
        return ("hydrocarbon", f"烃 C{c}H{h}")
    if o > 0 and n == 0 and p == 0 and not others:
        # 聚乙二醇 / 聚氧乙烯：C_nH_(2n+2)O_(n/2)（PEO20 = C40H82O20）
        if h == 2 * c + 2 and c >= 4 and o * 2 == c:
            return ("peo", f"聚乙二醇（PEO / PEG）链 {_hill_formula(counts)}")
        return ("oxygenate", f"含氧有机物 {_hill_formula(counts)}")
    if n > 0 and p == 0 and not others:
        return ("nitrogenous", f"含氮有机物 C{c}H{h}" + (f"N{n}" if n > 1 else "N"))
    if p > 0:
        return ("phosphorous", "含磷有机物")
    if s > 0:
        return ("sulfur", "含硫有机物")
    if any(k in ("NA", "CL", "K", "CA", "MG") for k in others) and n_atoms <= 2:
        return ("ion", "离子 / 离子对")
    return ("organic", f"有机分子（{_hill_formula(counts)}）")


def identify_component(ag) -> dict:
    """**初步识别这个组分是什么分子**：按分子式给中文名 + 分子式 + 分子数。

    这是"看得懂"的辅助信息，不参与任何计算：
    用户实测痛点 —— 界面上只写 ``polymer[ETH]`` / ``water`` 这种标签，
    体系里有水、烃链、离子混在一起时看不出"这到底是些什么分子"。

    返回::

        {"label": "水 H₂O", "kind": "water", "formula": "H2O",
         "n_molecules": 2000, "atoms_per_molecule": 3,
         "n_kinds": 1, "detail": "2000 个分子 · 每个 3 原子"}

    多分散（同一组分里有多种分子式）时 ``n_kinds > 1``，``label`` 会带「N 种」。
    """
    out = {"label": "未知", "kind": "unknown", "formula": "",
           "n_molecules": 0, "atoms_per_molecule": 0, "n_kinds": 0, "detail": ""}
    if ag is None or int(getattr(ag, "n_atoms", 0)) <= 0:
        return out

    # 分子划分：优先用分子编号（与其它分析同一口径），拿不到就当整组一个分子
    try:
        from .analysis.interface import _atom_molnums

        m = _atom_molnums(ag)
    except Exception:                                    # noqa: BLE001
        m = None
    els = _element_array(ag)
    if m is None or len(m) != len(els):
        m = np.zeros(len(els), dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)

    uniq, inv = np.unique(m, return_inverse=True)
    n_mol = int(uniq.size)
    per_mol: list[dict] = [{} for _ in range(n_mol)]
    for el, k in zip(els, inv.tolist()):
        d = per_mol[int(k)]
        d[el] = d.get(el, 0) + 1

    # 按分子式归并（同式子 = 同种分子）。虚拟位点（TIP4P 的 ``M``、无元素的 ``X``）
    # 不进分子式 —— 否则水会显示成 ``H2OX``（用户看起来像坏掉了）。
    groups: "OrderedDict[str, int]" = OrderedDict()
    sample: dict[str, dict] = {}
    for d in per_mol:
        clean = {k: v for k, v in d.items() if k not in ("X", "M", "", "UNK")}
        f = _hill_formula(clean) or "?"
        groups[f] = groups.get(f, 0) + 1
        sample.setdefault(f, clean)
    n_kinds = len(groups)

    main_f, main_cnt = max(groups.items(), key=lambda kv: kv[1])
    kind, zh = _classify_formula(sample.get(main_f, {}))
    # 生物大分子先按**残基名**判（分子式看不出"这是蛋白"）
    rns = _residue_set(ag)
    if rns and all(_is_amino_resname(r) for r in rns):
        kind, zh = "protein", f"蛋白质 / 多肽（{len(rns)} 种氨基酸残基）"
    elif rns and all(r in _NUCLEIC_NAMES for r in rns):
        kind, zh = "nucleic", f"核酸链（{len(rns)} 种核苷酸残基）"
    elif rns and all(is_sugar_resname(r) for r in rns):
        kind, zh = "sugar", f"糖 / 糖链（{len(rns)} 种糖残基）"
    atoms_each = sum(int(v) for v in sample.get(main_f, {}).values())
    label = zh if n_kinds <= 1 else f"{zh} 等 {n_kinds} 种"
    detail = f"{n_mol:,} 个分子"
    if atoms_each:
        detail += f" · 每个 {atoms_each} 原子"
    if n_kinds > 1:
        detail += f" · 分子式共 {n_kinds} 种（最多的是 {main_f}，{main_cnt} 个）"
    out.update(label=label, kind=kind, formula=main_f, n_molecules=n_mol,
               atoms_per_molecule=int(atoms_each), n_kinds=int(n_kinds),
               detail=detail)
    return out


# ---------------------------------------------------------------- 链/分子
@dataclass
class ChainType:
    """一类链/分子的汇总信息（同一 segid + resname）。

    同一类里可能含有不同大小的碎片（例如 TPR 中的水分子可能被解析成
    "3 个真实原子 + 1 个孤立虚拟位点"），因此用 ``variants`` 记录各尺寸的出现次数。
    """

    label: str
    segid: str
    resname: str
    count: int = 0
    #: ``{(n_residues, n_atoms): 条数}``
    variants: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        """该类中最主要尺寸的原子数。"""
        if not self.variants:
            return 0
        return max(self.variants, key=lambda k: (self.variants[k] * k[1], k[1]))[1]

    @property
    def n_residues(self) -> int:
        if not self.variants:
            return 0
        return max(self.variants, key=lambda k: (self.variants[k] * k[1], k[1]))[0]

    def __str__(self) -> str:
        if len(self.variants) == 1:
            (nr, na), cnt = next(iter(self.variants.items()))
            return f"{self.label}: {cnt} 条, 每条 {nr} residue / {na} 原子"
        parts = ", ".join(
            f"{cnt} 条×{na} 原子" for (nr, na), cnt in
            sorted(self.variants.items(), key=lambda kv: -kv[0][1])
        )
        return f"{self.label}: 共 {self.count} 个碎片 —— {parts}"


def _first_resname(ag) -> str:
    """取第一个原子的残基名；拓扑若**没有 resnames**（LAMMPS data）返回空串。

    g.resnames 在缺属性时抛 NoDataError，而链清单（list_chains）与
    chains() 是打开体系必走的路径 —— 一个缺属性就能让整个体系打不开。
    """
    try:
        return str(ag.resnames[0]) if ag.n_atoms else ""
    except Exception:                                        # noqa: BLE001
        try:
            return str(residue_names(ag.universe)[int(ag.resindices[0])]) if ag.n_atoms else ""
        except Exception:                                    # noqa: BLE001
            return ""


def _fragments(universe) -> list:
    """返回分子/链列表，**元素一定是 AtomGroup**（这点很关键）。

    有键信息时按连通性切分；没有键信息时退化为"按 segment 分组"，
    再退化为"整个体系算一条链"。

    注意：不能用 ``list(u.atoms.residues)`` 兜底 —— 那返回的是
    :class:`MDAnalysis.core.groups.Residue`，没有 ``n_atoms`` / ``positions``，
    会让下游分析全部报 ``Residue has no attribute n_atoms``。

    ⚠️ **不用** ``u.atoms.fragments``：它第一次调用要建整张 ``_fragdict``，
    AdK（47 681 原子）实测 ~2 s，而打开体系时 `largest_chains` 和
    `list_chains` 各要调一次。改成自己按键表并查集（0.29 s）并把结果缓存
    在 Universe 上，两个调用只算一遍。实测单读 .tpr 从 4.6 s 降到 ~3 s。
    """
    u = universe
    cached = getattr(u, "_mdta_fragments_cache", None)
    if cached is not None:
        return cached
    try:
        if len(u.bonds) > 0:               # NoDataError 表示没有 bonds 属性
            import numpy as _np

            from .topology import bond_components

            n = int(u.atoms.n_atoms)
            comp = bond_components(
                n, _np.asarray(u.bonds.indices, dtype=_np.int64))
            order = _np.argsort(comp, kind="stable")
            cs = comp[order]
            starts = _np.flatnonzero(_np.r_[True, cs[1:] != cs[:-1]])
            ends = _np.r_[starts[1:], n]
            frags = [u.atoms[order[s:e]]
                     for s, e in zip(starts.tolist(), ends.tolist())]
            if frags:
                try:
                    u._mdta_fragments_cache = frags
                except Exception:                            # noqa: BLE001
                    pass
                return frags
    except Exception:  # noqa: BLE001
        pass

    # 没有键信息：按 segment 分组（.gro 之类常见只有 1~3 个 segment）
    try:
        segs = list(u.segments)
        if segs:
            return [s.atoms for s in segs]
    except Exception:  # noqa: BLE001
        pass
    return [u.atoms]


def has_bond_info(universe) -> bool:
    """当前体系是否含键连接信息（决定"链"能否按连通性划分）。"""
    try:
        return len(universe.bonds) > 0
    except Exception:  # noqa: BLE001
        return False


def list_chains(universe, min_atoms: int = 1) -> list[ChainType]:
    """按（segid, resname）归类所有分子/链。

    这样即便体系里有上万个水分子，也只显示一行汇总，而高分子链会逐条列出。
    同一 segid+resname 下若存在不同尺寸的碎片，会一并记录在 ``variants`` 中。
    """
    u = universe
    frags = _fragments(u)
    buckets: "OrderedDict[tuple, ChainType]" = OrderedDict()
    for i, f in enumerate(frags):
        if f.n_atoms < min_atoms:
            continue
        seg = str(f.segids[0]) if f.n_atoms else ""
        resn = _first_resname(f)
        key = (seg, resn)
        if key not in buckets:
            label = f"{seg}:{resn}" if seg else resn
            buckets[key] = ChainType(label=label, segid=seg, resname=resn)
        ct = buckets[key]
        n_res = len(set(f.resids.tolist()))
        vkey = (int(n_res), int(f.n_atoms))
        ct.variants[vkey] = ct.variants.get(vkey, 0) + 1
        ct.count += 1
    return list(buckets.values())


def chains(universe, *, segid: str | None = None, resname: str | None = None,
           min_atoms: int = 1, max_atoms: int | None = None) -> list:
    """返回符合条件的分子/链（AtomGroup 列表）。

    参数
    ----
    segid, resname
        按段名/residue 名过滤，``None`` 表示不限。
    min_atoms, max_atoms
        按原子数过滤，便于挑出"高分子链"而排除水分子。
    """
    out = []
    for f in _fragments(universe):
        if f.n_atoms < min_atoms:
            continue
        if max_atoms is not None and f.n_atoms > max_atoms:
            continue
        if segid is not None and str(f.segids[0]) != segid:
            continue
        if resname is not None and _first_resname(f) != resname:
            continue
        out.append(f)
    return out


def largest_chains(universe, n: int = 5, min_atoms: int = 20) -> list:
    """按原子数返回最大的 ``n`` 条分子/链。

    对高分子体系，最大的若干条通常就是待研究的高分子链。
    """
    fs = [f for f in _fragments(universe) if f.n_atoms >= min_atoms]
    fs.sort(key=lambda f: f.n_atoms, reverse=True)
    return fs[:n]


# ---------------------------------------------------------------- 选择器
def _fmt_list(values: Iterable) -> str:
    return " ".join(str(v) for v in values)


def select(
    universe,
    query: str | None = None,
    *,
    name: Sequence[str] | str | None = None,
    type: Sequence[str] | str | None = None,  # noqa: A002 - 与 MDAnalysis 关键字一致
    resname: Sequence[str] | str | None = None,
    resid: Sequence[int] | range | tuple | int | None = None,
    segid: Sequence[str] | str | None = None,
    index: Sequence[int] | int | None = None,
    category: str | None = None,
    within: float | None = None,
    of: str | None = None,
    around: float | None = None,
):
    """统一的原子选择接口。

    参数
    ----
    query
        原生 MDAnalysis 选择语句，例如 ``"resname SOL and name OW"``。
        提供后其余参数会以 ``and`` 追加。
    name, type, resname, segid
        按原子名/原子类型/residue 名/段名选择，可给单个字符串或列表。
    resid
        按 residue 编号选择，支持 ``range`` 或 ``(start, stop)`` 元组。
    index
        按绝对原子编号（0 起）选择。
    category
        按自动识别的组分类型选择（protein/water/ion/polymer/nucleic）。
    within / of
        选择距离 ``of`` 语句所定义原子组 ``within`` Å 以内的原子。
    around
        选择 ``of`` 周围 ``around`` Å 以内的原子（同 ``within``，语义更直观）。

    返回
    ----
    ``MDAnalysis.AtomGroup``
    """
    parts: list[str] = []
    if query:
        parts.append(f"({query})")

    def add(kw: str, val) -> None:
        if val is None:
            return
        if isinstance(val, str):
            parts.append(f"{kw} {val}")
        elif isinstance(val, (range, tuple)):
            parts.append(f"{kw} {val[0]}:{val[-1]}")
        elif isinstance(val, (int, np.integer)):
            parts.append(f"{kw} {int(val)}")
        else:
            vals = list(val)
            if not vals:
                raise SelectionError(f"{kw} 的候选列表为空")
            parts.append(f"{kw} {_fmt_list(vals)}")

    add("name", name)
    add("type", type)
    add("resname", resname)
    add("segid", segid)
    add("resid", resid)
    add("index", index)

    if (within or around) and of:
        d = within if within else around
        # MDAnalysis 语法： (...group...) and around <d> (...of...)
        parts.append(f"around {float(d):g} ({of})")

    if not parts and not category:
        raise SelectionError("没有给出任何选择条件。")

    if parts:
        expr = " and ".join(parts)
        try:
            ag = universe.select_atoms(expr)
        except Exception as exc:  # noqa: BLE001
            raise SelectionError(f"选择语句无效: {expr!r} -> {exc}") from exc
    else:
        ag = universe.atoms

    if category:
        cat_ag = select_category(universe, category)
        if parts:
            mask = np.isin(ag.indices, cat_ag.indices)
            ag = ag[mask]
        else:
            ag = cat_ag

    return ag


def select_category(universe, category: str):
    """按组分类型选择原子组。"""
    cats = classify_residues(universe)
    if category not in cats:
        raise SelectionError(f"体系中不存在类别 {category!r}；可选类别: {list(cats)}")
    return universe.residues[cats[category]].atoms


def interface_region(universe, group_a, group_b, cutoff: float = 5.0):
    """界面区域：组分 A 中距组分 B 小于 ``cutoff`` Å 的原子。

    对应设计大纲 7.2 的"选择界面区域"。
    """
    from MDAnalysis.lib.distances import capped_distance

    pos_a = group_a.positions
    pos_b = group_b.positions
    if pos_a.size == 0 or pos_b.size == 0:
        return group_a[[]]
    pairs, _ = capped_distance(pos_a, pos_b, max_cutoff=float(cutoff),
                               box=universe.dimensions, return_distances=False)
    if pairs.size == 0:
        return group_a[[]]
    keep = np.unique(pairs[:, 0])
    return group_a[keep]


def summarize_selection(ag) -> "OrderedDict[str, object]":
    """给出一个原子组的可读摘要，用于界面/日志显示。"""
    d: "OrderedDict[str, object]" = OrderedDict()
    d["原子数"] = int(ag.n_atoms)
    d["residue 数"] = int(ag.residues.n_residues)
    if ag.n_atoms:
        d["涉及段"] = sorted(set(str(s) for s in np.asarray(ag.segids, dtype=str)))
        d["主要原子名"] = Counter(np.asarray(ag.names, dtype=str)).most_common(6)
        try:
            d["总质量 (amu)"] = round(float(ag.masses.sum()), 3)
        except Exception:  # noqa: BLE001
            pass
    return d
