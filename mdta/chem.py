# -*- coding: utf-8 -*-
"""把"一个分子由哪些原子组成"翻译成人能看懂的名字。

用于「结构层级」里给每条链/每种组分一个身份：分子式、常见名或类别、
以及（对纯烃链）估计的聚合度。**只做确定性推断**：元素组成 + 环数 +
resname 提示，不做力场级识别，拿不准就只报分子式与类别。

示例
----
>>> summarize({"C": 12, "H": 26}, resname="ETH")
{'formula': 'C12H26', 'name': '十二烷', 'category': '烷烃', ...}
>>> summarize({"C": 100, "H": 200}, resname="PE")
{'formula': 'C100H200', 'name': '聚乙烯（重复单元 CH2）', 'dp': 100, ...}
"""
from __future__ import annotations

#: 常见分子式（Hill 序）→ 中文名
COMMON = {
    "H2O": "水", "CH4": "甲烷", "C2H6": "乙烷", "C2H4": "乙烯", "C2H2": "乙炔",
    "C3H8": "丙烷", "C3H6": "丙烯", "C4H10": "丁烷", "C5H12": "戊烷",
    "C6H14": "己烷", "C7H16": "庚烷", "C8H18": "辛烷", "C9H20": "壬烷",
    "C10H22": "癸烷", "C11H24": "十一烷", "C12H26": "十二烷",
    "C6H6": "苯", "C7H8": "甲苯", "C6H12": "环己烷", "C6H12O6": "葡萄糖",
    "CH4O": "甲醇", "C2H6O": "乙醇", "C3H8O": "丙醇", "CH2O": "甲醛",
    "CH2O2": "甲酸", "C2H4O2": "乙酸", "CO2": "二氧化碳", "CO": "一氧化碳",
    "NH3": "氨", "N2": "氮气", "O2": "氧气", "H2": "氢气",
    "C4H4S": "噻吩", "C5H5N": "吡啶",
    "NA": "钠离子", "CL": "氯离子", "K": "钾离子", "MG": "镁离子", "CA": "钙离子",
}

#: 常见 resname → 名字（Materials Studio / 力场的习惯命名）
RESNAME_HINT = {
    "ETH": "乙烯型片段（C2H4）", "ETOH": "乙醇", "GRA": "石墨烯（含环）",
    "SOL": "水", "WAT": "水", "HOH": "水", "TIP3": "水",
    "NA": "钠离子", "NA+": "钠离子", "CL": "氯离子", "CL-": "氯离子",
    "PE": "聚乙烯", "PP": "聚丙烯", "PS": "聚苯乙烯", "PVC": "聚氯乙烯",
    "PEG": "聚乙二醇", "PEO": "聚乙二醇", "PMMA": "聚甲基丙烯酸甲酯",
    "ALA": "丙氨酸", "GLY": "甘氨酸", "MET": "甲硫氨酸", "LEU": "亮氨酸",
    "LYS": "赖氨酸", "SER": "丝氨酸", "THR": "苏氨酸", "VAL": "缬氨酸",
    "PHE": "苯丙氨酸", "TYR": "酪氨酸", "TRP": "色氨酸", "PRO": "脯氨酸",
}

#: 聚合物的"重复单元"（元素组成），用于估聚合度
REPEAT_UNIT = {
    "CH2": ("聚乙烯", 1, 2),          # C1H2
    "C2H4": ("聚乙烯", 2, 4),         # 与乙烯同组成
    "C2H3": ("聚氯乙烯?/聚丙烯?（无法定）", 2, 3),
    "C3H6": ("聚丙烯", 3, 6),
    "C8H8": ("聚苯乙烯", 8, 8),
    "C2H4O": ("聚乙二醇", 2, 4),
}


def hill_formula(counts: dict) -> str:
    """按 Hill 序写分子式：有 C 先 C+H，其余按字母序（如 C12H26、H2O、C2H6O）。"""
    c = {k: int(v) for k, v in (counts or {}).items() if v}
    if not c:
        return ""
    if "C" in c:
        order = ["C"] + (["H"] if "H" in c else []) + \
                sorted(k for k in c if k not in ("C", "H"))
    else:
        order = sorted(c)
    parts = []
    for k in order:
        n = c[k]
        parts.append(k if n == 1 else f"{k}{n}")
    return "".join(parts)


def _hydrocarbon_stem(n_c: int) -> str:
    """碳数 → 中文前缀（1..20 常见）。"""
    names = {1: "甲", 2: "乙", 3: "丙", 4: "丁", 5: "戊", 6: "己", 7: "庚",
             8: "辛", 9: "壬", 10: "癸", 11: "十一", 12: "十二", 13: "十三",
             14: "十四", 15: "十五", 16: "十六", 17: "十七", 18: "十八",
             19: "十九", 20: "二十"}
    return names.get(int(n_c), f"C{int(n_c)}")


def summarize(counts: dict, *, resname: str = "", n_rings: int = 0,
              is_chain: bool | None = None, n_atoms: int | None = None) -> dict:
    """由元素组成（``{"C": 12, "H": 26}``）+ 环数 + resname 给出身份描述。

    返回 ``{"formula", "name", "category", "dp", "note"}``；判断不了就只给分子式。
    """
    f = hill_formula(counts)
    out = {"formula": f, "name": "", "alias": "", "category": "", "dp": None,
           "note": ""}
    if not f:
        return out
    big = (n_atoms is not None and int(n_atoms) > 60)
    small = (n_atoms is None or int(n_atoms) <= 40)

    # ---- ⓿ 长链饱和烃：按用户口径就是"聚合度 n 的聚乙烯"
    #      （C12H26 = CH₂ 重复 12 个 + 两个 CH₃ 封端；小分子名"十二烷"降级为别名）
    els0 = set(counts)
    n_c0 = int(counts.get("C", 0))
    n_h0 = int(counts.get("H", 0))
    if els0 <= {"C", "H"} and n_c0 >= 6 and n_h0 == 2 * n_c0 + 2 and not big:
        # 聚合度按**聚乙烯重复单元 −CH₂−CH₂−（C2H4）**计：DP = 碳数 / 2。
        # （用户实测：100 个碳的链应报 DP≈50，以前报 100 是把 CH₂ 当成了重复单元）
        dp0 = n_c0 // 2 if n_c0 % 2 == 0 else n_c0 / 2.0
        out["name"] = f"聚乙烯类（DP≈{dp0:g}）"
        out["alias"] = f"{_hydrocarbon_stem(n_c0)}烷（饱和烃，{n_c0} 个碳）"
        out["category"] = "饱和烃（CₙH₂ₙ₊₂）"
        out["dp"] = dp0
        dp0s = f"{dp0:g}"
        out["note"] = (f"{f}：按 C2H4（−CH₂−CH₂−）重复单元计聚合度 ≈ {dp0s}"
                       f"（{n_c0} 个碳）；两端 CH₃ 封端，"
                       f"按小分子命名即{_hydrocarbon_stem(n_c0)}烷")
        return out

    # ---- ① 常见分子式直接命中（大分子不套小分子名）
    if f in COMMON and not big:
        out["name"] = COMMON[f]
        out["category"] = "常见分子"
        if not small:
            out["category"] = "常见分子（较大片段）"
        return out

    els = set(counts)
    n_c = int(counts.get("C", 0))
    n_h = int(counts.get("H", 0))

    # ---- 聚苯乙烯：每个重复单元（C8H8）带 1 个苯环 —— 判据用"每 8 个碳 1 个环"，
    #      这样即使导出时丢了部分氢（实测 C160H122 只有 20 个环）也能认出来；
    #      聚合度直接取**环数**（= 苯乙烯单元个数）。
    if els == {"C", "H"} and n_rings >= 2 and n_c >= 16:
        ratio = n_c / float(n_rings)
        if 6.0 <= ratio <= 10.0:
            dp_s = int(n_rings)
            out["name"] = f"聚苯乙烯（DP≈{dp_s}）"
            out["category"] = "含环烃（苯乙烯聚合物）"
            out["dp"] = dp_s
            out["note"] = (f"{f}：含 {n_rings} 个苯环、平均每环 {ratio:.1f} 个碳，"
                           f"按 C8H8（−CH₂−CH(C₆H₅)−）重复单元计聚合度 ≈ {dp_s}")
            return out

    # ---- ② 纯烃：按 C/H 比判烷/烯/炔，再按碳数命名
    if els <= {"C", "H"} and n_c > 0:
        if n_h == 0:
            out["category"] = "纯碳骨架" + ("（含环）" if n_rings else "")
            out["name"] = f"碳骨架 C{n_c}" + ("（含环/石墨烯类）" if n_rings else "")
            return out
        if n_h == 2 * n_c + 2:
            out["category"] = "饱和烃（CₙH₂ₙ₊₂）"
            # 用户口径：C12H26 这种"12 个碳的长链"就是**聚合度 12 的聚乙烯**，
            # 所以两条都给：聚乙烯（DP≈n）为正式名字，烷烃俗名放 alias。
            if n_c >= 6:
                dp = n_c // 2 if n_c % 2 == 0 else n_c / 2.0
                out["name"] = f"聚乙烯类（DP≈{dp:g}）"
                out["alias"] = f"{_hydrocarbon_stem(n_c)}烷（饱和烃，{n_c} 个碳）"
                out["dp"] = dp
                out["note"] = (f"{f}：按 C2H4（−CH₂−CH₂−）重复单元计聚合度≈{dp:g}"
                               f"（{n_c} 个碳）；两端为 CH₃ 封端，"
                               f"按小分子命名即{_hydrocarbon_stem(n_c)}烷")
            else:
                out["name"] = f"{_hydrocarbon_stem(n_c)}烷"
        elif n_h == 2 * n_c:
            if n_rings > 0:
                out["category"] = "环烷烃 / 含环烃（CₙH₂ₙ）"
                out["name"] = f"含环烃（{_hydrocarbon_stem(n_c)}环/烯）"
            else:
                out["category"] = "烯烃 / 聚乙烯类（CₙH₂ₙ）"
                # 碳数够多且是链 → 更像聚乙烯片段，报"重复单元 CH2 + 聚合度"
                if n_c >= 6:
                    dp2 = n_c // 2 if n_c % 2 == 0 else n_c / 2.0
                    out["name"] = "聚乙烯类（C2H4 重复单元）"
                    out["dp"] = dp2
                    out["note"] = ("按 C2H4（−CH₂−CH₂−）重复单元计聚合度 ≈ "
                                   f"{dp2:g}（{n_c} 个碳）；"
                                   f"若这是小分子则应为烯烃 C{n_c}H{2 * n_c}")
                else:
                    out["name"] = f"{_hydrocarbon_stem(n_c)}烯"
        elif n_h == 2 * n_c - 2:
            out["category"] = "炔烃 / 二烯（CₙH₂ₙ₋₂）"
            out["name"] = f"{_hydrocarbon_stem(n_c)}炔/二烯"
        else:
            out["category"] = "不饱和烃"
            out["name"] = f"烃类 C{n_c}H{n_h}"
        if n_rings and not out["name"].startswith("含环"):
            out["category"] += "（含环）"
        return out

    # ---- ③ 含杂原子：给类别
    if "O" in els and "H" in els:
        out["category"] = "含氧有机物"
    elif "N" in els:
        out["category"] = "含氮有机物"
    elif "S" in els:
        out["category"] = "含硫化合物"
    elif n_c == 0:
        out["category"] = "无机物/离子"
    else:
        out["category"] = "有机物"
    # resname 提示词只对小分子可信（整条蛋白的 resname 只是"第一个残基的类型"）
    hint = RESNAME_HINT.get(str(resname).upper(), "") if small else ""
    if hint:
        out["name"] = hint
    elif big:
        # 大分子：说清"大分子 + 分子式"，不要硬套小分子名
        out["name"] = f"{out['category']}（{f}，{int(n_atoms):,} 原子/分子）"
        if "生物" not in out["category"] and els & {"N", "S", "P"}:
            out["note"] = ("含 N/S/P 且规模较大，通常是蛋白质/聚合物链；"
                           "元素组成只是总量，不构成结构鉴定")
    else:
        out["name"] = f"{out['category']} {f}"
    return out
