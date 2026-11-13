# -*- coding: utf-8 -*-
"""下载文件的**英文短名**（图表区保存图片 / 数据表区下载 CSV / 导出板块共用）。

为什么统一在这里
----------------
文件名要满足三条硬要求：

1. **纯 ASCII** —— 浏览器下载时文件名走 HTTP 头的 ``Content-Disposition``，
   Starlette/uvicorn 按 latin-1 编码头字段，一旦带中文就直接抛
   ``UnicodeEncodeError``，用户看到的就是"下载不了"（实测踩过）；
2. **短** —— 打开文件夹一眼能认出来，别把整个中文标题塞进去；
3. **图表区/数据表区/导出板块三处一致** —— 否则同一个面板会导出三个不同名字。

所以按 ``(分析项英文名, 面板序号)`` 给一个短英文 token，兜底取标题里的 ASCII
单词，再兜底用 ``p{n}``。前端通过接口下发的 ``panels[i].slug`` 直接用同一个值，
不需要在前端再抄一份规则。
"""

from __future__ import annotations

import re

__all__ = ["PANEL_SLUGS", "panel_slug", "safe_ascii"]

#: ``(分析项, 面板序号) -> 短英文 token``。取名词尽量短（3–8 字符）且能自解释。
PANEL_SLUGS: dict[tuple[str, int], str] = {
    # 链构象
    ("rg", 0): "time",          # Rg(t)
    ("rg", 1): "dist",          # Rg 分布
    ("rg", 2): "permol",        # 各分子平均 Rg
    ("ree", 0): "time",         # R_ee(t)
    ("ree", 1): "dist",
    ("dihedral", 0): "dist",    # 二面角分布（概率密度）
    ("dihedral", 1): "trans",   # trans/gauche 随时间
    # 空间结构
    ("density", 0): "profile",  # 组分密度分布
    ("density", 1): "norm",     # 归一化密度
    ("rdf", 0): "rdf",
    ("rdf", 1): "halves",       # 前半段 vs 后半段
    ("rdf2d", 0): "inplane",    # 面内 RDF
    ("contact", 0): "count",    # 接触数随时间
    ("contact", 1): "mindist",  # 最小间距
    ("contact", 2): "matrix",   # 接触矩阵
    ("contact", 3): "pairs",    # 接触对数分布
    ("contact", 4): "coord",    # 配位数分布
    ("interface", 0): "profile",  # 密度分布与界面
    ("interface", 1): "norm",     # 归一化密度与宽度
    ("comdist", 0): "dist",       # 质心距随时间
    ("comdist", 1): "axis",       # 沿轴分量
    # 取向与结晶
    ("orientation", 0): "s",      # 取向参数 S
    ("orientation", 1): "p2",     # 二阶序参数 P2
    ("orientation", 2): "dist",   # 链段取向分布
    ("order", 0): "components",   # 各有序度分量
    ("order", 1): "index",        # 多分量指数
    ("boo", 0): "qbar6",          # q̄6 随时间
    ("boo", 1): "qbar6_dist",     # q̄6 分布
    ("boo", 2): "q4q6",           # q̄4–q̄6 平面
    ("crystal", 0): "frac",       # 晶区原子占比
    ("crystal", 1): "cluster",    # 最大晶簇
    # 动力学与输运
    ("msd", 0): "tau",            # MSD(τ)
    ("msd", 1): "fit",            # 线性区拟合
    ("msd", 2): "axes",           # 各轴分解
    ("msd", 3): "power",          # 幂律
    ("vacf", 0): "vacf",
    ("vacf", 1): "fit",
}

_ASCII_WORD = re.compile(r"[A-Za-z0-9]+")


def safe_ascii(text: str, fallback: str = "result") -> str:
    """把任意文本压成**纯 ASCII** 的文件名片段（非 ASCII 字符直接丢弃）。"""
    s = str(text or "")
    s = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "._-")) else "_"
                for ch in s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or fallback


def panel_slug(name: str, index: int = 0, title: str = "") -> str:
    """面板的**短英文**文件名 token（纯 ASCII）。

    查找顺序：``PANEL_SLUGS[(分析项, 面板号)]`` → 标题里的 ASCII 单词
    （最多两个，小写下划线连起来）→ ``p{面板号+1}``。
    """
    key = (str(name or ""), int(index or 0))
    if key in PANEL_SLUGS:
        return PANEL_SLUGS[key]
    words = _ASCII_WORD.findall(str(title or ""))
    if words:
        return "_".join(w.lower() for w in words[:2])[:24]
    return f"p{int(index or 0) + 1}"
