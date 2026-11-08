# -*- coding: utf-8 -*-
"""**自适应可视化引擎**：先看数据长什么样，再决定画什么图。

用户要求（自适应可视化引擎）
----------------------------
1. 数据分布识别：连续型 / 离散型 / 周期型 / 高度集中 / 恒定值 / 无效值；
2. 自动图表选择：类型 → 推荐图（直方图 / 频数柱状图 / 角度分布 / 折线 / 热力图 /
   接触矩阵 / 时间曲线 / 数值卡片）；
3. 高度集中时的处理：识别窄分布 → 给分箱宽度与坐标范围的建议 → 必要时退化成
   单点图或数值卡片；**不改原始数据、不加噪声**；
4. 统计信息：Mean / Std / Min / Max / Median / Count / Distribution。

这里只做"判断 + 建议"，不画图（画图在 Web 前端）。所有结论都由数据本身决定，
调用方可以原样展示，也可以忽略。
"""

from __future__ import annotations

import math

import numpy as np

#: 推荐的图型（前端按这个选 ECharts series / 布局）
CHARTS = {
    "line": "折线图",            # 连续型 / 空间分布 / 时间序列
    "hist": "直方图",            # 连续型数据的分布
    "bar": "频数柱状图",         # 离散型
    "angle": "角度分布图",       # 周期型（0–360° / 0–π）
    "polar": "极坐标图",         # 周期型（方向分布）
    "scatter": "散点图",         # 点太少、集中
    "card": "数值卡片",          # 单一数值 / 恒定值
}

#: 数据分布类型
TYPES = {
    "continuous": "连续型",
    "discrete": "离散型",
    "periodic": "周期型",
    "concentrated": "高度集中",
    "constant": "恒定值",
    "invalid": "无效/异常",
}

#: 判定为"高度集中"的阈值：标准差 / (最大值−最小值)。
#: 0.08 是有意的：**正态分布**的 σ/极差 ≈ 1/6 ≈ 0.17，所以只有明显比正态更尖的
#: 数据才算"高度集中"（否则正常分布也会被误判成一条竖线）。
_CONCENTRATED = 0.08
#: 角度类数据的判断：单位含 ° 或 deg，且值域落在周期区间内
_ANGLE_HINTS = ("°", "deg", "角度", "theta", "θ", "phi", "φ", "取向角")


def _clean(y) -> np.ndarray:
    a = np.asarray(y, dtype=float).ravel()
    return a[np.isfinite(a)]


def stats_of(y) -> dict:
    """统计信息：Mean / Std / Min / Max / Median / Count。空数据给 ``None``。"""
    a = _clean(y)
    if not a.size:
        return {"count": 0, "mean": None, "std": None, "min": None,
                "max": None, "median": None}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
    }


def _is_periodic(values: np.ndarray, label: str) -> bool:
    """周期型（角度）判定。

    ⚠️ 只在**标签明确是角度**、或**值域明显是度数**（0–360、跨度 ≥150°）时才认。
    早先还允许"值落在 0..π 且跨度 ≥1.4"当弧度 —— 结果把归一化的 **Rg 分布**
    （取值 0~1.6 的密度曲线）误判成角度分布，画成极坐标图（实测踩到）。
    """
    txt = str(label or "").lower()
    if any(h.lower() in txt for h in _ANGLE_HINTS):
        return True
    if not values.size:
        return False
    lo, hi = float(values.min()), float(values.max())
    return (lo >= -181.0 and hi <= 361.0 and (hi - lo) >= 150.0)


def _is_discrete(values: np.ndarray) -> bool:
    """离散型：取值种类少、且全是整数（接触数、分子编号、配位数、晶簇尺寸这种）。"""
    if values.size < 5:
        return False
    uniq = np.unique(np.round(values, 9))
    if uniq.size > min(40, max(3, values.size // 3)):
        return False
    return bool(np.all(np.abs(uniq - np.round(uniq)) < 1e-9))


def classify(x=None, y=None, *, label: str = "", xlabel: str = "") -> dict:
    """判断一条曲线的数据类型并给出图表建议 + 统计量。

    返回 ``{type, type_label, chart, chart_label, stats, note, bins, xrange}``。
    ``bins`` / ``xrange`` 是**窄分布**时的建议（分箱宽度、建议放大范围），
    前端可以用它做局部放大 —— **不修改任何原始数据**。
    """
    ys = _clean(y)
    xs = np.asarray(x, dtype=float).ravel() if x is not None else np.array([])
    st = stats_of(ys)
    out = {"type": "continuous", "type_label": TYPES["continuous"],
           "chart": "line", "chart_label": CHARTS["line"],
           "stats": st, "note": "", "bins": None, "xrange": None}

    if not ys.size:
        out.update(type="invalid", type_label=TYPES["invalid"],
                   chart="card", chart_label=CHARTS["card"],
                   note="没有有效数据点（全是 NaN/inf 或空）")
        return out

    # ---- 恒定值
    if st["max"] is not None and st["max"] - st["min"] <= 1e-12 * max(1.0, abs(st["max"])):
        out.update(type="constant", type_label=TYPES["constant"],
                   chart="card", chart_label=CHARTS["card"],
                   note=f"所有点都等于 {st['mean']:.6g}（恒定值，画成卡片更清楚）")
        return out

    # ---- 单点
    if st["count"] == 1:
        out.update(type="constant", type_label=TYPES["constant"],
                   chart="card", chart_label=CHARTS["card"],
                   note=f"只有 1 个数据点：{st['mean']:.6g}")
        return out

    # ---- 周期型（角度）
    if _is_periodic(ys, label) or _is_periodic(ys, xlabel):
        out.update(type="periodic", type_label=TYPES["periodic"],
                   chart="angle", chart_label=CHARTS["angle"],
                   note="角度类数据：按周期区间统计（边界处会平滑，不做人为外推）")
        return out

    span = float(st["max"] - st["min"])
    rel = float(st["std"] / span) if span > 0 else 0.0

    # ---- 时间序列（x 是时间）：画折线
    is_time = ("时间" in str(xlabel)) or ("time" in str(xlabel).lower())
    if is_time:
        out.update(chart="line", chart_label=CHARTS["line"])
        if rel <= _CONCENTRATED:
            out.update(type="concentrated", type_label=TYPES["concentrated"],
                       note="数值高度集中：时间曲线几乎是平的，可看右侧统计量卡片；"
                            "纵轴已按数据自适应，未做任何平滑或加噪")
        return out

    # ---- 离散型
    if _is_discrete(ys):
        out.update(type="discrete", type_label=TYPES["discrete"],
                   chart="bar", chart_label=CHARTS["bar"],
                   note="取值种类少且为整数：按频数画柱状图")
        return out

    # ---- 连续型；再看是不是"高度集中"（窄分布）
    if span > 0 and rel <= _CONCENTRATED:
        # 用 Freedman–Diaconis 给个分箱宽度建议（只为显示，不动数据）
        iqr = float(np.subtract(*np.percentile(ys, [75, 25])))
        if iqr > 0:
            bins = max(5, min(60, int(math.ceil(span / (2 * iqr / np.cbrt(ys.size))))))
        else:
            bins = max(5, min(60, int(math.sqrt(ys.size))))
        w = span / bins
        pad = max(span * 0.5, abs(st["mean"]) * 0.02)
        out.update(type="concentrated", type_label=TYPES["concentrated"],
                   chart="hist", chart_label=CHARTS["hist"],
                   bins=bins,
                   xrange=[st["mean"] - pad - w, st["mean"] + pad + w],
                   note=("数值高度集中（标准差/极差 = "
                         f"{rel:.3%}）：建议按 {bins} 个窄分箱看分布，"
                         "或只画单点/数值卡片；坐标范围建议 "
                         f"[{st['mean'] - pad:.4g}, {st['mean'] + pad:.4g}]"))
        return out

    out.update(type="continuous", type_label=TYPES["continuous"],
               chart="hist", chart_label=CHARTS["hist"],
               note="连续型数据：直方图看分布，折线看趋势")
    return out


def annotate_curves(result) -> None:
    """给一个 :class:`mdta.core.AnalysisResult` 的每条曲线补上 ``viz`` 提示。

    只在**曲线自己还不知道**的时候补（分析内部若显式给了 ``kind`` 就尊重它）。
    """
    for c in getattr(result, "curves", []) or []:
        try:
            panel = result.panel(int(getattr(c, "panel", 0) or 0))
            xlabel = str(getattr(panel, "xlabel", "") or "")
        except Exception:                                        # noqa: BLE001
            xlabel = ""
        viz = classify(getattr(c, "x", None), getattr(c, "y", None),
                       label=str(getattr(c, "label", "") or ""), xlabel=xlabel)
        c.viz = viz
