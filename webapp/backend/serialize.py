# -*- coding: utf-8 -*-
"""把 :class:`mdta.core.AnalysisResult` 序列化成前端可用的 JSON。

分工
----
后端只负责"给数据"，不负责画图：每个结果被拆成

- ``panels``  —— 每个面板的标题、坐标轴名称、坐标尺度；
- ``curves``  —— 每条曲线的 ``x`` / ``y`` 数组与画法（line / step / bar / scatter）；
- ``summary`` —— 标量统计量；
- ``notes``   —— 算法的文字说明。

前端用 ECharts 把这些数组画出来，所以图表的缩放 / 平移 / hover / 图例开关
都由前端负责，后端不需要生成任何图片。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

#: 单条曲线返回给前端的最大点数（超过则等间隔抽稀，并标记 ``downsampled``）
MAX_POINTS = 4000


def _num(v: Any) -> Any:
    """把 numpy 标量转成原生 Python 类型（NaN / Inf 转成 None）。"""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_num(x) for x in v.tolist()]
    return v


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS):
    """曲线太长时等间隔抽稀（并保留末点），避免 JSON 过大拖慢前端。"""
    n = int(x.size)
    if n <= max_points:
        return x, y, False
    idx = np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))
    return x[idx], y[idx], True


def curve_to_json(curve, max_points: int = MAX_POINTS) -> dict:
    x, y, ds = _downsample(np.asarray(curve.x, dtype=float),
                           np.asarray(curve.y, dtype=float), max_points)
    return {
        "label": str(curve.label),
        "kind": str(curve.kind),          # line | step | bar | scatter
        "panel": int(curve.panel),
        "x": [_num(v) for v in x.tolist()],
        "y": [_num(v) for v in y.tolist()],
        "n": int(curve.n),
        "downsampled": bool(ds),
    }


def panel_to_json(result, index: int) -> dict:
    p = result.panel(index)
    return {
        "index": int(index),
        "title": p.title or "",
        "xlabel": p.xlabel or "",
        "ylabel": p.ylabel or "",
        "xscale": p.xscale or "linear",
        "yscale": p.yscale or "linear",
        "legend": bool(p.legend),
        "curves": [str(c.label) for c in result.curves_of(index)],
    }


def result_to_json(result, max_points: int = MAX_POINTS) -> dict:
    n_panels = max(result.panel_count, 1)
    return {
        "name": result.name,
        "title": result.title,
        "n_panels": n_panels,
        "panels": [panel_to_json(result, i) for i in range(n_panels)],
        "curves": [curve_to_json(c, max_points) for c in result.curves],
        "summary": {str(k): _num(v) for k, v in result.summary.items()},
        "notes": [str(x) for x in result.notes],
        "meta": {str(k): _num(v) for k, v in (result.meta or {}).items()
                 if not isinstance(v, (list, dict, tuple))},
    }


def results_to_json(results: Mapping[str, Any], max_points: int = MAX_POINTS) -> dict:
    """``{分析名: AnalysisResult}`` -> ``{分析名: {...}}``（保持顺序）。"""
    return {name: result_to_json(res, max_points) for name, res in results.items()}


def summary_rows(results_json: Mapping[str, dict]) -> list[dict]:
    """汇总表：一行一个分析。"""
    rows = []
    for name, r in results_json.items():
        rows.append({
            "name": name,
            "title": r.get("title", ""),
            "n_panels": r.get("n_panels", 0),
            "n_curves": len(r.get("curves", [])),
            "n_stats": len(r.get("summary", {})),
        })
    return rows
