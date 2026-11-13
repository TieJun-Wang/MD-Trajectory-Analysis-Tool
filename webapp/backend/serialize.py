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
import os
from typing import Any, Mapping

import numpy as np

from mdta.qc import checks_digest


def _max_points_from_env() -> int | None:
    """单条曲线返回给前端的点数上限；``0``/``空``/``off`` = **不抽稀**（默认）。

    为什么默认改成不抽稀：用户要看**完整分辨率**的曲线与数据表 —— 抽稀只发生在
    展示层，数值（D、误差棒、QC）本来就用全量数据算，展示时再丢 1/3 的点会让人
    怀疑"是不是少算了"。代价是 JSON 更大、前端渲染更慢，需要时可以调回去：

        set MDTA_MAX_POINTS=4000      # 恢复等间隔抽稀（保留末点）
        set MDTA_MAX_POINTS=0         # （默认）原样返回全部点
    """
    raw = str(os.environ.get("MDTA_MAX_POINTS", "0")).strip().lower()
    if raw in ("", "0", "none", "off", "false", "all"):
        return None
    try:
        return max(2, int(float(raw)))
    except ValueError:
        return None


#: 单条曲线返回给前端的最大点数；``None`` = 不抽稀（默认，见上面的环境变量说明）
MAX_POINTS: int | None = _max_points_from_env()


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


def _downsample(x: np.ndarray, y: np.ndarray,
                max_points: int | None = MAX_POINTS):
    """曲线太长时等间隔抽稀（并保留末点）。

    ``max_points`` 为 ``None`` / ``0`` → **原样返回**（不抽稀，当前默认行为）。
    """
    n = int(x.size)
    if not max_points or n <= max_points:
        return x, y, False
    idx = np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))
    return x[idx], y[idx], True


def _viz_json(viz):
    """把自适应可视化的判定转成纯 JSON（numpy 标量/数组都转掉）。"""
    if not isinstance(viz, dict):
        return None
    out = {}
    for k, v in viz.items():
        if isinstance(v, dict):
            out[str(k)] = {str(a): _num(b) for a, b in v.items()}
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [_num(x) for x in v]
        elif isinstance(v, (str, bool)) or v is None:
            out[str(k)] = v
        elif isinstance(v, int):
            out[str(k)] = int(v)
        else:
            out[str(k)] = _num(v)
    return out


def curve_to_json(curve, max_points: int | None = MAX_POINTS) -> dict:
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
        # 所属分析对象（空 = 单系列）：前端据此"默认只显示第一个对象"
        "group": str(getattr(curve, "group", "") or ""),
        # 自适应可视化引擎的判定（数据类型 / 推荐图型 / 统计量 / 窄分布建议）
        "viz": _viz_json(getattr(curve, "viz", None)),
    }


def _heat_json(h) -> dict:
    """二维热力图 → JSON。``z`` 是 ``len(y) x len(x)`` 的二维表。"""
    h = dict(h or {})
    z = h.get("z") or []
    out = {
        "panel": int(h.get("panel") or 0),
        "title": str(h.get("title") or ""),
        "xlabel": str(h.get("xlabel") or ""),
        "ylabel": str(h.get("ylabel") or ""),
        "note": str(h.get("note") or ""),
        "x": [_num(v) for v in (h.get("x") or [])],
        "y": [_num(v) for v in (h.get("y") or [])],
        "z": [[_num(v) for v in row] for row in z],
    }
    for k in ("vmin", "vmax"):
        if h.get(k) is not None:
            out[k] = _num(h[k])
    return out


def panel_to_json(result, index: int) -> dict:
    p = result.panel(index)
    from mdta.naming import panel_slug

    return {
        "index": int(index),
        "title": p.title or "",
        #: 下载文件名用的**短英文** token（纯 ASCII）——图表区「保存图片」、数据表区
        #: 「下载 CSV」、导出板块三处共用，避免前端再抄一份命名规则
        "slug": panel_slug(getattr(result, "name", ""), index, p.title or ""),
        "xlabel": p.xlabel or "",
        "ylabel": p.ylabel or "",
        "xscale": p.xscale or "linear",
        "yscale": p.yscale or "linear",
        "legend": bool(p.legend),
        # 极坐标图（角度分布）/ 二维热力图（接触矩阵、空间热力图）
        "polar": bool(getattr(p, "polar", False)),
        "heatmap": bool(getattr(p, "heatmap", False)),
        "curves": [str(c.label) for c in result.curves_of(index)],
    }


def result_to_json(result, max_points: int | None = MAX_POINTS) -> dict:
    n_panels = max(result.panel_count, 1)
    return {
        "name": result.name,
        "title": result.title,
        "n_panels": n_panels,
        "panels": [panel_to_json(result, i) for i in range(n_panels)],
        "curves": [curve_to_json(c, max_points) for c in result.curves],
        # 二维热力图（接触矩阵 / 空间热力图）：与曲线并行的一条通道
        "heatmaps": [_heat_json(h) for h in
                     (getattr(result, "heatmaps", None) or [])],
        "summary": {str(k): _num(v) for k, v in result.summary.items()},
        "notes": [str(x) for x in result.notes],
        # 每条说明的作用域（属于哪张图 / 哪条曲线）——前端据此只显示与当前
        # 图表可见曲线相关的说明
        "note_meta": result.notes_with_scope(),
        # 科研 QC 检查项（名称/级别/结论/依据）+ 总览：前端「科研 QC」页直接用
        "checks": [dict(c) for c in (getattr(result, "checks", None) or [])],
        "checks_digest": checks_digest(getattr(result, "checks", None) or []),
        "meta": {str(k): _num(v) for k, v in (result.meta or {}).items()
                 if not isinstance(v, (list, dict, tuple))},
    }


def results_to_json(results: Mapping[str, Any], max_points: int | None = MAX_POINTS) -> dict:
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
