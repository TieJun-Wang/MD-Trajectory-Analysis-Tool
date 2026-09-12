# -*- coding: utf-8 -*-
"""分析结果的数据容器。

所有分析模块统一返回 :class:`AnalysisResult`，从而让可视化模块、导出模块、
命令行界面和图形界面可以完全通用地处理任意分析结果：

- 一个结果由若干"面板"(:class:`Panel`) 组成，每个面板是一张子图；
- 每个面板内可以画若干条曲线(:class:`Curve`)；
- 标量统计量放在 ``summary`` 中，导出到 ``summary.xlsx``。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class Panel:
    """一个绘图面板（子图）的元信息。"""

    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    xscale: str = "linear"
    yscale: str = "linear"
    legend: bool = True


@dataclass
class Curve:
    """一条待绘制的曲线。"""

    label: str
    x: np.ndarray
    y: np.ndarray
    kind: str = "line"  # line | step | bar | scatter
    panel: int = 0

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"曲线 {self.label!r} 的 x/y 长度不一致: {self.x.shape} vs {self.y.shape}"
            )

    @property
    def n(self) -> int:
        return int(self.x.size)


@dataclass
class AnalysisResult:
    """一次分析的全部输出。"""

    name: str
    title: str
    panels: list[Panel] = field(default_factory=list)
    curves: list[Curve] = field(default_factory=list)
    summary: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 构造
    def add_curve(
        self,
        label: str,
        x: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
        kind: str = "line",
        panel: int = 0,
    ) -> Curve:
        curve = Curve(label=label, x=np.asarray(x, float), y=np.asarray(y, float),
                      kind=kind, panel=panel)
        self.curves.append(curve)
        return curve

    def add_summary(self, key: str, value: Any) -> None:
        self.summary[key] = value

    def add_notes(self, *lines: str) -> None:
        self.notes.extend(str(x) for x in lines)

    # ---------------------------------------------------------------- 查询
    @property
    def panel_count(self) -> int:
        return max((c.panel for c in self.curves), default=-1) + 1

    def curves_of(self, panel: int) -> list[Curve]:
        return [c for c in self.curves if c.panel == panel]

    def panel(self, i: int) -> Panel:
        while len(self.panels) <= i:
            self.panels.append(Panel())
        return self.panels[i]

    # ---------------------------------------------------------------- 导出
    def dataframes(self) -> "OrderedDict[str, pd.DataFrame]":
        """把结果拆成若干 DataFrame（每个面板一张表）。

        面板内所有曲线共享同一 x 轴时导出为宽表（x 列 + 每条曲线一列），
        否则导出为长表（curve, x, y）。
        """
        out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        multi = self.panel_count > 1
        for p in range(self.panel_count):
            cs = self.curves_of(p)
            if not cs:
                continue
            key = f"{self.name}__panel{p}" if multi else self.name
            xs = [c.x for c in cs]
            shared = all(x.shape == xs[0].shape and np.array_equal(x, xs[0]) for x in xs)
            if shared:
                df = pd.DataFrame({self.panel(p).xlabel or "x": xs[0]})
                for c in cs:
                    df[c.label] = c.y
            else:
                frames = [
                    pd.DataFrame(
                        {
                            "curve": c.label,
                            "x": c.x,
                            "y": c.y,
                        }
                    )
                    for c in cs
                ]
                df = pd.concat(frames, ignore_index=True)
            out[key] = df
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """主表（第一个面板）。"""
        dfs = self.dataframes()
        return next(iter(dfs.values())) if dfs else pd.DataFrame()

    # ---------------------------------------------------------------- 绘图
    def plot(self, figsize: tuple[float, float] | None = None):
        """绘制结果，返回 ``matplotlib.figure.Figure``。"""
        from .plotting import plot_result

        return plot_result(self, figsize=figsize)

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试
        return (
            f"AnalysisResult(name={self.name!r}, curves={len(self.curves)}, "
            f"panels={self.panel_count}, summary_keys={list(self.summary)[:6]})"
        )


# -------------------------------------------------------------------- 工具
def describe_array(y: Sequence[float] | np.ndarray, prefix: str = "") -> "OrderedDict[str, float]":
    """数组的常用统计量：平均、标准差、标准误、极值、中位数。"""
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    d: "OrderedDict[str, float]" = OrderedDict()
    if a.size == 0:
        for k in ("mean", "std", "sem", "min", "max", "median"):
            d[f"{prefix}{k}"] = float("nan")
        d[f"{prefix}n"] = 0
        return d
    d[f"{prefix}mean"] = float(a.mean())
    d[f"{prefix}std"] = float(a.std(ddof=1)) if a.size > 1 else 0.0
    d[f"{prefix}sem"] = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    d[f"{prefix}min"] = float(a.min())
    d[f"{prefix}max"] = float(a.max())
    d[f"{prefix}median"] = float(np.median(a))
    d[f"{prefix}n"] = int(a.size)
    return d


def block_average(y: Sequence[float] | np.ndarray, n_blocks: int = 5) -> tuple[np.ndarray, float]:
    """块平均法估计统计误差。

    返回 ``(各块平均值, 块平均标准误)``。MD 数据存在时间相关性，
    块平均给出的误差棒比普通标准误更接近真实值。
    """
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2 or n_blocks < 2:
        return a, float("nan")
    n_blocks = min(int(n_blocks), a.size)
    blocks = np.array_split(a, n_blocks)
    means = np.array([b.mean() for b in blocks if b.size])
    if means.size < 2:
        return means, float("nan")
    return means, float(means.std(ddof=1) / np.sqrt(means.size))


def wrap_angle_deg(a: Sequence[float] | np.ndarray) -> np.ndarray:
    """把角度折算到 (-180, 180] 度。"""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0
