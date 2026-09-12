# -*- coding: utf-8 -*-
"""单位与坐标轴辅助函数。

工具内部统一使用 MDAnalysis / GROMACS 原生单位：

- 长度：Å
- 时间：ps
- 质量：amu
- 电荷：e

只在绘图和导出时做可读性转换。
"""

from __future__ import annotations

import numpy as np

#: 1 amu/Å³ 折算成 g/cm³
AMU_PER_A3_TO_G_CM3 = 1.66053906660

#: 玻尔兹曼常数，kJ/(mol·K)
KB_KJ_MOL_K = 0.008314462618


def time_axis(times_ps: np.ndarray) -> tuple[np.ndarray, str]:
    """自动选择 ps / ns 作为时间轴单位，返回 ``(数值, 轴标签)``。"""
    t = np.asarray(times_ps, dtype=float)
    if t.size == 0:
        return t, "Time (ps)"
    span = float(np.nanmax(t) - np.nanmin(t))
    if span >= 1000.0:
        return t / 1000.0, "时间 Time (ns)"
    return t, "时间 Time (ps)"


def mass_density_to_g_cm3(density_amu_per_a3: np.ndarray) -> np.ndarray:
    return np.asarray(density_amu_per_a3, dtype=float) * AMU_PER_A3_TO_G_CM3


def bin_edges_to_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def auto_bins(x: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """给定样本自动生成 bin 区间。"""
    x = np.asarray(x, dtype=float)
    if nbins is None:
        nbins = int(np.clip(np.sqrt(max(x.size, 1)) * 2, 10, 100))
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(0.0, 1.0, nbins + 1)
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, int(nbins) + 1)
