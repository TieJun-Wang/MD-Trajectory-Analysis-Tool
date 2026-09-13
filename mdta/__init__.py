# -*- coding: utf-8 -*-
"""MD Trajectory Analysis Tool (MDT)

面向高分子材料研究的分子动力学轨迹分析工具。

核心流程::

    Trajectory -> Selection -> Analysis -> Visualization -> Export

主要子模块
----------
- :mod:`mdta.io`          文件读取模块
- :mod:`mdta.systeminfo`  体系信息模块
- :mod:`mdta.selection`   原子/分子/链选择模块
- :mod:`mdta.preprocess`  轨迹预处理模块（PBC、抽帧、平衡段）
- :mod:`mdta.analysis`    结构与性质分析模块
- :mod:`mdta.plotting`    数据可视化模块
- :mod:`mdta.export`      数据导出模块
- :mod:`mdta.pipeline`    分析流程编排 / 批量分析
"""

from .core import AnalysisResult, Curve, Panel
from .io import MDTrajectory, load_trajectory
from .systeminfo import SystemInfo, describe_system, format_system_info

__version__ = "1.0.2"
__all__ = [
    "AnalysisResult",
    "Curve",
    "Panel",
    "MDTrajectory",
    "load_trajectory",
    "SystemInfo",
    "describe_system",
    "format_system_info",
    "__version__",
]
