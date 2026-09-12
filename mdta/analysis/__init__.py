# -*- coding: utf-8 -*-
"""结构与性质分析模块。

子模块
------
- :mod:`~mdta.analysis.conformation`   链构象分析（Rg、端到端距离、二面角）
- :mod:`~mdta.analysis.interface`     界面相容性分析（密度分布、RDF、接触、界面宽度）
- :mod:`~mdta.analysis.crystallinity` 结晶行为分析（链段取向、结构有序度）
- :mod:`~mdta.analysis.dynamics`      动力学辅助分析（MSD、扩散系数）
"""

from .base import (
    ANALYSIS_REGISTRY,
    frame_iterator,
    positions_for,
    register,
    run_analysis,
)
from . import conformation, crystallinity, dynamics, interface  # noqa: F401

__all__ = [
    "frame_iterator",
    "positions_for",
    "register",
    "run_analysis",
    "ANALYSIS_REGISTRY",
    "conformation",
    "interface",
    "crystallinity",
    "dynamics",
]
