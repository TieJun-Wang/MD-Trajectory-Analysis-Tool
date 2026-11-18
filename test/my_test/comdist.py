# -*- coding: utf-8 -*-
r"""两组分质心距（工具函数：``mdta.analysis.interface.analyze_com_distance``）。

**做什么**：逐帧算两个组分**质心**之间的距离（默认按最小镜像算，否则质心分处
盒子两侧时会得到 ~L 的假跳变），给出时间序列与分布。

**A/B 怎么选**（引擎参数 ``pair=(A, B)``；界面「质心距 A 组 / B 组」）：
留空 = 自动取原子数最多的两个组分；只有一个组分时该项**跳过并说明原因**
（质心距是"两组分之间"的量，单组分没有意义）。

**输出**：``comdist__dist.csv``（质心距随时间）、``comdist__axis.csv``（沿某轴的
分量）。

**单独跑**：``python test/my_test/comdist.py``

**多组分体系**：``MDTA_MYTEST_SYSTEM=mix python test/my_test/comdist.py --frames 20``
（那里会自动识别 polymer / water / ion，A=最大组分，B=次大且≠A 的组分）
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "comdist"
TITLE = "两组分质心距"
SUB = "逐帧算两个组分质心之间的距离（最小镜像），看两相是靠近还是分离"


def test_comdist(eng, *, axis=2, pbc=True, pair=None, outdir=None,
                 formats=("csv",), **over):
    """跑工具的质心距分析，打印统计量并导出 CSV。"""
    params = {"axis": int(axis), "pbc": bool(pbc), "pair": pair}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("axis", "分量方向", "choice", "2",
      choices=[("2", "c (2)"), ("1", "b (1)"), ("0", "a (0)")]),
    Q("pbc", "用最小镜像", "bool", True,
      help="默认勾上：两个质心常分处盒边界两侧，直接相减会得到 ~L 的假跳变"),
    Q("pair_a", "A 组（留空 = 原子数最多）", "text", ""),
    Q("pair_b", "B 组（留空 = 次多且≠A）", "text", ""),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_comdist, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
