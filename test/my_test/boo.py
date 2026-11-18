# -*- coding: utf-8 -*-
r"""键取向序参数 BOO（工具函数：``mdta.analysis.boo.analyze_boo``）。

**做什么**：Steinhardt 键取向序 —— 对每个原子的近邻做球谐展开求 ``q_l`` /
``q̄_l``（先对邻居平均再求模，更稳），``l = 4, 6``。

- 固相/结晶：q̄₆ 明显偏高（0.4 以上）且分布窄；液相/无定形：q̄₆ 低（0.2–0.3）、
  分布宽。20PE30 实测 q̄₆ ≈ 0.26 —— 典型无定形链堆积。
- 这是取代"晶体/非晶区域识别"的正式判据（那一项已在 1.0.8 下线）。

**引擎参数**：``cutoff``（近邻半径，留空 = 按第一配位层自动估）、
``averaged``（用 q̄_l 而不是 q_l，默认 True）、``l_list``（默认 4、6）。

**输出**：``boo__qbar6.csv``（q̄₆ 随时间）、``boo__qbar6_dist.csv``（q̄₆ 分布）、
``boo__q4q6.csv``（q̄₄–q̄₆ 平面散点）。

**单独跑**：``python test/my_test/boo.py --frames 20 -y``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "boo"
TITLE = "键取向序参数 BOO"
SUB = "q̄₆ 高（>0.4）且分布窄 = 固相/结晶；低且宽 = 液相/无定形"


def test_boo(eng, *, cutoff=None, averaged=True, outdir=None,
             formats=("csv",), **over):
    """跑工具的 BOO 分析，打印统计量并导出 CSV。"""
    params = {"averaged": bool(averaged)}
    if cutoff:
        params["cutoff"] = float(cutoff)
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("cutoff", "近邻半径 (Å，留空 = 自动)", "float", None, lo=0,
      help="留空 = 按第一配位层自动估"),
    Q("averaged", "用平均版 q̄_l", "bool", True,
      help="推荐勾上：先用邻居平均再求模，区分固相/液相更稳"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_boo, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
