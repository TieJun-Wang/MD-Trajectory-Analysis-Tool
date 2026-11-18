# -*- coding: utf-8 -*-
r"""接触分析（工具函数：``mdta.analysis.interface.analyze_contacts``）。

**做什么**：逐帧统计两组原子之间**距离 < cutoff** 的原子对，给出

- 接触对数随时间（界面面积的代理量，越大 = 两相接触越充分）；
- 最小原子间距随时间（判断有没有"插到一起"、有没有异常重叠）；
- 接触对数分布、原子接触概率（哪些原子最常参与接触）；
- 可选：按原子对累计出现的**接触矩阵 / 占据率**（谁和谁总贴着）。

**配对口径**（引擎参数 ``mode``）：

- ``inter``（默认）：只看**跨分子**的接触（A、B 两组若是同一种分子，就是不同分子之间）；
- ``intra``：只看分子内部；
- ``total``：全算。

**A/B 怎么选**（``pair=(A, B)``，界面「接触 A 组 / B 组」）：留空时引擎按
"A = 最大组分，B = 与 A 不同分子的最大组分"挑；只有一个组分时退回**同组 A–A 自接触**
（无序对，不含 i==i）。选组规则与「未参与的是 …」提示都由引擎给出，脚本原样打印。

**性能提示**：邻居搜索用 CSR 网格（`mdta.neighbors`），20PE30 / 3640 原子实测
约 26 ms/帧（100 帧 ≈ 3 s），大体系先用 ``--frames 20`` 试。

**输出**：``contact__count.csv``、``contact__mindist.csv``、
``contact__occupancy.csv``、``contact__pairs.csv`` 等（多面板）。

**单独跑**：``python test/my_test/contact.py --frames 50 --cutoff 4.5 -y``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "contact"
TITLE = "接触分析"
SUB = "两组原子距离 < cutoff 的原子对：接触对数 = 界面面积代理量，最小间距看有没有重叠"


def test_contact(eng, *, cutoff=5.0, mode="inter", top_n=0, pair=None,
                 outdir=None, formats=("csv",), **over):
    """跑工具的接触分析，打印统计量并导出 CSV。"""
    params = {"cutoff": float(cutoff), "mode": str(mode), "pair": pair}
    if top_n:
        params["top_n"] = int(top_n)
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("cutoff", "cutoff (Å)", "float", 5.0, lo=0.1,
      help="小于这个距离算一对接触；界面通常取 3.5–5 Å"),
    Q("mode", "配对模式", "choice", "inter",
      choices=[("inter", "分子间（默认）"), ("intra", "分子内"),
               ("total", "总体（含分子内）")]),
    Q("pair_a", "A 组（留空 = 最大组分）", "text", ""),
    Q("pair_b", "B 组（留空 = 不同分子的最大组分）", "text", ""),
    Q("top_n", "只留前 N 对原子（0 = 全部）", "int", 0, lo=0,
      help="占据率/接触矩阵很大时用它限量"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_contact, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
