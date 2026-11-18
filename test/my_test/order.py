# -*- coding: utf-8 -*-
r"""结构有序度分析（工具函数：``mdta.analysis.crystallinity.analyze_structural_order``）。

**做什么**：把三路证据合成一个 0–1 的**有序度指数**，并给出各分量随时间的曲线：

1. **二面角分量**：``trans`` 构象（|φ| ≈ 180°）的比例 —— 全反式 = 结晶/伸展；
2. **取向分量**：链段取向参数 S（与「链段取向分析」同一算法，
   ``orient_mode`` / ``stride`` / ``orient_weighting`` 与那边共享设置）；
3. **局部结构分量**：最近邻 RDF 第一峰高度 ``g_ref``（留空则不计入）。

三个分量按 ``weights`` 加权合成总有序度。想要"一图看结晶趋势"就看这一项；
想要"为什么高"就看各分量。

**引擎参数**：``gauche_edge``（trans 判定阈值，默认 120°）、``g_ref``（局部结构
参考峰高，留空 = 不计入）、``weights``（三分量权重，默认等权）。

**输出**：``order__components.csv``（各分量随时间）、``order__index.csv``（总指数）。

**单独跑**：``python test/my_test/order.py --frames 50 -y``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "order"
TITLE = "结构有序度分析"
SUB = "二面角 trans 比例 + 链段取向 S + 局部结构 g_ref → 0–1 的有序度指数"


def test_order(eng, *, gauche_edge=120.0, g_ref=None, outdir=None,
               formats=("csv",), **over):
    """跑工具的结构有序度分析，打印统计量并导出 CSV。

    链段取向那一路的链段定义/权重与「链段取向分析」共享（引擎 ``_shared_params``
    会把那边的设置透传过来），所以先跑过取向再跑这一项，两者数字能对上。
    """
    params = {"gauche_edge": float(gauche_edge)}
    if g_ref is not None:
        params["g_ref"] = float(g_ref)
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("gauche_edge", "trans/gauche 阈值 (°)", "float", 120.0, lo=60, hi=180,
      help="|φ| ≤ 阈值记 gauche±，其余记 trans（与「二面角分析」同一约定）"),
    Q("g_ref", "局部结构 g_ref（留空 = 不计入）", "float", None, lo=0,
      help="最近邻 RDF 第一峰高度；留空表示这一分量不参与合成"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_order, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
