# -*- coding: utf-8 -*-
r"""径向分布函数 RDF（工具函数：``mdta.analysis.interface.analyze_rdf``）。

**做什么**：统计"以 A 组原子为中心，距离 r 处找到 B 组原子的概率"相对理想气体的
偏离 ``g(r) = (dn / (4πr²dr)) / ρ₀``。第一个峰的峰位 = 最近邻距离，峰高 = 结构
有序程度，是判断"两种物质是互溶还是分相"最直接的曲线。

**配对口径**（引擎参数 ``mode``）：

- ``total``（默认）：A–A + A–B + B–B 全算（只看整体结构）；
- ``inter``：只看**跨分子**的原子对（排除同一分子内的近邻键）；
- ``intra``：只看分子内的原子对；
- ``all``：三者同时给出（三条曲线对比）。

归一化按**有序原子对**（同种粒子 N−1）—— 与 MDAnalysis ``InterRDF`` 的 N²
差一个 ``(N−1)/N``，`test/mda_test/rdf_mdat.py` 里有逐位比对。

**前提**：需要 ≥2 个组分（或同组自相关）与周期盒子；界面宽度/质心距这类
"两组分之间"的量在单组分体系会跳过，RDF 不会。

**输出**：``rdf__rdf.csv``（g(r)）、``rdf__halves.csv``（前后半段对比，开了才有）。

**单独跑**：``python test/my_test/rdf.py``

**多组分体系**：``MDTA_MYTEST_SYSTEM=mix python test/my_test/rdf.py --frames 20``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "rdf"
TITLE = "径向分布函数 RDF"
SUB = "g(r)：第一个峰位 = 最近邻距离，峰高 = 有序程度（互溶 / 分相的判断依据）"


def test_rdf(eng, *, rmax=12.0, nbins=120, mode="total", outdir=None,
             formats=("csv",), **over):
    """跑工具的 RDF 分析，打印统计量并导出 CSV。"""
    params = {"rmax": float(rmax), "nbins": int(nbins), "mode": str(mode)}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("rmax", "最大 r (Å)", "float", 12.0, lo=1,
      help="统计到多远；一般取到 g(r) 已经平到 1 的位置"),
    Q("nbins", "分箱数", "int", 120, lo=1, help="步长 = rmax / 分箱数"),
    Q("mode", "配对模式", "choice", "total",
      choices=[("total", "总体（默认）"), ("inter", "分子间"),
               ("intra", "分子内"), ("all", "三者对比（三条曲线）")]),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_rdf, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
