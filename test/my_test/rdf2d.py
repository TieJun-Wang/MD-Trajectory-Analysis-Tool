# -*- coding: utf-8 -*-
r"""面内径向分布 RDF (2D)（工具函数：``mdta.analysis.interface.analyze_rdf_inplane``）。

**做什么**：把原子**投影到膜/片层平面**上，只统计平面内距离的二维径向分布
``g₂(r)``（分子在膜里横向怎么排布）；还可以指定"叶层"（slab）只看上半层/下半层。

**为什么单拎出来**：三维 g(r) 对层状体系会把"层内近邻"和"跨层近邻"混在一起，
看不出膜内的六方/无定形排布；把 z 分量丢掉后峰形才干净。

**引擎参数**：

- ``plane_axis``：投影掉哪个轴（2 = c 轴 → 看 ab 面，默认）；
- ``rmax`` / ``nbins``：面内统计半径与分箱数；
- ``slab``：叶层范围 ``(z_lo, z_hi)``，留空 = 全盒。

**前提**：需要周期盒子（面内距离要按最小镜像算）。

**输出**：``rdf2d__inplane.csv``（g₂(r)，可能每组分一条）。

**单独跑**：``python test/my_test/rdf2d.py``（多组分体系更合适：
``MDTA_MYTEST_SYSTEM=mix python test/my_test/rdf2d.py``）
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "rdf2d"
TITLE = "面内径向分布 RDF (2D)"
SUB = "把原子投影到平面，只统计面内距离的 g₂(r)：看膜内横向怎么排布"


def test_rdf2d(eng, *, plane_axis=2, rmax=70.0, nbins=140, slab_lo=None,
               slab_hi=None, outdir=None, formats=("csv",), **over):
    """跑工具的面内 RDF 分析，打印统计量并导出 CSV。"""
    params = {"plane_axis": int(plane_axis), "rmax": float(rmax),
              "nbins": int(nbins)}
    if slab_lo is not None and slab_hi is not None:
        params["slab"] = (float(slab_lo), float(slab_hi))
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("plane_axis", "投影轴（法向）", "choice", "2",
      choices=[("2", "c (2)"), ("1", "b (1)"), ("0", "a (0)")],
      help="投影掉这个轴 → 统计剩下两个方向的平面内距离"),
    Q("rmax", "面内最大 r (Å)", "float", 70.0, lo=1),
    Q("nbins", "分箱数", "int", 140, lo=10),
    Q("slab_lo", "叶层 z 下限 (Å)", "float", None,
      help="与下面的上限一起填才生效；留空 = 全盒"),
    Q("slab_hi", "叶层 z 上限 (Å)", "float", None),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_rdf2d, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
