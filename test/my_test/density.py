# -*- coding: utf-8 -*-
r"""密度分布（工具函数：``mdta.analysis.interface.analyze_density``）。

**做什么**：把盒子沿某一个方向切成 N 个薄片，统计每个薄片里的质量（或原子数），
再除以薄片体积 → 沿该方向的密度分布 ``ρ(z)``。均相体系应当是一条平线，
分层/界面体系会出现台阶（这正是判断"有没有分层"的依据）。

**方向**（引擎参数 ``axis``，界面「密度方向」）：``"all"`` = 三个方向都算
（默认，一张图三条曲线）；``2/1/0`` = 只算某一个晶轴方向。

**口径**（``mode``）：``mass`` 质量密度（g/cm³）/ ``number`` 数密度（1/Å³）。

**前提**：必须有周期盒子，否则跳过（引擎抛 ``BoxMissing``，脚本会打印原因）。

**输出**：``density__profile.csv``（各方向 ρ 曲线）、``density__norm.csv``（归一化）。

**单独跑**：``python test/my_test/density.py``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "density"
TITLE = "密度分布"
SUB = "沿某个方向切薄片算密度 ρ(z)：均相体系是平线，分层体系出现台阶"


def test_density(eng, *, axis="all", mode="mass", nbins=None, outdir=None,
                 formats=("csv",), **over):
    """跑工具的密度分布分析，打印统计量并导出 CSV。"""
    axis = str(axis)
    params = {"axis": axis if axis == "all" else int(axis), "mode": str(mode)}
    if nbins:
        params["nbins"] = int(nbins)
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("axis", "密度方向", "choice", "all",
      choices=[("all", "三个方向（默认）"), ("2", "C001（c 轴）"),
               ("1", "C010（b 轴）"), ("0", "C100（a 轴）")],
      help="all = 三个方向都算，一张图三条曲线"),
    Q("mode", "密度口径", "choice", "mass",
      choices=[("mass", "质量密度 (g/cm³)"), ("number", "数密度 (1/Å³)")]),
    Q("nbins", "分箱数（留空 = 自动）", "int", None, lo=1),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_density, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
