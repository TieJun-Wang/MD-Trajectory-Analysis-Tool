# -*- coding: utf-8 -*-
r"""链段取向分析（工具函数：``mdta.analysis.crystallinity.analyze_orientation``）。

**做什么**：把每条链切成**链段**（相邻两个重复单元的连线），算链段方向与参考轴
的夹角，给出

- **取向参数 S** 随时间（``S = (3⟨cos²θ⟩ − 1)/2``，0 = 各向同性，1 = 完全对齐）；
- **P₂(cos θ)** 随时间（按 x/y/z 三个参考轴分别算）；
- 链段方向的**取向分布**（球面/极角分布）；
- ⟨cos²θ⟩ 与平均夹角 θ。

**链段定义**（引擎参数 ``mode``，界面「取向链段定义」）：

- ``repeat``（默认）：**化学重复单元**（PE 的 C–C 主链单元）—— 与分子拓扑一致，
  拉伸取向时最灵敏；
- ``backbone``：几何骨架（每 ``stride`` 个相邻重原子连一段，链很软时更平滑）；
- ``bonds``：所有重原子键（最细粒度，噪声也最大）。

**统计权重**（``weighting``）：``segment`` 链段加权（长链权重大，默认）/
``molecule`` 每条链等权（链长不一的体系两者会明显不同：310K 的 C6–C50 混油实测
链段加权 P₂ ≈ −0.13、每条链等权 P₂ ≈ −0.09）。

**参考轴**（``p2_axis``）：``z``（默认）/ ``x`` / ``y`` / ``xyz``（三个都算）。

**输出**：``orientation__s.csv``（S 随时间）、``orientation__p2.csv``（P₂）、
``orientation__dist.csv``（链段方向分布）。

**单独跑**：``python test/my_test/orientation.py --frames 50 --mode backbone -y``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "orientation"
TITLE = "链段取向分析"
SUB = "链段与参考轴的夹角 → 取向参数 S = (3⟨cos²θ⟩−1)/2：0 = 各向同性，1 = 完全对齐"


def test_orientation(eng, *, mode="repeat", stride=1, p2_axis="z",
                     weighting="segment", nbins=60, outdir=None,
                     formats=("csv",), **over):
    """跑工具的链段取向分析，打印统计量并导出 CSV。"""
    params = {"mode": str(mode), "stride": max(1, int(stride)),
              "p2_axis": str(p2_axis), "weighting": str(weighting),
              "nbins": int(nbins)}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("mode", "链段定义", "choice", "repeat",
      choices=[("repeat", "化学重复单元（默认）"), ("backbone", "几何骨架"),
               ("bonds", "所有重原子键")],
      help="repeat：按化学重复单元切链段（PE 的主链单元）；"
           "backbone：几何骨架，相邻重原子连段"),
    Q("stride", "几何骨架 stride", "int", 1, lo=1,
      help="只对 backbone / bonds 生效：每隔几个重原子连一段"),
    Q("p2_axis", "P2 参考轴", "choice", "z",
      choices=[("z", "z 轴（默认）"), ("x", "x 轴"), ("y", "y 轴"),
               ("xyz", "x + y + z 三者")]),
    Q("weighting", "统计权重", "choice", "segment",
      choices=[("segment", "链段加权（默认，长链权重大）"),
               ("molecule", "每条链等权")]),
    Q("nbins", "取向分布分箱数", "int", 60, lo=6),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_orientation, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
