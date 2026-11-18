# -*- coding: utf-8 -*-
r"""端到端距离 R_ee（工具函数：``mdta.analysis.conformation.analyze_end_to_end``）。

**做什么**：逐帧取每个分子的两个链端原子，算它们的直线距离（先按分子连通性
解包裹，避免被周期边界切断成"跨盒长键"）。

**链端怎么找**（引擎参数 ``ends``，界面「R_ee 链端来源」）：

- ``bond_graph``（默认，推荐）：在键图上找**最远的一对重原子**（首尾），
  对任意拓扑都成立；
- ``residue``：逐残基取首末原子（拓扑残基边界清楚时最省事）；
- ``custom``：手动指定两个原子（配合 ``atom_indices``）。

**输出**：``ree__time.csv``（逐帧 R_ee）、``ree__dist.csv``（分布）。

**单独跑**：``python test/my_test/ree.py``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "ree"
TITLE = "端到端距离 R_ee"
SUB = "逐分子取两个链端原子的直线距离（解包裹后），看链是伸展还是卷曲"


def test_ree(eng, *, ends="bond_graph", per_molecule=True, outdir=None,
             formats=("csv",), **over):
    """跑工具的 R_ee 分析，打印统计量并导出 CSV。"""
    params = {"ends": str(ends), "per_molecule": bool(per_molecule)}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("ends", "链端来源", "choice", "bond_graph",
      choices=[("bond_graph", "键图端（自动，推荐）"),
               ("residue", "逐残基首末"),
               ("custom", "手动指定两原子（需 atom_indices）")],
      help="bond_graph：在键图上找最远的一对重原子；residue：逐残基首末"),
    Q("per_molecule", "按分子分别算", "bool", True,
      help="勾上 = 每个分子各算自己的 R_ee（推荐）；"
           "不勾 = 只看对象整体的两端"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_ree, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
