# -*- coding: utf-8 -*-
r"""二面角分析（工具函数：``mdta.analysis.conformation.analyze_dihedrals``）。

**做什么**：沿分子链枚举四原子二面角 φ，统计**二面角分布**，并按
``trans / gauche±`` 归类给出构象比例。主链二面角分布是判断 PE 类链"伸展 / 卷曲"
最直接的证据：``trans``（|φ| ≈ 180°）越多，链越像锯齿形伸展。

**四原子怎么来**（引擎参数 ``mode``，界面「二面角模式」）：

- ``auto``（默认）：有蛋白主链就用 φ/ψ，否则沿每条分子链的几何骨架；
- ``phi_psi``：蛋白质主链 φ/ψ（-180…180，按残基编号成对出现）；
- ``chain``：沿每条链的几何骨架（每 4 个相邻重原子一组）；
- ``topology``：按拓扑里定义好的二面角。

**归类阈值**（``gauche_edge``，默认 120°）：``|φ| ≤ 阈值`` 记 ``gauche±``，
其余记 ``trans``；与 Forcite 的 torsion 统计一致（边值包含在 gauche 一侧）。

**输出**：``dihedral__dist.csv``（分布）、``dihedral__trans.csv``（trans 比例
随时间）、可能还有按分子/残基的面板。

**单独跑**：``python test/my_test/dihedral.py``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "dihedral"
TITLE = "二面角分析"
SUB = "沿分子链统计二面角分布 + trans / gauche 构象比例（PE 伸展程度最直接的证据）"


def test_dihedral(eng, *, mode="auto", gauche_edge=120.0, bin_width=1.0,
                  outdir=None, formats=("csv",), **over):
    """跑工具的二面角分析，打印统计量并导出 CSV。"""
    params = {"mode": str(mode), "gauche_edge": float(gauche_edge),
              "bin_width": float(bin_width)}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("mode", "二面角模式", "choice", "auto",
      choices=[("auto", "自动（推荐）"), ("phi_psi", "蛋白质主链 φ/ψ"),
               ("chain", "沿每条分子链（几何骨架）"), ("topology", "按拓扑定义")],
      help="auto：有蛋白主链走 φ/ψ，否则沿分子链骨架"),
    Q("gauche_edge", "trans/gauche 阈值 (°)", "float", 120.0, lo=60, hi=180,
      help="|φ| ≤ 阈值记 gauche±，其余记 trans（Forcite 同约定：边值算 gauche）"),
    Q("bin_width", "分布柱宽 Bin width (°)", "float", 1.0, lo=0.05, hi=10,
      help="与 Forcite 的 Bin width 对应：1 → −181…181（362 格）"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_dihedral, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
