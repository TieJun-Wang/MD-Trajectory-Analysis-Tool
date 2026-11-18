# -*- coding: utf-8 -*-
r"""回转半径 Rg（工具函数：``mdta.analysis.conformation.analyze_rg``）。

**做什么**：逐帧算回转半径 ``Rg = sqrt(Σ m_i (r_i − r_cm)² / Σ m_i)``。

**两档口径**（界面「分析对象」单选，对应引擎参数 ``object``）：

- ``molecule``（默认）：**每个分子各算一个 Rg，再对帧内所有分子取平均** ——
  多分子体系里这才是"分子有多大"的正确回答，也与 Forcite 的 Rg 一致；
- ``system``：把所选对象当成**一个整体**算（多分子时数值接近盒子尺寸）。

**输出**（``export_result`` → 与界面「下载 CSV」同名）：
``rg__time.csv``（逐帧时间序列）、``rg__dist.csv``（Rg 分布直方图）、
``rg__permol.csv``（各分子平均 Rg，排序后）。

**单独跑**：``python test/my_test/rg.py``（不带参数 = 交互菜单）。
"""

from __future__ import annotations

try:                                            # 作为包导入
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:                             # 直接 python test/my_test/rg.py
    from _menu import Q, main_cli, run_analysis_menu

NAME = "rg"
TITLE = "回转半径 Rg"
SUB = "逐分子算 Rg 再取平均（与 Forcite 一致），或把整个对象当一个整体算"


def test_rg(eng, *, mass_weighted=True, object="molecule", outdir=None,
            formats=("csv",), **over):
    """跑工具的 Rg 分析，打印统计量并导出 CSV。

    ``object``：``"molecule"``（逐分子）/ ``"system"``（整体）。
    ``**over`` 可以透传引擎的其它参数（``unwrap`` / ``nbins`` / ``per_molecule``…）。
    """
    params = {"mass_weighted": bool(mass_weighted), "object": str(object)}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("object", "分析对象", "choice", "molecule",
      choices=[("molecule", "分子/链（逐分子再平均）"), ("system", "整体体系")],
      help="分子/链：每个分子各算 Rg 再平均（与 Forcite 一致）；"
           "整体：把所选对象当一个整体算"),
    Q("mass_weighted", "质量加权 Rg", "bool", True,
      help="Forcite 默认按质量加权；取消勾选 = 所有原子等权"),
]


def menu():
    """交互菜单 / 命令行入口：``python test/my_test/rg.py``。"""
    return run_analysis_menu(NAME, TITLE, SPECS, test_rg, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
