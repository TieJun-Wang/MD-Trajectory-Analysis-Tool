# -*- coding: utf-8 -*-
r"""界面宽度分析（工具函数：``mdta.analysis.interface.analyze_interface_width``）。

**做什么**：沿某个方向做两组分的密度剖面，把"从本体值降到 10% / 90%"之间的
距离差当作**界面宽度**（10–90 判据）。界面越宽 = 两种物质互溶/缠结越厉害，
越窄 = 越像"油水分层"。

**判据**（引擎参数 ``method``）：``10-90``（默认）/ ``10-90（拟合）``；平滑窗口
``smooth`` 帧、分箱 ``nbins`` 可调。

**前提**：需要**至少 2 个组分**（界面是"两组分之间"的量）。单组分体系会被
引擎跳过并给出原因 —— 这时到「分析集合」里按元素/原子数把物种拆成独立集合，
或换多组分体系（``MDTA_MYTEST_SYSTEM=mix``）。还需要周期盒子。

**输出**：``interface__profile.csv``（两组分的密度剖面）、``interface__norm.csv``
（归一化后的 10–90 判读图）。

**单独跑**：``MDTA_MYTEST_SYSTEM=mix python test/my_test/interface.py --frames 20``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "interface"
TITLE = "界面宽度分析"
SUB = "密度剖面从 10% 到 90% 的距离差 = 界面宽度（互溶程度 / 分相程度的量化）"


def test_interface(eng, *, axis=2, nbins=120, method="10-90", smooth=5,
                   mode="mass", pair=None, outdir=None, formats=("csv",), **over):
    """跑工具的界面宽度分析，打印统计量并导出 CSV。

    只有一个组分时引擎会跳过（返回 ``None``），脚本打印原因并正常返回。
    """
    params = {"axis": int(axis), "nbins": int(nbins), "method": str(method),
              "smooth": int(smooth), "mode": str(mode), "pair": pair}
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("axis", "界面法向", "choice", "2",
      choices=[("2", "c (2)"), ("1", "b (1)"), ("0", "a (0)")]),
    Q("nbins", "分箱数", "int", 120, lo=10),
    Q("method", "判据", "choice", "10-90",
      choices=[("10-90", "10–90% 判据（默认）")],
      help="10-90：密度从 10% 升到 90% 的距离差"),
    Q("smooth", "平滑窗口（帧）", "int", 5, lo=0),
    Q("mode", "密度口径", "choice", "mass",
      choices=[("mass", "质量密度"), ("number", "数密度")]),
    Q("pair_a", "A 组（留空 = 原子数最多）", "text", ""),
    Q("pair_b", "B 组（留空 = 次多且≠A）", "text", ""),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_interface, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
