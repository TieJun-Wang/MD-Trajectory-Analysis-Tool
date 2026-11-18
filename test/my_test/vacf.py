# -*- coding: utf-8 -*-
r"""速度自相关函数 VACF（工具函数：``mdta.analysis.dynamics.analyze_vacf``）。

**做什么**：``C(t) = ⟨v(0)·v(t)⟩`` —— 速度的自相关。它在 t=0 最大、随后衰减到 0，
衰减快慢反映"速度记忆"有多久；对 ``C(t)`` 积分还能得到扩散系数（Green–Kubo）。

**前提**：轨迹里必须**带速度**（Materials Studio ``.trj`` 转 TRR 时带速度；
多数 ``.xtc`` 不带，会让这一项跳过并说明原因）。

**引擎参数**：``max_lag``（τ 上限帧数，留空 = 全段）、``normalize``
（按 C(0) 归一化，默认不归一化 —— 与 Forcite 的原始量一致）。

**输出**：``vacf__vacf.csv``（C(t)）、``vacf__fit.csv``（归一化后与拟合段）。

**单独跑**：``python test/my_test/vacf.py --frames 100 -y``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "vacf"
TITLE = "速度自相关函数 VACF"
SUB = "C(t) = ⟨v(0)·v(t)⟩：速度记忆多长；积分还能给 Green–Kubo 扩散系数"


def test_vacf(eng, *, max_lag=None, normalize=False, outdir=None,
              formats=("csv",), **over):
    """跑工具的 VACF 分析，打印统计量并导出 CSV。"""
    params = {"normalize": bool(normalize)}
    if max_lag:
        params["max_lag"] = int(max_lag)
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("max_lag", "τ 上限 (帧，留空 = 全段)", "int", None, lo=1),
    Q("normalize", "按 C(0) 归一化", "bool", False,
      help="不归一化 = Forcite 的原始量；归一化后首点恒为 1，便于横向比较"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_vacf, sub=SUB,
                             default_frames=100, velocities=True)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
