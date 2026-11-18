# -*- coding: utf-8 -*-
r"""均方位移 MSD（工具函数：``mdta.analysis.dynamics.analyze_msd``）。

**做什么**：``MSD(t) = ⟨|r(t) − r(0)|²⟩``，再对长时段做线性拟合得到自扩散系数

```
D = MSD(t) / (2·dim·t)        dim=3 → D = MSD/(6t)
```

**追踪对象**（引擎参数 ``object``，界面「MSD 追踪对象」）：

- ``atom``（默认）：逐**原子**位移 —— 与 Materials Studio Forcite 的 MSD 口径一致，
  可以逐点对齐；
- ``molecule``：按**分子质心**位移 —— 看整条链怎么移动（链内原子振动被平均掉，
  曲线明显更平滑、D 更小）。

⚠️ 这只是口径选择，不是"哪种更准"：两者数值本来就差一个量级
（实测 PE-PP-MAPE 单链 t=1 ps：原子 0.96 Å²、质心 0.066 Å²）。

**去漂移**（``remove_drift``）：NPT/NVT 里整盒在缓慢平移时，全体原子的共同平移会
让 MSD 抬出一条额外的抛物线；勾上先减掉"全体平均位移"。盒子固定、质心不动不用勾。

**输出**：``msd__tau.csv``（MSD 随时间）、``msd__fit.csv``（拟合段与 D）。

**单独跑**：``python test/my_test/msd.py --frames 100 -y``
"""

from __future__ import annotations

try:
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _menu import Q, main_cli, run_analysis_menu

NAME = "msd"
TITLE = "均方位移 MSD"
SUB = "MSD(t) = ⟨|r(t)−r(0)|²⟩，线性拟合 → 扩散系数 D = MSD/(6t)"


def test_msd(eng, *, object="atom", remove_drift=False, max_particles=None,
             outdir=None, formats=("csv",), **over):
    """跑工具的 MSD 分析，打印统计量并导出 CSV。"""
    params = {"object": str(object), "remove_drift": bool(remove_drift)}
    if max_particles:
        params["max_particles"] = int(max_particles)
    params.update(over)
    res = eng.run(NAME, params)
    return eng.report(res, outdir=outdir, formats=formats, head=f"— {TITLE} —")


SPECS = [
    Q("object", "追踪对象", "choice", "atom",
      choices=[("atom", "原子（Forcite 口径）"), ("molecule", "分子质心")],
      help="原子：逐原子位移，与 Forcite 可逐点对齐；质心：看整条链怎么移动"),
    Q("remove_drift", "减去整体漂移", "bool", False,
      help="NPT/NVT 整盒缓慢平移时勾上；盒子固定、质心不动不用勾"),
    Q("max_particles", "最多追踪多少个粒子（留空 = 全部）", "int", None, lo=1),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_msd, sub=SUB,
                             default_frames=100)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
