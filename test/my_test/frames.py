# -*- coding: utf-8 -*-
r"""帧选择（工具函数：``mdta.preprocess.select_frames``）。

**做什么**：决定"参与统计的是哪几帧"。所有分析的统计量都是在这批帧上算的，
所以选帧方式直接决定结果 —— 例如把前 100 ps 的平衡段丢掉，Rg / 有序度会整体
偏高或偏低，而程序只会如实反映它拿到的帧。

**五个旋钮**（对应界面「帧范围」那一排输入框，单位都是 ps，留空 = 不限）：

| 参数 | 作用 |
|---|---|
| ``start_ps`` | 从哪个时刻开始（``0`` = 首帧） |
| ``stop_ps`` | 到哪个时刻结束（含端点） |
| ``interval_ps`` | 每隔多少 ps 取一帧（抽稀，省时间；但会丢掉相关时间内的细节） |
| ``equil_ps`` | **平衡段**：前多少 ps 丢掉不算（MD 开头那段还没平衡） |
| ``max_frames`` | 最多用多少帧（超了就按间隔抽稀） |

返回 :class:`mdta.preprocess.FrameSelection`（``indices`` / ``times_ps`` /
``notes``），``describe()`` 打印"用了多少帧、跨多长"，``notes`` 会说明哪条规则真正生效
（比如"间隔 5 ps → 60 帧"）。``halves()`` 给前后半段（``compare_halves`` 类对比用）。

**本脚本做三件事**：① 把不同参数组合的结果列成一张表；② 打印最终选择的
describe/notes 与首尾帧；③ 导出 ``frames__selection.csv``（帧号 + 时刻），
方便核对某个分析到底吃了哪些帧。

**单独跑**：``python test/my_test/frames.py --max-frames 20 --equil-ps 50 -y``
"""

from __future__ import annotations

try:
    from ._common import out_dir
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _common import out_dir
    from _menu import Q, main_cli, run_analysis_menu

NAME = "frames"
TITLE = "帧选择"
SUB = "时间窗 / 抽帧间隔 / 平衡段 / 帧数上限：同一批轨迹选出不同帧，统计量就不一样"


def test_frames(eng, *, start_ps=None, stop_ps=None, interval_ps=None,
                equil_ps=None, max_frames=None, outdir=None, formats=("csv",),
                **over):
    """按给定旋钮重新选帧，打印对照表并导出帧清单。"""
    import os

    import numpy as np
    import pandas as pd

    from mdta.preprocess import select_frames

    times = np.asarray(eng.az.times_ps, dtype=float)
    sel = select_frames(times, start_ps=start_ps, stop_ps=stop_ps,
                        interval_ps=interval_ps, equil_ps=equil_ps,
                        max_frames=max_frames)
    eng.frames = sel
    eng.az.frames = sel

    print(f"\n— {TITLE} —")
    print(f"轨迹共 {len(times):,} 帧，时间轴 {times[0]:g} – {times[-1]:g} ps，"
          f"帧间隔 {float(np.median(np.diff(times))) if len(times) > 1 else 0:g} ps")

    # ---------------------------------------------------------- ① 参数组合对照
    variants = [
        ("全部帧", dict()),
        ("只丢平衡段", dict(equil_ps=equil_ps)),
        ("只抽稀", dict(interval_ps=interval_ps)),
        ("只限上限", dict(max_frames=max_frames)),
        ("时间窗", dict(start_ps=start_ps, stop_ps=stop_ps)),
        ("本次选择（全部旋钮）", dict(start_ps=start_ps, stop_ps=stop_ps,
                                interval_ps=interval_ps, equil_ps=equil_ps,
                                max_frames=max_frames)),
    ]
    print("\n[参数组合对照]（同一批轨迹，不同选法选出多少帧）")
    for label, kw in variants:
        try:
            v = select_frames(times, **kw)
            span = (f"{v.times_ps[0]:g}–{v.times_ps[-1]:g} ps"
                    if v.n_frames else "（空）")
            print(f"  {label:<20} {v.n_frames:>6,} 帧   {span}")
        except Exception as exc:                            # noqa: BLE001
            print(f"  {label:<20} 失败：{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------- ② 最终选择
    print(f"\n[本次选择] {sel.describe()}")
    for note in sel.notes:
        print(f"  · {note}")
    if sel.n_frames:
        half_a, half_b = sel.halves()
        print(f"  首帧 index={int(sel.indices[0])} (t={sel.times_ps[0]:g} ps) ｜ "
              f"末帧 index={int(sel.indices[-1])} (t={sel.times_ps[-1]:g} ps)")
        print(f"  前后半段对比：前 {half_a.n_frames} 帧 / 后 {half_b.n_frames} 帧"
              f"（RDF、接触这类开了 compare_halves 会用到）")
        if sel.n_frames > 4:
            print(f"  前 5 帧 index：{np.asarray(sel.indices[:5]).tolist()} …")

    # ---------------------------------------------------------- ③ 导出帧清单
    files: list[str] = []
    if formats and sel.n_frames:
        target = str(outdir or out_dir())
        path = os.path.join(target, "frames__selection.csv")
        pd.DataFrame({"帧号 index": np.asarray(sel.indices, dtype=int),
                      "时间 (ps)": np.asarray(sel.times_ps, dtype=float)}
                     ).to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
        print("\n  导出：" + os.path.basename(path) + f"  →  {target}")

    return {"name": NAME, "panels": 0,
            "summary": {"选中帧数": sel.n_frames,
                        "时间跨度 (ps)": (f"{sel.times_ps[0]:g}–{sel.times_ps[-1]:g}"
                                          if sel.n_frames else "—")},
            "files": files, "seconds": 0.0, "res": None}


SPECS = [
    Q("start_ps", "起始时刻 start_ps (ps)", "float", None, lo=0,
      help="留空 = 首帧"),
    Q("stop_ps", "结束时刻 stop_ps (ps)", "float", None, lo=0,
      help="留空 = 末帧（含端点）"),
    Q("interval_ps", "抽帧间隔 interval_ps (ps)", "float", None, lo=0,
      help="每隔多少 ps 取一帧；留空 = 不抽稀"),
    Q("equil_ps", "丢掉的平衡段 equil_ps (ps)", "float", None, lo=0,
      help="前多少 ps 不算（MD 开头还没平衡的那段）"),
    Q("max_frames", "帧数上限 max_frames", "int", None, lo=1,
      help="超过就按间隔抽稀；留空 = 不限"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_frames, sub=SUB,
                             default_frames=0)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
