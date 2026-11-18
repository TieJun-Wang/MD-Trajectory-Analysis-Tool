# -*- coding: utf-8 -*-
r"""一键分析（工具函数：``mdta.pipeline.Analyzer.run_all`` + ``mdta.export.export_all``）。

**做什么**：这就是图形界面点「开始分析」背后的那一次调用 —— 按分析项顺序逐项跑，
跑完把**每项结果 + 汇总表（Excel）+ 图（PNG）**一起导出，目录结构固定：

```
<输出目录>/
├── rg__time.csv / rg__dist.csv / rg__permol.csv …   ← 每项按面板切片（与界面「下载 CSV」同名）
├── rg.png / contact.png …                            ← 每项一张多面板汇总图
└── summary.xlsx                                      ← 全部项的统计量汇总（「汇总」+ 每项一张明细表）
```

**几件只有批量跑才有的事**：

- **预热**（``warmup``）：分子编号 / 键图 / 首次解包裹这些一次性开销先做掉，
  否则会记在队首那一项头上，看起来像"这项特别慢"；
- **短项优先**（``order="shortest"`` + ``estimate_times``）：先跑几帧试算，按预估
  用时从短到长排队，好让结果尽早出现在页面上。默认 ``"given"`` 保持命令行原顺序；
- **跳过 ≠ 失败**：体系不满足前提的项（单组分做界面宽度、无速度做 VACF…）记进
  ``skips`` 并给出原因，不会让整轮失败 —— 界面就是按这个显示"⏭ 跳过：原因"。

**单独跑**：``python test/my_test/pipeline.py --frames 20 --estimate -y``
（``--which rg,contact`` 只跑其中几项）
"""

from __future__ import annotations

try:
    from ._common import fmt, out_dir
    from ._menu import Q, main_cli, run_analysis_menu
except ImportError:
    from _common import fmt, out_dir
    from _menu import Q, main_cli, run_analysis_menu

NAME = "pipeline"
TITLE = "一键分析（全部项）"
SUB = "Analyzer.run_all：预热 → 逐项跑 → 导出 CSV/PNG/Excel；跳过项给原因而不是失败"


def test_pipeline(eng, *, which="", order="given", estimate=False,
                  panel_pngs=False, excel=True, outdir=None, formats=("csv",),
                  **params_over):
    """跑全部（或 ``which`` 指定的）分析项并导出，打印耗时与跳过表。

    ``params_over`` 可以给某一项的参数：``params_over={"contact": {"cutoff": 4.5}}``。
    """
    import os

    from mdta.pipeline import ANALYSIS_GROUPS, ANALYSIS_TITLES, DEFAULT_ORDER

    names = [n.strip() for n in str(which).replace("，", ",").split(",") if n.strip()]
    names = names or list(DEFAULT_ORDER)
    unknown = [n for n in names if n not in DEFAULT_ORDER]
    if unknown:
        raise SystemExit(f"未知分析项 {unknown}；可选：{list(DEFAULT_ORDER)}")

    if eng.frames is not None:
        eng.az.frames = eng.frames

    print(f"\n— {TITLE} —")
    grouped = " ｜ ".join(f"{label}: {'、'.join(mods)}" for label, mods in ANALYSIS_GROUPS)
    print(f"  分组：{grouped}")
    print(f"  本次 {len(names)} 项：{names}")

    # ---------------------------------------------------------- ① 预估用时
    est = None
    if estimate:
        print("\n[预估用时] 先跑几帧实测，按用时从短到长排队（界面也是这么排的）")
        est = eng.az.estimate_times(names, probe_frames=8)
        for n, rec in est.items():
            sec = rec.get("est_sec")
            print(f"  {n:<11} {'—' if sec is None else f'{sec:>8.2f} s'}"
                  f"  {rec.get('note', '')}")
        order = "shortest"

    # ---------------------------------------------------------- ② 正式跑
    target = str(outdir or out_dir())
    print(f"\n[运行] 导出到 {target}")
    results = eng.az.run_all(names, params=params_over, outdir=target,
                             formats=tuple(formats), excel=bool(excel),
                             panel_pngs=bool(panel_pngs), verbose=True,
                             order=order,
                             # run_all 要的是 ``{名字: 预估秒数}``（``sort_by_estimate``
                             # 的口径）；estimate_times 返回的是每项一条记录，这里取出 est_sec
                             estimates=({n: rec.get("est_sec") for n, rec in est.items()}
                                        if est else None))

    # ---------------------------------------------------------- ③ 汇总
    print("\n[每项耗时]")
    timings = dict(getattr(eng.az, "timings", {}) or {})
    for n in names:
        if n in results:
            print(f"  ✓ {ANALYSIS_TITLES.get(n, n):<14} {timings.get(n, 0.0):>7.2f} s  "
                  f"{len(results[n].curves)} 条曲线 / {len(results[n].summary)} 项统计")
    skips = dict(getattr(eng.az, "skips", {}) or {})
    for n, why in skips.items():
        print(f"  ⏭ {ANALYSIS_TITLES.get(n, n):<14} 跳过：{why}")

    print("\n[统计量摘要]")
    for n, res in results.items():
        head = "；".join(f"{k} = {fmt(res.summary[k])}" for k in list(res.summary)[:3])
        print(f"  {n:<11} {head}")

    # 只数这次写出的文件（输出目录里可能还留着别的分析项的旧文件）
    files = []
    if os.path.isdir(target):
        mine = {f"{n}.png" for n in results} | {f"{n}.csv" for n in results}
        for f in sorted(os.listdir(target)):
            path = os.path.join(target, f)
            if f.startswith(tuple(f"{n}__" for n in results)) or f in mine \
                    or f == "summary.xlsx":
                files.append(path)
        print(f"\n[导出] {len(files)} 个文件在 {target}")
    return {"name": NAME, "panels": 0,
            "summary": {"完成项": len(results), "跳过项": len(skips),
                        "导出文件": len(files)},
            "files": files, "seconds": sum(timings.values()), "res": None}


SPECS = [
    Q("which", "只跑哪几项（逗号分隔，留空 = 全部）", "text", "",
      help="例如 rg,contact,orientation"),
    Q("estimate", "先预估用时（按短项优先排队）", "bool", False,
      help="先跑几帧实测每项用时；帧数少的体系关掉它直接跑更划算"),
    Q("excel", "导出 summary.xlsx 汇总表", "bool", True),
    Q("panel_pngs", "每个面板单独导一张 PNG", "bool", False,
      help="默认每项只导一张多面板合并图；勾上再按面板各导一张"),
]


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_pipeline, sub=SUB,
                             default_frames=20)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
