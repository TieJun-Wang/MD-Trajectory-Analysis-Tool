# -*- coding: utf-8 -*-
r"""一次跑完全部模块（``python test/my_test/run_all.py`` 或 ``python -m my_test``）。

只加载一次体系，然后依次跑：体系信息 → 帧选择 → 14 项分析（+ 可选的一键分析），
最后打印一张"✅ 完成 / ⏭ 跳过 / ✗ 失败"的汇总表。

```powershell
python test/my_test/run_all.py                      # 默认体系 20PE30，20 帧
python test/my_test/run_all.py --system mix --frames 10   # 多组分体系
python test/my_test/run_all.py --frames 200 --pipeline    # 多跑点帧 + 再跑一遍一键分析
python test/my_test/run_all.py --top a.xsd --trj a.trj    # 换自己的体系
```

⚠️ 单组分体系里「界面宽度」「两组分质心距」会**跳过并说明原因**（它们要两组分），
这是预期行为，不是脚本坏了；用 ``--system mix`` 就能看到它们跑出结果。
"""

from __future__ import annotations

import argparse
import time
import traceback

try:                                        # 作为包导入
    from ._common import SYSTEMS, Engine, out_dir, summarize
    from . import (boo, comdist, contact, density, dihedral, frames, interface,
                   msd, order, orientation, pipeline, rdf, rdf2d, ree, rg,
                   system, vacf)
except ImportError:                         # 直接 python test/my_test/run_all.py
    from _common import SYSTEMS, Engine, out_dir, summarize
    import boo
    import comdist
    import contact
    import density
    import dihedral
    import frames
    import interface
    import msd
    import order
    import orientation
    import pipeline
    import rdf
    import rdf2d
    import ree
    import rg
    import system
    import vacf


#: 执行顺序：先"看体系"再"跑分析"。每一项 ``(名字, 标题, 函数, 额外参数)``。
STEPS = [
    ("system", "体系信息与能力预判", system.test_system, {}),
    ("frames", "帧选择", frames.test_frames, {}),
    ("rg", "回转半径 Rg", rg.test_rg, {}),
    ("ree", "端到端距离 R_ee", ree.test_ree, {}),
    ("dihedral", "二面角分析", dihedral.test_dihedral, {}),
    ("density", "密度分布", density.test_density, {}),
    ("rdf", "径向分布函数 RDF", rdf.test_rdf, {}),
    ("rdf2d", "面内径向分布 RDF (2D)", rdf2d.test_rdf2d, {}),
    ("comdist", "两组分质心距", comdist.test_comdist, {}),
    ("contact", "接触分析", contact.test_contact, {}),
    ("interface", "界面宽度分析", interface.test_interface, {}),
    ("orientation", "链段取向分析", orientation.test_orientation, {}),
    ("order", "结构有序度分析", order.test_order, {}),
    ("boo", "键取向序参数 BOO", boo.test_boo, {}),
    ("msd", "均方位移 MSD", msd.test_msd, {}),
    ("vacf", "速度自相关函数 VACF", vacf.test_vacf, {}),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="把 test/my_test 里的功能脚本一次跑完",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--top", default=None, help="拓扑文件（不给 = 用预置体系）")
    ap.add_argument("--trj", default=None, help="轨迹文件（不给 = 用预置体系）")
    ap.add_argument("--system", default=None, choices=list(SYSTEMS),
                    help="预置体系：pe = 20PE30（单组分）/ mix = 5PE20…（多组分）")
    ap.add_argument("--select", default=None, help="分析对象选择串（默认自动识别）")
    ap.add_argument("--frames", type=int, default=20, help="最多用多少帧（0 = 全部）")
    ap.add_argument("--outdir", default="", help="CSV 输出目录（默认 test/my_test/_out）")
    ap.add_argument("--pipeline", action="store_true",
                    help="最后再跑一遍「一键分析」（Analyzer.run_all + 导出全部）")
    ap.add_argument("--only", default="",
                    help="只跑这些（逗号分隔，如 rg,contact）")
    ap.add_argument("--velocities", action="store_true",
                    help="加载时保留速度（只有 VACF 需要；MS 的 .trj 会转 TRR）")
    args = ap.parse_args(argv)

    outdir = str(args.outdir or out_dir())
    print("=" * 76)
    print("MD 轨迹分析工具 —— 功能脚本自检（test/my_test）")
    print("=" * 76)

    t0 = time.time()
    eng = Engine(args.top, args.trj, system=args.system, select=args.select,
                 max_frames=args.frames or None, velocities=args.velocities or None)
    only = {s.strip() for s in str(args.only).replace("，", ",").split(",") if s.strip()}

    rows, failed = [], []
    for name, title, fn, extra in STEPS:
        if only and name not in only:
            continue
        print("\n" + "=" * 76)
        print(f"[{title}]")
        print("=" * 76)
        t1 = time.time()
        try:
            kw = {"outdir": outdir, "formats": ("csv",), **extra}
            if name == "frames" and args.frames:
                kw["max_frames"] = int(args.frames)
            out = fn(eng, **kw)
            sec = time.time() - t1
            if out and out.get("skipped"):
                rows.append((name, "⏭", "体系不满足前提，已跳过", f"{sec:6.2f} s"))
            else:
                note = (f"{out.get('panels', 0)} 面板 / "
                        f"{len(out.get('files') or [])} 文件") if out else "—"
                rows.append((name, "✅", note, f"{sec:6.2f} s"))
        except SystemExit:
            raise
        except Exception as exc:                        # noqa: BLE001
            sec = time.time() - t1
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            rows.append((name, "✗", f"{type(exc).__name__}: {exc}"[:40], f"{sec:6.2f} s"))
            print(f"✗ 失败：{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)

    if args.pipeline:
        print("\n" + "=" * 76)
        print("[一键分析（全部项）]")
        print("=" * 76)
        t1 = time.time()
        try:
            pipeline.test_pipeline(eng, outdir=outdir, formats=("csv",),
                                   panel_pngs=False)
            rows.append(("pipeline", "✅", f"{len(eng.az.results)} 项结果",
                         f"{time.time() - t1:6.2f} s"))
        except Exception as exc:                        # noqa: BLE001
            failed.append(("pipeline", f"{type(exc).__name__}: {exc}"))
            rows.append(("pipeline", "✗", f"{type(exc).__name__}: {exc}"[:40],
                         f"{time.time() - t1:6.2f} s"))

    summarize(rows)
    n_ok = sum(1 for r in rows if r[1] == "✅")
    n_skip = sum(1 for r in rows if r[1] == "⏭")
    print(f"\n完成 {n_ok} 项，跳过 {n_skip} 项，失败 {len(failed)} 项；"
          f"总用时 {time.time() - t0:.1f} s")
    print(f"输出目录：{outdir}")
    if failed:
        for name, msg in failed:
            print(f"  ✗ {name} —— {msg}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
