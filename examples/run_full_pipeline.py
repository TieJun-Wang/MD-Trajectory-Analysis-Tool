# -*- coding: utf-8 -*-
"""完整流程示例 —— 跑完全部 10 项分析并导出。

对应设计大纲第 27 章的"完整版本"。

运行::

    python examples/run_full_pipeline.py
    python examples/run_full_pipeline.py sys.tpr sys.xtc --equil 20000 --interval 100
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from mdta.pipeline import DEFAULT_ORDER, Analyzer      # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="MD 轨迹分析工具 —— 完整流程示例")
    ap.add_argument("topology", nargs="?",
                    default=os.path.join(_ROOT, "adk_oplsaa.tpr"))
    ap.add_argument("trajectory", nargs="?",
                    default=os.path.join(_ROOT, "adk_oplsaa.xtc"))
    ap.add_argument("--equil", type=float, default=None, help="平衡段长度 (ps)")
    ap.add_argument("--interval", type=float, default=None, help="抽帧间隔 (ps)")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--cutoff", type=float, default=5.0, help="接触判据 (Å)")
    ap.add_argument("--axis", type=int, default=2, choices=(0, 1, 2),
                    help="密度/界面方向 0=a 1=b 2=c")
    ap.add_argument("--rmax", type=float, default=12.0, help="RDF 最大距离 (Å)")
    ap.add_argument("--which", default=None, help="只跑指定分析项，逗号分隔")
    ap.add_argument("-o", "--outdir", default=os.path.join(_ROOT, "analysis_results"))
    args = ap.parse_args(argv[1:])

    # ---- 建立分析会话
    az = Analyzer(args.topology,
                  args.trajectory if os.path.isfile(args.trajectory) else None)
    az.set_frames(start_ps=args.start, stop_ps=args.stop,
                  interval_ps=args.interval, equil_ps=args.equil)

    # ---- 体系信息
    print(az.info_text(max_chain_types=10))

    # ---- 自动推荐主链与组分，也可以手动覆盖：
    #      az.set_primary(select(az.universe, resname="PE"), "PE 链")
    #      az.set_components({"PE": ..., "PEO": ...})
    #      az.add_interface_component("PE", "PEO", cutoff=5.0)   # 把界面区也作为组分
    setup = az.auto_setup()
    print(f"[选择] 主链: {setup['primary']}")
    print(f"[选择] 组分: {setup['components']}")
    print()

    which = [w.strip() for w in args.which.split(",")] if args.which else DEFAULT_ORDER

    # ---- 逐项分析 + 导出
    results = az.run_all(
        which,
        params={
            "density": {"axis": args.axis, "nbins": 100},
            "interface": {"axis": args.axis, "nbins": 120},
            "rdf": {"rmax": args.rmax, "nbins": 120, "compare_halves": True},
            "contact": {"cutoff": args.cutoff},
        },
        outdir=args.outdir,
        formats=("csv", "png"),
        excel=True,
        verbose=True,
        progress=lambda f, m: None,
    )

    # ---- 打印关键统计量
    print("\n" + "=" * 70)
    print("关键结果")
    print("=" * 70)
    for name, res in results.items():
        print(f"\n【{res.title}】")
        for k, v in list(res.summary.items())[:8]:
            if isinstance(v, float):
                print(f"    {k:<34} = {v:.6g}")
            else:
                print(f"    {k:<34} = {v}")

    print(f"\n全部结果已导出到: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
