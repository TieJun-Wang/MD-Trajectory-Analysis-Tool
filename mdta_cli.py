# -*- coding: utf-8 -*-
"""MD 轨迹分析工具 —— 命令行入口。

用法示例::

    # 1) 查看体系信息
    python mdta_cli.py info -t adk_oplsaa.tpr -x adk_oplsaa.xtc

    # 2) 列出分子/链，挑选要分析的对象
    python mdta_cli.py chains -t adk_oplsaa.tpr

    # 3) 运行全部分析并导出到 analysis_results/
    python mdta_cli.py run -t adk_oplsaa.tpr -x adk_oplsaa.xtc \\
        --equil 0 --interval 100 --cutoff 5 -o analysis_results

    # 4) 只跑 Rg / 端到端距离 / RDF
    python mdta_cli.py run -t sys.tpr -x sys.xtc --which rg,ree,rdf

    # 5) 批量分析一个目录下的所有 tpr+xtc
    python mdta_cli.py batch ./md_data -o ./results

    # 6) 启动图形界面
    python mdta_cli.py gui
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import sys

# ---------------------------------------------------------------- 控制台编码
def _fix_console_encoding() -> None:
    """在 Windows 的 GBK 控制台下也保证中文与 Å 等字符能打印。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


_fix_console_encoding()

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mdta import __version__                                   # noqa: E402
from mdta.io import TrajectoryError, load_trajectory, scan_input_files   # noqa: E402
from mdta.pipeline import ANALYSIS_TITLES, DEFAULT_ORDER, Analyzer        # noqa: E402
from mdta.plotting import set_agg_backend                                 # noqa: E402
from mdta.selection import list_chains, select                            # noqa: E402
from mdta.systeminfo import describe_system, format_system_info           # noqa: E402

# 命令行只把图存成文件，用非交互的 Agg 后端
set_agg_backend()


# ---------------------------------------------------------------- 公共参数
def add_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-t", "--top", dest="topology", help="拓扑文件（.tpr/.gro/.pdb ...）")
    p.add_argument("-x", "--traj", dest="trajectory", help="轨迹文件（.xtc/.trr/.dcd ...）")
    p.add_argument("--no-autodetect", action="store_true",
                   help="不自动寻找同名轨迹文件")


def add_frame_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("轨迹预处理（时间单位 ps）")
    g.add_argument("--start", type=float, default=None, help="起始时间 (ps)")
    g.add_argument("--stop", type=float, default=None, help="结束时间 (ps)")
    g.add_argument("--equil", type=float, default=None,
                   help="平衡段长度 (ps)，其内的数据不参与统计")
    g.add_argument("--interval", type=float, default=None, help="分析间隔 (ps)")
    g.add_argument("--max-frames", type=int, default=None, help="最多使用的帧数")


def add_selection_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("选择与参数")
    g.add_argument("--primary", default=None,
                   help="主链的选择语句（MDAnalysis 语法）；默认自动取最大链")
    g.add_argument("--components", default=None,
                   help="参与界面分析的组分，逗号分隔；默认自动识别")
    g.add_argument("--cutoff", type=float, default=5.0, help="接触判据截断距离 (Å)，默认 5")
    g.add_argument("--axis", type=int, default=2, choices=(0, 1, 2),
                   help="密度/界面分析方向（0=a,1=b,2=c），默认 2")
    g.add_argument("--nbins", type=int, default=100, help="密度分布 bin 数，默认 100")
    g.add_argument("--rmax", type=float, default=12.0, help="RDF 最大距离 (Å)，默认 12")
    g.add_argument("--rdf-bins", type=int, default=120, help="RDF bin 数，默认 120")
    g.add_argument("--dihedral-mode", default="auto",
                   choices=("auto", "phi_psi", "chain", "topology"),
                   help="二面角模式，默认 auto")
    g.add_argument("--gauche-edge", type=float, default=120.0,
                   help="trans/gauche 分界角 (°)，默认 120")
    g.add_argument("--g-ref", type=float, default=None, dest="g_ref",
                   help="结构有序度里「完全有序态」的 RDF 堆积峰高度；"
                        "不给则该分量不参与综合指数（先跑一次，"
                        "从 RDF 堆积峰高度 那一项读出数值后再传入）")


def add_output_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("输出")
    g.add_argument("-o", "--outdir", default="analysis_results", help="输出目录")
    g.add_argument("--formats", default="csv,png",
                   help="导出格式，逗号分隔（csv,png），默认 csv,png")
    g.add_argument("--no-excel", action="store_true", help="不生成 summary.xlsx")
    g.add_argument("--panel-pngs", action="store_true", dest="panel_pngs",
                   help="除每个分析一张合并 PNG 外，再为每个面板单独导出一张 PNG"
                        "（便于一张一张查看或排版）")
    g.add_argument("--dpi", type=int, default=200, help="图片分辨率，默认 200")
    g.add_argument("-q", "--quiet", action="store_true", help="安静模式")


def build_analyzer(args) -> Analyzer:
    top = args.topology
    traj = getattr(args, "trajectory", None)
    if not top:
        raise SystemExit("错误：必须用 -t/--top 指定拓扑文件。")
    az = Analyzer(top, traj)
    if args.no_autodetect and traj is None:
        pass
    az.set_frames(start_ps=args.start, stop_ps=args.stop, interval_ps=args.interval,
                  equil_ps=args.equil, max_frames=args.max_frames)
    return az


def setup_selection(az: Analyzer, args, *, verbose: bool = True) -> None:
    az.auto_setup()
    if getattr(args, "primary", None):
        ag = select(az.universe, query=args.primary)
        if ag.n_atoms == 0:
            raise SystemExit(f"错误：主链选择语句 {args.primary!r} 没有选中任何原子。")
        az.set_primary(ag, f"自定义选择: {args.primary}")
    if getattr(args, "components", None):
        names = [s.strip() for s in args.components.split(",") if s.strip()]
        all_comps = az.components
        chosen = {}
        for n in names:
            if n in all_comps:
                chosen[n] = all_comps[n]
            else:
                ag = select(az.universe, query=n)
                if ag.n_atoms == 0:
                    raise SystemExit(f"错误：组分选择语句 {n!r} 没有选中任何原子。")
                chosen[n] = ag
        az.set_components(chosen)
    if verbose:
        print(f"[选择] 主链: {az.primary_label} ({az.primary.n_atoms} 原子)")
        for n, ag in az.components.items():
            print(f"[选择] 组分 {n}: {ag.n_atoms} 原子 / {ag.residues.n_residues} residue")


def collect_params(args) -> dict:
    return {
        "density": {"axis": args.axis, "nbins": args.nbins},
        "interface": {"axis": args.axis, "nbins": max(args.nbins, 60)},
        "rdf": {"rmax": args.rmax, "nbins": args.rdf_bins},
        "contact": {"cutoff": args.cutoff},
        "dihedral": {"mode": args.dihedral_mode, "gauche_edge": args.gauche_edge},
        "order": ({"g_ref": args.g_ref} if getattr(args, "g_ref", None) else {}),
        "msd": {},
    }


# ---------------------------------------------------------------- 子命令
def cmd_info(args) -> int:
    mdt = load_trajectory(args.topology, args.trajectory,
                          autodetect=not args.no_autodetect)
    info = describe_system(mdt)
    print(format_system_info(info, max_chain_types=args.max_chains))
    if args.frames:
        from mdta.preprocess import select_frames

        fsel = select_frames(mdt.times_ps, start_ps=args.start, stop_ps=args.stop,
                             interval_ps=args.interval, equil_ps=args.equil)
        print(f"\n[帧选择] {fsel.describe()}")
        for n in fsel.notes:
            print(f"  - {n}")
    return 0


def cmd_chains(args) -> int:
    mdt = load_trajectory(args.topology, args.trajectory,
                          autodetect=not args.no_autodetect)
    u = mdt.universe
    print(f"分子/链清单（按 segid + residue 名归类，最小原子数 {args.min_atoms}）")
    print("-" * 72)
    cts = sorted(list_chains(u, min_atoms=args.min_atoms),
                 key=lambda c: -c.n_atoms)
    for c in cts:
        print(f"  {c}")
    print("-" * 72)
    print("提示：用 --primary \"<MDAnalysis 选择语句>\" 指定主链；")
    print("      例如 --primary \"segid seg_0_PE\"，或 --primary \"resid 1:200\"。")
    return 0


def cmd_run(args) -> int:
    az = build_analyzer(args)
    if not args.quiet:
        print(az.info_text())
    setup_selection(az, args, verbose=not args.quiet)
    which = None
    if args.which:
        which = [w.strip() for w in args.which.split(",") if w.strip()]
        bad = [w for w in which if w not in DEFAULT_ORDER]
        if bad:
            raise SystemExit(f"错误：未知分析项 {bad}；可选: {DEFAULT_ORDER}")
    az.run_all(which, params=collect_params(args), outdir=args.outdir,
               formats=[f.strip() for f in args.formats.split(",") if f.strip()],
               excel=not args.no_excel, panel_pngs=args.panel_pngs,
               verbose=not args.quiet)
    if not args.quiet:
        print(f"\n完成。共 {len(az.results)} 项分析，结果在 {os.path.abspath(args.outdir)}")
    return 0


def cmd_list(args) -> int:
    print("可用的分析项：")
    for k in DEFAULT_ORDER:
        print(f"  {k:12} {ANALYSIS_TITLES.get(k, k)}")
    return 0


def cmd_batch(args) -> int:
    files = []
    for pat in args.paths:
        if os.path.isdir(pat):
            for ext in ("*.tpr", "*.gro", "*.pdb", "*.xtc", "*.trr", "*.dcd"):
                files += globmod.glob(os.path.join(pat, ext))
        else:
            files += globmod.glob(pat)
    tops, trajs = scan_input_files(sorted(set(files)))
    if not tops:
        raise SystemExit("错误：没有找到任何拓扑文件。")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"批量分析：{len(tops)} 个体系 -> {os.path.abspath(args.outdir)}")
    all_results = []

    ok = 0
    for top in tops:
        stem = os.path.splitext(os.path.basename(top))[0]
        traj = None
        for t in trajs:
            if os.path.splitext(os.path.basename(t))[0] == stem:
                traj = t
                break
        sub = os.path.join(args.outdir, stem)
        print(f"\n=== {stem} (traj={os.path.basename(traj) if traj else '无'}) ===")
        try:
            ns = argparse.Namespace(**vars(args))
            ns.topology, ns.trajectory = top, traj
            ns.no_autodetect = False
            az = build_analyzer(ns)
            setup_selection(az, ns, verbose=False)
            which = [w.strip() for w in args.which.split(",")] if args.which else None
            res = az.run_all(which, params=collect_params(args), outdir=sub,
                             formats=[f.strip() for f in args.formats.split(",")],
                             excel=not args.no_excel, panel_pngs=args.panel_pngs,
                             verbose=not args.quiet, raise_errors=False)
            for r in res.values():
                r = r
                all_results.append((stem, r))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {stem} 失败: {type(exc).__name__}: {exc}")
    # 汇总表
    if all_results:
        try:
            import pandas as pd
            from mdta.export import summary_frame as sf

            frames = []
            for stem, r in all_results:
                df = sf([r])
                df.insert(0, "体系", stem)
                frames.append(df)
            combined = pd.concat(frames, ignore_index=True)
            path = os.path.join(args.outdir, "batch_summary.xlsx")
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                combined.to_excel(w, sheet_name="汇总", index=False)
            combined.to_csv(os.path.join(args.outdir, "batch_summary.csv"),
                            index=False, encoding="utf-8-sig")
            print(f"\n批量汇总已保存: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"汇总导出失败: {exc}")
    print(f"\n完成：{ok}/{len(tops)} 个体系分析成功。")
    return 0


def cmd_gui(args) -> int:
    try:
        from mdta_gui import main as gui_main
    except ImportError as exc:
        raise SystemExit(f"无法加载图形界面: {exc}") from exc
    return gui_main()


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdta",
        description="MD 轨迹分析工具 —— 面向高分子材料研究的分子动力学轨迹分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-V", "--version", action="version", version=f"MDTA {__version__}")
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("info", help="打印体系信息")
    add_io_args(pi)
    add_frame_args(pi)
    pi.add_argument("--max-chains", type=int, default=20, help="最多显示多少类分子/链")
    pi.add_argument("--frames", action="store_true", help="同时显示帧选择结果")
    pi.set_defaults(func=cmd_info)

    pc = sub.add_parser("chains", help="列出分子/链")
    add_io_args(pc)
    pc.add_argument("--min-atoms", type=int, default=1, help="只显示不小于该原子数的链")
    pc.set_defaults(func=cmd_chains)

    pr = sub.add_parser("run", help="运行分析并导出结果")
    add_io_args(pr)
    add_frame_args(pr)
    add_selection_args(pr)
    add_output_args(pr)
    pr.add_argument("--which", default=None,
                    help=f"要运行的分析项，逗号分隔；默认全部: {','.join(DEFAULT_ORDER)}")
    pr.set_defaults(func=cmd_run)

    pl = sub.add_parser("list", help="列出可用分析项")
    pl.set_defaults(func=cmd_list)

    pb = sub.add_parser("batch", help="批量分析多个体系")
    pb.add_argument("paths", nargs="+", help="目录或通配符")
    add_frame_args(pb)
    add_selection_args(pb)
    add_output_args(pb)
    pb.add_argument("--which", default=None, help="要运行的分析项，逗号分隔")
    pb.set_defaults(func=cmd_batch, topology=None, trajectory=None)

    pg = sub.add_parser("gui", help="启动图形界面")
    pg.set_defaults(func=cmd_gui)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except TrajectoryError as exc:
        print(f"轨迹读取错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
