# -*- coding: utf-8 -*-
"""RDF —— ``util_rdf.py``（MDTransport / Shah 组）算法的移植版。

这份脚本把第三方实现 ``test/util_rdf.py`` 的**算法本身**搬过来（不依赖它那个
包：原文件用的是 ``from .util import BaseAnalysis, tqdm`` 的相对导入，还带
matplotlib / multiprocessing，单独拷出来跑不起来）。算法逐条对应：

============================  ==========================================
原实现                         这里
============================  ==========================================
`process_frame_rdf()`         :func:`frame_histogram`
`compute_and_save_rdf()`      :func:`rdf_mdat`
`init_rdf_process()`          —— 省掉（改为单进程；数值完全相同）
============================  ==========================================

它的口径（与 MDAnalysis ``InterRDF`` 的差别就在这里）
----------------------------------------------------
1. **逐帧对全部原子对算距离**：``diffs = c1[:,None,:] − c2[None,:,:]``，
   然后**当场做最小镜像** ``diffs −= box*round(diffs/box)``（不预先折叠坐标）；
2. 同种粒子（A–A）**只剔除对角自配对**，把 ``N×N`` 矩阵里非对角的 N(N−1)
   个距离**全部**塞进直方图 —— 即有序对都算，与 ``InterRDF`` 的
   ``N(N−1)`` 分子分母口径一致；
3. **归一化的理想气体密度用"邻居数"而不是"原子数"**：
   ``ideal_rho = (n2 − 1)/V``（同种）或 ``n2/V``（异种）—— 这就是同种粒子
   分母要减 1 的原因；
4. 直方图是**按帧累加**再统一除，``g(r) = (Σhist / (n_frames·n1)) / (shell_V · ideal_rho)``；
5. **配位数曲线** = 直接对"每个中心粒子每帧的平均邻居数"做 cumsum，不做密度归一；
6. ``r`` 取**bin 中心**（``bins[:-1] + dr/2``）；
7. ``r_max`` 默认取 **最小盒边的一半**（保证最小镜像不漏）；
8. 盒体积用**参与帧的平均体积**。

代价（原实现也是）：每帧一次 ``n1×n2×3`` 的临时数组 —— 2,000 个原子就是
48 MB/帧、距离计算 O(n1·n2)。所以大体系请先抽稀/只取代表性原子（这里给了
``chunk_rows`` 分块，结果与不分块**逐位相同**）。

单独跑：``python test/mda_test/rdf_mdat.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    from ._common import frame_indices, save_csv
except ImportError:
    from _common import frame_indices, save_csv

#: 原实现 ``args.RBIN`` 的默认值（Å）
DR_DEFAULT = 0.02
#: 单块临时数组的字节上限（分块只是为了内存，结果与不分块逐位相同）
_CHUNK_BYTES = 256 * 1024 * 1024


def _bins(box_lengths, dr, rmax):
    """与 ``compute_and_save_rdf`` 相同的 bin 划分：默认到最小盒边的一半。"""
    L = np.asarray(box_lengths, dtype=float).ravel()[:3]
    if rmax is None:
        rmax = float(np.min(L)) / 2.0
    dr = float(dr) if dr else DR_DEFAULT
    return np.arange(0.0, float(rmax) + dr, dr)


def frame_histogram(c1, c2, box, bins, same):
    """一帧的距离直方图（`process_frame_rdf` 的等价实现，可分段算）。

    ``c1`` / ``c2`` 是**该帧**的坐标 ``(n1,3)`` / ``(n2,3)``，``box`` 是
    ``(3,)`` 或 ``(6,)``（只用前三个边长，与原实现一样按正交盒处理）。
    """
    c1 = np.asarray(c1, dtype=np.float32)
    c2 = np.asarray(c2, dtype=np.float32)
    L = np.asarray(box, dtype=np.float32).ravel()[:3]
    n1, n2 = c1.shape[0], c2.shape[0]
    hist = np.zeros(bins.size - 1, dtype=np.float64)
    if n1 == 0 or n2 == 0:
        return hist
    # 一块多少行才能让 n_chunk*n2*3*4 字节 ≤ 预算
    row_bytes = max(1, n2 * 3 * 4)
    chunk = max(1, int(_CHUNK_BYTES // row_bytes))
    chunk = min(chunk, n1)
    for s in range(0, n1, chunk):
        e = min(s + chunk, n1)
        diffs = c1[s:e, None, :] - c2[None, :, :]
        if np.all(L > 0):
            diffs -= L * np.round(diffs / L)          # 当场最小镜像
        dists = np.linalg.norm(diffs, axis=2)
        if same:
            rows = np.arange(s, e)
            dists[rows - s, rows] = np.inf            # 剔除对角自配对
            d = dists[np.isfinite(dists)]
        else:
            d = dists.ravel()
        if d.size:
            h, _ = np.histogram(d, bins=bins)
            hist += h
    return hist


def rdf_mdat(u, ag1, ag2=None, *, dr=DR_DEFAULT, rmax=None, frames=None,
             verbose=True):
    """按 ``util_rdf.py`` 的口径算 RDF。

    返回 ``{"r", "g_r", "coordination_number", "hist", "bins", "n_frames",
    "n1", "n2", "volume"}``。
    """
    a = ag1
    b = a if ag2 is None else ag2
    same = (a is b) or (a.n_atoms == b.n_atoms and np.array_equal(a.indices,
                                                               b.indices))
    if a.n_atoms == 0 or b.n_atoms == 0:
        raise SystemExit(f"原子组为空（A={a.n_atoms}，B={b.n_atoms}）：检查选择串，"
                         f"例如 1000water 的水分子是 name O1／H1／H2，氧用 'element O'")
    frames = frame_indices(u) if frames is None else [int(f) for f in frames]
    if not frames:
        raise SystemExit("没有可用的帧")

    # 先把每一帧的盒边长取出来（原实现用参与帧的平均体积）
    d = u.trajectory[frames[0]].dimensions
    bins = _bins(np.asarray(d, dtype=float) if d is not None
                 else np.array([u.atoms.n_atoms ** (1 / 3)] * 3), dr, rmax)
    hist = np.zeros(bins.size - 1, dtype=np.float64)
    vols = []
    for fi in frames:
        ts = u.trajectory[fi]
        dim = ts.dimensions
        L = np.asarray(dim, dtype=float).ravel()[:3] if dim is not None else None
        c1 = np.asarray(a.positions, dtype=np.float32)
        c2 = c1 if same else np.asarray(b.positions, dtype=np.float32)
        hist += frame_histogram(c1, c2, L if L is not None else np.zeros(3), bins,
                               same)
        vols.append(float(L[0] * L[1] * L[2]) if L is not None else float("nan"))
        if verbose and (len(frames) <= 10 or fi == frames[-1]):
            print(f"  帧 {fi}: 累计距离 {int(hist.sum()):,}")

    n_f = len(frames)
    n1, n2 = int(a.n_atoms), int(b.n_atoms)
    shell_v = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    avg_vol = float(np.nanmean(vols))
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        raise SystemExit("体系没有盒子信息：RDF 需要周期盒（最小镜像）")
    per_center = hist / float(n_f * n1)                    # 每个中心粒子每帧的平均邻居数
    num_neighbors = (n2 - 1) if same else n2               # ← 同种粒子减 1
    ideal_rho = num_neighbors / avg_vol
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = per_center / (shell_v * ideal_rho)
    g_r[~np.isfinite(g_r)] = 0.0
    cn = np.cumsum(per_center)
    r = bins[:-1] + (bins[1] - bins[0]) / 2.0              # bin 中心

    if verbose:
        print(f"RDF：{n1} × {n2} 个原子，{n_f} 帧，dr = {bins[1] - bins[0]:g} Å，"
              f"r ≤ {bins[-1]:g} Å（同种={same}，理想密度 {ideal_rho:.6g} Å⁻³）")
    return {"r": r, "g_r": g_r, "coordination_number": cn, "hist": hist,
            "bins": bins, "n_frames": n_f, "n1": n1, "n2": n2,
            "volume": avg_vol, "same": same}


def test_rdf_mdat(u, ag1=None, ag2=None, dr=DR_DEFAULT, rmax=None, frames=None,
                  prefix="rdf_mdat", compare=False, step=1, stop=None):
    """出图/出数：RDF + 配位数曲线，导出 ``{prefix}.csv``。

    CSV 列与原实现一致：``r (Ang.), g_r, coordination_number``。

    ``compare=True`` 时再跑一遍 MDAnalysis 的 ``InterRDF``（**同一批帧**）并打印
    最大偏差 —— 两者口径相同的话应当只差浮点舍入。
    """
    if ag1 is None:
        ag1 = u.atoms
    res = rdf_mdat(u, ag1, ag2, dr=dr, rmax=rmax, frames=frames)
    r, g = res["r"], res["g_r"]
    cn = res["coordination_number"]
    save_csv(f"{prefix}.csv", [r, g, cn],
             header="r (Ang.),g_r,coordination_number", fmt="%.6f,%.6f,%.6f")

    fin = np.isfinite(g)
    if fin.any():
        k = int(np.argmax(np.where(fin, g, -np.inf)))
        print(f"  首峰 r = {r[k]:.3f} Å，g = {g[k]:.4f}；"
              f"r = {r[-1]:.2f} Å 处配位数 = {cn[-1]:.3f}")
    if compare:
        # 对拍时统一按 range(0, stop, step) 取帧（InterRDF 只支持这种取法）
        start = int(frames[0]) if frames else 0
        compare_interrdf(u, ag1, ag2, dr=dr, rmax=rmax, start=start,
                         stop=stop, step=step)
    return res


def compare_interrdf(u, ag1, ag2=None, *, dr=DR_DEFAULT, rmax=None, start=0,
                     stop=None, step=1):
    """与 MDAnalysis ``InterRDF`` 逐点比对 **同一批帧**（口径相同 → 只差浮点舍入）。

    两边都按 ``range(start, stop, step)`` 取帧：``InterRDF`` 只支持
    ``start/stop/step``，所以这里不传任意帧列表，避免"比的是不同帧"这种假差异。
    """
    from MDAnalysis.analysis.rdf import InterRDF

    a = ag1
    b = a if ag2 is None else ag2
    same = (a is b) or (a.n_atoms == b.n_atoms and np.array_equal(a.indices,
                                                               b.indices))
    stop = len(u.trajectory) if stop is None else int(stop)
    step = max(1, int(step))
    fl = list(range(int(start), stop, step))
    mine = rdf_mdat(u, a, b, dr=dr, rmax=rmax, frames=fl, verbose=False)
    r_min, r_max = float(mine["bins"][0]), float(mine["bins"][-1])
    nbins = int(mine["bins"].size - 1)
    ana = InterRDF(a, b, range=(r_min, r_max), nbins=nbins,
                   exclusion_block=(1, 1) if same else None)
    ana.run(start=int(start), stop=stop, step=step)
    g_mda = np.asarray(ana.results.rdf, dtype=float)
    r_mda = np.asarray(ana.results.bins, dtype=float)
    n = min(g_mda.size, mine["g_r"].size)
    ok = np.isfinite(mine["g_r"][:n]) & np.isfinite(g_mda[:n]) & (g_mda[:n] > 1e-9)
    if not ok.any():
        print("  （无可比点）")
        return None
    absd = float(np.max(np.abs(mine["g_r"][:n][ok] - g_mda[:n][ok])))
    rel = float(np.max(np.abs(mine["g_r"][:n][ok] - g_mda[:n][ok])
                       / np.abs(g_mda[:n][ok])))
    print(f"  vs InterRDF（帧 {fl[0]}…{fl[-1]} step {step}，"
          f"{len(fl)} 帧；r {r_mda[0]:.3f}–{r_mda[-1]:.3f} Å，{nbins} bin）："
          f"最大绝对偏差 {absd:.3e}，最大相对偏差 {rel:.3e}")
    return {"abs": absd, "rel": rel, "g_interrdf": g_mda, "r_interrdf": r_mda}


def compare_engine(u, ag1, ag2=None, *, dr=DR_DEFAULT, rmax=None, frames=None,
                   mode="inter"):
    """与**本工具引擎**（``mdta.analysis.interface.analyze_rdf``）对拍同一批帧。

    引擎的 bin 是 ``linspace(0, rmax, nbins+1)``，这里把 ``nbins`` 取成
    "同样步长"的整数，两边网格一致后才比较。
    """
    from types import SimpleNamespace

    from mdta.analysis.interface import analyze_rdf
    from mdta.io import MDTrajectory

    a = ag1
    b = a if ag2 is None else ag2
    fl = [int(f) for f in (frame_indices(u) if frames is None else frames)]
    mine = rdf_mdat(u, a, b, dr=dr, rmax=rmax, frames=fl, verbose=False)
    rmax_used = float(mine["bins"][-1])
    nbins = int(mine["bins"].size - 1)

    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(fl, dtype=int),
                          times_ps=times[np.asarray(fl, dtype=int)],
                          n_frames=len(fl),
                          describe=f"{len(fl)} 帧（指定）", notes=[],
                          head=lambda self, k: self)
    groups = {"A": a} if (a is b) else {"A": a, "B": b}
    pairs = [("A", "A")] if (a is b) else [("A", "B")]
    res = analyze_rdf(mdt, groups, sel, pairs=pairs, rmax=rmax_used,
                      nbins=nbins, exclude_bonded=False, mode=mode)
    g_eng = None
    for c in res.curves:
        if str(c.kind) != "bar":
            g_eng = np.asarray(c.y, dtype=float)
            break
    if g_eng is None or g_eng.size == 0:
        print("  （引擎没给出 g(r) 曲线）")
        return None
    n = min(g_eng.size, mine["g_r"].size)
    ok = np.isfinite(mine["g_r"][:n]) & np.isfinite(g_eng[:n]) & (g_eng[:n] > 1e-9)
    absd = float(np.max(np.abs(mine["g_r"][:n][ok] - g_eng[:n][ok])))
    rel = float(np.max(np.abs(mine["g_r"][:n][ok] - g_eng[:n][ok])
                       / np.abs(g_eng[:n][ok])))
    print(f"  vs 工具引擎（mode={mode}，{nbins} bin，同样 {len(fl)} 帧）："
          f"最大绝对偏差 {absd:.3e}，最大相对偏差 {rel:.3e}")
    return {"abs": absd, "rel": rel, "g_engine": g_eng}


def main(argv=None):
    """命令行入口（带 ``-m/--menu`` 时改走交互菜单）。

    例（1000water 全体系、100 帧）::

        python test/mda_test/rdf_mdat.py --top "C:\\temp\\dataset\\1000water\\1000water.xtd" --trj "C:\\temp\\dataset\\1000water\\1000water.trj" --select all --frames 100 --dr 0.02
    """
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-m", "--menu"):
        return menu()

    ap = argparse.ArgumentParser(
        description="按 util_rdf.py（MDTransport）口径算 RDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--top", default=None, help="拓扑文件（.xsd/.xtd/.pdb/.gro/.tpr…）")
    ap.add_argument("--trj", default=None, help="轨迹（.trj/.trr/.xtc/.dcd…），可留空=单帧")
    ap.add_argument("--select", default="all",
                    help='中心粒子选择串，默认 "all"（全部原子）；'
                         '水中常用 "element O" 或 "name O1"')
    ap.add_argument("--select-b", default=None,
                    help="第二组选择串（留空 = 与 --select 同组，算 A–A）")
    ap.add_argument("--frames", type=int, default=0,
                    help="跑多少帧（0 = 全部；N>0 时在整条轨迹上**均匀取 N 帧**）")
    ap.add_argument("--start", type=int, default=0, help="起始帧（默认 0）")
    ap.add_argument("--stop", type=int, default=0, help="结束帧（不含，0 = 到末尾）")
    ap.add_argument("--step", type=int, default=0,
                    help="帧间隔（0 = 自动：按 --frames 均匀取）")
    ap.add_argument("--dr", type=float, default=DR_DEFAULT,
                    help=f"bin 宽（Å），默认 {DR_DEFAULT}（原实现 RBIN 默认值）")
    ap.add_argument("--rmax", type=float, default=None,
                    help="最大 r（Å）；默认 = 最小盒边的一半（原实现同款）")
    ap.add_argument("--outdir", default=None, help="CSV 输出目录（默认当前目录）")
    ap.add_argument("--prefix", default="rdf_mdat", help="CSV 文件名前缀")
    ap.add_argument("--compare", action="store_true",
                    help="同时和 MDAnalysis InterRDF 对拍（同一批帧）")
    ap.add_argument("--compare-engine", action="store_true",
                    help="同时和本工具引擎 analyze_rdf 对拍")
    args = ap.parse_args(argv)

    if args.outdir:
        os.environ["MDTA_MDA_OUT"] = args.outdir

    try:
        from ._common import build_universe
    except ImportError:
        from _common import build_universe

    u, ag_a = build_universe(args.top, args.trj, select=args.select)
    ag_b = None
    if args.select_b:
        ag_b = u.select_atoms(args.select_b)
        print(f"第二组 {args.select_b!r} → {ag_b.n_atoms} 个原子")

    # ---- 帧挑选：均匀取 N 帧（把实际帧号打印出来，便于复现）
    n_all = len(u.trajectory)
    start = int(args.start)
    stop = args.stop if args.stop and args.stop > 0 else n_all
    if args.frames and args.frames > 0 and args.frames < (stop - start):
        # 均匀取**正好 N 帧**（np.round 后可能重复 → unique）
        frames = np.unique(np.round(np.linspace(start, stop - 1,
                                                args.frames)).astype(int)).tolist()
        step_used = (frames[-1] - frames[0]) / max(len(frames) - 1, 1)
    else:
        step_used = int(args.step) if args.step and args.step > 0 else 1
        frames = list(range(start, stop, step_used))
    print(f"轨迹共 {n_all} 帧 → 本次跑 {len(frames)} 帧"
          f"（{frames[0]} … {frames[-1]}，平均间隔 {step_used:g}）")

    res = test_rdf_mdat(u, ag_a, ag_b, dr=args.dr, rmax=args.rmax,
                        frames=frames, prefix=args.prefix)
    r, cn = res["r"], res["coordination_number"]
    for cut in (1.2, 2.0, 3.5, 5.5):
        k = int(np.searchsorted(r, cut) - 1)
        if 0 <= k < cn.size:
            print(f"  r ≤ {cut:.1f} Å 的配位数 = {cn[k]:.3f}")

    if args.compare:
        compare_interrdf(u, ag_a, ag_b, dr=args.dr, rmax=args.rmax,
                         start=start, stop=stop, step=int(max(1, round(step_used))))
    if args.compare_engine:
        compare_engine(u, ag_a, ag_b, dr=args.dr, rmax=args.rmax, frames=frames)
    return 0


def menu():
    """交互菜单入口：``python test/mda_test/rdf_mdat.py``（不带参数即可）。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("dr", "bin 宽 dr (Å)", "float", DR_DEFAULT, lo=0.001,
          help="原实现 RBIN 默认 0.02；0.05 更快、曲线更平滑"),
        Q("rmax", "最大 r (Å，0 = 自动取最小盒边的一半)", "float", 0.0, lo=0.0),
        Q("select_b", "第二组选择串", "text", "",
          help="留空 = 与第一组同种（A–A）；填了就算 A–B"),
        Q("compare", "同时和 MDAnalysis InterRDF 对拍？", "bool", False),
        Q("compare_engine", "同时和本工具引擎 analyze_rdf 对拍？", "bool", False),
    ]

    def _run(cfg):
        u = cfg["u"]
        a = cfg["ag"]
        b = u.select_atoms(cfg["select_b"]) if cfg.get("select_b") else None
        rmax = cfg["rmax"] or None
        res = test_rdf_mdat(u, a, b, dr=cfg["dr"], rmax=rmax,
                            frames=cfg["frames"], prefix=cfg["prefix"])
        r, cn = res["r"], res["coordination_number"]
        for cut in (1.2, 2.0, 3.5, 5.5):
            k = int(np.searchsorted(r, cut) - 1)
            if 0 <= k < cn.size:
                print(f"  r ≤ {cut:.1f} Å 的配位数 = {cn[k]:.3f}")
        if cfg.get("compare"):
            step = max(1, int(round((cfg["frames"][-1] - cfg["frames"][0])
                                    / max(len(cfg["frames"]) - 1, 1)))) \
                if len(cfg["frames"]) > 1 else 1
            compare_interrdf(u, a, b, dr=cfg["dr"], rmax=rmax,
                             start=cfg["frames"][0], stop=cfg["frames"][-1] + 1,
                             step=step)
        if cfg.get("compare_engine"):
            compare_engine(u, a, b, dr=cfg["dr"], rmax=rmax, frames=cfg["frames"])
        return res

    return run_module("RDF（util_rdf.py / MDTransport 口径）", _run, specs,
                      sub="逐帧全原子对 + 当场最小镜像 + 同种粒子数−1 归一化",
                      default_frames=100, default_prefix="rdf_mdat")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(main))
