# -*- coding: utf-8 -*-
r"""量出 1.0.1 改动前后的真实数值差异（框架要求：旧/新差异说明）。

对照：
- RDF：旧 = 只管 total（1.0.0），新 = 默认 inter（分子间）
- MSD：旧 = 原子、暴力 O(N_帧²)，新 = 分子质心、FFT、逐轴各向异性

用法: python -u _diff_101.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                          # noqa: E402

from mdta.analysis import dynamics as dyn                   # noqa: E402
from mdta.analysis import interface as ifc                  # noqa: E402
from mdta.io import load_trajectory                         # noqa: E402
from mdta.preprocess import select_frames                   # noqa: E402
from mdta.selection import select                           # noqa: E402


def curve(res, label):
    for c in res.curves:
        if c.label == label:
            return c
    raise KeyError(label)


def rdf_table():
    print("=" * 78)
    print("RDF：旧口径（total） vs 新口径（inter，默认）")
    print("=" * 78)

    # ---------------------------------------------------- AdK 水（全部分子）
    mdt = load_trajectory("adk_oplsaa.tpr", "adk_oplsaa.xtc")
    u = mdt.universe
    w = select(u, category="water")
    sel = select_frames(mdt.times_ps, max_frames=4)
    print(f"\n【AdK 水，{w.n_atoms:,} 原子 / {len(np.unique(np.asarray(w.molnums))):,} 个分子】"
          f"  rmax=6 Å")
    res = {}
    for mode in ("total", "inter", "intra"):
        t0 = time.time()
        res[mode] = ifc.analyze_rdf(mdt, {"W": w}, sel, rmax=6.0, nbins=120, mode=mode)
        print(f"   {mode:<6} 用时 {time.time() - t0:.2f}s  曲线 {len(res[mode].curves)} 条")
    near = curve(res["total"], "W-W").x < 1.4
    rows = [
        ("r<1.4 Å 区间 g_max（共价 O–H 峰）",
         float(curve(res["total"], "W-W").y[near].max()),
         float(curve(res["inter"], "W-W").y[near].max())),
        ("第一峰位置 (Å)",
         res["total"].summary.get("W-W 第一峰位置 (Å)"),
         res["inter"].summary.get("W-W 第一峰位置 (Å)")),
        ("第一峰高度 g_max",
         res["total"].summary.get("W-W 第一峰高度 g_max"),
         res["inter"].summary.get("W-W 第一峰高度 g_max")),
        ("配位数 (第一壳层)",
         res["total"].summary.get("W-W 配位数 (第一壳层)"),
         res["inter"].summary.get("W-W 配位数 (第一壳层)")),
        ("归一化分母（有序原子对）",
         res["total"].summary.get("W-W 归一化分母 (有序原子对)"),
         res["inter"].summary.get("W-W 归一化分母 (有序原子对)")),
        ("分子内配对占比",
         res["total"].summary.get("W-W 分子内配对占比"),
         res["inter"].summary.get("W-W 分子内配对占比")),
    ]
    print(f"\n   {'指标':<34}{'旧 total':>16}{'新 inter':>16}")
    for name, old, new in rows:
        fo = f"{old:,.4g}" if isinstance(old, float) else str(old)
        fn = f"{new:,.4g}" if isinstance(new, float) else str(new)
        print(f"   {name:<34}{fo:>16}{fn:>16}")

    # ---------------------------------------------------- 46 体系 DPE
    print("\n【46 体系 DPE（180 个分子 × 70 原子）】 rmax=10 Å，3 帧")
    m46 = load_trajectory(r"C:\temp\dataset\46\input_46.gro")
    m46.trajectory = r"C:\temp\dataset\46\traj_46.xtc"
    u46 = m46.universe
    dpe = u46.select_atoms("resname DPE")
    n_mol = len(np.unique(np.asarray(dpe.molnums)))
    sel46 = select_frames(m46.times_ps, max_frames=3)
    print(f"   DPE 原子 {dpe.n_atoms:,}，分子 {n_mol:,}")
    out = {}
    for mode in ("total", "inter"):
        t0 = time.time()
        out[mode] = ifc.analyze_rdf(m46, {"DPE": dpe}, sel46, rmax=10.0, nbins=100,
                                    mode=mode)
        print(f"   {mode:<6} 用时 {time.time() - t0:.2f}s")
    print(f"\n   {'指标':<34}{'旧 total':>16}{'新 inter':>16}")
    for name, key in (("第一峰位置 (Å)", "DPE-DPE 第一峰位置 (Å)"),
                      ("第一峰高度 g_max", "DPE-DPE 第一峰高度 g_max"),
                      ("第一峰是否显著", "DPE-DPE 第一峰是否显著"),
                      ("配位数 (第一壳层)", "DPE-DPE 配位数 (第一壳层)"),
                      ("归一化分母", "DPE-DPE 归一化分母 (有序原子对)"),
                      ("分子内配对占比", "DPE-DPE 分子内配对占比")):
        old = out["total"].summary.get(key)
        new = out["inter"].summary.get(key)
        fo = f"{old:,.4g}" if isinstance(old, float) else str(old)
        fn = f"{new:,.4g}" if isinstance(new, float) else str(new)
        print(f"   {name:<34}{fo:>16}{fn:>16}")
    print("\n   新口径下为何没有配位数：")
    for note in out["inter"].notes:
        if "没有显著的结构峰" in note:
            print(f"     {note}")
            break


def msd_table():
    print("\n" + "=" * 78)
    print("MSD：旧（原子、暴力法） vs 新（分子质心、FFT、逐轴）")
    print("=" * 78)

    mdt = load_trajectory("adk_oplsaa.tpr", "adk_oplsaa.xtc")
    u = mdt.universe
    ow = select(u, category="water").select_atoms("name OW")
    sel = select_frames(mdt.times_ps, max_frames=10)
    print(f"\n【AdK 水氧 OW，{ow.n_atoms:,} 原子 = {ow.n_atoms:,} 个分子，10 帧】")
    t0 = time.time()
    lag, m_atom, diag_a = dyn.compute_msd(mdt, ow, sel, object="atom")
    t_atom = time.time() - t0
    t0 = time.time()
    lag, m_com, diag_c = dyn.compute_msd(mdt, ow, sel, object="molecule")
    t_com = time.time() - t0
    d_atom = dyn.diffusion_coefficient(lag, m_atom)["D (m²/s)"]
    d_com = dyn.diffusion_coefficient(lag, m_com)["D (m²/s)"]
    print(f"   原子   D = {d_atom:.4e} m²/s   用时 {t_atom:.2f}s  "
          f"MSD(最大τ) = {m_atom[-1]:.2f} Å²")
    print(f"   质心   D = {d_com:.4e} m²/s   用时 {t_com:.2f}s  "
          f"MSD(最大τ) = {m_com[-1]:.2f} Å²")
    print(f"   → OW 每分子恰好 1 个原子，两者应几乎相同（D 相对差 "
          f"{abs(d_com - d_atom) / d_atom * 100:.1f}%）")

    # 46 体系：DPE 分子扩散 + 各向异性（含真实规模与耗时）
    print("\n【46 体系 DPE 分子质心扩散，1000 帧 × 10 ps = 10 ns】")
    m46 = load_trajectory(r"C:\temp\dataset\46\input_46.gro")
    m46.trajectory = r"C:\temp\dataset\46\traj_46.xtc"
    u46 = m46.universe
    dpe = u46.select_atoms("resname DPE")
    sel46 = select_frames(m46.times_ps, max_frames=1000)
    t0 = time.time()
    res = dyn.analyze_msd(m46, {"DPE": dpe}, sel46, object="molecule",
                          remove_drift=True)
    el = time.time() - t0
    keys = ["DPE 追踪对象", "DPE 追踪粒子数", "DPE 去漂移", "DPE 算法",
            "DPE D (m²/s)", "DPE α (log-log 斜率)", "DPE 拟合可靠性",
            "DPE 最小镜像失效比例",
            "DPE D∥ (m²/s)", "DPE D⊥ (m²/s)", "DPE 各向异性 D∥/D⊥",
            "DPE MSD(最大 τ) (Å²)", "DPE MSD∥(最大 τ) (Å²)", "DPE MSD⊥(最大 τ) (Å²)"]
    print(f"   用时 {el:.1f}s（{len(sel46.indices):,} 帧）")
    for k in keys:
        v = res.summary.get(k)
        if isinstance(v, float):
            v = f"{v:.6g}"
        print(f"     {k:<28} {v}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("rdf", "both"):
        rdf_table()
    if which in ("msd", "both"):
        msd_table()
    print("\nDONE")
