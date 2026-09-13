# -*- coding: utf-8 -*-
r"""核查"接触概率恒为 1"：把接触分析里所有"概率类"指标摊开看。

同时手工算出**平均接触数（每个 A 原子平均接触多少个 B 原子）**——这才是
有区分度的物理量，用来对比"概率"指标为什么不成立。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                              # noqa: E402

from mdta.analysis import interface as ifc                      # noqa: E402
from mdta.io import load_trajectory                             # noqa: E402
from mdta.preprocess import select_frames                       # noqa: E402
from mdta.selection import component_groups                     # noqa: E402

CASES = [
    ("AdK", "adk_oplsaa.tpr", "adk_oplsaa.xtc", 6),
    ("糖蛋白+DOL", r"C:\temp\dataset\md_biopolymer_nowater\md_biopolymer_nowater.tpr", None, 6),
    ("46", r"C:\temp\dataset\46\input_46.gro", r"C:\temp\dataset\46\traj_46.xtc", 4),
]


def probe(label, top, xtc, max_frames):
    print("=" * 78)
    print(f"【{label}】")
    mdt = load_trajectory(top)
    if xtc:
        mdt.trajectory = xtc
    u = mdt.universe
    comps = component_groups(u)
    names = list(comps)
    print("  组分: " + " / ".join(f"{k}({v.n_atoms:,})" for k, v in comps.items()))
    if len(names) < 2:
        print("  组分不足 2 个，跳过")
        return
    a, b = comps[names[0]], comps[names[1]]
    sel = select_frames(mdt.times_ps, max_frames=max_frames)
    res = ifc.analyze_contacts(mdt, a, b, sel, cutoff=5.0)
    print(f"\n  接触对：{names[0]} × {names[1]}   cutoff=5 Å   "
          f"{len(sel.indices)} 帧")
    for k, v in res.summary.items():
        if isinstance(v, float):
            v = f"{v:,.6g}"
        print(f"    {k:<34} {v}")

    # ---- 手工算"平均接触数"（每个 A 原子平均接触的 B 原子数）----
    from MDAnalysis.lib.distances import capped_distance
    from mdta.analysis.base import frame_iterator
    counts = np.zeros(a.n_atoms, dtype=float)
    nfr = 0
    for _frame, _t in frame_iterator(mdt, sel):
        pa = np.asarray(a.positions, dtype=float)
        pb = np.asarray(b.positions, dtype=float)
        pr, _ = capped_distance(pa, pb, max_cutoff=5.0, box=u.dimensions,
                                return_distances=True)
        if pr.size:
            np.add.at(counts, pr[:, 0], 1)
        nfr += 1
    per_atom = counts / max(nfr, 1)
    print(f"\n    —— 有区分度的量（手工核算）——")
    print(f"    每个 A 原子的平均接触数         {per_atom.mean():.3f} "
          f"(min {per_atom.min():.2f} / 中位 {np.median(per_atom):.2f} / max {per_atom.max():.2f})")
    print(f"    接触数恰为 0 的 A 原子比例       {np.mean(per_atom == 0) * 100:.1f}%")
    print(f"    平均接触数 < 0.5 的原子比例      {np.mean(per_atom < 0.5) * 100:.1f}%")
    print(f"    → 概率类指标把上面这些差异**全部压平**：只要 >0 就算 1")


if __name__ == "__main__":
    for label, top, xtc, nf in CASES:
        try:
            probe(label, top, xtc, nf)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] 失败: {type(exc).__name__}: {exc}")
        print()
    print("DONE")
