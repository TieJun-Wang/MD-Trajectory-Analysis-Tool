# -*- coding: utf-8 -*-
r"""阶段 3 实测：Rg 按分子、R_ee 用键图端原子，与 1.0.0 口径对比。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                              # noqa: E402

from mdta.analysis import conformation as conf                  # noqa: E402
from mdta.analysis.base import positions_for                    # noqa: E402
from mdta.io import load_trajectory                             # noqa: E402
from mdta.preprocess import select_frames                       # noqa: E402
from mdta.selection import component_groups, select             # noqa: E402


def adk():
    print("=" * 78)
    print("AdK 蛋白：链端识别（键图 vs 首尾原子）")
    print("=" * 78)
    mdt = load_trajectory("adk_oplsaa.tpr", "adk_oplsaa.xtc")
    u = mdt.universe
    prot = select(u, category="protein")
    info = conf.bond_graph_ends(prot)
    print(f"  蛋白 {prot.n_atoms} 原子，组分（molecules）数：" 
          f"{len(np.unique(np.asarray(prot.molnums)))}")
    print(f"  键图连通分量 {info['n_components']}，度为 1 的原子 {info['n_degree1']} 个，"
          f"可定义={info['ok']}")
    old = (int(prot.indices[0]), int(prot.indices[-1]))
    new = (int(info["ends"][0]), int(info["ends"][1])) if info["ok"] else None
    print(f"  旧口径（首尾原子）  : {old}  "
          f"{u.atoms[old[0]].resname}{u.atoms[old[0]].resid}({u.atoms[old[0]].name}) – "
          f"{u.atoms[old[1]].resname}{u.atoms[old[1]].resid}({u.atoms[old[1]].name})")
    if new:
        print(f"  新口径（键图端原子）: {new}  "
              f"{u.atoms[new[0]].resname}{u.atoms[new[0]].resid}({u.atoms[new[0]].name}) – "
              f"{u.atoms[new[1]].resname}{u.atoms[new[1]].resid}({u.atoms[new[1]].name})")
    sel = select_frames(mdt.times_ps)
    for tag, kw in (("旧口径", {"ends": "selection"}), ("新口径", {})):
        r = conf.analyze_end_to_end(mdt, prot, sel, **kw)
        print(f"  {tag}: R_ee 平均 = {r.summary['R_ee mean']:.3f} Å，"
              f"min {r.summary['R_ee min']:.3f} / max {r.summary['R_ee max']:.3f} Å，"
              f"链端来源 = {r.summary['链端来源']}")
    r = conf.analyze_rg(mdt, prot, sel)
    print(f"  Rg = {r.summary['Rg mean']:.3f} Å，口径 = {r.summary['统计口径']}，"
          f"分子数 = {r.summary['分子数']}，面板 {len(r.panels)} 个")
    return r


def sys46():
    print("\n" + "=" * 78)
    print("46 体系：含 541 个分子的组分 —— Rg / R_ee 按分子")
    print("=" * 78)
    mdt = load_trajectory(r"C:\temp\dataset\46\input_46.gro")
    mdt.trajectory = r"C:\temp\dataset\46\traj_46.xtc"
    u = mdt.universe
    comps = component_groups(u)
    poly = comps["polymer"]
    sel = select_frames(mdt.times_ps, max_frames=4)
    n_mol = len(np.unique(np.asarray(poly.molnums)))
    print(f"  polymer 组分：{poly.n_atoms:,} 原子 / {n_mol:,} 个分子")

    t0 = time.time()
    r = conf.analyze_rg(mdt, poly, sel, label="polymer")
    el = time.time() - t0
    print(f"\n  【Rg】用时 {el:.1f}s，面板 {len(r.panels)} 个")
    for k in ("统计口径", "分子数", "Rg mean", "单分子 Rg 平均 (Å)",
              "单分子 Rg 标准差 (Å)", "单分子 Rg 最小 (Å)", "单分子 Rg 最大 (Å)"):
        v = r.summary.get(k)
        print(f"     {k:<24} {v if not isinstance(v, float) else f'{v:.4f}'}")
    print(f"     → 整组 Rg / 单分子 Rg = "
          f"{r.summary['Rg mean'] / max(r.summary['单分子 Rg 平均 (Å)'], 1e-9):.2f} 倍"
          f"（1.0.0 只给前者，会被读成后者）")

    t0 = time.time()
    r2 = conf.analyze_end_to_end(mdt, poly, sel, label="polymer")
    el2 = time.time() - t0
    print(f"\n  【R_ee】用时 {el2:.1f}s，面板 {len(r2.panels)} 个")
    for k in ("R_ee 是否可定义", "链端来源", "参与统计的链数", "R_ee mean",
              "单链 R_ee 平均 (Å)", "单链 R_ee 标准差 (Å)"):
        v = r2.summary.get(k)
        print(f"     {k:<24} {v if not isinstance(v, float) else f'{v:.4f}'}")
    for n in r2.notes:
        print(f"     note: {n}")
    for n in r.notes:
        print(f"     Rg note: {n}")


if __name__ == "__main__":
    adk()
    sys46()
    print("\nDONE")
