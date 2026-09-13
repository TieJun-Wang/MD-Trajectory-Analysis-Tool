import sys
sys.path.insert(0, r"C:\temp\MDT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from mdta.analysis import dynamics as dyn
from mdta.io import load_trajectory
from mdta.preprocess import select_frames

mdt = load_trajectory("adk_oplsaa.tpr", "adk_oplsaa.xtc")
sel = select_frames(mdt.times_ps)
their = {"SOL": [595.0873,1163.9418,1732.3644,2297.1367,2861.8252,3419.4478,3973.8428,4517.2607,5062.3350],
         "ALA": [7.3435,10.2876,14.7542,18.4960,30.7659,31.0848,45.1164,57.0954,78.5772],
         "ALA900": 78.5772, "NA+900": 2775.4165}
print(f"  {'物种':<6}{'我的 MSD(900 ps)':>18}{'他们的':>12}{'比值':>8}")
for nm, key in (("SOL","SOL"), ("ALA","ALA")):
    ag = mdt.universe.select_atoms(f"resname {nm}")
    if ag.n_atoms == 0:
        print(f"  {nm}: 找不到"); continue
    lag, msd, diag = dyn.compute_msd_axes(mdt, ag, sel, object="molecule", remove_drift=False)
    mine = float(msd["total"][-1]); theirs = their[key][-1]
    print(f"  {nm:<6}{mine:>18.2f}{theirs:>12.2f}{theirs/max(mine,1e-9):>8.2f}")
    lag2, msd2, _ = dyn.compute_msd_axes(mdt, ag, sel, object="atom", remove_drift=False)
    print(f"  {nm:<6}{'(原子)':>12}{float(msd2['total'][-1]):>6.2f}")
# 逐个 τ 的比值（SOL）
ag = mdt.universe.select_atoms("resname SOL")
lag, msd, _ = dyn.compute_msd_axes(mdt, ag, sel, object="molecule", remove_drift=False)
print("\n  SOL 逐点比值:", [f"{their['SOL'][i]/msd['total'][i+1]:.2f}" for i in range(9)])
