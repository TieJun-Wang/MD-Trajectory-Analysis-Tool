import sys
sys.path.insert(0, r"C:\temp\MDT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from mdta.analysis import dynamics as dyn
from mdta.io import load_trajectory
from mdta.preprocess import select_frames
from mdta.selection import select

mdt = load_trajectory("adk_oplsaa.tpr", "adk_oplsaa.xtc")
u = mdt.universe
w = select(u, category="water")
sel = select_frames(mdt.times_ps)
idx = np.asarray(sel.indices, dtype=int)

# 故意"不解 PBC"：逐帧取水分子质心，但**不做跨帧最小镜像累加**
series, boxes, meta = dyn.particle_series(mdt, w, idx, object="molecule")
naive = dyn.msd_fft(series[:, :, 0]) + dyn.msd_fft(series[:, :, 1]) + dyn.msd_fft(series[:, :, 2])
# 正确做法：先做最小镜像累加再算
unw, _ = dyn._unwrap_series(series, boxes)
good = dyn.msd_fft(unw[:, :, 0]) + dyn.msd_fft(unw[:, :, 1]) + dyn.msd_fft(unw[:, :, 2])

their = [595.0873,1163.9418,1732.3644,2297.1367,2861.8252,3419.4478,3973.8428,4517.2607,5062.3350]
t = np.arange(100, 1000, 100, dtype=float)
print(f"  {'τ(ps)':>6}{'解PBC(我的)':>14}{'不解PBC':>12}{'他们的':>12}{'比(不解/解)':>13}{'比(他们/解)':>13}")
for i in range(len(t)):
    print(f"  {int(t[i]):>6}{good[i]:>14.2f}{naive[i]:>12.2f}{their[i]:>12.2f}"
          f"{naive[i]/max(good[i],1e-9):>13.2f}{their[i]/max(good[i],1e-9):>13.2f}")
d_naive = (naive[7]-naive[0])/700/6*1e-8
d_good = (good[7]-good[0])/700/6*1e-8
print(f"\n  D(解PBC)  = {d_good:.4e} m²/s   ← 文献水 2.5–4.5e-9")
print(f"  D(不解PBC)= {d_naive:.4e} m²/s")
print(f"  他们报的  = {(4517.2607-595.0873)/700/6*1e-8:.4e} m²/s")
