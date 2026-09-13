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
print(f"AdK 轨迹: {len(u.trajectory)} 帧, dt = {u.trajectory.dt} ps, "
      f"times_ps = {np.asarray(mdt.times_ps).round(1).tolist()}")
w = select(u, category="water")
sel = select_frames(mdt.times_ps)
lag, msd, diag = dyn.compute_msd_axes(mdt, w, sel, object="molecule", remove_drift=False)
their = {100:595.0873,200:1163.9418,300:1732.3644,400:2297.1367,500:2861.8252,
         600:3419.4478,700:3973.8428,800:4517.2607,900:5062.3350}
print(f"\n  {'τ(ps)':>7}{'我的 MSD(Å²)':>14}{'他们的 MSD(Å²)':>16}{'比值':>8}")
for i, t in enumerate(lag):
    ti = int(round(t))
    if ti in their:
        print(f"  {ti:>7}{msd['total'][i]:>14.2f}{their[ti]:>16.2f}{their[ti]/max(msd['total'][i],1e-9):>8.1f}")
fit = dyn.diffusion_coefficient(lag, msd["total"], fit_fraction=(100/900, 800/900))
print(f"\n  我的水 D（同样 100–800 ps 窗口）= {fit['D (m²/s)']:.4e} m²/s，α = {fit['α (log-log 斜率)']:.3f}")
print(f"  他们表里 SOL 的 D 值 = 935.132（单位字符串在 CSV 里是乱码，无法确认量级）")
print(f"  他们自己的 msd_SOL.csv 反推 D = {((their[800]-their[100])/700/6)*1e-8:.4e} m²/s")
print(f"  文献水自扩散 ≈ 2.5–4.5e-9 m²/s")
