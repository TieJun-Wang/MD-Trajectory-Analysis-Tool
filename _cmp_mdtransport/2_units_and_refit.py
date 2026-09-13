# -*- coding: utf-8 -*-
import io, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = r"C:\temp\MDT_project\MDTransport-main\temp_file\results_diffusion\diffusion_summary.csv"
raw = io.open(p, "rb").read()
txt = raw.decode("utf-8", errors="replace")
hdr = txt.splitlines()[0]
print("表头(UTF-8 正确解码):", repr(hdr))
rows = [l.split(",") for l in txt.splitlines()[1:] if l.strip()]
sol = [r for r in rows if r[0] == "SOL"][0]
print("SOL 行:", [f"{c}" for c in sol])

# 用他们自己的 msd_SOL.csv 全 8 点做 OLS，反推 D（Å²/ps -> m²/s 用 1e-8）
msd = [595.0873,1163.9418,1732.3644,2297.1367,2861.8252,3419.4478,3973.8428,4517.2607]
t = np.arange(100, 801, 100, dtype=float)
slope, icept = np.polyfit(t, np.asarray(msd), 1)
D_a2ps = slope / 6.0
print(f"\n他们的 MSD(100-800 ps) OLS 斜率 = {slope:.4f} Å²/ps -> D = {D_a2ps:.6f} Å²/ps")
for lab, scale in (("10^-9", 1e-9), ("10^-10", 1e-10), ("10^-12", 1e-12)):
    print(f"  若表头单位是 {lab} m²/s：D = {D_a2ps*1e-8/scale:.3f}")
print("  表里 SOL 的 D 值 =", sol[6])
