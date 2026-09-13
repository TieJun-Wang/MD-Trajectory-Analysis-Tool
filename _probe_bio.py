"""Inspect md_biopolymer_nowater and run the new estimate -> shortest-first flow."""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\temp\MDT")
from mdta.pipeline import DEFAULT_ORDER, ANALYSIS_TITLES, Analyzer

TOP = r"C:\temp\dataset\md_biopolymer_nowater\md_biopolymer_nowater.tpr"
XTC = r"C:\temp\dataset\md_biopolymer_nowater\md_biopolymer_nowater.xtc"
for p in (TOP, XTC):
    print(f"  {os.path.basename(p):32s} {os.path.getsize(p)/1048576:9.1f} MB", flush=True)

t = time.time()
az = Analyzer(TOP, XTC)
az.auto_setup()
print(f"\n== 体系（打开 + 识别组分 {time.time()-t:.1f}s）==", flush=True)
i = az.info
print(f"  原子 {i.n_atoms:,} / residue {i.n_residues:,} / 段 {i.n_segments} / 帧 {i.n_frames:,}", flush=True)
print(f"  时间 {i.first_time_ps:.1f} – {i.last_time_ps:.1f} ps，dt={i.dt_ps} ps", flush=True)
print(f"  盒 {[round(float(x),2) for x in i.box_dimensions]} Å  体积 {i.box_volume:,.0f} Å³", flush=True)
print(f"  键/角/二面角 {i.n_bonds}/{i.n_angles}/{i.n_dihedrals}  电荷 {'有' if i.has_charges else '无'}"
      f"  净电荷 {i.total_charge_e:+.4f} e", flush=True)
print(f"  组分: {[(n, g.n_atoms) for n, g in az.components.items()]}", flush=True)
print(f"  主链: {az.primary_label}", flush=True)

sel = az.require_frames()
print(f"  帧选择: {sel.describe()}", flush=True)

print(f"\n== 预估全部 {len(DEFAULT_ORDER)} 项 ==", flush=True)
t = time.time()
est = az.estimate_times(list(DEFAULT_ORDER),
                        on_item=lambda a, b, nm: print(f"    [{a+1}/{b}] {nm}", flush=True))
print(f"  预估阶段耗时 {time.time()-t:.1f}s", flush=True)

rows = sorted(((r["est_sec"] if r["est_sec"] is not None else float("inf"), n, r)
               for n, r in est.items()))
tot = 0.0
print(f"\n  {'分析项':22s} {'预估':>10s}   说明", flush=True)
for e, n, r in rows:
    if r["est_sec"] is not None:
        tot += r["est_sec"]
    print(f"  {ANALYSIS_TITLES.get(n, n):22s} {str(r['est_sec']):>10s}   {r['note']}", flush=True)
print(f"\n  预计总用时 ≈ {tot:.0f}s ({tot/60:.1f} min)", flush=True)
print(f"  执行顺序（短→长）: {[ANALYSIS_TITLES.get(n, n) for _, n, _ in rows]}", flush=True)
