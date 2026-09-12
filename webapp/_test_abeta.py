# -*- coding: utf-8 -*-
"""用真实数据集 Abeta_4_16_Cu 验证 gro/pdb/top 三种拓扑路径。"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mdta.io import TrajectoryError, load_trajectory
from mdta.plotting import set_agg_backend
from mdta.pipeline import DEFAULT_ORDER, Analyzer
from mdta.selection import component_groups, largest_chains, list_chains

set_agg_backend()
D = ROOT / "Abeta_4_16_Cu" / "Abeta_4_16_Cu"
XTC = str(D / "md_dt200.xtc")


def P(*a):
    print(*a, flush=True)


P("=" * 76)
P("数据集 Abeta_4_16_Cu —— 拓扑格式验证")
P("=" * 76)

# ---------------------------------------------------------------- 错误路径
P("\n【A】用错文件时是否给得出有用的提示")

P("\n1) 用只有 216 原子的 pdb 配 40896 原子的 xtc：")
try:
    load_trajectory(str(D / "ab_4_16_cu.pdb"), XTC).n_atoms
    P("   ✗ 竟然成功")
except TrajectoryError as e:
    P("   ✓ 报错：")
    for line in str(e).splitlines():
        P("     " + line)

P("\n2) 用 GROMACS 力场拓扑 topol.top：")
try:
    load_trajectory(str(D / "topol.top"), XTC).n_atoms
    P("   ✗ 竟然成功")
except TrajectoryError as e:
    P("   ✓ 报错：")
    for line in str(e).splitlines():
        P("     " + line)

# ---------------------------------------------------------------- 正确路径
P("\n\n【B】用 min_nopbc.gro（40896 原子）配 md_dt200.xtc")

t0 = time.time()
mdt = load_trajectory(str(D / "min_nopbc.gro"), XTC)
u = mdt.universe
P(f"   加载成功，用时 {time.time() - t0:.1f}s")
P(f"   原子 {u.atoms.n_atoms:,}  残基 {u.residues.n_residues:,}  帧 {len(u.trajectory):,}")
P(f"   盒 {[round(float(x), 2) for x in u.dimensions]}  "
  f"（{u.dimensions[0]:.0f}×{u.dimensions[1]:.0f}×{u.dimensions[2]:.0f}）")
P(f"   时间间隔 {mdt.dt_ps:.1f} ps，总时长 {mdt.total_time_ps / 1000:.1f} ns")

gi = mdt.bond_guess_info
P(f"\n   猜键结果: {gi}")
assert gi and gi.get("bonds"), gi

P(f"\n   组分识别:")
for k, ag in component_groups(u).items():
    P(f"     {k:22} {ag.n_atoms:7,} 原子 / {ag.residues.n_residues:6,} 残基")

P(f"\n   链类型（前 8）:")
for c in sorted(list_chains(u), key=lambda c: -c.n_atoms)[:8]:
    P(f"     {c}")

big = largest_chains(u, n=3, min_atoms=50)
P(f"\n   最大链:")
for f in big:
    P(f"     {f.n_atoms:,} 原子 / {len(set(f.resids.tolist())):,} 残基  "
      f"resname={f.resnames[0]}")

# ---------------------------------------------------------------- 跑分析
P("\n\n【C】跑全部分析（每 10 帧取 1 帧 = 51 帧）")
az = Analyzer(str(D / "min_nopbc.gro"), XTC)
az.set_frames(interval_ps=2000)
az.auto_setup()
P(f"   自动主链: {az.primary_label} / {az.primary.n_atoms:,} 原子")
P(f"   组分: {list(az.components)}")

t0 = time.time()
res = az.run_all(list(DEFAULT_ORDER), verbose=False, raise_errors=False)
P(f"\n   完成 {len(res)}/10 项，用时 {time.time() - t0:.1f}s")
P(f"   {'分析':12} {'面板':>4} {'曲线':>4} {'统计':>4}   用时")
for name in DEFAULT_ORDER:
    if name in res:
        r = res[name]
        P(f"   {name:12} {r.panel_count:>4} {len(r.curves):>4} "
          f"{len(r.summary):>4}   {az.timings.get(name, 0):.1f}s")
    else:
        P(f"   {name:12}    —— 未产出")

P("\n   关键结果:")
for k in ("Rg mean", "Rg std", "Rg 块平均标准误"):
    if k in res.get("rg", {}).summary if "rg" in res else {}:
        P(f"     {k} = {res['rg'].summary[k]:.4f}")

if "density" in res:
    den = res["density"]
    P("     密度:")
    for k, v in den.summary.items():
        if "体相密度" in k or "平均密度" in k:
            P(f"       {k} = {v:.4f} g/cm³")
    P(f"     组分: {[c.label for c in den.curves if c.panel == 0]}")

if "interface" in res:
    it = res["interface"]
    P("     界面（板层体系应列出每个界面并给出 erf 拟合）:")
    for k, v in it.summary.items():
        P(f"       {k} = {v:.4f}" if isinstance(v, float) else f"       {k} = {v}")

if "rdf" in res:
    rdf = res["rdf"]
    pk = {k: v for k, v in rdf.summary.items() if "第一峰位置" in k}
    P(f"     RDF 峰（{len(pk)} 对）:")
    for k, v in list(pk.items())[:6]:
        P(f"       {k.replace(' 第一峰位置 (Å)', ''):32} {v:.2f} Å")

if "msd" in res:
    P("     MSD:")
    for k, v in res["msd"].summary.items():
        if k.endswith("D (m²/s)") or "失效比例" in k:
            P(f"       {k} = {v}")

P("\n" + "=" * 76)
