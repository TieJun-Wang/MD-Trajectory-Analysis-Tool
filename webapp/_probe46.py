# -*- coding: utf-8 -*-
"""46 号膜体系对象盘点（精简版：只做必要步骤，逐行刷盘）。"""
import sys
import time
from collections import Counter
from pathlib import Path

LOG = Path(__file__).with_suffix(".log")
LOGF = LOG.open("w", encoding="utf-8", buffering=1)
T0 = time.time()


def say(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    LOGF.write(line + "\n")
    sys.stderr.write(line + "\n")


sys.path.insert(0, r"C:\temp\MDT")
import numpy as np                                              # noqa: E402
from mdta.io import load_trajectory                             # noqa: E402

D = Path(r"C:\temp\dataset\46")
say("开始加载 input_46.gro + traj_46.xtc ...")
mdt = load_trajectory(str(D / "input_46.gro"), str(D / "traj_46.xtc"))
u = mdt.universe
say(f"加载完成：原子 {u.atoms.n_atoms:,}  残基 {u.residues.n_residues:,}")

say(f"  帧数 {len(u.trajectory):,}")
say(f"  dt  {getattr(mdt, 'dt_ps', float('nan')):.2f} ps")
say(f"  盒  {[round(float(x), 3) for x in u.dimensions]}")
for a in ("molnums", "segids", "chainIDs", "elements", "charges", "masses"):
    say(f"  属性 {a:9} {'有' if hasattr(u.atoms, a) else '无'}")
try:
    say(f"  bonds     {len(u.bonds):,}")
except Exception as e:  # noqa: BLE001
    say(f"  bonds     无 ({type(e).__name__})")
say(f"  猜键信息  {getattr(mdt, 'bond_guess_info', None)}")

say("统计残基名 ...")
rn = Counter(u.residues.resnames)
say(f"  残基名 {len(rn)} 种，前 20:")
for name, n in rn.most_common(20):
    say(f"    {name:8} {n:6}")

say("计算连通分量（分子）...")
frags = u.atoms.fragments
sizes = np.array([len(f) for f in frags])
say(f"  分子数 {len(frags):,}  原子/分子 min={sizes.min()} max={sizes.max():,} "
    f"中位={int(np.median(sizes))}")

say("按首个残基名给分子分组 ...")
per = {}
for f in frags:
    names = f.resnames
    key = names[0] if len(names) else "?"
    per.setdefault(key, []).append(len(f))
for key, lst in sorted(per.items(), key=lambda kv: -len(kv[1]))[:20]:
    arr = np.array(lst)
    say(f"  {key:8} {len(lst):6} 个分子  原子/分子 {arr.min()}~{arr.max()} "
        f"(中位 {int(np.median(arr))})")

say("完成")
LOGF.close()
