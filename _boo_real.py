import sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\temp\MDT")
from mdta.pipeline import Analyzer
from mdta.qc import checks_digest, worst_level

az = Analyzer(r"C:\temp\dataset\PEG_parametrization\eqpbc.gro",
              r"C:\temp\dataset\PEG_parametrization\trajpbc.xtc")
az.auto_setup()
az.frames = az.require_frames().head(3)
print(f"体系 {az.universe.atoms.n_atoms} 原子，取 3 帧\n", flush=True)
for name in ("boo", "crystal"):
    t = time.time()
    r = az.run(name)
    dt = time.time() - t
    if r is None:
        print(f"--- {name}: 跳过", flush=True)
        continue
    print(f"--- {name}  ({dt:.1f}s) ---", flush=True)
    for k, v in r.summary.items():
        print(f"    {k} = {v}", flush=True)
    print(f"    曲线 {len(r.curves)} 条", flush=True)
    print(f"    QC: {worst_level(r.checks)} {checks_digest(r.checks)}", flush=True)
    for c in r.checks:
        print(f"      [{c['级别']}] {c['名称']}: {c['结论']}", flush=True)
    print("    说明:", flush=True)
    for nt in r.notes[:4]:
        print(f"      · {nt[:110]}", flush=True)
    print(flush=True)
