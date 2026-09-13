import sys
sys.path.insert(0, r"C:\temp\MDT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from mdta.io import load_trajectory

for tag, top in (("AdK(.tpr)", "adk_oplsaa.tpr"),
                 ("46(.gro)", r"C:\temp\dataset\46\input_46.gro")):
    try:
        mdt = load_trajectory(top); u = mdt.universe
        try:
            q = np.asarray(u.atoms.charges, dtype=float)
            print(f"  {tag}: 电荷可用 ✓  总和={q.sum():+.4f} e  "
                  f"非零原子 {int((q!=0).sum()):,}/{q.size:,}  范围 [{q.min():+.3f},{q.max():+.3f}]")
            # 按残基名汇总净电荷，看能否直接做物种电导
            import collections
            agg = collections.defaultdict(float)
            for a in u.atoms:
                agg[str(a.resname)] += float(a.charge)
            top5 = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:6]
            print(f"     主要物种净电荷: " + ", ".join(f"{k}={v:+.2f}" for k, v in top5))
        except Exception as exc:
            print(f"  {tag}: 电荷不可用 — {type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"  {tag}: 加载失败 {exc}")
