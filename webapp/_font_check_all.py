# -*- coding: utf-8 -*-
"""跑完全部分析并导出所有图，检查：① 无缺字 ② 后端是 Agg ③ 无其他警告。"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mdta.plotting import sanitize_text, set_agg_backend   # noqa: E402

backend = set_agg_backend()

from mdta.export import export_all                          # noqa: E402
from mdta.pipeline import DEFAULT_ORDER, Analyzer           # noqa: E402


def P(*a):
    print(*a, flush=True)


P(f"matplotlib 后端 = {backend}")
assert backend.lower() == "agg", backend

az = Analyzer(str(ROOT / "adk_oplsaa.tpr"), str(ROOT / "adk_oplsaa.xtc"))
az.set_frames(interval_ps=300)
az.auto_setup()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    results = az.run_all(DEFAULT_ORDER, verbose=False)
    outdir = ROOT / "_fontcheck_out"
    shutil.rmtree(outdir, ignore_errors=True)
    out = export_all(list(results.values()), str(outdir), formats=("png",),
                     excel=False, panel_pngs=True, verbose=False)

P(f"分析 {len(results)} 项，导出 PNG {len(out['png'])} 合并 + "
  f"{len(out['panel_png'])} 单面板 = {len(out['png']) + len(out['panel_png'])} 张")

missing = {}
other = {}
for w in caught:
    msg = str(w.message)
    (missing if "missing from font" in msg else other)[msg] = \
        missing.get(msg, 0) + (1 if "missing from font" in msg else 0) \
        if "missing from font" in msg else other.get(msg, 0) + 1

P(f"\n缺字警告 = {len(missing)}")
for m, n in missing.items():
    P(f"   ×{n}  {m}")
P(f"其他警告 = {len(other)}")
for m, n in other.items():
    P(f"   ×{n}  {m[:110]}")

# 直接检查所有会被画出来的文字：经 sanitize_text 后不应再有渲染不了的字符
texts = []
for r in results.values():
    texts.append(r.title)
    texts.extend(r.notes)
    for i in range(max(r.panel_count, 1)):
        p = r.panel(i)
        texts.extend([p.title, p.xlabel, p.ylabel])
    texts.extend(c.label for c in r.curves)

cps = __import__("mdta.plotting", fromlist=["x"])._font_coverage()
raw_bad = []          # 原始文本里首选字体没有的字符（绘制时会被替换掉，属正常）
still_bad = []        # sanitize_text 之后仍然渲染不了的字符（才是真问题）
for t in texts:
    for ch in str(t):
        if ch in "\n\t":
            continue
        if ord(ch) not in cps:
            raw_bad.append((ch, hex(ord(ch))))
            if ch in sanitize_text(ch):
                still_bad.append((ch, hex(ord(ch)), str(t)[:40]))
P(f"\n扫描 {len(texts)} 条文本：")
P(f"   首选字体缺字的字符 {len(set(raw_bad))} 个 "
  f"{sorted({b[0] for b in raw_bad})}（绘制时会替换成等价写法）")
P(f"   sanitize_text 之后仍无法渲染的字符 {len(still_bad)} 个 {still_bad[:5]}")
assert not missing, missing
assert not still_bad, still_bad

# 抽查替换效果
demo = "界面位置 x₀；取向张量 uuᵀ；P₂ 二阶序参数；ρ θ σ Å ° ² × 保留"
P(f"\n替换示例:\n   原文: {demo}\n   绘制: {sanitize_text(demo)}")

shutil.rmtree(outdir, ignore_errors=True)
P("\n字体与后端检查通过。")
