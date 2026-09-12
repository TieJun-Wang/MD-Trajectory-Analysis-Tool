# -*- coding: utf-8 -*-
r"""真实 HTTP 下的实时运行验证（不用 TestClient，直接打 127.0.0.1:8000）。

跑的是 Abeta_4_16_Cu 板层体系（40 896 原子）——它是"跑起来最慢"的一个，
最能体现"边算边出"的价值：以前要干等 20 多秒才有第一张图。

用法:  python webapp\_test_live_http.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"

pass_n = 0
fails: list[str] = []


def ok(cond: bool, label: str, extra: str = "") -> None:
    global pass_n
    if cond:
        pass_n += 1
        print(f"  ✓ {label}" + (f"  {extra}" if extra else ""), flush=True)
    else:
        fails.append(label)
        print(f"  ✗ {label}" + (f"  {extra}" if extra else ""), flush=True)


def req(method: str, path: str, body=None, timeout: float = 120.0):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


D = ROOT / "Abeta_4_16_Cu" / "Abeta_4_16_Cu"
TOP, XTC = str(D / "min_nopbc.gro"), str(D / "md_dt200.xtc")

print("=" * 74)
print("实时运行验证（真实 HTTP）—— Abeta_4_16_Cu 板层体系")
print("=" * 74, flush=True)

h = req("GET", "/api/health")
ok(h["ok"] and h["frontend_built"], "服务在线且前端已构建", f"v{h['version']}")

sid = req("POST", "/api/session", {"topology": TOP, "trajectory": XTC})["sid"]
print(f"会话 {sid}\n", flush=True)

body = {
    "which": ["rg", "ree", "dihedral", "density", "rdf", "contact",
              "interface", "orientation", "order", "msd"],
    "frames": {"interval_ps": 2000},
    "params": {"interface": {"axis": 2, "nbins": 120},
               "rdf": {"rmax": 9.0, "nbins": 90}},
    "selection": {"primary": {"mode": "auto"}, "components": ["water", "polymer"]},
}

t0 = time.time()
st = req("POST", f"/api/session/{sid}/run/start", body)
start_dt = time.time() - t0
ok(start_dt < 3.0, "POST /run/start 立即返回", f"{start_dt * 1000:.0f} ms")
ok(st["status"] == "running" and st["n_total"] == 10,
   "返回初始状态与总数", f'{st["status"]} n_total={st["n_total"]}')

print("\n  轮询（每 400 ms），记录每项结果到达的时刻：", flush=True)
got = 0
arrivals: list[tuple[float, str]] = []
t0 = time.time()
while True:
    time.sleep(0.4)
    p = req("GET", f"/api/session/{sid}/run/progress?after={got}")
    for name in (p.get("results") or {}):
        arrivals.append((time.time() - t0, name))
        title = (p.get("titles") or {}).get(name, name)
        print(f"    {arrivals[-1][0]:6.2f}s  {name:12} {title}", flush=True)
    got += len(p.get("results") or {})
    if p["status"] != "running":
        total_dt = time.time() - t0
        break
    if time.time() - t0 > 300:
        ok(False, "轮询超时")
        total_dt = time.time() - t0
        break

print(flush=True)
ok(p["status"] == "done", "最终状态 done", p["status"])
ok(got == 10, "收到全部 10 项结果", str(got))
ok(len({n for _, n in arrivals}) == 10, "结果无重复")

first = arrivals[0][0] if arrivals else total_dt
ok(first < total_dt * 0.35, "第一项结果在总时长的 1/3 之前就出现了",
   f"首项 {first:.1f}s / 总 {total_dt:.1f}s")
spread = arrivals[-1][0] - arrivals[0][0] if len(arrivals) > 1 else 0.0
ok(spread > 3.0, "结果分散在整段时间里（不是最后一次性到达）",
   f"首尾相差 {spread:.1f}s")

# 关键：第一项到达时，用户已经能看图了
ok(first < 8.0, "第一张图在 8 秒内可用", f"{first:.1f}s")

print("\n  界面结果抽查：", flush=True)
full = req("GET", f"/api/session/{sid}/run/progress?after=0")
iface = full["results"].get("interface", {}).get("summary", {})
for k in ("界面位置 (Å)", "界面宽度 10-90 (Å)", "界面2 界面位置 (Å)",
          "界面判据可靠性"):
    if k in iface:
        v = iface[k]
        print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}",
              flush=True)
ok("界面判据可靠性" in iface, "界面结果完整送达")

print("\n  汇总表（随着结果增加而增长）：", flush=True)
print(f"    最终 {len(full['summary_rows'])} 行（应为 10）", flush=True)
ok(len(full["summary_rows"]) == 10, "汇总表 10 行")

req("DELETE", f"/api/session/{sid}")
print("\n" + "=" * 74)
print(f"通过 {pass_n} 项" + (f"，失败 {len(fails)} 项: {' | '.join(fails)}"
                           if fails else "，全部通过"))
print("=" * 74)
sys.exit(1 if fails else 0)
