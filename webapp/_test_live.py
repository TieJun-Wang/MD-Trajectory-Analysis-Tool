# -*- coding: utf-8 -*-
"""实时运行接口验证：结果必须**边算边出**，而不是最后一次性返回。

用 TestClient 打开一个真实体系，调用 /run/start，然后按 300 ms 轮询
/run/progress，记录每次拿到新结果的**时刻**。若所有结果都出现在同一时刻，
说明并没有真正实时更新。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient          # noqa: E402

from webapp.backend import app as appmod           # noqa: E402

client = TestClient(appmod.app)
TOP = str(ROOT / "adk_oplsaa.tpr")
XTC = str(ROOT / "adk_oplsaa.xtc")

pass_n = 0
fails: list[str] = []


def ok(cond: bool, label: str, extra: str = "") -> None:
    global pass_n
    if cond:
        pass_n += 1
        print(f"  ✓ {label}" + (f"  {extra}" if extra else ""))
    else:
        fails.append(label)
        print(f"  ✗ {label}" + (f"  {extra}" if extra else ""))


print("=" * 74)
print("实时运行接口验证（边算边出）")
print("=" * 74)

r = client.post("/api/session", json={"topology": TOP, "trajectory": XTC})
assert r.status_code < 400, r.text[:400]
sid = r.json()["sid"]
print(f"会话 {sid} 已打开")

body = {
    "which": ["rg", "ree", "dihedral", "density", "rdf", "contact",
              "interface", "orientation", "order", "msd"],
    "frames": {"interval_ps": 200},
    "params": {"rdf": {"rmax": 9.0, "nbins": 90},
               "interface": {"axis": 2, "nbins": 60}},
    "selection": {"primary": {"mode": "auto"}, "components": ["protein", "water"]},
}

print("\n[1] POST /run/start 必须立即返回")
t0 = time.time()
r = client.post(f"/api/session/{sid}/run/start", json=body)
start_dt = time.time() - t0
assert r.status_code < 400, r.text[:400]
st = r.json()
ok(start_dt < 2.0, "start 立即返回（未等分析跑完）", f"{start_dt * 1000:.0f} ms")
ok(st["status"] == "running", "初始状态为 running", st["status"])
ok(st["n_total"] == 10, "报告了总项数", str(st["n_total"]))
ok(len(st["pending"]) == 10, "报告了排队中的项", f"{len(st['pending'])} 项")

print("\n[2] 轮询 /run/progress，记录每项到达的时刻")
got = 0
arrivals: list[tuple[float, str]] = []
t0 = time.time()
poll = 0
while True:
    poll += 1
    time.sleep(0.3)
    p = client.get(f"/api/session/{sid}/run/progress",
                   params={"after": got}).json()
    fresh = list((p.get("results") or {}).keys())
    for name in fresh:
        arrivals.append((time.time() - t0, name))
    got += len(fresh)
    if poll == 1:
        ok(p["status"] == "running", "第一轮轮询时仍在运行")
    if p["status"] != "running":
        ok(p["status"] == "done", "最终状态为 done", p["status"])
        ok(p["n_done"] == 10, "完成 10 项", str(p["n_done"]))
        break
    if time.time() - t0 > 300:
        ok(False, "轮询超时", "> 300s")
        break
total_dt = time.time() - t0

print("\n    每项结果的到达时刻：")
for dt, name in arrivals:
    print(f"      {dt:6.2f}s  {name}")

ok(len(arrivals) == 10, "共收到 10 项结果", str(len(arrivals)))
ok(len({n for _, n in arrivals}) == 10, "结果互不重复")

# 关键断言：结果必须分散在不同时刻，而不是全部挤在最后一刻
spread = (arrivals[-1][0] - arrivals[0][0]) if len(arrivals) > 1 else 0.0
ok(spread > 0.5, "结果分批到达（真正的实时更新）",
   f"首尾相差 {spread:.2f}s")
early = [n for dt, n in arrivals if dt < total_dt - 1.0]
ok(len(early) >= 3, "多数结果在全部跑完之前就已送达",
   f"{len(early)}/10 项在结束前 1s 以上到达")

print("\n[3] 增量拉取：after=已收到条数 时不重复发送")
p_again = client.get(f"/api/session/{sid}/run/progress",
                     params={"after": 10}).json()
ok(not p_again["results"], "after=10 时不再重传结果",
   f"results 条数 {len(p_again['results'])}")
p_mid = client.get(f"/api/session/{sid}/run/progress", params={"after": 7}).json()
ok(len(p_mid["results"]) == 3, "after=7 时只回传最后 3 项",
   f"{list(p_mid['results'])}")
ok(len(p_mid["summary_rows"]) == 10, "汇总表始终是完整的 10 行")

print("\n[4] 运行期间禁止改动选择（防止两个线程同抢一个 Universe）")
r2 = client.post(f"/api/session/{sid}/run/start", json=body)
print(f"    再次 start: HTTP {r2.status_code}"
      + (f"  {r2.json().get('detail', '')[:40]}" if r2.status_code >= 400 else ""))
# 上一轮已结束，所以这次应当成功 —— 先把它取消掉
if r2.status_code < 400:
    r3 = client.post(f"/api/session/{sid}/run/start", json=body)
    ok(r3.status_code == 400, "运行中再次 start 被拒绝", f"HTTP {r3.status_code}")
    if r3.status_code == 400:
        print(f"      {r3.json().get('detail', '')[:60]}")
    sel = client.post(f"/api/session/{sid}/select",
                      json={"primary": {"mode": "auto"}, "components": ["protein"]})
    ok(sel.status_code == 400, "运行中修改选择被拒绝", f"HTTP {sel.status_code}")
    c = client.post(f"/api/session/{sid}/run/cancel")
    ok(c.status_code == 200, "取消请求被接受")
    # 等它停下来
    for _ in range(200):
        time.sleep(0.3)
        p = client.get(f"/api/session/{sid}/run/progress", params={"after": 0}).json()
        if p["status"] != "running":
            break
    ok(p["status"] == "cancelled", "取消后状态为 cancelled", p["status"])
    ok(p["n_done"] >= 1, "取消后仍保留已完成的结果", f"{p['n_done']} 项")

print("\n[5] 同步接口 /run 仍可用（向后兼容）")
r = client.post(f"/api/session/{sid}/run", json=body)
ok(r.status_code < 400, "POST /run 正常", f"HTTP {r.status_code}")
if r.status_code < 400:
    j = r.json()
    ok(len(j["results"]) == 10, "同步接口返回全部 10 项", str(len(j["results"])))
    ok("summary_rows" in j and "timings" in j, "同步接口字段完整")

client.delete(f"/api/session/{sid}")
print("\n" + "=" * 74)
print(f"通过 {pass_n} 项" + (f"，失败 {len(fails)} 项: {' | '.join(fails)}"
                           if fails else "，全部通过"))
print("=" * 74)
sys.exit(1 if fails else 0)
