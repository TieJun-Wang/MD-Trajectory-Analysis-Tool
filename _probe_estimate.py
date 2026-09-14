# -*- coding: utf-8 -*-
"""探针：「先预估用时再跑」这条链路到底还在不在（含新分析项 rdf2d / comdist）。

用真实后端（TestClient）+ 真实轨迹走一遍界面会走的路径：

    POST /api/session/{sid}/run/start {estimate, which, frames}
    GET  /api/session/{sid}/run/progress   ← 轮询，看 phase / estimates / estimate_note / order

覆盖四种情况（前端的「预估用时」区块在四种情况下都必须有内容可显示）：
  ① 多项 + 开预估        → 应给出每项预估秒数并按短→长重排执行顺序
  ② 只勾 1 项            → 预估无意义，但**必须说明原因**（否则看起来像模块没了）
  ③ 关掉开关             → 同上，必须说明"已关闭"
  ④ 帧数 < 40            → estimate_times 自己给出"只有 N 帧…直接开跑"
打印的字段就是前端 RunBlock 会渲染的那些。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient            # noqa: E402

from webapp.backend import app as appmod             # noqa: E402

SI = os.environ.get("MDTA_SI_DIR", r"C:\temp\dataset\Supporting_Info")
CG = os.path.join(SI, "64mer", "64mer_P1_random1_RBC", "6_CG-takeCURRENT")
TOP = os.path.join(CG, "3-run.tpr")
XTC = os.path.join(CG, "3-run_center.xtc")

client = TestClient(appmod.app)


def P(*a):
    print(*a, flush=True)


def wait_idle(sid: str, budget: float = 120.0) -> None:
    t0 = time.time()
    while time.time() - t0 < budget:
        p = client.get(f"/api/session/{sid}/run/progress", params={"after": 0}).json()
        if p.get("status") != "running":
            return
        time.sleep(0.4)


def case(sid: str, label: str, body: dict, budget: float = 240.0) -> dict:
    """跑一种情况，拿到前端会渲染的字段后立刻取消。"""
    P(f"\n--- {label} ---")
    r = client.post(f"/api/session/{sid}/run/start", json=body)
    assert r.status_code < 400, r.text[:300]
    t0, seen, out = time.time(), [], {}
    while time.time() - t0 < budget:
        p = client.get(f"/api/session/{sid}/run/progress", params={"after": 0}).json()
        ph = p.get("phase", "")
        if ph not in seen:
            seen.append(ph)
            P(f"  [{time.time() - t0:6.1f}s] phase -> {ph!r}  {p.get('message')!r}")
        out = p
        if ph == "run" and (p.get("estimates") or p.get("estimate_note")):
            break
        if p.get("status") != "running":
            break
        time.sleep(0.5)
    P(f"  phase          = {seen}")
    P(f"  estimates      = {out.get('estimates')}")
    P(f"  estimate_note  = {out.get('estimate_note')!r}")
    P(f"  order          = {out.get('order')}   （勾选 {body['which']}）")
    client.post(f"/api/session/{sid}/run/cancel")
    wait_idle(sid)
    return out


P("=" * 78)
if not (os.path.isfile(TOP) and os.path.isfile(XTC)):
    P(f"~ 跳过：找不到 CG 数据集 {CG}")
    raise SystemExit(0)

a = client.get("/api/analyses").json()
flat = [n for g in a.get("groups", []) for n in g["names"]]
P(f"/api/analyses：{len(a['order'])} 个分析项；分组覆盖 {len(flat)} 项，"
  f"差集 = {sorted(set(a['order']) - set(flat))}")
P(f"  rdf2d / comdist 在 order = {'rdf2d' in a['order']} / {'comdist' in a['order']}；"
  f"在 groups = {'rdf2d' in flat} / {'comdist' in flat}")

s = client.post("/api/session", json={"topology": TOP, "trajectory": XTC})
assert s.status_code < 400, s.text[:300]
sj = s.json()
sid = sj["sid"]
P(f"\n打开体系：{sj['info']['n_atoms']} 原子 / {sj['info']['n_frames']} 帧；"
  f"组分 {[c['name'] for c in sj['components']]}")

base = {"params": {}, "selection": {"primary": {"mode": "auto"}, "components": []}}
r1 = case(sid, "① 3 项 + 开预估（含新项 rdf2d / comdist）",
          {**base, "which": ["rg", "rdf2d", "comdist"], "estimate": True,
           "frames": {"max_frames": 60}})
r2 = case(sid, "② 只勾 1 项 + 开预估",
          {**base, "which": ["rg"], "estimate": True, "frames": {"max_frames": 60}})
r3 = case(sid, "③ 2 项 + 关掉预估",
          {**base, "which": ["rg", "comdist"], "estimate": False,
           "frames": {"max_frames": 60}})
r4 = case(sid, "④ 2 项 + 开预估，但只有 20 帧（< ESTIMATE_MIN_FRAMES=40）",
          {**base, "which": ["rg", "comdist"], "estimate": True,
           "frames": {"max_frames": 20}})
client.delete(f"/api/session/{sid}")

P("\n" + "=" * 78)
P("结论（前端的「预估用时」区块要能显示下面这些内容）：")
checks = [
    ("① 有预估秒数（含 rdf2d/comdist）", bool(r1.get("estimates"))),
    ("① 执行顺序被预估改写", r1.get("order") != ["rg", "rdf2d", "comdist"]),
    ("② 只勾 1 项时给出跳过原因", "1 项" in (r2.get("estimate_note") or "")),
    ("③ 关掉开关时给出跳过原因", "关闭" in (r3.get("estimate_note") or "")),
    ("④ 帧数不足时给出具体原因", "帧" in (r4.get("estimate_note") or "")),
]
for label, okv in checks:
    P(f"  {'✓' if okv else '✗'} {label}")
P(f"\n全部通过：{all(v for _, v in checks)}")
