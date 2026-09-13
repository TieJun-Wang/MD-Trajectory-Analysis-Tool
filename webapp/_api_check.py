# -*- coding: utf-8 -*-
"""后端 API 冒烟测试：直接用 TestClient 走一遍完整流程，不需要浏览器。"""
from __future__ import annotations

import json
import os
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


def P(*a):
    print(*a, flush=True)


def show(r, label, limit=400):
    P(f"   {label}: HTTP {r.status_code}")
    assert r.status_code < 400, r.text[:500]
    return r.json()


t0 = time.time()
P("=" * 76)
P("后端 API 冒烟测试")
P("=" * 76)

# 1) 健康检查
h = show(client.get("/api/health"), "GET /api/health")
P(f"      版本={h['version']}  前端已构建={h['frontend_built']}")
a = show(client.get("/api/analyses"), "GET /api/analyses")
P(f"      分析项 {len(a['order'])} 个: {a['order']}")
# 大纲模块分组（图表导航与数据表导航的分层都用它）
P("      按大纲模块分组:")
for g in a.get("groups", []):
    P(f"        - {g['label']:6} {g['names']}")
_group_flat = [n for g in a.get("groups", []) for n in g["names"]]
assert a.get("groups"), "缺少 groups（大纲模块分组）"
assert sorted(_group_flat) == sorted(a["order"]), (
    "groups 未恰好覆盖 order", sorted(_group_flat), sorted(a["order"]))
assert len(_group_flat) == len(set(_group_flat)), "groups 里有重复项"
assert [g["label"] for g in a["groups"]] == [
    "链构象", "空间结构", "取向与结晶", "动力学与输运"], "模块名或顺序与分组定义不符"
# 快捷按钮用的缩写必须每个分组都有、且足够短（那一行要求单行放下，
# 见 webapp/_css_check.py 的静态宽度估算：完整名会让整行溢出）
_short = [g.get("short") for g in a["groups"]]
assert all(isinstance(s, str) and 1 <= len(s) <= 3 for s in _short), _short
assert len(set(_short)) == len(_short), f"缩写重复: {_short}"
P("      groups 覆盖全部 order 且无重复、模块名与按钮缩写正确")

# 2) 文件选择器
f = show(client.get("/api/files", params={"dir": str(ROOT)}), "GET /api/files")
P(f"      目录={f['dir']}")
P(f"      子目录 {len(f['dirs'])} 个，MD 文件 {len(f['files'])} 个: "
  f"{[(x['name'], x['kind']) for x in f['files']]}")
P(f"      盘符: {f.get('drives')}")
assert any(x['name'] == 'adk_oplsaa.tpr' for x in f['files'])
assert any(x['name'] == 'adk_oplsaa.xtc' for x in f['files'])

# 3) 打开体系
s = show(client.post("/api/session", json={"topology": TOP, "trajectory": XTC}),
         "POST /api/session")
sid = s["sid"]
info = s["info"]
P(f"      sid={sid}")
P(f"      原子 {info['n_atoms']}  残基 {info['n_residues']}  帧 {info['n_frames']}  "
  f"dt {info['dt_ps']:.1f} ps  盒 {info['box_type']}")
P(f"      组分: {[(c['name'], c['n_atoms']) for c in s['components']]}")
P(f"      主链: {s['primary_label']} ({s['primary_atoms']} 原子)")
P(f"      链类型 {len(s['chains'])} 类")
assert info["n_atoms"] == 47681 and info["n_frames"] == 10

# 体系信息的「完整报告」必须是结构化表格，而不是让前端去解析定宽文本
secs = info.get("sections")
assert secs, "info 缺少 sections（结构化表格）"
P(f"      体系信息小节 {len(secs)} 个:")
for sec in secs:
    cols, rows = sec["columns"], sec["rows"]
    assert all(len(r) == len(cols) for r in rows), (sec["title"], cols, rows[:2])
    assert cols and rows, sec["title"]
    P(f"        - {sec['title']:26} {len(cols)} 列 × {len(rows)} 行")
# 报告里不允许再出现 markdown 语法（用户明确要求用表格，不要 md 文本）
_flat = " ".join(
    str(c) for sec in secs for row in sec["rows"] for c in row
) + " " + " ".join(str(sec.get("note", "")) for sec in secs)
assert "**" not in _flat, "报告里出现了 markdown 加粗语法"
assert not any(line.startswith("|") for line in _flat.splitlines()), "报告里出现了 md 表格语法"
P(f"      报告为纯表格结构（无 markdown 语法）")

# 4) 选择
sel = show(client.post(f"/api/session/{sid}/select",
                       json={"primary": {"mode": "component", "name": "protein"},
                             "components": ["protein", "water"]}),
           "POST /select")
P(f"      主链={sel['primary_label']}  组分={[c['name'] for c in sel['components']]}")

# 5) 跑分析
body = {
    "which": ["rg", "ree", "dihedral", "density", "rdf", "contact",
              "interface", "orientation", "order", "msd"],
    "frames": {"interval_ps": 300, "max_frames": None},
    "params": {"density": {"axis": 2, "nbins": 60},
               "interface": {"axis": 2, "nbins": 60},
               "rdf": {"rmax": 9.0, "nbins": 90},
               "contact": {"cutoff": 4.0},
               "dihedral": {"mode": "auto"}},
    "selection": {"primary": {"mode": "auto"}, "components": ["protein", "water"]},
}
t1 = time.time()
r = show(client.post(f"/api/session/{sid}/run", json=body), "POST /run")
P(f"      耗时 {time.time() - t1:.1f}s")
P(f"      帧: {r['frames']['describe']}")
P(f"      结果 {len(r['results'])} 项")
total_curves = sum(len(v['curves']) for v in r['results'].values())
total_pts = sum(len(c['x']) for v in r['results'].values() for c in v['curves'])
P(f"      曲线 {total_curves} 条，总点数 {total_pts}，JSON 约 "
  f"{len(json.dumps(r)) / 1024:.0f} KB")

for name, res in r["results"].items():
    P(f"      - {name:12} 面板 {res['n_panels']}  曲线 {len(res['curves']):2}  "
      f"统计 {len(res['summary']):2}  说明 {len(res['notes'])}")
    for p in res["panels"]:
        assert set(p) >= {"title", "xlabel", "ylabel", "xscale", "yscale", "curves"}
    for c in res["curves"]:
        assert len(c["x"]) == len(c["y"]) and set(c) >= {"label", "kind", "panel"}
assert len(r["results"]) == 10

# 数据结构抽查
rg = r["results"]["rg"]
P(f"\n      Rg 面板: {[p['title'] for p in rg['panels']]}")
P(f"      Rg 曲线: {[(c['label'], c['kind'], len(c['x'])) for c in rg['curves']]}")
P(f"      Rg 前 3 点: x={rg['curves'][0]['x'][:3]} y="
  f"{[round(v,4) for v in rg['curves'][0]['y'][:3]]}")
P(f"      Rg 统计: Rg mean = {rg['summary']['Rg mean']:.6f}")

msd = r["results"]["msd"]
P(f"\n      MSD 轴尺度: x={msd['panels'][0]['xscale']} y={msd['panels'][0]['yscale']}"
  f"  (前端应切到对数轴)")
di = r["results"]["dihedral"]
kinds = sorted({c["kind"] for c in di["curves"]})
P(f"      二面角曲线的画法: {kinds}  (bar + line)")

# 界面分析摘要：板层体系应逐个列出界面并给出 erf 拟合结果
ifc_res = r["results"]["interface"]
P(f"\n      界面分析摘要（{ifc_res['summary'].get('组分 A')} / "
  f"{ifc_res['summary'].get('组分 B')}）:")
for k, v in ifc_res["summary"].items():
    P(f"        {k} = {v:.4f}" if isinstance(v, float) else f"        {k} = {v}")

# 6) 导出
e = show(client.post(f"/api/session/{sid}/export",
                     json={"formats": ["csv", "png"], "panel_pngs": True,
                           "excel": True}), "POST /export")
P(f"      目录={e['dir']}")
P(f"      CSV {e['n_csv']}  PNG {e['n_png']}  Excel={os.path.basename(e['excel'] or '')}")
P(f"      文件总数 {len(e['files'])}")

# 7) 下载一个导出文件
png = next((f for f in e["files"] if f["name"].endswith(".png")), None)
if png:
    resp = client.get(png["url"])
    P(f"\n      GET {png['url']} -> HTTP {resp.status_code}, "
      f"{len(resp.content)} bytes, type={resp.headers.get('content-type')}")
    assert resp.status_code == 200 and len(resp.content) > 5000
    from PIL import Image
    import io
    P(f"      PNG 尺寸 = {Image.open(io.BytesIO(resp.content)).size}")

# 8) PEG 数据集：链类型的「全部 N 条」必须真的合并全部链
#    回归目标：1.0.1 之前 apply_selection 无论 count 是多少都只取 got[0]，
#    于是一个标着"200 条链"的选项实际只分析了 1 条（30 原子）。
P("\n   PEG 数据集（链类型合并回归）:")
PEG_DIR = Path(os.environ.get("MDTA_PEG_DIR", r"C:\temp\dataset\PEG_parametrization"))
PEG_TOP = PEG_DIR / "eqpbc.gro"
PEG_XTC = PEG_DIR / "trajpbc.xtc"
if not (PEG_TOP.is_file() and PEG_XTC.is_file()):
    P(f"      ~ 跳过：找不到 {PEG_DIR}（可用 MDTA_PEG_DIR 覆盖）")
else:
    dp = show(client.post("/api/session",
                          json={"topology": str(PEG_TOP), "trajectory": str(PEG_XTC)}),
              "POST /api/session (PEG)", limit=200)
    sid_p = dp["sid"]
    comp = next(c for c in dp["components"] if c["name"] == "polymer")
    ch = dp["chains"][0]
    P(f"      {dp['info']['n_atoms']} 原子 / {dp['info']['n_frames']} 帧；"
      f"组分 polymer = {comp['n_molecules']} 个分子；"
      f"链类型 {ch['label']} count={ch['count']} 每条 {ch['n_atoms']} 原子")
    assert comp["n_molecules"] == ch["count"] == 200
    assert ch["n_atoms"] == 30

    def _rg(primary, tag):
        body = {"which": ["rg"], "estimate": False, "frames": {"max_frames": 20},
                "params": {"rg": {"per_molecule": True}},
                "selection": {"primary": primary, "components": ["polymer"]}}
        r = show(client.post(f"/api/session/{sid_p}/run", json=body), f"run {tag}")
        s = r["results"]["rg"]["summary"]
        return r["selection"], s

    sel_all, s_all = _rg({"mode": "chain", "segid": ch["segid"],
                          "resname": ch["resname"], "min_atoms": ch["n_atoms"],
                          "all": True}, "全部链")
    sel_comp, s_comp = _rg({"mode": "component", "name": "polymer"}, "组分 polymer")
    sel_one, s_one = _rg({"mode": "chain", "segid": ch["segid"],
                          "resname": ch["resname"], "min_atoms": ch["n_atoms"]},
                         "第 1 条")
    P(f"      全部 {ch['count']} 条链 -> {sel_all['primary_atoms']} 原子 / "
      f"{sel_all['primary_molecules']} 分子，整组 Rg={s_all['Rg mean']:.3f} Å")
    P(f"      组分 polymer  -> {sel_comp['primary_atoms']} 原子 / "
      f"{sel_comp['primary_molecules']} 分子，整组 Rg={s_comp['Rg mean']:.3f} Å")
    P(f"      第 1 条        -> {sel_one['primary_atoms']} 原子 / "
      f"{sel_one['primary_molecules']} 分子，整组 Rg={s_one['Rg mean']:.3f} Å")
    assert sel_all["primary_atoms"] == comp["n_atoms"] == 6000, sel_all
    assert sel_all["primary_molecules"] == 200
    # 「全部链」与「组分 polymer」是同一批原子 → 数值必须一致
    assert abs(s_all["Rg mean"] - s_comp["Rg mean"]) < 1e-9, (s_all["Rg mean"],
                                                              s_comp["Rg mean"])
    # 而「第 1 条」只有 30 原子，数值必须明显不同（否则说明合并没生效）
    assert sel_one["primary_atoms"] == 30
    assert abs(s_one["Rg mean"] - s_all["Rg mean"]) > 5.0, (s_one["Rg mean"],
                                                            s_all["Rg mean"])
    P(f"      ✓ 全部链 == 组分 polymer（同批原子，Rg 差 <1e-9）；"
      f"第 1 条相差 {abs(s_one['Rg mean'] - s_all['Rg mean']):.2f} Å")
    show(client.delete(f"/api/session/{sid_p}"), "DELETE /api/session/{sid_p} (PEG)")

# 9) 错误处理
P("\n   错误处理:")
r404 = client.get("/api/session/deadbeef/info")
P(f"      不存在的会话 -> HTTP {r404.status_code}")
assert r404.status_code == 404
r400 = client.post("/api/session", json={"topology": str(ROOT / "nope.tpr")})
P(f"      不存在的文件 -> HTTP {r400.status_code}: {r400.json()['detail'][:50]}")
assert r400.status_code == 400
rbad = client.post(f"/api/session/{sid}/run", json={"which": ["nonsense"]})
P(f"      未知分析项 -> HTTP {rbad.status_code}")
assert rbad.status_code in (400, 500)

# 10) 关闭会话
d = show(client.delete(f"/api/session/{sid}"), "DELETE /api/session/{sid}")
P(f"      已关闭 {d['closed']}")
r404b = client.get(f"/api/session/{sid}/info")
P(f"      关闭后再访问 -> HTTP {r404b.status_code}")
assert r404b.status_code == 404
P(f"      导出目录已清理: {not os.path.isdir(e['dir'])}")

P("\n" + "=" * 76)
P(f"后端 API 全部通过，总耗时 {time.time() - t0:.1f}s")
P("=" * 76)
