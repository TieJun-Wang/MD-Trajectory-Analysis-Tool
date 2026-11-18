# -*- coding: utf-8 -*-
"""后端 API 冒烟测试：直接用 TestClient 走一遍完整流程，不需要浏览器。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent.parent   # test/webapp → test → 仓库根
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
assert "vacf" in _group_flat, "分析清单里缺少 VACF（速度自相关函数）"
assert a.get("groups"), "缺少 groups（大纲模块分组）"
assert sorted(_group_flat) == sorted(a["order"]), (
    "groups 未恰好覆盖 order", sorted(_group_flat), sorted(a["order"]))
assert len(_group_flat) == len(set(_group_flat)), "groups 里有重复项"
assert [g["label"] for g in a["groups"]] == [
    "链构象", "空间结构", "取向", "动力学与输运"], "模块名或顺序与分组定义不符"
# 快捷按钮用的缩写必须每个分组都有、且足够短（那一行要求单行放下，
# 见 test/webapp/_css_check.py 的静态宽度估算：完整名会让整行溢出）
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
# 格式说明由后端下发（唯一来源 mdta.formats），前端不再自己写一份
fmt = show(client.get("/api/formats"), "GET /api/formats")
P(f"      支持格式: 拓扑 {fmt['topology_count']} 种 / 轨迹 {fmt['trajectory_count']} 种")
assert fmt["topology_count"] >= 15 and fmt["trajectory_count"] >= 20
assert ".pdb" in fmt["topology"] and ".xtc" in fmt["trajectory"]
# LAMMPS：`.lmp`/`.dat` 也该出现在拓扑里（内容其实是 data，靠 topology_format 读）
for ext in (".lmp", ".dat", ".lammpsdump"):
    assert ext in fmt["topology"] or ext in fmt["trajectory"], ext

# 2a) LAMMPS 项目里同一批后缀混着 data / 输入脚本 / 日志，文件列表要按**内容**标注
from mdta.formats import sniff_lammps_kind                      # noqa: E402

lum = ROOT / "test" / "_lammps_fixture"     # 夹具由 test/selftest.py 生成
if lum.is_dir():
    lf = show(client.get("/api/files", params={"dir": str(lum)}), "GET /api/files (LAMMPS)")
    lnotes = {x["name"]: x.get("note") for x in lf["files"]}
    P(f"      内容警示: {[(k, (v or '')[:18]) for k, v in lnotes.items() if v]}")
    assert "输入脚本" in (lnotes.get("in.lammps") or ""), lnotes.get("in.lammps")
    assert "日志" in (lnotes.get("log.lammps") or ""), lnotes.get("log.lammps")
    # 真正的 data 文件不能被误标
    for name in ("sys.lmp", "sys.dat", "sys.data", "sys.lammps"):
        assert lnotes.get(name) is None, (name, lnotes.get(name))
        assert sniff_lammps_kind(str(lum / name)) == "data"
    assert "data" in (lnotes.get("result.dat") or ""), lnotes.get("result.dat")

# 2b) 后缀 → 槽位分类 + 放错槽位必须是**可读的 400**（以前是 500 HTML）
kinds = {x["name"]: x for x in f["files"]}
P(f"      两用途文件（roles 含两项）: "
  f"{[k for k, v in kinds.items() if len(v.get('roles', [])) == 2]}")
for name, want in (("adk_oplsaa.tpr", "topology"), ("adk_oplsaa.xtc", "trajectory")):
    assert kinds[name]["kind"] == want, (name, kinds[name])
# ⚠️ `kind` **只能**是 topology / trajectory 两个值：第一版曾让 .pdb 返回
# `kind="both"`，结果页面上还是老前端（只认 kind === 'topology'）时，
# .pdb 会掉进 else 分支被塞进**轨迹槽** —— 用户报"pdb 怎么读到轨迹去了"。
# 两用途的信息必须走新增字段 roles，老客户端不认识会直接忽略。
assert all(v["kind"] in ("topology", "trajectory") for v in kinds.values()), kinds
assert kinds["adk_oplsaa.xtc"].get("roles") == ["trajectory"]
bad = client.post("/api/session", json={"topology": XTC, "trajectory": TOP})
P(f"      轨迹放进拓扑槽 -> HTTP {bad.status_code}（应为 400，且带 JSON 说明）")
assert bad.status_code == 400, bad.status_code
P(f"        {bad.json()['detail'][:96]}")
junk = client.post("/api/session", json={"topology": str(ROOT / "README.md")})
P(f"      拿 README.md 当拓扑 -> HTTP {junk.status_code}")
assert junk.status_code == 400 and "可用拓扑格式" in junk.json()["detail"]

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
# 主链原子清单（R_ee 指定原子的悬浮提示：序号 → 原子名）
_pa = client.get(f"/api/session/{sid}/primary-atoms")
assert _pa.status_code == 200, _pa.text
_pa = _pa.json()
assert _pa["n_atoms"] == s["primary_atoms"], "主链原子数与 /info 不一致"
assert len(_pa["names"]) == len(_pa["indices"]) == _pa["n_atoms"], "主链原子名/编号数量不对"
P(f"      主链原子清单 {_pa['n_atoms']} 个，前三个 {_pa['names'][:3]}")
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
              "interface", "orientation", "order", "msd", "vacf"],
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
assert "diffusion" not in r["results"], "「扩散系数 D」应已删除（log–log 已并入 MSD）"
# 10 项（vacf 需要速度：这条 AdK 轨迹是 xtc，没有速度 → 会跳过，所以 10/11 都可能）
assert len(r["results"]) in (10, 11), (len(r["results"]), list(r["results"]))

# 数据结构抽查
rg = r["results"]["rg"]
P(f"\n      Rg 面板: {[p['title'] for p in rg['panels']]}")
P(f"      Rg 曲线: {[(c['label'], c['kind'], len(c['x'])) for c in rg['curves']]}")
P(f"      Rg 前 3 点: x={rg['curves'][0]['x'][:3]} y="
  f"{[round(v,4) for v in rg['curves'][0]['y'][:3]]}")
P(f"      Rg 统计: Rg mean = {rg['summary']['Rg mean']:.6f}")

msd = r["results"]["msd"]
P(f"\n      MSD 轴尺度: x={msd['panels'][0]['xscale']} y={msd['panels'][0]['yscale']}"
  f"  (用户要求：第一张图**线性-线性**，τ=0 那个点也要画出来)")
assert msd['panels'][0]['xscale'] == 'linear' and msd['panels'][0]['yscale'] == 'linear', (
    "MSD 第一张图必须是线性-线性（用户实测反馈：横轴还是对数就看不出是线性图）",
    msd['panels'][0])
# 一张图里多个对象时前端默认只画第一个 → 曲线必须带 group 才分得出来
_grp = {}
for _name, _res in r["results"].items():
    for _c in _res["curves"]:
        if _c.get("group"):
            _grp.setdefault(_name, set()).add(_c["group"])
P(f"      带 group 的分析项: { {k: len(v) for k, v in _grp.items()} }")
assert "msd" in _grp and len(_grp["msd"]) >= 2, ("MSD 曲线要带 group", _grp.get("msd"))
assert "density" in _grp and len(_grp["density"]) >= 2, (
    "密度分布（多组分）曲线也要带 group，否则图上所有组分一起画", _grp.get("density"))
di = r["results"]["dihedral"]
kinds = sorted({c["kind"] for c in di["curves"]})
P(f"      二面角曲线的画法: {kinds}  (bar + line)")

# 接触分析：逐原子柱图删掉了；接触矩阵的 panel 号必须与 heatmap=True 的面板序号一致
# （1.0.4 实测 bug：插入热力面板时忘了挪后面的面板号 → 热力图挂到了别的面板上，
#   接触矩阵那张图是空的、接触对数分布画到了热力面板上，整块错位。）
ct = r["results"]["contact"]
P(f"\n      接触分析面板: {[p['title'] for p in ct['panels']]}")
assert not any("各原子的" in p["title"] for p in ct["panels"]), (
    "接触分析不应再有「各原子的…」逐原子柱图（用户要求删掉）", [p["title"] for p in ct["panels"]])
assert any("接触矩阵" in p["title"] for p in ct["panels"]), (
    "接触分析应保留接触矩阵面板", [p["title"] for p in ct["panels"]])
_heat = ct.get("heatmaps") or []
if _heat:
    _hm = _heat[0]
    _skip = [p.get("heatmap") for p in ct["panels"]].index(True)
    assert _hm["panel"] == _skip, (
        "接触矩阵的 panel 号必须与 heatmap=True 的面板序号一致",
        _hm["panel"], _skip, [p["title"] for p in ct["panels"]])
    P(f"      接触矩阵 panel={_hm['panel']}（= heatmap 面板序号 ✓）"
      f"  x{len(_hm['x'])} × y{len(_hm['y'])}")
    # 每张面板都要有曲线（除热力面板本身）
    for _p in ct["panels"]:
        if _p.get("heatmap"):
            continue
        assert any(c["panel"] == _p["index"] for c in ct["curves"]), (
            "面板没有任何曲线（面板号错位的最典型症状）", _p["title"])

# 原「扩散系数 D」已并入 MSD：第 4 张图是 log–log（幂律 α 判据），
# 且 MSD 的 meta 里带「拟合区间 / 斜率 / R² / D」给图上覆盖层用
_msd = r["results"]["msd"]
_ll = [c for c in _msd["curves"] if c["panel"] == 3]
P(f"\n      MSD log–log 曲线: {[(c['label'], len(c['x'])) for c in _ll]}")
assert _ll, ("MSD 应该有第 4 张 log–log 幂律图", [c["panel"] for c in _msd["curves"]])
assert any(str(c["label"]).startswith("参考斜率") for c in _ll), (
    "log–log 图要带 α=1 参考线", [c["label"] for c in _ll])
# 拟合覆盖层用的统计量在 summary 里（形如「<组> 拟合起点 (ps)」）
_fit_keys = [k for k in _msd["summary"] if str(k).endswith("拟合起点 (ps)")]
assert _fit_keys, ("MSD summary 缺少「<组> 拟合起点 (ps)」", list(_msd["summary"])[:12])
for _k in _fit_keys:
    _g = str(_k)[:-len(" 拟合起点 (ps)")]
    for _suf in ("拟合起点 (ps)", "拟合终点 (ps)", "拟合斜率 (Å²/ps)"):
        assert f"{_g} {_suf}" in _msd["summary"], ("拟合覆盖层缺字段", _g, _suf)
P(f"      拟合覆盖层: {[(k[:-len(' 拟合起点 (ps)')],
                        round(float(_msd['summary'].get(k, 0) or 0), 3)) for k in _fit_keys]}")

# 数据表分页：一页一传（前端只渲染一页，避免上万行 <td> 卡死）
_t0 = show(client.get(f"/api/session/{sid}/table",
                      params={"name": "rg", "panel": 0, "offset": 0, "limit": 10}),
           "GET /table?name=rg&limit=10")
P(f"      数据表第一页 {len(_t0['rows'])} 行 / 共 {_t0['total']} 行，"
  f"列 {_t0['columns']}")
assert _t0["rows"] and len(_t0["rows"]) <= 10, ("limit 没生效", len(_t0["rows"]))
assert len(_t0["columns"]) == len(_t0["rows"][0]), ("列数与行长度不一致", _t0["columns"])
_mid = 5 if _t0["total"] > 5 else 0
_t1 = client.get(f"/api/session/{sid}/table",
                 params={"name": "rg", "panel": 0, "offset": _mid, "limit": 10})
assert _t1.status_code == 200, _t1.text
assert _t1.json()["offset"] == _mid and _t1.json()["total"] == _t0["total"]
P("      翻页 offset 生效（第二页与原页 total 一致）")

# 整表 CSV（复制 / 下载用）：行数 = total+1（含表头），不受分页限制
_csv = client.get(f"/api/session/{sid}/table.csv",
                  params={"name": "rg", "panel": 0})
assert _csv.status_code == 200, _csv.text[:200]
assert "text/csv" in _csv.headers.get("content-type", ""), _csv.headers
_lines = [x for x in _csv.text.splitlines() if x.strip()]
assert len(_lines) == _t0["total"] + 1, (
    "整表 CSV 行数应为 total+1（含表头）", len(_lines), _t0["total"])
P(f"      整表 CSV：{len(_lines) - 1} 行（= total，不分页）")

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

    def _rg(primary, tag, rg_object="system"):
        # 「分析对象」两档互斥：这里选「整体体系」，用来判断选择合并有没有生效
        # （整组 Rg 对"合并了 200 条 vs 只有 1 条"极其敏感）。
        body = {"which": ["rg"], "estimate": False, "frames": {"max_frames": 20},
                "params": {"rg": {"object": rg_object}},
                "selection": {"primary": primary, "components": ["polymer"]}}
        r = show(client.post(f"/api/session/{sid_p}/run", json=body), f"run {tag}")
        return r["results"]["rg"], r["selection"]

    res_all, sel_all = _rg({"mode": "chain", "segid": ch["segid"],
                            "resname": ch["resname"], "min_atoms": ch["n_atoms"],
                            "all": True}, "全部链")
    res_comp, sel_comp = _rg({"mode": "component", "name": "polymer"}, "组分 polymer")
    res_one, sel_one = _rg({"mode": "chain", "segid": ch["segid"],
                            "resname": ch["resname"], "min_atoms": ch["n_atoms"]},
                           "第 1 条")
    s_all, s_comp, s_one = res_all["summary"], res_comp["summary"], res_one["summary"]
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

    # 「分析对象」两档互斥：整体体系 vs 分子 / 链 必须是两套不同的结果
    res_mol, _ = _rg({"mode": "component", "name": "polymer"}, "组分 polymer（分子/链）",
                     rg_object="molecule")
    s_mol = res_mol["summary"]
    labels = [c["label"] for c in res_mol["curves"]]
    P(f"      「分子 / 链」模式：各分子 Rg 平均={s_mol['Rg mean']:.3f} Å，"
      f"曲线 {labels}")
    assert s_mol["分析对象"] == "分子 / 链", s_mol.get("分析对象")
    assert s_all["分析对象"] == "整体体系", s_all.get("分析对象")
    assert s_mol["Rg mean"] < s_all["Rg mean"] - 5.0, (s_mol["Rg mean"],
                                                       s_all["Rg mean"])
    # 互斥：分子/链模式不产出整组那条曲线；整体体系模式不产出逐分子统计量
    assert any("逐帧各分子 Rg 平均" in x for x in labels), labels
    assert not any(x in ("polymer", "全部 200 条链") for x in labels), labels
    assert "单分子 Rg 平均 (Å)" in s_mol
    assert "单分子 Rg 平均 (Å)" not in s_all
    # 只有一条 Rg 随时间变化的曲线是「主曲线」（另一条是常数平均线）
    assert sum(1 for c in res_mol["curves"]
               if c["label"] in ("逐帧各分子 Rg 平均", "平均 Rg")) == 2
    assert set(p["title"] for p in res_mol["panels"]) >= {"Rg 随时间变化", "Rg 分布"}
    P(f"      ✓ 两档互斥：整体体系 {s_all['Rg mean']:.2f} Å vs 分子 / 链 "
      f"{s_mol['Rg mean']:.3f} Å（各只算一种，不重复扫描）")
    show(client.delete(f"/api/session/{sid_p}"), "DELETE /api/session/{sid_p} (PEG)")

# 9) 错误处理
# 9b) 分析集合（Set）：像 Forcite 的 Set 那样按元素 / 形态切分
P("\n   分析集合（Set）:")
el = s.get("elements") or []
pres = s.get("set_presets") or []
P(f"      可选元素 {el}；预设 {[p['name'] for p in pres]}")
assert el, "打开体系的响应里应带 elements"
sel_sets = [{"name": "C", "elements": ["C"]},
            {"name": "链状", "shape": "chain"}]
ss = show(client.post(f"/api/session/{sid}/select",
                      json={"components": ["water", "set:C"], "sets": sel_sets}),
          "POST /select (sets)")
info = {x["name"]: x for x in ss["sets"]}
P(f"      集合预览: {[(k, v['kind'], v['n_atoms']) for k, v in info.items()]}")
assert info["set:C"]["kind"] == "atom", "只取 C 应该按原子追踪"
assert info["set:C"]["n_atoms"] > 0
names = [x["name"] for x in ss["components"]]
P(f"      组分清单含集合: {[n for n in names if n.startswith('set:')]}")
assert "set:C" in names
assert "set:链状" not in names, "没勾选的集合不应出现在组分里"
only_sets = show(client.post(f"/api/session/{sid}/select",
                             json={"components": ["set:C"],
                                   "sets": sel_sets}),
                 "POST /select (只勾集合)")
only_names = [x["name"] for x in only_sets["components"]]
P(f"      只勾集合时的组分: {only_names}")
assert only_names == ["set:C"], ("只勾集合时不应残留自动组分", only_names)
# 只勾集合 + 主链=自动 → 分析对象就是集合本身（否则 Rg / R_ee / 二面角 看不见集合）
P(f"      只勾集合时的分析对象: {only_sets['primary_label']}")
assert only_sets["primary_atoms"] == info["set:C"]["n_atoms"], (
    "主链应等于勾选的集合", only_sets["primary_atoms"], info["set:C"]["n_atoms"])
assert only_sets["primary_label"].startswith("集合"), only_sets["primary_label"]
# 主链显式选成 set:C（界面下拉框里「组分 set:C」这一档）
pick = show(client.post(f"/api/session/{sid}/select",
                        json={"components": ["water", "set:C"], "sets": sel_sets,
                              "primary": {"mode": "component", "name": "set:C"}}),
            "POST /select (主链=set:C)")
P(f"      主链显式 = set:C -> {pick['primary_label']}（{pick['primary_atoms']} 原子）")
assert pick["primary_atoms"] == info["set:C"]["n_atoms"], pick["primary_atoms"]

# 候选清单（catalog）必须保留**全部**自动识别的组分，不管勾没勾。
# 用户实测：第一轮没勾自动组分（只跑 MSD），第二轮／刷新后组分表整张空了。
_cat = [c["name"] for c in only_sets.get("catalog") or []]
P(f"      候选清单（取消勾选后仍在）: {_cat[:8]}{'…' if len(_cat) > 8 else ''}")
assert _cat, "select 响应必须带候选清单 catalog"
assert "water" in _cat, ("候选清单必须保留自动识别的组分（用户实测 bug）", _cat)
assert "set:C" in _cat, ("已定义的集合也要在候选清单里", _cat)
_w = next(c for c in only_sets["catalog"] if c["name"] == "water")
assert _w["active"] is False, ("没勾的组分应为 active=False 但仍留在清单里", _w)
# 全部取消勾选（第一轮只跑 MSD 的情形）：清单仍在，只是全不 active
empty = show(client.post(f"/api/session/{sid}/select",
                         json={"components": [], "sets": sel_sets}),
             "POST /select (一个都不勾)")
assert [x["name"] for x in empty["components"]] == [], empty["components"]
_cat0 = [c["name"] for c in empty.get("catalog") or []]
P(f"      一个都不勾：参与 {len(empty['components'])} 个，候选 {len(_cat0)} 个：{_cat0[:8]}")
assert _cat0, "一个都不勾时候选清单也必须还在（否则第二轮没得选）"
assert not any(c["active"] for c in empty["catalog"]), "一个都不勾时都应为 active=False"
# 主链直接选未勾选的集合 → 自动启用（不必先去勾它）
_auto_on = show(client.post(f"/api/session/{sid}/select",
                            json={"components": [], "sets": sel_sets,
                                  "primary": {"mode": "component", "name": "set:C"}}),
                "POST /select (主链=未勾选的 set:C)")
P(f"      主链=未勾选集合 -> {_auto_on['primary_label']}，"
  f"参与组分 {[x['name'] for x in _auto_on['components']]}")
assert _auto_on["primary_atoms"] == info["set:C"]["n_atoms"], _auto_on["primary_atoms"]
assert "set:C" in [x["name"] for x in _auto_on["components"]], "主链选中的集合应自动启用"

# 主链选「候选清单里有、但这次没勾」的**普通组分**：用户实测选了 water 跑 Rg，
# 报错「没有 water 这个组分」—— 因为「分析对象」下拉列的是候选清单（catalog），
# 勾选框列的是本次参与项，两者本来就可以不一致。现在必须自动启用并真的按它算。
w = show(client.post(f"/api/session/{sid}/select",
                     json={"components": [], "sets": sel_sets,
                           "primary": {"mode": "component", "name": "water"}}),
         "POST /select (主链=未勾选的 water)")
P(f"      主链=water（没勾它）-> {w['primary_label']}；"
  f"参与组分 {[x['name'] for x in w['components']]}")
assert w["primary_atoms"] > 0, w
assert "water" in [x["name"] for x in w["components"]], (
    "主链选中的组分必须自动启用（用户实测：以前报「没有 water 这个组分」）",
    w["components"])
rg_w = show(client.post(f"/api/session/{sid}/run",
                        json={"which": ["rg"], "frames": {}, "estimate": False,
                              "params": {"rg": {"object": "molecule"}},
                              "selection": {"components": [],
                                            "primary": {"mode": "component",
                                                        "name": "water"}}}),
            "POST /run (Rg, 分析对象=water)")
_rg = rg_w["results"]["rg"]
_lbl = _rg.get("title") or ""
P(f"      Rg 标题: {_lbl}；曲线 {[_c['label'] for _c in _rg['curves']]}")
assert "water" in _lbl, f"Rg 的分析对象不是 water：{_lbl}"

# 9c) 结构选择（三维视图 / 层级树 / 可视化选择 → 集合）
P("\n   结构选择（三维视图 / 层级 / 可视化选择 → 集合）:")
st3 = show(client.get(f"/api/session/{sid}/structure",
                      params={"frame": 0, "max_atoms": 5000}),
           "GET /structure")
P(f"      {st3['n_atoms']} 原子，显示 {st3['n_displayed']}"
  f"（残基 {st3['n_residues_shown']}/{st3['n_residues']}），"
  f"键 {st3['n_bonds']} 条（来源 {st3['bond_source']}）")
assert st3["n_displayed"] > 0 and len(st3["index"]) == st3["n_displayed"]
assert st3["n_displayed"] <= st3["n_atoms"]
assert len(st3["xyz"]) == st3["n_displayed"] == len(st3["elements"])
assert all(0 <= i < st3["n_atoms"] for i in st3["index"][:100]), "索引必须是原始编号"
assert st3["truncated"] is True, "max_atoms 应触发按残基截断"
# 芳香环（画虚线圈用）：字段必须在；AdK 是蛋白 + 水，没有苯环 → 空列表
assert "rings" in st3, "结构返回里缺 rings（三维视图的芳香环虚线圈要用）"
assert isinstance(st3["rings"], list) and not st3["rings"], (
    "AdK 没有芳香六元环，rings 应为空", st3["rings"][:3])
# ⚠️ 键索引必须是**显示数组的位置**（< n_displayed）。用户实测白屏就是因为它
#    用了原始原子编号（抽稀后能到 4 万多），前端索引 xyz 越界 → React 卸掉整棵树。
_bonds_flat = st3["bonds"]
assert len(_bonds_flat) % 2 == 0, ("键表应是扁平的成对数组", len(_bonds_flat))
_bmax = max(_bonds_flat, default=-1)
assert _bmax < st3["n_displayed"], (
    "键索引越界（会白屏）", _bmax, st3["n_displayed"])
assert st3["n_bonds"] == len(_bonds_flat) // 2, (st3["n_bonds"], len(_bonds_flat))
assert st3["bond_source"] == "topology", ("这体系自带键表，来源应为 topology",
                                         st3["bond_source"])
assert st3["n_bonds"] > 0, "球棍模型需要键表（有拓扑键却没返回）"
P(f"      键索引最大值 {_bmax} < 显示原子数 {st3['n_displayed']}（显示空间，不会越界）")
P(f"      键表是扁平成对数组 {len(_bonds_flat)} 个数 → {st3['n_bonds']} 根键")

# 模拟盒（用户要求：白色条状显示盒子）—— 盒矢量要能还原成 3 条棱 + 12 根边
_bx = st3.get("box")
assert _bx and len(_bx) >= 3, ("应返回模拟盒矢量", _bx)
assert all(isinstance(float(x), float) for x in _bx[:6]), _bx
_frac = st3.get("box_frac_inside")
assert _frac is not None, "应给出'多少原子在盒内'（判断坐标是否已折回）"
P(f"      模拟盒 {[round(float(x), 2) for x in _bx[:3]]} Å "
  f"角度 {[round(float(x), 1) for x in _bx[3:6]]}；"
  f"盒内原子 {_frac * 100:.0f}%（已折回={st3.get('box_wrapped')}）")
assert 0.0 <= _frac <= 1.0, _frac

# MDAnalysis 原生选择语句（用户要求：补上 select_atoms）
_q1 = show(client.get(f"/api/session/{sid}/query",
                      params={"expr": "resid 1:3 and name CA"}),
           "GET /query (resid 1:3 and name CA)")
P(f"      选择语句「{_q1['query']}」-> {_q1['n_atoms']} 原子 / {_q1['n_residues']} 残基")
assert _q1["n_atoms"] == 3 and _q1["n_residues"] == 3, _q1
assert all(0 <= i < st3["n_atoms"] for i in _q1["indices"]), "选择结果必须是原始编号"
_q2 = show(client.get(f"/api/session/{sid}/query", params={"expr": "prop mass > 12"}),
           "GET /query (prop mass > 12)")
P(f"      选择语句「{_q2['query']}」-> {_q2['n_atoms']} 原子")
assert _q2["n_atoms"] > 1000, _q2
_q3 = show(client.get(f"/api/session/{sid}/query", params={"expr": "resname NOPE"}),
           "GET /query (空结果给出名字提示)")
P(f"      空结果提示: {_q3.get('hint', '')[:80]}")
assert _q3["n_atoms"] == 0 and _q3.get("hint"), _q3
# 坏语句必须 400 + 可读原因（不能让用户看栈）
_qbad = client.get(f"/api/session/{sid}/query", params={"expr": "resid 1:10 and"})
P(f"      坏语句 -> HTTP {_qbad.status_code}: "
  f"{(_qbad.json().get('detail') or '')[:70]}")
assert _qbad.status_code == 400, _qbad.status_code
assert "选择语句无效" in (_qbad.json().get("detail") or ""), _qbad.json()

# 结构检索（用户要求：搜索要具体到 ETH32 这类残基 + 编号）
sr = show(client.get(f"/api/session/{sid}/search",
                     params={"q": "MET1", "limit": 3}), "GET /search (MET1)")
P(f"      搜 MET1 -> {[(m['label'], m['n_atoms']) for m in sr['matches']]}")
assert sr["matches"] and sr["matches"][0]["label"] == "MET1", sr
_si = show(client.get(f"/api/session/{sid}/search", params={"q": "5"}),
           "GET /search (原子索引 5)")
P(f"      搜 5 -> {[(m['label'], m['n_atoms']) for m in _si['matches']]}")
assert _si["matches"] and _si["matches"][0]["indices"] == [5], _si
assert "重原子" not in [p["name"] for p in pres], (
    "「重原子」预设已按用户要求删除", [p["name"] for p in pres])
_sc = show(client.get(f"/api/session/{sid}/search",
                      params={"q": "C1", "limit": 2}), "GET /search (C1 = 第 1 个碳)")
P(f"      搜 C1 -> {[_m['label'] for _m in _sc['matches']]}")
assert _sc["matches"], ("C1 应当能搜到（元素+序号）", _sc)
_sp = show(client.get(f"/api/session/{sid}/search",
                      params={"q": "SOL", "limit": 2}), "GET /search (SOL 前缀)")
P(f"      搜 SOL -> total {_sp['total']}，前 2 个 {[m['label'] for m in _sp['matches']]}")
assert _sp["total"] > 100 and len(_sp["matches"]) == 2, _sp

h3 = show(client.get(f"/api/session/{sid}/hierarchy"), "GET /hierarchy")
P(f"      组分 {len(h3['components'])} / 链 {len(h3['chains'])} / "
  f"残基 {h3['n_residues']}（键表 {h3['has_bonds']}）")
assert h3["components"] and h3["chains"]
_ch = h3["chains"][0]
r3 = show(client.get(f"/api/session/{sid}/hierarchy",
                     params={"segid": _ch["segid"], "resname": _ch["resname"],
                             "limit": 3}),
          "GET /hierarchy (残基分页)")
_tot3 = (r3["residues"] or {}).get("total", 0)
P(f"      链 {_ch['label']} 的残基 {_tot3} 个（返回 {len(r3['residues']['rows'])} 行）")
assert _tot3 > 0 and r3["residues"]["rows"], r3.get("residues")
_first = r3["residues"]["rows"][0]
e3 = show(client.get(f"/api/session/{sid}/expand",
                     params={"level": "residue", "index": _first["first_index"]}),
          "GET /expand (残留)")
P(f"      点原子 #{_first['first_index']} 展开成「{e3['label']}」：{e3['n_atoms']} 原子")
assert e3["n_atoms"] == _first["n_atoms"], (e3["n_atoms"], _first["n_atoms"])

# 9d) 分子 / 聚合物链识别（结构层级）
P("\n   分子与聚合物链识别（结构层级）:")
_rec = show(client.get(f"/api/session/{sid}/recognition"), "GET /recognition")
_rs = _rec["summary"]
P(f"      {_rs['total_molecules']} 分子 / {_rs['total_polymer_chains']} 聚合物链 / "
  f"{_rs['total_network_components']} 网络 / {_rs['total_unknown']} 未识别；"
  f"键表来源 {_rs['bond_source']}，用时 {_rs['elapsed_sec']}s")
assert _rs["bond_source"] == "topology", _rs
assert _rs["assigned_atoms"] == _rs["total_atoms"], _rs
assert _rs["unassigned_atoms"] == 0, _rs
assert _rec["sections"], _rec
P(f"      类别：{[(s['label'], s['n_objects'], len(s['groups'])) for s in _rec['sections']]}")
# 每个原子的归属之和必须等于体系原子数（不重不漏，recognize.md 测试 4）
_tot = sum(g["n_atoms"] for s in _rec["sections"] for g in s["groups"])
assert _tot == _rs["total_atoms"], (_tot, _rs["total_atoms"])
P(f"      各类别原子数之和 {_tot} == 体系 {_rs['total_atoms']}（不重不漏）")
# 二次请求必须命中缓存（识别只算一次）
_rec2 = show(client.get(f"/api/session/{sid}/recognition"), "GET /recognition（缓存）")
assert _rec2["cached"] is True, _rec2.get("cached")
assert _rec2["summary"]["elapsed_sec"] == _rs["elapsed_sec"], "第二次请求不应重算"
_s0 = _rec["sections"][0]
_g = _s0["groups"][0]
_obj = show(client.get(f"/api/session/{sid}/recognition/objects",
                       params={"group": _g["group_index"], "limit": 3}),
            "GET /recognition/objects")
P(f"      组 {_g['label']}（{_g['n_objects']} 个）前 3 个："
  f"{[(o['name'], o['n_atoms'], o['chain_type_label']) for o in _obj['rows']]}")
assert _obj["total"] == _g["n_objects"] and _obj["rows"], _obj
_exp = show(client.get(f"/api/session/{sid}/recognition/expand",
                       params={"object_type": "group", "object_id": _g["group_id"]}),
            "GET /recognition/expand（整组）")
P(f"      点整组 {_g['label']} -> {_exp['n_atoms']} 原子")
assert _exp["n_atoms"] == _g["n_atoms"], (_exp["n_atoms"], _g["n_atoms"])
_o1 = _obj["rows"][0]
_e1 = show(client.get(f"/api/session/{sid}/recognition/expand",
                      params={"object_type": _o1["object_type"],
                              "object_id": _o1["object_id"]}),
           "GET /recognition/expand（单个对象）")
P(f"      点单个对象 {_o1['name']} -> {_e1['n_atoms']} 原子")
assert _e1["n_atoms"] == _o1["n_atoms"], (_e1["n_atoms"], _o1["n_atoms"])
assert all(0 <= i < _rs["total_atoms"] for i in _e1["indices"]), "索引必须是原始编号"
# 类别展开（聚合物 / 小分子）也要能直接选中
_ep = show(client.get(f"/api/session/{sid}/recognition/expand",
                      params={"object_type": "section", "object_id": "polymer"}),
           "GET /recognition/expand（整个类别）")
P(f"      选中「聚合物」类别 -> {_ep['n_atoms']} 原子")
assert _ep["n_atoms"] > 0
# 人工确认：改名 / 改类型只影响显示，不动原子归属
_cf = show(client.post(f"/api/session/{sid}/recognition/confirm",
                       json={"object_id": _o1["molecule_id"], "name": "手工改名"}),
           "POST /recognition/confirm")
assert _cf["detail"]["name"] == "手工改名", _cf["detail"]
assert _cf["detail"]["status"] == "manually_confirmed", _cf["detail"]
assert len(_cf["detail"]["atom_indices"]) == _o1["n_atoms"], "人工确认不能改归属"
P(f"      人工确认 -> {_cf['detail']['name']}（{_cf['detail']['status_label']}），"
  f"原子数仍为 {len(_cf['detail']['atom_indices'])}")
# 换体系必须换掉旧识别结果：新建一个 MS 体系，识别结果不能沿用 AdK 的
_ms = show(client.post("/api/session", json={"topology": str(ROOT / "122C12-450K.xsd")}),
           "POST /session（MS 体系）")
_sid_ms = _ms["sid"]
_rec_ms = show(client.get(f"/api/session/{_sid_ms}/recognition"), "GET /recognition（MS）")
_rm = _rec_ms["summary"]
P(f"      MS 体系：{_rm['total_molecules']} 分子 / {_rm['total_polymer_chains']} 链 / "
  f"{_rm['total_network_components']} 网络（AdK 为 {_rs['total_molecules']} 分子）")
assert _rm["total_atoms"] != _rs["total_atoms"], "换了体系必须重算识别"
assert _rm["total_polymer_chains"] == 122 and _rm["total_network_components"] == 1, _rm
assert _rm["assigned_atoms"] == _rm["total_atoms"]
# 「先选中一条链，再取这条链的氢」——多选工作流的两侧数据都要能拿到。
# 交集在前端做（元素筛选作用在当前已选上），这里验证两侧口径：
# 链原子必须 ⊆ 体系，元素集合可用，且交集既非空也不等于全体系的氢。
_ch1 = show(client.get(f"/api/session/{_sid_ms}/recognition/objects",
                       params={"group": 0, "limit": 1}),
            "GET /recognition/objects（MS 组 0）")
_o = _ch1["rows"][0]
_ce = show(client.get(f"/api/session/{_sid_ms}/recognition/expand",
                      params={"object_type": _o["object_type"],
                              "object_id": _o["object_id"]}),
           f"GET /recognition/expand（{_o['name']}）")
_h = show(client.get(f"/api/session/{_sid_ms}/query", params={"expr": "element H"}),
          "GET /query（element H）")
_inter = set(_ce["indices"]) & set(_h["indices"])
P(f"      选中 {_o['name']}（{_ce['n_atoms']} 原子）再取它的氢 -> "
  f"{len(_inter)} 个 H（全体系 {_h['n_atoms']} 个）")
assert 0 < len(_inter) < _ce["n_atoms"], (len(_inter), _ce["n_atoms"])
assert len(_inter) < _h["n_atoms"], "元素筛选不能退化成'全体系的氢'"
# /sets/atoms：元素与预设（链状分子/含环分子/大分子）都要能算成原子索引，
# 前端把两者当作"与层级树同级、可叠加、可取消"的筛选条件。
def _spec_atoms(spec):
    r = client.post(f"/api/session/{_sid_ms}/sets/atoms", json={"spec": spec})
    assert r.status_code == 200, r.text[:300]
    return r.json()


_el_c = _spec_atoms({"elements": ["C"]})
_el_h = _spec_atoms({"elements": ["H"]})
_pr_chain = _spec_atoms({"shape": "chain"})
_pr_ring = _spec_atoms({"shape": "ring"})
_pr_big = _spec_atoms({"min_atoms": 20})
P(f"      元素 C={_el_c['n_atoms']} / H={_el_h['n_atoms']}；"
  f"链状={_pr_chain['n_atoms']}、含环={_pr_ring['n_atoms']}、"
  f"大分子={_pr_big['n_atoms']}（体系 {_rm['total_atoms']}）")
assert _el_c["n_atoms"] == 2424 and _el_h["n_atoms"] == 3172, (_el_c, _el_h)
assert _pr_chain["n_atoms"] == 4636, _pr_chain          # 122 条 C12H26
assert _pr_ring["n_atoms"] == 960, _pr_ring             # 石墨片
assert _pr_big["n_atoms"] == 5596, _pr_big              # 全部都 ≥20 原子
# 「范围（一条链）∩ 元素 H」= 这条链的氢；取消条件后范围原封不动
_scope = set(_ce["indices"])
_inter2 = _scope & set(_el_h["indices"])
P(f"      {_o['name']}（{len(_scope)} 原子）∩ 元素 H = {len(_inter2)} 个；"
  f"取消条件后仍是 {len(_scope)} 原子")
assert len(_inter2) == 26, len(_inter2)
assert len(_scope) == 38, len(_scope)
# 条件之间也能叠加（C ∩ 链状 = 122×12，石墨片的 960 个碳被"含环"排除）
assert len(set(_el_c["indices"]) & set(_pr_chain["indices"])) == 1464
# 算这些条件不能污染分析器（不能凭空多出集合）
_names = [c["name"] for c in
          show(client.get(f"/api/session/{_sid_ms}/components"),
               "GET /components（不该被污染）")["components"]]
P(f"      组分清单未被污染: {_names}")
assert not any(n == "set:_probe" or n.startswith("set:_probe") for n in _names), _names
client.delete(f"/api/session/{_sid_ms}")

# 9e) 分析**运行中**也能选取 / 建集合（用户实测：以前会报「分析正在运行中」）
P("\n   分析运行中的选取（选取与分析互不影响）:")
_base_sel = {"components": ["protein", "water"], "primary": {"mode": "auto"},
             "sets": []}
show(client.post(f"/api/session/{sid}/select", json=_base_sel), "POST /select（先设好）")
show(client.post(f"/api/session/{sid}/run/start", json={
    "selection": _base_sel, "frames": {"max_frames": 10},
    "which": ["rdf", "contact"], "estimate": False}), "POST /run/start（慢分析）")
time.sleep(0.8)
_prog = client.get(f"/api/session/{sid}/run/progress", params={"after": 0}).json()
P(f"      运行状态 {_prog['status']}（当前 {_prog.get('current') or '—'}）")
assert _prog["status"] == "running", _prog
_sel_run = show(client.post(f"/api/session/{sid}/select", json={
    "components": ["protein", "water"], "primary": {"mode": "auto"},
    "sets": [{"name": "运行中建的", "elements": ["O"]}]}),
    "POST /select（运行中）")
assert _sel_run.get("deferred") is True, _sel_run.get("deferred")
P(f"      deferred=True，集合预览 "
  f"{[(x['name'], x.get('n_atoms')) for x in _sel_run['sets']]}"
  f"；候选清单里有它："
  f"{any(c['name'] == 'set:运行中建的' for c in _sel_run.get('catalog') or [])}")
assert any(c["name"] == "set:运行中建的" for c in _sel_run.get("catalog") or []), \
    "运行中新建的集合要出现在候选清单里（下一轮能勾）"
assert any(x["name"] == "set:运行中建的" and x.get("n_atoms", 0) > 0
           for x in _sel_run["sets"]), _sel_run["sets"]
_vis_run = show(client.post(f"/api/session/{sid}/selection",
                            json={"name": "运行中选的", "indices": list(range(30))}),
                "POST /selection（运行中）")
assert _vis_run.get("deferred") is True and _vis_run["set"]["n_atoms"] == 30, _vis_run
# 只读的三维/识别/选择语句接口在运行中必须照常可用
for _p, _kw in (
    (f"/api/session/{sid}/structure", {"params": {"max_atoms": 2000}}),
    (f"/api/session/{sid}/expand", {"params": {"level": "residue", "index": 0}}),
    (f"/api/session/{sid}/query", {"params": {"expr": "element O"}}),
    (f"/api/session/{sid}/recognition", {}),
):
    _r = client.get(_p, **_kw)
    P(f"      运行中 GET {_p.rsplit('/', 1)[-1]}: HTTP {_r.status_code}")
    assert _r.status_code == 200, (_p, _r.text[:200])
show(client.delete(f"/api/session/{sid}/selections/运行中选的"),
     "DELETE /selections（运行中）")
# 取消这一轮，确认"只存不算"的集合在**下一轮**真的生效
show(client.post(f"/api/session/{sid}/run/cancel"), "POST /run/cancel")
for _ in range(120):
    time.sleep(0.3)
    if client.get(f"/api/session/{sid}/run/progress",
                  params={"after": 0}).json()["status"] != "running":
        break
_r2 = show(client.post(f"/api/session/{sid}/run/start", json={
    "selection": {"components": ["protein", "water", "set:运行中建的"],
                  "primary": {"mode": "auto"},
                  "sets": [{"name": "运行中建的", "elements": ["O"]}]},
    "frames": {"max_frames": 3}, "which": ["rg"], "estimate": False}),
    "POST /run/start（下一轮）")
for _ in range(120):
    time.sleep(0.3)
    _p2 = client.get(f"/api/session/{sid}/run/progress",
                     params={"after": 0}).json()
    if _p2["status"] != "running":
        break
_lbl = (_p2["results"].get("rg") or {}).get("title", "")
P(f"      下一轮 Rg 标题: {_lbl}")
assert "运行中建的" in _lbl, ("运行中只存不算的集合必须在下一轮真参与分析", _lbl)
show(client.post(f"/api/session/{sid}/clear"), "POST /clear（收拾干净）")
show(client.post(f"/api/session/{sid}/select", json=_base_sel), "POST /select（复原）")

# 9d') 分析项可行性预判：跑不了的项要能提前说清楚（用户要求"先预判再选"）
P("\n   分析项可行性预判（点「开始分析」之前就拦）:")
_no_comp = show(client.post(f"/api/session/{sid}/select",
                            json={"components": [], "sets": [],
                                  "primary": {"mode": "auto"}}),
                "POST /select（一个组分都不勾）")
_fe = _no_comp.get("feasible") or {}
P(f"      /select 自带预判：{len(_fe)} 项；"
  f"rdf={(_fe.get('rdf') or {}).get('level')}，"
  f"comdist={(_fe.get('comdist') or {}).get('level')}")
assert _fe and (_fe.get("rdf") or {}).get("ok") is False, _fe.get("rdf")
assert "组分" in str((_fe.get("rdf") or {}).get("reason")), _fe.get("rdf")
assert (_fe.get("comdist") or {}).get("ok") is False, _fe.get("comdist")
assert (_fe.get("interface") or {}).get("ok") is False, _fe.get("interface")
assert (_fe.get("rg") or {}).get("ok") is True, _fe.get("rg")
# MSD / 密度也必须按"有没有勾组分"拦：以前预判说 ok、跑完却是 skip（实测）
assert (_fe.get("msd") or {}).get("ok") is False, _fe.get("msd")
assert (_fe.get("density") or {}).get("ok") is False, _fe.get("density")
# 界面默认一个都不勾 → 传空 which 必须报错，而不是"悄悄跑全部 14 项"
_e = client.post(f"/api/session/{sid}/run/start",
                 json={"selection": {"components": [], "sets": [],
                                     "primary": {"mode": "auto"}},
                       "which": [], "estimate": False})
P(f"      空 which -> HTTP {_e.status_code}: "
  f"{(_e.json().get('detail') or '')[:46]}")
assert _e.status_code == 400, _e.status_code
assert "勾选" in str(_e.json().get("detail")), _e.json()
_fg = show(client.get(f"/api/session/{sid}/feasible"), "GET /feasible")
P(f"      GET /feasible: {len(_fg['feasible'])} 项")
assert len(_fg["feasible"]) >= 12, _fg
_two = show(client.post(f"/api/session/{sid}/select",
                        json={"components": ["protein", "water"], "sets": [],
                              "primary": {"mode": "auto"}}),
            "POST /select（勾 2 个组分）")
_fe2 = _two.get("feasible") or {}
P(f"      勾 2 个组分后：rdf={(_fe2.get('rdf') or {}).get('level')}，"
  f"comdist={(_fe2.get('comdist') or {}).get('level')}")
assert (_fe2.get("rdf") or {}).get("ok") is True, _fe2.get("rdf")
assert (_fe2.get("comdist") or {}).get("ok") is True, _fe2.get("comdist")
# 界面上「勾了组分 → RDF 却仍然跑不了」的 bug 就断在这里：可行性预判读的是
# **后端**的组分（az.components），前端只在"开始分析"时把选择塞进请求体，
# 于是勾了胶囊、预判里仍是"没勾组分"。现在前端每次改动都会 POST /select 同步，
# 所以这条不变量必须成立：/select 之后再单独 POST /feasible，rdf 必须是可用的。
_fsync = show(client.post(f"/api/session/{sid}/feasible", json={"frames": {}}),
              "POST /feasible（/select 之后，只带 frames —— 前端就是这样问的）")["feasible"]
P(f"      同步后 rdf={(_fsync.get('rdf') or {}).get('level')}，"
  f"density={(_fsync.get('density') or {}).get('level')}")
assert (_fsync.get("rdf") or {}).get("ok") is True, _fsync.get("rdf")
assert (_fsync.get("density") or {}).get("ok") is True, _fsync.get("density")
# 反向：取消勾选后再问一次，必须重新变回 blocked（不能"沿用上一次的放行"）
_off = show(client.post(f"/api/session/{sid}/select",
                        json={"components": [], "sets": [],
                              "primary": {"mode": "auto"}}),
            "POST /select（全部取消勾选）")["feasible"]
assert (_off.get("rdf") or {}).get("ok") is False, _off.get("rdf")
show(client.post(f"/api/session/{sid}/select",
                 json={"components": ["protein", "water"], "sets": [],
                       "primary": {"mode": "auto"}}),
     "POST /select（恢复 2 个组分）")
# **帧选择**也要算进去：把最大帧数设成 1、或把时间窗收窄，MSD 立刻跑不了
# （用户实测：预判说能跑，一跑报「MSD 至少需要 2 帧」）
_f1 = show(client.post(f"/api/session/{sid}/feasible",
                       json={"frames": {"max_frames": 1}}),
           "POST /feasible（max_frames=1）")["feasible"]
P(f"      max_frames=1 -> msd={(_f1.get('msd') or {}).get('level')}："
  f"{(_f1.get('msd') or {}).get('reason', '')[:40]}")
assert (_f1.get("msd") or {}).get("ok") is False, _f1.get("msd")
assert "帧" in str((_f1.get("msd") or {}).get("reason")), _f1.get("msd")
_f2 = show(client.post(f"/api/session/{sid}/feasible",
                       json={"frames": {"start_ps": 0, "stop_ps": 0.5}}),
           "POST /feasible（时间窗只剩 1 帧）")["feasible"]
assert (_f2.get("msd") or {}).get("ok") is False, _f2.get("msd")
_f3 = show(client.post(f"/api/session/{sid}/feasible",
                       json={"frames": {"max_frames": 10}}),
           "POST /feasible（10 帧）")["feasible"]
P(f"      max_frames=10 -> msd={(_f3.get('msd') or {}).get('level')}")
assert (_f3.get("msd") or {}).get("ok") is True, _f3.get("msd")

# 可视化选择 → 集合（走原有 Set 机制）
_vis = show(client.post(f"/api/session/{sid}/selection",
                        json={"name": "可视A", "indices": _first["indices"]}),
            "POST /selection (创建集合)")
P(f"      {_vis['set']['label']}")
assert _vis["set"]["name"] == "set:可视A"
assert "set:可视A" in [c["name"] for c in _vis["selection"]["components"]]
assert [x["name"] for x in _vis["selections"]] == ["可视A"], _vis["selections"]

# 只勾这个可视化集合跑 MSD：分组必须就是它（证明选择真的进了分析引擎）
_ = show(client.post(f"/api/session/{sid}/clear"), "POST /clear")
# 前端只回传名字（上万个索引不必来回传）；后端用 self.selections 把索引补回去 ——
# 这里刻意传 `elements: []` 这种"没有筛选条件"的空壳，检验补全逻辑真的生效。
show(client.post(f"/api/session/{sid}/run/start",
                 json={"which": ["msd"], "estimate": False, "frames": {},
                       "selection": {"components": ["set:可视A"],
                                     "sets": [{"name": "可视A"}]}}),
     "POST /run/start (MSD, 只用可视化集合)")
for _ in range(600):
    prog3 = client.get(f"/api/session/{sid}/run/progress?after=0").json()
    if prog3["status"] != "running":
        break
    time.sleep(0.2)
_msd3 = client.get(f"/api/session/{sid}/state").json()["results"].get("msd") or {}
_grp3 = sorted({c["group"] for c in _msd3.get("curves", []) if c.get("group")})
P(f"      MSD 分组: {_grp3}")
assert _grp3 == ["set:可视A"], ("MSD 必须用可视化集合的原子", _grp3)

# 删除选择：组分与集合都摘掉
_d3 = show(client.delete(f"/api/session/{sid}/selections/可视A"),
           "DELETE /selections/可视A")
P(f"      删除后组分: {[c['name'] for c in _d3['components']]}")
assert "set:可视A" not in [c["name"] for c in _d3["components"]]
assert _d3["selections"] == []

# 复原：把上面那批集合重新定义回来（后面的「刷新恢复」段要用 state.sets）
show(client.post(f"/api/session/{sid}/select",
                 json={"components": ["set:C"], "sets": sel_sets}),
     "POST /select (复原集合定义)")


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

# ---------------------------------------------------------------- 刷新恢复 / 增量追加
P("\n   刷新恢复（state）与增量追加（extend）:")
# 先用实时接口跑 1 项，模拟真实前端
st0 = show(client.post(f"/api/session/{sid}/run/start",
                       json={"which": ["rg"], "estimate": False, "frames": {}}),
           "POST /run/start (rg)")
for _ in range(200):
    prog = client.get(f"/api/session/{sid}/run/progress?after=0").json()
    if prog["status"] != "running":
        break
    time.sleep(0.2)
P(f"      运行结束: {prog['status']}，结果 {sorted(prog['results'])}")
assert set(prog["results"]) == {"rg"}, prog["results"]

# 模拟 F5：只读 /state，不依赖任何前端状态
st = show(client.get(f"/api/session/{sid}/state"), "GET /state (模拟 F5)")
P(f"      恢复：结果 {sorted(st['results'])} | 组分 {len(st['components'])} "
  f"| 帧 {(st.get('frames') or {}).get('n_frames')} | job {st['job']['status']}")
assert set(st["results"]) == {"rg"}
assert st["components"] and st["frames"], "state 缺少组分或帧选择"
assert st["selection"]["primary_label"], "state 缺少当前主链"
# 刷新恢复必须把「分析集合（Set）」接回来（用户实测：一刷新集合就没了）
P(f"      state 里的集合: {[(x.get('name'), x.get('on')) for x in st.get('sets') or []]}")
assert st.get("sets"), "state 必须回传集合定义（刷新后不能丢）"
assert st.get("sets_info"), "state 必须回传集合预览（原子数 / 追踪方式）"
assert st.get("elements"), "state 必须回传可选元素（集合编辑器要用）"
assert st["selection"].get("primary"), "state 必须回传主链那一档（auto / component …）"
# 候选清单（catalog）vs 勾选清单（active）：刷新后组分胶囊要能重新勾回来
_catS = [c["name"] for c in st.get("catalog") or []]
_actS = list(st["selection"].get("active") or [])
P(f"      state 候选 {len(_catS)} 项 / 勾选 {len(_actS)} 项：{_catS[:6]}")
assert _catS, "state 必须回传候选清单 catalog（用户实测：刷新后组分没了）"
assert _actS and set(_actS) <= set(_catS), ("勾选清单必须是候选清单的子集",
                                            _actS, _catS[:8])

# 增量追加：只补 ree，rg 结果必须保留
show(client.post(f"/api/session/{sid}/run/start",
                 json={"which": ["ree"], "extend": True, "estimate": False,
                       "frames": {}}), "POST /run/start (extend ree)")
for _ in range(300):
    prog2 = client.get(f"/api/session/{sid}/run/progress?after=0").json()
    if prog2["status"] != "running":
        break
    time.sleep(0.2)
st2 = client.get(f"/api/session/{sid}/state").json()
P(f"      扩展后 state 结果: {sorted(st2['results'])}")
assert {"rg", "ree"} <= set(st2["results"]), st2["results"].keys()

# ---------------------------------------------------------------- 加组分后必须重算
# 用户实测：第一轮只勾一两个组分跑了 MSD，第二轮**加了组分**再跑 —— 后端把 MSD
# 当成"已有结果"直接跳过，界面上看起来就是"加了组分却不跑"（图还是旧组分算的）。
P("\n   加组分 / 改参数后扩展运行必须重算:")
show(client.post(f"/api/session/{sid}/clear"), "POST /clear（先清干净）")


def _run_ext(tag, which, comps, params=None, extend=True):
    p0 = show(client.post(f"/api/session/{sid}/run/start",
                          json={"which": which, "extend": extend, "estimate": False,
                                "frames": {}, "params": params or {},
                                "selection": {"components": comps}}), tag)
    for _ in range(600):
        p = client.get(f"/api/session/{sid}/run/progress?after=0").json()
        if p["status"] != "running":
            return p0, p
        time.sleep(0.2)
    raise AssertionError(f"{tag} 超时")


_, f1 = _run_ext("POST /run/start (msd, 只勾 water)", ["msd"], ["water"])
m1 = client.get(f"/api/session/{sid}/state").json()["results"]["msd"]
g1 = sorted({c["group"] for c in m1["curves"] if c.get("group")})
P(f"      第一轮 MSD 分组: {g1}")
assert g1 and not any("protein" in g for g in g1), g1

p0b, f2 = _run_ext("POST /run/start (+protein, extend)",
                   ["msd"], ["water", "protein"])
m2 = client.get(f"/api/session/{sid}/state").json()["results"]["msd"]
g2 = sorted({c["group"] for c in m2["curves"] if c.get("group")})
P(f"      加组分后 MSD 分组: {g2}（start 时 completion={p0b['completion']}）")
assert any("protein" in g for g in g2), (
    "加了组分就必须重算 MSD（用户实测：以前直接跳过不跑）", g2)
assert p0b["completion"] == [], ("组分变了 → 旧 MSD 结果作废，不该被算作已保留",
                                 p0b["completion"])

# 反过来：输入完全没变 → 保留，不重算
p0c, f3 = _run_ext("POST /run/start (原样再跑)", ["msd"], ["water", "protein"])
P(f"      原样再跑: completion={p0c['completion']} / {f3['message']}")
assert p0c["completion"] == ["msd"], ("输入没变应保留", p0c["completion"])
assert "无需重算" in f3["message"], f3["message"]

# 清空
show(client.post(f"/api/session/{sid}/clear"), "POST /clear")
st3 = client.get(f"/api/session/{sid}/state").json()
P(f"      清空后结果数: {len(st3['results'])}")
assert not st3["results"]
assert client.get("/api/session/nope/state").status_code == 404

d = show(client.delete(f"/api/session/{sid}"), "DELETE /api/session/{sid}")
P(f"      已关闭 {d.get('closed', sid)}")
r404b = client.get(f"/api/session/{sid}/info")
P(f"      关闭后再访问 -> HTTP {r404b.status_code}")
assert r404b.status_code == 404
P(f"      导出目录已清理: {not os.path.isdir(e['dir'])}")
P(f"      已关闭 {d['closed']}")
r404b = client.get(f"/api/session/{sid}/info")
P(f"      关闭后再访问 -> HTTP {r404b.status_code}")
assert r404b.status_code == 404
P(f"      导出目录已清理: {not os.path.isdir(e['dir'])}")

# ---------------------------------------------------------------------------
# 集合「重名自动加序号」必须只看**现在**（用户实测的 bug：删掉 A 再建 A 得到 A_2）
P("\n" + "=" * 76)
P("[集合重名编号：只看现在，不看历史]")
P("=" * 76)
_s2 = show(client.post("/api/session", json={"topology": TOP, "trajectory": XTC}),
           "POST /api/session（编号检查用）")
sid2 = _s2["sid"]


def _mk_set(sid_, name, idx=(0, 1)):
    r = client.post(f"/api/session/{sid_}/selection",
                    json={"name": name, "indices": list(idx)})
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    return (j.get("set") or {}).get("set_name") or j.get("name")


def _saved(sid_):
    r = client.get(f"/api/session/{sid_}/selections")
    assert r.status_code == 200, r.text[:200]
    return [x["name"] for x in r.json()["selections"]]


_a = _mk_set(sid2, "dup")
assert _a == "dup", _a
P(f"      建 dup        -> {_a}")
_b = _mk_set(sid2, "dup")
assert _b == "dup_2", ("现在已存在同名 → 应加序号", _b)
P(f"      再建 dup      -> {_b}")
# 删掉（后端 DELETE：定义 / 已保存选择一起清）
client.delete(f"/api/session/{sid2}/selections/dup_2")
_c = _mk_set(sid2, "dup")
assert _c == "dup_2", ("dup 还在 → 新集合应是 dup_2", _c)
P(f"      删 dup_2 再建 -> {_c}")
client.delete(f"/api/session/{sid2}/selections/dup")
client.delete(f"/api/session/{sid2}/selections/dup_2")
assert _saved(sid2) == [], _saved(sid2)
_d = _mk_set(sid2, "dup")
assert _d == "dup", ("现在没有 dup 了 → 应直接叫 dup（不能翻历史给 dup_2）", _d)
P(f"      全删后再建     -> {_d}")

# 复现用户看到的那条路径：界面只把集合从列表里删掉（旧版没发 DELETE），
# 靠 /select 同步 → 后端必须把残留的"已保存选择"清掉，同名才能复用
client.delete(f"/api/session/{sid2}/selections/dup")
assert _saved(sid2) == []
client.post(f"/api/session/{sid2}/selection",
            json={"name": "dup2", "indices": [0, 1]})
show(client.post(f"/api/session/{sid2}/select",
                 json={"components": [], "sets": [],
                       "primary": {"mode": "auto"}}),
     "POST /select (sets=[]，模拟界面删光集合后同步)")
assert _saved(sid2) == [], ("同步后旧记录必须清掉，否则重名判断会翻历史", _saved(sid2))
_e = _mk_set(sid2, "dup2")
assert _e == "dup2", ("旧记录清掉后同名应可直接用（以前会得到 dup2_2）", _e)
P(f"      删光后同步再建 -> {_e}")
show(client.delete(f"/api/session/{sid2}"), "DELETE /api/session/{sid2}")

P("\n" + "=" * 76)
P(f"后端 API 全部通过，总耗时 {time.time() - t0:.1f}s")
P("=" * 76)
