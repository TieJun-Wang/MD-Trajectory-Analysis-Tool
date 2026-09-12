# -*- coding: utf-8 -*-
"""导出功能验证：目标目录、模块选择、文件形式、打开目录、只回报目录+时间。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
B = "http://127.0.0.1:8000"

pass_n = 0
fails: list[str] = []


def ok(cond, label, extra=""):
    global pass_n
    if cond:
        pass_n += 1
        print(f"  ✓ {label}" + (f"  {extra}" if extra else ""), flush=True)
    else:
        fails.append(label)
        print(f"  ✗ {label}" + (f"  {extra}" if extra else ""), flush=True)


def req(method, path, body=None, expect_error=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(B + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        if expect_error:
            return {"__status": e.code, "detail": e.read().decode("utf-8", "replace")}
        raise


print("=" * 74)
print("导出功能验证")
print("=" * 74)

D = ROOT / "adk_oplsaa.tpr"
X = ROOT / "adk_oplsaa.xtc"
s = req("POST", "/api/session", {"topology": str(D), "trajectory": str(X)})
sid = s["sid"]
print(f"会话 {sid}\n", flush=True)

req("POST", f"/api/session/{sid}/run",
    {"which": ["rg", "ree", "msd"], "frames": {"interval_ps": 200},
     "params": {}, "selection": {"primary": {"mode": "auto"},
                                 "components": ["protein", "water"]}})

# ------------------------------------------------------------ 1) 目标目录
custom = ROOT / "_export_probe"
shutil.rmtree(custom, ignore_errors=True)
r = req("POST", f"/api/session/{sid}/export",
        {"formats": ["csv"], "outdir": str(custom), "which": ["rg"]})
print("\n[1] 目标目录 + 模块选择 + 单一格式")
ok(os.path.isdir(r["dir"]), "自定义目标目录已创建", r["dir"])
ok(str(custom) in r["dir"], "导出到了指定目录")
names = {f["name"] for f in r["files"]}
ok(names and all(n.startswith("rg") for n in names),
   "只导出了勾选的模块 rg", f"{sorted(names)}")
ok(all(n.endswith(".csv") for n in names), "只导出了 csv 格式")
ok("time" in r and r["time"], "返回了导出时间", r["time"])
ok(r["which"] == ["rg"], "回报了本次导出的分析项", str(r["which"]))
ok(r["formats"] == ["csv"], "回报了本次的文件形式", str(r["formats"]))

# ------------------------------------------------------------ 2) 全部模块 + 多格式
r2 = req("POST", f"/api/session/{sid}/export",
         {"formats": ["csv", "png", "excel"], "outdir": str(custom)})
names2 = {f["name"] for f in r2["files"]}
ok(any(n.startswith("msd") for n in names2), "不传 which 时导出全部模块")
ok(any(n.endswith(".png") for n in names2), "导出了 PNG")
ok(any(n.endswith(".xlsx") for n in names2), "导出了 Excel")
ok(r2["n_csv"] > 0 and r2["n_png"] > 0, "回报了 CSV / PNG 计数",
   f'CSV {r2["n_csv"]} / PNG {r2["n_png"]}')

# ------------------------------------------------------------ 3) 参数校验
print("\n[2] 参数校验")
e = req("POST", f"/api/session/{sid}/export",
        {"formats": [], "outdir": str(custom)}, expect_error=True)
ok(e.get("__status") == 400, "空格式列表被拒绝", str(e.get("detail", ""))[:60])
e = req("POST", f"/api/session/{sid}/export",
        {"formats": ["pdf"], "outdir": str(custom)}, expect_error=True)
ok(e.get("__status") == 400 and "csv" in str(e.get("detail")), "不支持的格式被拒绝",
   str(e.get("detail", ""))[:70])
e = req("POST", f"/api/session/{sid}/export",
        {"formats": ["csv"], "which": ["nope"], "outdir": str(custom)},
        expect_error=True)
ok(e.get("__status") == 400, "导出不存在的分析项被拒绝", str(e.get("detail", ""))[:60])

# ------------------------------------------------------------ 4) 打开目录
print("\n[3] 打开目标目录")
o = req("POST", f"/api/session/{sid}/open-dir", {"dir": r["dir"]})
ok(o.get("ok") and o.get("dir") == r["dir"], "打开已存在目录成功", o.get("dir"))
o2 = req("POST", f"/api/session/{sid}/open-dir", {}, expect_error=False)
ok(o2.get("dir") == r["dir"], "不传 dir 时打开最近导出的目录", o2.get("dir"))
e = req("POST", f"/api/session/{sid}/open-dir",
        {"dir": str(ROOT / "no_such_dir_xyz")}, expect_error=True)
ok(e.get("__status") == 400, "不存在的目录被拒绝", str(e.get("detail", ""))[:60])

# ------------------------------------------------------------ 5) 下载 & 越权
print("\n[4] 自定义目录下的文件下载（含路径逃逸防护）")
f0 = r["files"][0]
u = f0["url"]
with urllib.request.urlopen(B + u, timeout=120) as resp:
    blob = resp.read()
ok(len(blob) > 100, "从自定义目录下载文件成功", f"{f0['name']} {len(blob)} B")
esc = req("GET", f"/api/session/{sid}/files/..%2F..%2Fsetup.py",
          expect_error=True)
ok(esc.get("__status") in (400, 404), "路径逃逸被挡住", str(esc.get("__status")))

# ------------------------------------------------------------ 6) /outputs
o3 = req("GET", f"/api/session/{sid}/outputs")
ok(o3.get("last_export") and o3["last_export"]["dir"] == r["dir"],
   "/outputs 带出最近导出信息（供页面显示目录+时间）")

req("DELETE", f"/api/session/{sid}")
shutil.rmtree(custom, ignore_errors=True)

print("\n" + "=" * 74)
print(f"通过 {pass_n} 项" + (f"，失败 {len(fails)} 项: {' | '.join(fails)}"
                           if fails else "，全部通过"))
print("=" * 74)
sys.exit(1 if fails else 0)
