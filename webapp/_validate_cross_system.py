# -*- coding: utf-8 -*-
"""**跨体系**回归：把作者留在每个体系里的原始输出，逐项与我们的实现对比。

为什么需要它：原来 `_validate_supporting_info.py` 只用了**1 个** CG 体系 + 1 个 20mer
体系，验证面太窄（换个体系换个脂质种类就可能翻车）。`Supporting_Info` 里其实有
**45 个 CG 体系 + 12 个 20mer 变体**，每个都带作者的输出文件，而且作者的分析脚本
（`Lipid_RDF.py` / `Rg.py` / `COM.py` / `Water.py` / `PONI_density.py` /
`Area_Compressibility.py`）就在数据集根目录 —— 口径可以逐行照抄。

本脚本按作者的原始口径复现 6 个量：

  ① `rg`        PONI 组（resname P1c/P1t/P1e）   vs `Rg/3-run/gyrate.xvg`（gmx gyrate）
  ② `rdf2d`     每种脂质一个头基珠子、**逐帧**膜质心之上的半叶、350 bin/7 nm
                                                vs `Lipid_RDF/3-run/<L>_<L>_rdf.txt`
  ③ `density`   PONI 组质量密度剖面               vs `PONI_dens/3-run/poni_density.xvg`
  ④ `comdist`   聚合物–膜 z 距离（不取最小镜像）   vs `COM/3-run/z_distance.csv`
  ⑤ 水珠子计数   逐帧膜质心 ±5 Å 内的 W 珠子数     vs `Water/3-run/Waters.csv`
  ⑥ 膜面积/KA   由轨迹盒矢量算 Lx·Ly、KA=kT·⟨A⟩/MSF vs `KA/3-run/msf.dat` /`KA_conv.csv`

用法：``python -u webapp/_validate_cross_system.py [体系数上限]``（默认全部）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mdta.analysis.interface import (analyze_density, analyze_rdf_inplane,   # noqa: E402
                                     density_profile)
from mdta.io import load_trajectory                                          # noqa: E402
from mdta.selection import select                                            # noqa: E402

SI = Path(os.environ.get("MDTA_SI_DIR", r"C:\temp\dataset\Supporting_Info"))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9

#: 作者脚本里的口径（逐行照抄，见 Supporting_Info/*.py）
LIPIDS = {
    "EC":   {"POPE": "PO4", "POPG": "PO4", "TOCL": "GLC"},
    "MRSA": {"POPGK": "PO4", "POPG": "PO4", "TOCL": "GLC"},
    "RBC":  {"POPC": "PO4", "OSM": "PO4", "POPE": "PO4", "POPS": "PO4", "CHOL": "ROH"},
}
RDF_MAX_NM, BIN_NM, START_PS = 7.0, 0.02, 4_000_000
KB_MN_M_PER_K, T_K = 1.380649e-20, 310.0        # Area_Compressibility.py

rows: list[dict] = []


def num(p):
    """读作者输出的数值表：兼容 xvg（空白分隔 + #/@ 注释）与 csv（逗号分隔 + 表头）。

    作者的 `z_distance.csv` / `Waters.csv` 是**逗号**分隔且 `comments=''`（表头留在
    文件里），`np.loadtxt` 默认按空白切分会把整行当一个 token 而报
    "could not convert string 'Time(碌s),Z_distance(nm)' to float64" —— 所以这里自己
    逐行解析：跳过注释/表头，逗号或空白都收。
    """
    rows = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s[0] in "#@;":
                continue
            try:
                rows.append([float(x) for x in s.replace(",", " ").split()])
            except ValueError:
                continue                      # 表头行
    return np.asarray(rows, float)


def systems() -> list[Path]:
    out = []
    for p in sorted(SI.rglob("analysis_output")):
        d = p.parent.parent
        if d not in out:
            out.append(d)
    return out


def run_system(sysdir: Path) -> dict:
    name = sysdir.name
    memb = next((m for m in LIPIDS if m in name), None)
    if memb is None:
        return {}
    rundir = sysdir / "6_CG-takeCURRENT"
    tpr, xtc = rundir / "3-run.tpr", rundir / "3-run_center.xtc"
    if not (tpr.is_file() and xtc.is_file()):
        return {}
    res: dict = {"sys": name, "memb": memb}
    mdt = load_trajectory(str(tpr), str(xtc))
    u = mdt.universe
    poly = select(u, "resname " + " ".join(f"{name.split('_')[1]}{s}" for s in "cte"))
    mem = select(u, "resname " + " ".join(LIPIDS[memb]))
    frames = np.asarray([ts.time for ts in u.trajectory], dtype=float)
    idx = np.where(frames >= START_PS)[0]
    from mdta.preprocess import select_frames
    sel = select_frames(np.asarray(mdt.times_ps, float)[idx])
    res["n_frames"] = sel.n_frames

    # ---------------------------------------------------------------- ① Rg
    try:
        from mdta.analysis.conformation import analyze_rg
        r = analyze_rg(mdt, poly, sel, label="PONI")
        mine = float(r.summary["Rg mean"]) / 10.0                    # Å → nm
        ref = num(rundir / "analysis_output/Rg/3-run/gyrate.xvg")[:, 1]
        res["rg"] = (mine, float(np.mean(ref)))
    except Exception as exc:                                          # noqa: BLE001
        res["rg_err"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------- ② 面内 RDF (2D)
    try:
        groups = {L: select(u, f"resname {L} and name {b}") for L, b in LIPIDS[memb].items()}
        r2 = analyze_rdf_inplane(mdt, groups, sel, plane_axis=2, rmax=RDF_MAX_NM * 10,
                                 nbins=int(RDF_MAX_NM / BIN_NM), slab_ref=mem,
                                 slab_side="upper")
        cors = {}
        skips = []
        for L in groups:
            f = rundir / f"analysis_output/Lipid_RDF/3-run/{L}_{L}_rdf.txt"
            if not f.is_file():
                continue
            ref = num(f)
            r_nm, ref_g = ref[:, 0], ref[:, 1]
            if np.isfinite(ref_g).sum() < 10:
                # 作者自己的参考在这条曲线上就是 NaN（0/0）：实测 POPS 在某些体系里
                # 几乎全在下半叶，上半叶每帧只有 0–1 个珠子 → 他们的脚本算出全 NaN。
                # 我们的实现会**直接不画这条曲线**（<2 个珠子算不出自 RDF），
                # 所以这里记为"参考无效"，不计入相关系数统计。
                skips.append(f"{L}(作者参考全 NaN)")
                continue
            cur = next((c for c in r2.curves if c.label.startswith(L)), None)
            if cur is None:
                skips.append(f"{L}(叶层内珠子 <2，本工具按定义跳过)")
                continue
            g = np.asarray(cur.y, float)
            n = min(len(g), len(ref_g))
            cor = float(np.corrcoef(g[:n], ref_g[:n])[0, 1])
            # 峰位两边都用**同一口径**：2.2 Å ≤ r ≤ 25 Å 内的最大值（我们那边就是
            # 结果 summary 里的「第一峰位置」；作者曲线同样按这个窗口取，
            # 否则比的是"定义"而不是"实现" —— 全区间 argmax 会被几十 Å 外的噪声骗走）
            pk_mine = float(r2.summary[f"{L}-{L} 第一峰位置 (Å)"])
            m = (ref[:, 0] * 10.0 >= 2.2) & (ref[:, 0] * 10.0 <= 25.0) & np.isfinite(ref_g)
            pk_ref = float(ref[:, 0][m][int(np.argmax(ref_g[m]))] * 10.0)
            sig_mine = str(r2.summary.get(f"{L}-{L} 第一峰是否显著", "")).startswith("是")
            sig_ref = float(np.max(ref_g[m])) >= 1.2
            n_leaf = int(r2.summary.get(f"{L}-{L} 原子数", 0))
            cors[L] = (cor, pk_mine, pk_ref, sig_mine == sig_ref, n_leaf)
        res["rdf2d"] = cors
        if skips:
            res["rdf2d_skip"] = skips
    except Exception as exc:                                          # noqa: BLE001
        res["rdf2d_err"] = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------- ③ 密度（PONI）
    try:
        cen, prof, _ = density_profile(mdt, {"PONI": poly}, sel, axis=2, nbins=100,
                                       mode="mass")
        mine = float(np.nanmean(prof["PONI"])) * 1000.0                # g/cm³ → kg/m³
        ref = num(rundir / "analysis_output/PONI_dens/3-run/poni_density.xvg")[:, 1]
        res["density"] = (mine, float(np.mean(ref)))
    except Exception as exc:                                          # noqa: BLE001
        res["density_err"] = f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------- ④ 质心距
    try:
        dz, zs = [], []
        for i in idx:
            u.trajectory[int(i)]
            dz.append(float(poly.center_of_mass()[2] - mem.center_of_mass()[2]) * 0.1)
            zs.append(float(u.trajectory.time))
        ref = num(rundir / "analysis_output/COM/3-run/z_distance.csv")
        t_ref, z_ref = ref[:, 0] * 1e6, ref[:, 1]                  # µs → ps
        pick = [int(np.argmin(np.abs(t_ref - t))) for t in zs]
        res["comdist"] = (float(np.std(dz)), float(np.std(z_ref[pick])))
    except Exception as exc:                                          # noqa: BLE001
        res["comdist_err"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------ ⑤ 水珠子窄带计数
    try:
        wat = select(u, "name W")
        cnt = []
        for i in idx:
            u.trajectory[int(i)]
            bz = float(u.dimensions[2])
            d = np.asarray(wat.positions, float)[:, 2] - float(mem.center_of_mass()[2])
            d -= bz * np.round(d / bz)
            cnt.append(float(np.sum(np.abs(d) <= 5.0)))
        ref = num(rundir / "analysis_output/Water/3-run/Waters.csv")
        t_ref = ref[:, 0] * 1e6
        pick = [int(np.argmin(np.abs(t_ref - t))) for t in zs]
        res["water"] = (float(np.mean(cnt)), float(np.mean(ref[pick, -1])))
    except Exception as exc:                                          # noqa: BLE001
        res["water_err"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------- ⑥ 膜面积与 KA
    try:
        areas = []
        for ts in u.trajectory:                    # 作者从 0 ps 起、不抽帧
            areas.append(float(ts.dimensions[0]) * float(ts.dimensions[1]) / 100.0)
        areas = np.asarray(areas, float)           # Å² → nm²
        msf = float(np.mean(areas ** 2) - np.mean(areas) ** 2)
        ka = KB_MN_M_PER_K * T_K * (float(np.mean(areas)) * 1e-18) / (msf * 1e-36)
        ms = num(rundir / "analysis_output/KA/3-run/msf.dat").reshape(-1)
        ka_ref = KB_MN_M_PER_K * T_K * (float(ms[1]) * 1e-18) / (float(ms[2]) * 1e-36)
        res["ka"] = (ka, ka_ref, float(np.mean(areas)), float(ms[1]))
        # KA 的**公式**自校验：把同一个公式用在作者自己的 area.xvg 上，
        # 应当复现他们的 KA（这样"公式对不对"与"采样密度不同"就分开了）。
        ref_area = num(rundir / "analysis_output/KA/3-run/area.xvg")[:, 1]
        m2 = float(np.mean(ref_area ** 2) - np.mean(ref_area) ** 2)
        ka_formula = (KB_MN_M_PER_K * T_K * (float(np.mean(ref_area)) * 1e-18)
                      / (m2 * 1e-36))
        res["ka_formula"] = (ka_formula, ka_ref)
    except Exception as exc:                                          # noqa: BLE001
        res["ka_err"] = f"{type(exc).__name__}: {exc}"
    return res


def rel(a, b) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


print("=" * 100)
print("跨体系回归：作者原始输出 vs 本工具（全部按作者脚本的口径复现）")
print("=" * 100)
all_sys = systems()
print(f"发现 {len(all_sys)} 个带 analysis_output 的体系，本次跑前 {min(LIMIT, len(all_sys))} 个\n")

t0 = time.time()
for i, sd in enumerate(all_sys[:LIMIT]):
    try:
        r = run_system(sd)
    except Exception as exc:                                          # noqa: BLE001
        print(f"[{i + 1:2}] {sd.name:26} ✗ 打开失败 {type(exc).__name__}: {exc}")
        continue
    if not r:
        continue
    rows.append(r)
    bits = []
    if "rg" in r:
        bits.append(f"Rg {r['rg'][0]:.3f}/{r['rg'][1]:.3f}nm({rel(*r['rg']) * 100:.2f}%)")
    if r.get("rdf2d"):
        worst = min(v[0] for v in r["rdf2d"].values())
        dpk = max(abs(v[1] - v[2]) for v in r["rdf2d"].values())
        bits.append(f"RDF r≥{worst:.4f} 峰位差≤{dpk:.2f}Å({len(r['rdf2d'])} 种脂质)")
    if r.get("rdf2d_skip"):
        bits.append("跳过 " + "、".join(r["rdf2d_skip"]))
    if "density" in r:
        bits.append(f"ρ {r['density'][0]:.1f}/{r['density'][1]:.1f}({rel(*r['density']) * 100:.2f}%)")
    if "comdist" in r:
        bits.append(f"COMσ {r['comdist'][0]:.2f}/{r['comdist'][1]:.2f}Å")
    if "water" in r:
        bits.append(f"nW {r['water'][0]:.2f}/{r['water'][1]:.2f}({rel(*r['water']) * 100:.0f}%)")
    if "ka" in r:
        bits.append(f"KA {r['ka'][0]:.0f}/{r['ka'][1]:.0f}mN/m")
    print(f"[{i + 1:2}] {r['sys']:26} {r['n_frames']:3}帧  " + "  ".join(bits), flush=True)

print("\n" + "=" * 100)
print(f"汇总（{len(rows)} 个体系，用时 {time.time() - t0:.0f}s）")
print("=" * 100)


def stats(key, fn, label, tol):
    vals = [fn(r) for r in rows if key in r]
    if not vals:
        print(f"  {label:22} 无数据")
        return
    v = np.asarray(vals, float)
    ok = int(np.sum(v <= tol))
    print(f"  {label:22} n={v.size:3}  中位偏差 {np.median(v) * 100:7.3f}%  "
          f"最差 {np.max(v) * 100:7.3f}%（{rows[int(np.argmax(v))]['sys']}）  "
          f"达标(≤{tol * 100:g}%) {ok}/{v.size}")


stats("rg", lambda r: rel(*r["rg"]), "① Rg(PONI)", 0.02)
stats("density", lambda r: rel(*r["density"]), "③ PONI 密度均值", 0.05)
stats("water", lambda r: rel(*r["water"]), "⑤ 水珠子计数", 0.25)
stats("ka", lambda r: rel(r["ka"][0], r["ka"][1]),
      "⑥ KA（我们 101 帧 vs 作者 EDR 5001）", 0.35)
stats("ka_formula", lambda r: rel(*r["ka_formula"]),
      "⑥ KA 公式（同一序列自校）", 0.001)

cors = [v[0] for r in rows if r.get("rdf2d") for v in r["rdf2d"].values()]
if cors:
    v = np.asarray(cors, float)
    print(f"  {'② 面内 RDF 相关系数':22} n={v.size:3}  中位 {np.median(v):.4f}  "
          f"最差 {np.min(v):.4f}  达标(≥0.95) {int(np.sum(v >= 0.95))}/{v.size}")
    print("      注：这里只用了 101 帧（作者 Rg.py 生成的 -pbc mol 居中轨迹）。"
          "用作者原始的完整 1001 帧时，同一个体系\n"
          "      （64mer_P1_random1_RBC）相关系数升到 POPC 0.9986 / OSM 0.9987 / "
          "POPE 0.9736 / CHOL 0.9994 ——\n"
          "      即残差主要来自**参考自身的有限采样**（珠子少的脂质如 POPE 更明显），"
          "不是算法（见 _diag_rdf_centering.py）。")
pk = [abs(v[1] - v[2]) for r in rows if r.get("rdf2d") for v in r["rdf2d"].values()]
if pk:
    v = np.asarray(pk, float)
    print(f"  {'② RDF 第一峰位置差 (Å)':22} n={v.size:3}  中位 {np.median(v):.3f}  "
          f"最差 {np.max(v):.3f}  达标(≤0.5 Å) {int(np.sum(v <= 0.5))}/{v.size}"
          "   ← 两边同口径（2.2–25 Å 内最大值）")
sg = [v[3] for r in rows if r.get("rdf2d") for v in r["rdf2d"].values()]
if sg:
    print(f"  {'② RDF 峰显著性判断一致':22} {int(np.sum(sg))}/{len(sg)} 条与按作者曲线"
          "（窗口内 g_max ≥ 1.2）的判定一致")

# 按"每叶层珠子数"拆开看：珠子少时参考本身的涨落就大，峰位不可比
allv = [(v[4], abs(v[1] - v[2]), v[0]) for r in rows if r.get("rdf2d")
        for v in r["rdf2d"].values()]
for label, sel_ in (("珠子充足（每叶层 ≥100）", lambda n: n >= 100),
                    ("珠子偏少（每叶层 <100）", lambda n: n < 100)):
    sub = [(dpk, cor) for n, dpk, cor in allv if sel_(n)]
    if not sub:
        continue
    dpk = np.asarray([s[0] for s in sub], float)
    cor = np.asarray([s[1] for s in sub], float)
    print(f"  {'  ' + label:22} n={dpk.size:3}  峰位差 中位 {np.median(dpk):.2f} / "
          f"最差 {np.max(dpk):.2f} Å（≤0.5 Å {int(np.sum(dpk <= 0.5))}/{dpk.size}）；"
          f"相关系数 中位 {np.median(cor):.4f}")

sig = [r["comdist"][0] / max(r["comdist"][1], 1e-9) for r in rows if "comdist" in r]
if sig:
    v = np.asarray(sig, float)
    print(f"  {'④ 质心距波动比(σ我/σ作)':22} n={v.size:3}  中位 {np.median(v):.3f}  "
          f"范围 [{np.min(v):.3f}, {np.max(v):.3f}]（居中方式不同，只比波动幅度）")

skipped = [(r["sys"], s) for r in rows for s in r.get("rdf2d_skip", [])]
if skipped:
    print(f"\n  ② 面内 RDF 里**按定义跳过**的曲线 {len(skipped)} 条"
          f"（作者参考本身全 NaN，或叶层内珠子 <2）：")
    for s, why in skipped[:12]:
        print(f"    {s:26} {why}")
    if len(skipped) > 12:
        print(f"    …… 其余 {len(skipped) - 12} 条同类")

errs = [(r["sys"], k, v) for r in rows for k, v in r.items() if k.endswith("_err")]
if errs:
    print("\n  异常：")
    for s, k, v in errs[:20]:
        print(f"    {s:26} {k[:-4]}: {v}")
