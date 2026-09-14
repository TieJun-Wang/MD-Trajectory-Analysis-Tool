"""用 Supporting_Info 的作者参考值回归验证本工具（一键跑完，全部与作者原始输出对比）。

覆盖：
  ① Rg      vs GROMACS ``gyrate.xvg``（20mer 全原子参考体系）
  ② R_ee    vs PLUMED ``e2e_prod_AA.txt``（首尾 O1，ATOMS=5,423 → 0-based 4,422）
  ③ 面内 RDF vs 作者 ``Lipid_RDF.py`` 输出（CG 膜体系，上层小叶，2D 口径）
  ④ density vs GROMACS ``gmx density``（同体系，比积分/均值，不比 z 原点）

数据集缺失时整段跳过（不报失败）；用 ``MDTA_SI_DIR`` 可指定根目录。
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# 本文件在 webapp/ 下，仓库根在**上一级**（mdta 包在那里）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdta.analysis.interface import analyze_density, analyze_rdf_inplane  # noqa: E402
from mdta.pipeline import Analyzer                                          # noqa: E402
from mdta.selection import select                                           # noqa: E402

SI = os.environ.get("MDTA_SI_DIR", r"C:\temp\dataset\Supporting_Info")
results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def numeric(path):
    rows = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        t = line.strip()
        if not t or t[0] in "#@!":
            continue
        try:
            rows.append([float(x) for x in t.replace(",", " ").split()])
        except ValueError:
            continue
    return np.array(rows)


# --------------------------------------------------------------- ① ② 20mer 全原子
AA = os.path.join(SI, "20mer", "20mer_BB_alt_T1", "1_AA-reference")
if not os.path.isfile(os.path.join(AA, "gyrate.xvg")):
    print(f"~ 跳过：找不到 {AA}")
else:
    print("=== ① Rg / ② R_ee（20mer 全原子参考体系）===")
    t0 = time.time()
    az = Analyzer(os.path.join(AA, "20mer_solv_prod.tpr"),
                  os.path.join(AA, "20mer_solv_prod_center.xtc"))
    az.auto_setup()

    gyr = numeric(os.path.join(AA, "gyrate.xvg"))
    ref_rg = float(gyr[:, 1].mean()) * 10.0                 # nm → Å
    r = az.run("rg")
    mine_rg = float(r.summary["Rg mean"])
    dev = abs(mine_rg - ref_rg) / ref_rg
    report("① Rg vs GROMACS gyrate", dev < 5e-4,
           f"我 {mine_rg:.4f} Å vs GROMACS {ref_rg:.4f} Å，偏差 {dev*100:.3f}%")

    e2e = numeric(os.path.join(AA, "e2e_prod_AA.txt"))
    ref_ee = float(e2e[:, 1].mean()) * 10.0                 # nm → Å
    r2 = az.run("ree", params={"atom_indices": [4, 422]})   # PLUMED ATOMS=5,423
    mine_ee = float(r2.summary["R_ee mean"])
    dev2 = abs(mine_ee - ref_ee) / ref_ee
    report("② R_ee vs PLUMED e2e", dev2 < 1e-3,
           f"我 {mine_ee:.4f} Å vs PLUMED {ref_ee:.4f} Å，偏差 {dev2*100:.4f}%"
           f"（链端 {r2.summary.get('链端原子')}）")
    print(f"  （20mer 两项合计 {time.time()-t0:.1f}s，{az.info.n_frames} 帧）", flush=True)

# ------------------------------------------------- ③ 面内 RDF / ④ 密度（CG 膜体系）
CG = os.path.join(SI, "64mer", "64mer_P1_random1_RBC", "6_CG-takeCURRENT")
if not os.path.isfile(os.path.join(CG, "3-run.tpr")):
    print(f"~ 跳过：找不到 {CG}")
else:
    print("\n=== ③ 面内 RDF (2D) / ④ density（CG 膜体系 64mer_P1_random1_RBC）===")
    az = Analyzer(os.path.join(CG, "3-run.tpr"), os.path.join(CG, "3-run_center.xtc"))
    az.auto_setup()
    u = az.universe
    sel = az.require_frames()
    if sel.n_frames > 60:                       # 逐对 O(N²)：抽样够用即可
        sel = sel[:: max(1, sel.n_frames // 60)]

    HEAD = {"CHOL": "ROH", "OSM": "PO4"}         # 作者 headgroup_dict（RBC 分支）
    lipids = "POPC OSM POPE POPS CHOL"
    zmid = 0.0
    for i in sel.indices:
        u.trajectory[int(i)]
        zmid += float(select(u, "resname " + lipids).center_of_mass()[2])
    zmid /= sel.n_frames
    Lz = float(np.asarray(u.dimensions[2], float))

    groups = {k: select(u, f"resname {k} and name {v}") for k, v in HEAD.items()}
    t1 = time.time()
    r2d = analyze_rdf_inplane(az.trajectory, groups, sel, plane_axis=2, rmax=70.0,
                              nbins=350, slab=(zmid, Lz))
    print(f"  （rdf2d {time.time()-t1:.1f}s，{sel.n_frames} 帧，上层阈值 z>{zmid:.1f} Å）",
          flush=True)
    for name in HEAD:
        p = os.path.join(CG, "analysis_output", "Lipid_RDF", "3-run",
                         f"{name}_{name}_rdf.txt")
        if not os.path.isfile(p):
            report(f"③ {name}-{name} RDF", False, f"找不到参考 {p}")
            continue
        ref = numeric(p)
        cur = [c for c in r2d.curves if c.label.startswith(name)]
        my_g = np.asarray(cur[0].y, float)
        ref_g = ref[:, 1]
        n = min(len(my_g), len(ref_g))
        corr = float(np.corrcoef(my_g[:n], ref_g[:n])[0, 1])
        pk_mine = float(ref[0, 0] * 0 + np.asarray(cur[0].x, float)[int(np.argmax(my_g[:n]))])
        pk_ref = float(ref[int(np.argmax(ref_g[:n])), 0] * 10.0)
        report(f"③ {name}-{name} RDF（2D 面内）", corr > 0.99 and abs(pk_mine - pk_ref) < 0.3,
               f"相关系数 {corr:.4f}；第一峰 我 {pk_mine:.2f} Å vs 作者 {pk_ref:.2f} Å")

    poni = select(u, "resname P1c P1t P1e")
    rden = analyze_density(az.trajectory, {"PONI": poni}, sel, axis=2, nbins=100,
                           mode="mass")
    my_mean = float(np.nanmean(np.asarray(rden.curves[0].y, float))) * 1000.0
    pref = os.path.join(CG, "analysis_output", "PONI_dens", "3-run", "poni_density.xvg")
    ref_den = numeric(pref)
    ref_mean = float(ref_den[:, 1].mean())
    dev3 = abs(my_mean - ref_mean) / ref_mean
    # 面积质量密度独立校验：PONI 总质量 / xy 面积
    mass_mg = float(np.asarray(poni.masses, float).sum()) / 6.02214076e23 * 1e-3
    area_m2 = float(u.dimensions[0]) * float(u.dimensions[1]) * 1e-20
    area_ref = float(np.sum(ref_den[:, 1]) * np.mean(np.diff(ref_den[:, 0]))
                     * 1e-9 * 1e6)
    report("④ density vs gmx density", dev3 < 0.05,
           f"剖面均值 我 {my_mean:.2f} vs GROMACS {ref_mean:.2f} kg/m³"
           f"（差 {dev3*100:.2f}%）；∫ρdz 作者 {area_ref:.4f} vs "
           f"独立校验 {mass_mg/area_m2*1e6:.4f} mg/m²")
    print("  注：作者剖面 z 原点与本文不同（未居中轨迹 / 相对膜质心），"
          "故只比均值与积分，不比峰位。", flush=True)

    # ---- ⑤ 两组分质心距（作者 COM.py）与 ⑥ 水珠子窄带计数（作者 Water.py）----
    print("\n=== ⑤ 质心距 / ⑥ 水珠子计数（与作者 COM.py / Water.py 同口径）===")
    from mdta.analysis.interface import analyze_com_distance
    from mdta.selection import select as _sel

    mem = _sel(u, "resname POPC OSM POPE POPS CHOL")
    poly = _sel(u, "resname P1c P1t P1e")
    water = _sel(u, "name W")
    if mem.n_atoms and poly.n_atoms:
        rc = analyze_com_distance(az.trajectory, {"polymer": poly, "membrane": mem},
                                  sel, pair=("polymer", "membrane"), axis=2, pbc=False)
        mine_z = float(rc.summary["沿 z 轴 平均 (Å)"])
        mine_sd = float(rc.summary["沿 z 轴 标准差 (Å)"])
        pcom = os.path.join(CG, "analysis_output", "COM", "3-run", "z_distance.csv")
        ref_z = ref_sd = None
        if os.path.isfile(pcom):
            rcom = numeric(pcom)
            ref_z = float(rcom[:, 1].mean()) * 10.0
            ref_sd = float(rcom[:, 1].std()) * 10.0
        ok = (ref_z is not None and abs(mine_z - ref_z) < 10.0
              and 0.5 < mine_sd / max(ref_sd, 1e-9) < 2.0)
        report("⑤ 两组分质心距 vs COM.py", ok,
               f"我 {mine_z:.2f}±{mine_sd:.2f} Å vs 作者 {ref_z:.2f}±{ref_sd:.2f} Å"
               f"（作者用未居中轨迹、本文用居中轨迹，故只比量级与波动幅度）")

    if water.n_atoms and mem.n_atoms:
        # 复现作者算法：逐帧膜质心 + z 最小镜像，统计 ±5 Å 内的水珠子数
        cnt = []
        for i in sel.indices:
            u.trajectory[int(i)]
            bz = float(u.dimensions[2])
            zm = float(mem.center_of_mass()[2])
            dz = np.asarray(water.positions, float)[:, 2] - zm
            dz -= bz * np.round(dz / bz)
            cnt.append(float(np.sum(np.abs(dz) <= 5.0)))
        mine_cnt = float(np.mean(cnt))
        pwat = os.path.join(CG, "analysis_output", "Water", "3-run", "Waters.csv")
        ref_cnt = None
        if os.path.isfile(pwat):
            rwat = numeric(pwat)
            ref_cnt = float(rwat[:, -1].mean())
        ok = ref_cnt is not None and abs(mine_cnt - ref_cnt) / max(ref_cnt, 1e-9) < 0.5
        report("⑥ 水珠子 ±5 Å 计数 vs Water.py", ok,
               f"我 {mine_cnt:.3f} vs 作者 {ref_cnt:.3f}"
               f"（逐帧动态膜质心；用时间平均质心会得到 7.76，差 15×）")

# 关于 ③「二面角 vs 5_target-distr」的结论（已核查，不再重复尝试）：
#   - `5_target-distr/{bonds,angles,dihedrals}_mapped/*.xvg` 由
#     `gmx distance -f ../3_mapped/mapped.xtc -n dihedrals.ndx -s CG.tpr` 生成，
#     **`gmx distance` 算的是距离而不是二面角**；且 `data_dihedrals.txt` 里解析出的
#     "均值/标准差" 全部 ≈ ±178°（标准差为负、数值也不合理），不能当二面角参考。
#   - 实测对比：本工具在 CG.tpr + mapped.xtc（99 珠子 / 501 帧）上用
#     `compute_dihedral_series` 算同一批 ndx 四元组，逐项均值落在 −74…+41°，
#     与上述文件完全不同 → **该文件不构成二面角参考**。
#   - 二面角实现本身的独立验证在 `selftest.py`：φ/ψ 对 MDAnalysis 原生
#     `Dihedral` 分析偏差 **0.00e+00**，另有 trans/gauche 分类与阈值共享的用例。

print("\n" + "=" * 74)
n_ok = sum(1 for _, ok, _ in results if ok)
print(f"支持信息回归：{n_ok}/{len(results)} 项通过")
for name, ok, _ in results:
    if not ok:
        print(f"   未通过: {name}")
print("=" * 74)
sys.exit(0 if n_ok == len(results) and results else 1)
