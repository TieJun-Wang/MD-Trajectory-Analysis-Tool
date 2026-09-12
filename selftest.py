# -*- coding: utf-8 -*-
"""MD 轨迹分析工具 —— 数值正确性自检。

本文件用两种方式验证工具的计算结果：

1. **与 MDAnalysis 原生实现对照** —— Rg、RDF、二面角等；
2. **与独立公式/文献值对照** —— 成对距离公式、液态水密度、水的 RDF 峰位与
   配位数、水的扩散系数、各向同性体系的取向参数等。

直接运行::

    python selftest.py                    # 用默认的 adk_oplsaa 轨迹
    python selftest.py sys.tpr sys.xtc    # 指定轨迹

也可以用 pytest 运行（文件中所有 ``test_*`` 函数都会被收集）。
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import MDAnalysis as mda                                    # noqa: E402
from MDAnalysis.analysis.dihedrals import Dihedral          # noqa: E402
from MDAnalysis.analysis.rdf import InterRDF                # noqa: E402

try:  # MDAnalysis < 2.8 提供 Phi / Psi；2.8+ 统一为 Dihedral
    from MDAnalysis.analysis.dihedrals import Phi, Psi      # type: ignore
except ImportError:  # pragma: no cover - 取决于 MDAnalysis 版本
    Phi = Psi = None


def _four_atom_groups(residues, kind: str):
    """收集主链 φ 或 ψ 的四原子组列表（兼容新旧 MDAnalysis）。"""
    groups = []
    for res in residues:
        sel = res.phi_selection() if kind == "phi" else res.psi_selection()
        if sel is not None and sel.n_atoms == 4:
            groups.append(sel)
    return groups


def _run_native_phi_psi(atomgroup, kind: str):
    """用 MDAnalysis 自带的实现计算 φ 或 ψ 的角度矩阵。"""
    res = atomgroup.residues
    groups = _four_atom_groups(res, kind)
    if Phi is not None:
        cls = Phi if kind == "phi" else Psi
        return np.asarray(cls(atomgroup).run().results.angles, dtype=float)
    return np.asarray(Dihedral(groups).run().results.angles, dtype=float)

from mdta.analysis import conformation as conf              # noqa: E402
from mdta.analysis import dynamics as dyn                   # noqa: E402
from mdta.analysis import interface as ifc                  # noqa: E402
from mdta.analysis.base import positions_for                # noqa: E402
from mdta.io import MDTrajectory, load_trajectory           # noqa: E402
from mdta.preprocess import select_frames                   # noqa: E402
from mdta.selection import classify_residues, select        # noqa: E402

TOP_DEFAULT = "adk_oplsaa.tpr"
XTRAJ_DEFAULT = "adk_oplsaa.xtc"

# 缓存的全局会话（避免重复读取大文件）
_CACHE: dict = {}
#: 由 main() 从命令行设置的 (拓扑, 轨迹)
_CLI_ARGS: tuple = (None, None)


def _load(top: str | None = None, xtc: str | None = None):
    top = top or os.environ.get("MDTA_TEST_TOP", TOP_DEFAULT)
    xtc = xtc or os.environ.get("MDTA_TEST_XTC", XTRAJ_DEFAULT)
    key = (os.path.abspath(top), os.path.abspath(xtc) if xtc else None)
    if key not in _CACHE:
        mdt = load_trajectory(top, xtc)
        u = mdt.universe
        data = {
            "mdt": mdt,
            "times": mdt.times_ps,
            "sel": select_frames(mdt.times_ps),
            "protein": select(u, category="protein"),
            "water": select(u, category="water"),
            "ion": select(u, category="ion"),
        }
        _CACHE[key] = data
    return _CACHE[key]


def _ctx():
    return _load(*_CLI_ARGS)


# ==================================================================== 测试
def test_rg_implementation_matches_mdanalysis_and_independent_formula():
    """Rg：与 MDAnalysis 实现、独立成对距离公式三者完全一致。"""
    d = _ctx()
    mdt, u = d["mdt"], d["mdt"].universe
    prot = d["protein"]
    u.trajectory[0]
    pos = np.asarray(prot.positions, dtype=float)
    m = np.asarray(prot.masses, dtype=float)

    mine = conf.compute_rg(pos, m)
    mda_rg = float(prot.radius_of_gyration())

    # 独立公式：Rg² = (1/(2M²)) ΣΣ m_i m_j r_ij²
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    pair = float(np.sqrt((np.outer(m, m) * d2).sum() / (2.0 * m.sum() ** 2)))

    assert abs(mine - mda_rg) < 1e-9, (mine, mda_rg)
    assert abs(mine - pair) < 1e-9, (mine, pair)
    return f"Rg(裸坐标) 本工具={mine:.6f} MDAnalysis={mda_rg:.6f} 独立公式={pair:.6f} Å"


def test_pbc_unwrap_changes_rg_and_is_correct():
    """周期性边界条件：验证蛋白确实被盒边界切断，且 unwrap 后结果与文献相符。"""
    d = _ctx()
    mdt = d["mdt"]
    u = mdt.universe
    prot = d["protein"]
    u.trajectory[0]
    raw = np.asarray(prot.positions, dtype=float)
    rg_raw = conf.compute_rg(raw, np.asarray(prot.masses, dtype=float))

    unw = positions_for(prot, unwrap=True)
    rg_unw = conf.compute_rg(unw, np.asarray(prot.masses, dtype=float))

    dmax_raw = float(np.sqrt(((raw[:, None] - raw[None, :]) ** 2).sum(-1).max()))
    dmax_unw = float(np.sqrt(((unw[:, None] - unw[None, :]) ** 2).sum(-1).max()))

    assert dmax_raw > dmax_unw * 1.2, (dmax_raw, dmax_unw)  # 裸坐标被拉长 => 被切断
    # 214 残基球蛋白的 Rg 经验值约 2.2·N^0.38 ≈ 17 Å，文献 AdK 约 19–20 Å
    assert 17.0 < rg_unw < 21.5, rg_unw
    return (f"裸坐标 Rg={rg_raw:.2f} Å（最大原子间距 {dmax_raw:.1f} Å）→ "
            f"unwrap 后 Rg={rg_unw:.2f} Å（{dmax_unw:.1f} Å），与 AdK 文献值一致")


def test_dihedral_matches_independent_atan2_formula():
    """二面角：与独立的解析公式在随机几何上逐点一致。"""
    rng = np.random.default_rng(20240501)
    n = 400
    pts = rng.normal(size=(n, 4, 3)) * 2.0

    def dihedral_ref(p0, p1, p2, p3):
        """独立的二面角实现（投影到垂直于中心键的平面上后取 atan2）。"""
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
        v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
        w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
        x = (v * w).sum(-1)
        y = (np.cross(b1, v) * w).sum(-1)
        return np.degrees(np.arctan2(y, x))

    ref = dihedral_ref(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])

    # 通过工具自身的计算路径（合成一个 4 原子体系 + 单帧轨迹）
    u = mda.Universe.empty(4, n_residues=1, atom_resindex=[0] * 4, trajectory=True)
    u.add_TopologyAttr("bonds", [(0, 1), (1, 2), (2, 3)])
    u.add_TopologyAttr("elements", ["C"] * 4)
    u.add_TopologyAttr("masses", [12.0] * 4)
    u.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    idx = np.array([[0, 1, 2, 3]])

    worst = 0.0
    for k in range(n):
        u.atoms.positions = pts[k]
        got = conf.compute_dihedral_series(mdt, idx, sel)[0, 0]
        diff = abs((got - ref[k] + 180.0) % 360.0 - 180.0)
        worst = max(worst, diff)
    # MDAnalysis 以 float32 存坐标，角度层面的舍入约为 1e-4 度量级
    assert worst < 1e-3, worst
    return (f"400 组随机几何上最大偏差 {worst:.3e}°"
            f"（与解析公式一致，残差为 float32 坐标精度）")


def test_phi_psi_matches_mdanalysis_native_analysis():
    """蛋白质 φ/ψ：与 MDAnalysis 的 Phi/Psi 分析类结果一致。

    注意：MDAnalysis 的 Phi/Psi 直接用轨迹里的坐标，不做 PBC 展开；
    本体系蛋白被盒边界切断，因此这里先把蛋白逐帧展开成连续坐标，
    再在两个实现之间比较（这样比较的才是同一个物理量）。
    """
    d = _ctx()
    mdt = d["mdt"]
    u = mdt.universe
    prot = d["protein"]
    sel = d["sel"]

    # 1) 逐帧展开蛋白，构造"连续坐标"的蛋白 Universe
    u2 = mda.Universe(mdt.topology, mdt.trajectory)
    prot2 = u2.select_atoms("protein")
    coords = np.empty((len(u2.trajectory), prot2.n_atoms, 3), dtype=np.float32)
    for i, _ts in enumerate(u2.trajectory):
        prot2.unwrap(compound="fragments", reference=None)
        coords[i] = np.asarray(prot2.positions, dtype=np.float32)

    pu = mda.Universe.empty(
        prot2.n_atoms, n_residues=prot2.n_residues,
        atom_resindex=np.asarray(prot2.atoms.resindices),
        residue_segindex=np.zeros(prot2.n_residues, dtype=int), trajectory=True)
    pu.add_TopologyAttr("names", np.asarray(prot2.names))
    pu.add_TopologyAttr("masses", np.asarray(prot2.masses))
    pu.add_TopologyAttr("resnames", np.asarray(prot2.residues.resnames))
    pu.add_TopologyAttr("resids", np.asarray(prot2.residues.resids))
    pu.add_TopologyAttr("segids", ["PROT"])
    pu.add_TopologyAttr("bonds", np.asarray(prot2.bonds.indices))
    pu.trajectory = mda.coordinates.memory.MemoryReader(coords, order="fac")

    # 2) 两个实现
    phi_idx, psi_idx = [], []
    for r in prot.residues:
        a = r.phi_selection()
        if a is not None and a.n_atoms == 4:
            phi_idx.append([int(x.index) for x in a])
        b = r.psi_selection()
        if b is not None and b.n_atoms == 4:
            psi_idx.append([int(x.index) for x in b])
    phi_idx = np.asarray(phi_idx, dtype=int)
    psi_idx = np.asarray(psi_idx, dtype=int)
    my_phi = conf.compute_dihedral_series(mdt, phi_idx, sel, unwrap_group=prot)
    my_psi = conf.compute_dihedral_series(mdt, psi_idx, sel, unwrap_group=prot)

    ref_phi = _run_native_phi_psi(pu, "phi")
    ref_psi = _run_native_phi_psi(pu, "psi")
    assert my_phi.shape == ref_phi.shape, (my_phi.shape, ref_phi.shape)
    assert my_psi.shape == ref_psi.shape, (my_psi.shape, ref_psi.shape)

    dp = float(np.abs((my_phi - ref_phi + 180.0) % 360.0 - 180.0).max())
    ds = float(np.abs((my_psi - ref_psi + 180.0) % 360.0 - 180.0).max())
    assert dp < 1e-3 and ds < 1e-3, (dp, ds)

    # 3) 记录 PBC 的影响（用未展开坐标跑 MDAnalysis 的实现）
    ref_phi_raw = _run_native_phi_psi(prot, "phi")
    dr = float(np.abs((ref_phi_raw - ref_phi + 180.0) % 360.0 - 180.0).max())
    return (f"φ 最大偏差 {dp:.2e}°，ψ 最大偏差 {ds:.2e}°"
            f"（{my_phi.shape[0]} 帧 × {my_phi.shape[1]} 个角）；"
            f"本轨迹的主链二面角恰好未被盒边界切断（裸坐标偏差 {dr:.1f}°），"
            f"但 Rg 等整体几何量受影响很大，故仍统一做 PBC 展开")


def test_rdf_matches_mdanalysis_interrdf():
    """RDF：与 MDAnalysis 的 InterRDF 结果一致（容差来自 bin 化差异）。"""
    d = _ctx()
    mdt = d["mdt"]
    prot, water = d["protein"], d["water"]
    sel = select_frames(mdt.times_ps, max_frames=3)
    ow = water.select_atoms("name OW")

    rmax, nbins = 8.0, 80
    res = ifc.analyze_rdf(mdt, {"OW": ow}, sel, rmax=rmax, nbins=nbins)
    mine = res.curves_of(0)[0].y

    ir = InterRDF(ow, ow, nbins=nbins, range=(0.0, rmax), exclusion_block=(1, 1))
    ir.run(start=0, stop=3, step=1)
    ref = ir.results.rdf
    r_ref = ir.results.bins
    r_mine = res.curves_of(0)[0].x
    assert np.allclose(r_ref, r_mine, atol=1e-9), (r_ref[:3], r_mine[:3])

    # 允许 bin 边界与实现细节带来的少量差异
    rel = np.abs(mine - ref) / np.maximum(ref, 1.0)
    worst = float(rel.max())
    assert worst < 0.05, (worst, np.argmax(rel), mine[np.argmax(rel)], ref[np.argmax(rel)])
    return (f"OW-OW g(r) 与 InterRDF 最大相对偏差 {worst * 100:.2f}%；"
            f"峰值 g={mine.max():.3f}")


def test_water_density_is_physical():
    """密度分布：纯水组分的体相质量密度应接近 1 g/cm³，且能回到总质量。"""
    d = _ctx()
    mdt = d["mdt"]
    water = d["water"]
    res = ifc.analyze_density(mdt, {"water": water}, d["sel"], axis=2, nbins=100)
    bulk = res.summary["water 体相密度"]
    mean = res.summary["water 平均密度"]
    assert 0.90 < bulk < 1.10, bulk
    assert 0.90 < mean < 1.10, mean

    # 内部一致性：平均密度 × 平均盒体积 应精确等于该组分总质量
    vbox = res.summary["平均盒体积 (Å³)"]
    mass_from_profile = mean * vbox / 1.66053906660
    mass_direct = float(np.asarray(water.masses, dtype=float).sum())
    rel = abs(mass_from_profile - mass_direct) / mass_direct
    assert rel < 1e-9, (mass_from_profile, mass_direct, rel)
    return (f"体相密度 {bulk:.4f} g/cm³，平均 {mean:.4f} g/cm³；"
            f"密度积分精确回到总质量（相对偏差 {rel:.1e}）")


def test_water_rdf_physics():
    """水的 OW–OW RDF：峰位、峰高、第一壳层配位数与文献一致。"""
    d = _ctx()
    mdt = d["mdt"]
    ow = d["water"].select_atoms("name OW")
    if ow.n_atoms == 0:
        return "跳过：体系中找不到水的 OW 原子"
    sel = select_frames(mdt.times_ps, max_frames=int(
        os.environ.get("MDTA_RDF_FRAMES", "4")))
    res = ifc.analyze_rdf(mdt, {"OW": ow}, sel, rmax=9.0, nbins=90)
    r0 = res.summary["OW-OW 第一峰位置 (Å)"]
    g0 = res.summary["OW-OW 第一峰高度 g_max"]
    cn = res.summary["OW-OW 配位数 (第一壳层)"]
    assert 2.55 <= r0 <= 2.95, r0
    assert 2.4 <= g0 <= 3.8, g0
    assert 3.6 <= cn <= 5.6, cn
    return f"第一峰 {r0:.2f} Å, g_max={g0:.2f}, 配位数 {cn:.2f}（文献 ≈2.8 Å / 2.8–3.2 / 4–5）"


def test_msd_water_diffusion_coefficient():
    """MSD：水的扩散系数与 MSD∝t 的指数应与文献一致。"""
    d = _ctx()
    mdt = d["mdt"]
    ow = d["water"].select_atoms("name OW")
    if ow.n_atoms == 0 or len(mdt.times_ps) < 4:
        return "跳过：体系或帧数不足"
    sel = select_frames(mdt.times_ps, max_frames=6)
    res = dyn.analyze_msd(mdt, {"OW": ow}, sel)
    D = res.summary["OW D (m²/s)"]
    alpha = res.summary["OW α (log-log 斜率)"]
    assert 1.0e-9 < D < 8.0e-9, D
    assert 0.85 < alpha < 1.15, alpha
    return (f"D = {D:.3e} m²/s（文献水 ≈ 2.5–4.5×10⁻⁹），"
            f"MSD∝t^α 中 α = {alpha:.3f}")


def test_orientation_isotropic_system():
    """取向分析：溶液中的球蛋白是各向同性的，S 与 P₂ 都应接近 0。"""
    from mdta.analysis import crystallinity as cry

    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]

    r = cry.analyze_orientation(mdt, prot, d["sel"], mode="backbone", stride=3)
    S = r.summary["S mean"]
    P2 = r.summary["P2 mean"]
    cos2 = r.summary["实测 <cos²θ>"]
    assert abs(S) < 0.10, S
    assert abs(P2) < 0.10, P2
    assert abs(cos2 - 1.0 / 3.0) < 0.02, cos2
    return (f"S = {S:.4f}，P₂ = {P2:.4f}（各向同性理论值 0）；"
            f"<cos²θ> = {cos2:.4f}（各向同性理论值 1/3）")


def test_nematic_order_parameter_on_known_distributions():
    """取向参数 S：在构造的已知分布上给出理论值。"""
    from mdta.analysis.crystallinity import nematic_order_parameter

    z = np.array([0.0, 0.0, 1.0])
    # 完全取向 -> S = 1
    v = np.tile(z, (500, 1))
    assert abs(nematic_order_parameter(v)["S"] - 1.0) < 1e-9
    # 完全各向同性 -> S -> 0
    rng = np.random.default_rng(7)
    iso = rng.normal(size=(200000, 3))
    s_iso = nematic_order_parameter(iso)["S"]
    assert abs(s_iso) < 0.01, s_iso
    # 圆锥内均匀分布（半角 θ0）-> S = 0.5·cosθ0·(1+cosθ0)
    theta0 = np.radians(30.0)
    phi = rng.uniform(0, 2 * np.pi, 60000)
    ct = rng.uniform(np.cos(theta0), 1.0, 60000)
    st = np.sqrt(1 - ct ** 2)
    cone = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
    expect = 0.5 * np.cos(theta0) * (1 + np.cos(theta0))
    got = nematic_order_parameter(cone)["S"]
    assert abs(got - expect) < 0.01, (got, expect)
    return (f"完全取向 S={1.0:.4f}；各向同性 S={s_iso:.5f}；"
            f"30° 圆锥 S={got:.4f}（理论 {expect:.4f}）")


def _build_slab_universe(sigma: float, n_atoms: int = 60000, box: float = 40.0,
                         n_bins: int = 200):
    """构造一个"两组分互补界面"的合成体系。

    沿 z 方向的归一化密度为

        f_A(z) = ½(1 − erf((z − 20)/(√2σ)))，   f_B(z) = 1 − f_A(z)

    即界面位于 z = 20 Å，其 10–90 宽度为 ``2.5631σ``。
    每个 bin 内的原子按期望数目**等间距**摆放，因此直方图与理论轮廓一致，
    没有抽样噪声。
    """
    from scipy.special import erf

    edges = np.linspace(0.0, box, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dz = box / n_bins
    fa = 0.5 * (1.0 - erf((centers - box / 2.0) / (np.sqrt(2.0) * sigma)))
    fa /= fa.sum()                       # 归一化：Σ fa = 1，均值 = 1/n_bins
    uniform = 1.0 / n_bins
    # 补集：f_B 的均值同样是 1/n_bins，故归一化后为 (2/n_bins − fa)
    fb = 2.0 * uniform - fa
    ca = np.round(fa * n_atoms).astype(int)
    cb = np.maximum(np.round(fb * n_atoms).astype(int), 0)

    def place(counts):
        zs = []
        for c, (lo, hi) in zip(counts, zip(edges[:-1], edges[1:])):
            if c <= 0:
                continue
            zs.append(np.linspace(lo + dz / (2 * c), hi - dz / (2 * c), int(c)))
        return np.concatenate(zs)

    za, zb = place(ca), place(cb)
    n_a, n_b = za.size, zb.size
    u = mda.Universe.empty(n_a + n_b, n_residues=2,
                           atom_resindex=[0] * n_a + [1] * n_b, trajectory=True)
    u.add_TopologyAttr("resnames", ["A", "B"])
    u.add_TopologyAttr("names", ["X"] * (n_a + n_b))
    u.add_TopologyAttr("masses", [12.0] * (n_a + n_b))
    u.add_TopologyAttr("elements", ["C"] * (n_a + n_b))
    u.dimensions = np.array([box, box, box, 90.0, 90.0, 90.0], dtype=np.float32)
    rng = np.random.default_rng(3)
    xy = rng.uniform(0.0, box, size=(n_a + n_b, 2))
    u.atoms.positions = np.column_stack([xy[:, 0], xy[:, 1], np.concatenate([za, zb])])
    return u, u.atoms[:n_a], u.atoms[n_a:]


def test_interface_width_on_synthetic_erf_profile():
    """界面宽度：在已知 erf 轮廓的合成体系上反解出正确的界面位置与宽度。"""
    # 注意 x0 在 20 Å，σ=1.5 Å  => 10–90 宽度 = 2.5631σ = 3.845 Å
    u, ag_a, ag_b = _build_slab_universe(sigma=1.5)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    res = ifc.analyze_interface_width(mdt, {"A": ag_a, "B": ag_b}, sel,
                                      pair=("A", "B"), axis=2, nbins=200, smooth=1)
    x0 = res.summary["界面位置 (Å)"]
    w = res.summary["界面宽度 10-90 (Å)"]
    w_fit = res.summary.get("界面宽度 10-90 (erf 拟合) (Å)", float("nan"))
    assert abs(x0 - 20.0) < 0.15, x0
    assert abs(w - 3.845) < 0.25, w
    assert abs(w_fit - 3.845) < 0.15, w_fit
    return (f"设定界面位置 20.00 Å、宽度 3.845 Å → 实测 {x0:.3f} Å / {w:.3f} Å；"
            f"erf 拟合 {w_fit:.3f} Å")


def test_interface_width_sharp_step():
    """界面宽度：尖锐台阶界面（σ→0）应给出接近 0 的界面宽度。"""
    u, ag_a, ag_b = _build_slab_universe(sigma=0.05)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    res = ifc.analyze_interface_width(mdt, {"A": ag_a, "B": ag_b}, sel,
                                      pair=("A", "B"), axis=2, nbins=200, smooth=1)
    x0 = res.summary["界面位置 (Å)"]
    w = res.summary["界面宽度 10-90 (Å)"]
    assert abs(x0 - 20.0) < 0.2, x0
    assert w < 0.5, w
    return f"尖锐界面：位置 {x0:.3f} Å（设定 20），宽度 {w:.3f} Å（应 ≈0）"


def _build_two_interface_universe(sigma: float, n_atoms: int = 80000,
                                  box: float = 40.0, n_bins: int = 200,
                                  z1: float = 12.0, z2: float = 28.0):
    """构造"两界面板层"合成体系：A 在 z₁..z₂ 之间，B 在两侧（周期闭合）。

    归一化密度为两个台阶之积::

        f_A(z) = ½(1+erf((z−z₁)/(√2σ))) · ½(1−erf((z−z₂)/(√2σ)))

    因此盒内出现**两个**界面，位置 z₁、z₂，各自的 10–90 宽度都是 ``2.5631σ``。
    回归用途：界面宽度必须在**局部**测量，不能跨到另一个界面上去
    （历史 bug：曾把宽度报成两个界面之间的距离 ≈ 19.8 Å）。
    """
    from scipy.special import erf

    edges = np.linspace(0.0, box, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dz = box / n_bins
    step1 = 0.5 * (1.0 + erf((centers - z1) / (np.sqrt(2.0) * sigma)))
    step2 = 0.5 * (1.0 - erf((centers - z2) / (np.sqrt(2.0) * sigma)))
    fa = step1 * step2                      # 0..1，平台值为 1
    fb = 1.0 - fa                           # 严格互补，平台值同为 1
    # 两个组分在各自体相区用**相同**的每 bin 原子数，这样两者的体相值相等，
    # 归一化后 f_A、f_B 恰好就是 fa、fb（不能用 2/nbins − fa 的写法：板层两侧
    # 占比不等时它会把体相算成负数并被截断，从而扭曲过渡区）。
    per_bin = float(n_atoms) / float(n_bins)
    ca = np.round(fa * per_bin).astype(int)
    cb = np.round(fb * per_bin).astype(int)

    def place(counts):
        zs = []
        for c, (lo, hi) in zip(counts, zip(edges[:-1], edges[1:])):
            if c <= 0:
                continue
            zs.append(np.linspace(lo + dz / (2 * c), hi - dz / (2 * c), int(c)))
        return np.concatenate(zs)

    za, zb = place(ca), place(cb)
    n_a, n_b = za.size, zb.size
    u = mda.Universe.empty(n_a + n_b, n_residues=2,
                           atom_resindex=[0] * n_a + [1] * n_b, trajectory=True)
    u.add_TopologyAttr("resnames", ["A", "B"])
    u.add_TopologyAttr("names", ["X"] * (n_a + n_b))
    u.add_TopologyAttr("masses", [12.0] * (n_a + n_b))
    u.add_TopologyAttr("elements", ["C"] * (n_a + n_b))
    u.dimensions = np.array([box, box, box, 90.0, 90.0, 90.0], dtype=np.float32)
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.0, box, size=(n_a + n_b, 2))
    u.atoms.positions = np.column_stack([xy[:, 0], xy[:, 1], np.concatenate([za, zb])])
    return u, u.atoms[:n_a], u.atoms[n_a:]


def test_interface_width_two_interfaces_are_measured_locally():
    """界面宽度：板层体系有两个界面时，各自宽度必须局部测量、互不串扰。"""
    sigma = 1.5
    expect = 2.5631 * sigma                 # 3.845 Å
    u, ag_a, ag_b = _build_two_interface_universe(sigma=sigma)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    res = ifc.analyze_interface_width(mdt, {"A": ag_a, "B": ag_b}, sel,
                                      pair=("A", "B"), axis=2, nbins=200, smooth=1)
    s = res.summary
    n_cross = s["检测到的界面交点数"]
    x0 = s["界面位置 (Å)"]
    w0 = s["界面宽度 10-90 (Å)"]
    x1 = s["界面2 界面位置 (Å)"]
    w1 = s["界面2 界面宽度 10-90 (Å)"]
    assert n_cross == 2, n_cross
    assert abs(x0 - 12.0) < 0.3, x0
    assert abs(x1 - 28.0) < 0.3, x1
    # 关键断言：宽度必须是局部过渡宽度，而不是 12 Å 与 28 Å 之间的距离
    for w in (w0, w1):
        assert abs(w - expect) < 0.4, (w, expect, "界面宽度跨到了另一个界面")
    # 两个界面的 erf 拟合也应各自成立
    for key in ("erf 拟合 R²", "界面2 erf 拟合 R²"):
        assert s.get(key, 0.0) > 0.99, (key, s.get(key))
    return (f"双界面板层（设定 z=12/28 Å、宽度 {expect:.3f} Å）→ "
            f"实测 {x0:.2f}/{w0:.3f} Å 与 {x1:.2f}/{w1:.3f} Å，"
            f"未退化成一个跨双界面的宽度")


def test_end_to_end_distance_manual():
    """端到端距离：与手工计算的首尾原子间距一致。"""
    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]
    sel = d["sel"]
    res = conf.analyze_end_to_end(mdt, prot, sel)
    u = mdt.universe
    manual = np.empty(len(sel.indices))
    for k, i in enumerate(sel.indices):
        u.trajectory[int(i)]
        p = positions_for(prot, unwrap=True)
        manual[k] = np.linalg.norm(p[-1] - p[0])
    mine = res.curves_of(0)[0].y
    worst = float(np.abs(mine - manual).max())
    assert worst < 1e-9, worst
    return f"逐帧最大偏差 {worst:.2e} Å，平均值 {mine.mean():.3f} Å"


def test_frame_selection_semantics():
    """帧选择：起止时间、抽帧间隔、平衡段的语义正确。"""
    times = np.arange(0.0, 1001.0, 10.0)          # 0,10,...,1000 ps
    s = select_frames(times, start_ps=100, stop_ps=300, interval_ps=50)
    assert list(s.times_ps) == [100, 150, 200, 250, 300], list(s.times_ps)

    s2 = select_frames(times, equil_ps=200)
    assert s2.times_ps[0] == 200, s2.times_ps[0]

    s3 = select_frames(times, interval_ps=1000)
    assert s3.n_frames >= 1

    # float32 时间戳的端点容差
    t32 = np.array([0.0, 100.00000762939453, 200.00001525878906, 900.0000610351562])
    s4 = select_frames(t32, stop_ps=900)
    assert s4.n_frames == 4, s4.n_frames

    s5 = select_frames(times, max_frames=5)
    assert s5.n_frames == 5
    return ("起点/终点/间隔/平衡段/最大帧数/float32 端点容差 全部通过")


def test_export_and_result_containers():
    """结果容器与导出：CSV/Excel 可写可读，数据能原样还原。"""
    import shutil
    import pandas as pd

    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]
    res = conf.analyze_rg(mdt, prot, select_frames(mdt.times_ps, max_frames=4))

    dfs = res.dataframes()
    assert len(dfs) == 2, len(dfs)
    main = res.to_dataframe()
    assert main.shape[0] == 4

    td = os.path.join(_HERE, "_selftest_out")
    if os.path.isdir(td):
        shutil.rmtree(td, ignore_errors=True)
    try:
        from mdta.export import export_all

        out = export_all([res], td, formats=("csv", "png"), excel=True)
        assert out["excel"] and os.path.isfile(out["excel"])
        assert all(os.path.isfile(p) for p in out["csv"])
        assert all(os.path.isfile(p) for p in out["png"])
        back = pd.read_csv(out["csv"][0], encoding="utf-8-sig")
        assert back.shape[0] == 4
        xl = pd.read_excel(out["excel"], sheet_name="汇总")
        assert len(xl) == 1
        assert os.path.getsize(out["png"][0]) > 5000
        return (f"导出 {len(out['csv'])} 个 CSV + {len(out['png'])} 个 PNG + summary.xlsx，"
                f"回读行数与原始一致（{main.shape[0]} 行）")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_selection_and_components():
    """选择模块：组分识别与原子选择在真实体系上给出预期结果。"""
    d = _ctx()
    u = d["mdt"].universe
    cats = classify_residues(u)
    assert "protein" in cats and "water" in cats, list(cats)
    assert d["protein"].n_atoms > 0 and d["water"].n_atoms > 0
    assert d["protein"].n_atoms + d["water"].n_atoms + d["ion"].n_atoms == u.atoms.n_atoms

    # 按原子名选择
    ow = select(u, name="OW")
    assert ow.n_atoms == 11084 or ow.n_atoms == len(cats["water"]), ow.n_atoms
    # 按残基号选择 + within 组合
    sub = select(u, resname="SOL", within=4.0, of="protein")
    assert 0 < sub.n_atoms < u.atoms.n_atoms
    return (f"protein={d['protein'].n_atoms} water={d['water'].n_atoms} "
            f"ion={d['ion'].n_atoms}；界面区内水原子 {sub.n_atoms}")


def test_exclude_bonded_removes_12_pairs():
    """RDF 键连剔除：剔除 1-2 原子对后，键长尺度上的峰必须消失。"""
    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]
    sel = select_frames(mdt.times_ps, max_frames=2)

    plain = ifc.analyze_rdf(mdt, {"P": prot}, sel, rmax=6.0, nbins=120,
                            exclude_bonded=False)
    excl = ifc.analyze_rdf(mdt, {"P": prot}, sel, rmax=6.0, nbins=120,
                           exclude_bonded=True)

    def band(res):
        c = res.curves_of(0)[0]
        m = c.x <= 1.6                       # 键长区间
        return float(c.y[m].sum())

    b_plain, b_excl = band(plain), band(excl)
    assert b_plain > 1.0, b_plain            # 未剔除时键长区间有大量原子对
    # 剔除后应只剩极少量 <1.6 Å 的 1-3 对（如 H–C–H），比未剔除时小两个数量级以上
    assert b_excl < 0.02 * b_plain, (b_plain, b_excl)

    p_plain = plain.summary["P-P 第一峰位置 (Å)"]
    p_excl = excl.summary["P-P 第一峰位置 (Å)"]
    assert p_plain < 1.7, p_plain            # 未剔除时峰落在键长尺度
    assert p_excl > 1.7, p_excl              # 剔除后峰移到非键连尺度
    g_plain = plain.summary["P-P 第一峰高度 g_max"]
    g_excl = excl.summary["P-P 第一峰高度 g_max"]
    return (f"未剔除：第一峰 {p_plain:.2f} Å, g_max={g_plain:.1f}；"
            f"剔除 1-2 对后：第一峰 {p_excl:.2f} Å, g_max={g_excl:.2f}；"
            f"r<1.6 Å 区间 g 之和 {b_plain:.1f} → {b_excl:.2f}")


def test_msd_of_static_system_is_zero():
    """MSD：完全不动的体系 MSD 必须恒为 0（回归测试）。"""
    u = mda.Universe.empty(20, n_residues=1, atom_resindex=[0] * 20, trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * 20)
    u.dimensions = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0], dtype=np.float32)
    n_frames = 5
    u.trajectory = mda.coordinates.memory.MemoryReader(
        np.tile(np.arange(60.0).reshape(20, 3), (n_frames, 1, 1)), order="fac")
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n_frames, dtype=float) * 10.0)
    lag, msd, diag = dyn.compute_msd(mdt, u.atoms, sel)
    assert np.allclose(msd, 0.0), msd
    return "静止体系 MSD 恒为 0，跨帧展开无漂移"


def test_msd_of_ballistic_motion():
    """MSD：匀速直线运动应给出 MSD = v²τ²（精确解）。"""
    v = np.array([0.01, 0.0, 0.0])          # Å/ps
    dt = 10.0
    n_frames = 6
    pos = np.array([v * (i * dt) for i in range(n_frames)])
    block = np.repeat(pos[:, None, :], 10, axis=1)      # 10 个原子同速运动
    u = mda.Universe.empty(10, n_residues=1, atom_resindex=[0] * 10, trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * 10)
    u.trajectory = mda.coordinates.memory.MemoryReader(
        block, order="fac",
        dimensions=np.array([1000.0, 1000.0, 1000.0, 90.0, 90.0, 90.0],
                            dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n_frames, dtype=float) * dt)
    lag, msd, diag = dyn.compute_msd(mdt, u.atoms, sel)
    # 盒子足够大（1000 Å）且位移很小，最小镜像不该有任何失效
    assert diag["失效比例"] == 0.0, diag
    expect = (np.linalg.norm(v) ** 2) * lag ** 2
    worst = float(np.abs(msd - expect).max())
    # MemoryReader 以 float32 存储坐标，容差取 float32 精度量级
    assert worst < 1e-6, (msd, expect, worst)
    return f"MSD 与 v²τ² 精确一致（最大偏差 {worst:.2e} Å²）"


def test_rdf_coordination_number_is_subsample_invariant():
    """RDF 抽稀：配位数不能随抽稀比例改变（g(r) 与 CN 都是密度归一化的量）。

    历史 bug：配位数用**抽稀后**的原子数算数密度，导致 CN 被整体缩小 step 倍
    （实测 4.67 → 2.31 → 1.17）。这里要求抽稀前后一致；
    抽稀本身只带来统计噪声，故容差按噪声量级给（远小于 bug 造成的成倍偏差）。
    """
    d = _ctx()
    mdt = d["mdt"]
    ow = d["water"].select_atoms("name OW")
    if ow.n_atoms == 0:
        return "跳过：体系中找不到水的 OW 原子"
    sel = select_frames(mdt.times_ps, max_frames=4)
    vals = {}
    for mga in (0, 3000, 6000):
        res = ifc.analyze_rdf(mdt, {"OW": ow}, sel, rmax=9.0, nbins=90,
                             max_group_atoms=mga)
        vals[mga] = (res.summary["OW-OW 配位数 (第一壳层)"],
                     res.summary["OW-OW 第一峰高度 g_max"])
    cn_full, gm_full = vals[0]
    for mga, (cn, gm) in vals.items():
        rel = abs(cn - cn_full) / cn_full
        # bug 版本这里是 0.75（1/4 抽稀），正常版本只是几个百分点的噪声
        assert rel < 0.10, (mga, cn, cn_full, "抽稀改变了配位数")
        assert abs(gm - gm_full) / gm_full < 0.12, (mga, gm, gm_full)
    detail = "，".join(f"上限{mga}: CN={cn:.3f}/g_max={gm:.3f}"
                       for mga, (cn, gm) in vals.items())
    return f"配位数与峰高不随抽稀变化（{detail}）"


def test_msd_flags_broken_minimum_image_tracking():
    """MSD 可靠性：最小镜像追踪失效时，即使 R²/α 很漂亮也必须降级提示。

    构造：小盒（20 Å）、每帧在 x/y 上各走 8 Å（最小化后 |d| = 11.3 Å > 半盒高
    10 Å），于是诊断判定最小镜像追踪不可靠。而 MSD 曲线本身仍是精确直线、
    R² 接近 1，若只看 R²/α 会得出"D 良好"的错误结论。
    """
    n_frames, n_atoms, box = 6, 40, 20.0
    # 每帧在 x、y 各走 8 Å：最小化后 |d| = 11.3 Å > 半盒高 10 Å，会被诊断捕获；
    # 同时每个分量都 < 10 Å，所以位移本身没有被回绕，MSD 仍是精确的直线。
    step = 8.0
    rng = np.random.default_rng(11)
    pos = np.zeros((n_frames, n_atoms, 3))
    for k in range(n_frames):
        pos[k, :, 0] = k * step
        pos[k, :, 1] = k * step
        pos[k, :, 2] = rng.uniform(0.0, box, size=n_atoms)
    pos[:, :, :2] %= box
    u = mda.Universe.empty(n_atoms, n_residues=1, atom_resindex=[0] * n_atoms,
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * n_atoms)
    boxvec = np.array([box, box, box, 90.0, 90.0, 90.0], dtype=np.float32)
    # 必须把 dimensions 交给 MemoryReader：只设 u.dimensions 会在赋值 trajectory
    # 时被覆盖成 None，于是 PBC 诊断整段被跳过（这个坑很隐蔽）。
    u.trajectory = mda.coordinates.memory.MemoryReader(pos, order="fac",
                                                       dimensions=boxvec)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n_frames, dtype=float) * 100.0)
    res = dyn.analyze_msd(mdt, {"X": u.atoms}, sel)
    fail = res.summary["X 最小镜像失效比例"]
    verdict = res.summary["X 拟合可靠性"]
    assert fail > 0.9, fail
    assert "最小镜像失效比例" in verdict, verdict
    assert "不建议引用" in verdict, verdict
    return f"失效比例 {fail * 100:.0f}% 时可靠性被降级（{verdict[:34]}…）"


def test_contact_same_group_matches_self_capped_distance():
    """接触分析：**同组（A–A）**时必须用无序对、不含 i==i 自配对。

    历史 bug：不分同组异组，一律 ``capped_distance(A, A)``，导致
    (1) 混进 N 个距离为 0 的自配对 → 最小原子间距恒为 0；
    (2) 每对无序对被记两次 → 接触对数翻倍；
    (3) 慢 5～7 倍（21384 原子实测 1.01 s vs 0.186 s 每帧）。
    这里把分析结果与 ``self_capped_distance`` 的独立结果直接对照。
    """
    from MDAnalysis.lib.distances import capped_distance, self_capped_distance

    d = _ctx()
    mdt = d["mdt"]
    ow = d["water"].select_atoms("name OW")
    if ow.n_atoms < 2:
        return "跳过：水的 OW 原子太少"
    sel = select_frames(mdt.times_ps, max_frames=1)
    mdt.universe.trajectory[int(np.asarray(sel.indices)[0])]
    box = mdt.universe.dimensions
    pos = np.asarray(ow.positions, dtype=float)
    pairs, dist = self_capped_distance(pos, max_cutoff=5.0, box=box,
                                       return_distances=True)

    res = ifc.analyze_contacts(mdt, ow, ow, sel, cutoff=5.0)
    got_pairs = float(res.curves[0].y[0])      # 面板 0：接触对数
    got_min = float(res.curves[2].y[0])        # 面板 1：最小原子间距
    assert int(got_pairs) == int(pairs.shape[0]), (got_pairs, pairs.shape[0])
    assert abs(got_min - float(np.min(dist))) < 1e-6, (got_min, np.min(dist))
    assert got_min > 0.5, ("同组最小间距不应为 0（自配对混进来了）", got_min)
    assert "同组" in res.summary.get("配对方式", ""), res.summary.get("配对方式")

    # 与"错误的旧做法"对比，确认差异确实存在
    old_pairs, old_dist = capped_distance(pos, pos, max_cutoff=5.0, box=box,
                                          return_distances=True)
    assert old_dist.min() == 0.0, "旧做法本应出现距离 0 的自配对"
    assert int(old_pairs.shape[0]) > int(pairs.shape[0]), (
        "旧做法的配对数本应更多（自配对 + 重复计数）")
    return (f"同组 {ow.n_atoms} 原子 OW：分析 {int(got_pairs):,} 对"
            f"（= self_capped {pairs.shape[0]:,}），最小间距 {got_min:.4f} Å；"
            f"旧做法会得到 {old_pairs.shape[0]:,} 对且最小间距 0")


def test_contact_cross_group_matches_capped_distance():
    """接触分析：**异组（A–B）**时就是 A×B 笛卡尔积，与 capped_distance 一致。"""
    from MDAnalysis.lib.distances import capped_distance

    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]
    ow = d["water"].select_atoms("name OW")
    if prot.n_atoms == 0 or ow.n_atoms == 0:
        return "跳过：缺少蛋白或水"
    sel = select_frames(mdt.times_ps, max_frames=1)
    mdt.universe.trajectory[int(np.asarray(sel.indices)[0])]
    box = mdt.universe.dimensions
    pa = np.asarray(prot.positions, dtype=float)
    pb = np.asarray(ow.positions, dtype=float)
    pairs, dist = capped_distance(pa, pb, max_cutoff=5.0, box=box,
                                  return_distances=True)

    res = ifc.analyze_contacts(mdt, prot, ow, sel, cutoff=5.0)
    assert int(res.curves[0].y[0]) == int(pairs.shape[0]), (
        res.curves[0].y[0], pairs.shape[0])
    assert abs(float(res.curves[2].y[0]) - float(np.min(dist))) < 1e-6
    assert "异组" in res.summary.get("配对方式", ""), res.summary.get("配对方式")
    return (f"异组 protein×OW：{int(pairs.shape[0]):,} 对，"
            f"最小间距 {dist.min():.4f} Å，与 capped_distance 完全一致")


def test_contact_min_distance_not_rescanned():
    """接触分析：已有接触时不应再调用 min_distance 重扫一遍（纯浪费一倍时间）。

    数学依据：``capped_distance(cutoff)`` 已经拿到 cutoff 内所有原子对，
    只要结果非空，其中最小距离就是**全局最小**（cutoff 外的只会更大）。
    """
    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]
    ow = d["water"].select_atoms("name OW")
    sel = select_frames(mdt.times_ps, max_frames=1)
    calls = {"n": 0}
    orig = ifc.min_distance

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ifc.min_distance = counting
    try:
        res = ifc.analyze_contacts(mdt, prot, ow, sel, cutoff=5.0)
    finally:
        ifc.min_distance = orig

    from MDAnalysis.lib.distances import capped_distance

    mdt.universe.trajectory[int(np.asarray(sel.indices)[0])]
    _, dist = capped_distance(np.asarray(prot.positions, dtype=float),
                              np.asarray(ow.positions, dtype=float),
                              max_cutoff=5.0, box=mdt.universe.dimensions,
                              return_distances=True)
    has_contact = dist.size > 0
    if has_contact:
        assert calls["n"] == 0, ("有接触时不该再扫一遍 min_distance",
                                 calls["n"])
        assert abs(float(res.curves[2].y[0]) - float(dist.min())) < 1e-6
        return (f"有接触（{dist.shape[0]:,} 对）：min_distance 调用 0 次，"
                f"最小间距仍精确 = {dist.min():.4f} Å")
    return f"该帧无接触，min_distance 调用 {calls['n']} 次（预期行为）"


# ==================================================================== 运行器
def main(argv: list[str] | None = None) -> int:
    global _CLI_ARGS
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    args = list(sys.argv[1:] if argv is None else argv)
    top = args[0] if len(args) > 0 and args[0].lower().endswith(
        (".tpr", ".gro", ".pdb", ".psf", ".prmtop")) else None
    xtc = args[1] if len(args) > 1 else None
    _CLI_ARGS = (top, xtc)
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    order = [
        "test_selection_and_components",
        "test_frame_selection_semantics",
        "test_rg_implementation_matches_mdanalysis_and_independent_formula",
        "test_pbc_unwrap_changes_rg_and_is_correct",
        "test_end_to_end_distance_manual",
        "test_dihedral_matches_independent_atan2_formula",
        "test_phi_psi_matches_mdanalysis_native_analysis",
        "test_water_density_is_physical",
        "test_rdf_matches_mdanalysis_interrdf",
        "test_exclude_bonded_removes_12_pairs",
        "test_water_rdf_physics",
        "test_rdf_coordination_number_is_subsample_invariant",
        "test_contact_same_group_matches_self_capped_distance",
        "test_contact_cross_group_matches_capped_distance",
        "test_contact_min_distance_not_rescanned",
        "test_nematic_order_parameter_on_known_distributions",
        "test_orientation_isotropic_system",
        "test_interface_width_on_synthetic_erf_profile",
        "test_interface_width_sharp_step",
        "test_interface_width_two_interfaces_are_measured_locally",
        "test_msd_of_static_system_is_zero",
        "test_msd_of_ballistic_motion",
        "test_msd_flags_broken_minimum_image_tracking",
        "test_msd_water_diffusion_coefficient",
        "test_export_and_result_containers",
    ]
    rank = {n: i for i, n in enumerate(order)}
    tests.sort(key=lambda kv: rank.get(kv[0], 999))

    print("=" * 78)
    print("MD 轨迹分析工具 —— 数值正确性自检")
    print("=" * 78)
    passed = failed = skipped = 0
    for name, fn in tests:
        label = name[5:].replace("_", " ")
        print(f"\n[测试] {label}")
        try:
            msg = fn()
            if isinstance(msg, str) and msg.startswith("跳过"):
                skipped += 1
                print(f"   ~ {msg}")
            else:
                passed += 1
                print(f"   ✓ 通过  {msg}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"   ✗ 失败  {type(exc).__name__}: {exc}")
            if os.environ.get("MDTA_TEST_TRACE"):
                traceback.print_exc()
    print("\n" + "=" * 78)
    print(f"结果：通过 {passed}，失败 {failed}，跳过 {skipped}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
