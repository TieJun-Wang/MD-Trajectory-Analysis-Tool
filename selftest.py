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
from mdta.analysis import crystallinity as cry              # noqa: E402
from mdta.analysis import dynamics as dyn                   # noqa: E402
from mdta.analysis import interface as ifc                  # noqa: E402
from mdta.analysis.base import positions_for                # noqa: E402
from mdta.io import MDTrajectory, load_trajectory           # noqa: E402
from mdta.preprocess import select_frames                   # noqa: E402
from mdta.selection import classify_residues, select        # noqa: E402
from mdta.systeminfo import describe_system                 # noqa: E402

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
    """（已废弃，保留为兼容壳）端到端距离的口径已改为键图/主链端基。

    1.0.0 的这条测试断言"结果 == 首尾原子间距"，即把 ``ag[[0, -1]]`` 当作链端。
    该定义本身是错的（见 ``test_end_to_end_distance_uses_bond_graph_ends``），
    所以这里改为复现旧口径，确保 ``ends="selection"`` 仍能给出与旧版完全相同的数。
    """
    d = _ctx()
    mdt = d["mdt"]
    prot = d["protein"]
    sel = d["sel"]
    u = mdt.universe
    res = conf.analyze_end_to_end(mdt, prot, sel, ends="selection")
    manual = np.empty(len(sel.indices))
    for k, i in enumerate(sel.indices):
        u.trajectory[int(i)]
        p = positions_for(prot, unwrap=True)
        manual[k] = np.linalg.norm(p[-1] - p[0])
    mine = res.curves_of(0)[0].y
    worst = float(np.abs(mine - manual).max())
    assert worst < 1e-9, worst
    return (f"旧口径（ends='selection'）逐帧最大偏差 {worst:.2e} Å，"
            f"平均值 {mine.mean():.3f} Å —— 与 1.0.0 完全一致")


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
def _curve_of(res, label: str):
    """按标签取一条曲线（找不到就报出全部标签，便于定位）。"""
    for c in res.curves:
        if c.label == label:
            return c
    raise AssertionError(f"找不到曲线 {label!r}；现有曲线：{[c.label for c in res.curves]}")


def test_rdf_modes_partition_pairs_exactly():
    """RDF 三模式：inter/intra 必须**穷尽且互斥**地划分全部原子对。

    合成体系：8 个双原子分子（键长 1.0 Å），分子间距 9 Å，于是 1.0 Å 处的
    计数只可能来自分子内配对，分子间配对全部 ≥ 8 Å——两类配对在距离上完全分开。
    断言：手算配对数成立、逐 bin 有 hist_total == hist_inter + hist_intra、
    键长峰只出现在 intra、归一化分母用的是该模式的真实配对数。
    """
    n_mol, spacing, bond = 8, 9.0, 1.0
    pos = []
    for m in range(n_mol):
        base = np.array([m * spacing, 0.0, 0.0])
        pos += [base, base + np.array([bond, 0.0, 0.0])]
    block = np.asarray(pos, dtype=np.float32)[None, :, :]           # (1, 16, 3)
    u = mda.Universe.empty(2 * n_mol, n_residues=n_mol,
                           atom_resindex=[m for m in range(n_mol) for _ in range(2)],
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * (2 * n_mol))
    u.add_TopologyAttr("molnums", np.arange(n_mol, dtype=int))       # 残基级
    u.trajectory = mda.coordinates.memory.MemoryReader(
        block, order="fac",
        dimensions=np.array([400.0, 400.0, 400.0, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))

    counts, why = ifc.mode_pair_counts(u.atoms, u.atoms, True)
    assert counts is not None, why
    assert counts["total"] == 16 * 15, counts
    assert counts["intra"] == n_mol * 2, counts                   # 每个分子 2×1 个有序对
    assert counts["inter"] + counts["intra"] == counts["total"], counts

    edges = np.linspace(0.0, 12.0, 121)
    hist = {}
    for mode in ("inter", "intra", "total"):
        h, nu, _ = ifc._rdf_accumulate(u, u.atoms, u.atoms, edges, [0], True, mode=mode)
        assert nu == 1
        hist[mode] = h
    assert np.array_equal(hist["inter"] + hist["intra"], hist["total"]), "三模式未穷尽划分"
    k = int(np.searchsorted(edges, bond) - 1)
    assert hist["intra"][k] == n_mol * 2, (hist["intra"][k], "键长峰未完整落在 intra")
    assert hist["intra"].sum() == n_mol * 2, "intra 出现多余的近距离配对"
    assert hist["inter"][:k + 2].sum() == 0, "分子间配对里混入了近程配对"

    res = ifc.analyze_rdf(mdt, {"X": u.atoms}, sel, rmax=12.0, nbins=120, mode="inter")
    assert res.summary["X-X 配对模式"] == "inter", res.summary
    assert res.summary["X-X 归一化分母 (有序原子对)"] == counts["inter"], res.summary
    assert res.summary["X-X 分子内配对占比"] == 0.0, res.summary
    return (f"三模式穷尽互斥：total {counts['total']:.0f} = 分子间 {counts['inter']:.0f}"
            f" + 分子内 {counts['intra']:.0f}；键长峰仅见于 intra")


def test_rdf_inter_equals_total_for_single_atom_molecules():
    """单原子分子（水氧）：inter 与 total 必须逐点相同——口径改动对它零影响。

    每个水分子只贡献 1 个 OW，既无分子内 OW–OW 配对，两种模式的配对数也相同
    （N²−Σ1² = N(N−1)）。这既验证分母推导，也说明对"每分子单原子"的
    日常情形（水氧、离子等）本改动不改变任何数值。
    """
    d = _ctx()
    mdt = d["mdt"]
    ow = d["water"].select_atoms("name OW")
    if ow.n_atoms == 0:
        return "跳过：体系中找不到水的 OW 原子"
    sel = select_frames(mdt.times_ps, max_frames=3)
    a = ifc.analyze_rdf(mdt, {"OW": ow}, sel, rmax=9.0, nbins=90, mode="inter")
    b = ifc.analyze_rdf(mdt, {"OW": ow}, sel, rmax=9.0, nbins=90, mode="total")
    ca, cb = _curve_of(a, "OW-OW"), _curve_of(b, "OW-OW")
    worst = float(np.abs(ca.y - cb.y).max())
    assert worst < 1e-12, (worst, "单原子分子的 inter 与 total 不一致")
    na = a.summary["OW-OW 归一化分母 (有序原子对)"]
    nb = b.summary["OW-OW 归一化分母 (有序原子对)"]
    assert na == nb, (na, nb)
    return f"单原子分子 inter≡total（最大差 {worst:.1e}），配对数 {na:,.0f}"


def test_rdf_inter_mode_excludes_covalent_bond():
    """真实水体系：分子间 RDF 里**不应**出现 O–H 共价键峰。

    这是 1.0.0 的口径错误——total 模式把同一水分子内的 O–H（~1.0 Å）计进了
    g(r)，于是"分子间结构"曲线上冒出一个纯共价键尖峰。改口径后 inter 在
    r<1.4 Å 必须严格为 0，而这部分计数完整地出现在 intra 里。
    """
    d = _ctx()
    mdt = d["mdt"]
    w = d["water"]
    if w is None or w.n_atoms == 0:
        return "跳过：体系中找不到水"
    sel = select_frames(mdt.times_ps, max_frames=3)
    rmax, nbins = 6.0, 120
    tot = ifc.analyze_rdf(mdt, {"W": w}, sel, rmax=rmax, nbins=nbins, mode="total")
    itr = ifc.analyze_rdf(mdt, {"W": w}, sel, rmax=rmax, nbins=nbins, mode="inter")
    ina = ifc.analyze_rdf(mdt, {"W": w}, sel, rmax=rmax, nbins=nbins, mode="intra")
    ct, ci, cn = _curve_of(tot, "W-W"), _curve_of(itr, "W-W"), _curve_of(ina, "W-W")
    r = ct.x
    near = r < 1.4                                   # 共价 O–H 键长区间
    assert ct.y[near].max() > 1.0, ("total 应含共价键峰", float(ct.y[near].max()))
    assert float(ci.y[near].max()) == 0.0, ("分子间 RDF 混入了共价键峰",
                                            float(ci.y[near].max()))
    assert cn.y[near].max() > 1.0, "intra 应含共价键峰"
    first = int(np.argmax(ci.y > 0))
    return (f"inter 在 r<1.4 Å 严格为 0（total 该处 g_max={ct.y[near].max():.1f}），"
            f"分子间配对自 r={r[first]:.2f} Å 起出现；"
            f"intra 含键峰 g_max={cn.y[near].max():.1f}")


def test_msd_fft_matches_brute_force_and_tidynamics():
    """MSD 的 FFT 快速算法必须与暴力 (t₀,τ) 双重求和、tidynamics 逐点一致。

    为什么值得单测：``msd_fft`` 是重写后的核心，它一次对成百上千个粒子做批量
    FFT；而 ``tidynamics.msd`` 把第二轴当**空间分量**求和，只能逐粒子调用。
    两者数值必须一致，否则"FFT 加速"就成了换算法的借口。
    """
    rng = np.random.default_rng(7)
    n, P = 60, 25
    xyz = np.cumsum(rng.normal(scale=0.5, size=(n, P, 3)), axis=0)

    got = dyn.msd_fft(xyz[:, :, 0].copy())                     # 单轴、多粒子
    bf = np.array([float(np.mean((xyz[k:, :, 0] - xyz[:n - k, :, 0]) ** 2))
                   for k in range(n)])
    w1 = float(np.abs(got - bf).max())
    assert w1 < 1e-9 * max(1.0, float(np.abs(bf).max())), (w1, "与暴力法不一致")

    import tidynamics
    # ⚠️ tidynamics.msd 的输入是"单个粒子、多分量"，它把第二轴当**空间分量**求和，
    #    所以它给出的是三维 MSD；要跟它比就必须比"三轴之和/粒子数"，不能比单轴。
    acc = np.zeros(n)
    for p in range(P):                                          # 逐粒子调 tidynamics
        acc += np.asarray(tidynamics.msd(xyz[:, p, :].copy()))
    tid = acc / P
    tot = sum(dyn.msd_fft(xyz[:, :, i].copy()) for i in range(3))
    w2 = float(np.abs(tot - tid).max())
    assert w2 < 1e-9 * max(1.0, float(np.abs(tid).max())), (w2, "与 tidynamics 不一致")

    bf3 = np.array([float(np.mean(np.sum((xyz[k:] - xyz[:n - k]) ** 2, axis=2)))
                    for k in range(n)])
    w3 = float(np.abs(tot - bf3).max())
    assert w3 < 1e-9 * max(1.0, float(np.abs(bf3).max())), (w3, "三轴之和与全维不一致")
    return (f"FFT 与暴力法最大偏差 {w1:.2e}、与 tidynamics {w2:.2e}、"
            f"三轴之和与全维 {w3:.2e} Å²（{n} 帧 × {P} 粒子）")


def test_msd_molecule_com_excludes_internal_motion():
    """分子质心 MSD：刚体平动时质心 MSD 只含平动，原子 MSD 会多出内部运动。

    合成体系：6 个双原子分子整体匀速平移（v=0.5 Å/帧），同时分子内键长按
    cos 振荡（纯内部运动）。于是
    1. 质心 MSD 必须精确等于 v²τ²；
    2. 原子 MSD ≥ 质心 MSD（多出的就是内部运动）；
    3. 全部分子位移完全相同时，去漂移会把质心 MSD 减到 0（漂移=整体平移）。
    """
    n_frames, n_mol, box, v = 8, 6, 80.0, 0.5
    pos = np.zeros((n_frames, n_mol * 2, 3))
    for k in range(n_frames):
        for m in range(n_mol):
            base = np.array([10.0 + m * 5.0 + v * k, 10.0, 10.0])
            sep = 3.0 * float(np.cos(0.4 * k))                  # 键长振荡
            pos[k, 2 * m] = base + np.array([sep / 2.0, 0.0, 0.0])
            pos[k, 2 * m + 1] = base - np.array([sep / 2.0, 0.0, 0.0])
    u = mda.Universe.empty(n_mol * 2, n_residues=n_mol,
                           atom_resindex=[m for m in range(n_mol) for _ in range(2)],
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * (n_mol * 2))
    u.add_TopologyAttr("molnums", np.arange(n_mol, dtype=int))
    u.trajectory = mda.coordinates.memory.MemoryReader(
        pos, order="fac",
        dimensions=np.array([box, box, box, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n_frames, dtype=float) * 10.0)

    lag, m_com, diag = dyn.compute_msd(mdt, u.atoms, sel, object="molecule",
                                       remove_drift=False)
    _, m_atom, _ = dyn.compute_msd(mdt, u.atoms, sel, object="atom",
                                   remove_drift=False)
    _, m_drift, _ = dyn.compute_msd(mdt, u.atoms, sel, object="molecule",
                                    remove_drift=True)
    assert diag["对象"] == "molecule", diag
    assert diag["粒子数"] == n_mol, diag
    # v 是 Å/**帧**，lag 是 ps：每帧 10 ps，故期望 MSD = (v·k)² = (v·τ/10)²
    dt = 10.0
    expect = (v * np.asarray(lag, dtype=float) / dt) ** 2
    w = float(np.abs(m_com - expect).max())
    assert w < 1e-6, (m_com, expect, w)
    gain = float(np.nanmax(m_atom - m_com))
    assert gain > 1e-3, (gain, "原子 MSD 应比质心 MSD 多出内部运动")
    assert float(np.nanmax(np.abs(m_drift))) < 1e-9, m_drift
    return (f"质心 MSD 与 v²τ² 最大偏差 {w:.1e} Å²；原子 MSD 额外含内部运动 "
            f"(最大 +{gain:.3f} Å²)；完全同向平移时去漂移后 MSD≈0")


def test_msd_anisotropy_reports_in_plane_vs_normal():
    """各向异性：给定 D_x=D_y=4·D_z 的随机行走，D∥/D⊥ 应还原为 4。

    这是"各向异性"这个数的**定义级**校验：MSD∥=MSD_x+MSD_y 按 2 维拟合
    （MSD∥=4D∥t），MSD⊥=MSD_z 按 1 维拟合（MSD⊥=2D⊥t）。
    """
    rng = np.random.default_rng(5)
    n, P = 600, 1000
    step = rng.normal(scale=np.array([1.0, 1.0, 0.5]), size=(n, P, 3))
    xyz = np.cumsum(step, axis=0).astype(np.float32)
    u = mda.Universe.empty(P, n_residues=P, atom_resindex=list(range(P)),
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * P)
    u.add_TopologyAttr("molnums", np.arange(P, dtype=int))
    u.trajectory = mda.coordinates.memory.MemoryReader(
        xyz, order="fac",
        dimensions=np.array([2000.0, 2000.0, 2000.0, 90.0, 90.0, 90.0],
                            dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n, dtype=float) * 1.0)
    res = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                          remove_drift=False, fit_fraction=(0.05, 0.5))
    ratio = float(res.summary["X 各向异性 D∥/D⊥"])
    rel = abs(ratio - 4.0) / 4.0
    assert rel < 0.08, (ratio, "D∥/D⊥ 应约为 4")
    return (f"σ=(1,1,0.5) 的随机行走 → D∥/D⊥ = {ratio:.2f}（期望 4.0，"
            f"相对偏差 {rel * 100:.1f}%）")


def test_contact_average_number_is_meaningful_and_probability_is_saturated():
    """接触指标：`存在接触的帧比例` 恒为 1（无区分度），真正的量是「平均接触数」。

    问题：该值定义为 ``mean(每帧接触对数 > 0)``。稠密体系里几万个原子对中永远
    至少有一对落在 cutoff 内，因此它对任何体系都恒为 1（实测 AdK / 糖蛋白 / 46
    三个体系全部为 1），**没有区分度**——不是算错了，是这个定义不成立。
    断言：
    1. 稠密体系中该值 == 1 且被显式标记为饱和、给出说明；
    2. 新增的「平均接触数」与手工逐原子统计**完全一致**；
    3. 「接触对占据率」覆盖的接触对数与手工统计一致，取值落在 (0, 1]。
    """
    d = _ctx()
    mdt = d["mdt"]
    u = mdt.universe
    prot, wat = d["protein"], d["water"]
    if prot.n_atoms == 0 or wat.n_atoms == 0:
        return "跳过：体系中缺少蛋白或水"
    sel = select_frames(mdt.times_ps, max_frames=4)
    res = ifc.analyze_contacts(mdt, prot, wat, sel, cutoff=5.0)
    assert float(res.summary["接触概率（存在接触的帧比例）"]) == 1.0, res.summary
    assert str(res.summary["接触概率是否饱和"]).startswith("是"), res.summary
    assert any("不能用来比较不同体系" in n for n in res.notes), res.notes

    from MDAnalysis.lib.distances import capped_distance

    from mdta.analysis.base import frame_iterator
    n_all = int(u.atoms.n_atoms)
    acc = np.zeros(prot.n_atoms)
    seen: dict[int, int] = {}
    nfr = 0
    for _f, _t in frame_iterator(mdt, sel):
        pa = np.asarray(prot.positions, dtype=float)
        pb = np.asarray(wat.positions, dtype=float)
        pr, _dd = capped_distance(pa, pb, max_cutoff=5.0, box=u.dimensions,
                                  return_distances=True)
        if pr.size:
            np.add.at(acc, pr[:, 0], 1)
            gi = (np.asarray(prot.indices)[pr[:, 0]] * n_all
                  + np.asarray(wat.indices)[pr[:, 1]])
            for key in np.unique(gi).tolist():
                seen[int(key)] = seen.get(int(key), 0) + 1
        nfr += 1
    manual = acc / nfr
    got = float(res.summary["平均接触数（每个 A 组原子）"])
    assert abs(got - float(manual.mean())) < 1e-9, (got, float(manual.mean()))
    assert int(res.summary["平均接触数为 0 的 A 组原子数"]) == int(np.sum(manual == 0))
    n_seen = int(res.summary["出现过的不同接触对总数"])
    assert n_seen == len(seen), (n_seen, len(seen))
    n_full = sum(1 for v in seen.values() if v == nfr)
    assert int(res.summary["始终接触（占据率 = 1）的接触对数"]) == n_full, n_full
    assert n_full < n_seen, "不应所有接触对都始终存在（否则占据率同样无区分度）"
    return (f"存在接触的帧比例={res.summary['接触概率（存在接触的帧比例）']}（已标记饱和）；"
            f"平均接触数={got:.3f} 与手工完全一致；"
            f"接触对 {n_seen:,} 个，其中始终接触 {n_full:,} 个")


def test_end_to_end_distance_uses_bond_graph_ends():
    """端到端距离：链端必须由键连接图/主链端基确定，而不是"首尾原子"。

    旧口径取 ``ag[[0, -1]]``（所选原子组里索引最小与最大的两个原子）。对蛋白它是
    文件里的第一个原子与最后一个原子（碰巧接近端基，但纯属偶然）；对含多个分子的
    组分则完全没有意义。断言：
    1. 识别出的链端与"首尾原子"可以是不同的原子（并报出两者差异）；
    2. 新的 R_ee 与手工按识别出的端原子算出的距离**逐帧一致**；
    3. ``ends="selection"`` 能复现旧口径（接口兼容）。
    """
    d = _ctx()
    mdt = d["mdt"]
    u = mdt.universe
    prot = d["protein"]
    sel = d["sel"]
    info = conf.bond_graph_ends(prot)
    assert info["ok"] and info["ends"] is not None, info
    e1, e2 = int(info["ends"][0]), int(info["ends"][1])

    res = conf.analyze_end_to_end(mdt, prot, sel)
    mine = res.curves_of(0)[0].y
    pair = u.atoms[[e1, e2]]
    manual = np.empty(len(sel.indices))
    for k, i in enumerate(sel.indices):
        u.trajectory[int(i)]
        p = positions_for(pair, unwrap=True)
        manual[k] = np.linalg.norm(p[1] - p[0])
    worst = float(np.abs(mine - manual).max())
    assert worst < 1e-9, worst

    # 旧口径（首尾原子）作为对照：两者是**不同**的原子对，数值也不同
    old_pair = (int(prot.indices[0]), int(prot.indices[-1]))
    res_old = conf.analyze_end_to_end(mdt, prot, sel, ends="selection")
    old = float(np.nanmean(res_old.curves_of(0)[0].y))
    new = float(np.nanmean(mine))
    assert res.summary["链端来源"] != res_old.summary["链端来源"]
    return (f"链端 {info.get('method')}：{res.summary.get('链端原子')} → "
            f"R_ee={new:.3f} Å；旧口径（首尾原子 {old_pair[0]}/{old_pair[1]}）"
            f"={old:.3f} Å（差 {abs(new - old) / old * 100:.1f}%）；"
            f"逐帧与手工最大偏差 {worst:.1e} Å")


def test_rg_per_molecule_for_multi_molecule_group():
    """Rg 按分子：多分子组分的"整组 Rg"不是"分子有多大"。

    合成体系：3 个正方形刚性分子（边长 a、2a、3a），各自 Rg = 边长/√2，
    分子之间相距 100 Å。于是
    1. 逐分子 Rg 必须精确等于解析值 (a+2a+3a)/√2/… 的平均；
    2. 整组 Rg 远大于单分子 Rg（这就是 1.0.0 会把两者混为一谈的地方）。
    """
    a = 2.0
    centers = [0.0, 100.0, 200.0]
    pos = []
    res_expected = []
    for m, (c, side) in enumerate(zip(centers, (a, 2 * a, 3 * a))):
        s = side
        pos += [[c, 0.0, 0.0], [c + s, 0.0, 0.0], [c + s, s, 0.0], [c, s, 0.0]]
        res_expected.append(side / np.sqrt(2.0))
    n_atoms = len(pos)
    n_mol = 3
    block = np.asarray(pos, dtype=np.float32)[None, :, :]
    u = mda.Universe.empty(n_atoms, n_residues=n_mol,
                           atom_resindex=[m for m in range(n_mol) for _ in range(4)],
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * n_atoms)
    u.add_TopologyAttr("molnums", np.arange(n_mol, dtype=int))
    u.add_TopologyAttr("bonds", [(4 * m + i, 4 * m + (i + 1) % 4)
                                 for m in range(n_mol) for i in range(4)])
    u.trajectory = mda.coordinates.memory.MemoryReader(
        np.repeat(block, 2, axis=0), order="fac",
        dimensions=np.array([500.0, 500.0, 500.0, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0, 1.0]))
    res = conf.analyze_rg(mdt, u.atoms, sel, label="X")

    assert int(res.summary["分子数"]) == n_mol, res.summary
    got = float(res.summary["单分子 Rg 平均 (Å)"])
    expect = float(np.mean(res_expected))
    assert abs(got - expect) < 1e-5, (got, expect)
    whole = float(res.summary["Rg mean"])
    assert whole > 5.0 * got, (whole, got, "整组 Rg 应远大于单分子 Rg")
    assert len(res.curves_of(2)) >= 1, "缺少逐分子 Rg 面板"
    return (f"3 个分子（边长 {a:g}/{2 * a:g}/{3 * a:g} Å）：单分子 Rg 平均 "
            f"{got:.4f} Å（解析值 {expect:.4f}）；整组 Rg {whole:.2f} Å "
            f"= 单分子的 {whole / got:.1f} 倍")


def test_ree_refuses_ring_and_handles_linear_chain():
    """R_ee 的边界：线形链给出精确端距；**环状分子必须拒绝**而不是硬算。

    合成体系：① 5 原子直链（键 0-1-2-3-4），两端相距 8 Å；
    ② 6 元环（键 0-1-…-5-0），每个原子度均为 2 → 没有端基。
    断言：直链 R_ee == 8 Å 且端原子就是 0 与 4；环状体系
    「R_ee 是否可定义」= 否、没有 R_ee 曲线、原因里说明是环状。
    """
    # ---- 直链
    lin = np.array([[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0], [6.0, 0, 0], [8.0, 0, 0]],
                   dtype=np.float32)
    u1 = mda.Universe.empty(5, n_residues=1, atom_resindex=[0] * 5, trajectory=True)
    u1.add_TopologyAttr("masses", [12.0] * 5)
    u1.add_TopologyAttr("molnums", [0])
    u1.add_TopologyAttr("bonds", [(0, 1), (1, 2), (2, 3), (3, 4)])
    u1.trajectory = mda.coordinates.memory.MemoryReader(
        lin[None, :, :], order="fac",
        dimensions=np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt1 = MDTrajectory.from_universe(u1)
    sel1 = select_frames(np.array([0.0]))
    info1 = conf.bond_graph_ends(u1.atoms)
    assert info1["ok"] and list(info1["ends"]) == [0, 4], info1
    r1 = conf.analyze_end_to_end(mdt1, u1.atoms, sel1)
    v1 = float(r1.curves_of(0)[0].y[0])
    assert abs(v1 - 8.0) < 1e-5, v1

    # ---- 6 元环（正六边形，边长 1.5 Å）
    ang = np.arange(6) * (np.pi / 3.0)
    ring = np.stack([1.5 * np.cos(ang), 1.5 * np.sin(ang), np.zeros(6)], axis=1)
    u2 = mda.Universe.empty(6, n_residues=1, atom_resindex=[0] * 6, trajectory=True)
    u2.add_TopologyAttr("masses", [12.0] * 6)
    u2.add_TopologyAttr("molnums", [0])
    u2.add_TopologyAttr("bonds", [(i, (i + 1) % 6) for i in range(6)])
    u2.trajectory = mda.coordinates.memory.MemoryReader(
        ring.astype(np.float32)[None, :, :], order="fac",
        dimensions=np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt2 = MDTrajectory.from_universe(u2)
    info2 = conf.bond_graph_ends(u2.atoms)
    assert not info2["ok"], info2
    assert "环状" in str(info2["reason"]), info2
    r2 = conf.analyze_end_to_end(mdt2, u2.atoms, select_frames(np.array([0.0])))
    assert r2.summary["R_ee 是否可定义"] == "否", r2.summary
    assert len(r2.curves) == 0, "环状分子不应给出 R_ee 曲线"
    return (f"直链 R_ee = {v1:.3f} Å（端原子 {list(info1['ends'])}）；"
            f"6 元环：度为 1 的原子 {info2['n_degree1']} 个 → 拒绝给出 R_ee"
            f"（{r2.summary['原因'][:28]}…）")


def _synth_chain(n_res: int, n_mol: int = 1, *, alternate: bool = False):
    """合成"每残基 2 个连接原子"的链：残基 i 的连接原子对是 (2i, 2i+1)。

    ``alternate=True`` 时奇数残基的矢量沿 y、偶数沿 x（用于构造已知 S 的分布）。
    """
    import MDAnalysis as mda

    pos = []
    for m in range(n_mol):
        y0 = 50.0 * m
        for i in range(n_res):
            if alternate and (i % 2 == 1):
                pos += [[i * 2.0, y0, 0.0], [i * 2.0, y0 + 1.0, 0.0]]
            else:
                pos += [[i * 2.0, y0, 0.0], [i * 2.0 + 1.0, y0, 0.0]]
    n_at = n_res * n_mol * 2
    u = mda.Universe.empty(n_at, n_residues=n_res * n_mol,
                           atom_resindex=[r for r in range(n_res * n_mol)
                                          for _ in range(2)], trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * n_at)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["MONO"] * (n_res * n_mol))
    u.add_TopologyAttr("molnums", np.repeat(np.arange(n_mol), n_res))
    bonds = []
    for m in range(n_mol):
        base = m * n_res * 2
        for i in range(n_res):
            bonds.append((base + 2 * i, base + 2 * i + 1))          # 残基内
            if i + 1 < n_res:
                bonds.append((base + 2 * i + 1, base + 2 * i + 2))  # 跨残基
    u.add_TopologyAttr("bonds", bonds)
    u.trajectory = mda.coordinates.memory.MemoryReader(
        np.asarray(pos, dtype=np.float32)[None, :, :], order="fac",
        dimensions=np.array([400.0, 400.0, 400.0, 90.0, 90.0, 90.0], dtype=np.float32))
    return MDTrajectory.from_universe(u), u


def test_orientation_uses_chemical_repeat_units():
    """取向链段默认必须是**化学重复单元**，且 S 在已知分布上给出理论值。

    合成体系：14 个残基的链，每个残基 2 个"连接原子"（连着相邻残基），
    于是中间的 12 个残基各给一个重复单元矢量。
    1. 链段原子对必须**精确等于**手算的连接原子对 (2i, 2i+1)；
    2. 全部平行 → S = 1；奇偶残基互相垂直 → S = 0.25（取向张量的解析值）。
    """
    mdt, u = _synth_chain(14)
    sel = select_frames(np.array([0.0]))
    pairs, info = cry.repeat_unit_pairs(u.atoms)
    expect = np.array([[2 * i, 2 * i + 1] for i in range(1, 13)], dtype=int)
    assert np.array_equal(pairs, expect), (pairs.tolist(), expect.tolist())
    assert info["n_skipped_terminal"] == 2, info

    r = cry.analyze_orientation(mdt, u.atoms, sel, label="合成")
    assert r.summary["链段来源"] == "chemical repeat unit", r.summary
    assert int(r.summary["链段数"]) == 12, r.summary
    s_align = float(r.summary["S mean"])
    assert abs(s_align - 1.0) < 1e-6, s_align

    mdt2, u2 = _synth_chain(14, alternate=True)
    r2 = cry.analyze_orientation(mdt2, u2.atoms, select_frames(np.array([0.0])), label="合成")
    s_alt = float(r2.summary["S mean"])
    assert abs(s_alt - 0.25) < 1e-6, (s_alt, "奇偶垂直分布的 S 应为 0.25")
    return (f"链段 = 化学重复单元（{pairs.shape[0]} 个，与手算连接原子对完全一致）；"
            f"全平行 S = {s_align:.4f}；奇偶垂直 S = {s_alt:.4f}（解析值 0.25）")


def test_orientation_ensemble_and_refuses_meaningless_cases():
    """取向必须做**多分子集合平均**；链段不足时拒绝给数而不是给假值。

    1. 2 个分子 × 14 残基 → 24 个链段、报告分子数 = 2（不是"只算一条链"）；
    2. 独立小分子（无跨残基连接键）→ 取向**拒绝给数**，多分量有序度指数
       也必须为 nan 且标记"不可定义"（不能给 0，0 会被读成"完全无序"）。
    """
    mdt, u = _synth_chain(14, n_mol=2)
    r = cry.analyze_orientation(mdt, u.atoms, select_frames(np.array([0.0])), label="双分子")
    assert int(r.summary["链段所属分子数"]) == 2, r.summary
    assert int(r.summary["链段数"]) == 24, r.summary
    assert any("集合平均" in n for n in r.notes), r.notes

    # 独立小分子：每个分子 3 个原子、分子之间没有键
    import MDAnalysis as mda
    pos = []
    for m in range(4):
        pos += [[20.0 * m, 0.0, 0.0], [20.0 * m + 1.5, 0.0, 0.0], [20.0 * m, 1.5, 0.0]]
    u2 = mda.Universe.empty(12, n_residues=12,
                            atom_resindex=list(range(12)), trajectory=True)
    u2.add_TopologyAttr("masses", [12.0] * 12)
    u2.add_TopologyAttr("elements", ["C"] * 12)
    u2.add_TopologyAttr("resnames", ["SOLV"] * 12)
    u2.add_TopologyAttr("molnums", list(range(12)))
    u2.add_TopologyAttr("bonds", [(3 * m, 3 * m + 1) for m in range(4)]
                        + [(3 * m, 3 * m + 2) for m in range(4)])
    u2.trajectory = mda.coordinates.memory.MemoryReader(
        np.asarray(pos, dtype=np.float32)[None, :, :], order="fac",
        dimensions=np.array([200.0, 200.0, 200.0, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt2 = MDTrajectory.from_universe(u2)
    sel2 = select_frames(np.array([0.0]))
    r2 = cry.analyze_orientation(mdt2, u2.atoms, sel2, label="小分子")
    assert r2.summary["取向分析是否可定义"] == "否", r2.summary
    assert "S mean" not in r2.summary, r2.summary
    assert len(r2.curves) == 0, "拒绝时应没有曲线"

    o2 = cry.analyze_structural_order(mdt2, u2.atoms, sel2, label="小分子")
    assert o2.summary["指数是否可定义"] == "否", o2.summary
    val = o2.summary.get("多分量有序度指数 平均")
    assert val is None or (isinstance(val, float) and np.isnan(val)), val
    return (f"2 分子 × 14 残基 → {r.summary['链段数']} 个链段、"
            f"{r.summary['链段所属分子数']} 个分子（集合平均）；"
            f"独立小分子：取向与指数均标记不可定义、无曲线、指数为 nan")


def _synth_two_component(*, mixed: bool, n: int = 40, box: float = 80.0):
    """沿 z 方向的两组分体系：``mixed=True`` 均匀交错，``False`` 上下分层。"""
    import MDAnalysis as mda

    za = np.linspace(1.0, box - 1.0, n)
    pos_a = np.stack([np.full(n, 10.0), np.full(n, 10.0), za], axis=1)
    if mixed:
        # 均匀交错：B 平移半个格点 → 两组分沿 z 都是平分布
        zb = za + (box / n) / 2.0
    else:
        # 上下分层：A 占下半盒、B 占上半盒 → 真正的台阶
        zb = np.linspace(box / 2.0 + 1.0, box - 1.0, n)
        za = np.linspace(1.0, box / 2.0 - 1.0, n)
        pos_a = np.stack([np.full(n, 10.0), np.full(n, 10.0), za], axis=1)
    pos_b = np.stack([np.full(n, 20.0), np.full(n, 10.0), zb], axis=1)
    pos = np.concatenate([pos_a, pos_b], axis=0)
    n_at = 2 * n
    u = mda.Universe.empty(n_at, n_residues=n_at,
                           atom_resindex=list(range(n_at)), trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * n_at)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["A"] * n + ["B"] * n)
    u.add_TopologyAttr("molnums", list(range(n_at)))
    u.trajectory = mda.coordinates.memory.MemoryReader(
        pos.astype(np.float32)[None, :, :], order="fac",
        dimensions=np.array([box, box, box, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    return mdt, u


def test_interface_width_refuses_uniform_mixture():
    """界面宽度：**未分层**的体系必须拒绝套用 1D 台阶模型，真台阶仍给数。

    1.0.0 的问题：均匀混合体系里两组分密度都只在体相值附近涨落，涨落曲线互相
    穿越同样产生"交点"，于是把涨落尺度当成界面宽度报出来（实测 AdK 均匀溶液
    被报成「高（板层体系…）」并给出 36.6 Å，而盒长只有 80 Å）。
    断言：均匀交错 → 判不适用、宽度/位置键被移除、只留如实标注的涨落尺度；
    同样两组分上下分层 → 判适用并给出正常量级的宽度。
    """
    sel = select_frames(np.array([0.0]))
    mdt_mix, u_mix = _synth_two_component(mixed=True)
    r_mix = ifc.analyze_interface_width(mdt_mix, {"A": u_mix.select_atoms("resname A"),
                                                 "B": u_mix.select_atoms("resname B")},
                                        sel, pair=("A", "B"), axis=2, nbins=80,
                                        mode="number")
    assert r_mix.summary["1D 台阶模型是否适用"] == "否", r_mix.summary
    assert r_mix.summary["界面宽度是否可用"] == "否", r_mix.summary
    assert not any(k == "界面宽度 10-90 (Å)" for k in r_mix.summary), r_mix.summary
    assert not any(k.startswith("界面位置 (Å)") for k in r_mix.summary), r_mix.summary
    assert "不适用" in str(r_mix.summary["界面判据可靠性"]), r_mix.summary

    mdt_slab, u_slab = _synth_two_component(mixed=False)
    r_slab = ifc.analyze_interface_width(mdt_slab, {"A": u_slab.select_atoms("resname A"),
                                                    "B": u_slab.select_atoms("resname B")},
                                         sel, pair=("A", "B"), axis=2, nbins=80,
                                         mode="number")
    assert r_slab.summary["1D 台阶模型是否适用"] == "是", r_slab.summary
    w = float(r_slab.summary["界面宽度 10-90 (Å)"])
    assert np.isfinite(w) and 0.0 < w < 0.25 * 80.0, w
    return (f"均匀交错体系：判「不适用」、宽度与位置键已移除；"
            f"分层体系：判「适用」，界面宽度 {w:.2f} Å")


def test_density_labels_its_own_convention():
    """密度：结果里必须自带口径标注（质量密度 g/cm³ 还是数密度 1/Å³）。

    两种口径数值差一个摩尔质量量级（水的体相：0.997 g/cm³ vs 0.0334 个/Å³），
    导出成表以后只看"体相密度 0.168"无法分辨，因此口径必须写进结果本身。
    """
    d = _ctx()
    mdt = d["mdt"]
    sel = select_frames(mdt.times_ps, max_frames=2)
    groups = {"water": d["water"]}
    r_mass = ifc.analyze_density(mdt, groups, sel, axis=2, nbins=40, mode="mass")
    r_num = ifc.analyze_density(mdt, groups, sel, axis=2, nbins=40, mode="number")
    assert r_mass.summary["密度口径"].startswith("质量密度"), r_mass.summary
    assert r_num.summary["密度口径"].startswith("数密度"), r_num.summary
    assert r_mass.summary["密度单位"] == "g/cm³", r_mass.summary
    assert r_num.summary["密度单位"] == "1/Å³", r_num.summary
    bm = float(r_mass.summary["water 体相密度"])
    bn = float(r_num.summary["water 体相密度"])
    assert 0.9 < bm < 1.1, bm
    # 数密度 ×（该组的**每原子平均质量**）/ N_A 应回到质量密度。
    # 注意这里要按原子平均质量换算：water 组是"水分子里的全部原子"，不是分子数密度。
    m_avg = float(np.mean(np.asarray(d["water"].masses, dtype=float)))
    conv = bn * m_avg / 0.602214076
    assert abs(conv - bm) / bm < 0.02, (conv, bm, m_avg)
    return (f"质量密度 {bm:.4f} g/cm³ ↔ 数密度 {bn:.4f} 1/Å³"
            f"（按每原子平均质量 {m_avg:.3f} u 换算，相对偏差 "
            f"{abs(conv - bm) / bm * 100:.2f}%），口径与单位已写入结果")


def test_contact_modes_partition_pairs_exactly():
    """接触配对模式：inter / intra 必须**穷尽且互斥**，且 total = inter + intra。

    用"水×水"做验证最干净：4 点水模型每个原子有 3 个同分子邻居、且都在 cutoff 内，
    因此 intra 模式下「每个原子的平均接触数」必须**恰好等于 3.000**——这是一个
    可精确预期的物理量，而不是"看起来差不多"。
    """
    d = _ctx()
    mdt = d["mdt"]
    w = d["water"]
    if w is None or w.n_atoms == 0:
        return "跳过：体系中找不到水"
    sel = select_frames(mdt.times_ps, max_frames=2)
    out = {}
    for name in ("inter", "intra", "total"):
        r = ifc.analyze_contacts(mdt, w, w, sel, cutoff=5.0, mode=name)
        out[name] = (float(r.summary["平均接触对数"]),
                     float(r.summary["平均接触数（每个 A 组原子）"]),
                     float(r.summary["分子内配对占比"]))
        assert r.summary["配对模式"] == name, r.summary
    assert abs(out["inter"][2]) < 1e-12, out["inter"]
    assert abs(out["intra"][2] - 1.0) < 1e-12, out["intra"]
    recon = out["inter"][0] + out["intra"][0]
    assert abs(recon - out["total"][0]) / out["total"][0] < 1e-9, (recon, out["total"])
    assert abs(out["intra"][1] - 3.0) < 1e-9, (out["intra"][1], "分子内邻居数应为 3")
    return (f"inter {out['inter'][0]:,.0f} + intra {out['intra'][0]:,.0f} = "
            f"total {out['total'][0]:,.0f} 对/帧（精确相加）；"
            f"intra 模式每个原子平均接触数 = {out['intra'][1]:.3f}（4 点水模型应恰为 3）")


def test_diffusion_coefficient_reports_trustworthy_error_bar():
    """扩散系数必须带**标准误**，而且这个标准误要真的覆盖真值。

    合成布朗运动：每步每轴方差 σ²、步长 dt → ``D_true = σ²/(2·dt)``（三维下
    MSD = 6Dt）。断言：
    1. |D − D_true| ≤ 3×报出的标准误（误差棒必须"够用"）；
    2. 标准误来自**分块平均**（经验散布），不是只有拟合协方差；
    3. 只用 OLS 协方差会**低估**（这正是加误差棒的理由）；
    4. 相对标准误随分块数/轨迹长度是合理量级（0 < 相对标准误 < 100%）。
    """
    import MDAnalysis as mda

    n_f, P, dt, sigma = 400, 300, 1.0, 0.2
    rng = np.random.default_rng(4)
    steps = rng.normal(scale=sigma, size=(n_f, P, 3))
    xyz = np.cumsum(steps, axis=0).astype(np.float32)
    u = mda.Universe.empty(P, n_residues=P, atom_resindex=list(range(P)),
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * P)
    u.add_TopologyAttr("molnums", np.arange(P, dtype=int))
    u.trajectory = mda.coordinates.memory.MemoryReader(
        xyz, order="fac",
        dimensions=np.array([2000.0, 2000.0, 2000.0, 90.0, 90.0, 90.0],
                            dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n_f, dtype=float) * dt)
    res = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                          remove_drift=False, fit_fraction=(0.1, 0.9))

    d_true = sigma ** 2 / (2.0 * dt) * 1e-8      # Å²/ps → m²/s
    d = float(res.summary["X D (m²/s)"])
    se = float(res.summary["X D 标准误 (m²/s)"])
    rel = float(res.summary["X D 相对标准误"])
    src = str(res.summary["X 标准误来源"])
    assert np.isfinite(d) and d > 0, d
    assert np.isfinite(se) and se > 0, se
    assert "分块" in src, src
    z = abs(d - d_true) / se
    assert z <= 3.0, (d, d_true, se, z, "误差棒没覆盖真值")
    assert 0.0 < rel < 1.0, rel

    # OLS 协方差（残差独立假设）应明显小于分块经验散布 —— 这就是要加修正的理由
    fit_ols = dyn.diffusion_coefficient(
        np.arange(n_f, dtype=float) * dt,
        np.asarray(res.curves_of(0)[0].y, dtype=float),
        fit_fraction=(0.1, 0.9), dim=3, fit_method="ols", block_curves=None)
    se_ols = float(fit_ols["D 标准误 (m²/s)"])
    assert se_ols < se, (se_ols, se, "OLS 协方差应当低估")
    return (f"D = {d:.4e} ± {se:.1e} m²/s（真值 {d_true:.4e}，偏差 {z:.2f}σ）；"
            f"来源={src}，相对标准误 {rel * 100:.1f}%；"
            f"OLS 协方差只给 {se_ols:.1e}（低估 {se / se_ols:.1f} 倍）")


def test_contact_coordination_distribution_is_exact():
    """配位数分布：水（4 点模型）在 intra 模式下必须是**恰好在 3 上的 δ 分布**。

    每个水分子的原子恰有 3 个同分子邻居（OW/HW1/HW2/MW 两两相连），且都在 cutoff
    内，所以「瞬时接触数」只能取 3 → 平均值 3.000、标准差 0.000、众数 3。
    分子间模式则应是展宽的分布（标准差 > 0）。这把"分布"这一维信息锁死：
    只报平均值时，两种完全不同的配位环境（均一 vs 高度异质）看起来一样。
    """
    d = _ctx()
    mdt = d["mdt"]
    w = d["water"]
    if w is None or w.n_atoms == 0:
        return "跳过：体系中找不到水"
    sel = select_frames(mdt.times_ps, max_frames=2)
    r_in = ifc.analyze_contacts(mdt, w, w, sel, cutoff=5.0, mode="intra")
    r_bt = ifc.analyze_contacts(mdt, w, w, sel, cutoff=5.0, mode="inter")
    assert abs(float(r_in.summary["瞬时接触数 平均"]) - 3.0) < 1e-9, r_in.summary
    assert float(r_in.summary["瞬时接触数 标准差"]) < 1e-9, r_in.summary
    assert abs(float(r_in.summary["瞬时接触数 众数"]) - 3.0) < 1e-9, r_in.summary
    assert float(r_bt.summary["瞬时接触数 标准差"]) > 1.0, r_bt.summary
    return (f"分子内：分布是 δ(3) → 平均 {r_in.summary['瞬时接触数 平均']:.3f}、"
            f"标准差 {r_in.summary['瞬时接触数 标准差']:.2e}；"
            f"分子间：平均 {r_bt.summary['瞬时接触数 平均']:.1f}、"
            f"标准差 {r_bt.summary['瞬时接触数 标准差']:.1f}（明显展宽）")


def test_notes_are_scoped_to_their_curve():
    """说明信息按作用域归属：RDF 每对的说明必须标到对应曲线，全局说明不标。

    为什么要这个：说明区现在是**动态**的——只显示与当前图表可见曲线相关的说明
    （RDF 跑了 10 个配对、图例只勾 1 个 → 只显示那 1 个的说明）。前提是后端先把
    "这条说明属于哪条曲线"标出来，否则前端只能把 10 份说明全堆出来。
    """
    from webapp.backend.serialize import result_to_json

    d = _ctx()
    mdt = d["mdt"]
    w = d["water"]
    ow = w.select_atoms("name OW")
    if ow.n_atoms == 0:
        return "跳过：体系中找不到水的 OW 原子"
    sel = select_frames(mdt.times_ps, max_frames=3)
    # 两个组分 → 三条配对曲线，说明天然会混在一起
    res = ifc.analyze_rdf(mdt, {"W": w, "OW": ow}, sel, rmax=6.0, nbins=60, mode="total")
    scope = res.notes_with_scope()
    assert len(scope) == len(res.notes), (len(scope), len(res.notes))
    tags = {s["curve"] for s in scope if s.get("curve")}
    assert len(tags) >= 2, f"说明没有归属到多条曲线: {tags}"
    for t, s in zip(res.notes, scope):
        if s.get("curve"):
            assert t.startswith(s["curve"]), (t[:40], s["curve"])
    assert any(s.get("curve") is None for s in scope), "全局说明不该被标上曲线"
    j = result_to_json(res)
    assert len(j.get("note_meta", [])) == len(j["notes"]), "作用域没有随结果下发"
    n_tag = sum(1 for s in scope if s.get("curve"))
    return (f"{n_tag} 条说明归属到曲线 {sorted(tags)}，"
            f"{len(scope) - n_tag} 条为全局说明；作用域已随结果下发")


# ------------------------------------------------- 时间轴：快路径与副作用
class _FakeTS:
    """替身 TimeStep：只提供 ``time`` / ``frame``。"""

    def __init__(self, time_, frame):
        self.time = float(time_)
        self.frame = int(frame)


class _FakeTraj:
    """替身 trajectory：**记录被访问过哪些帧**，用来证明"没有全扫"。"""

    def __init__(self, times, dt):
        self._t = [float(x) for x in times]
        self.dt = float(dt)
        self.ts = _FakeTS(self._t[0], 0)
        self.reads = []

    def __getitem__(self, i):
        self.reads.append(int(i))
        self.ts = _FakeTS(self._t[int(i)], int(i))
        return self.ts

    def __len__(self):
        return len(self._t)


class _FakeUniverse:
    def __init__(self, times, dt):
        self.trajectory = _FakeTraj(times, dt)


def _fake_mdt(times, dt):
    mdt = MDTrajectory(topology=None, trajectory=None)
    mdt._universe = _FakeUniverse(times, dt)
    return mdt


def test_time_range_is_constant_cost_and_exact():
    """首末时刻必须 O(1) 拿到，且调用后当前帧要复原。

    XTC 的 reader 没有 ``.times``，取一次时刻要读一整帧坐标：
    46 体系（10001 帧 / 1.7 GB）逐帧要 28 s。早期 ``total_time_ps``
    每次都走这条路，"读取完成"后静默 50 s 才返回，就是要在这里挡住。
    """
    times = np.arange(0.0, 100010.0, 10.0)          # 10001 帧
    mdt = _fake_mdt(times, 10.0)
    assert mdt.n_frames == 10001, mdt.n_frames
    rng = mdt.time_range_ps()
    assert rng == (0.0, 100000.0), rng
    reads = list(mdt.universe.trajectory.reads)
    assert len(reads) <= 8, f"读了 {len(reads)} 帧（{reads[:12]}…），说明不是 O(1)"
    assert mdt.total_time_ps == 100000.0, mdt.total_time_ps
    # 副作用：调用前停在哪一帧，调用后还得在哪一帧
    m2 = _fake_mdt(times, 10.0)
    m2.universe.trajectory[123]
    m2.time_range_ps()
    assert m2.universe.trajectory.ts.frame == 123, m2.universe.trajectory.ts.frame
    return (f"首末时刻只读 {len(reads)} 帧即得 ({rng[0]:g}, {rng[1]:g}) ps，"
            f"调用后当前帧复原")


def test_times_fast_path_equals_full_scan():
    """均匀时间轴走解析构造，结果必须与逐帧读出的**完全一致**，并命中缓存。"""
    times = np.arange(0.0, 101.0, 10.0)
    mdt = _fake_mdt(times, 10.0)
    fast = np.array(mdt.times_ps, dtype=float)
    assert np.array_equal(fast, times), (fast, times)
    assert len(mdt.universe.trajectory.reads) <= 10, mdt.universe.trajectory.reads
    n_before = len(mdt.universe.trajectory.reads)
    assert mdt.times_ps is mdt.times_ps
    assert len(mdt.universe.trajectory.reads) == n_before, "第二次访问不该再读帧"
    return f"{fast.size} 帧解析构造与逐帧一致，二次访问零读取"


def test_times_rejects_nonuniform_grid():
    """非等差时间轴必须挡下并回退精确逐帧，否则帧选择会整体错位。"""
    bad = [0.0, 10.0, 20.0, 30.0, 100.0]            # 末帧与 dt 不自洽
    m1 = _fake_mdt(bad, 10.0)
    assert np.array_equal(np.array(m1.times_ps, dtype=float), np.array(bad))
    assert m1._uniform_times_ps(0.0, 100.0) is None

    jump = [0.0, 10.0, 55.0, 30.0, 40.0]            # 首末自洽、中间跳变
    m2 = _fake_mdt(jump, 10.0)
    assert m2._uniform_times_ps(0.0, 40.0) is None, "中间跳变被误采信"
    assert np.array_equal(np.array(m2.times_ps, dtype=float), np.array(jump))

    ok = [0.0, 10.0, 20.0, 30.0, 40.0]              # 真正均匀 -> 采信
    m3 = _fake_mdt(ok, 10.0)
    got = m3._uniform_times_ps(0.0, 40.0)
    assert got is not None and np.array_equal(got, np.array(ok))

    m4 = _fake_mdt([7.5], 10.0)                     # 单帧
    assert list(m4.times_ps) == [7.5]
    return "末帧不自洽/中间跳变被挡下并回退逐帧，均匀与单帧正确"


def test_systeminfo_time_range_matches_full_scan():
    """体系信息的首末时刻来自 O(1) 快路径，且与逐帧读出的一致。"""
    mdt = _load()["mdt"]
    rng = mdt.time_range_ps()
    if rng is None:
        return "跳过：轨迹没有时间列"
    exact = np.asarray([ts.time for ts in mdt.universe.trajectory], dtype=float)
    if exact.size == 0:
        return "跳过：轨迹没有时间列"
    dev = float(np.max(np.abs(np.array([rng[0], rng[1]], dtype=float)
                              - np.array([exact[0], exact[-1]], dtype=float))))
    # XTC 的时间按 float32 存，10⁵ ps 量级的量化台阶约 0.008 ps
    assert dev < 1e-3, f"首末时刻偏差 {dev:.3e} ps"
    info = describe_system(mdt)
    assert info.first_time_ps == rng[0] and info.last_time_ps == rng[1], (
        info.first_time_ps, info.last_time_ps, rng
    )
    assert abs(mdt.total_time_ps - (exact[-1] - exact[0])) < 1e-3
    return (f"{mdt.n_frames} 帧：快路径首末与逐帧一致（偏差 {dev:.1e} ps），"
            f"总时长 {mdt.total_time_ps:.1f} ps")


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
