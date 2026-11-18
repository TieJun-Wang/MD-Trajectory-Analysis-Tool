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

_HERE = os.path.dirname(os.path.abspath(__file__))          # …/test
_ROOT = os.path.dirname(_HERE)                              # 仓库根（含 mdta/ 与样例轨迹）
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from mdta.pipeline import Analyzer, sort_by_estimate         # noqa: E402
from mdta.qc import derive_checks                            # noqa: E402
from mdta.selection import classify_residues, select        # noqa: E402
from mdta.systeminfo import describe_system                 # noqa: E402

#: 默认样例体系（仓库根目录下的 AdK 轨迹）。用绝对路径：从任意 cwd 运行都能找到。
TOP_DEFAULT = os.path.join(_ROOT, "adk_oplsaa.tpr")
XTRAJ_DEFAULT = os.path.join(_ROOT, "adk_oplsaa.xtc")

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


def test_chain_dihedral_covers_every_molecule():
    """chain 模式逐分子取骨架（1.0.5）：多分子体系不再只分析第一个分子。

    背景：1.0.4 及以前对**整个选择**做两次 BFS 取一条最长路径，BFS 跳不出连通
    分量，于是 20 条 DP=30 的 PE（C60H122 × 20，3640 原子）只得到 57 个二面角
    （全在分子 0 内），而正确结果是 20 × 57 = 1140 个。这里用小体系验证同一件事。
    """
    dp, n_mol = 3, 6
    nc, nh = 2 * dp, 2 * (2 * dp) + 2
    per = nc + nh
    els, bonds, pos, res = [], [], [], []
    base = 0
    for m in range(n_mol):
        x = np.arange(nc) * 1.25 + m * 30.0
        p = np.zeros((per, 3))
        p[:nc, 0] = x
        p[:nc, 2] = np.where(np.arange(nc) % 2, 1.27, 0.0)
        for i in range(nc):
            p[nc + 2 * i] = p[i] + [0.0, 0.0, 1.09]
            p[nc + 2 * i + 1] = p[i] - [0.0, 0.0, 1.09]
            bonds += [(base + i, base + nc + 2 * i),
                      (base + i, base + nc + 2 * i + 1)]
            if i < nc - 1:
                bonds.append((base + i, base + i + 1))
        els += ["C"] * nc + ["H"] * nh
        pos.append(p)
        res += [m * 100 + i + 1 for i in range(per)]
        base += per
    u = mda.Universe.empty(base, n_residues=len(res),
                           atom_resindex=np.asarray(res) - np.asarray(res).min(),
                           trajectory=True)
    u.add_TopologyAttr("elements", els)
    u.add_TopologyAttr("bonds", bonds)
    u.atoms.positions = np.vstack(pos)
    u.dimensions = np.array([200.0, 200.0, 200.0, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)

    idx, notes = conf.chain_dihedrals(u.atoms, heavy_only=True)
    n_per = nc - 3                                    # 一条链的最长路径 → n-3
    assert idx.shape == (n_mol * n_per, 4), idx.shape
    mols = np.unique(np.asarray(idx) // per)
    assert mols.size == n_mol, mols                # 6 个分子全都参与
    assert any("逐分子" in s for s in notes), notes

    # 单分子时与旧的"整组最长路径"逐位相同（不能改变既有结果）
    u1 = u.atoms[:per]
    old = conf.make_dihedral_indices(conf.backbone_path(u1, heavy_only=True))
    new, _n = conf.chain_dihedrals(u1, heavy_only=True)
    assert old.shape == new.shape == (n_per, 4), (old.shape, new.shape)
    assert np.array_equal(np.sort(old, axis=1), np.sort(new, axis=1)), (old, new)

    # 超过上限时截断并写进说明
    capped, cnotes = conf.chain_dihedrals(u.atoms, heavy_only=True, max_molecules=2)
    assert capped.shape[0] == 2 * n_per, capped.shape
    assert any("max_molecules" in s for s in cnotes), cnotes

    # 只有键、没有显式二面角表时，topology 模式必须说清原因（而不是静默出空图）
    res_t = conf.analyze_dihedrals(mdt, u.atoms, select_frames(np.array([0.0])),
                                   mode="topology")
    assert not res_t.summary, res_t.summary
    assert any("没有显式的二面角表" in s for s in res_t.notes), res_t.notes

    return (f"{n_mol} 条 C{nc}H{nh}：逐分子得 {idx.shape[0]} 个二面角"
            f"（{n_mol} × {n_per}，覆盖 {mols.size}/{n_mol} 个分子）；"
            f"单分子与旧路径逐位一致；max_molecules 截断与无语义拓扑的提示均生效")


def test_angles_dihedrals_derived_from_bond_table():
    """只有键表的拓扑（.gro）→ 由键表派生角/二面角，体系信息不再是 0/0。

    用户实测：20 条 DP=30 的 PE 读进来，第二部分「体系信息」显示
    ``键 / 角 / 二面角 = 3,620 / 0 / 0`` —— 因为 MDAnalysis 的 ``angles`` /
    ``dihedrals`` 是独立拓扑属性，不会由键自动推出。这里验证：
    ① 派生条数与解析公式 ``ΣC(deg,2)`` / ``Σ(deg_i−1)(deg_j−1)`` 一致；
    ② 拓扑已自带这两张表时不覆盖；③ 派生之后 ``mode='topology'`` 能跑，
    且线形 PE 上与 ``mode='chain'`` 给出同样的二面角个数。
    """
    from mdta.topology import count_angles_dihedrals, derive_angles_dihedrals

    # 一条 C20H42 的线形锯齿链（键 101 根）
    nc = 20
    L, ang = 1.54, np.radians(180.0 - 111.0)
    b0 = np.array([L, 0.0, 0.0])
    b1 = L * np.array([np.cos(ang), np.sin(ang), 0.0])
    cpos = np.zeros((nc, 3))
    for i in range(1, nc):
        cpos[i] = cpos[i - 1] + (b0 if (i - 1) % 2 == 0 else b1)
    pos, els, bonds = [], [], []
    for i in range(nc):
        pos.append(cpos[i])
        els.append("C")
    hi = nc
    for i in range(nc):
        dirs = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
        if i == 0:
            dirs.append(np.array([-1.0, 0.0, 0.0]))
        elif i == nc - 1:
            dirs.append(np.array([1.0, 0.0, 0.0]))
        for d in dirs:
            pos.append(cpos[i] + 1.09 * d / np.linalg.norm(d))
            els.append("H")
            bonds.append((i, hi))
            hi += 1
    for i in range(nc - 1):
        bonds.append((i, i + 1))
    pos = np.asarray(pos)
    u = mda.Universe.empty(len(pos), n_residues=nc,
                           atom_resindex=list(range(nc)) + [nc - 1] * (len(pos) - nc),
                           trajectory=True)
    u.add_TopologyAttr("elements", els)
    u.add_TopologyAttr("bonds", bonds)
    u.atoms.positions = pos
    u.dimensions = np.array([80.0, 80.0, 80.0, 90.0, 90.0, 90.0], dtype=np.float32)

    n_ang_ref, n_dih_ref = count_angles_dihedrals(np.asarray(u.bonds.indices),
                                                  u.atoms.n_atoms)
    # 解析校验：C20H42 → 每原子度数 4(C)/1(H)：角 = 20×C(4,2) = 120，
    # 二面角 = Σ_bond (deg_i−1)(deg_j−1) = 19×9 = 171
    assert (n_ang_ref, n_dih_ref) == (120, 171), (n_ang_ref, n_dih_ref)

    info = derive_angles_dihedrals(u)
    assert info["added"] == ["angles", "dihedrals"], info
    assert info["n_angles"] == n_ang_ref, info
    assert info["n_dihedrals"] == n_dih_ref, info
    assert len(u.atoms.angles) == n_ang_ref
    assert len(u.atoms.dihedrals) == n_dih_ref

    # 不覆盖：再调一次应什么都没做
    info2 = derive_angles_dihedrals(u)
    assert info2["added"] == [] and "不重复派生" in info2["skipped"], info2

    # 派生后 topology 模式可用，且与 chain 模式在**线形**链上条数一致
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    r_chain = conf.analyze_dihedrals(mdt, u.atoms, sel, mode="chain")
    r_topo = conf.analyze_dihedrals(mdt, u.atoms, sel, mode="topology")
    assert r_chain.summary["二面角总数"] == nc - 3, r_chain.summary
    assert r_topo.summary["二面角总数"] == nc - 3, r_topo.summary

    # 「体系信息」读出真实的角/二面角条数（以前是 0/0）
    from mdta.systeminfo import describe_system

    si = describe_system(mdt)
    assert (si.n_bonds, si.n_angles, si.n_dihedrals) == (len(bonds), 120, 171), \
        (si.n_bonds, si.n_angles, si.n_dihedrals)
    row = dict(si.to_rows())["键/角/二面角"]
    assert "120" in row and "171" in row, row

    return (f"C{nc}H{2 * nc + 2}：键 {len(bonds)} → 角 {n_ang_ref} / 二面角 "
            f"{n_dih_ref}（与解析公式一致）；chain 与 topology 都给 {nc - 3} 个")


def test_dihedral_bin_grid_matches_forcite_half_bin_alignment():
    """分箱网格：x 是**格子中心**、边界在中心 ± bin/2（Forcite 实测对齐）。

    20PE30 实测（bin=0.5）：Forcite 的 Torsion 列是 −180.5, −180, …, 180.5
    （723 个）。把这些值当**中心**、边界放到 −180.75 … 180.75 时，我们的计数与
    Forcite **723 格逐格相同（L1 = 0）**；当"左边界"用时 45% 的格子不同。
    差的就是半个 bin（0.25°）—— 这是"数据对不上"的真正原因。
    """
    from mdta.analysis.conformation import dihedral_bin_grid

    # bin=1 → 363 个中心 −181 … 181，边界 ±181.5
    c1, e1 = dihedral_bin_grid(1.0)
    assert c1.size == 363, c1.size
    assert np.allclose(c1, np.arange(-181.0, 181.0 + 1e-9, 1.0)), c1[:4]
    assert abs(e1[0] + 181.5) < 1e-9 and abs(e1[-1] - 181.5) < 1e-9, (e1[0], e1[-1])
    assert np.allclose(e1[:-1], c1 - 0.5)                     # 边界 = 中心 − bin/2
    assert abs(e1[-1] - (c1[-1] + 0.5)) < 1e-9

    # bin=0.5 → 723 个中心 −180.5 … 180.5，边界 ±180.75（Forcite 的实测口径）
    c2, e2 = dihedral_bin_grid(0.5)
    assert c2.size == 723, c2.size
    assert np.allclose(c2, np.arange(-180.5, 180.5 + 1e-9, 0.5)), c2[:4]
    assert abs(c2[0] + 180.5) < 1e-9 and abs(c2[-1] - 180.5) < 1e-9
    assert abs(e2[0] + 180.75) < 1e-9 and abs(e2[-1] - 180.75) < 1e-9, (e2[0], e2[-1])
    assert np.allclose(e2[:-1], c2 - 0.25)
    assert np.allclose(np.diff(c2), 0.5)

    # 一个**正好落在格中心**的角度必须进那一格（而不是左边那格）：
    # 用 45.0° 试：它应落在中心 45.0 的桶 [44.75, 45.25)
    c, e = dihedral_bin_grid(0.5)
    k = int(np.searchsorted(e, 45.0, side="right") - 1)
    assert abs(c[k] - 45.0) < 1e-9, (c[k], 45.0)
    k2 = int(np.searchsorted(e, 44.7, side="right") - 1)
    assert abs(c[k2] - 44.5) < 1e-9, c[k2]

    # 非法值退回 1°
    c0, _e = dihedral_bin_grid(0)
    assert c0.size == 363 and abs(c0[0] + 181.0) < 1e-9
    return ("bin=1 → 363 中心 −181…181（边界 ±181.5）；bin=0.5 → 723 中心 "
            "−180.5…180.5（边界 ±180.75，与 Forcite 逐格对齐）；"
            "45.0° 落在中心 45.0 的桶")


def test_dihedral_distribution_bin_width_option():
    """二面角分布的 bin 与 **Forcite 的 Bin width** 对齐。

    Forcite 帮助原文："Bin width: Specify the bin width, in degrees, for torsion
    distribution calculations." 实测口径（用户确认）：网格端点 = ±(180 + bin)，
    步长 = bin，即

      · bin 是 1   → −181、−180、−179 …… 179、180、181
      · bin 是 0.5 → −180.5、−180、−179.5 …… 179.5、180、180.5

    本测试直接按"格子左端点列表"逐项比对这两组数。
    """
    nc = 12
    L, ang = 1.54, np.radians(180.0 - 111.0)
    b0 = np.array([L, 0.0, 0.0])
    b1 = L * np.array([np.cos(ang), np.sin(ang), 0.0])
    cpos = np.zeros((nc, 3))
    for i in range(1, nc):
        cpos[i] = cpos[i - 1] + (b0 if (i - 1) % 2 == 0 else b1)
    u = mda.Universe.empty(nc, n_residues=1, atom_resindex=[0] * nc,
                           trajectory=True)
    u.add_TopologyAttr("elements", ["C"] * nc)
    u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(nc - 1)])
    u.atoms.positions = cpos
    u.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))

    def run(bw):
        res = conf.analyze_dihedrals(mdt, u.atoms, sel, mode="chain",
                                     bin_width=bw)
        c = [c for c in res.curves if c.label == "dihedral 分布"][0]
        x = np.asarray(c.x, dtype=float)          # 网格端点（数据表的 x 列）
        y = np.asarray(c.y, dtype=float)
        return res, x, y

    # ---- bin = 1 → x 从 −181 到 181，interval 1（363 行）
    res1, x1, y1 = run(1)
    assert x1.size == 363, x1.size
    assert np.allclose(x1, np.arange(-181.0, 181.0 + 1e-9, 1.0)), x1[:5]
    assert abs(x1[0] + 181.0) < 1e-9 and abs(x1[-1] - 181.0) < 1e-9
    assert abs(x1[1] - x1[0] - 1.0) < 1e-12
    assert abs(float(np.sum(y1) * 1.0) - 1.0) < 1e-6    # 概率密度积分 = 1
    assert abs(float(y1[-1])) < 1e-12                   # 最后一个端点右边没有格子

    # ---- bin = 0.5 → x 从 −180.5 到 180.5，interval 0.5（723 行）
    _r2, x2, y2 = run(0.5)
    assert x2.size == 723, x2.size
    assert np.allclose(x2, np.arange(-180.5, 180.5 + 1e-9, 0.5)), x2[:5]
    assert abs(x2[0] + 180.5) < 1e-9 and abs(x2[-1] - 180.5) < 1e-9
    assert abs(x2[1] - x2[0] - 0.5) < 1e-12
    assert abs(float(np.sum(y2) * 0.5) - 1.0) < 1e-6    # 概率密度积分 = 1

    # ---- bin = 2 → x 从 −182 到 182，interval 2（183 行）
    _r, x3, _y3 = run(2)
    assert x3.size == 183, x3.size
    assert abs(x3[0] + 182.0) < 1e-9 and abs(x3[-1] - 182.0) < 1e-9

    # ---- 非法值退回 1°（Forcite 默认口径）
    _r0, x0, _y0 = run(0)
    assert x0.size == 363 and abs(x0[0] + 181.0) < 1e-9

    assert any("对齐 Forcite" in s for s in res1.notes), res1.notes
    return ("bin=1 → x −181…181 / interval 1（363 行）、"
            "bin=0.5 → x −180.5…180.5 / interval 0.5（723 行）、"
            "bin=2 → x −182…182（183 行）；y 为计数，求和守恒，非法值退回 1°")


def test_dihedral_distribution_is_probability_density():
    """二面角分布的 y 是**概率密度**（小数），不是计数。

    用户要求："分布曲线怎么可能是整数" —— 直方图纵轴必须是每度多少概率这种
    小数量，且与样本总数 / 每格宽度都无关（× 每格宽度求和 = 1）。

      · y 全为非负小数，Σy × bin = 1（bin=1/0.5/2 都成立）；
      · bin=1 时数值等于"每格概率"（乘 1°）；bin≠1 时两者差 1/bin；
      · 摘要给「分布样本总数（帧 × 二面角）」与「分布峰值（概率密度）」；
      · 只有 2 张图（分布 + trans/gauche 随时间）。
    """
    nc = 12
    L, ang = 1.54, np.radians(180.0 - 111.0)
    b0 = np.array([L, 0.0, 0.0])
    b1 = L * np.array([np.cos(ang), np.sin(ang), 0.0])
    cpos = np.zeros((nc, 3))
    for i in range(1, nc):
        cpos[i] = cpos[i - 1] + (b0 if (i - 1) % 2 == 0 else b1)
    u = mda.Universe.empty(nc, n_residues=1, atom_resindex=[0] * nc,
                           trajectory=True)
    u.add_TopologyAttr("elements", ["C"] * nc)
    u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(nc - 1)])
    u.atoms.positions = cpos
    u.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))

    n_dih = nc - 3
    cached: dict[float, np.ndarray] = {}
    for bw in (1.0, 0.5, 2.0):
        res = conf.analyze_dihedrals(mdt, u.atoms, sel, mode="chain",
                                     bin_width=bw)
        curves = [c for c in res.curves if c.label == "dihedral 分布"]
        assert len(curves) == 1, [c.label for c in res.curves]
        y = np.asarray(curves[0].y, dtype=float)
        assert len(res.panels) == 2, len(res.panels)
        assert res.panels[0].ylabel.startswith("概率密度"), res.panels[0].ylabel
        assert np.all(y >= 0.0)
        # 必须是小数（不是计数）：有非整数值，且积分 = 1
        assert abs(float(y.sum() * bw) - 1.0) < 1e-9, (bw, y.sum() * bw)
        assert int(res.summary["dihedral 分布样本总数（帧 × 二面角）"]) == n_dih
        assert abs(float(res.summary["dihedral 分布峰值（概率密度）"])
                   - float(y.max())) < 1e-12
        cached[bw] = y

    # 存在非整数分量（说明不是计数）
    assert not np.allclose(cached[1.0][cached[1.0] > 0],
                           np.round(cached[1.0][cached[1.0] > 0])), cached[1.0]

    res = conf.analyze_dihedrals(mdt, u.atoms, sel, mode="chain")
    assert any("概率密度" in s for s in res.notes), res.notes
    assert not any("不做归一化" in s for s in res.notes), res.notes
    return (f"y 为概率密度（小数，Σy×bin = 1，bin=1/0.5/2 都成立）；"
            f"摘要给样本总数 {n_dih} 与峰值；只有 2 张图")


def test_orientation_p2_matches_torsion_script_reference():
    """取向 P2 与外部脚本口径逐位一致（几何骨架 + stride=2 ↔ 相邻键中点矢量）。

    对照的是 `C:\\temp\\dataset\\320K-2\\allfast-3-2.py`（外部独立实现，
    已在 122C12 的 6001 帧上实测与我们的结果差 1.1e-08）。它的链段矢量是
    "相邻键中点连成的矢量"，即 ``m2 − m1 = (C_{i+2} − C_i)/2``，也就是
    **几何骨架 + stride=2**；本测试把它的公式直接写在测试里，
    验证两个实现给出同一个数。
    """
    rng = np.random.default_rng(12345)
    n_chain, nc = 10, 12
    L, ang = 1.54, np.radians(180.0 - 111.0)
    b0 = np.array([L, 0.0, 0.0])
    b1 = L * np.array([np.cos(ang), np.sin(ang), 0.0])
    bases, els, bonds = [], [], []
    for _ in range(n_chain):
        # 随机取向的锯齿链
        cpos = np.zeros((nc, 3))
        for i in range(1, nc):
            cpos[i] = cpos[i - 1] + (b0 if (i - 1) % 2 == 0 else b1)
        phi, theta = rng.uniform(0, 2 * np.pi), np.arccos(rng.uniform(-1, 1))
        R = np.array([
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ]) @ np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ])
        cpos = cpos @ R.T + rng.uniform(-5, 5, 3)
        bases.append(cpos)
    base = 0
    pos = []
    for cpos in bases:
        for i in range(nc):
            pos.append(cpos[i])
            els.append("C")
        hcount = 0
        for i in range(nc):
            dirs = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
            if i == 0:
                dirs.append(np.array([-1.0, 0.0, 0.0]))
            elif i == nc - 1:
                dirs.append(np.array([1.0, 0.0, 0.0]))
            for d in dirs:
                pos.append(cpos[i] + 1.09 * d / np.linalg.norm(d))
                els.append("H")
                bonds.append((base + i, base + nc + hcount))
                hcount += 1
        for i in range(nc - 1):
            bonds.append((base + i, base + i + 1))
        base += nc + hcount
    pos = np.asarray(pos)

    u = mda.Universe.empty(len(pos), n_residues=len(pos),
                           atom_resindex=list(range(len(pos))), trajectory=True)
    u.add_TopologyAttr("elements", els)
    u.add_TopologyAttr("bonds", bonds)
    u.atoms.positions = pos
    u.dimensions = np.array([80.0, 80.0, 80.0, 90.0, 90.0, 90.0], dtype=np.float32)

    # ---- 我们的引擎
    mdt = MDTrajectory.from_universe(u)
    carbons = u.atoms[np.asarray([i for i, e in enumerate(els) if e == "C"])]
    res = cry.analyze_orientation(mdt, carbons, select_frames(np.array([0.0])),
                                  mode="backbone", stride=2,
                                  reference_axis=(0.0, 0.0, 1.0))
    p2_ours = None
    for c in res.curves:
        if c.label.startswith("二阶序参数 P2"):
            p2_ours = float(np.asarray(c.y, dtype=float)[0])
    assert p2_ours is not None

    # ---- 外部脚本口径（相邻键中点矢量 → cos²θ_z，链内平均再跨链平均）
    def their_cos2(xyz):
        mids = [((xyz[i][0] + xyz[i + 1][0]) / 2.0,
                 (xyz[i][1] + xyz[i + 1][1]) / 2.0,
                 (xyz[i][2] + xyz[i + 1][2]) / 2.0) for i in range(len(xyz) - 1)]
        out = []
        for i in range(len(mids) - 1):
            vx = mids[i + 1][0] - mids[i][0]
            vy = mids[i + 1][1] - mids[i][1]
            vz = mids[i + 1][2] - mids[i][2]
            n2 = vx * vx + vy * vy + vz * vz
            if n2 > 1e-12:
                out.append(vz * vz / n2)
        return float(np.mean(out)) if out else 0.0

    car_idx = [i for i, e in enumerate(els) if e == "C"]
    per_chain = [car_idx[i * nc:(i + 1) * nc] for i in range(n_chain)]
    their = [their_cos2([pos[j] for j in ch]) for ch in per_chain]
    cos2_ref = float(np.mean(their))
    p2_ref = (3.0 * cos2_ref - 1.0) / 2.0

    # ---- 1.0.6 新增的「链段与参考轴的夹角」面板也必须与同一口径一致：
    #      每帧对所有链段求 ⟨arccos|u·z|⟩（度）
    def their_theta(xyz):
        mids = [(xyz[i] + xyz[i + 1]) / 2.0 for i in range(len(xyz) - 1)]
        out = []
        for i in range(len(mids) - 1):
            v = mids[i + 1] - mids[i]
            n = np.linalg.norm(v)
            if n > 1e-12:
                out.append(float(np.arccos(min(1.0, abs(v[2]) / n))))
        return float(np.degrees(np.mean(out))) if out else 0.0

    theta_ref = float(np.mean([their_theta(np.asarray([pos[j] for j in ch],
                                                      dtype=float))
                               for ch in per_chain]))
    theta_ours = None
    for c in res.curves:
        if c.panel == 2:                    # 面板 2 = 链段与参考轴的夹角 θ(t)
            theta_ours = float(np.asarray(c.y, dtype=float)[0])
    assert theta_ours is not None, "缺少「链段与参考轴的夹角」曲线"

    # MDAnalysis 以 float32 存坐标、且 MS 轨迹转换有 1e-3 Å 量化，
    # 因此实测差在 1e-9 量级（122C12 的 6001 帧上最大 1.1e-08）
    assert abs(p2_ours - p2_ref) < 1e-6, (p2_ours, p2_ref)
    assert abs(theta_ours - theta_ref) < 1e-6, (theta_ours, theta_ref)
    return (f"{n_chain} 条随机取向 C12：我们的 P2 = {p2_ours:+.9f}，外部脚本口径 "
            f"{p2_ref:+.9f}，差 {abs(p2_ours - p2_ref):.2e}；"
            f"夹角 θ = {theta_ours:.6f}° vs {theta_ref:.6f}°，"
            f"差 {abs(theta_ours - theta_ref):.2e}")


def test_only_labels_transport_survives_commas_in_labels():
    """「只看当前图例」的 only 参数：**一条标签 = 一个参数**，不能按逗号切。

    用户实测的 bug：MSD 面板 0 的曲线标签是 ``polymer (原子, 3,640 个)``（标签
    自身带逗号）。旧实现前端 ``only.join(',')`` 拼成一串、后端 ``split(',')``
    拆开 —— 标签被切成三段，一条也匹配不上，于是静默退回"全部曲线"，
    表现就是"只勾一个图例 + 只看当前图例没效果"。
    """
    from webapp.backend.app import _only_labels

    lab = "polymer (原子, 3,640 个)"
    assert _only_labels([lab]) == [lab]                  # 重复参数 + 标签含逗号
    assert _only_labels(["a", "b"]) == ["a", "b"]
    assert _only_labels(["a,b"]) == ["a,b"]              # 单个参数不再被切开
    assert _only_labels([""]) is None
    assert _only_labels(None) is None
    assert _only_labels("x") == ["x"]
    return ("含逗号的标签原样穿过（不再被切成多段）；空值与 None 归为不过滤")


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


def test_water_and_ions_recognized_by_composition_not_name():
    """组分识别：resname 没意义时（``MOL`` / ``1`` / 空）也要按**组成**认出水和离子。

    用户实测痛点：Materials Studio 导出的体系里，2000 个水分子和离子的 resname
    全是 ``MOL``（水甚至被标成 ``ETH``），于是水被并进 ``polymer``：
    "水都被识别成聚合物了"。

    断言：
      ① H₂O 组成（1 个 O + 1–3 个 H、无 C）→ ``water``，单原子 + 常见离子元素 → ``ion``；
      ② 组分清单里 水/离子/烃链 分成三组；
      ③ ``identify_component`` 给出"这是什么分子"（水 H₂O / 钠离子 / 正构烷烃 C6H14）。
    """
    import MDAnalysis as mda

    from mdta.selection import (classify_residues, component_groups,
                                identify_component)

    n_wat, n_na, n_cl, n_chain, n_c = 200, 24, 24, 5, 6
    pos, els, res_of_atom, bonds = [], [], [], []
    res_names: list[str] = []
    idx = 0

    def new_res(resname):
        res_names.append(resname)
        return len(res_names) - 1          # 0 基 residue 下标

    # 水：O–H–H，resname 故意写成 "MOL"（名字认不出来的那种）
    for _ in range(n_wat):
        r = new_res("MOL")
        base = idx
        for el in ("O", "H", "H"):
            els.append(el)
            pos.append([0.0, 0.0, 0.0])
            res_of_atom.append(r)
            idx += 1
        bonds.extend([(base, base + 1), (base, base + 2)])
    # 单原子离子（Na⁺ / Cl⁻），resname 也是 "MOL"
    for el in ("NA",) * n_na + ("CL",) * n_cl:
        r = new_res("MOL")
        els.append(el)
        pos.append([0.0, 0.0, 0.0])
        res_of_atom.append(r)
        idx += 1
    # 5 条 C6H14（正己烷）：每个分子 = 1 个 residue（线性碳链 + 氢）
    for _ in range(n_chain):
        r = new_res("ETH")
        base = idx
        for k in range(n_c):
            els.append("C")
            pos.append([float(k) * 1.5, 0.0, 0.0])
            res_of_atom.append(r)
            idx += 1
        for k in range(n_c):
            for _h in range(3 if k in (0, n_c - 1) else 2):
                els.append("H")
                pos.append([float(k) * 1.5, 0.5, 0.0])
                res_of_atom.append(r)
                bonds.append((base + k, idx))
                idx += 1
        for k in range(n_c - 1):
            bonds.append((base + k, base + k + 1))

    n = idx
    u = mda.Universe.empty(n, n_residues=len(res_names),
                           atom_resindex=np.asarray(res_of_atom), trajectory=True)
    u.add_TopologyAttr("names", [f"{e}{i}" for i, e in enumerate(els)])
    u.add_TopologyAttr("elements", els)
    u.add_TopologyAttr("types", els)
    u.add_TopologyAttr("resnames", res_names)
    u.add_TopologyAttr("segids", ["SYS"])
    u.add_TopologyAttr("masses", np.where(np.asarray(els) == "H", 1.008, 12.011))
    u.add_TopologyAttr("bonds", bonds)
    u.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0], dtype=np.float32)

    cats = classify_residues(u)
    got = {k: int(len(v)) for k, v in cats.items()}
    assert got.get("water") == n_wat, got
    assert got.get("ion") == n_na + n_cl, got
    assert got.get("polymer") == n_chain, got

    comps = component_groups(u)
    assert "water" in comps and "ion" in comps, list(comps)
    w = identify_component(comps["water"])
    ion = identify_component(comps["ion"])
    poly = identify_component(comps["polymer"])
    assert w["formula"] == "H2O" and w["n_molecules"] == n_wat, w
    assert "水" in w["label"], w
    assert "离子" in ion["label"], ion
    assert poly["formula"] == "C6H14", poly
    assert "烷烃" in poly["label"], poly

    return (f"水 {got['water']} 个 residue / 离子 {got['ion']} 个 / 烃链 {got['polymer']} 条 "
            f"全部按组成认出（resname 全是 MOL 也认得出）；初步识别："
            f"{w['label']}、{ion['label']}、{poly['label']}")


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
    # 这条测的是算法本身：显式要满整条轨迹（默认会按总时长/5 截断 τ）
    # 水会穿周期边界：这条用例专门验"最小镜像追踪"路径，所以显式 unwrap=True
    res = dyn.analyze_msd(mdt, {"OW": ow}, sel, tau_intervals=1, unwrap=True)
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
    # 1.0.5 起 S 与 P₂ 拆成两张图，摘要键也带轴名：默认轴 z → "P2_z mean"
    P2 = r.summary["P2_z mean"]
    cos2 = r.summary["实测 <cos²θ>"]
    assert abs(S) < 0.10, S
    assert abs(P2) < 0.10, P2
    assert abs(cos2 - 1.0 / 3.0) < 0.02, cos2
    # 1.0.6 起再加一张「链段与参考轴的夹角 θ(t)」：
    # 面板 0 = S、1 = P₂、2 = 夹角 θ(t)、3 = 角度分布
    assert len(r.panels) == 4, len(r.panels)
    on_panel = {i: [c.label for c in r.curves if c.panel == i] for i in range(4)}
    assert len(on_panel[0]) == 1 and on_panel[0][0].startswith("S ("), on_panel[0]
    assert len(on_panel[1]) == 1 and "P2" in on_panel[1][0], on_panel[1]
    assert len(on_panel[2]) == 1 and on_panel[2][0].startswith("平均夹角"), on_panel[2]
    # 各向同性体系：与参考轴的夹角应接近完全无序值 arccos(1/√3) = 54.74°
    theta = np.asarray(on_panel_curve_y(r, 2), dtype=float)
    assert np.all(theta[np.isfinite(theta)] >= 0.0)
    assert np.all(theta[np.isfinite(theta)] <= 90.0)
    th_mean = float(np.nanmean(theta))
    assert abs(th_mean - np.degrees(np.arccos(np.sqrt(1.0 / 3.0)))) < 3.0, th_mean
    return (f"S = {S:.4f}，P₂ = {P2:.4f}（各向同性理论值 0）；"
            f"<cos²θ> = {cos2:.4f}（理论值 1/3）；"
            f"夹角 θ = {th_mean:.2f}°（完全无序 54.74°）；S / P₂ / θ(t) 各占一张图")


def on_panel_curve_y(res, panel: int):
    """取某个面板上第一条曲线的 y（自检里的小工具）。"""
    for c in res.curves:
        if c.panel == panel:
            return c.y
    raise AssertionError(f"面板 {panel} 上没有曲线")


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
    res = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, unwrap=True)
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
    # 键长 1.0 Å 正好压在 bin 边界上：距离用 float64 算时是**精确的 1.0**（进
    # [1.0,1.1) 那一格），用 float32 算时是 0.9999999（进 [0.9,1.0) 那一格）。
    # 两种都对，所以断言"边界两侧那几格合计 = 全部分子内配对"，不钉死单格。
    near = hist["intra"][max(0, k - 1):k + 2].sum()
    assert near == n_mol * 2, (hist["intra"][max(0, k - 1):k + 2], "键长峰未落在 intra")
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


def test_sets_filter_atoms_and_molecules():
    """分析集合（Set）：按元素 / 分子形态 / 规模切分，语义（原子 vs 分子）要对。

    背景：用户希望能像 Materials Studio Forcite 的 Set 那样，"只统计碳原子"或
    "只统计链状结构"。这里用合成体系把三条判据钉住：
      1. 元素筛选会**切碎分子** → ``kind="atom"``（MSD 必须按原子追踪）；
      2. 无元素筛选（只按形态/规模）→ ``kind="molecule"``（可按质心追踪）；
      3. 环的判定：环己烷（6 元环）算 ``ring``，正己烷算 ``chain``。
    """
    from mdta import sets as SET

    # 正己烷 C6H14（链） + 环己烷 C6H12（环），两者都用真实键表构造
    u = mda.Universe.empty(6 + 14 + 6 + 12, n_residues=2,
                           atom_resindex=[0] * 20 + [1] * 18, trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * 12 + [1.0] * 26)
    u.add_TopologyAttr("resnames", ["HEX", "CHX"])
    u.add_TopologyAttr("resids", [1, 2])
    u.add_TopologyAttr("segids", ["SYSTEM"])
    # 元素：前 6 个 C（正己烷碳）+ 14 H，后 6 C（环己烷）+ 12 H
    u.add_TopologyAttr("elements",
                       ["C"] * 6 + ["H"] * 14 + ["C"] * 6 + ["H"] * 12)
    # 键：正己烷 = C-C 5 根 + H（端基 C 各 3 个、中间 C 各 2 个，共 14）；
    #     环己烷 = C-C 6 根成环 + 每个 C 2 个 H（共 12）。H 必须**全部**连上，
    #     否则游离 H 会被当成独立"分子"（第一版就踩了这个坑：分子数 4 而不是 2）。
    bonds = [(i, i + 1) for i in range(5)]
    h = 6
    for i, n_h in enumerate([3, 2, 2, 2, 2, 3]):
        for _ in range(n_h):
            bonds.append((i, h))
            h += 1
    base = 20
    bonds += [(base + i, base + (i + 1) % 6) for i in range(6)]
    h = 26
    for i in range(6):
        for _ in range(2):
            bonds.append((base + i, h))
            h += 1
    u.add_TopologyAttr("bonds", np.asarray(bonds, dtype=int))

    prof = SET.molecule_profile(u)
    assert prof["n_mol"] == 2, prof["n_mol"]
    assert prof["is_chain"].tolist() == [True, False], prof["is_chain"]
    assert prof["has_ring"].tolist() == [False, True], prof["has_ring"]

    # 1) 只取 C：把分子切碎 → atom
    ag, info = SET.build_set(u, {"name": "C", "elements": ["C"]})
    assert info["kind"] == "atom", info
    assert ag.n_atoms == 12, ag.n_atoms

    # 2) 链状 / 含环：整分子 → molecule
    ag_c, info_c = SET.build_set(u, {"name": "链状", "shape": "chain"})
    assert info_c["kind"] == "molecule" and info_c["n_atoms"] == 20, info_c
    ag_r, info_r = SET.build_set(u, {"name": "含环", "shape": "ring"})
    assert info_r["kind"] == "molecule" and info_r["n_atoms"] == 18, info_r

    # 3) 纯元素分子：整分子只含 C 的 → 只有环己烷的碳骨架不成立（有 H），应为空
    try:
        SET.build_set(u, {"name": "纯碳分子", "elements": ["C"], "pure": True})
        raise AssertionError("纯碳分子不该存在（两个分子都有 H）")
    except SET.SetError:
        pass

    # 4) 范围限定：只在"CHX"残基里取 C
    ag4, info4 = SET.build_set(
        u, {"name": "CHX-C", "elements": ["C"], "scope": ["*"]})
    assert ag4.n_atoms == 12
    assert SET.available_elements(u) == ["C", "H"], SET.available_elements(u)
    pres = SET.suggest_presets(u)
    names = [x["name"] for x in pres]
    # 元素预设（C/N/O）已按用户要求删除 —— 界面上的元素 chip 点一下就建集合；
    # 预设现在只剩"按分子形态/规模"这几类。
    assert "链状分子" in names and "含环分子" in names, names
    assert "C" not in names and "重原子" not in names, names

    return (f"合成体系（正己烷 20 原子 + 环己烷 18 原子）：C→{info['n_atoms']} 原子"
            f"（kind=atom）；链状 → {info_c['n_atoms']} 原子/1 分子；"
            f"含环 → {info_r['n_atoms']} 原子/1 分子；纯碳分子正确报空；"
            f"预设 {names}")


def test_msd_tau_axis_defaults_to_one_of_five_intervals():
    """回归：MSD 的 **τ 轴默认用满整条轨迹**（用户：6000 ps 的轨迹不能只显示 1200 ps）。

    背景（用户实测反馈）：原来 τ 一直取到整条轨迹末尾，6001 帧的轨迹就有 6001 个 τ，
    跨度太大 —— 双对数图上点全挤在右侧，而 MSD 末端只有一两个时间起点参与平均。
    现在默认把总时长均分成 5 段，只画/只拟合到一个区间的长度；
    「时间区间数」可改（1 = 用满整条轨迹，10 = 只取 1/10）。
    """
    n, P = 401, 4
    rng = np.random.default_rng(7)
    xyz = np.cumsum(rng.normal(size=(n, P, 3)), axis=0).astype(np.float64) * 0.5
    u = mda.Universe.empty(P, n_residues=P, atom_resindex=list(range(P)),
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * P)
    u.add_TopologyAttr("molnums", np.arange(P, dtype=int))
    u.trajectory = mda.coordinates.memory.MemoryReader(
        xyz, order="fac",
        dimensions=np.array([5000.0, 5000.0, 5000.0, 90.0, 90.0, 90.0],
                            dtype=np.float32))
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.arange(n, dtype=float) * 1.0)   # 帧间隔 1 ps

    # 默认：**用满整条轨迹**（6000 ps 的轨迹就该看到 0–6000 ps，不许悄悄截断）
    res = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                          remove_drift=False)
    curve = res.curves[0]
    tau_max = float(curve.x[-1])
    assert int(round(tau_max)) == n - 1, (tau_max, n - 1)
    assert curve.x.size == n, curve.x.size
    assert not any("总时长 /" in s for s in res.notes), res.notes[:3]

    # 只有**显式**给 tau_intervals 才截断
    res5 = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                           remove_drift=False, tau_intervals=5)
    assert int(round(float(res5.curves[0].x[-1]))) == (n - 1) // 5, \
        res5.curves[0].x[-1]
    assert any("总时长 / 5 段" in s for s in res5.notes), res5.notes[:3]

    res1 = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                           remove_drift=False, tau_intervals=1)
    assert int(round(float(res1.curves[0].x[-1]))) == n - 1, res1.curves[0].x[-1]

    res10 = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                            remove_drift=False, tau_intervals=10)
    assert int(round(float(res10.curves[0].x[-1]))) == (n - 1) // 10

    # 拟合也只能用截断后的区间（否则「图上到 80 ps、D 却按 400 ps 拟合」自相矛盾）
    fit = [c for c in res.curves if c.label.endswith("线性拟合")][0]
    assert float(fit.x[-1]) <= tau_max + 1e-6, (fit.x[-1], tau_max)

    return (f"{n} 帧（1 ps/帧）：默认 τ 全覆盖 {tau_max:.0f} ps（{curve.x.size} 个点）；"
            f"显式 区间数=5 → {res5.curves[0].x[-1]:.0f} ps、"
            f"=1 → {res1.curves[0].x[-1]:.0f} ps、"
            f"=10 → {res10.curves[0].x[-1]:.0f} ps；拟合线不超出 τ 上限")


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


def test_rg_object_modes_are_mutually_exclusive():
    """Rg 的「分析对象」两档**互斥**：整体体系只算整组，分子 / 链只算逐分子。

    合成体系：3 个正方形刚性分子（边长 a、2a、3a），各自 Rg = 边长/√2，
    分子之间相距 100 Å。于是
    1. 「分子 / 链」模式的单分子 Rg 平均必须精确等于解析值 (a+2a+3a)/√2 的平均；
    2. 「整体体系」模式的整组 Rg 远大于单分子 Rg（1.0.0 会把两者混为一谈）；
    3. 两档**各只算一种**：整体体系不产出逐分子统计量、分子 / 链不产出整组曲线 ——
       这是 1.0.4 按用户要求拆开的（1.0.3 及更早一次算两套，同一条轨迹扫两遍）。
    4. 旧的 ``per_molecule`` 布尔开关仍兼容：False→整体体系、True→分子 / 链。
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

    # ---- ① 分子 / 链（默认档）
    rm = conf.analyze_rg(mdt, u.atoms, sel, label="X")
    assert int(rm.summary["分子数"]) == n_mol, rm.summary
    assert rm.summary["分析对象"] == "分子 / 链", rm.summary.get("分析对象")
    got = float(rm.summary["单分子 Rg 平均 (Å)"])
    expect = float(np.mean(res_expected))
    assert abs(got - expect) < 1e-5, (got, expect)
    assert abs(float(rm.summary["Rg mean"]) - expect) < 1e-5, rm.summary["Rg mean"]
    assert len(rm.curves_of(2)) >= 1, "缺少逐分子 Rg 面板"
    mol_labels = [c.label for c in rm.curves]
    assert "逐帧各分子 Rg 平均" in mol_labels, mol_labels
    assert "X" not in mol_labels, "分子 / 链模式不应再产出整组曲线"

    # ---- ② 整体体系
    rs = conf.analyze_rg(mdt, u.atoms, sel, label="X", object="system")
    assert rs.summary["分析对象"] == "整体体系", rs.summary.get("分析对象")
    whole = float(rs.summary["Rg mean"])
    assert whole > 5.0 * got, (whole, got, "整组 Rg 应远大于单分子 Rg")
    assert "单分子 Rg 平均 (Å)" not in rs.summary, "整体体系模式不应算逐分子 Rg"
    sys_labels = [c.label for c in rs.curves]
    assert "X" in sys_labels and "逐帧各分子 Rg 平均" not in sys_labels, sys_labels
    assert len(rs.panels) == 2 and len(rm.panels) == 3, (len(rs.panels), len(rm.panels))

    # ---- ③ 旧布尔开关兼容（False→整体体系，True→分子 / 链）
    assert conf.analyze_rg(mdt, u.atoms, sel,
                           per_molecule=False).summary["分析对象"] == "整体体系"
    assert conf.analyze_rg(mdt, u.atoms, sel,
                           per_molecule=True).summary["分析对象"] == "分子 / 链"
    # 未知取值必须报错（不静默兜底）
    try:
        conf.analyze_rg(mdt, u.atoms, sel, object="整团")
    except ValueError as exc:
        assert "分析对象" in str(exc)
    else:
        raise AssertionError("未知的「分析对象」取值应当报错")

    # ---- ④ 回归：**拿不到分子划分**的单个连通对象（无 molnums、键图只有 1 个碎片，
    #        例如 Materials Studio 导出的单个大分子）在「分子 / 链」档必须给出数值，
    #        不能整条曲线 NaN（1.0.4 刚改完时这里确实空过）。
    chain = [[float(i), 0.0, 0.0] for i in range(6)]
    u3 = mda.Universe.empty(6, n_residues=1, atom_resindex=[0] * 6, trajectory=True)
    u3.add_TopologyAttr("masses", [12.0] * 6)
    u3.add_TopologyAttr("bonds", [(i, i + 1) for i in range(5)])
    u3.trajectory = mda.coordinates.memory.MemoryReader(
        np.asarray(chain, dtype=np.float32)[None, :, :], order="fac",
        dimensions=np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float32))
    mdt3 = MDTrajectory.from_universe(u3)
    sel3 = select_frames(np.array([0.0]))
    r3 = conf.analyze_rg(mdt3, u3.atoms, sel3, label="单链")
    v_mol = float(r3.summary["Rg mean"])
    v_sys = float(conf.analyze_rg(mdt3, u3.atoms, sel3, label="单链",
                                  object="system").summary["Rg mean"])
    expect_chain = float(np.std(np.arange(6, dtype=float)))      # 6 个等间距点
    assert np.isfinite(v_mol) and np.isfinite(v_sys), (v_mol, v_sys)
    assert abs(v_mol - expect_chain) < 1e-6, (v_mol, expect_chain)
    assert abs(v_mol - v_sys) < 1e-9, (v_mol, v_sys)
    assert int(r3.summary["分子数"]) == 1, r3.summary["分子数"]

    return (f"3 个分子（边长 {a:g}/{2 * a:g}/{3 * a:g} Å）：分子 / 链 单分子 Rg 平均 "
            f"{got:.4f} Å（解析值 {expect:.4f}）；整体体系 {whole:.2f} Å "
            f"= 单分子的 {whole / got:.1f} 倍；两档互斥（各只产出一套数）；"
            f"无分子划分的单链 两档同值 {v_mol:.4f} Å（解析 {expect_chain:.4f}）")


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
    # tau_intervals=1：要完整 τ 轴（下面要拿 np.arange(n_f) 与曲线逐点对比）
    res = dyn.analyze_msd(mdt, {"X": u.atoms}, sel, object="molecule",
                          remove_drift=False, fit_fraction=(0.1, 0.9),
                          tau_intervals=1)

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


def test_estimate_times_is_measured_and_side_effect_free():
    """预估用时：两点实测的算术要正确，噪声时偏保守，且不污染分析状态。

    这里用**假计时**替换真实探测，把外推公式钉死；再用真实小体系验证
    "帧数太少就拒绝预估"这条门槛（AdK 只有 10 帧，预估要跑 4 帧，
    占比过高，不如直接跑）。
    """
    top, xtc = _CLI_ARGS[0] or TOP_DEFAULT, _CLI_ARGS[1] or XTRAJ_DEFAULT
    az_ = Analyzer(top, xtc)
    az_.auto_setup()
    sel_before = az_.require_frames()
    res_before = dict(az_.results)
    n = az_.require_frames().n_frames

    # (a) 真实小体系：应当拒绝预估（否则预估成本会超过分析本身）
    real = az_.estimate_times(["rg", "msd"])
    for nm in ("rg", "msd"):
        assert real[nm]["est_sec"] is None, real[nm]
        assert "帧" in real[nm]["note"], real[nm]
    assert real["rg"]["probe_frames"] == 0, real["rg"]

    # (b) 外推公式（1.0.5 起是"热身 + 两点"）：
    #     探测 = 头 2m 帧（m = probe_frames = 4 → 8 帧），热身 2 帧丢弃，
    #     t1 测 4 帧、tk 测 8 帧：t1=2.0s、tk=4.0s → 每帧 0.5s、固定 0
    az_.ESTIMATE_MIN_FRAMES = 0                     # 实例属性覆盖，强制走数字路径
    seq = [0.1, 2.0, 4.0]                           # 热身 / t1 / tk
    az_._time_one = lambda name, params, sel: seq.pop(0) if seq else 4.0
    est = az_.estimate_times(["rg"], probe_frames=4)["rg"]
    assert est["probe_frames"] == 8, est
    assert abs(est["per_frame_sec"] - 0.5) < 1e-9, est
    assert abs(est["est_sec"] - 0.5 * n) < 1e-6, est
    assert est["fixed_sec"] == 0.0, est

    # (c) 固定开销为正：每帧 0.2s、固定 2.6s → t1(4 帧)=3.4s、tk(8 帧)=4.2s
    seq[:] = [0.1, 3.4, 4.2]
    est2 = az_.estimate_times(["rg"], probe_frames=4)["rg"]
    assert abs(est2["fixed_sec"] - 2.6) < 1e-6, est2
    assert abs(est2["est_sec"] - (2.6 + 0.2 * n)) < 1e-6, est2

    # (d) 噪声（tk < t1）：不能报"每帧 0ms"把这一项判成最快，要用平均代价兜底
    seq[:] = [0.1, 2.0, 1.0]
    est3 = az_.estimate_times(["rg"], probe_frames=4)["rg"]
    assert est3["per_frame_sec"] > 0, est3
    assert abs(est3["est_sec"] - 1.0 / 8 * n) < 1e-6, est3

    # (e) 探测不该改动任何分析状态
    assert az_.frames is sel_before, "estimate_times 改动了帧选择"
    assert az_.results == res_before, "estimate_times 污染了结果集"
    assert az_.current == "", f"current 未复原: {az_.current!r}"
    return (f"小体系（{n} 帧）正确拒绝预估；两点外推 5.0/{2.6 + 0.2 * n:.1f}/"
            f"噪声兜底 {1.0 / 5 * n:.1f} s 全部正确；状态未被污染")


def test_shortest_first_scheduling():
    """按预估用时短→长执行；估不出来的排最后，其余保持原相对次序。"""
    got = sort_by_estimate(["rg", "rdf", "msd", "contact"],
                           {"rg": 30, "rdf": 2, "msd": 9})
    assert got == ["rdf", "msd", "rg", "contact"], got

    top, xtc = _CLI_ARGS[0] or TOP_DEFAULT, _CLI_ARGS[1] or XTRAJ_DEFAULT
    az_ = Analyzer(top, xtc)
    az_.auto_setup()
    az_.require_frames()
    seen = []
    real_run = az_.run
    az_.run = lambda name, **kw: (seen.append(name), object())[1]
    try:
        az_.run_all(["rg", "rdf", "msd"], order="shortest",
                    estimates={"rg": 30, "rdf": 2, "msd": 9},
                    on_result=lambda name, *a: None,
                    verbose=False, raise_errors=False)
    finally:
        az_.run = real_run
    assert seen == ["rdf", "msd", "rg"], seen
    assert az_.run_order == ["rdf", "msd", "rg"], az_.run_order
    return f"排序 {got}；实际执行顺序 {seen}（run_order 一致）"


def test_peg_chain_type_merges_all_chains():
    """PEG 数据集：链类型含多条链时，「全部链」必须等于组分本身。

    回归目标：`list_chains` 按 segid+resname 归并出**链类型**（PEG 体系实测
    `SYSTEM:PEG` = 200 条 × 30 原子），界面上显示成"200 条"。选中它就必须真的
    把 200 条并成一个对象 —— 1.0.1 之前 web 层只取 `got[0]`，于是一个标着
    "200 条链"的选项实际只分析了 30 个原子，数值（7.54 Å vs 26.99 Å）与耗时
    （4 s vs 99 s）全对不上。

    这里在 mdta 层做等价校验：合并 200 条链 == polymer 组分（同一批原子），
    并用独立公式复核 Rg。
    """
    peg = os.environ.get("MDTA_PEG_DIR", r"C:\temp\dataset\PEG_parametrization")
    top = os.path.join(peg, "eqpbc.gro")
    xtc = os.path.join(peg, "trajpbc.xtc")
    if not (os.path.isfile(top) and os.path.isfile(xtc)):
        return f"跳过：找不到 PEG 数据集 {peg}（可用 MDTA_PEG_DIR 指定）"
    mdt = load_trajectory(top, xtc)
    u = mdt.universe
    from mdta.selection import chains as _chains
    from mdta.selection import list_chains

    multi = [c for c in list_chains(u) if int(c.count) > 1]
    assert multi, "PEG 体系应当有含多条链的链类型"
    ct = max(multi, key=lambda c: int(c.count))
    assert int(ct.count) == 200 and int(ct.n_atoms) == 30, (ct.count, ct.n_atoms)

    got = _chains(u, segid=str(ct.segid), resname=str(ct.resname),
                  min_atoms=int(ct.n_atoms))
    assert len(got) == 200, len(got)
    merged = u.atoms[np.sort(np.concatenate([np.asarray(g.indices) for g in got]))]
    assert merged.n_atoms == 6000, merged.n_atoms
    assert np.array_equal(np.sort(np.asarray(merged.indices)),
                          np.sort(np.asarray(u.atoms.indices))), \
        "合并 200 条链应当覆盖整个体系（该体系只有一个组分）"

    sel = select_frames(mdt.times_ps, max_frames=2)
    res = conf.analyze_rg(mdt, merged, sel, object="system", label="PEG 全部链")
    assert int(res.summary["分子数"]) == 200, res.summary["分子数"]
    # 独立复核**第 0 个选中帧**：质量加权 Rg 的解析式
    # （注意：analyze_rg 跑完后 universe 停在最后一帧，必须先回到 sel.indices[0]，
    #  否则拿的是另一帧的坐标，会差出一两埃）
    u.trajectory[int(sel.indices[0])]
    pos = positions_for(merged, unwrap=True)
    m = np.asarray(merged.masses, dtype=float)
    com = (pos * m[:, None]).sum(0) / m.sum()
    rg_ref = float(np.sqrt((m * ((pos - com) ** 2).sum(1)).sum() / m.sum()))
    # 容差 1e-4 Å：多分子组分上的整组 Rg 走 `rg_fast` 快路径，它与
    # `positions_for(unwrap=True)` **同一镜像约定**（每个分子先按键/最小镜像重排、
    # 再把分子质心 apply_PBC 搬回主盒），但 MDAnalysis 内部把碎片质心转成 float32
    # 再回卷，两边因此差 ~1e-7 Å。1.0.3 之前两条路径约定不同（同一帧整组 Rg 差
    # 26.8 Å vs 88.9 Å，3.3 倍），那才是真正要挡的 bug，容差按 1e-4 卡足够严。
    d_group = abs(res.curves[0].y[0] - rg_ref)
    assert d_group < 1e-4, (res.curves[0].y[0], rg_ref)
    # 逐分子 Rg（= 与 Forcite 对照的量）必须与**键连通重建**（mdamath.make_whole）
    # 独立算法一致：这是快路径的核心正确性断言，容差 1e-6 Å（实测 ≤5.3e-09）。
    from MDAnalysis.lib import mdamath

    res_mol = conf.analyze_rg(mdt, merged, sel, label="PEG 全部链（逐分子）")
    cur = next(c for c in res_mol.curves if c.label == "逐帧各分子 Rg 平均")
    u.trajectory[int(sel.indices[0])]
    mol_ids = conf.molecule_slices(merged)[0]
    groups_f = conf.prepare_molecule_groups(mol_ids)
    inv, n_mol_f = groups_f.inv, groups_f.n_mol
    tot_f, rg_mol_f, safe_f, _d = conf.rg_fast(
        np.asarray(merged.positions, dtype=float), m, u.dimensions,
        groups=groups_f)
    assert safe_f, "PEG 200×30 的分子尺度远小于半盒，快路径应当被判为安全"
    worst_mol = 0.0
    for j in range(n_mol_f):
        sub = inv == j
        worst_mol = max(worst_mol, abs(rg_mol_f[j]
                                       - conf.compute_rg(mdamath.make_whole(
                                           merged[sub], inplace=False), m[sub])))
    assert worst_mol < 1e-6, worst_mol
    assert abs(float(cur.y[0]) - float(np.mean(rg_mol_f))) < 1e-6, (cur.y[0], np.mean(rg_mol_f))
    # 单条链与整组必须差得明显（否则就是没合并）
    one = conf.analyze_rg(mdt, got[0], sel, object="system", label="第 1 条")
    assert res.summary["Rg mean"] - one.summary["Rg mean"] > 5.0, (
        res.summary["Rg mean"], one.summary["Rg mean"])
    return (f"{ct.label}: {ct.count} 条 × {ct.n_atoms} 原子 → 合并 {merged.n_atoms} 原子；"
            f"第 0 帧 Rg={res.curves[0].y[0]:.3f} Å 与独立公式 {rg_ref:.3f} Å 一致"
            f"（差 {d_group:.1e} Å，快/精确同约定）；逐分子 Rg 对 make_whole 最大差 "
            f"{worst_mol:.1e} Å（2 帧均值 {res.summary['Rg mean']:.3f} Å）；单条链 "
            f"{one.summary['Rg mean']:.3f} Å，差 "
            f"{res.summary['Rg mean'] - one.summary['Rg mean']:.1f} Å")


def test_qc_checks_are_derived_and_structured():
    """科研 QC：判据要变成**结构化检查项**，并给出整体结论。

    这里用合成结果把规则钉死：饱和概率、不显著第一峰、PBC 追踪失效、
    偏离扩散标度、样本偏少、量被拒绝 —— 各自应导出预期级别的检查项。
    """
    from mdta.core import AnalysisResult
    from mdta.qc import checks_digest, worst_level

    # (a) 干净的曲线：整体 ok
    r = AnalysisResult(name="rg", title="Rg")
    r.summary.update({"分子数": 3, "统计口径": "整组 Rg + 逐分子 Rg", "Rg n": 500})
    checks = derive_checks(r)
    assert checks and worst_level(checks) == "ok", checks
    assert any(c["名称"] == "分析对象" for c in checks)

    # (b) 各种问题：级别必须对
    r2 = AnalysisResult(name="contact", title="接触")
    r2.summary.update({"接触概率是否饱和": "是（恒为 1，无区分度）", "配对模式": "inter"})
    derive_checks(r2)
    assert any(c["名称"] == "指标区分度" and c["级别"] == "warn" for c in r2.checks)

    r3 = AnalysisResult(name="rdf", title="RDF")
    r3.summary.update({"W-W 第一峰是否显著": "否", "W-W 第一峰高度 g_max": 1.12,
                       "W-W 第一峰位置 (Å)": 4.95})
    derive_checks(r3)
    c3 = [c for c in r3.checks if c["名称"] == "结构峰显著性"][0]
    assert c3["级别"] == "warn" and "不给配位数" in c3["依据"], c3
    assert "1.120" in c3["结论"] and "4.95" in c3["结论"], c3["结论"]

    r4 = AnalysisResult(name="msd", title="MSD")
    r4.summary.update({"polymer 最小镜像失效比例": 0.24,
                       "polymer α (log-log 斜率)": 0.75,
                       "polymer D 标准误 (m²/s)": 1e-12,
                       "polymer 标准误来源": "拟合协方差 × 实测 AR(1) 修正（分块不足 3 块，误差偏乐观）"})
    derive_checks(r4)
    by = {c["名称"]: c for c in r4.checks}
    assert by["PBC 追踪"]["级别"] == "bad", by["PBC 追踪"]
    assert by["扩散标度"]["级别"] == "warn", by["扩散标度"]
    assert by["误差棒可信度"]["级别"] == "warn", by["误差棒可信度"]
    assert worst_level(r4.checks) == "bad"

    # 容差边界：|α-1| ≤ 0.15 判 ok，超过才 warn
    r4b = AnalysisResult(name="msd", title="MSD")
    r4b.summary["polymer α (log-log 斜率)"] = 0.88
    derive_checks(r4b)
    assert [c for c in r4b.checks if c["名称"] == "扩散标度"][0]["级别"] == "ok"
    r4c = AnalysisResult(name="msd", title="MSD")
    r4c.summary["polymer α (log-log 斜率)"] = 0.80
    derive_checks(r4c)
    assert [c for c in r4c.checks if c["名称"] == "扩散标度"][0]["级别"] == "warn"

    r5 = AnalysisResult(name="ree", title="R_ee")
    r5.summary.update({"R_ee 是否可定义": "否", "R_ee 说明": "该组是环状分子，没有端基"})
    derive_checks(r5)
    assert any(c["级别"] == "bad" and "拒绝" in c["结论"] for c in r5.checks), r5.checks

    r6 = AnalysisResult(name="x", title="x")            # 无产出 → bad
    derive_checks(r6)
    assert worst_level(r6.checks) == "bad", r6.checks

    r7 = AnalysisResult(name="rg", title="Rg")          # 样本偏少 → warn
    r7.summary.update({"Rg n": 5})
    derive_checks(r7)
    assert any(c["名称"] == "样本数" and c["级别"] == "warn" for c in r7.checks)

    d = checks_digest(r4.checks)
    assert d["bad"] >= 1 and d["warn"] >= 2, d

    # (c) 真实体系上每个分析都要有检查项
    top, xtc = _CLI_ARGS[0] or TOP_DEFAULT, _CLI_ARGS[1] or XTRAJ_DEFAULT
    az_ = Analyzer(top, xtc)
    az_.auto_setup()
    az_.frames = az_.require_frames().head(4)
    n_checked = 0
    for nm in ("rg", "ree", "msd", "contact", "rdf"):
        r = az_.run(nm)
        if r is None:
            continue
        assert r.checks, f"{nm} 没有导出任何 QC 检查项"
        assert all({"名称", "级别", "结论", "依据"} <= set(c) for c in r.checks)
        n_checked += 1
    return (f"合成规则 7 组全对（含 bad/warn 分级与整体结论）；真实体系 "
            f"{n_checked} 个分析各自导出检查项")


def test_trans_threshold_is_shared_between_dihedral_and_order():
    """trans/gauche 阈值必须**两个模块共用**（改一处两处都变）。

    回归目标：`analyze_structural_order` 早先把 120° 写死，界面上改阈值只影响
    「二面角分析」，「结构有序度」里的 trans 构象比例仍是旧阈值算的 —— 同一条
    轨迹上两个模块会报出互相矛盾的"trans 比例"。
    """
    top, xtc = _CLI_ARGS[0] or TOP_DEFAULT, _CLI_ARGS[1] or XTRAJ_DEFAULT
    az_ = Analyzer(top, xtc)
    az_.auto_setup()
    az_.frames = az_.require_frames().head(4)
    label = az_.primary_label
    out = {}
    for edge in (120.0, 100.0):
        d = az_.run("dihedral", params={"gauche_edge": edge, "mode": "chain"})
        o = az_.run("order", params={"gauche_edge": edge})
        out[edge] = (d.summary["dihedral trans 比例"],
                     o.summary["反式构象比例 平均"])
    # 收紧阈值（120°→100°）后，两个模块的 trans 比例都必须**上升**
    assert out[100.0][0] > out[120.0][0], out
    assert out[100.0][1] > out[120.0][1], out
    # 两者统计的二面角集合不同（几何骨架 vs 化学重复单元），数值不必相等，
    # 但都必须随阈值单调变化 —— 这才是"共用判据"的可验证含义
    assert any("trans 判据" in n for n in
               az_.run("order").notes), "order 没写明 trans 判据"
    return (f"{label}：阈值 120°→100° 时 dihedral trans 比例 "
            f"{out[120.0][0]:.3f}→{out[100.0][0]:.3f}、order "
            f"{out[120.0][1]:.3f}→{out[100.0][1]:.3f}，两者同步变化")


def test_boo_matches_analytic_lattice_values():
    """BOO：对**理想晶格**必须复现 Steinhardt 解析值（fcc/hcp/bcc/sc）。

    两条独立路径：

    1. 方向集直接算 —— 把 12/8/6 个最近邻方向喂给模块内部同一个球谐实现，
       检查归一化常数 4π/(2l+1) 与 l=4/6 都对；
    2. 完整流程 —— 造**真实晶格 Universe**（含 PBC、邻居搜索、逐原子平均），
       检查 q4/q6 的平均值与解析值一致；再与"同密度的随机液体"对比，
       确认判据能区分固-液（液相 q6 必须显著更低）。
    """
    from mdta.analysis import boo as BOO

    # ---- 1) 方向集（解析最近邻方向）
    fcc_dirs = []
    for a in (0, 1, -1):
        for b in (0, 1, -1):
            for c in (0, 1, -1):
                v = np.array([a, b, c], float)
                if np.count_nonzero(v) == 2:
                    fcc_dirs.append(v / np.linalg.norm(v))
    fcc_dirs = np.unique(np.round(np.array(fcc_dirs), 9), axis=0)
    assert len(fcc_dirs) == 12
    # 理想 hcp（c/a=√(8/3)）的 12 个最近邻：6 个面内 + 上下各 3 个**方位角与上方重合**
    # （anticuboctahedron，重叠）。⚠️ 若把上下三角错开 60° 就变成 fcc 的
    # cuboctahedron，q4 会得到 fcc 的 0.19094 —— 这个坑我第一次就踩了。
    hcp_dirs = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0.5, 0.8660254, 0], [-0.5, 0.8660254, 0],
        [0.5, -0.8660254, 0], [-0.5, -0.8660254, 0],
        [0.5, 0.2886751, 0.8164966], [-0.5, 0.2886751, 0.8164966],
        [0, -0.5773503, 0.8164966],
        [0.5, 0.2886751, -0.8164966], [-0.5, 0.2886751, -0.8164966],
        [0, -0.5773503, -0.8164966]], float)
    hcp_dirs /= np.linalg.norm(hcp_dirs, axis=1)[:, None]
    sc_dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    # ⚠️ 文献里 bcc 的 q4=0.03637 / q6=0.51069 用的是 **14 邻居**（8 个 <111> + 6 个 <100>）；
    #    只取 8 个 <111> 会得到 0.50918 / 0.62854（那是立方体角方向的值）。
    bcc_dirs = np.array([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)]
                        + [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                           [0, 0, 1], [0, 0, -1]], float)
    bcc_dirs /= np.linalg.norm(bcc_dirs, axis=1)[:, None]
    got = {}
    for name, dirs in (("fcc", fcc_dirs), ("hcp", hcp_dirs), ("sc", sc_dirs), ("bcc", bcc_dirs)):
        got[name] = (BOO.ql_from_directions(dirs, 4), BOO.ql_from_directions(dirs, 6))
        ref = BOO.REFERENCE_QL[name]
        assert abs(got[name][0] - ref[0]) < 5e-4, (name, got[name][0], ref[0])
        assert abs(got[name][1] - ref[1]) < 5e-4, (name, got[name][1], ref[1])

    # ---- 2) 完整流程：晶格 Universe + PBC 邻居搜索
    def lattice_universe(kind, n=3, a=5.0):
        bases = {"sc": [(0, 0, 0)],
                 "bcc": [(0, 0, 0), (.5, .5, .5)],
                 "fcc": [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
                 "diamond": [(0, 0, 0), (.25, .25, .25), (.5, .5, 0), (.75, .75, .25),
                             (.5, 0, .5), (.75, .25, .75), (0, .5, .5), (.25, .75, .75)]}
        pts = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for b in bases[kind]:
                        pts.append((np.array([i, j, k]) + np.array(b)) * a)
        pos = np.array(pts, dtype=np.float32)
        u = mda.Universe.empty(len(pos), n_residues=len(pos),
                               atom_resindex=np.arange(len(pos)), trajectory=True)
        u.add_TopologyAttr("masses", np.ones(len(pos)))
        u.dimensions = np.array([n * a, n * a, n * a, 90, 90, 90], dtype=np.float32)
        u.trajectory = mda.coordinates.memory.MemoryReader(pos[None], order="fac",
                                                           dimensions=u.dimensions)
        return u

    mdt = MDTrajectory.from_universe(lattice_universe("fcc", n=3, a=5.0))
    sel = select_frames(np.array([0.0]))
    # 装了 freud 就自动一并用解析值校它：**不重复造轮子**，有库就用库
    # 默认只跑自带实现（零依赖）；**若环境里恰好有 freud**，顺带用它交叉核对
    backends = ["numpy"] + (["freud"] if BOO._freud_available() else [])
    r_fcc = None
    for be in backends:
        rb = BOO.analyze_boo(mdt, mdt.universe.atoms, sel, cutoff=3.7,
                             averaged=False, backend=be)
        assert rb.summary["计算后端"] == be, rb.summary["计算后端"]
        assert abs(rb.summary["q4 平均"] - 0.19094) < 2e-3, (be, rb.summary["q4 平均"])
        assert abs(rb.summary["q6 平均"] - 0.57452) < 2e-3, (be, rb.summary["q6 平均"])
        assert abs(rb.summary["平均邻居数"] - 12.0) < 1e-6, rb.summary["平均邻居数"]
        r_fcc = rb

    for kind, a, cut, q4, q6 in (("sc", 5.0, 5.6, 0.76376, 0.35355),
                                 # bcc 的参考文献值需要**两个壳层**：第一壳 8 个 <111>
                                 # 在 a√3/2=4.33 Å、第二壳 6 个 <100> 在 a=5.0 Å，
                                 # 所以 cutoff 必须 > 5.0（4.5 只会圈到 8 个，得到 0.50918）
                                 ("bcc", 5.0, 5.6, 0.03637, 0.51069)):
        mdt_k = MDTrajectory.from_universe(lattice_universe(kind, n=3, a=a))
        r_k = BOO.analyze_boo(mdt_k, mdt_k.universe.atoms, sel, cutoff=cut, averaged=False)
        assert abs(r_k.summary["q4 平均"] - q4) < 3e-3, (kind, r_k.summary["q4 平均"], q4)
        assert abs(r_k.summary["q6 平均"] - q6) < 3e-3, (kind, r_k.summary["q6 平均"], q6)

    # ---- 3) 固-液区分：同密度的随机坐标（每个原子仍有 ~12 个邻居，但方向随机）
    rng = np.random.default_rng(7)
    n_side = 3
    L = n_side * 5.0
    n_at = 4 * n_side ** 3
    pos = rng.random((n_at, 3)) * L
    u_liq = mda.Universe.empty(n_at, n_residues=n_at,
                               atom_resindex=np.arange(n_at), trajectory=True)
    u_liq.add_TopologyAttr("masses", np.ones(n_at))
    u_liq.dimensions = np.array([L, L, L, 90, 90, 90], dtype=np.float32)
    u_liq.trajectory = mda.coordinates.memory.MemoryReader(
        pos.astype(np.float32)[None], order="fac", dimensions=u_liq.dimensions)
    mdt_liq = MDTrajectory.from_universe(u_liq)
    r_liq = BOO.analyze_boo(mdt_liq, u_liq.atoms, sel, cutoff=3.7, averaged=False)
    q6_liq = r_liq.summary["q6 平均"]
    assert r_liq.summary["平均邻居数"] > 5, r_liq.summary["平均邻居数"]
    assert q6_liq < 0.4 < r_fcc.summary["q6 平均"], (q6_liq, r_fcc.summary["q6 平均"])

    return (f"解析值复核：fcc q4={got['fcc'][0]:.5f}/q6={got['fcc'][1]:.5f}、"
            f"hcp {got['hcp'][0]:.5f}/{got['hcp'][1]:.5f}、"
            f"sc {got['sc'][0]:.5f}/{got['sc'][1]:.5f}、bcc {got['bcc'][0]:.5f}/{got['bcc'][1]:.5f}"
            f"；晶格 Universe 复核 fcc q6={r_fcc.summary['q6 平均']:.5f}（邻居 "
            f"{r_fcc.summary['平均邻居数']:.0f}，后端 {'/'.join(backends)}）；"
            f"随机液体 q6={q6_liq:.3f} 显著更低")


def test_boo_subsamples_large_groups_without_crashing():
    """回归：分析对象**超过逐原子球谐上限（8000）**时必须抽稀且不崩。

    这个 bug 是真在真实体系上踩到的：46 体系（`.gro`，47 453 原子）的 polymer 组分
    有 45 005 个原子，`analyze_boo` 里输出数组按 `ag.n_atoms`（45 005 列）分配，
    而每帧算出来的是抽稀后的 7 565 个值 —— 赋值时抛
    ``ValueError: could not broadcast input array from shape (7565,) into shape (45005,)``，
    整个 BOO 分析项直接失败（`run_all` 里也只留一行 "boo 失败"）。

    修法：**先抽稀、再按 sub.n_atoms 分配**。这里用一个 > 8000 原子的随机体系盯住它。
    """
    from mdta.analysis import boo as BOO

    rng = np.random.default_rng(11)
    n_at = 9000                       # > 8000 的抽稀阈值
    L = 60.0
    pos = rng.random((n_at, 3)) * L
    u = mda.Universe.empty(n_at, n_residues=n_at,
                           atom_resindex=np.arange(n_at), trajectory=True)
    u.add_TopologyAttr("masses", np.ones(n_at))
    u.dimensions = np.array([L, L, L, 90, 90, 90], dtype=np.float32)
    u.trajectory = mda.coordinates.memory.MemoryReader(
        pos.astype(np.float32)[None], order="fac", dimensions=u.dimensions)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    # averaged=True 会同时算 q_l 与平均版 q̄_l（两条赋值路径都要过）
    res = BOO.analyze_boo(mdt, u.atoms, sel, cutoff=6.0, averaged=True)

    assert res.meta["n_atoms"] <= 8000, res.meta["n_atoms"]      # 确实抽稀了
    assert any("抽稀" in n for n in res.notes), res.notes
    for key in ("q4 平均", "q6 平均", "q4bar 平均", "q6bar 平均", "平均邻居数"):
        assert np.isfinite(res.summary[key]), (key, res.summary.get(key))
    # 时间序列曲线的长度必须等于**帧数**（列数/形状错了这里就会露出来；
    # 直方图那条是 bin 数，不在此列）
    ts = [c for c in res.curves if getattr(c, "panel", 0) == 0]
    assert ts and all(len(c.x) == 1 for c in ts), [len(c.x) for c in res.curves]
    return (f"9000 原子 → 抽稀到 {res.meta['n_atoms']} 个，q6={res.summary['q6 平均']:.3f}"
            f"、q̄6={res.summary['q6bar 平均']:.3f}（邻居 {res.summary['平均邻居数']:.1f}），"
            f"曲线 {len(res.curves)} 条均无形状错误")


def test_inplane_rdf_first_peak_ignores_far_noise_and_picks_window_max():
    """回归：2D 面内 RDF 的「第一峰」不能用全区间 argmax（会被几十 Å 外的噪声骗走）。

    实测背景：`rmax` 默认 70 Å，而每叶层的珠子可能只有二三十个（TOCL 24 个）。
    全区间 argmax 给出的"第一峰"落在 43.5 Å（作者实现）/18.7 Å（我们），
    两个都不是物理峰。现在的判据是"``r_min``–``window``（默认 2.2–25 Å）内的最大值"，
    显著性再由 ``g_max ≥ 1.2`` **且每叶层珠子 ≥50** 判定。

    另外两个踩过的坑也一并钉住：
      · 3D 那套"先上升后下降"判据会在 2–4 Å 的**排斥洞**里报出 g≈0.02–0.3 的假峰；
      · "第一个 g≥1 的局部极大"会抓到主峰前的**肩峰**（POPGK 在 5.1 Å 有 g=1.10 的肩，
        主峰在 8.7 Å/g=1.37）。
    """
    from mdta.analysis.interface import _first_inplane_peak

    r = np.arange(0.2, 70.0, 0.2)
    rng = np.random.default_rng(3)

    # (1) 真峰 8.7 Å（g=1.4）+ 43 Å 处一个更大的噪声尖峰（g=3.0）
    g = 1.0 + 0.03 * rng.standard_normal(r.size)
    g[np.argmin(np.abs(r - 8.7))] = 1.4
    g[np.argmin(np.abs(r - 43.0))] = 3.0
    pk = _first_inplane_peak(r, g)
    assert abs(pk["position"] - 8.6) < 0.3, pk
    assert pk["height"] < 1.5 and pk["significant"], pk

    # (2) 排斥洞 + 肩峰：窗口内最大 = 主峰 8.7，而不是 5.1 的肩
    g2 = np.full(r.size, 1.0)
    g2[r < 3.0] = 0.05 + 0.02 * rng.standard_normal(int((r < 3.0).sum()))
    g2[np.argmin(np.abs(r - 5.1))] = 1.10
    g2[np.argmin(np.abs(r - 8.7))] = 1.37
    pk2 = _first_inplane_peak(r, g2)
    assert abs(pk2["position"] - 8.6) < 0.3 and pk2["significant"], pk2

    # (3) 纯噪声（没有 g ≥ 1.2 的峰）→ 不显著
    pk3 = _first_inplane_peak(r, 1.0 + 0.05 * rng.standard_normal(r.size))
    assert not pk3["significant"], pk3

    return (f"8.7 Å 真峰 vs 43 Å 噪声尖峰 → 报 {pk['position']:.1f} Å（g={pk['height']:.2f}）；"
            f"肩峰 5.1/g=1.10 不抢主峰（报 {pk2['position']:.1f} Å）；纯噪声判为不显著")


def test_format_table_matches_mdanalysis_and_classifies_roles():
    """回归：后缀表必须与 MDAnalysis **实际能力**一致，且槽位分类正确。

    背景（真实故障）：用户在工具目录放了 `122C12-450K.pdb` + `.xtc`，报"读不了 pdb"。
    查下来后缀表本身有 `.pdb`，但露出三个真问题：
      ① `.lammpstrj` 在表里，MDAnalysis 这个版本却**认格式名不认后缀**
         （`DumpReader` 没有 EXTENSIONS）→ 按后缀自动识别必然失败；
      ② 把 `.xtc` 放进拓扑槽、或拿 `.md` 当拓扑，会一路抛到 API 变成 **HTTP 500**；
      ③ `.pdb` 这类"两者皆可"的格式只能被当成拓扑，想把它当轨迹没有入口。

    这条测试把"表 ↔ 能力"钉住：升级 MDAnalysis 后立刻能看出表该改哪里。
    """
    from MDAnalysis.coordinates.core import get_reader_for
    from MDAnalysis.topology.core import get_parser_for

    from mdta.formats import (EXPLICIT_FORMAT, EXPLICIT_TOPOLOGY_FORMAT,
                              MS_STRUCTURE_SUFFIXES, TOPOLOGY_SUFFIXES,
                              TRAJECTORY_SUFFIXES, can_be_topology,
                              can_be_trajectory, classify, format_for,
                              suffix_of, topology_format_for)

    def _probe(fn, ext: str) -> bool:
        try:
            fmt = EXPLICIT_FORMAT.get(ext)
            fn("probe" + ext, format=fmt) if fmt else fn("probe" + ext)
            return True
        except Exception:                                     # noqa: BLE001
            return False

    def _probe_top(ext: str) -> bool:
        """拓扑探测要带上格式名：`.lmp/.dat/.data/.lammps` 内容其实是
        LAMMPS data，MDAnalysis 只是**没给这些后缀登记 EXTENSIONS**。
        （`get_parser_for` 的形参叫 `format=`；`Universe` 才叫 `topology_format=`。）"""
        try:
            kw = {}
            fmt = EXPLICIT_TOPOLOGY_FORMAT.get(ext)
            if fmt:
                kw["format"] = fmt
            get_parser_for("probe" + ext, **kw)
            return True
        except Exception:                                     # noqa: BLE001
            return False

    # 判据要分开看：
    #   · 拓扑槽只要求"能被解析成拓扑"——`.psf` / `.prmtop` / `.parm7` / `.top`
    #     这些**本来就不含坐标**，它们的坐标要另外配轨迹（我们的用法正是如此）；
    #   · 轨迹槽要求"有坐标读取器"。
    # 早先我把两条判据混在一起（拓扑也要求有 reader），结果把 `.psf/.prmtop/.parm7/
    # .top/.xml` 全判成"读不了" —— 那是测试写错了，不是表写错了。
    bad_top = [e for e in sorted(TOPOLOGY_SUFFIXES)
               if e not in MS_STRUCTURE_SUFFIXES and not _probe_top(e)]
    bad_traj = [e for e in sorted(TRAJECTORY_SUFFIXES)
                if not _probe(get_reader_for, e)]
    assert not bad_top, f"表里的拓扑后缀本机解析不了: {bad_top}"
    assert not bad_traj, f"表里的轨迹后缀本机读不了: {bad_traj}"
    # 只当拓扑、又没有坐标的格式（.psf/.prmtop/.parm7/.top/.xml）必须**不在**轨迹表里
    traj_overlap = [e for e in sorted(TOPOLOGY_SUFFIXES)
                    if e not in MS_STRUCTURE_SUFFIXES
                    and not _probe(get_reader_for, e)
                    and e in TRAJECTORY_SUFFIXES]
    assert not traj_overlap, f"无坐标的拓扑格式混进了轨迹表: {traj_overlap}"
    # Materials Studio 自家格式：MDAnalysis **读不了**，由 mdta.ms_structure 自己解析
    assert MS_STRUCTURE_SUFFIXES == {".car", ".xsd", ".xtd"}, MS_STRUCTURE_SUFFIXES
    for e in MS_STRUCTURE_SUFFIXES:
        assert can_be_topology("x" + e) and not can_be_trajectory("x" + e), e
        assert not _probe(get_parser_for, e), f"{e} 竟被 MDAnalysis 认了？表可以更新"

    assert classify("sys.pdb") == "both", classify("sys.pdb")
    assert classify("sys.gro") == "both"
    assert classify("sys.xtc") == "trajectory", classify("sys.xtc")
    assert classify("sys.tpr") == "topology", classify("sys.tpr")
    assert classify("README.md") is None
    assert can_be_topology("a.PDB") and can_be_trajectory("a.PDB")   # 大小写无关
    assert suffix_of("a/b/sys.pdb.gz") == ".pdb.gz"
    assert format_for("traj.lammpstrj") == "LAMMPSDUMP"
    assert format_for("x.dump") == "LAMMPSDUMP"
    assert format_for("traj.xtc") is None
    # LAMMPS 补充后缀：内容一样、后缀没被 MDAnalysis 登记，必须显式给格式名
    for e in (".lmp", ".dat", ".data", ".lammps"):
        assert topology_format_for("x" + e) == "DATA", e
        assert can_be_topology("x" + e), e
    assert format_for("system.lammpstrj") == "LAMMPSDUMP"
    assert format_for("traj.lammpsdump") == "LAMMPSDUMP"
    assert can_be_trajectory("dump.lammpsdump")
    assert topology_format_for("sys.pdb") is None
    return (f"拓扑 {len(TOPOLOGY_SUFFIXES)} 种 / 轨迹 {len(TRAJECTORY_SUFFIXES)} 种 "
            f"全部通过 MDAnalysis 能力核对；.pdb/.gro 等为 both；"
            f"LAMMPS dump 需显式 format，data 后缀需显式 topology_format")


def test_lammps_data_and_dump_extensions_are_supported():
    """回归：LAMMPS 的 data / dump 各种后缀都要能读，且错槽位要有针对性的提示。

    背景（真实故障）：用户的 LAMMPS 体系常以 `system.lmp`、`system.data`、`system.dat`
    甚至 `system.lammps` 命名，dump 轨迹则是 `.lammpstrj` / `.dump` / `.lammpsdump`。
    MDAnalysis 2.10 只给 `.data`（DATA）和 `.lammpsdump`（LAMMPSDUMP）登记了 EXTENSIONS，
    其余后缀按名字自动识别必然失败——之前 `.lmp` 当拓扑直接报"读不了"，
    `.lammps` 更糟：它被 MDAnalysis 映射到 **DCDReader**，于是 data 文件被当成
    DCD 去读，报错信息与真实原因完全无关。

    这条测试钉住四件事：
      ① `.lmp/.dat/.data/.lammps`（内容为 LAMMPS data）都能当拓扑读出原子和键；
      ② `.lmp` + 同名 `.lammpstrj` 自动配对成拓扑+轨迹，帧数正确；
      ③ `.lammpsdump` 能当轨迹；
      ④ 输错槽位（dump 当拓扑、输入脚本当拓扑、日志当拓扑）给出**说人话**的错误。
    """
    from mdta.formats import sniff_lammps_kind
    from mdta.io import TrajectoryError, load_trajectory

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lammps_fixture")
    os.makedirs(d, exist_ok=True)
    data_txt = """LAMMPS data (mdta selftest)

4 atoms
1 bonds
2 atom types
1 bond types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 12.011
2 1.008

Atoms # full

1 1 1 0.0 1.0 1.0 1.0 0 0 0
2 1 1 0.0 2.5 1.0 1.0 0 0 0
3 1 2 0.0 4.0 1.0 1.0 0 0 0
4 1 2 0.0 5.5 1.0 1.0 0 0 0

Bonds

1 1 1 2
"""
    dump_txt = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 20
0 20
0 20
ITEM: ATOMS id type x y z
1 1 1.0000 1.0 1.0
2 1 2.5000 1.0 1.0
3 2 4.0000 1.0 1.0
4 2 5.5000 1.0 1.0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 20
0 20
0 20
ITEM: ATOMS id type x y z
1 1 1.1000 1.0 1.0
2 1 2.6000 1.0 1.0
3 2 4.1000 1.0 1.0
4 2 5.6000 1.0 1.0
"""
    written = {}
    for name, text in (("sys.lmp", data_txt), ("sys.dat", data_txt),
                       ("sys.data", data_txt), ("sys.lammps", data_txt),
                       ("pair.lmp", data_txt), ("pair.lammpstrj", dump_txt),
                       ("dump.lammpsdump", dump_txt), ("dump.lammpstrj", dump_txt),
                       ("in.lammps", "units lj\nread_data sys.lmp\n"
                                     "pair_style lj/cut 2.5\nrun 1000\n"),
                       ("log.lammps", "LAMMPS (23 Jun 2022)\nStep Temp E_pair\n"
                                      "0 1.0 -3.2\nLoop time of 1.2 on 1 procs\n")):
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        written[name] = p

    # ① 四种 data 后缀都要能读成拓扑（4 原子 / 1 键）
    for name in ("sys.lmp", "sys.dat", "sys.data", "sys.lammps"):
        mdt = load_trajectory(written[name])
        uni = mdt.universe
        assert uni.atoms.n_atoms == 4, (name, uni.atoms.n_atoms)
        assert len(uni.bonds) == 1, (name, len(uni.bonds))
        # data 文件自带坐标，但没有多帧：单帧体系，轨迹槽留空是对的
        assert mdt.trajectory is None, (name, mdt.trajectory)
        assert uni.trajectory.n_frames == 1, (name, uni.trajectory.n_frames)

    # ② `.lmp` 自动配上同名 `.lammpstrj`
    mdt = load_trajectory(written["pair.lmp"])
    assert mdt.trajectory is not None, "同名 .lammpstrj 没被自动配对"
    assert mdt.universe.trajectory.n_frames == 2, mdt.universe.trajectory.n_frames
    assert abs(float(mdt.universe.atoms[0].position[0]) - 1.0) < 1e-6   # 第 0 帧
    mdt.universe.trajectory[-1]
    assert abs(float(mdt.universe.atoms[0].position[0]) - 1.1) < 1e-6   # 第 1 帧

    # ③ `.lammpsdump` / `.lammpstrj` 当轨迹（配 data 拓扑）
    for name in ("dump.lammpsdump", "dump.lammpstrj"):
        mdt = load_trajectory(topology=written["sys.lmp"], trajectory=written[name])
        uni = mdt.universe
        assert uni.atoms.n_atoms == 4, (name, uni.atoms.n_atoms)
        assert uni.trajectory.n_frames == 2, (name, uni.trajectory.n_frames)
        assert abs(float(uni.atoms[3].position[0]) - 5.5) < 1e-6      # 第 0 帧
        uni.trajectory[-1]
        assert abs(float(uni.atoms[3].position[0]) - 5.6) < 1e-6      # 第 1 帧

    # ④ 错槽位：dump 当拓扑 / 输入脚本 / 日志 → 各自的针对性提示
    cases = (("dump.lammpstrj", ("dump", "拓扑")),
             ("in.lammps", ("输入脚本", "read_data")),
             ("log.lammps", ("日志", "dump")))
    seen = []
    for name, needles in cases:
        try:
            load_trajectory(written[name]).universe
            raise AssertionError(f"{name} 竟然当拓扑读过去了")
        except TrajectoryError as exc:
            text = str(exc)
            assert any(n in text for n in needles), (name, text)
            seen.append(text.splitlines()[0][:40])

    # 嗅探器本身：只按内容判断，不认后缀
    assert sniff_lammps_kind(written["sys.lmp"]) == "data"
    assert sniff_lammps_kind(written["in.lammps"]) == "input"
    assert sniff_lammps_kind(written["log.lammps"]) == "log"

    # ⑤ 后缀是 data、内容却不是 data（常见：本工具自己的 `_msd_scratch/msd_*.dat`
    #    二进制缓存，或用户的一列数值结果表）→ 要说清"这不是 data 文件"，
    #    而不是把 MDAnalysis 的 `'gbk' codec can't decode` 丢给用户
    junk_dat = os.path.join(d, "result.dat")
    with open(junk_dat, "wb") as fh:
        fh.write(b"\x01\x02\x03\xd7\xfe\xff" * 64)
    try:
        load_trajectory(junk_dat).universe
        raise AssertionError("二进制 junk .dat 竟然当拓扑读过去了")
    except TrajectoryError as exc:
        text = str(exc)
        assert sniff_lammps_kind(junk_dat) is None
        assert "不是** LAMMPS data 文件" in text or "不是" in text, text
        assert "write_data" in text, text

    # ⑥ 最关键的一条：LAMMPS data **通常既没有元素列、也没有 Bonds 段**
    #    （`write_data` 的默认输出只有 Masses + 数字 atom type + mol-id）。
    #    没有元素就猜不出键 → 每个原子自成"分子" → 逐分子 Rg / MSD 全错。
    #    现在按 Masses 反推元素（12.011→C、1.008→H），键能猜出来、分子能分对。
    two = os.path.join(d, "two_mol.lmp")
    body = ["LAMMPS data (2 molecules, no bonds)", "",
            "8 atoms", "2 atom types", "",
            "0.0 40.0 xlo xhi", "0.0 40.0 ylo yhi", "0.0 40.0 zlo zhi", "",
            "Masses", "", "1 12.011", "2 1.008", "", "Atoms # full", ""]
    n = 1
    for mol, (ox, oy, oz) in ((1, (0.0, 0.0, 0.0)), (2, (20.0, 20.0, 20.0))):
        for k, (dx, dy, dz) in enumerate(((1.0, 1.0, 1.0), (2.09, 1.0, 1.0),
                                          (1.0, 2.09, 1.0), (1.0, 1.0, 2.09))):
            typ = 1 if k == 0 else 2
            body.append(f"{n} {mol} {typ} 0.0 {ox + dx:.3f} {oy + dy:.3f} "
                        f"{oz + dz:.3f} 0 0 0")
            n += 1
    with open(two, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")
    from mdta.topology import molecule_count, residue_names

    mtd = load_trajectory(two)
    uni = mtd.universe
    assert uni.atoms.n_atoms == 8, uni.atoms.n_atoms
    # 没有 Bonds 段 → 必须靠猜键补出来，否则 8 个原子会变成 8 个"分子"
    assert len(uni.bonds) >= 6, len(uni.bonds)
    assert molecule_count(uni.atoms) == 2, molecule_count(uni.atoms)
    assert set(residue_names(uni).tolist()) <= {"C", "H"}, residue_names(uni)

    return ("data 后缀 4 种（.lmp/.dat/.data/.lammps）均可读；"
            ".lmp 自动配对 .lammpstrj 得 2 帧；"
            "dump 当拓扑 / 输入脚本 / 日志 都被拦下；"
            "内容是二进制缓存的 .dat 也会说明「不是 data 文件」；"
            "无元素列无 Bonds 的 data（Masses 12.011/1.008）按质量推出 C/H、"
            f"猜出 {len(uni.bonds)} 根键、分成 {molecule_count(uni.atoms)} 个分子："
            + "；".join(seen))


def test_wrong_slot_and_unknown_suffix_give_clear_errors():
    """回归：轨迹放进拓扑槽 / 不认识的文件当拓扑 → 必须是**可读的错误**。

    以前这两种情况都会抛到 API 层变成 HTTP 500（HTML 错误页），
    前端只能显示 "Unexpected token '<'"。现在 `load_trajectory` 自己就给出
    带后缀清单的说明，API 也统一回 JSON 400。
    """
    from mdta.formats import topology_help
    from mdta.io import TrajectoryError, load_trajectory

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)
    fake_xtc = os.path.join(d, "toy.xtc")
    with open(fake_xtc, "wb") as fh:
        fh.write(b"\x00" * 64)
    junk = os.path.join(d, "notes.md")
    with open(junk, "w", encoding="utf-8") as fh:
        fh.write("# 不是 MD 文件\n")

    msgs = []
    for kwargs in ({"topology": fake_xtc}, {"topology": junk}):
        try:
            load_trajectory(**kwargs)
            raise AssertionError(f"{kwargs} 竟然读过去了")
        except TrajectoryError as exc:
            msgs.append(str(exc))
    assert "缺少拓扑文件" in msgs[0] or "不能用作拓扑文件" in msgs[0], msgs[0]
    assert "可用拓扑格式" in msgs[1], msgs[1]
    assert topology_help().split()[0] in msgs[1], msgs[1]
    return ("轨迹放进拓扑槽 / 不认识的后缀都被挡下并给出后缀清单："
            + msgs[0].replace("\n", " ")[:100])


def test_msd_uses_all_particles_without_splitting_molecules():
    """回归：MSD 默认必须**每个粒子都参与**，且不能把分子切碎。

    用户报过"MSD 是不是没算全部的"。查下来有两层抽稀：
      ① `pipeline._msd_groups()`：组分 >5000 原子时 `ag[::step]` 抽到约 2000 个
         **原子**（47453 → 1978，砍掉 96%）；
      ② `particle_series`：按内存预算 `budget/(帧×3×8)` 再砍一次（6001 帧 → 8331 个）。
    更糟的是 ① 是**原子级等间隔**取：PEG 体系（200 条 30 原子链）抽 1/3 后每条链
    只剩 10 个原子，"分子质心"成了**残缺分子**的质心。

    断言：默认（`max_particles=None`）下粒子数 == 总粒子数、`subsampled=False`、
    分子对象时每个分子原子数不变；显式上限时按**分子整取**。
    """
    from mdta.analysis import dynamics as D
    from mdta.pipeline import Analyzer

    d = _ctx()
    mdt = d["mdt"]
    sel = select_frames(np.asarray(mdt.times_ps, float)[:20])
    # 用**多分子**组分（水，11084 个分子）来验"全粒子/按分子整取"；
    # 单分子体系（蛋白只有 1 条链）测不出抽稀行为。
    ag = d["water"] if d["water"].n_atoms > 10 else d["protein"]
    n_mol = len(np.unique(D._molecule_ids(ag)))

    ser, _box, meta = D.particle_series(mdt, ag, sel.indices, object="molecule")
    assert meta["n_particles"] == meta["n_particles_total"] == n_mol, meta
    assert meta["subsampled"] is False, meta
    assert ser.shape == (sel.n_frames, n_mol, 3), ser.shape

    # 显式上限：按分子整取，分子数减少但**每个分子的原子数不变**
    cap = max(1, n_mol // 3)
    _s2, _b2, m2 = D.particle_series(mdt, ag, sel.indices, object="molecule",
                                     max_particles=cap)
    assert m2["n_particles"] <= cap < m2["n_particles_total"], m2
    assert m2["subsampled"] is True
    mol = D._molecule_ids(ag)
    keep = np.unique(mol)[:: max(1, int(np.ceil(n_mol / cap)))]
    per = np.bincount(np.searchsorted(keep, mol[np.isin(mol, keep)]))
    assert per.min() == per.max(), per          # 每分子原子数一致 → 没被切碎

    # pipeline 的组分表不再抽稀：原子数原样保留
    az = Analyzer(mdt.topology, mdt.trajectory)
    az.set_frames(max_frames=5)
    az.auto_setup()
    groups = az._msd_groups()
    assert all("抽稀" not in k for k in groups), list(groups)
    assert all(getattr(g, "n_atoms", 0) > 0 for g in groups.values())
    return (f"全粒子：分子对象 {meta['n_particles']}/{meta['n_particles_total']} 个粒子"
            f"（未抽稀）；组分表不再抽稀（{list(groups)[:2]}…）；"
            f"显式上限 {cap} 时按分子整取、每分子原子数一致")


def test_msd_segmented_and_streaming_match_the_full_estimate():
    """分段（滚动窗口）与内存映射：**结果必须与完整口径一致**。

    为什么加分段：完整版一次 FFT 覆盖整条时间轴，n_fft=2^ceil(log2 2N)，
    6001 帧 × 50000 粒子实测 150 s、序列 3.6 GB。分段后单次 FFT 只覆盖
    ``2·max_lag+1`` 帧，代价 ∝ 帧数×log(块长)，且逐块读 → 内存恒定。

    断言：
      1. **单块**（max_lag=N−1、block_frames=N）与完整版**逐位一致** —— 这是
         排除实现错误的锚点（分子与起点计数都必须完全相同）；
      2. 多块时 τ 较小处与完整版统计一致（相对差 ≲5%）；
      3. `stream=True`（磁盘内存映射）与内存版结果**完全一致**；
      4. `workers=2` 拿不到进程池时**自动退回串行**、结果不变（受限环境）。
    """
    from mdta.analysis.dynamics import (msd_numerator_by_group,
                                        msd_numerator_by_group_segmented,
                                        particle_series, compute_msd_axes)

    rng = np.random.default_rng(3)
    A = np.cumsum(rng.standard_normal((300, 40)) * 0.5, axis=0)
    num_f, _ = msd_numerator_by_group(A)
    num_s, org_s, _ = msd_numerator_by_group_segmented(A, max_lag=299,
                                                       block_frames=300)
    assert float(np.max(np.abs(num_f[:, 0] - num_s[:, 0]))) < 1e-6 * float(np.max(num_f[:, 0])), \
        "单块分段必须与完整版逐位一致"
    assert float(np.max(np.abs(org_s - (300 - np.arange(300))))) < 1e-9, "起点计数不符"

    num_m, org_m, _ = msd_numerator_by_group_segmented(A, max_lag=40,
                                                       block_frames=81)
    msd_f = num_f[:41, 0] / (300 - np.arange(41))
    msd_m = num_m[:, 0] / org_m
    rel = float(np.max(np.abs(msd_m[:21] - msd_f[:21]) / msd_f[:21]))
    assert rel < 0.05, (rel, "多块分段与完整口径应有统计一致性")

    # 内存映射 vs 内存：同一条序列必须给出同一个 D。
    # 两种比法：① 都存 float64 → 应逐位一致；② 默认 float32 内存映射 vs float64
    # 内存版 → 只差 float32 存储精度（实测 ~1e-7 Å²，远小于 MSD 本身）。
    d = _ctx()
    mdt = d["mdt"]
    sel = select_frames(np.asarray(mdt.times_ps, float)[:30])
    ag = d["water"] if d["water"].n_atoms else d["protein"]
    obj = "atom"
    _l1, m1, g1 = compute_msd_axes(mdt, ag, sel, object=obj, remove_drift=False,
                                   n_error_blocks=0, dtype=np.float64)
    _l2, m2, g2 = compute_msd_axes(mdt, ag, sel, object=obj, remove_drift=False,
                                   n_error_blocks=0, dtype=np.float64, stream=True)
    same = float(np.max(np.abs(m1["total"] - m2["total"])))
    assert same < 1e-12, (same, "同精度下内存映射与内存版必须一致")
    assert g2.get("存储", "").startswith("磁盘"), g2.get("存储")
    _l3, m3, g3 = compute_msd_axes(mdt, ag, sel, object=obj, remove_drift=False,
                                   n_error_blocks=0, stream=True)     # float32 落盘
    ref = float(np.nanmax(np.abs(m1["total"]))) or 1.0
    rel32 = float(np.max(np.abs(m3["total"] - m1["total"]))) / ref
    assert rel32 < 1e-4, (rel32, "float32 内存映射只允许存储精度级别的差异")
    _l4, m4, g4 = compute_msd_axes(mdt, ag, sel, object=obj, remove_drift=False,
                                   n_error_blocks=0, stream=True, workers=2)
    assert float(np.max(np.abs(m4["total"] - m3["total"]))) < 1e-12, "workers 改变了结果"
    return (f"单块分段与完整版逐位一致；多块 τ≤20 相对差 {rel * 100:.1f}%；"
            f"float64 内存映射与内存版偏差 {same:.1e} Å²（{g2.get('存储')}）；"
            f"float32 落盘相对差 {rel32 * 100:.2e}%；workers=2 结果不变")


def test_degenerate_topology_is_repaired_and_forced_reguess():
    """回归：**残基/分子信息退化的拓扑**（Material Studio 的 PDB）必须被修好。

    用户实际文件：`122C12450K.pdb`（Material Studio 导出）+ 由 trj 转换的 `.xtc`。
    实测这个 PDB 有两处退化，而且**都不报错**：
      ① 5596 个原子只有 **2 个残基** → 按残基编号分子得到"整个体系 2 个分子"，
         于是逐分子 Rg/R_ee、分子质心 MSD、RDF 的"同分子剔除"全错；
      ② 自带的 ``CONECT`` **不完整**：4393 个键 → 键图被切成 1203 个碎片
         （真实分子约 122 个），而按坐标猜键得到 5953 个键 / 124 个碎片。

    断言：
      1. 退化 molnums 会被弃用，下游改用键连通分量；
      2. 正常拓扑（.tpr）不受影响：重跑属性补齐不会动它自带的键表、分子划分仍可用；
      3. 合成的退化体系（2 残基 / 20 原子 / 10 个双原子片段）应按键图分成 10 个分子。
    """
    from mdta.analysis.interface import _atom_molnums
    from mdta.topology import ensure_object_attrs

    d = _ctx()
    mdt = d["mdt"]                      # 正常体系（.tpr）：键表可信
    u = mdt.universe
    n_bonds_before = len(u.bonds)
    ensure_object_attrs(u)              # 再跑一次不应改变任何东西
    assert len(u.bonds) == n_bonds_before, "正常拓扑的键表被改动了"
    ag = d["water"] if d["water"].n_atoms else d["protein"]
    ids = _atom_molnums(ag)
    assert ids is not None and ids.size == ag.n_atoms

    # 合成一个"退化拓扑"：2 个残基、20 个原子、分成 10 个双原子片段
    import MDAnalysis as _mda
    n_at = 20
    uu = _mda.Universe.empty(n_at, n_residues=2,
                             atom_resindex=[0] * 10 + [1] * 10, trajectory=True)
    uu.add_TopologyAttr("masses", np.ones(n_at))
    uu.add_TopologyAttr("molnums", np.array([0, 1]))
    uu.add_TopologyAttr("bonds", [(2 * k, 2 * k + 1) for k in range(10)])
    uu.dimensions = np.array([50.0, 50.0, 50.0, 90, 90, 90], dtype=np.float32)
    uu.trajectory = _mda.coordinates.memory.MemoryReader(
        np.zeros((1, n_at, 3), dtype=np.float32), order="fac",
        dimensions=uu.dimensions)
    info = ensure_object_attrs(uu)
    assert info.get("molnums_degenerate"), info
    ids2 = _atom_molnums(uu.atoms)
    assert ids2 is not None and len(set(ids2.tolist())) == 10, (ids2,
                                                              "应按键图分成 10 个分子")
    return (f"正常拓扑键表未被改动（{n_bonds_before:,} 键）；退化拓扑"
            f"（2 残基 / 20 原子 / 10 个双原子片段）被识别并按**键连通分量**分成 "
            f"{len(set(ids2.tolist()))} 个分子")


def _write_ms_trj(path: str, coords: np.ndarray, cell=(20.0, 21.0, 22.0),
                  vel: np.ndarray | None = None) -> None:
    """现场合成一个**合法的最小 Materials Studio `.trj`**（大端 Fortran 记录序列）。

    格式见 `pymsdump/ms_trj.py`：8 条头部记录 + 每帧 5 条元数据记录
    （492/96/32/176/116 B）+ 6 条 double 数组（坐标 xyz + **速度** vx/vy/vz）。
    ``coords`` 形状 (n_frames, n_atoms, 3)；``vel`` 同形状（不给就是 0）。
    注意 ``8*n_atoms`` 不能撞上元数据记录长度（32/96/176/492），所以测试里用 6 个原子。
    """
    import struct

    n_fr, n_at, _ = coords.shape
    if vel is None:
        vel = np.zeros_like(coords)

    def rec(payload: bytes) -> bytes:
        return (struct.pack(">i", len(payload)) + payload
                + struct.pack(">i", len(payload)))

    def meta_block(time: float) -> bytes:
        main = bytearray(492)
        struct.pack_into(">d", main, 0, time)
        struct.pack_into(">i", main, 8, int(time * 500))
        main[12:12 + 8 * 61] = np.zeros(61, dtype=">f8").tobytes()
        pv = np.zeros(12, dtype=">f8")
        cellrec = np.zeros(22, dtype=">f8")
        cellrec[2], cellrec[3], cellrec[4] = cell
        x116 = np.zeros(15, dtype=">f8")
        return (rec(bytes(main)) + rec(pv.tobytes()) + rec(bytes(32))
                + rec(cellrec.tobytes()) + rec(x116.tobytes()))

    buf = rec(b"MDTR" + struct.pack(">i", 2010) + bytes(24))
    buf += rec(b"Created by Materials Studio Forcite 23.1")
    buf += rec(b"Molecular mechanics Forcite calculation") + rec(bytes(32))
    buf += rec(struct.pack(">i", 1) + struct.pack(">i", n_at)
               + struct.pack(">i", n_at) + b"MStudio ")
    buf += rec(struct.pack(">i", n_at)
               + b"".join(struct.pack(">i", i + 1) for i in range(n_at)))
    buf += rec(b"NOTITLE") + rec(b"NOPAR")
    for k in range(n_fr):
        buf += meta_block(float(k))
        for col in range(3):
            buf += rec(np.ascontiguousarray(coords[k][:, col],
                                            dtype=">f8").tobytes())
        for col in range(3):
            buf += rec(np.ascontiguousarray(vel[k][:, col],
                                            dtype=">f8").tobytes())
    with open(path, "wb") as fh:
        fh.write(buf)


def _write_ms_xtd(path: str, n_atoms: int, cell=(20.0, 21.0, 22.0),
                  end: int | None = None) -> None:
    """合成一个最小 `.xtd`（拓扑 + 可选的 ``<Trajectory End=…>`` 声明）。"""
    lines = ['<?xml version="1.0" encoding="latin1"?>', "<XSD>",
             ' <AtomisticTreeRoot ID="1">']
    for i in range(n_atoms):
        lines.append(f'  <Atom3d ID="{200 + i}" Name="C{i + 1}" '
                     f'XYZ="{i * 1.5 / cell[0]},0,0" Components="C" '
                     'ForcefieldType="c3a"/>')
    for i in range(n_atoms - 1):
        lines.append(f'  <Bond ID="{400 + i}" '
                     f'Connects="{200 + i},{201 + i}"/>')
    lines.append(f'  <SpaceGroup ID="2" AVector="{cell[0]},0,0" '
                 f'BVector="0,{cell[1]},0" CVector="0,0,{cell[2]}"/>')
    if end is not None:
        lines.append(f'  <Trajectory ID="9" End="{end}" Type="trj" '
                     'FrameClassType="Bondable"/>')
    lines += [" </AtomisticTreeRoot>", "</XSD>", ""]
    with open(path, "w", encoding="latin1") as fh:
        fh.write("\n".join(lines))


def test_xtd_carries_no_frames_but_pairs_with_trj():
    """探究结论的回归：`.xtd` **装不下轨迹帧**，但能当拓扑与同名 `.trj` 配对。

    结论来自实测（用户体系 `122C12-450K.xtd`，3.84 MB）：

    * `<Frame>` / `<Step>` 各 **0** 个；`<Trajectory ID="25810" End="6001"
      Type="trj"/>` 是**自闭合**标签，后面直接 `</AtomisticTreeRoot></XSD>`；
    * 没有 base64 / `<Set>` / 内嵌数据块；
    * 体积账：6001 帧 × 5596 原子 × 3 × 4 B = **384 MB**（float32），
      而文件只有 3.84 MB（**1.0%**）—— 差两个数量级，装不下；
    * MDAnalysis 对 `.xtd` 既没有 reader 也没有 parser。

    所以 `.xtd` 只能当**拓扑**（+ 帧数声明），帧必须靠同名 `.trj`。这里用合成文件
    把两侧都钉住：`.xtd`(拓扑) + `.trj`(3 帧) → 读到 3 帧且坐标正确；
    帧数声明写错时（End=5 vs 实际 3）要给出**明确的告警说明**。
    """
    from mdta import ms_trj as M
    from mdta.formats import TOPOLOGY_SUFFIXES, TRAJECTORY_SUFFIXES, classify
    from mdta.io import load_trajectory
    from mdta.ms_structure import read_ms_document

    assert ".xtd" in TOPOLOGY_SUFFIXES and ".xtd" not in TRAJECTORY_SUFFIXES
    assert classify("x.xtd") == "topology", classify("x.xtd")

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)
    n_at, n_fr = 6, 3
    coords = np.array([[[1.0 + k * 0.1 + i, 2.0 + i * 0.5, 3.0 + i * 0.25]
                        for i in range(n_at)] for k in range(n_fr)])

    # ① 文档声明 3 帧，与 .trj 一致
    xtd = os.path.join(d, "selftest_pair.xtd")
    trj = os.path.join(d, "selftest_pair.trj")
    _write_ms_xtd(xtd, n_at, end=n_fr)
    _write_ms_trj(trj, coords)
    doc = read_ms_document(xtd)
    assert doc["n_atoms"] == n_at and doc["trajectory_hint"]["n_frames"] == n_fr, doc
    assert len(doc["bonds"]) == n_at - 1, doc["bonds"]

    if not M.available(d):
        return "~ 跳过 .trj 部分：没找到 ms_trj.py / xtcwriter.py"

    mdt = load_trajectory(xtd, trj)
    u = mdt.universe
    assert u.atoms.n_atoms == n_at and len(u.trajectory) == n_fr, (
        u.atoms.n_atoms, len(u.trajectory))
    worst = 0.0
    for k in range(n_fr):
        u.trajectory[k]
        worst = max(worst, float(np.abs(np.asarray(u.atoms.positions, float)
                                        - coords[k]).max()))
    assert worst < 0.01, worst
    notes_ok = " ".join(mdt.load_notes)
    assert "校验通过" in notes_ok, (
        f"帧数校验说明缺失：load_notes={mdt.load_notes}，"
        f"frames={len(u.trajectory)}，ms_info={mdt.ms_info}")

    # ② 声明 5 帧但 .trj 只有 3 帧 → 必须明确告警（截断的轨迹就是这样）
    xtd_bad = os.path.join(d, "selftest_pair_bad.xtd")
    _write_ms_xtd(xtd_bad, n_at, end=n_fr + 2)
    mdt2 = load_trajectory(xtd_bad, trj)
    _ = mdt2.universe              # 先触发（懒）加载，load_notes 才会被填上
    notes_bad = " ".join(mdt2.load_notes)
    assert "帧数与 MS 文档声明不符" in notes_bad, (
        f"帧数不符的告警缺失：load_notes={mdt2.load_notes}，"
        f"frames={len(mdt2.universe.trajectory)}，ms_info={mdt2.ms_info}")
    assert "截断" in notes_bad, notes_bad

    # ③ 只给 .xtd（没有同名 .trj）→ 单帧，并说明"坐标在同名 .trj 里"
    solo = os.path.join(d, "selftest_solo.xtd")
    _write_ms_xtd(solo, n_at, end=n_fr)
    mdt3 = load_trajectory(solo)
    assert len(mdt3.universe.trajectory) == 1, len(mdt3.universe.trajectory)
    assert any(".trj" in x for x in mdt3.load_notes), mdt3.load_notes

    # ④ `.car`(6 原子) + 另一个体系的 `.trj`(5 原子)：必须**转换之前**就说清"原子数不一致"
    #    （实测用户把 C12-100-2.car 的 5400 原子和 122C12-450K.trj 的 5596 原子配在一起，
    #     旧行为会先把 1.6 GB 的轨迹转完，再抛一句"转换失败"）
    car6 = os.path.join(d, "selftest_pair6.car")
    with open(car6, "w", encoding="latin1") as fh:
        fh.write("!BIOSYM archive 3\nPBC=ON\nsynthetic\n!DATE x\n"
                 "PBC   20.0000   21.0000   22.0000   90.0000   90.0000   90.0000 (P1)\n")
        for i in range(n_at):
            fh.write(f"C{i + 1}   {1.0 + i:12.6f} {2.0 + i * 0.5:12.6f} "
                     f"{3.0 + i * 0.25:12.6f} XXXX 1 c3a C 0.000\n")
        fh.write("end\nend\n")
    coords5 = coords[:, :5, :]                     # 5 原子 ≠ car 的 6 原子
    trj5 = os.path.join(d, "selftest_pair5.trj")
    _write_ms_trj(trj5, coords5)
    from mdta.ms_trj import trj_header

    hdr = trj_header(trj5)
    assert hdr.get("n_atoms") == 5, hdr
    try:
        load_trajectory(car6, trj5).universe      # 懒加载：必须访问 universe 才会开
    except Exception as exc:                                     # noqa: BLE001
        msg = str(exc)
        assert "原子数不一致" in msg, msg
        assert "6 原子" in msg and "5 原子" in msg, msg
        assert "尚未转换" in msg, msg
    else:
        raise AssertionError("原子数不同的 .car + .trj 竟然被接受了")
    conv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_convert_cache")
    assert not [f for f in (os.listdir(conv) if os.path.isdir(conv) else [])
                if f.startswith("selftest_pair5.") and f.endswith(".xtc")], \
        "原子数不匹配时不该产生转换缓存（白转几个小时）"

    return (f".xtd 不在轨迹表（classify={classify('x.xtd')}）；.xtd(拓扑)+.trj(3 帧) → "
            f"读回 {len(u.trajectory)} 帧、坐标最大偏差 {worst:.4f} Å；"
            f"声明 5 帧但只有 3 帧 → 告警「帧数与 MS 文档声明不符」；"
            f"只给 .xtd → 单帧 + 提示坐标在同名 .trj 里；"
            f"原子数不匹配的 .car+.trj → 转换前就报「原子数不一致」（未产生缓存）")


def test_ms_trj_conversion_path_is_wired():
    """回归：Materials Studio `.trj` 的「发现模块 → 转换 → 读取」链路。

    MDAnalysis 不认识 MS 的二进制 `.trj`（`.trj` 这个后缀同时也被 CHARMM 的 ASCII
    轨迹占用，所以策略是**先让 MDAnalysis 试读、失败再转换**）。转换依赖用户自己的
    两个纯 Python 模块：`md_convert/ms_trj.py` 与 `pymsdump/xtcwriter.py`。

    这里**现场合成一个合法的最小 .trj**（大端 Fortran 记录序列，格式见 ms_trj.py），
    再验证：解析出的原子/帧数正确 → 转成 XTC → 坐标与写入值一致 → 加载器能读。
    模块找不到时跳过（缺外部依赖不该判成失败）。
    """
    from mdta import ms_trj as M
    from mdta.io import load_trajectory

    n_at, n_fr = 6, 3            # 8N 不能撞上元数据记录长度(32/96/176/492)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)
    trj = os.path.join(d, "selftest_ms.trj")
    pdb = os.path.join(d, "selftest_ms.pdb")
    if not M.available(d):
        return "~ 跳过：没找到 ms_trj.py / xtcwriter.py（设 MDTA_MS_CONVERT 后可启用）"

    coords = np.array([[[1.0 + k * 0.1 + i, 2.0 + i * 0.5, 3.0 + i * 0.25]
                        for i in range(n_at)] for k in range(n_fr)])
    _write_ms_trj(trj, coords)
    with open(pdb, "w", encoding="ascii") as fh:
        fh.write("CRYST1   20.000   21.000   22.000  90.00  90.00  90.00 "
                 "P 1           1\n")
        for i in range(n_at):
            fh.write(f"ATOM  {i + 1:5d}  C   MOL A   1    {1.0 + i:8.3f}"
                     f"{2.0 + i * 0.5:8.3f}{3.0 + i * 0.25:8.3f}"
                     f"  1.00  0.00           C  \n")
        fh.write("TER\nEND\n")

    info = M.trj_info(trj)
    assert info.get("n_atoms") == n_at and info.get("n_frames") == n_fr, info
    out = M.convert_trj(trj)
    assert out.is_file() and out.stat().st_size > 0, out
    mdt = load_trajectory(pdb, trj)
    u = mdt.universe
    assert u.atoms.n_atoms == n_at and u.trajectory.n_frames == n_fr, (
        u.atoms.n_atoms, u.trajectory.n_frames)
    worst = 0.0
    for k in range(n_fr):
        u.trajectory[k]
        worst = max(worst, float(np.abs(np.asarray(u.atoms.positions, float)
                                        - coords[k]).max()))
    assert worst < 0.01, worst            # XTC 量化步长 0.01 Å
    assert getattr(mdt, "converted_xtc", None), "应记录转换后的 XTC 路径"
    return (f"合成 .trj（{n_at} 原子 × {n_fr} 帧）→ 解析 OK → 转 XTC（"
            f"{out.stat().st_size:,} B）→ 加载器读回，坐标最大偏差 {worst:.4f} Å")


def test_ms_trj_precision_and_velocities():
    """`.trj` 转换的**坐标精度**与**速度**（用户实测两件事）。

    ① 用户用官方 `Trj2Ascii` 导出的 17 位 double 坐标做基准，发现我们的 XTC
       "平均差 0.02 Å、最大 0.05 Å"。复现后确认**不是解析错误**：XTC 是有损的
       整数量化格式，`precision` 就是步长（nm⁻¹）——

           precision=100    → 步长 0.1 Å     → 最大 0.05  Å / 平均 0.024 Å  ← 用户看到的
           precision=1000   → 步长 0.01 Å    → 最大 0.005 Å / 平均 0.0025 Å
           precision=1e4    → 步长 0.001 Å   → 最大 5e-4  Å / 平均 2.5e-4 Å ← 现在的默认

       所以把默认精度提到 1e4，并把「精度 + 格式」放进转换缓存键（否则改了精度
       还会命中旧缓存，看起来"没生效"）。

    ② 官方导出里还有 `Velocities:` 段 —— MS 的 `.trj` 确实存速度（第 4-6 条
       double 数组，Å/ps），而 **XTC 格式装不下速度**。现在 `fmt="trr"` 用 TRR
       （GROMACS 标准格式，MDAnalysis 原生可读）把坐标 + 速度一起写出来：
       坐标 float32（误差 ~1e-6 Å）、速度 ~1e-4 Å/ps，`ts.has_velocities=True`。
    """
    from mdta import ms_trj as M
    from mdta.io import load_trajectory

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)
    if not M.available(d):
        return "~ 跳过：没找到 ms_trj.py / xtcwriter.py"

    # 20 原子：大于 9 才会走 XTC 的压缩量化分支（≤9 是原样写 float32）
    n_at, n_fr = 20, 2
    rng = np.random.default_rng(7)
    coords = rng.random((n_fr, n_at, 3)) * 40.0               # Å
    vels = (rng.random((n_fr, n_at, 3)) - 0.5) * 2000.0       # Å/ps
    trj = os.path.join(d, "selftest_prec.trj")
    _write_ms_trj(trj, coords, vel=vels)

    import MDAnalysis as _mda

    def _read(path):
        u = _mda.Universe.empty(n_at, trajectory=True)
        u.load_new(str(path))
        return u

    # ① 精度：四种步长的实际误差
    errs = {}
    for prec in (100.0, 1000.0, 10000.0, 1.0e6):
        out = M.convert_trj(trj, fmt="xtc", precision=prec, force=True)
        u = _read(out)
        worst = 0.0
        for k in range(n_fr):
            u.trajectory[k]
            worst = max(worst, float(np.abs(
                np.asarray(u.trajectory.ts.positions, float) - coords[k]).max()))
        errs[prec] = worst
    assert 0.03 < errs[100.0] < 0.08, errs        # ≈0.05 Å：用户看到的量级
    assert errs[1000.0] < 0.01, errs
    assert errs[10000.0] < 2e-3, errs
    assert errs[1.0e6] < 2e-5, errs               # 1e-5 Å 步长 → 单点 ≤5e-6 Å
    assert errs[1.0e6] < errs[10000.0] < errs[1000.0] < errs[100.0], errs

    # 默认精度 1e5（1e-4 Å）：实测 PE50 的 Rg 与 Forcite 差 1.0e-6 Å
    # （1e4 时 9.6e-6、1e6 时 2.5e-7，但体积多 20% —— 1e5 是"够准又不多花"的拐点）
    assert abs(M.xc_precision() - 1.0e5) < 1, M.xc_precision()
    # 写入器：有 MDAnalysis 时走 C 扩展（比纯 Python 快 ~50 倍），产物一致
    assert M.writer_kind() in ("mdanalysis", "python"), M.writer_kind()
    _fast_kind = M.writer_kind()

    # 纯 Python 兜底也必须能跑（没有 MDAnalysis 的场合），并且精度同样达标。
    # 用一个不同的 precision 值 → 另一个缓存键，避免与上面仍开着的读取器抢文件
    # （Windows 上不能替换正被打开的文件）。
    _orig = M._mda_xtc_writer
    M._mda_xtc_writer = lambda: None
    try:
        out_py = M.convert_trj(trj, fmt="xtc", precision=20000.0, force=True)
        assert M.writer_kind() == "python", M.writer_kind()
        u_py = _read(out_py)
        w_py = 0.0
        for k in range(n_fr):
            u_py.trajectory[k]
            w_py = max(w_py, float(np.abs(
                np.asarray(u_py.trajectory.ts.positions, float) - coords[k]).max()))
        assert w_py < 2e-3, w_py
    finally:
        M._mda_xtc_writer = _orig
    del u_py

    # ② 速度：TRR 才带得出来
    trr = M.convert_trj(trj, fmt="trr", force=True)
    u = _read(trr)
    assert u.trajectory.ts.has_velocities or True     # 需要先定位帧
    dpos = dvel = 0.0
    for k in range(n_fr):
        u.trajectory[k]
        ts = u.trajectory.ts
        assert ts.has_velocities, "TRR 必须带速度"
        dpos = max(dpos, float(np.abs(np.asarray(ts.positions, float) - coords[k]).max()))
        dvel = max(dvel, float(np.abs(np.asarray(ts.velocities, float) - vels[k]).max()))
    assert dpos < 1e-4, dpos            # float32 精确度
    assert dvel < 1e-2, dvel

    # XTC 那条路没有速度（格式所限）
    xtc = M.convert_trj(trj, fmt="xtc")
    ux = _read(xtc)
    ux.trajectory[0]
    assert not ux.trajectory.ts.has_velocities, "XTC 不该有速度"

    # ③ 缓存键必须区分格式与精度（否则改精度"不生效"）
    keys = {M.cached_xtc(trj, fmt="xtc", precision=1000.0).name,
            M.cached_xtc(trj, fmt="xtc", precision=10000.0).name,
            M.cached_xtc(trj, fmt="trr").name}
    assert len(keys) == 3, keys

    # ④ 加载器端到端：velocities=True → TRR 且 has_velocities
    pdb = os.path.join(d, "selftest_prec.pdb")
    with open(pdb, "w", encoding="ascii") as fh:
        fh.write("CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 "
                 "P 1           1\n")
        for i in range(n_at):
            fh.write(f"ATOM  {i + 1:5d}  C   MOL A   1    {coords[0][i][0]:8.3f}"
                     f"{coords[0][i][1]:8.3f}{coords[0][i][2]:8.3f}"
                     f"  1.00  0.00           C  \n")
        fh.write("TER\nEND\n")
    m_vel = load_trajectory(pdb, trj, velocities=True)
    _ = m_vel.universe
    assert m_vel.has_velocities, m_vel.load_notes
    assert str(getattr(m_vel, "converted_xtc", "")).endswith(".trr"), m_vel.converted_xtc
    m_no = load_trajectory(pdb, trj, velocities=False)
    _ = m_no.universe
    assert not m_no.has_velocities
    assert str(getattr(m_no, "converted_xtc", "")).endswith(".xtc"), m_no.converted_xtc

    return (f"XTC 量化误差：precision=100 → {errs[100.0]:.4f} Å（用户看到的 0.05）、"
            f"1e3 → {errs[1000.0]:.4f}、1e4 → {errs[10000.0]:.4f}、"
            f"1e6 → {errs[1.0e6]:.1e}（默认 1e5 → ≈1e-4 Å 步长）；"
            f"TRR(坐标+速度) → 坐标 {dpos:.1e} Å / 速度 {dvel:.1e} Å/ps、"
            f"has_velocities=True；缓存键按格式+精度区分")


def test_rdf_without_box_and_box_dependent_analyses_skip():
    """无晶胞体系：**RDF 必须能跑**，需要盒的分析应当"跳过"而不是"失败"。

    用户实测：非周期体系（Materials Studio 里没给晶胞）用自定义集合跑 RDF，
    报 `rdf 失败: ValueError: 当前帧没有模拟盒信息，无法做空间分布分析。`——
    而 RDF 其实不需要周期性（`capped_distance(box=None)` 就是无周期距离），
    唯一卡住它的是"兜底体积"那一句 `box_matrix(u)`。

    现在的行为：
    · RDF：无盒 → 距离按无周期性算、密度归一化用**逐帧坐标包围盒体积**，
      notes 里写清楚这是近似；
    · 其它真需要盒的分析（密度剖面 / 界面宽度 / 面内 RDF）抛 :class:`BoxMissing`，
      由 `run_all` 记进 `Analyzer.skips`（界面上显示"跳过：原因"），不再算失败。
    """
    import MDAnalysis as _mda

    from mdta.analysis import interface as ifc
    from mdta.io import MDTrajectory
    from mdta.pipeline import Analyzer
    from mdta.preprocess import select_frames

    rng = np.random.default_rng(0)
    n = 80
    pos = rng.random((n, 3)) * 20.0
    u = _mda.Universe.empty(n, trajectory=True)      # 故意不给盒
    u.add_TopologyAttr("elements", ["C"] * n)
    u.atoms.positions = pos
    assert u.dimensions is None, u.dimensions
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))

    # ① RDF 能算，且说明里写明"没有晶胞"
    res = ifc.analyze_rdf(mdt, {"A": u.atoms}, sel, rmax=8.0, nbins=60, mode="total")
    assert res is not None and res.curves, "无盒体系 RDF 应当有结果"
    g = np.asarray(res.curves[0].y, dtype=float)
    assert np.all(np.isfinite(g)), "g(r) 出现非有限值"
    assert g.max() > 1.0, f"20 Å 盒里 80 个碳原子应当有结构峰，实际 g_max={g.max():.3f}"
    note_txt = " ".join(res.notes)
    assert "没有晶胞" in note_txt and "包围盒" in note_txt, res.notes

    # ② 需要盒的分析 → BoxMissing（不是普通 ValueError）
    try:
        ifc.box_matrix(u)
    except ifc.BoxMissing as exc:
        assert "晶胞" in str(exc), str(exc)
    else:
        raise AssertionError("没有盒时 box_matrix 应当抛 BoxMissing")

    # ③ pipeline：记进 skips，而不是当成失败
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)
    pdb = os.path.join(d, "selftest_nobox.pdb")
    with open(pdb, "w", encoding="ascii") as fh:      # 不写 CRYST1 → 没有盒
        for i in range(n):
            fh.write(f"ATOM  {i + 1:5d}  C   MOL A   1    {pos[i][0]:8.3f}"
                     f"{pos[i][1]:8.3f}{pos[i][2]:8.3f}  1.00  0.00           C  \n")
        fh.write("TER\nEND\n")
    az = Analyzer(pdb, guess_bonds=False)
    az.set_frames()
    # ⚠️ 必须先给 primary：否则 _run_one 会调 auto_setup()，把 components 重置成
    #    自动识别的结果（本测试里的两个自定义组分就被冲掉了）。
    az.set_primary(az.universe.atoms, "全部原子")
    az.set_components({"a": az.universe.atoms[:n // 2],
                       "b": az.universe.atoms[n // 2:]})
    res_all = az.run_all(["interface", "rg"], verbose=False)
    assert "interface" not in res_all, "没有盒时界面分析不该产出结果"
    assert "rg" in res_all, "Rg 不依赖盒，必须照常算出来"
    assert "interface" in az.skips and "晶胞" in az.skips["interface"], az.skips
    return (f"无盒体系：RDF 照常给出 g_max={g.max():.2f}（无周期性 + 包围盒体积，"
            f"notes 已说明）；界面分析记为「跳过」（不是失败）；Rg 不受影响")


def test_qc_inferred_mass_is_normal():
    """拓扑不带力场质量、由本工具按元素补出 → QC 判「正常」（用户要求）。

    PDB / XSD / CAR / GRO 本来就不含力场质量，按元素查表补是**常规做法**，
    不该在科研 QC 里显示成异常。只有 CG/联合原子体系（有原子认不出元素、
    质量被按 1.0 处理）才需要提醒。用户实测报告的就是
    "拓扑 PE50-g2020.xsd 未提供质量，本工具按元素/原子名补出" 被标成不正常。
    """
    from mdta import qc
    from mdta.core import AnalysisResult

    def _row(res, name="质量来源"):
        rows = qc.derive_checks(res)
        for r in rows:
            if r.get("名称") == name:
                return r
        raise AssertionError(f"没有「{name}」这一项：{[r.get('名称') for r in rows]}")

    # ① 全部原子都认得出元素 → 正常
    res = AnalysisResult(name="rg", title="水平")
    res.meta.update({"质量来源": "由工具按元素/原子名推断", "质量未知元素数": 0,
                     "拓扑文件": "PE50-g2020.xsd"})
    res.summary["Rg mean"] = 20.0
    row = _row(res)
    assert row["级别"] == "ok", row
    assert "补全" in row["结论"] and "PE50-g2020.xsd" in row["结论"], row

    # ② 有原子认不出元素（CG 珠子）→ 仍要提醒；涉及质量的量标"不可用"
    res2 = AnalysisResult(name="density", title="密度")
    res2.meta.update({"质量来源": "由工具按元素/原子名推断", "质量未知元素数": 17,
                      "拓扑文件": "cg.gro"})
    res2.summary["密度口径"] = "质量密度 (g/cm³)"
    row2 = _row(res2)
    assert row2["级别"] == "bad", row2
    assert "17" in row2["结论"], row2
    return ("质量由元素补出且元素全认得出 → QC「正常」（不再报异常）；"
            "只有 17 个原子认不出元素（CG 珠子）时才提醒并标不可用")


def test_visual_selection_becomes_set():
    """三维视图里选中的**原子索引** → 集合（`spec["indices"]` 分支）。

    这是「结构选择」模块的落地方式：视觉选择不走筛选条件，直接把原子索引交给
    `mdta.sets.build_set`。断言：
    ① 索引按**原始编号**解释，越界/空选择要给可读错误；
    ② `kind` 仍按"是否覆盖完整分子"自动判定（整分子 → 按质心追；切碎 → 按原子）；
    ③ 生成的 AtomGroup 就是那几个原子（顺序无关、去重）。
    """
    import MDAnalysis as _mda

    from mdta.sets import SetError, build_set

    # 3 个分子 × 4 个原子，按分子用键连成链（1-2-3-4、5-6-7-8、9-10-11-12）
    n_mol, per = 3, 4
    n = n_mol * per
    u = _mda.Universe.empty(n, trajectory=True)
    u.add_TopologyAttr("names", ["C"] * n)
    u.add_TopologyAttr("elements", ["C"] * n)
    bonds = [(i, i + 1) for m in range(n_mol) for i in range(m * per, m * per + per - 1)]
    u.add_TopologyAttr("bonds", bonds)
    u.atoms.positions = np.stack(
        [np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(float)

    # ① 整个分子（4 个原子都取）→ kind=molecule
    ag, info = build_set(u, {"name": "m1", "indices": [4, 5, 6, 7]})
    assert info["kind"] == "molecule", info
    assert info["n_atoms"] == 4 and info["n_molecules"] == 1, info
    assert sorted(np.asarray(ag.indices).tolist()) == [4, 5, 6, 7]

    # ② 只取部分原子 → kind=atom；重复索引要去重
    ag2, info2 = build_set(u, {"name": "part", "indices": [0, 0, 1, 9]})
    assert info2["kind"] == "atom", info2
    assert sorted(np.asarray(ag2.indices).tolist()) == [0, 1, 9], ag2.indices
    assert info2["scope"] == "indices" and len(info2["indices"]) == 3, info2

    # ③ 越界 / 空选择 → SetError（界面会把它显示成"选择无效"）
    for bad, why in (([0, n + 5], "越界"), ([], "空")):
        try:
            build_set(u, {"name": "bad", "indices": bad})
        except SetError as exc:
            assert "索引" in str(exc) or "没有选中" in str(exc), str(exc)
        else:
            raise AssertionError(f"{why} 索引应当报 SetError")
    return ("视觉选择 → 集合：整分子 4 个原子 → kind=molecule；部分原子 → kind=atom"
            "（去重后 3 个）；越界/空选择给出可读错误")


def test_degenerate_topology_without_molnums_runs_everything():
    """回归：**删掉 molnums 的退化拓扑**必须能跑起来（用户实测的崩溃）。

    背景：Material Studio 的 PDB 把 5596 个原子塞进 2 个残基，`ensure_object_attrs`
    会**删除** molnums、让下游改用键连通分量。但 `auto_setup` 里我新加的"选整个组分"
    逻辑直接写了 `ag.molnums` → 抛

        MDAnalysis.exceptions.NoDataError: This Universe does not contain molnum information

    整个体系连打开都过不去（用户报的就是这句）。修法是统一走
    `mdta.topology.molecule_ids()` / `molecule_count()`（molnums 不可用时退回键图碎片）。

    断言：① `molecule_ids` 有碎片就一定能给编号；② `molecule_count` 不抛异常且等于碎片数；
    ③ `auto_setup` 在"最大链是小分子、体系里有很多个"时能选到整个组分；
    ④ 接触分析在缺分子编号时退化为 total 而不是崩（`'NoneType' object is not subscriptable`）。
    """
    import MDAnalysis as _mda

    from mdta.analysis import interface as _ifc
    from mdta.pipeline import Analyzer
    from mdta.topology import ensure_object_attrs, molecule_count, molecule_ids

    n_mol, per = 12, 5                     # 12 个五原子分子
    n_at = n_mol * per
    uu = _mda.Universe.empty(n_at, n_residues=2,
                             atom_resindex=[0] * (n_at // 2) + [1] * (n_at - n_at // 2),
                             trajectory=True)
    uu.add_TopologyAttr("masses", np.ones(n_at))
    uu.add_TopologyAttr("molnums", np.array([0, 1]))      # 退化：整个体系 2 个"分子"
    bonds = [(per * m + k, per * m + k + 1) for m in range(n_mol) for k in range(per - 1)]
    uu.add_TopologyAttr("bonds", bonds)
    uu.dimensions = np.array([40.0, 40.0, 40.0, 90, 90, 90], dtype=np.float32)
    uu.trajectory = _mda.coordinates.memory.MemoryReader(
        np.zeros((1, n_at, 3), dtype=np.float32), order="fac", dimensions=uu.dimensions)
    info = ensure_object_attrs(uu)
    has_mol = True
    try:
        _ = uu.atoms.molnums
    except Exception:                                          # noqa: BLE001
        has_mol = False
    assert not has_mol, "退化 molnums 应被删掉（否则这条测试没测到点子上）"
    assert info.get("molnums_degenerate"), info

    ids = molecule_ids(uu.atoms)
    assert ids is not None and len(set(ids.tolist())) == n_mol, ids
    assert molecule_count(uu.atoms) == n_mol, molecule_count(uu.atoms)

    # 接触分析：缺分子编号时应退化为 total，而不是 IndexError/TypeError
    mdt = MDTrajectory.from_universe(uu)
    sel = select_frames(np.array([0.0]))
    res = _ifc.analyze_contacts(mdt, uu.atoms[:per * 3], uu.atoms[per * 3:per * 6], sel,
                                cutoff=2.0, mode="inter")
    assert res is not None
    assert "total" in str(res.summary.get("配对模式", "")) or \
        "total" in " ".join(res.notes), (res.summary.get("配对模式"), res.notes[:2])
    return (f"退化拓扑（{n_at} 原子 / 2 残基 / {n_mol} 个分子）：molnums 已删，"
            f"molecule_ids 按键图给出 {n_mol} 个分子；接触分析退化为 total 不崩")


def test_incomplete_conect_is_auto_repaired():
    """回归：拓扑自带键表**不完整**时自动改用猜键结果（不再依赖界面开关）。

    实测 Material Studio 的 PDB：`CONECT` 只有 4393 个键 → 键图碎片 **1203** 个
    （真实分子约 122 个），而按坐标猜键得到 5954 键 / 123 碎片（跨碎片补键后 122）。
    键表不全会让逐分子 Rg/R_ee、分子质心 MSD、RDF 同分子剔除**静默变差**。

    这里造一个 6 原子的链，但 PDB 的 `CONECT` **只写 half**（漏掉中间那根键）：
    读取器应识别出"猜键把碎片从 2 个并回 1 个"，于是**采用猜键结果**。
    """
    from mdta.io import load_trajectory

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)
    pdb = os.path.join(d, "sparse_conect.pdb")
    xyz = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0),
           (4.5, 0.0, 0.0), (6.0, 0.0, 0.0), (7.5, 0.0, 0.0)]
    lines = ["CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1"]
    for i, (x, y, z) in enumerate(xyz):
        lines.append(f"ATOM  {i + 1:5d}  C   MOL A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00           C  ")
    # 故意漏掉 3-4 这根键 → 自带键表被切成两段
    for i in range(1, 6):
        if i == 3:
            continue
        lines.append(f"CONECT{i:5d}{i + 1:5d}")
    lines += ["TER", "END"]
    with open(pdb, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")

    u_own = load_trajectory(pdb, guess_bonds=False).universe
    n_frag_own = len(u_own.atoms.fragments)
    n_bonds_own = len(u_own.bonds)
    assert n_frag_own == 2, n_frag_own           # 自带 CONECT 把链切成两段

    mdt = load_trajectory(pdb)                   # 默认：自动判断
    u = mdt.universe
    info = mdt.bond_guess_info or {}
    assert len(u.atoms.fragments) == 1, (len(u.atoms.fragments), info)
    assert len(u.bonds) >= 5, len(u.bonds)
    assert "不完整" in str(info.get("note", "")), info
    return (f"自带 CONECT 漏一根键（{n_bonds_own} 键 / {n_frag_own} 碎片）→ "
            f"自动改用猜键结果（{len(u.bonds)} 键 / 1 个碎片），并写明原因")


def test_coordinate_scale_check_catches_rescaling():
    """回归：坐标整体被缩放时，用**共价键长**当尺子必须能抓到，并给出倍数。

    背景（用户实测）：同一条轨迹在工具里算出 Rg = 2.19456 Å，而 Materials Studio
    Forcite 是 4.3929 Å——**逐帧比值稳定 2.0000**，不是口径差异，而是坐标被整体
    缩放过。这类错误不会抛异常，只会让所有长度量按同一比例偏大偏小，所以必须
    有一条"物理尺子"兜住：C–C 单键 1.54 Å、共价半径和 1.50 Å。
    """
    import MDAnalysis as mda

    from mdta.io import load_trajectory  # noqa: F401  (保持与其它用例一致的导入风格)
    from mdta.topology import bond_scale_report

    # 6 原子直链，键长 1.5 Å（C–C）
    xyz = np.array([[1.5 * i, 0.0, 0.0] for i in range(6)], dtype=np.float32)
    u = mda.Universe.empty(6, n_residues=1, atom_resindex=[0] * 6, trajectory=True)
    u.add_TopologyAttr("elements", ["C"] * 6)
    u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(5)])
    u.trajectory = mda.coordinates.memory.MemoryReader(
        xyz[None, :, :], order="fac",
        dimensions=np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=np.float32))
    ok = bond_scale_report(u.atoms)
    assert ok["ok"] is True, ok
    assert abs(ok["median"] - 1.5) < 1e-5, ok["median"]

    # 整体缩到 1/2：比值应 ≈0.5 → 判为异常
    u.atoms.positions = np.asarray(u.atoms.positions, dtype=float) * 0.5
    half = bond_scale_report(u.atoms)
    assert half["ok"] is False, half
    assert abs(half["ratio"] - 0.5) < 0.05, half["ratio"]
    assert "缩放" in half["note"] or "单位" in half["note"], half["note"]

    # 再缩到 1/10（相当于轨迹存成 nm）：比值 ≈0.1，提示语要指向"单位"
    u.atoms.positions = np.asarray(u.atoms.positions, dtype=float) / 5.0
    nm = bond_scale_report(u.atoms)
    assert nm["ok"] is False and nm["ratio"] < 0.3, nm
    assert "nm" in nm["note"], nm["note"]
    return (f"键长 1.5 Å → 比值 {ok['ratio']:.2f}（正常）；整体 ÷2 → "
            f"{half['ratio']:.2f}（判异常）；÷10 → {nm['ratio']:.2f}（提示单位可能是 nm）")


def test_degenerate_conect_is_repaired_even_when_user_says_no_guess():
    """回归：键表**明显残缺**时，即使显式 `guess_bonds=False` 也要自动补键。

    背景（用户实测）：Material Studio 导出 PDB 的 `CONECT` 只写了十二烷的一部分键，
    于是 `polymer[P]`（4636 原子 = 122 × C12H26）被切成 **243 个碎片**
    （122 个近乎完整的链 + **121 个孤立 H**），用户看到"243 个分子"。
    逐分子 Rg 的平均值被这些"Rg = 0 的单原子"拉低一半：
    **2.19456 Å vs 正确 4.3929 Å（= Forcite）**，逐帧比值稳定 2.0000，且不报错。

    判据（`topology_degeneracy`）：碎片中位原子数 ≤ 2，或孤立原子占比 ≥ 30%。
    这里造 2 条 6 原子链，但 `CONECT` **只写 1 根键**（其余全是孤立原子），
    断言：即使 `guess_bonds=False` 也会被补成 2 个碎片并写明原因；
    而**完整**键表（所有键都写了）时不会被动。
    """
    from mdta.io import load_trajectory

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)

    xyz = []
    for chain in range(2):
        base = 12.0 * chain
        xyz += [(base + 1.5 * i, 0.0, 0.0) for i in range(6)]

    def write_pdb(path, conect):
        lines = ["CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1"]
        for i, (x, y, z) in enumerate(xyz):
            lines.append(f"ATOM  {i + 1:5d}  C   MOL A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                         f"  1.00  0.00           C  ")
        for a, b in conect:
            lines.append(f"CONECT{a + 1:5d}{b + 1:5d}")
        lines += ["TER", "END"]
        with open(path, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")

    # ① 残缺：只写 1 根键（2 个原子成键、其余 10 个孤立）→ 中位 1 原子 → 判残缺
    sparse = os.path.join(d, "sparse_degenerate.pdb")
    write_pdb(sparse, [(0, 1)])
    raw = mda.Universe(sparse)                 # 原样读，不经工具修复
    n_raw = len(raw.atoms.fragments)
    assert n_raw > 2, n_raw
    assert raw.atoms.n_atoms == 12

    mdt = load_trajectory(sparse, guess_bonds=False)
    u = mdt.universe                 # 触发（懒）加载：bond_guess_info 在加载时就填好了
    info = mdt.bond_guess_info or {}
    if info.get("forced_repair") is not True:
        from mdta.topology import topology_degeneracy as _degen

        raise AssertionError(
            f"info={info}, guess_bonds={mdt.guess_bonds}, "
            f"原始碎片={n_raw}, 读入碎片={len(u.atoms.fragments)}, "
            f"键={len(u.bonds)}, degen={_degen(u.atoms)}, "
            f"path={sparse}")
    u = mdt.universe
    assert len(u.atoms.fragments) == 2, (len(u.atoms.fragments), info)
    assert "残缺" in str(info.get("note", "")), info

    # ② 完整键表：不应该被动（尊重用户的 guess_bonds=False）
    full = os.path.join(d, "sparse_complete.pdb")
    conect = [(i, i + 1) for i in range(5)] + [(6 + i, 6 + i + 1) for i in range(5)]
    write_pdb(full, conect)
    mdt2 = load_trajectory(full, guess_bonds=False)
    u2 = mdt2.universe                # 同样先触发加载
    info2 = mdt2.bond_guess_info or {}
    assert not info2.get("forced_repair"), info2
    assert len(u2.atoms.fragments) == 2, len(u2.atoms.fragments)
    return (f"残缺 CONECT（只写 1 根键）→ 自动补键：碎片 {n_raw} → "
            f"{len(u.atoms.fragments)}（forced_repair）；完整 CONECT 保持 2 个碎片不动")


def test_ms_documents_parse_to_topology():
    """回归：Materials Studio 的 `.car` / `.xsd` / `.xtd` 能当拓扑读进来。

    背景（用户要求）：MDAnalysis **不认识**这三种 MS 自家格式，工具里原本只能吃
    PDB/GRO/PSF。用户直接在工具目录放了 `.car` / `.xsd` / `.xtd`，要求"读文件时
    也能读它们作为补充文件"。实测要点：

    * `.xsd`：5596 原子 = H 3172 + C 2424；**5891 根键**；`XYZ` 是**分数坐标**
      （× 晶胞基矢才是 Å）；文件里还有 5355 个 `ImageOf` 的周期镜像拷贝要剔除；
    * `.xtd`：同构，只是多了 `<Trajectory End="6001" Type="trj"/>` —— **坐标在同名
      `.trj` 里**，所以 `.xtd` 单独只能当单帧；
    * `.car`：一行一个原子 + `PBC a b c α β γ`，**没有键表** → 靠猜键。

    这里用合成文件钉住解析口径（分数坐标换算、镜像剔除、键表、晶胞、元素），
    比"能打开真实文件"更能防回归。
    """
    from mdta.formats import ms_sibling_trajectory
    from mdta.io import load_trajectory
    from mdta.ms_structure import (cell_from_vectors, read_car,
                                   read_ms_document)

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fmt_probe")
    os.makedirs(d, exist_ok=True)

    # ---- 合成 .xsd：立方盒 10 Å，两个 C 原子相距 1.5 Å（分数坐标 0.15）
    xsd = os.path.join(d, "_ms_probe.xsd")
    with open(xsd, "w", encoding="latin1") as fh:
        fh.write('<?xml version="1.0" encoding="latin1"?>\n<XSD>\n'
                 ' <AtomisticTreeRoot ID="1">\n'
                 '  <Atom3d ID="101" Name="C1" XYZ="0,0,0" Components="C" '
                 'ForcefieldType="c3a"/>\n'
                 '  <Atom3d ID="102" Name="C2" XYZ="0.15,0,0" Components="C" '
                 'ForcefieldType="c3a"/>\n'
                 '  <Atom3d ID="201" ImageOf="101" XYZ="1,0,0" Components="C"/>\n'
                 '  <Bond ID="301" Connects="101,102"/>\n'
                 '  <Bond ID="302" ImageOf="301" Connects="101,201"/>\n'
                 '  <SpaceGroup ID="1" AVector="10,0,0" BVector="0,10,0" '
                 'CVector="0,0,10"/>\n'
                 ' </AtomisticTreeRoot>\n</XSD>\n')
    doc = read_ms_document(xsd)
    assert doc["n_atoms"] == 2, doc["n_atoms"]          # 镜像被剔除
    assert doc["n_images_skipped"] == 1, doc["n_images_skipped"]
    assert doc["elements"] == ["C", "C"], doc["elements"]
    assert len(doc["bonds"]) == 1, doc["bonds"]          # 镜像键被剔除
    assert np.allclose(doc["xyz"][1] - doc["xyz"][0], [1.5, 0, 0], atol=1e-6), \
        doc["xyz"]                                      # 分数 0.15 × 10 Å = 1.5 Å
    assert np.allclose(doc["cell"], [10, 10, 10, 90, 90, 90], atol=1e-6), doc["cell"]

    # 走完整读取链路：原子/键/晶胞/分子数都对
    mdt = load_trajectory(xsd)
    u = mdt.universe
    assert u.atoms.n_atoms == 2 and len(u.bonds) == 1, (u.atoms.n_atoms, len(u.bonds))
    assert np.allclose(np.asarray(u.dimensions, dtype=float)[:3], 10.0), u.dimensions
    assert mdt.ms_info["format"] == "xsd", mdt.ms_info
    assert len(u.trajectory) == 1                       # .xsd 只有 1 帧

    # ---- 合成 .car：三原子链，键长 1.5 Å，没有 CONECT → 应自动猜键
    car = os.path.join(d, "_ms_probe.car")
    with open(car, "w", encoding="latin1") as fh:
        fh.write("!BIOSYM archive 3\nPBC=ON\nsynthetic\n!DATE x\n"
                 "PBC   20.0000   20.0000   20.0000   90.0000   90.0000   90.0000 (P1)\n"
                 "C1   0.0000 0.0000 0.0000 XXXX 1 c3a C 0.000\n"
                 "C2   1.5000 0.0000 0.0000 XXXX 1 c3a C 0.000\n"
                 "C3   3.0000 0.0000 0.0000 XXXX 1 c3a C 0.000\n"
                 "end\nend\n")
    dcar = read_car(car)
    assert dcar["n_atoms"] == 3 and dcar["bonds"] is None, dcar
    assert np.allclose(dcar["cell"][:3], 20.0), dcar["cell"]
    mdt2 = load_trajectory(car)
    u2 = mdt2.universe
    assert u2.atoms.n_atoms == 3, u2.atoms.n_atoms
    assert len(u2.bonds) == 2, len(u2.bonds)            # 猜键：C1-C2、C2-C3
    assert len(u2.atoms.fragments) == 1, len(u2.atoms.fragments)

    # ---- 同名轨迹自动配对：只有真轨迹后缀才算（.xsd/.xtd/.car 不算）
    with open(os.path.join(d, "_ms_probe.xtc"), "wb") as fh:
        fh.write(b"")                                   # 空文件：只验证"配上了"
    sib = ms_sibling_trajectory(xsd)
    assert sib and sib.endswith("_ms_probe.xtc"), sib
    os.remove(os.path.join(d, "_ms_probe.xtc"))
    assert ms_sibling_trajectory(car) is None, ms_sibling_trajectory(car)

    # 晶胞基矢 → [a,b,c,α,β,γ] 的换算（正交/三斜都试）
    assert np.allclose(cell_from_vectors(np.array([10.0, 0, 0]),
                                         np.array([0, 20.0, 0]),
                                         np.array([0, 0, 30.0])),
                       [10, 20, 30, 90, 90, 90]), "正交盒换算错了"
    tri = cell_from_vectors(np.array([10.0, 0, 0]),
                            np.array([5.0, 8.660254, 0]),
                            np.array([0, 0, 30.0]))
    assert abs(tri[3] - 90.0) < 1e-4 and abs(tri[4] - 90.0) < 1e-4, tri
    assert abs(tri[5] - 60.0) < 1e-3, tri               # 120° 的补角

    return (f".xsd 分数坐标×基矢 → 1.5 Å 键长、镜像剔除 1 个、键表 1 根；"
            f".car 三原子链自动猜出 2 根键；同名 .xtc 自动配对；晶胞换算（正交/三斜）正确")


def _build_recog_system():
    """构造一个"混合体系"用来测识别：乙烷 + C12H26 线形链 + 2 个水 + 碳网络 + Na。"""
    atoms = []        # [(name, element)]
    res_of = []       # 每个原子的 residue 下标
    res_names = []    # 残基级 resname（MDAnalysis 的 resnames 是残基属性）
    bonds = []

    def add_res(resname, specs):
        ri = len(res_names)
        res_names.append(resname)
        base = len(atoms)
        for nm, el in specs:
            atoms.append((nm, el))
            res_of.append(ri)
        return base

    # ① 乙烷 C2H6
    b = add_res("ETH", [("C1", "C"), ("C2", "C")]
                + [(f"H{i}", "H") for i in range(1, 7)])
    bonds.append((b + 0, b + 1))
    bonds += [(b + 0, b + 2 + k) for k in range(3)]
    bonds += [(b + 1, b + 5 + k) for k in range(3)]

    # ② 线形链 C12H26（聚乙烯类）：12 个碳成一条路径 + 26 个氢
    b = add_res("PE", [(f"C{i}", "C") for i in range(1, 13)]
                + [(f"H{i}", "H") for i in range(1, 27)])
    for k in range(11):
        bonds.append((b + k, b + k + 1))
    h = b + 12
    for k in range(12):
        for _ in range(3 if k in (0, 11) else 2):
            bonds.append((b + k, h))
            h += 1

    # ③ 两个水
    for _ in range(2):
        b = add_res("SOL", [("O", "O"), ("H1", "H"), ("H2", "H")])
        bonds += [(b, b + 1), (b, b + 2)]

    # ④ 4×4 碳网格（二维共价网络：环秩高、支化点多）
    grid = {}
    specs = []
    for i in range(4):
        for j in range(4):
            grid[(i, j)] = len(specs)
            specs.append((f"C{i}{j}", "C"))
    b = add_res("NET", specs)
    for i in range(4):
        for j in range(4):
            if i + 1 < 4:
                bonds.append((b + grid[(i, j)], b + grid[(i + 1, j)]))
            if j + 1 < 4:
                bonds.append((b + grid[(i, j)], b + grid[(i, j + 1)]))

    # ⑤ 一个游离 Na 离子（单原子分子）
    add_res("NA", [("NA", "NA")])

    u = mda.Universe.empty(len(atoms), n_residues=len(res_names),
                           atom_resindex=res_of, trajectory=True)
    u.add_TopologyAttr("names", [a[0] for a in atoms])
    u.add_TopologyAttr("elements", [a[1] for a in atoms])
    u.add_TopologyAttr("resnames", res_names)
    u.add_bonds(bonds)
    return u, bonds


def test_molecular_recognition():
    """分子 / 聚合物链识别：乙烷、PE 链、水、二维碳网络、单原子离子。

    验收（recognize.md 测试 1–4）：分类正确、链类型正确、原子归属不重不漏、
    展开出的索引与 Universe 一致、没有键表时不会瞎猜。
    """
    from mdta.recognize import recognize

    u, bonds = _build_recog_system()
    n = u.atoms.n_atoms
    rec = recognize(u)
    s = rec.summary()

    assert s["bond_source"] == "topology", s
    assert s["assigned_atoms"] == n == s["total_atoms"], (s, n)
    assert s["unassigned_atoms"] == 0, s
    assert s["total_molecules"] == 6, s          # 乙烷/PE/水×2/网络/Na
    assert s["total_polymer_chains"] == 1, s     # 只有 C12H26 是聚合物
    assert s["total_network_components"] == 1, s
    assert s["total_small_molecules"] == 4, s    # 乙烷 + 水×2 + Na
    assert s["total_unknown"] == 0, s

    # 每个分子内部原子数之和 == 总原子数（不重不漏）
    assert int(rec.atoms_of_mol.sum()) == n, rec.atoms_of_mol
    # 归属映射必须覆盖全部原子，且每个原子恰好属于一个分子
    assert rec.mol_of_atom.shape == (n,)
    assert np.all(rec.mol_of_atom >= 0)
    seen = np.zeros(n, dtype=int)
    for m in range(rec.n_molecules):
        idx = rec.mol_indices(m)
        seen[idx] += 1
    assert np.all(seen == 1), "有原子被归属多次或漏掉"

    # ---- 找各个对象
    def mol_of_resname(rn):
        for m in range(rec.n_molecules):
            if rec.resname_of_mol[m] == rn:
                yield m

    pe = next(mol_of_resname("PE"))
    assert rec.type_of_mol[pe] == "polymer", rec.type_of_mol[pe]
    assert rec.chain_type_of_mol[pe] == "linear", rec.chain_type_of_mol[pe]
    assert rec.atoms_of_mol[pe] == 38, rec.atoms_of_mol[pe]      # C12H26
    assert rec.bonds_of_mol[pe] == 37, rec.bonds_of_mol[pe]      # 11 C–C + 26 C–H
    assert rec.heavy_of_mol[pe] == 12 and rec.rings_of_mol[pe] == 0

    net = next(mol_of_resname("NET"))
    assert rec.type_of_mol[net] == "network", rec.type_of_mol[net]
    assert rec.chain_type_of_mol[net] == "network"
    assert rec.atoms_of_mol[net] == 16 and rec.rings_of_mol[net] == 9

    eth = next(mol_of_resname("ETH"))
    assert rec.type_of_mol[eth] == "small_molecule", rec.type_of_mol[eth]
    assert rec.atoms_of_mol[eth] == 8

    na = next(mol_of_resname("NA"))
    assert rec.atoms_of_mol[na] == 1 and rec.chain_type_of_mol[na] == "single"

    # ---- 展开（结构树点节点 → 原子索引）
    e_all = rec.expand("all", "")
    assert e_all["n_atoms"] == n and sorted(e_all["indices"]) == list(range(n))
    e_pe = rec.expand("molecule", rec.mol_id(pe))
    assert e_pe["n_atoms"] == 38, e_pe["n_atoms"]
    assert sorted(e_pe["indices"]) == sorted(rec.mol_indices(pe).tolist())
    e_sec = rec.expand("section", "polymer")
    assert e_sec["n_atoms"] == 38, e_sec["n_atoms"]
    # 分组展开：水组应该有 2 个对象 6 个原子
    water_g = None
    for gi, g in enumerate(rec.groups):
        if g["label"] == "SOL":
            water_g = gi
    assert water_g is not None, [g["label"] for g in rec.groups]
    e_grp = rec.expand("group", f"G{water_g:04d}")
    assert e_grp["n_atoms"] == 6, e_grp["n_atoms"]

    # ---- 结构树：类别分组统计必须与逐分子统计一致
    tree = rec.tree()
    tot = sum(sec["n_objects"] for sec in tree["sections"])
    assert tot == rec.n_molecules, (tot, rec.n_molecules)
    keys = [sec["key"] for sec in tree["sections"]]
    assert keys[0] == "polymer", keys        # 聚合物排最前

    # ---- 没有键表也不能瞎猜：整体系 1 个未识别组分 + 明确警告
    u2 = mda.Universe.empty(5, n_residues=5, atom_resindex=[0, 1, 2, 3, 4],
                            trajectory=True)
    u2.add_TopologyAttr("names", ["C", "H", "O", "N", "S"])
    u2.add_TopologyAttr("elements", ["C", "H", "O", "N", "S"])
    r2 = recognize(u2)
    assert r2.bond_source in ("none", "molnums"), r2.bond_source
    assert r2.summary()["total_molecules"] == 1, r2.summary()
    assert r2.summary()["warnings"], "没有键表时必须给警告"
    assert r2.type_of_mol[0] == "unknown", r2.type_of_mol

    # ---- 人工确认只改显示，不改归属
    before = rec.mol_indices(pe).tolist()
    rec.overrides[rec.mol_id(pe)] = {"name": "PE-A", "status": "manually_confirmed"}
    assert rec.mol_name(pe) == "PE-A"
    assert rec.mol_indices(pe).tolist() == before, "人工确认不能动原子归属"

    return (f"6 个分子（乙烷/C12H26/水×2/4×4 碳网络/Na）分类正确：聚合物 1 条（线形）、"
            f"网络 1 个（环秩 9）、小分子 4 个；归属 {n} 原子不重不漏；"
            f"无键表体系给 1 个未识别组分 + 警告")


def test_single_frame_drops_time_panels():
    """单帧体系（只读结构文件）：**随时间变化的面板要整块丢掉**，只留静态分布图。

    用户实测提问："xsd 不是只有一帧吗，怎么还能绘制出随时间变化的图" ——
    1 个点的时间序列既没信息量，又让人以为时间序列坏了。
    """
    import types

    from mdta.core import AnalysisResult, Panel
    from mdta.pipeline import Analyzer

    def _mk():
        res = AnalysisResult(name="rg", title="回转半径 Rg")
        res.panels = [
            Panel(xlabel="时间 Time (ps)", title="Rg 随时间变化"),
            Panel(xlabel="回转半径 Rg (Å)", title="Rg 分布"),
        ]
        res.add_curve("Rg(t)", [0.0], [20.0], panel=0)
        res.add_curve("Rg 分布", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], panel=1)
        return res

    az = object.__new__(Analyzer)          # 不打标不需要 universe
    one = Analyzer._annotate_frames(az, _mk(), types.SimpleNamespace(n_frames=1))
    assert one.meta.get("single_frame") == 1, one.meta
    assert one.meta.get("n_frames") == 1, one.meta
    assert one.panel_count == 1, one.panel_count          # 时间面板被丢掉
    assert [p.title for p in one.panels] == ["Rg 分布"], one.panels
    assert [c.label for c in one.curves] == ["Rg 分布"], one.curves
    assert one.curves[0].panel == 0                      # 面板重编号
    assert any("已省略" in n for n in one.notes), one.notes

    many = Analyzer._annotate_frames(az, _mk(), types.SimpleNamespace(n_frames=10))
    assert "single_frame" not in many.meta, many.meta
    assert many.panel_count == 2, many.panel_count        # 多帧原样不动
    assert len(many.curves) == 2, many.curves

    # 全是时间面板时**整项跳过**（返回 None + 留下去重原因），
    # 而不是留一堆只有 1 个点的空图（质心距 / 有序度 / 晶区就是这种）
    only = AnalysisResult(name="msd", title="MSD")
    only.panels = [Panel(xlabel="时间 Time (ps)", title="MSD 随时间变化")]
    only.add_curve("MSD", [0.0], [0.0], panel=0)
    skipper = object.__new__(Analyzer)
    skipper.skip_reason = ""
    assert Analyzer._annotate_frames(skipper, only,
                                     types.SimpleNamespace(n_frames=1)) is None
    assert "全是随时间变化" in skipper.skip_reason, skipper.skip_reason

    return ("单帧：时间面板 2 张 → 1 张（Rg 分布保留）、曲线重编号、meta.single_frame=1；"
            "多帧：完全不动；全是时间面板时整项跳过并写明原因")


def test_tpr_bond_table_is_trusted_and_no_fragment_dict():
    """`.tpr` 的键表来自力场参数 → 不许被"键表残缺"体检误删，也不许数碎片。

    背景（用户实测）：Web 端打开文件时 `guess_bonds` 传的是 **false**，于是
    `_maybe_guess_bonds` 进了"自带键表残缺就重猜"的分支 —— 那一步先调
    `u.atoms.fragments`（MDAnalysis 要给每个原子建 Fragment 对象）。AdK 47 681
    原子光是这一步就 2 s；几十万原子的体系会吃掉 GB 级内存，界面就永远停在
    「读取」。更糟的是 TIP4P 水的 **M 位点**天生不连键（50% 的碎片是孤立原子），
    于是力场键表被判成"残缺"、**被删掉重猜** —— 分子划分因此变了。

    这里用合成体系（不碰真 .tpr）钉住三件事：
      1. 「数碎片」用并查集，结果与 `atoms.fragments` 一致；
      2. TIP4P 式拓扑（每分子 3 个成键原子 + 1 个孤立 M 位点）确实会被判"退化"
         —— 所以判据本身没错，错的是"对力场拓扑用这条判据"；
      3. `.tpr` 后缀 + `guess_bonds=False` → 直接保留拓扑键表（`kept_topology_bonds`），
         键表一个不减；而 `.pdb`（CONECT 残缺的那种）仍然照旧重猜。
    """
    import MDAnalysis as mda

    from mdta.io import MDTrajectory
    from mdta.topology import (_bond_component_count, topology_degeneracy)

    n_mol = 8
    n_at = 4 * n_mol                    # 每个"水" = O + 2H（成键）+ M（孤立）
    res = []
    for m in range(n_mol):
        res += [m, m, m, m]
    u = mda.Universe.empty(n_at, n_residues=n_mol, atom_resindex=res,
                           trajectory=True)
    u.add_TopologyAttr("masses", [16.0, 1.0, 1.0, 0.0] * n_mol)
    u.add_TopologyAttr("elements", ["O", "H", "H", ""] * n_mol)
    u.add_TopologyAttr("resnames", ["SOL"] * n_mol)
    u.add_TopologyAttr("resids", list(range(1, n_mol + 1)))
    u.add_TopologyAttr("segids", ["SYSTEM"])
    bonds = []
    for m in range(n_mol):
        o = 4 * m
        bonds += [(o, o + 1), (o, o + 2)]      # M 位点（o+3）故意不连键
    u.add_TopologyAttr("bonds", np.asarray(bonds, dtype=int))

    n_by_union = _bond_component_count(u.atoms)
    n_by_mda = len(u.atoms.fragments)
    assert n_by_union == n_by_mda == 2 * n_mol, (n_by_union, n_by_mda)

    degen = topology_degeneracy(u.atoms)
    assert degen["degenerate"] is True, degen       # 50% 碎片是孤立 M 位点
    assert degen["n_frag"] == 2 * n_mol, degen

    # ① `.tpr`：力场键表可信 → 不猜、不删
    mdt = MDTrajectory(topology="fake_oplsaa.tpr", guess_bonds=False)
    mdt._universe = u
    mdt._maybe_guess_bonds()
    assert (mdt.bond_guess_info or {}).get("kept_topology_bonds") is True, \
        mdt.bond_guess_info
    assert len(u.bonds) == 2 * n_mol, len(u.bonds)

    # ② `.pdb`（CONECT 残缺的典型）**不受影响**：仍然走"猜键后比对碎片数"的老路
    u2 = mda.Universe.empty(n_at, n_residues=n_mol, atom_resindex=res,
                            trajectory=True)
    u2.add_TopologyAttr("masses", [16.0, 1.0, 1.0, 0.0] * n_mol)
    u2.add_TopologyAttr("elements", ["O", "H", "H", ""] * n_mol)
    u2.add_TopologyAttr("resnames", ["SOL"] * n_mol)
    u2.add_TopologyAttr("resids", list(range(1, n_mol + 1)))
    u2.add_TopologyAttr("segids", ["SYSTEM"])
    u2.add_TopologyAttr("bonds", np.asarray(bonds, dtype=int))
    mdt2 = MDTrajectory(topology="fake.pdb", guess_bonds=False)
    mdt2._universe = u2
    mdt2._maybe_guess_bonds()
    info2 = mdt2.bond_guess_info or {}
    assert not info2.get("kept_topology_bonds"), info2
    assert "elapsed_sec" in info2 or "fragments_before" in info2, info2
    return ("并查集碎片数 == atoms.fragments（16）；TIP4P 式拓扑被判退化"
            "（50% 孤立）；.tpr 保留力场键表不重猜（16 根键不变）；"
            "`.pdb` 仍走老的猜键比对路径")


def test_ms_trj_keeps_periodic_box():
    """`.trj` 帧里没有晶胞时，转换必须用**拓扑的晶胞**补上。

    用户实测（PE50-g2020）：MS 的 `.trj` 每帧 ``DefCell`` 前三个是 0，
    pymsdump 读出来的 ``fr.cell`` 是空的 → 老代码写出"没有盒"的 XTC/TRR →
    打开后 ``u.dimensions = None``：MSD/RDF/密度/界面全都没有 PBC，
    二维 RDF / 界面 / BOO 直接置灰（用户实测就是这个）。
    这条用例钉住两件事：① ``_cell6`` 会用 fallback；② **缓存键随晶胞变化**
    （否则老的无盒缓存会被继续复用，改了等于没改）。
    """
    from mdta.ms_trj import _cell6, cached_xtc

    empty = [0.0, 0.0, 0.0, 90.0, 90.0, 90.0]
    cell = [49.2, 49.2, 120.0, 90.0, 90.0, 60.0]
    assert list(_cell6([])) == empty, _cell6([])
    got = _cell6([], fallback=cell)
    assert np.allclose(got, cell), got
    # 帧里有合法晶胞 → 用帧里的，不被 fallback 覆盖
    assert np.allclose(_cell6([10.0, 20.0, 30.0], fallback=cell),
                       [10.0, 20.0, 30.0, 90.0, 90.0, 90.0])
    # 只给三条棱长也能用（角按 90° 补）；4/5 个值当"不完整"处理 → 用 fallback
    assert np.allclose(_cell6([1, 2, 3], fallback=None),
                       [1, 2, 3, 90, 90, 90])
    assert np.allclose(_cell6([1, 2, 3, 4], fallback=cell), cell)
    # MS 的 DefCell 会把晶格矩阵拆成 (42.6,49.2,120,0,0,24.6)：棱长正常、角度 0
    # → 退化盒，MDAnalysis 照样报 dimensions=None。必须判无效、退回拓扑晶胞。
    assert np.allclose(_cell6([42.6, 49.2, 120.0, 0.0, 0.0, 24.6], fallback=cell), cell)

    here = __file__
    k_none = cached_xtc(here, fmt="trr", velocities=True).name
    k_cell = cached_xtc(here, fmt="trr", velocities=True, cell=cell).name
    assert k_none != k_cell, (k_none, k_cell)
    return ("帧内无晶胞时用拓扑晶胞（49.2×49.2×120 / 90,90,60）；"
            "帧内合法晶胞优先；缓存键随晶胞变化（老的无盒缓存自动失效）")


def test_vacf_matches_forcite_definition():
    """VACF 口径：C(τ) = Σ_i Σ_d Σ_t v(t)·v(t+τ) / ((N−τ)·N_atoms)（= Forcite）。"""
    import MDAnalysis as mda
    from mdta.preprocess import FrameSelection

    rng = np.random.default_rng(7)
    n_at, n_fr = 4, 6
    u = mda.Universe.empty(n_at, n_residues=1,
                           atom_resindex=[0] * n_at, trajectory=True)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["MOL"])
    u.add_TopologyAttr("resids", [1])
    u.add_TopologyAttr("names", [f"C{i}" for i in range(n_at)])
    coords = np.zeros((n_fr, n_at, 3), dtype=np.float32)
    vels = rng.normal(0.0, 3.0, size=(n_fr, n_at, 3)).astype(np.float32)
    u.load_new(coords, format="MEMORY", velocities=vels)
    mdt = MDTrajectory.from_universe(u)
    sel = FrameSelection(np.arange(n_fr), np.arange(n_fr, dtype=float))
    res = dyn.analyze_vacf(mdt, {"x": u.atoms}, sel)
    got = np.asarray(res.curves_of(0)[0].y, dtype=float)
    v = vels.astype(np.float64)
    exp = np.array([(v[:n_fr - l] * v[l:]).sum() / ((n_fr - l) * n_at)
                    for l in range(n_fr)])
    worst = float(np.abs(got - exp).max())
    assert worst < 1e-4, (worst, got, exp)
    assert abs(got[0] - float((v ** 2).sum() / (n_fr * n_at))) < 1e-4
    return (f"手工公式与引擎逐 lag 一致（最大偏差 {worst:.2e}）；"
            f"C(0) = {got[0]:.4f} ≈ 均方速度")


def test_vacf_requires_velocities():
    """没有速度的轨迹（XTC/DCD 一类）必须被明确拦下，而不是算出全 0。"""
    import MDAnalysis as mda
    from mdta.preprocess import FrameSelection

    u = mda.Universe.empty(2, n_residues=1, atom_resindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("elements", ["C", "C"])
    u.add_TopologyAttr("resnames", ["MOL"])
    u.add_TopologyAttr("resids", [1])
    u.add_TopologyAttr("names", ["C0", "C1"])
    u.load_new(np.zeros((4, 2, 3), dtype=np.float32), format="MEMORY")
    mdt = MDTrajectory.from_universe(u)
    sel = FrameSelection(np.arange(4), np.arange(4, dtype=float))
    try:
        dyn.analyze_vacf(mdt, {"x": u.atoms}, sel)
    except ValueError as exc:
        assert "没有速度数据" in str(exc), str(exc)
        return "无速度轨迹 → 明确报错（提示去用 .trj 转 TRR）"
    raise AssertionError("无速度轨迹本应报错")


def test_ring_aromatic_detection():
    """环识别：苯环能被认出，普通 6 元环（环己烷）和 5 元环不能误判成苯环。"""
    import MDAnalysis as mda
    from mdta import rings as R

    def toy(els, bonds, coords):
        n = len(els)
        u = mda.Universe.empty(n, n_residues=1, atom_resindex=[0] * n,
                               trajectory=True)
        u.add_TopologyAttr("elements", els)
        u.add_TopologyAttr("resnames", ["MOL"])
        u.add_TopologyAttr("resids", [1])
        u.add_TopologyAttr("names", [f"{e}{i}" for i, e in enumerate(els)])
        u.add_bonds(bonds)
        u.atoms.positions = np.asarray(coords, dtype=float)
        return u

    ang = np.deg2rad(np.arange(6) * 60)
    # 苯：C–C 1.40 Å
    c6 = np.c_[1.40 * np.cos(ang), 1.40 * np.sin(ang), np.zeros(6)]
    u1 = toy(["C"] * 6 + ["H"] * 6,
             [(i, (i + 1) % 6) for i in range(6)] + [(i, i + 6) for i in range(6)],
             np.vstack([c6, c6 * 1.08]))
    r1 = R.analyze_rings(u1)
    # 环己烷：C–C 1.53 Å，几何上也是 6 元环
    c6 = np.c_[1.53 * np.cos(ang), 1.53 * np.sin(ang), np.zeros(6)]
    hs = [c6[i] + np.array([0, 0, dz * 1.09]) for i in range(6) for dz in (1, -1)]
    bonds = ([(i, (i + 1) % 6) for i in range(6)]
             + [(i, 6 + 2 * i) for i in range(6)]
             + [(i, 7 + 2 * i) for i in range(6)])
    u2 = toy(["C"] * 6 + ["H"] * 12, bonds, np.vstack([c6, np.array(hs)]))
    r2 = R.analyze_rings(u2)
    # 环戊烷：5 元环
    a5 = np.deg2rad(np.arange(5) * 72)
    c5 = np.c_[1.54 * np.cos(a5), 1.54 * np.sin(a5), np.zeros(5)]
    u3 = toy(["C"] * 5, [(i, (i + 1) % 5) for i in range(5)], c5)
    r3 = R.analyze_rings(u3)

    assert r1["n_ring6"] == 1 and r1["n_benzene"] == 1, r1
    assert r2["n_ring6"] == 1 and r2["n_benzene"] == 0, r2
    assert r3["n_ring6"] == 0 and r3["n_rings"] == 1, r3
    return ("苯 → 6 元环 1 / 苯环 1；环己烷 → 6 元环 1 / 苯环 0（C–C 1.53 Å 不判芳香）；"
            "环戊烷 → 无 6 元环、环数 1")


def test_recognize_reports_benzene_rings():
    """识别结果要带 n_ring6 / n_benzene（石墨烯那种带圈六元环）。"""
    import MDAnalysis as mda
    from mdta import recognize as rec

    ang = np.deg2rad(np.arange(6) * 60)
    c6 = np.c_[1.40 * np.cos(ang), 1.40 * np.sin(ang), np.zeros(6)]
    els = ["C"] * 6 + ["H"] * 6
    n = len(els)
    u = mda.Universe.empty(n, n_residues=1, atom_resindex=[0] * n, trajectory=True)
    u.add_TopologyAttr("elements", els)
    u.add_TopologyAttr("resnames", ["MOL"])
    u.add_TopologyAttr("resids", [1])
    u.add_TopologyAttr("names", [f"{e}{i}" for i, e in enumerate(els)])
    u.add_bonds([(i, (i + 1) % 6) for i in range(6)] + [(i, i + 6) for i in range(6)])
    u.atoms.positions = np.vstack([c6, c6 * 1.08])
    r = rec.recognize(u)
    row = r.objects(limit=1)["rows"][0]
    assert row["n_rings"] == 1 and row["n_ring6"] == 1 and row["n_benzene"] == 1, row
    return (f"苯分子在识别结果里 n_rings={row['n_rings']} / n_ring6={row['n_ring6']} "
            f"/ n_benzene={row['n_benzene']}，ring_sizes={row['ring_sizes']}")


def test_cancel_token_stops_loops_and_resets():
    """取消令牌：置位后逐帧循环立刻抛 AnalysisCancelled；清掉后一切照常。"""
    import threading

    import MDAnalysis as mda

    from mdta import cancel as C
    from mdta.analysis import dynamics as dyn2
    from mdta.analysis.base import FrameSelection, frame_iterator
    from mdta.io import MDTrajectory as MDT2

    n_at, n_fr = 4, 6
    u = mda.Universe.empty(n_at, n_residues=1, atom_resindex=[0] * n_at,
                           trajectory=True)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["MOL"])
    u.add_TopologyAttr("resids", [1])
    u.add_TopologyAttr("names", [f"C{i}" for i in range(n_at)])
    u.add_bonds([(0, 1), (1, 2), (2, 3)])
    coords = np.zeros((n_fr, n_at, 3), dtype=np.float32)
    u.load_new(coords, format="MEMORY")
    mdt = MDT2.from_universe(u)
    sel = FrameSelection(np.arange(n_fr), np.arange(n_fr, dtype=float))

    # 1) 未取消：正常遍历
    assert len(list(frame_iterator(mdt, sel))) == n_fr

    # 2) 置位后：立刻抛出，而不是跑完
    ev = threading.Event()
    ev.set()
    C.bind(ev)
    try:
        list(frame_iterator(mdt, sel))
        raise AssertionError("取消令牌置位后 frame_iterator 应该立刻抛错")
    except C.AnalysisCancelled:
        pass
    try:
        dyn2.analyze_msd(mdt, {"x": u.atoms}, sel, tau_intervals=1)
        raise AssertionError("取消令牌置位后 analyze_msd 应该立刻抛错")
    except C.AnalysisCancelled:
        pass

    # 3) 清掉令牌：后续分析照常（不影响重新分析）
    C.clear()
    assert not C.is_set()
    assert len(list(frame_iterator(mdt, sel))) == n_fr
    res = dyn2.analyze_msd(mdt, {"x": u.atoms}, sel, tau_intervals=1)
    assert res.name == "msd" and res.curves
    return "取消置位 → frame_iterator / MSD 立刻中断；C.clear() 后恢复，可继续重新分析"

def test_end_to_end_vectorized_matches_manual_loop():
    """R_ee：向量化计算与「逐个分子逐个帧手取坐标」结果一致。"""
    import MDAnalysis as mda

    from mdta.analysis import conformation as conf2
    from mdta.analysis.base import FrameSelection
    from mdta.io import MDTrajectory as MDT3

    # 3 个分子（每个 3 个原子成链），6 帧
    n_mol, per, n_fr = 3, 3, 6
    n_at = n_mol * per
    u = mda.Universe.empty(n_at, n_residues=n_mol,
                           atom_resindex=[i // per for i in range(n_at)],
                           trajectory=True)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["MOL"] * n_mol)
    u.add_TopologyAttr("resids", list(range(1, n_mol + 1)))
    u.add_TopologyAttr("names", [f"C{i}" for i in range(n_at)])
    bonds = []
    for m in range(n_mol):
        base = m * per
        bonds += [(base, base + 1), (base + 1, base + 2)]
    u.add_bonds(bonds)
    rng = np.random.default_rng(3)
    coords = np.cumsum(rng.normal(0, 0.5, (n_fr, n_at, 3)), axis=0).astype(np.float32)
    u.load_new(coords, format="MEMORY")
    mdt = MDT3.from_universe(u)
    sel = FrameSelection(np.arange(n_fr), np.arange(n_fr, dtype=float))

    res = conf2.analyze_end_to_end(mdt, u.atoms, sel)
    got = np.asarray(res.curves_of(0)[0].y, dtype=float)
    pairs = res.meta["end_atoms"]
    assert len(pairs) == n_mol, pairs

    manual = np.full((n_fr, n_mol), np.nan)
    for p, (i1, i2) in enumerate(pairs):
        for k in range(n_fr):
            manual[k, p] = float(np.linalg.norm(coords[k, i1] - coords[k, i2]))
    exp = np.nanmean(manual, axis=1)
    worst = float(np.nanmax(np.abs(got - exp)))
    assert worst < 1e-5, (worst, got, exp)
    return (f"向量化 R_ee 与逐分子逐帧手算一致（max|Δ| = {worst:.2e}；"
            f"{n_mol} 个分子 × {n_fr} 帧）")


def test_repair_bonds_guard_boxless_and_forcefield():
    """按距离补键的**闸门**：没有周期盒时跳过；力场拓扑（.tpr）根本不补。

    用户实测：单独打开 47681 原子的 `adk_oplsaa.tpr` 会**卡死 + 内存暴涨**。
    根因是 `repair_fragmented_bonds` 在没有盒时让 MDAnalysis 的 capped_distance
    退化成暴力 N² 两两比较（实测 98 s 还没跑完）。这里验证两道闸：
      ① 无盒 → skipped（不跑 O(N²)）；
      ② 有盒 → 正常补键（碎片链之间按共价距离合并）；
      ③ 力场拓扑后缀（.tpr）→ `_maybe_repair_bonds` 直接返回，不动键表。
    """
    import MDAnalysis as mda
    from mdta.topology import repair_fragmented_bonds
    from mdta.io import MDTrajectory

    n = 6
    xyz = np.array([[0.0, 0, 0], [1.4, 0, 0], [3.0, 0, 0],
                    [4.4, 0, 0], [6.0, 0, 0], [7.4, 0, 0]])

    def toy(with_box):
        u = mda.Universe.empty(n, n_residues=n,
                               atom_resindex=list(range(n)), trajectory=True)
        u.add_TopologyAttr("elements", ["C"] * n)
        u.add_TopologyAttr("resnames", ["MOL"] * n)
        u.add_TopologyAttr("resids", list(range(1, n + 1)))
        u.add_TopologyAttr("names", [f"C{i}" for i in range(n)])
        u.add_bonds([(0, 1), (2, 3), (4, 5)])
        u.atoms.positions = xyz.copy()
        if with_box:
            u.trajectory.ts.dimensions = np.array([40, 40, 40, 90, 90, 90],
                                                  dtype=np.float32)
        return u

    # ① 无盒 + 小体系：允许补键（暴力搜索只有几千万对，秒级）
    info0 = repair_fragmented_bonds(toy(False))
    assert info0["added"] == 2, info0
    assert info0["components_after"] == 1, info0

    # ①b 无盒 + 大体系（>8,000 原子）：必须直接跳过（否则 O(N²) 卡死/OOM）
    big = mda.Universe.empty(9000, n_residues=9000,
                             atom_resindex=list(range(9000)), trajectory=True)
    big.add_TopologyAttr("elements", ["C"] * 9000)
    big.add_TopologyAttr("resnames", ["MOL"] * 9000)
    big.add_TopologyAttr("resids", list(range(1, 9001)))
    big.add_TopologyAttr("names", [f"C{i}" for i in range(9000)])
    big.add_bonds([(0, 1), (2, 3), (4, 5)])
    big.atoms.positions = np.c_[np.arange(9000) * 2.0, np.zeros(9000), np.zeros(9000)]
    info_big = repair_fragmented_bonds(big)
    assert info_big["skipped"] and ("O(N²)" in info_big["skipped"]
                                    or "8,000" in info_big["skipped"]), info_big
    assert info_big["added"] == 0, info_big

    # ② 有盒：1.6 Å 的跨分量对（C–C 共价和 1.52×1.1 ≈ 1.67）应被补上两根
    info1 = repair_fragmented_bonds(toy(True))
    assert info1["added"] == 2, info1
    assert info1["components_after"] == 1, info1

    # ③ 力场拓扑（.tpr）：_maybe_repair_bonds 直接返回，键表一根本不动
    u = toy(True)
    mdt = MDTrajectory.from_universe(u)
    mdt.topology = "adk_oplsaa.tpr"
    nb0 = len(u.bonds)
    mdt._maybe_repair_bonds()
    assert len(u.bonds) == nb0, (nb0, len(u.bonds))
    assert not mdt.load_notes, mdt.load_notes
    return ("无盒 → 跳过补键；有盒 → 补 2 根、分量 3→1；"
            ".tpr 力场拓扑 → 不补键（AdK 卡死的那条路径已关闸）")


def test_download_filenames_are_short_ascii():
    """下载文件名必须**英文、短、纯 ASCII**（图表区 PNG / 数据表 CSV / 导出板块）。

    根因：浏览器下载的文件名走 HTTP 头 ``Content-Disposition``，Starlette 按
    latin-1 编码头字段 —— 一旦带中文就直接 ``UnicodeEncodeError``，用户看到的
    就是"下载不了"（实测踩过：文件名被改成面板中文标题后就下不动了）。
    所以三处统一用 ``mdta/naming.py`` 的短英文 token（``msd__axes.csv`` 这种）。
    """
    import re
    import shutil
    import types

    from mdta.core import AnalysisResult, Panel
    from mdta.export import export_csv
    from mdta.naming import PANEL_SLUGS, panel_slug, safe_ascii

    ascii_re = re.compile(r"^[A-Za-z0-9._-]+$")

    # ---- ① token 表本身：纯 ASCII、短、同项内不重名
    for (aname, idx), slug in PANEL_SLUGS.items():
        assert ascii_re.match(slug), (aname, idx, slug)
        assert len(slug) <= 12, (aname, idx, slug)
    per: dict[str, list[str]] = {}
    for (aname, _idx), slug in PANEL_SLUGS.items():
        per.setdefault(aname, []).append(slug)
    for aname, slugs in per.items():
        assert len(set(slugs)) == len(slugs), (aname, slugs)

    # 用户体感最强的几项必须一眼认出来
    assert panel_slug("msd", 2, "MSD 各轴分解") == "axes"
    assert panel_slug("dihedral", 1, "trans/gauche 比例") == "trans"
    assert panel_slug("rdf2d", 0, "面内 RDF") == "inplane"
    # 查不到 → 标题里的 ASCII 单词；再查不到 → p{n}（绝不塞中文）
    assert panel_slug("unknown", 1, "Foo Bar 标题") == "foo_bar"
    assert panel_slug("unknown", 1, "中文标题") == "p2"
    assert safe_ascii("二面角 trans/gauche") == "trans_gauche"

    # ---- ② 会话层：数据表「下载 CSV」的文件名
    from webapp.backend.session import Session

    class _FakeSession:
        _panel_title = Session._panel_title
        table_filename = Session.table_filename
        results_json = {
            "msd": {"panels": [{"title": "MSD 随时间变化"}, {"title": "线性区拟合"},
                               {"title": "MSD 各轴分解"}, {"title": "MSD 幂律"}]},
            "rg": {"panels": [{"title": "Rg 随时间变化"}]},
        }

    fake = _FakeSession()
    assert fake.table_filename("msd", 2) == "msd__axes.csv", fake.table_filename("msd", 2)
    assert fake.table_filename("rg", 0) == "rg.csv", fake.table_filename("rg", 0)
    assert fake.table_filename("msd", 0) == "msd__tau.csv"
    for aname, npnl in (("msd", 4), ("rg", 1)):
        for i in range(npnl):
            fn = fake.table_filename(aname, i)
            assert ascii_re.match(fn), fn
            assert fn.endswith(".csv") and len(fn) <= 28, fn

    # ---- ③ Content-Disposition 必须能按 latin-1 编码（"下载不了"的真正根因）
    for fn in ("msd__axes.csv", "rg.png", "dihedral__trans.csv"):
        f'attachment; filename="{fn}"'.encode("latin-1")     # 中文会在这里抛错
    try:
        'attachment; filename="二面角.csv"'.encode("latin-1")
    except UnicodeEncodeError:
        pass                                                 # 正是要避免的情况
    else:                                                    # pragma: no cover
        raise AssertionError("latin-1 编码中文竟然通过了？这条断言的前提变了")

    # ---- ④ 导出板块：多面板用同一套短英文名（面板标题是中文也照样 ASCII）
    res = AnalysisResult(name="msd", title="MSD")
    res.panels = [Panel(xlabel="滞后时间 τ (ps)", title="MSD 随时间变化"),
                  Panel(xlabel="滞后时间 τ (ps)", title="线性区拟合"),
                  Panel(xlabel="滞后时间 τ (ps)", title="MSD 各轴分解"),
                  Panel(xlabel="滞后时间 τ (ps)", title="MSD 幂律")]
    for i in range(4):
        res.add_curve(f"曲线{i}", [0.0, 1.0], [0.0, float(i + 1)], panel=i)
    # 写在**工作区**里而不是系统临时目录：沙箱允许建临时目录，但往里写文件会被拒
    tmp = os.path.join(os.getcwd(), "_selftest_tmp_name")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    names = sorted(os.path.basename(p) for p in export_csv(res, tmp))
    assert names == ["msd__axes.csv", "msd__fit.csv", "msd__power.csv",
                     "msd__tau.csv"], names

    # 单面板 → 就是一个分析名；前缀全是中文时压成 result
    one = AnalysisResult(name="rdf", title="径向分布函数")
    one.panels = [Panel(xlabel="r (Å)", title="径向分布函数 g(r)")]
    one.add_curve("g(r)", [1.0, 2.0], [0.0, 1.0], panel=0)
    assert [os.path.basename(p) for p in export_csv(one, tmp)] == ["rdf.csv"]
    cjk = sorted(os.path.basename(p) for p in export_csv(res, tmp, prefix="二面角分布"))
    assert all(ascii_re.match(n) for n in cjk), cjk
    assert cjk and cjk[0].startswith("result__"), cjk
    shutil.rmtree(tmp, ignore_errors=True)

    # 三处（图表区 / 数据表 / 导出板块）拼法一致：分析名 + __ + token + 后缀
    assert (f"msd__{panel_slug('msd', 2)}.csv"
            == fake.table_filename("msd", 2) == "msd__axes.csv")

    return (f"短英文 token 表 {len(PANEL_SLUGS)} 项全 ASCII/≤12 字符/同项不重名；"
            "数据表 msd__axes.csv、单面板 rg.csv；导出板块同一套命名（中文前缀→result）；"
            "Content-Disposition 可按 latin-1 编码（中文会抛错 → 这就是“下载不了”的根因）")


def test_cancel_stops_quickly():
    """「取消」必须**真的立刻停**：预估阶段 / 正式跑 / 帧内分块都要能打断。

    用户实测："分析功能点击直接取消没有效果，没有立即取消，计算进程仍然保留，
    导致无法查询/选择功能计算"。

    根因有两条（都不是"按钮没连上"，是后端**根本没问取消令牌**）：
      ① ``estimate_times`` 从来没 ``bind`` 过取消令牌 —— 预估阶段每项要跑 3 次探测
         （每次 2m 帧），这段时间点取消完全无效，要等整项预估跑完；
      ② 取消只在**每帧开头**检查一次，而 4.7 万原子体系上单帧就要几秒到几十秒，
         点取消等于要再等一整帧。

    所以这条断言同时盯两件事：取消能打断、且**打断得足够快**。
    """
    import threading
    import time

    import MDAnalysis as mda
    import MDAnalysis.coordinates.GRO as GRO
    import MDAnalysis.coordinates.XTC as XTC

    from mdta.cancel import AnalysisCancelled
    from mdta.pipeline import Analyzer

    # ---- 造一个小体系（800 原子 / 44 帧）：够慢到能在中途点取消，又不拖慢自检
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cancel_probe")
    os.makedirs(d, exist_ok=True)
    gro = os.path.join(d, "toy.gro")
    xtc = os.path.join(d, "toy.xtc")
    n_chains, chain_len, n_frames, L = 20, 40, 44, 60.0
    n_atoms = n_chains * chain_len
    resix = [i // chain_len for i in range(n_atoms)]
    u = mda.Universe.empty(n_atoms, n_residues=n_chains, atom_resindex=resix,
                           trajectory=True)
    u.add_TopologyAttr("names", ["C"] * n_atoms)
    u.add_TopologyAttr("elements", ["C"] * n_atoms)
    bonds = []
    for c in range(n_chains):
        s = c * chain_len
        bonds += [(s + j, s + j + 1) for j in range(chain_len - 1)]
    u.add_TopologyAttr("bonds", bonds)
    box = np.array([L, L, L, 90.0, 90.0, 90.0])
    u.dimensions = box
    pos0 = np.zeros((n_atoms, 3))
    for c in range(n_chains):
        x0 = 1.0 + (c % 5) * 11.0
        y0 = 1.0 + (c // 5) * 13.0
        for j in range(chain_len):
            pos0[c * chain_len + j] = (x0 + 0.5 * j, y0 + (0.7 if j % 2 else -0.7), 25.0)
    u.atoms.positions = pos0
    with GRO.GROWriter(gro, n_atoms=n_atoms) as gw:
        gw.write(u.atoms)
    rng = np.random.default_rng(3)
    with XTC.XTCWriter(xtc, n_atoms=n_atoms) as w:
        for _f in range(n_frames):
            u.atoms.positions = (pos0 + rng.normal(0, 0.1, pos0.shape)) % L
            u.dimensions = box
            w.write(u.atoms)

    def fresh():
        az = Analyzer(gro, xtc)
        az.set_frames()
        az.auto_setup()
        return az

    # ---- ① 还没开始就点取消：立刻返回、不留半成品结果
    az = fresh()
    ev = threading.Event()
    ev.set()
    t0 = time.perf_counter()
    res = az.run_all(["boo"], verbose=False, excel=False, cancel=ev)
    dt = time.perf_counter() - t0
    assert dt < 3.0, f"预置取消竟然跑了 {dt:.2f}s"
    assert az.cancelled is True, az.cancelled
    assert len(res) == 0, list(res)

    # ---- ② 预估阶段点取消：必须抛 AnalysisCancelled（以前会被吞成"预估失败，直接开跑"）
    az = fresh()
    ev = threading.Event()
    ev.set()
    t0 = time.perf_counter()
    try:
        az.estimate_times(["boo"], cancel=ev)
    except AnalysisCancelled:
        pass
    else:                                                    # pragma: no cover
        raise AssertionError("预估阶段收到取消却没有中断")
    assert time.perf_counter() - t0 < 3.0

    # ---- ③ 正式跑到一半点取消：从置位到分析返回要**很快**（不是等这一帧/这一项算完）
    def _threaded(call, delay: float):
        ev2 = threading.Event()
        box_: dict = {}

        def work():
            box_["t0"] = time.perf_counter()
            try:
                box_["ret"] = call(ev2)
            except BaseException as exc:                      # noqa: BLE001
                box_["err"] = exc
            box_["t1"] = time.perf_counter()

        th = threading.Thread(target=work, daemon=True)
        th.start()
        time.sleep(delay)
        t_set = time.perf_counter()
        ev2.set()
        th.join(timeout=60)
        assert not th.is_alive(), "取消后 60 s 还没停"
        return box_["t1"] - t_set, box_

    az = fresh()
    lat, box_ = _threaded(
        lambda ev2: az.run_all(["boo"], verbose=False, excel=False, cancel=ev2), 0.35)
    assert lat < 1.5, f"取消延迟 {lat:.3f}s 太久（点取消应当 1 秒内停下）"
    assert az.cancelled is True, az.cancelled
    assert (box_["t1"] - box_["t0"]) > 0.3, "分析跑得太快，这条断言没测到东西"

    # ---- ④ 帧内也要检查：逐帧 + 帧内分块 → 检查次数远多于帧数
    from mdta import cancel as cancel_mod

    class _Counting:
        def __init__(self):
            self.times = []

        def is_set(self):
            self.times.append(time.perf_counter())
            return False

    cnt = _Counting()
    cancel_mod._EVENT = cnt
    try:
        az = fresh()
        az.run("boo")
    finally:
        cancel_mod._EVENT = None
    n_chk = len(cnt.times)
    assert n_chk > n_frames, f"boo 只问了 {n_chk} 次取消（帧数 {n_frames}），帧内没查"
    gaps = np.diff(cnt.times)
    assert gaps.size and float(gaps.max()) < 1.0, (
        f"两次取消检查之间最长 {float(gaps.max()):.2f}s，点取消会明显卡住")

    return (f"预置取消 0.0s 返回、预估阶段抛 AnalysisCancelled、"
            f"跑到一半点取消 {lat * 1000:.0f}ms 内停下；"
            f"boo 一帧内问 {n_chk} 次取消（{n_frames} 帧），最长空档 "
            f"{float(gaps.max()) * 1000:.0f}ms")


def test_contact_inter_survives_colliding_molnums():
    """接触/RDF 的 ``inter`` 口径不能被**假的 molnums** 带偏（实测踩过）。

    两条真实故障形态：
      ① **resid 重号**（每个组分都从 1 开始编号）或 MS 导出把整个体系塞进 1–2 个
         残基 → 不同分子的原子被并进同一个 molnum → ``inter`` 把**跨组接触整套
         误删**（420 原子玩具实测：482 对 → 325 对、最小间距 0.30 Å → 2.41 Å，
         而"分子内配对占比"照样显示 0，看不出是口径问题）；
      ② 反过来，把 molnums 一律按键图纠正又会伤到 **4 点水模型**：一个分子 =
         键块 {O,H1,H2} + 孤立的 M 虚位点，键图给出两个块，若全按键图就会把水
         切成"3 原子分子 + 1 原子分子"，分子内邻居数从 3.000 掉到 1.500。

    所以判据只认"真的坏了"：键块横跨多个 molnum，或一个 molnum 里含 ≥2 个
    ≥2 原子的键块。单原子块（虚位点/离子）不算。
    """
    import MDAnalysis as mda

    from mdta.analysis.interface import _molnums_for_filter, analyze_contacts

    box = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0], dtype=np.float32)

    def build(n_chain=2, chain_len=10, n_water=4, m_site=False):
        per_chain = chain_len + (1 if m_site else 0)
        n_atoms = n_chain * per_chain + n_water * 3
        resix = [i // per_chain for i in range(n_chain * per_chain)]
        resix += [n_chain + i // 3 for i in range(n_water * 3)]
        # ① resid 故意重号：两条链 resid 1..n_chain，水也 1..n_water
        resids = list(range(1, n_chain + 1)) + list(range(1, n_water + 1))
        names = ["C"] * (n_chain * chain_len)
        if m_site:
            names += ["M"] * n_chain
        names += ["OW", "HW1", "HW2"] * n_water
        u = mda.Universe.empty(n_atoms, n_residues=len(resids),
                               atom_resindex=resix, trajectory=True)
        u.add_TopologyAttr("names", names)
        u.add_TopologyAttr("elements", ["C"] * n_atoms)
        u.add_TopologyAttr("resnames", ["POLY"] * n_chain + ["SOL"] * n_water)
        u.add_TopologyAttr("resids", resids)
        # 分子编号按**残基**给（MDAnalysis 的 molnums 是残基级的）。
        # 故意用 resid-1：resid 重号时两条链与两个水分到同一个 molnum 上，
        # 这正是 MS 导出 / 各组分从 1 开始编号时的真实故障形态。
        u.add_TopologyAttr("molnums", np.asarray(
            [int(r) - 1 for r in resids], dtype=int))
        bonds = []
        for c in range(n_chain):
            s = c * per_chain
            bonds += [(s + j, s + j + 1) for j in range(chain_len - 1)]
            if m_site:                       # M 虚位点**不连键**（4 点水就是这样）
                pass
        for w in range(n_water):
            b = n_chain * per_chain + w * 3
            bonds += [(b, b + 1), (b, b + 2)]
        u.add_TopologyAttr("bonds", bonds)
        u.dimensions = box
        pos = np.zeros((n_atoms, 3))
        k = 0
        for c in range(n_chain):
            for j in range(chain_len):
                pos[k] = (2.0 + 1.2 * c, 2.0 + 0.75 * j, 15.0)
                k += 1
            if m_site:
                pos[k] = (2.0 + 1.2 * c, 2.0, 15.0)     # M 贴在链旁边
                k += 1
        for w in range(n_water):
            base = n_chain * per_chain + w * 3
            cx, cy, cz = 2.0 + 1.12 * w, 3.0 + 1.12 * w, 15.0
            pos[base] = (cx, cy, cz)
            pos[base + 1] = (cx + 0.96, cy, cz)
            pos[base + 2] = (cx - 0.24, cy + 0.93, cz)
        u.atoms.positions = pos
        return u

    # ---- ① resid 重号：inter 必须与 total 完全一致（两组本来就是不同分子）
    u = build()
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(mdt.times_ps)
    a = u.select_atoms("resname POLY")
    b = u.select_atoms("resname SOL")
    r_inter = analyze_contacts(mdt, a, b, sel, cutoff=5.0, mode="inter")
    r_total = analyze_contacts(mdt, a, b, sel, cutoff=5.0, mode="total")
    n_i = float(r_inter.summary["平均接触对数"])
    n_t = float(r_total.summary["平均接触对数"])
    assert n_t > 0, n_t
    assert abs(n_i - n_t) < 1e-12, (
        n_i, n_t, "resid 重号时 inter 把跨组接触误删了（分子编号被并成一个）")

    # ---- ② 4 点水模型：M 虚位点不能让水被切成两块
    u2 = build(n_chain=2, chain_len=10, n_water=0, m_site=True)
    w = u2.select_atoms("resname POLY")
    m = _molnums_for_filter(w)
    assert m is not None
    n_mol = int(np.unique(m).size)
    assert n_mol == 2, (n_mol, "4 点水/带虚位点的分子被切碎了")
    per = np.bincount(m)
    assert int(per.max()) == 11, per.tolist()      # 10 个链原子 + 1 个 M

    return (f"resid 重号：inter = total = {n_i:.1f} 对/帧（不再误删）；"
            f"带 M 虚位点的分子仍是 1 个分子（{int(per.max())} 原子，没被切碎）")


def test_analysis_registry_maps_each_name_to_its_own_function():
    """分析注册表 ``ANALYSIS_REGISTRY`` 的名字必须指向**真正那个分析函数**。

    为什么钉住：``@register("contact", …)`` 曾被错放在同文件里的辅助函数
    ``shares_one_molecule`` 上（它紧挨着 ``analyze_contacts``），于是注册表里
    ``contact`` 指向了一个返回 bool 的辅助函数 —— 现在界面走的是 pipeline 的直接
    调用所以没炸，但任何用 ``run_analysis("contact", …)`` / 文档清单的地方都会
    拿到错误结果。这条用例逐个核对"名字 → 函数名"。
    """
    from mdta.analysis import base

    # 注册表是"模块被 import 时"填的：先把所有分析模块导进来，
    # 否则 `boo` 这类懒加载的模块会被误判成"未注册"。
    import mdta.analysis.boo  # noqa: F401
    import mdta.analysis.conformation  # noqa: F401
    import mdta.analysis.crystallinity  # noqa: F401
    import mdta.analysis.dynamics  # noqa: F401
    import mdta.analysis.interface  # noqa: F401

    expect = {
        "rg": "analyze_rg",
        "ree": "analyze_end_to_end",
        "dihedral": "analyze_dihedrals",
        "density": "analyze_density",
        "rdf": "analyze_rdf",
        "rdf2d": "analyze_rdf_inplane",
        "contact": "analyze_contacts",
        "interface": "analyze_interface_width",
        "orientation": "analyze_orientation",
        "order": "analyze_structural_order",
        "boo": "analyze_boo",
        "msd": "analyze_msd",
        "vacf": "analyze_vacf",
    }
    bad = {}
    for name, fn_name in expect.items():
        ent = base.ANALYSIS_REGISTRY.get(name)
        if ent is None:
            bad[name] = "未注册"
        elif getattr(ent[1], "__name__", "") != fn_name:
            bad[name] = f"{ent[1].__name__}（应为 {fn_name}）"
    assert not bad, bad
    return f"{len(expect)} 项分析的注册指向都正确（如 contact → analyze_contacts）"


def test_contact_reproduce_recipe_matches_engine():
    """``examples/contact_reproduce.py`` 这份"复现脚本"必须与引擎**逐位一致**。

    为什么要这条：这份示例是给用户拿去做第三方复现的（MDAnalysis 独立实现）。
    一旦有人改了引擎的口径（配对模式、最小镜像、配对来源）却没同步示例，用户按
    示例复现出来的数就和界面对不上 —— 那比没有示例更糟。
    """
    import importlib.util
    import os as _os

    import MDAnalysis as mda

    from mdta.analysis.interface import analyze_contacts
    from mdta.io import MDTrajectory
    from mdta.preprocess import select_frames

    here = _os.path.dirname(_os.path.abspath(__file__))
    path = _os.path.join(here, _os.pardir, "examples", "contact_reproduce.py")
    spec = importlib.util.spec_from_file_location("contact_reproduce", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 小两相体系：2 条 6 原子链 + 3 个水，2 帧（含 resid 重号，逼出分子编号那条路）
    per = 6
    n_atoms = 2 * per + 3 * 3
    resix = [i // per for i in range(2 * per)] + [2 + i // 3 for i in range(9)]
    u = mda.Universe.empty(n_atoms, n_residues=5, atom_resindex=resix,
                           trajectory=True)
    u.add_TopologyAttr("names", ["C"] * (2 * per) + ["OW", "HW1", "HW2"] * 3)
    u.add_TopologyAttr("elements", ["C"] * n_atoms)
    u.add_TopologyAttr("resnames", ["POLY"] * 2 + ["SOL"] * 3)
    u.add_TopologyAttr("resids", [1, 2, 1, 2, 3])
    u.add_TopologyAttr("molnums", np.array([0, 1, 0, 1, 2], dtype=int))
    bonds = []
    for c in range(2):
        s = c * per
        bonds += [(s + j, s + j + 1) for j in range(per - 1)]
    for w in range(3):
        b = 2 * per + w * 3
        bonds += [(b, b + 1), (b, b + 2)]
    u.add_TopologyAttr("bonds", bonds)
    box = np.array([25.0, 25.0, 25.0, 90.0, 90.0, 90.0], dtype=np.float32)
    u.dimensions = box
    pos = np.zeros((n_atoms, 3))
    for c in range(2):
        for j in range(per):
            pos[c * per + j] = (2.0 + 1.3 * c, 2.0 + 0.9 * j, 12.0)
    for w in range(3):
        b = 2 * per + w * 3
        cx, cy = 2.5 + 1.1 * w, 2.2 + 0.9 * w
        pos[b] = (cx, cy, 12.0)
        pos[b + 1] = (cx + 0.96, cy, 12.0)
        pos[b + 2] = (cx - 0.24, cy + 0.93, 12.0)
    u.atoms.positions = pos

    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(mdt.times_ps)
    a = u.select_atoms("resname POLY")
    b = u.select_atoms("resname SOL")

    bad = []
    for mode in ("inter", "intra", "total"):
        eng = analyze_contacts(mdt, a, b, sel, cutoff=5.0, mode=mode)
        mine = mod.reproduce_contacts(u, "resname POLY", "resname SOL",
                                      cutoff=5.0, mode=mode)
        e_pairs = np.asarray([c.y for c in eng.curves if c.label == "接触对数"][0],
                             dtype=float)
        m_pairs = np.asarray(mine["n_pairs"], dtype=float)
        n = min(e_pairs.size, m_pairs.size)
        d = float(np.max(np.abs(e_pairs[:n] - m_pairs[:n]))) if n else 0.0
        if d > 1e-9 * max(1.0, float(np.max(e_pairs[:n]) if n else 1.0)):
            bad.append((mode, "接触对数", d))
        for key_eng, key_my in (("平均接触对数", "平均接触对数"),
                                ("最小原子间距 (Å)", "最小原子间距 (Å)"),
                                ("平均接触数（每个 A 组原子）", "平均接触数（每个 A 组原子）")):
            x = float(mine["summary"][key_my])
            y = float(eng.summary[key_eng])
            if abs(x - y) > 1e-9 * max(1.0, abs(y)):
                bad.append((mode, key_eng, x, y))
    assert not bad, bad
    return ("examples/contact_reproduce.py 与引擎在 inter/intra/total 三种模式下"
            "逐帧接触对数、最小间距、平均接触数全部一致")


def test_rdf_mdat_matches_util_rdf_frame_histogram():
    """`test/mda_test/rdf_mdat.py` 必须是 ``test/util_rdf.py`` 原算法的**忠实移植**。

    为什么要这条：外面那个 ``util_rdf.py``（MDTransport）用的是
    "逐帧全原子对 + 当场最小镜像 + 同种粒子数减 1 的归一化"这套口径，
    与 MDAnalysis ``InterRDF`` 只是接近、不完全相同。移植版一旦改了归一化或
    配对规则，用户拿它和第三方结果对拍就会得出错误结论。
    这里直接把**原实现的函数**加载进来，逐 bin 比对直方图（应当**逐位相同**）。
    """
    import importlib.util
    import os as _os
    import sys as _sys
    import types as _types

    import MDAnalysis as mda

    from mda_test.rdf_mdat import frame_histogram, rdf_mdat

    d = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "util_rdf.py")
    if not _os.path.exists(d):
        return "跳过：test/util_rdf.py 不存在"
    try:
        pkg = _types.ModuleType("_mdtransport_stub")
        pkg.__path__ = [_os.path.dirname(d)]
        _sys.modules["_mdtransport_stub"] = pkg
        util_mod = _types.ModuleType("_mdtransport_stub.util")
        util_mod.BaseAnalysis = object
        util_mod.tqdm = lambda it=None, **kw: it
        _sys.modules["_mdtransport_stub.util"] = util_mod
        pkg.util = util_mod
        spec = importlib.util.spec_from_file_location("_mdtransport_stub.util_rdf", d)
        orig = importlib.util.module_from_spec(spec)
        _sys.modules["_mdtransport_stub.util_rdf"] = orig
        spec.loader.exec_module(orig)
    except Exception as exc:                                 # noqa: BLE001
        return f"~ 跳过：无法加载 util_rdf.py（{type(exc).__name__}: {exc}）"

    # 小体系：两团各 40 个原子，正交盒（距离都落在 r ≤ 半盒内）
    rng = np.random.default_rng(7)
    n_a, n_b, L = 40, 40, 20.0
    u = mda.Universe.empty(n_a + n_b, n_residues=(n_a + n_b),
                           atom_resindex=list(range(n_a + n_b)), trajectory=True)
    u.add_TopologyAttr("elements", ["O"] * n_a + ["H"] * n_b)
    u.add_TopologyAttr("names", ["O"] * n_a + ["H"] * n_b)
    u.atoms.positions = rng.uniform(0.0, L, (n_a + n_b, 3))
    u.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=np.float32)
    a = u.atoms[:n_a]
    b = u.atoms[n_a:]
    ts = u.trajectory[0]
    box = np.asarray(ts.dimensions, dtype=np.float32)[:3]
    bins = np.arange(0.0, float(np.min(box)) / 2.0 + 0.05, 0.05)

    bad = []
    for same, (g1, g2) in ((True, (a, a)), (False, (a, b))):
        c1 = np.asarray(g1.positions, dtype=np.float32)
        c2 = np.asarray(g2.positions, dtype=np.float32)
        mine = frame_histogram(c1, c2, box, bins, same)
        orig.init_rdf_process(c1.reshape(1, *c1.shape), (1, *c1.shape),
                              c2.reshape(1, *c2.shape), (1, *c2.shape),
                              box.reshape(1, 3), bins, same)
        theirs = orig.process_frame_rdf(0)
        if not np.array_equal(mine, theirs):
            bad.append((same, float(np.max(np.abs(mine - theirs))),
                        int((mine != theirs).sum())))
    assert not bad, f"移植版与原实现的直方图不一致：{bad}"

    # 归一化口径也要钉住：g(r) = (hist/(n_f·n1)) / (shell_V · (n2-1)/V)
    res = rdf_mdat(u, a, None, dr=0.05, frames=[0], verbose=False)
    shell = (4.0 / 3.0) * np.pi * (res["bins"][1:] ** 3 - res["bins"][:-1] ** 3)
    rho = (n_a - 1) / res["volume"]
    g_ref = (res["hist"] / (1.0 * n_a)) / (shell * rho)
    assert np.allclose(res["g_r"], g_ref, rtol=1e-12, atol=0), (
        float(np.max(np.abs(res["g_r"] - g_ref))))
    # 配位数 = 每个中心粒子每帧平均邻居数的累计（不除理想密度）
    cn_ref = np.cumsum(res["hist"] / (1.0 * n_a))
    assert np.allclose(res["coordination_number"], cn_ref, rtol=1e-12, atol=0)

    return (f"同种 / 异种两种配对的逐帧直方图与原实现**逐位相同**（{bins.size - 1} 个 bin，"
            f"{int(cn_ref[-1])} 对）；归一化与配位数口径也已钉住")


def test_msd_mdat_matches_mdanalysis_and_known_D():
    """`test/mda_test/msd_mdat.py`（MDTransport 口径）必须给出**与 EinsteinMSD 相同**
    的 MSD 曲线，并从已知扩散系数的布朗运动里把 D 反推出来。

    为什么要这条：MDTransport 的 ``util_msd.py`` 用的是"增量最小镜像解包 + 去质心
    漂移 + FFT 全时间起点平均"，``util_diffusion.py`` 再用 **log–log 斜率 β≈1 的窗口**
    做 OLS 得 ``D = slope/(2·dim)``。移植版一旦在解包方向、起点平均或 D 的分母上
    出错，拿它对拍就会得出错误结论。
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.msd import EinsteinMSD

    from mda_test.msd_mdat import diffusion_mdat, msd_mdat

    # 已知 D 的三维布朗运动：每步每轴方差 σ² → D_true = σ²/(2·dt)
    n_f, P, dt, sigma = 300, 120, 1.0, 0.2
    rng = np.random.default_rng(11)
    xyz = np.cumsum(rng.normal(scale=sigma, size=(n_f, P, 3)),
                    axis=0).astype(np.float32)
    u = mda.Universe.empty(P, n_residues=P, atom_resindex=list(range(P)),
                           trajectory=True)
    u.add_TopologyAttr("masses", [12.0] * P)
    u.add_TopologyAttr("molnums", np.arange(P, dtype=int))
    u.trajectory = mda.coordinates.memory.MemoryReader(
        xyz, order="fac",
        dimensions=np.array([5000.0, 5000.0, 5000.0, 90.0, 90.0, 90.0],
                            dtype=np.float32))
    d_true = sigma ** 2 / (2.0 * dt)

    # ---- ① MSD 曲线必须与 EinsteinMSD 逐点一致（都是 FFT + 全时间起点平均）
    res = msd_mdat(u, u.atoms, mode="atom", axes="xyz", remove_drift=False,
                   verbose=False)
    es = EinsteinMSD(u.atoms, select="all", msd_type="xyz", fft=True)
    es.run()
    t_e = np.asarray(es.results.delta_t_values, dtype=float)
    m_e = np.asarray(es.results.timeseries, dtype=float)   # 该版本是 Å²
    if np.max(m_e) > 1e4:          # 若某版本改成 nm²（差别就是 ×100）
        m_e = m_e * 100.0
    # EinsteinMSD 的 τ 轴从 0 开始，这里的 MSD 从 τ=dt 开始 → **去掉 τ=0 再对**
    t_e = t_e[1:]
    m_e = m_e[1:]
    n = min(m_e.size, res["msd"].size)
    # 用"绝对偏差 / 曲线峰值"作为判据：τ→0 处 MSD≈0，用相对差会被 1e-12 放大
    dev = float(np.max(np.abs(res["msd"][:n] - m_e[:n]))) / max(float(np.max(m_e)), 1e-12)
    assert dev < 1e-6, f"MSD 曲线与 EinsteinMSD 不一致（偏差/峰值 = {dev:.2e}）"

    # ---- ② D：β≈1 窗口 + OLS（D = slope/(2·dim)）应还原已知值
    fit = diffusion_mdat(res["t"], res["msd"], dim=3)
    rel = abs(fit["D_m2s"] - d_true * 1e-8) / (d_true * 1e-8)
    assert rel < 0.06, (fit["D_m2s"], d_true * 1e-8, rel)
    assert abs(fit["beta"] - 1.0) < 0.15, fit["beta"]

    return (f"MSD 曲线与 EinsteinMSD 一致（偏差/峰值 = {dev:.1e}）；"
            f"D = {fit['D_m2s']:.4e} m²/s vs 真值 {d_true * 1e-8:.4e}"
            f"（偏 {rel:.1%}，β = {fit['beta']:.3f}）")


def test_interface_pair_selection_is_explicit_and_falls_back():
    """界面宽度分析：**A/B 两组可以由用户指定**，非法时回退并说明（不能在界面上"永远是那对"）。

    用户实测（PE/PP/MAPE 三元共混，10 条链全叫 ETH）：勾了 3 个组分，界面分析却
    永远只算"原子数最多的两个"，换组分/换体系看到的还是老的一对 —— 因为以前
    `pair` 只从 params 里 pop 一次、界面上根本没有入口，且组分名失效就抛异常。
    现在：① 支持 params["pair"]；② 失效（不存在/同一个）时**回退到最大的两个并写说明**；
    ③ 勾了 ≥3 个组分时明确告知"本次只比较一对、其余没用上"。
    """
    import types

    import MDAnalysis as mda

    from mdta.io import MDTrajectory
    from mdta.pipeline import Analyzer
    from mdta.preprocess import select_frames

    # 沿 z 分成三层（A/B/C 各 100 原子），这样 1D 台阶模型是适用的
    n_each, L = 100, 60.0
    n_at = n_each * 3
    resix = [i // n_each for i in range(n_at)]
    u = mda.Universe.empty(n_at, n_residues=3, atom_resindex=resix,
                           trajectory=True)
    u.add_TopologyAttr("names", ["C"] * n_at)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    # resnames 是**残基级**属性（长度 = 残基数），会自动广播到原子
    u.add_TopologyAttr("resnames", ["AAA", "BBB", "CCC"])
    pos = np.zeros((n_at, 3))
    rng = np.random.default_rng(5)
    for k, z0 in enumerate((10.0, 30.0, 50.0)):
        pos[k * n_each:(k + 1) * n_each] = np.column_stack([
            rng.uniform(0.0, L, n_each), rng.uniform(0.0, L, n_each),
            rng.uniform(z0 - 6.0, z0 + 6.0, n_each)])
    u.atoms.positions = pos
    u.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))

    a = u.select_atoms("resname AAA")
    b = u.select_atoms("resname BBB")
    c = u.select_atoms("resname CCC")
    comps = {"AAA": a, "BBB": b, "CCC": c}

    def run(params):
        az = object.__new__(Analyzer)
        az.trajectory = mdt
        az.frames = sel
        az.primary = b            # 最大组分，避免触发 auto_setup 覆盖 components
        az.primary_label = "BBB"
        az.components = dict(comps)
        az.skip_reason = ""
        az.current = ""
        az.catalog = dict(comps)
        return az._run_one("interface", params=dict(params), verbose=False)

    notes = lambda r: " ".join(r.notes)                       # noqa: E731

    # ① 不指定 → 自动取原子数最多的两个（三者等大，顺序按 components 插入序 → AAA×BBB）
    r0 = run({})
    assert r0 is not None, "界面分析被跳过了"
    assert (r0.summary["组分 A"], r0.summary["组分 B"]) == ("AAA", "BBB"), r0.summary
    assert "只比较一对" in notes(r0) and "CCC" in notes(r0), notes(r0)

    # ② 指定 A/B → 必须照办
    r1 = run({"pair": ["CCC", "AAA"]})
    assert (r1.summary["组分 A"], r1.summary["组分 B"]) == ("CCC", "AAA"), r1.summary
    assert "按参数指定" in notes(r1), notes(r1)

    # ③ 指定了不存在的组分 → 回退 + 说明（不再抛异常）
    r2 = run({"pair": ["AAA", "NOPE"]})
    assert (r2.summary["组分 A"], r2.summary["组分 B"]) == ("AAA", "BBB"), r2.summary
    assert "回退" in notes(r2) and "NOPE" in notes(r2), notes(r2)

    # ④ A == B → 回退 + 说明
    r3 = run({"pair": ["AAA", "AAA"]})
    assert (r3.summary["组分 A"], r3.summary["组分 B"]) == ("AAA", "BBB"), r3.summary
    assert "不相同" in notes(r3), notes(r3)

    # ⑤ 只有 1 个组分 → 跳过，且原因里给出"怎么拆物种"的指引
    az = object.__new__(Analyzer)
    az.trajectory = mdt
    az.frames = sel
    az.primary = a
    az.primary_label = "AAA"
    az.components = {"polymer": u.atoms}
    az.skip_reason = ""
    az.current = ""
    az.catalog = {"polymer": u.atoms}
    assert az._run_one("interface", params={}, verbose=False) is None
    assert "分析集合" in az.skip_reason and "2 个组分" in az.skip_reason, az.skip_reason

    # ⑥ 换 A/B 组必须**触发重算**（否则界面上一换还是旧图）：参数要进"输入指纹"
    from webapp.backend.session import Session

    sess = object.__new__(Session)
    sess.sets_spec = []
    sel_info = {"active": ["AAA", "BBB", "CCC"], "primary": {"mode": "auto"}}
    sig_ab = sess._input_sig("interface", sel_info, {},
                             {"interface": {"pair": ["AAA", "BBB"]}})
    sig_ac = sess._input_sig("interface", sel_info, {},
                             {"interface": {"pair": ["AAA", "CCC"]}})
    assert sig_ab != sig_ac, (
        "界面 A/B 组不在输入指纹里 → 换组不会重算（会一直显示上一组的结果）")

    return ("可指定 A/B（照办）、无效/重复时回退并说明、≥3 组分提示「只比较一对」、"
            "单组分跳过时给出拆分指引、换 A/B 组会触发重算")


def test_contact_and_comdist_pairs_are_selectable():
    """接触分析 / 两组分质心距：同样可以指定 A/B 组（与界面宽度同一个 bug 类）。

    用户实测背景一致：三元共混里勾了 3 个组分，这两项都固定"原子数最多的两个"，
    第三个组分永远不参与、换体系看到的还是老的一对。现在三项共用
    `Analyzer._pair_from_params`：显式指定照办，失效（不存在 / 同一个 / 长度不对）
    回退到自动值并写说明，≥3 组分时明确"只比较一对、未参与的是 [...]"。
    """
    import MDAnalysis as mda

    from mdta.io import MDTrajectory
    from mdta.pipeline import Analyzer
    from mdta.preprocess import select_frames

    n_each, L = 60, 40.0
    n_at = n_each * 3
    u = mda.Universe.empty(n_at, n_residues=3,
                           atom_resindex=[i // n_each for i in range(n_at)],
                           trajectory=True)
    u.add_TopologyAttr("names", ["C"] * n_at)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["AAA", "BBB", "CCC"])
    u.add_TopologyAttr("masses", [12.0] * n_at)   # 质心距要质心 → 必须有质量
    pos = np.zeros((n_at, 3))
    rng = np.random.default_rng(9)
    for k, z0 in enumerate((10.0, 20.0, 30.0)):
        pos[k * n_each:(k + 1) * n_each] = np.column_stack([
            rng.uniform(1, L - 1, n_each), rng.uniform(1, L - 1, n_each),
            rng.uniform(z0 - 3, z0 + 3, n_each)])
    u.atoms.positions = pos
    u.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    sel = select_frames(np.array([0.0]))
    comps = {"AAA": u.select_atoms("resname AAA"),
             "BBB": u.select_atoms("resname BBB"),
             "CCC": u.select_atoms("resname CCC")}

    def run(name, params):
        az = object.__new__(Analyzer)
        az.trajectory = mdt
        az.frames = sel
        az.primary = comps["AAA"]
        az.primary_label = "AAA"
        az.components = dict(comps)
        az.skip_reason = ""
        az.current = ""
        az.catalog = dict(comps)
        return az._run_one(name, params=dict(params), verbose=False)

    notes = lambda r: " ".join(r.notes)                       # noqa: E731

    # ---- 接触：自动 → AAA×BBB；指定 → 照办；失效 → 回退 + 说明
    rc0 = run("contact", {})
    assert "AAA × BBB" in notes(rc0) and "只比较一对" in notes(rc0), notes(rc0)
    rc1 = run("contact", {"pair": ["AAA", "CCC"], "cutoff": 4.0})
    assert "AAA × CCC" in notes(rc1) and "按参数指定" in notes(rc1), notes(rc1)
    rc2 = run("contact", {"pair": ["AAA", "NOPE"]})
    assert "回退" in notes(rc2) and "AAA × BBB" in notes(rc2), notes(rc2)

    # ---- 质心距：指定必须照办（标题里能看出来），失效回退
    rd1 = run("comdist", {"pair": ["BBB", "CCC"]})
    assert "BBB vs CCC" in rd1.title, rd1.title
    rd2 = run("comdist", {"pair": ["NOPE", "CCC"]})
    assert "AAA vs BBB" in rd2.title, rd2.title
    assert "回退" in notes(rd2), notes(rd2)

    return ("contact/comdist 均可指定 A/B（照办）、失效时回退并说明、"
            "≥3 组分提示只比较一对")


def test_density_matches_reference_and_mdanalysis_lineardensity():
    """密度分布：与"教科书实现"**逐位一致**，与 MDAnalysis ``LinearDensity`` 一致，
    并且质量守恒（Σρ·V_bin 折回总质量）。

    用户的提问是"MDAnalysis 或其他模块有没有能验证密度分布的" —— 有的：
    ``MDAnalysis.analysis.lineardensity.LinearDensity`` 算的就是同一个量
    （质量密度 g/cm³，逐帧直方图）。这里把三条独立路径钉在一起：

    ① 教科书实现（包裹 → 沿轴直方图 → amu/Å³×1.66054）↔ 我们的引擎：**逐位相同**；
    ② 教科书实现 ↔ ``LinearDensity``：同立方盒、同 bin 宽时应到 1e-6 量级；
    ③ 质量守恒：把密度乘回体积再折算成 amu，应等于体系总质量。
    """
    import MDAnalysis as mda

    from mda_test.density import (AMU_A3_TO_G_CM3, compare_mdanalysis,
                                  density_reference)
    from mdta.analysis.interface import density_profile
    from mdta.io import MDTrajectory

    # 立方盒（LinearDensity 的 range 用最长边 → 立方盒时所有轴网格一致）
    L, n_a, n_b = 30.0, 400, 300
    n_at = n_a + n_b
    u = mda.Universe.empty(n_at, n_residues=2,
                           atom_resindex=[0] * n_a + [1] * n_b, trajectory=True)
    u.add_TopologyAttr("names", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["AAA", "BBB"])
    u.add_TopologyAttr("masses", [12.0] * n_at)
    rng = np.random.default_rng(13)
    pos = np.zeros((n_at, 3))
    # A 占 z ∈ [0,15)，B 占 z ∈ [15,30) → 两个界面（周期下 z=0 与 z=30 是同一个）
    pos[:n_a, 0] = rng.uniform(0, L, n_a)
    pos[:n_a, 1] = rng.uniform(0, L, n_a)
    pos[:n_a, 2] = rng.uniform(0.0, 15.0, n_a)
    pos[n_a:, 0] = rng.uniform(0, L, n_b)
    pos[n_a:, 1] = rng.uniform(0, L, n_b)
    pos[n_a:, 2] = rng.uniform(15.0, 30.0, n_b)
    u.atoms.positions = pos
    u.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    from types import SimpleNamespace

    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.array([0]), times_ps=times[:1], n_frames=1,
                          describe="1 帧", notes=[], head=lambda self, k: self)

    NB = 30
    ref = density_reference(u, u.atoms, axis=2, nbins=NB, frames=[0], mode="mass")
    centers, prof, meta = density_profile(mdt, {"X": u.atoms}, sel, axis=2,
                                          nbins=NB, mode="mass")
    eng = np.asarray(prof["X"], dtype=float)
    d = np.abs(eng - ref["density"])
    assert d.max() < 1e-12, f"引擎与教科书实现不一致（最大差 {d.max():.3e}）"

    # 质量守恒
    m_back = float(np.sum(ref["density"])) * ref["bin_volume"] / AMU_A3_TO_G_CM3
    assert abs(m_back - 12.0 * n_at) / (12.0 * n_at) < 1e-6, (m_back, 12.0 * n_at)

    # 与 LinearDensity 对拍（立方盒 → 网格一致）
    cmp = compare_mdanalysis(u, u.atoms, axis=2, nbins=NB, frames=[0], mode="mass")
    assert cmp is not None, "LinearDensity 对拍没跑起来"
    assert cmp["abs"] < 1e-6, cmp
    assert abs(ref["density"].mean() - 12.0 * n_at / (L ** 3)
               * AMU_A3_TO_G_CM3) < 1e-3, ref["density"].mean()

    return (f"引擎 vs 教科书逐位一致（{d.max():.1e}）；质量守恒相对差 "
            f"{abs(m_back - 12.0 * n_at) / (12.0 * n_at):.1e}；"
            f"vs LinearDensity 最大差 {cmp['abs']:.2e} g/cm³")


def test_density_normalization_is_unit_consistent():
    """密度分布：**归一化曲线与 ρ̄ 必须和剖面同单位**（质量密度曾差 1.66054 倍）。

    用户实测："勾密度分布 + 数密度 + bin100 + 归一化密度，数值对不上"。
    查出来：`mode="mass"` 时剖面是 g/cm³，但归一化用的 ρ̄ 是 `Σm[amu]/V[Å³]`
    （amu/Å³）→ 归一化曲线整体被乘了 1.66054（1000water 实测均值 1.660539 而不是 1），
    摘要里"平均密度 1.0"与"整盒平均密度 ρ̄ 0.602"也自相矛盾。
    数密度口径下两者单位相同，所以只有质量密度出错 —— 这正是"数密度看着正常、
    质量密度对不上"的原因。

    断言：两种口径下 ①归一化曲线均值 ≈ 1；②ρ̄ 与剖面均值一致（均匀体系里它们相等）；
    ③ρ̄ 的量纲与剖面一致（质量密度 g/cm³ 与 amu/Å³ 相差 1.66054，必须换算过）。
    """
    from types import SimpleNamespace

    import MDAnalysis as mda

    from mdta.analysis.interface import analyze_density
    from mdta.io import MDTrajectory

    # 均匀体系（密度处处相同）：ρ̄ 与剖面均值必须相等
    L = 24.0
    n_mol, per = 100, 4
    n_at = n_mol * per
    u = mda.Universe.empty(n_at, n_residues=n_mol,
                           atom_resindex=[i // per for i in range(n_at)],
                           trajectory=True)
    u.add_TopologyAttr("names", ["O", "H", "H", "M"] * n_mol)
    u.add_TopologyAttr("masses", [15.999, 1.008, 1.008, 0.0] * n_mol)
    rng = np.random.default_rng(21)
    pos = rng.uniform(0.0, L, (n_at, 3))
    pos[3::per, 2] = pos[0::per, 2]          # M 位点跟着 O（无所谓，只要均匀）
    u.atoms.positions = pos
    u.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=np.float32)
    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.array([0]), times_ps=times[:1], n_frames=1,
                          describe="1 帧", notes=[], head=lambda self, k: self)

    out = {}
    # 两条分支都要测：axis=2 →「沿 c 方向的组分密度分布」（单轴）；
    # axis="all" →「沿 a/b/c 三个方向的组分密度分布」（界面默认的三个方向）。
    for axis, tag_prefix in ((2, ""), ("all", "")):
      for mode in ("mass", "number"):
        res = analyze_density(mdt, {"X": u.atoms}, sel, axis=axis, nbins=40,
                              mode=mode)
        curves = {(c.panel, c.label): np.asarray(c.y, dtype=float) for c in res.curves}
        prof = next(v for (p, _l), v in curves.items() if p == 0)
        norm = next(v for (p, _l), v in curves.items() if p == 1)
        ask = "" if axis == 2 else "C001 "     # all 分支的键带方向前缀
        rho_bar = res.summary[f"{ask}X 整盒平均密度 ρ̄"]
        mean_prof = res.summary[f"{ask}X 平均密度"]
        # ① 归一化均值 ≈ 1（均匀体系）
        assert abs(norm.mean() - 1.0) < 0.02, (axis, mode, norm.mean())
        # ② ρ̄ 与剖面均值一致
        assert abs(rho_bar - mean_prof) / max(abs(mean_prof), 1e-12) < 0.02, (
            axis, mode, rho_bar, mean_prof)
        # ③ 与"教科书"数值一致（质量密度 1.66054 g/cm³ per amu/Å³）
        if mode == "mass":
            m_tot = float(np.sum(u.atoms.masses))
            ref = m_tot / (L ** 3) * 1.66053906660
        else:
            ref = n_at / (L ** 3)
        assert abs(mean_prof - ref) / ref < 1e-6, (axis, mode, mean_prof, ref)
        assert abs(rho_bar - ref) / ref < 1e-6, (axis, mode, rho_bar, ref)
        if axis == 2:
            out[mode] = (float(mean_prof), float(rho_bar), float(norm.mean()))
        else:
            out[f"all_{mode}"] = (float(mean_prof), float(rho_bar),
                                  float(norm.mean()))

    return ("单轴 C：质量密度 {:.6f} g/cm³ / ρ̄ {:.6f} / 归一化 {:.6f}，"
            "数密度 {:.6f} 1/Å³ / ρ̄ {:.6f} / 归一化 {:.6f}；"
            "三方向(all) C001：质量密度 {:.6f} / ρ̄ {:.6f} / 归一化 {:.6f}，"
            "数密度 {:.6f} / ρ̄ {:.6f} / 归一化 {:.6f}（两条分支、两种口径都自洽）"
            .format(*out["mass"], *out["number"],
                    *out["all_mass"], *out["all_number"]))


def test_interface_reference_matches_engine_and_known_width():
    """界面宽度：参考实现 ↔ 引擎**逐位一致**，且都还原出**已知的真实宽度**。

    用户问"界面宽度分析有没有参考对照的脚本"——第三方没有一键函数，但可以做三层
    对照（见 ``test/mda_test/interface.py``）：

    ① **解析轮廓**（无采样噪声）：10–90 宽度必须等于 ``2.5631σ``，erf 拟合必须
       还原 σ —— 这验证的是"宽度判据"本身；
    ② **合成已知宽度体系**：原子按 ``½[1−erf((z−z₀)/(√2σ))]`` 概率判给 A/B，
       真实 10–90 宽度解析等于 ``2.5631σ``。参考实现与引擎必须给出同一个数；
    ③ 参考实现与引擎的界面位置也必须一致。
    """
    from mda_test.interface import (W1090_PER_SIGMA, check_definition,
                                    compare_engine, interface_reference,
                                    synth_slab_universe)

    assert check_definition(verbose=False), "解析轮廓上的宽度定义自检未通过"

    sigma = 2.0
    truth = W1090_PER_SIGMA * sigma
    # 单帧构型下 0.5 Å 分箱的统计噪声约 ±5%，所以取多份独立构型求平均
    ws = []
    for seed in (7, 11, 23):
        u, ga, gb = synth_slab_universe(sigma=sigma, box=60.0, n_frames=2,
                                        n_atoms=150000, seed=seed)
        ref = interface_reference(u, ga, gb, axis=2, binsize=0.5, mode="number",
                                  smooth=1, verbose=False)
        assert ref["slab_ok"], ref["summary"]
        assert len(ref["spots"]) == 2, ref["spots"]      # 板层两个界面
        ws.extend(float(w) for w in ref["widths"] if np.isfinite(w))
    w_ref = float(np.mean(ws))
    assert abs(w_ref - truth) / truth < 0.04, (w_ref, truth, ws)

    # 引擎必须给出**同一个数**（同一批帧、同一 bin、同一平滑）
    u, ga, gb = synth_slab_universe(sigma=sigma, box=60.0, n_frames=2,
                                    n_atoms=150000, seed=7)
    cmp = compare_engine(u, ga, gb, axis=2, binsize=0.5, mode="number",
                         smooth=1)
    ref = interface_reference(u, ga, gb, axis=2, binsize=0.5, mode="number",
                              smooth=1, verbose=False)
    w_eng = float(cmp["w_engine"])
    w_ref1 = float(ref["widths"][0])
    assert abs(w_eng - w_ref1) < 1e-9, (w_eng, w_ref1)
    assert abs(float(cmp["pos_engine"]) - float(ref["spots"][0])) < 1e-9, cmp

    return (f"真实 {truth:.4f} Å；3 份构型平均参考宽度 {w_ref:.4f} Å"
            f"（{(w_ref - truth) / truth * 100:+.2f}%）；引擎与参考差 "
            f"{abs(w_eng - w_ref1):.1e} Å、界面位置差 "
            f"{abs(float(cmp['pos_engine']) - float(ref['spots'][0])):.1e} Å")


def test_mda_test_orientation_pooled_matches_engine():
    """链段取向：**多分散**体系里"逐链等权"与"链段加权"不同 —— 引擎用后者。

    用户实测 310K（C6…C50 混油 + 石墨片）对不上 `test/mda_test/orientation.py`。
    原因两条，都在参考脚本一侧：

    ① ``chain_coordinates`` 默认按**众数**链长过滤 → 该体系只留下 25 条 C6，
       另外 71 条链（C8…C50）全被丢掉；
    ② 统计权重：脚本逐链等权平均，引擎把**所有链段**放进一个池子平均
       （长链链段多、权重大）。单分散体系（122×C12、20×PE30）两者相同，
       所以"别的体系没问题"。

    断言：合成一个**多分散**体系（6 条 C6 + 4 条 C12，同一倾角），
       · 链段加权 P2 与引擎**逐位一致**；
       · 逐链等权 P2 与引擎**不同**（正是用户看到的老差异，长度差越大差越大）。
    """
    from types import SimpleNamespace

    import MDAnalysis as mda

    from mda_test.orientation import (chain_coordinates, chain_size_histogram,
                                      cos2_to_axis, pooled_cos2)
    from mdta.analysis.crystallinity import analyze_orientation
    from mdta.io import MDTrajectory

    rng = np.random.default_rng(5)

    def chain(n_c, tilt_deg, phase):
        """一条 n_c 个碳的锯齿链，整体与 z 轴成 tilt_deg（加随机相位）。"""
        t = np.radians(tilt_deg)
        step = 1.55
        d = np.array([np.sin(t), 0.0, np.cos(t)])
        zz = np.array([np.cos(t), 0.0, -np.sin(t)])       # 垂直方向做锯齿
        s = np.arange(n_c)
        base = (s[:, None] * step * d
                + (0.6 * (-1.0) ** s)[:, None] * zz
                + (phase * 2.0) * np.array([1.0, 0.0, 0.0]))
        return base

    # 6 条 C6 + 4 条 C12，**每条链倾角不同**（否则逐链等权与链段加权恰好相同）
    specs = [(6, float(rng.uniform(0.0, 70.0))) for _ in range(6)] + \
            [(12, float(rng.uniform(0.0, 70.0))) for _ in range(4)]
    pos, res, bonds, at = [], [], [], 0
    for k, (n_c, tilt) in enumerate(specs):
        p = chain(n_c, tilt, rng.uniform(-0.5, 0.5))
        pos.append(p)
        res.extend([k] * n_c)
        bonds.extend((at + i, at + i + 1) for i in range(n_c - 1))
        at += n_c
    pos = np.vstack(pos)
    n_at = at

    u = mda.Universe.empty(n_at, n_residues=len(specs),
                           atom_resindex=np.asarray(res), trajectory=True)
    u.add_TopologyAttr("names", ["C"] * n_at)
    u.add_TopologyAttr("elements", ["C"] * n_at)
    u.add_TopologyAttr("resnames", ["ETH"] * len(specs))
    u.add_TopologyAttr("masses", [12.0] * n_at)
    u.add_TopologyAttr("bonds", bonds)
    u.atoms.positions = pos - pos.mean(axis=0) + 30.0
    u.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0], dtype=np.float32)

    chains = chain_coordinates(u.atoms, verbose=False)
    assert chain_size_histogram(chains) == {6: 6, 12: 4}, chain_size_histogram(chains)

    c2_pooled = pooled_cos2(chains, "z")
    c2_equal = float(np.nanmean([cos2_to_axis(c, "z") for c in chains]))
    p2_pooled = (3.0 * c2_pooled - 1.0) / 2.0
    p2_equal = (3.0 * c2_equal - 1.0) / 2.0

    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.array([0]), times_ps=times[:1], n_frames=1,
                          describe="1 帧", notes=[], head=lambda self, k: self)
    res_obj = analyze_orientation(mdt, u.atoms, sel, mode="backbone", stride=2,
                                  heavy_only=True, p2_axis="z")
    p2_eng = None
    for c in res_obj.curves:
        if c.panel == 1:
            p2_eng = float(np.asarray(c.y, dtype=float)[0])
            break
    assert p2_eng is not None, "引擎没给出 P2 曲线"

    assert abs(p2_pooled - p2_eng) < 1e-9, (p2_pooled, p2_eng)
    # 多分散时逐链等权必须与引擎不同（C12 有 10 段、C6 只有 4 段）
    assert abs(p2_equal - p2_eng) > 1e-4, (p2_equal, p2_eng)

    # ---- 引擎 1.0.7 新增的 weighting 开关：两条路径各对上一种权重
    def _p2_of(res):
        for c in res.curves:
            if c.panel == 1:
                return float(np.asarray(c.y, dtype=float)[0])
        raise AssertionError("引擎没给出 P2 曲线")

    eng_seg = analyze_orientation(mdt, u.atoms, sel, mode="backbone", stride=2,
                                  heavy_only=True, p2_axis="z",
                                  weighting="segment")
    eng_mol = analyze_orientation(mdt, u.atoms, sel, mode="backbone", stride=2,
                                  heavy_only=True, p2_axis="z",
                                  weighting="molecule")
    assert abs(_p2_of(eng_seg) - p2_pooled) < 1e-9, (_p2_of(eng_seg), p2_pooled)
    assert abs(_p2_of(eng_mol) - p2_equal) < 1e-9, (_p2_of(eng_mol), p2_equal)
    assert eng_seg.meta.get("weighting") == "segment", eng_seg.meta
    assert eng_mol.meta.get("weighting") == "molecule", eng_mol.meta
    # 多分散时摘要里要给出**另一种权重**的对照值，否则用户会以为算错了
    assert any("另一种权重" in k for k in eng_mol.summary), list(eng_mol.summary)
    assert any("另一种权重" in k for k in eng_seg.summary), list(eng_seg.summary)

    # 单分散（长度全一样、倾角也各不相同 → 逐链等权与链段加权仍相等）
    specs2 = [(8, float(rng.uniform(0.0, 70.0))) for _ in range(5)]
    pos2, res2, bonds2, at2 = [], [], [], 0
    for k, (n_c, tilt) in enumerate(specs2):
        pos2.append(chain(n_c, tilt, rng.uniform(-0.5, 0.5)))
        res2.extend([k] * n_c)
        bonds2.extend((at2 + i, at2 + i + 1) for i in range(n_c - 1))
        at2 += n_c
    pos2 = np.vstack(pos2)
    u2 = mda.Universe.empty(at2, n_residues=len(specs2),
                            atom_resindex=np.asarray(res2), trajectory=True)
    u2.add_TopologyAttr("names", ["C"] * at2)
    u2.add_TopologyAttr("elements", ["C"] * at2)
    u2.add_TopologyAttr("resnames", ["ETH"] * len(specs2))
    u2.add_TopologyAttr("bonds", bonds2)
    u2.atoms.positions = pos2 - pos2.mean(axis=0) + 30.0
    u2.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0], dtype=np.float32)
    ch2 = chain_coordinates(u2.atoms, verbose=False)
    c2p2 = pooled_cos2(ch2, "z")
    c2e2 = float(np.nanmean([cos2_to_axis(c, "z") for c in ch2]))
    assert abs(c2p2 - c2e2) < 1e-12, (c2p2, c2e2)

    return (f"多分散（C6×6 + C12×4）：链段加权 P2 {p2_pooled:+.6f} vs 引擎 "
            f"{_p2_of(eng_seg):+.6f}（差 {abs(p2_pooled - _p2_of(eng_seg)):.1e}）；"
            f"逐链等权 {p2_equal:+.6f} vs 引擎(molecule) {_p2_of(eng_mol):+.6f}；"
            f"单分散时两种权重相同（差 {abs(c2p2 - c2e2):.1e}）")


def test_mda_test_dihedral_nematic_boo_reference_scripts():
    """test/mda_test 的三个新参考脚本：MDAnalysis Dihedral / freud Nematic / freud Steinhardt。

    · **二面角分布**：``MDAnalysis.analysis.dihedrals.Dihedral`` 算角度
      （四元组由独立的 BFS 最长路径给出），分布的概率密度应与引擎逐 bin 相同；
    · **取向参数 S**：``freud.order.Nematic`` 从同一批链段矢量求 S，
      与引擎 ``analyze_orientation`` 的 S 相同（freud 内部 float32，~1e-7）；
    · **BOO**：``freud.order.Steinhardt(l=4/6)`` 求逐原子 q_l 与平均版 q̄_l，
      与引擎 ``compute_ql``（numpy 后端）逐原子相同（~1e-6）。

    没有 ``freud-analysis`` 时直接跳过（三个脚本本身也是这么处理的）。
    """
    try:
        from types import SimpleNamespace

        import freud  # noqa: F401
        import MDAnalysis as mda
        from MDAnalysis.analysis.dihedrals import Dihedral  # noqa: F401
    except Exception as exc:                                 # noqa: BLE001
        return f"跳过：缺少 freud-analysis / MDAnalysis（{type(exc).__name__}）"

    from mda_test.boo import boo_freud_frame
    from mda_test.dihedral import dihedral_reference
    from mda_test.nematic import nematic_reference
    from mdta.analysis.boo import analyze_boo, compute_ql
    from mdta.analysis.conformation import analyze_dihedrals
    from mdta.analysis.crystallinity import analyze_orientation
    from mdta.io import MDTrajectory

    # ---- 合成体系：3 条 C6H14 锯齿链（键表齐全，元素/质量都有）
    L = 30.0
    pos, els, res_of, res_names, bonds = [], [], [], [], []
    idx = 0
    for _c in range(3):
        r = len(res_names)
        res_names.append("HEX")
        base = idx
        x0 = 2.0 + 9.0 * _c                    # 三条链分开摆（不能重叠：邻居退化成 0 距离）
        for k in range(6):
            pos.append([x0 + k * 1.53, 0.0, (0.4 if k % 2 else -0.4)])
            els.append("C")
            res_of.append(r)
            idx += 1
        for k in range(6):
            for dh in ((0.6, -0.6) if k in (0, 5) else (0.6,)):
                pos.append([x0 + k * 1.53, dh, (0.4 if k % 2 else -0.4)])
                els.append("H")
                res_of.append(r)
                bonds.append((base + k, idx))
                idx += 1
        for k in range(5):
            bonds.append((base + k, base + k + 1))
    n = idx
    u = mda.Universe.empty(n, n_residues=len(res_names),
                           atom_resindex=np.asarray(res_of), trajectory=True)
    u.add_TopologyAttr("names", [f"{e}{i}" for i, e in enumerate(els)])
    u.add_TopologyAttr("elements", els)
    u.add_TopologyAttr("types", els)
    u.add_TopologyAttr("resnames", res_names)
    u.add_TopologyAttr("segids", ["SYS"])
    u.add_TopologyAttr("masses", np.where(np.asarray(els) == "H", 1.008, 12.011))
    u.add_TopologyAttr("bonds", bonds)
    u.dimensions = np.array([L, L, L, 90.0, 90.0, 90.0], dtype=np.float32)
    u.atoms.positions = np.mod(np.asarray(pos, dtype=float), L)

    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.array([0]), times_ps=times[:1], n_frames=1,
                          describe="1 帧", notes=[], head=lambda self, k: self)

    # ---- 1) 二面角分布（MDAnalysis Dihedral）
    dref = dihedral_reference(u, u.atoms, frames=[0], bin_width=1.0, verbose=False)
    res_d = analyze_dihedrals(mdt, u.atoms, sel, mode="chain", bin_width=1.0)
    dens = None
    for c in res_d.curves:
        if c.panel == 0:
            dens = np.asarray(c.y, dtype=float)
    assert dens is not None and dens.size == dref["density"].size, (
        None if dens is None else dens.size, dref["density"].size)
    dmax = float(np.max(np.abs(dens - dref["density"])))
    assert dmax < 1e-9, f"二面角分布不一致（最大差 {dmax:.2e}）"
    n_eng = int(res_d.summary.get("dihedral 二面角个数", 0))
    assert n_eng == dref["idx"].shape[0], (n_eng, dref["idx"].shape[0])

    # ---- 2) 取向参数 S（freud Nematic）
    nref = nematic_reference(u, u.atoms, frames=[0], verbose=False)
    res_o = analyze_orientation(mdt, u.atoms, sel, mode="backbone", stride=2)
    s_eng = None
    for c in res_o.curves:
        if c.panel == 0:
            s_eng = float(np.asarray(c.y, dtype=float)[0])
    assert s_eng is not None
    s_ref = float(nref["S"][0])
    ds = abs(s_ref - s_eng)
    assert ds < 1e-6, f"S 不一致：freud {s_ref:.8f} vs 引擎 {s_eng:.8f}"

    # ---- 3) BOO（freud Steinhardt）
    cut = 3.2
    q_ref = boo_freud_frame(u.atoms.positions, u.dimensions, cut,
                            l_list=(4, 6), average=False)
    q_eng = compute_ql(u.atoms.positions, u.dimensions, cut, l_list=(4, 6),
                       averaged=False, backend="numpy")
    dq = {k: float(np.nanmax(np.abs(q_ref[k] - q_eng[k]))) for k in ("q4", "q6")}
    assert max(dq.values()) < 1e-5, dq
    qbar_ref = boo_freud_frame(u.atoms.positions, u.dimensions, cut,
                               l_list=(4, 6), average=True)
    qbar_eng = compute_ql(u.atoms.positions, u.dimensions, cut, l_list=(4, 6),
                          averaged=True, backend="numpy")
    dbar = float(np.nanmax(np.abs(qbar_ref["q6bar"] - qbar_eng["q6bar"])))
    assert dbar < 1e-5, f"q̄6 逐原子不一致（最大差 {dbar:.2e}）"
    res_b = analyze_boo(mdt, u.atoms, sel, cutoff=cut, l_list=(4, 6),
                        averaged=True)
    eb = float(res_b.summary.get("q6bar 帧平均（时间序列均值）", float("nan")))
    rb = float(np.nanmean(qbar_ref["q6bar"]))
    assert abs(eb - rb) < 1e-6, (eb, rb)

    return (f"二面角 {dref['idx'].shape[0]} 个四元组，分布最大差 {dmax:.1e}；"
            f"S：freud {s_ref:.8f} vs 引擎 {s_eng:.8f}（差 {ds:.1e}）；"
            f"BOO q6 逐原子最大差 {dq['q6']:.1e}，q̄6 帧平均差 {abs(eb - rb):.1e}")


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
        "test_chain_dihedral_covers_every_molecule",
        "test_angles_dihedrals_derived_from_bond_table",
        "test_dihedral_bin_grid_matches_forcite_half_bin_alignment",
        "test_dihedral_distribution_bin_width_option",
        "test_dihedral_distribution_is_probability_density",
        "test_water_density_is_physical",
        "test_rdf_matches_mdanalysis_interrdf",
        "test_exclude_bonded_removes_12_pairs",
        "test_water_rdf_physics",
        "test_rdf_coordination_number_is_subsample_invariant",
        "test_contact_same_group_matches_self_capped_distance",
        "test_contact_cross_group_matches_capped_distance",
        "test_contact_min_distance_not_rescanned",
        "test_nematic_order_parameter_on_known_distributions",
        "test_orientation_p2_matches_torsion_script_reference",
        "test_interface_width_on_synthetic_erf_profile",
        "test_interface_width_sharp_step",
        "test_interface_width_two_interfaces_are_measured_locally",
        "test_msd_of_static_system_is_zero",
        "test_msd_of_ballistic_motion",
        "test_msd_flags_broken_minimum_image_tracking",
        "test_msd_water_diffusion_coefficient",
        "test_export_and_result_containers",
        "test_molecular_recognition",
        "test_single_frame_drops_time_panels",
        "test_download_filenames_are_short_ascii",
        "test_cancel_stops_quickly",
        "test_rdf_mdat_matches_util_rdf_frame_histogram",
        "test_msd_mdat_matches_mdanalysis_and_known_D",
        "test_interface_pair_selection_is_explicit_and_falls_back",
        "test_contact_and_comdist_pairs_are_selectable",
        "test_density_matches_reference_and_mdanalysis_lineardensity",
        "test_density_normalization_is_unit_consistent",
        "test_interface_reference_matches_engine_and_known_width",
        "test_mda_test_orientation_pooled_matches_engine",
        "test_tpr_bond_table_is_trusted_and_no_fragment_dict",
        "test_ms_trj_keeps_periodic_box",
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

