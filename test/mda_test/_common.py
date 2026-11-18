# -*- coding: utf-8 -*-
"""公共部分：加载体系（Materials Studio `.xsd` + 转换好的 `.trr`）、补质量、
帧窗口、分子编号、最小镜像距离、统一导出 CSV。

其余每个模块（`rg` / `ree` / `msd` / `rdf` / `comdis` / `contact`）都只负责自己的
算法，用

    from ._common import load, save_csv

拿 `(u, ag)`，或者用各自的 `python test/mda_test/xxx.py` 单独跑。

可用环境变量（三个都可不设，用下面写死的默认值）：

| 变量 | 作用 |
|---|---|
| `MDTA_MDA_XSD` | 拓扑文件（`.xsd` / `.pdb` / `.gro` / `.tpr` …） |
| `MDTA_MDA_TRR` | 轨迹文件（`.trr` / `.xtc` / `.trj` …），可留空 = 单帧结构 |
| `MDTA_MDA_SELECT` | 默认选择串，默认 `resname ETH` |
| `MDTA_MDA_FRAMES` | 最多用多少帧（大轨迹先用 20 试，省时间） |
| `MDTA_MDA_OUT` | CSV 输出目录，默认当前目录 |
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ 默认路径
_ROOT = Path(__file__).resolve().parents[2]          # 仓库根 C:\temp\MDT
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

#: 默认体系：**20PE30**（20 条 PE30 链 / 3640 原子 / 301 帧 / 立方盒 33.6 Å）
#: 拓扑 + Materials Studio 的 `.trj`（`.trj` 会自动转 TRR，有缓存直接复用）。
#: 换体系不用改代码：设 MDTA_MDA_XSD / MDTA_MDA_TRR 即可。
DEFAULT_XSD = os.environ.get("MDTA_MDA_XSD",
                             r"C:\temp\dataset\20PE30\20PE30.xsd")
DEFAULT_TRR = os.environ.get("MDTA_MDA_TRR",
                             r"C:\temp\dataset\20PE30\20PE30.trj")
DEFAULT_SELECT = os.environ.get("MDTA_MDA_SELECT", "resname ETH")

#: 元素 → 质量（拓扑里没有质量时按原子名首字母补；查不到给 1.008 不崩）
ELEMENT_MASS = {
    "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007,
    "S": 32.06, "P": 30.97, "Na": 22.99, "Cl": 35.45,
    "F": 18.998, "B": 10.81, "Si": 28.085, "Br": 79.90, "I": 126.90,
}


# ------------------------------------------------------------------ 输出
def out_dir() -> str:
    """CSV 的输出目录（默认当前目录，可用 `MDTA_MDA_OUT` 指定）。"""
    d = os.environ.get("MDTA_MDA_OUT") or os.getcwd()
    os.makedirs(d, exist_ok=True)
    return d


def out_path(name: str) -> str:
    """把文件名拼到输出目录上（`name` 用 ASCII，别用中文）。"""
    return os.path.join(out_dir(), name)


def save_csv(name: str, data, header: str, fmt: str) -> str:
    """统一导出 CSV：一行一个数据点，`fmt` 给每列格式（如 `"%.4f,%.6f"`）。

    统一用 **UTF-8 with BOM** 写：表头里常有中文（`P2=(3*cos2-1)/2 二阶序参数…`），
    不指定编码时 Windows 会按 GBK 写，别的程序（包括本包自己的读取）就乱码；
    带 BOM 的 UTF-8 还能让 Excel 直接正确打开。
    """
    path = out_path(name)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        np.savetxt(fh, np.column_stack(data), header=header, fmt=fmt, comments="")
    print(f"✅ 已导出 {path}")
    return path


# ------------------------------------------------------------------ 加载体系
def fill_masses(u, verbose: bool = True):
    """拓扑没带质量时按**原子名首字母**查元素表补上（查不到给 1.008，不崩）。

    为什么必须补：`radius_of_gyration()` / `center_of_mass()` / 质量密度都依赖质量，
    没有质量会直接抛 `NoDataError`。Materials Studio 导出的 `.xsd` 通常没有质量。
    """
    masses = np.zeros(u.atoms.n_atoms)
    unknown = set()
    for atom in u.atoms:
        key = atom.name[0]
        if key not in ELEMENT_MASS:
            unknown.add(atom.name)
            masses[atom.index] = 1.008
        else:
            masses[atom.index] = ELEMENT_MASS[key]
    u.add_TopologyAttr("masses", masses)
    if unknown and verbose:
        print(f"⚠️ 未知命名原子（已赋默认质量 1.008）: {sorted(unknown)}")
    return masses


def build_universe(xsd: str | None = None, trr: str | None = None,
                   select: str | None = None, verbose: bool = True):
    """加载 `(拓扑, 轨迹)` → `(u, ag)`；轨迹为空时是单帧结构。

    · `.xsd/.xtd` 走本工具的 `mdta.ms_structure`（Materials Studio 文档 → Universe）；
    · 轨迹是 Materials Studio 的 **`.trj`** 时走本工具的 `mdta.io.load_trajectory`
      —— 它会先把 `.trj` 转成带速度的 TRR（缓存在 `_convert_cache/`），
      MDAnalysis 自己读不了 `.trj`；
    · 其它格式（`.pdb/.gro/.tpr/.xtc/.trr/.dcd/…`）走 MDAnalysis。
    """
    import MDAnalysis as mda

    xsd = xsd or DEFAULT_XSD
    trr = trr if trr is not None else DEFAULT_TRR
    select = select or DEFAULT_SELECT
    for p in (xsd, trr):
        if p and not os.path.exists(p):
            raise SystemExit(f"找不到文件：{p}\n"
                             f"可用环境变量 MDTA_MDA_XSD / MDTA_MDA_TRR 指定。")

    suffix = os.path.splitext(str(trr))[1].lower() if trr else ""
    if suffix == ".trj":
        from mdta.io import load_trajectory

        if verbose:
            print(f"轨迹是 Materials Studio .trj：先转 TRR（有缓存直接复用）…")
        u = load_trajectory(xsd, trr).universe
    elif str(xsd).lower().endswith((".xsd", ".xtd")):
        from mdta import ms_structure

        u = ms_structure.build_universe(ms_structure.read_ms_document(xsd))
        if trr:
            u.load_new(trr)
    else:
        u = mda.Universe(xsd, trr) if trr else mda.Universe(xsd)

    try:
        u.atoms.masses
    except Exception:                                    # noqa: BLE001 - 没有质量就补
        fill_masses(u, verbose=verbose)

    ag = u.select_atoms(select) if select else u.atoms
    if verbose:
        print(f"体系：{u.atoms.n_atoms} 原子 / {len(u.trajectory)} 帧；"
              f"选择 {select!r} → {ag.n_atoms} 原子"
              f"（{len(getattr(ag, 'residues', []))} 个残基）")
    return u, ag


def load(select: str | None = None, verbose: bool = True):
    """按默认路径加载 `(u, ag)`（各模块 `__main__` 用它）。"""
    return build_universe(select=select, verbose=verbose)


# ------------------------------------------------------------------ 帧窗口
def max_frames() -> int:
    """最多用几帧（`MDTA_MDA_FRAMES` 没设或 ≤0 = 不限）。"""
    try:
        n = int(os.environ.get("MDTA_MDA_FRAMES", "0") or 0)
    except ValueError:
        n = 0
    return max(0, n)


def frame_indices(u) -> list[int]:
    """要遍历的帧号列表（受 `MDTA_MDA_FRAMES` 限制时按等间隔抽）。"""
    n = len(u.trajectory)
    lim = max_frames()
    if not lim or lim >= n:
        return list(range(n))
    step = max(1, int(np.ceil(n / float(lim))))
    return list(range(0, n, step))


def frame_window(u) -> dict:
    """MDAnalysis 分析类（InterRDF / EinsteinMSD）用的 ``start/stop/step``。"""
    n = len(u.trajectory)
    lim = max_frames()
    if not lim or lim >= n:
        return {"start": 0, "stop": n, "step": 1}
    step = max(1, int(np.ceil(n / float(lim))))
    idx = list(range(0, n, step))
    return {"start": idx[0], "stop": idx[-1] + 1, "step": step}


def iterate(u, frames=None):
    """按 `frames`（等差帧号列表）逐帧产出 ``Timestep``；不给就用 `frame_indices`。"""
    idx = frame_indices(u) if frames is None else [int(f) for f in frames]
    for fi in idx:
        yield u.trajectory[fi]


def frames_window(frames, u=None) -> dict:
    """把等差帧号列表转成 MDAnalysis 分析类要的 ``start/stop/step``。

    ``InterRDF`` / ``EinsteinMSD`` 只能按 start/stop/step 取帧，所以菜单里
    挑出来的帧都保证是等差的（见 ``_menu.choose_frames``）。
    """
    fl = sorted({int(f) for f in (frames or [])})
    if not fl:
        n = len(u.trajectory) if u is not None else 1
        return {"start": 0, "stop": n, "step": 1}
    if len(fl) == 1:
        return {"start": fl[0], "stop": fl[0] + 1, "step": 1}
    step = int(round((fl[-1] - fl[0]) / float(len(fl) - 1))) or 1
    return {"start": fl[0], "stop": fl[-1] + 1, "step": step}


# ------------------------------------------------------------------ 分子编号
def _same_partition(a, b) -> bool:
    """两个编号数组是否描述同一个划分（只差重新编号）。"""
    n_a, n_b = int(np.unique(a).size), int(np.unique(b).size)
    if n_a != n_b:
        return False
    return int(np.unique(np.stack([a, b], axis=1), axis=0).shape[0]) == n_a


def _molnums_unreliable(m, comp) -> bool:
    """``molnums`` 是否真的不可信（与 mdta 引擎 `interface._molnums_unreliable` 同判据）。

    只认两种：① 键块横跨多个 molnum（分子被切碎）；② 一个 molnum 里含 ≥2 个
    "有实质"（≥2 原子）的键块（不同分子被并成一个）。
    单原子块（虚位点 / 离子 / 4 点水的 M 位点）**不算**，所以它仍按 molnums 算一个分子。
    """
    if m.size != comp.size:
        return False
    if int(np.unique(m).size) == int(np.unique(comp).size) and _same_partition(m, comp):
        return False
    order = np.argsort(comp, kind="stable")
    cs, ms = comp[order], m[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], order.size]
    for s, t in zip(starts.tolist(), ends.tolist()):     # ①
        if np.unique(ms[s:t]).size > 1:
            return True
    order = np.argsort(m, kind="stable")
    ms, cs = m[order], comp[order]
    starts = np.flatnonzero(np.r_[True, ms[1:] != ms[:-1]])
    ends = np.r_[starts[1:], order.size]
    for s, t in zip(starts.tolist(), ends.tolist()):     # ②
        _, cnt = np.unique(cs[s:t], return_counts=True)
        if int(np.sum(cnt >= 2)) >= 2:
            return True
    return False


def atom_mol_ids(u) -> np.ndarray | None:
    """**全体系**的原子级分子编号（长度 = `u.atoms.n_atoms`）。

    ⚠️ 一定要在整个体系上算一次、再按原子序号取子集：按组各算一套时两套编号
    互不可比（resid 重号时 A 组的 "molnum 0" 与 B 组的 "molnum 0" 是两个不同分子），
    `inter` 口径会把跨组接触整套误删。
    """
    n = int(u.atoms.n_atoms)
    m = None
    try:
        cand = np.asarray(u.atoms.molnums, dtype=np.int64)
        if cand.size == n:
            m = cand
    except Exception:                                    # noqa: BLE001 - 无该属性
        m = None
    comp = None
    try:
        frags = u.atoms.fragments
        if len(frags) > 1:
            comp = np.empty(n, dtype=np.int64)
            for k, f in enumerate(frags):
                comp[np.asarray(f.indices, dtype=np.int64)] = k
    except Exception:                                    # noqa: BLE001
        comp = None
    if m is None:
        return comp
    if comp is not None and _molnums_unreliable(m, comp):
        m = comp
    neg = m < 0
    if neg.any():                # 未归属残基（-1）各自独立编号，别被当成同一分子
        m = m.copy()
        m[neg] = -(np.arange(m.size, dtype=np.int64)[neg] + 1)
    return m


# ------------------------------------------------------------------ 距离
def min_image_min_distance(pa, pb, box, start=5.0, max_doublings=6) -> float:
    """两组原子之间的**最小镜像**距离；找不到就把截断距离翻倍重搜。

    与工具 `interface.min_distance` 同一算法 —— 所以在"这一帧一对接触都没有"时，
    得到的仍是真实最小间距，而不是 cutoff 本身。
    """
    from MDAnalysis.lib.distances import capped_distance

    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    if pa.size == 0 or pb.size == 0:
        return float("inf")
    c = max(float(start), 0.5)
    for _ in range(int(max_doublings)):
        if box is not None and c > 0.49 * float(np.min(np.asarray(box[:3], dtype=float))):
            break
        try:
            _, d = capped_distance(pa, pb, max_cutoff=float(c), box=box,
                                   return_distances=True)
        except ValueError:
            break
        if d.size:
            return float(np.min(d))
        c *= 2.0
    return float("inf")


def auto_pair(u):
    """按工具的选组规则挑 A / B：A = 原子数最多的组分；B = 次多的**别的**组分。

    只有一个组分时退回同组自接触（A–A），与 `pipeline._contact_pair` 一致。
    """
    names = list(dict.fromkeys([str(r) for r in u.atoms.resnames]))
    if not names:
        return u.atoms, u.atoms, "全部原子", "全部原子"
    sizes = sorted(((n, int((u.atoms.resnames == n).sum())) for n in names),
                   key=lambda kv: -kv[1])
    a_name, a = sizes[0][0], u.select_atoms(f"resname {sizes[0][0]}")
    if len(sizes) < 2:
        return a, a, a_name, a_name
    b_name = sizes[1][0]
    return a, u.select_atoms(f"resname {b_name}"), a_name, b_name
