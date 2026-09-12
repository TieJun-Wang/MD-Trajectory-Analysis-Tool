# -*- coding: utf-8 -*-
"""体系信息模块（设计大纲第 6 章）。

读取轨迹后，输出原子数量、模拟时间、帧数、模拟盒尺寸、原子名称、
residue 信息以及分子/链信息。
"""

from __future__ import annotations

import os
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .selection import ChainType, classify_residues, list_chains


@dataclass
class SystemInfo:
    """体系摘要信息。"""

    topology: str = ""
    trajectory: str | None = None
    n_atoms: int = 0
    n_residues: int = 0
    n_segments: int = 0
    n_frames: int = 0
    dt_ps: float = 0.0
    first_time_ps: float = 0.0
    last_time_ps: float = 0.0
    box_dimensions: np.ndarray | None = None
    box_angles: np.ndarray | None = None
    box_type: str = "unknown"
    total_mass_amu: float = 0.0
    total_charge_e: float = 0.0
    n_bonds: int = 0
    n_angles: int = 0
    n_dihedrals: int = 0
    has_bonds: bool = False
    has_charges: bool = False
    has_velocities: bool = False
    atom_names: Counter = field(default_factory=Counter)
    residue_names: Counter = field(default_factory=Counter)
    segment_names: list[str] = field(default_factory=list)
    elements: Counter = field(default_factory=Counter)
    categories: "OrderedDict[str, int]" = field(default_factory=OrderedDict)
    chain_types: list[ChainType] = field(default_factory=list)

    # -------------------------------------------------------------- 尺寸
    @property
    def box_volume(self) -> float:
        """模拟盒体积（Å³）。"""
        if self.box_dimensions is None:
            return float("nan")
        a, b, c = self.box_dimensions[:3]
        if self.box_type == "orthorhombic":
            return float(a * b * c)
        al, be, ga = np.radians(self.box_angles[:3])
        return float(a * b * c * np.sqrt(
            1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
            + 2 * np.cos(al) * np.cos(be) * np.cos(ga)
        ))

    @property
    def total_time_ns(self) -> float:
        return (self.last_time_ps - self.first_time_ps) / 1000.0

    def to_rows(self) -> list[tuple[str, str]]:
        """转成（项目, 数值）列表，供界面/文本展示。"""
        rows: list[tuple[str, str]] = []
        rows.append(("拓扑文件", os.path.basename(self.topology)))
        rows.append(("轨迹文件", os.path.basename(self.trajectory) if self.trajectory else "（无，仅结构）"))
        rows.append(("原子数", f"{self.n_atoms:,}"))
        rows.append(("residue 数", f"{self.n_residues:,}"))
        rows.append(("段(segment)数", f"{self.n_segments:,}"))
        rows.append(("帧数", f"{self.n_frames:,}"))
        rows.append(("时间间隔", f"{self.dt_ps:.4g} ps"))
        rows.append(("模拟时间", f"{self.first_time_ps:.3f} – {self.last_time_ps:.3f} ps"
                                    f"  (共 {self.total_time_ns:.3f} ns)"))
        if self.box_dimensions is not None:
            a, b, c = self.box_dimensions[:3]
            rows.append(("模拟盒", f"{a:.4f} × {b:.4f} × {c:.4f} Å  ({self.box_type})"))
            if self.box_angles is not None:
                rows.append(("盒角度", " ".join(f"{x:.2f}°" for x in self.box_angles[:3])))
            rows.append(("盒体积", f"{self.box_volume:,.1f} Å³"))
        rows.append(("总质量", f"{self.total_mass_amu:,.2f} amu"))
        if self.has_charges:
            rows.append(("体系净电荷", f"{self.total_charge_e:+.4f} e"))
        rows.append(("键/角/二面角", f"{self.n_bonds:,} / {self.n_angles:,} / {self.n_dihedrals:,}"))
        rows.append(("原子名种类", f"{len(self.atom_names)}"))
        rows.append(("residue 名种类", f"{len(self.residue_names)}"))
        for cat, n in self.categories.items():
            rows.append((f"组分 {cat}", f"{n:,} residues"))
        return rows

    def format_text(self, max_chain_types: int = 20) -> str:
        return format_system_info(self, max_chain_types=max_chain_types)


def describe_system(mdt_or_universe) -> SystemInfo:
    """收集体系信息。

    参数可以是 :class:`mdta.io.MDTrajectory`，也可以是 ``MDAnalysis.Universe``。
    """
    if hasattr(mdt_or_universe, "universe"):
        mdt = mdt_or_universe
        u = mdt.universe
        info = SystemInfo(topology=str(mdt.topology), trajectory=mdt.trajectory)
        times = mdt.times_ps
        info.n_frames = mdt.n_frames
        info.dt_ps = mdt.dt_ps
        if times.size:
            info.first_time_ps = float(times[0])
            info.last_time_ps = float(times[-1])
    else:
        u = mdt_or_universe
        info = SystemInfo(topology=str(getattr(u, "filename", "") or ""))

    info.n_atoms = int(u.atoms.n_atoms)
    info.n_residues = int(u.residues.n_residues)
    info.n_segments = int(u.segments.n_segments)

    dims = u.dimensions
    if dims is not None:
        dims = np.asarray(dims, dtype=float)
        info.box_dimensions = dims[:3].copy()
        info.box_angles = dims[3:6].copy()
        info.box_type = ("orthorhombic"
                         if np.allclose(dims[3:6], 90.0, atol=1e-3)
                         else "triclinic")

    try:
        info.total_mass_amu = float(np.asarray(u.atoms.masses, dtype=float).sum())
    except Exception:  # noqa: BLE001
        pass
    try:
        q = np.asarray(u.atoms.charges, dtype=float)
        info.has_charges = q.size == u.atoms.n_atoms
        if info.has_charges:
            info.total_charge_e = float(q.sum())
    except Exception:  # noqa: BLE001
        info.has_charges = False

    info.atom_names = Counter(np.asarray(u.atoms.names, dtype=str).tolist())
    info.residue_names = Counter(np.asarray(u.residues.resnames, dtype=str).tolist())
    info.segment_names = [str(s) for s in np.asarray(u.segments.segids, dtype=str)]

    try:
        el = np.asarray(u.atoms.elements, dtype=str)
        info.elements = Counter(x for x in el.tolist() if x)
    except Exception:  # noqa: BLE001
        pass

    try:
        info.n_bonds = len(u.bonds) if u.bonds is not None else 0
        info.has_bonds = info.n_bonds > 0
    except Exception:  # noqa: BLE001
        info.n_bonds = 0
    for attr, key in (("angles", "n_angles"), ("dihedrals", "n_dihedrals")):
        try:
            top = getattr(u, attr, None)
            setattr(info, key, len(top) if top is not None else 0)
        except Exception:  # noqa: BLE001
            setattr(info, key, 0)

    try:
        info.has_velocities = bool(u.trajectory.ts.has_velocities)
    except Exception:  # noqa: BLE001
        info.has_velocities = False

    cats = classify_residues(u)
    info.categories = OrderedDict((k, int(len(v))) for k, v in cats.items())

    try:
        info.chain_types = list_chains(u)
    except Exception:  # noqa: BLE001
        info.chain_types = []

    return info


def format_system_info(info: SystemInfo, max_chain_types: int = 20) -> str:
    """把体系信息格式化成可打印的中文报告。"""
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("MD 轨迹分析工具 —— 体系信息")
    lines.append("=" * 68)
    for k, v in info.to_rows():
        lines.append(f"  {k:<16}: {v}")

    if info.atom_names:
        top = info.atom_names.most_common(12)
        lines.append(f"  {'常见原子名':<16}: " + ", ".join(f"{n}({c})" for n, c in top))
    if info.elements:
        top = info.elements.most_common(10)
        lines.append(f"  {'元素组成':<16}: " + ", ".join(f"{n}({c})" for n, c in top))
    if info.residue_names:
        top = info.residue_names.most_common(12)
        lines.append(f"  {'常见 residue':<16}: " + ", ".join(f"{n}({c})" for n, c in top))

    if info.chain_types:
        lines.append("-" * 68)
        lines.append("  分子/链信息（按组成归类）")
        ordered = sorted(info.chain_types, key=lambda c: (-c.n_atoms, c.label))
        for c in ordered[:max_chain_types]:
            lines.append(f"    - {c}")
        if len(ordered) > max_chain_types:
            lines.append(f"    ... 另有 {len(ordered) - max_chain_types} 类未显示")
    lines.append("=" * 68)
    return "\n".join(lines)


def info_tables(info: SystemInfo, max_chain_types: int = 20) -> list[dict]:
    """把体系信息整理成**结构化表格**，供 Web 界面直接渲染。

    返回 ``[{"title": 小节名, "columns": [...], "rows": [[...], ...]}, ...]``。
    所有小节都用同一种形状（列 + 行），前端就能用同一个表格组件渲染，
    不必去解析 ``format_system_info`` 那种定宽文本报告 —— 定宽文本在窄栏里
    要么被截断、要么被迫左右滚动，体验很差。
    """
    sections: list[dict] = []

    def kv(title: str, pairs: list[tuple[str, str]]) -> None:
        sections.append({
            "title": title,
            "columns": ["项目", "值"],
            "rows": [[k, v] for k, v in pairs],
        })

    # ---------------------------------------------------------- 文件与规模
    basic = [
        ("拓扑文件", os.path.basename(info.topology) or "—"),
        ("轨迹文件", os.path.basename(info.trajectory) if info.trajectory
         else "（无，仅结构）"),
        ("原子数", f"{info.n_atoms:,}"),
        ("residue 数", f"{info.n_residues:,}"),
        ("段 (segment) 数", f"{info.n_segments:,}"),
    ]
    kv("文件与规模", basic)

    # -------------------------------------------------------------- 时间轴
    kv("时间轴", [
        ("帧数", f"{info.n_frames:,}"),
        ("时间间隔", f"{info.dt_ps:.4g} ps"),
        ("首帧时刻", f"{info.first_time_ps:.3f} ps"),
        ("末帧时刻", f"{info.last_time_ps:.3f} ps"),
        ("总模拟时长", f"{info.total_time_ns:.3f} ns"),
    ])

    # -------------------------------------------------------------- 模拟盒
    box: list[tuple[str, str]] = []
    if info.box_dimensions is not None:
        a, b, c = (list(info.box_dimensions) + [0, 0, 0])[:3]
        box.append(("盒矢量长度", f"{a:.4f} × {b:.4f} × {c:.4f} Å"))
        box.append(("盒类型", {"triclinic": "三斜", "orthorhombic": "正交"}
                    .get(info.box_type, info.box_type)))
        if info.box_angles is not None:
            ang = (list(info.box_angles) + [0, 0, 0])[:3]
            box.append(("盒角度", " / ".join(f"{x:.2f}°" for x in ang)))
        box.append(("盒体积", f"{info.box_volume:,.1f} Å³"))
    box.append(("总质量", f"{info.total_mass_amu:,.2f} amu"))
    if info.has_charges:
        box.append(("体系净电荷", f"{info.total_charge_e:+.4f} e"))
    kv("模拟盒与质量", box)

    # -------------------------------------------------------- 拓扑与属性
    kv("拓扑与属性", [
        ("键 / 角 / 二面角", f"{info.n_bonds:,} / {info.n_angles:,} / {info.n_dihedrals:,}"),
        ("含化学键", "是" if info.has_bonds else "否（依赖键的分析会跳过或自动猜键）"),
        ("含电荷", "是" if info.has_charges else "否"),
        ("含速度", "是" if info.has_velocities else "否"),
        ("原子名种类", f"{len(info.atom_names):,}"),
        ("residue 名种类", f"{len(info.residue_names):,}"),
        ("段名", " / ".join(info.segment_names) if info.segment_names else "—"),
    ])

    # ------------------------------------------------------------ 组分构成
    if info.categories:
        n_res_total = max(int(info.n_residues), 1)
        sections.append({
            "title": "组分构成", "columns": ["组分", "residue 数", "占比"],
            "rows": [[str(cat), f"{int(n):,}",
                      f"{100.0 * int(n) / n_res_total:.1f}%"]
                     for cat, n in info.categories.items()],
            "note": "组分按 residue 名与元素自动识别（protein / nucleic / sugar / "
                    "polymer / water / ion / other）。",
        })

    # ------------------------------------------------- 计数器类（元素/名字）
    def counter_rows(counter, limit: int) -> list[list[str]]:
        top = counter.most_common(limit)
        tot = sum(int(v) for v in counter.values()) or 1
        return [[str(k), f"{int(v):,}", f"{100.0 * int(v) / tot:.2f}%"] for k, v in top]

    if info.elements:
        sections.append({
            "title": "元素组成", "columns": ["元素", "原子数", "占比"],
            "rows": counter_rows(info.elements, 12),
            "note": f"共 {len(info.elements)} 种元素，按原子数降序显示前 12 种。",
        })
    if info.residue_names:
        sections.append({
            "title": "常见 residue 名", "columns": ["residue", "数量", "占比"],
            "rows": counter_rows(info.residue_names, 12),
            "note": f"共 {len(info.residue_names)} 种 residue 名，按数量降序显示前 12 种。",
        })
    if info.atom_names:
        sections.append({
            "title": "常见原子名", "columns": ["原子名", "数量", "占比"],
            "rows": counter_rows(info.atom_names, 12),
            "note": f"共 {len(info.atom_names)} 种原子名，按数量降序显示前 12 种。",
        })

    # ---------------------------------------------------------- 分子/链信息
    if info.chain_types:
        ordered = sorted(info.chain_types, key=lambda c: (-c.n_atoms, c.label))
        rows = []
        for c in ordered[:max_chain_types]:
            # n_atoms / n_residues 是**每条链**的取值（取该类最主要的尺寸），
            # 不是该类的合计，所以列名必须写清楚"每条"。
            if len(c.variants) <= 1:
                dist = "—"
            else:
                dist = " / ".join(
                    f"{cnt} 条×{na} 原子" for (nr, na), cnt in
                    sorted(c.variants.items(), key=lambda kv: -kv[0][1])
                )
            rows.append([c.label, f"{int(c.count):,}", f"{int(c.n_atoms):,}",
                         f"{int(c.n_residues):,}", dist])
        note = ("「每条原子数 / residue 数」是单条链的规模（取该类里最主要的"
                "尺寸），不是合计。")
        if len(ordered) > max_chain_types:
            note += f" 另有 {len(ordered) - max_chain_types} 类未显示（按原子数降序）。"
        sections.append({
            "title": "分子 / 链信息（按组成归类）",
            "columns": ["分组", "条数", "每条原子数", "每条 residue 数", "尺寸分布"],
            "rows": rows, "note": note,
        })

    return sections
