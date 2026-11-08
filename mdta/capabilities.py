# -*- coding: utf-8 -*-
"""**数据能力检测引擎**：不靠文件后缀，直接看 Universe 里到底有什么。

为什么要有它（用户要求）
------------------------
以前"这个分析能不能跑"是各处零散判断，还容易退化成"看后缀猜"。
这里统一回答 7 个问题，所有上层（可行性预判、界面提示、静态/动态模式）都读这一份：

    Coordinates       有没有原子坐标
    Trajectory        有没有**多帧**轨迹（>1 帧）
    Bonds             有没有可靠键连接（数量 + 键长尺度是否合理）
    Masses            有没有原子质量
    Charges           有没有原子电荷
    Molecule Groups   能不能把原子归到分子（molnums / 键连通分量）
    Box Dimensions    有没有有效的周期性盒子

并在最后给出 ``mode``：``static``（只能做单帧结构分析）/ ``dynamic``（可做时间演化）。
所有检查都是**只读**的，不触发任何分析计算。
"""

from __future__ import annotations

import numpy as np

#: 一条能力的统一形态：``{ok, label, detail, level}``，level ∈ ok / warn / bad
_LEVELS = ("ok", "warn", "bad")


def _entry(ok: bool, label: str, detail: str = "", level: str = "") -> dict:
    lv = level or ("ok" if ok else "bad")
    if lv not in _LEVELS:
        lv = "warn"
    return {"ok": bool(ok), "label": label, "detail": detail, "level": lv}


def detect(az) -> dict:
    """检测当前体系的数据能力，返回 ``{items: {...}, mode, warnings, counts}``。"""
    u = az.universe
    out: dict = {}
    warnings: list[str] = []

    # ---------------------------------------------------------- Coordinates
    n_atoms = int(u.atoms.n_atoms)
    try:
        pos = np.asarray(u.atoms.positions, dtype=float)
        finite = bool(pos.size and np.all(np.isfinite(pos)))
        span = float(np.max(pos.max(axis=0) - pos.min(axis=0))) if finite else 0.0
    except Exception:                                            # noqa: BLE001
        finite, span = False, 0.0
    out["coordinates"] = _entry(
        finite, "原子坐标",
        f"{n_atoms:,} 个原子，包围盒最大边长 {span:.1f} Å" if finite
        else "读不到有限坐标（全 NaN/inf 或没有坐标）")
    if not finite:
        warnings.append("坐标不可用：几乎所有分析都依赖坐标，请检查文件是否读对了")

    # ----------------------------------------------------------- Trajectory
    try:
        n_frames = int(az.trajectory.n_frames)
    except Exception:                                            # noqa: BLE001
        try:
            n_frames = len(u.trajectory)
        except Exception:                                        # noqa: BLE001
            n_frames = 0
    dt = None
    try:
        dt = float(az.trajectory.dt_ps) if n_frames > 1 else None
    except Exception:                                            # noqa: BLE001
        dt = None
    dyn = n_frames > 1
    out["trajectory"] = _entry(
        dyn, "多帧轨迹",
        (f"{n_frames:,} 帧" + (f"，帧间隔 {dt:g} ps" if dt else "")) if dyn
        else "只有 1 帧（只读了结构文件）：随时间变化的分析不可用",
        level="ok" if dyn else "warn")
    if not dyn:
        warnings.append("只有单帧结构：静态分析（结构 / 空间分布 / 构象统计）可用，"
                        "动力学分析（MSD、扩散系数）不可用")

    # ---------------------------------------------------------------- Bonds
    try:
        n_bonds = int(len(u.bonds))
    except Exception:                                            # noqa: BLE001
        n_bonds = 0
    bond_note = ""
    bond_ok = n_bonds > 0
    if bond_ok:
        try:
            from .topology import bond_scale_report

            rep = bond_scale_report(u.atoms)
            if rep.get("n_bonds"):
                ratio = rep.get("ratio")
                if ratio is not None and ratio < 0.6:
                    bond_ok = False
                    bond_note = (f"键长中位数 {rep['median']:.2f} Å 明显偏离共价半径尺度"
                                 f"（比值 {ratio:.2f}），键表可能不可靠")
                else:
                    bond_note = f"{n_bonds:,} 根键，键长中位数 {rep['median']:.2f} Å"
        except Exception:                                        # noqa: BLE001
            bond_note = f"{n_bonds:,} 根键"
    out["bonds"] = _entry(bond_ok, "键连接",
                          bond_note or "没有键表：链端 / 二面角 / 按分子划分都不可靠",
                          level=("ok" if bond_ok else "bad"))
    if not bond_ok:
        warnings.append("键连接缺失或不可靠：R_ee / 二面角 / 链段取向 / 分子归属会退化")

    # --------------------------------------------------------------- Masses
    try:
        m = np.asarray(u.atoms.masses, dtype=float)
        ok_m = bool(m.size == n_atoms and np.all(np.isfinite(m)) and np.all(m > 0))
        bad_m = int(np.sum(~(m > 0))) if m.size == n_atoms else n_atoms
    except Exception:                                            # noqa: BLE001
        ok_m, bad_m = False, n_atoms
    out["masses"] = _entry(
        ok_m, "原子质量",
        "全部原子的质量均 > 0" if ok_m else f"{bad_m:,} 个原子缺质量（质量加权量会退化）",
        level="ok" if ok_m else "warn")

    # -------------------------------------------------------------- Charges
    try:
        q = np.asarray(u.atoms.charges, dtype=float)
        has_q = q.size == n_atoms
        nz = int(np.count_nonzero(q)) if has_q else 0
        ok_q = has_q and nz > 0
    except Exception:                                            # noqa: BLE001
        ok_q, nz = False, 0
    out["charges"] = _entry(
        ok_q, "原子电荷",
        (f"有电荷，{nz:,} 个非零" if ok_q else
         "没有电荷（本工具的分析不依赖电荷，只是标记出来）"),
        level="ok" if ok_q else "warn")

    # ------------------------------------------------------ Molecule Groups
    try:
        from .topology import molecule_ids

        mol = molecule_ids(u.atoms)
        n_mol = int(np.unique(mol).size) if mol is not None else 0
    except Exception:                                            # noqa: BLE001
        n_mol = 0
    ok_mol = n_mol > 1
    out["molecules"] = _entry(
        ok_mol, "分子归属",
        (f"{n_mol:,} 个分子（来自分子编号或键连通分量）" if ok_mol
         else "识别不出多个分子：按分子累计的分析（逐分子 Rg、质心 MSD）不可用"),
        level="ok" if ok_mol else "warn")

    # ------------------------------------------------------ Box Dimensions
    try:
        d = np.asarray(u.dimensions, dtype=float)
        ok_box = bool(d.size >= 3 and np.all(np.isfinite(d[:3])) and np.all(d[:3] > 0))
        box_txt = ("×".join(f"{float(x):.2f}" for x in d[:3]) + " Å") if ok_box else ""
    except Exception:                                            # noqa: BLE001
        ok_box, box_txt = False, ""
    out["box"] = _entry(
        ok_box, "周期盒",
        (f"{box_txt}" + (f"（αβγ {d[3]:.0f}/{d[4]:.0f}/{d[5]:.0f}°）"
                         if ok_box and d.size >= 6 else "")) if ok_box
        else "没有有效晶胞：密度分布 / 面内 RDF / 界面 / BOO / 晶区识别不可用",
        level="ok" if ok_box else "warn")
    if not ok_box:
        warnings.append("没有周期性盒子：依赖 PBC 的分析（密度、面内 RDF、界面、BOO、晶区）不可用")

    mode = "dynamic" if dyn else "static"
    return {
        "mode": mode,
        "mode_label": "动态分析（拓扑 + 轨迹）" if dyn else "静态分析（单帧结构）",
        "items": out,
        "warnings": warnings,
        "counts": {"n_atoms": n_atoms, "n_frames": n_frames,
                   "n_bonds": n_bonds, "n_molecules": n_mol},
        # 一页能看完的紧凑清单（界面直接用）
        "summary": " · ".join(
            f"{v['label']}{'✓' if v['ok'] else '✗'}" for v in out.values()),
    }
