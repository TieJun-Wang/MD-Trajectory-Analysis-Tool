# -*- coding: utf-8 -*-
"""**分析项可行性预判**：在点「开始分析」之前，先说清楚"这套体系 + 当前选择"能不能跑。

为什么要单独一个模块：以前是"跑起来才知道"——不满足前提的分析项在结果里
标成「跳过」（见 :class:`mdta.pipeline.Analyzer.skips`），用户得等一轮才看见。
这里把同一批前提**提前**算一遍，界面据此把跑不了的项置灰并给出简短原因。

判据原则：**只在确定跑不了时拦**（宁可放过，不可错拦）。所有判断都基于只读信息：
晶胞、帧数、键表、当前勾选的组分与分析对象，不碰任何分析算法。
"""

from __future__ import annotations

import numpy as np

#: 取向/有序度这类需要"链段"的分析，主链至少要有这么多个原子才谈得上方向
_MIN_CHAIN_ATOMS = 8
#: 二面角至少 4 个原子（i, j, k, l）
_MIN_DIHEDRAL_ATOMS = 4
#: **图全是"随时间变化"**的分析项（没有静态面板）：单帧下整项没有可看内容。
#: 逐项核对过实现：comdist（质心距随时间 + 沿 z 的符号距离随时间）、
#: order（各有序度分量随时间）、crystal（晶区占比 / 最大晶簇随时间）、
#: msd（MSD 随时间 / MSD(τ)）。其余分析都至少有一张静态分布图。
_ALL_TIME_PANELS = ("msd", "diffusion", "vacf", "comdist", "order", "crystal")

_OK = {"ok": True, "level": "ok", "reason": ""}


def _blocked(reason: str) -> dict:
    return {"ok": False, "level": "blocked", "reason": reason}


def _warn(reason: str) -> dict:
    return {"ok": True, "level": "warn", "reason": reason}


def _has_box(u) -> bool:
    try:
        d = np.asarray(u.dimensions, dtype=float)
        return d.size >= 3 and bool(np.all(np.isfinite(d[:3]))) and bool(
            np.all(d[:3] > 0))
    except Exception:                                            # noqa: BLE001
        return False


def _has_bonds(u) -> bool:
    try:
        return len(u.bonds) > 0
    except Exception:                                            # noqa: BLE001
        return False


def _chain_ends(u, ag) -> int:
    """``ag`` 里"只连了一根键以内的原子"数量（R_ee 的链端候选）。0 = 环状/交联。"""
    try:
        edges = np.asarray(u.bonds.indices, dtype=np.int64).reshape(-1, 2)
    except Exception:                                            # noqa: BLE001
        return 0
    if not edges.size or ag is None or not ag.n_atoms:
        return 0
    idx = np.asarray(ag.indices, dtype=np.int64)
    deg = np.zeros(int(u.atoms.n_atoms), dtype=np.int64)
    np.add.at(deg, edges.ravel(), 1)
    return int(np.sum(deg[idx] <= 1))


def preflight(az, *, n_frames: int | None = None,
              total_frames: int | None = None) -> dict:
    """返回 ``{分析名: {ok, level, reason}}``；``level`` ∈ ``ok`` / ``warn`` / ``blocked``。

    ``az`` 是 :class:`mdta.pipeline.Analyzer`（只用它的 universe / primary /
    components，不触发任何计算）。

    ``n_frames``：**当前帧选择**（起始/结束/间隔/平衡段/最大帧数）实际选中的帧数。
    传了就用它，不传才退回整个体系的帧数 —— 因为"体系有 100 帧"不等于"这次算
    100 帧"：用户把最大帧数设成 1、或把时间窗收得很窄，MSD 依然没法算
    （实测就是这个：预判说能跑，跑起来报 ``MSD 至少需要 2 帧``）。

    判据**逐条对齐** `Analyzer._run_one` 里真正会 ``skip`` 的条件（那里跳过、
    这里就该置灰），否则就会出现"预判说能跑、跑完却是跳过"的假放行。
    """
    u = az.universe
    box = _has_box(u)
    bonds = _has_bonds(u)
    # 帧数用**加载器自己算出来的**那个（`MDTrajectory.n_frames`），而不是
    # `len(universe.trajectory)` —— 后者在"只给了拓扑、没配轨迹"时可能不真实。
    try:
        total = int(az.trajectory.n_frames)
    except Exception:                                            # noqa: BLE001
        try:
            total = len(u.trajectory)
        except Exception:                                        # noqa: BLE001
            total = 0
    if total_frames is not None:
        total = int(total_frames)
    if n_frames is None:
        n_frames = total
    n_frames = int(n_frames)
    primary = getattr(az, "primary", None)
    n_primary = int(primary.n_atoms) if primary is not None else 0
    try:
        n_comp = len(az.components)
    except Exception:                                            # noqa: BLE001
        n_comp = 0

    out: dict[str, dict] = {}
    no_comp = "还没勾选任何组分（在「分析设置」里勾一个组分或集合）"
    two_comp = "需要至少勾选 2 个组分"
    box_need = "体系没有周期盒（晶胞）信息：这类分析依赖周期性边界"
    no_primary = "还没选「分析对象」（在「分析设置」里选一个组分或集合）"

    # ---- 链构象与取向：都以「分析对象（主链）」为输入
    if n_primary <= 0:
        for name in ("rg", "ree", "dihedral", "orientation", "order"):
            out[name] = _blocked(no_primary)
    else:
        out["rg"] = _OK
        # 端到端距离：没有键就找不到链端；全环/交联也无定义
        if not bonds:
            out["ree"] = _blocked("体系没有键连接信息，无法确定链端")
        elif _chain_ends(u, primary) == 0:
            out["ree"] = _blocked("这组原子没有链端（环状或交联），端到端距离无定义")
        else:
            out["ree"] = _OK
        if n_primary < _MIN_DIHEDRAL_ATOMS:
            out["dihedral"] = _blocked(
                f"分析对象只有 {n_primary} 个原子（二面角至少要 4 个）")
        elif not bonds:
            out["dihedral"] = _blocked("体系没有键连接信息，算不出二面角")
        else:
            out["dihedral"] = _OK
        for name in ("orientation", "order"):
            if not bonds:
                out[name] = _blocked("体系没有键连接信息，无法划分链段")
            elif n_primary < _MIN_CHAIN_ATOMS:
                out[name] = _blocked(
                    f"分析对象只有 {n_primary} 个原子（链段取向至少要 "
                    f"{_MIN_CHAIN_ATOMS} 个）")
            else:
                out[name] = _OK

    # ---- 空间结构：多数要晶胞（PBC）+ 勾选的组分
    if n_comp < 1:
        for name in ("density", "rdf", "rdf2d", "contact"):
            out[name] = _blocked(no_comp)
        for name in ("comdist", "interface"):
            out[name] = _blocked(two_comp)
    else:
        out["density"] = _OK
        out["rdf"] = _OK
        out["rdf2d"] = _OK if box else _blocked(box_need)
        out["contact"] = _OK if bonds else _warn(
            "体系没有键信息：分子内 / 分子间的区分不可靠")
        if n_comp < 2:
            out["comdist"] = _blocked(two_comp)
            out["interface"] = _blocked(two_comp)
        else:
            out["comdist"] = _OK
            out["interface"] = _OK if box else _blocked(box_need)
    out["boo"] = _OK if box else _blocked(box_need)
    out["crystal"] = _OK if box else _blocked(box_need)

    # ---- 动力学：要组分，且至少 2 帧
    if n_comp < 1:
        out["msd"] = _blocked(no_comp.replace("（在", "（MSD 至少需要一个组分；在"))
        out["diffusion"] = _blocked(
            no_comp.replace("（在", "（扩散系数至少需要一个组分；在"))
    elif n_frames < 2:
        # 两种情况要分开说：体系本身只有 1 帧（没配轨迹），还是**帧选择**把它压到 1 帧
        _why = ("当前帧选择只剩 1 帧：检查「帧」里的起始/结束时间、抽帧间隔、最大帧数"
                if total >= 2 else
                f"体系只有 {total} 帧（没配轨迹文件？）：位移无从谈起，至少要 2 帧")
        out["msd"] = _blocked(_why)
        out["diffusion"] = _blocked(_why)
        out["vacf"] = _blocked(_why)
    else:
        out["msd"] = _OK
        out["diffusion"] = _OK
        out["vacf"] = _OK

    # ---- VACF：除了帧数，还要轨迹里**有速度**（XTC/DCD 不存速度）
    try:
        _has_vel = bool(getattr(getattr(az, "trajectory", None),
                                 "has_velocities", False))
    except Exception:                                        # noqa: BLE001
        _has_vel = False
    if not _has_vel and (out.get("vacf") or {}).get("level") == "ok":
        out["vacf"] = _blocked(
            "这条轨迹没有速度数据：VACF 需要带速度的轨迹"
            "（MS 的 .trj 读取时会转成 TRR、自带速度；XTC/DCD 不含速度）")

    # ---- 单帧：其它项能跑，但只是"一张快照"，标 warn 提醒
    if n_frames < 2:
        why = ("当前帧选择只剩 1 帧" if total >= 2
               else f"只有 {total} 帧（只读了结构文件，没配轨迹）")
        # 图**全是时间序列**的项：单帧下没有可看内容，直接拦（不留一堆 1 点的空图）
        for name in _ALL_TIME_PANELS:
            if (out.get(name) or {}).get("level") == "ok":
                out[name] = _blocked(
                    f"{why}：这一项的图全是随时间变化，单帧下没有可看的内容")
        snap = (why + "；随时间变化的面板会自动省略，只出 RDF / 密度 / 各种分布这类"
                       "静态图（与帧数无关，结果有效）")
        for name, v in list(out.items()):
            if v["level"] == "ok":
                out[name] = _warn(snap)

    # 补齐没有单独判定的项（默认放行）
    for name in ("rg", "ree", "dihedral", "density", "rdf", "rdf2d", "comdist",
                 "contact", "interface", "orientation", "order", "boo",
                 "crystal", "msd", "diffusion"):
        out.setdefault(name, _OK)
    return out
