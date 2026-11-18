# -*- coding: utf-8 -*-
"""链段取向（allfast-3-2.py 口径）—— 键中点矢量法。

这份脚本把外部工具 ``allfast-3-2.py`` 的计算逻辑**原样搬到 MDAnalysis 上**，
方便与工具「链段取向分析」和 Forcite 三方对拍。

一、它的链段是怎么定义的（与工具的区别所在）
-------------------------------------------
对一条链的**碳骨架坐标** ``C_0 … C_{n-1}``：

1. 先取相邻碳的**键中点** ``m_i = (C_i + C_{i+1}) / 2``（n−1 个）；
2. 链段矢量 = **相邻两个键中点连成的矢量** ``v_i = m_{i+1} − m_i = (C_{i+2} − C_i)/2``
   （n−2 个）。

也就是说：它跨过**一个碳原子**取方向（锯齿烷烃里这样正好把 C–C–C 的折角抹平，
得到的才是"链的走向"）。等价的工具写法是 **几何骨架 + stride=2**：

    analyze_orientation(..., mode="backbone", stride=2, p2_axis="z")

二、它算的三个量（本脚本逐个实现）

=====================  ==========================================================
函数                    含义
=====================  ==========================================================
`cos2_to_axis()`        ``⟨cos²θ_z⟩``：链段矢量与参考轴的夹角余弦平方，**链内平均**
`straightness()`        "取向参数"：**相邻链段矢量之间**夹角的 ``cos²`` 平均
                        （≈ 链的局部直度：1 = 局部笔直，越低越弯折）
`mean_angle_to_axis()`  ``⟨θ⟩``（度）：链段与参考轴的夹角（工具 1.0.6 新增的曲线）
=====================  ==========================================================

三、两种统计模式（与脚本一致）

- **模式 A（单链）**：跟踪指定的那一条链：``chain_index`` 给第几条；
- **模式 B（全链平均）**：每帧对该帧所有链求平均（默认）。

四、与工具的等价关系

    P₂(工具, backbone+stride=2) = (3 · ⟨cos²θ⟩ − 1) / 2      ← 脚本的 cos2_to_axis

⚠️ **权重口径（多分散体系才会显现，310K 那个 C6–C50 混油就是这种）**

程序的「链段取向分析」现在有 **统计权重** 参数（1.0.7 起）：

- ``segment``（**默认**）**链段加权**：所有链段的矢量放进一个池子平均，
  长链链段多、权重大 —— ``P₂ = ⟨(3cos²−1)/2⟩`` over **全部链段**；
- ``molecule`` **每条链等权**：先按分子求平均，再在分子之间等权平均。

本脚本**两个权重都算、都导出**：``cos2_pooled`` / ``P2_pooled`` 对上默认的
链段加权，``cos2_axis`` / ``P2`` 对上「每条链等权」。两者在**单分散**体系
（122×C12、20×PE30）里完全等价，链长不一就会差：实测 310K
（C6 25 条 … C50 3 条）逐链等权 P₂ = +0.16，链段加权 P₂ = −0.007。

⚠️ **分子集合**：``chain_coordinates`` 默认只保留**线性链**（成键数 = 重原子数−1
且无 3 度以上节点），这样石墨/二维网络、环状分子不会被当成"链"；链长**不做**
"众数过滤"（310K 那种多分散体系里按众数只会挑出 25 条 C6，其余 71 条全丢）。
要按长度筛，显式给 ``n_atoms=<整数>`` 或 ``n_atoms="mode"``。

单独跑：``python test/mda_test/orientation.py``
"""

from __future__ import annotations

import os

import numpy as np

try:
    from ._common import frame_indices, save_csv
except ImportError:
    from _common import frame_indices, save_csv

_AXES = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}


# ------------------------------------------------------------------ 链段几何
def bond_midpoints(xyz) -> np.ndarray:
    """相邻原子的**键中点** ``m_i = (r_i + r_{i+1})/2``，形状 ``(n-1, 3)``。"""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape[0] < 2:
        return np.zeros((0, 3))
    return 0.5 * (xyz[:-1] + xyz[1:])


def midpoint_vectors(xyz) -> np.ndarray:
    """链段矢量 = 相邻键中点的差 ``v_i = m_{i+1} − m_i = (r_{i+2} − r_i)/2``。

    形状 ``(n-2, 3)``；``n < 3`` 时为空（这一条链给不出链段方向）。
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape[0] < 3:
        return np.zeros((0, 3))
    return xyz[2:] - xyz[:-2]          # 等价于 (m_{i+1} − m_i)，省一次中点运算


def _unit(v) -> np.ndarray:
    n = np.linalg.norm(v, axis=1)
    good = n > 1e-12
    return v[good] / n[good, None], good


def cos2_to_axis(xyz, axis="z") -> float:
    """链段与参考轴夹角的 ``cos²`` 在**这一条链内**的平均（脚本的 ``⟨cos²θ_z⟩``）。

    与 ``allfast-3-2.py::compute_molecular_cos2_z`` 同一口径：逐链段算
    ``v_ref² / |v|²`` 再取算术平均。
    """
    ref = _AXES.get(str(axis).lower(), _AXES["z"])
    v = np.asarray(midpoint_vectors(xyz), dtype=float)
    if v.shape[0] == 0:
        return float("nan")
    n2 = (v * v).sum(axis=1)
    good = n2 > 1e-12
    if not good.any():
        return float("nan")
    return float(np.mean((v[good] @ ref) ** 2 / n2[good]))


def straightness(xyz) -> float:
    """脚本的「取向参数」：**相邻**链段矢量之间夹角的 ``cos²`` 平均。

    1.0 = 局部完全笔直（相邻链段同向）；越低说明链在弯折。
    与 ``allfast-3-2.py::compute_orientation_parameter`` 同一口径。
    """
    v = np.asarray(midpoint_vectors(xyz), dtype=float)
    if v.shape[0] < 2:
        return float("nan")
    u, _ = _unit(v)
    if u.shape[0] < 2:
        return float("nan")
    cos = np.clip((u[:-1] * u[1:]).sum(axis=1), -1.0, 1.0)
    return float(np.mean(cos ** 2))


def mean_angle_to_axis(xyz, axis="z") -> float:
    """链段与参考轴的**平均夹角**（度，0–90°）。

    链段没有方向性（v 与 −v 等价），所以先取 ``|cos|`` 再 ``arccos``；
    完全无序时理论值 **54.74°**——与工具 1.0.6 新增的「链段与参考轴的夹角
    随时间变化」面板同一口径。
    """
    ref = _AXES.get(str(axis).lower(), _AXES["z"])
    v = np.asarray(midpoint_vectors(xyz), dtype=float)
    if v.shape[0] == 0:
        return float("nan")
    u, _ = _unit(v)
    if u.shape[0] == 0:
        return float("nan")
    cos = np.clip(u @ ref, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(np.abs(cos)))))


# ------------------------------------------------------------------ 链的取法
def _fragment_degrees(mol):
    """碎片内部的**原子**度数（每个原子连了几个碎片内的原子）。

    返回 ``(deg_heavy, heavy_mask)``：``deg_heavy`` 只数**重原子之间**的键
    （C–H 不改变"是不是一条链"），``heavy_mask`` 标出重原子。
    没有键表时返回 ``None``（此时无法判断线性/网络，按"线性"处理）。
    """
    try:
        bonds = mol.bonds
    except Exception:                                   # noqa: BLE001 - 没有键表
        return None
    if bonds is None or len(bonds) == 0:
        return None
    try:
        idx = np.asarray(bonds.indices, dtype=np.int64)   # 全体系原子下标
    except Exception:                                   # noqa: BLE001
        return None
    if idx.size == 0:
        return None
    # 全体系下标 → 碎片内局部下标
    order = np.asarray(mol.indices, dtype=np.int64)
    pos = np.searchsorted(order, idx)
    pos_clipped = np.clip(pos, 0, max(order.size - 1, 0))
    hit = order[pos_clipped] == idx
    if not hit.all():
        idx = idx[hit]
        pos_clipped = pos_clipped[hit]
    try:
        elem = np.asarray([str(e) for e in mol.elements])
    except Exception:                                   # noqa: BLE001
        elem = np.asarray([str(n) for n in mol.names])
    heavy = ~np.char.startswith(elem, "H")
    is_heavy = heavy[pos_clipped].all(axis=1)
    deg_heavy = np.zeros(int(mol.n_atoms), dtype=int)
    if is_heavy.any():
        np.add.at(deg_heavy, pos_clipped[is_heavy].ravel(), 1)
    return deg_heavy, heavy


def _is_linear_chain(mol) -> bool:
    """是不是一条**线性链**：重原子子图无 3 度节点、且成键数 = 节点数 − 1。

    · 直链烷烃 ✓；环状分子（成环 → 键数 ≥ 节点数）✗；石墨/二维网络（3 度）✗；
    · 支化（PP 的支点 3 度）✗ —— 这类分子的"链段"不是一条路，取向含义不同。
    """
    got = _fragment_degrees(mol)
    if got is None:
        return True                                     # 没键表：无从判断，放行
    deg, heavy = got
    n_heavy = int(heavy.sum())
    if n_heavy < 3:
        return False
    if int(np.max(deg)) > 2:
        return False
    n_bonds = int(np.sum(deg)) // 2
    return n_bonds == n_heavy - 1


def chain_coordinates(ag, heavy_only=True, n_atoms="auto", require_linear=True,
                      unwrap=False, verbose=False):
    """把原子组按**分子**（键连通分量）拆成一条条**线性链**，返回坐标列表。

    ``heavy_only=True``（默认）只取重原子：脚本里算的是**碳骨架**，氢会把
    键中点算乱。判据是元素/原子名不以 H 开头。

    ``require_linear=True``（默认）丢掉支化/环状/二维网络碎片（石墨片、苯环…）：
    这些碎片不是"链"，它们的"链段"混进来会把 P₂ 拉偏。

    ``unwrap=True`` 先用 ``make_whole`` 把每条链接成整链（跨周期边界的分子
    不会被算成一条"横穿盒子的链段"）。工具引擎内部用的是解包坐标，
    要与它逐值对照就设 ``unwrap=True``。

    ``n_atoms``（链长=重原子数）：

      · ``"auto"``（默认）：**不按长度过滤**，保留全部线性链。多分散体系
        （例如 C6–C50 混油）必须这样，否则只会算到一种链长；
      · ``"mode"``：只保留**众数**长度（旧 ``auto`` 的行为：模拟原脚本写死
        "12 个碳"的口径，适合单分散体系里剔除杂质）；
      · 给整数：只保留该长度；
      · ``None``：同 ``"auto"``（不过滤）。
    """
    out = []
    dropped = []
    try:
        mols = list(ag.fragments)
    except Exception:                                   # noqa: BLE001 - 没有键表
        mols = [ag]
    for mol in mols:
        if require_linear and not _is_linear_chain(mol):
            dropped.append(int(mol.n_atoms))
            continue
        if unwrap:
            try:
                from MDAnalysis.lib.mdamath import make_whole

                make_whole(mol)
            except Exception:                           # noqa: BLE001
                pass
        pos = np.asarray(mol.positions, dtype=float)
        if heavy_only:
            try:
                keep = np.asarray([not str(e).startswith("H") for e in mol.elements])
            except Exception:                           # noqa: BLE001 - 无元素
                keep = np.asarray([not str(n).startswith("H") for n in mol.names])
            if keep.any():
                pos = pos[keep]
        if pos.shape[0] >= 3:
            out.append(pos)
    if verbose and dropped:
        print(f"（跳过 {len(dropped)} 个非线性的碎片：石墨/环/支化，共 "
              f"{sum(dropped)} 原子）")
    if not out:
        return out
    counts = np.asarray([c.shape[0] for c in out])
    if n_atoms is None or str(n_atoms).lower() == "auto":
        return out
    if str(n_atoms).lower() == "mode":
        target = int(np.bincount(counts).argmax())
    else:
        target = int(n_atoms)
    return [c for c in out if c.shape[0] == target]


def chain_size_histogram(chains) -> dict:
    """链长（重原子数）→ 条数，用来发现"多分散"体系。"""
    if not chains:
        return {}
    counts = np.asarray([c.shape[0] for c in chains])
    uniq, freq = np.unique(counts, return_counts=True)
    return {int(u): int(f) for u, f in zip(uniq, freq)}


def pooled_cos2(chains, axis="z") -> float:
    """**链段加权**的 ``⟨cos²θ⟩``（把所有链的所有链段放进一个池子）。

    这是工具引擎的口径（``analyze_orientation`` 里对所有链段取平均），
    单分散时与逐链等权相同，多分散时不同（长链链段多、权重更大）。
    """
    ref = _AXES.get(str(axis).lower(), _AXES["z"])
    v = np.concatenate([midpoint_vectors(c) for c in chains]) if chains else None
    if v is None or v.shape[0] == 0:
        return float("nan")
    n2 = (v * v).sum(axis=1)
    good = n2 > 1e-12
    if not good.any():
        return float("nan")
    return float(np.mean((v[good] @ ref) ** 2 / n2[good]))


def pooled_angle(chains, axis="z") -> float:
    """**链段加权**的 ``⟨arccos|cosθ|⟩``（度），引擎 θ(t) 面板的口径。"""
    ref = _AXES.get(str(axis).lower(), _AXES["z"])
    v = np.concatenate([midpoint_vectors(c) for c in chains]) if chains else None
    if v is None or v.shape[0] == 0:
        return float("nan")
    u, _ = _unit(v)
    if u.shape[0] == 0:
        return float("nan")
    return float(np.degrees(np.mean(np.arccos(np.abs(np.clip(u @ ref, -1, 1))))))


def pooled_straightness(chains) -> float:
    """**链段加权**的"相邻链段夹角 cos²"平均（逐链先算再按链段数加权）。"""
    vals, w = [], []
    for c in chains:
        v = midpoint_vectors(c)
        if v.shape[0] < 2:
            continue
        u, _ = _unit(v)
        if u.shape[0] < 2:
            continue
        vals.append(np.mean(np.clip((u[:-1] * u[1:]).sum(axis=1), -1, 1) ** 2))
        w.append(v.shape[0] - 1)
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=w))


def _stack_stats(fn, chains, *args):
    """对每条链调用 ``fn``，返回 (逐链数组, nan 安全平均)。"""
    vals = np.asarray([fn(c, *args) for c in chains], dtype=float)
    fin = vals[np.isfinite(vals)]
    return vals, (float(np.mean(fin)) if fin.size else float("nan"))


def _note(cos2: float) -> str:
    """脚本里的定性分类（阈值照抄 allfast-3-2.py）。"""
    if not np.isfinite(cos2):
        return "无链段"
    return "高度有序" if cos2 > 0.7 else ("部分倾斜" if cos2 > 0.4 else "近似随机")


# ------------------------------------------------------------------ 三个模块
def test_cos2_axis(u, ag, axis="z", frames=None, prefix="ori_cos2"):
    """``⟨cos²θ⟩`` 与由它换算的 **P2** 随时间变化 —— 脚本的模式 B（全链平均）。

    ⚠️ **这里输出的才是 P2**（二阶序参数，相对参考轴）：

        P2(t) = (3·⟨cos²θ⟩ − 1) / 2      ← 与工具「二阶序参数 P₂」同一口径

    它跟 :func:`test_orientation_parameter` 的「取向参数」**不是一回事**：
    后者是"**相邻链段矢量之间**夹角的 cos² 平均"（局部直度，不需要参考轴）。

    导出的两种权重（单分散体系里两者相同）：

    - ``cos2_axis`` / ``P2``：**逐链等权**（每条链先算 ⟨cos²⟩ 再平均，原脚本口径）；
    - ``cos2_pooled`` / ``P2_pooled``：**链段加权**（所有链段进一个池子），
      **与工具引擎逐值一致**，多分散体系请用这两个。

    导出 ``{prefix}.csv``：
    ``frame, time(ps), cos2_axis, P2, n_chains, cos2_pooled, P2_pooled``
    """
    chains0 = chain_coordinates(ag, unwrap=True, verbose=True)
    if not chains0:
        print("⚠️ 该原子组里没有 ≥3 个重原子的线性链，跳过")
        return None
    hist = chain_size_histogram(chains0)
    print(f"链数 {len(chains0)}，重原子数分布 {hist}；参考轴 {str(axis).upper()}"
          + ("（**多分散**：逐链等权与链段加权会不同，取 P2_pooled 对引擎）"
             if len(hist) > 1 else ""))
    n_atom = int(round(float(np.mean([c.shape[0] for c in chains0]))))
    if n_atom < 4:
        print("⚠️ 每链重原子太少（<4），链段只有 1–2 个，取向参数没有统计意义")

    rows = []
    for fi in (frame_indices(u) if frames is None else [int(f) for f in frames]):
        ts = u.trajectory[fi]
        chains = chain_coordinates(ag, unwrap=True, verbose=True)
        _, cos2 = _stack_stats(cos2_to_axis, chains, axis)
        p2 = (3.0 * cos2 - 1.0) / 2.0
        cos2p = pooled_cos2(chains, axis)
        p2p = (3.0 * cos2p - 1.0) / 2.0
        rows.append((fi, float(ts.time), cos2, p2, len(chains), cos2p, p2p))
        print(f"  帧 {fi}: <cos²θ> = {cos2:.6f} (P2 = {p2:+.6f})"
              f"  链段加权 <cos²θ> = {cos2p:.6f} (P2 = {p2p:+.6f})"
              f"  {_note(cos2)}")

    save_csv(f"{prefix}.csv",
             [[r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
              [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
              [r[6] for r in rows]],
             header=("frame,time(ps),cos2_axis,P2,n_chains,cos2_pooled,P2_pooled  # "
                     "cos2_axis=<cos^2(theta)> 逐链等权; P2=(3*cos2-1)/2; "
                     "cos2_pooled=链段加权（与工具引擎同口径）"),
             fmt="%d,%.4f,%.6f,%.6f,%d,%.6f,%.6f")
    fin = np.asarray([r[2] for r in rows], dtype=float)
    finp = np.asarray([r[5] for r in rows], dtype=float)
    p2_mean = (3.0 * np.nanmean(fin) - 1.0) / 2.0
    p2p_mean = (3.0 * np.nanmean(finp) - 1.0) / 2.0
    print(f"全轨迹 <cos²θ> = {np.nanmean(fin):.6f} → **P2 = {p2_mean:+.6f}**（逐链等权）")
    print(f"全轨迹 <cos²θ> = {np.nanmean(finp):.6f} → **P2 = {p2p_mean:+.6f}**"
          f"（链段加权，= 工具引擎）")
    return {"frame": np.asarray([r[0] for r in rows]),
            "time_ps": np.asarray([r[1] for r in rows]),
            "cos2": fin,
            "P2": np.asarray([r[3] for r in rows], dtype=float),
            "cos2_pooled": finp,
            "P2_pooled": np.asarray([r[6] for r in rows], dtype=float),
            "n_chains": np.asarray([r[4] for r in rows], dtype=int),
            "chain_histogram": hist,
            "note": [_note(r[2]) for r in rows]}


def test_orientation_parameter(u, ag, mode="average", chain_index=0,
                               frames=None, prefix="ori_param"):
    """「取向参数」（相邻链段矢量夹角的 cos²）随时间变化。

    ⚠️ **这不是 P2**：P2 是"链段相对某个参考轴"的二阶序参数
    （``P2 = (3⟨cos²θ⟩−1)/2``，见 :func:`test_cos2_axis`）；
    这里的量只关心**链自己直不直**——相邻两段链段的夹角余弦平方取平均，
    1 = 局部完全笔直，越低说明链在弯折。不需要参考轴。

    ``mode="average"``（默认，脚本的模式 B）：每帧对所有链求平均（另给出
    **链段加权**值 ``pooled``，与工具引擎的"逐链段"统计同口径）；
    ``mode="single"``（脚本的模式 A）：只跟踪第 ``chain_index`` 条链。

    导出 ``{prefix}.csv``：``frame, time(ps), orientation_param, pooled``
    """
    if str(mode).lower() not in ("average", "single"):
        raise ValueError("mode 只能是 average（全链平均）或 single（跟踪单链）")
    rows = []
    for fi in (frame_indices(u) if frames is None else [int(f) for f in frames]):
        ts = u.trajectory[fi]
        chains = chain_coordinates(ag, unwrap=True, verbose=True)
        if not chains:
            continue
        if str(mode).lower() == "single":
            k = int(chain_index) % len(chains)
            val = straightness(chains[k])
            pooled = val
        else:
            _, val = _stack_stats(straightness, chains)
            pooled = pooled_straightness(chains)
        rows.append((fi, float(ts.time), val, pooled))
        print(f"  帧 {fi}: 取向参数 = {val:.6f}（链段加权 {pooled:.6f}）"
              + (f"（第 {int(chain_index)} 条链）" if str(mode).lower() == "single" else ""))

    save_csv(f"{prefix}.csv",
             [[r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
              [r[3] for r in rows]],
             header=("frame,time(ps),orientation_param,pooled  # "
                     "<cos^2(angle between adjacent segment vectors)>; "
                     f"mode={mode}; pooled=链段加权"),
             fmt="%d,%.4f,%.6f,%.6f")
    fin = np.asarray([r[2] for r in rows], dtype=float)
    finp = np.asarray([r[3] for r in rows], dtype=float)
    print(f"全轨迹平均取向参数 = {np.nanmean(fin):.6f}"
          f"（链段加权 {np.nanmean(finp):.6f}；1 = 局部完全笔直）")
    return {"frame": np.asarray([r[0] for r in rows]),
            "time_ps": np.asarray([r[1] for r in rows]),
            "param": fin,
            "param_pooled": finp}


def test_angle_to_axis(u, ag, axis="z", frames=None, prefix="ori_angle"):
    """链段与参考轴的**平均夹角 θ(t)**（度）—— 与工具 1.0.6 新增面板同一口径。

    同时给两种权重：``theta_deg``（逐链等权）与 ``theta_pooled``（链段加权，
    **与工具引擎逐值一致**）。

    导出 ``{prefix}.csv``：``frame, time(ps), theta_deg, theta_pooled``
    """
    rows = []
    iso = float(np.degrees(np.arccos(np.sqrt(1.0 / 3.0))))
    for fi in (frame_indices(u) if frames is None else [int(f) for f in frames]):
        ts = u.trajectory[fi]
        chains = chain_coordinates(ag, unwrap=True, verbose=True)
        _, th = _stack_stats(mean_angle_to_axis, chains, axis)
        thp = pooled_angle(chains, axis)
        rows.append((fi, float(ts.time), th, thp))
        print(f"  帧 {fi}: <θ> = {th:.3f}°（链段加权 {thp:.3f}°；"
              f"完全无序参考 {iso:.2f}°）")

    save_csv(f"{prefix}.csv",
             [[r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
              [r[3] for r in rows]],
             header=("frame,time(ps),theta_deg,theta_pooled  # "
                     "<arccos|unit_v . ref|> in degrees; isotropic = 54.7356; "
                     "theta_pooled=链段加权（= 工具引擎）"),
             fmt="%d,%.4f,%.4f,%.4f")
    fin = np.asarray([r[2] for r in rows], dtype=float)
    finp = np.asarray([r[3] for r in rows], dtype=float)
    print(f"全轨迹平均夹角 = {np.nanmean(fin):.3f}°"
          f"（链段加权 {np.nanmean(finp):.3f}°）")
    return {"frame": np.asarray([r[0] for r in rows]),
            "time_ps": np.asarray([r[1] for r in rows]),
            "theta_deg": fin,
            "theta_pooled": finp}


#: 本模块的默认体系：与其它模块一致 = **20PE30**（20 条 PE30 链 / 301 帧）
DATA_DIR = r"C:\temp\dataset\20PE30"
DEFAULT_TOP = os.path.join(DATA_DIR, "20PE30.xsd")
#: 首选 `.trj`（会自动转 TRR）；目录里没有时自动回退到"已转换的 TRR/XTC 缓存"
#: （缓存目录里按 `<名>.<哈希>.trr` 存放）
DEFAULT_TRJ_CANDIDATES = [
    os.path.join(DATA_DIR, "20PE30.trj"),
    r"C:\temp\MDT\_convert_cache\20PE30.*.trr",
    r"C:\temp\MDT\_convert_cache\20PE30.*.xtc",
]


def menu():
    """菜单 / 命令行入口：``python test/mda_test/orientation.py``。

    默认体系 = ``C:\\temp\\dataset\\20PE30``（20 条 PE30 链 / 301 帧），
    与其它模块一致；换体系用 ``--top/--trj`` 或环境变量。
    """
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("axis", "参考轴", "choice", "z",
          choices=[("z", "Z 轴（默认，膜法向常用）"), ("x", "X 轴"), ("y", "Y 轴")]),
        Q("what", "算哪几项", "choice", "all",
          choices=[("all", "全部（⟨cos²θ⟩+P2、取向参数、夹角 θ(t)）"),
                   ("cos2", "【P2 在这里】⟨cos²θ⟩ 与 P2 = (3⟨cos²θ⟩−1)/2（相对参考轴）"),
                   ("param", "取向参数（相邻链段夹角 cos²，局部直度；**不是 P2**）"),
                   ("angle", "夹角 θ(t)（相对参考轴的度数，对应工具新面板）")]),
        Q("mode", "取向参数的统计方式", "choice", "average",
          choices=[("average", "每帧所有链平均（模式 B）"),
                   ("single", "只跟踪第 0 条链（模式 A）")]),
        Q("n_atoms", "只保留多长的链（重原子数）", "int", 0, lo=0,
          help="0 = 不过滤（含全部线性链；多分散体系请保持 0）"),
    ]

    def _run(cfg):
        u, ag = cfg["u"], cfg["ag"]
        n_atoms = cfg["n_atoms"] or "auto"
        chains = chain_coordinates(ag, n_atoms=n_atoms, unwrap=True, verbose=True)
        if not chains:
            print("⚠️ 该原子组里没有 ≥3 个重原子的线性链，换个对象试试")
            return None
        hist = chain_size_histogram(chains)
        print(f"链数 {len(chains)}，重原子数分布 {hist}")
        if len(hist) > 1:
            print("⚠️ **多分散**体系（链长不一）：程序的「统计权重」现在有"
                  "「链段加权（默认）」和「每条链等权」两档 —— "
                  "前者对上导出的 *_pooled 列（P2_pooled / theta_pooled / pooled），"
                  "后者对上逐链等权列（P2 / theta_deg / orientation_param）。"
                  "两边选同一种权重才能逐值对上。")
        out = {}
        what = cfg["what"]
        if what in ("all", "cos2"):
            out["cos2"] = test_cos2_axis(u, ag, axis=cfg["axis"],
                                         frames=cfg["frames"],
                                         prefix=cfg["prefix"] + "_cos2")
        if what in ("all", "param"):
            out["param"] = test_orientation_parameter(
                u, ag, mode=cfg["mode"], frames=cfg["frames"],
                prefix=cfg["prefix"] + "_param")
        if what in ("all", "angle"):
            out["angle"] = test_angle_to_axis(u, ag, axis=cfg["axis"],
                                             frames=cfg["frames"],
                                             prefix=cfg["prefix"] + "_angle")
        return out

    return run_module("链段取向（allfast-3-2.py 口径）", _run, specs,
                      sub="键中点矢量法：⟨cos²θ⟩ / P2 / 取向参数 / 夹角 θ(t)；"
                          "同时给逐链等权与链段加权（= 程序引擎）两套数",
                      default_frames=100, default_prefix="orientation",
                      default_top=DEFAULT_TOP,
                      default_trj=DEFAULT_TRJ_CANDIDATES)


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
