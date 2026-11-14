# -*- coding: utf-8 -*-
"""界面相容性分析模块（设计大纲第 10–14 章）。

提供四个分析：

- :func:`analyze_density`          沿指定方向的组分密度分布
- :func:`analyze_rdf`              径向分布函数（A-A / B-B / A-B）
- :func:`analyze_contacts`         接触数 / 接触概率
- :func:`analyze_interface_width`  界面位置与界面宽度

说明
----
所有涉及空间分布的量都按模拟盒的**分数坐标**计算，因此对正交盒和三斜盒
（例如 GROMACS 用三斜盒表示的截角八面体盒）都成立。方向 ``axis`` 指的是
第 ``axis`` 个盒矢量方向（0=a, 1=b, 2=c），这与 ``gmx density -sl`` 的含义一致。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import numpy as np

from ..core import AnalysisResult, Panel
from ..preprocess import FrameSelection
from ..units import AMU_PER_A3_TO_G_CM3, bin_edges_to_centers
from .base import frame_iterator, register

__all__ = [
    "fractional_coordinates",
    "box_vector_lengths",
    "density_profile",
    "analyze_density",
    "analyze_rdf",
    "analyze_contacts",
    "analyze_interface_width",
]


# ------------------------------------------------------------------ 工具
class BoxMissing(ValueError):
    """当前体系没有晶胞（盒）信息。

    单独成类是为了让上层**把它当成"这项做不了"而不是"程序出错"**：
    `pipeline` 会把它记成"跳过 + 原因"，而不是在界面上抛一句 ValueError。
    典型来源：非周期体系（MS 里没给晶胞）、只给了坐标没有盒的轨迹（.xyz 等）。
    """


def box_matrix(universe) -> np.ndarray:
    """返回模拟盒矩阵，行向量为 a、b、c 三个盒矢量（Å）。"""
    from MDAnalysis.lib.mdamath import triclinic_vectors

    dims = universe.dimensions
    if dims is None:
        # 有的轨迹格式把盒放在别处（例如拓扑里给了晶胞）：允许调用方在
        # Universe 上挂一个兜底盒（``u._mdta_fallback_box = dims6``）。
        dims = getattr(universe, "_mdta_fallback_box", None)
    if dims is None:
        raise BoxMissing(
            "当前体系没有**晶胞（盒）信息**，做不了需要周期性的空间分布分析"
            "（密度沿某方向的分布 / 界面宽度 / 面内 RDF 等）。\n"
            "  常见原因：非周期体系（Materials Studio 里没有晶胞），"
            "或只有坐标没有盒的轨迹（.xyz 等）。\n"
            "  做法：换用带盒的轨迹（.trj / .tpr + .xtc / .gro 等），"
            "或在 MS 里给结构加上周期晶胞后重新导出。\n"
            "  （不依赖盒的 RDF / Rg / MSD 等照常可用。）")
    return np.asarray(triclinic_vectors(np.asarray(dims, dtype=float)), dtype=float)


def box_vector_lengths(universe) -> np.ndarray:
    """三个盒矢量的长度（Å）。"""
    return np.linalg.norm(box_matrix(universe), axis=1)


def box_volume_from_dims(dims) -> float:
    """由某一帧的 ``dimensions`` 直接算盒体积（Å³）。

    **必须按帧取**：NPT 轨迹的盒子会波动（AdK 实测 80.017–80.135 Å，
    体积相差 0.22%），若像 1.0.0 那样只取"调用分析时轨迹恰好停在哪一帧"的
    体积，同一个分析的结果就会依赖轨迹的偶然位置——实测同一份数据两次
    连续调用得到不同的 g(r)，且无法复现。
    """
    from MDAnalysis.lib.mdamath import triclinic_vectors

    m = np.asarray(triclinic_vectors(np.asarray(dims, dtype=float)), dtype=float)
    return float(abs(np.linalg.det(m)))


def box_min_length_from_dims(dims) -> float:
    """某一帧三个盒矢量长度的最小值（Å），用于判断 rmax 是否安全。"""
    from MDAnalysis.lib.mdamath import triclinic_vectors

    m = np.asarray(triclinic_vectors(np.asarray(dims, dtype=float)), dtype=float)
    return float(np.min(np.linalg.norm(m, axis=1)))


def fractional_coordinates(universe, positions: np.ndarray | None = None,
                           wrap: bool = True) -> np.ndarray:
    """把笛卡尔坐标转成沿盒矢量的分数坐标，可选折回 ``[0, 1)``。"""
    m = box_matrix(universe)
    pos = np.asarray(universe.atoms.positions if positions is None else positions, dtype=float)
    frac = pos @ np.linalg.inv(m)
    return frac % 1.0 if wrap else frac


def _positions_along_axis(universe, ag, axis: int, wrap: bool = True) -> np.ndarray:
    """返回原子组沿第 ``axis`` 个盒矢量的坐标（Å，范围 ``[0, L)``）。"""
    m = box_matrix(universe)
    vec = m[axis]
    length = float(np.linalg.norm(vec))
    pos = np.asarray(ag.positions, dtype=float)
    frac = pos @ np.linalg.inv(m)
    f = frac[:, axis] % 1.0 if wrap else frac[:, axis]
    return f * length


def _group_masses(ag) -> np.ndarray:
    try:
        m = np.asarray(ag.masses, dtype=float)
        if m.size == ag.n_atoms and np.isfinite(m).all() and m.sum() > 0:
            return m
    except Exception:  # noqa: BLE001
        pass
    return np.ones(ag.n_atoms, dtype=float)


def box_heights(universe) -> np.ndarray:
    """三个方向的盒"高度"（Å）：``h_i = V / |ε_ijk·b_j×b_k|``。

    最小镜像近似只在距离小于最小高度的一半时严格成立。
    """
    m = box_matrix(universe)
    v = float(abs(np.linalg.det(m)))
    h = np.empty(3, dtype=float)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        area = float(np.linalg.norm(np.cross(m[j], m[k])))
        h[i] = v / area if area > 0 else np.inf
    return h


def safe_max_cutoff(universe, frac: float = 0.49) -> float:
    """最小镜像/网格搜索允许的最大截断距离（Å）。"""
    return float(np.min(box_heights(universe)) * frac)


def min_distance(universe, pos_a: np.ndarray, pos_b: np.ndarray,
                 start_cutoff: float = 4.0, max_cutoff: float | None = None,
                 max_doublings: int = 6) -> float:
    """两个原子集合之间的最小距离（Å），逐步放大截断距离以兼顾速度与正确性。

    先用较小的截断距离试探（稠密体系几乎总能立刻找到原子对），
    若找不到再翻倍，直到达到盒条件允许的上限。
    """
    from MDAnalysis.lib.distances import capped_distance

    pa = np.asarray(pos_a, dtype=float)
    pb = np.asarray(pos_b, dtype=float)
    if pa.size == 0 or pb.size == 0:
        return float("inf")
    if max_cutoff is None:
        max_cutoff = safe_max_cutoff(universe)
    c = max(float(start_cutoff), 0.5)
    for _ in range(int(max_doublings)):
        if c > max_cutoff:
            break
        try:
            _, d = capped_distance(pa, pb, max_cutoff=float(c),
                                   box=universe.dimensions, return_distances=True)
        except ValueError:
            break
        if d.size:
            return float(np.min(d))
        c *= 2.0
    # 最后退路：分批暴力计算（很慢，仅在极端情况下触发）
    best = np.inf
    step = 256
    for i in range(0, pa.shape[0], step):
        try:
            _, d = capped_distance(pa[i:i + step], pb, max_cutoff=float(max_cutoff),
                                   box=universe.dimensions, return_distances=True)
        except ValueError:
            return float("inf")
        if d.size:
            best = min(best, float(np.min(d)))
    return best


# ------------------------------------------------------------------ 密度分布
def density_profile(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                    axis: int = 2, nbins: int = 100, mode: str = "mass",
                    wrap: bool = True, verbose: bool = False,
                    progress=None):
    """沿第 ``axis`` 个盒矢量方向统计各组分的密度分布。

    做法：把原子在**每帧**的盒坐标换算成沿该盒矢量的分数坐标（0–1），
    在分数坐标上做直方图，最后用整个时间窗口的**平均盒尺寸**把 bin 换算成 Å
    并计算密度。这样即使 NPT 模拟中盒子在漂移，结果也不会被单帧盒子尺寸污染。

    参数
    ----
    groups
        ``{组分名: AtomGroup}``。
    axis
        0/1/2 分别对应盒矢量 a/b/c。
    nbins
        沿该方向的 bin 数。
    mode
        ``"mass"`` 返回质量密度（g/cm³），``"number"`` 返回数密度（Å⁻³）。

    返回
    ----
    ``(bin 中心位置/Å, {组分名: 密度数组}, 元信息 dict)``
    """
    u = mdt.universe
    if axis not in (0, 1, 2):
        raise ValueError("axis 必须是 0、1 或 2")
    nbins = int(nbins or 0)
    if nbins <= 0:
        # 自适应：**0.5 Å 一个箱**（= Forcite Concentration Profile 的默认）。
        # 长度取所选帧第一帧该方向的盒矢量长度（NPT 下各帧差别很小，
        # 而 bin 本身是按分数坐标均分的，不会因盒长波动错位）。
        try:
            idx0 = int(np.asarray(selection.indices)[0])
            u.trajectory[idx0]
            L = float(np.linalg.norm(np.asarray(box_matrix(u))[axis]))
        except Exception:                                     # noqa: BLE001
            L = 100.0
        nbins = max(10, int(round(L / 0.5)))
    fedges = np.linspace(0.0, 1.0, nbins + 1)

    weights = {}
    for name, ag in groups.items():
        weights[name] = (_group_masses(ag) if mode == "mass"
                         else np.ones(ag.n_atoms, dtype=float))

    acc = {name: np.zeros(nbins, dtype=float) for name in groups}
    n_frames = len(selection.indices)
    sum_vbox = 0.0
    sum_len = 0.0
    n_used = 0
    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        m = box_matrix(u)
        inv = np.linalg.inv(m)
        sum_vbox += float(abs(np.linalg.det(m)))
        sum_len += float(np.linalg.norm(m[axis]))
        n_used += 1
        for name, ag in groups.items():
            if ag.n_atoms == 0:
                continue
            pos = np.asarray(ag.positions, dtype=float)
            frac = (pos @ inv)[:, axis]
            if wrap:
                frac = frac % 1.0
            h, _ = np.histogram(frac, bins=fedges, weights=weights[name])
            acc[name] += h
        if progress is not None:
            progress(k + 1, n_frames)

    if n_used == 0:
        raise ValueError("没有任何帧可用于密度分布计算")

    mean_vbox = sum_vbox / n_used
    mean_len = sum_len / n_used
    centers = (np.arange(nbins) + 0.5) / nbins * mean_len
    vbin = mean_vbox / nbins

    for name in acc:
        acc[name] /= float(n_used)
        if mode == "mass":
            acc[name] = acc[name] / vbin * AMU_PER_A3_TO_G_CM3      # g/cm³
        else:
            acc[name] = acc[name] / vbin                            # Å⁻³

    meta = {
        "axis": axis,
        "nbins": nbins,
        "mode": mode,
        "mean_box_length": float(mean_len),
        "mean_box_volume": float(mean_vbox),
        "n_frames": int(n_used),
    }
    return centers, OrderedDict(acc), meta


def density_profile_multi(mdt, groups, selection, axes=(0, 1, 2), *,
                          nbins: int | None = None, mode: str = "mass",
                          wrap: bool = True, verbose: bool = False):
    """沿**多个盒矢量方向**同时统计密度分布（只遍历一次轨迹）。

    为什么单独写：axis="all" 以前是分别调用三次 density_profile —— 同样一帧要
    读三遍、反矩阵算三遍。这里把帧循环提到外面，每帧只读一次坐标、只求一次
    ``inv``，三个方向各自积累直方图，耗时接近单方向。

    返回 ``{axis: (centers, {组分: 密度数组}, meta)}``。
    """
    u = mdt.universe
    axes = tuple(int(a) for a in axes)
    if any(a not in (0, 1, 2) for a in axes):
        raise ValueError("axis 必须是 0、1 或 2")
    idx0 = int(np.asarray(selection.indices)[0])
    u.trajectory[idx0]
    m0 = box_matrix(u)
    nb = {}
    for ax in axes:
        n = int(nbins or 0)
        if n <= 0:
            L = float(np.linalg.norm(m0[ax]))
            n = max(10, int(round(L / 0.5)))          # 自适应 0.5 Å/箱
        nb[ax] = n
    fedges = {ax: np.linspace(0.0, 1.0, nb[ax] + 1) for ax in axes}

    weights = {}
    for name, ag in groups.items():
        weights[name] = (_group_masses(ag) if mode == "mass"
                         else np.ones(ag.n_atoms, dtype=float))

    acc = {ax: {name: np.zeros(nb[ax]) for name in groups} for ax in axes}
    sum_vbox = 0.0
    lens = {ax: 0.0 for ax in axes}
    n_used = 0
    n_total = len(selection.indices)
    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        m = box_matrix(u)
        inv = np.linalg.inv(m)
        sum_vbox += float(abs(np.linalg.det(m)))
        for ax in axes:
            lens[ax] += float(np.linalg.norm(m[ax]))
        n_used += 1
        for name, ag in groups.items():
            if ag.n_atoms == 0:
                continue
            frac_all = np.asarray(ag.positions, dtype=float) @ inv
            for ax in axes:
                fr = frac_all[:, ax]
                if wrap:
                    fr = fr % 1.0
                h, _ = np.histogram(fr, bins=fedges[ax], weights=weights[name])
                acc[ax][name] += h
    if n_used == 0:
        raise ValueError("没有任何帧可用于密度分布计算")
    mean_vbox = sum_vbox / n_used
    out = {}
    for ax in axes:
        mean_len = lens[ax] / n_used
        centers = (np.arange(nb[ax]) + 0.5) / nb[ax] * mean_len
        vbin = mean_vbox / nb[ax]
        prof = OrderedDict()
        for name in groups:
            y = acc[ax][name] / float(n_used)
            prof[name] = (y / vbin * AMU_PER_A3_TO_G_CM3 if mode == "mass"
                          else y / vbin)
        out[ax] = (centers, prof, {
            "axis": ax, "nbins": nb[ax], "mode": mode,
            "mean_box_length": float(mean_len),
            "mean_box_volume": float(mean_vbox),
            "n_frames": int(n_used),
        })
    return out


@register("density", "密度分布")
def analyze_density(mdt, groups: Mapping[str, object] | None = None,
                    selection: FrameSelection | None = None, *,
                    group=None, axis: int = 2, nbins: int = 100,
                    mode: str = "mass", verbose: bool = False) -> AnalysisResult:
    """组分密度分布分析（设计大纲第 11 章）。

    输出 Density–Position 曲线（绝对密度 + 归一化密度），
    归一化曲线用于后续判断界面位置与界面宽度。
    """
    if groups is None:
        if group is None:
            raise ValueError("必须提供 groups 或 group")
        groups = {"组分": group}
    groups = OrderedDict((k, v) for k, v in groups.items() if v is not None and v.n_atoms > 0)
    if not groups:
        raise ValueError("没有有效的组分可用于密度分析")
    if selection is None:
        raise ValueError("必须提供 frame selection")

    # 分箱自适应：nbins 为空/<=0 时按 **0.5 Å/箱** 取（Forcite 的 Concentration
    # Profile 默认就是 0.5 Å：PE50 的 120 Å → 240 箱、graphite 27 的 40 Å → 80 箱，
    # 与它们导出的 CSV 行数完全一致）。所以界面上不用每次手填。
    # axis="all"：三个晶向一起算（C100/C010/C001），各自一条曲线画在同一张图上。
    # 这是默认行为：Forcite 的 Concentration Profile 也是分别按三个方向导出的。
    if str(axis) == "all" or (isinstance(axis, int) and axis < 0):
        # 一次遍历同时算三个方向（不再分别扫三遍轨迹；见 density_profile_multi）。
        multi = density_profile_multi(mdt, groups, selection, axes=(0, 1, 2),
                                      nbins=nbins, mode=mode, verbose=verbose)
        ylabel = "质量密度 ρ (g/cm³)" if str(mode) == "mass" else "数密度 ρ (1/Å³)"
        res = AnalysisResult(
            name="density", title="沿 a / b / c 三个方向的组分密度分布",
            meta={"axes": [0, 1, 2],
                  "nbins": 0 if nbins is None else int(nbins),
                  "mode": str(mode)},
        )
        res.panels = [
            Panel(xlabel="位置 (Å)", ylabel=ylabel,
                  title="组分密度分布（C100 / C010 / C001）"),
            Panel(xlabel="位置 (Å)", ylabel="归一化密度 ρ/ρ_bulk",
                  title="归一化密度分布（C100 / C010 / C001）"),
        ]
        for ax in (0, 1, 2):
            tag = {0: "C100", 1: "C010", 2: "C001"}[ax]
            centers, prof, dmeta = multi[ax]
            vbox = float(dmeta.get("mean_box_volume") or 0.0)
            for name, y in prof.items():
                res.add_curve(f"{name} ({tag})", centers, y, panel=0, group=name)
                bulk = _bulk_value(y)
                res.summary[f"{tag} {name} 体相密度"] = float(bulk)
                res.summary[f"{tag} {name} 平均密度"] = float(np.nanmean(y))
                try:
                    ga = groups[name]
                    rho_bar = _rho_bar_in_profile_units(ga, vbox, mode)
                except Exception:                         # noqa: BLE001
                    rho_bar = float("nan")
                res.summary[f"{tag} {name} 整盒平均密度 ρ̄"] = rho_bar
                den = rho_bar if np.isfinite(rho_bar) and rho_bar > 0 else bulk
                res.add_curve(f"{name} (Forcite 相对浓度 ρ/ρ̄) ({tag})", centers,
                              y / den if den > 0 else y * np.nan,
                              panel=1, group=name)
        res.add_notes("三个方向一次算出（C100=a / C010=b / C001=c），"
                      "与 Forcite Concentration Profile 的方向一一对应。")
        res.add_notes(
            "归一化曲线（面板 2）= ρ/ρ̄，ρ̄ = **整盒平均密度**（与 Forcite 的 "
            "`Relative concentration` 同口径）：数密度口径下 ρ̄ = N/V，可直接与 "
            "Forcite 逐箱对比；质量密度口径下 ρ̄ = Σm/V（g/cm³），曲线无量纲、均值 1。\n"
            "摘要里的「体相密度」是另一回事（上四分位区中位数），只用来判断 plateau。")

        return res

    centers, prof, dmeta = density_profile(mdt, groups, selection, axis=axis,
                                           nbins=nbins, mode=mode, verbose=verbose)
    u = mdt.universe
    axis_name = "abc"[axis]

    ylabel = "质量密度 ρ (g/cm³)" if mode == "mass" else "数密度 ρ (1/Å³)"
    res = AnalysisResult(
        name="density",
        title=f"沿 {axis_name} 方向的组分密度分布",
        meta={"axis": axis, "nbins": 0 if nbins is None else int(nbins),
              "mode": mode, **dmeta},
    )
    res.panels = [
        Panel(xlabel=f"位置沿 {axis_name} 方向 (Å)", ylabel=ylabel,
              title="组分密度分布"),
        Panel(xlabel=f"位置沿 {axis_name} 方向 (Å)", ylabel="归一化密度 ρ/ρ_bulk",
              title="归一化密度分布（用于确定界面）"),
    ]

    # 归一化基准：**Forcite 口径**——ρ(z)/ρ̄，ρ̄ = N/V_box（整盒平均密度）。
    # 用户实测（graphite 27 的 eth_Concentration Profile.csv）：Forcite 的
    # "Relative concentration" 峰值 16.211，正好等于 ρ(z)_max / (N/V_box)（16.0）。
    # 以前这里用"体相上四分位中位数"归一，方便找界面，但与 Forcite 差常数因子，
    # 现在统一成 Forcite 口径；体相密度仍单独在摘要里给出，界面检测照旧可用。
    for name, y in prof.items():
        # `group=name`：一张图里有多个组分时，前端默认只显示**第一个组分**
        # （其余在图例里点开）—— 见 chartOption.buildOption 的 legend.selected。
        res.add_curve(name, centers, y, panel=0, group=name)
        bulk = _bulk_value(y)
        res.summary[f"{name} 体相密度"] = float(bulk)
        res.summary[f"{name} 平均密度"] = float(np.nanmean(y))
        # ρ̄：整盒平均（与 Forcite 相同口径）
        try:
            ga = groups[name]
            vbox_mean = float(dmeta.get("mean_box_volume") or 0.0)
            rho_bar = _rho_bar_in_profile_units(ga, vbox_mean, mode)
        except Exception:                                     # noqa: BLE001
            rho_bar = float("nan")
        res.summary[f"{name} 整盒平均密度 ρ̄"] = rho_bar
        den = rho_bar if np.isfinite(rho_bar) and rho_bar > 0 else bulk
        res.add_curve(f"{name} (Forcite 相对浓度 ρ/ρ̄)", centers,
                      y / den if den > 0 else y * np.nan,
                      panel=1, group=name)

    res.summary["沿轴平均盒长 (Å)"] = dmeta["mean_box_length"]
    res.summary["平均盒体积 (Å³)"] = dmeta["mean_box_volume"]
    res.summary["统计帧数"] = dmeta["n_frames"]
    # 口径必须写进结果本身（不能只藏在 metadata/纵轴标签里）：质量密度与数密度
    # 数值差一个摩尔质量量级（水 0.0334 个/Å³ vs 0.997 g/cm³），导出成表以后
    # 光看"体相密度 0.168"根本分不清是哪个。
    res.summary["密度口径"] = ("质量密度 (g/cm³)" if str(mode) == "mass"
                          else "数密度 (1/Å³)")
    res.summary["密度单位"] = "g/cm³" if str(mode) == "mass" else "1/Å³"
    res.add_notes("方向按模拟盒矢量的分数坐标计算，对三斜盒同样成立；"
                  "密度用时间窗口内的平均盒尺寸归一化。")
    res.add_notes("体相密度取分布上四分位区的中位数，作为归一化基准。")
    res.add_notes(
        "Forcite 的 Relative concentration 是**按原子数**归一化的（= 数密度口径），"
        "要与它逐箱对比请把密度口径选「数密度」；选「质量密度」时 ρ/ρ̄ 用质量加权，"
        "形状会略有不同（重原子主导）。\n"
        "归一化曲线（面板 2）= ρ/ρ̄，ρ̄ = N/V_box 整盒平均 —— 与 Forcite 的 "
        "`Relative concentration` 同口径（实测 graphite 27 峰值 16.211 = 16.0 一次对齐）。"
        "⚠️ 本剖面是**时间平均**结果。若要统计**窄带**（±几 Å）内的原子数或密度，"
        "必须**逐帧**用动态参考（膜质心/界面位置）重算，不能从时间平均剖面里取窄带 —— "
        "实测某 CG 膜体系：逐帧膜质心得 0.515 个水珠子，用时间平均质心固定带得 7.76 个"
        "（**15×**），因为膜质心 z 的波动 std 就有 5.53 Å（范围 20.6–46.3 Å）。")
    res.add_notes(
        f"本次密度口径 = {res.summary['密度口径']}；"
        f"各组分的「体相密度」「平均密度」都按这个口径给出（质量密度与数密度"
        f"相差一个摩尔质量量级，不可直接互比）。用 mode=\"number\"/\"mass\" 切换。")
    return res


def _rho_bar_in_profile_units(ga, vbox: float, mode) -> float:
    """整盒平均密度 ρ̄，**与密度剖面的单位一致**（Forcite 的 ρ/ρ̄ 基准）。

    ⚠️ 这里以前是个真 bug：剖面在 ``mode="mass"`` 时是 **g/cm³**，但 ρ̄ 用的是
    ``Σm[amu] / V[Å³]``（= amu/Å³），两者差 **1.66054** 倍 —— 于是"归一化密度"
    面板整体被乘了 1.66（实测 1000water：均值 1.660539 而不是 1.0），
    摘要里"平均密度 1.0"和"整盒平均密度 ρ̄ 0.602"也自相矛盾。
    数密度口径下两者单位相同（1/Å³），所以只有质量密度出错。

    返回：mass → ρ̄[g/cm³]；number → ρ̄[1/Å³]。
    """
    v = float(vbox or 0.0)
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    if str(mode) == "mass":
        m = float(np.sum(np.asarray(ga.masses, dtype=float)))   # amu
        return m / v * AMU_PER_A3_TO_G_CM3                      # g/cm³
    return float(ga.n_atoms) / v                                # 1/Å³


def _safe_ratio(y: np.ndarray, denom: float) -> np.ndarray:
    """按体相值归一化；体相值为 0 或非有限时返回全 0（而不是 inf/nan）。"""
    y = np.asarray(y, dtype=float)
    d = float(denom)
    if not np.isfinite(d) or d <= 0:
        return np.zeros_like(y)
    return y / d


def _bulk_value(y: np.ndarray, frac: float = 0.25) -> float:
    """估计体相（plateau）密度。

    用**上四分位区的中位数**：对均匀体系（纯水）就是平均密度，对分层体系
    就是致密相的体相值。

    但要注意一种会出错的情形：某个组分只占很窄的一段（例如一条多肽溶解在
    204 Å 长的盒子里，只占不到 15% 的 bin），这时"最高的 25% 的 bin"里
    大部分是**空 bin**，中位数会退化成 0，归一化时就会炸成无穷大。
    所以先做一个判断：若上四分位区的中位数太小（相对峰值而言），
    就改用**峰值本身**作为该组分的参考密度。
    """
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    peak = float(np.percentile(a, 99.0))
    k = max(1, int(round(a.size * frac)))
    bulk = float(np.median(np.sort(a)[-k:]))
    # 体相值远低于峰值 → 该组分没有"体相"，用峰值当参考
    if not np.isfinite(bulk) or bulk <= 0.0:
        return peak if peak > 0 else 0.0
    if peak > 0 and bulk < 0.5 * peak:
        return peak
    return bulk


# ------------------------------------------------------------------ RDF
#: RDF 的配对模式（框架 §三：默认只统计**分子间**配对）
RDF_MODES = ("inter", "intra", "total")

RDF_MODE_LABELS = {
    "inter": "分子间（不同分子，默认）",
    "intra": "分子内（同一分子）",
    "total": "总体（两者都含，1.0.0 旧行为）",
}


#: 剔除 1-2 成键配对的距离下限（Å）：比任何共价键都长
#: （C–C 1.54 / C–O 1.43 / C–H 1.09），用于面内 RDF 的近邻过滤。
_BOND_EXCLUDE = 2.2


def _inplane_pair_hist(pos2d: np.ndarray, L_in: np.ndarray, rmax: float,
                       edges: np.ndarray, min_dist: float = _BOND_EXCLUDE):
    """面内 **2D 周期** 距离直方图（按**有序对**计数，返回 ``hist`` 或 ``None``）。

    为什么要它：面内 RDF 以前直接展开 ``N×N`` 距离矩阵（N=3000 → 144 MB 临时数组、
    约 900 ms/帧，是这一项最慢的唯一原因）。现在按"候选对"走两条分支：

    - ``rmax < 半盒边``：``cKDTree(pos, boxsize=L_in)`` 建在 2D 周期环面上，
      ``query_pairs(r)`` 直接给出无序对，只对这几对算距离 —— 候选对骤降；
    - 否则（cutoff 大、几乎全配对）：**分块**枚举 ``i<j``，每块只开两个
      ``(chunk, N)`` 的 float64 临时数组（最小镜像用 ``min(dx, L−dx)``，
      不用 ``round``），内存与缓存都友好；``rmax`` 大到覆盖整个盒对角时
      **不做距离筛选**（本来全都在范围内），省掉一次掩码。

    无序对 ``i<j`` 各贡献 2 个有序对，所以最后 ``hist *= 2`` —— 与旧的
    全矩阵（有序对）直方图**逐位相同**。
    """
    pos = np.asarray(pos2d, dtype=float)
    if pos.shape[0] < 2:
        return None
    L = np.asarray(L_in, dtype=float).ravel()
    if L.size != 2 or not np.all(L > 0):
        return None
    w = np.mod(pos, L)
    n = w.shape[0]
    rmax = float(rmax)
    half = 0.5 * float(np.min(L))
    hist = np.zeros(edges.size - 1, dtype=float)

    def _acc(dvals):
        if dvals.size:
            keep = dvals > min_dist
            if keep.any():
                h, _ = np.histogram(dvals[keep], bins=edges)
                hist[:] += h

    if rmax < half:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(w, boxsize=L)
            pr = np.asarray(tree.query_pairs(r=rmax, output_type="ndarray"),
                            dtype=np.int64).reshape(-1, 2)
            if pr.size:
                dp = w[pr[:, 0]] - w[pr[:, 1]]
                dp -= L[None, :] * np.round(dp / L[None, :])
                _acc(np.hypot(dp[:, 0], dp[:, 1]))
            return hist * 2.0
        except Exception:                                  # noqa: BLE001 - 退回分块
            pass
    # 分块全配对：最小镜像用 min(|dx|, L−|dx|)，避免 round 与 3 维临时数组
    L0, L1 = float(L[0]), float(L[1])
    max_d = float(np.hypot(0.5 * L0, 0.5 * L1))            # 盒内最大面内距离
    need_cut = rmax < max_d
    r2cut = rmax * rmax
    chunk = max(64, int(3_000_000 // max(n, 1)))
    rows = np.arange(n)
    from ..cancel import tick as _cancel_tick

    for s in range(0, n, chunk):
        _cancel_tick(s // max(chunk, 1))    # 每块问一次取消（单块 ~几十 ms）
        e = min(s + chunk, n)
        a = w[s:e]
        b = w
        # 只取 j > i 的那一半（无序对）→ 每行起点 s+row+1
        for r in range(a.shape[0]):
            gi = s + r
            if gi + 1 >= n:
                break
            _cancel_tick(gi, 256)
            dx = b[gi + 1:, 0] - a[r, 0]
            dy = b[gi + 1:, 1] - a[r, 1]
            dx = np.abs(dx)
            dy = np.abs(dy)
            dx = np.minimum(dx, L0 - dx)
            dy = np.minimum(dy, L1 - dy)
            r2 = dx * dx + dy * dy
            if need_cut:
                sel = r2 <= r2cut
                if not sel.any():
                    continue
                _acc(np.sqrt(r2[sel]))
            else:
                _acc(np.sqrt(r2))
    return hist * 2.0


def _atom_label(a) -> str:
    """原子的可读标签，**容忍缺属性**：``resname+resid(name)`` → ``name`` → ``#索引``。

    为什么：`.xyz` 之类拓扑没有 residue 信息、甚至没有原子名，直接读
    ``a.resname`` 会抛 ``NoDataError`` 把整项分析搞崩（实测无 resname 的合成体系）。
    """
    try:
        return f"{a.resname}{a.resid}({a.name})"
    except Exception:                                          # noqa: BLE001
        pass
    try:
        return str(a.name)
    except Exception:                                          # noqa: BLE001
        return f"#{int(a.index)}"


def _atom_molnums(ag) -> np.ndarray | None:
    """原子级分子编号。

    ``molnums`` 在 MDAnalysis 里是**残基级**属性，取 ``ag.molnums`` 时会自动
    广播到原子级。返回 ``None`` 表示体系没有分子信息（例如猜键失败），
    此时按分子区分配对无法进行。

    ⚠️ 自带 molnums 也可能是**退化**的：Material Studio 导出的 PDB 会把整个体系
    塞进 1–2 个残基（实测 5596 原子 / 2 个残基 → 全体原子算"2 个分子"），
    这会让"分子间 RDF"把所有配对都当成分子内而全部剔除、逐分子 Rg 变成整组 Rg，
    且**不报错**。所以这里加一道判据：若"分子数"远小于键图连通分量数，
    就改用**键连通分量**作为分子编号（那才是真实的分子划分）。
    """
    m = None
    try:
        cand = np.asarray(ag.molnums, dtype=np.int64)
        if cand.size == ag.n_atoms:
            m = cand
    except Exception:  # noqa: BLE001 - 无该属性时抛 NoDataError
        m = None

    frag_ids = None
    try:
        frags = ag.fragments
        if len(frags) > 1:
            frag_ids = np.empty(ag.n_atoms, dtype=np.int64)
            for k, f in enumerate(frags):
                pos = np.searchsorted(np.asarray(ag.indices), np.asarray(f.indices))
                frag_ids[pos] = k
    except Exception:  # noqa: BLE001
        frag_ids = None

    if m is None:
        return frag_ids
    n_mol = int(np.unique(m).size)
    if frag_ids is not None and n_mol * 4 < int(np.unique(frag_ids).size):
        return frag_ids          # 自带划分退化 → 用键连通分量
    return m


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """两个编号数组是否描述了**同一个划分**（只差重新编号）。

    用法：判断 ``molnums`` 与键连通分量是不是同一套分子划分。不一致就说明
    molnums 不可信（要么把分子切碎了，要么把不同分子并成了一个），
    RDF / 接触 / 界面 / 密度的 ``inter``/``intra`` 口径必须改用键图。
    """
    if a.size != b.size:
        return False
    try:
        n_a = int(np.unique(a).size)
        n_b = int(np.unique(b).size)
        if n_a != n_b:
            return False
        # 每个元素一个 (a,b) 组合；组合数 == 分子数 当且仅当两个划分同构
        combos = np.unique(np.stack([a, b], axis=1), axis=0)
        return int(combos.shape[0]) == n_a
    except Exception:                                        # noqa: BLE001
        return False


def _molnums_for_filter(ag, force_bonds: bool = False) -> np.ndarray | None:
    """用于配对过滤的分子编号；未归属残基（``-1``）给每个原子独立编号。

    为什么：``-1`` 表示该残基没被归入任何分子。若原样参与比较，所有 ``-1``
    原子会被判成"同一个分子"，于是分子间 RDF 会把它们整体丢掉。给它们各自
    独立编号后，它们之间的配对按"分子间"处理（更保守，也不会凭空消失）。

    ⚠️ **只在 RDF / 接触 / 界面 / 密度这条链上使用**（``conformation.py`` 与
    ``dynamics.py`` 用的是 :func:`_atom_molnums`，不受这里的修正影响 —— Rg/MSD
    已经与 Forcite 对齐，不能再动）。

    退化判据（两个方向都算退化，两种都改用**键连通分量**）：
      · ``molnums`` 比键块**细**（用户实测 PE50）：残基级 ``molnums`` 把**同一个键
        连通块切成几份**（每个分子 32 原子 = 16 个 4 原子残基），跨残基的键算出
        0.01–0.28 Å 的假距离，分子间 g(r) 在 r<2 Å 冒出 8~9（Forcite 那边严格为 0）；
      · ``molnums`` 比键块**粗**（resid 重号 / 整体系被塞进 1–2 个残基）：
        **不同分子**的原子被并成同一个 molnum，于是 ``inter`` 口径把**跨组接触整套
        误删** —— 实测 420 原子两相玩具：482 对 → 325 对、最小间距 0.30 → 2.41 Å，
        而且报出来的"分子内配对占比"是 0，完全看不出是口径造成的。

    ⚠️ **必须在整个体系上算、再按原子序号取子集**（见 :func:`global_molecule_ids`）。
    按组各算一套会得到两套互不可比的编号：resid 重号时 A 组的 "molnum 0" 与 B 组的
    "molnum 0" 其实是两个不同分子，逐组检查各自都"自洽"，跨组比较却全错。
    """
    return global_molecule_ids(ag.universe, force_bonds=force_bonds,
                               _indices=np.asarray(ag.indices, dtype=np.int64))


_GLOBAL_CACHE: dict[int, tuple] = {}


def global_molecule_ids(universe, *, force_bonds: bool = False,
                        _indices: np.ndarray | None = None) -> np.ndarray | None:
    """**整个体系**的原子级分子编号（长度 = ``universe.atoms.n_atoms``）。

    为什么要全局算：``molnums`` / 键连通分量都是"全体系"属性，按组切片后再各自
    纠偏，A、B 两组的编号就不可比了（resid 重号是这个坑的典型：两条链 resid 1、2，
    水也是 1、2 → A 组看 molnum 0 是"链 1"、B 组看 molnum 0 是"水 1"，而
    ``inter`` 只做相等比较，于是把这两个不同分子的接触判成"分子内"删掉）。

    ``_indices`` 只用于内部取子集（避免多取一次 ``ag.indices``）。
    """
    u = universe
    n_u = int(u.atoms.n_atoms)
    key = id(u)
    hit = _GLOBAL_CACHE.get(key)
    if hit is not None and hit[0] == n_u and hit[1] == bool(force_bonds):
        ids = hit[2]
    else:
        ids = _compute_global_ids(u, force_bonds=bool(force_bonds))
        if len(_GLOBAL_CACHE) > 8:
            _GLOBAL_CACHE.clear()
        _GLOBAL_CACHE[key] = (n_u, bool(force_bonds), ids)
    if ids is None:
        return None
    if _indices is None:
        return ids
    return np.asarray(ids, dtype=np.int64)[_indices]


def _compute_global_ids(universe, *, force_bonds: bool = False) -> np.ndarray | None:
    u = universe
    n_u = int(u.atoms.n_atoms)
    if n_u == 0:
        return None
    # 键连通分量（全体系，一次）
    comp = None
    try:
        from ..topology import bond_components

        edges = np.asarray(u.bonds.indices, dtype=np.int64).reshape(-1, 2)
        if edges.size:
            comp = bond_components(n_u, edges).astype(np.int64)
            if int(np.unique(comp).size) < 2:
                comp = None
    except Exception:                                        # noqa: BLE001
        comp = None
    if force_bonds and comp is not None:
        return comp                      # Forcite 口径：直接按键图分子
    m = _atom_molnums(u.atoms)
    if m is None:
        return comp
    if comp is not None and _molnums_unreliable(m, comp):
        m = comp
    neg = m < 0
    if neg.any():
        m = m.copy()
        m[neg] = -(np.arange(m.size, dtype=np.int64)[neg] + 1)
    return m


def _molnums_unreliable(m: np.ndarray, comp: np.ndarray) -> bool:
    """``molnums`` 的划分与键图**不一致到必须弃用**吗？

    两个方向都可能坏，但**不能**一见不一致就换成键图 —— 4 点水模型（TIP4P）
    就是这样：一个分子 = 键块 {O,H1,H2} + **孤立的 M 虚位点**，molnums 把 4 个
    原子算一个分子（正确），键图却给出两个块（一个是单原子）。实测若一律信键图，
    水的「分子内邻居数」会从 3.000 掉到 1.500（自检当场失败）。

    所以只认这两种"真的坏了"：
      ① **键块横跨多个 molnum**（molnums 比键图细）：同一分子被切碎，跨碎片的键
         算出 0.01–0.28 Å 的假距离（PE50 实测，分子间 g(r) 在 r<2 Å 冒出 8~9）；
      ② **一个 molnum 里含 ≥2 个"有实质"的键块**（每块 ≥2 原子）：不同分子被并成
         一个 molnum —— resid 重号或 MS 导出把整个体系塞进 1–2 个残基时就是这样，
         ``inter`` 口径会把**跨组接触整套误删**（实测 482 对 → 325 对、最小间距
         0.30 Å → 2.41 Å，且"分子内配对占比"仍显示 0，完全看不出是口径问题）。
    单原子块（虚位点 / 离子 / M 位点）不算"实质"，因此不会触发 ②。
    """
    try:
        if m.size != comp.size or int(np.unique(m).size) == int(np.unique(comp).size) \
                and _same_partition(m, comp):
            return False
        # ① 键块横跨多个 molnum
        order = np.argsort(comp, kind="stable")
        cs, ms = comp[order], m[order]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        ends = np.r_[starts[1:], order.size]
        for s, t in zip(starts.tolist(), ends.tolist()):
            if np.unique(ms[s:t]).size > 1:
                return True
        # ② 一个 molnum 里含 ≥2 个"有实质"的键块
        order = np.argsort(m, kind="stable")
        ms, cs = m[order], comp[order]
        starts = np.flatnonzero(np.r_[True, ms[1:] != ms[:-1]])
        ends = np.r_[starts[1:], order.size]
        for s, t in zip(starts.tolist(), ends.tolist()):
            _, cnt = np.unique(cs[s:t], return_counts=True)
            if int(np.sum(cnt >= 2)) >= 2:
                return True
        return False
    except Exception:                                        # noqa: BLE001
        return False


def mode_pair_counts(ga, gb, same: bool) -> tuple[dict[str, float] | None, str]:
    """三种模式下的**理想有序原子对数**，供 g(r) 归一化使用。

    为什么不能用 ``N_a·N_b`` 一把梭：``total`` 含分子内配对，``inter`` 把它们
    全部剔除，``intra`` 只留它们。三种模式若共用同一个分母，``g(r)`` 就不再是
    密度归一化的量（``inter`` 被系统性压低、``intra`` 被抬高），由 g(r) 积分出来
    的配位数会跟着错——这正是 1.0.0 的问题。

    恒等式（自检用）：``inter + intra == total``。
    """
    ma = _molnums_for_filter(ga)
    mb = ma if same else _molnums_for_filter(gb)
    if ma is None or mb is None:
        return None, "体系没有分子编号（molnums），无法按分子区分配对"
    na, nb = int(ga.n_atoms), int(gb.n_atoms)
    nmax = int(max(int(ma.max(initial=0)), int(mb.max(initial=0)))) + 1
    ca = np.bincount(ma, minlength=nmax).astype(float)
    cb = ca if same else np.bincount(mb, minlength=nmax).astype(float)
    overlap = float(np.sum(ca * cb))            # 同一分子内的有序对（含自配对）
    if same:
        total = float(na) * (na - 1)
        intra = overlap - float(na)             # 扣掉 i==j 的自配对
    else:
        total = float(na) * float(nb)
        intra = overlap
    inter = total - intra
    return ({"total": total, "inter": inter, "intra": intra},
            f"有序原子对：分子间 {inter:,.0f} / 分子内 {intra:,.0f} / 合计 {total:,.0f}")


def _subsample_group(ag, max_atoms: int):
    """抽稀：**按分子整取**，而不是按原子等间隔抽。

    为什么必须按分子：等间隔抽原子会把分子拆散——4 原子/分子的水按 stride 8
    抽取后每个保留分子只剩 1 个原子，分子内配对被清空，intra 模式的 g(r)
    直接变成空（实测踩到）。整分子抽取既保留分子内结构，对分子间结构也仍是
    等价抽样（g(r) 是密度归一化的量，抽稀只增加噪声）。

    返回 ``(新原子组, 原原子数, 新原子数, 说明)``。
    """
    n0 = int(ag.n_atoms)
    m = _atom_molnums(ag)
    if m is None:
        step = max(1, int(np.ceil(n0 / float(max_atoms))))
        sub = ag[::step]
        return sub, n0, int(sub.n_atoms), f"原子等间隔（每 {step} 个取 1 个，无分子信息）"
    uniq = np.unique(m)
    if uniq.size <= 1:
        return ag, n0, n0, "仅 1 个分子，未抽稀"
    atoms_per_mol = n0 / float(uniq.size)
    n_mol_keep = max(1, int(max_atoms / max(atoms_per_mol, 1.0)))
    if n_mol_keep >= uniq.size:
        return ag, n0, n0, "未抽稀"
    step = max(1, int(np.ceil(uniq.size / float(n_mol_keep))))
    kept = uniq[::step]
    sub = ag[np.isin(m, kept)]
    return (sub, n0, int(sub.n_atoms),
            f"按分子整取（每 {step} 个分子取 1 个，共 {kept.size:,} 个分子）")


#: 小于这个距离的原子对一律当异常（数值重叠 / 周期镜像残留），不参与 g(r)。
_RDF_MIN_DIST = 0.3


def _wrap_by_molecule(pa, pb, box, ma, mb):
    """把每个**分子**作为刚体平移进盒（分子内构象不变）。

    ``ma`` / ``mb`` 是原子级分子编号（与 ``pa`` / ``pb`` 的行一一对应）。
    做法：分数坐标下把该分子中心减去 ``floor(中心)``（即整格平移），
    于是分子重心落进 [0,1)³，而分子内部的相对位置一个数都不变。
    """
    from MDAnalysis.lib.mdamath import triclinic_vectors

    dims = np.asarray(box, dtype=float)
    M = np.asarray(triclinic_vectors(dims.astype(np.float32)), dtype=float)
    Minv = np.linalg.inv(M)

    def _one(p, comp):
        if comp is None or p is None or not len(p):
            return p
        frac = np.asarray(p, dtype=float) @ Minv.T
        out = np.array(p, dtype=float, copy=True)
        for k in np.unique(comp):
            idx = comp == k
            if idx.sum() == 0:
                continue
            center = frac[idx].mean(axis=0)
            shift = np.floor(center)
            if np.any(shift != 0.0):
                out[idx] -= shift @ M.T
        return out

    pa2 = _one(pa, ma)
    pb2 = None if pb is None else _one(pb, mb)
    return pa2, pb2


def _pairs_bruteforce(pa, pb, rmax: float, box, same: bool):
    """**大 cutoff 兜底**：MDAnalysis 的网格搜索（FastNS）要求
    ``cutoff < 半个最短盒边``，超过就抛
    ``ValueError: Cutoff 20.0 too large for box (max 15.68…)``；而 Materials Studio
    Forcite 的 RDF 允许把 cutoff 设到更大（用户实测设 20 Å，两边就对不上）。
    这里退化成**分块暴力距离**：内存按块控制，结果与网格法逐点一致。
    ``same=True`` 只返回 ``i<j``（不含自配对），与 ``self_capped_distance`` 同约定。
    """
    from MDAnalysis.lib.distances import distance_array

    other = pa if same else pb
    n = int(pa.shape[0])
    if not n or other is None or not len(other):
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    chunk = max(32, min(1024, int(4_000_000 // max(int(other.shape[0]), 1))))
    I, J, D = [], [], []
    from ..cancel import tick as _cancel_tick

    for s in range(0, n, chunk):
        _cancel_tick(s // max(chunk, 1))    # 每块问一次取消（分块暴力是帧内最重的活）
        d = np.asarray(distance_array(pa[s:s + chunk], other, box=box))
        if same:
            for r in range(d.shape[0]):
                gi = s + r
                row = d[r]
                idx = np.flatnonzero(row[gi + 1:] <= rmax) + (gi + 1)
                if idx.size:
                    I.append(np.full(idx.size, gi, dtype=np.int64))
                    J.append(idx.astype(np.int64))
                    D.append(row[idx])
        else:
            flat = np.flatnonzero(d <= rmax)
            if flat.size:
                r_idx, c_idx = np.divmod(flat, d.shape[1])
                I.append(s + r_idx.astype(np.int64))
                J.append(c_idx.astype(np.int64))
                D.append(d.ravel()[flat])
    if not I:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    return np.column_stack([np.concatenate(I), np.concatenate(J)]), np.concatenate(D)


def _rdf_accumulate(u, ref, conf, edges, frame_indices, same: bool,
                    exclude_bonded: bool = False, mode: str = "total",
                    verbose: bool = False, wrap_inbox: bool = False,
                    mol_from_bonds: bool = False,
                    distance: str = "minimage", drop_overlap: float = 0.0):
    """手工累积 RDF 直方图，支持任意帧号列表与配对模式。

    配对计数约定
    ------------
    RDF 的归一化基于**有序**原子对 ``Σ_i Σ_{j≠i}``。``self_capped_distance``
    每个无序对只返回一次（且不含自配对），因此同组分情形要把直方图乘以 2，
    分子端仍使用 ``N(N−1)``，两者才自洽。``same=False`` 时 ``capped_distance``
    的返回天然是有序对（reference 与 configuration 的笛卡尔积），无需修正。

    配对模式（``mode``）
    -------------------
    ``inter`` 只保留 ``molnums[i] != molnums[j]``，``intra`` 只保留相等者，
    ``total`` 全要（1.0.0 行为）。``bonded`` 排除是**独立开关**，与模式无关。

    返回 ``(hist, n_used, qc)``；``qc`` 给出"这个数是谁的数"所需的诊断量。
    """
    from MDAnalysis.lib.distances import capped_distance, self_capped_distance

    rmax = float(edges[-1])
    hist = np.zeros(edges.size - 1, dtype=float)
    #: mode="all" 时一次遍历就给出三条直方图（total / inter / intra），
    #: 配对筛选只是布尔掩码，所以**耗时与单模式相同**。
    hist_all = {"total": np.zeros(edges.size - 1, dtype=float),
                "inter": np.zeros(edges.size - 1, dtype=float),
                "intra": np.zeros(edges.size - 1, dtype=float)}
    na, nb = ref.n_atoms, conf.n_atoms
    qc = {"有效原子对数": 0, "统计帧数": 0, "分子内配对占比": 0.0,
          "模式": str(mode), "模式退化": "",
          # 归一化用：逐帧累加 1/V 与最小盒边长（NPT 盒波动必须逐帧计入）
          "体积倒数和": 0.0, "n_体积": 0, "最小盒边长": float("inf")}
    if na == 0 or nb == 0:
        return hist, 0, qc

    ma = _molnums_for_filter(ref, force_bonds=mol_from_bonds)
    mb = ma if same else _molnums_for_filter(conf, force_bonds=mol_from_bonds)
    eff_mode = str(mode)
    if eff_mode in ("inter", "intra") and (ma is None or mb is None):
        eff_mode = "total"                      # 没有分子信息 → 只能给总体
        qc["模式退化"] = "体系无 molnums，已退化为 total"

    n_used = 0
    n_pairs = 0
    n_intra = 0
    from ..cancel import check as _cancel_check

    #: 成键对查表结构（帧间不变，跨帧复用 —— 见 _bonded_state 的说明）
    bond_cache: dict = {}

    for idx in frame_indices:
        _cancel_check()
        u.trajectory[int(idx)]
        pa = np.asarray(ref.positions, dtype=float)
        pb = None if same else np.asarray(conf.positions, dtype=float)
        box = u.dimensions
        # Forcite 口径（`wrap_inbox=True`）：统计前把坐标折回主盒。MS 的 .trj
        # 坐标是**展开**的，多组分体系里不同分子的原子经最小镜像会在 0.3~0.5 Å
        # 处产生假近距离（实测 PE50 的 eth 组分前几箱 g≈9，而 Forcite 为 0）。
        # **默认关**，以保持与 MDAnalysis InterRDF（假设盒内坐标）一致的现有行为。
        if wrap_inbox and box is not None:
            try:
                m_box = box_matrix(u)
                inv_b = np.linalg.inv(m_box)
                pa = (pa @ inv_b) % 1.0 @ m_box
                if pb is not None:
                    pb = (pb @ inv_b) % 1.0 @ m_box
            except Exception:                             # noqa: BLE001
                pass
        # 距离一律用**周期最小镜像**（与 MDAnalysis InterRDF / Forcite 同口径，
        # selftest 有对账）。此前试过"坐标不在盒内就不用周期"的启发式：它能让
        # PE50 的分子间短程归零，但会把"坐标恰好不在盒内、却本就该周期统计"的
        # 体系（selftest 的水盒子、GROMACS 轨迹）算错，故不采用。
        qc["坐标前处理"] = ("周期最小镜像" if str(distance) == "minimage"
                            else "无周期（簇模式原始距离）")
        box_d = box if str(distance) == "minimage" else None
        if str(distance) != "minimage":
            qc["距离口径"] = "cluster"
        if box is not None:
            try:
                vol = box_volume_from_dims(box)
                if vol > 0:
                    qc["体积倒数和"] += 1.0 / vol
                    qc["n_体积"] += 1
                qc["最小盒边长"] = min(float(qc["最小盒边长"]),
                                       box_min_length_from_dims(box))
            except Exception:  # noqa: BLE001
                pass
        else:
            # ---- 没有盒信息（非周期体系 / 只带坐标的轨迹）----
            # 距离按**无周期性**算（capped_distance(box=None)），密度归一化用的
            # 体积取这一帧坐标的**包围盒体积**。这是对"有限体系"最自然的近似：
            # g(r) 的峰位/峰形仍然有效，长程与绝对值只作定性参考。
            pts = pa if pb is None else np.vstack([pa, pb])
            if pts.size:
                ext = np.ptp(pts, axis=0)
                ext = np.maximum(ext, 1e-6)
                vol = float(np.prod(ext))
                if vol > 0:
                    qc["体积倒数和"] += 1.0 / vol
                    qc["n_体积"] += 1
                    qc["最小盒边长"] = min(float(qc["最小盒边长"]), float(ext.min()))
            qc["无盒帧"] = int(qc.get("无盒帧", 0)) + 1
        if same:
            try:
                # 同组：走 mdta.neighbors（正交盒/无盒用 cKDTree，快约 2×；
                # 三斜盒自动退回 MDAnalysis 的 cell list）
                from ..neighbors import self_pairs as _self_pairs

                pairs, dist = _self_pairs(pa, box_d, float(rmax))
            except Exception as exc:                          # noqa: BLE001
                # 网格搜索要求 cutoff < 半盒边；Forcite 允许更大 cutoff（用户实测 20 Å）
                if "too large for box" not in str(exc):
                    raise
                pairs, dist = _pairs_bruteforce(pa, pa, rmax, box_d, same=True)
                qc["大cutoff兜底帧"] = int(qc.get("大cutoff兜底帧", 0)) + 1
            mc = ma
        else:
            try:
                from ..neighbors import cross_pairs as _cross_pairs

                pairs, dist = _cross_pairs(pa, pb, box_d, float(rmax))
            except Exception as exc:                          # noqa: BLE001
                if "too large for box" not in str(exc):
                    raise
                pairs, dist = _pairs_bruteforce(pa, pb, rmax, box_d, same=False)
                qc["大cutoff兜底帧"] = int(qc.get("大cutoff兜底帧", 0)) + 1
            mc = mb
        if pairs.size:
            keep = np.ones(dist.size, dtype=bool)
            if exclude_bonded:
                keep &= _drop_bonded_pairs(ref, conf, pairs, dist, same,
                                           return_mask=True, cache=bond_cache)
            masks = {"total": keep}
            if mc is not None:
                mi = ma[np.asarray(pairs[:, 0], dtype=np.int64)]
                mj = (ma if same else mb)[np.asarray(pairs[:, 1], dtype=np.int64)]
                same_mol = mi == mj
                masks["inter"] = keep & ~same_mol
                masks["intra"] = keep & same_mol
                # ---- Forcite 口径：剔除「伪重叠」分子对 --------------------
                # MS 的 .trj 坐标是展开的，两个分子的原子经最小镜像可能落到
                # 0.2~1.5 Å（物理上不可能）。Forcite 的 RDF 里这些对不出现
                # （实测 PE50 的 eth 分子间 g(r) 在 r<1.7 Å 严格为 0），
                # 用法是把**存在伪近距离（< drop_overlap）的整对分子**从
                # 分子间统计里剔除（只剔除该距离以下的原子对会把整对分子的
                # 其它配对仍然留在曲线里，与 Forcite 形状不符）。
                if drop_overlap and drop_overlap > 0 and np.any(same_mol == False):  # noqa: E712
                    _bad = masks["inter"] & (dist < float(drop_overlap))
                    if _bad.any():
                        _nmax = int(max(mi.max(initial=0), mj.max(initial=0))) + 1
                        _key = mi * _nmax + mj
                        _bkey = np.unique(_key[_bad])
                        _drop = np.isin(_key, _bkey) & masks["inter"]
                        if _drop.any():
                            masks["inter"] &= ~_drop
                            masks["total"] &= ~_drop
                            qc["伪重叠原子对"] = (int(qc.get("伪重叠原子对", 0))
                                              + int(_drop.sum()))
                # 统计口径跟着**实际模式**走：inter 时分子内配对占比必须是 0，
                # intra 时是 1，total 时才是真实比例（qc 里要如实反映）。
                _sel = masks.get(eff_mode, masks["total"])
                n_intra += int(np.sum(same_mol[_sel]))
            n_pairs += int(keep.sum())
            # ⚠️ 极近距离对（严重重叠/镜像残留）必须剔除：它会把第 0 箱顶到几十上百。
            # 真实成键距离 ≥0.7 Å，< 0.3 Å 只可能是数值/镜像异常。
            for k, m in masks.items():
                dsel = dist[m]
                if not dsel.size:
                    continue
                tiny = dsel < _RDF_MIN_DIST
                if tiny.any():
                    qc["异常近距离对数"] = (int(qc.get("异常近距离对数", 0))
                                           + int(tiny.sum()))
                    dsel = dsel[~tiny]
                if dsel.size:
                    h, _ = np.histogram(dsel, bins=edges)
                    hist_all[k] += h * (2.0 if same else 1.0)
                    if k == eff_mode:
                        hist += h * (2.0 if same else 1.0)
        n_used += 1
    qc["有效原子对数"] = n_pairs
    qc["统计帧数"] = n_used
    qc["分子内配对占比"] = float(n_intra / n_pairs) if n_pairs else 0.0
    if str(mode) == "all":
        return hist_all, n_used, qc
    return hist, n_used, qc


def _bonded_state(ref, conf, same, cache: dict | None = None):
    """预计算"成键对"的快速查找结构（**帧间不变，可跨帧复用**）。

    返回 ``(lut_r, lut_c, nb, probe)``，其中 ``probe`` 是可直接索引的判定函数：
    大体系用 ``searchsorted``（O(log B)），小体系用**布尔位图**（O(1) 查表）。

    为什么要它：``_drop_bonded_pairs`` 以前每帧都用 ``np.isin`` 把上百万个
    ``min*N+max`` 编码 key 拿去和键表比对（``np.isin`` 内部要排序整条数组），
    实测 20PE30 上 rmax=12 Å、126 万对时**单帧 112 ms**，而真正的直方图只要 8 ms。
    键表在一条轨迹上是不变的，所以这套映射与 key 集合只该算一次。
    """
    key = None
    try:
        b = ref.bonds
        if b is None or len(b) == 0:
            return None
        key = (int(ref.universe.atoms.n_atoms), int(len(b)), int(ref.n_atoms),
               int(conf.n_atoms), bool(same))
        if cache is not None and key in cache:
            return cache[key]
        n_u = int(ref.universe.atoms.n_atoms)
        lut_r = np.full(n_u, -1, dtype=np.int64)
        lut_r[np.asarray(ref.indices, dtype=np.int64)] = np.arange(
            ref.n_atoms, dtype=np.int64)
        lut_c = lut_r if same else np.full(n_u, -1, dtype=np.int64)
        if not same:
            lut_c[np.asarray(conf.indices, dtype=np.int64)] = np.arange(
                conf.n_atoms, dtype=np.int64)
        nb = max(int(conf.n_atoms), int(ref.n_atoms), 1)
        bidx = np.asarray(ref.bonds.indices, dtype=np.int64).reshape(-1, 2)
        bi = lut_r[bidx[:, 0]]
        bj = lut_c[bidx[:, 1]]
        m = (bi >= 0) & (bj >= 0)
        lo = np.minimum(bi[m], bj[m])
        hi = np.maximum(bi[m], bj[m])
        bonded = lo * nb + hi
        cap = int(nb) * int(nb)
        if bonded.size and cap <= 60_000_000:              # ≤ 60 MB 位图 → 直接查表
            bitmap = np.zeros(cap, dtype=bool)
            bitmap[bonded] = True

            def probe(keys, _bm=bitmap, _cap=cap):
                out = np.zeros(keys.size, dtype=bool)
                ok = (keys >= 0) & (keys < _cap)
                if ok.any():
                    out[ok] = _bm[keys[ok]]
                return out
        else:                                              # 大体系：排序 + searchsorted
            srt = np.sort(np.unique(bonded))

            def probe(keys, _srt=srt):
                if _srt.size == 0:
                    return np.zeros(keys.size, dtype=bool)
                pos = np.searchsorted(_srt, keys)
                pos = np.clip(pos, 0, _srt.size - 1)
                return _srt[pos] == keys
        state = (lut_r, lut_c, nb, probe)
        if cache is not None:
            cache[key] = state
        return state
    except Exception:                                      # noqa: BLE001
        return None


def _drop_bonded_pairs(ref, conf, pairs, dist, same, depth: int = 1,
                       return_mask: bool = False, cache: dict | None = None):
    """剔除键连（1-2）原子对，避免分子内成键原子污染 RDF。

    用 numpy 向量化实现（原子对数量可达百万级，纯 Python 循环会慢几个数量级）。
    每一对原子映射到局部索引后用 ``min*N+max`` 编码成整型 key，再与键表编码出的
    key 集合比对 —— 比对用的是 :func:`_bonded_state` 预计算的查找结构
    （位图或 ``searchsorted``）；``cache`` 传一个 dict 时，映射与 key 集合
    **只算一次、逐帧复用**（键表在一条轨迹上不变）。

    ``return_mask=True`` 时返回布尔掩码而不是过滤后的距离——调用方要在同一批
    原子上叠加别的过滤条件（例如分子间/分子内模式）时必须用掩码，否则两次
    过滤会因索引错位而互相污染。
    """
    def _ret(mask):
        return mask if return_mask else dist[mask]

    try:
        if pairs.size == 0:
            return _ret(np.ones(dist.size, dtype=bool))
        state = _bonded_state(ref, conf, same, cache)
        if state is None:
            return _ret(np.ones(dist.size, dtype=bool))
        lut_r, lut_c, nb, probe = state
        li = lut_r[np.asarray(pairs[:, 0], dtype=np.int64)]
        lj = lut_c[np.asarray(pairs[:, 1], dtype=np.int64)]
        valid = (li >= 0) & (lj >= 0)
        keep = np.ones(dist.size, dtype=bool)
        if valid.any():
            lo = np.minimum(li[valid], lj[valid])
            hi = np.maximum(li[valid], lj[valid])
            is_bonded = np.zeros(dist.size, dtype=bool)
            is_bonded[valid] = probe(lo * nb + hi)
            keep = ~is_bonded
        return _ret(keep)
    except Exception:  # noqa: BLE001
        return _ret(np.ones(dist.size, dtype=bool))


@register("rdf", "径向分布函数 RDF")
def analyze_rdf(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                pairs: Sequence[tuple[str, str]] | None = None,
                rmax: float = 12.0, nbins: int = 120,
                exclude_bonded: bool = False,
                mode: str = "inter",
                forcite: bool = False,
                wrap_inbox: bool = False, additive: bool = False,
                mol_from_bonds: bool = False,
                distance: str = "minimage", drop_overlap: float = 0.0,
                rho_scale: float = 1.0,
                compare_halves: bool = False,
                max_group_atoms: int = 6000,
                verbose: bool = False) -> AnalysisResult:
    """径向分布函数分析（设计大纲第 12、15.2 章）。

    参数
    ----
    groups
        ``{组分名: AtomGroup}``，例如 ``{"A": ag_a, "B": ag_b}``。
    pairs
        需要计算的原子对，例如 ``[("A","A"), ("A","B")]``；
        默认计算所有组合（含自身配对）。
    rmax, nbins
        最大距离与 bin 数。
    exclude_bonded
        是否剔除成键原子对（分子内 RDF 通常需要剔除 1-2 对）。
        这是**独立开关**，与 ``mode`` 无关：先按模式筛配对，再按键表剔 1-2 对。
    mode
        配对模式，取值 ``"inter"``（默认，只统计**不同分子**之间的配对）、
        ``"intra"``（只统计同一分子内部的配对）、``"total"``（两者都含，
        即 1.0.0 的旧行为）。判定依据是 ``molnums``。
        **三种模式各自使用自己的真实有序原子对数做归一化**——这是关键：
        若都用 ``N_a·N_b``，``g(r)`` 与由它积分的配位数都会系统性偏移。
        体系缺少 ``molnums`` 时自动退化为 ``total`` 并给出说明。
    compare_halves
        是否额外计算"模拟前半段 vs 后半段"的 RDF，用于观察结构有序化趋势。
    max_group_atoms
        单个组分的原子数上限。超过时**等间隔抽取代表原子**再做 RDF：
        ``g(r)`` 本身是按密度归一化的，用子集不改变它的物理含义，只增加噪声，
        但计算量按原子数平方下降（17202 原子的自 RDF 要算 ~3 亿对，非常慢）。
    """
    u = mdt.universe
    names = list(groups)
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i:]]
    edges = np.linspace(0.0, float(rmax), int(nbins) + 1)
    centers = bin_edges_to_centers(edges)
    # 兜底体积：所选帧完全没有盒信息时才用（非周期体系）。
    # 以前这里直接 `box_matrix(u)`，没有盒就抛 ValueError —— 一条没有晶胞的轨迹
    # 连 RDF 都跑不了（用户实测：自定义集合跑 RDF 报"当前帧没有模拟盒信息"）。
    # 现在：没有盒就按**逐帧坐标包围盒**当体积、距离按**无周期性**算（见
    # `_rdf_accumulate`），并在结果说明里写清楚这是近似。
    try:
        vbox_fallback = float(abs(np.linalg.det(box_matrix(u))))
    except BoxMissing:
        vbox_fallback = 0.0
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    # 大组分抽稀（g(r) 是密度归一化的量，等间隔取子集不改变其数值，只降噪 + 大幅提速）
    subsampled: dict[str, tuple[int, int]] = {}
    n_orig = {nm: int(groups[nm].n_atoms) for nm in names if groups[nm] is not None}
    if max_group_atoms and max_group_atoms > 0:
        for nm in names:
            ag = groups[nm]
            if ag is not None and ag.n_atoms > max_group_atoms:
                sub, n0, n1, how = _subsample_group(ag, int(max_group_atoms))
                groups = dict(groups)
                groups[nm] = sub
                subsampled[nm] = (n0, n1, how)

    idx = np.asarray(selection.indices, dtype=int)
    res = AnalysisResult(
        name="rdf",
        title="径向分布函数 g(r)",
        meta={"rmax": float(rmax), "nbins": int(nbins), "mode": str(mode),
              "n_frames": int(idx.size), "pairs": [f"{a}-{b}" for a, b in pairs]},
    )
    res.panels = [
        Panel(xlabel="距离 r (Å)", ylabel="径向分布函数 g(r)", title="RDF"),
        Panel(xlabel="距离 r (Å)", ylabel="径向分布函数 g(r)",
              title="模拟前半段 vs 后半段（结构有序化趋势）"),
    ]

    res.add_notes(
        "与 MDAnalysis `InterRDF` 对账时注意三点（同一批原子 / 同样 rmax,nbins 时）："
        "(1) 本工具默认 `mode=inter`，会**剔除同一分子内的原子对**；"
        "要和 InterRDF 比请把「配对模式」切到 `total`（= 1.0.0 的旧行为）。"
        "(2) 本工具把 <0.3 Å 的极近距离（自配对 / 镜像残留）当作异常剔除，"
        "所以第 0 箱是 0；InterRDF 会把**自配对**留在第 0 箱（数值极大）。"
        "(3) 归一化：本工具按 **N(N−1)**（同种原子对只数一次），InterRDF 按 **N²**"
        "（含自配对），所以同一条曲线 `g_ours ≈ g_InterRDF × (1 + 1/N)`"
        "（N=164 时约 0.6%）。这三条都能在「说明」里看到，不是数值错误。")

    # rmax 安全性：按所选帧的**最小**盒边长判断（逐帧在循环里收集），
    # 不再用"调用时轨迹停在哪一帧"的盒边长——那会随轨迹位置飘。
    min_blen_seen = float("inf")
    vbox_last = vbox_fallback

    halves = {}
    degraded_modes: set[str] = set()
    auto_intra: set[str] = set()
    n_boxless = 0                    # 有多少帧没有盒信息（见下方说明）
    if compare_halves and idx.size >= 2:
        half = idx.size // 2
        halves = {"前半段": idx[:half], "后半段": idx[half:]}

    # Forcite 口径开关的可见记录（两条分支共用）
    res.summary["RDF 距离口径"] = ("周期最小镜像" if str(distance) == "minimage"
                                 else "无周期（簇模式，原始坐标距离）")
    if float(drop_overlap or 0.0) > 0:
        res.summary["RDF 伪重叠剔除阈值 (Å)"] = float(drop_overlap)
        res.add_notes(
            f"已启用「伪重叠分子对剔除」（阈值 {float(drop_overlap):g} Å）："
            "MS 的 .trj 坐标是展开的，两个分子的原子经最小镜像可能落到 0.2~1.5 Å"
            "（物理上不可能，Forcite 的分子间 RDF 里也不出现）。这里把"
            "**存在该阈值以下原子对的整对分子**从分子间统计中剔除，"
            "与 Forcite 的分子间 g(r) 口径一致。默认关闭（与 MDAnalysis "
            "InterRDF 一致）。\n"
            "  注：Forcite 的帮助只说明「与周期镜像的相互作用会纳入计算，"
            "且 cutoff 大于最短格矢时会出现对应格矢的峰」，没说它是否做"
            "逐原子最小镜像；cutoff < 最短格矢时 periodic 勾/不勾本来就一样，"
            "不能拿这一点反推它的折叠方式。")
    if float(rho_scale or 1.0) != 1.0:
        res.summary["RDF 密度因子"] = float(rho_scale)
        res.add_notes(
            f"RDF 归一化密度乘了因子 {float(rho_scale):g}（g 除以它，"
            f"即放大 {1.0 / float(rho_scale):.3g} 倍）：Forcite 的分子间 g(r) "
            "与我们的「无周期簇模式」曲线形状逐箱一致、只差一个密度常数 —— "
            "实测 PE50 的 eth 为 2.95 倍（归一化体积口径不同）。"
            "要逐箱对上 Forcite，用 distance=cluster + 密度因子≈0.339。")

    for a, b in pairs:
        ga, gb = groups[a], groups[b]
        same = (ga.n_atoms == gb.n_atoms
                and np.array_equal(np.asarray(ga.indices), np.asarray(gb.indices)))
        if ga.n_atoms == 0 or gb.n_atoms == 0:
            continue
        # 先算三种模式的配对数，据此确定**实际使用**的模式，再累积直方图：
        # 这样"分子间配对恒为空"的退化组（例如单条蛋白链的 P–P）不会白算一遍。
        counts, why = mode_pair_counts(ga, gb, same)
        eff_mode = str(mode)
        if counts is None:
            eff_mode = "total"          # 没有分子信息：退回旧口径
        elif (eff_mode == "inter" and counts["inter"] <= 0 and counts["intra"] > 0):
            # 该组分的原子全在同一个分子里（分子数 = 1），分子间 RDF 恒为空。
            # 直接给空曲线或 0 都没有物理意义，因此自动改用分子内口径并写明。
            eff_mode = "intra"
            auto_intra.add(f"{a}-{b}")
        hist, n_used, qc = _rdf_accumulate(
            u, ga, gb, edges, idx, same, exclude_bonded=exclude_bonded,
            mode=eff_mode, verbose=verbose,
            wrap_inbox=bool(forcite or wrap_inbox),
            mol_from_bonds=bool(forcite or mol_from_bonds),
            distance=str(distance), drop_overlap=float(drop_overlap or 0.0))
        if n_used == 0:
            continue
        # mode="all"：一次遍历已经拿到 total/inter/intra 三条直方图，这里分别用
        # 各自的**真实有序原子对数**归一化，画在同一张 RDF 图上（耗时与单模式相同）。
        if isinstance(hist, dict):
            _siv = float(qc.get("体积倒数和", 0.0) or 0.0)
            for _m in ("total", "inter", "intra"):
                _h = hist.get(_m)
                if _h is None or not _h.size:
                    continue
                _neff = (float(counts.get(_m, counts["total"])) if counts is not None
                         else float(ga.n_atoms) * float(gb.n_atoms)
                         - (float(ga.n_atoms) if same else 0.0))
                if _neff <= 0 or not np.any(_h):
                    continue
                _nm = 1
                if _m == "intra":
                    try:
                        _mm = _molnums_for_filter(ga)
                        _nm = (max(1, int(np.unique(_mm).size))
                               if _mm is not None else 1)
                    except Exception:                         # noqa: BLE001
                        _nm = 1
                if _siv > 0:
                    _ideal = _neff * shell * _siv * _nm
                else:
                    _ideal = float(n_used) * _neff * shell / max(vbox_fallback, 1e-12) * _nm
                # 密度因子（见单模式分支说明）：rho_scale<1 → 归一化密度更小 → g 更大。
                # ⚠️ 只作用于 inter / total：Forcite 的**分子内**曲线与我们的
                # 逐箱一致（本就按"每个分子一组配对"归一化），再乘因子反而偏大。
                if float(rho_scale or 1.0) != 1.0 and _m == "inter":
                    _ideal = _ideal * float(rho_scale)
                _ideal = np.asarray(_ideal, dtype=float)
                if not np.any(_ideal > 0):
                    continue
                _g = np.divide(_h, _ideal, out=np.zeros_like(_h), where=_ideal > 0)
                _lab = f"{a}-{b} ({_m})"
                res.add_curve(_lab, centers, _g, panel=0, group=f"{a}-{b}")
                res.summary[f"{_lab} 归一化分母 (有序原子对)"] = _neff
                res.summary[f"{_lab} g(rmax)"] = float(
                    np.nanmean(_g[-max(3, _g.size // 20):]))
                _pk = _first_peak(centers, _g)
                if _pk and _pk["height"] >= 1.2 and _pk.get("minimum_position"):
                    _vb = (float(n_used) / _siv) if _siv > 0 else vbox_fallback
                    _rho = float(n_orig.get(b, gb.n_atoms)) / max(_vb, 1e-12)
                    res.summary[f"{_lab} 第一峰位置 (Å)"] = _pk["position"]
                    res.summary[f"{_lab} 第一峰高度 g_max"] = _pk["height"]
                    res.summary[f"{_lab} 配位数 (第一壳层)"] = _coordination_number(
                        centers, _g, _pk["minimum_position"], _rho)
            continue
        n_boxless = max(n_boxless, int(qc.get("无盒帧", 0)))
        if qc.get("模式退化"):
            degraded_modes.add(str(qc["模式退化"]))
        if counts is None:
            neff = float(ga.n_atoms) * float(gb.n_atoms) - (float(ga.n_atoms) if same else 0.0)
            why = why or "无 molnums"
        else:
            neff = float(counts.get(eff_mode, counts["total"]))
        if neff <= 0:
            continue
        # 归一化：逐帧用该帧的盒体积，等效于对 g(r) 做帧平均（而非先平均成一条
        # 曲线再除以某个固定体积）。NPT 盒波动时这才是可复现且物理正确的做法。
        sum_inv_vol = float(qc.get("体积倒数和", 0.0) or 0.0)
        if sum_inv_vol > 0:
            vbox_eff = float(n_used) / sum_inv_vol
        else:
            vbox_eff = vbox_fallback
        # 归一化：
        # · 默认（additive=False）= **各模式用自己的真实配对数**，这是与
        #   MDAnalysis InterRDF 对得上的口径（total 实测逐点一致）；
        # · additive=True（Forcite 口径）= 三种模式**共用 total 的分母**，
        #   于是 total = inter + intra（Forcite 三列满足该恒等式，误差 1e-8）。
        _neff_total = (float(counts["total"]) if counts is not None
                       else float(ga.n_atoms) * float(gb.n_atoms)
                       - (float(ga.n_atoms) if same else 0.0))
        if forcite or additive:
            if sum_inv_vol > 0:
                ideal = _neff_total * shell * sum_inv_vol
            else:
                ideal = n_used * _neff_total * shell / vbox_fallback
        elif eff_mode == "intra":
            n_mol_set = 1
            try:
                mm = _molnums_for_filter(ga)
                if mm is not None:
                    n_mol_set = max(1, int(np.unique(mm).size))
            except Exception:                                 # noqa: BLE001
                pass
            _den = (sum_inv_vol if sum_inv_vol > 0
                    else float(n_used) / vbox_fallback)
            ideal = neff * shell * n_mol_set * _den
            res.summary[f"{a}-{b} 分子内按分子数归一"] = n_mol_set
        elif sum_inv_vol > 0:
            ideal = neff * shell * sum_inv_vol
        else:
            ideal = n_used * neff * shell / vbox_fallback
        # 密度因子 rho_scale：归一化密度 ρ_eff = rho_scale·ρ(N/V_box)，
        # 于是 g 放大 1/rho_scale 倍。
        # 实测（PE50 eth，xsd 参考结构，xsd_rdf_periodic.csv）：
        # Forcite 的分子间 RDF = 我们的「无周期簇模式」曲线 × 2.946 —— 形状逐箱
        # 一致（mean|Δ| 从 1.955 降到 0.106），只差一个密度常数，说明 Forcite
        # 归一化用的密度比 N/V_box 小约 2.95 倍（体积口径不同，等效体积 ≈
        # V_box/2.95 ≈ 85,400 Å³）。所以对齐 Forcite 用 distance=cluster +
        # rho_scale≈0.339。
        # 分子内（intra）不乘：它的归一化本来就按"每分子一组配对"做的，
        # Forcite 的 intra 与我们的逐箱一致（76.2 vs 76.8 @1.15 Å）。
        if float(rho_scale or 1.0) != 1.0 and eff_mode == "inter":
            ideal = ideal * float(rho_scale)
        min_blen_seen = min(min_blen_seen, float(qc.get("最小盒边长", float("inf"))))
        vbox_last = vbox_eff
        g = np.divide(hist, ideal, out=np.zeros_like(hist), where=ideal > 0)

        label = f"{a}-{b}"
        res.add_curve(label, centers, g, panel=0, group=label)
        # QC：把"这个数是谁的数"写清楚——模式、分母、以及分子内配对的实际占比
        res.summary[f"{label} 配对模式"] = eff_mode
        res.summary[f"{label} 归一化分母 (有序原子对)"] = neff
        res.summary[f"{label} 有效原子对数/帧"] = (
            float(qc.get("有效原子对数", 0)) / n_used if n_used else 0.0)
        res.summary[f"{label} 分子内配对占比"] = float(qc.get("分子内配对占比", 0.0))
        # 数密度（Å⁻³）：用于把 RDF 积分成第一壳层配位数。
        # 必须用**原始**原子数——抽稀只影响 g(r) 的统计噪声，不该改变配位数；
        # 若用子集原子数，配位数会被整体缩小 step 倍。
        # 体积同样用逐帧平均体积，与 g(r) 的归一化保持一致。
        rho_b = float(n_orig.get(b, gb.n_atoms)) / vbox_eff
        pk = _first_peak(centers, g)
        if pk:
            # 结构峰的显著性判据：g 必须明显高于 1（相对均匀分布有富集）。
            # 没有这条判据时，曲线上的任意小起伏都会被当成"第一峰"，
            # 进而积分出一个毫无物理含义的配位数——例如 46 体系的 DPE–DPE
            # 分子间 g(r)（该组分较稀，r<3 Å 根本没有结构峰），旧代码会报出
            # "第一峰 1.75 Å、配位数 0.0077"这种数。
            sig = float(pk["height"]) >= 1.2
            res.summary[f"{label} 第一峰位置 (Å)"] = pk["position"]
            res.summary[f"{label} 第一峰高度 g_max"] = pk["height"]
            res.summary[f"{label} 第一峰是否显著"] = "是" if sig else "否"
            if pk.get("minimum_position"):
                res.summary[f"{label} 第一极小位置 (Å)"] = pk["minimum_position"]
            if sig:
                cn = _coordination_number(centers, g, pk["minimum_position"], rho_b)
                res.summary[f"{label} 配位数 (第一壳层)"] = cn
            else:
                res.add_notes(
                    f"{label} 在 r ≤ {rmax:g} Å 内**没有显著的结构峰**"
                    f"（最大 g 仅 {pk['height']:.2f}，未明显高于 1）："
                    f"此时的「第一峰位置/第一极小」只是曲线的局部起伏，由它积分出的"
                    f"配位数没有物理含义，因此不给该数。常见原因是该组分在体系中被"
                    f"稀释，或 rmax 尚未覆盖它的关联距离。")
            if sig and pk["position"] < 1.7 and not exclude_bonded:
                res.add_notes(
                    f"{label} 的第一峰位于 {pk['position']:.2f} Å，属于**成键原子对**的"
                    f"距离尺度（1-2 对），并非局部结构信息。若要研究结构有序性，"
                    f"请设置 exclude_bonded=True 剔除成键原子对。"
                )

        # 大 r 端是否已收敛到 1（未收敛说明 rmax 小于结构关联长度）
        g_tail = float(np.nanmean(g[-max(3, g.size // 20):]))
        res.summary[f"{label} g(rmax)"] = g_tail
        if np.isfinite(g_tail) and not (0.85 < g_tail < 1.15):
            res.add_notes(
                f"{label} 在 r={rmax:g} Å 处的 g = {g_tail:.3f}，尚未收敛到 1："
                f"说明该尺度仍在结构关联范围内（例如球蛋白内部、或两组分体积占比悬殊），"
                f"g(r) 的绝对数值不宜直接与稀溶液文献值比较。"
            )

        for hname, hidx in halves.items():
            h, nu, qc_h = _rdf_accumulate(u, ga, gb, edges, hidx, same,
                                          exclude_bonded=exclude_bonded,
                                          mode=eff_mode,
                                          distance=str(distance),
                                          drop_overlap=float(drop_overlap or 0.0))
            if nu == 0:
                continue
            siv = float(qc_h.get("体积倒数和", 0.0) or 0.0)
            ideal_h = (neff * shell * siv) if siv > 0 else (nu * neff * shell / vbox_fallback)
            gh = np.divide(h, ideal_h, out=np.zeros_like(h), where=ideal_h > 0)
            res.add_curve(f"{label} ({hname})", centers, gh, panel=1, group=label)
            pkh = _first_peak(centers, gh)
            if pkh:
                res.summary[f"{label} {hname}第一峰高度 g_max"] = pkh["height"]

    if np.isfinite(min_blen_seen) and rmax > 0.5 * min_blen_seen:
        res.add_notes(
            f"注意：rmax={rmax:g} Å 超过所选帧中最小盒边长的一半 "
            f"({0.5 * min_blen_seen:.2f} Å)，最小镜像近似可能失效，建议减小 rmax。")
    # 没有盒信息的体系：说清楚做了什么近似（否则用户会拿 g(r) 的绝对值去比文献）
    if n_boxless > 0:
        res.add_notes(
            f"⚠️ 这条轨迹**没有晶胞（盒）信息**（{n_boxless} 个所选帧）：距离按"
            f"**无周期性**计算（不做最小镜像），密度归一化用**逐帧坐标包围盒体积**。"
            f"峰位/峰形仍然有效，但 g(r) 的绝对值与长程部分只作定性参考，"
            f"配位数同理。要对文献做定量比较，请换带盒的轨迹"
            f"（.trj 转换后自带盒；或 .tpr + .xtc / .gro）。")
    res.summary["帧平均盒体积 (Å³)"] = vbox_last
    for nm, (n0, n1, how) in subsampled.items():
        res.add_notes(
            f"组分 {nm} 原子数 {n0:,} 较多，RDF 使用抽稀后的 {n1:,} 个代表原子"
            f"（{how}）。g(r) 按密度归一化，子集只增加少量统计噪声，不改变其数值；"
            f"配位数仍按原始原子数的数密度计算。"
        )
    _mode_note = {
        "inter": "只统计**不同分子之间**的原子对（同分子配对已剔除）。"
                 "这是 g(r) 的标准口径：它描述的是分子间的空间关联。",
        "intra": "只统计**同一分子内部**的原子对，反映分子内构象分布"
                 "（分子间结构与它无关）。",
        "total": "同时包含分子内与分子间配对（1.0.0 旧口径）。分子内配对会把"
                 "成键/近邻原子的尖峰混进来，若关注分子间结构请改用 inter。",
    }.get(str(mode), "")
    if _mode_note:
        res.add_notes(f"配对模式 = {mode}：{_mode_note}")
    res.add_notes(
        "g(r) 的归一化分母用的是**该模式下的真实有序原子对数**"
        "（inter/intra/total 各不相同），因此三种模式的 g(r) 与配位数不可直接互比；"
        "每种模式的配位数都按原始原子数的数密度积分得到。"
    )
    if auto_intra:
        res.add_notes(
            "以下组分对的所有原子属于**同一个分子**，分子间配对恒为空，"
            "因此已自动改用 intra（分子内）口径：" + "、".join(sorted(auto_intra))
            + "。若要研究分子间结构，请选择含多个分子的组分。")
    if degraded_modes:
        res.add_notes("配对模式未能生效：" + "；".join(sorted(degraded_modes))
                      + "——已按 total 口径计算，结果与 1.0.0 一致。")
    if exclude_bonded:
        res.add_notes("已剔除成键（1-2）原子对（独立于配对模式的开关）。")
    return res


def _first_inplane_peak(x: np.ndarray, g: np.ndarray, *, r_min: float = 2.2,
                        window: float = 25.0, min_g: float = 1.2) -> dict | None:
    """2D 面内 RDF 的**第一壳层**峰：``r_min ≤ r ≤ window`` 内的最大值。

    三种定义都试过，实测对比（跨体系回归 3 个体系的原始曲线）：

    ==========================  ==========================================
    定义                        问题（都是实测踩到的）
    ==========================  ==========================================
    全区间 ``argmax``（原实现）   rmax=70 Å 时远端噪声尖峰会盖过真峰：TOCL
                                 （每叶层 24 个珠子）的"第一峰"落到 43.5 Å
    "先上升后下降"（3D 的判据）  2D 曲线在 2–4 Å 有**排斥洞**（g≈0 且噪声
                                 大），"第一次下降"发生在洞里 → 报出 3–4 Å
                                 的假峰、g 只有 0.02–0.3
    "第一个 g≥1 的局部极大"      会抓到真峰前的**肩峰**：POPGK 的曲线在
                                 5.1 Å 有 g=1.10 的肩，主峰在 8.7 Å(g=1.37)
    ==========================  ==========================================

    因此取"窗口内最大值"：窗口默认 25 Å（第一壳层与第二壳层的合理范围），
    既不会被几十 Å 外的噪声骗走，也只会抓到该窗口内的主峰。显著性另由调用方
    按 ``g_max ≥ min_g`` **且每叶层珠子数够多**判定（珠子少时曲线涨落本来就大，
    实测每叶层 24 个珠子时 g 能到 1.3，那不是可引用的结构峰）。
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    m = (x >= float(r_min)) & (x <= float(window)) & np.isfinite(g)
    if not m.any() or int(m.sum()) < 3:
        return None
    xs, gs = x[m], g[m]
    i = int(np.argmax(gs))
    return {"position": float(xs[i]), "height": float(gs[i]),
            "significant": bool(gs[i] >= float(min_g)),
            "reason": "" if gs[i] >= float(min_g) else
                      f"窗口内最大 g={gs[i]:.2f} < {min_g:g}"}


def _first_peak(r: np.ndarray, g: np.ndarray, r_min: float = 0.8) -> dict | None:
    """找第一峰位置、高度以及第一峰之后的第一极小。

    ``r_min`` 用于跳过近距离区域：默认 0.8 Å；若要只看"堆积峰"（排除键长与
    1-3 距离的影响），可传 2.5 Å。
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    ok = np.isfinite(g)
    if not ok.any() or g.size < 5:
        return None
    # 跳过 r 很小的区域
    start = int(np.argmax(r > float(r_min)))
    gi = g[start:]
    ri = r[start:]
    if gi.size < 3:
        return None
    # 第一峰：出现下降之前的最大值
    k = 1
    while k < gi.size - 1 and gi[k + 1] >= gi[k]:
        k += 1
    while k < gi.size - 1 and gi[k + 1] < gi[k]:
        k += 1
    pidx = int(np.argmax(gi[:k + 1]))
    if gi[pidx] <= 0:
        return None
    # 第一极小：峰之后的最小值
    min_pos = None
    if pidx < gi.size - 2:
        tail = gi[pidx:]
        j = int(np.argmin(tail))
        if j > 0:
            min_pos = float(ri[pidx + j])
    return {"position": float(ri[pidx]), "height": float(gi[pidx]),
            "minimum_position": min_pos}


def _coordination_number(r: np.ndarray, g: np.ndarray, rmin: float | None,
                         number_density: float) -> float:
    """第一壳层配位数：``n = ρ · ∫₀^{r_min} 4πr² g(r) dr``。

    ``number_density`` 是 B 组分的数密度（Å⁻³），缺少它得到的只是
    ``∫4πr²g dr``，不是配位数。
    """
    if rmin is None or not np.isfinite(number_density) or number_density <= 0:
        return float("nan")
    r = np.asarray(r, dtype=float)
    m = r <= rmin
    if m.sum() < 2:
        return float("nan")
    dr = float(np.median(np.diff(r)))
    return float(number_density * np.sum(g[m] * 4.0 * np.pi * r[m] ** 2) * dr)


# ------------------------------------------------------------------ 接触分析
@register("comdist", "两组分质心距")
def analyze_com_distance(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                         pair: tuple | None = None, axis: int = 2, pbc: bool = True,
                         verbose: bool = False) -> AnalysisResult:
    """两个组分的**质心距**随时间（作者 ``COM.py`` 的量：聚合物–膜 z 距离）。

    口径（**1.0.5 起默认用最小镜像**，用户要求）：

    - ``pair`` 默认取**原子数最多**的两个组分（作者：聚合物 vs 膜）；
    - ``axis`` 给"沿该轴的分量"，同时输出总距离；
    - ``pbc=True``（**默认**）：对质心差取**最小镜像** —— 两个质心可能分别落在
      盒边界两侧（膜跨盒、体系被折回），直接相减会得到 ~L 的假跳变；
    - ``pbc=False``：按包裹坐标直接相减，与作者 ``COM.py`` 的
      ``z_poly - z_mem`` 一致（要逐点复现那篇的图时才用）。

    ⚠️ 注意：这是**质心距**，不是界面宽度，也不能用来推"某窄带内的粒子数" ——
    后者必须逐帧用**动态参考**（本函数逐帧取质心正是这个原因；实测膜质心 z 的
    波动 std 可达 5 Å 量级，用时间平均参考会把 ±5 Å 窄带的计数算错一个数量级）。
    """
    names = list(groups.keys())
    if len(names) < 2:
        raise ValueError("质心距需要两个组分")
    if pair is not None:
        for nm in pair:
            if nm not in groups:
                raise ValueError(f"组分 {nm!r} 不存在；可用 {names}")
        a_name, b_name = str(pair[0]), str(pair[1])
    else:
        ordered = sorted(groups.items(), key=lambda kv: -kv[1].n_atoms)
        a_name, b_name = ordered[0][0], ordered[1][0]
    from ..units import time_axis          # 本模块里 time_axis 是局部导入

    ga, gb = groups[a_name], groups[b_name]

    times = np.asarray(selection.times_ps, dtype=float)
    n_f = times.size
    d_tot = np.full(n_f, np.nan)
    d_axis = np.full(n_f, np.nan)
    ca = np.full((n_f, 3), np.nan)
    cb = np.full((n_f, 3), np.nan)
    for k, (_frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pa = np.asarray(ga.center_of_mass(), dtype=float)
        pb = np.asarray(gb.center_of_mass(), dtype=float)
        diff = pa - pb
        if pbc:
            from MDAnalysis.lib.distances import minimize_vectors

            diff = minimize_vectors(diff[None, :], box=mdt.universe.dimensions)[0]
        ca[k], cb[k] = pa, pb
        d_tot[k] = float(np.linalg.norm(diff))
        d_axis[k] = float(diff[int(axis)])

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="comdist",
        title=f"两组分质心距 —— {a_name} vs {b_name}",
        meta={"A": a_name, "B": b_name, "axis": int(axis), "pbc": bool(pbc),
              "n_frames": n_f},
    )
    axn = "xyz"[int(axis)]
    res.panels = [Panel(xlabel=tlabel, ylabel="质心距 (Å)", title="质心距随时间"),
                  Panel(xlabel=tlabel, ylabel=f"沿 {axn} 轴分量 (Å)",
                        title=f"{a_name} − {b_name} 沿 {axn} 轴的有符号距离")]
    res.add_curve(f"{a_name}–{b_name} 距离", tx, d_tot, kind="line", panel=0)
    res.add_curve(f"{a_name}–{b_name} 平均", tx, np.full_like(tx, np.nanmean(d_tot)),
                  kind="line", panel=0)
    res.add_curve(f"沿 {axn} 轴", tx, d_axis, kind="line", panel=1)
    for tag, arr in (("距离", d_tot), (f"沿 {axn} 轴", d_axis)):
        fin = arr[np.isfinite(arr)]
        if fin.size:
            res.summary[f"{tag} 平均 (Å)"] = float(np.mean(fin))
            res.summary[f"{tag} 标准差 (Å)"] = float(np.std(fin))
            res.summary[f"{tag} 最小 (Å)"] = float(np.min(fin))
            res.summary[f"{tag} 最大 (Å)"] = float(np.max(fin))
    res.summary["A 组分"] = a_name
    res.summary["B 组分"] = b_name
    res.summary["质心口径"] = ("最小镜像" if pbc
                              else "包裹坐标直接相减（与作者 COM.py 一致）")
    res.add_notes(
        f"逐帧取 {a_name} 与 {b_name} 的**质心**再作差（质心口径："
        f"{'最小镜像' if pbc else '包裹坐标直接相减'}）。"
        "这是质心距，不是界面宽度；也不能由它推「窄带内粒子数」。")
    res.add_notes(
        "为什么要逐帧取质心：膜/界面的位置本身在涨落（实测某 CG 膜体系膜质心 z 的 "
        "std 达 5.53 Å）。任何以界面为参考的**窄带**统计都必须用**逐帧**参考，"
        "用时间平均参考会把结果算错一个数量级。")
    return res

@register("rdf2d", "面内径向分布 RDF (2D)")
def analyze_rdf_inplane(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                        plane_axis: int = 2, rmax: float = 70.0, nbins: int = 140,
                        slab: tuple | None = None, max_group_atoms: int = 3000,
                        slab_ref=None, slab_side: str = "upper",
                        verbose: bool = False) -> AnalysisResult:
    """膜平面内的 **2D 自 RDF** ``g2D(r_xy)``（膜体系的标准做法，3D g(r) 不适用）。

    与 3D 的区别（照 ``Supporting_Info/Lipid_RDF.py`` 的口径实现）：

    - 距离只取**面内两个分量**（忽略法向），且只对这两个分量做最小镜像；
    - 归一化用**面内面密度** ``rho2D = N/(Lx*Ly)`` 与 **2D 壳面积** ``2*pi*r*dr``
      （而不是 3D 的球壳与体积密度）；
    - ``slab=(lo, hi)`` 只统计某一叶层（固定阈值，用于"膜整体位置不涨落"的场合）；
    - ``slab_ref=<AtomGroup>``：**逐帧**用该组的质心当阈值 —— 作者的做法就是
      ``z > 膜质心(t)``（``Lipid_RDF.py`` 里 ``z_mid = membrane_beads.center_of_mass()``
      在帧循环内取），``slab_side="upper"`` 取上半叶、``"lower"`` 取下半叶。
      ⚠️ 膜质心 z 的涨落 std 实测可达 5 Å 量级，用**固定**阈值（时间平均质心或盒中心）
      会把叶层圈错、归一化直接失真 —— 这就是必须有逐帧模式的原因。

    作者参考实现见 ``Lipid_RDF.py`` 的 ``compute_rdf()``：逐对（i != j）取面内距离、
    直方图计数后除以 ``N * rho2D * 2*pi*r*dr``。本实现向量化等价，并保留他们
    "rho 用整个面内面积"的约定（分母不随叶层厚度变化）。
    """
    times = np.asarray(selection.times_ps, dtype=float)
    n_f = times.size
    ax = int(plane_axis)
    in_ax = [i for i in range(3) if i != ax]
    edges = np.linspace(0.0, float(rmax), int(nbins) + 1)
    centers = bin_edges_to_centers(edges)
    dr = edges[1] - edges[0]
    shell = 2.0 * np.pi * centers * dr
    gsum: dict[str, np.ndarray] = {}
    n_atoms: dict[str, int] = {}
    notes_subsample: list[str] = []
    n_acc = 0
    for _k, (_frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        dims = np.asarray(mdt.universe.dimensions, dtype=float)
        L_in = dims[in_ax]
        A = float(np.prod(L_in))
        # 逐帧叶层阈值：取参考组（膜）的质心，而不是任何固定数值
        band = slab
        if slab_ref is not None:
            z_ref = float(np.asarray(slab_ref.center_of_mass(), dtype=float)[ax])
            band = ((z_ref, float(dims[ax])) if str(slab_side) == "upper"
                    else (0.0, z_ref))
        for name, ag in groups.items():
            allpos = np.asarray(ag.positions, dtype=float)
            if band is not None:
                z = allpos[:, ax]
                allpos = allpos[(z >= float(band[0])) & (z <= float(band[1]))]
            n = allpos.shape[0]
            n_atoms[name] = max(n_atoms.get(name, 0), n)
            if n < 2:
                continue
            if n > int(max_group_atoms):
                # 逐对 O(N²)：大基团按**整分子**抽稀（2D RDF 是密度归一化量，
                # 抽稀只增加噪声、不改峰位与峰高），而不是直接报错 ——
                # 否则界面上"全选"在大体系上会整项失败。
                keep_n = int(max_group_atoms)
                step = max(1, n // keep_n)
                allpos = allpos[::step]
                n = allpos.shape[0]
                note = f"面内 RDF 为逐对统计，{name} 超过上限，已抽稀到 {n} 个原子"
                if note not in notes_subsample:
                    notes_subsample.append(note)
            pos = allpos[:, in_ax]
            # ---- 面内配对：空间划分 + 直方图一次算完（不再展开 N×N 距离矩阵）----
            # 以前 `delta = pos[:,None,:] - pos[None,:,:]`（N=3000 → 144 MB 临时数组、
            # 约 900 ms/帧）。现在见 _inplane_pair_hist：小 cutoff 走 2D 周期 KD 树，
            # 大 cutoff 走分块无序对；返回的直方图已按**有序对**计数，
            # 与旧实现逐位相同（含 _BOND_EXCLUDE 那段 1-2 成键剔除）。
            hist = _inplane_pair_hist(pos, L_in, float(rmax), edges)
            if hist is None:
                continue
            rho = n / A
            g = hist / (n * rho * np.maximum(shell, 1e-12))
            gsum[name] = gsum.get(name, np.zeros_like(g)) + g
        n_acc += 1

    res = AnalysisResult(
        name="rdf2d",
        title=f"面内径向分布 RDF (2D) —— 法向轴 {ax}",
        meta={"plane_axis": ax, "rmax_A": float(rmax), "nbins": int(nbins),
              "n_frames": n_f, "slab_A": list(slab) if slab else None},
    )
    res.panels = [Panel(xlabel="面内距离 r_xy (Å)", ylabel="g2D(r)",
                        title="面内径向分布")]
    if slab is not None:
        res.add_notes(f"仅统计法向位置在 {slab[0]:.1f}-{slab[1]:.1f} Å 之间的原子（叶层）。")
    elif slab_ref is not None:
        res.add_notes(
            f"叶层阈值**逐帧**取参考组的质心（{'上半叶 z > z_ref' if str(slab_side) == 'upper' else '下半叶 z < z_ref'}）。"
            "膜/界面的位置本身在涨落（实测膜质心 z 的 std 可达 5 Å 量级），"
            "固定阈值会把叶层圈错、归一化失真。")
    for name, gs in gsum.items():
        g = gs / max(n_acc, 1)
        res.add_curve(f"{name}-{name}", centers, g, kind="line", panel=0, group=name)
        # ⚠️ 不能对整条曲线取 argmax：rmax 默认 70 Å，珠子少的脂质曲线远端噪声
        # 尖峰（g≈1.1）会盖过真正的第一壳层峰；也不能用 3D 那套"先上升后下降"
        # 判据（2D 曲线在小 r 有排斥洞，会在 3–4 Å 报假峰）。判据见
        # _first_inplane_peak 的文档。显著性额外要求每叶层珠子够多 —— 实测每叶层
        # 二三十个珠子时（TOCL）曲线涨落大到 g 能到 1.4，那不是可引用的结构峰。
        pk = _first_inplane_peak(centers, g, r_min=float(_BOND_EXCLUDE))
        n_leaf = int(n_atoms.get(name, 0))
        if pk is None:
            i = int(np.argmax(np.where(np.isfinite(g), g, -np.inf)))
            pk = {"position": float(centers[i]), "height": float(g[i]),
                  "significant": False, "reason": "曲线没有可用的极大"}
        sig = bool(pk.get("significant")) and n_leaf >= 50
        why = ""
        if not sig:
            why = (pk.get("reason") or "") if n_leaf >= 50 else (
                f"每叶层只有 {n_leaf} 个珠子，涨落过大（<50 时第一峰不可引用）")
        res.summary[f"{name}-{name} 第一峰位置 (Å)"] = float(pk["position"])
        res.summary[f"{name}-{name} 第一峰高度 g_max"] = float(pk["height"])
        res.summary[f"{name}-{name} 第一峰是否显著"] = (
            "是" if sig else f"否（{why}）" if why else "否")
        res.summary[f"{name}-{name} 原子数"] = n_leaf
    res.summary["统计帧数"] = int(n_acc)
    for note in notes_subsample:
        res.add_notes(note)
    res.add_notes(
        "2D 面内 RDF：距离只取面内分量、只对面内做最小镜像；归一化用 "
        "rho2D = N/(Lx*Ly) 与 2D 壳面积 2*pi*r*dr。膜平面内的横向结构必须用这个口径 —— "
        "3D g(r) 会把法向层间关联混进来，对膜体系没有意义。")
    res.add_notes(
        f"已剔除面内距离 < {_BOND_EXCLUDE:g} Å 的配对（1-2 成键）：作者的实现对每个脂质"
        "只取一个头基珠子、不存在分子内配对；若对整个组分统计而不剔除，最近邻就是共价键，"
        "会出现 0.25 Å 量级的假「第一峰」。")
    res.add_notes(
        "「第一峰位置」按**第一壳层**判据给出（跳过近距区、取首次下降之前的极大），"
        "不是整条曲线的最大值 —— 叶层内珠子很少时（实测 TOCL 每叶层约 24 个珠子）"
        "远端噪声尖峰会让「全区间最大值」落到几十 Å 处，那个位置没有物理含义。"
        "「第一峰是否显著」按 g_max ≥ 1.2 判定，与 3D g(r) 用同一阈值。")
    return res


@register("contact", "接触分析")
def shares_one_molecule(ga, gb) -> bool:
    """两组是否**整个落在同一个分子**里（如蛋白与它共价相连的糖基）。

    这是"组分被切成了共价相连的两块"的判据：此时 ``mode="inter"``（只统计
    不同分子之间的原子对）会**一对都不剩**，报出来的"接触数 = 0"是假象 ——
    实测 md_biopolymer_nowater 里 protein×sugar 就是这种情形，最小原子间距
    其实只有 1.39 Å（N-糖苷键）。没有分子编号时返回 ``False``（无从判断）。
    """
    ma, mb = _molnums_for_filter(ga), _molnums_for_filter(gb)
    if ma is None or mb is None:
        return False
    sa, sb = np.unique(ma), np.unique(mb)
    return bool(sa.size == 1 and sb.size == 1 and sa[0] == sb[0])


def analyze_contacts(mdt, group_a, group_b, selection: FrameSelection, *,
                     cutoff: float = 5.0, top_n: int = 0,
                     mode: str = "inter",
                     track_pairs: bool = True,
                     occupancy_max_frames: int = 300,
                     verbose: bool = False) -> AnalysisResult:
    """接触分析（设计大纲第 13 章）。

    定义 ``r < cutoff`` 为发生接触，统计：

    - 接触对数量随时间变化
    - 参与接触的 A 组原子数随时间变化
    - 最小原子间距随时间变化
    - 平均接触数、接触概率（只进**摘要**，不再画逐原子柱图 —— 用户要求）
    - 接触对占据率（持续性，只进摘要）与接触矩阵（A×B 二维热力图）
    - 配位数分布（瞬时接触数的直方图）

    ``top_n`` 是旧版"逐原子柱状图画前 N 个"的参数，逐原子图删掉后不再生效，
    保留仅为兼容已保存的参数。

    同组 vs 异组（重要）
    --------------------
    ``group_a is group_b``（同组，即 A–A 接触）必须与 A–B 分开处理：

    ============  ==============================  ==================================
    情形          配对来源                         配对含义
    ============  ==============================  ==================================
    异组 A ≠ B    ``capped_distance(A, B)``        A×B 笛卡尔积，每对 (i∈A, j∈B) 一次
    同组 A = B    ``self_capped_distance(A)``      无序对，**不含 i==i 自配对**，
                                                   且每对 {i,j} 只记一次
    ============  ==============================  ==================================

    早先不分同组异组、一律用 ``capped_distance(A, A)``，会有三个后果：
    (1) 混进 N 个距离为 0 的 ``i==i`` 自配对，于是**最小原子间距恒为 0**；
    (2) 每对无序对 {i,j} 被记两次（(i,j) 与 (j,i)），接触对数**翻倍**；
    (3) 慢得多 —— ``capped_distance(A,A)`` 实测比 ``self_capped_distance(A)``
        慢 5～7 倍（21384 原子：1.01 s vs 0.186 s 每帧）。
    """
    from MDAnalysis.lib.distances import capped_distance, self_capped_distance

    u = mdt.universe
    na, nb = group_a.n_atoms, group_b.n_atoms
    if na == 0 or nb == 0:
        raise ValueError("接触分析的两个原子组都不能为空")

    # 同组判定：同一个对象，或原子序号完全一致
    same = (group_a is group_b) or (
        na == nb and np.array_equal(np.asarray(group_a.indices),
                                    np.asarray(group_b.indices)))

    # ---------------------------------------------------------- 配对模式（阶段 5）
    # 与 RDF 同一套口径：inter = 只保留不同分子的配对（默认），intra = 只保留同一
    # 分子内部的配对，total = 全要（1.0.0 行为）。判定依据是 molnums。
    # 为什么默认改成 inter：同组分接触（polymer×polymer、protein×protein）里，
    # 同一分子内的相邻原子天然满足 r < cutoff，会把"接触数"抬得虚高，也让
    # "每个原子接触几个"变成"这个原子在自己分子里有几个邻居"。
    ma = _molnums_for_filter(group_a)
    mb = ma if same else _molnums_for_filter(group_b)
    eff_mode = str(mode)
    mode_note = ""
    if eff_mode not in ("inter", "intra", "total"):
        raise ValueError(f"未知的接触配对模式 {mode!r}；可用 inter / intra / total")
    if eff_mode != "total" and (ma is None or mb is None):
        eff_mode = "total"
        mode_note = "体系没有分子编号（molnums），配对模式已退化为 total"
    if eff_mode == "inter" and shares_one_molecule(group_a, group_b):
        # 两组是同一个分子被切成的两块（蛋白 + 共价糖基是典型）：
        # inter 会把**所有**配对都剔除，于是"平均接触对数 = 0"—— 这不是物理
        # 结论，而是口径造成的假象。此时改用 total 并写明，绝不报 0。
        eff_mode = "total"
        mode_note = ("两组的原子落在**同一个分子**里（如蛋白与共价相连的糖基被"
                     "识别成两个组分），分子间配对恒为空；已自动改用 total 口径"
                     "统计，若只想看分子间接触请自行换一组配对")

    times = np.asarray(selection.times_ps, dtype=float)
    n = times.size
    n_pairs = np.full(n, np.nan)
    n_atoms_a = np.full(n, np.nan)
    min_dist = np.full(n, np.nan)
    hit_count = np.zeros(na, dtype=float)
    # 每个 A 原子累计接触到的 B 原子数（逐帧累加）——这是真正有区分度的量：
    # "这个原子平均接触到几个 B 原子"。而"帧比例 > 0"这种概率在稠密体系里恒为 1。
    contact_sum = np.zeros(na, dtype=float)
    #: 瞬时配位数直方图：每个 A 原子在**每一帧**接触到几个 B 原子，对（原子×帧）汇总
    cn_hist: np.ndarray | None = None
    n_intra_pairs = 0
    n_kept_pairs = 0
    min_dist_all = np.inf
    # 接触对占据率：某一对 (i,j) 在多少帧里处于接触
    track = bool(track_pairs) and n <= int(occupancy_max_frames)
    occ_key: np.ndarray | None = None
    occ_cnt: np.ndarray | None = None
    nb_u = int(mdt.universe.atoms.n_atoms)
    # **接触矩阵**（A×B）：原子数可能上万，直接画矩阵既看不清也传不动，
    # 所以按原子序号**分块**统计"每帧每对块之间的平均接触数"。
    # 块数上限 48，保证前端热力图 ≤48×48。
    _nb = 48
    n_ba = int(min(_nb, max(1, na)))
    n_bb = int(min(_nb, max(1, nb)))
    bin_a = np.minimum((np.arange(na) * n_ba) // max(na, 1), n_ba - 1)
    bin_b = np.minimum((np.arange(nb) * n_bb) // max(nb, 1), n_bb - 1)
    mat = np.zeros((n_ba, n_bb), dtype=float)

    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pa = np.asarray(group_a.positions, dtype=float)
        pb = pa if same else np.asarray(group_b.positions, dtype=float)
        if same:
            # 无序对、无自配对 —— 既正确又快得多（走 mdta.neighbors：正交盒 cKDTree）
            from ..neighbors import self_pairs as _self_pairs

            pairs, dist = _self_pairs(pa, u.dimensions, float(cutoff))
        else:
            from ..neighbors import cross_pairs as _cross_pairs

            pairs, dist = _cross_pairs(pa, pb, u.dimensions, float(cutoff))
        if pairs.size and eff_mode != "total" and ma is not None:
            mi = ma[np.asarray(pairs[:, 0], dtype=np.int64)]
            mj = (ma if same else mb)[np.asarray(pairs[:, 1], dtype=np.int64)]
            same_mol = mi == mj
            keep = same_mol if eff_mode == "intra" else ~same_mol
            if not keep.all():
                pairs, dist = pairs[keep], dist[keep]
            n_intra_pairs += int(np.sum(same_mol[keep]))
            n_kept_pairs += int(keep.sum())
        else:
            n_kept_pairs += int(pairs.shape[0])
            # ⚠️ 不能只判 `ma is not None`：`mb` 在异组（same=False）时可能为 None
            # （B 组没有分子信息），下面立刻下标访问就会
            # `TypeError: 'NoneType' object is not subscriptable`（实测 MS 体系踩过）。
            if pairs.size and ma is not None and (same or mb is not None):
                mi = ma[np.asarray(pairs[:, 0], dtype=np.int64)]
                mj = (ma if same else mb)[np.asarray(pairs[:, 1], dtype=np.int64)]
                n_intra_pairs += int(np.sum(mi == mj))
        n_pairs[k] = pairs.shape[0]
        if pairs.size:
            # 同组时两个端点都属于 A，不能只看第 0 列
            uniq = np.unique(pairs) if same else np.unique(pairs[:, 0])
            n_atoms_a[k] = uniq.size
            hit_count[uniq] += 1.0
            if same:
                # 无序对：每个端点各算一次接触
                np.add.at(contact_sum, np.asarray(pairs[:, 0], dtype=np.int64), 1.0)
                np.add.at(contact_sum, np.asarray(pairs[:, 1], dtype=np.int64), 1.0)
            else:
                np.add.at(contact_sum, np.asarray(pairs[:, 0], dtype=np.int64), 1.0)
            # 接触矩阵：把每帧的接触对按块累加（同组时两个端点都在 A，
            # 就对称地算两次，避免只填上三角）
            _pa = bin_a[np.asarray(pairs[:, 0], dtype=np.int64)]
            _pb = bin_b[np.asarray(pairs[:, 1], dtype=np.int64)]
            np.add.at(mat, (_pa, _pb), 1.0)
            if same:
                np.add.at(mat, (_pb, _pa), 1.0)
            # 瞬时配位数（复用同一批配对，零额外距离计算）
            _i0 = np.asarray(pairs[:, 0], dtype=np.int64)
            cnt = np.bincount(_i0, minlength=na)
            if same:
                cnt = cnt + np.bincount(np.asarray(pairs[:, 1], dtype=np.int64),
                                        minlength=na)
            need = int(cnt.max()) + 1
            if cn_hist is None:
                cn_hist = np.zeros(max(8, need), dtype=float)
            elif need > cn_hist.size:
                cn_hist = np.concatenate([cn_hist, np.zeros(need - cn_hist.size)])
            np.add.at(cn_hist, cnt, 1.0)
        else:
            n_atoms_a[k] = 0
        if track and pairs.size:
            gi = np.asarray(group_a.indices, dtype=np.int64)[np.asarray(pairs[:, 0], dtype=np.int64)]
            if same:
                gj = np.asarray(group_a.indices, dtype=np.int64)[
                    np.asarray(pairs[:, 1], dtype=np.int64)]
            else:
                gj = np.asarray(group_b.indices, dtype=np.int64)[
                    np.asarray(pairs[:, 1], dtype=np.int64)]
            lo = np.minimum(gi, gj) * nb_u + np.maximum(gi, gj)
            uk, uc = np.unique(lo, return_counts=True)
            if occ_key is None:
                occ_key, occ_cnt = uk, uc.astype(np.int64)
            else:
                allk = np.concatenate([occ_key, uk])
                allc = np.concatenate([occ_cnt, uc.astype(np.int64)])
                newk, inv = np.unique(allk, return_inverse=True)
                acc = np.zeros(newk.size, dtype=np.int64)
                np.add.at(acc, inv, allc)
                occ_key, occ_cnt = newk, acc
        # 最小间距：**已经有接触时 np.min(dist) 就是全局最小**
        # （cutoff 之外的距离只会更大），无需再放大搜索一遍。
        # 只有"这一刻完全没接触"时才需要 min_distance 去放大截断距离找。
        if dist.size:
            dmin = float(np.min(dist))
        else:
            dmin = min_distance(u, pa, pb, start_cutoff=float(cutoff))
        min_dist[k] = dmin if np.isfinite(dmin) else np.nan
        if np.isfinite(dmin):
            min_dist_all = min(min_dist_all, dmin)

    from ..units import time_axis

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="contact",
        title=f"接触分析（cutoff = {cutoff:g} Å）",
        meta={"cutoff": float(cutoff), "n_atoms_a": int(na), "n_atoms_b": int(nb),
              "n_frames": int(n)},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="接触数", title="接触数随时间变化"),
        Panel(xlabel=tlabel, ylabel="最小原子间距 (Å)", title="最小原子间距随时间变化"),
        # **接触矩阵**（A×B，按原子序号分块）：静态模式下的主力图
        Panel(xlabel="B 组原子（按序号分块）", ylabel="A 组原子（按序号分块）",
              title="接触矩阵（每帧平均接触数）", heatmap=True),
        Panel(xlabel="接触对数", ylabel="频数", title="接触对数分布"),
    ]
    res.add_curve("接触对数", tx, n_pairs, panel=0)
    res.add_curve(f"参与接触的 A 组原子数 (共 {na})", tx, n_atoms_a, panel=0)
    res.add_curve("最小原子间距", tx, min_dist, panel=1)

    if n:
        res.summary["平均接触对数"] = float(np.nanmean(n_pairs))
        res.summary["最大接触对数"] = float(np.nanmax(n_pairs))
        res.summary["平均参与接触原子数"] = float(np.nanmean(n_atoms_a))
        res.summary["接触概率（存在接触的帧比例）"] = float(np.mean(n_pairs > 0))
        res.summary["最小原子间距 (Å)"] = float(min_dist_all) if np.isfinite(min_dist_all) else float("nan")
        res.summary["平均最小原子间距 (Å)"] = float(np.nanmean(min_dist))
        res.summary["配对方式"] = (
            "同组 A–A（无序对，不含 i==i 自配对，每对只记一次）" if same
            else "异组 A–B（A×B 笛卡尔积，每对 (i∈A, j∈B) 记一次）")
    if same:
        res.add_notes(
            f"这是**同组**接触（A 与 B 是同一个原子组，共 {na} 原子）：接触对数按"
            f"无序对统计，不含 i==i 自配对，也没有把 (i,j) 与 (j,i) 记两次。"
        )
        res.add_notes(
            "同组时用 MDAnalysis 的 self_capped_distance（无序、无自配对），"
            "而不是 capped_distance(A, A) —— 后者会混进 N 个距离为 0 的自配对"
            "（最小原子间距恒为 0），把接触对数翻倍，而且实测慢 5～7 倍。"
        )

    # ------------------------------------------------ 单原子：平均接触数（**只进摘要**）
    # 用户要求（1.0.4）：「各原子的平均接触数」「各原子的接触概率」这两张逐原子柱状图
    # 删掉 —— 它们是"上万个原子按接触数排序"的柱图，图上看不出结构，还占大量传参；
    # 有区分度的信息留在摘要里（均值 / 中位数 / 最大值 / 0 接触原子数…）。
    per_atom = contact_sum / float(n) if n else contact_sum
    res.summary["平均接触数（每个 A 组原子）"] = (float(np.mean(per_atom))
                                            if per_atom.size else float("nan"))
    res.summary["单原子平均接触数 中位数"] = (float(np.median(per_atom))
                                        if per_atom.size else float("nan"))
    res.summary["单原子平均接触数 最大值"] = (float(np.max(per_atom))
                                        if per_atom.size else float("nan"))
    res.summary["平均接触数为 0 的 A 组原子数"] = int(np.sum(per_atom == 0))

    # ------------------------------------------------ 单原子：接触概率（**只进摘要**）
    prob = hit_count / float(n) if n else hit_count
    res.summary["原子接触概率平均值"] = float(np.mean(prob)) if prob.size else float("nan")
    res.summary["接触概率 > 0 的 A 组原子数"] = int(np.sum(prob > 0))
    res.summary["完全无接触的 A 组原子数"] = int(np.sum(prob == 0))
    res.summary["接触概率恒为 1 的 A 组原子数"] = int(np.sum(prob >= 1.0))

    # 饱和 QC：「存在接触的帧比例」在稠密体系里恒为 1，没有区分度，必须点明
    any_contact = float(np.mean(n_pairs > 0)) if n else float("nan")
    saturated = bool(n and np.isfinite(any_contact) and any_contact >= 1.0)
    res.summary["接触概率是否饱和"] = "是（恒为 1，无区分度）" if saturated else "否"
    if saturated:
        res.add_notes(
            f"⚠️「接触概率（存在接触的帧比例）」= 1.0：本体系里**每一帧**都至少存在"
            f"一对距离 < {cutoff:g} Å 的 A–B 原子对（本组 {na}×{nb} 个原子对，"
            f"每帧平均 {float(np.nanmean(n_pairs)):,.0f} 对）。这是稠密体系的必然结果，"
            f"不是计算错误，但它对任何体系都恒为 1，**不能用来比较不同体系**。"
            f"要看接触强度请用「平均接触数」（每个 A 原子平均接触到几个 B 原子，"
            f"本体系为 {float(np.mean(per_atom)):.3g}）或下面的「接触对占据率」。")
    if int(np.sum(prob >= 1.0)) == prob.size and prob.size:
        res.add_notes(
            "A 组**全部**原子在每一帧都至少有一次接触（接触概率恒为 1）："
            "这时该指标同样没有区分度，请改看「平均接触数」的分布。")

    # ------------------------------------------------ 接触对占据率（持续性）
    if occ_key is not None and occ_key.size:
        occ = occ_cnt.astype(float) / float(n)
        n_pairs_seen = int(occ.size)
        res.summary["出现过的不同接触对总数"] = n_pairs_seen
        res.summary["接触对占据率平均值"] = float(np.mean(occ))
        res.summary["始终接触（占据率 = 1）的接触对数"] = int(np.sum(occ >= 1.0))
        res.summary["占据率 ≥ 0.5 的接触对数"] = int(np.sum(occ >= 0.5))
        top = np.argsort(-occ)[:3]
        atoms = u.atoms
        for rank, idx in enumerate(top, start=1):
            key = int(occ_key[idx])
            ia, ib = divmod(key, nb_u)
            ra, rb = atoms[int(ia)], atoms[int(ib)]
            # ⚠️ 不是每个拓扑都有 residue 信息：`.xyz` 顶多只有原子名，
            # 甚至二者皆无 —— 直接读 `.resname` 会抛
            # `NoDataError: This Universe does not contain resname information`
            # 把整项接触分析搞崩（实测：无 resname 的合成体系）。这里逐段降级：
            # `resname+resid(name)` → `name` → `#索引`。
            res.summary[f"最持久接触对 #{rank}"] = (
                f"{_atom_label(ra)} – {_atom_label(rb)}"
                f"  占据率 {occ[idx]:.3f}")
        # 「占据率分布」面板已按用户要求删除：占据率本身仍在摘要里
        # （平均值 / 始终接触的接触对数 / 最持久接触对），只是不再单独画分布图。
        res.add_notes(
            f"接触对占据率 = 某一对 (i∈A, j∈B) 处于接触的帧数 / 总帧数，"
            f"共出现 {n_pairs_seen:,} 个不同的接触对，平均占据率 "
            f"{float(np.mean(occ)):.3f}。这个量才是**有区分度的「概率」**："
            f"它区分「始终黏在一起」与「偶发碰到」，而「存在接触的帧比例」做不到。")
    elif track_pairs:
        res.add_notes(
            f"帧数 {n} 超过接触对占据率的统计上限 "
            f"({int(occupancy_max_frames)} 帧)，已跳过该指标以保证速度；"
            f"需要时可减少帧数或调大 occupancy_max_frames。")

    # ------------------------------------------------ 配位数分布（瞬时接触数）
    if cn_hist is not None and cn_hist.sum() > 0:
        cn_vals = np.arange(cn_hist.size, dtype=float)
        cn_p = cn_hist / cn_hist.sum()
        cn_mean = float(np.sum(cn_p * cn_vals))
        cn_std = float(np.sqrt(np.sum(cn_p * (cn_vals - cn_mean) ** 2)))
        pidx = len(res.panels)                      # 面板号动态取，避免与占据率面板冲突
        res.panels.append(Panel(xlabel="瞬时接触数（每个 A 原子每帧）",
                                ylabel="概率",
                                title="配位数分布（瞬时接触数）"))
        res.add_curve("配位数分布", cn_vals, cn_p, kind="bar", panel=pidx)
        res.summary["瞬时接触数 平均"] = cn_mean
        res.summary["瞬时接触数 标准差"] = cn_std
        res.summary["瞬时接触数 众数"] = float(cn_vals[int(np.argmax(cn_hist))])
        res.add_notes(
            "「配位数分布」= 每个 A 原子在**每一帧**的瞬时接触数，对（原子 × 帧）"
            "汇总成直方图。它比「平均接触数」多出**分布宽度**这一维信息：标准差大"
            "说明配位环境高度异质（一部分原子被完全包裹、一部分裸露），"
            "只报平均值会掩盖这种差异。")

    finite = n_pairs[np.isfinite(n_pairs)]
    if finite.size:
        edges = np.linspace(finite.min(), max(finite.max(), finite.min() + 1.0),
                            min(30, max(5, finite.size)) + 1)
        hist, _ = np.histogram(finite, bins=edges)
        res.add_curve("接触对数分布", bin_edges_to_centers(edges), hist.astype(float),
                      kind="bar", panel=3)

    res.add_notes(f"接触判据：任意 A–B 原子对距离 < {cutoff:g} Å。")
    _mode_txt = {
        "inter": "只统计**不同分子之间**的原子对（同分子内的配对已剔除）",
        "intra": "只统计**同一分子内部**的原子对（分子间接触已剔除）",
        "total": "同时包含分子内与分子间配对（1.0.0 旧口径）",
    }[eff_mode]
    res.summary["配对模式"] = eff_mode
    if n_kept_pairs:
        res.summary["分子内配对占比"] = float(n_intra_pairs / n_kept_pairs)
    res.add_notes(f"配对模式 = {eff_mode}：{_mode_txt}。"
                  f"（1.0.0 没有这个开关，等价于 total。）")
    if mode_note:
        res.add_notes(f"注意：{mode_note}。")
    if eff_mode == "inter" and n_kept_pairs == 0 and n_pairs.size:
        res.add_notes(
            "该组分对的全部原子对都属于**同一个分子**，分子间接触恒为空："
            "若要看分子内的接触，请把配对模式改成 intra。")
    # ------------------------------------------------ 接触矩阵（A×B 二维热力图）
    if mat.size and float(mat.sum()) > 0:
        # 每帧平均接触数；块中心用原子序号（1 基）表示"这一段原子"
        z = mat / max(int(n), 1)
        cx = (np.arange(n_bb) + 0.5) * (nb / max(n_bb, 1)) + 0.5
        cy = (np.arange(n_ba) + 0.5) * (na / max(n_ba, 1)) + 0.5
        res.heatmaps.append({
            "panel": 2,                     # 与上面的 heatmap 面板对齐            "title": f"接触矩阵（{n_ba}×{n_bb} 分块，每帧平均接触数）",
            "xlabel": f"B 组原子序号（1–{nb}，按序号分 {n_bb} 块）",
            "ylabel": f"A 组原子序号（1–{na}，按序号分 {n_ba} 块）",
            "x": [float(v) for v in cx],
            "y": [float(v) for v in cy],
            "z": [[float(v) for v in row] for row in z],
            "vmin": 0.0,
            "vmax": float(z.max()),
            "note": ("每个格 = 这一段 A 原子与这一段 B 原子之间的**每帧平均接触对数**"
                     f"（判据 distance < {cutoff:g} Å）。原子数太多时按序号分块，"
                     "块内取平均，所以格值不是整数。"),
        })
        res.summary["接触矩阵最大格值"] = float(z.max())
        res.add_notes(
            f"「接触矩阵」按原子序号把 A 组分成 {n_ba} 段、B 组分成 {n_bb} 段，"
            "每格是该段之间的**每帧平均接触对数**。它是静态模式下判断"
            "「哪一段贴着哪一段」的主图（曲线表达不了二维关系）。")
    res.add_notes(f"A 组 {na} 个原子，B 组 {nb} 个原子，统计 {n} 帧。")
    res.add_notes(
        "「平均接触数」= 该原子在每一帧接触到的 B 原子数，对帧取平均；"
        "它随配位环境连续变化，是描述接触强度的主指标。")
    return res


# ------------------------------------------------------------------ 界面宽度
@register("interface", "界面宽度分析")
def analyze_interface_width(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                            pair: tuple[str, str] | None = None, axis: int = 2,
                            nbins: int = 120, method: str = "10-90",
                            smooth: int = 5, mode: str = "mass",
                            verbose: bool = False) -> AnalysisResult:
    """界面位置与界面宽度分析（设计大纲第 14 章）。

    做法
    ----
    1. 沿指定方向分别得到两个组分的密度分布 ρ_A(x)、ρ_B(x)；
    2. 各自用体相值归一化，得到 f_A(x)、f_B(x)；
    3. 界面位置取 f_A 与 f_B 的交点（即两组分"各占一半"的位置）；
       ``method="10-90"`` 时界面宽度取**该界面附近** f 从 0.9 变到 0.1 的距离；
    4. 同时对每个界面做误差函数拟合，宽度按 ``2.563σ``（即 10–90 宽度）给出，
       便于与文献对比。两种估计互相独立，可交叉验证。

    板层（slab）体系在周期性边界下会出现**两个**界面，且两者未必等价
    （例如溶质只吸附在其中一侧）。所有交点都会按 z 顺序分别报告位置与宽度，
    宽度一律在局部窗口内测量，并且允许界面跨越周期边界。

    本方法对"分层/界面"体系有意义；对均匀混合体系，该指标只反映密度涨落，
    不应作为界面宽度解读。
    """
    if pair is None:
        names = list(groups)
        if len(names) < 2:
            raise ValueError("界面分析需要至少两个组分")
        pair = (names[0], names[1])
    a_name, b_name = pair
    if a_name not in groups or b_name not in groups:
        raise ValueError(f"组分 {pair} 不在 groups 中")

    centers, prof, dmeta = density_profile(mdt, groups, selection, axis=axis,
                                           nbins=nbins, mode=mode, verbose=verbose)
    rho_a = np.asarray(prof[a_name], dtype=float)
    rho_b = np.asarray(prof[b_name], dtype=float)
    bulk_a, bulk_b = _bulk_value(rho_a), _bulk_value(rho_b)
    # 体相值为 0（该组分在这个方向根本没有分布）时不能做归一化
    fa = _smooth(_safe_ratio(rho_a, bulk_a), smooth)
    fb = _smooth(_safe_ratio(rho_b, bulk_b), smooth)

    info = _interface_metrics(centers, fa, fb, method=method)
    axis_name = "abc"[axis]

    res = AnalysisResult(
        name="interface",
        title=f"界面宽度分析 —— {a_name} / {b_name}（沿 {axis_name} 方向）",
        meta={"axis": axis, "nbins": nbins, "pair": list(pair), "method": method,
              **dmeta},
    )
    res.panels = [
        Panel(xlabel=f"位置沿 {axis_name} 方向 (Å)",
              ylabel="质量密度 ρ (g/cm³)" if mode == "mass" else "数密度 ρ (1/Å³)",
              title=f"{a_name} / {b_name} 密度分布与界面"),
        Panel(xlabel=f"位置沿 {axis_name} 方向 (Å)", ylabel="归一化密度 ρ/ρ_bulk",
              title="归一化密度与界面宽度定义"),
    ]
    res.add_curve(a_name, centers, rho_a, panel=0, group=a_name)
    res.add_curve(b_name, centers, rho_b, panel=0, group=b_name)
    res.add_curve(f"{a_name} (归一化)", centers, fa, panel=1, group=a_name)
    res.add_curve(f"{b_name} (归一化)", centers, fb, panel=1, group=b_name)

    # ---------------- 1D 台阶模型的适用性判据（阶段 5）
    # 为什么必须判：均匀混合体系里 fa、fb 都只在 1 附近涨落，两条涨落曲线互相
    # 穿越同样会产生"交点"，于是 1.0.0 会把密度涨落的特征尺度当成"界面宽度"——
    # 实测 AdK 均匀溶液（根本没有界面）被判成「高（板层体系，检测到 2 个界面，
    # erf 拟合通过）」，并报出 36.6 Å 的"界面宽度"（盒长才 80 Å）。
    L_axis = float("nan")
    if centers.size > 1:
        L_axis = float(centers[-1] - centers[0] + (centers[1] - centers[0]))
    fmin_a = float(np.nanmin(fa)) if fa.size else float("nan")
    fmax_a = float(np.nanmax(fa)) if fa.size else float("nan")
    # 真实的分层台阶：ρ_A/ρ_bulk 必须同时接近 1（A 相内）与接近 0（B 相内）
    demixed = bool(np.isfinite(fmin_a) and fmin_a < 0.15 and fmax_a > 0.85)
    w_meas = float(info.get("界面宽度 10-90 (Å)", np.nan)) if info else float("nan")
    too_wide = bool(np.isfinite(w_meas) and np.isfinite(L_axis)
                    and w_meas > 0.25 * L_axis)
    slab_ok = bool(demixed and not too_wide)

    if info:
        res.summary.update(info)
        res.summary["组分 A"] = a_name
        res.summary["组分 B"] = b_name
        res.summary["1D 台阶模型是否适用"] = "是" if slab_ok else "否"
        if not slab_ok:
            # 拒绝套用 1D 模型：宽度与位置一律不给，数值只作为"涨落尺度"如实保留
            diag = float(res.summary.get("界面宽度 10-90 (Å)", float("nan")))
            for key in [k for k in list(res.summary)
                        if "界面宽度" in k or "界面位置" in k]:
                res.summary.pop(key, None)
            res.summary["界面宽度是否可用"] = "否"
            res.summary["界面位置是否可用"] = "否"
            if np.isfinite(diag):
                res.summary["密度涨落特征尺度 (Å)（不是界面宽度）"] = diag
            if not demixed:
                why = (f"该方向两组分**没有分层**：归一化密度只在 1 附近涨落"
                       f"（{a_name} 的 ρ/ρ_bulk 跨度仅 {fmax_a - fmin_a:.2f}，"
                       f"没有同时接近 0 与 1）")
            else:
                why = (f"测得的过渡尺度 {w_meas:.1f} Å 超过盒长的 1/4"
                       f"（{0.25 * L_axis:.1f} Å），不是局部界面")
            res.summary["界面判据可靠性"] = f"不适用——{why}；已拒绝套用 1D 台阶模型"
            res.add_notes(
                f"**未给出界面宽度与界面位置**：{why}。此时曲线上的「交点」只是"
                f"两组分密度涨落互相穿越，交点间的间距没有界面含义——这正是"
                f"1.0.0 的问题（实测 AdK 均匀溶液会给出 36.6 Å 的「界面宽度」）。")
            res.add_notes(
                "界面宽度只在体系**确实沿该方向分层**时才有定义（板层、液滴、"
                "真空/液面等）。请先看密度分布曲线是否呈台阶状；若不呈台阶，"
                "请改看密度分布本身或换一个方向。")
        else:
            fit_ok = "界面宽度 10-90 (erf 拟合) (Å)" in info
            n_cross = info.get("检测到的界面交点数", 0)
            if fit_ok and n_cross <= 1:
                res.summary["界面判据可靠性"] = "高（存在台阶状界面，erf 拟合通过）"
            elif fit_ok and n_cross <= 2:
                res.summary["界面判据可靠性"] = (
                    f"高（板层体系，检测到 {n_cross} 个界面，均已单独测量，erf 拟合通过）")
            elif fit_ok:
                res.summary["界面判据可靠性"] = f"中（检测到 {n_cross} 个交点，需人工确认）"
            else:
                res.summary["界面判据可靠性"] = (
                    "低——该方向不存在台阶状界面，上面的「界面宽度」只是密度涨落的"
                    "特征尺度，不能解释为真实界面宽度"
                )
            w0 = info.get("界面宽度 10-90 (Å)", float("nan"))
            parts = [f"界面位置 x₀ = {info['界面位置 (Å)']:.3f} Å",
                     f"界面宽度（10–90%）= {w0:.3f} Å" if np.isfinite(w0)
                     else "界面宽度（10–90%）无法直接测量"]
            if n_cross > 1 and "界面2 界面位置 (Å)" in info:
                w1 = info.get("界面2 界面宽度 10-90 (Å)", float("nan"))
                parts.append(
                    f"第二个界面 x₀ = {info['界面2 界面位置 (Å)']:.3f} Å、"
                    + (f"宽度 = {w1:.3f} Å" if np.isfinite(w1) else "宽度不可测"))
            res.add_notes("；".join(parts) + f"；检测到 {n_cross} 个密度交点。")
    else:
        res.add_notes("未能从密度分布中识别出界面（两个组分密度没有明显的过渡区）。")

    res.add_notes("界面位置定义为归一化密度曲线的交点，即两组分各占一半的位置。")
    res.add_notes("界面宽度是**局部**测量的：只取该界面两侧、同一过渡区内的 "
                  "0.9 / 0.1 穿越点，因此板层体系的多个界面不会互相串扰。")
    res.add_notes("对均匀混合体系该结果只反映密度涨落，不能解释为真实界面。")
    return res


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or y.size < w:
        return y
    kernel = np.ones(w, dtype=float) / w
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(w - 1 - pad, y[-1])])
    return np.convolve(ypad, kernel, mode="valid")[:y.size]


def _erf_width(x: np.ndarray, f: np.ndarray, x0: float | None = None,
               half_window: float | None = None,
               max_sigma: float | None = None,
               sigma_guess: float | None = None) -> dict | None:
    """用误差函数拟合**一个界面**的归一化轮廓 f(x) = 0.5·(1 ∓ erf((x−x0)/(√2 σ)))。

    台阶体系常是"板层"构型（周期性下会出现两个界面），直接对整条曲线拟合
    必然失败。因此给定界面位置 ``x0`` 时，只在 ``x0 ± half_window`` 的窗口内
    拟合，这样单侧过渡被正确分离出来。

    界面的**取向由数据自身判定**：先看 x0 左右两侧谁的平均密度高，再选择
    上升或下降的 erf。否则同一条曲线上的两个界面只有一个能拟合上
    （板层两侧恰好一升一降）。

    只在轮廓确实像台阶时才接受：要求 R² > 0.9 且 σ 不超过 ``max_sigma``。
    """
    try:
        from scipy.optimize import curve_fit
        from scipy.special import erf
    except Exception:  # noqa: BLE001
        return None

    xs_all = np.asarray(x, float)
    fs_all = np.asarray(f, float)
    m = np.isfinite(fs_all)
    if m.sum() < 8:
        return None
    xs, fs = xs_all[m], fs_all[m]
    span = float(xs[-1] - xs[0])

    # 只取界面附近的一段
    if x0 is not None:
        hw = half_window if half_window else max(span / 6.0, 1.0)
        sel = np.abs(xs - float(x0)) <= hw
        if sel.sum() >= 8:
            xs, fs = xs[sel], fs[sel]
    win = float(xs[-1] - xs[0]) if xs.size > 1 else span
    if max_sigma is None:
        max_sigma = win / 4.0
    if xs.size < 8 or win <= 0:
        return None

    # 取向：x0 右侧更高 → 上升台阶（sgn = +1），否则下降台阶（sgn = −1）
    xc_ref = float(x0) if x0 is not None else float(np.median(xs))
    left = fs[xs < xc_ref]
    right = fs[xs > xc_ref]
    sgn = 1.0 if (right.size and left.size and right.mean() > left.mean()) else -1.0

    def model(xx, xc, sigma):
        return 0.5 * (1.0 + sgn * erf((xx - xc) / (np.sqrt(2.0) * sigma)))

    # 初值：优先用 10-90 宽度换算的 σ（σ = w1090 / 2.5631），否则用窗口的 1/10。
    # 不这样做时最优化容易掉进"σ→很大 ⇒ 模型≈常数 0.5"的退化解（R²≈0）。
    if sigma_guess is not None and np.isfinite(sigma_guess) and sigma_guess > 0:
        s0 = float(sigma_guess)
        lo_s, hi_s = 0.2 * s0, min(win, 5.0 * s0)
    else:
        s0 = win / 10.0
        lo_s, hi_s = 1e-6, min(win, max_sigma)
    if hi_s <= lo_s:
        hi_s = lo_s * 1.01

    try:
        popt, _ = curve_fit(
            model, xs, fs,
            p0=[xc_ref, s0],
            maxfev=40000,
            bounds=([xs[0] - win, lo_s], [xs[-1] + win, hi_s]))
    except Exception:  # noqa: BLE001
        return None
    pred = model(xs, *popt)
    ss_res = float(np.sum((fs - pred) ** 2))
    ss_tot = float(np.sum((fs - fs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    xc, sigma = float(popt[0]), abs(float(popt[1]))
    if not np.isfinite(r2) or r2 < 0.9 or sigma > max_sigma:
        return {"rejected": True, "R2": r2, "sigma": sigma}
    return {"x0": xc, "sigma": sigma, "width_1090": 2.5631 * sigma, "R2": r2,
            "window": (float(xs[0]), float(xs[-1]))}


def _width_1090_local(x: np.ndarray, f: np.ndarray, x0: float, span: float,
                      window_frac: float = 0.35) -> float:
    """**局部**测量单个界面处的 10-90 宽度。

    板层体系里同一条曲线可能有多个界面；全局地找 0.9 / 0.1 穿越点会把两个界面
    混在一起（实测曾给出 89 Å 的荒谬宽度）。这里只接受
    ``|x − x0| ≤ window_frac · span`` 内、且**位于 x0 两侧**的一对穿越点：
    组分富集侧取 0.9（或 0.1），另一侧取 0.1（或 0.9）。
    """
    w = float(window_frac) * float(span)
    if not np.isfinite(w) or w <= 0:
        return float("nan")
    lo90 = [c for c in _level_crosses(x, f, 0.9) if c < x0 and x0 - c <= w]
    hi90 = [c for c in _level_crosses(x, f, 0.9) if c > x0 and c - x0 <= w]
    lo10 = [c for c in _level_crosses(x, f, 0.1) if c < x0 and x0 - c <= w]
    hi10 = [c for c in _level_crosses(x, f, 0.1) if c > x0 and c - x0 <= w]
    # 取离界面最近的候选，避免抓到远处的次级穿越
    near = lambda cs: min(cs, key=lambda v: abs(v - x0)) if cs else None  # noqa: E731
    a, b = near(lo90), near(hi10)          # 富集在左侧
    if a is None or b is None:
        a, b = near(hi90), near(lo10)      # 富集在右侧
    if a is None or b is None:
        return float("nan")
    return float(abs(b - a))


def _periodic_extend(x: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """把轮廓按盒长做周期延拓，返回 (x_ext, f_ext, 盒长)。

    板层体系的一个界面常落在周期盒的边界上：轮廓在采样范围内还没有走完
    整个 0 → 1 过渡（例如水的 0.1 穿越点已经在盒外）。延拓一份副本后，
    跨越边界的界面就能被正常测量。
    """
    xs = np.asarray(x, dtype=float)
    fs = np.asarray(f, dtype=float)
    if xs.size < 3:
        return xs, fs, 0.0
    d = np.diff(xs)
    dx = float(np.median(d))
    if not np.isfinite(dx) or dx <= 0:
        return xs, fs, 0.0
    box = dx * xs.size
    return np.concatenate([xs, xs + box]), np.concatenate([fs, fs]), box


def _level_crosses(x: np.ndarray, f: np.ndarray, level: float) -> list[float]:
    """返回 f(x) = level 的**全部**穿越点（线性插值）。"""
    d = np.asarray(f, dtype=float) - float(level)
    xs = np.asarray(x, dtype=float)
    cands: list[float] = []
    for k in range(xs.size - 1):
        if not (np.isfinite(d[k]) and np.isfinite(d[k + 1])):
            continue
        if d[k] == 0.0:
            cands.append(float(xs[k]))
        elif d[k] * d[k + 1] < 0:
            t = (0.0 - d[k]) / (d[k + 1] - d[k])
            cands.append(float(xs[k] + t * (xs[k + 1] - xs[k])))
    return cands


def _interface_metrics(x: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                       method: str = "10-90", max_interfaces: int = 3) -> dict | None:
    """由归一化密度曲线计算界面位置与宽度。

    板层（slab）体系在周期性边界下天然有**两个**界面，且两者未必等价
    （例如多肽只吸附在其中一侧）。因此这里不武断地只报一个：所有交点都按
    沿 z 的顺序列出（``界面位置 (Å)`` / ``界面2 位置 (Å)`` …），每个界面的
    宽度都是**局部**测量的，并允许界面跨越周期边界。
    """
    x = np.asarray(x, dtype=float)
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    if x.size < 5:
        return None

    # 周期延拓，用于测量跨边界 / 靠边界的界面
    xe, fae, box = _periodic_extend(x, fa)
    _, fbe, _ = _periodic_extend(x, fb)
    use_ext = box > 0
    span = box if use_ext else float(x[-1] - x[0])

    # 1) 界面位置：f_A 与 f_B 的交点
    d = fa - fb
    sign = np.sign(d)
    cross = np.nonzero(np.diff(sign) != 0)[0]
    n_cross = int(cross.size)
    spots: list[float] = []
    for k in cross:
        d_lo, d_hi = d[k], d[k + 1]
        if not (np.isfinite(d_lo) and np.isfinite(d_hi)):
            continue
        if d_hi != d_lo:
            spots.append(float(x[k] + (0.0 - d_lo) * (x[k + 1] - x[k]) / (d_hi - d_lo)))
        else:
            spots.append(float(0.5 * (x[k] + x[k + 1])))
    if not spots:
        x_fb = _level_cross(x, fa, 0.5, rising=False)
        if x_fb is None:
            return None
        spots = [float(x_fb)]
    spots.sort()                             # 沿 z 排序，输出稳定可复现

    out = OrderedDict()
    out["检测到的界面交点数"] = n_cross
    shown = spots[:max(1, int(max_interfaces))]
    methods: list[str] = []
    n_fit = 0
    for i, xi in enumerate(shown):
        tag = "" if i == 0 else f"界面{i + 1} "
        w_a = _width_1090_local(xe, fae, xi, span)
        w_b = _width_1090_local(xe, fbe, xi, span)
        width = (float(np.nanmean([w_a, w_b]))
                 if np.isfinite(w_a) or np.isfinite(w_b) else float("nan"))
        out[f"{tag}界面位置 (Å)"] = float(xi)
        out[f"{tag}界面宽度 10-90 (Å)"] = width
        out[f"{tag}组分 A 界面宽度 10-90 (Å)"] = float(w_a)
        out[f"{tag}组分 B 界面宽度 10-90 (Å)"] = float(w_b)

        # erf 拟合：给出与 10-90 独立的第二种宽度估计（板层畸变时可交叉验证）
        fit_a = _erf_width(x, fa, x0=xi, sigma_guess=_sigma_guess(xe, fae, xi, span))
        fit_b = _erf_width(x, fb, x0=xi, sigma_guess=_sigma_guess(xe, fbe, xi, span))
        if fit_a and not fit_a.get("rejected"):
            n_fit += 1
            out[f"{tag}界面位置 (erf 拟合) (Å)"] = fit_a["x0"]
            out[f"{tag}界面宽度 σ (erf 拟合) (Å)"] = fit_a["sigma"]
            out[f"{tag}界面宽度 10-90 (erf 拟合) (Å)"] = fit_a["width_1090"]
            out[f"{tag}erf 拟合 R²"] = fit_a["R2"]
            if fit_b and not fit_b.get("rejected"):
                out[f"{tag}界面宽度 σ (erf 拟合, 组分 B) (Å)"] = fit_b["sigma"]
                out[f"{tag}erf 拟合 R² (组分 B)"] = fit_b["R2"]
            if not np.isfinite(width):
                # 10-90 无法直接测量（例如界面被溶质严重扰动）时，采用拟合结果
                out[f"{tag}界面宽度 10-90 (Å)"] = fit_a["width_1090"]
                methods.append(f"界面{i + 1} 采用 erf 拟合（10-90 直接测量不可用）")
        else:
            r2 = fit_a.get("R2") if fit_a else None
            if r2 is not None and np.isfinite(r2):
                methods.append(f"界面{i + 1} erf 拟合被拒绝（R²={r2:.3f}），"
                               f"该处不是清晰的台阶状过渡")
            if not np.isfinite(width):
                out[f"{tag}界面宽度说明"] = (
                    "既找不到同一侧的 0.9 / 0.1 穿越点，erf 拟合也未通过："
                    "该处不是单调台阶（界面可能被溶质严重扰动），或采样太稀")

    out["界面宽度方法"] = method if not methods else f"{method}；" + "；".join(methods)
    if n_fit and not methods:
        out["界面宽度方法"] = f"{method}（并用 erf 拟合交叉验证）"
    if len(shown) > 1:
        if n_fit:
            out["界面说明"] = (
                f"周期盒中出现 {n_cross} 个界面交点，已按 z 顺序列出 {len(shown)} 个；"
                f"板层体系两侧界面通常等价，但吸附了溶质的一侧可能明显不同，"
                f"请分别比较宽度")
        else:
            out["界面说明"] = (
                f"出现 {n_cross} 个密度交点，但都没有通过台阶判据："
                f"这些交点应是**密度涨落**而不是真实界面，"
                f"上面的「界面宽度」不能按界面宽度解释")
    return out


def _sigma_guess(x: np.ndarray, f: np.ndarray, x0: float, span: float) -> float | None:
    """由局部 10-90 宽度给出 erf 拟合的 σ 初值（σ = w / 2.5631）。"""
    w = _width_1090_local(x, f, x0, span)
    if not np.isfinite(w) or w <= 0:
        return None
    return float(w / 2.5631)


def _level_cross(x: np.ndarray, f: np.ndarray, level: float,
                 rising: bool | None = None, near: float | None = None) -> float | None:
    """找 f(x) = level 的穿越点；``near`` 指定优先取靠近该位置的穿越点。"""
    d = np.asarray(f, dtype=float) - float(level)
    cands: list[float] = []
    for k in range(x.size - 1):
        if not (np.isfinite(d[k]) and np.isfinite(d[k + 1])):
            continue
        if d[k] == 0.0:
            cands.append(float(x[k]))
            continue
        if d[k] * d[k + 1] < 0:
            t = (0.0 - d[k]) / (d[k + 1] - d[k])
            cands.append(float(x[k] + t * (x[k + 1] - x[k])))
    if not cands:
        return None
    if near is not None:
        return min(cands, key=lambda v: abs(v - near))
    if rising is True:
        return cands[-1]
    if rising is False:
        return cands[0]
    return cands[len(cands) // 2]
