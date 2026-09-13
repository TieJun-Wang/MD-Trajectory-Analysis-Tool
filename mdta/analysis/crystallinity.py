# -*- coding: utf-8 -*-
"""结晶行为 / 结构有序性分析模块（设计大纲第 15–19 章）。

提供：

- :func:`analyze_orientation`      链段取向分析（取向分布、取向参数 S）
- :func:`analyze_structural_order` 结构有序度与有序化趋势（综合指标）

RDF 与二面角分布见 :mod:`~mdta.analysis.interface` 与
:mod:`~mdta.analysis.conformation`，它们同时服务于本模块的分析目标。

说明
----
设计大纲第 18、19 章明确指出"具体计算方法需要根据研究体系和导师要求确定"。
本模块给出的有序度指标是**可配置的、定义明确的一般性指标**，
其权重与参考值都可以由使用者调整，不应直接当作绝对结晶度使用。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core import AnalysisResult, Panel, block_average, describe_array
from ..preprocess import FrameSelection
from ..units import bin_edges_to_centers, time_axis
from .base import frame_iterator, positions_for, register
from .conformation import (
    backbone_path,
    classify_dihedrals,
    compute_dihedral_series,
    make_dihedral_indices,
)

#: 取向参数 S 所需的最少链段数。S 是 N 个矢量的取向张量最大本征值，其统计
#: 不确定度约为 √(2/5N)：N=10 时约 ±0.20、N=100 时约 ±0.06。
#: 链段太少（例如整个组分只凑出 2 个链段）时 S 没有区分度，因此低于此数**拒绝给数**。
MIN_SEGMENTS_DEFAULT = 10

#: 多分量有序度指数的显示名（1.0.1 由「综合有序度指数」更正而来）
INDEX_NAME = "多分量有序度指数"
INDEX_LONG_NAME = "多分量有序度指数（自定义权重，不是结晶度）"

__all__ = [
    "segment_pairs",
    "resolve_segments",
    "repeat_unit_pairs",
    "nematic_order_parameter",
    "analyze_orientation",
    "analyze_structural_order",
    "INDEX_NAME",
]


# ------------------------------------------------------------------ 链段定义
def segment_pairs(ag, mode: str = "backbone", *, custom_pairs=None,
                  stride: int = 1, heavy_only: bool = True) -> np.ndarray:
    """生成链段（取向矢量）所对应的原子对，返回 ``(n, 2)`` 绝对索引数组。

    参数
    ----
    mode
        - ``"backbone"``：沿键连接图提取的链骨架，取骨架原子中相距
          ``stride`` 个键的原子对作为链段；
        - ``"bonds"``：所选原子组内的所有重原子键；
        - ``"custom"``：使用 ``custom_pairs``。
    stride
        ``backbone`` 模式下链段的跨度（以骨架原子个数计）。
    """
    ag = ag if ag.n_atoms else ag
    if mode == "custom":
        if custom_pairs is None:
            raise ValueError("mode='custom' 时必须提供 custom_pairs")
        return np.asarray(custom_pairs, dtype=int).reshape(-1, 2)

    if str(mode) == "repeat":
        pairs, _info = repeat_unit_pairs(ag, heavy_only=heavy_only)
        if pairs.shape[0] > 0:
            return pairs
        # 识别不出化学重复单元 → 回退到几何骨架（调用方可用 resolve_segments
        # 拿到回退标记，写进 metadata）
        return segment_pairs(ag, mode="backbone", stride=stride, heavy_only=heavy_only)

    if mode == "bonds":
        rows = []
        idx = set(int(a) for a in ag.indices)
        for b in ag.bonds:
            i, j = int(b[0].index), int(b[1].index)
            if i in idx and j in idx:
                rows.append((i, j))
        return np.asarray(rows, dtype=int).reshape(-1, 2)

    if mode != "backbone":
        raise ValueError(f"未知的链段模式: {mode!r}")

    path = backbone_path(ag, heavy_only=heavy_only)
    if path.size < 2:
        return np.zeros((0, 2), dtype=int)
    s = max(1, int(stride))
    rows = [(int(path[i]), int(path[i + s])) for i in range(path.size - s)]
    return np.asarray(rows, dtype=int).reshape(-1, 2)


def repeat_unit_pairs(ag, heavy_only: bool = True):
    """**化学重复单元**的取向矢量：两端是连着相邻残基的两个原子。

    做法：对每一个残基，找出它内部那些与**别的残基**成键的原子（"连接原子"）。
    恰好两个 → 该残基的重复单元矢量 = 这两点的连线；这正是化学意义上的重复
    单元方向（蛋白质的 N→C、糖的 O4→C1、合成高分子的单体轴）。

    为什么不能用几何骨架：``backbone_path`` 取的是键连接图的**最长路径**，
    对支化或带长侧基的分子，这条路径可能拐进侧基，得到的"链段"不是化学重复
    单元（规范表 ⑤）。因此 1.0.1 起把化学重复单元作为首选定义，几何法只作
    兜底，且在 metadata 里标成 ``geometric segment orientation``。

    返回 ``(pairs, info)``；``info`` 说明用了什么、跳过了多少。
    """
    from .conformation import _bonds_of

    info = {"source": "chemical repeat unit", "n_pairs": 0, "n_residues": 0,
            "n_skipped_terminal": 0, "n_skipped_branched": 0, "reason": ""}
    bonds = _bonds_of(ag)
    if bonds is None:
        info["reason"] = "拓扑没有键连接信息，无法识别化学重复单元"
        return np.zeros((0, 2), dtype=int), info

    in_group = set(int(a) for a in ag.indices)
    try:
        res_of = {int(a.index): int(a.resindex) for a in ag}
    except Exception:  # noqa: BLE001
        info["reason"] = "无法读取残基归属"
        return np.zeros((0, 2), dtype=int), info

    junctions: dict[int, list[int]] = {}
    for b in bonds:
        i, j = int(b[0].index), int(b[1].index)
        if i not in in_group or j not in in_group:
            continue
        if res_of[i] == res_of[j]:
            continue                       # 残基内部的键不算连接键
        junctions.setdefault(res_of[i], []).append(i)
        junctions.setdefault(res_of[j], []).append(j)

    heavy: dict[int, str] | None = None
    if heavy_only:
        try:
            heavy = {int(a.index): str(a.element) for a in ag}
        except Exception:  # noqa: BLE001
            heavy = None

    rows: list[tuple[int, int]] = []
    for _r, atoms in junctions.items():
        uniq = sorted(set(atoms))
        if len(uniq) < 2:
            info["n_skipped_terminal"] += 1      # 端基残基（只连一侧）
            continue
        if len(uniq) > 2:
            info["n_skipped_branched"] += 1      # 支化重复单元：方向不唯一
            continue
        a, b2 = uniq[0], uniq[1]
        if heavy is not None and (heavy.get(a) in ("H", "") or heavy.get(b2) in ("H", "")):
            info["n_skipped_branched"] += 1
            continue
        rows.append((a, b2))
    info["n_residues"] = len(junctions)
    info["n_pairs"] = len(rows)
    return np.asarray(rows, dtype=int).reshape(-1, 2), info


def resolve_segments(ag, mode: str = "repeat", custom_pairs=None, stride: int = 1,
                     heavy_only: bool = True):
    """确定取向分析用的链段集合，返回 ``(pairs, info)``。

    ``mode`` 取值：

    - ``"repeat"``（**默认**）：化学重复单元（:func:`repeat_unit_pairs`）；
      若该组分识别不出重复单元（例如整个分子就是一个残基），自动退化为
      ``"backbone"`` 并在 ``info`` 里标成几何法；
    - ``"backbone"``：几何法（键图最长路径），``info["source"]`` 标为
      ``geometric segment orientation``；
    - ``"bonds"`` / ``"custom"``：同 1.0.0。
    """
    info = {"mode": str(mode), "source": "chemical repeat unit",
            "fallback": False, "detail": ""}
    if str(mode) == "repeat":
        pairs, rinfo = repeat_unit_pairs(ag, heavy_only=heavy_only)
        if pairs.shape[0] > 0:
            info["detail"] = (f"化学重复单元：{pairs.shape[0]} 个（来自 "
                              f"{rinfo['n_residues']} 个成键残基；跳过端基残基 "
                              f"{rinfo['n_skipped_terminal']} 个、支化/含氢 "
                              f"{rinfo['n_skipped_branched']} 个）")
            info["n_pairs"] = int(pairs.shape[0])
            return pairs, info
        info["fallback"] = True
        info["source"] = "geometric segment orientation（回退）"
        info["detail"] = ("该组分识别不出化学重复单元"
                          f"（{rinfo.get('reason') or '没有跨残基的连接键'}），"
                          "已回退到几何骨架法——结果的 metadata 标为 "
                          "geometric segment orientation")
        pairs = segment_pairs(ag, mode="backbone", stride=stride, heavy_only=heavy_only)
        info["n_pairs"] = int(pairs.shape[0])
        return pairs, info
    if str(mode) == "backbone":
        info["source"] = "geometric segment orientation"
        info["detail"] = "几何法：键连接图最长路径上的连续原子对（不保证是化学重复单元）"
    pairs = segment_pairs(ag, mode=mode, custom_pairs=custom_pairs, stride=stride,
                          heavy_only=heavy_only)
    info["n_pairs"] = int(pairs.shape[0])
    return pairs, info


def nematic_order_parameter(vectors: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """由取向矢量计算二阶取向张量及其最大本征值（标准"向列序参数"）。

    ``Q = <(3 u uᵀ - I)/2>``，最大本征值 ``S`` 即取向参数：
    ``S = 0`` 完全无序，``S = 1`` 完全取向。同时返回对应的取向矢
    （director，即最大本征值对应的本征向量）。
    """
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        return {"S": float("nan"), "biaxiality": float("nan"),
                "director": np.array([np.nan] * 3), "n": 0}
    norm = np.linalg.norm(v, axis=1)
    good = norm > 1e-12
    if not good.any():
        return {"S": float("nan"), "biaxiality": float("nan"),
                "director": np.array([np.nan] * 3), "n": 0}
    u = v[good] / norm[good, None]
    w = None if weights is None else np.asarray(weights, dtype=float)[good]
    if w is None:
        w = np.ones(u.shape[0])
    w = w / w.sum()
    q = np.einsum("n,ni,nj->ij", w, u, u) * 3.0 - np.eye(3)
    q *= 0.5
    evals, evecs = np.linalg.eigh(q)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    return {
        "S": float(evals[0]),
        "biaxiality": float(evals[1] - evals[2]),
        "director": np.asarray(evecs[:, 0], dtype=float),
        "n": int(u.shape[0]),
        "eigenvalues": evals,
    }


def _pair_unit_vectors(mdt, ag, pairs: np.ndarray, selection: FrameSelection,
                       verbose: bool = False) -> np.ndarray:
    """逐帧计算链段单位矢量，返回 ``(n_frames, n_pairs, 3)``。"""
    pairs = np.asarray(pairs, dtype=int)
    n_f = len(selection.indices)
    out = np.full((n_f, pairs.shape[0], 3), np.nan, dtype=float)
    if pairs.size == 0:
        return out
    abs_to_local = {int(a): i for i, a in enumerate(np.asarray(ag.indices))}
    try:
        i0 = np.array([abs_to_local[int(x)] for x in pairs[:, 0]])
        i1 = np.array([abs_to_local[int(x)] for x in pairs[:, 1]])
    except KeyError as exc:
        raise ValueError(f"链段原子 {exc} 不在所选链内") from exc
    for k, (frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pos = positions_for(ag, unwrap=True)
        v = pos[i1] - pos[i0]
        out[k] = v
    return out


# ------------------------------------------------------------------ 取向分析
@register("orientation", "链段取向分析")
def analyze_orientation(mdt, ag, selection: FrameSelection, *,
                        mode: str = "repeat", custom_pairs=None,
                        stride: int = 1, heavy_only: bool = True,
                        reference_axis: Sequence[float] = (0.0, 0.0, 1.0),
                        nbins: int = 60, verbose: bool = False,
                        label: str = "链",
                        min_segments: int = MIN_SEGMENTS_DEFAULT) -> AnalysisResult:
    """链段取向分析（设计大纲第 17 章）。

    链段定义（1.0.1 起）
    -------------------
    默认 ``mode="repeat"``：**化学重复单元**优先（残基内连接相邻残基的两原子
    连线，见 :func:`repeat_unit_pairs`）；识别不出来时回退几何骨架，并在
    metadata 里标成 ``geometric segment orientation``。1.0.0 只有几何法
    （``mode="backbone"``），对支化/带侧基的分子可能把侧基当成"链段"。

    统计对象
    --------
    取向是**集成量**：传入的原子组里有多少分子就应平均多少分子。因此调用方
    应传**整个组分**（例如全部 polymer 分子），而不是一条链——结果里会写明
    链段来自多少个分子，避免"只算一条链却命名整个膜"。

    输出：

    - 取向参数 S 随时间变化（取向张量最大本征值，无需外部参考方向）
    - 相对参考方向的二阶序参数 P₂ = <(3cos²θ − 1)/2>
    - 链段取向角分布（相对时间平均取向矢）
    - 取向矢方向随时间的变化
    """
    times = np.asarray(selection.times_ps, dtype=float)
    pairs, sinfo = resolve_segments(ag, mode=mode, custom_pairs=custom_pairs,
                                    stride=stride, heavy_only=heavy_only)
    vecs = _pair_unit_vectors(mdt, ag, pairs, selection, verbose=verbose)
    try:
        from .conformation import molecule_slices
        ms = molecule_slices(ag)
        n_mol = len(ms[1]) if ms is not None else 1
    except Exception:  # noqa: BLE001
        n_mol = 1

    # ---------------------------------------------------------- 可用性判据
    # 取向参数 S 是取向张量 Q=<(3uuᵀ-I)/2> 的最大本征值：只有 **1 个链段** 时
    # Q 恒为该方向的完全取向态，S ≡ 1（不管这唯一一个链段指向哪里）——这是
    # 无意义的数；链段数为 0（例如把水当分析对象：每个水分子只有 1 个重原子、
    # 残基之间也没有连接键）则连方向都定义不了。
    if pairs.shape[0] < max(2, int(min_segments)):
        r_bad = AnalysisResult(
            name="orientation",
            title=f"链段取向分析 —— {label}",
            meta={"selection": label, "mode": mode, "n_segments": int(pairs.shape[0]),
                  "n_atoms": int(ag.n_atoms), "n_molecules": int(n_mol),
                  "defined": False, "segment_source": sinfo.get("source")},
        )
        r_bad.panels = [Panel(xlabel=time_axis(times)[1], ylabel="取向参数",
                              title="取向参数随时间变化")]
        r_bad.summary["取向分析是否可定义"] = "否"
        if pairs.shape[0] == 0:
            reason = (f"该组分识别不出任何取向链段："
                      f"{sinfo.get('detail') or '没有可用的化学重复单元或几何骨架'}。"
                      f"常见情形是它由独立小分子/单原子离子构成（例如水、离子），"
                      f"此时「链段取向」本身不适用。")
        else:
            reason = (f"只识别出 {pairs.shape[0]} 个链段，少于取向参数所需的最少链段数 "
                      f"{int(min_segments)}。S 是 N 个矢量取向张量的最大本征值，"
                      f"其统计不确定度约为 √(2/5N)：N={pairs.shape[0]} 时误差与 S 本身"
                      f"同量级，这个数没有区分度，因此不给。"
                      f"请把分析对象换成含更多重复单元的对象（例如整个组分的全部分子），"
                      f"或用 custom_pairs 给出足够多的链段。")
        r_bad.summary["原因"] = reason
        r_bad.add_notes(f"**未给出取向参数**：{reason}")
        r_bad.add_notes(f"该结果的 metadata 里 `defined=False`，界面/导出应据此跳过。")
        return r_bad

    n_f = times.size
    S_t = np.full(n_f, np.nan)
    p2_t = np.full(n_f, np.nan)
    dirs = np.full((n_f, 3), np.nan)
    ref = np.asarray(reference_axis, dtype=float)
    ref = ref / np.linalg.norm(ref) if np.linalg.norm(ref) > 0 else np.array([0.0, 0.0, 1.0])

    for k in range(n_f):
        st = nematic_order_parameter(vecs[k])
        S_t[k] = st["S"]
        dirs[k] = st["director"]
        v = vecs[k]
        nrm = np.linalg.norm(v, axis=1)
        good = nrm > 1e-12
        if good.any():
            u = v[good] / nrm[good, None]
            cos = np.clip(u @ ref, -1.0, 1.0)
            p2_t[k] = float(np.mean(0.5 * (3.0 * cos ** 2 - 1.0)))

    mean_dir = np.nanmean(dirs, axis=0)
    if np.linalg.norm(mean_dir) > 0:
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
    # 取向矢方向可差一个正负号，统一到与平均方向同向
    dirs_aligned = dirs * np.sign(dirs @ mean_dir)[:, None]

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="orientation",
        title=f"链段取向分析 —— {label}",
        meta={"selection": label, "mode": mode, "stride": int(stride),
              "n_segments": int(pairs.shape[0]), "n_frames": n_f, "n_atoms": int(ag.n_atoms),
              "n_molecules": int(n_mol), "segment_source": sinfo.get("source"),
              "segment_fallback": bool(sinfo.get("fallback"))},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="取向参数", title="取向参数随时间变化"),
        Panel(xlabel="链段与平均取向矢的夹角 θ (°)", ylabel="概率密度 P(θ)",
              title="链段取向分布"),
        Panel(xlabel=tlabel, ylabel="取向矢分量",
              title="取向矢 (director) 随时间变化"),
    ]
    res.add_curve("S (二阶取向张量最大本征值)", tx, S_t, panel=0)
    res.add_curve("二阶序参数 P2 (相对参考轴)", tx, p2_t, panel=0)

    # 取向角分布（相对时间平均取向矢）
    if pairs.size:
        cos_all = []
        for k in range(n_f):
            v = vecs[k]
            nrm = np.linalg.norm(v, axis=1)
            good = nrm > 1e-12
            if good.any():
                cos_all.append(np.abs((v[good] / nrm[good, None]) @ mean_dir))
        if cos_all:
            cos_all = np.concatenate(cos_all)
            theta = np.degrees(np.arccos(np.clip(cos_all, -1.0, 1.0)))
            edges = np.linspace(0.0, 90.0, int(nbins) + 1)
            hist, _ = np.histogram(theta, bins=edges, density=True)
            res.add_curve("取向角分布", bin_edges_to_centers(edges), hist,
                          kind="bar", panel=1)
            res.summary["平均夹角 θ (°)"] = float(np.mean(theta))
            res.summary["完全无序参考值 <cos²θ>"] = 1.0 / 3.0
            res.summary["实测 <cos²θ>"] = float(np.mean(cos_all ** 2))

    for i, comp in enumerate("xyz"):
        res.add_curve(f"director {comp}", tx, dirs_aligned[:, i], panel=2)

    res.summary.update(describe_array(S_t, prefix="S "))
    _bm, sem = block_average(S_t, n_blocks=min(5, max(2, n_f // 2)))
    res.summary["S 块平均标准误"] = float(sem)
    res.summary.update(describe_array(p2_t, prefix="P2 "))
    res.summary["平均取向矢 (director)"] = " ".join(f"{x:+.4f}" for x in mean_dir)
    res.summary["链段数"] = int(pairs.shape[0])
    if pairs.shape[0] >= 2:
        res.summary["S 估计标准误 (≈√(2/5N))"] = float(
            np.sqrt(2.0 / (5.0 * float(pairs.shape[0]))))
    res.summary["链段来源"] = str(sinfo.get("source"))
    res.summary["链段所属分子数"] = int(n_mol)
    res.summary["参考轴"] = " ".join(f"{x:g}" for x in ref)
    res.add_notes("S 由取向张量 Q=<(3uuᵀ-I)/2> 的最大本征值给出；S=0 完全无序，S=1 完全取向。")
    res.add_notes("P₂ 为相对给定参考轴的二阶序参数，便于与文献对比。")
    res.add_notes(f"链段定义：mode={mode}，共 {pairs.shape[0]} 个链段"
                  f"（{sinfo.get('detail') or ''}）")
    if n_mol > 1:
        res.add_notes(
            f"该结果是**{n_mol} 个分子的集合平均**（链段来自全部这些分子），"
            f"不是单条链的结果。取向是集成量：分子数越多，S 的统计意义越强。")
    if sinfo.get("fallback"):
        res.add_notes(
            "⚠️ 该组分识别不出化学重复单元，本次结果用的是**几何骨架法**，"
            "metadata 已标记 `segment_fallback=True`、"
            "`segment_source=geometric segment orientation`。"
            "几何骨架是键图最长路径，不等于化学重复单元（对支化/带侧基的分子"
            "可能拐进侧基）。若要知道化学重复单元的取向，请用 mode=\"repeat\" "
            "并确认所选原子组的残基之间确实有连接键，或用 custom_pairs 显式给出。")
    return res


# ------------------------------------------------------------------ 综合有序度
@register("order", "结构有序度分析")
def analyze_structural_order(mdt, ag, selection: FrameSelection, *,
                             dihedral_mode: str = "auto",
                             custom_dihedrals=None,
                             orient_mode: str = "repeat",
                             orient_pairs=None,
                             stride: int = 1,
                             rdf_group=None,
                             rdf_rmax: float = 8.0,
                             rdf_nbins: int = 80,
                             g_ref: float | None = None,
                             weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
                             verbose: bool = False,
                             label: str = "链") -> AnalysisResult:
    """结构有序度分析（设计大纲第 18、19 章）。

    计算三个定义明确的"有序度分量"，并给出可配置的综合有序度指数：

    ==================  =========================================  ==============
    分量                 定义                                        取值范围
    ==================  =========================================  ==============
    取向有序度 S         取向张量最大本征值                            0 – 1
    二面角有序度 S_φ     ``-<cos 3φ>``（全反式构象为 +1）              -1 – 1
    局部结构有序度 S_r   ``(g_max - 1)/(g_ref - 1)``，截断到 [0,1]      0 – 1
    ==================  =========================================  ==============

    综合有序度指数 = 各分量（统一到 0–1）按 ``weights`` 的加权平均。

    关于 ``g_ref``（重要）
    ---------------------
    RDF 的堆积峰高度**依赖体系的数密度**（同一个致密相放在大盒子里
    g_peak 就大、放在小盒子里就小），所以它的绝对值无法跨体系比较，
    更不能直接当成有序度。因此：

    - ``g_ref=None``（默认）：**局部结构分量不参与综合指数**，只作为参考量
      输出（曲线 + 统计量）。此时综合有序度指数由取向与二面角这两个
      **不需要外部参考态**的分量构成。
    - 使用者在自己体系里先确定"完全有序态"的 RDF 堆积峰高度，再把它作为
      ``g_ref`` 传进来，局部结构分量才会计入综合指数。

    ``weights`` 按实际参与的分量重新归一化。本指数用于比较同体系不同时间
    或不同体系的**相对**有序程度，**不应直接当作绝对结晶度**。
    """
    times = np.asarray(selection.times_ps, dtype=float)
    n_f = times.size
    notes: list[str] = []

    # ---- 1. 取向（链段默认取化学重复单元；识别不出则回退几何法并标注）
    pairs, sinfo = resolve_segments(ag, mode=orient_mode, custom_pairs=orient_pairs,
                                    stride=stride)
    vecs = _pair_unit_vectors(mdt, ag, pairs, selection, verbose=verbose)
    s_orient = np.full(n_f, np.nan)
    for k in range(n_f):
        s_orient[k] = nematic_order_parameter(vecs[k])["S"]
    try:
        from .conformation import molecule_slices
        _ms = molecule_slices(ag)
        n_mol = len(_ms[1]) if _ms is not None else 1
    except Exception:  # noqa: BLE001
        n_mol = 1
    notes.append(f"取向分量：{pairs.shape[0]} 个链段（mode={orient_mode}，"
                 f"来源={sinfo.get('source')}，来自 {n_mol} 个分子）")
    if sinfo.get("fallback"):
        notes.append("⚠️ 取向分量的链段用的是**几何骨架法**（识别不出化学重复单元），"
                     "metadata 标记 segment_fallback=True。")

    # ---- 2. 二面角
    mode_used = dihedral_mode
    if mode_used == "auto":
        try:
            is_prot = ag.select_atoms("protein").n_atoms > 0
        except Exception:  # noqa: BLE001
            is_prot = False
        mode_used = "phi_psi" if is_prot else "chain"

    if mode_used == "phi_psi":
        series = _protein_psi_series(mdt, ag, selection, verbose=verbose)
        dname = "psi"
    else:
        if mode_used == "chain":
            path = backbone_path(ag, heavy_only=True)
            idx = make_dihedral_indices(path)
        elif mode_used == "custom":
            if custom_dihedrals is None:
                raise ValueError("dihedral_mode='custom' 时必须提供 custom_dihedrals")
            idx = np.asarray(custom_dihedrals, dtype=int).reshape(-1, 4)
        elif mode_used == "topology":
            from .conformation import topology_dihedrals

            idx = topology_dihedrals(ag, heavy_only=True)
        else:
            raise ValueError(f"未知的二面角模式: {mode_used!r}")
        series = compute_dihedral_series(mdt, idx, selection, unwrap_group=ag,
                                         verbose=verbose)
        dname = "dihedral"

    s_tors = np.full(n_f, np.nan)
    f_trans = np.full(n_f, np.nan)
    for k in range(n_f):
        a = series[k]
        a = a[np.isfinite(a)]
        if a.size:
            s_tors[k] = float(-np.mean(np.cos(np.radians(3.0 * a))))
            f_trans[k] = classify_dihedrals(a)["trans"]
    notes.append(f"二面角分量：{series.shape[1]} 个二面角（{dname}）")

    # ---- 3. 局部结构（RDF 第一峰高度）
    from .interface import _first_peak, _rdf_accumulate

    gmax_t = np.full(n_f, np.nan)
    rg = rdf_group if rdf_group is not None else ag
    # 局部结构分量现在跑的是**整个组分**（可能几万原子），自 RDF 的代价按平方增长；
    # 超过阈值时按分子整流抽稀（g(r) 是密度归一化的量，抽稀只增加噪声、不改数值）。
    if rg.n_atoms > 20000:
        try:
            from .interface import _subsample_group

            rg, n0, n1, how = _subsample_group(rg, 12000)
            notes.append(f"局部结构分量的自 RDF 原本要算 {n0:,} 个原子，"
                         f"已抽稀到 {n1:,} 个（{how}）——g(r) 按密度归一化，"
                         f"抽稀只增加统计噪声、不改变峰高。")
        except Exception:  # noqa: BLE001
            pass
    edges = np.linspace(0.0, float(rdf_rmax), int(rdf_nbins) + 1)
    centers = bin_edges_to_centers(edges)
    from .interface import box_matrix

    vbox = float(abs(np.linalg.det(box_matrix(mdt.universe))))
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    neff = float(rg.n_atoms) * (float(rg.n_atoms) - 1.0)
    if neff > 0 and rg.n_atoms > 1:
        for k, idxk in enumerate(selection.indices):
            # 有序度关注的是"非键连"的局部堆积，因此这里剔除 1-2 成键原子对，
            # 否则第一峰会落在键长尺度上（截断后恒为 1，失去区分度）。
            # mode="total"：保持与 1.0.0 完全一致的口径（本模块的分母仍是
            # N(N−1) 的总体有序对数）。分子间/分子内口径的切换属于阶段 4
            # "综合结构有序度指数"的改造范围，这里不动，避免牵连无关模块。
            hist, nu, _ = _rdf_accumulate(mdt.universe, rg, rg, edges, [int(idxk)], True,
                                          exclude_bonded=True, mode="total")
            if nu == 0:
                continue
            ideal = nu * neff * shell / vbox
            g = np.divide(hist, ideal, out=np.zeros_like(hist), where=ideal > 0)
            # r > 2.5 Å 才算"堆积峰"：更近的峰属于化学键与 1-3 距离，
            # 反映的是键合拓扑而不是结构有序程度。
            pk = _first_peak(centers, g, r_min=2.5)
            if pk:
                gmax_t[k] = pk["height"]
    notes.append(f"局部结构分量：以 {rg.n_atoms} 个原子的自 RDF 的"
                 f"第一「堆积峰」(r > 2.5 Å) 高度衡量（已剔除 1-2 成键原子对）")

    # ---- 4. 综合
    def _clipped(x):
        return np.clip(x, 0.0, 1.0)

    # 局部结构分量：RDF 堆积峰高度依赖数密度，只有在给出参考态数值 g_ref
    # 时才归一化并计入综合指数，否则仅作为参考量输出。
    use_local = g_ref is not None and np.isfinite(g_ref) and g_ref > 1.0
    s_local = (_clipped((gmax_t - 1.0) / (float(g_ref) - 1.0)) if use_local
               else np.full(n_f, np.nan, dtype=float))

    w3 = np.asarray(weights, dtype=float)
    if w3.size != 3:
        w3 = np.ones(3, dtype=float)
    w3 = w3.copy()
    if not use_local:
        w3[2] = 0.0
    # 分量不可用（例如该组分没有链段、或没有二面角）时必须把权重置 0 并说明，
    # 否则指数会由 np.nan_to_num 把缺失分量当 0 参与加权 —— 悄悄给出一个假数。
    use_orient = bool(np.isfinite(s_orient).any()) and pairs.shape[0] >= MIN_SEGMENTS_DEFAULT
    use_tors = bool(np.isfinite(s_tors).any())
    if not use_orient:
        w3[0] = 0.0
        notes.append("取向分量不可用（可用链段 < 2，或该组分没有取向链段），"
                     "已从指数中移除（权重置 0）。")
    if not use_tors:
        w3[1] = 0.0
        notes.append("二面角分量不可用（该组分没有可用的二面角集合），"
                     "已从指数中移除（权重置 0）。")
    parts = [nm for nm, ok in (("取向", use_orient), ("二面角", use_tors),
                               ("局部结构", use_local)) if ok]
    index_defined = bool(parts) and float(w3.sum()) > 0
    w = w3 / w3.sum() if index_defined else np.zeros(3)

    # 各分量统一到「0 = 完全无序，1 = 完全有序」的尺度：
    #   取向有序度 S 本身即 0–1；
    #   二面角有序度 -<cos3φ> 完全随机时为 0，负值（反有序）截断为 0；
    #   局部结构有序度按 g_ref 归一化（未给 g_ref 时该分量权重为 0）。
    comps = np.vstack([_clipped(s_orient), _clipped(s_tors), s_local])
    if index_defined:
        order_index = np.nansum(w[:, None] * np.nan_to_num(comps, nan=0.0), axis=0)
        active = comps[np.nonzero(w)[0]]
        order_index[~np.isfinite(active).any(axis=0)] = np.nan
    else:
        # 没有任何可用分量时必须给 nan，不能给 0——0 会被读成"完全无序"，
        # 而真实情况是"这个指标无定义"。
        order_index = np.full(n_f, np.nan, dtype=float)

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="order",
        title=f"结构有序度分析 —— {label}",
        meta={"selection": label, "n_frames": n_f, "g_ref": (float(g_ref) if use_local
                                                             else None),
              "weights": list(map(float, w)), "dihedral_mode": mode_used},
    )
    res.panels = [
        Panel(xlabel=tlabel, ylabel="有序度分量（0=完全无序, 1=完全有序）",
              title="各有序度分量随时间变化"),
        Panel(xlabel=tlabel, ylabel="多分量有序度指数",
              title="多分量有序度指数随时间变化（自定义权重，非结晶度）"),
    ]
    res.add_curve("取向有序度 S", tx, _clipped(s_orient), panel=0)
    res.add_curve("二面角有序度 -<cos3φ>", tx, _clipped(s_tors), panel=0)
    res.add_curve("反式构象比例", tx, f_trans, panel=0)
    if use_local:
        res.add_curve("局部结构有序度", tx, s_local, panel=0)
    res.add_curve(INDEX_NAME, tx, order_index, panel=1)

    res.summary["指数名称"] = INDEX_LONG_NAME
    res.summary[f"{INDEX_NAME} 平均"] = float(np.nanmean(order_index))
    res.summary[f"{INDEX_NAME} 标准差"] = (float(np.nanstd(order_index, ddof=1))
                                            if n_f > 1 else 0.0)
    res.summary[f"{INDEX_NAME} 首帧"] = float(order_index[0]) if n_f else float("nan")
    res.summary[f"{INDEX_NAME} 末帧"] = float(order_index[-1]) if n_f else float("nan")
    if np.isfinite(order_index).sum() > 1:
        x = tx[np.isfinite(order_index)]
        y = order_index[np.isfinite(order_index)]
        slope = float(np.polyfit(x, y, 1)[0]) if x.size > 1 else float("nan")
        res.summary[f"{INDEX_NAME} 变化斜率 (1/时间单位)"] = slope
    res.summary.update(describe_array(s_orient, prefix="取向有序度 S "))
    res.summary.update(describe_array(s_tors, prefix="二面角有序度 -<cos3φ> "))
    res.summary.update(describe_array(gmax_t, prefix="RDF 堆积峰高度 "))
    res.summary["反式构象比例 平均"] = float(np.nanmean(f_trans))
    res.summary["指数包含的分量"] = (" + ".join(parts) if parts
                                     else "无（指数不可定义）")
    res.summary["指数是否可定义"] = "是" if index_defined else "否"
    if not index_defined:
        res.add_notes(
            "**未给出指数值**：该组分没有任何可用的有序度分量（取向链段不足、"
            "无二面角、且未提供 g_ref），指标无定义。这通常说明所选对象不适合"
            "做取向/有序度分析（例如独立小分子或单原子离子）。")
    res.summary["权重 (取向/二面角/局部)"] = " / ".join(f"{v:.3f}" for v in w)
    res.summary["链段来源"] = str(sinfo.get("source"))
    res.summary["链段所属分子数"] = int(n_mol)
    if use_local:
        res.summary["g_ref"] = float(g_ref)

    res.add_notes(*notes)
    res.add_notes(
        f"「{INDEX_NAME}」= 各参与分量（统一到 0=完全无序、1=完全有序）按权重的"
        f"加权平均，用于比较同体系不同时间或不同体系的**相对**有序程度。"
        f"该名称由 1.0.0 的「综合（结构）有序度指数」更正而来："
        f"它是一个**自定义权重的多分量指标**，不是结晶度。")
    res.add_notes(
        f"权重理由（已写入 metadata）：本次权重 = "
        f"{' / '.join(f'{v:.3f}' for v in w)}（取向 / 二面角 / 局部结构）。"
        f"默认三者等权；未给 g_ref 时局部结构分量权重**强制为 0**，"
        f"理由是 RDF 堆积峰高度依赖体系数密度、无法跨体系比较"
        f"（详见下一条说明）。使用者可用 weights= 调整，但请把理由写进报告。")
    res.add_notes("二面角有序度 -<cos3φ>：全反式或全旁式构象为 +1，完全随机为 0，"
                  "负值（反有序）在综合指标中截断为 0。")
    if use_local:
        res.add_notes(f"局部结构有序度以用户给定参考值 g_ref={float(g_ref):g} "
                      f"（完全有序态的 RDF 堆积峰高度）归一化。")
    else:
        res.add_notes("未提供 g_ref，因此综合指数只由取向与二面角两个分量构成。"
                      "RDF 堆积峰高度仅作为参考量输出：它依赖体系的数密度"
                      "（同一相放在不同大小的盒子里数值就不同），"
                      "必须由使用者在自己体系里确定「完全有序态」的数值后，"
                      "通过 g_ref 参数传入，局部结构分量才会参与综合指数。")
    res.add_notes("本指标为一般性结构有序性指标，不等同于绝对结晶度；"
                  "若需定量结晶度请按研究体系确定判据。")
    return res


def _protein_psi_series(mdt, ag, selection, verbose: bool = False) -> np.ndarray:
    """取蛋白质 ψ 二面角序列（用于蛋白质体系的构象有序度）。"""
    prot = ag.select_atoms("protein") if ag.select_atoms("protein").n_atoms else ag
    rows = []
    for res in prot.residues:
        try:
            b = res.psi_selection()
            if b is not None and b.n_atoms == 4:
                rows.append([int(x.index) for x in b])
        except Exception:  # noqa: BLE001
            continue
    idx = np.asarray(rows, dtype=int).reshape(-1, 4)
    if idx.size == 0:
        return np.zeros((len(selection.indices), 0), dtype=float)
    return compute_dihedral_series(mdt, idx, selection, unwrap_group=ag, verbose=verbose)
