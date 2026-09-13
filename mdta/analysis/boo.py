# -*- coding: utf-8 -*-
"""键取向序参数（BOO）与晶体/非晶区域识别。

设计要点
--------
1. **BOO（Steinhardt）**：对每个原子的邻居方向做球谐展开

       q_lm(i) = (1/N_i) Σ_j Y_lm(r̂_ij)
       q_l(i)  = sqrt( 4π/(2l+1) Σ_m |q_lm(i)|² )

   ``l=6`` 对**局部晶序**最敏感（fcc/hcp/bcc 各不相同），``l=4`` 用来区分 fcc 与其它
   密堆结构。本模块同时给出 **Lechner–Dellago 平均版** ``q̄_l``（先用邻居的 q_lm 平均
   再求模），它才是识别固-液最稳的判据：单原子 q_l 在液相里也会有涨落，而 q̄_l 在
   液相里被邻居平均压低。

   解析参考值（本模块的测试就是拿它钉死实现，见 ``selftest.py``）：

   ============  ========  ========
   结构            q4        q6
   ============  ========  ========
   fcc            0.19094   0.57452
   hcp            0.09722   0.48476
   bcc            0.03637   0.51069
   sc             0.76376   0.35355
   ============  ========  ========

2. **晶体/非晶区域识别**：``q̄6 > q6_solid`` 的原子判为"固相原子"，再用**空间连通性**
   （键长 cutoff 内的邻居关系）做并查集聚类 → 逐个晶簇；在线统计**结晶分数 φ_c(t)**、
   最大晶簇尺寸、晶簇个数。这套做法不依赖序参量的绝对刻度，只看"局部有序 + 空间成片"。

3. **Avrami**：φ_c(t) 归一化为结晶度 X(t) 后拟合

       ln(-ln(1-X)) = n·ln t + n·ln k

   斜率给 Avrami 指数 n（成核/生长机制），截距给速率常数 k。**只在满足条件时才给**：
   X 的覆盖范围足够、且随时间单调增长；否则明确拒绝（体系没结晶、或已结晶完毕），
   绝不给一个"能算但没意义"的 n。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core import AnalysisResult, Panel
from ..preprocess import FrameSelection
from ..units import bin_edges_to_centers, time_axis
from .base import frame_iterator, register

#: 参考相的解析 q4/q6（Steinhardt 1983；用于结果里标注"离哪个晶型近"）
REFERENCE_QL: dict[str, tuple[float, float]] = {
    "fcc": (0.19094, 0.57452),
    "hcp": (0.09722, 0.48476),
    "bcc": (0.03637, 0.51069),
    "sc": (0.76376, 0.35355),
}


def _estimate_cutoff(pos: np.ndarray, box, probe: float = 8.0) -> float:
    """自动推定近邻壳层半径 = 「每个原子到最近邻居的距离」中位数 × 1.35。

    两个坑都踩过，写在这里免得回头再犯：

    1. 曾用「到质心距离的 40 分位」当探针半径：在伸展体系上它会是几十埃，
       ``capped_distance`` 于是枚举近乎**全部原子对** —— 6000 原子直接跑不完。
       第一壳层只能从**最近邻距离**估计，不能从整体尺度估计。
    2. 曾把「最近邻」直接当第一壳层：**共价键内的原子**才是最近的（联合原子
       C–C 约 1.5 Å），于是自动 cutoff 落到 1.97 Å、平均邻居数只有 2、
       q6 虚高到 0.77，结晶识别把 80% 原子误判成固相。所以这里必须
       **跳过 1-2 配对**（用 ``_BOND_EXCLUDE`` 这个比任何共价键都长的下限）。
    """
    from MDAnalysis.lib.distances import self_capped_distance

    n = pos.shape[0]
    pr, dd = self_capped_distance(pos, max_cutoff=float(probe), box=box,
                                  return_distances=True)
    if pr.size == 0:
        return 5.0
    m = (dd > _BOND_EXCLUDE) & (pr[:, 0] != pr[:, 1])
    if not m.any():
        return 5.0
    nn = np.full(n, np.inf)
    np.minimum.at(nn, pr[m, 0], dd[m])
    fin = nn[np.isfinite(nn)]
    return float(np.median(fin) * 1.35) if fin.size else 5.0

# ------------------------------------------------------------------ 球谐
#: 估算近邻壳层时**跳过**比这个更近的配对（任何共价键都在此以内：C–C 1.54、
#: C–O 1.43、C–H 1.09 Å）——否则会把成键原子当成"最近邻居"。
_BOND_EXCLUDE = 2.2


def _sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """球谐函数；兼容 scipy 新旧 API（旧 ``sph_harm(m,l,az,pol)`` / 新 ``sph_harm_y(l,m,pol,az)``）。"""
    try:                                            # scipy >= 1.15
        from scipy.special import sph_harm_y

        return sph_harm_y(l, m, theta, phi)
    except ImportError:                             # pragma: no cover
        from scipy.special import sph_harm

        return sph_harm(m, l, phi, theta)


def ql_from_directions(unit_dirs: np.ndarray, l: int) -> float:
    """给定一组**单位方向**，返回其（未平均的）q_l。解析测试与内部复用同一实现。"""
    d = np.asarray(unit_dirs, dtype=float).reshape(-1, 3)
    if d.size == 0:
        return float("nan")
    z = np.clip(d[:, 2], -1.0, 1.0)
    theta = np.arccos(z)                            # 极角
    phi = np.arctan2(d[:, 1], d[:, 0])              # 方位角
    s = 0.0
    for m in range(-l, l + 1):
        s += abs(np.mean(_sph_harm(l, m, theta, phi))) ** 2
    return float(np.sqrt(4.0 * np.pi / (2 * l + 1) * s))


def _neighbor_lists(positions: np.ndarray, box, cutoff: float):
    """返回 ``(i, j)`` 邻居对（i ≠ j，距离 < cutoff，考虑 PBC）。"""
    from MDAnalysis.lib.distances import capped_distance

    pairs, _ = capped_distance(positions, positions, max_cutoff=float(cutoff),
                               box=box, return_distances=True)
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    keep = pairs[:, 0] != pairs[:, 1]
    return pairs[keep]


def _freud_ql(positions: np.ndarray, box_dims, cutoff: float,
              l_list: Sequence[int], averaged: bool) -> dict:
    """用 **freud** 计算 Steinhardt 序参数（**可选**的交叉校验通道）。

    本仓库**不把 freud 列为依赖**（依赖已经够多，且离线环境装不上）。
    自带实现严格按规范定义写（Steinhardt 1983 的 q_l、Lechner–Dellago 2008 的
    平均版 q̄_l），因此结果可与 freud / pyscal / OVITO 直接对比；需要时用
    ``backend="freud"`` 显式指定，用它对同一帧做数值交叉核对即可。
    """
    import freud                                     # noqa: PLC0415

    L = np.asarray(box_dims, dtype=float)
    if np.allclose(L[3:6], 90.0, atol=1e-6):
        fbox = freud.box.Box(L[0], L[1], L[2])
    else:
        from MDAnalysis.lib.mdamath import triclinic_vectors

        fbox = freud.box.Box.from_matrix(triclinic_vectors(L))
    pos = np.asarray(positions, dtype=np.float32)
    aq = freud.locality.AABBQuery(fbox, pos)
    nl = aq.query(pos, {"r_max": float(cutoff), "exclude_ii": True}).toNeighborList()
    out: dict[str, object] = {"n_neighbors": np.asarray(nl.neighbor_counts, dtype=int),
                              "backend": "freud"}
    for l in l_list:
        for avg in ((False, True) if averaged else (False,)):
            st = freud.order.Steinhardt(l=int(l), average=bool(avg))
            st.compute((fbox, pos), neighbors=nl)
            out[f"q{l}" + ("bar" if avg else "")] = np.asarray(st.particle_order,
                                                               dtype=float)
    return out


def _freud_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("freud") is not None


def compute_ql(positions: np.ndarray, box, cutoff: float,
               l_list: Sequence[int] = (4, 6), averaged: bool = True,
               backend: str = "numpy") -> dict:
    """逐原子计算 q_l 与（可选）Lechner–Dellago 平均版 q̄_l。

    ``backend``：默认 ``"numpy"``（自带实现，零依赖、可复现）；
    显式传 ``"freud"`` 时改用 freud 交叉校验（未安装则报错，不会静默换实现）；
    ``"auto"`` 表示有 freud 就用 freud。
    返回 ``{"q4", "q6", "q4bar", "q6bar", "n_neighbors", "backend"}``
    （``averaged=False`` 时不含 ``*bar``）。
    """
    pos_in = np.asarray(positions, dtype=float)
    if backend in ("auto", "freud") and _freud_available():
        try:
            return _freud_ql(pos_in, box, cutoff, l_list, averaged)
        except Exception:                            # noqa: BLE001 - 退回自带实现
            if backend == "freud":
                raise
    pos = pos_in
    n = pos.shape[0]
    pairs = _neighbor_lists(pos, box, cutoff)
    n_nb = np.zeros(n, dtype=int)
    if pairs.size:
        np.add.at(n_nb, pairs[:, 0], 1)

    d = pos[pairs[:, 1]] - pos[pairs[:, 0]] if pairs.size else np.zeros((0, 3))
    if pairs.size and box is not None:
        from MDAnalysis.lib.distances import minimize_vectors

        d = minimize_vectors(d, box=box)
    norm = np.linalg.norm(d, axis=1) if d.size else np.zeros(0)
    ok = norm > 1e-12
    dirs = d[ok] / norm[ok, None]
    ii = pairs[ok, 0] if pairs.size else np.zeros(0, dtype=int)
    theta = np.arccos(np.clip(dirs[:, 2], -1, 1))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])

    out: dict[str, np.ndarray] = {"n_neighbors": n_nb}
    for l in l_list:
        # 每帧的 Y_lm：形状 (2l+1, n_pairs)
        Y = np.array([_sph_harm(l, m, theta, phi) for m in range(-l, l + 1)])
        # 逐原子求平均：q_lm(i) = mean_j Y_lm(r_ij)
        s_lm = np.zeros((2 * l + 1, n), dtype=complex)
        cnt = np.zeros(n, dtype=float)
        if ii.size:
            np.add.at(cnt, ii, 1.0)
            for k in range(2 * l + 1):
                np.add.at(s_lm[k], ii, Y[k])
        nz = cnt > 0
        s_lm[:, nz] /= cnt[nz]
        out[f"q{l}"] = np.sqrt(4.0 * np.pi / (2 * l + 1)
                               * np.sum(np.abs(s_lm) ** 2, axis=0))
        if averaged:
            # Lechner–Dellago 平均版：q̄_lm(i) = ⟨q_lm⟩，**中心原子自己也计入平均**
            # （分母 N_i+1）——这是标准定义，与 freud 的 average=True 一致。
            a_lm = np.array(s_lm, copy=True)
            acnt = np.ones(n, dtype=float)
            if ii.size:
                np.add.at(acnt, ii, 1.0)
                for k in range(2 * l + 1):
                    np.add.at(a_lm[k], ii, s_lm[k][pairs[ok, 1]])
            a_lm /= acnt[None, :]
            out[f"q{l}bar"] = np.sqrt(4.0 * np.pi / (2 * l + 1)
                                      * np.sum(np.abs(a_lm) ** 2, axis=0))
    out["backend"] = "numpy"
    return out


# --------------------------------------------------------------- 连通聚类
def _clusters(pairs: np.ndarray, mask: np.ndarray, n: int) -> list[np.ndarray]:
    """在 ``mask`` 为真的原子上，用并查集按邻居关系求连通分量。"""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if pairs.size:
        a = pairs[:, 0][mask[pairs[:, 0]] & mask[pairs[:, 1]]]
        b = pairs[:, 1][mask[pairs[:, 0]] & mask[pairs[:, 1]]]
        for x, y in zip(a.tolist(), b.tolist()):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
    groups: dict[int, list[int]] = {}
    for idx in np.flatnonzero(mask).tolist():
        groups.setdefault(find(idx), []).append(idx)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def _avrami(times_ps: np.ndarray, phi: np.ndarray) -> dict:
    """对结晶分数曲线做 Avrami 拟合；条件不满足时明确拒绝。"""
    ok = np.isfinite(phi)
    if ok.sum() < 10:
        return {"可拟合": False, "说明": "有效数据点不足 10 个"}
    t = np.asarray(times_ps, dtype=float)[ok] / 1000.0        # ns
    x = np.asarray(phi, dtype=float)[ok]
    phi0, phi1 = float(x[0]), float(x[-1])
    span = phi1 - phi0
    if span < 0.05:
        return {"可拟合": False, "起始": phi0, "结束": phi1,
                "说明": f"结晶分数变化只有 {span:.3f}（< 0.05），体系没有明显结晶"}
    xs = (x - phi0) / span if span > 0 else x * 0
    xs = np.clip(xs, 1e-6, 1 - 1e-6)
    y = np.log(-np.log(1.0 - xs))
    m = np.isfinite(y) & (t > 0)
    if m.sum() < 5:
        return {"可拟合": False, "说明": "可用于拟合的点不足 5 个"}
    A = np.vstack([np.log(t[m]), np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    n_av, b = float(coef[0]), float(coef[1])
    yhat = A @ coef
    ss_res = float(np.sum((y[m] - yhat) ** 2))
    ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2)) or 1.0
    if not (0.1 < n_av < 10):
        return {"可拟合": False, "Avrami 指数 n": n_av,
                "说明": f"拟合出的 n={n_av:.2f} 超出合理范围（0.1–10），判定不可靠"}
    return {"可拟合": True, "Avrami 指数 n": round(n_av, 3),
            "速率常数 k (1/ns^n)": float(np.exp(b / n_av)),
            "R²": round(1.0 - ss_res / ss_tot, 4),
            "归一化区间": [round(phi0, 4), round(phi1, 4)],
            "说明": "X(t) 已按 (φ-φ0)/(φ1-φ0) 归一化；n 为成核/生长机制指数"}


def _mode_common(kw: Mapping | None) -> dict:
    p = dict(kw or {})
    return p


# ------------------------------------------------------------------ 分析项
@register("boo", "键取向序参数 BOO")
def analyze_boo(mdt, ag, selection: FrameSelection, *,
                cutoff: float | None = None, l_list: Sequence[int] = (4, 6),
                averaged: bool = True, nbins: int = 60, backend: str = "numpy",
                verbose: bool = False, label: str = "链") -> AnalysisResult:
    """逐原子 q4/q6（及平均版 q̄4/q̄6）随时间演化。

    ``cutoff=None`` 时按体系自动取一个**近邻壳**：用 g(r) 第一极小不适合逐帧做，
    这里用简单稳妥的经验值 ``1.30 × (最近邻距离中位数)``——对密堆体系等价于取第一壳层。
    """
    times = np.asarray(selection.times_ps, dtype=float)
    n_f = times.size
    u = mdt.universe
    ls = [int(x) for x in l_list]
    keys = [f"q{l}" for l in ls] + ([f"q{l}bar" for l in ls] if averaged else [])
    series = {k: np.full((n_f, ag.n_atoms), np.nan) for k in keys}
    n_nb = np.full((n_f, ag.n_atoms), np.nan)
    sub = ag
    note_sub = ""
    if ag.n_atoms > 8000:                    # 逐原子球谐是大开销，给个上限
        from .interface import _subsample_group

        sub, n0, n1, how = _subsample_group(ag, 8000)
        note_sub = (f"分析对象 {n0:,} 个原子超过逐原子球谐的实用上限，"
                    f"已抽稀到 {n1:,} 个（{how}）——q_l 是**局部**量，抽稀不改分布。")

    cut = cutoff
    for k, (_frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pos = np.asarray(sub.positions, dtype=float)
        if cut is None:
            cut = _estimate_cutoff(pos, u.dimensions)
        q = compute_ql(pos, u.dimensions, float(cut), l_list=ls, averaged=averaged,
                       backend=backend)
        backend_used = q.get("backend", backend)
        for kk in keys:
            series[kk][k] = q[kk]
        n_nb[k] = q["n_neighbors"]

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="boo",
        title=f"键取向序参数 BOO —— {label}",
        meta={"selection": label, "n_atoms": int(sub.n_atoms), "cutoff_Å": round(float(cut), 3),
              "l": ls, "averaged": bool(averaged), "n_frames": n_f},
    )
    res.panels = [Panel(xlabel=tlabel, ylabel="q̄6", title="平均键取向序 q̄6 随时间"),
                  Panel(xlabel="q̄6", ylabel="概率密度 P(q̄6)", title="q̄6 分布"),
                  Panel(xlabel="q̄4", ylabel="q̄6", title="q̄4–q̄6 平面（参考相位置）")]
    if note_sub:
        res.add_notes(note_sub)
    for kk in keys:
        arr = series[kk]
        mean_t = np.nanmean(arr, axis=1)
        res.add_curve(kk, tx, mean_t, kind="line", panel=0)
        fin = arr[np.isfinite(arr)]
        if fin.size:
            res.summary[f"{kk} 平均"] = float(np.mean(fin))
            res.summary[f"{kk} 标准差（原子间）"] = float(np.std(fin))
        res.summary[f"{kk} 帧平均（时间序列均值）"] = float(np.nanmean(mean_t))
    if averaged:
        q6b = series["q6bar"]
        fin = q6b[np.isfinite(q6b)]
        if fin.size:
            edges = np.linspace(float(np.percentile(fin, 0.5)),
                                float(np.percentile(fin, 99.5)), nbins + 1)
            h, _ = np.histogram(fin, bins=edges, density=True)
            res.add_curve("q̄6 分布（全帧全原子）", bin_edges_to_centers(edges), h,
                          kind="bar", panel=1)
        q4b = series["q4bar"]
        res.add_curve("q̄4–q̄6", np.nanmean(q4b, axis=1), np.nanmean(q6b, axis=1),
                      kind="scatter", panel=2)
        for name, (r4, r6) in REFERENCE_QL.items():
            res.add_curve(f"参考 {name}", np.array([r4]), np.array([r6]),
                          kind="scatter", panel=2)
    res.summary["邻域半径 (Å)"] = round(float(cut), 3)
    res.summary["平均邻居数"] = float(np.nanmean(n_nb))
    res.summary["是否用平均版 q̄_l"] = "是" if averaged else "否"
    res.summary["计算后端"] = backend_used
    res.add_notes(
        "q_l 定义：q_lm(i)=⟨Y_lm(r̂_ij)⟩_j，q_l=√(4π/(2l+1)Σ_m|q_lm|²)。"
        "l=6 对局部晶序最敏感、l=4 用于区分 fcc 与其它密堆结构。")
    res.add_notes(
        "参考相解析值：" + "；".join(f"{k} q4={v[0]:.5f} q6={v[1]:.5f}"
                                     for k, v in REFERENCE_QL.items())
        + "。液相的单原子 q6 也会有涨落，**平均版 q̄6 才是识别固-液的稳定判据**（用邻居的 q_lm 平均后再求模）。")
    return res


@register("crystal", "晶体/非晶区域识别")
def analyze_crystal_regions(mdt, ag, selection: FrameSelection, *,
                            cutoff: float | None = None, q6_solid: float = 0.5,
                            min_cluster: int = 10, averaged: bool = True,
                            backend: str = "numpy",
                            avrami: bool = True, max_group_atoms: int = 20000,
                            verbose: bool = False, label: str = "链") -> AnalysisResult:
    """判"固相原子"→ 空间连通聚类 → 结晶分数 φ_c(t) 与 Avrami 动力学。

    判据：``q̄6 > q6_solid``（默认 0.5：fcc/hcp 的 q̄6 约 0.5–0.6，液相约 0.2–0.35）。
    聚类：固相原子之间若在 ``cutoff`` 内互为邻居则连边，并查集求连通分量；
    **只有规模 ≥ min_cluster 的连通分量算作晶区**（滤掉热涨落造成的零星固相原子）。

    φ_c(t) = 属于晶区的原子数 / 总原子数；再对 φ_c(t) 做 Avrami 拟合（条件不满足则拒绝）。
    """
    times = np.asarray(selection.times_ps, dtype=float)
    n_f = times.size
    u = mdt.universe
    sub = ag
    note_sub = ""
    if ag.n_atoms > max_group_atoms:
        from .interface import _subsample_group

        sub, n0, n1, how = _subsample_group(ag, int(max_group_atoms))
        note_sub = (f"分析对象 {n0:,} 个原子超过上限 {max_group_atoms:,}，"
                    f"已抽稀到 {n1:,} 个（{how}）；φ_c 是分数，抽稀不改其期望值。")
    n = sub.n_atoms

    phi = np.full(n_f, np.nan)          # 晶区原子占比
    n_solid = np.full(n_f, np.nan)      # 固相原子占比（含零星）
    n_clu = np.full(n_f, np.nan)
    big = np.full(n_f, np.nan)          # 最大晶簇原子数
    for k, (_frame, _t) in enumerate(frame_iterator(mdt, selection, verbose=verbose)):
        pos = np.asarray(sub.positions, dtype=float)
        cut = cutoff if cutoff is not None else _estimate_cutoff(pos, u.dimensions)
        q = compute_ql(pos, u.dimensions, float(cut), l_list=(4, 6),
                       averaged=averaged, backend=backend)
        backend_used = q.get("backend", backend)
        key = "q6bar" if averaged else "q6"
        mask = np.isfinite(q[key]) & (q[key] > float(q6_solid))
        n_solid[k] = mask.mean()
        pairs = _neighbor_lists(pos, u.dimensions, float(cut))
        cl = _clusters(pairs, mask, n)
        sizes = np.array([c.size for c in cl]) if cl else np.zeros(0, dtype=int)
        keep = sizes >= int(min_cluster)
        n_clu[k] = int(keep.sum())
        n_in = int(sizes[keep].sum()) if keep.any() else 0
        phi[k] = n_in / max(n, 1)
        big[k] = float(sizes[keep].max()) if keep.any() else 0.0

    tx, tlabel = time_axis(times)
    res = AnalysisResult(
        name="crystal",
        title=f"晶体/非晶区域识别 —— {label}",
        meta={"selection": label, "n_atoms": int(n), "q6_solid": float(q6_solid),
              "min_cluster": int(min_cluster), "averaged": bool(averaged),
              "cutoff_Å": round(float(cut), 3), "n_frames": n_f},
    )
    res.panels = [Panel(xlabel=tlabel, ylabel="结晶分数 φ_c", title="晶区原子占比随时间"),
                  Panel(xlabel=tlabel, ylabel="最大晶簇原子数", title="最大晶簇尺寸随时间"),
                  Panel(xlabel="ln t", ylabel="ln(−ln(1−X))", title="Avrami 图")]
    if note_sub:
        res.add_notes(note_sub)
    res.add_curve("结晶分数 φ_c", tx, phi, kind="line", panel=0)
    res.add_curve("固相原子占比（含零星）", tx, n_solid, kind="line", panel=0)
    res.add_curve("最大晶簇原子数", tx, big, kind="line", panel=1)
    res.add_curve("晶簇个数", tx, n_clu, kind="line", panel=1)

    fin = phi[np.isfinite(phi)]
    if fin.size:
        res.summary["结晶分数 平均"] = float(np.mean(fin))
        res.summary["结晶分数 首帧"] = float(fin[0])
        res.summary["结晶分数 末帧"] = float(fin[-1])
        res.summary["结晶分数 最大"] = float(np.max(fin))
        res.summary["固相原子占比 平均"] = float(np.nanmean(n_solid))
        res.summary["晶簇个数 平均"] = float(np.nanmean(n_clu))
        res.summary["最大晶簇原子数 平均"] = float(np.nanmean(big))
        res.summary["统计帧数"] = int(fin.size)
    res.summary["固相判据"] = f"{'q̄6' if averaged else 'q6'} > {q6_solid:g}"
    res.summary["晶簇最小规模"] = int(min_cluster)
    res.summary["计算后端"] = backend_used
    res.summary["邻域半径 (Å)"] = round(float(cut), 3)
    if avrami and fin.size:
        av = _avrami(times, phi)
        for k2, v2 in av.items():
            res.summary[f"Avrami {k2}" if k2 != "可拟合" else "Avrami 是否可拟合"] = v2
        if av.get("可拟合"):
            xs = (phi - fin[0]) / max(fin[-1] - fin[0], 1e-9)
            xs = np.clip(xs, 1e-6, 1 - 1e-6)
            tt = np.where(times > 0, times, np.nan)
            res.add_curve("Avrami 线性化", np.log(tt), np.log(-np.log(1 - xs)),
                          kind="scatter", panel=2)
            res.add_notes(
                f"Avrami 拟合：n = {av['Avrami 指数 n']}，k = "
                f"{av['速率常数 k (1/ns^n)']:.3g} 1/ns^n，R² = {av['R²']}"
                f"（X(t) 按 (φ−φ0)/(φ1−φ0) 归一化，φ0={av['归一化区间'][0]}，"
                f"φ1={av['归一化区间'][1]}）。")
        else:
            res.add_notes(f"**未给出 Avrami 参数**：{av.get('说明')}。"
                          f"（体系没有明显结晶、或已结晶完毕时，n 与 k 没有物理意义。）")
    res.add_notes(
        f"固相判据：{'平均版 q̄6' if averaged else '单原子 q6'} > {q6_solid:g}"
        f"（fcc/hcp 的 q̄6 约 0.5–0.6，液相约 0.2–0.35，故 0.5 是常用分界）。")
    res.add_notes(
        f"晶区定义：固相原子中规模 ≥ {min_cluster} 的**空间连通簇**（邻居半径 "
        f"{float(cut):.2f} Å）；零星固相原子（热涨落）不计入 φ_c。")
    res.add_notes(
        "φ_c 是**结构判据给出的有序原子占比**，与量热/密度法的结晶度口径不同；"
        "跨体系比较前请确认判据一致。")
    return res
