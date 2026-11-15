# -*- coding: utf-8 -*-
"""键取向序参数（BOO，Steinhardt）。

设计要点
--------
对每个原子的邻居方向做球谐展开

    q_lm(i) = (1/N_i) Σ_j Y_lm(r̂_ij)
    q_l(i)  = sqrt( 4π/(2l+1) Σ_m |q_lm(i)|² )

``l=6`` 对**局部晶序**最敏感（fcc/hcp/bcc 各不相同），``l=4`` 用来区分 fcc 与其它
密堆结构。本模块同时给出 **Lechner–Dellago 平均版** ``q̄_l``（先用邻居的 q_lm 平均
再求模），它对固-液差别比单原子 q_l 更敏感：单原子 q_l 在液相里也会有涨落，而
q̄_l 在液相里被邻居平均压低。

解析参考值（本模块的测试就是拿它钉死实现，见 ``test/selftest.py``）：

============  ========  ========
结构            q4        q6
============  ========  ========
fcc            0.19094   0.57452
hcp            0.09722   0.48476
bcc            0.03637   0.51069
sc             0.76376   0.35355
============  ========  ========

> 1.0.8：原来附带的「晶体/非晶区域识别」（φ_c、晶簇尺寸、Avrami）已按用户要求
> **删除**。需要晶区占比时请从 q̄6 的分布自行判断，或用「结构有序度」里的分量。
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

    性能：不再为了"找最近邻"去枚举 probe（8 Å）内的**全部**邻居对（20PE30 上
    3640 原子实测 ~50 ms/帧），改成 ``cKDTree.query(k=16)`` 的 k 近邻查询 ——
    每个原子只需要头几个邻居就能跳过成键对拿到最近邻，实测 ~7 ms/帧。

    ⚠️ 顺带修正了一个**顺序依赖**的老偏差：旧实现用
    ``self_capped_distance`` 的**无序对**（i<j）算"每个原子的最近非成键邻居"，
    却只把距离记到 ``pr[:, 0]``（较小下标那个原子）上 —— 于是与原子排序有关、
    整体偏大（20PE30 实测中位 2.3401 Å vs 真实 2.2481 Å，cutoff 3.159 vs 3.035）。
    现在两边端点都记（k 近邻天然如此），得到的是真正的最近邻距离。
    """
    n = int(pos.shape[0])
    if n < 2:
        return 5.0
    try:
        from scipy.spatial import cKDTree

        from ..neighbors import _ortho_lengths

        arr = np.asarray(pos, dtype=np.float32)
        L = _ortho_lengths(box)
        if box is None or L is not None:
            pts = np.mod(arr, L) if L is not None else arr
            tree = cKDTree(pts, boxsize=(L.astype(np.float64) if L is not None
                                         else None))
            k = int(min(16, n))
            dd, _ii = tree.query(pts, k=k)
            dd = np.atleast_2d(dd)
            if dd.shape[0] == 1:                    # n==1 的退化情形
                dd = dd.T
            # 第 0 列是自己（距离 0）；逐个原子取第一个 > _BOND_EXCLUDE 的距离
            ok = dd > _BOND_EXCLUDE
            first = np.argmax(ok, axis=1)
            has = ok.any(axis=1)
            fin = np.asarray(dd[np.arange(dd.shape[0])[has], first[has]],
                             dtype=float)
            fin = fin[np.isfinite(fin)]
            if fin.size:
                return float(np.median(fin) * 1.35)
    except Exception:                                      # noqa: BLE001 - 退回老路
        pass
    from MDAnalysis.lib.distances import self_capped_distance

    pr, dd = self_capped_distance(pos, max_cutoff=float(probe), box=box,
                                  return_distances=True)
    if pr.size == 0:
        return 5.0
    m = (dd > _BOND_EXCLUDE) & (pr[:, 0] != pr[:, 1])
    if not m.any():
        return 5.0
    nn = np.full(n, np.inf)
    # **两个端点都记**（无序对只记 pr[:,0] 会漏掉"该原子是较大下标"的那些对）
    np.minimum.at(nn, pr[m, 0], dd[m])
    np.minimum.at(nn, pr[m, 1], dd[m])
    fin = nn[np.isfinite(nn)]
    return float(np.median(fin) * 1.35) if fin.size else 5.0

# ------------------------------------------------------------------ 球谐
#: 估算近邻壳层时**跳过**比这个更近的配对（任何共价键都在此以内：C–C 1.54、
#: C–O 1.43、C–H 1.09 Å）——否则会把成键原子当成"最近邻居"。
_BOND_EXCLUDE = 2.2


def _ylm_real_parts(dirs: np.ndarray, l_max: int):
    """球谐的**实分量**（免掉逐 m 调 ``scipy.special.sph_harm_y``）。

    为什么不用 scipy 逐 m 算：``sph_harm_y`` 是按超几何函数求值的，单次约
    0.24 µs；l=4,6 一共 22 个分量、十几万邻居对时，光球谐就占 BOO 的 **67%**
    （实测 246 ms / 369 ms 一帧）。这里改成"连带勒让德递推 + 三角函数"，纯 numpy：

        P_m^m(x)   = (2m−1)·√(1−x²)·P_{m−1}^{m−1}(x)
        P_{m+1}^m  = (2m+1)·x·P_m^m(x)
        P_l^m(x)   = [x(2l−1)P_{l−1}^m − (l+m−1)P_{l−2}^m] / (l−m)

    返回 ``(P, cos_m, sin_m)``：``P[l][m]`` 与 ``cos(mφ)/sin(mφ)``（m ≥ 1）。
    归一化常数由调用方乘上（见 :func:`_ylm_norm`），这样同一份 ``P`` 供多个 ``l`` 复用。

    Condon–Shortley 的 ``(−1)^m`` 相位在这里**故意不乘**：它只给每个 (l,m) 一个
    全局符号，而 q_l 用的是 ``|⟨Y_lm⟩|²``，符号平方后消失；跳过多一次乘法，
    数值结果与 scipy 版本逐位一致（selftest 的解析晶格值会盯住这点）。
    """
    x = np.clip(np.asarray(dirs, dtype=float)[:, 2], -1.0, 1.0)
    rxy = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    P: list[list] = [[None] * (l_max + 1) for _ in range(l_max + 1)]
    P[0][0] = np.ones_like(x)
    for m in range(1, l_max + 1):
        P[m][m] = (2 * m - 1) * rxy * P[m - 1][m - 1]
    for m in range(0, l_max + 1):
        if m + 1 <= l_max:
            P[m + 1][m] = x * (2 * m + 1) * P[m][m]
        for l in range(m + 2, l_max + 1):
            P[l][m] = (x * (2 * l - 1) * P[l - 1][m]
                       - (l + m - 1) * P[l - 2][m]) / (l - m)
    cos_m = [None] + [np.cos(m * phi) for m in range(1, l_max + 1)]
    sin_m = [None] + [np.sin(m * phi) for m in range(1, l_max + 1)]
    return P, cos_m, sin_m


def _ylm_norm(l: int, m: int) -> float:
    """实分量的归一化常数 ``K_lm = sqrt((2l+1)/(4π) · (l−m)!/(l+m)!)``。"""
    from math import factorial

    return float(np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi)
                         * factorial(l - m) / factorial(l + m)))


def _sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """球谐函数；兼容 scipy 新旧 API（旧 ``sph_harm(m,l,az,pol)`` / 新 ``sph_harm_y(l,m,pol,az)``）。

    ⚠️ 生产路径已经不用它（见 :func:`_ylm_real_parts`，快 5–10 倍），
    保留它是为了 ``ql_from_directions`` 这类**小样本**调用与外部对拍。
    """
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
    """返回双向邻居对 ``(i, j)``（i ≠ j，距离 < cutoff，考虑 PBC）。

    实现：先拿**无序对**（i<j，省一半搜索与一半内存），再翻倍成双向对 —— 与
    ``capped_distance(pos, pos)`` 的结果集合完全相同，但少扫一遍、也不用再过滤
    N 个 ``i==j`` 的 0 距离自配对。底层走 :func:`mdta.neighbors.self_pairs`
    （正交盒用 cKDTree，比 MDAnalysis 的 cell list 快约 2×）。
    """
    from ..neighbors import self_pairs as _self_pairs

    # 只要原子对（方向由坐标自己算），省掉一趟距离计算
    pu, _d = _self_pairs(positions, box, float(cutoff), want_dist=False)
    pu = np.asarray(pu, dtype=np.int64).reshape(-1, 2)
    if pu.size == 0:
        return pu
    return np.vstack([pu, pu[:, ::-1]])


def _segmented_sum(values: np.ndarray, ii: np.ndarray, n: int) -> np.ndarray:
    """把 ``values`` 按 ``ii``（逐元素所属的中心原子序号）求和成长度 ``n`` 的数组。

    为什么不用 ``np.add.at``：它是**无缓冲**的 ``ufunc.at``，逐元素开销 ~1 µs；
    q_l 要对 ``(2l+1)`` 个分量各做一次（l=4,6 共 22 次），几十万邻居对时它一个
    就能吃掉整帧的时间。这里改成"按 ii 排序 + ``np.add.reduceat``"：C 层连续求和，
    实测快一到两个数量级，且结果逐位相同（浮点求和顺序变了，但按行求和与
    按元素累加的差异在 1e-15 量级）。
    """
    out = np.zeros(n, dtype=values.dtype)
    if values.size == 0:
        return out
    order = np.argsort(ii, kind="stable")
    ii_sorted = ii[order]
    uniq, starts = np.unique(ii_sorted, return_index=True)
    out[uniq] = np.add.reduceat(values[order], starts)
    return out


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
    # 大体系上"邻居搜索"和"球谐"各自都是几百毫秒到秒级的一段重活：
    # 两段之间各问一次取消，用户点取消就不用等整帧算完。
    from ..cancel import check as _cancel_check

    _cancel_check()
    pos = pos_in
    n = pos.shape[0]
    pairs = _neighbor_lists(pos, box, cutoff)
    _cancel_check()

    d = pos[pairs[:, 1]] - pos[pairs[:, 0]] if pairs.size else np.zeros((0, 3))
    if pairs.size and box is not None:
        from MDAnalysis.lib.distances import minimize_vectors

        d = minimize_vectors(d, box=box)
    norm = np.linalg.norm(d, axis=1) if d.size else np.zeros(0)
    ok = norm > 1e-12
    dirs = d[ok] / norm[ok, None]
    ii = pairs[ok, 0] if pairs.size else np.zeros(0, dtype=int)
    jj = pairs[ok, 1] if pairs.size else np.zeros(0, dtype=int)
    n_nb = np.zeros(n, dtype=int)
    if ii.size:
        n_nb = np.bincount(ii, minlength=n).astype(int)

    out: dict[str, np.ndarray] = {"n_neighbors": n_nb}
    if ii.size == 0:
        for l in l_list:
            out[f"q{l}"] = np.full(n, np.nan)
            if averaged:
                out[f"q{l}bar"] = np.full(n, np.nan)
        out["backend"] = "numpy"
        return out

    # ---- 球谐：**实分量**一次算好（P_l^m 递推），各 l 复用，不再逐 m 调 scipy
    l_max = int(max(l_list))
    P, cos_m, sin_m = _ylm_real_parts(dirs, l_max)
    _cancel_check()                     # 球谐（O(邻居对 × l²)）之后再看一次
    cnt = n_nb.astype(float)
    nz = cnt > 0
    for l in l_list:
        _cancel_check()
        # 分量按 m=0..l 组织：c_m = K·P_l^m·cos(mφ)、s_m = K·P_l^m·sin(mφ)
        # （m=0 只有 c）。|⟨Y_lm⟩|² 拆成这两组实数的平方和，完全等价于复数版：
        #   Σ_m |⟨Y_lm⟩|² = c_0² + 2 Σ_{m≥1} (c_m² + s_m²)
        c_avg = np.zeros((l + 1, n), dtype=float)
        s_avg = np.zeros((l + 1, n), dtype=float)
        for m in range(l + 1):
            K = _ylm_norm(l, m)
            c_avg[m] = _segmented_sum(K * P[l][m] * (cos_m[m] if m else 1.0), ii, n)
            if m:
                s_avg[m] = _segmented_sum(K * P[l][m] * sin_m[m], ii, n)
        c_avg[:, nz] /= cnt[nz]
        s_avg[:, nz] /= cnt[nz]
        acc = c_avg[0] ** 2 + 2.0 * (
            (c_avg[1:] ** 2).sum(axis=0) + (s_avg[1:] ** 2).sum(axis=0))
        out[f"q{l}"] = np.sqrt(4.0 * np.pi / (2 * l + 1) * acc)
        if averaged:
            # Lechner–Dellago 平均版：q̄_lm(i) = ⟨q_lm⟩，**中心原子自己也计入平均**
            # （分母 N_i+1）——这是标准定义，与 freud 的 average=True 一致。
            acnt = 1.0 + cnt
            ac = c_avg.copy()
            as_ = s_avg.copy()
            for m in range(l + 1):
                ac[m] = (c_avg[m] + _segmented_sum(c_avg[m][jj], ii, n)) / acnt
                if m:
                    as_[m] = (s_avg[m] + _segmented_sum(s_avg[m][jj], ii, n)) / acnt
            accb = ac[0] ** 2 + 2.0 * (
                (ac[1:] ** 2).sum(axis=0) + (as_[1:] ** 2).sum(axis=0))
            out[f"q{l}bar"] = np.sqrt(4.0 * np.pi / (2 * l + 1) * accb)
    out["backend"] = "numpy"
    return out


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
    # 抽稀必须放在**分配输出数组之前**：series / n_nb 的列数要等于真正参与计算的
    # 原子数（sub 的），否则 "ag 45005 列 而 q 只有 7565 个" 会在赋值时抛
    # ValueError: could not broadcast ... —— 46 体系（polymer 组分 45,005 原子，
    # 超过逐原子球谐上限 8,000）实测就是这样整项崩掉的。
    sub = ag
    note_sub = ""
    if ag.n_atoms > 8000:                    # 逐原子球谐是大开销，给个上限
        from .interface import _subsample_group

        sub, n0, n1, how = _subsample_group(ag, 8000)
        note_sub = (f"分析对象 {n0:,} 个原子超过逐原子球谐的实用上限，"
                    f"已抽稀到 {n1:,} 个（{how}）——q_l 是**局部**量，抽稀不改分布。")
    n_sel = int(sub.n_atoms)
    ls = [int(x) for x in l_list]
    keys = [f"q{l}" for l in ls] + ([f"q{l}bar" for l in ls] if averaged else [])
    series = {k: np.full((n_f, n_sel), np.nan) for k in keys}
    n_nb = np.full((n_f, n_sel), np.nan)

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
