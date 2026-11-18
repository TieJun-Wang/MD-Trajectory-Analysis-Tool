# -*- coding: utf-8 -*-
"""MSD / 扩散系数 —— ``util_msd.py`` + ``util_diffusion.py``（MDTransport）口径。

这个包（MDTransport，Shah 组）里能和我们**逐值对照**的，除了 RDF 就是 MSD 这一套；
这里把它的算法原样搬过来，用来交叉验证本工具「MSD」模块与 MDAnalysis 的
``EinsteinMSD``。

它们的 MSD 口径（与工具/Forcite 的差别都在细节里）
--------------------------------------------------
1. **解包用"增量最小镜像"**：``shift = round((r_f − r_{f−1})/L)·L``，逐帧累加
   （不是对每个粒子搜全轨迹，也不是"直接信文件坐标"）；
2. **去漂移**：把选定粒子集合的**质心漂移**（``com(t) − com(0)``）从每个粒子上减掉
   —— 等价于本工具 ``analyze_msd(remove_drift=True)``；
3. **MSD 用 FFT + 全时间起点平均**（每个 τ 用 ``n_frame − τ`` 个起点），
   按**方向求和**：``xyz`` → 3D MSD（``6Dt``），只给 ``z`` → 1D（``2Dt``）；
4. **D 的求法**：先在 MSD(τ) 上找 **log–log 斜率 β ≈ 1** 的"扩散区窗口"
   （``--min_window`` 给最小窗口，默认取总长的一半/8000 ps），再在窗口内做
   **OLS 线性拟合**：``D = slope / (2·dim) × 1e-8``（Å²/ps → m²/s）。
   工具那边是 GLS + 分块误差条，口径不同但**斜率/截距同源**，可以互证。

单独跑：``python test/mda_test/msd_mdat.py``（菜单）
``python test/mda_test/msd_mdat.py --top a.xtd --trj a.trj --mode com --frames 200 -y``
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, save_csv
except ImportError:
    from _common import frame_indices, save_csv

#: Å²/ps → m²/s
A2PS_TO_M2S = 1e-8
_DIRS = {"x": 0, "y": 1, "z": 2}


# ------------------------------------------------------------------ 解包 / 去漂移
def unwrap_incremental(coords, boxes):
    """增量最小镜像解包：``coords`` 形如 ``(T, N, 3)``，``boxes`` 形如 ``(T, 3)``。

    与 ``util_msd.py::unwrap_and_group`` 第一段完全一致：对**相邻帧的位移**取最小
    镜像（``round(dx/L)*L``）再累加，得到连续轨迹。
    """
    out = np.array(coords, dtype=np.float64, copy=True)
    boxes = np.asarray(boxes, dtype=np.float64)
    if out.ndim != 3 or out.shape[0] < 2 or not np.all(np.asarray(boxes) > 0):
        return out
    for f in range(1, out.shape[0]):
        L = boxes[f]
        if not np.all(L > 0):
            continue
        d = out[f] - out[f - 1]
        out[f] -= np.round(d / L[None, :]) * L[None, :]
    return out


def remove_com_drift(coords, weights=None):
    """把整组质心的漂移从每个粒子上减掉（与 ``unwrap_and_group`` 的后半段一致）。"""
    x = np.asarray(coords, dtype=np.float64)
    if x.shape[0] < 2:
        return x
    w = np.ones(x.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    com = np.einsum("tni,n->ti", x, w)
    return x - (com - com[0])[:, None, :]


# ------------------------------------------------------------------ FFT MSD
def msd_fft_single(coords2d):
    """**单个粒子**的 MSD（FFT），返回 ``(deltas, ssd, counts)``。

    ``coords2d`` 形如 ``(T, n_dirs)``；``ssd[d-1, k]`` 是 τ=d 时**所有时间起点**的
    位移平方和（按方向分开）——与 ``util_msd.py::process_msd_fft_array`` 同式：
    ``ssd = Q[T] − Q[τ] + Q[T−τ] − 2·ACF[τ]``。
    """
    c = np.asarray(coords2d, dtype=np.float64)
    T = c.shape[0]
    if T < 2:
        return np.zeros(0), np.zeros((0, c.shape[1])), np.zeros(0)
    try:
        from scipy.fft import irfft, next_fast_len, rfft

        nfft = next_fast_len(2 * T)
        F = rfft(c, n=nfft, axis=0)
        acf = irfft(F * np.conj(F), n=nfft, axis=0)[:T]
    except Exception:                                    # noqa: BLE001
        nfft = 1 << int(np.ceil(np.log2(2 * T)))
        F = np.fft.rfft(c, n=nfft, axis=0)
        acf = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:T]
    Q = np.zeros((T + 1, c.shape[1]))
    np.cumsum(c * c, axis=0, out=Q[1:])
    d = np.arange(1, T)
    ssd = (Q[T] - Q[d]) + Q[T - d] - 2.0 * acf[d]
    return d, ssd, (T - d).astype(float)


def msd_mdat(u, ag=None, *, mode="com", axes="xyz", frames=None,
             remove_drift=True, verbose=True):
    """按 MDTransport 口径算 MSD。

    参数
        u        Universe
        ag       被追踪的原子组（默认全部原子）
        mode     ``"com"``：每个**分子**取质心作为一个粒子（他们的默认口径）；
                 ``"atom"``：每个原子一个粒子
        axes     ``"xyz"``（3D）/ ``"x"`` / ``"xy"`` …
        frames   帧号列表（等差；默认全部）
        remove_drift  是否减掉整组质心漂移（他们的默认是减）
    返回 ``{"t", "msd", "msd_per_axis", "n_particles", "dim", "dt", "mode"}``
    """
    a = u.atoms if ag is None else ag
    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    if len(use) < 2:
        raise SystemExit(f"MSD 至少要 2 帧（当前 {len(use)} 帧）")
    ax = "".join(c for c in "xyz" if c in str(axes).lower()) or "xyz"
    cols = [_DIRS[c] for c in ax]

    sub = np.asarray(a.indices, dtype=np.int64)
    n_at = sub.size
    T = len(use)
    pos = np.empty((T, n_at, 3), dtype=np.float64)
    boxes = np.zeros((T, 3), dtype=np.float64)
    times = np.zeros(T, dtype=np.float64)
    for k, fi in enumerate(use):
        ts = u.trajectory[fi]
        pos[k] = np.asarray(a.positions, dtype=np.float64)
        dim = ts.dimensions
        boxes[k] = (np.asarray(dim, dtype=np.float64)[:3] if dim is not None
                    else np.zeros(3))
        times[k] = float(ts.time)
    pos = unwrap_incremental(pos, boxes)

    masses = None
    try:
        masses = np.asarray(a.masses, dtype=np.float64)
        if not np.any(masses > 0):
            masses = None
    except Exception:                                    # noqa: BLE001
        masses = None
    if remove_drift:
        pos = remove_com_drift(pos, masses)

    if str(mode).lower() == "com":
        try:
            frags = list(a.fragments)
        except Exception:                                # noqa: BLE001
            frags = []
        mols = [np.searchsorted(sub, np.asarray(m.indices, dtype=np.int64))
                for m in frags]
        mols = [m for m in mols if m.size]
        if mols:
            coords = np.empty((T, len(mols), 3), dtype=np.float64)
            for j, idx in enumerate(mols):
                w = (np.ones(idx.size) if masses is None
                     else masses[idx])
                w = w / w.sum()
                coords[:, j, :] = np.einsum("tni,n->ti", pos[:, idx, :], w)
        else:                             # 没有键表 → 退化成按原子
            mode = "atom"
            coords = pos
    else:
        mode = "atom"
        coords = pos

    n_p = coords.shape[1]
    n_lag = T - 1
    ssd_sum = np.zeros((n_lag, len(cols)), dtype=np.float64)
    cnt_sum = np.zeros(n_lag, dtype=np.float64)
    for p in range(n_p):
        d, ssd, cnt = msd_fft_single(coords[:, p, cols])
        if d.size:
            ssd_sum += ssd
            cnt_sum += cnt
    ok = cnt_sum > 0
    msd_dir = np.zeros((n_lag, len(cols)))
    msd_dir[ok] = ssd_sum[ok] / cnt_sum[ok, None]
    dt = float(np.median(np.diff(times))) if T > 1 else 1.0
    t = np.arange(1, T, dtype=float) * dt
    res = {"t": t, "msd": msd_dir.sum(axis=1), "msd_per_axis": msd_dir,
           "n_particles": int(n_p), "dim": len(cols), "dt": dt,
           "mode": str(mode), "axes": ax, "frames": use}
    if verbose:
        print(f"MSD（MDTransport 口径）：{mode} 模式，{n_p} 个粒子，{T} 帧，"
              f"dt = {dt:g} ps，方向 {ax.upper()}（{len(cols)}D）"
              f"{'，已去质心漂移' if remove_drift else ''}")
    return res


# ------------------------------------------------------------------ β 窗口 + D
def beta_window(t, msd, *, min_window_ps=0.0, n_grid=40):
    """找 **log–log 斜率 β ≈ 1** 的时间窗口（返回下标区间 ``(i1, i2)``）。

    对应 ``util_diffusion.find_best_time_window``：先按 ``--min_window`` 定最小窗口
    （没给就取总长的一半），再搜"β 最接近 1"的窗口，同等偏差时**取更早、更长**的。
    """
    t = np.asarray(t, dtype=float)
    m = np.asarray(msd, dtype=float)
    ok = np.isfinite(t) & np.isfinite(m) & (t > 0) & (m > 0)
    idx = np.flatnonzero(ok)
    if idx.size < 4:
        return 0, max(idx.size, 1)
    t, m = t[idx], m[idx]
    n = t.size
    min_len = max(4, int(round(min_window_ps / (t[1] - t[0]))) if min_window_ps > 0
                  else n // 2)
    min_len = min(min_len, n)
    best = None
    starts = np.linspace(0, max(n - min_len, 0), num=min(n_grid, n), dtype=int)
    for i1 in np.unique(starts):
        for i2 in range(i1 + min_len, n + 1):
            lt = np.log(t[i1:i2])
            lm = np.log(m[i1:i2])
            if lt.size < 4 or np.ptp(lt) <= 0:
                continue
            beta = float(np.polyfit(lt, lm, 1)[0])
            score = (abs(beta - 1.0), -float(i2 - i1), float(i1))
            if best is None or score < best[0]:
                best = (score, int(i1), int(i2), beta)
    if best is None:
        return 0, n
    return int(idx[best[1]]), int(idx[best[2] - 1]) + 1


def diffusion_mdat(t, msd, dim=3, *, min_window_ps=0.0, window=None):
    """在扩散区窗口内 OLS 拟合 → ``D = slope/(2·dim)·1e-8``（m²/s）。

    与 ``util_diffusion.calculate_diffusivity`` 的 OLS 分支同式；返回
    ``{"D_m2s", "D_slope", "beta", "window", "n_points"}``。
    """
    t = np.asarray(t, dtype=float)
    m = np.asarray(msd, dtype=float)
    i1, i2 = window if window is not None else beta_window(
        t, m, min_window_ps=min_window_ps)
    i1, i2 = max(0, int(i1)), min(int(i2), t.size)
    if i2 - i1 < 2:
        return {"D_m2s": float("nan"), "D_slope": float("nan"),
                "beta": float("nan"), "window": (i1, i2), "n_points": i2 - i1}
    tt, mm = t[i1:i2], m[i1:i2]
    slope = float(np.polyfit(tt, mm, 1)[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = float(np.polyfit(np.log(tt), np.log(mm), 1)[0]) \
            if np.all(mm > 0) else float("nan")
    return {"D_m2s": slope / (2.0 * int(dim)) * A2PS_TO_M2S,
            "D_slope": slope, "beta": beta,
            "window": (i1, i2), "n_points": int(i2 - i1),
            "t_window_ps": (float(tt[0]), float(tt[-1]))}


# ------------------------------------------------------------------ 出数 / 对拍
def test_msd_mdat(u, ag=None, *, mode="com", axes="xyz", frames=None,
                  min_window_ps=0.0, prefix="msd_mdat"):
    """算 MSD + D，打印并导出 ``{prefix}.csv``（``tau_ps, msd_A2, msd_x, msd_y, msd_z``）。"""
    res = msd_mdat(u, ag, mode=mode, axes=axes, frames=frames)
    fit = diffusion_mdat(res["t"], res["msd"], dim=res["dim"],
                         min_window_ps=min_window_ps)
    cols = [res["t"], res["msd"]] + [res["msd_per_axis"][:, k]
                                     for k in range(res["msd_per_axis"].shape[1])]
    names = ["tau_ps", "msd_A2"] + [f"msd_{c}" for c in res["axes"]]
    save_csv(f"{prefix}.csv", cols, header=",".join(names),
             fmt=",".join(["%.4f"] * len(cols)))
    i1, i2 = fit["window"]
    print(f"  扩散区窗口：τ = {fit.get('t_window_ps', (np.nan, np.nan))[0]:g} – "
          f"{fit.get('t_window_ps', (np.nan, np.nan))[1]:g} ps"
          f"（{fit['n_points']} 点，β = {fit['beta']:.3f}）")
    print(f"  D = {fit['D_m2s']:.4e} m²/s"
          f"（slope = {fit['D_slope']:.6g} Å²/ps，D = slope/(2·{res['dim']})）")
    return {**res, "fit": fit}


def compare_engine(u, ag=None, *, mode="com", frames=None, object_engine=None,
                   remove_drift=True):
    """与**本工具引擎**（``mdta.analysis.dynamics.analyze_msd``）对拍 MSD 曲线与 D。

    两边跑**同一批帧**；原子组按序号对齐。返回 ``{"msd_max_rel", "D_rel"}``。
    """
    from types import SimpleNamespace

    from mdta.analysis.dynamics import analyze_msd
    from mdta.io import MDTrajectory

    fl = [int(f) for f in (frame_indices(u) if frames is None else frames)]
    mine = msd_mdat(u, ag, mode=mode, frames=fl, remove_drift=remove_drift,
                    verbose=False)
    a = u.atoms if ag is None else ag

    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(fl, dtype=int),
                          times_ps=times[np.asarray(fl, dtype=int)],
                          n_frames=len(fl), describe=f"{len(fl)} 帧（指定）",
                          notes=[], head=lambda self, k: self)
    ga = mdt.universe.atoms[np.asarray(a.indices, dtype=np.int64)]
    eng_object = object_engine or ("molecule" if mine["mode"] == "com" else "atom")
    res = analyze_msd(mdt, {"X": ga}, sel, object=eng_object,
                      remove_drift=remove_drift, unwrap=True,
                      fit_fraction=(0.1, 1.0), tau_intervals=1, dim=mine["dim"])
    curves = [c for c in res.curves if str(c.kind) != "bar"]
    if not curves:
        print("  引擎没给出 MSD 曲线")
        return None
    e_t = np.asarray(curves[0].x, dtype=float)
    e_msd = np.asarray(curves[0].y, dtype=float)
    # ⚠️ 两边的 τ 轴起点不同：引擎从 τ=0 开始，这里从 τ=dt 开始
    #    → **按 τ 对齐**，不能按下标硬对（否则整体错一格，τ→0 处会报出巨大偏差）
    if e_t.size > 1:
        dt_e = float(np.median(np.diff(e_t))) or mine["dt"]
        j = np.rint(mine["t"] / dt_e).astype(int)
        keep = (j >= 0) & (j < e_msd.size)
        mine_msd = mine["msd"][keep]
        eng_msd = e_msd[j[keep]]
    else:
        n = min(e_msd.size, mine["msd"].size)
        mine_msd, eng_msd = mine["msd"][:n], e_msd[:n]
    ok = np.isfinite(eng_msd) & (eng_msd > 0)
    # 只在 MSD 达到峰值 5% 以上的 lag 上比相对偏差：τ→0 处 MSD 只有几 Å²，
    # float32 坐标的舍入就能造成很大的"相对差"，那是噪声不是口径差异
    big = ok & (eng_msd > 0.05 * np.max(eng_msd[ok])) if ok.any() else ok
    rel = float(np.max(np.abs(mine_msd[big] - eng_msd[big]) / eng_msd[big])) \
        if big.any() else float("nan")
    absd = float(np.max(np.abs(mine_msd[big] - eng_msd[big]))) if big.any() else float("nan")
    d_eng = None
    for k, v in res.summary.items():
        if str(k).endswith("D (m²/s)") or str(k).endswith("D∥ (m²/s)"):
            d_eng = float(v)
            break
    print(f"\n  === 与引擎对拍（{len(fl)} 帧，object={eng_object}）===")
    print(f"    MSD 曲线（{int(big.sum())}/{eng_msd.size} 个 lag，MSD > 5% 峰值）："
          f"最大相对偏差 {rel:.3e}，最大绝对偏差 {absd:.3e} Å²"
          f"   {'一致 ✅' if rel < 0.02 else '★ 偏差较大'}")
    if d_eng is not None:
        print(f"    引擎 D = {d_eng:.4e} m²/s")
    return {"msd_max_rel": rel, "msd_max_abs": absd, "D_engine": d_eng}


def menu():
    """菜单 / 命令行入口：``python test/mda_test/msd_mdat.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("mode", "追踪对象", "choice", "com",
          choices=[("com", "分子质心（MDTransport 默认口径）"),
                   ("atom", "每个原子各算一个粒子")]),
        Q("axes", "MSD 方向（求和）", "choice", "xyz",
          choices=[("xyz", "xyz → 3D MSD（6Dt）"), ("x", "只有 x（2Dt）"),
                   ("y", "只有 y（2Dt）"), ("z", "只有 z（2Dt）"),
                   ("xy", "xy 平面")]),
        Q("min_window_ps", "扩散区最小窗口 (ps，0=自动取总长一半)", "float", 0.0, lo=0.0),
        Q("compare", "再和本工具引擎对拍 MSD 曲线？", "bool", True),
    ]

    def _run(cfg):
        out = test_msd_mdat(cfg["u"], cfg["ag"], mode=cfg["mode"],
                            axes=cfg["axes"], frames=cfg["frames"],
                            min_window_ps=cfg["min_window_ps"],
                            prefix=cfg["prefix"])
        if cfg.get("compare", True):
            compare_engine(cfg["u"], cfg["ag"], mode=cfg["mode"],
                           frames=cfg["frames"], object_engine=(
                               "molecule" if cfg["mode"] == "com" else "atom"))
        return out

    return run_module("MSD / 扩散系数（MDTransport 口径）", _run, specs,
                      sub="增量最小镜像解包 + 去质心漂移 + FFT 全时间起点平均；"
                          "D = MSD 斜率/(2·dim)",
                      default_frames=200, default_prefix="msd_mdat")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
