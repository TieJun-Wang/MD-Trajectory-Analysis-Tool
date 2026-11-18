# -*- coding: utf-8 -*-
"""界面宽度（interface）—— 参考实现与交叉验证。

我们有 :func:`mdta.analysis.interface.analyze_interface_width`：沿某个盒矢量求两组分
的密度剖面 → 各自用**体相值**归一化 → 取两曲线**交点**当界面位置 → 在该交点附近
取归一化曲线从 0.9 到 0.1 的距离（10–90 宽度）→ 再用误差函数拟合给 ``2.5631σ``。

第三方没有"一键界面宽度"的函数，但这个量可以**完全独立**地重算：

1. **本文件的参考实现**（:func:`interface_reference`）：剖面用 ``density.py`` 里那套
   教科书直方图（已与 MDAnalysis ``LinearDensity`` 对齐到 1e-7），归一化 / 交点 /
   10–90 / erf 拟合全部在本文件里独立写一遍；
2. **剖面换成 MDAnalysis**（-:func:`compare_mdanalysis_interface`）：用
   ``LinearDensity`` 出的质量密度当输入，再套同一套几何判据 —— 这样"剖面来源"和
   "宽度判据"两条链路各自独立；
3. **已知宽度的合成界面**：按 ``f_A(z) = ½[1 − erf((z−z₀)/(√2σ))]`` 生成两相原子
   （每个原子先均匀采样、再按该概率判给 A/B），则真实 10–90 宽度**解析等于
   2.5631σ** —— 这是唯一能验证"宽度定义本身"的办法。

⚠️ 与界面宽度相关的两个常见坑（引擎里已处理，参考实现同样处理）：
  · **局部**测量：板层体系有两个界面，全局找 0.9/0.1 会把两个界面配成一对
    （历史上出过 89 Å 的假宽度）；
  · **周期延拓**：界面常落在盒边界上，轮廓在采样范围内没走完 0→1 的过渡。

单独跑：``python test/mda_test/interface.py``（菜单）
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, save_csv
    from .density import AMU_A3_TO_G_CM3, density_reference
except ImportError:
    from _common import frame_indices, save_csv
    from density import AMU_A3_TO_G_CM3, density_reference

#: 10–90 宽度与 erf 宽度的换算：w_1090 = 2·√2·erfinv(0.8)·σ = 2.5631σ
W1090_PER_SIGMA = 2.5631


# ------------------------------------------------------------------ 小工具
def synth_erf_profile(sigma=2.0, box=60.0, nbins=120, center=None):
    """解析的弥散界面轮廓 f(x)：上升台阶 ``½[1−erf((x−x₀)/(√2σ))]``。

    真实 10–90 宽度 = ``2.5631σ``，用来检验宽度**定义本身**（不涉及采样噪声）。
    """
    from scipy.special import erf as _erf

    x = (np.arange(int(nbins)) + 0.5) / int(nbins) * float(box)
    x0 = 0.5 * float(box) if center is None else float(center)
    f = 0.5 * (1.0 - _erf((x - x0) / (np.sqrt(2.0) * float(sigma))))
    return x, f, x0


def check_definition(verbose=True):
    """解析轮廓上的自检：10–90 宽度 == 2.5631σ、erf 拟合能还原 σ。

    这一步与体系/采样无关，专门验证"宽度判据"本身实现正确。用 0.1 Å 的细网格
    （线性插值求穿越点的离散误差 ~1e-3；0.5 Å 分箱时约为 1%，见下面的说明）。
    """
    ok = True
    rows = []
    for sigma in (1.0, 2.0, 4.0):
        x, f, x0 = synth_erf_profile(sigma=sigma, box=60.0, nbins=600)
        box = float(x[-1] - x[0] + (x[1] - x[0]))
        w = _local_width(x, f, x0, box)
        want = W1090_PER_SIGMA * sigma
        fit = _erf_width(x, f, x0, box)
        sig_fit = None if fit is None or fit.get("rejected") else fit["sigma"]
        e1 = abs(w - want) / want
        e2 = (abs(sig_fit - sigma) / sigma) if sig_fit else float("inf")
        rows.append((sigma, w, want, e1, sig_fit, e2))
        ok = ok and e1 < 5e-3 and e2 < 5e-2
    if verbose:
        print("解析轮廓自检（10–90 宽度应 = 2.5631σ；erf 拟合应还原 σ）：")
        for sigma, w, want, e1, sig_fit, e2 in rows:
            print(f"   σ={sigma:<4} 测得 {w:.4f} Å vs 理论 {want:.4f} Å"
                  f"（相对差 {e1:.1e}）；erf 拟合 σ="
                  f"{'—' if sig_fit is None else round(sig_fit, 4)}"
                  f"（相对差 {e2:.1e}）")
        print("  ✅ 宽度定义与 erf 换算一致" if ok else "  ★ 宽度定义自检未通过")
        print("  注：0.5 Å 分箱时线性插值求穿越有约 1% 的离散误差（变细网格即收敛），"
              "这是分箱本身的限制，不是判据错误。")
    return ok


def _smooth(y, n=5):
    """滑动平均（与引擎 ``_smooth`` 同宽），只压高频噪声、不改台阶位置。"""
    a = np.asarray(y, dtype=float)
    n = int(n or 0)
    if n <= 1 or a.size < n:
        return a
    k = np.ones(n) / n
    pad = n // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, k, mode="valid")[: a.size]


def _bulk(y, frac=0.25):
    """体相值 = 上四分位区的中位数（与引擎同一定义；峰值明显更高时退回峰值）。"""
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    peak = float(np.percentile(a, 99.0))
    k = max(1, int(round(a.size * frac)))
    bulk = float(np.median(np.sort(a)[-k:]))
    if not np.isfinite(bulk) or bulk <= 0:
        return peak if peak > 0 else 0.0
    return peak if (peak > 0 and bulk < 0.5 * peak) else bulk


def _level_crosses(x, y, level):
    """y(x) = level 的全部穿越点（线性插值）。"""
    xs = np.asarray(x, dtype=float)
    d = np.asarray(y, dtype=float) - float(level)
    out = []
    for k in range(xs.size - 1):
        if not (np.isfinite(d[k]) and np.isfinite(d[k + 1])):
            continue
        if d[k] == 0.0:
            out.append(float(xs[k]))
        elif d[k] * d[k + 1] < 0:
            t = -d[k] / (d[k + 1] - d[k])
            out.append(float(xs[k] + t * (xs[k + 1] - xs[k])))
    return out


def _local_width(x, f, x0, span, window_frac=0.35):
    """在界面 x0 附近**局部**取 10–90 宽度（富集侧 0.9 ↔ 另一侧 0.1）。

    窗口 0.35·span 与引擎一致；局部测量是为了不让板层的两个界面互相串扰。
    """
    w = float(window_frac) * float(span)
    if not np.isfinite(w) or w <= 0:
        return float("nan")

    def near(cs):
        return min(cs, key=lambda v: abs(v - x0)) if cs else None

    lo90 = [c for c in _level_crosses(x, f, 0.9) if c < x0 and x0 - c <= w]
    hi90 = [c for c in _level_crosses(x, f, 0.9) if c > x0 and c - x0 <= w]
    lo10 = [c for c in _level_crosses(x, f, 0.1) if c < x0 and x0 - c <= w]
    hi10 = [c for c in _level_crosses(x, f, 0.1) if c > x0 and c - x0 <= w]
    a, b = near(lo90), near(hi10)
    if a is None or b is None:
        a, b = near(hi90), near(lo10)
    if a is None or b is None:
        return float("nan")
    return float(abs(b - a))


def _erf_width(x, f, x0, span, half_window=None, max_sigma=None):
    """误差函数拟合单界面 f = ½(1 ± erf((x−x₀)/(√2σ)))，返回 σ / 2.5631σ / R²。

    只在 ``x0 ± max(span/6, 1)`` 内拟合（板层有两个界面，整条曲线拟合必然失败）；
    取向由数据自身判定（看 x0 两侧谁的平均密度高）。与引擎同一套判据：
    σ 初值取 10–90 换算值，``σ ≤ 窗宽/4`` 且 ``R² > 0.9`` 才接受。
    """
    from scipy.optimize import curve_fit
    from scipy.special import erf as _erf

    xs_all = np.asarray(x, dtype=float)
    fs_all = np.asarray(f, dtype=float)
    ok = np.isfinite(fs_all)
    xs, fs = xs_all[ok], fs_all[ok]
    if xs.size < 8:
        return None
    span = float(span) if np.isfinite(span) and span > 0 else float(xs[-1] - xs[0])
    hw = float(half_window) if half_window else max(span / 6.0, 1.0)
    m = np.abs(xs - float(x0)) <= hw
    if int(m.sum()) < 8:
        return None
    xw, fw = xs[m], fs[m]
    win = float(xw[-1] - xw[0])
    if max_sigma is None:
        max_sigma = win / 4.0
    left, right = fw[xw < float(x0)], fw[xw > float(x0)]
    sgn = 1.0 if (right.size and left.size and right.mean() > left.mean()) else -1.0
    s0 = _local_width(x, f, x0, span) / W1090_PER_SIGMA
    s0 = float(s0) if np.isfinite(s0) and s0 > 0 else win / 10.0
    lo_s, hi_s = 0.2 * s0, min(win, 5.0 * s0)
    if hi_s <= lo_s:
        hi_s = lo_s * 1.01

    def model(t, xc, sig):
        return 0.5 * (1.0 + sgn * _erf((t - xc) / (np.sqrt(2.0) * sig)))

    try:
        popt, _ = curve_fit(model, xw, fw, p0=[float(x0), s0], maxfev=40000,
                            bounds=([xw[0] - win, lo_s], [xw[-1] + win, hi_s]))
    except Exception:                                    # noqa: BLE001
        return None
    xc, sig = float(popt[0]), float(abs(popt[1]))
    pred = model(xw, xc, sig)
    ss_res = float(np.sum((fw - pred) ** 2))
    ss_tot = float(np.sum((fw - fw.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if not np.isfinite(r2) or r2 < 0.9 or sig > max_sigma:
        return {"rejected": True, "R2": r2, "sigma": sig}
    return {"x0": xc, "sigma": sig, "width_1090": W1090_PER_SIGMA * sig, "R2": r2}


def interface_reference(u, ga, gb, *, axis=2, nbins=None, binsize=0.5, frames=None,
                        mode="mass", smooth=5, verbose=True):
    """**独立实现**的界面宽度（剖面来自 :func:`density.density_reference`）。

    返回 ``{"x", "rho_a", "rho_b", "fa", "fb", "spots", "widths", "erf", "summary"}``。
    """
    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    ra = density_reference(u, ga, axis=axis, nbins=nbins, binsize=binsize,
                           frames=use, mode=mode, verbose=False)
    rb = density_reference(u, gb, axis=axis, nbins=nbins, binsize=binsize,
                           frames=use, mode=mode, verbose=False)
    x = np.asarray(ra["centers"], dtype=float)
    rho_a = np.asarray(ra["density"], dtype=float)
    rho_b = np.asarray(rb["density"], dtype=float)
    bulk_a, bulk_b = _bulk(rho_a), _bulk(rho_b)
    fa = _smooth(rho_a / bulk_a if bulk_a > 0 else np.zeros_like(rho_a), smooth)
    fb = _smooth(rho_b / bulk_b if bulk_b > 0 else np.zeros_like(rho_b), smooth)

    # 周期延拓（跨边界的界面才量得出来）
    dx = float(np.median(np.diff(x))) if x.size > 1 else 0.0
    box = dx * x.size
    xe = np.concatenate([x, x + box]) if box > 0 else x
    fae = np.concatenate([fa, fa]) if box > 0 else fa
    fbe = np.concatenate([fb, fb]) if box > 0 else fb
    span = box if box > 0 else float(x[-1] - x[0])

    # 交点 = 界面位置：在**原始**曲线上找（延拓后的副本会重复同一个交点）
    d = fa - fb
    sign = np.sign(d)
    spots = []
    for k in np.nonzero(np.diff(sign) != 0)[0]:
        d0, d1 = d[k], d[k + 1]
        if not (np.isfinite(d0) and np.isfinite(d1)):
            continue
        if d1 != d0:
            spots.append(float(x[k] + (0.0 - d0) * (x[k + 1] - x[k]) / (d1 - d0)))
        else:
            spots.append(float(0.5 * (x[k] + x[k + 1])))
    spots = sorted(spots)

    # 每个界面：A、B 两条曲线各量一次 10–90，取平均（与引擎一致）
    widths, widths_a, widths_b, erfs = [], [], [], []
    for s in spots:
        wa = _local_width(xe, fae, s, span)
        wb = _local_width(xe, fbe, s, span)
        widths_a.append(float(wa))
        widths_b.append(float(wb))
        good = [w for w in (wa, wb) if np.isfinite(w)]
        widths.append(float(np.mean(good)) if good else float("nan"))
        erfs.append(_erf_width(x, fa, s, span))

    # 1D 台阶模型适用性（与引擎同一判据）：归一化密度必须同时接近 1 与 0
    fmin_a = float(np.nanmin(fa)) if fa.size else float("nan")
    fmax_a = float(np.nanmax(fa)) if fa.size else float("nan")
    demixed = bool(np.isfinite(fmin_a) and fmin_a < 0.15 and fmax_a > 0.85)
    w0 = widths[0] if widths else float("nan")
    too_wide = bool(np.isfinite(w0) and np.isfinite(span) and w0 > 0.25 * span)
    slab_ok = bool(demixed and not too_wide)

    summary = {"组分 A": getattr(ga, "name", "A"), "组分 B": getattr(gb, "name", "B"),
               "轴": "abc"[int(axis)], "bin 数": int(x.size),
               "体相密度 A": float(bulk_a), "体相密度 B": float(bulk_b),
               "界面数": len(spots),
               "f_A 跨度": (fmax_a - fmin_a) if np.isfinite(fmin_a) else float("nan"),
               "1D 台阶模型是否适用": "是" if slab_ok else "否"}
    for i, (s, w) in enumerate(zip(spots, widths), start=1):
        tag = "界面" if i == 1 else f"界面{i} "
        summary[f"{tag}位置 (Å)"] = float(s)
        summary[f"{tag}宽度 10-90 (Å)"] = float(w)
        summary[f"{tag}宽度 10-90 (A 曲线) (Å)"] = widths_a[i - 1]
        summary[f"{tag}宽度 10-90 (B 曲线) (Å)"] = widths_b[i - 1]
        f = erfs[i - 1]
        if f is not None and not f.get("rejected"):
            summary[f"{tag}宽度 10-90 (erf) (Å)"] = float(f["width_1090"])
            summary[f"{tag}sigma (Å)"] = float(f["sigma"])
            summary[f"{tag}erf R²"] = float(f["R2"])
    if verbose:
        print(f"参考实现：{summary['轴']} 轴、{x.size} bin、{len(spots)} 个界面")
        for k, v in summary.items():
            print(f"   {k:<30} {v:.6f}" if isinstance(v, float) else f"   {k:<30} {v}")
    return {"x": x, "rho_a": rho_a, "rho_b": rho_b, "fa": fa, "fb": fb,
            "spots": spots, "widths": widths, "widths_a": widths_a,
            "widths_b": widths_b, "erf": erfs, "slab_ok": slab_ok,
            "summary": summary}


def synth_slab_universe(sigma=2.0, box=60.0, n_frames=8, n_atoms=120000,
                        seed=7, jitter=0.1):
    """生成**已知界面宽度**的合成两相体系（A 层居中，两侧各一个 erf 弥散界面）。

    做法：每个原子先均匀撒在盒内，再以 ``p(z₀) = ½[1 − erf((d−hw)/(√2σ))]``
    的概率判给 A（``d`` = 到层中心的周期距离，``hw = L/4``）。这样单帧的
    ρ_A(z) 就精确按 erf 变化，真实 10–90 宽度 = ``2.5631σ``；后续帧只加很小的
    热噪声（``σ_jitter ≪ σ``）。

    返回 ``(u, ag_a, ag_b)``，可直接喂给 :func:`interface_reference` 或引擎。
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    from scipy.special import erf as _erf

    rng = np.random.default_rng(seed)
    n = int(n_atoms)
    L = float(box)
    zc, hw = L / 2.0, L / 4.0
    base = np.empty((n, 3), dtype=float)
    base[:, :2] = rng.uniform(0.0, L, size=(n, 2))
    base[:, 2] = rng.uniform(0.0, L, size=n)
    d = np.abs(base[:, 2] - zc)
    d = np.minimum(d, L - d)
    p_a = 0.5 * (1.0 - _erf((d - hw) / (np.sqrt(2.0) * float(sigma))))
    is_a = rng.random(n) < p_a
    names = np.where(is_a, "A", "B")

    u = mda.Universe.empty(n, n_residues=n, atom_resindex=np.arange(n),
                           trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("resnames", names)
    u.add_TopologyAttr("resids", np.arange(1, n + 1))
    u.add_TopologyAttr("types", names)
    u.add_TopologyAttr("masses", np.where(is_a, 12.0, 1.0))
    frames = np.empty((int(n_frames), n, 3), dtype=np.float32)
    for k in range(int(n_frames)):
        frames[k] = base + rng.normal(0.0, jitter, size=(n, 3))
    u.trajectory = MemoryReader(
        frames, dimensions=np.tile([L, L, L, 90.0, 90.0, 90.0], (int(n_frames), 1)))
    return u, u.select_atoms("resname A"), u.select_atoms("resname B")


def test_synthetic(sigma=2.0, box=60.0, n_frames=8, n_atoms=120000, mode="number",
                   smooth=1, tol=0.08, compare=True, prefix="interface_synth"):
    """合成已知宽度界面 → 验证引擎与参考实现都能给出 ``2.5631σ``。

    ⚠️ 平滑会把窄界面加宽：``smooth=5``、0.5 Å bin 时对 σ=2 Å 的界面约 +9%，
    所以**核对宽度时用 ``smooth=1``**；真实数据为了压噪声才用 default(5)。
    """
    truth = W1090_PER_SIGMA * float(sigma)
    u, ga, gb = synth_slab_universe(sigma=sigma, box=box, n_frames=n_frames,
                                    n_atoms=n_atoms)
    ref = interface_reference(u, ga, gb, axis=2, binsize=0.5, mode=mode,
                              smooth=smooth, verbose=False)
    widths = [w for w in ref["widths"] if np.isfinite(w)]
    got = float(widths[0]) if widths else float("nan")   # 与引擎摘要口径一致（第 1 个界面）
    got2 = float(np.mean(widths[1:])) if len(widths) > 1 else float("nan")
    ok = bool(np.isfinite(got) and abs(got - truth) / truth <= tol)
    print(f"\n=== 合成已知宽度界面（σ={sigma} Å，真实 10–90 = {truth:.4f} Å）===")
    print(f"  参考实现：界面 {len(ref['spots'])} 个，位置 "
          f"{[round(s, 2) for s in ref['spots']]}，宽度 "
          f"{[round(w, 3) for w in ref['widths']]}")
    print(f"  第 1 个界面宽度 {got:.4f} Å vs 真实 {truth:.4f} Å"
          f"（相对差 {(got - truth) / truth * 100:+.2f}%）"
          f"  {'✅' if ok else f'★ 超出 {tol*100:.0f}% 容差'}")
    if np.isfinite(got2):
        print(f"  其余界面平均宽度 {got2:.4f} Å"
              f"（相对差 {(got2 - truth) / truth * 100:+.2f}%；"
              f"单帧构型的统计噪声约 ±3%）")
    save_csv(f"{prefix}.csv",
             [ref["x"], ref["fa"], ref["fb"]],
             header=("position(A),f_A,f_B  # synthetic erf slab, "
                     f"sigma={sigma}, truth_1090={truth:.4f}A, smooth={smooth}"),
             fmt="%.4f,%.6f,%.6f")
    out = {"truth": truth, "measured": got, "ok": ok}
    if compare:
        out["engine"] = compare_engine(u, ga, gb, axis=2, binsize=0.5, mode=mode,
                                       smooth=smooth)
        w_eng = out["engine"].get("w_engine")
        if w_eng is not None:
            e = abs(float(w_eng) - truth) / truth
            print(f"  引擎宽度 {float(w_eng):.4f} Å → 相对真实 {e * 100:+.2f}%"
                  f"  {'✅' if e <= tol else '★'}")
            out["engine_ok"] = bool(e <= tol)
    return out


def compare_engine(u, ga, gb, *, axis=2, nbins=None, binsize=0.5, frames=None,
                   mode="mass", smooth=5):
    """与工具引擎 ``analyze_interface_width`` 比界面位置/宽度（同一批帧、同一 bin）。"""
    from types import SimpleNamespace

    from mdta.analysis.interface import analyze_interface_width
    from mdta.io import MDTrajectory

    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    ref = interface_reference(u, ga, gb, axis=axis, nbins=nbins, binsize=binsize,
                              frames=use, mode=mode, smooth=smooth, verbose=False)
    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(use, dtype=int),
                          times_ps=times[np.asarray(use, dtype=int)],
                          n_frames=len(use), describe=f"{len(use)} 帧", notes=[],
                          head=lambda self, k: self)
    res = analyze_interface_width(mdt, {"A": ga, "B": gb}, sel, pair=("A", "B"),
                                  axis=int(axis), nbins=int(ref["x"].size),
                                  mode=mode, smooth=int(smooth))
    got = res.summary
    w_eng = got.get("界面宽度 10-90 (Å)")
    w_ref = ref["widths"][0] if ref["widths"] else None
    out = {"w_engine": w_eng, "w_ref": w_ref, "pos_engine": got.get("界面位置 (Å)"),
           "slab_ok": got.get("1D 台阶模型是否适用"), "ref_slab_ok": ref["slab_ok"]}
    print(f"\n=== 与引擎对拍（{len(use)} 帧，nbins={ref['x'].size}）===")
    if w_eng is None:
        why = (got.get("界面判据可靠性")
               or (getattr(res, "notes", None) or ["（引擎未识别出界面）"])[0])
        print(f"  引擎判为「1D 台阶模型不适用」：{str(why)[:140]}")
        print(f"  参考实现同样给出："
              f"{'不适用' if not ref['slab_ok'] else '适用（引擎与参考不一致，需排查）'}"
              f"（界面数 {len(ref['spots'])}）")
    if w_eng is not None and w_ref is not None and np.isfinite(w_ref):
        d = abs(float(w_eng) - float(w_ref))
        print(f"  10–90 宽度：引擎 {float(w_eng):.4f} Å vs 参考 {float(w_ref):.4f} Å"
              f"（差 {d:.3e} Å，相对 {d / max(abs(float(w_ref)), 1e-12):.2e}）"
              f"  {'一致 ✅' if d / max(abs(float(w_ref)), 1e-12) < 0.02 else '★ 偏差较大'}")
        out["dw"] = d
    if got.get("界面位置 (Å)") is not None and ref["spots"]:
        dp = abs(float(got["界面位置 (Å)"]) - float(ref["spots"][0]))
        print(f"  界面位置：引擎 {float(got['界面位置 (Å)']):.4f} Å vs 参考 "
              f"{float(ref['spots'][0]):.4f} Å（差 {dp:.3e} Å）")
        out["dpos"] = dp
    # 第二个界面（板层体系）宽度也核对一下
    w_eng2 = got.get("界面2 界面宽度 10-90 (Å)")
    w_ref2 = ref["widths"][1] if len(ref["widths"]) > 1 else None
    if w_eng2 is not None and w_ref2 is not None and np.isfinite(w_ref2):
        d2 = abs(float(w_eng2) - float(w_ref2))
        print(f"  第二界面宽度：引擎 {float(w_eng2):.4f} Å vs 参考 {float(w_ref2):.4f} Å"
              f"（差 {d2:.3e} Å）")
        out["dw2"] = d2
    return out


def compare_mdanalysis_interface(u, ga, gb, *, axis=2, nbins=None, binsize=0.5,
                                 frames=None):
    """剖面换成 **MDAnalysis LinearDensity**，再套本文件的 10–90 判据。

    这样"剖面来源"（MDA）与"宽度判据"（本文件）都与引擎独立。
    """
    try:
        from MDAnalysis.analysis.lineardensity import LinearDensity
    except Exception as exc:                                  # noqa: BLE001
        print(f"  （MDAnalysis LinearDensity 不可用：{exc}）")
        return None

    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    u.trajectory[use[0]]
    L = np.asarray(u.dimensions, dtype=float)[:3]
    Lmax = float(np.max(L))
    if nbins is None:
        nbins = max(10, int(round(float(L[int(axis)]) / float(binsize))))
    bs = float(L[int(axis)]) / int(nbins)
    if abs(float(L[int(axis)]) - Lmax) > 1e-6:
        print(f"  ⚠️ LinearDensity 的直方图范围恒为最长边，{('abc')[int(axis)]} 不是最长边 → 跳过")
        return None
    try:
        u.atoms.charges
    except Exception:                                        # noqa: BLE001
        u.add_TopologyAttr("charges", np.zeros(u.atoms.n_atoms))

    step = max(1, int(round((use[-1] - use[0]) / max(len(use) - 1, 1))))
    profs = {}
    for tag, ag in (("A", ga), ("B", gb)):
        ld = LinearDensity(ag, binsize=bs)
        ld.run(start=use[0], stop=use[-1] + 1, step=step)
        res = ld.results["xyz"[int(axis)]]
        edges = np.asarray(res.hist_bin_edges, dtype=float)
        profs[tag] = (0.5 * (edges[:-1] + edges[1:]),
                      np.asarray(res.mass_density, dtype=float))
    x = profs["A"][0]
    fa, fb = profs["A"][1], profs["B"][1]
    ba, bb = _bulk(fa), _bulk(fb)
    fAn = _smooth(fa / ba if ba > 0 else np.zeros_like(fa))
    fBn = _smooth(fb / bb if bb > 0 else np.zeros_like(fb))

    span = float(x[-1] - x[0]) + float(np.median(np.diff(x)))
    d = fAn - fBn
    sign = np.sign(d)
    spots = []
    for k in np.nonzero(np.diff(sign) != 0)[0]:
        d0, d1 = d[k], d[k + 1]
        if np.isfinite(d0) and np.isfinite(d1) and d1 != d0:
            spots.append(float(x[k] + (0.0 - d0) * (x[k + 1] - x[k]) / (d1 - d0)))
    widths = [_local_width(x, fAn, s, span) for s in spots]
    print(f"\n=== MDAnalysis 剖面 + 本判据（{len(use)} 帧，{x.size} bin）===")
    print(f"  界面 {len(spots)} 个：位置 {np.round(spots, 3).tolist()}")
    if widths:
        print(f"  10–90 宽度：{np.round(widths, 4).tolist()} Å")
    return {"spots": spots, "widths": widths,
            "x": x, "fa": fAn, "fb": fBn}


def test_interface(u, ga, gb, *, axis=2, nbins=None, binsize=0.5, frames=None,
                   mode="mass", smooth=1, prefix="interface", compare=True):
    """算界面位置/宽度 + 导出 CSV + （可选）与引擎、MDAnalysis 对拍。"""
    ref = interface_reference(u, ga, gb, axis=axis, nbins=nbins, binsize=binsize,
                              frames=frames, mode=mode, smooth=smooth)
    save_csv(f"{prefix}.csv",
             [ref["x"], ref["rho_a"], ref["rho_b"], ref["fa"], ref["fb"]],
             header=("position(A),rho_A,rho_B,f_A,f_B  # "
                     f"axis={'abc'[int(axis)]}; nbins={ref['x'].size}; mode={mode}"),
             fmt="%.4f,%.6f,%.6f,%.6f,%.6f")
    if compare:
        compare_engine(u, ga, gb, axis=axis, nbins=ref["x"].size,
                       frames=frames, mode=mode, smooth=smooth)
        compare_mdanalysis_interface(u, ga, gb, axis=axis, nbins=ref["x"].size,
                                     frames=frames)
    return ref


def menu():
    """菜单 / 命令行入口：``python test/mda_test/interface.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("synthetic", "先跑合成已知宽度界面自检？", "bool", True),
        Q("axis", "界面法向", "choice", "2",
          choices=[("2", "c 轴（z）"), ("1", "b 轴（y）"), ("0", "a 轴（x）")]),
        Q("nbins", "bin 数（0 = 按 bin 宽自动）", "int", 100, lo=0),
        Q("binsize", "bin 宽 (Å，nbins=0 时生效)", "float", 0.5, lo=0.0),
        Q("mode", "密度口径", "choice", "number",
          choices=[("number", "数密度 1/Å³"), ("mass", "质量密度 g/cm³")]),
        Q("smooth", "平滑窗口（核对绝对宽度用 1；真实数据可 5）", "int", 1, lo=1),
        Q("select_b", "B 组选择串（留空 = 自动取另一个组分）", "text", ""),
        Q("compare", "同时与引擎 / MDAnalysis 对拍？", "bool", True),
    ]

    def _run(cfg):
        if cfg.get("synthetic", True):
            check_definition()
            test_synthetic(mode=cfg.get("mode", "number"), smooth=1)
            print()
        u, a = cfg["u"], cfg["ag"]
        try:
            from ._common import auto_pair
        except ImportError:
            from _common import auto_pair
        b = u.select_atoms(cfg["select_b"]) if cfg.get("select_b") else None
        if b is None:
            _, b, _, _ = auto_pair(u)
            if b is None or b.n_atoms == 0:
                print("⚠️ 需要两个不同的组分才能算界面；请用「分析集合」把两相拆开，"
                      "或在 --select-b 里指定 B 组")
                return None
            if np.array_equal(np.sort(np.asarray(b.indices)),
                              np.sort(np.asarray(a.indices))):
                print("⚠️ 该体系只有一个组分（A、B 取到同一组原子）→ 界面分析必然判为"
                      "「1D 台阶模型不适用」，下面的结果只作对照；要量界面宽度请换分层"
                      "体系，或在「分析集合」里把两相拆开。")
        nb = int(cfg.get("nbins") or 0)
        return test_interface(u, a, b, axis=int(cfg["axis"]), nbins=(nb or None),
                              binsize=(cfg["binsize"] or None),
                              frames=cfg["frames"], mode=cfg["mode"],
                              smooth=int(cfg.get("smooth") or 1),
                              prefix=cfg["prefix"],
                              compare=bool(cfg.get("compare", True)))

    return run_module("界面宽度（10–90 + erf）", _run, specs,
                      sub="参考实现：教科书剖面 + 局部 10–90 + erf 拟合；"
                          "与引擎、MDAnalysis 剖面双向对拍",
                      default_frames=10, default_prefix="interface")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
