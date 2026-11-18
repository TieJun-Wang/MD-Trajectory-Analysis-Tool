# -*- coding: utf-8 -*-
"""密度分布（density）—— 用 MDAnalysis 与"教科书实现"交叉验证。

我们的口径（``mdta/analysis/interface.py::density_profile``）
------------------------------------------------------------
把原子换算成沿某个**盒矢量**的分数坐标（0–1）→ 在分数空间做直方图 →
用整个时间窗口的**平均盒长/平均盒体积**把 bin 换算成 Å 与 Å³ →
质量密度 ``ρ = Σm /(V_bin) × 1.66054``（g/cm³）或数密度（Å⁻³）。
用分数坐标的好处：NPT 里盒子漂移不会让 bin 错位。

验证手段（三条，互相独立）
--------------------------
1. **教科书实现**（本文件的 :func:`density_reference`）：包裹坐标 → 沿轴直方图 →
   ``amu/Å³ × 1.66054``。与我们的引擎**逐位相同**（同约定时偏差 0）；
2. **MDAnalysis ``LinearDensity``**（:func:`compare_mdanalysis`）：
   ``mass_density`` 单位就是 g/cm³，computed 每帧包裹后直方图；
   ⚠️ 它有两个已知限制：
     · ``range=(0, max(盒边))`` 对**所有**轴都用最长边 → 非最长轴（例如
       24.66×24.66×49.19 的盒子里的 x、y）网格被**拉长**，只能比较**最长轴**；
     · 盒尺寸取**初始化时那一帧**（NPT 下与我们用平均值略有差别）；
3. 与**质量守恒**自检：Σρ·V_bin 应等于体系总质量。

单独跑：``python test/mda_test/density.py``（菜单）
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, save_csv
except ImportError:
    from _common import frame_indices, save_csv

#: 1 amu/Å³ = 1.66053906660 g/cm³
AMU_A3_TO_G_CM3 = 1.66053906660


def _frames(u, frames):
    return frame_indices(u) if frames is None else [int(f) for f in frames]


def _axis_len(u, axis):
    d = np.asarray(u.dimensions, dtype=float)
    return float(d[:3][int(axis)])


def density_reference(u, ag=None, *, axis=2, nbins=None, binsize=0.5,
                      frames=None, mode="mass", wrap="floor", verbose=True):
    """**教科书实现**的 1D 密度分布（独立于引擎，用来验证它）。

    ``wrap``：
      · ``"floor"``（默认）：分数坐标取模 —— 与我们的引擎同一约定；
      · ``"center"``：MDAnalysis ``AtomGroup.wrap(compound="atoms")`` 的约定
        （把原子中心折进盒内，边界原子可能落进相邻 bin）。

    ``mode``：``"mass"``（g/cm³）/ ``"number"``（Å⁻³）。
    返回 ``{"centers", "density", "n_frames", "volume", "bin_volume", "total_mass",
    "total_number"}``。
    """
    a = u.atoms if ag is None else ag
    use = _frames(u, frames)
    axis = int(axis)
    if nbins is None:
        L0 = _axis_len(u, axis)
        nbins = max(10, int(round(L0 / float(binsize))))
    nbins = int(nbins)
    masses = None
    if str(mode).lower() == "mass":
        try:
            masses = np.asarray(a.masses, dtype=float)
            if masses.size != a.n_atoms or masses.sum() <= 0:
                masses = None
        except Exception:                                # noqa: BLE001
            masses = None
    w = (masses if masses is not None
         else np.ones(a.n_atoms, dtype=float))

    acc = np.zeros(nbins, dtype=float)
    vols, lens = [], []
    for fi in use:
        ts = u.trajectory[fi]
        L = np.asarray(ts.dimensions, dtype=float)[:3]
        if not np.all(L > 0):
            raise SystemExit("密度分布需要周期盒（当前帧没有盒子信息）")
        vols.append(float(np.prod(L)))
        lens.append(float(L[axis]))
        pos = np.asarray(a.positions, dtype=float)
        if str(wrap).lower() == "center":
            a.wrap(compound="atoms")
            pos = np.asarray(a.positions, dtype=float)
            x = pos[:, axis]
            edges = np.linspace(0.0, float(L[axis]), nbins + 1)
        else:
            x = (pos[:, axis] / float(L[axis])) % 1.0
            edges = np.linspace(0.0, 1.0, nbins + 1)
        h, _ = np.histogram(x, bins=edges, weights=w)
        acc += h

    n_f = max(len(use), 1)
    mean_len = float(np.mean(lens))
    mean_vol = float(np.mean(vols))
    vbin = mean_vol / nbins
    per = acc / n_f
    if str(mode).lower() == "mass":
        dens = per / vbin * AMU_A3_TO_G_CM3
    else:
        dens = per / vbin
    centers = (np.arange(nbins) + 0.5) / nbins * mean_len
    if verbose:
        print(f"教科书密度（{write_axis(axis)} 轴，{nbins} bin，{n_f} 帧，"
              f"{'质量' if masses is not None else '数'}密度）："
              f"平均 {dens.mean():.6f}")
    return {"centers": centers, "density": dens, "n_frames": n_f,
            "volume": mean_vol, "bin_volume": vbin,
            "total_mass": float(w.sum()), "total_number": int(a.n_atoms),
            "axis": axis, "nbins": nbins, "wrap": wrap}


def write_axis(axis) -> str:
    return "abc"[int(axis)]


def compare_mdanalysis(u, ag=None, *, axis=2, nbins=None, binsize=0.5,
                       frames=None, mode="mass", allow_stretched=False):
    """与 **MDAnalysis ``LinearDensity``** 对拍（单位同为 g/cm³）。

    ⚠️ ``LinearDensity`` 对**所有**轴都用 ``range=(0, max(盒边))``，所以只有
    **最长轴**的网格与物理坐标一致；其它轴的 bin 被拉长（数据只占前一半）。
    非最长轴默认**跳过并说明**；``allow_stretched=True`` 时仍然比（用物理位置
    把 MDA 的曲线插值到我们的 bin 中心）。
    """
    from MDAnalysis.analysis.lineardensity import LinearDensity as LD

    a = u.atoms if ag is None else ag
    use = _frames(u, frames)
    axis = int(axis)
    u.trajectory[use[0]]
    L = np.asarray(u.dimensions, dtype=float)[:3]
    Lmax = float(np.max(L))
    if nbins is None:
        nbins = max(10, int(round(float(L[axis]) / float(binsize))))
    bs = float(L[axis]) / int(nbins)

    if abs(float(L[axis]) - Lmax) > 1e-6 and not allow_stretched:
        print(f"  ⚠️ LinearDensity 在 {write_axis(axis)} 方向（长度 {L[axis]:.3f} Å）"
              f"不是最长边（{Lmax:.3f} Å）：它的直方图范围恒为 [0, 最长边]，"
              f"这个方向的 bin 会被拉长 → 本方向跳过对拍。"
              f"（想看就用最长轴，或 allow_stretched=True 看趋势）")
        return None

    mine = density_reference(u, a, axis=axis, nbins=int(nbins), frames=use,
                             mode=mode, wrap="floor", verbose=False)
    # LinearDensity 这个版本在 __init__ 里就读 charges；没有就补 0
    try:
        a.charges
    except Exception:                                    # noqa: BLE001
        a.universe.add_TopologyAttr("charges", np.zeros(a.universe.atoms.n_atoms))

    ld = LD(a, binsize=bs)
    step = max(1, int(round((use[-1] - use[0]) / max(len(use) - 1, 1))))
    ld.run(start=use[0], stop=use[-1] + 1, step=step)
    res = ld.results["xyz"[axis]]
    ref = np.asarray(res.mass_density, dtype=float)
    edges = np.asarray(res.hist_bin_edges, dtype=float)
    c_ref = 0.5 * (edges[:-1] + edges[1:])

    ours = np.asarray(mine["density"], dtype=float)
    # ⚠️ LinearDensity 只给**质量密度**：数密度口径要换算
    #    ρ_number[1/Å³] = ρ_mass[g/cm³] / (1.66054 × 平均原子质量[amu])
    if str(mode).lower() != "mass":
        try:
            m_mean = float(np.mean(np.asarray(a.masses, dtype=float)))
        except Exception:                                 # noqa: BLE001
            m_mean = 0.0
        if m_mean <= 0:
            print("  （数密度口径需要质量才能把 LinearDensity 的质量密度换算过来，跳过）")
            return None
        ref = ref / (AMU_A3_TO_G_CM3 * m_mean)
    # MDA 的 bin 网格更宽（range 到最长边）：按**物理位置**插值到我们的中心
    if c_ref.size != ours.size or not np.allclose(c_ref, mine["centers"], atol=1e-6):
        ours_cmp = np.interp(mine["centers"], c_ref, ref,
                             left=np.nan, right=np.nan)
    else:
        ours_cmp = ref
    ok = np.isfinite(ours_cmp)
    if not ok.any():
        print("  ⚠️ 两个网格没有重叠区，无法对拍")
        return None
    absd = float(np.max(np.abs(mine["density"][ok] - ours_cmp[ok])))
    scale = max(float(np.nanmean(np.abs(ours_cmp[ok]))), 1e-12)
    print(f"  vs MDAnalysis LinearDensity：{int(ok.sum())}/{ours.size} 个 bin，"
          f"最大绝对差 {absd:.3e} g/cm³（相对 {absd / scale:.3e}），"
          f"平均 {mine['density'][ok].mean():.6f} vs {ours_cmp[ok].mean():.6f}")
    return {"abs": absd, "rel": absd / scale, "n": int(ok.sum()),
            "rho_mda": ours_cmp, "centers": mine["centers"]}


def compare_engine(u, ag=None, *, axis=2, nbins=None, binsize=0.5, frames=None,
                   mode="mass"):
    """与本工具**引擎**（``analyze_density``）对拍：面板 1（剖面）+ 面板 2（ρ/ρ̄）。

    走的就是界面用的那条代码路径（单轴分支），所以能被用户直接拿去核对导出的
    ``density.csv`` / ``density__norm.csv``。
    """
    from types import SimpleNamespace

    from mdta.analysis.interface import analyze_density
    from mdta.io import MDTrajectory

    a = u.atoms if ag is None else ag
    use = _frames(u, frames)
    mine = density_reference(u, a, axis=axis, nbins=nbins, binsize=binsize,
                             frames=use, mode=mode, verbose=False)
    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(use, dtype=int),
                          times_ps=times[np.asarray(use, dtype=int)],
                          n_frames=len(use), describe=f"{len(use)} 帧",
                          notes=[], head=lambda self, k: self)
    res = analyze_density(mdt, {"X": a}, sel, axis=int(axis),
                          nbins=int(mine["nbins"]), mode=mode)
    prof = None
    norm = None
    for c in res.curves:
        if c.panel == 0 and prof is None:
            prof = np.asarray(c.y, dtype=float)
        elif c.panel == 1 and norm is None:
            norm = np.asarray(c.y, dtype=float)
    out = {"abs": None, "abs_norm": None, "n": 0}
    if prof is not None:
        n = min(prof.size, mine["density"].size)
        d = np.abs(prof[:n] - mine["density"][:n])
        out.update(abs=float(d.max()), n=int(n))
        print(f"  vs 工具引擎（面板 1，密度剖面）：{n} 个 bin，最大绝对差 {d.max():.3e}，"
              f"{'逐位一致 ✅' if d.max() < 1e-12 else '★ 不一致'}")
    if norm is not None:
        rb = rho_bar(a, mine["volume"], mode)
        n = min(norm.size, mine["density"].size)
        if rb > 0:
            mine_n = mine["density"][:n] / rb
            dn = float(np.max(np.abs(norm[:n] - mine_n)))
            out["abs_norm"] = dn
            print(f"  vs 工具引擎（面板 2，ρ/ρ̄）：最大绝对差 {dn:.3e}，"
                  f"{'逐位一致 ✅' if dn < 1e-12 else '★ 不一致'}"
                  f"；ρ̄ = {rb:.6f}")
    return out


def rho_bar(ag, vbox: float, mode="mass") -> float:
    """整盒平均密度 ρ̄ —— **与剖面同单位**（就是程序里归一化用的基准）。

    mass → Σm[amu]/V[Å³] × 1.66054 = g/cm³；number → N/V = 1/Å³。
    """
    v = float(vbox or 0.0)
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    if str(mode).lower() == "mass":
        return float(np.sum(np.asarray(ag.masses, dtype=float))) / v * AMU_A3_TO_G_CM3
    return float(ag.n_atoms) / v


def test_density(u, ag=None, *, axis=2, nbins=None, binsize=0.5, frames=None,
                 mode="mass", prefix="density", compare=True, export_norm=True):
    """算 1D 密度分布 + **归一化曲线** + 导出 CSV + （可选）与引擎/MDAnalysis 对拍。

    导出的两个文件和程序「导出板块」完全同口径，可以直接逐值比对：

    · ``{prefix}.csv``       → ``position(A), density``（面板 1：ρ）
    · ``{prefix}__norm.csv`` → ``position(A), rho_over_rbar``（面板 2：ρ/ρ̄）

    ⚠️ 归一化基准是 **ρ̄（整盒平均密度）**，不是「体相密度」（上四分位中位数）——
    两者在 20PE30 上差 14–16%，拿体相去核对会得到 ~0.86 的假偏差。
    """
    ref = density_reference(u, ag, axis=axis, nbins=nbins, binsize=binsize,
                            frames=frames, mode=mode)
    save_csv(f"{prefix}.csv", [ref["centers"], ref["density"]],
             header=("position(A),density  # "
                     + ("g/cm^3" if str(mode).lower() == "mass" else "1/A^3")
                     + f"; axis={'abc'[int(axis)]}; nbins={ref['nbins']}"),
             fmt="%.4f,%.6f")
    a0 = u.atoms if ag is None else ag
    rb = rho_bar(a0, ref["volume"], mode)
    norm = (np.asarray(ref["density"], dtype=float) / rb) if rb > 0 else None
    if export_norm and norm is not None:
        save_csv(f"{prefix}__norm.csv", [ref["centers"], norm],
                 header=("position(A),rho_over_rbar  # "
                         f"rho_bar={rb:.6f}（整盒平均密度，"
                         + ("g/cm^3" if str(mode).lower() == "mass" else "1/A^3")
                         + "）；与程序面板 2 同口径"),
                 fmt="%.4f,%.6f")
    d = np.asarray(ref["density"], dtype=float)
    a = a0
    vol = ref["volume"]
    print(f"  bin 数 {ref['nbins']}，平均盒体积 {vol:.2f} Å³，"
          f"密度：均 {d.mean():.6f} / 最小 {d.min():.6f} / 最大 {d.max():.6f}")
    if norm is not None:
        print(f"  ρ̄（整盒平均，归一化基准）= {rb:.6f}；"
              f"归一化 ρ/ρ̄：均 {norm.mean():.6f} / 最小 {norm.min():.6f} / "
              f"最大 {norm.max():.6f}")
    if str(mode).lower() == "mass":
        try:
            m_tot = float(np.sum(a.masses))
        except Exception:                                # noqa: BLE001
            m_tot = float(ref["total_mass"])
        # ρ[g/cm³] × V_bin[Å³] ÷ 1.66054 = 该 bin 的 amu → 全部加起来应等于总质量
        m_back = float(d.sum()) * float(ref["bin_volume"]) / AMU_A3_TO_G_CM3
        rel = abs(m_back - m_tot) / max(m_tot, 1e-12)
        print(f"  质量守恒自检：由密度积回的总质量 {m_back:.3f} amu，"
              f"体系总质量 {m_tot:.3f} amu（相对差 {rel:.2e}）"
              f"{'  ✅' if rel < 1e-6 else '  ★ 不一致'}")
    if compare:
        compare_engine(u, a, axis=axis, nbins=ref["nbins"], frames=frames,
                       mode=mode)
        compare_mdanalysis(u, a, axis=axis, nbins=ref["nbins"], frames=frames,
                           mode=mode)
    return ref


def menu():
    """菜单 / 命令行入口：``python test/mda_test/density.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("axis", "密度方向", "choice", "2",
          choices=[("2", "c 轴（z，第 3 个盒矢量）"),
                   ("1", "b 轴（y）"), ("0", "a 轴（x）")]),
        Q("nbins", "bin 数（0 = 按 bin 宽自动）", "int", 100, lo=0,
          help="程序界面上填的「密度bin」就是这个；0 则按下面的 bin 宽算"),
        Q("binsize", "bin 宽 (Å，nbins=0 时生效)", "float", 0.5, lo=0.0,
          help="与 Forcite Concentration Profile 的默认 0.5 Å 一致"),
        Q("mode", "密度口径", "choice", "number",
          choices=[("number", "数密度 1/Å³（Forcite 的 Relative concentration 用它）"),
                   ("mass", "质量密度 g/cm³")]),
        Q("compare", "同时与引擎 / MDAnalysis LinearDensity 对拍？", "bool", True),
    ]

    def _run(cfg):
        nb = int(cfg.get("nbins") or 0)
        return test_density(cfg["u"], cfg["ag"], axis=int(cfg["axis"]),
                            nbins=(nb or None),
                            binsize=(cfg["binsize"] or None),
                            frames=cfg["frames"], mode=cfg["mode"],
                            prefix=cfg["prefix"],
                            compare=bool(cfg.get("compare", True)))

    return run_module("密度分布（1D）", _run, specs,
                      sub="分数坐标直方图 + 平均盒体积归一化；"
                          "与教科书实现逐位一致、与 LinearDensity 差 ~1e-7",
                      default_frames=10, default_prefix="density")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
