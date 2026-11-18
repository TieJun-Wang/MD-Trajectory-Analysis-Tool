# -*- coding: utf-8 -*-
"""MSD（MDAnalysis 的 ``EinsteinMSD``，FFT 版）。

口径：`EinsteinMSD(ag, select='all', msd_type='xyz', fft=True)`，MSD 单位 **nm²**、
时间单位 **ps**（MDAnalysis 内部把 Å 换成 nm），与工具「MSD」的 τ 轴同一含义。

单独跑：``python test/mda_test/msd.py``
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_window, frames_window, save_csv
except ImportError:
    from _common import frame_window, frames_window, save_csv


def test_msd(u, ag, dt_per_frame_ps=10, fit_start_ns=0.05, fit_end_ns=0.2,
             frames=None, prefix="msd"):
    """测试 MSD 模块（可选在给定窗口里拟合扩散系数）。

    ``frames`` 给了就只跑这些帧（``EinsteinMSD`` 只能按 start/stop/step 取帧，
    菜单里挑的帧保证是等差的）。导出 ``eth_msd_mda.csv``：``time(ps), MSD(nm^2)``
    """
    from MDAnalysis.analysis.msd import EinsteinMSD

    w = frames_window(frames, u) if frames is not None else frame_window(u)
    print(f"总帧数: {len(u.trajectory)}（用 start={w['start']} stop={w['stop']} "
          f"step={w['step']}）")
    print(f"选中原子数: {ag.n_atoms}")

    msd_anal = EinsteinMSD(ag, select="all", msd_type="xyz", fft=True)
    msd_anal.run(start=w["start"], stop=w["stop"], step=w["step"])

    tau_ps = np.asarray(msd_anal.results.delta_t_values, dtype=float)
    msd = np.asarray(msd_anal.results.timeseries, dtype=float)
    save_csv(f"{prefix}.csv", [tau_ps, msd],
             header="time(ps),MSD(nm^2)", fmt="%.4f,%.6f")

    # 可选：在 [fit_start_ns, fit_end_ns] 里做线性拟合，D = 斜率 / 6（三维）
    sel = (tau_ps >= fit_start_ns * 1000.0) & (tau_ps <= fit_end_ns * 1000.0)
    if int(sel.sum()) >= 3:
        k, b = np.polyfit(tau_ps[sel], msd[sel], 1)
        print(f"拟合窗口 {fit_start_ns}–{fit_end_ns} ns：斜率 {k:.6g} nm²/ps "
              f"→ D = {k / 6.0 * 1000:.4g} ×10⁻⁹ m²/s（三维、MSD=6Dt）")
    return {"tau_ps": tau_ps, "msd_nm2": msd}


def menu():
    """菜单 / 命令行入口：``python test/mda_test/msd.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("fit_start_ns", "拟合窗口起点 (ns)", "float", 0.05, lo=0.0,
          help="只用这段 τ 范围拟合扩散系数"),
        Q("fit_end_ns", "拟合窗口终点 (ns)", "float", 0.2, lo=0.0),
    ]

    def _run(cfg):
        return test_msd(cfg["u"], cfg["ag"], fit_start_ns=cfg["fit_start_ns"],
                        fit_end_ns=cfg["fit_end_ns"], frames=cfg["frames"],
                        prefix=cfg["prefix"])

    return run_module("MSD 均方位移（EinsteinMSD）", _run, specs,
                      sub="FFT 版；同时在指定窗口里拟合扩散系数",
                      default_frames=100, default_prefix="msd")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
