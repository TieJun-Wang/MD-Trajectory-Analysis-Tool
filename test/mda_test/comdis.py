# -*- coding: utf-8 -*-
"""两组分质心距（COM distance）。

口径：每帧取两组的**质心**（按质量加权），用 ``calc_bonds`` 算**最小镜像**距离
（带 `box=ts.dimensions`）—— 与工具「质心距」默认的 PBC 口径一致。

单独跑：``python test/mda_test/comdis.py``
"""

from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from ._common import frame_indices, iterate, save_csv
except ImportError:
    from _common import frame_indices, iterate, save_csv


def test_comdis(u, ag_a=None, ag_b=None, frames=None, prefix="comdis"):
    """测试两组分质心距（``frames`` 给了就只跑这些帧）。

    不给 `ag_a/ag_b` 时按残基名统计**原子数最多的前两个组分**。
    导出 ``comdis_mda.csv``：``time(ps), comdis(A)``
    """
    from MDAnalysis.lib.distances import calc_bonds

    if ag_a is None or ag_b is None:
        com = Counter(u.atoms.resnames)
        print("含有的组分及其原子数量：")
        print(dict(com))
        if len(com) < 2:
            print("⚠️ 整体组分少于 2 个，只做单组分统计，跳过质心距")
            return None
        names = [n for n, _ in com.most_common()]
        ag_a = u.select_atoms(f"resname {names[0]}")
        ag_b = u.select_atoms(f"resname {names[1]}")

    use = frame_indices(u) if frames is None else list(frames)
    print(f"组分A 原子数 {ag_a.n_atoms}，总质量 {ag_a.masses.sum():.3f}")
    print(f"组分B 原子数 {ag_b.n_atoms}，总质量 {ag_b.masses.sum():.3f}")
    print(f"轨迹总帧数 {len(u.trajectory)}（本次 {len(use)} 帧）")

    t_list, d_list = [], []
    for ts in iterate(u, use):
        d = calc_bonds(ag_a.center_of_mass(), ag_b.center_of_mass(),
                       box=ts.dimensions)
        t_list.append(float(ts.time))
        d_list.append(float(d))

    save_csv(f"{prefix}.csv", [t_list, d_list],
             header="time(ps),comdis(A)", fmt="%.4f,%.6f")
    print(f"平均质心距 {np.mean(d_list):.3f} ± {np.std(d_list):.3f} Å")
    return {"time_ps": np.asarray(t_list), "comdist": np.asarray(d_list)}


def menu():
    """菜单 / 命令行入口：``python test/mda_test/comdis.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("select_b", "B 组选择串", "text", "",
          help="留空 = 自动取原子数次多的另一个组分；填了就按它算 A–B"),
    ]

    def _run(cfg):
        u, a = cfg["u"], cfg["ag"]
        b = u.select_atoms(cfg["select_b"]) if cfg.get("select_b") else None
        return test_comdis(u, a, b, frames=cfg["frames"],
                           prefix=cfg["prefix"])

    return run_module("两组分质心距 (COM distance)", _run, specs,
                      sub="最小镜像；B 组留空则自动取另一个组分",
                      default_frames=100, default_prefix="comdis")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
