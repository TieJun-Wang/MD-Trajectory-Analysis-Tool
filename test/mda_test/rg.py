# -*- coding: utf-8 -*-
"""回转半径 Rg（逐分子 + 帧内平均）。

口径：每个**分子**（残基）单独算 ``Rg = sqrt(Σm_i r_i² / Σm_i)``（相对该分子质心），
再对这一帧的所有分子取**算术平均**。与工具「链构象 → Rg」的逐分子平均口径一致。

单独跑：``python test/mda_test/rg.py``
"""

from __future__ import annotations

import numpy as np

try:                                  # 作为包导入：from mda_test import rg
    from ._common import frame_indices, iterate, save_csv
except ImportError:                   # 直接 python test/mda_test/rg.py
    from _common import frame_indices, iterate, save_csv


def test_rg(u, ag, frames=None, prefix="rg"):
    """测试回转半径模块：逐帧逐分子算 Rg，再对帧内所有分子取平均。

    ``frames`` 给了就只跑这些帧（等差帧号列表，菜单里选出来的）。

    导出 ``eth_rg_mda.csv``：``time(ps), avg_Rg_per_mol(Angstrom)``
    """
    mols = ag.residues
    use = frame_indices(u) if frames is None else list(frames)
    print(f"总帧数: {len(u.trajectory)}（本次 {len(use)} 帧）")
    print(f"选中原子数: {ag.n_atoms}，共 {len(mols)} 个独立分子")

    t_list, rg_list = [], []
    for ts in iterate(u, use):
        per_mol = [mol.atoms.radius_of_gyration() for mol in mols]
        t_list.append(float(ts.time))
        rg_list.append(float(np.mean(per_mol)))

    save_csv(f"{prefix}.csv", [t_list, rg_list],
             header="time(ps),avg_Rg_per_mol(Angstrom)", fmt="%.4f,%.6f")
    print(f"全轨迹平均单分子 Rg: {np.mean(rg_list):.3f} ± {np.std(rg_list):.3f} Å")
    return {"time_ps": np.asarray(t_list), "rg_mean": np.asarray(rg_list)}


def menu():
    """菜单 / 命令行入口：``python test/mda_test/rg.py``。"""
    try:
        from ._common import DEFAULT_TRR, DEFAULT_XSD, build_universe  # noqa: F401
        from ._menu import run_module
    except ImportError:
        from _common import DEFAULT_TRR, DEFAULT_XSD, build_universe  # noqa: F401
        from _menu import run_module

    def _run(cfg):
        return test_rg(cfg["u"], cfg["ag"], frames=cfg["frames"],
                    prefix=cfg["prefix"])

    return run_module("回转半径 Rg", _run,
                      sub="逐分子算 Rg，再对帧内所有分子取平均",
                      default_frames=100, default_prefix="rg")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
