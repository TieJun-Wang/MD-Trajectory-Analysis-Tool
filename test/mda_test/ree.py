# -*- coding: utf-8 -*-
"""末端距 R_ee（每个分子首尾原子间距离 + 帧内平均）。

口径：对每个分子取**第一个原子**与**最后一个原子**的欧氏距离（不展开、不折叠 ——
与工具「链构象 → R_ee」的默认做法一致；要最小镜像请自己乘上盒子再算）。

单独跑：``python test/mda_test/ree.py``
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_indices, iterate, save_csv
except ImportError:
    from _common import frame_indices, iterate, save_csv


def test_ree(u, ag, frames=None, prefix="ree"):
    """测试末端距模块（``frames`` 给了就只跑这些帧）。

    导出 ``eth_ree_mda.csv``：``time(ps), avg_End2End(Angstrom)``
    """
    mols = ag.residues
    use = frame_indices(u) if frames is None else list(frames)
    print(f"组内含有分子数量: {len(mols)}；本次 {len(use)} 帧")
    t_list, re_list = [], []
    for ts in iterate(u, use):
        t_list.append(float(ts.time))
        per_mol = []
        for mol in mols:
            head = mol.atoms[0].position
            tail = mol.atoms[-1].position
            per_mol.append(float(np.linalg.norm(tail - head)))   # 纯 numpy 欧氏距离
        re_list.append(float(np.mean(per_mol)))

    save_csv(f"{prefix}.csv", [t_list, re_list],
             header="time(ps),avg_End2End(Angstrom)", fmt="%.4f,%.6f")
    print(f"全轨迹平均 R_ee: {np.mean(re_list):.3f} ± {np.std(re_list):.3f} Å")
    return {"time_ps": np.asarray(t_list), "ree_mean": np.asarray(re_list)}


def menu():
    """菜单 / 命令行入口：``python test/mda_test/ree.py``。"""
    try:
        from ._menu import run_module
    except ImportError:
        from _menu import run_module

    def _run(cfg):
        return test_ree(cfg["u"], cfg["ag"], frames=cfg["frames"],
                     prefix=cfg["prefix"])

    return run_module("末端距 R_ee", _run,
                      sub="每个分子首尾原子的距离，再对分子取平均",
                      default_frames=100, default_prefix="ree")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
