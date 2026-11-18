# -*- coding: utf-8 -*-
"""径向分布函数 RDF（MDAnalysis 的 ``InterRDF``）。

口径：``InterRDF(ag, ag, range=(0, r_max), nbins=nbins, exclusion_block=(1,1))``
—— 同组 A–A、**排除自身配对**（`exclusion_block=(1,1)` 就是"不含 i==i"）。

⚠️ 与工具「RDF」的差别：工具默认 `mode="inter"`（剔除同分子配对），而
`InterRDF` 只排除自配对、**仍含同分子对**（≈ 工具的 `total`）。要和工具的
`inter` 比，得自己按分子编号过滤，或用 `examples/contact_reproduce.py` 的做法。

单独跑：``python test/mda_test/rdf.py``
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import frame_window, frames_window, save_csv
except ImportError:
    from _common import frame_window, frames_window, save_csv


def test_rdf(u, ag, r_max_nm=1.2, nbins=120, frames=None, prefix="rdf_mda"):
    """测试 RDF 模块。

    ``frames`` 给了就只跑这些帧（``InterRDF`` 只能按 start/stop/step 取帧）。
    导出 ``eth_rdf_mda.csv``：``r(0.1nm), g(r)``（r 单位 Å，等于 nm×10）
    """
    from MDAnalysis.analysis.rdf import InterRDF

    w = frames_window(frames, u) if frames is not None else frame_window(u)
    r_max_ang = float(r_max_nm) * 10.0
    print(f"总帧数: {len(u.trajectory)}（用 start={w['start']} stop={w['stop']} "
          f"step={w['step']}）")
    print(f"选中原子数: {ag.n_atoms}，共 {len(ag.residues)} 个独立分子；"
          f"r_max={r_max_ang} Å，{nbins} 个 bin")

    rdf_anal = InterRDF(ag, ag, range=(0.0, r_max_ang), nbins=nbins,
                        exclusion_block=(1, 1))
    rdf_anal.run(start=w["start"], stop=w["stop"], step=w["step"])

    r = np.asarray(rdf_anal.results.bins, dtype=float)
    g_r = np.asarray(rdf_anal.results.rdf, dtype=float)
    save_csv(f"{prefix}.csv", [r, g_r], header="r(A),g(r)", fmt="%.4f,%.6f")

    first = int(np.argmax(g_r))
    print(f"首峰 r = {r[first]:.3f} Å，g = {g_r[first]:.3f}")
    return {"r": r, "g_r": g_r}


def menu():
    """菜单 / 命令行入口：``python test/mda_test/rdf.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("r_max_nm", "最大 r (nm)", "float", 1.2, lo=0.05,
          help="1.2 nm = 12 Å；也可以写 0.6 只看第一壳层"),
        Q("nbins", "分箱数", "int", 120, lo=5),
    ]

    def _run(cfg):
        return test_rdf(cfg["u"], cfg["ag"], r_max_nm=cfg["r_max_nm"],
                        nbins=cfg["nbins"], frames=cfg["frames"],
                        prefix=cfg["prefix"])

    return run_module("径向分布函数 RDF（InterRDF）", _run, specs,
                      sub="同组 A–A、排除自身配对（含分子内配对）",
                      default_frames=50, default_prefix="rdf_mda")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
