# -*- coding: utf-8 -*-
"""一次跑完全部模块（``python -m mda_test`` 或 ``python test/mda_test/run_all.py``）。

只加载一次体系，然后依次跑 Rg / R_ee / MSD / RDF / 质心距 / 接触，最后打印一张
小汇总表。大轨迹先用 ``MDTA_MDA_FRAMES=20`` 限帧。
"""

from __future__ import annotations

import traceback

try:
    from ._common import frame_indices, load
    from .boo import test_boo
    from .comdis import test_comdis
    from .density import test_density
    from .contact import test_contact, test_contact_matrix, test_contact_reproduce
    from .dihedral import test_dihedral
    from .interface import test_synthetic
    from .msd import test_msd
    from .msd_mdat import test_msd_mdat
    from .nematic import test_nematic
    from .orientation import (test_angle_to_axis, test_cos2_axis,
                              test_orientation_parameter)
    from .rdf import test_rdf
    from .rdf_mdat import test_rdf_mdat
    from .ree import test_ree
    from .rg import test_rg
except ImportError:                       # 直接 python test/mda_test/run_all.py
    from _common import frame_indices, load
    from boo import test_boo
    from comdis import test_comdis
    from density import test_density
    from contact import test_contact, test_contact_matrix, test_contact_reproduce
    from dihedral import test_dihedral
    from interface import test_synthetic
    from msd import test_msd
    from msd_mdat import test_msd_mdat
    from nematic import test_nematic
    from orientation import (test_angle_to_axis, test_cos2_axis,
                             test_orientation_parameter)
    from rdf import test_rdf
    from rdf_mdat import test_rdf_mdat
    from ree import test_ree
    from rg import test_rg


def _oxygen(u):
    """挑一组"中心粒子"做 O(n²) 的 RDF：优先氧原子（水的标准做法）。"""
    for sel in ("element O", "name O*", "name OW"):
        try:
            ag = u.select_atoms(sel)
        except Exception:                                 # noqa: BLE001
            continue
        if ag.n_atoms > 0:
            return ag
    return u.atoms


def main(argv=None):
    u, ag = load()
    steps = [
        ("Rg（回转半径）", lambda: test_rg(u, ag)),
        ("R_ee（末端距）", lambda: test_ree(u, ag)),
        ("MSD（EinsteinMSD）", lambda: test_msd(u, ag)),
        ("MSD / 扩散系数（MDTransport 口径）",
         lambda: test_msd_mdat(u, ag, mode="atom", frames=frame_indices(u)[:5])),
        ("RDF（InterRDF）", lambda: test_rdf(u, ag)),
        ("RDF（util_rdf / MDTransport 口径，只看氧原子）",
         lambda: test_rdf_mdat(u, _oxygen(u), None, dr=0.05,
                               frames=frame_indices(u)[:3])),
        ("质心距", lambda: test_comdis(u)),
        ("密度分布（1D，与 LinearDensity 对拍）",
         lambda: test_density(u, ag, axis=2, binsize=0.5,
                             frames=frame_indices(u)[:3])),
        ("界面宽度（合成已知宽度体系 + 与引擎对拍）",
         lambda: test_synthetic(n_atoms=60000, n_frames=4)),
        ("链段取向：⟨cos²θ⟩（allfast 口径）", lambda: test_cos2_axis(u, ag)),
        ("链段取向：取向参数（相邻链段夹角 cos²）",
         lambda: test_orientation_parameter(u, ag)),
        ("链段取向：与参考轴的平均夹角 θ(t)", lambda: test_angle_to_axis(u, ag)),
        ("取向参数 S（freud Nematic 对拍）",
         lambda: test_nematic(u, ag, frames=frame_indices(u)[:3],
                              compare=True)),
        ("二面角分布（MDAnalysis Dihedral 对拍）",
         lambda: test_dihedral(u, ag, frames=frame_indices(u)[:3],
                               compare=True)),
        ("键取向序 BOO（freud Steinhardt 对拍）",
         lambda: test_boo(u, ag, frames=frame_indices(u)[:2], compare=True)),
        ("接触分析", lambda: test_contact(u, frames=frame_indices(u)[:3])),
        ("接触分析（独立复现 + 与引擎对拍）",
         lambda: test_contact_reproduce(u, frames=frame_indices(u)[:3],
                                        compare=True)),
        ("接触矩阵", lambda: test_contact_matrix(u, frames=frame_indices(u)[:3])),
    ]
    done, failed = [], []
    for title, fn in steps:
        print("\n" + "=" * 70)
        print(f"[{title}]")
        print("=" * 70)
        try:
            fn()
            done.append(title)
        except SystemExit:
            raise
        except Exception as exc:                       # noqa: BLE001
            failed.append((title, f"{type(exc).__name__}: {exc}"))
            print(f"✗ 失败：{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    print("\n" + "=" * 70)
    print(f"完成 {len(done)} 项，失败 {len(failed)} 项")
    for t in done:
        print(f"  ✓ {t}")
    for t, msg in failed:
        print(f"  ✗ {t} —— {msg}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
