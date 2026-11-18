# -*- coding: utf-8 -*-
r"""MDAnalysis 交叉验证脚本集（按分析模块拆分）。

目录结构（每个模块都能**单独运行**）：

====================  ==================================================
脚本                   用途
====================  ==================================================
`_common.py`          公共：加载体系、补质量、帧窗口、分子编号、最小镜像距离、导出
`rg.py`               回转半径 Rg（逐分子 + 帧内平均）
`ree.py`              末端距 R_ee
`msd.py`              MSD（EinsteinMSD，FFT）
`rdf.py`              径向分布函数（InterRDF）
`comdis.py`           两组分质心距（最小镜像）
`contact.py`          接触分析 + 接触矩阵（与引擎逐位一致）
`run_all.py`          一次跑完全部，打印汇总
====================  ==================================================

三种用法：

```powershell
# ① 单独跑某个模块（在自己目录或仓库根都行）
python test/mda_test/contact.py
python test/mda_test/rg.py

# ② 当包用
cd C:\\temp\\MDT\\test ; python -m mda_test
```

```python
# ③ 在自己的脚本里 import
import sys; sys.path.insert(0, r"C:\temp\MDT\test")
from mda_test import load, test_contact, test_contact_matrix

u, ag = load()                      # 默认 1000water（.xsd + .trr）
test_contact(u)                     # 自动挑两个组分，inter 口径
test_contact(u, u.select_atoms("resname ETH"), None, cutoff=4.5, mode="total")
test_contact_matrix(u)
```

```python
# ④ 交互菜单（每个模块都不带参数运行就是这样）
#    ① 拓扑 → ② 轨迹 → ③ 分析对象 → ④ 帧 → ⑤ 模块参数 → ⑥ 输出 → 确认
```

环境变量（都可省）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `MDTA_MDA_XSD` | 拓扑文件 | `C:\\temp\\dataset\\1000water\\1000water.xtd` |
| `MDTA_MDA_TRR` | 轨迹文件 | `_convert_cache\\1000water.abac1eb9a5e1.trr` |
| `MDTA_MDA_SELECT` | 默认选择串 | `resname ETH` |
| `MDTA_MDA_FRAMES` | 最多用多少帧（大轨迹先设 20） | 不限 |
| `MDTA_MDA_OUT` | CSV 输出目录 | 当前目录 |

口径说明见仓库根的 `接触分析复现指南.md`。
"""

from __future__ import annotations

from ._common import (  # noqa: F401 - 对外统一出口
    DEFAULT_SELECT,
    DEFAULT_TRR,
    DEFAULT_XSD,
    atom_mol_ids,
    auto_pair,
    build_universe,
    fill_masses,
    frame_indices,
    frame_window,
    iterate,
    load,
    max_frames,
    min_image_min_distance,
    out_dir,
    out_path,
    save_csv,
)
from .comdis import test_comdis                 # noqa: F401
from .boo import (                               # noqa: F401
    boo_freud_frame,
    boo_reference,
    estimate_cutoff_freud,
    test_boo,
)
from .boo import compare_engine as compare_boo_engine            # noqa: F401
from .dihedral import (                          # noqa: F401
    chain_quadruplets,
    dihedral_reference,
    test_dihedral,
)
from .dihedral import compare_engine as compare_dihedral_engine  # noqa: F401
from .nematic import (                           # noqa: F401
    nematic_reference,
    segment_vectors,
    test_nematic,
)
from .nematic import compare_engine as compare_nematic_engine    # noqa: F401
from .density import (                           # noqa: F401
    density_reference,
    test_density,
    compare_mdanalysis as compare_density_linear,
)
from .density import compare_engine as compare_density_engine  # noqa: F401
from .contact import (                          # noqa: F401
    reproduce_contacts,
    test_contact,
    test_contact_matrix,
    test_contact_reproduce,
)
from .contact import compare_engine as compare_contact_engine   # noqa: F401
from .msd import test_msd                       # noqa: F401
from .msd_mdat import (                          # noqa: F401
    beta_window,
    diffusion_mdat,
    msd_fft_single,
    msd_mdat,
    test_msd_mdat,
    unwrap_incremental,
)
from .msd_mdat import compare_engine as compare_msd_engine      # noqa: F401
from .orientation import (                       # noqa: F401
    bond_midpoints,
    chain_coordinates,
    chain_size_histogram,
    cos2_to_axis,
    mean_angle_to_axis,
    midpoint_vectors,
    pooled_angle,
    pooled_cos2,
    pooled_straightness,
    straightness,
    test_angle_to_axis,
    test_cos2_axis,
    test_orientation_parameter,
)
from .rdf import test_rdf                       # noqa: F401
from .interface import (                         # noqa: F401
    W1090_PER_SIGMA,
    check_definition,
    interface_reference,
    synth_erf_profile,
    synth_slab_universe,
    test_interface,
    test_synthetic,
)
from .interface import compare_engine as compare_interface_engine        # noqa: F401
from .interface import compare_mdanalysis_interface                      # noqa: F401
from .rdf_mdat import (                          # noqa: F401
    compare_engine,
    compare_interrdf,
    frame_histogram,
    rdf_mdat,
    test_rdf_mdat,
)
from .ree import test_ree                       # noqa: F401
from .rg import test_rg                         # noqa: F401

__all__ = [
    # 公共
    "load", "build_universe", "fill_masses", "save_csv", "out_dir", "out_path",
    "iterate", "frame_indices", "frame_window", "max_frames",
    "atom_mol_ids", "min_image_min_distance", "auto_pair",
    "DEFAULT_XSD", "DEFAULT_TRR", "DEFAULT_SELECT",
    # 各模块
    "test_rg", "test_ree", "test_msd", "test_rdf", "test_comdis",
    "test_contact", "test_contact_matrix",
    # 接触分析的独立复现 + 引擎对拍（原 examples/contact_reproduce.py）
    "reproduce_contacts", "test_contact_reproduce", "compare_contact_engine",
    # RDF（util_rdf.py / MDTransport 口径）
    "rdf_mdat", "test_rdf_mdat", "compare_interrdf", "compare_engine",
    "frame_histogram",
    # 密度分布（1D）：教科书实现 + MDAnalysis LinearDensity 对拍
    "density_reference", "test_density", "compare_density_linear",
    "compare_density_engine",
    # MSD / 扩散系数（util_msd.py + util_diffusion.py 口径）
    "msd_mdat", "test_msd_mdat", "diffusion_mdat", "beta_window",
    "msd_fft_single", "unwrap_incremental", "compare_msd_engine",
    # 链段取向（allfast-3-2.py 口径）
    "test_cos2_axis", "test_orientation_parameter", "test_angle_to_axis",
    "bond_midpoints", "midpoint_vectors", "cos2_to_axis", "straightness",
    "mean_angle_to_axis", "chain_coordinates", "chain_size_histogram",
    "pooled_cos2", "pooled_angle", "pooled_straightness",
    # 界面宽度：参考实现 + 合成已知宽度体系 + 双向对拍
    "interface_reference", "test_interface", "test_synthetic",
    "compare_interface_engine", "compare_mdanalysis_interface",
    "check_definition", "synth_erf_profile", "synth_slab_universe",
    "W1090_PER_SIGMA",
    # 二面角分布（MDAnalysis Dihedral）/ 取向 S（freud Nematic）/ BOO（freud Steinhardt）
    "chain_quadruplets", "dihedral_reference", "test_dihedral",
    "compare_dihedral_engine",
    "segment_vectors", "nematic_reference", "test_nematic",
    "compare_nematic_engine",
    "estimate_cutoff_freud", "boo_freud_frame", "boo_reference", "test_boo",
    "compare_boo_engine",
]
