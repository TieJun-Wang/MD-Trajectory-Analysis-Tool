# -*- coding: utf-8 -*-
r"""本工具自己的功能脚本集（每项分析一个文件，都能**单独运行**）。

和 `test/mda_test/` 的分工
--------------------------
`mda_test` 用第三方实现（MDAnalysis / freud / 作者脚本口径）**交叉验证**"算得对不对"；
本目录用工具自己的引擎（``mdta``）演示"这个功能怎么用、输出长什么样、文件叫什么名"：：

====================  ==========================================================
脚本                   对应的工具功能
====================  ==========================================================
`_common.py`          公共：``Engine`` = 加载体系 → 识别组分 → 选帧 → 跑分析 → 导出
`_menu.py`            交互菜单 / 命令行（复用 `test/mda_test/_menu.py`，只换加载器）
`system.py`           体系信息 / 组分识别 / 能力检测 / 14 项分析的可行性预判
`frames.py`           帧选择（时间窗 / 抽帧间隔 / 平衡段 / 帧数上限）
`rg.py`               回转半径 Rg
`ree.py`              端到端距离 R_ee
`dihedral.py`         二面角分布 + trans/gauche 构象比例
`density.py`          密度分布 ρ(z)
`rdf.py`              径向分布函数 g(r)
`rdf2d.py`            面内径向分布 g₂(r)
`comdist.py`          两组分质心距
`contact.py`          接触分析（接触对数 / 最小间距 / 占据率）
`interface.py`        界面宽度（10–90 判据）
`orientation.py`      链段取向（S、P₂、⟨cos²θ⟩）
`order.py`            结构有序度（二面角 + 取向 + 局部结构三路合成）
`boo.py`              键取向序参数 BOO（q̄₄ / q̄₆）
`msd.py`              均方位移 MSD + 扩散系数 D
`vacf.py`             速度自相关函数 VACF
`pipeline.py`         一键分析：``Analyzer.run_all`` + 导出 CSV/PNG/Excel
`run_all.py`          一次跑完全部模块，打印汇总表
====================  ==========================================================

三种用法
--------
```powershell
# ① 单独跑一项：不带参数 = 交互菜单，带参数 = 命令行
python test/my_test/contact.py
python test/my_test/contact.py --frames 50 --cutoff 4.5 -y
python test/my_test/run_all.py --system mix --frames 10

# ② 当包用
cd C:\temp\MDT\test ; python -m my_test
```

```python
# ③ 在自己的脚本里 import
import sys; sys.path.insert(0, r"C:\temp\MDT\test")
from my_test import Engine, test_contact, test_orientation

eng = Engine(max_frames=20)          # 默认体系 20PE30（单组分）
test_rg(eng)                         # 20 帧的 Rg
test_contact(eng, cutoff=4.5)        # 20 帧的接触分析
test_orientation(eng, mode="backbone", weighting="molecule")
```

环境变量（都可省）
------------------
| 变量 | 作用 | 默认 |
|---|---|---|
| `MDTA_MYTEST_SYSTEM` | 预置体系：`pe`（20PE30）/ `mix`（5PE20…多组分） | `pe` |
| `MDTA_MYTEST_TOP` | 拓扑文件 | 预置体系 |
| `MDTA_MYTEST_TRJ` | 轨迹文件（`.trj` 自动转 TRR） | 预置体系 |
| `MDTA_MYTEST_SELECT` | 分析对象选择串 | 自动识别 |
| `MDTA_MYTEST_FRAMES` | 最多用多少帧 | 100 |
| `MDTA_MYTEST_OUT` | CSV 输出目录 | `test/my_test/_out` |

口径说明见仓库根的 `使用手册.md`；与第三方实现逐位对拍的脚本在 `test/mda_test/`。
"""

from __future__ import annotations

from ._common import (  # noqa: F401 - 对外统一出口
    DEFAULT_FRAMES,
    DEFAULT_SELECT,
    DEFAULT_SYSTEM,
    DEFAULT_TOP,
    DEFAULT_TRJ,
    SYSTEMS,
    Engine,
    fmt,
    load,
    out_dir,
    summarize,
)
from .boo import test_boo                              # noqa: F401
from .comdist import test_comdist                      # noqa: F401
from .contact import test_contact                      # noqa: F401
from .density import test_density                      # noqa: F401
from .dihedral import test_dihedral                    # noqa: F401
from .frames import test_frames                        # noqa: F401
from .interface import test_interface                  # noqa: F401
from .msd import test_msd                              # noqa: F401
from .order import test_order                          # noqa: F401
from .orientation import test_orientation              # noqa: F401
from .pipeline import test_pipeline                    # noqa: F401
from .rdf import test_rdf                              # noqa: F401
from .rdf2d import test_rdf2d                          # noqa: F401
from .ree import test_ree                              # noqa: F401
from .rg import test_rg                                # noqa: F401
from .system import test_system                        # noqa: F401
from .vacf import test_vacf                            # noqa: F401

__all__ = [
    # 公共
    "Engine", "load", "out_dir", "fmt", "summarize", "SYSTEMS",
    "DEFAULT_SYSTEM", "DEFAULT_TOP", "DEFAULT_TRJ", "DEFAULT_SELECT",
    "DEFAULT_FRAMES",
    # 体系 / 帧
    "test_system", "test_frames",
    # 链构象
    "test_rg", "test_ree", "test_dihedral",
    # 空间结构
    "test_density", "test_rdf", "test_rdf2d", "test_comdist", "test_contact",
    "test_interface",
    # 取向与结晶
    "test_orientation", "test_order", "test_boo",
    # 动力学与输运
    "test_msd", "test_vacf",
    # 一键
    "test_pipeline",
]
