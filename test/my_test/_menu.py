# -*- coding: utf-8 -*-
"""交互菜单 + 命令行（复用 `test/mda_test/_menu.py` 的那一套，只换"怎么加载体系"）。

为什么只换加载这一步
--------------------
`mda_test` 的菜单已经解决了这些杂事：拓扑/轨迹文件怎么问、体系里有哪些组分、
帧怎么抽、摘要确认、``-y`` 免确认，以及"不带参数 = 交互 / 带参数 = 命令行"。
本目录只是**换掉加载器**：`mda_test` 那边是"自己按 MDAnalysis 拼一个 Universe"，
这里必须走工具自己的 `mdta.io.load_trajectory`（Materials Studio ``.trj`` 自动转
TRR、MS ``.xsd`` 走 `mdta.ms_structure`、缺质量/键表自动补……），否则脚本测的
就不是工具真正用的那条链路了。

做法是给 `mda_test._menu` 的 ``_load`` 打个补丁（见 :func:`_load_engine`），
菜单/命令行的其余部分一行不改 —— 于是两个目录的用法保持完全一致。

环境变量见 `_common.py`（``MDTA_MYTEST_SYSTEM`` / ``_TOP`` / ``_TRJ`` / ``_OUT``）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# `mda_test` 是 `test/` 下的包：把 `test/` 放进 sys.path 才能 `import mda_test`
_TEST_DIR = Path(__file__).resolve().parents[1]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import mda_test._menu as _mm                                    # noqa: E402

try:                                                            # 作为包导入
    from ._common import DEFAULT_FRAMES, DEFAULT_TOP, DEFAULT_TRJ, Engine
except ImportError:                                             # 直接运行本文件
    from _common import DEFAULT_FRAMES, DEFAULT_TOP, DEFAULT_TRJ, Engine

#: 复用：问题规格、选项读取、摘要、确认、命令行解析、退出码处理
Q = _mm.Q                                                       # noqa: F401
first_existing = _mm.first_existing                             # noqa: F401
choose_frames = _mm.choose_frames                               # noqa: F401
print_summary = _mm.print_summary                               # noqa: F401

#: ``_load`` 建好的引擎（菜单与命令行都在同一次进程里，逐项跑完即取走）
_SHARED: dict = {}

#: 加载选项（模块通过 :func:`run_analysis_menu` 的 ``velocities`` 参数设置）。
#: 目前只有 VACF 需要 ``velocities=True``：MS 的 ``.trj`` 默认只转坐标（XTC），
#: 带速度要转 TRR —— 转换慢一次，之后有缓存。
_LOAD: dict = {}


def _load_engine(cfg):
    """替换 `mda_test._menu` 的加载器：用工具自己的 ``mdta.io.load_trajectory``。

    ``cfg`` 里只有 top/trj/select 这三样有意义（``select`` 在这里**先不管**：
    交互菜单拿到 Universe 之后才会问"分析对象"，最终的选择串在
    :func:`run_analysis_menu` 的 ``_run`` 里再 ``apply_select``）。
    """
    eng = Engine(cfg.get("top"), cfg.get("trj"), select=None,
                 velocities=_LOAD.get("velocities"), verbose=True)
    _SHARED["eng"] = eng
    ag = eng.primary if eng.primary is not None else eng.u.atoms
    return eng.u, ag


def run_analysis_menu(name: str, title: str, specs, fn, *, sub: str = "",
                      default_frames: int | None = None,
                      default_top: str | None = None,
                      default_trj: str | None = None,
                      velocities: bool | None = None):
    """一项分析的统一入口：**不带参数 → 交互菜单，带参数 → 命令行**。

    ``fn(eng, *, outdir=None, formats=("csv",), **specs 的值)`` 是模块的分析函数；
    菜单负责把"拓扑/轨迹/分析对象/帧/本模块参数/输出目录/前缀"问齐，
    这里负责把它们翻译成引擎调用。

    约定：模块的 ``specs`` 里若出现 ``pair_a`` / ``pair_b`` 两个键，就自动拼成
    引擎的 ``pair=(A, B)``（接触 / 界面宽度 / 质心距"每次只比较一对"用的那个）。

    ``velocities=True`` 时加载会保留速度（MS 的 ``.trj`` 转 TRR 而不是 XTC）——
    VACF 需要它，其余项用不到（转换更慢、文件更大）。
    """
    _LOAD["velocities"] = velocities
    specs = list(specs) + [Q("png", "同时导出 PNG 图", "bool", False,
                             help="默认只导 CSV；加 --png（或菜单里选 y）再多导一张汇总图")]
    frames_n = DEFAULT_FRAMES if default_frames is None else int(default_frames)

    # 预置体系的默认路径也要换成我们的（`mda_test._common` 里写死的是 20PE30）
    import mda_test._common as _mc

    _mc.DEFAULT_XSD = default_top or DEFAULT_TOP
    _mc.DEFAULT_TRR = default_trj if default_trj is not None else DEFAULT_TRJ
    _mc.DEFAULT_SELECT = "all"

    def _run(cfg):
        eng = _SHARED.pop("eng", None)
        if eng is None:                                   # 兜底（正常不会走到）
            eng = Engine(cfg.get("top"), cfg.get("trj"), verbose=True)
        sel = str(cfg.get("select") or "").strip()
        if sel and sel.lower() not in ("all", "*", "全部", "全部原子"):
            eng.apply_select(sel)
        eng.use_frames(cfg.get("frames"))
        eng.export_prefix = str(cfg.get("prefix") or "").strip() or None

        kwargs = {}
        pair_a = str(cfg.pop("pair_a", "") or "").strip()
        pair_b = str(cfg.pop("pair_b", "") or "").strip()
        for s in specs:
            if s.key in ("pair_a", "pair_b"):
                continue
            kwargs[s.key] = cfg.get(s.key, s.default)
        if {"pair_a", "pair_b"} & {s.key for s in specs}:
            if pair_a and pair_b:
                kwargs["pair"] = (pair_a, pair_b)
            elif pair_a or pair_b:
                print("⚠️ A/B 两组要同时给（只给了一组，已按自动配对处理）")
        kwargs.pop("png", None)
        formats = ("csv", "png") if cfg.get("png") else ("csv",)
        return fn(eng, outdir=str(cfg.get("outdir") or "").strip() or None,
                  formats=formats, **kwargs)

    _mm._load = _load_engine                             # 只换加载器，其余照旧
    return _mm.run_module(title, _run, specs, sub=sub,
                          default_select="all", default_frames=frames_n,
                          default_prefix=name,
                          default_top=default_top or DEFAULT_TOP,
                          default_trj=(default_trj if default_trj is not None
                                       else DEFAULT_TRJ))


def main_cli(menu_fn) -> int:
    """``if __name__ == "__main__": sys.exit(main_cli(menu))``（同 `mda_test`）。"""
    return _mm.main_cli(menu_fn)
