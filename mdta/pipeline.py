# -*- coding: utf-8 -*-
"""分析流程编排模块。

把"读取 → 选择 → 预处理 → 分析 → 可视化 → 导出"串成一条流水线，
让命令行界面和图形界面都只需要调用 :class:`Analyzer`。

示例
----
::

    from mdta.pipeline import Analyzer

    az = Analyzer("sys.tpr", "sys.xtc")
    az.set_frames(equil_ps=20000, interval_ps=100)   # 跳过前 20 ns，每 100 ps 取一帧
    print(az.info_text())
    az.auto_setup()                                  # 自动推荐主链与组分
    results = az.run_all(outdir="analysis_results")
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Mapping, Sequence

import numpy as np

from .core import AnalysisResult
from .io import MDTrajectory, load_trajectory
from .preprocess import FrameSelection, select_frames
from .selection import (
    SelectionError,
    component_groups,
    interface_region,
    largest_chains,
)
from .systeminfo import SystemInfo, describe_system, format_system_info

#: 全部分析项及其默认参数
DEFAULT_PARAMS: dict = {
    "rg": {"mass_weighted": True},
    "ree": {},
    "dihedral": {"mode": "auto", "gauche_edge": 120.0},
    "density": {"axis": 2, "nbins": 100, "mode": "mass"},
    "rdf": {"rmax": 12.0, "nbins": 120, "compare_halves": False},
    "contact": {"cutoff": 5.0},
    "interface": {"axis": 2, "nbins": 120},
    "orientation": {"mode": "backbone", "stride": 1},
    "order": {},
    "msd": {},
}

#: 默认执行顺序（按设计大纲第 25–27 章）
DEFAULT_ORDER = ["rg", "ree", "dihedral", "density", "rdf", "contact",
                 "interface", "orientation", "order", "msd"]

#: 中文标题
ANALYSIS_TITLES = {
    "rg": "回转半径 Rg",
    "ree": "端到端距离 R_ee",
    "dihedral": "二面角分析",
    "density": "密度分布",
    "rdf": "径向分布函数 RDF",
    "contact": "接触分析",
    "interface": "界面宽度分析",
    "orientation": "链段取向分析",
    "order": "结构有序度分析",
    "msd": "均方位移 MSD",
}

#: 按设计大纲的**模块**给分析项分组：``[(模块名, (分析项, ...)), ...]``。
#: 命令行、桌面操作台与 Web 界面共用这一份定义，界面上「分析功能」的预设按钮
#: 与「图表导航」的分层都从这里生成，不会各写一份导致对不上。
ANALYSIS_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("链构象", ("rg", "ree", "dihedral")),
    ("界面", ("density", "rdf", "contact", "interface")),
    ("结晶", ("orientation", "order")),
    ("辅助", ("msd",)),
]


def analysis_groups() -> list[dict]:
    """给界面用的分组结构：``[{"label": 模块名, "names": [...]}, ...]``。"""
    return [{"label": label, "names": list(names)} for label, names in ANALYSIS_GROUPS]


class Analyzer:
    """一次完整的分析会话。"""

    def __init__(self, topology: str, trajectory: str | None = None):
        self.trajectory: MDTrajectory = load_trajectory(topology, trajectory)
        self.frames: FrameSelection | None = None
        self.primary = None            # 主链（构象/结晶分析对象）
        self.primary_label: str = "链"
        self.components: "OrderedDict[str, object]" = OrderedDict()
        self.extra_components: "OrderedDict[str, object]" = OrderedDict()
        self.results: "OrderedDict[str, AnalysisResult]" = OrderedDict()
        self._info: SystemInfo | None = None

    # ------------------------------------------------------------ 基本信息
    @property
    def universe(self):
        return self.trajectory.universe

    @property
    def times_ps(self) -> np.ndarray:
        return self.trajectory.times_ps

    @property
    def info(self) -> SystemInfo:
        if self._info is None:
            self._info = describe_system(self.trajectory)
        return self._info

    def info_text(self, max_chain_types: int = 20) -> str:
        return format_system_info(self.info, max_chain_types=max_chain_types)

    # ------------------------------------------------------------ 帧选择
    def set_frames(self, *, start_ps: float | None = None, stop_ps: float | None = None,
                   interval_ps: float | None = None, equil_ps: float | None = None,
                   max_frames: int | None = None) -> FrameSelection:
        """设置参与统计的帧（起始/结束时间、抽帧间隔、平衡段）。"""
        self.frames = select_frames(self.times_ps, start_ps=start_ps, stop_ps=stop_ps,
                                    interval_ps=interval_ps, equil_ps=equil_ps,
                                    max_frames=max_frames)
        return self.frames

    def require_frames(self) -> FrameSelection:
        if self.frames is None or self.frames.n_frames == 0:
            self.set_frames()
        return self.frames

    # ------------------------------------------------------------ 选择
    def auto_setup(self, *, max_components: int = 4, min_chain_atoms: int = 20) -> dict:
        """自动推荐主链与组分。

        - **主链**：体系中最大的分子/链（高分子链通常最大）；
        - **组分**：protein / polymer / water / ion 等自动识别结果，
          最多取 ``max_components`` 个。

        返回 ``{"primary": 描述, "components": [...]}``。
        """
        u = self.universe
        chains = largest_chains(u, n=1, min_atoms=min_chain_atoms)
        if chains:
            self.primary = chains[0]
            seg = str(chains[0].segids[0])
            self.primary_label = f"最大链 ({seg}, {chains[0].n_atoms} 原子)"
        else:
            self.primary = u.atoms
            self.primary_label = "全部原子"

        comps = component_groups(u)
        # protein/polymer 优先，其余按原子数降序
        priority = {"protein": 0, "nucleic": 1, "sugar": 2, "polymer": 3,
                    "water": 4, "ion": 5, "other": 6}
        items = sorted(comps.items(),
                       key=lambda kv: (priority.get(kv[0].split("[")[0], 9), -kv[1].n_atoms))
        self.components = OrderedDict(items[:max_components])
        self.extra_components = OrderedDict()
        if not self.components:
            self.components = OrderedDict([("all", u.atoms)])
        return {"primary": self.primary_label, "components": list(self.components)}

    def set_primary(self, ag, label: str = "链") -> None:
        self.primary = ag
        self.primary_label = label

    def set_components(self, components: Mapping[str, object]) -> None:
        self.components = OrderedDict(components)

    def add_interface_component(self, name_a: str, name_b: str, cutoff: float = 5.0,
                                label: str | None = None):
        """把"界面区域"作为新组分加入（组分 A 中距 B 小于 cutoff 的原子）。"""
        ga = self.components.get(name_a)
        gb = self.components.get(name_b)
        if ga is None or gb is None:
            raise SelectionError(f"组分 {name_a!r} 或 {name_b!r} 不存在")
        u = self.universe
        u.trajectory[0]
        reg = interface_region(u, ga, gb, cutoff=cutoff)
        lbl = label or f"界面区({name_a}∩{name_b}<{cutoff:g}Å)"
        self.extra_components[lbl] = reg
        return lbl, reg

    # ------------------------------------------------------------ 运行
    def run(self, name: str, *, params: Mapping | None = None,
            verbose: bool = False) -> AnalysisResult | None:
        """运行单个分析项。"""
        from .analysis import conformation as conf
        from .analysis import crystallinity as cry
        from .analysis import dynamics as dyn
        from .analysis import interface as ifc

        sel = self.require_frames()
        p = dict(DEFAULT_PARAMS.get(name, {}))
        if params:
            p.update(params)

        if self.primary is None:
            self.auto_setup()

        label = self.primary_label
        if name == "rg":
            return conf.analyze_rg(self.trajectory, self.primary, sel, label=label,
                                   verbose=verbose, **p)
        if name == "ree":
            return conf.analyze_end_to_end(self.trajectory, self.primary, sel,
                                           label=label, verbose=verbose, **p)
        if name == "dihedral":
            return conf.analyze_dihedrals(self.trajectory, self.primary, sel,
                                          label=label, verbose=verbose, **p)
        if name == "density":
            return ifc.analyze_density(self.trajectory, self.components, sel,
                                       verbose=verbose, **p)
        if name == "rdf":
            groups = self._rdf_groups()
            if len(groups) < 1:
                return None
            return ifc.analyze_rdf(self.trajectory, groups, sel, verbose=verbose, **p)
        if name == "contact":
            a, b = self._contact_pair()
            if a is None:
                return None
            return ifc.analyze_contacts(self.trajectory, a, b, sel, verbose=verbose, **p)
        if name == "interface":
            if len(self.components) < 2:
                return None
            # 默认取**原子数最多的两个组分**当作界面两侧：分层体系里这两相
            # 才是真正的界面两侧；若按字典序取前两个，可能取到"溶解在里面的
            # 小分子 + 基体"，那对组合没有平界面。可用 params 显式指定 pair。
            ordered = sorted(self.components.items(), key=lambda kv: -kv[1].n_atoms)
            pair = p.pop("pair", None) or (ordered[0][0], ordered[1][0])
            for nm in pair:
                if nm not in self.components:
                    raise SelectionError(f"界面分析指定的组分 {nm!r} 不存在")
            return ifc.analyze_interface_width(self.trajectory, self.components, sel,
                                               pair=tuple(pair),
                                               verbose=verbose, **p)
        if name == "orientation":
            return cry.analyze_orientation(self.trajectory, self.primary, sel,
                                           label=label, verbose=verbose, **p)
        if name == "order":
            return cry.analyze_structural_order(self.trajectory, self.primary, sel,
                                                label=label, verbose=verbose, **p)
        if name == "msd":
            groups = self._msd_groups()
            if not groups:
                return None
            return dyn.analyze_msd(self.trajectory, groups, sel, verbose=verbose, **p)
        raise KeyError(f"未知分析项: {name!r}")

    def run_all(self, which: Sequence[str] | None = None, *,
                params: Mapping[str, Mapping] | None = None,
                outdir: str | None = None, formats: Sequence[str] = ("csv", "png"),
                excel: bool = True, panel_pngs: bool = False,
                verbose: bool = True,
                raise_errors: bool = False,
                progress: Callable | None = None,
                on_result: Callable | None = None,
                cancel: "threading.Event | None" = None) -> "OrderedDict[str, AnalysisResult]":
        """依次运行多项分析，可选直接导出。

        ``progress`` 是一个 ``callable(frac: float, message: str)`` 回调，
        用于向图形界面汇报进度（``frac`` 取 0–1）。

        ``on_result`` 是一个 ``callable(name, result, elapsed_sec, done, total)``
        回调，在**每一项分析算完后立即**调用 —— Web 版靠它把已算好的结果
        实时推给页面，而不必等全部跑完。

        ``cancel`` 传入一个 :class:`threading.Event` 时，会在**每项分析开始前**
        检查它；已置位则停止并把已完成的项作为结果返回。
        ``panel_pngs=True`` 时额外为每个面板单独导出一张 PNG。
        """
        names = list(which) if which else list(DEFAULT_ORDER)
        params = dict(params or {})
        self.results = OrderedDict()
        #: ``{分析名: 耗时秒数}``，供界面显示"哪一项最慢"
        self.timings: dict[str, float] = {}
        sel = self.require_frames()
        if verbose:
            print(f"[帧选择] {sel.describe()}")
            for n in sel.notes:
                print(f"          {n}")
        total = max(len(names), 1)
        self.cancelled = False
        for i, name in enumerate(names):
            if cancel is not None and cancel.is_set():
                self.cancelled = True
                if verbose:
                    print("[取消] 收到取消请求，停止后续分析")
                break
            title = ANALYSIS_TITLES.get(name, name)
            if progress:
                progress(i / total, f"正在计算：{title}")
            t0 = time.time()
            if verbose:
                print(f"[分析] {title} ...")
            try:
                res = self.run(name, params=params.get(name), verbose=False)
            except Exception as exc:  # noqa: BLE001
                if raise_errors:
                    raise
                print(f"  !! {name} 失败: {type(exc).__name__}: {exc}")
                continue
            if res is None:
                if verbose:
                    print("  -- 跳过（体系缺少所需组分）")
                continue
            self.results[name] = res
            self.timings[name] = time.time() - t0
            if verbose:
                print(f"  -> {len(res.curves)} 条曲线, {len(res.summary)} 项统计, "
                      f"{time.time() - t0:.1f}s")
            # 关键：算完一项就立刻回调，Web 版据此实时刷新页面
            if on_result:
                on_result(name, res, self.timings[name], len(self.results), total)
        if progress:
            progress(1.0, "正在导出结果…")
        if outdir:
            from .export import export_all

            out = export_all(list(self.results.values()), outdir, formats=formats,
                             excel=excel, panel_pngs=panel_pngs, verbose=verbose)
            if verbose:
                print(f"[导出] {out['dir']}")
        if progress:
            progress(1.0, "完成")
        return self.results

    # ------------------------------------------------------------ 内部
    def _rdf_groups(self) -> "OrderedDict[str, object]":
        """RDF 用的原子组：优先选原子数适中的组分，避免超大体系过慢。

        对水这类原子数极多的组分，自动只取氧原子（或每个分子一个代表原子），
        这样既保持物理意义又大幅加速。
        """
        out: "OrderedDict[str, object]" = OrderedDict()
        u = self.universe
        for name, ag in self.components.items():
            category = name.split("[")[0]
            if category == "water":
                try:
                    ow = ag.select_atoms("name OW O")
                    if ow.n_atoms > 0:
                        out[f"{name}:O"] = ow
                        continue
                except Exception:  # noqa: BLE001
                    pass
                # 没有标准水原子名时，每个水分子取第一个原子
                try:
                    reps = [r.atoms[0] for r in ag.residues]
                    out[f"{name}:代表原子"] = u.atoms[[int(a.index) for a in reps]]
                    continue
                except Exception:  # noqa: BLE001
                    pass
            out[name] = ag
        return out

    def _contact_pair(self):
        names = list(self.components)
        if len(names) >= 2:
            return self.components[names[0]], self.components[names[1]]
        if len(names) == 1:
            return self.components[names[0]], self.components[names[0]]
        return None, None

    def _msd_groups(self) -> "OrderedDict[str, object]":
        out: "OrderedDict[str, object]" = OrderedDict()
        for name, ag in self.components.items():
            category = name.split("[")[0]
            if category == "water":
                try:
                    ow = ag.select_atoms("name OW O")
                    if ow.n_atoms > 0:
                        out[f"{name}(O)"] = ow
                        continue
                except Exception:  # noqa: BLE001
                    pass
            if ag.n_atoms > 5000:
                # 抽稀以避免过大的内存与计算量
                step = int(np.ceil(ag.n_atoms / 2000))
                out[f"{name}(抽稀 1/{step})"] = ag[::step]
            else:
                out[name] = ag
        for name, ag in self.extra_components.items():
            if ag.n_atoms > 5000:
                step = int(np.ceil(ag.n_atoms / 2000))
                out[f"{name}(抽稀 1/{step})"] = ag[::step]
            elif ag.n_atoms:
                out[name] = ag
        return out


# ---------------------------------------------------------------- 便捷函数
def analyze_all(topology: str, trajectory: str | None = None, *,
                which: Sequence[str] | None = None,
                outdir: str | None = "analysis_results",
                start_ps: float | None = None, stop_ps: float | None = None,
                interval_ps: float | None = None, equil_ps: float | None = None,
                max_frames: int | None = None,
                components: Mapping[str, object] | None = None,
                primary=None, primary_label: str = "链",
                params: Mapping[str, Mapping] | None = None,
                formats: Sequence[str] = ("csv", "png"), excel: bool = True,
                verbose: bool = True) -> "OrderedDict[str, AnalysisResult]":
    """一步完成"读取 → 选择 → 分析 → 导出"。"""
    az = Analyzer(topology, trajectory)
    az.set_frames(start_ps=start_ps, stop_ps=stop_ps, interval_ps=interval_ps,
                  equil_ps=equil_ps, max_frames=max_frames)
    if verbose:
        print(az.info_text())
    if components is None:
        az.auto_setup()
    else:
        az.set_components(components)
        if primary is None:
            az.auto_setup()
    if primary is not None:
        az.set_primary(primary, primary_label)
    return az.run_all(which, params=params, outdir=outdir, formats=formats,
                      excel=excel, verbose=verbose)
