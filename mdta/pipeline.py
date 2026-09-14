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
    "rdf2d": {"plane_axis": 2, "rmax": 70.0, "nbins": 140},
    "comdist": {"axis": 2, "pbc": False},
    "contact": {"cutoff": 5.0},
    "interface": {"axis": 2, "nbins": 120},
    # 取向链段默认取**化学重复单元**（1.0.1 起；1.0.0 为几何骨架 "backbone"）
    "orientation": {"mode": "repeat", "stride": 1},
    # trans/gauche 判据与「二面角分析」共用同一阈值：改一处两处都变，
    # 免得同一条轨迹上两个模块报出不同的"trans 构象比例"。
    # ⚠️ 每个分析项在这张表里只能出现**一次**：字典字面量里重复的键会被后者
    #    静默覆盖（这里曾同时写了 `"order": {"gauche_edge": …}` 和 `"order": {}`，
    #    结果默认阈值被吃掉，只有显式传参才生效）。
    "order": {"gauche_edge": 120.0},
    "msd": {},
    # 键取向序参数 / 晶体-非晶识别（1.0.2）：默认用 Lechner–Dellago 平均版 q̄6，
    # 邻域半径留空 = 自动按最近邻距离中位数推定
    "boo": {"cutoff": None, "averaged": True},
    "crystal": {"cutoff": None, "q6_solid": 0.5, "min_cluster": 10},
}

#: 默认执行顺序（按设计大纲第 25–27 章）
DEFAULT_ORDER = ["rg", "ree", "dihedral", "density", "rdf", "contact",
                 "interface", "orientation", "order", "boo", "crystal", "msd"]

#: 中文标题
ANALYSIS_TITLES = {
    "rg": "回转半径 Rg",
    "ree": "端到端距离 R_ee",
    "dihedral": "二面角分析",
    "density": "密度分布",
    "rdf": "径向分布函数 RDF",
    "rdf2d": "面内径向分布 RDF (2D)",
    "comdist": "两组分质心距",
    "contact": "接触分析",
    "interface": "界面宽度分析",
    "orientation": "链段取向分析",
    "order": "结构有序度分析",
    "boo": "键取向序参数 BOO",
    "crystal": "晶体/非晶区域识别",
    "msd": "均方位移 MSD",
}

#: 按设计大纲的**模块**给分析项分组：``[(模块名, (分析项, ...)), ...]``。
#: 命令行、桌面操作台与 Web 界面共用这一份定义，界面上「分析功能」的预设按钮
#: 与「图表导航」的分层都从这里生成，不会各写一份导致对不上。
ANALYSIS_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("链构象", ("rg", "ree", "dihedral")),
    ("空间结构", ("density", "rdf", "rdf2d", "comdist", "contact", "interface")),
    ("取向与结晶", ("orientation", "order", "boo", "crystal")),
    ("动力学与输运", ("msd",)),
]

#: 分组在**快捷按钮行**上用的 2 字缩写。
#:
#: 为什么需要：快捷行要求「全选 + 各分组的仅选按钮」在**一行内**放下（用户明确要求），
#: 而 `webapp/_css_check.py` 会用真实 CSS 字号做静态宽度估算。分组名改长之后
#: 「仅动力学与输运」这类按钮会把整行挤爆，所以按钮用缩写、完整名放在 tooltip 与
#: 分组标题上。改分组名时**必须同步更新这张表**（有测试盯着）。
GROUP_SHORT: dict[str, str] = {
    "链构象": "构象",
    "空间结构": "结构",
    "取向与结晶": "取向",
    "动力学与输运": "输运",
}


def analysis_groups() -> list[dict]:
    """给界面用的分组结构：``[{"label": 模块名, "short": 按钮缩写, "names": [...]}, ...]``。"""
    return [{"label": label, "short": GROUP_SHORT.get(label, label[:2]),
             "names": list(names)} for label, names in ANALYSIS_GROUPS]


def sort_by_estimate(names: Sequence[str],
                     estimates: Mapping[str, float | None] | None) -> list[str]:
    """按**预估用时从短到长**稳定排序（估不出来的排最后、其余保持原次序）。

    ``run_all(order="shortest")`` 与 Web 端的队列显示共用这一个函数，
    避免两处各写一套排序而慢慢跑偏。
    """
    base = {n: i for i, n in enumerate(names)}
    est = dict(estimates or {})

    def _key(n: str):
        e = est.get(n)
        return (float(e) if e is not None else float("inf"), base.get(n, 0))

    return sorted(names, key=_key)


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
        #: **正在计算的**分析名（``run_all`` 逐项设置）。界面靠它显示
        #: "当前分析项：xxx"，不必去解析进度文字（那种做法一改文案就失效）。
        self.current: str = ""
        #: 上一次 ``run_all`` 的**实际执行顺序**（``order="shortest"`` 时会与
        #: 传入顺序不同），用于自检与界面显示。
        self.run_order: list[str] = []

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
        """运行单个分析项（跑完自动挂上科研 QC 检查项）。"""
        from . import qc

        res = self._run_one(name, params=params, verbose=verbose)
        if res is not None:
            # 把"拓扑能不能提供质量"这一信息随结果带下去，供 QC 判断
            # （.gro/.pdb/.xyz 只有坐标：质量由元素/原子名**推断**，CG/联合原子
            #  体系会算错 —— 实测 MARTINI 珠子 51 amu 被推成 9.0 amu，密度差 5.7×）
            top = str(getattr(self.trajectory, "topology", "") or "")
            ff = top.lower().endswith((".tpr", ".psf", ".prmtop", ".parm7", ".top"))
            res.meta["拓扑文件"] = top.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            res.meta["拓扑含力场信息"] = bool(ff)
            qc.derive_checks(res)          # 统一把散落的判据变成结构化结论
        return res

    def _run_one(self, name: str, *, params: Mapping | None = None,
                 verbose: bool = False) -> AnalysisResult | None:
        """单项分析的实际实现（不含 QC 导出）。"""
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
        if name == "rdf2d":
            groups = self._rdf_groups()
            if len(groups) < 1:
                return None
            return ifc.analyze_rdf_inplane(self.trajectory, groups, sel,
                                           verbose=verbose, **p)
        if name == "comdist":
            if len(self.components) < 2:
                return None
            return ifc.analyze_com_distance(self.trajectory, self.components, sel,
                                            verbose=verbose, **p)
        if name == "contact":
            a, b, la, lb = self._contact_pair()
            if a is None:
                return None
            res = ifc.analyze_contacts(self.trajectory, a, b, sel, verbose=verbose, **p)
            if res is not None and la:
                res.add_notes(f"配对对象：{la} × {lb}"
                              f"（A={a.n_atoms} 原子，B={b.n_atoms} 原子）。"
                              f"选组规则：A 取最大组分，B 取与 A 不同分子的最大组分。")
            return res
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
            ag, lab = self._orientation_target()
            return cry.analyze_orientation(self.trajectory, ag, sel,
                                           label=lab, verbose=verbose, **p)
        if name == "order":
            ag, lab = self._orientation_target()
            return cry.analyze_structural_order(self.trajectory, ag, sel,
                                                label=lab, verbose=verbose, **p)
        if name == "boo":
            from .analysis import boo as boo_mod

            ag, lab = self._orientation_target()
            return boo_mod.analyze_boo(self.trajectory, ag, sel,
                                       label=lab, verbose=verbose, **p)
        if name == "crystal":
            from .analysis import boo as boo_mod

            ag, lab = self._orientation_target()
            return boo_mod.analyze_crystal_regions(self.trajectory, ag, sel,
                                                   label=lab, verbose=verbose, **p)
        if name == "msd":
            groups = self._msd_groups()
            if not groups:
                return None
            return dyn.analyze_msd(self.trajectory, groups, sel, verbose=verbose, **p)
        raise KeyError(f"未知分析项: {name!r}")

    #: 预估用时：探测帧数（先测 2 帧、再测 k 帧，用来分离固定开销与每帧代价）
    ESTIMATE_PROBE_FRAMES = 4
    #: 帧数少于这个数就不预估 —— 预估本身要跑 ``k`` 帧，小体系里这笔开销
    #: 可能比正式分析还大（AdK 10 帧实测：预估 12 s，正式跑 18 s）。
    #: 大体系（46 体系 10001 帧）预估算 17 s，占预计总时长不到 0.2%，非常划算。
    ESTIMATE_MIN_FRAMES = 40

    def warmup(self) -> dict:
        """把**一次性初始化**开销先做掉（不归属于任何分析项）。

        为什么必须做：molnums 展开、键图/连通分量、解包裹机制都是**首次访问才建**。
        实测（AdK water 组分、3 帧、全新进程）：``rg`` 关掉逐分子统计后第一次调用
        **6.40 s**、之后每次 **0.03 s**；开着时稳定 **1.25 s**。那几秒会记在
        "队首那一项"头上 —— 于是只要改一个开关让排序变一下，同一项就可能从
        1.3 s "变成" 6.4 s，看起来像变慢了（总时长其实更短）。预热之后，
        逐项耗时与预估都能跨设置比较。
        """
        from .analysis.base import positions_for

        touched: list[str] = []
        u = self.universe
        for label, fn in (("molnums", lambda: u.atoms.molnums),
                          ("fragments", lambda: u.atoms.fragments)):
            try:
                fn()
                touched.append(label)
            except Exception:  # noqa: BLE001
                pass
        groups = ([self.primary] if self.primary is not None else []) + \
                 [g for g in self.components.values() if g is not None]
        for ag in groups:
            try:
                ag.fragments
            except Exception:  # noqa: BLE001
                pass
            try:
                positions_for(ag, unwrap=True)
            except Exception:  # noqa: BLE001
                pass
        return {"touched": touched, "n_groups": len(groups)}

    def _time_one(self, name: str, params: Mapping, sel) -> float:
        """把帧选择临时换成 ``sel``，跑一次并计时（**结果丢弃**）。"""
        self.frames = sel
        t0 = time.time()
        self.run(name, params=params.get(name), verbose=False)
        return time.time() - t0

    def estimate_times(self, which: Sequence[str] | None = None, *,
                       params: Mapping[str, Mapping] | None = None,
                       probe_frames: int | None = None,
                       cancel: "threading.Event | None" = None,
                       on_item: Callable | None = None) -> dict[str, dict]:
        """先跑几帧，实测每项分析的用时并外推整段跑完要多久。

        为什么不是"按体系规模套公式"：各项分析的常数因子相差好几个量级
        （RDF/接触要建 CSR 邻居表、MSD 要解包裹坐标、取向要按重复单元重组），
        拍系数必然错。这里改成**两点实测**，把固定开销和每帧代价分开：

            per_frame = (t_k - t_1) / (k - n1)      n1 = min(2, k)
            fixed     = t_1 - per_frame * n1
            est       = fixed + per_frame * n_frames

        代价是每项多跑约 ``k`` 帧 —— 对 10001 帧的体系不到千分之一。
        返回 ``{name: {"probe_frames", "fixed_sec", "per_frame_sec",
        "est_sec", "n_frames", "note"}}``；某项估不出来时 ``est_sec=None``
        （帧数太少、组分缺失或直接报错），界面据此显示"—"。

        探测**不写入** ``self.results``，并保证恢复 ``self.frames`` 与
        ``self.current`` —— 预估只是量个速度，不能改动分析状态。
        """
        names = list(which) if which else list(DEFAULT_ORDER)
        params = dict(params or {})
        k = int(self.ESTIMATE_PROBE_FRAMES if probe_frames is None else probe_frames)
        sel_full = self.require_frames()
        n = sel_full.n_frames
        keep_frames, keep_current = self.frames, self.current
        out: dict[str, dict] = {}
        try:
            # 先预热：否则"第一个被探测的项"会把一次性初始化算进自己的预估里，
            # 排序就会因此跑偏
            try:
                self.warmup()
            except Exception:  # noqa: BLE001
                pass
            if n < max(int(self.ESTIMATE_MIN_FRAMES), 2 * (k + 1)):
                # 体系太小：预估要跑 k 帧，占比过高，直接跑更省事
                note = (f"只有 {n} 帧，预估本身的成本已不可忽略"
                        f"（要跑 {k} 帧），直接开跑")
                for name in names:
                    out[name] = {"probe_frames": 0, "n_frames": n, "fixed_sec": None,
                                 "per_frame_sec": None, "est_sec": None, "note": note}
                return out
            if n > k:
                probe = sel_full[::max(1, int(round(n / max(k, 1))))]
            else:
                probe = sel_full
            pk = probe.n_frames
            n1 = min(2, pk)
            for i, name in enumerate(names):
                if cancel is not None and cancel.is_set():
                    break
                if on_item:
                    on_item(i, len(names), name)
                rec = {"probe_frames": pk, "n_frames": n, "fixed_sec": None,
                       "per_frame_sec": None, "est_sec": None, "note": ""}
                try:
                    t1 = self._time_one(name, params, probe.head(n1))
                    tk = self._time_one(name, params, probe)
                except Exception as exc:  # noqa: BLE001
                    rec["note"] = f"估不出来（{type(exc).__name__}: {exc}）"
                    out[name] = rec
                    continue
                per_raw = (tk - t1) / max(pk - n1, 1)
                if per_raw <= 0:
                    # 两次测量没测出差异（固定开销主导或噪声）：**不能**因此报
                    # "每帧 0 ms" 去外推 —— 那会把这一项判成最快，排到队首却跑很久。
                    # 退一步用"平均每帧代价"（含固定开销）兜底，结果偏保守。
                    per, fixed = tk / max(pk, 1), 0.0
                else:
                    per = per_raw
                    fixed = max(0.0, t1 - per * n1)
                est = fixed + per * n
                rec["fixed_sec"] = round(fixed, 3)
                rec["per_frame_sec"] = round(per, 6)
                rec["est_sec"] = round(est, 2)
                rec["note"] = (f"按 {pk} 帧实测外推：固定 {fixed:.2f}s + "
                               f"每帧 {per * 1000:.1f}ms × {n} 帧")
                out[name] = rec
        finally:
            self.frames, self.current = keep_frames, keep_current
        return out

    def run_all(self, which: Sequence[str] | None = None, *,
                params: Mapping[str, Mapping] | None = None,
                outdir: str | None = None, formats: Sequence[str] = ("csv", "png"),
                excel: bool = True, panel_pngs: bool = False,
                verbose: bool = True,
                raise_errors: bool = False,
                progress: Callable | None = None,
                on_result: Callable | None = None,
                order: str = "given",
                estimates: Mapping[str, float | None] | None = None,
                cancel: "threading.Event | None" = None) -> "OrderedDict[str, AnalysisResult]":
        """依次运行多项分析，可选直接导出。

        ``progress`` 是一个 ``callable(frac: float, message: str)`` 回调，
        用于向图形界面汇报进度（``frac`` 取 0–1）。

        ``on_result`` 是一个 ``callable(name, result, elapsed_sec, done, total)``
        回调，在**每一项分析算完后立即**调用 —— Web 版靠它把已算好的结果
        实时推给页面，而不必等全部跑完。

        ``order="shortest"`` 且给了 ``estimates``（:meth:`estimate_times` 的结果）
        时，按**预估用时从短到长**执行，好让结果尽早出现在页面上；估不出来的项
        排在最后，其余保持原有相对次序（稳定排序）。默认 ``"given"`` 保持
        CLI / 桌面版的原行为不变。

        ``cancel`` 传入一个 :class:`threading.Event` 时，会在**每项分析开始前**
        检查它；已置位则停止并把已完成的项作为结果返回。
        ``panel_pngs=True`` 时额外为每个面板单独导出一张 PNG。
        """
        names = list(which) if which else list(DEFAULT_ORDER)
        params = dict(params or {})
        if order == "shortest" and estimates:
            names = sort_by_estimate(names, estimates)
        self.results = OrderedDict()
        #: ``{分析名: 耗时秒数}``，供界面显示"哪一项最慢"
        self.timings: dict[str, float] = {}
        self.run_order: list[str] = list(names)
        # 预热放在计时之前：一次性初始化（分子编号展开/键图/解包裹）不该算进
        # 队首那一项的耗时里（否则改个开关就能让同一项"看起来变慢"）
        if progress:
            progress(0.0, "预热（一次性初始化：分子编号 / 键图 / 解包裹）…")
        try:
            self.warmup()
        except Exception:  # noqa: BLE001
            pass
        sel = self.require_frames()
        if verbose:
            print(f"[帧选择] {sel.describe()}")
            for n in sel.notes:
                print(f"          {n}")
        total = max(len(names), 1)
        self.cancelled = False
        self.current = ""
        for i, name in enumerate(names):
            if cancel is not None and cancel.is_set():
                self.cancelled = True
                if verbose:
                    print("[取消] 收到取消请求，停止后续分析")
                break
            title = ANALYSIS_TITLES.get(name, name)
            self.current = name                    # 供界面显示"当前分析项"
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
        """接触分析的 A/B 两组。

        选法：A = 原子数最多的组分；B = **与 A 不在同一批分子里**的、原子数最多的
        组分；都不满足时退回同组自接触。

        为什么不能直接取"前两个组分"（1.0.0 的做法）：遇到"糖基被单独识别成一个
        组分"的糖蛋白（protein + sugar 共价相连）时，A/B 会选成一对**同分子**的
        组分，inter 口径下接触恒为 0 —— 实测 md_biopolymer_nowater 报出的
        "平均接触对数 = 0" 而最小原子间距只有 1.39 Å（N-糖苷键长），是个毫无意义
        的数。返回 ``(group_a, group_b, name_a, name_b)``。
        """
        from .analysis.interface import shares_one_molecule

        items = sorted(self.components.items(), key=lambda kv: -kv[1].n_atoms)
        if not items:
            return None, None, "", ""
        name_a, a = items[0]
        for name_b, b in items[1:]:
            if not shares_one_molecule(a, b):
                return a, b, name_a, name_b
        return a, a, name_a, name_a

    def _orientation_target(self):
        """取向/有序度的分析对象：**整个组分**（含该组分的全部分子）。

        为什么改：1.0.0 只把"最大的一条链"（``self.primary``）交给取向与有序度
        分析，却用整个体系/组分的名义输出——典型症状就是"只算一条链却命名整个
        膜"。取向与有序度都是**集成量**，必须对该类分子的全部实例做集合平均；
        结果里会写明链段来自多少个分子。

        选法：取与主链原子重叠最多的那个组分；没有组分信息时退回主链。
        """
        import numpy as _np

        if self.components and self.primary is not None:
            pset = {int(a) for a in _np.asarray(self.primary.indices)}
            best_name = None
            best_ag = None
            best_ov = -1
            for nm, ag in self.components.items():
                if ag is None or ag.n_atoms == 0:
                    continue
                ov = len(pset & {int(a) for a in _np.asarray(ag.indices)})
                if ov > best_ov:
                    best_name, best_ag, best_ov = nm, ag, ov
            if best_ag is not None and best_ov > 0:
                try:
                    from .analysis.conformation import molecule_slices

                    ms = molecule_slices(best_ag)
                    n_mol = len(ms[1]) if ms is not None else 1
                except Exception:  # noqa: BLE001
                    n_mol = 1
                return best_ag, (f"{best_name}（{n_mol} 个分子，{best_ag.n_atoms:,} 原子）")
        return self.primary, self.primary_label

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
