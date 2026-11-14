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
    # 「分析对象」默认 **分子 / 链**（逐分子算 Rg；与 Forcite / "分子有多大"同口径）。
    # 另一档是 "system" = 整体体系（整组 Rg）—— 两档互斥，一次只算一个。
    "rg": {"mass_weighted": True, "object": "molecule"},
    "ree": {},
    "dihedral": {"mode": "auto", "gauche_edge": 120.0, "bin_width": 1.0},
    # 默认三个晶向一起出（C100/C010/C001）；界面/2D RDF/质心距仍用 axis=2
    "density": {"axis": "all", "nbins": None, "mode": "mass"},
    "rdf": {"rmax": 12.0, "nbins": 120, "compare_halves": False,
            "mode": "total",
            "distance": "minimage", "drop_overlap": 0.0, "rho_scale": 1.0,
            "forcite": False},
    "rdf2d": {"plane_axis": 2, "rmax": 70.0, "nbins": 140},
    # 质心距默认**用最小镜像**（1.0.5 起，用户要求）：两个质心常分处盒边界两侧，
    # 直接相减会得到 ~L 的假跳变。要复现作者 COM.py 的"包裹坐标相减"再手动关掉。
    "comdist": {"axis": 2, "pbc": True},
    "contact": {"cutoff": 5.0},
    "interface": {"axis": 2, "nbins": 120},
    # 取向链段默认取**化学重复单元**（1.0.1 起；1.0.0 为几何骨架 "backbone"）
    # p2_axis：P₂ 的参考轴，默认 z 轴（与 Forcite / 外部脚本口径一致）
    "orientation": {"mode": "repeat", "stride": 1, "p2_axis": "z"},
    # trans/gauche 判据与「二面角分析」共用同一阈值：改一处两处都变，
    # 免得同一条轨迹上两个模块报出不同的"trans 构象比例"。
    # ⚠️ 每个分析项在这张表里只能出现**一次**：字典字面量里重复的键会被后者
    #    静默覆盖（这里曾同时写了 `"order": {"gauche_edge": …}` 和 `"order": {}`，
    #    结果默认阈值被吃掉，只有显式传参才生效）。
    "order": {"gauche_edge": 120.0},
    # `object` 必须是 **atom**：Forcite 的 MSD 是逐原子追踪后平均（实测 PE-PP-MAPE
    # 单链：t=1 ps 0.96075665、t=999 ps 21.51613092，逐点一致）。默认写成
    # `molecule` 时同一组原子会给 2.873 / 75.707 —— 用户实测的"MSD 对不上、
    # 但 Rg/密度对得上"就是这一处默认值造成的。
    "msd": {"object": "atom", "max_particles": None, "workers": 0},
    # VACF：max_lag 留空 = 用满整条轨迹（与 Forcite 一样逐 lag 全给）
    "vacf": {"max_lag": None},
    # τ 轴默认**用满整条轨迹**；想收缩就传 {"tau_intervals": 5}（= 总时长/5）
    # 键取向序参数 / 晶体-非晶识别（1.0.2）：默认用 Lechner–Dellago 平均版 q̄6，
    # 邻域半径留空 = 自动按最近邻距离中位数推定
    "boo": {"cutoff": None, "averaged": True},
    "crystal": {"cutoff": None, "q6_solid": 0.5, "min_cluster": 10},
}

#: 默认执行顺序（按设计大纲第 25–27 章）
DEFAULT_ORDER = ["rg", "ree", "dihedral", "density", "rdf", "contact",
                 "interface", "orientation", "order", "boo", "crystal", "msd",
                 "vacf"]

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
    "vacf": "速度自相关函数 VACF",
}

#: 按设计大纲的**模块**给分析项分组：``[(模块名, (分析项, ...)), ...]``。
#: 命令行、桌面操作台与 Web 界面共用这一份定义，界面上「分析功能」的预设按钮
#: 与「图表导航」的分层都从这里生成，不会各写一份导致对不上。
ANALYSIS_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("链构象", ("rg", "ree", "dihedral")),
    ("空间结构", ("density", "rdf", "rdf2d", "comdist", "contact", "interface")),
    ("取向与结晶", ("orientation", "order", "boo", "crystal")),
    ("动力学与输运", ("msd", "vacf")),
]

#: 分组在**快捷按钮行**上用的 2 字缩写。
#:
#: 为什么需要：快捷行要求「全选 + 各分组的仅选按钮」在**一行内**放下（用户明确要求），
#: 而 `test/webapp/_css_check.py` 会用真实 CSS 字号做静态宽度估算。分组名改长之后
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

    def __init__(self, topology: str, trajectory: str | None = None, *,
                 guess_bonds: bool | None = None,
                 velocities: bool | None = None,
                 autodetect: bool | None = None):
        # `guess_bonds=True`：忽略拓扑里已有的键、按坐标重新猜（CONECT 不完整的
        # PDB 用）。**必须能一路传到 Analyzer** —— 否则 Web 端把参数传给了
        # `load_trajectory` 做校验、Analyzer 又重新加载一次、参数就丢了。
        # `velocities=True`：MS 的 .trj 转成 TRR（坐标 + 速度）后再读，同理必须透传。
        self.trajectory: MDTrajectory = load_trajectory(
            topology, trajectory, guess_bonds=guess_bonds, velocities=velocities,
            **(dict() if autodetect is None else {"autodetect": bool(autodetect)}))
        self.frames: FrameSelection | None = None
        self.primary = None            # 主链（构象/结晶分析对象）
        self.primary_label: str = "链"
        self.components: "OrderedDict[str, object]" = OrderedDict()
        self.extra_components: "OrderedDict[str, object]" = OrderedDict()
        #: **候选组分清单**（自动识别出的全部组分 + 已定义的集合），与"当前勾选
        #: 参与分析的 `components`"分开存。界面的组分胶囊 / 主链下拉框都读这一份，
        #: 所以取消勾选不会让它们从列表里消失（详见 `auto_setup` 的说明）。
        self.catalog: "OrderedDict[str, object]" = OrderedDict()
        #: 已定义的集合名 → 原子组（不参与分析时也能被"改选为主链"重新启用）
        self.set_groups: dict[str, object] = {}
        #: `extra_components` / 组分 → MSD 的追踪语义（molecule / atom）。
        #: 由「分析集合（Set）」决定：整分子选中→质心，被切碎→按原子。
        self.msd_group_objects: dict[str, str] = {}
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
        comps = component_groups(u)
        if chains:
            ch = chains[0]
            # ⭐「最大链」是**一个小分子**、而同类分子有很多个时，主链应该取**整个组分**。
            # 为什么：这种体系（例如 122 个 C12 分子）里"分子有多大"的正确回答是
            # **逐分子算 Rg 再平均**，而不是随便挑一个分子 —— 单分子的 Rg 涨落大、
            # 曲线形状与"全部分子的平均"完全不同。实测用户拿 Forcite 的 Rg
            # （=122 个分子的平均）与界面上导出的"最大链"单分子曲线比，逐帧相关系数
            # 只有 0.84、均值差 1%，看起来像"算错了"，其实是**分析对象选错**。
            # 判据：该链占体系原子数 < 10%，且包含它的那个组分有 ≥10 个分子、
            #       原子数是该链的 ≥5 倍。
            chosen = None
            if u.atoms.n_atoms >= 10 * ch.n_atoms:
                idx_ch = set(np.asarray(ch.indices, dtype=int).tolist())
                for name, ag in comps.items():
                    # ⚠️ 用 molecule_count() 而不是直接 ag.molnums：退化拓扑下
                    # molnums 会被删掉（Material Studio 的 PDB 只有 2 个残基），
                    # 直接访问会抛 NoDataError 让整个体系打不开。
                    from .topology import molecule_count

                    n_mol = molecule_count(ag) if getattr(ag, "n_atoms", 0) else 0
                    if (ag.n_atoms >= 5 * ch.n_atoms and n_mol >= 10
                            and idx_ch.issubset(set(np.asarray(ag.indices,
                                                              dtype=int).tolist()))):
                        chosen = (name, ag, n_mol)
                        break
            if chosen is not None:
                name, ag, n_mol = chosen
                self.primary = ag
                self.primary_label = (f"组分 {name}（{n_mol} 个分子，"
                                      f"{ag.n_atoms} 原子）")
            else:
                self.primary = ch
                seg = str(ch.segids[0])
                self.primary_label = f"最大链 ({seg}, {ch.n_atoms} 原子)"
        else:
            self.primary = u.atoms
            self.primary_label = "全部原子"

        # protein/polymer 优先，其余按原子数降序
        priority = {"protein": 0, "nucleic": 1, "sugar": 2, "polymer": 3,
                    "water": 4, "ion": 5, "other": 6}
        items = sorted(comps.items(),
                       key=lambda kv: (priority.get(kv[0].split("[")[0], 9), -kv[1].n_atoms))
        #: **候选清单**（全部自动识别出的组分，不截断）。`components` 只是"当前勾选
        #: 参与分析的那几个"。两者必须分开：界面上把组分全部取消勾选之后，候选清单
        #: 仍要在（否则第二轮补充分析时就没得选了 —— 用户实测：第一轮只跑 MSD、
        #: 没勾任何自动组分，刷新一次整张组分表就空了）。
        self.catalog = OrderedDict(items)
        self.components = OrderedDict(items[:max_components])
        self.extra_components = OrderedDict()
        if not self.components:
            self.components = OrderedDict([("all", u.atoms)])
        if not self.catalog:
            self.catalog = OrderedDict([("all", u.atoms)])
        return {"primary": self.primary_label, "components": list(self.components),
                "catalog": list(self.catalog)}

    def set_primary(self, ag, label: str = "链") -> None:
        self.primary = ag
        self.primary_label = label

    def set_components(self, components: Mapping[str, object]) -> None:
        self.components = OrderedDict(components)

    def add_sets(self, specs, components: Mapping[str, object] | None = None) -> list:
        """按「分析集合（Set）」定义生成原子组，注册为 `set:<名字>` 成分。

        返回每个集合的 info（含 `kind`：`molecule` = 整分子选中、可按质心追踪；
        `atom` = 被元素筛选切碎、只能按原子追踪）。
        """
        from .sets import build_set

        # 先清掉上一次定义的集合（重新定义时不能残留旧集合）
        for key in [k for k in self.components if str(k).startswith("set:")]:
            self.components.pop(key, None)
        for key in [k for k in self.catalog if str(k).startswith("set:")]:
            self.catalog.pop(key, None)
        for key in [k for k in self.msd_group_objects if str(k).startswith("set:")]:
            self.msd_group_objects.pop(key, None)
        self.set_groups = {k: v for k, v in self.set_groups.items()
                           if not str(k).startswith("set:")}
        base = dict(components or self.components)
        infos = []
        for spec in specs or []:
            spec = dict(spec or {})
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            ag, info = build_set(self.universe, spec, base)
            key = f"set:{name}"
            # 作为**一等公民组分**放进 components：RDF / 接触 / 界面 / 密度 也能选它
            self.components[key] = ag
            # 同时进候选清单：取消勾选后它仍在下拉框里，随时能再勾回来
            self.catalog[key] = ag
            self.set_groups[key] = ag
            self.msd_group_objects[key] = info["kind"]
            # 「可视化选择」的集合带着完整原子索引列表 —— 那是给后端重建用的，
            # 不必回传界面（可能上万个数），只报个数。
            out_info = {**info, "name": key}
            idx = out_info.pop("indices", None)
            if idx is not None:
                out_info["n_indices"] = len(idx)
            infos.append(out_info)
        return infos

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
            # 把"质量从哪来"随结果带下去，供 QC 判断。**区分两种情况**（用户要求）：
            #   ① MDAnalysis 自己读到了质量（PDB 带元素列、力场拓扑自带…）→ QC 不提醒；
            #   ② 连 MDAnalysis 都没有质量，是本工具按元素表补的 → 才提醒
            #      （CG/联合原子体系会把珠子质量补错数倍：实测 MARTINI 51 → 9 amu，
            #       质量密度差 5.7×，而且**不报错**）。
            top = str(getattr(self.trajectory, "topology", "") or "")
            ff = top.lower().endswith((".tpr", ".psf", ".prmtop", ".parm7", ".top"))
            res.meta["拓扑文件"] = top.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            res.meta["拓扑含力场信息"] = bool(ff)
            src = str(((getattr(self.trajectory, "object_info", None) or {})
                       .get("sources") or {}).get("masses") or "")
            inferred = ("由元素查表" in src) or ("由元素" in src) or ("推断" in src)
            res.meta["质量来源"] = ("由工具按元素/原子名推断" if inferred
                                    else "拓扑/MDAnalysis 提供")
            # 「推断出来的质量**可不可信**」取决于有多少原子连元素都认不出来。
            # PDB / XSD / GRO 这类拓扑本来就不带力场质量，按元素查表补是**常规做法**
            # （用户明确要求：这种情况 QC 应判「正常」，不该报异常）；
            # 只有 CG/联合原子体系（珠子名推不出元素 → 质量被按 1.0 处理）才需要提醒。
            oi = getattr(self.trajectory, "object_info", None) or {}
            res.meta["质量未知元素数"] = int(oi.get("n_missing_mass", 0) or 0)
            res.meta["质量来源原文"] = src
            # 键表完整性：拓扑自带键表残缺时（MS 的 PDB 常见）工具会按坐标补键，
            # 这里把"补过"如实带进结果 —— 否则用户看到"122 分子"会以为是原样，
            # 看到"243 分子"又不知道为什么（实测正是后者让逐分子 Rg 少了一半）。
            binfo = getattr(self.trajectory, "bond_guess_info", None) or {}
            if binfo.get("forced_repair") or binfo.get("replaced_existing"):
                res.summary["键表修复"] = str(binfo.get("note", ""))
                res.meta["键表修复"] = {
                    "fragments_before": binfo.get("fragments_before"),
                    "fragments": binfo.get("fragments"),
                    "bonds_before": binfo.get("replaced_existing"),
                    "bonds": binfo.get("bonds"),
                    "degenerate_note": binfo.get("degenerate_note", ""),
                    "forced": bool(binfo.get("forced_repair")),
                }
                res.add_notes("**键表修复**：" + str(binfo.get("note", "")))
            # 坐标尺度体检（用共价键长当尺子）：轨迹被缩放过 / 单位不是 Å 时，
            # Rg / R_ee / 界面厚度等**所有长度量**都会按同一比例偏大偏小，而且不报错。
            # 这里把结论随结果带下去，qc.derive_checks 会据此出一条检查项。
            scale = self.bond_scale()
            if scale.get("ok") is not None:
                res.summary["坐标尺度校验"] = scale["note"]
                res.meta["坐标尺度"] = {k: scale.get(k) for k in
                                       ("ok", "median", "expected", "ratio")}
                if not scale["ok"]:
                    res.add_notes("⚠️ 坐标尺度异常 —— " + scale["note"])
            qc.derive_checks(res)          # 统一把散落的判据变成结构化结论
        return res

    def bond_scale(self) -> dict:
        """用共价键长标定坐标尺度（每个 Analyzer 只算一次，便于多处复用）。"""
        cached = getattr(self, "_bond_scale_cache", None)
        if cached is not None:
            return cached
        out: dict = {"ok": None, "note": ""}
        try:
            from .topology import bond_scale_report

            ag = self.primary if self.primary is not None else self.universe.atoms
            out = bond_scale_report(ag)
        except Exception as exc:                                # noqa: BLE001
            out = {"ok": None, "note": f"坐标尺度校验失败：{type(exc).__name__}: {exc}"}
        self._bond_scale_cache = out
        return out

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
        self.skip_reason = ""          # 每次分析前清掉上一次的原因

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
                self.skip_reason = "没有勾选任何组分（RDF 至少需要一个组分）"
                return None
            return ifc.analyze_rdf(self.trajectory, groups, sel, verbose=verbose, **p)
        if name == "rdf2d":
            groups = self._rdf_groups()
            if len(groups) < 1:
                self.skip_reason = "没有勾选任何组分（面内 RDF 至少需要一个组分）"
                return None
            # 逐帧叶层：`slab_ref_from` 给"膜"那一侧组分的名字，就用它的**逐帧质心**
            # 当叶层阈值（作者 Lipid_RDF.py 正是 z > 膜质心(t)）。给固定 slab 时优先用
            # 固定值（两者互斥，见 analyze_rdf_inplane 的文档）。
            p2 = dict(p)
            ref_name = p2.pop("slab_ref_from", None)
            if ref_name and p2.get("slab") is None:
                ref = (self.components.get(str(ref_name))
                       or self.extra_components.get(str(ref_name)))
                if ref is None:
                    raise SelectionError(f"组分 {ref_name!r} 不存在，无法作为叶层参考")
                p2["slab_ref"] = ref
            return ifc.analyze_rdf_inplane(self.trajectory, groups, sel,
                                           verbose=verbose, **p2)
        if name == "comdist":
            if len(self.components) < 2:
                self.skip_reason = "两组分质心距需要至少勾选 2 个组分"
                return None
            return ifc.analyze_com_distance(self.trajectory, self.components, sel,
                                            verbose=verbose, **p)
        if name == "contact":
            a, b, la, lb = self._contact_pair()
            if a is None:
                self.skip_reason = "接触分析需要至少勾选 1 个组分"
                return None
            res = ifc.analyze_contacts(self.trajectory, a, b, sel, verbose=verbose, **p)
            if res is not None and la:
                res.add_notes(f"配对对象：{la} × {lb}"
                              f"（A={a.n_atoms} 原子，B={b.n_atoms} 原子）。"
                              f"选组规则：A 取最大组分，B 取与 A 不同分子的最大组分。")
            return res
        if name == "interface":
            if len(self.components) < 2:
                names = list(self.components)
                self.skip_reason = (
                    f"界面宽度分析需要至少 **2 个组分**，当前只有 {len(self.components)} 个"
                    f"（{names}）。如果体系其实是多物种混合、但被识别成了一个组分"
                    f"（典型情形：resname 全一样的中文/自定义拓扑，例如 PE/PP/MAPE 共混"
                    f"10 条链都叫 ETH），请到「分析集合」里按**原子数 / 元素 / 链长**"
                    f"把各物种拆成独立集合，再勾选它们参与分析。")
                return None
            # 默认取**原子数最多的两个组分**当作界面两侧：分层体系里这两相
            # 才是真正的界面两侧；若按字典序取前两个，可能取到"溶解在里面的
            # 小分子 + 基体"，那对组合没有平界面。
            #
            # ⚠️ 但"最大的两个"对三元/多元共混是不够的（用户实测：勾了 3 个组分，
            #    界面分析永远只算最大的那两个，换体系还是老的一对）→ 现在支持
            #    params["pair"] 显式指定 A/B，并且**失效时回退**而不是直接报错。
            ordered = sorted(self.components.items(), key=lambda kv: -kv[1].n_atoms)
            auto_pair = (ordered[0][0], ordered[1][0])
            want, pair_note = None, ""
            raw = p.pop("pair", None)
            if raw:
                try:
                    cand = tuple(str(x) for x in raw)[:2]
                except TypeError:
                    cand = ()
                if len(cand) == 2 and cand[0] != cand[1]:
                    missing = [n for n in cand if n not in self.components]
                    if missing:
                        pair_note = (f"指定的界面组分 {missing} 不在当前勾选的组分里"
                                     f"（现有：{list(self.components)}），"
                                     f"已回退到原子数最多的 {auto_pair[0]} × {auto_pair[1]}。")
                    else:
                        want = cand
                else:
                    pair_note = (f"界面 A/B 组必须是不相同的两个组分名（收到 {raw!r}），"
                                 f"已回退到原子数最多的 {auto_pair[0]} × {auto_pair[1]}。")
            pair = want or auto_pair
            res = ifc.analyze_interface_width(self.trajectory, self.components, sel,
                                              pair=pair, verbose=verbose, **p)
            if res is not None:
                if pair_note:
                    res.add_notes("⚠️ " + pair_note)
                if len(self.components) > 2:
                    others = [n for n in self.components if n not in pair]
                    res.add_notes(
                        f"当前勾选了 {len(self.components)} 个组分，本项**每次只比较一对**："
                        f"{pair[0]} × {pair[1]}"
                        + ("（自动取原子数最多的两个）" if not want else "（按参数指定）")
                        + f"。未参与的是 {others}；要换组合请在参数里设置"
                          f"「界面 A 组 / 界面 B 组」。")
            return res
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
                self.skip_reason = "没有勾选任何组分（MSD 至少需要一个组分）"
                return None
            # ⚠️ **不要**再用集合的 kind 覆盖用户的「MSD 追踪对象」选择。
            # 旧代码把 `msd_group_objects`（= 集合的 kind，例如"整分子"集合是
            # molecule）塞进 `object_by_group`，于是界面上明明选了「原子」，
            # 集合仍按**分子质心**追踪 —— 实测 PE-PP-MAPE 单链：原子口径
            # t=1 ps 0.9608 / t=999 ps 21.516（= Forcite），质心口径 0.066 /
            # 0.94，用户看到的就是这个差异。口径应当只由界面那一个选择决定
            # （"想按质心追踪" = 直接选「分子质心」）。
            return dyn.analyze_msd(self.trajectory, groups, sel, verbose=verbose, **p)
        if name == "vacf":
            groups = self._msd_groups()
            if not groups:
                self.skip_reason = "没有勾选任何组分（VACF 至少需要一个组分）"
                return None
            return dyn.analyze_vacf(self.trajectory, groups, sel,
                                    verbose=verbose, **p)
        raise KeyError(f"未知分析项: {name!r}")

    #: 预估用时：**连续**探测帧数（先丢 2 帧热身，再测 m 帧与 2m 帧，两点定斜率）
    ESTIMATE_PROBE_FRAMES = 16
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
        """把帧选择临时换成 ``sel``，跑一次并计时（**结果丢弃**）。

        ⚠️ **计时前必须清空展开坐标缓存**（``positions_for`` 的 ``_UNWRAP_CACHE``）。
        以前不清：两次探测里 ``t1`` 先访问了前 n1 帧、``tk`` 再访问前 n1 帧（命中
        缓存）+ 后面几帧（冷），于是 ``tk − t1`` 只量到"1 帧的冷开销"，斜率被
        严重低估 —— 正式跑时每一帧都是冷的。实测 20PE30（301 帧）：

        · rg：预估 1.34 s / 实际 10.18 s（**−87%**）
        · dihedral：预估 2.43 s / 实际 12.67 s（**−81%**）
        · orientation：预估 0.42 s / 实际 0.86 s（−51%）

        清缓存后两次测量都是"从冷开始"，固定开销各算一次、斜率才是真实的每帧代价。
        """
        from .analysis.base import clear_unwrap_cache

        clear_unwrap_cache()
        # 每一次探测前问一次取消：3 次探测 × 各 2m 帧在大体系上也要好几十秒
        from .cancel import check as _cancel_check

        _cancel_check()
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
        # 预估阶段也必须能取消：原先这里**没挂**取消令牌，于是"点了取消没效果"
        # （每项要跑 3 次探测、每次 2m 帧，4.7 万原子体系上一项就好几分钟）。
        from .cancel import AnalysisCancelled
        from .cancel import bind as _cancel_bind
        from .cancel import clear as _cancel_clear
        from .cancel import check as _cancel_check

        _cancel_bind(cancel)
        try:
            # 预热同样可能很慢（分子划分 / 键表 / 首次 unwrap），先查一次
            _cancel_check()
            if cancel is not None and cancel.is_set():
                raise AnalysisCancelled("用户取消了本次分析")
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
            # 探测帧必须是**连续的一段**（头 2m 帧），而且先丢一小段热身：
            #   · 正式跑是连续读帧，"跨步抽 4 帧"的每帧代价明显偏低
            #     （实测 20PE30 上 rg：跨步 16 ms/帧 vs 连续 26–33 ms/帧）；
            #   · 第一次 ``run`` 会带上懒初始化（分子划分/键表/首次 unwrap），
            #     不热身就会把它算进斜率，把每帧代价压到离谱的小值
            #     （实测 rg 曾估出 3.8 ms/帧，真实 33 ms/帧 → 预估偏小 80%）。
            m = max(int(k), 4)
            probe = sel_full.head(min(2 * m, n))
            pk = probe.n_frames
            m = max(2, min(m, pk // 2))
            for i, name in enumerate(names):
                if cancel is not None and cancel.is_set():
                    break
                if on_item:
                    on_item(i, len(names), name)
                rec = {"probe_frames": pk, "n_frames": n, "fixed_sec": None,
                       "per_frame_sec": None, "est_sec": None, "note": ""}
                m_head = m
                if name == "vacf":
                    if pk < 3:
                        rec["note"] = "帧数太少（VACF 至少需要 3 帧），不给预估"
                        out[name] = rec
                        continue
                    m_head = max(3, min(m, pk - 3))   # VACF 的定义要求 ≥3 帧
                try:
                    # 热身（丢弃）：让懒初始化不落到斜率里
                    self._time_one(name, params, probe.head(min(2, pk)))
                    t1 = self._time_one(name, params, probe.head(m_head))
                    tk = self._time_one(name, params, probe)
                except AnalysisCancelled:
                    raise                       # 用户取消：明确向上报，别当"估不出来"
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    if "没有速度" in msg:
                        rec["note"] = ("这条轨迹没有速度数据：VACF 需要带速度的轨迹"
                                       "（MS 的 .trj 会转 TRR 后自带速度）")
                    else:
                        rec["note"] = f"估不出来（{type(exc).__name__}: {exc}）"
                    out[name] = rec
                    continue
                per_raw = (tk - t1) / max(pk - m_head, 1)
                if per_raw <= 0:
                    # 两次测量没测出差异（固定开销主导或噪声）：**不能**因此报
                    # "每帧 0 ms" 去外推 —— 那会把这一项判成最快，排到队首却跑很久。
                    # 退一步用"平均每帧代价"（含固定开销）兜底，结果偏保守。
                    per, fixed = tk / max(pk, 1), 0.0
                else:
                    per = per_raw
                    fixed = max(0.0, t1 - per * m_head)
                # 探测只跑 32 帧，正式跑可能持续几十秒 —— 满载后 CPU 降频会让
                # 真实每帧代价明显变高（实测 20PE30 的 rg：探测 16 ms/帧、
                # 正式 33 ms/帧）。所以这个数只是量级参考；跑起来之后
                # 前端/后端会**用实测耗时在线校正**（见 session._on_result）。
                est = fixed + per * n
                rec["fixed_sec"] = round(fixed, 3)
                rec["per_frame_sec"] = round(per, 6)
                rec["est_sec"] = round(est, 2)
                rec["note"] = (f"按前 {pk} 帧实测外推：固定 {fixed:.2f}s + "
                               f"每帧 {per * 1000:.1f}ms × {n} 帧"
                               f"（长时间满载会降频，实际可能偏高 30–60%）")
                out[name] = rec
        finally:
            self.frames, self.current = keep_frames, keep_current
            _cancel_clear()          # 预估结束后把令牌摘掉，避免影响后续 run_all
        return out

    def _annotate_frames(self, res, sel):
        """**单帧体系**（只读了结构文件）给结果打标 + 一条说明。

        用户实测关心的问题：只上传结构文件时"图表是不是就没了"。实际分成两类：
        · **随时间变化**的曲线（Rg(t)、接触数(t)、MSD…）只有 1 个点 —— 相当于快照；
        · **静态**面板（RDF、密度分布、各种分布直方图）本来就是单帧即可，照常有效。
        所以这里只加标记与一句话说明，让界面把这件事写清楚，而不是让人以为图坏了。
        """
        try:
            n = int(getattr(sel, "n_frames", 0) or 0)
        except Exception:                                        # noqa: BLE001
            n = 0
        if res is None or n > 1:
            return res
        res.meta["single_frame"] = 1
        res.meta["n_frames"] = n
        # **把"随时间变化"的面板整个去掉**：只有 1 帧时它就是一个点，
        # 画出来既没信息量、又让人以为"时间序列坏掉了"（用户实测提问：
        # "xsd 不是只有一帧吗，怎么还能绘制出随时间变化的图"）。
        # 静态面板（RDF / 密度 / 各种分布）照常保留 —— 那些本来就不依赖帧数。
        try:
            n_panel = int(res.panel_count)
        except Exception:                                    # noqa: BLE001
            n_panel = 0
        remap: dict[int, int] = {}
        dropped: list[int] = []
        for i in range(n_panel):
            xl = str(getattr(res.panel(i), "xlabel", "") or "")
            if ("时间" in xl) or ("time" in xl.lower()):
                dropped.append(i)
                continue
            remap[i] = len(remap)
        if dropped and not remap:
            # **所有面板都是时间序列**（质心距 / 有序度 / 晶区…）：单帧下这张图
            # 没有任何可看的东西，整项记为"跳过"，而不是留一堆 1 点的空图。
            self.skip_reason = (
                f"只有 {n} 帧（只读了结构文件）：这一项的 {len(dropped)} 张图"
                "全是随时间变化，单帧下没有可看的内容，已省略")
            return None
        if dropped:
            keep: list = []
            for c in res.curves:
                if c.panel in remap:
                    c.panel = remap[c.panel]
                    keep.append(c)
            res.curves = keep
            res.panels = [res.panels[i] for i in sorted(remap)
                          if i < len(res.panels)]
        note = (f"只算了 {n} 帧（只读了结构文件，没配轨迹？）："
                + (f"**随时间变化**的 {len(dropped)} 张面板已省略（1 个点没有意义），"
                   "下面都是**静态分布**图（RDF / 密度 / 各种分布），这些与帧数无关、"
                   "结果有效。"
                   if dropped else
                   "结果是**单帧快照**；RDF / 密度 / 各种分布这类静态面板不受影响。"))
        res.add_notes(note)
        return res

    def run_all(self, which: Sequence[str] | None = None, *,                params: Mapping[str, Mapping] | None = None,
                outdir: str | None = None, formats: Sequence[str] = ("csv", "png"),
                excel: bool = True, panel_pngs: bool = False,
                verbose: bool = True,
                raise_errors: bool = False,
                progress: Callable | None = None,
                on_result: Callable | None = None,
                order: str = "given",
                estimates: Mapping[str, float | None] | None = None,
                keep_existing: bool = False,
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
        ``keep_existing=True``（Web 端"扩展分析"用）：**不清空** ``self.results`` /
        ``self.timings``，本次只算 ``which`` 里还没结果的项，旧结果原样保留 ——
        否则扩展跑完以后，之前那几项在 ``self.results`` 里就没了，导出会漏掉它们。
        ``panel_pngs=True`` 时额外为每个面板单独导出一张 PNG。
        """
        names = list(which) if which else list(DEFAULT_ORDER)
        params = dict(params or {})
        if order == "shortest" and estimates:
            names = sort_by_estimate(names, estimates)
        if keep_existing:
            self.results = getattr(self, "results", None) or OrderedDict()
        else:
            self.results = OrderedDict()
        #: ``{分析名: 耗时秒数}``，供界面显示"哪一项最慢"
        self.timings = dict(getattr(self, "timings", {}) or {}) if keep_existing else {}
        #: ``{分析名: 跳过原因}``：体系不满足前提（如没有晶胞信息 / 缺组分）的项。
        #: 与"失败"区分开 —— 这类不是程序出错，界面上应当安静地跳过并给出原因。
        self.skips: dict[str, str] = {}
        #: 单项分析内部判断"这项做不了"时写的原因（供 :attr:`skips` 使用）
        self.skip_reason: str = ""
        self.run_order: list[str] = list(names)
        # 取消令牌挂到模块级：分析内部的长循环（逐帧 / 帧内分块）会检查它，
        # 于是「取消」不必等当前这一项算完（用户要求立即生效）。
        # ⚠️ 必须在**热身之前**挂上：热身要建分子划分 / 键图 / 首次解包裹，
        #    4.7 万原子体系上本身就要几秒到几十秒，之前这段是完全不可中断的。
        from .cancel import AnalysisCancelled
        from .cancel import bind as _cancel_bind
        from .cancel import clear as _cancel_clear

        _cancel_bind(cancel)
        # 预热放在计时之前：一次性初始化（分子编号展开/键图/解包裹）不该算进
        # 队首那一项的耗时里（否则改个开关就能让同一项"看起来变慢"）
        if progress:
            progress(0.0, "预热（一次性初始化：分子编号 / 键图 / 解包裹）…")
        try:
            self.warmup()
        except AnalysisCancelled:
            self.cancelled = True
            _cancel_clear()
            return self.results
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
            except AnalysisCancelled:
                # 用户取消：不算失败；已完成的项保留，剩余项不再跑
                self.cancelled = True
                break
            except Exception as exc:  # noqa: BLE001
                # 「做不了」和「算错了」要分开：像"没有盒信息"这种，是体系本身
                # 不满足该项的前提（用户实测：非周期体系跑 RDF 直接抛
                # ValueError: 当前帧没有模拟盒信息），不该在界面上显示成"失败"。
                from .analysis.interface import BoxMissing

                if isinstance(exc, BoxMissing):
                    self.skips[name] = str(exc).split("\n")[0]
                    if verbose:
                        print(f"  -- 跳过：{self.skips[name]}")
                    continue
                if raise_errors:
                    raise
                print(f"  !! {name} 失败: {type(exc).__name__}: {exc}")
                continue
            if res is None:
                self.skips[name] = self.skip_reason or "体系缺少这项分析所需的组分"
                if verbose:
                    print(f"  -- 跳过：{self.skips[name]}")
                continue
            res = self._annotate_frames(res, sel)
            if res is not None:
                # 自适应可视化：判定每条曲线的数据类型 / 推荐图型 / 统计量
                try:
                    from .autoviz import annotate_curves

                    annotate_curves(res)
                except Exception:                            # noqa: BLE001
                    pass
            if res is None:
                # 单帧体系里"全是时间序列"的分析项：整项记为跳过，不留空图
                self.skips[name] = (self.skip_reason
                                    or "只有 1 帧：这一项的图全是随时间变化，已省略")
                if verbose:
                    print(f"  -- 跳过：{self.skips[name]}")
                continue
            self.results[name] = res
            self.timings[name] = time.time() - t0
            if verbose:
                print(f"  -> {len(res.curves)} 条曲线, {len(res.summary)} 项统计, "
                      f"{time.time() - t0:.1f}s")
            # 关键：算完一项就立刻回调，Web 版据此实时刷新页面
            if on_result:
                on_result(name, res, self.timings[name], len(self.results), total)
        _cancel_clear()
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
            # 「分析集合（Set）」不参与"水→只看 O"的特判：集合已经是用户挑好的原子
            if category == "water" and not name.startswith("set:"):
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
            # ⚠️ 这里**不再**按原子数抽稀（1.0.2 改）。
            # 老代码对 >5000 原子的组分做 `ag[::step]`（抽到约 2000 个**原子**），
            # 两个问题：
            #   1. 用户要"每个原子都参与"，抽稀直接违背需求；
            #   2. 原子级等间隔取会把**分子切碎** —— PEG 体系 200 条 30 原子链
            #      抽成每条 10 个原子，"分子质心"变成残缺分子的质心，D 失真。
            # MSD 现在自己会分块 + 落盘内存映射，因此不需要在这里丢粒子；
            # 真要限制规模请用 MSD 参数 `max_particles`（分子对象按分子整取）。
            if ag.n_atoms:
                out[name] = ag
        for name, ag in self.extra_components.items():
            if ag.n_atoms:
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

