# -*- coding: utf-8 -*-
"""Web 会话管理：一个"已打开的体系"对应一个 :class:`Session`。

会话持有 :class:`mdta.pipeline.Analyzer`（也就是已打开的 MDAnalysis Universe），
分析结果与导出目录都挂在会话上；前端用 ``sid`` 引用它。
"""

from __future__ import annotations

import json
import math
import re
import os
import shutil
import sys
import threading
import time
import uuid

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from mdta import __version__
import time

from mdta.io import TrajectoryError, load_trajectory, set_progress_sink
from mdta.io import _progress


class OpenProgress:
    """「文件读取」这一步的进度记录：阶段、逐阶段耗时、总耗时、是否完成/失败。

    为什么需要：打开大轨迹首次要扫全文件建立帧索引（46 体系 1.7 GB 实测 >135 s），
    期间界面完全没有反馈，用户会以为程序卡死。这里把加载器的阶段回报攒下来，
    前端在 POST /api/session **进行中**并发轮询它（该接口是同步 def，FastAPI 走
    线程池，所以请求未返回时仍能响应其它请求）。
    """

    def __init__(self) -> None:
        self.t0 = time.time()
        self.stages: list[dict] = []
        self.done = False
        self.error = ""
        # 完成后的会话 id / 本次打开的文件：前端 F5 后靠它把会话接回来
        self.sid: str = ""
        self.topology: str = ""
        self.trajectory: str = ""
        self._lock = threading.Lock()

    def __call__(self, stage: str, **info) -> None:
        with self._lock:
            now = time.time()
            prev = self.stages[-1]["t"] if self.stages else self.t0
            self.stages.append({"stage": str(stage), "t": now,
                                "elapsed": round(now - self.t0, 2),
                                "dt": round(now - prev, 2),
                                **{k: v for k, v in info.items() if v is not None}})

    def snapshot(self) -> dict:
        with self._lock:
            stages = list(self.stages)
        return {
            "elapsed": round(time.time() - self.t0, 2),
            "stages": stages,
            "stage": stages[-1]["stage"] if stages else "准备中",
            "n_stages": len(stages),
            "done": bool(self.done),
            "error": self.error,
            # 刷新恢复用：会话建好后前端凭 `sid` 直接接回（见 App.restoreSession）
            "sid": self.sid,
            "topology": self.topology,
            "trajectory": self.trajectory,
        }
from mdta.pipeline import (ANALYSIS_TITLES, DEFAULT_ORDER, Analyzer,
                           sort_by_estimate)
from mdta.selection import chains as _chains
from mdta.selection import list_chains, select
from mdta.systeminfo import info_tables

from . import serialize

#: 支持的扩展名（用于文件选择器过滤）—— **唯一来源是 mdta.formats**，
#: 这里只做排序后的元组，供前端展示；不要再往这里加格式（会与其它入口脱节）。
from mdta.formats import (                                   # noqa: E402
    ALL_SUFFIXES as _ALL_SUFFIXES,
    BOTH_SUFFIXES as _BOTH_SUFFIXES,
    TOPOLOGY_SUFFIXES as _TOP_SUFFIXES,
    TRAJECTORY_SUFFIXES as _TRAJ_SUFFIXES,
    can_be_topology as _can_be_topology,
    can_be_trajectory as _can_be_trajectory,
    classify as _classify_suffix,
    sniff_lammps_kind as _sniff_lammps_kind,
    suffix_of as _suffix_of,
    topology_help as _topology_help,
    trajectory_help as _trajectory_help,
    unsupported_hint as _unsupported_hint,
)

TOPOLOGY_EXT = tuple(sorted(_TOP_SUFFIXES))
TRAJECTORY_EXT = tuple(sorted(_TRAJ_SUFFIXES))
MD_EXT = tuple(sorted(_ALL_SUFFIXES))


def file_kind(name: str) -> str:
    """文件在界面里的**主用途**：``"topology"`` 或 ``"trajectory"``（只有这两种值）。

    ⚠️ 这里**故意不返回第三种值** `"both"`。教训（真实故障）：第一版让 `.pdb` 返回
    `kind="both"`，而用户页面上还是**上一版前端**，它只判断 `kind === 'topology'`，
    其它一律塞进轨迹槽 —— 现象就是"pdb 明明是拓扑，怎么读到轨迹槽里去了"。
    只要接口新增枚举值、而客户端可能滞后一版，就会出现这种"新后端 + 老前端"事故。

    所以：`kind` 保持两种老值（`.pdb`/`.gro` 这类主用途就是**拓扑**），
    "还能当轨迹"的信息走**新增字段** `roles`（老前端不认识就会直接忽略）。
    """
    return "trajectory" if _classify_suffix(name) == "trajectory" else "topology"


def file_roles(name: str) -> list[str]:
    """该文件能用在哪几个槽：``["topology"]`` / ``["trajectory"]`` / 两者都有。"""
    kind = _classify_suffix(name)
    if kind == "both":
        return ["topology", "trajectory"]
    if kind == "trajectory":
        return ["trajectory"]
    return ["topology"]


def file_note(path: str) -> str | None:
    """文件的**内容警示**（新字段，供文件列表提前提示，选中后才报错太晚）。

    LAMMPS 项目里同一批后缀混着三种东西：`write_data` 的 data、`in.lammps`
    输入脚本、`log.lammps` 日志。后缀分不出来，但内容能 —— 所以这里嗅一下，
    让文件列表直接标出"这不是数据文件"（用户实测就是这么被绊住的）。
    """
    try:
        kind = _sniff_lammps_kind(path)
    except Exception:                                        # noqa: BLE001
        kind = None
    if kind == "input":
        return "LAMMPS 输入脚本，不是数据文件（data 看 read_data 指到的那份）"
    if kind == "log":
        return "LAMMPS 日志/热力学输出，不含坐标"
    if kind == "data":
        return None                     # 正常 data 文件，别乱标
    try:
        hint = _unsupported_hint(path)
        if hint:
            return hint
        # 后缀像 LAMMPS data、内容却不是 data（本工具自己的 `_msd_scratch/*.dat`
        # 二进制缓存、或用户的一列数值结果表）—— 提前标出来，别等选中才报错
        from mdta.formats import LAMMPS_TEXT_SUFFIXES

        if _suffix_of(path) in LAMMPS_TEXT_SUFFIXES:
            return "后缀像 LAMMPS data，但内容不是 data 文件（没有 atoms / atom types 表头）"
    except Exception:                                        # noqa: BLE001
        return None
    return None


def _available_elements(universe) -> list[str]:
    """体系里出现过的元素（给"分析集合"的多选项）。"""
    try:
        from mdta.sets import available_elements

        return available_elements(universe)
    except Exception:                                    # noqa: BLE001
        return []


def _set_presets(universe, components) -> list[dict]:
    """一键预设（碳原子 / 链状分子 / 含环 / 大分子 / 重原子…），只列能选中的。"""
    try:
        from mdta.sets import suggest_presets

        return suggest_presets(universe, components)
    except Exception:                                    # noqa: BLE001
        return []


def supported_formats() -> dict:
    """给界面的格式说明（唯一来源 `mdta.formats`，前端 tooltip 与错误信息共用）。"""
    return {"topology": _topology_help(), "trajectory": _trajectory_help(),
            "topology_count": len(_TOP_SUFFIXES),
            "trajectory_count": len(_TRAJ_SUFFIXES),
            "both": sorted(_BOTH_SUFFIXES)}



class SessionError(RuntimeError):
    """会话相关的错误（会被 API 层转成 400/404）。"""


@dataclass
class RunJob:
    """一次"实时运行"的状态。

    后台线程逐项算分析，每算完一项就更新这里；前端轮询
    ``GET /run/progress?after=N`` 只取自己还没有的结果，从而边算边显示。
    """

    #: extend 模式下已有结果的条数（completion 列表里排在最前面）
    preexisting: int = 0
    order: list[str] = field(default_factory=list)
    #: 已完成的**分析名，按完成顺序**——``after`` 索引就是它的下标
    completion: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "running"            # running | done | error | cancelled
    #: 子阶段：``estimate``（先跑几帧预估用时）→ ``run``（正式跑）
    phase: str = ""
    frac: float = 0.0
    message: str = "准备中…"
    current: str = ""
    #: ``{分析名: 预估秒数}``（只含估出来的项）
    estimates: dict[str, float] = field(default_factory=dict)
    estimate_note: str = ""
    error: str = ""
    last_result: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: Any = field(default=None, repr=False, compare=False)


@dataclass
class Session:
    """一个已打开的体系 + 它的分析结果。"""

    sid: str
    az: Analyzer
    outdir: str
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    results_json: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    #: 用 RLock 而不是 Lock：``run_progress`` / ``_payload`` 等辅助方法本身会取锁，
    #: 而它们常常在已持锁的代码段里被调用；普通 Lock 不可重入会直接死锁。
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _job: "RunJob | None" = field(default=None, repr=False)
    #: 最近一次导出的信息（目录 / 时间 / 计数），前端只显示这一条
    last_export: "dict | None" = field(default=None, repr=False)
    #: 用户定义的「分析集合（Set）」定义与生成结果（见 mdta.sets）
    sets_spec: list = field(default_factory=list, repr=False)
    sets_info: list = field(default_factory=list, repr=False)
    #: 最近一次「主链」（分析对象）请求，前端 F5 后据此把下拉框接回来
    primary_req: dict = field(default_factory=dict, repr=False)
    #: ``{分析名: 输入指纹}``：这次结果是用哪套"选择 + 帧 + 参数"算出来的。
    #: 扩展分析（extend）时用它判断哪些结果还能留、哪些必须重算 —— 用户实测：
    #: 第一轮只勾了几个组分跑 MSD，第二轮加了组分再跑，后端把 MSD 当成"已有结果"
    #: 直接跳过，于是"加了组分却不跑"（结果还是按旧组分算的）。
    sigs: dict = field(default_factory=dict, repr=False)
    #: 三维视图里保存下来的选择：``{名字: {name, indices, n_atoms, kind, label}}``
    selections: dict = field(default_factory=dict, repr=False)
    #: 分子/聚合物链识别结果（:class:`mdta.recognize.Recognition`）+ 缓存标识。
    #: 识别只跑一次：结构树反复展开/点选都读缓存，切换体系时整个 Session 重建 → 自然失效。
    _recog: object = field(default=None, repr=False)
    _recog_key: tuple = field(default=(), repr=False)
    #: 分析正在跑时新建/改动的集合（`set:<名字>` → AtomGroup），只建不算。
    #: 它们会出现在组分候选清单里，等这一轮结束后的下一次 run 真正注册进分析器。
    _pending_sets: dict = field(default_factory=dict, repr=False)
    #: 数据能力检测结果缓存（见 :meth:`capabilities`）
    _caps: dict | None = field(default=None, repr=False)
    _caps_key: tuple = field(default=(), repr=False)

    # ------------------------------------------------------------- 基本信息
    @property
    def lock(self) -> "threading.RLock":
        return self._lock

    def touch(self) -> None:
        self.last_used = time.time()

    def info(self) -> dict:
        mdt = self.az.trajectory
        info = self.az.info
        return {
            "topology": mdt.topology,
            "trajectory": mdt.trajectory,
            "n_atoms": mdt.n_atoms,
            "n_frames": mdt.n_frames,
            "dt_ps": mdt.dt_ps,
            "first_time_ps": info.first_time_ps,
            "last_time_ps": info.last_time_ps,
            "total_time_ps": mdt.total_time_ps,
            "n_residues": info.n_residues,
            "n_segments": info.n_segments,
            "box_dimensions": serialize._num(info.box_dimensions),
            "box_angles": serialize._num(info.box_angles),
            "box_type": info.box_type,
            "box_volume": serialize._num(info.box_volume),
            "total_mass_amu": serialize._num(info.total_mass_amu),
            "total_charge_e": serialize._num(info.total_charge_e),
            "n_bonds": info.n_bonds,
            "n_angles": info.n_angles,
            "n_dihedrals": info.n_dihedrals,
            "categories": dict(info.categories),
            # 结构化表格：Web 界面直接用表格渲染，不再显示定宽文本报告
            "sections": self._info_sections(info),
            "text": self.az.info_text(max_chain_types=25),
        }

    def _info_sections(self, info) -> list[dict]:
        """体系信息表格 + 追加「组分与残基构成」（按残基细分，不止 protein/water/ion）。"""
        from mdta.systeminfo import info_tables

        sections = info_tables(info, max_chain_types=25)
        try:
            rows = []
            for c in self.catalog():
                mix = c.get("residue_mix") or []
                brief = "、".join(f"{m['resname']}×{m['n']}" for m in mix[:5])
                if c.get("n_residue_kinds", 0) > len(mix[:5]):
                    brief += f" …（共 {c['n_residue_kinds']} 种）"
                rows.append([str(c["name"]), f"{c['n_atoms']:,}",
                             f"{c.get('n_residue_kinds', 0)}", brief or "—"])
            if rows:
                sections.append({"title": "组分与残基构成",
                                 "columns": ["组分", "原子数", "残基种类", "主要残基"],
                                 "rows": rows})
        except Exception:                                    # noqa: BLE001
            pass
        return sections

    def load_notes(self) -> list[str]:
        """加载阶段的诊断说明（猜键 / 分子划分 / 偏移缓存重建）。

        为什么要给前端：下面这些情况**不会报错**，但会实质影响结果 ——
        `CONECT` 不完整的 PDB 导致分子被切碎、退化 molnums 导致"整个体系算 1 个分子"、
        偏移缓存过期被重建。把话说出来，用户才知道要不要换拓扑或勾选重新猜键。
        """
        mdt = self.az.trajectory
        notes: list[str] = []
        bi = mdt.bond_guess_info or {}
        if bi.get("note"):
            notes.append(str(bi["note"]))
        if bi.get("bonds"):
            notes.append(f"键表：{bi.get('bonds'):,} 个键 / "
                         f"{bi.get('fragments', 0):,} 个碎片"
                         + (f"（{bi.get('note')}）" if bi.get("note") else ""))
        if bi.get("skipped"):
            notes.append(f"未猜键：{bi['skipped']}")
        oi = mdt.object_info or {}
        src = str((oi.get("sources") or {}).get("molnums") or "")
        if src:
            notes.append("分子划分：" + src)
        if oi.get("molnums_degenerate"):
            notes.append(f"⚠️ 拓扑的残基/分子信息退化 {oi['molnums_degenerate']}；"
                         f"分析已改用键连通分量划分分子")
        if getattr(mdt, "offsets_rebuilt", False):
            notes.append("偏移缓存与轨迹不符，已自动重建（轨迹可能在缓存建立后被"
                         "替换/重新转换过）")
        # Materials Studio 文档（.car/.xsd/.xtd）：拓扑是自研解析的，把摘要说出来
        # （帧数校验的说明由 mdta.io 写进 load_notes，下面统一 append）
        mi = getattr(mdt, "ms_info", None) or {}
        if mi:
            nb = int(mi.get("n_bonds") or 0)
            notes.append(
                f"Materials Studio 文档（.{mi.get('format')}）："
                f"{mi.get('n_atoms', 0):,} 原子 / "
                + (f"{nb:,} 键" if nb else "文件里没有键表（已按坐标猜键）")
                + (f"；剔除周期镜像 {mi['n_images_skipped']:,} 个"
                   if mi.get("n_images_skipped") else ""))
        notes.extend(str(x) for x in (getattr(mdt, "load_notes", None) or []))
        return notes

    def chains(self) -> list[dict]:
        out = []
        for c in sorted(list_chains(self.az.universe), key=lambda c: -c.n_atoms):
            out.append({
                "label": c.label,
                "segid": c.segid,
                "resname": c.resname,
                "count": int(c.count),
                "n_atoms": int(c.n_atoms),
                "n_residues": int(c.n_residues),
                "description": str(c),
            })
        return out

    def _residue_mix(self, ag, top: int = 6) -> tuple[list, int]:
        """该组分的**残基构成**：主要残基名 + 各自残基数 + 原子数；返回 (mix, 种类数)。

        为什么需要：界面上只写「protein / water / ion」太笼统（用户实测）。
        有了残基构成，用户能一眼看出"这个 protein 里到底有哪些残基"。
        """
        try:
            r = ag.residues
            names = [str(x) for x in r.resnames]
            if not names:
                return [], 0
            from collections import Counter

            kinds = Counter(names)
            atom_of = Counter()
            for res in r:
                atom_of[str(res.resname)] += int(res.atoms.n_atoms)
            mix = [{"resname": k, "n": int(v), "atoms": int(atom_of.get(k, 0))}
                   for k, v in kinds.most_common(top)]
            return mix, len(kinds)
        except Exception:                                    # noqa: BLE001
            return [], 0

    def _component_item(self, name, ag, *, active: bool) -> dict:
        """一个组分在界面上的那条记录（原子数 / 分子数 / 残基构成 / 追踪方式）。"""
        from mdta.analysis.interface import _atom_molnums

        n_mol = 1
        try:
            m = _atom_molnums(ag)
            if m is not None:
                n_mol = int(np.unique(m).size) if m.size else 1
        except Exception:  # noqa: BLE001
            n_mol = 1
        mix, n_kinds = self._residue_mix(ag)
        item = {
            "name": name,
            "n_atoms": int(ag.n_atoms),
            "n_residues": int(ag.residues.n_residues),
            "n_molecules": n_mol,
            # 这一条现在是否参与分析（取消勾选后仍在候选清单里，active=False）
            "active": bool(active),
            # 残基构成（protein/water/ion 太笼统，界面按这个进一步区分）
            "residue_mix": mix,
            "n_residue_kinds": int(n_kinds),
        }
        if str(name).startswith("set:"):
            item["kind"] = "set"
            item["trace"] = self.az.msd_group_objects.get(name)
        return item

    def components(self) -> list[dict]:
        """**当前参与分析**的组分清单。``n_molecules`` 是给界面用的：**逐分子统计只在
        多分子组分上有意义**（单分子组分里"整组 Rg"和"该分子 Rg"是同一个数），
        界面据此把开关置灰。
        代价极小（``molnums`` 直接读属性，AdK water 44336 原子实测 0.000 s）。
        """
        out = []
        for name, ag in list(self.az.components.items()) + list(
                self.az.extra_components.items()):
            out.append(self._component_item(name, ag, active=True))
        # 分析跑着的时候新建的集合：只建了没注册，但界面上要能勾（下一轮才生效）
        seen = {str(x.get("name")) for x in out}
        for name, ag in (self._pending_sets or {}).items():
            if str(name) in seen:
                continue
            item = self._component_item(name, ag, active=True)
            item["pending"] = True
            out.append(item)
        return out

    def catalog(self) -> list[dict]:
        """**候选成分清单**：自动识别出的全部组分 + 已定义的集合（不管勾没勾）。

        为什么必须和 :meth:`components` 分开：界面上的组分胶囊与「主链」下拉框读的是
        这一份。以前两者共用 `components`，于是"把自动识别的组分全部取消勾选"之后，
        列表里就一条都不剩了 —— 用户实测：第一轮只跑 MSD（没勾任何自动组分），
        第二轮想补算别的分析时发现无组分可选，刷新后也一样。
        """
        active = {str(k) for k in self.az.components}
        out = []
        seen: set[str] = set()
        for name, ag in list(self.az.catalog.items()) + list(
                self.az.extra_components.items()):
            key = str(name)
            if key in seen:
                continue
            seen.add(key)
            out.append(self._component_item(name, ag, active=key in active))
        # 兜底：勾选了但不在候选清单里的（用户自定义选择语句）也要能勾掉
        for name, ag in self.az.components.items():
            key = str(name)
            if key in seen:
                continue
            seen.add(key)
            out.append(self._component_item(name, ag, active=True))
        # 分析跑着时新建的集合（尚未注册进分析器，下一轮 run 才生效）
        pending = self._pending_sets or {}
        if pending:
            for name, ag in pending.items():
                key = str(name)
                if key in seen:
                    continue
                seen.add(key)
                item = self._component_item(name, ag, active=True)
                item["pending"] = True
                out.append(item)
        return out

    def primary_molecules(self) -> int:
        """当前主链对象含多少个分子（1 = 逐分子统计无意义）。"""
        from mdta.analysis.interface import _atom_molnums

        ag = self.az.primary
        if ag is None or ag.n_atoms == 0:
            return 0
        try:
            m = _atom_molnums(ag)
            return int(np.unique(m).size) if m is not None and m.size else 1
        except Exception:  # noqa: BLE001
            return 1

    def primary_atoms_info(self) -> dict:
        """主链（分析对象 = Rg/R_ee/二面角/取向/有序度/结晶用的那组）的原子清单。

        给界面的「R_ee 指定原子」做悬浮提示用：用户填「第几个原子」时，能立刻看到
        对应的**原子名与全局编号**。返回顺序就是主链内部顺序（1 起编号 = 下标+1）。
        """
        ag = self.az.primary
        if ag is None or ag.n_atoms == 0:
            return {"n_atoms": 0, "names": [], "elements": [], "indices": [],
                    "label": str(getattr(self, "primary_label", "") or "")}
        try:
            names = [str(x) for x in ag.names]
        except Exception:  # noqa: BLE001
            names = [f"#{int(i)}" for i in ag.indices]
        try:
            elements = [str(x) for x in ag.elements]
        except Exception:  # noqa: BLE001
            elements = ["" for _ in names]
        return {"n_atoms": int(ag.n_atoms), "names": names, "elements": elements,
                "indices": [int(i) for i in ag.indices],
                "label": str(getattr(self, "primary_label", "") or "")}

    # ----------------------------------------------------------------- 选择
    def apply_selection(self, req: Mapping[str, Any] | None) -> dict:
        """按前端请求设置主链与组分（含「分析集合（Set）」）。

        顺序很关键：**先组分 → 再集合 → 最后主链**。集合是作为 `set:<名字>` 写进
        `components` 的一等公民，所以主链解析必须排在集合之后 —— 否则界面上把主链
        选成 `set:C` 会报「不存在组分 'set:C'」（用户实测：集合对回转半径不起作用，
        一半原因在这里；另一半是主链还停在自动识别的最大链上）。
        """
        # 分析进行中：**选择只存索引/定义，不碰分析器**（用户实测要求 ——
        # "选取"和"分析"必须互不影响：分析跑着的时候照样能筛条件、建集合）。
        if self.is_running:
            return self._apply_selection_deferred(req)
        self._pending_sets = {}
        req = dict(req or {})
        az = self.az
        az.auto_setup()

        wanted = req.get("components")
        if wanted is not None:
            # ⚠️ 必须严格按用户勾选来（哪怕一个都没勾）：以前写成 `if wanted: ... if chosen:`，
            # 结果"只勾了分析集合"时 `chosen` 为空 → 根本没有覆盖 auto_setup 自动识别出的
            # protein / water / ion，用户看到的是"我没选它，却在分析它"。
            chosen = OrderedDict()
            for n in wanted:
                if str(n).startswith("set:"):
                    # 「分析集合（Set）」是下面单独处理的"一等公民"，不能当选择语句解析
                    continue
                if n in az.components:
                    chosen[n] = az.components[n]
                else:
                    ag = select(az.universe, query=str(n))
                    if ag.n_atoms == 0:
                        raise SessionError(f"组分选择语句没有选中任何原子: {n!r}")
                    chosen[str(n)] = ag
            az.set_components(chosen)      # 允许为空 = 本次只用集合

        # ---- 分析集合（Set）：可选的"只看某类原子/分子"
        if "sets" in req and req.get("sets") is not None:
            self.sets_spec = [dict(x) for x in (req.get("sets") or [])
                              if isinstance(x, Mapping)]
        # 「可视化选择」存下来的集合：前端只回传名字（上万个索引不必来回传），
        # 索引由后端从 `self.selections` 补回 —— 否则重建时会退化成"没有筛选条件"
        # 的集合（那会选中**整个体系**，比丢还好一点的是错得更离谱）。
        for spec in self.sets_spec:
            if "indices" not in spec:
                rec = (self.selections or {}).get(str(spec.get("name") or ""))
                if rec and rec.get("indices"):
                    spec["indices"] = list(rec["indices"])
        if self.sets_spec:
            from mdta.sets import SetError as _SetError

            try:
                self.sets_info = az.add_sets(self.sets_spec, components=az.components)
            except _SetError as exc:
                raise SessionError(f"分析集合定义有误：{exc}") from exc
            # 参与与否**严格看 `components` 里的 `set:` 名字**：
            # 界面勾上哪个集合就只保留哪个；一个都没勾 → 本次不参与分析。
            if wanted is not None:
                chosen_sets = {str(n) for n in wanted if str(n).startswith("set:")}
                for key in list(az.components):
                    if str(key).startswith("set:") and key not in chosen_sets:
                        az.components.pop(key, None)
                        az.msd_group_objects.pop(key, None)
        else:
            self.sets_info = []

        # ---- 主链 / 构象分析对象（Rg / R_ee / 二面角 / 取向 / 结晶都用它）
        # 放在集合之后：`set:` 名字这时才在 components 里，能直接当主链。
        primary = req.get("primary") or {"mode": "auto"}
        self.primary_req = dict(primary)
        mode = primary.get("mode", "auto")
        checked_sets = [str(n) for n in (wanted or [])
                        if str(n).startswith("set:") and str(n) in az.components]
        if mode == "custom" and primary.get("query"):
            q = str(primary["query"])
            ag = select(az.universe, query=q)
            if ag.n_atoms == 0:
                raise SessionError(f"主链选择语句没有选中任何原子: {q!r}")
            az.set_primary(ag, f"自定义: {q}")
        elif mode == "component" and primary.get("name"):
            name = str(primary["name"])
            if name not in az.components:
                catalog = getattr(az, "catalog", {}) or {}
                if name in catalog:
                    # 主链选的是**候选清单里有、但这次没勾选**的组分（界面上
                    # 「分析对象」下拉框列的是候选清单，勾选框列的是本次参与项，
                    # 两者本来就可以不一致）。用户既然把它选成分析对象，
                    # 就必须让它在这一次分析里参与 —— 以前这里直接报
                    # "不存在组分 'water'"（用户实测：选了 water 跑 Rg 就报错）。
                    az.components[name] = catalog[name]
                elif name.startswith("set:") and name in getattr(az, "set_groups", {}):
                    # 同上，只是集合的原子组单独存在 set_groups 里
                    az.components[name] = az.set_groups[name]
                elif name.startswith("set:"):
                    raise SessionError(
                        f"集合 {name} 不存在或没有定义（先在上面添加集合）")
                else:
                    raise SessionError(
                        f"不存在组分 {name!r}；当前体系可选的组分："
                        f"{list(catalog) or list(az.components)}")
            az.set_primary(az.components[name], f"组分 {name}")
        elif mode == "chain" and primary.get("segid"):
            got = _chains(az.universe, segid=str(primary["segid"]),
                          resname=primary.get("resname"),
                          min_atoms=int(primary.get("min_atoms", 1)))
            if not got:
                raise SessionError("按 segid/resname 没有找到任何链")
            tag = f"{primary.get('segid')}:{primary.get('resname')}"
            if primary.get("all") and len(got) > 1:
                # 界面上的「链」是按 segid+resname 归并出来的**链类型**（如 PEG 的
                # "200 条，每条 30 原子"）。标了"全部 N 条"就必须真的把 N 条并起来：
                # 早先无论 count 是多少都只取 got[0]，于是一个写着"200 条"的选项
                # 实际只分析了 30 个原子（1 条链）—— 实测 Rg 因此给出 7.57 Å
                # 而不是 200 条整组的 26.98 Å，耗时也差 25 倍。
                idx = np.sort(np.concatenate(
                    [np.asarray(g.indices, dtype=np.int64) for g in got]))
                ag = az.universe.atoms[idx]
                az.set_primary(ag, f"{tag} 全部 {len(got)} 条链 ({ag.n_atoms} 原子)")
            else:
                az.set_primary(got[0], tag + (f" 第 1 条（共 {len(got)} 条）"
                                              if len(got) > 1 else ""))
        elif mode == "auto" and checked_sets:
            # 「自动」+ 勾了集合 → 主链就是勾选的集合（多个取并集）。
            # 为什么：Rg / R_ee / 二面角 / 取向 / 结晶的**分析对象就是主链**，
            # 集合若只进 components，这些分析根本看不见它（用户实测：设了 set:C，
            # 回转半径还是按"最大链"算的 —— 那显然不是他要的碳骨架）。
            parts = [az.components[n] for n in checked_sets]
            if len(parts) == 1:
                ag = parts[0]
            else:
                idx = np.unique(np.concatenate(
                    [np.asarray(g.indices, dtype=np.int64) for g in parts]))
                ag = az.universe.atoms[idx]
            label = ("集合 " + " + ".join(checked_sets) if len(checked_sets) > 1
                     else f"集合 {checked_sets[0]}")
            az.set_primary(ag, f"{label}（{ag.n_atoms:,} 原子）")

        return {
            "primary_label": az.primary_label,
            "primary": dict(primary),
            "primary_atoms": int(az.primary.n_atoms) if az.primary is not None else 0,
            "primary_molecules": self.primary_molecules(),
            # 当前参与分析的组分（只有勾选的）
            "components": self.components(),
            # 勾选名字清单：界面据此恢复"哪些胶囊是亮的"
            "active": [str(n) for n in az.components],
            # 候选清单（全部自动组分 + 已定义集合，带 active 标记）
            "catalog": self.catalog(),
            "sets": list(self.sets_info),
            # 可选元素与预设（界面做多选项与一键按钮）
            "elements": _available_elements(az.universe),
            "set_presets": _set_presets(az.universe, az.components),
            # 分析项能不能跑（界面据此置灰 + tooltip 说明）
            "feasible": self.feasible(),
            "capabilities": self.capabilities(),
        }

    def capabilities(self, *, refresh: bool = False) -> dict:
        """**数据能力检测**：坐标 / 多帧轨迹 / 键连接 / 质量 / 电荷 /
        分子归属 / 周期盒，外加 ``mode``（static / dynamic）。

        只读，算一次就缓存（换体系即换 Session，缓存自然失效）。
        """
        from mdta.capabilities import detect

        key = (id(self.az.universe), int(self.az.universe.atoms.n_atoms))
        if refresh or self._caps is None or self._caps_key != key:
            self._caps = detect(self.az)
            self._caps_key = key
        return self._caps

    def feasible(self, frames: Mapping[str, Any] | None = None) -> dict:
        """分析项**可行性预判**（点「开始分析」之前就把跑不了的项说清楚）。

        只看晶胞 / 帧数 / 键表 / 当前勾选的组分与分析对象，不触发任何分析计算，
        所以开销可以忽略；`/select` 的返回里也带一份，界面据此把项置灰。

        ``frames``：界面上的**帧选择**（起始/结束/间隔/平衡段/最大帧数）。
        一定要传 —— "体系有 100 帧"不等于"这次要算 100 帧"：把最大帧数设成 1
        或者把时间窗收窄，MSD 就只剩下 1 帧，跑起来直接报「MSD 至少需要 2 帧」。
        """
        from mdta.feasible import preflight

        n_frames = None
        total_frames = None
        try:
            total_frames = int(self.az.trajectory.n_frames)
        except Exception:                                        # noqa: BLE001
            total_frames = None
        if frames:
            def _f(key):
                v = dict(frames).get(key)
                if v in ("", None):
                    return None
                try:
                    return float(v)
                except Exception:                                # noqa: BLE001
                    return None

            mf = dict(frames).get("max_frames")
            try:
                self.az.set_frames(
                    start_ps=_f("start_ps"), stop_ps=_f("stop_ps"),
                    interval_ps=_f("interval_ps"), equil_ps=_f("equil_ps"),
                    max_frames=int(mf) if mf not in ("", None) else None)
                n_frames = int(self.az.require_frames().n_frames)
            except Exception:                                    # noqa: BLE001
                n_frames = 0
        try:
            return preflight(self.az, n_frames=n_frames,
                             total_frames=total_frames)
        except Exception as exc:                                 # noqa: BLE001
            return {"_error": {"ok": True, "level": "warn",
                               "reason": f"可行性预判失败：{exc}"}}

    # ------------------------------------------------- 三维结构与交互式选择
    def _sets_only(self, specs) -> tuple[list[dict], dict]:
        """**只构建**分析集合（不碰分析器状态），返回 ``(infos, {set:<名>: AtomGroup})``。

        用途：分析正在跑的时候，界面上照样能增删集合、看预览 —— 这一步只读 Universe
        的拓扑/首帧坐标，写进会话自己的 ``sets_spec``/``sets_info``，等下一次 run
        才真正注册进 ``Analyzer``。这样"选取"与"分析"互不干扰（用户实测要求）。
        """
        from mdta.sets import SetError as _SetError
        from mdta.sets import build_set

        base = dict(getattr(self.az, "components", {}) or {})
        infos: list[dict] = []
        groups: dict = {}
        for spec in specs or []:
            spec = dict(spec or {})
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            try:
                ag, info = build_set(self.az.universe, spec, base)
            except _SetError as exc:
                raise SessionError(f"分析集合定义有误：{exc}") from exc
            key = f"set:{name}"
            groups[key] = ag
            out = {**info, "name": key}
            idx = out.pop("indices", None)
            if idx is not None:
                out["n_indices"] = len(idx)
            infos.append(out)
        return infos, groups

    def _apply_selection_deferred(self, req: Mapping[str, Any]) -> dict:
        """分析正在跑时的 `apply_selection`：**只存选择，不动分析器**。

        存的东西：集合定义（含可视化选的索引）+ 集合预览 + 主链/勾选意图。
        返回结构与空闲时同形（界面不用分支），多一个 ``deferred: True``。
        """
        az = self.az
        if req.get("sets") is not None:
            self.sets_spec = [dict(x) for x in (req.get("sets") or [])
                              if isinstance(x, Mapping)]
        # 可视化选择只回传名字，索引从会话里补回来（与空闲分支同一套规则）
        for spec in self.sets_spec:
            if "indices" not in spec:
                rec = (self.selections or {}).get(str(spec.get("name") or ""))
                if rec and rec.get("indices"):
                    spec["indices"] = list(rec["indices"])
        infos, groups = self._sets_only(self.sets_spec)
        self.sets_info = infos
        self._pending_sets = groups
        if req.get("primary") is not None:
            self.primary_req = dict(req.get("primary") or {"mode": "auto"})
        return {
            "deferred": True,
            "primary_label": az.primary_label,
            "primary": dict(self.primary_req or {"mode": "auto"}),
            "primary_atoms": (int(az.primary.n_atoms)
                              if az.primary is not None else 0),
            "primary_molecules": self.primary_molecules(),
            "components": self.components(),
            "active": [str(n) for n in az.components],
            "catalog": self.catalog(),
            "sets": list(self.sets_info),
            "elements": _available_elements(az.universe),
            "set_presets": _set_presets(az.universe, az.components),
            "feasible": self.feasible(),
            "capabilities": self.capabilities(),
            "note": "分析正在运行：选择已保存，将在下一次运行时生效",
        }

    def _chain_chem(self, segid: str, resname: str) -> dict:
        """一条链类型（segid+resname）的化学身份：分子式 / 名称 / 类别 / 聚合度。

        做法：取该类型**第一个残基**所属的**完整分子**（键连通分量），统计元素组成
        与独立环数，交给 :func:`mdta.chem.summarize`。拿不到就返回空 dict。
        """
        try:
            from mdta.chem import summarize
            from mdta.sets import molecule_profile, _elements_of

            u = self.az.universe
            res = u.residues
            mask = np.ones(int(res.n_residues), dtype=bool)
            try:
                mask &= np.asarray(res.segids, dtype=str) == str(segid)
            except Exception:                                # noqa: BLE001
                pass
            mask &= np.asarray(res.resnames, dtype=str) == str(resname)
            hits = np.nonzero(mask)[0]
            if not hits.size:
                return {}
            ridx = int(hits[0])
            atom_res = np.asarray(u.atoms.resindices, dtype=int)
            seed = np.nonzero(atom_res == ridx)[0]
            if not seed.size:
                return {}
            prof = molecule_profile(u)
            mol = prof["mol"]
            mid = int(mol[seed[0]])
            sel = np.nonzero(mol == mid)[0]
            elem = _elements_of(u)[sel]
            counts: dict[str, int] = {}
            for e in elem:
                k = str(e).strip().upper() or "X"
                counts[k] = counts.get(k, 0) + 1
            n_rings = 1 if bool(prof["has_ring"][mid]) else 0
            info = summarize(counts, resname=resname, n_rings=n_rings,
                             is_chain=bool(prof["is_chain"][mid]),
                             n_atoms=int(sel.size))
            info["n_atoms_in_molecule"] = int(sel.size)
            info["n_molecules"] = int(len(hits))
            return info
        except Exception:                                    # noqa: BLE001
            return {}

    def search(self, q: str, limit: int = 20) -> dict:
        """按**残基名 + 编号**（如 ``ETH32``）、原子名或原子索引检索结构。

        用户要求（1.0.4）：搜索要具体到某个残基/原子（例如 ``ETH32`` ``ETH34``），
        命中后在三维视图里高亮。匹配顺序：

        1. ``残基名+残基号`` 精确匹配（``ETH32`` / ``eth 32`` / ``ETH-32``）→ 残基；
        2. 纯数字 → 当作**原子索引**（``123`` / ``#123``）；
        3. 原子名精确匹配（``C1`` ``H12``）→ 若干原子；
        4. 以上都不中 → 残基名**前缀**匹配（``ETH`` → 该类型的前若干个残基）。
        """
        u = self.az.universe
        n = int(u.atoms.n_atoms)
        raw = str(q or "").strip()
        out: list[dict] = []
        if not raw:
            return {"query": "", "matches": [], "total": 0}

        resnames = np.array([str(x) for x in u.residues.resnames], dtype=object)
        resids = np.asarray(u.residues.resids, dtype=np.int64)
        segids = np.array([str(x) for x in u.residues.segids], dtype=object)
        atom_res = np.asarray(u.atoms.resindices, dtype=np.int64)

        def _add_residue(i: int) -> None:
            idx = np.nonzero(atom_res == i)[0]
            out.append({
                "label": f"{resnames[i]}{resids[i]}",
                "level": "residue",
                "segid": str(segids[i]),
                "resname": str(resnames[i]),
                "resid": int(resids[i]),
                "n_atoms": int(idx.size),
                "indices": [int(x) for x in idx[:5000]],
            })

        # ---- ① 残基名 + 编号（ETH32、eth 32、ETH-32）
        m = re.match(r"^\s*([A-Za-z+0-9_.\-]+?)\s*[-_ ]?\s*(\d+)\s*$", raw)
        if m:
            want_name, want_id = m.group(1).upper(), int(m.group(2))
            hits = np.nonzero((np.char.upper(resnames.astype(str)) == want_name)
                              & (resids == want_id))[0]
            for i in hits[:limit]:
                _add_residue(int(i))
            if out:
                return {"query": raw, "matches": out, "total": int(hits.size)}

        # ---- ② 纯数字 → 原子索引
        if re.fullmatch(r"#?\d+", raw):
            i = int(raw.lstrip("#"))
            if 0 <= i < n:
                out.append({"label": f"原子 #{i}", "level": "atom",
                            "n_atoms": 1, "indices": [i]})
                return {"query": raw, "matches": out, "total": 1}

        # ---- ③ 原子名匹配（**大小写不敏感**；先精确、再前缀）
        names = np.array([str(x) for x in u.atoms.names], dtype=object)
        up = np.char.upper(names.astype(str))
        hit = np.nonzero(up == raw.upper())[0]
        if not hit.size:
            hit = np.nonzero(np.char.startswith(up, raw.upper()))[0]
        if hit.size:
            for i in hit[:limit]:
                r = int(atom_res[i])
                out.append({"label": f"原子 {names[i]}（{resnames[r]}{resids[r]}）",
                            "level": "atom", "n_atoms": 1, "indices": [int(i)]})
            return {"query": raw, "matches": out, "total": int(hit.size)}

        # ---- ③' 元素 + 序号（C1 = 第 1 个碳原子，C2 = 第 2 个…）
        # 为什么需要：有的拓扑里所有碳都叫 "C"，只写 "C1" 既不是残基也不是原子名，
        # 但用户显然想指"第一个碳"（实测反馈 C1/C2 搜不到）。
        m2 = re.fullmatch(r"([A-Za-z]{1,2})\s*\[?\s*(\d+)\s*\]?", raw)
        if m2:
            el = m2.group(1).upper()
            k = int(m2.group(2))
            same = np.nonzero(np.char.upper(np.array(
                [str(x) for x in u.atoms.elements], dtype=object).astype(str)) == el)[0]
            if not same.size:
                same = np.nonzero(np.char.startswith(
                    np.char.upper(names.astype(str)), el))[0]
            if same.size and 1 <= k <= same.size:
                i = int(same[k - 1])
                r = int(atom_res[i])
                out.append({"label": f"{el} 的第 {k} 个原子 → #{i}"
                                     f"（{resnames[r]}{resids[r]}）",
                            "level": "atom", "n_atoms": 1, "indices": [i]})
                return {"query": raw, "matches": out, "total": int(same.size)}

        # ---- ④ 残基名前缀（ETH → 前若干个 ETH 残基）
        pre = raw.lstrip("#").upper()
        hits = np.nonzero(np.char.startswith(
            np.char.upper(resnames.astype(str)), pre))[0]
        for i in hits[:limit]:
            _add_residue(int(i))
        return {"query": raw, "matches": out, "total": int(hits.size)}

    def expand(self, level: str, *, index: int | None = None,
               segid: str | None = None, resname: str | None = None,
               component: str | None = None) -> dict:
        """把一次点击/一个名字**展开成完整原子索引列表**（按原始编号）。

        为什么单独做：三维视图为性能可能只画了一部分原子（按整残基截断），
        而"选中整个残基 / 链 / 分子"必须是**完整**的 —— 所以展开交给后端用
        Universe 直接算，和显示无关。
        """
        from mdta.topology import molecule_ids

        az = self.az
        u = az.universe
        n = int(u.atoms.n_atoms)
        lvl = str(level or "atom").lower()

        if component is not None:
            ag = (az.components.get(str(component))
                  or az.catalog.get(str(component))
                  or az.set_groups.get(str(component)))
            if ag is None:
                raise SessionError(f"不存在组分/集合 {component!r}")
            idx = np.asarray(ag.indices, dtype=np.int64)
            label = f"组分 {component}"
        elif level is not None and index is not None:
            i = int(index)
            if not 0 <= i < n:
                raise SessionError(f"原子索引越界：{i}（应在 0..{n - 1}）")
            if lvl in ("atom", "atoms"):
                idx = np.array([i], dtype=np.int64)
                label = f"原子 #{i}"
            elif lvl in ("residue", "residues"):
                rid = np.asarray(u.atoms.resindices, dtype=int)
                idx = np.nonzero(rid == rid[i])[0]
                r = u.residues[int(rid[i])]
                label = f"残基 {r.resname}{r.resid}"
            elif lvl in ("chain", "chains"):
                try:
                    seg = str(u.atoms.segids[i])
                    idx = np.nonzero(np.asarray(u.atoms.segids, dtype=str) == seg)[0]
                    label = f"链 {seg}"
                except Exception as exc:                  # noqa: BLE001
                    raise SessionError(f"体系没有 segid 信息，无法按链选择：{exc}") from exc
            elif lvl in ("molecule", "mol"):
                mol = molecule_ids(u.atoms)
                if mol is None:
                    raise SessionError(
                        "体系缺少分子/键连接信息，无法按分子选择（可改用残基或链）")
                mol = np.asarray(mol, dtype=int)
                idx = np.nonzero(mol == mol[i])[0]
                label = f"分子 #{int(mol[i])}"
            else:
                raise SessionError(f"不认识的层级 {level!r}（atom / residue / chain / molecule）")
        elif segid or resname:
            res = u.residues
            mask = np.ones(int(res.n_residues), dtype=bool)
            if segid:
                mask &= np.asarray(res.segids, dtype=str) == str(segid)
            if resname:
                mask &= np.asarray(res.resnames, dtype=str) == str(resname)
            keep = np.zeros(int(u.atoms.n_atoms), dtype=bool)
            keep[np.asarray(u.atoms.resindices, dtype=int)[np.isin(
                np.asarray(u.atoms.resindices, dtype=int),
                np.nonzero(mask)[0])]] = True
            idx = np.nonzero(keep)[0]
            label = f"{segid or ''}:{resname or ''}"
        else:
            idx = np.arange(n, dtype=np.int64)
            label = "全部原子"

        return {"level": lvl, "label": label, "n_atoms": int(idx.size),
                "indices": [int(x) for x in idx]}

    def query(self, expr: str, *, limit: int = 400000) -> dict:
        """**MDAnalysis 原生选择语句**（``select_atoms``）→ 原子索引列表。

        用户要求（1.0.4）：结构选择里补上 select 语句，这样 ``resid 1:10 and name CA``
        ``resname ETH and around 5.0 resname GRA`` ``prop mass > 12`` 这类写法都能直接用。

        语句写错时 MDAnalysis 会抛各种类型（``SelectionError`` / ``SyntaxError`` /
        ``ValueError``），一律翻成``选择语句无效``+ 原因，别让前端看栈。
        """
        u = self.az.universe
        raw = str(expr or "").strip()
        if not raw:
            raise SessionError("选择语句不能为空")
        try:
            ag = u.select_atoms(raw)
        except Exception as exc:                                 # noqa: BLE001
            raise SessionError(
                f"选择语句无效：「{raw}」→ {type(exc).__name__}: {exc}") from exc
        idx = np.asarray(ag.indices, dtype=np.int64)
        try:
            n_res = int(np.unique(np.asarray(ag.resindices, dtype=np.int64)).size)
        except Exception:                                        # noqa: BLE001
            n_res = 0
        out = {
            "query": raw,
            "label": f"选择「{raw}」",
            "n_atoms": int(idx.size),
            "n_residues": n_res,
            # 一次最多回 40 万个索引：足够覆盖实测体系，同时保护 JSON 体积
            "indices": idx[:limit].tolist(),
            "truncated": bool(idx.size > limit),
        }
        if not idx.size:
            # 空结果基本只有一种原因：名字写错了（体系里根本没有 "name C"，
            # 实际叫 C1/C2…）。把体系里真实存在的名字给几个，用户立刻能改对。
            out["hint"] = self._name_hint(u)
        return out

    @staticmethod
    def _name_hint(u) -> str:
        """空结果时给"体系里到底有什么名字"的样本（最多各 8 个）。"""
        def _sample(getter, n=8):
            try:
                vals = [str(x) for x in getter]
            except Exception:                                    # noqa: BLE001
                return []
            seen, keep = set(), []
            for v in vals:
                if v and v not in seen:
                    seen.add(v)
                    keep.append(v)
                    if len(keep) >= n:
                        break
            return keep

        parts = []
        rn = _sample(u.residues.resnames)
        if rn:
            parts.append("残基名 " + "、".join(rn))
        an = _sample(u.atoms.names)
        if an:
            parts.append("原子名 " + "、".join(an))
        try:
            sn = _sample(u.atoms.segids)
            if sn:
                parts.append("segid " + "、".join(sn))
        except Exception:                                        # noqa: BLE001
            pass
        return ("这条语句没有命中任何原子。体系里实际的名字："
                + "；".join(parts) if parts else "这条语句没有命中任何原子")

    def _view_bonds(self, u, idx) -> tuple[list[int], str]:
        """拓扑键表 → **显示位置**的扁平数组 ``[a0, b0, a1, b1, ...]``。

        为什么必须在这里映射：三维视图的下标是抽稀后的**显示位置**
        （``0..n_displayed-1``），而键表给的是 Universe **原始编号**。上一版把原始
        编号当显示下标用，越界索引直接抛异常、React 把整棵树卸掉（白屏）。所以映射
        只做一次，前端拿到的就是"成对的显示位置"，不再做任何换算。

        只取**两端都在显示集合里**的键：跨截断边界的键画不出来（另一端不画），
        留着只会得到悬空的棍子。
        """
        n_disp = int(np.asarray(idx).size)
        if not n_disp:
            return [], "none"
        try:
            cons = getattr(u, "bonds", None)
            n_b = 0 if cons is None else len(cons)
        except Exception:                                        # noqa: BLE001
            n_b = 0
        if not n_b:
            return [], "none"
        try:
            pairs = np.asarray(u.bonds.indices, dtype=np.int64).reshape(-1, 2)
        except Exception:                                        # noqa: BLE001
            return [], "none"
        pos_of = np.full(int(u.atoms.n_atoms), -1, dtype=np.int64)
        pos_of[np.asarray(idx, dtype=np.int64)] = np.arange(n_disp, dtype=np.int64)
        a = pos_of[pairs[:, 0]]
        b = pos_of[pairs[:, 1]]
        keep = (a >= 0) & (b >= 0)
        if not keep.any():
            return [], "topology"
        flat = np.stack([a[keep], b[keep]], axis=1).ravel()
        return flat.astype(np.int64).tolist(), "topology"

    @staticmethod
    def _cell_matrix(box):
        """``[a,b,c,alpha,beta,gamma]`` → 3×3 矩阵（列是三条棱，三斜标准取向）。

        与前端 `StructureView.jsx::cellVectors` 用同一套公式，两边的盒子必须一致。
        """
        try:
            a, b, c = (float(x) for x in box[:3])
        except Exception:                                        # noqa: BLE001
            return None
        if not (a > 0 and b > 0 and c > 0):
            return None

        def _ang(v, dflt=90.0):
            try:
                x = float(v)
            except Exception:                                    # noqa: BLE001
                return math.radians(dflt)
            return math.radians(x) if 0 < x < 180 else math.radians(dflt)

        al = _ang(box[3] if len(box) > 3 else None)
        be = _ang(box[4] if len(box) > 4 else None)
        ga = _ang(box[5] if len(box) > 5 else None)
        ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
        sg = math.sin(ga) or 1e-9
        cx = c * cb
        cy = c * (ca - cb * cg) / sg
        cz = math.sqrt(max(0.0, c * c - cx * cx - cy * cy))
        return np.array([[a, b * cg, cx],
                         [0.0, b * sg, cy],
                         [0.0, 0.0, cz]], dtype=float)

    def _box_report(self, u, idx) -> dict:
        """模拟盒是否**把原子真正围住了**（没围住就说明坐标没按周期折回）。

        为什么值得算：Material Studio 的 .xsd 经常存的是展开后的坐标（盒子
        51×49×120 Å，原子却铺开 288×273×136 Å）。盒子照画不误，只是会切过
        分子云 —— 界面上要说清楚是"坐标没折回"，而不是看起来像个 bug。
        """
        dims = u.dimensions
        if dims is None or not np.all(np.isfinite(dims[:3])):
            return {"box": None, "box_frac_inside": None, "box_wrapped": None}
        M = self._cell_matrix(dims)
        if M is None:
            return {"box": None, "box_frac_inside": None, "box_wrapped": None}
        try:
            pos = np.asarray(u.atoms.positions, dtype=float)[np.asarray(idx, dtype=int)]
            # ⚠️ M 的**列**才是三条棱（A/B/C），所以分数坐标是 solve(M, p)，
            #    不是 `p @ inv(M)`（那是转置，实测会把 74% 误算成 11%）。
            frac = np.linalg.solve(M, pos.T).T
            inside = bool(np.all((frac > -1e-6) & (frac < 1 + 1e-6), axis=1).mean() > 0.98)
            pct = float(np.all((frac > -1e-6) & (frac < 1 + 1e-6), axis=1).mean())
        except Exception:                                        # noqa: BLE001
            inside, pct = None, None
        return {"box": [float(x) for x in dims[:6]],
                "box_frac_inside": pct, "box_wrapped": inside}

    def visual_rings(self, limit_atoms: int = 30000) -> list:
        """芳香六元环的**原始原子编号**列表（给三维视图画虚线圈）。

        判定见 ``mdta.rings``：6 元碳环 + 环内 6 根键长 ≤1.50 Å。按「原子数 +
        键数」缓存；体系太大（>limit_atoms）或没有键表时返回空。
        """
        u = self.az.universe
        try:
            nb = len(u.bonds)
        except Exception:                                        # noqa: BLE001
            nb = 0
        key = (int(u.atoms.n_atoms), int(nb))
        if getattr(self, "_rings_key", None) == key:
            return list(getattr(self, "_rings_cache", []))
        rings: list = []
        if nb and u.atoms.n_atoms <= int(limit_atoms):
            try:
                from mdta.rings import analyze_rings

                st = analyze_rings(u)
                rings = [[int(x) for x in cyc]
                         for cyc, is_benz in st["rings"] if is_benz]
            except Exception:                                    # noqa: BLE001
                rings = []
        self._rings_key = key
        self._rings_cache = rings
        return list(rings)

    def structure(self, frame: int = 0, max_atoms: int = 20000,
                  max_bonds: int = 0) -> dict:
        """给三维视图**拓扑（首帧）坐标** + 元素 + 层级信息。

        1.0.4 精简（用户要求）：**只显示拓扑的坐标**，不逐帧读取 —— 逐帧要重新
        读偏移/解压整帧，纯属浪费；这里最多 seek 到第 0 帧一次。`frame` 参数保留
        但不再用于"看第几帧"（界面上的帧选择已删除）。

        为什么一次把该给的都给了：前端要画球（坐标 / 元素 / 半径）、可选的键
        （只在**拓扑自带可靠键表**时才给，绝不拿距离推断冒充），以及每个原子的
        层级信息（残基 / 链 / 分子）用来做"点一个原子选中整个残基/分子"。

        ``max_atoms`` 只影响**显示用的抽稀**：抽稀时返回的 ``index`` 仍是
        **原始 Universe 的原子编号**（0 基），前端提交选择时用的就是它 ——
        编号绝不因抽稀或隐藏而改变。
        """
        from mdta.topology import molecule_ids

        az = self.az
        u = az.universe
        n = int(u.atoms.n_atoms)
        # 只用**拓扑/首帧**坐标：已经在第 0 帧就不动，否则 seek 一次（不做逐帧读取）
        k = 0
        try:
            if len(u.trajectory) and int(u.trajectory.ts.frame) != 0:
                u.trajectory[0]
        except Exception:                                        # noqa: BLE001
            pass
        pos = np.asarray(u.atoms.positions, dtype=float)

        # ---- 元素（缺了就用原子名/类型推；再缺就标 X）
        try:
            el = [str(x).strip().upper() or "X" for x in u.atoms.elements]
        except Exception:                                        # noqa: BLE001
            el = ["X"] * n
        names = []
        try:
            names = [str(x) for x in u.atoms.names]
        except Exception:                                        # noqa: BLE001
            names = [""] * n
        resid = np.asarray(u.atoms.resindices, dtype=int) if n else np.zeros(0, int)
        try:
            resids = np.asarray(u.residues.resids, dtype=int)
            resnames = [str(x) for x in u.residues.resnames]
        except Exception:                                        # noqa: BLE001
            resids, resnames = np.arange(int(u.residues.n_residues)), ["RES"] * int(
                u.residues.n_residues)
        try:
            segids = [str(x) for x in u.atoms.segids]
        except Exception:                                        # noqa: BLE001
            segids = [""] * n
        mol = molecule_ids(u.atoms)
        mol = np.zeros(n, dtype=int) if mol is None else np.asarray(mol, dtype=int)

        # ---- 抽稀（只影响显示，index 保持原始编号）
        # ⚠️ 按**整残基**取，而不是等间隔抽原子：等间隔会把分子切成碎片，键表就画不
        #    出来了（分子内键的两端很少同时落在同一个抽稀子集里）。按整残基取既能
        #    保住化学键，又保证"抽稀只是少画几个残基"。
        n_res = int(u.residues.n_residues)
        truncated = False
        if max_atoms and n > max_atoms:
            counts = np.bincount(np.asarray(u.atoms.resindices, dtype=int),
                                 minlength=n_res)
            cum = np.cumsum(counts)
            n_keep_res = int(np.searchsorted(cum, int(max_atoms), side="right"))
            n_keep_res = max(1, min(n_keep_res, n_res))
            keep_res = np.zeros(n_res, dtype=bool)
            keep_res[:n_keep_res] = True
            idx = np.nonzero(keep_res[np.asarray(u.atoms.resindices, dtype=int)])[0]
            truncated = True
        else:
            idx = np.arange(n)
        step = 1 if idx.size == n else 0

        # ---- 键表：球棍模型的"棍"。1.0.4 用户要求按 Material Studio 的样子画，
        #      所以这里把**拓扑自带的键**映射成显示位置后返回（绝不用距离推断冒充，
        #      没有键表就只画球，`bond_source="none"` 说明原因）。
        bonds, bond_kind = self._view_bonds(u, idx)

        # radii 用共价半径当显示半径（相对大小看得出来即可，单位 Å）
        from mdta.topology import COVALENT_RADII       # type: ignore

        radii = [float(COVALENT_RADII.get(e, 1.5)) for e in el]

        mol_in_view = mol[idx] if idx.size else np.zeros(0, int)
        out = {
            "frame": k,
            "time_ps": float(getattr(u.trajectory.ts, "time", 0.0) or 0.0),
            "n_atoms": n,
            "n_displayed": int(idx.size),
            "step": int(step),
            "n_residues": n_res,
            "n_residues_shown": (int(np.unique(
                np.asarray(u.atoms.resindices, dtype=int)[idx]).size)
                if idx.size else 0),
            "index": idx.tolist(),
            "xyz": np.round(pos[idx], 3).tolist() if idx.size else [],
            "elements": [el[i] for i in idx],
            "names": [names[i] for i in idx],
            "radii": [radii[i] for i in idx],
            "resid": [int(resids[resid[i]]) for i in idx],
            "resname": [resnames[resid[i]] for i in idx],
            "segid": [segids[i] for i in idx],
            "molid": mol_in_view.tolist(),
            "bonds": bonds,
            "bond_source": bond_kind,
            "n_bonds": int(len(bonds) // 2),
            "box": serialize._num(u.dimensions),
            # 盒子照画，但要说清楚它有没有把原子围住（坐标未按周期折回时盒外还有原子）
            **{k: v for k, v in self._box_report(u, idx).items() if k != "box"},
            "truncated": bool(truncated),
            # 拓扑是否够用来做"分子/残基"层级选择
            "has_bonds": bond_kind == "topology",
            "n_molecules": int(mol_in_view.max()) + 1 if mol_in_view.size else 0,
        }
        # 芳香六元环（苯环）的原子编号：三维视图画环心虚线圆用（按拓扑缓存）
        try:
            out["rings"] = self.visual_rings()
        except Exception:                                        # noqa: BLE001
            out["rings"] = []
        return out

    # --------------------------------------------------- 分子 / 聚合物链识别
    def recognition(self, *, refresh: bool = False) -> dict:
        """执行（或读取缓存的）**分子与聚合物链识别**结果，返回摘要 + 结构树。

        识别只做一次并缓存在会话里（recognize.md §5.2）：结构树反复展开、
        点选对象都读缓存；换体系就是新 Session，旧结果自然失效。
        缓存标识取 ``(universe id, 原子数, 键数)``，万一有人动了拓扑也会重算。
        """
        rec = self._recognition(refresh=refresh)
        out = rec.tree()
        out["cached"] = not refresh
        return out

    def _recognition(self, *, refresh: bool = False):
        from mdta.recognize import recognize

        u = self.az.universe
        try:
            nb = len(u.bonds)
        except Exception:                                        # noqa: BLE001
            nb = 0
        key = (id(u), int(u.atoms.n_atoms), int(nb))
        if refresh or self._recog is None or self._recog_key != key:
            self._recog = recognize(u)
            self._recog_key = key
        return self._recog

    def recognition_objects(self, *, group: int | None = None, section: str = "",
                            offset: int = 0, limit: int = 100) -> dict:
        """某个分组（或某类别）下的对象清单（分页）—— 结构树展开到对象层用。"""
        rec = self._recognition()
        try:
            return rec.objects(group, section=section, offset=int(offset),
                               limit=int(limit))
        except ValueError as exc:
            raise SessionError(str(exc)) from exc

    def recognition_expand(self, object_type: str, object_id: str) -> dict:
        """把结构树上的节点展开成**完整原子索引**（原始 Universe 编号）。

        识别结果里的 ``mol_of_atom`` 就是权威归属，所以这里只是查表 ——
        不存在"前端自己算归属"的情况。
        """
        rec = self._recognition()
        try:
            return rec.expand(object_type, object_id)
        except ValueError as exc:
            raise SessionError(str(exc)) from exc

    def recognition_detail(self, object_id: str) -> dict:
        """单个对象/链的详情（右侧属性区）。"""
        from mdta.recognize import _parse_mol_id

        rec = self._recognition()
        m = _parse_mol_id(object_id)
        if m is None or not 0 <= m < rec.n_molecules:
            raise SessionError(f"不存在的对象 {object_id!r}")
        return rec.detail(m)

    def recognition_confirm(self, payload: Mapping[str, Any]) -> dict:
        """人工确认 / 改名 / 改类型（**只影响显示，不动拓扑数据**）。

        请求：``{"object_id": "C000001", "name": "PE-A", "type": "polymer",
                 "chain_type": "linear", "status": "manually_confirmed"}``
        """
        rec = self._recognition()
        from mdta.recognize import _parse_mol_id

        oid = str(payload.get("object_id") or "").strip()
        m = _parse_mol_id(oid)
        if m is None or not 0 <= m < rec.n_molecules:
            raise SessionError(f"不存在的对象 {oid!r}")
        cur = dict(rec.overrides.get(rec.mol_id(m)) or {})
        for k in ("name", "type", "chain_type", "status"):
            v = payload.get(k)
            if v is not None and str(v).strip():
                cur[k] = str(v).strip()
        cur.setdefault("status", "manually_confirmed")
        rec.overrides[rec.mol_id(m)] = cur
        return {"ok": True, "object_id": rec.mol_id(m), "override": cur,
                "detail": rec.detail(m)}

    def hierarchy(self, *, segid: str | None = None, resname: str | None = None,                  offset: int = 0, limit: int = 400) -> dict:
        """结构层级树：体系 → 组分 → 链（segid+resname）→ 残基（分页）。

        层级完全按**实际拓扑**生成，不假设"残基 = 分子"：
        · 组分来自 :meth:`catalog`（自动识别的化学组分 + 已定义的集合）；
        · 链来自 :func:`mdta.selection.list_chains`（按 segid + resname 归并的类型）；
        · 残基按 ``(segid, resname)`` 分页返回（上万个残基不可能一次给完）。
        """
        az = self.az
        u = az.universe
        comps = [{"name": c["name"], "n_atoms": c["n_atoms"],
                  "n_molecules": c.get("n_molecules"),
                  # 残基构成：界面上按残基进一步区分（protein / water / ion 太笼统）
                  "residue_mix": c.get("residue_mix") or [],
                  "n_residue_kinds": c.get("n_residue_kinds", 0)}
                 for c in self.catalog()]
        chains = self.chains()
        # 给每条链一个"身份"（分子式 / 名称 / 类别 / 聚合度）：取该链类型**第一个分子**
        # 的元素组成与环信息，交给 mdta.chem 推断（不猜力场，只做确定性描述）。
        chem_cache: dict[tuple, dict] = {}
        for ch in chains:
            key = (str(ch.get("segid")), str(ch.get("resname")))
            if key not in chem_cache:
                chem_cache[key] = self._chain_chem(*key)
            ch["chem"] = chem_cache[key]

        # 每个组分只挂**自己**的链（以前前端把全局链表摊在每个组分下面，
        # 于是两个组分展开后看起来一模一样 —— 用户实测）。
        try:
            res_seg = np.array([str(x) for x in u.residues.segids], dtype=object)
            res_rn = np.array([str(x) for x in u.residues.resnames], dtype=object)
            for c in comps:
                ag = (az.components.get(c["name"]) or az.catalog.get(c["name"])
                      or az.set_groups.get(c["name"]))
                if ag is None or not ag.n_atoms:
                    c["chains"] = []
                    continue
                ridx = np.unique(np.asarray(ag.atoms.resindices, dtype=np.int64))
                pairs = set(zip(res_seg[ridx].tolist(), res_rn[ridx].tolist()))
                c["chains"] = [ch for ch in chains
                               if (str(ch["segid"]), str(ch["resname"])) in pairs]
        except Exception:                                    # noqa: BLE001
            for c in comps:
                c["chains"] = list(chains)

        out = {
            "n_atoms": int(u.atoms.n_atoms),
            "n_residues": int(u.residues.n_residues),
            "n_chains": len(chains),
            "components": comps,
            "chains": chains,
            "has_bonds": bool(len(u.bonds) > 0),
            "residues": None,
        }
        if segid is not None or resname is not None:
            # ⚠️ 不用 `select_atoms`：MDAnalysis 的选择语法在这个版本里对**带引号的
            # 值**会匹配到 0 个原子（`segid "X"` → 0），而不带引号时 `resname water`
            # 又会撞上它自己的 `water` 关键字（`Unexpected token 'water'`）。
            # 这里要的只是"按 segid + resname 精确匹配"，直接用数组比较最稳。
            res = u.residues
            n_res = int(res.n_residues)
            mask = np.ones(n_res, dtype=bool)
            if segid:
                try:
                    mask &= np.asarray(res.segids, dtype=str) == str(segid)
                except Exception as exc:                      # noqa: BLE001
                    raise SessionError(f"体系没有 segid 信息：{exc}") from exc
            if resname:
                mask &= np.asarray(res.resnames, dtype=str) == str(resname)
            picked = np.nonzero(mask)[0]
            total = int(picked.size)
            page = picked[int(offset):int(offset) + int(limit)]
            rows = []
            for i in page:
                r = res[i]
                idxs = np.asarray(r.atoms.indices, dtype=int)
                rows.append({
                    "resid": int(r.resid),
                    "resname": str(r.resname),
                    "segid": str(r.segid),
                    "n_atoms": int(idxs.size),
                    "first_index": int(idxs.min()) if idxs.size else -1,
                    # 前 200 个原子索引足够做"点残基选中它"（更大的残基走 /structure）
                    "indices": [int(x) for x in idxs[:200]],
                })
            out["residues"] = {"total": total, "offset": int(offset),
                               "limit": int(limit), "rows": rows,
                               "query": {"segid": segid or "", "resname": resname or ""}}
        return out

    def set_atoms(self, spec: Mapping[str, Any]) -> dict:
        """按一条集合定义算出**原子索引**（只读；分析跑着的时候也能用）。

        前端拿它把"元素 C / 链状分子 / 含环分子 / 大分子"做成**可切换的选择条件**：
        点一下 = 把这些原子并进三维选择，再点一下 = 取消 —— 而不是一点就偷偷建一个集合。
        这里只读 Universe（`_sets_only`），不碰 `Analyzer`，所以运行中同样安全。
        """
        spec = dict(spec or {})
        # 只做选择用的临时名字（`build_set` 需要 name 字段才会处理）
        spec.setdefault("name", "_probe")
        infos, groups = self._sets_only([spec])
        if not groups:
            raise SessionError("集合定义为空，算不出原子")
        key, ag = next(iter(groups.items()))
        info = infos[0] if infos else {}
        idx = np.sort(np.asarray(ag.indices, dtype=np.int64))
        out = {
            "label": str(info.get("label") or key),
            "kind": str(info.get("kind") or "atom"),
            "n_atoms": int(idx.size),
            "indices": idx.tolist(),
            "spec": {k: v for k, v in spec.items() if k != "name"},
        }
        try:
            out["n_molecules"] = int(info.get("n_molecules") or 0)
        except Exception:                                        # noqa: BLE001
            pass
        return out

    def add_visual_selection(self, name: str, indices, *, save: bool = True,
                             from_meta: Mapping | None = None) -> dict:
        """把三维视图里选中的**原子索引**变成一条集合（`set:<名字>`）。

        走的是现有 Set 机制（`mdta.sets.build_set` 的 ``indices`` 分支）：
        于是它和手工定义的集合完全等价 —— 出现在组分清单里、可勾选参与分析、
        MSD 的追踪语义也按"是否整分子"自动判定。

        ``from_meta``：这条集合是**由什么选出来的**（范围多少原子、加了哪些筛选
        条件）。只用于界面悬浮说明，不参与任何计算，随 `sets_spec` 一起存会话。
        """
        from mdta.sets import SetError as _SetError

        nm = str(name or "").strip()
        if not nm:
            raise SessionError("集合需要一个名字")
        if nm.startswith("set:"):
            nm = nm[4:]
        spec = {"name": nm, "indices": [int(x) for x in (indices or [])]}
        if isinstance(from_meta, Mapping) and from_meta:
            spec["from"] = dict(from_meta)
        if not spec["indices"]:
            raise SessionError("没有选中任何原子（先在三维视图里点选，或在结构树里选中对象）")
        # 后端的校验（索引越界 / 空选择）在这里先做一遍，错误直接回给界面
        try:
            from mdta.sets import build_set as _build

            _ag, info = _build(self.az.universe, spec, self.az.components)
        except _SetError as exc:
            raise SessionError(f"选择无效：{exc}") from exc

        with self._lock:
            # 同名就覆盖（用户在可视化里改选后按"更新集合"）
            self.sets_spec = [dict(s) for s in self.sets_spec
                              if str((s or {}).get("name") or "") != nm]
            self.sets_spec.append(spec)
            if save:
                self.selections[nm] = {"name": nm, "n_atoms": int(info["n_atoms"]),
                                       "kind": info["kind"],
                                       "indices": spec["indices"],
                                       "label": info["label"]}
            if self.is_running:
                # 分析正在跑：只存索引与定义，不重建分析器里的集合（下一轮 run 生效）。
                # 用户实测要求：跑分析的时候照样能选原子、建集合。
                infos, groups = self._sets_only(self.sets_spec)
                self.sets_info = infos
                self._pending_sets = groups
                cur = next((i for i in infos if i.get("name") == f"set:{nm}"), None)
                out_info = cur or {**{k: v for k, v in info.items()
                                      if k != "indices"}, "name": f"set:{nm}"}
                return {"ok": True, "deferred": True,
                        "set": {**out_info, "set_name": nm,
                                "from": dict(from_meta or {})},
                        "selection": None,
                        "selections": self.selection_list()}
            # 立刻生效：作为参与分析的组分重建集合
            wanted = [str(k) for k in self.az.components] + [f"set:{nm}"]
            info_list = self.apply_selection({"components": wanted,
                                              "sets": list(self.sets_spec),
                                              "primary": dict(self.primary_req
                                                              or {"mode": "auto"})})
        return {"ok": True,
                # `name` 用带前缀的 `set:<名字>`（与 components 里的键一致）；
                # `set_name` 是没有前缀的原始名字，界面回显时用哪个都行。
                "set": {**{k: v for k, v in info.items() if k != "indices"},
                        "name": f"set:{nm}", "set_name": nm,
                        "from": dict(from_meta or {})},
                "selection": info_list, "selections": self.selection_list()}

    def selection_list(self) -> list[dict]:
        """已保存的可视化选择（名字 / 原子数 / 类型），供界面"加载已保存的选择"。"""
        out = []
        for nm, rec in (self.selections or {}).items():
            out.append({"name": nm, "n_atoms": int(rec.get("n_atoms") or 0),
                        "kind": rec.get("kind", ""), "label": rec.get("label", "")})
        return out

    def load_selection(self, name: str) -> dict:
        """把某个已保存的可视化选择重新变成集合（并参与分析）。"""
        rec = (self.selections or {}).get(str(name))
        if rec is None:
            raise SessionError(f"没有保存过名为 {name!r} 的选择")
        return self.add_visual_selection(name, rec.get("indices") or [])

    def delete_selection(self, name: str) -> dict:
        """删掉一条已保存的选择（同时把它从组分 / 集合定义里摘掉）。"""
        nm = str(name)
        if nm.startswith("set:"):
            nm = nm[4:]
        with self._lock:
            self.selections.pop(nm, None)
            self.sets_spec = [dict(s) for s in self.sets_spec
                              if str((s or {}).get("name") or "") != nm]
            self._pending_sets.pop(f"set:{nm}", None)
            # 分析跑着的时候只摘定义（不碰分析器的字典，避免与运行线程抢同一个 dict）；
            # 下一轮 run 会按新的 sets_spec 重建，等于同样删掉。
            if not self.is_running:
                self.az.components.pop(f"set:{nm}", None)
                self.az.catalog.pop(f"set:{nm}", None)
                self.az.set_groups.pop(f"set:{nm}", None)
                self.az.msd_group_objects.pop(f"set:{nm}", None)
            self.sets_info = [x for x in (self.sets_info or [])
                              if str(x.get("name")) != f"set:{nm}"]
        return {"ok": True, "selections": self.selection_list(),
                "components": self.components(), "catalog": self.catalog()}

    # ----------------------------------------------------------------- 运行
    def _prepare_run(self, req: Mapping[str, Any]) -> dict:
        """校验并落实「选择 + 帧 + 分析项 + 参数」。调用方须持有锁。"""
        req = dict(req or {})
        sel_info = self.apply_selection(req.get("selection"))

        frames = dict(req.get("frames") or {})

        def _f(key):
            v = frames.get(key)
            if v in ("", None):
                return None
            return float(v)

        mf = frames.get("max_frames")
        self.az.set_frames(
            start_ps=_f("start_ps"), stop_ps=_f("stop_ps"),
            interval_ps=_f("interval_ps"), equil_ps=_f("equil_ps"),
            max_frames=int(mf) if mf not in ("", None) else None,
        )
        selection = self.az.require_frames()

        raw_which = req.get("which")
        # ⚠️ 以前写的是 `list(req.get("which") or DEFAULT_ORDER)` —— 前端传**空列表**
        #    （界面默认一个都不勾）时，`[] or DEFAULT_ORDER` 会变成"全跑 14 项"，
        #    跟"没选就不分析"完全相反。现在只有**没传**才用默认顺序，传空就是空。
        which = (list(raw_which) if isinstance(raw_which, (list, tuple))
                 else list(DEFAULT_ORDER))
        if not which:
            raise SessionError("还没有勾选任何分析项（在「分析功能」里勾选后开始分析）")
        # 校验要针对**全部已注册分析项**，而不是 DEFAULT_ORDER（后者只是"默认
        # 勾选的那批"）。否则界面里能勾上、提交却报"未知分析项"的分析（如
        # rdf2d / comdist 这类可选项）在前端就完全用不了。
        known = set(ANALYSIS_TITLES)
        bad = [w for w in which if w not in known]
        if bad:
            raise SessionError(f"未知分析项 {bad}；可用: {sorted(known)}")

        params = {k: dict(v) for k, v in (req.get("params") or {}).items()
                  if isinstance(v, Mapping)}
        return {"sel_info": sel_info, "selection": selection,
                "which": which, "params": params,
                # 每项的"输入指纹"：选择（主链 + 参与的组分 + 集合定义）+ 帧 + 该项参数。
                # 扩展分析时拿它和上次算这份结果时的指纹比，不一致就**重算**。
                "sig": {n: self._input_sig(n, sel_info, frames, params) for n in which},
                # 先跑几帧预估用时（默认开）；传 false 可跳过
                "estimate": bool(req.get("estimate", True))}

    _SIG_KEYS = ("start_ps", "stop_ps", "equil_ps", "interval_ps", "max_frames")

    def _selection_sig(self, sel_info: Mapping[str, Any], frames: Mapping[str, Any]) -> str:
        """「选择 + 帧」的指纹（与具体分析项无关的那一半）。"""
        fr = {k: frames.get(k) for k in self._SIG_KEYS}
        sel = {
            "primary": sel_info.get("primary") or {"mode": "auto"},
            # 参与分析的组分：排序后比较，勾选顺序不同不算变
            "components": sorted(str(n) for n in (sel_info.get("active") or [])),
            "sets": sorted(
                ({"name": str(s.get("name") or ""), "elements": sorted(
                    str(e) for e in (s.get("elements") or [])),
                  "shape": str(s.get("shape") or "all"),
                  "min_atoms": int(s.get("min_atoms") or 0),
                  "max_atoms": int(s.get("max_atoms") or 0),
                  "pure": bool(s.get("pure"))}
                 for s in (self.sets_spec or [])),
                key=lambda d: d["name"]),
        }
        return json.dumps({"sel": sel, "frames": fr}, sort_keys=True, default=str)

    def _input_sig(self, name: str, sel_info: Mapping[str, Any],
                   frames: Mapping[str, Any], params: Mapping[str, Any]) -> str:
        """某一项分析的完整输入指纹（选择 + 帧 + 该项自己的参数）。"""
        return (self._selection_sig(sel_info, frames)
                + "|" + json.dumps(params.get(name) or {}, sort_keys=True, default=str))

    def _payload(self, prep: Mapping[str, Any], *, extra: Mapping | None = None) -> dict:
        """组装返回给前端的运行结果（调用方须持有锁）。"""
        selection = prep["selection"]
        timings = dict(getattr(self.az, "timings", {}))
        out = {
            "frames": {
                "n_frames": selection.n_frames,
                "times_ps": serialize._num(selection.times_ps),
                "notes": list(selection.notes),
                "describe": selection.describe(),
            },
            "selection": prep["sel_info"],
            "results": self.results_json,
            "summary_rows": serialize.summary_rows(self.results_json),
            "titles": {k: ANALYSIS_TITLES.get(k, k) for k in DEFAULT_ORDER},
            "timings": {k: round(v, 2) for k, v in timings.items()},
            # 「跳过」而不是「失败」的项（体系不满足前提，如没有晶胞信息）
            "skipped": dict(getattr(self.az, "skips", {}) or {}),
            "elapsed_sec": round(sum(timings.values()), 2),
        }
        if extra:
            out.update(extra)
        return out

    def run(self, req: Mapping[str, Any]) -> dict:
        """同步跑完所有分析并返回结果（保留给 CLI / 测试用）。"""
        with self._lock:
            prep = self._prepare_run(req)
            raw = self.az.run_all(prep["which"], params=prep["params"], outdir=None,
                                  verbose=False, raise_errors=False)
            self.results_json = OrderedDict(
                (name, serialize.result_to_json(res)) for name, res in raw.items())
            self.sigs = dict(prep["sig"])
            return self._payload(prep)

    # ------------------------------------------------------------- 状态恢复
    def state(self) -> dict:
        """完整的会话快照：前端 **F5 刷新后**据此把界面恢复原样。

        包含：体系信息、组分/链清单、当前选择、帧选择、**全部已完成结果**、
        以及后台运行的当前状态（正在跑 / 已跑完 / 报错）。
        分析本身跑在后端线程里，所以刷新页面**不会**让计算停下来，
        前端恢复后接着拉进度即可；只有显式「取消」才会停。
        """
        with self._lock:
            az = self.az
            sel_info = {
                "primary_label": az.primary_label,
                # 主链的**请求**（auto / component set:C / chain / custom）：
                # 刷新后前端要把下拉框恢复成同一档，否则下一次运行会悄悄退回
                # "自动"，分析对象就变了。
                "primary": dict(self.primary_req or {"mode": "auto"}),
                "primary_atoms": int(az.primary.n_atoms) if az.primary is not None else 0,
                "primary_molecules": self.primary_molecules(),
                "components": self.components(),
                # 勾选名字清单（前端恢复"哪些胶囊是亮的"；`components` 只有勾选的）
                "active": [str(n) for n in az.components],
            }
            # 「分析集合（Set）」：定义 + 预览（原子数 / 追踪方式）+ 是否勾选参与。
            # 勾选状态以 `components` 里有没有 `set:<名字>` 为准（唯一的真相来源）。
            live = {str(n) for n in az.components}
            sets_echo = []
            for spec in self.sets_spec:
                spec = dict(spec or {})
                nm = str(spec.get("name") or "")
                idx = spec.pop("indices", None)
                echo = {**spec, "name": nm, "on": f"set:{nm}" in live}
                if idx is not None:
                    # 可视化选择：只回报个数（上万个索引不必来回传）
                    echo["kind"] = "visual"
                    echo["n_indices"] = len(idx)
                sets_echo.append(echo)
            sets_info = [dict(x) for x in (self.sets_info or [])]
            frames = None
            try:
                fs = self.az.frames
                if fs is not None:
                    frames = {
                        "n_frames": fs.n_frames,
                        "describe": fs.describe(),
                        "notes": list(fs.notes),
                        "start_ps": getattr(fs, "start_ps", None),
                        "stop_ps": getattr(fs, "stop_ps", None),
                        "equil_ps": getattr(fs, "equil_ps", None),
                        "interval_ps": getattr(fs, "interval_ps", None),
                    }
            except Exception:                                # noqa: BLE001
                frames = None
            prog = self.run_progress(0)
            prog.pop("results", None)       # 完整结果走顶层 results
            prog.pop("summary_rows", None)
            prog.pop("titles", None)
            return {
                "sid": self.sid,
                "info": self.info(),
                "load_notes": self.load_notes(),
                "chains": self.chains(),
                # 候选成分清单：**全部**自动识别的组分 + 已定义集合（带 active 标记）。
                # 前端恢复后组分胶囊 / 主链下拉框都读这一份 —— 只回 `components`
                # 会导致"取消勾选后列表空了、刷新后也没得选"（用户实测）。
                "catalog": self.catalog(),
                "components": self.components(),
                "selection": sel_info,
                # 刷新恢复（用户要求：Set 模块不能一刷新就消失）
                "sets": sets_echo,
                "sets_info": sets_info,
                # 已保存的可视化选择（三维视图 → 集合）
                "selections": self.selection_list(),
                # 集合编辑器要用的元素多选项 / 一键预设（刷新后同样要能继续编辑）
                "elements": _available_elements(az.universe),
                "set_presets": _set_presets(az.universe, az.components),
                "frames": frames,
                "results": self.results_json,
                "summary_rows": serialize.summary_rows(self.results_json),
                "titles": {k: ANALYSIS_TITLES.get(k, k) for k in DEFAULT_ORDER},
                "timings": dict(getattr(self.az, "timings", {})),
                "skipped": dict(getattr(self.az, "skips", {}) or {}),
                "elapsed_sec": round(sum(getattr(self.az, "timings", {}).values()), 2),
                "job": prog,
            }

    def clear_results(self) -> dict:
        """清空已有结果（界面上"清空结果重跑"用）。"""
        with self._lock:
            self.assert_idle()
            self.results_json = OrderedDict()
            self.az.results = OrderedDict()
            self.az.timings = {}
            self.sigs = {}
            self._job = None
        return {"ok": True, "results": 0}

    # ------------------------------------------------------- 实时（流式）运行
    @property
    def job(self) -> "RunJob | None":
        return self._job

    def assert_idle(self) -> None:
        """正在跑分析时，禁止别的接口去动同一个 Universe。"""
        if self.is_running:
            raise SessionError("分析正在运行中，请先等待完成或点「取消」")

    @property
    def is_running(self) -> bool:
        job = self._job
        return job is not None and job.status == "running"

    def start_run(self, req: Mapping[str, Any]) -> dict:
        """**立即返回**并在线程里跑分析；前端轮询 :meth:`run_progress` 拿增量结果。

        ``extend=True``（用户要求）：**保留还能用的已有结果**，只算本次新勾的项 ——
        这样"先跑几项、再补几项"不用重算前面的。

        ⚠️ 但"已有结果"必须**输入没变**才算能用：早先只要名字在结果集里就跳过，
        于是用户第一轮只勾了一两个组分跑 MSD、第二轮加了组分再跑时，MSD 被当成
        "已有结果"直接跳过 —— 界面上看起来就是"加了组分却不跑"（而且真的没跑，
        图还是旧组分算的）。现在每份结果记一个输入指纹（选择 + 帧 + 该项参数），
        指纹对不上就**作废重算**，指纹对得上才保留。
        默认 ``extend=False`` = 清空重跑（老行为）。
        """
        req = dict(req or {})
        extend = bool(req.get("extend"))
        with self._lock:
            self.assert_idle()
            prep = self._prepare_run(req)
            prep["extend"] = extend
            job = RunJob(order=list(prep["which"]))
            if extend:
                keep, stale = [], []
                for nm in prep["which"]:
                    if nm not in self.results_json:
                        continue
                    if self.sigs.get(nm) == prep["sig"][nm]:
                        keep.append(nm)
                    else:
                        stale.append(nm)
                # 输入变了的旧结果必须扔掉：留着会让"结果与当前选择不一致"，
                # 而界面上又看不出来是哪一轮算的。
                for nm in stale:
                    self.results_json.pop(nm, None)
                    self.az.results.pop(nm, None)
                    self.az.timings.pop(nm, None)
                    self.sigs.pop(nm, None)
                # 没被本次勾选的其它结果（用户这次没勾它们）保持原样，不进 completion
                job.preexisting = len(keep)
                job.completion = list(keep)
                new = [n for n in prep["which"] if n not in keep]
                job.message = (
                    f"扩展分析：保留 {len(keep)} 项结果"
                    + (f"，重算 {len(stale)} 项（组分 / 帧 / 参数变了）" if stale else "")
                    + (f"，新算 {len(new) - len(stale)} 项" if new else "")
                    + ("，没有需要新算的项" if not new and not stale else "")
                )
            else:
                # 新一轮开始：清空上一轮，避免旧结果与新选择混在一起
                self.results_json = OrderedDict()
                self.az.results = OrderedDict()
                self.az.timings = {}
                self.sigs = {}
            self._job = job
        th = threading.Thread(target=self._run_worker, args=(prep,),
                              name=f"mdta-run-{self.sid}", daemon=True)
        job.thread = th
        th.start()
        return self.run_progress(0)

    def _estimate_progress(self, i: int, n: int, name: str) -> None:
        """预估阶段的进度：告诉用户正在量哪一项的速度。"""
        job = self._job
        if job is None:
            return
        with self._lock:
            job.message = (f"正在预估用时 {i + 1}/{n}："
                           f"{ANALYSIS_TITLES.get(name, name)}（先跑几帧实测）")
            job.current = name

    @staticmethod
    def _shortest_first(which, estimates) -> list[str]:
        """按预估用时短→长排序（估不出来的排最后），与 ``run_all`` 用同一函数。"""
        return sort_by_estimate(list(which), estimates)

    def _run_worker(self, prep: Mapping[str, Any]) -> None:
        """后台线程：逐项分析，每算完一项就把结果写进结果集并通知轮询方。"""
        job = self._job
        if job is None:
            return
        try:
            def _progress(frac, message):
                with self._lock:
                    job.frac = max(job.frac, float(frac))
                    job.message = str(message)
                    # 当前分析项：直接取 Analyzer 记录的名字（不解析进度文字，
                    # 文案一改就失效）。收尾阶段（导出）名字为空时保留上一项，
                    # 免得顶栏那一行闪成空白。
                    if getattr(self.az, "current", ""):
                        job.current = self.az.current

            def _on_result(name, res, elapsed, done, total):
                payload = serialize.result_to_json(res)
                with self._lock:
                    self.results_json[name] = payload
                    job.timings[name] = round(float(elapsed), 2)
                    # 记下这份结果是用哪套输入算的（扩展分析时据此判断能不能留）
                    self.sigs[name] = prep["sig"].get(name, "")
                    job.completion.append(name)
                    job.frac = max(job.frac, done / max(total, 1))
                    job.message = f"已完成 {done}/{total}：{ANALYSIS_TITLES.get(name, name)}"
                    job.last_result = name

            # ---- 阶段一：先跑几帧，实测每项用时并外推；按短→长排好序 -----
            estimates: dict[str, float | None] = {}
            # ⚠️ 以前写成 `len(which) > 1`：只勾**一项**时完全不预估，
            #    界面上就看不到"预计约 X"，用户实测反馈过（"只测一项也要给预计用时"）。
            #    现在哪怕只勾一项也预估 —— 排序对单项没意义，但"要跑多久"照样有用。
            if prep.get("estimate", True) and len(prep["which"]) >= 1:
                with self._lock:
                    job.phase = "estimate"
                    job.message = "正在预估各分析项用时（每项先跑几帧实测）…"
                try:
                    est = self.az.estimate_times(
                        prep["which"], params=prep["params"], cancel=job.cancel,
                        on_item=lambda i, n, nm: self._estimate_progress(i, n, nm))
                    estimates = {n: r.get("est_sec") for n, r in est.items()}
                    with self._lock:
                        job.estimates = {n: v for n, v in estimates.items()
                                         if v is not None}
                        if not job.estimates:
                            # 一项都没估出来：直接把原因说清楚（通常是"帧数太少"）
                            first = next(iter(est.values()), {})
                            job.estimate_note = (first.get("note")
                                                 or "没有可用的预估，按勾选顺序跑")
                        else:
                            missing = [n for n, v in estimates.items() if v is None]
                            job.estimate_note = (
                                "预估完成" if not missing else
                                "预估完成（" + "、".join(
                                    ANALYSIS_TITLES.get(m, m) for m in missing)
                                + " 估不出来）")
                        job.order = self._shortest_first(prep["which"], estimates)
                        job.phase = "run"
                        job.message = ("预估完成，按用时短 → 长依次计算…"
                                       if job.estimates else "开始计算…")
                except Exception as exc:  # noqa: BLE001 - 估不出来不该阻断分析
                    with self._lock:
                        job.estimate_note = f"预估失败，直接开跑（{exc}）"
                        job.phase = "run"
            else:
                with self._lock:
                    job.phase = "run"
                    job.order = list(prep["which"])
                    # 为什么必须写一句原因：界面上的「预估用时」区块以前只在"估出
                    # 东西"时才有内容，于是"这次没预估"看起来就像整个模块被删了。
                    # 跳过预估只有两种情况（帧数 < 40 由 estimate_times 自己给出
                    # 更具体的原因，走的是上面那条分支），都要在这里讲清楚。
                    job.estimate_note = (
                        "已关闭「先预估用时」，按勾选顺序直接跑"
                        if not prep.get("estimate", True)
                        else "只勾选了 1 项，无需预估排序，直接跑")

            # 扩展模式：已经有结果、且**输入没变**的项不重算（省掉重复计算）。
            # 输入变了的项已在 start_run 里作废，所以这里自然会重算它们。
            to_run = [n for n in prep["which"] if n not in self.results_json]
            if not to_run:
                with self._lock:
                    job.status = "done"
                    job.frac = 1.0
                    job.message = ("勾选的这些分析项都已有结果，且组分 / 帧 / 参数没变，"
                                   "无需重算（改了其中任何一项都会自动重算）")
                    job.finished = time.time()
                return
            self.az.run_all(to_run, params=prep["params"], outdir=None,
                            verbose=False, raise_errors=False,
                            progress=_progress, on_result=_on_result,
                            order="shortest" if estimates else "given",
                            estimates=estimates,
                            keep_existing=bool(prep.get("extend")),
                            cancel=job.cancel)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = f"失败：{job.error}"
                job.finished = time.time()
            return
        with self._lock:
            if job.cancel.is_set() and not self.az.results:
                job.status = "cancelled"
                job.message = "已取消（没有算出任何结果）"
            elif job.cancel.is_set():
                job.status = "cancelled"
                job.message = f"已取消（已完成 {len(job.completion)} 项的结果保留）"
            else:
                job.status = "done"
                job.frac = 1.0
                # 只报"这次新算了多少"，保留的那几项单独说 —— 否则扩展跑 1 项时
                # 消息却写"完成 5 项分析"，用户会以为又全算了一遍。
                new_n = max(0, len(job.completion) - job.preexisting)
                job.message = (f"完成：新算 {new_n} 项分析"
                               + (f"（另有 {job.preexisting} 项沿用之前的结果）"
                                  if job.preexisting else ""))
            job.finished = time.time()

    def run_progress(self, after: int = 0) -> dict:
        """轮询用：返回运行状态 + **只包含客户端还没有的结果**。

        ``after`` 是客户端已收到的结果条数（按完成顺序）。这样轮询的
        数据量只与"新增了几项"有关，不会每次重复传送全部曲线。
        """
        with self._lock:
            job = self._job
            if job is None:
                return {"status": "idle", "n_total": 0, "n_done": 0,
                        "completion": [], "results": {}, "summary_rows": [],
                        "titles": {k: ANALYSIS_TITLES.get(k, k) for k in DEFAULT_ORDER},
                        "message": "", "frac": 0.0, "current": "", "timings": {},
                        "phase": "", "estimates": {}, "estimate_note": "",
                        "order": [], "pending": [],
                        "elapsed_sec": 0.0, "error": ""}
            after = max(0, int(after))
            fresh = job.completion[after:]
            results = OrderedDict(
                (n, self.results_json[n]) for n in fresh if n in self.results_json)
            elapsed = (job.finished or time.time()) - job.started
            return {
                "status": job.status,
                "phase": job.phase,
                "frac": round(float(job.frac), 4),
                "message": job.message,
                "current": job.current,
                "estimates": dict(job.estimates),
                "estimate_note": job.estimate_note,
                "after": after,
                "n_total": len(job.order),
                "n_done": max(0, len(job.completion) - job.preexisting),
                "order": list(job.order),
                "completion": list(job.completion),
                "pending": [n for n in job.order if n not in job.completion],
                "results": results,
                "summary_rows": serialize.summary_rows(self.results_json),
                "titles": {k: ANALYSIS_TITLES.get(k, k) for k in DEFAULT_ORDER},
                "timings": dict(job.timings),
                # 体系不满足前提而**跳过**的分析（不是失败）：例如非周期体系
                # 做不了界面宽度、没勾组分做不了 RDF。界面按"跳过：原因"显示。
                "skipped": dict(getattr(self.az, "skips", {}) or {}),
                "elapsed_sec": round(elapsed, 2),
                "error": job.error,
                "cancelled": job.cancel.is_set(),
            }

    def cancel_run(self) -> dict:
        """请求取消：会在**当前这项算完后**停下，已完成的结果保留。"""
        # 注意：不能在持锁的情况下调用 run_progress()——它会再次获取同一把锁，
        # 而 threading.Lock 不可重入，会直接死锁（实测卡死在这里）。
        with self._lock:
            job = self._job
            if job is not None and job.status == "running":
                job.cancel.set()
                job.message = "已请求取消，等当前这项算完…"
        return self.run_progress(0)

    # ----------------------------------------------------------------- 导出
    def export(self, req: Mapping[str, Any] | None) -> dict:
        req = dict(req or {})
        self.assert_idle()
        if not self.results_json:
            raise SessionError("还没有分析结果，请先运行分析")
        from mdta.export import export_all

        # 注意区分「没传 formats」和「传了空列表」：前者给默认值，
        # 后者是用户把格式全取消勾选了，应当明确报错而不是悄悄用默认值。
        raw_formats = req.get("formats")
        formats = (["csv", "png"] if raw_formats is None
                   else [str(f).lower() for f in raw_formats])
        bad_fmt = [f for f in formats if f not in ("csv", "png", "xlsx", "excel")]
        if bad_fmt:
            raise SessionError(f"不支持的导出格式 {bad_fmt}；可用: csv / png / excel")
        if not formats:
            raise SessionError("至少要选择一种导出格式（CSV / PNG / Excel）")
        dpi = int(req.get("dpi") or 200)
        panel_pngs = bool(req.get("panel_pngs"))
        # 只导出选中的分析项（前端可按大纲模块勾选）；不传就是全部
        wanted = req.get("which")
        if wanted:
            unknown = [w for w in wanted if w not in self.results_json]
            if unknown:
                raise SessionError(f"这些分析项没有结果，无法导出: {unknown}")
        # 目标目录：用户可自选；不存在就建
        outdir = str(req.get("outdir") or self.outdir).strip() or self.outdir
        outdir = os.path.abspath(os.path.expanduser(outdir))
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as exc:
            raise SessionError(f"无法创建/写入目标目录 {outdir}: {exc}") from exc

        with self._lock:
            names = [n for n in self.results_json if (not wanted or n in wanted)]
            rows = {name: self.az.results[name] for name in names
                    if name in self.az.results}
            if not rows:
                raise SessionError("没有可导出的结果（检查勾选的模块）")
            out = export_all(list(rows.values()), outdir, formats=formats, dpi=dpi,
                             excel=("excel" in formats or "xlsx" in formats),
                             panel_pngs=panel_pngs, verbose=False)

        # 逐文件筛选：前端可以按"导出哪些文件"精确勾选（按大纲模块分板块展示），
        # 没勾上的文件导出后删掉。这里先导出再删，而不是让 export_all 支持
        # 逐面板过滤 —— 少写一层参数，代价只是几个小文件的 I/O。
        kept_files = {"csv": [], "png": [], "panel_png": [], "excel": None}
        want_files = req.get("files")
        if want_files is None:
            kept_files = {"csv": list(out["csv"]), "png": list(out["png"]),
                          "panel_png": list(out.get("panel_png", [])),
                          "excel": out["excel"]}
        else:
            keep = {os.path.basename(str(f)) for f in want_files}
            for key in ("csv", "png", "panel_png"):
                for p in out.get(key, []):
                    if os.path.basename(p) in keep:
                        kept_files[key].append(p)
                    else:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            if out["excel"] and os.path.basename(out["excel"]) in keep:
                kept_files["excel"] = out["excel"]
            elif out["excel"]:
                try:
                    os.remove(out["excel"])
                except OSError:
                    pass

        self.last_export = {
            "dir": out["dir"],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_csv": len(kept_files["csv"]),
            "n_png": len(kept_files["png"]) + len(kept_files["panel_png"]),
            "excel": kept_files["excel"],
            "which": list(names),
            "formats": formats,
        }
        return {**self.last_export, "files": self.list_output_files(outdir)}

    def open_directory(self, path: str | None = None) -> dict:
        """在操作系统的文件管理器里打开目录（"打开目标目录"按钮）。

        只允许打开**确实存在的目录**；路径来自用户自己选择/导出的位置，
        与 ``/api/files`` 的文件浏览能力一致（该接口本来就能浏览整个文件系统）。
        """
        target = str(path or (self.last_export or {}).get("dir") or self.outdir)
        target = os.path.abspath(os.path.expanduser(target))
        if not os.path.isdir(target):
            raise SessionError(f"目录不存在: {target}")
        try:
            if os.name == "nt":
                # explorer 对已存在目录返回 1，属正常，不能据此判失败
                import subprocess
                subprocess.Popen(["explorer", os.path.normpath(target)],
                                 close_fds=True)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", target], close_fds=True)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", target], close_fds=True)
        except Exception as exc:  # noqa: BLE001
            raise SessionError(f"无法打开目录（可能没有桌面环境）: {exc}") from exc
        return {"ok": True, "dir": target}

    def list_output_files(self, outdir: str | None = None) -> list[dict]:
        base = outdir or self.outdir
        out = []
        if not os.path.isdir(base):
            return out
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                out.append({
                    "name": name,
                    "size": os.path.getsize(p),
                    "url": f"/api/session/{self.sid}/files/{name}"
                           + (f"?dir={quote(base)}" if outdir else ""),
                    "path": p,
                })
        return out

    def output_file(self, name: str, outdir: str | None = None) -> str:
        """返回输出目录里某个文件的安全绝对路径。

        ``outdir`` 用于用户自选导出目录的情况；只取文件名部分，
        防止 ``../`` 逃逸出该目录。
        """
        safe = os.path.basename(name)
        base = os.path.abspath(os.path.expanduser(outdir)) if outdir else self.outdir
        path = os.path.join(base, safe)
        if not os.path.isfile(path):
            raise SessionError(f"找不到文件 {safe}")
        return path


class SessionRegistry:
    """会话注册表（进程内，带 TTL 清理）。"""

    def __init__(self, base_outdir: str, ttl_sec: float = 6 * 3600,
                 max_sessions: int = 8):
        self.base_outdir = base_outdir
        self.ttl = float(ttl_sec)
        self.max_sessions = int(max_sessions)
        self._items: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()
        # 「文件读取」进度：token → OpenProgress（前端打开文件时并发轮询）
        self._open_progress: "dict[str, OpenProgress]" = {}
        self._prog_lock = threading.Lock()

    def open_progress(self, token: str) -> dict:
        """查询某次「文件读取」的进度快照（未知 token 返回空快照）。"""
        with self._prog_lock:
            prog = self._open_progress.get(token)
        if prog is None:
            return {"elapsed": 0.0, "stages": [], "stage": "", "n_stages": 0,
                    "done": False, "error": "", "unknown": True}
        return prog.snapshot()

    # ------------------------------------------------------------------ 建立
    def open(self, topology: str, trajectory: str | None = None,
             progress_token: str | None = None,
             guess_bonds: bool | None = None,
             velocities: bool | None = None,
             autodetect: bool | None = None) -> Session:
        """打开一个体系。``progress_token`` 非空时记录加载进度供前端轮询。

        ``guess_bonds=True``：忽略拓扑里已有的键、按坐标重新猜。用于
        ``CONECT`` 不完整的 PDB（Material Studio 导出常见；实测某体系只有
        4393 个键、键图碎片 1203 个，重新猜键后 5953 键 / 124 个分子）。
        ``velocities=True``：读 MS 的 ``.trj`` 时把**速度**一并保留（转成 TRR）。
        ``autodetect=False``：**不自动配对同名轨迹** —— 只给了结构文件时
        就老老实实按单帧结构打开（否则 .xsd 同目录里有个 .xtc 就会被悄悄
        读成六千帧的轨迹分析，用户实测踩过）。
        """
        prog: OpenProgress | None = None
        if progress_token:
            prog = OpenProgress()
            prog.topology = str(topology or "")
            prog.trajectory = str(trajectory or "")
            with self._prog_lock:
                self._open_progress[progress_token] = prog
                if len(self._open_progress) > 8:          # 只留最近几次
                    for k in list(self._open_progress)[:-8]:
                        self._open_progress.pop(k, None)
            set_progress_sink(prog)
        try:
            return self._open_inner(topology, trajectory, prog,
                                    guess_bonds=guess_bonds,
                                    velocities=velocities,
                                    autodetect=autodetect)
        except Exception as exc:  # noqa: BLE001 - 记录失败原因后原样抛出
            if prog is not None:
                prog.error = str(exc)
            raise
        finally:
            set_progress_sink(None)

    def _open_inner(self, topology: str, trajectory: str | None,
                    prog: "OpenProgress | None",
                    guess_bonds: bool | None = None,
                    velocities: bool | None = None,
                    autodetect: bool | None = None) -> Session:
        if not topology:
            raise SessionError("必须提供拓扑文件；可用格式：" + _topology_help())
        if not os.path.isfile(topology):
            raise SessionError(f"找不到拓扑文件: {topology}")
        if trajectory and not os.path.isfile(trajectory):
            raise SessionError(f"找不到轨迹文件: {trajectory}")
        # 槽位校验放在最前面：把"轨迹放进拓扑槽""不认识的后缀"当场说清楚，
        # 而不是丢给 MDAnalysis 抛一个看不懂的错（那种错以前会变成 HTTP 500）。
        if not _can_be_topology(topology):
            if _can_be_trajectory(topology):
                raise SessionError(
                    f"「{os.path.basename(topology)}」是**轨迹**格式"
                    f"（后缀 {_suffix_of(topology)}），不能放在「拓扑」槽里——"
                    f"它没有原子名/元素/键表，按拓扑读进来质量与键表都会错。\n"
                    f"请把它放到「轨迹」槽，并在「拓扑」槽选同体系的文件。\n"
                    f"可用拓扑格式：{_topology_help()}")
            raise SessionError(
                f"不支持的文件格式：{os.path.basename(topology)}"
                f"（后缀 {_suffix_of(topology) or '无'}）。\n"
                f"可用拓扑格式：{_topology_help()}")
        if trajectory and not _can_be_trajectory(trajectory):
            raise SessionError(
                f"「{os.path.basename(trajectory)}」不能用作轨迹"
                f"（后缀 {_suffix_of(trajectory)}）。\n"
                f"可用轨迹格式：{_trajectory_help()}")
        try:
            mdt = load_trajectory(topology, trajectory, guess_bonds=guess_bonds,
                                  velocities=velocities,
                                  **(dict() if autodetect is None
                                     else {"autodetect": bool(autodetect)}))
        except TrajectoryError as exc:
            raise SessionError(str(exc)) from exc

        sid = uuid.uuid4().hex[:12]
        stem = os.path.splitext(os.path.basename(topology))[0]
        outdir = os.path.join(self.base_outdir, f"{stem}_{sid}")
        os.makedirs(outdir, exist_ok=True)
        if prog is not None:
            set_progress_sink(prog)          # 打开轨迹与组分识别都要上报
        # 注意：load_trajectory 是**惰性**的，真正的读取发生在 auto_setup() 里
        # （它第一次访问 .universe），所以"读取"相关的阶段由 io 层上报、
        # "识别组分"必须在 auto_setup 之后才报，否则顺序会前后颠倒。
        _progress("打开体系")
        # `guess_bonds` 必须传给 Analyzer（它会**重新加载**拓扑）：否则校验用的
        # load_trajectory 带了参数、真正分析的这一份却没带，等于开关失效。
        az = Analyzer(topology, trajectory, guess_bonds=guess_bonds,
                      velocities=velocities, autodetect=autodetect)
        az.auto_setup()
        _progress("识别组分与主链")
        # 体系信息在这里就算好（并缓存）：它的耗时全部来自"重复扫盘"，
        # 修掉之后只要 ~0.3 s。提前算掉有两个好处：
        #   1) 「完成」真的是最后一步，POST 返回时页面能立刻拿到全部信息；
        #   2) 万一将来某个体系的信息真的变慢，它也会出现在进度条里，
        #      而不是像以前那样在「完成」之后静默 50 s（用户只能看到卡住）。
        _progress("统计体系信息")
        _ = az.info          # 预热并缓存体系信息（耗时全在重复扫盘上，现已消除）
        _progress("完成")
        if prog is not None:
            prog.done = True
            prog.sid = sid          # 刷新后前端据此接回会话
        sess = Session(sid=sid, az=az, outdir=outdir)

        with self._lock:
            self._evict_locked()
            self._items[sid] = sess
        return sess

    # ------------------------------------------------------------------ 取用
    def get(self, sid: str) -> Session:
        with self._lock:
            sess = self._items.get(sid)
        if sess is None:
            raise KeyError(sid)
        sess.touch()
        return sess

    def close(self, sid: str) -> None:
        with self._lock:
            sess = self._items.pop(sid, None)
        if sess is not None:
            # 若还有后台分析在跑，先请求取消：否则线程会继续占用 Universe，
            # 还会在结果集上写入已经被丢弃的会话数据
            job = sess.job
            if job is not None and job.status == "running":
                job.cancel.set()
                th = job.thread
                if th is not None:
                    th.join(timeout=5.0)
            shutil.rmtree(sess.outdir, ignore_errors=True)

    def list(self) -> list[dict]:
        with self._lock:
            return [{"sid": s.sid, "topology": s.az.trajectory.topology,
                     "trajectory": s.az.trajectory.trajectory,
                     "n_results": len(s.results_json),
                     "age_sec": round(time.time() - s.created, 1)}
                    for s in self._items.values()]

    def _evict_locked(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._items.items() if now - v.last_used > self.ttl]:
            self._items.pop(sid, None)
        while len(self._items) >= self.max_sessions:
            self._items.popitem(last=False)


def list_dir(path: str | None) -> dict:
    """文件选择器：列出目录下的子目录与 MD 相关文件。"""
    import string

    if not path:
        path = os.getcwd()
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise SessionError(f"不是有效目录: {path}")

    dirs, files = [], []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                dirs.append({"name": name, "path": full})
            elif name.lower().endswith(MD_EXT):
                files.append({
                    "name": name,
                    "path": full,
                    "size": os.path.getsize(full),
                    # kind 只有 topology / trajectory（老前端也认）；
                    # roles 是新增字段，列出"这个文件能放哪几个槽"
                    "kind": file_kind(name),
                    "roles": file_roles(name),
                    # note 也是新增字段：内容是 LAMMPS 输入脚本/日志、
                    # 或是本机读不了的格式时，提前在列表里标出来
                    "note": file_note(full),
                })
    except PermissionError as exc:
        raise SessionError(f"没有权限读取目录: {path}") from exc

    parent = os.path.dirname(path)
    drives = []
    if os.name == "nt":
        drives = [f"{d}:\\" for d in string.ascii_uppercase
                  if os.path.isdir(f"{d}:\\")]
    return {"dir": path, "parent": parent if parent != path else None,
            "dirs": dirs, "files": files, "drives": drives,
            # 界面提示/错误信息共用同一份后缀说明（来自 mdta.formats）
            "help": {"topology": _topology_help(), "trajectory": _trajectory_help()}}


__all__ = ["Session", "SessionRegistry", "SessionError", "list_dir",
           "file_kind", "file_roles", "supported_formats",
           "ANALYSIS_TITLES", "DEFAULT_ORDER", "__version__"]
