# -*- coding: utf-8 -*-
"""Web 会话管理：一个"已打开的体系"对应一个 :class:`Session`。

会话持有 :class:`mdta.pipeline.Analyzer`（也就是已打开的 MDAnalysis Universe），
分析结果与导出目录都挂在会话上；前端用 ``sid`` 引用它。
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from mdta import __version__
from mdta.io import TrajectoryError, load_trajectory
from mdta.pipeline import ANALYSIS_TITLES, DEFAULT_ORDER, Analyzer
from mdta.selection import chains as _chains
from mdta.selection import list_chains, select
from mdta.systeminfo import info_tables

from . import serialize

#: 支持的扩展名（用于文件选择器过滤）
TOPOLOGY_EXT = (".tpr", ".gro", ".pdb", ".psf", ".prmtop", ".parm7", ".top", ".xyz")
TRAJECTORY_EXT = (".xtc", ".trr", ".dcd", ".nc", ".trj", ".lammpstrj")
MD_EXT = TOPOLOGY_EXT + TRAJECTORY_EXT


class SessionError(RuntimeError):
    """会话相关的错误（会被 API 层转成 400/404）。"""


@dataclass
class RunJob:
    """一次"实时运行"的状态。

    后台线程逐项算分析，每算完一项就更新这里；前端轮询
    ``GET /run/progress?after=N`` 只取自己还没有的结果，从而边算边显示。
    """

    order: list[str] = field(default_factory=list)
    #: 已完成的**分析名，按完成顺序**——``after`` 索引就是它的下标
    completion: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "running"            # running | done | error | cancelled
    frac: float = 0.0
    message: str = "准备中…"
    current: str = ""
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
            "sections": info_tables(info, max_chain_types=25),
            "text": self.az.info_text(max_chain_types=25),
        }

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

    def components(self) -> list[dict]:
        out = []
        for name, ag in self.az.components.items():
            out.append({
                "name": name,
                "n_atoms": int(ag.n_atoms),
                "n_residues": int(ag.residues.n_residues),
            })
        return out

    # ----------------------------------------------------------------- 选择
    def apply_selection(self, req: Mapping[str, Any] | None) -> dict:
        """按前端请求设置主链与组分。"""
        # 分析进行中时不许改选择：会动到同一个 Universe
        self.assert_idle()
        req = dict(req or {})
        az = self.az
        az.auto_setup()

        primary = req.get("primary") or {"mode": "auto"}
        mode = primary.get("mode", "auto")
        if mode == "custom" and primary.get("query"):
            q = str(primary["query"])
            ag = select(az.universe, query=q)
            if ag.n_atoms == 0:
                raise SessionError(f"主链选择语句没有选中任何原子: {q!r}")
            az.set_primary(ag, f"自定义: {q}")
        elif mode == "component" and primary.get("name"):
            name = str(primary["name"])
            if name not in az.components:
                raise SessionError(f"不存在组分 {name!r}")
            az.set_primary(az.components[name], f"组分 {name}")
        elif mode == "chain" and primary.get("segid"):
            got = _chains(az.universe, segid=str(primary["segid"]),
                          resname=primary.get("resname"),
                          min_atoms=int(primary.get("min_atoms", 1)))
            if not got:
                raise SessionError("按 segid/resname 没有找到任何链")
            az.set_primary(got[0], f"{primary.get('segid')}:{primary.get('resname')}")

        wanted = req.get("components")
        if wanted:
            chosen = OrderedDict()
            for n in wanted:
                if n in az.components:
                    chosen[n] = az.components[n]
                else:
                    ag = select(az.universe, query=str(n))
                    if ag.n_atoms == 0:
                        raise SessionError(f"组分选择语句没有选中任何原子: {n!r}")
                    chosen[str(n)] = ag
            if chosen:
                az.set_components(chosen)

        return {
            "primary_label": az.primary_label,
            "primary_atoms": int(az.primary.n_atoms) if az.primary is not None else 0,
            "components": self.components(),
        }

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

        which = list(req.get("which") or DEFAULT_ORDER)
        bad = [w for w in which if w not in DEFAULT_ORDER]
        if bad:
            raise SessionError(f"未知分析项 {bad}；可用: {list(DEFAULT_ORDER)}")

        params = {k: dict(v) for k, v in (req.get("params") or {}).items()
                  if isinstance(v, Mapping)}
        return {"sel_info": sel_info, "selection": selection,
                "which": which, "params": params}

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
            return self._payload(prep)

    # ------------------------------------------------------- 实时（流式）运行
    @property
    def job(self) -> "RunJob | None":
        return self._job

    def assert_idle(self) -> None:
        """正在跑分析时，禁止别的接口去动同一个 Universe。"""
        job = self._job
        if job is not None and job.status == "running":
            raise SessionError("分析正在运行中，请先等待完成或点「取消」")

    def start_run(self, req: Mapping[str, Any]) -> dict:
        """**立即返回**并在线程里跑分析；前端轮询 :meth:`run_progress` 拿增量结果。"""
        with self._lock:
            self.assert_idle()
            prep = self._prepare_run(req)
            job = RunJob(order=list(prep["which"]))
            # 新一轮开始：清空上一轮，避免旧结果与新选择混在一起
            self.results_json = OrderedDict()
            self.az.results = OrderedDict()
            self.az.timings = {}
            self._job = job
        th = threading.Thread(target=self._run_worker, args=(prep,),
                              name=f"mdta-run-{self.sid}", daemon=True)
        job.thread = th
        th.start()
        return self.run_progress(0)

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

            def _on_result(name, res, elapsed, done, total):
                payload = serialize.result_to_json(res)
                with self._lock:
                    self.results_json[name] = payload
                    job.timings[name] = round(float(elapsed), 2)
                    job.completion.append(name)
                    job.frac = max(job.frac, done / max(total, 1))
                    job.message = f"已完成 {done}/{total}：{ANALYSIS_TITLES.get(name, name)}"
                    job.last_result = name

            self.az.run_all(prep["which"], params=prep["params"], outdir=None,
                            verbose=False, raise_errors=False,
                            progress=_progress, on_result=_on_result,
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
                job.message = f"完成：{len(job.completion)} 项分析"
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
                        "elapsed_sec": 0.0, "error": ""}
            after = max(0, int(after))
            fresh = job.completion[after:]
            results = OrderedDict(
                (n, self.results_json[n]) for n in fresh if n in self.results_json)
            elapsed = (job.finished or time.time()) - job.started
            return {
                "status": job.status,
                "frac": round(float(job.frac), 4),
                "message": job.message,
                "current": job.current,
                "after": after,
                "n_total": len(job.order),
                "n_done": len(job.completion),
                "order": list(job.order),
                "completion": list(job.completion),
                "pending": [n for n in job.order if n not in job.completion],
                "results": results,
                "summary_rows": serialize.summary_rows(self.results_json),
                "titles": {k: ANALYSIS_TITLES.get(k, k) for k in DEFAULT_ORDER},
                "timings": dict(job.timings),
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
        self.last_export = {
            "dir": out["dir"],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_csv": len(out["csv"]),
            "n_png": len(out["png"]) + len(out.get("panel_png", [])),
            "excel": out["excel"],
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

    # ------------------------------------------------------------------ 建立
    def open(self, topology: str, trajectory: str | None = None) -> Session:
        if not topology:
            raise SessionError("必须提供拓扑文件（.tpr/.gro/.pdb）")
        if not os.path.isfile(topology):
            raise SessionError(f"找不到拓扑文件: {topology}")
        if trajectory and not os.path.isfile(trajectory):
            raise SessionError(f"找不到轨迹文件: {trajectory}")
        try:
            mdt = load_trajectory(topology, trajectory)
        except TrajectoryError as exc:
            raise SessionError(str(exc)) from exc

        sid = uuid.uuid4().hex[:12]
        stem = os.path.splitext(os.path.basename(topology))[0]
        outdir = os.path.join(self.base_outdir, f"{stem}_{sid}")
        os.makedirs(outdir, exist_ok=True)
        az = Analyzer(topology, trajectory)
        az.auto_setup()
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
                    "kind": ("trajectory"
                             if name.lower().endswith(TRAJECTORY_EXT) else "topology"),
                })
    except PermissionError as exc:
        raise SessionError(f"没有权限读取目录: {path}") from exc

    parent = os.path.dirname(path)
    drives = []
    if os.name == "nt":
        drives = [f"{d}:\\" for d in string.ascii_uppercase
                  if os.path.isdir(f"{d}:\\")]
    return {"dir": path, "parent": parent if parent != path else None,
            "dirs": dirs, "files": files, "drives": drives}


__all__ = ["Session", "SessionRegistry", "SessionError", "list_dir",
           "ANALYSIS_TITLES", "DEFAULT_ORDER", "__version__"]
