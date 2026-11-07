# -*- coding: utf-8 -*-
"""文件读取模块。

格式支持面由 :mod:`mdta.formats` 统一定义（**唯一来源**，别在本文件里再写一份）：
GROMACS（`.tpr/.gro/.xtc/.trr/.tng`）、AMBER（`.prmtop/.parm7/.nc/.mdcrd/.crd`）、
CHARMM/NAMD（`.psf/.dcd/.trj/.trz`）、LAMMPS（`.data/.lammps/.lammpstrj/.dump`）、
PDB / XYZ / MOL2 / DMS / H5MD / GSD 等 MDAnalysis 原生支持的格式。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .formats import (
    BOND_GUESSABLE_SUFFIXES as _BOND_GUESSABLE,
    CONVERT_SUFFIXES,
    TOPOLOGY_SUFFIXES,
    TRAJECTORY_SUFFIXES,
    can_be_topology,
    can_be_trajectory,
    format_for,
    is_ms_structure,
    ms_sibling_trajectory,
    suffix_of,
    topology_format_for,
    topology_help,
    trajectory_help,
    unsupported_hint,
)

SUPPORTED_SUFFIXES = set(TOPOLOGY_SUFFIXES) | set(TRAJECTORY_SUFFIXES)

#: GROMACS 的力场拓扑 ``.top``（含 ``#include`` 的 .itp）：MDAnalysis **不能**解析，
#: 给出明确的提示而不是让它用错误的解析器报一堆看不懂的错。
GROMACS_TOP_SUFFIXES = {".top"}



# ------------------------------------------- 只读数据目录的偏移缓存（防死锁）
# MDAnalysis 会在轨迹文件**旁边**写 ``.<名字>_offsets.npz``，并且为了多进程安全
# 先用 ``filelock`` 锁住 ``.<名字>_offsets.lock``。
#
# ⚠️ 当数据目录**不可写**时（只读共享盘、容器挂载、沙箱工作区之外的目录），
# MDAnalysis 本意是回退到"慢速重扫"（``XDR._load_offsets`` 捕获
# ``PermissionError`` 并 warn），但 ``filelock`` 在内部是**无限重试**，
# ``PermissionError`` 根本不会抛出来 —— 于是回退分支永远不触发，表现为
# **加载永久卡死**（实测 >600 s，CPU 仅约 2%，栈停在
# ``filelock/_api.py acquire`` ← ``XDR._load_offsets``）。
#
# 更糟的是：在这种受限环境下，**连"试着写一个临时文件来判断目录可写"都会卡住**
# （实测栈停在 ``tempfile._mkstemp_inner``），所以不能用"先探测再决定"的写法。
# 因此这里的策略是：**默认不在数据目录里写任何东西**，偏移与锁文件一律落到
# 本工具目录下的 ``_offsets_cache/``（不可写则退到系统临时目录）。
# 需要 MDAnalysis 原生"就近缓存"行为时设 ``MDTA_OFFSETS_INPLACE=1``。
_OFFSETS_CACHE_DIR: Path | None = None
_OFFSETS_REDIRECT: dict[str, str] = {}
_OFFSETS_WARNED: set[str] = set()
_OFFSETS_PATCHED = False


def _offsets_cache_dir() -> Path:
    """选一个可写的偏移缓存目录：优先本工具目录，其次系统临时目录。"""
    global _OFFSETS_CACHE_DIR

    if _OFFSETS_CACHE_DIR is not None:
        return _OFFSETS_CACHE_DIR
    for cand in (Path(__file__).resolve().parent.parent / "_offsets_cache",
                 Path(tempfile.gettempdir()) / "mdta_offsets"):
        try:
            cand.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        _OFFSETS_CACHE_DIR = cand
        return cand
    _OFFSETS_CACHE_DIR = Path(tempfile.gettempdir())
    return _OFFSETS_CACHE_DIR


def ensure_offsets_writable() -> Path | None:
    """装上偏移缓存重定向（幂等）。返回缓存目录；已禁用时返回 ``None``。"""
    global _OFFSETS_PATCHED

    if _OFFSETS_PATCHED:
        return _OFFSETS_CACHE_DIR
    if os.environ.get("MDTA_OFFSETS_INPLACE"):
        return None

    import shutil
    import warnings

    from MDAnalysis.coordinates import XDR

    orig = XDR.offsets_filename

    def offsets_filename(filename, ending="npz"):
        src = os.path.abspath(filename)
        inplace = orig(filename, ending)
        # 已有就地缓存就搬过来复用，省掉一次整文件重扫（1.7 GB 轨迹差别很大）。
        # ⚠️ 只在"缓存比轨迹新"时复用：如果轨迹在缓存建立之后被替换/重新转换过，
        # 那份偏移就是**过期**的，复用会让读取器 seek 到帧中间，报
        # `XDR read error = notfound`（用户实测踩到过）。这种情况下宁可重扫一次。
        if ending == "npz" and os.path.isfile(inplace):
            fresh = True
            try:
                if os.path.isfile(filename):
                    fresh = os.path.getmtime(inplace) >= os.path.getmtime(filename)
            except OSError:
                fresh = False
            if fresh:
                h0 = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
                dst = _offsets_cache_dir() / f"{Path(filename).name}.{h0}_offsets.npz"
                if not dst.exists():
                    try:
                        shutil.copy2(inplace, dst)
                    except OSError:
                        pass
        key = f"{src}|{ending}"
        if key not in _OFFSETS_REDIRECT:
            h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
            _OFFSETS_REDIRECT[key] = str(
                _offsets_cache_dir() / f"{Path(filename).name}.{h}_offsets.{ending}")
            '''if src not in _OFFSETS_WARNED:
                _OFFSETS_WARNED.add(src)
                warnings.warn(
                    f"为避免 MDAnalysis 在数据目录不可写时因 filelock 无限重试而卡死，"
                    f"偏移缓存已重定向到 {_offsets_cache_dir()}（数据目录不再被写入）。"
                    f"如需原生就地缓存请设 MDTA_OFFSETS_INPLACE=1。",
                    RuntimeWarning, stacklevel=2)'''
        return _OFFSETS_REDIRECT[key]

    XDR.offsets_filename = offsets_filename
    _OFFSETS_PATCHED = True
    return _OFFSETS_CACHE_DIR


def offsets_redirect_info() -> dict:
    """偏移缓存重定向的诊断信息，便于写进结果 metadata。"""
    return {
        "cache_dir": str(_OFFSETS_CACHE_DIR) if _OFFSETS_CACHE_DIR else None,
        "n_trajectories": len({k.split("|", 1)[0] for k in _OFFSETS_REDIRECT}),
    }


#: 加载进度钩子：``callable(stage: str, **info)``。
#: Web 层在打开会话时挂上它，用来给"文件读取"显示阶段与耗时——大轨迹首次打开
#: 要扫全文件建立帧索引（46 体系 1.7 GB 实测 >135 s），没有反馈时用户会以为卡死。
_PROGRESS_SINK = None


def _needs_frame_scan(trajectory: str | None) -> bool | None:
    """该轨迹是否**还没有**帧偏移缓存（首次打开要扫全文件，慢得多）。

    ``None`` 表示不适用（没有轨迹 / 不是 XDR 格式）。
    """
    if not trajectory:
        return None
    try:
        from MDAnalysis.coordinates import XDR

        off = XDR.offsets_filename(str(trajectory))
        return not os.path.isfile(off)
    except Exception:  # noqa: BLE001
        return None


def set_progress_sink(sink) -> None:
    """挂上/清除加载进度钩子（传 ``None`` 清除）。"""
    global _PROGRESS_SINK
    _PROGRESS_SINK = sink


def _progress(stage: str, **info) -> None:
    sink = _PROGRESS_SINK
    if sink is None:
        return
    try:
        sink(stage, **info)
    except Exception:  # noqa: BLE001 - 进度上报绝不能影响加载
        pass


def mda_universe(path: str):
    """只读打开一个拓扑文件（内部用，便于在错误信息里探测候选文件）。"""
    import MDAnalysis as mda

    ensure_offsets_writable()
    return mda.Universe(str(path))


def count_topology_atoms(path: str) -> int | None:
    """快速读出拓扑文件里的原子数（不依赖 MDAnalysis 能否完整解析）。

    ``.gro`` 的原子数在**第 2 行**（第 1 行是标题），``.pdb`` 数
    ``ATOM``/``HETATM`` 行 —— 都很快，其余格式交给 MDAnalysis。
    """
    suf = Path(path).suffix.lower()
    try:
        if suf == ".gro":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.readline()                      # 第 1 行是标题
                return int(f.readline().strip() or 0)
        if suf in (".pdb", ".ent"):
            n = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        n += 1
            return n
    except Exception:  # noqa: BLE001
        return None
    try:
        return int(mda_universe(path).atoms.n_atoms)
    except Exception:  # noqa: BLE001
        return None


def count_trajectory_atoms(path: str) -> int | None:
    """从轨迹文件头里读出原子数。"""
    import MDAnalysis as mda

    try:
        if Path(path).suffix.lower() == ".xtc":
            r = mda.coordinates.XTC.XTCReader(path, convert_units=False)
            n = int(r.n_atoms)
            r.close()
            return n
    except Exception:  # noqa: BLE001
        pass
    return None


class TrajectoryError(RuntimeError):
    """轨迹读取相关的错误。"""


#: 这些格式**无法存储键连接**，加载后若发现没有键，就按坐标猜一遍
#: （见 :mod:`mdta.topology`）。清单来自 :mod:`mdta.formats`（唯一来源）。
BOND_GUESSABLE_SUFFIXES = _BOND_GUESSABLE
#: 超过这个原子数就不自动猜键了（猜键是 O(N) 但常数不小）
AUTO_GUESS_MAX_ATOMS = 300_000


@dataclass
class MDTrajectory:
    """轨迹容器：封装 ``MDAnalysis.Universe`` 并补充常用元信息。

    参数
    ----
    topology
        拓扑文件（``.tpr`` / ``.gro`` / ``.pdb`` ...）。
    trajectory
        轨迹文件（``.xtc`` / ``.trr`` / ``.dcd`` ...）。为 ``None`` 时只读单帧结构。
    guess_bonds
        没有键连接信息时，是否按共价半径从坐标猜键。
        ``None``（默认）表示**自动**：``.gro`` / 无 ``CONECT`` 的 ``.pdb`` / ``.xyz``
        这类格式猜，``.tpr`` / ``.psf`` / ``.prmtop`` 本身带键表就不猜。
    """

    topology: str | None = None
    trajectory: str | None = None
    _universe: object | None = None
    guess_bonds: bool | None = None
    #: 读 ``.trj`` 时是否**连带保存速度/力**：``True`` 就转成 TRR（坐标+速度，
    #: 文件约为 XTC 的 5~6 倍），``False``/``None`` 只转坐标（XTC，最小）。
    #: 速度是 MS ``.trj`` 里本来就有的（第 4-6 条 double 数组），XTC 装不下。
    velocities: bool | None = None
    #: 猜键结果统计（未猜时为 ``None``）
    bond_guess_info: dict | None = None
    #: 对象层属性补齐诊断（elements / masses / molnums）
    object_info: dict | None = None
    #: Materials Studio 文档（``.car`` / ``.xsd`` / ``.xtd``）的解析摘要
    ms_info: dict | None = None
    #: 读取过程中值得告诉用户的话（自动配对轨迹、MS 单帧说明…）
    load_notes: list[str] = field(default_factory=list)
    #: 是否因偏移缓存过期而重建过（读取被替换/重转换的轨迹时会置位）
    offsets_rebuilt: bool = False
    #: 每帧时间的**惰性缓存**（见 :attr:`times_ps`）
    _times_cache: np.ndarray | None = field(default=None, repr=False, compare=False)
    #: 首帧/末帧时间的**惰性缓存**（见 :meth:`time_range_ps`）
    _time_range_cache: tuple[float, float] | None = field(
        default=None, repr=False, compare=False)

    # -------------------------------------------------------------- 构造
    def __post_init__(self) -> None:
        if self.topology is not None:
            self.topology = str(Path(str(self.topology)).expanduser())
        if self.trajectory is not None:
            self.trajectory = str(Path(self.trajectory).expanduser())

    @classmethod
    def from_universe(cls, universe, label: str = "<内存中的体系>") -> "MDTrajectory":
        """直接用一个已有的 ``MDAnalysis.Universe`` 构造（便于测试与二次开发）。"""
        obj = cls(topology=None, trajectory=None)
        obj._universe = universe
        obj.topology = label
        return obj

    def load(self):
        """打开（或返回已打开的）Universe。"""
        if self._universe is None:
            import MDAnalysis as mda

            if not self.topology:
                raise TrajectoryError("没有指定拓扑文件。")
            if not os.path.isfile(self.topology):
                raise TrajectoryError(f"找不到拓扑文件: {self.topology}")
            # Materials Studio 文档（.xsd/.xtd）常常**与轨迹同名不同后缀**：
            # 只给了一个时自动把另一个配上（.xtd 里写着 <Trajectory End=…/>，
            # 真正的坐标在同名 .trj 里；.trj 再由 ms_trj 转成 XTC 读取）。
            if self.trajectory is None and is_ms_structure(self.topology):
                sib = ms_sibling_trajectory(self.topology)
                if sib:
                    self.trajectory = sib
                    self.load_notes.append(
                        f"已自动配对同名轨迹：{os.path.basename(sib)}"
                        f"（与拓扑 {os.path.basename(self.topology)} 同名前缀）")
            if self.trajectory is not None and not os.path.isfile(self.trajectory):
                raise TrajectoryError(f"找不到轨迹文件: {self.trajectory}")
            if Path(self.topology).suffix.lower() in GROMACS_TOP_SUFFIXES:
                raise TrajectoryError(self._gromacs_top_message())
            # 槽位校验：把"轨迹文件当拓扑用""不认识的后缀"这类误操作挡在解析之前，
            # 给一句人能看懂的话（以前会一路抛到 API 变成 HTTP 500，
            # 实测 `.xtc` 当拓扑 / `.md` 当拓扑都会 500）。
            if not can_be_topology(self.topology):
                raise TrajectoryError(self._bad_topology_message())
            if self.trajectory and not can_be_trajectory(self.trajectory):
                raise TrajectoryError(self._bad_trajectory_message())
            # MS 的 .trj 头部写着原子数：**转换之前**先跟拓扑对一下。
            # 否则 `.car`(5400 原子) + 别的体系的 `.trj`(5596 原子 / 约 1.6 GB) 会先
            # 把整个轨迹转完，最后抛一句"转换失败 / .trj 被截断"（实测用户踩到）。
            self._precheck_ms_trj_atoms()
            # `.lammps` / `.lmp` / `.dat` 可能是**输入脚本或日志**（`in.lammps`、`log.lammps`），
            # 丢给 DATA 解析器只会报 EOFError；这里先闻一下内容。
            self._check_lammps_text_file()
            _progress("读取拓扑与轨迹", topology=os.path.basename(self.topology),
                      trajectory=(os.path.basename(self.trajectory)
                                  if self.trajectory else None),
                      trajectory_mb=(round(os.path.getsize(self.trajectory) / 1e6, 1)
                                     if self.trajectory
                                     and os.path.isfile(self.trajectory) else None),
                      first_scan=_needs_frame_scan(self.trajectory))
            # `.trj` 可能是 Materials Studio 的二进制轨迹（MDAnalysis 读不了）：
            # 先按下标让 MDAnalysis 试（CHARMM ASCII 的 .trj 它能读），失败就调用
            # 用户的 pymsdump/ms_trj 转成 XTC 再读（见 mdta.ms_trj）。
            load_traj = self.trajectory
            try:
                ensure_offsets_writable()   # 数据目录只读时重定向偏移缓存，避免死锁
                # `.lammpstrj` / `.dump` 这类"认格式名不认后缀"的格式必须显式传
                # format（见 mdta.formats.EXPLICIT_FORMAT），否则按后缀找不到读取器。
                kw = {}
                fmt = format_for(load_traj or self.topology or "")
                if fmt:
                    kw["format"] = fmt
                try:
                    self._universe = self._open_universe(mda, kw, traj=load_traj)
                except Exception as exc:  # noqa: BLE001 - 先判断是否偏移缓存问题
                    if not (load_traj and self._looks_like_offsets_error(exc)):
                        raise
                    # 自愈：轨迹被替换/重新转换过时，旧偏移缓存会让"定位帧"失败
                    # （实测用户遇到 `XDR read error = notfound`）。这里重建一次缓存，
                    # 成功就继续并记一条说明，失败则把原异常抛出去做可读诊断。
                    self._universe = self._open_universe(mda, kw, refresh=True,
                                                         traj=load_traj)
                    self.offsets_rebuilt = True
            except Exception as first_exc:  # noqa: BLE001
                # 第二次机会：`.trj` 走"转换后再读"
                if (self.trajectory
                        and suffix_of(self.trajectory) in CONVERT_SUFFIXES):
                    try:
                        load_traj = self._convert_trajectory()
                        kw2 = {}
                        self._universe = self._open_universe(mda, kw2, traj=load_traj)
                    except Exception as conv_exc:  # noqa: BLE001
                        raise TrajectoryError(
                            self._read_error_message(conv_exc, converted=True)
                        ) from conv_exc
                else:
                    n_top, n_traj = self._read_counts()
                    if n_top and n_traj and n_top != n_traj:
                        raise TrajectoryError(
                            self._mismatch_message(n_top, n_traj)) from first_exc
                    raise TrajectoryError(
                        self._read_error_message(first_exc)) from first_exc
            _progress("解析拓扑与帧数")
            self._validate()
            self._maybe_guess_bonds()
            _progress("按坐标补全键表",
                      bonds=(self.bond_guess_info or {}).get("bonds"),
                      fragments=(self.bond_guess_info or {}).get("fragments"),
                      skipped=(self.bond_guess_info or {}).get("skipped"))
            self._ensure_object_attrs()
            _progress("补齐对象属性（元素/质量/分子）",
                      **(self.object_info or {}))
            _progress("轨迹已就绪")
        return self._universe

    def _read_counts(self) -> tuple[int | None, int | None]:
        n_top = count_topology_atoms(self.topology) if self.topology else None
        n_traj = (count_trajectory_atoms(self.trajectory)
                  if self.trajectory else None)
        return n_top, n_traj

    #: 这些错误字样说明"读取器定位帧失败"，最典型的原因是**偏移缓存过期**
    #: （轨迹在缓存建立之后被替换/重新转换过，或转换写到一半就被读）
    _OFFSET_HINTS = ("xdr", "notfound", "offsets", "seek", "unexpected end",
                     "error reading", "bad magic")

    def _looks_like_offsets_error(self, exc: BaseException) -> bool:
        msg = f"{type(exc).__name__}: {exc}".lower()
        return any(h in msg for h in self._OFFSET_HINTS)

    def _check_lammps_text_file(self) -> None:
        """`.lammps` / `.lmp` / `.dat` 若是**输入脚本或日志**，给出针对性报错。

        真实情况：LAMMPS 项目里 `in.lammps`（输入脚本）、`log.lammps`（日志）非常常见，
        而它们与 data 文件共用同一批后缀。直接丢给 DATA 解析器，用户看到的是
        ``EOFError: Compressed file ended before the end-of-stream marker`` 这种
        完全不相关的话（实测）。
        """
        from .formats import sniff_lammps_kind

        for path, role in ((self.topology, "拓扑"), (self.trajectory, "轨迹")):
            if not path or not os.path.isfile(path):
                continue
            kind = sniff_lammps_kind(path)
            if kind == "input":
                raise TrajectoryError(
                    f"「{os.path.basename(path)}」是 LAMMPS 的**输入脚本**，不是结构/"
                    f"轨迹文件，不能放在「{role}」槽里。\n"
                    f"  脚本里 `read_data …` 指到的 data 文件（`.lmp`/`.data`）才是拓扑，"
                    f"`dump … custom` 输出的 `.lammpstrj` 才是轨迹。")
            if kind == "log":
                raise TrajectoryError(
                    f"「{os.path.basename(path)}」是 LAMMPS 的**日志/热力学输出**，"
                    f"不含坐标，不能放在「{role}」槽里。\n"
                    f"  坐标请看 `dump` 输出的轨迹（`.lammpstrj`），"
                    f"原子身份请看 `write_data` 的 data 文件（`.lmp`/`.data`）。")

    def _precheck_ms_trj_atoms(self) -> None:
        """``.trj`` 的头部原子数 vs 拓扑原子数 —— **转换之前**先比一次。

        真实故障：用户把 `C12-100-2.car`（5400 原子）和一个 5596 原子的
        `122C12-450K.trj` 配在一起。旧行为是"先转成 XTC 再让 MDAnalysis 报
        topology/轨迹原子数不一致"，而错误信息被包成
        「.trj 被截断 / 模块放错位置」—— 与事实无关，而且为一个 1.6 GB 的轨迹
        白转几分钟（C 盘只剩 0 GB 时还会把缓存写坏）。现在这里直接给结论。
        """
        if not self.trajectory or suffix_of(self.trajectory) != ".trj":
            return
        try:
            from .ms_trj import trj_header

            hdr = trj_header(self.trajectory)
        except Exception:                                        # noqa: BLE001
            return
        n_trj = hdr.get("n_atoms")
        if not n_trj:
            return
        n_top = None
        try:
            if is_ms_structure(self.topology or ""):
                from .ms_structure import read_ms_structure

                n_top = int(read_ms_structure(self.topology).get("n_atoms") or 0) or None
            else:
                n_top = count_topology_atoms(self.topology)
        except Exception:                                        # noqa: BLE001
            n_top = None
        if not n_top or int(n_top) == int(n_trj):
            self._trj_n_atoms = int(n_trj)
            return
        raise TrajectoryError(
            f"拓扑与轨迹的**原子数不一致**，这两个文件不是同一个体系：\n"
            f"  · 拓扑 {os.path.basename(self.topology)}：{int(n_top):,} 原子\n"
            f"  · 轨迹 {os.path.basename(self.trajectory)}：{int(n_trj):,} 原子"
            f"（读自 .trj 头部，尚未转换）\n"
            f"Materials Studio 的 .trj 必须与**导出它的那份结构**配套："
            f".car / .xsd / .xtd 与 .trj 的原子数与原子顺序都要一一对应。\n"
            f"请检查是不是把不同次导出（或不同体系）的拓扑与轨迹配在了一起 —— "
            f"文件名相近不代表同一体系。")

    def _open_universe(self, mda, kw: dict, refresh: bool = False,
                       traj: str | None = None) -> object:
        """打开 Universe；``refresh=True`` 时让 XDR 读取器**重建偏移缓存**。"""
        k = dict(kw)
        t = self.trajectory if traj is None else traj
        if refresh and suffix_of(t or "") in (".xtc", ".trr"):
            k["refresh_offsets"] = True
        # LAMMPS data（`.lmp`/`.dat`/`.lammps`）按后缀找不到拓扑解析器，必须显式
        # 给 `topology_format="DATA"`；轨迹一侧的 `format` 由 `format_for()` 单独给。
        tfmt = topology_format_for(self.topology or "")
        if tfmt and "topology_format" not in k:
            k["topology_format"] = tfmt
        if is_ms_structure(self.topology or ""):
            return self._open_ms_universe(t, k)
        if t:
            return mda.Universe(self.topology, t, **k)
        return mda.Universe(self.topology, **k)

    def _open_ms_universe(self, traj: str | None, kw: dict):
        """用 :mod:`mdta.ms_structure` 解析 Materials Studio 拓扑，再挂轨迹。

        MDAnalysis 不认识 ``.car`` / ``.xsd`` / ``.xtd``，所以拓扑这一侧自己解析
        （原子/元素/键表/晶胞），然后按普通方式把轨迹挂上去：

        * ``.xsd`` 带**完整键表**（实测 5596 原子 / 5891 键），分子划分直接可用；
        * ``.car`` 通常没有键表 → 后面由 `_maybe_guess_bonds` 按坐标猜；
        * ``.xtd`` 只有拓扑 + ``<Trajectory End="6001">`` 声明，坐标在同名 ``.trj`` 里
          —— 没显式给轨迹时自动配上（见 :func:`mdta.formats.ms_sibling_trajectory`）。
        """
        from .ms_structure import build_universe, read_ms_structure

        _progress("解析 Materials Studio 拓扑",
                  topology=os.path.basename(self.topology or ""))
        try:
            doc = read_ms_structure(self.topology)
        except Exception as exc:                                # noqa: BLE001
            raise TrajectoryError(
                f"读取失败：{os.path.basename(self.topology or '')} 不是可解析的 "
                f"Materials Studio 文件（{type(exc).__name__}: {exc}）") from exc
        u = build_universe(doc)
        self.ms_info = {
            "format": doc.get("format"),
            "n_atoms": int(doc.get("n_atoms") or 0),
            "n_bonds": len(doc.get("bonds") or []),
            "n_images_skipped": int(doc.get("n_images_skipped") or 0),
            "trajectory_hint": doc.get("trajectory_hint"),
            "cell": (None if doc.get("cell") is None
                     else [float(x) for x in doc["cell"]]),
        }
        _progress("Materials Studio 拓扑已解析",
                  atoms=self.ms_info["n_atoms"], bonds=self.ms_info["n_bonds"])
        if traj:
            u.load_new(traj, **kw)
            hint = self.ms_info.get("trajectory_hint") or {}
            n_decl = hint.get("n_frames")
            try:
                got = len(u.trajectory)
            except Exception:                                   # noqa: BLE001
                got = None
            if n_decl and got is not None:
                if int(n_decl) != int(got):
                    self.load_notes.append(
                        f"⚠️ 轨迹帧数与 MS 文档声明不符：文档写的是 {int(n_decl):,} 帧"
                        f"（<Trajectory End=\"{int(n_decl)}\" "
                        f"Type=\"{hint.get('type')}\"/>），实际读到 {int(got):,} 帧"
                        f"—— 轨迹文件可能被截断，或与这份拓扑不是同一次计算。")
                else:
                    self.load_notes.append(
                        f"轨迹帧数校验通过：MS 文档声明 {int(n_decl):,} 帧，"
                        f"实际读到 {int(got):,} 帧")
        elif self.ms_info["trajectory_hint"]:
            self.load_notes.append(
                f"{os.path.basename(self.topology or '')} 声明了 "
                f"{self.ms_info['trajectory_hint'].get('n_frames')} 帧轨迹"
                f"（<Trajectory Type=\"{self.ms_info['trajectory_hint'].get('type')}\">），"
                f"但坐标不在这个文件里 —— 同名的 .trj 才是轨迹；"
                f"没找到同名轨迹文件时只能按单帧分析。")
        return u

    def _convert_trajectory(self) -> str:
        """把需要转换的轨迹（Materials Studio ``.trj``）转成 XTC / TRR，返回新路径。

        转换器来自本仓库的 ``pymsdump``（纯 Python，无需编译），见 :mod:`mdta.ms_trj`；
        结果按「路径+大小+修改时间+格式+精度」缓存，重复打开秒开。

        ``self.velocities=True`` 时转成 **TRR**（坐标 + 速度），否则只转坐标（XTC）。
        精度：XTC 默认 1e4（0.001 Å 量化步长，单点误差 ≤5e-4 Å），可用环境变量
        ``MDTA_XTC_PRECISION`` 调整。
        """
        from . import ms_trj as _ms

        if not _ms.available(self.trajectory):
            raise TrajectoryError(_ms.unavailable_message(self.trajectory))
        fmt = "trr" if self.velocities else "xtc"
        what = "TRR（坐标 + 速度）" if fmt == "trr" else "XTC"
        info = _ms.trj_info(self.trajectory)
        _progress(f"转换 Materials Studio .trj → {what}",
                  frames=info.get("n_frames"), atoms=info.get("n_atoms"))
        out = _ms.convert_trj(
            self.trajectory, fmt=fmt,
            progress=lambda msg: _progress(f"转换 Materials Studio .trj → {what}",
                                           note=msg))
        self.converted_xtc = str(out)
        prec = _ms.xc_precision()
        if fmt == "trr":
            self.load_notes.append(
                f"轨迹 {os.path.basename(self.trajectory)} 是 Materials Studio 的 .trj，"
                f"已转换成 **TRR**（坐标 + 速度）后读取：坐标 float32（误差 ~1e-6 Å）、"
                f"速度单位 Å/ps，可在分析里直接用（缓存：{out.name}，"
                f"{out.stat().st_size / 1e6:.1f} MB）。")
        else:
            self.load_notes.append(
                f"轨迹 {os.path.basename(self.trajectory)} 是 Materials Studio 的 .trj，"
                f"MDAnalysis 读不了；已用 pymsdump/ms_trj 转成 XTC 后读取"
                f"（缓存：{out.name}；坐标量化步长 1e-3 Å，单点误差 ≤5e-4 Å —— "
                f"官方 Trj2Ascii 导出的是完整 double，这点差别就是量化本身）。"
                f"需要速度数据请用 TRR 模式。")
        self.converted_note = self.load_notes[-1]
        return str(out)

    def _read_error_message(self, exc: BaseException, *,
                            converted: bool = False) -> str:
        """读取失败时的可读诊断：**点名是哪个文件**+可能原因+可操作建议。

        为什么要专门写：用户实测遇到的是
        ``读取失败: 122C12450K.pdb: XDR read error = notfound`` ——
        报的是**拓扑**文件名，实际失败的是**轨迹**（XDR 是轨迹读取器），
        而且没有任何"该怎么办"的信息。这里改成点名 + 给原因 + 给做法。
        """
        traj = os.path.basename(self.trajectory or "")
        top = os.path.basename(self.topology or "")
        offsets_like = self._looks_like_offsets_error(exc)
        who = (f"轨迹文件 {traj}" if (self.trajectory and offsets_like)
               else f"{top}（拓扑）" if not self.trajectory
               else f"{top} + {traj}")
        lines = [f"读取失败（{who}）：{type(exc).__name__}: {exc}"]
        n_mismatch = "number of atoms" in str(exc) or "原子数" in str(exc)
        if n_mismatch and self.trajectory:
            n_top, n_traj = self._atom_counts_for_message()
            lines += [
                "  这是**拓扑与轨迹的原子数不一致** —— 两个文件不是同一个体系"
                "（或不是同一次导出），改名/挪目录都不会让它配套。",
                f"    · 拓扑 {top}：{n_top if n_top is not None else '?'} 原子",
                f"    · 轨迹 {traj}：{n_traj if n_traj is not None else '?'} 原子",
                "  Materials Studio 的 .trj 必须与**导出它的那份结构**配套；"
                "请换成同一批导出的 .car/.xsd/.xtd ↔ .trj 组合。",
            ]
        # LAMMPS 的 `.lmp` / `.dat` / `.data` / `.lammps` 里，只有 `write_data`
        # 导出的才是 data 文件；同名后缀也常被普通结果表/缓存占用（本工具自己的
        # `_msd_scratch/msd_*.dat` 就是二进制缓存）。内容嗅探认不出来时直接说清楚，
        # 而不是把 `gbk codec can't decode` 这种底层错误丢给用户。
        if not converted and not n_mismatch:
            for path in (self.topology, self.trajectory):
                if not path or not os.path.isfile(path):
                    continue
                try:
                    from .formats import LAMMPS_TEXT_SUFFIXES, sniff_lammps_kind

                    if (suffix_of(path) not in LAMMPS_TEXT_SUFFIXES
                            or sniff_lammps_kind(path) is not None):
                        continue
                except Exception:                               # noqa: BLE001
                    continue
                lines += [
                    f"  「{os.path.basename(path)}」的后缀像 LAMMPS data，但前 40 行里"
                    "没有 data 文件该有的表头（`N atoms` / `M atom types` / `Masses` 段），"
                    "所以它**不是** LAMMPS data 文件。",
                    "    · 真正的 data 文件由 LAMMPS 的 `write_data` 导出"
                    "（后缀 `.lmp` / `.data` / `.dat` / `.lammps` 都行）；",
                    "    · LAMMPS 的输入脚本（`in.lammps`）与日志（`log.lammps`）不是数据文件；",
                    "    · 若这个 `.dat` 其实是一列数值结果/缓存表，请换一份真拓扑"
                    "（`.pdb` / `.gro` / `.tpr` / `.xsd` …）。",
                ]
                break

        if converted:
            lines += [
                "  该文件是 `.trj`：MDAnalysis 读不了 Materials Studio 的二进制 .trj，"
                "本工具尝试用 pymsdump/ms_trj 转换后读取，**转换或读取失败了**。",
                "  常见原因：",
                "    · .trj 被截断/拷贝不完整（解析器报「文件长度与帧步长不匹配」）；",
                "    · 该 .trj 不是 Materials Studio 格式（也可能是坏文件）；",
                "    · pymsdump / md_convert 模块放错位置。",
                r"  可用环境变量指定模块目录：set MDTA_MS_CONVERT=D:\md_convert;D:\pymsdump",
            ]
            # 有头部信息就能把"截断"这件事说得很具体（用户实测就是磁盘满写不完）
            try:
                from .ms_trj import trj_header

                hdr = trj_header(self.trajectory or "")
            except Exception:                                   # noqa: BLE001
                hdr = {}
            if hdr.get("n_atoms"):
                got_fr = hdr.get("n_frames_declared")
                size_mb = (os.path.getsize(self.trajectory) / 1048576
                           if self.trajectory and os.path.isfile(self.trajectory) else 0)
                decl = ((self.ms_info or {}).get("trajectory_hint") or {}).get("n_frames")
                lines.append(
                    f"  实测这个 .trj：头部写着 {int(hdr['n_atoms']):,} 原子，"
                    f"按帧长推算**只有 {got_fr:,} 帧是完整的**（文件 {size_mb:.1f} MB）"
                    + (f"，而这套 MS 文档声明了 {int(decl):,} 帧" if decl else "")
                    + " —— 典型是**磁盘写满/拷贝中断**导致的截断。")
                lines.append("  建议：腾出磁盘空间后**重新从 Materials Studio 导出**"
                             "完整轨迹（或直接导出 .xtc），再打开。")
        elif offsets_like and self.trajectory:
            lines += [
                "  这是**轨迹读取/定位帧**失败，最常见的原因：",
                "    · 轨迹文件在「偏移缓存」建立之后被**替换或重新转换**过"
                "（例如 trj→xtc 又跑了一遍、或转换还没写完就开始读）；",
                "    · 文件被截断（下载/拷贝中断），末尾帧不完整；",
                "    · 它其实不是该格式（例如把别的格式改名成 .xtc）。",
                f"  已自动尝试**重建偏移缓存**仍失败；建议：",
                "    · 确认文件大小不再变化后重试（拷贝/转换完成再打开）；",
                "    · 重新转换一次，或改用其它轨迹格式（.trr / .dcd / .nc）；",
                "    · 用 gmx check / gmx trjconv 校验一下这条轨迹。",
            ]
        elif self.trajectory and not n_mismatch:
            lines.append("  如果怀疑是拓扑与轨迹不配套，请确认两者原子数一致"
                         f"（当前：拓扑 {count_topology_atoms(self.topology)}、"
                         f"轨迹 {count_trajectory_atoms(self.trajectory)}）。")
        return "\n".join(lines)

    def _atom_counts_for_message(self) -> tuple[int | None, int | None]:
        """给"原子数不一致"这类报错用的原子数（MS 文档要自己解析才算得出）。"""
        n_top = n_traj = None
        try:
            if is_ms_structure(self.topology or ""):
                from .ms_structure import read_ms_structure

                n_top = int(read_ms_structure(self.topology).get("n_atoms") or 0) or None
            else:
                n_top = count_topology_atoms(self.topology)
        except Exception:                                        # noqa: BLE001
            n_top = None
        try:
            n_traj = count_trajectory_atoms(self.trajectory)
        except Exception:                                        # noqa: BLE001
            n_traj = None
        if n_traj is None and self.trajectory:
            try:
                from .ms_trj import trj_header

                n_traj = trj_header(self.trajectory).get("n_atoms")
            except Exception:                                    # noqa: BLE001
                n_traj = None
        return n_top, n_traj

    def _bad_topology_message(self) -> str:
        """选错槽位/后缀时的提示：说清"这个文件不能当拓扑"+ 该选什么。"""
        top = os.path.basename(self.topology or "")
        suf = suffix_of(self.topology or "") or "（无后缀）"
        lines = [f"「{top}」不能用作拓扑文件（后缀 {suf}）。"]
        hint = unsupported_hint(self.topology or "")
        if can_be_trajectory(self.topology or ""):
            lines += [
                "  它看起来是**轨迹**格式（只有坐标，没有原子名/元素/键表）。"
                "把轨迹当拓扑读进来，质量与键表都会是错的（QC 会标「不可引用」）。",
            ]
            if suf in (".lammpstrj", ".dump", ".lammpsdump"):
                lines.append(
                    "  LAMMPS 的 dump 只记录原子**类型号**，没有元素/质量 —— "
                    "请把同体系的 **data 文件**（`.lmp` / `.dat` / `.data`）放进「拓扑」槽，"
                    "dump 放进「轨迹」槽（两个文件同名时会自动配对）。")
            else:
                lines.append("  请把它在「轨迹」槽里选择，并在「拓扑」槽里选同体系的"
                             " .pdb / .gro / .tpr 等。")
        elif hint:
            lines.append("  " + hint)
        else:
            lines.append("  该后缀不在支持列表里。")
        lines.append(f"  可用拓扑格式：{topology_help()}")
        return "\n".join(lines)

    def _bad_trajectory_message(self) -> str:
        traj = os.path.basename(self.trajectory or "")
        suf = suffix_of(self.trajectory or "") or "（无后缀）"
        lines = [f"「{traj}」不能用作轨迹文件（后缀 {suf}）。"]
        hint = unsupported_hint(self.trajectory or "")
        if hint:
            lines.append("  " + hint)
        lines.append(f"  可用轨迹格式：{trajectory_help()}")
        return "\n".join(lines)

    def _gromacs_top_message(self) -> str:
        """``.top`` 是 GROMACS 力场拓扑，MDAnalysis 解析不了——告诉用户该用什么。"""
        lines = [
            f"MDAnalysis 无法解析 GROMACS 力场拓扑文件（{os.path.basename(self.topology or '')}）。",
            "  .top 里是 [atoms]/[bonds]/#include 形式的力场定义，没有坐标，"
            "本工具的分析需要带坐标的拓扑。",
            "  请改用下面任一方式：",
            "    · 同目录/同体系的 .gro 或 .pdb（.gro 最省事，会自动按坐标补键）；",
            "    · 用 GROMACS 生成 .tpr：gmx grompp -f md.mdp -c conf.gro -p topol.top -o topol.tpr",
        ]
        try:
            d = os.path.dirname(os.path.abspath(self.topology)) or "."
            cands = [n for n in sorted(os.listdir(d))
                     if Path(n).suffix.lower() in TOPOLOGY_SUFFIXES
                     and not n.lower().endswith(".top")]
            if cands:
                lines += ["  同目录下这些文件可以用作拓扑："]
                lines += [f"    · {c}" for c in cands[:8]]
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines)

    # ------------------------------------------------------------ 键补全
    def _maybe_guess_bonds(self) -> None:
        """按共价半径补/修键表，把分子与链的划分弄对。

    三种情形：

    1. **拓扑没有键**（``.gro`` / 无 CONECT 的 ``.pdb`` / ``.xyz``）→ 猜键（老行为）；
    2. **拓扑有键但明显不完整**（``CONECT`` 只写了一部分）→ **自动改用猜键结果**
       并写明；实测 Material Studio 导出的 PDB：自带 4393 个键 → 键图碎片 1203 个
       （真实分子约 122 个），而按坐标猜键得到 5954 个键 / 123 个碎片（补键后 122）。
       键表不全会让"逐分子 Rg/R_ee、分子质心 MSD、RDF 的同分子剔除"**静默变差**，
       所以这里不靠用户勾选，直接在**证据充分**时自动替换（判据见下）。
    3. **拓扑有键且完整**（``.tpr``/``.psf``/作者给的 PDB）→ 什么都不做（也不猜）。
    """
        from .topology import guess_bonds as _guess
        from .topology import needs_bond_guessing

        u = self._universe
        has_bonds = False
        try:
            has_bonds = len(u.bonds) > 0
        except Exception:  # noqa: BLE001
            has_bonds = False
        forced = self.guess_bonds is True
        if not forced and not has_bonds and not needs_bond_guessing(u):
            return
        # 用户/旧版界面显式说了"不猜键"：一般照办，但**自带键表明显残缺时仍然修**
        # （实测 MS 的 PDB：polymer[P] 被 CONECT 切成 243 个碎片，其中 121 个是孤立
        #  H —— 逐分子 Rg 被这些"0 Å 的单原子"拉低到 2.19 Å，而真值 4.39 Å）。
        # 不修就得靠用户自己发现，而这条错误**不会报任何异常**。
        repair_broken = False
        degen_note = ""
        if self.guess_bonds is False:
            from .topology import topology_degeneracy

            degen = topology_degeneracy(u.atoms)
            if not degen.get("degenerate"):
                return
            repair_broken = True
            degen_note = str(degen.get("note", ""))
            forced = True                     # 走"替换旧键表"的路径
        suf = Path(self.topology or "").suffix.lower()
        if self.guess_bonds is None and suf not in BOND_GUESSABLE_SUFFIXES:
            return                                   # .tpr 等本应带键，不强猜
        if u.atoms.n_atoms > AUTO_GUESS_MAX_ATOMS:
            self.bond_guess_info = {
                "skipped": f"原子数 {u.atoms.n_atoms:,} 超过自动猜键上限 "
                           f"{AUTO_GUESS_MAX_ATOMS:,}，已跳过",
            }
            return

        n_bonds_before = len(u.bonds) if has_bonds else 0
        n_frag_before = 0
        old_bonds = None
        if has_bonds and not forced:
            try:
                n_frag_before = len(u.atoms.fragments)
                # `u.bonds.indices` 是 (n_bonds, 2)：**逐行**才是一对原子
                # （写成 `for a, b in arr.T` 会试图把长度为 n 的列拆成两个 → ValueError，
                #  于是快照失败、退化成"在旧键表上追加"，实测键数 5953 而不是 5954）
                old_bonds = [(int(a), int(b)) for a, b in
                             np.asarray(u.bonds.indices, dtype=int)]
            except Exception:  # noqa: BLE001
                old_bonds = None
        try:
            # 有旧键表时一律"替换后重猜"（旧键表已快照，判定不通过会还原）
            info = _guess(u, replace=bool(forced or has_bonds))
            info = dict(info or {})
        except Exception as exc:  # noqa: BLE001 - 猜键失败不应阻断加载
            self.bond_guess_info = {"skipped": f"猜键失败: {exc}"}
            return

        if old_bonds is not None:
            # 猜键 vs 自带键表：只有当"自带键表把分子切得更碎"时才采用猜键结果。
            # 判据：碎片数降到自带结果的 60% 以下（说明自带键表漏了成键关系）。
            n_frag_guess = int(info.get("fragments", 0) or 0)
            adopt = bool(n_frag_guess and n_frag_before
                         and n_frag_guess < 0.6 * n_frag_before)
            if not adopt:
                try:
                    u.delete_bonds(u.bonds)
                except Exception:                             # noqa: BLE001
                    u.delete_TopologyAttr("bonds")
                u.add_TopologyAttr("bonds", old_bonds)
                self.bond_guess_info = {
                    "skipped": (f"拓扑自带的键表看起来是完整的"
                                f"（碎片 {n_frag_before:,} vs 按坐标猜键 "
                                f"{n_frag_guess:,}），保留原键表不猜"),
                    "kept_topology_bonds": True,
                    "bonds": n_bonds_before,
                    "fragments": n_frag_before,
                }
                return
            info["note"] = (
                f"拓扑自带的键表**不完整**（碎片 {n_frag_before:,} → 猜键 "
                f"{n_frag_guess:,}，键 {n_bonds_before:,} → {info.get('bonds', 0):,}），"
                f"已自动改用按坐标猜键的结果 —— 逐分子 Rg/R_ee、分子质心 MSD、"
                f"RDF 的同分子剔除都依赖这个划分。")
            info["replaced_existing"] = n_bonds_before
            info["fragments_before"] = n_frag_before
        elif forced:
            info["replaced_existing"] = n_bonds_before
            if repair_broken:
                info["forced_repair"] = True
                info["degenerate_note"] = degen_note
                info["fragments_before"] = int(degen.get("n_frag", 0) or 0)
                info["isolated_frac"] = float(degen.get("isolated_frac", 0.0) or 0.0)
                info["note"] = (
                    f"拓扑自带的键表**明显残缺**（{degen_note}），已自动改用**按坐标"
                    f"猜键**的结果（忽略原有 {n_bonds_before:,} 个键，重猜 "
                    f"{info.get('bonds', 0):,} 个）—— 否则逐分子 Rg/R_ee、分子质心 "
                    f"MSD、RDF 的同分子剔除都会被碎片污染，而这些都不会报错。")
            else:
                info["note"] = (f"已忽略拓扑里原有的 {n_bonds_before:,} 个键，"
                                f"按坐标重新猜出 {info.get('bonds', 0):,} 个")
        self.bond_guess_info = info

    def _ensure_object_attrs(self) -> None:
        """补齐对象模型必需的 ``elements`` / ``masses`` / ``molnums``。

        ``.gro`` 三者都不提供，而"按 molecule 统计 Rg / RDF 排除同分子配对 /
        质心 MSD"全都要用；这里统一在加载末尾补齐（见
        :func:`mdta.topology.ensure_object_attrs`）。失败不阻断加载。
        """
        from .topology import ensure_object_attrs

        try:
            self.object_info = ensure_object_attrs(self._universe)
        except Exception as exc:  # noqa: BLE001
            self.object_info = {"skipped": f"对象层属性补齐失败: {exc}"}

    def _validate(self) -> None:
        u = self._universe
        if u.atoms.n_atoms == 0:
            raise TrajectoryError("拓扑文件中没有原子。")
        try:
            n_frames = len(u.trajectory)
        except Exception as exc:  # noqa: BLE001
            raise TrajectoryError(f"无法读取轨迹帧数: {exc}") from exc
        if n_frames == 0:
            raise TrajectoryError("轨迹文件中没有帧。")
        if self.trajectory:
            # 轨迹里的原子数必须与拓扑一致（有些格式 MDAnalysis 不会自己查）。
            # ⚠️ 这里**直接用已经建好的 Universe** 的原子数，不再回头调
            #    `count_topology_atoms()` —— 那会为 .tpr 这种二进制拓扑**再完整解析
            #    一遍**（实测 47681 原子的 tpr 多花约 1.8 s，占整个打开时间的 40%）。
            #    参考对象本来就该是"我们真正要用的那个 Universe"。
            n_top = int(u.atoms.n_atoms)
            n_traj = (count_trajectory_atoms(self.trajectory)
                      if self.trajectory else None)
            if n_traj is not None and n_traj != n_top:
                raise TrajectoryError(self._mismatch_message(n_top, n_traj))

    def _mismatch_message(self, n_top: int, n_traj: int) -> str:
        """原子数不匹配时，顺手在同目录里找原子数对得上的候选拓扑文件。"""
        lines = [f"原子数不匹配：拓扑文件 {n_top:,} 个原子，"
                 f"轨迹文件 {n_traj:,} 个原子。",
                 f"  拓扑: {os.path.basename(self.topology or '')}"]

        cands: list[str] = []
        try:
            d = os.path.dirname(os.path.abspath(self.topology)) or "."
            for name in sorted(os.listdir(d)):
                suf = Path(name).suffix.lower()
                if suf not in TOPOLOGY_SUFFIXES or suf in GROMACS_TOP_SUFFIXES:
                    continue
                p = os.path.join(d, name)
                if os.path.abspath(p) == os.path.abspath(self.topology):
                    continue
                n = count_topology_atoms(p)
                if n == n_traj:
                    cands.append(f"{name}（{n:,} 原子 — 与轨迹一致 ✓）")
        except Exception:  # noqa: BLE001
            pass

        if cands:
            lines.append("  同目录下这个（些）文件的原子数与轨迹一致，"
                         "应该就是正确的拓扑：")
            lines.extend(f"    · {c}" for c in cands)
        else:
            lines.append("  请确认拓扑与轨迹来自同一次模拟；"
                         "或改用与该轨迹配套的 .tpr / .gro / .pdb。")
        return "\n".join(lines)

    @property
    def universe(self):
        """底层 ``MDAnalysis.Universe``。"""
        return self.load()

    # -------------------------------------------------------------- 属性
    @property
    def n_atoms(self) -> int:
        return int(self.universe.atoms.n_atoms)

    @property
    def has_velocities(self) -> bool:
        """轨迹里是否带速度（MDAnalysis 的 ``ts.velocities`` 可用）。

        Materials Studio ``.trj`` 本身有速度（第 4-6 条 double 数组），但 XTC 装不下；
        只有按 ``velocities=True`` 转成 **TRR** 时才能在这里读到 ``True``。
        """
        try:
            return bool(getattr(self.universe.trajectory.ts, "has_velocities", False))
        except Exception:                                    # noqa: BLE001
            return False

    @property
    def n_frames(self) -> int:
        return int(len(self.universe.trajectory))

    @property
    def dt_ps(self) -> float:
        """相邻帧时间间隔（ps）。"""
        u = self.universe
        try:
            return float(u.trajectory.dt)
        except Exception:  # noqa: BLE001
            t = self.times_ps
            return float(np.median(np.diff(t))) if t.size > 1 else 0.0

    def _uniform_times_ps(self, t0: float, t1: float) -> np.ndarray | None:
        """若时间轴是**等差数列**，直接解析构造每帧时间；否则返回 ``None``。

        判据（全部通过才采用，任何一条不满足就回退全文件扫描）：

        1. 表头 ``dt`` 有限且为正；
        2. 实测首末时刻与 ``dt`` 自洽：``t1 - t0 == dt * (n - 1)``；
        3. **抽样复核**：抽查 5 帧，直接定位读出的时间与解析值一致
           （容差按 float32 存储精度给：XTC 的时间是 float32，
           10⁵ ps 量级的量化台阶约 0.008 ps）。

        这样"均匀采样"（GROMACS 输出的常态）就是 O(1)，而拼接轨迹、
        变速输出、restart 这类非均匀时间轴会被第 2/3 条挡下，走精确全扫。
        """
        n = self.n_frames
        if n <= 0 or not (np.isfinite(t0) and np.isfinite(t1)):
            return None
        if n == 1:
            return np.array([t0], dtype=float)
        try:
            dt = float(self.universe.trajectory.dt)
        except Exception:  # noqa: BLE001
            return None
        if not (np.isfinite(dt) and dt > 0):
            return None
        span = t1 - t0
        tol = max(1e-3, 3e-7 * max(abs(t0), abs(t1), abs(span), 1.0))
        if abs(span - dt * (n - 1)) > tol:
            return None
        t = t0 + dt * np.arange(n, dtype=float)
        u = self.universe
        idx = sorted({int(round(x)) for x in np.linspace(0, n - 1, 5)})
        prev: int | None = None
        try:
            prev = int(u.trajectory.ts.frame)
        except Exception:  # noqa: BLE001
            prev = None
        try:
            for i in idx:
                u.trajectory[i]
                if abs(float(u.trajectory.ts.time) - float(t[i])) > tol:
                    return None
        except Exception:  # noqa: BLE001
            return None
        finally:
            if prev is not None:
                try:
                    u.trajectory[prev]
                except Exception:  # noqa: BLE001
                    pass
        return t

    @property
    def times_ps(self) -> np.ndarray:
        """每一帧的模拟时间（ps）。**结果会缓存**（首次可能较慢）。

        ⚠️ 慢路径的代价极大：XTC/TRR 的 reader **没有** ``.times`` 属性，
        于是会掉进下面的逐帧兜底循环 —— 每取一个时刻都要读一整帧坐标，
        46 体系（10001 帧 / 1.7 GB）实测 **22 s**。而 ``total_time_ps``、
        体系信息、帧选择都会用到它 —— 早期版本每次访问都重算，于是
        "文件读取完成"之后还要沉默 50 s 才返回（信息显示两次各扫一次）。

        现在两条路：均匀时间轴走 :meth:`_uniform_times_ps`（O(1)，抽样复核），
        否则才全扫一次并缓存。返回的是**同一个数组**，调用方不得就地改写。
        """
        if self._times_cache is not None:
            return self._times_cache
        rng = self.time_range_ps()
        if rng is not None:
            fast = self._uniform_times_ps(*rng)
            if fast is not None:
                self._times_cache = fast
                return fast
        u = self.universe
        try:
            t = np.asarray(u.trajectory.times, dtype=float)
            if t.size == self.n_frames:
                self._times_cache = t
                return t
        except Exception:  # noqa: BLE001
            pass
        # 回退：逐帧迭代取时间
        out = np.empty(self.n_frames, dtype=float)
        for i, ts in enumerate(u.trajectory):
            out[i] = ts.time
        self._times_cache = out
        return out

    def time_range_ps(self) -> tuple[float, float] | None:
        """首帧/末帧时刻 ``(t0, t1)``（ps），**O(1)**，不扫全文件。

        只定位第 0 帧与最后一帧读时间（有偏移缓存时是两次随机访问），
        并在返回前把当前帧**恢复原位**，所以可以安全地当作只读查询使用。
        轨迹没有时间列时返回 ``None``。
        """
        if self._time_range_cache is not None:
            return self._time_range_cache
        if self._times_cache is not None and self._times_cache.size:
            self._time_range_cache = (float(self._times_cache[0]),
                                      float(self._times_cache[-1]))
            return self._time_range_cache
        u = self.universe
        n = self.n_frames
        if n == 0:
            return None
        prev: int | None = None
        try:
            prev = int(u.trajectory.ts.frame)
        except Exception:  # noqa: BLE001
            prev = None
        try:
            u.trajectory[0]
            t0 = float(u.trajectory.ts.time)
            if n == 1:
                t1 = t0
            else:
                u.trajectory[n - 1]
                t1 = float(u.trajectory.ts.time)
        except Exception:  # noqa: BLE001
            t = self.times_ps                      # 回退：全量数组
            if not t.size:
                return None
            self._time_range_cache = (float(t[0]), float(t[-1]))
            return self._time_range_cache
        finally:
            if prev is not None:
                try:
                    u.trajectory[prev]             # 恢复调用前的帧，避免副作用
                except Exception:  # noqa: BLE001
                    pass
        if not (np.isfinite(t0) and np.isfinite(t1)):
            return None
        self._time_range_cache = (t0, t1)
        return self._time_range_cache

    @property
    def total_time_ps(self) -> float:
        rng = self.time_range_ps()
        return float(rng[1] - rng[0]) if rng else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MDTrajectory(topology={os.path.basename(self.topology)!r}, "
            f"trajectory={os.path.basename(self.trajectory) if self.trajectory else None!r}, "
            f"n_atoms={self.n_atoms}, n_frames={self.n_frames})"
        )


# ------------------------------------------------------------------ 加载器
def _autodetect_trajectory(topology: str) -> str | None:
    """给定了拓扑文件时，自动寻找同名轨迹文件。

    候选后缀按"常用程度"排序（`.xtc` 最先），**只挑真能当轨迹的格式**；
    `.pdb` / `.gro` 这类"两者皆可"的格式**不参与自动匹配** —— 否则拿
    `sys.pdb` 打开时会把自己当成轨迹，变成"拓扑=轨迹=同一文件"的怪状态。
    """
    stem = Path(topology).with_suffix("")
    order = [".xtc", ".trr", ".dcd", ".nc", ".ncdf", ".trj", ".trz",
             ".tng", ".lammpstrj", ".dump", ".lammpsdump", ".mdcrd", ".crd",
             ".h5md", ".gsd"]
    for suf in order:
        if not can_be_trajectory("x" + suf):
            continue
        for cand in (stem.with_suffix(suf), Path(str(stem) + suf)):
            if cand.is_file() and str(cand) != str(topology):
                return str(cand)
    return None


def load_trajectory(
    topology: str | None = None,
    trajectory: str | None = None,
    *,
    autodetect: bool = True,
    guess_bonds: bool | None = None,
    velocities: bool | None = None,
) -> MDTrajectory:
    """读取 MD 轨迹。

    支持三种调用方式：

    - ``load_trajectory("sys.tpr", "sys.xtc")``
    - ``load_trajectory("sys.tpr")``  —— 自动寻找同名 ``.xtc``/``.trr``/``.dcd``
    - ``load_trajectory("sys.gro")``  —— 只读单帧结构

    若只给一个参数且它是轨迹文件（例如 ``.xtc``），会因为缺少拓扑而报错，
    此时需要显式提供拓扑文件。

    ``guess_bonds`` 控制"没有键表时是否按坐标猜键"（见 :class:`MDTrajectory`）。

    ``velocities``（只对 Materials Studio ``.trj`` 有效）：``True`` 时转成 **TRR**，
    把 ``.trj`` 里本来就有的速度一并保存（坐标 + 速度；文件约为 XTC 的 5~6 倍）；
    默认 ``None/False`` 只转坐标（XTC，最小、分析最快）。
    """
    if topology is None and trajectory is None:
        raise TrajectoryError("必须提供至少一个文件路径。")

    # 只给了一个参数：判断它是拓扑还是轨迹
    if topology is not None and trajectory is None:
        if not can_be_topology(topology):
            if can_be_trajectory(topology):
                suf = suffix_of(topology)
                extra = ""
                if suf in (".lammpstrj", ".dump", ".lammpsdump"):
                    extra = ("\n  LAMMPS 的 dump 只有原子类型号，没有元素/质量 —— "
                             "请把同体系的 **data 文件**（`.lmp` / `.dat` / `.data`）"
                             "当拓扑，dump 当轨迹（同名会自动配对）。")
                raise TrajectoryError(
                    f"{topology} 看起来是轨迹文件，缺少拓扑文件；"
                    f"请同时指定拓扑文件。{extra}"
                    f"可用拓扑格式：{topology_help()}")
            hint = unsupported_hint(topology)
            raise TrajectoryError(
                f"不认识的文件格式: {topology}"
                + (f"\n  {hint}" if hint else "")
                + f"；可用拓扑格式：{topology_help()}")
        if autodetect:
            found = _autodetect_trajectory(topology)
            if found:
                return MDTrajectory(topology, found, guess_bonds=guess_bonds,
                                    velocities=velocities)

    if topology is None and trajectory is not None:
        raise TrajectoryError("缺少拓扑文件。")

    return MDTrajectory(topology, trajectory, guess_bonds=guess_bonds,
                        velocities=velocities)


def scan_input_files(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """把一批文件分成（拓扑文件, 轨迹文件）。用于命令行/图形界面。

    "两者皆可"的格式（`.pdb` / `.gro` / `.xyz` …）优先算**拓扑**，
    与旧行为一致（命令行给一堆文件时，`.pdb` 通常是结构文件）。
    """
    tops: list[str] = []
    trajs: list[str] = []
    for p in paths:
        if can_be_topology(p):
            tops.append(p)
        elif can_be_trajectory(p):
            trajs.append(p)
    return tops, trajs
