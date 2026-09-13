# -*- coding: utf-8 -*-
"""文件读取模块。

第一阶段优先保证 **GROMACS TPR + XTC** 稳定运行；
同时支持 :mod:`.gro` / :mod:`.pdb` / :mod:`.trr` / :mod:`.dcd` / :mod:`.xyz`
等 MDAnalysis 原生支持的格式作为可扩展格式。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# ------------------------------------------------------------------ 格式表
TOPOLOGY_SUFFIXES = {".tpr", ".gro", ".pdb", ".psf", ".prmtop", ".parm7", ".top", ".xyz", ".data"}
TRAJECTORY_SUFFIXES = {".xtc", ".trr", ".dcd", ".nc", ".nctraj", ".xyz", ".lammpstrj", ".trj"}
SUPPORTED_SUFFIXES = TOPOLOGY_SUFFIXES | TRAJECTORY_SUFFIXES

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
        # 已有就地缓存就搬过来复用，省掉一次整文件重扫（1.7 GB 轨迹差别很大）
        if ending == "npz" and os.path.isfile(inplace):
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
            if src not in _OFFSETS_WARNED:
                _OFFSETS_WARNED.add(src)
                warnings.warn(
                    f"为避免 MDAnalysis 在数据目录不可写时因 filelock 无限重试而卡死，"
                    f"偏移缓存已重定向到 {_offsets_cache_dir()}（数据目录不再被写入）。"
                    f"如需原生就地缓存请设 MDTA_OFFSETS_INPLACE=1。",
                    RuntimeWarning, stacklevel=2)
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
#: （见 :mod:`mdta.topology`）。
BOND_GUESSABLE_SUFFIXES = {".gro", ".pdb", ".ent", ".xyz", ".pdb1"}
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
    #: 猜键结果统计（未猜时为 ``None``）
    bond_guess_info: dict | None = None
    #: 对象层属性补齐诊断（elements / masses / molnums）
    object_info: dict | None = None
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
            if self.trajectory is not None and not os.path.isfile(self.trajectory):
                raise TrajectoryError(f"找不到轨迹文件: {self.trajectory}")
            if Path(self.topology).suffix.lower() in GROMACS_TOP_SUFFIXES:
                raise TrajectoryError(self._gromacs_top_message())
            _progress("读取拓扑与轨迹", topology=os.path.basename(self.topology),
                      trajectory=(os.path.basename(self.trajectory)
                                  if self.trajectory else None),
                      trajectory_mb=(round(os.path.getsize(self.trajectory) / 1e6, 1)
                                     if self.trajectory
                                     and os.path.isfile(self.trajectory) else None),
                      first_scan=_needs_frame_scan(self.trajectory))
            try:
                ensure_offsets_writable()   # 数据目录只读时重定向偏移缓存，避免死锁
                if self.trajectory:
                    self._universe = mda.Universe(self.topology, self.trajectory)
                else:
                    self._universe = mda.Universe(self.topology)
            except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
                # MDAnalysis 自己也会检查原子数；这时给出更有用的提示
                n_top, n_traj = self._read_counts()
                if n_top and n_traj and n_top != n_traj:
                    raise TrajectoryError(self._mismatch_message(n_top, n_traj)) from exc
                raise TrajectoryError(
                    f"读取失败: {os.path.basename(self.topology)}: {exc}") from exc
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
                     if Path(n).suffix.lower() in {".gro", ".pdb", ".tpr", ".psf", ".prmtop"}]
            if cands:
                lines.append("  同目录下这些文件可以用作拓扑：")
                lines.extend(f"    · {c}" for c in cands[:8])
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines)

    # ------------------------------------------------------------ 键补全
    def _maybe_guess_bonds(self) -> None:
        """``.gro`` 之类没有键表的格式：按坐标猜键，把分子/链划分补回来。"""
        from .topology import guess_bonds as _guess
        from .topology import needs_bond_guessing

        u = self._universe
        if not needs_bond_guessing(u):
            return                                   # 拓扑自带键表
        if self.guess_bonds is False:
            return
        suf = Path(self.topology or "").suffix.lower()
        if self.guess_bonds is None and suf not in BOND_GUESSABLE_SUFFIXES:
            return                                   # .tpr 等本应带键，不强猜
        if u.atoms.n_atoms > AUTO_GUESS_MAX_ATOMS:
            self.bond_guess_info = {
                "skipped": f"原子数 {u.atoms.n_atoms:,} 超过自动猜键上限 "
                           f"{AUTO_GUESS_MAX_ATOMS:,}，已跳过",
            }
            return
        try:
            self.bond_guess_info = _guess(u)
        except Exception as exc:  # noqa: BLE001 - 猜键失败不应阻断加载
            self.bond_guess_info = {"skipped": f"猜键失败: {exc}"}

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
            # 轨迹里的原子数必须与拓扑一致（有些格式 MDAnalysis 不会自己查）
            n_top, n_traj = self._read_counts()
            n_top = n_top or u.atoms.n_atoms
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
    """给定了拓扑文件时，自动寻找同名轨迹文件。"""
    stem = Path(topology).with_suffix("")
    for suf in (".xtc", ".trr", ".dcd", ".nc", ".trj", ".lammpstrj"):
        for cand in (stem.with_suffix(suf), Path(str(stem) + suf)):
            if cand.is_file():
                return str(cand)
    return None


def load_trajectory(
    topology: str | None = None,
    trajectory: str | None = None,
    *,
    autodetect: bool = True,
    guess_bonds: bool | None = None,
) -> MDTrajectory:
    """读取 MD 轨迹。

    支持三种调用方式：

    - ``load_trajectory("sys.tpr", "sys.xtc")``
    - ``load_trajectory("sys.tpr")``  —— 自动寻找同名 ``.xtc``/``.trr``/``.dcd``
    - ``load_trajectory("sys.gro")``  —— 只读单帧结构

    若只给一个参数且它是轨迹文件（例如 ``.xtc``），会因为缺少拓扑而报错，
    此时需要显式提供拓扑文件。

    ``guess_bonds`` 控制"没有键表时是否按坐标猜键"（见 :class:`MDTrajectory`）。
    """
    if topology is None and trajectory is None:
        raise TrajectoryError("必须提供至少一个文件路径。")

    # 只给了一个参数：判断它是拓扑还是轨迹
    if topology is not None and trajectory is None:
        suf = Path(topology).suffix.lower()
        if suf in TRAJECTORY_SUFFIXES and suf not in TOPOLOGY_SUFFIXES:
            raise TrajectoryError(
                f"{topology} 看起来是轨迹文件，缺少拓扑文件；"
                f"请同时指定 .tpr/.gro/.pdb 等拓扑文件。"
            )
        if autodetect:
            found = _autodetect_trajectory(topology)
            if found:
                return MDTrajectory(topology, found, guess_bonds=guess_bonds)

    if topology is None and trajectory is not None:
        raise TrajectoryError("缺少拓扑文件。")

    return MDTrajectory(topology, trajectory, guess_bonds=guess_bonds)


def scan_input_files(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """把一批文件分成（拓扑文件, 轨迹文件）。用于命令行/图形界面。"""
    tops: list[str] = []
    trajs: list[str] = []
    for p in paths:
        suf = Path(p).suffix.lower()
        if suf in TOPOLOGY_SUFFIXES:
            tops.append(p)
        elif suf in TRAJECTORY_SUFFIXES:
            trajs.append(p)
    return tops, trajs
