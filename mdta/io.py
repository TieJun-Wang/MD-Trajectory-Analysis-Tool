# -*- coding: utf-8 -*-
"""文件读取模块。

第一阶段优先保证 **GROMACS TPR + XTC** 稳定运行；
同时支持 :mod:`.gro` / :mod:`.pdb` / :mod:`.trr` / :mod:`.dcd` / :mod:`.xyz`
等 MDAnalysis 原生支持的格式作为可扩展格式。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


def mda_universe(path: str):
    """只读打开一个拓扑文件（内部用，便于在错误信息里探测候选文件）。"""
    import MDAnalysis as mda

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
            try:
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
            self._validate()
            self._maybe_guess_bonds()
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

    @property
    def times_ps(self) -> np.ndarray:
        """每一帧的模拟时间（ps）。"""
        u = self.universe
        try:
            t = np.asarray(u.trajectory.times, dtype=float)
            if t.size == self.n_frames:
                return t
        except Exception:  # noqa: BLE001
            pass
        # 回退：逐帧迭代取时间
        out = np.empty(self.n_frames, dtype=float)
        for i, ts in enumerate(u.trajectory):
            out[i] = ts.time
        return out

    @property
    def total_time_ps(self) -> float:
        t = self.times_ps
        return float(t[-1] - t[0]) if t.size else 0.0

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
