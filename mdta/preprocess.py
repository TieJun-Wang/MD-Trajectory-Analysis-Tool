# -*- coding: utf-8 -*-
"""轨迹预处理模块。

包含三部分内容：

1. **周期性边界条件（PBC）** —— 把被模拟盒边界切断的分子拼回完整分子
   （``unwrap`` / ``make_whole``），保证 Rg、端到端距离等分子内几何量正确。
2. **轨迹抽帧** —— 按起始时间、结束时间、分析间隔挑选参与统计的帧。
3. **平衡阶段选择** —— 丢弃模拟初期未平衡的数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameSelection:
    """被选中的帧集合及其来源说明。"""

    indices: np.ndarray
    times_ps: np.ndarray
    start_ps: float | None = None
    stop_ps: float | None = None
    interval_ps: float | None = None
    equil_ps: float | None = None
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, key) -> "FrameSelection":
        """支持切片/索引，便于只取前若干帧或某个时间段做快速试算。"""
        if isinstance(key, slice):
            idx = self.indices[key]
            t = self.times_ps[key]
        else:
            idx = np.atleast_1d(self.indices[key])
            t = np.atleast_1d(self.times_ps[key])
        return FrameSelection(indices=np.asarray(idx, dtype=int),
                              times_ps=np.asarray(t, dtype=float),
                              start_ps=self.start_ps, stop_ps=self.stop_ps,
                              interval_ps=self.interval_ps, equil_ps=self.equil_ps,
                              notes=list(self.notes))

    def head(self, n: int) -> "FrameSelection":
        return self[:int(n)]

    def tail(self, n: int) -> "FrameSelection":
        return self[-int(n):] if n else self[0:0]

    def halves(self) -> tuple["FrameSelection", "FrameSelection"]:
        """把选中的帧等分成前半段与后半段（用于比较结构有序化趋势）。"""
        k = self.indices.size // 2
        return self[:k], self[k:]

    @property
    def n_frames(self) -> int:
        return int(self.indices.size)

    def describe(self) -> str:
        if self.n_frames == 0:
            return "未选中任何帧"
        return (
            f"选中 {self.n_frames} 帧，"
            f"时间范围 {self.times_ps[0]:.3f} – {self.times_ps[-1]:.3f} ps"
        )


def select_frames(
    times_ps: np.ndarray,
    *,
    start_ps: float | None = None,
    stop_ps: float | None = None,
    interval_ps: float | None = None,
    equil_ps: float | None = None,
    max_frames: int | None = None,
    strict_interval: bool = False,
) -> FrameSelection:
    """按时间区间 + 间隔挑选帧。

    参数
    ----
    times_ps
        每帧的模拟时间（ps）。
    start_ps / stop_ps
        参与分析的起止时间（含端点）。``None`` 表示不限制。
    interval_ps
        分析间隔（ps）。取距离目标时间点最近的一帧（``strict_interval=True``
        时要求时间差不超过间隔的一半，否则跳过）。
    equil_ps
        平衡阶段时长（ps），等价于 ``start_ps = max(start_ps, equil_ps)``；
        用于"排除模拟初期未平衡的数据"。
    max_frames
        最多使用的帧数（等间隔抽样），用于控制计算量。
    """
    times = np.asarray(times_ps, dtype=float)
    sel = FrameSelection(indices=np.arange(times.size), times_ps=times,
                         start_ps=start_ps, stop_ps=stop_ps,
                         interval_ps=interval_ps, equil_ps=equil_ps)
    if times.size == 0:
        sel.indices = np.array([], dtype=int)
        sel.times_ps = np.array([], dtype=float)
        sel.notes.append("轨迹中没有帧")
        return sel

    t0, t1 = float(times[0]), float(times[-1])

    lo = t0 if start_ps is None else float(start_ps)
    if equil_ps is not None:
        lo = max(lo, t0 + float(equil_ps))
    hi = t1 if stop_ps is None else float(stop_ps)
    if hi < lo:
        sel.notes.append(f"结束时间 {hi} 早于起始时间 {lo}，没有可用数据")
        sel.indices = np.array([], dtype=int)
        sel.times_ps = np.array([], dtype=float)
        return sel

    sel.notes.append(f"时间窗口: {lo:.3f} – {hi:.3f} ps")

    # 轨迹时间常以 float32 存储（如 XTC），累积误差可达 1e-7 相对量级，
    # 因此用相对容差判定区间端点，避免恰好落在端点上的帧被漏掉。
    tol = 1e-6 * max(1.0, abs(lo), abs(hi))
    idx = np.nonzero((times >= lo - tol) & (times <= hi + tol))[0]
    if idx.size == 0:
        sel.notes.append("时间窗口内没有任何帧，已回退为全轨迹")
        idx = np.arange(times.size)

    if interval_ps and interval_ps > 0:
        targets = np.arange(times[idx[0]], times[idx[-1]] + 1e-9, float(interval_ps))
        picked: list[int] = []
        for tgt in targets:
            j = int(np.argmin(np.abs(times[idx] - tgt)))
            cand = int(idx[j])
            if strict_interval and abs(times[cand] - tgt) > interval_ps / 2.0:
                continue
            if not picked or cand != picked[-1]:
                picked.append(cand)
        # 保证末端帧尽可能被纳入
        if picked and picked[-1] != int(idx[-1]) and (times[idx[-1]] - times[picked[-1]]) >= interval_ps * 0.5:
            picked.append(int(idx[-1]))
        idx = np.array(sorted(set(picked)), dtype=int)
        sel.notes.append(f"按 {interval_ps:g} ps 抽帧后剩余 {idx.size} 帧")

    if max_frames and idx.size > max_frames:
        keep = np.unique(np.linspace(0, idx.size - 1, int(max_frames)).round().astype(int))
        sel.notes.append(f"帧数超过上限 {max_frames}，等间隔抽样为 {keep.size} 帧")
        idx = idx[keep]

    sel.indices = idx
    sel.times_ps = times[idx]
    return sel


def unwrap_trajectory(universe, compound: str = "fragments", verbose: bool = False):
    """把整条轨迹的坐标展开（unwrap），使分子保持完整。

    返回**新的** Universe，原 Universe 不被修改。

    说明
    ----
    ``compound`` 可取 ``"fragments"``（推荐，按连通分子/链处理）、
    ``"residues"`` 或 ``"segments"``。对高分子链，如果力场拓扑中有键连接，
    ``"fragments"`` 恰好对应一条链。
    """
    u = universe.copy()
    if u.trajectory is None:
        return u
    for ts in u.trajectory:
        try:
            u.atoms.unwrap(compound=compound)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [warn] unwrap({compound}) 失败，尝试 residues: {exc}")
            u.atoms.unwrap(compound="residues")
    u.trajectory.rewind()
    return u


def make_whole_inplace(ag) -> None:
    """在当前帧把选定的原子组按分子连通性补完整（会修改坐标）。"""
    ag.unwrap(compound="fragments")


def minimize_to_anchor(ag, anchor_pos: np.ndarray, box) -> np.ndarray:
    """把原子组按最小镜像平移，使其尽量靠近 ``anchor_pos``。

    用于界面/接触分析中避免周期性镜像导致的假距离。
    返回平移后的坐标副本（不修改原原子组）。
    """
    from MDAnalysis.lib.distances import minimize_vectors

    pos = np.asarray(ag.positions, dtype=float)
    if pos.size == 0 or box is None:
        return pos
    rel = minimize_vectors(pos - np.asarray(anchor_pos, dtype=float), box)
    return rel + np.asarray(anchor_pos, dtype=float)
