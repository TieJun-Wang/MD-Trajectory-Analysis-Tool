# -*- coding: utf-8 -*-
"""辅助动力学分析模块（设计大纲第 20 章）。

均方位移 MSD：

.. math::

    MSD(t) = \\langle |\\mathbf r(t) - \\mathbf r(0)|^2 \\rangle

MSD 在本项目中作为辅助分析，可用于比较原子运动、分子运动、链段运动、
界面区域运动差异以及结晶前后的运动变化。

因为要统计粒子的真实位移，必须跨越周期性边界追踪粒子，所以本模块使用
``NoJump`` 变换消除帧间的坐标跳变，而不是只做帧内的分子展开。
"""

from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from ..core import AnalysisResult, Panel
from ..preprocess import FrameSelection
from .base import frame_iterator, register

__all__ = ["compute_msd", "compute_msd_axes", "particle_series", "msd_fft",
           "analyze_msd", "diffusion_coefficient"]

#: Å²/ps -> m²/s
A2_PER_PS_TO_M2_PER_S = 1e-8

#: 粒子坐标序列的内存上限（字节）。**不再用来抽稀粒子**（见下）：
#: 超过它就切换到"分块 + 内存映射"模式，粒子一个不少，只是不再一次性驻留内存。
DEFAULT_MEMORY_BUDGET = 1.2e9

#: FFT 工作区预算（字节/批）。`rfft` 的中间量是 `n_fft × 批宽 × 16 B`（复数），
#: 按它反推每批处理多少列，保证内存与"粒子数"无关。
FFT_WORKSPACE_BUDGET = 2.0e8

#: 序列落盘（内存映射）时用的目录
SCRATCH_DIRNAME = "_msd_scratch"


def _scratch_dir() -> str:
    """放内存映射文件的目录（默认在本工具目录下，不可写则退到系统临时目录）。"""
    import tempfile
    from pathlib import Path

    cand = Path(__file__).resolve().parent.parent.parent / SCRATCH_DIRNAME
    for d in (cand, Path(tempfile.gettempdir()) / "mdta_msd"):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".w"
            probe.write_bytes(b"0")
            probe.unlink()
            return str(d)
        except Exception:                                     # noqa: BLE001
            continue
    return tempfile.gettempdir()


def msd_numerator_by_group(A, *, groups=None, batch: int | None = None,
                           workers: int = 0, verbose: bool = False):
    """MSD 的**分子**（``Σ_{t<N-τ}(A_{t+τ}-A_t)²``）按组求和，分块做 FFT。

    为什么要"按组 + 分块"这两件事：

    1. **按组**：总 MSD = MSD_x + MSD_y + MSD_z。老实现把三个分量各调一次
       ``msd_fft``（还要再乘误差分块数），一次分析要跑 3×(1+块数) ≈ 18 遍 FFT。
       这里把三个分量当成同一批**列**（`x` 的 P 列 + `y` 的 P 列 + `z` 的 P 列）
       一次 FFT 算完，再按组求和 —— 结果逐位相同，FFT 次数降到 1/3。
    2. **分块**：`rfft` 的中间量是 `n_fft × 列数 × 16 B`，P=50000、帧数 6001 时
       单个分量就要十几 GB。按列切批后内存与粒子数无关，**因此不需要再抽稀粒子**：
       每个原子/分子都能参与统计（这正是用户要的）。

    参数
    ----
    A : ``(n_frames, n_columns)`` 数组或内存映射
    groups : 长度 ``n_columns`` 的整数数组，列 → 组号；``None`` 表示全部一列一组
    batch : 每批列数；``None`` 时按 ``FFT_WORKSPACE_BUDGET`` 自动定
    workers : >1 时用多进程并行批（仅在 ``A`` 是磁盘映射时启用，避免拷贝数据）

    返回 ``(num, counts)``：``num`` 形状 ``(n_frames, n_groups)``，
    ``counts`` 形状 ``(n_groups,)``。第 g 组的 ``MSD(τ) = num[τ, g] / ((N-τ)·counts[g])``。
    """
    arr = A
    n = int(arr.shape[0])
    n_col = int(arr.shape[1])
    if n < 2 or n_col < 1:
        return np.zeros((max(n, 0), 1)), np.zeros(1, dtype=np.int64)
    if groups is None:
        groups = np.zeros(n_col, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    n_grp = int(groups.max()) + 1
    counts = np.bincount(groups, minlength=n_grp).astype(np.int64)

    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    if not batch or batch <= 0:
        batch = max(1, int(FFT_WORKSPACE_BUDGET / (n_fft * 16)))
    batch = int(min(batch, n_col))
    tau = np.arange(n)
    num = np.zeros((n, n_grp), dtype=np.float64)

    def _one(lo: int, hi: int):
        """一批列的分子贡献（返回 ``(hi-lo, n_frames, n_groups)`` 的累加）。"""
        blk = np.asarray(arr[:, lo:hi], dtype=np.float64)          # 逐批转 float64
        F = np.fft.rfft(blk, n=n_fft, axis=0)
        corr = np.fft.irfft(F * np.conjugate(F), n=n_fft, axis=0)[:n]
        del F
        cs = np.cumsum(blk * blk, axis=0)
        tot = cs[-1]
        p2 = cs[n - 1 - tau]
        p1 = tot - np.concatenate([np.zeros((1, blk.shape[1])), cs[:-1]], axis=0)
        del cs
        nb = p1 + p2 - 2.0 * corr
        del p1, p2, corr, blk
        # 按列归属累加到组：列数很大时用 bincount 逐帧做，避免 (n, n_col, n_grp) 大数组
        g = groups[lo:hi]
        if n_grp == 1:
            return nb.sum(axis=1)[:, None]
        out = np.zeros((n, n_grp), dtype=np.float64)
        for gi in range(n_grp):
            m = (g == gi)
            if m.any():
                out[:, gi] = nb[:, m].sum(axis=1)
        return out

    use_mp = int(workers or 0) > 1 and isinstance(arr, np.memmap)
    if use_mp:
        # 多进程：只在"序列已落盘（内存映射）"时启用，避免把大数组 pickle 出去。
        # 受限环境（沙箱禁命名管道等）里进程池可能建不起来 —— 这时**退回串行**，
        # 只记一条说明，绝不能因为这个让整项分析失败。
        try:
            from concurrent.futures import ProcessPoolExecutor

            spans = [(lo, min(lo + batch, n_col)) for lo in range(0, n_col, batch)]
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                for out in ex.map(_batch_worker,
                                  [(str(arr.filename), arr.shape, arr.dtype.str,
                                    lo, hi, groups[lo:hi], n, n_fft, n_grp)
                                   for lo, hi in spans]):
                    num += out
            return num, counts
        except Exception as exc:                              # noqa: BLE001
            if verbose:
                print(f"  [MSD] 多进程不可用（{type(exc).__name__}: {exc}），退回串行")
    for lo in range(0, n_col, batch):
        hi = min(lo + batch, n_col)
        num += _one(lo, hi)
        if verbose:
            print(f"  [MSD] 列 {hi:,}/{n_col:,}")
    return num, counts


def msd_numerator_by_group_segmented(A, *, groups=None, max_lag: int,
                                     block_frames: int | None = None,
                                     overlap: float = 0.5,
                                     batch: int | None = None, verbose: bool = False):
    """**分段（滚动窗口）**版 MSD 分子：内存恒定、长轨迹可跑。

    为什么需要它：完整版要在一次 FFT 里覆盖整条时间轴，长度取到
    ``n_fft = 2^ceil(log2 2N)``。N=6001 时 n_fft=16384，50 000 个粒子实测
    **150 s**（还得先把 3.6 GB 序列准备好）。而扩散系数的拟合窗口通常只用到
    τ ≤ 20–50% 的最大 τ，也就是说**长 τ 的样本根本用不上**。

    做法：把时间轴切成 ``block_frames`` 长的块（默认 ``2·max_lag+1``，相邻块
    50% 重叠），每块只算 τ ≤ max_lag，再按"每个 τ 上实际用到的起点数"加权合并：

        num[τ] = Σ_块 Σ_{t<L−τ} |r_{t+τ}−r_t|²，   origins[τ] = Σ_块 (L−τ)
        MSD(τ) = num[τ] / (origins[τ] · n_particles)

    好处：序列不必整体驻留内存（配合内存映射就是**常数内存**，任意长轨迹都能跑），
    且每个 τ 的起点数由块内统计给出、不需要跨块配对。

    ⚠️ **它不是"计算量上的加速"**：50% 重叠时总 FFT 工作量约为
    ``2n·log L``，完整版是 ``n·log(2n)``，两者同量级（有时分段还更慢 ——
    实测 47 453 原子 × 1001 帧：完整 47.9 s、分段 τ≤200 57.4 s）。
    它的价值在于**内存与流式**：帧数上万时完整版 n_fft 会涨到 32768 且要 5.7 GB 序列，
    分段版固定为块长的 FFT 且只驻留一块。要真正省时间，应该减少**帧数**
    （时间分辨率）而不是开分段。

    代价：τ 越接近块长，起点数越少、噪声越大 —— 所以默认块长取 ``2·max_lag+1``，
    保证每个 τ 至少有约一半的起点。

    返回 ``(num, origins, counts)``，形状 ``(n_lag, n_groups)`` / ``(n_lag,)``。
    """
    arr = A
    n = int(arr.shape[0])
    n_col = int(arr.shape[1])
    if max_lag is None or max_lag <= 0:
        raise ValueError("分段模式必须给正的最大 τ")
    max_lag = int(min(max_lag, n - 1))
    if groups is None:
        groups = np.zeros(n_col, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    n_grp = int(groups.max()) + 1
    counts = np.bincount(groups, minlength=n_grp).astype(np.int64)

    L = int(block_frames or (2 * max_lag + 1))
    L = max(L, max_lag + 1)
    L = min(L, n)
    ov = float(overlap if overlap is not None else 0.5)
    ov = min(max(ov, 0.0), 0.95)
    step = max(1, int(round(L * (1.0 - ov))))     # 默认 50% 重叠
    n_fft = 1 << int(np.ceil(np.log2(2 * L)))
    if not batch or batch <= 0:
        batch = max(1, int(FFT_WORKSPACE_BUDGET / (n_fft * 16)))
    batch = int(min(batch, n_col))

    n_lag = max_lag + 1
    num = np.zeros((n_lag, n_grp), dtype=np.float64)
    origins = np.zeros(n_lag, dtype=np.float64)

    for lo in range(0, n - max_lag, step):
        hi = min(lo + L, n)
        blk_frames = hi - lo
        if blk_frames <= max_lag:
            continue
        # 该块在每个 τ 上贡献的起点数：min(L_b − τ, n_lag)，只累加到 τ ≤ max_lag
        k_ = min(blk_frames, n_lag)
        origins[:k_] += (blk_frames - np.arange(k_))
        for c0 in range(0, n_col, batch):
            c1 = min(c0 + batch, n_col)
            sub = np.asarray(arr[lo:hi, c0:c1], dtype=np.float64)
            F = np.fft.rfft(sub, n=n_fft, axis=0)
            corr = np.fft.irfft(F * np.conjugate(F), n=n_fft, axis=0)[:blk_frames]
            del F
            cs = np.cumsum(sub * sub, axis=0)
            tot = cs[-1]
            tt = np.arange(blk_frames)
            p2 = cs[blk_frames - 1 - tt]
            p1 = tot - np.concatenate([np.zeros((1, sub.shape[1])), cs[:-1]], axis=0)
            nb = p1 + p2 - 2.0 * corr
            del cs, p1, p2, corr, sub
            nb = nb[:n_lag]                       # 只留 τ ≤ max_lag 的部分
            g = groups[c0:c1]
            if n_grp == 1:
                num[:, 0] += nb.sum(axis=1)
            else:
                for gi in range(n_grp):
                    m = (g == gi)
                    if m.any():
                        num[:, gi] += nb[:, m].sum(axis=1)
        if verbose:
            print(f"  [MSD] 分段块 {lo}–{hi}/{n}")
    return num, origins, counts


def _batch_worker(args):
    """多进程 worker：按文件名重新打开内存映射，只算自己那一段列。"""
    fname, shape, dtype, lo, hi, g, n, n_fft, n_grp = args
    arr = np.memmap(fname, dtype=np.dtype(dtype), mode="r", shape=tuple(shape))
    g = np.asarray(g, dtype=np.int64)
    tau = np.arange(n)
    blk = np.asarray(arr[:, lo:hi], dtype=np.float64)
    F = np.fft.rfft(blk, n=n_fft, axis=0)
    corr = np.fft.irfft(F * np.conjugate(F), n=n_fft, axis=0)[:n]
    del F
    cs = np.cumsum(blk * blk, axis=0)
    tot = cs[-1]
    p2 = cs[n - 1 - tau]
    p1 = tot - np.concatenate([np.zeros((1, blk.shape[1])), cs[:-1]], axis=0)
    nb = p1 + p2 - 2.0 * corr
    del cs, p1, p2, corr, blk
    out = np.zeros((n, n_grp), dtype=np.float64)
    for gi in range(n_grp):
        m = (g == gi)
        if m.any():
            out[:, gi] = nb[:, m].sum(axis=1)
    return out



def _molecule_ids(ag) -> np.ndarray | None:
    """原子级分子编号；无 ``molnums`` 时退回键图碎片，都没有则返回 ``None``。"""
    from .interface import _atom_molnums

    m = _atom_molnums(ag)
    if m is not None:
        return m
    try:
        if len(ag.bonds) == 0:
            return None
        allidx = np.asarray(ag.indices, dtype=np.int64)
        m = np.full(ag.n_atoms, -1, dtype=np.int64)
        for k, frag in enumerate(ag.fragments):
            m[np.searchsorted(allidx, np.asarray(frag.indices, dtype=np.int64))] = k
        return m if bool((m >= 0).all()) else None
    except Exception:  # noqa: BLE001
        return None


def _safe_masses(ag) -> np.ndarray:
    """质量数组；缺失或非正时退回单位质量（几何中心），并在调用方注明。"""
    try:
        w = np.asarray(ag.masses, dtype=float)
        if w.size == ag.n_atoms and bool(np.all(np.isfinite(w))) and bool(np.all(w > 0)):
            return w
    except Exception:  # noqa: BLE001
        pass
    return np.ones(ag.n_atoms, dtype=float)


def _frame_com(ag, mol, n_mol, box, weights) -> np.ndarray:
    """当前帧的**分子质心**坐标 ``(n_mol, 3)``（质量加权）。

    ⚠️ 这里**不做**逐原子最小镜像折叠（旧实现用 ``minimize_vectors(pos-ref)``
    把分子内原子折到第 1 个原子附近）。实测 PE-PP-MAPE 的单条 MAPE 链
    （311 原子、链长 ~50 Å、盒 40 Å 量级）：折叠会把"离第 1 个原子超过半盒"的
    原子折到错的镜像上，逐帧质心随机跳，分子质心 MSD 被抬成 35.7 Å²
    （原子口径 17.0 Å²，Forcite 17.0 Å²）—— 质心 MSD 反比原子 MSD 大，
    物理上不可能。MS 的 .trj 坐标本来就是**展开**的；被折回盒内的轨迹，
    `particle_series` 的跨帧展开已经在原子层面解开了。所以这里直接对
    （已展开的）逐帧坐标做加权平均即可。
    """
    pos = np.asarray(ag.positions, dtype=float)
    num = np.zeros((n_mol, 3), dtype=float)
    for d in range(3):
        num[:, d] = np.bincount(mol, weights=weights * pos[:, d], minlength=n_mol)
    den = np.bincount(mol, weights=weights, minlength=n_mol)
    den[den <= 0] = 1.0
    return num / den[:, None]


def particle_series(mdt, ag, frame_indices, *, object="molecule",
                    max_particles: int | None = None,
                    memory_budget: float | None = DEFAULT_MEMORY_BUDGET,
                    dtype=None, stream: bool | None = None,
                    scratch_dir: str | None = None,
                    workers: int = 0,
                    verbose: bool = False):
    """把原子组变成逐帧的"被追踪粒子"坐标序列 ``(n_frames, n_particles, 3)``。

    ``object="molecule"``（默认）→ 每个**分子质心**是一个粒子（质量加权）；
    ``object="atom"`` → 每个原子是一个粒子。

    返回 ``(series, boxes, meta)``；``boxes`` 是逐帧的 ``dimensions``。
    """
    u = mdt.universe
    idx = np.asarray(frame_indices, dtype=int)
    n_frame = int(idx.size)
    obj = str(object).lower()
    use_com = obj.startswith("mol") or obj in ("com", "分子", "分子质心")
    notes: list[str] = []

    # ⚠️ 这里**不再**用内存预算反推"粒子数上限"。老实现按
    # `budget / (帧×3×8)` 抽稀粒子，两个问题：
    #   1. 用户要的是"每个原子都参与"，抽稀直接违背需求；
    #   2. 抽稀是**原子级等间隔**取（`ag[::step]`），会把分子切碎 —— PEG
    #      200×30 原子按 1/3 抽稀后每条链只剩 10 个原子，"分子质心"成了残缺
    #      分子的质心，位移里混进内部振动，D 直接失真。
    # 现在只有**显式** max_particles 才会抽稀（且按分子整取），超大序列改用
    # 磁盘内存映射（下面 stream 分支），内存与粒子数无关。

    mol = _molecule_ids(ag) if use_com else None
    if use_com and mol is None:
        use_com = False
        notes.append("该组分没有分子编号（molnums）也无法从键图分组，"
                     "本轮退化为**按原子**追踪。")

    if use_com:
        uniq = np.unique(mol)
        n_part_all = int(uniq.size)
        keep = uniq
        if max_particles and n_part_all > max_particles:
            step = int(np.ceil(n_part_all / float(max_particles)))
            keep = uniq[::step]
            notes.append(f"分子数 {n_part_all:,} 超过**显式**上限 {int(max_particles):,}，"
                         f"每 {step} 个分子取 1 个，抽稀到 {keep.size:,} 个"
                         f"（按分子整取，不会切碎分子）。")
        mask = np.isin(mol, keep)
        ag_use = ag[mask]
        _, inv = np.unique(mol[mask], return_inverse=True)
        mol_use = np.asarray(inv, dtype=np.int64)
        n_part = int(keep.size)
        weights = _safe_masses(ag_use)
        try:
            w_chk = np.asarray(ag_use.masses, dtype=float)
            has_mass = (w_chk.size == ag_use.n_atoms
                        and bool(np.all(np.isfinite(w_chk))) and bool(np.all(w_chk > 0)))
        except Exception:  # noqa: BLE001
            has_mass = False
        if not has_mass:
            notes.append("该组分没有质量信息，质心退化为几何中心。")
    else:
        n_part_all = int(ag.n_atoms)
        if max_particles and n_part_all > max_particles:
            step = int(np.ceil(n_part_all / float(max_particles)))
            ag_use = ag[::step]
            notes.append(f"原子数 {n_part_all:,} 超过**显式**上限 {int(max_particles):,}，"
                         f"每 {step} 个取 1 个，抽稀到 {ag_use.n_atoms:,} 个。")
        else:
            ag_use = ag
        mol_use = None
        n_part = int(ag_use.n_atoms)
        weights = None

    if n_part == 0:
        raise ValueError("没有可追踪的粒子。")

    # 序列存储：默认 float32（省一半内存与 IO；FFT 前逐批转 float64 累积，
    # 精度由 selftest 与 float64 对照钉住）。超过预算就落盘做内存映射。
    dt = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
    nbytes = int(n_frame) * n_part * 3 * dt.itemsize
    use_stream = (bool(stream) if stream is not None
                  else bool(memory_budget and nbytes > float(memory_budget)))
    path = None
    if use_stream:
        import uuid

        d = scratch_dir or _scratch_dir()
        path = os.path.join(d, f"msd_{uuid.uuid4().hex[:8]}.dat")
        series = np.memmap(path, dtype=dt, mode="w+", shape=(n_frame, n_part, 3))
        mode = "memmap"
        notes.append(
            f"全粒子序列 {nbytes / 1e9:.2f} GB 超过内存预算 "
            f"{float(memory_budget or 0) / 1e9:.2f} GB，改用**磁盘内存映射**"
            f"（{os.path.basename(path)}）：粒子一个没少，只是不再一次性驻留内存。")
    else:
        series = np.empty((n_frame, n_part, 3), dtype=dt)
        mode = "ram"

    boxes: list = []
    from ..cancel import check as _cancel_check

    for k, i in enumerate(idx):
        _cancel_check()          # 用户取消 → 立刻停
        u.trajectory[int(i)]
        box = u.dimensions
        boxes.append(None if box is None else np.asarray(box, dtype=float))
        if use_com:
            series[k] = _frame_com(ag_use, mol_use, n_part, box, weights)
        else:
            series[k] = np.asarray(ag_use.positions, dtype=float)

    if hasattr(series, "flush"):
        series.flush()

    meta = {"object": "molecule" if use_com else "atom",
            "n_particles": n_part, "n_particles_total": n_part_all,
            "n_frames": n_frame, "notes": notes, "mode": mode,
            "series_bytes": int(nbytes), "path": path, "dtype": dt.name,
            "subsampled": bool(max_particles and n_part_all > max_particles)}
    if verbose:
        print(f"  [MSD] 追踪对象={meta['object']}，粒子 {n_part:,}"
              f"（总 {n_part_all:,}），帧 {n_frame:,}，存储 {mode}")
    return series, boxes, meta


def _unwrap_series(series, boxes, *, verbose: bool = False, inplace: bool = False):
    """跨帧追踪粒子：对相邻帧做**最小镜像位移**累加，返回连续轨迹与诊断。

    比 ``NoJump`` 更可控的地方是它会统计"位移超过最小盒高度一半"的粒子比例，
    该比例即最小镜像近似可能失效的比例——这种误差是**随机**的，会让 MSD 曲线
    依然平滑线性（R² 甚至接近 1）却整体失真，所以必须显式报出来。

    ``inplace=True``（磁盘内存映射时用）：直接改写 ``series``，不再分配第二份
    同样大小的数组 —— 否则"落盘省内存"就白省了。注意 ``series`` 必须是可写的
    ``np.memmap``（或普通 ndarray 且调用方愿意被改写）。
    """
    from MDAnalysis.lib.distances import minimize_vectors

    n = int(series.shape[0])
    out = series if inplace else np.empty_like(series)
    if n == 0:
        return out, {"最小镜像近似失效粒子数": 0, "检查的粒子-帧数": 0, "失效比例": 0.0}
    # ⚠️ inplace 时 `out is series`：`out[k-1]` 一旦被写成**展开后**的坐标，
    #    就再也不能拿它当"上一帧原始坐标"用了 —— 否则位移算成
    #    `raw[k] − unwrapped[k−1]`，误差会随帧数累积（实测 10 帧就差 0.43 Å，
    #    MSD 偏 2.5 Å²）。所以这里把**上一帧的原始坐标**单独存一份，
    #    并且**位移与累加都在 float64 里做**（只在写回 float32 内存映射时舍入一次），
    #    这样内存映射版与内存版的差别只剩 float32 存储精度（实测 ~1e-7 Å²）。
    prev_raw = np.array(series[0], dtype=np.float64)
    out[0] = series[0]
    n_big = 0
    n_tot = 0
    for k in range(1, n):
        # 必须**拷贝**（np.array 默认 copy）：dtype 相同时 asarray 会返回视图，
        # 而 `out[k] = ...` 马上就会覆盖 memmap 的同一块内存 → 视图变脏
        raw = np.array(series[k], dtype=np.float64)
        d = raw - prev_raw
        box = boxes[k] if k < len(boxes) else None
        if box is not None and bool(np.all(np.asarray(box[:3], dtype=float) > 0)):
            d = minimize_vectors(d, np.asarray(box, dtype=np.float32))
            h = float(np.min(_box_heights_from_dims(box)))
            mag = np.linalg.norm(d, axis=1)
            n_big += int(np.sum(mag > 0.5 * h))
            n_tot += mag.size
        out[k] = out[k - 1] + d
        prev_raw = raw
        if inplace and hasattr(out, "flush") and (k % 200 == 0):
            out.flush()
    if hasattr(out, "flush"):
        out.flush()
    diag = {
        "最小镜像近似失效粒子数": int(n_big),
        "检查的粒子-帧数": int(n_tot),
        "失效比例": float(n_big / n_tot) if n_tot else 0.0,
    }
    return out, diag


def msd_fft(series: np.ndarray) -> np.ndarray:
    """``(n_frames, n_particles)`` → MSD(τ)，对全部时间起点与全部粒子求平均。

    用快速相关算法（FFT）算，复杂度 ``O(N log N)``，**保留完整时间分辨率**
    （不做分块、不牺牲长 τ）。内部走 :func:`msd_numerator_by_group`
    （分块 + 按组求和），因此粒子数很大时会自动分批，不会一次吃掉十几 GB。

    为什么不用 ``tidynamics.msd``：它把输入的第二轴当作**空间分量**求和
    （源码里逐列 ``autocorrelation_1d`` 后相加，``rsq = Σ_d pos²``），因此只
    能算"单个粒子、多分量"的 MSD，无法一次对几百上千个分子求平均；逐粒子
    调用又需要 P 次 FFT。这里用同一套快速相关算法，但把 FFT 沿**时间轴批量**
    做一次算完所有粒子。数值上与暴力 ``(t₀, τ)`` 双重求和逐点一致（selftest
    与 tidynamics、暴力法双向校验）。

    估计量与暴力法相同（无偏，按实际样本数 ``N−τ`` 平均）：

    .. math:: \\mathrm{MSD}(\\tau) = \\frac{1}{N-\\tau}\\sum_{t=0}^{N-\\tau-1}
              |\\mathbf r_{t+\\tau}-\\mathbf r_t|^2
    """
    A = np.asarray(series)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError(f"msd_fft 需要 1D/2D 输入，得到 shape={A.shape}")
    n = int(A.shape[0])
    if n < 2:
        return np.zeros(n, dtype=float)
    num, counts = msd_numerator_by_group(A)
    tau = np.arange(n)
    den = (n - tau).astype(float) * float(counts[0] if counts.size else 1)
    return num[:, 0] / den


def unwrap_over_frames(mdt, ag, frame_indices, *, verbose: bool = False):
    """兼容入口：跨帧追踪原子，返回 ``(连续坐标, 诊断)``。

    现在内部统一走 :func:`particle_series` + :func:`_unwrap_series`（按原子）。
    """
    series, boxes, _ = particle_series(mdt, ag, frame_indices, object="atom",
                                       verbose=verbose)
    return _unwrap_series(series, boxes, verbose=verbose)


def _box_heights_from_dims(dims) -> np.ndarray:
    from MDAnalysis.lib.mdamath import triclinic_vectors

    m = np.asarray(triclinic_vectors(np.asarray(dims, dtype=float)), dtype=float)
    v = float(abs(np.linalg.det(m)))
    h = np.empty(3, dtype=float)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        area = float(np.linalg.norm(np.cross(m[j], m[k])))
        h[i] = v / area if area > 0 else np.inf
    return h


def compute_msd_axes(mdt, ag, selection: FrameSelection, *, object: str = "molecule",
                     remove_drift: bool = False, unwrap: bool = False,
                     max_particles: int | None = None,
                     max_lag: int | None = None, n_error_blocks: int = 5,
                     memory_budget: float | None = DEFAULT_MEMORY_BUDGET,
                     stream: bool | None = None, dtype=None, workers: int = 0,
                     max_lag_frames: int | None = None,
                     block_frames: int | None = None,
                     verbose: bool = False):
    """逐轴 MSD：返回 ``(lag_t, msd_dict, diag)``。

    ``msd_dict`` 的键：``total``（x+y+z）、``x`` / ``y`` / ``z``、
    ``para``（=x+y，平行于 z 轴的面内分量）、``perp``（=z）。

    为什么要把各轴分开：只有分子在某个方向上**取向有序**（界面吸附、拉伸、
    受限孔道）时，MSD 才会各向异性；``D∥/D⊥`` 是这类体系最直接的可观测量，
    而单一的 MSD 总值会把它平均掉。

    ``max_particles``：默认 ``None`` = **全部粒子参与**（1.0.2 起不再按内存
    预算自动抽稀）。真需要限制规模时显式给数，分子对象会**按分子整取**。
    ``memory_budget`` / ``stream``：序列超过预算时自动落到磁盘内存映射
    （粒子一个不少）；``dtype`` 默认 float32 存储、FFT 前逐批转 float64。
    ``workers``：>1 时用多进程并行 FFT 分块（仅内存映射模式生效）。

    ``remove_drift``：减去全部粒子的平均位移轨迹（消除整体漂移，例如 NPT
    下整盒的缓慢平移）。粒子数 < 4 时不做（少数粒子时"平均位移"本身就是
    被测对象，减掉会把信号一起减没）。
    """
    idx = np.asarray(selection.indices, dtype=int)
    if idx.size < 2:
        raise ValueError("MSD 至少需要 2 帧")

    series, boxes, meta = particle_series(mdt, ag, idx, object=object,
                                          max_particles=max_particles,
                                          memory_budget=memory_budget,
                                          stream=stream, dtype=dtype,
                                          workers=workers, verbose=verbose)
    on_disk = isinstance(series, np.memmap)
    # ---- 跨帧追踪：两种口径（用户实测：Materials Studio 的 .trj 坐标**已经是解包的**，
    #      再"补"一次最小镜像反而会把 >L/2 的位移改错 → 与 Forcite 差一截）。
    #      · ``unwrap=False``（**默认**，= MDAnalysis ``analysis.msd.EinsteinMSD`` 口径）：
    #        直接用文件里的坐标做差，不做任何 PBC 修正。Forcite 也是这个口径；
    #      · ``unwrap=True``：对相邻帧做最小镜像累加，给"坐标被折回盒内"的轨迹用。
    #        判据是 diag 里的"最小镜像近似失效比例"：≈0 说明折与不折一样（无副作用），
    #        明显偏离 0 却仍要折，就可能把已经解包的轨迹改坏。
    if unwrap:
        unwrapped, diag = _unwrap_series(series, boxes, verbose=verbose,
                                         inplace=on_disk)
    else:
        unwrapped = series
        diag = {"最小镜像近似失效粒子数": 0, "检查的粒子-帧数": 0, "失效比例": 0.0}
    drift_removed = False
    if remove_drift and unwrapped.shape[1] >= 4:
        if on_disk:
            # 逐帧减均值：内存映射上不能一次性做 (F,P,3) 的临时副本
            for k in range(unwrapped.shape[0]):
                unwrapped[k] -= unwrapped[k].mean(axis=0, keepdims=True)
            unwrapped.flush()
        else:
            unwrapped = unwrapped - unwrapped.mean(axis=1, keepdims=True)
        drift_removed = True

    # 一次 FFT 算完 x/y/z 三个分量：把三组列拼在一起（列 → 组号 0/1/2），
    # 再按组求和。老实现三个分量各跑一次 msd_fft（还有误差分块），
    # 一次分析要 3×(1+块数) 遍 FFT；这里降到 1/3，结果逐位相同。
    n_p = int(unwrapped.shape[1])
    # ⚠️ 列的顺序必须与 `reshape` 出来的实际布局一致：`(F, P, 3)` 是 **C 序**，
    # 展平成 `(F, 3P)` 后是**逐粒子**排的 `[p1x,p1y,p1z, p2x,p2y,p2z, …]`，
    # 所以列 c 属于分量 `c % 3`（用 `tile`），**不是** `[x…, y…, z…]` 那种分量排布
    # （那种要用 `repeat`）。写错的话总 MSD 还是对的（三项都加起来除以粒子数），
    # 但 para/perp 会互相串味 —— 实测各向异性 D∥/D⊥ 从 4.0 变成 1.04，
    # 正是 `test/selftest.py::test_msd_anisotropy_reports_in_plane_vs_normal` 抓到的。
    col_groups = np.tile(np.arange(3, dtype=np.int64), n_p)
    A = unwrapped.reshape(unwrapped.shape[0], n_p * 3)
    assert col_groups.size == A.shape[1]
    n_lag_max = A.shape[0] if max_lag_frames is None else int(max_lag_frames) + 1
    n_lag_max = max(2, min(int(n_lag_max), A.shape[0]))
    segmented = bool(max_lag_frames is not None) and n_lag_max < A.shape[0]
    if segmented:
        # 分段（滚动窗口）：内存恒定、长轨迹可跑，见函数文档
        num, origins, counts = msd_numerator_by_group_segmented(
            A, groups=col_groups, max_lag=n_lag_max - 1,
            block_frames=block_frames, verbose=verbose)
        tau = np.arange(n_lag_max)
        den = np.maximum(origins, 1.0)

        def _msd_of(cols) -> np.ndarray:
            s = np.zeros(n_lag_max, dtype=np.float64)
            for c in cols:
                s += num[:, c]
            return s / (den * max(n_p, 1))
        algorithm = (f"FFT 分段快速相关（块 {int(block_frames or 2 * (n_lag_max - 1) + 1)} 帧、"
                     f"50% 重叠，τ ≤ {n_lag_max - 1} 帧）")
    else:
        num, counts = msd_numerator_by_group(A, groups=col_groups, workers=workers)
        tau = np.arange(A.shape[0])
        den = (A.shape[0] - tau).astype(float)

        def _msd_of(cols) -> np.ndarray:
            """分量组求和后除以**粒子数**（不是列数）。

            ⚠️ 必须除以 ``n_p`` 而不是 ``3·n_p``：总 MSD 的定义是
            ``(1/n_p)Σ_p[(Δx²+Δy²+Δz²)]``，即"逐粒子三分量之和再对粒子平均"，
            而不是"三个分量各自平均后相加再除 3"。分母用错会让 MSD 与 D **小 3 倍**
            —— 这个坑我在改"分量合并"时真踩过（三条轴分开算是对的，合并后分母
            写成列数就错了），所以这里单独写明，并靠 selftest 与暴力法对照钉住。
            """
            s = np.zeros(A.shape[0], dtype=np.float64)
            for c in cols:
                s += num[:, c]
            return s / (den * max(n_p, 1))
        algorithm = "FFT 快速相关（分量合并 + 分列分块）"

    msd = {"x": _msd_of([0]), "y": _msd_of([1]), "z": _msd_of([2])}
    msd["total"] = _msd_of([0, 1, 2])
    msd["para"] = _msd_of([0, 1])
    msd["perp"] = _msd_of([2])

    times = np.asarray(selection.times_ps, dtype=float)
    n_lag = times.size if max_lag is None else min(times.size, int(max_lag) + 1)
    # 分段模式只算到 max_lag_frames：时间轴必须跟着截断，否则 lag_t 比 MSD 长，
    # 后面拟合时两个数组广播不上（实测 ValueError: shapes (6001,) (301,)）。
    n_lag = min(n_lag, int(np.asarray(msd["total"]).size))
    lag_t = times[:n_lag] - times[0]
    for k in list(msd):
        msd[k] = np.asarray(msd[k][:n_lag], dtype=float)

    # ---- 误差分块：把时间轴切块，每块**独立**算一条 MSD。
    # 扩散系数的标准误由这些块的 D 散布给出（经验值），比"假设一个残差相关矩阵"
    # 更可靠。块要足够长才落在同一个拟合窗口里，所以块数按帧数自适应。
    block_curves: dict[str, list] = {k: [] for k in ("total", "para", "perp")}
    n_blk = int(n_error_blocks or 0)
    if n_blk >= 2 and unwrapped.shape[0] >= 8:
        edges = np.linspace(0, unwrapped.shape[0], n_blk + 1).astype(int)
        for b in range(n_blk):
            lo_b, hi_b = int(edges[b]), int(edges[b + 1])
            if hi_b - lo_b < 4:
                continue
            sub = np.asarray(unwrapped[lo_b:hi_b], dtype=np.float64)
            Ab = sub.reshape(sub.shape[0], n_p * 3)
            nb, cb = msd_numerator_by_group(Ab, groups=col_groups, workers=workers)
            db = (Ab.shape[0] - np.arange(Ab.shape[0])).astype(float)

            def _blk(cols) -> np.ndarray:
                s = np.zeros(Ab.shape[0], dtype=np.float64)
                for c in cols:
                    s += nb[:, c]
                return s / (db * max(n_p, 1))   # 同样除以粒子数（见上面 _msd_of）

            mb = {"total": _blk([0, 1, 2]), "para": _blk([0, 1]), "perp": _blk([2])}
            # 分段模式下主曲线只到 n_lag：块的曲线也要截到同一长度，
            # 否则拟合时 lag_t(301) 与块曲线(1200)广播不上
            n_b = min(int(mb["total"].size), int(lag_t.size))
            bt = lag_t[:n_b]
            for k in block_curves:
                block_curves[k].append((bt, np.asarray(mb[k][:n_b], dtype=float)))


    diag = dict(diag)
    diag["误差分块_curves"] = block_curves
    diag["误差分块数"] = len(block_curves.get("total", []))
    diag.update({"对象": meta["object"], "粒子数": int(meta["n_particles"]),
                 "粒子数_总": int(meta["n_particles_total"]),
                 "已抽稀": bool(meta.get("subsampled")),
                 "存储": (f"{'磁盘内存映射' if meta.get('mode') == 'memmap' else '内存'}"
                          f"（{meta.get('dtype', '?')}，"
                          f"{float(meta.get('series_bytes', 0)) / 1e9:.2f} GB）"),
                 "去漂移": bool(drift_removed), "算法": "FFT 快速相关（分量合并 + 分块）",
                 "说明": list(meta["notes"])})
    # 清理落盘的序列（分析结束就不需要了）
    if on_disk and meta.get("path"):
        try:
            unwrapped.flush()
            del unwrapped, series
            os.remove(meta["path"])
        except Exception:                                     # noqa: BLE001
            pass
    return lag_t, msd, diag


def compute_msd(mdt, ag, selection: FrameSelection, *, max_lag: int | None = None,
                object: str = "atom", remove_drift: bool = False,
                unwrap: bool = False,
                max_particles: int | None = None,
                memory_budget: float | None = DEFAULT_MEMORY_BUDGET,
                stream: bool | None = None, dtype=None, workers: int = 0,
                verbose: bool = False):
    """计算原子组的总 MSD，返回 ``(lag_times_ps, msd_Å², diagnostics)``。

    默认仍按**原子**追踪（保持既有调用方的语义）；``analyze_msd`` 默认按
    **分子质心**追踪，那才是"分子扩散"的正确对象。
    """
    lag_t, msd, diag = compute_msd_axes(mdt, ag, selection, object=object,
                                        remove_drift=remove_drift, unwrap=unwrap,
                                        max_particles=max_particles,
                                        max_lag=max_lag,
                                        memory_budget=memory_budget,
                                        stream=stream, dtype=dtype,
                                        workers=workers, verbose=verbose)
    return lag_t, msd["total"], diag


def _fit_slope_with_error(t, y, method: str = "gls"):
    """线性拟合斜率并给出**标准误**，返回 ``(slope, intercept, se, r2, rho)``。

    为什么不能只用 ``np.polyfit(cov=True)``：那个误差假设各点残差独立，而 MSD 在
    不同 τ 上是**同一段轨迹算出来的**，强相关，因此会系统性**低估**误差。

    ``method="gls"`` 时用**本曲线自己估出来的**残差滞后 1 自相关系数 ρ 做有效样本量
    修正：``se_eff = se_ols·sqrt((1+ρ)/(1−ρ))``（ρ=0 时退化为 OLS），并把 ρ 一起报出来。
    强调"估出来"是因为 MDTransport 那类实现是**假设**一个相关矩阵（如
    ``2^{-|i-j|/2}``）；从数据估计至少是自洽的，而且 ρ 可被检查。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    dof = max(n - 2, 1)
    sxx = float(np.sum((t - t.mean()) ** 2))
    se = float(np.sqrt(float(np.sum(resid ** 2)) / dof / sxx)) if sxx > 0 else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    r0 = resid - resid.mean()
    den = float(np.sum(r0 * r0))
    rho = float(np.clip(np.sum(r0[1:] * r0[:-1]) / den, -0.99, 0.99)) if den > 0 else 0.0
    if str(method).lower() == "gls" and rho > 0:
        se *= float(np.sqrt((1.0 + rho) / (1.0 - rho)))
    return float(slope), float(intercept), float(se), float(r2), rho


def diffusion_coefficient(lag_t: np.ndarray, msd: np.ndarray,
                          fit_fraction: tuple[float, float] = (0.3, 1.0),
                          dim: int = 3, fit_method: str = "gls",
                          block_curves=None) -> dict:
    """由 MSD 的线性区拟合扩散系数（含**标准误**）。

    ``MSD = 2·dim·D·t``（三维时 ``MSD = 6Dt``）。返回 ``D``（Å²/ps 与 m²/s）、
    标准误、拟合区间、对数-对数斜率 ``α``（MSD ∝ t^α）与可靠性判断。

    标准误怎么来
    ------------
    1. **分块平均（首选）**：把轨迹切成若干时间块，每块**独立**算一条 MSD 并各自
       拟合，取各块 D 的标准误 ``std(D_i)/√n``。这是**经验散布**，不依赖任何模型假设。
    2. 退路（块数 < 3，例如轨迹太短）：拟合协方差（``fit_method="gls"`` 时再乘实测
       AR(1) 修正）。此时误差偏乐观，结果里会写明来源。

    物理约束
    --------
    MSD 在真实体系里是**单调不减**的；扩散系数绝不能为负。如果拟合斜率 ≤ 0，
    说明该组分的 MSD 是一条平坦的曲线——粒子基本不动（玻璃态/被约束/结晶相），
    或者帧间隔太大导致曲线不具扩散性。这时返回 ``Nan`` 并说明原因，
    而不是给出一个看似精确、实则无意义的负数。
    """
    t = np.asarray(lag_t, dtype=float)
    y = np.asarray(msd, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & (t > 0)
    out = {"D (Å²/ps)": float("nan"), "D (m²/s)": float("nan"),
           "D 标准误 (m²/s)": float("nan"), "D 相对标准误": float("nan"),
           "标准误来源": "", "残差 AR(1) ρ": float("nan"),
           "拟合区间 (ps)": "", "R²": float("nan"), "α (log-log 斜率)": float("nan"),
           "拟合可靠性": ""}
    if ok.sum() < 3:
        out["拟合可靠性"] = "点太少，无法拟合"
        return out
    t_ok, y_ok = t[ok], y[ok]
    lo = fit_fraction[0] * t_ok[-1]
    hi = fit_fraction[1] * t_ok[-1]
    m = (t_ok >= lo) & (t_ok <= hi)
    if m.sum() < 3:
        m = np.ones_like(t_ok, dtype=bool)
    tt, yy = t_ok[m], y_ok[m]
    slope, _intercept, se_slope, r2, rho = _fit_slope_with_error(tt, yy, fit_method)
    out["拟合区间 (ps)"] = f"{tt[0]:.1f} – {tt[-1]:.1f}"
    out["R²"] = r2
    out["残差 AR(1) ρ"] = rho
    pos = (t_ok > 0) & (y_ok > 0)
    if pos.sum() >= 3:
        out["α (log-log 斜率)"] = float(np.polyfit(np.log(t_ok[pos]),
                                                   np.log(y_ok[pos]), 1)[0])

    if slope <= 0:
        out["拟合可靠性"] = ("MSD 曲线平坦或下降，粒子基本没有扩散"
                             "（玻璃态/被约束/结晶相，或帧间隔过大）——"
                             "不给出扩散系数")
        out["标准误来源"] = "未拟合（斜率 ≤ 0）"
        return out

    d_a2ps = float(slope) / (2.0 * dim)
    out["D (Å²/ps)"] = d_a2ps
    out["D (m²/s)"] = d_a2ps * A2_PER_PS_TO_M2_PER_S

    # ---- 标准误：优先用分块的经验散布
    d_blocks: list[float] = []
    for item in (block_curves or []):
        try:
            bt, by = np.asarray(item[0], dtype=float), np.asarray(item[1], dtype=float)
        except Exception:  # noqa: BLE001
            continue
        m2 = np.isfinite(bt) & np.isfinite(by) & (bt >= lo) & (bt <= hi)
        if m2.sum() < 3:
            continue
        s_i = _fit_slope_with_error(bt[m2], by[m2], method="ols")[0]
        if np.isfinite(s_i):
            d_blocks.append(float(s_i) / (2.0 * dim))
    se_D = float("nan")
    if len(d_blocks) >= 3:
        arr = np.asarray(d_blocks, dtype=float)
        se_D = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        out["分块 D 个数"] = int(arr.size)
        out["标准误来源"] = f"分块平均（{arr.size} 块各自独立拟合）"
    elif np.isfinite(se_slope):
        se_D = float(se_slope) / (2.0 * dim)
        out["标准误来源"] = ("拟合协方差 × 实测 AR(1) 修正（分块不足 3 块，误差偏乐观）"
                        if (str(fit_method).lower() == "gls" and rho > 0)
                        else "拟合协方差（残差独立假设，会低估；分块不足 3 块）")
    if np.isfinite(se_D):
        out["D 标准误 (m²/s)"] = se_D * A2_PER_PS_TO_M2_PER_S
        out["D 相对标准误"] = float(se_D / d_a2ps) if d_a2ps > 0 else float("nan")

    alpha = out["α (log-log 斜率)"]
    if np.isfinite(r2) and r2 < 0.5:
        out["拟合可靠性"] = "拟合优度低（R²<0.5），D 只能作定性参考"
    elif np.isfinite(alpha) and not (0.8 <= alpha <= 1.2):
        out["拟合可靠性"] = (f"MSD∝t^{alpha:.2f} 偏离扩散标度（α≈1），"
                             f"D 只能作定性参考")
    else:
        out["拟合可靠性"] = "良好（线性区拟合正常）"
    return out


@register("msd", "均方位移 MSD")
def analyze_msd(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                fit_fraction: tuple[float, float] = (0.3, 1.0),
                dim: int = 3, object: str = "atom",
                remove_drift: bool = False, unwrap: bool = False,
                max_particles: int | None = None,
                fit_method: str = "gls",
                n_error_blocks: int = 5,
                memory_budget: float | None = DEFAULT_MEMORY_BUDGET,
                stream: bool | None = None, dtype=None, workers: int = 0,
                tau_intervals: int | None = None,
                max_lag: int | None = None,
                object_by_group: Mapping[str, str] | None = None,
                max_lag_frames: int | None = None,
                block_frames: int | None = None,
                verbose: bool = False) -> AnalysisResult:
    """均方位移 MSD（默认追踪对象：**原子**；跨帧用文件坐标，不额外展开）。

    参数
    ----
    groups
        ``{组分名: AtomGroup}``，可一次比较多个组分（例如界面区 vs 体相区）。
    object
        ``"atom"``（**默认**）→ 每个原子各算一个粒子 —— 这是 Materials Studio
        Forcite 以及 MDAnalysis ``analysis.msd.EinsteinMSD`` 的口径；
        ``"molecule"`` → 每个分子取**质心**作为一个被追踪粒子（"分子的平动扩散"，
        数值会比原子口径大，且与 Forcite 的 MSD 不可直接对比）。
    unwrap
        ``False``（**默认**）→ 直接用文件里的坐标做差。MS 的 ``.trj`` 坐标通常
        **已经是解包的**，再补一次最小镜像反而会把 >L/2 的位移改错（用户实测：
        graphite 27 与 Forcite 对不上就是这个）。``True`` → 对相邻帧做最小镜像
        累加，给"坐标被折回盒内"的轨迹用（判据看诊断里的"最小镜像近似失效比例"）。
    max_particles
        ``None``（默认）= **全部粒子参与**。1.0.2 起不再按内存预算自动抽稀；
        序列过大时改用磁盘内存映射（见 ``memory_budget`` / ``stream``）。
    max_lag_frames
        ``None``（默认）= 完整时间分辨率（一次 FFT 覆盖整条时间轴）。
        给正整数则切到**分段（滚动窗口）**模式：单次 FFT 只覆盖
        ``block_frames``（默认 ``2·max_lag_frames+1``）帧、相邻块 50% 重叠。
        **它的作用是省内存/支持流式**（常数内存，长轨迹也能跑），不是加速 ——
        实测 1001 帧时反而略慢（47.9 s → 57.4 s）。要省时间请减少帧数。
        代价是 τ 越接近块长样本越少，拟合窗口应落在 τ ≤ max_lag_frames 之内。
    object_by_group
        ``{组分名: "molecule"|"atom"}`` —— **逐组**覆盖 ``object``。
        「分析集合（Set）」用得到：`set:C`（只取碳原子）必须按**原子**追踪，
        而 `set:链状分子`（整分子选中）应该按**分子质心**追踪。
    tau_intervals
        **可选的 τ 上限**：把总时长均分成 ``tau_intervals`` 个区间，MSD 只画/只拟合到
        **一个区间的长度**（例如 5 → τ_max = 总时长 / 5）。
        **默认 ``None`` = 用满整条轨迹**（用户要求：6000 ps 的轨迹就该看到 0–6000 ps，
        不要悄悄截成 1200 ps）。想收缩范围时再在界面填「时间区间数」
        （5 = 每段 T/5，10 = T/10）。
    max_lag
        直接指定 τ 上限（帧数，不经过 ``tau_intervals`` 换算）；``None`` 时由
        ``tau_intervals`` 决定。两者都给时 ``max_lag`` 优先。
    block_frames
        分段模式下的块长（帧）；``None`` 时取 ``2·max_lag_frames+1``。
    remove_drift
        是否减去全体粒子的平均位移轨迹（消除整体漂移）。粒子数 < 4 时不做。
        **默认关**：Materials Studio Forcite 的 MSD 不去漂移，默认关掉才能与它
        逐点对齐（用户实测 PE50：关掉差 0.000%，开着差 −0.013%…−0.187%）。
        NPT 体系里整盒缓慢平移会把 MSD 抬高，那时再打开（界面上是「减去整体漂移」
        复选框）。
    fit_fraction
        线性区拟合区间（相对最大 lag 时间的比例）。
    dim
        总 MSD 的拟合维度，三维取 3（``MSD = 6Dt``）；``D∥`` 用 2、``D⊥`` 用 1。
    """
    if not groups:
        raise ValueError("必须提供至少一个组分")

    # ---- τ 上限：默认把总时长**均分成 5 个区间**，只用到其中一个区间的长度
    n_frame = int(len(selection.indices))
    tau_cap: int | None = max_lag
    tau_note = ""
    try:
        n_int = int(tau_intervals or 0)
    except (TypeError, ValueError):
        n_int = 0
    if tau_cap is None and n_int >= 1 and n_frame >= 2:
        tau_cap = max(1, (n_frame - 1) // n_int)
        # 帧数太少时不截断：6 帧按 5 段切只剩 τ=1 一个点，拟合直接失效。
        if tau_cap < 8 or n_frame < 20:
            tau_note = (f"帧数只有 {n_frame}，未按总时长均分 {n_int} 段截断 τ"
                        f"（否则 τ 只剩 {tau_cap} 步，拟合没有意义）")
            tau_cap = None
        else:
            tau_note = (
                f"τ 上限 = 总时长 / {n_int} 段 = {tau_cap} 帧"
                f"（共 {n_frame - 1} 个 τ 步，占整条轨迹的"
                f" {tau_cap / float(n_frame - 1) * 100:.0f}%）")

    res = AnalysisResult(
        name="msd",
        title="均方位移 MSD",
        meta={"n_frames": n_frame,
              "fit_fraction": list(map(float, fit_fraction)), "dim": int(dim),
              "object": str(object), "remove_drift": bool(remove_drift),
              "tau_intervals": n_int, "tau_max_frames": tau_cap},
    )
    # 坐标轴：**全部线性**（用户要求）。
    # 演进过程：①原来是双对数；②用户要求纵轴改线性 → 横轴仍是对数（界面上写着
    # "对数轴（仅横轴）"）；③用户实测后仍认为"MSD 随时间变化这张图没用线性图表"
    # —— 因为横轴还是 10⁰/10¹/10² 的对数刻度，整张图看起来依旧是双对数图。
    # 现在把第一张图（含 τ=0 那个 MSD=0 的点）彻底改成线性-线性：τ=0 的点在对数轴上
    # 画不出来，改线性后"全部时间点"都在图上。
    # 想看双对数（判断幂律指数 α）请用「各轴分解」面板的数据自行换轴，或告诉我加回。
    res.panels = [
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)", title="MSD 随时间间隔变化",
              xscale="linear", yscale="linear"),
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)", title="MSD 线性区拟合",
              xscale="linear", yscale="linear"),
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)",
              title="各轴分解与各向异性（x/y/z 及 ∥、⊥）",
              xscale="linear", yscale="linear"),
        Panel(xlabel="时间间隔 τ (ps)", ylabel="MSD (Å²)",
              title="MSD 幂律（log–log，α=1 参考线）",
              xscale="log", yscale="log"),
    ]

    fail_by_group: dict[str, float] = {}
    per_group_obj = dict(object_by_group or {})
    for name, ag in groups.items():
        if ag is None or ag.n_atoms == 0:
            continue
        g_object = str(per_group_obj.get(name) or object)
        lag_t, msd, diag = compute_msd_axes(mdt, ag, selection, object=g_object,
                                            remove_drift=remove_drift, unwrap=unwrap,
                                            max_particles=max_particles,
                                            n_error_blocks=n_error_blocks,
                                            memory_budget=memory_budget,
                                            stream=stream, dtype=dtype,
                                            workers=workers,
                                            max_lag=tau_cap,
                                            max_lag_frames=max_lag_frames,
                                            block_frames=block_frames,
                                            verbose=verbose)
        bl = diag.get("误差分块_curves") or {}
        if diag.get("粒子数", 0) < 2:
            res.add_notes(
                f"{name} 只追踪到 {diag.get('粒子数')} 个粒子，MSD 没有集合平均，"
                f"结果只是单条轨迹的位移平方，统计意义有限。")
        fit = diffusion_coefficient(lag_t, msd["total"], fit_fraction=fit_fraction,
                                    dim=dim, fit_method=fit_method,
                                    block_curves=bl.get("total"))
        fit_para = diffusion_coefficient(lag_t, msd["para"], fit_fraction=fit_fraction,
                                         dim=2, fit_method=fit_method,
                                         block_curves=bl.get("para"))
        fit_perp = diffusion_coefficient(lag_t, msd["perp"], fit_fraction=fit_fraction,
                                         dim=1, fit_method=fit_method,
                                         block_curves=bl.get("perp"))
        fail_by_group[name] = float(diag.get("失效比例", 0.0) or 0.0)

        # 「MSD 线性区拟合」图上的覆盖层（拟合区间 / 斜率 / R² / D）由前端画，
        # 数值以这里算的为准；前端只有在用户**自定义区间**时才自己重算（最小二乘）。
        # ⚠️ 放 summary（序列化会保留标量），不要放 meta —— meta 里的 dict 会被
        #    序列化层过滤掉（dict/list/tuple 一律丢）。
        _tf = np.asarray(lag_t, dtype=float)
        _tok = _tf[np.isfinite(_tf) & (_tf > 0)]
        _tmax = float(_tok[-1]) if _tok.size else float("nan")
        try:
            _slope_a2ps = float(fit.get("D (Å²/ps)", np.nan)) * 2.0 * float(dim)
        except Exception:                                    # noqa: BLE001
            _slope_a2ps = float("nan")
        res.summary[f"{name} 拟合起点 (ps)"] = float(fit_fraction[0] * _tmax)
        res.summary[f"{name} 拟合终点 (ps)"] = float(fit_fraction[1] * _tmax)
        res.summary[f"{name} τ 上限 (ps)"] = _tmax
        res.summary[f"{name} 拟合斜率 (Å²/ps)"] = _slope_a2ps
        res.summary[f"{name} 拟合方法"] = str(fit_method)

        obj_label = "分子质心" if diag.get("对象") == "molecule" else "原子"
        res.add_curve(f"{name} ({obj_label}, {diag.get('粒子数', 0):,} 个)",
                      lag_t, msd["total"], panel=0, group=name)

        # 线性区的拟合直线（沿用旧口径，画在同一张图上便于目视检查）
        t = np.asarray(lag_t, dtype=float)
        y_tot = np.asarray(msd["total"], dtype=float)
        ok = np.isfinite(t) & np.isfinite(y_tot) & (t > 0)
        if ok.sum() >= 3:
            t_ok, y_ok = t[ok], y_tot[ok]
            m = ((t_ok >= fit_fraction[0] * t_ok[-1])
                 & (t_ok <= fit_fraction[1] * t_ok[-1]))
            if m.sum() < 3:
                m = np.ones_like(t_ok, dtype=bool)
            slope = np.polyfit(t_ok[m], y_ok[m], 1)
            res.add_curve(f"{name} 线性拟合", t_ok[m], np.polyval(slope, t_ok[m]),
                          panel=1, group=name)
        res.add_curve(f"{name}", lag_t, y_tot, panel=1, group=name)

        # 各轴分解：各向异性是界面/受限体系最直接的可观测量
        for key, clabel in (("x", f"{name} MSD_x"), ("y", f"{name} MSD_y"),
                            ("z", f"{name} MSD_z"), ("para", f"{name} MSD∥ (x+y)"),
                            ("perp", f"{name} MSD⊥ (z)")):
            res.add_curve(clabel, lag_t, msd[key], panel=2, group=name)

        # log–log：幂律指数 α 的判据（原来在「扩散系数 D」那一项里，现并入 MSD）
        _pos = np.isfinite(t) & np.isfinite(y_tot) & (t > 0) & (y_tot > 0)
        if _pos.sum() >= 3:
            res.add_curve(f"{name}", t[_pos], y_tot[_pos], panel=3, group=name)

        for k, v in fit.items():
            res.summary[f"{name} {k}"] = v
        if np.isfinite(fit.get("D 标准误 (m²/s)", np.nan)):
            res.add_notes(
                f"{name}：D = {fit['D (m²/s)']:.3g} ± {fit['D 标准误 (m²/s)']:.3g} m²/s"
                f"（相对 {fit['D 相对标准误'] * 100:.1f}%）。标准误来源："
                f"{fit.get('标准误来源') or '未说明'}。")
        res.summary[f"{name} 原子数"] = int(ag.n_atoms)
        res.summary[f"{name} 追踪对象"] = obj_label
        res.summary[f"{name} 追踪粒子数"] = int(diag.get("粒子数", 0))
        res.summary[f"{name} 追踪粒子数（总）"] = int(diag.get("粒子数_总",
                                                              diag.get("粒子数", 0)))
        res.summary[f"{name} 是否抽稀过粒子"] = "是" if diag.get("已抽稀") else "否（全部参与）"
        res.summary[f"{name} 序列存储"] = str(diag.get("存储") or "")
        res.summary[f"{name} 去漂移"] = "是" if diag.get("去漂移") else "否"
        res.summary[f"{name} 跨帧追踪"] = (
            "最小镜像累加（适合坐标被折回盒内的轨迹）" if unwrap
            else "文件坐标（不展开；Forcite / EinsteinMSD 口径）")
        res.summary[f"{name} 算法"] = str(diag.get("算法", ""))
        res.summary[f"{name} τ 上限 (ps)"] = float(lag_t[-1]) if lag_t.size else float("nan")
        res.summary[f"{name} τ 轴点数"] = int(lag_t.size)
        res.summary[f"{name} MSD(最大 τ) (Å²)"] = float(y_tot[-1]) if y_tot.size else float("nan")
        res.summary[f"{name} MSD∥(最大 τ) (Å²)"] = float(msd["para"][-1]) if lag_t.size else float("nan")
        res.summary[f"{name} MSD⊥(最大 τ) (Å²)"] = float(msd["perp"][-1]) if lag_t.size else float("nan")
        res.summary[f"{name} D∥ (m²/s)"] = fit_para["D (m²/s)"]
        res.summary[f"{name} D⊥ (m²/s)"] = fit_perp["D (m²/s)"]
        d_para, d_perp = fit_para["D (m²/s)"], fit_perp["D (m²/s)"]
        ratio = (d_para / d_perp if (np.isfinite(d_para) and np.isfinite(d_perp)
                                    and d_perp > 0) else float("nan"))
        res.summary[f"{name} 各向异性 D∥/D⊥"] = ratio
        res.summary[f"{name} α∥ (log-log 斜率)"] = fit_para["α (log-log 斜率)"]
        res.summary[f"{name} α⊥ (log-log 斜率)"] = fit_perp["α (log-log 斜率)"]

        fail = float(diag.get("失效比例", 0.0) or 0.0)
        res.summary[f"{name} 最小镜像失效比例"] = fail
        # 最小镜像追踪的可靠性必须并入总判定：失效比例高时，被错误回绕的粒子贡献的
        # 是**随机**大位移，会让 MSD 看起来依然线性（R² 甚至接近 1）却整体失真。
        if fail > 0.05:
            base = str(res.summary.get(f"{name} 拟合可靠性", "") or "")
            if fail > 0.20:
                tag = (f"最小镜像失效比例 {fail * 100:.0f}%（严重）——"
                       f"帧间隔内多数粒子的位移已超过最小盒高的一半，"
                       f"跨周期追踪不可靠，D 很可能被污染，**不建议引用**；"
                       f"请用更密的帧间隔（该体系建议 ≤ 数百 ps）重算")
            else:
                tag = (f"最小镜像失效比例 {fail * 100:.1f}%，D 可能偏离真值，"
                       f"建议缩小帧间隔复核")
            res.summary[f"{name} 拟合可靠性"] = f"{base}；{tag}" if base else tag
        if ag.n_atoms < 20:
            res.add_notes(
                f"{name} 只有 {ag.n_atoms} 个原子，其扩散系数与 MSD 曲线的统计误差"
                f"很大，只能作为定性参考。"
            )
        for extra in diag.get("说明", []) or []:
            res.add_notes(f"{name}：{extra}")
        if np.isfinite(ratio):
            if 0.8 <= ratio <= 1.25:
                res.add_notes(
                    f"{name} 的扩散是**近似各向同性**的（D∥/D⊥ = {ratio:.2f}）。")
            else:
                res.add_notes(
                    f"{name} 的扩散**各向异性**明显：D∥/D⊥ = {ratio:.2f}"
                    f"（D∥={d_para:.3g}、D⊥={d_perp:.3g} m²/s）。"
                    f"这通常意味着取向有序（界面吸附、拉伸或受限孔道），"
                    f"只报总 D 会把这个信息平均掉。")

    bad_fail = [n for n, f in fail_by_group.items() if f > 0.20]
    # α=1 参考线（与旧「扩散系数 D」里那条同口径：过第一个点、斜率 1）
    try:
        _ref = res.curves_of(3)[0]
        _rx = np.asarray(_ref.x, dtype=float)
        _ry = np.asarray(_ref.y, dtype=float)
        if _rx.size >= 2 and _ry.size >= 2 and _ry[0] > 0:
            res.add_curve("参考斜率 α=1", _rx, _ry[0] * (_rx / _rx[0]),
                          kind="step", panel=3)
    except Exception:                                        # noqa: BLE001
        pass
    res.add_notes("MSD = <|r(t+τ) − r(t)|²>，对全部时间起点与全部粒子求平均。")
    if tau_note:
        res.add_notes(tau_note + "。这样做是为了让横轴范围合理：τ 接近整条轨迹长度时"
                      "只有极少数时间起点参与平均，曲线末端的统计噪声很大，"
                      "而且分子可能早已跑出盒子。想用满整条轨迹就把「时间区间数」设为 1"
                      "（或留空 = 不限制）。")
    if max_lag_frames:
        res.add_notes(f"τ 上限还受「分段模式」限制（max_lag_frames={max_lag_frames}）。")
    res.add_notes("追踪对象可选：**原子**（= Forcite / MDAnalysis EinsteinMSD 口径，"
                  "默认）或**分子质心**（只留平动扩散）。选「分子质心」时曲线整体偏小"
                  "（实测 PE-PP-MAPE 单链：原子口径 t=1 ps 0.961、t=999 ps 21.516，"
                  "与 Forcite 逐点一致；分子质心同一条链是 2.873 / 75.707 —— "
                  "两者不能直接对比）。")
    res.add_notes("MSD 用 FFT 快速相关算法一次算完，保留完整时间分辨率"
                  "（不做分块、不牺牲长 τ）。")
    res.add_notes("跨周期性边界通过「最小镜像位移累加」追踪粒子，相当于把分子在时间上解开。")
    res.add_notes("MSD∥ = MSD_x + MSD_y（垂直于 z 轴的面内分量），MSD⊥ = MSD_z；"
                  "D∥ 按 2 维、D⊥ 按 1 维拟合（MSD∥ = 4D∥t，MSD⊥ = 2D⊥t）。")
    if bad_fail:
        res.add_notes(
            "以下组分的「最小镜像失效比例」超过 20%，其 D 不应直接引用："
            + "、".join(bad_fail)
            + "。建议用更密的输出间隔重新计算 MSD。")
    res.add_notes(f"总扩散系数由 MSD = 2·dim·D·t（dim={dim}）的线性区斜率得到。")

    bad = [n for n in groups
           if not np.isfinite(res.summary.get(f"{n} D (Å²/ps)", float("nan")))]
    if bad:
        res.add_notes("以下组分的 MSD 曲线平坦或下降，粒子基本没有扩散，"
                      "因此不给扩散系数（D = nan）：" + "、".join(bad)
                      + "。这通常意味着该组分处于玻璃态/被约束/结晶相，"
                        "也可能是帧间隔太大导致曲线不具扩散性。")
    return res


def _vacf_raw(mdt, abs_idx, selection, chunk_atoms: int, verbose: bool = False):
    """按原子分块收集速度并算**未归一化的**相关和。

    返回 ``(raw, n_atoms, n_frames)``：``raw[lag] = Σ_i Σ_d Σ_t v_i,d(t)·v_i,d(t+lag)``。
    分块是为了内存与原子数无关（每块只驻留 ``n_frames × chunk × 3`` 个 double）。
    """
    u = mdt.universe
    n = int(len(selection.indices))
    n_at = int(len(abs_idx))
    raw = np.zeros(n, dtype=np.float64)
    nfft = 1 << int(np.ceil(np.log2(max(2, 2 * n))))
    chunk_atoms = int(max(1, min(chunk_atoms, n_at)))
    for start in range(0, n_at, chunk_atoms):
        sub = np.asarray(abs_idx[start:start + chunk_atoms], dtype=int)
        v = np.empty((n, sub.size, 3), dtype=np.float64)
        for k, (fi, _t) in enumerate(frame_iterator(mdt, selection, verbose=False)):
            u.trajectory[fi]
            v[k] = u.atoms[sub].velocities
        for d in range(3):
            F = np.fft.rfft(v[:, :, d], n=nfft, axis=0)
            acf = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)
            raw += acf[:n].sum(axis=1)
    return raw, n_at, n


@register("vacf", "速度自相关函数 VACF")
def analyze_vacf(mdt, groups: Mapping[str, object], selection: FrameSelection, *,
                 max_lag: int | None = None, chunk_atoms: int = 2048,
                 normalize: bool = False, verbose: bool = False) -> AnalysisResult:
    r"""速度自相关函数 VACF。

    .. math::

        C(\tau) = \frac{1}{(N-\tau)\,n_\mathrm{atoms}}
                   \sum_i \sum_{t_0} \mathbf v_i(t_0)\cdot\mathbf v_i(t_0+\tau)

    分母是 **(N-lag)·原子数**（无偏），也就是"对全部时间起点、全部原子求平均"，
    单位 Å²/ps²（速度用 MD 的 Å/ps）。这个口径与 Materials Studio Forcite 的
    ``VACF`` 分析**逐点一致**（实测 PE50 的 ETH 组分：1001 个 lag 全段平均偏差
    0.0000，C(0)=1043.7678 Å²/ps²）。

    只对**带速度的轨迹**可用：MS 的 ``.trj`` 转 TRR 后自带速度；XTC/DCD 不含速度。
    """
    if not groups:
        raise ValueError("必须提供至少一个组分")
    try:
        has_vel = bool(mdt.has_velocities)
    except Exception:                                            # noqa: BLE001
        has_vel = False
    if not has_vel:
        raise ValueError(
            "这条轨迹没有速度数据（XTC/DCD 等格式不存速度）。VACF 需要用带速度的"
            "轨迹：Materials Studio 的 .trj 读取时会转成 TRR（坐标 + 速度），"
            "或者换用 GROMACS 带速度的 TRR。")

    n_frame = int(len(selection.indices))
    if n_frame < 3:
        raise ValueError(f"VACF 至少要 3 帧（当前 {n_frame} 帧）")
    times = np.asarray(selection.times_ps, dtype=float)
    dt = float(np.median(np.diff(times))) if times.size > 1 else float(mdt.dt_ps or 1.0)

    lag_cap = int(max_lag) if max_lag else n_frame - 1
    lag_cap = int(max(1, min(lag_cap, n_frame - 1)))

    res = AnalysisResult(
        name="vacf",
        title="速度自相关函数 VACF",
        meta={"n_frames": n_frame, "dt_ps": dt, "max_lag_frames": lag_cap,
              "normalization": "mean/(N-lag)（= Forcite 口径）"},
    )
    res.panels = [
        Panel(xlabel="滞后时间 τ (ps)", ylabel="VACF (Å²/ps²)",
              title="速度自相关函数 VACF(τ)"),
        Panel(xlabel="滞后时间 τ (ps)", ylabel="归一化 VACF  C(τ)/C(0)",
              title="归一化速度自相关函数"),
    ]

    for label, ag in groups.items():
        idx = np.asarray(ag.indices, dtype=int)
        if idx.size == 0:
            continue
        raw, n_at, _n = _vacf_raw(mdt, idx, selection, chunk_atoms, verbose=verbose)
        lag = np.arange(n_frame, dtype=float)
        c = raw / np.maximum(n_frame - lag, 1.0) / max(n_at, 1)
        t = lag * dt
        m = lag <= lag_cap
        c = c[m]
        t = t[m]
        c0 = float(c[0]) if c.size else float("nan")
        res.add_curve(str(label), t, c, kind="line", panel=0, group=str(label))
        if np.isfinite(c0) and abs(c0) > 0:
            res.add_curve(str(label), t, c / c0, kind="line", panel=1,
                          group=str(label))
        else:
            res.add_curve(str(label), t, np.zeros_like(c), kind="line", panel=1,
                          group=str(label))
        # τ_c = ∫0^T C(τ)/C(0) dτ（梯形法），单位 ps
        tau_c = (float(np.trapezoid(c / c0, t)) if np.isfinite(c0) and c0 > 0
                 and c.size > 1 else float("nan"))
        d_est = (float(np.trapezoid(c, t)) / 3.0 if c.size > 1 else float("nan"))
        res.summary[f"{label} 原子数"] = int(n_at)
        res.summary[f"{label} VACF(0) 均方速度 (Å²/ps²)"] = c0
        res.summary[f"{label} 归一化 C(τ=1步)"] = (
            float(c[1] / c0) if c.size > 1 and np.isfinite(c0) and c0 > 0
            else float("nan"))
        res.summary[f"{label} 弛豫时间 τ_c (ps)"] = tau_c
        res.summary[f"{label} 由 VACF 积分估的 D (Å²/ps)"] = d_est
        if normalize:
            res.summary[f"{label} 说明"] = "曲线已按 C(0) 归一化（界面开关）"

    res.add_notes(
        "VACF(τ) = <v(0)·v(τ)>，对**全部时间起点与全部原子**求平均，"
        "分母用 (N-τ)·原子数（无偏），速度单位 Å/ps，所以单位 Å²/ps²。"
        "实测与 Forcite 的 VACF 输出逐点一致（PE50 ETH：全段平均偏差 0.0000）。")
    res.add_notes(
        "C(0) = 均方速度：3kT/m 的算术平均。可用来核对轨迹温度"
        "（氢原子贡献远大于碳，所以按原子平均时不要拿单一元素的质量去估）。")
    res.add_notes(
        "归一化曲线 C(τ)/C(0) 的积分是速度弛豫时间 τ_c；"
        "由 VACF 积分直接估的 D = (1/3)∫C dτ 与 MSD 斜率应大致相当"
        "（长轨迹上两者渐近一致，短轨迹上 VACF 估计受截断影响）。")
    return res
