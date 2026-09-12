# -*- coding: utf-8 -*-
"""分析模块的公共基础设施。

- :func:`frame_iterator`  按 :class:`~mdta.preprocess.FrameSelection` 逐帧遍历
- :func:`positions_for`   取某一帧中选定原子的坐标（可选按 PBC 展开）
- :func:`register` / :data:`ANALYSIS_REGISTRY`  分析功能注册表，
  供命令行与图形界面自动列出可用分析项
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterator

import numpy as np

from ..preprocess import FrameSelection

#: 分析功能注册表： ``{名称: (中文标题, 调用函数)}``
ANALYSIS_REGISTRY: "OrderedDict[str, tuple[str, Callable]]" = OrderedDict()


def register(name: str, title: str):
    """把一个分析函数登记到注册表（装饰器）。"""

    def deco(fn: Callable) -> Callable:
        ANALYSIS_REGISTRY[name] = (title, fn)
        return fn

    return deco


def frame_iterator(mdt, selection: FrameSelection, verbose: bool = False) -> Iterator[tuple[int, float]]:
    """按选中的帧号逐帧遍历，产出 ``(帧号, 时间/ps)``。"""
    u = mdt.universe
    for k, idx in enumerate(selection.indices):
        ts = u.trajectory[int(idx)]
        if verbose and (k % 10 == 0 or k == len(selection.indices) - 1):
            print(f"    帧 {k + 1}/{len(selection.indices)}  (index={idx}, t={ts.time:.1f} ps)")
        yield int(idx), float(ts.time)


#: 展开坐标的缓存：``(组指纹, 帧号, compound, 坐标指纹) -> 坐标``。
#:
#: 为什么需要：``ag.unwrap()`` 每次都要重新做"按分子连通性分组"那套工作，
#: 对千原子级的主链单次要 0.14 s 左右。而 order / dihedral 这类分析在一次
#: 遍历里会**对同一帧、同一个组**反复取展开坐标（实测各 4 次），
#: 于是 unwrap 占了这些分析一半以上的时间。按帧号缓存后只算一次。
#:
#: 键里**必须带上坐标本身**，不能只认帧号：调用方完全可能在不换帧的情况下
#: 直接改 ``ag.positions``（合成体系的测试就是这么做的），只认帧号会命中过期
#: 结果。这里用坐标字节的哈希，任何一位变化都会失效，而代价（一次 O(N) 的
#: tobytes+hash）比 unwrap 低三个数量级。
_UNWRAP_CACHE: dict[tuple, np.ndarray] = {}
_UNWRAP_CACHE_MAX = 24


def _group_fingerprint(ag) -> tuple:
    idx = np.asarray(ag.indices)
    if idx.size == 0:
        return (0, -1, -1)
    return (int(ag.n_atoms), int(idx[0]), int(idx[-1]))


def clear_unwrap_cache() -> None:
    """清空展开坐标缓存（换了体系/轨迹后调用）。"""
    _UNWRAP_CACHE.clear()


def positions_for(ag, unwrap: bool = True, compound: str = "fragments") -> np.ndarray:
    """返回原子组在当前帧的坐标；``unwrap=True`` 时先按分子连通性展开。

    展开是为了修正周期性边界条件把分子"切断"的问题。函数会恢复原始坐标，
    不改变调用者的状态。同一帧上对同一原子组的重复调用会命中缓存
    （见 :data:`_UNWRAP_CACHE`）。
    """
    if not unwrap or ag.n_atoms == 0:
        return np.asarray(ag.positions, dtype=float).copy()
    saved = np.asarray(ag.positions, dtype=float).copy()
    try:
        frame = int(ag.universe.trajectory.ts.frame)
    except Exception:  # noqa: BLE001
        frame = -1
    # 坐标指纹：任何一位变化都会让缓存失效（见 _UNWRAP_CACHE 的说明）
    key = (_group_fingerprint(ag), frame, str(compound),
           hash(saved.tobytes()))
    hit = _UNWRAP_CACHE.get(key)
    if hit is not None and hit.shape == (ag.n_atoms, 3):
        return hit.copy()

    out = saved
    for kwargs in ({"compound": compound, "reference": "com"},
                   {"compound": compound, "reference": "cog"},
                   {"compound": compound, "reference": None},
                   {"compound": "residues", "reference": None}):
        try:
            ag.positions = saved
            ag.unwrap(**kwargs)
            out = np.asarray(ag.positions, dtype=float).copy()
            break
        except Exception:  # noqa: BLE001 - 无键信息/零质量碎片时逐步退让
            continue
    else:
        # 最终退路：按最小镜像把原子聚拢到第一个原子附近
        if ag.n_atoms > 1:
            try:
                box = ag.universe.dimensions
                if box is not None:
                    from MDAnalysis.lib.distances import minimize_vectors

                    out = minimize_vectors(saved - saved[0], box) + saved[0]
            except Exception:  # noqa: BLE001
                out = saved
    ag.positions = saved
    if len(_UNWRAP_CACHE) >= _UNWRAP_CACHE_MAX:
        _UNWRAP_CACHE.clear()
    _UNWRAP_CACHE[key] = out
    return out.copy()


def run_analysis(name: str, *args, **kwargs):
    """按注册名调用分析函数。"""
    if name not in ANALYSIS_REGISTRY:
        raise KeyError(f"未知分析项 {name!r}；可用: {list(ANALYSIS_REGISTRY)}")
    return ANALYSIS_REGISTRY[name][1](*args, **kwargs)


def list_analyses() -> "OrderedDict[str, str]":
    """返回 ``{名称: 中文标题}``。"""
    return OrderedDict((k, v[0]) for k, v in ANALYSIS_REGISTRY.items())
