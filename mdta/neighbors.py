# -*- coding: utf-8 -*-
"""邻居表工具：**周期感知的邻居搜索** + 一次性缓存的短时复用。

为什么单独抽一层
----------------
分析模块里"找 cutoff 内的原子对"出现在 RDF / 接触 / BOO / 面内 RDF 里，语义完全
一样但各写一遍。这里统一成两个入口：

- :func:`self_pairs`  —— 同组（返回**无序对** ``i<j``，不含 ``i==j``）
- :func:`cross_pairs` —— 异组（A×B 笛卡尔积）

底层按盒子类型挑最快的实现（实测 3640 原子、rmax=12 Å、126 万对）：

============================  ==================
实现                          耗时
============================  ==================
MDAnalysis cell list            258 ms（同组）
scipy cKDTree(正交周期盒)        **130 ms**（同组）
cKDTree 只取对、不取距离          71 ms
MDAnalysis cell list（异组）     **70 ms**（异组，比 cKDTree 快）
============================  ==================

所以：**同组走 cKDTree**（快约 2×），**异组继续用 MDAnalysis**（它的 cell list
在异组上更快 —— ``query_ball_point`` 返回的是 Python 列表，扁平化的开销吃掉优势）。
三斜盒 scipy 的 ``boxsize`` 不支持（它只接受正交边长），自动退回 MDAnalysis。

缓存（``cache=True``）
----------------------
同一次分析里同一帧可能被查多遍（预估阶段会重跑前几帧、``estimate_times`` 先测 2 帧
再测 2m 帧；用户同时勾了多项时也常复用同一帧）。缓存按
``(类型, 坐标指纹, 盒指纹, cutoff)`` 存 ``(pairs, dist)``：

- 命中且缓存 cutoff ≥ 请求 cutoff 时，**按距离裁剪**即可（省掉整次搜索）；
- 命中但 cutoff 更小则视为未命中（重建更大的）；
- 内存上限：单条缓存对超过 ``_MAX_PAIRS``、或总字节超 ``_MAX_BYTES`` 就不缓存。

⚠️ 跨分析复用只在"两项分析处理到**同一帧**时"才发生；因为各分析是各自逐帧循环的，
正常跑一遍时同帧复用主要来自预估阶段（``estimate_times`` 会重复跑前几帧）。
"""

from __future__ import annotations

import zlib

import numpy as np

__all__ = ["self_pairs", "cross_pairs", "clear_cache", "cache_stats"]

#: 单条缓存允许的最大原子对数（≈ 40 MB 的 int64+float64）
_MAX_PAIRS = 3_000_000
#: 缓存总字节上限
_MAX_BYTES = 240_000_000
_MAX_ENTRIES = 6

#: cKDTree 与 MDAnalysis cell list 的**交叉点**（Å）：实测 20PE30（3640 原子，
#: 盒 33.6）与 122C12（5596 原子，盒 49.2）都在 5～6 Å 附近翻盘 ——
#: cutoff 5 Å 以下 MDA 的 cell list 更快（3 Å：3.9 ms vs 6.1 ms），
#: 5 Å 以上 cKDTree 明显更快（12 Å：161 ms vs 257 ms）。
#: 无盒时一律用 cKDTree（MDAnalysis 在没有周期盒时退化成 O(N²) 暴力）。
_KD_MIN_CUTOFF = 5.0

_CACHE: dict[tuple, tuple] = {}
_ORDER: list[tuple] = []
_STATS = {"hit": 0, "miss": 0, "filtered": 0, "bytes": 0}


def clear_cache() -> None:
    """清空邻居表缓存（换体系/轨迹后调用）。"""
    _CACHE.clear()
    _ORDER.clear()
    _STATS.update({"hit": 0, "miss": 0, "filtered": 0, "bytes": 0})


def cache_stats() -> dict:
    """缓存命中情况（诊断/测试用）。"""
    return dict(_STATS, entries=len(_CACHE))


# ------------------------------------------------------------------ 指纹/盒子
def _fp(arr: np.ndarray) -> int:
    """数组指纹（内容变了就不命中，与展开缓存同一套思路）。"""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return zlib.crc32(a.tobytes())


def _box_fp(box) -> int:
    if box is None:
        return 0
    b = np.asarray(box, dtype=np.float32).ravel()
    return zlib.crc32(b.tobytes())


def _ortho_lengths(box):
    """正交周期盒返回 ``(Lx, Ly, Lz)``（float32），否则 ``None``。

    scipy 的 ``cKDTree(boxsize=...)`` 只支持**轴对齐**的边长；三斜盒（角度≠90°）
    必须走 MDAnalysis 的 cell list。
    """
    if box is None:
        return None
    try:
        b = np.asarray(box, dtype=float).ravel()
    except Exception:                                          # noqa: BLE001
        return None
    if b.size < 3:
        return None
    L = b[:3]
    if not np.all(np.isfinite(L)) or np.any(L <= 0):
        return None
    if b.size >= 6:
        ang = b[3:6]
        if not np.allclose(ang, 90.0, atol=1e-6):
            return None
    return L.astype(np.float32)


# ------------------------------------------------------------------ 缓存
def _min_image_dist(pa, pb, pairs, box) -> np.ndarray:
    """由原子对直接算**最小镜像距离**（float64 精度；不需要再做一次邻居搜索）。"""
    if pairs.size == 0:
        return np.zeros(0, dtype=float)
    d = (np.asarray(pa, dtype=np.float32)[pairs[:, 0]].astype(np.float64)
         - np.asarray(pb, dtype=np.float32)[pairs[:, 1]].astype(np.float64))
    L = _ortho_lengths(box)
    if L is not None:
        d -= np.round(d / L.astype(np.float64)) * L.astype(np.float64)
    elif box is not None:
        # 三斜盒：折到分数坐标再转回来（与 MDAnalysis 的最小镜像同口径）
        try:
            from MDAnalysis.lib.distances import minimize_vectors

            d = minimize_vectors(d, box=box)
        except Exception:                                      # noqa: BLE001
            pass
    return np.sqrt((d * d).sum(axis=1))


def _cache_get(kind: str, pa: np.ndarray, pb, box, cutoff: float):
    """命中返回 ``(pairs, dist_or_None)``；未命中返回 ``None``。"""
    key = (kind, _fp(pa), 0 if pb is None else _fp(pb), _box_fp(box))
    ent = _CACHE.get(key)
    if ent is None:
        _STATS["miss"] += 1
        return None
    c0, pairs, dist = ent
    if c0 + 1e-9 < cutoff:                      # 缓存的是更小的 cutoff → 不可用
        _STATS["miss"] += 1
        return None
    _STATS["hit"] += 1
    other = pa if pb is None else pb
    if c0 == cutoff:
        return pairs, dist
    d_all = _min_image_dist(pa, other, pairs, box)   # 大 cutoff 裁剪成小 cutoff
    keep = d_all <= cutoff
    _STATS["filtered"] += 1
    return pairs[keep], (d_all[keep] if dist is not None else None)


def _cache_put(kind: str, pa: np.ndarray, pb, box, cutoff: float,
               pairs: np.ndarray, dist) -> None:
    if pairs.shape[0] > _MAX_PAIRS:
        return
    need = pairs.nbytes + (0 if dist is None else dist.nbytes)
    if _STATS["bytes"] + need > _MAX_BYTES:
        return
    key = (kind, _fp(pa), 0 if pb is None else _fp(pb), _box_fp(box))
    if key in _CACHE:
        old = _CACHE.pop(key)
        _ORDER.remove(key)
        _STATS["bytes"] -= old[1].nbytes + (0 if old[2] is None else old[2].nbytes)
    _CACHE[key] = (float(cutoff), pairs, dist)
    _ORDER.append(key)
    _STATS["bytes"] += need
    while len(_ORDER) > _MAX_ENTRIES:
        old_key = _ORDER.pop(0)
        e = _CACHE.pop(old_key, None)
        if e is not None:
            _STATS["bytes"] -= e[1].nbytes + (0 if e[2] is None else e[2].nbytes)


# ------------------------------------------------------------------ 同组
def self_pairs(pos, box, cutoff: float, *, cache: bool = True,
               want_dist: bool = True):
    """同组邻居对 ``(pairs, dist)``：无序对 ``i<j``、不含 ``i==j``，考虑 PBC。

    优先用 ``scipy.spatial.cKDTree(boxsize=L)``（正交周期盒）/ 无盒时用不建 boxsize
    的树（比 MDAnalysis 在无盒时的 N² 暴力快几个数量级）；三斜盒或
    ``cutoff ≥ 半盒边`` 时退回 ``MDAnalysis.self_capped_distance``。

    ``want_dist=False`` 时只返回 ``(pairs, None)`` —— BOO 这类"从坐标自己算方向"
    的调用方不需要距离，省掉一趟 O(n_pairs) 的 sqrt/sum（实测 20–40 ms/帧）。
    """
    pa = np.asarray(pos, dtype=np.float32)
    if pa.shape[0] < 2 or float(cutoff) <= 0:
        return np.zeros((0, 2), dtype=np.int64), (np.zeros(0, dtype=float) if want_dist else None)
    if cache:
        hit = _cache_get("self", pa, None, box, float(cutoff))
        if hit is not None:
            pr, dd = hit
            if want_dist and dd is None:
                dd = _min_image_dist(pa, pa, pr, box)
            elif not want_dist:
                dd = None
            return pr, dd
    L = _ortho_lengths(box)
    pairs = np.zeros((0, 2), dtype=np.int64)
    dist = np.zeros(0, dtype=float)
    done = False
    # 只有"无盒"或"正交盒"才能用 cKDTree；三斜盒必须交给 MDAnalysis 的 cell list
    # （scipy 的 boxsize 只接受轴对齐边长，用非周期的树会把周期性丢掉 → 结果偏小）
    usable = (box is None) or (L is not None)
    big_enough = (box is None) or (float(cutoff) >= _KD_MIN_CUTOFF)
    if usable and big_enough and (L is None or float(cutoff) < 0.5 * float(np.min(L))):
        try:
            from scipy.spatial import cKDTree

            if L is None:                                  # 无盒：纯欧氏
                tree = cKDTree(pa)
                pr = np.asarray(tree.query_pairs(r=float(cutoff),
                                                 output_type="ndarray"),
                                dtype=np.int64).reshape(-1, 2)
            else:                                          # 正交周期盒
                wrapped = np.mod(pa, L)
                tree = cKDTree(wrapped, boxsize=L.astype(np.float64))
                pr = np.asarray(tree.query_pairs(r=float(cutoff),
                                                 output_type="ndarray"),
                                dtype=np.int64).reshape(-1, 2)
                # ⚠️ 不能把 pa 换成 wrapped：缓存指纹按**传入坐标**算，
                #    换掉之后下一次调用就永远命不中了
            pairs = pr
            if want_dist:
                dist = _min_image_dist(pa, pa, pairs, box)
            else:
                dist = None
            done = True
        except Exception:                                  # noqa: BLE001 - 退回 MDA
            done = False
    if not done:
        from MDAnalysis.lib.distances import self_capped_distance

        if want_dist:
            pr, dd = self_capped_distance(pa, max_cutoff=float(cutoff), box=box,
                                          return_distances=True)
            dist = np.asarray(dd, dtype=float)
        else:
            pr = self_capped_distance(pa, max_cutoff=float(cutoff), box=box,
                                      return_distances=False)
            dist = None
        pairs = np.asarray(pr, dtype=np.int64).reshape(-1, 2)
    if not want_dist:
        dist = None
    if cache:
        _cache_put("self", pa, None, box, float(cutoff), pairs, dist)
    return pairs, dist


# ------------------------------------------------------------------ 异组
def cross_pairs(pa, pb, box, cutoff: float, *, cache: bool = True):
    """异组邻居对 ``(pairs, dist)``：A×B 的笛卡尔积（每对一次），考虑 PBC。

    异组**继续用 MDAnalysis 的 cell list**：实测 1820×1820、rmax=12 Å 时它
    70 ms，而 ``cKDTree.query_ball_point`` 要 128 ms（返回 Python 列表，扁平化的
    开销把收益吃掉）、分块 numpy 220 ms。
    """
    a = np.asarray(pa, dtype=np.float32)
    b = np.asarray(pb, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0 or float(cutoff) <= 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    if cache:
        hit = _cache_get("cross", a, b, box, float(cutoff))
        if hit is not None:
            return hit
    from MDAnalysis.lib.distances import capped_distance

    pairs, dist = capped_distance(a, b, max_cutoff=float(cutoff), box=box,
                                  return_distances=True)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    dist = np.asarray(dist, dtype=float)
    if cache:
        _cache_put("cross", a, b, box, float(cutoff), pairs, dist)
    return pairs, dist
