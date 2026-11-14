#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 **MDAnalysis** 独立复现本工具「接触分析」的结果（逐位一致）。

这份脚本不 import mdta 就能跑；加了 ``--engine-compare`` 时会把工具的引擎也算一遍，
两个结果逐个数字对比 —— 这是"复现"最硬的证据。

一、工具里的口径（`mdta/analysis/interface.py::analyze_contacts`）
------------------------------------------------------------------
判据
    ``r < cutoff``（默认 **5.0 Å**）算一次接触。距离一律是**最小镜像距离**
    （正交盒用 ``round`` 折叠，三斜盒交给 ``MDAnalysis.lib.distances``），
    所以只要体系带周期盒，就不要自己用裸欧氏距离。

配对来源（同组 / 异组很重要）
    ==========  ===========================  ================================
    情形        工具用的调用                 每帧得到的对
    ==========  ===========================  ================================
    异组 A ≠ B  ``capped_distance(A, B)``     A×B 笛卡尔积，(i∈A, j∈B) 各一次
    同组 A = B  ``self_capped_distance(A)``   无序对 i<j，**不含 i==i 自配对**
    ==========  ===========================  ================================

配对模式（``inter`` / ``intra`` / ``total``）
    按**分子编号**过滤：``inter`` 只留不同分子的对（默认，避免"同分子相邻原子天然
    < cutoff"把接触数抬虚高）、``intra`` 只留同分子、``total`` 全要（旧口径）。
    分子编号 = ``molnums``；缺失或退化（分子数远小于键连通分量数）时改用
    **键连通分量**；编号为 ``-1``（未归属残基）的原子各自独立编号。

每条曲线 / 每个摘要量的确切定义
    · 「接触对数(t)」        = 该帧过滤后的原子对数
    · 「参与接触的 A 组原子数(t)」= 该帧至少出现一次接触的 A 组原子个数
                                （同组时两个端点都算 A）
    · 「最小原子间距(t)」     = 该帧过滤后所有对的最小距离；**若该帧一对都没有**，
                                就把截断距离逐次翻倍重新搜索（所以它不受 cutoff 影响）
    · 「平均接触数（每个 A 原子）」= 对每帧"该 A 原子接触到的 B 原子数"求帧平均
    · 「配位数分布」          = 每个 A 原子**每帧**的瞬时接触数，对（原子×帧）做直方图
    · 「接触矩阵」            = A、B 各按原子序号分 ≤48 块，格值 = 该块对之间
                                **每帧平均接触对数**（不除以块内原子数）
    · 「接触对占据率」        = 某一对 (i,j) 处于接触的帧数 / 总帧数（≤300 帧才统计）

二、用法
--------
    python examples/contact_reproduce.py TOP TRAJ "resname POLY" "resname HOH"
    python examples/contact_reproduce.py TOP TRAJ "index 0:99" "index 100:199" \
           --cutoff 4.5 --mode intra --stride 5
    python examples/contact_reproduce.py TOP TRAJ "resname POLY" "resname HOH" \
           --engine-compare            # 顺便和工具引擎对拍

    不给 ``--engine-compare`` 时只依赖 MDAnalysis + numpy。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# --------------------------------------------------------------------- 分子编号
_GLOBAL_CACHE = {}


def global_molecule_ids(universe):
    """**整个体系**的原子级分子编号（长度 = ``universe.atoms.n_atoms``）。

    关键：一定在**全体系**上算、再按原子序号取子集。按组各算一套会得到两套互不
    可比的编号 —— resid 重号时（每个组分都从 1 开始编号，或 MS 导出把整个体系塞进
    1–2 个残基）A 组的 "molnum 0" 与 B 组的 "molnum 0" 其实是两个不同分子，而
    ``inter`` 只做相等比较，于是把**跨组接触整套误删**（实测 482 对 → 325 对）。
    """
    key = id(universe)
    hit = _GLOBAL_CACHE.get(key)
    if hit is not None and hit[0] == int(universe.atoms.n_atoms):
        return hit[1]
    ids = _compute_global_ids(universe)
    if len(_GLOBAL_CACHE) > 8:
        _GLOBAL_CACHE.clear()
    _GLOBAL_CACHE[key] = (int(universe.atoms.n_atoms), ids)
    return ids


def _compute_global_ids(u):
    n_u = int(u.atoms.n_atoms)
    m = None
    try:
        cand = np.asarray(u.atoms.molnums, dtype=np.int64)
        if cand.size == n_u:
            m = cand
    except Exception:                                        # noqa: BLE001
        m = None
    comp = None
    try:
        frags = u.atoms.fragments
        if len(frags) > 1:
            comp = np.empty(n_u, dtype=np.int64)
            for k, f in enumerate(frags):
                comp[np.asarray(f.indices, dtype=np.int64)] = k
    except Exception:                                        # noqa: BLE001
        comp = None
    if m is None:
        return comp
    if comp is not None and _molnums_unreliable(m, comp):
        m = comp
    neg = m < 0
    if neg.any():
        m = m.copy()
        m[neg] = -(np.arange(m.size, dtype=np.int64)[neg] + 1)
    return m


def molecule_ids(ag, universe=None):
    """某个原子组的分子编号 —— 从全局编号里按原子序号取（与工具同一规则）。"""
    u = universe if universe is not None else ag.universe
    ids = global_molecule_ids(u)
    if ids is None:
        return None
    return np.asarray(ids, dtype=np.int64)[np.asarray(ag.indices, dtype=np.int64)]


def _same_partition(a, b) -> bool:
    """两个编号数组是否描述同一个划分（只差重新编号）。"""
    n_a, n_b = int(np.unique(a).size), int(np.unique(b).size)
    if n_a != n_b:
        return False
    combos = np.unique(np.stack([a, b], axis=1), axis=0)
    return int(combos.shape[0]) == n_a


def _molnums_unreliable(m, comp) -> bool:
    """``molnums`` 是否真的不可信（与 ``interface._molnums_unreliable`` 同判据）。

    只认两种：① 键块横跨多个 molnum（分子被切碎）；② 一个 molnum 里含 ≥2 个
    "有实质"（≥2 原子）的键块（不同分子被并成一个）。
    4 点水模型的 M 虚位点是**单原子块**，不触发 —— 所以它仍然按 molnums 算一个分子。
    """
    if m.size != comp.size:
        return False
    if int(np.unique(m).size) == int(np.unique(comp).size) and _same_partition(m, comp):
        return False
    order = np.argsort(comp, kind="stable")
    cs, ms = comp[order], m[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], order.size]
    for s, t in zip(starts.tolist(), ends.tolist()):
        if np.unique(ms[s:t]).size > 1:
            return True
    order = np.argsort(m, kind="stable")
    ms, cs = m[order], comp[order]
    starts = np.flatnonzero(np.r_[True, ms[1:] != ms[:-1]])
    ends = np.r_[starts[1:], order.size]
    for s, t in zip(starts.tolist(), ends.tolist()):
        _, cnt = np.unique(cs[s:t], return_counts=True)
        if int(np.sum(cnt >= 2)) >= 2:
            return True
    return False


# --------------------------------------------------------------------- 最小距离
def _min_distance(pa, pb, box, start_cutoff, u=None):
    """最小镜像距离；找不到就翻倍截断距离（与工具 ``min_distance`` 同算法）。"""
    from MDAnalysis.lib.distances import capped_distance

    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    if pa.size == 0 or pb.size == 0:
        return float("inf")
    if box is not None and np.all(np.asarray(box[:3], dtype=float) > 0):
        max_cut = 0.49 * float(np.min(np.asarray(box[:3], dtype=float)))
    else:
        max_cut = np.inf
    c = max(float(start_cutoff), 0.5)
    for _ in range(6):
        if c > max_cut:
            break
        try:
            _, d = capped_distance(pa, pb, max_cutoff=float(c), box=box,
                                   return_distances=True)
        except ValueError:
            break
        if d.size:
            return float(np.min(d))
        c *= 2.0
    if not np.isfinite(max_cut):                  # 无盒：直接暴力（分批省内存）
        best = np.inf
        for i in range(0, pa.shape[0], 256):
            _, d = capped_distance(pa[i:i + 256], pb, max_cutoff=1e9, box=None,
                                   return_distances=True)
            if d.size:
                best = min(best, float(np.min(d)))
        return best
    return float("inf")


# --------------------------------------------------------------------- 主计算
def reproduce_contacts(u, sel_a, sel_b, *, cutoff: float = 5.0, mode: str = "inter",
                       stride: int = 1, verbose: bool = False) -> dict:
    """按工具的口径复现接触分析。

    返回 ``{}`` 里全是与界面一一对应的量：

    ``times_ps`` / ``n_pairs`` / ``n_atoms_a`` / ``min_dist``  —— 三条曲线
    ``per_atom_mean``   每个 A 原子的"平均接触数"（帧平均）
    ``cn_hist``         瞬时接触数直方图（对 原子×帧 汇总）与取值
    ``matrix``          A×B 分块接触矩阵（每帧平均接触对数）
    ``summary``         摘要里那几个关键数字
    """
    from MDAnalysis.lib.distances import capped_distance, self_capped_distance

    A = u.select_atoms(sel_a) if isinstance(sel_a, str) else sel_a
    B = A if sel_b is None else (u.select_atoms(sel_b)
                                 if isinstance(sel_b, str) else sel_b)
    na, nb = A.n_atoms, B.n_atoms
    if na == 0 or nb == 0:
        raise SystemExit("两个原子组都不能为空")
    same = (A is B) or (na == nb and np.array_equal(A.indices, B.indices))
    if mode not in ("inter", "intra", "total"):
        raise SystemExit(f"未知配对模式 {mode!r}：可用 inter / intra / total")

    ma = molecule_ids(A, u)
    mb = ma if same else molecule_ids(B, u)
    eff = mode
    if eff != "total" and (ma is None or mb is None):
        eff = "total"                     # 与工具一致：没有分子编号就退化成 total
        if verbose:
            print("  [提示] 体系没有分子编号，配对模式退化为 total")

    # 帧：与工具一致，按选中的帧遍历
    frames = list(range(0, len(u.trajectory), max(int(stride), 1)))
    times = []
    n_pairs = np.zeros(len(frames), dtype=float)
    n_atoms_a = np.zeros(len(frames), dtype=float)
    min_dist = np.full(len(frames), np.nan, dtype=float)
    contact_sum = np.zeros(na, dtype=float)
    hit_count = np.zeros(na, dtype=float)
    cn_hist = None

    nb_u = int(u.atoms.n_atoms)
    n_ba = int(min(48, max(1, na)))
    n_bb = int(min(48, max(1, nb)))
    bin_a = np.minimum((np.arange(na) * n_ba) // max(na, 1), n_ba - 1)
    bin_b = np.minimum((np.arange(nb) * n_bb) // max(nb, 1), n_bb - 1)
    mat = np.zeros((n_ba, n_bb), dtype=float)
    occ_key = None
    occ_cnt = None

    for k, fi in enumerate(frames):
        ts = u.trajectory[fi]
        times.append(float(ts.time))
        pa = np.asarray(A.positions, dtype=float)
        pb = pa if same else np.asarray(B.positions, dtype=float)
        box = u.dimensions
        if same:                                     # 同组：无序对、无自配对
            pairs, dist = self_capped_distance(pa, max_cutoff=float(cutoff), box=box,
                                              return_distances=True)
        else:                                        # 异组：A×B 每对一次
            pairs, dist = capped_distance(pa, pb, max_cutoff=float(cutoff), box=box,
                                          return_distances=True)
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        dist = np.asarray(dist, dtype=float)
        # 配对模式过滤（同分子 / 不同分子）
        if pairs.size and eff != "total" and ma is not None:
            mi = ma[pairs[:, 0]]
            mj = (ma if same else mb)[pairs[:, 1]]
            same_mol = mi == mj
            keep = same_mol if eff == "intra" else ~same_mol
            pairs, dist = pairs[keep], dist[keep]
        n_pairs[k] = pairs.shape[0]
        if pairs.size:
            uniq = np.unique(pairs) if same else np.unique(pairs[:, 0])
            n_atoms_a[k] = uniq.size
            hit_count[uniq] += 1.0
            if same:
                np.add.at(contact_sum, pairs[:, 0], 1.0)
                np.add.at(contact_sum, pairs[:, 1], 1.0)
            else:
                np.add.at(contact_sum, pairs[:, 0], 1.0)
            _pa = bin_a[pairs[:, 0]]
            _pb = bin_b[pairs[:, 1]]
            np.add.at(mat, (_pa, _pb), 1.0)
            if same:
                np.add.at(mat, (_pb, _pa), 1.0)
            cnt = np.bincount(pairs[:, 0], minlength=na)
            if same:
                cnt = cnt + np.bincount(pairs[:, 1], minlength=na)
            need = int(cnt.max()) + 1
            if cn_hist is None:
                cn_hist = np.zeros(max(8, need), dtype=float)
            elif need > cn_hist.size:
                cn_hist = np.concatenate([cn_hist, np.zeros(need - cn_hist.size)])
            np.add.at(cn_hist, cnt, 1.0)
            # 接触对占据率（绝对原子序号编码）
            if len(frames) <= 300:
                gi = np.asarray(A.indices, dtype=np.int64)[pairs[:, 0]]
                gj = (np.asarray(B.indices, dtype=np.int64) if not same
                      else np.asarray(A.indices, dtype=np.int64))[pairs[:, 1]]
                lo = np.minimum(gi, gj) * nb_u + np.maximum(gi, gj)
                uk, uc = np.unique(lo, return_counts=True)
                if occ_key is None:
                    occ_key, occ_cnt = uk, uc.astype(np.int64)
                else:
                    allk = np.concatenate([occ_key, uk])
                    allc = np.concatenate([occ_cnt, uc.astype(np.int64)])
                    newk, inv = np.unique(allk, return_inverse=True)
                    acc = np.zeros(newk.size, dtype=np.int64)
                    np.add.at(acc, inv, allc)
                    occ_key, occ_cnt = newk, acc
        min_dist[k] = (float(np.min(dist)) if dist.size
                       else _min_distance(pa, pb, box, float(cutoff)))
        if verbose and (k % 10 == 0 or k == len(frames) - 1):
            print(f"  帧 {k + 1}/{len(frames)}  接触对 {int(n_pairs[k])}")

    n = max(len(frames), 1)
    per_atom_mean = contact_sum / n
    summary = {
        "平均接触对数": float(np.mean(n_pairs)),
        "最大接触对数": float(np.max(n_pairs)) if n_pairs.size else float("nan"),
        "平均参与接触原子数": float(np.mean(n_atoms_a)),
        "接触概率（存在接触的帧比例）": float(np.mean(n_pairs > 0)),
        "最小原子间距 (Å)": float(np.nanmin(min_dist)) if np.isfinite(
            min_dist).any() else float("nan"),
        "平均最小原子间距 (Å)": float(np.nanmean(min_dist)),
        "平均接触数（每个 A 组原子）": float(np.mean(per_atom_mean)) if na else
        float("nan"),
        "接触概率平均值": float(np.mean(hit_count / n)) if na else float("nan"),
        "配对模式": eff,
    }
    if occ_key is not None and occ_key.size:
        occ = occ_cnt.astype(float) / n
        summary["出现过的不同接触对总数"] = int(occ.size)
        summary["接触对占据率平均值"] = float(np.mean(occ))
        summary["始终接触（占据率 = 1）的接触对数"] = int(np.sum(occ >= 1.0))
        summary["占据率 ≥ 0.5 的接触对数"] = int(np.sum(occ >= 0.5))
    if cn_hist is not None and cn_hist.sum() > 0:
        vals = np.arange(cn_hist.size, dtype=float)
        p = cn_hist / cn_hist.sum()
        mean = float(np.sum(p * vals))
        summary["瞬时接触数 平均"] = mean
        summary["瞬时接触数 标准差"] = float(np.sqrt(np.sum(p * (vals - mean) ** 2)))
        summary["瞬时接触数 众数"] = float(vals[int(np.argmax(cn_hist))])
    return {
        "times_ps": np.asarray(times, dtype=float),
        "n_pairs": n_pairs,
        "n_atoms_a": n_atoms_a,
        "min_dist": min_dist,
        "per_atom_mean": per_atom_mean,
        "cn_hist": cn_hist,
        "matrix": mat / n,
        "matrix_bins": (bin_a, bin_b),
        "summary": summary,
    }


# --------------------------------------------------------------------- 与引擎对拍
def engine_contacts(topology, trajectory, sel_a, sel_b, *, cutoff=5.0, mode="inter",
                    stride=1):
    """用工具自己的引擎算一遍（需要 ``mdta`` 可 import）。"""
    from mdta.analysis.interface import analyze_contacts
    from mdta.pipeline import Analyzer

    az = Analyzer(topology, trajectory)
    if stride > 1:
        n_frames = len(az.universe.trajectory)
        keep = list(range(0, n_frames, stride))
        az.frames = type("FS", (), {"indices": np.asarray(keep),
                                    "times_ps": np.asarray(
                                        [az.times_ps[i] for i in keep]),
                                    "n_frames": len(keep),
                                    "describe": f"{len(keep)} 帧（stride {stride}）",
                                    "notes": [], "head": lambda self, k: self})()
    else:
        az.set_frames()
    A = az.universe.select_atoms(sel_a)
    B = A if sel_b is None else az.universe.select_atoms(sel_b)
    res = analyze_contacts(az.trajectory, A, B, az.require_frames(),
                           cutoff=float(cutoff), mode=mode)
    out = {"summary": dict(res.summary)}
    curves = {c.label: c for c in res.curves}
    out["n_pairs"] = np.asarray(curves.get("接触对数").y, dtype=float) \
        if "接触对数" in curves else None
    out["min_dist"] = np.asarray(curves.get("最小原子间距").y, dtype=float) \
        if "最小原子间距" in curves else None
    for c in res.curves:
        if str(c.label).startswith("参与接触"):
            out["n_atoms_a"] = np.asarray(c.y, dtype=float)
    return out


# --------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="用 MDAnalysis 复现本工具「接触分析」的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("topology", help="拓扑文件（.gro/.pdb/.tpr/.data…）")
    ap.add_argument("trajectory", nargs="?", default=None, help="轨迹文件（可省）")
    ap.add_argument("sel_a", help='A 组选择串，如 "resname POLY"')
    ap.add_argument("sel_b", nargs="?", default=None,
                    help='B 组选择串（省略 = 与 A 同组，算 A–A 接触）')
    ap.add_argument("--cutoff", type=float, default=5.0, help="接触判据 (Å)，默认 5.0")
    ap.add_argument("--mode", default="inter", choices=["inter", "intra", "total"],
                    help="配对模式，默认 inter（只算分子间）")
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧取 1 帧")
    ap.add_argument("--engine-compare", action="store_true",
                    help="同时用 mdta 引擎算一遍并逐个数字对拍")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    import MDAnalysis as mda

    u = mda.Universe(args.topology, args.trajectory) if args.trajectory else \
        mda.Universe(args.topology)
    res = reproduce_contacts(u, args.sel_a, args.sel_b, cutoff=args.cutoff,
                             mode=args.mode, stride=args.stride,
                             verbose=args.verbose)
    print("\n=== 接触分析（MDAnalysis 复现）===")
    print(f"  frames = {len(res['times_ps'])}   cutoff = {args.cutoff} Å   "
          f"mode = {res['summary']['配对模式']}")
    for k, v in res["summary"].items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.6g}")
        else:
            print(f"  {k:<28} {v}")
    print(f"  每帧接触对数（前 10）：{np.round(res['n_pairs'][:10], 1)}")

    if not args.engine_compare:
        return 0
    eng = engine_contacts(args.topology, args.trajectory, args.sel_a, args.sel_b,
                          cutoff=args.cutoff, mode=args.mode, stride=args.stride)
    print("\n=== 与工具引擎对拍 ===")
    bad = []
    for key in ("平均接触对数", "最小原子间距 (Å)", "平均接触数（每个 A 组原子）",
                "瞬时接触数 平均"):
        a, b = res["summary"].get(key), eng["summary"].get(key)
        if a is None or b is None:
            print(f"  {key:<28} 引擎没给这一项")
            continue
        same = abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        print(f"  {key:<28} 复现 {a:.6g} | 引擎 {b:.6g} | {'一致' if same else '不一致'}")
        if not same:
            bad.append(key)
    for key in ("n_pairs", "min_dist", "n_atoms_a"):
        mine, theirs = res.get(key), eng.get(key)
        if mine is None or theirs is None:
            continue
        m = min(mine.size, theirs.size)
        d = np.nanmax(np.abs(mine[:m] - theirs[:m])) if m else 0.0
        print(f"  {key:<28} 逐帧最大偏差 {d:.3g}")
        if d > 1e-9 * max(1.0, float(np.nanmax(np.abs(theirs[:m]) if m else 1.0))):
            bad.append(key)
    print("  => " + ("全部一致" if not bad else f"不一致：{bad}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
