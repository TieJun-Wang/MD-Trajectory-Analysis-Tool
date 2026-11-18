# -*- coding: utf-8 -*-
"""接触分析（contact）—— 与工具 `analyze_contacts` 同一口径。

口径要点（详见 `接触分析复现指南.md`）：

| 项目 | 定义 |
|---|---|
| 判据 | 任意 A–B 原子对的**最小镜像**距离 `r < cutoff`（默认 5.0 Å） |
| 异组 A≠B | `capped_distance(A, B)` —— A×B 每对一次 |
| 同组 A=B | `self_capped_distance(A)` —— 无序对 `i<j`、**不含** `i==i` 自配对 |
| 模式 | `inter`（默认，只留不同分子的对）/ `intra` / `total` |
| 分子编号 | **全体系**算一次再取子集（见 `_common.atom_mol_ids`） |

两套入口，口径完全一致：

1. :func:`test_contact` / :func:`test_contact_matrix`
   —— 逐帧打印 + 导出 CSV（曲线级，看趋势）；
2. :func:`reproduce_contacts` / :func:`compare_engine` / :func:`test_contact_reproduce`
   —— **独立复现 + 与工具引擎逐帧对拍**（原 ``examples/contact_reproduce.py``）：
   一次性给出摘要里的全部量，并能把引擎算的同一批帧拿过来比对偏差。

单独跑：``python test/mda_test/contact.py``（交互菜单）；
``python test/mda_test/contact.py --top a.xtd --trj a.trj --compare-engine -y``（命令行）。
"""

from __future__ import annotations

import numpy as np

try:
    from ._common import (atom_mol_ids, auto_pair, frame_indices,
                          min_image_min_distance, save_csv)
except ImportError:
    from _common import (atom_mol_ids, auto_pair, frame_indices,
                         min_image_min_distance, save_csv)


def _frames_to_use(u, frames, stride):
    """要遍历的帧号：显式 `frames` > `stride` > 环境变量 `MDTA_MDA_FRAMES` > 全部。"""
    if frames is not None:
        return [int(f) for f in frames]
    if int(stride) > 1:
        return list(range(0, len(u.trajectory), int(stride)))
    return frame_indices(u)


def _pairs_at(pa, pb, box, cutoff, same):
    """当前帧的原子对（同组→无序对；异组→A×B）。"""
    from MDAnalysis.lib.distances import capped_distance, self_capped_distance

    if same:
        pairs, dist = self_capped_distance(pa, max_cutoff=float(cutoff), box=box,
                                           return_distances=True)
    else:
        pairs, dist = capped_distance(pa, pb, max_cutoff=float(cutoff), box=box,
                                      return_distances=True)
    return (np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
            np.asarray(dist, dtype=float))


def test_contact(u, ag_a=None, ag_b=None, cutoff=5.0, mode="inter", stride=1,
                 frames=None, prefix="contact_mda"):
    """测试接触分析模块（对应界面「接触分析」的曲线 + 摘要）。

    参数
        u         Universe（已 load_new 轨迹）
        ag_a/ag_b 两个原子组；留空则按工具规则自动挑两个组分
        cutoff    接触判据（Å），默认 5.0
        mode      inter / intra / total，默认 inter（只算分子间）
        stride    每 N 帧取 1 帧（大轨迹先用 10 试试）
        frames    直接给帧号列表（给了就忽略 stride；与工具"指定帧"对拍时用）
        prefix    导出 CSV 的前缀
    导出
        {prefix}_pairs.csv   time, 接触对数, 参与接触的 A 原子数, 最小原子间距(Å)
        {prefix}_coord.csv   瞬时接触数, 概率（配位数分布）
    """
    if ag_a is None:
        ag_a, ag_b, la, lb = auto_pair(u)
    else:
        la = lb = "自定义"
    a = ag_a
    b = a if ag_b is None else ag_b          # ag_b 留空 = 同组自接触（A–A）
    na, nb = a.n_atoms, b.n_atoms
    same = (a is b) or (na == nb and np.array_equal(a.indices, b.indices))

    mol = atom_mol_ids(u)
    idx_a = np.asarray(a.indices, dtype=np.int64)
    ma = None if mol is None else np.asarray(mol, dtype=np.int64)[idx_a]
    if same:
        mb = ma
    else:
        mb = (None if mol is None
              else np.asarray(mol, dtype=np.int64)[np.asarray(b.indices, dtype=np.int64)])

    eff = str(mode)
    if eff not in ("inter", "intra", "total"):
        raise ValueError(f"未知配对模式 {mode!r}：可用 inter / intra / total")
    if eff != "total" and (ma is None or mb is None):
        eff = "total"
        print("⚠️ 体系没有分子编号，配对模式退化为 total")

    frames = _frames_to_use(u, frames, stride)
    print(f"总帧数 {len(u.trajectory)}（取 {len(frames)} 帧）  A={la}({na} 原子)  "
          f"B={lb}({nb} 原子)  cutoff={cutoff} Å  mode={eff}")
    if same:
        print("  同组接触：无序对 i<j，不含 i==i 自配对")

    n_u = int(u.atoms.n_atoms)
    t_list, n_pair, n_atom_a, d_min = [], [], [], []
    per_atom_sum = np.zeros(na, dtype=float)
    cn_hist = None
    occ: dict[int, int] = {}
    for fi in frames:
        ts = u.trajectory[fi]
        pa = np.asarray(a.positions, dtype=float)
        pb = pa if same else np.asarray(b.positions, dtype=float)
        pairs, dist = _pairs_at(pa, pb, ts.dimensions, cutoff, same)
        if pairs.size and eff != "total" and ma is not None:
            keep = (ma[pairs[:, 0]] == mb[pairs[:, 1]])
            if eff == "inter":
                keep = ~keep
            pairs, dist = pairs[keep], dist[keep]
        t_list.append(float(ts.time))
        n_pair.append(int(pairs.shape[0]))
        if pairs.size:
            uniq = np.unique(pairs) if same else np.unique(pairs[:, 0])
            n_atom_a.append(int(uniq.size))
            np.add.at(per_atom_sum, pairs[:, 0], 1.0)
            if same:
                np.add.at(per_atom_sum, pairs[:, 1], 1.0)
            cnt = np.bincount(pairs[:, 0], minlength=na)
            if same:
                cnt = cnt + np.bincount(pairs[:, 1], minlength=na)
            need = int(cnt.max()) + 1
            if cn_hist is None:
                cn_hist = np.zeros(max(8, need), dtype=float)
            elif need > cn_hist.size:
                cn_hist = np.concatenate([cn_hist, np.zeros(need - cn_hist.size)])
            np.add.at(cn_hist, cnt, 1.0)
            gi = idx_a[pairs[:, 0]]
            gj = (idx_a if same else np.asarray(b.indices, dtype=np.int64))[pairs[:, 1]]
            for key in np.minimum(gi, gj) * n_u + np.maximum(gi, gj):
                occ[int(key)] = occ.get(int(key), 0) + 1
        else:
            n_atom_a.append(0)
        d_min.append(float(np.min(dist)) if dist.size
                     else min_image_min_distance(pa, pb, ts.dimensions, float(cutoff)))

    n = max(len(frames), 1)
    save_csv(f"{prefix}_pairs.csv",
             [t_list, n_pair, n_atom_a, d_min],
             header="time(ps),n_pairs,n_atoms_a,min_dist(A)",
             fmt="%.4f,%.1f,%.1f,%.6f")
    if cn_hist is not None and cn_hist.sum() > 0:
        vals = np.arange(cn_hist.size, dtype=float)
        save_csv(f"{prefix}_coord.csv", [vals, cn_hist / cn_hist.sum()],
                 header="instant_coordination,probability", fmt="%.1f,%.8f")

    per_atom_mean = per_atom_sum / n
    print(f"  平均接触对数            {np.mean(n_pair):,.1f} 对/帧（最大 {max(n_pair):,}）")
    print(f"  最小原子间距            {min(d_min):.4f} Å（全帧最小）")
    print(f"  参与接触的 A 原子数      {np.mean(n_atom_a):.1f} / {na}")
    print(f"  平均接触数（每个 A 原子）{np.mean(per_atom_mean):.4f}  ← 描述接触强度的主指标")
    print(f"  接触概率（有接触的帧比例）{np.mean([x > 0 for x in n_pair]):.3f}"
          "（稠密体系恒为 1，无区分度）")
    if occ:
        o = np.asarray(list(occ.values()), dtype=float) / n
        print(f"  接触对占据率            平均 {o.mean():.3f}，"
              f"出现过 {o.size:,} 个不同的对，始终接触 {int(np.sum(o >= 1.0)):,} 个")
    return {"n_pairs": np.asarray(n_pair, dtype=float),
            "n_atoms_a": np.asarray(n_atom_a, dtype=float),
            "min_dist": np.asarray(d_min, dtype=float),
            "per_atom_mean": per_atom_mean,
            "cn_hist": cn_hist}


def test_contact_matrix(u, ag_a=None, ag_b=None, cutoff=5.0, nbin=48, stride=1,
                        frames=None, prefix="contact_matrix_mda"):
    """接触矩阵（对应界面「接触矩阵」热力图）。

    A、B 各按原子序号分 ≤`nbin` 块，格值 = 该块对之间的**每帧平均接触对数**
    （不除以块内原子数）。导出 ``{prefix}.csv``（纯矩阵，逗号分隔）。
    """
    if ag_a is None:
        ag_a, ag_b, la, lb = auto_pair(u)
    else:
        la = lb = "自定义"
    a = ag_a
    b = a if ag_b is None else ag_b
    same = (a is b) or (a.n_atoms == b.n_atoms
                        and np.array_equal(a.indices, b.indices))
    na, nb = a.n_atoms, b.n_atoms
    n_ba, n_bb = int(min(nbin, na)), int(min(nbin, nb))
    bin_a = np.minimum((np.arange(na) * n_ba) // max(na, 1), n_ba - 1)
    bin_b = np.minimum((np.arange(nb) * n_bb) // max(nb, 1), n_bb - 1)

    frames = _frames_to_use(u, frames, stride)
    mat = np.zeros((n_ba, n_bb), dtype=float)
    for fi in frames:
        ts = u.trajectory[fi]
        pa = np.asarray(a.positions, dtype=float)
        pb = pa if same else np.asarray(b.positions, dtype=float)
        pairs, _ = _pairs_at(pa, pb, ts.dimensions, cutoff, same)
        if not pairs.size:
            continue
        np.add.at(mat, (bin_a[pairs[:, 0]], bin_b[pairs[:, 1]]), 1.0)
        if same:                        # 同组：两端都在 A，对称地补上另一半
            np.add.at(mat, (bin_b[pairs[:, 1]], bin_a[pairs[:, 0]]), 1.0)
    mat /= max(len(frames), 1)

    path = save_csv(f"{prefix}.csv", mat,
                    header=f"A({la}) x B({lb}) mean_contacts_per_frame cutoff={cutoff}A",
                    fmt="%.4f")
    print(f"接触矩阵 {n_ba}×{n_bb}（A={la} {na} 原子，B={lb} {nb} 原子，"
          f"{len(frames)} 帧，cutoff={cutoff} Å）")
    print(f"  最大格值 {mat.max():.4f}，非零格 {int(np.sum(mat > 0))}/{mat.size}")
    print(f"  {path}")
    return mat


def reproduce_contacts(u, ag_a=None, ag_b=None, *, cutoff=5.0, mode="inter",
                       frames=None, verbose=False):
    """**独立复现**一遍接触分析，一次性返回摘要里的全部量（不依赖 mdta）。

    这是 ``examples/contact_reproduce.py`` 里那套实现的包内版本（同口径、同结果），
    区别只是直接吃一个 Universe + 两个原子组，不需要再传文件路径。

    返回 ``{"times_ps", "n_pairs", "n_atoms_a", "min_dist", "per_atom_mean",
    "cn_hist", "matrix", "summary"}`` —— ``summary`` 里的键名与界面摘要一一对应。
    """
    a = ag_a if ag_a is not None else u.atoms
    b = a if ag_b is None else ag_b
    na, nb = int(a.n_atoms), int(b.n_atoms)
    if na == 0 or nb == 0:
        raise SystemExit(f"原子组为空（A={na}，B={nb}）")
    same = (a is b) or (na == nb and np.array_equal(a.indices, b.indices))
    eff = str(mode)
    if eff not in ("inter", "intra", "total"):
        raise ValueError(f"未知配对模式 {mode!r}：可用 inter / intra / total")

    mol = atom_mol_ids(u)
    idx_a = np.asarray(a.indices, dtype=np.int64)
    ma = None if mol is None else np.asarray(mol, dtype=np.int64)[idx_a]
    mb = ma if same else (None if mol is None else
                          np.asarray(mol, dtype=np.int64)[
                              np.asarray(b.indices, dtype=np.int64)])
    if eff != "total" and (ma is None or mb is None):
        eff = "total"
        if verbose:
            print("  ⚠️ 体系没有分子编号，配对模式退化为 total")

    use = frame_indices(u) if frames is None else [int(f) for f in frames]
    n = max(len(use), 1)
    times = np.zeros(len(use), dtype=float)
    n_pairs = np.zeros(len(use), dtype=float)
    n_atoms_a = np.zeros(len(use), dtype=float)
    min_dist = np.full(len(use), np.nan)
    per_atom_sum = np.zeros(na, dtype=float)
    hit = np.zeros(na, dtype=float)
    cn_hist = None
    occ: dict[int, int] = {}
    n_u = int(u.atoms.n_atoms)
    n_ba, n_bb = int(min(48, max(1, na))), int(min(48, max(1, nb)))
    bin_a = np.minimum((np.arange(na) * n_ba) // max(na, 1), n_ba - 1)
    bin_b = np.minimum((np.arange(nb) * n_bb) // max(nb, 1), n_bb - 1)
    mat = np.zeros((n_ba, n_bb), dtype=float)

    for k, fi in enumerate(use):
        ts = u.trajectory[fi]
        pa = np.asarray(a.positions, dtype=float)
        pb = pa if same else np.asarray(b.positions, dtype=float)
        pairs, dist = _pairs_at(pa, pb, ts.dimensions, cutoff, same)
        times[k] = float(ts.time)
        if pairs.size and eff != "total" and ma is not None:
            keep = (ma[pairs[:, 0]] == mb[pairs[:, 1]])
            if eff == "inter":
                keep = ~keep
            pairs, dist = pairs[keep], dist[keep]
        n_pairs[k] = pairs.shape[0]
        if pairs.size:
            uniq = np.unique(pairs) if same else np.unique(pairs[:, 0])
            n_atoms_a[k] = uniq.size
            hit[uniq] += 1.0
            np.add.at(per_atom_sum, pairs[:, 0], 1.0)
            if same:
                np.add.at(per_atom_sum, pairs[:, 1], 1.0)
            np.add.at(mat, (bin_a[pairs[:, 0]], bin_b[pairs[:, 1]]), 1.0)
            if same:
                np.add.at(mat, (bin_b[pairs[:, 1]], bin_a[pairs[:, 0]]), 1.0)
            cnt = np.bincount(pairs[:, 0], minlength=na)
            if same:
                cnt = cnt + np.bincount(pairs[:, 1], minlength=na)
            need = int(cnt.max()) + 1
            if cn_hist is None:
                cn_hist = np.zeros(max(8, need), dtype=float)
            elif need > cn_hist.size:
                cn_hist = np.concatenate([cn_hist, np.zeros(need - cn_hist.size)])
            np.add.at(cn_hist, cnt, 1.0)
            if len(use) <= 300:
                gi = idx_a[pairs[:, 0]]
                gj = (idx_a if same else np.asarray(b.indices, dtype=np.int64))[
                    pairs[:, 1]]
                for key in np.minimum(gi, gj) * n_u + np.maximum(gi, gj):
                    occ[int(key)] = occ.get(int(key), 0) + 1
        min_dist[k] = (float(np.min(dist)) if dist.size
                       else min_image_min_distance(pa, pb, ts.dimensions,
                                                   float(cutoff)))
        if verbose:
            print(f"  帧 {fi}: 接触对 {int(n_pairs[k])}，最小间距 {min_dist[k]:.4f} Å")

    per_atom_mean = per_atom_sum / n
    summary = {
        "平均接触对数": float(np.mean(n_pairs)),
        "最大接触对数": float(np.max(n_pairs)) if n_pairs.size else float("nan"),
        "平均参与接触原子数": float(np.mean(n_atoms_a)),
        "接触概率（存在接触的帧比例）": float(np.mean(n_pairs > 0)),
        "最小原子间距 (Å)": float(np.nanmin(min_dist)) if np.isfinite(
            min_dist).any() else float("nan"),
        "平均最小原子间距 (Å)": float(np.nanmean(min_dist)),
        "平均接触数（每个 A 组原子）": float(np.mean(per_atom_mean)),
        "原子接触概率平均值": float(np.mean(hit / n)),
        "配对模式": eff,
        "帧数": int(len(use)),
    }
    if occ:
        o = np.asarray(list(occ.values()), dtype=float) / n
        summary["出现过的不同接触对总数"] = int(o.size)
        summary["接触对占据率平均值"] = float(np.mean(o))
        summary["始终接触（占据率 = 1）的接触对数"] = int(np.sum(o >= 1.0))
        summary["占据率 ≥ 0.5 的接触对数"] = int(np.sum(o >= 0.5))
    if cn_hist is not None and cn_hist.sum() > 0:
        vals = np.arange(cn_hist.size, dtype=float)
        p = cn_hist / cn_hist.sum()
        mean = float(np.sum(p * vals))
        summary["瞬时接触数 平均"] = mean
        summary["瞬时接触数 标准差"] = float(np.sqrt(np.sum(p * (vals - mean) ** 2)))
        summary["瞬时接触数 众数"] = float(vals[int(np.argmax(cn_hist))])
    return {"times_ps": times, "n_pairs": n_pairs, "n_atoms_a": n_atoms_a,
            "min_dist": min_dist, "per_atom_mean": per_atom_mean,
            "cn_hist": cn_hist, "matrix": mat / n, "summary": summary}


def compare_engine(u, ag_a=None, ag_b=None, *, cutoff=5.0, mode="inter",
                   frames=None, verbose=True):
    """把**工具引擎**（``mdta.analysis.interface.analyze_contacts``）在同一批帧上
    算一遍，与 :func:`reproduce_contacts` 逐帧比对。

    只比"能一一对应的量"：逐帧接触对数、逐帧最小原子间距、平均接触数。
    返回 ``{"n_pairs", "min_dist", "per_atom"}`` 三组偏差（None = 该项没比成）。
    """
    from types import SimpleNamespace

    from mdta.analysis.interface import analyze_contacts
    from mdta.io import MDTrajectory

    a = ag_a if ag_a is not None else u.atoms
    b = a if ag_b is None else ag_b
    fl = [int(f) for f in (frame_indices(u) if frames is None else frames)]
    mine = reproduce_contacts(u, a, b, cutoff=cutoff, mode=mode, frames=fl,
                              verbose=False)

    mdt = MDTrajectory.from_universe(u)
    times = np.asarray(mdt.times_ps, dtype=float)
    sel = SimpleNamespace(indices=np.asarray(fl, dtype=int),
                          times_ps=times[np.asarray(fl, dtype=int)],
                          n_frames=len(fl), describe=f"{len(fl)} 帧（指定）",
                          notes=[], head=lambda self, k: self)
    # 引擎只吃 MDAnalysis 的 AtomGroup：按同一批原子序号从它的 Universe 里取
    uu = mdt.universe
    ga = uu.atoms[np.asarray(a.indices, dtype=np.int64)]
    gb = uu.atoms[np.asarray(b.indices, dtype=np.int64)]
    eng = analyze_contacts(mdt, ga, gb, sel, cutoff=float(cutoff), mode=mode)

    def _curve(res, prefix):
        for c in res.curves:
            if str(c.label).startswith(prefix):
                return np.asarray(c.y, dtype=float)
        return None

    out = {"n_pairs": None, "min_dist": None, "per_atom": None}
    e_pairs = _curve(eng, "接触对数")
    if e_pairs is not None:
        m = min(e_pairs.size, mine["n_pairs"].size)
        out["n_pairs"] = float(np.max(np.abs(e_pairs[:m] - mine["n_pairs"][:m])))
    e_min = _curve(eng, "最小原子间距")
    if e_min is not None:
        m = min(e_min.size, mine["min_dist"].size)
        ok = np.isfinite(e_min[:m]) & np.isfinite(mine["min_dist"][:m])
        if ok.any():
            out["min_dist"] = float(np.max(np.abs(e_min[:m][ok]
                                                 - mine["min_dist"][:m][ok])))
    try:
        out["per_atom"] = abs(float(eng.summary["平均接触数（每个 A 组原子）"])
                              - float(mine["summary"]["平均接触数（每个 A 组原子）"]))
    except Exception:                                     # noqa: BLE001
        pass

    if verbose:
        print(f"\n  === 与引擎对拍（{len(fl)} 帧，mode={mine['summary']['配对模式']}）===")
        # 判据：接触对数必须**完全相同**（整数对，差 1 对都算错）；
        # 最小间距容差 1e-5 Å —— 两条邻居搜索链路的浮点舍入
        # （引擎 cKDTree+float32 坐标 vs MDA cell list）实测差 1e-6 量级，
        # 与工具自检里 `test_contact_...` 的容差同源。
        for key, label, unit, tol in (("n_pairs", "逐帧接触对数", "", 0.0),
                                      ("min_dist", "逐帧最小间距", " Å", 1e-5),
                                      ("per_atom", "平均接触数", "", 1e-9)):
            v = out[key]
            if v is None:
                print(f"    {label:<12} 引擎没给这一项")
            else:
                ok = v <= tol
                print(f"    {label:<12} 最大偏差 {v:.3e}{unit}"
                      f"   {'一致 ✅' if ok else '★ 不一致'}")
    return out


def test_contact_reproduce(u, ag_a=None, ag_b=None, *, cutoff=5.0, mode="inter",
                           frames=None, prefix="contact_repro", compare=False):
    """独立复现 + 导出摘要 CSV（+ 可选与引擎对拍）。

    导出 ``{prefix}_pairs.csv``（time / 接触对数 / 参与接触的 A 原子数 / 最小间距）
    与 ``{prefix}_summary.csv``（摘要逐项，便于和别的工具并排看）。
    """
    res = reproduce_contacts(u, ag_a, ag_b, cutoff=cutoff, mode=mode, frames=frames)
    s = res["summary"]
    save_csv(f"{prefix}_pairs.csv",
             [res["times_ps"], res["n_pairs"], res["n_atoms_a"], res["min_dist"]],
             header="time(ps),n_pairs,n_atoms_a,min_dist(A)",
             fmt="%.4f,%.1f,%.1f,%.6f")
    keys = list(s.keys())
    vals = []
    for k in keys:
        v = s[k]
        vals.append(float(v) if isinstance(v, (int, float)) else float("nan"))
    with open(_out(f"{prefix}_summary.csv"), "w", encoding="utf-8-sig") as fh:
        fh.write("quantity,value\n")
        for k, v in zip(keys, vals):
            fh.write(f"{k},{v:.6g}\n")
    print(f"✅ 已导出 {_out(f'{prefix}_summary.csv')}")

    print(f"\n  === 接触分析摘要（独立复现，{s['帧数']} 帧，mode={s['配对模式']}）===")
    for k in ("平均接触对数", "最大接触对数", "平均参与接触原子数",
              "最小原子间距 (Å)", "平均最小原子间距 (Å)",
              "平均接触数（每个 A 组原子）", "原子接触概率平均值",
              "接触概率（存在接触的帧比例）", "出现过的不同接触对总数",
              "接触对占据率平均值", "瞬时接触数 平均", "瞬时接触数 标准差"):
        if k in s:
            print(f"    {k:<22} {s[k]:,.4f}" if isinstance(s[k], float)
                  else f"    {k:<22} {s[k]:,}")
    if compare:
        compare_engine(u, ag_a, ag_b, cutoff=cutoff, mode=mode, frames=frames)
    return res


def _out(name: str) -> str:
    try:
        from ._common import out_path
    except ImportError:
        from _common import out_path
    return out_path(name)


def menu():
    """菜单 / 命令行入口：``python test/mda_test/contact.py``。"""
    try:
        from ._menu import Q, run_module
    except ImportError:
        from _menu import Q, run_module

    specs = [
        Q("cutoff", "接触判据 cutoff (Å)", "float", 5.0, lo=0.1,
          help="任意 A–B 原子对的最小镜像距离小于它算一次接触"),
        Q("mode", "配对模式", "choice", "inter",
          choices=[("inter", "分子间（默认，剔除同分子配对）"),
                   ("intra", "分子内（只留同分子配对）"),
                   ("total", "总体（含分子内，旧口径）")]),
        Q("select_b", "B 组选择串", "text", "",
          help="留空 = 同组自接触（A–A）；填了就按 A–B 算"),
        Q("matrix", "同时算接触矩阵（热力图）？", "bool", True),
        Q("repro", "再跑一遍「独立复现 + 摘要 CSV」（原 examples/contact_reproduce.py）？",
          "bool", True, help="一次性给出平均接触数/接触概率/占据率/配位数等全部摘要量"),
        Q("compare", "再和工具引擎逐帧对拍？", "bool", False,
          help="同一批帧比接触对数、最小间距、平均接触数（应当一致到 1e-9）"),
    ]

    def _run(cfg):
        u = cfg["u"]
        a = cfg["ag"]
        b = u.select_atoms(cfg["select_b"]) if cfg.get("select_b") else None
        res = test_contact(u, a, b, cutoff=cfg["cutoff"], mode=cfg["mode"],
                           frames=cfg["frames"], prefix=cfg["prefix"])
        if cfg.get("matrix"):
            test_contact_matrix(u, a, b, cutoff=cfg["cutoff"],
                                frames=cfg["frames"],
                                prefix=cfg["prefix"] + "_matrix")
        if cfg.get("repro", True):
            test_contact_reproduce(u, a, b, cutoff=cfg["cutoff"],
                                   mode=cfg["mode"], frames=cfg["frames"],
                                   prefix=cfg["prefix"] + "_repro",
                                   compare=bool(cfg.get("compare")))
        elif cfg.get("compare"):
            compare_engine(u, a, b, cutoff=cfg["cutoff"], mode=cfg["mode"],
                           frames=cfg["frames"])
        return res

    return run_module("接触分析（contact）", _run, specs,
                      sub="r < cutoff 算接触；同组无序对、异组 A×B；"
                          "含独立复现与引擎对拍",
                      default_frames=100, default_prefix="contact")


if __name__ == "__main__":
    import sys as _sys

    try:
        from ._menu import main_cli
    except ImportError:
        from _menu import main_cli

    _sys.exit(main_cli(menu))
