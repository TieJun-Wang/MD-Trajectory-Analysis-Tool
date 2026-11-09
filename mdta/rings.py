# -*- coding: utf-8 -*-
"""环结构识别：环的**大小分布** + 苯环（6 元芳香碳环）判定。

为什么需要单独一个模块
----------------------
``recognize.py`` 里的 ``n_rings`` 只是**环秩**（E − V + 1），它知道"有几个独立环"，
但不知道"环是 6 元的还是 5 元的"、"是不是苯环那样的芳香六元环"。
Materials Studio 里石墨烯/苯环画的是带虚线圈的六元环 —— 要回答"能不能认出苯环、
有没有把普通环认错"，需要真正枚举环并检查每个环的原子与键长。

判定口径（**只做确定性推断**，不外推）
------------------------------------
- 环：把成键图当作无向图，取**最小环基**（networkx 的 minimum_cycle_basis，
  即 SSSR 口味的最小权重环），每个环给出原子序列与环大小。
- 6 元碳环：环大小 = 6 且 6 个原子都是 C。
- **苯环（芳香六元碳环）**：6 元碳环且**全部 6 根环键长 ≤ ``ARO_MAX_BOND``（1.50 Å）**。
  苯/石墨的 C–C ≈ 1.39–1.45 Å；环己烷的 C–C ≈ 1.53–1.55 Å，因此不会被误判成苯环。
"""
from __future__ import annotations

import numpy as np

#: 苯环判定：6 元碳环内所有环键的**最大**键长上限（Å）。
#: 苯/石墨烯 1.39–1.45；环己烷 1.53+；取 1.50 能把两者分开。
ARO_MAX_BOND = 1.50


def _bond_list(u, atoms_idx=None):
    """当前体系的键：(N, 2) int 数组；``atoms_idx`` 给定时只保留两端都在其中的键。"""
    try:
        b = np.asarray([(int(x[0].index), int(x[1].index)) for x in u.bonds],
                       dtype=np.int64)
    except Exception:                                        # noqa: BLE001
        return np.zeros((0, 2), dtype=np.int64)
    if atoms_idx is not None and b.size:
        keep = np.zeros(int(u.atoms.n_atoms), dtype=bool)
        keep[np.asarray(atoms_idx, dtype=np.int64)] = True
        b = b[keep[b[:, 0]] & keep[b[:, 1]]]
    return b


def ring_cycles(adj, n: int) -> list[list[int]]:
    """最小环基（SSSR 口味）。没有 networkx 时退回简单 DFS 环（只用于兜底）。"""
    try:
        import networkx as nx
    except Exception:                                        # noqa: BLE001
        nx = None
    if nx is not None:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((int(a), int(b)) for a in range(n) for b in adj[a])
        try:
            return [list(c) for c in nx.minimum_cycle_basis(g)]
        except Exception:                                    # noqa: BLE001
            return [list(c) for c in nx.cycle_basis(g)]
    # 兜底：DFS 找最小的几个环（不保证 SSSR，只避免依赖缺失时直接崩）
    out: list[list[int]] = []
    seen_edges = set()
    for s in range(n):
        for a in adj[s]:
            e = (min(s, a), max(s, a))
            if e in seen_edges:
                continue
            seen_edges.add(e)
            prev, cur = s, a
            path = [s, a]
            for _ in range(40):
                nxt = [x for x in adj[cur] if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                if cur == s:
                    out.append(path)
                    break
                if cur in path:
                    break
                path.append(cur)
    return out



def six_cycles(adj, n: int, max_cycles: int = 200000) -> list[list[int]]:
    """枚举**全部 6 元环**（简单、无重复），不依赖 networkx 的最小环基。

    为什么不用 ``nx.minimum_cycle_basis``：石墨烯这种二维网络上一个 DFS 基本环基
    给的是几十上百元的"大圈"，而最小环基在几千条边的图上要跑很久（实测 1700 原子
    的石墨片超过 1 分钟）。这里直接枚举长度 6 的简单环：每个环只从**最小下标**的
    原子出发搜一次，度数 ≤3 时总代价很小（1700 原子 ≈ 0.2 s）。
    """
    out: list[list[int]] = []
    for s in range(n):
        # 只从小下标出发，避免同一个环被枚举 6 次
        stack = [(s, [s])]
        while stack:
            v, path = stack.pop()
            if len(path) == 6:                       # 6 个原子 = 6 元环
                # 同一个环会沿两个方向各找到一次：只保留下标较小的一侧
                if s in adj[v] and path[1] < path[5]:
                    out.append(list(path))
                    if len(out) >= max_cycles:
                        return out
                continue
            for w in adj[v]:
                if w == s or w < s:                  # 最小下标必须是起点
                    continue
                if w in path:
                    continue
                stack.append((w, path + [w]))
    return out

def analyze_rings(universe, atoms_idx=None, *, coords=None) -> dict:
    """识别体系（或指定原子子集）里的环，返回大小分布与苯环统计。

    参数
    ----
    atoms_idx
        ``None`` = 整个体系；给定时只在这个子集内找环（分子内环）。
    coords
        坐标（Å）；``None`` 时用当前帧坐标。用于判断环键长。

    返回
    ----
    ``{"n_rings", "sizes": {环大小: 个数}, "n_ring6", "n_benzene",
       "n_aromatic_atoms", "rings": [(原子下标列表, 是否苯环)] 或 None}``
    """
    idx = (np.arange(int(universe.atoms.n_atoms), dtype=np.int64)
           if atoms_idx is None else np.asarray(atoms_idx, dtype=np.int64))
    pos = np.asarray(coords if coords is not None else universe.atoms.positions,
                     dtype=float)
    # 局部编号：0..n-1
    if atoms_idx is not None:
        remap = {int(a): i for i, a in enumerate(idx)}
        bonds = _bond_list(universe, idx)
        edges = [(remap[int(a)], remap[int(b)]) for a, b in bonds]
        local_pos = pos[idx]
    else:
        edges = [tuple(map(int, e)) for e in _bond_list(universe)]
        local_pos = pos
    n = int(idx.size)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    # 每个键的长度（判苯环用）
    blen = {}
    for a, b in edges:
        d = float(np.linalg.norm(local_pos[a] - local_pos[b]))
        blen[(min(a, b), max(a, b))] = d

    # 环秩（E − V + 连通分量数）：快的"有几个独立环"
    seen = [False] * n
    n_comp = 0
    for s0 in range(n):
        if seen[s0] or not adj[s0]:
            continue
        n_comp += 1
        st = [s0]
        seen[s0] = True
        while st:
            v = st.pop()
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    st.append(w)
    rank = max(0, len(edges) - n + n_comp)

    try:
        els = [str(x) for x in universe.atoms.elements]
    except Exception:                                        # noqa: BLE001
        els = [""] * int(universe.atoms.n_atoms)

    # 6 元环（苯环判定的基础）
    sixes = six_cycles(adj, n)
    n_ring6 = len(sixes)
    n_benzene = 0
    rings_out = []
    for cyc in sixes:
        all_c = all(els[int(idx[i])] == "C" for i in cyc)
        worst = 0.0
        if all_c:
            for k in range(6):
                a, b = cyc[k], cyc[(k + 1) % 6]
                worst = max(worst, blen.get((min(a, b), max(a, b)), 99.0))
        is_benzene = bool(all_c and worst <= ARO_MAX_BOND)
        if is_benzene:
            n_benzene += 1
        rings_out.append(([int(idx[i]) for i in cyc], is_benzene))

    # 环大小分布：小图才枚举（大网络的基本环基既慢又不可解释）
    sizes = {6: n_ring6}
    if n <= 400:
        try:
            cycles = [c for c in ring_cycles(adj, n) if 3 <= len(c) <= 80]
            sizes = {}
            for c in cycles:
                sizes[len(c)] = sizes.get(len(c), 0) + 1
        except Exception:                                    # noqa: BLE001
            pass

    return {
        "n_rings": int(max(rank, n_ring6)),
        "rank": int(rank),
        "sizes": {int(k): int(v) for k, v in sorted(sizes.items())},
        "n_ring6": int(n_ring6),
        "n_benzene": int(n_benzene),
        "atoms_in_rings": np.array(
            sorted({int(idx[i]) for c, _b in rings_out for i in c}),
            dtype=np.int64),
        "rings": rings_out,
    }
