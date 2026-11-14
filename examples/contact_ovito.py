#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 **OVITO** 复现本工具「接触分析」里的两个核心量。

⚠️ 本仓库的开发机上**没有装 OVITO**（``pip install ovito`` 才能 import），
所以这份脚本是**按 OVITO 3.10+ 的 Python API 写的**，没有在本机跑过；
MDAnalysis 那条路（``examples/contact_reproduce.py``）是逐位对拍验证过的，
要"复现结果"请优先用它，OVITO 更适合可视化 + 交叉检查。

固定采用的定义（与工具一致）
----------------------------
``r < cutoff``（默认 5.0 Å）、**最小镜像**距离（OVITO 用 simulation cell 的
周期性处理，等价）。OVITO 的 ``Coordination`` 属性是"该原子在 cutoff 内的
邻居总数（不分类型）"，所以要用**两次计算相减**才能只数 A–B：

    A–B 接触数(每个 A 原子) = c_all(A) − c_AA(A)

``c_all(A)`` = 全体系里 A 原子的邻居数；``c_AA(A)`` = 把 B 全删掉后 A 原子的
邻居数。两者相减正好是"每个 A 原子周围有几个 B 原子"。

用法
----
    # 轨迹/结构文件里 A、B 用 **particle type** 区分（LAMMPS dump、带 type 的 xyz…）
    python examples/contact_ovito.py dump.lammpstrj --type-a 1 --type-b 2 --cutoff 5
    python examples/contact_ovito.py dump.lammpstrj --type-a 1 --type-b 2 \
           --frame 0            # 只看第 0 帧

    # 想用 .xtc / .trr / .gro：OVITO Basic 不认，需要 OVITO Pro 的 MDAnalysis 导入器；
    # 或者先用 MDAnalysis 转成带 type 的 xyz/dump。

输出
----
· 每个 A 原子的 A–B 邻居数（可选存成 numpy 文件）
· 平均值 = 工具的「平均接触数（每个 A 组原子）」
· 总接触对数 = Σ(A 的 B 邻居数) —— 与工具「接触对数」同一口径（异组 A×B 每对一次）
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="用 OVITO 复现接触分析的两个核心量")
    ap.add_argument("file", help="结构/轨迹文件（dump / xyz / …）")
    ap.add_argument("--type-a", type=int, required=True, help="A 组的 particle type")
    ap.add_argument("--type-b", type=int, required=True, help="B 组的 particle type")
    ap.add_argument("--cutoff", type=float, default=5.0, help="接触判据 (Å)，默认 5.0")
    ap.add_argument("--frame", type=int, default=0, help="第几帧（从 0 起）")
    ap.add_argument("--save", default=None, help="把每个 A 原子的邻居数存成 .npy")
    args = ap.parse_args(argv)

    try:
        from ovito.io import import_file
        from ovito.modifiers import (CoordinationAnalysisModifier,
                                     DeleteSelectedModifier, SelectTypeModifier)
    except Exception as exc:                                  # noqa: BLE001
        raise SystemExit(f"需要先安装 OVITO Python 包：pip install ovito\n({exc})")

    import numpy as np

    # ---- ① 全体系：A 原子的邻居总数（含 A–A、A–B）
    pipe_all = import_file(args.file)
    pipe_all.modifiers.append(
        CoordinationAnalysisModifier(cutoff=float(args.cutoff), number_of_bins=200))
    data_all = pipe_all.compute(frame=int(args.frame))
    types = np.asarray(data_all.particles["Particle Type"])
    c_all = np.asarray(data_all.particles["Coordination"], dtype=float)
    is_a = types == int(args.type_a)
    is_b = types == int(args.type_b)
    if not is_a.any() or not is_b.any():
        raise SystemExit(f"体系里没有 type {args.type_a} 或 {args.type_b} 的粒子")

    # ---- ② 只留 A：A 原子在 A 内部的邻居数（A–A）
    pipe_a = import_file(args.file)
    pipe_a.modifiers.append(SelectTypeModifier(types={int(args.type_b)}))
    pipe_a.modifiers.append(DeleteSelectedModifier())        # 删掉 B
    pipe_a.modifiers.append(
        CoordinationAnalysisModifier(cutoff=float(args.cutoff), number_of_bins=200))
    data_a = pipe_a.compute(frame=int(args.frame))
    c_aa = np.asarray(data_a.particles["Coordination"], dtype=float)
    # 删完 B 之后原子序号会变：只留 A 时 A 是**原顺序**里的 A（OVITO 保序）
    if c_aa.size != int(is_a.sum()):
        raise SystemExit(f"只留 A 之后粒子数 {c_aa.size} 与 A 的个数 "
                         f"{int(is_a.sum())} 不符，OVITO 版本行为可能不同")

    n_ab = c_all[is_a] - c_aa                        # 每个 A 原子的 B 邻居数
    n_pairs = float(np.sum(n_ab))                    # 总 A–B 接触对数
    mean_per_a = float(np.mean(n_ab))

    print(f"文件 {args.file}  帧 {args.frame}  cutoff {args.cutoff} Å")
    print(f"  A 原子 {int(is_a.sum())} 个，B 原子 {int(is_b.sum())} 个")
    print(f"  总 A–B 接触对数          = {n_pairs:.0f}"
          f"   ← 对应工具「接触对数」")
    print(f"  平均接触数（每个 A 原子）  = {mean_per_a:.6f}"
          f"   ← 对应工具「平均接触数（每个 A 组原子）」")
    print(f"  每个 A 原子的邻居数：min {n_ab.min():.0f} / "
          f"中位数 {np.median(n_ab):.0f} / max {n_ab.max():.0f}")
    if args.save:
        np.save(args.save, n_ab)
        print(f"  已存 {args.save}")

    print("\n提示：OVITO 的 CoordinationAnalysis 用的是**截断半径 + 周期性**的邻居搜索，"
          "与工具的 capped_distance/self_capped_distance 语义一致；\n"
          "      但要拿到「接触概率」「接触对占据率」「接触矩阵」，OVITO 没有现成输出，"
          "请用 examples/contact_reproduce.py（MDAnalysis）或本工具本身。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
