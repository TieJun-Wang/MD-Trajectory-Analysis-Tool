# -*- coding: utf-8 -*-
"""最小可用示例 —— 对应设计大纲第 32 章的"第一版开发目标"。

目标：
    读取一个真实的 .tpr + .xtc
        -> 查看体系信息
        -> 选择一条高分子链
        -> 计算 Rg
        -> 绘制 Rg-Time
        -> 导出 CSV

运行::

    python examples/quickstart.py                      # 用同目录的 adk_oplsaa
    python examples/quickstart.py sys.tpr sys.xtc
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from mdta.analysis import conformation as conf          # noqa: E402
from mdta.export import export_csv, export_png          # noqa: E402
from mdta.io import load_trajectory                     # noqa: E402
from mdta.preprocess import select_frames               # noqa: E402
from mdta.selection import component_groups, largest_chains  # noqa: E402
from mdta.systeminfo import describe_system             # noqa: E402


def main(argv: list[str]) -> int:
    top = argv[1] if len(argv) > 1 else os.path.join(_ROOT, "adk_oplsaa.tpr")
    xtc = argv[2] if len(argv) > 2 else os.path.join(_ROOT, "adk_oplsaa.xtc")
    outdir = os.path.join(_ROOT, "analysis_results")

    # 1) 读取轨迹
    mdt = load_trajectory(top, xtc if os.path.isfile(xtc) else None)
    print(f"读取成功: {mdt}")

    # 2) 查看体系信息
    print(describe_system(mdt).format_text(max_chain_types=6))

    # 3) 选择一条"链"：默认取最大的分子/链
    #    高分子体系里最大的链通常就是待研究的高分子链。
    #    也可以用 component_groups(u) 按组分（protein / polymer / water / ion）选。
    chains = largest_chains(mdt.universe, n=1, min_atoms=20)
    chain = chains[0] if chains else mdt.universe.atoms
    label = f"最大链 ({chain.n_atoms} 原子)"
    print(f"[选择] {label}")
    for name, ag in component_groups(mdt.universe).items():
        print(f"        组分 {name}: {ag.n_atoms} 原子")

    # 4) 选择参与统计的帧（这里全部 10 帧都用）
    sel = select_frames(mdt.times_ps)
    print(f"[帧选择] {sel.describe()}")

    # 5) 计算 Rg
    res = conf.analyze_rg(mdt, chain, sel, label=label)
    print("\n[结果] 回转半径 Rg")
    for k in ("Rg mean", "Rg std", "Rg min", "Rg max", "Rg 块平均标准误"):
        if k in res.summary:
            print(f"    {k:<16} = {res.summary[k]:.4f}")

    # 6) 绘图 + 导出
    os.makedirs(outdir, exist_ok=True)
    csv_paths = export_csv(res, outdir, prefix="Rg")
    png_path = export_png(res, outdir, prefix="Rg")
    print("\n[导出]")
    for p in csv_paths + [png_path]:
        print(f"    {p}")

    print("\n完成。可以打开 Rg.png 查看 Rg–Time 曲线。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
