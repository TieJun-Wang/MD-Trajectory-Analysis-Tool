# -*- coding: utf-8 -*-
r"""体系信息 / 组分识别 / 能力预判（工具模块：``mdta.systeminfo``、
``mdta.selection``、``mdta.capabilities``、``mdta.feasible``）。

**做什么**：把打开一个体系时"程序先看一遍"的那几件事做成脚本，一次看清：

1. **体系概况**（``mdta.systeminfo.describe_system`` → ``info_text()``）：
   原子数 / 残基 / 段 / 帧数 / 时间间隔 / 模拟盒 / 体积 / 总质量 / 净电荷；
2. **明细表**（``info_tables``）：文件与规模、时间轴、模拟盒与质量、拓扑与属性、
   组分构成、分子识别、元素组成、常见 residue / 原子名、分子链信息；
3. **组分识别**（``mdta.selection.identify_component``）：每个组分到底是什么
   （``正构烷烃 C60H122`` / ``H2O`` / ``Na+`` …）—— 界面「组分胶囊」上那段文字的
   来源，也是接触 / RDF / 界面能不能跑的依据；
4. **能力检测**（``mdta.capabilities.detect``）：有坐标 / 多帧 / 键 / 质量 / 电荷 /
   分子归属 / 周期盒吗？警告写清楚（"没有周期盒 → 密度、面内 RDF、界面、BOO 不可用"）；
5. **分析项预判**（``mdta.feasible.preflight``）：14 项里**哪些能跑、哪些会被跳过、
   为什么** —— 界面「分析功能」里置灰的那些项就是按它判定，"先预判再选，别等跑完才知道"。

**导出**：``system__components.csv``（组分识别表）、``system__feasible.csv``
（各项可行性与原因）。

**单独跑**：``python test/my_test/system.py``
"""

from __future__ import annotations

try:
    from ._common import out_dir
    from ._menu import main_cli, run_analysis_menu
except ImportError:
    from _common import out_dir
    from _menu import main_cli, run_analysis_menu

NAME = "system"
TITLE = "体系信息与能力预判"
SUB = "体系概况 / 组分识别 / 能力检测 / 14 项分析的可行性预判（界面置灰的依据）"


def test_system(eng, outdir=None, formats=("csv",), **over):
    """打印体系信息、组分识别、能力检测与可行性预判，并导出两张 CSV。"""
    import pandas as pd

    from mdta import capabilities as C
    from mdta import feasible as F
    from mdta import systeminfo as SI
    from mdta.selection import identify_component

    az = eng.az
    info = az.info

    # ---------------------------------------------------------- ① 体系概况
    print(f"\n— {TITLE} —")
    print(az.info_text())

    # ---------------------------------------------------------- ② 明细表
    for tab in SI.info_tables(info):
        rows = tab.get("rows") or []
        if not rows:
            continue
        print(f"\n[{tab['title']}]")
        for row in rows:
            print("  " + "  ".join(str(c) for c in row))

    # ---------------------------------------------------------- ③ 组分识别
    print("\n[组分识别]（界面「组分胶囊」上那段文字就是这里来的）")
    comp_rows = []
    for name, ag in eng.catalog.items():
        ic = identify_component(ag) if ag is not None and ag.n_atoms else {}
        chosen = "✓ 参与分析" if name in eng.components else "（未勾选）"
        print(f"  {name:<22} {ag.n_atoms:>7,} 原子  "
              f"{ic.get('label', '?')} · {ic.get('formula', '')} · "
              f"{ic.get('detail', '')}  {chosen}")
        comp_rows.append({"组分": name, "原子数": int(ag.n_atoms),
                          "种类": ic.get("kind", ""), "识别": ic.get("label", ""),
                          "分子式": ic.get("formula", ""),
                          "分子数": ic.get("n_molecules", ""),
                          "每分子原子数": ic.get("atoms_per_molecule", ""),
                          "参与分析": name in eng.components})

    # ---------------------------------------------------------- ④ 能力检测
    cap = C.detect(az)
    print(f"\n[能力检测] {cap.get('mode_label', '')}")
    for key, item in (cap.get("items") or {}).items():
        mark = "✓" if item.get("ok") else "✗"
        print(f"  {mark} {item.get('label', key):<8} {item.get('detail', '')}")
    for warn in cap.get("warnings") or []:
        print(f"  ⚠️ {warn}")

    # ---------------------------------------------------------- ⑤ 可行性预判
    verdicts = F.preflight(az)
    print("\n[分析项预判]（界面「分析功能」里置灰的项就是这些 —— 先预判再选）")
    feas_rows = []
    for name, v in verdicts.items():
        ok = bool(v.get("ok"))
        print(f"  {'✓ 可跑' if ok else '✗ 会跳过'}  {name:<11} {v.get('reason', '')}")
        feas_rows.append({"分析项": name, "可跑": ok,
                          "级别": v.get("level", ""), "原因": v.get("reason", "")})

    # ---------------------------------------------------------- 导出
    target = str(outdir or out_dir())
    files = []
    if formats:
        import os

        for fname, df in (("system__components.csv", pd.DataFrame(comp_rows)),
                          ("system__feasible.csv", pd.DataFrame(feas_rows))):
            path = os.path.join(target, fname)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            files.append(path)
        print("\n  导出：" + "、".join(os.path.basename(f) for f in files)
              + f"  →  {target}")
    n_ok = sum(1 for v in verdicts.values() if v.get("ok"))
    return {"name": NAME, "panels": 0, "summary": {"可跑项": n_ok,
                                                   "总项数": len(verdicts)},
            "files": files, "seconds": eng.last_seconds, "res": None}


SPECS: list = []


def menu():
    return run_analysis_menu(NAME, TITLE, SPECS, test_system, sub=SUB,
                             default_frames=1)


if __name__ == "__main__":
    raise SystemExit(main_cli(menu))
