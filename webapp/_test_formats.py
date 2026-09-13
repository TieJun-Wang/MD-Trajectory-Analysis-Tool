# -*- coding: utf-8 -*-
r"""拓扑格式一致性验证：.tpr / .gro / .pdb 是否给出同一套化学键。

`_fmt_test/` 里是从 `.tpr` 转换出来的同名 `.gro` 与 `.pdb`（真实体系的完整副本），
用它们检验"没有键表时的自动猜键"能否复现 `.tpr` 的键表：

- 若 `.gro` 的键数、碎片数与 `.tpr` 一致，说明猜键判据（共价半径、
  离子不参与成键、虚拟位点不成键）是可靠的；
- `.pdb` 自带 CONECT 记录时，键应来自文件而**不再**猜键，且键数同样与 `.tpr` 一致；
- 组分识别也必须跨格式一致——注意 `.gro` 的 residue 名字段只有 5 个字符，
  6 字符的 GLYCAM 名（``BGLCNA``）会被截断成 ``BGLCN``，糖识别必须兼容；
- 顺带验证 `.top`（GROMACS 力场拓扑）被**明确拒绝**且报错可读。

用法:  python webapp\_test_formats.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mdta.io import TrajectoryError, load_trajectory          # noqa: E402
from mdta.plotting import set_agg_backend                     # noqa: E402
from mdta.selection import component_groups                   # noqa: E402

set_agg_backend()

# --- 数据位置解析：工作区 / 上一级 / 上一级 dataset（兼容一层或两层目录）---
_DATA_ROOTS = [ROOT, ROOT.parent, ROOT.parent / "dataset"]


def find_data(name: str) -> Path:
    """按文件名在候选根目录里找（含一层子目录），找不到就返回工作区下的预期路径。"""
    for base in _DATA_ROOTS:
        direct = base / name
        if direct.exists():
            return direct
        try:
            for hit in sorted(base.glob(f"*/{name}")):
                if hit.exists():
                    return hit
        except OSError:
            continue
    return ROOT / name

SYSTEMS = [
    ("adk_oplsaa", "AdK 蛋白水溶液"),
    ("md_biopolymer_nowater", "糖蛋白 + DOL"),
]

pass_n = 0
fails: list[str] = []


def ok(cond: bool, label: str, extra: str = "") -> None:
    global pass_n
    if cond:
        pass_n += 1
        print(f"  ✓ {label}" + (f"  {extra}" if extra else ""))
    else:
        fails.append(label)
        print(f"  ✗ {label}" + (f"  {extra}" if extra else ""))


def counts(mdt) -> tuple[int, int]:
    """返回 (键数, 片段数)。没有键表时返回 (-1, 片段数)。"""
    u = mdt.universe
    try:
        nb = len(u.bonds)
    except Exception:  # noqa: BLE001
        nb = -1
    return nb, len(u.atoms.fragments)


def has_conect(path: Path) -> bool:
    """pdb 是否自带 CONECT 键表。

    注意 CONECT 记录在**文件末尾**（可达数万行），不能只看前若干字节。
    """
    if path.suffix.lower() not in (".pdb", ".ent"):
        return False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("CONECT"):
                return True
    return False


print("=" * 74)
print("拓扑格式一致性验证 —— .tpr / .gro / .pdb")
print("=" * 74)

for stem, desc in SYSTEMS:
    tpr = find_data(f"{stem}.tpr")
    xtc = find_data(f"{stem}.xtc")
    gro = ROOT / "_fmt_test" / f"{stem}.gro"
    pdb = ROOT / "_fmt_test" / f"{stem}.pdb"
    if not tpr.is_file() or not xtc.is_file():
        print(f"\n[跳过] {stem}：轨迹文件不在")
        continue
    print(f"\n【{stem}】{desc}")

    t0 = time.time()
    ref = load_trajectory(str(tpr), str(xtc))
    n_atoms = ref.universe.atoms.n_atoms
    n_b_ref, n_f_ref = counts(ref)
    print(f"  .tpr  原子 {n_atoms:,}  键 {n_b_ref:,}  片段 {n_f_ref:,}  "
          f"({time.time() - t0:.1f}s)")
    ok(n_b_ref > 0, f"{stem}: .tpr 自带键表", f"{n_b_ref:,} 键")

    for fmt, path in ((".gro", gro), (".pdb", pdb)):
        if not path.is_file():
            print(f"  [跳过] {fmt}：文件不存在 {path}")
            continue
        t0 = time.time()
        try:
            mdt = load_trajectory(str(path), str(xtc))
        except TrajectoryError as e:
            ok(False, f"{stem}: 读取 {fmt}", str(e).splitlines()[0])
            continue
        dt = time.time() - t0
        n_b, n_f = counts(mdt)
        conect = has_conect(path)
        print(f"  {fmt:5} 原子 {mdt.universe.atoms.n_atoms:,}  键 {n_b:,}  "
              f"片段 {n_f:,}  ({dt:.1f}s)")
        ok(mdt.universe.atoms.n_atoms == n_atoms,
           f"{stem}: {fmt} 原子数与 .tpr 一致", f"{n_atoms:,}")
        ok(n_b == n_b_ref, f"{stem}: {fmt} 键数与 .tpr 一致",
           f"{n_b:,} vs {n_b_ref:,}"
           + ("（来自 CONECT 记录）" if conect else "（来自自动猜键）"))
        ok(n_f == n_f_ref, f"{stem}: {fmt} 片段数与 .tpr 一致",
           f"{n_f:,} vs {n_f_ref:,}")
        gi = mdt.bond_guess_info
        if conect:
            # pdb 自带 CONECT：键应来自文件，**不应该**再猜键
            ok(not gi, f"{stem}: {fmt} 有 CONECT 时不再猜键",
               "键来自文件的 CONECT 记录")
        else:
            ok(bool(gi), f"{stem}: {fmt} 报告了猜键信息",
               f"猜键 {gi.get('bonds', 0):,} 键 / {gi.get('fragments', 0):,} 片段"
               if gi else "无")
        # 组分识别也必须一致
        c_ref = {k: v.n_atoms for k, v in component_groups(ref.universe).items()}
        c_new = {k: v.n_atoms for k, v in component_groups(mdt.universe).items()}
        ok(c_ref == c_new, f"{stem}: {fmt} 组分识别与 .tpr 一致",
           " / ".join(f"{k}({v})" for k, v in c_new.items())
           + ("" if c_ref == c_new else f"  ← .tpr 为 "
              + " / ".join(f"{k}({v})" for k, v in c_ref.items())))

print("\n【.gro 会截断 residue 名，糖识别必须兼容】")
from mdta.selection import is_sugar_resname                       # noqa: E402

# GROMACS .gro 的 residue 名字段只有 5 字符：BGLCNA → BGLCN
ok(is_sugar_resname("BGLCN"), "截断名 BGLCN 仍识别为糖", "（.gro 里的 BGLCNA）")
ok(is_sugar_resname("BGLCNA"), "完整名 BGLCNA 识别为糖")
ok(not is_sugar_resname("ALA") and not is_sugar_resname("MARG")
   and not is_sugar_resname("MHSE") and not is_sugar_resname("GLY"),
   "标准氨基酸与 CHARMM 变体不被误判为糖")
ok(not is_sugar_resname("DOL") and not is_sugar_resname("SOL"),
   "DOL / SOL 不被误判为糖")

print("\n【.top 必须被明确拒绝】")
top = find_data("topol.top")
xtc = find_data("md_dt200.xtc")
if top.is_file():
    try:
        load_trajectory(str(top), str(xtc)).universe.atoms.n_atoms
        ok(False, ".top 被拒绝", "竟然读取成功")
    except TrajectoryError as e:
        msg = str(e)
        ok("top" in msg.lower() and ("力场" in msg or "MDAnalysis" in msg),
           ".top 被拒绝且提示可读", msg.splitlines()[0][:60])
    except Exception as e:  # noqa: BLE001
        ok(False, ".top 被拒绝", f"抛出的是 {type(e).__name__} 而不是 TrajectoryError")
else:
    print("  [跳过] 找不到 topol.top")

print("\n【原子数不匹配时的提示】")
bad = find_data("ab_4_16_cu.pdb")
if bad.is_file():
    try:
        load_trajectory(str(bad), str(xtc)).universe.atoms.n_atoms
        ok(False, "原子数不匹配被捕获", "竟然读取成功")
    except TrajectoryError as e:
        msg = str(e)
        ok("不匹配" in msg or "216" in msg, "原子数不匹配被捕获并说明", msg.splitlines()[0][:70])
else:
    print("  [跳过] 找不到 ab_4_16_cu.pdb")

print("\n" + "=" * 74)
print(f"通过 {pass_n} 项" + (f"，失败 {len(fails)} 项: {' | '.join(fails)}"
                           if fails else "，全部通过"))
print("=" * 74)
sys.exit(1 if fails else 0)
