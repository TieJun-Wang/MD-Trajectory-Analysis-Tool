# -*- coding: utf-8 -*-
"""交互式输入菜单 + 命令行参数（一份规格，两种用法）。

为什么单独一层
--------------
`mda_test` 里每个分析模块都要问同一批东西：拓扑文件、轨迹文件、分析对象（组分 /
选择串）、帧怎么取；再各自加 1–3 个自己的参数（cutoff、dr、参考轴…）。
以前只能改脚本里的常量或调 Python 函数，用起来门槛高。这里把它做成一份
**问题规格**（:class:`Q`），两种入口共用：

- **交互菜单**（`python test/mda_test/contact.py`，不带参数）：
  一步步问，显示默认值、列出体系里有哪些组分、最后给一份摘要让你确认；
- **命令行**（`python test/mda_test/contact.py --top a.xtd --trj a.trj --frames 100 -y`）：
  规格自动变成 `--cutoff` / `--mode` 这类开关，适合批处理。

两种方式得到的 ``cfg`` 完全一样，模块只负责"拿 cfg 干活"。
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ------------------------------------------------------------------ 问题规格
class Q:
    """一个问题：既是菜单的一步，也是命令行的一个开关。

    ``kind``：``text`` / ``int`` / ``float`` / ``bool`` / ``choice``。
    """

    def __init__(self, key: str, label: str, kind: str = "text", default=None, *,
                 cli: str | None = None, choices=None, lo=None, hi=None,
                 help: str = "", show_default: bool = True):
        self.key = key
        self.label = label
        self.kind = kind
        self.default = default
        self.cli = cli or ("--" + key.replace("_", "-"))
        self.choices = choices            # [(value, 说明), ...]（kind="choice" 时）
        self.lo = lo
        self.hi = hi
        self.help = help
        self.show_default = show_default

    # ---------------------------------------------------------- 值
    def cast(self, raw):
        """把用户输入的字符串转成值（非法时抛 ValueError）。"""
        if self.kind == "int":
            return int(str(raw).strip())
        if self.kind == "float":
            return float(str(raw).strip())
        if self.kind == "bool":
            s = str(raw).strip().lower()
            if s in ("", "y", "yes", "1", "true", "t", "是"):
                return True
            if s in ("n", "no", "0", "false", "f", "否"):
                return False
            raise ValueError("请输入 y/n")
        if self.kind == "choice":
            s = str(raw).strip()
            for v, _txt in (self.choices or []):
                if s == str(v):
                    return v
            idx = int(s) - 1               # 也允许按序号选
            opts = self.choices or []
            if 0 <= idx < len(opts):
                return opts[idx][0]
            raise ValueError(f"可选：{' / '.join(str(v) for v, _ in opts)} 或 1..{len(opts)}")
        return str(raw).strip()

    def validate(self, v):
        if self.lo is not None and isinstance(v, (int, float)) and v < self.lo:
            raise ValueError(f"不能小于 {self.lo}")
        if self.hi is not None and isinstance(v, (int, float)) and v > self.hi:
            raise ValueError(f"不能大于 {self.hi}")
        return v

    def default_text(self) -> str:
        if self.default is None:
            return ""
        if self.kind == "bool":
            return "Y" if self.default else "N"
        if self.kind == "choice":
            for v, txt in (self.choices or []):
                if str(v) == str(self.default):
                    return f"{v}（{txt}）"
        return str(self.default)


# ------------------------------------------------------------------ 读输入
def _readline() -> str | None:
    """读一行；EOF / Ctrl-C 返回 None（调用方据此走默认值或退出）。

    管道/重定向（非 tty）时输入不回显，这里补一个换行，免得提示与下文接在一行。
    """
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return None
    if line == "":                       # EOF
        return None
    try:
        if not sys.stdin.isatty():
            print()
    except Exception:                    # noqa: BLE001
        pass
    return line.rstrip("\r\n")


def _clean_path(s: str) -> str:
    """Windows 拖拽会带引号/空格，这里统一清掉。"""
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.strip()


def banner(title: str, sub: str = ""):
    print("\n" + "=" * 68)
    print(f"  {title}")
    if sub:
        print(f"  {sub}")
    print("=" * 68)


def note(text: str):
    print(f"   {text}")


def ask_text(label: str, default=None, hint: str = "") -> str:
    d = "" if default is None else str(default)
    tail = f" [默认: {d}]" if d else ""
    print(f"\n{label}{tail}")
    if hint:
        print(f"   （{hint}）")
    print("> ", end="", flush=True)
    line = _readline()
    if line is None:
        print(d)
        return d
    line = line.strip()
    return line if line else d


def ask_path(label: str, default=None, *, must_exist=True, allow_empty=False,
             hint: str = "") -> str:
    d = "" if default is None else str(default)
    tail = f" [默认: {d}]" if d else (" [留空]" if allow_empty else "")
    while True:
        print(f"\n{label}{tail}")
        if hint:
            print(f"   （{hint}）")
        print("> ", end="", flush=True)
        line = _readline()
        if line is None:
            if must_exist and d and not os.path.exists(d):
                raise SystemExit(f"默认文件不存在：{d}")
            return d
        got = _clean_path(line) or d
        if not got:
            if allow_empty:
                return ""
            print("   ✗ 必须给一个文件")
            continue
        if must_exist and not os.path.exists(got):
            print(f"   ✗ 找不到文件：{got}")
            continue
        return got


def ask_value(spec: Q, default=None):
    """按规格问一个值（带回显默认、非法重问）。"""
    d = spec.default if default is None else default
    while True:
        if spec.kind == "choice":
            print(f"\n{spec.label}")
            for i, (v, txt) in enumerate(spec.choices or [], 1):
                mark = " ←" if str(v) == str(d) else ""
                print(f"   {i}) {v:<8} {txt}{mark}")
            print(f"   选择 [默认: {d}] > ", end="", flush=True)
        elif spec.kind == "bool":
            print(f"\n{spec.label} (y/n) [默认: {'y' if d else 'n'}] > ",
                  end="", flush=True)
        else:
            tail = f" [默认: {d}]" if d is not None and spec.show_default else ""
            print(f"\n{spec.label}{tail}")
            if spec.help:
                print(f"   （{spec.help}）")
            print("> ", end="", flush=True)
        line = _readline()
        if line is None:                       # EOF → 用默认值
            print(d)
            return spec.validate(d)
        if not line.strip():
            return spec.validate(d)
        try:
            return spec.validate(spec.cast(line))
        except ValueError as exc:
            print(f"   ✗ {exc}")


# ------------------------------------------------------------------ 体系与帧
def sibling_trj(topology: str) -> str:
    """拓扑旁边有没有同名 `.trj`（Materials Studio 常见命名）。"""
    if not topology:
        return ""
    base = os.path.splitext(topology)[0]
    for suf in (".trj", ".TRJ", ".trr", ".xtc"):
        p = base + suf
        if os.path.exists(p):
            return p
    return ""


def first_existing(candidates):
    """按顺序挑第一个"存在"的候选；支持通配符。都没有则返回第一个（用于显示）。"""
    import glob

    cands = [c for c in (candidates or []) if c]
    for c in cands:
        if any(ch in str(c) for ch in "*?["):
            hits = sorted(glob.glob(str(c)))
            if hits:
                return hits[0]
        elif os.path.exists(str(c)):
            return str(c)
    return str(cands[0]) if cands else ""


def choose_group(u, default: str = "all", title: str = "分析对象"):
    """列出体系里的组分让人选（也可以直接敲选择串）。返回选择串。"""
    try:
        names = list(dict.fromkeys([str(r) for r in u.atoms.resnames]))
    except Exception:                                     # noqa: BLE001
        names = []
    print(f"\n{title}：")
    print(f"   0) 全部原子  {u.atoms.n_atoms:,} 原子")
    for i, nm in enumerate(names, 1):
        try:
            ag = u.select_atoms(f"resname {nm}")
            n_res = len(getattr(ag, "residues", []))
            print(f"   {i}) resname {nm:<10} {ag.n_atoms:>8,} 原子 / {n_res:>6,} 个分子")
        except Exception:                                 # noqa: BLE001
            print(f"   {i}) resname {nm}")
    print("   （也可以直接输入 MDAnalysis 选择串，如 \"element O\" / \"index 0:999\"）")
    print(f"   选择 [默认: {default}] > ", end="", flush=True)
    line = _readline()
    line = "" if line is None else line.strip()
    if not line:
        return default
    if line == "0":
        return "all"
    if line.isdigit() and names and 1 <= int(line) <= len(names):
        return f"resname {names[int(line) - 1]}"
    return line


def choose_frames(u, default_n: int = 0) -> list[int]:
    """帧怎么取：全部 / 均匀取 N 帧 / 指定区间。返回**等差**帧号列表。

    返回值一定是 ``range(start, stop, step)`` 的形式 —— MDAnalysis 的
    ``InterRDF`` / ``EinsteinMSD`` 只能用 ``start/stop/step`` 取帧，
    保持等差才能让"菜单里选的帧"和它们实际跑的帧完全一致。
    """
    n_all = len(u.trajectory)
    print(f"\n帧选择（轨迹共 {n_all} 帧）：")
    print(f"   1) 全部 {n_all} 帧")
    print("   2) 均匀取 N 帧（推荐：够统计又省时间）")
    print("   3) 指定区间 start / stop / step")
    print(f"   选择 [默认: {'2' if default_n else '1'}] > ", end="", flush=True)
    line = _readline()
    if line is None:
        line = "2" if default_n else "1"
    line = line.strip() or ("2" if default_n else "1")

    if line == "3":
        start = int(ask_text("起始帧", 0) or 0)
        e = ask_text("结束帧（不含，留空=到末尾）", "")
        stop = int(e) if e else n_all
        step = max(1, int(ask_text("帧间隔 step", 1) or 1))
        return list(range(start, stop, step)) or list(range(n_all))
    if line == "2":
        want = int(default_n or 0)
        if not want:
            want = int(ask_text("取多少帧", min(100, n_all)) or min(100, n_all))
        want = max(1, min(int(want), n_all))
        step = max(1, int(round((n_all - 1) / max(want - 1, 1))))
        return list(range(0, n_all, step))
    return list(range(n_all))


def print_summary(items):
    print("\n" + "-" * 68)
    print("  即将按以下设置运行：")
    for k, v in items:
        if isinstance(v, list):
            v = (f"{len(v)} 帧（{v[0]} … {v[-1]}，等差）" if v else "（空）")
        elif isinstance(v, bool):
            v = "是" if v else "否"
        elif isinstance(v, float):
            v = f"{v:g}"
        print(f"    {k:<20} {v}")
    print("-" * 68)


def confirm(default: bool = True) -> bool:
    print(f"\n开始运行？(y/n) [默认: {'y' if default else 'n'}] > ", end="", flush=True)
    line = _readline()
    if line is None or not line.strip():
        print("y" if default else "n")
        return default
    return line.strip().lower() in ("y", "yes", "1", "是", "")


# ------------------------------------------------------------------ 统一入口
def _common_questions():
    return [
        Q("top", "拓扑文件", "text",
          help=".xtd/.xsd/.pdb/.gro/.tpr…（可以把文件直接拖进来）"),
        Q("trj", "轨迹文件", "text",
          help=".trj/.trr/.xtc/.dcd…；留空 = 只读单帧结构"),
        Q("select", "分析对象（选择串）", "text", default="all"),
        Q("frames", "最多用多少帧（0 = 全部）", "int", default=0, lo=0),
        Q("outdir", "CSV 输出目录", "text", default="",
          help="留空 = 当前目录"),
        Q("prefix", "输出文件名前缀", "text", default="result"),
    ]


def add_args(ap, specs, common: bool = True):
    """把规格挂到 argparse 上（命令行用法）。"""
    if common:
        ap.add_argument("--top", default=None, help="拓扑文件（不给 = 用默认体系）")
        ap.add_argument("--trj", default=None,
                        help="轨迹文件（不给 = 用默认体系；--no-traj = 只读单帧结构）")
        ap.add_argument("--no-traj", action="store_true",
                        help="即使给了 --top 也只读单帧结构（忽略轨迹）")
        ap.add_argument("--select", default="all", help="分析对象选择串")
        ap.add_argument("--frames", type=int, default=0,
                        help="最多用多少帧（0=全部；N>0 时均匀取 N 帧）")
        ap.add_argument("--outdir", default="", help="CSV 输出目录")
        ap.add_argument("--prefix", default="result", help="输出文件名前缀")
    for s in specs:
        if s.kind == "bool":
            ap.add_argument(s.cli, dest=s.key, action="store_true",
                            default=bool(s.default), help=s.help or s.label)
            ap.add_argument("--no-" + s.key.replace("_", "-"), dest=s.key,
                            action="store_false", help=argparse_suppress())
        elif s.kind == "choice":
            ap.add_argument(s.cli, dest=s.key, default=s.default,
                            choices=[str(v) for v, _ in (s.choices or [])],
                            help=s.help or s.label)
        elif s.kind in ("int", "float"):
            ap.add_argument(s.cli, dest=s.key, type=(int if s.kind == "int" else float),
                            default=s.default, help=s.help or s.label)
        else:
            ap.add_argument(s.cli, dest=s.key, default=s.default,
                            help=s.help or s.label)
    ap.add_argument("-y", "--yes", action="store_true", help="不确认，直接跑")
    return ap


def argparse_suppress():
    import argparse

    return argparse.SUPPRESS


def _cfg_from_args(ns, specs) -> dict:
    cfg = {"top": ns.top, "trj": ns.trj, "select": ns.select,
           "frames_n": ns.frames, "outdir": ns.outdir, "prefix": ns.prefix}
    for s in specs:
        v = getattr(ns, s.key, s.default)
        if s.kind == "choice" and v is not None:
            v = s.cast(v)
        if s.kind == "bool":
            v = bool(v)
        cfg[s.key] = v
    return cfg


def _load(cfg):
    """按 cfg 里的 top/trj/select 把体系载进来。"""
    try:
        from ._common import build_universe
    except ImportError:
        from _common import build_universe
    if cfg.get("outdir"):
        os.environ["MDTA_MDA_OUT"] = str(cfg["outdir"])
    sel = cfg.get("select") or "all"
    return build_universe(cfg["top"], cfg["trj"], select=sel)


def run_module(title: str, run_fn, specs=(), *, sub: str = "",
               default_select: str = "all", default_frames: int = 0,
               default_prefix: str = "result", auto_confirm: bool = False,
               default_top=None, default_trj=None):
    """统一入口：**无参数 → 交互菜单**；**有参数 → 命令行**。

    ``run_fn(cfg)`` 拿到：``u`` / ``ag``（已按 select 选好）、``frames``
    （等差帧号列表）、``prefix`` / ``outdir``，以及每个规格问题的值 ``cfg[key]``。
    """
    import argparse

    specs = list(specs)

    # ================================================== ① 命令行
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser(
            description=title,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ap.add_argument("--top", default=None, help="拓扑文件（不给 = 用默认体系）")
        ap.add_argument("--trj", default=None,
                        help="轨迹文件（不给 = 用默认体系；--no-traj = 只读单帧结构）")
        ap.add_argument("--no-traj", action="store_true",
                        help="即使给了 --top 也只读单帧结构（忽略轨迹）")
        ap.add_argument("--select", default=default_select, help="分析对象选择串")
        ap.add_argument("--frames", type=int, default=default_frames,
                        help="最多用多少帧（0=全部；N>0 时均匀取 N 帧）")
        ap.add_argument("--outdir", default="", help="CSV 输出目录")
        ap.add_argument("--prefix", default="", help="输出文件名前缀")
        for s in specs:
            if s.kind == "bool":
                ap.add_argument(s.cli, dest=s.key, action="store_true",
                                default=bool(s.default), help=s.help or s.label)
            elif s.kind == "choice":
                ap.add_argument(s.cli, dest=s.key, default=str(s.default),
                                choices=[str(v) for v, _ in (s.choices or [])],
                                help=s.help or s.label)
            elif s.kind in ("int", "float"):
                ap.add_argument(s.cli, dest=s.key,
                                type=(int if s.kind == "int" else float),
                                default=s.default, help=s.help or s.label)
            else:
                ap.add_argument(s.cli, dest=s.key, default=s.default,
                                help=s.help or s.label)
        ap.add_argument("-y", "--yes", action="store_true", help="不确认直接跑")
        ns = ap.parse_args()

        # ⚠️ 以前 `--top/--trj` 默认空字符串，结果"不给参数"就变成"只读单帧结构"
        # （默认体系其实没被用上，20PE30 只读到 1 帧）。现在：不给 → 用默认体系，
        # 要单帧请显式 --no-traj。
        try:
            from ._common import DEFAULT_TRR, DEFAULT_XSD
        except ImportError:
            from _common import DEFAULT_TRR, DEFAULT_XSD
        top = ns.top or DEFAULT_XSD
        trj = "" if ns.no_traj else (DEFAULT_TRR if ns.trj is None else ns.trj)
        cfg = {"top": top, "trj": trj, "select": ns.select,
               "outdir": ns.outdir,
               "prefix": ns.prefix or default_prefix}
        for s in specs:
            v = getattr(ns, s.key, s.default)
            cfg[s.key] = s.cast(v) if (s.kind == "choice" and v is not None) else v
        print(f"\n=== {title} ===")
        u, ag = _load(cfg)
        cfg["u"], cfg["ag"] = u, ag
        cfg["frames"] = _frames_n(u, int(ns.frames))
        print_summary([("拓扑", cfg["top"]), ("轨迹", cfg["trj"] or "（单帧）"),
                       ("对象", cfg["select"]), ("帧", cfg["frames"])]
                      + [(s.label, cfg[s.key]) for s in specs])
        return run_fn(cfg)

    # ================================================== ② 交互菜单
    banner(title, sub)
    print("  直接回车 = 用方括号里的默认值；Ctrl-C 退出。")
    try:
        from ._common import DEFAULT_SELECT, DEFAULT_TRR, DEFAULT_XSD
    except ImportError:
        from _common import DEFAULT_SELECT, DEFAULT_TRR, DEFAULT_XSD

    # 模块可以指定自己的默认文件（拓扑 + 轨迹候选列表，按顺序挑存在的）
    top_cands = ([default_top] if isinstance(default_top, str)
                 else list(default_top or [])) or [DEFAULT_XSD]
    trj_cands = ([default_trj] if isinstance(default_trj, str)
                 else list(default_trj or [])) or [DEFAULT_TRR]
    default_top_path = first_existing(top_cands)
    if default_top and not os.path.exists(str(default_top)):
        print(f"  ⚠️ 默认拓扑不存在，改用：{default_top_path}")

    top = ask_path("① 拓扑文件（.xtd/.xsd/.pdb/.gro/.tpr…）", default_top_path,
                   hint=r"例如 C:\temp\dataset\1000water\1000water.xtd")
    guess = first_existing(trj_cands)
    if default_trj and not os.path.exists(str([default_trj] if isinstance(
            default_trj, str) else default_trj[0])):
        print(f"  ⚠️ 指定轨迹不存在（{default_trj if isinstance(default_trj, str) else default_trj[0]}），"
              f"自动改用：{guess}")
    if not guess:
        guess = sibling_trj(top)
    trj = ask_path("② 轨迹文件（留空 = 只读单帧结构）", guess,
                   must_exist=False, allow_empty=True,
                   hint=r"例如 C:\temp\dataset\1000water\1000water.trj"
                        "（.trj 会自动转 TRR，有缓存直接复用）")

    print("\n正在读取体系…")
    u, _ = _load({"top": top, "trj": trj, "select": "all", "outdir": ""})

    sel = choose_group(u, default_select if default_select != "all" else DEFAULT_SELECT)
    print(f"   分析对象：{sel!r} = {u.select_atoms(sel).n_atoms:,} 原子")
    extra = ask_text("③ 在对象里再筛原子（留空 = 不筛）", "",
                     hint=r'例如 "element O" 只算氧原子（水体系做 RDF 常用）')
    if extra:
        sel = f"({sel}) and ({extra})"
        print(f"   再筛后：{u.select_atoms(sel).n_atoms:,} 原子")
    ag = u.select_atoms(sel)

    frames = choose_frames(u, int(default_frames))

    cfg = {"top": top, "trj": trj, "select": sel, "u": u, "ag": ag,
           "frames": frames, "outdir": "", "prefix": ""}

    if specs:
        print("\n---- 本模块参数 ----")
    for s in specs:
        cfg[s.key] = ask_value(s)

    print("\n---- 输出 ----")
    cfg["outdir"] = ask_path("CSV 输出目录（留空 = 当前目录）", "",
                             must_exist=False, allow_empty=True)
    cfg["prefix"] = ask_text("输出文件名前缀（不带扩展名）", default_prefix)

    print_summary([("拓扑", top), ("轨迹", trj or "（单帧）"),
                   ("对象", sel), ("帧", frames),
                   ("输出目录", cfg["outdir"] or os.getcwd())]
                  + [(s.label, cfg[s.key]) for s in specs])
    if not (auto_confirm or confirm(True)):
        print("已取消。")
        return None

    if cfg["outdir"]:
        os.environ["MDTA_MDA_OUT"] = str(cfg["outdir"])
    return run_fn(cfg)


def main_cli(menu_fn):
    """``if __name__ == "__main__": sys.exit(main_cli(menu))``。

    只关心退出码：**不要**把 ``menu_fn()`` 的返回值（dict/ndarray）交给
    ``sys.exit``，否则 Python 会把整个结果当错误信息打到 stderr 并返回 1。
    """
    try:
        menu_fn()
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except SystemExit:
        raise
    except Exception as exc:                                  # noqa: BLE001
        import traceback

        print(f"\n✗ 运行失败：{type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return 1
    return 0


def _frames_n(u, n: int) -> list[int]:
    """命令行用：0=全部；N>0 = 均匀取 N 帧（等差，供 MDAnalysis 分析类取帧）。"""
    n_all = len(u.trajectory)
    n = int(n or 0)
    if n <= 0 or n >= n_all:
        return list(range(n_all))
    step = max(1, int(round((n_all - 1) / max(n - 1, 1))))
    return list(range(0, n_all, step))
