# -*- coding: utf-8 -*-
"""Materials Studio ``.trj`` 轨迹支持。

MDAnalysis **不认识** ``.trj``（它是 MS 的私有二进制格式），所以这里走"先转换、
再按普通轨迹读取"的路子：

    .trj  ──(pymsdump 的 ms_trj 解析器 + xtcwriter)──▶  缓存 .xtc  ──▶  正常分析

需要的两个**纯 Python** 模块（无需编译）**都在本仓库的 `pymsdump/` 里**
（1.0.4 起把 `.trj` 解析器也集成进来了，换机器不用再单独拷 md_convert）：

    pymsdump/ms_trj.py        Materials Studio .trj 解析器（MSTrajectory）
    pymsdump/xtcwriter.py     纯 Python XTC 写入器（XTCWriter）

查找顺序（第一个找到的目录生效）：

1. 环境变量 ``MDTA_MS_CONVERT``（多个目录用 ``;`` 分隔，最省事）；
2. 轨迹文件所在目录及其附近的常见布局：``<dir>``、``<dir>/md_convert``、
   ``<dir>/pymsdump``、``<dir>/trj2xtc/pymsdump``、``<dir>/../md_convert``；
3. 本工具目录下的 ``md_convert`` / ``pymsdump``；
4. 用户桌面下的有限深度搜索（``~/Desktop`` 里找 ``xtcwriter.py`` / ``ms_trj.py``）。

转好的 XTC 会按「源文件绝对路径 + 大小 + 修改时间」做键缓存到
``_convert_cache/``，第二次打开同一份 ``.trj`` 就是秒开。

为什么不做"边转边算"：MS 的 ``.trj`` 是顺序格式，逐帧解析本身很便宜，但我们的
分析要多次随机访问帧；先落一份 XTC 才能复用已有的缓存/随机访问机制。
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import Callable, Iterable

#: 模块名 → 需要的目录（任一目录里能找到对应文件即可）
TRJ_MODULE = "ms_trj.py"
XTC_MODULE = "xtcwriter.py"

_module_cache: dict[str, dict[str, str | None]] = {}
_dir_cache: dict[str, list[str]] = {}


def _candidate_dirs(hint: str | None = None) -> list[str]:
    """按优先级给出所有可能的模块目录（去重）。"""
    out: list[str] = []
    env = os.environ.get("MDTA_MS_CONVERT") or ""
    for part in env.split(os.pathsep):
        if part.strip():
            out.append(str(Path(part.strip()).expanduser()))
    if hint:
        d = Path(hint).expanduser().resolve()
        base = d if d.is_dir() else d.parent
        out += [str(base), str(base / "md_convert"), str(base / "pymsdump"),
                str(base / "trj2xtc" / "pymsdump"),
                str(base.parent / "md_convert"), str(base.parent / "pymsdump")]
    here = Path(__file__).resolve().parent.parent
    out += [str(here / "md_convert"), str(here / "pymsdump"),
            str(here / "trj2xtc" / "pymsdump")]
    # 桌面下的有限深度搜索（用户常把 pymsdump 放在毕设目录里）
    try:
        desktop = Path.home() / "Desktop"
        if desktop.is_dir():
            out += _find_dirs(desktop, (XTC_MODULE, TRJ_MODULE), max_depth=4)
    except Exception:                                            # noqa: BLE001
        pass
    seen, uniq = set(), []
    for p in out:
        if p not in seen and os.path.isdir(p):
            seen.add(p)
            uniq.append(p)
    return uniq


def _find_dirs(root: Path, names: Iterable[str], max_depth: int = 4) -> list[str]:
    """在 ``root`` 下做有限深度 BFS，返回包含任一目标文件的目录。"""
    key = f"{root}|{','.join(names)}|{max_depth}"
    if key in _dir_cache:
        return _dir_cache[key]
    want = set(names)
    hits: list[str] = []
    level = [(root, 0)]
    while level and len(hits) < 8:
        nxt = []
        for d, depth in level:
            try:
                for child in d.iterdir():
                    if child.is_dir() and depth + 1 <= max_depth:
                        if child.name.lower() in ("node_modules", "__pycache__",
                                                  ".git", "eigen-3.3.9"):
                            continue
                        nxt.append((child, depth + 1))
                    elif child.name in want:
                        hits.append(str(d))
                        break
            except (PermissionError, OSError):
                continue
        level = nxt
    _dir_cache[key] = hits
    return hits


def module_dirs(hint: str | None = None) -> dict[str, str | None]:
    """返回 ``{"trj": dir|None, "xtc": dir|None}``。"""
    key = str(hint)
    if key in _module_cache:
        return _module_cache[key]
    found: dict[str, str | None] = {"trj": None, "xtc": None}
    for d in _candidate_dirs(hint):
        for mod, fname in (("trj", TRJ_MODULE), ("xtc", XTC_MODULE)):
            if found[mod] is None and os.path.isfile(os.path.join(d, fname)):
                found[mod] = d
        if found["trj"] and found["xtc"]:
            break
    _module_cache[key] = found
    return found


def available(hint: str | None = None) -> bool:
    """``.trj`` 转换是否可用（两个模块都找到）。"""
    m = module_dirs(hint)
    return bool(m["trj"] and m["xtc"])


def unavailable_message(hint: str | None = None) -> str:
    """说清楚**到底缺哪个**模块、以及三种放法（用户换机器时最常卡在这）。"""
    m = module_dirs(hint)
    lines = ["读取 Materials Studio 的 .trj 需要两个**纯 Python** 模块（都不需要编译）："]
    if m["trj"]:
        lines.append(f"  ✓ ms_trj.py      （.trj 解析器）  已找到：{m['trj']}")
    else:
        lines.append("  ✗ ms_trj.py      （.trj 解析器）  **缺失**"
                     " —— 它应在本仓库的 pymsdump/ 里（1.0.4 起已集成），"
                     "检查是否被删或被 .gitignore 排除")
    if m["xtc"]:
        lines.append(f"  ✓ xtcwriter.py   （XTC 写入器）  已找到：{m['xtc']}")
    else:
        lines.append("  ✗ xtcwriter.py   （XTC 写入器）  **缺失**"
                     " —— 本仓库自带 pymsdump/xtcwriter.py，检查是否被删了")
    if not m["trj"]:
        lines += [
            "",
            "从仓库找回这个文件（`git checkout -- pymsdump/ms_trj.py`），或放到下面任一位置：",
            "  ① 与 .trj 文件同一个文件夹里（或它的上一级）；",
            "  ② <仓库>/pymsdump/ms_trj.py（默认位置）；",
            "  ③ 任意位置，然后用环境变量点名：",
            r"     set MDTA_MS_CONVERT=D:\你的路径\md_convert;C:\...\MDT\pymsdump",
            "",
            "  临时验证：python -c 'from mdta import ms_trj; "
            "print(ms_trj.module_dirs())'",
            "  另一条省事的路：在别的电脑上先把 .trj 转成 .xtc（产物在 _convert_cache/ 里），"
            "再把那个 .xtc 拷过来直接分析。",
        ]
    return "\n".join(lines)


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "_convert_cache"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        import tempfile

        d = Path(tempfile.gettempdir()) / "mdta_convert"
        d.mkdir(parents=True, exist_ok=True)
    return d


def cached_xtc(trj: str) -> Path:
    """按「绝对路径 + 大小 + 修改时间」算缓存文件名。"""
    p = Path(trj).expanduser().resolve()
    st = p.stat()
    key = hashlib.sha1(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
    return _cache_dir() / f"{p.stem}.{key.hexdigest()[:12]}.xtc"


def convert_trj(trj: str, *, precision: float = 1000.0,
                progress: Callable[[str], None] | None = None,
                force: bool = False) -> Path:
    """把 ``.trj`` 转成 XTC 并返回路径（命中缓存就直接返回）。

    ``precision`` 默认 1000（= 0.01 Å 量化步长，与 MS_Trajdump 一致）。
    """
    p = Path(trj).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"找不到轨迹文件: {p}")
    out = cached_xtc(str(p))
    if out.is_file() and not force and out.stat().st_size > 0:
        if progress:
            progress(f"使用已转换的 XTC 缓存（{out.name}）")
        return out

    m = module_dirs(str(p))
    if not (m["trj"] and m["xtc"]):
        raise RuntimeError(unavailable_message(str(p)))
    for d in (m["xtc"], m["trj"]):
        if d and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from ms_trj import MSTrajectory                          # type: ignore
        from xtcwriter import XTCWriter                          # type: ignore
    except Exception as exc:                                     # noqa: BLE001
        raise RuntimeError(
            f"{unavailable_message(str(p))}\n导入失败：{type(exc).__name__}: {exc}"
        ) from exc

    if progress:
        progress(f"转换 Materials Studio .trj（{p.name}）")
    src = MSTrajectory(str(p))
    tmp = out.with_suffix(".part")
    n = 0
    try:
        w = XTCWriter(str(tmp), precision=precision)
        for fr in src.iter_frames(step=1):
            w.write(fr.xyz, fr.cell[:3], step=fr.step, time=fr.time)
            n += 1
            if progress and n % 500 == 0:
                progress(f"转换 .trj → XTC：{n}/{getattr(src, 'n_frames', '?')} 帧")
        w.close()
    except Exception as exc:                                     # noqa: BLE001
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"转换 .trj 失败（{type(exc).__name__}: {exc}）") from exc
    finally:
        try:
            src.close()
        except Exception:                                        # noqa: BLE001
            pass
    os.replace(tmp, out)
    if progress:
        progress(f"转换完成：{n} 帧 → {out.name}")
    return out


def trj_info(trj: str) -> dict:
    """不转换，先读 .trj 的帧数/原子数（用于界面提示与进度估算）。"""
    m = module_dirs(str(trj))
    if not m["trj"]:
        return {}
    if m["trj"] not in sys.path:
        sys.path.insert(0, m["trj"])
    try:
        from ms_trj import MSTrajectory                          # type: ignore

        t = MSTrajectory(str(trj))
        info = {"n_atoms": int(t.n_atoms), "n_frames": int(t.n_frames)}
        t.close()
        return info
    except Exception:                                            # noqa: BLE001
        return {}


def trj_header(trj: str) -> dict:
    """**只读头部**拿 ``{"n_atoms", "n_frames_declared"}``（不依赖 pymsdump、不读整文件）。

    为什么要单独写一个：`trj_info` 走用户的 `MSTrajectory`，遇到**截断**的 `.trj`
    会直接抛异常返回 ``{}``；而"这个轨迹跟拓扑是不是同一个体系"只看头部就够了，
    并且必须在**转换之前**判断 —— 否则一个 5,400 原子的 `.car` 配一个 5,596 原子的
    1.6 GB `.trj`，会先老老实实转几个 GB，最后才报一句让人误解的"转换失败"。

    格式（与 `md_convert/ms_trj.py` 同）：大端 Fortran 记录序列；第 5 条记录长 20，
    内容是 ``int 1`` + ``int n_atoms`` + ``int n_atoms`` + 8 字节 ``MStudio ``。
    """
    out: dict = {"n_atoms": None, "n_frames_declared": None}
    try:
        with open(trj, "rb") as fh:
            head = fh.read(4096)
        size = os.path.getsize(trj)
    except OSError:
        return out
    pos = 0
    for _ in range(8):
        if pos + 4 > len(head):
            break
        (n,) = struct.unpack(">i", head[pos:pos + 4])
        if n <= 0 or n > 1_000_000:
            break
        body = head[pos + 4:pos + 4 + n]
        if len(body) >= 20 and body[12:20] == b"MStudio ":
            cand = int(struct.unpack(">i", body[4:8])[0])
            if cand > 0:
                out["n_atoms"] = cand
                break
        pos += 8 + n
    if out["n_atoms"] is None:                  # 兜底：直接找 "MStudio " 标记
        i = head.find(b"MStudio ")
        if i >= 8:
            cand = int(struct.unpack(">i", head[i - 8:i - 4])[0])
            if cand > 0:
                out["n_atoms"] = cand
    # 帧数（按格式推算；截断的文件只给出"完整的那部分"帧数，作参考用）
    #   头部 = 92+92+92+40+28 +(8+4(n+1)) +15+12 = 379 + 4(n+1)
    #   每帧 = (492+96+32+176+116) + 5×8 + 6×(8+8n) = 1000 + 48n
    #   （实测 5596 原子：每帧 269,608 B ✓ 与解析器报的完全一致）
    n_at = out["n_atoms"]
    if n_at:
        header = 379 + 4 * (n_at + 1)
        frame = 1000 + 48 * n_at
        body_bytes = size - header
        if frame > 0 and body_bytes > 0:
            out["n_frames_declared"] = int(body_bytes // frame)
    return out


__all__ = ["available", "cached_xtc", "convert_trj", "module_dirs", "trj_header",
           "trj_info", "unavailable_message"]


def _selfcheck() -> int:
    """``python -m mdta.ms_trj`` —— 打印两个模块找到没、从哪找到的。"""
    m = module_dirs()
    ok = bool(m["trj"] and m["xtc"])
    print(f"{'可用' if ok else '不可用'}：Materials Studio .trj → XTC 转换")
    print(f"  ms_trj.py    : {m['trj'] or '（缺失）'}")
    print(f"  xtcwriter.py : {m['xtc'] or '（缺失）'}")
    if not ok:
        print()
        print(unavailable_message())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
