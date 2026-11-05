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

转好的缓存会按「源文件绝对路径 + 大小 + 修改时间 + 输出格式 + 精度」做键存到
``_convert_cache/``，第二次打开同一份 ``.trj`` 就是秒开。

输出格式与精度
--------------
* **XTC**（默认）：只存坐标，文件最小 —— 量化步长 1e-4 nm = 0.001 Å
  （单点误差 ≤5e-4 Å）。官方 ``Trj2Ascii`` 导出的是完整 double，所以这点差别
  就是"我们 vs 官方"的全部偏差；想更准可以设 ``MDTA_XTC_PRECISION=100000``。
* **TRR**（``fmt="trr"``，界面上「保留速度数据」）：坐标 **+ 速度**，
  坐标 float32（误差 ~1e-6 Å），速度单位 Å/ps。``.trj`` 里第 4-6 条 double
  数组就是速度，XTC 格式没有这个字段，只有 TRR（MDAnalysis 原生可读、
  ``u.atoms.velocities`` 直接可用）能带出来。代价是文件更大（每帧每原子 24 B）。

速度（同一份 5596 原子 × 300 帧的 .trj 实测）
---------------------------------------------
=====================  ==========  ==========
写入方式               耗时        体积
=====================  ==========  ==========
纯 Python XTC 写入器     18.9 s      9.9 MB
MDAnalysis C 写入器      0.4 s      9.9 MB   ← 现在默认走这条（快 49×）
TRR（坐标 + 速度）       0.6 s     40.3 MB
=====================  ==========  ==========

所以"转得慢"的根源从来不是 .trj 解析（读 300 帧只要 0.06 s），而是**纯 Python
的 XTC 位压缩**（每帧几千次 Python 级的位操作）。MDAnalysis 自带的写入器是
C 扩展（libmdaxdr），产物大小与精度完全一致，只在它不可用时才退回纯 Python
（``writer_kind()`` 可以看到这次用的是哪个）。

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

import numpy as np

#: 模块名 → 需要的目录（任一目录里能找到对应文件即可）
TRJ_MODULE = "ms_trj.py"
XTC_MODULE = "xtcwriter.py"

#: XTC 坐标量化精度：坐标（nm）× precision 后取整。10000 = 步长 1e-4 nm = 0.001 Å，
#: 单点最大误差 5e-4 Å（旧的 1000 是 0.01 Å 步长 / 5e-3 误差，100 更粗到 0.05 Å）。
#: 官方 Trj2Ascii 导出的是**完整 double**，所以这个步长就是"我们 vs 官方"的全部偏差。
XTC_PRECISION = 10000.0
#: 想改精度可以设环境变量 ``MDTA_XTC_PRECISION``（越大越准、文件越大）。
PRECISION_ENV = "MDTA_XTC_PRECISION"

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


def _mda_xtc_writer():
    """MDAnalysis 的 XTC 写入器（C 扩展）。有它就不需要 pymsdump/xtcwriter.py。"""
    try:
        from MDAnalysis.coordinates.XTC import XTCWriter as _W  # type: ignore

        return _W
    except Exception:                                            # noqa: BLE001
        return None


def available(hint: str | None = None) -> bool:
    """``.trj`` 转换是否可用。

    ``.trj`` 解析器（``ms_trj.py``）是**必需**的；写 XTC 的那一步优先用
    MDAnalysis 自带的 C 写入器，只有在它不可用时才需要 ``xtcwriter.py``。
    """
    m = module_dirs(hint)
    if not m["trj"]:
        return False
    return bool(m["xtc"]) or _mda_xtc_writer() is not None


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


def xc_precision() -> float:
    """实际使用的 XTC 量化精度（环境变量可覆盖）。"""
    raw = os.environ.get(PRECISION_ENV, "")
    if raw.strip():
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return XTC_PRECISION


def cached_xtc(trj: str, *, fmt: str = "xtc", precision: float | None = None,
               velocities: bool = False) -> Path:
    """按「绝对路径 + 大小 + 修改时间 + 输出格式 + 精度」算缓存文件名。

    ⚠️ 精度与格式**必须进缓存键**：以前只按路径/大小/时间，改了精度或想换 TRR
    时会直接命中旧文件，看上去"改了没生效"（实测用户遇到 0.02/0.05 Å 的旧精度
    缓存，而代码里早就是 1e-3 Å 了）。
    """
    p = Path(trj).expanduser().resolve()
    st = p.stat()
    prec = xc_precision() if precision is None else float(precision)
    fmt = "trr" if str(fmt).lower() == "trr" else "xtc"
    tag = f"{fmt}|{prec:g}|{int(bool(velocities))}"
    key = hashlib.sha1(
        f"{p}|{st.st_size}|{int(st.st_mtime)}|{tag}".encode("utf-8")
    ).hexdigest()[:12]
    return _cache_dir() / f"{p.stem}.{key}.{fmt}"


#: 最近一次 XTC 用的是哪个写入器（``"mdanalysis"`` 或 ``"python"``），供自检/日志查看
_xtc_writer_kind = ""


def writer_kind() -> str:
    """XTC 写入器：``"mdanalysis"``（C 扩展，默认）还是 ``"python"``（兜底）。"""
    return _xtc_writer_kind


def _cell6(cell) -> np.ndarray:
    """MS 的 ``cell=(a,b,c,α,β,γ)`` → MDAnalysis 的 6 值尺寸（缺角就按 90° 补）。"""
    c = np.asarray(cell, dtype=np.float32)
    if c.size >= 6:
        return c[:6]
    if c.size == 3:
        return np.concatenate([c[:3], np.array([90.0, 90.0, 90.0], dtype=np.float32)])
    return np.array([0, 0, 0, 90, 90, 90], dtype=np.float32)


def xtc_precision_digits(precision: float) -> int:
    """把"nm⁻¹ 精度"（如 1e4）换成 MDAnalysis XTCWriter 的"小数位数"（如 4）。

    MDAnalysis 的 ``precision=N`` 表示坐标按 **10^-N nm** 量化，也就是步长
    10^(1-N) Å：N=4 → 1e-3 Å（与我们的 1e4 等价）。
    """
    import math

    p = float(precision)
    if p <= 0:
        return 3
    return max(0, min(9, int(round(math.log10(p)))))


def _write_xtc(src, out: Path, precision: float,
               progress: Callable[[str], None] | None) -> int:
    """``.trj`` → XTC（只存坐标；文件小、分析最快）。

    **优先用 MDAnalysis 自带的 XTC 写入器**（C 扩展 libxrd / libmdaxdr）：
    同一批数据实测 0.11 s vs 纯 Python 的 8.8 s（**82×**），产物大小与精度
    完全一致（4.0 MB、最大误差 5.0e-4 Å）。纯 Python 的 ``pymsdump/xtcwriter.py``
    只在 MDAnalysis 缺失/导入失败时兜底 —— 它是"零编译依赖"的保险。
    """
    global _xtc_writer_kind
    _MdaXTC = _mda_xtc_writer()

    n = 0
    if _MdaXTC is not None:
        import MDAnalysis as mda

        _xtc_writer_kind = "mdanalysis"
        n_at = int(src.n_atoms)
        u = mda.Universe.empty(n_at, trajectory=True)
        ts = u.trajectory.ts
        w = _MdaXTC(str(out), n_at, precision=xtc_precision_digits(precision))
        try:
            for fr in src.iter_frames(step=1):
                ts.positions = fr.xyz
                ts.dimensions = _cell6(fr.cell)
                ts.time = float(fr.time)
                try:
                    ts.data["step"] = int(fr.step)
                except Exception:                            # noqa: BLE001
                    pass
                w.write(u)
                n += 1
                if progress and n % 500 == 0:
                    progress(f"转换 .trj → XTC：{n}/"
                             f"{getattr(src, 'n_frames', '?')} 帧")
        finally:
            w.close()
        return n

    # ---- 兜底：纯 Python 写入器（慢 80 倍，但没有编译依赖） ----
    _xtc_writer_kind = "python"
    m = module_dirs(str(src.path))
    if not m["xtc"]:
        raise RuntimeError(unavailable_message(str(src.path)))
    if m["xtc"] not in sys.path:
        sys.path.insert(0, m["xtc"])
    from xtcwriter import XTCWriter                          # type: ignore

    w = XTCWriter(str(out), precision=precision)
    try:
        for fr in src.iter_frames(step=1):
            w.write(fr.xyz, fr.cell[:3], step=fr.step, time=fr.time)
            n += 1
            if progress and n % 500 == 0:
                progress(f"转换 .trj → XTC（纯 Python 兜底）：{n}/"
                         f"{getattr(src, 'n_frames', '?')} 帧")
    finally:
        w.close()
    return n


def _write_trr(src, out: Path, progress: Callable[[str], None] | None,
               with_velocities: bool = True) -> int:
    """``.trj`` → TRR（坐标，可选 **+ 速度**；float32，坐标误差 ~1e-6 Å）。

    为什么用 TRR：XTC 装不下速度（格式里没有这个字段），而 MS 的 ``.trj`` 里
    第 4-6 条数组就是速度（Å/ps，与官方 Trj2Ascii 导出的 ``Velocities:`` 一致）。
    TRR 是 GROMACS 的标准格式，MDAnalysis 原生可读，``u.atoms.velocities`` 直接可用。

    体积/速度实测（5596 原子 × 120 帧基准）：坐标+速度 16.1 MB、顺序读 0.19 s；
    只写坐标时体积减半、读取更快。换算到 6001 帧：约 800 MB（含速度）/ 400 MB（仅坐标），
    而同等精度的 XTC 约 130 MB。**转换本身反而比 XTC 快 30 倍**（TRR 是裸写入，
    XTC 要走纯 Python 的位压缩）。
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.TRR import TRRWriter        # type: ignore

    n_at = int(src.n_atoms)
    u = mda.Universe.empty(n_at, trajectory=True)
    ts = u.trajectory.ts
    if with_velocities:
        # 先分配，TRRWriter 才会把速度块写出去
        ts.velocities = np.zeros((n_at, 3), dtype=np.float32)
    w = TRRWriter(str(out), n_at, convert_units=True)
    n = 0
    try:
        for fr in src.iter_frames(step=1, vel=with_velocities):
            ts.positions = fr.xyz
            if with_velocities and fr.vel is not None:
                ts.velocities = fr.vel
            # 盒矢量：fr.cell = (a, b, c, alpha, beta, gamma)；MDAnalysis 要 6 个值
            ts.dimensions = _cell6(fr.cell)
            ts.time = float(fr.time)
            try:
                ts.data["step"] = int(fr.step)
            except Exception:                                # noqa: BLE001
                pass
            w.write(u)
            n += 1
            if progress and n % 500 == 0:
                progress(f"转换 .trj → TRR（{'含' if with_velocities else '不含'}速度）："
                         f"{n}/{getattr(src, 'n_frames', '?')} 帧")
    finally:
        w.close()
    return n


def convert_trj(trj: str, *, fmt: str = "xtc",
                precision: float | None = None,
                velocities: bool | None = None,
                progress: Callable[[str], None] | None = None,
                force: bool = False) -> Path:
    """把 ``.trj`` 转成缓存轨迹并返回路径（命中缓存就直接返回）。

    ``fmt``
        ``"xtc"``（默认，只存坐标，文件最小）或 ``"trr"``（坐标 **+ 速度**）。
    ``precision``
        XTC 的量化精度（nm⁻¹）。默认 :data:`XTC_PRECISION` = 1e4（0.001 Å 步长，
        单点最大误差 5e-4 Å）。**精度与格式都参与缓存键**，改了就自动重建。
    ``velocities``
        ``fmt="trr"`` 时默认带上速度；显式给 ``False`` 可只写坐标（体积减半）。
    """
    p = Path(trj).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"找不到轨迹文件: {p}")
    fmt = "trr" if str(fmt).lower() == "trr" else "xtc"
    if velocities is None:
        velocities = (fmt == "trr")
    prec = xc_precision() if precision is None else float(precision)
    out = cached_xtc(str(p), fmt=fmt, precision=prec, velocities=velocities)
    if out.is_file() and not force and out.stat().st_size > 0:
        if progress:
            progress(f"使用已转换的缓存（{out.name}）")
        return out

    m = module_dirs(str(p))
    if not m["trj"]:
        raise RuntimeError(unavailable_message(str(p)))
    # XTC 优先用 MDAnalysis 的 C 写入器（快 80 倍）；没有它才需要 pymsdump 的
    # 纯 Python 写入器，所以这里只在"两者都没有"时才报错。
    if fmt == "xtc" and not m["xtc"] and _mda_xtc_writer() is None:
        raise RuntimeError(unavailable_message(str(p)))
    for d in (m["trj"], m["xtc"]):
        if d and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from ms_trj import MSTrajectory                          # type: ignore
    except Exception as exc:                                     # noqa: BLE001
        raise RuntimeError(
            f"{unavailable_message(str(p))}\n导入失败：{type(exc).__name__}: {exc}"
        ) from exc

    if progress:
        what = "TRR（坐标 + 速度）" if fmt == "trr" else f"XTC（精度 {prec:g}）"
        progress(f"转换 Materials Studio .trj → {what}（{p.name}）")
    src = MSTrajectory(str(p))
    tmp = out.with_name(out.name + ".part")
    try:
        n = (_write_trr(src, tmp, progress, with_velocities=bool(velocities))
             if fmt == "trr" else _write_xtc(src, tmp, prec, progress))
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
        progress(f"转换完成：{n} 帧 → {out.name}（{out.stat().st_size / 1e6:.1f} MB）")
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
           "trj_info", "unavailable_message", "writer_kind", "xc_precision",
           "XTC_PRECISION"]


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
