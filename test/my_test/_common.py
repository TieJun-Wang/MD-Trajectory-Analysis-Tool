# -*- coding: utf-8 -*-
r"""公共部分：用**本工具自己的引擎**跑一项分析，并把结果打印 + 导出成 CSV。

和 `test/mda_test/` 的分工
--------------------------
`test/mda_test/` 是**交叉验证**：拿第三方实现（MDAnalysis 的 `InterRDF` / freud 的
`Steinhardt` / 作者自己的 `allfast-3-2.py` 口径…）重算一遍，再和引擎逐位比对 ——
回答"算得对不对"。

`test/my_test/`（本目录）是**功能脚本**：直接调用工具里的分析函数
（``mdta.pipeline.Analyzer`` → ``mdta.analysis.*``），一个模块演示一项功能 ——
回答"这个功能怎么用、输出长什么样、导出的文件叫什么名"。

因此这里**不重写任何算法**：组件识别、逐帧解包裹、CSV 列名、面板切片名
（``rg__time.csv`` / ``dihedral__trans.csv``）全部走引擎本体，脚本只负责
"喂参数 + 打印 + 导出"，与图形界面「一键分析 → 导出」得到的东西完全一致。

三种用法
--------
```powershell
# ① 单独跑一项（不带参数 = 交互菜单；带参数 = 命令行）
python test/my_test/contact.py
python test/my_test/contact.py --frames 50 --cutoff 4.5 -y

# ② 一次跑完 14 项（打印一张汇总表）
python test/my_test/run_all.py

# ③ 在自己的脚本里 import
import sys; sys.path.insert(0, r"C:\temp\MDT\test")
from my_test import Engine, test_contact

eng = Engine(max_frames=20)         # 默认体系 20PE30
test_contact(eng, cutoff=4.5)
```

环境变量（都可省）
------------------
| 变量 | 作用 | 默认 |
|---|---|---|
| `MDTA_MYTEST_SYSTEM` | 预置体系：`pe`（20PE30，单组分）/ `mix`（5PE20…，多组分） | `pe` |
| `MDTA_MYTEST_TOP` | 拓扑文件（覆盖预置） | 预置体系 |
| `MDTA_MYTEST_TRJ` | 轨迹文件（覆盖预置；`.trj` 会自动转 TRR） | 预置体系 |
| `MDTA_MYTEST_SELECT` | 分析对象选择串 | 自动识别 |
| `MDTA_MYTEST_FRAMES` | 最多用多少帧 | 100 |
| `MDTA_MYTEST_OUT` | CSV 输出目录 | `test/my_test/_out` |
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ 路径
_ROOT = Path(__file__).resolve().parents[2]           # 仓库根 C:\temp\MDT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HERE = Path(__file__).resolve().parent
_OUT_DEFAULT = _HERE / "_out"                        # 默认输出目录（已 gitignore）

# ------------------------------------------------------------------ 预置体系
#: 预置体系：换体系用 ``--top`` / ``--trj``，或用 ``MDTA_MYTEST_SYSTEM=mix``。
SYSTEMS: "OrderedDict[str, dict]" = OrderedDict([
    ("pe", {
        "label": "20PE30：20 条 PE30 链 / 3640 原子 / 301 帧 / 立方盒 33.6 Å（**单组分**）",
        "top": r"C:\temp\dataset\20PE30\20PE30.xsd",
        "trj": r"C:\temp\dataset\20PE30\20PE30.trj",
    }),
    ("mix", {
        "label": ("5PE20-2000H20-24Na-24Cl：5 条 PEO20 + 2000 水 + 48 离子 / "
                  "6758 原子 / 1001 帧（**多组分**，界面/质心距/面内 RDF 用这个）"),
        "top": r"C:\temp\dataset\5PE20-2000H20-24Na-24Cl\5PE20-2000H20-24Na-24Cl.xsd",
        "trj": r"C:\temp\dataset\5PE20-2000H20-24Na-24Cl\5PE20-2000H20-24Na-24Cl.trj",
    }),
])

#: 默认用哪个预置体系（``MDTA_MYTEST_SYSTEM=pe|mix``）
DEFAULT_SYSTEM = os.environ.get("MDTA_MYTEST_SYSTEM", "pe")
if DEFAULT_SYSTEM not in SYSTEMS:
    DEFAULT_SYSTEM = "pe"

DEFAULT_TOP = os.environ.get("MDTA_MYTEST_TOP") or SYSTEMS[DEFAULT_SYSTEM]["top"]
DEFAULT_TRJ = os.environ.get("MDTA_MYTEST_TRJ")
if DEFAULT_TRJ is None:
    DEFAULT_TRJ = SYSTEMS[DEFAULT_SYSTEM]["trj"]
DEFAULT_SELECT = os.environ.get("MDTA_MYTEST_SELECT", "all")

#: 菜单里默认跑多少帧（0 = 全部）。大体系先用 100 帧试，省时间。
DEFAULT_FRAMES = int(os.environ.get("MDTA_MYTEST_FRAMES", "100") or 0)


# ------------------------------------------------------------------ 输出目录
def out_dir() -> Path:
    """CSV 的默认输出目录（``MDTA_MYTEST_OUT`` 或 ``test/my_test/_out``）。"""
    d = Path(os.environ.get("MDTA_MYTEST_OUT") or _OUT_DEFAULT)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ 数值格式
def fmt(v, nd: int = 4) -> str:
    """把统计量格式化成好读的字符串（大数加千分位、None/NaN 说人话）。"""
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,d}"
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if not np.isfinite(x):
            return "—" if np.isnan(x) else ("∞" if x > 0 else "-∞")
        if abs(x) >= 1e5 or (0 < abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:,.{nd}g}"
    return str(v)


# ------------------------------------------------------------------ 引擎
class Engine:
    """一个分析会话：加载体系 → 识别组分 → 选帧 → 逐项跑分析 → 导出。

    内部就是工具的 :class:`mdta.pipeline.Analyzer`（Web 后端用的同一个类），
    所以这里的任何一步都与图形界面**同一条代码路径**：

    · 加载：``mdta.io.load_trajectory``（Materials Studio ``.xsd/.trj`` 自动转 TRR）；
    · 组分：``Analyzer.auto_setup()`` → ``mdta.selection.component_groups``；
    · 选帧：``mdta.preprocess.select_frames``；
    · 分析：``Analyzer.run(name, params=...)``；
    · 导出：``mdta.export.export_csv``（文件名与界面「下载 CSV」一致）。
    """

    def __init__(self, topology: str | None = None, trajectory: str | None = None, *,
                 select: str | None = None, max_frames: int | None = None,
                 start_ps: float | None = None, stop_ps: float | None = None,
                 interval_ps: float | None = None, equil_ps: float | None = None,
                 guess_bonds: bool | None = None, velocities: bool | None = None,
                 system: str | None = None, verbose: bool = True):
        from mdta.pipeline import Analyzer

        self.system = system or DEFAULT_SYSTEM
        preset = SYSTEMS.get(self.system, SYSTEMS[DEFAULT_SYSTEM])
        self.topology = str(topology or preset["top"])
        self.trajectory = str(trajectory if trajectory is not None else preset["trj"])
        self.trajectory = self.trajectory or None          # 空串 = 只读单帧结构
        for p in (self.topology, self.trajectory):
            if p and not os.path.exists(p):
                raise SystemExit(
                    f"找不到文件：{p}\n"
                    f"可用 --top/--trj 指定，或用环境变量 MDTA_MYTEST_TOP/TRJ；\n"
                    f"预置体系：{ {k: v['label'] for k, v in SYSTEMS.items()} }")

        # 速度：MS 的 .trj 默认只转坐标（XTC，快）；VACF 要速度，得转 TRR。
        # 这里对 `.trj` **默认保留速度**，让 `python test/my_test/vacf.py` 开箱可用；
        # 想回到"只转坐标"就设 MDTA_MYTEST_VELOCITIES=0（转出来的文件小一半、快一倍）。
        if velocities is None:
            env = os.environ.get("MDTA_MYTEST_VELOCITIES")
            if env is not None:
                velocities = env.strip().lower() not in ("0", "false", "no", "off")
            elif self.trajectory and self.trajectory.lower().endswith(".trj"):
                velocities = True
        self.velocities = velocities

        t0 = time.time()
        self.az = Analyzer(self.topology, self.trajectory, guess_bonds=guess_bonds,
                           velocities=velocities)
        self.az.auto_setup()                    # 主链 + 组分（与界面点「自动识别」一致）
        self.select: str | None = None
        if select and str(select).strip().lower() not in ("all", "*", "全部", "全部原子"):
            self.apply_select(str(select))
        self.frames = self.az.set_frames(
            start_ps=start_ps, stop_ps=stop_ps, interval_ps=interval_ps,
            equil_ps=equil_ps,
            max_frames=(DEFAULT_FRAMES or None) if max_frames is None else
            (int(max_frames) or None))
        self.load_seconds = time.time() - t0
        self.last_seconds = 0.0
        #: 导出文件名前缀（菜单里问的「输出文件名前缀」；``None`` = 用分析名）
        self.export_prefix: str | None = None
        if verbose:
            self.describe()

    # ---------------------------------------------------------- 属性
    @property
    def u(self):
        """MDAnalysis 的 ``Universe``（引擎在读轨迹时建的那个）。"""
        return self.az.universe

    @property
    def mdt(self):
        """``mdta.io.MDTrajectory``（坐标、时间、拓扑元信息都在它上面）。"""
        return self.az.trajectory

    @property
    def components(self):
        """当前勾选参与分析的组分：``{名字: AtomGroup}``（有序）。"""
        return self.az.components

    @property
    def catalog(self):
        """自动识别出的**全部**候选组分（含没勾选的）。"""
        return self.az.catalog

    @property
    def primary(self):
        """构象分析的主对象（``Analyzer.auto_setup`` 推荐的那一个）。"""
        return self.az.primary

    @property
    def sel(self):
        """当前的帧选择 :class:`mdta.preprocess.FrameSelection`。"""
        return self.az.frames

    # ---------------------------------------------------------- 打印体系
    def describe(self) -> None:
        """打印体系概况 + 组分识别结果（等同于界面顶部的「体系信息」）。"""
        u = self.u
        box = getattr(u, "dimensions", None)
        print(f"体系：{self.topology.rsplit(os.sep, 1)[-1]}"
              f" + {(self.trajectory or '（单帧结构）').rsplit(os.sep, 1)[-1]}")
        print(f"  {u.atoms.n_atoms:,} 原子 / {len(u.trajectory)} 帧"
              f" / {len(getattr(u, 'residues', [])):,} 残基"
              + (f" / 盒 {box[0]:.4g}×{box[1]:.4g}×{box[2]:.4g} Å" if box is not None else
                 " / **无周期盒**（依赖 PBC 的项会跳过）"))
        print(f"  主链：{self.az.primary_label}")
        try:
            has_vel = bool(self.mdt.has_velocities)
        except Exception:                                    # noqa: BLE001
            has_vel = False
        print(f"  速度：{'有（VACF 可用）' if has_vel else '无（VACF 会跳过：XTC/DCD 不存速度）'}")
        print(f"  组分（勾选 {len(self.components)} / 候选 {len(self.catalog)}）：", end="")
        print("、".join(f"{k}({v.n_atoms:,} 原子)" for k, v in self.components.items()))
        if self.sel is not None:
            print(f"  帧选择：{self.sel.describe()}")
            for note in self.sel.notes:
                print(f"          {note}")

    # ---------------------------------------------------------- 选择 / 帧
    def apply_select(self, select: str) -> None:
        """把分析对象收窄到一个选择串（例如 ``resname ETH`` / ``element O``）。

        组件识别照常跑过一次（``catalog`` 里还在），只是"参与分析"的组分换成
        这个选择串 —— 与界面上「分析对象」下拉框选「自定义选择」等价。
        只有 1 个组分时，接触分析退回同组 A–A，界面宽度/质心距会**跳过并说明**
        原因（它们是"两组分之间"的量），这不是出错。
        """
        ag = self.u.select_atoms(select)
        if ag.n_atoms == 0:
            raise SystemExit(f"选择串 {select!r} 没有选中任何原子")
        name = f"选择[{select}]"
        self.az.set_components(OrderedDict({name: ag}))
        self.az.set_primary(ag, name)
        self.select = select

    def set_frames(self, *, start_ps=None, stop_ps=None, interval_ps=None,
                   equil_ps=None, max_frames=None):
        """按时间/间隔/平衡段重新选帧（对应界面「帧范围」那几个输入框）。"""
        self.frames = self.az.set_frames(start_ps=start_ps, stop_ps=stop_ps,
                                        interval_ps=interval_ps, equil_ps=equil_ps,
                                        max_frames=max_frames)
        return self.frames

    def use_frames(self, frames) -> "object":
        """用**帧号列表**（菜单选出来的那些）当帧选择。

        菜单支持"跨步抽 N 帧"，而 ``select_frames`` 是按时间选的，
        两者对不上时直接把 ``indices`` 写进 ``FrameSelection`` —— 引擎的每个
        分析函数只读这个对象，所以完全等价。
        """
        from mdta.preprocess import FrameSelection

        idx = np.asarray(sorted({int(f) for f in (frames or [])}), dtype=int)
        n = len(self.u.trajectory)
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size == 0:
            self.frames = self.az.require_frames()
            return self.frames
        times = np.asarray(self.az.times_ps, dtype=float)[idx]
        self.frames = FrameSelection(indices=idx, times_ps=times,
                                     notes=[f"手动指定帧号：{idx.size} 帧"
                                            f"（{idx[0]}…{idx[-1]}）"])
        self.az.frames = self.frames
        return self.frames

    # ---------------------------------------------------------- 跑一项
    def run(self, name: str, params: dict | None = None, *, verbose: bool = False):
        """跑一项分析，返回 :class:`mdta.core.AnalysisResult`（跳过时返回 ``None``）。

        ``params`` 就是 ``mdta.pipeline.DEFAULT_PARAMS[name]`` 那一套键
        （``cutoff`` / ``rmax`` / ``nbins`` / ``axis`` / ``mode`` …），
        另外接触/界面/质心距多一个 ``pair=(A, B)`` 用来指定要比哪两组。
        """
        from mdta.analysis.interface import BoxMissing

        if self.frames is not None:
            self.az.frames = self.frames
        t0 = time.time()
        try:
            res = self.az.run(name, params=dict(params or {}), verbose=verbose)
        except BoxMissing as exc:                       # 体系不满足前提，不是出错
            self.last_seconds = time.time() - t0
            print(f"⏭ 跳过：{str(exc).splitlines()[0]}")
            return None
        self.last_seconds = time.time() - t0
        if res is None:
            print(f"⏭ 跳过：{self.az.skip_reason or '体系缺少这项分析所需的组分'}"
                  f"（可用 --top/--trj 换个体系，或把组分补上）")
        return res

    # ---------------------------------------------------------- 打印 + 导出
    def report(self, res, *, outdir=None, formats=("csv",), head: str = "",
               max_summary: int | None = 10) -> dict:
        """把一项结果打印成人能看的统计表，并导出 CSV（``formats`` 里有 png 就再导图）。

        返回 ``{"name", "title", "panels", "summary", "files", "seconds", "res"}``；
        ``res`` 为 ``None``（该项被跳过）时返回 ``{"name": ..., "skipped": 原因}``。
        """
        if res is None:
            return {"name": head or "?", "skipped": True}

        if head:
            print(f"\n{head}")
        print(f"[{res.name}] {res.title}")
        print(f"  用时 {self.last_seconds:.2f} s ｜ 面板 {res.panel_count} 个 ｜ "
              f"曲线 {len(res.curves)} 条 ｜ 统计量 {len(res.summary)} 项")

        items = list(res.summary.items())
        shown = items if max_summary is None else items[:max_summary]
        if shown:
            width = max(len(str(k)) for k, _ in shown)
            for k, v in shown:
                print(f"    {str(k):<{width}}  {fmt(v)}")
            if len(items) > len(shown):
                print(f"    …（其余 {len(items) - len(shown)} 项统计见导出的 Excel 汇总表）")

        notes = [str(n) for n in (getattr(res, "notes", []) or [])]
        for note in notes[:3]:
            print(f"  · {note if len(note) <= 200 else note[:200] + '…'}")
        if len(notes) > 3:
            print(f"  · …（另有 {len(notes) - 3} 条说明，见界面「说明」区）")

        # 面板表头（哪些列 / 多少行）—— 与界面「数据表」一致
        for i in range(res.panel_count):
            try:
                panel = res.panel(i)
                df = res.dataframes()[list(res.dataframes())[i]]
            except Exception:                            # noqa: BLE001
                continue
            cols = "、".join(str(c) for c in df.columns)
            print(f"  ├ 面板 {i}「{panel.title}」{df.shape[0]} 行 × {df.shape[1]} 列：{cols}")

        files: list[str] = []
        if formats:
            from mdta.export import export_result

            target = str(outdir or out_dir())
            files = export_result(res, target, formats=tuple(formats),
                                  prefix=self.export_prefix)
            print("  导出：" + "、".join(os.path.basename(f) for f in files)
                  + f"  →  {target}")
        return {"name": res.name, "title": res.title, "panels": res.panel_count,
                "summary": dict(res.summary), "files": files,
                "seconds": self.last_seconds, "res": res}

    # ---------------------------------------------------------- 批量
    def run_many(self, names, *, params=None, outdir=None, formats=("csv",)) -> dict:
        """依次跑多项（每项独立，某项跳过不影响其余），返回 ``{名字: 结果或 None}``。"""
        out: dict = {}
        params = dict(params or {})
        for name in names:
            out[name] = self.run(name, params=params.get(name) or {})
            self.report(out[name], outdir=outdir, formats=formats, head=f"— {name} —")
        return out


# ------------------------------------------------------------------ 便捷入口
def load(**kwargs) -> Engine:
    """按默认路径建一个 :class:`Engine`（各模块 ``__main__`` 用它）。"""
    return Engine(**kwargs)


def summarize(rows) -> None:
    """打印一张小汇总表：``[(名字, 状态, 关键数, 用时)]``。"""
    if not rows:
        return
    w = max(len(str(r[0])) for r in rows)
    line = "-" * (w + 52)
    print("\n" + line)
    for name, status, note, sec in rows:
        print(f"  {str(name):<{w}}  {status}  {note:<28} {sec}")
    print(line)
