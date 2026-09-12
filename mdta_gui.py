# -*- coding: utf-8 -*-
"""MD 轨迹分析工具 —— 桌面操作台（Tkinter）。

.. note::
   本界面**不显示图表**。它的定位是"操作台 + 数据台"：
   选文件、选对象、设参数、跑分析、看数值、导出结果。
   图表由 :mod:`mdta.plotting` 在导出时生成 PNG（``analysis_results/*.png``），
   或由后续的 Web 前端读取同一批数据来渲染。

界面结构::

    ┌──────────────────────────────────────────────────────────┐
    │  MD Trajectory Analysis Tool                             │
    ├────────────────┬─────────────────────────────────────────┤
    │ 1. 轨迹文件     │ [分析结果] [汇总] [统计量明细] [运行日志]│
    │ 2. 体系信息     │  ┌────────────┬───────────────────────┐ │
    │ 3. 分析对象选择 │  │ 分析/数据表 │  当前数据表的行与列    │ │
    │ 4. 参数设置     │  │  ▸ 回转半径 │  （可导出为 CSV）      │ │
    │ 5. 分析功能     │  │  ▸ 二面角   │                       │ │
    │ 6. [开始分析]   │  └────────────┴───────────────────────┘ │
    ├────────────────┴─────────────────────────────────────────┤
    │ [导出全部] [CSV] [PNG] [Excel] [打开输出目录]  输出目录:… │
    └──────────────────────────────────────────────────────────┘

界面约定
--------
**布局固定，没有可拖拽的分隔条**：左侧操作面板为固定宽度
（:data:`MDTAGui.BASE_LEFT_PANEL`），右侧数据区自适应剩余空间。
两侧都用 ``pack`` + ``pack_propagate(False)`` 固定，不使用 ``ttk.Panedwindow``。

**以数据表代替图表**：

- 「分析结果」页左边是"分析 → 数据表"目录树（例如 Rg 有
  ``Rg 随时间变化``、``Rg 分布`` 两张表），点一下右边就显示该表的行列；
- 「汇总」页一眼看完所有分析；「统计量明细」页列出全部标量结果；
- 「运行日志」页记录读取与运行过程。

**图表怎么来**：曲线数据与导出功能面对的是同一份结果，命令行
``mdta_cli.py run`` 或本界面的「导出全部结果」都会生成
``analysis_results/<分析名>.png``；点「打开输出目录」即可查看。
同一批数据也可以直接交给 Web 前端（React / Vue + ECharts）渲染。

启动方式::

    python mdta_gui.py
    python mdta_cli.py gui
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from mdta import __version__                             # noqa: E402
from mdta.pipeline import (                              # noqa: E402
    ANALYSIS_TITLES,
    DEFAULT_ORDER,
    Analyzer,
)
from mdta.plotting import set_agg_backend                # noqa: E402
from mdta.selection import select                        # noqa: E402

# 本界面不显示 matplotlib 图，只在导出时把图存成文件，
# 因此统一用非交互的 Agg 后端，避免和 Tk 事件循环互相干扰。
set_agg_backend()

#: 分析功能分组（对应设计大纲第 9/10/15/20 章）
ANALYSIS_GROUPS = [
    ("链构象分析", ["rg", "ree", "dihedral"]),
    ("界面相容性分析", ["density", "rdf", "contact", "interface"]),
    ("结晶行为分析", ["orientation", "order"]),
    ("辅助分析", ["msd"]),
]


def enable_high_dpi() -> float:
    """开启 Windows 高 DPI 感知，返回显示缩放系数（1.0 = 100%）。

    为什么必须做这件事
    ------------------
    如果进程不声明 DPI 感知，Windows 会先按 96 dpi 渲染整个窗口，再把位图
    按缩放比拉伸（例如 150% 显示器放大 1.5 倍），于是**界面文字会明显发虚**。
    开启 per-monitor DPI 感知后，Tk 拿到的是真实分辨率，文字按真实像素渲染，
    界面就是清晰的。

    必须在创建 ``tk.Tk()`` **之前**调用。
    """
    if not sys.platform.startswith("win"):
        return 1.0
    import ctypes

    try:
        u32 = ctypes.windll.user32
        ok = False
        # ① Windows 10 1703+：per-monitor DPI aware v2（最佳）
        try:
            fn = u32.SetProcessDpiAwarenessContext
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_bool
            ok = bool(fn(ctypes.c_void_p(-4)))          # PER_MONITOR_AWARE_V2
        except Exception:  # noqa: BLE001
            ok = False
        # ② Windows 8.1+：per-monitor aware
        if not ok:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
        # ③ Windows Vista+：system aware
        if not ok:
            try:
                u32.SetProcessDPIAware()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    scale = 1.0
    try:
        scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:  # noqa: BLE001
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)   # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            scale = dpi / 96.0
        except Exception:  # noqa: BLE001
            scale = 1.0
    return float(scale) if scale and scale > 0 else 1.0


def _fmt_cell(v) -> str:
    """把表格里的一个数值格式化成便于阅读的字符串。"""
    try:
        if v is None:
            return ""
        if isinstance(v, float):
            if v != v:                     # NaN
                return "nan"
            if v == int(v) and abs(v) < 1e15:
                return str(int(v))
            return f"{v:.6g}"
        return str(v)
    except Exception:  # noqa: BLE001
        return str(v)


class MDTAGui:
    """主窗口。

    ``dpi_scale`` 是显示缩放系数（1.0 = 100%，1.5 = 150%）。
    所有像素尺寸都按它放大，配合已开启的 DPI 感知，界面在任意缩放下都清晰。
    """

    #: 以 100% 缩放为基准的设计尺寸（物理像素）
    BASE_WIDTH = 1440
    BASE_HEIGHT = 900
    BASE_LEFT_PANEL = 430        # 左侧操作面板：固定宽度，不可拖动
    BASE_TABLE_NAV = 260         # 「分析结果」页左侧目录树的固定宽度

    def __init__(self, root: tk.Tk, dpi_scale: float = 1.0):
        self.root = root
        self.scale = float(dpi_scale) if dpi_scale and dpi_scale > 0 else 1.0

        root.title(f"MD Trajectory Analysis Tool  v{__version__}")
        self._apply_geometry()

        self.analyzer: Analyzer | None = None
        self.results = {}
        self.result_order: list[str] = []
        #: ``{目录树 item id: (分析名, 表序号)}``
        self._table_map: dict[str, tuple[str, int]] = {}
        #: 当前在「分析结果」页里选中的分析名
        self._current_name: str | None = None
        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self._busy = False
        self._after_id: str | None = None

        self.var_top = tk.StringVar()
        self.var_traj = tk.StringVar()
        self.var_start = tk.StringVar(value="")
        self.var_stop = tk.StringVar(value="")
        self.var_equil = tk.StringVar(value="")
        self.var_interval = tk.StringVar(value="")
        self.var_maxframes = tk.StringVar(value="")
        self.var_cutoff = tk.StringVar(value="5.0")
        self.var_axis = tk.StringVar(value="c (2)")
        self.var_nbins = tk.StringVar(value="100")
        self.var_rmax = tk.StringVar(value="12.0")
        self.var_dihedral = tk.StringVar(value="auto (自动)")
        self.var_primary = tk.StringVar(value="自动（最大链）")
        self.var_custom_sel = tk.StringVar(value="")
        self.var_outdir = tk.StringVar(value=os.path.join(_HERE, "analysis_results"))
        self.var_status = tk.StringVar(value="就绪")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.analysis_vars: dict[str, tk.BooleanVar] = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._after_id = self.root.after(100, self._drain_queue)
        self.log(f"显示缩放 {self.scale * 100:.0f}%，已启用高 DPI 感知（画面按真实像素渲染）")

    # ============================================================ DPI / 字体
    def px(self, value: float) -> int:
        """把 100% 缩放下的设计像素值换算成当前缩放下的物理像素。"""
        return int(round(float(value) * self.scale))

    def _apply_geometry(self) -> None:
        """按屏幕实际分辨率与缩放比设置窗口大小，避免超出屏幕或过大过小。"""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
        except Exception:  # noqa: BLE001
            sw, sh = 1920, 1080
        w = min(self.px(self.BASE_WIDTH), int(sw * 0.96))
        h = min(self.px(self.BASE_HEIGHT), int(sh * 0.94))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(min(self.px(900), w), min(self.px(560), h))

    def _mono_font(self) -> tkfont.Font:
        """等宽字体，用于体系信息与日志文本框（字号由 Tk scaling 统一换算）。"""
        for fam in ("Cascadia Mono", "Consolas", "Courier New"):
            try:
                if fam in tkfont.families(self.root):
                    return tkfont.Font(root=self.root, family=fam, size=10)
            except Exception:  # noqa: BLE001
                continue
        return tkfont.Font(root=self.root, size=10)

    # ================================================================ 布局
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=6)
        outer.pack(fill="both", expand=True)

        # ---------------- 底部按钮栏
        bottom = ttk.Frame(outer)
        bottom.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(bottom, text="导出全部结果", command=self.export_all,
                   width=16).pack(side="left", padx=2)
        self.btn_csv = ttk.Button(bottom, text="导出当前分析 CSV",
                                  command=lambda: self.export_current("csv"), width=20)
        self.btn_csv.pack(side="left", padx=2)
        self.btn_png = ttk.Button(bottom, text="保存当前图片 PNG",
                                  command=lambda: self.export_current("png"), width=20)
        self.btn_png.pack(side="left", padx=2)
        self.btn_xlsx = ttk.Button(bottom, text="导出汇总 Excel",
                                   command=lambda: self.export_current("excel"), width=18)
        self.btn_xlsx.pack(side="left", padx=2)
        ttk.Button(bottom, text="打开输出目录", command=self.open_outdir,
                   width=14).pack(side="left", padx=2)
        ttk.Label(bottom, text="输出目录:").pack(side="left", padx=(16, 2))
        ttk.Entry(bottom, textvariable=self.var_outdir, width=30).pack(side="left")
        ttk.Button(bottom, text="选择…", width=7,
                   command=self.choose_outdir).pack(side="left", padx=2)

        # ---------------- 状态栏
        status = ttk.Frame(outer)
        status.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Label(status, textvariable=self.var_status, anchor="w").pack(
            side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(status, variable=self.var_progress,
                                        maximum=100.0, length=260)
        self.progress.pack(side="right")

        # ---------------- 主体：左操作面板（固定宽度）+ 右数据区（自适应）
        # 用 pack + pack_propagate(False) 固定左侧宽度，不使用可拖拽的
        # ttk.Panedwindow —— 布局不可拖动，两边宽度恒定。
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=self.px(self.BASE_LEFT_PANEL))
        left.pack(side="left", fill="y", expand=False)
        left.pack_propagate(False)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    # ---------------------------------------------------------- 左侧面板
    def _build_left(self, parent: ttk.Frame) -> None:
        # 用一个可滚动区域，窗口小的时候也能看到全部控件
        canvas = tk.Canvas(parent, width=self.px(self.BASE_LEFT_PANEL),
                           highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        def _on_conf(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_conf)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _wheel)

        # ---- 1. 文件
        f1 = ttk.LabelFrame(inner, text="1. 轨迹文件", padding=8)
        f1.pack(fill="x", pady=4)
        self._file_row(f1, "TPR / 拓扑文件:", self.var_top, self.choose_top)
        self._file_row(f1, "XTC / 轨迹文件:", self.var_traj, self.choose_traj)
        row = ttk.Frame(f1)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="读取体系信息", command=self.load_system,
                   width=16).pack(side="left")
        ttk.Label(row, text="（只给拓扑文件时会自动寻找同名 .xtc）",
                  foreground="#666").pack(side="left", padx=4)

        # ---- 2. 体系信息
        f2 = ttk.LabelFrame(inner, text="2. 体系信息", padding=8)
        f2.pack(fill="x", pady=4)
        self.txt_info = tk.Text(f2, height=13, width=46, wrap="none",
                                font=self._mono_font())
        isb = ttk.Scrollbar(f2, orient="vertical", command=self.txt_info.yview)
        self.txt_info.configure(yscrollcommand=isb.set)
        self.txt_info.pack(side="left", fill="both", expand=True)
        isb.pack(side="right", fill="y")

        # ---- 3. 选择
        f3 = ttk.LabelFrame(inner, text="3. 分析对象选择", padding=8)
        f3.pack(fill="x", pady=4)
        ttk.Label(f3, text="主链（构象 / 结晶分析对象）:").pack(anchor="w")
        self.cmb_primary = ttk.Combobox(f3, textvariable=self.var_primary,
                                        state="readonly", width=50)
        self.cmb_primary.pack(fill="x", pady=(0, 4))
        ttk.Label(f3, text="自定义选择语句（MDAnalysis 语法，可选）:").pack(anchor="w")
        ttk.Entry(f3, textvariable=self.var_custom_sel, width=52).pack(fill="x")
        ttk.Label(f3, text="参与界面 / RDF / MSD 分析的组分:", ).pack(anchor="w",
                                                                    pady=(6, 0))
        self.lst_comp = tk.Listbox(f3, selectmode="extended", height=5,
                                   exportselection=False)
        self.lst_comp.pack(fill="x")
        ttk.Label(f3, text="（多选；不选则自动使用识别到的全部组分）",
                  foreground="#666").pack(anchor="w")

        # ---- 4. 参数
        f4 = ttk.LabelFrame(inner, text="4. 参数设置", padding=8)
        f4.pack(fill="x", pady=4)
        grid = ttk.Frame(f4)
        grid.pack(fill="x")
        rows = [
            ("起始时间 (ps):", self.var_start, "留空 = 从第一帧开始"),
            ("结束时间 (ps):", self.var_stop, "留空 = 到最后一帧"),
            ("平衡段 (ps):", self.var_equil, "模拟前段不参与统计"),
            ("分析间隔 (ps):", self.var_interval, "留空 = 每帧都算"),
            ("最多帧数:", self.var_maxframes, "限制计算量"),
        ]
        for i, (lab, var, tip) in enumerate(rows):
            ttk.Label(grid, text=lab).grid(row=i, column=0, sticky="w", pady=1)
            ttk.Entry(grid, textvariable=var, width=10).grid(row=i, column=1,
                                                             sticky="w", padx=4)
            ttk.Label(grid, text=tip, foreground="#777").grid(row=i, column=2,
                                                              sticky="w")
        grid2 = ttk.Frame(f4)
        grid2.pack(fill="x", pady=(6, 0))
        ttk.Label(grid2, text="接触 cutoff (Å):").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid2, textvariable=self.var_cutoff, width=8).grid(row=0, column=1,
                                                                     sticky="w", padx=4)
        ttk.Label(grid2, text="密度方向:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Combobox(grid2, textvariable=self.var_axis, width=8, state="readonly",
                     values=["a (0)", "b (1)", "c (2)"]).grid(row=0, column=3,
                                                              sticky="w", padx=4)
        ttk.Label(grid2, text="密度 bin 数:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(grid2, textvariable=self.var_nbins, width=8).grid(row=1, column=1,
                                                                    sticky="w", padx=4)
        ttk.Label(grid2, text="RDF 最大 r (Å):").grid(row=1, column=2, sticky="w",
                                                      padx=(8, 0))
        ttk.Entry(grid2, textvariable=self.var_rmax, width=8).grid(row=1, column=3,
                                                                   sticky="w", padx=4)
        ttk.Label(grid2, text="二面角模式:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(grid2, textvariable=self.var_dihedral, width=22, state="readonly",
                     values=["auto (自动)", "phi_psi (蛋白质主链)",
                             "chain (链骨架)", "topology (拓扑定义)"]
                     ).grid(row=2, column=1, columnspan=3, sticky="w", padx=4)

        # ---- 5. 分析功能
        f5 = ttk.LabelFrame(inner, text="5. 分析功能", padding=8)
        f5.pack(fill="x", pady=4)
        for group, names in ANALYSIS_GROUPS:
            ttk.Label(f5, text=group, foreground="#00508a").pack(anchor="w",
                                                                 pady=(4, 0))
            line = ttk.Frame(f5)
            line.pack(fill="x")
            for n in names:
                var = tk.BooleanVar(value=True)
                self.analysis_vars[n] = var
                ttk.Checkbutton(line, text=ANALYSIS_TITLES.get(n, n),
                                variable=var).pack(side="left", padx=(0, 8))
        line = ttk.Frame(f5)
        line.pack(fill="x", pady=(6, 0))
        ttk.Button(line, text="全选", width=8,
                   command=lambda: self.set_all(True)).pack(side="left", padx=2)
        ttk.Button(line, text="全不选", width=8,
                   command=lambda: self.set_all(False)).pack(side="left", padx=2)
        ttk.Button(line, text="仅链构象", width=10,
                   command=lambda: self.set_only(["rg", "ree", "dihedral"])
                   ).pack(side="left", padx=2)

        # ---- 6. 运行
        f6 = ttk.LabelFrame(inner, text="6. 运行", padding=8)
        f6.pack(fill="x", pady=4)
        self.btn_run = ttk.Button(f6, text="开始分析", command=self.start_analysis)
        self.btn_run.pack(fill="x")
        ttk.Label(f6, text="运行过程与结果记录在右侧「运行日志」页；\n"
                           "图表由「导出全部结果」生成为 PNG。",
                  foreground="#666", justify="left").pack(anchor="w",
                                                          pady=(self.px(6), 0))

    def _file_row(self, parent, label, var, command) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="选择…", width=8, command=command).pack(side="left",
                                                                     padx=(4, 0))

    # ---------------------------------------------------------- 右侧区域
    def _build_right(self, parent: ttk.Frame) -> None:
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill="both", expand=True)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ---------------- 1. 分析结果：左目录树 + 右数据表
        self.tab_data = ttk.Frame(self.nb)
        self.nb.add(self.tab_data, text="分析结果")

        head = ttk.Frame(self.tab_data)
        head.pack(fill="x", padx=self.px(8), pady=(self.px(6), self.px(2)))
        self.lbl_table = ttk.Label(head, text="还没有结果：读取体系后点「开始分析」",
                                   foreground="#555")
        self.lbl_table.pack(side="left")
        ttk.Button(head, text="导出当前表 CSV", width=16,
                   command=self.export_current_table).pack(side="right", padx=2)
        ttk.Button(head, text="复制当前表", width=12,
                   command=self.copy_current_table).pack(side="right", padx=2)

        split = ttk.Frame(self.tab_data)
        split.pack(fill="both", expand=True, padx=self.px(8),
                   pady=(0, self.px(8)))

        # 左：分析 → 数据表 目录树（固定宽度，不可拖动）
        nav = ttk.Frame(split, width=self.px(self.BASE_TABLE_NAV))
        nav.pack(side="left", fill="y", expand=False)
        nav.pack_propagate(False)
        nvsb = ttk.Scrollbar(nav, orient="vertical")
        self.tbl_tree = ttk.Treeview(nav, show="tree", selectmode="browse",
                                     yscrollcommand=nvsb.set, height=20)
        nvsb.configure(command=self.tbl_tree.yview)
        self.tbl_tree.column("#0", width=self.px(self.BASE_TABLE_NAV - 26), anchor="w")
        self.tbl_tree.pack(side="left", fill="both", expand=True)
        nvsb.pack(side="right", fill="y")
        self.tbl_tree.bind("<<TreeviewSelect>>", self._on_table_select)

        # 右：数据表
        table_wrap = ttk.Frame(split)
        table_wrap.pack(side="left", fill="both", expand=True, padx=(self.px(8), 0))
        self.tbl = ttk.Treeview(table_wrap, show="headings", selectmode="extended")
        ysb = ttk.Scrollbar(table_wrap, orient="vertical", command=self.tbl.yview)
        xsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self.tbl.xview)
        self.tbl.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tbl.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)

        # ---------------- 2. 汇总
        self.tab_summary = ttk.Frame(self.nb)
        self.nb.add(self.tab_summary, text="汇总")
        cols = ("分析", "标题", "数据表数", "统计项")
        self.tree = ttk.Treeview(self.tab_summary, columns=cols, show="headings",
                                 height=18)
        for c, w in zip(cols, (110, 520, 90, 80)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=self.px(w), anchor="w")
        tsb = ttk.Scrollbar(self.tab_summary, orient="vertical",
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_summary_select)

        # ---------------- 3. 统计量明细
        self.tab_stats = ttk.Frame(self.nb)
        self.nb.add(self.tab_stats, text="统计量明细")
        self.txt_stats = tk.Text(self.tab_stats, wrap="none",
                                 font=self._mono_font())
        ssb = ttk.Scrollbar(self.tab_stats, orient="vertical",
                            command=self.txt_stats.yview)
        self.txt_stats.configure(yscrollcommand=ssb.set)
        self.txt_stats.pack(side="left", fill="both", expand=True)
        ssb.pack(side="right", fill="y")

        # ---------------- 4. 运行日志
        self.tab_log = ttk.Frame(self.nb)
        self.nb.add(self.tab_log, text="运行日志")
        log_head = ttk.Frame(self.tab_log)
        log_head.pack(fill="x", padx=self.px(8), pady=(self.px(6), self.px(2)))
        ttk.Button(log_head, text="清空日志", width=10,
                   command=lambda: self.txt_log.delete("1.0", "end")).pack(side="right")
        self.txt_log = tk.Text(self.tab_log, wrap="word", font=self._mono_font())
        lsb = ttk.Scrollbar(self.tab_log, orient="vertical",
                            command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=lsb.set)
        self.txt_log.pack(side="left", fill="both", expand=True,
                          padx=(self.px(8), 0), pady=(0, self.px(8)))
        lsb.pack(side="right", fill="y", padx=(0, self.px(8)),
                 pady=(0, self.px(8)))

    # ================================================================ 文件
    def choose_top(self) -> None:
        p = filedialog.askopenfilename(
            title="选择拓扑文件",
            filetypes=[("拓扑文件", "*.tpr *.gro *.pdb *.psf *.prmtop *.top *.xyz"),
                       ("GROMACS TPR", "*.tpr"), ("所有文件", "*.*")])
        if p:
            self.var_top.set(p)
            if not self.var_traj.get():
                stem = os.path.splitext(p)[0]
                for suf in (".xtc", ".trr", ".dcd", ".nc"):
                    if os.path.isfile(stem + suf):
                        self.var_traj.set(stem + suf)
                        self.log(f"自动匹配到轨迹文件: {os.path.basename(stem + suf)}")
                        break

    def choose_traj(self) -> None:
        p = filedialog.askopenfilename(
            title="选择轨迹文件",
            filetypes=[("轨迹文件", "*.xtc *.trr *.dcd *.nc *.trj"),
                       ("所有文件", "*.*")])
        if p:
            self.var_traj.set(p)

    def choose_outdir(self) -> None:
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.var_outdir.set(p)

    # ================================================================ 读取
    def load_system(self) -> None:
        top = self.var_top.get().strip()
        traj = self.var_traj.get().strip() or None
        if not top:
            messagebox.showwarning("提示", "请先选择拓扑文件（.tpr / .gro / .pdb）。")
            return
        self.set_status("正在读取体系…")
        self.root.update_idletasks()
        try:
            az = Analyzer(top, traj)
            az.auto_setup()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("读取失败", f"{type(exc).__name__}: {exc}")
            self.set_status("读取失败")
            return
        self.analyzer = az
        self.results = {}
        self.result_order = []
        self._reset_tabs()

        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", az.info_text(max_chain_types=25))

        # 填充主链下拉框
        options = ["自动（最大链）"]
        for c in sorted(az.info.chain_types, key=lambda c: -c.n_atoms):
            options.append(f"{c.label}（{c.n_atoms} 原子 / {c.count} 条）")
        for name, ag in az.components.items():
            options.append(f"组分 {name}（{ag.n_atoms} 原子）")
        self.cmb_primary.configure(values=options)
        self.var_primary.set(options[0])

        # 填充组分列表
        self.lst_comp.delete(0, "end")
        self._component_names = list(az.components)
        for name, ag in az.components.items():
            self.lst_comp.insert("end", f"{name}   ({ag.n_atoms} 原子)")
        for i in range(len(self._component_names)):
            self.lst_comp.selection_set(i)

        self.log(f"读取成功: {az.trajectory}")
        self.log(f"原子数 {az.trajectory.n_atoms}, 帧数 {az.trajectory.n_frames}, "
                 f"时间 {az.trajectory.total_time_ps:.1f} ps")
        self.set_status("体系已读取，可以开始分析")

    def _reset_tabs(self) -> None:
        self._table_map = {}
        self._current_name = None
        self._current_table = None
        for item in self.tbl_tree.get_children():
            self.tbl_tree.delete(item)
        self.tbl.delete(*self.tbl.get_children())
        self.tbl.configure(columns=(), show="headings")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.txt_stats.delete("1.0", "end")
        try:
            self.lbl_table.configure(text="还没有结果：读取体系后点「开始分析」")
        except Exception:  # noqa: BLE001
            pass

    # ================================================================ 选择
    def _apply_selection(self, az: Analyzer) -> None:
        # 主链
        label = self.var_primary.get()
        custom = self.var_custom_sel.get().strip()
        if custom:
            ag = select(az.universe, query=custom)
            if ag.n_atoms == 0:
                raise ValueError(f"自定义选择语句没有选中任何原子: {custom!r}")
            az.set_primary(ag, f"自定义: {custom}")
        elif label.startswith("组分 "):
            name = label[3:].split("（")[0].strip()
            if name in az.components:
                az.set_primary(az.components[name], f"组分 {name}")
        elif label != "自动（最大链）":
            name = label.split("（")[0].strip()
            for c in az.info.chain_types:
                if c.label == name:
                    from mdta.selection import chains

                    got = chains(az.universe, segid=c.segid, resname=c.resname,
                                 min_atoms=c.n_atoms)
                    if got:
                        az.set_primary(got[0], f"{name} 的第 1 条链")
                    break
        # 组分
        picked = [self._component_names[i] for i in self.lst_comp.curselection()]
        if picked:
            az.set_components({k: v for k, v in az.components.items() if k in picked})

    # ================================================================ 运行
    def _read_params(self) -> dict:
        def fnum(var, name):
            s = var.get().strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError as exc:
                raise ValueError(f"{name} 必须是数字，当前为 {s!r}") from exc

        start = fnum(self.var_start, "起始时间")
        stop = fnum(self.var_stop, "结束时间")
        equil = fnum(self.var_equil, "平衡段")
        interval = fnum(self.var_interval, "分析间隔")
        mf = self.var_maxframes.get().strip()
        max_frames = int(mf) if mf else None
        cutoff = float(self.var_cutoff.get().strip() or 5.0)
        nbins = int(self.var_nbins.get().strip() or 100)
        rmax = float(self.var_rmax.get().strip() or 12.0)
        axis = int(self.var_axis.get().strip()[-2])
        dmode = self.var_dihedral.get().split(" ")[0]
        return {
            "_frames": dict(start_ps=start, stop_ps=stop, equil_ps=equil,
                            interval_ps=interval, max_frames=max_frames),
            "density": {"axis": axis, "nbins": nbins},
            "interface": {"axis": axis, "nbins": max(nbins, 60)},
            "rdf": {"rmax": rmax, "nbins": 120},
            "contact": {"cutoff": cutoff},
            "dihedral": {"mode": dmode},
        }

    def start_analysis(self) -> None:
        if self._busy:
            messagebox.showinfo("提示", "分析正在进行中，请稍候。")
            return
        if self.analyzer is None:
            messagebox.showwarning("提示", "请先读取体系。")
            return
        which = [n for n in DEFAULT_ORDER if self.analysis_vars[n].get()]
        if not which:
            messagebox.showwarning("提示", "请至少勾选一项分析功能。")
            return
        try:
            params = self._read_params()
            self._apply_selection(self.analyzer)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("参数错误", str(exc))
            return

        frames = params.pop("_frames")
        self.analyzer.set_frames(**frames)
        self.results = {}
        self.result_order = []
        self._reset_tabs()
        self._busy = True
        self.btn_run.configure(state="disabled")
        self.var_progress.set(0.0)
        self.set_status("分析中…")
        self.log("=" * 60)
        self.log(f"开始分析 {len(which)} 项: {', '.join(which)}")

        az = self.analyzer
        q = self.msg_queue

        def report(frac: float, message: str) -> None:
            q.put(("progress", float(frac) * 100.0))
            if message:
                q.put(("tick", message))

        def work():
            try:
                az.run_all(which, params=params, outdir=None, verbose=False,
                           raise_errors=True, progress=report)
                q.put(("done", list(az.results.items())))
            except Exception:  # noqa: BLE001
                q.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "progress":
                    self.var_progress.set(payload)
                elif kind == "tick":
                    self.set_status(payload)
                elif kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self._after_id = self.root.after(120, self._drain_queue)

    def on_close(self) -> None:
        """关闭窗口：停掉事件轮询，避免销毁后回调报错。"""
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:  # noqa: BLE001
                pass
            self._after_id = None
        self.root.destroy()

    def _on_done(self, items) -> None:
        self._busy = False
        self.btn_run.configure(state="normal")
        self.var_progress.set(100.0)
        self.results = dict(items)
        self.result_order = [k for k, _ in items]
        for name, res in items:
            self._add_result_entry(name, res)
            self.log(f"  ✓ {ANALYSIS_TITLES.get(name, name)}: "
                     f"{len(res.dataframes())} 张数据表, {len(res.summary)} 项统计")
        self.set_status(f"完成：{len(items)} 项分析（图表可用「导出全部结果」生成 PNG）")
        if items:
            self.nb.select(self.tab_data)
            self._select_table(self.result_order[0], 0)

    def _on_error(self, tb: str) -> None:
        self._busy = False
        self.btn_run.configure(state="normal")
        self.var_progress.set(0.0)
        self.set_status("分析失败")
        self.log(tb)
        messagebox.showerror("分析失败", tb.strip().splitlines()[-1])

    # ================================================================ 结果
    @staticmethod
    def _table_title(res, index: int) -> str:
        """数据表的标题：多面板时用面板标题，单面板时用分析标题。"""
        panel = res.panel(index)
        if max(res.panel_count, 1) == 1:
            return res.title or panel.title or "数据表"
        return panel.title or f"表 {index + 1}"

    def _add_result_entry(self, name: str, res) -> None:
        """在「分析结果」目录树和「汇总」表里登记一项分析（不绘制任何图）。"""
        self.results.setdefault(name, res)
        frames = res.dataframes()
        n_tables = max(len(frames), 1)

        parent = self.tbl_tree.insert("", "end",
                                      text=ANALYSIS_TITLES.get(name, name),
                                      open=True)
        for i, key in enumerate(frames.keys()):
            iid = self.tbl_tree.insert(parent, "end", text="  " + key)
            self._table_map[iid] = (name, i)

        self.tree.insert("", "end", values=(name, res.title, n_tables,
                                            len(res.summary)))

    # ---------------------------------------------------------- 数据表浏览
    def _frames_of(self, name: str):
        res = self.results.get(name)
        if res is None:
            return {}
        return res.dataframes()

    def _select_table(self, name: str, index: int) -> None:
        """在「分析结果」页里显示第 ``index`` 张数据表。"""
        frames = self._frames_of(name)
        if not frames:
            return
        keys = list(frames.keys())
        index = max(0, min(int(index), len(keys) - 1))
        df = frames[keys[index]]
        res = self.results[name]
        self._current_name = name
        self._current_table = (name, index)
        self._fill_table(df)
        self.lbl_table.configure(
            text=f"{res.title}  ▸  {keys[index]}    "
                 f"（{len(df)} 行 × {len(df.columns)} 列）")
        self.set_status(f"{res.title} ▸ {keys[index]}：{len(df)} 行 × "
                        f"{len(df.columns)} 列")

    def _fill_table(self, df, max_rows: int = 2000) -> None:
        """把 DataFrame 填进 Treeview（大表只显示前 ``max_rows`` 行）。"""
        self.tbl.delete(*self.tbl.get_children())
        cols = [str(c) for c in df.columns]
        self.tbl.configure(columns=cols, show="headings")
        for c in cols:
            self.tbl.heading(c, text=c)
            try:
                width = max(90, min(220, 12 + 9 * max(len(c), 6)))
            except Exception:  # noqa: BLE001
                width = 110
            self.tbl.column(c, width=self.px(width), anchor="e", stretch=True)

        n = len(df)
        shown = min(n, int(max_rows))
        vals = df.iloc[:shown].to_numpy()
        for row in vals:
            self.tbl.insert("", "end", values=[_fmt_cell(v) for v in row])
        if n > shown:
            self.log(f"  （表共 {n} 行，界面只显示前 {shown} 行；"
                     f"完整数据请用「导出当前表 CSV」）")

    def _on_table_select(self, _event=None) -> None:
        sel = self.tbl_tree.selection()
        if not sel:
            return
        target = self._table_map.get(sel[0])
        if target is None:
            return
        self._select_table(*target)

    def _on_tab_changed(self, _event=None) -> None:
        try:
            tab = self.nb.nametowidget(self.nb.select())
        except Exception:  # noqa: BLE001
            return
        if tab is self.tab_stats:
            self._show_stats()

    def _on_summary_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0], "values")[0]
        if name in self.results:
            self.nb.select(self.tab_data)
            self._select_table(name, 0)

    def _show_stats(self) -> None:
        self.txt_stats.delete("1.0", "end")
        for name in self.result_order:
            res = self.results[name]
            self.txt_stats.insert("end", f"【{res.title}】\n")
            for k, v in res.summary.items():
                if isinstance(v, float):
                    self.txt_stats.insert("end", f"    {k:<34} = {v:.6g}\n")
                else:
                    self.txt_stats.insert("end", f"    {k:<34} = {v}\n")
            for note in res.notes:
                self.txt_stats.insert("end", f"    注: {note}\n")
            self.txt_stats.insert("end", "\n")

    # ---------------------------------------------------------- 当前表 / 目录
    def _current_frame(self):
        """返回当前选中的数据表 ``(分析名, 表序号, 表名, DataFrame)``。"""
        target = getattr(self, "_current_table", None)
        if not target:
            return None
        name, index = target
        frames = self._frames_of(name)
        keys = list(frames.keys())
        if not keys:
            return None
        index = max(0, min(index, len(keys) - 1))
        return name, index, keys[index], frames[keys[index]]

    def copy_current_table(self) -> None:
        cur = self._current_frame()
        if cur is None:
            messagebox.showwarning("提示", "还没有可复制的数据表。")
            return
        _name, _i, _key, df = cur
        text = df.to_csv(index=False)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.set_status(f"已复制 {len(df)} 行到剪贴板（可直接粘进 Excel）")
        self.log(f"已复制数据表到剪贴板：{len(df)} 行 × {len(df.columns)} 列")

    def export_current_table(self) -> None:
        cur = self._current_frame()
        if cur is None:
            messagebox.showwarning("提示", "还没有可导出的数据表。")
            return
        name, index, key, df = cur
        outdir = self.var_outdir.get().strip() or "analysis_results"
        os.makedirs(outdir, exist_ok=True)
        from mdta.export import safe_name

        path = os.path.join(outdir, f"{safe_name(key)}.csv")
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", f"{type(exc).__name__}: {exc}")
            return
        self.set_status(f"已导出 {os.path.basename(path)}")
        self.log(f"已导出 {path}")
        messagebox.showinfo("导出完成", path)

    def open_outdir(self) -> None:
        outdir = os.path.abspath(self.var_outdir.get().strip() or "analysis_results")
        os.makedirs(outdir, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(outdir)          # noqa: S606 - 打开资源管理器
            elif sys.platform == "darwin":
                subprocess.Popen(["open", outdir])
            else:
                subprocess.Popen(["xdg-open", outdir])
            self.log(f"已打开输出目录: {outdir}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showinfo("输出目录", f"{outdir}\n\n（自动打开失败: {exc}）")

    # ================================================================ 导出
    def _current_result(self):
        """当前在「分析结果」页里选中的分析，返回 ``(名称, 结果)`` 或 ``None``。"""
        name = getattr(self, "_current_name", None)
        if name in self.results:
            return name, self.results[name]
        if self.result_order:
            n = self.result_order[0]
            return n, self.results[n]
        return None

    def export_current(self, kind: str) -> None:
        if not self.results:
            messagebox.showwarning("提示", "还没有可导出的结果。")
            return
        cur = self._current_result()
        outdir = self.var_outdir.get().strip() or "analysis_results"
        try:
            from mdta.export import export_all, export_csv, export_png, export_excel

            if kind == "excel":
                path = export_excel(list(self.results.values()),
                                    os.path.join(outdir, "summary.xlsx"))
                self.log(f"已导出汇总表: {path}")
                messagebox.showinfo("导出完成", path)
                return
            targets = [cur] if cur else list(self.results.items())
            if kind == "csv":
                for name, res in targets:
                    for p in export_csv(res, outdir):
                        self.log(f"已导出: {p}")
            elif kind == "png":
                for name, res in targets:
                    p = export_png(res, outdir)
                    self.log(f"已导出: {p}")
            self.set_status(f"已导出到 {os.path.abspath(outdir)}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", f"{type(exc).__name__}: {exc}")

    def export_all(self) -> None:
        if not self.results:
            messagebox.showwarning("提示", "还没有可导出的结果。")
            return
        outdir = self.var_outdir.get().strip() or "analysis_results"
        try:
            from mdta.export import export_all

            out = export_all(list(self.results.values()), outdir,
                             formats=("csv", "png"), excel=True)
            self.log(f"全部导出完成: {out['dir']}")
            self.set_status(f"已导出到 {out['dir']}")
            messagebox.showinfo("导出完成",
                                f"CSV: {len(out['csv'])} 个\n"
                                f"PNG: {len(out['png'])} 个\n"
                                f"Excel: {os.path.basename(out['excel'] or '')}\n\n"
                                f"目录: {out['dir']}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", f"{type(exc).__name__}: {exc}")

    # ================================================================ 杂项
    def set_all(self, value: bool) -> None:
        for var in self.analysis_vars.values():
            var.set(value)

    def set_only(self, names) -> None:
        for k, var in self.analysis_vars.items():
            var.set(k in names)

    def set_status(self, text: str) -> None:
        self.var_status.set(text)

    def log(self, text: str) -> None:
        self.txt_log.insert("end", text.rstrip() + "\n")
        self.txt_log.see("end")


def main(argv: list[str] | None = None) -> int:
    """图形界面入口。

    可用环境变量 ``MDTA_GUI_SCALE`` 强制指定显示缩放（例如 ``1.5``），
    用于系统 DPI 探测不准的情况。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    # ① 必须在 tk.Tk() 之前开启高 DPI 感知，否则 Windows 会拉伸位图导致发虚
    scale = enable_high_dpi()
    env = os.environ.get("MDTA_GUI_SCALE", "").strip()
    if env:
        try:
            scale = float(env)
        except ValueError:
            pass
    scale = max(0.5, min(4.0, scale))

    root = tk.Tk()
    try:
        # ② Tk 的 scaling 单位是"每磅多少像素"，100% 时为 96/72
        root.tk.call("tk", "scaling", scale * 96.0 / 72.0)
    except Exception:  # noqa: BLE001
        pass
    style = ttk.Style()
    for theme in ("vista", "clam", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break

    app = MDTAGui(root, dpi_scale=scale)
    args = [a for a in (sys.argv[1:] if argv is None else argv) if a]
    if args:
        app.var_top.set(args[0])
        if len(args) > 1:
            app.var_traj.set(args[1])
        root.after(300, app.load_system)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
