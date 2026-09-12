# -*- coding: utf-8 -*-
"""数据可视化模块（设计大纲第 21 章）。

统一把 :class:`~mdta.core.AnalysisResult` 画成 matplotlib 图：

- 一个结果可以有多个"面板"（子图）；
- 每个面板内可以有多条曲线；
- 支持 line / step / bar / scatter 四种画法；
- 自动配置中文字体，避免中文标题与坐标轴出现方框。
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

_FONT_CANDIDATES = [
    "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun", "KaiTi",
    "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Arial Unicode MS",
    "PingFang SC", "Heiti SC", "Hiragino Sans GB",
]

_font_ready = False
_cjk_available = False


def set_agg_backend() -> str:
    """强制使用非交互的 Agg 后端。

    本工具的画图只用于**保存文件**（PNG），不需要任何窗口。默认后端在
    Windows 上可能是 TkAgg，那样在后台线程里画图会报警告甚至崩溃
    （``Tcl_AsyncDelete: async handler deleted by the wrong thread``）。
    命令行、桌面操作台、Web 服务都应当先调用本函数。
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    return matplotlib.get_backend()


#: 字体缺字时的等价替换（仅在第一选择字体确实没有该字形时才用）
_FALLBACK_CHARS = {
    "ᵀ": "^T", "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁻": "-", "⁺": "+", "⟨": "<", "⟩": ">", "∝": "~", "≈": "~",
    "≤": "<=", "≥": ">=", "→": "->", "←": "<-", "−": "-", "⋅": ".",
    "·": ".", "√": "sqrt",
}

_coverage_cache: set[int] | None = None


def _font_coverage() -> set[int]:
    """**首选字体**支持的码点集合（用于检测缺字）。

    注意不能用"字体列表的并集"：matplotlib 渲染时实际是由列表解析出的
    第一个字体来排版的，并集判据会漏掉"首选字体没有该字形"的情况
    （例如 Microsoft YaHei 没有 U+2080 下标 0、也没有 U+1D40 ``ᵀ``）。
    因此这里只取 :func:`matplotlib.font_manager.findfont` 解析出的那一个字体。
    """
    global _coverage_cache
    if _coverage_cache is not None:
        return _coverage_cache
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    cps: set[int] = set()
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(
                family=matplotlib.rcParams.get("font.sans-serif")),
            fallback_to_default=True)
        cps = set(FT2Font(path).get_charmap().keys())
    except Exception:  # noqa: BLE001
        cps = set()
    _coverage_cache = cps
    return cps


def sanitize_text(text) -> str:
    """把字体渲染不了的字符换成等价写法，避免图上出现方框。

    例如 Microsoft YaHei 没有 ``ᵀ``（U+1D40）与下标 ``₀``（U+2080），
    直接画出来会是缺字方框；这里检测后替换成 ``^T`` / ``0``。
    """
    s = str(text)
    cps = _font_coverage()
    if not cps:
        return s
    out = []
    for ch in s:
        if ch in "\n\t" or ord(ch) in cps:
            out.append(ch)
        else:
            out.append(_FALLBACK_CHARS.get(ch, ""))
    return "".join(out)


def setup_matplotlib(force: bool = False, backend: str | None = None) -> bool:
    """配置 matplotlib 的中文字体与后端。返回是否成功找到中文字体。

    ``backend`` 传入 ``"Agg"`` 时会强制切到非交互后端（保存文件用）。
    """
    global _font_ready, _cjk_available, _coverage_cache
    if backend:
        set_agg_backend()
    if _font_ready and not force:
        return _cjk_available
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in _FONT_CANDIDATES if f in available]
    base = ["DejaVu Sans", "Arial", "Helvetica"]
    matplotlib.rcParams["font.sans-serif"] = chosen + base
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 110
    matplotlib.rcParams["savefig.dpi"] = 200
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.3
    matplotlib.rcParams["legend.frameon"] = False
    # 图形界面会为每张图缓存一个 Figure（10 项分析约 20+ 张），这是刻意设计，
    # 因此关掉 matplotlib 的"打开图过多"警告；切换/重置时会显式 close。
    matplotlib.rcParams["figure.max_open_warning"] = 0
    _cjk_available = bool(chosen)
    _font_ready = True
    _coverage_cache = None          # 字体列表变了，重新算覆盖范围
    return _cjk_available


def cjk_available() -> bool:
    """当前环境是否找到中文字体。"""
    setup_matplotlib()
    return _cjk_available


def _bar_width(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    dx = np.diff(np.sort(x))
    dx = dx[np.isfinite(dx) & (dx > 0)]
    return float(np.median(dx)) if dx.size else 1.0


def _apply_font_sizes(dpi: float | None, base_fontsize: float) -> float:
    """按渲染 dpi 同步放大基础字号，保证图元在屏幕上的物理尺寸不变。"""
    import matplotlib.pyplot as plt

    fs = float(base_fontsize) * (max(1.0, float(dpi) / 100.0)
                                 if (dpi and float(dpi) > 0) else 1.0)
    plt.rcParams["font.size"] = fs
    plt.rcParams["axes.titlesize"] = fs * 1.1
    plt.rcParams["axes.labelsize"] = fs
    plt.rcParams["xtick.labelsize"] = fs * 0.9
    plt.rcParams["ytick.labelsize"] = fs * 0.9
    plt.rcParams["legend.fontsize"] = fs * 0.85
    return fs


def _draw_curves(ax, curves, max_legend: int = 14) -> None:
    """把一组曲线画到给定坐标轴上（图例文字先做缺字替换）。"""
    for c in curves:
        label = sanitize_text(c.label)
        if c.kind == "bar":
            ax.bar(c.x, c.y, width=_bar_width(c.x), align="center",
                   label=label, alpha=0.75, edgecolor="none")
        elif c.kind == "step":
            ax.step(c.x, c.y, where="mid", label=label, linewidth=1.4)
        elif c.kind == "scatter":
            ax.plot(c.x, c.y, "o", label=label, markersize=3.5)
        else:
            ax.plot(c.x, c.y, label=label, linewidth=1.6)


def _style_axes(ax, panel, n_curves: int, max_legend: int = 14) -> None:
    """按面板信息设置坐标轴标签、标题、坐标尺度与图例。"""
    if panel.xlabel:
        ax.set_xlabel(sanitize_text(panel.xlabel))
    if panel.ylabel:
        ax.set_ylabel(sanitize_text(panel.ylabel))
    if panel.title:
        ax.set_title(sanitize_text(panel.title))
    if panel.xscale != "linear":
        ax.set_xscale(panel.xscale)
    if panel.yscale != "linear":
        ax.set_yscale(panel.yscale)
    if panel.legend and 1 < n_curves <= max_legend:
        ax.legend(loc="best", ncol=1 if n_curves <= 6 else 2)


#: :func:`plot_panel` 的**初始**边距 ``(left, right, bottom, top)``。
#: 真正的边距会在绘制时按文字实际尺寸微调，见 :func:`fit_panel_margins`。
PANEL_MARGINS = (0.13, 0.93, 0.13, 0.93)

#: 坐标轴（绘图内容）应占画布的比例
PANEL_CONTENT = (0.80, 0.80)


def panel_content_fraction() -> tuple[float, float]:
    """返回坐标轴框应占画布的 (宽度比, 高度比)，即 4/5。"""
    return PANEL_CONTENT


def pixel_figsize(width_px: int, height_px: int,
                  dpi: float) -> tuple[float, float]:
    """给出能**精确**渲染成 ``width_px × height_px`` 像素的 figsize（英寸）。

    matplotlib 的画布尺寸是把 ``figsize × dpi`` 截断取整得到的，
    浮点误差会让 823 px 变成 822 px；这里补 0.5 px 消除该误差，
    从而保证"固定尺寸"在像素级别也严格一致。
    """
    d = max(float(dpi), 1.0)
    return ((float(width_px) + 0.5) / d, (float(height_px) + 0.5) / d)


def fit_panel_margins(fig, ax, content: tuple[float, float] = PANEL_CONTENT,
                      max_iter: int = 4) -> tuple[float, float, float, float]:
    """设置边距：坐标轴框占画布 ``content`` 比例，同时保证文字不被裁切。

    做法是先按 :data:`PANEL_MARGINS` 画一次，量出坐标轴**外面**真正需要多少
    空间（标题、x/y 轴标签、刻度标签），再在"内容占 4/5"的目标下把坐标轴框
    摆到能容纳这些文字的位置；剩余空隙左右（上下）均分。

    因为测量的是渲染后的实际像素尺寸，所以无论字号、单位符号（Å、°）、
    还是刻度位数如何变化，都不会出现标题或坐标轴超出画布的情况。

    返回最终使用的 ``(left, right, bottom, top)``（0–1 的画布比例）。
    """
    W, H = float(fig.bbox.width), float(fig.bbox.height)
    left, right, bottom, top = PANEL_MARGINS
    if W <= 0 or H <= 0:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        return (left, right, bottom, top)

    target_w, target_h = content
    for _ in range(max(1, int(max_iter))):
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        try:
            renderer = fig.canvas.get_renderer()
            tb = ax.get_tightbbox(renderer)      # 含标题/标签/刻度，单位像素
            ab = ax.get_window_extent(renderer)
        except Exception:  # noqa: BLE001 - 拿不到渲染器就退回固定边距
            return (left, right, bottom, top)
        need_l = max(0.0, float(ab.x0 - tb.x0))
        need_r = max(0.0, float(tb.x1 - ab.x1))
        need_b = max(0.0, float(ab.y0 - tb.y0))
        need_t = max(0.0, float(tb.y1 - ab.y1))
        if not all(np.isfinite(v) for v in (need_l, need_r, need_b, need_t)):
            return (left, right, bottom, top)

        w = min(target_w * W, max(W - need_l - need_r, 1.0))
        h = min(target_h * H, max(H - need_b - need_t, 1.0))
        x0 = need_l + max(0.0, (W - need_l - need_r - w) / 2.0)
        y0 = need_b + max(0.0, (H - need_b - need_t - h) / 2.0)
        new = (x0 / W, (x0 + w) / W, y0 / H, (y0 + h) / H)
        converged = all(abs(a - b) < 1e-4
                        for a, b in zip((left, right, bottom, top), new))
        left, right, bottom, top = new
        if converged:
            break
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return (left, right, bottom, top)


def plot_panel(result, index: int, figsize: tuple[float, float] | None = None,
               dpi: float | None = None, base_fontsize: float = 10.0,
               max_legend: int = 14, with_note: bool = False,
               margins: tuple[float, float, float, float] | None = None,
               content: tuple[float, float] = PANEL_CONTENT):
    """只画结果中的**第 ``index`` 个面板**，返回单张图的 ``Figure``。

    图形界面用它实现"一个分析一页，通过图标题导航切换"。

    版面约定（配合图形界面的"固定显示、不缩放"）
    --------------------------------------------
    不再用 ``tight_layout``，而是由 :func:`fit_panel_margins` 显式排版：

    - 坐标轴（真正的绘图内容）占画布宽度与高度的 **4/5**；
    - 边距按标题、坐标轴标签、刻度标签的**实际渲染尺寸**分配，
      因此**不会有任何文字超出画布**；
    - 不给 ``margins`` 时使用自动结果；给了就按给定值固定。
    - 画布尺寸由调用者给定后固定不变，显示与导出都稳定可复现。
    """
    import matplotlib.pyplot as plt

    setup_matplotlib()
    fs = _apply_font_sizes(dpi, base_fontsize)
    panel = result.panel(index)
    cs = result.curves_of(index)
    n_panels = max(result.panel_count, 1)
    if figsize is None:
        figsize = (8.0, 6.0)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=(float(dpi) if dpi else None))
    _draw_curves(ax, cs, max_legend=max_legend)
    _style_axes(ax, panel, len(cs), max_legend=max_legend)

    if n_panels == 1:
        # 只有一张图时直接用结果标题，避免标题重复
        ax.set_title(sanitize_text(result.title))
    else:
        ax.set_title(panel.title or f"图 {index + 1}")

    if margins is not None:
        l, r, b, t = margins
        fig.subplots_adjust(left=l, right=r, bottom=b, top=t)
    else:
        fit_panel_margins(fig, ax, content=content)

    if with_note and result.notes:
        fig.text(0.005, 0.002, sanitize_text("注: " + result.notes[0]),
                 fontsize=fs * 0.7, va="bottom", alpha=0.75)
    return fig


def plot_result(result, figsize: tuple[float, float] | None = None,
                max_legend: int = 14, dpi: float | None = None,
                base_fontsize: float = 10.0):
    """把 :class:`AnalysisResult` 的全部面板拼成一张图，返回 ``Figure``。

    参数
    ----
    figsize
        ``(宽, 高)``，单位英寸；``None`` 时按面板数自动取。
    dpi
        图形渲染分辨率。嵌入图形界面时应传入与屏幕一致的值
        （例如 ``100 × 显示缩放``），这样文字与曲线按真实像素渲染，不会发虚。
    base_fontsize
        基础字号；传入 ``dpi`` 时会同步放大，保证图元在屏幕上的物理尺寸不变。
    """
    import matplotlib.pyplot as plt

    setup_matplotlib()
    n_panels = max(result.panel_count, 1)
    if figsize is None:
        figsize = (8.0, min(3.4 * n_panels, 16.0))
    fs = _apply_font_sizes(dpi, base_fontsize)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False,
                             dpi=(float(dpi) if dpi else None))
    axes = axes[:, 0]

    for p in range(n_panels):
        ax = axes[p]
        cs = result.curves_of(p)
        _draw_curves(ax, cs, max_legend=max_legend)
        _style_axes(ax, result.panel(p), len(cs), max_legend=max_legend)

    fig.suptitle(sanitize_text(result.title), fontsize=fs * 1.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if result.notes:
        fig.text(0.005, 0.002, sanitize_text("注: " + result.notes[0]),
                 fontsize=fs * 0.7, va="bottom", alpha=0.75)
    return fig


def save_figure(fig, path: str, dpi: int = 200, close: bool = True) -> str:
    """保存图片，返回实际路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path


def plot_many(results: Sequence, figsize=None, max_cols: int = 2):
    """把多个结果画到一张大图上（每个结果占一列子图列）。"""
    import matplotlib.pyplot as plt

    setup_matplotlib()
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("没有可绘制的结果")
    ncols = min(max_cols, len(results))
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (7 * ncols, 4 * nrows),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        cs = r.curves_of(0)
        for c in cs:
            if c.kind == "bar":
                ax.bar(c.x, c.y, width=_bar_width(c.x), align="center",
                       label=c.label, alpha=0.75)
            else:
                ax.plot(c.x, c.y, label=c.label, linewidth=1.5)
        ax.set_title(r.title, fontsize=10)
        p0 = r.panel(0)
        if p0.xlabel:
            ax.set_xlabel(p0.xlabel, fontsize=8)
        if p0.ylabel:
            ax.set_ylabel(p0.ylabel, fontsize=8)
        if p0.xscale != "linear":
            ax.set_xscale(p0.xscale)
        if p0.yscale != "linear":
            ax.set_yscale(p0.yscale)
        if 1 < len(cs) <= 6:
            ax.legend(fontsize=7)
    for j in range(len(results), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return fig
