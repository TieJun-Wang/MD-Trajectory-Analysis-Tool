# -*- coding: utf-8 -*-
"""数据导出模块（设计大纲第 22 章）。

支持导出::

    analysis_results/
    ├── Rg.csv / Rg.png
    ├── RDF.csv / RDF.png
    ├── density.csv / density.png
    ├── dihedral.csv / dihedral.png
    └── summary.xlsx
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def safe_name(name: str) -> str:
    """把分析名转成安全的文件名。"""
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in str(name)]
    out = "".join(keep).strip("_")
    return out or "result"


def result_to_frames(result) -> "OrderedDict[str, pd.DataFrame]":
    """结果 -> ``{表名: DataFrame}``。"""
    return result.dataframes()


def export_csv(result, outdir: str = ".", *, prefix: str | None = None,
               long_format: bool = False) -> list[str]:
    """把结果导出为 CSV。

    - 单面板结果（如 Rg）导出成一个宽表 ``Rg.csv``；
    - 多面板结果（如二面角）每个面板一个文件 ``dihedral__panel0.csv`` …；
    - ``long_format=True`` 时改为一个长表 ``panel, curve, x_label, y_label, x, y``。
    """
    os.makedirs(outdir, exist_ok=True)
    base = safe_name(prefix or result.name)
    if long_format:
        return [export_csv_long(result, outdir, prefix=prefix)]

    frames = result_to_frames(result)
    multi = len(frames) > 1
    written: list[str] = []
    for i, (_key, df) in enumerate(frames.items()):
        fname = f"{base}__panel{i}.csv" if multi else f"{base}.csv"
        path = os.path.join(outdir, fname)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    return written


def export_csv_long(result, outdir: str = ".", *, prefix: str | None = None) -> str:
    """把结果导成一张长表 CSV：``panel, curve, x_label, y_label, x, y``。"""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for p in range(result.panel_count):
        panel = result.panel(p)
        for c in result.curves_of(p):
            rows.append(pd.DataFrame({
                "panel": p,
                "panel_title": panel.title,
                "curve": c.label,
                "x_label": panel.xlabel,
                "y_label": panel.ylabel,
                "x": c.x,
                "y": c.y,
            }))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["panel", "panel_title", "curve", "x_label", "y_label", "x", "y"])
    path = os.path.join(outdir, f"{safe_name(prefix or result.name)}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_png(result, outdir: str = ".", *, prefix: str | None = None,
               dpi: int = 200, figsize=None) -> str:
    """把结果画成图并存成 PNG。"""
    from .plotting import plot_result, save_figure

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{safe_name(prefix or result.name)}.png")
    fig = plot_result(result, figsize=figsize)
    return save_figure(fig, path, dpi=dpi)


def export_panel_pngs(result, outdir: str = ".", *, prefix: str | None = None,
                      dpi: int = 200, panel_px: tuple[int, int] | None = None,
                      figsize=None) -> list[str]:
    """把结果里的**每个面板分别**导出成一张 PNG。

    方便"一张图一个文件"地查看或排版（例如二面角的分布 / 构象比例 / 时间序列
    各自独立成图），与 :func:`export_png` 的多面板合并图并存。
    """
    from .plotting import pixel_figsize, plot_panel, save_figure

    os.makedirs(outdir, exist_ok=True)
    base = safe_name(prefix or result.name)
    n = max(result.panel_count, 1)
    multi = n > 1
    if figsize is None and panel_px is not None:
        figsize = pixel_figsize(panel_px[0], panel_px[1], dpi)
    written: list[str] = []
    for i in range(n):
        fig = plot_panel(result, i, figsize=figsize)
        fname = f"{base}__p{i + 1}.png" if multi else f"{base}.png"
        path = os.path.join(outdir, fname)
        written.append(save_figure(fig, path, dpi=dpi))
    return written


def summary_frame(results: Iterable) -> pd.DataFrame:
    """把多个结果的 summary 合并成一张表（每行一个分析）。"""
    rows = []
    for r in results:
        if r is None:
            continue
        row: "OrderedDict[str, object]" = OrderedDict()
        row["分析"] = r.name
        row["标题"] = r.title
        for k, v in r.summary.items():
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            elif isinstance(v, np.ndarray):
                v = np.array2string(v, precision=4)
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        # 把"分析/标题"放前面，其余列保持出现顺序
        cols = list(df.columns)
        first = [c for c in ("分析", "标题") if c in cols]
        df = df[first + [c for c in cols if c not in first]]
    return df


def export_excel(results: Sequence, path: str,
                 extra_tables: Mapping[str, pd.DataFrame] | None = None) -> str:
    """导出 Excel 汇总：``汇总`` 表 + 每个分析一张明细表。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    results = [r for r in results if r is not None]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_frame(results).to_excel(writer, sheet_name="汇总", index=False)
        used: set[str] = {"汇总"}
        for r in results:
            for key, df in result_to_frames(r).items():
                sheet = safe_name(key)[:31] or "sheet"
                n = 1
                base = sheet
                while sheet in used:
                    n += 1
                    sheet = f"{base[:28]}_{n}"
                used.add(sheet)
                df.to_excel(writer, sheet_name=sheet, index=False)
        if extra_tables:
            for name, df in extra_tables.items():
                sheet = safe_name(name)[:31] or "sheet"
                n = 1
                base = sheet
                while sheet in used:
                    n += 1
                    sheet = f"{base[:28]}_{n}"
                used.add(sheet)
                df.to_excel(writer, sheet_name=sheet, index=False)
    return path


def export_result(result, outdir: str = ".", *, formats: Sequence[str] = ("csv", "png"),
                  dpi: int = 200, prefix: str | None = None) -> list[str]:
    """按指定格式导出单个结果。"""
    written: list[str] = []
    fmts = {f.lower() for f in formats}
    if "csv" in fmts:
        written += export_csv(result, outdir, prefix=prefix)
    if "png" in fmts or "figure" in fmts:
        written.append(export_png(result, outdir, prefix=prefix, dpi=dpi))
    return written


def export_all(results: Sequence, outdir: str = "analysis_results", *,
               formats: Sequence[str] = ("csv", "png"), dpi: int = 200,
               excel: bool = True, excel_name: str = "summary.xlsx",
               panel_pngs: bool = False,
               extra_tables: Mapping[str, pd.DataFrame] | None = None,
               verbose: bool = False) -> dict:
    """按设计大纲第 22 章的目录结构导出全部结果。

    ``panel_pngs=True`` 时，除每个分析一张合并 PNG 外，再为**每个面板**
    单独导出一张 ``<分析名>__pN.png``，便于一张一张查看或排版。
    """
    os.makedirs(outdir, exist_ok=True)
    results = [r for r in results if r is not None]
    out: dict = {"csv": [], "png": [], "panel_png": [], "excel": None,
                 "dir": os.path.abspath(outdir)}
    for r in results:
        for path in export_result(r, outdir, formats=formats, dpi=dpi):
            key = "png" if path.lower().endswith(".png") else "csv"
            out[key].append(path)
            if verbose:
                print(f"  导出 {path}")
        if panel_pngs:
            for path in export_panel_pngs(r, outdir, dpi=dpi):
                out["panel_png"].append(path)
                if verbose:
                    print(f"  导出 {path}")
    if excel and results:
        out["excel"] = export_excel(results, os.path.join(outdir, excel_name),
                                    extra_tables=extra_tables)
        if verbose:
            print(f"  导出 {out['excel']}")
    return out
