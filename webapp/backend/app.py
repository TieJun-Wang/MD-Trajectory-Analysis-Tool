# -*- coding: utf-8 -*-
"""FastAPI 应用：把 :mod:`mdta` 分析引擎暴露成 REST API，并托管前端构建产物。

启动::

    python -m webapp.backend.run              # 默认 http://127.0.0.1:8000
    python -m webapp.backend.run --port 8080

API 一览::

    GET    /api/health                        健康检查 + 能力清单
    GET    /api/analyses                      可用分析项与中文标题
    GET    /api/files?dir=                    文件选择器（列目录 / 盘符）
    POST   /api/session                       打开体系 -> {sid, info, ...}
    GET    /api/session/{sid}/info            体系信息
    GET    /api/session/{sid}/chains          分子/链清单
    GET    /api/session/{sid}/components      组分清单
    POST   /api/session/{sid}/select          只设置选择，返回当前选择
    POST   /api/session/{sid}/run             跑分析 -> 全部曲线数据（JSON）
    POST   /api/session/{sid}/export          导出 CSV / PNG / Excel
    GET    /api/session/{sid}/outputs         列出已导出的文件
    GET    /api/session/{sid}/files/{name}    下载某个导出文件
    DELETE /api/session/{sid}                 关闭会话（释放并清理导出目录）

设计要点
--------
- **后端只给数据，不画图**：曲线以 ``x`` / ``y`` 数组返回，前端用 ECharts 渲染，
  缩放 / 平移 / hover / 图例开关都在前端完成。
- 分析在**线程池**里执行（MDAnalysis 是计算密集的同步代码），避免阻塞事件循环。
- 只监听 ``127.0.0.1``，供本机使用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------- 路径准备
_BACKEND = Path(__file__).resolve().parent
_WEBAPP = _BACKEND.parent
_ROOT = _WEBAPP.parent
for _p in (str(_ROOT), str(_WEBAPP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mdta import __version__                                     # noqa: E402
from mdta.pipeline import ANALYSIS_TITLES, DEFAULT_ORDER         # noqa: E402
from mdta.pipeline import analysis_groups                        # noqa: E402
from mdta.plotting import set_agg_backend                        # noqa: E402

from .session import (                                           # noqa: E402
    SessionError,
    SessionRegistry,
    list_dir,
)

# 服务端只把图**存成文件**，绝不弹窗；必须先切到 Agg，
# 否则默认的 TkAgg 后端会在工作线程里报警告甚至崩溃。
set_agg_backend()

#: 前端构建产物目录（``npm run build`` 生成）
DIST_DIR = _WEBAPP / "frontend" / "dist"
#: 导出结果的默认根目录
DEFAULT_OUTDIR = str(_ROOT / "webapp" / "outputs")

registry = SessionRegistry(DEFAULT_OUTDIR)

app = FastAPI(
    title="MD Trajectory Analysis Tool — Web API",
    version=__version__,
    description="分子动力学轨迹分析工具的 Web 后端（分析引擎复用 mdta 包）",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sess(sid: str):
    try:
        return registry.get(sid)
    except KeyError:
        raise HTTPException(404, f"会话不存在或已过期: {sid}") from None


def _fail(exc: Exception) -> HTTPException:
    return HTTPException(400, f"{type(exc).__name__}: {exc}")


# ==================================================================== 基础
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "version": __version__,
        "frontend_built": DIST_DIR.is_dir(),
        "outdir": DEFAULT_OUTDIR,
        "python": sys.version.split()[0],
    }


@app.get("/api/analyses")
def analyses() -> dict:
    return {
        "order": list(DEFAULT_ORDER),
        "titles": {k: ANALYSIS_TITLES.get(k, k) for k in DEFAULT_ORDER},
        # 按大纲模块分组（链构象 / 界面 / 结晶 / 辅助）：
        # 前端「分析功能」预设按钮与「图表导航」分层都用这一份，
        # 避免前后端各维护一份分组表而对不上。
        "groups": analysis_groups(),
        "sessions": registry.list(),
    }


@app.get("/api/files")
def files(dir: str | None = None) -> dict:
    try:
        return list_dir(dir)
    except SessionError as exc:
        raise _fail(exc) from exc


# ==================================================================== 会话
@app.post("/api/session")
def open_session(payload: Mapping[str, Any] = Body(...)) -> dict:
    top = str(payload.get("topology") or "").strip()
    traj = str(payload.get("trajectory") or "").strip() or None
    try:
        sess = registry.open(top, traj)
    except SessionError as exc:
        raise _fail(exc) from exc
    return {
        "sid": sess.sid,
        "info": sess.info(),
        "chains": sess.chains(),
        "components": sess.components(),
        "primary_label": sess.az.primary_label,
        "primary_atoms": int(sess.az.primary.n_atoms) if sess.az.primary else 0,
        "outdir": sess.outdir,
    }


@app.get("/api/session/{sid}/info")
def session_info(sid: str) -> dict:
    return _sess(sid).info()


@app.get("/api/session/{sid}/chains")
def session_chains(sid: str) -> dict:
    return {"chains": _sess(sid).chains()}


@app.get("/api/session/{sid}/components")
def session_components(sid: str) -> dict:
    return {"components": _sess(sid).components()}


@app.post("/api/session/{sid}/select")
def session_select(sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.apply_selection(payload)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/run")
async def session_run(sid: str, payload: Mapping[str, Any] = Body(...)) -> JSONResponse:
    """跑分析（**同步**等待全部完成）。放在线程池里执行，避免阻塞事件循环。

    前端用的是下面的 ``/run/start`` + ``/run/progress`` 实时接口；
    这个同步接口保留给脚本与测试使用。
    """
    sess = _sess(sid)
    from starlette.concurrency import run_in_threadpool

    try:
        data = await run_in_threadpool(sess.run, payload)
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"分析失败: {type(exc).__name__}: {exc}") from exc
    return JSONResponse(data)


# ---------------------------------------------------------------- 实时运行
@app.post("/api/session/{sid}/run/start")
def session_run_start(sid: str, payload: Mapping[str, Any] = Body(...)) -> dict:
    """开始跑分析并**立即返回**；之后用 ``GET /run/progress`` 拉增量结果。"""
    sess = _sess(sid)
    try:
        return sess.start_run(payload)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/run/progress")
def session_run_progress(sid: str, after: int = 0) -> dict:
    """运行状态 + 客户端还没收到的那几项结果（``after`` = 已有条数）。"""
    return _sess(sid).run_progress(after)


@app.post("/api/session/{sid}/run/cancel")
def session_run_cancel(sid: str) -> dict:
    """请求取消：当前这项算完后停下，已完成的项保留。"""
    return _sess(sid).cancel_run()


@app.post("/api/session/{sid}/export")
async def session_export(sid: str,
                         payload: Mapping[str, Any] = Body(default={})) -> dict:
    sess = _sess(sid)
    from starlette.concurrency import run_in_threadpool

    try:
        return await run_in_threadpool(sess.export, payload)
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"导出失败: {type(exc).__name__}: {exc}") from exc


@app.post("/api/session/{sid}/open-dir")
def session_open_dir(sid: str,
                     payload: Mapping[str, Any] = Body(default={})) -> dict:
    """在操作系统的文件管理器里打开目标目录（「打开目标目录」按钮）。"""
    sess = _sess(sid)
    try:
        return sess.open_directory(payload.get("dir"))
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/outputs")
def session_outputs(sid: str) -> dict:
    sess = _sess(sid)
    return {"dir": sess.outdir,
            "last_export": sess.last_export,
            "files": sess.list_output_files()}


@app.get("/api/session/{sid}/files/{name}")
def session_file(sid: str, name: str, dir: str | None = None):
    sess = _sess(sid)
    try:
        path = sess.output_file(name, outdir=dir)
    except SessionError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, filename=os.path.basename(path))


@app.delete("/api/session/{sid}")
def close_session(sid: str) -> dict:
    registry.close(sid)
    return {"closed": sid}


# ==================================================================== 前端
if DIST_DIR.is_dir():
    # 放在最后注册，保证 /api/* 优先匹配
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="ui")
else:
    @app.get("/")
    def _no_ui() -> dict:
        return {
            "message": "前端还没构建。请在 webapp/frontend 目录执行 "
                       "`npm install && npm run build`，或开发时用 `npm run dev`"
                       "（Vite 会把 /api 代理到本服务）。",
            "api_docs": "/docs",
            "frontend_dir": str(DIST_DIR),
        }
