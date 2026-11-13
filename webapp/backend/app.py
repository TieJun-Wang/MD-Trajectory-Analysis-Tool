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

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                                StreamingResponse)
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
    _available_elements,
    _set_presets,
    list_dir,
    supported_formats,
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
    description="分子动力学轨迹分析工具Web后端",
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


@app.exception_handler(Exception)
async def _unhandled(request, exc: Exception):
    """兜底：**任何**没被捕获的异常都回一个 JSON 400/500，而不是 HTML 错误页。

    为什么需要：前端 `request()` 直接 `JSON.parse` 响应体，而 Starlette 默认的
    未处理异常会返回 **HTML** 错误页 —— 实测"把 .xtc 放进拓扑槽""拿 .md 当拓扑"
    都会走到这里，页面只能显示一句 "Unexpected token '<'"，完全看不出发生了什么。
    现在至少能把异常类型与消息带回界面。
    """
    from fastapi.responses import JSONResponse

    detail = f"{type(exc).__name__}: {exc}"
    return JSONResponse(
        status_code=400,
        content={"detail": detail,
                 "hint": "请求本身可以重试；若与文件格式相关，请确认后缀在支持列表内"
                         "（见 GET /api/formats）"},
    )


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
        "order": list(ANALYSIS_TITLES),
        "titles": {k: ANALYSIS_TITLES.get(k, k) for k in ANALYSIS_TITLES},
        # 按大纲模块分组（链构象 / 空间结构 / 取向与结晶 / 动力学与输运）：
        # 前端「分析功能」预设按钮与「图表导航」分层都用这一份，
        # 避免前后端各维护一份分组表而对不上。
        "groups": analysis_groups(),
        "sessions": registry.list(),
    }


@app.get("/api/formats")
def formats() -> dict:
    """支持的文件格式（后缀白名单的唯一来源是 `mdta.formats`）。

    前端用它渲染提示文案与"这个文件能放哪个槽"的判断，避免前后端各写一份。
    """
    return supported_formats()


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
    # 前端会带一个 token 并**并发轮询** /api/session/open/progress 显示读取进度
    token = str(payload.get("progress_token") or "").strip() or None
    # `guess_bonds=True`：忽略拓扑里已有的键、按坐标重新猜（CONECT 不完整的 PDB，
    # 例如 Material Studio 导出的文件）
    gb = payload.get("guess_bonds")
    guess_bonds = bool(gb) if gb is not None else None
    # `velocities=True`：读 Materials Studio 的 .trj 时**连带保留速度**
    # （转成 TRR；XTC 格式装不下速度，文件约为 XTC 的 5~6 倍）
    vel = payload.get("velocities")
    # `autodetect`：是否**自动配对同名轨迹**。Web 端默认 **关**：用户只选了
    # 一个结构文件时不该悄悄把同目录的轨迹也读进来（那样"只看结构"会变成
    # 几千帧的轨迹分析）。想要自动配对就在界面上勾「自动配对同名轨迹」。
    auto = payload.get("autodetect")
    autodetect = bool(auto) if auto is not None else False
    velocities = bool(vel) if vel is not None else None
    try:
        sess = registry.open(top, traj, progress_token=token,
                             guess_bonds=guess_bonds, velocities=velocities,
                             autodetect=autodetect)
    except SessionError as exc:
        raise _fail(exc) from exc
    return {
        "sid": sess.sid,
        "info": sess.info(),
        "chains": sess.chains(),
        # 候选成分清单（全部自动组分 + 已定义集合，带 active 标记）——
        # 界面上的组分胶囊与「主链」下拉框读它，取消勾选不会让条目消失
        "catalog": sess.catalog(),
        "components": sess.components(),
        # 已保存的可视化选择（三维视图里存下来的）
        "selections": sess.selection_list(),
        "primary_label": sess.az.primary_label,
        "primary_atoms": int(sess.az.primary.n_atoms) if sess.az.primary else 0,
        "outdir": sess.outdir,
        # 轨迹里有没有速度（MS .trj 用 TRR 模式转换后为 True）
        "has_velocities": bool(getattr(sess.az.trajectory, "has_velocities", False)),
        # 加载诊断（猜键 / 分子划分 / 偏移缓存重建），前端写进「运行日志」
        "load_notes": sess.load_notes(),
        # 「分析集合（Set）」用：可选元素 + 一键预设（只列能选中的）
        "elements": _available_elements(sess.az.universe),
        "set_presets": _set_presets(sess.az.universe, sess.az.components),
    }


@app.get("/api/session/open/progress")
def open_progress(token: str = "") -> dict:
    """「文件读取」进度（轮询版）：阶段、逐阶段耗时、总耗时。"""
    return registry.open_progress(token)


@app.get("/api/session/open/stream")
async def open_stream(token: str = "") -> StreamingResponse:
    """「文件读取」进度的**流式**推送（SSE）。

    为什么用流式：读取大轨迹首次要扫全文件（46 体系 1.7 GB 实测 >135 s），
    轮询只能"定期取样"，阶段切换最多晚一个轮询周期；SSE 在阶段变化时**立即**
    推送，客户端不必反复发请求。结束/出错时发一个 ``event: end`` 让前端收流。

    实现要点：本接口是 ``async``，而 ``registry.open_progress()`` 是同步的
    （内部有锁，耗时是微秒级），在事件循环里直接调用不会造成阻塞。
    """
    import asyncio
    import json
    import time as _time

    async def gen():
        last = None
        beat = _time.time()
        hard_deadline = _time.time() + 3600.0        # 最长 1 小时，避免僵尸连接
        grace_until = _time.time() + 15.0            # 等 token 注册（见下）
        while True:
            snap = registry.open_progress(token)
            # 前端通常**先开流、再 POST**，所以刚开始查不到 token 是正常的
            # （实测：不等就会立刻收流，前端永远看不到进度）。这里等它出现，
            # 超过宽限期才当作"未知 token"收流。
            if snap.get("unknown") and _time.time() < grace_until:
                await asyncio.sleep(0.2)
                continue
            key = (snap.get("n_stages"), snap.get("done"), bool(snap.get("error")))
            now = _time.time()
            if key != last or now - beat >= 1.0:      # 变化即时推；否则每秒一次心跳
                last, beat = key, now
                yield "data: " + json.dumps(snap, ensure_ascii=False) + "\n\n"
            if snap.get("unknown") or snap.get("done") or snap.get("error"):
                yield "event: end\ndata: {}\n\n"
                return
            if now > hard_deadline:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.12)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.get("/api/session/{sid}/info")
def session_info(sid: str) -> dict:
    return _sess(sid).info()


@app.get("/api/session/{sid}/chains")
def session_chains(sid: str) -> dict:
    return {"chains": _sess(sid).chains()}


@app.get("/api/session/{sid}/components")
def session_components(sid: str) -> dict:
    sess = _sess(sid)
    return {"components": sess.components(), "catalog": sess.catalog()}


@app.get("/api/session/{sid}/primary-atoms")
def session_primary_atoms(sid: str) -> dict:
    """主链（分析对象）的原子名 / 元素 / 全局编号 —— R_ee 指定原子的悬浮提示用。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.primary_atoms_info()
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:                                     # noqa: BLE001
        raise _fail(SessionError(f"取主链原子清单失败：{type(exc).__name__}: {exc}")) from exc


@app.post("/api/session/{sid}/select")
def session_select(sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.apply_selection(payload)
    except SessionError as exc:
        raise _fail(exc) from exc


# ------------------------------------------------ 三维结构（可视化选择用）
@app.get("/api/session/{sid}/structure")
def session_structure(sid: str, frame: int = 0, max_atoms: int = 20000) -> dict:
    """一帧的原子坐标 / 元素 / 可选键表 / 层级信息（三维视图画球用）。

    ``max_atoms`` 只控制**显示抽稀**：返回的 ``index`` 始终是原始原子编号，
    前端提交选择时用它 —— 抽稀与隐藏都不改变编号。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.structure(frame=frame, max_atoms=max_atoms)
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:                                     # noqa: BLE001
        raise _fail(SessionError(f"取三维结构失败：{type(exc).__name__}: {exc}")) from exc


@app.get("/api/session/{sid}/hierarchy")
def session_hierarchy(sid: str, segid: str = "", resname: str = "",
                      offset: int = 0, limit: int = 400) -> dict:
    """结构层级树：体系 → 组分 → 链（segid+resname）→ 残基（分页）。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.hierarchy(segid=segid or None, resname=resname or None,
                                  offset=offset, limit=limit)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/search")
def session_search(sid: str, q: str = "", limit: int = 20) -> dict:
    """按残基名+编号（``ETH32``）、原子名或原子索引检索，命中即可在三维视图高亮。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.search(q, limit=limit)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/query")
def session_query(sid: str, expr: str = "", limit: int = 400000) -> dict:
    """**MDAnalysis 原生选择语句**（``select_atoms``）→ 原子索引。

    例：``resid 1:10 and name CA``、``resname ETH and around 5.0 resname GRA``。
    语句写错时返回 400 + 具体原因（不让前端看栈）。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.query(expr, limit=limit)
    except SessionError as exc:
        raise _fail(exc) from exc


# ------------------------------------------- 分子 / 聚合物链识别（结构层级）
@app.get("/api/session/{sid}/capabilities")
def session_capabilities(sid: str, refresh: bool = False) -> dict:
    """数据能力检测：坐标 / 多帧轨迹 / 键连接 / 质量 / 电荷 / 分子归属 / 周期盒。

    另外给出 `mode`：`static`（只单帧结构，只能做结构/空间分布/构象统计）或
    `dynamic`（有轨迹，可做时间演化与动力学）。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.capabilities(refresh=bool(refresh))
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/feasible")
def session_feasible(sid: str) -> dict:
    """分析项可行性预判：`{项: {ok, level, reason}}`（界面据此置灰 + 提示）。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return {"feasible": sess.feasible()}
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/feasible")
def session_feasible_post(sid: str,
                          payload: Mapping[str, Any] = Body(default={})) -> dict:
    """同上，但带上界面当前的**帧选择**（`{"frames": {...}}`）。

    必须带：把最大帧数设成 1、或把时间窗收窄到只剩 1 帧时，MSD 就跑不了 ——
    不带帧选择的话预判会以为"体系有 N 帧就能跑"，然后一跑就报
    「MSD 至少需要 2 帧」（用户实测）。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return {"feasible": sess.feasible(payload.get("frames") or None)}
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/recognition")
def session_recognition(sid: str, refresh: bool = False) -> dict:
    """按**拓扑键图**识别独立分子 / 聚合物链 / 网络组分，返回摘要与结构树。

    `refresh=1` 强制重算；不传则读会话缓存（识别只跑一次）。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition(refresh=bool(refresh))
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:                                     # noqa: BLE001
        raise _fail(SessionError(f"识别失败：{type(exc).__name__}: {exc}")) from exc


@app.get("/api/session/{sid}/recognition/objects")
def session_recognition_objects(sid: str, group: int | None = None,
                                section: str = "", offset: int = 0,
                                limit: int = 100) -> dict:
    """结构树展开到对象层：某个分组 / 某类别下的分子或链（分页）。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition_objects(group=group, section=section,
                                            offset=offset, limit=limit)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/recognition/expand")
def session_recognition_expand(sid: str, object_type: str = "all",
                               object_id: str = "") -> dict:
    """把结构树节点展开成**完整原子索引**（`section` / `group` / `molecule` / `chain` / `all`）。

    例：选择某条链的 id → 返回这条链的全部原子索引（原始 Universe 编号）。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition_expand(object_type, object_id)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/recognition/expand")
def session_recognition_expand_post(
        sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    """同上，POST 形式（recognize.md §5.1 的写法）。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition_expand(str(payload.get("object_type") or "all"),
                                           str(payload.get("object_id") or ""))
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/recognition/molecules/{object_id}")
def session_recognition_molecule(sid: str, object_id: str) -> dict:
    """单个分子的详情（含全部原子索引）。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition_detail(object_id)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/recognition/chains/{object_id}")
def session_recognition_chain(sid: str, object_id: str) -> dict:
    """单条聚合物链的详情（含全部原子索引）。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition_detail(object_id)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/recognition/confirm")
def session_recognition_confirm(sid: str,
                                payload: Mapping[str, Any] = Body(default={})) -> dict:
    """人工确认：改名 / 改类型 / 标记未识别。**只影响显示，不动拓扑数据**。"""
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.recognition_confirm(payload)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/sets/atoms")
def session_set_atoms(sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    """按一条集合定义算出**原子索引**（只读，分析运行中也安全）。

    界面用它把「元素 C / 链状分子 / 含环分子 / 大分子」做成**可切换的选择条件**：
    点一下并进三维选择，再点一下取消；要留成分析用的集合再点「＋ 添加集合」。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.set_atoms(payload.get("spec") or payload)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/expand")
def session_expand(sid: str, level: str = "atom", index: int | None = None,
                   segid: str = "", resname: str = "",
                   component: str = "") -> dict:
    """把一次点击（或一个结构名）**展开成完整原子索引**（按原始编号）。

    三维视图为性能可能只画了部分原子（按整残基截断），但"选中整个残基/链/分子"
    必须完整 —— 所以展开由后端用 Universe 直接算，与显示无关。
    """
    sess = _sess(sid)
    try:
        with sess.lock:
            return sess.expand(level, index=index, segid=segid or None,
                               resname=resname or None,
                               component=component or None)
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/selection")
def session_selection(sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    """把三维视图里选中的**原子索引**变成一条集合（可直接参与后续分析）。

    请求：``{"name": "我选的碳链", "indices": [0, 5, 7, ...]}``
    （给了 ``name`` 但没给 ``indices`` ＝ 加载同名的已保存选择）
    """
    sess = _sess(sid)
    name = str(payload.get("name") or "").strip()
    raw = payload.get("indices")
    try:
        if raw is None:
            return sess.load_selection(name)
        if not isinstance(raw, (list, tuple)):
            raise SessionError("indices 必须是原子索引数组")
        return sess.add_visual_selection(name, raw,
                                         from_meta=payload.get("from"))
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/pick")
def session_pick(sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    """三维视图手势 → 一批原子索引。

    ``{"index": 123, "kind": "chain"}``；``kind`` ∈ ``monomer`` / ``chain`` /
    ``same_type``（双击取单体、右击取所在链、再右击取同类全部链）。
    """
    sess = _sess(sid)
    raw = payload.get("index")
    if raw is None:
        raise _fail(SessionError("缺少 index"))
    try:
        return sess.pick_atom(int(raw), str(payload.get("kind") or "chain"))
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/set-rename")
def session_set_rename(sid: str, payload: Mapping[str, Any] = Body(default={})) -> dict:
    """给已定义的分析集合改名（``{"old": "A", "new": "B"}``）。

    新名字与别的集合重名时**自动加序号**（``B_2``、``B_3``…），返回值里的
    ``name`` 是最终生效的名字、``renamed`` 表示是否加过序号。
    """
    sess = _sess(sid)
    try:
        return sess.rename_set(payload.get("old"), payload.get("new"))
    except SessionError as exc:
        raise _fail(exc) from exc


@app.get("/api/session/{sid}/selections")
def session_selections(sid: str) -> dict:
    """已保存的可视化选择（名字 / 原子数 / 类型）。"""
    return {"selections": _sess(sid).selection_list()}


@app.delete("/api/session/{sid}/selections/{name}")
def session_selection_delete(sid: str, name: str) -> dict:
    sess = _sess(sid)
    try:
        return sess.delete_selection(name)
    except SessionError as exc:
        raise _fail(exc) from exc


def _only_labels(only) -> list[str] | None:
    """把 ``only`` 查询参数整理成标签列表。

    **必须支持"重复参数"**（``?only=A&only=B``），且**绝不能按逗号再切**：
    曲线标签自身就含逗号（MSD 的 ``polymer (原子, 3,640 个)``）。旧实现是前端
    ``only.join(',')`` 拼成一串、后端 ``split(',')`` 拆开 —— 标签被切碎后一条也
    匹配不上，于是"只勾一个图例 + 只看当前图例"静默退回全部曲线（用户实测的 bug）。

    现在一个参数就是**一条完整标签**；老客户端若是自己拼了逗号，会得到一条匹配
    不上的标签，界面会明确提示"没匹配到曲线"，而不是悄悄显示全部。
    """
    if only is None:
        return None
    vals = [str(x) for x in only] if isinstance(only, (list, tuple)) else [str(only)]
    vals = [v for v in vals if v != ""]
    return vals or None


@app.get("/api/session/{sid}/table.csv")
def session_table_csv(sid: str, name: str, panel: int = 0,
                      only: list[str] | None = Query(default=None)):
    """整表 CSV（不分页）：浏览器直接下载，行数再多也不会撑爆页面。"""
    sess = _sess(sid)
    labels = _only_labels(only)
    try:
        with sess.lock:
            text = sess.table_csv(name, panel=panel, only=labels)
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:                                     # noqa: BLE001
        raise _fail(SessionError(f"导出数据表失败：{type(exc).__name__}: {exc}")) from exc
    # 文件名用**面板标题**（用户要求：不要再有 panel0/panel1）
    fname = sess.table_filename(name, panel)
    return PlainTextResponse(
        text, media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/session/{sid}/table")
def session_table(sid: str, name: str, panel: int = 0, offset: int = 0,
                  limit: int = 200, only: list[str] | None = Query(default=None)) -> dict:
    """数据表分页（一页一传）：``only`` 是**重复出现的**曲线标签（只看这些）。"""
    sess = _sess(sid)
    labels = _only_labels(only)
    try:
        with sess.lock:
            return sess.table_slice(name, panel=panel, offset=offset, limit=limit,
                                    only=labels)
    except SessionError as exc:
        raise _fail(exc) from exc
    except Exception as exc:                                     # noqa: BLE001
        raise _fail(SessionError(f"取数据表失败：{type(exc).__name__}: {exc}")) from exc


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
@app.get("/api/session/{sid}/state")
def session_state(sid: str) -> dict:
    """**刷新页面后恢复现场**：体系信息 + 选择 + 帧选择 + 全部结果 + 运行状态。

    分析跑在后端线程里，F5 只是前端断开连接 —— 所以这里把"当前有什么"
    一次性交给新的页面实例，正在跑的任务继续跑，前端接着轮询进度。
    """
    sess = _sess(sid)
    try:
        return sess.state()
    except SessionError as exc:
        raise _fail(exc) from exc


@app.post("/api/session/{sid}/clear")
def session_clear(sid: str) -> dict:
    """清空已有结果（界面上"清空结果重跑"）。"""
    sess = _sess(sid)
    try:
        return sess.clear_results()
    except SessionError as exc:
        raise _fail(exc) from exc


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
