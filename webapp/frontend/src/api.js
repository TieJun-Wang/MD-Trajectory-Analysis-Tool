/**
 * 与后端 REST API 的通信层。
 *
 * 开发时由 Vite 把 `/api` 代理到 FastAPI（127.0.0.1:8000）；
 * 发布时前端由 FastAPI 托管，同源访问 `/api`，因此这里统一用相对路径。
 */

const BASE = '/api'

/** 「文件读取」进度的 SSE 地址（EventSource 只能 GET，故单独开流） */
export const openStreamUrl = (token) =>
  `${BASE}/session/open/stream?token=${encodeURIComponent(token)}`

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { detail: text }
  }
  if (!res.ok) {
    const msg = data?.detail || `${res.status} ${res.statusText}`
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return data
}

/**
 * **流式**订阅「文件读取」进度：服务端在阶段变化时立即推送（SSE），
 * 不必轮询；流断了或环境不支持 EventSource 时自动退回 400ms 轮询，
 * 保证任何情况下都有反馈。本地秒表则一直跑，避免"看起来卡死"。
 * 返回 ``stop()``，打开结束时调用。
 */
export function streamOpenProgress(token, { onStage, onTick } = {}) {
  let es = null
  let timer = null
  let stopped = false
  const startPoll = () => {
    if (stopped || timer) return
    timer = setInterval(async () => {
      onTick?.()
      try {
        const p = await api.openProgress(token)
        if (p && !p.unknown) onStage?.(p)
      } catch { /* 轮询失败不打断打开流程 */ }
    }, 400)
  }
  const stop = () => {
    stopped = true
    if (es) { try { es.close() } catch { /* ignore */ } es = null }
    if (timer) { clearInterval(timer); timer = null }
  }
  timer = setInterval(() => onTick?.(), 400)     // 本地秒表（两条路径都用）
  if (typeof EventSource === 'undefined') { startPoll(); return stop }
  try {
    es = new EventSource(openStreamUrl(token))
    es.onmessage = (ev) => {
      try { onStage?.(JSON.parse(ev.data)) } catch { /* 忽略坏消息 */ }
    }
    es.addEventListener('end', () => stop())
    es.onerror = () => {                          // 流断了 → 退回轮询
      if (stopped) return
      if (es) { try { es.close() } catch { /* ignore */ } es = null }
      startPoll()
    }
  } catch {
    startPoll()
  }
  return stop
}

export const api = {
  health: () => request('/health'),
  analyses: () => request('/analyses'),

  /** 文件选择器：列出目录下的子目录与 MD 相关文件 */
  files: (dir) => request(`/files${dir ? `?dir=${encodeURIComponent(dir)}` : ''}`),

  /** 打开体系（读取 tpr/xtc），返回 sid 与体系信息 */
  open: (topology, trajectory, progressToken) =>
    request('/session', {
      method: 'POST',
      // progress_token 走**请求体**（后端从 payload 取）；前端在 POST 进行中
      // 并发轮询 /session/open/progress?token= 显示文件读取进度
      body: JSON.stringify({ topology, trajectory: trajectory || null,
        progress_token: progressToken || null }),
    }),

  info: (sid) => request(`/session/${sid}/info`),
  chains: (sid) => request(`/session/${sid}/chains`),
  components: (sid) => request(`/session/${sid}/components`),

  select: (sid, payload) =>
    request(`/session/${sid}/select`, { method: 'POST', body: JSON.stringify(payload) }),

  /** 跑分析：后端返回全部曲线数据（JSON），前端用 ECharts 画 */
  run: (sid, payload) =>
    request(`/session/${sid}/run`, { method: 'POST', body: JSON.stringify(payload) }),

  /**
   * 实时跑分析：立即返回，随后用 runProgress 轮询增量结果。
   * 后台每算完一项就写入结果集，因此页面可以边算边显示。
   */
  runStart: (sid, payload) =>
    request(`/session/${sid}/run/start`, { method: 'POST', body: JSON.stringify(payload) }),

  /** 拉取运行状态 + 客户端尚未收到的结果（after = 已收到的条数） */
  runProgress: (sid, after = 0) =>
    request(`/session/${sid}/run/progress?after=${after}`),

  /** 「文件读取」进度（打开文件时并发轮询，token 由前端生成） */
  openProgress: (token) =>
    request(`/session/open/progress?token=${encodeURIComponent(token)}`),

  /** 请求取消：当前这项算完即停，已完成的结果保留 */
  runCancel: (sid) =>
    request(`/session/${sid}/run/cancel`, { method: 'POST' }),

  /** 导出 CSV / PNG / Excel 到服务器磁盘（可指定目录、模块、格式） */
  export: (sid, payload) =>
    request(`/session/${sid}/export`, { method: 'POST', body: JSON.stringify(payload) }),

  /** 在操作系统的文件管理器里打开目标目录 */
  openDir: (sid, dir) =>
    request(`/session/${sid}/open-dir`, {
      method: 'POST', body: JSON.stringify({ dir: dir || null }),
    }),

  outputs: (sid) => request(`/session/${sid}/outputs`),

  close: (sid) => request(`/session/${sid}`, { method: 'DELETE' }),
}

/** 自动匹配同名轨迹文件（选完 tpr 后找同目录的 xtc/trr/dcd） */
export function guessTrajectory(topologyPath, files) {
  if (!topologyPath) return null
  const stem = topologyPath.replace(/\.[^.]+$/, '')
  const exts = ['.xtc', '.trr', '.dcd', '.nc', '.trj']
  for (const ext of exts) {
    const hit = files.find((f) => f.path === stem + ext)
    if (hit) return hit.path
  }
  return null
}
