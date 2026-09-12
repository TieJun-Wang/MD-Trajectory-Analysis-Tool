/**
 * 与后端 REST API 的通信层。
 *
 * 开发时由 Vite 把 `/api` 代理到 FastAPI（127.0.0.1:8000）；
 * 发布时前端由 FastAPI 托管，同源访问 `/api`，因此这里统一用相对路径。
 */

const BASE = '/api'

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

export const api = {
  health: () => request('/health'),
  analyses: () => request('/analyses'),

  /** 文件选择器：列出目录下的子目录与 MD 相关文件 */
  files: (dir) => request(`/files${dir ? `?dir=${encodeURIComponent(dir)}` : ''}`),

  /** 打开体系（读取 tpr/xtc），返回 sid 与体系信息 */
  open: (topology, trajectory) =>
    request('/session', {
      method: 'POST',
      body: JSON.stringify({ topology, trajectory: trajectory || null }),
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
