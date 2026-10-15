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

  /** 打开体系（读取拓扑/轨迹），返回 sid 与体系信息（含 load_notes 加载诊断） */
  open: (topology, trajectory, progressToken, guessBonds = false) =>
    request('/session', {
      method: 'POST',
      // progress_token 走**请求体**（后端从 payload 取）；前端在 POST 进行中
      // 并发轮询 /session/open/progress?token= 显示文件读取进度
      // guess_bonds=true：忽略拓扑里已有键、按坐标重猜（CONECT 不完整的 PDB）
      body: JSON.stringify({ topology, trajectory: trajectory || null,
        progress_token: progressToken || null,
        guess_bonds: guessBonds === true }),
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
  const exts = ['.xtc', '.trr', '.dcd', '.nc', '.ncdf', '.trj', '.trz',
    '.tng', '.lammpstrj', '.dump', '.mdcrd', '.h5md', '.gsd']
  for (const ext of exts) {
    const hit = files.find((f) => f.path === stem + ext)
    if (hit) return hit.path
  }
  return null
}

/**
 * 一个文件能用在哪几个槽：`['topology']` / `['trajectory']` / 两者都有。
 *
 * 兼容三种后端写法（避免"新旧版本打架"）：
 *   · 新：`roles: ['topology','trajectory']`
 *   · 过渡版：`kind === 'both'`
 *   · 老：`kind === 'topology' | 'trajectory'`
 * 用户实际踩过的坑：后端先返回了 `kind="both"`，而**老前端**只认
 * `kind === 'topology'`，于是 `.pdb` 掉进了 else 分支、被当成**轨迹** →
 * 现象就是"pdb 明明是拓扑，怎么读到轨迹槽去了"。所以这里必须容错。
 */
export function fileRoles(file) {
  const r = file?.roles
  if (Array.isArray(r) && r.length) return r.map(String)
  if (file?.kind === 'both') return ['topology', 'trajectory']
  if (file?.kind === 'trajectory') return ['trajectory']
  return ['topology']
}

/**
 * 点击文件行时该填哪个槽 —— 纯函数，便于自检（之前这段逻辑埋在组件里没法测）。
 *
 * 规则（按用户预期定）：**能给拓扑就一定进拓扑槽**。`.pdb` / `.gro` 本来就是结构
 * 文件，哪怕两个槽都空、哪怕拓扑槽已满，主点击也不该把它塞进轨迹槽。
 * 想让它当轨迹，只能点行右侧的「轨迹」小按钮（OpenPanel 的 useAs）。
 * 返回 `{role, next}`，`next` 是合并后的 `{topology, trajectory}`。
 */
export function assignFile(file, cur = {}) {
  const roles = fileRoles(file)
  const role = roles.includes('topology') ? 'topology' : 'trajectory'
  return { role, next: { ...cur, [role]: file?.path || '' } }
}
