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
  open: (topology, trajectory, progressToken, guessBonds = false,
         velocities = false, autodetect = false) =>
    request('/session', {
      method: 'POST',
      // progress_token 走**请求体**（后端从 payload 取）；前端在 POST 进行中
      // 并发轮询 /session/open/progress?token= 显示文件读取进度
      // guess_bonds=true：忽略拓扑里已有键、按坐标重猜（CONECT 不完整的 PDB）
      // ⚠️ 未勾选时发 **null**（= 用加载器的默认策略），不是 false：
      //    false 会让后端去做"键表残缺"体检（数碎片），.tpr 这类力场拓扑本来
      //    键表就是权威的，体检既慢（AdK 光这一步 2 s）又会在几十万原子的
      //    体系上吃光内存（用户实测卡在「读取」），还会把力场键表误判成残缺。
      // velocities=true：MS 的 .trj 转 TRR 时把速度一起存下来（XTC 装不下速度）
      // autodetect=true：**自动配对同名轨迹**（.xsd↔.xtc/.trj、.tpr↔.xtc…）。
      //   默认 false —— 只选了结构文件时按单帧结构打开，不偷偷带上轨迹
      body: JSON.stringify({ topology, trajectory: trajectory || null,
        progress_token: progressToken || null,
        guess_bonds: guessBonds === true ? true : null,
        velocities: velocities === true,
        autodetect: autodetect === true }),
    }),

  info: (sid) => request(`/session/${sid}/info`),
  chains: (sid) => request(`/session/${sid}/chains`),
  components: (sid) => request(`/session/${sid}/components`),
  /** 主链（分析对象）的原子名/元素/编号 —— R_ee 手动指定原子的悬浮提示用 */
  primaryAtoms: (sid) => request(`/session/${sid}/primary-atoms`),

  // ---------------------------------------------- 三维结构 / 交互式选择
  /**
   * 一帧的坐标 + 元素 + 可选键表 + 层级信息（三维视图画球用）。
   * `maxAtoms` 只控制**显示抽稀**（按整残基取，保住分子内化学键）；
   * 返回的 `index` 始终是原始原子编号 —— 提交选择时用它。
   */
  structure: (sid, frame = 0, maxAtoms = 20000) =>
    request(`/session/${sid}/structure?frame=${frame}&max_atoms=${maxAtoms}`),

  /** 结构层级树：体系 → 组分 → 链（segid+resname）→ 残基（分页） */  hierarchy: (sid, { segid = '', resname = '', offset = 0, limit = 400 } = {}) =>
    request(`/session/${sid}/hierarchy?segid=${encodeURIComponent(segid)}`
      + `&resname=${encodeURIComponent(resname)}&offset=${offset}&limit=${limit}`),

  /**
   * 把一次点击（或一个结构名）**展开成完整原子索引**。
   * 三维视图可能只画了部分原子（按整残基截断），而"选中整个残基/链/分子"
   * 必须完整 —— 所以这一步交后端用 Universe 算，与显示无关。
   */
  expand: (sid, { level = 'atom', index = null, segid = '', resname = '',
                  component = '' } = {}) => {
    const qs = [`level=${encodeURIComponent(level)}`]
    if (index !== null && index !== undefined) qs.push(`index=${index}`)
    if (segid) qs.push(`segid=${encodeURIComponent(segid)}`)
    if (resname) qs.push(`resname=${encodeURIComponent(resname)}`)
    if (component) qs.push(`component=${encodeURIComponent(component)}`)
    return request(`/session/${sid}/expand?${qs.join('&')}`)
  },

  /**
   * 按一条**集合定义**算出原子索引（只读，分析运行中也安全）。
   * 用来把「元素 C / 链状分子 / 含环分子 / 大分子」做成**可切换的选择条件**：
   * 点一下并进三维选择、再点一下取消 —— 而不是一点就自动建一个集合。
   */
  setAtoms: (sid, spec) =>
    request(`/session/${sid}/sets/atoms`, {
      method: 'POST', body: JSON.stringify({ spec }),
    }),

  /** 把三维视图里选中的**原子索引**变成一条集合（可直接参与分析）
   *  `from` = 这条集合是由什么选出来的（范围原子数 / 加了哪些筛选条件），
   *  只用于集合行的悬浮说明。 */
  createSelection: (sid, name, indices, from = null) =>
    request(`/session/${sid}/selection`, {
      method: 'POST', body: JSON.stringify({ name, indices, from }),
    }),

  /** 数据能力检测：坐标 / 多帧轨迹 / 键连接 / 质量 / 电荷 / 分子归属 / 周期盒
   *  （外加 mode：static / dynamic）。不靠文件后缀，直接看体系里有什么。 */
  capabilities: (sid, refresh = false) =>
    request(`/session/${sid}/capabilities?refresh=${refresh ? 1 : 0}`),

  /** 分析项可行性预判（界面据此把跑不了的项置灰 + 说明原因）。
   *  必须带上当前**帧选择** —— 把最大帧数设成 1 或时间窗收窄后，MSD 就跑不了。 */
  feasible: (sid, frames = null) =>
    request(`/session/${sid}/feasible`, {
      method: 'POST', body: JSON.stringify({ frames: frames || null }),
    }),

  /** 按残基名+编号（ETH32）/ 原子名 / 原子索引检索结构（命中即可高亮） */
  search: (sid, q, limit = 20) =>
    request(`/session/${sid}/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  /**
   * **MDAnalysis 原生选择语句**（select_atoms）→ 原子索引。
   * 例：`resid 1:10 and name CA`、`resname ETH and around 5.0 resname GRA`。
   * 语句写错时后端返回具体原因（前端直接把它显示在面板里）。
   */
  query: (sid, expr, limit = 400000) =>
    request(`/session/${sid}/query?expr=${encodeURIComponent(expr)}&limit=${limit}`),

  // ------------------------------------- 分子 / 聚合物链识别（结构层级）
  /** 按拓扑键图识别独立分子 / 聚合物链 / 网络组分（结果在后端缓存，只算一次） */
  recognition: (sid, refresh = false) =>
    request(`/session/${sid}/recognition?refresh=${refresh ? 1 : 0}`),

  /** 结构树展开到对象层：某分组（或某类别）下的分子或链，分页 */
  recognitionObjects: (sid, { group = null, section = '', offset = 0,
                              limit = 100 } = {}) => {
    const qs = [`offset=${offset}`, `limit=${limit}`]
    if (group !== null && group !== undefined) qs.push(`group=${group}`)
    if (section) qs.push(`section=${encodeURIComponent(section)}`)
    return request(`/session/${sid}/recognition/objects?${qs.join('&')}`)
  },

  /** 把识别树上的节点展开成**完整原子索引**（section / group / molecule / chain / all） */
  recognitionExpand: (sid, { objectType = 'all', objectId = '' } = {}) =>
    request(`/session/${sid}/recognition/expand?object_type=`
      + `${encodeURIComponent(objectType)}&object_id=${encodeURIComponent(objectId)}`),

  /** 单个分子 / 链的详情 */
  recognitionDetail: (sid, objectId, kind = 'molecules') =>
    request(`/session/${sid}/recognition/${kind}/${encodeURIComponent(objectId)}`),

  /** 人工确认（改名 / 改类型）—— 只影响显示，不动拓扑数据 */
  recognitionConfirm: (sid, payload) =>
    request(`/session/${sid}/recognition/confirm`,
      { method: 'POST', body: JSON.stringify(payload) }),

  /** 已保存的可视化选择 */
  selections: (sid) => request(`/session/${sid}/selections`),

  /** 数据表分页：只取一页（offset/limit），`only` = 只看这些曲线标签（可空） */
  table: (sid, { name, panel = 0, offset = 0, limit = 200, only = null } = {}) => {
    const q = new URLSearchParams({ name: String(name), panel: String(panel),
      offset: String(offset), limit: String(limit) })
    if (Array.isArray(only) && only.length) q.set('only', only.join(','))
    return request(`/session/${sid}/table?${q.toString()}`)
  },
  deleteSelection: (sid, name) =>
    request(`/session/${sid}/selections/${encodeURIComponent(name)}`,
      { method: 'DELETE' }),

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

  /**
   * **刷新页面后恢复现场**：体系信息 + 选择 + 帧选择 + 全部已完成结果 + 运行状态。
   * 分析跑在后端线程里，所以 F5 只断开前端连接，任务本身继续；恢复后接着轮询即可。
   */
  sessionState: (sid) => request(`/session/${sid}/state`),

  /** 清空已有结果（"清空结果重跑"按钮） */
  clearResults: (sid) =>
    request(`/session/${sid}/clear`, { method: 'POST' }),

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

/** 自动匹配同名轨迹文件（选完 tpr 后找同目录的 xtc/trr/dcd…） */
export function guessTrajectory(topologyPath, files) {
  if (!topologyPath) return null
  const stem = topologyPath.replace(/\.[^.]+$/, '')
  const exts = ['.xtc', '.trr', '.dcd', '.nc', '.ncdf', '.trj', '.trz',
    '.tng', '.lammpstrj', '.dump', '.lammpsdump', '.mdcrd', '.h5md', '.gsd']
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
