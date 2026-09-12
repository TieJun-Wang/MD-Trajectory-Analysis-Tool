/**
 * Web 版端到端验证（真实 HTTP，不经过 TestClient）。
 *
 * 用法: node webapp/_e2e.mjs [baseUrl]
 * 覆盖：静态前端资源、健康检查、文件浏览、打开体系、选择、跑全部分析、
 *       导出、下载导出文件、错误处理、关闭会话。
 */

const BASE = process.argv[2] || 'http://127.0.0.1:8000'
const ROOT = 'C:\\temp\\MDT'
const TOP = `${ROOT}\\adk_oplsaa.tpr`
const XTC = `${ROOT}\\adk_oplsaa.xtc`

let pass = 0
const fails = []

function ok(cond, label, extra = '') {
  if (cond) { pass += 1; console.log(`  ✓ ${label}${extra ? '  ' + extra : ''}`) }
  else { fails.push(label); console.log(`  ✗ ${label}${extra ? '  ' + extra : ''}`) }
}

async function req(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' }, ...options,
  })
  const text = await res.text()
  let data
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  return { status: res.status, data, headers: res.headers }
}

console.log('='.repeat(74))
console.log('Web 版端到端验证  ' + BASE)
console.log('='.repeat(74))

// ---------------------------------------------------------------- 静态资源
console.log('\n[1] 静态前端')
const home = await fetch(BASE + '/')
const html = await home.text()
ok(home.status === 200, 'GET / 返回首页', `HTTP ${home.status}`)
ok(/<div id="root">/.test(html), '首页含 React 挂载点')
const m = html.match(/src="(\/assets\/[^"]+\.js)"/)
ok(!!m, '首页引用打包后的 JS', m ? m[1] : '')
if (m) {
  const js = await fetch(BASE + m[1])
  const body = await js.text()
  ok(js.status === 200 && body.length > 100000,
    'JS 产物可下载', `${(body.length / 1024).toFixed(0)} KB`)
  ok(/echarts/i.test(body), 'JS 产物里包含 ECharts')
}
const cssMatch = html.match(/href="(\/assets\/[^"]+\.css)"/)
if (cssMatch) {
  const css = await fetch(BASE + cssMatch[1])
  ok(css.status === 200, 'CSS 产物可下载', `${(await css.text()).length} bytes`)
}

// ---------------------------------------------------------------- API
console.log('\n[2] 健康检查 / 分析项目录')
const health = await req('/api/health')
ok(health.status === 200 && health.data.ok, 'GET /api/health', JSON.stringify(health.data))
ok(health.data.frontend_built === true, '后端确认前端已构建')

const ana = await req('/api/analyses')
ok(ana.status === 200 && ana.data.order.length === 10, 'GET /api/analyses 返回 10 项',
  ana.data.order.join(','))

console.log('\n[3] 文件浏览')
const files = await req(`/api/files?dir=${encodeURIComponent(ROOT)}`)
const names = (files.data.files || []).map((f) => f.name)
ok(files.status === 200, 'GET /api/files', files.data.dir)
ok(names.includes('adk_oplsaa.tpr') && names.includes('adk_oplsaa.xtc'),
  '列出 tpr 与 xtc', names.join(', '))

console.log('\n[4] 打开体系')
const opened = await req('/api/session', {
  method: 'POST', body: JSON.stringify({ topology: TOP, trajectory: XTC }),
})
ok(opened.status === 200, 'POST /api/session', `sid=${opened.data?.sid}`)
const sid = opened.data.sid
const info = opened.data.info
ok(info.n_atoms === 47681 && info.n_frames === 10, '体系信息正确',
  `${info.n_atoms} 原子 / ${info.n_frames} 帧`)
ok(opened.data.components.length === 3, '识别到 3 个组分',
  opened.data.components.map((c) => `${c.name}(${c.n_atoms})`).join(' '))

console.log('\n[5] 选择')
const sel = await req(`/api/session/${sid}/select`, {
  method: 'POST',
  body: JSON.stringify({
    primary: { mode: 'component', name: 'protein' },
    components: ['protein', 'water'],
  }),
})
ok(sel.status === 200 && sel.data.components.length === 2, 'POST /select',
  sel.data.primary_label)

console.log('\n[6] 跑全部分析')
const t0 = Date.now()
const run = await req(`/api/session/${sid}/run`, {
  method: 'POST',
  body: JSON.stringify({
    which: ana.data.order,
    frames: { interval_ps: 300 },
    params: {
      density: { axis: 2, nbins: 60 },
      interface: { axis: 2, nbins: 60 },
      rdf: { rmax: 9, nbins: 90 },
      contact: { cutoff: 4 },
      dihedral: { mode: 'auto' },
    },
    selection: { primary: { mode: 'auto' }, components: ['protein', 'water'] },
  }),
})
ok(run.status === 200, 'POST /run', `${((Date.now() - t0) / 1000).toFixed(1)}s`)
const results = run.data.results
ok(Object.keys(results).length === 10, '返回 10 项分析结果')

// 逐个校验结构，并模拟前端"图表导航 + 单图"的取数逻辑
let nCharts = 0
let allGood = true
const problems = []
for (const [name, res] of Object.entries(results)) {
  const hasPanels = Array.isArray(res.panels) && res.panels.length > 0
  const curvesOk = res.curves.every((c) =>
    Array.isArray(c.x) && Array.isArray(c.y) && c.x.length === c.y.length
    && ['line', 'step', 'bar', 'scatter'].includes(c.kind)
    && Number.isInteger(c.panel))
  const panelRefOk = res.curves.every((c) => c.panel >= 0 && c.panel < res.panels.length)
  const xyFinite = res.curves.every((c) => c.x.every((v) => v === null || Number.isFinite(v)))
  nCharts += res.panels.length
  if (!(hasPanels && curvesOk && panelRefOk && xyFinite)) {
    allGood = false
    problems.push(`${name}: panels=${hasPanels} curves=${curvesOk} ref=${panelRefOk} finite=${xyFinite}`)
  }
}
ok(allGood, '每项结果的面板/曲线结构合法', problems.join('; '))
ok(nCharts === 23, '图总数为 23（与桌面版一致）', String(nCharts))

const rg = results.rg
ok(/回转半径/.test(rg.title), 'Rg 标题正确', rg.title.slice(0, 30))
ok(rg.panels.length === 2, 'Rg 有 2 个面板',
  rg.panels.map((p) => p.title).join(' / '))
const rgMean = rg.summary['Rg mean']
ok(Math.abs(rgMean - 19.6) < 1.0, 'Rg 平均值在合理区间', rgMean?.toFixed(4))

const msd = results.msd
ok(msd.panels[0].xscale === 'log' && msd.panels[0].yscale === 'log',
  'MSD 面板标记为对数轴（前端据此切换坐标轴）')

const di = results.dihedral
ok(di.curves.some((c) => c.kind === 'bar') && di.curves.some((c) => c.kind === 'line'),
  '二面角同时有分布(bar)与时间序列(line)')

console.log('\n[7] 导出')
const exp = await req(`/api/session/${sid}/export`, {
  method: 'POST',
  body: JSON.stringify({ formats: ['csv', 'png'], dpi: 150, panel_pngs: true, excel: true }),
})
ok(exp.status === 200, 'POST /export', `CSV ${exp.data.n_csv} / PNG ${exp.data.n_png}`)
ok(exp.data.files.length > 40, '导出文件清单', `${exp.data.files.length} 个文件`)
const png = exp.data.files.find((f) => f.name === 'rg.png')
ok(!!png, '存在 rg.png')
if (png) {
  const r = await fetch(BASE + png.url)
  const buf = Buffer.from(await r.arrayBuffer())
  ok(r.status === 200 && buf.length > 5000, '下载 rg.png',
    `${(buf.length / 1024).toFixed(0)} KB`)
  ok(buf[0] === 0x89 && buf.toString('ascii', 1, 4) === 'PNG', 'PNG magic 正确')
}
const csvFile = exp.data.files.find((f) => f.name.endsWith('.csv'))
if (csvFile) {
  const r = await fetch(BASE + csvFile.url)
  const txt = await r.text()
  ok(r.status === 200 && txt.split('\n').length > 2, '下载 CSV 并解析',
    `${csvFile.name}: ${txt.split('\n').length} 行`)
}

console.log('\n[8] 错误处理')
const e404 = await req('/api/session/notexist/info')
ok(e404.status === 404, '未知会话返回 404')
const e400 = await req('/api/session', {
  method: 'POST', body: JSON.stringify({ topology: `${ROOT}\\nope.tpr` }),
})
ok(e400.status === 400, '不存在的文件返回 400', String(e400.data.detail).slice(0, 40))
const ebad = await req(`/api/session/${sid}/run`, {
  method: 'POST', body: JSON.stringify({ which: ['bogus'] }),
})
ok(ebad.status === 400, '未知分析项返回 400')

console.log('\n[9] 关闭会话')
const closed = await req(`/api/session/${sid}`, { method: 'DELETE' })
ok(closed.status === 200, 'DELETE /api/session/{sid}')
const after = await req(`/api/session/${sid}/info`)
ok(after.status === 404, '关闭后无法再访问')

console.log('\n' + '='.repeat(74))
console.log(`通过 ${pass} 项` + (fails.length ? `，失败 ${fails.length} 项: ${fails.join(' | ')}` : '，全部通过'))
console.log('='.repeat(74))

// 不要用 process.exit()：fetch(undici) 的 keep-alive 套接字可能正在关闭，
// 此时强制退出会触发 libuv 断言（STATUS_STACK_BUFFER_OVERRUN），
// 使退出码变成 0xC0000409 —— 明明全部通过却报失败。
// 改为设置 exitCode 并让事件循环自然排空（undici 默认 4s 后关掉空闲连接）。
process.exitCode = fails.length ? 1 : 0
