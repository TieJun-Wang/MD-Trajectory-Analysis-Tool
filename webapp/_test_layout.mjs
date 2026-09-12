/**
 * 用真实浏览器验证布局（Headless Chrome + CDP）。
 *
 * 为什么必须有这个测试：说明栏"能不能上下滚动"取决于 flex 高度链在真实排版
 * 引擎里的**解析结果**，而"规则写在 CSS 里"和"规则真的作用到这个元素上"是两回事。
 * 这个 bug 连续漏过两次，两次都是因为只检查了 CSS 文本：
 *   `.notes > .notes-body` 这条规则一直在样式表里，但加了 .notes-clip 之后
 *   .notes-body 不再是 .notes 的直接子元素，选择器**匹配不上**，
 *   于是 display/flex/min-height 一条都没生效 —— 内容被裁掉且滚不动。
 *
 * 这里把说明栏的 DOM 原样搭出来、挂上**构建产物里的真 CSS**，然后量：
 *   - .notes-body 的 flex 尺寸是否真的生效（display:flex / min-height:0 / flex-grow>0）
 *   - .notes-scroll 是否真的能滚（scrollHeight > clientHeight 且 scrollTop 能变动）
 *   - 横向是否锁死（scrollWidth <= clientWidth）
 *   - 各类表格的单元格是否水平+垂直居中
 *
 * 运行前置条件：需要先起一个带调试端口的 Chrome。
 *   ⚠️ 受限沙箱里 Chrome 起不来（Mojo 需要创建命名管道，会以
 *      "platform_channel.cc Check failed 拒绝访问" 直接崩），
 *      必须用 danger-full-access 运行那条启动命令：
 *
 *   pwsh -Command "Start-Process 'C:\Program Files\Google\Chrome\Application\chrome.exe'
 *     -ArgumentList '--headless=new','--remote-debugging-port=9222',
 *     '--user-data-dir=C:\temp\MDT\webapp\_probe\chrome-profile','about:blank'"
 *
 * 用法:  node webapp/_test_layout.mjs
 * 没有可连的 Chrome 时会**跳过并返回 0**，不会把正常测试套件搞挂。
 */
import { readdirSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

const ROOT = resolve(new URL('..', import.meta.url).pathname.replace(/^\//, ''))
const DIST_ASSETS = join(ROOT, 'webapp', 'frontend', 'dist', 'assets')
const PROBE_DIR = join(ROOT, 'webapp', '_probe')
const PORT = process.env.MDTA_CDP_PORT || '9222'

let pass = 0
const fails = []
function ok(cond, label, extra = '') {
  if (cond) { pass += 1; console.log(`  ✓ ${label}${extra ? '  ' + extra : ''}`) }
  else { fails.push(label); console.log(`  ✗ ${label}${extra ? '  ' + extra : ''}`) }
}

console.log('='.repeat(74))
console.log('真实浏览器布局验证（Headless Chrome + CDP）')
console.log('='.repeat(74))

// ---------------------------------------------------------------- 前置检查
let target = null
try {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()
  target = list.find((t) => t.type === 'page')
} catch { /* 没起 Chrome */ }
if (!target?.webSocketDebuggerUrl) {
  console.log(`\n  跳过：连不上 Chrome 调试端口 ${PORT}。`)
  console.log('  说明见本文件顶部注释（受限沙箱下需要 danger-full-access 才能启动 Chrome）。')
  console.log(`\n通过 ${pass} 项，跳过 1 项（未连接浏览器）`)
  process.exit(0)
}

// ---------------------------------------------------------------- 探针页面
const cssFile = readdirSync(DIST_ASSETS).find((f) => f.endsWith('.css'))
if (!cssFile) throw new Error('找不到构建产物 CSS，先跑 webapp/frontend/_build.mjs')
const cssHref = 'file:///' + join(DIST_ASSETS, cssFile).replace(/\\/g, '/')

const NOTES = Array.from({ length: 14 }, (_, i) =>
  `<div class="noterow"><span class="notebullet"></span>`
  + `<span class="notetext">第 ${i + 1} 条说明：一段比较长的算法说明文字，`
  + `用来把说明栏撑高，验证它到底能不能上下滚动查看超出的内容。</span></div>`).join('\n')
const STATS = Array.from({ length: 26 }, (_, i) =>
  `<tr><th>统计量键名很长的第 ${i + 1} 项</th><td>${(19.6 + i / 100).toFixed(4)}</td></tr>`).join('\n')
const INFO = Array.from({ length: 12 }, (_, i) =>
  `<tr><td>项目 ${i + 1}</td><td>值 ${i + 1}</td></tr>`).join('\n')
const DATA = Array.from({ length: 40 }, (_, i) =>
  `<tr><td>${i * 100}</td><td>${(19 + i / 10).toFixed(3)}</td>`
  + `<td>${(19 + i / 10).toFixed(3)}</td><td>${i}</td></tr>`).join('\n')

const HTML = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<link rel="stylesheet" href="${cssHref}">
<style>.app{height:100%;display:flex;flex-direction:column;overflow:hidden}
.topbar{flex:0 0 56px;height:56px}</style>
</head><body><div id="root"><div class="app">
  <header class="topbar">顶栏</header>
  <div class="layout">
    <aside class="ops collapsed"><div class="acc"></div>
      <button class="rail rail-l"><span class="rail-text">展开操作台</span></button></aside>
    <main class="results"><nav class="tabs"></nav><div class="tabbody"></div></main>
    <aside class="notes">
      <div class="notes-clip"><div class="notes-body">
        <div class="notes-head"><div class="ntitle">回转半径 Rg</div>
          <div class="nsub">第 1 / 2 张</div></div>
        <div class="notes-scroll">
          <div class="sect-title">图表说明</div>
          <div class="notelist">${NOTES}</div>
          <div class="sect-title">关键统计量</div>
          <table class="stats notes-table"><thead><tr><th>统计量</th><th>数值</th></tr></thead>
            <tbody>${STATS}</tbody></table>
        </div>
        <div class="notes-foot dim small">完整统计量见「统计量 / 导出」页。</div>
      </div></div>
      <button class="rail rail-r"><span class="rail-text">收起说明</span></button>
    </aside>
  </div>
</div>
<!-- 其他会用到的表格类型，用于统一检查单元格对齐 -->
<div style="position:absolute;left:-9999px;top:0;width:600px">
  <table class="kv"><tbody><tr><th>原子数</th><td>47,681</td></tr>${INFO}</tbody></table>
  <table class="stats"><thead><tr><th>统计量</th><th>数值</th></tr></thead>
    <tbody>${STATS}</tbody></table>
  <table class="kv inforows"><thead><tr><th>项目</th><th>值</th></tr></thead>
    <tbody>${INFO}</tbody></table>
  <div class="tablewrap"><table class="datatable">
    <thead><tr><th>时间</th><th>Rg</th><th>Rg2</th><th>n</th></tr></thead>
    <tbody>${DATA}</tbody></table></div>
  <table class="stats filetable"><thead><tr><th>文件</th><th>大小</th><th>下载</th></tr></thead>
    <tbody><tr><td>rg.png</td><td>80 KB</td><td><a href="#">下载</a></td></tr></tbody></table>
</div>
</div></body></html>`

mkdirSync(PROBE_DIR, { recursive: true })
const probePath = join(PROBE_DIR, 'layout-probe.html')
writeFileSync(probePath, HTML, 'utf8')

// ---------------------------------------------------------------- CDP 连接
const ws = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((res, rej) => {
  ws.addEventListener('open', res, { once: true })
  ws.addEventListener('error', rej, { once: true })
})
let seq = 0
const pending = new Map()
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data)
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) }
})
const send = (method, params = {}) => new Promise((res) => {
  const id = ++seq
  pending.set(id, res)
  ws.send(JSON.stringify({ id, method, params }))
})
async function evaluate(expression) {
  const r = await send('Runtime.evaluate', {
    expression: `(() => { ${expression} })()`, returnByValue: true, awaitPromise: true,
  })
  if (r.result?.exceptionDetails) {
    throw new Error(r.result.exceptionDetails.exception?.description || 'evaluate 失败')
  }
  return r.result?.result?.value
}

await send('Page.enable')
await send('Runtime.enable')
await send('Page.navigate', { url: 'file:///' + probePath.replace(/\\/g, '/') })
for (let i = 0; i < 60; i += 1) {
  if (await evaluate('return document.readyState') === 'complete') break
  await new Promise((r) => setTimeout(r, 150))
}
await new Promise((r) => setTimeout(r, 250))

const M = await evaluate(`
  const g = (sel) => document.querySelector(sel)
  const cs = (el, p) => getComputedStyle(el)[p]
  const box = (sel) => { const el = g(sel); if (!el) return null
    return { clientH: el.clientHeight, scrollH: el.scrollHeight,
             clientW: el.clientWidth, scrollW: el.scrollWidth } }
  const scrollable = (sel) => { const el = g(sel); if (!el) return null
    const before = el.scrollTop
    el.scrollTop = 1e6
    const max = el.scrollTop
    el.scrollTop = before
    return { max, overflows: el.scrollHeight > el.clientHeight } }
  const body = g('.notes-body')
  const tables = {}
  for (const sel of ['.notes-table', '.kv:not(.inforows)', '.stats:not(.notes-table)',
                     '.inforows', '.datatable', '.filetable']) {
    const t = document.querySelectorAll(sel)[0] || document.querySelector(sel)
    if (!t) { tables[sel] = { missing: true }; continue }
    const cells = [...t.querySelectorAll('th,td')]
    const aligns = [...new Set(cells.map((c) => cs(c, 'textAlign')))]
    const valigns = [...new Set(cells.map((c) => cs(c, 'verticalAlign')))]
    tables[sel] = { n: cells.length, aligns, valigns }
  }
  return {
    bodyFlex: body ? { display: cs(body, 'display'), dir: cs(body, 'flexDirection'),
                       grow: cs(body, 'flexGrow'), minH: cs(body, 'minHeight'),
                       h: body.clientHeight } : null,
    noteFont: g('.noterow') ? cs(g('.noterow'), 'fontSize') : null,
    noteBullet: g('.notebullet') ? { w: cs(g('.notebullet'), 'width'),
                                     radius: cs(g('.notebullet'), 'borderRadius') } : null,
    notesScroll: box('.notes-scroll'),
    notesScrollable: scrollable('.notes-scroll'),
    clipH: g('.notes-clip')?.clientHeight,
    notesH: g('.notes')?.clientHeight,
    tables,
  }
`)

console.log('\n[1] 说明栏高度链真的生效（不只是 CSS 里写了规则）')
ok(M.bodyFlex?.display === 'flex', '.notes-body display=flex',
  `display=${M.bodyFlex?.display}`)
ok(M.bodyFlex?.grow === '1', '.notes-body flex-grow=1', `grow=${M.bodyFlex?.grow}`)
ok(M.bodyFlex?.minH === '0px', '.notes-body min-height=0',
  `min-height=${M.bodyFlex?.minH}`)
ok(M.bodyFlex?.h === M.notesH,
  '.notes-body 高度被约束在面板高度内（不是被内容撑高）',
  `body=${M.bodyFlex?.h} notes=${M.notesH}`)

console.log('\n[2] 说明栏能上下滚动')
ok(M.notesScroll?.scrollH > M.notesScroll?.clientH,
  '内容确实超出可视高度', `${M.notesScroll?.scrollH} > ${M.notesScroll?.clientH}`)
ok(M.notesScrollable?.max > 0, 'scrollTop 能真的往上滚（可查看跨页内容）',
  `maxScroll=${M.notesScrollable?.max}px`)
ok(M.notesScrollable?.overflows === true, 'overflow 判定为超出')

console.log('\n[3] 说明栏横向锁死（不出现左右滑动）')
ok(M.notesScroll?.scrollW <= M.notesScroll?.clientW,
  '横向无溢出', `scrollW=${M.notesScroll?.scrollW} clientW=${M.notesScroll?.clientW}`)

console.log('\n[3b] 说明文字字号 9.5px + 圆形项目符号')
ok(M.noteFont === '9.5px', '.noterow 字号 = 9.5px', `font-size=${M.noteFont}`)
ok(M.noteBullet?.radius === '50%', '项目符号是圆点',
  `${M.noteBullet?.w} ${M.noteBullet?.radius}`)

console.log('\n[4] 所有表格：单元格水平居中 + 垂直居中')
for (const [sel, info] of Object.entries(M.tables)) {
  if (info.missing) { ok(false, `${sel} 存在`, '未找到'); continue }
  const hOK = info.aligns.length === 1 && info.aligns[0] === 'center'
  const vOK = info.valigns.length === 1 && info.valigns[0] === 'middle'
  ok(hOK, `${sel} 水平居中`, `${info.n} 格  text-align=${info.aligns.join('/')}`)
  ok(vOK, `${sel} 垂直居中`, `vertical-align=${info.valigns.join('/')}`)
}

ws.close()
console.log('\n' + '='.repeat(74))
console.log(`通过 ${pass} 项` + (fails.length
  ? `，失败 ${fails.length} 项: ${fails.join(' | ')}` : '，全部通过'))
console.log('='.repeat(74))
process.exitCode = fails.length ? 1 : 0
