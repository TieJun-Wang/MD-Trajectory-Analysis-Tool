/**
 * 用真实高分子/两组分体系验证 Web 版（md_biopolymer_nowater）。
 *
 * 用法: node webapp/_e2e_polymer.mjs [baseUrl]
 */
const BASE = process.argv[2] || 'http://127.0.0.1:8000'
const ROOT = 'C:\\temp\\MDT'
const TOP = `${ROOT}\\md_biopolymer_nowater.tpr`
const XTC = `${ROOT}\\md_biopolymer_nowater.xtc`

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
  return { status: res.status, data }
}

console.log('='.repeat(76))
console.log('Web 版 —— 真实两组分体系验证（md_biopolymer_nowater）')
console.log('='.repeat(76))

console.log('\n[1] 打开体系')
const opened = await req('/api/session', {
  method: 'POST', body: JSON.stringify({ topology: TOP, trajectory: XTC }),
})
ok(opened.status === 200, 'POST /api/session', `sid=${opened.data?.sid}`)
if (opened.status !== 200) { console.log(opened.data); process.exit(1) }
const sid = opened.data.sid
const info = opened.data.info

ok(info.n_atoms === 24616, '原子数 24,616', String(info.n_atoms))
ok(info.n_frames === 301, '帧数 301', String(info.n_frames))
ok(Math.abs(info.dt_ps - 1000) < 1, '时间间隔 1000 ps', String(info.dt_ps))
ok(Math.abs(info.total_time_ps / 1000 - 300) < 1, '总时长 300 ns',
  `${(info.total_time_ps / 1000).toFixed(1)} ns`)
ok(info.box_type === 'orthorhombic', '正交盒', info.box_type)
ok(info.n_residues === 853, 'residue 数 853', String(info.n_residues))

console.log('\n[2] 组分与链识别')
const comps = opened.data.components.map((c) => `${c.name}(${c.n_atoms})`)
console.log('   组分:', comps.join('  '))
ok(opened.data.components.some((c) => c.name === 'protein'), '识别出 protein 组分')
ok(opened.data.components.some((c) => c.name.startsWith('polymer')), '识别出 polymer 组分')
ok(opened.data.components.some((c) => c.name === 'ion'), '识别出 ion 组分')
const chains = opened.data.chains.map((c) => `${c.label}×${c.count}`)
console.log('   链类型:', chains.join('  '))
ok(chains.length >= 3, '链按组成归类（不逐个列 648 个 DOL）')
ok(opened.data.primary_atoms > 3000, '自动主链选到最大的链',
  `${opened.data.primary_label} / ${opened.data.primary_atoms} 原子`)

// ------------------------------------------------------------- 跑分析
console.log('\n[3] 跑全部分析（每 5000 ps 取一帧 = 61 帧）')
const names = ['rg', 'ree', 'dihedral', 'density', 'rdf', 'contact',
  'interface', 'orientation', 'order', 'msd']
const t0 = Date.now()
const run = await req(`/api/session/${sid}/run`, {
  method: 'POST',
  body: JSON.stringify({
    which: names,
    frames: { interval_ps: 5000 },
    params: {
      density: { axis: 2, nbins: 100 },
      interface: { axis: 2, nbins: 100 },
      rdf: { rmax: 10, nbins: 100 },
      contact: { cutoff: 4 },
      dihedral: { mode: 'auto' },
    },
    selection: { primary: { mode: 'auto' }, components: null },
  }),
})
const secs = ((Date.now() - t0) / 1000).toFixed(1)
ok(run.status === 200, 'POST /run 成功', `${secs}s`)
if (run.status !== 200) { console.log(run.data); process.exit(1) }

const results = run.data.results
console.log(`   帧选择: ${run.data.frames.describe}`)
run.data.frames.notes.forEach((n) => console.log(`           ${n}`))
console.log('   逐项耗时/规模:')
for (const [name, r] of Object.entries(results)) {
  const pts = r.curves.reduce((a, c) => a + c.x.length, 0)
  console.log(`     ${name.padEnd(12)} 面板 ${r.n_panels}  曲线 ${String(r.curves.length).padStart(2)}  `
    + `点数 ${String(pts).padStart(6)}  统计 ${String(Object.keys(r.summary).length).padStart(2)}`)
}
ok(Object.keys(results).length === 10, '10 项分析全部成功')

// ---------------------------------------------------- 关键物理量
console.log('\n[4] 关键物理量')
const rg = results.rg
console.log(`   Rg：均值 ${rg.summary['Rg mean']?.toFixed(3)} Å，`
  + `std ${rg.summary['Rg std']?.toFixed(4)}，块平均标准误 ${rg.summary['Rg 块平均标准误']?.toFixed(4)}`)
ok(rg.curves[0].x.length === 61, 'Rg 曲线 61 点', String(rg.curves[0].x.length))
ok(rg.summary['Rg mean'] > 5 && rg.summary['Rg mean'] < 60, 'Rg 数值合理')

const den = results.density
const wBulk = Object.entries(den.summary).find(([k]) => k.endsWith(' 体相密度'))
console.log(`   密度：${Object.entries(den.summary).filter(([k]) => k.includes('体相密度'))
  .map(([k, v]) => `${k.replace(' 体相密度', '')}=${v.toFixed(4)}`).join(', ')} g/cm³`)
ok(den.curves.filter((c) => c.panel === 0).length >= 2, '密度分布含多个组分',
  den.curves.filter((c) => c.panel === 0).map((c) => c.label).join(', '))

const itf = results.interface
console.log(`   界面：位置 ${itf.summary['界面位置 (Å)']?.toFixed(2)} Å，`
  + `宽度(10-90) ${itf.summary['界面宽度 10-90 (Å)']?.toFixed(2)} Å，`
  + `交点数 ${itf.summary['检测到的界面交点数']}，`
  + `可靠性 = ${itf.summary['界面判据可靠性']}`)
ok(itf.summary['界面位置 (Å)'] !== undefined, '界面分析给出界面位置')

const ct = results.contact
console.log(`   接触：平均接触对 ${ct.summary['平均接触对数']?.toFixed(0)}，`
  + `接触概率 ${ct.summary['接触概率（存在接触的帧比例）']}`)
ok(ct.summary['平均接触对数'] > 0, '蛋白-聚合物有接触')

const msd = results.msd
const msdKeys = Object.keys(msd.summary).filter((k) => k.endsWith('D (m²/s)'))
console.log('   MSD 扩散系数与可靠性:')
msdKeys.forEach((k) => {
  const name = k.replace(' D (m²/s)', '')
  const d = msd.summary[k]
  const rel = msd.summary[`${name} 拟合可靠性`]
  console.log(`     ${name.padEnd(28)} D = ${typeof d === 'number' ? d.toExponential(2) : 'nan'}`
    + ` m²/s   ${rel || ''}`)
})
ok(msdKeys.length > 0, 'MSD 给出扩散系数')
// 关键：扩散系数不得为负（MSD 平坦时应返回 nan 并说明原因）
ok(msdKeys.every((k) => msd.summary[k] === null || msd.summary[k] >= 0),
  '扩散系数没有负值（平坦 MSD 返回 nan 并注明原因）')
ok(msdKeys.some((k) => msd.summary[k] === null),
  'DOL 这类"不动"的组分被正确判为不给扩散系数')

const ori = results.orientation
console.log(`   取向参数 S = ${ori.summary['S mean']?.toFixed(4)} ± ${ori.summary['S std']?.toFixed(4)}`)
console.log(`   综合有序度指数 = ${results.order.summary['综合有序度指数 平均']?.toFixed(4)}`
  + `（分量: ${results.order.summary['综合指数包含的分量']}）`)

const rdfPks = Object.keys(results.rdf.summary).filter((k) => k.includes('第一峰位置'))
console.log('   RDF 第一峰:')
rdfPks.forEach((k) => {
  const label = k.replace(' 第一峰位置 (Å)', '')
  const g = results.rdf.summary[`${label} 第一峰高度 g_max`]
  console.log(`     ${label.padEnd(30)} ${results.rdf.summary[k].toFixed(2)} Å  `
    + `g_max=${typeof g === 'number' ? g.toFixed(2) : 'n/a'}`)
})
ok(rdfPks.length >= 2, 'RDF 覆盖多个组分对', `${rdfPks.length} 对`)
ok(rdfPks.every((k) => {
  const label = k.replace(' 第一峰位置 (Å)', '')
  return typeof results.rdf.summary[`${label} 第一峰高度 g_max`] === 'number'
}), '每对 RDF 都给出第一峰高度')
ok(Object.keys(results.rdf.summary).some((k) => k.includes('配位数')), 'RDF 给出第一壳层配位数')

// ------------------------------------------------------------- 导出
console.log('\n[5] 导出')
const exp = await req(`/api/session/${sid}/export`, {
  method: 'POST',
  body: JSON.stringify({ formats: ['csv', 'png'], dpi: 150, panel_pngs: true, excel: true }),
})
ok(exp.status === 200, 'POST /export', `CSV ${exp.data.n_csv} / PNG ${exp.data.n_png}`)
ok(exp.data.files.length > 20, '导出文件清单', `${exp.data.files.length} 个`)

const itfPng = exp.data.files.find((f) => f.name === 'interface.png')
if (itfPng) {
  const r = await fetch(BASE + itfPng.url)
  const buf = Buffer.from(await r.arrayBuffer())
  ok(r.status === 200 && buf.length > 5000, '下载 interface.png',
    `${(buf.length / 1024).toFixed(0)} KB`)
}

console.log('\n[6] 关闭会话')
const closed = await req(`/api/session/${sid}`, { method: 'DELETE' })
ok(closed.status === 200, 'DELETE /api/session/{sid}')

console.log('\n' + '='.repeat(76))
console.log(`通过 ${pass} 项` + (fails.length ? `，失败 ${fails.length} 项: ${fails.join(' | ')}` : '，全部通过'))
console.log('='.repeat(76))

// 不要用 process.exit()：fetch(undici) 的 keep-alive 套接字可能正在关闭，
// 此时强制退出会触发 libuv 断言（STATUS_STACK_BUFFER_OVERRUN），
// 使退出码变成 0xC0000409 —— 明明全部通过却报失败。
// 改为设置 exitCode 并让事件循环自然排空（undici 默认 4s 后关掉空闲连接）。
process.exitCode = fails.length ? 1 : 0
