/**
 * React 组件渲染自检。
 *
 * 做法：先用 Vite 把 `_ssr_entry.js` 打成一个 Node 可用的 SSR bundle
 * （react / echarts 作为外部依赖由 Node 自己加载），再逐个
 * `renderToString`，从而在**没有浏览器**的环境下捕获渲染期的运行时报错。
 *
 * 用法（在 frontend 目录下）:  node _ui_render.mjs
 */
import { build } from 'vite'
import react from '@vitejs/plugin-react'
import { renderToString } from 'react-dom/server'
import React from 'react'
import { readFileSync } from 'node:fs'

const ROOT = new URL('.', import.meta.url).pathname.replace(/^\//, '')
const OUT = '_ssr_build'

let pass = 0
const fails = []
function ok(cond, label, extra = '') {
  if (cond) { pass += 1; console.log(`  ✓ ${label}${extra ? '  ' + extra : ''}`) }
  else { fails.push(label); console.log(`  ✗ ${label}${extra ? '  ' + extra : ''}`) }
}

console.log('='.repeat(74))
console.log('React 组件渲染自检')
console.log('='.repeat(74))
console.log('\n[0] 打包 SSR bundle')
await build({
  configFile: false,                 // 不加载 vite.config.js（沙箱里会 EPERM）
  root: ROOT,
  plugins: [react()],
  logLevel: 'error',
  build: {
    ssr: '_ssr_entry.js',
    outDir: OUT,
    emptyOutDir: true,
    minify: false,
    ssrEmitAssets: false,
    rollupOptions: { output: { entryFileNames: 'entry.mjs', format: 'es' } },
  },
})
const M = await import(`./${OUT}/entry.mjs`)
ok(!!M.App && !!M.ChartPanel && !!M.buildOption, 'SSR bundle 打包并导入成功')

const render = (el) => renderToString(React.createElement(el))

// React 在 SSR 时会在相邻文本节点之间插入 <!-- --> 分隔符
// （例如 `第 {n} / {m} 张` 变成 `第 <!-- -->1<!-- --> / ...`），
// 断言前先去掉，否则字符串匹配会误判。
const flat = (s) => s.replace(/<!--\s*-->/g, '')

// ---------------------------------------------------------------- 空状态
console.log('\n[1] App 空状态 + 折叠面板')
const html = flat(render(M.App))
ok(html.includes('MD Trajectory Analysis Tool'), '渲染出标题')
ok(html.includes('Molecular Dynamics Simulation Analysis Platform'),
  '副标题是英文平台名')
ok(!html.includes('● 后端离线'), '可见文案不再用中文「● 后端离线」（tooltip 里保留说明）')
ok(html.includes('status: Offline'), '右对齐显示 status: Offline')
ok(html.includes('version:'), '右对齐显示 version')
ok(!html.includes('class="logo"'), '顶栏去掉了左侧 logo 方块')
ok(!html.includes('Web 版 · 分析引擎复用'), '旧的中文副标题已移除')
ok(!html.includes('sid '), '顶栏不再直接显示 sid（移到 status 的 tooltip）')

// 三块折叠面板
ok(html.includes('1. 文件读取'), '含「1. 文件读取」折叠块')
ok(html.includes('2. 体系信息'), '含「2. 体系信息」折叠块')
ok(html.includes('3. 分析设置'), '含「3. 分析设置」折叠块')
ok(!html.includes('4. 分析功能'), '不再有独立的「分析功能」块（已并入分析设置）')
ok(!html.includes('参数设置</'), '参数设置已并入分析设置折叠块')
ok(html.includes('轨迹文件') || html.includes('拓扑'), '默认展开「文件读取」')
ok(!html.includes('class="acc-item open"') === false, '有一块处于展开状态')

// 折叠块数量与互斥标记
const heads = (html.match(/class="acc-head"/g) || []).length
const openItems = (html.match(/acc-item open/g) || []).length
ok(heads === 3, '恰好 3 个折叠面板', String(heads))
ok(openItems === 1, '一次只展开一块', `open=${openItems}`)

// 未读体系时后两块禁用
const disabledHeads = (html.match(/class="acc-head" disabled/g) || []).length
ok(disabledHeads === 2, '未读体系时「体系信息/分析设置」禁用', String(disabledHeads))

// 五个结果标签页：统计量与导出是**两个同级页面**
ok(html.includes('图表') && html.includes('数据表')
  && html.includes('统计量') && html.includes('导出')
  && html.includes('运行日志'), '五个标签页都在')
ok(!html.includes('统计量 / 导出'), '不再是合并的「统计量 / 导出」')
const tabOrder = ['图表', '数据表', '统计量', '导出', '运行日志']
const idxs = tabOrder.map((t) => html.indexOf(`>${t}<`))
ok(idxs.every((i) => i >= 0) && idxs.every((v, i) => i === 0 || v > idxs[i - 1]),
  '标签顺序：图表 / 数据表 / 统计量 / 导出 / 运行日志',
  idxs.join(','))
ok(html.includes('还没有结果'), '结果区显示空状态提示')

// 左侧操作台可折叠：竖条永远在最左侧，展开时显示「收起」、收起时显示「展开」
ok(html.includes('收起操作台'), '最左侧竖条上有「收起操作台」')
ok(!html.includes('展开操作台'), '默认展开时竖条不显示「展开操作台」')
ok(/class="ops\s*"/.test(html) || /class="ops"/.test(html),
  '默认不折叠（.ops 无 collapsed）')
ok(!html.includes('ops collapsed'), '默认 .ops 不带 collapsed')
ok(html.includes('class="rail rail-l"'), '竖条 rail-l 在最左侧且始终渲染')
ok(html.includes('收起操作台，让图表占满整屏'), '竖条 title 说明收起的效果')
ok(!html.includes('ops-toggle'), '顶栏那个重复的收起按钮已移除（统一到最左侧竖条）')

// 左右两栏互斥 + 跑完后自动收起操作台 / 展开说明栏：由 App 源码断言
// （SSR 无法点击，这里检查逻辑确实写在 App 里，行为由人工/后续 e2e 验证）
const appSrc = readFileSync(new URL('./src/App.jsx', import.meta.url), 'utf8')
ok(appSrc.includes('const toggleOps'), 'App 有 toggleOps（互斥切换）')
ok(appSrc.includes('const toggleNotes'), 'App 有 toggleNotes（互斥切换）')
ok(/toggleOps[\s\S]{0,200}setNotesCollapsed\(true\)/.test(appSrc),
  '展开操作台时会收起说明栏')
ok(/toggleNotes[\s\S]{0,200}setOpsCollapsed\(true\)/.test(appSrc),
  '展开说明栏时会收起操作台')
ok(appSrc.includes('const focusChart') && /focusChart\(\)/.test(appSrc),
  '点击开始分析后调用 focusChart')
// 关键：必须在**开始分析时**就让位，而不是等全部跑完
const runBodyIdx = appSrc.indexOf('const doRun')
const focusIdx = appSrc.indexOf('focusChart()', runBodyIdx)
const firstAwait = appSrc.indexOf('await api.runStart', runBodyIdx)
ok(focusIdx > runBodyIdx && firstAwait > 0 && focusIdx < firstAwait,
  'focusChart() 在第一个 await 之前调用（点开始就隐藏，不是跑完才隐藏）')
ok(!/if \(names\.length\) \{\s*focusChart\(\)/.test(appSrc),
  '不会再在跑完后重复调用 focusChart')
ok(/if \(!which\.length\)/.test(appSrc), '未选任何分析项时给出提示而不是空跑')
ok(appSrc.includes('topbar-live'), '顶栏有实时进度（收起操作台后仍能看到进度）')

// ---------------------------------------------------------- 各组件渲染
console.log('\n[2] 各组件单独渲染')
const info = {
  n_atoms: 47681, n_residues: 11302, n_segments: 3, n_frames: 10, dt_ps: 100,
  first_time_ps: 0, last_time_ps: 900, total_time_ps: 900,
  box_dimensions: [80.017, 80.017, 80.017], box_angles: [60, 60, 90],
  box_type: 'triclinic', box_volume: 362269.6, total_mass_amu: 223356.7,
  total_charge_e: 0, n_bonds: 25533, n_angles: 6123, n_dihedrals: 7481,
  categories: { protein: 214, water: 11084, ion: 4 }, text: '体系信息报告正文',
  // 后端 info_tables() 给的结构化小节（完整报告改用表格渲染）
  sections: [
    {
      title: '文件与规模', columns: ['项目', '值'],
      rows: [['拓扑文件', 'adk_oplsaa.tpr'], ['原子数', '47,681']],
    },
    {
      title: '元素组成', columns: ['元素', '原子数', '占比'],
      rows: [['H', '23,853', '65.18%'], ['O', '11,404', '31.16%']],
      note: '共 2 种元素。',
    },
    {
      title: '分子 / 链信息（按组成归类）',
      columns: ['分组', '条数', '每条原子数', '每条 residue 数', '尺寸分布'],
      rows: [['seg_0_AKeco:MET', '1', '3,341', '214', '—']],
      note: '「每条原子数」是单条链的规模。',
    },
  ],
}
let h = flat(render(() => React.createElement(M.InfoPanel, {
  info, onNext: () => {},
})))
ok(h.includes('47,681') || h.includes('47681'), 'InfoPanel 显示原子数')
ok(h.includes('protein: 214'), 'InfoPanel 显示组分标签')
ok(h.includes('体系信息报告正文') === false, '完整报告默认折叠')
ok(h.includes('展开完整报告'), 'InfoPanel 有「展开完整报告」按钮')
ok(h.includes('下一步：分析设置'), 'InfoPanel 提供「下一步」按钮')
ok(!h.includes('inforeport'), '未展开时不渲染报告容器')
ok(!h.includes('<pre'), 'InfoPanel 不再用 <pre> 定宽文本展示报告')

// 完整报告：结构化表格（横向锁定、单元格换行由 CSS 保证）
h = flat(render(() => React.createElement(M.InfoReport, { sections: info.sections })))
ok(h.includes('inforeport'), 'InfoReport 渲染报告容器')
ok((h.match(/<table/g) || []).length === 3, '三个小节渲染成三张表',
  `table=${(h.match(/<table/g) || []).length}`)
ok(h.includes('<thead>'), '表格有表头')
ok(h.includes('文件与规模') && h.includes('元素组成'), '渲染小节标题')
ok(h.includes('分子 / 链信息（按组成归类）'), '渲染分子/链小节')
ok(h.includes('每条原子数') && h.includes('尺寸分布'),
  '链信息列名写清「每条」（不是合计）')
ok(h.includes('65.18%') && h.includes('3,341'), '渲染百分比与数值单元格')
ok(h.includes('共 2 种元素。'), '渲染小节备注')
ok(!h.includes('<pre'), '报告里没有 <pre>')
const h0 = flat(render(() => React.createElement(M.InfoReport, { sections: [] })))
ok(h0 === '', 'sections 为空时不渲染')

// 旧版后端只给 text 时：退回文本，但外层仍是锁横向的容器
h = flat(render(() => React.createElement(M.InfoPanel, {
  info: { ...info, sections: [] }, onNext: () => {},
})))
ok(h.includes('展开完整报告'), '没有 sections 时仍可展开旧版文本报告')

const sess = {
  primary_label: '最大链', primary_atoms: 3341,
  components: [{ name: 'protein', n_atoms: 3341, n_residues: 214 },
    { name: 'water', n_atoms: 44336, n_residues: 11084 }],
  chains: [{ label: 'seg_0:MET', segid: 'seg_0', resname: 'MET', n_atoms: 3341,
    description: '1 条' }],
}
h = flat(render(() => React.createElement(M.SelectionBlock, {
  sess, primary: { mode: 'auto' }, setPrimary: () => {}, components: ['protein'],
  setComponents: () => {},
})))
ok(h.includes('分析对象'), 'SelectionBlock 有「分析对象」小标题')
ok(h.includes('主链') && h.includes('组分 protein'), 'SelectionBlock 渲染主链与组分选项')
ok(h.includes('chip-pick on'), '已选组分高亮（胶囊式多选）')
ok(h.includes('water'), 'SelectionBlock 列出全部组分')

// 「帧选择」独立成板块（排在「分析功能」之前），用**表格约束布局**
h = flat(render(() => React.createElement(M.FrameBlock, {
  frames: { interval_ps: '100' }, setFrames: () => {},
})))
ok(h.includes('帧选择'), 'FrameBlock 有「帧选择」小标题')
ok(h.includes('起始 (ps)') && h.includes('平衡 (ps)') && h.includes('抽帧 (ps)'),
  'FrameBlock 渲染 5 个帧选择参数')
ok(/<table class="ptab">/.test(h),
  'FrameBlock 用表格约束布局（列在各行间共享 → 输入框左边缘天然对齐）')
ok(/<th scope="row">起始 \(ps\)<\/th><td colSpan="1"><input/.test(h),
  '表格式：标签放 th、输入放紧随的 td')
const nRows = (h.match(/<tr>/g) || []).length
ok(nRows === 2, '帧选择 5 个参数排成 2 行（3 个 + 2 个）', `rows=${nRows}`)

// **提示文字必须完整显示**：每个带 placeholder 的输入框都要有内联 min-width，
// 且 ≥ 提示文字宽度 + 内边距12 + 边框2（元素级保证，不依赖布局推算）。
function checkPlaceholders(html, label, minN = 1) {
  const re = /placeholder="([^"]+)"[^>]*?min-width:([0-9.]+)px/g
  let m
  let n = 0
  const bad = []
  while ((m = re.exec(html)) !== null) {
    n += 1
    const need = [...m[1]].reduce(
      (w, ch) => w + (ch.codePointAt(0) < 0x2E80 ? 0.55 : 1.0) * 12.5, 0) + 14
    if (Number(m[2]) + 0.5 < need) bad.push(`${m[1]}: ${m[2]}px < ${need.toFixed(1)}px`)
  }
  ok(n >= minN, `${label}：带提示文字的输入框都有内联 min-width`, `n=${n}`)
  ok(bad.length === 0, `${label}：提示文字都放得下（不会被截断）`, bad.join('; '))
}
checkPlaceholders(h, '帧选择', 5)

// 「分析参数」只含按分析项分组的参数（帧选择已独立出去）
let ph = flat(render(() => React.createElement(M.ParamBlock, {
  params: { cutoff: 5, axis: 2, nbins: 100, rmax: 12, dihedral_mode: 'auto' },
  setParams: () => {},
})))
ok(ph.includes('分析参数'), 'ParamBlock 有「分析参数」小标题')
ok(!ph.includes('起始 (ps)') && !ph.includes('最多帧数'),
  'ParamBlock 不再包含帧选择参数（已独立成板块）')
ok(/<table class="ptab">/.test(ph), 'ParamBlock 也用表格约束布局')
ok(/<th scope="row">trans\/gauche 阈值 \(°\)<\/th><td colSpan="5">/.test(ph),
  '长参数（trans/gauche 阈值）独占一行（colSpan=5）')
ok(ph.includes('cutoff (Å)') && ph.includes('密度方向') && ph.includes('二面角模式'),
  'ParamBlock 渲染分析参数')

checkPlaceholders(ph, '分析参数')

// 参数**按分析项分组**：只显示当前勾选的分析项的参数（which=null 时全显示）
ph = flat(render(() => React.createElement(M.ParamBlock, {
  params: {}, setParams: () => {},
  which: ['rdf'],
})))
ok(ph.includes('径向分布函数 RDF') && ph.includes('RDF 最大 r (Å)'),
  '参数分组：选中 RDF 时显示 RDF 组的参数')
ok(!ph.includes('cutoff (Å)') && !ph.includes('trans/gauche 阈值'),
  '参数分组：未选中的接触/二面角参数被隐藏')
ok(!ph.includes('MSD 追踪对象') && !ph.includes('取向链段定义'),
  '参数分组：未选中的 MSD/取向参数被隐藏')
ok(!ph.includes('起始 (ps)') && !ph.includes('最多帧数'),
  '参数分组：帧选择已独立成板块，不再出现在分析参数里')
ok(ph.includes('随上方「分析功能」的勾选出现'),
  '参数分组：标题说明这些参数随勾选出现')
ph = flat(render(() => React.createElement(M.ParamBlock, {
  params: {}, setParams: () => {}, which: ['order'],
})))
ok(ph.includes('局部结构 g_ref'), '参数分组：结构有序度暴露 g_ref（留空则不计入指数）')
ok(/<th scope="row">局部结构 g_ref<\/th><td colSpan="2">/.test(ph),
  'g_ref 占 2 列（一行 2 个），且提示文字有内联 min-width')
checkPlaceholders(ph, '结构有序度')
ph = flat(render(() => React.createElement(M.ParamBlock, {
  params: {}, setParams: () => {}, which: [],
})))
ok(ph.includes('尚未勾选任何分析项'), '参数分组：一个都没勾时给出提示而不是空白')
ok(!ph.includes('cutoff (Å)'), '参数分组：切到 order 后接触参数仍隐藏')

h = flat(render(() => React.createElement(M.FunctionBlock, {
  titles: { rg: '回转半径 Rg' }, order: ['rg'], which: ['rg'], setWhich: () => {},
})))
ok(h.includes('分析功能'), 'FunctionBlock 有「分析功能」小标题')
ok(h.includes('回转半径 Rg'), 'FunctionBlock 渲染勾选项')
ok(h.includes('链构象') && h.includes('空间结构') && h.includes('取向与结晶')
  && h.includes('动力学与输运'), 'FunctionBlock 四组分类齐全')
ok(h.includes('已选 1 / 1'), 'FunctionBlock 显示已选数量')

// 「仅 XX」预设按钮：每一类都要有一个，参考「仅链构象」
const FB_TITLES = {
  rg: '回转半径 Rg', ree: '端到端距离 R_ee', dihedral: '二面角分析',
  density: '密度分布', rdf: '径向分布函数 RDF', contact: '接触分析',
  interface: '界面宽度分析', orientation: '链段取向分析',
  order: '结构有序度分析', msd: '均方位移 MSD',
}
const FB_ORDER = Object.keys(FB_TITLES)
h = flat(render(() => React.createElement(M.FunctionBlock, {
  titles: FB_TITLES, order: FB_ORDER, which: [...FB_ORDER], setWhich: () => {},
})))
// 按钮用 2 字缩写（完整名在 tooltip 里）：完整名太长会把这一行挤爆，
// 而这一行要求在单行内放下（webapp/_css_check.py 有静态宽度估算盯着）
for (const label of ['仅构象', '仅结构', '仅取向', '仅输运']) {
  ok(h.includes(label), `FunctionBlock 有「${label}」按钮`)
}
ok(h.includes('title="只勾选「动力学与输运」这一类'), '按钮 tooltip 保留分组完整名')
ok(h.includes('全选'), 'FunctionBlock 保留「全选」')
ok(!h.includes('全不选'), '已去掉「全不选」（全不选 = 什么都不勾，无需按钮）')
const nBtns = (h.match(/class="mini/g) || []).length
ok(nBtns === 5, '按钮行恰好 5 个（全选 + 4 个仅 XX），能排在一行', `buttons=${nBtns}`)

// 当前选择正好等于「界面」这一类时，该按钮要高亮（active）
h = flat(render(() => React.createElement(M.FunctionBlock, {
  titles: FB_TITLES, order: FB_ORDER,
  which: ['density', 'rdf', 'contact', 'interface'], setWhich: () => {},
})))
ok(/class="mini active"[^>]*>仅结构/.test(h) || /仅结构/.test(h) && /mini active/.test(h),
  '「仅结构」在选中空间结构类时高亮')
ok(h.includes('已选 4 / 10'), '高亮时已选数量正确')

// 选择为「取向与结晶」时应高亮「仅取向」而不是「仅结构」
h = flat(render(() => React.createElement(M.FunctionBlock, {
  titles: FB_TITLES, order: FB_ORDER, which: ['orientation', 'order'],
  setWhich: () => {},
})))
const activeCount = (h.match(/mini active/g) || []).length
ok(activeCount === 1, '同时只有一个预设处于高亮态', `active=${activeCount}`)
ok(h.includes('仅取向'), '取向类被高亮时「仅取向」按钮存在')

// 不带参数也能渲染（例如 order 为空时）
h = flat(render(() => React.createElement(M.FunctionBlock, { titles: {}, order: [] })))
ok(h.includes('仅结构') && h.includes('仅取向'), 'order 为空时预设按钮仍在')

h = flat(render(() => React.createElement(M.RunBlock, {
  busy: false, progress: 0, onRun: () => {}, onCancel: () => {},
  lastLine: '[12:00:00] 完成：10 项分析', onOpenLog: () => {},
})))
ok(h.includes('开始分析'), 'RunBlock 有运行按钮')
ok(h.includes('完成：10 项分析'), 'RunBlock 显示最近一行日志')
ok(h.includes('查看完整运行日志'), 'RunBlock 提供「查看完整日志」链接')

// ---------------------------------------------- 实时运行（边算边出）
const live = {
  status: 'running', message: '已完成 4/10：径向分布函数 g(r)',
  nDone: 4, nTotal: 10, elapsed: 12.3,
  pending: ['contact', 'interface', 'orientation', 'order', 'msd'],
}
h = flat(render(() => React.createElement(M.RunBlock, {
  busy: true, progress: 0.4, onRun: () => {}, onCancel: () => {}, live,
  which: ['rg', 'ree', 'dihedral', 'density', 'rdf', 'contact', 'interface',
          'orientation', 'order', 'msd'],
  titles: { contact: '接触分析', interface: '界面宽度分析' },
  lastLine: '', onOpenLog: () => {},
})))
ok(h.includes('实时更新中'), 'RunBlock 显示「实时更新中」徽标')
ok(h.includes('4/10'), 'RunBlock 显示已完成 / 总数', '4/10')
ok(h.includes('已完成 4/10：径向分布函数 g(r)'), 'RunBlock 显示后端进度消息')
ok(h.includes('已用 12s'), 'RunBlock 显示已用时间')
ok(h.includes('接触分析') && h.includes('界面宽度分析'), 'RunBlock 列出排队中的分析（用中文标题）')
ok(h.includes('bar'), 'RunBlock 显示进度条')
ok(h.includes('取消') && h.includes('保留已算完的'), 'RunBlock 说明取消会保留已完成结果')
ok(!h.includes('丢弃本次请求'), 'RunBlock 不再说"丢弃本次请求"')

// 实时运行时、第一项还没算完：不能说"还没有结果"
h = flat(render(() => React.createElement(M.ChartPanel, {
  run: null, chart: { name: '', index: 0 }, setChart: () => {},
  live: { status: 'running', nDone: 0, nTotal: 10, pending: ['rg'] },
  titles: { rg: '回转半径 Rg' },
})))
ok(h.includes('正在计算第一项'), 'ChartPanel 在实时运行时显示「正在计算第一项」')
ok(h.includes('回转半径 Rg'), 'ChartPanel 显示正在算的分析名')
ok(h.includes('实时更新中'), 'ChartPanel 空态也带实时徽标')
ok(!h.includes('还没有结果'), 'ChartPanel 实时运行时不再显示「还没有结果」')

h = flat(render(() => React.createElement(M.OpenPanel, { onOpened: () => {}, log: () => {} })))
ok(h.includes('拓扑') && h.includes('轨迹'), 'OpenPanel 显示两个文件槽')
ok(h.includes('转到') && h.includes('读取体系'), 'OpenPanel 含目录浏览器与读取按钮')

h = flat(render(() => React.createElement(M.LogPanel, {
  log: ['[12:00:00] 开始分析'], onClear: () => {},
})))
ok(h.includes('运行日志') && h.includes('开始分析'), 'LogPanel 渲染日志内容')
ok(h.includes('清空'), 'LogPanel 有清空按钮')

// -------------------------------------------------- 带完整结果的页面
console.log('\n[3] 带结果的页面（模拟后端返回）')
const run = {
  results: {
    rg: {
      name: 'rg', title: '回转半径 Rg —— 最大链', n_panels: 2,
      panels: [
        { index: 0, title: 'Rg 随时间变化', xlabel: '时间 Time (ps)', ylabel: '回转半径 Rg (Å)', xscale: 'linear', yscale: 'linear', legend: true, curves: ['Rg'] },
        { index: 1, title: 'Rg 分布', xlabel: 'Rg (Å)', ylabel: 'P(Rg)', xscale: 'linear', yscale: 'linear', legend: false, curves: ['Rg 分布'] },
      ],
      curves: [
        { label: 'Rg', kind: 'line', panel: 0, x: [0, 300, 600], y: [19.65, 19.83, 19.57], n: 3, downsampled: false },
        { label: 'Rg 分布', kind: 'bar', panel: 1, x: [19.3, 19.6, 19.9], y: [0.3, 0.7, 0.3], n: 3, downsampled: false },
      ],
      summary: { 'Rg mean': 19.669698, 'Rg std': 0.19, 'Rg 块平均标准误': 0.07 },
      notes: ['质量加权: True；PBC 展开: True'],
    },
    msd: {
      name: 'msd', title: '均方位移 MSD', n_panels: 1,
      panels: [{ index: 0, title: 'MSD 随时间间隔变化', xlabel: 'τ (ps)', ylabel: 'MSD (Å²)', xscale: 'log', yscale: 'log', legend: true, curves: ['water OW'] }],
      curves: [{ label: 'water OW', kind: 'line', panel: 0, x: [0, 300, 600, 900], y: [0, 600, 1200, 1800], n: 4, downsampled: false }],
      summary: { 'water OW D (m²/s)': 3.277e-9 },
      notes: [],
    },
  },
}

// 跟 App 一样：由 chart({name,index}) 推出「当前这张图」，再传给 ChartPanel。
// 说明文字已移到右侧 NotesPanel，所以这里要显式把 current 传进来。
const pick = (r, chart) => {
  const names = Object.keys(r.results)
  const result = r.results[chart.name] || r.results[names[0]]
  const panelIndex = Math.max(0, Math.min(chart.index ?? 0,
    (result.panels?.length || 1) - 1))
  return { result, panelIndex }
}
const chartPanel = (r, chart, extra = {}) => {
  const { result, panelIndex } = pick(r, chart)
  return flat(render(() => React.createElement(M.ChartPanel, {
    run: r, chart, setChart: () => {}, current: result, panelIndex, ...extra,
  })))
}

h = chartPanel(run, { name: 'rg', index: 0 })
ok(h.includes('图表导航'), '图表导航栏标题')
ok(h.includes('Rg 随时间变化') && h.includes('Rg 分布'), '导航里列出全部图标题')
ok(h.includes('回转半径 Rg —— 最大链'), '右侧显示当前图的分析标题')
ok(h.includes('第 1 / 2 张'), '显示「第 N / M 张」')
ok(h.includes('上一张') && h.includes('下一张'), '上一张/下一张按钮')
ok(h.includes('chart-box'), '含 ECharts 容器')
// 说明文字与统计量已移到右侧面板，图表区不再有 chartfoot
ok(!h.includes('chartfoot'), '图表下方不再有 chartfoot（已移到右侧）')
ok(!h.includes('Rg mean'), '图表区不再重复显示统计量卡片')

// 全屏按钮
ok(h.includes('全屏'), 'ChartPanel 有全屏按钮')
ok(h.includes('⤢'), '全屏按钮带图标')
ok(h.includes('全屏显示这张图'), '全屏按钮 title 说明用途')
ok(!h.includes('退出全屏'), '未全屏时显示的是「全屏」而不是「退出全屏」')

// 部分结果已到：导航里标出还有几项在算（实时运行的核心表现）
h = chartPanel(run, { name: 'rg', index: 0 },
  { live: { status: 'running', nDone: 1, nTotal: 3, pending: ['rdf', 'msd'] } })
ok(h.includes('还有 2 项在算'), 'ChartPanel 导航显示还有几项在算')
ok(h.includes('Rg 随时间变化'), 'ChartPanel 仍正常显示已到达的结果')

// 非运行态（跑完了）不应出现实时徽标
h = chartPanel(run, { name: 'rg', index: 0 },
  { live: { status: 'done', nDone: 3, nTotal: 3, pending: [] } })
ok(!h.includes('还有'), '跑完后不再显示「还有 N 项在算」')

h = chartPanel(run, { name: 'rg', index: 1 })
ok(h.includes('第 2 / 2 张'), '切到第 2 张图')

h = chartPanel(run, { name: 'msd', index: 0 })
ok(h.includes('对数轴'), 'MSD 标出「对数轴」')

// 非法 index（超出该分析的面板数）要被夹住，不能崩
h = chartPanel(run, { name: 'msd', index: 99 })
ok(h.includes('第 1 / 1 张'), '超出范围的面板序号被夹到最后一页')

// 切到不存在的分析名时回退到第一项
h = chartPanel(run, { name: 'nope', index: 0 })
ok(h.includes('回转半径 Rg —— 最大链'), '未知分析名回退到第一项结果')

// ------------------------------------------------ 右侧「图表说明」面板
h = flat(render(() => React.createElement(M.NotesPanel, pick(run, { name: 'rg', index: 0 }))))
ok(h.includes('图表说明'), 'NotesPanel 有「图表说明」标题')
ok(h.includes('PBC 展开'), 'NotesPanel 显示算法说明')
ok(h.includes('Rg mean'), 'NotesPanel 显示关键统计量')
ok(h.includes('关键统计量'), 'NotesPanel 有「关键统计量」小标题')
ok(h.includes('第 1 / 2 张'), 'NotesPanel 标出当前是第几张')
ok(h.includes('1 条'), 'NotesPanel 显示说明条数')
// 折叠动画的载体：内容必须被包在 .notes-clip 里，否则折叠后竖条会被挤出可视区
ok(h.includes('notes-clip'), 'NotesPanel 内容包在 .notes-clip 中（折叠后竖条仍可点）')
ok(/notes-clip[^>]*>\s*<div class="notes-body"/.test(h),
  '.notes-body 在 .notes-clip 内部')
// 排版：说明用悬挂缩进的 noterow，统计量用受约束的表格（不要再错位）
ok(h.includes('notelist') && h.includes('noterow'), '说明条目用 noterow（悬挂缩进，换行对齐）')
ok(h.includes('notebullet') && h.includes('notetext'),
  '项目符号与正文分开（换行后不会缩到符号底下）')
ok(h.includes('notes-table'), '统计量用 notes-table（固定列宽 + 换行，不横向滚动）')
ok(h.includes('<table'), '统计量确实是表格')
ok(!h.includes('statcards'), 'NotesPanel 不再用卡片网格（改用表格约束对齐）')
ok(h.includes('<th>Rg mean</th>'), '统计量键在 <th> 列')

h = flat(render(() => React.createElement(M.NotesPanel, pick(run, { name: 'msd', index: 0 }))))
ok(h.includes('这一项没有额外的算法说明'), '没有说明时给出明确提示')
ok(h.includes('water OW D'), '没有说明也能显示统计量')

h = flat(render(() => React.createElement(M.NotesPanel, {})))
ok(h === '', 'NotesPanel 没有结果时不渲染')

h = flat(render(() => React.createElement(M.DataPanel, { run, groups: M.FALLBACK_GROUPS })))
ok(h.includes('数据表'), 'DataPanel 标题')
// 数据表导航用与图表导航相同的模块分层
ok(h.includes('navmodule'), '数据表导航有模块层')
ok(h.includes('class="mtitle">链构象'), '数据表导航模块标题=链构象')
ok(h.includes('class="mtitle">动力学与输运'), '数据表导航模块标题=动力学与输运')
ok(h.includes('1 项 / 2 张'), '数据表模块显示「N 项 / M 张」')
ok(!h.includes('class="mtitle">界面'), '数据表里没结果的模块也不显示')
h = flat(render(() => React.createElement(M.DataPanel, { run })))
ok(h.includes('navmodule'), 'DataPanel 不传 groups 时也能分层（用兜底）')
ok(h.includes('时间 Time (ps)'), '表头使用面板的 x 轴名称')
ok(h.includes('19.65') && h.includes('19.83'), '表格里出现 Rg 数值')
ok(h.includes('3 行 × 2 列') || h.includes('3 行'), '显示行列表头信息')
ok(h.includes('下载 CSV'), '有下载 CSV 按钮')

// ------------------------------------------------ 「导出」页（只管导出）
h = flat(render(() => React.createElement(M.ExportPanel, { run, sid: 'abc', log: () => {} })))
ok(h.includes('导出结果') && h.includes('开始导出'), '导出页渲染')
// 1) 目标目录：显示 + 选择按钮
ok(h.includes('目标目录'), '有「目标目录」一行')
ok(h.includes('（默认：会话输出目录）'), '未选目录时显示默认')
ok(h.includes('选择…'), '有「选择…」按钮')
// 2) 可选模块（按大纲）+ 可选文件形式
ok(h.includes('导出模块'), '有「导出模块」一行')
ok(h.includes('chinline'), '模块用胶囊勾选')
ok(h.includes('文件形式'), '有「文件形式」一行')
for (const f of ['CSV 数据表', 'PNG 图', 'Excel 汇总']) {
  ok(h.includes(f), `文件形式可勾选：${f}`)
}
ok(h.includes('2 / 2 项分析'), '默认全部模块勾选，显示已选分析项数', '2 / 2')
// 3) 打开目标目录按钮（排在开始导出右边）
ok(h.includes('📂 打开目标目录'), '有「打开目标目录」按钮')
ok(h.indexOf('开始导出') < h.indexOf('打开目标目录'),
  '「打开目标目录」排在「开始导出」右边')
// 4) 未导出时不显示"最近导出"，且导出页**不显示统计量明细**
ok(!h.includes('最近导出'), '未导出时不显示最近导出信息')
ok(!h.includes('Rg mean'), '导出页不掺统计量明细')
ok(!h.includes('stattabs'), '导出页没有统计量的模块标签栏')
ok(!h.includes('statblock'), '导出页没有统计量明细区块')
ok(!h.includes('<th>统计量</th>'), '导出页没有「统计量 / 数值」表')
// 5) 下方「选择要导出的文件」：按模块分板块（参照统计量页的设计）
ok(h.includes('filetabs'), '有「选择要导出的文件」区（独立类名 filetabs）')
ok(h.includes('选择要导出的文件'), '区块标题正确')
ok(h.includes('filegroup'), '文件按模块分板块')
ok(h.includes('filegroup-head'), '每个板块有标题栏（含全选 / 全不选）')
ok(h.includes('rg__panel0.csv'), '列出实际会产出的文件名（多面板带 __panelN）')
ok(h.includes('rg.png'), '列出合并图 PNG')
ok(h.includes('summary.xlsx'), '列出 Excel 汇总')
ok(h.includes('已选'), '显示已选文件数')
ok(h.includes('type="checkbox"'), '每个文件有勾选框')

// ------------------------------------------------ 「统计量」页（只显示指标）
h = flat(render(() => React.createElement(M.StatsPanel, { run, groups: M.FALLBACK_GROUPS })))
ok(h.includes('stattabs'), '统计量页有模块标签栏')
ok(!h.includes('导出结果') && !h.includes('开始导出'), '统计量页不含任何导出控件')
ok(!h.includes('目标目录') && !h.includes('文件形式'), '统计量页不含导出设置')
ok(h.includes('链构象') && h.includes('动力学与输运'), '标签栏含各模块')
ok(!h.includes('statmodule'), '不再一次性铺开全部模块')
// 默认停在第一个有结果的模块
ok(h.includes('Rg mean'), '默认显示第一个模块（链构象）的统计量')
ok(h.includes('<thead>') && h.includes('<th>统计量</th>') && h.includes('<th>数值</th>'),
  '统计量表带表头「统计量 / 数值」')
ok(!h.includes('water OW D'), '只显示本模块（不含辅助模块的 MSD 统计量）')

h = flat(render(() => React.createElement(M.StatsPanel, {
  run, groups: M.FALLBACK_GROUPS, defaultView: '动力学与输运',
})))
ok(h.includes('water OW D'), '切到「辅助」标签显示 MSD 统计量')
ok(!h.includes('Rg mean'), '切模块后不显示其它模块的统计量')

h = flat(render(() => React.createElement(M.StatsPanel, {
  run, groups: M.FALLBACK_GROUPS, defaultView: '空间结构',
})))
ok(h.includes('这个模块下没有结果'), '没有结果的模块给出明确提示')

h = flat(render(() => React.createElement(M.StatsPanel, { run, groups: null })))
ok(h.includes('Rg mean'), 'groups 为 null 时用兜底分组')

// 目录选择器可以单独渲染（导出页的「选择…」展开后就是它）
h = flat(render(() => React.createElement(M.DirPicker, {
  value: '', onPick: () => {}, onClose: () => {},
})))
ok(h.includes('dirpicker'), 'DirPicker 渲染目录浏览器')
ok(h.includes('用这个目录'), 'DirPicker 有确认按钮')
ok(h.includes('读取中') || h.includes('dp-path'), 'DirPicker 显示当前路径')

h = flat(render(() => React.createElement(M.SummaryView, { result: run.results.rg, compact: true })))
ok(h.includes('statcard'), 'SummaryView compact 模式渲染卡片')

// 不可测量的统计量：后端把 NaN 序列化成 JSON null，界面上必须显示成破折号，
// 不能出现字面量 "null" / "NaN"（界面分析在非分层体系里就是这种情况）
const withNull = {
  name: 'interface', title: '界面', summary: {
    '界面位置 (Å)': 109.32, '组分 A 界面宽度 10-90 (Å)': null,
    '界面2 界面宽度 10-90 (Å)': null, '界面宽度 10-90 (Å)': 9.02,
  },
}
h = flat(render(() => React.createElement(M.SummaryView, { result: withNull })))
ok(!/null|NaN|undefined/.test(h), 'SummaryView 不显示 null/NaN 字面量', h.slice(0, 80))
ok(h.includes('—'), 'SummaryView 把不可测值显示成「—」')
ok(h.includes('109.32'), 'SummaryView 可测值正常显示', '109.32')
h = flat(render(() => React.createElement(M.SummaryView, { result: withNull, compact: true })))
ok(!/null|NaN|undefined/.test(h), 'SummaryView compact 模式同样不显示 null')
ok(M.formatCell(null) === '—' && M.formatCell(NaN) === '不可测',
  'formatCell 处理 null / NaN', `${M.formatCell(null)} / ${M.formatCell(NaN)}`)
ok(M.formatCell(0) === '0' && M.formatCell(false) === '否',
  'formatCell 不把 0/false 误当空值')

// ---------------------------------------------------- ECharts option
console.log('\n[4] ECharts option 构造')
// 单曲线面板：自动隐藏图例
const opt = M.buildOption(run.results.rg, 0)
ok(opt.series.length === 1 && opt.series[0].type === 'line', '线性面板 -> line series')
ok(opt.grid.containLabel === true, 'grid.containLabel = true（标签不越界）')
ok(opt.dataZoom.length === 2, 'dataZoom 同时有 inside 与 slider（缩放/平移）')
ok(opt.tooltip.trigger === 'axis', 'tooltip 轴触发（hover 读数）')
ok(!!opt.toolbox.feature.saveAsImage, 'toolbox 支持保存图片')
ok(opt.xAxis.name === '时间 Time (ps)', 'x 轴名称取自面板')

// 多曲线面板：图例可开关
const multi = {
  name: 'rg', title: 'Rg', n_panels: 1,
  panels: [{ index: 0, title: 'Rg 随时间变化', xlabel: 't', ylabel: 'Rg',
    xscale: 'linear', yscale: 'linear', legend: true, curves: ['a', 'b'] }],
  curves: [
    { label: '链 A', kind: 'line', panel: 0, x: [0, 1], y: [1, 2], n: 2, downsampled: false },
    { label: '链 B', kind: 'line', panel: 0, x: [0, 1], y: [2, 1], n: 2, downsampled: false },
  ],
  summary: {}, notes: [],
}
const optMulti = M.buildOption(multi, 0)
ok(optMulti.legend.data.includes('链 A') && optMulti.legend.data.includes('链 B'),
  '多曲线面板图例含全部曲线名（可开关）')
ok(optMulti.series.length === 2 && optMulti.series[0].color !== optMulti.series[1].color,
  '多条曲线使用不同颜色')

const optBar = M.buildOption(run.results.rg, 1)
ok(optBar.series[0].type === 'bar', '分布面板 -> bar series')
ok(optBar.legend.show === false, '单曲线面板隐藏图例')

const optLog = M.buildOption(run.results.msd, 0)
ok(optLog.xAxis.type === 'log' && optLog.yAxis.type === 'log', '对数面板 -> log 轴')
ok(optLog.series[0].data.every((p) => p[0] > 0 && p[1] > 0), '对数轴自动剔除 ≤0 的点')

// step / scatter 画法
const kinds = {
  name: 'k', title: 'k', n_panels: 1,
  panels: [{ index: 0, title: 'k', xlabel: 'x', ylabel: 'y', xscale: 'linear',
    yscale: 'linear', legend: true, curves: ['s', 'p'] }],
  curves: [
    { label: '阶跃', kind: 'step', panel: 0, x: [0, 1], y: [1, 2], n: 2, downsampled: false },
    { label: '散点', kind: 'scatter', panel: 0, x: [0, 1], y: [2, 1], n: 2, downsampled: false },
  ],
  summary: {}, notes: [],
}
const optKinds = M.buildOption(kinds, 0)
ok(optKinds.series[0].step === 'middle', 'step 画法 -> ECharts step: middle')
ok(optKinds.series[1].type === 'scatter', 'scatter 画法 -> scatter series')

// --------------------------------------------------------- 工具函数
console.log('\n[5] 工具函数')
ok(M.formatNumber(19.669698) === '19.67', 'formatNumber 四舍五入', M.formatNumber(19.669698))
ok(M.formatNumber(3.277e-9).includes('e-9'), 'formatNumber 科学计数')
ok(M.formatNumber(NaN) === 'nan', 'formatNumber 处理 NaN')
ok(M.formatNumber(null) === '', 'formatNumber 处理 null')
ok(M.guessTrajectory('C:/d/sys.tpr', [{ path: 'C:/d/sys.xtc' }]) === 'C:/d/sys.xtc',
  'guessTrajectory 自动匹配同名轨迹')
ok(M.guessTrajectory('C:/d/sys.tpr', [{ path: 'C:/d/other.xtc' }]) === null,
  'guessTrajectory 不匹配其它文件')
ok(typeof M.api.open === 'function' && typeof M.api.run === 'function',  'api 暴露 open / run')
ok(typeof M.api.runStart === 'function' && typeof M.api.runProgress === 'function'
   && typeof M.api.runCancel === 'function', 'api 暴露实时运行三件套 runStart/runProgress/runCancel')

// 分组来自 /api/analyses 的 groups（权威定义在后端），带本地兜底
console.log('\n[6] 分析分组与图表导航分层')
ok(Array.isArray(M.FALLBACK_GROUPS) && M.FALLBACK_GROUPS.length === 4,
  '兜底分组有 4 个模块', `n=${M.FALLBACK_GROUPS?.length}`)
ok(M.FALLBACK_GROUPS.map(([l]) => l).join('/')
  === '链构象/空间结构/取向与结晶/动力学与输运',
  '兜底模块名与后端定义一致', M.FALLBACK_GROUPS.map(([l]) => l).join('/'))
ok(M.FALLBACK_GROUPS.every(([, , s]) => typeof s === 'string' && s.length >= 1 && s.length <= 3),
  '每个兜底分组都带 2 字按钮缩写', M.FALLBACK_GROUPS.map(([, , s]) => s).join('/'))
const allNames = M.FALLBACK_GROUPS.flatMap(([, ns]) => ns)
ok(allNames.length === 10 && new Set(allNames).size === 10,
  '兜底分组覆盖 10 项且不重复', `n=${allNames.length}`)
ok(M.normalizeGroups(null) === M.FALLBACK_GROUPS, 'groups 缺失时用兜底')
ok(M.normalizeGroups([]) === M.FALLBACK_GROUPS, 'groups 为空时用兜底')
ok(M.normalizeGroups([{ label: 'X', names: ['rg'] }])[0][0] === 'X',
  '接受后端 {label,names} 形状')
ok(M.normalizeGroups([['X', ['rg']]])[0][0] === 'X', '也接受 [label,names] 形状')
ok(M.normalizeGroups([{ label: 'X' }, { label: 'Y', names: ['rg'] }]).length === 1,
  '丢弃缺 names 的模块')

// groupPresent：只保留已有结果、空模块丢掉、漏网的归「其他」
const gp = M.groupPresent(['rg', 'msd', 'zzz'], M.FALLBACK_GROUPS)
ok(gp.length === 3, '只保留有结果的模块', gp.map(([l]) => l).join('/'))
ok(gp[0][0] === '链构象' && gp[0][1].length === 1, '链构象模块只含 rg')
ok(gp[1][0] === '动力学与输运' && gp[2][0] === '其他', '未归类的进「其他」')
ok(gp[2][1][0] === 'zzz', '「其他」里是 zzz')
ok(M.groupPresent([], M.FALLBACK_GROUPS).length === 0, '没有结果时不显示任何模块')

// 图表导航的分层：模块 → 分析项 → 各张图
h = chartPanel(run, { name: 'rg', index: 0 }, { groups: M.FALLBACK_GROUPS })
ok(h.includes('navmodule'), '导航有模块层')
ok(h.includes('class="mtitle">链构象'), '模块标题=链构象')
ok(h.includes('class="mtitle">动力学与输运'), '模块标题=动力学与输运')
ok(h.includes('1 项 / 2 张'), '模块显示「N 项 / M 张」')
ok(h.includes('navgroup'), '模块内还有分析项一层')
ok(h.includes('回转半径 Rg —— 最大链'), '分析项标题仍在')
ok(h.includes('Rg 随时间变化'), '面板项仍在')
ok(!h.includes('class="mtitle">界面'), '没有结果的模块不显示（界面未算出）')
ok(h.includes('链构象 ▸ 回转半径 Rg —— 最大链 ▸ Rg 随时间变化'),
  'title 里带上模块路径')

// 没有 groups 时退化为兜底分组，不能崩
h = chartPanel(run, { name: 'rg', index: 0 })
ok(h.includes('navmodule') && h.includes('class="mtitle">链构象'),
  'ChartPanel 不传 groups 时也能分层（用兜底）')

console.log('\n' + '='.repeat(74))
console.log(`通过 ${pass} 项` + (fails.length
  ? `，失败 ${fails.length} 项: ${fails.join(' | ')}` : '，全部通过'))
console.log('='.repeat(74))

// 同 _e2e*.mjs：不要用 process.exit()，否则 undici 的 keep-alive 套接字
// 正在关闭时会触发 libuv 断言，退出码变成 0xC0000409（假阴性）
process.exitCode = fails.length ? 1 : 0
