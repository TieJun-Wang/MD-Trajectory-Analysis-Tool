/**
 * Web 版主界面。
 *
 * 布局（整页一屏，无页面级滚动）：
 *   顶栏
 *   ├─ 左操作台：三块**互斥折叠**面板，一次只展开一块
 *   │    1. 文件读取
 *   │    2. 体系信息
 *   │    3. 分析设置（分析对象 + 参数设置 + 分析功能 + 运行）
 *   └─ 右结果区：标签页（图表 / 数据表 / 统计量·导出 / 运行日志）
 *
 * 所有分析都在后端做，前端只负责发请求和用 ECharts 画图。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { normalizeGroups } from './analysisGroups'
import { fmtSec } from './format'
import OpenPanel from './components/OpenPanel'
import InfoPanel from './components/InfoPanel'
import DataPanel from './components/DataPanel'
import StatsPanel from './components/StatsPanel'
import ExportPanel from './components/ExportPanel'
import LogPanel from './components/LogPanel'
import QcPanel from './components/QcPanel'
import ChartPanel from './components/ChartPanel'
import NotesPanel from './components/NotesPanel'
import StructurePanel from './components/StructurePanel'
import StructureView from './components/StructureView'
import { useStructureState } from './structureState'
import {
  FunctionBlock,
  FrameBlock,
  ParamBlock,
  RunBlock,
  SelectionBlock,
  fmtElapsed,
} from './components/SetupPanel'

//: 结果区标签页。统计量与导出是**两个同级的独立页面**：
//: 统计量只管显示指标，导出只管导出，互不掺在一起。
const TABS = [
  ['struct', '三维结构'],
  ['charts', '图表'],
  ['data', '数据表'],
  ['stats', '统计量'],
  ['qc', '科研 QC'],
  ['export', '导出'],
  ['log', '运行日志'],
]

//: 三块折叠面板的 id
const SEC_FILE = 'file'
const SEC_INFO = 'info'
const SEC_STRUCT = 'struct'
const SEC_SETUP = 'setup'

export default function App() {
  const [health, setHealth] = useState(null)
  const [sess, setSess] = useState(null)
  // 当前标签页：F5 之后**回到原来那一页**（用户实测：一刷新就跳到「图表」）
  const [tab, setTab] = useState(() => {
    try {
      const saved = window.localStorage.getItem('mdta.tab')
      return TABS.some(([k]) => k === saved) ? saved : 'charts'
    } catch { return 'charts' }
  })
  useEffect(() => {
    try { window.localStorage.setItem('mdta.tab', tab) } catch { /* 忽略 */ }
  }, [tab])
  const [log, setLog] = useState([])

  // 折叠面板：一次只展开一块
  // 哪个小块展开也要跨刷新保持（用户在「分析设置」里刷新，不该跳回体系信息）
  const [open, setOpen] = useState(() => {
    try {
      const saved = window.localStorage.getItem('mdta.open')
      return saved || SEC_FILE
    } catch { return SEC_FILE }
  })
  useEffect(() => {
    try { window.localStorage.setItem('mdta.open', open || '') } catch { /* 忽略 */ }
  }, [open])
  const toggle = (id) => setOpen((cur) => (cur === id ? '' : id))

  const [frames, setFrames] = useState({
    start_ps: '', stop_ps: '', equil_ps: '', interval_ps: '', max_frames: '',
  })
  const [params, setParams] = useState({
    cutoff: 5.0, axis: 'all', nbins: '', rmax: 12, dihedral_mode: 'auto',
    // 密度口径：mass = 质量密度 g/cm³（Forcite 默认）/ number = 数密度 1/Å³
    density_mode: 'mass',
    // MSD 默认按**原子**追踪（= Forcite / MDAnalysis EinsteinMSD 口径）；
    // 「分子质心」会大一圈（分子内振动/转动被去掉），两边不能直接比。
    // RDF 设置只留三项（用户要求）：最大 r / 分箱数 / 配对模式；
    // 默认 12 / 120 / 总体（total）。
    rdf_mode: 'total', rdf_nbins: 120, msd_object: 'atom',
    rdf_distance: 'minimage', rdf_drop_overlap: '', rdf_rho_scale: '',
    // VACF 的 τ 上限（帧）；留空 = 全段（与 Forcite 一样逐 lag 全给）
    vacf_max_lag: '',
    gauche_edge: 120,
    // 二面角分布柱宽 = Forcite 的 Bin width（度）：网格端点 = ±(180 + bin)。
    // bin=1 → −181…181，bin=0.5 → −180.5…180.5（与 Forcite 逐格一致）
    bin_width: 1,
    // 「减去整体漂移」默认**关**：Materials Studio Forcite 的 MSD 不去漂移，
    // 默认关掉才能与它逐点对齐（实测 PE50：关掉差 0.000%，开着差 −0.01…−0.19%）。
    // NPT 体系里整盒缓慢平移会污染 MSD，那时再勾上。
    msd_drift: false,
    // 取向：链段定义 / 几何骨架 stride / **P₂ 参考轴**（x / y / z=默认 / xyz 三者）
    orient_mode: 'repeat', orient_stride: 1, orient_p2_axis: 'z',
    contact_mode: 'inter', g_ref: '',
    // Rg：「分析对象」是**二选一**（整体体系 / 分子·链，默认分子·链），程序只算
    // 选中的那种，不再同时算整组与逐分子两套。`per_molecule` 现在只归 R_ee 用。
    mass_weighted: true, rg_object: 'molecule', per_molecule: true,
    ree_ends: 'bond_graph', ree_atoms: '',
    // BOO / 晶体识别：邻域半径留空 = 自动（按最近邻距离中位数推定）
    boo_cutoff: '', boo_averaged: true, q6_solid: 0.5, min_cluster: 10,
    // 面内 RDF (2D) / 两组分质心距
    // 质心距默认**用最小镜像**（PBC），见 groups 里的「质心距 PBC」勾选框
    rdf2d_rmax: 70, rdf2d_nbins: 140, slab_lo: '', slab_hi: '', comdist_pbc: true,
  })
  const [primary, setPrimary] = useState({ mode: 'auto' })
  //: **真正决定参与分析**的组分（第 4 部分「参与分析的对象」勾选）
  const [components, setComponents] = useState([])
  //: 第 3 部分（结构选择）里**不展示**的组分：只影响第 4 部分的可选列表，
  //: 不直接决定谁参与分析（用户实测：在第 3 部分没勾的组分，RDF 还是会跑出来）
  const [hiddenComps, setHiddenComps] = useState([])
  // 分析集合（Set）：定义 + 后端预览（原子数 / 分子数 / 追踪方式）
  /** 第 3 部分的组分勾选：只决定**第 4 部分是否展示**该组分；
   *  取消展示时顺手把它从「参与分析」里摘掉（看不见的组分不该还在算）。 */
  const toggleShownComp = (name) => {
    const hiding = !hiddenComps.includes(name)
    setHiddenComps((prev) => (hiding ? [...prev, name] : prev.filter((c) => c !== name)))
    if (hiding) {
      setComponents((prev) => prev.filter((c) => c !== name))
      setPrimary((cur) => (cur && cur.mode === 'component' && cur.name === name
        ? { mode: 'auto' } : cur))
    }
  }

  const [sets, setSets] = useState([])
  const [setsInfo, setSetsInfo] = useState([])

  const [titles, setTitles] = useState({})
  const [order, setOrder] = useState([])
  const [which, setWhich] = useState([])
  const [groups, setGroups] = useState([])
  //: 分析项可行性预判（后端 /select 一起返回）：跑不了的项在界面上置灰 + 说明原因
  const [feasible, setFeasible] = useState({})
  //: 数据能力检测（坐标/轨迹/键/质量/电荷/分子归属/周期盒 + 静态 or 动态模式）
  const [caps, setCaps] = useState(null)

  const [run, setRun] = useState(null)
  const [chart, setChart] = useState(() => {
    try {
      const raw = window.localStorage.getItem('mdta.chart')
      if (raw) {
        const v = JSON.parse(raw)
        if (v && typeof v.name === 'string') {
          return { name: v.name, index: Number(v.index) || 0 }
        }
      }
    } catch { /* 忽略 */ }
    return { name: '', index: 0 }
  })
  useEffect(() => {
    try {
      window.localStorage.setItem('mdta.chart', JSON.stringify({
        name: chart?.name || '', index: Number(chart?.index) || 0,
      }))
    } catch { /* 忽略 */ }
  }, [chart?.name, chart?.index])
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  //: 开跑前先预估各项用时（每项先跑几帧实测），再按短→长排序
  const [estimateOn, setEstimateOn] = useState(true)
  const [live, setLive] = useState(null)
  const [error, setError] = useState('')
  // 折叠状态要**跨刷新保持**（用户：F5 之后操作台/说明栏都被自动展开了）
  const readBool = (key, dflt) => {
    try {
      const v = window.localStorage.getItem(key)
      return v == null ? dflt : v === '1'
    } catch { return dflt }
  }
  const writeBool = (key, v) => {
    try { window.localStorage.setItem(key, v ? '1' : '0') } catch { /* 隐私模式 */ }
  }
  const [opsCollapsed, setOpsCollapsed] = useState(() => readBool('mdta.opsCollapsed', false))
  const [notesCollapsed, setNotesCollapsed] = useState(() => readBool('mdta.notesCollapsed', false))
  useEffect(() => { writeBool('mdta.opsCollapsed', opsCollapsed) }, [opsCollapsed])
  useEffect(() => { writeBool('mdta.notesCollapsed', notesCollapsed) }, [notesCollapsed])
  // 当前图表里**仍然显示**的曲线（由图例点选决定）：说明区据此动态过滤
  const [visibleCurves, setVisibleCurves] = useState(null)
  //: 每个（分析项:面板）各自记住图例里当前显示的曲线 —— 切页签/换图后再回来
  //: 依然能取到，数据表的「只看当前图例」就靠它（用户实测：切到数据表后失效）。
  const legendByKeyRef = useRef({})
  const abortRef = useRef(null)
  const liveRunRef = useRef(null)
  const cancelledRef = useRef(false)
  //: 已经写进日志的「跳过」项（后端每轮轮询都会重复回传 skipped，只记一次）
  const skipLoggedRef = useRef(new Set())
  //: 选择同步计数：每成功把「主链 + 组分 + 集合」发给后端一次就 +1。
  //: 可行性预判读的是**后端**状态，所以同步完成后要让它重新问一次
  //: （否则界面上勾了组分、预判里仍是"没勾组分"，RDF 一直置灰 —— 用户实测的 bug）。
  const [selSync, setSelSync] = useState(0)
  const selKeyRef = useRef('')

  /**
   * **选择同步**：把"主链 + 参与分析的组分 + 集合"整份发给后端。
   *
   * 为什么必须同步（用户实测的真 bug）：可行性预判、`/capabilities`、以及任何
   * 读 `/select` 之后状态的接口，都是读**后端**的组分/主链。以前只在「开始分析」
   * 时把选择塞进请求体，于是界面上"分析设置 → 分析对象"里勾了组分，预判里却
   * 还是"没勾任何组分"，RDF / 密度 / 接触 这些就一直置灰 —— 看着像"选了也跑不了"。
   *
   * 由下面那个防抖 effect 触发（打字/连点不会每个动作都发一次请求）。
   * 注意：这里故意用普通函数而不是 useCallback —— 它依赖 pushLog，
   * 而 pushLog 定义在后面，useCallback 的依赖数组会在渲染期访问未初始化的
   * const（之前踩过 `Cannot access before initialization`）。
   */
  const syncSelection = async (payload) => {
    const r = await api.select(sess.sid, payload)
    setSetsInfo(r?.sets || [])
    // 可行性**不在这里设**：统一由下面那个 effect 取（它还必须带帧选择），
    // 两个来源写同一个 state 会互相覆盖 —— 正是"预判没生效"的一半原因。
    // 但同步完成后要**戳一下**那个 effect：它读的是后端状态，刚变过。
    setSelSync((n) => n + 1)
    if (r?.deferred) {
      pushLog('分析正在运行：选择已保存，下一次运行分析时生效')
    }
    if (r?.catalog || r?.components) {
      setSess((cur) => (cur ? {
        ...cur,
        // 组件胶囊 / 主链下拉框读**候选清单**（catalog）：没勾的条目也要在，
        // 否则取消勾选后整张表就空了（用户实测）。
        components: r.catalog || cur.components,
        // 「自动」主链会跟着勾选的集合走（后端 apply_selection 的约定），
        // 这里把新的标签带回来，界面上"自动（集合 set:C …）"才是真的
        primary_label: r.primary_label ?? cur.primary_label,
        primary_atoms: r.primary_atoms ?? cur.primary_atoms,
      } : cur))
      // 新集合默认参与分析：补进 components（胶囊里会显示成选中态）。
      // 已经是"关"状态的集合**不要**补（否则界面上关了、后端还在算）。
      const off = new Set((payload.sets || [])
        .filter((x) => x.on === false).map((x) => x.name))
      const setNames = (r.components || []).map((c) => c.name)
        .filter((n) => n.startsWith('set:') && !off.has(n.slice(4)))
      setComponents((prev) => [...prev,
        ...setNames.filter((n) => !prev.includes(n))])
    }
    return r
  }

  /**
   * 分析集合变更：只改本地状态（+ 立刻在三维视图里高亮），
   * 后端同步交给下面的防抖 effect —— 这样"勾选框 / 主链 / 组分胶囊 / 集合"
   * 走的是**同一条**同步路径，不会出现某一条路忘了通知后端。
   */
  /**
   * 给已定义的集合改名（重名时后端自动加序号 `_2`/`_3`…）。
   * 名字在四处都是键（components / 主链 / 集合定义 / 已保存选择），所以改完
   * 要把本地这四份一起改写，否则防抖同步会立刻把旧名字又发回后端。
   */
  const renameSet = async (oldName, newName) => {
    const sid = sess?.sid
    if (!sid) return
    const nm = String(newName || '').trim()
    if (!nm || nm === oldName) return
    try {
      const r = await api.renameSet(sid, oldName, nm)
      const fin = r?.name || nm
      setSets((prev) => prev.map((x) => (x.name === oldName ? { ...x, name: fin } : x)))
      if (r?.sets) setSetsInfo(r.sets)
      setComponents((prev) => prev.map((c) => (c === `set:${oldName}` ? `set:${fin}` : c)))
      setPrimary((cur) => (cur && cur.name === `set:${oldName}`
        ? { ...cur, name: `set:${fin}` } : cur))
      setSelSync((n) => n + 1)
      pushLog(`集合改名：set:${oldName} → set:${fin}`
        + (r?.renamed ? `（原名已存在，自动加序号）` : ''))
    } catch (e) {
      setError(`集合改名失败: ${e.message}`)
    }
  }

  const applySets = (next, highlight = '') => {
    setSets(next)
    if (!sess?.sid) return
    // 新建/更新的集合 → 三维视图里**立刻高亮**（用户要求：点 ＋C 要标红）
    if (highlight) struct.highlightSet(highlight)
  }

  /**
   * 三维「结构选择」里创建 / 载入 / 删除集合之后，把后端的状态接回本地：
   * · 组分清单（set:<名字> 已由后端变成参与分析的组分）；
   * · 集合预览（setsInfo）与已保存选择（sess.selections）；
   * · 本地 `sets` 里补一条 —— 可视化集合**只存名字**，原子索引一直在后端
   *   （`Session.apply_selection` 会用 `self.selections` 把它补回去）。
   */
  const onVisualSelection = (r) => {
    if (!r) return
    const sel = r.selection || {}
    if (sel.components || sel.catalog) {
      setSess((cur) => (cur ? { ...cur, components: sel.catalog || cur.components } : cur))
    }
    if (sel.sets) setSetsInfo(sel.sets)
    if (r.selections) {
      setSess((cur) => (cur ? { ...cur, selections: r.selections } : cur))
      // 删除后同步掉本地那条（后端已把它从组分里摘掉）
      // ⚠️ 这里**必须**同时看两个来源（用户实测 bug：建了第二个集合，第一个就没了）：
      //   · r.components  —— 只在 `/select` 的响应里有；`/selection`（新建/加载
      //     一个集合）的响应顶层只有 set / selection / selections。
      //     以前直接读 `r.components` 拿到 undefined → live 为空 → 把本地**所有**
      //     集合都过滤掉了，于是"选一个再选下一个，前一个被覆盖"。
      //   · r.selection.components —— 新建走的是这一层（apply_selection 的返回）。
      //   · r.selections —— 后端仍然保存着的可视化选择（含已关掉、不参与分析的
      //     集合）。已关掉的集合不在 components 里，但**不能因此删掉**。
      const compsResp = r.components
        || (r.selection && r.selection.components) || []
      const live = new Set(compsResp.map((x) => (typeof x === 'string' ? x : x.name)))
      const saved = new Set((r.selections || []).map((x) => x.name))
      setSets((prev) => prev.filter((x) => (live.has(`set:${x.name}`)
        || saved.has(x.name))))
    }
    if (r.set?.name) {
      const nm = String(r.set.name).replace(/^set:/, '')
      setSets((prev) => (prev.some((x) => x.name === nm) ? prev
        : [...prev, { name: nm, kind: 'visual', on: true,
          ...(r.set.from ? { from: r.set.from } : {}) }]))
      setComponents((prev) => (prev.includes(r.set.name) ? prev : [...prev, r.set.name]))
    }

  }

  const pushLog = useCallback((line) => {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    setLog((prev) => [...prev.slice(-300), `[${ts}] ${line}`])
  }, [])

  /**
   * 「结构选择」的共享状态：左侧操作台与中间「三维结构」页都用这一份
   * （三维画面在中间显示区，控制项在左侧 —— 两边必须看到同一份选择）。
   */
  const struct = useStructureState({ sess, log: pushLog,
    onSetsChanged: onVisualSelection })

  // 刷新恢复：把 sid 存在 localStorage，F5 之后据此把整个界面接回去
  const SID_KEY = 'mdta.sid'

  // 轮询一次并合并结果（doRun 与"刷新后接着跑"共用）
  const pumpOnce = async (sid, acc, got) => {
    const p = await api.runProgress(sid, got)
    let nextGot = got
    const fresh = Object.entries(p.results || {})
    for (const [name, res] of fresh) acc.results[name] = res
    nextGot += fresh.length
    const merged = {
      results: { ...acc.results },
      timings: p.timings || acc.timings || {},
      titles: p.titles || acc.titles || {},
      frames: p.frames || acc.frames || null,
      summary_rows: p.summary_rows || acc.summary_rows || [],
      elapsed_sec: p.elapsed_sec,
    }
    setRun(merged)
    setProgress(p.frac || 0)
    setLive({
      status: p.status, message: p.message, nDone: p.n_done,
      nTotal: p.n_total, pending: p.pending || [], elapsed: p.elapsed_sec,
      current: p.current || '',
      phase: p.phase || '',
      estimates: p.estimates || {},
      estimateNote: p.estimate_note || '',
    })
    return { p, nextGot, node: merged }
  }

  /** F5 之后：若后端还活着（会话还在），把界面接回去并继续轮询。 */
  const restoreSession = useCallback(async (sid) => {
    const st = await api.sessionState(sid)
    setSess({
      sid,
      info: st.info,
      load_notes: st.load_notes || [],
      // ⚠️ 这里必须是**候选清单**（catalog），不是"当前勾选的组分"：
      // 用户实测 —— 第一轮没勾任何自动识别的组分（只跑了 MSD），刷新后
      // 组分胶囊全空、第二轮没得选。catalog 才是不随勾选变化的完整清单。
      components: st.catalog || st.components || [],
      chains: st.chains || [],
      // 下面这些是「分析对象 / 分析集合」界面要用的（刷新后也得在）
      primary_label: st.selection?.primary_label || '',
      primary_atoms: st.selection?.primary_atoms || 0,
      elements: st.elements || [],
      set_presets: st.set_presets || [],
    })
    // 勾选状态：优先用 active（纯名字清单），老后端退回 selection.components
    setComponents(((st.selection?.active)
      || st.selection?.components || st.components || [])
      .map((c) => (typeof c === 'string' ? c : c.name)))
    // 分析集合（Set）：定义、预览、勾选状态都要接回来 —— 否则刷新一次
    // 用户刚建好的集合就"消失"了（用户实测反馈）。
    setSets((st.sets || []).map((x) => ({
      name: x.name,
      elements: x.elements || [],
      shape: x.shape || 'all',
      min_atoms: x.min_atoms ?? 0,
      max_atoms: x.max_atoms ?? 0,
      pure: !!x.pure,
      on: x.on !== false,
    })))
    setSetsInfo(st.sets_info || [])
    if (st.selection?.primary) setPrimary(st.selection.primary)
    const f = st.frames || {}
    setFrames((prev) => ({ ...prev,
      start_ps: f.start_ps ?? prev.start_ps,
      stop_ps: f.stop_ps ?? prev.stop_ps,
      equil_ps: f.equil_ps ?? prev.equil_ps,
      interval_ps: f.interval_ps ?? prev.interval_ps,
      max_frames: prev.max_frames,
    }))
    const results = st.results || {}
    setRun({
      results,
      timings: st.timings || {},
      titles: st.titles || {},
      frames: st.frames || null,
      summary_rows: st.summary_rows || [],
      elapsed_sec: st.elapsed_sec || 0,
    })
    const names = Object.keys(results)
    if (names.length) {
      // 只有"存档里没有选过图"时才退回第一项；否则保持刷新前那张
      let savedChart = null
      try {
        const raw = window.localStorage.getItem('mdta.chart')
        savedChart = raw ? JSON.parse(raw) : null
      } catch { savedChart = null; }
      if (!(savedChart && results[savedChart.name])) {
        setChart({ name: names[0], index: 0 })
      }
      pushLog(`已恢复上次的会话：${names.length} 项结果（${st.sid}）`)
    } else {
      pushLog(`已恢复上次打开的体系（还没有分析结果）`)
    }
    // 刷新要停在原来展开的那一块，不再强制 setOpen(SEC_INFO)

    // 后端还在跑 → 接着轮询；只有「取消」才会让它停
    const job = st.job || {}
    if (job.status === 'running') {
      pushLog('后端仍在计算中，已重新接上进度…')
      setBusy(true)
      let acc = {
        results: { ...results }, timings: st.timings || {},
        titles: st.titles || {}, frames: st.frames || null,
        summary_rows: st.summary_rows || [],
      }
      // 索引以服务端的 completion 长度为准（= 后端认为"已经交付过"的项数）
      let got = (job.completion || []).length || names.length
      try {
        for (;;) {
          await new Promise((r) => setTimeout(r, 500))
          const { p, nextGot, node } = await pumpOnce(sid, acc, got)
          acc = node
          got = nextGot
          if (p.status !== 'running') {
            if (p.status === 'error') throw new Error(p.error || '分析失败')
            break
          }
        }
        pushLog('恢复后的计算已完成')
      } catch (e) {
        setError(e.message)
        pushLog(`恢复进度失败: ${e.message}`)
      } finally {
        setBusy(false)
      }
    } else if (job.message) {
      pushLog(`上次运行状态：${job.message}`)
    }
  }, [pushLog])

  useEffect(() => {
    api.health().then(setHealth).catch((e) => pushLog(`后端连接失败: ${e.message}`))
    api.analyses().then((a) => {
      setTitles(a.titles || {})
      setOrder(a.order || [])
      // 默认**一个都不勾**（用户要求）：只有自己选了才会分析 —— 免得一上来
      // 点「开始分析」就把 14 项全跑一遍（其中不少项当前体系根本跑不了）
      setWhich([])
      setGroups(normalizeGroups(a.groups))
      pushLog(`已连接后端，可用分析 ${a.order?.length || 0} 项`)
      // 上次的会话还在就接回去（F5 / 重新打开页面）
      let sid = null
      try { sid = window.localStorage.getItem(SID_KEY) } catch { sid = null }
      if (sid) {
        restoreSession(sid).catch(() => {
          try { window.localStorage.removeItem(SID_KEY) } catch { /* 忽略 */ }
          pushLog('上次的会话已失效（后端重启或超时），请重新读取文件')
        })
      }
      // 读取文件途中刷新 = 这次读取作废（用户定的方案）：顺手清掉可能残留的旧记录
      try { window.localStorage.removeItem('mdta.openPending') } catch { /* 忽略 */ }
    }).catch((e) => pushLog(`读取分析项目录失败: ${e.message}`))
  }, [pushLog])

  const onOpened = (data) => {
    try { window.localStorage.setItem(SID_KEY, data.sid) } catch { /* 隐私模式忽略 */ }
    setSets([])          // 新体系：清掉上一次定义的集合
    setSetsInfo([])
    // `components` 用**候选清单**（catalog）：它是界面上可勾选的全部条目，
    // 与"现在勾了哪几个"（下面的 setComponents）是两回事。
    setSess({ ...data, components: data.catalog || data.components || [] })
    setRun(null)
    setLive(null)
    setComponents((data.components || []).map((c) => c.name))
    setPrimary({ mode: 'auto' })
    setError('')
    setOpen(SEC_INFO)                 // 打开后自动切到「体系信息」
  }

  // 左右两栏互斥：展开一边就把另一边收起（屏幕宽度有限，两个都开着图表就太窄了）
  const toggleOps = () => {
    const next = !opsCollapsed
    setOpsCollapsed(next)
    if (!next) setNotesCollapsed(true)      // 展开操作台 → 收起说明栏
  }
  const toggleNotes = () => {
    const next = !notesCollapsed
    setNotesCollapsed(next)
    if (!next) setOpsCollapsed(true)        // 展开说明栏 → 收起操作台
  }
  // 跑完分析后：自动收起操作台、展开说明栏，把注意力交给图表与解释
  const focusChart = () => {
    setOpsCollapsed(true)
    setNotesCollapsed(false)
  }

  const doRun = async () => {
    if (!sess) { setError('请先读取体系'); return }
    if (!which.length) {
      setError('请至少选择一个分析项（勾选左侧「分析功能」里的复选框，或点「仅 XX」预设）')
      return
    }
    setBusy(true); setError(''); setProgress(0); setLive(null)
    // 用户要求：点「开始分析」立刻切到图表区（不要停在三维结构页等结果）
    setTab('charts')
    setVisibleCurves(null)
    skipLoggedRef.current = new Set()      // 新一轮：跳过项重新记录
    cancelledRef.current = false
    const t0 = performance.now()
    const existingNames = Object.keys(run?.results || {})
    pushLog(existingNames.length
      ? `开始分析 ${which.length} 项（已有 ${existingNames.length} 项结果，`
        + '组分 / 帧 / 参数没变的保留，变了的自动重算）…'
      : `开始分析 ${which.length} 项（实时刷新）…`)

    // 一点「开始分析」就让位给图表：立刻收起操作台、展开说明栏。
    // 进度改由顶栏的实时指示器显示，所以收起操作台后不会看不到进度。
    focusChart()

    liveRunRef.current = { cancelled: false }

    // R_ee 手工链端：形如 "0, 3034"（逗号/空格分隔）；不是两个非负整数就当留空，
    // 由后端的「链端来源」自动判定（bond_graph = 键连图端基）。
    // 可选数值：留空 → null（后端按默认/自动处理），否则转成数字
    const optNum = (v) => (v === '' || v == null ? null : Number(v))

    // 叶层范围：两个都填才生效（rdf2d 的 slab），任一留空 = 全盒
    const slabRange = (() => {
      const lo = optNum(params.slab_lo)
      const hi = optNum(params.slab_hi)
      return (lo === null || hi === null) ? null : [lo, hi]
    })()

    const reeAtoms = (() => {
      const raw = String(params.ree_atoms ?? '').trim()
      if (!raw) return null
      const nums = raw.split(/[^0-9]+/).filter((s) => s !== '').map(Number)
      return (nums.length === 2 && nums.every((n) => Number.isInteger(n) && n >= 0))
        ? nums : null
    })()

    // 已有结果时**默认追加**：只补跑本次新勾的项，旧结果原样保留
    // （用户："跑完几项之后想补加别的功能，在原有基础上做扩展"）
    const existing = Object.keys(run?.results || {})
    const body = {
      which,
      extend: existing.length > 0,
      // 先跑几帧预估各项用时，再按短→长排序开跑（可在「运行」里关掉）
      estimate: estimateOn,
      frames: {
        start_ps: frames.start_ps || null,
        stop_ps: frames.stop_ps || null,
        equil_ps: frames.equil_ps || null,
        interval_ps: frames.interval_ps || null,
        max_frames: frames.max_frames || null,
      },
      params: {
        // 密度默认三个方向一起算（C100/C010/C001）；其它模块仍用单个轴
        density: { axis: (params.axis === 'all' ? 'all' : Number(params.axis)),
                   // 留空 = 自适应（0.5 Å/箱，与 Forcite 一致）
                   nbins: params.nbins ? Number(params.nbins) : null,
                   mode: params.density_mode === 'number' ? 'number' : 'mass' },
        interface: { axis: (params.axis === 'all' ? 2 : Number(params.axis)),
                     nbins: Math.max(Number(params.nbins), 60) },
        // RDF 分箱数以前写死 120 —— Forcite 是按「间隔(Å)」给的（如 20 Å / 0.02 Å
        // = 1000 箱），界面能设分箱数，两边才能逐 bin 对比。
        rdf: { rmax: Number(params.rmax),
               nbins: Math.max(2, Number(params.rdf_nbins) || 120),
               mode: params.rdf_mode,
               // Forcite 口径旋钮已从界面移除（后端仍保留参数，默认即
               // MDAnalysis InterRDF 口径，不再从界面下发）。
               forcite: false },
        // `remove_drift`：默认**关**（= Materials Studio Forcite 的口径，实测逐点一致）。
        // NPT 体系里整盒平移会把 MSD 抬起来，那时把「减去整体漂移」勾上。
        // `unwrap`：不再让用户选，按轨迹来源自动定 ——
        //   MS 的 .trj 坐标本来就是**展开的**（跨盒边的分子用 >L 的坐标）→ 不补；
        //   GROMACS 的 xtc 等一般折回盒内 → 默认补最小镜像。
        msd: { object: params.msd_object,
               remove_drift: params.msd_drift === true,
               unwrap: !/\.trj$/i.test(String(sess?.info?.trajectory || '')) },
        // VACF：max_lag 留空（null）= 用满整条轨迹
        vacf: { max_lag: optNum(params.vacf_max_lag) },
        contact: { cutoff: Number(params.cutoff), mode: params.contact_mode },
        dihedral: { mode: params.dihedral_mode,
                    gauche_edge: Number(params.gauche_edge) || 120,
                    // 分布柱宽 bin（度）：留空/非法 → 1°
                    bin_width: Number(params.bin_width) || 1 },
        orientation: { mode: params.orient_mode,
                       stride: Number(params.orient_stride) || 1,
                       // P2 相对哪个参考轴：x / y / z（默认）/ xyz（三者都算）
                       p2_axis: ['x', 'y', 'z', 'xyz'].includes(params.orient_p2_axis)
                         ? params.orient_p2_axis : 'z' },
        // 结构有序度的局部结构分量：留空 → g_ref=None（该分量不计入指数）
        order: { g_ref: String(params.g_ref).trim() === ''
          ? null : Number(params.g_ref),
          // 与「二面角分析」共用同一个 trans/gauche 阈值（后端两个模块都收这个键）
          gauche_edge: Number(params.gauche_edge) || 120 },
        // Rg：「分析对象」二选一（互斥）—— object="molecule" 逐分子算 Rg（默认），
        // object="system" 整组算 Rg。后端只算选中的那种。
        rg: { mass_weighted: params.mass_weighted !== false,
              object: params.rg_object === 'system' ? 'system' : 'molecule' },
        // 链端来源三选一：bond_graph / residue / custom（custom 由 atom_indices 生效）
        ree: { ends: (params.ree_ends === 'residue' ? 'residue' : 'bond_graph'),
               per_molecule: params.per_molecule !== false,
               // 手工指定两个成键原子作为链端（留空 = None，按 ends 自动判定）
               atom_indices: reeAtoms },
        // 键取向序 BOO / 晶体-非晶识别：邻域半径留空 → null（后端自动推定）
        boo: { cutoff: optNum(params.boo_cutoff), averaged: params.boo_averaged !== false },
        crystal: { cutoff: optNum(params.boo_cutoff),
                   q6_solid: Number(params.q6_solid) || 0.5,
                   min_cluster: Number(params.min_cluster) || 10 },
        // 面内 RDF (2D)：slab 留空 = 全盒
        // 两组分质心距：**默认用最小镜像**（PBC），取消勾选才退回"包裹坐标相减"
        rdf2d: { plane_axis: (params.axis === 'all' ? 2 : Number(params.axis)) ?? 2,
                 rmax: Number(params.rdf2d_rmax) || 70,
                 nbins: Number(params.rdf2d_nbins) || 140,
                 slab: slabRange },
        comdist: { axis: (params.axis === 'all' ? 2 : Number(params.axis)) ?? 2,
                   pbc: params.comdist_pbc !== false },
      },
      selection: {
        primary,
        // 集合参与与否用 `set:<名字>` 表达（后端严格按此过滤）。
        // `components` 里可能残留旧集合名（同步时已过滤），这里统一再过滤一遍，
        // 免得"界面上关掉的集合"被当成参与者。
        components: [...components.filter((c) => !String(c).startsWith('set:')),
          ...sets.filter((x) => x.on !== false).map((x) => `set:${x.name}`)],
        sets: sets.map((x) => ({
          name: x.name, on: x.on !== false,
          elements: x.elements || [], shape: x.shape || 'all',
          min_atoms: x.min_atoms || 0, max_atoms: x.max_atoms || 0,
          pure: !!x.pure,
        })),
      },
    }

    // 结果按完成顺序累积；后端只回传我们还没有的那几项。
    // 追加模式下先把已有结果带上，这样界面不会在开跑瞬间"清空"。
    let acc = {
      results: { ...(run?.results || {}) },
      timings: { ...(run?.timings || {}) },
      frames: run?.frames || null,
      titles: run?.titles || {},
      summary_rows: run?.summary_rows || [],
    }
    let got = existing.length

    const merge = (p, quiet = false) => {
      const fresh = Object.entries(p.results || {})
      if (fresh.length) {
        for (const [name, res] of fresh) acc.results[name] = res
        got += fresh.length
        // 第一项算完就切到图表页，让用户马上看到东西
        if (!quiet && got === fresh.length) {
          setChart({ name: fresh[0][0], index: 0 })
          setVisibleCurves(null)
          setTab('charts')
        }
        if (!quiet) {
          for (const [name] of fresh) {
            const panels = acc.results[name]?.panels?.length || 1
            pushLog(`  ✓ ${titles[name] || name} 完成（${panels} 张图）`)
          }
        }
      }
      // 「跳过」的项（体系不满足前提，例如没有晶胞信息做不了密度/界面）
      // 一次性写进日志 —— 它们不是错误，界面不该显示成"失败"。
      const skipped = p.skipped || {}
      for (const [n, why] of Object.entries(skipped)) {
        if (skipLoggedRef.current.has(n)) continue
        skipLoggedRef.current.add(n)
        pushLog(`  – ${titles[n] || n} 跳过：${why}`)
      }
      acc = {
        ...acc,
        results: { ...acc.results },
        timings: p.timings || acc.timings,
        titles: p.titles || acc.titles,
        frames: p.frames || acc.frames,
        summary_rows: p.summary_rows || acc.summary_rows,
        elapsed_sec: p.elapsed_sec,
      }
      setRun({ ...acc })
      setProgress(p.frac || 0)
      setLive({
        status: p.status, message: p.message, nDone: p.n_done,
        nTotal: p.n_total, pending: p.pending || [], elapsed: p.elapsed_sec,
        current: p.current || '',
        phase: p.phase || '',
        estimates: p.estimates || {},
        estimateNote: p.estimate_note || '',
      })
    }

    try {
      // 先拿 start 的返回：后端会告诉我们**保留了哪几项**（completion），
      // 以及本次要跑哪几项（order − completion）。输入变了的旧结果必须从本地
      // 丢掉，否则界面上留着的是按旧组分算的图（用户实测：加了组分却不跑）。
      const p0 = await api.runStart(sess.sid, body)
      {
        const keep = p0.completion || []
        const pend = new Set((p0.order || []).filter((n) => !keep.includes(n)))
        const dropped = Object.keys(acc.results).filter((n) => pend.has(n))
        for (const n of dropped) delete acc.results[n]
        if (dropped.length) {
          pushLog(`  这些项的设置变了，重算：${
            dropped.map((n) => titles[n] || n).join('、')}`)
        }
        // 索引以服务端的"已有结果数"为准（本地条数可能多出被作废的那几项）
        merge(p0, true)
        got = keep.length
      }

      // 轮询：后端每算完一项，这里就能拿到一项，页面边算边更新。
      // 取消后也继续轮询到后端把状态置为 cancelled —— 这样"当前这项"
      // 算完的结果不会丢（后端会在下一项开始前停下）。
      let fails = 0
      for (;;) {
        await new Promise((r) => setTimeout(r, 500))
        let p
        try {
          p = await api.runProgress(sess.sid, got)
          fails = 0
        } catch (e) {
          if (++fails >= 3) throw e
          continue
        }
        merge(p)
        if (p.status !== 'running') {
          if (p.status === 'error') throw new Error(p.error || '分析失败')
          break
        }
      }

      const dt = ((performance.now() - t0) / 1000).toFixed(1)
      const names = Object.keys(acc.results)
      const nCharts = names.reduce((a, n) => a + (acc.results[n].panels?.length || 1), 0)
      if (cancelledRef.current) {
        pushLog(`已取消：保留已完成的 ${names.length} 项（用时 ${dt}s）`)
      } else {
        pushLog(`完成：${names.length} 项分析 / ${nCharts} 张图，用时 ${dt}s`
          + (acc.frames ? `，${acc.frames.describe}` : ''))
        acc.frames?.notes?.forEach((n) => pushLog(`  帧选择: ${n}`))
        const slow = Object.entries(acc.timings || {}).sort((a, b) => b[1] - a[1]).slice(0, 3)
        if (slow.length) {
          pushLog('  最慢的三项: ' + slow.map(([k, v]) => `${k} ${v}s`).join('，'))
        }
      }
      if (!names.length) pushLog('  没有产出任何结果（检查组分选择与分析项）')
      setProgress(1)
    } catch (e) {
      setError(e.message)
      pushLog(`分析失败: ${e.message}`)
    } finally {
      liveRunRef.current = null
      cancelledRef.current = false
      setBusy(false)
      setLive((l) => (l ? { ...l, status: 'idle', pending: [] } : l))
    }
  }

  /** 清空已有结果（重置为"刚打开体系"的状态，便于整套重跑） */
  const clearResults = async () => {
    if (!sess) return
    try {
      await api.clearResults(sess.sid)
      setRun(null)
      setLive(null)
      setProgress(0)
      setError('')
      pushLog('已清空所有分析结果')
    } catch (e) {
      pushLog(`清空失败: ${e.message}`)
    }
  }

  const cancelRun = async () => {
    cancelledRef.current = true
    pushLog('已请求取消：正在中断当前计算（帧内分块也会检查，通常 1 秒内停下），已完成的结果会保留')
    try {
      const p = await api.runCancel(sess.sid)
      setLive((l) => (l ? { ...l, message: p.message || l.message } : l))
    } catch (e) {
      pushLog(`取消失败: ${e.message}`)
    }
  }

  const lastLine = useMemo(() => (log.length ? log[log.length - 1] : ''), [log])

  /**
   * **分析项可行性**：凡影响"能不能跑"的东西变了就重新问后端 ——
   * 体系、主链、勾选的组分、集合的参与开关、**帧选择**（把最大帧数设成 1，
   * MSD 就跑不了了）。以前只在集合变化时顺带取一次，导致很多时候
   * 预判根本没生效：选了 MSD 却一跑就报「至少需要 2 帧」（用户实测）。
   */
  // 能力只跟体系有关：换 sid 才重新问一次
  useEffect(() => {
    const sid = sess?.sid
    if (!sid) { setCaps(null); return undefined }
    let alive = true
    api.capabilities(sid)
      .then((r) => { if (alive) setCaps(r || null) })
      .catch(() => { if (alive) setCaps(null) })
    return () => { alive = false }
  }, [sess?.sid])

  /**
   * **把选择同步给后端**（防抖 350ms）。
   *
   * 三种触发都走这一条：组分胶囊点选、主链切换、集合增删/开关。
   * 后端拿到之后 `az.components / az.primary` 就与界面一致，
   * 于是 `/feasible`（可行性预判）与真正开跑时的口径相同 ——
   * 勾上组分后 RDF / 密度 / 接触 立刻可用（用户实测的 bug 就是这里断的）。
   */
  useEffect(() => {
    const sid = sess?.sid
    if (!sid) return undefined
    // 组分里不带 `set:`（集合的参与开关由 sets 的 on 决定，写两份会互相打架：
    // 关掉的集合会因为还留在 components 里而被后端继续当成参与者）
    const comps = (components || []).filter((c) => !String(c).startsWith('set:'))
    const on = (sets || []).filter((x) => x.on !== false).map((x) => `set:${x.name}`)
    const payload = {
      primary,
      components: [...comps, ...on],
      sets: (sets || []).map((x) => ({
        name: x.name, on: x.on !== false,
        elements: x.elements || [], shape: x.shape || 'all',
        min_atoms: x.min_atoms || 0, max_atoms: x.max_atoms || 0,
        pure: !!x.pure,
        // 这条集合"怎么来的"（范围 / 筛选条件）—— 后端原样存进 sets_spec，
        // 集合行的悬浮提示靠它
        ...(x.from ? { from: x.from } : {}),
      })),
    }
    const key = JSON.stringify(payload)
    if (key === selKeyRef.current) return undefined
    const timer = setTimeout(() => {
      selKeyRef.current = key
      syncSelection(payload).catch((e) => {
        setError(`选择同步失败: ${e.message}`)
      })
    }, 350)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sess?.sid, primary, components, sets])

  // 主链（分析对象）原子清单：给「R_ee 指定原子」做悬浮提示（序号 → 原子名）
  const [primaryAtoms, setPrimaryAtoms] = useState({ n_atoms: 0, names: [] })
  useEffect(() => {
    const sid = sess?.sid
    if (!sid) { setPrimaryAtoms({ n_atoms: 0, names: [] }); return undefined }
    let dead = false
    api.primaryAtoms(sid)
      .then((r) => { if (!dead && r) setPrimaryAtoms(r) })
      .catch(() => { /* 拿不到就只显示原子数，不打断界面 */ })
    return () => { dead = true }
  }, [sess?.sid, sess?.primary_label, selSync])

  const feasibleKey = useMemo(() => JSON.stringify([
    sess?.sid || '', primary, components,
    (sets || []).map((x) => [x.name, x.on !== false]), frames, selSync,
  ]), [sess?.sid, primary, components, sets, frames, selSync])

  useEffect(() => {
    const sid = sess?.sid
    if (!sid) { setFeasible({}); return undefined }
    let alive = true
    api.feasible(sid, frames)
      .then((r) => { if (alive) setFeasible((r && r.feasible) || {}) })
      .catch(() => { if (alive) setFeasible({}) })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feasibleKey])

  // 当前正在看的那张图：图表区与右侧「图表说明」都要用它，
  // 所以在这里统一算好再分别传下去（避免两边各算一套出现不一致）。
  const current = useMemo(() => {
    const results = run?.results || {}
    const names = Object.keys(results)
    if (!names.length) return null
    const result = results[chart.name] || results[names[0]]
    if (!result) return null
    const panelIndex = Math.max(0, Math.min(chart.index ?? 0,
      (result.panels?.length || 1) - 1))
    return { result, panelIndex }
  }, [run, chart])
  // 当前主链含多少个分子：单分子时「逐分子统计」没有意义（整组 Rg 就是该分子的 Rg），
  // 界面据此把开关置灰并写明原因。auto = 最大链（按构造必然是 1 个分子）；
  // 组分模式直接查组分表（后端已给出 n_molecules）；segid/自定义不猜，返回 null。
  const primaryMolecules = useMemo(() => {
    if (!sess) return null
    if (primary.mode === 'component' && primary.name) {
      const c = (sess.components || []).find((x) => x.name === primary.name)
      return c?.n_molecules ?? null
    }
    if (primary.mode === 'auto') return 1
    return null
  }, [sess, primary])

  // 顶栏第二行「当前分析项」：取后端记录的**分析名**再用标题表翻译成中文，  // 不去解析进度文字（那种做法一改文案就失效）。排队数放在同一行的括号里。
  const curTitle = live?.current
    ? ((run?.titles || {})[live.current] || live.current) : ''

  const sections = [
    { id: SEC_FILE, title: '1. 文件读取', hint: sess ? sess.info.topology.split(/[\\/]/).pop() : '' },
    {
      id: SEC_INFO, title: '2. 体系信息', disabled: !sess,
      hint: sess ? `${sess.info.n_atoms.toLocaleString()} 原子 / ${sess.info.n_frames} 帧` : '先读取体系',
    },
    {
      id: SEC_STRUCT, title: '3. 结构选择', disabled: !sess,
      hint: sess ? `${(sets?.length || 0)} 集合` : '先读取体系',
    },
    {
      id: SEC_SETUP, title: '4. 分析设置', disabled: !sess,
      hint: sess ? `${components.length} 组分 / ${which.length} 项分析` : '先读取体系',
    },
  ]

  return (
    <div className="app">
      <header className="topbar">
        {/* 左对齐：第一行加粗标题，第二行灰色小 2px 的英文副标题 */}
        <div className="brand">
          <div>
            <div className="t1">MD Trajectory Analysis Tool</div>
            <div className="t2">Molecular Dynamics Simulation Analysis Platform</div>
          </div>
        </div>
        {/* 右对齐：运行状态 + 版本 */}
        <div className="status">
          {/* 收起操作台后进度条在那边就看不见了 —— 顶栏这里始终显示。
              两行：第一行徽标 + 进度条 + n/N·耗时；第二行「当前分析项」。 */}
          {busy && (
            <span className="topbar-live"
                  title={`${live?.message || '正在计算…'}`
                    + (live?.pending?.length
                      ? `\n排队中：${live.pending
                        .map((n) => (run?.titles || {})[n] || n).join('、')}` : '')}>
              <span className="tbl-row">
                <span className="badge-live">● 实时更新中</span>
                <span className="tbl-bar">
                  <span className="tbl-fill"
                        style={{ width: `${Math.round((live?.frac ?? progress) * 100)}%` }} />
                </span>
                <span className="tbl-text">
                  {live?.nDone ?? 0}/{live?.nTotal ?? which.length}
                  {live?.elapsed ? ` · ${live.elapsed.toFixed(0)}s` : ''}
                </span>
              </span>
              <span className="topbar-cur">
                当前分析项：{curTitle || '准备中…'}
                {curTitle && live?.estimates?.[live.current] != null
                  ? `（预计 ${fmtSec(live.estimates[live.current])}）` : ''}
                {live?.pending?.length ? ` · 排队 ${live.pending.length} 项` : ''}
              </span>
            </span>
          )}
          <span className={health ? 'ok' : 'bad'}
                title={health
                  ? `后端在线${sess ? ` · 会话 ${sess.sid}` : ''}`
                  : '后端离线：连不上分析服务'}>
            ● status: {health ? 'Online' : 'Offline'}
          </span>
          <span className="ver" title={health ? `Python ${health.python}` : ''}>
            ● version: {health?.version || '—'}
          </span>
        </div>
      </header>

      <div className="layout">
        {/* ---------------------------------------------- 左：互斥折叠操作台 */}
        <aside className={`ops ${opsCollapsed ? 'collapsed' : ''}`}>
          {/* 最左侧的竖条：展开时是「收起」，收起时是「展开」，位置始终不变 */}
          <button className="rail rail-l"
                  title={opsCollapsed ? '展开左侧操作台（会收起右侧说明栏）'
                    : '收起操作台，让图表占满整屏'}
                  onClick={toggleOps}>
            <span className="rail-chev">{opsCollapsed ? '»' : '«'}</span>
            <span className="rail-text">{opsCollapsed ? '展开操作台' : '收起操作台'}</span>
          </button>

          <div className="acc">
            {sections.map((s) => {
              const isOpen = open === s.id
              return (
                <div key={s.id} className={`acc-item ${isOpen ? 'open' : ''}`}>
                  <button className="acc-head" disabled={s.disabled}
                          onClick={() => toggle(s.id)}>
                    <span className="caret">{isOpen ? '▼' : '▶'}</span>
                    <span>{s.title}</span>
                    <span className="hint" title={s.hint}>{s.hint}</span>
                  </button>

                  {isOpen && (
                    <div className="acc-body">
                      {s.id === SEC_FILE && (
                        <OpenPanel onOpened={onOpened} log={pushLog} />
                      )}
                      {s.id === SEC_INFO && sess && (
                        <InfoPanel info={sess.info} caps={caps}
                                   onNext={() => {
                                     // 「下一步」：直接打开结构选择并**自动渲染**结构图
                                     // （不再需要手动点"载入结构"）
                                     setOpen(SEC_STRUCT)
                                     setTab('struct')
                                     if (!struct.data && !struct.loading) struct.load()
                                   }} />
                      )}
                      {s.id === SEC_STRUCT && sess && (
                        <StructurePanel sess={sess} st={struct}
                                        sets={sets} setsInfo={setsInfo}
                                        onSetsChange={applySets}
                                        onRenameSet={renameSet}
                                        shownComps={(sess?.components || [])
                                          .map((c) => c.name)
                                          .filter((n) => !hiddenComps.includes(n))}
                                        onToggleShown={toggleShownComp}
                                        components={components}
                                        onToggleComponent={(name) => {
                                          // 集合的参与状态有两个入口：分析集合行里的
                                          // 勾选框（sets[].on）和「分析对象」里的胶囊
                                          // （components）。两者必须是**同一份状态** ——
                                          // 否则在胶囊上取消勾选集合后，开跑时
                                          // payload 又会按 sets[].on=true 把它加回来
                                          //（用户实测：没勾的集合照样进了 MSD）。
                                          const key = String(name)
                                          if (!key.startsWith('set:')) {
                                            setComponents((prev) => (prev.includes(key)
                                              ? prev.filter((c) => c !== key)
                                              : [...prev, key]))
                                            return
                                          }
                                          const nm = key.slice(4)
                                          setSets((prev) => prev.map((x) => (x.name === nm
                                            ? { ...x, on: x.on === false } : x)))
                                          setComponents((prev) => (prev.includes(key)
                                            ? prev.filter((c) => c !== key)
                                            : [...prev, key]))
                                        }}
                                        onNext={() => setOpen(SEC_SETUP)}
                                        log={pushLog} />
                      )}
                      {s.id === SEC_SETUP && sess && (
                        <>
                          <SelectionBlock sess={sess} primary={primary}
                                          setPrimary={setPrimary}
                                          components={components}
                                          setComponents={setComponents}
                                          hidden={hiddenComps}
                                          sets={sets} setsInfo={setsInfo}
                                          onSetsChange={applySets}
                                          onRenameSet={renameSet} />
                          <FrameBlock frames={frames} setFrames={setFrames} />
                          <FunctionBlock titles={titles} order={order}
                                         which={which}
                                         setWhich={(next) => {
                                           // 运行中把分析项取消掉 → 立刻打断当前计算
                                           const running = live?.status === 'running'
                                           const removed = which.filter((n) => !next.includes(n))
                                           if (running && removed.length) {
                                             pushLog(`已取消分析项：${removed.join('、')}`
                                               + '（当前计算会在 1 秒内中断，可重新开始）')
                                             cancelRun()
                                           }
                                           setWhich(next)
                                         }}
                                         groups={groups}
                                         feasible={feasible}
                                         estimates={live?.estimates} />
                          <ParamBlock params={params} setParams={setParams}
                                      which={which}
                                      primaryMolecules={primaryMolecules}
                                      nAtoms={Number(sess?.primary_atoms) || 0}
                                      primaryAtomInfo={primaryAtoms} />
                          <RunBlock busy={busy} progress={progress} onRun={doRun}
                                    onCancel={cancelRun} live={live}
                                    which={which} titles={titles} run={run}
                                    estimateOn={estimateOn}
                                    setEstimateOn={setEstimateOn}
                                    lastLine={lastLine}
                                    onClear={clearResults}
                                    onOpenLog={() => setTab('log')} />
                          {error && <div className="error">{error}</div>}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </aside>

        {/* -------------------------------------------------- 中：结果区 */}
        <main className="results">
          <nav className="tabs">
            {TABS.map(([k, label]) => (
              <button key={k} className={`tab ${tab === k ? 'active' : ''}`}
                      onClick={() => setTab(k)}>{label}</button>
            ))}
            {error && tab !== 'charts' && (
              <span className="error inline-error inlinetab">{error}</span>
            )}
          </nav>

          <div className="tabbody">
            {tab === 'struct' && <StructureView st={struct} />}
            {tab === 'charts' && (
              <ChartPanel onVisibleChange={(vis) => {
                // 记住"这张图（分析项 + 面板）"的图例选择，供数据表「只看当前图例」用。
                // 键同时写精确键与两个残缺键：图例事件上报时 `chart` 有可能还没就位
                // （那时 key 会变成 `:0`），两条路都留着才不会让数据表取不到选择。
                const nm = chart?.name || ''
                const idx = chart?.index ?? 0
                const ref = legendByKeyRef.current
                ref[`${nm}:${idx}`] = vis
                if (!nm) ref[`:${idx}`] = vis
                setVisibleCurves(vis)
              }}
                                 run={run} chart={chart} setChart={setChart}
                          live={live} titles={titles} groups={groups}
                          current={current?.result} panelIndex={current?.panelIndex ?? 0} />
            )}
            {/* 数据表 / 统计量 / 科研 QC 都跟着「图表里当前这张」走：
                在图表页选到 msd 的第 1 张图，切到数据表就是 msd 那一页的数据，
                统计量与科研 QC 会自动滚到对应分析项（用户要求）。 */}
            {/* 数据表跟随图表页：不仅同一张图，还要**同一批可见曲线**
                （图例里没勾的组不出现，用户要求） */}
            {tab === 'data' && <DataPanel run={run} groups={groups}
                                          chart={chart} setChart={setChart}
                                          visible={visibleCurves}
                                          legendByKey={legendByKeyRef.current}
                                          sid={sess?.sid || ''} />}
            {/* 统计量：只显示指标 */}
            {tab === 'stats' && <StatsPanel run={run} groups={groups}
                                            focus={chart?.name || ''} />}
            {/* 科研 QC：每项分析导出的检查项（PBC/样本/误差/峰显著性…） */}
            {tab === 'qc' && <QcPanel run={run} groups={groups}
                                      focus={chart?.name || ''} />}
            {/* 导出：只管导出 */}
            {tab === 'export' && (
              <ExportPanel run={run} sid={sess?.sid} log={pushLog} groups={groups}
                           focus={chart?.name || ''}
                           focusIndex={chart?.index ?? 0} />
            )}
            {tab === 'log' && (
              <LogPanel log={log} onClear={() => setLog([])} />
            )}
          </div>
        </main>

        {/* ------------------------------------- 右：图表说明（同样可折叠） */}
        {tab === 'charts' && current && (
          <aside className={`notes ${notesCollapsed ? 'collapsed' : ''}`}>
            <NotesPanel result={current.result} panelIndex={current.panelIndex}
                                 visibleCurves={visibleCurves} />
            {/* 右侧同理：竖条放在最右边，展开时是「收起」 */}
            <button className="rail rail-r"
                    title={notesCollapsed ? '展开图表说明（会收起左侧操作台）'
                      : '收起图表说明，让图表更宽'}
                    onClick={toggleNotes}>
              <span className="rail-chev">{notesCollapsed ? '«' : '»'}</span>
              <span className="rail-text">
                {notesCollapsed ? '展开说明' : '收起说明'}
              </span>
            </button>
          </aside>
        )}
      </div>
    </div>
  )
}
