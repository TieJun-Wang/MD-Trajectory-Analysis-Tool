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
const SEC_SETUP = 'setup'

export default function App() {
  const [health, setHealth] = useState(null)
  const [sess, setSess] = useState(null)
  const [tab, setTab] = useState('charts')
  const [log, setLog] = useState([])

  // 折叠面板：一次只展开一块
  const [open, setOpen] = useState(SEC_FILE)
  const toggle = (id) => setOpen((cur) => (cur === id ? '' : id))

  const [frames, setFrames] = useState({
    start_ps: '', stop_ps: '', equil_ps: '', interval_ps: '', max_frames: '',
  })
  const [params, setParams] = useState({
    cutoff: 5.0, axis: 2, nbins: 100, rmax: 12, dihedral_mode: 'auto',
    rdf_mode: 'inter', msd_object: 'molecule', gauche_edge: 120,
    orient_mode: 'repeat', orient_stride: 1, contact_mode: 'inter', g_ref: '',
    // Rg：「分析对象」是**二选一**（整体体系 / 分子·链，默认分子·链），程序只算
    // 选中的那种，不再同时算整组与逐分子两套。`per_molecule` 现在只归 R_ee 用。
    mass_weighted: true, rg_object: 'molecule', per_molecule: true,
    ree_ends: 'bond_graph', ree_atoms: '',
    // BOO / 晶体识别：邻域半径留空 = 自动（按最近邻距离中位数推定）
    boo_cutoff: '', boo_averaged: true, q6_solid: 0.5, min_cluster: 10,
    // 面内 RDF (2D) / 两组分质心距
    rdf2d_rmax: 70, rdf2d_nbins: 140, slab_lo: '', slab_hi: '', comdist_pbc: false,
  })
  const [primary, setPrimary] = useState({ mode: 'auto' })
  const [components, setComponents] = useState([])
  // 分析集合（Set）：定义 + 后端预览（原子数 / 分子数 / 追踪方式）
  const [sets, setSets] = useState([])
  const [setsInfo, setSetsInfo] = useState([])

  const [titles, setTitles] = useState({})
  const [order, setOrder] = useState([])
  const [which, setWhich] = useState([])
  const [groups, setGroups] = useState([])

  const [run, setRun] = useState(null)
  const [chart, setChart] = useState({ name: '', index: 0 })
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
  const abortRef = useRef(null)
  const liveRunRef = useRef(null)
  const cancelledRef = useRef(false)

  /**
   * 分析集合变更：立刻让后端算一次预览（原子数/分子数/追踪方式），
   * 同时刷新组分清单（集合是"一等公民组分"，RDF/接触/界面/MSD 都能选到）。
   * 注意：这里故意用普通函数而不是 useCallback —— 它依赖 pushLog，
   * 而 pushLog 定义在后面，useCallback 的依赖数组会在渲染期访问未初始化的
   * const（之前踩过 `Cannot access before initialization`）。
   */
  const applySets = async (next) => {
    setSets(next)
    if (!sess?.sid) return
    const on = next.filter((x) => x.on !== false).map((x) => `set:${x.name}`)
    try {
      const r = await api.select(sess.sid, {
        primary,
        components: [...components, ...on],
        sets: next.map((x) => ({
          name: x.name, elements: x.elements || [], shape: x.shape || 'all',
          min_atoms: x.min_atoms || 0, max_atoms: x.max_atoms || 0,
          pure: !!x.pure,
        })),
      })
      setSetsInfo(r?.sets || [])
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
        // 新集合默认参与分析：补进 components（chips 里会显示成选中态）
        const setNames = (r.components || []).map((c) => c.name)
          .filter((n) => n.startsWith('set:'))
        setComponents((prev) => [...prev,
          ...setNames.filter((n) => !prev.includes(n))])
      }
    } catch (e) {
      setError(`集合预览失败: ${e.message}`)
    }
  }

  const pushLog = useCallback((line) => {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    setLog((prev) => [...prev.slice(-300), `[${ts}] ${line}`])
  }, [])

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
      setChart({ name: names[0], index: 0 })
      setTab('charts')
      pushLog(`已恢复上次的会话：${names.length} 项结果（${st.sid}）`)
    } else {
      pushLog(`已恢复上次打开的体系（还没有分析结果）`)
    }
    setOpen(SEC_INFO)

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
      setWhich(a.order || [])
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
        density: { axis: Number(params.axis), nbins: Number(params.nbins) },
        interface: { axis: Number(params.axis), nbins: Math.max(Number(params.nbins), 60) },
        rdf: { rmax: Number(params.rmax), nbins: 120, mode: params.rdf_mode },
        msd: { object: params.msd_object, remove_drift: true },
        contact: { cutoff: Number(params.cutoff), mode: params.contact_mode },
        dihedral: { mode: params.dihedral_mode,
                    gauche_edge: Number(params.gauche_edge) || 120 },
        orientation: { mode: params.orient_mode,
                       stride: Number(params.orient_stride) || 1 },
        // 结构有序度的局部结构分量：留空 → g_ref=None（该分量不计入指数）
        order: { g_ref: String(params.g_ref).trim() === ''
          ? null : Number(params.g_ref),
          // 与「二面角分析」共用同一个 trans/gauche 阈值（后端两个模块都收这个键）
          gauche_edge: Number(params.gauche_edge) || 120 },
        // Rg：「分析对象」二选一（互斥）—— object="molecule" 逐分子算 Rg（默认），
        // object="system" 整组算 Rg。后端只算选中的那种。
        rg: { mass_weighted: params.mass_weighted !== false,
              object: params.rg_object === 'system' ? 'system' : 'molecule' },
        ree: { ends: params.ree_ends || 'bond_graph',
               per_molecule: params.per_molecule !== false,
               // 手工指定两个成键原子作为链端（留空 = None，按 ends 自动判定）
               atom_indices: reeAtoms },
        // 键取向序 BOO / 晶体-非晶识别：邻域半径留空 → null（后端自动推定）
        boo: { cutoff: optNum(params.boo_cutoff), averaged: params.boo_averaged !== false },
        crystal: { cutoff: optNum(params.boo_cutoff),
                   q6_solid: Number(params.q6_solid) || 0.5,
                   min_cluster: Number(params.min_cluster) || 10 },
        // 面内 RDF (2D)：slab 留空 = 全盒；两组分质心距：默认包裹坐标相减(与作者一致)
        rdf2d: { plane_axis: Number(params.axis) ?? 2,
                 rmax: Number(params.rdf2d_rmax) || 70,
                 nbins: Number(params.rdf2d_nbins) || 140,
                 slab: slabRange },
        comdist: { axis: Number(params.axis) ?? 2,
                   pbc: params.comdist_pbc === true },
      },
      selection: {
        primary,
        // 集合参与与否用 `set:<名字>` 表达（后端严格按此过滤）
        components: [...components,
          ...sets.filter((x) => x.on !== false).map((x) => `set:${x.name}`)],
        sets: sets.map((x) => ({
          name: x.name, elements: x.elements || [], shape: x.shape || 'all',
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
    pushLog('已请求取消：当前这项算完后停止，已完成的结果会保留')
    try {
      const p = await api.runCancel(sess.sid)
      setLive((l) => (l ? { ...l, message: p.message || l.message } : l))
    } catch (e) {
      pushLog(`取消失败: ${e.message}`)
    }
  }

  const lastLine = useMemo(() => (log.length ? log[log.length - 1] : ''), [log])

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
      id: SEC_SETUP, title: '3. 分析设置', disabled: !sess,
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
                        <InfoPanel info={sess.info}
                                   onNext={() => setOpen(SEC_SETUP)} />
                      )}
                      {s.id === SEC_SETUP && sess && (
                        <>
                          <SelectionBlock sess={sess} primary={primary}
                                          setPrimary={setPrimary}
                                          components={components}
                                          setComponents={setComponents}
                                          sets={sets} setsInfo={setsInfo}
                                          onSetsChange={applySets} />
                          <FrameBlock frames={frames} setFrames={setFrames} />
                          <FunctionBlock titles={titles} order={order}
                                         which={which} setWhich={setWhich}
                                         groups={groups}
                                         estimates={live?.estimates} />
                          <ParamBlock params={params} setParams={setParams}
                                      which={which}
                                      primaryMolecules={primaryMolecules} />
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
            {tab === 'charts' && (
              <ChartPanel onVisibleChange={setVisibleCurves}
                                 run={run} chart={chart} setChart={setChart}
                          live={live} titles={titles} groups={groups}
                          current={current?.result} panelIndex={current?.panelIndex ?? 0} />
            )}
            {tab === 'data' && <DataPanel run={run} groups={groups} />}
            {/* 统计量：只显示指标 */}
            {tab === 'stats' && <StatsPanel run={run} groups={groups} />}
            {/* 科研 QC：每项分析导出的检查项（PBC/样本/误差/峰显著性…） */}
            {tab === 'qc' && <QcPanel run={run} groups={groups} />}
            {/* 导出：只管导出 */}
            {tab === 'export' && (
              <ExportPanel run={run} sid={sess?.sid} log={pushLog} groups={groups} />
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
