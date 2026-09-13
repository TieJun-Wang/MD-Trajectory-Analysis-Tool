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
    // Rg / R_ee：两个布尔开关默认开（与后端函数签名一致），
    // 关掉「逐分子统计」就只看整组 Rg（一团物质的尺寸），
    // 关掉「质量加权」就用等权（几何）口径。
    mass_weighted: true, per_molecule: true,
    ree_ends: 'bond_graph', ree_atoms: '',
    // BOO / 晶体识别：邻域半径留空 = 自动（按最近邻距离中位数推定）
    boo_cutoff: '', boo_averaged: true, q6_solid: 0.5, min_cluster: 10,
  })
  const [primary, setPrimary] = useState({ mode: 'auto' })
  const [components, setComponents] = useState([])

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
  const [opsCollapsed, setOpsCollapsed] = useState(false)
  const [notesCollapsed, setNotesCollapsed] = useState(false)
  // 当前图表里**仍然显示**的曲线（由图例点选决定）：说明区据此动态过滤
  const [visibleCurves, setVisibleCurves] = useState(null)
  const abortRef = useRef(null)
  const liveRunRef = useRef(null)
  const cancelledRef = useRef(false)

  const pushLog = useCallback((line) => {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    setLog((prev) => [...prev.slice(-300), `[${ts}] ${line}`])
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch((e) => pushLog(`后端连接失败: ${e.message}`))
    api.analyses().then((a) => {
      setTitles(a.titles || {})
      setOrder(a.order || [])
      setWhich(a.order || [])
      setGroups(normalizeGroups(a.groups))
      pushLog(`已连接后端，可用分析 ${a.order?.length || 0} 项`)
    }).catch((e) => pushLog(`读取分析项目录失败: ${e.message}`))
  }, [pushLog])

  const onOpened = (data) => {
    setSess(data)
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
    pushLog(`开始分析 ${which.length} 项（实时刷新）…`)

    // 一点「开始分析」就让位给图表：立刻收起操作台、展开说明栏。
    // 进度改由顶栏的实时指示器显示，所以收起操作台后不会看不到进度。
    focusChart()

    liveRunRef.current = { cancelled: false }

    // R_ee 手工链端：形如 "0, 3034"（逗号/空格分隔）；不是两个非负整数就当留空，
    // 由后端的「链端来源」自动判定（bond_graph = 键连图端基）。
    // 可选数值：留空 → null（后端按默认/自动处理），否则转成数字
    const optNum = (v) => (v === '' || v == null ? null : Number(v))

    const reeAtoms = (() => {
      const raw = String(params.ree_atoms ?? '').trim()
      if (!raw) return null
      const nums = raw.split(/[^0-9]+/).filter((s) => s !== '').map(Number)
      return (nums.length === 2 && nums.every((n) => Number.isInteger(n) && n >= 0))
        ? nums : null
    })()

    const body = {
      which,
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
        // Rg / R_ee：布尔开关直接传（后端 analyze_rg / analyze_end_to_end 接受）
        rg: { mass_weighted: params.mass_weighted !== false,
              per_molecule: params.per_molecule !== false },
        ree: { ends: params.ree_ends || 'bond_graph',
               per_molecule: params.per_molecule !== false,
               // 手工指定两个成键原子作为链端（留空 = None，按 ends 自动判定）
               atom_indices: reeAtoms },
        // 键取向序 BOO / 晶体-非晶识别：邻域半径留空 → null（后端自动推定）
        boo: { cutoff: optNum(params.boo_cutoff), averaged: params.boo_averaged !== false },
        crystal: { cutoff: optNum(params.boo_cutoff),
                   q6_solid: Number(params.q6_solid) || 0.5,
                   min_cluster: Number(params.min_cluster) || 10 },
      },
      selection: { primary, components },
    }

    // 结果按完成顺序累积；后端只回传我们还没有的那几项
    let acc = { results: {}, timings: {}, frames: null, titles: {}, summary_rows: [] }
    let got = 0

    const merge = (p) => {
      const fresh = Object.entries(p.results || {})
      if (fresh.length) {
        for (const [name, res] of fresh) acc.results[name] = res
        got += fresh.length
        // 第一项算完就切到图表页，让用户马上看到东西
        if (got === fresh.length) {
          setChart({ name: fresh[0][0], index: 0 })
          setVisibleCurves(null)
          setTab('charts')
        }
        for (const [name] of fresh) {
          const panels = acc.results[name]?.panels?.length || 1
          pushLog(`  ✓ ${titles[name] || name} 完成（${panels} 张图）`)
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
      merge(await api.runStart(sess.sid, body))

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
                                          setComponents={setComponents} />
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
