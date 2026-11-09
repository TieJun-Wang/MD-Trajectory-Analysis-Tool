/**
 * 图表页：左侧是「分析 → 图标题」导航，右侧一次只显示一张图。
 * 点击标题即跳转到对应的图。
 *
 * 图表的文字说明与统计量已移到整页右侧的 NotesPanel，
 * 这里只负责导航 + 画布 + 全屏，图表因此能占满整个高度。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import ChartView from './ChartView'
import { groupPresent, normalizeGroups } from '../analysisGroups'
import { soloAxisInfo } from '../chartOption'

export default function ChartPanel({ run, chart, setChart, live, titles = {},
                                    current = null, panelIndex = 0, groups,
                                    onVisibleChange }) {
  const results = run?.results || {}
  const order = useMemo(() => Object.keys(results), [results])
  const pending = live?.pending || []
  const running = live?.status === 'running'

  const [openSet, setOpenSet] = useState(() => new Set(order))
  // 图例里当前可见的曲线：只留 1 条时按它自适应坐标轴（见 chartOption）
  const [visible, setVisible] = useState(null)
  const solo = current ? soloAxisInfo(current, panelIndex, visible) : null
  const handleVisible = (vis) => {
    setVisible(vis)
    onVisibleChange?.(vis)
  }
  // 折叠起来的**模块**（默认全部展开）；记"关掉的"这样新模块默认是展开的
  const [closedModules, setClosedModules] = useState(() => new Set())
  // 全屏：优先用浏览器 Fullscreen API；被拒绝/不支持时退化成整页覆盖
  const [cssFs, setCssFs] = useState(false)
  const [nativeFs, setNativeFs] = useState(false)
  const areaRef = useRef(null)

  useEffect(() => {
    setOpenSet(new Set(order))
  }, [order])

  useEffect(() => {
    if (typeof document === 'undefined') return undefined
    const on = () => setNativeFs(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', on)
    return () => document.removeEventListener('fullscreenchange', on)
  }, [])

  // 退化成 CSS 覆盖时，Esc 退出
  useEffect(() => {
    if (!cssFs) return undefined
    const onKey = (e) => { if (e.key === 'Escape') setCssFs(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cssFs])

  const isFs = nativeFs || cssFs
  const toggleFullscreen = async () => {
    const el = areaRef.current
    if (!el) return
    if (typeof document === 'undefined') return
    if (document.fullscreenElement) {
      try { await document.exitFullscreen() } catch { /* 忽略 */ }
      return
    }
    if (cssFs) { setCssFs(false); return }
    try {
      if (typeof el.requestFullscreen !== 'function') throw new Error('不支持')
      await el.requestFullscreen()
    } catch {
      // 有些浏览器/嵌入环境会拒绝（需要用户手势、iframe 权限等），退回覆盖模式
      setCssFs(true)
    }
  }

  // ---------------------------------------------------------------- MSD 拟合覆盖层
  // 用户要求：MSD「线性区拟合」那张图上，左上角写拟合区间（可自定义），
  // 右上角一行一个数据给斜率 / R² / D；改区间后**立刻重算**（纯前端最小二乘）。
  // 默认区间来自后端 meta.msd_fit（= fit_fraction × τ_max，与摘要里的 D 同窗口）。
  const isMsdFit = current?.name === 'msd' && panelIndex === 1
  // 拟合窗口/统计量来自结果 summary（meta 里的 dict 后端会过滤掉）
  const fitSum = (current?.summary) || {}
  const fitOf = (g) => ({
    'τ 起 (ps)': Number(fitSum[`${g} 拟合起点 (ps)`]),
    'τ 止 (ps)': Number(fitSum[`${g} 拟合终点 (ps)`]),
    'R²': Number(fitSum[`${g} R²`]),
    'D (m²/s)': Number(fitSum[`${g} D (m²/s)`]),
  })
  const dimOf = Number(fitSum['dim']) || 3
  const [fitWin, setFitWin] = useState({ a: '', b: '' })
  useEffect(() => { setFitWin({ a: '', b: '' }) }, [chart.name, panelIndex])

  const fitGroups = useMemo(() => {
    if (!isMsdFit) return []
    const gs = []
    for (const c of current.curves || []) {
      if (c.panel !== 1) continue
      if (/拟合/.test(String(c.label))) continue
      const g = String(c.group || c.label)
      if (!gs.includes(g)) gs.push(g)
    }
    return gs
  }, [isMsdFit, current])

  const fitInfo = useMemo(() => {
    if (!isMsdFit || !fitGroups.length) return null
    const dim = dimOf
    const rows = []
    const extra = []
    for (const g of fitGroups) {
      const base = (current.curves || []).find(
        (c) => c.panel === 1 && !/拟合/.test(String(c.label))
          && String(c.group || c.label) === g)
      if (!base) continue
      const meta = fitOf(g)
      let t1 = Number(fitWin.a)
      let t2 = Number(fitWin.b)
      const custom = Number.isFinite(t1) && Number.isFinite(t2) && t2 > t1
      if (!custom) {
        t1 = Number(meta['τ 起 (ps)'])
        t2 = Number(meta['τ 止 (ps)'])
      }
      const x = base.x || []
      const y = base.y || []
      const px = []
      const py = []
      for (let i = 0; i < x.length; i += 1) {
        const xi = Number(x[i])
        const yi = Number(y[i])
        if (!Number.isFinite(xi) || !Number.isFinite(yi)) continue
        if (xi < t1 || xi > t2) continue
        px.push(xi)
        py.push(yi)
      }
      let slope = NaN
      let inter = NaN
      let r2 = NaN
      if (px.length >= 3) {
        const n = px.length
        let mx = 0
        let my = 0
        for (let i = 0; i < n; i += 1) { mx += px[i]; my += py[i] }
        mx /= n
        my /= n
        let sxy = 0
        let sxx = 0
        let syy = 0
        for (let i = 0; i < n; i += 1) {
          sxy += (px[i] - mx) * (py[i] - my)
          sxx += (px[i] - mx) ** 2
          syy += (py[i] - my) ** 2
        }
        slope = sxx > 0 ? sxy / sxx : NaN
        inter = my - slope * mx
        r2 = (sxx > 0 && syy > 0) ? (sxy * sxy) / (sxx * syy) : NaN
        if (custom && Number.isFinite(slope)) {
          extra.push({
            label: `${g} 拟合（自定义）`, kind: 'line', panel: 1, group: g,
            x: [t1, t2], y: [slope * t1 + inter, slope * t2 + inter],
          })
        }
      }
      rows.push({
        g, t1, t2, slope, r2, n: px.length, custom,
        d: Number.isFinite(slope) ? (slope / (2 * dim)) * 1e-8 : NaN,
      })
    }
    if (!rows.length) return null
    return { rows, extra, custom: rows.some((r) => r.custom) }
  }, [isMsdFit, current, fitGroups, fitSum, fitWin])

  // 自定义区间时：用前端算的拟合线替掉后端那条（默认区间的线）
  const viewResult = useMemo(() => {
    if (!current || !fitInfo || !fitInfo.custom) return current
    const curves = (current.curves || [])
      .filter((c) => !(c.panel === 1 && /拟合/.test(String(c.label))))
      .concat(fitInfo.extra || [])
    return { ...current, curves }
  }, [current, fitInfo])

  const fmtNum = (v, digits = 4) => (
    Number.isFinite(v) ? Number(v.toPrecision(digits)).toString() : '—')
  const fmtD = (v) => (Number.isFinite(v) ? v.toExponential(3) : '—')

  if (!order.length) {
    // 实时运行时，第一项还没算完就到这里——不能说"还没有结果"
    if (running) {
      const firstName = pending[0]
      return (
        <div className="placeholder">
          <p className="chart-wait">
            <span className="badge-live">● 实时更新中</span>
            正在计算第一项{pending.length ? `：${titles[firstName] || firstName}` : ''}…
          </p>
          <p className="dim">
            算完一项就会立刻出现在这里，不用等全部跑完（
            {live?.nDone ?? 0}/{live?.nTotal ?? 0}）。
          </p>
        </div>
      )
    }
    return (
      <div className="placeholder">
        <p>还没有结果。</p>
        <p className="dim">在左侧读取体系、勾选分析功能，然后点「开始分析」。</p>
      </div>
    )
  }

  const toggleOpen = (name) => {
    const next = new Set(openSet)
    if (next.has(name)) next.delete(name); else next.add(name)
    setOpenSet(next)
  }

  const toggleModule = (label) => {
    const next = new Set(closedModules)
    if (next.has(label)) next.delete(label); else next.add(label)
    setClosedModules(next)
  }

  return (
    <div className="chartsplit">
      <aside className="chartnav">
        <div className="chartnav-head">
          图表导航
          <span className="dim">{order.length} 项 / {order.reduce(
            (a, n) => a + (results[n].panels?.length || 1), 0)} 张</span>
          {running && pending.length > 0 && (
            <span className="badge-live">还有 {pending.length} 项在算</span>
          )}
        </div>
        <div className="chartnav-list">
          {/* 两层：大纲模块（链构象 / 空间结构 / 取向与结晶 / 动力学与输运）→ 分析项 → 各张图。
              模块只显示"已经有结果"的项，所以实时运行时模块会一项项长出来。 */}
          {groupPresent(order, normalizeGroups(groups)).map(([modLabel, names]) => {
            const modOpen = !closedModules.has(modLabel)
            const nPanels = names.reduce(
              (a, n) => a + (results[n].panels?.length || 1), 0)
            return (
              <div key={modLabel} className="navmodule">
                <button className="navmodule-head"
                        onClick={() => toggleModule(modLabel)}>
                  <span className="caret">{modOpen ? '▾' : '▸'}</span>
                  <span className="mtitle">{modLabel}</span>
                  <span className="count">{names.length} 项 / {nPanels} 张</span>
                </button>

                {modOpen && names.map((name) => {
                  const res = results[name]
                  const n = res.panels?.length || 1
                  const open = openSet.has(name)
                  return (
                    <div key={name} className="navgroup">
                      <button className="navgroup-head" onClick={() => toggleOpen(name)}>
                        <span className="caret">{open ? '▾' : '▸'}</span>
                        <span className="gtitle">{res.title}</span>
                        <span className="count">{n}</span>
                      </button>
                      {open && (res.panels || []).map((p, i) => (
                        <button key={i}
                                className={`navitem ${chart.name === name && panelIndex === i ? 'active' : ''}`}
                                onClick={() => setChart({ name, index: i })}
                                title={`${modLabel} ▸ ${res.title} ▸ ${p.title}`}>
                          <span className="idx">{i + 1}</span>
                          <span className="ptitle">{p.title}</span>
                        </button>
                      ))}
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>
      </aside>

      <section ref={areaRef} className={`chartarea ${cssFs ? 'fs' : ''}`}>
        {current && (
          <>
            <div className="charthead">
              <div>
                {/* 用户要求：标题下方不再有小字。原来那串「第 k/n 张 · 对数轴 ·
                    N 点 · 多个对象 · 单帧…」全部挪进标题的悬浮提示，需要时悬停即见。 */}
                <div className="charttitle"
                     title={(() => {
                       const parts = [`第 ${panelIndex + 1} / ${current.panels.length} 张`]
                       const pn = current.panels[panelIndex] || {}
                       const lx = pn.xscale === 'log'
                       const ly = pn.yscale === 'log'
                       if (lx && ly) parts.push('对数轴')
                       else if (lx) parts.push('对数轴（仅横轴）')
                       else if (ly) parts.push('对数轴（仅纵轴）')
                       if (solo) {
                         parts.push(`单曲线自适应（横轴${solo.xscale === 'log'
                           ? '对数' : '线性'}、纵轴${solo.yscale === 'log'
                           ? '对数' : '线性'}）`)
                       }
                       const cs = current.curves.filter((c) => c.panel === panelIndex)
                       if (cs.length) {
                         const nmax = Math.max(...cs.map((c) => (c.x ? c.x.length : 0)))
                         parts.push(`${nmax.toLocaleString()} 点`
                           + (cs.some((c) => c.downsampled) ? '（已抽稀）' : '（未抽稀）'))
                         const gs = [...new Set(cs.map((c) => c.group).filter(Boolean))]
                         if (gs.length > 1) {
                           parts.push(`${gs.length} 个对象（默认只画「${gs[0]}」，`
                             + '其余点图例打开）')
                         }
                       }
                       if (current.meta && current.meta.single_frame) {
                         parts.push(`单帧（${current.meta.n_frames ?? 1} 帧）：`
                           + '只出静态分布图，随时间变化的面板已省略')
                       }
                       if (isFs) parts.push('全屏中（Esc 退出）')
                       return parts.join(' · ')
                     })()}>
                  {current.title}
                </div>
              </div>
              <div className="charttools">
                <button className="mini" disabled={panelIndex <= 0}
                        onClick={() => setChart({ name: chart.name, index: panelIndex - 1 })}>
                  ← 上一张
                </button>
                <button className="mini"
                        disabled={panelIndex >= current.panels.length - 1}
                        onClick={() => setChart({ name: chart.name, index: panelIndex + 1 })}>
                  下一张 →
                </button>
                <button className="mini fsbtn" onClick={toggleFullscreen}
                        title={isFs ? '退出全屏（Esc）' : '全屏显示这张图，便于仔细看数据'}>
                  {isFs ? '⤡ 退出全屏' : '⤢ 全屏'}
                </button>
              </div>
            </div>

            <div className="chartwrap">
              <ChartView result={viewResult} panelIndex={panelIndex}
                         visible={visible}
                         onVisibleChange={handleVisible} />
              {isMsdFit && fitInfo && (
                <>
                  <div className="fitov l">
                    <span>拟合区间 </span>
                    <b>{fmtNum(fitInfo.rows[0].t1)} – {fmtNum(fitInfo.rows[0].t2)} ps</b>
                    <span className="dim">{fitInfo.custom ? '（自定义）' : '（默认）'}</span>
                    <span className="fitov-in">
                      <input value={fitWin.a} placeholder="起 ps" size={6}
                             onChange={(e) => setFitWin({ ...fitWin, a: e.target.value })} />
                      <input value={fitWin.b} placeholder="止 ps" size={6}
                             onChange={(e) => setFitWin({ ...fitWin, b: e.target.value })} />
                      {(fitWin.a || fitWin.b) && (
                        <button className="mini" onClick={() => setFitWin({ a: '', b: '' })}>
                          恢复默认
                        </button>
                      )}
                    </span>
                  </div>
                  <div className="fitov r">
                    {fitInfo.rows.map((r) => (
                      <div key={r.g} className="fitov-row">
                        {fitInfo.rows.length > 1 && <div className="fitov-name">{r.g}</div>}
                        {r.n < 3 ? (
                          <div className="dim">区间内只有 {r.n} 个点，无法拟合</div>
                        ) : (
                          <>
                            <div>斜率 {fmtNum(r.slope)} Å²/ps</div>
                            <div>R² {fmtNum(r.r2)}</div>
                            <div>D {fmtD(r.d)} m²/s</div>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  )
}
