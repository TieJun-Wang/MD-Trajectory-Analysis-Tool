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
                <div className="charttitle">{current.title}</div>
                <div className="chartsub">
                  第 {panelIndex + 1} / {current.panels.length} 张
                  {/* 坐标轴尺度：原来只写「对数轴」。现在只在单条曲线被单独选中时
                      才由 chartOption 自适应（见下），所以这里把两种情形都写清楚。 */}
                  {(() => {
                    const pn = current.panels[panelIndex] || {}
                    const lx = pn.xscale === 'log'
                    const ly = pn.yscale === 'log'
                    if (lx && ly) return ' · 对数轴'
                    if (lx) return ' · 对数轴（仅横轴）'
                    if (ly) return ' · 对数轴（仅纵轴）'
                    return ''
                  })()}
                  {solo
                    ? ` · 单曲线自适应（横轴${solo.xscale === 'log' ? '对数' : '线性'}、`
                      + `纵轴${solo.yscale === 'log' ? '对数' : '线性'}）`
                    : ''}

                  {/* 点数与抽稀状态：让"图上有多少点"一眼可见（后端默认不抽稀） */}
                  {(() => {
                    const cs = current.curves.filter((c) => c.panel === panelIndex)
                    if (!cs.length) return null
                    const nmax = Math.max(...cs.map((c) => (c.x ? c.x.length : 0)))
                    const ds = cs.some((c) => c.downsampled)
                    return ` · ${nmax.toLocaleString()} 点${ds ? '（已抽稀）' : '（未抽稀）'}`
                  })()}
                  {/* 一张图里有多个对象（多个组分 / 集合）时，图表默认只显示
                      第一个，其余在图例里点开 —— 这里把这件事说出来，
                      否则用户会以为"第二个组分没算"。 */}
                  {(() => {
                    const cs = current.curves.filter((c) => c.panel === panelIndex)
                    const gs = [...new Set(cs.map((c) => c.group).filter(Boolean))]
                    return gs.length > 1
                      ? ` · ${gs.length} 个对象（默认只画「${gs[0]}」，其余点图例打开）`
                      : ''
                  })()}
                  {isFs ? ' · 全屏中（Esc 退出）' : ''}
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

            <ChartView result={current} panelIndex={panelIndex}
                       visible={visible}
                       onVisibleChange={handleVisible} />
          </>
        )}
      </section>
    </div>
  )
}
