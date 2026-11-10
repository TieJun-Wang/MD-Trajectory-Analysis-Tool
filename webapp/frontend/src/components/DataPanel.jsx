/**
 * 数据表页：左侧「模块 → 分析 → 数据表」导航，右侧把曲线的 x/y 以表格列出，
 * 可复制 / 下载。
 *
 * 导航分层和图表页保持一致（模块 → 分析项 → 各张表），用的是同一份
 * `analysisGroups` 定义，所以两边的分法不会不一致。
 */
import { useMemo, useState } from 'react'
import { useEffect } from 'react'
import { api } from '../api'
import { buildOption, formatCell } from '../chartOption'
import { groupPresent, normalizeGroups } from '../analysisGroups'

/**
 * 这张图**当前应该显示**的曲线：
 * 1) 用户在图例里点过（visible 非空）→ 用那批；
 * 2) 没点过 → 用图表的默认图例（`legend.selected`，例如 MSD 各轴分解只勾第一条）；
 * 3) 没有图例（单曲线）→ 全显示。
 * 数据表与图表必须同口径，否则会看到"图上没有、表里还有"的曲线。
 */
function shownCurves(result, panelIndex, visible) {
  const curves = result.curves.filter((c) => c.panel === panelIndex)
  if (Array.isArray(visible) && visible.length) {
    const set = new Set(visible)
    const hit = curves.filter((c) => set.has(c.label))
    return hit.length ? hit : curves
  }
  const opt = buildOption(result, panelIndex)
  const sel = opt && opt.legend && opt.legend.selected
  if (sel) {
    const hit = curves.filter((c) => sel[c.label] !== false)
    if (hit.length) return hit
  }
  return curves
}

function buildTable(result, panelIndex, onlyShown, visible) {
  // 默认把该面板**全部图例**的数据都列出来（用户要求）；
  // 勾上「只看当前图例」才按图上实际显示的曲线过滤。
  const curves = onlyShown
    ? shownCurves(result, panelIndex, visible)
    : result.curves.filter((c) => c.panel === panelIndex)
  if (!curves.length) return { columns: [], rows: [] }
  const panel = result.panels[panelIndex] || {}
  const xlabel = panel.xlabel || 'x'
  const xs = curves[0].x
  const columns = [xlabel, ...curves.map((c) => c.label)]
  const rows = []
  for (let i = 0; i < xs.length; i += 1) {
    rows.push([xs[i], ...curves.map((c) => c.y[i])])
  }
  return { columns, rows }
}

export default function DataPanel({ run, groups, chart = null, setChart = null,
                                   visible = null, sid = '' }) {
  const results = run?.results || {}
  const order = useMemo(() => Object.keys(results), [results])
  // 选中的「分析项 + 第几张图」：与图表页**共用同一个状态**（切页不丢），
  // 没有外部状态时退回内部 state（单独渲染这个组件时也能用）。
  const [selLocal, setSelLocal] = useState(null)
  const sel = setChart ? chart : selLocal
  const setSel = (v) => { if (setChart) setChart(v); else setSelLocal(v) }
  // 折叠起来的模块（默认全展开；记"关掉的"，新模块默认展开）
  const [closedModules, setClosedModules] = useState(() => new Set())

  // 「只看当前图例」：默认关（数据表列出所有曲线的数据，用户要求）
  const [onlyShown, setOnlyShown] = useState(false)

  const active = sel && results[sel.name] ? sel : (order.length ? { name: order[0], index: 0 } : null)
  const result = active ? results[active.name] : null

  // ---- 分页：默认从后端只取一页（PB 级曲线全渲染成 <td> 会卡死浏览器）
  const PAGE_DEFAULT = 200
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(PAGE_DEFAULT)
  const [slice, setSlice] = useState(null)       // {columns, rows, total, offset}
  const [sliceErr, setSliceErr] = useState('')
  useEffect(() => { setPage(0) }, [active?.name, active?.index, onlyShown])

  const onlyLabels = useMemo(() => {
    if (!onlyShown || !result) return null
    if (Array.isArray(visible)) return visible
    try {
      const opt = buildOption(result, active.index)
      const sel = opt && opt.legend && opt.legend.selected
      if (sel) return Object.keys(sel).filter((k) => sel[k])
    } catch { /* 忽略 */ }
    return null
  }, [onlyShown, result, active, visible])

  // 没有会话（自检 / 单独渲染）：同步用本地数据建表，首帧就有表头与行
  const local = useMemo(
    () => (result ? buildTable(result, active.index, onlyShown, visible)
      : { columns: [], rows: [] }),
    [result, active, onlyShown, visible],
  )

  useEffect(() => {
    if (!result || !sid) return undefined
    let alive = true
    setSliceErr('')
    api.table(sid, { name: active.name, panel: active.index,
      offset: page * pageSize, limit: pageSize, only: onlyLabels })
      .then((r) => { if (alive) setSlice(r) })
      .catch((e) => { if (alive) setSliceErr(e.message) })
    return () => { alive = false }
  }, [sid, result, active, page, pageSize, onlyShown, visible, onlyLabels])

  const { columns, rows } = sid
    ? (slice || { columns: local.columns, rows: [] })
    : local
  const totalRows = sid ? (slice?.total ?? rows.length) : local.rows.length
  const nPage = Math.max(1, Math.ceil(totalRows / pageSize))

  if (!order.length) {
    return <div className="placeholder"><p>还没有结果。</p></div>
  }

  const toggleModule = (label) => {
    const next = new Set(closedModules)
    if (next.has(label)) next.delete(label); else next.add(label)
    setClosedModules(next)
  }

  const downloadCsv = () => {
    const esc = (v) => {
      const s = v === null || v === undefined ? '' : String(v)
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
    }
    const csv = [columns.map(esc).join(','), ...rows.map((r) => r.map(esc).join(','))]
      .join('\r\n')
    // 加 BOM，Excel 打开不乱码
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${active.name}__p${active.index + 1}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="datasplit">
      <aside className="datanav">
        <div className="chartnav-head">
          数据表
          <span className="dim">{order.length} 项 / {order.reduce(
            (a, n) => a + (results[n].panels?.length || 1), 0)} 张</span>
        </div>
        <div className="chartnav-list">
          {/* 与图表导航同样的两层：大纲模块 → 分析项 → 各张表 */}
          {groupPresent(order, normalizeGroups(groups)).map(([modLabel, names]) => {
            const modOpen = !closedModules.has(modLabel)
            const nTables = names.reduce(
              (a, n) => a + (results[n].panels?.length || 1), 0)
            return (
              <div key={modLabel} className="navmodule">
                <button className="navmodule-head"
                        onClick={() => toggleModule(modLabel)}>
                  <span className="caret">{modOpen ? '▾' : '▸'}</span>
                  <span className="mtitle">{modLabel}</span>
                  <span className="count">{names.length} 项 / {nTables} 张</span>
                </button>

                {modOpen && names.map((name) => {
                  const res = results[name]
                  return (
                    <div key={name} className="navgroup">
                      <div className="navgroup-head static">{res.title}</div>
                      {(res.panels || []).map((p, i) => (
                        <button key={i}
                                className={`navitem ${active?.name === name && active.index === i ? 'active' : ''}`}
                                title={`${modLabel} ▸ ${res.title} ▸ ${p.title}`}
                                onClick={() => setSel({ name, index: i })}>
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

      <section className="dataarea">
        <div className="charthead">
          <div>
            <div className="charttitle">{result.title}</div>
            <div className="chartsub">
              {result.panels[active.index]?.title} · {rows.length} 行 × {columns.length} 列
            </div>
          </div>
          <div className="charttools">
            <label className="mini"
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                   title={'勾上后只列「图表页图例里当前显示」的曲线；'
                     + '不勾则列出这个面板全部曲线的数据'}>
              <input type="checkbox" checked={onlyShown}
                     onChange={(e) => setOnlyShown(e.target.checked)} />
              只看当前图例
            </label>
            <button className="mini" onClick={() => {
              const text = [columns.join('\t'), ...rows.map((r) => r.join('\t'))].join('\n')
              navigator.clipboard?.writeText(text)
            }}>复制</button>
            <button className="mini" onClick={downloadCsv}>下载 CSV</button>
          </div>
        </div>

        <div className="tablefoot">
          <span className="dim">
            共 {totalRows.toLocaleString()} 行
            {totalRows > pageSize
              ? ` · 第 ${Math.min(page + 1, nPage)} / ${nPage} 页（每页 ${pageSize} 行）` : ''}
            {sliceErr ? ` · ${sliceErr}` : ''}
          </span>
          <span className="charttools">
            <button className="mini" disabled={page <= 0}
                    onClick={() => setPage((p) => Math.max(0, p - 1))}>上一页</button>
            <button className="mini" disabled={page + 1 >= nPage}
                    onClick={() => setPage((p) => p + 1)}>下一页</button>
            <select value={pageSize} onChange={(e) => {
              setPageSize(Number(e.target.value)); setPage(0)
            }}>
              {[200, 500, 1000, 2000].map((n) => (
                <option key={n} value={n}>每页 {n} 行</option>
              ))}
            </select>
          </span>
        </div>

        <div className="tablewrap">
          <table className="datatable">
            <thead>
              <tr>{columns.map((c, i) => <th key={i}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {/* 用户要求数据表**不抽稀**：x/y 与导出的 CSV 一致，全部行都渲染出来。
                  行数很大时（上万行）浏览器会慢一些，但数值一个不少；想快就少选帧。 */}
              {rows.map((r, i) => (
                <tr key={i}>
                  {r.map((v, j) => (
                    <td key={j}>{formatCell(v, 6)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length > 20000 && (
          <div className="dim small">共 {rows.length.toLocaleString()} 行，已全部渲染（不抽稀）；滚动或「下载 CSV」查看。</div>
        )}
      </section>
    </div>
  )
}
