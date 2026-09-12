/**
 * 数据表页：左侧「模块 → 分析 → 数据表」导航，右侧把曲线的 x/y 以表格列出，
 * 可复制 / 下载。
 *
 * 导航分层和图表页保持一致（模块 → 分析项 → 各张表），用的是同一份
 * `analysisGroups` 定义，所以两边的分法不会不一致。
 */
import { useMemo, useState } from 'react'
import { formatCell } from '../chartOption'
import { groupPresent, normalizeGroups } from '../analysisGroups'

function buildTable(result, panelIndex) {
  const curves = result.curves.filter((c) => c.panel === panelIndex)
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

export default function DataPanel({ run, groups }) {
  const results = run?.results || {}
  const order = useMemo(() => Object.keys(results), [results])
  const [sel, setSel] = useState(null)
  // 折叠起来的模块（默认全展开；记"关掉的"，新模块默认展开）
  const [closedModules, setClosedModules] = useState(() => new Set())

  const active = sel && results[sel.name] ? sel : (order.length ? { name: order[0], index: 0 } : null)
  const result = active ? results[active.name] : null
  const { columns, rows } = useMemo(
    () => (result ? buildTable(result, active.index) : { columns: [], rows: [] }),
    [result, active],
  )

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
            <button className="mini" onClick={() => {
              const text = [columns.join('\t'), ...rows.map((r) => r.join('\t'))].join('\n')
              navigator.clipboard?.writeText(text)
            }}>复制</button>
            <button className="mini" onClick={downloadCsv}>下载 CSV</button>
          </div>
        </div>

        <div className="tablewrap">
          <table className="datatable">
            <thead>
              <tr>{columns.map((c, i) => <th key={i}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.slice(0, 500).map((r, i) => (
                <tr key={i}>
                  {r.map((v, j) => (
                    <td key={j}>{formatCell(v, 6)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length > 500 && (
          <div className="dim small">共 {rows.length} 行，表格只显示前 500 行；需要完整数据请「下载 CSV」。</div>
        )}
      </section>
    </div>
  )
}
