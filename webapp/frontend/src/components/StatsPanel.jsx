/**
 * 「统计量」标签页 —— **只负责显示统计量指标**，不做任何导出。
 *
 * 按大纲模块拆成单一模块标签（链构象 / 空间结构 / 取向与结晶 / 动力学与输运），
 * 一次只看一个板块，不必在长页面里翻。
 * 分组与图表导航、数据表导航共用同一份 analysisGroups。
 */
import { useState } from 'react'
import { formatCell } from '../chartOption'
import { groupPresent, normalizeGroups } from '../analysisGroups'

export default function StatsPanel({ run, groups, defaultView }) {
  const results = run?.results || {}
  const order = Object.keys(results)
  const mods = groupPresent(order, normalizeGroups(groups))
  // 默认停在第一个有结果的模块
  const [view, setView] = useState(defaultView || mods[0]?.[0] || '')

  if (!order.length) {
    return <div className="placeholder"><p>还没有结果。</p></div>
  }

  const names = (mods.find(([l]) => l === view) || [, []])[1]

  return (
    <div className="statsview">
      <div className="stattabs">
        {mods.map(([label, ns]) => (
          <button key={label}
                  className={`tab ${view === label ? 'active' : ''}`}
                  onClick={() => setView(label)}>
            {label}
            <span className="dim">（{ns.length}）</span>
          </button>
        ))}
      </div>

      <div className="stattabbody">
        {!names.length ? (
          <div className="tabhint">这个模块下没有结果（可能没勾选对应分析项）。</div>
        ) : names.map((name) => {
          const res = results[name]
          const entries = Object.entries(res.summary || {})
          return (
            <details key={name} open className="statblock">
              <summary>
                {res.title}
                <span className="dim">（{entries.length} 项）</span>
              </summary>
              <table className="stats">
                <thead>
                  <tr><th>统计量</th><th>数值</th></tr>
                </thead>
                <tbody>
                  {entries.map(([k, v]) => (
                    <tr key={k}>
                      <th>{k}</th>
                      <td>{formatCell(v, 6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {res.notes?.length > 0 && (
                <div className="statnotes">
                  {res.notes.map((n, i) => <div key={i}>· {n}</div>)}
                </div>
              )}
            </details>
          )
        })}
      </div>
    </div>
  )
}
