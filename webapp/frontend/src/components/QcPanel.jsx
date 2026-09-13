/**
 * 「科研 QC」标签页 —— 排版与「统计量」一致：按大纲模块分标签，一次看一个板块，
 * 每个分析项一张表。
 *
 * 表头固定三列（与统计量的「统计量 / 数值」同一套表格样式）：
 *
 * - **QC判断**：级别徽标（正常 / 注意 / 不可引用）+ 检查项名称；
 * - **数据参考**：该结论依据的实测数据（判据、阈值、实测值都在这里）；
 * - **解释**：为什么这么判（口径理由、物理含义、该怎么处理）。
 *
 * 级别语义：``ok`` 可引用 / ``warn`` 数可用但需保留意见 / ``bad`` **不可引用**
 * （该量已被拒绝，或口径造成假象）。
 */
import { useState } from 'react'
import { groupPresent, normalizeGroups } from '../analysisGroups'

const LEVEL_TXT = { ok: '正常', warn: '注意', bad: '不可引用' }
const LEVEL_CLS = { ok: 'qc-ok', warn: 'qc-warn', bad: 'qc-bad' }
//: 排序：不可引用 → 注意 → 正常（问题优先暴露）
const RANK = { bad: 0, warn: 1, ok: 2 }
const rankOf = (c) => RANK[c.级别] ?? 2
/** 一项分析里最严重的级别（没有检查项时按"正常"处理） */
const worst = (checks) => (checks.length ? Math.min(...checks.map(rankOf)) : 2)

export default function QcPanel({ run, groups, defaultView }) {
  const results = run?.results || {}
  const order = Object.keys(results)
  const mods = groupPresent(order, normalizeGroups(groups))
  const [view, setView] = useState(defaultView || mods[0]?.[0] || '')

  if (!order.length) {
    return (
      <div className="placeholder">
        <p>还没有结果。</p>
        <p className="dim">
          跑完分析后，这里会按模块列出每一项的科研 QC 判断：
          PBC 追踪、样本数、误差棒可信度、结构峰显著性、指标区分度、指标性质…
        </p>
      </div>
    )
  }

  // 总览：所有已算出结果的检查项计数
  const sum = { ok: 0, warn: 0, bad: 0 }
  for (const n of order) {
    const d = results[n].checks_digest || {}
    sum.ok += d.ok || 0
    sum.warn += d.warn || 0
    sum.bad += d.bad || 0
  }

  const names = (mods.find(([l]) => l === view) || [, []])[1]
  // 模块内按严重程度排序（不可引用的排最前），同级按标题
  const ranked = [...names].sort((a, b) => {
    const ra = worst(results[a]?.checks || [])
    const rb = worst(results[b]?.checks || [])
    return ra - rb || String(results[a]?.title || a).localeCompare(
      String(results[b]?.title || b))
  })

  return (
    <div className="statsview">
      <div className="stattabs">
        {mods.map(([label, ns]) => {
          const bad = ns.filter((n) => worst(results[n]?.checks || []) === 0).length
          const warn = ns.filter((n) => worst(results[n]?.checks || []) === 1).length
          return (
            <button key={label}
                    className={`tab ${view === label ? 'active' : ''}`}
                    onClick={() => setView(label)}>
              {label}
              <span className="dim">（{ns.length}）</span>
              {bad > 0 && <span className="qc-pill qc-bad">{bad}</span>}
              {bad === 0 && warn > 0 && <span className="qc-pill qc-warn">{warn}</span>}
            </button>
          )
        })}
      </div>

      <div className="stattabbody">
        <div className="qc-sum">
          <span className="dim">全部结果：</span>
          <span className="qc-pill qc-ok">正常 {sum.ok}</span>
          <span className="qc-pill qc-warn">注意 {sum.warn}</span>
          <span className="qc-pill qc-bad">不可引用 {sum.bad}</span>
        </div>

        {!names.length ? (
          <div className="tabhint">这个模块下没有结果（可能没勾选对应分析项）。</div>
        ) : ranked.map((name) => {
          const res = results[name]
          const checks = [...(res.checks || [])].sort((a, b) => rankOf(a) - rankOf(b))
          const d = res.checks_digest || {}
          return (
            <details key={name} open className="statblock">
              <summary>
                {res.title}
                <span className="dim">
                  {`（正常 ${d.ok || 0} / 注意 ${d.warn || 0} / 不可引用 ${d.bad || 0}）`}
                </span>
              </summary>
              {checks.length ? (
                <table className="stats qc">
                  {/* 四列等宽由 colgroup 定死（配 table-layout:fixed），
                      不靠内容长度决定列宽 */}
                  <colgroup>
                    <col />
                    <col />
                    <col />
                    <col />
                  </colgroup>
                  <thead>
                    <tr>
                      <th className="qc-c1">QC判断类型</th>
                      <th className="qc-c2">QC判断结果</th>
                      <th className="qc-c3">QC数据参考</th>
                      <th className="qc-c4">解释</th>
                    </tr>
                  </thead>
                  <tbody>
                    {checks.map((c, i) => (
                      <tr key={i}>
                        <td className="qc-c1">{c.名称}</td>
                        <td className="qc-c2">
                          <span className={`qc-pill ${LEVEL_CLS[c.级别] || 'qc-ok'}`}>
                            {LEVEL_TXT[c.级别] || c.级别}
                          </span>
                        </td>
                        <td className="qc-c3">{c.结论}</td>
                        <td className="qc-c4">{c.依据 || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="statnotes"><div>· 该项没有导出检查项</div></div>
              )}
            </details>
          )
        })}
      </div>
    </div>
  )
}
