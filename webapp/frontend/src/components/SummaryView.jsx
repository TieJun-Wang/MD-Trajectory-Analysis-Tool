/**
 * 统计量展示：compact 模式显示成卡片（图表页用），full 模式显示成完整表格。
 *
 * ``tableClass`` 用于给表格追加样式类：右侧「图表说明」栏很窄，
 * 需要 ``notes-table`` 那套「固定列宽 + 自动换行 + 无横向滚动」的约束，
 * 否则长键名会把单元格挤得参差不齐（看起来就是"文字错位"）。
 */
import { formatCell } from '../chartOption'

export default function SummaryView({ result, compact = false, tableClass = '' }) {
  if (!result) return null
  const entries = Object.entries(result.summary || {})
  if (!entries.length) return null

  if (compact) {
    return (
      <div className="statcards">
        {entries.slice(0, 12).map(([k, v]) => (
          <div key={k} className="statcard">
            <div className="skey">{k}</div>
            <div className="sval">{formatCell(v)}</div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <table className={`stats ${tableClass}`.trim()}>
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{formatCell(v, 6)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
