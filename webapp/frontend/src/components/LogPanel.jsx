/**
 * 运行日志标签页（完整日志放在右侧，避免左操作台被撑长）。
 */
export default function LogPanel({ log, onClear }) {
  return (
    <div className="logview">
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8,
        position: 'sticky', top: 0,
      }}>
        <strong style={{ color: '#cbd5e1', fontSize: 12.5 }}>
          运行日志（{log?.length || 0} 行）
        </strong>
        <button className="mini" onClick={onClear}>清空</button>
      </div>
      {(log || []).length === 0
        ? <div className="logline">（暂无日志）</div>
        : (log || []).map((line, i) => (
          <div key={i} className="logline">{line}</div>
        ))}
    </div>
  )
}
