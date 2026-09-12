/**
 * 右侧「图表说明」面板。
 *
 * 原来这些文字（算法说明 + 关键统计量）压在图表下方，把图挤得很矮；
 * 现在移到整页右侧独立成一栏，并且和左侧操作台一样可以整体折叠。
 * 内容跟随「当前正在看的那张图」——由 App 统一算好 current 传进来。
 *
 * 排版约束（窄栏里最要紧的一条）：
 * 长键名 / 长说明必须**在表格里换行**，不能把栏撑宽去做左右滑动。
 * 说明用 flex 悬挂缩进（换行后仍与首行左缘对齐），统计量用固定列宽的表格。
 */
import SummaryView from './SummaryView'

export default function NotesPanel({ result, panelIndex = 0 }) {
  if (!result) return null
  const panel = result.panels?.[panelIndex]
  const nPanels = result.panels?.length || 1
  const notes = result.notes || []
  const nStats = Object.keys(result.summary || {}).length

  return (
    /* `.notes-clip` 是折叠动画的载体：它会真正缩到 0 宽，
       里面的 `.notes-body` 始终保持固定宽度（只是被裁掉，文字不重排）。
       这样折叠后右侧竖条一定落在面板可见区域内 —— 之前把固定宽度的内容
       直接放在竖条前面，折叠时内容占了全部宽度、把竖条挤到可视区之外，
       于是"折叠就再也点不开了"。 */
    <div className="notes-clip">
      <div className="notes-body">
        <div className="notes-head">
          <div className="ntitle" title={result.title}>{result.title}</div>
          <div className="nsub">
            {panel ? `${panel.title} · ` : ''}第 {panelIndex + 1} / {nPanels} 张
            {panel?.xscale === 'log' || panel?.yscale === 'log' ? ' · 对数轴' : ''}
          </div>
        </div>

        <div className="notes-scroll">
          <div className="sect-title" style={{ marginTop: 0 }}>
            图表说明
            <span className="dim" style={{ fontWeight: 400 }}>
              {' '}（{notes.length} 条）
            </span>
          </div>
          {notes.length > 0 ? (
            /* flex 悬挂缩进：换行后的文字与首行左缘对齐，不会缩到项目符号底下 */
            <div className="notelist">
              {notes.map((n, i) => (
                <div key={i} className="noterow">
                  <span className="notebullet">·</span>
                  <span className="notetext">{n}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="hint">这一项没有额外的算法说明。</div>
          )}

          <div className="sect-title" style={{ marginTop: 12 }}>
            关键统计量
            <span className="dim" style={{ fontWeight: 400 }}>
              {' '}（{nStats} 项）
            </span>
          </div>
          {/* 表格约束：固定列宽 + 单元格换行，超长键名换行而不横向滚动 */}
          <SummaryView result={result} tableClass="notes-table" />
        </div>

        <div className="notes-foot dim small">
          完整统计量见「统计量」页；这里只显示当前这张图相关的部分。
        </div>
      </div>
    </div>
  )
}
