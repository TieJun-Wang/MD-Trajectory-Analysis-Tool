/**
 * 2. 体系信息 —— 内容部分（放进折叠面板，不带卡片外壳）。
 *
 * 完整报告用**表格**渲染（后端 `info.sections` 给出「列 + 行」结构），
 * 不再把定宽文本报告塞进 <pre>：定宽文本在窄栏里要么被截断、
 * 要么逼着用户左右拖动。
 */
import { useState } from 'react'

function fmt(v, digits = 3) {
  if (v === null || v === undefined) return '—'
  if (Array.isArray(v)) return v.map((x) => Number(x).toFixed(digits)).join(' × ')
  if (typeof v === 'number') return Number(v.toPrecision(6)).toLocaleString()
  return String(v)
}

/** 一行摘要：原子数 / 帧数 / 盒 —— 始终可见，不用展开 */
function summaryRows(info) {
  return [
    ['原子数', info.n_atoms?.toLocaleString()],
    ['residue 数', info.n_residues?.toLocaleString()],
    ['段 (segment)', info.n_segments],
    ['帧数', info.n_frames?.toLocaleString()],
    ['时间间隔', `${fmt(info.dt_ps, 4)} ps`],
    ['模拟时间', `${fmt(info.total_time_ps / 1000)} ns`
      + `（${fmt(info.first_time_ps)}–${fmt(info.last_time_ps)} ps）`],
    ['模拟盒', info.box_dimensions
      ? `${fmt(info.box_dimensions)} Å（${info.box_type === 'triclinic' ? '三斜' : '正交'}）`
      : '—'],
    ['盒体积', info.box_volume ? `${fmt(info.box_volume, 1)} Å³` : '—'],
    ['总质量', `${fmt(info.total_mass_amu, 1)} amu`],
    ['净电荷', info.total_charge_e !== null && info.total_charge_e !== undefined
      ? `${info.total_charge_e.toFixed(4)} e` : '—'],
    ['键/角/二面角', `${info.n_bonds} / ${info.n_angles} / ${info.n_dihedrals}`],
  ]
}

/** 通用表格：列 + 行，全部来自后端结构化的 sections */
function SectionTable({ section }) {
  const cols = section.columns || []
  return (
    <div className="infosec">
      <div className="infosec-title">
        {section.title}
        <span className="dim" style={{ fontWeight: 400 }}>
          {' '}（{section.rows?.length || 0} 行）
        </span>
      </div>
      <table className="kv inforows">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {(section.rows || []).map((r, i) => (
            <tr key={i}>
              {r.map((cell, j) => (
                <td key={j} className={j === 0 ? 'cell-key' : ''}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {section.note && <div className="infosec-note">{section.note}</div>}
    </div>
  )
}

/**
 * 完整报告：把后端的结构化小节渲染成一串表格。
 *
 * 纵向滚动、横向锁定 —— 单元格内容自动换行，不会出现左右滚动条。
 * 单独导出是为了能直接渲染测试（不必先「展开」）。
 */
export function InfoReport({ sections }) {
  if (!sections || !sections.length) return null
  return (
    <div className="inforeport">
      {sections.map((s) => <SectionTable key={s.title} section={s} />)}
    </div>
  )
}

export default function InfoPanel({ info, onNext }) {
  const [expanded, setExpanded] = useState(false)
  if (!info) return null

  const rows = summaryRows(info)
  const sections = info.sections || []

  return (
    <>
      <table className="kv">
        <tbody>
          {rows.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}
        </tbody>
      </table>

      <div className="chips">
        {Object.entries(info.categories || {}).map(([k, v]) => (
          <span key={k} className="chip">{k}: {v}</span>
        ))}
      </div>

      {sections.length > 0 ? (
        <>
          <button className="link" onClick={() => setExpanded((s) => !s)}>
            {expanded ? '收起完整报告 ▲' : '展开完整报告 ▼'}
          </button>
          {expanded && <InfoReport sections={sections} />}
        </>
      ) : (
        /* 后端没给出结构化小节时（旧版本）退回文本，但同样锁定横向滚动 */
        <>
          <button className="link" onClick={() => setExpanded((s) => !s)}>
            {expanded ? '收起完整报告 ▲' : '展开完整报告 ▼'}
          </button>
          {expanded && (
            <div className="inforeport"><pre className="report">{info.text}</pre></div>
          )}
        </>
      )}

      {onNext && (
        <button className="primary mt6" onClick={onNext}>下一步：分析设置 →</button>
      )}
    </>
  )
}
