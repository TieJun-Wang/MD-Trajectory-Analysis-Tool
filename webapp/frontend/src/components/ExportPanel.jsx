/**
 * 「导出」标签页 —— **只负责导出功能**，不显示统计量明细。
 *
 * 目标目录（内置目录浏览器）、导出模块、文件形式、开始导出、打开目标目录，
 * 以及导出完成后的「最近导出：目录 + 时间」。
 */
import { useEffect, useState } from 'react'
import { api } from '../api'
import { groupPresent, normalizeGroups } from '../analysisGroups'

const FORMATS = [
  ['csv', 'CSV 数据表'],
  ['png', 'PNG 图'],
  ['excel', 'Excel 汇总'],
]

/** 目标目录选择器：复用 /api/files 的目录浏览能力 */
export function DirPicker({ value, onPick, onClose }) {
  const [cur, setCur] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const go = async (dir) => {
    setBusy(true); setErr('')
    try {
      setCur(await api.files(dir))
    } catch (e) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }
  useEffect(() => { go(value || undefined) }, [])   // eslint-disable-line

  return (
    <div className="dirpicker">
      <div className="dp-head">
        <button className="mini" disabled={!cur?.parent} onClick={() => go(cur.parent)}>
          ↑ 上级
        </button>
        <code className="dp-path" title={cur?.dir}>{cur?.dir || '读取中…'}</code>
        <button className="mini" onClick={onClose}>关闭</button>
      </div>
      {err && <div className="error inline-error">{err}</div>}
      <div className="dp-list">
        {(cur?.drives || []).length > 1 && (
          <div className="dp-drives">
            {cur.drives.map((d) => (
              <button key={d} className="mini" onClick={() => go(d)}>{d}</button>
            ))}
          </div>
        )}
        {(cur?.dirs || []).map((d) => (
          <button key={d.path} className="dp-item" onClick={() => go(d.path)}>
            📁 {d.name}
          </button>
        ))}
        {cur && !busy && !(cur.dirs || []).length && (
          <div className="hint">这个目录下没有子目录。</div>
        )}
      </div>
      <div className="dp-foot">
        <button className="primary" disabled={!cur}
                onClick={() => { onPick(cur.dir); onClose() }}>
          用这个目录
        </button>
        <span className="dim small">不选就用会话默认输出目录</span>
      </div>
    </div>
  )
}

export default function ExportPanel({ run, sid, log, groups }) {
  const results = run?.results || {}
  const order = Object.keys(results)
  const mods = groupPresent(order, normalizeGroups(groups))

  const [dpi, setDpi] = useState(200)
  const [panelPngs, setPanelPngs] = useState(false)
  const [formats, setFormats] = useState(['csv', 'png', 'excel'])
  const [whichMods, setWhichMods] = useState({})       // {模块名: bool}
  const [outdir, setOutdir] = useState('')
  const [showPicker, setShowPicker] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [last, setLast] = useState(null)

  // 模块勾选默认全开；新出现的模块也自动勾上
  const modOn = (label) => whichMods[label] !== false
  const toggleMod = (label) => setWhichMods((s) => ({ ...s, [label]: !modOn(label) }))
  const chosenNames = mods.filter(([l]) => modOn(l)).flatMap(([, ns]) => ns)
  const toggleFormat = (f) => setFormats((s) => (
    s.includes(f) ? s.filter((x) => x !== f) : [...s, f]))

  const doExport = async () => {
    if (!sid) return
    if (!chosenNames.length) { setError('请至少勾选一个导出模块'); return }
    if (!formats.length) { setError('请至少选择一种文件形式'); return }
    setBusy(true); setError('')
    try {
      const data = await api.export(sid, {
        formats, dpi: Number(dpi) || 200, panel_pngs: panelPngs,
        which: chosenNames, outdir: outdir || null,
      })
      setLast(data)
      log?.(`已导出到 ${data.dir}（CSV ${data.n_csv}，PNG ${data.n_png}）`)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const openDir = async () => {
    try {
      const dir = last?.dir || outdir || undefined
      await api.openDir(sid, dir)
      log?.(`已在文件管理器中打开 ${last?.dir || outdir || '默认输出目录'}`)
    } catch (e) {
      setError(e.message)
    }
  }

  if (!order.length) {
    return <div className="placeholder"><p>还没有结果。</p></div>
  }

  return (
    <div className="statsview">
      <div className="exportbar">
        <div className="exptitle">
          <strong>导出结果</strong>
          <span className="dim">{chosenNames.length} / {order.length} 项分析</span>
        </div>

        {/* ---------------------------------------------- 目标目录 */}
        <div className="exprow">
          <span className="explabel">目标目录</span>
          <code className="exppath" title={outdir || '（默认：会话输出目录）'}>
            {outdir || '（默认：会话输出目录）'}
          </code>
          <button className="mini" onClick={() => setShowPicker((v) => !v)}>选择…</button>
          {outdir && <button className="mini" onClick={() => setOutdir('')}>用默认</button>}
        </div>
        {showPicker && (
          <DirPicker value={outdir} onPick={setOutdir}
                     onClose={() => setShowPicker(false)} />
        )}

        {/* ---------------------------------------------- 模块与格式 */}
        <div className="exprow">
          <span className="explabel">导出模块</span>
          <div className="chiprow">
            {mods.map(([label, ns]) => (
              <label key={label} className={`chinline ${modOn(label) ? 'on' : ''}`}>
                <input type="checkbox" checked={modOn(label)}
                       onChange={() => toggleMod(label)} />
                {label}
                <span className="dim">（{ns.length}）</span>
              </label>
            ))}
            <button className="mini" onClick={() => setWhichMods({})}>全选</button>
          </div>
        </div>

        <div className="exprow">
          <span className="explabel">文件形式</span>
          <div className="chiprow">
            {FORMATS.map(([f, label]) => (
              <label key={f} className={`chinline ${formats.includes(f) ? 'on' : ''}`}>
                <input type="checkbox" checked={formats.includes(f)}
                       onChange={() => toggleFormat(f)} />
                {label}
              </label>
            ))}
            <label className="inline">dpi
              <input type="number" className="tiny" value={dpi}
                     onChange={(e) => setDpi(e.target.value)} />
            </label>
            <label className="inline">
              <input type="checkbox" checked={panelPngs}
                     onChange={(e) => setPanelPngs(e.target.checked)} />
              每个面板单独出图
            </label>
          </div>
        </div>

        <div className="exprow expactions">
          <button className="primary" onClick={doExport} disabled={busy || !sid}>
            {busy ? '导出中…' : '开始导出'}
          </button>
          {/* 打开目标目录（排在开始导出右边） */}
          <button className="mini" onClick={openDir} disabled={!sid}
                  title="在文件管理器中打开目标目录">
            📂 打开目标目录
          </button>
          {error && <span className="error inline-error">{error}</span>}
        </div>

        {/* 导出完成后只报「目录 + 时间」，不列文件 */}
        {last && (
          <div className="lastout">
            <span className="ok">✓ 最近导出</span>
            <span className="lo-time">{last.time}</span>
            <code className="lo-dir" title={last.dir}>{last.dir}</code>
            <span className="dim">
              （CSV {last.n_csv} / PNG {last.n_png}
              {last.excel ? ` / ${String(last.excel).split(/[\\/]/).pop()}` : ''}）
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
