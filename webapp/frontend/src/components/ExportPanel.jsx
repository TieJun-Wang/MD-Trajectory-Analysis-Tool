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

/**
 * 列出某个分析在当前格式设置下**会产出哪些文件**。
 * 命名规则与 mdta/export.py 保持一致：
 *   单面板 → ``rg.csv`` / ``rg.png``；多面板 → ``rg__panel0.csv``；
 *   每面板单独出图 → ``rg__p1.png``；汇总 → ``summary.xlsx``。
 */
export function filesFor(res, name, formats, panelPngs) {
  const panels = res.panels || []
  const n = panels.length || 1
  const multi = n > 1
  const out = []
  if (formats.includes('csv')) {
    if (multi) {
      for (let i = 0; i < n; i += 1) {
        out.push({ file: `${name}__panel${i}.csv`, kind: 'CSV',
                   label: panels[i]?.title || `面板 ${i + 1}` })
      }
    } else {
      out.push({ file: `${name}.csv`, kind: 'CSV', label: '数据表' })
    }
  }
  if (formats.includes('png')) {
    out.push({ file: `${name}.png`, kind: 'PNG', label: `合并图（${n} 张面板）` })
    if (panelPngs) {
      for (let i = 0; i < n; i += 1) {
        out.push({ file: `${name}__p${i + 1}.png`, kind: 'PNG',
                   label: `单图 · ${panels[i]?.title || `面板 ${i + 1}`}` })
      }
    }
  }
  return out
}

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
  // 逐文件勾选：存"取消勾选的"，这样格式一变、新出现的文件默认是勾上的
  const [offFiles, setOffFiles] = useState(() => new Set())

  // 模块勾选默认全开；新出现的模块也自动勾上
  const modOn = (label) => whichMods[label] !== false
  const toggleMod = (label) => setWhichMods((s) => ({ ...s, [label]: !modOn(label) }))
  const chosenNames = mods.filter(([l]) => modOn(l)).flatMap(([, ns]) => ns)
  const toggleFormat = (f) => setFormats((s) => (
    s.includes(f) ? s.filter((x) => x !== f) : [...s, f]))

  // ---- 当前格式下每个模块会产出的文件（按大纲模块分板块展示）
  const fileMods = mods
    .filter(([label]) => modOn(label))
    .map(([label, ns]) => [label, ns.flatMap((n) => filesFor(
      results[n], n, formats,
      // 每面板单独出图只对 PNG 有意义
      panelPngs,
    ).map((f) => ({ ...f, name: n })))])

  const excelFile = formats.includes('excel')
    ? [{ file: 'summary.xlsx', kind: 'EXCEL', label: '全部统计量汇总', name: '__excel' }]
    : []
  const allFiles = [...fileMods.flatMap(([, fs]) => fs), ...excelFile]
  const fileOn = (f) => !offFiles.has(f)
  const toggleFile = (f) => setOffFiles((s) => {
    const next = new Set(s)
    if (next.has(f)) next.delete(f); else next.add(f)
    return next
  })
  const setModFiles = (fs, on) => setOffFiles((s) => {
    const next = new Set(s)
    fs.forEach((f) => (on ? next.delete(f.file) : next.add(f.file)))
    return next
  })
  const chosenFiles = allFiles.filter((f) => fileOn(f.file)).map((f) => f.file)

  const doExport = async () => {
    if (!sid) return
    if (!chosenNames.length) { setError('请至少勾选一个导出模块'); return }
    if (!formats.length) { setError('请至少选择一种文件形式'); return }
    if (!chosenFiles.length) { setError('请至少勾选一个要导出的文件'); return }
    setBusy(true); setError('')
    try {
      const data = await api.export(sid, {
        formats, dpi: Number(dpi) || 200, panel_pngs: panelPngs,
        which: chosenNames, outdir: outdir || null, files: chosenFiles,
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

      {/* 可选文件：按大纲模块分板块（与「统计量」页同一套分组与视觉），
          逐个文件勾选，导出时只产出勾上的那些。 */}
      <div className="filetabs">
        <span className="filetabs-title">选择要导出的文件</span>
        <span className="dim">
          已选 {chosenFiles.length} / {allFiles.length}
        </span>
        <button className="mini" onClick={() => setOffFiles(new Set())}>全选</button>
        <button className="mini"
                onClick={() => setOffFiles(new Set(allFiles.map((f) => f.file)))}>
          全不选
        </button>
      </div>

      <div className="stattabbody">
        {!allFiles.length ? (
          <div className="tabhint">
            当前没有可导出的文件 —— 请先在上面勾选「文件形式」。
          </div>
        ) : (
          <>
            {[...fileMods, ...(excelFile.length
              ? [['汇总表', excelFile]] : [])].map(([label, fs]) => {
              const onCount = fs.filter((f) => fileOn(f.file)).length
              return (
                <section key={label} className="filegroup">
                  <div className="filegroup-head">
                    <strong>{label}</strong>
                    <span className="dim">{onCount} / {fs.length}</span>
                    <button className="mini"
                            onClick={() => setModFiles(fs, true)}>全选</button>
                    <button className="mini"
                            onClick={() => setModFiles(fs, false)}>全不选</button>
                  </div>
                  <table className="stats filetable">
                    <thead>
                      <tr>
                        <th style={{ width: '46px' }}>选择</th>
                        <th>文件名</th>
                        <th style={{ width: '74px' }}>类型</th>
                        <th>内容</th>
                      </tr>
                    </thead>
                    <tbody>
                      {fs.map((f) => (
                        <tr key={f.file}>
                          <td>
                            <input type="checkbox" checked={fileOn(f.file)}
                                   onChange={() => toggleFile(f.file)}
                                   aria-label={f.file} />
                          </td>
                          <td><code>{f.file}</code></td>
                          <td>{f.kind}</td>
                          <td>{f.label}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}
