/**
 * 1. 文件读取 —— 内容部分（放进折叠面板，不带卡片外壳）。
 */
import { useEffect, useState } from 'react'
import {api, guessTrajectory, streamOpenProgress } from '../api'

export default function OpenPanel({ onOpened, log }) {
  const [dir, setDir] = useState('')
  const [listing, setListing] = useState(null)
  const [topology, setTopology] = useState('')
  const [trajectory, setTrajectory] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const browse = async (target) => {
    try {
      setError('')
      const data = await api.files(target)
      setListing(data)
      setDir(data.dir)
    } catch (e) {
      setError(e.message)
    }
  }
  useEffect(() => { browse('') }, [])

  const pick = (file) => {
    if (file.kind === 'topology') {
      setTopology(file.path)
      log?.(`选择拓扑文件: ${file.name}`)
      if (listing) {
        const guess = guessTrajectory(file.path, listing.files)
        if (guess) {
          setTrajectory(guess)
          log?.(`自动匹配到同名轨迹: ${guess.split(/[\\/]/).pop()}`)
        }
      }
    } else {
      setTrajectory(file.path)
      log?.(`选择轨迹文件: ${file.name}`)
    }
  }

  // 「文件读取」进度：打开大轨迹首次要扫全文件（46 体系 1.7 GB 实测 >135 s），
  // 这里一边等 POST 返回、一边并发轮询后端记录的阶段与耗时，给出进度条与已用时间。
  const [phase, setPhase] = useState(null)
  const [tick, setTick] = useState(0)

  const open = async () => {
    if (!topology) { setError('请先选择拓扑文件（.tpr/.gro/.pdb）'); return }
    setBusy(true); setError('')
    const token = `open-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setPhase({ stage: '准备中', elapsed: 0, stages: [], done: false })
    const t0 = Date.now()
    // **流式**接收读取进度（SSE）：阶段一变就推过来；断流自动退回轮询
    let stopStream = () => {}
    try {
      stopStream = streamOpenProgress(token, {
        onStage: (p) => setPhase(p),
        onTick: () => setTick((n) => n + 1),   // 本地秒表：保证时间一直在走
      })
    } catch (e) {
      stopStream = () => {}                     // 订阅失败也不该阻断打开
    }
    try {
      const data = await api.open(topology, trajectory, token)
      log?.(`体系已打开: sid=${data.sid}, ${data.info?.n_atoms ?? '?'} 原子, `
        + `${data.info?.n_frames ?? '?'} 帧（用时 ${((Date.now() - t0) / 1000).toFixed(1)} s）`)
      onOpened(data)
    } catch (e) {
      setError(e.message)
    } finally {
      // 清理绝不能因为某个变量未定义就中断——否则 busy 一直是 true、
      // 进度块会永远停在最后一个阶段（这正是之前踩过的坑：
      // 注释行吞掉了 `const stopStream = ...`，导致这里抛 ReferenceError）。
      try { stopStream() } catch { /* ignore */ }
      setBusy(false)
      setPhase(null)
    }
  }

  const short = (p) => (p ? p.split(/[\\/]/).slice(-2).join('/') : '未选择')

  return (
    <>
      {busy && (
        <div className="openprog">
          <div className="progress">
            <div className="bar" style={{ width: `${Math.min(
              96, 12 + (phase?.n_stages || 0) * 17)}%`,
            }} />
          </div>
          <div className="openmeta dim small">
            正在读取文件：{phase?.stage || '准备中'}
            {' '}· 已用 <b>{(((phase?.elapsed ?? 0) + (tick % 3) * 0.4)).toFixed(1)} s</b>
            {phase?.first_scan ? ' · 首次打开需扫描全文件建立帧索引（之后会走缓存）' : ''}
          </div>
          {(phase?.stages || []).slice(-4).map((s, i) => (
            <div key={i} className="openstage dim small">
              · {s.stage}{s.dt != null ? `　${s.dt} s` : ''}
            </div>
          ))}
        </div>
      )}
      <div className="picked">
        <div className="row">
          <span className="tag">拓扑</span>
          <span className={`path ${topology ? 'sel' : 'dim'}`} title={topology}>
            {short(topology)}
          </span>
        </div>
        <div className="row">
          <span className="tag">轨迹</span>
          <span className={`path ${trajectory ? 'sel' : 'dim'}`} title={trajectory}>
            {short(trajectory)}
          </span>
        </div>
      </div>

      <div className="browser">
        <div className="browser-head">
          <button className="mini" onClick={() => browse(listing?.parent || '')}
                  disabled={!listing?.parent}>↑</button>
          <input className="dirbox" value={dir}
                 onChange={(e) => setDir(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && browse(dir)} />
          <button className="mini" onClick={() => browse(dir)}>转到</button>
        </div>
        <div className="browser-list">
          {(listing?.drives || []).map((d) => (
            <button key={d} className="item dir" onClick={() => browse(d)}>💽 {d}</button>
          ))}
          {(listing?.dirs || []).map((d) => (
            <button key={d.path} className="item dir" onClick={() => browse(d.path)}
                    title={d.path}>📁 {d.name}</button>
          ))}
          {(listing?.files || []).map((f) => (
            <button key={f.path} className={`item file ${f.kind}`}
                    onClick={() => pick(f)} title={f.path}>
              {f.kind === 'topology' ? '🧬' : '📈'} {f.name}
              <span className="size">{(f.size / 1048576).toFixed(1)} MB</span>
            </button>
          ))}
          {listing && !listing.dirs.length && !listing.files.length && (
            <div className="empty">此目录下没有子目录或 MD 文件</div>
          )}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <button className="primary mt6" onClick={open} disabled={busy || !topology}>
        {busy ? '正在读取…' : '读取体系'}
      </button>
    </>
  )
}
