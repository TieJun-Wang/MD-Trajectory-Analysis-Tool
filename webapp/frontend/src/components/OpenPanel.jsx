/**
 * 1. 文件读取 —— 内容部分（放进折叠面板，不带卡片外壳）。
 */
import { useEffect, useState } from 'react'
import { api, guessTrajectory } from '../api'

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

  const open = async () => {
    if (!topology) { setError('请先选择拓扑文件（.tpr/.gro/.pdb）'); return }
    setBusy(true); setError('')
    try {
      const data = await api.open(topology, trajectory)
      log?.(`体系已打开: sid=${data.sid}, ${data.info.n_atoms} 原子, `
        + `${data.info.n_frames} 帧`)
      onOpened(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const short = (p) => (p ? p.split(/[\\/]/).slice(-2).join('/') : '未选择')

  return (
    <>
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
