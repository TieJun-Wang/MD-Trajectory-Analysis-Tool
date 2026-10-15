/**
 * 1. 文件读取 —— 内容部分（放进折叠面板，不带卡片外壳）。
 */
import { useEffect, useState } from 'react'
import {api, assignFile, fileRoles, guessTrajectory, streamOpenProgress } from '../api'

export default function OpenPanel({ onOpened, log }) {
  const [dir, setDir] = useState('')
  const [listing, setListing] = useState(null)
  const [topology, setTopology] = useState('')
  const [trajectory, setTrajectory] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  //: 忽略拓扑里已有的键、按坐标重新猜。给 CONECT 不完整的 PDB（Material Studio
  //: 导出的常见）用：实测某体系自带 4393 个键 → 键图碎片 1203 个（分子被切碎），
  //: 重新猜键后 5953 键 / 124 个分子。
  const [reguess, setReguess] = useState(false)
  //: 上一次读取的加载诊断（猜键/分子划分/缓存重建），读进来后显示给用户
  const [loadNotes, setLoadNotes] = useState([])

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

  //: 文件按"能放哪个槽"分三类：topology / trajectory / **both**（`.pdb` `.gro` `.xyz`
  //: 这类多帧文本格式两个槽都能用）。判定与"该填哪个槽"的逻辑在 api.js 的
  //: `fileRoles` / `assignFile` 里（纯函数，有自检），这里只负责更新 state。
  //:
  //: ⚠️ 规则是"**能给拓扑就一定进拓扑槽**"：`.pdb` 本来就是结构文件，
  //: 主点击绝不能把它塞进轨迹槽。用户实际踩过的坑就是 `.pdb` 掉进了轨迹槽
  //: （后端先返回了 `kind="both"`，老前端只认 `kind === 'topology'`）。
  const pick = (file) => {
    const { role, next } = assignFile(file, { topology, trajectory })
    if (role === 'topology') {
      setTopology(next.topology)
      const prev = trajectory
      log?.(`选择拓扑文件: ${file.name}`)
      if (listing) {
        const guess = guessTrajectory(file.path, listing.files)
        if (guess && !prev) {
          setTrajectory(guess)
          log?.(`自动匹配到同名轨迹: ${guess.split(/[\\/]/).pop()}`)
        }
      }
    } else {
      setTrajectory(next.trajectory)
      log?.(`选择轨迹文件: ${file.name}`)
    }
  }

  /** 「两者皆可」的格式：明确指定放到哪个槽（主点击 = 拓扑，这里可以改成轨迹） */
  const useAs = (file, role) => {
    if (role === 'topology') {
      setTopology(file.path)
      log?.(`用作拓扑: ${file.name}`)
      if (listing) {
        const guess = guessTrajectory(file.path, listing.files)
        if (guess && guess !== file.path) {
          setTrajectory(guess)
          log?.(`自动匹配到同名轨迹: ${guess.split(/[\\/]/).pop()}`)
        }
      }
    } else {
      setTrajectory(file.path)
      log?.(`用作轨迹: ${file.name}`)
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
      const data = await api.open(topology, trajectory, token, reguess)
      log?.(`体系已打开: sid=${data.sid}, ${data.info?.n_atoms ?? '?'} 原子, `
        + `${data.info?.n_frames ?? '?'} 帧（用时 ${((Date.now() - t0) / 1000).toFixed(1)} s）`)
      const notes = data.load_notes || []
      setLoadNotes(notes)
      notes.forEach((n) => log?.(`  · ${n}`))
      onOpened(data)
    } catch (e) {
      setError(e.message)
      setLoadNotes([])
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
          {(listing?.files || []).map((f) => {
            const roles = fileRoles(f)
            const canTop = roles.includes('topology')
            const canTraj = roles.includes('trajectory')
            const dual = canTop && canTraj
            return (
              <div key={f.path} className="filerow">
                <button className={`item file ${canTop ? 'topology' : 'trajectory'}`}
                        onClick={() => pick(f)}
                        title={`${f.path}\n${dual
                          ? '该格式拓扑/轨迹都能用：点一下进**拓扑**槽；'
                            + '想当轨迹请点右侧「轨迹」'
                          : canTop ? '点击选择为拓扑' : '点击选择为轨迹'}`}>
                  {canTop ? '🧬' : '📈'}
                  <span className="fname">{f.name}</span>
                  <span className="size">{(f.size / 1048576).toFixed(1)} MB</span>
                </button>
                {dual && (
                  <span className="roles">
                    <button className="mini" title="把这个文件用作拓扑（主点击的默认行为）"
                            onClick={() => useAs(f, 'topology')}>拓扑</button>
                    <button className="mini" title="把这个文件用作轨迹"
                            onClick={() => useAs(f, 'trajectory')}>轨迹</button>
                  </span>
                )}
              </div>
            )
          })}
          {listing && !listing.dirs.length && !listing.files.length && (
            <div className="empty">此目录下没有子目录或 MD 文件</div>
          )}
        </div>
        {listing?.help && (
          <div className="hint dim" style={{ fontSize: 10.5 }}
               title={`拓扑：${listing.help.topology}\n轨迹：${listing.help.trajectory}`}>
            支持格式：拓扑 {String(listing.help.topology).split(' ').length} 种、
            轨迹 {String(listing.help.trajectory).split(' ').length} 种
            （鼠标悬停看全部；带「拓扑/轨迹」小按钮的文件两个槽都能用）
          </div>
        )}
      </div>

      {/* 键表重猜：给 CONECT 不完整的 PDB（Material Studio 导出常见）用。
          实测某体系自带 4393 键 → 分子被切成 1203 个碎片；重新猜键 5953 键 /
          124 个分子，逐分子 Rg/R_ee、分子质心 MSD、RDF 同分子剔除才对。 */}
      <label className="check" style={{ marginTop: 6 }}
             title={'勾上后忽略拓扑里已有的键（.pdb 的 CONECT / .gro 无键），'
               + '按共价半径重新猜键。适用于 CONECT 不完整的 PDB：'
               + '实测某 Material Studio 体系自带 4393 个键 → 分子被切成 1203 个碎片，'
               + '重新猜键后 5953 个键 / 124 个分子。'
               + '注意：力场拓扑（.tpr/.psf/.prmtop）本身键表可信，不要勾。'}>
        <input type="checkbox" checked={reguess} disabled={busy}
               onChange={(e) => setReguess(e.target.checked)} />
        忽略拓扑里的键，按坐标重新猜键
      </label>

      {loadNotes.length > 0 && (
        <div className="hint dim" style={{ fontSize: 10.5 }}
             title={loadNotes.join('\n')}>
          读取诊断：{loadNotes[0].slice(0, 42)}…（共 {loadNotes.length} 条，悬停看全部）
        </div>
      )}

      {error && <div className="error">{error}</div>}

      <button className="primary mt6" onClick={open} disabled={busy || !topology}>
        {busy ? '正在读取…' : '读取体系'}
      </button>
    </>
  )
}
