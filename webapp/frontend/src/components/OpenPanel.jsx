/**
 * 1. 文件读取 —— 内容部分（放进折叠面板，不带卡片外壳）。
 */
import { useEffect, useState } from 'react'
import { api, assignFile, fileRoles, streamOpenProgress } from '../api'

export default function OpenPanel({ onOpened, log }) {
  const [dir, setDir] = useState('')
  const [listing, setListing] = useState(null)
  const [topology, setTopology] = useState('')
  const [trajectory, setTrajectory] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
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
  //:
  //: ⚠️ 1.0.4 起**不再自动匹配同名轨迹**（用户要求）：有时候只想看拓扑信息，
  //: 一选拓扑就被塞进一个轨迹槽反而挡路。而且**再点一次同一个文件 = 取消**，
  //: 这样误选的文件随时能摘掉。
  const pick = (file) => {
    const { role, next } = assignFile(file, { topology, trajectory })
    if (role === 'topology') {
      if (topology === file.path) {          // 再点一次 → 取消
        setTopology('')
        log?.(`已取消拓扑文件: ${file.name}`)
        return
      }
      setTopology(next.topology)
      log?.(`选择拓扑文件: ${file.name}`)
    } else {
      if (trajectory === file.path) {        // 再点一次 → 取消
        setTrajectory('')
        log?.(`已取消轨迹文件: ${file.name}`)
        return
      }
      setTrajectory(next.trajectory)
      log?.(`选择轨迹文件: ${file.name}`)
    }
  }

  /** 「两者皆可」的格式：明确指定放到哪个槽（主点击 = 拓扑，这里可以改成轨迹） */
  const useAs = (file, role) => {
    if (role === 'topology') {
      if (topology === file.path) {
        setTopology('')
        log?.(`已取消拓扑文件: ${file.name}`)
        return
      }
      setTopology(file.path)
      log?.(`用作拓扑: ${file.name}`)
    } else {
      if (trajectory === file.path) {
        setTrajectory('')
        log?.(`已取消轨迹文件: ${file.name}`)
        return
      }
      setTrajectory(file.path)
      log?.(`用作轨迹: ${file.name}`)
    }
  }

  // 「文件读取」进度：打开大轨迹首次要扫全文件（46 体系 1.7 GB 实测 >135 s），
  // 这里一边等 POST 返回、一边并发轮询后端记录的阶段，给出进度条与"当前在做什么"。
  const [phase, setPhase] = useState(null)
  // 读取已用时间：每秒自增，配合"动态闪烁"告诉用户还在动
  const [openElapsed, setOpenElapsed] = useState(0)
  useEffect(() => {
    if (!busy) { setOpenElapsed(0); return undefined }
    const t0 = Date.now()
    setOpenElapsed(0)
    const timer = setInterval(() => setOpenElapsed((Date.now() - t0) / 1000), 500)
    return () => clearInterval(timer)
  }, [busy])

  const open = async () => {
    if (!topology) { setError('请先选择拓扑文件（.tpr/.gro/.pdb）'); return }
    setBusy(true); setError('')
    const token = `open-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setPhase({ stage: '准备中', stages: [], done: false })
    const t0 = Date.now()

    // **流式**接收读取进度（SSE）：阶段一变就推过来；断流自动退回轮询
    let stopStream = () => {}
    try {
      stopStream = streamOpenProgress(token, { onStage: (p) => setPhase(p) })
    } catch (e) {
      stopStream = () => {}                     // 订阅失败也不该阻断打开
    }
    try {
      // 用户要求：MS 的 .trj 一律转 TRR（坐标 + 速度），不再给复选框
      const data = await api.open(topology, trajectory, token, false, true, false)
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

/** 秒 → mm:ss（超过 1 小时显示 h:mm:ss），给进度条旁边的"已用时间"用。 */
function fmtElapsed(sec) {
  const t = Math.max(0, Math.floor(Number(sec) || 0))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const ss = String(t % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

  const short = (p) => (p ? p.split(/[\\/]/).slice(-2).join('/') : '未选择')

  return (
    <>
      {busy && (
        // 只保留「进度条 + 当前工作项」。
        // 以前这里还显示"已用 X s"、首次扫描提示、以及最近 4 个阶段各自的耗时
        // （`· 阶段名  0.4 s`）—— 用户要求删掉这些计时展示，只留进度与当前在做什么。
        <div className="openprog">
          {/* 阶段数推出来的百分比；刚开始阶段还少时进度条走"不确定"样式（来回跑） */}
          <div className="progress">
            <div className={`bar shimmer${(phase?.n_stages || 0) < 2 ? ' indeterminate' : ''}`}
                 style={{ width: `${Math.min(96, 12 + (phase?.n_stages || 0) * 17)}%` }} />
          </div>
          <div className="openmeta dim small progress-meta">
            <span>正在读取文件：{phase?.stage || '准备中'}</span>
            <span className="elapsed" title="本次读取已用时间">
              已用 {fmtElapsed(openElapsed)}
            </span>
          </div>
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
            // 后端按**内容**嗅出来的警示（LAMMPS 的 in.lammps / log.lammps 与 data
            // 文件共用后缀，只靠后缀分不出来）；老后端没有这个字段，为 undefined
            const note = f.note ? String(f.note) : ''
            return (
              <div key={f.path} className="filerow">
                <button className={`item file ${canTop ? 'topology' : 'trajectory'}`}
                        onClick={() => pick(f)}
                        title={`${f.path}${note ? '\n⚠ ' + note : ''}\n${dual
                          ? '该格式拓扑/轨迹都能用：点一下进**拓扑**槽；'
                            + '想当轨迹请点右侧「轨迹」'
                          : canTop ? '点击选择为拓扑' : '点击选择为轨迹'}`}>
                  {canTop ? '🧬' : '📈'}
                  <span className="fname">{f.name}</span>
                  {note ? <span className="fnote" title={note}>⚠</span> : null}
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
      </div>

      {loadNotes.length > 0 && (
        <div className="hint dim" style={{ fontSize: 10.5 }}
             title={loadNotes.join('\n')}>
          读取诊断：{loadNotes[0].slice(0, 42)}…（共 {loadNotes.length} 条，悬停看全部）
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {/* 速度数据开关已按用户要求删掉：MS 的 .trj 一律转 TRR（坐标 + 速度）。 */}

      {/* 「自动配对同名轨迹」开关已按用户要求删掉：固定为**只按选中的文件**读。
          要跑动力学就把轨迹放进「轨迹」槽。 */}

      <button className="primary mt6" onClick={open} disabled={busy || !topology}>
        {busy ? '正在读取…' : '读取体系'}
      </button>
    </>
  )
}
