/**
 * 3. 分析设置 —— 把「分析对象 + 参数设置 + 分析功能 + 运行」合成一块，
 *    放进同一个折叠面板里，所以三块都用紧凑排版、不带卡片外壳。
 */
import { api } from '../api'
import { normalizeGroups } from '../analysisGroups'

//: 运行日志留一小条在设置板块底部，完整日志在右侧「运行日志」标签页
export function SelectionBlock({ sess, primary, setPrimary, components, setComponents }) {
  if (!sess) return null
  const chains = sess.chains || []
  const comps = sess.components || []

  const toggleComponent = (name) => {
    setComponents(components.includes(name)
      ? components.filter((c) => c !== name)
      : [...components, name])
  }

  return (
    <>
      <div className="sect-title">分析对象</div>

      <label className="field">
        <span>主链（构象 / 结晶分析对象）</span>
        <select
          value={primary.mode === 'custom' ? 'custom' : JSON.stringify(primary)}
          onChange={(e) => {
            const v = e.target.value
            setPrimary(v === 'custom'
              ? { mode: 'custom', query: primary.query || '' }
              : JSON.parse(v))
          }}>
          <option value={JSON.stringify({ mode: 'auto' })}>
            自动（最大链：{sess.primary_label}，{sess.primary_atoms} 原子）
          </option>
          {comps.map((c) => (
            <option key={c.name} value={JSON.stringify({ mode: 'component', name: c.name })}>
              组分 {c.name}（{c.n_atoms} 原子）
            </option>
          ))}
          {chains.map((c) => (
            <option key={c.label}
                    value={JSON.stringify({ mode: 'chain', segid: c.segid,
                                            resname: c.resname, min_atoms: c.n_atoms })}>
              {c.label}（{c.description}）
            </option>
          ))}
          <option value="custom">自定义选择语句…</option>
        </select>
      </label>

      {primary.mode === 'custom' && (
        <label className="field">
          <span>MDAnalysis 选择语句</span>
          <input value={primary.query || ''}
                 placeholder="例如 resid 1:200 或 segid seg_0_PE"
                 onChange={(e) => setPrimary({ mode: 'custom', query: e.target.value })} />
        </label>
      )}

      <div className="field">
        <span>参与界面 / RDF / MSD 分析的组分（点选切换）</span>
        <div className="chips-pick">
          {comps.map((c) => (
            <button key={c.name}
                    className={`chip-pick ${components.includes(c.name) ? 'on' : ''}`}
                    onClick={() => toggleComponent(c.name)}
                    title={`${c.n_atoms.toLocaleString()} 原子 / ${c.n_residues} 残基`}>
              {c.name}
            </button>
          ))}
        </div>
      </div>
    </>
  )
}

export function ParamBlock({ frames, setFrames, params, setParams }) {
  const f = (key) => ({
    value: frames[key] ?? '',
    onChange: (e) => setFrames({ ...frames, [key]: e.target.value }),
  })
  const p = (key) => ({
    value: params[key] ?? '',
    onChange: (e) => setParams({ ...params, [key]: e.target.value }),
  })

  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>参数设置</div>
      <div className="grid3">
        <label className="field"><span>起始 (ps)</span>
          <input type="number" placeholder="首帧" {...f('start_ps')} /></label>
        <label className="field"><span>结束 (ps)</span>
          <input type="number" placeholder="末帧" {...f('stop_ps')} /></label>
        <label className="field"><span>平衡段 (ps)</span>
          <input type="number" placeholder="不统计" {...f('equil_ps')} /></label>
        <label className="field"><span>抽帧间隔 (ps)</span>
          <input type="number" placeholder="每帧" {...f('interval_ps')} /></label>
        <label className="field"><span>最多帧数</span>
          <input type="number" placeholder="不限" {...f('max_frames')} /></label>
        <label className="field"><span>接触 cutoff (Å)</span>
          <input type="number" step="0.5" {...p('cutoff')} /></label>
        <label className="field"><span>密度方向</span>
          <select value={params.axis}
                  onChange={(e) => setParams({ ...params, axis: e.target.value })}>
            <option value={2}>c (2)</option>
            <option value={1}>b (1)</option>
            <option value={0}>a (0)</option>
          </select></label>
        <label className="field"><span>密度 bin 数</span>
          <input type="number" {...p('nbins')} /></label>
        <label className="field"><span>RDF 最大 r (Å)</span>
          <input type="number" step="1" {...p('rmax')} /></label>
        <label className="field col-span-all"><span>二面角模式</span>
          <select value={params.dihedral_mode}
                  onChange={(e) => setParams({ ...params, dihedral_mode: e.target.value })}>
            <option value="auto">auto（自动）</option>
            <option value="phi_psi">phi_psi（蛋白质主链 φ/ψ）</option>
            <option value="chain">chain（链骨架）</option>
            <option value="topology">topology（拓扑定义）</option>
          </select></label>
      </div>
    </>
  )
}

export function FunctionBlock({ titles = {}, order = [], which = [], setWhich,
                                groups }) {
  const toggle = (n) => setWhich(which.includes(n)
    ? which.filter((x) => x !== n) : [...which, n])

  // 分组来自 /api/analyses 的 groups（权威定义在 mdta.pipeline.ANALYSIS_GROUPS），
  // normalizeGroups 里带本地兜底
  const GROUPS = normalizeGroups(groups)

  // 某个「仅 XX」预设是否正好等于当前选择（用于高亮当前所处的预设）
  const isPreset = (names) =>
    names.length === which.length && names.every((n) => which.includes(n))

  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>
        分析功能
        <span className="dim" style={{ fontWeight: 400 }}>
          {' '}（已选 {which.length} / {order.length}）
        </span>
      </div>
      {GROUPS.map(([group, names]) => (
        <div key={group} className="fgroup">
          <span className="glabel">{group}</span>
          <div className="gitems">
            {names.map((n) => (
              <label key={n} className="check">
                <input type="checkbox" checked={which.includes(n)}
                       onChange={() => toggle(n)} />
                {titles[n] || n}
              </label>
            ))}
          </div>
        </div>
      ))}
      <div className="btnrow">
        <button className="mini" onClick={() => setWhich([...order])}>全选</button>
        {GROUPS.map(([group, names]) => (
          <button key={group}
                  className={`mini ${isPreset(names) ? 'active' : ''}`}
                  title={`只勾选「${group}」这一类（${names.map(
                    (n) => titles[n] || n).join('、')}）；要全部取消就逐个点掉勾选框`}
                  onClick={() => setWhich([...names])}>
            仅{group}
          </button>
        ))}
      </div>
    </>
  )
}

export function RunBlock({ busy, progress, onRun, onCancel, lastLine, onOpenLog,
                          live, which = [], titles = {}, run }) {
  const done = live?.nDone ?? (run ? Object.keys(run.results || {}).length : 0)
  const total = live?.nTotal || which.length || 0
  const pending = live?.pending || []
  const pct = Math.round((busy ? (live?.frac ?? progress) : progress) * 100)

  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>
        运行
        {busy && <span className="badge-live">● 实时更新中</span>}
      </div>
      <button className="primary" onClick={onRun} disabled={busy}>
        {busy ? `分析中… ${done}/${total}` : '开始分析'}
      </button>

      {busy && (
        <>
          <div className="progress">
            <div className="bar" style={{ width: `${pct}%` }} />
          </div>
          <div className="hint" title={live?.message}>
            {live?.message || '准备中…'}
            {live?.elapsed ? `（已用 ${live.elapsed.toFixed(0)}s）` : ''}
          </div>
          {pending.length > 0 && (
            <div className="hint dim" title={pending.map((n) => titles[n] || n).join('、')}>
              排队中：{pending.map((n) => titles[n] || n).join('、')}
            </div>
          )}
          {onCancel && (
            <button className="mini mt6" onClick={onCancel}>取消（保留已算完的）</button>
          )}
        </>
      )}

      {lastLine && !busy && (
        <div className="hint" title={lastLine}>
          {lastLine.length > 56 ? lastLine.slice(0, 56) + '…' : lastLine}
        </div>
      )}
      {onOpenLog && (
        <button className="link" onClick={onOpenLog}>查看完整运行日志 →</button>
      )}
    </>
  )
}

export default { SelectionBlock, ParamBlock, FunctionBlock, RunBlock }
