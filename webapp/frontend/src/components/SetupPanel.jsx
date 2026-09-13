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

// ------------------------------------------------------------ 参数区布局
// 用**表格约束**排版（用户建议）：`<table class="ptab">` 是 6 列基准网格，
// 一行最多放 3 个参数（label|input 各占列），同一列的输入框左边缘**天然对齐**。
// 每个参数按"标签宽 + 提示文字宽"决定占 2 / 3 / 6 列（即每行 3 / 2 / 1 个）。
//
// **提示文字必须完整显示**：不再只靠布局推算，而是给每个输入框直接设定
// 内联 `min-width = max(40, 提示文字宽 + 内边距12 + 边框2)`——这是元素级保证，
// 布局怎么变都不会截断。数字输入的上下箭头也一并隐藏（约占 18px 宽，
// 而且会让数字框与下拉框外观不一致、看着不齐）。
const LABEL_FS = 11.5
const INPUT_FS = 12.5
const CELL_3 = 113.3           // 三个一排时每格宽度（6 列基准、间距 4px）
const CELL_2 = 172.0           // 两个一排时每格宽度
const GRID_GAP = 4.0
const IN_MIN = 40.0            // 数字输入的绝对最小宽
const SEL_MIN = 70.0           // 下拉框最小宽（选项文字长）
const PH_CHROME = 14.0         // 输入框左右内边距 12 + 边框 2

const textPx = (t, fs) => [...String(t)].reduce(
  (w, ch) => w + (ch.codePointAt(0) < 0x2E80 ? 0.55 : 1.0) * fs, 0)
const labelPx = (t) => textPx(t, LABEL_FS)
/** 输入框的最小宽度：必须放得下提示文字（元素级保证）。 */
const inputMinWidth = (ph) => Math.max(IN_MIN, textPx(ph || '', INPUT_FS) + PH_CHROME)

/** 按需要宽度选跨度（列数）：2 = 一行 3 个、3 = 一行 2 个、6 = 独占一行。 */
function spanOf(label, kind = 'input', ph = '') {
  const inNeed = kind === 'select' ? SEL_MIN : inputMinWidth(ph)
  const need = labelPx(label) + GRID_GAP + inNeed
  if (need <= CELL_3) return 2
  if (need <= CELL_2) return 3
  return 6
}

/** 贪心装箱：每行跨度合计不超过 6。 */
function packRows(fields) {
  const rows = []
  let cur = []
  let used = 0
  for (const f of fields) {
    if (used + f.span > 6 && cur.length) { rows.push(cur); cur = []; used = 0 }
    cur.push(f)
    used += f.span
  }
  if (cur.length) rows.push(cur)
  return rows
}

/** 数字输入：内联 min-width 保证提示文字完整显示。 */
const numInput = (ph, props) => (
  <input type="number" placeholder={ph || undefined} title={ph || undefined}
         style={{ minWidth: inputMinWidth(ph) }} {...props} />
)

/** 参数表格：th（标签）+ td（输入），td 的 colSpan = 跨度 − 1。 */
function ParamTable({ fields }) {
  const rows = packRows(fields.filter(Boolean))
  if (!rows.length) return null
  return (
    <table className="ptab">
      <tbody>
        {rows.map((row, ri) => (
          <tr key={ri}>
            {row.flatMap((f) => [
              <th key={`${f.key}-l`} scope="row">{f.label}</th>,
              <td key={`${f.key}-i`} colSpan={f.span - 1}>{f.node}</td>,
            ])}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** 帧选择（所有分析共用）：独立板块，排在「分析功能」之前。 */
export function FrameBlock({ frames, setFrames }) {
  const f = (key) => ({
    value: frames[key] ?? '',
    onChange: (e) => setFrames({ ...frames, [key]: e.target.value }),
  })
  const SPEC = [
    ['start_ps', '起始 (ps)', '首帧'],
    ['stop_ps', '结束 (ps)', '末帧'],
    ['equil_ps', '平衡 (ps)', '不统计'],
    ['interval_ps', '抽帧 (ps)', '每帧'],
    ['max_frames', '最多帧数', '不限'],
  ]
  const fields = SPEC.map(([key, label, ph]) => ({
    key, label, span: spanOf(label, 'input', ph),
    node: numInput(ph, f(key)),
  }))
  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>帧选择</div>
      <ParamTable fields={fields} />
    </>
  )
}

/** 分析参数：按分析项分组，**只显示已选分析项的参数**（which=null → 全显示）。 */
export function ParamBlock({ params, setParams, which = null }) {
  const p = (key) => ({
    value: params[key] ?? '',
    onChange: (e) => setParams({ ...params, [key]: e.target.value }),
  })
  const sel = (key, opts) => (
    <select value={params[key]}
            onChange={(e) => setParams({ ...params, [key]: e.target.value })}>
      {opts.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
    </select>
  )
  const F = (key, label, kind, node, ph = '') => ({ key, label, span: spanOf(label, kind, ph), node })

  const show = (mods) => {
    if (mods === null) return true
    if (which === null || !Array.isArray(which)) return true
    return mods.some((m) => which.includes(m))
  }

  const GROUPS = [
    ['接触分析', ['contact'], [
      F('cutoff', 'cutoff (Å)', 'input', numInput('', { step: 0.5, ...p('cutoff') })),
      F('contact_mode', '接触配对模式', 'select', sel('contact_mode', [
        ['inter', '分子间（默认，剔除同分子配对）'],
        ['intra', '分子内（只算同一分子内部）'],
        ['total', '总体（含分子内，1.0.0 旧口径）'],
      ])),
    ]],
    ['密度分布 / 界面宽度', ['density', 'interface'], [
      F('axis', '密度方向', 'select', sel('axis', [[2, 'c (2)'], [1, 'b (1)'], [0, 'a (0)']])),
      F('nbins', '密度 bin 数', 'input', numInput('', p('nbins'))),
    ]],
    ['径向分布函数 RDF', ['rdf'], [
      F('rmax', 'RDF 最大 r (Å)', 'input', numInput('', { step: 1, ...p('rmax') })),
      F('rdf_mode', 'RDF 配对模式', 'select', sel('rdf_mode', [
        ['inter', '分子间（默认，g(r) 标准口径）'],
        ['intra', '分子内（同一分子内构象）'],
        ['total', '总体（含分子内，1.0.0 旧口径）'],
      ])),
    ]],
    ['二面角分析', ['dihedral'], [
      F('dihedral_mode', '二面角模式', 'select', sel('dihedral_mode', [
        ['auto', 'auto（自动）'],
        ['phi_psi', 'phi_psi（蛋白质主链 φ/ψ）'],
        ['chain', 'chain（几何连续原子模式：键图最长路径）'],
        ['topology', 'topology（拓扑定义）'],
      ])),
      F('gauche_edge', 'trans/gauche 阈值 (°)', 'input',
        numInput('', { step: 5, min: 60, max: 180, ...p('gauche_edge') })),
    ]],
    ['链段取向分析', ['orientation'], [
      F('orient_mode', '取向链段定义', 'select', sel('orient_mode', [
        ['repeat', 'repeat（化学重复单元，默认）'],
        ['backbone', 'backbone（几何骨架：键图最长路径）'],
        ['bonds', 'bonds（所有重原子键）'],
      ])),
      F('orient_stride', '几何骨架 stride', 'input',
        numInput('', { step: 1, min: 1, ...p('orient_stride') })),
    ]],
    ['结构有序度分析', ['order'], [
      F('g_ref', '局部结构 g_ref', 'input', numInput('留空不计入', { step: 1, ...p('g_ref') }),
        '留空不计入'),
    ]],
    ['均方位移 MSD', ['msd'], [
      F('msd_object', 'MSD 追踪对象', 'select', sel('msd_object', [
        ['molecule', '分子质心（默认，分子平动扩散）'],
        ['atom', '原子（含分子内部运动）'],
      ])),
    ]],
  ]

  const visible = GROUPS.filter(([, mods, fields]) => show(mods) && fields.length)
  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>
        分析参数
        <span className="dim" style={{ fontWeight: 400 }}>
          {Array.isArray(which)
            ? ' （随上方「分析功能」的勾选出现）'
            : ' （未按分析项筛选，显示全部）'}
        </span>
      </div>
      {visible.length ? visible.map(([title, , fields]) => (
        <div className="pblock" key={title}>
          <div className="pgroup">{title}</div>
          <ParamTable fields={fields} />
        </div>
      )) : (
        <div className="pempty">尚未勾选任何分析项（在上方「分析功能」里勾选后，这里会出现对应参数）</div>
      )}
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
        {GROUPS.map(([group, names, short]) => (
          <button key={group}
                  className={`mini ${isPreset(names) ? 'active' : ''}`}
                  title={`只勾选「${group}」这一类（${names.map(
                    (n) => titles[n] || n).join('、')}）；要全部取消就逐个点掉勾选框`}
                  onClick={() => setWhich([...names])}>
            仅{short}
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

export default { SelectionBlock, FrameBlock, ParamBlock, FunctionBlock, RunBlock }
