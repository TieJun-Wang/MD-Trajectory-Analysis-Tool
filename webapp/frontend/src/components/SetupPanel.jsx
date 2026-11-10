/**
 * 3. 分析设置 —— 把「分析对象 + 参数设置 + 分析功能 + 运行」合成一块，
 *    放进同一个折叠面板里，所以三块都用紧凑排版、不带卡片外壳。
 */
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { normalizeGroups } from '../analysisGroups'
import { fmtSec } from '../format'

//: 运行日志留一小条在设置板块底部，完整日志在右侧「运行日志」标签页
/**
 * 「分析集合（Set）」板块 —— 1.0.4 起**迁到「结构选择」**（三维视图旁边），
 * 因为集合既可以手写筛选条件，也可以由三维视图里框选出来的原子直接生成。
 *
 * 只依赖 `sess`（元素/预设）、`sets`（本地定义）、`setsInfo`（后端预览）与
 * `onSetsChange`（变更回调），和原来的 SelectionBlock 用法一致。
 */
export function SetsBlock({ sess, sets = [], setsInfo = [], onSetsChange = null,
                          selectedCount = 0, onCreateFromIndices = null,
                          onHighlight = null, activeSet = '',
                          onFilterElement = null, activeElement = '',
                          onToggleFilter = null, activeKeys = null,
                          filterErr = '', filterBusy = false, filterCount = 0,
                          scopeSize = 0, onClearFilters = null, filters = [],
                          components = [], onToggleComponent = null }) {
  if (!sess) return null
  //: 三维视图里已经选中了原子 → 默认用"这批原子"建集合
  //: （1.0.4 用户要求：预设/元素 chip 只改选择、不自动建集合，所以"添加集合"
  //:   就应该默认拿当前选择来建，改名/选条件时才切到按筛选条件建）
  const [useSel, setUseSel] = useState(true)
  // ---- 分析集合（Set）：本地只存"定义"，原子数/追踪方式来自后端预览（setsInfo）
  const els = sess.elements || []
  const presets = sess.set_presets || []
  const [draft, setDraft] = useState({
    name: '', elements: [], shape: 'all', min_atoms: '', max_atoms: '', pure: false,
  })
  //: 变更集合；`highlight` = 这次新建/更新的集合名（父组件会在三维视图里高亮它）
  const emit = (next, highlight = '') => { onSetsChange?.(next, highlight) }
  const addSet = (spec) => {
    const name = String(spec.name || '').trim()
    if (!name) return
    if (sets.some((x) => x.name === name)) {
      // 同名覆盖
      emit([...sets.filter((x) => x.name !== name), { ...spec, name }], name)
      return
    }
    emit([...sets, { ...spec, name }], name)
  }
  const removeSet = (name) => emit(sets.filter((x) => x.name !== name))
  const toggleSetOn = (name) => emit(sets.map((x) => (x.name === name
    ? { ...x, on: x.on === false ? true : false } : x)))
  const toggleEl = (e) => setDraft((d) => ({ ...d,
    elements: d.elements.includes(e)
      ? d.elements.filter((x) => x !== e)
      : [...d.elements, e] }))
  const submitDraft = () => {
    const nm = draft.name.trim() || (draft.elements.join('') || '自定义')
    // ① 三维视图里选中了原子、且用户没改成"按筛选条件" → 用这批原子建集合
    if (useSel && selectedCount > 0 && onCreateFromIndices) {
      // 记下这条集合"怎么来的"：集合行只显示名字，全部说明放悬浮提示
      const from = {
        n_atoms: Number(selectedCount) || 0,
        scope: Number(scopeSize) || 0,
        filters: (filters || []).map((f) => ({
          label: f.label || f.key, n_atoms: Number(f.nAtoms || f.atoms?.size || 0),
        })),
      }
      onCreateFromIndices(nm, from)
      setDraft({ name: '', elements: [], shape: 'all', min_atoms: '', max_atoms: '', pure: false })
      return
    }
    // ② 否则按筛选条件建
    const spec = {
      name: nm,
      elements: draft.elements,
      shape: draft.shape,
      min_atoms: draft.min_atoms === '' ? 0 : Number(draft.min_atoms),
      max_atoms: draft.max_atoms === '' ? 0 : Number(draft.max_atoms),
      pure: !!draft.pure,
    }
    addSet(spec)
    setDraft({ name: '', elements: [], shape: 'all', min_atoms: '', max_atoms: '', pure: false })
  }
  const infoOf = (name) => (setsInfo || []).find((x) => x.name === `set:${name}`)
  //: **自动识别出的组分**（protein / water / ion / polymer[seg] …）——
  //: 用户要求：分析集合区也要能看到、能勾，且与「分析设置 → 分析对象」里的
  //: 胶囊是**同一份选择**（不是两条独立的路）。`set:` 开头的条目属于自定义集合，
  //: 下面单独一块，别混进来。
  const autoComps = (sess.components || [])
    .filter((c) => c && c.name && !String(c.name).startsWith('set:'))
  //: 当前生效的筛选条件键（`element:H` / `preset:链状分子`），用来把 chip 标红
  const elActive = String(activeElement || '')
  const keys = activeKeys instanceof Set ? activeKeys : new Set()

  return (
    <>
      <div className="sect-title" style={{ marginTop: 0 }}>
        分析集合（Set）
        <span className="dim" style={{ fontWeight: 400 }}>（可选：只算某一类原子或分子）</span>
      </div>
      <div className="field">
        {/* 选择现状：范围 + 条件 → 实际选中。点了 chip 这里会立刻变，避免"没反应" */}
        <div className="tree-mini dim" style={{ padding: '2px 0 4px' }}>
          当前选择 {Number(selectedCount).toLocaleString()} 原子
          {`（范围 ${Number(scopeSize).toLocaleString()}`
            + (filterCount ? ` ∩ 条件 ${filterCount} 条` : '') + '）'}
          {filterBusy ? ' · 计算中…' : ''}
          {filterCount > 0 && onClearFilters && (
            <button className="mini" style={{ marginLeft: 6 }}
                    onClick={onClearFilters}
                    title="只清掉筛选条件，保留层级树的选中范围">
              清条件
            </button>
          )}
        </div>
        {filterErr && <div className="tree-mini warn">{filterErr}</div>}
        {presets.length > 0 && (
          <div className="chips-pick mt4"
               title={'这些和结构层级里的选中是**同一套选择**：点一下把它们加进三维选择，'
                 + '再点一下取消；要留成分析用的集合，再点下面的「＋ 添加集合」'}>
            {presets.map((p) => {
              const k = `preset:${p.name}`
              const on = keys.has(k)
              return (
                <button key={p.name}
                        className={`chip-pick${on ? ' sel' : ''}`}
                        title={`${p.hint}\n预览：${p.preview}`
                          + (on ? '\n（已选中，再点一次取消）' : '')}
                        onClick={() => {
                          if (onToggleFilter) {
                            onToggleFilter({ ...p.spec, name: p.name }, k, p.name)
                            return
                          }
                          addSet({ ...p.spec, name: p.name })
                        }}>
                  {on ? '✓ ' : '＋'}{p.name}
                </button>
              )
            })}
          </div>
        )}

        {els.length > 0 && (
          <div className="chips-pick mt4"
               title={'点元素 = 在当前选择上加一条**筛选条件**（与层级树的选中同级）：'
                 + '已经选中一条链，再点 H 就是"这条链的氢"；再点一次取消这条条件。'
                 + '什么都没选时就是整个体系的该元素'}>
            {els.slice(0, 12).map((e) => (
              // 点元素 = 加/取消一条筛选条件（不建集合）。
              // "选中一条链 → 点 H" = 这条链的氢；再点 H = 取消该条件（用户实测要求）。
              <button key={e}
                      className={`chip-pick ${draft.elements.includes(e) ? 'on' : ''}`
                        + `${elActive && elActive === e ? ' sel' : ''}`}
                      title={keys.has(`element:${e}`)
                        ? `取消筛选条件：元素 ${e}`
                        : `加上筛选条件：元素 ${e}`
                          + (selectedCount > 0 ? '（与当前选择取交集）' : '')}
                      onClick={() => {
                        if (onFilterElement) {
                          onFilterElement(e)
                          return
                        }
                        // 兜底（没接筛选回调时退回老行为：一键建集合）
                        addSet({ name: e, elements: [e] })
                      }}>{e}</button>
            ))}
            {/* 「其他原子」下拉：元素多于 12 个时也能点到；作用范围与上面 chip 一致 */}
            {els.length > 12 && (
              <select value="" onChange={(e) => {
                const el = e.target.value
                if (el) {
                  if (onFilterElement) onFilterElement(el)
                  else toggleEl(el)
                }
                e.target.value = ''
              }} title={selectedCount > 0
                ? `在当前已选里取某个元素`
                : '取整个体系的某个元素'}>
                <option value="">其他原子…</option>
                {els.slice(12).map((e) => (
                  <option key={e} value={e}>{e}</option>
                ))}
              </select>
            )}
            {/* 始终提供一个"其他原子"下拉：列出全部元素，方便按名字挑 */}
            {els.length > 0 && els.length <= 12 && (
              <select value="" onChange={(e) => {
                const el = e.target.value
                if (el) {
                  if (onFilterElement) onFilterElement(el)
                  else toggleEl(el)
                }
                e.target.value = ''
              }} title={selectedCount > 0
                ? `在当前已选里取某个元素`
                : '取整个体系的某个元素'}>
                <option value="">其他原子…</option>
                {els.map((e) => (
                  <option key={`all-${e}`} value={e}>{e}</option>
                ))}
              </select>
            )}
            <select value={draft.shape}
                    onChange={(e) => setDraft({ ...draft, shape: e.target.value })}
                    title="链状 = 分子没有环；含环 = 分子里有环">
              <option value="all">形态不限</option>
              <option value="chain">链状（无环）</option>
              <option value="ring">含环</option>
            </select>
            <input type="number" min="0" placeholder="分子最少原子"
                   value={draft.min_atoms} style={{ width: 92 }}
                   onChange={(e) => setDraft({ ...draft, min_atoms: e.target.value })} />
            <input type="number" min="0" placeholder="最多原子"
                   value={draft.max_atoms} style={{ width: 78 }}
                   onChange={(e) => setDraft({ ...draft, max_atoms: e.target.value })} />
            <label className="check" style={{ fontSize: 10.5 }}
                   title="只要整分子都由所选元素组成的（例如纯碳分子）">
              <input type="checkbox" checked={draft.pure}
                     onChange={(e) => setDraft({ ...draft, pure: e.target.checked })} />
              整分子只含所选元素
            </label>
            {selectedCount > 0 && (
              <label className="check" style={{ fontSize: 10.5 }}
                     title={'勾上：用三维视图里当前高亮的 ' + selectedCount
                       + ' 个原子建集合；不勾：按上面的筛选条件建'}>
                <input type="checkbox" checked={useSel}
                       onChange={(e) => setUseSel(e.target.checked)} />
                用选中的 {selectedCount.toLocaleString()} 个原子
              </label>
            )}
            <input placeholder="集合名（留空自动）" value={draft.name}
                   style={{ width: 120 }}
                   onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
            <button className="mini" onClick={submitDraft}>＋ 添加集合</button>
          </div>
        )}

        {autoComps.length > 0 && (
          <div className="mt4">
            <div className="tree-mini dim" style={{ padding: '2px 0 3px' }}>
              自动识别的组分（勾上就参与分析，和「分析设置 → 分析对象」是同一份选择）
            </div>
            <div className="sets-list">
              {autoComps.map((c) => {
                const on = (components || []).includes(c.name)
                return (
                  <div key={c.name}
                       className={`set-row${on ? '' : ' off'}`}>
                    <label className="check"
                           title={(on ? '点此取消参与分析' : '点此参与分析')
                             + `\n${Number(c.n_atoms || 0).toLocaleString()} 原子`
                             + (c.n_residues != null ? ` / ${c.n_residues} 残基` : '')
                             + (c.n_molecules != null ? ` / ${c.n_molecules} 个分子` : '')}>
                      <input type="checkbox" checked={on}
                             onChange={() => onToggleComponent?.(c.name)} />
                      <b>{c.name}</b>
                    </label>
                    <span className="set-desc">
                      {Number(c.n_atoms || 0).toLocaleString()} 原子
                      {c.n_molecules != null ? ` · ${c.n_molecules} 分子` : ''}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {(sets || []).length > 0 && (
          <div className="sets-list mt4">
            {sets.map((x) => {
              const info = infoOf(x.name)
              const on = x.on !== false
              // 行内**只显示名字**（用户实测：后面的小字太长会把「删除」挤到换行）。
              // 原子数 / 选择来源全部放进悬浮提示：
              //   ① 这个集合有多少原子
              //   ② 层级树那边选了什么（范围）
              //   ③ 分析集合那边加了哪些筛选条件
              const fr = x.from || {}
              const lines = []
              const nA = Number(fr.n_atoms || info?.n_atoms || 0)
              lines.push(nA ? `① 原子数：${nA.toLocaleString()}` : '① 原子数：（自动预览）')
              const nScope = Number(fr.scope || 0)
              lines.push(nScope
                ? `② 层级树选择：范围 ${nScope.toLocaleString()} 原子`
                : '② 层级树选择：（没有先在层级树里选中）')
              const fl = Array.isArray(fr.filters) ? fr.filters : []
              lines.push(fl.length
                ? '③ 分析集合选择：' + fl.map((f) => `${f.label}`
                  + (f.n_atoms ? `（${Number(f.n_atoms).toLocaleString()} 原子）` : ''))
                  .join('、')
                : '③ 分析集合选择：（没有加筛选条件）')
              if (info?.label) lines.push(`· 后端识别为：${info.label}`)
              return (
                <div key={x.name}
                     className={`set-row${on ? '' : ' off'}${activeSet === x.name ? ' active' : ''}`}
                     onClick={() => onHighlight?.(x.name)}
                     title={lines.join('\n') + '\n\n点击：在三维视图里高亮这个集合的原子'}>
                  <label className="check"
                         title={on ? '点此取消参与分析' : '点此参与分析'}
                         onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" checked={on} onChange={() => toggleSetOn(x.name)} />
                    <b>set:{x.name}</b>
                  </label>
                  <button className="mini" onClick={(e) => { e.stopPropagation(); removeSet(x.name) }}>
                    删除
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>

    </>
  )
}


/**
 * 各分析项**用哪个对象**算 —— 界面上只显示一行短文字，完整解释放悬浮提示
 * （用户要求：一行解释一个功能，别在界面上铺一大段字）。
 *
 * 这份清单必须与 `mdta/analysis/*` 里真正读的对象一致（改实现时一并改这里）。
 */
const PRIMARY_USES = [
  ['Rg 回转半径', '按这个对象的全部原子算 Rg(t)；对象里有多个分子时还给出逐分子 Rg 平均'],
  ['R_ee 端到端距离', '取这个对象的两端原子（键连图里只连一根键的原子）算直线距离'],
  ['二面角', '沿这个对象的成键关系找出所有可转二面角，给 P(φ) 与 trans/gauche 比例'],
  ['取向', '把这个对象切成链段，算二阶序参数 S、P2 与链段取向角分布'],
  ['结构有序度', '这个对象每个原子的局部有序度加权成指数（按主链的近邻算）'],
  ['结晶 / 非晶识别', '用这个对象的 q6/q̄6 判固相原子，再连通成晶簇数晶区占比'],
]
const COMPONENT_USES = [
  ['密度', '沿选定方向分箱，分别统计每个组分的质量/数密度剖面'],
  ['RDF', '组分内或组分对之间的径向分布 g(r)，同分子配对可剔除'],
  ['二维 RDF', 'x–y 平面内的径向分布（厚层体系看面内结构）'],
  ['接触', 'A 组分与 B 组分之间距离 < cutoff 的原子对数（随时间）'],
  ['界面', '两组分的密度剖面求交点定位界面，再给 10–90 界面宽度'],
  ['质心距', '两组分的质心距离随时间变化（需要 ≥2 个组分）'],
  ['MSD / 扩散系数', '每个勾选的组分各算一条曲线（集合也一样）'],
]

export function SelectionBlock({ sess, primary, setPrimary, components, setComponents,
                                      sets = [], setsInfo = [], onSetsChange = null }) {
  if (!sess) return null
  const chains = sess.chains || []
  const comps = sess.components || []

  const toggleComponent = (name) => {
    setComponents(components.includes(name)
      ? components.filter((c) => c !== name)
      : [...components, name])
  }
  const nOn = comps.filter((c) => components.includes(c.name)).length
  const primaryTip = '主链（分析对象）用来算这些：\n'
    + PRIMARY_USES.map(([k, v]) => `· ${k}：${v}`).join('\n')
  const compTip = '「参与分析的对象」用来算这些（一行一个功能）：\n'
    + COMPONENT_USES.map(([k, v]) => `· ${k}：${v}`).join('\n')
    + '\n\n勾几个就按几个算：MSD / 扩散系数每个对象各一条曲线；'
    + 'RDF / 接触 / 界面 / 质心距按组分两两（或 A–B 对）计算。'

  return (
    <>
      <div className="sect-title">分析对象</div>

      <label className="field">
        <span>主链</span>
        <span className="hint dim" style={{ fontSize: 10.5 }} title={primaryTip}>
          {sets.length > 0 ? '勾了集合就用集合 · ' : ''}
          Rg / R_ee / 二面角 / 取向 / 有序度 / 结晶 按它算
        </span>
        <select
          value={primary.mode === 'custom' ? 'custom' : JSON.stringify(primary)}
          onChange={(e) => {
            const v = e.target.value
            setPrimary(v === 'custom'
              ? { mode: 'custom', query: primary.query || '' }
              : JSON.parse(v))
          }}>
          <option value={JSON.stringify({ mode: 'auto' })}>
            自动 · {sess.primary_label || '最大链'}
          </option>
          {comps.map((c) => (
            <option key={c.name} value={JSON.stringify({ mode: 'component', name: c.name })}>
              组分 {c.name}（{c.n_atoms} 原子
              {c.n_molecules != null ? ` / ${c.n_molecules} 分子` : ''}）
            </option>
          ))}
          {chains.flatMap((c) => {
            // 一个 ChainType 可能含多条同种链（PEG 体系：200 条 × 30 原子）。
            // 标"全部 N 条"就必须真的选全部（后端 all=true 会合并），
            // 否则用户按标签理解的对象和实际分析对象不是一回事。
            const base = { mode: 'chain', segid: c.segid, resname: c.resname,
                           min_atoms: c.n_atoms }
            if ((c.count || 1) > 1) {
              return [
                <option key={`${c.label}-all`}
                        value={JSON.stringify({ ...base, all: true })}>
                  全部 {c.count} 条链 {c.label}（共 {(c.n_atoms * c.count).toLocaleString()} 原子）
                </option>,
                <option key={`${c.label}-first`} value={JSON.stringify(base)}>
                  第 1 条（共 {c.count} 条，{c.n_atoms} 原子）
                </option>,
              ]
            }
            return [(
              <option key={c.label} value={JSON.stringify(base)}>
                {c.label}（{c.description}）
              </option>
            )]
          })}
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
        <span>参与分析的对象</span>
        <span className="hint dim" style={{ fontSize: 10.5 }} title={compTip}>
          密度 / RDF / 接触 / 界面 / 质心距 / MSD / 扩散系数 按勾选的组分算
        </span>
        <span className="tree-mini dim" style={{ padding: '2px 0 4px' }}
              title={compTip}>
          {`已勾 ${nOn} / ${comps.length}，点胶囊切换`
            + (nOn === 0 ? '（RDF / 密度 / 接触 这些需要至少 1 个组分）' : '')}
        </span>
        <div className="chips-pick">
          {comps.map((c) => (
            <button key={c.name}
                    className={`chip-pick ${components.includes(c.name) ? 'on' : ''}`}
                    onClick={() => toggleComponent(c.name)}
                    title={`${Number(c.n_atoms || 0).toLocaleString()} 原子`
                      + (c.n_residues != null ? ` / ${c.n_residues} 残基` : '')
                      + (c.n_molecules != null
                        ? ` / ${c.n_molecules} 个分子${c.n_molecules === 1
                          ? '（单分子）' : ''}` : '')}>
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
/** 秒 → mm:ss（>1h 显示 h:mm:ss）。进度条旁边的"已用时间"与读取面板共用同样式。 */
export function fmtElapsed(sec) {
  const t = Math.max(0, Math.floor(Number(sec) || 0))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const ss = String(t % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

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
export function ParamBlock({ params, setParams, which = null, primaryMolecules = null,
                            nAtoms = 0, primaryAtomInfo = null }) {
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
  // span 可显式覆盖：传 6 表示**独占一行**（用于让某项换行到下一行，
  // 而不是被贪心装箱挤在同一行右侧）
  const F = (key, label, kind, node, ph = '', span = null) =>
    ({ key, label, span: span || spanOf(label, kind, ph), node })

  // 布尔开关：值必须是**真布尔**（后端按 bool 收），不能用上面 p() 的字符串写法
  const chk = (key, label, title = '', { disabled = false } = {}) => (
    <label className={`pchk${disabled ? ' off' : ''}`} title={title || undefined}>
      <input type="checkbox" checked={params[key] !== false} disabled={!!disabled}
             onChange={(e) => setParams({ ...params, [key]: e.target.checked })} />
      {label}
    </label>
  )

  // 单选组：**互斥**选项（目前只有 Rg 的「分析对象」）用 radio，不用复选框 ——
  // 两个复选框会让人以为可以同时选。值是字符串，直接原样传给后端。
  const radio = (key, opts, title = '') => (
    <span className="pradio" title={title || undefined}>
      {opts.map(([v, t]) => (
        <label key={v} className={`pchk${params[key] === v ? ' on' : ''}`}>
          <input type="radio" name={key} value={v}
                 checked={params[key] === v}
                 onChange={() => setParams({ ...params, [key]: v })} />
          {t}
        </label>
      ))}
    </span>
  )

  const show = (mods) => {
    if (mods === null) return true
    if (which === null || !Array.isArray(which)) return true
    return mods.some((m) => which.includes(m))
  }

  //: 主链只有一个分子时，「逐分子统计」在数学上没有作用（整组 Rg == 该分子 Rg）
  const singleMol = primaryMolecules === 1
  //: R_ee 手动指定原子：提示里给主链原子数，并把填的序号翻成「原子名（#全局编号）」
  const reeScope = Number(nAtoms) || Number(primaryAtomInfo?.n_atoms) || 0
  const reeNames = primaryAtomInfo?.names || []
  const reeIds = primaryAtomInfo?.indices || []
  const reeLookup = (x) => {
    const xi = Number(x)
    if (!Number.isFinite(xi) || !reeNames.length) return null
    const n = reeNames.length
    const i = (xi >= 1 && xi <= n) ? xi - 1 : ((xi >= 0 && xi < n) ? xi : -1)
    if (i < 0) return null
    const id = Number.isFinite(Number(reeIds[i])) ? Number(reeIds[i]) : i
    return `${reeNames[i]}（#${id}）`
  }
  const reeParts = String(params.ree_atoms ?? '').trim().split(/[\s,，]+/).filter(Boolean)
  const reePick = reeParts.slice(0, 2).map(reeLookup).filter(Boolean)

  const GROUPS = [
    ['面内 RDF (2D) / 两组分质心距', ['rdf2d', 'comdist'], [
      F('axis', '法向轴', 'select',
        sel('axis', [[2, 'c (2)'], [1, 'b (1)'], [0, 'a (0)']])),
      F('rdf2d_rmax', '面内最大 r (Å)', 'input',
        numInput('70', { step: 5, min: 1, ...p('rdf2d_rmax') }), '70'),
      F('rdf2d_nbins', '面内 bin 数', 'input',
        numInput('140', { step: 10, min: 10, ...p('rdf2d_nbins') }), '140'),
      F('slab_lo', '叶层 z 下限 (Å)', 'input',
        numInput('留空=全盒', { step: 1, ...p('slab_lo') }), '留空=全盒'),
      F('slab_hi', '叶层 z 上限 (Å)', 'input',
        numInput('留空=盒顶', { step: 1, ...p('slab_hi') }), '留空=盒顶'),
      F('comdist_pbc', '质心距 PBC', 'check',
        chk('comdist_pbc', '用最小镜像',
          '默认按包裹坐标相减；质心可能跨盒边界时勾上'), '用最小镜像', 6),
    ]],
    ['键取向序 BOO / 晶体识别', ['boo', 'crystal'], [
      F('boo_cutoff', '邻域半径 (Å)', 'input',
        numInput('自动', { step: 0.1, min: 0, ...p('boo_cutoff') }), '自动'),
      F('boo_averaged', '用平均版 q̄_l', 'check',
        chk('boo_averaged', '是（推荐）',
          '推荐勾上：先用邻居平均再求模，区分固相/液相更稳'), '是（推荐）'),
      F('q6_solid', '固相判据 q̄6 >', 'input',
        numInput('0.5', { step: 0.05, min: 0, max: 1, ...p('q6_solid') }),
        '0.5'),
      F('min_cluster', '晶簇最小规模', 'input',
        numInput('10', { step: 1, min: 1, ...p('min_cluster') }), '10'),
    ]],
    ['回转半径 Rg', ['rg'], [
      // 「分析对象」是**互斥**的两档：选哪个就只算哪个（后端 object=...）。
      // 独占一行放在最前面，避免和下面的口径开关混在一起看漏。
      F('rg_object', '分析对象', 'radio',
        radio('rg_object', [['system', '整体体系'], ['molecule', '分子 / 链']],
          singleMol
            ? '这个对象只有 1 个分子，两档给出同一个数'
            : '整体体系：把所选对象当一个整体算（多分子时接近盒子尺寸）；'
              + '分子 / 链：每个分子各算再平均（与 Forcite 一致）'),
        '整体体系 分子 / 链', 6),
      F('mass_weighted', '质量加权 Rg', 'check',
        chk('mass_weighted', '用质量加权',
          '默认按原子质量加权（= Materials Studio Forcite 的口径）；\n'
          + '关掉则每个原子权重相同（= MDAnalysis analyze.Rg 的默认，'
          + '也是作者 mda_test.py 的算法）。\n'
          + '两种都对，但数值差 ~1%（PE 链上 H 多）：实测 20PE30 同一体系'
          + '加权 10.685 Å、不加权 10.812 Å，后者与 eth_rg_mda.csv 逐点一致。'),
        '用质量加权'),
    ]],
    ['R_ee 端到端距离', ['ree'], [
      // 下面这个开关**只影响 R_ee**：Rg 的「整组 / 逐分子」由上面的「分析对象」决定。
      F('per_molecule', 'R_ee 逐分子', 'check',
        chk('per_molecule', '按分子分别算',
          singleMol
            ? '这个对象只有 1 个分子'
            : '只管 R_ee：每个分子各算自己的端到端距离',
          { disabled: singleMol }), '按分子分别算'),
      F('ree_ends', 'R_ee 链端来源', 'select', sel('ree_ends', [
        ['bond_graph', '键图端（自动，推荐）'],
        ['residue', '逐残基首末'],
        ['custom', '手动指定两原子'],
      ])),
      F('ree_atoms', 'R_ee 指定原子', 'text',
        <input type="text"
               placeholder={reeScope > 1 ? `如 1,${reeScope}（1–${reeScope}）` : '如 0,3034（留空自动）'}
               style={{ minWidth: inputMinWidth('如 1,99999（1–99999）') }}
               value={params.ree_atoms ?? ''}
               title={`分析对象（主链）共 ${reeScope} 个原子：填两个原子序号（从 1 起，逗号或空格分隔），如 1,10`
                 + (reePick.length === 2 ? `。当前选的是 ${reePick[0]} 与 ${reePick[1]}` : '')}
               onChange={(e) => setParams({ ...params, ree_atoms: e.target.value })} />,
        '如 0,3034（留空自动）'),
    ]],
    ['接触分析', ['contact'], [
      F('cutoff', 'cutoff (Å)', 'input', numInput('', { step: 0.5, ...p('cutoff') })),
      F('contact_mode', '接触配对模式', 'select', sel('contact_mode', [
        ['inter', '分子间（默认）'],
        ['intra', '分子内'],
        ['total', '总体（含分子内）'],
      ])),
    ]],
    ['密度分布 / 界面宽度', ['density', 'interface'], [
      F('axis', '密度方向', 'select', sel('axis', [[2, 'c (2)'], [1, 'b (1)'], [0, 'a (0)']])),
      F('nbins', '密度 bin 数', 'input', numInput('', p('nbins'))),
    ]],
    ['径向分布函数 RDF', ['rdf'], [
      F('rmax', 'RDF 最大 r (Å)', 'input', numInput('', { step: 1, ...p('rmax') })),
      F('rdf_nbins', 'RDF 分箱数', 'input',
        numInput('120', { step: 1, min: 1, ...p('rdf_nbins') }), '步长 = rmax/分箱数'),
      F('rdf_mode', 'RDF 配对模式', 'select', sel('rdf_mode', [
        ['inter', '分子间（默认）'],
        ['intra', '分子内'],
        ['total', '总体（含分子内）'],
      ])),
    ]],
    ['二面角分析', ['dihedral'], [
      F('dihedral_mode', '二面角模式', 'select', sel('dihedral_mode', [
        ['auto', '自动（推荐）'],
        ['phi_psi', '蛋白质主链 φ/ψ'],
        ['chain', '沿最长链'],
        ['topology', '按拓扑定义'],
      ])),
      F('gauche_edge', 'trans/gauche 阈值 (°)', 'input',
        numInput('', { step: 5, min: 60, max: 180, ...p('gauche_edge') })),
    ]],
    ['链段取向分析', ['orientation'], [
      F('orient_mode', '取向链段定义', 'select', sel('orient_mode', [
        ['repeat', '化学重复单元（默认）'],
        ['backbone', '几何骨架'],
        ['bonds', '所有重原子键'],
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
        ['atom', '原子（默认，= Forcite / EinsteinMSD 口径）'],
        ['molecule', '分子质心（比 Forcite 大一圈）'],
      ])),
      // 「跨帧补最小镜像」不再暴露：MS 的 .trj 坐标本来就是展开的 → 不补；
      // GROMACS/NPT 的 xtc 一般折回盒内 → 默认补。由 App.jsx 按轨迹类型自动决定。
      F('msd_drift', '去漂移', 'check',
        chk('msd_drift', '减去整体漂移',
          '默认不勾（= Materials Studio Forcite 的口径，可逐点对齐）。\n'
          + '什么时候勾：NPT/NVT 里**整盒在缓慢平移**（质心漂移）时，'
          + '全体原子的共同平移会把 MSD 抬出一条额外抛物线上翘，勾上先减掉'
          + '「全体平均位移」再看单分子扩散；盒子固定、质心不动的体系不用勾。\n'
          + '勾上后 MSD 一般小 0.01%~0.2%（实测 PE50）。'), '减去整体漂移', 3),
      F('vacf_max_lag', 'VACF τ 上限 (帧)', 'input',
        numInput('留空 = 全段', { step: 1, min: 1, ...p('vacf_max_lag') }),
        '留空 = 全段'),
      F('tau_intervals', 'τ 分段数', 'input',
        numInput('5', { step: 1, min: 1, ...p('tau_intervals') }), '留空不限'),
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
        <div className="pempty">勾选上方「分析功能」后，这里出现对应参数</div>
      )}
    </>
  )
}

export function FunctionBlock({ titles = {}, order = [], which = [], setWhich,
                                groups, estimates = {}, feasible = {} }) {
  const toggle = (n) => setWhich(which.includes(n)
    ? which.filter((x) => x !== n) : [...which, n])

  //: 后端预判：`{项: {ok, level, reason}}`。level='blocked' 的项直接置灰，
  //: 并在 tooltip 里说明**为什么**跑不了（用户要求：先预判再选，别等跑完才知道）
  const verdict = (n) => {
    const v = feasible && feasible[n]
    if (!v) return { ok: true, level: 'ok', reason: '' }
    if (v.ok === false || v.level === 'blocked') {
      return { ok: false, level: 'blocked', reason: v.reason || '当前体系不满足前提' }
    }
    return { ok: true, level: v.level || 'ok', reason: v.reason || '' }
  }
  const blockedCount = order.filter((n) => !verdict(n).ok).length
  //: 当前**跑得了**的分析项。快捷按钮（全选 / 构象 / 结构 …）只许碰这些 ——
  //: 否则会出现"置灰的项被按钮勾上、然后一跑就失败"（用户实测）。
  const okNames = useMemo(() => order.filter((n) => verdict(n).ok),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [order, feasible])
  const okSet = useMemo(() => new Set(okNames), [okNames])

  // 可行性变了（换体系 / 改主链 / 改帧选择）→ 把已经勾上但**现在跑不了**的项撤掉。
  // 这样"置灰"和"已选"永远是同一套体系，不会留下偷偷跑不了的残留。
  useEffect(() => {
    const keep = which.filter((n) => okSet.has(n))
    if (keep.length !== which.length) setWhich(keep)
  }, [which, okSet, setWhich])

  // 分组来自 /api/analyses 的 groups（权威定义在 mdta.pipeline.ANALYSIS_GROUPS），
  // normalizeGroups 里带本地兜底
  const GROUPS = normalizeGroups(groups)

  // 某个模块是否整类已选 / 部分已选（按钮高亮用）
  const groupState = (names) => {
    const allIn = names.length > 0 && names.every((n) => which.includes(n))
    const some = !allIn && names.some((n) => which.includes(n))
    return { allIn, some }
  }

  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>
        分析功能
        <span className="dim" style={{ fontWeight: 400 }}>
          {' '}（已选 {which.length} / {order.length}
          {blockedCount ? `，${blockedCount} 项当前跑不了（灰）` : ''}）
        </span>
      </div>
      {GROUPS.map(([group, names]) => (
        <div key={group} className="fgroup">
          <span className="glabel">{group}</span>
          <div className="gitems">
            {names.map((n) => {
              const v = verdict(n)
              const tip = [v.reason,
                estimates?.[n] != null
                  ? `上次实测外推：预计约 ${fmtSec(estimates[n])}` : '',
                v.level === 'warn' ? v.reason : '',
              ].filter(Boolean).join('\n')
              return (
                <label key={n} className={`check${v.ok ? '' : ' disabled'}`}
                       title={tip || undefined}>
                  <input type="checkbox" checked={which.includes(n)}
                         disabled={!v.ok}
                         onChange={() => { if (v.ok) toggle(n) }} />
                  {titles[n] || n}
                  {/* 预估用时来自「开始分析」时的实测外推，直接标在分析项上，
                      省得用户去别的区块里对名字 */}
                  {estimates?.[n] != null && (
                    <span className="est-tag">≈{fmtSec(estimates[n])}</span>
                  )}
                  {!v.ok && <span className="est-tag" style={{ color: '#b45309' }}>不可用</span>}
                </label>
              )
            })}
          </div>
        </div>
      ))}
      <div className="btnrow">
        {/* 「全选 / 全不选」只针对**当前跑得了**的项：置灰的直接不勾（用户要求
            "不可用的话不能被快捷按钮勾上"）。 */}
        <button className={`mini ${okNames.length && which.length === okNames.length ? 'active' : ''}`}
                title={which.length === okNames.length && okNames.length
                  ? '取消全部勾选'
                  : `把当前可用的 ${okNames.length} 项都勾上`
                    + (blockedCount ? `（${blockedCount} 项跑不了，不会被勾上）` : '')}
                onClick={() => setWhich(which.length === okNames.length ? [] : [...okNames])}>
          {which.length === okNames.length && okNames.length ? '全不选' : '全选'}
        </button>
        {/* 各模块按钮 = **可叠加的开关**，且只加**该模块里跑得了**的项：
            点一下加进去，再点一下撤掉；不同模块可以同时选。 */}
        {GROUPS.map(([group, names, short]) => {
          const usable = names.filter((n) => okSet.has(n))
          const allIn = usable.length > 0 && usable.every((n) => which.includes(n))
          const some = !allIn && usable.some((n) => which.includes(n))
          const bad = names.length - usable.length
          return (
            <button key={group}
                    className={`mini${allIn ? ' active' : ''}${some ? ' partial' : ''}`}
                    disabled={!usable.length}
                    title={`点一下把「${group}」这一类的 ${usable.length} 项加进已选，`
                      + '再点一下取消这一类；不同模块可以同时选\n'
                      + usable.map((n) => titles[n] || n).join('、')
                      + (bad ? `\n（这一类里还有 ${bad} 项当前跑不了，不会被勾上）` : '')}
                    onClick={() => setWhich(allIn
                      ? which.filter((n) => !usable.includes(n))
                      : [...which, ...usable.filter((n) => !which.includes(n))])}>
              {short || String(group).slice(0, 2)}
            </button>
          )
        })}
      </div>
    </>
  )
}

export function RunBlock({ busy, progress, onRun, onCancel, lastLine, onOpenLog,
                          live, which = [], titles = {}, run,
                          estimateOn = true, setEstimateOn = null,
                          onClear = null }) {
  // 已用时间：本地每秒自增（后端 elapsed 只在轮询时刷新）
  const [runElapsed, setRunElapsed] = useState(0)
  useEffect(() => {
    if (!busy) return undefined
    const t0 = Date.now()
    setRunElapsed(0)
    const timer = setInterval(() => setRunElapsed((Date.now() - t0) / 1000), 500)
    return () => clearInterval(timer)
  }, [busy])

  const nDoneNow = run ? Object.keys(run.results || {}).length : 0
  const done = live?.nDone ?? nDoneNow
  const total = live?.nTotal || which.length || 0
  const pending = live?.pending || []
  const pct = Math.round((busy ? (live?.frac ?? progress) : progress) * 100)
  // 预估用时按短→长排列（这正是实际执行顺序），估不出来的项不列
  const estRows = Object.entries(live?.estimates || {})
    .filter(([, s]) => s != null)
    .sort((a, b) => a[1] - b[1])

  // 预估用时区块：**必须有东西可看**。
  // 以前它整个套在 `{busy && …}` 里、且只在 `estRows.length > 0` 时才渲染 ——
  // 于是"这次没预估出任何东西"时这块地方**一个字都不说**，看起来就像功能被删了。
  // 而"没预估"其实有明确原因（帧数 < 40 / 只勾了 1 项 / 开关被关掉），后端都会
  // 写在 estimate_note 里，这里必须把它显示出来；有名单时再列出短→长的顺序。
  const estBox = (estRows.length > 0 || live?.estimateNote) ? (
    <div className="est-box">
      <div className="hint" title={live?.estimateNote}>
        <b>预估用时</b>
        <span className="dim">
          {estRows.length
            ? `（${busy ? (live?.estimateNote || '按实测外推')
              : `上一次运行：${live?.estimateNote || '按实测外推'}`}，已按短 → 长排序）`
            : `（${live?.estimateNote || '本次没有预估，按勾选顺序直接跑'}）`}
        </span>
      </div>
      {estRows.length > 0 && (
        <ul className="est-list">
          {estRows.map(([n, s]) => {
            // 算完的项显示**实测**秒数（后端的 timings），没算完才显示 ≈ 预估
            const done = run?.timings?.[n]
            return (
              <li key={n} className={n === live?.current ? 'cur' : ''}>
                <span className="est-name">{titles[n] || n}</span>
                <span className={`est-sec${done != null ? ' done' : ''}`}>
                  {done != null ? `实测 ${fmtSec(done)}` : `≈${fmtSec(s)}`}
                </span>
              </li>
            )
          })}
          {/* 有实测但没进预估列表的项（例如关掉开关后估不出来、或分析被跳过）*/}
          {Object.entries(run?.timings || {})
            .filter(([n]) => !(n in (live?.estimates || {})))
            .map(([n, s]) => (
              <li key={`t-${n}`}>
                <span className="est-name">{titles[n] || n}</span>
                <span className="est-sec done">实测 {fmtSec(s)}</span>
              </li>
            ))}
        </ul>
      )}
    </div>
  ) : null

  return (
    <>
      <div className="sect-title" style={{ marginTop: 10 }}>
        运行
        {busy && <span className="badge-live">● 实时更新中</span>}
      </div>
      <button className="primary" onClick={onRun} disabled={busy}>
        {busy ? `分析中… ${done}/${total}`
          : (nDoneNow > 0 ? `开始分析（保留 ${nDoneNow} 项已有结果）` : '开始分析')}
      </button>
      {/* 已有结果时说明"追加"语义，并提供清空入口 */}
      {nDoneNow > 0 && (
        <div className="hint dim" style={{ fontSize: 10.5 }}>
          已有 <b>{nDoneNow}</b> 项结果：再跑只补算新勾的项，组分 / 帧 / 参数改过的
          会自动重算。整套重跑请先
          {onClear ? (
            <button className="mini" style={{ marginLeft: 4 }}
                    onClick={onClear} disabled={busy}>清空结果</button>
          ) : '清空结果'}
        </div>
      )}
      {setEstimateOn && (
        <label className="check" style={{ marginTop: 4 }}
               title={'开跑前每项先跑几帧实测，外推出各自用时，再按用时短→长执行，'
                 + '让结果尽早出现；关掉则按勾选顺序直接开跑'}>
          <input type="checkbox" checked={estimateOn} disabled={busy}
                 onChange={(e) => setEstimateOn(e.target.checked)} />
          先预估用时并按短 → 长跑
        </label>
      )}

      {busy && (
        <>
          {/* 进度条：**流光闪烁**（用户：别让人以为是卡死）；刚开始没比例时走"不确定"样式 */}
          <div className="progress">
            <div className={`bar shimmer${pct < 3 ? ' indeterminate' : ''}`}
                 style={{ width: `${Math.max(3, pct)}%` }} />
          </div>
          <div className="hint progress-meta" title={live?.message}>
            <span>{live?.message || '准备中…'}</span>
            <span className="elapsed" title="本次分析已用时间">
              已用 {fmtElapsed(runElapsed)}
            </span>
          </div>
          {/* 预估用时：后端在正式跑之前，每项先跑几帧实测再外推
              （固定开销 + 每帧代价），这里按短→长列出来 —— 排序就是执行顺序。
              「没预估出东西」时也必须显示原因，见上面 estBox 的注释。 */}
          {estBox}
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

      {/* 跑完之后这块**继续留着**：预估秒数变实测秒数，用户能核对
          "关掉预估到底快没快"，也避免刚跑完就"这个模块不见了"的错觉。 */}
      {!busy && estBox}

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
