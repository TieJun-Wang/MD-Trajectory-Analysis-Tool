/**
 * 3. 结构选择 —— **结构层级** 与 **分析集合（Set）** 两块。
 *
 * 结构层级有两个视图（同一块区域里切换，不新建页面）：
 * · **识别分类**（默认）：分子 / 聚合物链 / 网络 / 未识别 —— 数据来自后端
 *   `mdta/recognize.py`（纯拓扑键图识别，算一次并缓存），点节点即选中整组原子；
 * · **组分 / 残基**：原来按 segid+resname 分的组分 / 链 / 残基树，功能保留。
 *
 * 三维画面本身在中间显示区的「三维结构」标签页（`StructureView`），
 * 这里只放树和集合；选中一律通过 `st.selected`（原始原子索引）联动。
 */
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { SetsBlock } from './SetupPanel'

//: 识别类别的图标（只用必要的几个字符，不加装饰）
const SEC_ICON = {
  polymer: '⛓',
  small_molecule: '○',
  network: '▦',
  unknown: '⚠',
}

export default function StructurePanel({ sess, st, sets = [], setsInfo = [],
                                        onSetsChange = null, log = null,
                                        onNext = null }) {
  const [mode, setMode] = useState('recog')     // recog | classic
  const [hier, setHier] = useState(null)
  const [q, setQ] = useState('')
  const [hits, setHits] = useState(null)   // 结构检索结果（点一项即高亮）
  const [searchErr, setSearchErr] = useState('')   // 只在面板本地显示，成功即清
  //: MDAnalysis 原生选择语句（select_atoms）
  const [expr, setExpr] = useState('')
  const [queryMsg, setQueryMsg] = useState(null)   // {ok, text}
  const [queryBusy, setQueryBusy] = useState(false)
  const [activeSet, setActiveSet] = useState('')
  const sid = sess?.sid || ''

  // 打开面板时若还没有结构数据 → 自动载入（「下一步」进来就能直接看到图）
  useEffect(() => {
    if (sid && !st.data && !st.loading) st.load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid])

  const loadHierarchy = useCallback(async () => {
    if (!sid) return
    try { setHier(await api.hierarchy(sid, {})) } catch (e) { /* 树失败不影响三维视图 */ }
  }, [sid])

  useEffect(() => {
    setHier(null); setActiveSet('')
    if (sid) loadHierarchy()
  }, [sid, loadHierarchy])

  // 识别结果是后端缓存的一次性计算：进面板就取（没有才真算）
  useEffect(() => {
    if (sid && !st.recog && !st.recogLoading) st.loadRecognition()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid])

  if (!sess) {
    return <div className="placeholder"><p>先读取体系，再在这里选择结构。</p></div>
  }

  /** 结构检索：回车或点「搜」 */
  const doSearch = async () => {
    const key = q.trim()
    if (!key) { setHits(null); setSearchErr(''); return }
    setSearchErr('')
    try {
      const list = await st.search(key, 20)
      setHits(list)
      if (!list.length) {
        log?.(`没有匹配「${key}」的残基 / 原子（试试 ETH32、C1、索引 123）`)
      }
    } catch (e) {
      setHits(null)
      // 后端没重启时是 404（Not Found）——直接说清楚该干什么，而不是一直挂个错误
      const notFound = /not found|404/i.test(e.message || '')
      setSearchErr(notFound
        ? '搜索接口不可用（后端还是旧版本）——重启后端后再试'
        : `搜索失败：${e.message}`)
    }
  }

  /** MDAnalysis 选择语句：回车或点「执行」→ 命中原子直接在三维视图高亮 */
  const doQuery = async () => {
    const raw = expr.trim()
    if (!raw) { setQueryMsg(null); return }
    setQueryBusy(true)
    setQueryMsg(null)
    try {
      const r = await st.runQuery(raw)
      if (!r) { setQueryMsg(null); return }
      if (!r.n_atoms) {
        setQueryMsg({ ok: false, text: r.hint
          || '这条语句没有命中任何原子（检查 resid / resname / 原子名是否写对）' })
      } else {
        setQueryMsg({ ok: true, text: `✓ ${r.n_atoms.toLocaleString()} 原子`
          + (r.n_residues ? ` / ${r.n_residues.toLocaleString()} 残基` : '')
          + (r.truncated ? '（索引过多，只高亮一部分）' : '') })
      }
    } catch (e) {
      setQueryMsg({ ok: false, text: e.message || String(e) })
    } finally {
      setQueryBusy(false)
    }
  }

  /** 点集合 → 三维视图高亮（同时记住哪一行是高亮的） */
  const highlight = async (name) => {
    setActiveSet(name)
    await st.highlightSet(name)
  }

  const selCount = st.selected?.size || 0
  const recog = st.recog
  //: 搜索命中项也要能标红（用户实测：点了命中项看不出选中）
  const isSel = (key) => !!(st.selKeys && st.selKeys.has(key))

  return (
    <>
      {/* ------------------------------------------------ ① 结构层级 */}
      <div className="sect-title" style={{ marginTop: 0 }}>结构层级</div>
      <div className="tree">
        <div className="tree-head">
          <span className="dim">
            {recog
              ? `${(recog.summary?.total_molecules || 0).toLocaleString()} 分子 / `
                + `${(recog.summary?.total_polymer_chains || 0).toLocaleString()} 链`
              : hier ? `${hier.n_atoms.toLocaleString()} 原子 / `
                + `${hier.n_residues.toLocaleString()} 残基` : '读取中…'}
          </span>
          <span className="grow" />
          <button className="mini"
                  onClick={() => {
                    st.resetAll()
                    setHits(null); setQ(''); setActiveSet(''); setSearchErr('')
                    setExpr(''); setQueryMsg(null)
                    // 用户实测：重置后集合还留在展示区 —— 这里一并清空分析集合
                    onSetsChange?.([], '')
                  }}
                  title="清空所有选择（高亮 / 只显示已选 / 搜索 / 分析集合）；三维视图的视角保持不变">
            重置
          </button>
          <button className={`mini ${st.additive ? 'active' : ''}`}
                  onClick={() => st.setAdditive(!st.additive)}
                  title={'累加选择：开时点击结构层级会把原子**加到**已有选择上，'
                    + '再点同一项则取消那部分（想同时选两条链、或"某条链 + 它的氢"就靠它）；'
                    + '关掉就退回"点谁选谁"'}>
            累加
          </button>
          <button className={`mini ${st.onlySelected ? 'active' : ''}`}
                  onClick={() => st.setOnlySelected(!st.onlySelected)}
                  title="只显示已选中的原子（其余不画；原子编号不变）">
            只显示已选
          </button>
        </div>

        {/* 两个视图在同一块区域里切换：识别分类 / 组分残基 */}
        <div className="tree-modes">
          <button className={`mini ${mode === 'recog' ? 'active' : ''}`}
                  onClick={() => setMode('recog')}
                  title="按拓扑键图识别出的分子 / 聚合物链 / 网络分类">识别分类</button>
          <button className={`mini ${mode === 'classic' ? 'active' : ''}`}
                  onClick={() => setMode('classic')}
                  title="按 segid + 残基组织的组分 / 链 / 残基树">组分残基</button>
          <span className="grow" />
          {mode === 'recog' && (
            <button className="mini" disabled={st.recogLoading}
                    onClick={() => st.loadRecognition(true)}
                    title="重新做一次识别（拓扑变了才需要）">
              {st.recogLoading ? '识别中…' : '重新识别'}
            </button>
          )}
        </div>

        {/* 检索 / 选择语句 —— 两个视图下都能用 */}
        <div className="tree-search-row">
          <input className="tree-search"
                 placeholder="搜残基/原子：ETH32、C1、123"
                 value={q}
                 onChange={(e) => setQ(e.target.value)}
                 onKeyDown={(e) => { if (e.key === 'Enter') doSearch() }} />
          <button className="mini" onClick={doSearch} title="按残基名+编号 / 原子名 / 原子索引检索">
            搜
          </button>
          {hits && (
            <button className="mini" onClick={() => { setHits(null); setQ('') }}>
              清除
            </button>
          )}
        </div>
        {searchErr && <div className="tree-mini warn">{searchErr}</div>}
        {hits && (
          <div className="tree-hits">
            {hits.length === 0 && (
              <div className="dim tree-mini">没有匹配项（试试 ETH32、C1、索引 123）</div>
            )}
            {hits.map((m, i) => (
              <button key={`${m.label}-${i}`}
                      className={`hit${isSel(`hit:${m.label}`) ? ' sel' : ''}`}
                      title={`${m.n_atoms} 原子`
                        + (isSel(`hit:${m.label}`) ? '（已选中，再点一次取消）' : '')}
                      onClick={() => st.highlightMatch(m)}>
                {m.label}
                <span className="count">{m.n_atoms}</span>
              </button>
            ))}
          </div>
        )}

        {/* ---- MDAnalysis 选择语句（select_atoms）：命中即高亮，可再建成集合 */}
        <div className="tree-search-row">
          <input className="tree-search sel-atoms"
                 placeholder="选择语句：resid 1:10 and name CA"
                 value={expr}
                 spellCheck={false}
                 title={['MDAnalysis select_atoms 语法，例：',
                   'resname ETH and name C1',
                   'resid 1:10 and name CA',
                   'prop mass > 12',
                   'around 5.0 resname GRA',
                   'same residue as (name C1 and resid 1)',
                   '命中后可用下面「分析集合」的“按已选原子”建成集合'].join('\n')}
                 onChange={(e) => setExpr(e.target.value)}
                 onKeyDown={(e) => { if (e.key === 'Enter') doQuery() }} />
          <button className="mini" onClick={doQuery} disabled={queryBusy}
                  title="按 MDAnalysis select_atoms 语法选取原子（回车同效）">
            {queryBusy ? '…' : '执行'}
          </button>
          {(queryMsg || expr) && (
            <button className="mini" onClick={() => { setExpr(''); setQueryMsg(null) }}>
              清除
            </button>
          )}
        </div>
        {queryMsg && (
          <div className={`tree-mini ${queryMsg.ok ? 'dim' : 'warn'}`}>{queryMsg.text}</div>
        )}

        {mode === 'recog'
          ? <RecogTree st={st} recog={recog} />
          : <ClassicTree st={st} hier={hier} />}
      </div>

      {/* ------------------------------------------------ ② 分析集合 */}
      <SetsBlock sess={sess} sets={sets} setsInfo={setsInfo}
                 onSetsChange={onSetsChange}
                 selectedCount={selCount}
                 onCreateFromIndices={(nm) => st.createFromIndices(nm)}
                 onHighlight={highlight}
                 activeSet={activeSet}
                 onFilterElement={(el) => st.filterElement(el)}
                 onToggleFilter={(spec, key, label) => st.toggleFilter(spec, key, label)}
                 activeKeys={st.selKeys}
                 filterErr={st.filterErr || ''}
                 filterBusy={!!st.filterBusy}
                 filterCount={(st.filters || []).length}
                 scopeSize={st.scopeSize || 0}
                 onClearFilters={() => st.clearFilters?.()}
                 activeElement={(() => {
                   const k = [...(st.selKeys || [])].find((x) => x.startsWith('element:'))
                   return k ? k.slice('element:'.length) : ''
                 })()} />

      {onNext && (
        <button className="primary mt6" onClick={onNext}>下一步：分析设置 →</button>
      )}
    </>
  )
}

/* ============================================================ 识别分类树 */
/**
 * System → 类别（聚合物 / 小分子 / 网络 / 未识别）→ 分组 → 单个分子或链。
 *
 * 全部统计数字来自后端识别结果，前端**不做任何推算**；对象层按需分页加载，
 * 上万分子的体系不会因为展开一次就把整棵树塞进 DOM。
 */
function RecogTree({ st, recog }) {
  const [openSec, setOpenSec] = useState(() => new Set())
  const [openGrp, setOpenGrp] = useState(() => new Set())
  const [objs, setObjs] = useState({})          // groupIndex → {total, rows, offset}
  const [busy, setBusy] = useState({})
  const sid = st.sid

  if (st.recogErr) {
    return <div className="tree-body"><div className="tree-mini warn">{st.recogErr}</div></div>
  }
  if (!recog) {
    return (
      <div className="tree-body">
        <div className="dim tree-mini">
          {st.recogLoading ? '正在按拓扑识别分子与聚合物链…' : '还没有识别结果'}
        </div>
      </div>
    )
  }

  const toggle = (setter, key) => setter((prev) => {
    const n = new Set(prev)
    if (n.has(key)) n.delete(key); else n.add(key)
    return n
  })

  const loadObjs = async (g, offset = 0) => {
    setBusy((p) => ({ ...p, [g]: true }))
    try {
      const r = await api.recognitionObjects(sid, { group: g, offset, limit: 60 })
      setObjs((prev) => ({ ...prev, [g]: { ...r, offset } }))
    } catch (e) {
      st.setErr?.(`读取对象失败：${e.message}`)
    } finally {
      setBusy((p) => ({ ...p, [g]: false }))
    }
  }

  const sections = recog.sections || []
  const summary = recog.summary || {}
  //: 哪些节点当前是选中的（用户实测：选中后没有标红，根本看不出选中了没）
  const isSel = (key) => !!(st.selKeys && st.selKeys.has(key))
  const selCls = (key) => (isSel(key) ? ' sel' : '')

  return (
    <div className="tree-body">
      <div className="tree-row lvl0">
        <b>体系</b>
        <span className="count">{(summary.total_atoms || 0).toLocaleString()}</span>
        <button className={`mini${selCls('recog:all:')}`}
                onClick={() => st.selectRecognition('all', '', '全部原子')}>
          {isSel('recog:all:') ? '已选' : '选中全部'}
        </button>
      </div>

      {sections.length === 0 && (
        <div className="dim tree-mini">没有识别出任何组分</div>
      )}

      {sections.map((sec) => {
        const isOpen = openSec.has(sec.key)
        return (
          <div key={sec.key}>
            <div className={`tree-row lvl1 ${isOpen ? 'open' : ''}`
              + selCls(`recog:section:${sec.key}`)}>
              <button className="caret" onClick={() => toggle(setOpenSec, sec.key)}>
                {isOpen ? '▾' : '▸'}
              </button>
              <span className="nm">
                <span className="badge">{SEC_ICON[sec.key] || '·'}</span>
                {sec.label}
              </span>
              <span className="count">{sec.n_objects.toLocaleString()} / {sec.n_groups}</span>
              <button className={`mini${selCls(`recog:section:${sec.key}`)}`}
                      onClick={() => st.selectRecognition('section', sec.key, sec.label)}>
                {isSel(`recog:section:${sec.key}`) ? '已选' : '选中'}
              </button>
            </div>
            {isOpen && sec.groups.map((g) => {
              const gi = g.group_index
              const gOpen = openGrp.has(gi)
              const page = objs[gi]
              const chem = g.chem || {}
              const gKey = `recog:group:${g.group_id}`
              const tip = [
                `类型 ${g.type_label}`,
                g.chain_type ? `链类型 ${g.chain_type_label}` : '',
                `对象 ${g.n_objects} 个 / 原子 ${g.n_atoms}`,
                chem.formula ? `分子式 ${chem.formula}` : '',
                chem.category ? `类别 ${chem.category}` : '',
                chem.note ? chem.note : '',
              ].filter(Boolean).join('\n')
              return (
                <div key={gi}>
                  <div className={`tree-row lvl2 ${gOpen ? 'open' : ''}` + selCls(gKey)}>
                    <button className="caret" onClick={() => {
                      toggle(setOpenGrp, gi)
                      if (!page) loadObjs(gi)
                    }}>{gOpen ? '▾' : '▸'}</button>
                    <span className="nm" title={tip}>
                      {g.label}
                      {g.chain_type ? ` · ${g.chain_type_label}` : ''}
                      {chem.dp ? `（n≈${chem.dp}）` : ''}
                      {g.status !== 'recognized'
                        ? <span className="dim"> · {g.status_label}</span> : ''}
                    </span>
                    <span className="count">
                      {g.n_objects.toLocaleString()}× / {g.n_atoms.toLocaleString()}
                    </span>
                    <button className={`mini${selCls(gKey)}`}
                            onClick={() => st.selectRecognition('group', g.group_id, g.label)}>
                      {isSel(gKey) ? '已选' : '选中'}
                    </button>
                  </div>

                  {gOpen && (
                    <div className="tree-kids">
                      {busy[gi] && !page && <div className="dim tree-mini">读取对象…</div>}
                      {page && (
                        <>
                          <div className="dim tree-mini">
                            {page.total.toLocaleString()} 个对象
                            （显示 {page.rows.length}）
                          </div>
                          {page.rows.map((o) => {
                            const oKey = `recog:${o.object_type}:${o.object_id}`
                            return (
                              <div key={o.molecule_id}
                                   className={`tree-row lvl3${selCls(oKey)}`}>
                                <span className="nm"
                                      title={[
                                        `${o.type_label} · ${o.chain_type_label}`,
                                        `原子 ${o.n_atoms}，键 ${o.n_bonds}，`
                                          + `重原子 ${o.n_heavy}，环秩 ${o.n_rings}`,
                                        o.status_label,
                                      ].join('\n')}>
                                  {o.name}
                                  {o.status !== 'recognized'
                                    ? <span className="dim"> · {o.status_label}</span> : ''}
                                </span>
                                <span className="count">{o.n_atoms}</span>
                                <button className={`mini${selCls(oKey)}`}
                                        onClick={() => st.selectRecognition(
                                          o.object_type, o.object_id,
                                          `${o.name} · ${o.chain_type_label || o.type_label}`)}>
                                  {isSel(oKey) ? '已选' : '选中'}
                                </button>
                              </div>
                            )
                          })}
                          {page.total > (page.offset || 0) + page.rows.length && (
                            <button className="mini" disabled={busy[gi]}
                                    onClick={() => loadObjs(
                                      gi, (page.offset || 0) + page.rows.length)}>
                              再看 {Math.min(60, page.total - (page.offset || 0)
                                - page.rows.length)} 个
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )
      })}

      {/* 当前选中的对象 + 识别警告（警告来自后端，不猜） */}
      {st.recogSel && (
        <div className="tree-mini dim">
          已选 {st.recogSel.label} · {(st.recogSel.nAtoms || 0).toLocaleString()} 原子
        </div>
      )}
      {(recog.warnings || []).map((w, i) => (
        <div key={`w${i}`} className="tree-mini warn">{w}</div>
      ))}
    </div>
  )
}

/* ============================================================ 组分 / 残基树 */
function ClassicTree({ st, hier }) {
  const [openComp, setOpenComp] = useState(() => new Set())
  const [openChain, setOpenChain] = useState(() => new Set())
  const [res, setRes] = useState({})       // key → {total, rows, offset}
  const [resErr, setResErr] = useState('')  // 残基分页失败的原因（以前被吞掉，看起来像"点了没反应"）
  const sid = st.sid

  const chains = (hier?.chains || [])
  const loadRes = async (c, offset = 0) => {
    const key = `${c.segid}|${c.resname}`
    setResErr('')
    try {
      const r = await api.hierarchy(sid, { segid: c.segid, resname: c.resname,
        offset, limit: 300 })
      setRes((prev) => ({ ...prev, [key]: { ...(r.residues || {}), offset } }))
    } catch (e) {
      setResErr(`读取残基失败：${e.message}`)
    }
  }

  const isSel = (key) => !!(st.selKeys && st.selKeys.has(key))
  const selCls = (key) => (isSel(key) ? ' sel' : '')

  return (
    <div className="tree-body">
      {resErr && <div className="tree-mini warn">{resErr}</div>}
      <div className={`tree-row lvl0${selCls('classic:all')}`}>
        <b>体系</b>
        <span className="count">{hier ? hier.n_atoms.toLocaleString() : '—'}</span>
        <button className={`mini${selCls('classic:all')}`}
                onClick={() => st.selectGroup({ level: 'all', key: 'classic:all' })}>
          {isSel('classic:all') ? '已选' : '选中全部'}
        </button>
      </div>
      {(hier?.components || []).map((c) => {
        const cKey = `classic:comp:${c.name}`
        return (
          <div key={c.name}>
            <div className={`tree-row lvl1 ${openComp.has(c.name) ? 'open' : ''}`
              + selCls(cKey)}>
              <button className="caret" onClick={() => setOpenComp((p) => {
                const n = new Set(p)
                if (n.has(c.name)) n.delete(c.name); else n.add(c.name)
                return n
              })}>{openComp.has(c.name) ? '▾' : '▸'}</button>
              <span className="nm"
                    title={(c.residue_mix || []).map((m) => `${m.resname}×${m.n}`)
                      .join('、') + (c.n_residue_kinds
                        ? `（共 ${c.n_residue_kinds} 种残基）` : '')}>
                {c.name}
                {c.residue_mix && c.residue_mix.length
                  ? ` · ${c.residue_mix.slice(0, 3).map((m) => m.resname).join('/')}`
                  : ''}
                {c.n_residue_kinds > 3 ? ` 等 ${c.n_residue_kinds} 种` : ''}
              </span>
              <span className="count">{c.n_atoms.toLocaleString()}</span>
              <button className={`mini${selCls(cKey)}`}
                      onClick={() => st.selectGroup({ level: 'atom',
                        component: c.name, key: cKey })}>
                {isSel(cKey) ? '已选' : '选中'}
              </button>
            </div>
            {openComp.has(c.name) && (c.chains || chains).map((ch) => {
              const key = `${ch.segid}|${ch.resname}`
              const chKey = `classic:chain:${key}`
              const isOpen = openChain.has(key)
              const page = res[key]
              return (
                <div key={key}>
                  <div className={`tree-row lvl2 ${isOpen ? 'open' : ''}` + selCls(chKey)}>
                  <button className="caret" onClick={() => {
                    setOpenChain((p) => {
                      const n = new Set(p)
                      if (n.has(key)) n.delete(key); else n.add(key)
                      return n
                    })
                    if (!page) loadRes(ch)
                  }}>{isOpen ? '▾' : '▸'}</button>
                  <span className="nm"
                        title={[
                          `segid ${ch.segid} / resname ${ch.resname}`,
                          ch.chem && ch.chem.formula
                            ? `分子式 ${ch.chem.formula}` : '',
                          ch.chem && ch.chem.category
                            ? `类别 ${ch.chem.category}` : '',
                          ch.chem && ch.chem.alias
                            ? `俗称 ${ch.chem.alias}` : '',
                          ch.chem && ch.chem.note ? ch.chem.note : '',
                        ].filter(Boolean).join('\n')}>
                    {/* 行内只留短标签；名字/俗称全文在 tooltip 里（否则长名字会把
                        「选中」按钮挤到第二行 —— 用户实测） */}
                    {ch.label}
                    {ch.chem && ch.chem.name
                      ? ` · ${String(ch.chem.name).replace(/（.*$/, '').slice(0, 12)}` : ''}
                    {ch.chem && ch.chem.dp ? `（n≈${ch.chem.dp}）` : ''}
                  </span>
                  <span className="count">{ch.count}× / {ch.n_atoms.toLocaleString()}</span>
                  <button className={`mini${selCls(chKey)}`}
                          onClick={() => st.selectGroup({ level: 'atom',
                            segid: ch.segid, resname: ch.resname,
                            key: chKey })}>{isSel(chKey) ? '已选' : '选中'}</button>
                </div>
                {isOpen && (
                  <div className="tree-kids">
                    {page ? (
                      <>
                        <div className="dim tree-mini">
                          残基 {page.total.toLocaleString()} 个（显示 {page.rows.length}）
                        </div>
                        {page.rows.map((r) => {
                          const rKey = `classic:res:${r.segid}:${r.resid}`
                          return (
                            <div key={`${r.segid}:${r.resid}`}
                                 className={`tree-row lvl3${selCls(rKey)}`}>
                              <span className="nm">{r.resname}{r.resid}</span>
                              <span className="count">{r.n_atoms}</span>
                              <button className={`mini${selCls(rKey)}`}
                                      onClick={() => st.selectGroup({
                                        level: 'residue', index: r.first_index,
                                        key: rKey })}>
                                {isSel(rKey) ? '已选' : '选中'}
                              </button>
                            </div>
                          )
                        })}
                        {page.total > page.rows.length && (
                          <button className="mini" onClick={() => loadRes(
                            ch, page.offset + page.rows.length)}>
                            再看 {Math.min(300,
                              page.total - page.offset - page.rows.length)} 个
                          </button>
                        )}
                      </>
                    ) : <div className="dim tree-mini">读取残基…</div>}
                  </div>
                )}
              </div>
            )
          })}
        </div>
        )
      })}
      {hier && !hier.has_bonds && (
        <div className="tree-mini warn">
          拓扑没有可靠键表：不能按「分子」归类（残基 / 链仍可用）。
        </div>
      )}
    </div>
  )
}
