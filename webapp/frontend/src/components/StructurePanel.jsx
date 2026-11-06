/**
 * 3. 结构选择 —— 只保留两块：**结构层级** 与 **分析集合（Set）**。
 *
 * 用户要求（1.0.4）：
 * · 去掉「第几帧」——三维视图只显示**拓扑的坐标**，不逐帧读取；
 * · 去掉「显示键 / 原子标签」；
 * · 点「分析集合」里的某一项（例如 set:C）→ 三维视图**实时高亮**那批原子；
 * · "添加集合"只有**一套**命名流程（在 SetsBlock 里：选中了原子就按索引建，
 *   否则按筛选条件建）。
 *
 * 三维画面本身在中间显示区的「三维结构」标签页（`StructureView`），
 * 这里只放树和集合。
 */
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { SetsBlock } from './SetupPanel'

export default function StructurePanel({ sess, st, sets = [], setsInfo = [],
                                        onSetsChange = null, log = null,
                                        onNext = null }) {
  const [hier, setHier] = useState(null)
  const [openComp, setOpenComp] = useState(() => new Set())
  const [openChain, setOpenChain] = useState(() => new Set())
  const [res, setRes] = useState({})       // key → {total, rows, offset}
  const [q, setQ] = useState('')
  const [hits, setHits] = useState(null)   // 结构检索结果（点一项即高亮）
  const [searchErr, setSearchErr] = useState('')   // 只在面板本地显示，成功即清
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
    setHier(null); setRes({}); setActiveSet('')
    if (sid) loadHierarchy()
  }, [sid, loadHierarchy])

  if (!sess) {
    return <div className="placeholder"><p>先读取体系，再在这里选择结构。</p></div>
  }

  const chains = (hier?.chains || []).filter((c) => {
    if (!q.trim()) return true
    const t = q.trim().toLowerCase()
    return String(c.label).toLowerCase().includes(t)
      || String(c.resname).toLowerCase().includes(t)
      || String(c.segid).toLowerCase().includes(t)
  })

  const loadRes = async (c, offset = 0) => {
    const key = `${c.segid}|${c.resname}`
    try {
      const r = await api.hierarchy(sid, { segid: c.segid, resname: c.resname,
        offset, limit: 300 })
      setRes((prev) => ({ ...prev, [key]: { ...(r.residues || {}), offset } }))
    } catch (e) { log?.(`读取残基失败：${e.message}`) }
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

  /** 点集合 → 三维视图高亮（同时记住哪一行是高亮的） */
  const highlight = async (name) => {
    setActiveSet(name)
    await st.highlightSet(name)
  }

  const selCount = st.selected?.size || 0

  return (
    <>
      {/* ------------------------------------------------ ① 结构层级 */}
      <div className="sect-title" style={{ marginTop: 0 }}>结构层级</div>
      <div className="tree">
        <div className="tree-head">
          <span className="dim">
            {hier ? `${hier.n_atoms.toLocaleString()} 原子 / `
              + `${hier.n_residues.toLocaleString()} 残基` : '读取中…'}
          </span>
          <span className="grow" />
          <button className="mini"
                  onClick={() => {
                    st.resetAll()
                    setHits(null); setQ(''); setActiveSet(''); setSearchErr('')
                    // 用户实测：重置后集合还留在展示区 —— 这里一并清空分析集合
                    onSetsChange?.([], '')
                  }}
                  title="清空所有选择（高亮 / 只显示已选 / 搜索 / 分析集合）+ 视角复位">
            重置
          </button>
          <button className={`mini ${st.onlySelected ? 'active' : ''}`}
                  onClick={() => st.setOnlySelected(!st.onlySelected)}
                  title="只显示已选中的原子（其余不画；原子编号不变）">
            只显示已选
          </button>
        </div>
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
              <button key={`${m.label}-${i}`} className="hit"
                      title={`${m.n_atoms} 原子`}
                      onClick={() => st.highlightMatch(m)}>
                {m.label}
                <span className="count">{m.n_atoms}</span>
              </button>
            ))}
          </div>
        )}
        <div className="tree-body">
          <div className="tree-row lvl0">
            <b>体系</b>
            <span className="count">{hier ? hier.n_atoms.toLocaleString() : '—'}</span>
            <button className="mini" onClick={() => st.selectGroup({ level: 'all' })}>
              选中全部
            </button>
          </div>
          {(hier?.components || []).map((c) => (
            <div key={c.name}>
              <div className={`tree-row lvl1 ${openComp.has(c.name) ? 'open' : ''}`}>
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
                <button className="mini"
                        onClick={() => st.selectGroup({ level: 'atom', component: c.name })}>
                  选中
                </button>
              </div>
              {openComp.has(c.name) && (c.chains || chains).map((ch) => {
                const key = `${ch.segid}|${ch.resname}`
                const isOpen = openChain.has(key)
                const page = res[key]
                return (
                  <div key={key}>
                    <div className={`tree-row lvl2 ${isOpen ? 'open' : ''}`}>
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
                              ch.chem && ch.chem.note ? ch.chem.note : '',
                            ].filter(Boolean).join('\n')}>
                        {ch.label}
                        {ch.chem && ch.chem.name ? ` · ${ch.chem.name}` : ''}
                        {ch.chem && ch.chem.dp ? `（n≈${ch.chem.dp}）` : ''}
                        {ch.chem && ch.chem.alias
                          ? <span className="dim"> {ch.chem.alias}</span> : ''}
                      </span>
                      <span className="count">{ch.count}× / {ch.n_atoms.toLocaleString()}</span>
                      <button className="mini"
                              onClick={() => st.selectGroup({ level: 'atom',
                                segid: ch.segid, resname: ch.resname })}>选中</button>
                    </div>
                    {isOpen && (
                      <div className="tree-kids">
                        {page ? (
                          <>
                            <div className="dim tree-mini">
                              残基 {page.total.toLocaleString()} 个（显示 {page.rows.length}）
                            </div>
                            {page.rows.map((r) => (
                              <div key={`${r.segid}:${r.resid}`} className="tree-row lvl3">
                                <span className="nm">{r.resname}{r.resid}</span>
                                <span className="count">{r.n_atoms}</span>
                                <button className="mini" onClick={() => st.selectGroup({
                                  level: 'residue', index: r.first_index })}>选中</button>
                              </div>
                            ))}
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
          ))}
          {hier && !hier.has_bonds && (
            <div className="tree-mini warn">
              拓扑没有可靠键表：不能按「分子」归类（残基 / 链仍可用）。
            </div>
          )}
        </div>
      </div>

      {/* ------------------------------------------------ ② 分析集合 */}
      <SetsBlock sess={sess} sets={sets} setsInfo={setsInfo}
                 onSetsChange={onSetsChange}
                 selectedCount={selCount}
                 onCreateFromIndices={(nm) => st.createFromIndices(nm)}
                 onHighlight={highlight}
                 activeSet={activeSet} />

      {onNext && (
        <button className="primary mt6" onClick={onNext}>下一步：分析设置 →</button>
      )}
    </>
  )
}
