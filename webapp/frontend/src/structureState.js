/**
 * 「结构选择」的共享状态（三维视图在中间显示区，左侧控制面板在右侧，两边共用）。
 *
 * 1.0.4 精简后只保留两件事：
 * · **结构层级**（点树节点 → 选中那批原子并在三维视图里高亮）；
 * · **分析集合（Set）**（点列表里的某个集合 → 立刻高亮它在三维视图里的原子；
 *   "添加集合"只有**一套**命名流程，选中了原子就按原子索引建，否则按筛选条件建）。
 *
 * 约定（对应 visual_select.md）
 * ---------------------------
 * · `selected` 里存的是**原始原子编号**（0 基）；抽稀只影响显示，不改编号；
 * · 只显示**拓扑/首帧坐标**，不逐帧读取（省掉读帧的时间）；
 * · 残基/链/分子的"整选"走 `api.expand`（后端用 Universe 算，与显示无关）。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'

export const STRUCT_MODES = [
  ['atom', '原子（Atom）'],
  ['residue', '残基（Residue）'],
  ['chain', '链（Chain）'],
  ['molecule', '分子（Molecule）'],
]

/** 显示抽稀上限：只影响画哪些原子，索引仍是原始编号 */
const MAX_ATOMS = 20000

/**
 * **最终选择 = 范围 ∩ 所有筛选条件**（纯函数，便于自检）。
 *
 * ⚠️ 这里踩过一个坑：`f.atoms` 是 **Set**，判空必须用 `.size` ——
 * 写成 `.length` 永远是 `undefined`（假值），于是所有条件都被当成"没生效"，
 * `selected` 直接返回 `scope`：界面上表现为"层级树选中了 + 点了 H，
 * 选中原子数一点没变"（用户实测），以及"点了预设没反应"（scope 为空 → 空选择）。
 */
export function computeSelected(scope, filters) {
  const live = (filters || []).filter(
    (f) => f && f.atoms && Number(f.atoms.size || 0) > 0)
  if (!live.length) return scope || new Set()
  let out = scope && scope.size ? new Set(scope) : null
  for (const f of live) {
    const fs = f.atoms instanceof Set ? f.atoms : new Set(f.atoms)
    if (!out) { out = new Set(fs); continue }
    for (const x of out) if (!fs.has(x)) out.delete(x)
    if (!out.size) break
  }
  return out || new Set()
}

export function useStructureState({ sess, log = null, onSetsChanged = null } = {}) {
  const sid = sess?.sid || ''
  const nAtoms = sess?.info?.n_atoms || 0

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  //: **范围**（`scope`）：结构层级/识别树/选择语句/搜索命中等"点了一下"给出的原子。
  //: **条件**（`filters`）：分析集合板块里「元素 C / 链状分子 / 含环分子 / 大分子」
  //: 这类可切换的筛选条件。
  //: 最终的三维选择 = 范围 ∩ 所有条件 —— 两者**同级、可叠加**（用户实测要求：
  //: 点了层级树的选中之后再点元素 C，不该把层级树的选择取消掉）。
  const [scope, setScope] = useState(() => new Set())
  const [filters, setFilters] = useState([])   // [{key, label, nAtoms, atoms:[...]}]
  const [hover, setHover] = useState(null)
  //: 「只显示已选」：打开后只画选中的原子（其余缩到不可见，**编号不变**）
  const [onlySelected, setOnlySelected] = useState(false)
  //: 只用于层级树点击时的"展开粒度"（原子/残基/链/分子），界面上不再给用户选
  const [mode, setMode] = useState('atom')

  const viewerRef = useRef(null)

  /** 最终选择：范围与所有条件取交集（都没有条件时就是范围本身） */
  const selected = useMemo(() => computeSelected(scope, filters), [scope, filters])

  /** 兼容旧接口：直接给一批原子当"范围" */
  const setSelected = useCallback((arg) => {
    setScope((prev) => {
      const v = typeof arg === 'function' ? arg(prev) : arg
      return new Set(v || [])
    })
  }, [])

  // 换体系：清空
  useEffect(() => {
    setData(null)
    setScope(new Set())
    setFilters([])
    setHover(null)
    setErr('')
    setOnlySelected(false)
    // 识别结果与体系一一对应：换体系必须丢掉旧的（不能拿旧原子索引去高亮新体系）
    setRecog(null)
    setRecogSel(null)
    setRecogErr('')
  }, [sid])

  /** 三维视图的本地存档键：F5 / 刷新后把**选择与视角**接回来（用户要求） */
  const saveKey = (id) => `mdta.structstate:${id}`
  const readSaved = (id) => {
    try {
      const raw = window.localStorage.getItem(saveKey(id))
      return raw ? JSON.parse(raw) : null
    } catch { return null }
  }
  const pendingViewRef = useRef(null)

  const load = useCallback(async () => {
    if (!sid) return
    setLoading(true); setErr('')
    try {
      const s = await api.structure(sid, 0, MAX_ATOMS)
      setData(s)
      // 刷新（F5 / 重新载入）后保持原样：把上次的选择与视角接回来
      const saved = readSaved(sid)
      if (saved) {
        if (Array.isArray(saved.scope) && saved.scope.length) {
          setScope(new Set(saved.scope.filter((i) => i >= 0 && i < s.n_atoms)))
        } else {
          setScope(new Set())
        }
        setFilters(Array.isArray(saved.filters)
          ? saved.filters.filter((f) => Array.isArray(f && f.atoms)) : [])
        setOnlySelected(!!saved.onlySelected)
        if (saved.mode) setMode(saved.mode)
        if (saved.view) pendingViewRef.current = saved.view
      } else {
        setScope(new Set())
        setFilters([])
      }
      setHover(null)
      log?.(`三维结构：显示 ${s.n_displayed.toLocaleString()} / `
        + `${s.n_atoms.toLocaleString()} 原子（坐标取自拓扑 / 首帧，不逐帧读取）`
        + (s.truncated
          ? `；按整残基截断到 ${s.n_residues_shown.toLocaleString()} 个残基` : ''))
    } catch (e) {
      setErr(`取三维结构失败：${e.message}`)
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [sid, log])

  // 定时存盘：选择/视角一变，最多 1.5 s 内落盘；刷新/关页面都能接回来
  useEffect(() => {
    if (!sid) return undefined
    const save = () => {
      try {
        const view = viewerRef.current && viewerRef.current.getView
          ? viewerRef.current.getView() : null
        // 把"上次看到的相机"留在内存里：切标签页 / 重建几何时也能恢复视角
        if (view) pendingViewRef.current = view
        const payload = {
          scope: [...scope], filters, onlySelected, mode, view,
        }
        if (payload.scope.length > 20000) payload.scope = []
        const txt = JSON.stringify(payload)
        if (txt.length < 400000) window.localStorage.setItem(saveKey(sid), txt)
      } catch { /* 配额满 / 隐私模式：忽略 */ }
    }
    save()
    const timer = setInterval(save, 1500)
    window.addEventListener('pagehide', save)
    return () => {
      clearInterval(timer)
      window.removeEventListener('pagehide', save)
      save()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, scope, filters, onlySelected, mode])

  /** **反选**：把"当前没被选中的原子"选上，已选中的取消（集合运算，不是猜）。 */
  const invertSelection = useCallback(() => {
    const n = Number(nAtoms) || 0
    if (!n) return
    const cur = selected || new Set()
    const next = []
    for (let i = 0; i < n; i += 1) {
      if (!cur.has(i)) next.push(i)
    }
    setScope(new Set(next))
    setFilters([])
    setSelKeys(new Set())
    keyAtomsRef.current.clear()
    setOnlySelected(false)
    setHover(null)
    log?.(`已反选：${next.length.toLocaleString()} 个原子`
      + `（原来选中 ${cur.size.toLocaleString()} 个）`)
  }, [nAtoms, selected, log])

  /** 选中一批原始原子编号（外部直接给定，例如集合载入） */
  const setIndices = useCallback((indices) => {
    setScope(new Set(indices || []))
    setFilters([])
    setSelKeys(new Set())
  }, [])

  //: **累加选择**（用户要求）：结构层级里的点击默认是"加到已有选择上"，
  //: 再点同一个节点就取消那部分。关掉这个开关就退回"点谁选谁"。
  const [additive, setAdditive] = useState(true)
  //: 当前处于选中状态的**节点键**（`recog:chain:C000001` / `element:H` …）。
  //: 只用于界面标红（用户实测：选中了哪个看不清），不参与任何计算。
  const [selKeys, setSelKeys] = useState(() => new Set())
  //: 节点键 → 它的原子（下面那个 effect 用它把红标与真实选择对齐）
  const keyAtomsRef = useRef(new Map())
  //: 筛选条件的错误与忙碌状态：以前失败只写进三维页，左侧看起来像"点了没反应"
  const [filterErr, setFilterErr] = useState('')
  const [filterBusy, setFilterBusy] = useState(false)

  /**
   * 统一的选中入口。
   *
   * `mode`：
   * · `'auto'`   —— 按「累加选择」开关走：开关开时并进去，且"整块已选"时再点一下取消；
   * · `'add'`    —— 只加不减（Ctrl/⌘ 单击用）；
   * · `'replace'`—— 直接替换（显式的选择语句结果）。
   *
   * `key` 是这次点击对应的树节点键，用来把那一行标红；不给就不标。
   */
  const applySelection = useCallback((indices, mode = 'auto', key = '') => {
    const list = (indices || []).map(Number).filter((x) => Number.isFinite(x))
    if (!list.length) return
    const replacing = mode === 'replace' || (mode === 'auto' && !additive)
    //: `toggle`（Ctrl/⌘ 单击）：整块已选 → 取消；否则加进来。
    //  以前 Ctrl 单击走的是 `add`（只加不减），于是"取消选中"永远做不到
    //  （用户实测）。
    const toggling = mode === 'toggle'
    const removing = mode === 'auto'
      ? list.every((x) => scope.has(x))
      : (toggling && list.every((x) => scope.has(x)))
    if (key) {
      // 记住这个节点键对应的原子，供"标红"与真实选择对齐（见下面的 prune 效果）
      if (removing || replacing) keyAtomsRef.current.delete(key)
      if (!removing) keyAtomsRef.current.set(key, new Set(list))
    }
    setScope((prev) => {
      const next = new Set(replacing ? [] : prev)
      if (removing) list.forEach((x) => next.delete(x))
      else list.forEach((x) => next.add(x))
      return next
    })
    setSelKeys((prev) => {
      if (replacing) return key ? new Set([key]) : new Set()
      const n = new Set(prev)
      if (!key) return n
      if (removing) n.delete(key)
      else n.add(key)
      return n
    })
    // 换"范围"时把不再适用的条件收掉？——不收：条件是叠加在范围上的筛选，
    // 换范围后它继续生效（用户要求两者同级、可叠加）。
  }, [additive, scope])

  /**
   * **让"标红"永远跟着真实选择走**（用户实测：点了取消，红标还留着）。
   *
   * 不靠"点击时顺手改一下 selKeys"这种手工同步 —— 只要某个节点键对应的原子
   * 已经不全在 `scope` 里了，就把它的红标摘掉。三层一致：红标 ⊆ 真实选择。
   * 元素/预设这类**条件键**由 `filters` 自己管，这里不碰。
   */
  useEffect(() => {
    setSelKeys((prev) => {
      if (!prev.size) return prev
      const map = keyAtomsRef.current
      let changed = false
      const next = new Set()
      for (const k of prev) {
        const isCond = k.startsWith('element:') || k.startsWith('preset:')
        const at = isCond ? null : map.get(k)
        if (at && at.size) {
          let all = true
          for (const x of at) {
            if (!scope.has(x)) { all = false; break }
          }
          if (!all) { changed = true; continue }
        }
        next.add(k)
      }
      return changed ? next : prev
    })
  }, [scope])

  /**
   * **切换一条筛选条件**（元素 / 链状分子 / 含环分子 / 大分子）。
   *
   * 用户实测要求：这些按钮不该"一点就自动建集合"，也不该"点了就取消不掉"。
   * 现在它们和层级树的选中是**同一个层级**：
   * · 没点过 → 把这条条件的原子并进选择（作为筛选条件）；
   * · 已经点过 → 取消这条条件（选择回到"范围"本身）；
   * · 多条条件同时生效时取交集（"这条链 且 氢原子"就是这个用法）。
   */
  const toggleFilter = useCallback(async (spec, key, label = '') => {
    if (!sid) return 0
    const k = String(key || '')
    const has = filters.some((f) => f.key === k)
    if (has) {
      setFilters((prev) => prev.filter((f) => f.key !== k))
      setSelKeys((prev) => {
        const n = new Set(prev)
        n.delete(k)
        return n
      })
      setFilterErr('')
      log?.(`取消筛选条件：${label || k}`)
      return 0
    }
    setFilterBusy(true)
    setFilterErr('')
    let r
    try {
      r = await api.setAtoms(sid, spec || {})
    } catch (e) {
      // 失败要说出来：以前这里只写进三维页的错误条，左侧面板上看起来像"点了没反应"
      const notFound = /not found|404/i.test(e.message || '')
      setFilterErr(notFound
        ? '筛选接口不可用（后端还是旧版本）——重启后端后再试'
        : `筛选条件不可用：${e.message}`)
      log?.(`筛选条件「${label || k}」失败：${e.message}`)
      return 0
    } finally {
      setFilterBusy(false)
    }
    const atoms = r.indices || []
    if (!atoms.length) {
      setFilterErr(`「${label || k}」在体系里没有命中任何原子`)
      log?.(`筛选条件「${label || k}」在体系里没有命中任何原子`)
      return 0
    }
    setFilterErr('')
    setFilters((prev) => [...prev, { key: k, label: label || k,
      nAtoms: atoms.length, atoms: new Set(atoms) }])
    setSelKeys((prev) => new Set([...prev, k]))
    log?.(`加上筛选条件 ${label || k}：${atoms.length.toLocaleString()} 原子`
      + (scope.size ? '（与已选范围取交集）' : ''))
    return atoms.length
  }, [sid, filters, scope, log])

  /** 清掉所有筛选条件（保留范围） */
  const clearFilters = useCallback(() => {
    setFilters([])
    setSelKeys((prev) => new Set(
      [...prev].filter((k) => !(k.startsWith('element:') || k.startsWith('preset:')))))
    setFilterErr('')
  }, [])

  /** 清掉所有"标红"标记（选择被外部改掉时用） */
  const clearSelKeys = useCallback(() => setSelKeys(new Set()), [])

  /** 点三维视图里的一个原子：按当前层级展开（结构树会设置层级） */
  const pick = useCallback(async (atomIndex, ctrl = false) => {
    if (!sid || atomIndex === null || atomIndex === undefined) return
    let idxs = [atomIndex]
    if (mode !== 'atom') {
      try {
        const r = await api.expand(sid, { level: mode, index: atomIndex })
        idxs = r.indices || [atomIndex]
      } catch (e) {
        log?.(`按 ${mode} 展开失败：${e.message}`)
      }
    }
    // Ctrl/⌘ 单击 = **切换**（整块已选就取消，否则加进来）；普通单击按
    // 「累加选择」开关走 —— 两者不冲突（累加开时普通单击=加、Ctrl=切换）。
    applySelection(idxs, ctrl ? 'toggle' : 'auto')
  }, [sid, mode, log, applySelection])

  /** 双击三维视图里的原子：选中它所在的**单体（重复单元）**。 */
  const pickMonomer = useCallback(async (atomIndex) => {
    if (!sid || atomIndex === null || atomIndex === undefined) return
    try {
      const r = await api.pickAtom(sid, atomIndex, 'monomer')
      applySelection(r.indices || [atomIndex], 'auto')
      log?.(`选中${r.label}（${(r.n_atoms || 0).toLocaleString()} 原子）`)
    } catch (e) {
      log?.(`单体选择失败：${e.message}`)
    }
  }, [sid, applySelection, log])

  /** 右击三维视图里的原子：选中所在的**一条链**；如果当前选择正好就是这条链，
   *  再右击升级为**同类全部链**。 */
  const pickChain = useCallback(async (atomIndex) => {
    if (!sid || atomIndex === null || atomIndex === undefined) return
    try {
      const r = await api.pickAtom(sid, atomIndex, 'chain')
      const idx = (r.indices || []).map(Number)
      const same = idx.length > 0 && idx.length === scope.size
        && idx.every((x) => scope.has(x))
      if (same) {
        const rt = await api.pickAtom(sid, atomIndex, 'same_type')
        applySelection(rt.indices || idx, 'replace')
        log?.(`选中${rt.label}（${(rt.n_atoms || 0).toLocaleString()} 原子）`)
      } else {
        applySelection(idx, 'auto')
        log?.(`选中${r.label}（${(r.n_atoms || 0).toLocaleString()} 原子）`)
      }
    } catch (e) {
      log?.(`链选择失败：${e.message}`)
    }
  }, [sid, applySelection, scope, log])

  /** 结构树点节点：按名字/层级取完整索引并选中 */
  const selectGroup = useCallback(async ({ level, index = null, segid = '',
                                           resname = '', component = '',
                                           key = '' }) => {
    if (!sid) return
    try {
      const r = await api.expand(sid, { level: level || 'atom', index,
        segid, resname, component })
      applySelection(r.indices, 'auto', key)
      log?.(`${additive ? '已加入' : '已选中'} ${r.label}`
        + `（${(r.n_atoms || 0).toLocaleString()} 原子）`)
    } catch (e) {
      setErr(`选中失败：${e.message}`)
    }
  }, [sid, log, applySelection, additive])

  /** 点集合列表里的某一项：把它的原子在三维视图里高亮出来（实时联动） */
  const highlightSet = useCallback(async (setName) => {
    if (!sid) return
    const name = String(setName || '').startsWith('set:') ? String(setName) : `set:${setName}`
    try {
      const r = await api.expand(sid, { level: 'atom', component: name })
      applySelection(r.indices, 'auto', `set:${name}`)
      log?.(`高亮 ${name}：${(r.n_atoms || 0).toLocaleString()} 原子`)
    } catch (e) {
      setErr(`高亮失败：${e.message}`)
    }
  }, [sid, log, applySelection])

  /**
   * **元素筛选**（兼容旧调用）：等价于 `toggleFilter({elements:[el]}, 'element:el')`。
   *
   * 以前点元素 chip = 直接对整个体系建一个 `H` 集合并把选择替换掉 —— 于是
   * "选中一条链再取它的氢"做不到，而且会把层级树的选中一起清掉（用户实测）。
   * 现在元素只是**一条可切换的筛选条件**：与范围取交集，再点一次取消。
   */
  const filterElement = useCallback((el) => {
    const e = String(el || '').trim()
    if (!e) return 0
    return toggleFilter({ elements: [e] }, `element:${e}`, `元素 ${e}`)
  }, [toggleFilter])

  /** 用"当前选中的原子"建集合（与手工筛选共用同一套命名流程）
   *  `from` = 这条集合怎么来的（范围原子数 / 筛选条件），只用于界面悬浮说明 */
  const createFromIndices = useCallback(async (name, from = null) => {
    if (!sid || !selected.size) return null
    const nm = String(name || '').trim()
    if (!nm) { setErr('集合需要一个名字'); return null }
    setErr('')
    try {
      const r = await api.createSelection(sid, nm, [...selected], from)
      log?.(`已创建集合 set:${nm}：${r.set?.n_atoms ?? selected.size} 原子（来自三维视图的选择）`)
      onSetsChanged?.(r)
      return r
    } catch (e) {
      setErr(`创建集合失败：${e.message}`)
      return null
    }
  }, [sid, selected, log, onSetsChanged])

  const resetView = useCallback(() => viewerRef.current?.resetView?.(), [])

  /** 「重置」：清空所有**选择**（选中 / 只显示已选 / 悬停 / 错误），
   * 但**不动三维视图的相机**。
   *
   * 用户实测要求：点了重置，原来看到哪儿就还在哪儿，只是把高亮清掉。
   * 想回到初始视角用三维结构页右上角那个「重置视角」，两件事分开。
   */
  const resetAll = useCallback(() => {
    setScope(new Set())
    setFilters([])
    setSelKeys(new Set())
    keyAtomsRef.current.clear()
    setFilterErr('')
    setOnlySelected(false)
    setHover(null)
    setErr('')
    // ⚠️ 这里**故意不调** viewerRef.current.resetView()：相机位置保持原样
  }, [])

  /** 结构检索：残基名+编号 / 原子名 / 原子索引 → 返回命中列表（点一项即高亮）。

  失败时**抛给调用方**，不在全局 err 上留字（以前失败后 "搜索失败：Not Found"
  会一直挂在界面上，而且成功也不会清）。
  */
  const search = useCallback(async (q, limit = 20) => {
    if (!sid || !String(q || '').trim()) return []
    const r = await api.search(sid, q, limit)
    return r.matches || []
  }, [sid])

  /** 点某个命中项：选中它的原子（三维视图立刻高亮） */
  const highlightMatch = useCallback((m) => {
    applySelection(m.indices, 'auto', `hit:${m.label}`)
    log?.(`${additive ? '已加入' : '已高亮'} ${m.label}`
      + `（${(m.n_atoms || 0).toLocaleString()} 原子）`)
  }, [log, applySelection, additive])

  /**
   * **MDAnalysis 选择语句**（`select_atoms`）→ 选中并高亮命中的原子。
   *
   * 失败时**抛给调用方**（面板里就地显示具体原因：语句哪个词写错了），
   * 不写到全局 err 上，免得错误信息一直挂着。
   *
   * 语句是**显式**的，所以这里一律替换（不累加）——想合并两次语句的结果，
   * 用 `resid 1:10 or resid 20:30` 这种写法，别靠点击累加。
   */
  const runQuery = useCallback(async (expr) => {
    const raw = String(expr || '').trim()
    if (!sid || !raw) return null
    const r = await api.query(sid, raw)
    applySelection(r.indices, 'replace', `query:${raw}`)
    log?.(`选择语句「${raw}」→ ${(r.n_atoms || 0).toLocaleString()} 原子`
      + (r.n_residues ? ` / ${r.n_residues.toLocaleString()} 残基` : '')
      + (r.truncated ? '（索引过多，只带回一部分用于高亮）' : ''))
    return r
  }, [sid, log, applySelection])

  // --------------------------------------- 分子 / 聚合物链识别（结构层级）
  /** 识别结果（后端算一次并缓存；前端只负责展示与选择） */
  const [recog, setRecog] = useState(null)
  const [recogLoading, setRecogLoading] = useState(false)
  const [recogErr, setRecogErr] = useState('')
  /** 当前从识别树里选中的对象（名字 / 类型 / 原子数 / 状态），给面板显示 */
  const [recogSel, setRecogSel] = useState(null)

  const loadRecognition = useCallback(async (refresh = false) => {
    if (!sid) return null
    setRecogLoading(true); setRecogErr('')
    try {
      const r = await api.recognition(sid, refresh)
      setRecog(r)
      const s = r.summary || {}
      log?.(`结构识别：${s.total_molecules?.toLocaleString()} 个分子 / `
        + `${s.total_polymer_chains?.toLocaleString()} 条聚合物链 / `
        + `${s.total_network_components?.toLocaleString()} 个网络组分`
        + `（键表来源 ${s.bond_source}，用时 ${s.elapsed_sec}s）`)
      return r
    } catch (e) {
      setRecogErr(`结构识别失败：${e.message}`)
      setRecog(null)
      return null
    } finally {
      setRecogLoading(false)
    }
  }, [sid, log])

  /** 点识别树上的节点（类别 / 分组 / 单个分子或链）→ 高亮它的全部原子 */
  const selectRecognition = useCallback(async (objectType, objectId, label = '') => {
    if (!sid) return null
    try {
      const r = await api.recognitionExpand(sid, { objectType, objectId })
      applySelection(r.indices, 'auto', `recog:${objectType}:${objectId}`)
      setRecogSel({ objectType, objectId, label: r.label || label,
        nAtoms: r.n_atoms || 0 })
      log?.(`${additive ? '已加入' : '已选中'} ${r.label || label}`
        + `（${(r.n_atoms || 0).toLocaleString()} 原子）`)
      return r
    } catch (e) {
      setRecogErr(`选中失败：${e.message}`)
      return null
    }
  }, [sid, log, applySelection, additive])

  return {
    sid, nAtoms,
    data, loading, err, setErr,
    selected, setSelected, setIndices, hover, setHover,
    onlySelected, setOnlySelected,
    mode, setMode,
    load, pick, pickMonomer, pickChain, selectGroup, highlightSet,
    createFromIndices, resetView, resetAll,
    invertSelection,
    search, highlightMatch, runQuery,
    additive, setAdditive, applySelection, filterElement,
    toggleFilter, clearFilters, filters,
    filterErr, filterBusy, clearFilterErr: () => setFilterErr(''),
    scopeSize: scope.size,
    selKeys, clearSelKeys,
    recog, recogLoading, recogErr, recogSel,
    loadRecognition, selectRecognition,
    viewerRef,
    //: 恢复的相机（load 从存档读出）；Viewer 重建几何时用一次后清空
    get pendingView() { return pendingViewRef.current },
  }
}
