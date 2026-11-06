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
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

export const STRUCT_MODES = [
  ['atom', '原子（Atom）'],
  ['residue', '残基（Residue）'],
  ['chain', '链（Chain）'],
  ['molecule', '分子（Molecule）'],
]

/** 显示抽稀上限：只影响画哪些原子，索引仍是原始编号 */
const MAX_ATOMS = 20000

export function useStructureState({ sess, log = null, onSetsChanged = null } = {}) {
  const sid = sess?.sid || ''
  const nAtoms = sess?.info?.n_atoms || 0

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const [selected, setSelected] = useState(() => new Set())
  const [hover, setHover] = useState(null)
  //: 「只显示已选」：打开后只画选中的原子（其余缩到不可见，**编号不变**）
  const [onlySelected, setOnlySelected] = useState(false)
  //: 只用于层级树点击时的"展开粒度"（原子/残基/链/分子），界面上不再给用户选
  const [mode, setMode] = useState('atom')

  const viewerRef = useRef(null)

  // 换体系：清空
  useEffect(() => {
    setData(null)
    setSelected(new Set())
    setHover(null)
    setErr('')
    setOnlySelected(false)
    // 识别结果与体系一一对应：换体系必须丢掉旧的（不能拿旧原子索引去高亮新体系）
    setRecog(null)
    setRecogSel(null)
    setRecogErr('')
  }, [sid])

  const load = useCallback(async () => {
    if (!sid) return
    setLoading(true); setErr('')
    try {
      const s = await api.structure(sid, 0, MAX_ATOMS)
      setData(s)
      setSelected(new Set())
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

  /** 选中一批原始原子编号 */
  const setIndices = useCallback((indices) => {
    setSelected(new Set(indices || []))
  }, [])

  /** 点三维视图里的一个原子：按当前层级展开（结构树会设置层级） */
  const pick = useCallback(async (atomIndex, additive = false) => {
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
    setSelected((prev) => {
      const next = additive ? new Set(prev) : new Set()
      const already = idxs.every((x) => prev.has(x))
      if (additive && already) idxs.forEach((x) => next.delete(x))
      else idxs.forEach((x) => next.add(x))
      return next
    })
  }, [sid, mode, log])

  /** 结构树点节点：按名字/层级取完整索引并选中 */
  const selectGroup = useCallback(async ({ level, index = null, segid = '',
                                           resname = '', component = '' }) => {
    if (!sid) return
    try {
      const r = await api.expand(sid, { level: level || 'atom', index,
        segid, resname, component })
      setSelected(new Set(r.indices || []))
      log?.(`已选中 ${r.label}（${(r.n_atoms || 0).toLocaleString()} 原子）`)
    } catch (e) {
      setErr(`选中失败：${e.message}`)
    }
  }, [sid, log])

  /** 点集合列表里的某一项：把它的原子在三维视图里高亮出来（实时联动） */
  const highlightSet = useCallback(async (setName) => {
    if (!sid) return
    const name = String(setName || '').startsWith('set:') ? String(setName) : `set:${setName}`
    try {
      const r = await api.expand(sid, { level: 'atom', component: name })
      setSelected(new Set(r.indices || []))
      log?.(`高亮 ${name}：${(r.n_atoms || 0).toLocaleString()} 原子`)
    } catch (e) {
      setErr(`高亮失败：${e.message}`)
    }
  }, [sid, log])

  /** 用"当前选中的原子"建集合（与手工筛选共用同一套命名流程） */
  const createFromIndices = useCallback(async (name) => {
    if (!sid || !selected.size) return null
    const nm = String(name || '').trim()
    if (!nm) { setErr('集合需要一个名字'); return null }
    setErr('')
    try {
      const r = await api.createSelection(sid, nm, [...selected])
      log?.(`已创建集合 set:${nm}：${r.set?.n_atoms ?? selected.size} 原子（来自三维视图的选择）`)
      onSetsChanged?.(r)
      return r
    } catch (e) {
      setErr(`创建集合失败：${e.message}`)
      return null
    }
  }, [sid, selected, log, onSetsChanged])

  const resetView = useCallback(() => viewerRef.current?.resetView?.(), [])

  /**
   * 「重置」：清空所有**选择**（选中 / 只显示已选 / 悬停 / 错误），
   * 但**不动三维视图的相机**。
   *
   * 用户实测要求：点了重置，原来看到哪儿就还在哪儿，只是把高亮清掉。
   * 想回到初始视角用三维结构页右上角那个「重置视角」，两件事分开。
   */
  const resetAll = useCallback(() => {
    setSelected(new Set())
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
    setSelected(new Set(m.indices || []))
    log?.(`已高亮 ${m.label}（${(m.n_atoms || 0).toLocaleString()} 原子）`)
  }, [log])

  /**
   * **MDAnalysis 选择语句**（`select_atoms`）→ 选中并高亮命中的原子。
   *
   * 失败时**抛给调用方**（面板里就地显示具体原因：语句哪个词写错了），
   * 不写到全局 err 上，免得错误信息一直挂着。
   */
  const runQuery = useCallback(async (expr) => {
    const raw = String(expr || '').trim()
    if (!sid || !raw) return null
    const r = await api.query(sid, raw)
    setSelected(new Set(r.indices || []))
    log?.(`选择语句「${raw}」→ ${(r.n_atoms || 0).toLocaleString()} 原子`
      + (r.n_residues ? ` / ${r.n_residues.toLocaleString()} 残基` : '')
      + (r.truncated ? '（索引过多，只带回一部分用于高亮）' : ''))
    return r
  }, [sid, log])

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
      setSelected(new Set(r.indices || []))
      setRecogSel({ objectType, objectId, label: r.label || label,
        nAtoms: r.n_atoms || 0 })
      log?.(`已选中 ${r.label || label}（${(r.n_atoms || 0).toLocaleString()} 原子）`)
      return r
    } catch (e) {
      setRecogErr(`选中失败：${e.message}`)
      return null
    }
  }, [sid, log])

  return {
    sid, nAtoms,
    data, loading, err, setErr,
    selected, setSelected, setIndices, hover, setHover,
    onlySelected, setOnlySelected,
    mode, setMode,
    load, pick, selectGroup, highlightSet, createFromIndices, resetView, resetAll,
    search, highlightMatch, runQuery,
    recog, recogLoading, recogErr, recogSel,
    loadRecognition, selectRecognition,
    viewerRef,
  }
}
