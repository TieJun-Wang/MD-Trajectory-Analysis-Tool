/**
 * 分析项按**模块**分组（链构象 / 空间结构 / 取向与结晶 / 动力学与输运）。
 *
 * 权威定义在后端 `mdta.pipeline.ANALYSIS_GROUPS`，通过 `/api/analyses`
 * 的 `groups` 字段下发；这里只保留一份**同内容的兜底**，用于后端暂时拿不到
 * 分组时（老版本 / 请求失败）界面依然能正常分组显示。
 *
 * 统一放在这个模块里，「分析功能」的预设按钮（SetupPanel）与「图表导航」的
 * 分层（ChartPanel）用的是同一份定义，不会出现两边对不上的情况。
 *
 * 每项是 `[完整名, 分析项, 按钮缩写]`；缩写只用于快捷按钮行（那一行必须在
 * 一行内放下，见 `webapp/_css_check.py`），完整名用于面板标题与 tooltip。
 */

/** 与后端 `mdta.pipeline.ANALYSIS_GROUPS` / `GROUP_SHORT` 保持一致 */
export const FALLBACK_GROUPS = [
  ['链构象', ['rg', 'ree', 'dihedral'], '构象'],
  ['空间结构', ['density', 'rdf', 'contact', 'interface'], '结构'],
  ['取向与结晶', ['orientation', 'order', 'boo', 'crystal'], '取向'],
  ['动力学与输运', ['msd'], '输运'],
]

/** 没被任何模块覆盖到的分析项兜底归到这里 */
export const OTHER_LABEL = '其他'

/**
 * 归一化成 `[[label, names, short], ...]`。
 * 后端给的是 `[{label, names, short}]`，本地兜底是 `[label, names, short]`，两种都收。
 */
export function normalizeGroups(fromApi) {
  if (!Array.isArray(fromApi) || !fromApi.length) return FALLBACK_GROUPS
  const out = []
  for (const g of fromApi) {
    const label = Array.isArray(g) ? g[0] : g?.label
    const names = Array.isArray(g) ? g[1] : g?.names
    const short = (Array.isArray(g) ? g[2] : g?.short) || ''
    if (!label || !Array.isArray(names) || !names.length) continue
    out.push([String(label), names.map(String),
      String(short || String(label).slice(0, 2))])
  }
  return out.length ? out : FALLBACK_GROUPS
}

/**
 * 把 `names` 里出现的分析项按模块归拢，供导航分层使用。
 *
 * - 只保留**实际已有结果**的项，所以实时运行时模块会一项项长出来；
 * - 空模块直接丢掉，不显示标题；
 * - 未在任何模块里的分析项归入「其他」，保证不会漏掉。
 */
export function groupPresent(names, groups) {
  const have = new Set(names)
  const used = new Set()
  const out = []
  for (const [label, members] of groups) {
    const present = members.filter((n) => have.has(n))
    present.forEach((n) => used.add(n))
    if (present.length) out.push([label, present])
  }
  const rest = names.filter((n) => !used.has(n))
  if (rest.length) out.push([OTHER_LABEL, rest])
  return out
}
