/**
 * 文件名/图标命名的**唯一实现** —— 前端三处（图表区保存图片、数据表区下载 CSV、
 * 导出页文件清单）与后端 `mdta/export.py::panel_slug` 必须逐字符一致。
 *
 * 用户要求：文件名用**面板的关键信息（标题）**，不要再出现 `panel0` / `__p1`
 * 这类编号 —— 那些名字打开文件夹根本看不出是哪张图。
 *
 * 规则（与 Python 端相同）：
 *   1. 非法字符（不是字母/数字/`.`/`_`/`-`）换成 `_`；
 *   2. 连续下划线折叠成一个，去掉首尾下划线；
 *   3. 截断到 48 字符；
 *   4. 标题为空 → `图N`（N = 面板号 + 1）。
 */

export function panelSlug(title, index = 0) {
  const raw = String(title == null ? '' : title).trim()
  if (!raw) return `图${Number(index) + 1}`
  const out = [...raw]
    .map((ch) => (/[\p{L}\p{N}._-]/u.test(ch) ? ch : '_'))
    .join('')
    .replace(/^_+|_+$/g, '')
    .replace(/__+/g, '_')
  return out.slice(0, 48) || `图${Number(index) + 1}`
}

/**
 * 单张图/表的文件名。
 *
 * @param {string} base    分析项名（如 `msd`）
 * @param {string} title   面板标题（如 `MSD 随时间间隔变化`）
 * @param {number} index   面板序号
 * @param {number} nPanels 该结果的面板总数
 * @param {string} ext     扩展名（`csv` / `png`，可带前导点）
 * @returns {string} 单面板 → `msd.csv`；多面板 → `msd__MSD_随时间间隔变化.csv`
 */
export function panelFileName(base, title, index, nPanels, ext = 'csv') {
  const e = String(ext || '').replace(/^\./, '')
  const b = String(base == null ? '' : base) || 'result'
  const dot = e ? `.${e}` : ''
  if (Number(nPanels) <= 1) return `${b}${dot}`
  return `${b}__${panelSlug(title, index)}${dot}`
}
