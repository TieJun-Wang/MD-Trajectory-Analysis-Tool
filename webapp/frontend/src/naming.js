/**
 * 下载文件名的**唯一实现**（图表区「保存图片」、数据表区「下载 CSV」、导出板块
 * 文件清单三处共用）。
 *
 * 用户要求：**英文、尽量短、纯 ASCII**。三条硬约束：
 *   1. 纯 ASCII —— 浏览器下载走 HTTP 头 `Content-Disposition`，中文会编码失败
 *      （实测"下载不了"）；
 *   2. 短 —— 一眼能认出来；
 *   3. 三处一致 —— 同一个面板不能导出三个不同名字。
 *
 * 面板的短英文 token 由**后端**下发给 `result.panels[i].slug`
 * （见 `mdta/naming.py::PANEL_SLUGS`，如 `msd` 的 `tau / fit / axes / power`），
 * 前端不再自己抄一份映射表；只有拿不到 slug 时才退回 ASCII 清洗。
 */

/** 兜底：把任意文本压成纯 ASCII 片段（非 ASCII 字符丢弃）。 */
export function safeAscii(text, fallback = 'result') {
  const s = [...String(text == null ? '' : text)]
    .map((ch) => (/[A-Za-z0-9._-]/.test(ch) ? ch : '_'))
    .join('')
    .replace(/^_+|_+$/g, '')
    .replace(/__+/g, '_')
  return s.slice(0, 24) || fallback
}

/** 一个面板的短英文 token（优先用后端下发的 slug）。 */
export function panelSlug(panel, index = 0) {
  const s = panel && typeof panel === 'object' ? panel.slug : panel
  if (s && /^[A-Za-z0-9._-]+$/.test(String(s))) return String(s)
  return `p${Number(index) + 1}`
}

/**
 * 单张图/表的文件名。
 *
 * @param {string} base    分析项名（`msd` / `dihedral` …）
 * @param {object} panel   面板对象（用它的 `slug`）
 * @param {number} index   面板序号
 * @param {number} nPanels 该结果的面板总数
 * @param {string} ext     `csv` / `png`
 * @returns {string} 单面板 → `msd.csv`；多面板 → `msd__axes.csv`
 */
export function panelFileName(base, panel, index, nPanels, ext = 'csv') {
  const e = String(ext || '').replace(/^\./, '')
  const dot = e ? `.${e}` : ''
  const b = safeAscii(base == null ? '' : base)
  if (Number(nPanels) <= 1) return `${b}${dot}`
  return `${b}__${panelSlug(panel, index)}${dot}`
}
