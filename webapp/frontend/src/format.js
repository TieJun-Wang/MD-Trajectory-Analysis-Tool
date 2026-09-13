/**
 * 数值/时间的可读化小工具（界面通用）。
 */

/** 秒数 → 人类可读。预估用，粗粒度即可：「<1 s」/「8 s」/「3 min 20 s」/「1 h 5 min」。 */
export function fmtSec(sec) {
  if (sec == null || !isFinite(Number(sec))) return '—'
  const s = Math.max(0, Number(sec))
  if (s < 1) return '<1 s'
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m} min ${Math.round(s - m * 60)} s`
  const h = Math.floor(m / 60)
  return `${h} h ${m - h * 60} min`
}
