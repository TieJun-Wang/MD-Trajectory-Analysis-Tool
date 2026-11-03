/**
 * 把后端返回的一个「面板 + 曲线」转成 ECharts option。
 *
 * 设计要点
 * --------
 * - `grid.containLabel: true` —— 让 ECharts 自己按标题 / 坐标轴标签 / 刻度标签的
 *   实际占位来收缩绘图区，因此**标签永远不会被容器裁掉**（等价于后端
 *   `fit_panel_margins` 的作用）。
 * - `dataZoom`（inside + slider）+ `tooltip.axis` + `legend` —— 满足
 *   「缩放 / 平移 / hover 读数 / 图例开关」四项基础交互。
 * - 对数轴（MSD 的双对数图）会自动剔除 ≤0 的点，否则 ECharts 无法画。
 */

const PALETTE = [
  '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
  '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#4f9dd9',
]

function buildPoints(curve, logX, logY) {
  const pts = []
  const n = Math.min(curve.x.length, curve.y.length)
  for (let i = 0; i < n; i += 1) {
    const x = curve.x[i]
    const y = curve.y[i]
    if (x === null || y === null) continue
    if (logX && !(x > 0)) continue
    if (logY && !(y > 0)) continue
    pts.push([x, y])
  }
  return pts
}

/** 一组数值的有限值统计（忽略 null / NaN / ±Inf）。 */
function rangeOf(values) {
  let lo = Infinity
  let hi = -Infinity
  let nonPositive = 0
  let n = 0
  for (const v of values) {
    if (typeof v !== 'number' || !Number.isFinite(v)) continue
    n += 1
    if (v <= 0) nonPositive += 1
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!n) return null
  return { lo, hi, n, nonPositive }
}

/**
 * 单条曲线时**按这条曲线的数据**决定用对数还是线性轴。
 *
 * 用户要求：「选取单个曲线的时候应该适配这个曲线的坐标方式，让其尽可能完整
 * 展示全部时间点的数据」。两条判据：
 *  1. **有 ≤0 的值就必须用线性**（对数轴画不出 0，例如 MSD/τ=0 那一点、
 *     有负值的残差曲线会被整点丢掉 —— 这正是"数据看不全"的主因）；
 *  2. 数据跨度决定对数是否有用：跨度 ≥ 2.5 个数量级才值得上对数，
 *     否则线性更好读（小数跨度的对数轴刻度全是 10⁰/10¹ 这种噪声）。
 */
export function pickScaleByData(values, preferLog) {
  const r = rangeOf(values)
  if (!r) return preferLog ? 'log' : 'linear'
  if (r.nonPositive > 0) return 'linear'
  const lo = r.lo
  const hi = r.hi
  if (!(lo > 0) || !(hi > 0) || hi === lo) return 'linear'
  const decades = Math.log10(hi / lo)
  // **只降级、不升级**：面板声明线性就保持线性（用户要线性就别偷偷改回对数）；
  // 声明对数时，跨度小于 0.7 个数量级才退回线性（对数刻度全是 10⁰/10¹ 噪声）。
  if (!preferLog) return 'linear'
  return decades >= 0.7 ? 'log' : 'linear'
}

/** 轴范围：线性留 2% 余量、对数按比例扩一点，保证首末点都在画布内。 */
function axisBounds(r, scale) {
  if (!r) return {}
  if (scale === 'log') {
    if (!(r.lo > 0) || !(r.hi > 0)) return {}
    if (r.hi === r.lo) return { min: r.lo / 2, max: r.hi * 2 }
    const pad = Math.log10(r.hi / r.lo) * 0.03
    return { min: r.lo / 10 ** pad, max: r.hi * 10 ** pad }
  }
  if (r.hi === r.lo) {
    const d = Math.abs(r.hi) * 0.05 || 1
    return { min: r.hi - d, max: r.hi + d }
  }
  const pad = (r.hi - r.lo) * 0.02
  return { min: r.lo - pad, max: r.hi + pad }
}

/**
 * @param {object} result   一个分析结果（panels / curves / summary / notes）
 * @param {number} panelIndex 第几张图
 * @param {string[]|null} visible 图例里当前可见的曲线名；只留下 1 条时
 *        按那条曲线自适应坐标（轴型），传 null/undefined = 全部可见
 */
export function buildOption(result, panelIndex, visible = null) {
  const panel = result.panels[panelIndex] || {
    title: result.title, xlabel: '', ylabel: '', xscale: 'linear', yscale: 'linear',
  }
  const curves = result.curves.filter((c) => c.panel === panelIndex)

  // ---- 只留一条曲线时：轴型按它自己的数据选（含 0/负数 → 线性）
  const solo = (Array.isArray(visible) && visible.length === 1)
    ? curves.find((c) => c.label === visible[0])
    : null
  let xscale = panel.xscale === 'log' ? 'log' : 'linear'
  let yscale = panel.yscale === 'log' ? 'log' : 'linear'
  if (solo) {
    const raw = buildPoints(solo, false, false)
    xscale = pickScaleByData(raw.map((p) => p[0]), xscale === 'log')
    yscale = pickScaleByData(raw.map((p) => p[1]), yscale === 'log')
  }
  const logX = xscale === 'log'
  const logY = yscale === 'log'

  // ---- 轴范围 = 当前可见曲线的数据范围（小余量，见 axisBounds）
  const shown = (Array.isArray(visible) && visible.length)
    ? curves.filter((c) => visible.includes(c.label))
    : curves
  const pool = shown.length ? shown : curves
  const xsAll = []
  const ysAll = []
  for (const c of pool) {
    for (const p of buildPoints(c, false, false)) {
      xsAll.push(p[0])
      ysAll.push(p[1])
    }
  }
  const xBounds = axisBounds(
    rangeOf(logX ? xsAll.filter((v) => v > 0) : xsAll), xscale)
  const yBounds = axisBounds(
    rangeOf(logY ? ysAll.filter((v) => v > 0) : ysAll), yscale)

  const legend = []
  const series = curves.map((c, ci) => {
    legend.push(c.label)
    const data = buildPoints(c, logX, logY)
    const color = PALETTE[ci % PALETTE.length]
    // ⚠️ 用户要求"图表不抽稀"：这里**不能**开 ECharts 的降采样/大数组渲染
    //    - LTTB 降采样会按像素宽度把折线压到几百个点（放大后看不出来，
    //      但导出的图与 hover 读数都是压缩后的）—— 已移除；
    //    - large 渲染路径（>3000 点自动切换）同样可能丢点 —— 已移除；
    //    6001 点的折线 ECharts 直接画完全没问题，只是动画要关掉。
    const common = {
      name: c.label,
      data,
      color,
      animation: data.length < 2000,
      // 点不太密时画出数据点标记，方便确认"每一帧都在图上"
      showSymbol: data.length <= 1200,
      symbolSize: 4,
    }
    if (c.kind === 'bar') {
      return { ...common, type: 'bar', barMaxWidth: 26, itemStyle: { opacity: 0.85 } }
    }
    if (c.kind === 'scatter') {
      return { ...common, type: 'scatter', symbolSize: 6, itemStyle: { opacity: 0.85 } }
    }
    return {
      ...common,
      type: 'line',
      step: c.kind === 'step' ? 'middle' : false,
      lineStyle: { width: 1.9 },
    }
  })

  const hasLegend = panel.legend !== false && legend.length > 1

  return {
    title: {
      text: panel.title || result.title,
      left: 'center',
      top: 6,
      textStyle: { fontSize: 15, fontWeight: 600, color: '#1f2733' },
    },
    // hover 读数
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', label: { backgroundColor: '#4b5563' } },
      confine: true,
      valueFormatter: (v) => (typeof v === 'number' ? v.toPrecision(6) : v),
    },
    // 图例开关
    legend: hasLegend
      ? {
          top: 34,
          type: 'scroll',
          data: legend,
          itemWidth: 16,
          itemHeight: 9,
          textStyle: { fontSize: 11 },
        }
      : { show: false },
    // 绘图区：containLabel=true 保证坐标轴标签不越界
    grid: {
      left: 18,
      right: 28,
      top: hasLegend ? 84 : 56,
      bottom: 62,
      containLabel: true,
    },
    toolbox: {
      right: 14,
      top: 8,
      itemSize: 14,
      feature: {
        saveAsImage: { title: '保存图片', pixelRatio: 2, name: `${result.name}_${panelIndex + 1}` },
        dataZoom: { title: { zoom: '框选缩放', back: '还原缩放' } },
        restore: { title: '重置' },
      },
    },
    // 缩放 / 平移
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
      { type: 'slider', height: 18, bottom: 10, filterMode: 'none' },
    ],
    xAxis: {
      type: logX ? 'log' : 'value',
      name: panel.xlabel,
      nameLocation: 'middle',
      nameGap: 34,
      nameTextStyle: { fontSize: 12, color: '#374151' },
      axisLabel: { hideOverlap: true, fontSize: 11 },
      splitLine: { show: true, lineStyle: { color: '#eef1f5' } },
      minorTick: { show: logX },
      ...xBounds,
    },
    yAxis: {
      type: logY ? 'log' : 'value',
      name: panel.ylabel,
      nameLocation: 'middle',
      nameGap: 56,
      nameTextStyle: { fontSize: 12, color: '#374151' },
      axisLabel: { hideOverlap: true, fontSize: 11 },
      splitLine: { show: true, lineStyle: { color: '#eef1f5' } },
      minorTick: { show: logY },
      ...yBounds,
    },
    series,
  }
}

/** 只有一条曲线可见时，这句会告诉界面"现在是按这条曲线自适应坐标"。 */
export function soloAxisInfo(result, panelIndex, visible) {
  if (!Array.isArray(visible) || visible.length !== 1) return null
  const panel = result.panels?.[panelIndex]
  const c = result.curves?.find((x) => x.panel === panelIndex && x.label === visible[0])
  if (!c || !panel) return null
  const raw = buildPoints(c, false, false)
  const xscale = pickScaleByData(raw.map((p) => p[0]), panel.xscale === 'log')
  const yscale = pickScaleByData(raw.map((p) => p[1]), panel.yscale === 'log')
  return { label: c.label, xscale, yscale }
}

/** 图表里的统计量卡片内容 */
export function keyStats(result, limit = 8) {
  const entries = Object.entries(result.summary || {})
  return entries.slice(0, limit).map(([k, v]) => ({
    key: k,
    value: formatCell(v),
  }))
}

/**
 * 统一样本展示：数值走 formatNumber，不可测（null/NaN/空串）显示成破折号
 * 而不是字符串 "null"/"NaN"。
 */
export function formatCell(v, digits = 5) {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return '不可测'
    if (!Number.isFinite(v)) return v > 0 ? '+∞' : '−∞'
    return formatNumber(v, digits)
  }
  if (typeof v === 'boolean') return v ? '是' : '否'
  return String(v)
}

export function formatNumber(v, digits = 4) {
  if (v === null || v === undefined) return ''
  if (typeof v !== 'number') return String(v)
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e5 || a < 1e-4) return v.toExponential(3)
  return String(Number(v.toPrecision(digits)))
}
