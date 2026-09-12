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

export function buildOption(result, panelIndex) {
  const panel = result.panels[panelIndex] || {
    title: result.title, xlabel: '', ylabel: '', xscale: 'linear', yscale: 'linear',
  }
  const logX = panel.xscale === 'log'
  const logY = panel.yscale === 'log'
  const curves = result.curves.filter((c) => c.panel === panelIndex)

  const legend = []
  const series = curves.map((c, ci) => {
    legend.push(c.label)
    const data = buildPoints(c, logX, logY)
    const color = PALETTE[ci % PALETTE.length]
    const common = {
      name: c.label,
      data,
      color,
      large: data.length > 3000,
      largeThreshold: 3000,
      animation: data.length < 2000,
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
      showSymbol: false,
      symbol: 'none',
      step: c.kind === 'step' ? 'middle' : false,
      lineStyle: { width: 1.9 },
      sampling: 'lttb',
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
    },
    series,
  }
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
