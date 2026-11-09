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
/** 坐标刻度：把 3.27e-9 这种"一串零"压成 3.27×10⁻⁹，短而清楚。 */
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴',
  5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' }
export function fmtSci(v) {
  const n = Number(v)
  if (!Number.isFinite(n)) return String(v)
  const a = Math.abs(n)
  if (a === 0 || (a >= 0.01 && a < 10000)) return String(Number(n.toPrecision(4)))
  const [m, e] = n.toExponential(2).split('e')
  const exp = String(Number(e)).split('').map((ch) => SUP[ch] || ch).join('')
  return `${Number(m)}×10${exp}`
}

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
/**
 * **二维热力图**：接触矩阵 / 空间热力图。
 *
 * 数据来自 `result.heatmaps`（后端单独一条通道）：`x` / `y` 是格中心，
 * `z[j][i]` 是第 j 行第 i 列的值。曲线表达不了"矩阵"，所以不硬塞进 series.data。
 */
function buildHeatmapOption(result, panel, heat) {
  const xs = (heat.x || []).map(Number)
  const ys = (heat.y || []).map(Number)
  const z = heat.z || []
  const data = []
  for (let j = 0; j < ys.length; j += 1) {
    for (let i = 0; i < xs.length; i += 1) {
      const v = Number((z[j] || [])[i])
      if (Number.isFinite(v)) data.push([i, j, v])
    }
  }
  const vmax = Number.isFinite(Number(heat.vmax)) ? Number(heat.vmax) : undefined
  return {
    title: {
      text: panel.title || heat.title || result.title,
      left: 'center', top: 6,
      textStyle: { fontSize: 15, fontWeight: 600, color: '#1f2733' },
    },
    tooltip: {
      confine: true,
      formatter: (p) => {
        const [i, j] = p.data
        return `${heat.xlabel || 'x'} ≈ ${xs[i]?.toPrecision(4)}\n`
          + `${heat.ylabel || 'y'} ≈ ${ys[j]?.toPrecision(4)}\n`
          + `值 = ${Number(p.data[2]).toPrecision(4)}`
      },
    },
    grid: { left: 18, right: 58, top: 66, bottom: 84, containLabel: true },
    xAxis: {
      type: 'category', name: heat.xlabel || panel.xlabel || '',
      nameLocation: 'middle', nameGap: 32,
      nameTextStyle: { fontSize: 12, color: '#374151' },
      data: xs.map((v) => Number(v).toPrecision(3)),
      axisLabel: { fontSize: 10, hideOverlap: true },
      splitArea: { show: false },
    },
    yAxis: {
      type: 'category', name: heat.ylabel || panel.ylabel || '',
      // 与折线图一致：纵轴名称水平放在轴顶端，避免被推到画布外
      nameLocation: 'end', nameGap: 12, nameRotate: 0,
      nameTextStyle: { fontSize: 12, color: '#374151', align: 'left',
        verticalAlign: 'bottom' },
      data: ys.map((v) => Number(v).toPrecision(3)),
      axisLabel: { fontSize: 10, hideOverlap: true },
    },
    visualMap: {
      min: 0, max: vmax || undefined, calculable: true,
      orient: 'vertical', right: 4, top: 'middle',
      itemHeight: 120, textStyle: { fontSize: 10 },
      inRange: { color: ['#eef2ff', '#93c5fd', '#3b82f6', '#1e3a8a'] },
    },
    toolbox: {
      right: 14, top: 8, itemSize: 14,
      feature: {
        saveAsImage: { title: '保存图片', pixelRatio: 2,
          name: `${result.name}_${Number(panel && panel.index) + 1 || 1}_matrix` },
        restore: { title: '重置' },
      },
    },
    series: [{
      name: heat.title || '矩阵', type: 'heatmap', data,
      emphasis: { itemStyle: { borderColor: '#111', borderWidth: 1 } },
      progressive: 4000,
    }],
  }
}

/** **极坐标图**（角度分布）：x 是角度、y 是密度。 */
function buildPolarOption(result, panel, panelIndex, visible) {
  const curves = result.curves.filter((c) => c.panel === panelIndex)
  const labels = (Array.isArray(visible) && visible.length)
    ? visible : curves.map((c) => c.label)
  const shown = curves.filter((c) => labels.includes(c.label))
  const list = shown.length ? shown : curves
  const series = list.map((c, ci) => {
    const pts = buildPoints(c, false, false)
    return {
      name: c.label,
      type: 'line',
      coordinateSystem: 'polar',
      smooth: true,
      showSymbol: pts.length <= 120,
      symbolSize: 3,
      lineStyle: { width: 1.8 },
      areaStyle: list.length === 1 ? { opacity: 0.18 } : undefined,
      color: PALETTE[ci % PALETTE.length],
      data: pts,
    }
  })
  const rmax = Math.max(1e-12, ...series.flatMap((s) => s.data.map((p) => p[1])))
  return {
    title: {
      text: panel.title || result.title,
      left: 'center', top: 6,
      textStyle: { fontSize: 15, fontWeight: 600, color: '#1f2733' },
    },
    tooltip: {
      trigger: 'item', confine: true,
      valueFormatter: (v) => (typeof v === 'number' ? v.toPrecision(6) : v),
    },
    legend: series.length > 1
      ? { top: 34, type: 'scroll', textStyle: { fontSize: 11 },
        itemWidth: 16, itemHeight: 9 }
      : { show: false },
    polar: { center: ['50%', '58%'], radius: ['12%', '72%'] },
    // 角度轴：0–360°，刻度每 30°
    angleAxis: {
      type: 'value', min: 0, max: 360, startAngle: 90,
      name: panel.xlabel || '角度 (°)',
      nameLocation: 'middle', nameGap: 26,
      nameTextStyle: { fontSize: 12, color: '#374151' },
      axisLabel: { fontSize: 10 },
      splitLine: { show: true, lineStyle: { color: '#eef1f5' } },
    },
    radiusAxis: {
      type: 'value', min: 0, max: rmax * 1.05,
      name: panel.ylabel || '',
      nameTextStyle: { fontSize: 11, color: '#374151' },
      axisLabel: { fontSize: 10 },
      splitLine: { show: true, lineStyle: { color: '#eef1f5' } },
    },
    toolbox: {
      right: 14, top: 8, itemSize: 14,
      feature: {
        saveAsImage: { title: '保存图片', pixelRatio: 2, name: `${result.name}_polar` },
        restore: { title: '重置' },
      },
    },
    series,
  }
}

export function buildOption(result, panelIndex, visible = null) {
  const panel = result.panels[panelIndex] || {
    title: result.title, xlabel: '', ylabel: '', xscale: 'linear', yscale: 'linear',
  }
  // ---- 二维热力图（接触矩阵 / 空间热力图）：走单独一条通道
  const heat = (result.heatmaps || []).find((h) => Number(h.panel) === Number(panelIndex))
  if (panel.heatmap && heat) return buildHeatmapOption(result, panel, heat)
  // ---- 极坐标（角度分布）
  if (panel.polar) return buildPolarOption(result, panel, panelIndex, visible)
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

  // ---- 默认只显示"第一个分析对象"（同面板多个 group 时）
  // group 由后端给（MSD/RDF/密度 等已带；其它分析通常是单系列，group 为空 → 全显示）
  const groupsInOrder = []
  for (const c of curves) {
    const g = String(c.group || '')
    if (g && !groupsInOrder.includes(g)) groupsInOrder.push(g)
  }
  const firstGroup = groupsInOrder.length > 1 ? groupsInOrder[0] : null
  // MSD 的「各轴分解与各向异性」：同一个对象有 5 条曲线（x/y/z/∥/⊥），
  // 一起画会互相压住 —— 默认只画第一条，其余由图例手动打开（用户要求）。
  const onlyFirstOfPanel = String(result?.name || '') === 'msd'
    && String(panel?.title || '').includes('各轴分解')
  const defaultSelected = {}
  for (const c of curves) {
    const g = String(c.group || '')
    if (onlyFirstOfPanel && !firstGroup) {
      defaultSelected[c.label] = c.label === curves[0].label
    } else {
      defaultSelected[c.label] = firstGroup ? g === firstGroup : true
    }
  }

  // ⚠️ 轴范围必须跟**实际画出来的曲线**一致：
  //    显式传了 visible（用户点图例）就用它；否则等于"默认选中的那些"
  //    （多对象时只有第一个对象）。以前这里 `visible` 为空就退回**全部曲线**，
  //    于是默认只显示第一组、坐标轴却按所有组取范围 → 第一组被压成一条平线，
  //    要等图例动一次（或刷新）才恢复（用户实测）。
  const visibleLabels = (Array.isArray(visible) && visible.length)
    ? visible
    : curves.filter((c) => defaultSelected[c.label]).map((c) => c.label)
  const shown = curves.filter((c) => visibleLabels.includes(c.label))
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
      // 点不太密时画出数据点标记，方便确认"每一帧都在图上"；
      // **只有 1 个点**时（单帧快照）把标记画大一点，否则看着像"图上什么都没有"
      showSymbol: data.length <= 1200,
      symbolSize: data.length <= 3 ? 9 : 4,
    }
    if (c.kind === 'bar') {
      return { ...common, type: 'bar', barMaxWidth: 26,
        // 「一个对象一根柱」（扩散系数 D 那种：每个对象一条曲线、各自一个图例）
        // 的点落在 value 轴上时默认宽度会算出接近 0，必须给固定像素，
        // 否则柱子细得看不见。
        ...(data.length === 1 ? { barWidth: 22 } : null),
        itemStyle: { opacity: 0.85 } }
    }
    // 自适应可视化引擎的判定优先：离散型 → 柱状图；恒定/单点 → 散点大标记
    const vz = c.viz && c.viz.chart
    if (c.kind === 'bar' || (vz === 'bar' && c.kind === 'line')) {
      return { ...common, type: 'bar', barMaxWidth: 26,
        itemStyle: { opacity: 0.85 } }
    }
    // ⚠️「恒定值 / 单点」**不要**退化成散点：用户要求点线图（带标记点的折线）。
    //    "平均 Rg" 这类水平线如果画成散点，看着就像一组孤立点而不是一条线。
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
          // 多个对象时默认只勾第一个（其它曲线在图例里可点开）；
          // MSD 的「各轴分解」同组有 5 条曲线，也默认只勾第一条
          ...(firstGroup || onlyFirstOfPanel ? { selected: defaultSelected } : null),
          itemWidth: 16,
          itemHeight: 9,
          textStyle: { fontSize: 11 },
        }
      : { show: false },
    // 绘图区：containLabel=true 保证坐标轴标签不越界
    // ⚠️ containLabel 只把**刻度数字**算进去，**轴名称**不算 —— 所以名称的空间
    //    必须由 grid 边距自己留出来。以前纵轴名称用 nameGap 64（还按名称长度再加
    //    18px）横着推开，刻度数字窄的面板里它直接被推到画布外 → 标题看不见
    //    （用户实测）。现在纵向名称改成**水平的、放在轴顶端**（见 yAxis），
    //    横向名称固定 34px、底部留够"数字 + 名称 + 缩放条"。
    grid: {
      left: 18,
      right: 28,
      top: hasLegend ? 78 : 54,
      bottom: hasLegend ? 92 : 84,
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
      // 横轴名称与刻度数字的距离：刻度字号 11 + margin 10 已经吃掉约 24px，
      // 34 留出 10px 空气。**不再按名称长度加大**（那会一路把名称推到缩放条上）。
      nameGap: 34,
      nameTextStyle: { fontSize: 12, color: '#374151' },
      axisLabel: {
        hideOverlap: true, fontSize: 11, margin: 10,
        formatter: (v) => (logX ? v : fmtSci(v)),
      },
      splitLine: { show: true, lineStyle: { color: '#eef1f5' } },
      minorTick: { show: logX },
      ...xBounds,
    },
    yAxis: {
      type: logY ? 'log' : 'value',
      name: panel.ylabel,
      // 纵轴名称**水平放在轴顶端**（像论文里的轴标题）：
      //   · 旋转 90° + 固定 nameGap 的老做法，刻度数字一窄（"0"/"5"）名称就顶到
      //     画布外被裁掉；数字一宽（"3.27×10⁻⁹"）又会压住名称 —— 两头不讨好；
      //   · 放到轴顶端后，它只跟 grid.top 的空白打交道，横竖都不会被裁。
      nameLocation: 'end',
      nameGap: 12,
      nameRotate: 0,
      nameTextStyle: { fontSize: 12, color: '#374151', align: 'left',
        verticalAlign: 'bottom' },
      axisLabel: {
        hideOverlap: true, fontSize: 11, margin: 10,
        formatter: (v) => (logY ? v : fmtSci(v)),
      },
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
