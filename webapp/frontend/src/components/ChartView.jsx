/**
 * ECharts 单图组件。
 *
 * 一次只显示一张图（由父组件通过 title/结果决定显示哪张），
 * 图表自适应容器大小；缩放 / 平移 / hover / 图例开关由 ECharts 提供。
 *
 * 图例只剩**一条曲线**时（用户点掉了其它曲线），按这条曲线的数据自适应
 * 坐标轴（轴型 + 范围），保证它的全部数据点都能画出来 —— 见
 * `chartOption.pickScaleByData`。这里只更新坐标轴，不动 series / 缩放状态。
 */
import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import { buildOption } from '../chartOption'

export default function ChartView({ result, panelIndex, height = '100%',
                                   onVisibleChange, visible = null,
                                   onChart = null, onLayout = null }) {
  const boxRef = useRef(null)
  const chartRef = useRef(null)
  // 最近一次「可见曲线」：图例事件里直接读它，避免闭包拿到旧值
  const visibleRef = useRef(null)
  //: 上一张图的身份（分析项名 + 面板号）。只有**真的换了图**才清掉图例筛选，
  //: 结果对象刷新（跑分析时每算完一项都会推新结果）不能清 —— 以前每刷新一次就
  //: `onVisibleChange(null)`，于是"数据表 → 只看当前图例"经常拿到空选择
  //: （用户实测：MSD 只勾了一个图例，点「只看当前图例」没效果）。
  const lastKeyRef = useRef(null)

  // 初始化 / 销毁
  useEffect(() => {
    if (!boxRef.current) return undefined
    const chart = echarts.init(boxRef.current, null, { renderer: 'canvas' })
    // 图例点选 → 上报"当前可见的曲线"（供右侧说明区过滤），同时让坐标轴跟着变
    chart.on('legendselectchanged', (ev) => {
      const vis = Object.entries(ev.selected || {}).filter(([, v]) => v).map(([k]) => k)
      visibleRef.current = vis
      onVisibleChange?.(vis)
      const res = chart.__mdta_result
      const idx = chart.__mdta_panel
      if (res && vis.length === 1) {
        const opt = buildOption(res, idx, vis)
        // 只换坐标轴：series / dataZoom / 图例选中态都保留
        chart.setOption({ xAxis: opt.xAxis, yAxis: opt.yAxis }, { lazyUpdate: true })
      } else if (res) {
        const opt = buildOption(res, idx, null)
        chart.setOption({ xAxis: opt.xAxis, yAxis: opt.yAxis }, { lazyUpdate: true })
      }
    })
    chartRef.current = chart
    onChart?.(chart)

    // 折叠操作台时容器会连续变化 ~240ms，ResizeObserver 每帧都会回调。
    // 直接每次都 resize() 会在动画期间反复重排（大曲线尤其卡），
    // 这里合并到每帧只做一次，图表就能跟着侧栏平滑展开。
    let raf = 0
    const scheduleResize = () => {
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        if (chartRef.current) {
          chartRef.current.resize()
          onLayout?.()
        }
      })
    }

    window.addEventListener('resize', scheduleResize)
    // 侧栏折叠等布局变化不会触发 window.resize，用 ResizeObserver 兜住
    const ro = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(scheduleResize)
      : null
    if (ro) ro.observe(boxRef.current)
    return () => {
      if (raf) cancelAnimationFrame(raf)
      window.removeEventListener('resize', scheduleResize)
      if (ro) ro.disconnect()
      chart.dispose()
      chartRef.current = null
      onChart?.(null)
    }
  }, [])

  // 数据变化时重画
  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !result) return
    chart.__mdta_result = result
    chart.__mdta_panel = panelIndex
    chart.clear()
    chart.setOption(buildOption(result, panelIndex, null), true)
    requestAnimationFrame(() => onLayout?.())
    // 只有**换了图**（分析项/面板变了）才清掉图例筛选；同一张图的数据刷新
    // （实时跑分析时每算完一项推一次）要保留用户刚才的勾选，并把勾选**重新画上**。
    const key = `${result.name || ''}:${panelIndex}`
    const same = lastKeyRef.current === key
    lastKeyRef.current = key
    const keep = same ? visibleRef.current : null
    if (Array.isArray(keep) && keep.length) {
      const labels = result.curves
        .filter((c) => c.panel === panelIndex).map((c) => c.label)
      const selected = {}
      labels.forEach((l) => { selected[l] = keep.includes(l) })
      chart.setOption({ legend: { selected } }, { lazyUpdate: true })
    } else {
      visibleRef.current = null
      onVisibleChange?.(null)   // 换图/换面板时清掉上一次的图例筛选
    }
  }, [result, panelIndex])

  // 外部（父组件）改动可见曲线时，同步坐标轴
  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !result) return
    if (visible === null || visible === undefined) return
    const opt = buildOption(result, panelIndex, visible)
    chart.setOption({ xAxis: opt.xAxis, yAxis: opt.yAxis }, { lazyUpdate: true })
  }, [visible, result, panelIndex])

  return <div ref={boxRef} className="chart-box" style={{ height }} />
}
