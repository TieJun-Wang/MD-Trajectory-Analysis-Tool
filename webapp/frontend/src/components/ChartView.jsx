/**
 * ECharts 单图组件。
 *
 * 一次只显示一张图（由父组件通过 title/结果决定显示哪张），
 * 图表自适应容器大小；缩放 / 平移 / hover / 图例开关由 ECharts 提供。
 */
import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import { buildOption } from '../chartOption'

export default function ChartView({ result, panelIndex, height = '100%' }) {
  const boxRef = useRef(null)
  const chartRef = useRef(null)

  // 初始化 / 销毁
  useEffect(() => {
    if (!boxRef.current) return undefined
    const chart = echarts.init(boxRef.current, null, { renderer: 'canvas' })
    chartRef.current = chart

    // 折叠操作台时容器会连续变化 ~240ms，ResizeObserver 每帧都会回调。
    // 直接每次都 resize() 会在动画期间反复重排（大曲线尤其卡），
    // 这里合并到每帧只做一次，图表就能跟着侧栏平滑展开。
    let raf = 0
    const scheduleResize = () => {
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        if (chartRef.current) chartRef.current.resize()
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
    }
  }, [])

  // 数据变化时重画
  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !result) return
    chart.clear()
    chart.setOption(buildOption(result, panelIndex), true)
  }, [result, panelIndex])

  return <div ref={boxRef} className="chart-box" style={{ height }} />
}
