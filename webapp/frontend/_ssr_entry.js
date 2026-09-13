/**
 * SSR 自检入口：把所有组件与工具函数统一导出，
 * 供 _ui_render.mjs 打包后逐个渲染验证。
 */
export { default as App } from './src/App.jsx'
export { default as OpenPanel } from './src/components/OpenPanel.jsx'
export { default as InfoPanel, InfoReport } from './src/components/InfoPanel.jsx'
export { default as ChartPanel } from './src/components/ChartPanel.jsx'
export { default as ChartView } from './src/components/ChartView.jsx'
export { default as DataPanel } from './src/components/DataPanel.jsx'
export { default as SummaryView } from './src/components/SummaryView.jsx'
export { default as NotesPanel } from './src/components/NotesPanel.jsx'
export { default as StatsPanel } from './src/components/StatsPanel.jsx'
export { default as ExportPanel, DirPicker } from './src/components/ExportPanel.jsx'
export { default as LogPanel } from './src/components/LogPanel.jsx'
export { default as QcPanel } from './src/components/QcPanel.jsx'
export {
  SelectionBlock,
  FrameBlock,
  ParamBlock,
  FunctionBlock,
  RunBlock,
} from './src/components/SetupPanel.jsx'
export { buildOption, formatNumber, formatCell, keyStats } from './src/chartOption.js'
export { FALLBACK_GROUPS, normalizeGroups, groupPresent } from './src/analysisGroups.js'
export { api, guessTrajectory } from './src/api.js'
