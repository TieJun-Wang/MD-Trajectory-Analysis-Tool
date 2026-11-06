import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

/**
 * 兜底错误边界：渲染/副作用里的异常**显示成一句话**，而不是整页白屏。
 *
 * 为什么必须有：三维视图跑在 WebGL + Three.js 上，一旦某处抛错（哪怕只是数据
 * 索引越界这种小问题），React 会把整棵树卸掉 —— 用户看到的就是"白屏"，除了
 * 控制台之外什么都没有。有了它，屏幕上是可复制的错误信息，能直接拿来报障。
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { err: null }
  }

  static getDerivedStateFromError(err) {
    return { err }
  }

  componentDidCatch(err, info) {
    console.error('[mdta] 界面渲染出错：', err, info)
  }

  render() {
    const { err } = this.state
    if (!err) return this.props.children
    return (
      <div style={{
        margin: '40px auto', maxWidth: 900, padding: '18px 22px',
        fontFamily: 'system-ui, Segoe UI, sans-serif', color: '#7f1d1d',
        background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 10,
      }}>
        <h2 style={{ margin: '0 0 8px', fontSize: 17 }}>界面渲染出错（分析结果不受影响）</h2>
        <p style={{ margin: '0 0 10px', fontSize: 13, color: '#991b1b' }}>
          把下面这段发给开发者即可定位；分析跑在后端，已算出的结果不会丢。
        </p>
        <pre style={{
          margin: 0, padding: 10, maxHeight: 320, overflow: 'auto',
          background: '#fff', border: '1px solid #fecaca', borderRadius: 6,
          fontSize: 11.5, whiteSpace: 'pre-wrap', color: '#111827',
        }}>{String(err?.stack || err)}</pre>
        <button type="button" style={{
          marginTop: 12, padding: '6px 14px', fontSize: 13, cursor: 'pointer',
          border: '1px solid #fca5a5', borderRadius: 6, background: '#fff',
        }} onClick={() => window.location.reload()}>重新加载页面</button>
      </div>
    )
  }
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
