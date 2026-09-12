/**
 * 生产构建（不经过 vite.config.js 的加载器）。
 *
 * 为什么不用 `vite build`：CLI 会先 `loadConfigFromFile`，其中
 * `windowsSafeRealPathSync` 会 `exec` 一个子进程去解析真实路径；在受限沙箱里
 * 该 spawn 会以 EPERM 失败。这里用编程式 build() 并 `configFile: false`，
 * 把 vite.config.js 的内容原样内联，效果与 `vite build` 完全一致：
 * 产物同样是 dist/，同样由 FastAPI 托管。
 *
 * 用法（在 frontend 目录下）:  node _build.mjs
 */
import { build } from 'vite'
import react from '@vitejs/plugin-react'

const ROOT = new URL('.', import.meta.url).pathname.replace(/^\//, '')

console.log('开始构建前端（configFile: false，等价于 vite build）…')
const t0 = Date.now()
await build({
  configFile: false,
  root: ROOT,
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
})
console.log(`构建完成，用时 ${((Date.now() - t0) / 1000).toFixed(1)}s → dist/`)
