import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时 `npm run dev` 起在 5173，/api 反向代理到 FastAPI（8000）；
// 发布时 `npm run build` 生成 dist/，由 FastAPI 直接托管，同源访问 /api。
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
})
