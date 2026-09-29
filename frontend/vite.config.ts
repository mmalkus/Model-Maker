import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        // Overridable so a second checkout (e.g. a git worktree) can run its
        // own API server alongside the usual one on 8001.
        target: process.env.MODELMAKER_API_URL ?? 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Emit into the Python package so it ships as part of the modelmaker
    // wheel/sdist and modelmaker.api can serve it directly.
    outDir: '../modelmaker/static',
    emptyOutDir: true,
  },
})
