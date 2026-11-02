import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

// `vite build --mode vscode` (npm run build:vscode): the same app, built for
// the VS Code extension's webview instead of the Python package. Assets are
// loaded relative to the extension's rewritten URIs, and nothing may inject
// an inline script or reach the network, which the webview's CSP forbids.
// The Python package's own build (below) is untouched by it.
const vscodeHtml: Plugin = {
  name: 'modelmaker-vscode-html',
  transformIndexHtml: (html) =>
    html
      // Google Fonts: blocked by the webview CSP; the font stacks fall back.
      .replace(/\s*<link rel="preconnect"[^>]*>/g, '')
      .replace(/\s*<link\s+href="https:\/\/fonts\.googleapis\.com[^>]*>/g, '')
      .replace(/\s*<link rel="icon"[^>]*>/g, ''),
}

export default defineConfig(({ mode }) =>
  mode === 'vscode'
    ? {
        plugins: [react(), vscodeHtml],
        base: './',
        build: {
          outDir: '../vscode/media/webview',
          emptyOutDir: true,
          modulePreload: { polyfill: false },
        },
      }
    : {
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
      },
)
