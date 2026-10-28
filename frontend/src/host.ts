// Everything the app needs from where it's running, behind one interface:
// a plain browser tab served by modelmaker-api (the default), or a VS Code
// webview, whose extension injects window.__MODELMAKER__ before the bundle
// loads. Webviews block alert/confirm/prompt and <a download>, can't use
// relative /api URLs, and have no localStorage worth trusting -- so every
// such call in the app goes through `host` instead.

export type PickOptions = {
  // Pick a folder rather than a file.
  folders?: boolean
  // Only offer files with this extension (e.g. '.csv').
  ext?: string
  startPath?: string | null
  // 'save' asks for a new path (a folder name to create), not an existing one.
  mode?: 'open' | 'save'
  defaultName?: string
  title?: string
}

export type HostEvent = { name: string; data?: unknown }

export interface Host {
  readonly kind: 'browser' | 'vscode'
  // Prefix for every API URL: '' in the browser (same origin), the backend's
  // http://127.0.0.1:<port> in VS Code.
  readonly apiBase: string
  authHeaders(): Record<string, string>
  // For URLs used without fetch (<img src>, downloads): adds the auth token
  // as a query parameter, which the backend accepts on GETs only.
  withToken(url: string): string
  saveFile(name: string, data: Blob): Promise<void>
  // Download an API URL (e.g. a library export) to a file of the user's choosing.
  downloadUrl(url: string): Promise<void>
  // A native path picker, or undefined to use the in-app FileBrowser.
  pickPath?: (opts: PickOptions) => Promise<string | null>
  confirm(message: string): Promise<boolean>
  prompt(message: string, defaultValue?: string): Promise<string | null>
  notify(message: string, kind?: 'error' | 'info'): void
  // Open text (e.g. the compiled script) in a real editor tab; VS Code only.
  openText?: (content: string, language: string) => void
  getState(key: string): string | null
  setState(key: string, value: string): void
  // Tell the host something happened (e.g. 'saved'); a no-op in the browser.
  post(name: string, data?: unknown): void
  // Subscribe to host-initiated events (e.g. VS Code's Save / Reload).
  onEvent(handler: (event: HostEvent) => void): () => void
}

declare global {
  interface Window {
    __MODELMAKER__?: { apiBase: string; token: string }
  }
}

function appendQuery(url: string, key: string, value: string): string {
  return `${url}${url.includes('?') ? '&' : '?'}${key}=${encodeURIComponent(value)}`
}

function anchorDownload(href: string, name: string) {
  const a = document.createElement('a')
  a.href = href
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
}

export const browserHost: Host = {
  kind: 'browser',
  apiBase: '',
  authHeaders: () => ({}),
  withToken: (url) => url,
  saveFile: async (name, data) => {
    const url = URL.createObjectURL(data)
    anchorDownload(url, name)
    URL.revokeObjectURL(url)
  },
  downloadUrl: async (url) => anchorDownload(url, ''),
  confirm: async (message) => window.confirm(message),
  prompt: async (message, defaultValue) => window.prompt(message, defaultValue),
  notify: (message) => window.alert(message),
  getState: (key) => {
    try {
      return window.localStorage.getItem(key)
    } catch {
      return null
    }
  },
  setState: (key, value) => {
    try {
      window.localStorage.setItem(key, value)
    } catch {
      // storage unavailable -- the value just won't be remembered
    }
  },
  post: () => {},
  onEvent: () => () => {},
}

// ---- VS Code ------------------------------------------------------------

type VsCodeApi = {
  postMessage(message: unknown): void
  getState(): Record<string, string> | undefined
  setState(state: Record<string, string>): void
}

declare function acquireVsCodeApi(): VsCodeApi

// Messages to and from the extension (see vscode/src/bridge.ts):
//   webview -> extension  {kind: 'request', id, method, params}  (answered)
//                         {kind: 'post', name, data}             (one-way)
//   extension -> webview  {kind: 'response', id, ok, result | error}
//                         {kind: 'event', name, data}
function makeVsCodeHost(config: { apiBase: string; token: string }): Host {
  const vscode = acquireVsCodeApi()
  const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>()
  const listeners = new Set<(event: HostEvent) => void>()
  let nextId = 1

  window.addEventListener('message', (e: MessageEvent) => {
    const msg = e.data
    if (!msg || typeof msg !== 'object') return
    if (msg.kind === 'response') {
      const p = pending.get(msg.id)
      if (!p) return
      pending.delete(msg.id)
      if (msg.ok) p.resolve(msg.result)
      else p.reject(new Error(msg.error ?? 'request failed'))
    } else if (msg.kind === 'event') {
      for (const l of listeners) l({ name: msg.name, data: msg.data })
    }
  })

  function call<T>(method: string, params: unknown): Promise<T> {
    const id = nextId++
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
      vscode.postMessage({ kind: 'request', id, method, params })
    })
  }

  async function blobToBase64(data: Blob): Promise<string> {
    const bytes = new Uint8Array(await data.arrayBuffer())
    let binary = ''
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
    return btoa(binary)
  }

  const authHeaders = () => ({ Authorization: `Bearer ${config.token}` })

  const saveFile = async (name: string, data: Blob) => {
    await call('saveFile', { name, base64: await blobToBase64(data) })
  }

  return {
    kind: 'vscode',
    apiBase: config.apiBase,
    authHeaders,
    withToken: (url) => appendQuery(url, 'token', config.token),
    saveFile,
    downloadUrl: async (url) => {
      const res = await fetch(url, { headers: authHeaders() })
      if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`)
      const disposition = res.headers.get('Content-Disposition') ?? ''
      const name = /filename="?([^";]+)"?/.exec(disposition)?.[1] ?? 'download'
      await saveFile(name, await res.blob())
    },
    pickPath: (opts) => call<string | null>('pickPath', opts),
    confirm: (message) => call<boolean>('confirm', { message }),
    prompt: (message, defaultValue) => call<string | null>('prompt', { message, value: defaultValue }),
    notify: (message, kind = 'error') => vscode.postMessage({ kind: 'post', name: 'notify', data: { message, kind } }),
    openText: (content, language) => vscode.postMessage({ kind: 'post', name: 'openText', data: { content, language } }),
    getState: (key) => vscode.getState()?.[key] ?? null,
    setState: (key, value) => vscode.setState({ ...(vscode.getState() ?? {}), [key]: value }),
    post: (name, data) => vscode.postMessage({ kind: 'post', name, data }),
    onEvent: (handler) => {
      listeners.add(handler)
      return () => listeners.delete(handler)
    },
  }
}

export const host: Host =
  typeof window !== 'undefined' && window.__MODELMAKER__ ? makeVsCodeHost(window.__MODELMAKER__) : browserHost
