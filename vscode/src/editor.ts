import { randomBytes } from 'node:crypto'
import * as path from 'node:path'
import * as vscode from 'vscode'
import { type Backend, type BackendManager } from './backend'
import { attachBridge } from './bridge'

export const VIEW_TYPE = 'modelmaker.model'

// A file event this soon after the canvas saved is that save, not an
// outside edit.
const OWN_SAVE_WINDOW_MS = 3000

/** The canvas for a project's model.json. The backend is the source of
 * truth: it loads the project, the canvas edits it through the API, and its
 * autosave (or Save) writes model.json -- so the editor itself is read-only
 * as far as VS Code's document model goes. */
export class ModelEditorProvider implements vscode.CustomReadonlyEditorProvider {
  // Open canvases, by project folder: restarted backends re-render them,
  // and Model-Maker: Save goes to the active one.
  private readonly panels = new Map<string, Set<vscode.WebviewPanel>>()
  private active: vscode.WebviewPanel | undefined

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly backends: BackendManager,
    private readonly log: vscode.OutputChannel,
  ) {
    context.subscriptions.push(
      backends.onDidRestart(async (folder) => {
        for (const panel of this.panels.get(folder) ?? []) await this.render(panel, folder)
      }),
    )
  }

  static register(context: vscode.ExtensionContext, backends: BackendManager, log: vscode.OutputChannel) {
    const provider = new ModelEditorProvider(context, backends, log)
    context.subscriptions.push(
      vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, {
        // The canvas's own state (selection, viewport, open panels) survives
        // switching tabs; rebuilding it from the backend would lose it.
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false,
      }),
    )
    return provider
  }

  /** Asks the active canvas to save (Model-Maker: Save, Ctrl/Cmd-S). */
  saveActive() {
    this.active?.webview.postMessage({ kind: 'event', name: 'save' })
  }

  openCustomDocument(uri: vscode.Uri): vscode.CustomDocument {
    return { uri, dispose: () => {} }
  }

  async resolveCustomEditor(document: vscode.CustomDocument, panel: vscode.WebviewPanel): Promise<void> {
    const folder = path.resolve(path.dirname(document.uri.fsPath))
    const set = this.panels.get(folder) ?? new Set()
    set.add(panel)
    this.panels.set(folder, set)
    this.active = panel

    let lastSave = 0
    const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(vscode.Uri.file(folder), 'model.json'))
    const disposables: vscode.Disposable[] = [
      watcher,
      attachBridge(panel.webview, folder, { onSaved: () => (lastSave = Date.now()) }),
      watcher.onDidChange(async () => {
        if (Date.now() - lastSave < OWN_SAVE_WINDOW_MS) return
        const choice = await vscode.window.showInformationMessage(
          `${path.basename(folder)}/model.json changed on disk. Reload the canvas from it?`,
          'Reload',
        )
        if (choice === 'Reload') panel.webview.postMessage({ kind: 'event', name: 'reload' })
      }),
      panel.onDidChangeViewState(() => {
        if (panel.active) this.active = panel
      }),
    ]
    panel.onDidDispose(() => {
      disposables.forEach((d) => d.dispose())
      set.delete(panel)
      if (this.active === panel) this.active = undefined
      this.backends.release(folder)
    })

    panel.webview.html = loadingHtml(`Starting the Model-Maker backend for ${path.basename(folder)}…`)
    await this.render(panel, folder, true)
  }

  private async render(panel: vscode.WebviewPanel, folder: string, first = false) {
    let backend: Backend
    try {
      backend = await (first ? this.backends.acquire(folder) : this.backends.acquire(folder).finally(() => this.backends.release(folder)))
      await backend.json('/project/load', { method: 'POST', body: { path: folder } })
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      this.log.appendLine(`[${path.basename(folder)}] ${message}`)
      panel.webview.html = loadingHtml(`Couldn't start Model-Maker: ${message}`, true)
      return
    }
    const media = vscode.Uri.joinPath(this.context.extensionUri, 'media', 'webview')
    panel.webview.options = {
      enableScripts: true,
      localResourceRoots: [media],
      // The backend listens on the extension host's machine; this forwards
      // the same port from the webview, local or remote (SSH, WSL, ...).
      portMapping: [{ webviewPort: backend.port, extensionHostPort: backend.port }],
    }
    panel.webview.html = await webviewHtml(panel.webview, media, backend)
  }
}

async function webviewHtml(webview: vscode.Webview, media: vscode.Uri, backend: Backend): Promise<string> {
  let html: string
  try {
    html = Buffer.from(await vscode.workspace.fs.readFile(vscode.Uri.joinPath(media, 'index.html'))).toString('utf8')
  } catch {
    return loadingHtml('The canvas bundle is missing from this build of the extension (media/webview).', true)
  }
  const nonce = randomBytes(16).toString('base64')
  const api = `http://127.0.0.1:${backend.port}`
  const csp = [
    "default-src 'none'",
    // The entry script by nonce; the chunks it imports come from media/.
    `script-src 'nonce-${nonce}' ${webview.cspSource}`,
    // React Flow and CodeMirror set inline styles.
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `img-src ${webview.cspSource} data: blob: ${api}`,
    `font-src ${webview.cspSource} data:`,
    `connect-src ${api}`,
  ].join('; ')
  const config = JSON.stringify({ apiBase: api, token: backend.token })
  const head = [
    `<base href="${webview.asWebviewUri(media)}/">`,
    `<meta http-equiv="Content-Security-Policy" content="${csp}">`,
    `<script nonce="${nonce}">window.__MODELMAKER__ = ${config}</script>`,
  ].join('\n    ')
  return html.replace('<head>', `<head>\n    ${head}`).replace(/<script type="module"/g, `<script nonce="${nonce}" type="module"`)
}

function loadingHtml(message: string, error = false): string {
  const escaped = message.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!)
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"></head>
<body style="font-family: var(--vscode-font-family); color: var(--vscode-${error ? 'errorForeground' : 'foreground'}); padding: 24px; white-space: pre-wrap">${escaped}${
    error ? '\n\nSee Model-Maker: Show Backend Log, then reopen the file or run Model-Maker: Restart Backend.' : ''
  }</body></html>`
}
