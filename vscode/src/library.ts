import * as path from 'node:path'
import * as vscode from 'vscode'
import type { BackendManager } from './backend'
import { attachBridge } from './bridge'
import { loadingHtml, type ModelEditorProvider, webviewHtml } from './editor'

// The block library in an editor tab of its own, one per project, beside
// its canvas: the same app (frontend/src/LibraryPage.tsx) on the same
// backend. What it does to the canvas -- place a block, open the update
// dialog, tell the palette the library changed -- is forwarded to the
// canvas, which does it, so it's in the canvas's undo history.

export const LIBRARY_VIEW_TYPE = 'modelmaker.library'

export class LibraryPanels implements vscode.Disposable {
  private readonly panels = new Map<string, vscode.WebviewPanel>()

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly backends: BackendManager,
    private readonly editor: ModelEditorProvider,
    private readonly log: vscode.OutputChannel,
  ) {
    editor.onOpenLibrary = (folder, view) => this.open(folder, view)
    context.subscriptions.push(
      backends.onDidRestart(async (folder) => {
        const panel = this.panels.get(folder)
        if (panel) await this.render(panel, folder, undefined)
      }),
    )
  }

  /** Opens (or brings to the front, on `view`) the library tab of a project. */
  async open(folder: string, view?: unknown) {
    const existing = this.panels.get(folder)
    if (existing) {
      existing.reveal(existing.viewColumn)
      if (view) existing.webview.postMessage({ kind: 'event', name: 'libraryView', data: view })
      return
    }
    const panel = vscode.window.createWebviewPanel(
      LIBRARY_VIEW_TYPE,
      `Block Library: ${path.basename(folder)}`,
      { viewColumn: vscode.ViewColumn.Active, preserveFocus: false },
      { enableScripts: true, retainContextWhenHidden: true },
    )
    panel.iconPath = vscode.Uri.joinPath(this.context.extensionUri, 'icon.png')
    this.panels.set(folder, panel)
    const bridge = attachBridge(panel.webview, folder, {
      onSaved: () => {},
      onPost: (name, data) => this.fromLibrary(folder, name, data),
    })
    let acquired = false
    panel.onDidDispose(() => {
      bridge.dispose()
      this.panels.delete(folder)
      if (acquired) this.backends.release(folder)
    })
    panel.webview.html = loadingHtml(`Opening the block library for ${path.basename(folder)}…`)
    try {
      await this.backends.acquire(folder)
      acquired = true
    } catch (e) {
      panel.webview.html = loadingHtml(`Couldn't start Model-Maker: ${e instanceof Error ? e.message : e}`, true)
      return
    }
    await this.render(panel, folder, view)
  }

  dispose() {
    for (const panel of this.panels.values()) panel.dispose()
  }

  private async render(panel: vscode.WebviewPanel, folder: string, view: unknown) {
    const backend = await this.backends.acquire(folder).finally(() => this.backends.release(folder))
    // Restarted while this waited: the restart renders the new one.
    if (backend.stopped) return
    const media = vscode.Uri.joinPath(this.context.extensionUri, 'media', 'webview')
    panel.webview.options = {
      enableScripts: true,
      localResourceRoots: [media],
      portMapping: [{ webviewPort: backend.port, extensionHostPort: backend.port }],
    }
    panel.webview.html = await webviewHtml(panel.webview, media, backend, { view: 'library', ...(view ? { libraryView: view } : {}) })
  }

  private fromLibrary(folder: string, name: string, data: unknown) {
    switch (name) {
      case 'placeLibraryBlock':
      case 'placeBuiltinBlock':
      case 'openUpdate':
        if (!this.editor.postToCanvas(folder, name, data)) {
          vscode.window.showWarningMessage(`Open ${path.basename(folder)}'s canvas to place blocks on it.`)
          return
        }
        this.editor.revealCanvas(folder)
        return
      case 'libraryChanged':
        this.editor.postToCanvas(folder, name, data)
        return
      case 'showCanvas':
        this.editor.revealCanvas(folder)
        return
      default:
        this.log.appendLine(`[library] unknown message: ${name}`)
    }
  }
}
