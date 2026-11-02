import * as path from 'node:path'
import * as vscode from 'vscode'

// The extension side of frontend/src/host.ts's VS Code host: native dialogs
// and notifications for the canvas, which a webview can't show itself.
//   webview -> extension  {kind: 'request', id, method, params}  (answered)
//                         {kind: 'post', name, data}             (one-way)
//   extension -> webview  {kind: 'response', id, ok, result | error}
//                         {kind: 'event', name, data}

type PickOptions = { folders?: boolean; ext?: string; startPath?: string | null; mode?: 'open' | 'save'; defaultName?: string; title?: string }

export type BridgeHooks = {
  // The canvas is writing, or just wrote, the project file (so a
  // file-watcher event now is ours).
  onSaved(): void
}

export function attachBridge(webview: vscode.Webview, projectFolder: string, hooks: BridgeHooks): vscode.Disposable {
  return webview.onDidReceiveMessage(async (msg) => {
    if (!msg || typeof msg !== 'object') return
    if (msg.kind === 'post') return handlePost(msg.name, msg.data, hooks)
    if (msg.kind !== 'request') return
    try {
      const result = await handleRequest(msg.method, msg.params ?? {}, projectFolder)
      webview.postMessage({ kind: 'response', id: msg.id, ok: true, result: result ?? null })
    } catch (e) {
      webview.postMessage({ kind: 'response', id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) })
    }
  })
}

async function handleRequest(method: string, params: Record<string, unknown>, projectFolder: string): Promise<unknown> {
  switch (method) {
    case 'confirm': {
      const choice = await vscode.window.showWarningMessage(String(params.message), { modal: true }, 'OK')
      return choice === 'OK'
    }
    case 'prompt':
      return (await vscode.window.showInputBox({ prompt: String(params.message), value: (params.value as string) ?? '' })) ?? null
    case 'saveFile': {
      const name = String(params.name)
      const target = await vscode.window.showSaveDialog({ defaultUri: vscode.Uri.file(path.join(projectFolder, name)) })
      if (!target) return false
      await vscode.workspace.fs.writeFile(target, Buffer.from(String(params.base64), 'base64'))
      return true
    }
    case 'pickPath':
      return pickPath(params as PickOptions, projectFolder)
    default:
      throw new Error(`unknown request: ${method}`)
  }
}

async function pickPath(opts: PickOptions, projectFolder: string): Promise<string | null> {
  const start = vscode.Uri.file(opts.startPath || projectFolder)
  if (opts.mode === 'save') {
    // A new project folder: the "file" the dialog names is the folder.
    const target = await vscode.window.showSaveDialog({
      defaultUri: opts.defaultName ? vscode.Uri.joinPath(start, opts.defaultName) : start,
      title: opts.title,
      saveLabel: 'Use this folder',
    })
    return target?.fsPath ?? null
  }
  const ext = opts.ext?.replace(/^\./, '')
  const picked = await vscode.window.showOpenDialog({
    defaultUri: start,
    canSelectFiles: !opts.folders,
    canSelectFolders: !!opts.folders,
    canSelectMany: false,
    title: opts.title,
    filters: ext ? { [ext.toUpperCase()]: [ext], 'All files': ['*'] } : undefined,
  })
  return picked?.[0]?.fsPath ?? null
}

async function handlePost(name: string, data: unknown, hooks: BridgeHooks) {
  const d = (data ?? {}) as Record<string, string>
  if (name === 'notify') {
    if (d.kind === 'info') vscode.window.showInformationMessage(d.message)
    else vscode.window.showErrorMessage(d.message)
  } else if (name === 'openText') {
    const doc = await vscode.workspace.openTextDocument({ content: d.content, language: d.language })
    await vscode.window.showTextDocument(doc, { preview: false })
  } else if (name === 'saving' || name === 'saved') {
    hooks.onSaved()
  }
}
