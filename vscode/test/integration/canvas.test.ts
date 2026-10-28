import assert from 'node:assert/strict'
import * as path from 'node:path'
import * as vscode from 'vscode'
import type { ExtensionApi } from '../../src/extension'

// Runs inside a real VS Code (npm test / CI's vscode job) on a workspace
// holding the demo project (`modelmaker-demo`), with modelmaker.pythonPath
// pointing at an interpreter that has the package installed.

const folder = () => path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, 'projects', 'demo_pd_model')

async function activate(): Promise<ExtensionApi> {
  const ext = vscode.extensions.getExtension<ExtensionApi>('quantology.modelmaker')
  assert.ok(ext, 'extension not found')
  return ext.activate()
}

async function until<T>(fn: () => Promise<T | undefined>, timeoutMs: number, what: string): Promise<T> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const value = await fn().catch(() => undefined)
    if (value !== undefined) return value
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`timed out waiting for ${what}`)
}

suite('Model-Maker canvas', () => {
  test('opens the demo project on an authenticated backend, runs it green, and saves', async () => {
    const { backends } = await activate()
    const file = vscode.Uri.file(path.join(folder(), 'model.json'))
    await vscode.commands.executeCommand('vscode.openWith', file, 'modelmaker.model')

    // Shares the editor's backend (same folder).
    const backend = await backends.acquire(folder())
    try {
      assert.equal((await fetch(`${backend.apiBase}/api/health`)).status, 200)
      assert.equal((await fetch(`${backend.apiBase}/api/graph`)).status, 401, 'the API must require the token')

      // The editor loads the project once its backend is up.
      const graph = await until(
        async () => {
          const g = await backend.json<{ blocks: Record<string, { status: string }>; project_path: string | null }>('/graph')
          return g.project_path ? g : undefined
        },
        60_000,
        'the project to load',
      )
      assert.equal(path.resolve(graph.project_path!), path.resolve(folder()))
      assert.ok(Object.keys(graph.blocks).length > 0)

      await backend.json('/run_all', { method: 'POST' })
      const done = await until(
        async () => {
          const g = await backend.json<{ blocks: Record<string, { status: string }>; sweep_running: boolean }>('/graph')
          const statuses = Object.values(g.blocks).map((b) => b.status)
          return !g.sweep_running && !statuses.includes('running') ? statuses : undefined
        },
        300_000,
        'the run to finish',
      )
      assert.deepEqual([...new Set(done)], ['green'])

      const before = (await vscode.workspace.fs.stat(file)).mtime
      await new Promise((r) => setTimeout(r, 1100))
      await backend.json('/project/save', { method: 'POST', body: {} })
      assert.ok((await vscode.workspace.fs.stat(file)).mtime > before, 'save should rewrite model.json')
    } finally {
      backends.release(folder())
    }
  })

  test('the Copilot relay is offered to the backend', async () => {
    const { backends } = await activate()
    const backend = await backends.acquire(folder())
    try {
      const settings = await backend.json<{ providers: string[]; settings: Record<string, { available?: boolean }> }>('/llm/settings')
      assert.ok(settings.providers.includes('copilot'))
      assert.equal(settings.settings.copilot.available, true)
    } finally {
      backends.release(folder())
    }
  })
})
