import assert from 'node:assert/strict'
import * as path from 'node:path'
import { test } from 'node:test'
import { BackendManager } from '../../src/backend'
import { ModelEditorProvider } from '../../src/editor'
import { EventEmitter } from '../stub/vscode'

// A backend restarted while a canvas was still on its first render: the
// render finished last and pointed the canvas, and Claude Code's discovery
// file, at the backend that had been stopped -- every modelmaker-mcp tool
// then said "no Model-Maker canvas is open".

type Fake = { port: number; token: string; apiBase: string; stopped: boolean; json: () => Promise<unknown> }

function fakeBackend(port: number, loaded: Promise<void> = Promise.resolve()): Fake {
  return { port, token: `tok-${port}`, apiBase: `http://127.0.0.1:${port}`, stopped: false, json: () => loaded.then(() => ({})) }
}

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>((r) => (resolve = r))
  return { promise, resolve }
}

function fakePanel() {
  return {
    active: true,
    viewColumn: 1,
    reveal: () => {},
    onDidChangeViewState: () => ({ dispose: () => {} }),
    onDidDispose: () => ({ dispose: () => {} }),
    webview: {
      html: '',
      options: {},
      cspSource: 'vscode-resource:',
      asWebviewUri: (u: { fsPath: string }) => u.fsPath,
      onDidReceiveMessage: () => ({ dispose: () => {} }),
      postMessage: () => {},
    },
  }
}

test('a render for a backend restarted meanwhile neither shows nor publishes it', async () => {
  const folder = path.resolve('/projects/fm_mortgages')
  const restarted = new EventEmitter<string>()
  // A's load is still in flight when the restart lands.
  const aLoaded = deferred()
  const a = fakeBackend(60071, aLoaded.promise)
  const b = fakeBackend(62334)
  let current = a
  const backends = { onDidRestart: restarted.event, acquire: async () => current, release: () => {} }
  const published: string[] = []
  const discovery = { publish: (_f: string, url: string) => published.push(url), touch: () => {}, withdraw: () => {} }
  const context = { subscriptions: [] as unknown[], extensionUri: { fsPath: '/ext' } }
  const log = { appendLine: () => {} }
  const editor = new ModelEditorProvider(context as never, backends as never, log as never, discovery as never)
  const panel = fakePanel()

  const opening = editor.resolveCustomEditor({ uri: { fsPath: path.join(folder, 'model.json') } } as never, panel as never)
  await new Promise((r) => setImmediate(r))

  // The restart: A stopped, B started and rendered.
  a.stopped = true
  current = b
  restarted.fire(folder)
  await new Promise((r) => setImmediate(r))
  assert.deepEqual(published, [b.apiBase])

  // Now A's first render finishes (or fails, the process being gone).
  aLoaded.resolve()
  await opening
  assert.deepEqual(published, [b.apiBase], 'the stopped backend must not replace the live one')
  assert.match(panel.webview.html, /127\.0\.0\.1:62334/)
  assert.doesNotMatch(panel.webview.html, /60071/)
})

test('restart(which) restarts only the backends it picks', async () => {
  const relay = {}
  const environments = {}
  const manager = new BackendManager({ appendLine: () => {} } as never, relay as never, environments as never)
  const started: string[] = []
  let port = 50000
  ;(manager as unknown as { start: (key: string) => Promise<unknown> }).start = async (key: string) => {
    started.push(path.basename(key))
    return { ...fakeBackend(port++), stop: () => {} }
  }
  const managed = path.resolve('/projects/managed')
  const own = path.resolve('/projects/own-python')
  await manager.acquire(managed)
  await manager.acquire(own)
  started.length = 0

  await manager.restart(undefined, (f) => f === own)
  assert.deepEqual(started, ['own-python'])
  manager.dispose()
})
