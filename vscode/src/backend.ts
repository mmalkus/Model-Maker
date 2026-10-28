import { type ChildProcess, execFile, spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import * as net from 'node:net'
import * as path from 'node:path'
import * as vscode from 'vscode'
import type { CopilotRelay } from './copilot'
import { resolvePython } from './python'

// The /api/version "api" level this extension speaks (modelmaker/api.py's
// API_COMPAT_LEVEL). A backend reporting another level -- or none, from
// before the endpoint existed -- is refused with an offer to upgrade.
export const API_COMPAT = 1

const PACKAGE = 'quantology-modelmaker'
const HEALTH_TIMEOUT_MS = 90_000
// The webview's origin; the backend admits it through CORS.
const WEBVIEW_ORIGIN_REGEX = '^vscode-webview://[a-z0-9-]+$'

export class BackendError extends Error {}

/** One `modelmaker-api` process, serving one project folder. The backend
 * keeps a single global session, so each open project gets its own. */
export class Backend {
  private stopping = false
  private readonly exitEmitter = new vscode.EventEmitter<number | null>()
  /** Fires if the process exits without being asked to. */
  readonly onDidExit = this.exitEmitter.event

  constructor(
    readonly folder: string,
    readonly python: string,
    readonly port: number,
    readonly token: string,
    private readonly proc: ChildProcess,
  ) {
    proc.on('exit', (code) => {
      if (!this.stopping) this.exitEmitter.fire(code)
    })
  }

  get apiBase() {
    return `http://127.0.0.1:${this.port}`
  }

  get exited() {
    return this.proc.exitCode !== null || this.proc.signalCode !== null
  }

  request(apiPath: string, init: { method?: string; body?: unknown } = {}): Promise<Response> {
    return fetch(`${this.apiBase}/api${apiPath}`, {
      method: init.method ?? 'GET',
      headers: { Authorization: `Bearer ${this.token}`, 'Content-Type': 'application/json' },
      body: init.body === undefined ? undefined : JSON.stringify(init.body),
    })
  }

  async json<T>(apiPath: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
    const res = await this.request(apiPath, init)
    const body = await res.json().catch(() => ({}))
    if (!res.ok) throw new BackendError((body as { detail?: string }).detail ?? `HTTP ${res.status} from ${apiPath}`)
    return body as T
  }

  stop() {
    this.stopping = true
    killTree(this.proc)
    this.exitEmitter.dispose()
  }
}

type Entry = { backend: Promise<Backend>; refs: number }

/** Starts, shares (by folder) and stops backends. */
export class BackendManager implements vscode.Disposable {
  private readonly entries = new Map<string, Entry>()
  private readonly restartEmitter = new vscode.EventEmitter<string>()
  /** Fires with a folder whose backend was replaced (restart, crash). */
  readonly onDidRestart = this.restartEmitter.event

  constructor(
    private readonly log: vscode.OutputChannel,
    private readonly relay: CopilotRelay,
  ) {}

  /** The folder's backend, started if needed; pair with release(). */
  acquire(folder: string): Promise<Backend> {
    const key = path.resolve(folder)
    let entry = this.entries.get(key)
    if (!entry) {
      entry = { backend: this.start(key), refs: 0 }
      this.entries.set(key, entry)
      entry.backend.catch(() => this.entries.delete(key))
    }
    entry.refs++
    return entry.backend
  }

  release(folder: string) {
    const key = path.resolve(folder)
    const entry = this.entries.get(key)
    if (!entry || --entry.refs > 0) return
    this.entries.delete(key)
    entry.backend.then((b) => b.stop()).catch(() => {})
  }

  /** Restarts every running backend (or one folder's), keeping its users. */
  async restart(folder?: string) {
    const keys = folder ? [path.resolve(folder)] : [...this.entries.keys()]
    await Promise.all(
      keys.map(async (key) => {
        const entry = this.entries.get(key)
        if (!entry) return
        entry.backend.then((b) => b.stop()).catch(() => {})
        entry.backend = this.start(key)
        await entry.backend.catch(() => {})
        this.restartEmitter.fire(key)
      }),
    )
  }

  dispose() {
    for (const entry of this.entries.values()) entry.backend.then((b) => b.stop()).catch(() => {})
    this.entries.clear()
    this.restartEmitter.dispose()
  }

  private async start(folder: string): Promise<Backend> {
    const uri = vscode.Uri.file(folder)
    const python = await resolvePython(uri)
    return vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: `Model-Maker: starting backend for ${path.basename(folder)}` },
      async (progress) => {
        await ensurePackage(python, this.log, progress)
        let backend = await this.spawnHealthy(folder, python, progress)
        if (!(await isCompatible(backend))) {
          backend.stop()
          await offerUpgrade(python, this.log, progress)
          backend = await this.spawnHealthy(folder, python, progress)
          if (!(await isCompatible(backend))) {
            backend.stop()
            throw new BackendError(
              `the ${PACKAGE} package in ${python} is too old or too new for this extension -- ` +
                `upgrade both, or set modelmaker.packageSpec to a matching version`,
            )
          }
        }
        backend.onDidExit((code) => this.crashed(folder, code))
        return backend
      },
    )
  }

  private async crashed(folder: string, code: number | null) {
    this.log.appendLine(`[${path.basename(folder)}] backend exited unexpectedly (code ${code})`)
    const choice = await vscode.window.showWarningMessage(
      `The Model-Maker backend for ${path.basename(folder)} stopped (exit code ${code}).`,
      'Restart',
      'Show Log',
    )
    if (choice === 'Restart') await this.restart(folder)
    else if (choice === 'Show Log') this.log.show()
  }

  private async spawnHealthy(folder: string, python: string, progress: Progress): Promise<Backend> {
    progress.report({ message: 'launching' })
    // A free port can be taken between checking and binding: retry then.
    for (let attempt = 1; ; attempt++) {
      const backend = await this.spawn(folder, python, await freePort())
      try {
        await waitHealthy(backend)
        return backend
      } catch (e) {
        backend.stop()
        if (!(e instanceof PortTaken) || attempt >= 3) throw e
      }
    }
  }

  private async spawn(folder: string, python: string, port: number): Promise<Backend> {
    const config = vscode.workspace.getConfiguration('modelmaker', vscode.Uri.file(folder))
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      MODELMAKER_HOST: '127.0.0.1',
      MODELMAKER_PORT: String(port),
      MODELMAKER_CORS_ORIGIN_REGEX: WEBVIEW_ORIGIN_REGEX,
      // Per project, so two windows never share a recovery snapshot.
      MODELMAKER_CACHE_DIR: path.join(folder, '.modelmaker-cache'),
    }
    if (config.get<boolean>('llm.useCopilot', true)) {
      env.MODELMAKER_COPILOT_BASE_URL = await this.relay.url()
      env.MODELMAKER_COPILOT_TOKEN = this.relay.token
      const hasCopilot = (await vscode.lm.selectChatModels({ vendor: 'copilot' }).then((m) => m.length > 0, () => false))
      if (hasCopilot && !env.MODELMAKER_LLM_PROVIDER) env.MODELMAKER_LLM_PROVIDER = 'copilot'
    }
    Object.assign(env, config.get<Record<string, string>>('backend.extraEnv', {}))

    // Every /api call must carry it (see modelmaker/api.py's _auth_token):
    // other local processes and web pages can't drive the backend.
    const token = randomBytes(24).toString('hex')
    env.MODELMAKER_AUTH_TOKEN = token

    const name = path.basename(folder)
    this.log.appendLine(`[${name}] ${python} -m modelmaker.api on port ${port}`)
    const proc = spawn(python, ['-c', 'from modelmaker.api import main; main()'], {
      cwd: folder,
      env,
      // Its own process group, so stop() takes the worker pool with it.
      detached: process.platform !== 'win32',
      windowsHide: true,
    })
    const backend = new Backend(folder, python, port, token, proc)
    proc.stdout?.on('data', (d: Buffer) => this.log.append(prefix(name, d)))
    proc.stderr?.on('data', (d: Buffer) => {
      const text = d.toString()
      stderrTail.set(backend, ((stderrTail.get(backend) ?? '') + text).slice(-4000))
      this.log.append(prefix(name, d))
    })
    proc.on('error', (e) => stderrTail.set(backend, String(e)))
    return backend
  }
}

type Progress = vscode.Progress<{ message?: string }>

const stderrTail = new WeakMap<Backend, string>()

class PortTaken extends Error {}

function prefix(name: string, data: Buffer): string {
  return data
    .toString()
    .split(/(?<=\n)/)
    .map((line) => `[${name}] ${line}`)
    .join('')
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

async function waitHealthy(backend: Backend) {
  const deadline = Date.now() + HEALTH_TIMEOUT_MS
  while (Date.now() < deadline) {
    if (backend.exited) {
      const tail = stderrTail.get(backend) ?? ''
      if (/address already in use|EADDRINUSE|10048/i.test(tail)) throw new PortTaken(tail)
      throw new BackendError(`the backend exited during startup:\n${tail.trim().split('\n').slice(-8).join('\n')}`)
    }
    try {
      const res = await fetch(`${backend.apiBase}/api/health`)
      if (res.ok) return
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 300))
  }
  throw new BackendError(`the backend didn't answer within ${HEALTH_TIMEOUT_MS / 1000}s -- see Model-Maker: Show Backend Log`)
}

async function isCompatible(backend: Backend): Promise<boolean> {
  const res = await backend.request('/version').catch(() => undefined)
  if (!res?.ok) return false
  const body = (await res.json().catch(() => ({}))) as { api?: number }
  return body.api === API_COMPAT
}

function run(python: string, args: string[], log: vscode.OutputChannel): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    const proc = spawn(python, args, { windowsHide: true })
    let out = ''
    const collect = (d: Buffer) => {
      out += d.toString()
      log.append(d.toString())
    }
    proc.stdout.on('data', collect)
    proc.stderr.on('data', collect)
    proc.on('error', (e) => resolve({ code: -1, out: String(e) }))
    proc.on('close', (code) => resolve({ code: code ?? -1, out }))
  })
}

/** The installed package version, or undefined if it isn't installed. */
export function installedVersion(python: string, log?: vscode.OutputChannel): Promise<string | undefined> {
  return new Promise((resolve, reject) => {
    execFile(
      python,
      ['-c', `import importlib.metadata as m; print(m.version(${JSON.stringify(PACKAGE)}))`],
      { windowsHide: true },
      (err, stdout, stderr) => {
        if (err && (err as NodeJS.ErrnoException).code === 'ENOENT') {
          reject(new BackendError(`no Python at ${python} -- pick one with Model-Maker: Select Python Interpreter`))
        } else if (err) {
          log?.append(stderr)
          resolve(undefined)
        } else {
          resolve(stdout.trim() || undefined)
        }
      },
    )
  })
}

/** pip install -U <modelmaker.packageSpec> into `python`. */
export async function installPackage(python: string, log: vscode.OutputChannel, progress?: Progress) {
  const spec = vscode.workspace.getConfiguration('modelmaker').get<string>('packageSpec', PACKAGE).trim() || PACKAGE
  progress?.report({ message: `pip install -U ${spec}` })
  log.appendLine(`$ ${python} -m pip install -U ${spec}`)
  const { code, out } = await run(python, ['-m', 'pip', 'install', '-U', ...spec.split(/\s+/)], log)
  if (code !== 0) {
    const hint = /externally-managed-environment/.test(out)
      ? ' This Python is managed by the OS -- select a virtual environment instead (Model-Maker: Select Python Interpreter).'
      : ''
    throw new BackendError(`pip install failed (exit ${code}).${hint} See Model-Maker: Show Backend Log.`)
  }
}

async function ensurePackage(python: string, log: vscode.OutputChannel, progress: Progress) {
  progress.report({ message: 'checking the Python package' })
  if (await installedVersion(python, log)) return
  await askToInstall(python, log, progress, `Model-Maker needs the ${PACKAGE} Python package, which isn't installed in ${python}.`)
}

async function offerUpgrade(python: string, log: vscode.OutputChannel, progress: Progress) {
  await askToInstall(python, log, progress, `The ${PACKAGE} package in ${python} doesn't match this extension.`, 'Upgrade')
}

async function askToInstall(python: string, log: vscode.OutputChannel, progress: Progress, why: string, verb = 'Install') {
  const mode = vscode.workspace.getConfiguration('modelmaker').get<string>('autoInstall', 'ask')
  if (mode === 'never') throw new BackendError(`${why} Install it yourself, or set modelmaker.autoInstall.`)
  if (mode === 'ask') {
    const choice = await vscode.window.showInformationMessage(`${why} ${verb} it now?`, { modal: true }, verb, 'Select Interpreter')
    if (choice === 'Select Interpreter') {
      await vscode.commands.executeCommand('modelmaker.selectInterpreter')
      throw new BackendError('interpreter changed -- reopen the project to use it')
    }
    if (choice !== verb) throw new BackendError(`${why} Not installed.`)
  }
  await installPackage(python, log, progress)
}

function killTree(proc: ChildProcess) {
  if (proc.pid === undefined || proc.exitCode !== null) return
  if (process.platform === 'win32') {
    // The worker pool's processes are children: /T takes the whole tree.
    spawn('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { windowsHide: true })
    return
  }
  try {
    process.kill(-proc.pid, 'SIGTERM')
  } catch {
    proc.kill('SIGTERM')
  }
  const pid = proc.pid
  setTimeout(() => {
    if (proc.exitCode === null && proc.signalCode === null) {
      try {
        process.kill(-pid, 'SIGKILL')
      } catch {
        // already gone
      }
    }
  }, 3000).unref()
}
