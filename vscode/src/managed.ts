import { execFile, spawn } from 'node:child_process'
import { existsSync, mkdirSync, rmSync, statSync } from 'node:fs'
import * as path from 'node:path'
import { ensureUv, UV_VERSION } from './uv'

// Model-Maker's own Python environment (modelmaker.environment = "managed",
// the default): uv creates it in the extension's storage and installs the
// package into it, so a user needs no Python setup at all. uv uses a Python
// 3.11-3.13 already on the machine, or downloads a standalone one. Free of
// the vscode module, so tests/unit can drive it in plain Node.

export const PACKAGE = 'quantology-modelmaker'
// The Pythons the package supports (pyproject.toml's classifiers).
const PYTHON_REQUEST = '>=3.11,<3.14'
// Another window setting the environment up holds this; a lock older than
// this is from a setup that died, and is taken over.
const LOCK_STALE_MS = 15 * 60_000

export type Report = (message: string) => void
export type Log = (text: string) => void

export class ManagedSetupError extends Error {}

export class ManagedEnvironment {
  constructor(
    private readonly storage: string,
    private readonly log: Log,
  ) {}

  get root(): string {
    return path.join(this.storage, 'env')
  }

  get python(): string {
    return process.platform === 'win32' ? path.join(this.root, 'Scripts', 'python.exe') : path.join(this.root, 'bin', 'python')
  }

  /** The environment's Python, with the package installed -- created first
   * if this is the first run (a minute or so; later runs are instant). */
  async ensure(spec: string, report: Report, env: NodeJS.ProcessEnv = process.env): Promise<string> {
    return this.locked(async () => {
      const uv = await this.uv(report)
      if (!existsSync(this.python)) {
        rmSync(this.root, { recursive: true, force: true }) // a half-made one from an interrupted setup
        const phase = 'setting up Python (first run only)'
        report(phase)
        await this.run(uv, ['venv', this.root, '--python', PYTHON_REQUEST], env, (m) => report(`${phase}: ${m}`))
      }
      if (!(await packageVersion(this.python))) {
        const phase = `installing ${PACKAGE} (first run only)`
        report(phase)
        await this.run(uv, ['pip', 'install', '--python', this.python, ...spec.split(/\s+/)], env, (m) => report(`${phase}: ${m}`))
      }
      return this.python
    })
  }

  /** Upgrades the package to the newest version `spec` allows. */
  async upgrade(spec: string, report: Report, env: NodeJS.ProcessEnv = process.env): Promise<void> {
    await this.ensure(spec, report, env)
    await this.locked(async () => {
      const phase = `upgrading ${PACKAGE}`
      report(phase)
      const args = ['pip', 'install', '--python', this.python, '--upgrade', ...spec.split(/\s+/)]
      await this.run(await this.uv(report), args, env, (m) => report(`${phase}: ${m}`))
    })
  }

  /** Deletes the environment; the next start builds a fresh one. */
  reset(): void {
    rmSync(this.root, { recursive: true, force: true })
  }

  private uv(report: Report): Promise<string> {
    return ensureUv(path.join(this.storage, `uv-${UV_VERSION}`), report, this.log).catch((e) => {
      throw new ManagedSetupError(`couldn't download uv: ${e instanceof Error ? e.message : e}`)
    })
  }

  /** Runs uv; `progress` gets what it's downloading (a big package can
   * take a while on a slow line, and the notification should say why). */
  private run(uv: string, args: string[], env: NodeJS.ProcessEnv, progress: Report = () => {}): Promise<void> {
    this.log(`$ uv ${args.join(' ')}\n`)
    return new Promise((resolve, reject) => {
      const proc = spawn(uv, args, {
        windowsHide: true,
        env: {
          ...env,
          // Its Pythons live with the environment, not in the user's uv dirs.
          UV_PYTHON_INSTALL_DIR: path.join(this.storage, 'python'),
          // The OS certificate store: corporate proxies' CAs are there.
          UV_NATIVE_TLS: env.UV_NATIVE_TLS ?? '1',
          UV_NO_PROGRESS: '1',
          VIRTUAL_ENV: undefined,
        },
      })
      let tail = ''
      const collect = (d: Buffer) => {
        const text = d.toString()
        tail = (tail + text).slice(-2000)
        this.log(text)
        const what = /Downloading (\S+(?: \([\d.]+\s*[KMG]i?B\))?)/.exec(text)?.[1]
        if (what) progress(`downloading ${what}`)
      }
      proc.stdout.on('data', collect)
      proc.stderr.on('data', collect)
      proc.on('error', (e) => reject(new ManagedSetupError(`couldn't run uv: ${e.message}`)))
      proc.on('close', (code) => {
        if (code === 0) resolve()
        else reject(new ManagedSetupError(`uv ${args[0]} ${args[1] ?? ''} failed (exit ${code}): ${tail.trim().split('\n').slice(-4).join(' ')}`))
      })
    })
  }

  /** Runs `fn` holding a lock directory, so two windows don't set up the
   * same environment at once. */
  private async locked<T>(fn: () => Promise<T>): Promise<T> {
    mkdirSync(this.storage, { recursive: true })
    const lock = path.join(this.storage, 'env.lock')
    for (;;) {
      try {
        mkdirSync(lock)
        break
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e
        const age = Date.now() - (statSync(lock, { throwIfNoEntry: false })?.mtimeMs ?? Date.now())
        if (age > LOCK_STALE_MS) rmSync(lock, { recursive: true, force: true })
        else await new Promise((r) => setTimeout(r, 1000))
      }
    }
    try {
      return await fn()
    } finally {
      rmSync(lock, { recursive: true, force: true })
    }
  }
}

/** The installed package version in `python`, or undefined. */
export function packageVersion(python: string): Promise<string | undefined> {
  return new Promise((resolve) => {
    execFile(
      python,
      ['-c', `import importlib.metadata as m; print(m.version(${JSON.stringify(PACKAGE)}))`],
      { windowsHide: true },
      (err, stdout) => resolve(err ? undefined : stdout.trim() || undefined),
    )
  })
}
