import { execFile, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { cp, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import * as path from 'node:path'

// The Python side of running Model-Maker from the extension, without VS Code
// APIs (unit-tested in test/unit/pyenv.test.ts; environment.ts is the glue):
//
// - the bundled package (python/ in the .vsix, see scripts/bundle-python.py),
//   copied to a stable folder so paths written elsewhere (.mcp.json) survive
//   extension updates;
// - the managed environment: a venv of the extension's own, made from any
//   Python 3.11+, holding only Model-Maker's open-source requirements;
// - for a user's own interpreter: which of those requirements it lacks.
//
// No Model-Maker package is ever pip installed by any of it.

export class EnvironmentError extends Error {}

export type Bundle = { version: string; requires_python: string; requirements: string[] }

export type Log = (text: string) => void
export type Report = (message: string) => void

export type Candidate = { command: string; args?: string[]; label?: string }

/** A Python found on the machine: its real executable and version. */
export type Found = { executable: string; version: [number, number] }

const IS_WIN = process.platform === 'win32'
const LOCK_STALE_MS = 30 * 60_000
const LOCK_WAIT_MS = 30 * 60_000

export async function readBundle(dir: string): Promise<Bundle | undefined> {
  try {
    return JSON.parse(await readFile(path.join(dir, 'bundle.json'), 'utf8')) as Bundle
  } catch {
    return undefined
  }
}

/** ">=3.11" -> [3, 11] (only the >= form pyproject uses). */
export function minimumPython(spec: string): [number, number] {
  const m = /(\d+)\.(\d+)/.exec(spec)
  return m ? [Number(m[1]), Number(m[2])] : [3, 11]
}

/** "scikit-learn>=1.4.0" -> "scikit-learn". */
export function requirementName(req: string): string {
  return req.trim().split(/[<>=!~;[\s]/, 1)[0]
}

function atLeast(v: [number, number], min: [number, number]): boolean {
  return v[0] > min[0] || (v[0] === min[0] && v[1] >= min[1])
}

/** Runs a command and returns its exit code and output. */
export function run(command: string, args: string[], log?: Log, env?: NodeJS.ProcessEnv): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    const proc = spawn(command, args, { windowsHide: true, env: env ?? process.env })
    let out = ''
    const collect = (d: Buffer) => {
      out += d.toString()
      log?.(d.toString())
    }
    proc.stdout.on('data', collect)
    proc.stderr.on('data', collect)
    proc.on('error', (e) => resolve({ code: -1, out: String(e) }))
    proc.on('close', (code) => resolve({ code: code ?? -1, out }))
  })
}

/** The Python a candidate runs, or undefined when there's none there. */
export function probe(c: Candidate): Promise<Found | undefined> {
  const script = 'import sys; print(sys.executable); print(sys.version_info[0], sys.version_info[1])'
  return new Promise((resolve) => {
    execFile(c.command, [...(c.args ?? []), '-c', script], { windowsHide: true, timeout: 20_000 }, (err, stdout) => {
      if (err) return resolve(undefined)
      const [exe, ver] = stdout.trim().split(/\r?\n/)
      const [major, minor] = (ver ?? '').split(' ').map(Number)
      resolve(exe && major ? { executable: exe.trim(), version: [major, minor] } : undefined)
    })
  })
}

/** The usual places a Python lives, after the user's own choices. */
export function defaultCandidates(): Candidate[] {
  return IS_WIN
    ? [{ command: 'py', args: ['-3'], label: 'py -3' }, { command: 'python' }, { command: 'python3' }]
    : [{ command: 'python3' }, { command: 'python' }]
}

/** The first candidate that is a new enough Python. */
export async function findPython(candidates: Candidate[], min: [number, number]): Promise<Found> {
  const tried: string[] = []
  for (const c of candidates) {
    const found = await probe(c)
    const label = c.label ?? c.command
    if (!found) {
      tried.push(`${label} (not found)`)
      continue
    }
    if (atLeast(found.version, min)) return found
    tried.push(`${label} (Python ${found.version.join('.')})`)
  }
  throw new EnvironmentError(
    `Model-Maker needs Python ${min.join('.')} or newer, and found none (tried ${tried.join(', ')}). ` +
      'Install one (python.org, or your company software centre), or set modelmaker.basePython to its path.',
  )
}

/** Copies the bundled package to `dest` unless that's already this bundle;
 * returns `dest`. A stable path, unlike the extension's versioned folder. */
export async function ensureSource(bundleDir: string, dest: string): Promise<string> {
  const want = await readFile(path.join(bundleDir, 'bundle.json'), 'utf8')
  const have = await readFile(path.join(dest, 'bundle.json'), 'utf8').catch(() => undefined)
  if (have === want) return dest
  const fresh = `${dest}.new-${process.pid}`
  await rm(fresh, { recursive: true, force: true })
  await cp(bundleDir, fresh, { recursive: true })
  const old = `${dest}.old-${process.pid}-${Date.now()}`
  if (existsSync(dest)) await rename(dest, old)
  await rename(fresh, dest)
  await rm(old, { recursive: true, force: true }).catch(() => {})
  return dest
}

export function venvPython(envDir: string): string {
  return IS_WIN ? path.join(envDir, 'Scripts', 'python.exe') : path.join(envDir, 'bin', 'python')
}

export function requirementsKey(requirements: string[]): string {
  return createHash('sha256').update(JSON.stringify([...requirements].sort())).digest('hex').slice(0, 16)
}

const MARKER = 'modelmaker-env.json'

/** The managed environment's python: `envDir` made with `base` (only when
 * it's missing or broken) and given `requirements` (only when they changed
 * since the last install). One window at a time, across windows. */
export async function ensureManagedEnv(opts: {
  envDir: string
  requirements: string[]
  base: () => Promise<Found>
  log: Log
  report: Report
}): Promise<string> {
  const { envDir, requirements, log, report } = opts
  const python = venvPython(envDir)
  const key = requirementsKey(requirements)
  const release = await lock(`${envDir}.lock`, report)
  try {
    const marker = await readFile(path.join(envDir, MARKER), 'utf8').then((t) => JSON.parse(t).requirements as string, () => undefined)
    const works = existsSync(python) && (await run(python, ['-c', 'import sys']).then((r) => r.code === 0))
    if (works && marker === key) return python

    if (!works) {
      const base = await opts.base()
      report(`creating its Python environment (Python ${base.version.join('.')})`)
      log(`$ ${base.executable} -m venv ${envDir}\n`)
      await rm(envDir, { recursive: true, force: true })
      const made = await run(base.executable, ['-m', 'venv', envDir], log)
      if (made.code !== 0) {
        const hint = /ensurepip|No module named venv/.test(made.out)
          ? ' This Python has no venv support; on Debian/Ubuntu install python3-venv, or set modelmaker.basePython to another Python.'
          : ''
        throw new EnvironmentError(`couldn't create the environment (exit ${made.code}).${hint} See Model-Maker: Show Backend Log.`)
      }
    }

    const names = requirements.map(requirementName).join(', ')
    report(`installing ${names} (the first time takes a few minutes)`)
    const reqFile = path.join(envDir, 'modelmaker-requirements.txt')
    await writeFile(reqFile, requirements.join('\n') + '\n')
    log(`$ ${python} -m pip install --upgrade -r ${reqFile}\n`)
    const installed = await run(python, ['-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', '-r', reqFile], log)
    if (installed.code !== 0) {
      throw new EnvironmentError(
        `installing Model-Maker's requirements failed (exit ${installed.code}). It installs them from the package index ` +
          'pip is configured for (PyPI, or your company mirror). See Model-Maker: Show Backend Log.',
      )
    }
    await writeFile(path.join(envDir, MARKER), JSON.stringify({ requirements: key, at: new Date().toISOString() }))
    return python
  } finally {
    await release()
  }
}

/** Forgets the managed environment's install, so the next start reinstalls
 * its requirements (and recreates it, if its python no longer runs). */
export async function invalidateManagedEnv(envDir: string): Promise<void> {
  await rm(path.join(envDir, MARKER), { force: true })
}

/** The requirements `python` doesn't have installed. */
export async function missingRequirements(python: string, requirements: string[]): Promise<string[]> {
  const script = [
    'import importlib.metadata as m, json, sys',
    'missing = []',
    'for req, name in json.loads(sys.argv[1]):',
    '    try: m.version(name)',
    '    except m.PackageNotFoundError: missing.append(req)',
    'print(json.dumps(missing))',
  ].join('\n')
  const arg = JSON.stringify(requirements.map((r) => [r, requirementName(r)]))
  const { code, out } = await run(python, ['-c', script, arg])
  if (code !== 0) throw new EnvironmentError(`couldn't check ${python}'s packages: ${out.trim().split('\n').slice(-1)[0]}`)
  return JSON.parse(out.trim().split('\n').slice(-1)[0]) as string[]
}

/** A lock directory shared by every window on the machine; waits for it. */
async function lock(dir: string, report: Report): Promise<() => Promise<void>> {
  await mkdir(path.dirname(dir), { recursive: true })
  const deadline = Date.now() + LOCK_WAIT_MS
  let said = false
  for (;;) {
    try {
      await mkdir(dir)
      return () => rm(dir, { recursive: true, force: true })
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e
    }
    const age = await stat(dir).then((s) => Date.now() - s.mtimeMs, () => 0)
    if (age > LOCK_STALE_MS) {
      await rm(dir, { recursive: true, force: true })
      continue
    }
    if (Date.now() > deadline) throw new EnvironmentError(`another window has been setting up the environment for too long; remove ${dir} if none is`)
    if (!said) {
      report('waiting for another window to finish setting up the environment')
      said = true
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
}
