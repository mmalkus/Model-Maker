import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { test } from 'node:test'
import { PythonEnvironments } from '../../src/environment'
import { settings } from '../stub/vscode'

// The glue that picks the Python a backend runs on, with a real Python
// (MODELMAKER_TEST_PYTHON, which has Model-Maker's requirements in CI) and
// the bundle scripts/bundle-python.py puts in python/ (skipped without it).
const python = process.env.MODELMAKER_TEST_PYTHON
const extensionPath = path.resolve(__dirname, '../..')
const bundled = existsSync(path.join(extensionPath, 'python', 'bundle.json'))
const opts = { skip: python ? (bundled ? false : 'run scripts/bundle-python.py first') : 'set MODELMAKER_TEST_PYTHON to run' }
const log = { append: () => {}, appendLine: () => {} }
const folder = { fsPath: '/nowhere' } as never

function environments(ext: string) {
  const storage = mkdtempSync(path.join(tmpdir(), 'mm-env-'))
  const context = { extensionPath: ext, globalStorageUri: { fsPath: storage } }
  return { envs: new PythonEnvironments(context as never, log as never), storage }
}

test('interpreter: your Python, with the bundled Model-Maker', opts, async () => {
  settings.clear()
  settings.set('modelmaker.python.environment', 'interpreter')
  settings.set('modelmaker.pythonPath', python)
  const { envs, storage } = environments(extensionPath)
  const runtime = await envs.runtime(folder)
  assert.equal(runtime.python, python)
  assert.equal(runtime.env.PYTHONPATH.split(path.delimiter)[0], path.join(storage, 'src'))

  // A backend from it reports the bundle's version, loaded from the copy.
  const port = 19000 + Math.floor(Math.random() * 1000)
  const cwd = mkdtempSync(path.join(tmpdir(), 'mm-proj-'))
  const proc = spawn(runtime.python, ['-c', 'from modelmaker.api import main; main()'], {
    cwd, env: { ...process.env, ...runtime.env, MODELMAKER_PORT: String(port), MODELMAKER_CACHE_DIR: path.join(cwd, '.c') }, stdio: 'ignore',
  })
  try {
    let version: { version?: string } = {}
    for (let i = 0; i < 120 && !version.version; i++) {
      version = (await fetch(`http://127.0.0.1:${port}/api/version`).then((r) => r.json(), () => ({}))) as { version?: string }
      if (!version.version) await new Promise((r) => setTimeout(r, 250))
    }
    assert.equal(version.version, runtime.version)
  } finally {
    proc.kill()
  }
})

test('interpreter: missing requirements are offered, not silently installed', opts, async () => {
  settings.clear()
  settings.set('modelmaker.python.environment', 'interpreter')
  settings.set('modelmaker.pythonPath', python)
  settings.set('modelmaker.autoInstall', 'never')
  const ext = mkdtempSync(path.join(tmpdir(), 'mm-ext-'))
  mkdirSync(path.join(ext, 'python'))
  writeFileSync(path.join(ext, 'python', 'bundle.json'), JSON.stringify({ version: '9', requires_python: '>=3.11', requirements: ['surely-not-a-real-package-mm>=1'] }))
  await assert.rejects(environments(ext).envs.runtime(folder), /doesn't have: surely-not-a-real-package-mm>=1.*Install them yourself/)
})

test('managed: its own environment, with only the requirements', opts, async () => {
  settings.clear()
  settings.set('modelmaker.basePython', python)
  const ext = mkdtempSync(path.join(tmpdir(), 'mm-ext-'))
  mkdirSync(path.join(ext, 'python'))
  writeFileSync(path.join(ext, 'python', 'bundle.json'), JSON.stringify({ version: '9', requires_python: '>=3.11', requirements: ['python-dotenv>=1.0.0'] }))
  const { envs, storage } = environments(ext)
  const runtime = await envs.runtime(folder)
  assert.equal(runtime.mode, 'managed')
  assert.ok(runtime.python.startsWith(path.join(storage, 'env')))
})

test('installedPackage, or a build without a bundle: the installed package', async () => {
  settings.clear()
  settings.set('modelmaker.pythonPath', '/some/python')
  const runtime = await environments(mkdtempSync(path.join(tmpdir(), 'mm-ext-'))).envs.runtime(folder)
  assert.deepEqual(runtime, { python: '/some/python', env: {}, mode: 'installedPackage' })
})
