import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { test } from 'node:test'
import {
  ensureManagedEnv,
  ensureSource,
  findPython,
  invalidateManagedEnv,
  minimumPython,
  missingRequirements,
  probe,
  requirementName,
  run,
  venvPython,
} from '../../src/pyenv'

// The managed environment and the bundled source, with a real Python
// (MODELMAKER_TEST_PYTHON, which CI sets; skipped without it). The
// environment test installs one small package from the configured index.
const python = process.env.MODELMAKER_TEST_PYTHON
const opts = { skip: python ? false : 'set MODELMAKER_TEST_PYTHON to run' }
const tmp = () => mkdtempSync(path.join(tmpdir(), 'mm-pyenv-'))

test('requirement names and the minimum Python', () => {
  assert.equal(requirementName('scikit-learn>=1.4.0'), 'scikit-learn')
  assert.equal(requirementName('mcp>=1.0.0,<2'), 'mcp')
  assert.equal(requirementName('uvicorn[standard] >=0.29'), 'uvicorn')
  assert.deepEqual(minimumPython('>=3.11'), [3, 11])
  assert.deepEqual(minimumPython(''), [3, 11])
})

test('finds a new enough Python, and says what it tried', opts, async () => {
  const found = await findPython([{ command: 'no-such-python-here' }, { command: python! }], [3, 11])
  assert.ok(existsSync(found.executable) && found.version[0] === 3)
  await assert.rejects(findPython([{ command: python! }], [9, 0]), /needs Python 9\.0 or newer.*\(Python 3\.\d+\)/)
  await assert.rejects(findPython([{ command: 'nope', label: 'nope' }], [3, 11]), /nope \(not found\)/)
  assert.equal(await probe({ command: 'no-such-python-here' }), undefined)
})

test('copies the bundle to a stable folder once per version', async () => {
  const root = tmp()
  const bundle = path.join(root, 'bundle')
  mkdirSync(path.join(bundle, 'modelmaker'), { recursive: true })
  writeFileSync(path.join(bundle, 'bundle.json'), '{"version":"1"}')
  writeFileSync(path.join(bundle, 'modelmaker', '__init__.py'), 'V = 1\n')
  const dest = path.join(root, 'src')
  await ensureSource(bundle, dest)
  assert.equal(readFileSync(path.join(dest, 'modelmaker', '__init__.py'), 'utf8'), 'V = 1\n')

  // Same bundle: left alone (even if touched).
  writeFileSync(path.join(dest, 'marker'), 'kept')
  await ensureSource(bundle, dest)
  assert.ok(existsSync(path.join(dest, 'marker')))

  // A new version replaces it whole.
  writeFileSync(path.join(bundle, 'bundle.json'), '{"version":"2"}')
  writeFileSync(path.join(bundle, 'modelmaker', '__init__.py'), 'V = 2\n')
  await ensureSource(bundle, dest)
  assert.equal(readFileSync(path.join(dest, 'modelmaker', '__init__.py'), 'utf8'), 'V = 2\n')
  assert.ok(!existsSync(path.join(dest, 'marker')))
})

test('the managed environment: made once, reinstalled only when asked', opts, async () => {
  const envDir = path.join(tmp(), 'env')
  const requirements = ['python-dotenv>=1.0.0']
  let bases = 0
  const logs: string[] = []
  const make = () =>
    ensureManagedEnv({
      envDir,
      requirements,
      base: async () => {
        bases++
        return findPython([{ command: python! }], [3, 11])
      },
      log: (t) => logs.push(t),
      report: () => {},
    })

  // Two at once (two windows): one environment, made once.
  const [a, b] = await Promise.all([make(), make()])
  assert.equal(a, venvPython(envDir))
  assert.equal(a, b)
  assert.equal(bases, 1)
  const check = await run(a, ['-c', 'import dotenv, sys; print(sys.prefix)'])
  assert.equal(check.code, 0, check.out)

  // Nothing to do the next time.
  const installs = () => logs.filter((t) => t.includes('pip install')).length
  const before = installs()
  await make()
  assert.equal(installs(), before)

  // Repair: reinstalls the requirements, keeps the environment.
  await invalidateManagedEnv(envDir)
  await make()
  assert.equal(installs(), before + 1)
  assert.equal(bases, 1)
  assert.deepEqual(await missingRequirements(a, requirements), [])
})

test('lists the requirements an interpreter lacks', opts, async () => {
  const missing = await missingRequirements(python!, ['pip', 'surely-not-a-real-package-mm>=1'])
  assert.deepEqual(missing, ['surely-not-a-real-package-mm>=1'])
})
