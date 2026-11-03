import assert from 'node:assert/strict'
import { existsSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { test } from 'node:test'
import { ManagedEnvironment, packageVersion } from '../../src/managed'

// The real thing: downloads uv from PyPI, makes the environment, installs
// the package from MODELMAKER_TEST_MANAGED_SPEC (CI: the checkout itself,
// as the version under test isn't on PyPI yet). Skipped without it.
const spec = process.env.MODELMAKER_TEST_MANAGED_SPEC
const opts = { skip: spec ? false : 'set MODELMAKER_TEST_MANAGED_SPEC to run (downloads uv and packages)', timeout: 15 * 60_000 }

test('sets up, reuses, survives concurrent setup, and upgrades', opts, async () => {
  const storage = mkdtempSync(path.join(tmpdir(), 'mm-managed-'))
  const steps: string[] = []
  const env = new ManagedEnvironment(storage, () => {})
  const report = (m: string) => steps.push(m)

  // Two windows opening projects at once: one sets up, the other waits.
  const [a, b] = await Promise.all([env.ensure(spec!, report), new ManagedEnvironment(storage, () => {}).ensure(spec!, () => {})])
  assert.equal(a, b)
  assert.ok(existsSync(a))
  assert.ok(await packageVersion(a))
  assert.ok(steps.some((s) => s.startsWith('downloading uv')), steps.join(' | '))
  assert.ok(steps.some((s) => s.startsWith('installing quantology-modelmaker')), steps.join(' | '))

  // Later starts are instant: nothing to download or install.
  steps.length = 0
  const t = Date.now()
  await env.ensure(spec!, report)
  assert.equal(steps.length, 0, steps.join(' | '))
  assert.ok(Date.now() - t < 5000)

  await env.upgrade(spec!, report)
  assert.ok(steps.some((s) => s.startsWith('upgrading')))

  env.reset()
  assert.ok(!existsSync(a))
})
