import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { test } from 'node:test'
import { Discovery } from '../../src/discovery'

// The files modelmaker-mcp (modelmaker/assistant_mcp.py) reads to find the
// open canvases: one per backend, rewritten on restart, gone on close.
test('publishes, touches and withdraws a canvas', () => {
  const dir = path.join(mkdtempSync(path.join(tmpdir(), 'mm-disc-')), 'vscode-backends')
  let enabled = true
  const d = new Discovery(() => enabled, dir)

  d.publish('/p/a', 'http://127.0.0.1:41001', 'tok-a', true)
  d.publish('/p/b', 'http://127.0.0.1:41002', 'tok-b', false)
  const files = readdirSync(dir)
  assert.equal(files.length, 2)
  const a = JSON.parse(readFileSync(path.join(dir, `${process.pid}-41001.json`), 'utf8'))
  assert.deepEqual([a.folder, a.url, a.token], ['/p/a', 'http://127.0.0.1:41001', 'tok-a'])
  assert.ok(a.active_at > 0)
  if (process.platform !== 'win32') assert.equal(statSync(path.join(dir, files[0])).mode & 0o077, 0)

  d.touch('/p/b')
  const b = JSON.parse(readFileSync(path.join(dir, `${process.pid}-41002.json`), 'utf8'))
  assert.ok(b.active_at >= a.active_at)

  // A restart moves the backend to a new port: one file per folder still.
  d.publish('/p/a', 'http://127.0.0.1:41003', 'tok-a2', false)
  assert.ok(!existsSync(path.join(dir, `${process.pid}-41001.json`)))
  assert.equal(JSON.parse(readFileSync(path.join(dir, `${process.pid}-41003.json`), 'utf8')).active_at, a.active_at)

  d.withdraw('/p/b')
  assert.deepEqual(readdirSync(dir), [`${process.pid}-41003.json`])
  enabled = false
  d.publish('/p/a', 'http://127.0.0.1:41003', 'tok-a2', true)
  assert.deepEqual(readdirSync(dir), [])
  d.dispose()
})
