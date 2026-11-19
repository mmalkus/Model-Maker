import assert from 'node:assert/strict'
import * as path from 'node:path'
import { test } from 'node:test'
import { editorKind, mcpConfigPath, withServer } from '../../src/mcpConfig'

// Model-Maker: Set Up AI Agent Tools writes modelmaker-mcp into Cursor's or
// Windsurf's MCP config (their agents can't use the vscode.lm chat tools).
test('tells the editors apart by app name', () => {
  assert.equal(editorKind('Cursor'), 'cursor')
  assert.equal(editorKind('Windsurf'), 'windsurf')
  assert.equal(editorKind('Visual Studio Code'), 'vscode')
  assert.equal(editorKind('VSCodium'), 'vscode')
})

test("finds each editor's MCP config file", () => {
  const dirs = { home: '/home/u', workspace: '/w' }
  assert.equal(mcpConfigPath('cursor', 'user', dirs), path.join('/home/u', '.cursor', 'mcp.json'))
  assert.equal(mcpConfigPath('cursor', 'workspace', dirs), path.join('/w', '.cursor', 'mcp.json'))
  assert.equal(mcpConfigPath('cursor', 'workspace', { home: '/home/u' }), undefined)
  assert.equal(mcpConfigPath('windsurf', 'user', dirs), path.join('/home/u', '.codeium', 'windsurf', 'mcp_config.json'))
  assert.equal(mcpConfigPath('windsurf', 'workspace', dirs), undefined)
})

test('adds or replaces the modelmaker server and keeps the rest', () => {
  const server = { command: '/py', args: ['-m', 'modelmaker.assistant_mcp'], env: { MODELMAKER_EDITOR_URI_SCHEME: 'cursor' } }
  assert.deepEqual(JSON.parse(withServer(undefined, 'modelmaker', server)), { mcpServers: { modelmaker: server } })
  assert.deepEqual(JSON.parse(withServer('  \n', 'modelmaker', server)), { mcpServers: { modelmaker: server } })

  const existing = JSON.stringify({ other: 1, mcpServers: { git: { command: 'git-mcp' }, modelmaker: { command: '/old' } } })
  assert.deepEqual(JSON.parse(withServer(existing, 'modelmaker', server)), {
    other: 1,
    mcpServers: { git: { command: 'git-mcp' }, modelmaker: server },
  })

  assert.throws(() => withServer('{ not json', 'modelmaker', server), SyntaxError)
  assert.throws(() => withServer('[]', 'modelmaker', server), SyntaxError)
})
