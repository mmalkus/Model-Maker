// Where Cursor and Windsurf read their MCP servers, and how to add
// Model-Maker's (modelmaker-mcp) to that file without touching the others.
// Their agents can't use the chat tools this extension registers with
// vscode.lm (only Copilot does), but they speak MCP. Kept free of the vscode
// module so it's unit-testable.
import * as path from 'node:path'

export type McpClient = 'cursor' | 'windsurf'
export type McpScope = 'user' | 'workspace'
export type McpServer = { command: string; args: string[]; env?: Record<string, string> }

export const MCP_CLIENTS: Record<McpClient, { name: string; uriScheme: string; scopes: McpScope[]; reload: string }> = {
  cursor: {
    name: 'Cursor',
    uriScheme: 'cursor',
    scopes: ['user', 'workspace'],
    reload: 'In Cursor Settings → MCP, check that "modelmaker" is listed and switched on, then start a new agent chat.',
  },
  windsurf: {
    name: 'Windsurf',
    uriScheme: 'windsurf',
    // Windsurf reads only its user-level file.
    scopes: ['user'],
    reload: 'In Cascade, open the MCP servers panel and refresh it: "modelmaker" should be listed.',
  },
}

/** Which editor this is, from vscode.env.appName ("Cursor", "Windsurf", "Visual Studio Code"). */
export function editorKind(appName: string): McpClient | 'vscode' {
  if (/cursor/i.test(appName)) return 'cursor'
  if (/windsurf/i.test(appName)) return 'windsurf'
  return 'vscode'
}

/** The client's MCP config file for that scope (workspace needs a folder). */
export function mcpConfigPath(client: McpClient, scope: McpScope, dirs: { home: string; workspace?: string }): string | undefined {
  if (!MCP_CLIENTS[client].scopes.includes(scope)) return undefined
  if (client === 'windsurf') return path.join(dirs.home, '.codeium', 'windsurf', 'mcp_config.json')
  if (scope === 'user') return path.join(dirs.home, '.cursor', 'mcp.json')
  return dirs.workspace ? path.join(dirs.workspace, '.cursor', 'mcp.json') : undefined
}

/** The config file's text with `name` set to `server` under mcpServers, the
 * rest as it was. Throws a SyntaxError when the existing text isn't JSON. */
export function withServer(existing: string | undefined, name: string, server: McpServer): string {
  const config = existing?.trim() ? JSON.parse(existing) : {}
  if (typeof config !== 'object' || config === null || Array.isArray(config)) throw new SyntaxError('the file is not a JSON object')
  const servers = typeof config.mcpServers === 'object' && config.mcpServers !== null ? config.mcpServers : {}
  config.mcpServers = { ...servers, [name]: server }
  return JSON.stringify(config, null, 2) + '\n'
}
