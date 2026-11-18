import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import * as path from 'node:path'

// How `modelmaker-mcp` (modelmaker/assistant_mcp.py) finds the canvases open
// in VS Code: one small file per open project's backend, with its URL and
// token, and when its canvas was last used -- the MCP tools act on the most
// recent, like the Copilot Chat tools do. Readable by this user only; the
// backend listens on 127.0.0.1 anyway. Removed when the canvas closes, and
// the MCP server skips any whose backend doesn't answer (a crashed window).

export const DISCOVERY_DIR = path.join(homedir(), '.modelmaker', 'vscode-backends')

export type DiscoveryEntry = { folder: string; url: string; token: string; pid: number; active_at: number }

export class Discovery {
  // folder -> its file and what's in it
  private readonly entries = new Map<string, { file: string; entry: DiscoveryEntry }>()

  constructor(
    private readonly enabled: () => boolean,
    private readonly dir = DISCOVERY_DIR,
  ) {}

  /** Records (or, after a restart, replaces) a folder's backend. */
  publish(folder: string, url: string, token: string, active: boolean) {
    if (!this.enabled()) return this.withdraw(folder)
    const file = path.join(this.dir, `${process.pid}-${new URL(url).port}.json`)
    const previous = this.entries.get(folder)
    if (previous && previous.file !== file) rmSync(previous.file, { force: true })
    this.write(folder, file, { folder, url, token, pid: process.pid, active_at: active ? Date.now() : (previous?.entry.active_at ?? 0) })
  }

  /** The folder's canvas is the one in use now. */
  touch(folder: string) {
    const known = this.entries.get(folder)
    if (known && this.enabled()) this.write(folder, known.file, { ...known.entry, active_at: Date.now() })
  }

  withdraw(folder: string) {
    const known = this.entries.get(folder)
    if (known) rmSync(known.file, { force: true })
    this.entries.delete(folder)
  }

  dispose() {
    for (const { file } of this.entries.values()) rmSync(file, { force: true })
    this.entries.clear()
  }

  private write(folder: string, file: string, entry: DiscoveryEntry) {
    try {
      mkdirSync(this.dir, { recursive: true, mode: 0o700 })
      writeFileSync(file, JSON.stringify(entry), { mode: 0o600 })
      this.entries.set(folder, { file, entry })
    } catch {
      // Not fatal: only the MCP tools need it.
    }
  }
}
