// Just enough of the vscode module for tests/unit to load src/tools.ts and
// src/backend.ts in plain Node: registered tools are captured in `tools`.
export const tools = new Map<string, { invoke: Function; prepareInvocation?: Function }>()

export class Disposable {
  constructor(private readonly fn: () => void = () => {}) {}
  dispose() {
    this.fn()
  }
}

export class EventEmitter<T> {
  private listeners: Array<(e: T) => void> = []
  event = (l: (e: T) => void) => {
    this.listeners.push(l)
    return new Disposable()
  }
  fire(e: T) {
    for (const l of this.listeners) l(e)
  }
  dispose() {
    this.listeners = []
  }
}

export class LanguageModelTextPart {
  constructor(readonly value: string) {}
}

export class LanguageModelToolResult {
  constructor(readonly content: unknown[]) {}
}

export const lm = {
  registerTool(name: string, tool: { invoke: Function }) {
    tools.set(name, tool)
    return new Disposable()
  },
}

// Settings the tests set, read through getConfiguration('modelmaker').
export const settings = new Map<string, unknown>()

export class RelativePattern {
  constructor(
    readonly base: unknown,
    readonly pattern: string,
  ) {}
}

export const workspace = {
  asRelativePath: (p: string) => p,
  // For src/editor.ts: a watcher that never fires, and the canvas bundle's
  // index.html.
  createFileSystemWatcher: () => ({ onDidChange: () => new Disposable(), dispose: () => {} }),
  fs: { readFile: async () => Buffer.from('<html><head></head><body></body></html>') },
  getConfiguration: (section?: string) => ({
    get: <T>(key: string, fallback?: T): T | undefined => {
      const full = section ? `${section}.${key}` : key
      return (settings.has(full) ? settings.get(full) : fallback) as T | undefined
    },
    update: async (key: string, value: unknown) => {
      settings.set(section ? `${section}.${key}` : key, value)
    },
  }),
}

export const extensions = { getExtension: () => undefined }

export const ConfigurationTarget = { Global: 1, Workspace: 2 }

// What showInformationMessage answers (the tests set it).
export const answers: Array<string | undefined> = []

export const window = {
  showInformationMessage: async () => answers.shift(),
}

export const Uri = {
  file: (fsPath: string) => ({ fsPath }),
  joinPath: (base: { fsPath: string }, ...parts: string[]) => ({ fsPath: [base.fsPath, ...parts].join('/') }),
}
