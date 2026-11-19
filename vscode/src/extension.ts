import { spawn } from 'node:child_process'
import { homedir } from 'node:os'
import * as path from 'node:path'
import * as vscode from 'vscode'
import { BackendManager, installPackage } from './backend'
import { CopilotRelay } from './copilot'
import { Discovery, DISCOVERY_DIR } from './discovery'
import { PythonEnvironments } from './environment'
import { LibraryPanels } from './library'
import { editorKind, MCP_CLIENTS, mcpConfigPath, type McpClient, type McpScope, type McpServer, withServer } from './mcpConfig'
import { ModelEditorProvider, VIEW_TYPE } from './editor'
import { onPythonChanged, resolvePython } from './python'
import { registerTools } from './tools'

const PROJECT_FILE = 'model.json'
// Present in every model.json Model-Maker writes (modelmaker/project.py),
// and how this tells one apart from unrelated files of the same name.
const FORMAT_MARKER = '"modelmaker_version"'
const HINT_DISMISSED_KEY = 'modelmaker.projectHintDismissed'

/** What activate() returns: for the integration tests, not a public API. */
export type ExtensionApi = { backends: BackendManager }

export function activate(context: vscode.ExtensionContext): ExtensionApi {
  const log = vscode.window.createOutputChannel('Model-Maker')
  const relay = new CopilotRelay(log)
  const environments = new PythonEnvironments(context, log)
  const backends = new BackendManager(log, relay, environments)
  // For Claude Code and other MCP clients (modelmaker-mcp): see discovery.ts.
  const discovery = new Discovery(() => vscode.workspace.getConfiguration('modelmaker').get<boolean>('mcp.discovery', true), DISCOVERY_DIR, vscode.env.uriScheme)
  context.subscriptions.push(log, relay, backends, discovery)
  const editor = ModelEditorProvider.register(context, backends, log, discovery)
  const libraries = new LibraryPanels(context, backends, editor, log)
  context.subscriptions.push(libraries)
  registerTools(context, backends, editor, (folder, create) => openProjectFolder(backends, folder, create))

  const command = (id: string, fn: (...args: never[]) => unknown) =>
    context.subscriptions.push(vscode.commands.registerCommand(id, fn))

  command('modelmaker.openCanvas', (uri?: vscode.Uri) => {
    const target = uri ?? vscode.window.activeTextEditor?.document.uri
    if (target) return openCanvas(target)
  })
  command('modelmaker.openProject', async () => {
    const picked = await vscode.window.showOpenDialog({
      canSelectFolders: true,
      canSelectFiles: false,
      title: 'Open a Model-Maker project folder',
      openLabel: 'Open Project',
    })
    if (!picked?.[0]) return
    const file = vscode.Uri.joinPath(picked[0], PROJECT_FILE)
    if (!(await exists(file))) return vscode.window.showErrorMessage(`${picked[0].fsPath} isn't a Model-Maker project (no ${PROJECT_FILE}).`)
    await openCanvas(file)
  })
  command('modelmaker.newProject', () => newProject(backends))
  command('modelmaker.setUpClaudeCode', () => setUpClaudeCode(environments, log))
  command('modelmaker.setUpMcp', () => setUpMcp(environments, log))
  command('modelmaker.repairEnvironment', async () => {
    await environments.repair()
    await backends.restart()
    vscode.window.showInformationMessage('Model-Maker: its Python environment is reinstalled the next time a project opens (now, for open ones).')
  })

  // vscode://quantology.modelmaker/open?folder=<path>[&create=1]: how
  // modelmaker-mcp's open_project asks for a canvas (see assistant_mcp.py).
  context.subscriptions.push(
    vscode.window.registerUriHandler({
      handleUri: async (uri) => {
        if (uri.path !== '/open') return
        const params = new URLSearchParams(uri.query)
        const folder = params.get('folder')
        if (!folder || !path.isAbsolute(folder)) return
        const create = params.get('create') === '1'
        if (create && !(await exists(vscode.Uri.joinPath(vscode.Uri.file(folder), PROJECT_FILE)))) {
          const ok = await vscode.window.showInformationMessage(`An assistant asks to create a Model-Maker project in ${folder}.`, { modal: true }, 'Create')
          if (ok !== 'Create') return
        }
        try {
          await openProjectFolder(backends, folder, create)
        } catch (e) {
          vscode.window.showErrorMessage(`Model-Maker: ${e instanceof Error ? e.message : e}`)
        }
      },
    }),
  )
  command('modelmaker.save', () => editor.saveActive())
  command('modelmaker.openLibrary', () => {
    const folder = editor.activeProject()
    if (!folder) return vscode.window.showInformationMessage('Open a project\'s canvas first: the library opens beside it, on its backend.')
    return libraries.open(folder)
  })
  command('modelmaker.restartBackend', () => backends.restart())
  command('modelmaker.showLog', () => log.show())
  command('modelmaker.selectInterpreter', selectInterpreter)
  command('modelmaker.installPackage', async () => {
    const folder = vscode.workspace.workspaceFolders?.[0]?.uri ?? vscode.Uri.file(process.cwd())
    if (environments.mode(folder) !== 'installedPackage') {
      return vscode.window.showInformationMessage(
        'Model-Maker ships with the extension and needs no package install. To redo its Python setup, run Model-Maker: Repair Python Environment.',
      )
    }
    const python = await resolvePython(folder)
    try {
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'Model-Maker: installing the Python package' },
        (progress) => installPackage(python, log, progress),
      )
      const choice = await vscode.window.showInformationMessage(`Installed into ${python}.`, 'Restart Backend')
      if (choice) await backends.restart()
    } catch (e) {
      vscode.window.showErrorMessage(String(e instanceof Error ? e.message : e))
    }
  })

  // A different interpreter means a different package: restart onto it --
  // only the backends that run on it. The managed environment doesn't, and
  // the Python extension reports an interpreter as it starts up, which
  // restarted backends that had only just started (racing their first
  // render, and leaving Claude Code pointed at the stopped one).
  onPythonChanged(() => backends.restart(undefined, (f) => environments.mode(vscode.Uri.file(f)) !== 'managed')).then((d) =>
    context.subscriptions.push(d),
  )

  // Settings a running backend was started with: offer to restart onto them.
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (e) => {
      const keys = ['modelmaker.pythonPath', 'modelmaker.python.environment', 'modelmaker.basePython', 'modelmaker.llm.useCopilot', 'modelmaker.backend.extraEnv']
      if (!keys.some((k) => e.affectsConfiguration(k))) return
      const choice = await vscode.window.showInformationMessage('Restart the Model-Maker backend to apply the new settings?', 'Restart')
      if (choice) await backends.restart()
    }),
  )

  offerCanvas(context)
  return { backends }
}

export function deactivate() {
  // Disposing the subscriptions (above) stops every backend.
}

async function openCanvas(uri: vscode.Uri) {
  await vscode.commands.executeCommand('vscode.openWith', uri, VIEW_TYPE)
}

async function exists(uri: vscode.Uri): Promise<boolean> {
  try {
    await vscode.workspace.fs.stat(uri)
    return true
  } catch {
    return false
  }
}

async function isModelMakerProject(uri: vscode.Uri): Promise<boolean> {
  try {
    const head = Buffer.from(await vscode.workspace.fs.readFile(uri)).toString('utf8', 0, 4096)
    return head.includes(FORMAT_MARKER)
  } catch {
    return false
  }
}

/** The canvas is an "option" editor for model.json (plenty of unrelated
 * repos have one), so say once per workspace that it's there. */
async function offerCanvas(context: vscode.ExtensionContext) {
  if (context.workspaceState.get(HINT_DISMISSED_KEY)) return
  const files = await vscode.workspace.findFiles(`**/${PROJECT_FILE}`, '**/node_modules/**', 20)
  const projects: vscode.Uri[] = []
  for (const f of files) if (await isModelMakerProject(f)) projects.push(f)
  if (projects.length === 0) return
  const label = projects.length === 1 ? path.basename(path.dirname(projects[0].fsPath)) : `${projects.length} projects`
  const choice = await vscode.window.showInformationMessage(
    `This workspace has a Model-Maker project (${label}). Open the canvas?`,
    'Open',
    "Don't Ask Again",
  )
  if (choice === 'Open') {
    const target =
      projects.length === 1
        ? projects[0]
        : (
            await vscode.window.showQuickPick(
              projects.map((uri) => ({ label: vscode.workspace.asRelativePath(path.dirname(uri.fsPath)), uri })),
              { title: 'Open which project?' },
            )
          )?.uri
    if (target) await openCanvas(target)
  } else if (choice === "Don't Ask Again") {
    await context.workspaceState.update(HINT_DISMISSED_KEY, true)
  }
}

async function newProject(backends: BackendManager) {
  const parent = vscode.workspace.workspaceFolders?.[0]?.uri
  const target = await vscode.window.showSaveDialog({
    defaultUri: parent ? vscode.Uri.joinPath(parent, 'my_model') : undefined,
    title: 'New Model-Maker project: choose a folder name',
    saveLabel: 'Create Project',
  })
  if (!target) return
  if (await exists(vscode.Uri.joinPath(target, PROJECT_FILE))) {
    return vscode.window.showErrorMessage(`${target.fsPath} already has a project; open it with Model-Maker: Open Project.`)
  }
  try {
    await createProject(backends, target)
    await openCanvas(vscode.Uri.joinPath(target, PROJECT_FILE))
  } catch (e) {
    vscode.window.showErrorMessage(`Couldn't create the project: ${e instanceof Error ? e.message : e}`)
  }
}

/** Writes a new, empty project into `target` (model.json, files/,
 * .gitignore, git init) exactly as the browser app's Save does. */
async function createProject(backends: BackendManager, target: vscode.Uri) {
  const folder = target.fsPath
  await vscode.workspace.fs.createDirectory(target)
  const backend = await backends.acquire(folder)
  try {
    await backend.json('/project/new', { method: 'POST' })
    await backend.json('/project/save', { method: 'POST', body: { path: folder } })
  } finally {
    backends.release(folder)
  }
}

/** The chat tools' modelmaker_open_project: a folder relative to the
 * (first) workspace folder, or absolute; with `create`, a missing project
 * is created first. Opens its canvas and returns the folder's path. */
async function openProjectFolder(backends: BackendManager, folder: string, create: boolean): Promise<string> {
  const root = vscode.workspace.workspaceFolders?.[0]?.uri
  const target = path.isAbsolute(folder) ? vscode.Uri.file(folder) : root ? vscode.Uri.joinPath(root, folder) : undefined
  if (!target) throw new Error('open a workspace folder first, or give an absolute path')
  const file = vscode.Uri.joinPath(target, PROJECT_FILE)
  if (!(await exists(file))) {
    if (!create) throw new Error(`${folder} isn't a Model-Maker project (no ${PROJECT_FILE}); pass create: true to make one there`)
    await createProject(backends, target)
  }
  await openCanvas(file)
  return target.fsPath
}

/** Registers the Model-Maker tools (modelmaker-mcp) with Claude Code, on
 * the same Python and Model-Maker the canvas runs on (see environment.ts):
 * for all the user's projects through the `claude` CLI (`claude mcp add
 * --scope user`), or in the workspace's .mcp.json. The tools act on the
 * canvas used last. */
async function setUpClaudeCode(environments: PythonEnvironments, log: vscode.OutputChannel) {
  const root = vscode.workspace.workspaceFolders?.[0]?.uri
  const scope = await vscode.window.showQuickPick(
    [
      { label: 'All my projects', description: 'claude mcp add --scope user', value: 'user' as const },
      ...(root ? [{ label: 'This workspace only', description: '.mcp.json in the workspace folder', value: 'project' as const }] : []),
    ],
    { title: 'Set up Claude Code: where should it find the Model-Maker tools?' },
  )
  if (!scope) return
  // Opening a canvas from Claude Code goes to this editor (Cursor's
  // cursor://, ...), not always VS Code.
  const server = await mcpServer(environments, vscode.env.uriScheme === 'vscode' ? undefined : vscode.env.uriScheme)
  if (!server) return
  const { command: python, args, env } = server
  const done = 'Start a new Claude Code conversation (or restart it) and check /mcp: "modelmaker" should be listed. It works on the canvas you used last.'

  if (scope.value === 'user') {
    // The name before -e, which takes several values and would swallow it.
    const cli = ['claude', 'mcp', 'add', '--scope', 'user', 'modelmaker', ...Object.entries(env ?? {}).flatMap(([k, v]) => ['-e', `${k}=${v}`]), '--', python, ...args]
    // Replaces an earlier registration (e.g. with an older Python path).
    await runQuiet('claude', ['mcp', 'remove', '--scope', 'user', 'modelmaker'])
    const { code, out } = await runQuiet(cli[0], cli.slice(1))
    log.appendLine(`[claude] $ ${cli.join(' ')}\n${out}`)
    if (code === 0) return vscode.window.showInformationMessage(`Added Model-Maker to Claude Code for all your projects. ${done}`)
    const command = cli.map((a) => (/[\s"]/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a)).join(' ')
    const choice = await vscode.window.showWarningMessage(
      "Couldn't run the claude command (is Claude Code's CLI on PATH?). Run this in a terminal instead:\n" + command,
      { modal: true },
      'Copy Command',
    )
    if (choice) await vscode.env.clipboard.writeText(command)
    return
  }

  const file = vscode.Uri.joinPath(root!, '.mcp.json')
  let config: { mcpServers?: Record<string, unknown> } = {}
  try {
    config = JSON.parse(Buffer.from(await vscode.workspace.fs.readFile(file)).toString('utf8'))
  } catch (e) {
    if (await exists(file)) return vscode.window.showErrorMessage(`.mcp.json isn't valid JSON (${e instanceof Error ? e.message : e}); fix it first.`)
  }
  config.mcpServers = { ...config.mcpServers, modelmaker: server }
  await vscode.workspace.fs.writeFile(file, Buffer.from(JSON.stringify(config, null, 2) + '\n'))
  const choice = await vscode.window.showInformationMessage(
    `Added Model-Maker to ${vscode.workspace.asRelativePath(file)}. Open Claude Code in this folder and approve the "modelmaker" server. ${done}`,
    'Open .mcp.json',
  )
  if (choice) await vscode.window.showTextDocument(file)
}

/** How an MCP client starts modelmaker-mcp: the same Python and Model-Maker
 * the canvas runs on (see environment.ts). `uriScheme` is the editor its
 * open_project should open canvases in, when not VS Code. */
async function mcpServer(environments: PythonEnvironments, uriScheme?: string): Promise<McpServer | undefined> {
  const root = vscode.workspace.workspaceFolders?.[0]?.uri
  let runtime
  try {
    runtime = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: 'Model-Maker: preparing its Python environment' },
      (progress) => environments.runtime(root ?? vscode.Uri.file(process.cwd()), progress),
    )
  } catch (e) {
    vscode.window.showErrorMessage(`Model-Maker: ${e instanceof Error ? e.message : e}`)
    return undefined
  }
  const env: Record<string, string> = {
    ...(runtime.env.PYTHONPATH ? { PYTHONPATH: runtime.env.PYTHONPATH } : {}),
    ...(uriScheme ? { MODELMAKER_EDITOR_URI_SCHEME: uriScheme } : {}),
  }
  return { command: runtime.python, args: ['-m', 'modelmaker.assistant_mcp'], ...(Object.keys(env).length ? { env } : {}) }
}

/** Registers modelmaker-mcp with an agent that speaks MCP but not VS Code's
 * chat tools API: Cursor's or Windsurf's (Claude Code has its own command).
 * This editor's comes first in the list. */
async function setUpMcp(environments: PythonEnvironments, log: vscode.OutputChannel) {
  const here = editorKind(vscode.env.appName)
  const workspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
  type Item = vscode.QuickPickItem & { client?: McpClient; scope?: McpScope; file?: string }
  const items: Item[] = []
  for (const client of Object.keys(MCP_CLIENTS) as McpClient[]) {
    for (const scope of MCP_CLIENTS[client].scopes) {
      const file = mcpConfigPath(client, scope, { home: homedir(), workspace })
      if (!file) continue
      const where = scope === 'user' ? 'all my projects' : 'this workspace only'
      items.push({ label: `${MCP_CLIENTS[client].name}: ${where}`, description: file, client, scope, file })
    }
  }
  items.sort((a, b) => Number(b.client === here) - Number(a.client === here))
  items.push({ label: 'Claude Code', description: 'Model-Maker: Set Up Claude Code' })
  const picked = await vscode.window.showQuickPick(items, { title: 'Set up Model-Maker tools for which AI agent?' })
  if (!picked) return
  if (!picked.client || !picked.file) return setUpClaudeCode(environments, log)

  const info = MCP_CLIENTS[picked.client]
  const server = await mcpServer(environments, info.uriScheme)
  if (!server) return
  const file = vscode.Uri.file(picked.file)
  let text
  try {
    const existing = (await exists(file)) ? Buffer.from(await vscode.workspace.fs.readFile(file)).toString('utf8') : undefined
    text = withServer(existing, 'modelmaker', server)
  } catch (e) {
    return vscode.window.showErrorMessage(`${picked.file} isn't valid JSON (${e instanceof Error ? e.message : e}); fix it first.`)
  }
  // writeFile creates the folder (~/.codeium/windsurf, .cursor) as needed.
  await vscode.workspace.fs.writeFile(file, Buffer.from(text))
  log.appendLine(`[mcp] added modelmaker to ${picked.file}: ${server.command} ${server.args.join(' ')}`)
  const choice = await vscode.window.showInformationMessage(
    `Added Model-Maker to ${info.name} (${picked.file}). ${info.reload} Its agent works on the canvas you used last.`,
    'Open File',
  )
  if (choice) await vscode.window.showTextDocument(file)
}

/** Runs a command; its exit code (-1 when it can't start) and output. */
function runQuiet(command: string, args: string[]): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    // Through the shell on Windows, where `claude` is a .cmd shim: quote
    // what has spaces (a user folder often does).
    const win = process.platform === 'win32'
    const quoted = win ? args.map((a) => (/[\s&()^]/.test(a) ? `"${a}"` : a)) : args
    const proc = spawn(command, quoted, { windowsHide: true, shell: win })
    let out = ''
    proc.stdout.on('data', (d: Buffer) => (out += d.toString()))
    proc.stderr.on('data', (d: Buffer) => (out += d.toString()))
    proc.on('error', (e) => resolve({ code: -1, out: String(e) }))
    proc.on('close', (code) => resolve({ code: code ?? -1, out }))
  })
}

async function selectInterpreter() {
  if (vscode.extensions.getExtension('ms-python.python')) {
    await vscode.commands.executeCommand('python.setInterpreter')
    return
  }
  const config = vscode.workspace.getConfiguration('modelmaker')
  const value = await vscode.window.showInputBox({
    title: 'Python interpreter for the Model-Maker backend',
    prompt: 'Path to a python executable (install the Python extension to pick from a list)',
    value: config.get<string>('pythonPath') ?? '',
  })
  if (value !== undefined) await config.update('pythonPath', value, vscode.ConfigurationTarget.Workspace)
}
