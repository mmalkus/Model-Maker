import * as path from 'node:path'
import * as vscode from 'vscode'
import { BackendManager, installPackage } from './backend'
import { CopilotRelay } from './copilot'
import { Discovery } from './discovery'
import { PythonEnvironments } from './environment'
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
  const discovery = new Discovery(() => vscode.workspace.getConfiguration('modelmaker').get<boolean>('mcp.discovery', true))
  context.subscriptions.push(log, relay, backends, discovery)
  const editor = ModelEditorProvider.register(context, backends, log, discovery)
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
  command('modelmaker.setUpClaudeCode', () => setUpClaudeCode(environments))
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

  // A different interpreter means a different package: restart onto it.
  onPythonChanged(() => backends.restart()).then((d) => context.subscriptions.push(d))

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

/** Writes the workspace's .mcp.json entry that lets Claude Code use the
 * Model-Maker tools (modelmaker-mcp) on the open canvas: the same Python and
 * Model-Maker the canvas runs on (see environment.ts). */
async function setUpClaudeCode(environments: PythonEnvironments) {
  const root = vscode.workspace.workspaceFolders?.[0]?.uri
  if (!root) return vscode.window.showErrorMessage('Open a workspace folder first.')
  let runtime
  try {
    runtime = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: 'Model-Maker: preparing its Python environment' },
      (progress) => environments.runtime(root, progress),
    )
  } catch (e) {
    return vscode.window.showErrorMessage(`Model-Maker: ${e instanceof Error ? e.message : e}`)
  }
  const file = vscode.Uri.joinPath(root, '.mcp.json')
  let config: { mcpServers?: Record<string, unknown> } = {}
  try {
    config = JSON.parse(Buffer.from(await vscode.workspace.fs.readFile(file)).toString('utf8'))
  } catch (e) {
    if (await exists(file)) return vscode.window.showErrorMessage(`.mcp.json isn't valid JSON (${e instanceof Error ? e.message : e}); fix it first.`)
  }
  const server: Record<string, unknown> = { command: runtime.python, args: ['-m', 'modelmaker.assistant_mcp'] }
  if (runtime.env.PYTHONPATH) server.env = { PYTHONPATH: runtime.env.PYTHONPATH }
  config.mcpServers = { ...config.mcpServers, modelmaker: server }
  await vscode.workspace.fs.writeFile(file, Buffer.from(JSON.stringify(config, null, 2) + '\n'))
  const choice = await vscode.window.showInformationMessage(
    'Added Model-Maker to .mcp.json. Start (or restart) Claude Code in this folder and approve the "modelmaker" server; it works on the canvas you used last.',
    'Open .mcp.json',
  )
  if (choice) await vscode.window.showTextDocument(file)
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
