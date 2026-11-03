import { execFile } from 'node:child_process'
import * as path from 'node:path'
import * as vscode from 'vscode'
import { BackendManager, installPackage, managedStep, packageSpec } from './backend'
import { CopilotRelay } from './copilot'
import { ModelEditorProvider, VIEW_TYPE } from './editor'
import { ManagedEnvironment } from './managed'
import { downloadEnv, onPythonChanged, resolvePython, usesSelectedInterpreter } from './python'
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
  // On the machine with the files (remote included), shared by all projects.
  const managed = new ManagedEnvironment(context.globalStorageUri.fsPath, (text) => log.append(text))
  const backends = new BackendManager(log, relay, managed)
  context.subscriptions.push(log, relay, backends)
  const editor = ModelEditorProvider.register(context, backends, log)
  registerTools(context, backends, editor)

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
  command('modelmaker.openDemo', () => openDemo(backends, log))
  command('modelmaker.save', () => editor.saveActive())
  command('modelmaker.restartBackend', () => backends.restart())
  command('modelmaker.showLog', () => log.show())
  command('modelmaker.selectInterpreter', () => selectInterpreter(managed))
  command('modelmaker.installPackage', async () => {
    const folder = vscode.workspace.workspaceFolders?.[0]?.uri ?? vscode.Uri.file(process.cwd())
    const choice = await resolvePython(folder)
    try {
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'Model-Maker: installing the Python package' },
        (progress) =>
          choice.managed
            ? managedStep(() => managed.upgrade(packageSpec(), (message) => progress.report({ message }), downloadEnv()))
            : installPackage(choice.python, log, progress),
      )
      const where = choice.managed ? "Model-Maker's environment" : choice.python
      const restart = await vscode.window.showInformationMessage(`Installed into ${where}.`, 'Restart Backend')
      if (restart) await backends.restart()
    } catch (e) {
      vscode.window.showErrorMessage(String(e instanceof Error ? e.message : e))
    }
  })
  command('modelmaker.resetEnvironment', async () => {
    const ok = await vscode.window.showWarningMessage(
      "Delete Model-Maker's managed Python environment? It's rebuilt on the next start (this needs internet access).",
      { modal: true },
      'Reset',
    )
    if (ok !== 'Reset') return
    // Open canvases restart onto a freshly built environment.
    try {
      await backends.restart(undefined, () => managed.reset())
    } catch (e) {
      vscode.window.showErrorMessage(`Couldn't reset the environment: ${e instanceof Error ? e.message : e}`)
    }
  })

  // A different interpreter means a different package: restart onto it --
  // when the Python extension's selection is what runs the backend.
  onPythonChanged(() => {
    if (usesSelectedInterpreter()) backends.restart()
  }).then((d) => context.subscriptions.push(d))

  // Settings a running backend was started with: offer to restart onto them.
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (e) => {
      const keys = ['modelmaker.pythonPath', 'modelmaker.environment', 'modelmaker.llm.useCopilot', 'modelmaker.backend.extraEnv']
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
  const folder = target.fsPath
  if (await exists(vscode.Uri.joinPath(target, PROJECT_FILE))) {
    return vscode.window.showErrorMessage(`${folder} already has a project; open it with Model-Maker: Open Project.`)
  }
  // The backend writes the project (model.json, files/, .gitignore, git init)
  // exactly as the browser app's Save does.
  try {
    await vscode.workspace.fs.createDirectory(target)
    const backend = await backends.acquire(folder)
    try {
      await backend.json('/project/new', { method: 'POST' })
      await backend.json('/project/save', { method: 'POST', body: { path: folder } })
      await openCanvas(vscode.Uri.joinPath(target, PROJECT_FILE))
    } finally {
      backends.release(folder)
    }
  } catch (e) {
    vscode.window.showErrorMessage(`Couldn't create the project: ${e instanceof Error ? e.message : e}`)
  }
}

/** Copies the demo projects (modelmaker-demo) into a folder and opens one:
 * with the managed environment there's no modelmaker-demo on PATH. */
async function openDemo(backends: BackendManager, log: vscode.OutputChannel) {
  const parent = vscode.workspace.workspaceFolders?.[0]?.uri
  const target = await vscode.window.showSaveDialog({
    defaultUri: parent ? vscode.Uri.joinPath(parent, 'modelmaker-demo') : undefined,
    title: 'Model-Maker demo: choose a folder to create',
    saveLabel: 'Create Demo',
  })
  if (!target) return
  const project = vscode.Uri.joinPath(target, 'projects', 'demo_pd_model', PROJECT_FILE)
  try {
    await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: 'Model-Maker: creating the demo' },
      async (progress) => {
        const { python } = await backends.preparePython(path.dirname(target.fsPath), progress)
        progress.report({ message: 'generating the demo data' })
        await new Promise<void>((resolve, reject) => {
          execFile(
            python,
            ['-c', 'from modelmaker.demo_data import workspace_main; workspace_main()', target.fsPath],
            { windowsHide: true },
            (err, stdout, stderr) => {
              log.append(stdout + stderr)
              if (err) reject(new Error(stderr.trim().split('\n').pop() || err.message))
              else resolve()
            },
          )
        })
      },
    )
    await openCanvas(project)
  } catch (e) {
    vscode.window.showErrorMessage(`Couldn't create the demo: ${e instanceof Error ? e.message : e}`)
  }
}

async function selectInterpreter(managed: ManagedEnvironment) {
  const config = vscode.workspace.getConfiguration('modelmaker')
  const target = vscode.ConfigurationTarget.Workspace
  const hasPythonExtension = !!vscode.extensions.getExtension('ms-python.python')
  type Item = vscode.QuickPickItem & { id: 'managed' | 'selected' | 'path' }
  const items: Item[] = [
    {
      id: 'managed',
      label: "Model-Maker's own environment",
      description: 'recommended',
      detail: `Set up automatically; nothing to install yourself (${managed.root})`,
    },
    {
      id: 'selected',
      label: hasPythonExtension ? "The Python extension's interpreter" : 'python3 / python on PATH',
      detail: 'Blocks can use the packages in your own environment; Model-Maker is pip-installed into it',
    },
    { id: 'path', label: 'Enter a path…', detail: 'A specific python executable' },
  ]
  const picked = await vscode.window.showQuickPick(items, { title: 'Which Python runs the Model-Maker backend?' })
  if (!picked) return
  if (picked.id === 'path') {
    const value = await vscode.window.showInputBox({ title: 'Path to a python executable', value: config.get<string>('pythonPath') ?? '' })
    if (value !== undefined) await config.update('pythonPath', value, target)
    return
  }
  await config.update('pythonPath', undefined, target)
  await config.update('environment', picked.id, target)
  if (picked.id === 'selected' && hasPythonExtension) await vscode.commands.executeCommand('python.setInterpreter')
}
