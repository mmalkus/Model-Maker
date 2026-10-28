# VS Code extension: build and release plan

A plan for shipping Model-Maker as a VS Code extension. The canvas opens
as a native editor tab for a project's `model.json`, and the AI features
can run on the user's GitHub Copilot models. The extension adds a second
way to use Model-Maker. **The pip package stays the product**: the
extension is a thin client that installs `quantology-modelmaker` into the
user's Python environment and drives `modelmaker-api`, just as the
browser does today.

Decisions already taken:

- **The UI loads as a native webview (option B).** The extension doesn't
  show the localhost page in an iframe. It loads the built frontend
  bundle into a VS Code webview, and that webview talks to a backend the
  extension starts.
- **The pip and browser workflow keeps working as it does today.** Every
  backend change is opt-in (an env var the extension sets) and leaves
  default behaviour byte-for-byte unchanged. One frontend source tree
  builds both targets.
- **Copilot is reached through a relay.** Only extension code can call
  Copilot (via `vscode.lm`), so the extension serves it to the backend as
  an OpenAI-compatible endpoint. The existing `OpenAILoop` and OpenAI
  provider do the rest.

## 1. Architecture

```
VS Code window
├─ Extension host (TypeScript, vscode/)
│   ├─ BackendManager: one `python -m modelmaker.api` per open project
│   │     env: MODELMAKER_PORT, MODELMAKER_AUTH_TOKEN, MODELMAKER_CACHE_DIR,
│   │          MODELMAKER_CORS_ORIGIN_REGEX, MODELMAKER_COPILOT_*
│   ├─ ModelEditorProvider: custom editor for model.json (webview)
│   │     message bridge: save file, pick path, confirm, notify, open file
│   └─ CopilotRelay: 127.0.0.1:<port>/v1/{models,chat/completions} → vscode.lm
│
├─ Webview (frontend/ built with --mode vscode)
│     fetch → http://127.0.0.1:<backend port>/api/*   (Bearer token, portMapping)
│
└─ Python backend (unchanged package, opt-in env flags)
      llm provider "copilot" → CopilotRelay (OpenAI wire format)
```

**One backend per open project.** `api.py` keeps a single global
`SESSION`, so one server holds one project. Rather than refactor that,
the extension starts one backend per open Model-Maker editor, each with
its own port and cache dir. Each backend prewarms its worker pool, so
memory cost grows with the number of open projects. Opening two projects
at once is rare, so this is acceptable.

**Remote development (SSH, WSL, Codespaces, Dev Containers).** The
extension is declared `"extensionKind": ["workspace"]`, so it runs, and
starts Python, wherever the files are. The webview's `portMapping`
forwards `127.0.0.1:<port>` from the client to that host, so the same
code covers the local and remote cases.

## 2. Compatibility guarantees for the Python package

These guarantees are invariants. CI enforces them (§7) and no phase may
break them:

1. With none of the new env vars set, `pip install quantology-modelmaker`
   followed by `modelmaker-api` serves the same app on port 8001, with no
   auth and the same CORS settings.
2. `cd frontend && npm run build` still writes to `modelmaker/static/`,
   and `setup.py` still bakes that output into the wheel. The VS Code
   build writes to its own directory and never touches
   `modelmaker/static/`.
3. Inside the frontend, the browser host is the default. VS Code
   behaviour switches on only when the webview injects its config (§4.1).
4. The TUI, `run.bat`, `modelmaker-demo` and the CLI are untouched.
5. `vscode/` doesn't ship in the sdist or the wheel: it isn't under the
   package directory and isn't grafted in `MANIFEST.in`.
6. The extension needs a minimum backend version, which it checks
   through `/api/version`. The Python package never depends on the
   extension.

## 3. Backend changes (Python, all opt-in)

| Change | Where | Default (env unset) |
|---|---|---|
| `MODELMAKER_AUTH_TOKEN`: when set, a middleware requires `Authorization: Bearer <t>` on `/api/*`. `GET` requests may pass `?token=` instead, because `<img src>` for `imageUrl` and library export downloads can't send headers. `/api/health` stays open. | `api.py` | No auth (as today) |
| `MODELMAKER_CORS_ORIGIN_REGEX`: added to the existing `CORSMiddleware` as `allow_origin_regex`. The extension sets it to `^vscode-webview://[a-z0-9-]+$`. | `api.py:53` | Only the Vite dev origins (as today) |
| `GET /api/version` returns `{"version": <package version>, "api": <int compat level>}` so the extension can refuse an incompatible backend. | `api.py` | Additive endpoint |
| A `copilot` LLM provider and agent loop. These are thin subclasses of the OpenAI provider and `OpenAILoop` that read `MODELMAKER_COPILOT_BASE_URL` and `MODELMAKER_COPILOT_TOKEN`. When those are unset they report "only available inside VS Code", the same way other providers report a missing key. Added to `AGENT_CAPABLE_PROVIDERS`. | `llm/copilot_provider.py`, `agent/loop.py` | Listed but unavailable |

Two things the extension handles with existing mechanisms, without
backend changes:

- **Per-project cache.** `MODELMAKER_CACHE_DIR` already exists
  (`session.py:24`). The extension points it at
  `<project>/.modelmaker-cache`, so two windows never clobber each
  other's recovery snapshot.
- **Default provider.** `MODELMAKER_LLM_PROVIDER=copilot` is set when the
  `modelmaker.llm.useCopilot` setting is on (the default when Copilot is
  available). The Settings panel can still switch providers.

The project is opened by calling `POST /api/project/load` after
`/api/health` responds, so the backend needs no new startup flag.

**Tests (pytest):**

- Without a token, every existing test passes unchanged; the existing
  suite covers this.
- With a token set, requests without it get 401. A header or a `?token=`
  on a GET passes. A `?token=` on a POST is rejected.
- The CORS regex is applied only when its env var is set.
- `/api/version` returns the package version.
- The `copilot` provider works against a fake OpenAI-compatible server,
  using the same pattern as the existing LM Studio and OpenAI tests.

## 4. Frontend changes (one source tree, two builds)

### 4.1 Host abstraction: `frontend/src/host.ts`

All browser-only behaviour goes behind one interface:

```ts
interface Host {
  apiBase: string                   // '' in the browser; 'http://127.0.0.1:<port>' in VS Code
  authHeaders(): Record<string, string>
  withToken(url: string): string    // for <img src> and downloads
  saveFile(name: string, data: Blob): Promise<void>
  pickPath(opts: { folders?: boolean; ext?: string; startPath?: string }): Promise<string | null>
  confirm(message: string): Promise<boolean>
  notify(message: string, kind?: 'error' | 'info'): void
  openFile?(path: string): void     // VS Code only: open compiled script or block code in an editor
  getState / setState               // localStorage in the browser; vscode.setState in VS Code
}
```

`browserHost` does exactly what the code does today. `vscodeHost` is
chosen when the webview injects `window.__MODELMAKER__ = { apiBase,
token }`, and it talks to the extension through `acquireVsCodeApi()`
`postMessage` request/response pairs.

### 4.2 Call sites to convert

Every one of these has to change, because VS Code webviews block
`alert`, `confirm` and `<a download>`:

- **`api.ts`:** `request()` (line 30), `imageUrl` (line 147) and
  `libraryExportUrl` (line 240) all build `/api` URLs. They switch to
  `host.apiBase`, `host.authHeaders()` and `host.withToken()`.
- **`alert()` (about 12 call sites)** in `App.tsx`, `Inspector.tsx`,
  `PortDataView.tsx`, `Toolbar.tsx`, `SettingsPanel.tsx` and
  `DataModal.tsx` become `host.notify()`.
- **`confirm()` (6 call sites)** become `await host.confirm()`. This
  changes those handlers from sync to async; the changes are small.
- **Downloads:** `BuildPanel.tsx:658`, `BuildReport.tsx:42` and
  `LibraryPanel.tsx:56` move to `host.saveFile()`. In VS Code that
  becomes `showSaveDialog` plus a file write.
- **`FileBrowser`:** `Toolbar.tsx:319/329` (save/load project) and
  `Inspector.tsx:570` (path params) call `host.pickPath()`. In the
  browser that opens the existing `FileBrowser` component; in VS Code it
  opens the native `showOpenDialog`. The backend-driven browser and
  `/api/browse` stay for the browser build.
- **Storage:** `roles.ts` moves from `localStorage` to
  `host.getState()`/`setState()`.
- **Inside VS Code only:** hide the Git panel (VS Code's own Source
  Control covers it) and show "Open in editor" for compiled scripts and
  block code.

A lint rule (`no-restricted-globals` for `alert`/`confirm`, plus a grep
check in CI for `'/api` outside `api.ts`) stops new call sites creeping
back in.

### 4.3 Build targets

- `npm run build` stays as it is today and writes to
  `../modelmaker/static`.
- `npm run build:vscode` runs `vite build --mode vscode`, with
  `base: './'` and `outDir: '../vscode/media/webview'`. It also disables
  module preload polyfills that inject inline scripts, so a strict CSP
  works.

### 4.4 Theming (polish)

Map the app's CSS colour tokens to `--vscode-*` variables when the
`body.vscode-*` theme classes are present, so the canvas follows the
user's light, dark or high-contrast theme. The browser keeps its
current palette.

## 5. Extension (`vscode/`, TypeScript)

```
vscode/
  package.json        manifest: contributes, activation, settings
  src/extension.ts    activate(): register editor, commands, relay
  src/backend.ts      BackendManager: find Python, install/upgrade, spawn, health, stop
  src/editor.ts       ModelEditorProvider (CustomEditorProvider) + webview HTML/CSP
  src/bridge.ts       webview message handlers (save, pick, confirm, notify, open)
  src/copilot.ts      CopilotRelay: OpenAI-compatible HTTP server over vscode.lm
  src/python.ts       interpreter discovery via @vscode/python-extension, fallback setting
  media/webview/      (build output, gitignored)
  test/               @vscode/test-electron integration tests
  README.md CHANGELOG.md LICENSE icon.png .vscodeignore esbuild.mjs
```

### 5.1 Manifest (`package.json` contributes)

- **Custom editor `modelmaker.model`** for `**/model.json`, with
  priority `option`. Plenty of unrelated repos contain a `model.json`,
  so the extension shouldn't take it over. Instead an editor-title
  button and a notification ("This folder is a Model-Maker project.
  Open the canvas?") appear when the file sits next to a Model-Maker
  project layout.
- **Commands:** `Model-Maker: Open Project…`, `New Project…`, `Restart
  Backend`, `Show Backend Log`, `Select Python Interpreter`, `Install /
  Upgrade Python Package`.
- **Settings:**
  - `modelmaker.pythonPath`: empty means use the Python extension's
    interpreter.
  - `modelmaker.autoInstall`: `ask` / `always` / `never`.
  - `modelmaker.packageSpec`: default
    `quantology-modelmaker>=<min version>`.
  - `modelmaker.llm.useCopilot`: default true.
  - `modelmaker.backend.extraEnv`.
- **Activation:** `onCustomEditor:modelmaker.model`, `onCommand:*`, and
  `workspaceContains:**/model.json`. The last one only shows the hint
  and never starts a backend.
- **Other fields:** `extensionKind: ["workspace"]`, and the Python
  extension (`ms-python.python`) as a soft dependency, used when present.

### 5.2 BackendManager

1. **Find the interpreter.** Use `modelmaker.pythonPath`, then the
   Python extension's active environment for the workspace, then
   `python3`/`py`.
2. **Check the package.** Run `python -c "import modelmaker, importlib.metadata
   as m; print(m.version('quantology-modelmaker'))"`. If it's missing or
   older than the minimum version, offer to install. The install runs
   `python -m pip install -U "<packageSpec>"` with progress in a
   notification and output in the log channel. The extension never
   installs into a system Python without asking.
3. **Spawn the server.** Run `python -m modelmaker.api` with `cwd` set to
   the project folder and env vars `MODELMAKER_HOST=127.0.0.1`,
   `MODELMAKER_PORT=<free port>`, `MODELMAKER_AUTH_TOKEN=<random>`,
   `MODELMAKER_CORS_ORIGIN_REGEX`, `MODELMAKER_CACHE_DIR` and the Copilot
   variables. If the port is taken, retry on a new one.
4. **Wait for it.** Poll `/api/health`, then check `/api/version`
   against the compat level. If they're incompatible, offer to upgrade
   the package.
5. **Supervise.** Stream stdout and stderr to an OutputChannel. If the
   process exits unexpectedly, show "Backend stopped. Restart?". Kill it
   when the editor closes and on deactivate (on Windows, kill the whole
   process tree because of the worker pool).

### 5.3 ModelEditorProvider

The editor is a `CustomEditorProvider` with a custom document. **The
backend is the source of truth**:

- **Open:** start a backend, call `/api/project/load` with the folder,
  then render the webview.
- **Dirty state:** the frontend already tracks `dirty`. It posts
  `dirty-changed`, which fires `onDidChangeCustomDocument` so VS Code
  shows the dot and asks before closing.
- **Save (Ctrl+S):** `POST /api/project/save`. **Revert:**
  `/api/project/load`. Undo and redo stay in the app, which has its own
  50-step stack, and are routed from VS Code's undo commands to the
  backend.
- **External edits:** a file watcher on `model.json` (from `git
  checkout` or another editor) offers to reload.
- **Webview options:** `enableScripts`, `retainContextWhenHidden` (keeps
  the canvas state), `localResourceRoots: [media/webview]` and
  `portMapping: [{webviewPort: p, extensionHostPort: p}]`.
- **HTML:** read `media/webview/index.html`, rewrite asset URLs with
  `asWebviewUri`, inject `window.__MODELMAKER__`, and set this CSP:
  `default-src 'none'; script-src 'nonce-…'; style-src ${cspSource}
  'unsafe-inline'; img-src ${cspSource} data: blob:
  http://127.0.0.1:${p}; connect-src http://127.0.0.1:${p}; font-src
  ${cspSource}`. React Flow and CodeMirror need inline styles.

### 5.4 CopilotRelay

- **Server:** Node's `http` server on `127.0.0.1:0`, with a random
  bearer token, started lazily the first time a backend starts with
  Copilot enabled.
- **`GET /v1/models`:** `vscode.lm.selectChatModels({ vendor: 'copilot'
  })` mapped to `{ data: [{ id }] }`.
- **`POST /v1/chat/completions`:** translates the request to
  `LanguageModelChatMessage`s. Tool definitions become
  `LanguageModelChatTool`s, and `tool_calls`/`tool` messages map to
  `LanguageModelToolCallPart`/`LanguageModelToolResultPart`.
  `model.sendRequest` is called with a `justification` string. The
  stream is collected into a single non-streaming completion
  (`OpenAILoop` doesn't stream). `usage` is filled from
  `model.countTokens` and marked as estimated.
- **Cancellation:** when the HTTP request closes, it cancels the
  `CancellationTokenSource`.
- **Errors:** `LanguageModelError` maps to HTTP status codes
  (`NoPermissions` → 403 with "approve Copilot access in VS Code",
  `NotFound` → 404, quota → 429), so the backend's existing error
  surfaces show something useful.
- **Consent:** the first call triggers VS Code's consent prompt. Users
  without Copilot see "Copilot unavailable" in the provider list and
  keep the other providers.
- **Tests:** unit tests for the message and tool translation; an
  integration test with a fake `vscode.lm` model.

### 5.5 Copilot Chat tools (later, optional)

Expose Model-Maker to Copilot agent mode by registering a project-level
MCP server through `vscode.lm.registerMcpServerDefinitionProvider`. This
needs a project-scoped variant of `agent/mcp_server.py`; today's server
is scoped to a single build's token. It's kept out of the first release
and tracked as a follow-up.

## 6. Phases

| Phase | Scope | Exit criteria | Estimate |
|---|---|---|---|
| **0. Scaffolding** | `vscode/` skeleton, esbuild, lint, `build:vscode` target, CI job that packages a `.vsix` | CI uploads a `.vsix` artifact; pip CI unchanged and green | 1–2 days |
| **1. Backend opt-ins** | Token middleware, CORS regex, `/api/version`, tests | New tests pass; all existing tests pass unchanged | 1–2 days |
| **2. Host abstraction** | `host.ts`, convert all call sites (§4.2), lint guard | Browser build behaves the same (manual pass + existing e2e); no `alert`/`confirm`/raw `/api` left | 3–4 days |
| **3. Editor + backend manager** | §5.2, §5.3, bridge, commands, settings | Demo project opens, runs green, saves, and reverts in VS Code locally and over Remote-SSH/WSL | 5–7 days |
| **4. Copilot** | `copilot` provider (Python), CopilotRelay (TS) | Draft, fix and full AI build of the demo work on a Copilot model | 3–5 days |
| **5. Polish** | Theming, open-in-editor, hide the Git panel, README/screenshots, walkthrough | Marketplace-ready listing | 2–3 days |
| **6. Release** | §8 pipeline, pre-release then stable | Published on the Marketplace and Open VSX | 1–2 days |

That's about 3–4 weeks in total. Phases 1 and 2 merge to `main` on their
own, because they're invisible to pip users.

## 7. CI (`.github/workflows/ci.yml`)

The existing `tests`, `frontend` and `package` jobs stay. The `package`
job's `test -f …/static/index.html` and demo-from-wheel check already
guard invariants 1 and 2. Changes:

- **The `frontend` job** also runs `npm run build:vscode` and checks
  that `modelmaker/static/` was not modified by it.
- **A new `vscode` job:**
  1. Set up Node 22 and Python 3.11, then `pip install -e .`.
  2. Run `npm ci` in `frontend/` and `vscode/`, then lint, typecheck and
     the unit tests (relay translation and the bridge).
  3. `npm run build:vscode`, then `npx vsce package` (no Marketplace
     auth needed).
  4. Run `xvfb-run npx vscode-test`. These integration tests open the
     demo project, assert the backend reaches health with auth enforced,
     the webview loads, Run all goes green, and Save writes `model.json`.
     Copilot is stubbed.
  5. Upload the `.vsix` as an artifact so reviewers can install it from
     any PR.

## 8. Deployment of the extension

### 8.1 One-time setup (manual, by the repo owner)

1. Create a Visual Studio Marketplace **publisher** (e.g. `quantology`)
   at marketplace.visualstudio.com/manage.
2. Create an Azure DevOps **PAT** with scope *Marketplace → Manage*, or
   set up `vsce publish --azure-credential` with a federated Entra ID
   identity to avoid a long-lived secret.
3. Create an **Open VSX** namespace and token at open-vsx.org, so Cursor,
   VSCodium and Gitpod users get it too.
4. Add the GitHub environment **`vscode-marketplace`** with secrets
   `VSCE_PAT` and `OVSX_PAT` and required reviewers. This mirrors the
   existing `pypi` environment.
5. Add an icon (128×128 PNG), a gallery banner colour, categories
   (`Data Science`, `Visualization`, `Machine Learning`), and links to
   the Apache-2.0 license and repo.

### 8.2 Versioning

- **Lockstep.** The extension version matches the Python package version
  (`0.3.0` ↔ `0.3.0`), and one GitHub release ships both. The extension
  still accepts any backend at or above its declared minimum version
  with the same `/api/version` compat level. A user with a newer pip
  package therefore isn't forced to update the extension.
- **Pre-release channel.** Odd minor versions are published with
  `vsce publish --pre-release` (Marketplace convention); even minors are
  stable.

### 8.3 Release workflow (extend `.github/workflows/publish.yml`)

The extension installs the package from PyPI, so it must publish after
PyPI. It's added as jobs in the same workflow, not a separate one:

```yaml
  build-vscode:
    runs-on: ubuntu-latest
    steps:
      - checkout; setup-node 22
      - npm ci (frontend/, vscode/)
      - check vscode/package.json version == pyproject version == release tag
      - npm run build:vscode
      - npx vsce package -o modelmaker.vsix   # --pre-release for odd minors
      - upload-artifact: modelmaker.vsix

  publish-vscode:
    needs: [publish, build-vscode]            # after PyPI is live
    environment: vscode-marketplace
    steps:
      - download-artifact
      - npx vsce publish --packagePath modelmaker.vsix -p ${{ secrets.VSCE_PAT }}
      - npx ovsx publish modelmaker.vsix -p ${{ secrets.OVSX_PAT }}
      - attach modelmaker.vsix to the GitHub release (offline/air-gapped installs)
```

The VSIX is universal (no native modules), so it needs no per-platform
targets. The `.vsix` attached to the release covers corporate machines
that can't reach the Marketplace (`code --install-extension
modelmaker.vsix`).

### 8.4 Release checklist

1. Bump the version in `pyproject.toml` and `vscode/package.json`, then
   update `vscode/CHANGELOG.md`.
2. Merge, then create a GitHub release with tag `vX.Y.Z`.
3. `publish.yml` runs: PyPI first, then the Marketplace and Open VSX
   (after approval in the `vscode-marketplace` environment).
4. Smoke test: install from the Marketplace into a clean profile with a
   fresh venv. It should offer to install the package, open the demo and
   run it green.

## 9. Risks and open questions

- **Publisher name and display name.** This needs the owner's choice
  before phase 6; it can't be changed after publishing.
- **Unrelated `model.json` files.** These are why the custom editor has
  `option` priority (§5.1). If that proves too hidden, it could detect a
  Model-Maker `model.json` by its format marker and switch to default
  priority for those files only.
- **Copilot policy.** Organisations can disable some models or
  third-party extension access. The relay must degrade cleanly (§5.4),
  and docs should say the other providers still work.
- **Memory with several projects open.** Each backend prewarms a worker
  pool. If this bites, add `MODELMAKER_WORKER_PREWARM=0` (opt-in) for
  extension-started backends.
- **The webview's `portMapping` across remotes.** It's supported, but
  needs testing on Codespaces early in phase 3. The fallback is
  `asExternalUri` plus proxying every request through `postMessage`.
- **Windows process cleanup.** Spawned workers can outlive the server
  if only the parent is killed. Use `taskkill /T` or tree-kill.
