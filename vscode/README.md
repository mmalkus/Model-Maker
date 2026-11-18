# Model-Maker for VS Code

The [Model-Maker](https://github.com/mmalkus/Model-Maker) canvas inside VS
Code. Build financial model pipelines visually, run them block by block,
and compile them to a single readable Python script, next to your code.
GitHub Copilot can be the AI that drafts blocks and runs AI model builds.

## What you need

- **Python 3.11 or newer** on the machine, and access to a package index:
  PyPI, or your company's mirror (whatever `pip` is configured for).
  Model-Maker itself ships inside the extension. The first time a canvas
  opens, the extension makes its own Python environment with Model-Maker's
  open-source requirements (polars, scikit-learn, scipy, numpy, matplotlib,
  fastapi and a few more; about 500 MB). It never installs a Model-Maker
  package and never touches your other Python environments.
- Optional: GitHub Copilot, for the AI features. Without it, the canvas's
  Settings offers the other providers (Anthropic, OpenAI, Gemini, a local
  LM Studio, the Claude Code CLI).

## Use it

- **Open a project:** right-click a project's `model.json`, choose
  **Open With… → Model-Maker Canvas**, or click the canvas button in the
  editor title bar. You can also run **Model-Maker: Open Project…**. When
  a workspace contains a project, the extension offers to open it.
- **Start a project:** run **Model-Maker: New Project…**.
- **Try the demo:** run `modelmaker-demo .` in a terminal, then open
  `projects/demo_pd_model/model.json`.

The canvas autosaves to `model.json` a couple of seconds after each edit,
and **Ctrl/Cmd+S** saves straight away. Git works through VS Code's
Source Control. **Compile** opens the generated script in a new editor
tab.

## Theme and layout

Inside VS Code the canvas uses your colour theme, whether light, dark or
high contrast, and switches when you switch theme. It's laid out for an
editor tab: one toolbar row (the rarer run modes are under **More**), a
block palette that folds to a rail (**Blocks ›** opens it), and an inspector
that appears when you select a block. Saving is VS Code's: Ctrl/Cmd+S, and
edits autosave.

## How it works

Each open project gets its own local backend: `python -m modelmaker.api`
on a random `127.0.0.1` port with a random access token, started in the
project folder and stopped when the canvas closes.

Which Python runs it is `modelmaker.python.environment`:

- `managed` (the default): the extension's own environment, in its storage
  folder. It's made once from a Python 3.11+ (`modelmaker.basePython`, else
  `py -3`, `python3` or `python`), and its requirements are reinstalled only
  when a new version of the extension needs different ones. **Model-Maker:
  Repair Python Environment** reinstalls them.
- `interpreter`: the interpreter you select (the Python extension's choice,
  or `modelmaker.pythonPath`). The extension offers to install whatever
  requirements it lacks. Use it to add your own packages for library blocks.
- `installedPackage`: the `quantology-modelmaker` package installed in the
  selected interpreter, for working on Model-Maker itself (`pip install -e`).

The first two run the Model-Maker that ships with the extension, so it
always matches the extension's version. The canvas is the same
app `modelmaker-api` serves in a browser. The extension declares
`extensionKind: workspace`, so over Remote-SSH, WSL, Dev Containers and
Codespaces the backend runs next to your files.

Copilot is offered to the backend through a small relay inside the
extension, an OpenAI-compatible endpoint over VS Code's Language Model
API. The first AI call asks you to allow Model-Maker to use Copilot.

## Copilot Chat tools

In Copilot Chat's agent mode, Copilot can work on the open canvas with these
tools. Mention them with `#` (for example `#mmGraph`) or let agent mode pick
them:

| Tool | Does |
|---|---|
| `#mmOpenProject` | Opens a project's canvas, or creates a new project |
| `#mmGraph` | Reads the blocks, wires, statuses and errors |
| `#mmFiles` | Finds data files (and the workspace's datasets, if any) |
| `#mmBlockTypes`, `#mmDescribeBlock` | Browse and read the block catalogue, including data sources and your library |
| `#mmOutput`, `#mmColumns` | Summarize a block's output, or list the columns it receives: schema and statistics, never rows |
| `#mmAddLane`, `#mmAddBlock`, `#mmSetParams`, `#mmColumnRole` | Lay out and configure the pipeline |
| `#mmConnect`, `#mmDisconnect`, `#mmDeleteBlock` | Wire, unwire and remove blocks |
| `#mmRun` | Runs the pipeline, or up to one block, and reports errors and leakage |
| `#mmCheck` | Checks for holdout leakage and suspicious features, with each block's decision hints |
| `#mmCompile` | Returns the compiled Python script |

Copilot can start from nothing: ask agent mode to "create a project in
`models/pd` and build a PD scorecard from `data/loans.csv`". It opens the
canvas and you watch the model take shape.

Every edit asks you first, lands in the canvas's undo history, and shows up
on the canvas straight away. Copilot sees the same data the AI builder
does: column names, types, roles and summary statistics, never your rows.

## Claude Code

Claude Code can use the same tools through a small MCP server in the
Python package, `modelmaker-mcp`, on the canvas you used last. Set it up
once per workspace:

1. Run **Model-Maker: Set Up Claude Code**. It adds a `modelmaker` entry
   to the workspace's `.mcp.json`, using the selected Python interpreter.
2. Start (or restart) Claude Code in that folder, in the panel or the
   terminal, and approve the `modelmaker` server.
3. Open a canvas, or ask Claude to create a project. Its edits show up on
   the canvas as it works.

The `.mcp.json` entry holds the paths of the extension's Python and its
copy of Model-Maker, which stay put across extension updates. Before committing it
for others, change `command` to a path that works for them. The server
finds the canvases through small files in `~/.modelmaker/vscode-backends/`
(each backend's local address and access token, readable by you only).
Turn this off with `modelmaker.mcp.discovery`.

The same server works with the browser app: `modelmaker-mcp --url
http://127.0.0.1:8001`.

## Commands

| Command | |
|---|---|
| Model-Maker: Open Project… | Open a project folder's canvas |
| Model-Maker: New Project… | Create a project folder (model.json, files/, git) |
| Model-Maker: Set Up Claude Code | Add the Model-Maker tools to the workspace's `.mcp.json` |
| Model-Maker: Save Project | Save the active canvas (Ctrl/Cmd+S) |
| Model-Maker: Restart Backend | Restart every running backend |
| Model-Maker: Show Backend Log | Backend output, pip installs, relay errors |
| Model-Maker: Repair Python Environment | Reinstall the managed environment's requirements |
| Model-Maker: Select Python Interpreter | Choose the interpreter for the `interpreter` and `installedPackage` environments |
| Model-Maker: Install / Upgrade Python Package | `installedPackage` only: `pip install -U` the package into the interpreter |

## Settings

| Setting | Default | |
|---|---|---|
| `modelmaker.python.environment` | `managed` | `managed`, `interpreter` or `installedPackage` (see How it works) |
| `modelmaker.basePython` | (empty) | The Python the managed environment is made from |
| `modelmaker.pythonPath` | (empty) | Interpreter for `interpreter`/`installedPackage`; empty uses the Python extension's selection |
| `modelmaker.autoInstall` | `ask` | `ask`, `always` or `never` install what a selected interpreter is missing |
| `modelmaker.packageSpec` | `quantology-modelmaker>=0.3.1` | `installedPackage` only: what gets installed, e.g. `-e /path/to/checkout` |
| `modelmaker.llm.useCopilot` | `true` | Offer Copilot to the backend and make it the default provider |
| `modelmaker.backend.extraEnv` | `{}` | Extra environment variables for the backend, e.g. API keys |
| `modelmaker.mcp.discovery` | `true` | Let `modelmaker-mcp` (Claude Code) find the open canvases |

## License

Apache-2.0
