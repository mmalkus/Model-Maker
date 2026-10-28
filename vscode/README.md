# Model-Maker for VS Code

The [Model-Maker](https://github.com/mmalkus/Model-Maker) canvas inside VS
Code. Build financial model pipelines visually, run them block by block,
and compile them to a single readable Python script, next to your code.
GitHub Copilot can be the AI that drafts blocks and runs AI model builds.

## What you need

- Python 3.11 or newer. The extension runs the
  [`quantology-modelmaker`](https://pypi.org/project/quantology-modelmaker/)
  package in it and offers to `pip install` the package if it's missing.
  A virtual environment is the safe choice; pick it with the
  [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  or the `modelmaker.pythonPath` setting.
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

## How it works

Each open project gets its own local backend: `python -m modelmaker.api`
on a random `127.0.0.1` port with a random access token, started in the
project folder and stopped when the canvas closes. The canvas is the same
app `modelmaker-api` serves in a browser. The extension declares
`extensionKind: workspace`, so over Remote-SSH, WSL, Dev Containers and
Codespaces the backend runs next to your files.

Copilot is offered to the backend through a small relay inside the
extension, an OpenAI-compatible endpoint over VS Code's Language Model
API. The first AI call asks you to allow Model-Maker to use Copilot.

## Commands

| Command | |
|---|---|
| Model-Maker: Open Project… | Open a project folder's canvas |
| Model-Maker: New Project… | Create a project folder (model.json, files/, git) |
| Model-Maker: Save Project | Save the active canvas (Ctrl/Cmd+S) |
| Model-Maker: Restart Backend | Restart every running backend |
| Model-Maker: Show Backend Log | Backend output, pip installs, relay errors |
| Model-Maker: Select Python Interpreter | Choose the interpreter that runs the backend |
| Model-Maker: Install / Upgrade Python Package | `pip install -U` the package into it |

## Settings

| Setting | Default | |
|---|---|---|
| `modelmaker.pythonPath` | (empty) | Interpreter for the backend; empty uses the Python extension's selection |
| `modelmaker.autoInstall` | `ask` | `ask`, `always` or `never` install the package when missing or too old |
| `modelmaker.packageSpec` | `quantology-modelmaker` | What gets installed, e.g. `quantology-modelmaker[anthropic]` or `-e /path/to/checkout` |
| `modelmaker.llm.useCopilot` | `true` | Offer Copilot to the backend and make it the default provider |
| `modelmaker.backend.extraEnv` | `{}` | Extra environment variables for the backend, e.g. API keys |

## License

Apache-2.0
