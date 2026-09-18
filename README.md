# Model-Maker

A visual pipeline builder for financial model development in Python, with
both a web GUI and a terminal UI, and an embedded, pluggable AI provider
that can draft and fix blocks for you. Blocks contain functional Python
code, are wired together via Polars DataFrames carrying metadata,
organized visually into phase-labelled swimlanes, and compile down to a
single, human-readable, self-contained Python script.

See [`modelmaker-v2-plan.md`](modelmaker-v2-plan.md) for the full design.

The project has two parts:

- **`modelmaker/`** — the Python backend: block registry, graph/session
  model, execution runner, single-file compiler, pluggable AI-assisted
  block drafting, a FastAPI HTTP API, and a Textual-based terminal UI
  (`modelmaker-tui`, see [Terminal UI](#terminal-ui)).
- **`frontend/`** — a React + Vite web UI for building and running
  pipelines against that API.

The [`quantology-modelmaker`](https://pypi.org/project/quantology-modelmaker/)
package on PyPI ships both: the frontend's built bundle is baked into the
wheel, so `modelmaker-api` alone serves the whole app on one port with no
separate frontend build or Node.js install required.

## Install

```bash
pip install quantology-modelmaker
modelmaker-api
```

Then open `http://127.0.0.1:8001` in a browser. Requires Python >= 3.11.

Optional extras:

```bash
pip install "quantology-modelmaker[anthropic]"  # call the Anthropic API directly
pip install "quantology-modelmaker[tui]"        # terminal UI, see below
```

(Optional) the [Claude Code](https://claude.com/claude-code) CLI (`claude`)
on `PATH` and logged in enables AI-assisted block drafting with no extra
install — it's the default LLM provider. See [LLM provider](#llm-provider)
for alternatives that don't need it.

## Quick start (Windows)

Download [`run.bat`](run.bat) by itself (no clone needed) and double-click
it. It creates a `.venv` next to itself if missing, installs/upgrades
`quantology-modelmaker` into it, and starts `modelmaker-api` at
`http://127.0.0.1:8001`. Safe to re-run any time to pick up updates.

## Projects

A project is a folder, not a file: **Save**/**Load** browse for a folder
rather than a filename, and a folder is a project once it holds a
`model.json` (everything else in it -- `files/`, a `.gitignore`, custom
block code sidecars -- is the project's own working area, versioned
alongside it). Saving for the first time scaffolds that folder (see
[The demo project](#the-demo-project) for an example) and git-inits it, so
it's ready to commit and push right away from the **Git** panel.

## The demo project

[`projects/demo_pd_model/`](projects/demo_pd_model) is a complete, runnable
PD pipeline over [`sample_data/pd_model_data.csv`](sample_data): load →
clean → weight-of-evidence → train/test split → logistic regression → Gini,
KS and PSI. Open it with **Load** (it's what the folder dialog opens on),
then **Run all** — it reads the CSV source itself since it's never been read
before, then runs the rest of the pipeline.

It's the quickest way to see what a finished graph looks like, and the test
suite runs it end to end so it can't rot. This folder lives in the git
repository rather than the PyPI package — clone the repo (or just download
it) to try it.

## Working in the tool

A few things worth knowing beyond the canvas itself.

- **Sample mode** — the `Sample` toggle in the top bar caps *every* source at
  the first N rows, so a whole pipeline stays fast to iterate on. It changes
  every block's cache key, so sampled and full results never mix; a banner
  keeps it obvious that what you're looking at isn't the full population.
  Switching back off needs the sources re-read.
- **Why a block is orange** — a stale block says what made it stale (a param
  it names, a code edit, a re-read source, sample mode), following a cascade
  back to the edit that started it rather than blaming its neighbour.
- **Undo/redo** — `Ctrl`/`Cmd-Z` and `Ctrl`/`Cmd-Shift-Z`, or the arrows in
  the top bar, over graph edits: adding, deleting, rewiring, renaming,
  retagging and param/code changes. Text fields and the code editor keep
  their own undo. Undo restores the graph, not what you've run: reverting a
  param lands back on a cache key that's already computed, so results you
  already have aren't thrown away.
- **Unsaved work** — once a project has a folder to save into, a dirty edit
  autosaves back to it a couple of seconds after you stop typing/dragging (a
  `•` on the Save button marks the brief window before that happens; the bar
  says "autosaved HH:MM:SS" once it has). Every edit is also snapshotted to
  `.modelmaker-cache/recovery.json` regardless -- crash recovery doesn't wait
  on the debounce, and it's what covers a project that's never been saved at
  all yet. After a crash or a closed browser, the app offers to restore it on
  next start. Autosave only ever writes to your project's own `model.json`;
  it never stages or commits anything -- git stays entirely manual, via the
  Git panel.
- **Editing during a run** — a run is pinned to the graph as it stood when
  it started, so you can keep editing (or undo, or toggle sample mode) while
  a long sweep executes. The run finishes against its own version; anything
  you changed in the meantime simply shows as stale afterwards, saying so.
- **Editing block code** — custom/AI-authored blocks use a real code editor
  (syntax highlighting, line numbers, bracket matching, `Ctrl-F` search);
  `Ctrl`/`Cmd-S` saves.

## Terminal UI

`modelmaker-tui` (installed by the `tui` extra, see [Install](#install)) is a
[Textual](https://textual.textualize.io/)-based terminal alternative to the
web UI, talking to the same backend over the same HTTP API. Lanes render as
titled, horizontally-scrolling rows of block chips (a free 2D canvas
doesn't map to a character grid at a readable size) colored by status; a
Wires panel lists connections as `from.port → to.port` rather than drawn
ASCII lines. Charts (`generate_image` blocks) render as ANSI plots via
[plotext](https://github.com/piccolomo/plotext) instead of matplotlib, and
can be saved to a `.txt`/`.html` file alongside the block's own PNG.

```bash
modelmaker-tui                          # spawns its own modelmaker-api, zero setup
modelmaker-tui projects/demo_pd_model   # ...and loads a project (a folder) on startup
modelmaker-tui --host 127.0.0.1 --port 8001  # attach to an already-running modelmaker-api instead
```

Standalone mode spawns a real `modelmaker-api` subprocess on a loopback
port rather than importing the backend in-process -- block execution
spawns its own `multiprocessing` subprocesses (see `runner.py`), and
running that inside the same process as the terminal driver risks fd-table
conflicts between the two. `--host`/`--port` skips spawning and attaches
to a server you already started (e.g. so the TUI and the web UI can drive
the same session live).

Key bindings (also shown in the footer): arrow keys move block selection;
`a` adds a block, `Delete` removes the focused block or wires-list row,
`w` starts/completes a wire (select the source block, `w`, select the
target block, `w` again); `r` runs the selected block, `Shift+R` runs to
it, `g`/`Ctrl+G` refresh the selected/all input sources, `Ctrl+R` runs
all, `x` cancels a run, `m` toggles sample mode; `t` tags a column's role;
`i` exports the selected block's chart/image; `c` compiles; `Ctrl+D`/
`Ctrl+F` draft with AI / suggest a fix for the selected block (`Ctrl+A`
applies the proposal, `Esc` discards it); `Ctrl+S` saves (the project, or
the block being edited if the inspector has focus), `Ctrl+O` opens,
`Ctrl+Z`/`Ctrl+Y` undo/redo.

## LLM provider

AI-assisted block drafting ("Draft with AI", suggest-a-fix) is pluggable:

| Provider | Value | Requirements |
|---|---|---|
| Claude Code CLI (default) | `claude_cli` | `claude` binary on `PATH`, logged in (`claude` or `claude /login`) |
| Anthropic API | `anthropic` | The `anthropic` extra (see [Install](#install)) and an API key (Settings, or `ANTHROPIC_API_KEY` / an `ant auth login` profile) |
| OpenAI API / OpenAI-compatible endpoint | `openai` | An API key (Settings, or `OPENAI_API_KEY`) and a model id. Defaults to `https://api.openai.com/v1`, but the base URL is configurable — point it at Groq, Together, OpenRouter, Fireworks, Azure OpenAI's `/openai` surface, or a self-hosted vLLM/text-generation-webui server instead |
| Google Gemini API | `gemini` | An API key (Settings, or `GEMINI_API_KEY` / `GOOGLE_API_KEY`) and a model id, e.g. `gemini-2.5-pro` |
| LM Studio (local) | `lmstudio` | LM Studio running with its local server started (Developer tab → Start Server) and a model loaded |
| Stub (no network, deterministic) | `stub` | none — used by default in tests |

The `openai` and `gemini` providers talk plain HTTPS via the standard
library, so they need no extra `pip install`.

The **Settings** menu in the app's top bar picks the active provider and,
per provider, its model, base URL (`lmstudio`/`openai`), and API key
(`anthropic`/`openai`/`gemini`) — "Fetch" queries the endpoint for the
models it has available, so you can pick one instead of typing an id by
hand. A "Include Polars reference examples in prompts" toggle there applies
to every provider; it's mainly useful for smaller/local models that know
Polars' shape but not its exact syntax.

These are in-memory server settings (no restart needed) that override the
env vars below when set — including API keys: a key entered in Settings is
held only in the running server process's memory, is never written to
disk, and is never sent back to the browser (the settings endpoint reports
only whether a key is currently set and whether it came from Settings or
an env var, never the value itself). Like every other setting here, it
does not survive a server restart — set the corresponding env var instead
for a value that should.

Env vars are still honored as defaults (useful for headless/CI use, or to
set a starting point before the server starts) -- copy
[`.env.example`](.env.example) to `.env` and fill in what you need; it's
loaded automatically on startup (and gitignored, so keys never get
committed) and a real exported environment variable always overrides it.
`MODELMAKER_LLM_PROVIDER`
picks the provider, `MODELMAKER_LLM_MODEL` pins a model, and
`MODELMAKER_LLM_INCLUDE_REFERENCE` (default on) controls the Polars
reference toggle. Per provider: `lmstudio` reads `MODELMAKER_LLM_BASE_URL`
(default `http://localhost:1234/v1`); `openai` reads `OPENAI_API_KEY` (or
`MODELMAKER_OPENAI_API_KEY`) and `MODELMAKER_OPENAI_BASE_URL`; `gemini`
reads `GEMINI_API_KEY` / `GOOGLE_API_KEY` (or `MODELMAKER_GEMINI_API_KEY`).

```bash
export MODELMAKER_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MODELMAKER_LLM_MODEL=gpt-5.1
modelmaker-api
```

## Project structure

```
modelmaker/
  api.py              FastAPI app and HTTP routes
  blocks/             Block registry (standard library, modelling, stat tests)
  llm/                Pluggable LLM providers for AI-assisted drafting
  graph.py            Graph/block/wire data model
  session.py          Mutable project session wrapping the graph + runner
  runner.py           Block execution + caching engine
  compiler.py         Compiles a graph to a single Python script
  packet.py           DataFramePacket: the typed value flowing over wires
  project.py          Load/save project files (JSON + block source files)
  static/             Built frontend bundle (generated, gitignored)
frontend/             React + Vite UI
tests/                Pytest suite for the backend
sample_data/          Example CSVs for building demo pipelines
```
