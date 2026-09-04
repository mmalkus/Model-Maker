# Model-Maker

A SAS Enterprise Guide–style visual pipeline builder for financial model
development in Python. Blocks contain functional Python code, are wired
together via Polars DataFrames carrying metadata, organized visually into
phase-labelled swimlanes, and compile down to a single, human-readable,
self-contained Python script.

See [`modelmaker-v2-plan.md`](modelmaker-v2-plan.md) for the full design.

The project has two parts:

- **`modelmaker/`** — the Python backend: block registry, graph/session
  model, execution runner, single-file compiler, and a FastAPI HTTP API.
- **`frontend/`** — a React + Vite UI for building and running pipelines
  against that API.

## Requirements

- Python >= 3.11
- Node.js >= 18 (for the frontend)
- (Optional) the [Claude Code](https://claude.com/claude-code) CLI (`claude`)
  on `PATH` and logged in, for AI-assisted block drafting — this is the
  default LLM provider. See [LLM provider](#llm-provider) below for
  alternatives.

## Install

Install the backend as an editable package:

```bash
pip install -e ".[dev]"
```

`[dev]` pulls in `pytest`/`httpx` for running the test suite. There's also
an optional `anthropic` extra if you want to call the Anthropic API
directly instead of shelling out to the `claude` CLI:

```bash
pip install -e ".[anthropic]"
```

Install the frontend dependencies:

```bash
cd frontend
npm install
```

## Running it

Run the backend and frontend in two terminals.

**Backend** (FastAPI, defaults to `http://127.0.0.1:8001`):

```bash
modelmaker-api
```

This is the console script installed by `pip install -e .`; it's
equivalent to `python -m modelmaker.api`. For auto-reload during
development, run uvicorn directly instead:

```bash
uvicorn modelmaker.api:app --reload --port 8001
```

Override the host/port with `MODELMAKER_HOST` / `MODELMAKER_PORT`
environment variables.

**Frontend** (Vite dev server on `http://localhost:5173`, proxies `/api`
requests to the backend on port 8001):

```bash
cd frontend
npm run dev
```

Then open `http://localhost:5173` in a browser.

## LLM provider

AI-assisted block drafting ("Draft with AI", suggest-a-fix) is pluggable
via the `MODELMAKER_LLM_PROVIDER` environment variable:

| Provider | Value | Requirements |
|---|---|---|
| Claude Code CLI (default) | `claude_cli` | `claude` binary on `PATH`, logged in (`claude` or `claude /login`) |
| Anthropic API | `anthropic` | `pip install -e ".[anthropic]"` and `ANTHROPIC_API_KEY` set (or an `ant auth login` profile) |
| Stub (no network, deterministic) | `stub` | none — used by default in tests |

Optionally pin the model with `MODELMAKER_LLM_MODEL`.

```bash
export MODELMAKER_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-...
modelmaker-api
```

## Tests

```bash
pytest
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
frontend/             React + Vite UI
tests/                Pytest suite for the backend
sample_data/          Example CSVs for building demo pipelines
```
