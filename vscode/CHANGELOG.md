# Changelog

The extension's version follows the `quantology-modelmaker` Python package
it runs (see vscode-extension-plan.md §8.2).

## 0.3.6

- **The canvas follows what assistants do.** When Copilot, Claude Code or
  the AI build adds blocks, the canvas zooms out to show them; when a run
  starts, it brings the running blocks into view (zooming out only as
  needed). Your own panning or zooming pauses this for a few seconds.
- **The AI build ends with a model document**: purpose and data, each stage
  in build order with the choices made and the numbers behind them,
  deviations, results per sample, limitations and next checks. Outside
  assistants are told to offer the same document as Markdown in the
  project folder.
- **Stage CSV as Parquet**, a new block: converts large text sources (a
  .csv/.txt, a .gz of one, or the text members of a .zip) to Parquet once,
  streaming straight from the archive in 8 MB blocks to keep memory low,
  and rebuilds only the parts whose source changed. Block-writing guidance
  now says to stage large text sources this way.

## 0.3.5

- **Cursor and Windsurf.** **Model-Maker: Set Up AI Agent Tools** registers
  the Model-Maker tools (`modelmaker-mcp`) with Cursor's or Windsurf's agent,
  which can't use the Copilot Chat tools. When an assistant opens or creates
  a project, the canvas now opens in the editor it works for, not always in
  VS Code. The extension also starts cleanly in editors without VS Code's
  language model API; the canvas's AI then uses a provider from its Settings.

## 0.3.4

- **Claude Code finds the open canvas again.** When VS Code started, the
  Python extension's interpreter report restarted Model-Maker's backend
  even in the managed environment, which doesn't use that interpreter.
  The canvas's first render could then finish after the restart and leave
  Claude Code (and the canvas) pointed at the backend that was stopped,
  so every Model-Maker tool answered "no Model-Maker canvas is open". Only
  backends that run on the selected interpreter restart now, and a render
  for a backend that was replaced meanwhile is dropped.

## 0.3.3

- **The block library opens in a tab of its own**, beside the canvas: the
  toolbar's new **Library** button, **Model-Maker: Open Block Library**, or
  the library icon in the editor title bar. Placing a block from it puts it
  on the canvas, in the canvas's undo history, and brings the canvas back.
- **Build with AI** is gone from the canvas in VS Code: the assistant beside
  it (Copilot Chat, Claude Code) builds models there.
- The Quantology mark is back on the toolbar, and **More** links to the
  Quantology site, GitHub and the license.
- **"model.json changed on disk"** only asks when the file really changed
  outside the canvas *and* the canvas has unsaved edits. A change with no
  unsaved edits reloads quietly; the canvas's own saves are recognised and
  ignored.
- **Set Up Claude Code** can register the tools for all your projects
  (`claude mcp add --scope user`), not just the workspace's `.mcp.json`.
- Blocks drafted by AI read large sources lazily and reduce them with
  Polars' streaming engine, instead of loading a table bigger than memory.
- New chat tools `get_block_guide` and `check_block_file` (Copilot and
  Claude Code): an assistant writing a block file for your library gets the
  rules and template first, then validates the file and runs its examples.
  Your library's data-source blocks can now be placed from chat too.

## 0.3.2

The first stable release: install it from the Marketplace without opting
in to pre-releases. Releases are stable from here on. (0.3.1 was never
released; its changes are here.)

- **Model-Maker ships inside the extension.** No `quantology-modelmaker`
  package to install: the extension runs its own copy, which always
  matches its version.
- **Its own Python environment.** The first time a canvas opens, the
  extension makes a Python environment in its storage folder, from any
  Python 3.11+ on the machine. It installs Model-Maker's open-source
  requirements there, from PyPI or the company mirror pip is configured for.
  Your other environments are never touched.
- `modelmaker.python.environment`: `managed` (the default), `interpreter`
  (your interpreter, with the bundled Model-Maker; missing requirements are
  offered for install) or `installedPackage` (the previous behaviour, for
  developing Model-Maker). New `modelmaker.basePython` and **Model-Maker:
  Repair Python Environment**.
- **Set Up Claude Code** points `.mcp.json` at the same Python and
  Model-Maker, at paths that survive extension updates.
- **Run all** re-reads a data source after its settings change (for example
  when a column role is tagged), so a chat assistant's run no longer leaves
  the pipeline red.

## 0.3.0 (pre-release)

The first release, on the Marketplace's pre-release channel. Needs
`quantology-modelmaker` 0.3.0 or newer, the first version with the backend
switches the extension uses.

- The Model-Maker canvas as a custom editor for a project's `model.json`.
- One local, token-protected backend per open project, started from the
  selected Python interpreter. The extension offers to install the package.
- GitHub Copilot as an AI provider, through a relay over VS Code's Language
  Model API.
- Native dialogs for saving, picking paths, confirmations and downloads.
  Compiled scripts open in an editor tab.
- The canvas follows the VS Code colour theme (light, dark, high contrast).
- Copilot Chat tools that can build a model from an empty canvas: open or
  create a project, find data files, add lanes, add, configure, rename,
  wire, unwire and delete blocks, tag column roles, run, check for holdout
  leakage and suspicious features (with the AI builder's decision hints),
  and compile.
- Claude Code can use the same tools on the canvas you used last, through
  `modelmaker-mcp` from the Python package. **Model-Maker: Set Up Claude
  Code** adds it to the workspace's `.mcp.json`.
- A compact canvas inside VS Code: one toolbar row with the rarer run modes
  under **More**, a palette that folds to a rail, and an inspector that
  appears only for a selected block.
