# Changelog

The extension's version follows the `quantology-modelmaker` Python package
it runs (see vscode-extension-plan.md §8.2).

## 0.3.2

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

## 0.3.1

The first stable release: install it from the Marketplace without opting
in to pre-releases. Releases are stable from here on.

- Needs `quantology-modelmaker` 0.3.1, whose **Run all** re-reads a data
  source after its settings change (for example when a column role is
  tagged), so a chat assistant's run no longer leaves the pipeline red.

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
