# Changelog

The extension's version follows the `quantology-modelmaker` Python package
it runs (see vscode-extension-plan.md §8.2).

## 0.3.0 (pre-release)

The first release. Odd minor versions go to the Marketplace's pre-release
channel; 0.4.0 is planned as the first stable one. Needs
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
- Copilot Chat tools: read the graph, the block catalogue and output
  summaries, and add, configure, wire, run and compile blocks.
