# Releasing

One GitHub release ships both the Python package (PyPI) and the VS Code
extension (Visual Studio Marketplace and Open VSX). Both carry the same
version. `.github/workflows/publish.yml` publishes PyPI first, then the
extension, because the extension installs the package from PyPI. The design
is in [`vscode-extension-plan.md`](../vscode-extension-plan.md) §8.

## One-time setup (repository owner)

The PyPI side (the `pypi` environment and trusted publishing) already exists.
Every release attaches the `.vsix` to the GitHub release. Each marketplace is
optional: the workflow publishes to one only once its token is set, and says
in a warning which ones it skipped. Open VSX needs only a GitHub sign-in. The
Visual Studio Marketplace token comes from Azure DevOps, and a new Azure
DevOps organization has to be linked to an Azure subscription (pay-as-you-go
is enough; Azure DevOps itself is free for small teams).

The extension needs:

1. **Marketplace publisher.** Sign in at
   <https://marketplace.visualstudio.com/manage> and create the publisher
   `quantology`. It must match `publisher` in `vscode/package.json`, and a
   published extension can't move to another publisher, so settle the name
   first.
2. **Marketplace token.** In Azure DevOps (<https://dev.azure.com>, any
   organization), create a personal access token with organization
   **All accessible organizations** and scope **Marketplace → Manage**.
3. **Open VSX.** Sign in at <https://open-vsx.org> with GitHub, sign the
   publisher agreement, create an access token, then create the namespace
   once:

   ```bash
   npx ovsx create-namespace quantology -p <open-vsx-token>
   ```

   Because `quantology` is also a Visual Studio Marketplace publisher, Open
   VSX keeps published versions inactive (not listed or installable) until
   the namespace's ownership is verified. Use **Claim Ownership** on the
   extension's page in your Open VSX settings, which opens an issue on
   `EclipseFdn/open-vsx.org`. Versions published before the approval become
   active once it's granted, with no new upload.

4. **GitHub environment.** Under *Settings → Environments*, add
   `vscode-marketplace` with secrets `VSCE_PAT` and/or `OVSX_PAT`, plus
   yourself as a required reviewer, so every extension publish waits for
   approval. After adding a token, **Re-run failed jobs** (or re-run the
   `publish-vscode` job) on the last release's run publishes that release's
   extension, with no new release. Both publish steps pass
   `--skip-duplicate`, so a marketplace that already has the version is
   left alone instead of failing the job.

## Each release

1. On a branch, set the same version in `pyproject.toml` and
   `vscode/package.json`. In `vscode/`, `npm version X.Y.Z
   --no-git-tag-version` updates the lockfile too. Add a `vscode/CHANGELOG.md`
   entry. CI fails if the two versions differ.
   - The extension is published as **stable**. To publish one to the
     Marketplace's pre-release channel instead, tick **Set as a
     pre-release** when creating the GitHub release. PyPI gets the version
     either way. A version can't move between channels later, so a fix for a
     pre-release ships as a new version.
   - The `.vsix` carries the package's own code (`vscode/python/`, made by
     `vscode/scripts/bundle-python.py` in CI from the same commit), so the
     extension never depends on the PyPI upload.
   - If the release changes the HTTP API in a way older clients can't
     handle, bump `API_COMPAT_LEVEL` in `modelmaker/api.py` and `API_COMPAT`
     in `vscode/src/backend.ts` together. Also raise the minimum version in
     `modelmaker.packageSpec`'s default (`vscode/package.json` and
     `DEFAULT_SPEC` in `backend.ts`).
2. Merge to `main` with CI green, including the `VS Code extension` job.
3. Create a GitHub release with the tag `vX.Y.Z`. The tag must match the
   version.
4. `publish.yml` then builds the wheel and publishes it to PyPI, and builds
   the `.vsix`. Once you approve the `vscode-marketplace` environment, it
   publishes the extension to both marketplaces and attaches the `.vsix` to
   the release, for machines that can install only from a file (`code
   --install-extension modelmaker.vsix`).
5. **Smoke test** in a clean profile:
   1. Run `code --profile release-check`, and in it install Model-Maker from
      the Marketplace (switch to the pre-release version if the release was
      one).
   2. Open an empty folder, create a virtual environment, and select it as
      the interpreter.
   3. Run `pip install quantology-modelmaker && modelmaker-demo .` in that
      environment, or let the extension offer the install when you open the
      project.
   4. Open `projects/demo_pd_model/model.json` with the canvas, then click
      **Run all**. Every block should turn green.
   5. If Copilot is signed in, ask agent mode to "describe this pipeline"
      (it should call `#mmGraph`).

## v0.3.3 release notes (draft)

> **Model-Maker 0.3.3**
>
> **VS Code.**
>
> - The block library opens in a tab of its own, beside the canvas (the
>   toolbar's **Library** button). Placing a block from it puts it on the
>   canvas.
> - **Build with AI** no longer shows on the canvas in VS Code: Copilot Chat
>   or Claude Code beside it builds models there.
> - The "model.json changed on disk" prompt only appears when reloading
>   would discard unsaved edits; otherwise the canvas reloads quietly, and
>   its own saves never trigger it.
> - **Set Up Claude Code** can register the Model-Maker tools for all your
>   projects.
>
> **Large data.** Blocks drafted by AI (Create with AI, Fix with AI, the AI
> builder) now read large sources lazily (`pl.scan_csv`/`scan_parquet`),
> select only the columns they need, reduce early and collect with Polars'
> streaming engine, so a table bigger than memory never has to be loaded
> whole. The block-authoring guide has a new "Large data" section. Chat
> assistants get the same rules, plus the block template, from two new
> tools: `get_block_guide` before writing a block file for your library, and
> `check_block_file` to validate it and run its examples. Your library's
> data-source blocks can now be placed from chat too.
>
> **Python package.** `GET /api/project/disk_state` says whether the
> project file changed on disk since the backend last read or wrote it.

## v0.3.2 release notes

0.3.1 was never released: everything prepared for it ships here.

> **Model-Maker 0.3.2**
>
> **VS Code: nothing to install but Python.** Model-Maker now ships inside
> the extension, which runs it in a Python environment of its own. That
> environment is made from any Python 3.11+ on the machine, and holds only
> Model-Maker's open-source requirements (polars, scikit-learn, scipy,
> numpy, matplotlib, fastapi and a few more), installed from PyPI or the
> company mirror pip is configured for. No `quantology-modelmaker` package
> is installed, and other Python environments aren't touched. The setting
> `modelmaker.python.environment` can switch to your own interpreter
> instead.
>
> **The extension is now stable.** Install it from the Marketplace without
> opting in to pre-releases.
>
> **Run all re-reads a data source whose settings changed** since it was
> read: an edited param, a tagged column role, sample mode toggled. Before,
> it left the source stale, and the blocks after it failed with "input block
> is not green". A source whose settings are unchanged is still re-read only
> by **Refresh sources**. The streaming Run all does the same.

## v0.3.0 release notes

Paste the PyPI part into the GitHub release; the extension's own notes are in
[`vscode/CHANGELOG.md`](../vscode/CHANGELOG.md).

> **Model-Maker 0.3.0**
>
> **Upgrading from 0.2:** a project that contains a canvas-only custom
> block won't load. Custom blocks now live in the block library. Recreate
> the block there and place it again. Projects with only built-in blocks
> load as before and are saved in the new format (`modelmaker_version: 2`).
>
> **Model-Maker for VS Code (pre-release).** Open a project's `model.json`
> as the canvas in a VS Code editor tab, laid out compactly for it. GitHub
> Copilot can be the AI that drafts blocks and runs AI builds. In Copilot
> Chat's agent mode, Copilot can build a model from an empty canvas: create
> the project, find the data, lay out lanes, add and wire blocks, run them
> and check the result for leakage, asking before every edit. Claude Code
> gets the same tools through `modelmaker-mcp` (**Model-Maker: Set Up
> Claude Code**). Install
> "Model-Maker" from the Marketplace (pre-release) or Open VSX, or use the
> `.vsix` attached to this release.
>
> **Block library.**
>
> - Custom blocks are files in your block library (`~/.modelmaker/blocks`,
>   or `MODELMAKER_BLOCK_LIBRARY`), which can be a git repo or a shared
>   drive. The palette has a library section, and a library panel browses,
>   edits, imports and exports them.
> - **Create with AI** drafts a library block and checks it with the block
>   validator before saving it.
> - A placed library block shows when a newer version is available.
>   **Update** shows the code diff and what changes in the graph before
>   applying it.
> - `modelmaker check-blocks` validates a library folder and runs the
>   blocks' examples. See `docs/block-authoring.md`.
> - Blocks carry tags, and the palette can filter by tag.
>
> **AI builder.**
>
> - Builds work in stages. The AI first plans an outline, then plans and
>   builds one stage at a time, with a review after each stage.
> - Guards: holdout leakage in the plan or the wiring is refused, an
>   outline must split the data before anything learns from the target,
>   and a feature with a suspicious IV is flagged to you.
> - Optional decision hints and a small-context mode, for smaller models.
> - Builds can use your library blocks. Custom block creation can be
>   switched off.
> - Questions from the build have clickable answers. A build pauses when it
>   hits a limit, and an ended build can be continued.
> - More accurate token and cost accounting.
>
> **Blocks.**
>
> - Display blocks show their result right on the canvas node.
> - New: forward default flag, for a PD target from a monthly panel.
> - Join offers polars' join strategies.
>
> **Execution.**

> - Block code never runs in the API process. Runs happen in worker
>   processes with a cleaned-up environment, and the result cache holds
>   data only.
> - Per-task memory and thread limits, with clear errors when a block hits
>   them.
>
> **LLM providers.**
>
> - New `copilot` provider, available when the backend runs inside VS
>   Code.
> - Connections by kind: set up an OpenAI-compatible service such as Azure
>   OpenAI, OpenRouter or an internal gateway under its own name, and list
>   the models users may pick.
>
> **Hosting Model-Maker for several users (for embedding applications).**
>
> - Several isolated workspaces in one process, each with its own project,
>   runs, settings, cache and optional folder root (paths are shown relative
>   to it).
> - Restricted mode: switch off custom block code, the AI builder, AI
>   drafting, LLM settings, environment variables or git per workspace.
> - Datasets, sources and exports providers in place of file paths, and
>   block capabilities that hide blocks a workspace can't use.
> - A runner agent and `RemoteExecutor`, to run block tasks in a separate,
>   sandboxed process behind a socket.
>
> **Claude Code and other MCP clients.** `modelmaker-mcp` gives an
> assistant the tools to build a model on the open canvas, in VS Code or
> the browser app (`--url`). The canvas shows edits made from outside
> within a couple of seconds.
>
> **Embedding API (opt-in).**
>
> - `GET /api/version`.
> - `/api/assistant/*`: read-only views of the graph, block catalogue and
>   redacted output summaries, a leakage and hints check, and block and lane
>   placement for chat assistants. `GET /api/graph/stamp` tells a canvas
>   the graph changed.
> - `MODELMAKER_AUTH_TOKEN` requires a token on `/api/*`, and
>   `MODELMAKER_CORS_ORIGIN_REGEX` admits extra origins.
>
> With none of the hosting or embedding switches set, `modelmaker-api` runs
> as a single local app, as in 0.2.0.
