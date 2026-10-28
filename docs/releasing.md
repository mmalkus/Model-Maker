# Releasing

One GitHub release ships both the Python package (PyPI) and the VS Code
extension (Visual Studio Marketplace and Open VSX). Both carry the same
version. `.github/workflows/publish.yml` publishes PyPI first, then the
extension, because the extension installs the package from PyPI. The design
is in [`vscode-extension-plan.md`](../vscode-extension-plan.md) §8.

## One-time setup (repository owner)

The PyPI side (the `pypi` environment and trusted publishing) already exists.
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

4. **GitHub environment.** Under *Settings → Environments*, add
   `vscode-marketplace` with secrets `VSCE_PAT` and `OVSX_PAT`, plus yourself
   as a required reviewer, so every extension publish waits for approval.

## Each release

1. On a branch, set the same version in `pyproject.toml` and
   `vscode/package.json`. In `vscode/`, `npm version X.Y.Z
   --no-git-tag-version` updates the lockfile too. Add a `vscode/CHANGELOG.md`
   entry. CI fails if the two versions differ.
   - An **odd minor** (0.3, 0.5, …) goes to the Marketplace's pre-release
     channel, and an even one to stable. The workflow picks the channel from
     the version.
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
      the Marketplace (switch to the pre-release version for an odd minor).
   2. Open an empty folder, create a virtual environment, and select it as
      the interpreter.
   3. Run `pip install quantology-modelmaker && modelmaker-demo .` in that
      environment, or let the extension offer the install when you open the
      project.
   4. Open `projects/demo_pd_model/model.json` with the canvas, then click
      **Run all**. Every block should turn green.
   5. If Copilot is signed in, ask agent mode to "describe this pipeline"
      (it should call `#mmGraph`).

## v0.3.0 release notes (draft)

> **Model-Maker for VS Code (pre-release).** Open a project's `model.json`
> as the canvas inside VS Code, and use GitHub Copilot as the AI that drafts
> blocks and runs AI builds. In Copilot Chat's agent mode, Copilot can read
> and edit the pipeline, asking before every edit. The canvas follows your
> VS Code theme. Install "Model-Maker" from the Marketplace (pre-release) or
> Open VSX, or the attached `.vsix`.
>
> **Python package.**
>
> - New `copilot` LLM provider, available when the backend runs inside VS
>   Code.
> - `GET /api/version`.
> - Read-only `/api/assistant/*` views.
> - Opt-in switches for embedding the backend: `MODELMAKER_AUTH_TOKEN`
>   requires a token on `/api/*`, and `MODELMAKER_CORS_ORIGIN_REGEX`
>   admits extra origins.
>
> With none of these set, `modelmaker-api` behaves exactly as in 0.2.0.
