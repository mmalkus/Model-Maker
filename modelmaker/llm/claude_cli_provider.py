from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import JSON_ONLY_INSTRUCTIONS, build_user_prompt, contract_for, extract_json_response

# A CLI alias rather than a pinned id: the CLI resolves it to the newest
# Sonnet the login can run, so the default never goes stale.
DEFAULT_MODEL = "sonnet"
# A library-block draft is a whole block (code, helpers, _example_ and its
# BLOCKS entry) -- several thousand output tokens, which takes minutes
# rather than seconds for a broad request.
CLI_TIMEOUT_SECONDS = 600


@register_provider("claude_cli")
class ClaudeCliProvider(LLMProvider):
    """Shells out to the locally installed `claude` (Claude Code) CLI in
    non-interactive print mode. Uses whatever the CLI is already
    authenticated with -- a Claude subscription (Pro/Max) login or an API
    key -- so it works without a separate ANTHROPIC_API_KEY. Requires the
    `claude` binary on PATH and a logged-in session (`claude` interactively,
    or `claude /login`)."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = float(os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS", CLI_TIMEOUT_SECONDS))
        self.binary = shutil.which("claude")
        if not self.binary:
            raise RuntimeError("claude CLI not found on PATH; install Claude Code or choose a different LLM provider")

    def draft(self, ctx: DraftContext) -> DraftResult:
        system = contract_for(ctx.mode, ctx.include_reference) + "\n" + JSON_ONLY_INSTRUCTIONS
        # Prompt via stdin and system prompt via a file: as argv they can
        # exceed Windows' 32K command-line limit (WinError 206).
        fd, system_path = tempfile.mkstemp(prefix="modelmaker-system-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(system)
        try:
            result = subprocess.run(
                [
                    self.binary,
                    "-p",
                    "--output-format",
                    "json",
                    "--tools",
                    "",
                    "--model",
                    self.model,
                    "--system-prompt-file",
                    system_path,
                ],
                input=build_user_prompt(ctx),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            # An unknown or unavailable model fails fast (is_error, below),
            # so a timeout is a long draft -- or a hung CLI.
            raise RuntimeError(
                f"claude CLI did not respond within {e.timeout:.0f}s using model "
                f"{self.model!r}. A broad request can take minutes to draft -- set "
                "MODELMAKER_LLM_TIMEOUT_SECONDS to wait longer, or narrow the request."
            ) from e
        finally:
            try:
                os.unlink(system_path)
            except OSError:
                pass
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI exited {result.returncode}: {(result.stderr or result.stdout).strip()}") from e
            raise RuntimeError(f"claude CLI produced non-JSON output: {result.stdout[:500]!r}") from e

        # Errors (e.g. a model this login can't run) still come back as a
        # JSON result, with a readable message, alongside a nonzero exit.
        if payload.get("is_error") or result.returncode != 0:
            raise RuntimeError(f"claude CLI error (model {self.model!r}): {payload.get('result') or result.stderr.strip()}")

        data = extract_json_response(payload["result"])

        try:
            code = data["code"]
        except KeyError as e:
            raise ValueError(f"claude CLI response missing required field: {e}") from e
        block = data.get("block") or {}

        return DraftResult(
            code=code,
            block=block if isinstance(block, dict) else {},
            explanation=data.get("explanation", ""),
            params=data.get("params", {}) or {},
        )
