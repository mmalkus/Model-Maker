from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import CONTRACT, build_user_prompt

DEFAULT_MODEL = "claude-opus-5"

JSON_ONLY_INSTRUCTIONS = """
Respond with ONLY a single JSON object -- no markdown code fences, no prose \
before or after it. It must have exactly these keys:
{
  "code": "<the complete function definition as a string>",
  "metadata_transform": {
    "kind": "passthrough" | "narrow" | "declared",
    "base": "<input port name, only when kind is 'declared', else null>",
    "drops": ["<column name>", ...],
    "adds": [{"name": "...", "dtype": "...", "role": "..."}, ...]
  },
  "params": {"<param name>": <default value>, ...},
  "explanation": "<one or two sentences>"
}
Omit "drops"/"adds"/"base" (send empty list / null) when kind is not \
"declared". The response must be valid JSON parseable by a standard parser."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"claude CLI did not return parseable JSON: {text[:500]!r}")


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
        self.binary = shutil.which("claude")
        if not self.binary:
            raise RuntimeError("claude CLI not found on PATH; install Claude Code or choose a different LLM provider")

    def draft(self, ctx: DraftContext) -> DraftResult:
        system = CONTRACT + "\n" + JSON_ONLY_INSTRUCTIONS
        result = subprocess.run(
            [
                self.binary,
                "-p",
                build_user_prompt(ctx),
                "--output-format",
                "json",
                "--tools",
                "",
                "--model",
                self.model,
                "--system-prompt",
                system,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI exited {result.returncode}: {(result.stderr or result.stdout).strip()}")

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI produced non-JSON output: {result.stdout[:500]!r}") from e

        if payload.get("is_error"):
            raise RuntimeError(f"claude CLI error: {payload.get('result')}")

        data = _extract_json(payload["result"])

        try:
            code = data["code"]
            transform = dict(data["metadata_transform"])
        except KeyError as e:
            raise ValueError(f"claude CLI response missing required field: {e}") from e

        if transform.get("kind") != "declared":
            transform = {"kind": transform.get("kind", "passthrough")}

        return DraftResult(
            code=code,
            metadata_transform=transform,
            explanation=data.get("explanation", ""),
            params=data.get("params", {}) or {},
        )
