from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import JSON_ONLY_INSTRUCTIONS, build_user_prompt, contract_for, extract_json_response

_TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT_SECONDS = 180
# Reasoning models (e.g. Qwen3.5) spend a chunk of this on an unstructured
# chain-of-thought before ever emitting the JSON -- capped so a rambling
# local model fails fast instead of hanging.
DEFAULT_MAX_TOKENS = 4096


def list_models(base_url: str) -> list[str]:
    """Model ids LM Studio's OpenAI-compatible /v1/models endpoint currently
    reports (downloaded models it can serve, not just the one already
    loaded) -- used both to auto-detect a model when none is configured and
    to populate the Settings panel's model picker."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
        return [m["id"] for m in data["data"]]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        raise RuntimeError(
            f"could not reach LM Studio at {base_url} -- is the local server running "
            "(LM Studio > Developer > Start Server)?"
        ) from e


@register_provider("lmstudio")
class LMStudioProvider(LLMProvider):
    """Calls a local LM Studio server's OpenAI-compatible chat-completions
    endpoint. Start the server in LM Studio (Developer tab -> Start Server)
    and load a model first. Uses only the standard library, so it needs no
    extra dependency. Configure with MODELMAKER_LLM_BASE_URL (default
    http://localhost:1234/v1) and MODELMAKER_LLM_MODEL (default: whichever
    model LM Studio currently has loaded).

    Defaults to including prompts.POLARS_REFERENCE on every call: small
    local models tend to know the shape of the Polars API but not its exact
    syntax (e.g. dropping the parens `.alias()` needs around a comparison),
    and the reference reliably fixes that. Set MODELMAKER_LLM_INCLUDE_REFERENCE
    to a falsy value (0/false/no/off) to turn it back off and save tokens."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        include_reference: bool | None = None,
    ):
        self.base_url = (base_url or os.environ.get("MODELMAKER_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout_seconds = float(os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        self.max_tokens = int(os.environ.get("MODELMAKER_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        if include_reference is None:
            include_reference = os.environ.get("MODELMAKER_LLM_INCLUDE_REFERENCE", "true").strip().lower() in _TRUTHY
        self.include_reference = include_reference
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL") or self._detect_loaded_model()

    def _detect_loaded_model(self) -> str:
        try:
            return list_models(self.base_url)[0]
        except IndexError as e:
            raise RuntimeError(
                f"LM Studio at {self.base_url} has no models loaded -- load one in the "
                "Developer tab, or set MODELMAKER_LLM_MODEL to skip auto-detection."
            ) from e

    def draft(self, ctx: DraftContext) -> DraftResult:
        system = contract_for(ctx.mode, ctx.include_reference or self.include_reference) + "\n" + JSON_ONLY_INSTRUCTIONS
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": build_user_prompt(ctx)},
                ],
                "temperature": 0.2,
                "max_tokens": self.max_tokens,
            }
        ).encode()

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                response = json.load(resp)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"could not reach LM Studio server at {self.base_url} -- is the local "
                "server running (LM Studio > Developer > Start Server) with a model loaded?"
            ) from e
        except TimeoutError as e:
            raise RuntimeError(
                f"LM Studio did not respond within {self.timeout_seconds:.0f}s -- local "
                "generation can be slow, especially on the first request while the model "
                "loads. Set MODELMAKER_LLM_TIMEOUT_SECONDS to wait longer."
            ) from e

        try:
            choice = response["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"LM Studio response missing expected fields: {response!r}") from e

        if choice.get("finish_reason") == "length" and not text.strip():
            raise RuntimeError(
                f"LM Studio hit the {self.max_tokens}-token cap before producing a reply -- "
                "the model spent it all on chain-of-thought reasoning. Set "
                "MODELMAKER_LLM_MAX_TOKENS higher to give it more room."
            )

        data = extract_json_response(text)

        try:
            code = data["code"]
            transform = dict(data["metadata_transform"])
        except KeyError as e:
            raise ValueError(f"LM Studio response missing required field: {e}") from e

        if transform.get("kind") != "declared":
            transform = {"kind": transform.get("kind", "passthrough")}

        return DraftResult(
            code=code,
            metadata_transform=transform,
            explanation=data.get("explanation", ""),
            params=data.get("params", {}) or {},
        )
