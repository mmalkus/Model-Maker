from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import JSON_ONLY_INSTRUCTIONS, build_user_prompt, contract_for, extract_json_response

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MAX_TOKENS = 4096


def list_models(base_url: str, api_key: str) -> list[str]:
    """Model ids the `/models` endpoint reports for this key -- works against
    the real OpenAI API and any OpenAI-compatible endpoint (Groq, Together,
    OpenRouter, Fireworks, a self-hosted vLLM/text-generation-webui server,
    ...) that implements the same surface, used to populate the Settings
    panel's model picker."""
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            data = json.load(resp)
        return sorted(m["id"] for m in data["data"])
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"could not list models from {base_url} -- HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"could not reach {base_url} -- check the base URL and network access") from e


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """Calls an OpenAI-compatible chat-completions endpoint over plain HTTPS
    (standard library only -- no `openai` package needed). Defaults to the
    real OpenAI API, but `base_url` can be pointed at any OpenAI-compatible
    endpoint instead: Groq, Together, OpenRouter, Fireworks, Azure OpenAI's
    `/openai` surface, or a self-hosted vLLM/text-generation-webui/LM Studio
    server -- anything that implements the same `/chat/completions` request
    shape.

    Configure with an API key (Settings panel, or MODELMAKER_OPENAI_API_KEY
    / OPENAI_API_KEY) and a model id. Neither is guessed: an incorrect
    default model could silently run (and bill) the wrong thing, so both
    must be set explicitly."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = (base_url or os.environ.get("MODELMAKER_OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("MODELMAKER_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = float(os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        self.max_tokens = int(os.environ.get("MODELMAKER_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL")
        if not self.api_key:
            raise RuntimeError(
                "no OpenAI API key configured -- set one in Settings, or export "
                "OPENAI_API_KEY / MODELMAKER_OPENAI_API_KEY"
            )
        if not self.model:
            raise RuntimeError(
                "no model configured for the openai provider -- set one in Settings "
                "(use Fetch models to see what your key/endpoint has available)"
            )

    def draft(self, ctx: DraftContext) -> DraftResult:
        system = contract_for(ctx.mode, ctx.include_reference) + "\n" + JSON_ONLY_INSTRUCTIONS
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": build_user_prompt(ctx)},
                ],
                "temperature": 0.2,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode()

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                response = json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"{self.base_url} returned HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"could not reach {self.base_url} -- check the base URL and network access") from e
        except TimeoutError as e:
            raise RuntimeError(
                f"{self.base_url} did not respond within {self.timeout_seconds:.0f}s -- set "
                "MODELMAKER_LLM_TIMEOUT_SECONDS to wait longer."
            ) from e

        try:
            choice = response["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"response missing expected fields: {response!r}") from e

        if choice.get("finish_reason") == "length" and not (text or "").strip():
            raise RuntimeError(
                f"hit the {self.max_tokens}-token cap before producing a reply -- set "
                "MODELMAKER_LLM_MAX_TOKENS higher to give it more room."
            )

        data = extract_json_response(text)

        try:
            code = data["code"]
            transform = dict(data["metadata_transform"])
        except KeyError as e:
            raise ValueError(f"response missing required field: {e}") from e

        if transform.get("kind") != "declared":
            transform = {"kind": transform.get("kind", "passthrough")}

        return DraftResult(
            code=code,
            metadata_transform=transform,
            explanation=data.get("explanation", ""),
            params=data.get("params", {}) or {},
        )
