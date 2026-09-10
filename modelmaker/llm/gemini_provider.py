from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import JSON_ONLY_INSTRUCTIONS, build_user_prompt, contract_for, extract_json_response

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MAX_TOKENS = 4096


def list_models(base_url: str, api_key: str) -> list[str]:
    """Model ids the Gemini API reports for this key, used to populate the
    Settings panel's model picker. The key is passed as a query param, per
    the Gemini REST API (there is no bearer-token auth for this endpoint)."""
    url = f"{base_url.rstrip('/')}/models?key={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
        # Drop the "models/" prefix Gemini returns (e.g. "models/gemini-2.5-pro")
        # and keep only models that actually support text generation.
        return sorted(
            m["name"].removeprefix("models/")
            for m in data["models"]
            if "generateContent" in m.get("supportedGenerationMethods", [])
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"could not list models from {base_url} -- HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"could not reach {base_url} -- check network access") from e


@register_provider("gemini")
class GeminiProvider(LLMProvider):
    """Calls the Google Gemini API's `generateContent` endpoint over plain
    HTTPS (standard library only -- no extra package needed).

    Configure with an API key (Settings panel, or MODELMAKER_GEMINI_API_KEY
    / GEMINI_API_KEY / GOOGLE_API_KEY) and a model id, e.g. `gemini-2.5-pro`
    or `gemini-2.5-flash`. Neither is guessed: an incorrect default model
    could silently run (and bill) the wrong thing, so both must be set
    explicitly."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = (base_url or os.environ.get("MODELMAKER_GEMINI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("MODELMAKER_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.timeout_seconds = float(os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        self.max_tokens = int(os.environ.get("MODELMAKER_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL")
        if not self.api_key:
            raise RuntimeError(
                "no Gemini API key configured -- set one in Settings, or export "
                "GEMINI_API_KEY / GOOGLE_API_KEY / MODELMAKER_GEMINI_API_KEY"
            )
        if not self.model:
            raise RuntimeError(
                "no model configured for the gemini provider -- set one in Settings "
                "(use Fetch models to see what your key has available)"
            )

    def draft(self, ctx: DraftContext) -> DraftResult:
        system = contract_for(ctx.mode, ctx.include_reference) + "\n" + JSON_ONLY_INSTRUCTIONS
        payload = json.dumps(
            {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": build_user_prompt(ctx)}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json",
                },
            }
        ).encode()

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                response = json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Gemini API returned HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"could not reach {self.base_url} -- check network access") from e
        except TimeoutError as e:
            raise RuntimeError(
                f"Gemini API did not respond within {self.timeout_seconds:.0f}s -- set "
                "MODELMAKER_LLM_TIMEOUT_SECONDS to wait longer."
            ) from e

        try:
            candidate = response["candidates"][0]
            if candidate.get("finishReason") == "MAX_TOKENS" and not candidate.get("content", {}).get("parts"):
                raise RuntimeError(
                    f"hit the {self.max_tokens}-token cap before producing a reply -- set "
                    "MODELMAKER_LLM_MAX_TOKENS higher to give it more room."
                )
            text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"response missing expected fields: {response!r}") from e

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
