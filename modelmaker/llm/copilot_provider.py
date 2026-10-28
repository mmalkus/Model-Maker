from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import register_provider
from .openai_provider import OpenAIProvider

UNAVAILABLE = (
    "GitHub Copilot is only available when Model-Maker runs inside VS Code "
    "(the Model-Maker extension relays it) -- pick another provider in Settings"
)


def relay() -> tuple[str, str] | None:
    """The VS Code extension's Copilot relay -- an OpenAI-compatible endpoint
    over vscode.lm -- as (base_url, token), or None outside VS Code."""
    base_url = os.environ.get("MODELMAKER_COPILOT_BASE_URL")
    token = os.environ.get("MODELMAKER_COPILOT_TOKEN")
    if not base_url or not token:
        return None
    return base_url.rstrip("/"), token


def relay_models(base_url: str, token: str) -> list[dict[str, Any]]:
    """The relay's /models entries -- {"id", "name", "max_input_tokens"} --
    in VS Code's preference order (unlike the openai provider's list_models,
    which sorts), so the first is a sensible default."""
    request = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return list(json.load(resp).get("data") or [])
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"the Copilot relay returned HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(f"could not reach the Copilot relay at {base_url} -- restart the backend from VS Code") from e


def default_model(base_url: str, token: str) -> str | None:
    """MODELMAKER_COPILOT_MODEL, else the first model the relay offers."""
    model = os.environ.get("MODELMAKER_COPILOT_MODEL")
    if model:
        return model
    models = relay_models(base_url, token)
    return models[0]["id"] if models else None


@register_provider("copilot")
class CopilotProvider(OpenAIProvider):
    """GitHub Copilot's models, through the VS Code extension's relay. The
    relay speaks the OpenAI wire format, so this is the openai provider
    pointed at it; outside VS Code it reports itself unavailable, the same
    way a key-based provider reports a missing key."""

    def __init__(self, model: str | None = None):
        endpoint = relay()
        if endpoint is None:
            raise RuntimeError(UNAVAILABLE)
        base_url, token = endpoint
        model = model or default_model(base_url, token)
        if not model:
            raise RuntimeError("the Copilot relay offers no models -- check that GitHub Copilot is signed in in VS Code")
        super().__init__(model=model, base_url=base_url, api_key=token)
