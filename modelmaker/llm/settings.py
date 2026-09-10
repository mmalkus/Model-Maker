from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMSettingsStore:
    """UI-editable LLM settings for the running server, kept in memory only
    (like ProjectSession -- nothing here persists across a restart). Every
    field defaults to "unset" (None/empty): until the Settings panel changes
    something, each provider falls back to its own constructor defaults
    (env vars, then hardcoded constants), exactly as before this store
    existed. Setting a field here overrides that provider's default until
    changed again or the server restarts.

    This includes API keys (per_provider[name]["api_key"]): they live only
    in this process's memory, are never written to disk, and the API layer
    (see api.py's _effective_llm_settings) never echoes a raw key value back
    to the client -- only whether one is currently set and where it came
    from (an explicit override here vs. an environment variable)."""

    active_provider: str | None = None
    per_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Global include-Polars-reference toggle (see DraftContext.include_reference),
    # applied across every provider's draft calls. None = each provider's own default.
    include_reference: bool | None = None

    def for_provider(self, name: str) -> dict[str, Any]:
        return self.per_provider.get(name, {})

    def update(self, active_provider: str | None, include_reference: bool | None, settings: dict[str, dict[str, Any]] | None) -> None:
        if active_provider is not None:
            self.active_provider = active_provider
        if include_reference is not None:
            self.include_reference = include_reference
        for name, values in (settings or {}).items():
            slot = self.per_provider.setdefault(name, {})
            for key, value in values.items():
                # An explicit null clears a previously-set override (e.g. an
                # emptied text field, or "Remove key") so the provider falls
                # back to its own default (env var / hardcoded constant)
                # instead of being stuck with a stale value.
                if value is None:
                    slot.pop(key, None)
                else:
                    slot[key] = value
