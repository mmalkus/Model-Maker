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
    changed again or the server restarts."""

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
            slot.update({k: v for k, v in values.items() if v is not None})
