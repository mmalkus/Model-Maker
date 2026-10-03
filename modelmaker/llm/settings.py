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
    # The AI builder's two LLMs (see /agent-builder-proposal.md §9.1), each
    # {"provider": ..., "model": ...} with either key optional -- unset
    # means the active provider and that provider's configured model.
    agent_plan: dict[str, Any] = field(default_factory=dict)
    agent_build: dict[str, Any] = field(default_factory=dict)
    # The AI builder's small-context mode (BuildOptions.small_context): a
    # fresh conversation per stage, for models with a small context window.
    agent_small_context: bool = False
    # The AI builder's decision hints (BuildOptions.decision_hints).
    agent_decision_hints: bool = False

    def for_provider(self, name: str) -> dict[str, Any]:
        return self.per_provider.get(name, {})

    def update(
        self,
        active_provider: str | None,
        include_reference: bool | None,
        settings: dict[str, dict[str, Any]] | None,
        agent_plan: dict[str, Any] | None = None,
        agent_build: dict[str, Any] | None = None,
        agent_small_context: bool | None = None,
        agent_decision_hints: bool | None = None,
    ) -> None:
        if agent_small_context is not None:
            self.agent_small_context = agent_small_context
        if agent_decision_hints is not None:
            self.agent_decision_hints = agent_decision_hints
        if active_provider is not None:
            self.active_provider = active_provider
        if include_reference is not None:
            self.include_reference = include_reference
        for slot, values in ((self.agent_plan, agent_plan), (self.agent_build, agent_build)):
            for key, value in (values or {}).items():
                if value in (None, ""):
                    slot.pop(key, None)
                else:
                    slot[key] = value
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
