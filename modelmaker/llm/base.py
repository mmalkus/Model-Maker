from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    role: str
    # Summary statistics (see packet.ColumnStats) -- optional and unused by
    # "author"/"params_only" drafting, populated only for "analyze_data" (see
    # DraftContext.mode) so the model has more than names/dtypes to go on
    # without ever being handed actual row data.
    count: int | None = None
    null_count: int | None = None
    n_unique: int | None = None
    mean: float | None = None
    std: float | None = None
    min: Any = None
    max: Any = None


@dataclass
class DraftContext:
    """Everything a provider needs to draft or fix a block's code, or (see
    mode="analyze_data") to analyze one block's output columns instead.
    Column schemas are passed as plain metadata (names/dtypes/roles/stats)
    -- never row data -- matching the plan's "schema, not necessarily full
    data" rule for the AI-assisted fix action, and now the analysis one too."""

    instruction: str
    function_name: str
    input_ports: dict[str, list[ColumnInfo]]
    param_names: list[str] = field(default_factory=list)
    existing_code: str | None = None
    error: str | None = None
    # "author": drafting a full llm_authored function body (the original
    # flow). "params_only": the block's code is fixed (a standard/input/
    # output block from the registry) -- the model may only choose values
    # for its existing parameters, given the fixed source for context.
    # "analyze_data": not drafting anything -- `code`/`metadata_transform`
    # come back as fixed placeholders (see prompts.CONTRACT_ANALYZE_DATA),
    # `params` carries per-column tag suggestions instead of parameter
    # values, and `explanation` carries the write-up itself rather than a
    # one-sentence note. Reuses the same DraftContext/DraftResult shape (and
    # every existing provider's draft() unchanged) for an otherwise
    # unrelated task, rather than adding a second provider-facing method.
    # "rename": same reuse, for proposing a block's display name and its
    # output ports' names instead -- `params["name"]` is the block name,
    # `params[<port>]` each port's proposed name (see
    # prompts.CONTRACT_RENAME).
    mode: Literal["author", "params_only", "analyze_data", "rename"] = "author"
    fixed_source: str | None = None
    # Off by default: appends a condensed Polars API cheat sheet + worked
    # example to the system prompt. Costs extra tokens on every call, so it's
    # opt-in -- mainly useful to compensate smaller/local models' weaker
    # recall of a less-common library's exact API surface.
    include_reference: bool = False


@dataclass
class DraftResult:
    code: str = ""
    metadata_transform: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def draft(self, ctx: DraftContext) -> DraftResult: ...


LLM_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(name: str):
    def _reg(cls: type[LLMProvider]) -> type[LLMProvider]:
        cls.name = name
        LLM_PROVIDER_REGISTRY[name] = cls
        return cls

    return _reg


def get_provider(name: str | None = None, model: str | None = None, **overrides: Any) -> LLMProvider:
    """Instantiate a registered provider. Extra keyword overrides (e.g.
    base_url, include_reference) are only passed through to providers whose
    constructor actually declares them -- most providers just take `model`,
    so this lets callers (the Settings panel, via the API) pass a uniform
    set of overrides without every provider needing to accept and ignore
    params it has no use for."""
    provider_name = name or os.environ.get("MODELMAKER_LLM_PROVIDER", "claude_cli")
    cls = LLM_PROVIDER_REGISTRY.get(provider_name)
    if cls is None:
        raise ValueError(
            f"unknown LLM provider {provider_name!r}; available: {sorted(LLM_PROVIDER_REGISTRY)}"
        )
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in overrides.items() if k in accepted and v is not None}
    return cls(model=model, **kwargs)
