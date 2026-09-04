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


@dataclass
class DraftContext:
    """Everything a provider needs to draft or fix a block's code. Column
    schemas are passed as plain metadata (names/dtypes/roles) -- never row
    data -- matching the plan's "schema, not necessarily full data" rule for
    the AI-assisted fix action."""

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
    mode: Literal["author", "params_only"] = "author"
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
