from __future__ import annotations

import os
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import build_user_prompt, contract_for

DEFAULT_MODEL = "claude-opus-5"


class _ColumnAdd(BaseModel):
    name: str
    dtype: str
    role: Literal["id", "target", "weight", "feature", "date", "segment", "excluded", "unassigned"] = "unassigned"


class _MetadataTransformDraft(BaseModel):
    kind: Literal["passthrough", "narrow", "declared"]
    base: str | None = Field(default=None, description="Input port name; required when kind is 'declared'.")
    drops: list[str] = Field(default_factory=list)
    adds: list[_ColumnAdd] = Field(default_factory=list)


class _BlockDraft(BaseModel):
    code: str = Field(description="The complete function definition, including the `def` line.")
    metadata_transform: _MetadataTransformDraft
    params: dict[str, str | float | bool | int] = Field(
        default_factory=dict, description="Suggested default values for any extra parameters the function takes."
    )
    explanation: str


class _ParamsDraft(BaseModel):
    params: dict[str, str | float | bool | int] = Field(
        default_factory=dict, description="Chosen values for the block's existing parameters."
    )
    explanation: str


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """Calls the Anthropic Messages API directly via the `anthropic` SDK.
    Requires an API key -- from the Settings panel, ANTHROPIC_API_KEY, or an
    `ant auth login` profile, in that order."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL", DEFAULT_MODEL)
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def draft(self, ctx: DraftContext) -> DraftResult:
        if ctx.mode == "params_only":
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=1024,
                system=contract_for("params_only"),
                messages=[{"role": "user", "content": build_user_prompt(ctx)}],
                output_format=_ParamsDraft,
            )
            draft = response.parsed_output
            return DraftResult(params=draft.params, explanation=draft.explanation)

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=contract_for(ctx.mode, ctx.include_reference),
            messages=[{"role": "user", "content": build_user_prompt(ctx)}],
            output_format=_BlockDraft,
        )
        draft = response.parsed_output
        transform = draft.metadata_transform.model_dump(exclude_none=True)
        if draft.metadata_transform.kind != "declared":
            transform.pop("base", None)
            transform.pop("drops", None)
            transform.pop("adds", None)
        return DraftResult(code=draft.code, metadata_transform=transform, explanation=draft.explanation, params=draft.params)
