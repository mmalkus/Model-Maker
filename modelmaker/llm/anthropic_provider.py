from __future__ import annotations

import json
import os

import anthropic
from pydantic import BaseModel, Field

from .base import DraftContext, DraftResult, LLMProvider, register_provider
from .prompts import build_user_prompt, contract_for

DEFAULT_MODEL = "claude-opus-5"


class _BlockDraft(BaseModel):
    code: str = Field(description="Python source: the block function, any _helpers, and its _example_.")
    block_json: str = Field(description="The block's BLOCKS entry, as a JSON object.")
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
        try:
            block = json.loads(draft.block_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"the model's BLOCKS entry isn't valid JSON: {e}") from e
        return DraftResult(code=draft.code, block=block if isinstance(block, dict) else {}, explanation=draft.explanation)
