from __future__ import annotations

from .base import DraftContext, DraftResult, LLMProvider, register_provider


@register_provider("stub")
class StubProvider(LLMProvider):
    """Deterministic, no-network provider. Used by default in tests and as a
    template for adding a real second provider alongside AnthropicProvider --
    register a class with @register_provider("name") and point
    MODELMAKER_LLM_PROVIDER at it."""

    def draft(self, ctx: DraftContext) -> DraftResult:
        if ctx.mode == "params_only":
            explanation = (
                "Stub provider: no LLM was called, so no parameter values were suggested -- "
                f"replace with a real provider to act on: {ctx.instruction!r}"
            )
            return DraftResult(explanation=explanation)

        code = f"def {ctx.function_name}(df):\n    return df\n"
        explanation = (
            "Stub provider: no LLM was called. Returned the input unchanged -- "
            f"replace with a real provider to act on: {ctx.instruction!r}"
        )
        return DraftResult(code=code, metadata_transform={"kind": "passthrough"}, explanation=explanation)
