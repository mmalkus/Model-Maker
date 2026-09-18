from __future__ import annotations

from .base import DraftContext, DraftResult, LLMProvider, register_provider


@register_provider("stub")
class StubProvider(LLMProvider):
    """Deterministic, no-network provider. Used by default in tests and as a
    template for adding a real second provider alongside AnthropicProvider --
    register a class with @register_provider("name") and point
    MODELMAKER_LLM_PROVIDER at it."""

    def __init__(self, model: str | None = None):
        pass

    def draft(self, ctx: DraftContext) -> DraftResult:
        if ctx.mode == "rename":
            # ctx.input_ports is keyed by each real output port name here
            # (see api.py's suggest_names) -- one placeholder name per key.
            params = {"name": "stub_renamed_block"}
            params.update({port: f"stub_{port}" for port in ctx.input_ports})
            return DraftResult(
                params=params,
                explanation="Stub provider: no LLM was called, so these are placeholder names.",
            )

        if ctx.mode == "analyze_data":
            columns = next(iter(ctx.input_ports.values()), [])
            document = (
                "Stub provider: no LLM was called, so this is not a real analysis -- "
                f"replace with a real provider to act on it. {len(columns)} column(s) were given."
            )
            # Deterministic placeholder tags, just enough for a caller to
            # exercise the apply path without a real provider configured.
            tags = {c.name: "stub-tag" for c in columns}
            return DraftResult(params=tags, explanation=document)

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
