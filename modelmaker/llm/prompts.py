from __future__ import annotations

from .base import ColumnInfo, DraftContext

CONTRACT = """You draft a single block's implementation for a visual, \
Polars-based data pipeline tool. Follow the contract exactly:

- Write exactly one top-level Python function named as instructed. It takes \
`df: pl.DataFrame` as its first parameter, plus any additional parameters you \
introduce for configurable values (each with a sensible default), and returns \
a single `pl.DataFrame`.
- Use only the `polars` library (imported as `pl`). Never use pandas.
- The function body is inlined verbatim into a generated script and also run \
directly by the engine -- do not reference anything outside the function \
(no globals, no I/O, no network calls). It must be a pure function of its \
arguments.
- Only reference columns that are listed in the provided input schema. If the \
instruction implies a new column, add it with `.with_columns(...)`.
- Also produce a `metadata_transform` describing how the function changes the \
column set, so the tool can track column roles/types without re-executing \
your code:
  - kind="passthrough": output columns are exactly the input columns, \
unchanged (dtype may differ; the tool re-infers dtype either way).
  - kind="narrow": output columns are a subset of the input columns (e.g. a \
select/filter that drops columns).
  - kind="declared": output adds and/or drops specific columns. Set `base` \
to the input port name ("df"), list any dropped column names in `drops`, and \
describe every added column in `adds` (name, a short polars dtype string like \
"Int64"/"Float64"/"Boolean"/"String"/"Date", and a role -- one of id, target, \
weight, feature, date, segment, excluded, unassigned).
- If `params` values are provided as suggested defaults, they must match the \
extra parameter names in your function signature exactly.
- If existing code and an error are provided, this is a bug-fix request: fix \
the described error while preserving the function's original intent as \
closely as possible. Do not rewrite unrelated behavior.
- Keep `explanation` to one or two plain-English sentences describing what \
the function does or what you changed and why."""


def format_columns(columns: list[ColumnInfo]) -> str:
    if not columns:
        return "(unknown -- this input hasn't been run yet, so its columns aren't available)"
    return "\n".join(f"  - {c.name}: {c.dtype}, role={c.role}" for c in columns)


def build_user_prompt(ctx: DraftContext) -> str:
    parts = [
        f"Function name: {ctx.function_name}",
        "",
        "Input schema (port `df`):",
        format_columns(ctx.input_ports.get("df", [])),
    ]

    if ctx.param_names:
        parts += ["", "Current parameter names: " + ", ".join(ctx.param_names)]

    if ctx.existing_code:
        parts += ["", "Existing code:", "```python", ctx.existing_code, "```"]

    if ctx.error:
        parts += ["", "This code just failed with the following error -- fix it:", ctx.error, "", f"Additional instruction: {ctx.instruction}"]
    else:
        parts += ["", f"Instruction: {ctx.instruction}"]

    return "\n".join(parts)
