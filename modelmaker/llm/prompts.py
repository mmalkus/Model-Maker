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

CONTRACT_PARAMS_ONLY = """You are configuring one block in a visual, \
Polars-based data pipeline tool. This block's Python function is fixed -- \
you cannot change its code, only choose values for the parameters it \
already accepts. Follow the contract exactly:

- The fixed function is given below (read-only). Read its signature and \
docstring to see exactly which parameters exist, their types, and their \
defaults.
- Respond with a `params` object containing values for whichever of those \
parameters the instruction implies should change. Only use keys that are \
real parameter names of the function -- never invent new ones, and never \
include the dataframe argument(s) themselves.
- Only reference columns listed in the provided input schema.
- Leave `code` as an empty string and `metadata_transform` as \
`{"kind": "passthrough"}` -- both are ignored for this block.
- Keep `explanation` to one sentence describing the values you chose and \
why."""


POLARS_REFERENCE = """
Condensed Polars reference -- consult this instead of relying on possibly \
stale training knowledge of the API, especially for smaller/local models:

All of `.select()`/`.filter()`/`.with_columns()`/`.group_by().agg()` are \
expression-based -- write `pl.col("name")`, not `df["name"]`. There is no \
pandas-style label indexing (`.loc`/`.iloc`/`df["col"] = ...`).

- Select/rename: `df.select(pl.col("a"), pl.col("b").alias("b2"))`
- Add/replace a column: `df.with_columns((pl.col("a") * 2).alias("a_doubled"))`
- Filter rows: `df.filter(pl.col("score") > 700)`
- Conditional column: `df.with_columns(pl.when(pl.col("x") > 0).then(pl.lit("pos")).otherwise(pl.lit("neg")).alias("sign"))`
- Group + aggregate: `df.group_by("segment").agg(pl.col("amount").sum().alias("total"))`
- Join: `df.join(other, on="id", how="left")`
- Sort: `df.sort("date", descending=True)`
- Cast dtype: `pl.col("x").cast(pl.Float64)`
- String ops: `pl.col("name").str.to_lowercase()`, `.str.contains("x")`, `.str.replace_all(a, b)`
- Date ops: `pl.col("d").dt.year()`, `.dt.strftime("%Y-%m")`
- Null handling: `pl.col("x").fill_null(0)`, `.is_null()`, `.drop_nulls()`
- Row count / unique: `df.height`, `df.select(pl.col("x").n_unique())`

Common mistakes to avoid:
- `.alias()` is required to name an expression's output column; without it \
the output keeps the input expression's original column name.
- `pl.when/then/otherwise` is the ternary/`np.where` equivalent -- there is \
no `np.where` and no `.apply(lambda ...)` in idiomatic polars code.
- Boolean masks combine with `&`/`|` (never `and`/`or`), each operand \
parenthesized: `df.filter((pl.col("a") > 1) & (pl.col("b") < 5))`.

Worked example -- instruction: "flag rows where income > 50000 as high_income"
```python
def bucket_income(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("income") > 50000).alias("high_income")
    )
```
This adds one boolean column, so `metadata_transform` would be \
`{"kind": "declared", "base": "df", "drops": [], "adds": [{"name": "high_income", "dtype": "Boolean", "role": "feature"}]}`.
"""


def contract_for(mode: str, include_reference: bool = False) -> str:
    if mode == "params_only":
        return CONTRACT_PARAMS_ONLY
    return CONTRACT + ("\n" + POLARS_REFERENCE if include_reference else "")


def format_columns(columns: list[ColumnInfo]) -> str:
    if not columns:
        return "    (unknown -- this input hasn't been run yet, so its columns aren't available)"
    return "\n".join(f"    - {c.name}: {c.dtype}, role={c.role}" for c in columns)


def build_user_prompt(ctx: DraftContext) -> str:
    parts = [f"Function name: {ctx.function_name}"]

    if ctx.fixed_source:
        parts += [
            "",
            "Fixed function source (read-only -- you are only choosing parameter values, not writing code):",
            "```python",
            ctx.fixed_source,
            "```",
        ]

    parts += ["", "Input schema:"]
    for port, cols in (ctx.input_ports or {"df": []}).items():
        parts.append(f"  Port `{port}`:")
        parts.append(format_columns(cols))

    if ctx.param_names:
        parts += ["", "Current parameter names: " + ", ".join(ctx.param_names)]

    if ctx.existing_code:
        parts += ["", "Existing code:", "```python", ctx.existing_code, "```"]

    if ctx.error:
        parts += ["", "This code just failed with the following error -- fix it:", ctx.error, "", f"Additional instruction: {ctx.instruction}"]
    else:
        parts += ["", f"Instruction: {ctx.instruction}"]

    return "\n".join(parts)
