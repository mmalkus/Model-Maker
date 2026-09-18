from __future__ import annotations

import json
import re

from .base import ColumnInfo, DraftContext

JSON_ONLY_INSTRUCTIONS = """
Respond with ONLY a single JSON object -- no markdown code fences, no prose \
before or after it. It must have exactly these keys:
{
  "code": "<the complete function definition as a string>",
  "metadata_transform": {
    "kind": "passthrough" | "narrow" | "declared",
    "base": "<input port name, only when kind is 'declared', else null>",
    "drops": ["<column name>", ...],
    "adds": [{"name": "...", "dtype": "...", "role": "..."}, ...]
  },
  "params": {"<param name>": <default value>, ...},
  "explanation": "<one or two sentences>"
}
Omit "drops"/"adds"/"base" (send empty list / null) when kind is not \
"declared". The response must be valid JSON parseable by a standard parser."""


def extract_json_response(text: str) -> dict:
    """Pull a JSON object out of a model's raw text reply, tolerating markdown
    code fences or stray prose around the object (small/local models in
    particular tend not to follow "JSON only" instructions exactly)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"model did not return parseable JSON: {text[:500]!r}")

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
- Never hardcode an *existing* input column's name as a string literal in the \
function body. For each existing column your logic reads, add a `str` \
parameter named `<descriptive_name>_col`, defaulting to that column's real \
name from the input schema, and reference the parameter (e.g. \
`pl.col(income_col)`) instead of the literal -- this lets the tool re-point \
the block at a differently-named upstream column later without touching the \
code. Columns you are newly creating still get fixed, literal names via \
`.alias(...)`; only pre-existing columns you read need a `_col` parameter.
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

CONTRACT_ANALYZE_DATA = """You are analyzing one block's output columns in \
a visual, Polars-based data pipeline tool, given only column names, dtypes, \
current role tags, and summary statistics -- never the full data. You are \
not writing or configuring any code; ignore the "Function name"/"Fixed \
function source" framing below, it's an artifact of a shared request format. \
Follow the contract exactly:

- Leave `code` as an empty string and `metadata_transform` as \
`{"kind": "passthrough"}` -- both are ignored for this action.
- In `params`, add one entry per column worth flagging, keyed by that \
column's exact name, whose value is a short comma-separated list of \
lowercase, hyphen-separated tags (e.g. "likely-id, high-cardinality" or \
"mostly-null, candidate-target"). Only include a column where you have \
something worth saying; omit the rest entirely rather than giving them an \
empty or filler tag.
- In `explanation`, write a short Markdown write-up (a few short \
paragraphs) describing what this dataset appears to be and what's worth \
knowing about it -- going well beyond the one-sentence limit used \
elsewhere in this tool, since a real write-up is the whole point of this \
action. Call out anything the statistics suggest is worth flagging (a \
column that's mostly null, a near-constant column, an id-shaped column, a \
column whose name/values suggest it's the modelling target), and be \
explicit that this is inferred from column names and statistics only, not \
the actual data."""

CONTRACT_RENAME = """You are proposing better names for one block in a \
visual, Polars-based data pipeline tool: its own display name, and a name \
for each of its output ports (used as the variable name for that data in \
the compiled script, and shown wherever that data is wired elsewhere on \
the canvas). You are not writing or configuring any code; ignore the \
"Function name"/"Fixed function source" framing below, it's an artifact of \
a shared request format. Follow the contract exactly:

- Leave `code` as an empty string and `metadata_transform` as \
`{"kind": "passthrough"}` -- both are ignored for this action.
- In `params`, propose a name for the block itself under the key "name", \
and a name for each output port under a key exactly matching that port's \
own name (e.g. "out"). Base each on what the block's instruction/category \
says it does and, when given, the columns its actual output shows -- \
short, lowercase, space- or underscore-separated, at most a few words \
(e.g. "clean applications", "scored_customers"). The instruction below \
lists names already used elsewhere in this graph -- never propose one of \
those; pick a different, still-descriptive name instead.
- In `explanation`, one sentence on why you chose these names."""

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
def bucket_income(df: pl.DataFrame, income_col: str = "income") -> pl.DataFrame:
    return df.with_columns(
        (pl.col(income_col) > 50000).alias("high_income")
    )
```
`income` is an existing column being read, so it becomes the `income_col` \
parameter (defaulted to its real name) instead of a literal; `high_income` is \
a new column, so it keeps a fixed, literal name via `.alias(...)`. `params` \
would be `{"income_col": "income"}`, and this adds one boolean column, so \
`metadata_transform` would be \
`{"kind": "declared", "base": "df", "drops": [], "adds": [{"name": "high_income", "dtype": "Boolean", "role": "feature"}]}`.
"""


def contract_for(mode: str, include_reference: bool = False) -> str:
    if mode == "params_only":
        return CONTRACT_PARAMS_ONLY
    if mode == "analyze_data":
        return CONTRACT_ANALYZE_DATA
    if mode == "rename":
        return CONTRACT_RENAME
    return CONTRACT + ("\n" + POLARS_REFERENCE if include_reference else "")


def format_columns(columns: list[ColumnInfo]) -> str:
    if not columns:
        return "    (unknown -- this input hasn't been run yet, so its columns aren't available)"
    lines = []
    for c in columns:
        stats = []
        if c.count is not None:
            stats.append(f"count={c.count}")
        if c.null_count:
            stats.append(f"nulls={c.null_count}")
        if c.n_unique is not None:
            stats.append(f"distinct={c.n_unique}")
        if c.mean is not None:
            stats.append(f"mean={c.mean:.4g}")
        if c.std is not None:
            stats.append(f"std={c.std:.4g}")
        if c.min is not None:
            stats.append(f"min={c.min}")
        if c.max is not None:
            stats.append(f"max={c.max}")
        suffix = f" ({', '.join(stats)})" if stats else ""
        lines.append(f"    - {c.name}: {c.dtype}, role={c.role}{suffix}")
    return "\n".join(lines)


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
    for port, cols in (ctx.input_ports or {}).items():
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
