from __future__ import annotations

import json
import re

from .base import ColumnInfo, DraftContext

JSON_ONLY_INSTRUCTIONS = """
Respond with ONLY a single JSON object -- no markdown code fences, no prose \
before or after it. It must have exactly these keys:
{
  "code": "<Python source: the block function, any _helpers, and its _example_ (or empty)>",
  "block": {<the block's BLOCKS entry (or {})>},
  "params": {"<param name>": <value>, ...},
  "explanation": "<one or two sentences>"
}
The response must be valid JSON parseable by a standard parser."""


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

CONTRACT = """You write one block for the block library of a visual, \
Polars-based model development tool. A block is a plain Python function plus \
a literal description of it (its BLOCKS entry). Follow the contract exactly:

`code` -- Python source holding:
- Exactly one public top-level function: the block. Name it in snake_case for \
what it does (or as instructed); the name is the block's id. Its first \
parameters are its input ports (usually `df: pl.DataFrame`), then any \
configurable values, each with a sensible default. It returns one value per \
output port (a tuple when there are several).
- Optional private helpers (names starting with `_`) the block calls.
- One `_example_<block name>()` function: a self-check that builds a tiny \
`pl.DataFrame` by hand, calls the block, and `assert`s something meaningful \
about the result. It must pass.
- Only `polars` is available at the top (as `pl`; don't import it). Import \
anything else (numpy, scipy, sklearn) *inside* the function that uses it. \
No pandas, no file or network I/O, no global state.
- Give the function a docstring: first line a one-sentence summary, then \
what it computes, its inputs and outputs, and its parameters.
- Never hardcode an *existing* input column's name in the body: read it \
through a `str` parameter named `<descriptive>_col` (or `<descriptive>_cols` \
for a list) defaulted to the real column name from the input schema. New \
columns you create get literal names via `.alias(...)`.

`block` -- the BLOCKS entry, a JSON object with these keys:
- "display_name": short title shown on the canvas.
- "version": 1 for a new block; the existing version + 1 when changing one.
- "block_type": "standard" (a transform or test), "input" (no inputs, reads \
data) or "output" (a report/sink at the end of a branch).
- "inputs" / "outputs": lists of {"name": <port>, "type": <type>}; types are \
dataframe, model, scalar_metric, image, master_scale, binning, any. Add \
"aggregate": true to a dataframe output that is a statistics table (one row \
per feature/bin/grade/period, never per record). Add "lazy": true to a \
dataframe input only if the function is pure polars expressions that also \
work on a pl.LazyFrame (filter/select/with_columns/join/group_by).
- "form" (optional): [{"key": <param>, "label": <text>, "kind": <kind>}] for \
params worth a label; kinds: text, number, select (+ "options"), column, \
columns, checkbox. Every key must be a parameter of the function.
- "meta" (optional): how output columns get their roles. Columns from an \
input keep its role; new columns start "unassigned". {"new_role": "feature"} \
tags every new column a feature; {"roles": [{"column": <name>, "role": \
<role>}]} tags specific columns (roles: id, target, predicted, weight, \
feature, date, segment, excluded, unassigned); {"inherit": "none"} for a \
block whose output is a new table (statistics, a summary).
- "tags" (optional): from the tag list given below only.
- "hints" (optional): a decision-hint rule from the list given below, only \
when this block's outputs have exactly the shape that rule reads.
- "requires" (optional): pip requirements beyond polars/numpy/scipy/sklearn.

When existing code and an error are given, this is a bug fix: fix the error, \
keep the block's name, ports and intent, and bump "version". When an existing \
block is given without an error, change it as instructed and bump "version". \
When other blocks of the target file are shown, reuse their helpers and match \
their style instead of duplicating code.

Leave `params` empty. Keep `explanation` to one or two plain-English \
sentences: what the block does, or what you changed and why."""

CONTRACT_ANALYZE_DATA = """You are analyzing one block's output columns in \
a visual, Polars-based data pipeline tool, given only column names, dtypes, \
current role tags, and summary statistics -- never the full data. You are \
not writing or configuring any code; ignore the "Function name"/"Fixed \
function source" framing below, it's an artifact of a shared request format. \
Follow the contract exactly:

- Leave `code` as an empty string and `block` as `{}` -- both are ignored \
for this action.
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

- Leave `code` as an empty string and `block` as `{}` -- both are ignored \
for this action.
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
- Leave `code` as an empty string and `block` as `{}` -- both are ignored \
for this block.
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
a new column, so it keeps a fixed, literal name via `.alias(...)`, and the \
block's "meta" would be `{"new_role": "feature"}`.
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
    parts = [f"Function name: {ctx.function_name}"] if ctx.function_name else ["Function name: (your choice)"]

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

    if ctx.existing_block:
        parts += ["", "Its current BLOCKS entry:", json.dumps(ctx.existing_block, indent=2)]

    if ctx.library_context:
        parts += ["", ctx.library_context]

    if ctx.error:
        parts += ["", "This code just failed with the following error -- fix it:", ctx.error, "", f"Additional instruction: {ctx.instruction}"]
    else:
        parts += ["", f"Instruction: {ctx.instruction}"]

    return "\n".join(parts)
