# Writing blocks

Every block in Model-Maker, the built-in ones included, is a plain Python
function plus a short literal description of it. Blocks live in `.py`
files, one or more per file. Your own blocks go in your **block library**:
`~/.modelmaker/blocks/`, or the folder `MODELMAKER_BLOCK_LIBRARY` points
at. The folder can be a git repo or a shared drive.

In the app, **Library** in the palette opens the library. From there you
can browse blocks, edit a file, create a block with AI, import files and
export them. Placed blocks keep their own copy of their code, so editing
the library never changes a model. The model only changes when you choose
**Update** on a placed block.

The template every new file starts from is
[`modelmaker/blocks/_template.py`](../modelmaker/blocks/_template.py).
It explains every key inline.

## A block file

```python
"""PD backtests: tests that compare predicted PDs with observed defaults."""

import polars as pl

BLOCKS = {
    "binomial_test": {
        "display_name": "Binomial test per grade",
        "version": 2,
        "block_type": "output",
        "tags": ["pd", "calibration"],
        "requires": ["scipy>=1.11"],
        "inputs": [{"name": "df", "type": "dataframe"}],
        "outputs": [
            {"name": "table", "type": "dataframe", "aggregate": True},
            {"name": "metric", "type": "scalar_metric"},
        ],
        "form": [
            {"key": "grade_col", "label": "Grade column", "kind": "column"},
            {"key": "pd_col", "label": "PD column", "kind": "column", "auto_role": "predicted"},
            {"key": "alpha", "label": "Significance", "kind": "number", "step": 0.01},
        ],
        "meta": {"inherit": "none"},
        "renamed": {"confidence": "alpha"},
    },
}


def binomial_test(df: pl.DataFrame, grade_col: str, pd_col: str, default_col: str, alpha: float = 0.05):
    """Per-grade one-sided binomial test of the observed default count.

    The first docstring line is the block's summary; the rest is its
    description in the library and in the AI catalogue."""
    from scipy.stats import binomtest

    rows = _per_grade(df, grade_col, pd_col, default_col)
    ...
    return table, {"share_failed": share}


def _per_grade(df, grade_col, pd_col, default_col):
    # A private helper: shared by the blocks in this file, not a block.
    return df.group_by(grade_col).agg(...)


def _example_binomial_test():
    # An optional self-check, run by "Try it" and `modelmaker check-blocks`.
    df = pl.DataFrame({"grade": ["A", "A", "B"], "pd": [0.01, 0.01, 0.2], "flag": [0, 0, 1]})
    table, metric = binomial_test(df, "grade", "pd", "flag")
    assert table.height == 2
```

The rules, all checked by the validator:

1. **`BLOCKS` is a plain literal.** It is read without running the file,
   so listing the library or previewing an import runs no code. It can
   hold no names, calls or expressions. Comments are fine.
2. **Each key names a public top-level function, and each public
   function has a key.** The key is the block's id. Ids are unique across
   the built-ins and your whole library. Functions whose names start with
   `_` are helpers. `_example_<id>` is that block's self-check.
3. **The function's docstring is the block's description.** The module
   docstring describes the file.
4. **Only `polars` (as `pl`) at the top of the file.** Import anything else
   inside the function that uses it. A compiled script emits each block
   function together with the helpers it uses, so every block has to be a
   self-contained unit.
5. **Input ports are parameters of the function, by name.** The function
   returns one value per output port, as a tuple when there are several.
6. **No key takes a function.** If a block seems to need one, that's a
   missing piece of the app, not of the block.

## Keys

| Key | Meaning |
|---|---|
| `display_name` | Title on the canvas and in the palette. |
| `version` | An integer. Bump it when you change the block: placed copies then offer **Update to vN**. |
| `block_type` | `"input"` (no upstream, can be refreshed and checked for changes), `"standard"`, or `"output"` (can take the engine's `output_dir`). |
| `group` | Palette group (optional). |
| `tags` | Catalogue tags, from the fixed list in `modelmaker/agent/catalogue.py` (`TAGS`). |
| `requires` | pip requirement strings, e.g. `"scipy>=1.11"`. They are checked against installed package metadata without importing anything. A block with unmet requirements can't be placed. |
| `inputs` / `outputs` | Ports, each `{"name", "type"}`. Types: `dataframe`, `model`, `scalar_metric`, `image`, `master_scale`, `binning`, `any`, and the stochastic types. Port flags are below. |
| `form` | Params form fields worth a label, placeholder or options. Every other param gets a field inferred from the signature. |
| `meta` | How output columns get their roles. See below. |
| `hints` | The decision-hint rule that judges the block's result in an AI build. See below. |
| `renamed` | `{"old_param": "new_param"}`. **Update** carries the old value over. |
| `renamed_ports` | `{"old_port": "new_port"}`. **Update** keeps the wires connected. |

### Port flags

- `"lazy": true` on an **input** port means the function also works on a
  `pl.LazyFrame` there, i.e. it is pure polars expressions
  (`filter`/`select`/`with_columns`/`join`/`group_by`). A streaming run
  fuses such blocks into one query.
- `"lazy": true` on an **output** port means the function returns a
  `pl.LazyFrame` (an input block's `pl.scan_csv(...)`, say). The engine
  collects it, pushing sample mode's row cap into the scan first.
- An output block with no outputs and one lazy input is a **sink**. In a
  streaming run it receives its upstream's uncollected plan.
  `df.lazy().sink_csv(path)` works whether it gets a DataFrame or a
  LazyFrame.
- `"aggregate": true` on a dataframe **output** marks a statistics table:
  one row per feature, bin, grade or period, never one per record. An AI
  build may read such a table's rows. It sees every other dataframe as
  schema and summary statistics only. The flag is honored for built-ins;
  for library blocks it is shown but not trusted.

### Form field kinds

`text`, `number` (`step`), `select` (`options`), `column` (`auto_role`:
`"target"` or `"predicted"` fills it from the tagged column), `columns`,
`checkbox`, and three kinds the app attaches behavior to:

- `path`: a file the block reads. It gets a **Browse…** button
  (`options` = accepted extensions), and its modified time and size are
  the block's change fingerprint for **Check for changes**.
- `sql`: a query, shown as a multi-line field. Together with an `env`
  field (the connection), it lets the block's optional `probe_query`
  instance setting detect new data.
- `env`: the *name* of an environment variable, such as a connection URI.
  The UI can set it for the session. The secret itself is never stored in
  the project.

### Column metadata: `meta`

Every block's output columns get their metadata from one rule:

- A column that exists in an input keeps that input's role, description
  and tags. When several inputs have it, the first-listed input port wins.
- A new column gets its dtype inferred and the role `new_role` (default
  `"unassigned"`).
- `roles` then tags specific columns.

```python
"meta": {
    "inherit": "all",          # or "none" (a new table), {"port": "df"}, {"columns_param": "by"}
    "new_role": "feature",
    "roles": [
        {"column": "lgd", "role": "target", "if_taken": "feature"},
        {"column_param": "output_col", "role": "predicted", "displaced": "feature"},
        {"column_param": "flag_col", "fallback": "default_{horizon}m", "role": "target"},
        {"column_suffix": "_bin", "role": "segment"},
    ],
}
```

- `inherit: "none"` is for a block whose output is a new table, such as a
  statistics table or a summary. `{"columns_param": "by"}` inherits only
  the columns a list param names, for example group-by keys.
- A role rule names its column directly (`column`), through a param
  (`column_param`, with a `fallback` template over the params for when
  it's unset), or by suffix (`column_suffix`, new columns only).
- Roles: `id`, `target`, `predicted`, `weight`, `feature`, `date`,
  `segment`, `excluded`, `unassigned`. The first five may sit on at most
  one column. For those, `if_taken` gives the column that role instead
  when another column already has it, and `displaced` re-tags any other
  column that has it.

### Decision hints: `hints`

`"hints": "<rule>"` picks a rule from
[`modelmaker/agent/hint_rules.py`](../modelmaker/agent/hint_rules.py). In
an AI build the rule reads the block's outputs and adds a short "what this
result means" note. Each rule reads a specific output shape. For example,
`auc_gini` reads a scalar_metric dict with a `gini` key on a `metric`
port. Only use a rule when your block produces that shape. The library
panel lists the rule names and the built-ins that use each one.

## What the app does around every block

None of this is written per block. The app does it for every block,
driven by what the block declares:

| What | How |
|---|---|
| Sample mode | The engine caps each input block's output at N rows (`.head(n)`, pushed into lazy scans). |
| Change detection | From `path` fields (file fingerprint), or `sql` + `env` fields plus a `probe_query`. |
| Streaming | From the `lazy` port flags. |
| Caching | Keyed by the definition's sha plus params and upstream, so an edited block never serves a stale result. |
| Column roles | `meta`, as above, plus the user's own role tags. |
| AI decision hints | `hints`. |
| Group by | Any block with a dataframe input can run once per group. |

## Self-checks and the CLI

`_example_<id>()` builds a tiny input by hand, calls the block and
asserts something about the result. **Try it** in the library runs it,
in a separate process with a 30-second timeout. So does

```bash
modelmaker check-blocks [DIR]   # default: the built-ins and your library
```

which validates every file, checks requirements and runs every example. It
exits 1 on any failure, so a team's library repo can run it in CI.

## Lifecycle

- **Placing** a library block copies its definition into the project, in
  `model.json` plus `blocks/<block id>.py` next to it. The project runs on
  a machine without your library.
- **Updating**: when the library has a newer version, the block shows
  **Update to vN**. The dialog shows the code diff and what happens to
  params and wires, using `renamed` and `renamed_ports`, before anything
  changes.
- **Built-ins** are pinned by sha too. After a Model-Maker upgrade changes
  one, placed copies say *built-in changed since this model was saved* and
  offer **Accept**. To change a built-in yourself, **Duplicate to
  library** and edit the copy.
- **Concurrent edits**: a save carries the file's sha from when you opened
  it. If someone saved in the meantime, the save is refused and you reload
  and re-apply instead of overwriting their change.
- **Import** previews every block (validated from the literal alone, with
  nothing run) and asks what to do with each clash: overwrite, import
  under a new id, or skip.
- **AI builds** that need a block no built-in covers write it to
  `drafts/build_<id>.py` in your library. The library lists it under *AI
  drafts*, where you **Keep** (move into a file) or **Discard** it.
  Discarding never breaks a model, because placed blocks embed their code.
- **AI builds use your library.** A build sees your library blocks next to
  the built-ins: under their own tags, and all of them under the `library`
  tag. That includes blocks earlier builds drafted. It places them like you
  would, as embedded copies linked to their file, and reuses an existing
  block rather than writing a duplicate. Input blocks, blocks with errors
  and blocks with unmet requirements are never offered. The build panel's
  *Use my library blocks* switch turns this off for one build.

## Create with AI

**✦ Create with AI** (in the palette and the library) drafts a block
from a description. Optionally pick a canvas block whose output is the
example input; the AI sees its column names, types and roles, never rows.
The draft goes through the same validator as a hand-written block. You
review and edit it, run its self-check or try it on that input, refine it
with follow-up instructions, and save it. Only saved blocks can be placed.
**Edit with AI** changes an existing library block. A red library block's
inspector offers **Fix with AI**, which drafts a fix from its code and
error and then updates the block.
