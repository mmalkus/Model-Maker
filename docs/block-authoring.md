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
7. **Defaults the app uses must be literals.** See below.

## When your code runs

The app reads everything it needs to know about a block from the file's
text, without running it: the `BLOCKS` literal, and each function's
signature (parameter names, which have defaults, their annotations as
written, and the default values). Listing the library, rendering a
block's form, placing and wiring it, working out whether it's stale,
validating it, compiling a model to a script and the AI catalogue all run
none of your code.

Your code runs only when a block actually runs: a run on the canvas, a
**Try it**, an `_example_` self-check, or `modelmaker check-blocks`. Each
of those happens in a separate, single-use task process. That includes
everything at module level that the block uses (helpers, constants), its
decorators and its default expressions, so a side effect there (writing a
file, reading the environment) happens in that task process, each time the
block runs, and never in the app itself. Task processes get the app's
environment minus its secrets (API tokens, LLM provider keys), and what
they hand back to the app is data only: dataframes, JSON-shaped values,
bytes, numpy arrays, or scikit-learn / scipy.sparse objects.

### Default values

The app reads a parameter's default with `ast.literal_eval`, so it knows
the value when the default is a **literal**: a number, string, bool or
`None`, or a list, tuple, dict or set of those (`-1`, `1e-6`, `(0.1, 0.9)`
and `["a", "b"]` all count).

Any other default -- `x=SOME_CONST`, `x=float("inf")`, `x=_helper()` -- is
still allowed. The block runs with it as usual, and calls may leave the
param out, but the app doesn't know its value: the form shows the field as
unset, the AI catalogue lists the expression's source text instead of a
value, and a block an AI build places doesn't copy it into its params. Two
things make a non-literal default an error:

- a `meta` rule reads the param (`column_param`, `columns_param`, or a
  `{param}` in a `fallback` template), because the rule needs the value;
- the expression (or a decorator) uses a name your code doesn't define,
  which the validator reports the way a `NameError` would have been.

The signature the app sees is the `def` line as written. A decorator that
changes the function's parameters (one that wraps it in a `*args,
**kwargs` function, say) isn't seen, so don't rely on one for the
parameters the app fills in (`output_dir`, `block_id`, role params such as
`target_col`) or for the form.

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
| `display` | Show the block's result right on its canvas node: `"table"`, `"kpi"` or `"value"`. See below. |
| `renamed` | `{"old_param": "new_param"}`. **Update** carries the old value over. |
| `renamed_ports` | `{"old_port": "new_port"}`. **Update** keeps the wires connected. |
| `capabilities` | What the block needs the workspace to allow, beyond pure computation: `"filesystem"`, `"datasets"`, `"sources"`, `"exports"`. See below. Field kinds add their own. |
| `use_instead` | A block id to suggest when the workspace doesn't allow this one, e.g. `"read_dataset"` for a block that reads a path. |

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
`checkbox`, and five kinds the app attaches behavior to:

- `path`: a file the block reads. It gets a **Browse…** button
  (`options` = accepted extensions), and its modified time and size are
  the block's change fingerprint for **Check for changes**.
- `sql`: a query, shown as a multi-line field. Together with an `env`
  field (the connection), it lets the block's optional `probe_query`
  instance setting detect new data.
- `env`: the *name* of an environment variable, such as a connection URI.
  The UI can set it for the session. The secret itself is never stored in
  the project. A value set from the UI is kept in memory by the app and
  handed to the block's run (and to its change probe) on top of the
  server's own environment; it isn't written into the server process's
  environment. Names the app treats as secrets (its own and the LLM
  providers' API keys, git host tokens, anything listed in
  `MODELMAKER_WORKER_ENV_DENY`) never reach block code however they're
  set, so give a connection variable an ordinary name such as `DWH_URI`.
- `dataset`: the *name* of one of the workspace's datasets, picked from a
  list (`GET /api/datasets`). Before the block runs, the engine resolves
  the name to the dataset's current file, read-only, and passes that
  **path** to the function in its place. The dataset's version is part of
  the block's cache key, so replacing the dataset makes the block stale.
  The embedding application provides the datasets (see
  `modelmaker/datasets.py`); the local app has none, so blocks with this
  kind aren't offered there.
- `source`: the name of a data source the application's administrator
  defined (`GET /api/sources`), used with a `sql` field. The engine runs
  the query through the application -- never in the block's process, so
  the credentials never reach block code -- and passes the path of the
  result (a Parquet file) in place of the name.

`path` and `env` fields need the workspace's `filesystem` capability,
`dataset` fields `datasets`, `source` fields `sources`.

### Capabilities

A workspace has a capability set (`Workspace(capabilities=...)`, see
`modelmaker/capabilities.py`); a block needs whatever it lists under
`capabilities` plus what its field kinds imply. The local app grants
`filesystem`, so it behaves as it always has. An application hosting
other people's workspaces typically withholds `filesystem` and grants
`datasets`, `sources` and `exports` instead, by handing the workspace a
dataset provider, a source provider and an export sink.

A block the workspace doesn't allow is left out of the palette,
`/api/registry`, the library panel's built-ins and the AI builder's
catalogue, and is refused when placed, when a project or recovery
snapshot holding it is opened, and when it runs. The refusal names the
block and its `use_instead`, e.g. *'Read CSV' (read_csv) reads or writes
files on the server by path, which this workspace doesn't allow -- use
'Read dataset' (read_dataset) instead*.

The built-ins: `read_csv`, `read_parquet`, `read_json`, `read_excel`,
`read_sql` and `write_csv` need `filesystem`; `read_dataset` needs
`datasets`, `read_source` `sources`, and `export` `exports`. A block
needing `exports` is an export block: after it runs successfully, its
first dataframe output goes to the workspace's export sink with the
block's `format` and `name` params. A block with a `dataset` or `source`
field never joins a streaming run's fusion (the engine resolves its file
first).

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
- `"new_only": true` applies a rule only when the block created the
  column, not when a column with that name came through from an input.
  The `predict` block uses it, so scoring a table that already carries
  another model's prediction makes the new one `predicted` and the old
  one a `feature`.

### Decision hints: `hints`

`"hints": "<rule>"` picks a rule from
[`modelmaker/agent/hint_rules.py`](../modelmaker/agent/hint_rules.py). In
an AI build the rule reads the block's outputs and adds a short "what this
result means" note. Each rule reads a specific output shape. For example,
`auc_gini` reads a scalar_metric dict with a `gini` key on a `metric`
port. Only use a rule when your block produces that shape. The library
panel lists the rule names and the built-ins that use each one.

### Display blocks: `display`

A block with `"display"` shows its first output on its canvas node, with
no click, and re-runs as soon as its upstream does, so the value follows
every run:

- `"table"`: the first 10 rows and 8 columns of a dataframe, plus the
  full row and column count. **all rows…** opens the full table.
- `"kpi"`: one headline number. The output is a `scalar_metric` dict with
  `label` and `value`, and optionally `number_format` (`number`,
  `compact` or `percent`), `decimals` and `unit`.
- `"value"`: whatever comes through: a table as above, a flat dict as
  key/value pairs, an image, or anything else as truncated JSON.

The built-ins are `display_table`, `display_kpi` (*Headline number*: a
column of a table reduced by `agg`, a metric dict's key, or a number) and
`display_value`. `display_table` and `display_value` also save a table
they show as `<block id>.csv` (or `filename`) in the run's output
directory, unless `save_csv` is off.

## What the app does around every block

None of this is written per block. The app does it for every block,
driven by what the block declares:

| What | How |
|---|---|
| Sample mode | The engine caps each input block's output at N rows (`.head(n)`, pushed into lazy scans). |
| Change detection | From `path` fields (file fingerprint), `dataset` fields (the dataset's version), or `sql` + `env` fields plus a `probe_query`. |
| Capabilities | From `capabilities` and the field kinds: a block the workspace doesn't allow isn't offered or run. |
| Streaming | From the `lazy` port flags. |
| Caching | Keyed by the definition's sha plus params and upstream, so an edited block never serves a stale result. Results are cached as data (Arrow and JSON), never pickled. |
| Column roles | `meta`, as above, plus the user's own role tags. |
| AI decision hints | `hints`. |
| Group by | Any block with a dataframe input can run once per group. |

## Self-checks and the CLI

`_example_<id>()` builds a tiny input by hand, calls the block and
asserts something about the result. **Try it** in the library runs it,
in a task process like a block run (see *When your code runs*) with a
30-second timeout. So does

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
