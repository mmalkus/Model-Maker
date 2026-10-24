# Block Library — proposal

A user-owned library of reusable blocks: browse it in the UI, add blocks to
it (from scratch or from a block on the canvas), and import/export them as
files. The goal is to make Model-Maker customizable without touching the
package source.

**Design rules**

1. **A block file describes the block's core function and nothing else:**
   what it computes (a plain function), its interface (ports and params form)
   and what its outputs *are* (e.g. "this table is aggregate", "this value is a
   Gini"). Sampling, change detection, streaming, sinks, AI decision hints,
   privacy of outputs, caching and grouping are **architecture**, applied
   to every block the same way, built-in or not.
2. **Blocks live in plain `.py` files, one or more blocks per file.** Group
   related blocks together (e.g. all PD backtests in `pd_backtests.py`). No
   manifest, no registry database, no sidecar JSON. Copying a file exports
   its blocks, and dropping a file into the folder imports them.
3. **Built-in blocks use the same format.** With the extras moved into
   the architecture, nothing is left that only a built-in can express, so
   `modelmaker/blocks/*.py` becomes ordinary block files that ship with the
   package. There is one loader, one validator and one code path.

---

## 1. Today vs. proposed

Today a registry block is a function plus a `BlockSpec` carrying
block-specific Python extras. Each of them maps to a piece of shared
architecture that the block only *describes* its way into:

| Today (per block, Python) | Count | Proposed (architecture, driven by the block's description) |
|---|---|---|
| `sample_rows` param on input blocks | ~10 | The **runner** applies sampling: it calls `.head(n)` on the input block's output, and that pushes down for free when the function returns a `LazyFrame`. Blocks never see `sample_rows`. |
| `probe=` (`_probe_path_mtime`, `_probe_sql`) | 5 | **Change detection** is picked from the *param kinds*. A form field of kind `path` gets an mtime/size fingerprint, and kind `sql` (+ connection) gets the shared SQL fingerprint. The block only says "this param is a file path". |
| `lazy_fn=` (read_csv/read_parquet twins + 5 identical reuses) | 7 | An input port marked `"lazy": true` means "this function works on a `LazyFrame`". The **fusion** pass treats a block whose dataframe inputs are all lazy as fusable. Input readers just return `pl.scan_*(...)` and the runner collects at checkpoints, so separate twin functions go away. |
| `lazy_sink_fn=` (write_csv) | 1 | The same port flag on an output block: when fused, it receives a `LazyFrame`. A writer written as `df.lazy().sink_csv(path)` works in both cases. |
| `hints=` (`_auc_gini_hints`, `_psi_test_hints`, …) | 31 | **Decision hints become a rule table** in `agent/hints.py`, keyed by *what a value measures*, not by block. An output declares `"measures": {"gini": "gini"}` or a table kind (`"table": "iv"`), and every block emitting a Gini gets the Gini rules. The rule bodies already live mostly in `hints.py` (the thresholds) and are keyed off metric names (`gini`, `ks_statistic`, `psi`, `p_value`), so this is mostly a move. |
| `aggregate_outputs=` | 11 | A port flag `"aggregate": true`. The **architecture** decides whether to honor it, based on the block's source (see §8), so an imported file can't opt its rows into reaching an LLM. |
| `metadata_transform=` (20 `infer_dtypes`, 9 generic, ~40 bespoke `_x_meta`/lambdas) | ~70 | The default is **automatic schema-diff inheritance**. Output columns that exist upstream keep their metadata, new ones are inferred, and dropped ones disappear. On top of that, declarative role annotations (`"roles": [{"column_param": "pd_column", "role": "predicted"}]`) cover the "tag the column this param names" pattern most bespoke transforms implement. Migration audits each bespoke transform (see §9). |
| `group_by`, caching, column-role overrides | n/a | Already architecture. Unchanged. |

## 2. The file format

```python
"""PD backtests: tests that compare predicted PDs with observed defaults.

The module docstring describes the file as a whole (shown as the group
heading in the library panel).
"""

BLOCKS = {
    "binomial_test": {
        "display_name": "Binomial test per grade",
        "version": 2,
        "block_type": "output",
        "group": "tests",
        "tags": ["pd", "validation"],
        "inputs": [{"name": "in", "type": "dataframe", "lazy": False}],
        "outputs": [
            {"name": "table", "type": "dataframe", "aggregate": True, "table": "grade_backtest"},
            {"name": "metric", "type": "scalar_metric", "measures": {"share_failed": "grade_failures"}},
        ],
        "form": [
            {"key": "grade", "label": "Grade column", "kind": "column"},
            {"key": "pd_column", "label": "PD column", "kind": "column", "auto_role": "predicted"},
            {"key": "alpha", "label": "Significance", "kind": "number", "step": 0.01},
        ],
    },
    "jeffreys_test": {
        "display_name": "Jeffreys test per grade",
        "version": 1,
        ...
    },
}


def binomial_test(df: pl.DataFrame, grade: str, pd_column: str, default_flag: str, alpha: float = 0.05):
    """Per-grade one-sided binomial test of the observed default count."""
    from scipy.stats import binomtest

    rows = _per_grade(df, grade, pd_column, default_flag)
    ...


def jeffreys_test(df: pl.DataFrame, grade: str, pd_column: str, default_flag: str, alpha: float = 0.05):
    """Per-grade Jeffreys (beta posterior) test, the ECB's preferred variant."""
    from scipy.stats import beta

    rows = _per_grade(df, grade, pd_column, default_flag)
    ...


def _per_grade(df, grade, pd_column, default_flag):
    # shared helper: private (leading underscore), not a block
    return df.group_by(grade).agg(...)
```

Rules (all checked by one `validate()` function):

1. **`BLOCKS` is a pure literal**, read with `ast.literal_eval` and never
   executed. Listing the library or previewing an import runs **no code**.
2. **Every key in `BLOCKS` names a public top-level function in the file,
   and every public top-level function has a key.** Functions starting with
   `_` are shared helpers, not blocks.
3. **Each block function's docstring is its description** (first line =
   summary). The module docstring describes the file and heads its group in
   the panel.
4. **Block ids are function names and must be unique across everything
   loaded** (built-ins + library). The validator rejects a clash. Built-in
   ids stay what they are today (`auc_gini`, `read_csv`, …), so existing
   `model.json` files keep working unchanged.
5. **Only `pl` at module level; other imports go inside functions.** This
   is today's built-in and custom-block convention. The compiler emits each
   used block function **plus the helpers it references** (found
   transitively from the function's AST names), and nothing else from the
   file. The compiled script stays a single self-contained file.
6. Every `form[].key` must be a parameter of its function (the same check
   `register_block` does today). Port types must be in `PortType`.
   `measures`/`table` names must be known to the hint table; an unknown one
   is kept but flagged (it gets no hints until a rule exists).
7. **No executable extras.** There is no key that takes a function. If a
   block seems to need one, that's a missing piece of architecture: add it
   there, for every block.

## 3. Where block files live

```
modelmaker/blocks/               # built-in, shipped with the package, read-only
    io.py  transforms.py  binning.py  stat_tests.py  modelling.py  ...
~/.modelmaker/blocks/            # user library (override with MODELMAKER_BLOCK_LIBRARY)
    pd_backtests.py
    preprocessing.py
```

- Both are scanned by the same loader. Each block carries its source: `builtin`
  or `library` (plus the file path).
- The library folder can be a git repo or a shared/network drive, so team
  sharing needs nothing extra. *(Later:* several search paths.)*

## 4. How a block reaches a model

All blocks resolve to one internal type:

```python
@dataclass
class BlockDefinition:
    id: str                 # function name
    meta: dict[str, Any]    # its BLOCKS entry
    code: str               # the function + the helpers it references
    source: Literal["builtin", "library", "custom"]
    origin: str | None      # file path for builtin/library
    sha: str                # hash of meta + code
```

- **Built-in block** on the canvas: the instance references the id, as today.
  The package version defines it.
- **Library block** on the canvas: the instance **embeds a copy** of the
  definition (meta + code) and records `library_ref = {file, id, version, sha}`.
  Because a project keeps the code in its sidecar, it runs on a machine
  without that library. Editing the library never silently changes a
  validated model, and the UI shows *Update to vN* with a diff instead.
- **AI/custom block:** an embedded definition with `source="custom"`. This
  merges today's `is_custom` path into the same type: a custom block is just
  a definition that isn't in any file yet, and *Save to library* writes it
  into one.

The runner, compiler, forms, hints and fusion all consume `BlockDefinition`,
and none of them branch on where the block came from (except §8's
trust rule for `aggregate`).

## 5. Backend

### `modelmaker/block_library.py` (loader + library I/O, no state)

```python
def parse_file(source: str, origin: str) -> list[BlockDefinition]  # ast only, never exec
def validate(defs: list[BlockDefinition], taken_ids: set[str]) -> dict[str, list[str]]  # id -> errors
def load_builtin() -> list[BlockDefinition]       # replaces register_block() at import time
def list_library() -> list[LibraryFile]           # file -> its blocks + per-block errors
def save_file(name: str, source: str, *, overwrite=False) -> LibraryFile
def add_block(file: str | None, definition) -> LibraryFile  # "save to library": new file or append to one
def extract(file: str, ids: list[str]) -> str     # one file holding just these blocks + their helpers
def delete_file(name: str) / delete_block(file: str, id: str)
```

`validate` runs code in exactly one place: it defines the functions without
calling them (the same as `_compile_code_to_fn` today). That happens on save
and on import, after the user has seen the code. Editing a file with several
blocks rewrites only that block's `BLOCKS` entry and function, using
`ast` positions, so comments and the rest of the file stay untouched.

### Architecture modules that take over the extras

- `runner.py`: sampling via `.head(n)`; fusion eligibility from `lazy` port flags.
- `sources.py` (new, small): change-detection strategies by param kind
  (`path` → mtime/size, `sql` → the existing SQL fingerprint). Replaces `probe`.
- `agent/hints.py`: rules keyed by `measures`/`table`, replacing the
  31 per-block hint functions.
- `metadata_transforms.py`: `schema_diff` as the default, plus declarative
  `roles`.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/library` | files → blocks (id, display_name, summary, group, tags, ports, version, sha, errors) |
| `GET /api/library/files/{file}` | full file source + parsed blocks |
| `PUT /api/library/files/{file}` | save a whole file from the editor |
| `POST /api/library/from_block/{block_id}` | *Save to library*: `{file?: existing or new, id, display_name, group}` |
| `DELETE /api/library/files/{file}` · `DELETE /api/library/files/{file}/blocks/{id}` | delete a file or one block from it |
| `GET /api/library/export?files=a,b` · `?blocks=x,y` | download a `.py` (one file) or `.zip`. Exporting single blocks uses `extract()` |
| `POST /api/library/import/preview` | upload `.py`/`.zip` → per-block validation + id clashes. **Runs no code.** |
| `POST /api/library/import` | commit, choosing per clash: overwrite / rename / skip |
| `POST /api/blocks` (existing) | gains `library: {file, id}` → embedded definition + `library_ref` |
| `POST /api/blocks/{id}/update_from_library` | swap in the current definition, keep params that still exist |

`GET /api/registry` keeps its shape but also lists library blocks (with
`source` and `file`), so the palette needs one fetch.

## 6. Frontend

Mock: https://claude.ai/artifact/QGEp3iNJeBWGfSStogzAU3 (the library list
there is still flat; per-file grouping is the change below).

1. **Palette**: a *Library* section, grouped by file (the file's docstring
   first line as the heading), with LIB badges. Broken blocks are greyed
   out with the error as a tooltip.
2. **Library panel**: a list grouped by file, then block. Search and tag
   filter. Built-ins are shown read-only under their own files. Block detail
   shows description, ports, form, *measures* (which hints it gets),
   change detection and streaming (both derived, shown so the user can see
   what the architecture will do), and usage in this project. Actions are
   per block (*Edit*, *Duplicate*, *Export block*, *Delete*) and per file
   (*Export file*, *New block in this file*).
3. **Editor**: edits a whole file in `CodeEditor.tsx`, with a live per-block
   checklist and errors on the exact line. Opening one block scrolls to it.
4. **Inspector**: *Save to library…* on custom blocks, asking for a new file
   or an existing file to add the block to. *From library: X v2 → Update to v3*
   with a diff on library-derived blocks.
5. **Import dialog**: one row per *block*, grouped by file, with validation
   status, clash state (*overwrite / rename / skip*) and expandable code.
6. **Built-in → library**: *Duplicate to library* copies a built-in (plus its
   helpers) into a library file. No extras are lost now, because extras are
   architecture.

## 7. Phasing

| Phase | Scope | Size |
|---|---|---|
| **1. Architecture for the extras** | Runner-side sampling, `lazy` port flags in fusion, `sources.py` change detection, hint rule table, `schema_diff` + declarative roles. Built-ins still use `BlockSpec`, but each extra is switched to the new mechanism one at a time, and the existing tests stay green at every step. | L |
| **2. File format + loader** | `BlockDefinition`, `parse_file`/`validate`, helper-aware compiler emission. Migrate built-ins module by module from `register_block(BlockSpec(...))` to `BLOCKS = {...}`. The goal is an identical registry and identical compiled scripts (checked by snapshot tests on the demo project). | L |
| **3. Library backend** | Library folder, embedded definitions + `library_ref`, the custom-block merge, API endpoints. | M |
| **4. UI: browse & add** | Palette section, library panel (read-only), drag to canvas. | M |
| **5. Authoring** | Editor, *Save to library*, *Duplicate to library*. | M |
| **6. Import / export + updates** | Preview-then-commit import, file/block export, outdated badge + *Update*. | M |
| **7. Extras** | TUI, several search paths, library blocks in the AI catalogue (§10). | S each |

Phases 1–2 are a refactor of existing code with no user-visible change, and
they can ship on their own. They are also where most of the risk is, so they
come first.

## 8. Risks and decisions

- **Imported files are code.** Opening someone's project today already
  means trusting their code. Preview never executes, the import dialog
  shows the code, and validation only *defines* functions.
- **Trust for `aggregate`.** That flag decides which tables an AI build may
  read row by row. The architecture honors it for `builtin` blocks and for
  library blocks the user has marked trusted in the panel. Imported blocks
  arrive untrusted, so their outputs reach the AI as schema + summary
  statistics only until the user opts in.
- **Edits never propagate silently** to models (§4). The cost is an
  explicit *Update* per model.
- **Id uniqueness across files.** Two files can't both define `psi_test`.
  This keeps `model.json` references simple (just the id, as today), at the
  cost of an occasional rename on import.

## 9. Migration notes (phase 1–2)

- **Bespoke metadata transforms (~40)** are the biggest unknown. The
  assumption is that most are "inherit + tag the column a param names with
  role R" or "add these columns", which `schema_diff` + `roles` cover.
  Each one gets audited, and any that doesn't fit becomes a new *generic*
  declarative kind (never a per-block function). If one genuinely can't be
  expressed, that's a signal to stop and redesign, not to add an escape hatch.
- **Hints with structure** (IV tables, data profile, stepwise selection)
  become `table` kinds with their rules in `hints.py`. The thresholds there
  stay as they are.
- `library.py` (804 lines, mixed I/O and transforms) splits into `io.py` and
  `transforms.py`. The other modules keep their names.
- `worker_pool.BASE_PRELOAD`, `run_worker` and `catalogue.ensure_blocks_registered`
  replace their hard-coded module lists with the loader.

## 10. Open questions

1. **Should the AI build loop see library blocks?** It would let a build use
   a team's house blocks. Proposal: phase 7, behind a setting, and only for
   trusted library blocks.
2. **Library location default:** `~/.modelmaker/blocks/` or next to the
   projects dir?
3. **Project-local block files** (`<project>/blocks/`, same format), so a
   project can ship its own block definitions? They aren't needed for
   portability (instances embed their code), so they're left out unless
   there's demand.
