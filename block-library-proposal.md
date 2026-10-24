# Block Library — proposal

A user-owned library of reusable blocks: browse it in the UI, create blocks
(by hand or with AI), and import/export them as files. The goal is to make
Model-Maker customizable without touching the package source.

> **Status: implemented** (phases 1–7). The author's guide is
> [docs/block-authoring.md](docs/block-authoring.md). The implementation
> differs from this proposal in a few places:
>
> - **Hints** are named per block (`"hints": "auc_gini"`), not per output
>   (`measures`/`table`). The rules moved, unchanged, into a table in
>   `modelmaker/agent/hint_rules.py`, and each rule reads a specific output
>   shape.
> - **Metadata** uses one `"meta"` key: `inherit` (`all`/`none`/a
>   port/a list param), `new_role`, and `roles` rules (`column` /
>   `column_param` + `fallback` / `column_suffix`, with `if_taken` and
>   `displaced` for unique roles). All ~70 transforms were expressed this
>   way. The only loss is the `missing_treatment:<method>` column tag,
>   which nothing read.
> - **Breaking change, no legacy loading.** Projects with old canvas-only
>   custom blocks are refused on load with a clear message. Project files
>   are format 2: a placed library block's code is stored in
>   `blocks/<block id>.py` next to `model.json`.
> - **`aggregate`** is honored for built-ins only. Library blocks show the
>   flag, but there is no "trusted library block" marking yet.
> - **Change detection for SQL** reads the block instance's `probe_query`
>   setting, which is never passed to the function.
> - **Decisions on the open questions:** the library lives in
>   `~/.modelmaker/blocks/` (`MODELMAKER_BLOCK_LIBRARY` overrides it); AI
>   drafts stay until the user keeps or discards them; agent builds *use*
>   library blocks (the user's own and earlier AI drafts; input blocks and
>   invalid blocks excluded) unless the build turns that off; there are no
>   project-local block files.
> - **Not done yet (phase 8):** a TUI library view, several search paths.

**Design rules**

1. **A block file describes the block's core function and nothing else:**
   what it computes (a plain function), its interface (ports and params
   form), what its outputs *are* (e.g. "this table is aggregate", "this value
   is a Gini") and what it needs (packages). Sampling, change detection,
   streaming, sinks, AI decision hints, privacy of outputs, caching and
   grouping are **architecture**, applied to every block the same way.
2. **Blocks live in plain `.py` files, one or more blocks per file.** Group
   related blocks together (e.g. all PD backtests in `pd_backtests.py`). No
   manifest, no registry database, no sidecar JSON.
3. **Built-in blocks use the same format.** With the extras moved into the
   architecture, nothing is left that only a built-in can express, so there
   is one loader, one validator and one code path.
4. **Custom blocks are created first, then used.** Custom blocks (hand-written
   or AI-written) stay, but a custom block is always *created* in the block
   library (by hand, or with **Create with AI**, §7) and saved there before
   it is placed on the canvas. What goes away is code that lives only
   inside one canvas node. A block's code therefore always has a name, a
   version, a description and a home, and every custom block is reusable
   in other models from the moment it exists.

---

## 1. Today vs. proposed

Today a registry block is a function plus a `BlockSpec` carrying
block-specific Python extras. Each maps to a piece of shared architecture
the block only *describes* its way into:

| Today (per block, Python) | Count | Proposed (architecture, driven by the block's description) |
|---|---|---|
| `sample_rows` param on input blocks | ~10 | The **runner** applies sampling: it calls `.head(n)` on the input block's output, and that pushes down for free when the function returns a `LazyFrame`. Blocks never see `sample_rows`. |
| `probe=` (`_probe_path_mtime`, `_probe_sql`) | 5 | **Change detection** is picked from the *param kinds*. A field of kind `path` gets an mtime/size fingerprint, and kind `sql` (+ connection) gets the shared SQL fingerprint. |
| `lazy_fn=` (read_csv/read_parquet twins + 5 identical reuses) | 7 | An input port marked `"lazy": true` means "this function works on a `LazyFrame`", and the **fusion** pass uses that. Input readers return `pl.scan_*(...)` and the runner collects at checkpoints. Twin functions disappear. |
| `lazy_sink_fn=` (write_csv) | 1 | The same port flag on an output block: when fused, it receives a `LazyFrame`, and `df.lazy().sink_csv(path)` works in both cases. |
| `hints=` (`_auc_gini_hints`, `_psi_test_hints`, …) | 31 | **A hint rule table** in `agent/hint_rules.py`, keyed by rule name; a block names its rule (`"hints": "auc_gini"`). Thresholds stay in `hints.py`. *(Proposed: keyed per output by `measures`/`table`.)* |
| `aggregate_outputs=` | 11 | A port flag `"aggregate": true`. The architecture decides whether to honor it, based on the block's source (§10). |
| `metadata_transform=` (20 `infer_dtypes`, 9 generic, ~40 bespoke) | ~70 | The default is **automatic schema-diff inheritance** (upstream columns keep their metadata, new ones are inferred), plus declarative role annotations (`"roles": [{"column_param": "pd_column", "role": "predicted"}]`). Each bespoke transform is audited during migration (§12). |
| `group_by`, caching, column-role overrides | n/a | Already architecture. Unchanged. |
| `is_custom` blocks (code stored on the canvas node) + `add_custom_block` agent tool | n/a | Custom blocks become library blocks: create (by hand or with AI, §7) → saved in the library → placed. The canvas-only code path is removed. |

## 2. The file format

```python
"""PD backtests: tests that compare predicted PDs with observed defaults.

The module docstring describes the file (the group heading in the panel).
"""

BLOCKS = {
    "binomial_test": {
        "display_name": "Binomial test per grade",
        "version": 2,
        "block_type": "output",
        "group": "tests",
        "tags": ["pd", "validation"],
        "requires": ["scipy>=1.11"],
        "inputs": [{"name": "in", "type": "dataframe", "lazy": False}],
        "outputs": [
            {"name": "table", "type": "dataframe", "aggregate": True},
            {"name": "metric", "type": "scalar_metric"},
        ],
        "hints": "grade_backtest",
        "form": [
            {"key": "grade", "label": "Grade column", "kind": "column"},
            {"key": "pd_column", "label": "PD column", "kind": "column", "auto_role": "predicted"},
            {"key": "alpha", "label": "Significance", "kind": "number", "step": 0.01},
        ],
        # v1 called it "confidence"; Update carries old values across
        "renamed": {"confidence": "alpha"},
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
    """Per-grade Jeffreys (beta posterior) test."""
    ...


def _per_grade(df, grade, pd_column, default_flag):
    # shared helper: private (leading underscore), not a block
    return df.group_by(grade).agg(...)


def _example_binomial_test():
    # optional self-check: tiny input -> call -> assert; run by "Try it" and `modelmaker check-blocks`
    df = pl.DataFrame({"grade": ["A", "A", "B"], "pd": [0.01, 0.01, 0.2], "default_flag": [0, 0, 1]})
    table, metric = binomial_test(df, "grade", "pd", "default_flag")
    assert table.height == 2
```

Rules (all checked by one `validate()` function):

1. **`BLOCKS` is a pure literal**, read with `ast.literal_eval` and never
   executed. Listing the library or previewing an import runs **no code**.
2. **Every key in `BLOCKS` names a public top-level function, and every
   public top-level function has a key.** `_`-prefixed functions are
   helpers. `_example_<id>` functions are that block's self-checks.
3. **Each block function's docstring is its description** (first line =
   summary). The module docstring describes the file.
4. **Block ids are function names, unique across everything loaded.** A
   new library block can't take a built-in's id. If a *later package
   version* adds a built-in with an existing library block's id, the
   built-in wins in the palette and the library block is flagged
   *"shadowed by built-in `x` — rename"* with a one-click rename. Models are
   unaffected, because placed library blocks embed their definition (§4).
5. **Only `pl` at module level; other imports go inside functions** (the
   current convention). The compiler emits each used block function plus
   the helpers it references (found transitively from AST names). Examples
   are never emitted. The compiled script stays one self-contained file.
6. **`requires`** lists pip requirement strings. The loader checks
   them with `importlib.metadata`, without importing anything. A block with
   unmet requirements lists with a "needs scipy>=1.11" badge, can't be
   placed, and the import dialog shows it. The compiled script header lists
   the combined requirements of the blocks it uses.
7. **`renamed`** maps old param names to new ones, so *Update* carries
   values over (§4). Ports may be renamed the same way
   (`"renamed_ports": {"in": "sample"}`) to keep wires connected.
8. Every `form[].key` must be a parameter of its function. Port types must
   be in `PortType`. `measures`/`table` names must be known to the hint
   table (unknown ones are kept but flagged). Tags must be in the tag
   vocabulary (§11).
9. **No executable extras.** There is no key that takes a function. If a
   block seems to need one, that's a missing piece of architecture.

## 3. Where block files live

```
modelmaker/blocks/               # built-in, shipped with the package, read-only
    io.py  transforms.py  binning.py  stat_tests.py  modelling.py  ...
~/.modelmaker/blocks/            # user library (override with MODELMAKER_BLOCK_LIBRARY)
    pd_backtests.py
    preprocessing.py
```

The same loader scans both. The library folder can be a git repo or a
shared drive. *(Later:* several search paths.)*

## 4. How a block reaches a model

All blocks resolve to one internal type:

```python
@dataclass
class BlockDefinition:
    id: str                 # function name
    meta: dict[str, Any]    # its BLOCKS entry
    code: str               # the function + the helpers it references
    source: Literal["builtin", "library"]
    origin: str | None      # file path
    sha: str                # hash of meta + code
```

- **Built-in block** on the canvas: the instance references the id, **and
  records the definition's `sha`** in `model.json`. On load, if the
  installed package's definition differs (an upgrade changed it), the block
  shows *"built-in changed since this model was saved"* with a diff and an
  *Accept* action. The `sha` is part of the cache key, so a changed built-in
  never serves a stale cached result. Built-ins can change behaviour after
  an upgrade, but never silently.
- **Library block** on the canvas: the instance **embeds a copy** of the
  definition and records `library_ref = {file, id, version, sha}`. The
  project runs on a machine without that library, and edits to the library
  never silently change a model. When the library has a newer version, the
  block shows *Update to vN*.
- **Update dialog** (library blocks, and *Accept* for built-ins): shows the
  code diff **and the structural impact**: params carried over (incl.
  `renamed`), params dropped, new required params, wires that stay
  (incl. `renamed_ports`), and wires that would be disconnected because a
  port was removed or changed type. You confirm with that information in
  front of you.
- **Legacy custom blocks** (existing `is_custom` instances in old
  projects) still load. Each becomes an embedded definition with
  `source="library"` and no `library_ref`, flagged *"not in a library — Save
  to library"*. That action asks for a file and a name and writes it out.
  New ones can't be created.

The runner, compiler, forms, hints and fusion all consume `BlockDefinition`
and never branch on its source (except §10's trust rule for `aggregate`).

**Provenance in the compiled script.** Each emitted function gets a header
line naming its origin:

```python
# === binomial_test | library pd_backtests.py v2 | sha 3f9c1a2 ===
# === logistic_regression | built-in modelmaker 0.14.2 | sha 81bd0e4 ===
```

The script header lists the requirements and the Model-Maker version it was
compiled with.

## 5. Backend

### `modelmaker/block_library.py` (loader + library I/O, no state)

```python
def parse_file(source: str, origin: str) -> list[BlockDefinition]  # ast only, never exec
def validate(defs, taken_ids) -> dict[str, list[str]]       # id -> errors (rules in §2)
def check_requirements(defs) -> dict[str, list[str]]        # id -> unmet requirements
def run_examples(file, ids=None) -> dict[str, ExampleResult]  # in a worker, with a timeout
def load_builtin() -> list[BlockDefinition]                 # replaces register_block() at import
def list_library() -> list[LibraryFile]
def save_file(name, source, *, expected_sha=None) -> LibraryFile
def add_block(file: str | None, definition) -> LibraryFile  # new file, or append to one
def extract(file, ids) -> str                               # just these blocks + their helpers
def delete_file(name) / delete_block(file, id)
def reload() -> None                                        # rescan; bumps a library generation
```

- `validate` runs code in exactly one place: it *defines* the functions
  without calling them. `run_examples` calls code, but only when the user
  asks (*Try it*) or runs the CLI.
- Editing one block in a multi-block file rewrites only that block's
  `BLOCKS` entry and function, using `ast` positions. Comments and the rest
  of the file stay untouched.
- **Safe writes**: write a temp file and then `os.replace`. A save
  carries the `expected_sha` the editor opened, and a mismatch (someone else
  saved meanwhile, e.g. on a shared drive) is refused. The editor then
  offers *reload & re-apply* instead of the last writer winning.

### Architecture modules that take over the extras

- `runner.py`: sampling via `.head(n)`, fusion eligibility from `lazy` port flags.
- `sources.py` (new): change-detection strategies by param kind. Replaces `probe`.
- `agent/hints.py`: rules keyed by `measures`/`table`. Replaces 31 per-block functions.
- `metadata_transforms.py`: `schema_diff` default + declarative `roles`.
- `worker_pool.py` / `run_worker.py` / `catalogue.ensure_blocks_registered`:
  use the loader instead of hard-coded module lists. Each task carries the
  library generation, so a worker started before a reload re-imports
  before it runs anything.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/library` | files → blocks (id, display_name, summary, group, tags, ports, version, sha, errors, unmet requirements, shadowed) |
| `GET /api/library/files/{file}` | full source + parsed blocks + sha |
| `PUT /api/library/files/{file}` | save a whole file (`expected_sha` required for an existing file) |
| `DELETE /api/library/files/{file}` · `…/blocks/{id}` | delete a file or one block |
| `POST /api/library/reload` | rescan the folder (also offered automatically when file mtimes change) |
| `POST /api/library/files/{file}/try` | run `_example_*` checks (or one block's on a chosen canvas input) |
| `GET /api/library/export?files=…` · `?blocks=…` | `.py` (one file) or `.zip`. Single blocks use `extract()` |
| `POST /api/library/import/preview` | upload → per-block validation, requirements, id clashes. **Runs no code** |
| `POST /api/library/import` | commit, choosing per clash: overwrite / rename / skip |
| `POST /api/library/ai/draft` | Create/edit with AI (§7): instruction (+ optional file, block, canvas input) → proposed file source |
| `POST /api/blocks` (existing) | gains `library: {file, id}` → embedded definition + `library_ref` |
| `POST /api/blocks/{id}/update_preview` · `…/update` | structural impact + diff, then apply (library *Update*, built-in *Accept*) |
| `POST /api/blocks/{id}/save_to_library` | legacy custom blocks only |

**Removed:** creating custom blocks (`POST /api/blocks` with `code`),
`PATCH` of a block's `code`/`metadata_transform`, and `/draft` for code.
`/draft` stays for *suggesting param values* on any block. `/suggest_fix`
moves to the library (§7).

### CLI

`modelmaker check-blocks [DIR]`: validates every file, checks requirements
and runs every `_example_*`, with exit status 1 on any failure. A team
library repo can run it in CI.

## 6. Frontend

Mock: https://claude.ai/artifact/QGEp3iNJeBWGfSStogzAU3 (to be updated:
the library is still flat there, and the palette and canvas still show AI blocks).

1. **Palette**: the *+ AI input / standard / output block* buttons go away.
   A *Library* section is grouped by file, with LIB badges, and its header
   has a **✦ Create with AI** button that opens §7 directly. Blocks that are
   broken, have unmet requirements or are shadowed are greyed out with the
   reason as a tooltip.
2. **Library panel**: grouped by file, then block, with search and a tag
   filter. Built-ins are shown read-only under their own files. Block detail
   shows description, ports, form, requirements, *measures* (which hints it
   gets), change detection and streaming (derived, so the user can see what
   the architecture will do), example status and usage in this project.
   Actions per block: *Edit*, *Edit with AI*, *Try it*, *Duplicate*,
   *Export block*, *Delete*. Per file: *Export file*, *New block in this file*.
   Header actions: *✦ Create with AI*, *New block*, *Import…*, *Export all*, *Reload*.
3. **Editor**: edits a whole file in `CodeEditor.tsx`, with a live per-block
   checklist and errors on the exact line, plus a *Try it* pane. *New block*
   starts from a commented template (§11).
4. **Inspector**: library blocks show *From library: X v2* and *Update to
   v3* (opens the update dialog). A red library block shows *Fix with AI*,
   which opens §7 with the error. Built-ins show *changed since saved* when
   applicable. Legacy custom blocks show *Save to library*.
5. **Update dialog**: the code diff plus the structural impact (§4), with
   wires that would break highlighted on a mini version of the graph.
6. **Import dialog**: one row per block, grouped by file, with validation,
   requirements, clash state and expandable code.

## 7. Create with AI

This is how AI-written custom blocks are made from now on, replacing today's
canvas-level AI blocks and draft/fix flow. The order is always **create →
save to library → use**: AI produces a **library block file entry**, through
the same validator as hand-written ones, and only a saved block can be
placed.

**Flow** (a side panel in the library, also reachable from the palette and
from a red block's *Fix with AI*):

1. **Describe it.** Free text ("PSI per segment between a reference and a
   current sample, flag segments above 0.25"). Optional settings:
   - **Context:** pick a block on the canvas whose output becomes the
     example input. The AI sees its **schema and column roles only**, the
     same leakage rules as today (`agent/leakage.py`). Rows are never sent.
   - **Where:** a new file, or *add to an existing file*. Then the AI also
     sees that file's other blocks and helpers, reuses helpers and matches
     the file's style.
   - **Edit / fix mode:** start from an existing block (*Edit with AI*), or
     from a red block's code + error + input schema (*Fix with AI*).
2. **Draft.** The AI returns a full `BLOCKS` entry (ports, form, `measures`,
   `requires`, `roles`), the function and an `_example_<id>`. The prompt
   carries the file-format rules and the known `measures`/`table` names, so
   drafts get hints and a self-check from the start.
3. **Review.** The panel shows the drafted code in the editor (editable),
   the live checklist, a preview of the node and params form, and the
   diff when it is added to an existing file or edits a block.
4. **Try it.** Runs the example and, if a canvas block was picked as
   context, a real run on that block's (sampled) output. The result is
   shown, but nothing is sent back to the AI unless the user clicks *Refine*.
5. **Refine.** Follow-up instructions ("make alpha a param", "also return a
   chart") iterate on the draft. A failed *Try it* offers *Fix with AI* with
   the error filled in.
6. **Save.** Writes to the library (version 1, or +1 when editing), with an
   *Add to canvas* checkbox (default on when started from the palette or
   a red block). For *Fix with AI* on a placed block, saving also runs that
   block's *Update* step, showing the update dialog when the structure changed.

**Agent builds.** `add_custom_block`/`update_custom_block` are replaced by
`create_library_block`/`update_library_block`, which call the same draft
pipeline and validator. To avoid every build filling the user's library,
an agent build writes into a per-build file `~/.modelmaker/blocks/drafts/<build>.py`.
The library panel shows that file under *AI drafts* with *Keep* (move to a
chosen file) and *Discard*. Blocks placed by the build embed their
definition as usual, so discarding a draft file never breaks the model.
`BuildOptions.allow_custom_blocks` is renamed `allow_new_blocks`, and the
`custom_blocks` limit carries over.

## 8. Phasing

| Phase | Scope | Size |
|---|---|---|
| **1. Architecture for the extras** | Runner-side sampling, `lazy` port flags in fusion, `sources.py`, hint rule table, `schema_diff` + declarative roles. Built-ins still use `BlockSpec`, and each extra is switched over one at a time with the tests green at every step. | L |
| **2. File format + loader** | `BlockDefinition`, parse/validate, `requires`, helper-aware compiler emission + provenance headers, built-in `sha` pinning. Migrate built-ins module by module, aiming for an identical registry and identical compiled scripts apart from the new header lines (snapshot tests on the demo project). | L |
| **3. Library backend** | Library folder, embedded definitions + `library_ref`, safe writes, reload, update preview/apply with `renamed`, legacy custom-block loading, CLI `check-blocks`. | M |
| **4. UI: browse, add, update** | Palette section, library panel, drag to canvas, update dialog, built-in *changed since saved*. | M |
| **5. Authoring** | Editor + template, *Try it*, *Duplicate to library*, *Save to library* for legacy blocks. | M |
| **6. Create with AI** | Draft endpoint + prompts, the AI panel, *Edit/Fix with AI*, agent tools moved to library drafts. **Then** remove canvas AI blocks (palette buttons, custom-code create/patch). | M |
| **7. Import / export** | Preview-then-commit import with hardening (§9), file/block export. | S |
| **8. Extras** | TUI, several search paths, library blocks in the AI catalogue (§13). | S each |

Canvas AI blocks are removed only at the end of phase 6, once Create with
AI covers everything they did, so there is never a release without a
way to have AI write a block.

## 9. Operational details

- **Files changing on disk.** A `git pull` in the library folder or a
  colleague saving on a shared drive changes files under a running server.
  `GET /api/library` compares directory mtimes and the panel shows *Library
  changed on disk — Reload*. Reload bumps the library generation, and
  workers re-import before their next task. Placed blocks are unaffected
  (they embed their definition) apart from new *Update* badges.
- **Concurrent edits** are handled by `expected_sha` + atomic replace (§5).
  Two people saving the same file get a refusal plus a re-apply, never a
  silent overwrite.
- **Import hardening.** Only `.py` (or `.zip` of `.py`), max 1 MB per file
  and 10 MB per upload. Zip entries must be plain names with no directories,
  `..` or absolute paths (no zip-slip). At most 200 blocks per import.
  `ast.parse` runs on the size-limited text, and a file that fails to parse
  is listed as an error, never executed.
- **Examples run sandboxed by process, not by permission.** *Try it* and
  `check-blocks` run examples in a worker with a timeout (30 s default),
  so a hanging example can't freeze the server. They are still trusted code.

## 10. Risks and decisions

- **Imported files are code.** Preview never executes, the import dialog
  shows the code, validation only *defines* functions, and examples run only
  on request.
- **Trust for `aggregate`.** That flag decides which tables an AI build may
  read row by row. It is honored for built-ins and for library blocks the
  user has marked trusted. Imported blocks, and AI-drafted blocks until they
  are kept, are untrusted: their outputs reach the AI as schema + summary
  statistics only.
- **Nothing changes a model silently:** library edits need *Update*, and
  built-in changes need *Accept* (§4).
- **Removing canvas AI blocks** gives up a quick "just make me a block here"
  flow. *Create with AI* from the palette with *Add to canvas* on is meant to
  be just as quick: describe it, review, save, and it's on the canvas.

## 11. Authoring support

- **Template.** *New block* (and a new file) starts from a commented
  template covering the docstring, a `BLOCKS` entry with every key
  explained inline, the function and an `_example_`. The template is a
  built-in file, so it also serves as reference documentation.
- **Guide.** `docs/block-authoring.md` covers the file rules, port flags
  (`lazy`, `aggregate`), `measures`/`table` names and what each triggers,
  `roles`, `requires`, `renamed`, examples and the CLI.
- **Tag vocabulary.** Tags stay a fixed list (`agent/catalogue.py`'s `TAGS`)
  so the palette filter and AI catalogue stay coherent. A library folder may
  add its own in an optional `tags.json`. Unknown tags are flagged, not
  dropped.

## 12. Migration notes (phases 1–2)

- **Bespoke metadata transforms (~40)** are the biggest unknown. The
  assumption is that most are "inherit + tag the column a param names" or
  "add these columns", which `schema_diff` + `roles` cover. Each one gets
  audited, and any that doesn't fit becomes a new *generic* declarative kind,
  never a per-block function. If one genuinely can't be expressed, stop and
  redesign rather than add an escape hatch.
- **Hints with structure** (IV tables, data profile, stepwise selection)
  become `table` kinds with their rules in `hints.py`.
- `library.py` (804 lines, mixed I/O and transforms) splits into `io.py` and
  `transforms.py`. The other modules keep their names.
- The agent's custom-block code (`agent/tools.py` `add_custom_block`,
  `_validate_custom`, `prompts._custom_contract`) is reworked into the
  library draft pipeline in phase 6, and its leakage checks are reused as-is.

## 13. Open questions

1. **Should agent builds be able to *use* library blocks** (not just create
   drafts)? It would let a build use a team's house blocks. Proposal: phase
   8, behind a setting, trusted library blocks only.
2. **Library location default:** `~/.modelmaker/blocks/` or next to the
   projects dir?
3. **Project-local block files** (`<project>/blocks/`)? They aren't needed
   for portability (instances embed their code), so they're left out
   unless there's demand.
4. **AI drafts retention:** auto-delete discarded or never-kept build draft
   files after N days, or keep them until the user acts?
