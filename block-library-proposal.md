# Block Library — proposal

A user-owned library of reusable blocks: browse it in the UI, add blocks to
it (from scratch or from a block on the canvas), and import/export blocks as
files. The goal is to make Model-Maker customizable without touching the
package source.

**Design constraint:** a library block is exactly **one `.py` file**. No
manifest, no registry database, no sidecar JSON. Copying the file *is*
exporting the block; dropping a file into the folder *is* importing it.

---

## 1. What exists today (and what we reuse)

- **Registry blocks** (`modelmaker/blocks/*.py`): a plain function plus a
  `BlockSpec` registered into `BLOCK_REGISTRY` at import time. Fixed code,
  and they carry Python-only extras (`lazy_fn`, `probe`, `hints`).
- **Custom blocks** (`BlockInstance.is_custom=True`): the instance carries
  its own `code` (exactly one top-level function, see
  `graph._compile_code_to_fn`), its `inputs`/`outputs` ports and a
  JSON-declared `metadata_transform` (`metadata_transforms.resolve_metadata_transform`).
  The runner, the worker pool (`run_worker` gets `code` passed in), the
  compiler (emits `block.code` verbatim) and project save (code sidecar
  `.py` via `code_ref`) already handle them end to end.

A custom block is already 90% of a library block: code + ports + metadata
transform. What it lacks is a name/description/group/tags, a params form, and
somewhere to live outside a single project.

## 2. Core decision: the library is a set of templates, not registry entries

When a library block is dropped on the canvas it becomes a **custom block
instance**: a copy of the file's code, ports, metadata transform and form,
plus a `library_ref` that remembers where it came from.

Why this instead of registering library files into `BLOCK_REGISTRY`:

| | Template (proposed) | Registry entry |
|---|---|---|
| Runner / worker pool / compiler changes | none (custom path already works) | workers must load the library dir; registry reload at runtime |
| Project portability | automatic: the code is saved in the project's sidecar | a project breaks on a machine without that library file |
| Editing a library block | never silently changes existing models | changes every model that uses it, including validated ones |
| Updating instances | explicit "Update from library" (diff shown) | implicit |

For model development, the second-to-last row matters: a validated model
should not change because someone edited a shared block. Updates are
explicit, which leaves an audit trail.

## 3. The file format

```python
"""Winsorize numeric columns at the given quantiles.

Clips each listed column to its [lower, upper] quantile range, computed on
the input frame itself. Longer description shown in the library panel.
"""

BLOCK = {
    "name": "winsorize",                 # unique id = file stem; [a-z0-9_]+
    "display_name": "Winsorize",
    "version": 1,                        # bumped on each save from the UI
    "block_type": "standard",            # input | standard | output
    "group": "preprocessing",            # palette group
    "tags": ["preprocessing"],
    "inputs": [{"name": "in", "type": "dataframe"}],
    "outputs": [{"name": "out", "type": "dataframe"}],
    "metadata_transform": {"kind": "passthrough"},
    "form": [
        {"key": "columns", "label": "Columns", "kind": "columns"},
        {"key": "lower", "label": "Lower quantile", "kind": "number", "step": 0.01},
    ],
    "author": "",
}


def winsorize(df: pl.DataFrame, columns: list[str], lower: float = 0.01, upper: float = 0.99) -> pl.DataFrame:
    import numpy as np  # extra imports go inside the function (see below)

    return df.with_columns(
        pl.col(c).clip(pl.col(c).quantile(lower), pl.col(c).quantile(upper)) for c in columns
    )
```

Rules (all checked by one `validate()` function):

1. **Module docstring:** the first line is the summary and the rest is the
   long description. This replaces a separate `description` key.
2. **`BLOCK` is a pure literal**, read with `ast.literal_eval`, never
   executed. Listing the library or previewing an import therefore runs
   **no code**.
3. **Exactly one top-level function**, the same rule as custom blocks
   today. `pl` is in scope, and any other import goes inside the function.
   The compiler emits only the function's source, so module-level imports
   would be lost from the compiled script. This keeps the
   "compiles to one clean script" property without changing the compiler.
4. Every `form[].key` must be a parameter of the function (the same check
   `register_block` does today).
5. `metadata_transform` must be one of the JSON kinds `resolve_metadata_transform`
   already accepts.
6. Port types must be in `PortType`. Unknown tags are kept but flagged
   (they won't appear in the AI catalogue's tag browse until added to `TAGS`).

No `lazy_fn`, `probe` or `hints`. Those stay a privilege of built-in blocks,
the same as custom blocks today. No `aggregate_outputs` either (see §8).

## 4. Where library files live

```
~/.modelmaker/blocks/            # user library (default; override with MODELMAKER_BLOCK_LIBRARY)
    winsorize.py
    psi_by_segment.py
```

- A single flat folder. The file stem is the block's id, so ids are unique
  by construction and a name clash on import is a file-exists clash.
- The folder can be a git repo or a shared/network drive. Team sharing then
  needs no extra feature.
- *(Later, optional)* a second read-only search path such as
  `MODELMAKER_BLOCK_LIBRARY=~/.modelmaker/blocks:/shared/team-blocks`. The
  first match wins and the panel shows which path a block came from.

## 5. Backend

### New module: `modelmaker/block_library.py` (~200 lines, no state)

Everything is read from disk on request. There is no cache, and that is
fine for tens to hundreds of small files. If it ever isn't, add an
mtime-keyed cache.

```python
@dataclass
class LibraryBlock:
    name: str
    path: Path
    meta: dict[str, Any]      # the BLOCK literal
    summary: str
    description: str
    code: str                 # the function source only (what an instance gets)
    source: str               # the whole file (what export returns)
    sha: str                  # hash of `source`, used for "update available"
    errors: list[str]         # validation problems; a broken file still lists, flagged

def library_dir() -> Path
def list_blocks() -> list[LibraryBlock]
def load(name: str) -> LibraryBlock
def parse(source: str) -> LibraryBlock          # ast only, never exec
def validate(block: LibraryBlock) -> list[str]  # rules in §3, plus a trial _compile_code_to_fn
def render(meta, docstring, code) -> str        # BLOCK dict + fn back into one file (for "save from canvas")
def save(source: str, *, overwrite: bool = False) -> LibraryBlock
def delete(name: str) -> None
```

`validate` is the only place that runs code: `_compile_code_to_fn(code)`,
which defines the function without calling it. It runs on save and import,
after the user has seen the code.

### Model change: `BlockInstance.library_ref`

```python
# Set when the block was created from a library file: {"name": ..., "version": ..., "sha": ...}.
# Descriptive only (outside compute_key), like provenance.
library_ref: dict[str, Any] | None = None
# Params form for a custom block (FieldSpec dicts); today only library blocks set it.
form: list[dict[str, Any]] | None = None
```

Both go through `project.graph_to_dict`/`dict_to_graph`. `_form_for_block` in
`api.py` returns `block.form` for a custom block when it is set (today it
returns `None`), so library blocks get the same labelled fields as built-ins
in both the web UI and the TUI.

### API (`api.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/library` | list: name, display_name, summary, group, tags, ports, version, sha, errors |
| `GET /api/library/{name}` | full entry incl. code + description |
| `POST /api/library` | create/overwrite from `{source}` (the editor's full file text) |
| `POST /api/library/from_block/{block_id}` | "Save to library" for a canvas custom block → `render()` + `save()` |
| `DELETE /api/library/{name}` | delete the file |
| `GET /api/library/{name}/export` | the `.py` file as a download |
| `GET /api/library/export?names=a,b` | `.zip` of several files |
| `POST /api/library/import/preview` | upload `.py`/`.zip` → parsed entries, errors, conflicts. **Runs no code.** |
| `POST /api/library/import` | commit the previewed files, with per-file `overwrite`/`rename`/`skip` |
| `POST /api/blocks` (existing) | gains `library: str`, which expands into the existing custom-block fields + `library_ref` |
| `POST /api/blocks/{id}/update_from_library` | replace code/ports/transform/form with the current file, keep params that still exist |

`GET /api/graph` marks a block `library_outdated: true` when its
`library_ref.sha` no longer matches the file. That costs one hash per
referenced name per poll, or a cheap mtime check.

### Untouched

Runner, worker pool, compiler, cache keys and the AI build loop need no
changes, because instances are ordinary custom blocks.

## 6. Frontend

1. **Palette → "Library" group** (`Palette.tsx`): library blocks are listed
   under their own `group` like registry blocks, with a small badge. Dragging
   one creates `{kind: 'library', name}`, which maps to `POST /api/blocks
   {library: name}`. Broken files are shown greyed out with the error as a
   tooltip.
2. **Library panel** (new `LibraryPanel.tsx`, opened from the toolbar): a
   list with search/tag filter, and a detail pane (description, ports,
   form fields, read-only code). Actions: *New*, *Edit*, *Duplicate*,
   *Export*, *Delete*, *Import…*, and a multi-select *Export selected*.
3. **Block editor** (reuse `CodeEditor.tsx`): edits the whole file as text.
   That keeps one source of truth and avoids a form editor that has to stay
   in sync with the file. Save → `POST /api/library`, and validation errors
   show inline. *(Later:* a small form for name/ports/tags that rewrites the
   `BLOCK` literal.)*
4. **Inspector on a custom block**: a *Save to library…* button (asks for a
   name/display name/group). On a library-derived block: "from library
   `winsorize` v3", plus an *Update to v4* button with a code diff when it
   is outdated.
5. **Import dialog**: file picker (`.py` or `.zip`) → preview table (name,
   summary, ✓/errors, *exists → overwrite / rename / skip*) with the code
   visible per row → *Import*.
6. **Built-ins in the library panel**: shown read-only with a *Duplicate to
   library* action, which writes a single file from the built-in's `fn`
   source + spec. Built-ins that rely on `lazy_fn`/`probe`/`hints` lose those
   extras in the copy, and the panel says so.

TUI: phase 2. It only needs the palette listing and "add from library",
both over the same endpoints via `tui/client.py`.

## 7. Phasing

| Phase | Scope | Size |
|---|---|---|
| **1. Backend core** | `block_library.py` (parse/validate/list/save/delete/render), `library_ref` + `form` on `BlockInstance` + project round-trip, `POST /api/blocks {library}`, list/get/save/delete endpoints. Tests: parse without exec, validation rules, instance round-trip, compile output identical to the equivalent custom block. | M |
| **2. UI: browse & add** | Palette library group, library panel (read-only), drag to canvas, form rendering for library instances. | M |
| **3. Authoring** | Editor (new/edit/duplicate), *Save to library* from a canvas block, *Duplicate to library* for built-ins. | M |
| **4. Import / export** | Single + zip export, preview-then-commit import with conflict handling. | S |
| **5. Updates** | `library_outdated` flag, *Update from library* with diff. | S |
| **6. Extras** | TUI support, multiple search paths, AI catalogue integration (§9). | S each |

## 8. Risks and decisions

- **Imported files are code.** Importing a `.py` is running someone
  else's Python, the same trust model as opening someone's project today
  ("trusted, project-local code"). Mitigations that cost little: preview
  never executes, the import dialog shows the full code, and the
  validation exec (`_compile_code_to_fn`) runs only on *Import*. Nothing
  runs until a block is actually executed on the canvas.
- **`aggregate_outputs` is not allowed in library files.** That flag decides
  which tables an AI build may read row by row (`agent/tools.summarize_value`).
  The code comment says it lives on the registry spec "so a block instance
  (or custom code) can never claim it for itself". Library blocks follow
  that rule, so an imported file cannot opt its outputs into being sent to
  an LLM.
- **Name clashes with built-ins.** Library names live in their own space
  (instances are custom blocks with `category="lib_<name>"`), so they can
  never shadow a built-in category.
- **Edits to library files never propagate silently.** This is deliberate
  (see §2). The cost is that a bug fix has to be pulled into each model with
  *Update*. The outdated badge makes that visible.

## 9. Open questions

1. **Migrate built-ins to the single-file format?** It is tempting for
   consistency, but built-ins use `lazy_fn`, `probe` and `hints`, which a
   literal-only `BLOCK` dict can't express. Proposal: not now. Built-ins stay
   as they are and show read-only in the library panel.
2. **Should the AI build loop see library blocks?** Adding them to
   `agent/catalogue.py` would let a build use a team's house blocks, which
   is valuable. It needs a decision on whether unvalidated user code should
   be offered to the agent. Proposal: phase 6, behind a setting.
3. **Library location default.** Is `~/.modelmaker/blocks/` right, or should
   it sit under the default projects dir so it is visible next to projects?
4. **Project-local blocks?** A `<project>/blocks/` folder using the same
   format would let a project ship its own templates. It isn't needed for
   portability (instances already embed their code), so it is left out
   unless there's demand.
