# Model-building tool — architecture plan

A SAS Enterprise Guide–style visual pipeline builder for financial model
development in Python. Blocks contain functional Python code, are wired
together via Polars DataFrames carrying metadata, organized visually into
phase-labelled swimlanes, and compile down to a single, human-readable,
self-contained Python script.

This is a fresh design — **not** built on the existing Modelmaker
(three-panel chat/IDE/iframe) tool.

---

## 1. Core data contract: `DataFramePacket`

Everything flowing over a wire is a packet, not a bare `pl.DataFrame`:

```
DataFramePacket:
    data: pl.DataFrame
    schema_meta: dict[str, ColumnMeta]     # per-column
    summary: dict[str, ColumnStats] | None # computed on demand, cached
    lineage: list[str]                     # block ids that touched it
```

`ColumnMeta` per column: description, **field function** (id / target /
weight / feature / date / segment / excluded / …), dtype, and optional
role-specific tags (e.g. `categorical`, `monotonic_required`, `pd_driver`).
This is what lets downstream blocks auto-configure instead of the user
re-typing column names everywhere.

- Every block declares not just a data transform but a **metadata
  transform** (select narrows it, join merges it, groupby invalidates
  row-level stats but not column roles, etc.).
- Metadata is authored/updated at **design time**, largely static — an LLM
  can be used to *propose* a block's metadata transform when the block is
  authored, but the result is a fixed declaration stored with the block,
  not something recomputed at runtime. Execution is deterministic: no LLM
  in the execution path.
- Summary stats are computed on demand (when a block's preview panel asks
  for them), not eagerly on every wire.

## 2. Block model — functional, partially executable

A block is a pure function:

```
def block(inputs: dict[str, DataFramePacket], params: dict) -> dict[str, DataFramePacket]
```

No side effects, no hidden state. This enables partial execution: the tool
runs block-by-block, caches each output keyed on
`(block_id, hash(inputs), hash(params), code_version)`, and only re-runs
downstream of whatever changed. Same mechanism drives single-file
compilation later (topologically sort + inline the function bodies).

### Block catalog

- **Input blocks** — read CSV/parquet/DB/etc., infer initial `ColumnMeta`
  (dtype at minimum; role left `unassigned` until tagged manually or via
  LLM suggestion from column names/samples).
- **Standard library blocks** — filter, join, groupby/agg, missing-value
  treatment, WoE/binning, scaling, train/test split, fit model
  (logistic/GBM/etc.), score, calibration/discrimination tests. Each ships
  with a declared input→output metadata transform.
- **LLM-authored blocks** — user describes intent in natural language; LLM
  generates the function body against the fixed block signature, using
  input metadata as context. Shown for review/edit before being wired in.
- **Output blocks** — tables, (interactive, e.g. Plotly) graphs. Sinks:
  they don't need to emit a packet onward, just render / write to disk.

### Ports

Blocks expose named input/output ports, not just one-in-one-out (e.g. a
`join` block has `left`/`right` inputs; a split block has `train`/`test`
outputs; a model-fit block may output both a fitted model object and a
scored dataframe). Ports are typed (`dataframe` / `model` /
`scalar-metric`); wires only connect compatible port types.

## 3. Column selection on consumption

Blocks declare what they need in terms of **roles**, not literal column
names (e.g. "one target, N≥1 features, optional weight"). The UI
auto-binds by role when unambiguous, or presents a picker pre-filtered by
role/dtype compatibility. Column selection happens in the **target
block's config panel**, not on the wire — the wire always carries the
whole packet.

## 4. Canvas & wiring

- **Swimlanes = phase label only.** Purely visual grouping (e.g. Data
  Prep → Feature Engineering → Estimation → Validation → Reporting). No
  effect on wiring or execution order — a block can freely consume from a
  block in a "later" lane. Lane-level collapse and multi-block drag
  between lanes supported, since lanes are relabelling, not rewiring.
- **Wire validity is checked on connection**, independent of block status:
  if a target port's role requirement isn't met by the incoming packet's
  metadata, the wire itself renders invalid (e.g. red/dashed) immediately.
- **Adding a block**: pick from the standard library (browsable by
  category), or "describe it" — a prompt box where an LLM drafts the
  function body + metadata transform + ports/params, dropped onto canvas
  in `grey` state for review before first run.
- **Editing a block's code** bumps its `code_version` → block itself goes
  `grey` (not `orange` — its own definition changed), cascades `orange`
  downstream as usual.
- **Removing a wire**: downstream block loses a required input → drops to
  `grey` with the missing-input reason shown (distinct from `red`, which
  means it was attempted and failed).
- **Node inspector panel** (on block select): config form driven off the
  block's declared schema, code (read-only for standard, editable for
  custom/LLM-authored), current status + timestamps/error, and for
  `green`/`orange` blocks a preview (head of output data, metadata table,
  on-demand summary stats).

## 5. Block status — state machine

Four states: `grey` (never run) → `green` (ran, output valid) → `orange`
(ran previously, now stale) → `red` (last run failed).

- Current cache key matches last **successful** run → `green`.
- Key differs but a prior successful run/output exists → `orange` (stale
  output stays visible, flagged as not current).
- Last run attempt raised an exception → `red`; previous good cached
  output (if any) stays available but marked not-current.
- Never run → `grey`.

**Propagation:** when a block's effective key changes (code/param edit, or
an upstream output changes), every currently-`green` downstream block that
depends on it flips to `orange`. Red/grey blocks are untouched.

**Execution actions:**
- **Run** (single block) — requires its inputs to currently be `green`.
- **Run to here** — cascades: runs whatever upstream chain isn't `green`,
  then this block.
- **Run all** — topological order, runs any non-`green` block (skips
  already-`green` ones); reports per-block status at the end. **Input
  blocks are excluded from Run all entirely** (see §6). If a required
  input block is still `grey`, Run all halts/flags rather than treating it
  as empty.
- **Force run all** — as Run all, but ignores cache entirely.

**AI-assisted fix on `red`:** a "Suggest fix" action sends the LLM the
block's code, params, the actual error/traceback, and the input packet's
schema (not necessarily full data). Returns a proposed diff — never
auto-applied. Accepting it edits the code (→ `grey`, version bump); user
re-runs manually to confirm it goes `green`.

## 6. Input blocks — explicit-only lifecycle

Input blocks never execute as part of Run all — treated as fixed until the
user deliberately acts.

- **Two independent timestamps:**
  - *Last successful read* — updates only on a successful Refresh; tied to
    the currently-cached (`green`) packet; shown by default.
  - *Last attempt* — updates on every Refresh attempt; only surfaced when
    more recent than the last successful read (i.e. there's a pending
    failure to flag). Normal display: "last successful read: t". Failure
    display: "last successful read: t1 · last attempt failed: t2" — makes
    clear the visible data is still from t1.
- **Check for changes** — cheap, read-only probe (mtime/checksum/row-count/
  schema-hash). Does not touch the cached packet or cascade anything;
  just annotates the block with a "source changed" badge. Global "Check
  all sources" sweeps every input block in one pass.
- **Refresh** — actually re-reads and swaps the packet, cascades `orange`
  downstream, updates the last-successful-read timestamp. A failed Refresh
  flips the block `red` but keeps the last-known-good packet and
  timestamp intact (never discards working data on a failed re-read).
- **Global "Refresh sources"** — runs Refresh on every input block; left
  as a deliberately separate step from Run all, so re-ingesting data is
  never accidental fallout from wanting to re-run downstream logic.

## 7. Compiling the graph to a single Python file

- **Scope:** only blocks reachable from the requested output(s) (or the
  whole graph if none specified) are emitted — no dead code.
- **Ordering:** topological sort of the DAG (lanes play no role); ties
  broken by canvas position / creation order for stable, diffable output.
- **Declare-then-call structure:** all function definitions first, then a
  linear call sequence below. Block functions stay **generic** — no
  literal parameter values baked into the body; all configuration (`cols`,
  `method`, etc.) comes in as explicit arguments. Actual configured values
  live only at the call site. Multiple output ports map to plain
  multi-return / tuple unpacking at the call site.
- **Fully inlined by default** (self-contained script, no dependency on
  the tool's runtime package being installed) — an optional "lean"
  import-based mode can exist alongside it.
- **`--with-metadata` toggle:** default/clean mode strips
  `DataFramePacket`/`ColumnMeta` scaffolding, emits plain functions over
  `pl.DataFrame` ("production" script); verbose mode keeps the packet
  wrapper so the compiled script still carries lineage/roles.
- **Provenance header:** comment block with graph name, compile timestamp,
  and each input block's "last successful read" timestamp captured at
  compile time.
- **`OUTPUT_DIR` constant** near the top of the file; every output block
  writes into it, with filenames tied to block id/display name so a file
  traces back to the exact block (and via its header comment, its phase)
  that produced it. Makes the compiled script relocatable.
- **Validation before compile:** refuse (or compile with inline warning
  comments) if any block in the required path is `grey`/`red`, or a
  required wire is unbound.

### Parseable comment markers

Comments carry only what plain Python can't recover: stable block id,
type/category, phase label, display name. Fixed minimal pattern above
each function def and each call:

```python
# === Block: b_042 | type=transform | phase="Feature Engineering" | name="Scale features" ===
def scale_features(df: pl.DataFrame, cols: list[str], method: str = "standard") -> pl.DataFrame:
    ...


# --- Call: b_042 ---
train_scaled = scale_features(df=train, cols=["income", "age"], method="standard")
```

Because call-site keyword arguments mirror declared port/param names, a
single parse pass (regex/AST) can reconstruct the full graph: match each
`# === Block ===` header to its `def`, each `# --- Call ---` to its
assignment, and derive edges from output-variable-name → later
call-argument matches. The compiled `.py` is therefore itself a valid
one-way serialization of the graph — a candidate import path later (open a
compiled script, rebuild the canvas), not just an export format.

---

## 8. Project storage & versioning

- **Live project format is JSON.** The canvas — blocks (code, params,
  metadata transform, position), wires, lanes, per-block status/cache
  pointers, in-review LLM-drafted code not yet accepted — is the source of
  truth, stored as JSON. This is what's opened, edited, and saved as the
  user works.
- **The compiled `.py` (§7) is export-only**, generated on demand from the
  JSON project. It is not read back in as the live format — the "parse the
  compiled script back into a graph" idea from §7 is a possible future
  import path for scripts that originated *outside* the tool, not the
  round-trip mechanism for the tool's own projects.
- **Versioning is git**, directly on the JSON project file(s) — no
  bespoke version-history feature inside the tool. "What changed between
  model version 3 and 4" is answered with a normal `git diff` between
  commits/branches/tags.
- **Diffing is git-branch-based**: comparing two states of a model means
  comparing two branches (or commits) of the project JSON, using ordinary
  git tooling. This implies the JSON should be structured to diff
  cleanly — stable block ids, deterministic key ordering, one
  block/wire per diffable unit rather than a single monolithic blob —
  since a JSON format that's hard to read in a diff defeats the point.

### 8.1 Project JSON — draft schema

`blocks` and `lanes` are keyed **objects**, not arrays — inserting a block
elsewhere in creation order then produces a one-line diff, not a reordered
array. Keys written in sorted order, 2-space indent, trailing newline
(same convention as a lockfile) so diffs are minimal and stable across
tools/OSes.

```json
{
  "modelmaker_version": 1,
  "project_name": "consumer_pd_model",
  "lanes": {
    "lane_data_prep":   { "name": "Data Prep",           "order": 0 },
    "lane_feature_eng": { "name": "Feature Engineering", "order": 1 },
    "lane_estimation":  { "name": "Estimation",           "order": 2 },
    "lane_validation":  { "name": "Validation",           "order": 3 },
    "lane_reporting":   { "name": "Reporting",            "order": 4 }
  },
  "blocks": {
    "b_001": {
      "block_type": "input",
      "category": "read_csv",
      "name": "Load applications",
      "lane": "lane_data_prep",
      "position": { "x": 40, "y": 120 },
      "code_version": 1,
      "params": { "path": "data/applications.csv" },
      "code_ref": null,
      "metadata_transform": { "kind": "infer_dtypes" },
      "ports": {
        "inputs": [],
        "outputs": [{ "name": "out", "type": "dataframe" }]
      }
    },
    "b_002": {
      "block_type": "standard",
      "category": "filter",
      "name": "Drop test accounts",
      "lane": "lane_data_prep",
      "position": { "x": 260, "y": 120 },
      "code_version": 1,
      "params": { "expr": "account_type != 'test'" },
      "code_ref": null,
      "metadata_transform": { "kind": "passthrough" },
      "ports": {
        "inputs": [{ "name": "in", "type": "dataframe" }],
        "outputs": [{ "name": "out", "type": "dataframe" }]
      }
    },
    "b_003": {
      "block_type": "llm_authored",
      "category": "custom",
      "name": "Bucket income",
      "lane": "lane_feature_eng",
      "position": { "x": 480, "y": 200 },
      "code_version": 2,
      "params": { "bins": [0, 30000, 60000, 100000, null] },
      "code_ref": "blocks/b_003.py",
      "metadata_transform": {
        "kind": "declared",
        "adds": [{ "name": "income_bucket", "role": "feature", "dtype": "categorical" }]
      },
      "ports": {
        "inputs": [{ "name": "in", "type": "dataframe" }],
        "outputs": [{ "name": "out", "type": "dataframe" }]
      }
    }
  },
  "wires": {
    "w_001": { "from": { "block": "b_001", "port": "out" }, "to": { "block": "b_002", "port": "in" } },
    "w_002": { "from": { "block": "b_002", "port": "out" }, "to": { "block": "b_003", "port": "in" } }
  }
}
```

Design decisions this schema bakes in:

- **Custom/LLM-authored block code lives in a sidecar `.py` file**
  (`code_ref`, e.g. `blocks/b_003.py`), not inlined as a JSON string.
  Code embedded as an escaped `"...\n..."` string diffs as one opaque
  line-change per edit; a sidecar file gets normal Python line-level
  diffs and syntax highlighting in any git tool. Standard-library blocks
  have `code_ref: null` since their body lives in the tool itself, not
  the project.
- **No run-state in this file.** Status (`grey`/`green`/`orange`/`red`),
  cache keys, timestamps, last-successful-read — all ephemeral,
  regenerated by re-deriving cache keys against a local cache store on
  load. None of it is committed; a `.modelmaker-cache/` directory
  (gitignored) holds cached packets keyed by `(block_id, inputs_hash,
  params_hash, code_version)`, separate from the versioned graph
  definition. This keeps every git diff meaningful (an actual graph
  edit) instead of noisy (a re-run touched a timestamp).
- **`params` may contain literal values only** (numbers, strings, bools,
  null, lists/objects of those) — never DataFrames or model objects — so
  every field in the versioned JSON is inherently diffable text.
