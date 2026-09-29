# AI model builder — build proposal

Proposal for letting an LLM build a whole model graph, not just one block
at a time. Today every AI action in Model-Maker is scoped to a single
block and is a one-shot call: `draft`, `suggest_fix`, `analyze_data` and
`suggest_names` each build one `DraftContext`, call `LLMProvider.draft()`
once and get back one structured `DraftResult` (see `llm/base.py`,
`api.py`). This proposal adds an **agent loop**. The model gets tools that
wrap the existing session operations: place blocks, wire them, set
params, run, read summary stats. It works step by step: observe, act,
run, inspect, correct.

Decisions already taken (from the design discussion):

- **Agent loop**, not a one-shot "emit the whole graph as JSON" plan. The
  right graph for a credit model depends on the data: which column is the
  target, what leaks, how features bin, whether a fit converges. Only a
  loop that runs blocks and reads their output can react to that.
- **The user prepares the inputs.** The agent starts from input blocks
  the user has already added, run and role-tagged. It never picks files,
  reads paths or touches credentials.
- **Plan first.** The agent proposes a plan and waits for approval before
  changing the graph.
- **Extending is allowed.** The agent can build onto an existing graph,
  not only into an empty one downstream of the inputs. The ownership rules
  in §6 are what make that safe.

The open decisions from the first draft are resolved in §11 and carried
through the sections below.

## Implementation status

**Phases 0–3 are built, plus the `claude_cli` part of phase 4.** That
provider was moved forward because it's the default and needs no API key.
The code is in `modelmaker/agent/` and the frontend's `BuildPanel.tsx` /
`GhostNode.tsx`, with tests in `tests/test_agent_foundations.py`,
`tests/test_agent_build.py` (a scripted loop against the real runner and
PD sample data) and `tests/test_agent_api.py` (including a real stdio MCP
client driving the bridge against a live server).

It has been verified end to end with the real `claude` CLI:
- **Plan:** it planned a PD model from a prepared `read_csv` block (70/30
  split → logistic regression on six numeric drivers → predict → AUC/Gini).
  It left out the excluded, id and date columns by itself.
- **Build:** after approval it built all five blocks green, first time,
  with test-set Gini 0.674.
- **Cost:** about $0.29 for the whole build, planning included.

Where the implementation differs from the text below:
- **Custom blocks** are written by the build LLM itself
  (`add_custom_block` / `update_custom_block`), checked against the same
  contract (one polars function, input ports as leading parameters, a valid
  `metadata_transform`, no excluded-column literals). They are not
  delegated to a second LLM call through the `draft` / `suggest_fix` flow.
  The agent already has the context, and a nested call per block would
  double the cost and latency.
- **Sampling (§7):** switching sample mode makes input blocks stale, and
  downstream runs then refuse to run. A sampled build therefore re-reads the
  in-scope input blocks with `run_block` (not `refresh`, which bumps the
  read counter and would orphan the full-data cache). This happens once on
  approval and again when sample mode is switched back off; the second read
  hits the cache. Sampling defaults to 50,000 rows, and only when the
  largest anchor output has more than 100,000 rows. If the user already has
  sample mode on, the build uses it and leaves the setting alone.
- **Preflight (§4.1):** no `target` at all is a warning, not a blocker.
  The app can't tell from the goal whether a target is needed. More than
  one distinct target is still blocking.
- **Discard** is available only while a build is live. Once it has ended,
  a plain Undo reverts it in one step.
- **Canvas lock (§7):** this is an HTTP middleware. While a build is
  `building`, `awaiting_input` or `final_run`, it returns 409 for mutating
  `/api/*` calls other than `/api/agent/*`, LLM settings, env vars, project
  save and source checks.
- **The MCP bridge** is a stateless stdio server
  (`python -m modelmaker.agent.mcp_server`) that forwards to
  `/api/agent/mcp/{tools,call}`, authenticated with a per-build token.
  Guards and logging therefore stay in the API process. It calls back to the
  URL the build was started from. `MODELMAKER_AGENT_CALLBACK_URL` overrides
  that.

**`openai` and `gemini` loops** are built too, over plain HTTPS (standard
library only, like their draft providers), with the same keys, models and
base URLs as Settings:
- **OpenAI:** `OpenAILoop` uses `/chat/completions` with `tools`. It sends
  no `temperature` and no token cap, because current reasoning models reject
  non-default values for both.
- **Gemini:** `GeminiLoop` uses `generateContent` with
  `functionDeclarations`. It passes schemas as `parametersJsonSchema`,
  because the OpenAPI-subset `parameters` rejects `additionalProperties` and
  property-less objects. It echoes each model turn back unchanged so
  thinking models' thought signatures survive, and it sends the key in the
  `x-goog-api-key` header rather than the URL.
- **Start-time check:** `POST /api/agent/builds` constructs both loops once
  up front, so a missing key or model is a 400 immediately rather than a
  failed build later.
- **Tests:** `tests/test_agent_loops.py` drives both loops against a local
  fake endpoint with scripted vendor responses. It checks our side of the
  wire format, but not that the real services accept it.

**`lmstudio`** is built as `LMStudioLoop`, an `OpenAILoop` in compact mode
for local OpenAI-compatible servers (LM Studio, llama.cpp `llama-server
--jinja`, Ollama, vLLM):
- **Setup:** no API key, and the model is auto-detected.
- **Compact catalogue:** the system prompt lists block *names per category*
  (about 170 tokens, against about 5,150 for the full catalogue), and the
  model fetches ports and summaries with `list_block_types(group=…)` and
  params with `describe_block_type` as needed.
- **Context:** tool results are capped at 6,000 chars. When the
  conversation passes about 85% of the context window, the oldest tool
  results are replaced by a stub. The window is read from the server
  (llama.cpp's `/v1/models` reports `n_ctx`), else
  `MODELMAKER_LLM_CONTEXT_TOKENS`, else 16k.
- **Reasoning traces:** `reasoning_content` is never echoed back, for any
  OpenAI-compatible server.

It has been verified live against a llama.cpp server (Ternary-Bonsai-2-27B
at 2-bit, about 10 tokens/s, 100k context):
- **Plan:** 10 minutes, using the per-category lookups as intended.
- **Build:** 9 minutes, five blocks all green on the first run, test Gini
  0.594, no deviations.
- **Size and cost:** 59k input and 9k output tokens, $0.

Provenance records the auto-detected model rather than "default".

That run exposed a **race** that fast hosted models had hidden. After
`submit_plan` the build showed "plan ready" while the model was still
writing a closing remark, and an Approve in that window returned 409, but
only *after* flipping the phase and opening the undo transaction. The build
was then stuck in "building" with nothing running. The fixes:
- **Terminal tools end the turn.** A successful `submit_plan`, `ask_user`
  or `finish` sets `build.turn_over`, and the in-process loops stop there
  with no extra request.
- **Resume after a terminal tool** merges into the trailing user turn
  (Anthropic, Gemini).
- **`approve` and `feedback` check `_require_idle()`** before touching any
  state.
- **The panel** disables its buttons while the AI is busy.

Not yet done:
- **Live tests** of the Anthropic, OpenAI and Gemini loops against the
  real APIs, which need keys. `claude_cli` and `lmstudio` have run real
  builds.
- **Phase 5 follow-ons:** `iterate`/`collect` in the catalogue, "improve
  this model" builds, and the build id in compiled-script block markers.

Written against the codebase as of `0f0499e`.

## 1. Design principles

1. **No new engine paths.** Every agent action goes through the same
   `ProjectSession` methods the canvas uses (`add_block`, `add_wire`,
   `update_block`, `set_column_role`, `runner.run_to_here`, …). Agent-built
   graphs therefore get validation, caching, staleness, compile and git
   exactly as hand-built ones do. The agent is just another client of the
   session, like the React canvas and the TUI.
2. **Schema and statistics, never rows.** This is the rule every existing
   LLM feature already follows (see the `DraftContext` docstring). Tool
   results carry column names, dtypes, roles, summary stats, scalar
   metrics and model summaries. They never carry `packet.data` rows. §5
   covers where this is and isn't airtight.
3. **The user's blocks are the user's.** The agent may read and wire from
   any block. It may only modify or delete blocks it created in the
   current build, plus the specific changes to existing blocks that the
   user approved in the plan (§6).
4. **One build, one undo.** A whole build, however many steps, is a
   single undo point.
5. **Bounded and stoppable.** Hard caps on steps, custom blocks and
   repeated failures. A Stop button that works mid-run. When stuck, the
   agent stops and asks instead of flailing.
6. **Provider-agnostic loop, Anthropic first.** The tool layer is plain
   Python with JSON-schema tool definitions. The loop that drives it sits
   behind a small interface, so each provider plugs in its own native
   tool-calling (§9).

## 2. User workflow

1. **Prepare data.** Add input blocks (`read_csv`, `read_parquet`,
   `read_sql`, …), run them, and tag roles in the Port Inspector: at
   minimum `target`, plus `excluded` for anything that must not become a
   feature (post-outcome fields, leaky columns). Optionally tag `id`,
   `date`, `weight` and `segment`.
2. **Start a build.** Select the blocks the agent should build from. This
   is usually the prepared inputs, but for an extension it can be any
   blocks, e.g. a model's scored output for "add a validation lane".
   Open the **Build** panel and write a goal, e.g.:
   > PD model, logistic regression on WoE-transformed features,
   > out-of-time validation on 2023 vintages, master scale with 10 grades.
   The Build panel also shows the **plan LLM** and **build LLM** (§9.1).
   Both default to the Settings panel's configuration and can be changed
   per build.
3. **Preflight.** The app checks the anchors and everything upstream of
   them (§4.1). It reports anything that blocks the build, such as "no
   column tagged `target`", and warns about anything out of date, before
   any LLM call.
4. **Plan.** The agent explores with read-only tools and returns a
   structured plan (§4.2): lanes, blocks, wiring, key params, any changes
   to existing blocks, its assumptions, and open questions. The plan
   appears as ghost blocks on the canvas and as a list in the Build
   panel. The real graph does not change.
5. **Review.** The user approves, or sends feedback ("use a 70/30 split,
   not out-of-time", "drop the correlation step") and the agent re-plans.
   There is no limit on rounds.
6. **Build.** With the plan approved, the agent builds in sample mode,
   lane by lane. It runs each new block, checks its output and fixes what
   fails. Progress streams into the Build panel as an event feed while
   blocks appear on the canvas. The canvas is read-only for the user
   during the build, apart from Stop.
7. **Finish.** A full-data `run_all` runs by default (§7). The agent then
   reports what it built, any deviations from the plan, the key results
   (e.g. Gini on train and test, computed on the full data) and open
   concerns.
   The user can keep the result, undo it in one step, or ask for a
   follow-up build.

## 3. Architecture

```
modelmaker/agent/
  __init__.py
  build.py       AgentBuild: state machine, ownership set, event log, limits
  tools.py       tool definitions (JSON schema) + implementations over SESSION
  catalogue.py   block catalogue for the model: categories, ports, params, docs
  layout.py      lane assignment -> x/y placement
  prompts.py     system prompts for the plan and build phases
  loop.py        AgentLoop interface + provider implementations
  mcp_server.py  (phase 4) the same tools exposed over MCP
```

**`AgentBuild`** is the one object that holds a build's state:

- `id`, `goal`, and `anchors`: the selected block ids.
- `phase`: `preflight → planning → awaiting_approval → building
  (⇄ awaiting_input) → final_run → done | done_with_errors | stopped |
  failed`.
- `plan`: the latest structured plan, plus a history of feedback rounds.
- `owned`: the set of block and wire ids created in this build. This is
  what the ownership guard checks.
- `approved_changes`: the edits to pre-existing blocks that the approved
  plan lists (§6).
- `events`: an append-only log of tool calls, results (truncated),
  assistant messages and status changes. The UI feed and the final report
  are both rendered from it.
- `counters`: steps, custom blocks, consecutive failures per block, and
  tokens.
- `snapshot_before`: the graph snapshot taken when building starts
  (§7).

There is one active build per session, the same as the existing
one-run-at-a-time lock. The build runs on a background thread, and the
frontend polls it just as it polls `/api/graph` during a sweep.

**API** (new, under `/api/agent`):

| Endpoint | Purpose |
|---|---|
| `POST /api/agent/builds` | start: `{goal, anchors, plan_llm?, build_llm?, final_full_run?}` → preflight + planning; each `*_llm` is `{provider, model}` (§9.1) |
| `POST /api/agent/builds/current/proceed` | continue past preflight warnings (§4.1) |
| `GET /api/agent/builds/current` | phase, plan, events since cursor, counters |
| `POST /api/agent/builds/current/feedback` | `{text}` → re-plan |
| `POST /api/agent/builds/current/approve` | start the build phase |
| `POST /api/agent/builds/current/stop` | cooperative stop (between tool calls) |
| `POST /api/agent/builds/current/discard` | stop + restore `snapshot_before` |

## 4. Phases

### 4.1 Preflight (no LLM)

Preflight covers the anchors **and their whole upstream**
(`Graph.ancestors(anchor)` for each anchor). This matters for
extensions, where an anchor can sit deep in an existing graph. It runs
before any tokens are spent.

**Blocking** (the build can't start):

- An anchor has no successful output yet (never run, or `red` with no
  earlier success), so there is no schema or stats for the agent to read.
  The panel offers a **Run upstream** button (`run_to_here` on each
  anchor). The agent never runs the user's input blocks itself, because
  those are the blocks that touch the outside world.
- The goal needs a target, but no column across the anchors' dataframe
  outputs is tagged `target`, or more than one is.
  `find_duplicate_unique_role` already detects duplicate unique roles.

**Warnings** (the user can Run upstream, or proceed anyway):

- **Out of date:** any anchor or upstream block that is `orange` (stale:
  the panel shows `Runner.stale_reason`), `grey`, or `red` (the panel
  shows `last_error`). The agent would be planning against outputs that
  no longer match the graph, so these are listed by block name with the
  reason. **Run upstream** fixes them in one click. **Proceed anyway**
  records the warning in the build's event log and final report.
- No `excluded` columns tagged, no `date` column when the goal says
  "out-of-time", or very high-cardinality string columns with role
  `feature`.

With warnings present, the build waits in phase `preflight` until the
user picks one of the two options.

### 4.2 Plan (read-only tools)

The planner only gets the read-only tools (§5). The final tool call is
`submit_plan`, whose argument is the plan itself. Using a tool for it
rather than free text means the plan is schema-validated, and a malformed
one comes back to the model as a tool error it can fix.

```jsonc
{
  "summary": "Logistic PD model with WoE features, OOT validation on 2023.",
  "assumptions": ["default_flag is the 12-month default indicator", "..."],
  "questions": [],                       // non-empty => UI asks before approving
  "lanes": [{"key": "prep", "name": "Data prep", "purpose": "..."}],
  "steps": [
    {
      "ref": "s1",                        // plan-local id, used by later steps
      "category": "train_test_split",     // registry category, or "custom"
      "instruction": null,                // for custom: what the block should do
      "lane": "prep",
      "name": "oot split",
      "inputs": [{"port": "df", "from": "blk_8f2", "from_port": "out"}],  // existing id or a ref
      "params": {"date_col": "obs_date", "cutoff": "2023-01-01"},
      "why": "Goal asks for out-of-time validation on 2023."
    }
  ],
  "changes_to_existing": [
    {"block": "blk_8f2", "change": "set group_by=segment", "why": "..."}
  ]
}
```

**Ghost preview.** The plan is shown two ways:

- **In the Build panel,** as a list grouped by lane.
- **On the canvas, as ghost blocks.** These are translucent, dashed
  nodes with dashed wires, placed by the same `layout.py` pass the build
  phase will use (§8). Planned lanes that don't exist yet show as ghost
  lane bands.

In detail:

- **Where ghosts live.** Ghost blocks are frontend-only. They are
  rendered from `plan.steps`, never added to `SESSION.graph`, so they
  can't be run, saved, compiled or undone.
- **Linking list and canvas.** Hovering a ghost highlights its plan step,
  and clicking one scrolls the list to it.
- **Changes to existing blocks.** Each entry in `changes_to_existing` is
  shown as a badge on the real block it would modify.
- **Placement.** `POST /api/agent/builds` responses and
  `GET .../current` return the plan with each step's computed position
  and lane, so the frontend doesn't reimplement layout.
- **Re-planning** replaces the ghosts. **Approve** hides them, and the
  real blocks appear as they're built. The build phase uses the same
  layout, so real blocks land roughly where their ghosts were.

If `questions` is non-empty, Approve is disabled until the user answers
through the feedback box.

### 4.3 Build (read + write tools)

The build phase starts a fresh conversation. It is seeded with the goal,
the approved plan, the anchor schemas and the block catalogue, not the
planning transcript. That keeps context small, and it makes the approved
plan (not the exploration) the contract.

The expected rhythm per step:

1. `add_block` with its params.
2. `connect` its inputs.
3. `run_to` it.
4. `get_output_summary`, then check the result: expected columns
   present, row counts plausible, the metric in a sane range.
5. On failure, fix it (`set_params`, or `fix_custom_block`, which reuses
   the existing `suggest_fix` draft flow) and retry.

**Deviations.** Small deviations are allowed and must be recorded with
`note_deviation(step_ref, what, why)`. Examples: a param value changed to
make a fit converge, or an extra custom cleaning block inserted to fix a
dtype. Structural deviations (dropping a planned lane, switching model
family) and any change to an existing block that the plan didn't list
are not. For those the agent calls `ask_user(question)`. This pauses the
build (phase `awaiting_input`) until the user answers or stops it.

**Finishing.** The agent calls `finish(report)`. The report is Markdown:
what was built, deviations, key metrics and concerns. It is stored as an
`Artifact` (the class `analyze_data` already uses for write-ups) attached
to the build's last output block, so the model's development notes stay
in the project.

## 5. Tools

All tools live in `agent/tools.py` as functions of
`(build: AgentBuild, **args) -> dict`. Each has a JSON schema generated
from its signature. Every tool validates, applies guards, calls the
session, and returns a compact JSON result or a structured error that the
model can act on. The same functions back the in-app loop and the MCP
server.

**Read-only (plan + build):**

| Tool | Returns |
|---|---|
| `list_block_types(group?)` | catalogue entries: category, display name, group, ports + types, one-line description |
| `describe_block_type(category)` | full docstring, parameters with types/defaults, role-bound params (e.g. `target_col` auto-fills from the `target` role) |
| `get_graph()` | blocks (id, category, name, lane, status, params, ports, `owned` flag), wires, lanes |
| `get_output_summary(block, port?)` | by port type: dataframe → columns, dtypes, roles, tags, row count, summary stats (**`rows=0`**); scalar_metric → the value; model → coefficients/fit stats; image → "image, N bytes" only; others → the JSON-shaped dict, truncated |
| `get_block_error(block)` | `last_error`, and the code for custom blocks |

**Write (build only):**

| Tool | Notes |
|---|---|
| `add_lane(name)` | reuses an existing lane with the same name |
| `add_block(category, lane, name?, params?)` | registry blocks only; placed by `layout.py` (§8) |
| `author_custom_block(lane, name, instruction, inputs)` | creates a custom block and drafts it via the existing `draft` flow (`DraftContext`, `mode="author"`) |
| `fix_custom_block(block, instruction?)` | wraps `suggest_fix` |
| `connect(from_block, from_port, to_block, to_port)` | the target must be owned (or an approved change); the wire must be `valid` |
| `disconnect(wire)` / `delete_block(block)` | owned only |
| `set_params(block, params)` | owned, or an approved change |
| `set_column_role(block, column, role)` | owned only; the user's role tags are read-only |
| `run_to(block)` | `runner.run_to_here`, in sample mode (§7); returns the final status + error |
| `note_deviation`, `ask_user`, `finish` | control tools (§4.3) |

**Guards, enforced in the tool layer and not only in the prompt:**

- Ownership (§6).
- A column tagged `excluded` on an anchor may not be named in any param
  that selects features. The guard checks param values against the
  excluded set for any param whose name ends in `_col`/`_cols` or is
  `features`. This is best-effort, because custom block code can still
  read any column. The final report lists every column that reached a
  model block's feature set, so the user can check it.
- No tool takes a file path, a URL, a SQL string or an env var. Input
  and output blocks that touch the outside world (`read_*`,
  `write_csv`) aren't offered in `add_block` at all. The agent can add
  display blocks (`display_table`, `display_value`) for the user's
  benefit.

**On "never rows":** `get_output_summary` never returns `packet.data`,
and `_packet_preview` is called with `rows=0`. That doesn't make the
channel leak-proof, though, and the proposal shouldn't pretend it does:

- `min`/`max` of string and id columns are real values.
- Custom code written by the agent could filter to one row, and that
  row's stats would then be its values.

Mitigation:

- `min`/`max` are suppressed for columns whose role is `id`, and for
  string columns (their `n_unique` is still reported).
- Every custom block the agent writes is logged in full in the event
  feed.

That's the same trust level as the current per-block features. It is not
a sandbox.

**The suppression applies to every LLM path, not only the agent.** Today
the only existing feature that sends stats is `analyze_data`: `api.py`
builds `ColumnInfo(min=…, max=…)` from the summary, and
`prompts.format_columns` prints them. `draft` and `suggest_fix` only
send names, dtypes and roles. A single helper, `llm/redact.py:
column_info_for_llm(name, meta, stats)`, becomes the only way
`ColumnInfo` gets built for a prompt. The agent's `get_output_summary`
uses it too. The helper nulls `min`/`max` for `id`-role and string
columns (and date columns with role `id`). The UI's own previews
(`_packet_preview` for the Port Inspector and DataModal) are unchanged,
because the user looking at their own data isn't the concern.

## 6. Ownership and extending existing graphs

Extending is what makes the builder useful after the first build ("add a
PSI monitoring lane", "try LightGBM alongside the logistic model"). It is
also where an agent could do real damage. The rules:

- **Readable:** every block in the graph.
- **Wirable from:** every block. Wiring *from* a user block never changes
  that block.
- **Modifiable/deletable:** blocks and wires in `build.owned`, i.e. those
  created in *this* build.
- **Approved changes:** each entry in the plan's `changes_to_existing`
  that the user approved authorizes exactly one described change to one
  pre-existing block. The tool layer checks the block id, and the event
  log records the change against the plan entry. Wiring *into* an
  existing block's input port also counts as a change to that block and
  must be listed. That's usually a replacement ("feed the new
  feature-selection output into the existing model block"), which changes
  the block's result.

Ownership is per build. When a build finishes, its blocks become ordinary
blocks, and a later build treats them as pre-existing. That keeps the
rule simple and matches how people think about it: "the model the agent
built last week" is now just part of the model.

**Provenance (decided: persisted).** Ownership for guard purposes stays
per build and in memory, as described above. Separately, every block the
agent creates gets a descriptive, persisted provenance record, so model
governance can always tell what was AI-built. The review document
(`model-developer-review.md`) repeatedly flags audit trail as a standing
requirement.

```python
# graph.BlockInstance
provenance: dict[str, Any] | None = None
# e.g. {"source": "agent", "build_id": "b_3f1c", "at": "2026-09-29T14:02:11Z",
#       "plan_llm": "anthropic/claude-opus-5-5", "build_llm": "anthropic/claude-sonnet-5-5",
#       "plan_step": "s4", "modified_by_user": false}
```

- **Hashing.** Like `column_tags`, provenance stays out of
  `Runner.compute_key`'s hash basis, so it never invalidates a cache.
- **Persistence.** It is saved in `model.json` through the existing
  project serializer.
- **Old project files.** A missing key loads as `None`, so no migration
  is needed.
- **Changes to existing blocks.** An approved change to a pre-existing
  block appends an entry to that block's provenance history (`"changes":
  [{"build_id", "at", "change"}]`) instead of replacing it. A user block
  never gets labelled `source: agent` just because the agent edited it.
- **User edits afterwards.** `ProjectSession.update_block` sets
  `modified_by_user: true` when the user later edits an agent-created
  block's code or params. "The AI built it and a person changed it" is
  exactly what a reviewer wants to see.
- **Where it shows.** The canvas shows a small AI badge on agent-created
  blocks, and the Inspector shows the full record.
- **Compiled scripts** (phase 5). The compiler will add the build id to
  the block's marker comment. The compiled-script provenance header
  already exists.
- **Build report.** Each build's report Artifact is keyed by `build_id`,
  so provenance links back to the plan and the reasoning.

## 7. Undo, sample mode and the run lock

**One undo.** `ProjectSession.edit()` pushes one snapshot per mutation.
A build will make dozens of mutations. Add
`ProjectSession.transaction()`: it pushes one snapshot on entry and turns
the `edit()` calls inside it into no-ops for the undo stack (while still
bumping `revision`). The build phase runs inside one transaction, so a
single undo reverts the whole build. `discard` restores
`snapshot_before` directly. Snapshots already carry run state, so undoing
a build costs no re-runs of the user's blocks.

**Canvas lock.** While a build is `building`, mutating endpoints called
from outside the agent return `409 "an AI build is in progress"`. Read
endpoints keep working, so the user can still inspect outputs as they
appear. Without the lock, the transaction and ownership model would
have to cope with concurrent edits, and nothing needs that.

**Sample mode.** On approve, the build stores the current `sample_rows`
and switches to a build default (e.g. 50 000 rows) if sampling is off.
It restores the setting on finish. Sample runs cache under different
keys from full runs, so this doesn't pollute the user's cache.

**Final full run (decided: on by default).** After the agent calls
`finish`, the build restores the user's `sample_rows` and runs one
`run_all` on the full data. The panel has a checkbox to turn this off,
which maps to `final_full_run` on the start request. The metrics in the
report (§4.3) are then refreshed from the full-data outputs:

1. The agent names the metric-bearing blocks in `finish(report,
   key_outputs=[...])`.
2. The build re-reads those blocks after the full run and fills them into
   a results table in the report. The agent is not called again.

If the full run turns a block `red` that was green on the sample, the
build ends in phase `done_with_errors`. The report flags that block and
its error, and the build stays in one undo step. The user can start a
follow-up build ("fix the failing blocks") or fix it by hand.

**Run lock.** `_start_background_run` serializes runs behind
`_RUN_LOCK`/`_RUN_THREAD`. The agent's `run_to` must go through the same
lock, so it should be lifted out of `api.py` into a small shared helper.
While the lock is held the agent is the only runner, and the canvas lock
already stops user-initiated runs.

## 8. Block catalogue and layout

**Catalogue.** The model can only choose blocks well if it knows what
they do. There are 52 registry blocks today, and **23 have no docstring
on `fn`**:

- input/output: `read_csv`, `read_parquet`, `read_json`, `read_excel`,
  `read_sql`, `write_csv`, `generate_image`
- data prep: `filter`, `select`, `groupby_agg`, `join`,
  `train_test_split`
- modelling and tests: `glm_fit`, `logistic_regression`, `ks_test`,
  `auc_gini`
- stochastic: `fit_distribution`, `sample_distribution`,
  `build_dependency`, `fit_proxy`, `evaluate_proxy`, `validate_proxy`,
  `iterate`

`BlockSpec` has no description field either. The catalogue needs, per
block:

- a one-line description, plus the full docstring on demand;
- the param list with types and defaults, from `inspect.signature`;
- which params auto-bind from a role (`util.find_role_param`);
- port types.

Frontend `PARAM_SPECS` (in `ParamsForm.tsx`) has the richest param
metadata, such as enum options, but it lives in TypeScript. Short term,
docstrings are the source of truth, so filling the 23 gaps is a phase 0
task. Longer term, moving param specs to the Python side would serve the
catalogue, the TUI and the frontend at once. That's out of scope here.

`iterate`/`collect` (fan-out) are left out of the build-phase catalogue
in v1. Pairing them by `iterate_block` id is easy to get subtly wrong,
and neither compiles. They can be added once the builder is proven.

**Layout.** The agent chooses the *lane*, and `layout.py` chooses x/y:
x from topological depth within the build, y stacked within the lane
band, avoiding existing blocks' bounding boxes. LLMs are poor at picking
coordinates, and the user can always drag blocks afterwards. Lanes the
agent creates are appended after existing lanes.

## 9. Providers and the loop

```python
class AgentLoop(ABC):
    def run(self, build: AgentBuild, system: str, tools: list[ToolDef],
            seed_messages: list[dict]) -> None: ...
```

The loop owns the message history and calls tools through
`build.call_tool(name, args)`. That one method applies the guards,
logging, counters and the stop flag. The loop checks `build.stop_requested`
between tool calls, which gives a cooperative stop. A block run that is
already in flight finishes first, or is cancelled through the existing
`/api/run/cancel` path.

| Provider | Approach | Phase |
|---|---|---|
| `anthropic` | Messages API with `tools=`, manual loop (not the SDK tool runner, so stop/events/counters stay in our hands); prompt caching on the system prompt + catalogue | 2 |
| `claude_cli` (the default) | Expose `agent/tools.py` as an MCP server (`mcp_server.py`, stdio). Planning = `claude -p --mcp-config … --allowedTools <read-only tools>`; build = a second invocation with write tools allowed. The CLI runs the loop; our tool layer still enforces guards and logs events. Uses the user's existing Claude login, no API key needed | 4 |
| `openai`, `gemini` | native function calling in the same manual-loop shape as `anthropic` | 4 |
| `lmstudio` | OpenAI-compatible tool calling; best-effort, marked experimental in the UI | 4 |
| `stub` | scripted tool-call sequences, for tests | 1 |

A side benefit of the MCP server: Claude Code or Claude Desktop pointed
at a running Model-Maker can build models directly, using the same guards.

### 9.1 Separate plan and build LLMs (decided)

Planning is short but reasoning-heavy. Building is long, with many tool
calls. They are configured separately, and each can be any provider and
model:

```python
# llm/settings.LLMSettingsStore
agent_plan: dict[str, str] | None = None   # {"provider": ..., "model": ...}
agent_build: dict[str, str] | None = None
```

- **Defaults.** Unset means the active provider with its default model,
  the same fallback every other LLM feature uses.
- **Settings panel.** A new "AI builder" section with two provider/model
  pickers, reusing the existing model dropdowns and `/api/llm/models`.
  API keys and base URLs still come from each provider's own settings.
- **Per-build override.** The Build panel pre-fills both from Settings.
  A build can override them through `plan_llm`/`build_llm` on
  `POST /api/agent/builds`.
- **Mixing providers works** because the build phase starts a fresh
  conversation seeded with the approved plan (§4.3). No message history
  has to cross providers. Planning with `claude_cli` and building with
  `anthropic`, or the other way round, is fine.
- **Tool-calling check.** Both choices are checked against the §9 table.
  A provider/model without tool-calling support can't be selected, and
  `lmstudio` shows an "experimental" hint.
- **Recording.** Both are recorded on the build and in each created
  block's provenance (§6).

## 10. Limits

Defaults, adjustable in the Build panel:

| Limit | Default | On hit |
|---|---|---|
| Tool calls per phase | plan 30, build 80 | stop, report |
| Custom blocks per build | 5 | tool error: "use a registry block or ask_user" |
| Consecutive failed runs of one block | 3 | forced `ask_user` |
| Wall time | 20 min | stop, report |
| Tokens | shown live, no default cap | — |

Stopping never leaves a half-applied tool call. Everything built so far
stays on the canvas as one undo point, with the report explaining where
the build stopped.

## 11. Decisions

Resolved after review of the first draft:

| # | Question | Decision | Where |
|---|---|---|---|
| 1 | Persist provenance on AI-built blocks? | **Yes.** `BlockInstance.provenance`, persisted, outside the hash basis; approved changes to user blocks append to its history; later user edits set `modified_by_user` | §6 |
| 2 | Final full-data run? | **On by default**, switchable off per build; report metrics refreshed from full-data outputs | §7 |
| 3 | Plan preview? | **Ghost blocks** on the canvas (frontend-only) alongside the plan list | §4.2 |
| 4 | Hide `min`/`max` for string/id columns? | **Yes, for every LLM path**, including the existing `analyze_data`, via one `column_info_for_llm` helper; UI previews unchanged | §5 |
| 5 | Which model(s)? | **Separately configurable plan LLM and build LLM** (provider + model each), set in Settings and overridable per build | §9.1 |
| 6 | Preflight scope for extensions? | **Anchors and their whole upstream**; anything out of date (`orange`/`grey`/`red`) is a **warning** with Run upstream / Proceed anyway; an anchor with no output at all blocks | §4.1 |

Still open, to settle during implementation:

- The default build sample size (50 000 rows proposed in §7). This
  should come out of timing the phase 2 live test on the demo data.
- Whether prompt caching of the catalogue makes a cheaper build model
  unnecessary in practice. Measure it in phase 2 and use the result to
  set the defaults for §9.1.

## 12. Phased delivery

Each phase is usable and tested on its own.

**Phase 0: prerequisites** (no AI)
- Docstrings for the 23 undocumented registry blocks.
- `agent/catalogue.py`, with a test that every registry block produces a
  complete entry.
- `ProjectSession.transaction()`, plus tests that N edits inside it make
  1 undo and undo restores the run state.
- Lift the run lock into a shared helper.
- `llm/redact.column_info_for_llm`, switching `analyze_data` over to it,
  with tests that `id`/string `min`/`max` never reach a prompt (§5,
  decision 4).
- `BlockInstance.provenance`: field, save/load round-trip, excluded from
  `compute_key`, `modified_by_user` set on user edits (§6, decision 1).

**Phase 1: tool layer**
- `AgentBuild`, `tools.py` and `layout.py`.
- Preflight over anchors + upstream, with blocking checks and warnings
  (§4.1, decision 6).
- Tests driven directly (no LLM): ownership guard, approved-change guard,
  excluded-column guard, no-rows guarantee on `get_output_summary`, wire
  validity, limits, and provenance stamping.
- A `stub` loop that replays a scripted tool sequence, building the demo
  PD model from `projects/demo_pd_model`'s input block end to end,
  including the final full run (decision 2).

**Phase 2: Anthropic loop + API**
- `loop.py` (Anthropic) and `prompts.py` for plan and build.
- The `/api/agent` endpoints.
- `agent_plan`/`agent_build` in `LLMSettingsStore` and
  `/api/llm/settings` (decision 5).
- Plan responses carry laid-out ghost positions.
- A live-LLM test, skipped without a key, that plans and builds a PD
  model on `sample_data/pd_model_data.csv` and asserts a sane result:
  the model block is green and the test-set Gini is within a band.

**Phase 3: frontend**
- Build panel: goal box, anchor chips from the current selection,
  plan/build LLM pickers, preflight warnings with Run upstream / Proceed,
  plan list + feedback, Approve/Stop/Discard, event feed, final report.
- Ghost-block plan preview on the canvas (decision 3).
- AI provenance badge on blocks, and the provenance record in the
  Inspector.
- Settings panel "AI builder" section.
- Canvas lock indicator, and highlighting of blocks the agent is
  touching.

**Phase 4: more providers**
- MCP server + `claude_cli`, then OpenAI, Gemini and LM Studio.

**Phase 5: follow-ons**
- `iterate`/`collect` in the catalogue.
- "Improve this model" builds that iterate on an existing agent-built
  graph against a metric target.
- Provenance in the compiled script's block markers.
