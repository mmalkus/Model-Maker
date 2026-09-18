export type Status = 'grey' | 'green' | 'orange' | 'red' | 'running'
export type PortType = 'dataframe' | 'model' | 'scalar_metric' | 'image' | 'master_scale' | 'any'
export type BlockType = 'input' | 'standard' | 'output'

export interface PortSpec {
  name: string
  type: PortType
  required: boolean
}

export interface RegistryEntry {
  category: string
  block_type: BlockType
  group: string
  display_name: string
  inputs: PortSpec[]
  outputs: PortSpec[]
}

export interface BlockOut {
  id: string
  block_type: BlockType
  is_custom: boolean
  category: string
  name: string
  lane: string | null
  position: { x: number; y: number }
  code_version: number
  params: Record<string, unknown>
  code: string | null
  source: string | null
  metadata_transform: Record<string, unknown> | null
  inputs: PortSpec[]
  outputs: PortSpec[]
  port_names: Record<string, string>
  // Column to run this block once per distinct value of (see
  // BlockInstance.group_by in graph.py) -- null means plain, ungrouped
  // execution. max_workers caps concurrent group workers; null means "use
  // the runner's default".
  group_by: string | null
  max_workers: number | null
  status: Status
  last_error: string | null
  last_successful_read_at: string | null
  last_attempt_at: string | null
  // Why an orange block's cached output is no longer current, in one phrase
  // (see Runner.stale_reason) -- null for every other status, since grey has
  // never run and red carries its error instead.
  stale_reason: string | null
  // Seconds since this block's (or its whole fused streaming group's --
  // see Runner.running_elapsed) current dispatch started; null unless
  // status is "running". A crude heartbeat, not real progress -- polars
  // gives no finer-grained signal for one collect() call.
  running_seconds: number | null
}

export interface WireOut {
  from_block: string
  from_port: string
  to_block: string
  to_port: string
  valid: boolean
}

export interface LaneOut {
  name: string
  order: number
  height: number
}

export interface RecoveryInfo {
  saved_at: string | null
  project_name: string | null
  project_path: string | null
  block_count: number
}

export interface GraphOut {
  project_name: string
  project_path: string | null
  // Whether there are edits the project file doesn't have yet. Edits are
  // always snapshotted for crash recovery, and autosave catches the project
  // file itself up a couple of seconds later once it has somewhere to save
  // to (see App.tsx) -- this is true in the gap before that happens.
  dirty: boolean
  can_undo: boolean
  can_redo: boolean
  // Row cap applied to every input block, or null when off (see
  // Runner.sample_rows).
  sample_rows: number | null
  lanes: Record<string, LaneOut>
  blocks: Record<string, BlockOut>
  // A precondition failure from the most recently *started* run (e.g. "Run"
  // clicked on a block whose upstream isn't green) that had nothing else to
  // attach to, since runs execute in the background -- see api.py's
  // _start_background_run. Present at most once: the server clears it as
  // soon as a GET /api/graph reads it.
  run_error: string | null
  wires: Record<string, WireOut>
  // True only while a Run all/Force run all sweep is in flight (see
  // Runner._sweep) -- see App.tsx's polling effect for why this matters
  // beyond the per-block `running` status.
  sweep_running: boolean
}

export interface PreviewColumn {
  name: string
  dtype: string
  role: string
  description: string | null
  // Free-text tags, hand-set or AI-suggested (see api.analyzeData) --
  // BlockInstance.column_tags.
  tags: string[]
}

export interface PreviewSummary {
  count: number
  null_count: number
  n_unique: number | null
  mean: number | null
  std: number | null
  min: unknown
  max: unknown
}

export interface PreviewOut {
  columns: PreviewColumn[]
  rows: Record<string, unknown>[]
  row_count: number
  lineage: string[]
  summary: Record<string, PreviewSummary> | null
}

export interface SchemaColumn {
  name: string
  dtype: string
  role: string
}

export type InputSchemaOut = Record<string, SchemaColumn[]>

export interface BrowseEntry {
  name: string
  path: string
  is_dir: boolean
  // Set (true/false) for directory entries only: whether this folder
  // already holds a saved project (see project.PROJECT_FILENAME).
  is_project?: boolean
}

export interface BrowseOut {
  path: string
  parent: string | null
  entries: BrowseEntry[]
}

export interface DraftOut {
  code: string
  metadata_transform: Record<string, unknown>
  params: Record<string, unknown>
  explanation: string
}

export interface AnalyzeDataOut {
  document: string
  tags: Record<string, string[]>
  // Where the document was written under the project's files/ folder, or
  // null when no project has been saved yet.
  document_path: string | null
  // The persistent Artifact this analysis was saved/updated as -- see
  // ArtifactSummary/Artifact below.
  artifact_id: string
}

export interface ArtifactSummary {
  id: string
  kind: string
  title: string
  block_id: string
  block_name: string | null
  port: string
  created_at: string
  updated_at: string
  // True once the source block has changed (re-run with different
  // params/code/upstream data) since this artifact was generated -- see
  // session.artifact_is_stale.
  stale: boolean
}

export interface Artifact extends ArtifactSummary {
  document: string
}

export interface SuggestNamesOut {
  name: string
  port_names: Record<string, string>
  explanation: string
}

export interface LLMProviderSettings {
  model?: string | null
  base_url?: string
  // Whether an API key is currently configured for this provider, and
  // where it came from -- the key's value is never sent to the client.
  api_key_set?: boolean
  api_key_source?: 'override' | 'env' | null
}

export interface LLMSettingsOut {
  providers: string[]
  active_provider: string
  include_reference: boolean | null
  settings: Record<string, LLMProviderSettings>
}

export interface GitChange {
  status: string
  path: string
}

export interface GitStatusOut {
  is_repo: boolean
  branch: string | null
  remote: string | null
  changes: GitChange[]
  ahead: number
  behind: number
}
