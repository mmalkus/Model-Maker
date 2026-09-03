export type Status = 'grey' | 'green' | 'orange' | 'red'
export type PortType = 'dataframe' | 'model' | 'scalar_metric'
export type BlockType = 'input' | 'standard' | 'output' | 'llm_authored'

export interface PortSpec {
  name: string
  type: PortType
  required: boolean
}

export interface RegistryEntry {
  category: string
  block_type: BlockType
  display_name: string
  inputs: PortSpec[]
  outputs: PortSpec[]
}

export interface BlockOut {
  id: string
  block_type: BlockType
  category: string
  name: string
  lane: string | null
  position: { x: number; y: number }
  code_version: number
  params: Record<string, unknown>
  code: string | null
  metadata_transform: Record<string, unknown> | null
  inputs: PortSpec[]
  outputs: PortSpec[]
  status: Status
  last_error: string | null
  last_successful_read_at: string | null
  last_attempt_at: string | null
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
}

export interface GraphOut {
  project_name: string
  project_path: string | null
  lanes: Record<string, LaneOut>
  blocks: Record<string, BlockOut>
  wires: Record<string, WireOut>
}

export interface PreviewColumn {
  name: string
  dtype: string
  role: string
  description: string | null
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

export interface DraftOut {
  code: string
  metadata_transform: Record<string, unknown>
  params: Record<string, unknown>
  explanation: string
}
