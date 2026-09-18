import type {
  AnalyzeDataOut,
  Artifact,
  ArtifactSummary,
  BlockOut,
  BlockType,
  BrowseOut,
  DraftOut,
  GitStatusOut,
  GraphOut,
  InputSchemaOut,
  LLMSettingsOut,
  PreviewOut,
  RecoveryInfo,
  RegistryEntry,
  SuggestNamesOut,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  registry: () => request<RegistryEntry[]>('/registry'),
  graph: () => request<GraphOut>('/graph'),

  createBlock: (body: {
    category: string
    block_type?: BlockType
    name?: string
    lane?: string | null
    x?: number
    y?: number
    params?: Record<string, unknown>
    code?: string
    inputs?: { name: string; type: string }[]
    outputs?: { name: string; type: string }[]
    metadata_transform?: Record<string, unknown>
  }) => request<BlockOut>('/blocks', { method: 'POST', body: JSON.stringify(body) }),

  updateBlock: (
    id: string,
    body: Partial<{
      name: string
      lane: string | null
      position: { x: number; y: number }
      params: Record<string, unknown>
      code: string
      metadata_transform: Record<string, unknown>
      group_by: string | null
      max_workers: number | null
    }>,
  ) => request<BlockOut>(`/blocks/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),

  deleteBlock: (id: string) => request<void>(`/blocks/${id}`, { method: 'DELETE' }),

  upsertLane: (id: string, name: string, order: number, height?: number) =>
    request<Record<string, { name: string; order: number; height: number }>>('/lanes', {
      method: 'PUT',
      body: JSON.stringify({ id, name, order, height }),
    }),
  deleteLane: (id: string) =>
    request<Record<string, { name: string; order: number; height: number }>>(`/lanes/${id}`, { method: 'DELETE' }),

  inputSchema: (id: string) => request<InputSchemaOut>(`/blocks/${id}/input_schema`),

  setColumnRole: (id: string, column: string, role: string) =>
    request<BlockOut>(`/blocks/${id}/column_role`, { method: 'POST', body: JSON.stringify({ column, role }) }),

  preview: (id: string, opts: { port?: string; rows?: number; summary?: boolean } = {}) => {
    const q = new URLSearchParams()
    if (opts.port) q.set('port', opts.port)
    if (opts.rows) q.set('rows', String(opts.rows))
    if (opts.summary) q.set('summary', 'true')
    const qs = q.toString()
    return request<PreviewOut>(`/blocks/${id}/preview${qs ? `?${qs}` : ''}`)
  },

  createWire: (body: { from_block: string; from_port: string; to_block: string; to_port: string }) =>
    request<{ id: string; valid: boolean }>('/wires', { method: 'POST', body: JSON.stringify(body) }),
  deleteWire: (id: string) => request<void>(`/wires/${id}`, { method: 'DELETE' }),
  renamePort: (blockId: string, port: string, name: string | null) =>
    request<BlockOut>(`/blocks/${blockId}/port_name`, { method: 'PATCH', body: JSON.stringify({ port, name }) }),

  run: (id: string) => request<BlockOut>(`/blocks/${id}/run`, { method: 'POST' }),
  runToHere: (id: string) => request<BlockOut>(`/blocks/${id}/run_to_here`, { method: 'POST' }),
  refresh: (id: string) => request<BlockOut>(`/blocks/${id}/refresh`, { method: 'POST' }),
  checkChanges: (id: string) => request<{ changed: boolean }>(`/blocks/${id}/check_changes`, { method: 'POST' }),

  runAll: () => request<Record<string, string>>('/run_all', { method: 'POST' }),
  forceRunAll: () => request<Record<string, string>>('/force_run_all', { method: 'POST' }),
  // Opt-in streaming run (see Runner.run_all_streaming): fuses whatever
  // contiguous stretch of compatible blocks it safely can into one polars
  // query per group, so a source larger than memory doesn't fully
  // materialize at every block boundary. Fused interior blocks report as
  // "fused" rather than turning green individually -- see the tooltip on
  // the toolbar button that calls this.
  runAllStreaming: () => request<Record<string, string>>('/run_all_streaming', { method: 'POST' }),
  refreshAll: () => request<Record<string, string>>('/refresh_all', { method: 'POST' }),
  checkAllSources: () => request<Record<string, boolean>>('/check_all_sources', { method: 'POST' }),
  cancelRun: () => request<{ cancelled: boolean }>('/run/cancel', { method: 'POST' }),

  compile: (output_blocks?: string[], stream?: boolean) =>
    request<{ source: string }>('/compile', { method: 'POST', body: JSON.stringify({ output_blocks, stream }) }),

  undo: () => request<GraphOut>('/undo', { method: 'POST' }),
  redo: () => request<GraphOut>('/redo', { method: 'POST' }),

  setSampleMode: (rows: number | null) =>
    request<GraphOut>('/sample_mode', { method: 'PUT', body: JSON.stringify({ rows }) }),

  recoveryInfo: () => request<{ recovery: RecoveryInfo | null }>('/project/recovery'),
  recover: () => request<GraphOut>('/project/recover', { method: 'POST' }),

  save: (path?: string) => request<{ path: string }>('/project/save', { method: 'POST', body: JSON.stringify({ path }) }),
  load: (path: string) => request<GraphOut>('/project/load', { method: 'POST', body: JSON.stringify({ path }) }),
  new: () => request<GraphOut>('/project/new', { method: 'POST' }),
  projectDefaultDir: () => request<{ path: string }>('/project/default_dir'),

  // Git status/commit/push for the currently-open project's folder -- see
  // gitops.py. Every write endpoint 404s (well, 400s) until a project has
  // been saved at least once, since there's no folder to operate on yet.
  gitStatus: () => request<GitStatusOut>('/project/git/status'),
  gitInit: () => request<GitStatusOut>('/project/git/init', { method: 'POST' }),
  gitSetRemote: (url: string) =>
    request<GitStatusOut>('/project/git/remote', { method: 'POST', body: JSON.stringify({ url }) }),
  gitCommit: (message: string) =>
    request<GitStatusOut>('/project/git/commit', { method: 'POST', body: JSON.stringify({ message }) }),
  gitPush: () => request<GitStatusOut>('/project/git/push', { method: 'POST' }),
  gitPull: () => request<GitStatusOut>('/project/git/pull', { method: 'POST' }),

  imageUrl: (id: string, port: string, cacheBust?: string | null) =>
    `/api/blocks/${id}/image?port=${encodeURIComponent(port)}${cacheBust ? `&t=${encodeURIComponent(cacheBust)}` : ''}`,

  value: (id: string, port: string) => request<unknown>(`/blocks/${id}/value?port=${encodeURIComponent(port)}`),

  browse: (path?: string, ext?: string) => {
    const q = new URLSearchParams()
    if (path) q.set('path', path)
    if (ext) q.set('ext', ext)
    const qs = q.toString()
    return request<BrowseOut>(`/browse${qs ? `?${qs}` : ''}`)
  },

  llmSettings: () => request<LLMSettingsOut>('/llm/settings'),
  updateLlmSettings: (body: {
    active_provider?: string
    include_reference?: boolean | null
    settings?: Record<string, { model?: string | null; base_url?: string | null; api_key?: string | null }>
  }) => request<LLMSettingsOut>('/llm/settings', { method: 'PUT', body: JSON.stringify(body) }),
  llmModels: (provider: string, baseUrl?: string) => {
    const q = new URLSearchParams({ provider })
    if (baseUrl) q.set('base_url', baseUrl)
    return request<{ models: string[] }>(`/llm/models?${q.toString()}`)
  },
  draftBlock: (id: string, instruction: string, provider?: string) =>
    request<DraftOut>(`/blocks/${id}/draft`, { method: 'POST', body: JSON.stringify({ instruction, provider }) }),
  suggestFix: (id: string, instruction?: string, provider?: string) =>
    request<DraftOut>(`/blocks/${id}/suggest_fix`, {
      method: 'POST',
      body: JSON.stringify({ instruction: instruction ?? '', provider }),
    }),
  analyzeData: (id: string, port?: string, provider?: string) =>
    request<AnalyzeDataOut>(`/blocks/${id}/analyze_data`, { method: 'POST', body: JSON.stringify({ port, provider }) }),
  suggestNames: (id: string, provider?: string) =>
    request<SuggestNamesOut>(`/blocks/${id}/suggest_names`, { method: 'POST', body: JSON.stringify({ provider }) }),

  listArtifacts: () => request<ArtifactSummary[]>('/artifacts'),
  getArtifact: (id: string) => request<Artifact>(`/artifacts/${id}`),
  renameArtifact: (id: string, title: string) =>
    request<Artifact>(`/artifacts/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteArtifact: (id: string) => request<void>(`/artifacts/${id}`, { method: 'DELETE' }),
}
