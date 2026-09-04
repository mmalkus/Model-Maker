import type {
  BlockOut,
  BlockType,
  BrowseOut,
  DraftOut,
  GraphOut,
  InputSchemaOut,
  LLMSettingsOut,
  PreviewOut,
  RegistryEntry,
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
  refreshAll: () => request<Record<string, string>>('/refresh_all', { method: 'POST' }),
  checkAllSources: () => request<Record<string, boolean>>('/check_all_sources', { method: 'POST' }),

  compile: (output_blocks?: string[]) =>
    request<{ source: string }>('/compile', { method: 'POST', body: JSON.stringify({ output_blocks }) }),

  save: (path?: string) => request<{ path: string }>('/project/save', { method: 'POST', body: JSON.stringify({ path }) }),
  load: (path: string) => request<GraphOut>('/project/load', { method: 'POST', body: JSON.stringify({ path }) }),
  projectDefaultDir: () => request<{ path: string }>('/project/default_dir'),

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
    settings?: Record<string, { model?: string | null; base_url?: string }>
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
}
