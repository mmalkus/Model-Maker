import { api } from "./api";

// Restricted mode: which of the app's tools this workspace offers (see the
// API's features.py). The local app offers all of them; an embedding
// application can switch some off, and the API then refuses them -- so the
// UI hides them instead of showing controls that fail. Read once, before
// the app renders (main.tsx).
export type Feature =
  | "custom_blocks"
  | "ai_builder"
  | "ai_drafting"
  | "llm_settings"
  | "env_vars"
  | "git"
  | "git_remote";

const ALL: Feature[] = [
  "custom_blocks",
  "ai_builder",
  "ai_drafting",
  "llm_settings",
  "env_vars",
  "git",
  "git_remote",
];

let enabled = new Set<string>(ALL);

export async function loadFeatures(): Promise<void> {
  try {
    const caps = await api.capabilities();
    // An older backend doesn't list features: it offers everything.
    if (Array.isArray(caps.features)) enabled = new Set(caps.features);
  } catch {
    // Unreachable for now: keep everything on; the API still refuses
    // whatever its workspace has switched off.
  }
}

export function has(feature: Feature): boolean {
  return enabled.has(feature);
}
