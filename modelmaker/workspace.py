"""A workspace: everything that belongs to one user of the API.

The API used to keep this state in module globals (the open project, the
run slot, the AI builder, the LLM settings, the env-var overrides), which
made one process serve exactly one user. A `Workspace` bundles all of it,
so several workspaces can share one process without seeing each other's
graph, runs or settings. The API looks up the workspace for each request
(see api.current_workspace and api.set_workspace_resolver); the local app
has exactly one, `api.DEFAULT_WORKSPACE`, so it behaves as it always did.

What a workspace owns:

- `session` -- the open project (graph, undo/redo, runner, recovery);
- `run_slot` -- at most one run at a time, *per workspace*;
- `agent` -- the AI builder's controller (its build, lock and MCP token)
  and `agent_callback_url`, where the claude CLI bridge calls back to;
- `llm_settings` -- the Settings panel's LLM choices and API keys;
- `env_vars` -- variables the user set from the UI (PUT /api/env_vars),
  which reach this workspace's task processes and source probes and
  nothing else (never this process's os.environ);
- `cache_dir` (the result cache, recovery snapshot, unsaved builds'
  logs), `projects_dir` (where Save/Load start) and `executor` (where its
  runs go; None = the process-wide default, runner.EXECUTOR).

What it deliberately does *not* own -- shared by every workspace in the
process, and safe to share: the executor's warm worker pool (generic
spares; each task carries its own environment), the block registry and
the content-hash-keyed caches of compiled block code, forms and embedded
definitions, and the block library folder (MODELMAKER_BLOCK_LIBRARY).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent.build import LLMChoice
from .agent.controller import BuildController
from .agent.loop import AgentLoop, make_loop
from .llm import LLMProvider, get_provider
from .llm.settings import LLMSettingsStore
from .runslot import RunSlot
from .session import CACHE_DIR, ProjectSession

if TYPE_CHECKING:
    from .execution import Executor

RECOVERY_FILENAME = "recovery.json"

# Env vars each key-requiring provider's API key can fall back to, in
# priority order -- mirrors what each provider's own __init__ checks. These
# are the *process's* variables (the shell, a .env file, a deployment's
# configuration), never a workspace's own `env_vars`: those are for block
# code, and a key typed into Settings lives in that workspace's
# llm_settings instead.
API_KEY_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["MODELMAKER_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "gemini": ["MODELMAKER_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


def default_provider_name() -> str:
    return os.environ.get("MODELMAKER_LLM_PROVIDER", "claude_cli")


class EnvVars:
    """A workspace's user-set environment variables: name -> value, in
    memory only (never written to disk, never echoed back by the API).
    Thread-safe: request threads set them while run threads snapshot them
    for a dispatch."""

    def __init__(self, initial: Mapping[str, str] | None = None) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, str] = dict(initial or {})

    def set(self, name: str, value: str) -> None:
        with self._lock:
            self._values[name] = value

    def unset(self, name: str) -> bool:
        """Remove an override; whether there was one."""
        with self._lock:
            return self._values.pop(name, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def get(self, name: str) -> str | None:
        with self._lock:
            return self._values.get(name)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._values)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self.snapshot())

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


class Workspace:
    """One user's state in the API (see the module docstring).

    `id` is only a label (logs, an embedding application's own registry);
    nothing in this package interprets it. `cache_dir` defaults to the
    process-wide MODELMAKER_CACHE_DIR, which is right for the one workspace
    of the local app and wrong for any second one: give every additional
    workspace a directory of its own, or they share result files and,
    worse, one recovery snapshot."""

    def __init__(
        self,
        id: str = "default",
        *,
        cache_dir: Path | str | None = None,
        projects_dir: Path | str | None = None,
        executor: Executor | None = None,
        llm_settings: LLMSettingsStore | None = None,
        recovery: bool = True,
    ) -> None:
        self.id = id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.projects_dir = (
            Path(projects_dir).expanduser().resolve()
            if projects_dir is not None
            else Path(os.environ.get("MODELMAKER_PROJECTS_DIR", "./projects")).expanduser().resolve()
        )
        self.executor = executor
        self.recovery_path: Path | None = self.cache_dir / RECOVERY_FILENAME if recovery else None
        self.env_vars = EnvVars()
        self.llm_settings = llm_settings if llm_settings is not None else LLMSettingsStore()
        self.run_slot = RunSlot()
        self._session = self.new_session()
        # Where the claude CLI loop's MCP bridge calls back to -- set from
        # the request that starts or continues a build (see api.agent_start).
        self.agent_callback_url = ""
        self.agent = BuildController(lambda: self.session, self.run_slot, self.agent_loop)

    def __repr__(self) -> str:
        return f"Workspace({self.id!r}, cache_dir={str(self.cache_dir)!r})"

    # ---- the open project ---------------------------------------------

    def new_session(self) -> ProjectSession:
        """A fresh ProjectSession wired to this workspace's cache, recovery
        snapshot, executor and env vars."""
        return ProjectSession(
            recovery_path=self.recovery_path,
            cache_dir=self.cache_dir,
            executor=self.executor,
            task_env=self.env_vars.snapshot,
        )

    @property
    def session(self) -> ProjectSession:
        return self._session

    @session.setter
    def session(self, session: ProjectSession) -> None:
        """Swap in another session (the test suite does, through the
        api.SESSION alias). One made without a task_env is adopted: its runs
        get this workspace's env vars, as a session made here would."""
        if session.task_env is None:
            session.task_env = self.env_vars.snapshot
        self._session = session

    # ---- LLMs ------------------------------------------------------------

    def resolve_api_key(self, provider_name: str) -> str | None:
        """The API key this workspace would use for `provider_name` right
        now: its Settings override, else the first set process env var.
        Only ever used server-side, never sent to the client."""
        override = self.llm_settings.for_provider(provider_name).get("api_key")
        if override:
            return override
        for var in API_KEY_ENV_VARS.get(provider_name, []):
            value = os.environ.get(var)
            if value:
                return value
        return None

    def active_provider(self) -> str:
        return self.llm_settings.active_provider or default_provider_name()

    def provider(self, name: str | None = None) -> LLMProvider:
        """A provider for a draft/suggest call: `name`, else this
        workspace's active provider, configured with its overrides (model,
        base_url, api_key, include_reference). A fresh instance per call,
        so nothing one workspace configured is left behind for another."""
        provider_name = name or self.active_provider()
        overrides: dict[str, Any] = dict(self.llm_settings.for_provider(provider_name))
        if self.llm_settings.include_reference is not None:
            overrides.setdefault("include_reference", self.llm_settings.include_reference)
        return get_provider(provider_name, **overrides)

    def agent_llm_choice(self, stored: dict[str, Any], override: dict[str, Any] | None) -> LLMChoice:
        """Resolve the plan or build LLM: a per-build override, else
        Settings' AI-builder choice, else the active provider -- with the
        model falling back to that provider's configured model (else the
        loop's default)."""
        provider = (override or {}).get("provider") or stored.get("provider") or self.active_provider()
        model = (override or {}).get("model") or stored.get("model") or self.llm_settings.for_provider(provider).get("model")
        return LLMChoice(provider=provider, model=model)

    def agent_loop(self, choice: LLMChoice, phase: str) -> AgentLoop:
        """The BuildController's loop factory: an agent loop for `choice`,
        with this workspace's key, endpoint, callback address and token."""
        provider = choice.provider or "claude_cli"
        return make_loop(
            provider,
            choice.model,
            api_base_url=self.agent_callback_url,
            token=self.agent.token or "",
            api_key=self.resolve_api_key(provider),
            base_url=self.llm_settings.for_provider(provider).get("base_url"),
        )
