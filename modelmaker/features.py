"""Which parts of the app a workspace offers: its feature flags (restricted mode).

Capabilities (capabilities.py) say what a workspace's *blocks* may reach.
Features say which of the app's own tools its *user* may use. The local
app offers all of them, as it always did. An application embedding the
API for other people (a hosted demo, a locked-down install) can switch
some off per workspace (``Workspace(features=...)``):

- ``custom_blocks`` -- block code a user writes or brings: the block
  library (create, edit, check, try, import, export, duplicate, drafts,
  AI drafting of blocks), placing library blocks, updating a placed block
  to a newer library version (accepting a built-in's update stays
  allowed), and opening a project or recovery snapshot that holds a
  library block's embedded code. Without it a workspace runs
  built-in blocks only. The AI builder then can't create blocks or use
  library ones either.
- ``ai_builder`` -- the AI builder (``/api/agent/...``).
- ``ai_drafting`` -- AI help on one block: draft params, suggest a fix,
  analyse data, suggest names (and, with ``custom_blocks``, drafting
  library code).
- ``llm_settings`` -- changing the LLM settings (the Settings panel) and
  listing a provider's models. Reading the settings stays allowed: the AI
  tools show which provider they use.
- ``env_vars`` -- environment variables set from the UI (env-var fields).
- ``git`` -- the Git panel: versioning a saved project with git.
- ``git_remote`` -- its remote features: setting a remote, push and pull
  (only meaningful with ``git``).

A switched-off feature is refused by the API (403, ``ROUTES``, checked
for every request by a middleware in api.py, plus the checks inside
routes that a path alone can't decide, such as placing a library block),
never only hidden. ``GET /api/capabilities`` lists the workspace's
features, so the UI hides what's off instead of showing controls that
fail.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

CUSTOM_BLOCKS = "custom_blocks"
AI_BUILDER = "ai_builder"
AI_DRAFTING = "ai_drafting"
LLM_SETTINGS = "llm_settings"
ENV_VARS = "env_vars"
GIT = "git"
GIT_REMOTE = "git_remote"

ALL: frozenset[str] = frozenset({CUSTOM_BLOCKS, AI_BUILDER, AI_DRAFTING, LLM_SETTINGS, ENV_VARS, GIT, GIT_REMOTE})
# The local app: everything.
LOCAL: frozenset[str] = ALL


class FeatureError(PermissionError):
    """Something a switched-off feature covers was asked for."""


_ANY = None
# (methods or None for any, path pattern, the feature it needs). Every
# matching row applies, so a route can need two (library AI drafting).
ROUTES: list[tuple[frozenset[str] | None, re.Pattern[str], str]] = [
    (_ANY, re.compile(r"^/api/library(/.*)?$"), CUSTOM_BLOCKS),
    (_ANY, re.compile(r"^/api/library/ai(/.*)?$"), AI_DRAFTING),
    (_ANY, re.compile(r"^/api/blocks/[^/]+/(draft|suggest_fix|analyze_data|suggest_names)$"), AI_DRAFTING),
    (_ANY, re.compile(r"^/api/agent(/.*)?$"), AI_BUILDER),
    (frozenset({"PUT", "POST", "PATCH", "DELETE"}), re.compile(r"^/api/llm/settings$"), LLM_SETTINGS),
    (_ANY, re.compile(r"^/api/llm/models$"), LLM_SETTINGS),
    (_ANY, re.compile(r"^/api/env_vars(/.*)?$"), ENV_VARS),
    (_ANY, re.compile(r"^/api/project/git(/.*)?$"), GIT),
    (_ANY, re.compile(r"^/api/project/git/(remote|push|pull)$"), GIT_REMOTE),
]

_DESCRIBE = {
    CUSTOM_BLOCKS: "Custom block code (the block library, its editor and imports)",
    AI_BUILDER: "The AI builder",
    AI_DRAFTING: "AI drafting",
    LLM_SETTINGS: "Changing the LLM settings",
    ENV_VARS: "Environment variables",
    GIT: "Git",
    GIT_REMOTE: "Git remotes (push and pull)",
}


def resolve(features: Iterable[str] | None) -> frozenset[str]:
    """None: every feature (the local app). Unknown names are refused, so a
    typo in an embedding application's configuration fails loudly."""
    if features is None:
        return LOCAL
    result = frozenset(features)
    unknown = result - ALL
    if unknown:
        raise ValueError(f"unknown features: {', '.join(sorted(unknown))} (known: {', '.join(sorted(ALL))})")
    return result


def needed(method: str, path: str) -> list[str]:
    """The features a request to `path` needs, in ROUTES order."""
    method = method.upper()
    out: list[str] = []
    for methods, pattern, feature in ROUTES:
        if (methods is None or method in methods) and pattern.match(path) and feature not in out:
            out.append(feature)
    return out


def refusal(feature: str) -> str:
    what = _DESCRIBE.get(feature, f"'{feature}'")
    return f"{what} isn't available in this workspace."


def denied(method: str, path: str, features: Iterable[str]) -> str | None:
    """Why the request is refused, or None when its features are on."""
    have = frozenset(features)
    for feature in needed(method, path):
        if feature not in have:
            return refusal(feature)
    return None


def check(feature: str, features: Iterable[str]) -> None:
    if feature not in frozenset(features):
        raise FeatureError(refusal(feature))
