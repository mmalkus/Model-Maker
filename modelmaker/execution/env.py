"""What environment a task process gets: this process's, minus the secrets.

Task processes run block code -- built-in, library, or AI-written -- and
used to inherit this process's whole environment, including the API's own
auth token and every LLM provider key. They can't simply get an *empty* environment: some
blocks read a variable by name at run time on purpose (Read SQL's
`connection_env` holds the name of the variable with the connection URI,
see blocks/library.py read_sql), and PATH, HOME, locale and thread-count
variables matter to the libraries blocks use. So it's a denylist: the
variables this app and deployments built on it keep secrets in, plus whatever
an admin adds with MODELMAKER_WORKER_ENV_DENY.

Removing them from `os.environ` inside the task process would not be
enough -- /proc/self/environ keeps showing the environment the process was
exec'd with. worker_pool.py therefore starts each task process with the
scrubbed environment from the start (see Worker), and this module only
decides what that environment is.

Remaining limitation: a task process runs as the same OS user as this one,
so on Linux it could still read *this* process's /proc/<ppid>/environ
(and on any OS, its memory, through a debugger interface). `harden()` --
opt-in, MODELMAKER_HARDEN=1 -- closes that on Linux by marking this process
non-dumpable; the real fix is a separate sandbox with no secrets in it at
all.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

# Secrets by exact name: this app's own (the API's bearer token, LLM
# provider keys it reads, see llm/), the providers' standard names, and the
# git host tokens the CLI tools around it look for.
SECRET_NAMES: frozenset[str] = frozenset(
    {
        "MODELMAKER_AUTH_TOKEN",
        "MODELMAKER_COPILOT_TOKEN",
        "MODELMAKER_OPENAI_API_KEY",
        "MODELMAKER_GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)

# Secrets by prefix: hosted deployments of Model Maker configure
# themselves through MMS_* variables, several of them secret (MMS_SECRET_KEY,
# MMS_ADMIN_PASSWORD, MMS_GATEWAY_TOKEN, MMS_*_API_KEY, ...). None is ever
# meant for block code.
SECRET_PREFIXES: tuple[str, ...] = ("MMS_",)

# MODELMAKER_WORKER_ENV_DENY: extra names, comma-separated; an entry ending
# in `*` is a prefix (e.g. "DWH_ADMIN_*").
EXTRA_DENY_VAR = "MODELMAKER_WORKER_ENV_DENY"


def _extra_denied(environ: Mapping[str, str]) -> tuple[frozenset[str], tuple[str, ...]]:
    names, prefixes = set(), []
    for item in environ.get(EXTRA_DENY_VAR, "").split(","):
        item = item.strip()
        if item.endswith("*") and len(item) > 1:
            prefixes.append(item[:-1])
        elif item:
            names.add(item)
    return frozenset(names), tuple(prefixes)


def _secret_test(environ: Mapping[str, str]):
    names, prefixes = _extra_denied(environ)
    names, prefixes = names | SECRET_NAMES, (*SECRET_PREFIXES, *prefixes)
    if sys.platform == "win32":  # environment names are case-insensitive there
        names = frozenset(n.upper() for n in names)
        prefixes = tuple(p.upper() for p in prefixes)
        return lambda name: name.upper() in names or name.upper().startswith(prefixes)
    return lambda name: name in names or name.startswith(prefixes)


def is_secret(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether `name` is kept out of task processes (`environ` supplies
    MODELMAKER_WORKER_ENV_DENY; default os.environ)."""
    return _secret_test(os.environ if environ is None else environ)(name)


def snapshot_environ() -> dict[str, str]:
    """A copy of os.environ that tolerates another thread setting a
    variable mid-copy (api.set_env_var runs on a request thread while a
    dispatch thread spawns): retry instead of failing the run."""
    for _ in range(10):
        try:
            return dict(os.environ)
        except (RuntimeError, KeyError):
            continue
    return dict(os.environ)


def scrubbed_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """`environ` (default: a snapshot of os.environ) without any secret --
    the environment a task process is started with, and the one each task
    re-applies before it runs (see worker_pool.Worker.send_task)."""
    env = snapshot_environ() if environ is None else dict(environ)
    secret = _secret_test(env)
    return {k: v for k, v in env.items() if not secret(k)}


def task_environ(overrides: Mapping[str, str] | None = None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment one task runs with: `environ` (default: a snapshot
    of os.environ) scrubbed, then `overrides` -- the variables the user set
    for their workspace (see execution.base.Task.env) -- on top.

    The denylist applies to the overrides too, and it's the one from
    `environ`, never from the overrides (so an override of
    MODELMAKER_WORKER_ENV_DENY can't lift it). Setting, say,
    ANTHROPIC_API_KEY from the UI therefore never hands it to block code,
    exactly as when the variable came from the shell; a variable a block
    is meant to read, such as the one holding a Read SQL connection URI,
    simply must not be named like a secret. An admin who wants a name kept
    out of blocks however it's set adds it to MODELMAKER_WORKER_ENV_DENY."""
    env = snapshot_environ() if environ is None else dict(environ)
    secret = _secret_test(env)
    out = {k: v for k, v in env.items() if not secret(k)}
    for name, value in (overrides or {}).items():
        if not secret(name):
            out[name] = value
    return out


# ---- opt-in hardening -----------------------------------------------------

HARDEN_VAR = "MODELMAKER_HARDEN"
_PR_SET_DUMPABLE = 4
_PR_GET_DUMPABLE = 3


def harden_requested() -> bool:
    return os.environ.get(HARDEN_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def harden() -> bool:
    """Mark this process non-dumpable (Linux only): its /proc/<pid>/environ,
    /proc/<pid>/mem and ptrace attach become unavailable to other processes
    of the same user -- task processes included, which otherwise could read
    the secrets scrubbing kept out of their own environment straight from
    the parent's. Costs core dumps and attaching a debugger or profiler
    (py-spy) to this process, which is why it's opt-in. execve resets the
    flag, so task processes themselves stay dumpable. Returns whether the
    flag is now set."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
            return False
        return libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False
