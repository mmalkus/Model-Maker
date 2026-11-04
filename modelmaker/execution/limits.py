"""Per-task resource limits: memory and threads, enforced on each task
process by the executor that runs it (local.py) -- opt-in, off unless set.

Why not just rlimits, or the container's own limits. A sandbox's
`--memory` and `--pids-limit` bound the sandbox as a whole, and how they
fail depends on the runtime: with runc the kernel OOM-kills some process
(a task, or the agent) and thread creation fails; under gVisor the pids
limit doesn't apply to the sandbox's own threads at all, and memory
exhaustion doesn't kill anything -- the sandbox is held at its limit and
thrashes until a wall-clock limit stops the task. Neither says *which*
block did it, or why. And a plain `RLIMIT_AS` budget doesn't work as the
memory limit: polars, numpy and the allocators reserve far more address
space than they touch (measured: ~1.7 GB of virtual size for ~200 MB
resident after a polars + sklearn import, a join and a logistic
regression, growing with the number of CPUs), so a budget tight enough to
mean anything breaks ordinary runs. (`RLIMIT_DATA` isn't enforced under
gVisor at all.)

So the limits here are enforced by a watchdog (`Watchdog`) that the
executor's wait loop polls, every POLL_INTERVAL, for each running task:

- `memory_mb`: the task's *private* resident memory -- anonymous and
  shared-anonymous pages (`RssAnon` + `RssShmem` in /proc/<pid>/status),
  summed over the task process and every process it started (when it has
  several, and that sum is over the limit, their proportional share
  instead -- Pss_Anon + Pss_Shmem from smaps_rollup -- so that the pages a
  fork shares with its parent count once). Not VmRSS:
  that counts the mapped files too (the polars library alone is ~150 MB),
  which are shared with every other process and reclaimable, and which
  gVisor reports as fully resident. gVisor doesn't report RssAnon, so
  there it's the `Anonymous:` lines of /proc/<pid>/smaps -- slower to read,
  so read only when VmRSS says the task might be over (see _Memory).
- `threads`: the threads of the task process and every process it started,
  together (a process counts as its threads).

A task over either is killed -- its whole process group: the task process
puts itself in a group of its own when it has limits (apply_in_child), so
nothing it started survives it -- and fails with an error naming the
limit. (With limits, whatever a task started is also killed when the task
ends, however it ends.) Two rlimits back the watchdog up in the task process, for what is
faster than a poll:

- `RLIMIT_AS`, set well above any normal virtual size (address_space_mb):
  a single huge allocation fails at once with a MemoryError instead of
  filling the sandbox between two polls;
- `RLIMIT_NPROC` = `total_threads`, when set: the threads and processes of
  this uid, all tasks *and the agent* together, the way a container's pids
  limit counts them. Under runc that's what --pids-limit already does;
  under gVisor, which ignores --pids-limit but enforces RLIMIT_NPROC, it's
  what makes a fork or thread bomb fail fast rather than run until the
  watchdog's next poll.

A block that hits one of the rlimits gets the ordinary Python error
(MemoryError, "can't start new thread"); the executor adds which limit it
was to the error (annotate_error).

Linux only: elsewhere (no /proc) the watchdog measures nothing and only
the rlimits apply.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, replace
from typing import Any

# How often a task's process tree is listed again (its processes are
# measured at every poll; finding new ones means scanning /proc).
TREE_INTERVAL = 0.2

# Errors a block gets from the rlimits, by what they say.
_THREAD_ERRORS = ("can't start new thread", "Resource temporarily unavailable", "failed to spawn thread")


@dataclass(frozen=True)
class TaskLimits:
    """Per-task limits (None: no limit). See the module docstring."""

    memory_mb: int | None = None
    threads: int | None = None
    total_threads: int | None = None

    def __bool__(self) -> bool:
        return any(v is not None for v in (self.memory_mb, self.threads, self.total_threads))

    def stricter(self, other: TaskLimits | None) -> TaskLimits:
        """The stricter of the two, field by field."""
        if other is None:
            return self

        def pick(a: int | None, b: int | None) -> int | None:
            return b if a is None else a if b is None else min(a, b)

        return replace(
            self,
            memory_mb=pick(self.memory_mb, other.memory_mb),
            threads=pick(self.threads, other.threads),
            total_threads=pick(self.total_threads, other.total_threads),
        )

    def child_fields(self) -> dict[str, int]:
        """What the task process needs to know (apply_in_child): that it has
        limits at all (so it makes its own process group), and the two it
        backs up with rlimits."""
        fields: dict[str, int] = {"group": 1}
        fields.update((k, v) for k, v in (("memory_mb", self.memory_mb), ("total_threads", self.total_threads)) if v is not None)
        return fields


def check_limit(value: Any, name: str) -> int | None:
    """A limit as the protocol or a caller gives it: None, or a positive int."""
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"bad {name}")
    return value


def address_space_mb(memory_mb: int) -> int:
    """The RLIMIT_AS backstop for a memory budget: far above normal virtual
    usage, which is mostly reservations (each thread's stack, malloc's
    per-thread arenas, polars' allocator) and grows with the CPU count."""
    return 2 * memory_mb + 4096 + 128 * (os.cpu_count() or 4)


# ---- the task process's side -------------------------------------------------


def apply_in_child(fields: dict[str, Any] | None) -> None:
    """Called by the task process itself, before running its task, when it
    has limits: its own process group (so the watchdog can kill all of it),
    and the rlimit backstops. Lowered hard limits too: block code can't
    raise them back."""
    if not fields:
        return
    try:
        os.setpgid(0, 0)
    except (AttributeError, OSError):
        pass
    try:
        import resource
    except ImportError:  # not POSIX
        return
    memory_mb = fields.get("memory_mb")
    if type(memory_mb) is int and memory_mb > 0:
        _lower(resource, resource.RLIMIT_AS, address_space_mb(memory_mb) * 1024 * 1024)
    total = fields.get("total_threads")
    if type(total) is int and total > 0 and hasattr(resource, "RLIMIT_NPROC"):
        _lower(resource, resource.RLIMIT_NPROC, total)


def _lower(resource: Any, which: int, value: int) -> None:
    try:
        _soft, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(which, (value, value))
    except (ValueError, OSError):
        pass


# ---- the executor's side -------------------------------------------------------


def memory_error(limit_mb: int) -> str:
    return f"stopped: the block used more than its memory limit of {limit_mb:,} MB"


def threads_error(limit: int) -> str:
    return f"stopped: the block ran more than its limit of {limit:,} threads and processes"


def annotate_error(error: str, limits: TaskLimits) -> str:
    """A block's own error, with the limit it most likely hit named."""
    if limits.memory_mb is not None and error.startswith("MemoryError"):
        return f"{error} (the block reached its memory limit of {limits.memory_mb:,} MB)"
    caps = [c for c in (limits.threads, limits.total_threads) if c is not None]
    if caps and any(s in error for s in _THREAD_ERRORS):
        return f"{error} (the block reached its limit on threads and processes: {min(caps):,})"
    return error


def _read_status(pid: int) -> dict[str, int] | None:
    """VmRSS, RssAnon, RssShmem (kB) and Threads of `pid`, as reported."""
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status", "rb") as f:
            for line in f:
                key, _, value = line.partition(b":")
                if key in (b"VmRSS", b"RssAnon", b"RssShmem", b"Threads"):
                    out[key.decode()] = int(value.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return out


def _rollup_pss_kb(pid: int) -> int | None:
    """Proportional private memory (Pss_Anon + Pss_Shmem) from Linux's
    smaps_rollup: what a forked child really adds, its copy-on-write pages
    shared with its parent counted once between them."""
    found, total = False, 0
    try:
        with open(f"/proc/{pid}/smaps_rollup", "rb") as f:
            for line in f:
                if line.startswith((b"Pss_Anon:", b"Pss_Shmem:")):
                    total += int(line.split()[1])
                    found = True
    except (OSError, ValueError, IndexError):
        return None
    return total if found else None


def _smaps_anon_kb(pid: int) -> int | None:
    total = 0
    try:
        with open(f"/proc/{pid}/smaps", "rb") as f:
            for line in f:
                if line.startswith(b"Anonymous:"):
                    total += int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return total


def _group_members(pgid: int) -> list[int]:
    """Every process in process group `pgid` (field 5 of /proc/<pid>/stat)."""
    out = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            fields = stat[stat.rindex(b")") + 2 :].split()
            if int(fields[2]) == pgid:
                out.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return out


class _Memory:
    """One process's private resident memory. Where /proc/<pid>/status has
    RssAnon (Linux), that's it, at the cost of one small read. Where it
    hasn't (gVisor), the smaps sum is the truth but costs tens of ms, so
    it's read only when it could matter: between reads, the estimate is the
    last smaps value plus however much VmRSS has grown from its lowest
    since -- never lower than the truth, so the task is never let past its
    limit, and smaps is read again only once that estimate is over it."""

    __slots__ = ("rss_floor", "smaps_kb")

    def __init__(self) -> None:
        self.smaps_kb: int | None = None
        self.rss_floor = 0

    def estimate(self, status: dict[str, int]) -> int | None:
        if "RssAnon" in status:
            return status["RssAnon"] + status.get("RssShmem", 0)
        rss = status.get("VmRSS")
        if rss is None or self.smaps_kb is None:
            return rss
        self.rss_floor = min(self.rss_floor, rss)
        return self.smaps_kb + rss - self.rss_floor

    def exact(self, pid: int, status: dict[str, int], shared: bool) -> int | None:
        """`shared`: the task has more than one process, which may share
        copy-on-write pages (a fork) -- counted proportionally then."""
        if "RssAnon" in status:
            pss = _rollup_pss_kb(pid) if shared else None
            return pss if pss is not None else self.estimate(status)
        anon = _smaps_anon_kb(pid)
        if anon is None:
            return status.get("VmRSS")
        self.smaps_kb, self.rss_floor = anon, status.get("VmRSS", 0)
        return anon


class Watchdog:
    """Measures one task's process tree against its limits (see the module
    docstring). `check()` returns the error to stop it with, or None."""

    def __init__(self, pid: int, limits: TaskLimits) -> None:
        self.pid = pid
        self.limits = limits
        self.enabled = os.path.isdir(f"/proc/{pid}") and (limits.memory_mb is not None or limits.threads is not None)
        self._members: list[int] = [pid]
        self._listed_at = 0.0
        self._memory: dict[int, _Memory] = {}

    def check(self, now: float | None = None) -> str | None:
        if not self.enabled:
            return None
        now = time.monotonic() if now is None else now
        if now - self._listed_at >= TREE_INTERVAL:
            # The task's group (it made its own, apply_in_child); its own
            # pid always, in case it hasn't got that far yet.
            self._members = sorted({self.pid, *_group_members(self.pid)})
            self._listed_at = now
            self._memory = {p: self._memory.get(p) or _Memory() for p in self._members}
        statuses = {}
        for pid in self._members:
            status = _read_status(pid)
            if status is not None:
                statuses[pid] = status
        if not statuses:
            return None
        threads = self.limits.threads
        if threads is not None and sum(s.get("Threads", 1) for s in statuses.values()) > threads:
            return threads_error(threads)
        limit_mb = self.limits.memory_mb
        if limit_mb is None:
            return None
        limit_kb = limit_mb * 1024
        mems = {pid: self._memory.setdefault(pid, _Memory()) for pid in statuses}
        if sum(mems[p].estimate(s) or 0 for p, s in statuses.items()) <= limit_kb:
            return None
        shared = len(statuses) > 1
        exact = sum(mems[p].exact(p, s, shared) or 0 for p, s in statuses.items())
        return memory_error(limit_mb) if exact > limit_kb else None

    def kill(self) -> None:
        """SIGKILL the task's whole process group, and the task process in
        case it never made its group. (Not the members by pid: one that has
        exited since it was listed may have had its pid reused. The task
        process can't have: its parent hasn't reaped it.)"""
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (AttributeError, OSError):
            pass
        try:
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass
