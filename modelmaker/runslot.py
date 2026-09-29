from __future__ import annotations

import threading
from typing import Any, Callable


class RunBusy(RuntimeError):
    """Another run (a canvas-triggered one, or an AI build's) already holds
    the slot."""


class RunFailed(RuntimeError):
    """A run raised before/while executing -- typically a precondition
    failure such as "upstream isn't green yet", which leaves nothing in the
    graph's own block statuses to show it happened."""


class RunSlot:
    """At most one run (a single block, a cascade, or a full sweep) is
    active at a time. Two ways to take the slot:

    - start_background(): used by the HTTP endpoints. Runs fn on a
      background thread, so a run that genuinely takes a while doesn't
      block the request indefinitely -- the frontend polls GET /api/graph
      (whose per-block `status` already reflects "running") to watch it
      progress. It joins the thread for up to `wait_seconds`, so the common
      case still gets a response reflecting the final state, as if the
      call had been synchronous; only a run still going after the wait
      comes back as None ("running") for the client to poll.
    - run_sync(): used by an AI build (see agent/build.py), which is
      already on its own background thread and needs each run's outcome
      before deciding its next step.

    The isolation that actually matters -- one subprocess per block call,
    so a crash or a runaway group_by can't take the server down -- is
    Runner's job and applies either way."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sync_active = False
        # A failure from a background run that outlived its wait window --
        # surfaced once via take_error() (see api._graph_out), which clears
        # it on read.
        self._last_error: str | None = None

    def busy(self) -> bool:
        return self._sync_active or (self._thread is not None and self._thread.is_alive())

    def take_error(self) -> str | None:
        err, self._last_error = self._last_error, None
        return err

    def start_background(self, fn: Callable[[], Any], wait_seconds: float = 20.0) -> Any:
        with self._lock:
            if self.busy():
                raise RunBusy("a run is already in progress")
            self._last_error = None
            holder: dict[str, Any] = {}

            def _target() -> None:
                try:
                    holder["result"] = fn()
                except Exception as e:  # noqa: BLE001 -- has no caller to raise to once the wait below gives up
                    self._last_error = f"{type(e).__name__}: {e}"

            thread = threading.Thread(target=_target, daemon=True)
            self._thread = thread
            thread.start()
        thread.join(timeout=wait_seconds)
        if thread.is_alive():
            return None
        err = self.take_error()
        if err:
            raise RunFailed(err)
        return holder.get("result")

    def run_sync(self, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if self.busy():
                raise RunBusy("a run is already in progress")
            self._sync_active = True
        try:
            return fn()
        finally:
            self._sync_active = False
