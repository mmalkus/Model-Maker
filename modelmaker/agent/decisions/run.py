"""One decision-based build against a live session: plan on the anchor's
output, lay the plan out on the canvas, attach the report, run it."""

from __future__ import annotations

from typing import Any

from .backends import get_backend
from .core import Decider
from .emit import anchor_output, emit, new_build_id, report, workspace_from
from .planner import plan


def decision_build(session, anchor: str, backend: str = "rules", run: bool = True) -> dict[str, Any]:
    port, packet = anchor_output(session, anchor)
    ws = workspace_from(packet)
    if ws.col("target") is None and ws.col("date") is None:
        raise ValueError("tag the target column (or, for a monthly panel, the id and date columns) on the anchor first")
    decider = Decider(get_backend(backend))
    result = plan(ws, decider)
    build_id = new_build_id()
    with session.transaction():  # one Undo reverts the whole build, report included
        with session.edit():
            emitted = emit(session, anchor, port, ws, result, build_id, backend)
        rb = session.graph.blocks[emitted["report_block"]]
        with session.edit():
            session.upsert_artifact(
                "build_report",
                rb.id,
                rb.outputs[0].name if rb.outputs else port,
                f"Decision-based PD build {build_id}",
                report(result, decider, backend, build_id),
                key=build_id,
            )
    failed = []
    if run:
        for sink in emitted["sinks"]:
            session.runner.run_to_here(sink)
        failed = [
            session.graph.blocks[b].name
            for b in emitted["blocks"]
            if (st := session.runner.state.get(b)) and st.last_error
        ]
    return {
        "build_id": build_id,
        "steps": [s["option"] for s in result.steps],
        "sign_off": ws.artifacts.get("sign_off"),
        "blocks": len(emitted["blocks"]),
        "decisions": len(decider.records),
        "to_confirm": sum(r.needs_user for r in decider.records),
        "failed": failed,
    }
