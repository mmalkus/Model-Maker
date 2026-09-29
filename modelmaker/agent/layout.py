"""Where an AI build's blocks go on the canvas. The model picks a lane;
this picks x/y -- LLMs are poor at coordinates, and a consistent layout
is easier to read. Mirrors the frontend's lane geometry (LaneBand.tsx):
lanes are horizontal bands stacked top to bottom by `order`, each
`height` tall, starting at y=0; blocks sit in a row inside their band."""

from __future__ import annotations

from dataclasses import dataclass

from ..graph import Graph

START_X = 40.0
X_SPACING = 240.0
Y_OFFSET = 60.0
DEFAULT_LANE_HEIGHT = 260.0


@dataclass
class LaneGeom:
    lane_id: str
    name: str
    order: int
    top: float
    height: float


def lane_geometry(graph: Graph, extra: list[tuple[str, str]] | None = None) -> dict[str, LaneGeom]:
    """Every lane's band, in order -- plus `extra` (lane_id, name) pairs for
    lanes that don't exist yet (a plan's new lanes), appended after the
    existing ones in the given order."""
    ordered = sorted(graph.lanes.items(), key=lambda kv: kv[1].order)
    out: dict[str, LaneGeom] = {}
    y = 0.0
    order = 0
    for lane_id, lane in ordered:
        out[lane_id] = LaneGeom(lane_id, lane.name, lane.order, y, lane.height)
        y += lane.height
        order = max(order, lane.order + 1)
    for lane_id, name in extra or []:
        if lane_id in out:
            continue
        out[lane_id] = LaneGeom(lane_id, name, order, y, DEFAULT_LANE_HEIGHT)
        y += DEFAULT_LANE_HEIGHT
        order += 1
    return out


def next_lane_order(graph: Graph) -> int:
    return max((lane.order for lane in graph.lanes.values()), default=-1) + 1


class Placer:
    """Hands out positions lane by lane, left to right after whatever is
    already in the lane. Used both to lay out a plan's ghost blocks and to
    place real blocks during the build -- same algorithm, same order, so a
    real block lands where its ghost was."""

    def __init__(self, graph: Graph, extra_lanes: list[tuple[str, str]] | None = None) -> None:
        self.lanes = lane_geometry(graph, extra_lanes)
        self.next_x: dict[str | None, float] = {}
        for block in graph.blocks.values():
            key = block.lane if block.lane in self.lanes else None
            self.next_x[key] = max(self.next_x.get(key, START_X), block.position.x + X_SPACING)

    def place(self, lane_id: str | None) -> tuple[float, float]:
        key = lane_id if lane_id in self.lanes else None
        x = self.next_x.get(key, START_X)
        self.next_x[key] = x + X_SPACING
        top = self.lanes[key].top if key is not None else 0.0
        return x, top + Y_OFFSET
