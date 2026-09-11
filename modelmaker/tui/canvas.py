from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.containers import HorizontalScroll, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Static

# Lanes render as a stack of titled, horizontally-scrolling rows of block
# "chips" instead of a literal pixel x/y canvas -- a free 2D canvas doesn't
# map to a character grid at a readable size, so each lane becomes a flow
# of chips in position order and wires are a separate list panel rather
# than drawn ASCII lines (fragile to route well; a list is exact and lets
# you jump straight to either endpoint).

STATUS_COLORS = {
    "grey": "#9ca3af",
    "green": "#22c55e",
    "orange": "#f59e0b",
    "red": "#ef4444",
}
STATUS_GLYPHS = {"grey": "○", "green": "●", "orange": "◐", "red": "✖"}
BLOCK_TYPE_GLYPHS = {"input": "▶", "standard": "■", "output": "◀"}


class BlockChip(Static, can_focus=True):
    """One block, rendered as a small bordered box colored by run status.
    Focusable (keyboard-navigable via the parent Canvas's arrow handling,
    or plain Tab) and clickable; posts Selected when it gains focus."""

    DEFAULT_CSS = """
    BlockChip {
        width: 26;
        height: 5;
        margin: 0 1 0 0;
        padding: 0 1;
        border: round #9ca3af;
        content-align: left top;
    }
    BlockChip:focus {
        border: heavy $accent;
    }
    """

    class Selected(Message):
        def __init__(self, block_id: str) -> None:
            self.block_id = block_id
            super().__init__()

    def __init__(self, block: dict[str, Any]) -> None:
        super().__init__(id=f"chip-{block['id']}")
        self.block_id: str = block["id"]
        self.set_block(block)

    def set_block(self, block: dict[str, Any]) -> None:
        self.block = block
        status = block.get("status", "grey")
        self.styles.border = ("round", STATUS_COLORS.get(status, STATUS_COLORS["grey"]))
        type_glyph = BLOCK_TYPE_GLYPHS.get(block.get("block_type", "standard"), "■")
        status_glyph = STATUS_GLYPHS.get(status, "○")
        name = block.get("name") or block.get("category", "?")
        category = block.get("category", "")
        text = Text()
        text.append(f"{type_glyph} ", style="dim")
        text.append(name, style="bold")
        text.append("\n")
        text.append(category, style="dim italic")
        text.append("\n")
        text.append(f"{status_glyph} {status}", style=STATUS_COLORS.get(status, "white"))
        if block.get("group_by"):
            text.append(f"  by:{block['group_by']}", style="dim")
        if block.get("is_custom"):
            text.append("  AI", style="magenta")
        self.update(text)

    def on_focus(self) -> None:
        self.post_message(self.Selected(self.block_id))

    def on_click(self) -> None:
        self.focus()


class LaneRow(Vertical):
    """One lane: a title bar plus a horizontally-scrolling strip of chips."""

    DEFAULT_CSS = """
    LaneRow { height: auto; margin: 0 0 1 0; }
    LaneRow > .lane-title { height: 1; color: $text-muted; text-style: bold; }
    LaneRow > HorizontalScroll { height: 7; }
    """

    def __init__(self, lane_id: str, title: str) -> None:
        super().__init__(id=f"lane-{lane_id}")
        self.lane_id = lane_id
        self._title_widget = Static(title, classes="lane-title")
        self._strip = HorizontalScroll()

    def compose(self):
        yield self._title_widget
        yield self._strip

    def set_title(self, title: str) -> None:
        self._title_widget.update(title)

    async def set_chips(self, blocks: list[dict[str, Any]]) -> None:
        await self._strip.remove_children()
        await self._strip.mount_all([BlockChip(b) for b in blocks])


class WiresTable(DataTable):
    """Read-only-by-default list of wires: from.port -> to.port, flagged
    invalid (e.g. a type mismatch, or an endpoint since deleted) same as
    the web UI's wire coloring. Row selection is used by the app for
    delete-wire and for jumping to an endpoint block."""

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.add_columns("From", "Port", "→", "To", "Port", "OK")

    def set_wires(self, wires: dict[str, dict[str, Any]], blocks: dict[str, dict[str, Any]]) -> None:
        self.clear()
        self._wire_ids: list[str] = []
        for wid, w in wires.items():
            from_name = blocks.get(w["from_block"], {}).get("name", w["from_block"])
            to_name = blocks.get(w["to_block"], {}).get("name", w["to_block"])
            ok = "✓" if w.get("valid", True) else "✗"
            self.add_row(from_name, w["from_port"], "→", to_name, w["to_port"], ok, key=wid)
            self._wire_ids.append(wid)

    def wire_id_for_row(self, row_index: int) -> str | None:
        if 0 <= row_index < len(self._wire_ids):
            return self._wire_ids[row_index]
        return None


class Canvas(VerticalScroll):
    """Top-level graph view: one LaneRow per lane (plus a synthetic
    'Unassigned' lane for lane=None blocks), followed by the wires list."""

    DEFAULT_CSS = """
    Canvas { height: 1fr; }
    Canvas > .wires-title { height: 1; color: $text-muted; text-style: bold; margin: 1 0 0 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._lane_rows: dict[str, LaneRow] = {}
        self._wires_title = Static("Wires", classes="wires-title")
        self._wires_table = WiresTable()
        self._chip_order: list[str] = []  # block ids, in canvas reading order

    def compose(self):
        yield self._wires_title
        yield self._wires_table

    async def update_graph(self, graph: dict[str, Any]) -> None:
        lanes = graph.get("lanes", {})
        blocks: dict[str, dict[str, Any]] = graph.get("blocks", {})
        wires = graph.get("wires", {})

        def sort_key(b: dict[str, Any]):
            lane_order = lanes.get(b.get("lane") or "", {}).get("order", 10**9)
            pos = b.get("position", {}) or {}
            return (lane_order, pos.get("y", 0), pos.get("x", 0), b["id"])

        lane_ids = sorted(lanes, key=lambda lid: lanes[lid]["order"])
        by_lane: dict[str, list[dict[str, Any]]] = {lid: [] for lid in lane_ids}
        unassigned: list[dict[str, Any]] = []
        for b in sorted(blocks.values(), key=sort_key):
            if b.get("lane") and b["lane"] in by_lane:
                by_lane[b["lane"]].append(b)
            else:
                unassigned.append(b)

        rows_needed = list(lane_ids) + (["__unassigned__"] if unassigned else [])
        for lane_id in list(self._lane_rows):
            if lane_id not in rows_needed:
                await self._lane_rows.pop(lane_id).remove()

        self._chip_order = []
        for lane_id in rows_needed:
            title = "Unassigned" if lane_id == "__unassigned__" else lanes[lane_id]["name"]
            row = self._lane_rows.get(lane_id)
            if row is None:
                row = LaneRow(lane_id, title)
                self._lane_rows[lane_id] = row
                await self.mount(row, before=self._wires_title)
            else:
                row.set_title(title)
            row_blocks = unassigned if lane_id == "__unassigned__" else by_lane[lane_id]
            await row.set_chips(row_blocks)
            self._chip_order.extend(b["id"] for b in row_blocks)

        self._wires_table.set_wires(wires, blocks)

    def chip(self, block_id: str) -> BlockChip | None:
        try:
            return self.query_one(f"#chip-{block_id}", BlockChip)
        except Exception:
            return None

    def focus_block(self, block_id: str) -> None:
        chip = self.chip(block_id)
        if chip is not None:
            chip.focus()

    def focus_adjacent(self, block_id: str, delta: int) -> None:
        """Move focus to the block delta positions away in canvas reading
        order (lane order, then position) -- used for Left/Right."""
        if block_id not in self._chip_order:
            return
        idx = self._chip_order.index(block_id)
        new_idx = max(0, min(len(self._chip_order) - 1, idx + delta))
        self.focus_block(self._chip_order[new_idx])
