from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.binding import Binding
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
# Selection-relative highlight tints -- distinct hues from STATUS_COLORS
# (and from each other) so "this is the selected block" vs. "this feeds
# it" vs. "it feeds this" read as three different things at a glance.
HIGHLIGHT_COLORS = {
    "selected": "#a855f7 35%",   # violet
    "upstream": "#06b6d4 30%",   # cyan -- blocks feeding the selected block
    "downstream": "#3b82f6 30%",  # blue -- blocks the selected block feeds
}


class BlockChip(Static, can_focus=True):
    """One block, rendered as a small bordered box colored by run status.
    Focusable (keyboard-navigable via its own arrow-key bindings, or plain
    Tab) and clickable; posts Selected when it gains focus."""

    # Bound here rather than on the App: Textual resolves a key against the
    # focused widget's own bindings before any ancestor's, but the chip
    # sits inside two nested scrollable containers (its lane's
    # HorizontalScroll, and Canvas itself, a VerticalScroll) that both bind
    # plain arrow keys to scrolling. An App-level binding is checked last,
    # so once either container actually has something to scroll (i.e. as
    # soon as a real terminal is short enough for the canvas to overflow),
    # it swallows the key first -- pressing Up/Down scrolls instead of
    # moving the selection. Binding on the chip itself always wins.
    BINDINGS = [
        Binding("left", "nav_left", "Prev block", show=False),
        Binding("right", "nav_right", "Next block", show=False),
        Binding("up", "nav_up", "Prev lane", show=False),
        Binding("down", "nav_down", "Next lane", show=False),
    ]

    DEFAULT_CSS = """
    BlockChip {
        width: 26;
        height: 6;
        margin: 0 1 0 0;
        padding: 0 1;
        border: round #9ca3af;
        content-align: left top;
        /* A long name/category/group_by must not wrap onto a second line --
        the chip has a fixed height, and a wrapped line pushes everything
        below it (including the in:/out: badge) out of the box entirely. */
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    class Selected(Message):
        def __init__(self, block_id: str) -> None:
            self.block_id = block_id
            super().__init__()

    def __init__(self, block: dict[str, Any], wire_counts: tuple[int, int] = (0, 0)) -> None:
        super().__init__(id=f"chip-{block['id']}")
        self.block_id: str = block["id"]
        self._border_color: str = STATUS_COLORS["grey"]
        self._highlight_role: str | None = None
        self.set_block(block, wire_counts)

    def set_highlight(self, role: str | None) -> None:
        """Tint the background by this chip's relationship to the currently
        selected block ('selected', 'upstream', 'downstream', or None) --
        the badges say *how many* wires a block has, this says *which other
        chip, and in which direction*, without routing lines."""
        if role == self._highlight_role:
            return
        self._highlight_role = role
        self.styles.background = HIGHLIGHT_COLORS.get(role) if role else None

    def _apply_border(self, focused: bool | None = None) -> None:
        # Inline styles (set here) take precedence over CSS, including
        # `:focus` rules -- so the focus highlight has to be applied by
        # hand rather than via a `BlockChip:focus` stylesheet rule. Also
        # don't default to reading `self.has_focus` from inside the
        # on_focus/on_blur handlers below: Textual dispatches a subclass's
        # `on_focus` *before* Widget's own internal `_on_focus`, which is
        # what actually flips `has_focus` to True, so it would still read
        # the stale pre-focus value at that point.
        if focused is None:
            focused = self.has_focus
        edge = "heavy" if focused else "round"
        self.styles.border = (edge, self._border_color)

    def set_block(self, block: dict[str, Any], wire_counts: tuple[int, int] = (0, 0)) -> None:
        self.block = block
        status = block.get("status", "grey")
        self._border_color = STATUS_COLORS.get(status, STATUS_COLORS["grey"])
        self._apply_border()
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
        inputs = block.get("inputs") or []
        outputs = block.get("outputs") or []
        if inputs or outputs:
            in_count, out_count = wire_counts
            text.append("\n")
            if inputs:
                # Red flags an input port this block declares but nothing
                # feeds -- almost always means the block can't run yet.
                in_style = STATUS_COLORS["green"] if in_count else STATUS_COLORS["red"]
                text.append(f"in:{in_count}", style=in_style)
            if inputs and outputs:
                text.append("  ")
            if outputs:
                # An unconsumed output isn't necessarily wrong (e.g. a
                # terminal metric block), so it's dim rather than red.
                out_style = STATUS_COLORS["green"] if out_count else "dim"
                text.append(f"out:{out_count}", style=out_style)
        self.update(text)

    def on_focus(self) -> None:
        self._apply_border(True)
        self.post_message(self.Selected(self.block_id))

    def on_blur(self) -> None:
        self._apply_border(False)

    def on_click(self) -> None:
        self.focus()

    def action_nav_left(self) -> None:
        self.screen.query_one(Canvas).focus_adjacent(self.block_id, -1)

    def action_nav_right(self) -> None:
        self.screen.query_one(Canvas).focus_adjacent(self.block_id, 1)

    def action_nav_up(self) -> None:
        self.screen.query_one(Canvas).focus_adjacent_lane(self.block_id, -1)

    def action_nav_down(self) -> None:
        self.screen.query_one(Canvas).focus_adjacent_lane(self.block_id, 1)


class LaneRow(Vertical):
    """One lane: a title bar plus a horizontally-scrolling strip of chips."""

    DEFAULT_CSS = """
    LaneRow { height: auto; margin: 0 0 1 0; }
    LaneRow > .lane-title { height: 1; color: $text-muted; text-style: bold; }
    LaneRow > HorizontalScroll { height: 8; }
    """

    def __init__(self, lane_id: str, title: str) -> None:
        super().__init__(id=f"lane-{lane_id}")
        self.lane_id = lane_id
        self._title_widget = Static(title, classes="lane-title")
        self._strip = HorizontalScroll()
        self._chips: dict[str, BlockChip] = {}

    def compose(self):
        yield self._title_widget
        yield self._strip

    def set_title(self, title: str) -> None:
        self._title_widget.update(title)

    async def set_chips(
        self, blocks: list[dict[str, Any]], wire_counts: dict[str, tuple[int, int]] | None = None
    ) -> None:
        # Update existing chips in place and only mount/remove what actually
        # changed -- remounting every chip on every refresh (this runs on
        # each 2s poll tick, even with unchanged data) would destroy
        # whatever chip currently has focus, kicking keyboard focus up to
        # this HorizontalScroll and silently breaking navigation/highlight.
        wire_counts = wire_counts or {}
        wanted_ids = [b["id"] for b in blocks]
        wanted_set = set(wanted_ids)
        for block_id in list(self._chips):
            if block_id not in wanted_set:
                await self._chips.pop(block_id).remove()
        blocks_by_id = {b["id"]: b for b in blocks}
        for block_id in wanted_ids:
            chip = self._chips.get(block_id)
            counts = wire_counts.get(block_id, (0, 0))
            if chip is None:
                chip = BlockChip(blocks_by_id[block_id], counts)
                self._chips[block_id] = chip
                await self._strip.mount(chip)
            else:
                chip.set_block(blocks_by_id[block_id], counts)
        ordered = [self._chips[block_id] for block_id in wanted_ids]
        if list(self._strip.children) != ordered:
            for index, chip in enumerate(ordered):
                self._strip.move_child(chip, before=index)


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
        self._chip_rows: list[list[str]] = []  # block ids, grouped by lane row
        self._upstream: dict[str, set[str]] = {}  # block id -> blocks feeding it
        self._downstream: dict[str, set[str]] = {}  # block id -> blocks it feeds
        self._highlighted: str | None = None

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

        wire_counts: dict[str, tuple[int, int]] = {}
        upstream: dict[str, set[str]] = {}
        downstream: dict[str, set[str]] = {}
        for w in wires.values():
            in_c, out_c = wire_counts.get(w["to_block"], (0, 0))
            wire_counts[w["to_block"]] = (in_c + 1, out_c)
            in_c, out_c = wire_counts.get(w["from_block"], (0, 0))
            wire_counts[w["from_block"]] = (in_c, out_c + 1)
            downstream.setdefault(w["from_block"], set()).add(w["to_block"])
            upstream.setdefault(w["to_block"], set()).add(w["from_block"])
        self._upstream = upstream
        self._downstream = downstream

        self._chip_order = []
        self._chip_rows = []
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
            await row.set_chips(row_blocks, wire_counts)
            row_ids = [b["id"] for b in row_blocks]
            self._chip_order.extend(row_ids)
            if row_ids:
                self._chip_rows.append(row_ids)

        self._wires_table.set_wires(wires, blocks)
        # Re-apply on every refresh, not just on selection change: a chip
        # rebuilt by set_chips() (new block, or the JSON-fallback churn
        # elsewhere) starts unhighlighted regardless of prior state.
        self.highlight_neighbors(self._highlighted)

    def highlight_neighbors(self, block_id: str | None) -> None:
        """Tint the selected chip and everything directly wired to it --
        feeding blocks (upstream) in a different color than blocks it feeds
        (downstream) -- so a connection, and its direction, is visible on
        the canvas itself without hunting through the wires list below."""
        self._highlighted = block_id
        upstream_ids = self._upstream.get(block_id, set()) if block_id else set()
        downstream_ids = self._downstream.get(block_id, set()) if block_id else set()
        for chip_id in self._chip_order:
            chip = self.chip(chip_id)
            if chip is None:
                continue
            if chip_id == block_id:
                chip.set_highlight("selected")
            elif chip_id in upstream_ids:
                chip.set_highlight("upstream")
            elif chip_id in downstream_ids:
                chip.set_highlight("downstream")
            else:
                chip.set_highlight(None)

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

    def focus_adjacent_lane(self, block_id: str, delta: int) -> None:
        """Move focus to the block in the lane row delta away, keeping
        (clamped) horizontal position -- used for Up/Down."""
        for row_idx, row in enumerate(self._chip_rows):
            if block_id not in row:
                continue
            new_row_idx = row_idx + delta
            if not (0 <= new_row_idx < len(self._chip_rows)):
                return
            col = row.index(block_id)
            new_row = self._chip_rows[new_row_idx]
            self.focus_block(new_row[min(col, len(new_row) - 1)])
            return
