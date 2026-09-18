from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from .canvas import BlockChip, Canvas, WiresTable
from .charts import render_chart, save_chart
from .client import APIError, ModelMakerClient
from .inspector import Inspector
from .preview import PreviewPane
from .screens import ColumnRolePickScreen, ConfirmScreen, FilterListScreen, TextInputScreen, TextViewScreen

# Same default as frontend/src/Toolbar.tsx's DEFAULT_SAMPLE_ROWS: small
# enough that a pipeline over millions of rows becomes interactive, large
# enough that a train/test split or a grouped metric still has something
# to work with.
DEFAULT_SAMPLE_ROWS = 1000

# How much of the upstream input a generate_image block's chart preview /
# export is built from -- a preview, like the web UI's own /preview calls,
# not the full dataset (sample mode aside, the point is a representative
# chart, not exhaustive data).
CHART_PREVIEW_ROWS = 5000


class StatusBar(Static):
    DEFAULT_CSS = "StatusBar { height: 1; padding: 0 1; background: $panel; }"

    def show(self, graph: dict[str, Any], wire_pending: tuple[str, str] | None) -> None:
        name = graph.get("project_name", "untitled")
        dirty = "*" if graph.get("dirty") else ""
        sample = f"   sample:{graph['sample_rows']}" if graph.get("sample_rows") else ""
        err = f"   [red]! {graph['run_error']}[/red]" if graph.get("run_error") else ""
        pending = f"   [yellow]wiring from {wire_pending[0]}.{wire_pending[1]} -- select target, w[/yellow]" if wire_pending else ""
        self.update(f"{name}{dirty}{sample}{err}{pending}")


class ModelMakerTUI(App):
    TITLE = "Model-Maker"

    CSS = """
    #main { height: 1fr; }
    #right { width: 62; border-left: solid $panel-lighten-2; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+s", "save_context", "Save"),
        Binding("ctrl+o", "open_project", "Open"),
        Binding("ctrl+z", "undo", "Undo"),
        Binding("ctrl+y", "redo", "Redo"),
        Binding("a", "add_block", "Add block"),
        Binding("delete", "delete_selected", "Delete"),
        Binding("r", "run_selected", "Run"),
        Binding("R", "run_to_here", "Run to here", show=False),
        Binding("g", "refresh_selected", "Refresh src"),
        Binding("ctrl+r", "run_all", "Run all"),
        Binding("ctrl+g", "refresh_all_sources", "Refresh all"),
        Binding("ctrl+t", "run_all_streaming", "Run all (stream)"),
        Binding("f", "toggle_fullscreen_preview", "Fullscreen preview"),
        Binding("x", "cancel_run", "Cancel run"),
        Binding("m", "toggle_sample", "Sample mode"),
        Binding("w", "wire_toggle", "Wire"),
        Binding("i", "export_output", "Export chart/image"),
        Binding("t", "tag_column", "Tag column"),
        Binding("c", "compile", "Compile"),
        Binding("ctrl+d", "draft_ai", "Draft w/ AI"),
        Binding("ctrl+f", "suggest_fix", "Suggest fix"),
        Binding("ctrl+a", "apply_draft", "Apply draft", show=False),
        Binding("escape", "discard_draft", "Discard draft", show=False),
    ]

    def __init__(self, base_url: str | None = None, initial_project: str | None = None):
        super().__init__()
        self.client = ModelMakerClient(base_url)
        self._initial_project = initial_project
        self.graph: dict[str, Any] = {"lanes": {}, "blocks": {}, "wires": {}}
        self.registry_cache: list[dict[str, Any]] = []
        self.selected_block_id: str | None = None
        self.wire_pending: tuple[str, str] | None = None
        self.pending_draft: dict[str, Any] | None = None
        self._did_initial_focus = False
        self.preview_fullscreen = False

        self.status = StatusBar()
        self.canvas = Canvas()
        self.inspector = Inspector()
        self.preview = PreviewPane()
        # Serializes refresh_all() calls (the 2s poll timer vs. an action's
        # own post-mutation refresh) so a slow poll that started before an
        # action can't finish after it and clobber self.graph with stale
        # pre-action state -- the lock guarantees whichever call runs second
        # fetches state that reflects everything before it.
        self._refresh_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield self.status
        with Horizontal(id="main"):
            yield self.canvas
            with Vertical(id="right"):
                yield self.inspector
                yield self.preview
        yield Footer()

    # ---- lifecycle -----------------------------------------------------

    async def on_mount(self) -> None:
        self.run_worker(self._startup(), exclusive=True)
        self.set_interval(2.0, self._poll)

    async def on_unmount(self) -> None:
        await self.client.aclose()

    async def _startup(self) -> None:
        try:
            self.registry_cache = await self.client.registry()
        except APIError as e:
            self.notify(f"could not load block registry: {e}", severity="error")
        await self._check_recovery()
        if self._initial_project:
            try:
                await self.client.load_project(self._initial_project)
            except APIError as e:
                self.notify(str(e), severity="error")
        await self.refresh_all()

    async def _check_recovery(self) -> None:
        try:
            info = await self.client.recovery_info()
        except APIError:
            return
        if info.get("recovery"):
            restore = await self.push_screen_wait(
                ConfirmScreen("A crash-recovery snapshot was found. Restore it?")
            )
            if restore:
                try:
                    await self.client.recover()
                    self.notify("recovered")
                except APIError as e:
                    self.notify(str(e), severity="error")
            else:
                # Declined -- don't ask again with the same stale snapshot
                # on every future startup.
                try:
                    await self.client.dismiss_recovery()
                except APIError as e:
                    self.notify(str(e), severity="error")

    def _poll(self) -> None:
        self.run_worker(self.refresh_all(), exclusive=True, group="poll")

    # ---- refresh ---------------------------------------------------------

    async def refresh_all(self) -> None:
        async with self._refresh_lock:
            try:
                self.graph = await self.client.get_graph()
            except APIError as e:
                self.notify(str(e), severity="error")
                return
            await self.canvas.update_graph(self.graph)
            if self.selected_block_id not in self.graph["blocks"]:
                self.selected_block_id = None
            self.status.show(self.graph, self.wire_pending)
            await self.refresh_inspector()
            await self.refresh_preview()
            if not self._did_initial_focus and self.canvas._chip_order:
                self._did_initial_focus = True
                first_id = self.canvas._chip_order[0]
                self.selected_block_id = first_id
                self.canvas.focus_block(first_id)
                await self.refresh_inspector()
                await self.refresh_preview()

    async def refresh_inspector(self) -> None:
        block = self.graph["blocks"].get(self.selected_block_id) if self.selected_block_id else None
        columns: list[str] = []
        if block is not None:
            try:
                schema = await self.client.input_schema(block["id"])
                columns = sorted({c["name"] for cols in schema.values() for c in cols})
            except APIError:
                columns = []
        await self.inspector.show(block, columns)

    async def refresh_preview(self) -> None:
        block = self.graph["blocks"].get(self.selected_block_id) if self.selected_block_id else None
        if block is None:
            self.preview.show_message("(select a block)")
            return
        status = block.get("status")
        if status == "grey":
            self.preview.show_message("not run yet")
            return
        if status == "red":
            self.preview.show_message(f"error: {block.get('last_error') or '(unknown)'}")
            return
        outputs = block.get("outputs") or []
        if not outputs:
            self.preview.show_message("(no output ports)")
            return
        port = outputs[0]
        try:
            if port["type"] == "dataframe":
                data = await self.client.preview(block["id"], port=port["name"], rows=200)
                self.preview.show_dataframe(data["columns"], data["rows"], data["row_count"])
            elif port["type"] == "image":
                if block["category"] == "generate_image":
                    await self._preview_chart(block)
                else:
                    self.preview.show_message('image output -- press "i" to export the PNG')
            else:
                value = await self.client.value(block["id"], port=port["name"])
                self.preview.show_value(value)
        except APIError as e:
            self.preview.show_message(f"preview error: {e}")

    def _upstream_df_wire(self, block_id: str, port: str = "df") -> dict[str, Any] | None:
        return next(
            (w for w in self.graph["wires"].values() if w["to_block"] == block_id and w["to_port"] == port), None
        )

    async def _fetch_chart_df(self, block: dict[str, Any]):
        import polars as pl

        wire = self._upstream_df_wire(block["id"])
        if wire is None:
            return None
        data = await self.client.preview(wire["from_block"], port=wire["from_port"], rows=CHART_PREVIEW_ROWS)
        if data["rows"]:
            return pl.DataFrame(data["rows"])
        return pl.DataFrame({c["name"]: [] for c in data["columns"]})

    async def _preview_chart(self, block: dict[str, Any]) -> None:
        df = await self._fetch_chart_df(block)
        if df is None:
            self.preview.show_message("no upstream data wired in")
            return
        params = block.get("params") or {}
        try:
            width, height = self.preview.size.width - 4 or 60, max(self.preview.size.height - 2, 10)
            ansi = render_chart(
                df,
                kind=params.get("kind", "hist"),
                x=params.get("x", ""),
                y=params.get("y", ""),
                bins=params.get("bins", 30),
                title=params.get("title", ""),
                width=width,
                height=height,
            )
            self.preview.show_chart(ansi)
        except Exception as e:  # noqa: BLE001 -- surfaced to the user, not fatal
            self.preview.show_message(f"chart error: {e}")

    def action_toggle_fullscreen_preview(self) -> None:
        """The preview panel is normally squeezed into a 62-column sidebar
        under the inspector -- too narrow to actually read a wide table.
        This hides the canvas and inspector so the preview can take the
        whole screen, and puts it back exactly as it was on toggle-off."""
        self.preview_fullscreen = not self.preview_fullscreen
        self.canvas.display = not self.preview_fullscreen
        self.inspector.display = not self.preview_fullscreen
        right = self.query_one("#right")
        if self.preview_fullscreen:
            right.styles.width = "1fr"
            self.preview.focus_active()
        else:
            right.styles.width = None
            if self.selected_block_id:
                self.canvas.focus_block(self.selected_block_id)

    # ---- selection ---------------------------------------------------------

    def on_block_chip_selected(self, event: BlockChip.Selected) -> None:
        self.selected_block_id = event.block_id
        self.canvas.highlight_neighbors(event.block_id)
        self.run_worker(self.refresh_inspector(), exclusive=True, group="inspector")
        self.run_worker(self.refresh_preview(), exclusive=True, group="preview")

    def _focus_in_inspector(self) -> bool:
        node = self.focused
        while node is not None:
            if node is self.inspector:
                return True
            node = node.parent
        return False

    # ---- save / load ---------------------------------------------------------

    @work(exclusive=True)
    async def action_save_context(self) -> None:
        if self._focus_in_inspector():
            await self._commit_block_edits()
        else:
            await self._save_project()

    async def _commit_block_edits(self) -> None:
        if self.selected_block_id is None:
            return
        params = self.inspector.params.collect()
        if params is None:
            self.notify("params JSON is invalid -- not saved", severity="error")
            return
        fields: dict[str, Any] = {"params": params}
        code = self.inspector.code.collect_code()
        if code is not None:
            fields["code"] = code
        try:
            await self.client.update_block(self.selected_block_id, **fields)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self.notify("block saved")
        await self.refresh_all()

    async def _save_project(self) -> None:
        if self.graph.get("project_path"):
            try:
                await self.client.save_project(None)
            except APIError as e:
                self.notify(str(e), severity="error")
                return
            self.notify("project saved")
            await self.refresh_all()
            return
        path = await self.push_screen_wait(TextInputScreen("Save project as", placeholder="projects/my_model"))
        if not path:
            return
        try:
            await self.client.save_project(path)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self.notify("project saved")
        await self.refresh_all()

    @work(exclusive=True)
    async def action_open_project(self) -> None:
        path = await self.push_screen_wait(TextInputScreen("Load project", placeholder="projects/demo_pd_model"))
        if not path:
            return
        try:
            await self.client.load_project(path)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self.selected_block_id = None
        self._did_initial_focus = False
        await self.refresh_all()
        self.notify("project loaded")

    # ---- undo/redo -----------------------------------------------------------

    async def action_undo(self) -> None:
        try:
            await self.client.undo()
        except APIError as e:
            self.notify(str(e), severity="warning")
            return
        await self.refresh_all()

    async def action_redo(self) -> None:
        try:
            await self.client.redo()
        except APIError as e:
            self.notify(str(e), severity="warning")
            return
        await self.refresh_all()

    # ---- blocks / wires -----------------------------------------------------------

    @work(exclusive=True)
    async def action_add_block(self) -> None:
        if not self.registry_cache:
            self.notify("block registry not loaded yet", severity="warning")
            return
        options = [(f"{s['display_name']}  [{s['group']}]", s["category"]) for s in self.registry_cache]
        category = await self.push_screen_wait(FilterListScreen("Add block", options))
        if not category:
            return
        lane_id = None
        if self.selected_block_id:
            lane_id = self.graph["blocks"].get(self.selected_block_id, {}).get("lane")
        if lane_id is None and self.graph["lanes"]:
            lane_id = min(self.graph["lanes"], key=lambda lid: self.graph["lanes"][lid]["order"])
        try:
            block = await self.client.create_block(category=category, lane=lane_id)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self.selected_block_id = block["id"]
        await self.refresh_all()
        self.canvas.focus_block(block["id"])

    @work(exclusive=True)
    async def action_delete_selected(self) -> None:
        focused = self.focused
        if isinstance(focused, BlockChip):
            block = self.graph["blocks"].get(focused.block_id)
            name = block.get("name") if block else focused.block_id
            confirmed = await self.push_screen_wait(
                ConfirmScreen(f"Delete block '{name}'? (Ctrl+Z undoes this)")
            )
            if not confirmed:
                return
            try:
                await self.client.delete_block(focused.block_id)
            except APIError as e:
                self.notify(str(e), severity="error")
                return
            if self.selected_block_id == focused.block_id:
                self.selected_block_id = None
            await self.refresh_all()
        elif isinstance(focused, WiresTable):
            row = focused.cursor_row
            wire_id = focused.wire_id_for_row(row) if row is not None else None
            if wire_id is None:
                return
            confirmed = await self.push_screen_wait(ConfirmScreen("Delete this wire? (Ctrl+Z undoes this)"))
            if not confirmed:
                return
            try:
                await self.client.delete_wire(wire_id)
            except APIError as e:
                self.notify(str(e), severity="error")
                return
            await self.refresh_all()

    @work(exclusive=True)
    async def action_wire_toggle(self) -> None:
        focused = self.focused
        if not isinstance(focused, BlockChip):
            self.notify("select a block first (Tab / arrow keys)", severity="warning")
            return
        block = self.graph["blocks"].get(focused.block_id)
        if block is None:
            return
        if self.wire_pending is None:
            outputs = block.get("outputs") or []
            if not outputs:
                self.notify("block has no output ports", severity="warning")
                return
            port = outputs[0]["name"]
            if len(outputs) > 1:
                picked = await self.push_screen_wait(
                    FilterListScreen("Output port", [(f"{p['name']} ({p['type']})", p["name"]) for p in outputs])
                )
                if not picked:
                    return
                port = picked
            self.wire_pending = (block["id"], port)
            self.status.show(self.graph, self.wire_pending)
        else:
            from_block, from_port = self.wire_pending
            self.wire_pending = None
            inputs = block.get("inputs") or []
            if not inputs:
                self.notify("block has no input ports", severity="warning")
                self.status.show(self.graph, self.wire_pending)
                return
            port = inputs[0]["name"]
            if len(inputs) > 1:
                picked = await self.push_screen_wait(
                    FilterListScreen("Input port", [(f"{p['name']} ({p['type']})", p["name"]) for p in inputs])
                )
                if not picked:
                    self.status.show(self.graph, self.wire_pending)
                    return
                port = picked
            try:
                await self.client.create_wire(from_block, from_port, block["id"], port)
            except APIError as e:
                self.notify(str(e), severity="error")
                self.status.show(self.graph, self.wire_pending)
                return
            await self.refresh_all()

    @work(exclusive=True)
    async def action_tag_column(self) -> None:
        if self.selected_block_id is None:
            return
        try:
            schema = await self.client.input_schema(self.selected_block_id)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        columns = sorted({c["name"] for cols in schema.values() for c in cols})
        if not columns:
            self.notify("no input columns available yet", severity="warning")
            return
        result = await self.push_screen_wait(ColumnRolePickScreen(columns))
        if result is None:
            return
        column, role = result
        try:
            await self.client.set_column_role(self.selected_block_id, column, role)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    # ---- run control -------------------------------------------------------

    async def action_run_selected(self) -> None:
        if self.selected_block_id is None:
            return
        try:
            await self.client.run_block(self.selected_block_id)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_run_to_here(self) -> None:
        if self.selected_block_id is None:
            return
        try:
            await self.client.run_to_here(self.selected_block_id)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_refresh_selected(self) -> None:
        if self.selected_block_id is None:
            return
        try:
            await self.client.refresh(self.selected_block_id)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_run_all(self) -> None:
        try:
            await self.client.run_all()
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_refresh_all_sources(self) -> None:
        try:
            await self.client.refresh_all()
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_run_all_streaming(self) -> None:
        try:
            await self.client.run_all_streaming()
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_cancel_run(self) -> None:
        try:
            await self.client.cancel_run()
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    async def action_toggle_sample(self) -> None:
        rows = None if self.graph.get("sample_rows") is not None else DEFAULT_SAMPLE_ROWS
        try:
            await self.client.set_sample_mode(rows)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        await self.refresh_all()

    # ---- compile -----------------------------------------------------------

    @work(exclusive=True)
    async def action_compile(self) -> None:
        try:
            result = await self.client.compile()
        except APIError as e:
            self.notify(str(e), severity="error")
            return

        async def _save(text: str) -> None:
            path = await self.push_screen_wait(TextInputScreen("Save script as", placeholder="pipeline.py"))
            if path:
                Path(path).write_text(text)
                self.notify(f"saved {path}")

        await self.push_screen_wait(TextViewScreen("Compiled script", result["source"], language="python", on_save=_save))

    # ---- chart / image export -----------------------------------------------------------

    @work(exclusive=True)
    async def action_export_output(self) -> None:
        if self.selected_block_id is None:
            return
        block = self.graph["blocks"].get(self.selected_block_id)
        if not block or not block.get("outputs"):
            self.notify("nothing to export here", severity="warning")
            return
        port = block["outputs"][0]
        if port["type"] != "image":
            self.notify("selected block's output isn't an image", severity="warning")
            return
        if block["category"] == "generate_image":
            df = await self._fetch_chart_df(block)
            if df is None:
                self.notify("no upstream data wired in", severity="warning")
                return
            path = await self.push_screen_wait(TextInputScreen("Save chart as (.txt/.html)", placeholder=f"{block['name']}.txt"))
            if not path:
                return
            params = block.get("params") or {}
            try:
                save_chart(
                    df,
                    path,
                    kind=params.get("kind", "hist"),
                    x=params.get("x", ""),
                    y=params.get("y", ""),
                    bins=params.get("bins", 30),
                    title=params.get("title", ""),
                )
            except Exception as e:  # noqa: BLE001
                self.notify(f"save failed: {e}", severity="error")
                return
            self.notify(f"chart saved to {path}")
        else:
            try:
                png = await self.client.image(block["id"])
            except APIError as e:
                self.notify(str(e), severity="error")
                return
            path = await self.push_screen_wait(TextInputScreen("Save image as", placeholder=f"{block['name']}.png"))
            if not path:
                return
            Path(path).write_bytes(png)
            self.notify(f"image saved to {path}")

    # ---- AI drafting -----------------------------------------------------------

    async def action_draft_ai(self) -> None:
        if self.selected_block_id is None:
            return
        instruction = self.inspector.ai.instruction.value.strip()
        if not instruction:
            self.inspector.ai.instruction.focus()
            self.notify("type an instruction first", severity="warning")
            return
        try:
            result = await self.client.draft(self.selected_block_id, instruction)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self._show_draft(result)

    async def action_suggest_fix(self) -> None:
        if self.selected_block_id is None:
            return
        block = self.graph["blocks"].get(self.selected_block_id)
        if not block or not block.get("last_error"):
            self.notify("block has no recorded error", severity="warning")
            return
        instruction = self.inspector.ai.instruction.value.strip()
        try:
            result = await self.client.suggest_fix(self.selected_block_id, instruction)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self._show_draft(result)

    def _show_draft(self, result: dict[str, Any]) -> None:
        self.pending_draft = {"block_id": self.selected_block_id, **result}
        self.inspector.ai.show_proposal(result.get("code"), result.get("params"), result.get("explanation"))
        self.notify("proposal ready -- Ctrl+A apply, Esc discard")

    async def action_apply_draft(self) -> None:
        if self.pending_draft is None:
            return
        draft = self.pending_draft
        self.pending_draft = None
        self.inspector.ai.clear_proposal()
        fields: dict[str, Any] = {}
        if draft.get("code") is not None:
            fields["code"] = draft["code"]
        if draft.get("metadata_transform") is not None:
            fields["metadata_transform"] = draft["metadata_transform"]
        if draft.get("params") is not None:
            fields["params"] = draft["params"]
        try:
            await self.client.update_block(draft["block_id"], **fields)
        except APIError as e:
            self.notify(str(e), severity="error")
            return
        self.notify("draft applied")
        await self.refresh_all()

    def action_discard_draft(self) -> None:
        if self.pending_draft is not None:
            self.pending_draft = None
            self.inspector.ai.clear_proposal()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.15)
    return False


def main() -> None:
    """Entry point for the modelmaker-tui console script. Standalone mode
    (no --host) spawns a real modelmaker-api subprocess on a loopback port
    and attaches to it over HTTP, rather than importing the FastAPI app
    in-process -- block execution spawns its own multiprocessing
    subprocesses (see runner.py's MP_CONTEXT), and running that inside the
    same process as the Textual app risks fd-table conflicts between
    Textual's terminal driver and multiprocessing's spawn bookkeeping.
    A real, separate modelmaker-api process is the same isolation model
    the web UI already relies on, just pointed at by this process instead
    of a browser."""
    parser = argparse.ArgumentParser(prog="modelmaker-tui", description="Terminal UI for Model-Maker.")
    parser.add_argument("project", nargs="?", help="project folder to load on startup")
    parser.add_argument("--host", help="attach to an already-running modelmaker-api instead of spawning one")
    parser.add_argument("--port", type=int, default=8001, help="port for --host, or for the spawned server")
    args = parser.parse_args()

    server_process: subprocess.Popen | None = None
    if args.host:
        base_url = f"http://{args.host}:{args.port}"
    else:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "MODELMAKER_HOST": "127.0.0.1", "MODELMAKER_PORT": str(port)}
        server_process = subprocess.Popen(
            [sys.executable, "-m", "modelmaker.api"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not _wait_for_health(base_url):
            server_process.terminate()
            raise SystemExit("modelmaker-api did not start in time")

    try:
        ModelMakerTUI(base_url=base_url, initial_project=args.project).run()
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    main()
