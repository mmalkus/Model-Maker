from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from modelmaker.tui.app import ModelMakerTUI
from modelmaker.tui.canvas import BlockChip

# Integration test: runs the TUI App against a *real* modelmaker-api
# subprocess, the same way modelmaker-tui's standalone launch does (see
# app.main) -- not the in-process ASGI transport. Block execution spawns
# its own multiprocessing subprocess (runner.py's MP_CONTEXT), and doing
# that inside the same process as the Textual app risks fd-table conflicts
# with Textual's driver (observed as `ValueError: bad value(s) in
# fds_to_keep` under the headless test driver) -- a real, separate server
# process is what main() actually does, so it's what this test exercises.


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


@pytest.fixture
def server_base_url(tmp_path):
    # Function-scoped (a fresh server per test) rather than shared: the
    # server's SESSION carries state (background run threads, the last-run
    # error slot) across requests, and a module-scoped server let one
    # test's run leak into the next's timing -- a clean process per test
    # is slightly slower but fully deterministic.
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "MODELMAKER_HOST": "127.0.0.1",
        "MODELMAKER_PORT": str(port),
        "MODELMAKER_LLM_PROVIDER": "stub",
        # Its own cache per server: a recovery snapshot left behind by an
        # earlier test (or another process) would otherwise make this
        # server's startup stop on an unanswered "restore snapshot?" prompt.
        "MODELMAKER_CACHE_DIR": str(tmp_path / "modelmaker-cache"),
    }
    # The server's own log goes to a file, echoed at teardown -- pytest only
    # shows it for a failed test, where a 500 or crash is otherwise invisible
    # (the client just sees "Server disconnected").
    log_path = tmp_path / "modelmaker-api.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen([sys.executable, "-m", "modelmaker.api"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not _wait_for_health(base_url):
            proc.terminate()
            pytest.fail("modelmaker-api subprocess did not become healthy in time")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        print(log_path.read_text(encoding="utf-8", errors="replace"))


@pytest.fixture
def demo_project_path(tmp_path):
    # A private copy so the test never writes back into the tracked demo
    # project folder, however it drives the app (Ctrl+S, autosave-on-load,
    # ...). Save/Load operate on the whole folder, not just model.json.
    src = os.path.join(os.path.dirname(__file__), "..", "projects", "demo_pd_model")
    dest = tmp_path / "demo_pd_model"
    shutil.copytree(src, dest)
    return str(dest)


async def _load(app: ModelMakerTUI, pilot) -> None:
    # Wait for the whole first refresh, not just for app.graph to be set:
    # refresh_all assigns the graph first and only builds the canvas and
    # focuses the first block after. Asserting in between failed the test
    # and tore the app down under the still-mounting canvas (surfacing as
    # a misleading WidgetError from the startup worker).
    for _ in range(100):
        await pilot.pause(0.2)
        if app.graph.get("blocks") and app._did_initial_focus and not app._refresh_lock.locked():
            return
    pytest.fail("project never finished loading")


async def _wait_for_sources(app: ModelMakerTUI, pilot) -> None:
    # Ctrl+G's handler re-reads every source before it returns; wait for
    # that to show in the graph rather than sleeping a fixed time after it.
    await _wait_until(
        pilot,
        lambda: all(b["status"] == "green" for b in app.graph["blocks"].values() if b["block_type"] == "input"),
        "every source to be read",
        timeout=30.0,
    )


async def _wait_for_run_all(app: ModelMakerTUI, pilot, timeout: float = 60.0) -> None:
    # Run all works in the background; a fixed pause races it on a slow
    # machine, and the next run request then fails "already in progress".
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.3)
        await app.refresh_all()
        statuses = {b["status"] for b in app.graph["blocks"].values()}
        if "running" not in statuses and "grey" not in statuses:
            return
    pytest.fail(f"run all never finished: {statuses}")


async def _wait_until(pilot, predicate, what: str, timeout: float = 10.0) -> None:
    # Poll for the state an action leads to instead of a fixed pause: under
    # CI load a key press's handler (or the Selected message a focus change
    # posts) can take longer than any fixed pause, and asserting early is
    # what made these tests flaky.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.1)
    pytest.fail(f"timed out waiting for {what}")


def test_loads_project_and_focuses_first_block(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            assert len(app.graph["blocks"]) == 8
            assert app.selected_block_id == app.canvas._chip_order[0]
            assert isinstance(app.focused, BlockChip)

    asyncio.run(go())


def test_left_right_navigation_moves_selection(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            first = app.selected_block_id
            await pilot.press("right")
            await _wait_until(pilot, lambda: app.selected_block_id != first, "selection to move right")
            await pilot.press("left")
            await _wait_until(pilot, lambda: app.selected_block_id == first, "selection to move back left")

    asyncio.run(go())


def test_run_block_turns_it_green(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            assert app.selected_block_id == "b_load"
            await pilot.press("r")
            await _wait_until(
                pilot, lambda: app.graph["blocks"]["b_load"]["status"] == "green", "b_load to turn green", timeout=30.0
            )

    asyncio.run(go())


def test_run_all_after_refreshing_sources(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            await pilot.press("ctrl+g")  # refresh all sources
            await _wait_for_sources(app, pilot)
            await pilot.press("ctrl+r")  # run all
            await _wait_for_run_all(app, pilot)
            statuses = {b["status"] for b in app.graph["blocks"].values()}
            assert statuses == {"green"}

    asyncio.run(go())


def test_add_delete_and_undo_block(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            before = len(app.graph["blocks"])

            await pilot.press("a")
            await _wait_until(pilot, lambda: len(app.screen_stack) == 2, "the add-block picker")
            for ch in "filter":
                await pilot.press(ch)
            await pilot.pause(0.2)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(app.graph["blocks"]) == before + 1, "the block to be added")
            new_id = app.selected_block_id

            app.canvas.focus_block(new_id)
            await _wait_until(pilot, lambda: app.selected_block_id == new_id, "the new block to be selected")
            await pilot.press("delete")
            await _wait_until(pilot, lambda: len(app.screen_stack) == 2, "the delete confirmation")
            await pilot.press("y")  # the delete ConfirmScreen
            await _wait_until(pilot, lambda: len(app.graph["blocks"]) == before, "the block to be deleted")

            await pilot.press("ctrl+z")
            await _wait_until(pilot, lambda: len(app.graph["blocks"]) == before + 1, "undo to restore the block")

    asyncio.run(go())


def test_sample_mode_toggle(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            assert app.graph.get("sample_rows") is None
            await pilot.press("m")
            await _wait_until(pilot, lambda: app.graph.get("sample_rows") == 1000, "sample mode to turn on")
            await pilot.press("m")
            await _wait_until(pilot, lambda: app.graph.get("sample_rows") is None, "sample mode to turn off")

    asyncio.run(go())


def test_compile_opens_and_closes_script_view(server_base_url, demo_project_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            await pilot.press("ctrl+g")  # refresh sources
            await _wait_for_sources(app, pilot)
            await pilot.press("ctrl+r")  # run all -- strict compile needs every block to have actually run
            await _wait_for_run_all(app, pilot)
            assert all(b["status"] == "green" for b in app.graph["blocks"].values())

            await pilot.press("c")
            for _ in range(20):
                await pilot.pause(0.2)
                if len(app.screen_stack) == 2:
                    break
            assert len(app.screen_stack) == 2
            await pilot.press("escape")
            await pilot.pause(0.3)
            assert len(app.screen_stack) == 1

    asyncio.run(go())


def test_generate_image_chart_preview_and_export(server_base_url, demo_project_path, tmp_path):
    async def go():
        app = ModelMakerTUI(base_url=server_base_url, initial_project=demo_project_path)
        async with app.run_test(size=(160, 50)) as pilot:
            await _load(app, pilot)
            await pilot.press("ctrl+g")  # refresh sources
            await _wait_for_sources(app, pilot)
            await pilot.press("ctrl+r")  # run all -- b_clean needs a cached output to wire a chart from
            await _wait_for_run_all(app, pilot)
            assert app.graph["blocks"]["b_clean"]["status"] == "green"

            await pilot.press("a")
            await _wait_until(pilot, lambda: len(app.screen_stack) == 2, "the add-block picker")
            for ch in "generate image":
                await pilot.press(ch)
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.5)
            gi_id = app.selected_block_id
            assert app.graph["blocks"][gi_id]["category"] == "generate_image"

            app.canvas.focus_block("b_clean")
            await pilot.pause(0.2)
            await pilot.press("w")
            await pilot.pause(0.2)
            app.canvas.focus_block(gi_id)
            await pilot.pause(0.2)
            await pilot.press("w")
            await pilot.pause(0.5)
            assert any(w["to_block"] == gi_id for w in app.graph["wires"].values())

            await app.client.update_block(gi_id, params={"kind": "hist", "x": "credit_score"})
            await app.client.run_block(gi_id)
            await app.refresh_all()
            assert app.graph["blocks"][gi_id]["status"] == "green"

            app.canvas.focus_block(gi_id)
            await _wait_until(pilot, lambda: app.selected_block_id == gi_id, "the chart block to be selected")
            await app.refresh_preview()
            assert app.preview._chart.display is True, app.preview._message.content

            out_path = tmp_path / "chart.txt"
            df = await app._fetch_chart_df(app.graph["blocks"][gi_id])
            assert df is not None
            from modelmaker.tui.charts import save_chart

            saved = save_chart(df, out_path, kind="hist", x="credit_score")
            assert saved.exists()
            assert saved.read_text(encoding="utf-8").strip()

    asyncio.run(go())
