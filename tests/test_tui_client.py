from __future__ import annotations

import asyncio

import pytest

import modelmaker.api as api_module
from modelmaker.session import ProjectSession
from modelmaker.tui.client import APIError, ModelMakerClient


@pytest.fixture
def isolated_session(tmp_path):
    # Same trick as test_api.py's `client` fixture: point the module-level
    # SESSION at a fresh ProjectSession with its recovery snapshot under
    # tmp_path, so a test run never writes into the real working directory
    # and each test starts from a clean graph.
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    return api_module.SESSION


def run(coro):
    return asyncio.run(coro)


def test_health_and_registry(isolated_session):
    async def go():
        async with ModelMakerClient() as c:
            assert await c.health() == {"status": "ok"}
            categories = {b["category"] for b in await c.registry()}
            assert {"read_csv", "filter", "generate_image"} <= categories

    run(go())


def test_create_wire_run_and_preview(isolated_session, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")

    async def go():
        async with ModelMakerClient() as c:
            read = await c.create_block("read_csv", params={"path": str(csv_path)})
            filt = await c.create_block("filter", params={"expr": "a > 1"})
            wire = await c.create_wire(read["id"], "out", filt["id"], "df")
            assert wire["valid"] is True

            await c.refresh(read["id"])
            result = await c.run_block(filt["id"])
            assert result["status"] == "green"

            preview = await c.preview(filt["id"])
            assert preview["row_count"] == 2

    run(go())


def test_api_error_surfaces_status_and_detail(isolated_session):
    async def go():
        async with ModelMakerClient() as c:
            with pytest.raises(APIError) as exc_info:
                await c.run_block("does-not-exist")
            assert exc_info.value.status_code == 404
            assert "does-not-exist" in str(exc_info.value)

    run(go())


def test_undo_redo(isolated_session):
    async def go():
        async with ModelMakerClient() as c:
            block = await c.create_block("filter", params={"expr": "a > 1"})
            graph = await c.get_graph()
            assert block["id"] in graph["blocks"]

            await c.undo()
            graph = await c.get_graph()
            assert block["id"] not in graph["blocks"]

            await c.redo()
            graph = await c.get_graph()
            assert block["id"] in graph["blocks"]

    run(go())


def test_sample_mode_round_trip(isolated_session):
    async def go():
        async with ModelMakerClient() as c:
            graph = await c.set_sample_mode(500)
            assert graph["sample_rows"] == 500
            graph = await c.set_sample_mode(None)
            assert graph["sample_rows"] is None

    run(go())


def test_save_and_load_project(isolated_session, tmp_path):
    path = tmp_path / "proj"

    async def go():
        async with ModelMakerClient() as c:
            await c.create_block("filter", params={"expr": "a > 1"})
            saved = await c.save_project(str(path))
            assert saved["path"] == str(path)
            assert (path / "model.json").exists()

            await c.create_block("select", params={"cols": ["a"]})
            assert len((await c.get_graph())["blocks"]) == 2

            await c.load_project(str(path))
            assert len((await c.get_graph())["blocks"]) == 1

    run(go())


def test_compile_produces_python_source(isolated_session, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n")

    async def go():
        async with ModelMakerClient() as c:
            await c.create_block("read_csv", params={"path": str(csv_path)})
            result = await c.compile(strict=False)
            assert "import polars" in result["source"]

    run(go())
