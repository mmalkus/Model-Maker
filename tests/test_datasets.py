"""Datasets, sources, exports and the capability set (capabilities.py,
datasets.py): a workspace without `filesystem` doesn't offer the
path-based blocks anywhere and refuses them clearly; read_dataset,
read_source and export work through an embedding application's providers
(fakes here); and the local app is unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import modelmaker.api as api
from modelmaker import capabilities
from modelmaker.agent import catalogue
from modelmaker.block_library import parse_source, validate_definition
from modelmaker.blocks.base import BLOCK_REGISTRY, FIELD_KINDS
from modelmaker.datasets import DatasetError, DatasetInfo, ExportRequest, SourceInfo
from modelmaker.forms import block_form
from modelmaker.project import save_project
from modelmaker.session import ProjectSession
from modelmaker.workspace import Workspace

HEADER = "X-Test-Workspace"
PATH_BLOCKS = {"read_csv", "read_parquet", "read_json", "read_excel", "read_sql", "write_csv"}
DATA_BLOCKS = {"read_dataset", "read_source", "export"}


class FakeDatasets:
    """Datasets in a folder, content-addressed like a real store: the
    version is the content's hash, the file is named after it."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.names: dict[str, DatasetInfo] = {}

    def put(self, name: str, data: bytes, fmt: str = "csv", origin: str = "upload") -> DatasetInfo:
        sha = hashlib.sha256(data).hexdigest()
        path = self.folder / f"{sha}.{fmt}"
        path.write_bytes(data)
        info = DatasetInfo(name, f"sha256:{sha}", fmt, len(data), str(path), origin=origin)
        self.names[name] = info
        return info

    def list_datasets(self):
        return list(self.names.values())

    def get_dataset(self, name):
        return self.names.get(name)


class FakeSources:
    def __init__(self, datasets: FakeDatasets) -> None:
        self.datasets = datasets
        self.queries: list[tuple[str, str]] = []

    def list_sources(self):
        return [SourceInfo("dwh", "the warehouse")]

    def query_source(self, source, query):
        if source != "dwh":
            raise DatasetError(f"no source named {source!r}")
        self.queries.append((source, query))
        df = pl.DataFrame({"id": [1, 2, 3], "q": [query] * 3})
        path = self.datasets.folder / "result.parquet"
        df.write_parquet(path)
        return self.datasets.put(f"{source}-result", path.read_bytes(), "parquet", origin="source")


class FakeExports:
    def __init__(self) -> None:
        self.received: list[ExportRequest] = []

    def export(self, request):
        self.received.append(request)
        return {"name": request.name}


@pytest.fixture
def hosted(tmp_path):
    """Two workspaces: `local` as the local app has it, `server` without
    `filesystem` but with datasets, a source and exports."""
    (tmp_path / "ds").mkdir()
    datasets = FakeDatasets(tmp_path / "ds")
    sources = FakeSources(datasets)
    exports = FakeExports()
    local = Workspace("local", cache_dir=tmp_path / "c-local", projects_dir=tmp_path / "p-local", root=tmp_path)
    server = Workspace(
        "server",
        cache_dir=tmp_path / "c-server",
        projects_dir=tmp_path / "p-server",
        root=tmp_path,
        capabilities={"datasets", "sources", "exports"},
        datasets=datasets,
        sources=sources,
        exports=exports,
    )
    by_id = {"local": local, "server": server}

    def resolve(request):
        ws = by_id.get(request.headers.get(HEADER, ""))
        if ws is None:
            raise HTTPException(404, "no such workspace")
        return ws

    api.set_workspace_resolver(resolve)
    try:
        yield local, server, datasets, sources, exports
    finally:
        api.set_workspace_resolver(None)
        for ws in (local, server):
            ws.session.runner.cancel()


def _client(ws: Workspace) -> TestClient:
    return TestClient(api.app, headers={HEADER: ws.id})


def _run(session: ProjectSession, block_id: str) -> str:
    return session.runner.run_block(block_id)


# ---- the capability set -----------------------------------------------------


def test_capabilities_of_blocks():
    for bid in PATH_BLOCKS:
        assert capabilities.required(BLOCK_REGISTRY[bid]) == {"filesystem"}, bid
    assert capabilities.required(BLOCK_REGISTRY["read_dataset"]) == {"datasets"}
    assert capabilities.required(BLOCK_REGISTRY["read_source"]) == {"sources"}
    assert capabilities.required(BLOCK_REGISTRY["export"]) == {"exports"}
    assert capabilities.required(BLOCK_REGISTRY["filter"]) == frozenset()
    # Outside any request: the local app's set.
    assert capabilities.granted() == capabilities.LOCAL == {"filesystem"}
    msg = capabilities.refusal(BLOCK_REGISTRY["read_csv"], ["filesystem"])
    assert "Read dataset" in msg and "read_dataset" in msg and "files" in msg


def test_a_library_block_with_a_path_field_needs_filesystem():
    src = (
        '"""Mine."""\n\nimport polars as pl\n\nBLOCKS = {"mine": {"block_type": "input", "inputs": [], '
        '"outputs": [{"name": "out"}], "form": [{"key": "where", "kind": "path"}]}}\n\n\n'
        "def mine(where: str) -> pl.DataFrame:\n    return pl.read_csv(where)\n"
    )
    (d,) = parse_source(src, "mine.py").definitions
    assert capabilities.required(d) == {"filesystem"}
    assert not capabilities.allowed(d, {"datasets"})


def test_capabilities_key_is_validated():
    src = (
        '"""Mine."""\n\nimport polars as pl\n\nBLOCKS = {"mine": {"capabilities": ["root"], '
        '"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}}\n\n\ndef mine(df):\n    return df\n'
    )
    (d,) = parse_source(src, "mine.py").definitions
    assert any("capabilities" in e for e in validate_definition(d))


def test_local_app_is_unchanged(hosted):
    local = hosted[0]
    c = _client(local)
    names = {e["category"] for e in c.get("/api/registry").json()}
    assert PATH_BLOCKS <= names
    assert not names & DATA_BLOCKS
    assert c.get("/api/capabilities").json() == {"capabilities": ["filesystem"]}
    assert c.get("/api/datasets").json() == [] and c.get("/api/sources").json() == []
    r = c.post("/api/blocks", json={"category": "read_csv", "params": {"path": "x.csv"}})
    assert r.status_code == 200
    r = c.post("/api/blocks", json={"category": "read_dataset", "params": {"dataset": "loans"}})
    assert r.status_code == 400 and "this workspace has no datasets" in r.json()["detail"]


def test_default_workspace_is_the_local_app():
    plain = TestClient(api.app)
    names = {e["category"] for e in plain.get("/api/registry").json()}
    assert PATH_BLOCKS <= names and not names & DATA_BLOCKS
    assert api.DEFAULT_WORKSPACE.capabilities == {"filesystem"}
    assert plain.get("/api/datasets").json() == []


def test_palette_and_catalogues_hide_path_blocks(hosted):
    server = hosted[1]
    c = _client(server)
    names = {e["category"] for e in c.get("/api/registry").json()}
    assert not names & PATH_BLOCKS
    assert DATA_BLOCKS <= names
    builtin = {b["id"] for f in c.get("/api/library").json()["builtin_files"] for b in f["blocks"]}
    assert not builtin & PATH_BLOCKS and "read_dataset" in builtin
    assert c.get("/api/capabilities").json() == {"capabilities": ["datasets", "exports", "sources"]}
    # The AI builder's catalogue, as seen from this workspace.
    with capabilities.granting(server.capabilities):
        known = catalogue.available()
        assert not set(known) & PATH_BLOCKS
        assert catalogue.lookup("read_csv") is None
        listed = {e["category"] for e in catalogue.list_block_types(include_disallowed=True)}
        assert not listed & PATH_BLOCKS
        with pytest.raises(KeyError):
            catalogue.describe_block_type("write_csv")
    # ...and through the API's own resolver (the assistant's catalogue).
    from modelmaker.api import workspace_context

    with workspace_context(server):
        assert catalogue.lookup("read_csv") is None
        assert "read_csv" not in catalogue.available()
    with workspace_context(hosted[0]):
        assert catalogue.lookup("read_csv") is not None


def test_path_blocks_refused_with_the_dataset_equivalent(hosted, tmp_path):
    local, server = hosted[0], hosted[1]
    c = _client(server)
    r = c.post("/api/blocks", json={"category": "read_csv", "params": {"path": "/etc/passwd"}})
    assert r.status_code == 400
    assert "read_dataset" in r.json()["detail"] and "Read dataset" in r.json()["detail"]
    r = c.post("/api/blocks", json={"category": "read_sql", "params": {"connection_env": "X", "query": "select 1"}})
    assert r.status_code == 400 and "read_source" in r.json()["detail"]
    r = c.post("/api/blocks", json={"category": "write_csv", "params": {"filename": "x.csv"}})
    assert r.status_code == 400 and "'Export' (export)" in r.json()["detail"]
    assert server.session.graph.blocks == {}

    # A project using one: refused on load, naming the block and the alternative.
    session = ProjectSession(recovery_path=None)
    session.add_block("read_csv", name="Loans", params={"path": str(tmp_path / "loans.csv")})
    folder = tmp_path / "proj"
    save_project(session.graph, folder / "model.json", project_name="proj")
    r = c.post("/api/project/load", json={"path": str(folder)})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "block 'Loans'" in detail and "read_dataset" in detail
    assert server.session.graph.blocks == {} and server.session.project_path is None
    # The local app opens it as always.
    assert _client(local).post("/api/project/load", json={"path": str(folder)}).status_code == 200

    # Even a graph that got there some other way doesn't run, and isn't probed.
    block = session.graph.blocks[next(iter(session.graph.blocks))]
    server.session.graph.blocks[block.id] = block
    assert _run(server.session, block.id) == "red"
    assert "read_dataset" in server.session.runner.state[block.id].last_error
    assert server.session.runner.check_for_changes(block.id) is False


def test_recovery_snapshot_with_a_path_block_is_refused(tmp_path):
    from modelmaker.datasets import DataAccess

    local = ProjectSession(recovery_path=tmp_path / "recovery.json")
    with local.edit():
        local.add_block("read_csv", params={"path": "x.csv"})
    restricted = ProjectSession(recovery_path=tmp_path / "recovery.json", data=DataAccess(frozenset()))
    with pytest.raises(capabilities.CapabilityError, match="read_dataset"):
        restricted.recover()


# ---- the dataset field --------------------------------------------------------


def test_dataset_field_kind():
    assert {"dataset", "source"} <= FIELD_KINDS
    fields = {f["key"]: f for f in block_form(BLOCK_REGISTRY["read_dataset"])["fields"]}
    assert fields["dataset"]["kind"] == "dataset"
    fields = {f["key"]: f for f in block_form(BLOCK_REGISTRY["read_source"])["fields"]}
    assert fields["source"]["kind"] == "source" and fields["query"]["kind"] == "sql"
    # A library block may use the kind; it then needs `datasets`.
    src = (
        '"""Mine."""\n\nimport polars as pl\n\nBLOCKS = {"mine": {"block_type": "input", "inputs": [], '
        '"outputs": [{"name": "out"}], "form": [{"key": "ds", "kind": "dataset"}]}}\n\n\n'
        "def mine(ds: str) -> pl.DataFrame:\n    return pl.read_csv(ds)\n"
    )
    (d,) = parse_source(src, "mine.py").definitions
    assert validate_definition(d) == []
    assert capabilities.required(d) == {"datasets"}


def test_dataset_picker_lists_without_paths(hosted):
    server, datasets = hosted[1], hosted[2]
    datasets.put("loans", b"a,b\n1,2\n")
    out = _client(server).get("/api/datasets").json()
    assert [d["name"] for d in out] == ["loans"]
    assert out[0]["format"] == "csv" and out[0]["size"] == 8
    assert "path" not in out[0] and str(datasets.folder) not in json.dumps(out)
    assert _client(server).get("/api/sources").json() == [{"name": "dwh", "description": "the warehouse"}]


# ---- read_dataset through a provider -------------------------------------------


def test_read_dataset_reads_and_follows_the_version(hosted):
    server, datasets = hosted[1], hosted[2]
    datasets.put("loans", b"id,amount\n1,10\n2,20\n")
    s = server.session
    reader = s.add_block("read_dataset", params={"dataset": "loans"})
    key1 = s.runner.compute_key(reader.id)
    assert _run(s, reader.id) == "green", s.runner.state[reader.id].last_error
    df = s.runner.cache.get(s.runner.state[reader.id].last_successful_key).outputs["out"].data
    assert df.to_dict(as_series=False) == {"id": [1, 2], "amount": [10, 20]}

    # Replaced: a new version, a new key, stale with the reason.
    datasets.put("loans", b"id,amount\n1,10\n2,20\n3,30\n")
    assert s.runner.compute_key(reader.id) != key1
    assert s.runner.status(reader.id) == "orange"
    assert "dataset was replaced" in s.runner.stale_reason(reader.id)
    assert s.runner.check_for_changes(reader.id) is False  # first probe: a baseline
    assert _run(s, reader.id) == "green"
    df = s.runner.cache.get(s.runner.state[reader.id].last_successful_key).outputs["out"].data
    assert df.height == 3

    # Formats follow the file.
    parquet = datasets.folder / "p.parquet"
    pl.DataFrame({"x": [1.5]}).write_parquet(parquet)
    datasets.put("pq", parquet.read_bytes(), "parquet")
    datasets.put("js", b'[{"x": 1}, {"x": 2}]', "json")
    datasets.put("nd", b'{"x": 1}\n{"x": 2}\n', "ndjson")
    for name, rows in (("pq", 1), ("js", 2), ("nd", 2)):
        b = s.add_block("read_dataset", params={"dataset": name})
        assert _run(s, b.id) == "green", s.runner.state[b.id].last_error
        assert s.runner.cache.get(s.runner.state[b.id].last_successful_key).outputs["out"].data.height == rows


def test_read_dataset_errors(hosted):
    server = hosted[1]
    s = server.session
    missing = s.add_block("read_dataset", params={"dataset": "nope"})
    assert _run(s, missing.id) == "red"
    assert "no dataset named 'nope'" in s.runner.state[missing.id].last_error
    unset = s.add_block("read_dataset", params={})
    assert _run(s, unset.id) == "red"
    assert "choose a dataset" in s.runner.state[unset.id].last_error


def test_read_dataset_through_the_api_with_sample_mode(hosted):
    server, datasets = hosted[1], hosted[2]
    datasets.put("loans", ("id\n" + "\n".join(str(i) for i in range(100)) + "\n").encode())
    c = _client(server)
    block = c.post("/api/blocks", json={"category": "read_dataset", "params": {"dataset": "loans"}}).json()
    assert c.put("/api/sample_mode", json={"rows": 10}).status_code == 200
    assert c.post(f"/api/blocks/{block['id']}/run").status_code == 200
    preview = c.get(f"/api/blocks/{block['id']}/preview").json()
    assert preview["row_count"] == 10


# ---- read_source ----------------------------------------------------------------


def test_read_source_queries_through_the_provider(hosted):
    server, sources = hosted[1], hosted[3]
    s = server.session
    b = s.add_block("read_source", params={"source": "dwh", "query": "SELECT * FROM loans"})
    assert _run(s, b.id) == "green", s.runner.state[b.id].last_error
    assert sources.queries == [("dwh", "SELECT * FROM loans")]
    df = s.runner.cache.get(s.runner.state[b.id].last_successful_key).outputs["out"].data
    assert df["q"].to_list() == ["SELECT * FROM loans"] * 3
    # Cached until refreshed; a refresh runs the query again.
    assert s.runner.status(b.id) == "green"
    assert s.runner.refresh(b.id) == "green"
    assert len(sources.queries) == 2
    bad = s.add_block("read_source", params={"source": "other", "query": "SELECT 1"})
    assert _run(s, bad.id) == "red" and "no source named 'other'" in s.runner.state[bad.id].last_error
    empty = s.add_block("read_source", params={"source": "dwh"})
    assert _run(s, empty.id) == "red" and "query" in s.runner.state[empty.id].last_error


# ---- export ------------------------------------------------------------------------


def test_export_goes_to_the_sink(hosted):
    server, datasets, exports = hosted[1], hosted[2], hosted[4]
    datasets.put("loans", b"id,amount\n1,10\n2,20\n")
    s = server.session
    reader = s.add_block("read_dataset", params={"dataset": "loans"})
    out = s.add_block("export", name="Scored", params={"format": "parquet"})
    s.add_wire(reader.id, "out", out.id, "df")
    report = s.runner.run_all()
    assert report[out.id] == "green", s.runner.state[out.id].last_error
    (req,) = exports.received
    assert (req.name, req.format, req.block_name, req.sampled) == ("Scored", "parquet", "Scored", False)
    assert req.data.to_dict(as_series=False) == {"id": [1, 2], "amount": [10, 20]}

    # A sink refusing (a quota, say) fails the block with its message.
    def refuse(request):
        raise DatasetError("export quota exceeded")

    exports.export = refuse
    s.update_block(out.id, params={"format": "csv", "name": "again"})
    assert _run(s, out.id) == "red"
    assert "export quota exceeded" in s.runner.state[out.id].last_error


def test_export_format_is_checked(hosted):
    server, datasets = hosted[1], hosted[2]
    datasets.put("loans", b"id\n1\n")
    s = server.session
    reader = s.add_block("read_dataset", params={"dataset": "loans"})
    out = s.add_block("export", params={"format": "xlsx"})
    s.add_wire(reader.id, "out", out.id, "df")
    report = s.runner.run_all()
    assert report[out.id] == "red"
    assert "format must be one of csv, parquet" in s.runner.state[out.id].last_error


def test_compiling_dataset_blocks_is_refused(hosted):
    from modelmaker.compiler import CompileError, compile_graph

    server, datasets = hosted[1], hosted[2]
    datasets.put("loans", b"id\n1\n")
    reader = server.session.add_block("read_dataset", params={"dataset": "loans"})
    with pytest.raises(CompileError, match="datasets or sources"):
        compile_graph(server.session.graph, strict=False)
    assert reader.id


def test_streaming_run_with_datasets_and_exports(hosted):
    server, datasets, exports = hosted[1], hosted[2], hosted[4]
    datasets.put("loans", b"id,amount\n1,10\n2,20\n3,30\n")
    s = server.session
    reader = s.add_block("read_dataset", params={"dataset": "loans"})
    keep = s.add_block("filter", params={"expr": "amount > 10"})
    out = s.add_block("export", params={"format": "csv", "name": "big"})
    s.add_wire(reader.id, "out", keep.id, "df")
    s.add_wire(keep.id, "out", out.id, "df")
    assert s.runner.refresh(reader.id) == "green"  # a streaming run starts from read inputs
    report = s.runner.run_all_streaming()
    assert report[out.id] == "green", (report, s.runner.state[out.id].last_error)
    assert exports.received[-1].data["id"].to_list() == [2, 3]
