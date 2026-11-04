"""The API process never executes library block code.

Everything the app does with a library block short of running it -- list
it, render its form, place it, wire it, work out staleness, compile the
model, validate the file, describe it to the AI -- reads the block's
static signature (signature.py). Running it ("Try it", examples,
check-blocks, a canvas run) happens in a task process through the
executor. The blocks below have side effects at every place definition
time code can hide: module level, a decorator and a default expression.
The autouse `no_block_code_in_this_process` fixture (conftest.py) also
fails any test here that tries to execute block code in this process."""

from __future__ import annotations

import os
import sys

import polars as pl
import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker import block_library as bl
from modelmaker import forms
from modelmaker import library_store as store
from modelmaker.agent import catalogue
from modelmaker.blocks.base import BLOCK_REGISTRY
from modelmaker.session import ProjectSession
from modelmaker.signature import UNKNOWN_DEFAULT, BlockSignature

ON_LINUX = sys.platform.startswith("linux")


def _source(marker_dir, tail: str = "") -> str:
    """A block file whose block's code writes a marker file (named for the
    place it ran from, holding the pid) at module level, in a decorator and
    in a default expression. `tail` is appended at module level."""
    m = str(marker_dir)
    return f'''"""Side effects everywhere."""

import polars as pl

BLOCKS = {{
    "noisy_double": {{
        "display_name": "Noisy double",
        "inputs": [{{"name": "df", "lazy": True}}],
        "outputs": [{{"name": "out"}}],
        "form": [{{"key": "col", "label": "Column", "kind": "column"}}],
        "meta": {{"roles": [{{"column_param": "col", "role": "feature"}}]}},
    }},
}}


def _mark(where):
    with open({m!r} + "/" + where, "a") as f:
        f.write(str(__import__("os").getpid()) + "\\n")
    return 2


_LOADED = _mark("module")


def _decorate(fn):
    _mark("decorator")
    return fn


@_decorate
def noisy_double(df, col: str = "a", factor=_mark("default"), target_col: str | None = None, block_id: str = ""):
    """Doubles a column, noisily."""
    return df.with_columns(pl.col(col) * factor * (_LOADED // 2))


def _example_noisy_double():
    assert noisy_double(pl.DataFrame({{"a": [1]}}))["a"].to_list() == [2]
{tail}'''


def _markers(marker_dir) -> dict[str, list[int]]:
    return {p.name: [int(x) for x in p.read_text().split()] for p in marker_dir.iterdir()}


@pytest.fixture
def client(tmp_path):
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    return TestClient(api_module.app)


def test_a_side_effecting_library_block_is_handled_without_running_it_here(client, tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    source = _source(markers)

    # Validate (the editor's live check) and save it.
    checked = client.post("/api/library/check", json={"source": source, "name": "noisy.py"}).json()
    assert checked["ok"], checked
    assert client.put("/api/library/files/noisy.py", json={"source": source, "create": True}).status_code == 200

    # List it, with its form and architecture notes.
    lib = client.get("/api/library").json()
    block = next(b for f in lib["files"] for b in f["blocks"] if b["id"] == "noisy_double")
    assert block["errors"] == []
    fields = {f["key"]: f for f in block["form"]["fields"]}
    assert fields["col"]["default"] == "a"
    assert fields["target_col"]["auto_role"] == "target"
    assert "factor" in block["form"]["other_params"]  # unannotated: edited as JSON

    # The AI catalogue's view of it.
    d = store.find("noisy.py", "noisy_double")
    params = {p["name"]: p for p in catalogue.block_params(d)}
    assert params["col"]["default"] == "a"
    assert params["factor"] == {"name": "factor", "type": None, "default_expr": "_mark('default')"}

    # Place it, wire it, look at the graph (forms, statuses, staleness).
    csv = tmp_path / "a.csv"
    csv.write_text("a\n1\n2\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv)}}).json()
    placed = client.post("/api/blocks", json={"library": {"file": "noisy.py", "id": "noisy_double"}}).json()
    wire = client.post(
        "/api/wires", json={"from_block": read["id"], "from_port": "out", "to_block": placed["id"], "to_port": "df"}
    ).json()
    assert wire["valid"]
    graph = client.get("/api/graph").json()
    assert {read["id"], placed["id"]} <= set(graph["blocks"])
    session = api_module.SESSION
    assert session.runner.status(placed["id"]) == "grey"
    session.runner.pin().key(placed["id"])  # the cache key basis asks whether it takes block_id

    # Compile the model to a script, plain and streaming.
    for stream in (False, True):
        compiled = client.post("/api/compile", json={"strict": False, "stream": stream})
        assert compiled.status_code == 200, compiled.text
        assert "def noisy_double(" in compiled.json()["source"]

    # An update preview (params migration reads the signature too).
    v2 = source.replace('"display_name": "Noisy double",', '"display_name": "Noisy double", "version": 2,')
    sha = store.get_file("noisy.py").sha
    assert client.put("/api/library/files/noisy.py", json={"source": v2, "expected_sha": sha}).status_code == 200
    assert client.get(f"/api/blocks/{placed['id']}/update_preview").json()["to_version"] == 2

    assert _markers(markers) == {}, "block code ran in the API process"

    # It still runs -- in a task process.
    client.post(f"/api/blocks/{read['id']}/refresh")
    assert session.runner.run_block(placed["id"]) == "green", session.runner.state[placed["id"]].last_error
    entry = session.runner.cache.get(session.runner.state[placed["id"]].last_successful_key)
    assert entry.outputs["out"].data["a"].to_list() == [2, 4]
    seen = _markers(markers)
    assert set(seen) == {"module", "decorator", "default"}
    assert os.getpid() not in {pid for pids in seen.values() for pid in pids}


@pytest.mark.parametrize(
    "tail, expected",
    [
        ("\nraise SystemExit(3)\n", "SystemExit"),
        ("\n_BYE = __import__('os')._exit(7)\n", bl.CRASHED),
    ],
)
def test_a_block_file_that_exits_at_import_never_takes_the_api_down(client, tmp_path, tail, expected):
    markers = tmp_path / "markers"
    markers.mkdir()
    source = _source(markers, tail)
    # Make the exit part of the block's own code, not only of the file.
    if "_BYE" in tail:
        source = source.replace("* (_LOADED // 2))", "* (_LOADED // 2) + 0 * (_BYE or 0))")
    assert client.put("/api/library/files/bye.py", json={"source": source, "create": True}).status_code == 200
    lib = client.get("/api/library").json()
    block = next(b for f in lib["files"] for b in f["blocks"] if b["id"] == "noisy_double")
    assert block["errors"] == []
    placed = client.post("/api/blocks", json={"library": {"file": "bye.py", "id": "noisy_double"}}).json()
    assert client.post("/api/compile", json={"strict": False}).status_code == 200
    assert client.get("/api/graph").status_code == 200

    examples = client.post("/api/library/files/bye.py/try", json={}).json()["examples"]
    assert expected in examples["__file__"]["error"]
    assert _markers(markers).get("module") and os.getpid() not in _markers(markers)["module"]
    assert placed["id"] in api_module.SESSION.graph.blocks  # still here, still serving


def test_check_blocks_runs_examples_as_tasks(tmp_path, capsys):
    from modelmaker.cli import check_blocks

    markers = tmp_path / "markers"
    markers.mkdir()
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "noisy.py").write_text(_source(markers))
    assert check_blocks(lib) == 0
    assert "(example passed)" in capsys.readouterr().out
    assert os.getpid() not in {pid for pids in _markers(markers).values() for pid in pids}
    assert check_blocks(lib, run_examples=False) == 0


# ---- the guard ---------------------------------------------------------------------------------


def test_defining_a_library_block_here_is_refused(no_block_code_in_this_process):
    d = bl.parse_source(_source("/nonexistent"), "x.py").definitions[0]
    with pytest.raises(bl.BlockCodeNotAllowed):
        _ = d.fn
    with pytest.raises(bl.BlockCodeNotAllowed):
        bl.compile_fn("noisy_double", d.code)
    assert not bl.in_task_process()
    no_block_code_in_this_process.clear()  # the attempts above were the point


def test_builtins_keep_their_real_functions():
    assert BLOCK_REGISTRY["filter"].fn.__module__ == "modelmaker.blocks.library"


# ---- static signatures ---------------------------------------------------------------------------


def test_static_signatures_match_the_live_ones_for_every_builtin():
    for bid, d in BLOCK_REGISTRY.items():
        static, live = d.signature, BlockSignature.from_callable(d.fn)
        assert static.names == live.names, bid
        for s, lv in zip(static.params, live.params):
            assert (s.kind, s.annotation, s.has_default) == (lv.kind, lv.annotation, lv.has_default), (bid, s.name)
            if s.has_default:
                assert s.default == lv.default and type(s.default) is type(lv.default), (bid, s.name)


def test_signature_reads_every_parameter_kind():
    code = (
        "def f(a, /, b: int, c: 'list[str]' = ['x'], *args, d: float = -1.5, e=(1, None), f=SOME, **kw) -> int:\n"
        "    return 1\n"
    )
    sig = BlockSignature.from_code(code, "f")
    kinds = {p.name: p.kind for p in sig.params}
    assert kinds == {
        "a": "positional_only", "b": "positional_or_keyword", "c": "positional_or_keyword", "args": "var_positional",
        "d": "keyword_only", "e": "keyword_only", "f": "keyword_only", "kw": "var_keyword",
    }
    assert sig.get("b").annotation == "int" and sig.get("b").required
    assert sig.get("c").default == ["x"] and sig.get("c").annotation == "'list[str]'"
    assert sig.get("d").default == -1.5 and sig.get("e").default == (1, None)
    assert sig.get("f").has_default and sig.get("f").default is UNKNOWN_DEFAULT and sig.get("f").default_source == "SOME"
    assert sig.literal_defaults() == {"c": ["x"], "d": -1.5, "e": (1, None)}
    assert sig.accepts("kw") and not sig.accepts("nope") and not sig.get("args").required


def test_non_literal_defaults_are_unknown_but_allowed():
    source = (
        "BLOCKS = {'f': {'inputs': [{'name': 'df'}], 'outputs': [{'name': 'out'}]}}\n\n"
        "_LIMIT = 3\n\n"
        "def f(df, limit: int = _LIMIT, scale: float = float('inf'), n: int = 2):\n"
        "    return df.head(limit)\n"
    )
    parsed = bl.validate(bl.parse_source(source, "x.py"))
    assert parsed.ok, parsed.errors
    d = parsed.definitions[0]
    assert d.param_defaults() == {"n": 2}
    fields = {f["key"]: f for f in forms.block_form(d)["fields"]}
    assert fields["limit"]["default"] is None and fields["scale"]["default"] is None and fields["n"]["default"] == 2
    # ...and the block runs with its real defaults, in a task process.
    ok, out = bl.try_call(d, {"df": pl.DataFrame({"a": range(10)})})
    assert ok and out.height == 3


@pytest.mark.parametrize(
    "meta, signature, message",
    [
        ({"roles": [{"column_param": "col", "role": "target"}]}, "col: str = _NAME", "used by a meta rule"),
        ({"inherit": {"columns_param": "by"}}, "by: list = list()", "used by a meta rule"),
        (
            {"roles": [{"column_param": "col", "fallback": "default_{horizon}m", "role": "target"}]},
            "col: str | None = None, horizon: int = _H",
            "'horizon' is used by a meta rule",
        ),
        ({}, "col: str = UNDEFINED_THING", "default of 'col' uses 'UNDEFINED_THING'"),
    ],
)
def test_non_literal_defaults_that_cant_work_are_errors(meta, signature, message):
    source = (
        f"BLOCKS = {{'f': {{'inputs': [{{'name': 'df'}}], 'outputs': [{{'name': 'out'}}], 'meta': {meta!r}}}}}\n\n"
        "_NAME = 'x'\n_H = 12\n\n"
        f"def f(df, {signature}):\n    return df\n"
    )
    parsed = bl.validate(bl.parse_source(source, "x.py"))
    assert any(message in e for e in parsed.errors["f"]), parsed.errors["f"]


def test_an_undefined_decorator_is_reported_statically():
    source = "BLOCKS = {'f': {}}\n\n@nowhere\ndef f():\n    return 1\n"
    parsed = bl.validate(bl.parse_source(source, "x.py"))
    assert any("decorator uses 'nowhere'" in e for e in parsed.errors["f"]), parsed.errors["f"]


def test_embedded_code_without_its_def_is_a_validation_error_not_an_exec():
    from modelmaker.blocks.base import BlockDefinition

    d = BlockDefinition(id="f", meta={}, code="f = 1\n", source="library", origin=None, sha="x")
    assert bl.validate_definition(d) == ["block code doesn't define f()"]
    broken = BlockDefinition(id="f", meta={}, code="def f(:\n", source="library", origin=None, sha="y")
    assert "code doesn't parse" in bl.validate_definition(broken)[0]


# ---- Try it and examples, through the executor ------------------------------------------------------


EXAMPLES = '''"""Examples."""

import polars as pl

BLOCKS = {"ok_block": {}, "bad_block": {}, "plain_block": {}}


def ok_block():
    return 1


def bad_block():
    return 2


def plain_block():
    return 3


def _example_ok_block():
    assert ok_block() == 1


def _example_bad_block():
    assert bad_block() == 3, "two is not three"
'''


def test_examples_report_results_and_the_failing_line():
    result = bl.run_examples(EXAMPLES, "ex.py")
    line = EXAMPLES.splitlines().index('    assert bad_block() == 3, "two is not three"') + 1
    assert result == {
        "ok_block": {"ok": True, "error": None},
        "bad_block": {"ok": False, "error": "AssertionError: two is not three", "line": line},
        "plain_block": {"ok": None, "error": "no _example_ for this block"},
    }
    assert bl.run_examples(EXAMPLES, "ex.py", ["ok_block"]) == {"ok_block": {"ok": True, "error": None}}


def test_an_example_that_crashes_its_process_is_reported():
    source = EXAMPLES.replace("    assert ok_block() == 1", "    __import__('os')._exit(9)")
    result = bl.run_examples(source, "ex.py")
    assert bl.CRASHED in result["__file__"]["error"]


def test_try_call_reports_errors_and_refuses_a_malicious_result(tmp_path):
    marker = tmp_path / "pwned"
    source = f'''BLOCKS = {{"evil": {{"outputs": [{{"name": "out", "type": "any"}}]}}, "raises": {{}}}}


class _Payload:
    def __reduce__(self):
        return (__import__("os").system, ("touch {marker}",))


def evil():
    return _Payload()


def raises():
    raise ValueError("nope")
'''
    defs = {d.id: d for d in bl.parse_source(source, "x.py").definitions}
    ok, error = bl.try_call(defs["evil"], {})
    assert not ok and "can't return" in error
    assert not marker.exists()
    assert bl.try_call(defs["raises"], {}) == (False, "ValueError: nope")


def test_try_it_and_examples_run_without_the_api_secrets(monkeypatch):
    from modelmaker.execution.local import LocalExecutor
    from modelmaker.runner import MP_CONTEXT
    from modelmaker.worker_pool import WorkerPool

    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "sekrit-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sekrit-key")
    monkeypatch.setenv("HARMLESS_SETTING", "fine")
    source = '''BLOCKS = {"peek": {"outputs": [{"name": "out", "type": "any"}]}}


def peek():
    import os

    proc = ""
    if os.path.exists("/proc/self/environ"):
        proc = open("/proc/self/environ", "rb").read().decode(errors="replace")
    return {k: (os.environ.get(k), k in proc) for k in ("MODELMAKER_AUTH_TOKEN", "ANTHROPIC_API_KEY", "HARMLESS_SETTING")}


def _example_peek():
    import os

    assert "MODELMAKER_AUTH_TOKEN" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
    assert os.environ["HARMLESS_SETTING"] == "fine"
'''
    # A fresh pool: no spare started before the variables were set.
    executor = LocalExecutor(WorkerPool(MP_CONTEXT, 0))
    try:
        d = bl.parse_source(source, "x.py").definitions[0]
        ok, seen = bl.try_call(d, {}, executor=executor)
        assert ok, seen
        assert seen["MODELMAKER_AUTH_TOKEN"] == (None, False) and seen["ANTHROPIC_API_KEY"] == (None, False)
        assert seen["HARMLESS_SETTING"][0] == "fine"
        if ON_LINUX:
            assert seen["HARMLESS_SETTING"][1] is True
        assert bl.run_examples(source, "x.py", executor=executor) == {"peek": {"ok": True, "error": None}}
    finally:
        executor.close()


def test_try_call_times_out():
    source = "BLOCKS = {'slow': {}}\n\ndef slow():\n    import time\n    time.sleep(30)\n"
    d = bl.parse_source(source, "x.py").definitions[0]
    assert bl.try_call(d, {}, timeout=2) == (False, "timed out after 2s")
