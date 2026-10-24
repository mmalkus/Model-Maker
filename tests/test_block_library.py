"""Block files (modelmaker/block_library.py), the declarative metadata rules
(metadata_transforms.schema_diff), change detection (sources.py), the
user's block library (library_store.py), moving placed blocks between
versions (library_updates.py), their API routes and the check-blocks CLI."""

from __future__ import annotations

import base64
import io
import os
import time
import zipfile

import polars as pl
import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker import block_library as bl
from modelmaker import library_store as store
from modelmaker import library_updates
from modelmaker.blocks.base import BLOCK_REGISTRY
from modelmaker.cache import CacheStore
from modelmaker.graph import Graph, Wire
from modelmaker.metadata_transforms import schema_diff
from modelmaker.packet import ColumnMeta, ColumnRole
from modelmaker.project import ProjectFormatError, graph_from_dict, load_project, save_project
from modelmaker.runner import Runner
from modelmaker.session import ProjectSession
from modelmaker.sources import fingerprint

from .helpers import make_block

GOOD = '''"""PD backtests."""

import polars as pl

BLOCKS = {
    # comment inside the literal
    "double_it": {
        "display_name": "Double it",
        "version": 2,
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [{"key": "col", "label": "Column", "kind": "column"}],
        "renamed": {"column": "col"},
    },
    "halve_it": {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]},
}

_FACTOR = 2


def _scaled(col, f):
    return pl.col(col) * f


def double_it(df, col: str = "a"):
    """Doubles a column.

    More words."""
    return df.with_columns(_scaled(col, _FACTOR).alias(col))


def halve_it(df, col: str = "a"):
    return df.with_columns(_scaled(col, 0.5).alias(col))


def _example_double_it():
    assert double_it(pl.DataFrame({"a": [1]}))["a"].to_list() == [2]
'''


# ---- parsing and validation ------------------------------------------------------------


def test_parse_reads_the_literal_and_finds_each_blocks_code_and_helpers():
    parsed = bl.validate(bl.parse_source(GOOD, "pd.py"))
    assert parsed.ok, (parsed.file_errors, parsed.errors)
    assert parsed.docstring == "PD backtests."
    d = {x.id: x for x in parsed.definitions}["double_it"]
    assert d.version == 2 and d.display_name == "Double it" and d.summary == "Doubles a column."
    assert d.inputs[0].lazy and d.renamed == {"column": "col"}
    # The function plus the helpers it reaches (transitively), never the example.
    assert "_FACTOR = 2" in d.code and "def _scaled(" in d.code and "def double_it(" in d.code
    assert "_example_" not in d.code and "def halve_it(" not in d.code
    assert parsed.examples == {"double_it": "_example_double_it"}
    assert d.fn(pl.DataFrame({"a": [3]}))["a"].to_list() == [6]


def test_parsing_and_listing_never_run_the_file(block_library_dir):
    source = GOOD.replace("_FACTOR = 2", "_FACTOR = open('/definitely/not/here').read()")
    parsed = bl.parse_source(source, "x.py")
    assert [d.id for d in parsed.definitions] == ["double_it", "halve_it"]  # no exception: nothing ran
    store.save_file("x.py", source)
    preview = store.import_preview([("x.py", source)])
    assert preview[0]["blocks"][0]["errors"] == []


def test_listing_checking_and_importing_never_execute_block_code(block_library_dir, tmp_path):
    # Code a block reaches at definition time -- a helper constant used as
    # a default, a decorator -- must not run until the block itself runs.
    marker = tmp_path / "ran"
    source = GOOD.replace(
        "_FACTOR = 2",
        f"_FACTOR = open({str(marker)!r}, 'a').write('x') and 2\n\n\ndef _deco(f):\n    open({str(marker)!r}, 'a').write('d')\n    return f",
    ).replace("def double_it(df, col: str = \"a\"):", "@_deco\ndef double_it(df, col: str = \"a\", n: int = _FACTOR):")
    store.import_commit([("x.py", source)], {})
    files = store.list_files(force=True)
    assert files[0].parsed.errors == {"double_it": [], "halve_it": []}
    bl.validate(bl.parse_source(source, "y.py"))
    store.import_preview([("y.py", source)])
    client = TestClient(api_module.app)
    assert client.get("/api/library").status_code == 200
    assert client.post("/api/library/check", json={"source": source, "name": "x.py"}).json()["ok"] is True
    assert not marker.exists()


def test_signature_stub_keeps_literal_defaults_only():
    code = (
        "_N = 5\n"
        "_COLS = ['a', 'b']\n\n\n"
        "@some_decorator(1)\n"
        "def blk(df, k: int = 3, n: int = _N, cols: list = _COLS, f=pl.col('x'), *, flag: bool = -1 < 0, req: str):\n"
        "    raise SystemExit\n"
    )
    import inspect

    params = inspect.signature(bl.signature_stub("blk", code)).parameters
    assert {k: p.default for k, p in params.items() if p.default is not inspect.Parameter.empty} == {
        "k": 3, "n": 5, "cols": ["a", "b"], "f": None, "flag": None,
    }
    assert params["req"].default is inspect.Parameter.empty
    assert params["k"].annotation == "int"
    with pytest.raises(ValueError, match="doesn't define"):
        bl.signature_stub("other", code)


def test_compiled_functions_are_cached_with_a_bound():
    assert bl.compile_fn.cache_info().maxsize is not None
    assert bl.signature_stub.cache_info().maxsize is not None


def test_requires_must_be_a_whole_pip_requirement():
    ok = ["scikit-learn", "scikit-learn>=1.3", "numpy>=1.20,<3", "xgboost[cuda]==2.0.3", "pyarrow; python_version>='3.9'"]
    for req in ok:
        assert not [e for e in bl.validate_definition(_definition({"requires": [req]})) if "requires" in e], req
    for req in ["numpy && rm -rf /", "numpy>=", "-e git+https://x"]:
        assert any("requires" in e for e in bl.validate_definition(_definition({"requires": [req]}))), req


def _definition(extra_meta):
    from modelmaker.blocks.base import BlockDefinition

    code = "def blk(df):\n    return df\n"
    meta = {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}], **extra_meta}
    return BlockDefinition(id="blk", meta=meta, code=code, source="library", origin="t.py", sha=bl._sha(meta, code))


def test_an_embedded_definition_is_keyed_on_its_actual_code():
    from types import SimpleNamespace

    meta = {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}
    one = {"id": "blk", "meta": meta, "code": "def blk(df):\n    return 1\n", "sha": "samesha"}
    two = {**one, "code": "def blk(df):\n    return 2\n"}  # hand-edited, sha left stale
    assert bl.definition_for(SimpleNamespace(definition=one)).fn(None) == 1
    assert bl.definition_for(SimpleNamespace(definition=two)).fn(None) == 2


@pytest.mark.parametrize(
    "source, message",
    [
        ("def f(df):\n    return df\n", "no BLOCKS"),
        ("BLOCKS = {'f': dict()}\n\ndef f(df):\n    return df\n", "plain literal"),
        ("import os\n\nBLOCKS = {}\n", "import 'os' at module level"),
        ("BLOCKS = {}\n\ndef f(df):\n    return df\n", "f() has no BLOCKS entry"),
    ],
)
def test_file_level_errors(source, message):
    parsed = bl.parse_source(source, "x.py")
    assert any(message in e for e in parsed.file_errors), parsed.file_errors


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"outputs": [{"name": "out", "type": "frame"}]}, "unknown type 'frame'"),
        ({"form": [{"key": "nope", "kind": "text"}]}, "form field 'nope' isn't a parameter"),
        ({"form": [{"key": "a", "kind": "slider"}]}, "unknown kind 'slider'"),
        ({"tags": ["nonsense"]}, "unknown tag(s) ['nonsense']"),
        ({"hints": "no_such_rule"}, "unknown hints rule"),
        ({"meta": {"roles": [{"column": "x", "role": "boss"}]}}, "isn't a role"),
        ({"wat": 1}, "unknown key(s) ['wat']"),
        ({"inputs": [{"name": "data"}]}, "input port(s) ['data'] aren't parameters"),
    ],
)
def test_block_level_errors(entry, message):
    source = f"BLOCKS = {{'f': {entry!r}}}\n\ndef f(df, a: int = 1):\n    return df\n"
    parsed = bl.validate(bl.parse_source(source, "x.py"))
    assert any(message in e for e in parsed.errors["f"]), parsed.errors["f"]


def test_a_missing_function_and_a_taken_id_are_errors():
    parsed = bl.validate(bl.parse_source("BLOCKS = {'filter': {}, 'g': {}}\n\ndef filter(df):\n    return df\n", "x.py"), set(BLOCK_REGISTRY))
    assert "already taken" in parsed.errors["filter"][0]
    assert "has no function def g" in parsed.errors["g"][0]


def test_every_builtin_and_the_template_validate():
    assert {k: v for k, v in bl.check_builtin().items() if v} == {}
    template = bl.TEMPLATE_PATH.read_text()
    parsed = bl.validate(bl.parse_source(template, "_template.py"))
    assert parsed.ok, (parsed.file_errors, parsed.errors)
    assert bl.run_examples(template, "_template.py") == {"flag_outliers": {"ok": True, "error": None}}


def test_requirements_are_checked_from_package_metadata_only():
    d = bl.parse_source("BLOCKS = {'f': {'requires': ['polars>=0.1', 'surely-not-a-package-xyz', 'polars<0.1']}}\n\ndef f():\n    return 1\n", "x.py").definitions[0]
    assert bl.check_requirements(d) == ["surely-not-a-package-xyz", "polars<0.1"]


def test_examples_report_failures_and_timeouts():
    source = GOOD.replace("== [2]", "== [3]")
    result = bl.run_examples(source, "x.py")
    assert result["double_it"]["ok"] is False and "AssertionError" in result["double_it"]["error"]
    assert result["halve_it"]["ok"] is None
    hang = GOOD.replace('    assert double_it(pl.DataFrame({"a": [1]}))["a"].to_list() == [2]', "    import time\n    time.sleep(30)")
    assert "timed out" in bl.run_examples(hang, "x.py", timeout=2)["__file__"]["error"]


def test_format_literal_round_trips():
    value = {"a": [1, 2.5, None, True], "b": {"c": "d\"e", "f": []}, "long": ["x" * 40, "y" * 40, "z" * 40]}
    import ast

    assert ast.literal_eval(bl.format_literal(value)) == value
    assert "\n" in bl.format_literal(value)


# ---- declarative metadata -------------------------------------------------------------


def _m(role="unassigned", dtype="Int64"):
    return ColumnMeta(dtype=dtype, role=ColumnRole(role))


def test_schema_diff_inherits_with_the_first_port_winning_and_infers_new_columns():
    left = {"k": _m("id"), "a": _m("feature")}
    right = {"k": _m("segment"), "b": _m("target")}
    out = pl.DataFrame({"k": [1], "a": [1.0], "b": [1], "new": ["x"]})
    meta = schema_diff({"new_role": "feature"})({"left": left, "right": right}, {"out": out}, {})["out"]
    assert meta["k"].role == ColumnRole.ID  # left wins
    assert meta["a"].dtype == "Float64" and meta["a"].role == ColumnRole.FEATURE  # dtype refreshed
    assert meta["b"].role == ColumnRole.TARGET
    assert meta["new"].role == ColumnRole.FEATURE and meta["new"].dtype == "String"


def test_schema_diff_inherit_none_port_and_columns_param():
    ins = {"df": {"g": _m("segment"), "x": _m("target")}, "other": {"y": _m("feature")}}
    out = pl.DataFrame({"g": [1], "x": [1], "y": [1]})
    none = schema_diff({"inherit": "none"})(ins, {"out": out}, {})["out"]
    assert {m.role for m in none.values()} == {ColumnRole.UNASSIGNED}
    port = schema_diff({"inherit": {"port": "df"}})(ins, {"out": out}, {})["out"]
    assert port["x"].role == ColumnRole.TARGET and port["y"].role == ColumnRole.UNASSIGNED
    keys = schema_diff({"inherit": {"columns_param": "by"}})(ins, {"out": out}, {"by": ["g"]})["out"]
    assert keys["g"].role == ColumnRole.SEGMENT and keys["x"].role == ColumnRole.UNASSIGNED


def test_schema_diff_role_rules():
    ins = {"df": {"target_old": _m("target"), "p_old": _m("predicted"), "flag": _m()}}
    out = pl.DataFrame({"target_old": [1], "p_old": [1], "flag": [1], "lgd": [0.5], "default_12m": [1], "x_bin": ["a"], "p_new": [1.0]})
    rules = {
        "roles": [
            {"column": "lgd", "role": "target", "if_taken": "feature"},
            {"column_param": "flag_col", "fallback": "default_{horizon}m", "role": "target", "displaced": "excluded"},
            {"column_param": "out_col", "role": "predicted", "displaced": "feature"},
            {"column_suffix": "_bin", "role": "segment"},
            {"column_param": "default_col", "role": "excluded"},
        ]
    }
    meta = schema_diff(rules, defaults={"horizon": 12, "out_col": "p_new"})(ins, {"out": out}, {"default_col": "flag"})["out"]
    assert meta["lgd"].role == ColumnRole.FEATURE  # target was taken
    assert meta["default_12m"].role == ColumnRole.TARGET and meta["target_old"].role == ColumnRole.EXCLUDED
    assert meta["p_new"].role == ColumnRole.PREDICTED and meta["p_old"].role == ColumnRole.FEATURE
    assert meta["x_bin"].role == ColumnRole.SEGMENT and meta["flag"].role == ColumnRole.EXCLUDED


# ---- change detection -------------------------------------------------------------------


def test_change_detection_comes_from_path_fields(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("a\n1\n")
    read_csv = BLOCK_REGISTRY["read_csv"]
    before = fingerprint(read_csv, {"path": str(csv)})
    os.utime(csv, (time.time() + 10, time.time() + 10))
    assert fingerprint(read_csv, {"path": str(csv)}) != before
    assert fingerprint(BLOCK_REGISTRY["filter"], {"expr": "a"}) is None

    graph = Graph(blocks={"r": make_block("r", "read_csv", params={"path": str(csv)})})
    runner = Runner(graph, CacheStore())
    assert runner.check_for_changes("r") is False
    csv.write_text("a\n1\n2\n")
    assert runner.check_for_changes("r") is True


def test_probe_query_is_never_passed_to_the_function(tmp_path, monkeypatch):
    import sqlite3

    pytest.importorskip("connectorx", reason="read_sql needs the optional connectorx package")

    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    con.execute("create table t (a int)")
    con.execute("insert into t values (1)")
    con.commit()
    con.close()
    monkeypatch.setenv("MM_DB", f"sqlite://{db}")
    block = make_block("s", "read_sql", params={"connection_env": "MM_DB", "query": "select * from t", "probe_query": "select count(*) from t"})
    runner = Runner(Graph(blocks={"s": block}), CacheStore())
    assert runner.refresh("s") == "green", runner.state["s"].last_error


# ---- the library folder -------------------------------------------------------------------


def test_save_list_and_refuse_a_stale_save(block_library_dir):
    f = store.save_file("pd.py", GOOD)
    assert (block_library_dir / "pd.py").read_text() == GOOD
    assert [d.id for d in f.parsed.definitions] == ["double_it", "halve_it"]
    with pytest.raises(store.ConflictError):
        store.save_file("pd.py", GOOD + "\n", expected_sha="stale")
    with pytest.raises(store.ConflictError):
        store.save_file("pd.py", GOOD + "\n")  # an existing file needs its sha
    store.save_file("pd.py", GOOD + "\n", expected_sha=f.sha)
    with pytest.raises(store.LibraryError):
        store.save_file("../escape.py", GOOD)
    with pytest.raises(store.LibraryError):
        store.save_file("sub/dir.py", GOOD)


def test_ids_clash_with_builtins_and_across_files(block_library_dir):
    store.save_file("a.py", GOOD)
    store.save_file("b.py", GOOD.replace('"halve_it": {', '"filter": {').replace("def halve_it(", "def filter("))
    b = store.get_file("b.py")
    assert "already used by another library file" in b.parsed.errors["double_it"][0]
    assert "shadowed by built-in" in b.parsed.errors["filter"][0]
    with pytest.raises(store.LibraryError, match="shadowed"):
        store.placeable("b.py", "filter")


def test_files_changing_on_disk_are_picked_up(block_library_dir):
    store.save_file("a.py", GOOD)
    gen = store.generation
    (block_library_dir / "a.py").write_text(GOOD.replace('"version": 2', '"version": 3'))
    store.reload()
    assert store.find("a.py", "double_it").version == 3 and store.generation > gen


@pytest.mark.parametrize(
    "blocks",
    [
        'BLOCKS = {\n    "a": {"display_name": "A"}  # note\n}',
        'BLOCKS = {\n    "a": {"display_name": "A"},  # note\n}',
        'BLOCKS = {"a": {"display_name": "A"}}',
        'BLOCKS = {\n    "a": {"display_name": "A"}\n    # trailing, comment\n}',
        "BLOCKS = {}",
    ],
)
def test_upsert_adds_an_entry_whatever_the_last_one_looks_like(blocks):
    import ast

    source = f'"""d"""\n\nimport polars as pl\n\n{blocks}\n\n\ndef a(df):\n    return df\n'
    if blocks == "BLOCKS = {}":
        source = source.replace("\n\n\ndef a(df):\n    return df\n", "\n")
    out = store.upsert_block(source, "b", {"display_name": "B"}, "def b(df):\n    return df\n")
    literal = ast.literal_eval(next(n for n in ast.parse(out).body if isinstance(n, ast.Assign)).value)
    assert list(literal)[-1] == "b"
    if "# note" in blocks:
        assert "# note\n" in out


def test_remove_block_needs_a_blocks_dict():
    with pytest.raises(store.LibraryError, match="no BLOCKS"):
        store.remove_block("def a(df):\n    return df\n", "a")


def test_renaming_matches_whole_names_only():
    code = "def bin(df):\n    return woe_bin(df) + bin(df) + x.bin(df)\n\n\ndef _example_bin():\n    bin(None)\n"
    assert store._rename_calls(code, "bin", "bin_2") == (
        "def bin_2(df):\n    return woe_bin(df) + bin_2(df) + x.bin(df)\n\n\ndef _example_bin_2():\n    bin_2(None)\n"
    )


def test_keeping_a_draft_restores_it_when_the_destination_fails(block_library_dir):
    store.add_block("drafts/b1.py", GOOD_ENTRY, "triple_it", TRIPLE_CODE)
    before = store.get_file("drafts/b1.py").source
    (block_library_dir / "Bad.py").write_text("this is not python (")
    store.reload()
    with pytest.raises(SyntaxError):
        store.move_block("drafts/b1.py", "triple_it", "Bad.py")
    assert store.get_file("drafts/b1.py").source == before


GOOD_ENTRY = {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}
TRIPLE_CODE = "def triple_it(df):\n    return df\n"


def test_upsert_rewrites_only_that_block_and_keeps_comments(block_library_dir):
    out = store.upsert_block(GOOD, "halve_it", {"display_name": "Halve", "inputs": [{"name": "df"}], "outputs": [{"name": "out"}]},
                             "def halve_it(df, col: str = 'b'):\n    return df\n")
    assert "# comment inside the literal" in out and '"Double it"' in out
    assert "col: str = 'b'" in out and "_scaled(col, 0.5)" not in out
    parsed = bl.validate(bl.parse_source(out, "x.py"))
    assert parsed.ok and {d.id: d.display_name for d in parsed.definitions}["halve_it"] == "Halve"

    added = store.upsert_block(out, "triple_it", {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]},
                               "def _three():\n    return 3\n\n\ndef triple_it(df):\n    return df * _three()\n")
    parsed = bl.validate(bl.parse_source(added, "x.py"))
    assert parsed.ok and [d.id for d in parsed.definitions] == ["double_it", "halve_it", "triple_it"]

    removed = store.remove_block(added, "double_it")
    parsed = bl.validate(bl.parse_source(removed, "x.py"))
    assert parsed.ok, parsed.file_errors
    assert [d.id for d in parsed.definitions] == ["halve_it", "triple_it"] and "_example_double_it" not in removed
    assert "def _scaled(" in removed  # helpers stay


def test_add_block_creates_the_file_and_keep_moves_a_draft(block_library_dir):
    meta = {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}
    store.add_block("drafts/build_x.py", meta, "ratio", "def ratio(df):\n    return df\n")
    assert store.get_file("drafts/build_x.py").is_draft
    store.move_block("drafts/build_x.py", "ratio", "kept.py")
    assert not (block_library_dir / "drafts" / "build_x.py").exists()  # empty drafts file removed
    assert store.find("kept.py", "ratio").id == "ratio"


def test_extract_exports_just_the_chosen_blocks_with_their_helpers(block_library_dir):
    store.save_file("pd.py", GOOD)
    text = store.extract("pd.py", ["double_it"])
    parsed = bl.validate(bl.parse_source(text, "x.py"))
    assert parsed.ok and [d.id for d in parsed.definitions] == ["double_it"]
    assert "_example_double_it" in text and "def halve_it" not in text


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def test_import_preview_flags_clashes_and_commit_applies_choices(block_library_dir):
    store.save_file("pd.py", GOOD)
    upload = store.read_upload("team.zip", _zip({"pd.py": GOOD.replace('"version": 2', '"version": 3'), "notes.py": "x = 1\n"}))
    preview = {p["file"]: p for p in store.import_preview(upload)}
    assert preview["notes.py"]["file_errors"] == ["no BLOCKS = {...} found"]
    clash = {b["id"]: b["clash"] for b in preview["pd.py"]["blocks"]}
    assert clash["double_it"] == {"kind": "library", "file": "pd.py", "version": 2}

    written = store.import_commit(upload, {"pd.py:double_it": "overwrite", "pd.py:halve_it": "rename"})
    assert written == ["double_it", "halve_it_2"]
    assert store.find("pd.py", "double_it").version == 3
    assert store.find("pd.py", "halve_it_2").fn(pl.DataFrame({"a": [4]}))["a"].to_list() == [2.0]


@pytest.mark.parametrize("name", ["../evil.py", "dir/evil.py", "/abs.py"])
def test_import_refuses_zip_paths(name):
    with pytest.raises(store.LibraryError, match="plain file name"):
        store.read_upload("x.zip", _zip({name: GOOD}))


def test_import_limits():
    with pytest.raises(store.LibraryError, match="over 1 MB"):
        store.read_upload("big.py", b"#" * (store.MAX_FILE_BYTES + 1))
    # Each entry under the per-file limit, but far too much once unpacked.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(store.MAX_UNPACKED_BYTES // store.MAX_FILE_BYTES + 1):
            zf.writestr(f"f{i}.py", "#" * (store.MAX_FILE_BYTES - 1))
    assert len(buf.getvalue()) < store.MAX_UPLOAD_BYTES
    with pytest.raises(store.LibraryError, match="uncompressed"):
        store.read_upload("x.zip", buf.getvalue())
    with pytest.raises(store.LibraryError, match="entries"):
        store.read_upload("x.zip", _zip({f"f{i}.py": "" for i in range(store.MAX_ZIP_ENTRIES + 1)}))


def test_import_skips_blocks_with_errors_unless_asked(block_library_dir):
    broken = GOOD.replace('"halve_it": {', '"halve_it": {"bogus_key": 1, ')
    assert store.import_commit([("pd.py", broken)], {}) == ["double_it"]
    assert store.import_commit([("pd2.py", broken.replace("double_it", "quad_it"))], {"pd2.py:halve_it": "rename"}) == ["quad_it", "halve_it_2"]
    with pytest.raises(store.LibraryError, match=".py file or a .zip"):
        store.read_upload("x.txt", b"")


# ---- placed library blocks: embed, update, accept ---------------------------------------------


def _session_with(source: str) -> tuple[ProjectSession, str]:
    store.save_file("pd.py", source)
    session = ProjectSession()
    embedded, ref = store.placeable("pd.py", "double_it")
    with session.edit():
        b = session.add_block("double_it", definition=embedded, library_ref=ref, params={"col": "a"})
    return session, b.id


def test_a_placed_block_embeds_its_definition_and_survives_without_the_library(block_library_dir, tmp_path):
    session, bid = _session_with(GOOD)
    path = tmp_path / "proj" / "model.json"
    save_project(session.graph, path)
    (block_library_dir / "pd.py").unlink()
    store.reload()
    graph = load_project(path)
    assert graph.blocks[bid].resolve().fn(pl.DataFrame({"a": [1]}))["a"].to_list() == [2]
    assert (tmp_path / "proj" / "blocks" / f"{bid}.py").read_text().startswith("_FACTOR = 2")
    assert library_updates.status_for(graph.blocks[bid])["in_library"] is False


def test_update_carries_renamed_params_and_wires_and_drops_what_cant_stay(block_library_dir):
    session, bid = _session_with(GOOD)
    with session.edit():
        read = session.add_block("display_table")
        session.add_wire(bid, "out", read.id, "df")
    assert library_updates.status_for(session.graph.blocks[bid])["update_available"] is False

    v3 = (
        GOOD.replace('"version": 2', '"version": 3')
        .replace('"renamed": {"column": "col"}', '"renamed": {"col": "column"}, "renamed_ports": {"out": "doubled"}')
        .replace('"outputs": [{"name": "out"}],\n        "form"', '"outputs": [{"name": "doubled"}],\n        "form"')
        .replace('{"key": "col", "label": "Column", "kind": "column"}', '{"key": "column", "label": "Column", "kind": "column"}')
        .replace('def double_it(df, col: str = "a"):', 'def double_it(df, column: str = "a", scale: float = 1.0):')
        .replace("_scaled(col, _FACTOR).alias(col)", "_scaled(column, _FACTOR * scale).alias(column)")
    )
    store.save_file("pd.py", v3, expected_sha=store.get_file("pd.py").sha)
    info = library_updates.status_for(session.graph.blocks[bid])
    assert info["update_available"] and info["latest_version"] == 3

    preview = library_updates.preview(session.graph, bid)
    assert preview["params_renamed"] == {"col": "column"} and preview["params_dropped"] == []
    assert list(preview["wires_renamed"].values()) == ["doubled"] and preview["wires_disconnected"] == []
    assert "scale" in preview["code_diff"]

    with session.edit():
        library_updates.apply(session, bid)
    block = session.graph.blocks[bid]
    assert block.params == {"column": "a"} and block.library_ref["version"] == 3
    assert [w.from_port for w in session.graph.wires.values()] == ["doubled"]
    assert library_updates.status_for(block)["update_available"] is False
    session.undo()
    assert session.graph.blocks[bid].params == {"col": "a"}


def test_a_changed_builtin_is_flagged_until_accepted():
    session = ProjectSession()
    with session.edit():
        b = session.add_block("filter", params={"expr": "a > 1"})
    assert library_updates.status_for(b) == {"source": "builtin", "sha": BLOCK_REGISTRY["filter"].sha, "changed_since_saved": False}
    b.builtin_sha = "0ld"
    assert library_updates.status_for(b)["changed_since_saved"] is True
    with session.edit():
        library_updates.apply(session, b.id)
    assert session.graph.blocks[b.id].builtin_sha == BLOCK_REGISTRY["filter"].sha


def test_old_custom_blocks_are_refused_with_a_clear_message():
    data = {"blocks": {"b1": {"block_type": "standard", "category": "x", "name": "Mine", "is_custom": True,
                              "ports": {"inputs": [], "outputs": []}}}}
    with pytest.raises(ProjectFormatError, match="custom block from an older Model-Maker"):
        graph_from_dict(data)


# ---- API ---------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    return TestClient(api_module.app)


def test_library_api_round_trip(client, block_library_dir):
    assert client.get("/api/library").json()["files"] == []
    created = client.put("/api/library/files/pd.py", json={"source": GOOD, "create": True}).json()
    assert [b["id"] for b in created["blocks"]] == ["double_it", "halve_it"]
    assert client.put("/api/library/files/pd.py", json={"source": GOOD, "expected_sha": "nope"}).status_code == 409

    lib = client.get("/api/library").json()
    double = lib["files"][0]["blocks"][0]
    assert double["architecture"]["streaming"].startswith("fuses")
    assert double["form"]["fields"][0]["key"] == "col"
    builtin = {b["id"]: b for f in lib["builtin_files"] for b in f["blocks"]}
    assert builtin["read_csv"]["architecture"]["change_detection"] == ["file changes (modified time and size)"]

    checked = client.post("/api/library/check", json={"source": GOOD.replace('"key": "col"', '"key": "c"'), "name": "pd.py"}).json()
    assert not checked["ok"] and "isn't a parameter" in checked["blocks"][0]["errors"][0]
    assert client.post("/api/library/files/pd.py/try", json={}).json()["examples"]["double_it"]["ok"] is True

    block = client.post("/api/blocks", json={"library": {"file": "pd.py", "id": "double_it"}}).json()
    assert block["definition_info"] == {"source": "library", "file": "pd.py", "version": 2, "sha": double["sha"],
                                        "in_library": True, "update_available": False, "library_older": False,
                                        "latest_version": 2}
    assert client.get("/api/library").json()["usage"]["double_it"] == [block["id"]]

    exported = client.get("/api/library/export", params={"file": "pd.py", "blocks": "halve_it"})
    assert exported.headers["content-disposition"].endswith('halve_it.py"')
    payload = {"filename": "halve_it.py", "content_base64": base64.b64encode(exported.content).decode()}
    preview = client.post("/api/library/import/preview", json=payload).json()
    assert preview["files"][0]["blocks"][0]["clash"]["kind"] == "library"
    done = client.post("/api/library/import", json={**payload, "actions": {"halve_it.py:halve_it": "rename"}}).json()
    assert done == {"imported": ["halve_it_2"]}

    v3 = GOOD.replace('"version": 2', '"version": 3')
    client.put("/api/library/files/pd.py", json={"source": v3, "expected_sha": created["sha"]})
    assert client.get(f"/api/blocks/{block['id']}/update_preview").json()["to_version"] == 3
    updated = client.post(f"/api/blocks/{block['id']}/update").json()
    assert updated["block"]["definition_info"]["version"] == 3

    assert client.delete("/api/library/files/pd.py/blocks/halve_it").json()["blocks"][0]["id"] == "double_it"
    assert client.delete("/api/library/files/halve_it.py").status_code == 422  # expected_sha is required
    sha = client.get("/api/library/files/halve_it.py").json()["sha"]
    assert client.delete("/api/library/files/halve_it.py", params={"expected_sha": "stale"}).status_code == 409
    assert client.delete("/api/library/files/halve_it.py", params={"expected_sha": sha}).status_code == 200


def test_try_a_block_on_a_canvas_blocks_output(client, tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("a\n1\n2\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv)}}).json()
    client.post(f"/api/blocks/{read['id']}/refresh")
    result = client.post("/api/library/files/new.py/try", json={"source": GOOD, "block": "double_it", "input_block": read["id"]}).json()
    assert result["ok"] and result["outputs"]["out"]["head"] == [{"a": 2}, {"a": 4}]


def test_drafts_keep_and_discard(client):
    meta = {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}
    store.add_block("drafts/build_1.py", meta, "ratio", "def ratio(df):\n    return df\n")
    embedded, ref = store.placeable("drafts/build_1.py", "ratio")
    with api_module.SESSION.edit():
        b = api_module.SESSION.add_block("ratio", definition=embedded, library_ref=ref)
    kept = client.post("/api/library/drafts/keep", json={"file": "drafts/build_1.py", "block": "ratio", "to": "mine.py"}).json()
    assert kept["name"] == "mine.py" and b.library_ref["file"] == "mine.py"
    store.add_block("drafts/build_2.py", meta, "other", "def other(df):\n    return df\n")
    assert client.post("/api/library/drafts/discard", json={"file": "drafts/build_2.py"}).status_code == 200
    assert client.post("/api/library/drafts/discard", json={"file": "mine.py"}).status_code == 400


# ---- CLI --------------------------------------------------------------------------------------------


def test_check_blocks_cli(tmp_path, capsys):
    from modelmaker.cli import check_blocks

    d = tmp_path / "lib"
    d.mkdir()
    (d / "pd.py").write_text(GOOD)
    assert check_blocks(d) == 0
    (d / "bad.py").write_text(GOOD.replace("== [2]", "== [9]").replace("double_it", "dbl").replace("halve_it", "hlv"))
    assert check_blocks(d) == 1
    out = capsys.readouterr().out
    assert "FAIL dbl v2" in out and "example failed" in out


def test_streaming_fuses_a_lazy_library_block(tmp_path, block_library_dir):
    csv = tmp_path / "a.csv"
    csv.write_text("a\n1\n2\n")
    store.save_file("pd.py", GOOD)
    embedded, ref = store.placeable("pd.py", "double_it")
    from modelmaker.graph import BlockInstance, Position

    d = BlockInstance("d", "standard", "double_it", "d", None, Position(1, 0), 1, {}, [], [], definition=embedded, library_ref=ref)
    d.inputs, d.outputs = list(d.resolve().inputs), list(d.resolve().outputs)
    graph = Graph(
        blocks={"r": make_block("r", "read_csv", params={"path": str(csv)}), "d": d, "s": make_block("s", "select", params={"cols": ["a"]}, x=2)},
        wires={"w1": Wire("w1", "r", "out", "d", "df"), "w2": Wire("w2", "d", "out", "s", "df")},
    )
    runner = Runner(graph, CacheStore())
    report = runner.run_all_streaming()
    assert report == {"r": "fused", "d": "fused", "s": "green"}, report
    out = runner.cache.get(runner.state["s"].last_successful_key).outputs["out"].data
    assert out["a"].to_list() == [2, 4]


# ---- Create with AI ----------------------------------------------------------------------------


@pytest.fixture
def ai_client(tmp_path, monkeypatch):
    from modelmaker.llm.settings import LLMSettingsStore

    monkeypatch.setenv("MODELMAKER_LLM_PROVIDER", "stub")
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    api_module.LLM_SETTINGS = LLMSettingsStore()
    return TestClient(api_module.app)


def test_create_with_ai_proposes_a_valid_file_and_saves_only_when_asked(ai_client, block_library_dir, tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("a\n1\n")
    read = ai_client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv)}}).json()
    ai_client.post(f"/api/blocks/{read['id']}/refresh")
    draft = ai_client.post("/api/library/ai/draft", json={"instruction": "keep it as is", "context_block": read["id"]}).json()
    assert draft["file"] == "stub_block.py" and draft["block"] == "stub_block" and draft["base_sha"] is None
    assert draft["check"] == {"ok": True, "file_errors": [], "errors": [], "has_example": True}
    assert not (block_library_dir / "stub_block.py").exists()  # nothing written yet

    saved = ai_client.put("/api/library/files/stub_block.py", json={"source": draft["source"], "create": True}).json()
    assert saved["blocks"][0]["id"] == "stub_block"
    assert ai_client.post("/api/library/files/stub_block.py/try", json={}).json()["examples"]["stub_block"]["ok"] is True
    placed = ai_client.post("/api/blocks", json={"library": {"file": "stub_block.py", "id": "stub_block"}}).json()
    assert placed["definition_info"]["source"] == "library"


def test_edit_with_ai_bumps_the_version_and_keeps_the_rest_of_the_file(ai_client, block_library_dir):
    store.save_file("pd.py", GOOD)
    draft = ai_client.post("/api/library/ai/draft", json={"instruction": "tidy", "file": "pd.py", "block": "halve_it"}).json()
    assert draft["entry"]["version"] == 2 and draft["base_sha"] == store.get_file("pd.py").sha
    assert "# comment inside the literal" in draft["source"] and draft["check"]["ok"]


def test_fix_with_ai_needs_a_red_library_block(ai_client, tmp_path):
    filt = ai_client.post("/api/blocks", json={"category": "filter", "params": {"expr": "x"}}).json()
    assert ai_client.post("/api/library/ai/draft", json={"instruction": "", "fix_block": filt["id"]}).status_code == 400


def test_duplicate_a_builtin_into_the_library(client):
    out = client.post("/api/library/duplicate", json={"id": "filter", "new_id": "my_filter", "to": "mine.py"}).json()
    block = out["blocks"][0]
    assert block["id"] == "my_filter" and block["version"] == 1 and block["display_name"] == "Filter (copy)" and not block["errors"]
    assert client.post("/api/library/duplicate", json={"id": "filter", "new_id": "my_filter", "to": "mine.py"}).status_code == 400
