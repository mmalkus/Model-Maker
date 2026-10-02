"""Phase 0 of the AI model builder (see /agent-builder-proposal.md): the
pieces the agent relies on that aren't themselves AI -- grouped undo,
block provenance, the shared run slot, LLM-bound stat redaction, and the
block catalogue."""

from __future__ import annotations

import json
import threading
import time

import pytest

from modelmaker.agent.build import ToolError
from modelmaker.agent.catalogue import (
    AGENT_DISALLOWED,
    AGENT_TAGS,
    describe_block_type,
    list_block_types,
    list_blocks_for_tag,
)
from modelmaker.agent.loop import LMStudioLoop
from modelmaker.agent.tools import TOOLS
from modelmaker.blocks.base import BLOCK_REGISTRY
from modelmaker.llm.redact import column_info_for_llm
from modelmaker.packet import ColumnMeta, ColumnRole, ColumnStats
from modelmaker.project import graph_from_dict, graph_to_dict
from modelmaker.runslot import RunBusy, RunFailed, RunSlot
from modelmaker.session import ProjectSession


@pytest.fixture
def session(tmp_path):
    return ProjectSession(recovery_path=tmp_path / "recovery.json")


# ---- transaction ------------------------------------------------------


def test_edits_inside_a_transaction_undo_as_one_step(session):
    with session.edit():
        keep = session.add_block("filter", params={"expr": "a > 1"})
    with session.transaction():
        for i in range(5):
            with session.edit():
                session.add_block("select", params={"cols": [f"c{i}"]})
    assert len(session.graph.blocks) == 6

    assert session.undo() is True
    assert set(session.graph.blocks) == {keep.id}
    assert session.undo() is True
    assert session.graph.blocks == {}


def test_an_empty_transaction_leaves_no_undo_point(session):
    with session.transaction():
        pass
    assert session.can_undo is False


def test_a_failing_transaction_keeps_its_edits_revertible(session):
    with pytest.raises(RuntimeError):
        with session.transaction():
            with session.edit():
                session.add_block("filter", params={"expr": "a > 1"})
            raise RuntimeError("build stopped")
    assert len(session.graph.blocks) == 1
    assert session.undo() is True
    assert session.graph.blocks == {}


def test_nested_transactions_record_once(session):
    with session.transaction():
        with session.edit():
            session.add_block("filter", params={"expr": "a > 1"})
        with session.transaction():
            with session.edit():
                session.add_block("filter", params={"expr": "a > 2"})
    assert session.undo() is True
    assert session.graph.blocks == {}
    assert session.can_undo is False


def test_restore_snapshot_is_itself_undoable(session):
    with session.transaction() as before:
        with session.edit():
            session.add_block("filter", params={"expr": "a > 1"})
    session.restore_snapshot(before)
    assert session.graph.blocks == {}
    assert session.undo() is True
    assert len(session.graph.blocks) == 1


# ---- provenance -------------------------------------------------------


def test_provenance_round_trips_and_is_omitted_when_unset(session):
    with session.edit():
        ai = session.add_block("filter", params={"expr": "a > 1"})
        plain = session.add_block("filter", params={"expr": "a > 2"})
    ai.provenance = {"source": "agent", "build_id": "b_1", "modified_by_user": False}

    data = graph_to_dict(session.graph)
    assert data["blocks"][ai.id]["provenance"]["build_id"] == "b_1"
    assert "provenance" not in data["blocks"][plain.id]

    restored = graph_from_dict(data)
    assert restored.blocks[ai.id].provenance == ai.provenance
    assert restored.blocks[plain.id].provenance is None


def test_provenance_does_not_change_the_cache_key(session):
    with session.edit():
        block = session.add_block("filter", params={"expr": "a > 1"})
    before = session.runner.compute_key(block.id)
    block.provenance = {"source": "agent", "build_id": "b_1"}
    assert session.runner.compute_key(block.id) == before


def test_user_edits_flag_an_ai_built_block_but_agent_edits_do_not(session):
    with session.edit():
        block = session.add_block("filter", params={"expr": "a > 1"})
    block.provenance = {"source": "agent", "build_id": "b_1", "modified_by_user": False}

    session.update_block(block.id, actor="agent", params={"expr": "a > 2"})
    assert block.provenance["modified_by_user"] is False
    session.update_block(block.id, name="renamed")  # not a params/code change
    assert block.provenance["modified_by_user"] is False
    session.update_block(block.id, params={"expr": "a > 3"})
    assert block.provenance["modified_by_user"] is True


def test_user_edits_leave_hand_built_blocks_without_provenance(session):
    with session.edit():
        block = session.add_block("filter", params={"expr": "a > 1"})
    session.update_block(block.id, params={"expr": "a > 2"})
    assert block.provenance is None


# ---- run slot -----------------------------------------------------------


def test_run_slot_rejects_a_second_run_while_one_is_active():
    slot = RunSlot()
    release = threading.Event()
    assert slot.start_background(release.wait, wait_seconds=0.05) is None
    assert slot.busy()
    with pytest.raises(RunBusy):
        slot.run_sync(lambda: None)
    release.set()
    for _ in range(100):
        if not slot.busy():
            break
        time.sleep(0.01)
    assert slot.run_sync(lambda: 42) == 42


def test_run_slot_sync_run_blocks_background_runs():
    slot = RunSlot()
    seen = {}

    def inner():
        with pytest.raises(RunBusy):
            slot.start_background(lambda: None)
        seen["ok"] = True

    slot.run_sync(inner)
    assert seen == {"ok": True}
    assert not slot.busy()


def test_run_slot_surfaces_a_fast_failure():
    slot = RunSlot()

    def boom():
        raise ValueError("upstream isn't green")

    with pytest.raises(RunFailed, match="upstream isn't green"):
        slot.start_background(boom)
    assert slot.take_error() is None


# ---- redaction ------------------------------------------------------------


def _stats(lo, hi):
    return ColumnStats(count=10, null_count=0, n_unique=10, min=lo, max=hi)


def test_redaction_keeps_numeric_ranges():
    info = column_info_for_llm("income", ColumnMeta(dtype="Float64", role=ColumnRole.FEATURE), _stats(1.0, 9.0))
    assert (info.min, info.max, info.n_unique) == (1.0, 9.0, 10)


def test_redaction_drops_text_min_max():
    info = column_info_for_llm("name", ColumnMeta(dtype="String", role=ColumnRole.UNASSIGNED), _stats("Aa", "Zz"))
    assert (info.min, info.max) == (None, None)
    assert info.n_unique == 10


def test_redaction_drops_id_min_max_whatever_the_dtype():
    info = column_info_for_llm("customer_id", ColumnMeta(dtype="Int64", role=ColumnRole.ID), _stats(1001, 9999))
    assert (info.min, info.max) == (None, None)


def test_analyze_data_prompt_never_contains_text_values(tmp_path, monkeypatch):
    """End to end through the real endpoint: the prompt analyze_data sends
    must not contain a string column's literal min/max."""
    import polars as pl
    from fastapi.testclient import TestClient

    from modelmaker import api
    from modelmaker.llm.base import DraftResult, LLMProvider

    csv = tmp_path / "people.csv"
    pl.DataFrame({"name": ["Zelda Secret", "Aaron Hidden", "Mia"], "age": [30, 40, 50]}).write_csv(csv)

    captured = {}

    class Capture(LLMProvider):
        def draft(self, ctx):
            captured["ctx"] = ctx
            return DraftResult(explanation="ok")

    monkeypatch.setattr(api, "SESSION", ProjectSession(recovery_path=tmp_path / "r.json"))
    monkeypatch.setattr(api, "_provider_for", lambda name: Capture())
    client = TestClient(api.app)
    block = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv)}}).json()
    assert client.post(f"/api/blocks/{block['id']}/run").status_code == 200
    assert client.post(f"/api/blocks/{block['id']}/analyze_data", json={}).status_code == 200

    from modelmaker.llm.prompts import build_user_prompt

    prompt = build_user_prompt(captured["ctx"])
    assert "Zelda Secret" not in prompt and "Aaron Hidden" not in prompt
    assert "min=30" in prompt  # numeric ranges still go through


# ---- catalogue ------------------------------------------------------------


def test_every_block_has_a_complete_catalogue_entry():
    listed = {e["category"] for e in list_block_types(include_disallowed=True)}
    assert listed == set(BLOCK_REGISTRY)
    for category in BLOCK_REGISTRY:
        entry = describe_block_type(category)
        assert entry["summary"], f"{category} has no docstring summary"
        for p in entry["params"]:
            assert p["name"] not in {"output_dir", "block_id", "sample_rows"}


def test_disallowed_blocks_are_hidden_by_default():
    listed = {e["category"] for e in list_block_types()}
    assert not listed & set(AGENT_DISALLOWED)
    assert "logistic_regression" in listed


def test_every_addable_block_is_tagged():
    for e in list_block_types():
        assert e["tags"], f"{e['category']} has no tag in AGENT_TAGS"
    for tag, (_, categories) in AGENT_TAGS.items():
        for category in categories:
            assert category in BLOCK_REGISTRY, f"tag {tag} names unknown block {category}"
            assert category not in AGENT_DISALLOWED, f"tag {tag} names {category}, which an AI build can't add"


def test_every_tag_listing_fits_a_local_models_tool_result():
    list_tool = TOOLS["list_block_types"].fn
    assert len(json.dumps(list_tool(None))) < LMStudioLoop.max_result_chars / 2
    for tag in AGENT_TAGS:
        assert len(json.dumps(list_tool(None, tag=tag))) < LMStudioLoop.max_result_chars / 2, tag


def test_list_block_types_by_tag():
    list_tool = TOOLS["list_block_types"].fn
    assert {t["tag"] for t in list_tool(None)["tags"]} == set(AGENT_TAGS)
    pd = {e["category"]: e["summary"] for e in list_blocks_for_tag("pd")}
    assert "logistic_regression" in pd and "auc_gini" not in pd
    assert pd["grade_backtest"] == "Grade-level PD back-test"  # trimmed to the first sentence
    with pytest.raises(ToolError, match="unknown tag 'modelling'"):
        list_tool(None, tag="modelling")


def test_role_bound_params_are_marked():
    params = {p["name"]: p for p in describe_block_type("auc_gini")["params"]}
    assert params["target_col"]["auto_fills_from_role"] == "target"
    assert params["score_col"]["auto_fills_from_role"] == "predicted"
    lr = {p["name"]: p for p in describe_block_type("logistic_regression")["params"]}
    assert lr["target"]["auto_fills_from_role"] == "target"
    assert lr["C"]["default"] == 1.0
