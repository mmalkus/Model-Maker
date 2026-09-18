from __future__ import annotations

import json
import threading
import time

import pytest

from modelmaker.session import ProjectSession


@pytest.fixture
def session(tmp_path):
    return ProjectSession(recovery_path=tmp_path / "recovery.json")


def test_undo_restores_a_deleted_block_and_its_wires(session):
    with session.edit():
        read = session.add_block("read_csv", params={"path": "data.csv"})
    with session.edit():
        filt = session.add_block("filter", params={"expr": "a > 1"})
    with session.edit():
        session.add_wire(read.id, "out", filt.id, "df")

    with session.edit():
        session.delete_block(filt.id)
    assert filt.id not in session.graph.blocks
    assert session.graph.wires == {}

    assert session.undo() is True
    assert filt.id in session.graph.blocks
    assert len(session.graph.wires) == 1
    assert session.graph.blocks[filt.id].params == {"expr": "a > 1"}


def test_redo_reapplies_an_undone_edit(session):
    with session.edit():
        block = session.add_block("filter", params={"expr": "a > 1"})
    with session.edit():
        session.update_block(block.id, params={"expr": "a > 99"})

    session.undo()
    assert session.graph.blocks[block.id].params == {"expr": "a > 1"}
    assert session.redo() is True
    assert session.graph.blocks[block.id].params == {"expr": "a > 99"}


def test_a_new_edit_clears_the_redo_stack(session):
    with session.edit():
        session.add_block("filter", params={"expr": "a > 1"})
    session.undo()
    assert session.can_redo is True
    with session.edit():
        session.add_block("select", params={"cols": ["a"]})
    assert session.can_redo is False


def test_a_rejected_edit_leaves_no_undo_point(session):
    with session.edit():
        a = session.add_block("read_csv", params={"path": "x.csv"})
    with session.edit():
        b = session.add_block("filter", params={"expr": "a > 1"})
    with session.edit():
        session.add_wire(a.id, "out", b.id, "df")
    undo_depth = len(session._undo)

    # A wire that would close a loop is refused -- and must not leave a
    # do-nothing entry behind for the user to step back through.
    with pytest.raises(ValueError):
        with session.edit():
            session.add_wire(b.id, "out", a.id, "df")
    assert len(session._undo) == undo_depth


def test_undo_keeps_cached_run_state(session, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n2\n")
    with session.edit():
        read = session.add_block("read_csv", params={"path": str(csv)})
    session.runner.refresh(read.id)
    assert session.runner.status(read.id) == "green"

    with session.edit():
        session.add_block("filter", params={"expr": "a > 1"})
    session.undo()

    # Undoing an unrelated edit must not cost the user a re-read.
    assert session.runner.status(read.id) == "green"


def test_dirty_clears_on_save_and_returns_on_the_next_edit(session, tmp_path):
    assert session.dirty is False
    with session.edit():
        session.add_block("filter", params={"expr": "a > 1"})
    assert session.dirty is True

    session.save(tmp_path / "proj.json")
    assert session.dirty is False

    with session.edit():
        session.add_block("select", params={"cols": ["a"]})
    assert session.dirty is True


def test_recovery_snapshot_is_written_on_every_edit_but_never_the_project_file(session, tmp_path):
    project = tmp_path / "proj.json"
    with session.edit():
        session.add_block("filter", params={"expr": "a > 1"})
    session.save(project)
    before = project.read_text(encoding="utf-8")

    with session.edit():
        session.add_block("select", params={"cols": ["a"]})

    # The edit lands in the recovery snapshot...
    info = session.recovery_info()
    assert info is not None and info["block_count"] == 2
    # ...and emphatically not in the user's versioned project file.
    assert project.read_text(encoding="utf-8") == before


def test_recover_rebuilds_the_graph_from_the_snapshot(session, tmp_path):
    with session.edit():
        session.add_block("filter", params={"expr": "a > 1"})
    saved = json.loads((tmp_path / "recovery.json").read_text(encoding="utf-8"))
    assert len(saved["graph"]["blocks"]) == 1

    fresh = ProjectSession(recovery_path=tmp_path / "recovery.json")
    assert fresh.graph.blocks == {}
    fresh.recover()
    assert len(fresh.graph.blocks) == 1


def test_recovery_info_is_none_when_there_is_nothing_worth_recovering(tmp_path):
    assert ProjectSession(recovery_path=tmp_path / "nope.json").recovery_info() is None


def test_group_by_survives_a_save_load_round_trip(session, tmp_path):
    with session.edit():
        block = session.add_block("filter", params={"expr": "a > 1"})
        session.update_block(block.id, group_by="region", max_workers=3)
    path = tmp_path / "proj.json"
    session.save(path)

    reloaded = ProjectSession(recovery_path=None)
    reloaded.load(path)
    assert reloaded.graph.blocks[block.id].group_by == "region"
    assert reloaded.graph.blocks[block.id].max_workers == 3


def test_undo_does_not_roll_back_runs_that_happened_after_the_edit(session, tmp_path):
    """Regression: undo used to restore a wholesale snapshot of run state,
    so an edit made while a run was still in flight would, on undo, drop
    every result that run had produced -- blocks that were green went back
    to never-having-run."""
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n2\n3\n")
    with session.edit():
        read = session.add_block("read_csv", params={"path": str(csv)})
    with session.edit():
        filt = session.add_block("filter", params={"expr": "a > 1"})
    with session.edit():
        session.add_wire(read.id, "out", filt.id, "df")
    session.runner.refresh(read.id)

    # An edit recorded before the downstream block has ever run...
    with session.edit():
        session.update_block(filt.id, params={"expr": "a > 2"})
    # ...and the run lands afterwards.
    session.runner.run_block(filt.id)
    assert session.runner.status(filt.id) == "green"

    session.undo()
    # The params are back, and because status derives from the cache key,
    # that is enough: the block is not stranded as never-run.
    assert session.graph.blocks[filt.id].params == {"expr": "a > 1"}
    assert session.runner.status(filt.id) != "grey"
    session.runner.run_block(filt.id)
    assert session.runner.status(filt.id) == "green"


def test_undoing_a_param_edit_returns_the_block_to_green(session, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n2\n3\n")
    with session.edit():
        read = session.add_block("read_csv", params={"path": str(csv)})
    with session.edit():
        filt = session.add_block("filter", params={"expr": "a > 1"})
    with session.edit():
        session.add_wire(read.id, "out", filt.id, "df")
    session.runner.refresh(read.id)
    session.runner.run_block(filt.id)

    with session.edit():
        session.update_block(filt.id, params={"expr": "a > 2"})
    assert session.runner.status(filt.id) == "orange"

    # Reverting the config reaches the key that was already run and cached.
    session.undo()
    assert session.runner.status(filt.id) == "green"


def test_undo_while_a_run_is_in_flight_does_not_disturb_it(session, tmp_path):
    """Undo replaces the graph object wholesale. A run pinned to the old one
    (see Runner.pin) carries on against its own copy rather than finding the
    blocks swapped underneath it mid-execution."""
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n2\n3\n")
    with session.edit():
        read = session.add_block("read_csv", params={"path": str(csv)})
    with session.edit():
        slow = session.add_block(
            "slow_block",
            block_type="standard",
            inputs=[{"name": "df", "type": "dataframe"}],
            outputs=[{"name": "out", "type": "dataframe"}],
            code="def slow_block(df):\n    import time\n    time.sleep(1.5)\n    return df\n",
            metadata_transform={"kind": "passthrough"},
        )
    with session.edit():
        session.add_wire(read.id, "out", slow.id, "df")
    session.runner.refresh(read.id)

    errors: list[BaseException] = []

    def _run():
        try:
            session.runner.run_all()
        except BaseException as e:  # noqa: BLE001 -- recorded so the test can assert on it
            errors.append(e)

    thread = threading.Thread(target=_run)
    thread.start()
    deadline = time.monotonic() + 20
    while not session.runner.is_running() and time.monotonic() < deadline:
        time.sleep(0.02)
    session.undo()  # removes the wire under the running block
    thread.join(timeout=60)

    assert not thread.is_alive()
    assert not errors, errors
    # The run completed against the graph it was pinned to.
    assert session.runner.state[slow.id].last_successful_key is not None
    assert session.graph.wires == {}


def test_new_discards_the_graph_and_project_identity(session, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a\n1\n")
    with session.edit():
        session.add_block("read_csv", params={"path": str(csv)})
    project_path = tmp_path / "project.json"
    session.save(project_path)
    assert session.project_path == project_path
    assert session.graph.blocks

    session.new()

    assert session.graph.blocks == {}
    assert session.project_path is None
    assert session.project_name == "untitled"
    assert session.can_undo is False
    assert session.can_redo is False
    assert session.dirty is False
    # The saved file itself is untouched -- New only affects in-memory state.
    assert project_path.exists()


def test_new_clears_the_recovery_snapshot(session, tmp_path):
    with session.edit():
        session.add_block("read_csv", params={"path": "data.csv"})
    assert session.recovery_info() is not None

    session.new()

    assert session.recovery_info() is None


def test_upsert_data_analysis_artifact_creates_then_updates_in_place(session):
    with session.edit():
        block = session.add_block("read_csv", params={"path": "data.csv"})

    with session.edit():
        first = session.upsert_data_analysis_artifact(block.id, "out", "First title", "first document")
    assert len(session.graph.artifacts) == 1
    assert first.kind == "data_analysis"

    with session.edit():
        second = session.upsert_data_analysis_artifact(block.id, "out", "Second title", "second document")

    # same (block, port) -- refreshed in place, not duplicated
    assert len(session.graph.artifacts) == 1
    assert second.id == first.id
    assert second.title == "Second title"
    assert second.document == "second document"
    assert second.updated_at >= first.created_at


def test_upsert_data_analysis_artifact_is_independent_per_port(session):
    with session.edit():
        block = session.add_block("read_csv", params={"path": "data.csv"})

    with session.edit():
        session.upsert_data_analysis_artifact(block.id, "out", "A", "doc a")
        session.upsert_data_analysis_artifact(block.id, "other_port", "B", "doc b")

    assert len(session.graph.artifacts) == 2


def test_rename_and_delete_artifact(session):
    with session.edit():
        block = session.add_block("read_csv", params={"path": "data.csv"})
    with session.edit():
        artifact = session.upsert_data_analysis_artifact(block.id, "out", "Title", "doc")

    with session.edit():
        renamed = session.rename_artifact(artifact.id, "New title")
    assert renamed.title == "New title"
    assert session.graph.artifacts[artifact.id].title == "New title"

    with session.edit():
        session.delete_artifact(artifact.id)
    assert artifact.id not in session.graph.artifacts


def test_artifact_is_stale_once_the_source_block_changes(session):
    with session.edit():
        block = session.add_block("read_csv", params={"path": "data.csv"})
    with session.edit():
        artifact = session.upsert_data_analysis_artifact(block.id, "out", "Title", "doc")

    assert session.artifact_is_stale(artifact) is False

    with session.edit():
        session.update_block(block.id, params={"path": "other.csv"})

    assert session.artifact_is_stale(artifact) is True


def test_artifact_is_stale_when_its_block_no_longer_exists(session):
    with session.edit():
        block = session.add_block("read_csv", params={"path": "data.csv"})
    with session.edit():
        artifact = session.upsert_data_analysis_artifact(block.id, "out", "Title", "doc")

    with session.edit():
        session.delete_block(block.id)

    assert session.artifact_is_stale(artifact) is True


def test_list_artifacts_sorts_newest_updated_first_and_reports_block_name(session):
    with session.edit():
        block = session.add_block("read_csv", name="My source", params={"path": "data.csv"})
    with session.edit():
        older = session.upsert_data_analysis_artifact(block.id, "out", "Older", "doc")
        older.updated_at = "2020-01-01T00:00:00+00:00"
        newer = session.upsert_data_analysis_artifact(block.id, "other_port", "Newer", "doc")
        newer.updated_at = "2030-01-01T00:00:00+00:00"

    listed = session.list_artifacts()

    assert [a["id"] for a in listed] == [newer.id, older.id]
    assert listed[0]["block_name"] == "My source"
    assert listed[0]["stale"] is False
