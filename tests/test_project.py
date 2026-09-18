import json

from modelmaker.blocks.base import PortSpec
from modelmaker.graph import Artifact, Graph, Lane, Wire
from modelmaker.project import ensure_project_scaffold, load_project, save_project

from .helpers import make_block


def _build_graph():
    read = make_block("b_001", "read_csv", params={"path": "data/applications.csv"}, lane="lane_data")
    custom = make_block(
        "b_002",
        "bucket_income",
        block_type="standard",
        params={"bins": [0, 30000, 60000]},
        code="def bucket_income(df, bins):\n    return df\n",
        code_version=2,
        lane="lane_feat",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        metadata_transform={"kind": "declared", "base": "df", "adds": [{"name": "income_bucket", "role": "feature"}]},
    )
    custom.port_names = {"out": "income_bucketed"}
    return Graph(
        lanes={"lane_data": Lane("Data Prep", 0), "lane_feat": Lane("Feature Engineering", 1)},
        blocks={"b_001": read, "b_002": custom},
        wires={"w_001": Wire("w_001", "b_001", "out", "b_002", "df")},
    )


def test_save_then_load_round_trips(tmp_path):
    graph = _build_graph()
    project_path = tmp_path / "project.json"
    save_project(graph, project_path, project_name="consumer_pd_model")

    assert project_path.exists()
    sidecar = tmp_path / "blocks" / "b_002.py"
    assert sidecar.exists()
    assert "def bucket_income" in sidecar.read_text()

    loaded = load_project(project_path)
    assert set(loaded.blocks) == {"b_001", "b_002"}
    assert loaded.blocks["b_002"].code == graph.blocks["b_002"].code
    assert loaded.blocks["b_002"].code_version == 2
    assert loaded.blocks["b_002"].metadata_transform["kind"] == "declared"
    assert loaded.blocks["b_002"].port_names == {"out": "income_bucketed"}
    assert loaded.wires["w_001"].from_block == "b_001"
    assert loaded.lanes["lane_feat"].order == 1


def test_save_then_load_round_trips_artifacts(tmp_path):
    graph = _build_graph()
    graph.artifacts["art_001"] = Artifact(
        id="art_001",
        kind="data_analysis",
        title="b_001 :: out -- data analysis",
        block_id="b_001",
        port="out",
        document="# Analysis\nLooks fine.",
        source_key="abc123",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
    )
    project_path = tmp_path / "project.json"
    save_project(graph, project_path)

    loaded = load_project(project_path)

    assert set(loaded.artifacts) == {"art_001"}
    loaded_artifact = loaded.artifacts["art_001"]
    assert loaded_artifact.title == "b_001 :: out -- data analysis"
    assert loaded_artifact.document == "# Analysis\nLooks fine."
    assert loaded_artifact.block_id == "b_001"
    assert loaded_artifact.source_key == "abc123"


def test_saved_json_has_no_run_state_and_is_sorted(tmp_path):
    graph = _build_graph()
    project_path = tmp_path / "project.json"
    save_project(graph, project_path)

    data = json.loads(project_path.read_text())
    assert "state" not in data
    for block in data["blocks"].values():
        assert "status" not in block
        assert "last_successful_key" not in block

    raw = project_path.read_text()
    # sort_keys=True was used -- re-dumping should be byte-identical
    redumped = json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert raw == redumped


def test_save_project_gives_the_folder_a_files_dir_gitignore_and_git_repo(tmp_path):
    graph = _build_graph()
    project_dir = tmp_path / "my_project"
    save_project(graph, project_dir / "model.json")

    assert (project_dir / "files").is_dir()
    assert (project_dir / ".gitignore").exists()
    assert ".modelmaker-cache/" in (project_dir / ".gitignore").read_text()
    assert (project_dir / ".git").is_dir()


def test_ensure_project_scaffold_never_overwrites_an_edited_gitignore(tmp_path):
    ensure_project_scaffold(tmp_path)
    (tmp_path / ".gitignore").write_text("my-own-rule\n")

    ensure_project_scaffold(tmp_path)

    assert (tmp_path / ".gitignore").read_text() == "my-own-rule\n"


def test_ensure_project_scaffold_is_safe_to_call_repeatedly(tmp_path):
    ensure_project_scaffold(tmp_path)
    ensure_project_scaffold(tmp_path)  # must not raise (idempotent git init)
    assert (tmp_path / ".git").is_dir()
