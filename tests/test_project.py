import json

from modelmaker.blocks.base import PortSpec
from modelmaker.graph import Graph, Lane, Wire
from modelmaker.project import load_project, save_project

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
    assert loaded.wires["w_001"].from_block == "b_001"
    assert loaded.lanes["lane_feat"].order == 1


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
