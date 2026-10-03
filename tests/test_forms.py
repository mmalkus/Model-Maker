"""Params forms (modelmaker/forms.py): each registry block's BlockSpec.form
plus fields inferred from its signature, served with the block to the web
UI and the TUI -- and the TUI's reading of them."""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker.agent.catalogue import INJECTED_PARAMS
from modelmaker.blocks.base import BLOCK_REGISTRY, BlockSpec, FieldSpec, PortSpec, register_block
from modelmaker.forms import block_form, infer_field
from modelmaker.session import ProjectSession
from modelmaker.tui.paramspecs import field_specs_for, json_param_keys


def _fields(category: str) -> dict[str, dict]:
    return {f["key"]: f for f in block_form(BLOCK_REGISTRY[category])["fields"]}


def test_every_param_has_a_field_or_is_edited_as_json():
    for category, spec in BLOCK_REGISTRY.items():
        form = block_form(spec)
        ports = {p.name for p in spec.inputs}
        params = {
            name
            for name, p in inspect.signature(spec.fn).parameters.items()
            if name not in ports and name not in INJECTED_PARAMS and p.kind is not p.VAR_KEYWORD
        }
        keys = [f["key"] for f in form["fields"]]
        assert len(keys) == len(set(keys)), category
        assert set(keys) | set(form["other_params"]) == params, category
        assert not set(keys) & set(form["other_params"]), category


def test_declared_fields_come_first_and_the_rest_are_inferred():
    fields = list(_fields("forward_default_flag").values())
    assert [f["key"] for f in fields][:3] == ["id_col", "date_col", "default_col"]
    by_key = {f["key"]: f for f in fields}
    assert by_key["incomplete"]["kind"] == "select" and by_key["incomplete"]["options"] == ["drop", "null"]
    # Not declared: inferred from `performing_only: bool = True`.
    assert by_key["performing_only"]["kind"] == "checkbox" and by_key["performing_only"]["default"] is True
    assert by_key["horizon"]["default"] == 12


def test_inference_rules():
    assert infer_field("ead_col", "str").kind == "column"
    assert infer_field("target", "str", "target").auto_role == "target"
    assert infer_field("features", "list[str] | None").kind == "columns"
    assert infer_field("seed", "int").kind == "number"
    assert infer_field("floor", "float | None").kind == "number"
    assert infer_field("keep_paths", "bool").kind == "checkbox"
    assert infer_field("title", "str").kind == "text"
    assert infer_field("aggs", "dict[str, str]") is None
    assert infer_field("alpha_levels", "list[float] | None") is None

    fit = _fields("fit_binning")
    assert fit["target"]["auto_role"] == "target"  # the runner auto-fills it from the target role
    assert fit["monotonic"]["kind"] == "checkbox"
    assert block_form(BLOCK_REGISTRY["groupby_agg"])["other_params"] == ["aggs"]


def test_a_form_field_must_be_a_real_param():
    def fn(df, a: int = 1):
        return df

    with pytest.raises(ValueError, match="aren't params"):
        register_block(
            BlockSpec(
                category="_bad_form",
                block_type="standard",
                display_name="bad",
                inputs=[PortSpec("df")],
                outputs=[PortSpec("out")],
                fn=fn,
                metadata_transform=lambda *_a, **_k: {},
                form=(FieldSpec("b", "B", "number"),),
            )
        )
    assert "_bad_form" not in BLOCK_REGISTRY


@pytest.fixture
def client(tmp_path):
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    return TestClient(api_module.app)


def test_forms_are_served_with_the_registry_and_each_block(client):
    registry = {b["category"]: b for b in client.get("/api/registry").json()}
    assert registry["join"]["form"]["fields"][0]["label"] == "Join on"

    block = client.post("/api/blocks", json={"category": "groupby_agg"}).json()
    assert block["form"] == block_form(BLOCK_REGISTRY["groupby_agg"])
    custom = client.post(
        "/api/blocks",
        json={
            "category": "mine",
            "block_type": "standard",
            "inputs": [{"name": "df", "type": "dataframe"}],
            "outputs": [{"name": "out", "type": "dataframe"}],
            "metadata_transform": {"kind": "passthrough"},
            "code": "def mine(df, amount_col: str = 'a'):\n    return df\n",
            "params": {"amount_col": "a"},
        },
    ).json()
    assert custom["form"] is None


def test_tui_reads_the_served_form():
    block = {"category": "groupby_agg", "is_custom": False, "params": {}, "form": block_form(BLOCK_REGISTRY["groupby_agg"])}
    assert [f.key for f in field_specs_for(block)] == ["by"]
    assert json_param_keys(block) == ["aggs"]

    custom = {"category": "mine", "is_custom": True, "params": {"amount_col": "a", "rate": 0.1}, "form": None}
    assert [(f.key, f.kind) for f in field_specs_for(custom)] == [("amount_col", "column")]
    assert json_param_keys(custom) is None
