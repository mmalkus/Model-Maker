"""GET/PUT/DELETE /api/env_vars/{name}: lets the UI set an env var (e.g. a
Read SQL block's connection_env) for the running server process without
ever writing it to disk or echoing it back -- see api._env_var_status.
"""

import os

import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module


@pytest.fixture
def client():
    # PUT writes straight into the real process os.environ (that's the
    # point -- see api.set_env_var), so unlike monkeypatch.setenv it isn't
    # auto-reverted; clean up explicitly on both sides of each test so one
    # test's override can't leak into the next.
    api_module._ENV_VAR_OVERRIDES = {}
    os.environ.pop("MM_TEST_DB_URL", None)
    yield TestClient(api_module.app)
    os.environ.pop("MM_TEST_DB_URL", None)


def test_unset_var_reports_not_set(client):
    resp = client.get("/api/env_vars/MM_TEST_NOT_SET_VAR")
    assert resp.json() == {"name": "MM_TEST_NOT_SET_VAR", "is_set": False, "source": None}


def test_set_var_is_never_echoed_back_but_presence_is_reported(client):
    resp = client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://user:super-secret@host/db"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "override"}
    assert "super-secret" not in resp.text

    resp2 = client.get("/api/env_vars/MM_TEST_DB_URL")
    assert resp2.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "override"}
    assert "super-secret" not in resp2.text


def test_set_var_is_visible_to_os_environ(client):
    import os

    client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://x"})
    assert os.environ["MM_TEST_DB_URL"] == "postgresql://x"
    client.delete("/api/env_vars/MM_TEST_DB_URL")


def test_clear_removes_a_var_this_endpoint_set(client):
    client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://x"})
    resp = client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json() == {"name": "MM_TEST_DB_URL", "is_set": False, "source": None}


def test_clear_restores_a_var_that_pre_existed_in_the_environment(client, monkeypatch):
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite://original.db")

    client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://overridden"})
    resp = client.get("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json()["source"] == "override"

    resp2 = client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert resp2.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "env"}

    import os

    assert os.environ["MM_TEST_DB_URL"] == "sqlite://original.db"


def test_clear_leaves_a_pre_existing_env_var_alone_if_never_overridden(client, monkeypatch):
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite://original.db")

    resp = client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "env"}

    import os

    assert os.environ["MM_TEST_DB_URL"] == "sqlite://original.db"


def test_reported_as_env_sourced_when_set_outside_the_endpoint(client, monkeypatch):
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite://from-dotenv.db")
    resp = client.get("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "env"}


def test_rejects_an_invalid_env_var_name(client):
    resp = client.put("/api/env_vars/not-a-valid-name", json={"value": "x"})
    assert resp.status_code == 400
