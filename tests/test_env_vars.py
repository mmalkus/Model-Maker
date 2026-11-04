"""GET/PUT/DELETE /api/env_vars/{name}: lets the UI set an env var (e.g. a
Read SQL block's connection_env) for the workspace's blocks without ever
writing it to disk, into the server's own os.environ, or echoing it back --
see api._env_var_status.
"""

import os

import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module


@pytest.fixture
def client():
    # PUT stores the value in the default workspace (never os.environ), which
    # lives as long as the module -- clear it on both sides of each test so
    # one test's override can't leak into the next.
    api_module.DEFAULT_WORKSPACE.env_vars.clear()
    os.environ.pop("MM_TEST_DB_URL", None)
    yield TestClient(api_module.app)
    api_module.DEFAULT_WORKSPACE.env_vars.clear()
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


def test_set_var_is_stored_in_the_workspace_not_os_environ(client):
    client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://x"})
    assert "MM_TEST_DB_URL" not in os.environ
    assert api_module.DEFAULT_WORKSPACE.env_vars.get("MM_TEST_DB_URL") == "postgresql://x"
    client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert "MM_TEST_DB_URL" not in api_module.DEFAULT_WORKSPACE.env_vars


def test_clear_removes_a_var_this_endpoint_set(client):
    client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://x"})
    resp = client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json() == {"name": "MM_TEST_DB_URL", "is_set": False, "source": None}


def test_clear_restores_a_var_that_pre_existed_in_the_environment(client, monkeypatch):
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite://original.db")

    client.put("/api/env_vars/MM_TEST_DB_URL", json={"value": "postgresql://overridden"})
    resp = client.get("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json()["source"] == "override"
    # The server's own environment is never touched, only the workspace's view.
    assert os.environ["MM_TEST_DB_URL"] == "sqlite://original.db"

    resp2 = client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert resp2.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "env"}

    assert os.environ["MM_TEST_DB_URL"] == "sqlite://original.db"


def test_clear_leaves_a_pre_existing_env_var_alone_if_never_overridden(client, monkeypatch):
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite://original.db")

    resp = client.delete("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "env"}

    assert os.environ["MM_TEST_DB_URL"] == "sqlite://original.db"


def test_reported_as_env_sourced_when_set_outside_the_endpoint(client, monkeypatch):
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite://from-dotenv.db")
    resp = client.get("/api/env_vars/MM_TEST_DB_URL")
    assert resp.json() == {"name": "MM_TEST_DB_URL", "is_set": True, "source": "env"}


def test_rejects_an_invalid_env_var_name(client):
    resp = client.put("/api/env_vars/not-a-valid-name", json={"value": "x"})
    assert resp.status_code == 400
