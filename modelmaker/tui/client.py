from __future__ import annotations

from typing import Any

import httpx

# Thin async wrapper around every modelmaker.api route the TUI needs. Two
# transports:
#  - standalone (default): talks straight to the FastAPI `app` object over
#    an in-process ASGI transport, no real socket -- `modelmaker-tui` works
#    with zero setup, exactly like a single binary.
#  - attach (base_url given): a real HTTP client against an already-running
#    `modelmaker-api`, so the TUI and the web UI can drive the same session
#    live.
# Either way the surface is identical: every route in api.py has a matching
# method here, returning the parsed JSON body (or bytes, for /image).


class APIError(RuntimeError):
    """A non-2xx response from the API, with the server's own detail
    message (FastAPI's {"detail": "..."} body) surfaced as str(err)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except ValueError:
        pass
    return resp.text or f"HTTP {resp.status_code}"


class ModelMakerClient:
    def __init__(self, base_url: str | None = None):
        """base_url=None -> standalone in-process mode (imports the FastAPI
        app directly); otherwise a real http(s)://host:port to attach to."""
        # Run-control endpoints (run_all, force_run_all, ...) block server-side
        # for up to 20s waiting for the run to finish (see api._start_background_run)
        # before falling back to "still running" -- httpx's 5s default would
        # time out well before that on anything but a trivial pipeline.
        timeout = httpx.Timeout(30.0)
        if base_url is None:
            from modelmaker.api import app as _app

            transport = httpx.ASGITransport(app=_app)
            self._client = httpx.AsyncClient(transport=transport, base_url="http://tui.local", timeout=timeout)
        else:
            self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ModelMakerClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _req(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        resp = await self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise APIError(resp.status_code, _detail(resp))
        return resp

    async def _get(self, path: str, **kwargs: Any) -> Any:
        return (await self._req("GET", path, **kwargs)).json()

    async def _post(self, path: str, **kwargs: Any) -> Any:
        return (await self._req("POST", path, **kwargs)).json()

    async def _patch(self, path: str, **kwargs: Any) -> Any:
        return (await self._req("PATCH", path, **kwargs)).json()

    async def _put(self, path: str, **kwargs: Any) -> Any:
        return (await self._req("PUT", path, **kwargs)).json()

    async def _delete(self, path: str, **kwargs: Any) -> Any:
        return (await self._req("DELETE", path, **kwargs)).json()

    # ---- health / registry -------------------------------------------

    async def health(self) -> dict[str, str]:
        return await self._get("/api/health")

    async def registry(self) -> list[dict[str, Any]]:
        return await self._get("/api/registry")

    # ---- graph ----------------------------------------------------------

    async def get_graph(self) -> dict[str, Any]:
        return await self._get("/api/graph")

    async def browse(self, path: str | None = None, ext: str | None = None) -> dict[str, Any]:
        params = {k: v for k, v in {"path": path, "ext": ext}.items() if v is not None}
        return await self._get("/api/browse", params=params)

    async def project_default_dir(self) -> dict[str, str]:
        return await self._get("/api/project/default_dir")

    async def load_project(self, path: str) -> dict[str, Any]:
        return await self._post("/api/project/load", json={"path": path})

    async def save_project(self, path: str | None = None) -> dict[str, Any]:
        return await self._post("/api/project/save", json={"path": path})

    async def recovery_info(self) -> dict[str, Any]:
        return await self._get("/api/project/recovery")

    async def recover(self) -> dict[str, Any]:
        return await self._post("/api/project/recover")

    async def dismiss_recovery(self) -> dict[str, Any]:
        return await self._post("/api/project/recovery/dismiss")

    async def undo(self) -> dict[str, Any]:
        return await self._post("/api/undo")

    async def redo(self) -> dict[str, Any]:
        return await self._post("/api/redo")

    async def set_sample_mode(self, rows: int | None) -> dict[str, Any]:
        return await self._put("/api/sample_mode", json={"rows": rows})

    # ---- blocks -----------------------------------------------------------

    async def create_block(
        self,
        category: str,
        block_type: str | None = None,
        name: str | None = None,
        lane: str | None = None,
        x: float = 0,
        y: float = 0,
        params: dict[str, Any] | None = None,
        code: str | None = None,
        inputs: list[dict] | None = None,
        outputs: list[dict] | None = None,
        metadata_transform: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "category": category,
            "block_type": block_type,
            "name": name,
            "lane": lane,
            "x": x,
            "y": y,
            "params": params or {},
            "code": code,
            "inputs": inputs,
            "outputs": outputs,
            "metadata_transform": metadata_transform,
        }
        return await self._post("/api/blocks", json=body)

    async def update_block(self, block_id: str, **fields: Any) -> dict[str, Any]:
        return await self._patch(f"/api/blocks/{block_id}", json=fields)

    async def set_column_role(self, block_id: str, column: str, role: str) -> dict[str, Any]:
        return await self._post(f"/api/blocks/{block_id}/column_role", json={"column": column, "role": role})

    async def input_schema(self, block_id: str) -> dict[str, list[dict[str, str]]]:
        return await self._get(f"/api/blocks/{block_id}/input_schema")

    async def delete_block(self, block_id: str) -> dict[str, str]:
        return await self._delete(f"/api/blocks/{block_id}")

    async def preview(self, block_id: str, port: str | None = None, rows: int = 20, summary: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {"rows": rows, "summary": summary}
        if port is not None:
            params["port"] = port
        return await self._get(f"/api/blocks/{block_id}/preview", params=params)

    async def image(self, block_id: str, port: str | None = None) -> bytes:
        params = {"port": port} if port is not None else {}
        return (await self._req("GET", f"/api/blocks/{block_id}/image", params=params)).content

    async def value(self, block_id: str, port: str | None = None) -> Any:
        params = {"port": port} if port is not None else {}
        return await self._get(f"/api/blocks/{block_id}/value", params=params)

    async def rename_port(self, block_id: str, port: str, name: str | None) -> dict[str, Any]:
        return await self._patch(f"/api/blocks/{block_id}/port_name", json={"port": port, "name": name})

    # ---- wires -----------------------------------------------------------

    async def create_wire(self, from_block: str, from_port: str, to_block: str, to_port: str) -> dict[str, Any]:
        body = {"from_block": from_block, "from_port": from_port, "to_block": to_block, "to_port": to_port}
        return await self._post("/api/wires", json=body)

    async def delete_wire(self, wire_id: str) -> dict[str, str]:
        return await self._delete(f"/api/wires/{wire_id}")

    # ---- lanes -----------------------------------------------------------

    async def upsert_lane(self, lane_id: str, name: str, order: int, height: float | None = None) -> dict[str, Any]:
        return await self._put("/api/lanes", json={"id": lane_id, "name": name, "order": order, "height": height})

    async def delete_lane(self, lane_id: str) -> dict[str, Any]:
        return await self._delete(f"/api/lanes/{lane_id}")

    # ---- run control -------------------------------------------------------

    async def run_block(self, block_id: str) -> dict[str, Any]:
        return await self._post(f"/api/blocks/{block_id}/run")

    async def run_to_here(self, block_id: str) -> dict[str, Any]:
        return await self._post(f"/api/blocks/{block_id}/run_to_here")

    async def refresh(self, block_id: str) -> dict[str, Any]:
        return await self._post(f"/api/blocks/{block_id}/refresh")

    async def check_changes(self, block_id: str) -> dict[str, Any]:
        return await self._post(f"/api/blocks/{block_id}/check_changes")

    async def run_all(self) -> dict[str, Any]:
        return await self._post("/api/run_all")

    async def force_run_all(self) -> dict[str, Any]:
        return await self._post("/api/force_run_all")

    async def run_all_streaming(self) -> dict[str, Any]:
        return await self._post("/api/run_all_streaming")

    async def refresh_all(self) -> dict[str, Any]:
        return await self._post("/api/refresh_all")

    async def cancel_run(self) -> dict[str, bool]:
        return await self._post("/api/run/cancel")

    async def check_all_sources(self) -> dict[str, bool]:
        return await self._post("/api/check_all_sources")

    # ---- compile -----------------------------------------------------------

    async def compile(self, output_blocks: list[str] | None = None, strict: bool = True) -> dict[str, str]:
        return await self._post("/api/compile", json={"output_blocks": output_blocks, "strict": strict})

    # ---- LLM ---------------------------------------------------------------

    async def llm_settings(self) -> dict[str, Any]:
        return await self._get("/api/llm/settings")

    async def update_llm_settings(
        self,
        active_provider: str | None = None,
        include_reference: bool | None = None,
        settings: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body = {"active_provider": active_provider, "include_reference": include_reference, "settings": settings}
        return await self._put("/api/llm/settings", json=body)

    async def llm_models(self, provider: str, base_url: str | None = None) -> dict[str, list[str]]:
        params = {"provider": provider}
        if base_url is not None:
            params["base_url"] = base_url
        return await self._get("/api/llm/models", params=params)

    async def draft(self, block_id: str, instruction: str, provider: str | None = None) -> dict[str, Any]:
        return await self._post(f"/api/blocks/{block_id}/draft", json={"instruction": instruction, "provider": provider})

    async def suggest_fix(self, block_id: str, instruction: str = "", provider: str | None = None) -> dict[str, Any]:
        return await self._post(
            f"/api/blocks/{block_id}/suggest_fix", json={"instruction": instruction, "provider": provider}
        )
