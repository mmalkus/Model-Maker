"""The conversation loop that drives an AI build -- one AgentLoop per phase
(plan, build), each possibly a different provider (see
/agent-builder-proposal.md §9). A loop holds its own conversation state so
it can be resumed: with the user's plan feedback, or their answer to an
ask_user question. Every tool call goes through AgentBuild.call_tool,
whichever loop makes it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .build import AgentBuild
from .tools import Tool


class LLMUnavailable(RuntimeError):
    """The model's endpoint timed out or couldn't be reached. Not the
    build's fault and often transient (a slow local model, a server still
    loading), so the controller pauses the build for a retry instead of
    failing it."""


@dataclass
class LoopOutcome:
    final_text: str = ""
    error: str | None = None


class AgentLoop(ABC):
    @abstractmethod
    def start(self, build: AgentBuild, system: str, prompt: str, tools: list[Tool]) -> LoopOutcome: ...

    @abstractmethod
    def resume(self, build: AgentBuild, prompt: str, tools: list[Tool]) -> LoopOutcome: ...

    def cancel(self) -> None:
        """Called from another thread on Stop. Loops also check
        build.stop_requested between tool calls."""


# ---- scripted (tests) --------------------------------------------------------

Turn = Callable[[Callable[[str, dict], dict]], str]


class ScriptedLoop(AgentLoop):
    """Replays scripted turns instead of calling a model -- each turn is a
    function given `call(name, args) -> result` that makes whatever tool
    calls it wants and returns the turn's final text. start() plays the
    first turn, each resume() the next."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.prompts: list[str] = []

    def _play(self, build: AgentBuild, prompt: str) -> LoopOutcome:
        self.prompts.append(prompt)
        if not self.turns:
            return LoopOutcome(final_text="(script exhausted)")
        turn = self.turns.pop(0)
        text = turn(build.call_tool)
        if text:
            build.log("assistant", text=text)
        return LoopOutcome(final_text=text or "")

    def start(self, build, system, prompt, tools):
        return self._play(build, prompt)

    def resume(self, build, prompt, tools):
        return self._play(build, prompt)


# ---- Anthropic Messages API -------------------------------------------------------

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 8000


class AnthropicLoop(AgentLoop):
    """Manual tool-use loop over the Messages API -- not the SDK's tool
    runner, so stop/limits/logging stay in AgentBuild.call_tool."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        import anthropic

        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL", ANTHROPIC_DEFAULT_MODEL)
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.system = ""
        self.messages: list[dict[str, Any]] = []

    def start(self, build, system, prompt, tools):
        self.system = system
        self.messages = [{"role": "user", "content": prompt}]
        return self._run(build, tools)

    def resume(self, build, prompt, tools):
        # A turn that ended on a terminal tool leaves a trailing user turn
        # (the tool results) -- add the new text to it rather than sending
        # two user turns in a row.
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append({"type": "text", "text": prompt})
        else:
            self.messages.append({"role": "user", "content": prompt})
        return self._run(build, tools)

    def _run(self, build: AgentBuild, tools: list[Tool]) -> LoopOutcome:
        defs = [t.definition() for t in tools]
        if defs:
            defs[-1] = {**defs[-1], "cache_control": {"type": "ephemeral"}}
        system = [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
        final = ""
        while not build.stop_requested:
            response = self.client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=system, tools=defs, messages=self.messages
            )
            usage = response.usage
            build.usage["input_tokens"] += (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            build.usage["output_tokens"] += usage.output_tokens or 0
            content = [block.model_dump(exclude_none=True) for block in response.content]
            self.messages.append({"role": "assistant", "content": content})
            texts = [c["text"] for c in content if c.get("type") == "text" and c.get("text", "").strip()]
            if texts:
                final = "\n".join(texts)
                build.log("assistant", text=final)
            calls = [c for c in content if c.get("type") == "tool_use"]
            if not calls:
                break
            results = []
            for call in calls:
                result = build.call_tool(call["name"], call.get("input") or {})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(result, default=str),
                        **({"is_error": True} if "error" in result else {}),
                    }
                )
            self.messages.append({"role": "user", "content": results})
            if build.turn_over:
                break
        return LoopOutcome(final_text=final)


# ---- OpenAI-compatible chat completions and Gemini (plain HTTPS) ------------------

HTTP_TIMEOUT_SECONDS = 300
# Self-hosted OpenAI-compatible servers (llama-server, vLLM, Ollama...) are
# often slow models on modest hardware and can take many minutes per turn.
LOCAL_HTTP_TIMEOUT_SECONDS = 900


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], what: str, timeout: float | None = None
) -> dict[str, Any]:
    """POST JSON with the standard library (same approach as the openai/gemini
    draft providers -- no extra packages), with errors worded for the user."""
    timeout = timeout or float(os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS", HTTP_TIMEOUT_SECONDS))
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:800]
        raise RuntimeError(f"{what} returned HTTP {e.code}: {body}") from e
    except TimeoutError as e:
        raise LLMUnavailable(f"{what} did not respond within {timeout:.0f}s -- set MODELMAKER_LLM_TIMEOUT_SECONDS to wait longer") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise LLMUnavailable(f"{what} did not respond within {timeout:.0f}s -- set MODELMAKER_LLM_TIMEOUT_SECONDS to wait longer") from e
        raise LLMUnavailable(f"could not reach {what} -- check the base URL and network access") from e


def _parse_args(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Tool-call arguments arrive as a JSON string (OpenAI) or an object
    (Gemini); a model occasionally emits malformed JSON, which goes back to
    it as a tool error rather than failing the build."""
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        return None, f"arguments weren't valid JSON ({e}) -- call the tool again with valid JSON"
    return (parsed, None) if isinstance(parsed, dict) else (None, "arguments must be a JSON object")


class OpenAILoop(AgentLoop):
    """Tool-use loop over an OpenAI-compatible /chat/completions endpoint --
    the real OpenAI API by default, or any compatible service via base_url
    (the same setting the openai draft provider uses). No temperature or
    token cap is sent: current reasoning models reject non-default values
    for both, and every compatible server has a sensible default."""

    # Subclass knobs (see LMStudioLoop): a compact system prompt, a context
    # budget in tokens that old tool results get elided to stay under, and
    # a cap on each tool result's length. Off for hosted APIs, whose
    # context windows dwarf a build.
    compact = False
    context_tokens: int | None = None
    max_result_chars: int | None = None
    timeout_seconds: float | None = None
    what = "OpenAI-compatible endpoint"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        from ..llm import openai_provider

        self.base_url = (base_url or os.environ.get("MODELMAKER_OPENAI_BASE_URL", openai_provider.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("MODELMAKER_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL")
        if "api.openai.com" not in self.base_url and not os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS"):
            self.timeout_seconds = LOCAL_HTTP_TIMEOUT_SECONDS
        if not self.api_key:
            raise RuntimeError("no OpenAI API key configured -- set one in Settings, or export OPENAI_API_KEY")
        if not self.model:
            raise RuntimeError("no model configured for openai -- set one in Settings (or for this build)")
        self.messages: list[dict[str, Any]] = []

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def start(self, build, system, prompt, tools):
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return self._run(build, tools)

    def resume(self, build, prompt, tools):
        self.messages.append({"role": "user", "content": prompt})
        return self._run(build, tools)

    def _fit_context(self, build: AgentBuild, defs: list[dict[str, Any]]) -> None:
        """Keep the conversation inside `context_tokens`: estimate its size
        (~3 chars/token, deliberately pessimistic), and while it's over ~85%
        of the budget, replace the oldest tool results with a stub. The
        model keeps every call it made and everything it wrote -- it just
        can't re-read old outputs, and can call the tool again."""
        if not self.context_tokens:
            return
        budget = self.context_tokens * 0.85

        def size() -> float:
            return (len(json.dumps(self.messages)) + len(json.dumps(defs))) / 3

        if size() <= budget:
            return
        # Never elide the results the model is about to read (after its last turn).
        last_assistant = max((i for i, m in enumerate(self.messages) if m.get("role") == "assistant"), default=0)
        elided = 0
        for m in self.messages[:last_assistant]:
            if size() <= budget:
                break
            if m.get("role") == "tool" and not m.get("_elided"):
                m["content"] = json.dumps(
                    {"elided": "older tool result removed to fit the context window -- call the tool again if you need it"}
                )
                m["_elided"] = True
                elided += 1
        if elided:
            build.log("context", note=f"elided {elided} old tool result(s) to fit a {self.context_tokens}-token context")

    def _run(self, build: AgentBuild, tools: list[Tool]) -> LoopOutcome:
        defs = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
            for t in tools
        ]
        final = ""
        while not build.stop_requested:
            self._fit_context(build, defs)
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.messages],
            }
            if defs:
                payload["tools"] = defs
            response = _post_json(
                f"{self.base_url}/chat/completions",
                payload,
                self._headers(),
                f"{self.what} at {self.base_url}",
                self.timeout_seconds,
            )
            usage = response.get("usage") or {}
            build.usage["input_tokens"] += usage.get("prompt_tokens") or 0
            build.usage["output_tokens"] += usage.get("completion_tokens") or 0
            try:
                message = response["choices"][0]["message"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"response missing expected fields: {str(response)[:500]}") from e
            # Echo the assistant turn back (tool_calls and all) -- the API
            # requires each tool result to follow the call it answers. Minus
            # any reasoning trace: it only costs context, and some compatible
            # servers (e.g. DeepSeek's) reject it being sent back.
            self.messages.append(
                {k: v for k, v in message.items() if v is not None and k not in ("reasoning_content", "reasoning")}
            )
            text = (message.get("content") or "").strip()
            if text:
                final = text
                build.log("assistant", text=text)
            calls = message.get("tool_calls") or []
            if not calls:
                break
            for call in calls:
                fn = call.get("function") or {}
                args, err = _parse_args(fn.get("arguments"))
                result = {"error": err} if err else build.call_tool(fn.get("name", ""), args)
                content = json.dumps(result, default=str)
                if self.max_result_chars and len(content) > self.max_result_chars:
                    content = content[: self.max_result_chars] + " ...[truncated to fit the context window]"
                self.messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
            if build.turn_over:
                break
        return LoopOutcome(final_text=final)


class LMStudioLoop(OpenAILoop):
    """Local models behind an OpenAI-compatible server: LM Studio, or
    llama.cpp's llama-server (started with --jinja, which tool calling
    needs), Ollama's /v1, vLLM... Uses the lmstudio provider's settings --
    base URL (default http://localhost:1234/v1), and a model that's
    auto-detected when unset. No API key.

    Local models tend to have small context windows and generate slowly,
    so this runs in compact mode: a short block catalogue in the system
    prompt (the model reads details with describe_block_type), tool
    results capped in length, old results elided to stay within the
    window, and long request timeouts. The window size comes from
    MODELMAKER_LLM_CONTEXT_TOKENS, else the server's own report of it
    (llama.cpp's /v1/models carries n_ctx), else a conservative 16k."""

    compact = True
    max_result_chars = 6000
    what = "local model server"
    DEFAULT_CONTEXT_TOKENS = 16384

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        from ..llm import lmstudio_provider

        self.base_url = (base_url or os.environ.get("MODELMAKER_LLM_BASE_URL", lmstudio_provider.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = None
        self.timeout_seconds = float(os.environ.get("MODELMAKER_LLM_TIMEOUT_SECONDS", LOCAL_HTTP_TIMEOUT_SECONDS))
        self.messages = []
        models = self._server_models()
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL") or (models[0]["id"] if models else None)
        if not self.model:
            raise RuntimeError(f"the server at {self.base_url} reports no models -- load one, or set a model in Settings")
        env_ctx = os.environ.get("MODELMAKER_LLM_CONTEXT_TOKENS")
        served = next((m for m in models if m.get("id") == self.model), models[0] if models else {})
        n_ctx = (served.get("meta") or {}).get("n_ctx")
        self.context_tokens = int(env_ctx) if env_ctx else int(n_ctx) if n_ctx else self.DEFAULT_CONTEXT_TOKENS

    def _server_models(self) -> list[dict[str, Any]]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as resp:
                return list(json.load(resp).get("data") or [])
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(
                f"could not reach the local model server at {self.base_url} -- is it running "
                "(LM Studio > Developer > Start Server, or llama-server --jinja)? Check the base URL in Settings."
            ) from e


class GeminiLoop(AgentLoop):
    """Function-calling loop over the Gemini API's generateContent. Tool
    schemas go in `parametersJsonSchema` (full JSON Schema) rather than the
    OpenAPI-subset `parameters`, which rejects e.g. additionalProperties
    and property-less objects. Each model turn is echoed back unchanged, so
    thinking models' thought signatures survive into the next request."""

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        from ..llm import gemini_provider

        self.base_url = (base_url or os.environ.get("MODELMAKER_GEMINI_BASE_URL", gemini_provider.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("MODELMAKER_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL")
        if not self.api_key:
            raise RuntimeError("no Gemini API key configured -- set one in Settings, or export GEMINI_API_KEY")
        if not self.model:
            raise RuntimeError("no model configured for gemini -- set one in Settings (or for this build)")
        self.system = ""
        self.contents: list[dict[str, Any]] = []

    def start(self, build, system, prompt, tools):
        self.system = system
        self.contents = [{"role": "user", "parts": [{"text": prompt}]}]
        return self._run(build, tools)

    def resume(self, build, prompt, tools):
        last = self.contents[-1] if self.contents else None
        if last and last["role"] == "user":
            last["parts"].append({"text": prompt})  # after a terminal tool's functionResponse
        else:
            self.contents.append({"role": "user", "parts": [{"text": prompt}]})
        return self._run(build, tools)

    def _run(self, build: AgentBuild, tools: list[Tool]) -> LoopOutcome:
        decls = [{"name": t.name, "description": t.description, "parametersJsonSchema": t.schema} for t in tools]
        final = ""
        while not build.stop_requested:
            payload: dict[str, Any] = {"system_instruction": {"parts": [{"text": self.system}]}, "contents": self.contents}
            if decls:
                payload["tools"] = [{"functionDeclarations": decls}]
            response = _post_json(
                f"{self.base_url}/models/{self.model}:generateContent",
                payload,
                {"x-goog-api-key": self.api_key},
                "Gemini API",
            )
            usage = response.get("usageMetadata") or {}
            build.usage["input_tokens"] += usage.get("promptTokenCount") or 0
            build.usage["output_tokens"] += (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
            try:
                content = response["candidates"][0].get("content") or {"role": "model", "parts": []}
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"response missing expected fields: {str(response)[:500]}") from e
            parts = content.get("parts") or []
            self.contents.append({"role": "model", "parts": parts})
            texts = [p["text"] for p in parts if p.get("text", "").strip() and not p.get("thought")]
            if texts:
                final = "\n".join(texts)
                build.log("assistant", text=final)
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not calls:
                break
            responses = []
            for call in calls:
                args, err = _parse_args(call.get("args") or {})
                result = {"error": err} if err else build.call_tool(call.get("name", ""), args)
                fr: dict[str, Any] = {"name": call.get("name", ""), "response": result}
                if call.get("id"):
                    fr["id"] = call["id"]
                responses.append({"functionResponse": fr})
            self.contents.append({"role": "user", "parts": responses})
            if build.turn_over:
                break
        return LoopOutcome(final_text=final)


# ---- claude CLI (via MCP) ----------------------------------------------------------

CLI_DEFAULT_MODEL = "claude-sonnet-5"
MCP_SERVER_NAME = "modelmaker"


class ClaudeCliLoop(AgentLoop):
    """Runs the conversation inside the locally installed `claude` CLI, on
    whatever login it already has (no API key needed). The CLI talks to
    this build's tools through a small stdio MCP server
    (agent/mcp_server.py) that forwards each call to this API server --
    so the tools, guards and event log are exactly the same as for the
    in-process loops. Resumable via the CLI's own --resume <session>."""

    def __init__(self, model: str | None, api_base_url: str, token: str) -> None:
        self.model = model or os.environ.get("MODELMAKER_LLM_MODEL", CLI_DEFAULT_MODEL)
        self.binary = shutil.which("claude")
        if not self.binary:
            raise RuntimeError("claude CLI not found on PATH; install Claude Code or choose a different provider")
        self.api_base_url = api_base_url.rstrip("/")
        self.token = token
        self.system = ""
        self.session_id: str | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self, build, system, prompt, tools):
        self.system = system
        self.session_id = None
        return self._run(build, prompt, tools)

    def resume(self, build, prompt, tools):
        return self._run(build, prompt, tools)

    def cancel(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()

    def _mcp_config(self) -> str:
        config = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "stdio",
                    "command": sys.executable,
                    # --token=<value>: a token_urlsafe token can start with "-",
                    # which argparse would take for an option.
                    "args": ["-m", "modelmaker.agent.mcp_server", f"--url={self.api_base_url}", f"--token={self.token}"],
                    "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                }
            }
        }
        fd, path = tempfile.mkstemp(prefix="modelmaker-mcp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return path

    def _run(self, build: AgentBuild, prompt: str, tools: list[Tool]) -> LoopOutcome:
        config_path = self._mcp_config()
        # The system prompt (block catalogue included) and the prompt (plan
        # JSON included) easily exceed Windows' 32K command-line limit
        # (WinError 206), so they go via a file and stdin, not argv.
        fd, system_path = tempfile.mkstemp(prefix="modelmaker-system-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(self.system)
        allowed = [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in tools]
        cmd = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--mcp-config",
            config_path,
            "--strict-mcp-config",
            "--tools",
            "",
            "--allowedTools",
            *allowed,
        ]
        cmd += ["--system-prompt-file", system_path]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        env = {**os.environ, "MCP_TOOL_TIMEOUT": str(30 * 60 * 1000), "MCP_TIMEOUT": "60000"}
        final, error = "", None
        try:
            with self._lock:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
            proc = self._proc
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "system" and msg.get("session_id"):
                    self.session_id = msg["session_id"]
                elif kind == "assistant":
                    for block in (msg.get("message") or {}).get("content") or []:
                        if block.get("type") == "text" and block.get("text", "").strip():
                            final = block["text"]
                            build.log("assistant", text=final)
                elif kind == "result":
                    self.session_id = msg.get("session_id") or self.session_id
                    cost = msg.get("total_cost_usd")
                    if cost:
                        build.usage["cost_usd"] = round(build.usage.get("cost_usd", 0.0) + cost, 4)
                    usage = msg.get("usage") or {}
                    build.usage["input_tokens"] += (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
                    build.usage["output_tokens"] += usage.get("output_tokens") or 0
                    if msg.get("is_error"):
                        error = str(msg.get("result") or msg.get("subtype") or "claude CLI error")
                    elif msg.get("result"):
                        final = msg["result"]
            proc.wait()
            if proc.returncode not in (0, None) and not build.stop_requested and error is None:
                stderr = (proc.stderr.read() if proc.stderr else "").strip()
                error = f"claude CLI exited {proc.returncode}: {stderr[-800:]}"
        finally:
            for path in (config_path, system_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return LoopOutcome(final_text=final, error=error)


# ---- factory ---------------------------------------------------------------------------

AGENT_CAPABLE_PROVIDERS = ("claude_cli", "anthropic", "openai", "gemini", "lmstudio")


def make_loop(
    provider: str,
    model: str | None,
    *,
    api_base_url: str,
    token: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AgentLoop:
    """`api_base_url`/`token` are this server's own address and the build's
    MCP token (claude_cli only); `base_url` is the provider endpoint
    override from Settings (openai/gemini)."""
    if provider == "claude_cli":
        return ClaudeCliLoop(model, api_base_url, token)
    if provider == "anthropic":
        return AnthropicLoop(model, api_key=api_key)
    if provider == "openai":
        return OpenAILoop(model, api_key=api_key, base_url=base_url)
    if provider == "gemini":
        return GeminiLoop(model, api_key=api_key, base_url=base_url)
    if provider == "lmstudio":
        return LMStudioLoop(model, base_url=base_url)
    raise ValueError(
        f"provider {provider!r} can't drive an AI build yet (tool calling isn't wired up for it); "
        f"use one of {', '.join(AGENT_CAPABLE_PROVIDERS)}"
    )
