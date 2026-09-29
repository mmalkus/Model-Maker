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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .build import AgentBuild
from .tools import Tool


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
                    "args": ["-m", "modelmaker.agent.mcp_server", "--url", self.api_base_url, "--token", self.token],
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
        allowed = [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in tools]
        cmd = [
            self.binary,
            "-p",
            prompt,
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
        cmd += ["--system-prompt", self.system]
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
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
            proc = self._proc
            assert proc.stdout is not None
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
            try:
                os.unlink(config_path)
            except OSError:
                pass
        return LoopOutcome(final_text=final, error=error)


# ---- factory ---------------------------------------------------------------------------

AGENT_CAPABLE_PROVIDERS = ("claude_cli", "anthropic")


def make_loop(provider: str, model: str | None, *, api_base_url: str, token: str, api_key: str | None = None) -> AgentLoop:
    if provider == "claude_cli":
        return ClaudeCliLoop(model, api_base_url, token)
    if provider == "anthropic":
        return AnthropicLoop(model, api_key=api_key)
    raise ValueError(
        f"provider {provider!r} can't drive an AI build yet (tool calling isn't wired up for it); "
        f"use one of {', '.join(AGENT_CAPABLE_PROVIDERS)}"
    )
