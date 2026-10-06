"""Decision backends: the rule policy, Laya, and a Claude model answering
like Laya (via the logged-in `claude` CLI, as the claude_cli LLM provider
does)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...llm.prompts import extract_json_response
from .core import Answer, labels, render


class RuleBackend:
    """The decision-hint thresholds as a backend: Decider answers each
    question with its rule (always sure). The baseline everything else is
    compared with."""

    name = "rules"
    uses_rules = True

    def decide(self, items):  # pragma: no cover - Decider never calls it
        raise NotImplementedError


def _answer(question: dict[str, Any], probs: dict[str, float]) -> Answer:
    """An Answer from a label -> probability map (normalised here)."""
    opts = labels(question)
    if question["type"] == "noul":
        p = float(probs.get("true", probs.get("yes", 0.5)))
        p = min(max(p, 0.0), 1.0)
        return Answer(p >= 0.5, max(p, 1 - p), {"true": p, "false": 1 - p})
    keyed = {str(o): max(float(probs.get(str(o), 0.0)), 0.0) for o in opts}
    total = sum(keyed.values()) or 1.0
    keyed = {k: v / total for k, v in keyed.items()}
    if question["type"] == "score":
        # Laya's score is the expected level; report the nearest level.
        expected = sum(i * keyed[str(o)] for i, o in enumerate(opts))
        best = opts[min(range(len(opts)), key=lambda i: abs(i - expected))]
        return Answer(best, keyed[str(best)], keyed)
    best = max(opts, key=lambda o: keyed[str(o)])
    return Answer(best, keyed[str(best)], keyed)


SYSTEM_PROMPT = """You are a decision model (System 1), emulating Laya: you read a STATE of facts \
and typed questions, and answer each with probabilities -- no explanations, no text outside JSON.

Question types:
- [choice]: give a probability for every option.
- [score]: give a probability for every level.
- [noul]: give the probability that the answer is yes.

Probabilities must be calibrated: near 1.0 only when the facts clearly decide it, spread out \
when they don't. Use only the facts in STATE and the criteria in the question.

Reply with one JSON object only, keyed by each QUESTION's exact id:
{"<qid>": {"probabilities": {"<option or level>": p, ...}}, "<noul qid>": {"yes": p}}"""


class ClaudeCliBackend:
    """A Claude model standing in for Laya: same input (one compact state +
    typed questions, within the token budget), same output shape. Calls the
    logged-in `claude` CLI with no tools and a minimal system prompt, a few
    asks in parallel."""

    def __init__(self, model: str = "claude-haiku-4-5", workers: int = 12, timeout: float = 120):
        self.model = model
        self.name = f"claude_cli:{model}"
        self.workers = workers
        self.timeout = timeout
        self.binary = shutil.which("claude")
        if not self.binary:
            raise RuntimeError("claude CLI not found on PATH")
        fd, self._system_path = tempfile.mkstemp(prefix="modelmaker-decide-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SYSTEM_PROMPT)

    def _one(self, item: tuple[dict, dict]) -> dict[str, Answer]:
        state, questions = item
        last_error = None
        for _ in range(2):  # one retry on an unparseable reply
            result = subprocess.run(
                [
                    self.binary,
                    "-p",
                    "--output-format",
                    "json",
                    "--tools",
                    "",
                    "--model",
                    self.model,
                    "--system-prompt-file",
                    self._system_path,
                ],
                input=render(state, questions),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
            if result.returncode != 0:
                last_error = RuntimeError(f"claude CLI exited {result.returncode}: {result.stderr.strip()[:300]}")
                continue
            try:
                payload = extract_json_response(json.loads(result.stdout)["result"])
                if len(questions) == 1 and len(payload) == 1:
                    payload = {next(iter(questions)): next(iter(payload.values()))}
                return {
                    qid: _answer(
                        q,
                        {
                            str(k).lower() if q["type"] == "noul" else str(k): v
                            for k, v in (payload[qid].get("probabilities") or payload[qid]).items()
                        },
                    )
                    for qid, q in questions.items()
                }
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                last_error = e
        raise RuntimeError(f"no usable answer from {self.model}: {last_error}")

    def decide(self, items):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, items))


class LayaBackend:
    """Laya itself (pip install laya; weights download from Hugging Face on
    first use). `model` "router" lets Laya pick the checkpoint."""

    def __init__(self, model: str = "convaiinnovations/laya"):
        import laya

        self.name = f"laya:{model}"
        self.agent = laya.Router() if model == "router" else laya.load(model)

    def decide(self, items):
        out = []
        for state, questions in items:
            res = self.agent.predict(state, questions)["answers"]
            answers = {}
            for qid, q in questions.items():
                a = res[qid]
                if q["type"] == "noul":
                    answers[qid] = _answer(q, {"true": a["noul"]})
                else:
                    answers[qid] = _answer(q, {str(k): v for k, v in a["probabilities"].items()})
            out.append(answers)
        return out


def get_backend(name: str) -> Any:
    if name == "rules":
        return RuleBackend()
    if name.startswith("laya"):
        return LayaBackend(name.split(":", 1)[1] if ":" in name else "convaiinnovations/laya")
    if name == "haiku":
        return ClaudeCliBackend("claude-haiku-4-5")
    if name.startswith("claude:"):
        return ClaudeCliBackend(name.split(":", 1)[1])
    raise ValueError(f"unknown decision backend {name!r}: rules, haiku, claude:<model>, laya[:<model>|router]")
