"""Typed questions, answers, the context budget, and the Decider that asks
a backend and records every decision next to what the rule policy says.

A question is Laya's shape, a plain dict:
    {"type": "choice", "instructions": str, "criteria": {label: description}}
    {"type": "score",  "instructions": str, "criteria": [label, ...]}  # ordered
    {"type": "noul",   "instructions": str}                            # yes/no
A state is a small dict of facts (never rows). One ask = one state plus one
or more questions about it, like Laya's predict()."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

# Laya's English checkpoint reads 512 tokens (state, questions and options
# together); the multilingual and typed-decisions ones 1024. Keep to the
# smaller one so any checkpoint fits.
MAX_CONTEXT_TOKENS = 512
# Below this a decision goes to the person as a choice card.
CONFIDENCE_THRESHOLD = 0.7


class ContextBudgetError(ValueError):
    pass


def choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def labels(question: dict[str, Any]) -> list[Any]:
    if question["type"] == "noul":
        return [True, False]
    return list(question["criteria"])  # a choice's labels, or a score's levels


@dataclass
class Answer:
    value: Any  # the choice label, the score level label, or a bool (noul)
    confidence: float  # the chosen label's probability
    probabilities: dict[str, float] = field(default_factory=dict)


# ---- context budget -----------------------------------------------------------------


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.3g}") if value == value else None
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def compact(state: dict[str, Any]) -> str:
    return json.dumps(_round(state), separators=(",", ":"), default=str)


def render(state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> str:
    """What a decision model reads: the state, then each question with its
    options -- the same text the token budget is measured on."""
    parts = ["STATE " + compact(state)]
    for qid, q in questions.items():
        line = f'QUESTION id="{qid}" [{q["type"]}] {q["instructions"]}'
        if q["type"] == "choice":
            line += " OPTIONS " + "; ".join(f"{k}: {v}" for k, v in q["criteria"].items())
        elif q["type"] == "score":
            line += " LEVELS (low to high) " + ", ".join(q["criteria"])
        parts.append(line)
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    # No tokenizer offline; ModernBERT's BPE averages ~4 chars a token on
    # prose and fewer on JSON and numbers, so count 3 to stay on the safe side.
    return math.ceil(len(text) / 3)


def fit(
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    max_tokens: int,
    drop_order: list[str] = (),
) -> tuple[dict[str, Any], int]:
    """The state cut down to fit max_tokens with its questions: drops the
    keys in drop_order (least important first) until it fits. Raises
    ContextBudgetError when it still doesn't -- split the ask instead."""
    state = dict(state)
    tokens = estimate_tokens(render(state, questions))
    for key in drop_order:
        if tokens <= max_tokens:
            break
        if state.pop(key, None) is not None:
            tokens = estimate_tokens(render(state, questions))
    if tokens > max_tokens:
        raise ContextBudgetError(f"decision needs ~{tokens} tokens, over the {max_tokens}-token budget")
    return state, tokens


# ---- backends + decider -----------------------------------------------------------------

Rule = Callable[[dict[str, Any]], Any]  # state -> the policy's answer value


class Backend(Protocol):
    name: str

    def decide(self, items: list[tuple[dict, dict]]) -> list[dict[str, Answer]]:
        """One answers-dict per (state, questions) item, in order."""
        ...


@dataclass
class DecisionRecord:
    name: str  # what was decided, e.g. "phase1.next" or "screen:annual_income"
    qid: str
    state: dict[str, Any]
    question: dict[str, Any]
    answer: Answer
    rule_answer: Any
    tokens: int
    needs_user: bool

    @property
    def agrees(self) -> bool:
        return self.answer.value == self.rule_answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qid": self.qid,
            "state": self.state,
            "answer": self.answer.value,
            "confidence": round(self.answer.confidence, 3),
            "rule": self.rule_answer,
            "agrees": self.agrees,
            "tokens": self.tokens,
            "needs_user": self.needs_user,
        }


@dataclass
class Ask:
    name: str
    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    rules: dict[str, Rule]
    drop_order: list[str] = field(default_factory=list)


class Decider:
    """Asks the backend, checks the budget, compares with the rule policy,
    flags low-confidence answers for the person, and logs it all."""

    def __init__(
        self,
        backend: Backend,
        max_tokens: int = MAX_CONTEXT_TOKENS,
        threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.backend = backend
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.records: list[DecisionRecord] = []

    def ask(
        self,
        name: str,
        state: dict,
        questions: dict,
        rules: dict,
        drop_order: list[str] = (),
    ) -> dict[str, Any]:
        return self.ask_many([Ask(name, state, questions, rules, list(drop_order))])[0]

    def ask_many(self, asks: list[Ask]) -> list[dict[str, Any]]:
        """Answer values per ask (backend batches/parallelises them)."""
        fitted = [fit(a.state, a.questions, self.max_tokens, a.drop_order) for a in asks]
        if getattr(self.backend, "uses_rules", False):
            # The policy itself: answers straight from the rules, sure of each.
            answers = [{qid: Answer(a.rules[qid](a.state), 1.0) for qid in a.questions} for a in asks]
        else:
            answers = self.backend.decide([(state, a.questions) for (state, _), a in zip(fitted, asks)])
        out = []
        for a, (state, tokens), ans in zip(asks, fitted, answers):
            values = {}
            for qid, q in a.questions.items():
                answer = ans[qid]
                rule_value = a.rules[qid](a.state) if qid in a.rules else None
                self.records.append(
                    DecisionRecord(
                        a.name,
                        qid,
                        state,
                        q,
                        answer,
                        rule_value,
                        tokens,
                        answer.confidence < self.threshold,
                    )
                )
                values[qid] = answer.value
            out.append(values)
        return out
