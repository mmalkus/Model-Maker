"""Decision-based model planning: the build as a fixed set of phases, each
a menu of blocks ("which block next?", DONE included) and, per block, typed
questions that set its parameters -- every step a small, bounded decision
instead of free-form agent reasoning.

The questions use Laya's typed-decision shape (choice / score / noul, see
https://pypi.org/project/laya/), so the decision model is pluggable: the
rule-based policy (`RuleBackend`, the decision-hint thresholds), Laya itself
(`LayaBackend`), or an LLM made to answer like Laya (`ClaudeCliBackend`, for
trying the idea before Laya's weights are at hand). Every state sent to a
model is kept within a Laya-sized context budget (see budget.py).

Entry point: planner.plan(workspace, decider)."""

from .core import Answer, Decider, DecisionRecord
from .planner import PlanResult, plan
from .workspace import Workspace

__all__ = ["Answer", "Decider", "DecisionRecord", "PlanResult", "Workspace", "plan"]
