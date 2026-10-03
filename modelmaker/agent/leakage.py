"""Holdout leakage in a graph's wiring: a fitted artifact (binning, model,
rating scale) applied to a test or out-of-time sample although it was
fitted on rows that include that sample.

Every dataframe flow is described by its row *atoms*: where its rows come
from (the source block) and which side of each split they took. A source
block's rows are one atom, {src: block}; a split's train/development port
adds {split: "train"} to each of its input's atoms and its test/out-of-time
port {split: "holdout"}; any other block passes its inputs' atoms through
(filters, derived columns, joins, scores -- rows are never invented). Two
atoms share rows unless they come from different sources or took
different sides of the same split.

A block that learns from the target (a dataframe input and a binning,
model or master_scale output -- fit_binning, logistic_regression,
calibrate_model, ...) is a fitter on its dataframe atoms, and its artifact
carries those (plus whatever artifacts it took in, so binning -> model ->
calibration -> predict is followed through). A block that applies an
artifact to a dataframe leaks when one of its data atoms is on the holdout
side of a split and shares rows with an atom the artifact was fitted on:
the fit saw the holdout rows. woe_transform, which fits and applies in one
go, leaks on any holdout-side data. Fitting on the full data to *look* at
it (an IV table) is fine -- only applying the result to a holdout sample
is flagged.

Pure: works on a plain node/edge description, so a stage plan can be
checked before anything is built (see tools.validate_steps)."""

from __future__ import annotations

from dataclasses import dataclass, field

ARTIFACT_TYPES = frozenset({"binning", "model", "master_scale"})
# Splits: output port -> side.
SPLIT_SIDES = {
    "train_test_split": {"train": "train", "test": "holdout"},
    "time_split": {"development": "train", "out_of_time": "holdout"},
}
# Fit and apply in one block, on whatever data it's given.
SELF_FITTING = frozenset({"woe_transform"})
# Choose features from the target's statistics, without a fitted artifact.
TARGET_SCREENS = frozenset({"iv_table", "stepwise_selection"})


def learns_from_target(spec) -> bool:
    """Whether a registry block learns from the target: it fits an artifact
    from a dataframe (fit_binning, logistic_regression, calibrate_model...),
    or screens features on it."""
    if spec.category in SELF_FITTING or spec.category in TARGET_SCREENS:
        return True
    return any(p.type == "dataframe" for p in spec.inputs) and any(p.type in ARTIFACT_TYPES for p in spec.outputs)


def outline_order_problems(stages: list[dict], registry) -> list[str]:
    """An outline that learns from the target in a stage before the one
    that splits the sample: its fits (and the features they pick) would
    see the holdout rows. Stages read by their `blocks`."""
    split_at = next(
        (i for i, st in enumerate(stages) if any(c in SPLIT_SIDES for c in st.get("blocks") or [])), None
    )
    if split_at is None:
        return []
    out = []
    for st in stages[:split_at]:
        learners = [c for c in st.get("blocks") or [] if c in registry and learns_from_target(registry[c])]
        if learners:
            out.append(
                f"stage {st.get('key')}: {', '.join(learners)} learn(s) from the target, but the sample is only split "
                f"in a later stage ({stages[split_at].get('key')}) -- move the split before it, so the fits and the "
                "features they pick see the development sample only"
            )
    return out

Atom = frozenset  # of (key, value): ("src", block id) and (split id, side)


@dataclass
class Node:
    id: str
    name: str
    category: str
    inputs: dict[str, str]  # port -> type
    outputs: dict[str, str]


@dataclass
class Edge:
    src: str
    src_port: str
    dst: str
    dst_port: str


@dataclass
class Leak:
    block: str  # the block that applies (or fits-and-applies) on holdout rows
    fitter: str  # the block whose fit saw those rows (== block for woe_transform)
    split: str
    message: str
    involves: set[str] = field(default_factory=set)


def _overlap(a: Atom, b: Atom) -> bool:
    da, db = dict(a), dict(b)
    return all(da[k] == db[k] for k in da.keys() & db.keys())


def _holdout_split(atom: Atom) -> str | None:
    return next((k for k, v in sorted(atom) if k != "src" and v == "holdout"), None)


def _topo(nodes: dict[str, Node], edges: list[Edge]) -> list[str]:
    preds: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        if e.src in nodes and e.dst in nodes:
            preds[e.dst].add(e.src)
    order: list[str] = []
    done: set[str] = set()

    def visit(n: str, stack: frozenset[str]) -> None:
        if n in done or n in stack:
            return
        for p in sorted(preds[n]):
            visit(p, stack | {n})
        done.add(n)
        order.append(n)

    for n in sorted(nodes):
        visit(n, frozenset())
    return order


def find_leaks(nodes: dict[str, Node], edges: list[Edge]) -> list[Leak]:
    into: dict[str, list[Edge]] = {n: [] for n in nodes}
    for e in edges:
        if e.src in nodes and e.dst in nodes:
            into[e.dst].append(e)
    data: dict[tuple[str, str], set[Atom]] = {}  # (block, port) -> row atoms
    fits: dict[tuple[str, str], set[tuple[str, Atom]]] = {}  # artifact port -> (fitter, atom)
    leaks: list[Leak] = []
    seen: set[tuple[str, str, str]] = set()

    for nid in _topo(nodes, edges):
        node = nodes[nid]
        df_in: set[Atom] = set()
        art_in: set[tuple[str, Atom]] = set()
        has_df_input = False
        for e in into[nid]:
            kind = node.inputs.get(e.dst_port) or nodes[e.src].outputs.get(e.src_port)
            if kind == "dataframe":
                has_df_input = True
                df_in |= data.get((e.src, e.src_port), set())
            elif kind in ARTIFACT_TYPES:
                art_in |= fits.get((e.src, e.src_port), set())
        if not has_df_input and not any(t == "dataframe" for t in node.inputs.values()):
            df_in = {Atom({("src", nid)})}

        # Applying an artifact to holdout rows its fit saw.
        for atom in df_in:
            split = _holdout_split(atom)
            if split is None:
                continue
            for fitter, fatom in art_in:
                if _overlap(atom, fatom) and (nid, fitter, split) not in seen:
                    seen.add((nid, fitter, split))
                    f, s = nodes[fitter], nodes[split]
                    leaks.append(Leak(nid, fitter, split, (
                        f"{node.category} {node.name!r} applies what {f.category} {f.name!r} learned to the holdout "
                        f"sample of {s.category} {s.name!r} -- but {f.name!r} was fitted on data that includes that "
                        f"sample, so its results there are leaked (optimistic). Fit {f.name!r} on the split's "
                        f"{'train' if s.category == 'train_test_split' else 'development'} output instead."
                    ), {nid, fitter, split}))
            if node.category in SELF_FITTING and (nid, nid, split) not in seen:
                seen.add((nid, nid, split))
                s = nodes[split]
                leaks.append(Leak(nid, nid, split, (
                    f"woe_transform {node.name!r} derives WoE from the holdout sample of {s.name!r} using that "
                    f"sample's own target -- leakage. Use fit_binning on the train sample and apply_binning here."
                ), {nid, split}))

        for port, kind in node.outputs.items():
            if kind == "dataframe":
                sides = SPLIT_SIDES.get(node.category)
                if sides and port in sides:
                    data[(nid, port)] = {Atom(a | {(nid, sides[port])}) for a in df_in}
                else:
                    data[(nid, port)] = set(df_in)
            elif kind in ARTIFACT_TYPES:
                learned = {(nid, a) for a in df_in} if has_df_input else set()
                fits[(nid, port)] = art_in | learned
    return leaks
