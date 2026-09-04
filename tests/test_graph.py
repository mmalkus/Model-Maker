import pytest

from modelmaker.graph import Graph, Wire

from .helpers import make_block


def test_topo_order_respects_dependencies():
    a = make_block("a", "read_csv", params={"path": "x.csv"}, x=0)
    b = make_block("b", "filter", params={"expr": "1=1"}, x=1)
    c = make_block("c", "select", params={"cols": []}, x=2)
    graph = Graph(
        blocks={"a": a, "b": b, "c": c},
        wires={
            "w1": Wire("w1", "a", "out", "b", "df"),
            "w2": Wire("w2", "b", "out", "c", "df"),
        },
    )
    order = graph.topo_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_topo_order_ignores_lane_for_correctness_uses_it_only_for_ties():
    # b depends on a even though b's lane sorts before a's lane.
    a = make_block("a", "read_csv", params={"path": "x.csv"}, lane="lane2", x=5)
    b = make_block("b", "filter", params={"expr": "1=1"}, lane="lane1", x=0)
    from modelmaker.graph import Lane

    graph = Graph(
        lanes={"lane1": Lane("Lane 1", 0), "lane2": Lane("Lane 2", 1)},
        blocks={"a": a, "b": b},
        wires={"w1": Wire("w1", "a", "out", "b", "df")},
    )
    order = graph.topo_order()
    assert order.index("a") < order.index("b")


def test_cycle_detection_raises():
    a = make_block("a", "filter", params={"expr": "1=1"})
    b = make_block("b", "filter", params={"expr": "1=1"})
    graph = Graph(
        blocks={"a": a, "b": b},
        wires={
            "w1": Wire("w1", "a", "out", "b", "df"),
            "w2": Wire("w2", "b", "out", "a", "df"),
        },
    )
    with pytest.raises(ValueError):
        graph.topo_order()


def test_creates_cycle_detects_would_be_cycle():
    a = make_block("a", "filter", params={"expr": "1=1"})
    b = make_block("b", "filter", params={"expr": "1=1"})
    graph = Graph(
        blocks={"a": a, "b": b},
        wires={"w1": Wire("w1", "a", "out", "b", "df")},
    )
    assert graph.creates_cycle("b", "a") is True
    assert graph.creates_cycle("a", "a") is True
    assert graph.creates_cycle("a", "b") is False


def test_ancestors_transitive():
    a = make_block("a", "read_csv", params={"path": "x.csv"})
    b = make_block("b", "filter", params={"expr": "1=1"})
    c = make_block("c", "select", params={"cols": []})
    graph = Graph(
        blocks={"a": a, "b": b, "c": c},
        wires={
            "w1": Wire("w1", "a", "out", "b", "df"),
            "w2": Wire("w2", "b", "out", "c", "df"),
        },
    )
    assert graph.ancestors(["c"]) == {"a", "b", "c"}
