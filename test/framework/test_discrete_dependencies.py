# SPDX-License-Identifier: MIT
"""
T-022 — discrete-update dependency analyzer + cycle detector tests.

The actual scheduler change (lower-triangular Phase 2) is tracked as
T-022a; this file exercises the static-analysis foundation:

  - ``find_cycles`` correctly identifies cycles in a hand-crafted graph
    (and reports none on a DAG).
  - ``topological_sort`` produces a lower-triangular order on a DAG.
  - ``topological_sort`` raises ``DependencyCycleError`` on a cyclic
    graph, with the cycle paths attached to the exception.
  - Self-loops (A → A) are detected.
"""

from __future__ import annotations

import pytest

from jaxonomy.framework.discrete_dependencies import (
    DependencyCycleError,
    find_cycles,
    topological_sort,
)


# ── find_cycles ──────────────────────────────────────────────────────────


def test_find_cycles_dag_returns_empty():
    g = {"a": set(), "b": {"a"}, "c": {"a", "b"}}
    assert find_cycles(g) == []


def test_find_cycles_simple_2_cycle():
    g = {"a": {"b"}, "b": {"a"}}
    cycles = find_cycles(g)
    assert len(cycles) >= 1
    # Cycle should mention both nodes.
    flat = sum((list(c) for c in cycles), [])
    assert "a" in flat and "b" in flat


def test_find_cycles_self_loop():
    g = {"a": {"a"}}
    cycles = find_cycles(g)
    assert len(cycles) == 1
    assert cycles[0] == ["a", "a"]


def test_find_cycles_3_cycle():
    g = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    cycles = find_cycles(g)
    assert len(cycles) >= 1
    flat = sum((list(c) for c in cycles), [])
    assert "a" in flat and "b" in flat and "c" in flat


# ── topological_sort ─────────────────────────────────────────────────────


def test_topological_sort_dag():
    """A DAG sorts so every node appears after its upstreams."""
    g = {"a": set(), "b": {"a"}, "c": {"a", "b"}, "d": {"c"}}
    order = topological_sort(g)
    pos = {n: i for i, n in enumerate(order)}
    assert pos["a"] < pos["b"]
    assert pos["a"] < pos["c"]
    assert pos["b"] < pos["c"]
    assert pos["c"] < pos["d"]


def test_topological_sort_disconnected_components():
    g = {"a": set(), "b": {"a"}, "x": set(), "y": {"x"}}
    order = topological_sort(g)
    pos = {n: i for i, n in enumerate(order)}
    assert pos["a"] < pos["b"]
    assert pos["x"] < pos["y"]


def test_topological_sort_cycle_raises():
    g = {"a": {"b"}, "b": {"a"}}
    with pytest.raises(DependencyCycleError, match="cycle"):
        topological_sort(g)


def test_cycle_error_carries_paths():
    g = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    try:
        topological_sort(g)
    except DependencyCycleError as e:
        assert hasattr(e, "cycles")
        assert len(e.cycles) >= 1
    else:
        pytest.fail("expected DependencyCycleError")


def test_topological_sort_singleton():
    """A single-node graph with no upstream sorts trivially."""
    g = {"only": set()}
    assert topological_sort(g) == ["only"]


def test_topological_sort_implicit_dependency_target():
    """When a graph references a node only as an upstream
    (never declared as a key), topological_sort still includes it."""
    g = {"b": {"a"}}  # 'a' is only on the upstream side
    order = topological_sort(g)
    assert "a" in order
    assert order.index("a") < order.index("b")
