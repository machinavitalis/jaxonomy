# SPDX-License-Identifier: MIT
"""
Discrete-update dependency analyzer (T-022).

The current discrete-update rule (see ``SystemBase.handle_discrete_update``)
is *Drake-style two-phase*:

  - Phase 1: every block computes its output ``y[n] = g(x[n], u[n])``.
    Sequential, no snapshot.
  - Phase 2: every block computes its next state ``x[n+1]`` against a
    snapshot of the post-Phase-1 context.  Each state update sees other
    blocks' ``y[n]`` (cache-updated) but their ``x[n]`` (pre-update).

T-022's lower-triangular rule would relax Phase 2 so that block B's
state update can see block A's ``x[n+1]`` whenever the discrete
dependency graph contains an edge A → B (B reads A's discrete state).
That requires (a) a topological sort of the dependency graph and (b)
cycle detection: a cycle is an algebraic loop on discrete state.

This module ships the analyzer and cycle detector — the static
preconditions for the scheduler change.  The actual reorder of Phase 2
is filed as T-022a.

Use::

    from jaxonomy.framework.discrete_dependencies import (
        build_discrete_dependency_graph, find_cycles,
    )
    g = build_discrete_dependency_graph(diagram)
    cycles = find_cycles(g)
    if cycles:
        raise BuilderError(f"Discrete-state cycle detected: {cycles}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional

if TYPE_CHECKING:
    from .system_base import SystemBase


__all__ = [
    "build_discrete_dependency_graph",
    "find_cycles",
    "topological_sort",
    "DependencyCycleError",
]


class DependencyCycleError(ValueError):
    """Raised when the discrete dependency graph contains a cycle."""


def build_discrete_dependency_graph(
    system: "SystemBase",
) -> dict[Hashable, set[Hashable]]:
    """Return ``{block_id: {upstream_block_ids}}`` for every block with a
    discrete state update.

    "Upstream" means: the upstream block's *discrete* state contributes
    to the downstream block's update, via an input-port connection in
    the diagram or via a direct shared-state read.  The current
    implementation conservatively traces input-port connections only —
    blocks that read each other's state through a shared global (rare
    and discouraged) are not detected.
    """
    from .diagram import Diagram

    graph: dict[Hashable, set[Hashable]] = {}

    if not isinstance(system, Diagram):
        # Single LeafSystem: there are no upstream blocks; the only
        # entry is itself with an empty upstream set if it has any
        # discrete events.
        if system.state_update_events.has_events:
            graph[system.system_id] = set()
        return graph

    # For each leaf in the diagram with discrete state updates, trace
    # back through its connected input ports to find upstream blocks
    # whose own state updates feed it.
    leaf_ids = {leaf.system_id for leaf in system.leaf_systems}
    for leaf in system.leaf_systems:
        if not leaf.state_update_events.has_events:
            continue
        upstream: set[Hashable] = set()
        for in_port in leaf.input_ports:
            src = _resolve_upstream(system, leaf, in_port.index)
            if src is not None and src.system_id in leaf_ids:
                if src.state_update_events.has_events:
                    upstream.add(src.system_id)
        graph[leaf.system_id] = upstream
    return graph


def _resolve_upstream(diagram, leaf, port_index):
    """Best-effort: return the LeafSystem feeding ``leaf.input_ports[port_index]``,
    or None if the port is unconnected / unresolvable."""
    cmap = getattr(diagram, "_connection_map", None) or getattr(
        diagram, "connection_map", None
    )
    if cmap is None:
        return None
    src_loc = cmap.get((leaf, port_index))
    if src_loc is None:
        return None
    src_system, _src_port_idx = src_loc
    return src_system


def find_cycles(graph: dict[Hashable, set[Hashable]]) -> list[list[Hashable]]:
    """Return a list of cycles in the dependency graph.  Each cycle is
    a list of block IDs in traversal order; an empty list means the
    graph is a DAG."""
    visited: dict[Hashable, str] = {}  # "white" / "gray" / "black"
    cycles: list[list[Hashable]] = []
    stack: list[Hashable] = []

    def _dfs(node):
        visited[node] = "gray"
        stack.append(node)
        for nxt in sorted(graph.get(node, ()), key=str):
            mark = visited.get(nxt, "white")
            if mark == "gray":
                # Found a back-edge → cycle.
                idx = stack.index(nxt)
                cycles.append(stack[idx:] + [nxt])
            elif mark == "white":
                _dfs(nxt)
        stack.pop()
        visited[node] = "black"

    for node in sorted(graph, key=str):
        if visited.get(node, "white") == "white":
            _dfs(node)
    return cycles


def topological_sort(
    graph: dict[Hashable, set[Hashable]],
    *,
    tiebreak_key: Optional[Callable[[Hashable], Any]] = None,
) -> list[Hashable]:
    """Return block IDs in lower-triangular order (every block appears
    after all its upstream dependencies).

    Args:
        graph: Adjacency map ``{node: {upstream_dependencies}}``.
        tiebreak_key: Optional callable that returns a sort key for a
            node when multiple nodes are simultaneously ready.  When
            ``None`` (default), uses ``str(node)`` — preserves the
            legacy byte-equivalent ordering from T-022a.  T-105-followup
            -priority-scheduler-hook plumbs a ``(rate_key, prio_key,
            name)`` callable through here so the lower-triangular
            scheduler honours user-declared block priorities within a
            same-rate group while still respecting topological edges.

    Raises:
        DependencyCycleError: if the graph contains a cycle.  The
            error's ``cycles`` attribute holds the offending paths
            from :func:`find_cycles`.
    """
    cycles = find_cycles(graph)
    if cycles:
        err = DependencyCycleError(
            f"Discrete-state dependency graph has {len(cycles)} cycle(s): "
            + "; ".join(" → ".join(str(n) for n in cyc) for cyc in cycles)
        )
        err.cycles = cycles
        raise err

    key = tiebreak_key if tiebreak_key is not None else str

    # Khan's algorithm — sort by upstream count, then by tiebreak key.
    in_count = {n: len(deps) for n, deps in graph.items()}
    # Reverse graph for downstream lookup.
    reverse: dict[Hashable, set[Hashable]] = {n: set() for n in graph}
    for n, deps in graph.items():
        for d in deps:
            reverse.setdefault(d, set()).add(n)
            in_count.setdefault(d, 0)

    ready = sorted([n for n, c in in_count.items() if c == 0], key=key)
    out: list[Hashable] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for nxt in sorted(reverse.get(n, ()), key=key):
            in_count[nxt] -= 1
            if in_count[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=key)

    if len(out) != len(in_count):
        # Defensive — find_cycles should have caught this.
        raise DependencyCycleError(
            "topological_sort: graph not fully sortable (unreported cycle)"
        )
    return out
