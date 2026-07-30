# SPDX-License-Identifier: MIT

"""Bounded, budgeted serialization of an influence graph.

Asking a language model "which upstream signal dominates ``x`` after the
handover?" about a five-thousand-block model has an obvious failure mode: dump
the whole structure and it does not fit; dump an arbitrary window and the answer
is missing. What makes the question answerable is that the influence graph
already knows which neighbours matter — so the window can be chosen by
*influence*, not by proximity in a file.

:func:`influence_subgraph` does exactly that: resolve the focus signals, expand
a bounded number of hops taking the strongest edges first, enrich with the units
(``framework/unit_propagation``) and rates (``simulation/rate_groups``) the
graph itself does not carry, and stop at a token budget. Every line is keyed by
a stable node id (block name path + port name), so an answer can cite one and
the citation survives a rebuild of the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .influence import InfluenceGraph


__all__ = ["influence_subgraph", "format_influence_subgraph", "CHARS_PER_TOKEN"]


# Token counting without pulling in a tokenizer dependency. Four characters per
# token is the standard rule of thumb for English-plus-identifiers text; the
# returned dict reports ``estimated_tokens`` so a caller who has a real
# tokenizer can re-measure rather than trust this.
CHARS_PER_TOKEN = 4


def _resolve_focus(graph: InfluenceGraph, focus) -> List[str]:
    """Focus specs to node ids, accepting block names as "all of that block"."""
    if isinstance(focus, str) or not isinstance(focus, Sequence):
        focus = [focus]
    resolved: List[str] = []
    for spec in focus:
        block_nodes = [
            node
            for node, data in graph.graph.nodes(data=True)
            if data["block"] == spec
        ]
        if block_nodes:
            resolved.extend(sorted(block_nodes))
            continue
        resolved.append(graph.resolve(spec))
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(resolved))


def _expand(
    graph: InfluenceGraph,
    focus: List[str],
    hops: int,
    threshold: float,
    direction: str,
    max_depth: int = 32,
) -> List[Tuple[int, float, Tuple[str, str]]]:
    """Edges within ``hops`` of the focus, ranked by what they carry TO the focus.

    Ranking by an edge's own weight is the obvious choice and it is wrong. A
    relative weight is an elasticity, so a summing junction whose two inputs
    nearly cancel has an enormous local weight — its output is a small
    difference of large numbers — while contributing almost nothing onward,
    precisely because that output is small. On a model with any near-cancelling
    junctions (an error signal, a balance, a bridge) those junctions have the
    largest weights anywhere in the graph, and a budget spent strongest-first
    goes entirely to them: measured on a 629-block plant, the serialization
    filled with junction pairs of weight ±139 and never reached the block that
    actually drove the target.

    What an edge is worth to a reader asking about the focus is the influence
    that travels *through* it: its own weight times the best onward path to the
    focus. That is what :meth:`InfluenceGraph.slice` already computes, so the
    scores come from the same traversal rather than a second heuristic.
    """
    underlying = graph.graph
    reach: Dict[str, Dict[str, float]] = {}
    for side in (("backward",) if direction == "backward"
                 else ("forward",) if direction == "forward"
                 else ("backward", "forward")):
        merged: Dict[str, float] = {}
        for node in focus:
            scores, _, _, _ = graph._reach(node, max_depth, side, threshold=0.0)
            for other, value in scores.items():
                if value > merged.get(other, -1.0):
                    merged[other] = value
        reach[side] = merged

    def carried(src: str, dst: str, side: str) -> float:
        """Influence reaching the focus through this edge."""
        data = underlying.edges[src, dst]
        if not data["local_gradient"]:
            return float("inf")      # unmeasurable: never rank it away
        onward = reach[side].get(dst if side == "backward" else src, 0.0)
        return abs(data["magnitude"]) * onward

    seen = set(focus)
    frontier = list(focus)
    collected: Dict[Tuple[str, str], Tuple[int, float]] = {}

    for hop in range(1, hops + 1):
        candidates: List[Tuple[float, Tuple[str, str], str]] = []
        for node in frontier:
            if direction in ("both", "backward"):
                for other in underlying.predecessors(node):
                    candidates.append(
                        (carried(other, node, "backward"), (other, node), other))
            if direction in ("both", "forward"):
                for other in underlying.successors(node):
                    candidates.append(
                        (carried(node, other, "forward"), (node, other), other))
        candidates.sort(key=lambda entry: -entry[0])
        next_frontier = []
        for value, edge, other in candidates:
            if value < threshold:
                continue
            if edge not in collected:
                collected[edge] = (hop, value)
            if other not in seen:
                seen.add(other)
                next_frontier.append(other)
        frontier = next_frontier
        if not frontier:
            break

    # Sorted by carried influence, NOT by hop: the budget is meant to be spent
    # on what matters most wherever it sits, so that what gets dropped is what
    # mattered least. Hop-major ordering would spend the budget on a negligible
    # one-hop edge and drop a dominant three-hop one, and the consumer of this
    # text cannot see what is missing.
    return sorted(
        ((hop, -value, edge) for edge, (hop, value) in collected.items()),
        key=lambda entry: (entry[1], entry[0], entry[2]),
    )


def _block_rates(graph: InfluenceGraph) -> Dict[str, str]:
    """Sample-time description per block, best-effort."""
    from ..simulation.rate_groups import infer_block_sample_time

    rates: Dict[str, str] = {}
    for leaf in _leaves(graph):
        try:
            sample_time = infer_block_sample_time(leaf)
        except Exception:  # noqa: BLE001 - rate inference is enrichment, not core
            continue
        # The dataclass repr spends ~50 characters of the token budget on
        # ``period=None, offset=None``; the kind plus the period is the whole
        # content.
        if sample_time.period is not None:
            rates[leaf.name_path_str] = f"{sample_time.kind}@{sample_time.period:g}s"
        else:
            rates[leaf.name_path_str] = str(sample_time.kind)
    return rates


def _block_types(graph: InfluenceGraph) -> Dict[str, str]:
    return {leaf.name_path_str: type(leaf).__name__ for leaf in _leaves(graph)}


def _leaves(graph: InfluenceGraph) -> List[Any]:
    system = graph.system
    return list(getattr(system, "leaf_systems", [system]))


def _units_text(units) -> str:
    return "-" if units is None else str(units)


def _value_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size == 1:
        return f"{float(array.reshape(())):.6g}"
    if array.size <= 4:
        return "[" + ", ".join(f"{v:.4g}" for v in array.reshape(-1)) + "]"
    flat = array.reshape(-1)
    return f"[{flat[0]:.4g} ... {flat[-1]:.4g}] (n={flat.size})"


def _weight_text(data: Dict[str, Any]) -> str:
    if not data["local_gradient"]:
        return "unknown"
    return f"{data['weight']:+.4g}"


def influence_subgraph(
    graph: InfluenceGraph,
    focus,
    *,
    budget_tokens: int = 1500,
    hops: int = 4,
    threshold: float = 0.0,
    direction: str = "both",
) -> Dict[str, Any]:
    """A bounded, budgeted, citable neighbourhood of ``focus``.

    Args:
        graph: An :class:`~jaxonomy.analysis.influence.InfluenceGraph`.
        focus: One or more focus points — node ids, port objects, name
            fragments, or a block name (which expands to all of that block's
            signals).
        budget_tokens: Approximate ceiling on the rendered text, at
            :data:`CHARS_PER_TOKEN` characters per token. Edges are dropped
            weakest-first to fit; the result reports what was dropped.
        hops: How many graph edges out from the focus to expand. Nodes are
            signals, so crossing one block costs two hops (wire in, block
            Jacobian out) — the default of 4 reaches roughly two blocks.
        threshold: Minimum edge ``|weight|`` to include at all.
        direction: ``"both"`` (default), ``"backward"`` (what influences the
            focus) or ``"forward"`` (what it influences).

    Returns:
        A dict with ``text`` (the rendered context), ``nodes``, ``edges``,
        ``blocks``, ``focus``, ``estimated_tokens``, ``dropped_edges`` and
        ``conventions``. The dict is JSON-serializable.
    """
    if direction not in ("both", "backward", "forward"):
        raise ValueError(
            f"direction must be 'both', 'backward' or 'forward', got {direction!r}"
        )
    if budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be positive, got {budget_tokens}")

    focus_nodes = _resolve_focus(graph, focus)
    ranked = _expand(graph, focus_nodes, hops, threshold, direction)

    types = _block_types(graph)
    rates = _block_rates(graph)

    # Grow the edge set strongest-first until the rendered text would exceed
    # the budget.  Rendering is cheap relative to building the graph, so the
    # budget is enforced on the real text rather than on an estimate of it.
    kept: List[Tuple[str, str]] = []
    dropped: List[Tuple[str, str]] = []
    text = format_influence_subgraph(graph, focus_nodes, [], types, rates)
    for index, (_hop, _negative, edge) in enumerate(ranked):
        candidate = kept + [edge]
        remaining = len(ranked) - index - 1
        rendered = format_influence_subgraph(
            graph, focus_nodes, candidate, types, rates, dropped_for_budget=remaining
        )
        if len(rendered) > budget_tokens * CHARS_PER_TOKEN and kept:
            # Edges arrive strongest-first, so once one does not fit, none of
            # the weaker ones would have been a better use of the budget.
            dropped = [entry[2] for entry in ranked[index:]]
            break
        kept, text = candidate, rendered
    # Re-render with the final count so the warning matches what was dropped.
    text = format_influence_subgraph(
        graph, focus_nodes, kept, types, rates, dropped_for_budget=len(dropped)
    )

    nodes = list(dict.fromkeys(focus_nodes + [n for edge in kept for n in edge]))
    return {
        "model": graph.system.name,
        "focus": focus_nodes,
        "text": text,
        "estimated_tokens": -(-len(text) // CHARS_PER_TOKEN),
        "blocks": sorted({graph.graph.nodes[n]["block"] for n in nodes}),
        "nodes": [
            {
                "id": node,
                "kind": graph.graph.nodes[node]["kind"],
                "block": graph.graph.nodes[node]["block"],
                "block_type": types.get(graph.graph.nodes[node]["block"], "?"),
                "port": graph.graph.nodes[node]["port"],
                "size": graph.graph.nodes[node]["size"],
                "value": _value_text(graph.graph.nodes[node]["value"]),
                "units": _units_text(graph.graph.nodes[node]["units"]),
                "sample_time": rates.get(graph.graph.nodes[node]["block"], "?"),
                "hybrid": bool(graph.graph.nodes[node]["hybrid"]),
            }
            for node in nodes
        ],
        "edges": [
            {
                "src": src,
                "dst": dst,
                "kind": graph.graph.edges[src, dst]["kind"],
                "weight": _weight_text(graph.graph.edges[src, dst]),
                "local_gradient": bool(
                    graph.graph.edges[src, dst]["local_gradient"]
                ),
                "note": graph.graph.edges[src, dst]["note"],
            }
            for src, dst in kept
        ],
        "dropped_edges": [{"src": src, "dst": dst} for src, dst in dropped],
        "conventions": {
            "at": graph.at,
            "normalize": graph.normalize,
            "tau_seconds": graph.tau,
            "scale_floor": graph.scale_floor,
            "reduce": graph.reduce if graph.at == "trajectory" else None,
            "weight_meaning": (
                "relative (elasticity) sensitivity; a path's product is the "
                "relative end-to-end sensitivity"
                if graph.normalize == "relative"
                else "raw partial derivative in model units"
            ),
        },
    }


def format_influence_subgraph(
    graph: InfluenceGraph,
    focus: List[str],
    edges: List[Tuple[str, str]],
    types: Optional[Dict[str, str]] = None,
    rates: Optional[Dict[str, str]] = None,
    dropped_for_budget: int = 0,
) -> str:
    """Render a node/edge selection as compact, citable text.

    The footer distinguishes the two reasons a block can be absent, because to
    a reader they mean opposite things. A block left out because its influence
    fell below the threshold is *known to be negligible* — that is an answer. A
    block left out because the budget ran out is simply *unknown*, and treating
    it as negligible would be a fabrication. Without the footer both look
    identical: missing.
    """
    types = types if types is not None else _block_types(graph)
    rates = rates if rates is not None else _block_rates(graph)
    nodes = list(dict.fromkeys(list(focus) + [n for edge in edges for n in edge]))

    lines = [
        f"# influence subgraph of model '{graph.system.name}'",
        f"# focus: {', '.join(focus)}",
        f"# weights: {'relative sensitivity (dimensionless)' if graph.normalize == 'relative' else 'raw partial derivative'}"
        f"; state-rate edges scaled by tau={graph.tau:g} s"
        f"; evaluated at={graph.at}"
        + (f" (reduce={graph.reduce})" if graph.at == "trajectory" else ""),
        "# signals: id | value | units | sample_time | block_type",
    ]
    for node in nodes:
        data = graph.graph.nodes[node]
        block = data["block"]
        marker = " *" if node in focus else ""
        lines.append(
            f"{node}{marker} | {_value_text(data['value'])} | "
            f"{_units_text(data['units'])} | {rates.get(block, '?')} | "
            f"{types.get(block, '?')}"
            + (" | hybrid" if data["hybrid"] else "")
        )
    lines.append("# edges: src -> dst | weight | kind")
    for src, dst in edges:
        data = graph.graph.edges[src, dst]
        line = f"{src} -> {dst} | {_weight_text(data)} | {data['kind']}"
        if data["note"]:
            line += f" | {data['note']}"
        lines.append(line)

    shown = {graph.graph.nodes[n]["block"] for n in nodes}
    negligible = graph.n_blocks - len(shown)
    lines.append(
        f"# coverage: {len(shown)}/{graph.n_blocks} blocks; {negligible} omitted "
        f"as below-cutoff (negligible, NOT unknown)"
    )
    if dropped_for_budget:
        lines.append(
            f"# WARNING: {dropped_for_budget} edges also dropped for budget — "
            f"anything reachable only via those is UNKNOWN, not negligible"
        )
    return "\n".join(lines)
