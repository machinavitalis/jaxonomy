# SPDX-License-Identifier: MIT

"""Sensitivity-weighted information flow: the influence graph.

``framework/dependency_graph.py`` answers *whether* information flows between
two points in a model.  Autodiff answers *how much*.  An
:class:`InfluenceGraph` is the two put together: the model's own leaf-level
dependency structure, with every edge carrying the exact local partial
derivative computed at an operating point (or resolved along a trajectory).

That combination is what makes the classic questions answerable
quantitatively rather than structurally:

* **Model slicing** — "which blocks actually matter to output ``y``?" A
  boolean slice returns everything structurally upstream. A
  :meth:`InfluenceGraph.slice` at ``threshold=0.01`` returns what carries at
  least 1% of the influence, plus the hops needed to connect it to the target.
* **Path attribution** — the chain rule along each path, so a signal that
  reaches ``y`` three ways can be decomposed, including the cancellation
  between paths that a boolean graph cannot express.
* **Bottlenecks and dead edges** — the nodes every strong path passes through,
  and the wires that exist structurally but transmit nothing here.

Conventions that make the numbers composable — read these before reading a
weight:

**Nodes** are signals and states, not blocks: one node per leaf input port,
one per leaf output port, one per state component group (``xc`` / ``xd``).
Block granularity is recovered by grouping (:attr:`InfluenceSlice.blocks`).

**Edge weights are relative (elasticity) by default.** For an edge whose local
Jacobian block is ``J``, the stored weight is
``J[i, j] · scale(src)[j] / scale(dst)[i]``, where ``scale`` is the
operating-point magnitude floored at ``scale_floor``.  This is dimensionless,
comparable across a mixed electrical/thermal/mechanical model, and — the
load-bearing property — **telescoping**: the intermediate scales cancel in a
path product, so the product of relative weights along a path equals the
relative end-to-end sensitivity ``(∂y/∂u)·scale(u)/scale(y)``.  Pass
``normalize="none"`` to store raw partials instead, in which case a path
product is literally ``∂y/∂u`` in the model's own units.

**State-mediated edges carry a time scale.** ``∂ẋ/∂u`` is a rate, so it cannot
be multiplied into a dimensionless chain as-is.  Edges whose destination is a
continuous-state *derivative* are therefore scaled by ``tau``: one explicit
Euler step, ``Δx ≈ tau · (∂ẋ/∂u) · Δu``.  ``tau`` is reported on the graph and
in every report; a weight through an integrator means "influence accumulated
over ``tau`` seconds", nothing more.  Discrete updates need no such factor —
``∂xd⁺/∂u`` is already a state-to-state map.

*``tau`` is a frequency choice, and it changes the answer.*  Every integrator on
a path contributes one factor of ``tau``, so a path crossing ``k`` states scales
as ``tau**k`` — which is exactly the magnitude of that path's transfer function
at ``ω = 1/tau``.  For a path of algebraic gains and ``k`` integrators,
``|G(jω)| = (∏ gains) / ω**k = (∏ gains) · tau**k``, and the graph reproduces
that identically (``test_influence.py::test_path_product_is_the_gain_at_one_over_tau``).

So ``tau`` selects the time scale the question is asked on, and there is no
single right value for a stiff model:

* a **large** ``tau`` (low frequency) makes integrator-rich paths dominate — the
  steady-state view, where an integral control term outweighs everything;
* a **small** ``tau`` (high frequency) makes algebraic and fast paths dominate.

Two consequences worth internalizing.  First, an absolute threshold only reads
as "1%" when ``tau`` is comparable to the time constants on the path; otherwise
scale the threshold to the strongest score, which
:meth:`InfluenceGraph.relative_threshold` computes.  Second, two nodes at
different integrator depths from the target are being compared through
different-order transfers, which is meaningful but is *not* "fraction of the
same quantity" — read a slice at one ``tau`` as one frequency of a Bode-style
sensitivity study, and re-run it at another to see the picture change.

**Multi-component signals report an upper bound.** A vector port's Jacobian
block is a matrix; the scalar edge weight is its induced ∞-norm (largest
absolute row sum), which is submultiplicative — so a path product bounds the
true end-to-end Jacobian rather than under-stating it.  Path products are
therefore exact for scalar signals and a conservative over-estimate when a
vector signal or a multi-component state sits on the path, the direction that
never drops a real dependency from a slice.  The full block is kept in
``edge["relative"]`` for drill-down.

**Non-differentiable blocks are labelled, never zeroed.** A comparator, a
quantizer, an integer mode signal, an ``io_callback`` block: the edge is
emitted with ``local_gradient=False`` and ``magnitude=nan``, and traversals
treat it as unknown-but-present — everything behind it stays in a slice and is
listed in :attr:`InfluenceSlice.unknown_nodes` — instead of pretending the
derivative is zero. What such a region does *not* get is a ranking: past an
unmeasurable edge there is no product left to maximize, so the search switches
from path enumeration to plain reachability there, and those nodes' scores are
placeholders rather than measurements. Blocks with zero-crossing events carry
``hybrid=True``: their Jacobians are exact for the mode they are currently in
and say nothing about the others.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..framework.diagram import Diagram
from ..framework.flatten import leaf_connections
from ..framework.port import InputPort, OutputPort
from .block_jacobians import (
    STATE_KINDS,
    is_sampled_output,
    leaf_jacobians,
    secant_jacobians,
)


__all__ = [
    "InfluenceGraph",
    "InfluenceSlice",
    "PathAttribution",
    "influence_graph",
]


# Edge kinds, in the order a report should present them.
WIRE = "wire"
FEEDTHROUGH = "feedthrough"
TO_STATE = "to_state"
FROM_STATE = "from_state"
STATE_TO_STATE = "state"


# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------


def _leaf_path(leaf) -> str:
    return leaf.name_path_str


def port_node_id(port) -> str:
    """Stable node id for an input or output port.

    Stable across rebuilds of the same model (it is derived from the block name
    path and port name, not from object identity or ``system_id``), which is
    what makes it usable as a citation in generated text.
    """
    direction = "in" if isinstance(port, InputPort) else "out"
    return f"{_leaf_path(port.system)}:{direction}:{port.name}"


def state_node_id(leaf, kind: str) -> str:
    return f"{_leaf_path(leaf)}:{kind}"


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------


def _flat(value) -> np.ndarray:
    """Signal value as a 1-D float vector.

    An unknown value (a block whose evaluation failed) still gets a
    one-component placeholder so the node has a well-formed size; edges out of
    such a block are labelled ``local_gradient=False`` regardless.
    """
    if value is None:
        return np.zeros(1)
    return np.atleast_1d(np.asarray(value, dtype=float).reshape(-1))


def _scale(value, floor: float) -> np.ndarray:
    """Per-component magnitude of a signal, floored away from zero."""
    return np.maximum(np.abs(_flat(value)), floor)


def _relative(jac: np.ndarray, src_scale, dst_scale, factor: float) -> np.ndarray:
    """``factor · J[i, j] · scale(src)[j] / scale(dst)[i]``."""
    return factor * jac * src_scale[None, :] / dst_scale[:, None]


def _scalarize(block: np.ndarray) -> float:
    """One number for a Jacobian block: signed if 1x1, else its induced ∞-norm.

    Sign is preserved for scalar signals because cancellation between paths is
    real information; for a matrix block there is no single sign to report.

    The matrix case uses ``max_i Σ_j |J[i, j]|`` rather than the largest single
    entry, because only a *submultiplicative* norm makes a path product an upper
    bound. ``max|AB| ≤ k · max|A| · max|B|`` for an inner dimension ``k``, so
    chaining largest-entries would under-estimate by up to a factor of ``k`` per
    hop — dropping real dependencies from a slice, the opposite of the intended
    direction. Induced norms satisfy ``‖AB‖ ≤ ‖A‖·‖B‖`` and dominate the largest
    entry, so the product bounds the true end-to-end Jacobian.
    """
    if block.size == 1:
        return float(block.reshape(()))
    return float(np.max(np.sum(np.abs(block), axis=1)))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class InfluenceSlice:
    """A quantitative model slice: what actually reaches a target.

    Attributes:
        target: Node id the slice was taken to (or from).
        threshold: Influence cutoff a path had to clear to be included.
        direction: ``"backward"`` (what influences the target) or
            ``"forward"`` (what the target influences).
        scores: ``{node_id: best |path product| between node and target}``. For
            a node reached only across an edge with no local gradient the value
            is a bound, not a measurement — see ``unknown_nodes``. Nodes kept
            only to connect an influential node to the target (see
            :meth:`InfluenceGraph.slice`) appear here with their own, possibly
            small, score.
        edges: ``(src, dst)`` pairs retained.
        blocks: Block name paths touched — the block-level slice.
        unknown_nodes: Nodes that some retained path reaches across an edge
            with no local gradient. Their score accounts only for the
            measurable routes, so it is not the whole story — treat it as a
            partial reading rather than a measurement.
        truncated: True if the path search hit its expansion budget, in which
            case the scores are lower bounds and the slice may be missing
            contributors.
        graph: The originating :class:`InfluenceGraph`.
    """

    target: str
    threshold: float
    direction: str
    scores: Dict[str, float]
    edges: List[Tuple[str, str]]
    blocks: List[str]
    unknown_nodes: List[str]
    graph: "InfluenceGraph"
    truncated: bool = False

    @property
    def unknown_paths(self) -> bool:
        """True if any retained path crosses an edge with no local gradient."""
        return bool(self.unknown_nodes)

    def __repr__(self) -> str:
        flags = ""
        if self.unknown_paths:
            flags += ", unknown paths"
        if self.truncated:
            flags += ", TRUNCATED"
        return (
            f"InfluenceSlice({self.target}, threshold={self.threshold:g}, "
            f"{len(self.blocks)} blocks, {len(self.scores)} nodes{flags})"
        )

    @property
    def subgraph(self) -> nx.DiGraph:
        """The retained portion of the influence graph.

        Built from the retained nodes *and* edges rather than as an edge-induced
        view, so a node with no retained edge — the target of a slice that keeps
        nothing else — is still present.
        """
        view = nx.DiGraph()
        for node in self.scores:
            view.add_node(node, **self.graph.graph.nodes[node])
        for src, dst in self.edges:
            view.add_edge(src, dst, **self.graph.graph.edges[src, dst])
        return view

    def report(self) -> str:
        lines = [
            f"Influence slice ({self.direction}) for {self.target}",
            f"  threshold={self.threshold:g}  normalize={self.graph.normalize}  "
            f"tau={self.graph.tau:g}  at={self.graph.at}",
            f"  {len(self.blocks)} of {self.graph.n_blocks} blocks retained",
        ]
        unknown = set(self.unknown_nodes)
        ranked = sorted(self.scores.items(), key=lambda kv: -abs(kv[1]))
        for node, score in ranked:
            if node == self.target:
                continue
            flag = "  (bound)" if node in unknown else ""
            lines.append(f"    {score:>10.4g}  {node}{flag}")
        if unknown:
            lines.append(
                f"  NOTE: {len(unknown)} nodes are reached only across an edge with "
                f"no local gradient; they are kept unconditionally and their "
                f"scores are upper bounds."
            )
        if self.truncated:
            lines.append(
                "  NOTE: the path search hit its expansion budget — these scores "
                "are lower bounds and contributors may be missing. Lower "
                "max_depth or raise the threshold."
            )
        return "\n".join(lines)


@dataclass
class PathAttribution:
    """Chain-rule decomposition of one source's influence on one target.

    Attributes:
        target: Destination node id.
        source: Origin node id.
        paths: One entry per path, ranked by ``|product|``, each a dict with
            ``nodes``, ``product``, ``signed`` (False when a matrix block on
            the path made the sign meaningless) and ``unknown`` (True when an
            edge on the path has no local gradient).
        total: Signed sum of path products when every path is signed, else
            ``None`` — a sum of magnitudes would hide cancellation.
        total_magnitude: Sum of ``|product|`` over paths, always available.
        truncated: True if enumeration hit ``max_paths`` or ``max_depth``.
    """

    target: str
    source: str
    paths: List[Dict[str, Any]]
    total: Optional[float]
    total_magnitude: float
    truncated: bool

    def __repr__(self) -> str:
        total = "n/a" if self.total is None else f"{self.total:.4g}"
        return (
            f"PathAttribution({self.source} -> {self.target}, "
            f"{len(self.paths)} paths, total={total})"
        )

    def report(self, max_paths: int = 10) -> str:
        lines = [f"Attribution {self.source} -> {self.target}"]
        if self.total is not None:
            lines.append(f"  total (signed sum over paths): {self.total:.6g}")
        lines.append(f"  total magnitude: {self.total_magnitude:.6g}")
        for entry in self.paths[:max_paths]:
            flags = []
            if not entry["signed"]:
                flags.append("magnitude-only")
            if entry["unknown"]:
                flags.append("no local gradient on path")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {entry['product']:+.6g}{suffix}")
            lines.append(f"      {' -> '.join(entry['nodes'])}")
        if len(self.paths) > max_paths:
            lines.append(f"  ... {len(self.paths) - max_paths} more paths")
        if self.truncated:
            lines.append("  NOTE: enumeration truncated; totals are partial.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass
class InfluenceGraph:
    """A model's dependency structure with autodiff-computed edge weights.

    Build with :func:`influence_graph`; see this module's docstring for the
    weighting conventions the numbers obey.

    Attributes:
        system: The analyzed ``Diagram`` or ``LeafSystem``.
        graph: The underlying ``networkx.DiGraph``. Node attributes describe
            the signal (``kind``, ``block``, ``port``, ``size``, ``value``,
            ``units``, ``sample_time``, ``hybrid``); edge attributes carry
            ``kind``, ``jacobian``, ``relative``, ``weight``, ``magnitude``,
            ``local_gradient``, ``note`` and — in trajectory mode —
            ``profile``.
        tau: Time scale applied to continuous-state-rate edges, in seconds.
        normalize: ``"relative"`` or ``"none"``.
        scale_floor: Lower bound on a signal's magnitude when normalizing.
        at: ``"operating_point"`` or ``"trajectory"``.
        times: Snapshot times in trajectory mode, else None.
        reduce: How a trajectory profile became the scalar weight.
        block_notes: Per-block explanations for anything not differentiated.
    """

    system: Any
    graph: nx.DiGraph
    tau: float
    normalize: str
    scale_floor: float
    at: str
    times: Optional[np.ndarray]
    reduce: str
    block_notes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    structure: Optional[nx.DiGraph] = None

    # -- basics ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InfluenceGraph({self.system.name}, {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, at={self.at!r}, tau={self.tau:g})"
        )

    @property
    def n_blocks(self) -> int:
        return len({d["block"] for _, d in self.graph.nodes(data=True)})

    @property
    def blocks(self) -> List[str]:
        return sorted({d["block"] for _, d in self.graph.nodes(data=True)})

    def resolve(self, spec) -> str:
        """Turn a port object, locator, or name fragment into a node id.

        Accepts an exact node id, an ``InputPort`` / ``OutputPort``, a
        ``(system, port_index)`` locator, or any unambiguous suffix of a node
        id (``"integ:out:out_0"``, ``"integ"``, ``"out:y"``).
        """
        if isinstance(spec, (InputPort, OutputPort)):
            return self._require(port_node_id(spec))
        if isinstance(spec, tuple) and len(spec) == 2:
            system, index = spec
            ports = system.output_ports if hasattr(system, "output_ports") else []
            if index < len(ports):
                return self._require(port_node_id(ports[index]))
        if not isinstance(spec, str):
            raise TypeError(
                f"Cannot resolve {spec!r} to an influence-graph node; pass a node "
                f"id, a port object, or a (system, port_index) locator."
            )
        if spec in self.graph:
            return spec
        matches = [n for n in self.graph if n.endswith(spec) or spec in n]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(
                f"No influence-graph node matches {spec!r}. "
                f"Nodes are named '<block path>:in|out:<port>' or "
                f"'<block path>:xc|xd'; {self.graph.number_of_nodes()} exist, "
                f"e.g. {sorted(self.graph)[:4]}."
            )
        raise KeyError(
            f"{spec!r} is ambiguous — it matches {len(matches)} nodes: "
            f"{sorted(matches)[:6]}. Use a longer fragment or the full node id."
        )

    def _require(self, node_id: str) -> str:
        if node_id not in self.graph:
            raise KeyError(
                f"{node_id!r} is not in this influence graph. It may belong to a "
                f"different model, or be a port of a sub-Diagram rather than a leaf."
            )
        return node_id

    # -- traversal ---------------------------------------------------------

    def _step_weight(self, src: str, dst: str) -> Tuple[float, bool]:
        """(magnitude used for propagation, whether it is actually known)."""
        data = self.graph.edges[src, dst]
        if not data["local_gradient"]:
            # Unknown, not zero. The unit gain is a placeholder that keeps a
            # product finite for reporting; callers must branch on the second
            # element rather than treat this as a measurement.
            return 1.0, False
        return abs(data["magnitude"]), True

    def _amplification(
        self, direction: str, max_depth: int
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, bool]]]:
        """The pruning bound: how much a continuation could still contribute.

        Returns ``(bounds, unmeasurable)``, both indexed by remaining depth.
        ``bounds[d][n]`` is the largest product over *measurable* paths of ≤ ``d``
        edges arriving at ``n``; ``unmeasurable[d][n]`` says whether any such
        path crosses an edge with no local gradient.

        The bound exists because a running product is not an upper bound on the
        finished path's — a relative weight is an elasticity and can exceed 1, so
        a partial path can dip below any threshold and climb back. Relaxing over
        depth gives the largest factor still available, in ``max_depth`` sweeps
        of the edge list, and being an over-estimate it can only prune paths that
        could not have qualified.

        The two results are kept separate on purpose. An unmeasurable
        continuation must never be pruned — "we could not compute it" is not
        "it is small" — but folding that into the *numeric* bound as an infinite
        gain would propagate infinity to every node upstream of any comparator
        or quantizer, switching pruning off across the whole model and
        collapsing the search back to an exponential one. A boolean reachability
        flag suppresses pruning exactly where an unknown edge is actually
        within reach, and leaves the numeric bound finite everywhere else.
        """
        incoming = (
            self.graph.predecessors if direction == "backward" else self.graph.successors
        )
        bounds = [{node: 1.0 for node in self.graph}]
        unmeasurable = [{node: False for node in self.graph}]
        for _ in range(max_depth):
            previous_bound, previous_flag = bounds[-1], unmeasurable[-1]
            current_bound, current_flag = dict(previous_bound), dict(previous_flag)
            for node in self.graph:
                for other in incoming(node):
                    if other == node:
                        continue
                    edge = (
                        (other, node) if direction == "backward" else (node, other)
                    )
                    magnitude, known = self._step_weight(*edge)
                    if not known:
                        current_flag[node] = True
                        continue
                    candidate = previous_bound[other] * magnitude
                    if candidate > current_bound[node]:
                        current_bound[node] = candidate
                    if previous_flag[other]:
                        current_flag[node] = True
            if current_bound == previous_bound and current_flag == previous_flag:
                break
            bounds.append(current_bound)
            unmeasurable.append(current_flag)
        return bounds, unmeasurable

    def _reach(
        self,
        target: str,
        max_depth: int,
        direction: str,
        threshold: float = 0.0,
        max_expansions: int = 200_000,
        amplification: Optional[
            Tuple[List[Dict[str, float]], List[Dict[str, bool]]]
        ] = None,
    ) -> Tuple[
        Dict[str, float], Dict[str, bool], Dict[str, Tuple[str, ...]], bool
    ]:
        """Best *simple*-path product between every node and ``target``.

        Three properties this has to get right, each of which a more obvious
        implementation gets wrong on a real model:

        **Simple paths only.** Allowing a path to revisit a node lets it
        circulate a feedback loop, multiplying by the loop gain each turn, and a
        node's "influence" then reports how many times the search went round
        rather than how much signal gets through — in a closed loop it even
        makes a node influence *itself* by a large factor. Restricting to simple
        paths is also the standard signal-flow-graph notion of a forward path,
        and matches what :meth:`attribute` enumerates.

        **No pruning on the running product.** A relative weight is an
        elasticity, not a gain bounded by 1: a summing junction whose output
        nearly cancels (any controller error signal near steady state) has an
        elasticity far above 1, so a partial product can dip below any threshold
        and climb back above it. Pruning uses :meth:`_amplification` instead,
        which bounds what a continuation could still contribute.

        **Unmeasurable edges are reachability, not search.** Past an edge with
        no local gradient there is no product left to maximize, and *before* one
        there is nothing to optimize either — only the question of what connects.
        Both are answered by two linear BFS sweeps after the path search, so a
        comparator or quantizer anywhere upstream costs a pass over the edge
        list rather than an unpruned walk of everything between it and the
        target.

        Returns ``(best, unknown, routes, truncated)``, where ``routes[n]`` is
        the node sequence realizing ``best[n]`` (from ``n`` to ``target``, or
        the reverse for a forward search) and ``truncated`` is True when the
        expansion budget ran out, in which case the scores are lower bounds.
        Scores for nodes the sweeps supplied are lower bounds by construction —
        one real route's product rather than the best one's.
        """
        step = (
            self.graph.predecessors if direction == "backward" else self.graph.successors
        )
        if amplification is None and threshold > 0.0:
            amplification = self._amplification(direction, max_depth)
        best: Dict[str, float] = {target: 1.0}
        unknown: Dict[str, bool] = {target: False}
        # The route realizing each node's best product. Slicing needs it to stay
        # connected: retaining nodes by their own influence alone leaves the
        # intermediate hops of a dominant route out, and a disconnected slice
        # makes `subgraph` and `bottlenecks` meaningless.
        routes: Dict[str, Tuple[str, ...]] = {target: (target,)}
        truncated = False
        expansions = 0

        # Path enumeration runs over fully measurable routes only, and prunes
        # on the numeric bound alone. Everything to do with unmeasurable edges
        # is handled by the two reachability sweeps below, because *both* halves
        # of that question — getting to an unknown edge and going past it — are
        # reachability, not optimization. Letting the DFS off its leash to find
        # the unknown frontier (the obvious alternative) means walking a dense
        # region as an unpruned simple-path enumeration before ever arriving at
        # the comparator that made it necessary.
        stack = [(target, 1.0, frozenset((target,)), 0, (target,))]
        while stack:
            node, product, trail, depth, route = stack.pop()
            if depth >= max_depth:
                continue
            expansions += 1
            if expansions > max_expansions:
                truncated = True
                break
            for other in step(node):
                if other in trail:
                    continue  # a repeat would be a loop turn, not a new path
                edge = (other, node) if direction == "backward" else (node, other)
                magnitude, known = self._step_weight(*edge)
                if not known:
                    continue  # picked up by the frontier sweep below
                new_product = product * magnitude
                new_route = (other,) + route
                if amplification is not None:
                    bounds, _ = amplification
                    remaining = min(max_depth - depth - 1, len(bounds) - 1)
                    if new_product * bounds[remaining][other] < threshold:
                        continue
                previous = best.get(other)
                if previous is None or new_product > previous:
                    best[other] = new_product
                    routes[other] = new_route
                stack.append(
                    (other, new_product, trail | {other}, depth + 1, new_route)
                )

        # Sweep 1: how far is each node from the target, by what route, and what
        # that route carries. Plain BFS, so it costs one pass whatever the
        # model's density. The product is needed because a seed's route runs
        # through nodes the DFS pruned; without a score of their own they would
        # be dropped from the slice while their edges were kept, leaving the
        # retained set inconsistent with the subgraph.
        hops: Dict[str, int] = {target: 0}
        to_target: Dict[str, Tuple[str, ...]] = {target: (target,)}
        hop_product: Dict[str, float] = {target: 1.0}
        hop_unknown: Dict[str, bool] = {target: False}
        frontier_queue = deque([target])
        while frontier_queue:
            node = frontier_queue.popleft()
            if hops[node] >= max_depth:
                continue
            for other in step(node):
                if other == node or other in hops:
                    continue
                edge = (other, node) if direction == "backward" else (node, other)
                magnitude, known = self._step_weight(*edge)
                hops[other] = hops[node] + 1
                to_target[other] = (other,) + to_target[node]
                hop_product[other] = hop_product[node] * magnitude
                hop_unknown[other] = hop_unknown[node] or not known
                frontier_queue.append(other)

        # Sweep 2: seed at every unmeasurable edge that can still reach the
        # target, then walk away from it. This is what keeps a comparator, a
        # quantizer, or anything behind them in the answer.
        unknown_seeds: List[Tuple[str, int, float, Tuple[str, ...]]] = []
        for source_node, destination, data in self.graph.edges(data=True):
            if data["local_gradient"] or source_node == destination:
                continue
            head, tail = (
                (destination, source_node)
                if direction == "backward"
                else (source_node, destination)
            )
            reached = hops.get(head)
            if reached is None or reached + 1 > max_depth:
                continue
            # Everything the seed's route passes through has to carry a score
            # too, or it would be filtered out of the slice while its edges
            # stayed in. The BFS route is one real path, so its product is a
            # lower bound on that node's influence — honest, and finite.
            for hop in to_target[head]:
                best.setdefault(hop, hop_product[hop])
                routes.setdefault(hop, to_target[hop])
                if hop_unknown[hop]:
                    unknown[hop] = True
            route = (tail,) + to_target[head]
            unknown[tail] = True
            best.setdefault(tail, 1.0)
            routes.setdefault(tail, route)
            unknown_seeds.append((tail, reached + 1, best[tail], route))

        # Scores past an unmeasurable edge are placeholders, not measurements —
        # `unknown` marks them so no caller reads them as one.
        queue = deque(unknown_seeds)
        shallowest: Dict[str, int] = {}
        while queue:
            node, depth, product, route = queue.popleft()
            if depth >= max_depth or shallowest.get(node, max_depth + 1) <= depth:
                continue
            shallowest[node] = depth
            for other in step(node):
                if other == node or other in route:
                    continue
                unknown[other] = True
                best.setdefault(other, product)
                extended = (other,) + route
                routes.setdefault(other, extended)
                queue.append((other, depth + 1, product, extended))

        return best, unknown, routes, truncated

    def slice(
        self,
        target,
        threshold: float = 0.01,
        *,
        direction: str = "backward",
        max_depth: int = 32,
    ) -> InfluenceSlice:
        """Quantitative model slice: what influences ``target`` by ≥ ``threshold``.

        The boolean answer — everything structurally upstream — is
        :meth:`structural_slice`; this one keeps only what lies on a path
        carrying at least ``threshold`` of the influence (in the
        relative-sensitivity sense described in the module docstring, so
        ``0.01`` reads as "1%").

        Two kinds of node are kept, and the distinction is load-bearing. A node
        is **influential** when its own best path to ``target`` clears the
        threshold. It is a **connector** when it merely lies on some influential
        node's best route: a relative weight is an elasticity, so a signal can
        pass through a junction that nearly cancels it and be amplified back
        afterwards, leaving a mid-route node with a small score of its own.
        Keeping only the influential ones would punch holes in the result —
        naming a block as influential while the route from it to the target ran
        through blocks that had been dropped, leaving
        :attr:`InfluenceSlice.subgraph` disconnected and :meth:`bottlenecks`
        meaningless. Connectors are read off the routes the search actually
        found, so nothing is added that no real path uses.

        ``scores`` reports every retained node's own best product to the target,
        which is the number to rank by.

        Args:
            target: Node id, port object, or name fragment (see :meth:`resolve`).
            threshold: Minimum ``|path product|`` for a path to be retained.
            direction: ``"backward"`` (default, what influences the target) or
                ``"forward"`` (what the target influences).
            max_depth: Hard bound on path length, and the only hard bound — a
                partial product is not a bound on the whole path's (see
                :meth:`_reach`), so nothing may be pruned on the running value.

        Returns:
            An :class:`InfluenceSlice`.
        """
        if direction not in ("backward", "forward"):
            raise ValueError(
                f"direction must be 'backward' or 'forward', got {direction!r}"
            )
        node = self.resolve(target)
        best, unknown, routes, truncated = self._reach(
            node, max_depth, direction, threshold=threshold
        )

        influential = {
            other
            for other, score in best.items()
            if other == node or score >= threshold or unknown.get(other, False)
        }
        # Pull in whatever each influential node's own best route passes
        # through, so the result is a connected sub-model rather than a set of
        # names with no way to get between them.
        retained_nodes = set(influential)
        for other in influential:
            retained_nodes.update(routes.get(other, (other,)))

        # Every node on a retained route gets a score. Dropping the ones the
        # numeric search never scored would leave `blocks` naming fewer blocks
        # than `edges` actually connects — the retained set and the subgraph
        # would disagree, and the route's interior blocks would vanish from the
        # answer even though they are the only way the influence travels.
        retained = {
            other: best[other] if other in best else 0.0 for other in retained_nodes
        }
        retained.setdefault(node, best[node])

        kept = set()
        for other in influential:
            route = routes.get(other, ())
            for first, second in zip(route, route[1:]):
                kept.add(
                    (first, second) if direction == "backward" else (second, first)
                )
        # Beyond the best routes, keep any edge between retained nodes that
        # itself carries a qualifying path — a parallel branch of comparable
        # strength belongs in the slice even though some other route was best.
        for other in retained:
            neighbours = (
                self.graph.successors(other)
                if direction == "backward"
                else self.graph.predecessors(other)
            )
            for downstream in neighbours:
                if downstream == other or downstream not in retained:
                    continue
                edge = (
                    (other, downstream)
                    if direction == "backward"
                    else (downstream, other)
                )
                magnitude, known = self._step_weight(*edge)
                if not known or magnitude * retained[downstream] >= threshold:
                    kept.add(edge)

        blocks = sorted({self.graph.nodes[n]["block"] for n in retained})
        return InfluenceSlice(
            target=node,
            threshold=threshold,
            direction=direction,
            scores=dict(retained),
            edges=sorted(kept),
            blocks=blocks,
            unknown_nodes=sorted(
                other for other in retained if unknown.get(other, False)
            ),
            graph=self,
            truncated=truncated,
        )

    def relative_threshold(
        self,
        target,
        fraction: float = 0.01,
        *,
        direction: str = "backward",
        max_depth: int = 32,
        floor: float = 1e-12,
    ) -> float:
        """A threshold set at ``fraction`` of the strongest influence on ``target``.

        An absolute threshold only reads as a percentage when ``tau`` is
        comparable to the time constants on the paths involved (see the module
        docstring). Scaling to the strongest score makes "keep what carries at
        least 1% of what the dominant contributor carries" mean the same thing at
        any ``tau``.

        Args:
            target: Node id, port object, or name fragment.
            fraction: Fraction of the strongest score to keep.
            direction: As in :meth:`slice`.
            max_depth: As in :meth:`slice`.
            floor: Threshold the reference sweep runs at, and the value returned
                when nothing upstream carries influence. It is passed to the
                search rather than left at zero so the sweep stays pruned; a
                model whose strongest contributor falls below it would yield
                ``floor`` itself.

        Returns:
            A threshold to pass to :meth:`slice` / :meth:`bottlenecks`.
        """
        node = self.resolve(target)
        best, _, _, _ = self._reach(node, max_depth, direction, threshold=floor)
        others = [value for other, value in best.items() if other != node]
        if not others or max(others) <= 0.0:
            return floor
        return max(fraction * max(others), floor)

    def structural_slice(self, target, *, direction: str = "backward") -> List[str]:
        """Boolean slice: every block structurally connected to ``target``.

        The over-approximation :meth:`slice` improves on, computed from the
        model's declared connectivity rather than from the weighted graph — so
        it stays a genuine bound even where a Jacobian could not be taken.
        Provided so the two can be compared directly on a real model.
        """
        node = self.resolve(target)
        structure = self.structure if self.structure is not None else self.graph
        if node not in structure:
            return [self.graph.nodes[node]["block"]]
        reached = (
            nx.ancestors(structure, node)
            if direction == "backward"
            else nx.descendants(structure, node)
        )
        return sorted({structure.nodes[n]["block"] for n in reached | {node}})

    def _enumerate_paths(
        self,
        source: str,
        target: str,
        threshold: float,
        max_depth: int,
        max_paths: int,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        paths: List[Dict[str, Any]] = []
        truncated = False

        # The running product alone is not a bound on the finished path's — an
        # elasticity above 1 further along can lift it back over the threshold
        # (see :meth:`_reach`). Pruning is therefore done against
        # running x (the most any continuation could still contribute), which is
        # admissible and so never discards a path that would have qualified.
        #
        # This bound comes from :meth:`_amplification`, not from :meth:`_reach`:
        # `_reach` is itself a simple-path DFS, and calling it here without a
        # threshold would be an unpruned exponential search whose expansion
        # budget could silently cut off nodes — making this enumeration report
        # "no path" while claiming it was complete.
        onward_bounds, onward_unmeasurable = self._amplification(
            "forward", max_depth
        )

        # Pruning is switched off once a path turns unknown (its product is a
        # placeholder, so the bound says nothing), which leaves the walk through
        # an unmeasurable region unbounded. `max_paths` only fires when paths are
        # actually found, so a region that reaches the target rarely needs this
        # second cap.
        expansions = 0
        max_expansions = 200_000

        stack = [(source, [source], 1.0, True, False)]
        while stack:
            node, trail, product, signed, unknown = stack.pop()
            if node == target and len(trail) > 1:
                paths.append(
                    {
                        "nodes": list(trail),
                        "product": product,
                        "signed": signed,
                        "unknown": unknown,
                    }
                )
                if len(paths) >= max_paths:
                    truncated = True
                    break
                continue
            if len(trail) > max_depth:
                truncated = True
                continue
            expansions += 1
            if expansions > max_expansions:
                truncated = True
                break
            for successor in self.graph.successors(node):
                if successor in trail:
                    continue  # a loop turn cannot add a new path
                data = self.graph.edges[node, successor]
                if not data["local_gradient"]:
                    stack.append(
                        (successor, trail + [successor], product, False, True)
                    )
                    continue
                weight = data["weight"]
                block = data["relative"]
                step_signed = signed and block is not None and block.size == 1
                new_product = product * (weight if step_signed else abs(weight))
                if not unknown:
                    depth_left = min(max_depth - len(trail), len(onward_bounds) - 1)
                    if depth_left < 0:
                        continue
                    # As in `_reach`: a branch that could still reach an
                    # unmeasurable edge is never pruned.
                    if not onward_unmeasurable[depth_left].get(successor, False) and (
                        abs(new_product) * onward_bounds[depth_left].get(successor, 1.0)
                        < threshold
                    ):
                        continue
                stack.append(
                    (successor, trail + [successor], new_product, step_signed, unknown)
                )
        paths.sort(key=lambda entry: -abs(entry["product"]))
        return paths, truncated

    def attribute(
        self,
        target,
        source,
        *,
        threshold: float = 1e-6,
        max_depth: int = 32,
        max_paths: int = 512,
    ) -> PathAttribution:
        """Decompose ``source``'s influence on ``target`` path by path.

        Each path's contribution is the chain-rule product of its edge weights;
        the signed sum over paths is the end-to-end sensitivity, which is where
        cancellation between two routes shows up as a total far below the
        largest single path.

        Args:
            target: Destination node (id, port, or fragment).
            source: Origin node.
            threshold: Prune a path once ``|product|`` falls below this.
            max_depth: Maximum path length.
            max_paths: Stop after this many paths and mark the result
                truncated, rather than enumerating a combinatorial blow-up.
        """
        src = self.resolve(source)
        dst = self.resolve(target)
        paths, truncated = self._enumerate_paths(
            src, dst, threshold, max_depth, max_paths
        )
        all_signed = bool(paths) and all(
            entry["signed"] and not entry["unknown"] for entry in paths
        )
        return PathAttribution(
            target=dst,
            source=src,
            paths=paths,
            total=sum(entry["product"] for entry in paths) if all_signed else None,
            total_magnitude=sum(abs(entry["product"]) for entry in paths),
            truncated=truncated,
        )

    def dominant_paths(
        self,
        target,
        k: int = 5,
        *,
        source=None,
        threshold: float = 1e-6,
        max_depth: int = 32,
        max_paths: int = 512,
    ) -> List[Dict[str, Any]]:
        """The ``k`` strongest paths into ``target`` (optionally from ``source``).

        With no ``source``, every node with no in-edges inside the search — the
        model's genuine independent inputs and states — is used as an origin.
        """
        dst = self.resolve(target)
        if source is not None:
            return self.attribute(
                dst,
                source,
                threshold=threshold,
                max_depth=max_depth,
                max_paths=max_paths,
            ).paths[:k]

        # Reuse the slice rather than an unpruned reachability sweep: it applies
        # the same admissible bound, so the candidate origins are found without
        # an exponential search that could silently truncate.
        origin_slice = self.slice(dst, threshold, max_depth=max_depth)
        if origin_slice.truncated:
            warnings.warn(
                f"dominant_paths({target!r}): the slice used to find candidate "
                f"origins hit its search budget, so the ranking may be missing "
                f"stronger paths. Raise the threshold or lower max_depth.",
                UserWarning,
                stacklevel=2,
            )
        reachable = origin_slice.scores
        origins = [
            node
            for node in reachable
            if node != dst and self.graph.in_degree(node) == 0
        ]
        if not origins:
            # Every candidate origin is driven by something — a closed loop.
            # The states are then the model's independent variables.
            origins = [
                node
                for node in reachable
                if node != dst and self.graph.nodes[node]["kind"] == "state"
            ]
        collected: List[Dict[str, Any]] = []
        for origin in origins:
            paths, _ = self._enumerate_paths(
                origin, dst, threshold, max_depth, max_paths
            )
            collected.extend(paths)
        collected.sort(key=lambda entry: -abs(entry["product"]))
        return collected[:k]

    def dead_edges(self, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Structural edges that transmit no influence at this operating point.

        A wire the model declares and the mathematics ignores: a gain of zero, a
        saturated nonlinearity, a term that cancels. This is the quantitative
        form of a dead-store warning — the connection is real, the influence is
        not. Edges with no local gradient are excluded (unknown is not dead), and
        so are the state self-loops, whose zero A block is the *definition* of a
        plain integrator rather than a defect.
        """
        found = []
        for src, dst, data in self.graph.edges(data=True):
            if not data["local_gradient"] or src == dst:
                continue
            if abs(data["magnitude"]) <= threshold:
                found.append(
                    {
                        "src": src,
                        "dst": dst,
                        "kind": data["kind"],
                        "magnitude": abs(data["magnitude"]),
                    }
                )
        found.sort(key=lambda entry: (entry["magnitude"], entry["src"]))
        return found

    def bottlenecks(
        self,
        target,
        *,
        threshold: float = 0.01,
        max_depth: int = 32,
    ) -> List[str]:
        """Nodes every influential path to ``target`` must pass through.

        Computed on the slice at ``threshold``: a node is a bottleneck when
        deleting it disconnects at least one slice origin from ``target``. These
        are the signals worth instrumenting, and the single points of failure in
        a redundancy argument.

        Returns a bare list, so it has nowhere to report that the underlying
        slice was truncated — a truncated slice is missing paths, and a missing
        path is exactly what turns a non-bottleneck into an apparent one. That
        case warns instead; take the slice yourself and check
        :attr:`InfluenceSlice.truncated` if you need to handle it.
        """
        model_slice = self.slice(target, threshold, max_depth=max_depth)
        if model_slice.truncated:
            warnings.warn(
                f"bottlenecks({target!r}): the underlying slice hit its search "
                f"budget, so paths are missing and a node can look like a "
                f"single point of failure when it is not. Raise the threshold "
                f"or lower max_depth.",
                UserWarning,
                stacklevel=2,
            )
        subgraph = model_slice.subgraph
        subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
        node = model_slice.target
        if node not in subgraph:
            return []
        origins = [
            other
            for other in subgraph
            if other != node and subgraph.in_degree(other) == 0
        ]
        if not origins:
            return []
        bottleneck = []
        for candidate in subgraph:
            if candidate == node or candidate in origins:
                continue
            trimmed = subgraph.copy()
            trimmed.remove_node(candidate)
            for origin in origins:
                if origin in trimmed and not nx.has_path(trimmed, origin, node):
                    bottleneck.append(candidate)
                    break
        return sorted(bottleneck)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        """Human-readable overview: size, conventions, and honesty labels."""
        unknown = [
            (src, dst)
            for src, dst, data in self.graph.edges(data=True)
            if not data["local_gradient"]
        ]
        hybrid = sorted(
            {d["block"] for _, d in self.graph.nodes(data=True) if d["hybrid"]}
        )
        lines = [
            f"InfluenceGraph for {self.system.name}",
            f"  {self.n_blocks} blocks, {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges",
            f"  at={self.at}  normalize={self.normalize}  tau={self.tau:g} s"
            f"  scale_floor={self.scale_floor:g}",
        ]
        if self.at == "trajectory":
            lines.append(
                f"  {len(self.times)} snapshots over "
                f"[{self.times[0]:g}, {self.times[-1]:g}] s, reduce={self.reduce}"
            )
        if unknown:
            lines.append(f"  {len(unknown)} edges with no local gradient:")
            for src, dst in unknown[:8]:
                note = self.graph.edges[src, dst]["note"]
                lines.append(f"    {src} -> {dst}  ({note})")
            if len(unknown) > 8:
                lines.append(f"    ... {len(unknown) - 8} more")
        if hybrid:
            lines.append(
                f"  {len(hybrid)} hybrid blocks (weights valid for the current "
                f"mode only): {', '.join(hybrid[:6])}"
            )
        floored = self.nodes_at_scale_floor()
        if floored:
            lines.append(
                f"  {len(floored)} signals sit at the scale floor (value ~0 at this "
                f"operating point), so elasticities through them are inflated by "
                f"the floor rather than measured: {', '.join(sorted(floored)[:5])}"
                + (f", ... (+{len(floored) - 5})" if len(floored) > 5 else "")
            )
            lines.append(
                "    Analyze at a settled operating point, raise scale_floor, or "
                "use normalize='none' if this matters."
            )
        dead = self.dead_edges()
        if dead:
            lines.append(f"  {len(dead)} dead edges (structural, zero influence)")
        return "\n".join(lines)

    def nodes_at_scale_floor(self) -> List[str]:
        """Signals whose normalizer came from ``scale_floor``, not from a value.

        A relative weight divides by the signal's magnitude, so a signal that is
        (near) zero at the operating point — an error signal at equilibrium, an
        integrator state at ``t=0`` — produces an elasticity governed by
        ``scale_floor`` rather than by the model. Those weights are not wrong so
        much as meaningless, and they are large, so they dominate any ranking.
        """
        if self.normalize != "relative":
            return []
        return sorted(
            node
            for node, data in self.graph.nodes(data=True)
            if "scale" in data
            and bool(np.any(np.asarray(data["scale"]) <= self.scale_floor))
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _node_specs(leaf, jacs) -> List[Tuple[str, Dict[str, Any]]]:
    """Node ids and attributes contributed by one leaf, with its values."""
    path = _leaf_path(leaf)
    hybrid = bool(leaf.has_zero_crossing_events)
    specs = []
    for index, port in enumerate(leaf.input_ports):
        value = jacs.u0[index] if index < len(jacs.u0) else None
        specs.append(
            (
                port_node_id(port),
                dict(
                    kind="input",
                    block=path,
                    port=port.name,
                    port_index=port.index,
                    value=_flat(value),
                    units=getattr(port, "units", None),
                    hybrid=hybrid,
                    sampled=False,
                ),
            )
        )
    for index, port in enumerate(leaf.output_ports):
        value = jacs.y0[index] if index < len(jacs.y0) else port.default_value
        specs.append(
            (
                port_node_id(port),
                dict(
                    kind="output",
                    block=path,
                    port=port.name,
                    port_index=port.index,
                    value=_flat(value),
                    units=getattr(port, "units", None),
                    hybrid=hybrid,
                    sampled=is_sampled_output(port),
                ),
            )
        )
    for kind in STATE_KINDS:
        if kind in jacs.x0:
            specs.append(
                (
                    state_node_id(leaf, kind),
                    dict(
                        kind="state",
                        state_kind=kind,
                        block=path,
                        port=kind,
                        port_index=None,
                        value=_flat(jacs.x0[kind]),
                        units=None,
                        hybrid=hybrid,
                        sampled=False,
                    ),
                )
            )
    for node_id, attributes in specs:
        attributes["size"] = int(attributes["value"].size)
    return specs


def _set_edge(
    graph: nx.DiGraph,
    src: str,
    dst: str,
    kind: str,
    jac: Optional[np.ndarray],
    relative: Optional[np.ndarray],
    *,
    local_gradient: bool = True,
    note: str = "",
    tau_applied: bool = False,
):
    weight = _scalarize(relative) if relative is not None else math.nan
    graph.add_edge(
        src,
        dst,
        kind=kind,
        jacobian=jac,
        relative=relative,
        weight=weight,
        magnitude=abs(weight),
        local_gradient=local_gradient,
        note=note,
        tau_applied=tau_applied,
    )


def _leaf_systems(system) -> List[Any]:
    return list(system.leaf_systems) if isinstance(system, Diagram) else [system]


def _snapshot_times(context, results, times, n_snapshots) -> np.ndarray:
    if times is not None:
        requested = np.atleast_1d(np.asarray(times, dtype=float))
    elif results is not None and results.time is not None:
        available = np.asarray(results.time, dtype=float)
        if n_snapshots >= available.size:
            requested = available
        else:
            picks = np.linspace(0, available.size - 1, n_snapshots)
            requested = available[np.round(picks).astype(int)]
    else:
        raise ValueError(
            "influence_graph(at='trajectory') needs either results= (a "
            "SimulationResults, whose time vector supplies the snapshot times) "
            "or an explicit times= sequence."
        )
    t0 = float(context.time)
    requested = np.unique(requested[requested >= t0])
    if requested.size == 0:
        raise ValueError(
            f"No snapshot time is at or after the context time t={t0:g}; "
            f"trajectory weights are computed by advancing the given context."
        )
    return requested


def _trajectory_contexts(system, context, times, simulator_options) -> List[Any]:
    """Contexts at each snapshot time, by advancing ``context`` in order.

    A ``SimulationResults`` records signals, not the full state of every
    stateful leaf, so the operating points have to be re-derived. Walking the
    snapshots in order and restarting from each one keeps the total integration
    work close to a single run of the same span.
    """
    from ..simulation import simulate

    contexts = []
    current = context
    for time in times:
        if float(time) > float(current.time):
            current = simulate(
                system,
                current,
                (float(current.time), float(time)),
                options=simulator_options,
            ).context
        contexts.append(current)
    return contexts


def _local_data(
    system, context, probe: Optional[float], scale_floor: float
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[str, Dict[str, str]]]:
    """Every leaf's local Jacobians (and optional secants) at ``context``."""
    jacobians = {}
    secants: Dict[Any, Any] = {}
    notes: Dict[str, Dict[str, str]] = {}
    for leaf in _leaf_systems(system):
        jacs = leaf_jacobians(leaf, context)
        jacobians[leaf.system_id] = jacs
        if jacs.notes:
            notes[_leaf_path(leaf)] = dict(jacs.notes)
        if probe is not None:
            secants[leaf.system_id] = secant_jacobians(
                leaf, context, probe, scale_floor
            )
    return jacobians, secants, notes


def _build_at(
    system,
    jacobians,
    *,
    tau: float,
    normalize: str,
    scale_floor: float,
    scales: Optional[Dict[str, np.ndarray]] = None,
    secants: Optional[Dict[Any, Any]] = None,
    probe: Optional[float] = None,
) -> nx.DiGraph:
    """One weighted graph from one set of local Jacobians.

    ``scales`` overrides the per-node normalizers.  Trajectory mode passes
    trajectory-wide scales so that every snapshot divides a given signal by the
    *same* number: relative weights only telescoping along a path when the
    intermediate scales agree, and a per-snapshot scale would mean a per-edge
    reduction (max over snapshots) silently combined an early snapshot's
    denominator with a late snapshot's numerator.
    """
    graph = nx.DiGraph()
    relative = normalize == "relative"
    leaves = _leaf_systems(system)

    for leaf in leaves:
        for node_id, attributes in _node_specs(leaf, jacobians[leaf.system_id]):
            graph.add_node(node_id, **attributes)

    def scale_of(node_id: str) -> np.ndarray:
        if not relative:
            return np.ones(max(int(graph.nodes[node_id]["size"]), 1))
        if scales is not None and node_id in scales:
            return scales[node_id]
        return _scale(graph.nodes[node_id]["value"], scale_floor)

    for node_id in graph:
        graph.nodes[node_id]["scale"] = scale_of(node_id)

    # --- wires: exact identity ----------------------------------------
    for input_locator, output_locator in leaf_connections(system):
        source = output_locator[0].output_ports[output_locator[1]]
        destination = input_locator[0].input_ports[input_locator[1]]
        src_id, dst_id = port_node_id(source), port_node_id(destination)
        if src_id not in graph or dst_id not in graph:
            continue
        # A connection is the identity in *value*, but not in relative terms
        # unless both endpoints normalize by the same scale — and they do not
        # when the source is a sample-and-hold output, whose node carries the
        # post-tick value while the consumer's carries the held one. Scaling the
        # identity like any other block keeps path products telescoping.
        size = graph.nodes[dst_id]["size"]
        identity = np.eye(size)
        src_scale, dst_scale = scale_of(src_id), scale_of(dst_id)
        scaled = (
            _relative(identity, src_scale, dst_scale, 1.0)
            if src_scale.size == size and dst_scale.size == size
            else identity  # a placeholder-sized endpoint; leave the wire exact
        )
        _set_edge(graph, src_id, dst_id, WIRE, identity, scaled)

    # --- block-local edges --------------------------------------------
    for leaf in leaves:
        jacs = jacobians[leaf.system_id]
        probed = secants.get(leaf.system_id) if secants else None
        feedthrough = set(_feedthrough_pairs(leaf))

        for kind, family, key, src_id, dst_id, rate_kind, block in _local_edges(
            leaf, jacs, feedthrough
        ):
            note = ""
            if probed is not None and not np.any(block):
                alternative = getattr(probed, family).get(key)
                if alternative is not None and np.any(alternative):
                    block = alternative
                    note = (
                        f"secant slope over a {probe:g} relative step: the exact "
                        f"local derivative here is zero (flat, quantized, or "
                        f"saturated at this operating point)"
                    )
            factor = tau if rate_kind == "xc" else 1.0
            _set_edge(
                graph,
                src_id,
                dst_id,
                kind,
                block,
                _relative(block, scale_of(src_id), scale_of(dst_id), factor),
                note=note,
                tau_applied=rate_kind == "xc",
            )

        _label_missing_gradients(graph, leaf, jacs, feedthrough)

    return graph


def _local_edges(leaf, jacs, feedthrough):
    """Every block-local edge one leaf contributes.

    Yields ``(edge_kind, jacobian_family, key, src_id, dst_id, rate_kind,
    block)``.  ``rate_kind`` is the destination state kind, which is what
    decides whether ``tau`` applies: only a *continuous*-state derivative needs
    a time scale to become a state increment.
    """
    for (out_index, in_index), block in jacs.d.items():
        if (in_index, out_index) not in feedthrough:
            continue  # structurally not a feedthrough path
        yield (
            FEEDTHROUGH,
            "d",
            (out_index, in_index),
            port_node_id(leaf.input_ports[in_index]),
            port_node_id(leaf.output_ports[out_index]),
            None,
            block,
        )
    for (kind, in_index), block in jacs.b.items():
        yield (
            TO_STATE,
            "b",
            (kind, in_index),
            port_node_id(leaf.input_ports[in_index]),
            state_node_id(leaf, kind),
            kind,
            block,
        )
    for (kind, out_index), block in jacs.c.items():
        yield (
            FROM_STATE,
            "c",
            (kind, out_index),
            state_node_id(leaf, kind),
            port_node_id(leaf.output_ports[out_index]),
            None,
            block,
        )
    for (src_kind, dst_kind), block in jacs.a.items():
        yield (
            STATE_TO_STATE,
            "a",
            (src_kind, dst_kind),
            state_node_id(leaf, src_kind),
            state_node_id(leaf, dst_kind),
            dst_kind,
            block,
        )


def _structural_graph(system) -> nx.DiGraph:
    """The model's declared connectivity, with no reference to any Jacobian.

    :meth:`InfluenceGraph.structural_slice` has to be the boolean
    over-approximation that the weighted slice improves on, so it cannot be
    derived from the weighted graph: an edge the Jacobian pass never produced
    (a block that would not evaluate, a state whose dtype has no derivative)
    would then be missing from the "over-approximation" too, and the two would
    agree for the wrong reason.
    """
    graph = nx.DiGraph()
    for leaf in _leaf_systems(system):
        inputs = [port_node_id(port) for port in leaf.input_ports]
        outputs = [port_node_id(port) for port in leaf.output_ports]
        for node_id in inputs + outputs:
            graph.add_node(node_id, block=_leaf_path(leaf))

        for in_index, out_index in _feedthrough_pairs(leaf):
            graph.add_edge(inputs[in_index], outputs[out_index], kind=FEEDTHROUGH)

        # Conservative on purpose: any state the block holds is assumed to be
        # driven by every input and read by every output.
        states = []
        if leaf.ode_callback is not None:
            states.append(state_node_id(leaf, "xc"))
        if getattr(leaf, "_state_update_events", None):
            states.append(state_node_id(leaf, "xd"))
        for state_id in states:
            graph.add_node(state_id, block=_leaf_path(leaf))
            for node_id in inputs:
                graph.add_edge(node_id, state_id, kind=TO_STATE)
            for node_id in outputs:
                graph.add_edge(state_id, node_id, kind=FROM_STATE)

    for input_locator, output_locator in leaf_connections(system):
        source = output_locator[0].output_ports[output_locator[1]]
        destination = input_locator[0].input_ports[input_locator[1]]
        src_id, dst_id = port_node_id(source), port_node_id(destination)
        if src_id in graph and dst_id in graph:
            graph.add_edge(src_id, dst_id, kind=WIRE)

    return graph


def _feedthrough_pairs(leaf) -> List[Tuple[int, int]]:
    """``(input_index, output_index)`` pairs with structural feedthrough.

    Falls back to "assume everything feeds through" if the block cannot report
    — an over-approximation, matching what the framework itself does when
    feedthrough is undecidable.
    """
    try:
        return leaf.get_feedthrough()
    except Exception:  # noqa: BLE001
        return [
            (inp.index, out.index)
            for inp in leaf.input_ports
            for out in leaf.output_ports
        ]


def _label_missing_gradients(graph, leaf, jacs, feedthrough):
    """Emit ``local_gradient=False`` edges wherever a Jacobian was refused.

    Without this, a comparator or a quantizer would simply have no edge, and a
    slice would read as "this input does not influence that output" when the
    truth is "we could not compute how much".
    """
    for in_index, out_index in sorted(feedthrough):
        if (out_index, in_index) in jacs.d:
            continue
        note = (
            jacs.notes.get(f"in:{leaf.input_ports[in_index].name}")
            or jacs.notes.get(f"out:{leaf.output_ports[out_index].name}")
            or jacs.notes.get("block")
        )
        if note is None:
            continue
        _set_edge(
            graph,
            port_node_id(leaf.input_ports[in_index]),
            port_node_id(leaf.output_ports[out_index]),
            FEEDTHROUGH,
            None,
            None,
            local_gradient=False,
            note=note,
        )

    # A state that exists but could not be differentiated (a boolean latch, a
    # PRNG key) still connects its block's inputs to its outputs. Which inputs
    # feed it is unknowable without a derivative, so every pair is labelled —
    # the same conservative direction the framework takes when feedthrough is
    # undecidable.
    for kind in STATE_KINDS:
        note = jacs.notes.get(f"state:{kind}")
        if note is None or kind not in jacs.x0:
            continue
        state_id = state_node_id(leaf, kind)
        if state_id not in graph:
            continue
        for port in leaf.input_ports:
            _set_edge(
                graph,
                port_node_id(port),
                state_id,
                TO_STATE,
                None,
                None,
                local_gradient=False,
                note=note,
            )
        for port in leaf.output_ports:
            _set_edge(
                graph,
                state_id,
                port_node_id(port),
                FROM_STATE,
                None,
                None,
                local_gradient=False,
                note=note,
            )

    # An input that could not be differentiated (a boolean gate, an integer
    # mode) may still drive this block's states — and for a block whose outputs
    # read only its state, the feedthrough loop above emits nothing at all, so
    # without this the input would be reported as having no influence
    # whatsoever. That is exactly the silent zero this function exists to
    # prevent.
    for index, port in enumerate(leaf.input_ports):
        note = jacs.notes.get(f"in:{port.name}") or jacs.notes.get("block")
        if note is None:
            continue
        for kind in STATE_KINDS:
            if kind not in jacs.x0 or (kind, index) in jacs.b:
                continue
            state_id = state_node_id(leaf, kind)
            if state_id in graph:
                _set_edge(
                    graph,
                    port_node_id(port),
                    state_id,
                    TO_STATE,
                    None,
                    None,
                    local_gradient=False,
                    note=note,
                )


def influence_graph(
    system,
    context=None,
    *,
    at: str = "operating_point",
    results=None,
    times: Optional[Sequence[float]] = None,
    n_snapshots: int = 5,
    tau: float = 1.0,
    normalize: str = "relative",
    scale_floor: float = 1e-6,
    probe: Optional[float] = None,
    reduce: str = "max",
    simulator_options=None,
) -> InfluenceGraph:
    """Build the sensitivity-weighted influence graph of a model.

    Args:
        system: A ``Diagram`` or a single ``LeafSystem``.
        context: Root context fixing the operating point. Defaults to
            ``system.create_context()``.
        at: ``"operating_point"`` weights every edge once, at ``context``.
            ``"trajectory"`` weights at several snapshots and stores per-edge
            profiles — the honest answer when a nonlinearity means one number
            per edge cannot be right everywhere (a block saturated at the
            operating point has a zero local gradient there and a large one
            elsewhere).
        results: A ``SimulationResults`` supplying the snapshot times for
            ``at="trajectory"``. The states are re-derived by advancing
            ``context``, because recorded signals do not pin down every
            stateful leaf — which costs one ``simulate`` call per snapshot.
            Budget for that on a large model: ``simulate``'s fixed setup cost
            scales with block count and dominates the integration itself (a
            1 µs span costs the same as a 4 s one), so ``n_snapshots=6`` on a
            2500-block model is minutes rather than seconds. Building at a
            single operating point is linear in block count and stays in
            seconds at that size.
        times: Explicit snapshot times, used instead of ``results``.
        n_snapshots: How many times to take from ``results.time``.
        tau: Seconds of integration represented by a continuous-state-rate
            edge; only affects edges into ``ẋc``. Set it from the *fastest*
            state on the paths you care about — every integrator on a path
            contributes a factor of ``tau``, so a value taken from the slow
            dynamics of a stiff model inflates multi-integrator path products
            (see the module docstring).
        normalize: ``"relative"`` (default, dimensionless elasticities) or
            ``"none"`` (raw partial derivatives in model units).
        scale_floor: Floor on a signal's operating-point magnitude when
            normalizing, so a signal that happens to sit at zero does not
            produce an infinite elasticity. Nodes at the floor are visible via
            their ``value`` attribute.
        probe: When set to a relative step size (``0.05`` = 5% of each signal's
            magnitude), every edge whose exact derivative is zero is re-checked
            with a central-difference secant, and the secant is used instead when
            it is non-zero. This is the cross-check for the one thing an exact
            local derivative gets wrong: a quantizer between steps, a saturation
            at its rail, or a dead zone inside the zone is *locally* flat while
            still transmitting information, and would otherwise be reported dead.
            Costs two extra block evaluations per signal component; ``None``
            (default) skips it.
        reduce: How a trajectory profile collapses to the scalar weight used by
            queries: ``"max"`` (default, conservative — never hides an
            influence that appears at some point), ``"mean"``, or ``"final"``.
        simulator_options: ``SimulatorOptions`` for the trajectory-mode
            re-integration.

    Returns:
        An :class:`InfluenceGraph`.

    Example:
        >>> import jaxonomy
        >>> from jaxonomy.library import Constant, Gain, Integrator
        >>> from jaxonomy.analysis import influence_graph
        >>> builder = jaxonomy.DiagramBuilder()
        >>> source = builder.add(Constant(1.0, name="src"))
        >>> gain = builder.add(Gain(3.0, name="gain"))
        >>> plant = builder.add(Integrator(1.0, name="plant"))
        >>> builder.connect(source.output_ports[0], gain.input_ports[0])
        >>> builder.connect(gain.output_ports[0], plant.input_ports[0])
        >>> diagram = builder.build(name="root")
        >>> graph = influence_graph(diagram)
        >>> graph.slice("plant:xc", threshold=0.01).blocks
        ['gain', 'plant', 'src']
    """
    if at not in ("operating_point", "trajectory"):
        raise ValueError(
            f"at must be 'operating_point' or 'trajectory', got {at!r}"
        )
    if normalize not in ("relative", "none"):
        raise ValueError(
            f"normalize must be 'relative' or 'none', got {normalize!r}"
        )
    if reduce not in ("max", "mean", "final"):
        raise ValueError(f"reduce must be 'max', 'mean' or 'final', got {reduce!r}")
    if tau <= 0:
        raise ValueError(f"tau must be positive (seconds), got {tau}")

    if context is None:
        context = system.create_context()

    if at == "operating_point":
        jacobians, secants, notes = _local_data(system, context, probe, scale_floor)
        graph = _build_at(
            system,
            jacobians,
            tau=tau,
            normalize=normalize,
            scale_floor=scale_floor,
            secants=secants,
            probe=probe,
        )
        return InfluenceGraph(
            system=system,
            graph=graph,
            tau=tau,
            normalize=normalize,
            scale_floor=scale_floor,
            at=at,
            times=None,
            reduce=reduce,
            block_notes=notes,
            structure=_structural_graph(system),
        )

    snapshot_times = _snapshot_times(context, results, times, n_snapshots)
    contexts = _trajectory_contexts(system, context, snapshot_times, simulator_options)

    per_snapshot = []
    merged_notes: Dict[str, Dict[str, str]] = {}
    for snapshot_context in contexts:
        jacobians, secants, notes = _local_data(
            system, snapshot_context, probe, scale_floor
        )
        per_snapshot.append((jacobians, secants))
        for block, block_notes in notes.items():
            merged_notes.setdefault(block, {}).update(block_notes)

    scales = _trajectory_scales(
        system, [jacobians for jacobians, _ in per_snapshot], normalize, scale_floor
    )
    graphs = [
        _build_at(
            system,
            jacobians,
            tau=tau,
            normalize=normalize,
            scale_floor=scale_floor,
            scales=scales,
            secants=secants,
            probe=probe,
        )
        for jacobians, secants in per_snapshot
    ]

    combined = _merge_profiles(graphs, reduce)
    return InfluenceGraph(
        system=system,
        graph=combined,
        tau=tau,
        normalize=normalize,
        scale_floor=scale_floor,
        at=at,
        times=snapshot_times,
        reduce=reduce,
        block_notes=merged_notes,
        structure=_structural_graph(system),
    )


def _trajectory_scales(
    system, per_snapshot: List[Dict[Any, Any]], normalize: str, scale_floor: float
) -> Optional[Dict[str, np.ndarray]]:
    """Per-node normalizer shared by every snapshot: the largest magnitude seen.

    Using the trajectory-wide magnitude rather than each snapshot's own also
    removes the pathology where a signal that transiently crosses zero (a
    tracking error settling out) acquires an enormous elasticity purely because
    its denominator collapsed.
    """
    if normalize != "relative":
        return None
    peaks: Dict[str, np.ndarray] = {}
    for leaf in _leaf_systems(system):
        for jacobians in per_snapshot:
            for node_id, attributes in _node_specs(leaf, jacobians[leaf.system_id]):
                magnitude = np.abs(attributes["value"])
                previous = peaks.get(node_id)
                if previous is None or previous.size != magnitude.size:
                    peaks[node_id] = magnitude
                else:
                    peaks[node_id] = np.maximum(previous, magnitude)
    return {node: np.maximum(peak, scale_floor) for node, peak in peaks.items()}


def _merge_profiles(graphs: List[nx.DiGraph], reduce: str) -> nx.DiGraph:
    """Collapse per-snapshot graphs into one carrying per-edge profiles.

    The union of nodes and edges is taken, not the intersection: an edge that
    only conducts during part of the trajectory has to survive into the merged
    graph, or the trajectory mode would report *less* than the operating-point
    mode it is meant to improve on.
    """
    merged = nx.DiGraph()
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **data)
            else:
                # Report the largest magnitude the signal reached, matching the
                # scale the weights were normalized by.
                merged.nodes[node]["value"] = np.maximum(
                    np.abs(merged.nodes[node]["value"]), np.abs(data["value"])
                )

    edges = {}
    for index, graph in enumerate(graphs):
        for src, dst, data in graph.edges(data=True):
            edges.setdefault((src, dst), {})[index] = data

    for (src, dst), per_snapshot in edges.items():
        profile = np.full(len(graphs), np.nan)
        signed = np.full(len(graphs), np.nan)
        for index, data in per_snapshot.items():
            if data["local_gradient"]:
                profile[index] = data["magnitude"]
                signed[index] = data["weight"]
        # An edge that is unknown at *any* snapshot has an unknown reduction:
        # the max over the trajectory cannot be established from the snapshots
        # that happened to be differentiable, and reporting the known ones as
        # the answer would let a "dead edge" verdict rest on the snapshots we
        # could measure. `profile` still records what was measured where.
        known = [d for d in per_snapshot.values() if d["local_gradient"]]
        fully_known = len(known) == len(graphs)
        if fully_known:
            # `representative` is the snapshot whose exact Jacobian is reported
            # alongside the reduced scalar, so the two always describe the same
            # instant.  For "mean" the scalar is a genuine average and its
            # magnitude can therefore differ from |weight| of any one snapshot.
            if reduce == "max":
                representative = int(np.nanargmax(profile))
                magnitude = float(profile[representative])
                weight = float(signed[representative])
            elif reduce == "final":
                # The last snapshot, not the last measurable one — otherwise
                # `reduce="final"` would silently report an earlier instant.
                representative = len(graphs) - 1
                magnitude = float(profile[representative])
                weight = float(signed[representative])
            else:
                representative = max(i for i in per_snapshot if not np.isnan(profile[i]))
                magnitude = float(np.nanmean(profile))
                weight = float(np.nanmean(signed))
            template = per_snapshot[representative]
            note = template["note"]
        else:
            template = per_snapshot[max(per_snapshot)]
            magnitude = math.nan
            weight = math.nan
            missing = len(graphs) - len(known)
            note = template["note"] or (
                f"no local gradient at {missing} of {len(graphs)} snapshots"
            )
        merged.add_edge(
            src,
            dst,
            kind=template["kind"],
            jacobian=template["jacobian"],
            relative=template["relative"],
            weight=weight,
            magnitude=magnitude,
            local_gradient=fully_known,
            note=note,
            tau_applied=template["tau_applied"],
            profile=profile,
            profile_signed=signed,
        )
    return merged
