# SPDX-License-Identifier: MIT

"""T-104 phase 2 — unit propagation through math blocks.

Phase 1 shipped connect-time consistency checking (DiagramBuilder.connect
calls assert_unit_compatible on source/destination port units). Phase 2
adds the *rules* that compute what units a math block's output should
carry given its input units and parameters — and a single walker
function that, after diagram build, propagates units forward through
the diagram and stamps each output port that doesn't already declare
one.

Design:

* The per-block algebra lives as pure functions in this module — no
  framework state, easy to unit-test in isolation. Each rule has a
  documented signature.
* :func:`propagate_diagram_units` is the optional one-line entry point
  callers invoke after :meth:`DiagramBuilder.build`. It walks the
  diagram in input-source-first order, gathers input units from
  upstream output ports, applies each block's registered rule, and
  stamps the result on the block's output port — but only when the
  block didn't already declare a unit explicitly (default-off
  byte-equivalence with the phase-1 flow).
* New block types register a rule via :func:`register_unit_rule`. This
  module pre-registers rules for the math-block set called out in the
  T-104 phase-2 phasing: ``Adder``, ``Gain``, ``Product``,
  ``Reciprocal``, ``Integrator``, ``Derivative`` (plus the obvious
  passthroughs).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .units import (
    Unit,
    UnitMismatchError,
    assert_unit_compatible,
    resolve_unit,
)


__all__ = [
    "units_for_adder",
    "units_for_gain",
    "units_for_product",
    "units_for_reciprocal",
    "units_for_integrator",
    "units_for_derivative",
    "units_for_passthrough",
    "register_unit_rule",
    "get_unit_rule",
    "propagate_diagram_units",
    "UNIT_OF_TIME",
]


# Canonical unit-of-time used by integrator / derivative rules.  Lives
# here (not in units.py) because units.py owns the algebra and the
# canonical instances; this module owns the *application* of those
# instances to block-level propagation rules.
UNIT_OF_TIME = Unit(dims=(0, 0, 1, 0, 0, 0, 0), name="s")


# ---------------------------------------------------------------------------
# Per-block algebra (pure functions).
# ---------------------------------------------------------------------------


def units_for_passthrough(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Identity rule: a one-input passthrough block (Abs, Slice,
    IOPort, ZeroOrderHold, etc.) carries its input unit through
    unchanged. Returns ``None`` when no input unit is set, preserving
    the default-off byte-equivalence policy.
    """
    del params
    if not input_units:
        return None
    return input_units[0]


def units_for_adder(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Sum / difference: all inputs must share the same unit; the
    output carries that same unit.

    Sign flips (``Adder(operators="+-+")``) preserve units — subtracting
    metres from metres is still metres. Raises :class:`UnitMismatchError`
    on a unit-incompatible input set so the user discovers the bug at
    propagation time rather than as a confusing runtime error.

    Returns ``None`` when no input declares a unit (default-off
    byte-equivalence).
    """
    del params
    # Drop unset inputs; if none survive, no propagation needed.
    typed = [u for u in input_units if u is not None]
    if not typed:
        return None
    first = typed[0]
    for u in typed[1:]:
        # Raises UnitMismatchError on incompatibility; the caller
        # (propagate_diagram_units) lets it propagate up so users see
        # the conflict immediately.
        assert_unit_compatible(first, u, src_label="Adder input",
                               dst_label="Adder input")
    return first


def units_for_gain(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Gain: ``y = gain * u`` so ``units(y) = units(gain) * units(u)``.

    The gain unit comes from ``params['gain_units']`` when set; absent
    that, the gain is treated as dimensionless and the output unit
    equals the input unit (the common case — most Gain blocks are
    used for pure-numeric scaling).
    """
    if not input_units or input_units[0] is None:
        return None
    gain_unit = (params or {}).get("gain_units")
    if gain_unit is None:
        return input_units[0]
    return gain_unit * input_units[0]


def units_for_product(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Product: ``y = u1 * u2 * ... * un`` so ``units(y) = ∏ units(ui)``.

    Inputs with no declared unit are treated as dimensionless under
    the standard "unit-less is wildcard" convention. Returns ``None``
    only when *every* input is unit-less (preserving default-off
    byte-equivalence).
    """
    del params
    typed = [u for u in input_units if u is not None]
    if not typed:
        return None
    out = typed[0]
    for u in typed[1:]:
        out = out * u
    return out


def units_for_reciprocal(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Reciprocal: ``y = 1/u`` so ``units(y) = 1 / units(u)``.

    Implemented as ``dimensionless / units(u)`` via the existing Unit
    algebra; returns ``None`` when the input has no declared unit.
    """
    del params
    if not input_units or input_units[0] is None:
        return None
    # ``Unit() / u`` gives ``1/u`` since Unit() is dimensionless.
    return Unit() / input_units[0]


def units_for_integrator(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Integrator: ``y(t) = ∫ u(τ) dτ`` so ``units(y) = units(u) · seconds``.

    The first (and only) input is the integrand. Returns ``None``
    when the input has no declared unit.
    """
    del params
    if not input_units or input_units[0] is None:
        return None
    return input_units[0] * UNIT_OF_TIME


def units_for_derivative(
    input_units: Sequence[Optional[Unit]],
    params: Optional[dict] = None,
) -> Optional[Unit]:
    """Derivative (continuous): ``y(t) = du/dt`` so
    ``units(y) = units(u) / seconds``.
    """
    del params
    if not input_units or input_units[0] is None:
        return None
    return input_units[0] / UNIT_OF_TIME


# ---------------------------------------------------------------------------
# Rule registry — block-class-name → rule callable.
# ---------------------------------------------------------------------------


# Class-name lookup table.  Keyed by ``cls.__name__`` rather than by
# the class object itself so the registry can be populated before the
# block modules are imported (the registry lives in the framework
# layer; block classes live in the library layer, which depends on
# framework).
_RULES: dict[str, Callable[..., Optional[Unit]]] = {}


def register_unit_rule(
    block_class_name: str,
    rule: Callable[[Sequence[Optional[Unit]], Optional[dict]], Optional[Unit]],
) -> None:
    """Register ``rule`` as the unit-propagation function for
    ``block_class_name`` (matched by ``cls.__name__``).

    Last write wins; downstream libraries can override the built-in
    rules for their own block subclasses.
    """
    _RULES[block_class_name] = rule


def get_unit_rule(
    block_class_name: str,
) -> Optional[Callable[..., Optional[Unit]]]:
    """Return the registered rule for ``block_class_name`` or ``None``
    when no rule has been registered (the walker then leaves the
    block's output unit untouched)."""
    return _RULES.get(block_class_name)


# Pre-register the math-block set called out in the T-104 phase-2
# phasing list. Block classes look these up by their own __name__ at
# propagation time, so registration here doesn't pull in the
# (heavier) block modules at framework import.
register_unit_rule("Adder", units_for_adder)
register_unit_rule("Gain", units_for_gain)
register_unit_rule("Product", units_for_product)
register_unit_rule("Reciprocal", units_for_reciprocal)
register_unit_rule("Integrator", units_for_integrator)
register_unit_rule("Derivative", units_for_derivative)
register_unit_rule("DerivativeDiscrete", units_for_derivative)
register_unit_rule("Abs", units_for_passthrough)
register_unit_rule("Slice", units_for_passthrough)
register_unit_rule("IOPort", units_for_passthrough)
register_unit_rule("ZeroOrderHold", units_for_passthrough)
register_unit_rule("UnitDelay", units_for_passthrough)
register_unit_rule("TransportDelay", units_for_passthrough)
register_unit_rule("Offset", units_for_passthrough)
register_unit_rule("Saturate", units_for_passthrough)
register_unit_rule("DeadZone", units_for_passthrough)


# ---------------------------------------------------------------------------
# Diagram walker.
# ---------------------------------------------------------------------------


def propagate_diagram_units(diagram, *, overwrite: bool = False) -> int:
    """Walk ``diagram`` and propagate units forward through registered
    block-type rules.

    For each leaf block that has a registered rule
    (:func:`get_unit_rule`), gather the units carried by the source
    output ports it consumes, run the rule, and stamp the result on
    the block's output port(s) when the block didn't already declare a
    unit. Multiple passes are performed until a fixed point is reached
    (a block whose inputs depend on another block's not-yet-propagated
    output stabilises after the upstream pass).

    Args:
        diagram: A built :class:`Diagram` (typically the return value
            of :meth:`DiagramBuilder.build`). A non-Diagram
            ``SystemBase`` is a no-op.
        overwrite: When ``False`` (default) the walker only stamps
            output ports whose ``units`` attribute is ``None`` —
            user-supplied unit annotations win. When ``True`` the
            walker overrides any existing annotation, useful for
            re-running propagation after a parameter swap.

    Returns:
        The number of output ports the walker stamped (zero if
        propagation reached a fixed point on the first pass without
        new assignments, e.g. because every input is already unit-less
        or every block lacks a registered rule).

    Raises:
        UnitMismatchError: When a rule (e.g. :func:`units_for_adder`)
            rejects an incompatible input set.
    """
    # Lazy import to avoid the framework→library cycle.
    from .diagram import Diagram

    if not isinstance(diagram, Diagram):
        return 0

    # Conn lookup: input port → driving output port.  Locators in
    # Diagram.connection_map are ``(system, port_index)`` pairs (the
    # full SystemBase object, not just its id).
    def _source_for(input_port):
        """Return the output Port driving ``input_port``, or None if
        the input is unconnected (e.g. an exported diagram input)."""
        sys = input_port.system
        parent = _find_parent_diagram(diagram, sys)
        if parent is None:
            return None
        conn_map = getattr(parent, "connection_map", None)
        if conn_map is None:
            return None
        loc_in = (sys, input_port.index)
        out_loc = conn_map.get(loc_in)
        if out_loc is None:
            return None
        src_sys, src_idx = out_loc
        if src_idx >= len(src_sys.output_ports):
            return None
        return src_sys.output_ports[src_idx]

    stamped_count = 0
    # Fixed-point iteration. Bounded by the diagram size so a
    # pathological case can't loop forever; in practice 2-3 passes
    # suffice for any reasonable signal flow.
    max_passes = max(8, 4 * sum(1 for _ in _iter_leaves(diagram)))
    for _ in range(max_passes):
        changed = False
        for leaf in _iter_leaves(diagram):
            rule = get_unit_rule(type(leaf).__name__)
            if rule is None:
                continue
            input_units: list[Optional[Unit]] = []
            for ip in leaf.input_ports:
                src = _source_for(ip)
                input_units.append(getattr(src, "units", None) if src is not None else None)
            params = _block_params_for_units(leaf)
            try:
                new_unit = rule(input_units, params)
            except UnitMismatchError:
                raise
            if new_unit is None:
                continue
            for op in leaf.output_ports:
                existing = getattr(op, "units", None)
                if existing is not None and not overwrite:
                    continue
                if existing == new_unit and existing is not None:
                    continue
                op.units = new_unit
                stamped_count += 1
                changed = True
        if not changed:
            break
    return stamped_count


def _iter_leaves(diagram):
    """Yield every leaf in ``diagram`` (recursively flattening nested
    sub-diagrams). Order follows the natural tree-traversal order."""
    # Lazy import.
    from .diagram import Diagram

    for node in diagram.nodes:
        if isinstance(node, Diagram):
            yield from _iter_leaves(node)
        else:
            yield node


def _find_parent_diagram(root, system):
    """Return the immediate parent Diagram of ``system`` within
    ``root``, or ``None`` if ``system`` is not in the tree.

    Walks the diagram tree depth-first. O(N) per call; called once per
    input port so total cost is O(N · in-degree), fine for diagrams
    of realistic size.
    """
    from .diagram import Diagram

    if not isinstance(root, Diagram):
        return None
    for node in root.nodes:
        if node is system:
            return root
        if isinstance(node, Diagram):
            found = _find_parent_diagram(node, system)
            if found is not None:
                return found
    return None


def _block_params_for_units(block) -> dict:
    """Return the subset of a block's instance attributes that the unit
    rules consult. Conservative: looks for documented keys
    (``gain_units`` on Gain, etc.) and returns ``{}`` when none are
    present.
    """
    out: dict = {}
    for key in ("gain_units",):
        val = getattr(block, key, None)
        if val is not None:
            out[key] = val
    return out
