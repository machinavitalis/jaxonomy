# SPDX-License-Identifier: MIT

"""Local Jacobians of a single ``LeafSystem`` at an operating point.

The influence graph needs, for one leaf block and one root context, the four
local Jacobian blocks an engineer would call A/B/C/D *for that block alone*:

    ∂yᵢ/∂uⱼ   direct feedthrough, input j → output i
    ∂yᵢ/∂x    state → output i
    ∂ẋ/∂uⱼ    input j → state derivative
    ∂ẋ/∂x     state → state derivative

with the discrete-time counterparts (``xd⁺`` from the block's periodic update
callback) computed the same way, so a discrete filter or a ``PIDDiscrete`` is
weighted rather than dropped.

This is deliberately *not* :func:`jaxonomy.library.linear_system.linearize`:
that function linearizes a whole system through a single chosen input/output
port pair and warns when the operating point isn't an equilibrium.  Here every
port pair of every leaf is wanted, no equilibrium is implied (a block mid-
trajectory is the normal case), and a block whose callback cannot be
differentiated must degrade to a labelled "no local gradient" edge instead of
raising.

Everything is evaluated with the leaf's inputs *fixed* to their operating-point
values, so a Jacobian is genuinely block-local: upstream blocks contribute
nothing, and the graph edges carry the composition instead.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpy as np
from jax.flatten_util import ravel_pytree


__all__ = ["LeafJacobians", "leaf_jacobians", "secant_jacobians", "STATE_KINDS"]


# Ordered so reports and graph construction are deterministic.
STATE_KINDS = ("xc", "xd")


@dataclass
class LeafJacobians:
    """Local Jacobian blocks for one leaf at one operating point.

    Attributes:
        leaf: The ``LeafSystem`` these Jacobians describe.
        u0: Operating-point value of each input port, in port order.
        y0: Operating-point value of each output port, in port order.
        x0: Operating-point value of each state kind present, keyed by
            ``"xc"`` / ``"xd"``.
        d: ``{(out_i, in_j): ndarray(m_i, n_j)}`` — direct feedthrough.
        c: ``{(kind, out_i): ndarray(m_i, n_x)}`` — state → output.
        b: ``{(kind, in_j): ndarray(n_x, n_j)}`` — input → state rate/update.
        a: ``{(src_kind, dst_kind): ndarray(n_dst, n_src)}`` — state → state
            rate/update, including the cross terms (an ODE reading discrete
            state, a periodic update reading continuous state).
        notes: ``{subject: reason}`` for every quantity that could *not* be
            differentiated, e.g. ``{"out:mode": "non-inexact dtype int32"}``.
            Callers turn these into ``local_gradient=None`` edge labels rather
            than silently reporting a zero.
    """

    leaf: Any
    u0: List[Any]
    y0: List[Any]
    x0: Dict[str, Any]
    d: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    c: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    b: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    a: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"LeafJacobians({self.leaf.name}, n_in={len(self.u0)}, "
            f"n_out={len(self.y0)}, states={sorted(self.x0)}, "
            f"notes={len(self.notes)})"
        )


# ---------------------------------------------------------------------------
# Isolated leaf evaluation
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fixed_inputs(leaf, values):
    """Fix every input port of ``leaf`` to ``values`` for the duration.

    Restores whatever each port had before — including a value the *user* had
    pinned with ``fix_value`` on a disconnected port, which the plain
    ``InputPort.fixed`` context manager would drop on exit.
    """
    saved = [(port, port.is_fixed, port.default_value) for port in leaf.input_ports]
    try:
        for port, value in zip(leaf.input_ports, values):
            port.fix_value(value)
        yield
    finally:
        for port, was_fixed, old_value in reversed(saved):
            port.unfix()
            if was_fixed:
                port.fix_value(old_value)


def _discrete_update_event(leaf):
    """The leaf's periodic discrete-state update, if it has exactly one.

    A ``LeafSystem`` holds a single discrete-state component, so multiple
    periodic updates all overwrite the same value and there is no single
    Jacobian to report; those leaves get a note instead.
    """
    events = [e for e in leaf.state_update_events.events if e is not None]
    if len(events) == 1:
        return events[0]
    return None


def is_sampled_output(port) -> bool:
    """True for a sample-and-hold output port (declared with a ``period``).

    Such a port's own callback just reads ``state.cache[cache_index]``, so
    differentiating it gives zero for every input and every state: within a
    step, the held value depends on nothing.  The block's actual transfer lives
    in the port's periodic update event.
    """
    return getattr(port, "cache_index", None) is not None and port.event is not None


def _output_value(leaf, port, context):
    """The value ``port`` produces, resolving sample-and-hold to its next tick.

    For a sample-and-hold port this is the value the port will hold after its
    next update — the quantity whose sensitivity an engineer means when they ask
    what drives a discrete block's output. Reporting the currently-held value
    instead would make every discrete block look like a dead end.
    """
    if is_sampled_output(port):
        return port.event.callback(context).cache[port.cache_index]
    return port.eval(context)


def _eval_leaf(leaf, root_context, us, xc, xd):
    """Evaluate ``(outputs, ẋc, xd⁺)`` for ``leaf`` alone.

    ``xc`` / ``xd`` of ``None`` mean "leave the context's value in place";
    passing them explicitly is how the state Jacobians get their perturbation
    in.
    """
    leaf_context = root_context[leaf.system_id]
    if xc is not None:
        leaf_context = leaf_context.with_continuous_state(xc)
    if xd is not None:
        leaf_context = leaf_context.with_discrete_state(xd)
    context = root_context.with_subcontext(leaf.system_id, leaf_context)

    with _fixed_inputs(leaf, us):
        outputs = [_output_value(leaf, port, context) for port in leaf.output_ports]
        xcdot = (
            leaf.eval_time_derivatives(context)
            if leaf.ode_callback is not None
            else None
        )
        event = _discrete_update_event(leaf)
        xd_plus = (
            event.callback(context).discrete_state
            if event is not None and leaf_context.has_discrete_state
            else None
        )
    return outputs, xcdot, xd_plus


# ---------------------------------------------------------------------------
# Differentiability screening
# ---------------------------------------------------------------------------


def _differentiable(value) -> Optional[str]:
    """Return None if ``value`` can carry a JVP tangent, else why it can't.

    Integer / boolean signals (a state-machine ``mode``, a comparator output,
    a PRNG key stored in discrete state) have no meaningful derivative.  JAX
    would hand back a ``float0`` tangent, and ``ravel_pytree`` over a mixed-
    dtype state would silently promote the key to float — both produce numbers
    that look like sensitivities and aren't.
    """
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        return "empty"
    for entry in leaves:
        dtype = jax.numpy.asarray(entry).dtype
        if not np.issubdtype(dtype, np.inexact):
            return f"non-inexact dtype {dtype}"
    return None


def _ravel(value):
    """Flat vector view of a value that may be a pytree.

    A leaf's continuous state, its derivative, and its outputs are all allowed
    to be NamedTuples (``BatteryCell.BatteryStateType``, a Kalman filter's
    ``DiscreteStateType``), so the differentiated functions return raveled
    vectors and the Jacobians come back as plain matrices.
    """
    flat, _ = ravel_pytree(value)
    return flat


def _size(value) -> int:
    return int(_ravel(value).size)


def _as_matrix(jac, n_out: int, n_in: int) -> np.ndarray:
    return np.asarray(jax.numpy.reshape(jac, (n_out, n_in)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def leaf_jacobians(leaf, root_context) -> LeafJacobians:
    """Compute every local Jacobian block of ``leaf`` at ``root_context``.

    Args:
        leaf: A ``LeafSystem`` belonging to the system ``root_context`` was
            created from.
        root_context: Root context supplying the operating point — time,
            parameters, this leaf's state, and (via upstream evaluation) the
            values arriving on its input ports.

    Returns:
        A :class:`LeafJacobians`.  Blocks that could not be differentiated are
        absent from ``d`` / ``c`` / ``b`` / ``a`` and explained in ``notes``;
        this function does not raise on a non-differentiable block.
    """
    # An enabled port cache would hand back stored constants instead of
    # re-evaluating under the JVP, zeroing every gradient.  Inputs are fixed
    # below, so nothing upstream needs the cache.
    if root_context.port_cache:
        root_context = root_context.with_port_cache({})

    leaf_context = root_context[leaf.system_id]
    try:
        u0 = [port.eval(root_context) for port in leaf.input_ports]
    except Exception as exc:  # noqa: BLE001 - e.g. a dangling, unfixed input
        return LeafJacobians(
            leaf=leaf,
            u0=[],
            y0=[],
            x0={},
            notes={
                "block": (
                    f"input-port evaluation failed: {type(exc).__name__}: {exc}"
                )
            },
        )

    x0: Dict[str, Any] = {}
    if leaf_context.has_continuous_state and leaf.ode_callback is not None:
        x0["xc"] = leaf_context.continuous_state
    if leaf_context.has_discrete_state and _discrete_update_event(leaf) is not None:
        x0["xd"] = leaf_context.discrete_state

    jacs = LeafJacobians(leaf=leaf, u0=u0, y0=[], x0=x0)

    if leaf_context.has_discrete_state and "xd" not in x0:
        jacs.notes["state:xd"] = (
            "discrete state with no single periodic update callback"
        )

    def evaluate(us, xc, xd):
        return _eval_leaf(leaf, root_context, us, xc, xd)

    # Baseline evaluation, and the reference for output sizes / dtypes.
    try:
        y0, xcdot0, xdp0 = evaluate(u0, None, None)
    except Exception as exc:  # noqa: BLE001 - a block we cannot evaluate at all
        jacs.notes["block"] = f"evaluation failed: {type(exc).__name__}: {exc}"
        return jacs
    jacs.y0 = y0

    # Which outputs / states are differentiable at all.
    out_ok: Dict[int, int] = {}
    for i, y in enumerate(y0):
        reason = _differentiable(y)
        if reason is None:
            out_ok[i] = _size(y)
        else:
            jacs.notes[f"out:{leaf.output_ports[i].name}"] = reason

    in_ok: Dict[int, int] = {}
    for j, u in enumerate(u0):
        reason = _differentiable(u)
        if reason is None:
            in_ok[j] = _size(u)
        else:
            jacs.notes[f"in:{leaf.input_ports[j].name}"] = reason

    rates = {}  # state kind -> (baseline rate/update value, its size)
    if "xc" in x0 and xcdot0 is not None:
        rates["xc"] = (xcdot0, _size(xcdot0))
    if "xd" in x0 and xdp0 is not None:
        rates["xd"] = (xdp0, _size(xdp0))

    state_ok: Dict[str, int] = {}
    for kind, value in x0.items():
        reason = _differentiable(value)
        if reason is None and kind in rates:
            state_ok[kind] = _size(value)
        elif reason is not None:
            jacs.notes[f"state:{kind}"] = reason

    # --- ∂(·)/∂uⱼ : one forward-mode trace per input port -------------------
    for j, n_j in in_ok.items():
        u_flat, unravel = ravel_pytree(u0[j])

        def wrt_input(u_vec, j=j, unravel=unravel):
            us = list(u0)
            us[j] = unravel(u_vec)
            outputs, xcdot, xd_plus = evaluate(us, None, None)
            return (
                [_ravel(outputs[i]) for i in sorted(out_ok)],
                _ravel(xcdot) if "xc" in rates else None,
                _ravel(xd_plus) if "xd" in rates else None,
            )

        try:
            jac_out, jac_xcdot, jac_xdp = jax.jacfwd(wrt_input)(u_flat)
        except Exception as exc:  # noqa: BLE001
            jacs.notes[f"in:{leaf.input_ports[j].name}"] = (
                f"not differentiable: {type(exc).__name__}: {exc}"
            )
            continue

        for slot, i in enumerate(sorted(out_ok)):
            jacs.d[(i, j)] = _as_matrix(jac_out[slot], out_ok[i], n_j)
        for kind, jac in (("xc", jac_xcdot), ("xd", jac_xdp)):
            if jac is not None and kind in rates:
                jacs.b[(kind, j)] = _as_matrix(jac, rates[kind][1], n_j)

    # --- ∂(·)/∂x : one forward-mode trace per state kind -------------------
    for kind, n_x in state_ok.items():
        x_flat, unravel = ravel_pytree(x0[kind])

        def wrt_state(x_vec, kind=kind, unravel=unravel):
            xc = unravel(x_vec) if kind == "xc" else None
            xd = unravel(x_vec) if kind == "xd" else None
            outputs, xcdot, xd_plus = evaluate(u0, xc, xd)
            return (
                [_ravel(outputs[i]) for i in sorted(out_ok)],
                _ravel(xcdot) if "xc" in rates else None,
                _ravel(xd_plus) if "xd" in rates else None,
            )

        try:
            jac_out, jac_xcdot, jac_xdp = jax.jacfwd(wrt_state)(x_flat)
        except Exception as exc:  # noqa: BLE001
            jacs.notes[f"state:{kind}"] = (
                f"not differentiable: {type(exc).__name__}: {exc}"
            )
            continue

        for slot, i in enumerate(sorted(out_ok)):
            jacs.c[(kind, i)] = _as_matrix(jac_out[slot], out_ok[i], n_x)
        for dst, jac in (("xc", jac_xcdot), ("xd", jac_xdp)):
            if jac is not None and dst in rates:
                jacs.a[(kind, dst)] = _as_matrix(jac, rates[dst][1], n_x)

    return jacs


def secant_jacobians(
    leaf, root_context, step_fraction: float, scale_floor: float
) -> LeafJacobians:
    """The same blocks as :func:`leaf_jacobians`, by central differences.

    Exists to catch the one failure mode an exact local derivative cannot see:
    a block that is *locally flat* but not actually dead.  A quantizer between
    its steps, a saturation at its rail, a dead zone inside the zone — all have
    a genuine zero derivative and a non-zero secant slope over a finite step.
    Reporting the derivative alone would call those connections dead.

    The step for each component is ``step_fraction · max(|value|, scale_floor)``,
    so it scales with the signal rather than assuming SI-sized quantities.
    Non-differentiable blocks are still skipped: a secant across a boolean or an
    integer signal is not a slope.
    """
    baseline = leaf_jacobians(leaf, root_context)
    if "block" in baseline.notes:
        return baseline

    if root_context.port_cache:
        root_context = root_context.with_port_cache({})

    u0 = baseline.u0
    x0 = baseline.x0
    out_indices = [
        index
        for index, value in enumerate(baseline.y0)
        if _differentiable(value) is None
    ]

    secant = LeafJacobians(
        leaf=leaf, u0=u0, y0=baseline.y0, x0=x0, notes=dict(baseline.notes)
    )
    if not out_indices and not x0:
        return secant

    def targets(us, xc, xd):
        outputs, xcdot, xd_plus = _eval_leaf(leaf, root_context, us, xc, xd)
        return (
            [_ravel(outputs[i]) for i in out_indices],
            _ravel(xcdot) if xcdot is not None and "xc" in x0 else None,
            _ravel(xd_plus) if xd_plus is not None and "xd" in x0 else None,
        )

    def sweep(flat, unravel, apply_perturbation):
        """Central-difference columns for one perturbed flat vector."""
        steps = step_fraction * np.maximum(np.abs(np.asarray(flat)), scale_floor)
        columns = []
        for index in range(int(np.asarray(flat).size)):
            delta = jax.numpy.zeros_like(flat).at[index].set(steps[index])
            plus = apply_perturbation(unravel(flat + delta))
            minus = apply_perturbation(unravel(flat - delta))
            columns.append(
                tuple(
                    None
                    if high is None
                    else (np.asarray(high) - np.asarray(low)) / (2.0 * steps[index])
                    for high, low in zip(_pack(plus), _pack(minus))
                )
            )
        return columns

    def assemble(columns, slot):
        blocks = [column[slot] for column in columns]
        if not blocks or blocks[0] is None:
            return None
        return np.stack(blocks, axis=-1)

    for j, value in enumerate(u0):
        if _differentiable(value) is not None:
            continue
        flat, unravel = ravel_pytree(value)

        def perturb(new_value, j=j):
            us = list(u0)
            us[j] = new_value
            return targets(us, None, None)

        columns = sweep(flat, unravel, perturb)
        for slot, i in enumerate(out_indices):
            block = assemble(columns, slot)
            if block is not None:
                secant.d[(i, j)] = block
        for offset, kind in enumerate(STATE_KINDS):
            block = assemble(columns, len(out_indices) + offset)
            if block is not None and kind in x0:
                secant.b[(kind, j)] = block

    for kind, value in x0.items():
        if _differentiable(value) is not None:
            continue
        flat, unravel = ravel_pytree(value)

        def perturb(new_value, kind=kind):
            return targets(
                u0,
                new_value if kind == "xc" else None,
                new_value if kind == "xd" else None,
            )

        columns = sweep(flat, unravel, perturb)
        for slot, i in enumerate(out_indices):
            block = assemble(columns, slot)
            if block is not None:
                secant.c[(kind, i)] = block
        for offset, dst in enumerate(STATE_KINDS):
            block = assemble(columns, len(out_indices) + offset)
            if block is not None and dst in x0:
                secant.a[(kind, dst)] = block

    return secant


def _pack(result):
    """Flatten a ``targets()`` result into one positional sequence.

    Order is ``[outputs..., ẋc, xd⁺]`` so :func:`secant_jacobians`'s ``slot``
    indices line up with ``STATE_KINDS`` after the outputs.
    """
    outputs, xcdot, xd_plus = result
    rates = {"xc": xcdot, "xd": xd_plus}
    return list(outputs) + [rates[kind] for kind in STATE_KINDS]
