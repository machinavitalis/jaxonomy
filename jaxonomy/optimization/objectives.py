# SPDX-License-Identifier: MIT

"""
Objective function factory helpers for system identification and optimal control.

Every time a user writes::

    sq_v  = b.add(Power(2.0, name="sq_v"))
    sq_x  = b.add(Power(2.0, name="sq_x"))
    cv    = b.add(Integrator(0.0, name="cost_v"))
    cx    = b.add(Integrator(0.0, name="cost_x"))
    obj   = b.add(Adder(2, operators="++", name="obj"))
    b.connect(v.output_ports[0], sq_v.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(x.output_ports[0], sq_x.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0], obj.input_ports[0])
    b.connect(cx.output_ports[0], obj.input_ports[1])

…they can instead write::

    obj_port = weighted_sum(b, [
        ise_objective(b, v.output_ports[0]),
        ise_objective(b, x.output_ports[0]),
    ])

Factory functions add primitive blocks directly to an existing
:class:`~jaxonomy.DiagramBuilder` and return the scalar output port that
should be evaluated in ``objective_from_context``.

Functions
---------
:func:`ise_objective`
    Integral of squared error: :math:`\\int_0^T w\\,\\|e(t)\\|^2\\,dt`
:func:`lqr_objective`
    LQR-style quadratic cost: :math:`\\int_0^T (x^\\top Q x + u^\\top R u)\\,dt`
:func:`tracking_mse`
    Dataset tracking: :math:`\\int_0^T w\\,\\|\\text{signal}(t) - y_{\\text{ref}}(t)\\|^2\\,dt`
:func:`weighted_sum`
    Weighted linear combination of objective ports.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import jax.numpy as jnp

# Import lazily inside functions to avoid circular imports at module level.

__all__ = [
    "ise_objective",
    "lqr_objective",
    "tracking_mse",
    "weighted_sum",
]


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sq_sum_port(builder, err_port, name):
    """Add Power(2) → SumOfElements blocks; return scalar port.

    Works correctly for scalar **and** vector signals:
    ``jnp.sum(scalar)`` is the identity, so no shape-detection is needed.
    """
    from jaxonomy.library import Power, SumOfElements

    sq = builder.add(Power(2.0, name=f"{name}_sq"))
    s = builder.add(SumOfElements(name=f"{name}_sum"))
    builder.connect(err_port, sq.input_ports[0])
    builder.connect(sq.output_ports[0], s.input_ports[0])
    return s.output_ports[0]


def _scale_port(builder, port, weight, name):
    """Optionally multiply *port* by a scalar *weight* via :class:`~jaxonomy.library.Gain`.

    Returns *port* unchanged when *weight* is exactly ``1.0``.
    """
    from jaxonomy.library import Gain

    w = float(weight)
    if w == 1.0:
        return port
    g = builder.add(Gain(w, name=f"{name}_gain"))
    builder.connect(port, g.input_ports[0])
    return g.output_ports[0]


def _integrate_port(builder, port, initial_cost, name):
    """Add :class:`~jaxonomy.library.Integrator`; return its output port."""
    from jaxonomy.library import Integrator

    integ = builder.add(Integrator(initial_cost, name=f"{name}_integral"))
    builder.connect(port, integ.input_ports[0])
    return integ.output_ports[0]


class _QuadraticIntegrand:
    """Single-input block that computes :math:`x^\\top W x`.

    This is an internal helper that wraps :class:`~jaxonomy.library.ReduceBlock`
    so that :func:`lqr_objective` can reuse the same calculation for both the
    state and the control cost terms without relying on :class:`~jaxonomy.library.QuadraticCost`,
    which requires two inputs.
    """

    def __new__(cls, W, name=None):  # factory: returns a ReduceBlock instance
        from jaxonomy.library import ReduceBlock

        W_arr = jnp.asarray(np.asarray(W), dtype=float)

        def _cost(inputs):
            x = inputs[0]
            return jnp.squeeze(jnp.dot(x, jnp.dot(W_arr, x)))

        return ReduceBlock(1, _cost, name=name)


# ──────────────────────────────────────────────────────────────────────────────
# Public factory functions
# ──────────────────────────────────────────────────────────────────────────────

def ise_objective(
    builder,
    signal_port,
    reference_port=None,
    weight: float = 1.0,
    initial_cost: float = 0.0,
    name: str = "ise",
):
    r"""Add blocks to compute the **Integral of Squared Error** (ISE).

    .. math::

        J = \int_0^T w \, \| \text{signal}(t) - \text{reference}(t) \|^2 \, dt

    When ``reference_port`` is ``None`` the reference is implicitly zero, so
    the objective is :math:`\int_0^T w \, \|\text{signal}(t)\|^2 \, dt`.

    The function adds the following blocks to *builder*:

    * (optional) :class:`~jaxonomy.library.Adder` computing
      ``signal − reference``
    * :class:`~jaxonomy.library.Power` ``(2.0)``
    * :class:`~jaxonomy.library.SumOfElements` (handles both scalar and
      vector signals transparently)
    * (optional) :class:`~jaxonomy.library.Gain` if ``weight ≠ 1``
    * :class:`~jaxonomy.library.Integrator` accumulating the cost

    Parameters
    ----------
    builder:
        The :class:`~jaxonomy.DiagramBuilder` to add blocks to.
    signal_port:
        Output port of the signal to penalise.
    reference_port:
        Output port of the reference signal.  ``None`` → reference is 0.
    weight:
        Scalar multiplier applied to the squared norm before integration.
        For per-component or matrix weighting use :func:`lqr_objective`.
    initial_cost:
        Initial value of the accumulating integrator (default ``0.0``).
    name:
        Name prefix for the added blocks.

    Returns
    -------
    OutputPort
        Scalar port whose value at the end of simulation equals *J*.

    Examples
    --------
    Minimise oscillation energy of a spring-mass system::

        obj = ise_objective(b, x.output_ports[0])  # ∫ x² dt
        # later:  return obj.eval(ctx)

    Multi-signal ISE with a shared reference of zero::

        cost_x = ise_objective(b, x.output_ports[0], name="ise_x")
        cost_v = ise_objective(b, v.output_ports[0], name="ise_v")
        total  = weighted_sum(b, [cost_x, cost_v], weights=[1.0, 0.5])
    """
    from jaxonomy.library import Adder

    # ── error ──────────────────────────────────────────────────────────────
    if reference_port is not None:
        err = builder.add(Adder(2, operators="+-", name=f"{name}_err"))
        builder.connect(signal_port, err.input_ports[0])
        builder.connect(reference_port, err.input_ports[1])
        err_port = err.output_ports[0]
    else:
        err_port = signal_port

    # ── ‖e‖² (scalar) ──────────────────────────────────────────────────────
    sq_port = _sq_sum_port(builder, err_port, name)

    # ── optional scalar weight ──────────────────────────────────────────────
    weighted_port = _scale_port(builder, sq_port, weight, name)

    # ── integrate ───────────────────────────────────────────────────────────
    return _integrate_port(builder, weighted_port, initial_cost, name)


def lqr_objective(
    builder,
    state_port,
    Q,
    control_port=None,
    R=None,
    initial_cost: float = 0.0,
    name: str = "lqr",
):
    r"""Add blocks to compute an **LQR-style quadratic cost**.

    .. math::

        J = \int_0^T \bigl( x(t)^\top Q\, x(t) \;+\; u(t)^\top R\, u(t) \bigr)\, dt

    When ``control_port`` or ``R`` is ``None`` only the state cost
    :math:`\int x^\top Q x\, dt` is computed.

    The function adds a single-input :class:`~jaxonomy.library.ReduceBlock`
    for :math:`x^\top Q x` (and optionally one for :math:`u^\top R u`), an
    optional :class:`~jaxonomy.library.Adder`, and an
    :class:`~jaxonomy.library.Integrator`.

    Parameters
    ----------
    builder:
        The :class:`~jaxonomy.DiagramBuilder` to add blocks to.
    state_port:
        Output port of the state vector :math:`x`.
    Q:
        Positive semi-definite state weight matrix, shape ``(nx, nx)``.
    control_port:
        Output port of the control vector :math:`u`.  ``None`` → no control
        penalty.
    R:
        Positive definite control weight matrix, shape ``(nu, nu)``.
        Required when *control_port* is provided.
    initial_cost:
        Initial value of the accumulating integrator (default ``0.0``).
    name:
        Name prefix for the added blocks.

    Returns
    -------
    OutputPort
        Scalar port whose value at the end of simulation equals *J*.

    Examples
    --------
    Pendulum regulation::

        # ∫ θ²·Q[0,0] + ω²·Q[1,1] dt  (diagonal Q)
        Q = jnp.diag(jnp.array([10.0, 1.0]))
        R = jnp.array([[0.1]])
        cost = lqr_objective(b, x.output_ports[0], Q, u.output_ports[0], R)
    """
    from jaxonomy.library import Adder

    Q = np.asarray(Q, dtype=float)

    # ── state cost: x^T Q x ────────────────────────────────────────────────
    x_cost_block = builder.add(_QuadraticIntegrand(Q, name=f"{name}_Qcost"))
    builder.connect(state_port, x_cost_block.input_ports[0])
    cost_port = x_cost_block.output_ports[0]

    # ── optional control cost: u^T R u ─────────────────────────────────────
    if control_port is not None and R is not None:
        R = np.asarray(R, dtype=float)
        u_cost_block = builder.add(_QuadraticIntegrand(R, name=f"{name}_Rcost"))
        builder.connect(control_port, u_cost_block.input_ports[0])

        total = builder.add(Adder(2, operators="++", name=f"{name}_total"))
        builder.connect(cost_port, total.input_ports[0])
        builder.connect(u_cost_block.output_ports[0], total.input_ports[1])
        cost_port = total.output_ports[0]

    # ── integrate ───────────────────────────────────────────────────────────
    return _integrate_port(builder, cost_port, initial_cost, name)


def tracking_mse(
    builder,
    signal_port,
    t_data,
    y_data,
    weight: float = 1.0,
    interpolation: str = "linear",
    initial_cost: float = 0.0,
    name: str = "tracking_mse",
):
    r"""Add blocks to compute the **dataset tracking MSE**.

    Computes

    .. math::

        J = \int_0^T w \, \| \text{signal}(t) - y_{\text{ref}}(t) \|^2 \, dt

    where :math:`y_{\text{ref}}(t)` is the reference signal *interpolated*
    from the dataset ``(t_data, y_data)`` at every simulation time step.

    The function wires:

    1. :class:`~jaxonomy.library.Clock` → current simulation time
    2. :class:`~jaxonomy.library.LookupTable1d` → interpolated reference
    3. :func:`ise_objective` → squared error integrator

    Parameters
    ----------
    builder:
        The :class:`~jaxonomy.DiagramBuilder` to add blocks to.
    signal_port:
        Output port of the simulated signal to compare against the data.
    t_data:
        1-D array of reference time points (must be strictly increasing).
    y_data:
        Array of reference values.  Shape ``(N,)`` for scalar signals or
        ``(N, ny)`` for vector signals.  Extrapolation clamps to the
        nearest endpoint value.
    weight:
        Scalar multiplier applied before integration.
    interpolation:
        Interpolation method passed to :class:`~jaxonomy.library.LookupTable1d`:
        ``"linear"`` (default), ``"nearest"``, or ``"flat"``.
    initial_cost:
        Initial value of the integrator.
    name:
        Name prefix for the added blocks.

    Returns
    -------
    OutputPort
        Scalar port equal to *J* at the end of simulation.

    Examples
    --------
    Fit a model to measured step-response data::

        import numpy as np
        t_meas = np.linspace(0, 5, 50)
        y_meas = 1 - np.exp(-t_meas)   # first-order step response

        cost = tracking_mse(b, plant.output_ports[0], t_meas, y_meas)
    """
    from jaxonomy.library import Clock, LookupTable1d

    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    clock = builder.add(Clock(name=f"{name}_clock"))
    ref = builder.add(
        LookupTable1d(t_data, y_data, interpolation, name=f"{name}_ref")
    )
    builder.connect(clock.output_ports[0], ref.input_ports[0])

    return ise_objective(
        builder,
        signal_port,
        reference_port=ref.output_ports[0],
        weight=weight,
        initial_cost=initial_cost,
        name=name,
    )


def weighted_sum(
    builder,
    objectives: Sequence,
    weights: Sequence[float] | None = None,
    name: str = "total_cost",
):
    r"""Combine multiple objective ports into a **weighted sum**.

    .. math::

        J_{\text{total}} = \sum_{i} w_i \, J_i

    Parameters
    ----------
    builder:
        The :class:`~jaxonomy.DiagramBuilder` to add blocks to.
    objectives:
        Sequence of scalar output ports, one per term.
    weights:
        Scalar weights :math:`w_i`.  ``None`` → uniform weight ``1.0``.
        Must have the same length as *objectives* when provided.
    name:
        Name of the final :class:`~jaxonomy.library.Adder` block (and prefix
        for :class:`~jaxonomy.library.Gain` blocks when weights differ from 1).

    Returns
    -------
    OutputPort
        Scalar port equal to :math:`J_{\text{total}}`.

    Raises
    ------
    ValueError
        If *objectives* is empty or *weights* has a different length.

    Examples
    --------
    Combine two ISE objectives with different priorities::

        cost_pos = ise_objective(b, x.output_ports[0], name="ise_x")
        cost_vel = ise_objective(b, v.output_ports[0], name="ise_v")
        total    = weighted_sum(b, [cost_pos, cost_vel], weights=[10.0, 1.0])
    """
    from jaxonomy.library import Adder, Gain

    objectives = list(objectives)
    n = len(objectives)

    if n == 0:
        raise ValueError("weighted_sum: 'objectives' must not be empty.")

    if weights is None:
        weights = [1.0] * n
    else:
        weights = [float(w) for w in weights]
        if len(weights) != n:
            raise ValueError(
                f"weighted_sum: 'objectives' has {n} elements but "
                f"'weights' has {len(weights)} elements."
            )

    # Apply individual weights via Gain blocks where w ≠ 1
    scaled = []
    for i, (port, w) in enumerate(zip(objectives, weights)):
        if w != 1.0:
            g = builder.add(Gain(w, name=f"{name}_w{i}"))
            builder.connect(port, g.input_ports[0])
            scaled.append(g.output_ports[0])
        else:
            scaled.append(port)

    # Short-circuit for single objective
    if n == 1:
        return scaled[0]

    # Sum all scaled objectives
    ops = "+" * n
    adder = builder.add(Adder(n, operators=ops, name=name))
    for i, p in enumerate(scaled):
        builder.connect(p, adder.input_ports[i])
    return adder.output_ports[0]
