# SPDX-License-Identifier: MIT
"""Phase-1 verification for T-107-followup-variable-tau ``VariableTransportDelay``.

These tests cover the signal-driven continuous-time transport delay block
shipped in ``jaxonomy.library.primitives.VariableTransportDelay``. They
exercise:

1. Constant-tau equivalence: ``VariableTransportDelay`` driven by a
   constant ``tau`` matches the fixed :class:`TransportDelay` (T-107
   phase 1) within tight tolerance.
2. Time-varying tau: ``tau(t) = 0.3 + 0.2 * sin(t)`` produces an output
   that reflects the changing delay (and is distinct from any constant-
   delay reference).
3. Differentiability: ``jax.grad`` of an integrated delayed signal w.r.t.
   the (constant) delay value is non-zero — gradient flows through the
   ``tau`` input port.
4. Bounds enforcement: requesting ``tau > max_delay_seconds`` clamps to
   ``max_delay_seconds`` (rather than raising or producing NaNs).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import library


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _build_variable_delay_with_constant_tau(
    constant_tau,
    max_delay,
    dt,
    history_length=None,
    t_final=2.0,
    amplitude=1.0,
    frequency=2 * np.pi,
):
    """Wire ``Sine -> VariableTransportDelay`` with a constant tau source."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(
        library.Sine(amplitude=amplitude, frequency=frequency, phase=0.0, bias=0.0)
    )
    tau_src = builder.add(library.Constant(value=float(constant_tau)))
    delay = builder.add(
        library.VariableTransportDelay(
            dt=dt,
            max_delay_seconds=max_delay,
            initial_output=0.0,
            history_length=history_length,
        )
    )
    builder.connect(src.output_ports[0], delay.input_ports[0])
    builder.connect(tau_src.output_ports[0], delay.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_final),
        recorded_signals={
            "u": src.output_ports[0],
            "y": delay.output_ports[0],
        },
    )
    return res


def _build_fixed_delay(
    delay_seconds,
    dt,
    history_length=None,
    t_final=2.0,
    amplitude=1.0,
    frequency=2 * np.pi,
):
    """Reference: same Sine wired through fixed :class:`TransportDelay`."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(
        library.Sine(amplitude=amplitude, frequency=frequency, phase=0.0, bias=0.0)
    )
    delay = builder.add(
        library.TransportDelay(
            dt=dt,
            delay_seconds=delay_seconds,
            initial_output=0.0,
            history_length=history_length,
        )
    )
    builder.connect(src.output_ports[0], delay.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_final),
        recorded_signals={"y": delay.output_ports[0]},
    )
    return res


# --------------------------------------------------------------------- #
# 1. Constant-tau equivalence with TransportDelay phase 1
# --------------------------------------------------------------------- #


def test_variable_transport_delay_matches_fixed_for_constant_tau():
    """Constant ``tau = 0.5`` should match ``TransportDelay(0.5)``."""
    delay = 0.5
    dt = 0.005
    history_length = int(np.ceil(delay / dt)) + 8
    t_final = 2.0

    res_var = _build_variable_delay_with_constant_tau(
        constant_tau=delay,
        max_delay=1.0,  # generous bound; clip is inactive
        dt=dt,
        history_length=history_length,
        t_final=t_final,
    )
    res_fix = _build_fixed_delay(
        delay_seconds=delay,
        dt=dt,
        history_length=history_length,
        t_final=t_final,
    )

    ts_var = np.asarray(res_var.time)
    ts_fix = np.asarray(res_fix.time)
    y_var = np.asarray(res_var.outputs["y"])
    y_fix = np.asarray(res_fix.outputs["y"])

    # Adaptive solvers may pick slightly different step sequences; align
    # by interpolating both onto a common grid past the initial-output
    # window.
    t_eval = np.linspace(delay + 8 * dt, ts_var[-1], 200)
    y_var_i = np.interp(t_eval, ts_var, y_var)
    y_fix_i = np.interp(t_eval, ts_fix, y_fix)

    err = float(np.max(np.abs(y_var_i - y_fix_i)))
    # Both blocks share the same ring-buffer machinery; with identical
    # static shape the only difference is how ``tau`` is sourced. The
    # constant Constant->port path adds no numerical noise, so we use a
    # tight tolerance with a small slack for solver-step alignment.
    assert err < 1e-5, f"max abs error {err:.4g} between variable and fixed delay"


# --------------------------------------------------------------------- #
# 2. Time-varying tau changes the output relative to constant-tau
# --------------------------------------------------------------------- #


def test_variable_transport_delay_time_varying_tau_changes_output():
    """``tau(t) = 0.3 + 0.2 * sin(t)`` must yield an output distinct
    from any constant delay (the variable delay must actually vary)."""
    dt = 0.01
    max_delay = 0.6
    history_length = int(np.ceil(max_delay / dt)) + 8
    t_final = 3.0

    # Build: Sine(input data) -> VariableTransportDelay <- (tau source)
    # where tau source = bias 0.3 + sine of amp 0.2.
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(
        library.Sine(amplitude=1.0, frequency=2 * np.pi, phase=0.0, bias=0.0)
    )
    tau_src = builder.add(
        library.Sine(amplitude=0.2, frequency=1.0, phase=0.0, bias=0.3)
    )
    delay = builder.add(
        library.VariableTransportDelay(
            dt=dt,
            max_delay_seconds=max_delay,
            initial_output=0.0,
            history_length=history_length,
        )
    )
    builder.connect(src.output_ports[0], delay.input_ports[0])
    builder.connect(tau_src.output_ports[0], delay.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_final),
        recorded_signals={
            "u": src.output_ports[0],
            "y": delay.output_ports[0],
            "tau": tau_src.output_ports[0],
        },
    )

    ts = np.asarray(res.time)
    y_var = np.asarray(res.outputs["y"])
    tau = np.asarray(res.outputs["tau"])

    # Reference: fixed delay at the mean tau (0.3) over same interval.
    res_fix = _build_fixed_delay(
        delay_seconds=0.3,
        dt=dt,
        history_length=history_length,
        t_final=t_final,
    )
    ts_fix = np.asarray(res_fix.time)
    y_fix = np.asarray(res_fix.outputs["y"])

    # Past the initial-output window, the variable-delay output should
    # differ meaningfully from any single fixed delay.
    mask = ts >= max_delay + 4 * dt
    y_fix_on_var = np.interp(ts[mask], ts_fix, y_fix)
    diff = float(np.max(np.abs(y_var[mask] - y_fix_on_var)))
    assert diff > 1e-2, (
        f"variable tau output should differ from fixed-tau reference; "
        f"max diff was {diff:.4g}"
    )

    # And the analytical signal-driven expectation is
    # ``y(t) ~= sin(2 pi (t - tau(t)))`` (within linear-interp error on a
    # ``dt`` grid). Check on the post-warmup window.
    expected = np.sin(2 * np.pi * (ts[mask] - tau[mask]))
    err = float(np.max(np.abs(y_var[mask] - expected)))
    # Loose tolerance: linear interp on a 10 ms grid for a 1 Hz sine,
    # plus solver-step misalignment slack.
    assert err < 0.05, f"variable-tau output deviates from analytic by {err:.4g}"


# --------------------------------------------------------------------- #
# 3. Differentiability w.r.t. the delay input
# --------------------------------------------------------------------- #


def test_variable_transport_delay_grad_through_tau():
    """``jax.grad`` of an integrated delayed signal w.r.t. a (constant)
    delay value is non-zero — gradient flows through the ``tau`` port."""
    dt = 0.05
    max_delay = 1.0
    history_length = 32
    n_steps = 40  # 0.0 .. 2.0 in dt strides

    delay_blk = library.VariableTransportDelay(
        dt=dt,
        max_delay_seconds=max_delay,
        initial_output=0.0,
        history_length=history_length,
    )
    delay_blk.system_id = "vtd_grad_test"
    # ``_signal_shape`` is normally populated when the diagram calls
    # ``initialize()``. We bypass the diagram for this micro-test, so
    # set it manually to ``()`` (scalar input/output).
    delay_blk._signal_shape = ()

    State = namedtuple("State", ["discrete_state"])
    BufferState = delay_blk._BufferState

    sentinel_t0 = -dt * (history_length + 1) - 1.0
    times0 = jnp.asarray(
        sentinel_t0 + dt * np.arange(history_length, dtype=np.float64)
    )[::-1]
    values0 = jnp.zeros((history_length,))
    init_buffer = BufferState(times=times0, values=values0)

    def integrate(tau_value):
        state = State(discrete_state=init_buffer)
        total = jnp.array(0.0)
        for k in range(n_steps):
            t = jnp.asarray(k * dt)
            u = jnp.sin(2 * jnp.pi * t)
            new_buf = delay_blk._update(t, state, u, tau_value)
            state = State(discrete_state=new_buf)
            # Sample the (continuous) output a half-step into the future
            # so we exercise interpolation rather than always landing on
            # a sample boundary.
            t_eval = jnp.asarray((k + 0.5) * dt)
            y = delay_blk._output(
                t_eval,
                state,
                u,           # input port 0 (data) — unused by _output
                tau_value,   # input port 1 (delay) — read inside _output
                initial_output=jnp.asarray(0.0),
            )
            total = total + y * dt
        return total

    # Gradient w.r.t. tau (the headline feature: differentiable delay).
    g_tau = jax.grad(integrate)(jnp.asarray(0.2))
    assert jnp.isfinite(g_tau)
    assert abs(float(g_tau)) > 1e-3, (
        f"gradient w.r.t. tau should be non-trivial; got {float(g_tau):.4g}"
    )


def test_variable_transport_delay_grad_through_data_input():
    """Sanity: gradient also flows through the data input (same as the
    fixed-delay block — VariableTransportDelay must not regress this)."""
    dt = 0.05
    max_delay = 1.0
    history_length = 32
    n_steps = 40
    fixed_tau = 0.2

    delay_blk = library.VariableTransportDelay(
        dt=dt,
        max_delay_seconds=max_delay,
        initial_output=0.0,
        history_length=history_length,
    )
    delay_blk.system_id = "vtd_grad_input_test"
    delay_blk._signal_shape = ()

    State = namedtuple("State", ["discrete_state"])
    BufferState = delay_blk._BufferState

    sentinel_t0 = -dt * (history_length + 1) - 1.0
    times0 = jnp.asarray(
        sentinel_t0 + dt * np.arange(history_length, dtype=np.float64)
    )[::-1]
    values0 = jnp.zeros((history_length,))
    init_buffer = BufferState(times=times0, values=values0)

    def integrate(amplitude):
        state = State(discrete_state=init_buffer)
        tau = jnp.asarray(fixed_tau)
        total = jnp.array(0.0)
        for k in range(n_steps):
            t = jnp.asarray(k * dt)
            u = amplitude * jnp.sin(2 * jnp.pi * t)
            new_buf = delay_blk._update(t, state, u, tau)
            state = State(discrete_state=new_buf)
            t_eval = jnp.asarray((k + 0.5) * dt)
            y = delay_blk._output(
                t_eval,
                state,
                u,
                tau,
                initial_output=jnp.asarray(0.0),
            )
            total = total + y * dt
        return total

    g_amp = jax.grad(integrate)(jnp.asarray(1.0))
    assert jnp.isfinite(g_amp)
    assert abs(float(g_amp)) > 1e-3


# --------------------------------------------------------------------- #
# 4. Bounds enforcement: tau > max_delay_seconds clips to max_delay
# --------------------------------------------------------------------- #


def test_variable_transport_delay_clips_oversize_tau():
    """``tau > max_delay_seconds`` should clamp to ``max_delay_seconds``;
    output must equal what we'd get at the saturated delay."""
    dt = 0.01
    max_delay = 0.4
    history_length = int(np.ceil(max_delay / dt)) + 8
    t_final = 1.5

    # tau driver = constant 0.9, well above the 0.4 ceiling.
    res_clipped = _build_variable_delay_with_constant_tau(
        constant_tau=0.9,
        max_delay=max_delay,
        dt=dt,
        history_length=history_length,
        t_final=t_final,
    )
    # Reference: tau driver = exactly max_delay (0.4).
    res_at_ceiling = _build_variable_delay_with_constant_tau(
        constant_tau=max_delay,
        max_delay=max_delay,
        dt=dt,
        history_length=history_length,
        t_final=t_final,
    )

    ts_a = np.asarray(res_clipped.time)
    ts_b = np.asarray(res_at_ceiling.time)
    y_a = np.asarray(res_clipped.outputs["y"])
    y_b = np.asarray(res_at_ceiling.outputs["y"])

    # Compare past the initial-output window on a common grid.
    t_eval = np.linspace(max_delay + 8 * dt, min(ts_a[-1], ts_b[-1]), 100)
    y_a_i = np.interp(t_eval, ts_a, y_a)
    y_b_i = np.interp(t_eval, ts_b, y_b)
    err = float(np.max(np.abs(y_a_i - y_b_i)))
    assert err < 1e-5, (
        f"clipped (tau=0.9 -> max_delay={max_delay}) output must equal "
        f"ceiling reference; max diff {err:.4g}"
    )


# --------------------------------------------------------------------- #
# 5. Default-path / __init__ smoke tests (parameter validation)
# --------------------------------------------------------------------- #


def test_variable_transport_delay_rejects_nonpositive_dt():
    from jaxonomy.framework.error import BlockParameterError

    with pytest.raises(BlockParameterError):
        library.VariableTransportDelay(dt=0.0, max_delay_seconds=0.5)


def test_variable_transport_delay_rejects_negative_max_delay():
    from jaxonomy.framework.error import BlockParameterError

    with pytest.raises(BlockParameterError):
        library.VariableTransportDelay(dt=0.01, max_delay_seconds=-0.1)


def test_variable_transport_delay_rejects_history_length_lt_2():
    from jaxonomy.framework.error import BlockParameterError

    with pytest.raises(BlockParameterError):
        library.VariableTransportDelay(
            dt=0.01, max_delay_seconds=0.0, history_length=1
        )
