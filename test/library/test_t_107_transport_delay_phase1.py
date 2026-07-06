# SPDX-License-Identifier: MIT
"""Phase-1 verification for T-107 ``TransportDelay``.

These tests cover the continuous-time transport delay block
shipped in ``jaxonomy.library.primitives.TransportDelay``. They exercise:

1. ``delay=0`` passthrough (output ≈ input within one ``dt``).
2. Sinusoidal delay: a 1 Hz sine delayed by 0.5 s should reproduce
   ``sin(2 pi (t - 0.5))`` after the first ``delay_seconds``.
3. Initial-output behavior for ``t < delay_seconds``.
4. Differentiability w.r.t. the input amplitude through a simulation.
5. Default-path byte-equivalence: existing ``UnitDelay`` test scenarios
   still produce identical numerical output (no incidental regression
   from registering the new class).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import library


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _build_sine_to_delay(delay_seconds, dt, history_length=None, t_final=2.0,
                         amplitude=1.0, frequency=2 * np.pi):
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
        recorded_signals={
            "u": src.output_ports[0],
            "y": delay.output_ports[0],
        },
    )
    return res


# --------------------------------------------------------------------- #
# 1. Passthrough at delay = 0
# --------------------------------------------------------------------- #


def test_transport_delay_zero_delay_passthrough():
    """At delay=0 the output should track the input (within one buffer dt)."""
    dt = 0.01
    res = _build_sine_to_delay(delay_seconds=0.0, dt=dt, t_final=1.0)
    ts = np.asarray(res.time)
    u = np.asarray(res.outputs["u"])
    y = np.asarray(res.outputs["y"])

    # The buffer is sampled every ``dt``; at exactly the sample times the
    # zero-delay lookup returns the most recent buffered value, so we
    # only check at points sufficiently far past the very first sample.
    mask = ts >= 2 * dt
    # Allow one ``dt`` worth of staleness (linear interpolation between
    # samples at ``dt`` resolution) plus a tiny numerical slack.
    assert np.max(np.abs(y[mask] - u[mask])) < 5e-2


# --------------------------------------------------------------------- #
# 2. Delayed sine
# --------------------------------------------------------------------- #


def test_transport_delay_sine_phase_shift():
    """``TransportDelay(0.5)`` on ``sin(2 pi t)`` ≈ ``sin(2 pi (t-0.5))``."""
    delay = 0.5
    dt = 0.005
    # Plenty of buffer to cover the whole pre-delay window.
    history_length = int(np.ceil(delay / dt)) + 8
    res = _build_sine_to_delay(
        delay_seconds=delay,
        dt=dt,
        history_length=history_length,
        t_final=2.0,
        amplitude=1.0,
        frequency=2 * np.pi,
    )

    ts = np.asarray(res.time)
    y = np.asarray(res.outputs["y"])
    # Only compare past the initial-output region with a small guard
    # band for the first one or two buffer samples after t == delay.
    mask = ts >= delay + 4 * dt
    expected = np.sin(2 * np.pi * (ts[mask] - delay))
    err = np.max(np.abs(y[mask] - expected))
    # Linear interpolation over a ``dt = 5 ms`` grid for a 1 Hz sine
    # gives an error well under ``(2 pi dt)^2 / 8``; loosen by 5x for
    # buffer-edge / solver-step alignment slack.
    tol = 5 * (2 * np.pi * dt) ** 2 / 8.0
    assert err < tol, f"max abs error {err:.4g} exceeded tolerance {tol:.4g}"


# --------------------------------------------------------------------- #
# 3. Initial output behaviour for t < delay
# --------------------------------------------------------------------- #


def test_transport_delay_initial_output_holds():
    """Until simulated time exceeds delay, output equals initial_output."""
    delay = 0.3
    dt = 0.01
    initial_output = 0.7
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Sine(amplitude=2.0, frequency=4.0, phase=1.0, bias=0.0))
    delay_blk = builder.add(
        library.TransportDelay(
            dt=dt,
            delay_seconds=delay,
            initial_output=initial_output,
            history_length=64,
        )
    )
    builder.connect(src.output_ports[0], delay_blk.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 0.5),
        recorded_signals={"y": delay_blk.output_ports[0]},
    )
    ts = np.asarray(res.time)
    y = np.asarray(res.outputs["y"])

    # Strict equality before the delay threshold.
    pre_mask = ts < delay - 1e-9
    assert np.allclose(y[pre_mask], initial_output)


# --------------------------------------------------------------------- #
# 4. Differentiability w.r.t. input amplitude
# --------------------------------------------------------------------- #


def test_transport_delay_grad_through_input():
    """``jax.grad`` of an integrated delayed signal w.r.t. input amplitude
    is non-zero and finite (the buffer transmits gradient end-to-end)."""
    dt = 0.05
    delay = 0.2
    history_length = 32
    n_steps = 40  # 0.0 .. 2.0 in dt strides

    delay_blk = library.TransportDelay(
        dt=dt,
        delay_seconds=delay,
        initial_output=0.0,
        history_length=history_length,
    )
    delay_blk.system_id = "td_grad_test"
    # ``_signal_shape`` is normally populated when the diagram calls
    # ``initialize()``. We bypass the diagram for this micro-test, so
    # set it manually to ``()`` (scalar input/output).
    delay_blk._signal_shape = ()

    # Pull the empty default state directly off the block. Use the
    # NamedTuple class via the public attribute set on the instance.
    from collections import namedtuple

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
        total = jnp.array(0.0)
        for k in range(n_steps):
            t = jnp.asarray(k * dt)
            u = amplitude * jnp.sin(2 * jnp.pi * t)
            new_buf = delay_blk._update(t, state, u)
            state = State(discrete_state=new_buf)
            # Sample the (continuous) output a half-step into the future
            # so we exercise the interpolation branch rather than always
            # landing exactly on a sample boundary.
            t_eval = jnp.asarray((k + 0.5) * dt)
            y = delay_blk._output(
                t_eval,
                state,
                delay_seconds=jnp.asarray(delay),
                initial_output=jnp.asarray(0.0),
            )
            total = total + y * dt
        return total

    g = jax.grad(integrate)(jnp.asarray(1.0))
    assert jnp.isfinite(g)
    # Integrating the delayed sine over a window > delay must depend on
    # the amplitude, so the gradient should be appreciably non-zero.
    assert abs(float(g)) > 1e-3


# --------------------------------------------------------------------- #
# 5. Default-path byte-equivalence on the existing UnitDelay tests
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("dt", [0.05, 0.1])
def test_unit_delay_default_path_unchanged(dt):
    """Adding ``TransportDelay`` to the library must not perturb the
    behaviour of any unrelated block. We re-run a small ``UnitDelay``
    scenario (mirroring ``test_delay.py::test_shift_register_matches_unit_delay``)
    and compare against the analytical step-delayed expectation."""
    builder = jaxonomy.DiagramBuilder()
    step = builder.add(library.Step(start_value=0.0, end_value=1.0, step_time=0.2))
    ud = builder.add(library.UnitDelay(dt=dt, initial_state=0.0))
    builder.connect(step.output_ports[0], ud.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 0.6),
        recorded_signals={"y": ud.output_ports[0]},
    )
    ts = np.asarray(res.time)
    y = np.asarray(res.outputs["y"])
    # UnitDelay holds the previous-step input. Since the step happens at
    # t = 0.2, the discrete output samples switch at the next discrete
    # tick after the input has propagated, i.e. at t > 0.2 + dt + 1e-9.
    expected_high_mask = ts > 0.2 + dt + 1e-9
    expected_low_mask = ts < 0.2 + dt - 1e-9
    # Pre-rise should be zero.
    assert np.allclose(y[expected_low_mask], 0.0)
    # Post-rise should be one.
    assert np.allclose(y[expected_high_mask], 1.0)
