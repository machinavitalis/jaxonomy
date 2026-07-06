# SPDX-License-Identifier: MIT

"""T-107 phase 4 — PCHIP interpolation on the VariableTransportDelay buffer.

Phase 4 adds ``method="linear"|"pchip"`` to ``VariableTransportDelay``.
``"linear"`` (default) keeps the phase-3 behaviour byte-equivalent;
``"pchip"`` routes the per-output interpolation through the T-106
backend's monotone cubic Hermite interpolant. The motivation is a
smooth (C^1) gradient w.r.t. ``tau`` across sample boundaries — under
``method="linear"`` the gradient has a jump discontinuity at every
buffer tick.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import library


def _run_constant_tau(method, tau, *, dt=0.01, max_delay=0.5, t_final=2.0,
                      amplitude=1.0, frequency=2 * np.pi):
    """Sine -> VariableTransportDelay(method) with constant tau."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(
        library.Sine(amplitude=amplitude, frequency=frequency, phase=0.0, bias=0.0)
    )
    tau_src = builder.add(library.Constant(value=float(tau)))
    delay = builder.add(
        library.VariableTransportDelay(
            dt=dt,
            max_delay_seconds=max_delay,
            initial_output=0.0,
            method=method,
        )
    )
    builder.connect(src.output_ports[0], delay.input_ports[0])
    builder.connect(tau_src.output_ports[0], delay.input_ports[1])
    diagram = builder.build()
    ctx = diagram.create_context()
    return jaxonomy.simulate(
        diagram, ctx, (0.0, t_final),
        recorded_signals={"u": src.output_ports[0], "y": delay.output_ports[0]},
    )


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


def test_unknown_method_raises_block_parameter_error():
    from jaxonomy.framework.error import BlockParameterError

    with pytest.raises(BlockParameterError, match="method must be"):
        library.VariableTransportDelay(
            dt=0.01, max_delay_seconds=0.5, method="cubic"
        )


def test_default_method_is_linear():
    blk = library.VariableTransportDelay(dt=0.01, max_delay_seconds=0.5)
    assert blk.method == "linear"


# ---------------------------------------------------------------------------
# Backward compatibility: method="linear" matches the pre-phase-4 output.
# ---------------------------------------------------------------------------


def test_linear_method_matches_omitted_method_default():
    """Passing method="linear" explicitly is byte-equivalent to the
    pre-phase-4 default (which was always linear interpolation)."""
    res_default = _run_constant_tau("linear", tau=0.3)
    # Build a second diagram without the method kwarg at all and compare.
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Sine(amplitude=1.0, frequency=2 * np.pi))
    tau_src = builder.add(library.Constant(value=0.3))
    delay = builder.add(
        library.VariableTransportDelay(
            dt=0.01, max_delay_seconds=0.5, initial_output=0.0
        )
    )
    builder.connect(src.output_ports[0], delay.input_ports[0])
    builder.connect(tau_src.output_ports[0], delay.input_ports[1])
    diag = builder.build()
    ctx = diag.create_context()
    res_omit = jaxonomy.simulate(
        diag, ctx, (0.0, 2.0),
        recorded_signals={"y": delay.output_ports[0]},
    )

    np.testing.assert_array_equal(
        np.asarray(res_default.outputs["y"]),
        np.asarray(res_omit.outputs["y"]),
    )


# ---------------------------------------------------------------------------
# PCHIP path: same general signal shape as linear (delayed sine), but
# differs at the sub-sample scale.
# ---------------------------------------------------------------------------


def test_pchip_approximates_delayed_sine():
    """For a constant tau, both methods should recover an approximately
    delayed sine. Tolerance is loose because the ring buffer is sampled
    at dt and tau is offset off-grid."""
    tau = 0.25
    res = _run_constant_tau("pchip", tau=tau, dt=0.01, t_final=1.5)
    t = np.asarray(res.time)
    y = np.asarray(res.outputs["y"])
    # Theoretical delayed sine, zero before t < tau (initial_output=0).
    expected = np.where(t < tau, 0.0, np.sin(2 * np.pi * (t - tau)))
    # PCHIP plus the ring-buffer discretisation gives a couple of percent
    # max error on this fixture.
    err = np.max(np.abs(y - expected))
    assert err < 0.05, f"PCHIP delayed-sine error {err:.4g} too high"


def test_pchip_differs_from_linear_at_sub_sample_resolution():
    """Verify the two methods actually produce different numbers on
    off-grid queries. ``tau`` is deliberately not a multiple of ``dt``
    so ``query_t = time - tau`` never lands on a ring-buffer tick —
    PCHIP and linear must then disagree on every recorded sample."""
    # tau=0.253 is offset by 3 ms from any dt=0.01 tick, so every
    # recorded output uses sub-sample interpolation.
    res_lin = _run_constant_tau("linear", tau=0.253, dt=0.01)
    res_pch = _run_constant_tau("pchip", tau=0.253, dt=0.01)
    diff = np.max(np.abs(np.asarray(res_lin.outputs["y"])
                         - np.asarray(res_pch.outputs["y"])))
    # Sine at this dt: PCHIP vs linear differs by O(1e-3) on smooth
    # interior segments; require a clearly-non-noise gap.
    assert diff > 1e-4, (
        f"expected linear vs pchip to disagree noticeably, got {diff:.4g}"
    )
    assert diff < 0.1, (
        f"linear vs pchip diverged by {diff:.4g} — sanity-check fail"
    )


# ---------------------------------------------------------------------------
# Differentiability: jax.grad w.r.t. tau flows under both methods.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "pchip"])
def test_jax_grad_through_tau_is_finite_and_nonzero(method):
    """Headline T-107 phase 4 check: under either method, jax.grad of an
    integrated delayed signal w.r.t. ``tau`` flows and produces a
    non-trivial gradient.

    Uses the same synthetic-loop pattern as the phase-1 grad test
    (bypassing the diagram simulator, which doesn't compose recorded
    signals with autodiff) so we can isolate the interpolation-method
    contribution to differentiability.
    """
    from collections import namedtuple

    dt = 0.05
    max_delay = 1.0
    history_length = 32
    n_steps = 40

    blk = library.VariableTransportDelay(
        dt=dt,
        max_delay_seconds=max_delay,
        initial_output=0.0,
        history_length=history_length,
        method=method,
    )
    blk.system_id = f"vtd_grad_{method}"
    blk._signal_shape = ()

    State = namedtuple("State", ["discrete_state"])
    BufferState = blk._BufferState

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
            new_buf = blk._update(t, state, u, tau_value)
            state = State(discrete_state=new_buf)
            # Half-step into the future so we exercise interpolation
            # rather than always landing on a buffer sample boundary —
            # that's where PCHIP vs linear actually matters for the
            # tau-gradient.
            t_eval = jnp.asarray((k + 0.5) * dt)
            y = blk._output(
                t_eval, state, u, tau_value,
                initial_output=jnp.asarray(0.0),
            )
            total = total + y * dt
        return total

    g = float(jax.grad(integrate)(jnp.asarray(0.2)))
    assert np.isfinite(g), f"non-finite gradient {g!r} under method={method!r}"
    assert abs(g) > 1e-3, (
        f"expected non-trivial tau gradient under method={method!r}, got {g:.4g}"
    )
