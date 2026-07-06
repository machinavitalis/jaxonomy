# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-discrete-filter-family.

T-127 phase 1 shipped :class:`PIDController2DOF`.  The deferred follow-up
``T-127-followup-discrete-filter-family`` adds the discrete-filter blocks
commonly needed alongside a PID:

* :class:`LowPassDiscrete` — single-pole RC low-pass filter,
  ``y[k] = alpha*x[k] + (1-alpha)*y[k-1]`` with
  ``alpha = dt / (dt + 1/(2*pi*cutoff_hz))``.
* :class:`LeadLag` — first-order lead-lag compensator
  ``G(s) = K * (1 + T_lead*s) / (1 + T_lag*s)`` discretised by Tustin
  (bilinear) transform.

These tests cover step / steady-state behaviour, the analytic identity
``T_lead == T_lag`` ⇒ pure-gain LeadLag, and ``jax.grad`` finiteness
through the tunable parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    LeadLag,
    LowPassDiscrete,
    Sine,
    Step,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_lpf(
    input_block,
    *,
    dt=0.01,
    cutoff_hz=1.0,
    initial_state=0.0,
    t_end=2.0,
):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(input_block)
    lpf = builder.add(
        LowPassDiscrete(
            dt=dt,
            cutoff_hz=cutoff_hz,
            initial_state=initial_state,
            name="lpf",
        )
    )
    builder.connect(src.output_ports[0], lpf.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"y": lpf.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["y"], results.time


def _simulate_leadlag(
    input_block,
    *,
    dt=0.01,
    K=1.0,
    T_lead=1.0,
    T_lag=1.0,
    initial_state=0.0,
    t_end=2.0,
):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(input_block)
    ll = builder.add(
        LeadLag(
            dt=dt,
            K=K,
            T_lead=T_lead,
            T_lag=T_lag,
            initial_state=initial_state,
            name="leadlag",
        )
    )
    builder.connect(src.output_ports[0], ll.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"y": ll.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["y"], results.time


# --------------------------------------------------------------------- #
# LowPassDiscrete: step response & cutoff sanity
# --------------------------------------------------------------------- #


class TestLowPassDiscreteStepResponse:
    """Step response should rise smoothly toward the input amplitude."""

    def test_step_response_rises_to_one(self):
        # Step input of amplitude 1 starting at t=0.0; cutoff well below
        # the sample rate so the discrete approximation is faithful.
        dt = 0.01
        cutoff_hz = 5.0  # tau ~ 0.032 s; 1 second is ~30 tau
        y, _ = _simulate_lpf(
            Step(start_value=0.0, end_value=1.0, step_time=0.0, name="u"),
            dt=dt,
            cutoff_hz=cutoff_hz,
            t_end=2.0,
        )
        # Final value should be close to 1.0 (steady state of unit-gain LPF).
        assert jnp.abs(y[-1] - 1.0) < 1e-3, (
            f"LPF step response did not settle near 1.0; y[-1] = {float(y[-1])}"
        )
        # Output should be monotonically non-decreasing during the rise.
        # Sample a few interior points and check they are non-decreasing.
        N = y.shape[0]
        sample_idx = jnp.arange(0, N, max(1, N // 50))
        y_samples = y[sample_idx]
        diffs = jnp.diff(y_samples)
        assert jnp.all(diffs >= -1e-12), (
            "LPF step response should be (weakly) monotonically increasing; "
            f"saw {float(jnp.min(diffs))}"
        )
        # And bounded above by the input amplitude.
        assert jnp.max(y) <= 1.0 + 1e-9

    def test_minus_3db_at_design_cutoff(self):
        """Sine input at the design cutoff should attenuate to ~ -3 dB.

        For the simple-RC update
            tau = 1/(2*pi*fc), alpha = dt/(dt+tau),
            y[k] = alpha*x[k] + (1-alpha)*y[k-1]
        the steady-state amplitude ratio at angular frequency w is
            |H(e^{jwT})| = alpha / sqrt(1 - 2*(1-alpha)*cos(wT) + (1-alpha)**2).
        For dt << tau (here dt=1e-4 s, fc=10 Hz → tau ~ 0.0159 s, dt/tau ~
        0.0063), this matches the continuous-time -3 dB at w = 1/tau
        (i.e. f = fc) within a few percent.

        Note: ``Sine(frequency=...)`` interprets its ``frequency`` argument
        as *angular* frequency (rad/s), so we drive at ``omega = 2*pi*fc``.
        """
        # Note: the simulator's recorded-signals buffer caps at ~500
        # samples per signal; ``t_end / dt`` must stay ≤ 500 or the
        # buffer wraps and only the last sample survives.  We pick
        # ``dt=0.01, t_end=5.0`` → 500 samples, 5 full periods at fc=1 Hz.
        dt = 0.01
        fc = 1.0  # Hz; tau ~ 0.159 s, well above dt → small discrete drift.
        omega = 2.0 * float(jnp.pi) * fc  # rad/s for the Sine block.
        t_end = 5.0
        y, t = _simulate_lpf(
            Sine(frequency=omega, amplitude=1.0, phase=0.0),
            dt=dt,
            cutoff_hz=fc,
            t_end=t_end,
        )
        # Take the last 2 seconds (steady state, ~2 full periods) and
        # measure the peak-to-peak amplitude.
        assert y.shape[0] > 100, f"too few recorded samples: {y.shape[0]}"
        cutoff_t = float(t[-1]) - 2.0
        mask = t >= cutoff_t
        y_ss = y[mask]
        amp = 0.5 * (jnp.max(y_ss) - jnp.min(y_ss))
        # -3 dB ≈ 1/sqrt(2) ≈ 0.7071.  Allow a generous window
        # (0.65 .. 0.78) because of: (a) the discrete-vs-continuous cutoff
        # drift, and (b) we measure peak-to-peak on a finite window so we
        # may miss the true peak by up to one sample.
        assert 0.65 < float(amp) < 0.78, (
            f"Expected ~-3 dB (amp ~ 0.707) at the design cutoff; "
            f"got peak-to-peak/2 = {float(amp)}"
        )

    def test_dc_passthrough(self):
        """Constant input → constant output equal to the input (unit gain)."""
        dt = 0.01
        y, _ = _simulate_lpf(
            Constant(2.5, name="dc"),
            dt=dt,
            cutoff_hz=2.0,
            initial_state=0.0,
            t_end=3.0,
        )
        # Final output should be ~2.5; a few tau is enough.
        assert jnp.abs(y[-1] - 2.5) < 1e-3, (
            f"DC gain should be 1; final y = {float(y[-1])}"
        )


# --------------------------------------------------------------------- #
# LeadLag: identity collapse and step behaviour
# --------------------------------------------------------------------- #


class TestLeadLagIdentity:
    """K = 1, T_lead = T_lag should collapse to a unity-gain passthrough."""

    def test_identity_on_constant_input(self):
        dt = 0.01
        y, _ = _simulate_leadlag(
            Constant(3.0, name="u"),
            dt=dt,
            K=1.0,
            T_lead=0.5,
            T_lag=0.5,
            initial_state=0.0,
            t_end=1.0,
        )
        # After the first tick, output should equal the input exactly.
        assert jnp.allclose(y[1:], 3.0, atol=1e-10), (
            f"Identity LeadLag on a constant should pass it through; got {y}"
        )

    def test_identity_on_sine_input(self):
        """K=1, T_lead=T_lag → output sine equals input sine exactly.

        ``Sine.frequency`` is angular (rad/s), so the analytic form is
        ``A * sin(omega * t + phi)`` with the same ``omega`` we pass to
        the block.
        """
        dt = 0.01
        omega = 4.0  # rad/s
        amp = 1.5
        phi = 0.3
        y, t = _simulate_leadlag(
            Sine(frequency=omega, amplitude=amp, phase=phi),
            dt=dt,
            K=1.0,
            T_lead=0.2,
            T_lag=0.2,
            initial_state=0.0,
            t_end=2.0,
        )
        # Compare against the analytic input everywhere except the very
        # first couple of ticks (the LeadLag's internal x_prev seed is 0).
        x_expected = amp * jnp.sin(omega * t + phi)
        # Need at least a few samples for the diff to be meaningful.
        assert y.shape[0] > 5, f"too few recorded samples: {y.shape[0]}"
        diff = jnp.abs(y[2:] - x_expected[2:])
        assert jnp.max(diff) < 1e-9, (
            f"Identity LeadLag on sine should pass it through; "
            f"max diff = {float(jnp.max(diff))}"
        )

    def test_pure_gain_when_K_only(self):
        """K, T_lead == T_lag arbitrary, only the gain should appear."""
        dt = 0.01
        K = 4.5
        y, _ = _simulate_leadlag(
            Constant(2.0, name="u"),
            dt=dt,
            K=K,
            T_lead=0.7,
            T_lag=0.7,
            initial_state=0.0,
            t_end=1.0,
        )
        assert jnp.allclose(y[1:], 2.0 * K, atol=1e-10), (
            f"Lead/lag with T_lead=T_lag should yield K*input; got {y}"
        )


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` is finite w.r.t. cutoff / K / T_lead / T_lag.

    Bypasses the simulator (whose recorded-signals path is not JAX-
    traceable) and exercises the recursive update / output methods
    directly across a small loop — the same pattern used by
    ``test_t_127_pid2dof_phase1.TestDifferentiability``.
    """

    @staticmethod
    def _make_lpf(cutoff_hz, dt=0.01):
        block = LowPassDiscrete(
            dt=dt, cutoff_hz=cutoff_hz, initial_state=0.0, name="lpf"
        )
        block.initialize(cutoff_hz=cutoff_hz, initial_state=0.0)
        return block

    @classmethod
    def _lpf_step_loss(cls, cutoff_hz, n_steps=10):
        """Run the LPF with a constant input and sum |y|."""
        from collections import namedtuple

        State = namedtuple("State", ["discrete_state"])
        block = cls._make_lpf(cutoff_hz)

        y_prev = jnp.asarray(0.0)
        x = jnp.asarray(1.0)
        params = dict(cutoff_hz=cutoff_hz)
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            state = State(discrete_state=y_prev)
            y_new = block._update(jnp.asarray(0.0), state, x, **params)
            total = total + jnp.abs(y_new)
            y_prev = y_new
        return total

    def test_lpf_grad_wrt_cutoff_finite(self):
        g = jax.grad(self._lpf_step_loss)(5.0)
        assert jnp.isfinite(g), f"grad wrt cutoff_hz not finite: {g}"
        # Higher cutoff → faster rise → larger sum of |y|; gradient > 0.
        assert float(g) > 0, f"grad wrt cutoff_hz should be > 0; got {g}"

    @staticmethod
    def _make_leadlag(K, T_lead, T_lag, dt=0.01):
        block = LeadLag(
            dt=dt,
            K=K,
            T_lead=T_lead,
            T_lag=T_lag,
            initial_state=0.0,
            name="leadlag",
        )
        block.initialize(K=K, T_lead=T_lead, T_lag=T_lag, initial_state=0.0)
        return block

    @classmethod
    def _leadlag_step_loss(cls, K, T_lead, T_lag, n_steps=10):
        from collections import namedtuple

        State = namedtuple("State", ["discrete_state"])
        block = cls._make_leadlag(K, T_lead, T_lag)

        xd = block.DiscreteStateType(
            x_prev=jnp.asarray(0.0), y_prev=jnp.asarray(0.0)
        )
        x = jnp.asarray(1.0)
        params = dict(K=K, T_lead=T_lead, T_lag=T_lag)
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            state = State(discrete_state=xd)
            y = block._output(jnp.asarray(0.0), state, x, **params)
            total = total + jnp.abs(y)
            xd = block._update(jnp.asarray(0.0), state, x, **params)
        return total

    def test_leadlag_grad_wrt_K_finite_and_nonzero(self):
        g = jax.grad(self._leadlag_step_loss, argnums=0)(2.0, 0.3, 0.5)
        assert jnp.isfinite(g), f"grad wrt K not finite: {g}"
        assert jnp.abs(g) > 0, f"grad wrt K should be nonzero; got {g}"

    def test_leadlag_grad_wrt_T_lead_finite(self):
        g = jax.grad(self._leadlag_step_loss, argnums=1)(2.0, 0.3, 0.5)
        assert jnp.isfinite(g), f"grad wrt T_lead not finite: {g}"

    def test_leadlag_grad_wrt_T_lag_finite(self):
        g = jax.grad(self._leadlag_step_loss, argnums=2)(2.0, 0.3, 0.5)
        assert jnp.isfinite(g), f"grad wrt T_lag not finite: {g}"


# --------------------------------------------------------------------- #
# T-005 default-float64 policy
# --------------------------------------------------------------------- #


class TestDefaultFloat64:
    """Discrete state should be the JAX/NumPy default float (float64 with x64)."""

    def test_lpf_state_is_default_float(self):
        dt = 0.01
        y, _ = _simulate_lpf(
            Constant(1.0, name="dc"),
            dt=dt,
            cutoff_hz=2.0,
            t_end=0.1,
        )
        assert y.dtype == jnp.asarray(0.0).dtype

    def test_leadlag_state_is_default_float(self):
        dt = 0.01
        y, _ = _simulate_leadlag(
            Constant(1.0, name="dc"),
            dt=dt,
            K=1.0,
            T_lead=0.2,
            T_lag=0.2,
            t_end=0.1,
        )
        assert y.dtype == jnp.asarray(0.0).dtype
