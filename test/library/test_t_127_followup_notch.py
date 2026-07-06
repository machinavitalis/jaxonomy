# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-notch-filter.

Completes the T-127 discrete-filter family (which already ships
``LowPassDiscrete`` and ``LeadLag`` via
T-127-followup-discrete-filter-family) by exercising the deferred
:class:`Notch` band-stop biquad.

Coverage:

* On-notch attenuation: a sinusoid at the notch centre frequency is
  reduced to a small residual relative to its drive amplitude.
* Off-notch passthrough: a sinusoid well outside the notch comes
  through with roughly its original amplitude.
* DC gain ~ 1 (constant input passes through after the delay-line
  warms up).
* ``jax.grad`` is finite (and non-zero) w.r.t. ``frequency_hz``.
* T-005: discrete-state dtype matches the JAX/NumPy default float.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    Notch,
    Sine,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_notch(
    input_block,
    *,
    dt=1.0 / 1000.0,
    frequency_hz=50.0,
    bandwidth_hz=5.0,
    depth=0.99,
    initial_state=0.0,
    t_end=0.5,
):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(input_block)
    notch = builder.add(
        Notch(
            dt=dt,
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            depth=depth,
            initial_state=initial_state,
            name="notch",
        )
    )
    builder.connect(src.output_ports[0], notch.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"y": notch.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["y"], results.time


# --------------------------------------------------------------------- #
# On-notch attenuation
# --------------------------------------------------------------------- #


class TestNotchOnNotch:
    """A sinusoid at the notch centre frequency should be deeply attenuated."""

    def test_50hz_attenuated_at_50hz_notch(self):
        # 50 Hz notch with 5 Hz bandwidth, drive at 50 Hz.  Use a 1 kHz
        # sample rate (dt = 1 ms) so the digital notch sits well below
        # Nyquist.  ``Sine.frequency`` is *angular* (rad/s), so we pass
        # ``2*pi*50``.
        dt = 1.0 / 1000.0
        f_notch = 50.0
        amp = 1.0
        # Keep the buffer below the simulator's per-signal cap (~500
        # samples), and give the filter ~25 periods to settle.
        t_end = 0.5
        y, t = _simulate_notch(
            Sine(
                frequency=2.0 * float(jnp.pi) * f_notch,
                amplitude=amp,
                phase=0.0,
            ),
            dt=dt,
            frequency_hz=f_notch,
            bandwidth_hz=5.0,
            depth=0.99,
            t_end=t_end,
        )
        assert y.shape[0] > 100, f"too few recorded samples: {y.shape[0]}"
        # Discard the warm-up (first ~half) and inspect the steady-state
        # peak-to-peak; on-notch attenuation should be deep.
        N = y.shape[0]
        y_ss = y[N // 2:]
        amp_out = 0.5 * (jnp.max(y_ss) - jnp.min(y_ss))
        assert float(amp_out) < 0.25 * amp, (
            "Notch failed to attenuate the on-notch sinusoid: "
            f"output amp = {float(amp_out)} (input amp = {amp})"
        )


# --------------------------------------------------------------------- #
# Off-notch passthrough
# --------------------------------------------------------------------- #


class TestNotchOffNotch:
    """A sinusoid well above the notch centre should pass through ~ unchanged."""

    def test_200hz_passthrough_with_50hz_notch(self):
        # Drive at 200 Hz, far above the 50 Hz notch + 5 Hz bandwidth.
        # The block should preserve the amplitude to within a few percent.
        dt = 1.0 / 2000.0  # 2 kHz sample rate: Nyquist = 1 kHz
        f_drive = 200.0
        f_notch = 50.0
        amp = 1.0
        t_end = 0.2  # 400 samples; 40 drive periods
        y, t = _simulate_notch(
            Sine(
                frequency=2.0 * float(jnp.pi) * f_drive,
                amplitude=amp,
                phase=0.0,
            ),
            dt=dt,
            frequency_hz=f_notch,
            bandwidth_hz=5.0,
            depth=0.99,
            t_end=t_end,
        )
        assert y.shape[0] > 100, f"too few recorded samples: {y.shape[0]}"
        N = y.shape[0]
        # Steady-state segment (skip warm-up).
        y_ss = y[N // 2:]
        amp_out = 0.5 * (jnp.max(y_ss) - jnp.min(y_ss))
        # Off-notch passthrough should be within ~10 % of the input
        # amplitude.  (The biquad has a modest skirt; allow some slack.)
        assert 0.85 * amp < float(amp_out) < 1.15 * amp, (
            "Off-notch sinusoid should pass through ~unchanged: "
            f"output amp = {float(amp_out)} (input amp = {amp})"
        )


# --------------------------------------------------------------------- #
# DC behaviour
# --------------------------------------------------------------------- #


class TestNotchDC:
    """DC input should pass through with unit gain (after the delay-line warms up)."""

    def test_dc_gain_unity(self):
        dt = 1.0 / 1000.0
        y, _ = _simulate_notch(
            Constant(2.5, name="dc"),
            dt=dt,
            frequency_hz=50.0,
            bandwidth_hz=5.0,
            depth=0.99,
            initial_state=0.0,
            t_end=0.3,
        )
        # After a handful of taps + biquad settling, the output should
        # equal the constant input.
        assert jnp.abs(y[-1] - 2.5) < 5e-3, (
            f"Notch DC gain should be ~1; final y = {float(y[-1])}"
        )


# --------------------------------------------------------------------- #
# Differentiability (jax.grad finite through frequency_hz)
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` is finite w.r.t. ``frequency_hz`` (and friends).

    Bypasses the simulator (whose recorded-signals path is not JAX-
    traceable) and exercises the recursive update / output methods
    directly across a small loop — same pattern as the other T-127
    differentiability tests.
    """

    @staticmethod
    def _make_notch(frequency_hz, bandwidth_hz=5.0, depth=0.99, dt=1.0 / 1000.0):
        block = Notch(
            dt=dt,
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            depth=depth,
            initial_state=0.0,
            name="notch",
        )
        block.initialize(
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            depth=depth,
            initial_state=0.0,
        )
        return block

    @classmethod
    def _notch_step_loss(
        cls, frequency_hz, bandwidth_hz=5.0, depth=0.99, n_steps=20
    ):
        """Drive the notch with a sinusoid and return sum-of-|y|.

        The loss depends smoothly on ``frequency_hz`` through ``omega0``,
        ``cos(omega0)`` and the resulting biquad recursion.
        """
        State = namedtuple("State", ["discrete_state"])
        block = cls._make_notch(frequency_hz, bandwidth_hz, depth)

        xd = block.DiscreteStateType(
            x_prev1=jnp.asarray(0.0),
            x_prev2=jnp.asarray(0.0),
            y_prev1=jnp.asarray(0.0),
            y_prev2=jnp.asarray(0.0),
        )
        params = dict(
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            depth=depth,
        )
        total = jnp.asarray(0.0)
        # Drive with sin(2*pi*40*k*dt) — a frequency near, but not at,
        # the nominal 50 Hz notch so the on-notch derivative is large.
        dt = 1.0 / 1000.0
        for k in range(n_steps):
            t = k * dt
            x = jnp.sin(2.0 * jnp.pi * 40.0 * t)
            state = State(discrete_state=xd)
            y = block._output(jnp.asarray(t), state, x, **params)
            total = total + jnp.abs(y)
            xd = block._update(jnp.asarray(t), state, x, **params)
        return total

    def test_notch_grad_wrt_frequency_finite(self):
        g = jax.grad(self._notch_step_loss, argnums=0)(50.0)
        assert jnp.isfinite(g), f"grad wrt frequency_hz not finite: {g}"

    def test_notch_grad_wrt_frequency_nonzero(self):
        # On-notch behaviour is sensitive to the notch frequency; the
        # derivative should be measurably non-zero around 50 Hz.
        g = jax.grad(self._notch_step_loss, argnums=0)(50.0)
        assert jnp.abs(g) > 0, f"grad wrt frequency_hz should be nonzero; got {g}"

    def test_notch_grad_wrt_bandwidth_finite(self):
        g = jax.grad(self._notch_step_loss, argnums=1)(50.0, 5.0)
        assert jnp.isfinite(g), f"grad wrt bandwidth_hz not finite: {g}"

    def test_notch_grad_wrt_depth_finite(self):
        g = jax.grad(self._notch_step_loss, argnums=2)(50.0, 5.0, 0.99)
        assert jnp.isfinite(g), f"grad wrt depth not finite: {g}"


# --------------------------------------------------------------------- #
# T-005 default-float64 policy
# --------------------------------------------------------------------- #


class TestDefaultFloat64:
    """Discrete state should be the JAX/NumPy default float (float64 with x64)."""

    def test_notch_state_is_default_float(self):
        y, _ = _simulate_notch(
            Constant(1.0, name="dc"),
            dt=1.0 / 1000.0,
            frequency_hz=50.0,
            bandwidth_hz=5.0,
            depth=0.99,
            t_end=0.05,
        )
        assert y.dtype == jnp.asarray(0.0).dtype


# --------------------------------------------------------------------- #
# depth = 0 ⇒ pass-through sanity check
# --------------------------------------------------------------------- #


class TestNotchDepthZero:
    """``depth = 0`` collapses numerator to denominator (unit pass-through)."""

    def test_depth_zero_passes_on_notch_sinusoid(self):
        # With depth = 0 the biquad numerator equals its denominator, so
        # the block is a pure pass-through (H(z) = 1).  Even a sinusoid
        # tuned exactly to the nominal notch frequency should survive
        # unchanged.
        dt = 1.0 / 1000.0
        f = 50.0
        amp = 1.0
        t_end = 0.2
        y, t = _simulate_notch(
            Sine(
                frequency=2.0 * float(jnp.pi) * f,
                amplitude=amp,
                phase=0.0,
            ),
            dt=dt,
            frequency_hz=f,
            bandwidth_hz=5.0,
            depth=0.0,
            t_end=t_end,
        )
        # Skip the first couple of ticks (delay-line warm-up) and check
        # the output matches the input sinusoid.
        N = y.shape[0]
        y_ss = y[N // 2:]
        amp_out = 0.5 * (jnp.max(y_ss) - jnp.min(y_ss))
        assert 0.95 * amp < float(amp_out) < 1.05 * amp, (
            f"depth=0 should be a pass-through; got amp = {float(amp_out)}"
        )
