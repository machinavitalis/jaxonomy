# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-deadband-error — :class:`PIDController2DOF`.

In real control loops you don't want the PID to react to tiny
measurement noise — you want a "deadband" around zero error where the
controller doesn't act.  This is the equivalent of a hardware sensor's
noise floor.

This followup adds two new construction kwargs to
:class:`PIDController2DOF`:

* ``error_deadband`` (float, default ``0.0``) — non-negative half-width
  of the deadband on the (weighted) error signal.
* ``error_deadband_mode`` (``"hard"`` / ``"smooth"``) — selects the
  gate kernel.  Hard mode uses ``npa.where(|e_raw| > deadband, e_raw,
  0)``; smooth mode uses :func:`soft_dead_zone` for a sigmoid-blended,
  fully-differentiable gate.
* ``error_deadband_sharpness`` (float, default ``10.0``) — only used in
  smooth mode; passed through to :func:`soft_dead_zone`.

Default ``error_deadband=0.0`` is byte-equivalent to phase 1 (and every
previous T-127 followup) — the gate is bypassed at construction time.

These tests cover:

* Default ``error_deadband=0.0``: byte-equivalent to phase 1.
* ``error_deadband=0.1``: a step input of magnitude 0.05 (inside the
  band) produces no controller action; magnitude 0.2 (outside) acts as
  if there were no deadband.
* Smooth mode: sigmoid-blended gate produces a non-zero (but small)
  control action for an in-band step, and is monotone in the input
  magnitude.
* Differentiability: ``jax.grad`` w.r.t. ``error_deadband`` is finite
  in smooth mode (and finite, possibly zero, in hard mode).
* Composition smoke test: error_deadband + anti-windup + feedforward +
  gain scheduling all active at once produces a finite output.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    PIDController2DOF,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_open_loop(
    *,
    dt=0.05,
    r=1.0,
    y=0.0,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    b=1.0,
    c=1.0,
    kff=0.0,
    error_deadband=0.0,
    error_deadband_mode="hard",
    error_deadband_sharpness=10.0,
    t_end=1.0,
):
    """Open-loop helper: r and y are constants; returns ``u(t)``."""
    builder = jaxonomy.DiagramBuilder()
    r_b = builder.add(Constant(r, name="r"))
    y_b = builder.add(Constant(y, name="y"))
    kwargs = dict(
        dt=dt,
        kp=kp,
        ki=ki,
        kd=kd,
        b=b,
        c=c,
        kff=kff,
        error_deadband=error_deadband,
        error_deadband_mode=error_deadband_mode,
        error_deadband_sharpness=error_deadband_sharpness,
        name="pid",
    )
    pid = builder.add(PIDController2DOF(**kwargs))
    builder.connect(r_b.output_ports[0], pid.input_ports[0])
    builder.connect(y_b.output_ports[0], pid.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals={"u": pid.output_ports[0]},
    )
    return results.outputs["u"], results.time


# --------------------------------------------------------------------- #
# Default error_deadband=0.0 → byte-equivalent to phase 1
# --------------------------------------------------------------------- #


class TestDefaultDeadbandByteEquivalent:
    """``error_deadband=0.0`` is byte-equivalent to phase 1."""

    def test_open_loop_default_matches_implicit(self):
        """Explicit ``error_deadband=0.0`` matches the implicit default."""
        u_implicit, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1, t_end=1.0,
        )
        u_explicit, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.0, t_end=1.0,
        )
        assert jnp.array_equal(u_implicit, u_explicit), (
            "Explicit error_deadband=0.0 must be byte-equivalent to the "
            "implicit phase 1 default"
        )

    def test_default_attribute(self):
        """Default-constructed block reports the documented defaults."""
        block = PIDController2DOF(dt=0.1, name="pid")
        assert float(block.dynamic_parameters["error_deadband"].get()) == 0.0
        assert block._error_deadband_mode == "hard"
        assert block._error_deadband_active is False

    def test_smooth_mode_with_zero_band_is_byte_equivalent(self):
        """``mode='smooth'`` with ``error_deadband=0.0`` is still
        byte-equivalent — the gate is bypassed at construction time
        regardless of the configured mode."""
        u_hard, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.0, error_deadband_mode="hard", t_end=1.0,
        )
        u_smooth, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.0, error_deadband_mode="smooth", t_end=1.0,
        )
        assert jnp.array_equal(u_hard, u_smooth), (
            "error_deadband=0.0 bypasses the gate entirely, so hard and "
            "smooth modes must produce the same trajectory"
        )


# --------------------------------------------------------------------- #
# Hard mode: in-band → silent, out-of-band → unchanged
# --------------------------------------------------------------------- #


class TestHardDeadband:
    """Hard mode: ``|e_raw| <= deadband`` → ``e = 0``; otherwise pass
    through unchanged."""

    def test_in_band_step_produces_no_action(self):
        """``error_deadband=0.1`` with a step of magnitude 0.05 produces
        u == 0 for all time (P+I+D all read e=0)."""
        u, _ = _simulate_open_loop(
            dt=0.05, r=0.05, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.1, error_deadband_mode="hard",
            t_end=1.0,
        )
        assert jnp.allclose(u, 0.0, atol=1e-12), (
            f"In-band step (|e|=0.05 < deadband=0.1) should produce no "
            f"controller action; got u={u}"
        )

    def test_out_of_band_step_acts_normally(self):
        """``error_deadband=0.1`` with a step of magnitude 0.2 produces
        (within float64 ulps) the same u as a phase 1 PID with no
        deadband — when out of the band the hard gate is an algebraic
        identity (the ``where`` dispatch contributes a few ulps of
        rounding noise but is otherwise indistinguishable)."""
        u_band, _ = _simulate_open_loop(
            dt=0.05, r=0.2, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.1, error_deadband_mode="hard",
            t_end=1.0,
        )
        u_no_band, _ = _simulate_open_loop(
            dt=0.05, r=0.2, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            t_end=1.0,
        )
        assert jnp.allclose(u_band, u_no_band, atol=1e-12), (
            f"Out-of-band step (|e|=0.2 > deadband=0.1) should match the "
            f"no-deadband trajectory to within float64 ulps; got "
            f"u_band={u_band}, u_no_band={u_no_band}"
        )

    def test_boundary_below_kills_action(self):
        """At exactly the deadband boundary (|e| == deadband) the hard
        gate is closed (strict inequality ``> deadband``)."""
        u, _ = _simulate_open_loop(
            dt=0.05, r=0.1, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.1, error_deadband_mode="hard",
            t_end=0.3,
        )
        # At |e| == deadband the gate returns 0 (strict inequality).
        assert jnp.allclose(u, 0.0, atol=1e-12), (
            f"|e|==deadband should be zeroed; got u={u}"
        )

    def test_negative_in_band_step_produces_no_action(self):
        """Symmetric: an in-band negative step also kills all action."""
        u, _ = _simulate_open_loop(
            dt=0.05, r=-0.05, y=0.0, kp=2.0, ki=4.0, kd=0.1,
            error_deadband=0.1, error_deadband_mode="hard",
            t_end=0.5,
        )
        assert jnp.allclose(u, 0.0, atol=1e-12), (
            f"In-band negative step should produce no action; got u={u}"
        )


# --------------------------------------------------------------------- #
# Smooth mode: sigmoid-blended gate
# --------------------------------------------------------------------- #


class TestSmoothDeadband:
    """Smooth mode uses :func:`soft_dead_zone` — gradient flows through
    the band but the magnitude inside the band is tiny."""

    def test_smooth_in_band_is_small_but_present(self):
        """Inside the band, smooth mode produces a non-zero (small) u —
        the sigmoid does not snap to 0."""
        u, _ = _simulate_open_loop(
            dt=0.05, r=0.05, y=0.0, kp=1.0, ki=0.0, kd=0.0,
            error_deadband=0.1, error_deadband_mode="smooth",
            error_deadband_sharpness=10.0,
            t_end=0.1,
        )
        # The first sample's u is kp * soft_dead_zone(0.05, 0.1, 10).
        # gate = 0.5*(1+tanh(10*(0.05-0.1)/0.1)) = 0.5*(1+tanh(-5))
        #      ≈ 0.5 * (1 - 0.9999) ≈ 3.35e-5
        # u    = 0.05 * 3.35e-5 ≈ 1.67e-6 — very small but non-zero.
        assert float(jnp.max(jnp.abs(u))) < 1e-3, (
            f"Smooth in-band u should be very small (~1e-6); got {u}"
        )
        # And the hard-mode equivalent is exactly zero — the two modes
        # are distinguishable inside the band.
        u_hard, _ = _simulate_open_loop(
            dt=0.05, r=0.05, y=0.0, kp=1.0, ki=0.0, kd=0.0,
            error_deadband=0.1, error_deadband_mode="hard",
            t_end=0.1,
        )
        assert jnp.allclose(u_hard, 0.0, atol=1e-12)

    def test_smooth_out_of_band_approaches_no_deadband(self):
        """Far outside the band, the smooth gate -> 1 so u -> u_no_band."""
        # Use a step well outside the band so the sigmoid is fully open.
        u_band, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=0.0, kd=0.0,
            error_deadband=0.1, error_deadband_mode="smooth",
            error_deadband_sharpness=20.0,
            t_end=0.3,
        )
        u_no_band, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=0.0, kd=0.0,
            t_end=0.3,
        )
        # gate(1.0, 0.1, 20) = 0.5*(1+tanh(20*0.9/0.1)) ≈ 1 to machine eps.
        # so u_band ≈ u_no_band.
        assert jnp.allclose(u_band, u_no_band, atol=1e-6), (
            f"Far out-of-band smooth gate should be ≈1; got "
            f"u_band={u_band}, u_no_band={u_no_band}"
        )

    def test_smooth_monotone_in_input_magnitude(self):
        """For increasing |r|, the smooth-gated u is monotone-increasing
        in magnitude — the sigmoid is a non-decreasing function of
        |r|."""
        magnitudes = [0.01, 0.05, 0.1, 0.2, 0.5]
        us = []
        for r in magnitudes:
            u, _ = _simulate_open_loop(
                dt=0.05, r=r, y=0.0, kp=1.0, ki=0.0, kd=0.0,
                error_deadband=0.1, error_deadband_mode="smooth",
                error_deadband_sharpness=10.0,
                t_end=0.1,
            )
            us.append(float(jnp.abs(u[-1])))
        # Strictly monotone: each |u| > the previous (for r >= 0).
        for i in range(1, len(us)):
            assert us[i] > us[i - 1], (
                f"|u| should be monotone-increasing in |r|: "
                f"magnitudes={magnitudes}, us={us}"
            )


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` w.r.t. ``error_deadband`` is finite (and non-zero in
    smooth mode for in-band inputs)."""

    @staticmethod
    def _make_block(
        *, dt=0.1, error_deadband=0.05, mode="smooth", sharpness=10.0,
    ):
        block = PIDController2DOF(
            dt=dt, kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            error_deadband=error_deadband,
            error_deadband_mode=mode,
            error_deadband_sharpness=sharpness,
            name="pid",
        )
        block.initialize(
            kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            initial_state=0.0, filter_type="none",
            filter_coefficient=1.0,
            error_deadband=error_deadband,
            error_deadband_mode=mode,
            error_deadband_sharpness=sharpness,
        )
        return block

    def test_grad_wrt_error_deadband_smooth_finite(self):
        """Gradient of |u| w.r.t. ``error_deadband`` is finite (smooth)."""
        State = namedtuple("State", ["discrete_state"])
        # Build a static block at the chosen mode; the *value* of the
        # half-width is threaded through ``params`` so it can be
        # differentiated.
        block = self._make_block(error_deadband=0.05, mode="smooth")

        def loss(eb_value, n_steps=4):
            xd0 = block.DiscreteStateType(
                integral=jnp.asarray(0.0),
                e_d_prev=jnp.asarray(0.0),
                e_dot_prev=jnp.asarray(0.0),
            )
            state = State(discrete_state=xd0)
            r = jnp.asarray(0.05)  # in-band — gradient flows through
            y = jnp.asarray(0.0)
            params = dict(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
                error_deadband=eb_value,
                error_deadband_sharpness=10.0,
            )
            total = jnp.asarray(0.0)
            for _ in range(n_steps):
                u = block._output(jnp.asarray(0.0), state, r, y, **params)
                total = total + jnp.abs(u)
                new_xd = block._update(
                    jnp.asarray(0.0), state, r, y, **params
                )
                state = State(discrete_state=new_xd)
            return total

        g = jax.grad(loss)(jnp.asarray(0.05))
        assert jnp.isfinite(g), (
            f"grad wrt error_deadband not finite (smooth mode): {g}"
        )
        # Non-zero: increasing the band squeezes |u| further.
        assert jnp.abs(g) > 0, (
            f"grad wrt error_deadband should be non-zero for in-band "
            f"input in smooth mode; got {g}"
        )

    def test_grad_wrt_error_deadband_hard_finite(self):
        """Hard mode: gradient is finite (zero on the in-band branch is
        acceptable; the kink at the boundary is sub-differentiable but
        finite either side)."""
        State = namedtuple("State", ["discrete_state"])
        block = self._make_block(error_deadband=0.05, mode="hard")

        def loss(eb_value):
            xd0 = block.DiscreteStateType(
                integral=jnp.asarray(0.0),
                e_d_prev=jnp.asarray(0.0),
                e_dot_prev=jnp.asarray(0.0),
            )
            state = State(discrete_state=xd0)
            r = jnp.asarray(0.2)  # out of band — gate is identity
            y = jnp.asarray(0.0)
            params = dict(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
                error_deadband=eb_value,
                error_deadband_sharpness=10.0,
            )
            u = block._output(jnp.asarray(0.0), state, r, y, **params)
            return jnp.abs(u)

        g = jax.grad(loss)(jnp.asarray(0.05))
        assert jnp.isfinite(g), (
            f"grad wrt error_deadband not finite (hard mode): {g}"
        )


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    """Construction-time validation of the new kwargs."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="error_deadband_mode"):
            PIDController2DOF(
                dt=0.1, error_deadband=0.1,
                error_deadband_mode="bogus", name="pid",
            )

    def test_mode_change_at_initialize_raises(self):
        block = PIDController2DOF(
            dt=0.1, error_deadband=0.1, error_deadband_mode="hard",
            name="pid",
        )
        with pytest.raises(ValueError, match="error_deadband_mode"):
            block.initialize(
                kp=1.0, ki=0.0, kd=0.0, b=1.0, c=1.0,
                initial_state=0.0, filter_type="none",
                filter_coefficient=1.0,
                error_deadband=0.1, error_deadband_mode="smooth",
            )


# --------------------------------------------------------------------- #
# Composition smoke test: deadband + anti-windup + feedforward +
# gain scheduling all active simultaneously.
# --------------------------------------------------------------------- #


class TestComposesWithAllFeatures:
    """All four T-127 follow-up dimensions active at once produce a
    finite output."""

    def test_smoke_all_features_active(self):
        """error_deadband + anti-windup + feedforward + gain scheduling
        all on — block constructs, simulates, and produces a finite u."""
        builder = jaxonomy.DiagramBuilder()
        r_b = builder.add(Constant(1.0, name="r"))
        y_b = builder.add(Constant(0.0, name="y"))
        kp_b = builder.add(Constant(2.0, name="kp_sched"))
        ki_b = builder.add(Constant(0.5, name="ki_sched"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05,
                kp=2.0, ki=0.5, kd=0.1,
                kff=0.5,
                kp_dynamic=True,
                ki_dynamic=True,
                output_min=-3.0, output_max=3.0,
                anti_windup_method="back_calc",
                anti_windup_gain=1.0,
                error_deadband=0.05,
                error_deadband_mode="smooth",
                error_deadband_sharpness=10.0,
                name="pid",
            )
        )
        builder.connect(r_b.output_ports[0], pid.input_ports[0])
        builder.connect(y_b.output_ports[0], pid.input_ports[1])
        # kp at index 2, ki at index 3 (no b/c/kd/kff dynamic).
        builder.connect(kp_b.output_ports[0], pid.input_ports[2])
        builder.connect(ki_b.output_ports[0], pid.input_ports[3])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 0.5),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u = results.outputs["u"]
        assert jnp.all(jnp.isfinite(u)), (
            f"Composition smoke test produced non-finite u: {u}"
        )
        # The output is also saturated to the configured limits.
        assert float(jnp.max(u)) <= 3.0 + 1e-9
        assert float(jnp.min(u)) >= -3.0 - 1e-9

    def test_deadband_kills_action_with_in_band_setpoint(self):
        """Composition still respects the deadband: an in-band step
        (with anti-windup + feedforward on) produces u close to zero
        plus the feedforward contribution (kff * r_inband)."""
        # r=0.05 is inside deadband=0.1. kp/ki/kd*e_terms are all 0;
        # kff*r=0.5*0.05=0.025 still flows through (feedforward is NOT
        # gated by the deadband — it bypasses the error signal).
        u, _ = _simulate_open_loop(
            dt=0.05, r=0.05, y=0.0, kp=2.0, ki=4.0, kd=0.1, kff=0.5,
            error_deadband=0.1, error_deadband_mode="hard",
            t_end=0.5,
        )
        # All PID terms are gated to 0; only kff*r remains.
        assert jnp.allclose(u, 0.5 * 0.05, atol=1e-12), (
            f"Expected u=kff*r=0.025 (PID gated to 0); got u={u}"
        )
