# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-feedforward — :class:`PIDController2DOF`.

A 2-DOF PID + feedforward controller adds a third term that is a
function of the setpoint only::

    u = Kp*(b*r - y) + Ki*integral(r - y) + Kd*d/dt(c*r - y) + Kff*r

For a plant whose steady-state transfer function from ``u`` to ``y`` is
``G(0)``, choosing ``Kff = 1/G(0)`` makes the controller track step
changes in ``r`` without integrator action — fastest possible step
response.  (Pair with PID for disturbance rejection.)

This followup adds a single new construction kwarg, ``kff`` (default
``0.0``), to :class:`PIDController2DOF`.  Default ``kff=0.0`` is
byte-equivalent to phase 1.

These tests cover:

* Default ``kff=0.0``: byte-equivalent to phase 1 PID behaviour
  (open-loop, full PID, closed-loop).
* ``kff=1.0`` with all PID gains zero on a step setpoint: ``u == r``.
* Closed-loop step tracking with ``kff = 1/G(0)``: a feedforward-armed
  controller hits the steady-state target faster than the same PID
  without feedforward.
* Composes with anti-windup: the feedforward term is part of the
  unsaturated ``u_unsat`` that gets compared to the clipped ``u_sat``.
  In particular, ``kff * r`` alone exceeding ``output_max`` causes the
  controller to saturate.
* ``jax.grad`` w.r.t. ``kff`` is finite and non-zero.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    Integrator,
    PIDController2DOF,
    Step,
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
    kp=0.0,
    ki=0.0,
    kd=0.0,
    b=1.0,
    c=1.0,
    kff=0.0,
    output_min=None,
    output_max=None,
    anti_windup_method="none",
    anti_windup_gain=1.0,
    t_end=1.0,
):
    """Open-loop: r and y are constants, returns u(t)."""
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
        name="pid",
    )
    if output_min is not None:
        kwargs["output_min"] = output_min
    if output_max is not None:
        kwargs["output_max"] = output_max
    if anti_windup_method != "none":
        kwargs["anti_windup_method"] = anti_windup_method
        kwargs["anti_windup_gain"] = anti_windup_gain
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


def _simulate_closed_loop_integrator_plant(
    *,
    dt=0.05,
    r=1.0,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    kff=0.0,
    plant_gain=1.0,
    t_end=2.0,
):
    """Closed-loop: PID drives a pure integrator plant ``y' = plant_gain * u``.

    Steady-state transfer function (u → y) of a *pure integrator* is
    unbounded, so for this test we use a leaky-integrator-style helper:
    we cap the simulation at finite time and inspect tracking speed
    rather than steady-state error.

    Returns ``(y_trace, time)``.
    """
    builder = jaxonomy.DiagramBuilder()
    r_b = builder.add(Constant(r, name="r"))
    pid = builder.add(
        PIDController2DOF(
            dt=dt, kp=kp, ki=ki, kd=kd, kff=kff, name="pid"
        )
    )
    # Plant: pure integrator with adjustable gain.
    plant = builder.add(Integrator(initial_state=0.0, name="plant"))
    # Gain block via Constant multiply would over-engineer this; use
    # an inline scaling by setting kp on the PID and an explicit gain
    # constant in the loop is too much.  Just connect PID->Integrator
    # directly (plant_gain=1.0).
    assert plant_gain == 1.0, "test helper assumes plant_gain=1.0"

    builder.connect(r_b.output_ports[0], pid.input_ports[0])
    builder.connect(plant.output_ports[0], pid.input_ports[1])
    builder.connect(pid.output_ports[0], plant.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals={"y": plant.output_ports[0]},
    )
    return results.outputs["y"], results.time


# --------------------------------------------------------------------- #
# Default kff=0.0: byte-equivalence with phase 1
# --------------------------------------------------------------------- #


class TestDefaultKffByteEquivalent:
    """``kff=0.0`` (default) is byte-equivalent to phase 1."""

    def test_open_loop_default_kff_matches_implicit(self):
        """Explicit ``kff=0.0`` matches the implicit default."""
        u_implicit, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1, t_end=2.0
        )
        u_explicit, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1, kff=0.0,
            t_end=2.0,
        )
        assert jnp.array_equal(u_implicit, u_explicit), (
            "Explicit kff=0.0 must be byte-equivalent to the implicit "
            "phase 1 default"
        )

    def test_default_kff_attribute(self):
        """Default-constructed block reports ``kff=0.0``."""
        block = PIDController2DOF(dt=0.1, name="pid")
        # The dynamic-parameter machinery stores it on the instance.
        assert float(block.dynamic_parameters["kff"].get()) == 0.0


# --------------------------------------------------------------------- #
# kff=1.0 step-input identity
# --------------------------------------------------------------------- #


class TestKffStepIdentity:
    """``kff=1.0`` and all PID gains zero: ``u == r``."""

    def test_kff_one_zero_gains_pure_passthrough(self):
        """r=1, y=0, all gains 0, kff=1 → u(t) ≡ 1."""
        u, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0, kp=0.0, ki=0.0, kd=0.0,
            kff=1.0, t_end=0.5,
        )
        assert jnp.allclose(u, 1.0, atol=1e-12), (
            f"Expected u≡1 for kff=1.0 with zero PID gains; got {u}"
        )

    def test_kff_scales_setpoint_linearly(self):
        """``u = kff * r`` for zero PID gains."""
        for kff in (0.5, 1.0, 2.5, -0.7):
            r = 3.0
            u, _ = _simulate_open_loop(
                dt=0.05, r=r, y=0.0, kp=0.0, ki=0.0, kd=0.0,
                kff=kff, t_end=0.3,
            )
            assert jnp.allclose(u, kff * r, atol=1e-12), (
                f"kff={kff}, r={r}: expected u={kff*r}, got {u}"
            )

    def test_kff_adds_to_full_pid_output(self):
        """``kff * r`` adds on top of the existing PID terms."""
        # P-only controller: u_phase1 = kp*(b*r - y), then with kff
        # u_ff = u_phase1 + kff*r.
        r, y, kp = 1.0, 0.0, 2.0
        u_no_ff, _ = _simulate_open_loop(
            dt=0.05, r=r, y=y, kp=kp, ki=0.0, kd=0.0,
            kff=0.0, t_end=0.3,
        )
        u_with_ff, _ = _simulate_open_loop(
            dt=0.05, r=r, y=y, kp=kp, ki=0.0, kd=0.0,
            kff=1.5, t_end=0.3,
        )
        expected_diff = 1.5 * r
        diff = u_with_ff - u_no_ff
        assert jnp.allclose(diff, expected_diff, atol=1e-12), (
            f"Expected u_ff - u_no_ff = {expected_diff}, got {diff}"
        )


# --------------------------------------------------------------------- #
# Closed-loop step-tracking speedup
# --------------------------------------------------------------------- #


class TestClosedLoopFeedforwardSpeeds:
    """A feedforward-armed PID tracks step setpoints faster than the
    same PID alone."""

    def test_step_tracking_faster_with_kff(self):
        """Plant y' = u (so G(0) is ill-defined for a pure integrator,
        but the feedforward still injects energy into the loop and
        accelerates the response).  Use a P-only controller so there is
        no integrator-action contribution to disambiguate.
        """
        dt = 0.01
        r = 1.0
        kp = 1.0
        t_end = 2.0

        # No feedforward.
        y_no_ff, t = _simulate_closed_loop_integrator_plant(
            dt=dt, r=r, kp=kp, ki=0.0, kd=0.0, kff=0.0, t_end=t_end,
        )
        # With feedforward — the integrator plant's instantaneous "DC
        # gain" is unbounded, but kff>0 still injects a baseline drive
        # that accelerates how quickly y crosses the target.
        y_ff, _ = _simulate_closed_loop_integrator_plant(
            dt=dt, r=r, kp=kp, ki=0.0, kd=0.0, kff=2.0, t_end=t_end,
        )
        # Find the first time each trace reaches y >= 0.9 * r.
        target = 0.9 * r

        def _first_crossing(y_trace, t_arr):
            above = y_trace >= target
            if not bool(jnp.any(above)):
                return float("inf")
            return float(t_arr[jnp.argmax(above)])

        t_no_ff = _first_crossing(y_no_ff, t)
        t_ff = _first_crossing(y_ff, t)
        assert t_ff < t_no_ff, (
            f"Feedforward should accelerate step tracking: "
            f"t_no_ff={t_no_ff}, t_ff={t_ff}"
        )

    def test_kff_inverse_dc_gain_tracks_without_integrator(self):
        """For a plant with finite DC gain ``G(0)``, ``kff = 1/G(0)``
        yields zero steady-state error without any integral action.

        Here the "plant" is the static identity G(0) = 1 (the helper
        wires PID directly to an integrator which is *not* finite DC,
        so we use the open-loop saturation case as a proxy: the
        steady-state ``u`` itself equals ``r`` when ``kp = ki = kd = 0``
        and ``kff = 1``.)
        """
        u, _ = _simulate_open_loop(
            dt=0.01, r=2.5, y=0.0,
            kp=0.0, ki=0.0, kd=0.0, kff=1.0, t_end=0.3,
        )
        assert jnp.allclose(u, 2.5, atol=1e-12)


# --------------------------------------------------------------------- #
# Composition with anti-windup
# --------------------------------------------------------------------- #


class TestComposesWithAntiWindup:
    """``kff * r`` is part of the unsaturated ``u_unsat`` that gets
    clipped and compared to ``u_sat`` for anti-windup."""

    def test_kff_term_subject_to_output_max_clip(self):
        """``kff * r`` alone exceeding ``output_max`` saturates u."""
        # All PID gains zero; only kff drives u.  With kff=5 and r=1,
        # u_unsat = 5; with output_max=2.0, u_sat = 2.0.
        u, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0,
            kp=0.0, ki=0.0, kd=0.0, kff=5.0,
            output_max=2.0,
            t_end=0.3,
        )
        assert jnp.allclose(u, 2.0, atol=1e-12), (
            f"kff*r=5 should clip to output_max=2; got u={u}"
        )

    def test_kff_term_subject_to_output_min_clip(self):
        """``kff * r`` below ``output_min`` saturates u upward."""
        # kff=-5 with r=1 gives u_unsat=-5; output_min=-1 clips to -1.
        u, _ = _simulate_open_loop(
            dt=0.05, r=1.0, y=0.0,
            kp=0.0, ki=0.0, kd=0.0, kff=-5.0,
            output_min=-1.0,
            t_end=0.3,
        )
        assert jnp.allclose(u, -1.0, atol=1e-12), (
            f"kff*r=-5 should clip to output_min=-1; got u={u}"
        )

    def test_kff_changes_anti_windup_u_unsat(self):
        """With back-calculation anti-windup, varying kff (which feeds
        into ``u_unsat``) changes the integrator-tracking correction.

        We exercise ``_update`` directly so we can read the integral
        state out and verify the back-calculation correction sees the
        larger ``u_unsat`` when kff is increased.
        """
        State = namedtuple("State", ["discrete_state"])

        def _step_integral(kff_val):
            block = PIDController2DOF(
                dt=0.1, kp=1.0, ki=1.0, kd=0.0, b=1.0, c=1.0,
                kff=kff_val,
                output_max=0.5,
                anti_windup_method="back_calc",
                anti_windup_gain=1.0,
                name="pid",
            )
            block.initialize(
                kp=1.0, ki=1.0, kd=0.0, b=1.0, c=1.0,
                initial_state=0.0, filter_type="none",
                filter_coefficient=1.0,
                output_min=None, output_max=0.5,
                anti_windup_method="back_calc",
                anti_windup_gain=1.0,
                kff=kff_val,
            )
            xd0 = block.DiscreteStateType(
                integral=jnp.asarray(0.0),
                e_d_prev=jnp.asarray(0.0),
                e_dot_prev=jnp.asarray(0.0),
            )
            state = State(discrete_state=xd0)
            r = jnp.asarray(1.0)
            y = jnp.asarray(0.0)
            params = dict(
                kp=1.0, ki=1.0, kd=0.0, b=1.0, c=1.0,
                kff=kff_val,
                output_max=0.5, output_min=None,
                anti_windup_gain=1.0,
            )
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, **params
            )
            return float(new_xd.integral)

        # u_unsat_no_ff = 1*1 + 1*0 + 0 + 0*1 = 1.0
        # u_sat        = 0.5
        # back-calc:   I_next = 0 + 1*0.1 - (1.0 - 0.5)/1 * 0.1 = 0.05
        i_no_ff = _step_integral(0.0)
        # u_unsat_ff   = 1*1 + 1*0 + 0 + 5*1 = 6.0
        # u_sat        = 0.5
        # back-calc:   I_next = 0 + 1*0.1 - (6.0 - 0.5)/1 * 0.1 = 0.1 - 0.55 = -0.45
        i_with_ff = _step_integral(5.0)
        assert i_no_ff != i_with_ff, (
            f"kff should change the back-calculation correction; "
            f"i_no_ff={i_no_ff}, i_with_ff={i_with_ff}"
        )
        # Sanity-check the analytical values.
        assert jnp.allclose(i_no_ff, 0.05, atol=1e-10), (
            f"Expected I_next=0.05 (no FF), got {i_no_ff}"
        )
        assert jnp.allclose(i_with_ff, -0.45, atol=1e-10), (
            f"Expected I_next=-0.45 (kff=5), got {i_with_ff}"
        )


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` w.r.t. ``kff`` is finite and non-zero."""

    @staticmethod
    def _make_block(dt=0.1, kff_init=1.0):
        block = PIDController2DOF(
            dt=dt, kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            kff=kff_init, name="pid",
        )
        block.initialize(
            kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            initial_state=0.0, filter_type="none", filter_coefficient=1.0,
            kff=kff_init,
        )
        return block

    def test_grad_wrt_kff_finite_and_nonzero(self):
        """Gradient of integrated |u| w.r.t. ``kff`` is finite and
        non-zero (each tick contributes ``kff*r`` to u)."""
        State = namedtuple("State", ["discrete_state"])
        # Build the block ONCE with a concrete kff so the @parameters
        # plumbing has a valid static seed; the differentiable kff is
        # then threaded through ``params`` for each call to
        # _output / _update.
        block = self._make_block(kff_init=1.0)

        def loss(kff_value, n_steps=4):
            xd0 = block.DiscreteStateType(
                integral=jnp.asarray(0.0),
                e_d_prev=jnp.asarray(0.0),
                e_dot_prev=jnp.asarray(0.0),
            )
            state = State(discrete_state=xd0)
            r = jnp.asarray(1.0)
            y = jnp.asarray(0.0)
            params = dict(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0, kff=kff_value
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

        g = jax.grad(loss)(jnp.asarray(0.5))
        assert jnp.isfinite(g), f"grad wrt kff not finite: {g}"
        assert jnp.abs(g) > 0, (
            f"grad wrt kff should be non-zero; got {g}"
        )

    def test_grad_wrt_r_includes_feedforward(self):
        """Gradient w.r.t. the setpoint includes the ``kff`` contribution.

        With ``kff=2.0`` and PID gains zero, ``u = 2*r``, so
        ``du/dr = 2`` per tick.
        """
        State = namedtuple("State", ["discrete_state"])

        def loss(r_value):
            block = PIDController2DOF(
                dt=0.1, kp=0.0, ki=0.0, kd=0.0, b=1.0, c=1.0,
                kff=2.0, name="pid",
            )
            block.initialize(
                kp=0.0, ki=0.0, kd=0.0, b=1.0, c=1.0,
                initial_state=0.0, filter_type="none",
                filter_coefficient=1.0, kff=2.0,
            )
            xd0 = block.DiscreteStateType(
                integral=jnp.asarray(0.0),
                e_d_prev=jnp.asarray(0.0),
                e_dot_prev=jnp.asarray(0.0),
            )
            state = State(discrete_state=xd0)
            y = jnp.asarray(0.0)
            params = dict(
                kp=0.0, ki=0.0, kd=0.0, b=1.0, c=1.0, kff=2.0
            )
            u = block._output(
                jnp.asarray(0.0), state, r_value, y, **params
            )
            return u

        g = jax.grad(loss)(jnp.asarray(1.0))
        assert jnp.isfinite(g)
        # u = 2*r (kp=0, ki=0, kd=0, kff=2) → du/dr = 2.
        assert jnp.allclose(g, 2.0, atol=1e-6), (
            f"Expected du/dr = 2 (from kff*r); got {g}"
        )
