# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-anti-windup — :class:`PIDController2DOF`.

The phase 1 :class:`PIDController2DOF` integrates the error every tick
without any awareness of actuator saturation: when the actuator is
hard-clamped, classical "integral windup" causes the integrator to grow
unbounded and overshoot when saturation lifts.

This followup adds three new construction kwargs:

* ``output_min``, ``output_max`` — saturation limits applied to the
  control output.
* ``anti_windup_method`` — ``"none"`` (default), ``"back_calc"``, or
  ``"clamping"``.
* ``anti_windup_gain`` — tracking time constant ``Tt`` for back-
  calculation.

Default behaviour (no limits, method = ``"none"``) must be byte-
equivalent to phase 1.

These tests cover:

* Default-off byte-equivalence with phase 1 on a closed-loop step-
  tracking scenario.
* Validation of the ``anti_windup_method`` string.
* Saturation alone (method = ``"none"``, limits set) clips the output
  but lets the integrator wind up — establishes the "no-anti-windup"
  baseline used to measure recovery improvement.
* ``"back_calc"`` keeps the integrator bounded against a saturating
  step disturbance and recovers from saturation faster than the no-
  anti-windup baseline.
* ``"clamping"`` stops the integrator from growing while the
  controller is pushed deeper into saturation.
* Differentiability: ``jax.grad`` w.r.t. ``anti_windup_gain`` is
  finite and non-zero.
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
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_closed_loop(
    *,
    dt=0.05,
    t_end=5.0,
    setpoint=1.0,
    plant_initial=0.0,
    kp=2.0,
    ki=8.0,
    kd=0.0,
    output_min=None,
    output_max=None,
    anti_windup_method="none",
    anti_windup_gain=1.0,
):
    """PI controller in closed loop with an integrator plant ``y' = u``."""
    builder = jaxonomy.DiagramBuilder()
    r = builder.add(Constant(setpoint, name="r"))
    pid_kwargs = dict(
        dt=dt,
        kp=kp,
        ki=ki,
        kd=kd,
        b=1.0,
        c=1.0,
        anti_windup_method=anti_windup_method,
        anti_windup_gain=anti_windup_gain,
        name="pid",
    )
    if output_min is not None:
        pid_kwargs["output_min"] = output_min
    if output_max is not None:
        pid_kwargs["output_max"] = output_max
    pid = builder.add(PIDController2DOF(**pid_kwargs))
    plant = builder.add(Integrator(initial_state=plant_initial, name="plant"))

    builder.connect(r.output_ports[0], pid.input_ports[0])
    builder.connect(plant.output_ports[0], pid.input_ports[1])
    builder.connect(pid.output_ports[0], plant.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    recorded = {
        "y": plant.output_ports[0],
        "u": pid.output_ports[0],
    }
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results


def _make_block(
    *,
    kp=1.0,
    ki=1.0,
    kd=0.0,
    b=1.0,
    c=1.0,
    initial_state=0.0,
    output_min=None,
    output_max=None,
    anti_windup_method="none",
    anti_windup_gain=1.0,
    dt=0.1,
):
    """Construct + run-initialise a PIDController2DOF outside a Diagram."""
    pid_kwargs = dict(
        dt=dt,
        kp=kp,
        ki=ki,
        kd=kd,
        b=b,
        c=c,
        initial_state=initial_state,
        anti_windup_method=anti_windup_method,
        anti_windup_gain=anti_windup_gain,
        name="pid",
    )
    if output_min is not None:
        pid_kwargs["output_min"] = output_min
    if output_max is not None:
        pid_kwargs["output_max"] = output_max
    block = PIDController2DOF(**pid_kwargs)
    block.initialize(
        kp=kp,
        ki=ki,
        kd=kd,
        b=b,
        c=c,
        initial_state=initial_state,
        filter_type="none",
        filter_coefficient=1.0,
        output_min=output_min,
        output_max=output_max,
        anti_windup_method=anti_windup_method,
        anti_windup_gain=anti_windup_gain,
    )
    return block


def _run_steps(block, r, y, *, n_steps, kp, ki, kd, b=1.0, c=1.0,
               output_min=None, output_max=None, anti_windup_gain=1.0,
               initial_integral=0.0):
    """Tick the block ``n_steps`` times against constant ``r``, ``y``.

    Returns ``(integrals, outputs)`` arrays of length ``n_steps``.
    """
    State = namedtuple("State", ["discrete_state"])
    xd0 = block.DiscreteStateType(
        integral=jnp.asarray(initial_integral),
        e_d_prev=jnp.asarray(0.0),
        e_dot_prev=jnp.asarray(0.0),
    )
    state = State(discrete_state=xd0)
    params = dict(kp=kp, ki=ki, kd=kd, b=b, c=c, anti_windup_gain=anti_windup_gain)
    if output_min is not None:
        params["output_min"] = output_min
    if output_max is not None:
        params["output_max"] = output_max

    integrals = []
    outputs = []
    r = jnp.asarray(r)
    y = jnp.asarray(y)
    for _ in range(n_steps):
        u = block._output(jnp.asarray(0.0), state, r, y, **params)
        outputs.append(u)
        new_xd = block._update(jnp.asarray(0.0), state, r, y, **params)
        integrals.append(new_xd.integral)
        state = State(discrete_state=new_xd)
    return jnp.asarray(integrals), jnp.asarray(outputs)


# --------------------------------------------------------------------- #
# Default-off byte-equivalence with phase 1
# --------------------------------------------------------------------- #


class TestDefaultOffByteEquivalence:
    """No saturation limits set → identical to phase 1 behaviour."""

    def test_closed_loop_step_tracking_unchanged(self):
        """Closed-loop tracking trace matches the phase 1 baseline.

        Without ``output_min`` / ``output_max``, the new kwargs default
        to ``anti_windup_method="none"`` and ``anti_windup_gain=1.0``;
        the result must be bit-identical to building the block with no
        anti-windup kwargs at all.
        """
        # Baseline — phase 1 construction (no kwargs).
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=2.0, ki=4.0, kd=0.0, b=1.0, c=1.0, name="pid"
            )
        )
        plant = builder.add(Integrator(initial_state=0.0, name="plant"))
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(plant.output_ports[0], pid.input_ports[1])
        builder.connect(pid.output_ports[0], plant.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        results_phase1 = jaxonomy.simulate(
            diagram, context, (0.0, 5.0),
            recorded_signals={"y": plant.output_ports[0]},
        )
        y_phase1 = results_phase1.outputs["y"]

        # New construction with explicit-default anti-windup kwargs.
        results_new = _simulate_closed_loop(
            kp=2.0, ki=4.0, kd=0.0,
            anti_windup_method="none", anti_windup_gain=1.0,
        )
        y_new = results_new.outputs["y"]

        assert y_phase1.shape == y_new.shape
        assert jnp.allclose(y_phase1, y_new, atol=0.0, rtol=0.0), (
            "default-off path is not byte-equivalent to phase 1; "
            f"max diff = {float(jnp.max(jnp.abs(y_phase1 - y_new)))}"
        )

    def test_per_step_update_byte_equal_with_no_limits(self):
        """``_update`` is bit-equal to phase 1 when no limits are set."""
        # Build the same block with and without explicit kwargs; they
        # exercise the same branch (no anti-windup correction).
        block_default = _make_block(kp=1.0, ki=2.0, kd=0.5)
        block_with_kwargs = _make_block(
            kp=1.0, ki=2.0, kd=0.5,
            anti_windup_method="back_calc", anti_windup_gain=1.0,
        )
        ints_default, us_default = _run_steps(
            block_default, r=1.0, y=0.0,
            n_steps=10, kp=1.0, ki=2.0, kd=0.5,
        )
        # back_calc is a no-op when neither output_min nor output_max
        # is configured (anti-windup is not "active") — must match.
        ints_kwargs, us_kwargs = _run_steps(
            block_with_kwargs, r=1.0, y=0.0,
            n_steps=10, kp=1.0, ki=2.0, kd=0.5,
        )
        assert jnp.allclose(ints_default, ints_kwargs, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_default, us_kwargs, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="anti_windup_method must be one of"):
            PIDController2DOF(
                dt=0.1, kp=1.0, ki=1.0, kd=0.0,
                anti_windup_method="bogus",
            )

    @pytest.mark.parametrize("method", ["none", "back_calc", "clamping"])
    def test_supported_methods_construct(self, method):
        # Smoke test — must not raise.
        PIDController2DOF(
            dt=0.1, kp=1.0, ki=1.0, kd=0.0,
            output_min=0.0, output_max=1.0,
            anti_windup_method=method,
        )


# --------------------------------------------------------------------- #
# Output saturation
# --------------------------------------------------------------------- #


class TestOutputSaturation:
    """``output_min`` / ``output_max`` clip the published control."""

    def test_output_clipped_to_max(self):
        # High kp with a constant setpoint and no measurement → P-term is
        # huge.  output_max=1 must clip every published sample to 1.
        block = _make_block(
            kp=100.0, ki=0.0, kd=0.0,
            output_min=0.0, output_max=1.0,
            anti_windup_method="none",
        )
        _, us = _run_steps(
            block, r=1.0, y=0.0,
            n_steps=5, kp=100.0, ki=0.0, kd=0.0,
            output_min=0.0, output_max=1.0,
        )
        assert jnp.all(us <= 1.0 + 1e-12)
        assert jnp.allclose(us, 1.0, atol=1e-10)

    def test_output_clipped_to_min(self):
        # Negative P term — should be clipped at 0.
        block = _make_block(
            kp=100.0, ki=0.0, kd=0.0,
            output_min=0.0, output_max=1.0,
            anti_windup_method="none",
        )
        _, us = _run_steps(
            block, r=-1.0, y=0.0,
            n_steps=5, kp=100.0, ki=0.0, kd=0.0,
            output_min=0.0, output_max=1.0,
        )
        assert jnp.all(us >= -1e-12)
        assert jnp.allclose(us, 0.0, atol=1e-10)


# --------------------------------------------------------------------- #
# Back-calculation anti-windup
# --------------------------------------------------------------------- #


class TestBackCalculation:
    """Back-calc bounds the integrator and improves recovery."""

    def test_integrator_does_not_wind_up_unboundedly(self):
        """Integrator stays bounded against a persistent saturating step.

        Setup: r = 1, y = 0, output_max = 0.1.  The pure-I controller
        with ki = 5 produces u_unsat that grows without bound under the
        phase 1 update; back-calc should keep the integrator close to
        the value that produces u_sat (≈ 0.1 / ki = 0.02).
        """
        n_steps = 200
        kp, ki, kd = 0.0, 5.0, 0.0
        out_max = 0.1
        # No-anti-windup baseline (method = "none", limits set).
        block_none = _make_block(
            kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            anti_windup_method="none",
        )
        ints_none, _ = _run_steps(
            block_none, r=1.0, y=0.0,
            n_steps=n_steps, kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
        )

        # Back-calc.
        block_bc = _make_block(
            kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            anti_windup_method="back_calc", anti_windup_gain=0.5,
        )
        ints_bc, _ = _run_steps(
            block_bc, r=1.0, y=0.0,
            n_steps=n_steps, kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            anti_windup_gain=0.5,
        )

        # No-anti-windup: integral should grow ~ linearly to a large
        # value (~ n_steps * dt = 200 * 0.1 = 20).
        assert float(ints_none[-1]) > 10.0
        # Back-calc: integral should be bounded near the equilibrium
        # value Ki * I = u_sat → I = 0.1 / 5 = 0.02.
        assert float(ints_bc[-1]) < 1.0, (
            f"back-calc integral grew too large: {float(ints_bc[-1])}"
        )
        # And critically, much smaller than the no-anti-windup case.
        assert float(ints_bc[-1]) < 0.1 * float(ints_none[-1])

    def test_recovery_faster_than_no_anti_windup(self):
        """After saturation lifts, back-calc tracks the setpoint sooner.

        Closed loop with an integrator plant.  Phase 1: r = 2, plant
        starts at -2, output_max = 1.  The actuator saturates at u = 1
        for a while.  Once y catches up the integrator should not have
        accumulated much (back-calc).  Without anti-windup the
        integrator overshoots and y overshoots r.
        """
        # Hard saturation: actuator can move at u_max = 1.
        common = dict(
            dt=0.05, t_end=8.0,
            setpoint=2.0, plant_initial=-2.0,
            kp=1.0, ki=4.0, kd=0.0,
            output_min=-1.0, output_max=1.0,
        )
        # Compare overshoot.
        res_none = _simulate_closed_loop(
            anti_windup_method="none", **common
        )
        res_bc = _simulate_closed_loop(
            anti_windup_method="back_calc", anti_windup_gain=0.2, **common
        )
        y_none = res_none.outputs["y"]
        y_bc = res_bc.outputs["y"]
        # Overshoot = max(y) - setpoint.
        os_none = float(jnp.max(y_none) - 2.0)
        os_bc = float(jnp.max(y_bc) - 2.0)
        # Without anti-windup, integral piles up while saturated and
        # causes a noticeable overshoot.
        assert os_none > 0.1, (
            f"no-anti-windup baseline did not overshoot enough to test "
            f"recovery; got overshoot = {os_none}"
        )
        # Back-calc should reduce overshoot substantially.
        assert os_bc < 0.5 * os_none, (
            f"back-calc did not improve recovery; "
            f"overshoot none={os_none:.3f}, back_calc={os_bc:.3f}"
        )


# --------------------------------------------------------------------- #
# Clamping anti-windup
# --------------------------------------------------------------------- #


class TestClamping:
    """Clamping freezes the integrator while saturation is being pushed."""

    def test_integrator_frozen_when_pushing_into_saturation(self):
        """``integral`` does not grow once the actuator saturates positively."""
        n_steps = 100
        kp, ki, kd = 0.0, 5.0, 0.0
        out_max = 0.1
        block = _make_block(
            kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            anti_windup_method="clamping",
        )
        ints, us = _run_steps(
            block, r=1.0, y=0.0,
            n_steps=n_steps, kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
        )
        # Output should be saturated at out_max from the second sample
        # onward (the first sample has integral=0 so u_unsat=0 < out_max
        # and no clipping is needed).
        assert jnp.allclose(us[1:], out_max, atol=1e-10)
        # Integrator: first tick lifts integral from 0 -> 0.1 (because
        # before update u_unsat = 0 == u_sat = 0, no clamp).  Second
        # tick: u_unsat = 5*0.1 = 0.5 > 0.1 → clamp; integral stays
        # at 0.1 forever after.
        # Allow a couple of free ticks at the boundary, then assert
        # the integrator is stuck.
        tail = ints[-30:]
        assert float(jnp.max(tail) - jnp.min(tail)) < 1e-10, (
            f"integrator should be frozen under clamping; tail = {tail}"
        )
        # And it's bounded — much smaller than the no-anti-windup case
        # (which would be ~ ki * n_steps * dt = 5).
        assert float(ints[-1]) < 1.0

    def test_integrator_releases_when_error_reverses(self):
        """Clamping releases the integrator when the error sign flips."""
        kp, ki, kd = 0.0, 5.0, 0.0
        out_max = 0.1
        block = _make_block(
            kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            anti_windup_method="clamping",
            initial_state=0.5,  # Pre-loaded integral, output saturated.
        )
        # First, push positive — should freeze near 0.5.
        ints_push, _ = _run_steps(
            block, r=1.0, y=0.0,
            n_steps=20, kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            initial_integral=0.5,
        )
        # Tail of push phase is the steady-state integral — frozen.
        i_after_push = float(ints_push[-1])
        assert abs(i_after_push - 0.5) < 1e-10

        # Now reverse: r = -1, y = 0 → e_i = -1, sign(e_i) = -1.
        # u_unsat with integral 0.5 = 5*0.5 = 2.5 (still saturated at
        # 0.1), but sign(sat_excess) = +1 ≠ sign(e_i) = -1, so the
        # integrator MUST update and be pulled back down.
        ints_pull, _ = _run_steps(
            block, r=-1.0, y=0.0,
            n_steps=20, kp=kp, ki=ki, kd=kd,
            output_min=-out_max, output_max=out_max,
            initial_integral=0.5,
        )
        # Integrator should release and decrease substantially from 0.5.
        # It will eventually re-clamp at the lower saturation boundary
        # (where ki*I = output_min and sign(e_i) == sign(sat_excess)),
        # so the final value is bounded but well below the frozen value.
        assert float(ints_pull[-1]) < 0.0, (
            f"clamping should release with reversed error; "
            f"final integral = {float(ints_pull[-1])}"
        )
        # Should have moved by ki * dt * (-1) per step = -0.5 per step,
        # so within ~5 steps the integral is driven to the lower bound
        # and re-clamps; final integral should be much less than 0.4.
        assert float(ints_pull[-1]) < 0.4 * float(ints_push[-1])


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` flows through the new kwargs."""

    @staticmethod
    def _loss(anti_windup_gain, n_steps=200):
        """Closed-loop step-tracking loss as a function of ``Tt``.

        The trajectory must be long enough that the actuator un-saturates
        — only then does the integrator value affect ``u`` (and hence the
        loss).  While the actuator is hard-clamped at ``output_max``,
        ``u`` is independent of ``Tt`` and the loss is locally constant.
        """
        kp, ki, kd = 1.0, 4.0, 0.0
        out_max = 0.5
        block = _make_block(
            kp=kp, ki=ki, kd=kd, dt=0.05,
            output_min=-out_max, output_max=out_max,
            anti_windup_method="back_calc",
            anti_windup_gain=anti_windup_gain,
        )

        State = namedtuple("State", ["discrete_state"])
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0),
            e_d_prev=jnp.asarray(0.0),
            e_dot_prev=jnp.asarray(0.0),
        )
        state = State(discrete_state=xd0)

        # Plant: ``y' = u``, starts far below the setpoint to force a
        # long saturated transient + an over-shoot once it catches up.
        r = jnp.asarray(2.0)
        y = jnp.asarray(-2.0)
        params = dict(
            kp=kp, ki=ki, kd=kd, b=1.0, c=1.0,
            output_min=-out_max, output_max=out_max,
            anti_windup_gain=anti_windup_gain,
        )
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, **params)
            # Integrator plant: y' = u, forward-Euler with dt=0.05.
            y = y + u * 0.05
            new_xd = block._update(jnp.asarray(0.0), state, r, y, **params)
            state = State(discrete_state=new_xd)
            total = total + (r - y) ** 2
        return total

    def test_grad_wrt_anti_windup_gain_finite(self):
        g = jax.grad(self._loss)(0.5)
        assert jnp.isfinite(g), f"grad wrt anti_windup_gain not finite: {g}"

    def test_grad_wrt_anti_windup_gain_nonzero(self):
        g = jax.grad(self._loss)(0.5)
        assert jnp.abs(g) > 0, (
            f"grad wrt anti_windup_gain should be non-zero (back-calc "
            f"depends on Tt while saturated); got {g}"
        )
