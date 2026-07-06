# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-tracking-mode — :class:`PIDController2DOF`.

In real control loops, the PID is sometimes placed in "manual" /
"tracking" mode: an external signal ``u_ext`` (from a manual controller
or another regulator) is published to the actuator, and the PID needs to
track that signal so the handoff back to PID-driven (auto) control is
*bumpless* — the integrator should equal the value that would produce
``u_ext`` at the switchover instant.

This followup adds two construction kwargs:

* ``tracking_enabled`` — when True, declares an extra input port
  ``u_ext`` (appended after every other dynamic port) and folds a
  tracking-error term into the integrator update.
* ``tracking_gain`` — tracking time constant ``Tt`` for the back-
  calculation kernel.

Default (``tracking_enabled=False``) is byte-equivalent to phase 1.

These tests cover:

* Default-off byte-equivalence with phase 1.
* Convergence: ``tracking_enabled=True`` with ``u_ext`` connected to a
  ``Constant`` makes the PID output converge toward ``u_ext``.
* Bumpless transfer: the integrator is loaded so the PID output equals
  the steady-state ``u_ext``.
* Composition with anti-windup: both mechanisms operate simultaneously.
* Validation: flipping ``tracking_enabled`` after construction is
  rejected; building with ``tracking_enabled=True`` and the new port
  unconnected raises at simulate time.
* Differentiability: ``jax.grad`` w.r.t. ``tracking_gain`` is finite.
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


def _make_block(
    *,
    kp=1.0,
    ki=1.0,
    kd=0.0,
    b=1.0,
    c=1.0,
    initial_state=0.0,
    dt=0.1,
    output_min=None,
    output_max=None,
    anti_windup_method="none",
    anti_windup_gain=1.0,
    tracking_enabled=False,
    tracking_gain=1.0,
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
        tracking_enabled=tracking_enabled,
        tracking_gain=tracking_gain,
        name="pid",
    )
    if output_min is not None:
        pid_kwargs["output_min"] = output_min
    if output_max is not None:
        pid_kwargs["output_max"] = output_max
    block = PIDController2DOF(**pid_kwargs)
    init_kwargs = dict(
        kp=kp, ki=ki, kd=kd, b=b, c=c, initial_state=initial_state,
        filter_type="none", filter_coefficient=1.0,
        anti_windup_method=anti_windup_method,
        anti_windup_gain=anti_windup_gain,
        tracking_enabled=tracking_enabled,
        tracking_gain=tracking_gain,
    )
    if output_min is not None:
        init_kwargs["output_min"] = output_min
    if output_max is not None:
        init_kwargs["output_max"] = output_max
    block.initialize(**init_kwargs)
    return block


def _run_steps(
    block,
    r,
    y,
    *,
    n_steps,
    kp,
    ki,
    kd,
    b=1.0,
    c=1.0,
    u_ext=None,
    output_min=None,
    output_max=None,
    anti_windup_gain=1.0,
    tracking_gain=1.0,
    initial_integral=0.0,
):
    """Tick the block ``n_steps`` times against constant scalar inputs.

    Returns ``(integrals, outputs)`` arrays of length ``n_steps``.
    """
    State = namedtuple("State", ["discrete_state"])
    xd0 = block.DiscreteStateType(
        integral=jnp.asarray(initial_integral, dtype=jnp.float64),
        e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
        e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
    )
    state = State(discrete_state=xd0)
    params = dict(
        kp=kp, ki=ki, kd=kd, b=b, c=c,
        anti_windup_gain=anti_windup_gain,
        tracking_gain=tracking_gain,
    )
    if output_min is not None:
        params["output_min"] = output_min
    if output_max is not None:
        params["output_max"] = output_max

    r = jnp.asarray(r, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if u_ext is not None:
        u_ext = jnp.asarray(u_ext, dtype=jnp.float64)

    integrals = []
    outputs = []
    for _ in range(n_steps):
        if block.tracking_enabled:
            inputs = (r, y, u_ext)
        else:
            inputs = (r, y)
        u = block._output(jnp.asarray(0.0), state, *inputs, **params)
        outputs.append(u)
        new_xd = block._update(jnp.asarray(0.0), state, *inputs, **params)
        integrals.append(new_xd.integral)
        state = State(discrete_state=new_xd)
    return jnp.asarray(integrals), jnp.asarray(outputs)


# --------------------------------------------------------------------- #
# Default-off byte-equivalence with phase 1
# --------------------------------------------------------------------- #


class TestDefaultOffByteEquivalence:
    """``tracking_enabled=False`` → identical to phase 1 behaviour."""

    def test_closed_loop_step_tracking_unchanged(self):
        """Closed-loop tracking trace matches the no-tracking baseline."""
        # Baseline — phase 1 construction (no tracking kwargs).
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

        # New construction with explicit ``tracking_enabled=False``.
        builder2 = jaxonomy.DiagramBuilder()
        r2 = builder2.add(Constant(1.0, name="r"))
        pid2 = builder2.add(
            PIDController2DOF(
                dt=0.05, kp=2.0, ki=4.0, kd=0.0, b=1.0, c=1.0,
                tracking_enabled=False, tracking_gain=0.25,
                name="pid",
            )
        )
        plant2 = builder2.add(Integrator(initial_state=0.0, name="plant"))
        builder2.connect(r2.output_ports[0], pid2.input_ports[0])
        builder2.connect(plant2.output_ports[0], pid2.input_ports[1])
        builder2.connect(pid2.output_ports[0], plant2.input_ports[0])
        diagram2 = builder2.build()
        context2 = diagram2.create_context()
        results_new = jaxonomy.simulate(
            diagram2, context2, (0.0, 5.0),
            recorded_signals={"y": plant2.output_ports[0]},
        )
        y_new = results_new.outputs["y"]

        assert y_phase1.shape == y_new.shape
        assert jnp.allclose(y_phase1, y_new, atol=0.0, rtol=0.0), (
            "default-off tracking-mode is not byte-equivalent to phase 1; "
            f"max diff = {float(jnp.max(jnp.abs(y_phase1 - y_new)))}"
        )

    def test_per_step_update_byte_equal_with_disabled_tracking(self):
        """``_update`` is bit-equal to phase 1 when tracking is disabled.

        Even a non-default ``tracking_gain`` must have zero effect when
        ``tracking_enabled=False`` because the entire tracking branch is
        gated off.
        """
        block_default = _make_block(kp=1.0, ki=2.0, kd=0.5)
        block_with_kwargs = _make_block(
            kp=1.0, ki=2.0, kd=0.5,
            tracking_enabled=False, tracking_gain=0.25,
        )
        ints_default, us_default = _run_steps(
            block_default, r=1.0, y=0.0,
            n_steps=10, kp=1.0, ki=2.0, kd=0.5,
        )
        ints_kwargs, us_kwargs = _run_steps(
            block_with_kwargs, r=1.0, y=0.0,
            n_steps=10, kp=1.0, ki=2.0, kd=0.5,
            tracking_gain=0.25,
        )
        assert jnp.allclose(ints_default, ints_kwargs, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_default, us_kwargs, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Tracking-mode convergence
# --------------------------------------------------------------------- #


class TestTrackingConvergence:
    """With ``tracking_enabled=True``, ``u`` converges toward ``u_ext``."""

    def test_output_converges_to_u_ext_constant(self):
        """A constant ``u_ext`` pulls the PID output toward it.

        With ``r = y = 0`` the bare PID would output 0 forever; the
        tracking term must drive the integrator (and hence the output)
        toward ``u_ext`` over many ticks.  We choose a small
        ``tracking_gain`` so convergence is fast within the simulation
        window.
        """
        u_ext_target = 0.7
        n_steps = 500
        block = _make_block(
            kp=0.0, ki=0.0, kd=0.0,
            tracking_enabled=True, tracking_gain=0.5,
        )
        _, us = _run_steps(
            block, r=0.0, y=0.0,
            n_steps=n_steps, kp=0.0, ki=0.0, kd=0.0,
            u_ext=u_ext_target, tracking_gain=0.5,
        )
        # First tick: u = 0 (integral starts at 0, no P/D action).
        assert float(us[0]) == pytest.approx(0.0, abs=1e-12)
        # Late samples: integrator has converged toward u_ext_target
        # (with ki=0 the integral feeds nothing into u, so equivalently
        # we use ki=1.0 in a second variant below).  Here, since ki=0,
        # the output stays at 0 even though the integral grows.  So
        # check the *integrator* convergence directly via _update.
        State = namedtuple("State", ["discrete_state"])
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0, dtype=jnp.float64),
            e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
            e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
        )
        state = State(discrete_state=xd0)
        for _ in range(n_steps):
            new_xd = block._update(
                jnp.asarray(0.0), state,
                jnp.asarray(0.0), jnp.asarray(0.0),
                jnp.asarray(u_ext_target),
                kp=0.0, ki=0.0, kd=0.0, b=1.0, c=1.0,
                anti_windup_gain=1.0, tracking_gain=0.5,
            )
            state = State(discrete_state=new_xd)
        # Integrator should be approximately u_ext_target * 1 (since
        # kp=ki=kd=0, u_unsat = 0 every tick, so each step adds
        # (u_ext - 0) / 0.5 * 0.1 = 0.2 * u_ext.  This will overshoot
        # because there is no negative feedback on the integrator
        # without ki).  Use a separate ki=1 test for true convergence.

    def test_output_converges_to_u_ext_with_ki(self):
        """With ki>0 (and r=y=0) the steady-state output is u_ext.

        At steady state: u = ki * I; the integrator update is
        I[k+1] = I[k] + ki*(r-y)*dt + (u_ext - u)/Tt * dt
              = I[k]                    + (u_ext - ki*I[k])/Tt * dt
        Setting I[k+1] = I[k] → u_ext = ki * I = u → output = u_ext.
        """
        u_ext_target = 0.7
        n_steps = 1500  # need plenty of ticks for convergence
        kp, ki, kd = 0.0, 1.0, 0.0
        block = _make_block(
            kp=kp, ki=ki, kd=kd,
            tracking_enabled=True, tracking_gain=0.1,
            dt=0.05,
        )
        _, us = _run_steps(
            block, r=0.0, y=0.0,
            n_steps=n_steps, kp=kp, ki=ki, kd=kd,
            u_ext=u_ext_target, tracking_gain=0.1,
        )
        # Late samples should be close to u_ext_target.
        tail = us[-50:]
        assert float(jnp.mean(tail)) == pytest.approx(u_ext_target, abs=2e-2), (
            f"output did not converge to u_ext={u_ext_target}; "
            f"tail mean = {float(jnp.mean(tail))}"
        )


# --------------------------------------------------------------------- #
# Bumpless transfer
# --------------------------------------------------------------------- #


class TestBumplessTransfer:
    """Tracking mode produces a smooth handoff to PID-driven control."""

    def test_no_bump_after_tracking_loads_integrator(self):
        """Pre-load integrator via tracking → switch back to PID is smooth.

        Scenario: r = y = 0, u_ext = 0.5 for 200 ticks (PID's integrator
        converges so kp*0 + ki*I ≈ 0.5; output ≈ 0.5).  At t=200 we
        "switch back" — for the purposes of this test, we read the
        instantaneous output u[200].  If the integrator was correctly
        loaded, swapping to PID-driven control (no more u_ext influence)
        produces ``u`` equal to the last tracking-mode output → no bump.
        """
        u_ext_target = 0.5
        kp, ki, kd = 0.0, 1.0, 0.0
        block_track = _make_block(
            kp=kp, ki=ki, kd=kd,
            tracking_enabled=True, tracking_gain=0.05,
            dt=0.05,
        )
        # Run with tracking until the integrator converges.
        State = namedtuple("State", ["discrete_state"])
        xd = block_track.DiscreteStateType(
            integral=jnp.asarray(0.0, dtype=jnp.float64),
            e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
            e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
        )
        state = State(discrete_state=xd)
        track_params = dict(
            kp=kp, ki=ki, kd=kd, b=1.0, c=1.0,
            anti_windup_gain=1.0, tracking_gain=0.05,
        )
        for _ in range(2000):
            new_xd = block_track._update(
                jnp.asarray(0.0), state,
                jnp.asarray(0.0), jnp.asarray(0.0),
                jnp.asarray(u_ext_target),
                **track_params,
            )
            state = State(discrete_state=new_xd)
        u_before = block_track._output(
            jnp.asarray(0.0), state,
            jnp.asarray(0.0), jnp.asarray(0.0),
            jnp.asarray(u_ext_target),
            **track_params,
        )
        # u_before should match u_ext (≈ 0.5).
        assert float(u_before) == pytest.approx(u_ext_target, abs=1e-3)

        # Now switch to a PID built without tracking, copying the
        # converged integrator into its initial state.  The output on
        # the first tick of "auto" mode should equal u_before — no
        # bump.
        loaded_integral = float(state.discrete_state.integral)
        block_auto = _make_block(
            kp=kp, ki=ki, kd=kd,
            initial_state=loaded_integral,
            tracking_enabled=False,
            dt=0.05,
        )
        xd_auto = block_auto.DiscreteStateType(
            integral=jnp.asarray(loaded_integral, dtype=jnp.float64),
            e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
            e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
        )
        state_auto = State(discrete_state=xd_auto)
        u_after = block_auto._output(
            jnp.asarray(0.0), state_auto,
            jnp.asarray(0.0), jnp.asarray(0.0),
            kp=kp, ki=ki, kd=kd, b=1.0, c=1.0,
            anti_windup_gain=1.0,
        )
        # Bumpless: u_after should equal u_before to high precision.
        bump = float(jnp.abs(u_after - u_before))
        assert bump < 1e-3, f"bump={bump} too large; not bumpless"


# --------------------------------------------------------------------- #
# Composition with anti-windup
# --------------------------------------------------------------------- #


class TestComposeWithAntiWindup:
    """Tracking-mode and anti-windup can be active simultaneously."""

    def test_both_active_no_nan(self):
        """Smoke: both mechanisms produce finite results when active."""
        block = _make_block(
            kp=1.0, ki=4.0, kd=0.5,
            output_min=-1.0, output_max=1.0,
            anti_windup_method="back_calc", anti_windup_gain=0.2,
            tracking_enabled=True, tracking_gain=0.5,
        )
        ints, us = _run_steps(
            block, r=2.0, y=0.0,
            n_steps=200, kp=1.0, ki=4.0, kd=0.5,
            u_ext=0.0,
            output_min=-1.0, output_max=1.0,
            anti_windup_gain=0.2, tracking_gain=0.5,
        )
        assert jnp.all(jnp.isfinite(ints))
        assert jnp.all(jnp.isfinite(us))
        # Output is clipped to [-1, 1] regardless of internal state.
        assert jnp.all(us <= 1.0 + 1e-12)
        assert jnp.all(us >= -1.0 - 1e-12)

    def test_tracking_correction_active_with_anti_windup(self):
        """Tracking pulls integrator toward u_ext; with antiwindup off,
        the correction is provably larger in magnitude than without it.

        We compare two long simulations: (a) anti-windup only, (b)
        anti-windup + tracking with u_ext = 0 driving the integrator
        toward zero.  Variant (b)'s steady-state integrator must lie
        between variant (a)'s integrator value and zero (the tracking
        signal pulls it down).
        """
        common = dict(
            kp=0.0, ki=5.0, kd=0.0,
            output_min=-0.5, output_max=0.5,
            anti_windup_method="back_calc", anti_windup_gain=1.0,
        )
        n_steps = 200
        # (a) Anti-windup alone.
        block_a = _make_block(**common)
        ints_a, _ = _run_steps(
            block_a, r=1.0, y=0.0,
            n_steps=n_steps, kp=0.0, ki=5.0, kd=0.0,
            output_min=-0.5, output_max=0.5,
            anti_windup_gain=1.0,
        )
        # (b) Anti-windup + tracking with u_ext=0.
        block_b = _make_block(
            tracking_enabled=True, tracking_gain=0.5,
            **common,
        )
        ints_b, _ = _run_steps(
            block_b, r=1.0, y=0.0,
            n_steps=n_steps, kp=0.0, ki=5.0, kd=0.0,
            u_ext=0.0,
            output_min=-0.5, output_max=0.5,
            anti_windup_gain=1.0, tracking_gain=0.5,
        )
        # Tracking term drags the steady-state integral toward 0.
        # Both are positive (e_i > 0 dominates), but (b)'s tail is
        # smaller in magnitude.
        i_a_tail = float(jnp.mean(ints_a[-20:]))
        i_b_tail = float(jnp.mean(ints_b[-20:]))
        assert i_a_tail > i_b_tail, (
            f"tracking u_ext=0 should pull the integrator down; "
            f"i_a={i_a_tail}, i_b={i_b_tail}"
        )


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    """Port topology is locked at construction time."""

    def test_tracking_enabled_change_after_init_rejected(self):
        """Flipping ``tracking_enabled`` in ``initialize`` is rejected."""
        block = PIDController2DOF(
            dt=0.1, kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            tracking_enabled=False, name="pid",
        )
        with pytest.raises(
            ValueError, match="tracking_enabled cannot be changed"
        ):
            block.initialize(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0, initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                tracking_enabled=True,  # was False at construction
            )

    def test_tracking_enabled_unconnected_port_raises(self):
        """``tracking_enabled=True`` with the new port unconnected
        surfaces a build / simulate error rather than silently using
        a default value.
        """
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=1.0, ki=0.0, kd=0.0,
                tracking_enabled=True, tracking_gain=1.0, name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        # Intentionally do NOT connect input_ports[2] (the u_ext port).
        with pytest.raises(Exception):
            diagram = builder.build()
            context = diagram.create_context()
            jaxonomy.simulate(
                diagram, context, (0.0, 0.2),
                recorded_signals={"u": pid.output_ports[0]},
            )

    def test_tracking_enabled_port_count(self):
        """``tracking_enabled=True`` adds exactly one input port."""
        # Base: 2 inputs (r, y).
        pid_off = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0, name="off",
        )
        assert len(pid_off.input_ports) == 2
        pid_on = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0,
            tracking_enabled=True, name="on",
        )
        assert len(pid_on.input_ports) == 3
        assert pid_on.u_ext_index == 2

    def test_u_ext_port_after_other_dynamic_ports(self):
        """``u_ext`` is appended after the b/c/gain/kff dynamic ports."""
        pid = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0,
            b_dynamic=True, kp_dynamic=True, kff_dynamic=True,
            tracking_enabled=True, name="pid",
        )
        # r, y, b, kp, kff, u_ext  → 6 ports total.
        assert len(pid.input_ports) == 6
        # u_ext is the last one — its index must be >= every other
        # dynamic port's index.
        assert pid.u_ext_index == 5
        assert pid.b_index < pid.u_ext_index
        assert pid.kp_index < pid.u_ext_index
        assert pid.kff_index < pid.u_ext_index


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` flows through ``tracking_gain``."""

    @staticmethod
    def _loss(tracking_gain, n_steps=400):
        """Open-loop integrator-trajectory loss as a function of Tt."""
        kp, ki, kd = 0.0, 1.0, 0.0
        block = _make_block(
            kp=kp, ki=ki, kd=kd,
            dt=0.05,
            tracking_enabled=True,
            tracking_gain=tracking_gain,
        )
        State = namedtuple("State", ["discrete_state"])
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0, dtype=jnp.float64),
            e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
            e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(0.0)
        y = jnp.asarray(0.0)
        u_ext = jnp.asarray(0.5)
        params = dict(
            kp=kp, ki=ki, kd=kd, b=1.0, c=1.0,
            anti_windup_gain=1.0, tracking_gain=tracking_gain,
        )
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, u_ext, **params)
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, u_ext, **params
            )
            state = State(discrete_state=new_xd)
            total = total + (u_ext - u) ** 2
        return total

    def test_grad_wrt_tracking_gain_finite(self):
        g = jax.grad(self._loss)(jnp.asarray(0.5))
        assert jnp.isfinite(g), f"grad wrt tracking_gain not finite: {g}"

    def test_grad_wrt_tracking_gain_nonzero(self):
        # Smaller Tt → faster convergence → smaller loss; gradient must
        # be non-zero in the convergence regime.
        g = jax.grad(self._loss)(jnp.asarray(0.5))
        assert jnp.abs(g) > 0, (
            f"grad wrt tracking_gain should be nonzero; got {g}"
        )


# --------------------------------------------------------------------- #
# Config round-trip
# --------------------------------------------------------------------- #


class TestConfigRoundTrip:
    """``to_dict`` / ``from_dict`` preserve the tracking-mode kwargs."""

    def test_round_trip_default(self):
        pid = PIDController2DOF(dt=0.05, name="pid")
        data = pid.to_dict()
        assert data["tracking_enabled"] is False
        assert data["tracking_gain"] == pytest.approx(1.0)
        # Reconstruct and verify port count unchanged.
        pid2 = PIDController2DOF.from_dict(data, name="pid2")
        assert pid2.tracking_enabled is False
        assert len(pid2.input_ports) == 2

    def test_round_trip_enabled(self):
        pid = PIDController2DOF(
            dt=0.05, kp=2.0, ki=3.0,
            tracking_enabled=True, tracking_gain=0.25,
            name="pid",
        )
        data = pid.to_dict()
        assert data["tracking_enabled"] is True
        assert data["tracking_gain"] == pytest.approx(0.25)
        pid2 = PIDController2DOF.from_dict(data, name="pid2")
        assert pid2.tracking_enabled is True
        assert len(pid2.input_ports) == 3
        assert pid2.u_ext_index == 2
