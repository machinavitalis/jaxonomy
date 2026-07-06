# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-derivative-on-measurement.

Covers the new ``derivative_on_measurement_only`` convenience kwarg and
the two factory class methods on :class:`PIDController2DOF`:

* ``PIDController2DOF.standard(kp, ki, kd, dt)`` — textbook 2-DOF PID
  with ``b = c = 1``.
* ``PIDController2DOF.with_derivative_on_measurement(kp, ki, kd, dt)`` —
  derivative-on-measurement-only with ``b = 1, c = 0``, the standard
  "no derivative kick" recipe for real-world controllers.

A step change in the setpoint produces a one-tick ``Kd / dt`` spike
through ``d/dt(c*r - y)`` when ``c = 1``; with ``c = 0`` (or the
convenience flag / factory) that spike disappears.

These tests cover:

* ``PIDController2DOF(derivative_on_measurement_only=True)`` is byte-
  equivalent to ``PIDController2DOF(c=0.0)``.
* Step setpoint change: with ``c = 0`` the derivative term shows NO
  spike at step time; with ``c = 1`` (default) the derivative spikes
  at step.
* ``PIDController2DOF.standard(...)`` returns a PID with ``b = c = 1``
  whose output matches the default-constructed block.
* ``PIDController2DOF.with_derivative_on_measurement(...)`` returns a
  PID with ``b = 1, c = 0`` whose output matches the explicit-``c=0``
  block.
* Validation: combining ``derivative_on_measurement_only=True`` with
  an explicit ``c`` or ``c_dynamic`` is rejected at construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    PIDController2DOF,
    Step,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_open_loop(pid, *, r_block, y_block, t_end=1.0):
    """Run ``pid`` in an open loop driven by ``r_block`` and ``y_block``.

    Returns the recorded control signal ``u`` and the time vector.
    """
    builder = jaxonomy.DiagramBuilder()
    r = builder.add(r_block)
    y = builder.add(y_block)
    pid_b = builder.add(pid)
    builder.connect(r.output_ports[0], pid_b.input_ports[0])
    builder.connect(y.output_ports[0], pid_b.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid_b.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.time


# --------------------------------------------------------------------- #
# Convenience-kwarg equivalence with explicit c=0
# --------------------------------------------------------------------- #


class TestDerivativeOnMeasurementFlag:
    """``derivative_on_measurement_only=True`` is sugar for ``c=0``."""

    def test_matches_explicit_c_zero(self):
        """Byte-for-byte: convenience flag vs explicit ``c=0``."""
        dt = 0.1
        kp, ki, kd = 1.0, 2.0, 0.5

        pid_flag = PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            derivative_on_measurement_only=True,
            name="pid_flag",
        )
        u_flag, _ = _simulate_open_loop(
            pid_flag,
            r_block=Constant(1.0, name="r"),
            y_block=Constant(0.0, name="y"),
            t_end=1.0,
        )

        pid_c0 = PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            b=1.0,
            c=0.0,
            name="pid_c0",
        )
        u_c0, _ = _simulate_open_loop(
            pid_c0,
            r_block=Constant(1.0, name="r"),
            y_block=Constant(0.0, name="y"),
            t_end=1.0,
        )

        assert u_flag.shape == u_c0.shape
        assert jnp.allclose(u_flag, u_c0, atol=0.0, rtol=0.0), (
            f"derivative_on_measurement_only=True differs from c=0; "
            f"max diff {float(jnp.max(jnp.abs(u_flag - u_c0)))}"
        )

    def test_default_is_c_equals_one(self):
        """Default (``derivative_on_measurement_only=False``) keeps ``c=1``."""
        dt = 0.1
        kp, ki, kd = 1.0, 2.0, 0.5

        # Default (no flag set) — c=1.
        pid_default = PIDController2DOF(
            dt=dt, kp=kp, ki=ki, kd=kd, name="pid_default"
        )
        u_default, _ = _simulate_open_loop(
            pid_default,
            r_block=Constant(1.0, name="r"),
            y_block=Constant(0.0, name="y"),
            t_end=1.0,
        )

        pid_c1 = PIDController2DOF(
            dt=dt, kp=kp, ki=ki, kd=kd, b=1.0, c=1.0, name="pid_c1"
        )
        u_c1, _ = _simulate_open_loop(
            pid_c1,
            r_block=Constant(1.0, name="r"),
            y_block=Constant(0.0, name="y"),
            t_end=1.0,
        )

        # Default must be byte-equivalent to phase 1 (c=1).
        assert jnp.allclose(u_default, u_c1, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Derivative-kick suppression on a setpoint step
# --------------------------------------------------------------------- #


class TestDerivativeKickSuppression:
    """A step in the setpoint kicks the derivative term unless ``c=0``."""

    def test_step_setpoint_no_kick_with_d_on_measurement(self):
        """With ``c = 0`` and ``y = 0``, a setpoint step has no D spike.

        Setup:
            kp = ki = 0, kd > 0, y == 0 constant, r = Step(0 → 1).
            Output is therefore *exclusively* the derivative term.
            With ``c = 0``: D = -Kd * d/dt(y) = 0 at all times.
            With ``c = 1``: D = Kd * d/dt(r - y) spikes by Kd/dt at
            the step.
        """
        dt = 0.1
        kd = 0.5
        # Place the step well inside the simulation window so we observe
        # both pre-step (D == 0) and post-step (no spike with c=0).
        step_block_factory = lambda: Step(
            start_value=0.0, end_value=1.0, step_time=0.3
        )

        # c = 0 (derivative on measurement only).
        pid_dm = PIDController2DOF(
            dt=dt,
            kp=0.0,
            ki=0.0,
            kd=kd,
            derivative_on_measurement_only=True,
            name="pid_dm",
        )
        u_dm, t_dm = _simulate_open_loop(
            pid_dm,
            r_block=step_block_factory(),
            y_block=Constant(0.0, name="y"),
            t_end=0.8,
        )

        # c = 1 (standard).
        pid_std = PIDController2DOF(
            dt=dt, kp=0.0, ki=0.0, kd=kd, b=1.0, c=1.0, name="pid_std"
        )
        u_std, t_std = _simulate_open_loop(
            pid_std,
            r_block=step_block_factory(),
            y_block=Constant(0.0, name="y"),
            t_end=0.8,
        )

        # With c=0 and y constant, the derivative term is identically 0
        # for all samples regardless of the setpoint step.
        assert jnp.allclose(u_dm, 0.0, atol=1e-10), (
            f"c=0 should suppress derivative kick entirely; got "
            f"max|u_dm|={float(jnp.max(jnp.abs(u_dm)))}"
        )

        # With c=1 the derivative term must spike at the step — the
        # maximum-magnitude sample exceeds zero by a clearly observable
        # margin (forward-difference kernel on a unit step at dt=0.1
        # gives Kd/dt = 5.0).
        peak_std = float(jnp.max(jnp.abs(u_std)))
        assert peak_std > 1.0, (
            f"c=1 should kick the derivative on a setpoint step; "
            f"got peak |u_std|={peak_std}"
        )

        # And explicitly: c=0 produces a strictly smaller peak than c=1.
        peak_dm = float(jnp.max(jnp.abs(u_dm)))
        assert peak_dm < peak_std, (
            f"c=0 peak {peak_dm} should be smaller than c=1 peak {peak_std}"
        )


# --------------------------------------------------------------------- #
# Factory class methods
# --------------------------------------------------------------------- #


class TestFactoryMethods:
    """``PIDController2DOF.standard`` and ``.with_derivative_on_measurement``."""

    def test_standard_returns_b_c_one(self):
        """``standard(kp, ki, kd, dt)`` matches the default constructor."""
        dt = 0.05
        kp, ki, kd = 1.5, 0.8, 0.2

        pid_factory = PIDController2DOF.standard(kp, ki, kd, dt, name="pid_f")
        # Underlying static weights are b=c=1.
        u_factory, _ = _simulate_open_loop(
            pid_factory,
            r_block=Constant(1.0, name="r"),
            y_block=Constant(0.0, name="y"),
            t_end=0.5,
        )

        pid_ref = PIDController2DOF(
            dt=dt, kp=kp, ki=ki, kd=kd, b=1.0, c=1.0, name="pid_r"
        )
        u_ref, _ = _simulate_open_loop(
            pid_ref,
            r_block=Constant(1.0, name="r"),
            y_block=Constant(0.0, name="y"),
            t_end=0.5,
        )

        assert jnp.allclose(u_factory, u_ref, atol=0.0, rtol=0.0)

    def test_with_derivative_on_measurement_returns_b1_c0(self):
        """``with_derivative_on_measurement`` ≡ ``c=0`` controller."""
        dt = 0.05
        kp, ki, kd = 1.5, 0.8, 0.2

        pid_factory = PIDController2DOF.with_derivative_on_measurement(
            kp, ki, kd, dt, name="pid_f"
        )
        u_factory, _ = _simulate_open_loop(
            pid_factory,
            r_block=Step(start_value=0.0, end_value=1.0, step_time=0.2),
            y_block=Constant(0.0, name="y"),
            t_end=0.5,
        )

        pid_ref = PIDController2DOF(
            dt=dt, kp=kp, ki=ki, kd=kd, b=1.0, c=0.0, name="pid_r"
        )
        u_ref, _ = _simulate_open_loop(
            pid_ref,
            r_block=Step(start_value=0.0, end_value=1.0, step_time=0.2),
            y_block=Constant(0.0, name="y"),
            t_end=0.5,
        )

        assert jnp.allclose(u_factory, u_ref, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Validation: incompatible kwargs
# --------------------------------------------------------------------- #


class TestValidation:
    """Combining the flag with explicit ``c`` / ``c_dynamic`` is rejected."""

    def test_rejects_explicit_c_with_flag(self):
        with pytest.raises(ValueError, match="derivative_on_measurement_only"):
            PIDController2DOF(
                dt=0.1,
                kp=1.0,
                ki=0.0,
                kd=0.5,
                c=0.5,
                derivative_on_measurement_only=True,
                name="pid",
            )

    def test_rejects_c_dynamic_with_flag(self):
        with pytest.raises(ValueError, match="derivative_on_measurement_only"):
            PIDController2DOF(
                dt=0.1,
                kp=1.0,
                ki=0.0,
                kd=0.5,
                c_dynamic=True,
                derivative_on_measurement_only=True,
                name="pid",
            )
