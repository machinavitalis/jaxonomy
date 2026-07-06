# SPDX-License-Identifier: MIT

"""Tests for T-127 Phase 1 — :class:`PIDController2DOF`.

T-127 (originally tracked as T-MW-304, renumbered in 124c178) ships the
two-degree-of-freedom discrete PID controller, plus follow-ups for the
discrete-filter family.  Phase 1 covers only the 2-DOF PID block.

The 2-DOF control law is::

    u = Kp * (b*r - y) + Ki * integral(r - y) + Kd * d/dt(c*r - y)

where ``r`` is the setpoint, ``y`` is the measurement, and ``b``, ``c``
are the proportional and derivative setpoint weights.

These tests cover:

* Numerical equivalence with the existing :class:`PIDDiscrete` block
  when ``b = c = 1``.
* Setpoint-weight semantics: with ``b = 0`` the proportional term does
  not see the setpoint (only the measurement contributes), so a step
  in the setpoint produces no instantaneous change in the proportional
  contribution to ``u``.
* Differentiability of the simulated output with respect to the gains
  (``Kp``, ``Ki``, ``Kd``) and weights (``b``, ``c``).
* Closed-loop tracking of a constant setpoint with a simple integrator
  plant (``y' = u``): the steady-state error is driven to zero by the
  integral action.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import (
    Constant,
    Integrator,
    PIDController2DOF,
    PIDDiscrete,
    Sine,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_open_loop_2dof(
    r_block,
    y_block,
    *,
    dt=0.1,
    kp=1.0,
    ki=1.0,
    kd=0.1,
    b=1.0,
    c=1.0,
    t_end=2.0,
    filter_type="none",
    filter_coefficient=1.0,
):
    """Run a 2-DOF PID block in an open loop driven by ``r`` and ``y``."""
    builder = jaxonomy.DiagramBuilder()
    r = builder.add(r_block)
    y = builder.add(y_block)
    pid = builder.add(
        PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            b=b,
            c=c,
            filter_type=filter_type,
            filter_coefficient=filter_coefficient,
            name="pid2",
        )
    )
    builder.connect(r.output_ports[0], pid.input_ports[0])
    builder.connect(y.output_ports[0], pid.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.time


def _simulate_open_loop_1dof(
    e_block,
    *,
    dt=0.1,
    kp=1.0,
    ki=1.0,
    kd=0.1,
    t_end=2.0,
    filter_type="none",
    filter_coefficient=1.0,
):
    """Run the existing :class:`PIDDiscrete` block on an error signal."""
    builder = jaxonomy.DiagramBuilder()
    e = builder.add(e_block)
    pid = builder.add(
        PIDDiscrete(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            filter_type=filter_type,
            filter_coefficient=filter_coefficient,
            name="pid",
        )
    )
    builder.connect(e.output_ports[0], pid.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.time


# --------------------------------------------------------------------- #
# Equivalence with PIDDiscrete (b = c = 1)
# --------------------------------------------------------------------- #


class TestEquivalenceWithPIDDiscrete:
    """``b = c = 1`` reduces to the standard 1-DOF PID."""

    @pytest.mark.parametrize(
        "kp, ki, kd",
        [
            (1.0, 0.0, 0.0),  # P only
            (0.0, 2.0, 0.0),  # I only
            (0.0, 0.0, 0.5),  # D only (no filter)
            (1.0, 2.0, 0.5),  # full PID
        ],
    )
    def test_matches_1dof_with_zero_measurement(self, kp, ki, kd):
        """When y == 0, the error is r, so 2-DOF should equal 1-DOF on r."""
        dt = 0.1

        # 2-DOF: setpoint = sine, measurement = 0  → error = sine.
        u2, _ = _simulate_open_loop_2dof(
            Sine(frequency=1.0, amplitude=1.0, phase=0.0),
            Constant(0.0, name="zero"),
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            b=1.0,
            c=1.0,
        )

        # 1-DOF: error signal is sine.
        u1, _ = _simulate_open_loop_1dof(
            Sine(frequency=1.0, amplitude=1.0, phase=0.0),
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
        )

        assert u2.shape == u1.shape
        assert jnp.allclose(u2, u1, atol=1e-10), (
            f"2-DOF (b=c=1) and 1-DOF disagree (max diff "
            f"{float(jnp.max(jnp.abs(u2 - u1)))})"
        )

    def test_matches_1dof_with_filter(self):
        """The same equivalence must hold with a derivative filter on."""
        dt = 0.1
        kwargs = dict(
            kp=1.0, ki=2.0, kd=0.3, filter_type="forward", filter_coefficient=10.0
        )

        u2, _ = _simulate_open_loop_2dof(
            Sine(frequency=1.0, amplitude=1.0, phase=0.0),
            Constant(0.0, name="zero"),
            dt=dt,
            b=1.0,
            c=1.0,
            **kwargs,
        )
        u1, _ = _simulate_open_loop_1dof(
            Sine(frequency=1.0, amplitude=1.0, phase=0.0),
            dt=dt,
            **kwargs,
        )

        assert jnp.allclose(u2, u1, atol=1e-10)


# --------------------------------------------------------------------- #
# Setpoint-weight semantics
# --------------------------------------------------------------------- #


class TestSetpointWeights:
    """``b`` and ``c`` correctly attenuate the setpoint into P and D paths."""

    def test_b_zero_kills_setpoint_in_proportional_path(self):
        """With kp != 0, ki = kd = 0, b = 0: the proportional term is -kp*y.

        So with y = 0 and any setpoint r != 0, the output should be ~0
        (modulo the initial-tick ZOH default), independent of r.
        """
        dt = 0.1
        u_b0, _ = _simulate_open_loop_2dof(
            Constant(5.0, name="r"),
            Constant(0.0, name="y"),
            dt=dt,
            kp=2.0,
            ki=0.0,
            kd=0.0,
            b=0.0,
            c=0.0,
            t_end=1.0,
        )
        # All steady-state samples (after the first tick) should be 0.
        assert jnp.allclose(u_b0[1:], 0.0, atol=1e-10), (
            f"b=0 with y=0 should give u=0 in steady state; got {u_b0}"
        )

        # And as a sanity contrast, b = 1 should give u = kp * r = 10.0.
        u_b1, _ = _simulate_open_loop_2dof(
            Constant(5.0, name="r"),
            Constant(0.0, name="y"),
            dt=dt,
            kp=2.0,
            ki=0.0,
            kd=0.0,
            b=1.0,
            c=1.0,
            t_end=1.0,
        )
        assert jnp.allclose(u_b1[1:], 10.0, atol=1e-10), (
            f"b=1 with y=0 should give u=kp*r=10; got {u_b1}"
        )

    def test_b_zero_proportional_responds_only_to_measurement(self):
        """With b = 0, P-term = -kp*y; varying r should not change u (P only)."""
        dt = 0.1
        u_r1, _ = _simulate_open_loop_2dof(
            Constant(1.0, name="r"),
            Constant(0.5, name="y"),
            dt=dt,
            kp=3.0,
            ki=0.0,
            kd=0.0,
            b=0.0,
            c=0.0,
            t_end=1.0,
        )
        u_r2, _ = _simulate_open_loop_2dof(
            Constant(7.0, name="r"),
            Constant(0.5, name="y"),
            dt=dt,
            kp=3.0,
            ki=0.0,
            kd=0.0,
            b=0.0,
            c=0.0,
            t_end=1.0,
        )
        # Both should give u = -kp * y = -1.5 in steady state.
        assert jnp.allclose(u_r1[1:], -1.5, atol=1e-10)
        assert jnp.allclose(u_r2[1:], -1.5, atol=1e-10)
        assert jnp.allclose(u_r1[1:], u_r2[1:], atol=1e-10)

    def test_integral_term_ignores_setpoint_weights(self):
        """The integral path uses e_i = r - y, independent of b and c."""
        dt = 0.1
        # Pure-I controller with b = c = 0; integral should still react to
        # the setpoint via e_i = r - y.
        u, _ = _simulate_open_loop_2dof(
            Constant(1.0, name="r"),
            Constant(0.0, name="y"),
            dt=dt,
            kp=0.0,
            ki=1.0,
            kd=0.0,
            b=0.0,
            c=0.0,
            t_end=1.0,
        )
        # u[k] = ki * sum_{j<k}(r-y)*dt = sum_{j<k}(0.1) = k*0.1.
        # Allow the first sample to be 0 (integral default).
        assert u[-1] > 0.5, f"integral should accumulate; got u[-1]={float(u[-1])}"


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """Gradients flow through ``Kp``, ``Ki``, ``Kd``, ``b``, ``c``.

    Bypasses the full simulator (whose recorded-signals path is not JAX-
    traceable as written) and exercises ``PIDController2DOF._update`` /
    ``_output`` directly across a small loop — the same pattern used by
    ``test_t_123_rate_transition_phase1.test_decimator_gradient_flows_through_input``.
    """

    @staticmethod
    def _make_block(kp, ki, kd, b, c, dt=0.1):
        """Construct and initialise a PIDController2DOF outside a Diagram.

        The ``@parameters`` decorator wires ``initialize(...)`` into the
        builder lifecycle; when the block is exercised standalone we have
        to call it ourselves so ``self.filter`` / ``self.filter_type`` are
        available before ``_update`` / ``_output`` run.
        """
        block = PIDController2DOF(
            dt=dt, kp=kp, ki=ki, kd=kd, b=b, c=c, name="pid2"
        )
        block.initialize(
            kp=kp,
            ki=ki,
            kd=kd,
            b=b,
            c=c,
            initial_state=0.0,
            filter_type="none",
            filter_coefficient=1.0,
        )
        return block

    @classmethod
    def _step_loss(cls, kp, ki, kd, b, c, n_steps=5):
        """Run the block for ``n_steps`` ticks with constant ``r=1``, ``y=0``."""
        from collections import namedtuple

        State = namedtuple("State", ["discrete_state"])
        block = cls._make_block(kp, ki, kd, b, c)

        # Mirror the discrete-state seed declared in `initialize`.
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0),
            e_d_prev=jnp.asarray(0.0),
            e_dot_prev=jnp.asarray(0.0),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(1.0)
        y = jnp.asarray(0.0)
        params = dict(kp=kp, ki=ki, kd=kd, b=b, c=c)

        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, **params)
            total = total + jnp.abs(u)
            new_xd = block._update(jnp.asarray(0.0), state, r, y, **params)
            state = State(discrete_state=new_xd)
        return total

    def test_grad_wrt_kp_finite(self):
        g = jax.grad(self._step_loss, argnums=0)(
            1.0, 0.5, 0.1, 1.0, 1.0
        )
        assert jnp.isfinite(g), f"grad wrt Kp not finite: {g}"
        # P term contributes b*r - y = 1 every step, so dL/dKp >= n_steps.
        assert float(g) > 0

    def test_grad_wrt_ki_finite(self):
        g = jax.grad(self._step_loss, argnums=1)(
            1.0, 0.5, 0.1, 1.0, 1.0
        )
        assert jnp.isfinite(g), f"grad wrt Ki not finite: {g}"

    def test_grad_wrt_b_finite_and_nonzero(self):
        """``b`` enters the output linearly through the P path."""
        g = jax.grad(self._step_loss, argnums=3)(
            1.0, 0.0, 0.0, 1.0, 1.0
        )
        assert jnp.isfinite(g)
        assert jnp.abs(g) > 0, f"grad wrt b should be nonzero; got {g}"

    def test_grad_wrt_c_finite(self):
        g = jax.grad(self._step_loss, argnums=4)(
            1.0, 0.0, 0.5, 1.0, 1.0
        )
        assert jnp.isfinite(g)


# --------------------------------------------------------------------- #
# Closed-loop tracking
# --------------------------------------------------------------------- #


class TestClosedLoopTracking:
    """Closed-loop integrator plant: integral action drives error to zero."""

    def test_step_setpoint_tracking(self):
        """With a PI controller and an integrator plant, y → r at steady state."""
        dt = 0.05
        builder = jaxonomy.DiagramBuilder()

        # Setpoint = 1, measurement comes from the integrator's output.
        r = builder.add(Constant(1.0, name="r"))
        pid = builder.add(
            PIDController2DOF(
                dt=dt,
                kp=2.0,
                ki=4.0,
                kd=0.0,
                b=1.0,
                c=1.0,
                name="pid",
            )
        )
        plant = builder.add(Integrator(initial_state=0.0, name="plant"))

        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(plant.output_ports[0], pid.input_ports[1])
        builder.connect(pid.output_ports[0], plant.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        recorded = {"y": plant.output_ports[0]}
        results = jaxonomy.simulate(
            diagram, context, (0.0, 5.0), recorded_signals=recorded
        )
        y = results.outputs["y"]
        # Final value should be near the setpoint (1.0).
        assert jnp.abs(y[-1] - 1.0) < 0.05, (
            f"Closed-loop tracking failed; y[-1] = {float(y[-1])}"
        )
