# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-external-weights — :class:`PIDController2DOF`.

Phase 1 of T-127 (and the T-127-followup-anti-windup followup) treated
the setpoint weights ``b`` (proportional) and ``c`` (derivative) as
construction-time scalars routed through ``@parameters(dynamic=...)``.

In a real-world controller these weights are sometimes scheduled — for
example, ``b`` is faded from 0 to 1 as the system approaches its target
to avoid setpoint kicks while still recovering full proportional action
once the loop is locked.  This followup adds two new construction
kwargs, ``b_dynamic`` and ``c_dynamic``, that mirror the
``enable_dynamic_*`` pattern used by :class:`Saturate` and
:class:`RateLimiter`.  When set, the corresponding weight is read from a
new input port instead of the static parameter.

Port indexing:

* (0) Setpoint ``r``.
* (1) Measurement ``y``.
* (2) ``b`` if ``b_dynamic=True``.
* (next) ``c`` if ``c_dynamic=True`` — index 2 when only ``c_dynamic``
  is set, index 3 when both are set.

Default-off (``b_dynamic=False, c_dynamic=False``) must remain byte-
equivalent to phase 1.

These tests cover:

* Default-off byte-equivalence with phase 1 on a closed-loop tracking
  scenario (no new ports, no behaviour change).
* ``b_dynamic=True`` with the new port wired to a ``Constant(0.5)``
  matches the static-``b=0.5`` reference.
* ``b_dynamic=True`` with the new port wired to a ``Step(0 → 1)``
  reflects the scheduled weight: pre-step the proportional path is
  inert, post-step it tracks ``Kp * r``.
* ``c_dynamic=True`` (alone): same equivalence with port at index 2.
* Both ``b_dynamic=True`` and ``c_dynamic=True``: ports land at indices
  2 and 3, weight values are correctly threaded.
* Differentiability: ``jax.grad(integrated_output)(b_value)`` is finite
  when ``b`` flows through the runtime port.
* Validation: flipping ``b_dynamic`` / ``c_dynamic`` after construction
  is rejected (port topology is locked at build time).
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


def _simulate_open_loop_static(
    *,
    dt=0.1,
    r=1.0,
    y=0.0,
    kp=2.0,
    ki=0.0,
    kd=0.0,
    b=1.0,
    c=1.0,
    t_end=1.0,
):
    """Static-weight reference: phase-1 path."""
    builder = jaxonomy.DiagramBuilder()
    r_block = builder.add(Constant(r, name="r"))
    y_block = builder.add(Constant(y, name="y"))
    pid = builder.add(
        PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            b=b,
            c=c,
            name="pid",
        )
    )
    builder.connect(r_block.output_ports[0], pid.input_ports[0])
    builder.connect(y_block.output_ports[0], pid.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.time


def _simulate_open_loop_dynamic_b(
    *,
    b_block,
    dt=0.1,
    r=1.0,
    y=0.0,
    kp=2.0,
    ki=0.0,
    kd=0.0,
    c=1.0,
    t_end=1.0,
):
    """Dynamic-``b`` variant: ``b`` is read from an input port."""
    builder = jaxonomy.DiagramBuilder()
    r_block = builder.add(Constant(r, name="r"))
    y_block = builder.add(Constant(y, name="y"))
    b_src = builder.add(b_block)
    pid = builder.add(
        PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            b=999.0,  # ignored (b_dynamic=True)
            c=c,
            b_dynamic=True,
            name="pid",
        )
    )
    builder.connect(r_block.output_ports[0], pid.input_ports[0])
    builder.connect(y_block.output_ports[0], pid.input_ports[1])
    builder.connect(b_src.output_ports[0], pid.input_ports[2])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.time


def _simulate_open_loop_dynamic_c(
    *,
    c_block,
    dt=0.1,
    r=1.0,
    y=0.0,
    kp=0.0,
    ki=0.0,
    kd=0.5,
    b=1.0,
    t_end=1.0,
):
    """Dynamic-``c`` variant: ``c`` port at index 2 (b_dynamic=False)."""
    builder = jaxonomy.DiagramBuilder()
    r_block = builder.add(Constant(r, name="r"))
    y_block = builder.add(Constant(y, name="y"))
    c_src = builder.add(c_block)
    pid = builder.add(
        PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            b=b,
            c=999.0,  # ignored (c_dynamic=True)
            c_dynamic=True,
            name="pid",
        )
    )
    builder.connect(r_block.output_ports[0], pid.input_ports[0])
    builder.connect(y_block.output_ports[0], pid.input_ports[1])
    builder.connect(c_src.output_ports[0], pid.input_ports[2])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.time


# --------------------------------------------------------------------- #
# Default-off byte-equivalence
# --------------------------------------------------------------------- #


class TestDefaultOff:
    """``b_dynamic=False, c_dynamic=False`` is byte-equivalent to phase 1."""

    def test_open_loop_default_matches_phase1(self):
        """Default kwargs leave the open-loop PID output unchanged."""
        u_static, _ = _simulate_open_loop_static(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1, t_end=2.0
        )
        # Re-build with the new kwargs explicitly defaulted -- the block
        # path must produce *exactly* the same output (no new ports, no
        # behavior change).
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05,
                kp=2.0,
                ki=4.0,
                kd=0.1,
                b=1.0,
                c=1.0,
                b_dynamic=False,
                c_dynamic=False,
                name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        diagram = builder.build()
        context = diagram.create_context()
        recorded = {"u": pid.output_ports[0]}
        results = jaxonomy.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded
        )
        u_explicit_off = results.outputs["u"]
        assert jnp.array_equal(u_static, u_explicit_off), (
            "Default-off path must be byte-equivalent to the implicit "
            "phase 1 default"
        )

    def test_default_off_does_not_add_input_ports(self):
        """Without the flags, only the (r, y) ports exist."""
        pid = PIDController2DOF(dt=0.1, name="pid")
        # Phase 1 declared exactly two input ports.
        assert len(pid.input_ports) == 2
        assert pid.b_dynamic is False
        assert pid.c_dynamic is False


# --------------------------------------------------------------------- #
# b_dynamic — runtime port matches static b
# --------------------------------------------------------------------- #


class TestBDynamicPort:
    """``b_dynamic=True`` reads ``b`` from a runtime port."""

    def test_b_port_adds_third_input(self):
        """``b_dynamic=True`` adds exactly one new input port at index 2."""
        pid = PIDController2DOF(dt=0.1, b_dynamic=True, name="pid")
        assert len(pid.input_ports) == 3
        assert pid.b_index == 2

    def test_b_constant_matches_static_b(self):
        """A Constant(0.5) port equals the static-``b=0.5`` reference."""
        # Use a P-only controller so b is the only knob that matters.
        u_static, _ = _simulate_open_loop_static(
            dt=0.05, r=2.0, y=0.0, kp=3.0, ki=0.0, kd=0.0, b=0.5, t_end=1.0
        )
        u_dyn, _ = _simulate_open_loop_dynamic_b(
            b_block=Constant(0.5, name="b_signal"),
            dt=0.05, r=2.0, y=0.0, kp=3.0, ki=0.0, kd=0.0, t_end=1.0,
        )
        assert u_static.shape == u_dyn.shape
        assert jnp.allclose(u_static, u_dyn, atol=1e-12), (
            "Dynamic-port b=0.5 must match static b=0.5 "
            f"(max diff {float(jnp.max(jnp.abs(u_static - u_dyn)))})"
        )

    def test_b_step_reflects_schedule(self):
        """``b`` stepped 0→1 mid-simulation: u tracks the schedule."""
        # Pure P controller, r = 4, y = 0.
        # Pre-step (b=0): u = kp*(0*r - y) = 0.
        # Post-step (b=1): u = kp*(1*r - y) = kp * r = 6.0.
        kp = 1.5
        r = 4.0
        u_dyn, t = _simulate_open_loop_dynamic_b(
            b_block=Step(start_value=0.0, end_value=1.0, step_time=0.5),
            dt=0.05,
            r=r,
            y=0.0,
            kp=kp,
            ki=0.0,
            kd=0.0,
            t_end=1.0,
        )
        # Find one sample well before and well after the step.
        t_arr = jnp.asarray(t)
        # Use samples >= 0.1s into the simulation but < 0.5s.
        pre_mask = (t_arr >= 0.1) & (t_arr < 0.5)
        post_mask = t_arr > 0.55
        assert jnp.any(pre_mask), "no pre-step samples found"
        assert jnp.any(post_mask), "no post-step samples found"

        u_pre = u_dyn[pre_mask]
        u_post = u_dyn[post_mask]
        assert jnp.allclose(u_pre, 0.0, atol=1e-10), (
            f"Pre-step (b=0): expected u≈0, got {u_pre}"
        )
        assert jnp.allclose(u_post, kp * r, atol=1e-10), (
            f"Post-step (b=1): expected u≈{kp*r}, got {u_post}"
        )


# --------------------------------------------------------------------- #
# c_dynamic — runtime port matches static c
# --------------------------------------------------------------------- #


class TestCDynamicPort:
    """``c_dynamic=True`` reads ``c`` from a runtime port."""

    def test_c_only_port_at_index_2(self):
        """When only ``c_dynamic`` is set, the c port lives at index 2."""
        pid = PIDController2DOF(
            dt=0.1, b_dynamic=False, c_dynamic=True, name="pid"
        )
        assert len(pid.input_ports) == 3
        assert pid.c_index == 2

    def test_c_constant_matches_static_c(self):
        """A Constant(0.3) c port matches static-``c=0.3``.

        Use a D-only controller with a sinusoidal-looking r so the
        derivative path is exercised.
        """
        # D-only on a constant r/y is uninteresting (derivative is 0).
        # Use a Sine setpoint to exercise the derivative path; both
        # paths should still match exactly because the rest of the
        # block is shared.
        from jaxonomy.library import Sine
        dt = 0.05
        # Static reference.
        builder = jaxonomy.DiagramBuilder()
        r_blk = builder.add(Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        y_blk = builder.add(Constant(0.0, name="y"))
        pid_static = builder.add(
            PIDController2DOF(
                dt=dt, kp=0.0, ki=0.0, kd=0.5, b=1.0, c=0.3, name="pid_s"
            )
        )
        builder.connect(r_blk.output_ports[0], pid_static.input_ports[0])
        builder.connect(y_blk.output_ports[0], pid_static.input_ports[1])
        diagram = builder.build()
        context = diagram.create_context()
        results_s = jaxonomy.simulate(
            diagram, context, (0.0, 1.0),
            recorded_signals={"u": pid_static.output_ports[0]},
        )

        u_dyn, _ = _simulate_open_loop_dynamic_c(
            c_block=Constant(0.3, name="c_signal"),
            dt=dt, r=0.0, y=0.0, kp=0.0, ki=0.0, kd=0.5, b=1.0, t_end=1.0,
        )
        # The dynamic-c helper above uses Constant for r, not Sine, so
        # we need to rebuild a Sine-driven version for the equivalence
        # check.
        builder2 = jaxonomy.DiagramBuilder()
        r2 = builder2.add(Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        y2 = builder2.add(Constant(0.0, name="y"))
        c2 = builder2.add(Constant(0.3, name="c_signal"))
        pid_dyn = builder2.add(
            PIDController2DOF(
                dt=dt, kp=0.0, ki=0.0, kd=0.5, b=1.0, c=999.0,
                c_dynamic=True, name="pid_d",
            )
        )
        builder2.connect(r2.output_ports[0], pid_dyn.input_ports[0])
        builder2.connect(y2.output_ports[0], pid_dyn.input_ports[1])
        builder2.connect(c2.output_ports[0], pid_dyn.input_ports[2])
        diagram2 = builder2.build()
        context2 = diagram2.create_context()
        results_d = jaxonomy.simulate(
            diagram2, context2, (0.0, 1.0),
            recorded_signals={"u": pid_dyn.output_ports[0]},
        )
        assert jnp.allclose(results_s.outputs["u"], results_d.outputs["u"], atol=1e-12)


# --------------------------------------------------------------------- #
# Both b_dynamic and c_dynamic
# --------------------------------------------------------------------- #


class TestBothDynamic:
    """Both flags active: ports at indices 2 and 3, in (b, c) order."""

    def test_port_order_b_then_c(self):
        pid = PIDController2DOF(
            dt=0.1, b_dynamic=True, c_dynamic=True, name="pid"
        )
        assert len(pid.input_ports) == 4
        assert pid.b_index == 2
        assert pid.c_index == 3

    def test_full_pid_matches_static_reference(self):
        """A full PID with dynamic b and c equals the static reference."""
        dt = 0.05
        r_val, y_val = 1.0, 0.0
        kp, ki, kd = 1.0, 2.0, 0.3
        b_val, c_val = 0.4, 0.7
        # Static.
        u_s, _ = _simulate_open_loop_static(
            dt=dt, r=r_val, y=y_val, kp=kp, ki=ki, kd=kd, b=b_val, c=c_val,
            t_end=1.5,
        )
        # Dynamic with both ports wired to constants.
        builder = jaxonomy.DiagramBuilder()
        r_b = builder.add(Constant(r_val, name="r"))
        y_b = builder.add(Constant(y_val, name="y"))
        b_b = builder.add(Constant(b_val, name="b"))
        c_b = builder.add(Constant(c_val, name="c"))
        pid = builder.add(
            PIDController2DOF(
                dt=dt, kp=kp, ki=ki, kd=kd, b=999.0, c=999.0,
                b_dynamic=True, c_dynamic=True, name="pid",
            )
        )
        builder.connect(r_b.output_ports[0], pid.input_ports[0])
        builder.connect(y_b.output_ports[0], pid.input_ports[1])
        builder.connect(b_b.output_ports[0], pid.input_ports[2])
        builder.connect(c_b.output_ports[0], pid.input_ports[3])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 1.5),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u_d = results.outputs["u"]
        assert jnp.allclose(u_s, u_d, atol=1e-12), (
            f"Static and fully-dynamic PIDs disagree "
            f"(max diff {float(jnp.max(jnp.abs(u_s - u_d)))})"
        )


# --------------------------------------------------------------------- #
# Differentiability through the runtime b port
# --------------------------------------------------------------------- #


class TestDifferentiabilityThroughPort:
    """``jax.grad`` w.r.t. the runtime ``b`` value is finite."""

    @staticmethod
    def _make_block(dt=0.1):
        block = PIDController2DOF(
            dt=dt, kp=1.0, ki=0.5, kd=0.1, b=999.0, c=1.0,
            b_dynamic=True, name="pid",
        )
        block.initialize(
            kp=1.0, ki=0.5, kd=0.1, b=999.0, c=1.0,
            initial_state=0.0, filter_type="none", filter_coefficient=1.0,
            b_dynamic=True, c_dynamic=False,
        )
        return block

    @classmethod
    def _step_loss(cls, b_value, n_steps=4):
        """Run the block for ``n_steps`` ticks with constant r=1, y=0,
        and the runtime-port ``b`` value supplied as the differentiable
        argument."""
        State = namedtuple("State", ["discrete_state"])
        block = cls._make_block()
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0),
            e_d_prev=jnp.asarray(0.0),
            e_dot_prev=jnp.asarray(0.0),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(1.0)
        y = jnp.asarray(0.0)
        params = dict(kp=1.0, ki=0.5, kd=0.1, b=999.0, c=1.0)
        # Inputs are ordered (r, y, b) per the b_index=2 port layout.
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, b_value, **params)
            total = total + jnp.abs(u)
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, b_value, **params
            )
            state = State(discrete_state=new_xd)
        return total

    def test_grad_wrt_b_port_finite_and_nonzero(self):
        """Gradient of integrated |u| w.r.t. the runtime ``b`` port."""
        g = jax.grad(self._step_loss)(jnp.asarray(0.5))
        assert jnp.isfinite(g), f"grad wrt runtime b not finite: {g}"
        # P term contributes b*r each step; varying b changes |u|, so
        # the gradient must be non-zero.
        assert jnp.abs(g) > 0, f"grad wrt runtime b should be nonzero; got {g}"


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    """Topology of dynamic ports is locked at construction time."""

    def test_b_dynamic_change_after_init_rejected(self):
        """Flipping ``b_dynamic`` in ``initialize`` is rejected."""
        block = PIDController2DOF(
            dt=0.1, kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            b_dynamic=False, name="pid",
        )
        with pytest.raises(ValueError, match="b_dynamic cannot be changed"):
            block.initialize(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0, initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                b_dynamic=True,  # was False at construction
            )

    def test_c_dynamic_change_after_init_rejected(self):
        """Flipping ``c_dynamic`` in ``initialize`` is rejected."""
        block = PIDController2DOF(
            dt=0.1, kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            c_dynamic=True, name="pid",
        )
        with pytest.raises(ValueError, match="c_dynamic cannot be changed"):
            block.initialize(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0, initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                c_dynamic=False,  # was True at construction
            )

    def test_b_dynamic_unconnected_port_raises(self):
        """``b_dynamic=True`` with the new port left unconnected is a
        clear build-time error.

        The DiagramBuilder owns input-port-connectedness validation; we
        simply confirm that build/simulate surfaces an exception rather
        than silently using the static ``b`` (or NaN-propagating).
        """
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=1.0, ki=0.0, kd=0.0, b=999.0, c=1.0,
                b_dynamic=True, name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        # Intentionally do NOT connect input_ports[2].
        with pytest.raises(Exception):
            diagram = builder.build()
            context = diagram.create_context()
            jaxonomy.simulate(
                diagram, context, (0.0, 0.2),
                recorded_signals={"u": pid.output_ports[0]},
            )
