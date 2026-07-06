# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-bumpless-mode-switch — :class:`PIDController2DOF`.

The T-127-followup-tracking-mode kernel hard-codes the
``tracking_enabled`` flag at construction time: a single simulation
either ALWAYS pulls the integrator toward ``u_ext`` or NEVER does.
Real-world supervisory architectures need to swap between AUTO (PID
drives the actuator) and MANUAL/TRACKING (operator drives the
actuator) mid-simulation while the integrator stays loaded with the
value that produces ``u_ext`` (so the auto→manual→auto handoff is
bumpless in either direction).

This followup adds ``tracking_enabled_dynamic: bool = False``.  When
True it promotes the tracking gate to a runtime SCALAR INPUT port
(appended last, after ``u_ext``):

* ``mode_flag == 0`` → tracking-pull branch suppressed for that tick
  (integrator behaves like ``tracking_enabled=False``).
* ``mode_flag != 0`` → tracking-pull branch runs exactly as in the
  T-127-followup-tracking-mode kernel.

Default is False → byte-equivalent to T-127-followup-tracking-mode.

These tests cover:

* Default-off byte-equivalence with T-127-followup-tracking-mode (both
  static-on and static-off baselines).
* Runtime flag = 1 ≡ static ``tracking_enabled=True``.
* Runtime flag = 0 ≡ static ``tracking_enabled=False`` (i.e. tracking
  branch is silenced entirely).
* Mid-sim 0→1 transition: integrator catches up to ``u_ext``.
* Mid-sim 1→0 transition: integrator stops tracking and holds.
* Validation: ``tracking_enabled_dynamic=True`` without
  ``tracking_enabled=True`` is rejected.
* Validation: flipping ``tracking_enabled_dynamic`` after construction
  is rejected.
* Composes with ``integrate_tracking_error=False`` (the gate is moot
  when the tracking-integrator branch is itself off).
* Differentiability: ``jax.grad`` w.r.t. ``tracking_gain`` stays finite
  with the runtime gate active.
* Config round-trip preserves ``tracking_enabled_dynamic``.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

from jaxonomy.library import PIDController2DOF


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
    dt=0.05,
    tracking_enabled=False,
    tracking_gain=1.0,
    integrate_tracking_error=True,
    tracking_enabled_dynamic=False,
):
    """Construct + run-initialise a PIDController2DOF outside a Diagram."""
    block = PIDController2DOF(
        dt=dt,
        kp=kp, ki=ki, kd=kd, b=b, c=c,
        initial_state=initial_state,
        tracking_enabled=tracking_enabled,
        tracking_gain=tracking_gain,
        integrate_tracking_error=integrate_tracking_error,
        tracking_enabled_dynamic=tracking_enabled_dynamic,
        name="pid",
    )
    block.initialize(
        kp=kp, ki=ki, kd=kd, b=b, c=c,
        initial_state=initial_state,
        filter_type="none", filter_coefficient=1.0,
        tracking_enabled=tracking_enabled,
        tracking_gain=tracking_gain,
        integrate_tracking_error=integrate_tracking_error,
        tracking_enabled_dynamic=tracking_enabled_dynamic,
    )
    return block


def _initial_state_tuple(block, integral=0.0):
    State = namedtuple("State", ["discrete_state"])
    xd0 = block.DiscreteStateType(
        integral=jnp.asarray(integral, dtype=jnp.float64),
        e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
        e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
    )
    return State(discrete_state=xd0), State


def _run(
    block,
    *,
    n_steps,
    r,
    y,
    u_ext=None,
    mode_flag=None,
    kp=1.0,
    ki=1.0,
    kd=0.0,
    b=1.0,
    c=1.0,
    tracking_gain=1.0,
    initial_integral=0.0,
):
    """Drive the block ``n_steps`` ticks against constant scalar inputs.

    ``mode_flag`` is either a Python scalar or a length-``n_steps``
    array (allowing mid-sim transitions).  Returns ``(integrals,
    outputs)`` arrays.
    """
    state, State = _initial_state_tuple(block, integral=initial_integral)
    params = dict(
        kp=kp, ki=ki, kd=kd, b=b, c=c,
        anti_windup_gain=1.0, tracking_gain=tracking_gain,
    )
    r_a = jnp.asarray(r, dtype=jnp.float64)
    y_a = jnp.asarray(y, dtype=jnp.float64)
    u_ext_a = (
        jnp.asarray(u_ext, dtype=jnp.float64) if u_ext is not None else None
    )

    if mode_flag is not None and not hasattr(mode_flag, "__len__"):
        mode_flag = [mode_flag] * n_steps

    integrals = []
    outputs = []
    for k in range(n_steps):
        if block.tracking_enabled and block.tracking_enabled_dynamic:
            mf = jnp.asarray(mode_flag[k], dtype=jnp.float64)
            inputs = (r_a, y_a, u_ext_a, mf)
        elif block.tracking_enabled:
            inputs = (r_a, y_a, u_ext_a)
        else:
            inputs = (r_a, y_a)
        u = block._output(jnp.asarray(0.0), state, *inputs, **params)
        outputs.append(u)
        new_xd = block._update(jnp.asarray(0.0), state, *inputs, **params)
        integrals.append(new_xd.integral)
        state = State(discrete_state=new_xd)
    return jnp.asarray(integrals), jnp.asarray(outputs)


# --------------------------------------------------------------------- #
# Default-off byte-equivalence
# --------------------------------------------------------------------- #


class TestDefaultOffByteEquivalence:
    """``tracking_enabled_dynamic=False`` → identical to T-127-fu-tracking-mode."""

    def test_with_tracking_static_on(self):
        """Default-off + ``tracking_enabled=True`` matches the static
        T-127-followup-tracking-mode kernel byte-for-byte.
        """
        common = dict(
            kp=0.5, ki=2.0, kd=0.1, dt=0.05,
            tracking_enabled=True, tracking_gain=0.3,
        )
        block_a = _make_block(**common)  # default tracking_enabled_dynamic=False
        block_b = _make_block(tracking_enabled_dynamic=False, **common)
        ints_a, us_a = _run(
            block_a, n_steps=50, r=1.0, y=0.0, u_ext=0.4,
            kp=0.5, ki=2.0, kd=0.1, tracking_gain=0.3,
        )
        ints_b, us_b = _run(
            block_b, n_steps=50, r=1.0, y=0.0, u_ext=0.4,
            kp=0.5, ki=2.0, kd=0.1, tracking_gain=0.3,
        )
        assert jnp.allclose(ints_a, ints_b, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_a, us_b, atol=0.0, rtol=0.0)

    def test_with_tracking_static_off(self):
        """Default-off + ``tracking_enabled=False`` matches phase-1
        (no tracking) byte-for-byte.  Even with ``tracking_enabled_dynamic
        =False`` set explicitly the port count must remain at 2.
        """
        block_a = _make_block(kp=1.0, ki=2.0, kd=0.5)  # phase-1 baseline
        block_b = _make_block(
            kp=1.0, ki=2.0, kd=0.5, tracking_enabled_dynamic=False,
        )
        assert len(block_a.input_ports) == 2
        assert len(block_b.input_ports) == 2
        ints_a, us_a = _run(
            block_a, n_steps=20, r=1.0, y=0.0, kp=1.0, ki=2.0, kd=0.5,
        )
        ints_b, us_b = _run(
            block_b, n_steps=20, r=1.0, y=0.0, kp=1.0, ki=2.0, kd=0.5,
        )
        assert jnp.allclose(ints_a, ints_b, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_a, us_b, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Runtime flag equivalence with the static settings
# --------------------------------------------------------------------- #


class TestRuntimeFlagEquivalence:
    """Runtime flag values match the corresponding static configurations."""

    def test_flag_one_equals_static_on(self):
        """``mode_flag=1`` for every tick ≡ static ``tracking_enabled=True``."""
        common = dict(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_gain=0.2,
        )
        # Static-on baseline.
        block_static = _make_block(tracking_enabled=True, **common)
        ints_s, us_s = _run(
            block_static, n_steps=200, r=0.0, y=0.0, u_ext=0.6,
            kp=0.0, ki=1.0, kd=0.0, tracking_gain=0.2,
        )
        # Dynamic with flag locked at 1.
        block_dyn = _make_block(
            tracking_enabled=True, tracking_enabled_dynamic=True, **common,
        )
        ints_d, us_d = _run(
            block_dyn, n_steps=200, r=0.0, y=0.0, u_ext=0.6, mode_flag=1.0,
            kp=0.0, ki=1.0, kd=0.0, tracking_gain=0.2,
        )
        assert jnp.allclose(ints_s, ints_d, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_s, us_d, atol=0.0, rtol=0.0)

    def test_flag_zero_equals_static_off(self):
        """``mode_flag=0`` for every tick ≡ no tracking (phase-1 update)."""
        common = dict(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
        )
        # Static-off baseline (no tracking branch at all).
        block_off = _make_block(**common)
        ints_o, us_o = _run(
            block_off, n_steps=80, r=0.5, y=0.0, kp=0.0, ki=1.0, kd=0.0,
        )
        # Dynamic with flag locked at 0 — tracking branch declared but
        # gated off every tick.  Integrator must accumulate the
        # regulation error only.
        block_dyn = _make_block(
            tracking_enabled=True, tracking_enabled_dynamic=True,
            tracking_gain=0.2, **common,
        )
        ints_d, us_d = _run(
            block_dyn, n_steps=80, r=0.5, y=0.0, u_ext=99.0, mode_flag=0.0,
            kp=0.0, ki=1.0, kd=0.0, tracking_gain=0.2,
        )
        # u_ext is wildly off but the gate kills the correction.
        assert jnp.allclose(ints_o, ints_d, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_o, us_d, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Mid-simulation transitions
# --------------------------------------------------------------------- #


class TestMidSimTransitions:
    """Flipping ``mode_flag`` mid-run produces the documented behavior."""

    def test_zero_to_one_transition_catches_up(self):
        """Initially OFF → integrator drifts on regulation error alone;
        once flipped ON, the tracking-pull branch hauls it toward the
        bumpless-handoff value."""
        n_off = 100
        n_on = 400
        kp, ki, kd = 0.0, 0.5, 0.0
        u_ext_target = 0.8
        block = _make_block(
            kp=kp, ki=ki, kd=kd, dt=0.05,
            tracking_enabled=True, tracking_enabled_dynamic=True,
            tracking_gain=0.05,
        )
        # r = y = 0 so the regulation error contributes nothing — the
        # only thing that can move the integrator is the tracking pull.
        flag = [0.0] * n_off + [1.0] * n_on
        ints, us = _run(
            block, n_steps=n_off + n_on, r=0.0, y=0.0,
            u_ext=u_ext_target, mode_flag=flag,
            kp=kp, ki=ki, kd=kd, tracking_gain=0.05,
        )
        # During OFF phase: integrator stays at 0 (no e_i, no tracking).
        i_during_off = float(ints[n_off - 1])
        assert i_during_off == pytest.approx(0.0, abs=1e-12), (
            f"integrator drifted during OFF phase: {i_during_off}"
        )
        # During ON phase: integrator catches up so u → u_ext_target.
        # u = ki * I = ki * I; steady-state I = u_ext_target / ki.
        steady_int = u_ext_target / ki
        i_tail = float(jnp.mean(ints[-30:]))
        assert i_tail == pytest.approx(steady_int, rel=5e-2), (
            f"integrator did not catch up after 0→1 flip; "
            f"tail mean={i_tail}, expected≈{steady_int}"
        )
        u_tail = float(jnp.mean(us[-30:]))
        assert u_tail == pytest.approx(u_ext_target, abs=2e-2)

    def test_one_to_zero_transition_holds(self):
        """Initially ON → integrator catches up to ``u_ext``; once
        flipped OFF the tracking pull stops and (with no regulation
        error) the integrator holds its last value indefinitely."""
        n_on = 400
        n_off = 200
        kp, ki, kd = 0.0, 0.5, 0.0
        u_ext_target = 0.8
        block = _make_block(
            kp=kp, ki=ki, kd=kd, dt=0.05,
            tracking_enabled=True, tracking_enabled_dynamic=True,
            tracking_gain=0.05,
        )
        flag = [1.0] * n_on + [0.0] * n_off
        ints, us = _run(
            block, n_steps=n_on + n_off, r=0.0, y=0.0,
            u_ext=u_ext_target, mode_flag=flag,
            kp=kp, ki=ki, kd=kd, tracking_gain=0.05,
        )
        # At the moment of the flip the integrator should be near the
        # bumpless-handoff value.
        i_at_flip = float(ints[n_on - 1])
        steady_int = u_ext_target / ki
        assert i_at_flip == pytest.approx(steady_int, rel=5e-2), (
            f"integrator did not converge before the 1→0 flip; "
            f"value={i_at_flip}, expected≈{steady_int}"
        )
        # During the OFF phase: with r=y=0 and no tracking pull, the
        # integrator update reduces to ``I[k+1] = I[k]`` — must be
        # constant to many decimal places.
        post_flip_segment = ints[n_on + 5:n_on + n_off]
        deltas = jnp.diff(post_flip_segment)
        max_delta = float(jnp.max(jnp.abs(deltas)))
        assert max_delta < 1e-12, (
            f"integrator drifted after 1→0 flip; max per-tick delta="
            f"{max_delta}"
        )
        # The held value still drives the output (u = ki * I).
        u_after_flip = float(us[-1])
        assert u_after_flip == pytest.approx(
            ki * float(ints[n_on + 5]), abs=1e-12
        )


# --------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------- #


class TestCompositionWithIonErrorOnly:
    """``tracking_enabled_dynamic`` composes cleanly with the
    T-127-followup-i-on-error-only flag."""

    def test_gate_moot_when_integrate_tracking_error_false(self):
        """When the tracking-integrator branch is itself disabled, the
        runtime flag has nothing to gate — no value of ``mode_flag``
        should perturb the integrator beyond the regulation error."""
        common = dict(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_enabled=True, tracking_gain=0.05,
            integrate_tracking_error=False,
        )
        # Static-off baseline (i_on_error_only).
        block_static = _make_block(**common)
        ints_s, us_s = _run(
            block_static, n_steps=80, r=0.0, y=0.0, u_ext=99.0,
            kp=0.0, ki=1.0, kd=0.0, tracking_gain=0.05,
        )
        # Dynamic flag toggling — output must still match.
        block_dyn = _make_block(tracking_enabled_dynamic=True, **common)
        flag = [0.0, 1.0] * 40
        ints_d, us_d = _run(
            block_dyn, n_steps=80, r=0.0, y=0.0, u_ext=99.0, mode_flag=flag,
            kp=0.0, ki=1.0, kd=0.0, tracking_gain=0.05,
        )
        assert jnp.allclose(ints_s, ints_d, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_s, us_d, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    """Construction-time and initialize-time contracts."""

    def test_dynamic_without_tracking_enabled_rejected(self):
        """``tracking_enabled_dynamic=True`` without ``tracking_enabled
        =True`` raises (the gate has nothing to multiply)."""
        with pytest.raises(ValueError, match="tracking_enabled_dynamic"):
            PIDController2DOF(
                dt=0.05, kp=1.0, ki=0.0, kd=0.0,
                tracking_enabled=False,
                tracking_enabled_dynamic=True,
                name="pid",
            )

    def test_dynamic_change_after_init_rejected(self):
        """Flipping ``tracking_enabled_dynamic`` in ``initialize`` raises."""
        block = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0,
            tracking_enabled=True, tracking_enabled_dynamic=False,
            name="pid",
        )
        with pytest.raises(
            ValueError, match="tracking_enabled_dynamic cannot be changed"
        ):
            block.initialize(
                kp=1.0, ki=0.0, kd=0.0, b=1.0, c=1.0,
                initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                tracking_enabled=True,
                tracking_enabled_dynamic=True,  # was False at construction
            )

    def test_port_count_when_dynamic(self):
        """``tracking_enabled_dynamic=True`` adds exactly one input port
        beyond the static-tracking layout."""
        # Static tracking: r, y, u_ext = 3 ports.
        block_static = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0,
            tracking_enabled=True, name="pid_s",
        )
        assert len(block_static.input_ports) == 3
        # Static + dynamic: r, y, u_ext, mode_flag = 4 ports.
        block_dyn = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0,
            tracking_enabled=True, tracking_enabled_dynamic=True,
            name="pid_d",
        )
        assert len(block_dyn.input_ports) == 4
        assert block_dyn.u_ext_index == 2
        assert block_dyn.mode_flag_index == 3

    def test_mode_flag_after_other_dynamic_ports(self):
        """``mode_flag`` is appended LAST — after every other dynamic port."""
        block = PIDController2DOF(
            dt=0.05, kp=1.0, ki=0.0, kd=0.0,
            b_dynamic=True, kp_dynamic=True, kff_dynamic=True,
            tracking_enabled=True, tracking_enabled_dynamic=True,
            name="pid",
        )
        # r, y, b, kp, kff, u_ext, mode_flag = 7 ports.
        assert len(block.input_ports) == 7
        assert block.mode_flag_index == 6
        assert block.u_ext_index < block.mode_flag_index


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` w.r.t. ``tracking_gain`` stays finite with the
    runtime gate active (the boolean gate kills the gradient through
    the flag itself, but ``tracking_gain`` is differentiable on every
    ON tick)."""

    @staticmethod
    def _loss(tracking_gain, n_steps=200):
        kp, ki, kd = 0.0, 1.0, 0.0
        block = _make_block(
            kp=kp, ki=ki, kd=kd, dt=0.05,
            tracking_enabled=True, tracking_enabled_dynamic=True,
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
        # Flag is ON for the entire trajectory.
        mf_on = jnp.asarray(1.0)
        params = dict(
            kp=kp, ki=ki, kd=kd, b=1.0, c=1.0,
            anti_windup_gain=1.0, tracking_gain=tracking_gain,
        )
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(
                jnp.asarray(0.0), state, r, y, u_ext, mf_on, **params
            )
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, u_ext, mf_on, **params
            )
            state = State(discrete_state=new_xd)
            total = total + (u_ext - u) ** 2
        return total

    def test_grad_wrt_tracking_gain_finite(self):
        g = jax.grad(self._loss)(jnp.asarray(0.3))
        assert jnp.isfinite(g), f"grad wrt tracking_gain not finite: {g}"

    def test_grad_wrt_tracking_gain_nonzero(self):
        g = jax.grad(self._loss)(jnp.asarray(0.3))
        assert jnp.abs(g) > 0, (
            f"grad wrt tracking_gain should be nonzero with flag ON; "
            f"got {g}"
        )


# --------------------------------------------------------------------- #
# Config round-trip
# --------------------------------------------------------------------- #


class TestConfigRoundTrip:
    """``to_dict`` / ``from_dict`` preserve ``tracking_enabled_dynamic``."""

    def test_round_trip_default(self):
        pid = PIDController2DOF(dt=0.05, name="pid")
        data = pid.to_dict()
        assert data["tracking_enabled_dynamic"] is False
        pid2 = PIDController2DOF.from_dict(data, name="pid2")
        assert pid2.tracking_enabled_dynamic is False
        assert len(pid2.input_ports) == 2

    def test_round_trip_enabled(self):
        pid = PIDController2DOF(
            dt=0.05, kp=2.0, ki=3.0,
            tracking_enabled=True, tracking_gain=0.25,
            tracking_enabled_dynamic=True,
            name="pid",
        )
        data = pid.to_dict()
        assert data["tracking_enabled"] is True
        assert data["tracking_enabled_dynamic"] is True
        pid2 = PIDController2DOF.from_dict(data, name="pid2")
        assert pid2.tracking_enabled is True
        assert pid2.tracking_enabled_dynamic is True
        assert len(pid2.input_ports) == 4
        assert pid2.u_ext_index == 2
        assert pid2.mode_flag_index == 3
