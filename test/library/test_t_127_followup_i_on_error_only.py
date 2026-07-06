# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-i-on-error-only — :class:`PIDController2DOF`.

The standard "tracking-mode" architecture (T-127-followup-tracking-mode)
folds *two* corrections into the integrator each tick:

* ``Ki * (r - y) * dt`` — the canonical regulation-error term.
* ``(u_ext - u_unsat) / Tt * dt`` — back-calculation against an
  external tracking signal.

Some users want JUST the first: their feedforward / manual-mode signal
should not perturb the integrator, only the parallel path.  This
followup adds an ``integrate_tracking_error: bool = True`` kwarg that
selects between the two architectures:

* ``True`` (default): T-127-followup-tracking-mode kernel preserved
  byte-for-byte.
* ``False``: tracking integrator term is suppressed entirely; ``u_ext``
  reaches the integrator only via the regulation error.

These tests cover:

* Default-on byte-equivalence with T-127-followup-tracking-mode.
* Off-mode: integrator ignores ``u_ext`` (only ``r - y`` accumulates).
* Composition with anti-windup (saturation-aware integrator still
  works; only the tracking term is silenced).
* Validation: flipping the flag after construction is rejected.
* Config round-trip preserves the flag.
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
    dt=0.1,
    output_min=None,
    output_max=None,
    anti_windup_method="none",
    anti_windup_gain=1.0,
    tracking_enabled=False,
    tracking_gain=1.0,
    integrate_tracking_error=True,
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
        integrate_tracking_error=integrate_tracking_error,
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
        integrate_tracking_error=integrate_tracking_error,
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
    """Tick the block ``n_steps`` times against constant scalar inputs."""
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
# Default-on byte-equivalence with T-127-followup-tracking-mode
# --------------------------------------------------------------------- #


class TestDefaultOnByteEquivalence:
    """``integrate_tracking_error=True`` reproduces T-127-followup-tracking-mode."""

    def test_default_value_is_true(self):
        """The kwarg defaults to True (T-127-followup-tracking-mode behavior)."""
        block = PIDController2DOF(
            dt=0.05, tracking_enabled=True, name="pid",
        )
        assert block.integrate_tracking_error is True

    def test_explicit_true_matches_implicit_default(self):
        """Explicitly passing ``True`` is bit-equal to omitting the kwarg."""
        # Block A: default (no integrate_tracking_error kwarg).
        block_a = PIDController2DOF(
            dt=0.05, kp=1.0, ki=2.0, kd=0.5,
            tracking_enabled=True, tracking_gain=0.5, name="a",
        )
        block_a.initialize(
            kp=1.0, ki=2.0, kd=0.5, b=1.0, c=1.0, initial_state=0.0,
            filter_type="none", filter_coefficient=1.0,
            tracking_enabled=True, tracking_gain=0.5,
        )
        # Block B: explicit integrate_tracking_error=True.
        block_b = _make_block(
            kp=1.0, ki=2.0, kd=0.5, dt=0.05,
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=True,
        )
        ints_a, us_a = _run_steps(
            block_a, r=1.0, y=0.0, u_ext=0.3,
            n_steps=50, kp=1.0, ki=2.0, kd=0.5,
            tracking_gain=0.5,
        )
        ints_b, us_b = _run_steps(
            block_b, r=1.0, y=0.0, u_ext=0.3,
            n_steps=50, kp=1.0, ki=2.0, kd=0.5,
            tracking_gain=0.5,
        )
        assert jnp.allclose(ints_a, ints_b, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_a, us_b, atol=0.0, rtol=0.0)

    def test_byte_equivalent_with_phase1_when_tracking_off(self):
        """``tracking_enabled=False`` still byte-equal regardless of flag."""
        # Even if integrate_tracking_error is set to False, with tracking
        # disabled the tracking branch is entirely skipped (the `tr_on`
        # gate fires before the new sub-gate); so behavior must match
        # phase 1.
        block_default = _make_block(
            kp=1.0, ki=2.0, kd=0.5, dt=0.05,
            tracking_enabled=False, integrate_tracking_error=True,
        )
        block_flag_off = _make_block(
            kp=1.0, ki=2.0, kd=0.5, dt=0.05,
            tracking_enabled=False, integrate_tracking_error=False,
        )
        ints_a, us_a = _run_steps(
            block_default, r=1.0, y=0.0,
            n_steps=20, kp=1.0, ki=2.0, kd=0.5,
        )
        ints_b, us_b = _run_steps(
            block_flag_off, r=1.0, y=0.0,
            n_steps=20, kp=1.0, ki=2.0, kd=0.5,
        )
        assert jnp.allclose(ints_a, ints_b, atol=0.0, rtol=0.0)
        assert jnp.allclose(us_a, us_b, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------- #
# Off-mode: integrator ignores u_ext
# --------------------------------------------------------------------- #


class TestIntegratorIgnoresUExt:
    """``integrate_tracking_error=False`` → ``u_ext`` does NOT pull the integrator."""

    def test_integrator_independent_of_u_ext_value(self):
        """With the flag off, varying ``u_ext`` leaves the integrator unchanged.

        With ``r = y = 0`` and ``Kp = Kd = 0``, the only way ``u_ext``
        could touch the integrator is via the tracking-error term.  When
        that term is gated off, the integrator must accumulate exactly
        ``Ki*(r-y)*dt = 0`` regardless of ``u_ext``.
        """
        # Two parallel runs with different u_ext.
        block_a = _make_block(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=False,
        )
        block_b = _make_block(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=False,
        )
        ints_a, _ = _run_steps(
            block_a, r=0.0, y=0.0, u_ext=0.0,
            n_steps=200, kp=0.0, ki=1.0, kd=0.0,
            tracking_gain=0.5,
        )
        ints_b, _ = _run_steps(
            block_b, r=0.0, y=0.0, u_ext=5.0,  # very different u_ext
            n_steps=200, kp=0.0, ki=1.0, kd=0.0,
            tracking_gain=0.5,
        )
        # Both should be identically zero everywhere (Ki*0 + 0).
        assert jnp.allclose(ints_a, 0.0, atol=1e-12, rtol=0.0)
        assert jnp.allclose(ints_b, 0.0, atol=1e-12, rtol=0.0)
        assert jnp.allclose(ints_a, ints_b, atol=0.0, rtol=0.0)

    def test_integrator_accumulates_regulation_error_only(self):
        """``r - y`` is the ONLY signal the integrator sees.

        With ``Ki = 1``, ``r = 1``, ``y = 0``, ``dt = 0.05``, the
        integrator should grow as ``k * 0.05`` after ``k`` ticks
        (forward-Euler on a constant unit error) — completely
        independent of ``u_ext``.
        """
        block_low = _make_block(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=False,
        )
        block_high = _make_block(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=False,
        )
        n_steps = 50
        ints_low, _ = _run_steps(
            block_low, r=1.0, y=0.0, u_ext=-10.0,
            n_steps=n_steps, kp=0.0, ki=1.0, kd=0.0,
            tracking_gain=0.5,
        )
        ints_high, _ = _run_steps(
            block_high, r=1.0, y=0.0, u_ext=10.0,
            n_steps=n_steps, kp=0.0, ki=1.0, kd=0.0,
            tracking_gain=0.5,
        )
        # Identical trajectories regardless of u_ext.
        assert jnp.allclose(ints_low, ints_high, atol=0.0, rtol=0.0)
        # Closed form: after k ticks (1-indexed in our list) the
        # integrator equals k * Ki * (r-y) * dt = k * 0.05.
        expected = jnp.arange(1, n_steps + 1, dtype=jnp.float64) * 0.05
        assert jnp.allclose(ints_low, expected, atol=1e-12, rtol=0.0)

    def test_on_vs_off_differs_when_u_ext_nonzero(self):
        """``True`` and ``False`` produce distinct trajectories when u_ext != u_unsat.

        Sanity-check: the flag is not a no-op.  With ``True`` the
        integrator gets pulled by ``u_ext``; with ``False`` it does not.
        """
        common = dict(
            kp=0.0, ki=1.0, kd=0.0, dt=0.05,
            tracking_enabled=True, tracking_gain=0.5,
        )
        block_on = _make_block(integrate_tracking_error=True, **common)
        block_off = _make_block(integrate_tracking_error=False, **common)
        ints_on, _ = _run_steps(
            block_on, r=0.0, y=0.0, u_ext=0.7,
            n_steps=50, kp=0.0, ki=1.0, kd=0.0,
            tracking_gain=0.5,
        )
        ints_off, _ = _run_steps(
            block_off, r=0.0, y=0.0, u_ext=0.7,
            n_steps=50, kp=0.0, ki=1.0, kd=0.0,
            tracking_gain=0.5,
        )
        # On-mode: integrator pulled toward a value that produces u_ext.
        # Off-mode: integrator stays at 0 (e_i = 0 everywhere).
        assert not jnp.allclose(ints_on, ints_off, atol=1e-6, rtol=0.0)
        assert jnp.allclose(ints_off, 0.0, atol=1e-12, rtol=0.0)
        # On-mode integrator should be nonzero (positive: u_ext = 0.7 > 0).
        assert float(ints_on[-1]) > 0.1


# --------------------------------------------------------------------- #
# Composition with anti-windup
# --------------------------------------------------------------------- #


class TestComposeWithAntiWindup:
    """Anti-windup integrator correction still works when tracking-int is gated off."""

    def test_anti_windup_unaffected_by_flag(self):
        """``integrate_tracking_error=False`` does NOT suppress anti-windup.

        Anti-windup operates via ``u_sat`` (the saturated output), not
        ``u_ext`` — the two corrections are independent contributions
        and the new flag must only silence the tracking one.
        """
        common = dict(
            kp=1.0, ki=4.0, kd=0.0, dt=0.05,
            output_min=-1.0, output_max=1.0,
            anti_windup_method="back_calc", anti_windup_gain=0.2,
        )
        # Anti-windup only (no tracking at all): baseline.
        block_baseline = _make_block(**common)
        ints_base, _ = _run_steps(
            block_baseline, r=5.0, y=0.0,
            n_steps=200, kp=1.0, ki=4.0, kd=0.0,
            output_min=-1.0, output_max=1.0,
            anti_windup_gain=0.2,
        )
        # Anti-windup + tracking_enabled=True + integrate_tracking_error=False.
        # The tracking branch should be a no-op for the integrator;
        # anti-windup must continue to prevent windup at saturation.
        block_flagged = _make_block(
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=False,
            **common,
        )
        ints_flagged, _ = _run_steps(
            block_flagged, r=5.0, y=0.0, u_ext=0.0,
            n_steps=200, kp=1.0, ki=4.0, kd=0.0,
            output_min=-1.0, output_max=1.0,
            anti_windup_gain=0.2, tracking_gain=0.5,
        )
        # Both trajectories should be identical: tracking_enabled=True
        # adds the port but the flag suppresses its only effect on the
        # integrator; anti-windup is unchanged.
        assert jnp.allclose(ints_base, ints_flagged, atol=1e-10, rtol=0.0), (
            f"anti-windup integrator perturbed by gated tracking term; "
            f"max diff = {float(jnp.max(jnp.abs(ints_base - ints_flagged)))}"
        )

    def test_off_mode_still_finite_with_anti_windup(self):
        """Smoke: both mechanisms produce finite results when active."""
        block = _make_block(
            kp=1.0, ki=4.0, kd=0.5, dt=0.05,
            output_min=-1.0, output_max=1.0,
            anti_windup_method="back_calc", anti_windup_gain=0.2,
            tracking_enabled=True, tracking_gain=0.5,
            integrate_tracking_error=False,
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
        assert jnp.all(us <= 1.0 + 1e-12)
        assert jnp.all(us >= -1.0 - 1e-12)


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    """The flag is locked at construction time."""

    def test_flag_change_after_init_rejected(self):
        """Flipping ``integrate_tracking_error`` in ``initialize`` is rejected."""
        block = PIDController2DOF(
            dt=0.1, kp=1.0, ki=0.5, kd=0.1,
            tracking_enabled=True,
            integrate_tracking_error=True,
            name="pid",
        )
        with pytest.raises(
            ValueError, match="integrate_tracking_error cannot be changed"
        ):
            block.initialize(
                kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0, initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                tracking_enabled=True,
                integrate_tracking_error=False,  # was True at construction
            )

    def test_port_count_unchanged_by_flag(self):
        """The flag does NOT add or remove input ports."""
        pid_on = PIDController2DOF(
            dt=0.05, tracking_enabled=True,
            integrate_tracking_error=True, name="on",
        )
        pid_off = PIDController2DOF(
            dt=0.05, tracking_enabled=True,
            integrate_tracking_error=False, name="off",
        )
        # Both have r, y, u_ext → 3 ports.
        assert len(pid_on.input_ports) == 3
        assert len(pid_off.input_ports) == 3
        assert pid_on.u_ext_index == 2
        assert pid_off.u_ext_index == 2


# --------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """Both modes remain differentiable through ``jax.grad``."""

    @staticmethod
    def _loss(tracking_gain, *, mode, n_steps=200):
        kp, ki, kd = 0.5, 1.0, 0.0
        block = _make_block(
            kp=kp, ki=ki, kd=kd, dt=0.05,
            tracking_enabled=True,
            tracking_gain=tracking_gain,
            integrate_tracking_error=mode,
        )
        State = namedtuple("State", ["discrete_state"])
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0, dtype=jnp.float64),
            e_d_prev=jnp.asarray(0.0, dtype=jnp.float64),
            e_dot_prev=jnp.asarray(0.0, dtype=jnp.float64),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(1.0)
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
            total = total + u ** 2
        return total

    def test_grad_finite_with_flag_on(self):
        g = jax.grad(self._loss)(jnp.asarray(0.5), mode=True)
        assert jnp.isfinite(g)

    def test_grad_finite_with_flag_off(self):
        # With the flag off, tracking_gain affects nothing in the
        # integrator path, so the grad should be exactly zero (and
        # finite, not NaN).  This confirms the gating is a clean
        # short-circuit rather than a runtime nan-producer.
        g = jax.grad(self._loss)(jnp.asarray(0.5), mode=False)
        assert jnp.isfinite(g)
        assert float(g) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------- #
# Config round-trip
# --------------------------------------------------------------------- #


class TestConfigRoundTrip:
    """``to_dict`` / ``from_dict`` preserve the new flag."""

    def test_default_is_true_in_dict(self):
        pid = PIDController2DOF(dt=0.05, name="pid")
        data = pid.to_dict()
        assert data["integrate_tracking_error"] is True

    def test_round_trip_off(self):
        pid = PIDController2DOF(
            dt=0.05, tracking_enabled=True,
            integrate_tracking_error=False, name="pid",
        )
        data = pid.to_dict()
        assert data["integrate_tracking_error"] is False
        pid2 = PIDController2DOF.from_dict(data, name="pid2")
        assert pid2.integrate_tracking_error is False
        assert pid2.tracking_enabled is True

    def test_round_trip_on(self):
        pid = PIDController2DOF(
            dt=0.05, tracking_enabled=True,
            integrate_tracking_error=True, name="pid",
        )
        data = pid.to_dict()
        assert data["integrate_tracking_error"] is True
        pid2 = PIDController2DOF.from_dict(data, name="pid2")
        assert pid2.integrate_tracking_error is True
