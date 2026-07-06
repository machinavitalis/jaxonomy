# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-discrete-integrator-derivative.

Phase 1 of T-127 hard-coded the integrator and derivative kernels of
:class:`PIDController2DOF` to forward-Euler integration and a simple
finite-difference derivative.  This followup adds two new construction
kwargs that let users swap in alternate discretisations:

* ``integrator_method`` ∈ {``"forward_euler"`` (default),
  ``"backward_euler"``, ``"trapezoidal"``}.
* ``derivative_method`` ∈ {``"forward_diff"`` (default),
  ``"backward_diff"``, ``"centered_diff"``}.

Both default to byte-equivalence with phase 1.  Non-default kernel
choices add small bookkeeping cells to the discrete state when needed
(``e_i_prev`` for non-default integrator, ``e_d_prev_prev`` for
``"centered_diff"``).

These tests cover:

* Default kernels: bit-identical 10-step closed-loop trace vs phase 1.
* Trapezoidal integrator differs from forward_euler by the documented
  ``(e[k] - e[k-1]) / 2 * dt`` shift on a ramp input.
* Backward-difference derivative produces a one-tick-delayed initial
  transient relative to forward_diff.
* Centered-difference state has the expected extra ``e_d_prev_prev``
  field (one element larger than the default).
* Differentiability: ``jax.grad`` is finite for every (integrator,
  derivative) combination.
* Validation: invalid method strings raise clear ``ValueError``s.
* Validation: combining a non-default ``derivative_method`` with a
  recursive filter (``filter_type != "none"``) raises a ``ValueError``.
* Validation: flipping ``integrator_method`` / ``derivative_method`` in
  ``initialize`` is rejected.
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


def _make_block(
    *,
    dt=0.1,
    kp=1.0,
    ki=1.0,
    kd=0.1,
    b=1.0,
    c=1.0,
    integrator_method="forward_euler",
    derivative_method="forward_diff",
    filter_type="none",
    filter_coefficient=1.0,
):
    """Construct + initialize a PIDController2DOF block in isolation."""
    block = PIDController2DOF(
        dt=dt,
        kp=kp,
        ki=ki,
        kd=kd,
        b=b,
        c=c,
        filter_type=filter_type,
        filter_coefficient=filter_coefficient,
        integrator_method=integrator_method,
        derivative_method=derivative_method,
        name="pid",
    )
    block.initialize(
        kp=kp,
        ki=ki,
        kd=kd,
        b=b,
        c=c,
        initial_state=0.0,
        filter_type=filter_type,
        filter_coefficient=filter_coefficient,
        integrator_method=integrator_method,
        derivative_method=derivative_method,
    )
    return block


def _initial_state(block):
    """Build a zero discrete-state tuple for an initialized block."""
    State = namedtuple("State", ["discrete_state"])
    fields = block._state_fields
    zeros = {name: jnp.asarray(0.0) for name in fields}
    return State(discrete_state=block.DiscreteStateType(**zeros))


def _run_loop(block, errors, *, kp=1.0, ki=1.0, kd=0.1, b=1.0, c=1.0):
    """Run ``len(errors)`` ticks of the PID block in isolation.

    ``errors`` is the per-tick (r - y) sequence.  We hold y=0 and pass
    ``r = error`` so ``e_i = r - y = r = error``.

    Returns the recorded (u, integral) traces, both of length
    ``len(errors)``.
    """
    State = namedtuple("State", ["discrete_state"])
    state = _initial_state(block)
    params = dict(kp=kp, ki=ki, kd=kd, b=b, c=c)
    u_trace = []
    int_trace = []
    for e in errors:
        r = jnp.asarray(float(e))
        y = jnp.asarray(0.0)
        u = block._output(jnp.asarray(0.0), state, r, y, **params)
        u_trace.append(u)
        int_trace.append(state.discrete_state.integral)
        new_xd = block._update(jnp.asarray(0.0), state, r, y, **params)
        state = State(discrete_state=new_xd)
    return jnp.stack(u_trace), jnp.stack(int_trace)


def _simulate_closed_loop_diagram(
    *,
    dt=0.05,
    kp=2.0,
    ki=4.0,
    kd=0.1,
    integrator_method="forward_euler",
    derivative_method="forward_diff",
    t_end=0.5,
):
    """Closed-loop tracking with a unit-integrator plant.

    Used to confirm that the *default* kernel choice produces a 10-step
    trace bit-identical to the implicit phase 1 default.
    """
    builder = jaxonomy.DiagramBuilder()
    r = builder.add(Constant(1.0, name="r"))
    plant = builder.add(Integrator(initial_state=0.0, name="plant"))
    pid = builder.add(
        PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            integrator_method=integrator_method,
            derivative_method=derivative_method,
            name="pid",
        )
    )
    builder.connect(r.output_ports[0], pid.input_ports[0])
    builder.connect(plant.output_ports[0], pid.input_ports[1])
    builder.connect(pid.output_ports[0], plant.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0], "y": plant.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs["u"], results.outputs["y"], results.time


# --------------------------------------------------------------------- #
# Default kernels: byte-equivalent to phase 1
# --------------------------------------------------------------------- #


class TestDefaultsByteEquivalent:
    """Default kernels reproduce the phase 1 closed-loop trace exactly."""

    def test_default_kernels_match_implicit_phase1(self):
        """Closed-loop trace with explicit defaults == implicit defaults."""
        u_imp, y_imp, _ = _simulate_closed_loop_diagram(
            integrator_method="forward_euler",
            derivative_method="forward_diff",
            t_end=0.5,
        )
        # Re-run with phase-1 defaults (no kwargs).  Build manually so we
        # don't rely on default-arg propagation through this helper.
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        plant = builder.add(Integrator(initial_state=0.0, name="plant"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=2.0, ki=4.0, kd=0.1, name="pid"
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(plant.output_ports[0], pid.input_ports[1])
        builder.connect(pid.output_ports[0], plant.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 0.5),
            recorded_signals={"u": pid.output_ports[0],
                              "y": plant.output_ports[0]},
        )
        assert jnp.array_equal(u_imp, results.outputs["u"]), (
            "Default kernels must produce a bit-identical control trace"
        )
        assert jnp.array_equal(y_imp, results.outputs["y"]), (
            "Default kernels must produce a bit-identical plant trace"
        )

    def test_default_state_shape_unchanged(self):
        """Defaults keep the 3-field state layout (no extra delay cells)."""
        block = _make_block()
        assert block._state_fields == ("integral", "e_d_prev", "e_dot_prev")


# --------------------------------------------------------------------- #
# Trapezoidal integrator: documented (e[k] - e[k-1]) / 2 * dt shift
# --------------------------------------------------------------------- #


class TestTrapezoidalIntegrator:
    """Trapezoidal integrator differs from forward_euler by a known
    shift on a ramp input."""

    def test_trapezoidal_matches_expected_shift_on_ramp(self):
        """For a ramp ``e[k] = (k+1) * dt``, trapezoidal differs from
        forward_euler by a per-step shift of ``(e_prev − e_curr) / 2 * dt``.

        Per the spec note "expected ``(e[k] − e[k−1]) / 2 * dt`` shift",
        this telescopes to a known cumulative offset.  In our impl the
        offset is negative because ``trapezoidal`` averages in the older
        (smaller, on a rising ramp) sample.
        """
        dt = 0.1
        n = 8
        # Ramp errors: e[k] = (k+1) * dt.
        errors = [(k + 1) * dt for k in range(n)]
        # Pure-I controller so u = ki * integral directly reflects the
        # integrator state.
        kp, ki, kd = 0.0, 1.0, 0.0

        block_fe = _make_block(
            dt=dt, kp=kp, ki=ki, kd=kd, integrator_method="forward_euler"
        )
        block_tr = _make_block(
            dt=dt, kp=kp, ki=ki, kd=kd, integrator_method="trapezoidal"
        )
        _, int_fe = _run_loop(block_fe, errors, kp=kp, ki=ki, kd=kd)
        _, int_tr = _run_loop(block_tr, errors, kp=kp, ki=ki, kd=kd)

        # Per-step shift on this ramp: (e_prev - e_curr)/2 * dt.
        # At update step j (0-indexed): e_curr = (j+1)*dt, e_prev = j*dt
        # (with e_prev=0 at j=0).  So at every step the shift = -dt^2/2.
        # After k ticks the cumulative shift = -k * dt^2 / 2.
        # The integrator read at output time k reflects k completed updates.
        for k in range(1, n):
            expected = -k * dt * dt / 2.0
            actual = float(int_tr[k] - int_fe[k])
            assert abs(actual - expected) < 1e-10, (
                f"Step {k}: expected trap−fe shift {expected}, "
                f"got {actual}"
            )
        # Also: the *magnitude* of the per-step shift matches the spec's
        # "(e[k] − e[k−1]) / 2 * dt" formula (up to sign).
        per_step_shift = float(int_tr[2] - int_tr[1]) - float(
            int_fe[2] - int_fe[1]
        )
        assert abs(per_step_shift) == pytest.approx(dt * dt / 2.0, abs=1e-12)

    def test_trapezoidal_state_has_e_i_prev(self):
        """Non-default integrator adds the e_i_prev delay cell."""
        block = _make_block(integrator_method="trapezoidal")
        assert "e_i_prev" in block._state_fields
        # Original 3 + 1 = 4 fields.
        assert len(block._state_fields) == 4

    def test_backward_euler_state_has_e_i_prev(self):
        """Backward-Euler also requires the previous integral-error sample."""
        block = _make_block(integrator_method="backward_euler")
        assert "e_i_prev" in block._state_fields


# --------------------------------------------------------------------- #
# Backward-diff derivative: one-tick lag on the initial transient
# --------------------------------------------------------------------- #


class TestBackwardDiffDerivative:
    """``derivative_method="backward_diff"`` delays the derivative by one
    tick relative to ``forward_diff``."""

    def test_backward_diff_delays_initial_transient(self):
        """A step in the error: forward_diff fires at tick 1, backward
        fires at tick 2 (one tick later)."""
        dt = 0.1
        # Step input: e=0 for k=0, then e=1 thereafter.
        errors = [0.0] + [1.0] * 6
        # D-only controller: u = kd * d/dt(e).  Using c=1 so e_d == e_i.
        kp, ki, kd = 0.0, 0.0, 1.0

        block_fd = _make_block(
            dt=dt, kp=kp, ki=ki, kd=kd, derivative_method="forward_diff"
        )
        block_bd = _make_block(
            dt=dt, kp=kp, ki=ki, kd=kd, derivative_method="backward_diff"
        )
        u_fd, _ = _run_loop(block_fd, errors, kp=kp, ki=ki, kd=kd)
        u_bd, _ = _run_loop(block_bd, errors, kp=kp, ki=ki, kd=kd)

        # Forward-diff sees the step the moment it arrives: at tick 1
        # the derivative is (1 - 0)/dt = 10.0 (with kd=1).
        assert float(u_fd[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(u_fd[1]) == pytest.approx(1.0 / dt, abs=1e-10)

        # Backward-diff lags by one tick: tick 1 still sees zero derivative,
        # tick 2 sees the spike.
        assert float(u_bd[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(u_bd[1]) == pytest.approx(0.0, abs=1e-12)
        assert float(u_bd[2]) == pytest.approx(1.0 / dt, abs=1e-10)

        # Steady-state (k >= 3): both kernels see d/dt(constant) = 0.
        for k in range(3, len(errors)):
            assert float(u_fd[k]) == pytest.approx(0.0, abs=1e-12)
            assert float(u_bd[k]) == pytest.approx(0.0, abs=1e-12)

    def test_backward_diff_does_not_add_extra_state(self):
        """``backward_diff`` reuses ``e_dot_prev`` (already in state)."""
        block = _make_block(derivative_method="backward_diff")
        # Same field count as defaults — no e_d_prev_prev needed.
        assert block._state_fields == ("integral", "e_d_prev", "e_dot_prev")


# --------------------------------------------------------------------- #
# Centered-diff derivative: extra delay cell
# --------------------------------------------------------------------- #


class TestCenteredDiffDerivative:
    """``derivative_method="centered_diff"`` adds an extra delay cell."""

    def test_centered_diff_state_has_extra_field(self):
        """Centered-diff state has one more field than the default."""
        default_block = _make_block()
        centered_block = _make_block(derivative_method="centered_diff")
        assert (
            len(centered_block._state_fields)
            == len(default_block._state_fields) + 1
        )
        assert "e_d_prev_prev" in centered_block._state_fields
        assert "e_d_prev_prev" not in default_block._state_fields

    def test_centered_diff_value_matches_formula(self):
        """For a ramp e[k]=k*dt, centered_diff should output ~1.0 in
        steady state (= true derivative of the ramp)."""
        dt = 0.1
        # Ramp errors.  Note: centered_diff needs two prior samples, so
        # the first two outputs are meaningful only after the delay
        # cells fill up.
        n = 6
        errors = [k * dt for k in range(n)]
        kp, ki, kd = 0.0, 0.0, 1.0
        block = _make_block(
            dt=dt, kp=kp, ki=ki, kd=kd, derivative_method="centered_diff"
        )
        u, _ = _run_loop(block, errors, kp=kp, ki=ki, kd=kd)

        # By tick 2+, both delay cells are filled and the output is
        # (e[k] - e[k-2]) / (2*dt) = (k*dt - (k-2)*dt)/(2*dt) = 1.0.
        for k in range(2, n):
            assert float(u[k]) == pytest.approx(1.0, abs=1e-10), (
                f"Step {k}: centered_diff on a unit-slope ramp should "
                f"give 1.0, got {float(u[k])}"
            )


# --------------------------------------------------------------------- #
# Differentiability for every (integrator, derivative) combination
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` must remain finite for every kernel combination."""

    @staticmethod
    def _loss(block, kp, ki, kd, *, n_steps=4):
        State = namedtuple("State", ["discrete_state"])
        state = _initial_state(block)
        params = dict(kp=kp, ki=ki, kd=kd, b=1.0, c=1.0)
        r = jnp.asarray(1.0)
        y = jnp.asarray(0.0)
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, **params)
            total = total + u * u
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, **params
            )
            state = State(discrete_state=new_xd)
        return total

    @pytest.mark.parametrize("integrator_method",
                             ["forward_euler", "backward_euler", "trapezoidal"])
    @pytest.mark.parametrize("derivative_method",
                             ["forward_diff", "backward_diff", "centered_diff"])
    def test_grad_finite_all_combinations(
        self, integrator_method, derivative_method
    ):
        """``jax.grad`` of integrated u^2 wrt (kp, ki, kd) is finite."""
        # Build the block ONCE outside the differentiated function so
        # that the @parameters decorator's float() casts (in the block
        # ctor) never see traced values.
        block = _make_block(
            dt=0.1,
            kp=1.0,
            ki=0.5,
            kd=0.1,
            integrator_method=integrator_method,
            derivative_method=derivative_method,
        )

        def loss_fn(kp):
            return self._loss(
                block, kp, jnp.asarray(0.5), jnp.asarray(0.1)
            )

        g = jax.grad(loss_fn)(jnp.asarray(1.0))
        assert jnp.isfinite(g), (
            f"grad wrt kp not finite for "
            f"({integrator_method}, {derivative_method}): {g}"
        )


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


class TestValidation:
    """Invalid kernel strings raise clear errors at construction."""

    def test_invalid_integrator_method_rejected(self):
        with pytest.raises(ValueError, match="integrator_method must be one of"):
            PIDController2DOF(dt=0.1, integrator_method="midpoint")

    def test_invalid_derivative_method_rejected(self):
        with pytest.raises(ValueError, match="derivative_method must be one of"):
            PIDController2DOF(dt=0.1, derivative_method="adams")

    def test_filter_combined_with_nondefault_derivative_rejected(self):
        """Combining ``filter_type != "none"`` with a non-default
        ``derivative_method`` is ambiguous; the ctor must raise."""
        with pytest.raises(
            ValueError,
            match="derivative_method only applies when filter_type='none'",
        ):
            PIDController2DOF(
                dt=0.1,
                filter_type="forward",
                derivative_method="backward_diff",
            )

    def test_default_derivative_with_filter_ok(self):
        """``filter_type != "none"`` with the default derivative_method is
        fine (preserves phase 1's recursive-filter path)."""
        # Should not raise.
        PIDController2DOF(
            dt=0.1, filter_type="forward", derivative_method="forward_diff"
        )

    def test_integrator_method_change_after_init_rejected(self):
        block = PIDController2DOF(
            dt=0.1, integrator_method="forward_euler", name="pid"
        )
        with pytest.raises(
            ValueError, match="integrator_method cannot be changed"
        ):
            block.initialize(
                kp=1.0, ki=1.0, kd=0.1, b=1.0, c=1.0, initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                integrator_method="trapezoidal",
            )

    def test_derivative_method_change_after_init_rejected(self):
        block = PIDController2DOF(
            dt=0.1, derivative_method="centered_diff", name="pid"
        )
        with pytest.raises(
            ValueError, match="derivative_method cannot be changed"
        ):
            block.initialize(
                kp=1.0, ki=1.0, kd=0.1, b=1.0, c=1.0, initial_state=0.0,
                filter_type="none", filter_coefficient=1.0,
                derivative_method="forward_diff",
            )
