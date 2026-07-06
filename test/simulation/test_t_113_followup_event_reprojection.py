# SPDX-License-Identifier: MIT
"""Tests for T-113-followup-event-reprojection — explicit DAE projection
immediately after discrete-event resets.

T-003a's ``dae_projection_enabled`` projects only at the end of each
major step.  Discrete updates (handled at the top of ``_major_step``)
and triggered ZC resets (handled inside ``_advance_continuous_time``)
modify state *within* a major step; if a reset map drops the algebraic
states off the constraint manifold, the worst case is one major step
of integration on infeasible state until the boundary projection
catches up.  This follow-up adds
``SimulatorOptions.dae_reproject_after_events`` which, when True, runs
``project_constraints`` immediately after the discrete-update reset and
again after the continuous-integration phase, re-establishing
``f_a = 0`` before continuous integration resumes.

These tests verify:

* Default-off byte-equivalence — the option's absence does not perturb
  recorded outputs on the planar pendulum.
* Pure-ODE no-op — setting the option on a non-DAE system is a no-op.
* Event-driven reset drift reduction — a toy index-1 DAE with a ZC
  reset that drops the algebraic state off the manifold sees its drift
  return to projection tolerance within one major step when the option
  is on.  Without the option (and without T-003a), the post-reset
  drift persists through the next continuous-integration step.
* Composes with T-003a — combining ``dae_reproject_after_events=True``
  with ``dae_projection_enabled=True`` keeps both hooks running and
  the drift bounded throughout.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation.dae_drift import constraint_residual_norm
from jaxonomy.testing.markers import skip_if_not_jax, requires_jax

skip_if_not_jax()


# Index-2 holonomic-constraint test bed (T-032).  Verbatim copy from
# test/simulation/test_dae_projection.py / T-113 phase 1 / Baumgarte
# follow-up tests.
class PlanarPendulum(jaxonomy.LeafSystem):
    def __init__(self, L=1.0, g0=9.8, name=None):
        super().__init__(name=name)
        x0 = np.array(
            [0.0, 0.8660254037844386, 0.0, -4.9, -0.5,
             -4.243524478543744, -7.35, -7.35, 0.0]
        )
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g0", g0)
        self.nx, self.nz = 2, 7
        M = np.concatenate([np.ones(self.nx), np.zeros(self.nz)])
        self.declare_continuous_state(default_value=x0, mass_matrix=M, ode=self.ode)
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **parameters):
        L, g0 = parameters["L"], parameters["g0"]
        x = state.continuous_state[:2]
        z = state.continuous_state[2:]
        f = jnp.array([z[3], x[0]])
        g = jnp.array([
            -(L**2) + x[1] ** 2 + z[2] ** 2,
            2 * z[0] * z[2] + 2 * x[1] * x[0],
            z[0] - z[6],
            2 * z[3] * x[1] + 2 * z[4] * z[2] + 2 * z[0] ** 2 + 2 * x[0] ** 2,
            z[4] - z[5],
            z[5] + g0 - z[1] * z[2],
            -z[1] * x[1] + z[3],
        ])
        return jnp.concatenate([f, g])


# Toy index-1 DAE with a TERMINAL ZC reset map that *deliberately*
# perturbs the algebraic state off the constraint manifold.  State is
# ``[x, z]`` with ``M = diag(1, 0)``, so the rhs algebraic row enforces
# ``z = x * x``.  The guard ``t - 0.4`` fires once at t=0.4; the reset
# adds ``_RESET_DRIFT`` to ``z`` taking ``g = z - x*x`` from 0 to
# ~_RESET_DRIFT.  Terminal=True so the simulation stops on the event —
# the final context captures state *immediately* after the reset, with
# no subsequent BDF step to silently re-converge the constraint.  This
# is the right observability hook: with re-projection ON, the final
# residual is at projection tol; with it OFF (and T-003a OFF), the
# final residual carries the full ``_RESET_DRIFT``.
_RESET_DRIFT = 0.1


class _ToyDAEWithReset(jaxonomy.LeafSystem):
    def __init__(self, name=None):
        super().__init__(name=name)
        x0 = np.array([1.0, 1.0])  # consistent: g(1, 1) = 1 - 1 = 0
        M = np.array([1.0, 0.0])
        self.declare_continuous_state(default_value=x0, mass_matrix=M, ode=self._ode)
        self.declare_continuous_state_output(name="x")

        def _reset(t, s, *i, **p):
            cs = s.continuous_state
            # Push z OFF the manifold by _RESET_DRIFT.  x is unchanged.
            new_cs = jnp.array([cs[0], cs[1] + _RESET_DRIFT])
            return s.with_continuous_state(new_cs)

        # Edge at t = 0.4: guard rises through zero exactly once.  We
        # mark the event terminal so the simulation halts immediately
        # after the reset — the final ``context.continuous_state``
        # captures the post-reset state with no further BDF iterations
        # that would otherwise silently re-converge the constraint.
        self.declare_zero_crossing(
            guard=lambda t, s, *i, **p: t - 0.4,
            reset_map=_reset,
            direction="negative_then_non_negative",
            terminal=True,
        )

    def _ode(self, time, state, **params):
        x = state.continuous_state[0]
        z = state.continuous_state[1]
        return jnp.array([-x, z - x * x])


def test_dae_reproject_option_default_off():
    """``SimulatorOptions()`` defaults ``dae_reproject_after_events`` to False."""
    opts = jaxonomy.SimulatorOptions()
    assert opts.dae_reproject_after_events is False


@requires_jax()
def test_default_off_byte_equivalent_on_pendulum():
    """Default-off path produces byte-equal recorded outputs on the
    planar pendulum compared to a baseline that omits the option."""
    model = PlanarPendulum()
    ctx = model.create_context()
    rec = {"x": model.output_ports[0]}

    base_opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
    )
    res_baseline = jaxonomy.simulate(
        model, ctx, (0.0, 0.5), recorded_signals=rec, options=base_opts,
    )

    same_opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
        dae_reproject_after_events=False,
    )
    res_default = jaxonomy.simulate(
        model, ctx, (0.0, 0.5), recorded_signals=rec, options=same_opts,
    )

    np.testing.assert_array_equal(
        np.asarray(res_baseline.outputs["x"]),
        np.asarray(res_default.outputs["x"]),
    )


@requires_jax()
def test_pure_ode_noop():
    """Enabling the option on a pure-ODE system is byte-equivalent —
    the hook short-circuits when the system has no mass matrix."""
    class Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(
                default_value=jnp.array(1.0), ode=self._ode,
            )
            self.declare_continuous_state_output(name="x")

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = Decay()
    ctx = sys.create_context()
    rec = {"x": sys.output_ports[0]}

    res_off = jaxonomy.simulate(
        sys, ctx, (0.0, 1.0), recorded_signals=rec,
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="dopri5",
        ),
    )
    res_on = jaxonomy.simulate(
        sys, ctx, (0.0, 1.0), recorded_signals=rec,
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="dopri5",
            dae_reproject_after_events=True,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(res_off.outputs["x"]),
        np.asarray(res_on.outputs["x"]),
        rtol=1e-12, atol=1e-14,
    )


@requires_jax()
def test_event_reprojection_reduces_post_reset_drift():
    """Toy DAE with a ZC reset that pushes ``z`` off the manifold:
    with ``dae_reproject_after_events=True``, the post-event drift drops
    back below the projection tolerance within the major step that
    contains the event.  Compare ``constraint_residual_norm`` on the
    final context with the option off vs on."""
    sys = _ToyDAEWithReset()
    ctx = sys.create_context()
    t_span = (0.0, 0.5)  # event fires at t=0.4

    # Baseline: no projection, no re-projection.  The post-event step
    # integrates on infeasible state; final residual carries the drift.
    opts_off = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-8, atol=1e-10,
    )
    res_off = jaxonomy.simulate(sys, ctx, t_span, options=opts_off)
    drift_off = constraint_residual_norm(sys, res_off.context)
    assert drift_off is not None, "test bed has no algebraic rows?"

    # On: project after the event reset.  Final residual should be at
    # or below the projection tolerance.
    opts_on = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-8, atol=1e-10,
        dae_reproject_after_events=True,
        dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
    )
    res_on = jaxonomy.simulate(sys, ctx, t_span, options=opts_on)
    drift_on = constraint_residual_norm(sys, res_on.context)
    assert drift_on is not None

    assert drift_on < 1e-7, (
        f"post-event projection failed to drive drift below 1e-7: "
        f"baseline={drift_off:.3e}, projected={drift_on:.3e}"
    )
    # And the reprojection must materially reduce the drift relative
    # to the unprotected baseline.  Use a healthy ratio so the test
    # is not noise-sensitive.
    assert drift_on < drift_off * 1e-2, (
        f"reprojection did not materially reduce drift: "
        f"baseline={drift_off:.3e}, projected={drift_on:.3e}"
    )


@requires_jax()
def test_composes_with_t003a_major_step_projection():
    """Both hooks can run together: event-reprojection + T-003a major-
    step projection.  Drift stays bounded; the simulation completes and
    final residual is small."""
    sys = _ToyDAEWithReset()
    ctx = sys.create_context()
    t_span = (0.0, 0.5)

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-8, atol=1e-10,
        dae_projection_enabled=True,
        dae_reproject_after_events=True,
        dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
    )
    res = jaxonomy.simulate(sys, ctx, t_span, options=opts)
    drift = constraint_residual_norm(sys, res.context)
    assert drift is not None
    assert drift < 1e-7, f"composed projections did not bound drift: {drift:.3e}"
    # Final state should still reflect the event firing (z was bumped).
    cs = np.asarray(res.context.continuous_state)
    # x decays toward exp(-0.5); z = x*x post-projection.
    np.testing.assert_allclose(cs[1], cs[0] * cs[0], atol=1e-6)
