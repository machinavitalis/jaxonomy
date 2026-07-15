# SPDX-License-Identifier: MIT
"""Tests for the DAE constraint projection step (T-003a).

Companion to ``test_dae_drift.py`` (the detection primitive).  Verifies
opt-in semantics, no-op on pure ODEs, residual reduction on the
:class:`PlanarPendulum` index-2 test bed, and that the projection
preserves the differential sub-vector.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation.dae_drift import (
    algebraic_row_mask,
    constraint_residual_norm,
)
from jaxonomy.simulation.dae_projection import project_constraints
from jaxonomy.testing.markers import skip_if_not_jax, requires_jax

skip_if_not_jax()


# Index-2 holonomic-constraint test bed (T-032).  Verbatim copy from
# ``test/simulation/test_mass_matrix.py``; the source class is not a
# public symbol so we re-declare it here.
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


def test_dae_projection_disabled_by_default():
    """A fresh ``SimulatorOptions()`` has projection off — the option is opt-in."""
    opts = jaxonomy.SimulatorOptions()
    assert opts.dae_projection_enabled is False
    # The numerical fields exist with sane defaults.
    assert opts.dae_projection_tol == 1e-8
    assert opts.dae_projection_max_iter == 20


@requires_jax()
def test_dae_projection_no_op_on_pure_ode():
    """``dae_projection_enabled=True`` on a non-DAE system is a no-op."""
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
        sys, ctx, (0.0, 2.0), recorded_signals=rec,
        options=jaxonomy.SimulatorOptions(math_backend="jax", ode_solver_method="dopri5"),
    )
    res_on = jaxonomy.simulate(
        sys, ctx, (0.0, 2.0), recorded_signals=rec,
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="dopri5",
            dae_projection_enabled=True,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(res_off.outputs["x"][-1]),
        np.asarray(res_on.outputs["x"][-1]),
        rtol=1e-10, atol=1e-12,
    )


@requires_jax()
def test_dae_projection_reduces_pendulum_residual():
    """Enabling projection holds ``||f_a||_∞`` below the configured tol."""
    model = PlanarPendulum(L=1.0, g0=9.8)
    ctx = model.create_context()
    tf = 1.0

    opts_off = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf", rtol=1e-6, atol=1e-8,
    )
    res_off = jaxonomy.simulate(model, ctx, (0.0, tf), options=opts_off)
    resid_off = constraint_residual_norm(model, res_off.context)

    opts_on = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf", rtol=1e-6, atol=1e-8,
        dae_projection_enabled=True, dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
    )
    res_on = jaxonomy.simulate(model, ctx, (0.0, tf), options=opts_on)
    resid_on = constraint_residual_norm(model, res_on.context)

    assert resid_on is not None and resid_off is not None
    assert resid_on < 1e-7, (
        f"projection failed: baseline={resid_off:.3e}, projected={resid_on:.3e}"
    )


@requires_jax()
def test_dae_projection_preserves_differential_states():
    """Newton step must not move the differential entries of the state."""
    model = PlanarPendulum(L=1.0, g0=9.8)
    ctx = model.create_context()
    mask = algebraic_row_mask(model)
    assert mask is not None and mask.any() and not mask.all()

    state = np.asarray(ctx.continuous_state).copy()
    diff_before = state[~mask].copy()
    state[mask] += 1e-3
    perturbed_ctx = ctx.with_continuous_state(jnp.asarray(state))

    projected = project_constraints(model, perturbed_ctx, tol=1e-10, max_iter=4)
    diff_after = np.asarray(projected.continuous_state)[~mask]

    np.testing.assert_allclose(
        diff_after, diff_before, atol=1e-12,
        err_msg="projection moved the differential states",
    )
    resid_after = constraint_residual_norm(model, projected)
    assert resid_after is not None and resid_after < 1e-7


@requires_jax()
def test_dae_projection_corrects_manual_perturbation():
    """Direct projection drives ``||f_a||_∞`` down by orders of magnitude."""
    model = PlanarPendulum(L=1.0, g0=9.8)
    ctx = model.create_context()
    mask = algebraic_row_mask(model)
    state = np.asarray(ctx.continuous_state).copy()
    state[mask] += 1e-2
    perturbed_ctx = ctx.with_continuous_state(jnp.asarray(state))

    resid_before = constraint_residual_norm(model, perturbed_ctx)
    projected = project_constraints(model, perturbed_ctx, tol=1e-10, max_iter=4)
    resid_after = constraint_residual_norm(model, projected)

    assert resid_before is not None and resid_after is not None
    assert resid_after < resid_before * 1e-3
