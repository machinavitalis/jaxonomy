# SPDX-License-Identifier: MIT
"""Tests for the DAE drift threshold warning (T-003b).

Companion to ``test_dae_drift.py`` (detection primitive) and
``test_dae_projection.py`` (Newton projection).  Verifies opt-in
semantics, threshold above/below residual, integration with T-003a,
and pure-ODE no-op behaviour.

The check fires from inside the jit'd ``_major_step`` via
``jax.debug.callback``; the callback gates ``warnings.warn`` host-side
so non-violating steps stay silent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation.dae_drift import algebraic_row_mask
from jaxonomy.testing.markers import skip_if_not_jax, requires_jax

skip_if_not_jax()


# Index-2 holonomic-constraint test bed (T-032).  Same model used in
# ``test_dae_projection.py``.
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


def _perturbed_ctx(model, perturb=1e-2):
    ctx = model.create_context()
    mask = algebraic_row_mask(model)
    state = np.asarray(ctx.continuous_state).copy()
    state[mask] += perturb
    return ctx.with_continuous_state(jnp.asarray(state))


def _has_drift_warning(records):
    return any(
        issubclass(r.category, UserWarning)
        and "DAE constraint residual" in str(r.message)
        for r in records
    )


def test_default_off_no_warning():
    """Default ``SimulatorOptions()`` emits no DAE drift warning."""
    assert jaxonomy.SimulatorOptions().dae_drift_threshold is None
    model = PlanarPendulum()
    ctx = _perturbed_ctx(model, perturb=1e-2)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        jaxonomy.simulate(
            model, ctx, (0.0, 0.05),
            options=jaxonomy.SimulatorOptions(
                math_backend="jax", ode_solver_method="bdf",
            ),
        )
    assert not _has_drift_warning(records)


@requires_jax()
def test_threshold_above_residual_no_warning():
    """Threshold well above residual: no warning."""
    model = PlanarPendulum()
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-8, atol=1e-10,
        dae_drift_threshold=1e-2,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        jaxonomy.simulate(model, ctx, (0.0, 0.1), options=opts)
    assert not _has_drift_warning(records)


@requires_jax()
def test_threshold_below_residual_warns():
    """Coarse-tolerance BDF lets residual grow above threshold —
    ``UserWarning`` fires.  BDF re-solves algebraic states aggressively
    each step, so a perturbed IC alone gets cleaned up before the first
    major-step check; the loose-tolerance long-sim path is the reliable
    way to provoke a measurable post-step drift."""
    model = PlanarPendulum()
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-3, atol=1e-3,
        dae_drift_threshold=1e-6,
    )
    with pytest.warns(UserWarning, match="DAE constraint residual"):
        jaxonomy.simulate(model, ctx, (0.0, 5.0), options=opts)


@requires_jax()
def test_threshold_with_projection_no_warning():
    """T-003a projection holds residual below threshold — no warning."""
    model = PlanarPendulum()
    ctx = _perturbed_ctx(model, perturb=1e-2)
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-8, atol=1e-10,
        dae_projection_enabled=True,
        dae_projection_tol=1e-12,
        dae_projection_max_iter=4,
        dae_drift_threshold=1e-6,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        jaxonomy.simulate(model, ctx, (0.0, 0.1), options=opts)
    assert not _has_drift_warning(records)


@requires_jax()
def test_pure_ode_threshold_is_noop():
    """Non-DAE system + threshold set: no warning, no error."""

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
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        dae_drift_threshold=1e-12,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = jaxonomy.simulate(
            sys, ctx, (0.0, 1.0),
            recorded_signals={"x": sys.output_ports[0]},
            options=opts,
        )
    assert not _has_drift_warning(records)
    assert float(result.outputs["x"][-1]) < 0.5
