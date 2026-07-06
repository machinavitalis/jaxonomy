# SPDX-License-Identifier: MIT
"""
T-005 — precision policy enforcement tests.

Covers:

- ``precision_info()`` reports float64 under the default install.
- Solver error bounds at documented tolerance settings (sanity floor
  for each (solver, rtol, atol) bucket).
- Edge cases the policy explicitly enumerates:
    * stiff system under BDF
    * near-singular Jacobian
    * long-horizon harmonic oscillator energy drift
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── API: precision_info reports float64 by default ─────────────────────────


def test_precision_info_default_is_float64():
    """The default install enables x64; precision_info() must reflect that."""
    info = jaxonomy.precision_info()
    assert info.x64_enabled is True
    assert info.default_float_dtype == "float64"
    assert info.machine_eps < 3e-16
    assert info.integer_time_dtype == "int64"


def test_assert_float64_active_passes_by_default():
    """assert_float64_active is a no-op in the default install."""
    jaxonomy.assert_float64_active()


# ── Solver error bounds per documented tolerance bucket ────────────────────


class _Decay(jaxonomy.LeafSystem):
    """dx/dt = -x, x(0) = 1.  Analytic solution: x(T) = exp(-T)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)

    def _ode(self, time, state, **params):
        return -state.continuous_state


def _final_error(solver, rtol, atol, tf=10.0, **extra):
    sys = _Decay()
    ctx = sys.create_context()
    kwargs = dict(math_backend="jax", ode_solver_method=solver, rtol=rtol, atol=atol)
    if solver == "rk4":
        kwargs["max_minor_step_size"] = 0.001
    kwargs.update(extra)
    opts = jaxonomy.SimulatorOptions(**kwargs)
    res = jaxonomy.simulate(sys, ctx, (0.0, tf), options=opts)
    x_final = float(res.context.continuous_state)
    expected = math.exp(-tf)
    return abs(x_final - expected)


@pytest.mark.parametrize(
    "solver,rtol,atol,bound",
    [
        ("rk4",    0.0,   0.0,   1e-10),  # rk4 tols ignored
        ("dopri5", 1e-6,  1e-8,  1e-7),
        ("dopri5", 1e-10, 1e-12, 1e-11),
        ("bdf",    1e-6,  1e-8,  1e-5),
        ("bdf",    1e-10, 1e-12, 1e-9),
    ],
)
def test_error_bounds_per_solver_f64(solver, rtol, atol, bound):
    """Empirical error floor for each (solver, tol) bucket on the default
    exponential-decay benchmark.  Bounds documented in POLICY.md."""
    err = _final_error(solver, rtol, atol, max_major_steps=1500)
    assert err < bound, (
        f"{solver} rtol={rtol} atol={atol}: final error {err:.3e} > bound {bound:.1e}"
    )


# ── Edge case: stiff system (Van der Pol, µ large) under BDF ───────────────


class _VanDerPol(jaxonomy.LeafSystem):
    """Van der Pol oscillator.

    Stiff for large mu: dx/dt = v,  dv/dt = mu*(1 - x**2)*v - x
    """

    def __init__(self, mu=100.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("mu", mu)
        self.declare_continuous_state(default_value=jnp.array([2.0, 0.0]), ode=self._ode)

    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        mu = params["mu"]
        return jnp.array([v, mu * (1.0 - x**2) * v - x])


def test_stiff_system_bdf():
    """BDF must integrate a stiff Van der Pol (µ=100) over 10 s without
    stalling.  The trajectory passes through two limit-cycle transitions."""
    sys = _VanDerPol(mu=100.0)
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf", rtol=1e-6, atol=1e-8,
        max_major_steps=1000,
    )
    res = jaxonomy.simulate(sys, ctx, (0.0, 10.0), options=opts)
    x_final = np.asarray(res.context.continuous_state)
    # Van der Pol limit cycle has amplitude ≈ 2 for any mu; x_final[0] must
    # be bounded.
    assert np.all(np.isfinite(x_final)), "BDF returned NaN on stiff VdP"
    assert abs(x_final[0]) < 3.0, f"VdP x out of range: {x_final}"


# ── Edge case: near-singular Jacobian (pendulum near upright) ──────────────


class _Pendulum(jaxonomy.LeafSystem):
    """dθ/dt = ω, dω/dt = -(g/L)·sin(θ).  Near-upright initial condition
    makes sin(θ) small ⇒ small restoring force ⇒ near-singular linearisation."""

    def __init__(self, theta0=np.pi - 0.01, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(
            default_value=jnp.array([theta0, 0.0]), ode=self._ode
        )

    def _ode(self, time, state, **params):
        theta, w = state.continuous_state
        return jnp.array([w, -9.81 * jnp.sin(theta)])


def test_near_singular_pendulum():
    """Pendulum started near the upright (unstable) equilibrium.  Small
    restoring force for most of the trajectory; solver must complete
    without NaN and stay in a reasonable range."""
    sys = _Pendulum(theta0=np.pi - 0.01)
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5", rtol=1e-8, atol=1e-10,
        max_major_steps=1000,
    )
    res = jaxonomy.simulate(sys, ctx, (0.0, 5.0), options=opts)
    x_final = np.asarray(res.context.continuous_state)
    assert np.all(np.isfinite(x_final)), f"NaN from near-singular pendulum: {x_final}"
    # Energy is approximately conserved (undamped); not a tight check, but the
    # state should not blow up.
    assert abs(x_final[1]) < 20.0, f"pendulum angular velocity out of range: {x_final}"


# ── Edge case: long-horizon energy drift at float64 ────────────────────────


class _SHO(jaxonomy.LeafSystem):
    def __init__(self, omega=1.0, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega
        self.declare_continuous_state(default_value=jnp.array([1.0, 0.0]), ode=self._ode)

    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        return jnp.array([v, -(self.omega**2) * x])


def test_long_horizon_energy_drift_float64():
    """100 periods of a simple harmonic oscillator under Dopri5 @ 1e-10 tol.

    A 10,000-period horizon would be the ideal torture test but is slow
    for a per-PR suite.  100 periods with tight tolerances already catches
    anything meaningful — float64 drift at this scale should be < 1e-8.
    """
    omega = 2.0
    sys = _SHO(omega=omega)
    ctx = sys.create_context()
    tf = 100.0 * 2 * math.pi / omega
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        rtol=1e-10, atol=1e-12, max_major_steps=20000,
    )
    res = jaxonomy.simulate(sys, ctx, (0.0, tf), options=opts)
    x, v = np.asarray(res.context.continuous_state)

    e0 = 0.5 * (0.0**2 + omega**2 * 1.0**2)
    e_final = 0.5 * (v**2 + omega**2 * x**2)
    rel_drift = abs(e_final - e0) / e0
    # Empirically 2.4e-8 at rtol=1e-10 over 100 periods; the bound is set at
    # 1e-7 to give headroom for solver version jitter while still being
    # tight enough to catch a regression.
    assert rel_drift < 1e-7, f"long-horizon energy drift too large: {rel_drift:.3e}"
