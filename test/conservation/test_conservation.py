# SPDX-License-Identifier: MIT
"""
T-004 — conservation-law property tests.

Verifies that energy (or angular momentum, for the rigid body) of closed
conservative systems stays within tolerance over 10–50 oscillation
periods for each selectable ODE solver.

Systems covered:

  - Simple harmonic oscillator (energy)
  - Undamped pendulum, small-angle regime (energy)
  - Free rigid body with diagonal inertia tensor (angular momentum)
  - LC circuit via the acausal library (energy)

Each system is exercised under rk4, dopri5, and bdf with tolerances
chosen so the conservation envelope is tighter than the "useful" solver
accuracy — i.e. a regression in the solver or the block math surfaces
as a drift failure, not a pass by luck.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import Integrator, Adder, Gain

from ._framework import assert_conserved
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# Allowed relative drift per (solver, horizon). rk4 is fixed-step so the
# envelope depends on step size; dopri5 / bdf use rtol=1e-10 so they are
# tighter.
_SOLVER_TOLS = {
    "rk4":    dict(rtol=1e-8,  atol=1e-10, allowed=5e-5),
    "dopri5": dict(rtol=1e-10, atol=1e-12, allowed=1e-7),
    "bdf":    dict(rtol=1e-10, atol=1e-12, allowed=1e-4),  # BDF dissipates more
}


# ── Simple Harmonic Oscillator (energy) ─────────────────────────────────────


class _SHO(jaxonomy.LeafSystem):
    """d²x/dt² + ω²·x = 0 → state = [x, v]. E = 0.5·(v² + ω²·x²)."""

    def __init__(self, omega=2.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("omega", omega)
        self.declare_continuous_state(default_value=jnp.array([1.0, 0.0]), ode=self._ode)

    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        w = params["omega"]
        return jnp.array([v, -(w**2) * x])


@pytest.mark.parametrize("solver", ["rk4", "dopri5", "bdf"])
def test_sho_energy_conserved(solver):
    tols = _SOLVER_TOLS[solver]
    omega = 2.0
    sys = _SHO(omega=omega)
    ctx = sys.create_context()

    def energy(ctx):
        x, v = ctx.continuous_state
        return 0.5 * (v**2 + omega**2 * x**2)

    # 10 oscillation periods; period = 2π/ω ≈ 3.14.
    tf = 10 * 2 * np.pi / omega
    assert_conserved(
        sys, ctx, (0.0, tf), energy,
        solver=solver, **{k: tols[k] for k in ("rtol", "atol")},
        max_major_steps=500,
        allowed_rel_drift=tols["allowed"],
        quantity="SHO energy",
    )


# ── Undamped pendulum (energy) ──────────────────────────────────────────────


class _Pendulum(jaxonomy.LeafSystem):
    """d²θ/dt² + (g/L)·sin(θ) = 0.  E = 0.5·L·ω² + g·(1 − cos(θ)).

    We test the moderate-amplitude regime (θ₀ = 0.4 rad ≈ 23°).
    """

    def __init__(self, g=9.81, L=1.0, theta0=0.4, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter("L", L)
        self.declare_continuous_state(
            default_value=jnp.array([theta0, 0.0]), ode=self._ode
        )

    def _ode(self, time, state, **params):
        theta, w = state.continuous_state
        return jnp.array([w, -(params["g"] / params["L"]) * jnp.sin(theta)])


@pytest.mark.parametrize("solver", ["rk4", "dopri5", "bdf"])
def test_pendulum_energy_conserved(solver):
    tols = _SOLVER_TOLS[solver]
    g, L, theta0 = 9.81, 1.0, 0.4
    sys = _Pendulum(g=g, L=L, theta0=theta0)
    ctx = sys.create_context()

    def energy(ctx):
        theta, w = ctx.continuous_state
        return 0.5 * L * w**2 + g * (1.0 - jnp.cos(theta))

    # Pendulum period (small-amplitude approximation): T ≈ 2π·√(L/g) ≈ 2.006 s
    # → 10 periods ~20 s.
    tf = 20.0
    assert_conserved(
        sys, ctx, (0.0, tf), energy,
        solver=solver, **{k: tols[k] for k in ("rtol", "atol")},
        max_major_steps=5000 if solver == "rk4" else 500,
        allowed_rel_drift=tols["allowed"],
        quantity="pendulum energy",
        extra={"theta0": theta0},
    )


# ── Free rigid body (angular momentum) ──────────────────────────────────────


class _FreeRigidBody(jaxonomy.LeafSystem):
    """Euler's equations, torque-free:
        I·dω/dt = −ω × (I·ω)
    With a diagonal inertia tensor I = diag(Ix, Iy, Iz).

    ||H||² = (Ix·ωx)² + (Iy·ωy)² + (Iz·ωz)² is exactly conserved.
    """

    def __init__(self, I_diag=(1.0, 2.0, 3.0), w0=(1.0, 0.5, 0.2), **kwargs):
        super().__init__(**kwargs)
        self.I_diag = jnp.asarray(I_diag)
        self.declare_continuous_state(
            default_value=jnp.asarray(w0), ode=self._ode
        )

    def _ode(self, time, state, **params):
        w = state.continuous_state
        I = self.I_diag
        H = I * w  # angular momentum components
        dw = -jnp.cross(w, H) / I
        return dw


@pytest.mark.parametrize("solver", ["rk4", "dopri5", "bdf"])
def test_rigid_body_angular_momentum_conserved(solver):
    tols = _SOLVER_TOLS[solver]
    I_diag = (1.0, 2.0, 3.0)
    w0 = (1.0, 0.5, 0.2)
    sys = _FreeRigidBody(I_diag=I_diag, w0=w0)
    ctx = sys.create_context()

    I_arr = jnp.asarray(I_diag)

    def ang_mom_sq(ctx):
        w = ctx.continuous_state
        H = I_arr * w
        return jnp.sum(H**2)

    # 20 s, enough for several tumble periods with these parameters.
    tf = 20.0
    assert_conserved(
        sys, ctx, (0.0, tf), ang_mom_sq,
        solver=solver, **{k: tols[k] for k in ("rtol", "atol")},
        max_major_steps=5000 if solver == "rk4" else 500,
        allowed_rel_drift=tols["allowed"],
        quantity="rigid-body ||H||²",
    )


# ── LC circuit (energy) via acausal library ─────────────────────────────────


def _build_lc():
    """Undamped LC oscillator: inductor + capacitor in a loop.

    E = 0.5·L·I² + 0.5·C·V²  (conserved).
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad = AcausalDiagram()
    L = elec.Inductor(ev, name="L", L=1.0, initial_current=0.0, initial_current_fixed=True)
    C = elec.Capacitor(
        ev, name="C", C=1.0, initial_voltage=1.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(L, "p", C, "p")
    ad.connect(L, "n", gnd, "p")
    ad.connect(C, "n", gnd, "p")
    sys_ = AcausalCompiler(ev, ad)()
    bld = jaxonomy.DiagramBuilder()
    blk = bld.add(sys_)
    return bld.build(), blk


@pytest.mark.parametrize("solver", ["bdf"])  # only BDF supports mass-matrix DAEs
def test_lc_energy_conserved(solver):
    tols = _SOLVER_TOLS[solver]
    diagram, blk = _build_lc()
    ctx = diagram.create_context()
    state0 = np.asarray(ctx[blk.system_id].continuous_state)
    # Identify the inductor current and capacitor voltage by initial values:
    # C.initial_voltage = 1.0 → whichever state equals 1.0 at t=0 is V.
    # L.initial_current = 0.0 → whichever equals 0.0 is I.
    # This gets fragile if more than one state happens to equal 0 or 1; we
    # therefore compute energy by asking for the time derivative and matching
    # against the block's labeling would be overkill.  For a 2-D LC system
    # after alias elimination the state is typically [V_C, I_L, *algebraic].
    # Pick indices by matching against known initial values.
    v_idx = int(np.argmin(np.abs(state0 - 1.0)))
    i_idx = int(np.argmin(np.abs(state0 - 0.0)))

    def energy(ctx_full):
        s = ctx_full[blk.system_id].continuous_state
        V = s[v_idx]
        I = s[i_idx]
        return 0.5 * 1.0 * I**2 + 0.5 * 1.0 * V**2

    # 10 oscillation periods ≈ 62.8 s.
    tf = 10 * 2 * np.pi
    try:
        assert_conserved(
            diagram, ctx, (0.0, tf), energy,
            solver=solver, **{k: tols[k] for k in ("rtol", "atol")},
            max_major_steps=1500,
            allowed_rel_drift=tols["allowed"],
            quantity="LC energy",
            extra={"v_idx": v_idx, "i_idx": i_idx},
        )
    except Exception as e:
        # LC via acausal is an edge case — if the state indexing we did above
        # was wrong (multiple zero-valued entries in the initial state, for
        # instance), surface that as a clear message rather than a silent
        # miss.
        raise AssertionError(
            f"LC conservation test failed: {e}\n"
            f"  initial state = {state0}\n"
            f"  guessed V_idx={v_idx}, I_idx={i_idx}"
        ) from e
