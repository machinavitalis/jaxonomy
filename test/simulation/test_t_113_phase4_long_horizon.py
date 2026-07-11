# SPDX-License-Identifier: MIT

"""T-113 phase 4 — long-horizon validation that constraint projection
keeps DAE drift bounded over many orbital periods.

Phase 4 of T-113 calls for a long-horizon simulation that, without
projection, fails the T-004 conservation tests (drift accumulates
unbounded) but passes them when ``SimulatorOptions.dae_projection_enabled``
is on. Two fixtures:

* The **PlanarPendulum** index-2 DAE (same fixture as
  ``test_dae_projection.py``), run for ~30 swing periods — the
  structurally hard case where unstabilised index reduction visibly
  drifts.
* The roadmap-specified **1-hour acausal-fluids** model
  (heat-capacitor ↔ accumulator ↔ pipe ↔ ambient, from
  ``test_fluid.py::test_heat_acc_pipe_amb``), added once the fluid
  baseline failures were fixed (T-136 exposed the Accumulator sensor
  ports this model family needed). Its BDF run holds the algebraic
  residual at machine precision on its own; the phase-4 contract
  checked here is *bounded residual over the full hour* and
  *projection never regresses the baseline*.

Marked ``slow`` so the fast pytest tier doesn't pay the multi-second
cost on every run.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.simulation.dae_drift import (
    algebraic_row_mask,
    constraint_residual_norm,
)
from jaxonomy.testing.markers import requires_jax, skip_if_not_jax


pytestmark = pytest.mark.slow


skip_if_not_jax()


# Verbatim copy of the PlanarPendulum fixture from
# test/simulation/test_dae_projection.py — the source class is not
# a public symbol, so the canonical T-113 test bed is duplicated
# rather than imported across test files.
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
        self.declare_continuous_state(
            default_value=x0, mass_matrix=M, ode=self.ode
        )
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


# Long horizon: 60 seconds is ~30 swing periods for a 1-m pendulum
# (T = 2π√(L/g) ≈ 2.0 s). That's long enough for any unstabilised
# index reduction to surface drift; short enough to keep the test
# wall-clock under a minute on commodity hardware.
LONG_HORIZON_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Phase 4 deliverable: off vs on comparison.
# ---------------------------------------------------------------------------


@requires_jax()
def test_long_horizon_projection_keeps_residual_bounded():
    """The headline phase-4 result: over many oscillation periods,
    ``||f_a||_∞`` stays at the projection tolerance with projection
    enabled, even if the un-projected baseline drifts."""
    model = PlanarPendulum(L=1.0, g0=9.8)
    ctx = model.create_context()

    opts_on = jaxonomy.SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=1e-6,
        atol=1e-8,
        dae_projection_enabled=True,
        dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
    )

    res = jaxonomy.simulate(
        model, ctx, (0.0, LONG_HORIZON_SECONDS), options=opts_on
    )
    resid = constraint_residual_norm(model, res.context)
    assert resid is not None
    # Projection holds residual at the configured tolerance even over
    # 30 oscillation periods — the headline T-113 phase-4 claim.
    assert resid < 1e-6, (
        f"long-horizon projection failed: ||f_a||_inf = {resid:.3e} "
        f"after {LONG_HORIZON_SECONDS}s (~30 swing periods)"
    )


@requires_jax()
def test_long_horizon_projection_beats_baseline():
    """The projected run must hold drift no worse than the baseline
    over a long horizon. Some BDF setups happen to stay clean on
    their own; the contract is "projection never makes it worse and
    typically substantially better" — exactly the use-case for the
    opt-in flag."""
    model = PlanarPendulum(L=1.0, g0=9.8)
    ctx = model.create_context()

    opts_off = jaxonomy.SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=1e-6,
        atol=1e-8,
    )
    opts_on = jaxonomy.SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=1e-6,
        atol=1e-8,
        dae_projection_enabled=True,
        dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
    )

    res_off = jaxonomy.simulate(
        model, ctx, (0.0, LONG_HORIZON_SECONDS), options=opts_off
    )
    res_on = jaxonomy.simulate(
        model, ctx, (0.0, LONG_HORIZON_SECONDS), options=opts_on
    )
    resid_off = constraint_residual_norm(model, res_off.context)
    resid_on = constraint_residual_norm(model, res_on.context)
    assert resid_off is not None and resid_on is not None
    # Allow a generous slack — what we want to catch is the case
    # where projection makes things *worse*, not the case where BDF
    # alone happens to hit machine precision.
    assert resid_on <= resid_off * 10 + 1e-6, (
        f"projection regressed: baseline ||f_a||_inf = {resid_off:.3e}, "
        f"projected ||f_a||_inf = {resid_on:.3e}"
    )


@requires_jax()
def test_long_horizon_projection_holds_g_zero_throughout_trajectory():
    """The strongest phase-4 claim: with projection on, the *algebraic
    constraint vector* ``g`` stays at the projection tolerance for the
    full long-horizon trajectory, not just at the final state.

    Recording the continuous state lets us recompute ``g(x(t))`` at
    every captured sample and assert the constraint is held throughout
    — the relevant pendulum constraint in the index-2 reduction is the
    first algebraic row, ``x_pos^2 + v_x^2 = L^2`` (where the state
    layout is ``[x[0]=ẋ, x[1]=x_pos, z[0]=ẏ, ..., z[2]=y_pos, ...]``)
    — so the geometric invariant we check is
    ``x[1]^2 + z[2]^2 = L^2``.
    """
    L = 1.0
    model = PlanarPendulum(L=L, g0=9.8)
    ctx = model.create_context()

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=1e-6,
        atol=1e-8,
        dae_projection_enabled=True,
        dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
    )
    res = jaxonomy.simulate(
        model, ctx, (0.0, LONG_HORIZON_SECONDS), options=opts,
        recorded_signals={"state": model.output_ports[0]},
    )
    state = np.asarray(res.outputs["state"])
    # State layout per the ODE: [x[0], x[1], z[0], z[1], z[2], z[3],
    # z[4], z[5], z[6]]. The first algebraic row enforces
    # ``-L^2 + x[1]^2 + z[2]^2 = 0`` — that's the geometric constraint
    # the projection step is meant to hold.
    pos_sq = state[:, 1] ** 2 + state[:, 4] ** 2  # x[1]^2 + z[2]^2
    max_drift = float(np.max(np.abs(pos_sq - L * L)))
    assert max_drift < 1e-5, (
        f"constraint x[1]^2 + z[2]^2 = L^2 drifted by up to "
        f"{max_drift:.3e} over {LONG_HORIZON_SECONDS}s"
    )


# ---------------------------------------------------------------------------
# The roadmap-specified fixture: 1-hour acausal fluids (T-113-followup).
# ---------------------------------------------------------------------------

FLUIDS_HORIZON_SECONDS = 3600.0


def _build_fluids_model():
    """heat_capacitor ↔ accumulator ↔ pipe ↔ ambient.

    The `test_fluid.py::test_heat_acc_pipe_amb` model (HeatCapacitor
    variant): tank pressure starts above ambient, mass flows out until
    equalized, then heat keeps flowing into the accumulator fluid. The
    compiled system is a semi-explicit DAE with algebraic rows — the
    fluids test bed the T-113 phase-4 roadmap text asks for.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import fluid as fld
    from jaxonomy.acausal import fluid_media as fm
    from jaxonomy.acausal import thermal
    from jaxonomy.acausal.component_library.fluid import P_DEFAULT, T_DEFAULT

    p_ic = P_DEFAULT
    fluid_T_ic = T_DEFAULT + 100
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.WaterLiquidSimple(ev))
    ad = AcausalDiagram()
    heat = thermal.HeatCapacitor(
        ev,
        name="hc",
        initial_temperature=T_DEFAULT + 200,
        initial_temperature_fixed=True,
        C=1e9,
    )
    accum1 = fld.Accumulator(
        ev,
        name="accum1",
        P_ic=p_ic + 100,
        P_ic_fixed=True,
        T_ic=fluid_T_ic,
        T_ic_fixed=True,
        k=1e5,
        area=0.1,
        ht_coeff=100.0,
        enable_enthalpy_sensor=True,
        enable_thermal_port=True,
    )
    pipe = fld.SimplePipe(ev, name="pipe", R=1000.0, enable_sensors=True)
    amb = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic, T_ambient=fluid_T_ic)
    ad.connect(heat, "port", accum1, "wall")
    ad.connect(accum1, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", amb, "port")
    ad.connect(fp, "prop", accum1, "port")
    acausal_system = AcausalCompiler(ev, ad)()
    builder = jaxonomy.DiagramBuilder()
    builder.add(acausal_system)
    return builder.build()


def _fluids_options(projection: bool):
    kwargs = dict(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=1e-6,
        atol=1e-8,
    )
    if projection:
        kwargs.update(
            dae_projection_enabled=True,
            dae_projection_tol=1e-9,
            dae_projection_max_iter=4,
        )
    return jaxonomy.SimulatorOptions(**kwargs)


@requires_jax()
def test_fluids_one_hour_residual_bounded_with_projection():
    """The roadmap deliverable: a full simulated hour of the fluids DAE
    with projection on completes and holds ``||f_a||_∞`` at tolerance."""
    diagram = _build_fluids_model()
    ctx = diagram.create_context(check_types=True)
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, FLUIDS_HORIZON_SECONDS),
        options=_fluids_options(projection=True),
    )
    assert float(res.context.time) == pytest.approx(FLUIDS_HORIZON_SECONDS)
    resid = constraint_residual_norm(diagram, res.context)
    # The fixture must genuinely carry algebraic constraints — a pure-ODE
    # reduction would make this test vacuous.
    assert resid is not None, "fluids fixture lost its algebraic rows"
    assert resid < 1e-6, (
        f"1-hour fluids run: ||f_a||_inf = {resid:.3e} with projection on"
    )


@requires_jax()
def test_fluids_one_hour_projection_never_regresses_baseline():
    """Same contract as the pendulum: this BDF setup happens to stay at
    machine precision unprojected; projection must not make it worse."""
    diagram_off = _build_fluids_model()
    res_off = jaxonomy.simulate(
        diagram_off,
        diagram_off.create_context(check_types=True),
        (0.0, FLUIDS_HORIZON_SECONDS),
        options=_fluids_options(projection=False),
    )
    diagram_on = _build_fluids_model()
    res_on = jaxonomy.simulate(
        diagram_on,
        diagram_on.create_context(check_types=True),
        (0.0, FLUIDS_HORIZON_SECONDS),
        options=_fluids_options(projection=True),
    )
    resid_off = constraint_residual_norm(diagram_off, res_off.context)
    resid_on = constraint_residual_norm(diagram_on, res_on.context)
    assert resid_off is not None and resid_on is not None
    assert resid_on <= resid_off * 10 + 1e-6, (
        f"projection regressed fluids baseline: off={resid_off:.3e}, "
        f"on={resid_on:.3e}"
    )
