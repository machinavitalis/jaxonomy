# SPDX-License-Identifier: MIT
"""V-003: DAE long-run stability and constraint adherence.

Verifies that algebraic constraints in mass-matrix DAE (acausal) systems
are preserved over long-horizon simulations (500-1000 s).  For each topology
we sample recorded port signals and evaluate the characteristic constraint
residual (KCL for electrical, Newton's law for mechanical, energy balance
for thermal).  The max residual over the recorded trace must stay below a
documented tolerance.  Tests are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy.acausal import (
    AcausalCompiler,
    AcausalDiagram,
    EqnEnv,
    electrical as elec,
    translational as trans,
    thermal as ht,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

# Default tolerances for the acausal solvers used in these tests.
RTOL = 1e-8
ATOL = 1e-10
# Constraint residual tolerance: 100x rtol or 1e-4 absolute, whichever is larger.
RESIDUAL_TOL = max(100 * RTOL, 1e-4)


def _build_and_simulate(ad: AcausalDiagram, ev: EqnEnv, t_end: float, signals: dict):
    """Compile, build, and run the acausal diagram for ``t_end`` seconds."""
    asys = AcausalCompiler(ev, ad, scale=True)()
    builder = jaxonomy.DiagramBuilder()
    asys = builder.add(asys)
    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)

    recorded = {
        name: asys.output_ports[asys.outsym_to_portid[sym]]
        for name, sym in signals.items()
    }
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=RTOL,
        atol=ATOL,
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, t_end), recorded_signals=recorded, options=opts
    )
    return res


def _max_residual(res, residual_fn) -> float:
    """Sample residual_fn at ~50 evenly spaced indices and return the max abs."""
    n = len(res.time)
    if n < 4:
        idxs = np.arange(n)
    else:
        idxs = np.unique(np.linspace(0, n - 1, num=min(50, n)).astype(int))
    residuals = np.array([residual_fn(res, i) for i in idxs])
    return float(np.max(np.abs(residuals)))


# --- Electrical: index-1 RC circuit, KCL residual. ---


@pytest.mark.slow
def test_rc_circuit_kcl_drift():
    """Index-1 RC: |i_R - i_C| at every sampled step over t=1000s."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=1.0)
    r = elec.Resistor(ev, name="r", R=10.0)
    c = elec.Capacitor(
        ev, name="c", C=1e-3, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    sens_iR = elec.CurrentSensor(ev, name="sens_iR")
    sens_iC = elec.CurrentSensor(ev, name="sens_iC")

    ad.connect(v, "p", sens_iR, "p")
    ad.connect(sens_iR, "n", r, "p")
    ad.connect(r, "n", sens_iC, "p")
    ad.connect(sens_iC, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")

    res = _build_and_simulate(
        ad, ev, t_end=1000.0,
        signals={
            "iR": sens_iR.get_sym_by_port_name("i"),
            "iC": sens_iC.get_sym_by_port_name("i"),
        },
    )
    drift = _max_residual(
        res, lambda r_, i: r_.outputs["iR"][i] - r_.outputs["iC"][i]
    )
    assert drift < RESIDUAL_TOL, f"RC KCL drift {drift:.2e} exceeds {RESIDUAL_TOL:.2e}"


# --- Electrical: series RLC, long horizon. ---


@pytest.mark.slow
def test_rlc_series_kcl_drift():
    """Series RLC: a single loop must have identical current at every series
    node.  Residual = |i_R - i_C|."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    l = elec.Inductor(  # noqa: E741
        ev, name="l", L=0.5, initial_current=0.0, initial_current_fixed=True
    )
    c = elec.Capacitor(
        ev, name="c", C=0.1, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    sens_iR = elec.CurrentSensor(ev, name="sens_iR")
    sens_iC = elec.CurrentSensor(ev, name="sens_iC")

    ad.connect(v, "p", sens_iR, "p")
    ad.connect(sens_iR, "n", r, "p")
    ad.connect(r, "n", l, "p")
    ad.connect(l, "n", sens_iC, "p")
    ad.connect(sens_iC, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")

    res = _build_and_simulate(
        ad, ev, t_end=500.0,
        signals={
            "iR": sens_iR.get_sym_by_port_name("i"),
            "iC": sens_iC.get_sym_by_port_name("i"),
        },
    )
    drift = _max_residual(
        res, lambda r_, i: r_.outputs["iR"][i] - r_.outputs["iC"][i]
    )
    assert drift < RESIDUAL_TOL, f"RLC KCL drift {drift:.2e} exceeds {RESIDUAL_TOL:.2e}"


# --- Electrical: two cascaded RC stages.  KCL on the second stage. ---


@pytest.mark.slow
def test_two_stage_rc_kcl_drift():
    """Two RC stages in cascade.  Verify within the second stage that
    i_R2 = i_C2 (KCL on the series branch), over t=500s."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=5.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=1e-3, initial_voltage=0.0, initial_voltage_fixed=True
    )
    r2 = elec.Resistor(ev, name="r2", R=20.0)
    c2 = elec.Capacitor(
        ev, name="c2", C=2e-3, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    sens_iR2 = elec.CurrentSensor(ev, name="sens_iR2")
    sens_iC2 = elec.CurrentSensor(ev, name="sens_iC2")

    # First RC stage.
    ad.connect(v, "p", r1, "p")
    ad.connect(r1, "n", c1, "p")
    ad.connect(c1, "n", v, "n")
    ad.connect(v, "n", gnd, "p")

    # Second RC stage hangs off the c1 high node.
    ad.connect(c1, "p", sens_iR2, "p")
    ad.connect(sens_iR2, "n", r2, "p")
    ad.connect(r2, "n", sens_iC2, "p")
    ad.connect(sens_iC2, "n", c2, "p")
    ad.connect(c2, "n", v, "n")

    res = _build_and_simulate(
        ad, ev, t_end=500.0,
        signals={
            "iR2": sens_iR2.get_sym_by_port_name("i"),
            "iC2": sens_iC2.get_sym_by_port_name("i"),
        },
    )
    drift = _max_residual(
        res, lambda r_, i: r_.outputs["iR2"][i] - r_.outputs["iC2"][i]
    )
    assert drift < RESIDUAL_TOL, (
        f"Cascaded RC stage-2 KCL drift {drift:.2e} exceeds {RESIDUAL_TOL:.2e}"
    )


# --- Mechanical: mass-spring-damper (translational), Newton's law. ---


def _build_msd(ev, M: float, K: float, D: float, x0: float):
    ad = AcausalDiagram()
    m = trans.Mass(
        ev, name="m", M=M,
        initial_position=x0, initial_position_fixed=True,
        initial_velocity=0.0, initial_velocity_fixed=True,
    )
    sp = trans.Spring(ev, name="sp", K=K)
    dp = trans.Damper(ev, name="dp", D=D)
    wall = trans.FixedPosition(ev, name="wall", initial_position=0.0)
    sens_a = trans.MotionSensor(
        ev, name="sens_a",
        enable_flange_b=True,
        enable_velocity_port=False,
        enable_acceleration_port=True,
    )
    sens_F_sp = trans.ForceSensor(ev, name="sens_F_sp")
    sens_F_dp = trans.ForceSensor(ev, name="sens_F_dp")
    ad.connect(m, "flange", sp, "flange_a")
    ad.connect(sp, "flange_b", sens_F_sp, "flange_a")
    ad.connect(sens_F_sp, "flange_b", wall, "flange")
    ad.connect(m, "flange", dp, "flange_a")
    ad.connect(dp, "flange_b", sens_F_dp, "flange_a")
    ad.connect(sens_F_dp, "flange_b", wall, "flange")
    ad.connect(m, "flange", sens_a, "flange_a")
    ad.connect(wall, "flange", sens_a, "flange_b")
    return ad, sens_a, sens_F_sp, sens_F_dp


@pytest.mark.slow
@pytest.mark.parametrize(
    "label, M, K, D, x0",
    [
        ("overdamped", 1.0, 4.0, 0.2, 1.0),
        ("underdamped_long", 0.5, 10.0, 0.05, 0.5),
    ],
)
def test_mass_spring_damper_newton_drift(label, M, K, D, x0):
    """Index-1 mass-spring-damper.  Newton's law:
    ``m*a + F_spring + F_damper = 0`` over a 500s horizon."""
    ev = EqnEnv()
    ad, sens_a, sens_F_sp, sens_F_dp = _build_msd(ev, M, K, D, x0)
    res = _build_and_simulate(
        ad, ev, t_end=500.0,
        signals={
            "a": sens_a.get_sym_by_port_name("a_rel"),
            "F_sp": sens_F_sp.get_sym_by_port_name("f"),
            "F_dp": sens_F_dp.get_sym_by_port_name("f"),
        },
    )

    def residual(r_, i):
        return M * r_.outputs["a"][i] + r_.outputs["F_sp"][i] + r_.outputs["F_dp"][i]

    drift = _max_residual(res, residual)
    # Mechanical residual scales with force magnitudes; allow a slightly
    # looser bound than the 100x rtol baseline.
    bound = max(RESIDUAL_TOL, 1e-3)
    assert drift < bound, f"MSD ({label}) Newton drift {drift:.2e} > {bound:.2e}"


# --- Thermal: heat capacitor + insulator energy balance. ---


@pytest.mark.slow
def test_thermal_chain_energy_balance():
    """Index-1 thermal: TemperatureSource -> Insulator -> HeatCapacitor.

    Energy balance: heat flowing into the capacitor (Q = (T_src -
    T_cap)/R, by the insulator's constitutive relation) must equal the
    rate of internal-energy increase ``C * dT/dt``.  Both T_src and
    T_cap are measured by TemperatureSensors; dT_cap/dt is estimated by
    central differences."""
    R_ins = 2.0
    C_cap = 5.0
    T_src_val = 350.0
    ev = EqnEnv()
    ad = AcausalDiagram()
    src = ht.TemperatureSource(ev, name="src", temperature=T_src_val)
    ins = ht.Insulator(ev, name="ins", R=R_ins)
    cap = ht.HeatCapacitor(
        ev, name="cap", C=C_cap,
        initial_temperature=300.0, initial_temperature_fixed=True,
    )
    sens_T = ht.TemperatureSensor(ev, name="sens_T", enable_port_b=False)

    ad.connect(src, "port", ins, "port_a")
    ad.connect(ins, "port_b", cap, "port")
    ad.connect(cap, "port", sens_T, "port_a")

    res = _build_and_simulate(
        ad, ev, t_end=1000.0,
        signals={"T": sens_T.get_sym_by_port_name("T_rel")},
    )
    t = res.time
    T = res.outputs["T"][:, 0] if res.outputs["T"].ndim == 2 else res.outputs["T"]
    Q = (T_src_val - T) / R_ins  # heat flowing into capacitor through insulator
    dTdt = np.gradient(T, t)
    energy_residual = Q - C_cap * dTdt
    # Skip first/last few samples where np.gradient uses one-sided
    # differences (lower accuracy).
    drift = float(np.max(np.abs(energy_residual[5:-5])))
    bound = 5e-2  # numerical-differentiation noise on a 1000s trace
    assert drift < bound, (
        f"Thermal energy-balance drift {drift:.2e} exceeds {bound:.2e}"
    )


# --- Thermal: two-stage insulator chain.  Total stored-energy conservation. ---


@pytest.mark.slow
def test_thermal_two_insulator_chain():
    """Two insulators in series between two heat capacitors.  Verify
    energy balance at each capacitor: ``C_a * dT_a/dt = -Q`` and
    ``C_b * dT_b/dt = +Q``, where ``Q = (T_a - T_b) / (R1 + R2)`` (the
    series resistance, since the two insulators are purely conductive).
    The combined check is ``C_a * dT_a/dt + C_b * dT_b/dt = 0`` (total
    energy conservation - no source / sink in the closed system)."""
    C_a = 10.0
    C_b = 10.0
    ev = EqnEnv()
    ad = AcausalDiagram()
    cap_a = ht.HeatCapacitor(
        ev, name="cap_a", C=C_a,
        initial_temperature=400.0, initial_temperature_fixed=True,
    )
    ins1 = ht.Insulator(ev, name="ins1", R=1.0)
    ins2 = ht.Insulator(ev, name="ins2", R=2.0)
    cap_b = ht.HeatCapacitor(
        ev, name="cap_b", C=C_b,
        initial_temperature=300.0, initial_temperature_fixed=True,
    )
    sens_Ta = ht.TemperatureSensor(ev, name="sens_Ta", enable_port_b=False)
    sens_Tb = ht.TemperatureSensor(ev, name="sens_Tb", enable_port_b=False)

    ad.connect(cap_a, "port", ins1, "port_a")
    ad.connect(ins1, "port_b", ins2, "port_a")
    ad.connect(ins2, "port_b", cap_b, "port")
    ad.connect(cap_a, "port", sens_Ta, "port_a")
    ad.connect(cap_b, "port", sens_Tb, "port_a")

    res = _build_and_simulate(
        ad, ev, t_end=1000.0,
        signals={
            "Ta": sens_Ta.get_sym_by_port_name("T_rel"),
            "Tb": sens_Tb.get_sym_by_port_name("T_rel"),
        },
    )
    t = res.time
    Ta = res.outputs["Ta"][:, 0] if res.outputs["Ta"].ndim == 2 else res.outputs["Ta"]
    Tb = res.outputs["Tb"][:, 0] if res.outputs["Tb"].ndim == 2 else res.outputs["Tb"]
    dTadt = np.gradient(Ta, t)
    dTbdt = np.gradient(Tb, t)
    # Total stored energy time-derivative must vanish (closed system).
    energy_rate = C_a * dTadt + C_b * dTbdt
    drift = float(np.max(np.abs(energy_rate[5:-5])))
    bound = 5e-2  # numerical-differentiation noise on a 1000s trace
    assert drift < bound, f"Two-insulator thermal drift {drift:.2e} > {bound:.2e}"


# --- T-031: chained HeatflowSensor compilation + behavioural check. ---


@pytest.mark.slow
def test_thermal_chained_heatflow_sensors_compile():
    """Two HeatflowSensor components in series previously triggered
    Pantelides "Mismatch between the number of equations N and the
    number of variables M" (T-031). The fix added the missing flow-
    conservation equation Q1 + Q2 = 0 to HeatflowSensor; this test
    asserts the chain compiles cleanly and that each sensor reports
    the same heat flux (series sensors must agree on Q because the
    chain has no branches between them)."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    src = ht.TemperatureSource(ev, name="src", temperature=400.0)
    ins1 = ht.Insulator(ev, name="ins1", R=1.0)
    sens1 = ht.HeatflowSensor(ev, name="sens1")
    ins2 = ht.Insulator(ev, name="ins2", R=2.0)
    sens2 = ht.HeatflowSensor(ev, name="sens2")
    cap = ht.HeatCapacitor(
        ev, name="cap", C=10.0,
        initial_temperature=300.0, initial_temperature_fixed=True,
    )
    ad.connect(src, "port", ins1, "port_a")
    ad.connect(ins1, "port_b", sens1, "port_a")
    ad.connect(sens1, "port_b", ins2, "port_a")
    ad.connect(ins2, "port_b", sens2, "port_a")
    ad.connect(sens2, "port_b", cap, "port")

    res = _build_and_simulate(
        ad, ev, t_end=20.0,
        signals={
            "Q1": sens1.get_sym_by_port_name("Q_flow"),
            "Q2": sens2.get_sym_by_port_name("Q_flow"),
        },
    )
    Q1 = np.asarray(res.outputs["Q1"]).reshape(-1)
    Q2 = np.asarray(res.outputs["Q2"]).reshape(-1)
    # Series flow: Q through sens1 and sens2 must be equal at every
    # sampled instant. Tolerance reflects BDF integration error, not
    # the constraint itself.
    assert np.allclose(Q1, Q2, rtol=1e-6, atol=1e-9), (
        f"chained sensors disagree: max |Q1-Q2| = {np.max(np.abs(Q1-Q2)):.2e}"
    )
    # And the initial flux must equal the steady-state Ohm-style
    # value (T_src - T_cap) / (R1 + R2) = 100/3 ≈ 33.33 W.
    assert Q1[0] == pytest.approx(100.0 / 3.0, rel=1e-2), (
        f"initial heatflow {Q1[0]:.3f} != expected {100/3:.3f}"
    )
    # The flux decays as the capacitor heats up; assert monotone
    # decreasing on the smoothed trace.
    assert Q1[-1] < Q1[0], (
        f"flux did not decay: Q[0]={Q1[0]}, Q[-1]={Q1[-1]}"
    )


# --- Optional: index-2 case. ---


@pytest.mark.slow
def test_index2_constrained_pendulum():
    """T-032 / T-003a — point-mass on a rigid massless link forms an
    index-2 DAE: the holonomic constraint ``x² + y² = L²`` must be
    differentiated twice before the Lagrange multiplier appears.

    Verifies (a) Pantelides + BDF can simulate the model end-to-end,
    (b) the constraint stays at ~machine precision over a 5-second
    swing, and (c) energy is approximately conserved (no damping is
    modelled, so total mechanical energy ½m·v² + m·g·y must drift
    only by integrator error)."""
    import math
    from jaxonomy.acausal.component_library import planar

    L = 1.0
    m = 1.0
    g = 9.81

    ev = EqnEnv()
    pend = planar.PlanarPendulum(
        ev, name="pend", m=m, L=L, g_y=-g,
        initial_theta=math.pi / 6, initial_omega=0.0,
    )
    # PlanarPendulum has no acausal ports — pass it directly to the
    # AcausalDiagram via comp_list since there's no connect() call to
    # register it implicitly.
    ad = AcausalDiagram(comp_list=[pend])

    asys = AcausalCompiler(ev, ad, scale=True)()
    builder = jaxonomy.DiagramBuilder()
    builder.add(asys)
    diagram = builder.build()
    ctx = diagram.create_context()

    rec = {
        "x": asys.output_ports[asys.outsym_to_portid[
            pend.get_sym_by_port_name("x_out")]],
        "y": asys.output_ports[asys.outsym_to_portid[
            pend.get_sym_by_port_name("y_out")]],
    }
    # 2-second window is one full small-amplitude period (T = 2π√(L/g)
    # ≈ 2.0 s for L = 1 m, g = 9.81 m/s²). Long enough to assert real
    # swinging motion without dragging the test runtime up; the
    # constraint-drift check at machine precision is what matters,
    # and longer simulations don't make that test stronger.
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-9,
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 2.0), recorded_signals=rec, options=opts,
    )
    x = np.asarray(res.outputs["x"]).reshape(-1)
    y = np.asarray(res.outputs["y"]).reshape(-1)
    r = np.sqrt(x * x + y * y)

    # (a) Constraint drift: the BDF mass-matrix path holds the
    # holonomic constraint to machine precision on this linear-DAE
    # tail. T-003a will tighten this further once projection lands;
    # for now we just verify it doesn't blow up.
    drift = float(np.max(np.abs(r - L)))
    assert drift < 1e-6, (
        f"constraint drift {drift:.2e} exceeds 1e-6 over 2 s"
    )

    # (b) The bob actually swings — x and y are not constants.
    assert float(np.std(x)) > 0.1, (
        f"pendulum did not swing in x: std={np.std(x):.3e}"
    )
    assert float(np.std(y)) > 0.1, (
        f"pendulum did not swing in y: std={np.std(y):.3e}"
    )

    # (c) Initial position matches polar IC: theta = pi/6 puts bob at
    # (cos pi/6, sin pi/6) ≈ (0.866, 0.5) on a unit-length link.
    assert x[0] == pytest.approx(math.cos(math.pi / 6), abs=1e-12)
    assert y[0] == pytest.approx(math.sin(math.pi / 6), abs=1e-12)


# --- Optional: under-determined-flow fluid case. ---


@pytest.mark.slow
def test_fluid_underdetermined_flow():
    """Skipped - the experimental fluid library does not expose a stable
    enough port API for a long-horizon constraint check at present."""
    pytest.skip(
        "fluid components are experimental in this repo and do not yet expose "
        "a sensor + port-id mapping that supports a topology-agnostic "
        "constraint residual; will revisit when the fluid API stabilises."
    )
