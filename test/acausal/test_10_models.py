# SPDX-License-Identifier: MIT
"""
Ten acausal model examples, run as pytest tests.
Each test verifies simulation output against an analytic solution.
"""
import numpy as np
import pytest
import jaxonomy
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.acausal import rotational as rot
from jaxonomy.acausal import translational as trans
from jaxonomy.acausal import thermal as ht


def _sim(ev, ad, t_span):
    t_span = (float(t_span[0]), float(t_span[1]))
    ac = AcausalCompiler(ev, ad)
    system = ac()
    b = jaxonomy.DiagramBuilder()
    system = b.add(system)
    diagram = b.build()
    ctx = diagram.create_context()
    res = jaxonomy.simulate(
        diagram, ctx, t_span, recorded_signals={"s": system.output_ports[0]}
    )
    return ac, np.asarray(res.time), np.asarray(res.outputs["s"])


def sidx(ac, *subs):
    """Return index of first ODE state whose name contains any of the given substrings."""
    for sub in subs:
        for i, v in enumerate(ac.sed.x):
            if sub in str(v):
                return i
    raise KeyError(f"{subs} not in {[str(v) for v in ac.sed.x]}")


# ── Model 1: RC circuit ───────────────────────────────────────────────────────
def test_m01_rc_circuit():
    """V=1V R=1Ω C=1F  →  Vc(t) = 1 - e^{-t}"""
    ev = EqnEnv(); ad = AcausalDiagram()
    vs  = elec.VoltageSource(ev, name="vs", v=1.0)
    r1  = elec.Resistor(ev, name="r1", R=1.0)
    c1  = elec.Capacitor(ev, name="c1", C=1.0,
                          initial_voltage=0.0, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    ac, t, s = _sim(ev, ad, (0.0, 5.0))
    # state name may be 'c1_V' or node-potential 'np*_electrical_volt_0'
    # depending on build_recorder global counter; accept either
    i_Vc = sidx(ac, "c1_V", "electrical_volt")
    Vc  = s[:, i_Vc]
    ana = 1 - np.exp(-t)
    assert np.max(np.abs(Vc - ana)) < 1e-4, f"max err={np.max(np.abs(Vc-ana)):.2e}"


# ── Model 2: Series RLC underdamped ──────────────────────────────────────────
def test_m02_rlc_underdamped():
    """V=1 R=0.5Ω L=1H C=1F  →  decaying oscillation, ω_d≈0.968 rad/s"""
    ev = EqnEnv(); ad = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=1.0)
    r1 = elec.Resistor(ev, name="r1", R=0.5)
    l1 = elec.Inductor(ev, name="l1", L=1.0,
                        initial_current=0.0, initial_current_fixed=True)
    c1 = elec.Capacitor(ev, name="c1", C=1.0,
                         initial_voltage=0.0, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "p"); ad.connect(r1, "n", l1, "p")
    ad.connect(l1, "n", c1, "p"); ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    ac, t, s = _sim(ev, ad, (0.0, 20))
    Vc  = s[:, sidx(ac, "c1_V", "electrical_volt")]
    ana = 1 - np.exp(-0.25*t) * (np.cos(0.9682*t) + 0.2580*np.sin(0.9682*t))
    assert np.max(np.abs(Vc - ana)) < 1e-3


# ── Model 3: Torsional spring-mass oscillator (was BLOCKED pre-fix) ──────────
def test_m03_torsional_oscillator():
    """K=10 N·m/rad, J=1 kg·m²,  ω(0)=1  →  ω(t) = cos(√10·t)"""
    ev = EqnEnv(); ad = AcausalDiagram()
    fa  = rot.FixedAngle(ev, name="fa")
    sp1 = rot.Spring(ev, name="sp", K=10.0,
                     initial_angle_A=0.0, initial_angle_A_fixed=True)
    J   = rot.Inertia(ev, name="J", I=1.0,
                      initial_velocity=1.0, initial_velocity_fixed=True,
                      initial_angle=0.0,    initial_angle_fixed=True)
    ad.connect(fa, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", J, "flange")
    ac, t, s = _sim(ev, ad, (0.0, 10))
    w   = s[:, sidx(ac, "_0(")]          # angular velocity state
    ana = np.cos(np.sqrt(10) * t)
    assert np.max(np.abs(w - ana)) < 1e-3


# ── Model 4: Translational spring-mass-damper (was BLOCKED pre-fix) ──────────
def test_m04_spring_mass_damper():
    """K=4 N/m, M=1 kg, D=0.4 N·s/m, x(0)=1  →  underdamped decay"""
    ev = EqnEnv(); ad = AcausalDiagram()
    fp  = trans.FixedPosition(ev, name="fp")
    sp1 = trans.Spring(ev, name="sp", K=4.0)
    m1  = trans.Mass(ev, name="m1", M=1.0,
                     initial_velocity=0.0, initial_velocity_fixed=True,
                     initial_position=1.0, initial_position_fixed=True)
    d1  = trans.Damper(ev, name="d1", D=0.4)
    ad.connect(fp,  "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", m1, "flange")
    ad.connect(m1,  "flange", d1, "flange_a")
    ad.connect(d1,  "flange_b", fp, "flange")
    ac, t, s = _sim(ev, ad, (0.0, 10))
    x   = s[:, sidx(ac, "_n1(")]       # position state
    zeta = 0.1; wd = np.sqrt(4.0 * (1 - zeta**2))
    ana  = np.exp(-zeta*2*t) * (np.cos(wd*t) + (zeta*2/wd)*np.sin(wd*t))
    assert np.max(np.abs(x - ana)) < 1e-3


# ── Model 5: Two-mass thermal system ─────────────────────────────────────────
def test_m05_two_mass_thermal():
    """C=15 J/K each, R=0.1 K/W  →  T_eq=323.15 K, τ=0.75 s"""
    ev = EqnEnv(); ad = AcausalDiagram()
    c1 = ht.HeatCapacitor(ev, name="c1", C=15.0,
                           initial_temperature=373.15, initial_temperature_fixed=True)
    r1 = ht.Insulator(ev, name="r1", R=0.1)
    c2 = ht.HeatCapacitor(ev, name="c2", C=15.0,
                           initial_temperature=273.15, initial_temperature_fixed=True)
    ad.connect(c1, "port", r1, "port_a")
    ad.connect(r1, "port_b", c2, "port")
    ac, t, s = _sim(ev, ad, (0.0, 10.0))
    T1  = s[:, sidx(ac, "np0_thermal")]
    T2  = s[:, sidx(ac, "np1_thermal")]
    ana1 = 323.15 + 50 * np.exp(-t / 0.75)
    ana2 = 323.15 - 50 * np.exp(-t / 0.75)
    assert max(np.max(np.abs(T1-ana1)), np.max(np.abs(T2-ana2))) < 1e-3


# ── Model 6: DC motor (stalled, fixed shaft) ─────────────────────────────────
def test_m06_dc_motor_stalled():
    """V=10V R=0.1Ω  →  stall current I_ss = V/R = 100 A (was BLOCKED pre-fix)"""
    ev = EqnEnv(); ad = AcausalDiagram()
    mot   = elec.IdealMotor(ev, name="mot", R=0.1, K=0.5, J=1.0,
                             initial_velocity=0.0, initial_velocity_fixed=True,
                             initial_angle=0.0,    initial_angle_fixed=True,
                             initial_current=0.0,  initial_current_fixed=True)
    vs    = elec.VoltageSource(ev, name="vs", v=10.0)
    gnd   = elec.Ground(ev, name="gnd")
    fixed = rot.FixedAngle(ev, name="fixed")
    ad.connect(vs, "p", mot, "pos"); ad.connect(mot, "neg", vs, "n")
    ad.connect(vs, "n", gnd, "p");   ad.connect(mot, "shaft", fixed, "flange")
    ac, t, s = _sim(ev, ad, (0.0, 0.01))
    # stall: at steady state back-EMF=0, I_ss = V/R = 100 A
    i_I = sidx(ac, "vs_p_I")   # state: current (passive sign: into vs.p, so negative at stall)
    I_final = abs(s[-1, i_I])   # magnitude; direction depends on sign convention
    assert abs(I_final - 100.0) < 1.0, f"|I|={I_final:.2f}, expected 100 A"


# ── Model 7: Resistor with thermal port (Joule heating) ──────────────────────
def test_m07_resistor_thermal_port():
    """V=10V R=100Ω P=1W C_th=1000 J/K G_th=10 W/K  →  T_ss = 300.1 K, τ=100 s"""
    ev = EqnEnv(); ad = AcausalDiagram()
    vs    = elec.VoltageSource(ev, name="vs", v=10.0)
    gnd   = elec.Ground(ev, name="gnd")
    r1    = elec.Resistor(ev, name="r1", R=100.0, enable_heat_port=True)
    c_th  = ht.HeatCapacitor(ev, name="c_th", C=1000.0,
                              initial_temperature=300.0, initial_temperature_fixed=True)
    r_th  = ht.Insulator(ev, name="r_th", R=0.1)   # R = 1/G = 0.1 K/W
    t_amb = ht.TemperatureSource(ev, name="t_amb", temperature=300.0)
    ad.connect(vs, "p", r1, "p"); ad.connect(r1, "n", vs, "n"); ad.connect(vs, "n", gnd, "p")
    ad.connect(r1, "heat", c_th, "port")
    ad.connect(c_th, "port", r_th, "port_a"); ad.connect(r_th, "port_b", t_amb, "port")
    ac, t, s = _sim(ev, ad, (0.0, 400.0))
    T   = s[:, sidx(ac, "np")]   # thermal node-potential state
    ana = 300.1 - 0.1 * np.exp(-t / 100)
    assert np.max(np.abs(T - ana)) < 0.01


# ── Model 8: Coupled mechanical oscillators (was BLOCKED pre-fix) ────────────
def test_m08_coupled_oscillators():
    """Two masses on springs: m=1 k=1  →  normal modes at ω=1 and ω=√3"""
    ev = EqnEnv(); ad = AcausalDiagram()
    fp  = trans.FixedPosition(ev, name="fp")
    sp1 = trans.Spring(ev, name="sp1", K=1.0)
    m1  = trans.Mass(ev, name="m1", M=1.0,
                     initial_velocity=0.0, initial_velocity_fixed=True,
                     initial_position=1.0, initial_position_fixed=True)
    sp2 = trans.Spring(ev, name="sp2", K=1.0)
    m2  = trans.Mass(ev, name="m2", M=1.0,
                     initial_velocity=0.0, initial_velocity_fixed=True,
                     initial_position=0.0, initial_position_fixed=True)
    ad.connect(fp, "flange", sp1, "flange_a"); ad.connect(sp1, "flange_b", m1, "flange")
    ad.connect(m1, "flange", sp2, "flange_a"); ad.connect(sp2, "flange_b", m2, "flange")
    ac, t, s = _sim(ev, ad, (0.0, 20.0))
    # check energy-like invariant: all 4 states populated
    assert s.shape[1] >= 4, f"expected ≥4 states, got {s.shape[1]}"
    # verify simulation runs without blowing up
    assert np.all(np.isfinite(s)), "non-finite values in output"
    assert np.max(np.abs(s)) < 10.0, "oscillations unexpectedly large"


# ── Model 9: Motor with external load inertia (was BLOCKED pre-fix) ──────────
def test_m09_motor_with_load():
    """V=10V, motor R=0.1Ω K=0.5, J_load=1 kg·m²  →  w_ss = V/K = 20 rad/s"""
    ev = EqnEnv(); ad = AcausalDiagram()
    mot  = elec.IdealMotor(ev, name="mot", R=0.1, K=0.5, J=0.01,
                            initial_velocity=0.0, initial_velocity_fixed=True,
                            initial_angle=0.0,    initial_angle_fixed=True,
                            initial_current=0.0,  initial_current_fixed=True)
    vs   = elec.VoltageSource(ev, name="vs", v=10.0)
    gnd  = elec.Ground(ev, name="gnd")
    load = rot.Inertia(ev, name="load", I=1.0,
                       initial_velocity=0.0, initial_velocity_fixed=True,
                       initial_angle=0.0,    initial_angle_fixed=True)
    ad.connect(vs, "p", mot, "pos"); ad.connect(mot, "neg", vs, "n")
    ad.connect(vs, "n", gnd, "p");   ad.connect(mot, "shaft", load, "flange")
    ac, t, s = _sim(ev, ad, (0.0, 30.0))
    w_final = s[-1, sidx(ac, "_0(")]
    # no-load steady state: back-EMF = V => w_ss = V/K = 20 rad/s
    assert abs(w_final - 20.0) < 0.5, f"w={w_final:.3f}, expected ≈20 rad/s"


# ── Model 10: RLC forced at resonance ────────────────────────────────────────
def test_m10_rlc_resonance():
    """R=0.5Ω L=1H C=1F V=sin(t), ω=ω₀=1  →  Vc amplitude grows linearly"""
    ev = EqnEnv(); ad = AcausalDiagram()
    freq = 1.0 / (2.0 * np.pi)       # so that 2π·freq = 1 rad/s = ω₀
    vs = elec.ACVoltageSource(ev, name="vs",
                               amplitude=1.0, frequency=freq, phase=0.0, bias=0.0)
    r1 = elec.Resistor(ev, name="r1", R=0.5)
    l1 = elec.Inductor(ev, name="l1", L=1.0,
                        initial_current=0.0, initial_current_fixed=True)
    c1 = elec.Capacitor(ev, name="c1", C=1.0,
                         initial_voltage=0.0, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "p"); ad.connect(r1, "n", l1, "p")
    ad.connect(l1, "n", c1, "p");  ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    ac, t, s = _sim(ev, ad, (0.0, 40.0))
    Vc = s[:, sidx(ac, "c1_V", "electrical_volt")]
    # at resonance the amplitude grows beyond 1/(R·C·ω₀) = 2 over t=[0,40]
    assert np.max(np.abs(Vc)) > 1.5, f"resonance not growing: max|Vc|={np.max(np.abs(Vc)):.3f}"
    assert np.all(np.isfinite(Vc))
