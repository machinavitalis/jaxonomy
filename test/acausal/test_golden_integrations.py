# SPDX-License-Identifier: MIT
"""
Phase 0 regression wall: minimal end-to-end acausal models vs closed-form solutions.

Each test builds an AcausalDiagram, compiles with AcausalCompiler, runs jaxonomy.simulate,
and compares recorded signals to analytic references. Intended to catch compiler / IR /
solver regressions across electrical, rotational, translational, and thermal domains.

Notes:
- Electrical RC uses V=R=C=1 so the first continuous state matches capacitor voltage
  (same convention as ``test_basic_RC``).
- Thermal golden uses TemperatureSource + Insulator + HeatCapacitor (first-order RC to
  setpoint); an isolated HeatflowSource + single capacitor node currently mis-indexes /
  diverges in simulation and is out of scope for this smoke layer.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.acausal import rotational as rot
from jaxonomy.acausal import thermal as ht
from jaxonomy.acausal import translational as trans
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

# Quieter than domain test files (which use DEBUG) while still allowing failures to log.
import jaxonomy.logging as logging

logging.set_log_level(logging.WARNING)


def test_golden_electrical_rc_lowpass():
    """RC toward V_s: V_C(t) = V_s * (1 - exp(-t/(RC)))."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    # Match test_basic_RC (V=1, R=1, C=1): first continuous state is capacitor voltage.
    v_s = 1.0
    Rv, Cv = 1.0, 1.0
    tau_rc = Rv * Cv

    v1 = elec.VoltageSource(ev, name="v1", V=v_s)
    r1 = elec.Resistor(ev, name="r1", R=Rv)
    c1 = elec.Capacitor(
        ev,
        name="c1",
        C=Cv,
        initial_voltage=0.0,
        initial_voltage_fixed=True,
    )
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")

    ac = AcausalCompiler(ev, ad, verbose=False)
    sys = ac()
    builder = jaxonomy.DiagramBuilder()
    sys = builder.add(sys)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals={"x": sys.output_ports[0]},
    )
    t = results.time
    x = results.outputs["x"]
    v_analytic = v_s * (1.0 - np.exp(-t / tau_rc))
    assert np.allclose(v_analytic, x[:, 0], atol=2e-4, rtol=0.0)


def test_golden_rotational_inertia_damper_constant_torque():
    """J * dw/dt = tau - D*w with w(0)=0  =>  w(t) = w_ss * (1 - exp(-D/J t)), w_ss = tau/D."""
    J, D, tau0 = 0.4, 0.2, 1.0
    w_ss = tau0 / D
    tau_relax = J / D

    ev = EqnEnv()
    ad = AcausalDiagram()
    trq = rot.TorqueSource(ev, name="trq", tau=tau0, enable_flange_b=False)
    jj = rot.Inertia(
        ev,
        name="J",
        I=J,
        initial_angle=0.0,
        initial_angle_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    d1 = rot.Damper(ev, name="d1", D=D)
    ref1 = rot.FixedAngle(ev, name="ref1")
    sns = rot.MotionSensor(ev, name="sns", enable_flange_b=False)
    ad.connect(trq, "flange_a", jj, "flange")
    ad.connect(jj, "flange", d1, "flange_a")
    ad.connect(d1, "flange_b", ref1, "flange")
    ad.connect(jj, "flange", sns, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=False)
    sys = ac()
    builder = jaxonomy.DiagramBuilder()
    sys = builder.add(sys)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    w_idx = sys.outsym_to_portid[sns.get_sym_by_port_name("w_rel")]
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 10.0 * tau_relax),
        recorded_signals={"w": sys.output_ports[w_idx]},
    )
    t = results.time
    w = results.outputs["w"]
    w_analytic = w_ss * (1.0 - np.exp(-D / J * t))
    assert np.allclose(w_analytic, w, atol=2e-3, rtol=0.0)


def test_golden_translational_mass_spring_damper_step_force():
    """M x'' + D x' + K x = F (constant), x(0)=x'(0)=0 (underdamped closed form)."""
    M, D, K, F0 = 1.0, 0.5, 4.0, 1.0
    omega_n = np.sqrt(K / M)
    zeta = D / (2.0 * np.sqrt(M * K))
    assert zeta < 1.0, "test assumes underdamped plant"
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)

    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(
        ev,
        name="m1",
        M=M,
        initial_position=0.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    f1 = trans.ForceSource(ev, name="f1", f=F0, enable_flange_b=True)
    sp1 = trans.Spring(
        ev,
        name="sp1",
        K=K,
        initial_position_A=0.0,
        initial_position_A_fixed=True,
        initial_velocity_A=0.0,
        initial_velocity_A_fixed=True,
        initial_position_B=0.0,
        initial_position_B_fixed=True,
        initial_velocity_B=0.0,
        initial_velocity_B_fixed=True,
    )
    d1 = trans.Damper(ev, name="d1", D=D)
    r1 = trans.FixedPosition(ev, name="r1")
    sns = trans.MotionSensor(
        ev,
        name="sns",
        enable_flange_b=True,
        enable_position_port=True,
    )
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", r1, "flange")
    ad.connect(sp1, "flange_a", d1, "flange_a")
    ad.connect(sp1, "flange_b", d1, "flange_b")
    ad.connect(m1, "flange", f1, "flange_a")
    ad.connect(r1, "flange", f1, "flange_b")
    ad.connect(m1, "flange", sns, "flange_a")
    ad.connect(r1, "flange", sns, "flange_b")

    ac = AcausalCompiler(ev, ad, verbose=False)
    sys = ac()
    builder = jaxonomy.DiagramBuilder()
    sys = builder.add(sys)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    x_idx = sys.outsym_to_portid[sns.get_sym_by_port_name("x_rel")]
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 15.0 / omega_n),
        recorded_signals={"x": sys.output_ports[x_idx]},
    )
    t = results.time
    x = results.outputs["x"]
    x_ss = F0 / K
    envelope = np.exp(-zeta * omega_n * t)
    x_analytic = x_ss * (
        1.0
        - envelope
        * (np.cos(omega_d * t) + (zeta * omega_n / omega_d) * np.sin(omega_d * t))
    )
    assert np.allclose(x_analytic, x, atol=5e-3, rtol=0.0)


def test_golden_thermal_rc_step_to_setpoint():
    """Thermal RC: T_src --R-- C. Same first-order form as electrical RC.

    T_cap(t) = T_src + (T0 - T_src) * exp(-t / (R*C)).
    Uses TemperatureSource + Insulator + HeatCapacitor (validated topology; a lone
    HeatflowSource + capacitor on one node is ill-posed in the current compiler).
    """
    T_src = 300.0
    T0 = 250.0
    R_th, C_heat = 1.0, 1.0
    tau = R_th * C_heat

    ev = EqnEnv()
    ad = AcausalDiagram()
    t1 = ht.TemperatureSource(ev, name="t1", temperature=T_src)
    r1 = ht.Insulator(ev, name="r1", R=R_th)
    c1 = ht.HeatCapacitor(
        ev,
        name="c1",
        C=C_heat,
        initial_temperature=T0,
        initial_temperature_fixed=True,
    )
    ad.connect(t1, "port", r1, "port_a")
    ad.connect(r1, "port_b", c1, "port")

    ac = AcausalCompiler(ev, ad, verbose=False)
    sys = ac()
    builder = jaxonomy.DiagramBuilder()
    sys = builder.add(sys)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals={"x": sys.output_ports[0]},
    )
    t = results.time
    x = results.outputs["x"]
    T_analytic = T_src + (T0 - T_src) * np.exp(-t / tau)
    assert np.allclose(T_analytic, x[:, 0], atol=2e-4, rtol=0.0)
