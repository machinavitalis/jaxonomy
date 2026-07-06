# SPDX-License-Identifier: MIT

import numpy as np
from matplotlib import pyplot as plt

# acausal imports
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import translational as trans
from jaxonomy.acausal import hydraulic as hd
from jaxonomy.acausal.component_library.base import SymKind

# jaxonomy imports
import jaxonomy
from jaxonomy import library as lib

import jaxonomy.logging as logging
from jaxonomy.testing.markers import skip_if_not_jax

logging.set_log_level(logging.DEBUG)
skip_if_not_jax()


def test_fluid_pressure_to_accumulator(show_plot=False):
    ev = EqnEnv()
    # fp = hd.HydraulicProperties(ev, fluid=hd.Water)
    fp = hd.HydraulicProperties(ev, fluid_name="water")
    ad = AcausalDiagram()
    ps = hd.PressureSource(ev, pressure=10)
    pipe = hd.Pipe(ev)
    acc = hd.Accumulator(ev, area=0.05, initial_pressure_fixed=True)
    sensP = hd.PressureSensor(ev, name="sensP", enable_port_b=False)
    ad.connect(ps, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", acc, "port")
    ad.connect(sensP, "port_a", acc, "port")
    ad.connect(fp, "prop", ps, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    # make jaxonomy diagram
    builder = jaxonomy.DiagramBuilder()
    acausal_system = builder.add(acausal_system)

    # 'compile' jaxonomy diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    p0_idx = acausal_system.outsym_to_portid[sensP.get_sym_by_port_name("P_rel")]
    recorded_signals = {
        "sensP": acausal_system.output_ports[p0_idx],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    sensP = results.outputs["sensP"]

    def rc_filter(t, v=1, r=1, c=1):
        return v * (1 - np.exp(-t / (r * c)))

    accp_sol = rc_filter(t, v=10, r=2.5)

    atol = 0.0002
    rtol = 0

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(t, accp_sol, label="accp_sol", marker="o")
        ax.plot(t, sensP, label="sensP")
        ax.legend()
        plt.show()

    assert np.allclose(accp_sol, sensP, atol=atol, rtol=rtol)


def test_fluid_inline_pump(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    fp = hd.HydraulicProperties(ev, fluid_name="HydraulicFluid")
    ps1 = hd.PressureSource(ev, name="ps1", pressure=0.01)
    pmp = hd.Pump(ev)
    sensP = hd.PressureSensor(ev, name="sensP", enable_port_b=False)
    sensMF = hd.MassflowSensor(ev, name="sensMF")
    acc = hd.Accumulator(ev, name="acc", initial_pressure_fixed=True)
    ad.connect(ps1, "port", pmp, "port_a")
    ad.connect(pmp, "port_b", sensMF, "port_a")
    ad.connect(sensMF, "port_b", acc, "port")
    ad.connect(fp, "prop", acc, "port")
    ad.connect(sensP, "port_a", acc, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    # make jaxonomy diagram
    builder = jaxonomy.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    pmp_pwr = builder.add(lib.Constant(value=1e3))
    builder.connect(pmp_pwr.output_ports[0], acausal_system.input_ports[0])

    # 'compile' jaxonomy diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    p0_idx = acausal_system.outsym_to_portid[sensP.get_sym_by_port_name("P_rel")]
    mf0_idx = acausal_system.outsym_to_portid[sensMF.get_sym_by_port_name("m_flow")]
    recorded_signals = {
        "sensP": acausal_system.output_ports[p0_idx],
        "sensMF": acausal_system.output_ports[mf0_idx],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    sensP = results.outputs["sensP"]
    sensMF = results.outputs["sensMF"]

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(t, sensP, label="sensP")
        ax2.plot(t, sensMF, label="sensMF")

        ax1.legend()
        ax2.legend()
        ax1.grid()
        ax2.grid()
        plt.show()

    # just ensure the final value
    print(sensP[-1])
    assert np.allclose(3.70, sensP[-1], rtol=0.05, atol=0.0)
    print(sensMF[-1])
    assert np.allclose(213, sensMF[-1], rtol=0.05, atol=0.0)


def test_hyd_act_and_spring(show_plot=False):
    ev = EqnEnv()
    ad = AcausalDiagram()
    fp = hd.HydraulicProperties(ev, fluid_name="HydraulicFluid")
    ps1 = hd.PressureSource(ev, name="ps1", pressure=1.0)
    ps2 = hd.PressureSource(ev, name="ps2", pressure=0.1)
    act = hd.HydraulicActuatorLinear(ev, name="act")
    ref1 = trans.FixedPosition(ev, name="ref1")
    mass = trans.Mass(
        ev,
        name="mass",
        initial_position=0.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    sprg = trans.Spring(ev, name="sprg")
    ref2 = trans.FixedPosition(ev, name="ref2")
    tspd = trans.MotionSensor(ev, name="tspd", enable_flange_b=False)
    tfrc = trans.ForceSensor(ev, name="tfrc")
    mf = hd.MassflowSensor(ev, name="mf")
    ad.connect(ps1, "port", fp, "prop")
    ad.connect(ps1, "port", act, "port_a")
    ad.connect(ps2, "port", mf, "port_a")
    ad.connect(mf, "port_b", act, "port_b")
    ad.connect(act, "flange_a", ref1, "flange")
    ad.connect(act, "flange_b", tfrc, "flange_a")
    ad.connect(tfrc, "flange_b", sprg, "flange_a")
    ad.connect(sprg, "flange_a", mass, "flange")
    ad.connect(sprg, "flange_b", ref2, "flange")
    ad.connect(tspd, "flange_a", mass, "flange")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    # make jaxonomy diagram
    builder = jaxonomy.DiagramBuilder()
    acausal_system = builder.add(acausal_system)

    # 'compile' jaxonomy diagram
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # run the simulation
    tspd_idx = acausal_system.outsym_to_portid[tspd.get_sym_by_port_name("v_rel")]
    tfrc_idx = acausal_system.outsym_to_portid[tfrc.get_sym_by_port_name("f")]
    mf_idx = acausal_system.outsym_to_portid[mf.get_sym_by_port_name("m_flow")]
    recorded_signals = {
        "tspd": acausal_system.output_ports[tspd_idx],
        "tfrc": acausal_system.output_ports[tfrc_idx],
        "mf": acausal_system.output_ports[mf_idx],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )
    t = results.time
    tspd = results.outputs["tspd"]
    tfrc = results.outputs["tfrc"]
    mf = results.outputs["mf"]

    tspd_sol = np.sin(t) * 0.9

    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3))
        ax1.plot(t, tspd, label="tspd", linewidth=3)
        ax1.plot(t, tspd_sol, label="tspd_sol")
        ax1.legend()
        ax1.grid()

        ax2.plot(t, tfrc, label="tfrc")
        ax2.legend()
        ax2.grid()

        ax3.plot(t, mf, label="mf")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert np.allclose(tspd, tspd_sol, atol=0.0, rtol=0.005)


def test_hydraulic_massflow_source_constant_flow(show_plot=False):
    ev = EqnEnv()
    src_param = hd.MassflowSource(
        ev,
        name="src_param",
        M=0.5,
        enable_massflow_port=False,
    )
    src_inp = hd.MassflowSource(
        ev,
        name="src_inp",
        enable_massflow_port=True,
    )

    assert src_param.port_idx_to_name == {-1: "port"}
    assert src_inp.port_idx_to_name == {-1: "port"}

    m_param = [
        s
        for s in src_param.syms
        if s.sym_name == "M" and "_port_" not in str(s.s) and "_port(" not in str(s.s)
    ][0]
    m_inp = [
        s
        for s in src_inp.syms
        if s.sym_name == "M" and "_port_" not in str(s.s) and "_port(" not in str(s.s)
    ][0]
    assert m_param.kind == SymKind.param
    assert m_inp.kind == SymKind.inp

    eqs_param = [str(eq.e) for eq in src_param.eqs]
    eqs_inp = [str(eq.e) for eq in src_inp.eqs]
    assert any("src_param_port_M" in e and "src_param_M" in e for e in eqs_param)
    assert any("src_inp_port_M" in e and "src_inp_M" in e for e in eqs_inp)


if __name__ == "__main__":
    show_plot = True
    # test_fluid_pressure_to_accumulator(show_plot=show_plot)
    # test_fluid_inline_pump(show_plot=show_plot)
    test_hyd_act_and_spring(show_plot=show_plot)
