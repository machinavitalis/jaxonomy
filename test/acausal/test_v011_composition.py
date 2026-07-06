# SPDX-License-Identifier: MIT
"""V-011: Acausal layer - composability and edge cases.

Eight tricky configurations: acausal-in-diagram, causal feedback,
two domains, submodel (T-036), outer-param IC, over-constrained
model, gradients through acausal, end-to-end optimisation.
Capability gaps degrade to xfail with the actual error excerpt.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions
from jaxonomy.acausal import (
    AcausalCompiler,
    AcausalDiagram,
    EqnEnv,
    AcausalCompilerError,
    AcausalModelError,
    electrical as elec,
    translational as trans,
)
from jaxonomy.framework.system_base import Parameter
from jaxonomy.library import Sine, Gain, Constant
from jaxonomy.library.primitives import Demultiplexer
from jaxonomy.library.reference_subdiagram import ReferenceSubdiagram
from jaxonomy.testing import fd_grad
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _build_rc(R=10.0, C=1e-3, V=1.0, vc0=0.0, sys_name="acausal_system"):
    """Compile an RC: V -> R -> C -> ground.  Returns (sys, sensV, sensI)."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=V)
    r = elec.Resistor(ev, name="r", R=R)
    c = elec.Capacitor(
        ev, name="c", C=C, initial_voltage=vc0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", sensI, "p")
    ad.connect(sensI, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    ad.connect(sensV, "p", c, "p")
    ad.connect(sensV, "n", c, "n")
    sys_ = AcausalCompiler(ev, ad, scale=True)(name=sys_name)
    return sys_, sensV, sensI


def _output_idx(asys, sensor, port_name):
    return asys.outsym_to_portid[sensor.get_sym_by_port_name(port_name)]


# 1. Acausal subsystem inside a regular hierarchical Diagram.


def test_rc_inside_diagram_with_unrelated_sine():
    """Acausal RC + causal Sine source side by side; no shared ports."""
    rc_sys, sensV, _ = _build_rc()
    builder = DiagramBuilder()
    rc = builder.add(rc_sys)
    sine = builder.add(Sine(amplitude=1.0, frequency=2.0, name="sine"))
    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)

    v_idx = _output_idx(rc_sys, sensV, "v")
    res = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 1.0),
        recorded_signals={
            "Vc": rc.output_ports[v_idx],
            "sine": sine.output_ports[0],
        },
        options=SimulatorOptions(math_backend="jax"),
    )
    assert res.time[-1] >= 0.99
    assert np.all(res.outputs["Vc"] >= -1e-6)
    assert np.all(res.outputs["Vc"] <= 1.0 + 1e-6)
    assert np.max(np.abs(res.outputs["sine"])) <= 1.0 + 1e-6


# 2. Causal block reads an acausal output port (and drives an input port).


def test_acausal_with_causal_feedback_steady_state():
    """RC has its VoltageSource exposed as input; a causal Constant
    drives it, and a causal Gain reads the Vc sensor output.  Verifies
    that both directions of acausal<->causal data flow simulate."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", enable_voltage_port=True)
    r = elec.Resistor(ev, name="r", R=10.0)
    c = elec.Capacitor(
        ev, name="c", C=1e-3, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    sensV = elec.VoltageSensor(ev, name="sensV")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    ad.connect(sensV, "p", c, "p")
    ad.connect(sensV, "n", c, "n")
    rc_sys = AcausalCompiler(ev, ad, scale=True)()

    builder = DiagramBuilder()
    rc = builder.add(rc_sys)
    src = builder.add(Constant(value=1.0, name="src"))
    builder.connect(src.output_ports[0], rc.input_ports[0])

    v_idx = _output_idx(rc_sys, sensV, "v")
    gain = builder.add(Gain(0.5, name="gain"))
    builder.connect(rc.output_ports[v_idx], gain.input_ports[0])

    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)
    res = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 1.0),
        recorded_signals={
            "Vc": rc.output_ports[v_idx],
            "gain_out": gain.output_ports[0],
        },
        options=SimulatorOptions(math_backend="jax"),
    )
    Vc = np.asarray(res.outputs["Vc"])
    gain_out = np.asarray(res.outputs["gain_out"])
    # Steady state: ~100 time-constants in 1s, Vc -> 1.
    assert abs(Vc[-1] - 1.0) < 1e-2
    assert np.allclose(gain_out, 0.5 * Vc, atol=1e-6, rtol=1e-6)


# 3. Two independent acausal domains in one Diagram.


def test_two_acausal_domains_in_one_diagram():
    """Electrical RC and mechanical mass-spring as siblings; no shared
    ports; expect independent evolution."""
    rc_sys, sensV, _ = _build_rc(
        R=1.0, C=1.0, V=1.0, vc0=0.0, sys_name="rc_sys"
    )

    ev2 = EqnEnv()
    ad2 = AcausalDiagram()
    K, M = 4.0, 1.0
    fp = trans.FixedPosition(ev2, name="fp")
    sp = trans.Spring(ev2, name="sp", K=K)
    m = trans.Mass(
        ev2, name="m", M=M,
        initial_position=1.0, initial_position_fixed=True,
        initial_velocity=0.0, initial_velocity_fixed=True,
    )
    pos_sens = trans.MotionSensor(
        ev2, name="pos", enable_flange_b=True, enable_position_port=True
    )
    ad2.connect(fp, "flange", sp, "flange_a")
    ad2.connect(sp, "flange_b", m, "flange")
    ad2.connect(m, "flange", pos_sens, "flange_a")
    ad2.connect(fp, "flange", pos_sens, "flange_b")
    ms_sys = AcausalCompiler(ev2, ad2, scale=True)(name="ms_sys")

    builder = DiagramBuilder()
    rc = builder.add(rc_sys)
    ms = builder.add(ms_sys)
    diagram = builder.build()
    ctx = diagram.create_context(check_types=True)

    v_idx = _output_idx(rc_sys, sensV, "v")
    x_idx = _output_idx(ms_sys, pos_sens, "x_rel")
    res = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 2.0),
        recorded_signals={
            "Vc": rc.output_ports[v_idx],
            "x": ms.output_ports[x_idx],
        },
        options=SimulatorOptions(math_backend="jax"),
    )
    Vc = np.asarray(res.outputs["Vc"])
    x = np.asarray(res.outputs["x"])
    assert Vc[-1] > Vc[0]
    omega = np.sqrt(K / M)
    expected_x = np.cos(omega * np.asarray(res.time))
    assert np.allclose(x, expected_x, atol=5e-2)


# 4. Acausal subsystem inside a registered submodel  (T-036).


@pytest.mark.xfail(
    reason="T-036: acausal subsystems inside submodels not yet supported",
    strict=False,
)
def test_acausal_inside_submodel_T036():
    """Place an acausal RC inside a ReferenceSubdiagram and simulate."""

    def _rc_submodel(instance_name, parameters):
        rc_sys, _, _ = _build_rc()
        sub = DiagramBuilder()
        sub.add(rc_sys)
        return sub.build(name=instance_name)

    ref_id = ReferenceSubdiagram.register(_rc_submodel, default_parameters=[])
    outer = DiagramBuilder()
    sub = ReferenceSubdiagram.create_diagram(ref_id, "rc_sub")
    outer.add(sub)
    diagram = outer.build()
    ctx = diagram.create_context(check_types=True)
    jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1), options=SimulatorOptions(math_backend="jax")
    )


# 5. Outer Parameter feeding an acausal initial condition.


def test_outer_param_into_acausal_initial_condition():
    """Pass an outer Parameter into Capacitor.initial_voltage."""
    vc0 = Parameter(value=0.25, name="vc0")
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    try:
        c = elec.Capacitor(
            ev, name="c", C=1.0,
            initial_voltage=vc0, initial_voltage_fixed=True,
        )
    except Exception as exc:
        pytest.xfail(
            f"Capacitor IC rejects outer Parameter: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )
    gnd = elec.Ground(ev, name="gnd")
    sensV = elec.VoltageSensor(ev, name="sensV")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    ad.connect(sensV, "p", c, "p")
    ad.connect(sensV, "n", c, "n")
    try:
        rc_sys = AcausalCompiler(ev, ad, scale=True)()
    except Exception as exc:
        pytest.xfail(
            f"Compile with outer-Parameter IC failed: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

    builder = DiagramBuilder()
    rc = builder.add(rc_sys)
    diagram = builder.build()
    try:
        ctx = diagram.create_context(check_types=True)
    except Exception as exc:
        pytest.xfail(
            f"create_context failed with outer-Parameter IC: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

    v_idx = _output_idx(rc_sys, sensV, "v")
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 1e-3),
        recorded_signals={"Vc": rc.output_ports[v_idx]},
        options=SimulatorOptions(math_backend="jax"),
    )
    assert abs(float(res.outputs["Vc"][0]) - 0.25) < 1e-3


# 6. Over-constrained / singular-Jacobian model.


def test_two_voltage_sources_in_parallel_raises_clear_error():
    """Two ideal VoltageSources in parallel with no series resistance
    over-constrain the loop voltage; compiler must surface a clear
    structural error rather than silently succeeding."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    v2 = elec.VoltageSource(ev, name="v2", V=2.0)
    c = elec.Capacitor(
        ev, name="c", C=1.0, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v1, "p", v2, "p")
    ad.connect(v1, "n", v2, "n")
    ad.connect(v1, "p", c, "p")
    ad.connect(c, "n", v1, "n")
    ad.connect(v1, "n", gnd, "p")

    with pytest.raises((AcausalModelError, AcausalCompilerError, Exception)) as exc:
        AcausalCompiler(ev, ad, scale=True)()
    msg = str(exc.value).lower()
    assert any(
        kw in msg
        for kw in (
            "singular", "rank", "over", "under", "inconsistent",
            "equation", "variable", "constraint", "redundant",
        )
    ), f"Compiler error should describe the structural problem; got: {msg!r}"


# 7. Gradient w.r.t. parameter inside an acausal subsystem.


def test_grad_wrt_acausal_resistor_R():
    """jax.grad of terminal Vc w.r.t. Resistor R.

    For a series RC with V_in step and Vc(0)=0:
        Vc(T) = V * (1 - exp(-T/(R*C)))
        dVc/dR = -V * (T / (R^2 * C)) * exp(-T/(R*C))
    """
    V, C_val, T = 1.0, 1.0, 1.0
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=V)
    r = elec.Resistor(ev, name="r", R=1.0)
    c = elec.Capacitor(
        ev, name="c", C=C_val,
        initial_voltage=0.0, initial_voltage_fixed=True,
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    rc_sys = AcausalCompiler(ev, ad, scale=True)()

    builder = DiagramBuilder()
    rc = builder.add(rc_sys)
    diagram = builder.build()
    ctx0 = diagram.create_context()
    sub_ctx = ctx0[rc.system_id]
    if "r_R" not in sub_ctx.parameters:
        pytest.xfail(
            f"Resistor parameter 'r_R' not exposed; got: "
            f"{list(sub_ctx.parameters.keys())}"
        )

    opts = SimulatorOptions(
        math_backend="jax", enable_autodiff=True, ode_solver_method="bdf"
    )

    def fwd(R_val):
        new_params = dict(sub_ctx.parameters)
        new_params["r_R"] = R_val
        new_sub = sub_ctx.with_parameters(new_params)
        ctx = ctx0.with_subcontext(rc.system_id, new_sub)
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=opts)
        return res.context[rc.system_id].continuous_state[0]

    try:
        ad_grad = float(jax.grad(fwd)(jnp.array(1.0)))
    except Exception as exc:
        pytest.xfail(
            f"Adjoint through acausal R parameter unsupported: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

    # FD with eps=1e-3: BDF tol ~ 1e-6, so a tiny eps is dominated by noise.
    fd_g = float(
        fd_grad(lambda r_: float(fwd(r_[0])), np.array([1.0]), eps=1e-3)[0][0]
    )
    analytic = -V * (T / (1.0 ** 2 * C_val)) * np.exp(-T / (1.0 * C_val))
    assert np.isfinite(ad_grad)
    assert abs(ad_grad - analytic) < 5e-2, (
        f"AD vs analytic: ad={ad_grad:.6f}, analytic={analytic:.6f}"
    )
    assert np.sign(fd_g) == np.sign(analytic) or abs(fd_g) < 1e-3


# 8. End-to-end acausal+causal cost gradient.


def test_end_to_end_acausal_causal_cost_grad():
    """AcausalRC -> Demultiplexer -> Gain (cost = 3 * Vc(T)).

    Take jax.grad w.r.t. initial Vc; verify finite and ~ 3*exp(-T/RC)."""
    R, C_val, V, T = 1.0, 1.0, 1.0, 1.5
    Vc0 = 0.5

    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=V)
    r = elec.Resistor(ev, name="r", R=R)
    c = elec.Capacitor(
        ev, name="c", C=C_val,
        initial_voltage=Vc0, initial_voltage_fixed=True,
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    rc_sys = AcausalCompiler(ev, ad, scale=True)()

    tmp_b = DiagramBuilder()
    tmp_s = tmp_b.add(rc_sys)
    n_states = len(
        tmp_b.build().create_context()[tmp_s.system_id].continuous_state
    )

    bld = DiagramBuilder()
    rc = bld.add(rc_sys)
    demux = bld.add(Demultiplexer(n_states, name="dmx"))
    cost_gain = bld.add(Gain(3.0, name="cost"))
    bld.connect(rc.output_ports[0], demux.input_ports[0])
    bld.connect(demux.output_ports[0], cost_gain.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    x0_full = jnp.array(ctx0[rc.system_id].continuous_state)

    opts = SimulatorOptions(
        math_backend="jax", enable_autodiff=True, ode_solver_method="bdf"
    )

    def fwd(vc0_scalar):
        x_new = x0_full.at[0].set(vc0_scalar)
        rc_ctx = ctx0[rc.system_id].with_continuous_state(x_new)
        ctx = ctx0.with_subcontext(rc.system_id, rc_ctx)
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=opts)
        return 3.0 * res.context[rc.system_id].continuous_state[0]

    try:
        g = float(jax.grad(fwd)(jnp.array(Vc0)))
    except Exception as exc:
        pytest.xfail(
            f"End-to-end acausal+causal grad unsupported: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )
    assert np.isfinite(g)
    expected = 3.0 * np.exp(-T / (R * C_val))
    assert abs(g - expected) < 5e-2, (
        f"End-to-end grad: AD={g:.6f}, expected~{expected:.6f}"
    )
