# SPDX-License-Identifier: MIT

"""StaticPipe: Darcy-Weisbach wall friction + static head, validated end-to-end.

The component was previously broken on arrival (crashed at construction
reading ``port.fluid`` before the network was bound, and referenced
``self.M1``/``self.M2``, which ``FluidTwoPort`` never defines). These tests
author it against physics:

* laminar discharge matches the analytic Hagen-Poiseuille flow,
* the recorded (dP, mdot) trajectory satisfies the Churchill/Darcy law
  (re-implemented independently in numpy),
* a raised port_b (``h_ab > 0``) drives the network to the hydrostatic
  equilibrium ``P_a - P_b = rho*g*h_ab`` through a flow-reversal transient.
"""

import numpy as np
import pytest

from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import fluid as fld
from jaxonomy.acausal import fluid_media as fm

import jaxonomy
from jaxonomy import SimulatorOptions
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

G_N = 9.80665

# Laminar-regime pipe: water, D=1 cm, L=1 m, ~10 Pa drop -> Re ~ 400.
D_PIPE = 0.01
L_PIPE = 1.0
P_AMB = 101325.0
DP0 = 10.0


def _churchill_darcy_dp(mdot, rho, mu, L, D, e=0.0, eps=1e-8):
    """Independent numpy re-implementation of the pipe's pressure-drop law."""
    mdot = np.asarray(mdot, dtype=float)
    Re = 4.0 * (np.abs(mdot) + eps) / (np.pi * D * mu)
    rel = 0.27 * e / D if e > 0 else 0.0
    A = (-2.457 * np.log((7.0 / Re) ** 0.9 + rel)) ** 16
    B = (37530.0 / Re) ** 16
    f = 8.0 * ((8.0 / Re) ** 12 + (A + B) ** -1.5) ** (1.0 / 12.0)
    area = np.pi * D**2 / 4.0
    return f * (L / D) * mdot * np.abs(mdot) / (2.0 * rho * area**2)


def _accumulator_pipe_ambient(h_ab=0.0, dp0=DP0, k_acc=1.0e6):
    """Accumulator(P_amb + dp0) -> StaticPipe -> Boundary_pT(P_amb), water."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    water = fm.WaterLiquidSimple(ev)
    fp = fld.FluidProperties(ev, fluid=water)
    acc = fld.Accumulator(ev, name="acc", P_ic=P_AMB + dp0, area=1.0, k=k_acc)
    pipe = fld.StaticPipe(
        ev, name="pipe", L=L_PIPE, D=D_PIPE, h_ab=h_ab, enable_sensors=True
    )
    amb = fld.Boundary_pT(ev, name="amb", p_ambient=P_AMB)
    ad.connect(acc, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", amb, "port")
    ad.connect(fp, "prop", acc, "port")
    return ev, ad, water, pipe


def _simulate(ev, ad, pipe, t_end):
    acausal_system = AcausalCompiler(ev, ad, scale=True)()
    builder = jaxonomy.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    def port_of(name):
        idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name(name)]
        return acausal_system.output_ports[idx]

    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_end),
        options=SimulatorOptions(
            math_backend="jax",
            ode_solver_method="bdf",
            # Project the caller-supplied context onto the constraint manifold:
            # the head test starts far from the algebraic solution (flow must
            # jump to the turbulent reversal branch at t=0).
            dae_initial_projection=True,
        ),
        recorded_signals={
            "m_flow": port_of("m_flow"),
            "pa": port_of("pa"),
            "pb": port_of("pb"),
        },
    )
    t = np.asarray(results.time)
    return (
        t,
        np.asarray(results.outputs["m_flow"]).ravel(),
        np.asarray(results.outputs["pa"]).ravel(),
        np.asarray(results.outputs["pb"]).ravel(),
    )


def test_static_pipe_constructs_without_network():
    # Regression: previously raised AttributeError at construction (read
    # port.fluid.density before DiagramProcessing bound the network).
    ev = EqnEnv()
    pipe = fld.StaticPipe(ev, name="p")
    assert pipe.name == "p"


def test_static_pipe_requires_viscosity():
    class NoViscosityMedium(fm.WaterLiquidSimple):
        def __init__(self, ev):
            super().__init__(ev)
            del self.viscosity_dyn

    ev, ad, _, _ = _accumulator_pipe_ambient()
    # Rebuild with a viscosity-less medium: finalize must raise a clear error.
    ev2 = EqnEnv()
    ad2 = AcausalDiagram()
    medium = NoViscosityMedium(ev2)
    fp = fld.FluidProperties(ev2, fluid=medium)
    acc = fld.Accumulator(ev2, name="acc", P_ic=P_AMB + DP0)
    pipe = fld.StaticPipe(ev2, name="pipe")
    amb = fld.Boundary_pT(ev2, name="amb", p_ambient=P_AMB)
    ad2.connect(acc, "port", pipe, "port_a")
    ad2.connect(pipe, "port_b", amb, "port")
    ad2.connect(fp, "prop", acc, "port")
    with pytest.raises(ValueError, match="viscosity_dyn"):
        AcausalCompiler(ev2, ad2, scale=True)()


def test_static_pipe_laminar_matches_hagen_poiseuille():
    """Discharge through a laminar pipe: flow matches the analytic law."""
    ev, ad, water, pipe = _accumulator_pipe_ambient(h_ab=0.0)
    t, m_flow, pa, pb = _simulate(ev, ad, pipe, t_end=3.0)

    rho, mu = water.density, water.viscosity_dyn
    dp = pa - pb

    # Trim the numerical initialization transient, keep laminar samples.
    mask = (t > 0.1) & (np.abs(dp) > 1.0)
    assert mask.sum() >= 5
    m, d = m_flow[mask], dp[mask]

    # (1) The compiled trajectory satisfies the Darcy/Churchill law
    #     (independent numpy re-implementation).
    dp_pred = _churchill_darcy_dp(m, rho, mu, L_PIPE, D_PIPE)
    np.testing.assert_allclose(d, dp_pred, rtol=2e-2)

    # (2) Analytic Hagen-Poiseuille in the laminar regime:
    #     mdot = pi*rho*D^4*dP / (128*mu*L)
    m_hp = np.pi * rho * D_PIPE**4 * d / (128.0 * mu * L_PIPE)
    Re = 4.0 * np.abs(m) / (np.pi * D_PIPE * mu)
    assert np.all(Re < 2000.0), f"expected laminar flow, Re up to {Re.max():.0f}"
    np.testing.assert_allclose(m, m_hp, rtol=5e-2)

    # (3) The accumulator drains: flow (and drop) decay monotonically-ish.
    assert abs(m_flow[-1]) < abs(m[0])


def test_static_pipe_static_head_equilibrium():
    """port_b raised by h_ab: flow reverses and the network settles at the
    hydrostatic equilibrium P_a - P_b = rho*g*h_ab."""
    h_ab = 0.2
    ev, ad, water, pipe = _accumulator_pipe_ambient(h_ab=h_ab, k_acc=1.0e7)
    t, m_flow, pa, pb = _simulate(ev, ad, pipe, t_end=8.0)

    rho, mu = water.density, water.viscosity_dyn
    head = rho * G_N * h_ab

    # Initial drop (10 Pa) is far below the head (~1956 Pa): flow must
    # reverse (into the accumulator through port_a).
    assert m_flow[np.searchsorted(t, 0.2)] < 0.0

    # The trajectory obeys friction + head throughout (turbulent at first,
    # relaminarizing as it settles).
    mask = t > 0.1
    dp_pred = _churchill_darcy_dp(m_flow[mask], rho, mu, L_PIPE, D_PIPE) + head
    np.testing.assert_allclose((pa - pb)[mask], dp_pred, rtol=2e-2, atol=2.0)

    # Settles at the hydrostatic equilibrium with vanishing flow.
    assert abs((pa - pb)[-1] - head) < 0.05 * head
    assert abs(m_flow[-1]) < 0.05 * abs(m_flow[np.searchsorted(t, 0.2)])
