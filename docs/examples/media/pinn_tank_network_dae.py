# SPDX-License-Identifier: MIT
"""Tank-manifold-pump network as an acausal DAE, with custom components.

Shared plant module for ``pinn_across_stacks_part_2_neural_dae.ipynb`` and
its offline publication script ``pinn_across_stacks_part_2_publication_offline.py``.
The notebook reproduces the two component classes inline as teaching
material (with a drift guard asserting the copies match this file); the
network *builder* is imported from here by both consumers so there is a
single source of truth.

Topology:

    reservoir --(pump, power u_p)--> manifold --(valve A, opening 1-v)--> tank 1
                                     manifold --(valve B, opening v)---> tank 2
    tank 1 --(fixed orifice)--> tank 2
    tank 2 --(fixed orifice)--> reservoir
    manifold --(small bypass)--> reservoir

The manifold is a pure acausal junction: its pressure is an algebraic
unknown determined by the flow-balance constraint the compiler keeps.
Custom components extend the hydraulic domain (potential = pressure Pa,
flow = massflow kg/s): GravityTank (open tank, P = rho*g*h) and
SqrtValve (regularized orifice law M = u*k*sign(dP)*sqrt(|dP|)).
"""
import sympy as sp

from jaxonomy.acausal.component_library.base import SymKind
from jaxonomy.acausal.component_library.hydraulic import (
    HydraulicOnePort,
    HydraulicTwoPort,
    HydraulicProperties,
    PressureSource,
    Pump,
)

G_N = 9.81  # m/s^2


class GravityTank(HydraulicOnePort):
    """Open gravity-drained tank; port at the bottom.

    P_port = rho*g*h (gauge),  A*rho*der(h) = M_net.
    The level h is the differential state.
    """

    def __init__(self, ev, name=None, area=1.0, h_ic=0.5, rho_ref=1000.0):
        self.name = self.__class__.__name__ if name is None else name
        # port pressure IC must be strong (fixed) or the compiler's IC search
        # flattens the levels to zero; rho_ref must match the fluid density.
        super().__init__(
            ev, self.name, P_ic=rho_ref * G_N * h_ic, P_ic_fixed=True
        )
        self.area = self.declare_symbol(
            ev, "area", self.name, kind=SymKind.param, val=area,
            validator=lambda a: a > 0.0,
            invalid_msg=f"GravityTank {self.name} must have area>0",
        )
        self.h = self.declare_symbol(ev, "h", self.name, kind=SymKind.var, ic=h_ic)
        self.dh = self.declare_symbol(
            ev, "dh", self.name, kind=SymKind.var, int_sym=self.h, ic=0.0
        )
        self.h.der_sym = self.dh
        # level sensor output (causal)
        h_out = self.declare_symbol(ev, "h_out", self.name, kind=SymKind.outp)
        from jaxonomy.acausal.component_library.base import EqnKind
        self.declare_equation(sp.Eq(h_out.s, self.h.s), kind=EqnKind.outp)

    def finalize(self, ev):
        fluid = self.ports["port"].fluid
        self.add_eqs(
            [
                # bottom pressure from hydrostatic head (gauge)
                sp.Eq(self.P.s, fluid.density * G_N * self.h.s),
                # mass balance: net inflow raises the level
                sp.Eq(self.dh.s, self.M.s / (fluid.density * self.area.s)),
            ]
        )


class SqrtValve(HydraulicTwoPort):
    """Orifice with square-root pressure-flow law and causal opening input.

    M1 = opening * k * dP / (dP^2 + eps)^(1/4)
       ~ opening * k * sign(dP) * sqrt(|dP|),  smooth at dP = 0.

    opening: causal input in [0, 1] if enable_opening_port else a parameter.
    """

    def __init__(self, ev, name=None, k=1.0, opening=1.0,
                 enable_opening_port=False, eps=1.0):
        # eps regularizes the sqrt at dP=0: slope there is k/eps^(1/4).
        # With dP in Pa (~1e3 in this network) eps=1.0 biases the law by
        # <0.01% while keeping the Newton iterations well-conditioned;
        # eps=1e-4 made the BDF corrector fragile on some compiles.
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        k = self.declare_symbol(
            ev, "k", self.name, kind=SymKind.param, val=k,
            validator=lambda k: k > 0.0,
            invalid_msg=f"SqrtValve {self.name} must have k>0",
        )
        if enable_opening_port:
            opening = self.declare_symbol(ev, "opening", self.name, kind=SymKind.inp)
        else:
            opening = self.declare_symbol(
                ev, "opening", self.name, kind=SymKind.param, val=opening
            )
        eps = sp.Float(eps)
        self.add_eqs(
            [
                sp.Eq(0, self.M1.s + self.M2.s),
                sp.Eq(
                    self.M1.s,
                    opening.s * k.s * self.dP.s / (self.dP.s**2 + eps) ** sp.Rational(1, 4),
                ),
            ]
        )


def build_network(ev, ad, *, h1_ic=0.3, h2_ic=0.1,
                  area1=0.02, area2=0.03,
                  k_in=6e-3, k_12=3e-3, k_out=2e-3,
                  pump_dPmax=3e4, pump_cop=80.0,
                  k_bypass=5e-5):
    """Assemble the tank-manifold network on an AcausalDiagram.

    Returns dict of components. Causal inputs: pump power, valve A opening,
    valve B opening (the caller wires a split fraction v as 1-v / v if
    desired).
    """
    props = HydraulicProperties(ev, fluid_name="water")
    reservoir = PressureSource(ev, name="reservoir", pressure=0.0)
    pump = Pump(ev, name="pump", dPmax=pump_dPmax, CoP=pump_cop)
    valve_a = SqrtValve(ev, name="valve_a", k=k_in, enable_opening_port=True)
    valve_b = SqrtValve(ev, name="valve_b", k=k_in, enable_opening_port=True)
    tank1 = GravityTank(ev, name="tank1", area=area1, h_ic=h1_ic)
    tank2 = GravityTank(ev, name="tank2", area=area2, h_ic=h2_ic)
    orifice_12 = SqrtValve(ev, name="orifice_12", k=k_12)
    orifice_out = SqrtValve(ev, name="orifice_out", k=k_out)
    # small always-open bypass from the manifold back to the reservoir.
    # Physically a relief path; numerically it keeps the manifold node
    # pressure pinned even when the pump idles (pwr -> 0 makes the ideal
    # pump equation lose all dP sensitivity and the Newton pivot goes
    # singular without it).
    bypass = SqrtValve(ev, name="bypass", k=k_bypass)

    # reservoir -> pump -> manifold
    ad.connect(reservoir, "port", pump, "port_a")
    # manifold node: pump outlet feeds both valves (3-way junction)
    ad.connect(pump, "port_b", valve_a, "port_a")
    ad.connect(pump, "port_b", valve_b, "port_a")
    # valves discharge into the tanks
    ad.connect(valve_a, "port_b", tank1, "port")
    ad.connect(valve_b, "port_b", tank2, "port")
    # tank1 drains into tank2; tank2 drains back to the reservoir
    ad.connect(orifice_12, "port_a", tank1, "port")
    ad.connect(orifice_12, "port_b", tank2, "port")
    ad.connect(orifice_out, "port_a", tank2, "port")
    ad.connect(orifice_out, "port_b", reservoir, "port")
    ad.connect(bypass, "port_a", pump, "port_b")
    ad.connect(bypass, "port_b", reservoir, "port")
    # fluid properties tag the whole network
    ad.connect(props, "prop", pump, "port_a")

    return dict(props=props, reservoir=reservoir, pump=pump,
                valve_a=valve_a, valve_b=valve_b,
                tank1=tank1, tank2=tank2,
                orifice_12=orifice_12, orifice_out=orifice_out,
                bypass=bypass)
