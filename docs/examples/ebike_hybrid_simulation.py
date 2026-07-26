"""Smart cargo e-bike — high-fidelity multi-domain acausal model.

A single, calibrated e-bike model built on Jaxonomy's acausal engine, coupling
four physical domains through acausal ports:

    electrical  (2-RC battery ECM, dq PMSM, inverter losses)
      <-> rotational  (crank, motor, chain compliance, drivetrain gearing)
      <-> translational / planar vehicle  (3-DOF body, Pacejka tyres, aero,
          rolling resistance, grade)
      <-> thermal  (battery + motor lumped nodes, speed-dependent convection)

plus causal control blocks (rider biomechanics, assist policy, field-oriented
current control).

The distinguishing feature versus a "cute demo" is that the model is
*instrumented for verification*: every power flow (human, battery-terminal,
aero, rolling, grade, drivetrain damping, chain damping, motor heat) is
integrated online into a dedicated energy accumulator, so the closing energy
balance can be checked to within a few percent against the change in stored
energy (kinetic + rotational + gravitational + chain-spring). See
:func:`energy_audit` and :func:`validate`.

Run directly to execute the reference drive cycle and print the audit:

    python docs/examples/ebike_hybrid_simulation.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

import jaxonomy
from jaxonomy.acausal.component_library.base import SymKind, EqnKind
from jaxonomy.acausal.component_library.component_base import ComponentBase
from jaxonomy.acausal import (
    AcausalCompiler,
    AcausalDiagram,
    EqnEnv,
    electrical as elec,
    rotational as rot,
    thermal as therm,
    battery as bat,
)
from jaxonomy.framework import LeafSystem
from jaxonomy.library import Integrator
from jaxonomy.simulation import SimulatorOptions


# ---------------------------------------------------------------------------
# DAE-safe smoothing helpers (avoid non-differentiable branches in the symbolic
# graph; the added 1e-4 keeps derivatives bounded at the kinks).
# ---------------------------------------------------------------------------
def safe_abs(x):
    return sp.sqrt(x**2 + 1e-4)


def safe_max(x, y):
    return 0.5 * (x + y + sp.sqrt((x - y) ** 2 + 1e-4))


def safe_min(x, y):
    return 0.5 * (x + y - sp.sqrt((x - y) ** 2 + 1e-4))


# ===========================================================================
# 1. Calibration — realistic Class-1 cargo e-bike parameters
# ===========================================================================
@dataclass
class EbikeConfig:
    """Physical + control parameters for a mid-drive cargo e-bike.

    Defaults are calibrated to a Class-1 (EU 25 km/h, 250 W nominal) longtail
    cargo bike carrying a rider plus cargo. Every value is a real, nameable
    quantity rather than a fitting knob.
    """

    # Masses [kg]
    m_bike_rider: float = 120.0   # frame + drivetrain + rider
    m_cargo: float = 60.0         # payload on the rack

    # Wheel / chassis
    r_wheel: float = 0.29         # loaded rolling radius [m] (26"/650b cargo)
    Crr: float = 0.008            # rolling resistance coefficient (asphalt)
    CdA: float = 0.80             # drag area Cd*A [m^2] (upright rider + cargo)
    rho_air: float = 1.20         # air density [kg/m^3]

    # Drivetrain gearing (see make_ebike_diagram for the kinematic chain)
    motor_reduction: float = 5.0  # motor:intermediate speed ratio
    wheel_reduction: float = 0.5  # intermediate:wheel speed ratio

    # Battery pack: 13S, ~15 Ah -> ~700 Wh, ~48 V class
    n_series: int = 13
    cell_capacity_Ah: float = 15.0
    initial_soc: float = 0.90

    # Assist policy
    K_assist: float = 2.0
    v_cutoff_kmh: float = 25.0    # EU legal assist cutoff
    v_fade_kmh: float = 23.0      # begin fading assist below the cutoff
    max_assist_torque: float = 12.0  # motor-shaft torque cap [Nm]

    # Reference drive cycle
    tf: float = 60.0
    grade_hold: float | None = None  # if set, hold this constant road grade
                                     # (thermal stress test) instead of the
                                     # default hill/descent profile

    @property
    def m_total(self) -> float:
        return self.m_bike_rider + self.m_cargo


# ===========================================================================
# 2. High-fidelity custom acausal components
#    Each exports power signals (kind=outp) so energy flows can be audited.
# ===========================================================================
class TorsionalSpringDamper(ComponentBase):
    """Compliant chain/belt as a torsional spring + damper.

    With ``one_way=True`` it becomes a freewheel (rear-hub ratchet): torque is
    transmitted only while the driving flange leads (torque > 0); when the wheel
    overruns the drivetrain the coupling releases, so the rider/motor can coast
    without being back-driven.

    Exports the net mechanical power into the element (stored + dissipated),
    which closes the energy audit in both the engaged and released states, and
    the deflection for plotting.
    """

    def __init__(self, ev, name=None, K=2000.0, D=50.0, one_way=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        trq_a, ang_a, w_a, alpha_a = self.declare_rotational_port(
            ev, "flange_a", ang_ic=0.0, ang_ic_fixed=True
        )
        trq_b, ang_b, w_b, alpha_b = self.declare_rotational_port(ev, "flange_b")

        K_sym = self.declare_symbol(ev, "K", self.name, kind=SymKind.param, val=K)
        D_sym = self.declare_symbol(ev, "D", self.name, kind=SymKind.param, val=D)

        defl = ang_a.s - ang_b.s
        dw = w_a.s - w_b.s

        T_full = K_sym.s * defl + D_sym.s * dw
        if one_way:
            # Soft ratchet: ~full coupling when driving (T>0), ~2% back-coupling
            # when the wheel overruns.  A soft asymmetry (rather than a hard
            # safe_max clamp to zero) keeps the drivetrain/wheel torque path
            # well-posed for the stiff DAE while still letting the rider coast.
            engaged = 0.5 * (1.0 + T_full / safe_abs(T_full))
            torque_expr = T_full * (0.02 + 0.98 * engaged)
        else:
            torque_expr = T_full

        # Net power into the coupling = trq_a * (w_a - w_b) = d(spring PE)/dt +
        # damper dissipation; correct whether engaged or released.
        p_chain = self.declare_symbol(ev, "p_chain", self.name, kind=SymKind.outp)
        defl_out = self.declare_symbol(ev, "defl", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(p_chain.s, trq_a.s * dw), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(defl_out.s, defl), kind=EqnKind.outp)

        self.add_eqs(
            [
                sp.Eq(trq_a.s, torque_expr),
                sp.Eq(trq_b.s, -trq_a.s),
            ]
        )
        self.port_idx_to_name = {-1: "flange_a", 1: "flange_b"}


class SpeedDependentCooling(ComponentBase):
    """Convective thermal link whose conductance rises with vehicle speed."""

    def __init__(self, ev, name=None, h_static=1.0, k_wind=0.5, A_case=0.1):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        T_a, Q_a = self.declare_thermal_port(ev, "port_a")
        T_b, Q_b = self.declare_thermal_port(ev, "port_b")
        v_bike = self.declare_symbol(ev, "speed", self.name, kind=SymKind.inp)

        h_static_sym = self.declare_symbol(ev, "h_static", self.name, kind=SymKind.param, val=h_static)
        k_wind_sym = self.declare_symbol(ev, "k_wind", self.name, kind=SymKind.param, val=k_wind)
        A_case_sym = self.declare_symbol(ev, "A_case", self.name, kind=SymKind.param, val=A_case)

        h_conv = (h_static_sym.s + k_wind_sym.s * safe_abs(v_bike.s)) * A_case_sym.s

        self.add_eqs(
            [
                sp.Eq(Q_a.s + Q_b.s, 0),
                sp.Eq(Q_a.s, h_conv * (T_a.s - T_b.s)),
            ]
        )
        self.port_idx_to_name = {-1: "port_a", 1: "port_b"}


class SurrogateCooling(ComponentBase):
    """Convective thermal link whose conductance is supplied by an external
    signal (e.g. a reduced-order surrogate of a cooling map), rather than an
    inline analytic correlation. ``h_cond`` [W/K] is an input port."""

    def __init__(self, ev, name=None):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        T_a, Q_a = self.declare_thermal_port(ev, "port_a")
        T_b, Q_b = self.declare_thermal_port(ev, "port_b")
        h_cond = self.declare_symbol(ev, "h_cond", self.name, kind=SymKind.inp)
        self.add_eqs(
            [
                sp.Eq(Q_a.s + Q_b.s, 0),
                sp.Eq(Q_a.s, safe_max(1e-3, h_cond.s) * (T_a.s - T_b.s)),
            ]
        )
        self.port_idx_to_name = {-1: "port_a", 1: "port_b"}


class HighFidelityBatteryCellECM(elec.ElecTwoPin):
    """2-RC equivalent-circuit battery (lumped n_series string) with SOC/SOH,
    thermal core node, and temperature/SOC-dependent parameters.

    Passive sign convention: ``Ip`` (current into the positive pin) is negative
    on discharge, so ``dSOC/dt = Ip/(n_series*cap*3600)`` decreases SOC when the
    pack drives the motor. Exports terminal power (V*Ip), chemical power
    (n_series*OCV*Ip) and internal heat generation for the energy audit.
    """

    def __init__(self, ev, name=None, capacity_Ah=15.0, T_ref=298.15,
                 R00=0.015, R10=0.01, R20=0.012, C10=2000.0, C20=10000.0,
                 alpha_R=-0.005, beta_R=0.5, alpha_OCV_T=-0.0005,
                 C_core=1500.0, R_core_case=0.5, k_aging=1e-4, alpha_aging=0.05,
                 initial_soc=0.9, initial_soc_fixed=True, n_series=13):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name, V_ic=3.9 * n_series, I_ic=0.0)

        SOC = self.declare_symbol(ev, "SOC", self.name, kind=SymKind.var, ic=initial_soc, ic_fixed=initial_soc_fixed)
        dSOC = self.declare_symbol(ev, "dSOC", self.name, kind=SymKind.var, int_sym=SOC, ic=0.0, ic_fixed=True)
        SOC.der_sym = dSOC

        V_RC1 = self.declare_symbol(ev, "V_RC1", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dV_RC1 = self.declare_symbol(ev, "dV_RC1", self.name, kind=SymKind.var, int_sym=V_RC1, ic=0.0)
        V_RC1.der_sym = dV_RC1

        V_RC2 = self.declare_symbol(ev, "V_RC2", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dV_RC2 = self.declare_symbol(ev, "dV_RC2", self.name, kind=SymKind.var, int_sym=V_RC2, ic=0.0)
        V_RC2.der_sym = dV_RC2

        SOH = self.declare_symbol(ev, "SOH", self.name, kind=SymKind.var, ic=1.0, ic_fixed=True)
        dSOH = self.declare_symbol(ev, "dSOH", self.name, kind=SymKind.var, int_sym=SOH, ic=0.0, ic_fixed=True)
        SOH.der_sym = dSOH

        T_core = self.declare_symbol(ev, "T_core", self.name, kind=SymKind.var, ic=298.15, ic_fixed=True)
        dT_core = self.declare_symbol(ev, "dT_core", self.name, kind=SymKind.var, int_sym=T_core, ic=0.0, ic_fixed=True)
        T_core.der_sym = dT_core

        cap = self.declare_symbol(ev, "capacity_Ah", self.name, kind=SymKind.param, val=capacity_Ah)

        T_cell, Q_cell = self.declare_thermal_port(ev, "heat")
        self.port_idx_to_name[2] = "heat"

        soc_out = self.declare_symbol(ev, "soc", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(soc_out.s, SOC.s), kind=EqnKind.outp)

        v_out = self.declare_symbol(ev, "v_out", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(v_out.s, self.V.s), kind=EqnKind.outp)

        soh_out = self.declare_symbol(ev, "soh_out", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(soh_out.s, SOH.s), kind=EqnKind.outp)

        soc_safe = safe_max(0.01, safe_min(0.99, SOC.s))

        OCV = (3.3 + 1.2 * soc_safe - 0.5 * soc_safe**2 + 0.15 * soc_safe**3
               + alpha_OCV_T * (T_core.s - T_ref))

        R0 = (R00 * (1.0 + alpha_R * (T_core.s - T_ref)) * (1.0 + beta_R * (1.0 - soc_safe))) / SOH.s
        R1 = R10 * (1.0 + alpha_R * (T_core.s - T_ref)) / soc_safe
        R2 = R20 * (1.0 + alpha_R * (T_core.s - T_ref)) / (1.0 - soc_safe + 1e-3)
        C1 = C10 * (1.0 - alpha_R * (T_core.s - T_ref)) * soc_safe
        C2 = C20 * (1.0 - alpha_R * (T_core.s - T_ref)) * (1.0 - soc_safe + 1e-3)

        Q_gen = self.Ip.s**2 * R0 + V_RC1.s**2 / R1 + V_RC2.s**2 / R2 - self.Ip.s * T_core.s * alpha_OCV_T
        cap_current = cap.s * SOH.s

        I_mag = sp.sqrt(self.Ip.s**2 + 1e-4)

        # Energy-audit exports.  Terminal power P=V*Ip is negative on discharge
        # (Ip<0); we export -V*Ip so that "power delivered by the pack" is
        # positive while discharging.
        p_term = self.declare_symbol(ev, "p_term", self.name, kind=SymKind.outp)
        p_chem = self.declare_symbol(ev, "p_chem", self.name, kind=SymKind.outp)
        q_heat = self.declare_symbol(ev, "q_heat", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(p_term.s, -self.V.s * self.Ip.s), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(p_chem.s, -n_series * OCV * self.Ip.s), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(q_heat.s, Q_gen), kind=EqnKind.outp)

        self.add_eqs(
            [
                # Series cells share the same current, so SOC drains at Ip/(cap*3600)
                # with NO 1/n_series factor (that would be correct only for a
                # parallel string).
                sp.Eq(dSOC.s, self.Ip.s / (cap_current * 3600.0)),
                sp.Eq(dV_RC1.s, self.Ip.s / C1 - V_RC1.s / (R1 * C1)),
                sp.Eq(dV_RC2.s, self.Ip.s / C2 - V_RC2.s / (R2 * C2)),
                sp.Eq(dSOH.s, -I_mag / (cap.s * 3600.0) * k_aging * (1.0 + alpha_aging * (T_core.s - T_ref))),
                sp.Eq(self.V.s, n_series * (OCV + R0 * self.Ip.s + V_RC1.s + V_RC2.s)),
                sp.Eq(C_core * dT_core.s, Q_gen - (T_core.s - T_cell.s) / R_core_case),
                sp.Eq(-Q_cell.s, n_series * (T_core.s - T_cell.s) / R_core_case),
            ]
        )


class HighFidelityPMSM(elec.ElecTwoPin):
    """dq-axis PMSM with inductance saturation, Steinmetz core loss, inverter
    conduction/switching loss, dual-node (stator/rotor) thermal model, and
    optional thermal demagnetisation.

    The DC-bus current is derived from a *signed* power balance
    ``Ip = (P_mech + P_heat)/V`` so that regeneration (P_mech<0) correctly
    charges the pack. Exports mechanical, electrical and heat power.
    """

    def __init__(self, ev, name=None, R_s=0.05, L_d0=0.0005, L_q0=0.0008, psi_m=0.10, P=4.0, J=0.01, B=0.001, C_core=1e-6,
                 ksat_d=1.5e-5, ksat_q=2.5e-5, kcross=1.0e-5,
                 C_stator=600.0, C_rotor=200.0, R_stator_case=1.0, R_airgap=5.0, w_ic=0.0,
                 R_ds_on=0.005, f_sw=10000.0, k_sw=3e-7, enable_inverter_losses=True,
                 enable_demagnetization=True, alpha_mag=0.001):
        # Torque constant Kt = 1.5*P*psi_m = 0.6 Nm/A (must match the assist
        # policy's torque_constant). psi_m=0.10 and the reduced switching-loss
        # coefficient k_sw give a realistic ~90% peak motor+inverter efficiency.
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name, V_ic=48.0, I_ic=0.0)

        trq, ang, w, alpha = self.declare_rotational_port(ev, "shaft", w_ic=w_ic, ang_ic=0.0)
        self.port_idx_to_name[1] = "shaft"

        vd_ctrl = self.declare_symbol(ev, "vd_ctrl", self.name, kind=SymKind.inp)
        vq_ctrl = self.declare_symbol(ev, "vq_ctrl", self.name, kind=SymKind.inp)

        id_sym = self.declare_symbol(ev, "id_sym", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        did_sym = self.declare_symbol(ev, "did_sym", self.name, kind=SymKind.var, int_sym=id_sym, ic=0.0)
        id_sym.der_sym = did_sym

        iq_sym = self.declare_symbol(ev, "iq_sym", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        diq_sym = self.declare_symbol(ev, "diq_sym", self.name, kind=SymKind.var, int_sym=iq_sym, ic=0.0)
        iq_sym.der_sym = diq_sym

        T_stator = self.declare_symbol(ev, "T_stator", self.name, kind=SymKind.var, ic=298.15, ic_fixed=True)
        dT_stator = self.declare_symbol(ev, "dT_stator", self.name, kind=SymKind.var, int_sym=T_stator, ic=0.0)
        T_stator.der_sym = dT_stator

        T_rotor = self.declare_symbol(ev, "T_rotor", self.name, kind=SymKind.var, ic=298.15, ic_fixed=True)
        dT_rotor = self.declare_symbol(ev, "dT_rotor", self.name, kind=SymKind.var, int_sym=T_rotor, ic=0.0)
        T_rotor.der_sym = dT_rotor

        id_out = self.declare_symbol(ev, "id_out", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(id_out.s, id_sym.s), kind=EqnKind.outp)

        iq_out = self.declare_symbol(ev, "iq_out", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(iq_out.s, iq_sym.s), kind=EqnKind.outp)

        T_stator_out = self.declare_symbol(ev, "T_stator_out", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(T_stator_out.s, T_stator.s), kind=EqnKind.outp)

        T_mot, Q_mot = self.declare_thermal_port(ev, "heat")
        self.port_idx_to_name[2] = "heat"

        Rs_sym = self.declare_symbol(ev, "R_s", self.name, kind=SymKind.param, val=R_s)
        psi_sym = self.declare_symbol(ev, "psi_m", self.name, kind=SymKind.param, val=psi_m)
        P_sym = self.declare_symbol(ev, "P", self.name, kind=SymKind.param, val=P)
        J_sym = self.declare_symbol(ev, "J", self.name, kind=SymKind.param, val=J)
        B_sym = self.declare_symbol(ev, "B", self.name, kind=SymKind.param, val=B)
        C_core_sym = self.declare_symbol(ev, "C_core", self.name, kind=SymKind.param, val=C_core)

        if enable_demagnetization:
            psi_eff = psi_sym.s * safe_max(0.2, 1.0 - alpha_mag * (T_rotor.s - 298.15))
        else:
            psi_eff = psi_sym.s

        Ld = safe_max(0.15 * L_d0, L_d0 - ksat_d * safe_abs(id_sym.s) - kcross * safe_abs(iq_sym.s))
        Lq = safe_max(0.15 * L_q0, L_q0 - ksat_q * safe_abs(iq_sym.s) - kcross * safe_abs(id_sym.s))

        w_e = P_sym.s * w.s
        tau_em = 1.5 * P_sym.s * (psi_eff * iq_sym.s + (Ld - Lq) * id_sym.s * iq_sym.s)

        Q_winding = 1.5 * Rs_sym.s * (id_sym.s**2 + iq_sym.s**2)

        # Steinmetz core loss: hysteresis (k_h*w_e) + eddy currents (C_core*w_e^2)
        k_h = 1e-4
        Q_core = k_h * safe_abs(w_e) + C_core_sym.s * w_e**2

        if enable_inverter_losses:
            I_rms = sp.sqrt(1.5 * (id_sym.s**2 + iq_sym.s**2) + 1e-4)
            P_inverter = 3.0 * I_rms**2 * R_ds_on + 3.0 * self.V.s * I_rms * f_sw * k_sw
        else:
            P_inverter = 0.0

        P_mech = tau_em * w.s              # signed: <0 during regeneration
        P_heat = Q_winding + Q_core + P_inverter
        V_safe = safe_max(1.0, self.V.s)

        # Energy-audit exports
        p_mech_out = self.declare_symbol(ev, "p_mech", self.name, kind=SymKind.outp)
        p_heat_out = self.declare_symbol(ev, "p_heat", self.name, kind=SymKind.outp)
        p_elec_out = self.declare_symbol(ev, "p_elec", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(p_mech_out.s, P_mech), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(p_heat_out.s, P_heat), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(p_elec_out.s, self.V.s * self.Ip.s), kind=EqnKind.outp)

        self.add_eqs(
            [
                sp.Eq(Ld * did_sym.s, vd_ctrl.s - Rs_sym.s * id_sym.s + w_e * Lq * iq_sym.s),
                sp.Eq(Lq * diq_sym.s, vq_ctrl.s - Rs_sym.s * iq_sym.s - w_e * Ld * id_sym.s - w_e * psi_eff),
                sp.Eq(0, trq.s + tau_em - B_sym.s * w.s - J_sym.s * alpha.s),
                # Signed DC-bus power balance (electrical in = mechanical + heat)
                sp.Eq(self.Ip.s, (P_mech + P_heat) / V_safe),
                sp.Eq(C_stator * dT_stator.s, Q_winding + Q_core + P_inverter - (T_stator.s - T_mot.s) / R_stator_case - (T_stator.s - T_rotor.s) / R_airgap),
                sp.Eq(C_rotor * dT_rotor.s, -(T_rotor.s - T_stator.s) / R_airgap),
                sp.Eq(-Q_mot.s, (T_stator.s - T_mot.s) / R_stator_case),
            ]
        )


class PlanarVehicleDynamics(ComponentBase):
    """3-DOF planar vehicle: longitudinal + lateral + yaw, with a Pacejka Magic
    Formula tyre, aerodynamic drag, rolling resistance, road grade, and optional
    pitch load transfer.

    Exports aero, rolling and grade (climb) power for the energy audit, plus
    kinematic outputs. Kinetic energy is reconstructed from the exported state
    outputs (u, v_lat, r).
    """

    def __init__(self, ev, name=None, r_wheel=0.29, J_wheel=0.15, B_wheel=0.01, M_total=180.0,
                 a=1.0, b=0.6, h_cg=0.8, I_z=120.0,
                 B_pacejka=10.0, C_pacejka=1.9, D_pacejka=1.0, E_pacejka=0.97,
                 B_pacejka_y=8.0, C_pacejka_y=1.8, D_pacejka_y=0.9, E_pacejka_y=0.95,
                 rho=1.2, CdA=0.8, Cs0=1.2, A_side=1.2, Crr=0.008,
                 v_headwind=0.0, v_crosswind=0.0, enable_dynamic_weight_transfer=True):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()

        trq_rear, ang_rear, w_rear, alpha_rear = self.declare_rotational_port(
            ev, "shaft_rear", w_ic=0.30303, w_ic_fixed=True, ang_ic=0.0, ang_ic_fixed=True
        )
        self.port_idx_to_name = {1: "shaft_rear"}

        delta = self.declare_symbol(ev, "steer_angle", self.name, kind=SymKind.inp)
        slope = self.declare_symbol(ev, "slope", self.name, kind=SymKind.inp)

        u = self.declare_symbol(ev, "u", self.name, kind=SymKind.var, ic=0.1, ic_fixed=True)
        du = self.declare_symbol(ev, "du", self.name, kind=SymKind.var, int_sym=u, ic=0.0)
        u.der_sym = du

        v_lat = self.declare_symbol(ev, "v_lat", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dv_lat = self.declare_symbol(ev, "dv_lat", self.name, kind=SymKind.var, int_sym=v_lat, ic=0.0)
        v_lat.der_sym = dv_lat

        r = self.declare_symbol(ev, "r", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dr = self.declare_symbol(ev, "dr", self.name, kind=SymKind.var, int_sym=r, ic=0.0)
        r.der_sym = dr

        psi = self.declare_symbol(ev, "psi", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dpsi = self.declare_symbol(ev, "dpsi", self.name, kind=SymKind.var, int_sym=psi, ic=0.0)
        psi.der_sym = dpsi

        x = self.declare_symbol(ev, "x", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dx = self.declare_symbol(ev, "dx", self.name, kind=SymKind.var, int_sym=x, ic=0.0)
        x.der_sym = dx

        y = self.declare_symbol(ev, "y", self.name, kind=SymKind.var, ic=0.0, ic_fixed=True)
        dy = self.declare_symbol(ev, "dy", self.name, kind=SymKind.var, int_sym=y, ic=0.0)
        y.der_sym = dy

        for sym_name, expr in [("u_out", u.s), ("v_out", v_lat.s), ("yaw_rate_out", r.s),
                               ("yaw_angle_out", psi.s), ("pos_x_out", x.s), ("pos_y_out", y.s)]:
            o = self.declare_symbol(ev, sym_name, self.name, kind=SymKind.outp)
            self.declare_equation(sp.Eq(o.s, expr), kind=EqnKind.outp)

        M_sym = self.declare_symbol(ev, "M_total", self.name, kind=SymKind.param, val=M_total)
        Iz_sym = self.declare_symbol(ev, "I_z", self.name, kind=SymKind.param, val=I_z)

        L_wheelbase = a + b

        if enable_dynamic_weight_transfer:
            delta_Fz = (h_cg / L_wheelbase) * M_sym.s * du.s
            F_z_front = (b / L_wheelbase) * M_sym.s * 9.81 - delta_Fz
            F_z_rear = (a / L_wheelbase) * M_sym.s * 9.81 + delta_Fz
        else:
            F_z_front = (b / L_wheelbase) * M_sym.s * 9.81
            F_z_rear = (a / L_wheelbase) * M_sym.s * 9.81

        Fzf_safe = safe_max(10.0, F_z_front)
        Fzr_safe = safe_max(10.0, F_z_rear)

        u_abs_max = safe_max(safe_abs(w_rear.s * r_wheel), safe_abs(u.s))
        slip_x = (w_rear.s * r_wheel - u.s) / (u_abs_max + 0.1)

        u_denom = safe_max(0.5, safe_abs(u.s))
        alpha_f = delta.s - sp.atan((v_lat.s + a * r.s) / u_denom)
        alpha_r = -sp.atan((v_lat.s - b * r.s) / u_denom)

        Fx_rear = Fzr_safe * D_pacejka * sp.sin(C_pacejka * sp.atan(B_pacejka * slip_x - E_pacejka * (B_pacejka * slip_x - sp.atan(B_pacejka * slip_x))))
        Fy_front = Fzf_safe * D_pacejka_y * sp.sin(C_pacejka_y * sp.atan(B_pacejka_y * alpha_f - E_pacejka_y * (B_pacejka_y * alpha_f - sp.atan(B_pacejka_y * alpha_f))))
        Fy_rear = Fzr_safe * D_pacejka_y * sp.sin(C_pacejka_y * sp.atan(B_pacejka_y * alpha_r - E_pacejka_y * (B_pacejka_y * alpha_r - sp.atan(B_pacejka_y * alpha_r))))

        # Aerodynamic drag (CdA lumped) with optional ambient wind
        u_app = u.s + v_headwind * sp.cos(psi.s) - v_crosswind * sp.sin(psi.s)
        v_app = v_lat.s - v_headwind * sp.sin(psi.s) - v_crosswind * sp.cos(psi.s)
        Fx_drag = 0.5 * rho * CdA * u_app * sp.sqrt(u_app**2 + 1e-4)
        Fy_drag = 0.5 * rho * A_side * Cs0 * v_app * sp.sqrt(v_app**2 + 1e-4)

        # Rolling resistance opposes motion (smoothed sign u/|u|; tanh is not
        # supported in acausal output expressions, sqrt is)
        F_roll = Crr * (Fzf_safe + Fzr_safe) * u.s / safe_abs(u.s)

        # Grade / gravity component along the direction of travel.  sin(slope)
        # is a transcendental function of an input, which the acausal compiler
        # forbids inside *output* equations, so it is captured in a declared
        # algebraic variable that both the dynamics and the audit export can use.
        F_grade = self.declare_symbol(ev, "F_grade", self.name, kind=SymKind.var)
        # Longitudinal tyre force as a declared variable so it can appear in the
        # (transcendental-free) slip-loss output equation.
        Fx_rear_v = self.declare_symbol(ev, "Fx_rear", self.name, kind=SymKind.var)

        # Energy-audit exports (positive = power dissipated / stored against motion)
        p_aero = self.declare_symbol(ev, "p_aero", self.name, kind=SymKind.outp)
        p_roll = self.declare_symbol(ev, "p_roll", self.name, kind=SymKind.outp)
        p_climb = self.declare_symbol(ev, "p_climb", self.name, kind=SymKind.outp)
        p_bearing = self.declare_symbol(ev, "p_bearing", self.name, kind=SymKind.outp)
        p_slip = self.declare_symbol(ev, "p_slip", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(p_aero.s, Fx_drag * u_app + Fy_drag * v_app), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(p_roll.s, F_roll * u.s), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(p_climb.s, F_grade.s * u.s), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(p_bearing.s, B_wheel * w_rear.s**2), kind=EqnKind.outp)
        # Contact-patch slip dissipation = Fx * (wheel surface speed - body speed)
        self.declare_equation(sp.Eq(p_slip.s, Fx_rear_v.s * (w_rear.s * r_wheel - u.s)), kind=EqnKind.outp)

        self.add_eqs(
            [
                sp.Eq(F_grade.s, M_sym.s * 9.81 * sp.sin(slope.s)),
                sp.Eq(Fx_rear_v.s, Fx_rear),
                sp.Eq(M_sym.s * (du.s - v_lat.s * r.s), Fx_rear_v.s - Fy_front * sp.sin(delta.s) - Fx_drag - F_roll - F_grade.s),
                sp.Eq(M_sym.s * (dv_lat.s + u.s * r.s), Fy_front * sp.cos(delta.s) + Fy_rear - Fy_drag),
                sp.Eq(Iz_sym.s * dr.s, a * Fy_front * sp.cos(delta.s) - b * Fy_rear),
                sp.Eq(dpsi.s, r.s),
                sp.Eq(dx.s, u.s * sp.cos(psi.s) - v_lat.s * sp.sin(psi.s)),
                sp.Eq(dy.s, u.s * sp.sin(psi.s) + v_lat.s * sp.cos(psi.s)),
                sp.Eq(0, trq_rear.s - Fx_rear_v.s * r_wheel - B_wheel * w_rear.s - J_wheel * alpha_rear.s),
            ]
        )


# ===========================================================================
# 3. Causal control & biomechanics blocks
# ===========================================================================
def _smooth_bump(t, t0, t1, width=1.0):
    """Smooth 0->1->0 pulse active on [t0, t1] with tanh edges of given width."""
    return 0.5 * (jnp.tanh((t - t0) / width) - jnp.tanh((t - t1) / width))


class DriveCycleSource(LeafSystem):
    """Open-loop reference: rider torque, smooth road grade, smooth steering.

    Grade and steering use tanh transitions rather than step discontinuities so
    the ODE solver does not have to integrate across kinks in the continuous
    plant inputs.
    """

    def __init__(self, name="drive_cycle",
                 trq_startup=25.0, trq_cruise=15.0, trq_amplitude=5.0,
                 pedal_freq=1.5, t_ramp=1.5, grade_max=0.06, descent=-0.03,
                 grade_hold=None):
        super().__init__(name=name)
        self.trq_startup = trq_startup
        self.trq_cruise = trq_cruise
        self.trq_amplitude = trq_amplitude
        self.pedal_freq = pedal_freq
        self.t_ramp = t_ramp
        self.grade_max = grade_max
        self.descent = descent
        self.grade_hold = grade_hold

        self.declare_output_port(self.calc_human_trq, name="human_trq", requires_inputs=False)
        self.declare_output_port(self.calc_slope, name="slope", requires_inputs=False)
        self.declare_output_port(self.calc_steer_angle, name="steer_angle", requires_inputs=False)

    def calc_human_trq(self, time, state, *inputs, **params):
        human_trq_raw = jnp.where(
            time < self.t_ramp,
            self.trq_startup,
            self.trq_cruise + self.trq_amplitude * jnp.sin(2.0 * jnp.pi * self.pedal_freq * time),
        )
        return jnp.array([human_trq_raw])

    def calc_slope(self, time, state, *inputs, **params):
        if self.grade_hold is not None:
            # Sustained constant grade (thermal stress test), ramped smoothly.
            slope = self.grade_hold * 0.5 * (1.0 + jnp.tanh((time - 3.0) / 1.5))
            return jnp.array([slope])
        # Smooth climb between 5-20 s, smooth descent after 30 s.
        slope = (self.grade_max * _smooth_bump(time, 5.0, 20.0, width=1.5)
                 + self.descent * 0.5 * (1.0 + jnp.tanh((time - 33.0) / 1.5)))
        return jnp.array([slope])

    def calc_steer_angle(self, time, state, *inputs, **params):
        # A gentle lane-change doublet around 15-21 s.
        steer = (0.02 * _smooth_bump(time, 15.0, 18.0, width=0.6)
                 - 0.02 * _smooth_bump(time, 18.0, 21.0, width=0.6))
        return jnp.array([steer])


class HumanRiderBiomechanics(LeafSystem):
    """W'-balance rider model: cadence-limited power envelope plus anaerobic work
    capacity (W') depletion above critical power (CP). Sampled (ZOH) to avoid an
    algebraic loop with the plant cadence.
    """

    def __init__(self, name="rider_biomechanics", dt=0.01,
                 trq_startup=25.0, trq_cruise=15.0, trq_amplitude=5.0,
                 pedal_freq=1.5, t_ramp=1.5, w_max=12.57,
                 W0=20000.0, CP=150.0):
        super().__init__(name=name)
        self.dt = dt
        self.trq_startup = trq_startup
        self.trq_cruise = trq_cruise
        self.trq_amplitude = trq_amplitude
        self.pedal_freq = pedal_freq
        self.t_ramp = t_ramp
        self.w_max = w_max
        self.W0 = W0
        self.CP = CP

        self.declare_input_port(name="cadence")
        self.declare_discrete_state(default_value=jnp.array([trq_startup, W0]))
        self.declare_output_port(self.calc_human_trq, name="human_trq", requires_inputs=False)
        self.declare_output_port(self.calc_w_prime, name="w_prime", requires_inputs=False)
        self.declare_periodic_update(self._update, period=dt, offset=0.0)

    def _update(self, time, state, *inputs, **params):
        cadence = jnp.squeeze(inputs[0])
        W_prime_curr = state.discrete_state[1]

        human_trq_raw = jnp.where(
            time < self.t_ramp,
            self.trq_startup,
            self.trq_cruise + self.trq_amplitude * jnp.sin(2.0 * jnp.pi * self.pedal_freq * time),
        )
        cadence_factor = jnp.clip(1.0 - (cadence / self.w_max) ** 2, 0.0, 1.0)
        human_trq_requested = human_trq_raw * cadence_factor

        P_demanded = human_trq_requested * cadence
        P_excess = jnp.maximum(0.0, P_demanded - self.CP)
        dW_prime = -P_excess * self.dt
        W_prime_next = jnp.maximum(0.0, W_prime_curr + dW_prime)

        fatigue_factor = jnp.where(
            W_prime_next > 100.0,
            1.0,
            jnp.clip(self.CP / (P_demanded + 1.0), 0.2, 1.0),
        )
        human_trq_actual = human_trq_requested * fatigue_factor
        return jnp.array([human_trq_actual, W_prime_next])

    def calc_human_trq(self, time, state, *inputs, **params):
        return state.discrete_state[0:1]

    def calc_w_prime(self, time, state, *inputs, **params):
        return state.discrete_state[1:2]


class AssistPolicy(LeafSystem):
    """Torque-assist manager: proportional assist with speed fade to the legal
    cutoff, battery/motor thermal derating, and optional regenerative braking.
    Runs at a fixed control rate and outputs a q-axis current reference.
    """

    def __init__(self, name="assist_policy", dt=0.01,
                 K_assist=2.0,
                 v_cutoff=6.94, v_fade_start=6.39,
                 T_derate_bat_start=315.0, T_derate_bat_end=330.0,
                 T_derate_motor_start=340.0, T_derate_motor_end=360.0,
                 max_assist_torque=12.0,
                 v_regen_start=7.78, K_regen=5.0, max_regen_torque=15.0,
                 torque_constant=0.60):  # Nm/A, must equal motor Kt = 1.5*P*psi_m
        super().__init__(name=name)
        self.dt = dt
        self.v_cutoff = v_cutoff
        self.v_fade_start = v_fade_start
        self.T_derate_bat_start = T_derate_bat_start
        self.T_derate_bat_end = T_derate_bat_end
        self.T_derate_motor_start = T_derate_motor_start
        self.T_derate_motor_end = T_derate_motor_end
        self.max_assist_torque = max_assist_torque
        self.v_regen_start = v_regen_start
        self.K_regen = K_regen
        self.max_regen_torque = max_regen_torque
        self.torque_constant = torque_constant

        self.declare_input_port(name="cadence")
        self.declare_input_port(name="speed")
        self.declare_input_port(name="bat_temp")
        self.declare_input_port(name="motor_temp")

        self.declare_dynamic_parameter("K_assist", K_assist)
        # Assist torque cap [Nm] exposed as a tunable parameter for the
        # trajectory-optimization example (it is the effective "assist level"
        # knob; K_assist alone saturates against it).
        self.declare_dynamic_parameter("max_assist_torque", max_assist_torque)
        self.declare_discrete_state(default_value=jnp.array([0.0, 0.0]))

        self.declare_output_port(self.calc_iq_ref, name="iq_ref", requires_inputs=False)
        self.declare_output_port(self.calc_speed_zoh, name="speed_zoh", requires_inputs=False)
        self.declare_periodic_update(self._update, period=dt, offset=0.0)

    def _update(self, time, state, *inputs, **params):
        cadence = jnp.squeeze(inputs[0])
        speed = jnp.squeeze(inputs[1])
        bat_temp = jnp.squeeze(inputs[2])
        motor_temp = jnp.squeeze(inputs[3])
        K_assist = params["K_assist"]
        max_assist_torque = params["max_assist_torque"]

        human_trq_est = 15.0

        eta_speed = jnp.where(
            speed < self.v_cutoff,
            1.0 - (speed - self.v_fade_start) / (self.v_cutoff - self.v_fade_start),
            0.0,
        )
        eta_speed = jnp.clip(eta_speed, 0.0, 1.0)

        eta_thermal_bat = jnp.clip(
            1.0 - (bat_temp - self.T_derate_bat_start) / (self.T_derate_bat_end - self.T_derate_bat_start),
            0.0, 1.0,
        )
        eta_thermal_motor = jnp.clip(
            1.0 - (motor_temp - self.T_derate_motor_start) / (self.T_derate_motor_end - self.T_derate_motor_start),
            0.0, 1.0,
        )

        assist_trq = K_assist * human_trq_est * eta_speed * eta_thermal_bat * eta_thermal_motor
        k_sat = 10.0
        assist_trq = max_assist_torque - jnp.logaddexp(0.0, k_sat * (max_assist_torque - assist_trq)) / k_sat

        regen_trq = jnp.where(
            speed > self.v_regen_start,
            -self.K_regen * (speed - self.v_regen_start),
            0.0,
        )
        regen_trq = -self.max_regen_torque + jnp.logaddexp(0.0, k_sat * (regen_trq + self.max_regen_torque)) / k_sat

        total_trq = assist_trq + regen_trq
        iq_ref = total_trq / self.torque_constant
        return jnp.array([iq_ref, jnp.squeeze(speed)])

    def calc_iq_ref(self, time, state, *inputs, **params):
        return state.discrete_state[0:1]

    def calc_speed_zoh(self, time, state, *inputs, **params):
        return state.discrete_state[1:2]


class FOCController(LeafSystem):
    """Field-oriented current controller: two discrete PI loops (d,q) with SVPWM
    voltage-magnitude saturation. id_ref = 0 (non-flux-weakening region).
    """

    def __init__(self, name="foc_controller", dt=0.01,
                 Kp_d=0.5, Ki_d=50.0, Kp_q=0.5, Ki_q=50.0):
        super().__init__(name=name)
        self.dt = dt
        self.Kp_d = Kp_d
        self.Ki_d = Ki_d
        self.Kp_q = Kp_q
        self.Ki_q = Ki_q

        self.declare_input_port(name="iq_ref")
        self.declare_input_port(name="id_curr")
        self.declare_input_port(name="iq_curr")
        self.declare_input_port(name="v_dc")

        self.declare_discrete_state(default_value=jnp.array([0.0, 0.0, 0.0, 0.0]))
        self.declare_output_port(self.calc_vd_ctrl, name="vd_ctrl", requires_inputs=False)
        self.declare_output_port(self.calc_vq_ctrl, name="vq_ctrl", requires_inputs=False)
        self.declare_periodic_update(self._update, period=dt, offset=0.0)

    def _update(self, time, state, *inputs, **params):
        iq_ref = jnp.squeeze(inputs[0])
        id_curr = jnp.squeeze(inputs[1])
        iq_curr = jnp.squeeze(inputs[2])
        v_dc = jnp.squeeze(inputs[3])

        id_ref = 0.0
        e_d = id_ref - id_curr
        e_q = iq_ref - iq_curr

        z_vd = state.discrete_state[2]
        z_vq = state.discrete_state[3]
        z_vd_next = z_vd + e_d * self.dt
        z_vq_next = z_vq + e_q * self.dt

        vd_req = self.Kp_d * e_d + self.Ki_d * z_vd_next
        vq_req = self.Kp_q * e_q + self.Ki_q * z_vq_next

        v_max = v_dc / jnp.sqrt(3.0)
        v_mag = jnp.sqrt(vd_req**2 + vq_req**2 + 1e-4)
        scale = jnp.minimum(1.0, v_max / v_mag)

        vd_ctrl = vd_req * scale
        vq_ctrl = vq_req * scale
        return jnp.array([vd_ctrl, vq_ctrl, z_vd_next, z_vq_next])

    def calc_vd_ctrl(self, time, state, *inputs, **params):
        return state.discrete_state[0:1]

    def calc_vq_ctrl(self, time, state, *inputs, **params):
        return state.discrete_state[1:2]


class AssistSpeedLimiter(LeafSystem):
    """Hybrid state machine for the legal assist cutoff.

    Two modes (ENABLED=0, CUTOFF=1) with hysteresis. Zero-crossing guards locate
    the exact instant the vehicle speed crosses the cutoff (rather than a sampled
    ``jnp.where``), so the solver places a step there and the enable state flips
    precisely. Output ``enable`` gates the motor assist current.
    """

    ENABLED = 0
    CUTOFF = 1

    def __init__(self, name="speed_limiter", v_cutoff=6.94, v_hyst=0.3):
        super().__init__(name=name)
        self.v_cutoff = v_cutoff
        self.v_reengage = v_cutoff - v_hyst

        self.declare_input_port(name="speed")
        self.declare_continuous_state(shape=(), ode=self._ode)  # dummy state for ZC tracking
        self.declare_default_mode(self.ENABLED)
        self.declare_output_port(self._enable, name="enable", requires_inputs=False)

        self.declare_zero_crossing(
            guard=self._guard_cutoff, start_mode=self.ENABLED, end_mode=self.CUTOFF,
            direction="negative_then_non_negative", name="cutoff",
        )
        self.declare_zero_crossing(
            guard=self._guard_reengage, start_mode=self.CUTOFF, end_mode=self.ENABLED,
            direction="positive_then_non_positive", name="reengage",
        )

    def _ode(self, time, state, *inputs, **params):
        return jnp.zeros(())

    def _guard_cutoff(self, time, state, *inputs, **params):
        return jnp.squeeze(inputs[0]) - self.v_cutoff

    def _guard_reengage(self, time, state, *inputs, **params):
        return jnp.squeeze(inputs[0]) - self.v_reengage

    def _enable(self, time, state, *inputs, **params):
        return jnp.where(state.mode == self.ENABLED, 1.0, 0.0)


class PowerProbe(LeafSystem):
    """Feedthrough that outputs the scalar product of two input signals, used to
    form instantaneous power (e.g. torque * angular velocity) for the audit."""

    def __init__(self, name="power_probe"):
        super().__init__(name=name)
        self.declare_input_port(name="a")
        self.declare_input_port(name="b")
        self.declare_output_port(self._calc, name="power", requires_inputs=True)

    def _calc(self, time, state, *inputs, **params):
        return jnp.squeeze(inputs[0]) * jnp.squeeze(inputs[1])


# ===========================================================================
# 4. Diagram assembly
# ===========================================================================
def _find_port(sys, name):
    for port in sys.output_ports:
        if port.name == name:
            return port
    raise ValueError(f"Output port {name} not found in {sys.name}")


def _find_in_port(sys, name):
    for port in sys.input_ports:
        if port.name == name:
            return port
    raise ValueError(f"Input port {name} not found in {sys.name}")


def make_ebike_diagram(config: EbikeConfig | None = None, name="ebike_system",
                       return_handles=False, enable_speed_event=True,
                       battery_thermal_network=False, cooling_rbf_model=None):
    """Assemble the full multi-domain e-bike diagram from ``config``.

    Drivetrain kinematic chain (all rigid gears, power-conserving):

        human torque --> crank_inertia --+
                                          |  (shared intermediate shaft node)
        motor --> motor_gear (5:1) -------+--> wheel_gear (1:2) --> chain --> wheel

    so the crank turns at half wheel speed and the motor at 2.5x wheel speed.
    """
    if config is None:
        config = EbikeConfig()

    builder = jaxonomy.DiagramBuilder()
    ev = EqnEnv()
    ad = AcausalDiagram()

    gnd = elec.Ground(ev, name="gnd")

    def cell_factory(ev, cname):
        return HighFidelityBatteryCellECM(
            ev, name=cname, capacity_Ah=config.cell_capacity_Ah,
            initial_soc=config.initial_soc, initial_soc_fixed=True,
            n_series=config.n_series,
        )

    pack = bat.battery_pack(ev, ad, n_modules=1, n_cells_per_module=1,
                            cell_factory=cell_factory, name="battery")
    motor = HighFidelityPMSM(ev, name="motor", w_ic=0.757575,
                             enable_inverter_losses=True, enable_demagnetization=True)
    pack.connect_pos(ad, motor, "p")
    pack.connect_neg(ad, motor, "n")
    ad.connect(gnd, "p", motor, "n")

    # Thermal network
    ambient_temp = therm.TemperatureSource(ev, name="ambient_temp", temperature=298.15)
    use_rom_cooling = cooling_rbf_model is not None

    # ---- Battery thermal: single lumped node, or a radial multi-node network
    # (core -> mid -> surface) that resolves a spatial hot-spot. ---------------
    bat_surf_sensor = None
    if battery_thermal_network:
        bat_core = therm.HeatCapacitor(ev, name="bat_core", C=800.0, initial_temperature=298.15, initial_temperature_fixed=True)
        bat_mid = therm.HeatCapacitor(ev, name="bat_mid", C=700.0, initial_temperature=298.15, initial_temperature_fixed=True)
        bat_surf = therm.HeatCapacitor(ev, name="bat_surf", C=500.0, initial_temperature=298.15, initial_temperature_fixed=True)
        r_cm = therm.Insulator(ev, name="bat_r_cm", R=0.08)   # core->mid conduction
        r_ms = therm.Insulator(ev, name="bat_r_ms", R=0.08)   # mid->surface conduction
        for cell in pack.cells:
            ad.connect(cell, "heat", bat_core, "port")        # heat generated at the core
        ad.connect(bat_core, "port", r_cm, "port_a")
        ad.connect(r_cm, "port_b", bat_mid, "port")
        ad.connect(bat_mid, "port", r_ms, "port_a")
        ad.connect(r_ms, "port_b", bat_surf, "port")
        bat_cool_node, bat_sense_node = bat_surf, bat_core    # cool the skin, sense the core hot-spot
        bat_surf_sensor = therm.TemperatureSensor(ev, name="bat_surf_sensor", enable_port_b=False)
        ad.connect(bat_surf_sensor, "port_a", bat_surf, "port")
    else:
        bat_thermal = therm.HeatCapacitor(ev, name="bat_thermal", C=2000.0, initial_temperature=298.15, initial_temperature_fixed=True)
        for cell in pack.cells:
            ad.connect(cell, "heat", bat_thermal, "port")
        bat_cool_node = bat_sense_node = bat_thermal

    # Battery cooling: analytic speed-dependent, or surrogate-conductance-driven
    if use_rom_cooling:
        bat_cooling = SurrogateCooling(ev, name="bat_cooling")
    else:
        bat_cooling = SpeedDependentCooling(ev, name="bat_cooling", h_static=1.0, k_wind=0.3, A_case=0.15)
    ad.connect(bat_cool_node, "port", bat_cooling, "port_a")
    ad.connect(ambient_temp, "port", bat_cooling, "port_b")

    motor_thermal = therm.HeatCapacitor(ev, name="motor_thermal", C=800.0, initial_temperature=298.15, initial_temperature_fixed=True)
    ad.connect(motor, "heat", motor_thermal, "port")
    motor_cooling = SpeedDependentCooling(ev, name="motor_cooling", h_static=1.5, k_wind=0.5, A_case=0.1)
    ad.connect(motor_thermal, "port", motor_cooling, "port_a")
    ad.connect(ambient_temp, "port", motor_cooling, "port_b")

    bat_temp_sensor = therm.TemperatureSensor(ev, name="bat_temp_sensor", enable_port_b=False)
    motor_temp_sensor = therm.TemperatureSensor(ev, name="motor_temp_sensor", enable_port_b=False)
    ad.connect(bat_temp_sensor, "port_a", bat_sense_node, "port")
    ad.connect(motor_temp_sensor, "port_a", motor_thermal, "port")

    # Rider + crank
    human_trq = rot.TorqueSource(ev, name="human_trq", enable_torque_port=True, enable_flange_b=False)
    crank_inertia = rot.Inertia(ev, name="crank_inertia", I=0.1, initial_angle=0.0,
                                initial_angle_fixed=False, initial_velocity=0.151515,
                                initial_velocity_fixed=True)
    ad.connect(human_trq, "flange_a", crank_inertia, "flange")
    crank_cadence_sensor = rot.MotionSensor(ev, name="crank_cadence_sensor", enable_flange_b=False, enable_velocity_port=True)
    ad.connect(crank_inertia, "flange", crank_cadence_sensor, "flange_a")

    # Drivetrain: crank + motor both drive a shared intermediate shaft, which
    # is geared to the wheel through the chain compliance.  The crank couples
    # through a 1:1 gear (kept as an explicit element so the index reducer sees
    # a balanced set of rotational initial conditions).
    crank_gear = rot.GearRatio(ev, name="crank_gear", ratio=1.0)
    motor_gear = rot.GearRatio(ev, name="motor_gear", ratio=config.motor_reduction)
    wheel_gear = rot.GearRatio(ev, name="wheel_gear", ratio=config.wheel_reduction)
    ad.connect(crank_inertia, "flange", crank_gear, "flange_a")
    ad.connect(motor, "shaft", motor_gear, "flange_a")
    ad.connect(crank_gear, "flange_b", wheel_gear, "flange_a")    # crank -> intermediate (1:1)
    ad.connect(motor_gear, "flange_b", wheel_gear, "flange_a")    # motor -> intermediate (5:1)

    # Chain compliance (torsional spring + damper).  NOTE: a true rear freewheel
    # (one_way=True) is implemented on this component but destabilises the stiff
    # acausal DAE without complementarity/event support, so the reference model
    # uses the two-way coupling.  Consequence: the crank cannot coast, so cadence
    # tracks wheel speed on descents (see KNOWN LIMITATIONS in the walkthrough).
    chain = TorsionalSpringDamper(ev, name="chain", K=2000.0, D=50.0, one_way=False)
    ad.connect(wheel_gear, "flange_b", chain, "flange_a")

    vehicle = PlanarVehicleDynamics(
        ev, name="vehicle", r_wheel=config.r_wheel, M_total=config.m_total,
        rho=config.rho_air, CdA=config.CdA, Crr=config.Crr,
        v_headwind=0.0, v_crosswind=0.0, enable_dynamic_weight_transfer=True,
    )
    ad.connect(chain, "flange_b", vehicle, "shaft_rear")

    compiler = AcausalCompiler(ev, ad, scale=True, verbose=False)
    phys_sys = builder.add(compiler())

    # Controllers
    drive_cycle = builder.add(DriveCycleSource(grade_hold=config.grade_hold))
    assist_policy = builder.add(AssistPolicy(
        dt=0.01, K_assist=config.K_assist,
        v_cutoff=config.v_cutoff_kmh / 3.6, v_fade_start=config.v_fade_kmh / 3.6,
        max_assist_torque=config.max_assist_torque,
    ))
    foc = builder.add(FOCController(dt=0.01))
    rider_bio = builder.add(HumanRiderBiomechanics())

    builder.connect(_find_port(phys_sys, "crank_cadence_sensor_w_rel"), assist_policy.input_ports[0])
    builder.connect(_find_port(phys_sys, "vehicle_u_out"), assist_policy.input_ports[1])
    builder.connect(_find_port(phys_sys, "bat_temp_sensor_T_rel"), assist_policy.input_ports[2])
    builder.connect(_find_port(phys_sys, "motor_temp_sensor_T_rel"), assist_policy.input_ports[3])

    # Precise legal-speed cutoff as a hybrid zero-crossing event: the limiter's
    # enable signal gates the assist current before it reaches the FOC loop.
    # Disabled for gradient-based optimization: the integer mode variable of the
    # zero-crossing state machine cannot carry a reverse-mode cotangent, so
    # optimization uses the (fully differentiable) smooth speed fade already
    # built into the assist policy, and the optimum is validated on the full
    # hybrid model with the event enabled.
    speed_limiter = None
    if enable_speed_event:
        speed_limiter = builder.add(AssistSpeedLimiter(
            name="speed_limiter", v_cutoff=config.v_cutoff_kmh / 3.6))
        builder.connect(_find_port(phys_sys, "vehicle_u_out"), speed_limiter.input_ports[0])
        assist_gate = builder.add(PowerProbe(name="assist_gate"))  # iq_ref * enable
        builder.connect(assist_policy.output_ports[0], assist_gate.input_ports[0])
        builder.connect(speed_limiter.output_ports[0], assist_gate.input_ports[1])
        builder.connect(assist_gate.output_ports[0], foc.input_ports[0])
    else:
        builder.connect(assist_policy.output_ports[0], foc.input_ports[0])

    builder.connect(_find_port(phys_sys, "motor_id_out"), foc.input_ports[1])
    builder.connect(_find_port(phys_sys, "motor_iq_out"), foc.input_ports[2])
    builder.connect(_find_port(phys_sys, "battery_mod0_cell0_v_out"), foc.input_ports[3])

    builder.connect(foc.output_ports[0], _find_in_port(phys_sys, "motor_vd_ctrl"))
    builder.connect(foc.output_ports[1], _find_in_port(phys_sys, "motor_vq_ctrl"))

    # W'-balance rider drives the human torque port
    builder.connect(_find_port(phys_sys, "crank_cadence_sensor_w_rel"), rider_bio.input_ports[0])
    builder.connect(rider_bio.output_ports[0], _find_in_port(phys_sys, "human_trq_tau"))
    trq_signal = rider_bio.output_ports[0]

    builder.connect(drive_cycle.output_ports[1], _find_in_port(phys_sys, "vehicle_slope"))
    builder.connect(drive_cycle.output_ports[2], _find_in_port(phys_sys, "vehicle_steer_angle"))

    builder.connect(assist_policy.output_ports[1], _find_in_port(phys_sys, "motor_cooling_speed"))
    if use_rom_cooling:
        # Reduced-order surrogate maps vehicle speed -> battery cooling
        # conductance and drives the physical cooling link directly.
        from jaxonomy.library.rom import RadialBasisSurrogate
        cooling_rom = builder.add(RadialBasisSurrogate(cooling_rbf_model, name="bat_cooling_rom"))
        builder.connect(assist_policy.output_ports[1], cooling_rom.input_ports[0])
        builder.connect(cooling_rom.output_ports[0], _find_in_port(phys_sys, "bat_cooling_h_cond"))
    else:
        builder.connect(assist_policy.output_ports[1], _find_in_port(phys_sys, "bat_cooling_speed"))

    # ---- Energy-audit instrumentation: integrate each power flow online -----
    def _integrate(signal_port, label):
        integ = builder.add(Integrator(0.0, name=f"E_{label}"))
        builder.connect(signal_port, integ.input_ports[0])
        builder.export_output(integ.output_ports[0], f"E_{label}")

    # human mechanical power = rider torque * crank cadence
    p_human = builder.add(PowerProbe(name="p_human"))
    builder.connect(trq_signal, p_human.input_ports[0])
    builder.connect(_find_port(phys_sys, "crank_cadence_sensor_w_rel"), p_human.input_ports[1])
    _integrate(p_human.output_ports[0], "human")

    _integrate(_find_port(phys_sys, "battery_mod0_cell0_p_term"), "batt_term")
    _integrate(_find_port(phys_sys, "battery_mod0_cell0_q_heat"), "batt_heat")
    _integrate(_find_port(phys_sys, "motor_p_heat"), "motor_heat")
    _integrate(_find_port(phys_sys, "vehicle_p_aero"), "aero")
    _integrate(_find_port(phys_sys, "vehicle_p_roll"), "roll")
    _integrate(_find_port(phys_sys, "vehicle_p_climb"), "climb")
    _integrate(_find_port(phys_sys, "vehicle_p_bearing"), "bearing")
    _integrate(_find_port(phys_sys, "vehicle_p_slip"), "slip")
    _integrate(_find_port(phys_sys, "chain_p_chain"), "chain")

    # Distance integrator: final value / tf gives mean speed as a final-state
    # quantity (usable as an objective term under enable_autodiff, where no time
    # series is recorded).
    dist_integ = builder.add(Integrator(0.0, name="distance"))
    builder.connect(_find_port(phys_sys, "vehicle_u_out"), dist_integ.input_ports[0])
    builder.export_output(dist_integ.output_ports[0], "distance")

    # ---- Signals for plots / metrics ---------------------------------------
    builder.export_output(_find_port(phys_sys, "battery_mod0_cell0_soc"), "soc")
    builder.export_output(_find_port(phys_sys, "bat_temp_sensor_T_rel"), "bat_temp")
    if bat_surf_sensor is not None:
        builder.export_output(_find_port(phys_sys, "bat_surf_sensor_T_rel"), "bat_surf_temp")
    builder.export_output(_find_port(phys_sys, "motor_temp_sensor_T_rel"), "motor_temp")
    builder.export_output(_find_port(phys_sys, "crank_cadence_sensor_w_rel"), "cadence")
    builder.export_output(_find_port(phys_sys, "vehicle_u_out"), "speed")
    builder.export_output(_find_port(phys_sys, "battery_mod0_cell0_v_out"), "v_dc")
    builder.export_output(_find_port(phys_sys, "battery_mod0_cell0_soh_out"), "soh")
    builder.export_output(_find_port(phys_sys, "motor_id_out"), "id_curr")
    builder.export_output(_find_port(phys_sys, "motor_iq_out"), "iq_curr")
    builder.export_output(_find_port(phys_sys, "motor_T_stator_out"), "T_stator")
    builder.export_output(_find_port(phys_sys, "vehicle_v_out"), "v_lat")
    builder.export_output(_find_port(phys_sys, "vehicle_yaw_rate_out"), "yaw_rate")
    builder.export_output(_find_port(phys_sys, "vehicle_pos_x_out"), "pos_x")
    builder.export_output(_find_port(phys_sys, "vehicle_pos_y_out"), "pos_y")
    builder.export_output(_find_port(phys_sys, "chain_defl"), "chain_defl")
    builder.export_output(trq_signal, "human_trq")
    builder.export_output(rider_bio.output_ports[1], "w_prime")
    if speed_limiter is not None:
        builder.export_output(speed_limiter.output_ports[0], "assist_enable")

    diagram = builder.build(name=name)
    if return_handles:
        return diagram, {"assist_policy": assist_policy, "speed_limiter": speed_limiter}
    return diagram


def simulate_ebike(config: EbikeConfig | None = None, tf=None):
    if config is None:
        config = EbikeConfig()
    if tf is None:
        tf = config.tf
    diagram = make_ebike_diagram(config)
    context = diagram.create_context()
    options = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6, buffer_length=120000)
    recorded_signals = {port.name: port for port in diagram.output_ports}
    return jaxonomy.simulate(diagram, context, (0.0, tf), options=options, recorded_signals=recorded_signals)


# ===========================================================================
# 5. Energy audit & validation
# ===========================================================================
# Rotational inertias in the stored-energy accounting (kg m^2) and the gear
# speed ratios that relate each shaft to the wheel.
_J_CRANK = 0.1
_J_MOTOR = 0.01
_J_WHEEL = 0.15


def energy_audit(results, config: EbikeConfig):
    """Close the energy balance for the whole vehicle+powertrain.

    Sources (into the mechanical/electrical system):
        E_human       rider mechanical work
        E_batt_term   electrical energy out of the pack terminals

    Sinks / storage (out of, or stored by, the system):
        dKE           change in translational + rotational kinetic energy
        dPE_chain     change in chain-spring potential energy
        E_climb       gravitational PE gained on grades
        E_aero        aerodynamic dissipation
        E_roll        rolling-resistance dissipation
        E_bearing     wheel-bearing viscous dissipation
        E_chain_damp  chain-damper dissipation
        E_motor_heat  motor copper + core + inverter heat

    (Battery-internal heat is *not* a sink here because we use terminal, not
    chemical, energy as the electrical source; it is reported separately.)
    """
    o = results.outputs

    def last(key):
        return float(np.asarray(o[key])[-1])

    def first(key):
        return float(np.asarray(o[key])[0])

    # Stored kinetic energy: translational + yaw + rotating inertias.  The crank
    # (and, through the rigid 5:1 gear, the motor) can decouple from the wheel
    # via the freewheel, so use the *measured* crank cadence rather than assuming
    # it tracks wheel speed.
    def ke(u, v, yaw_rate, w_wheel, w_crank):
        w_motor = 5.0 * w_crank          # rigid gear: motor at 5x crank speed
        ke_trans = 0.5 * config.m_total * (u**2 + v**2)
        ke_yaw = 0.5 * 120.0 * yaw_rate**2
        ke_rot = 0.5 * (_J_WHEEL * w_wheel**2 + _J_CRANK * w_crank**2 + _J_MOTOR * w_motor**2)
        return ke_trans + ke_yaw + ke_rot

    u0, uf = first("speed"), last("speed")
    v0, vf = first("v_lat"), last("v_lat")
    r0, rf = first("yaw_rate"), last("yaw_rate")
    cad0, cadf = first("cadence"), last("cadence")
    # Wheel speed reconstructed from vehicle speed (u = w_wheel * r_wheel).
    ww0, wwf = u0 / config.r_wheel, uf / config.r_wheel

    dKE = ke(uf, vf, rf, wwf, cadf) - ke(u0, v0, r0, ww0, cad0)

    terms = {
        "E_human": last("E_human"),
        "E_batt_term": last("E_batt_term"),
        "dKE": dKE,
        "E_climb": last("E_climb"),
        "E_aero": last("E_aero"),
        "E_roll": last("E_roll"),
        "E_bearing": last("E_bearing"),
        "E_slip": last("E_slip"),
        "E_chain": last("E_chain"),     # net into freewheel = stored + dissipated
        "E_motor_heat": last("E_motor_heat"),
        "E_batt_heat": last("E_batt_heat"),  # reported, not in the balance
    }

    E_in = terms["E_human"] + terms["E_batt_term"]
    E_out = (terms["dKE"] + terms["E_climb"] + terms["E_aero"]
             + terms["E_roll"] + terms["E_bearing"] + terms["E_slip"]
             + terms["E_chain"] + terms["E_motor_heat"])
    residual = E_in - E_out
    closure_pct = abs(residual) / (abs(E_in) + 1e-9) * 100.0

    terms.update({
        "E_in": E_in,
        "E_out": E_out,
        "residual": residual,
        "closure_error_pct": closure_pct,
    })
    return terms


def validate(results, config: EbikeConfig, closure_tol_pct=3.0):
    """Run sanity checks and return (ok, messages)."""
    o = results.outputs
    msgs = []
    ok = True

    audit = energy_audit(results, config)
    if audit["closure_error_pct"] > closure_tol_pct:
        ok = False
        msgs.append(f"[FAIL] energy closure error {audit['closure_error_pct']:.2f}% > {closure_tol_pct}%")
    else:
        msgs.append(f"[ok]   energy closure error {audit['closure_error_pct']:.2f}% (<= {closure_tol_pct}%)")

    v_max_kmh = float(np.max(np.asarray(o["speed"]))) * 3.6
    if not (5.0 < v_max_kmh < 60.0):
        ok = False
        msgs.append(f"[FAIL] max speed {v_max_kmh:.1f} km/h out of plausible range")
    else:
        msgs.append(f"[ok]   max speed {v_max_kmh:.1f} km/h")

    soc_f = float(np.asarray(o["soc"])[-1])
    soc_drop = config.initial_soc - soc_f
    if not (0.0 <= soc_drop < 0.5):
        ok = False
        msgs.append(f"[FAIL] SOC drop {soc_drop*100:.2f}% implausible")
    else:
        msgs.append(f"[ok]   SOC drop {soc_drop*100:.2f}% over {config.tf:.0f}s")

    T_stator_f = float(np.asarray(o["T_stator"])[-1]) - 273.15
    if not (15.0 < T_stator_f < 180.0):
        ok = False
        msgs.append(f"[FAIL] final stator temp {T_stator_f:.1f} C implausible")
    else:
        msgs.append(f"[ok]   final stator temp {T_stator_f:.1f} C")

    return ok, msgs


def print_report(results, config: EbikeConfig):
    o = results.outputs
    audit = energy_audit(results, config)

    print("\n" + "=" * 68)
    print("  SMART CARGO E-BIKE — REFERENCE DRIVE CYCLE REPORT")
    print("=" * 68)
    print(f"  Total mass            : {config.m_total:.0f} kg "
          f"(bike+rider {config.m_bike_rider:.0f} + cargo {config.m_cargo:.0f})")
    print(f"  Mean speed            : {float(np.asarray(o['speed']).mean())*3.6:6.2f} km/h")
    print(f"  Max speed             : {float(np.max(np.asarray(o['speed'])))*3.6:6.2f} km/h")
    print(f"  Peak cadence          : {float(np.max(np.asarray(o['cadence'])))*60/(2*np.pi):6.1f} rpm")
    print(f"  Final SOC             : {float(np.asarray(o['soc'])[-1]):6.4f}")
    print(f"  Final stator temp     : {float(np.asarray(o['T_stator'])[-1])-273.15:6.2f} C")
    print(f"  Final battery temp    : {float(np.asarray(o['bat_temp'])[-1])-273.15:6.2f} C")

    print("\n  ENERGY AUDIT (J)")
    print("  " + "-" * 50)
    print(f"  {'IN  human work':<26}: {audit['E_human']:10.1f}")
    print(f"  {'IN  battery (terminal)':<26}: {audit['E_batt_term']:10.1f}")
    print(f"  {'TOTAL IN':<26}: {audit['E_in']:10.1f}")
    print("  " + "-" * 50)
    print(f"  {'OUT dKE (stored)':<26}: {audit['dKE']:10.1f}")
    print(f"  {'OUT climb (grav. PE)':<26}: {audit['E_climb']:10.1f}")
    print(f"  {'OUT aero loss':<26}: {audit['E_aero']:10.1f}")
    print(f"  {'OUT rolling loss':<26}: {audit['E_roll']:10.1f}")
    print(f"  {'OUT bearing loss':<26}: {audit['E_bearing']:10.1f}")
    print(f"  {'OUT tyre-slip loss':<26}: {audit['E_slip']:10.1f}")
    print(f"  {'OUT freewheel (net)':<26}: {audit['E_chain']:10.1f}")
    print(f"  {'OUT motor heat':<26}: {audit['E_motor_heat']:10.1f}")
    print(f"  {'TOTAL OUT':<26}: {audit['E_out']:10.1f}")
    print("  " + "-" * 50)
    print(f"  {'RESIDUAL':<26}: {audit['residual']:10.1f}")
    print(f"  {'CLOSURE ERROR':<26}: {audit['closure_error_pct']:9.2f} %")
    print(f"  (battery-internal heat, reported : {audit['E_batt_heat']:.1f} J)")

    ok, msgs = validate(results, config)
    print("\n  VALIDATION")
    print("  " + "-" * 50)
    for m in msgs:
        print("  " + m)
    print("=" * 68)
    return audit, ok


if __name__ == "__main__":
    print("Building and simulating the reference cargo e-bike drive cycle...")
    cfg = EbikeConfig()
    t0 = time.time()
    res = simulate_ebike(cfg)
    print(f"Simulated {cfg.tf:.0f} s in {time.time()-t0:.2f} s wall-clock.")
    print_report(res, cfg)
