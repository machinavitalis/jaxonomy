"""f1_lts_common.py — F1 lap-time simulator plant + helpers.

Extracted from docs/examples/f1_part_2_setup_optimization.ipynb (cells 1-9)
because Part 2 already factors the plant cleanly with no exploratory plotting
in those cells. This module is imported by the F1-series offline-publication
scripts (parts 2, 5, 6) so they can run the full LTS at publication fidelity
without re-deriving the plant.

The plant + helpers exposed:
  pacejka, friction_ellipse_split            (cell 3)
  _normal_loads, _drag, car_ode_rhs, BicycleCar  (cell 4)
  Powertrain                                 (cell 5)
  kappa_track, centerline_xy, mu_eff_at_speed, build_speed_profile,
    lookup_vref, lookup_kappa, Driver, MuxControls, CarStateSplit,
    DemuxDriver                              (cell 6)
  SETUP_BASELINE, SETUP_LOWER, SETUP_UPPER, SETUP_NAMES, N_SETUP,
    setup_to_physics                         (cell 7)
  LapTimeAccumulator, CarArcLength           (cell 8)
  build_lap_diagram, DIAG, CAR_BLK, DRV_BLK, PT_BLK, LAP_BLK, CTX0   (cell 9)
"""



# Standard scientific Python + Part-1 inheritance.
from __future__ import annotations

import time as _time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize

# JAX — float64 throughout, same as Part 1.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

# jaxonomy
import jaxonomy
from jaxonomy import DiagramBuilder, LeafSystem, simulate
from jaxonomy.simulation import SimulatorOptions
from jaxonomy.library import Constant
from jaxonomy.diagnostics import analyze_saturation, analyze_control_oscillation
from jaxonomy import logging as jx_logging

jx_logging.set_log_level(jx_logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*ring-buffer.*")  # tame the T-002b warning during sweeps

# One RNG_SEED for the whole notebook.
RNG_SEED = 0
np.random.seed(RNG_SEED)
print(f"jaxonomy {jaxonomy.__version__}  |  jax {jax.__version__}")


# ## 1. Why setup optimisation matters in F1
# 
# Every car that takes the grid on Sunday is the product of a *setup* — a vector of mechanical, aerodynamic, and electronic parameters chosen the day before. The choices are tightly coupled: stiffer rear springs raise the rear ride height under load, which steepens the underfloor and shifts the aerodynamic centre of pressure forward; softer front dampers slow weight transfer into corners, which delays the front-axle tyre saturation and changes mid-corner balance. There is no analytic optimum — every track, weather, and tyre compound has its own. So teams run laps and adjust.
# 
# The economics of those laps are extreme. A 2025-spec power-unit upgrade costs several million euros; a few hours of wind-tunnel time, hundreds of thousands; a single bad setup that costs three grid positions on Saturday can erase the upgrade entirely. Worse, the FIA's *aerodynamic-testing restriction* (Appendix 6 to the International Sporting Code) gives the constructors-championship leader the **smallest** CFD and wind-tunnel budget, so the cost of trial-and-error scales inversely with championship position. Sample-efficient setup search is therefore not a nice-to-have — it is a structural advantage.
# 
# Today's commercial lap-time simulators address this by **finite-difference sensitivity sweeps** around a baseline setup. For each setup parameter $\theta_i$, you re-run the simulator at $\theta_i \pm \Delta$ and use $(\text{lap}(\theta_i+\Delta) - \text{lap}(\theta_i-\Delta)) / (2\Delta)$ as the sensitivity. With $N$ parameters that's $2N$ extra simulations for a gradient, and $3^N$ simulations for a one-step-per-axis grid. The 8-D slice we tackle in this notebook would cost $6561$ forward runs of the LTS as a grid search — roughly an hour and three quarters at one second per simulation. `jax.grad` does the same job in a single backward pass.
# 
# > **Why one backward pass.** The simulator we built in Part 1 is a composition of differentiable JAX primitives — Pacejka magic-formula, ODE integration via Diffrax under the hood, PCHIP-interpolated engine map, driver feedforward. Each piece is `jax.grad`-able; the composition therefore is too. `jax.grad` reverse-mode AD computes the gradient of a scalar w.r.t. an $N$-vector in $O(\text{forward-pass-time})$ — *independent of $N$*. With $N = 8$ that's ~$2\times$ a forward pass; with $N = 100$ (real-team scale) it would still be ~$2\times$ a forward pass. The grid-search cost, in contrast, scales as $3^N$.
# 

# ## 2. Recap: the Part-1 lap-time simulator
# 
# Part 1 built five blocks that we re-use here verbatim: the 8-state `BicycleCar` (longitudinal-lateral-yaw with Pacejka 5.2 tyres + friction-ellipse + body-frame Newton + passive position/arc-length/wheel integrators), the `Powertrain` (engine map + 7-speed gearbox + brake-bias splitter), the `Driver` (quasi-steady-state lookahead tracker against a Casanova forward-backward speed profile), the `MuxControls` adapter, and the synthetic 4-corner track $\kappa(s)$. Because Part 1 is a notebook rather than an importable module, we paste the class definitions inline below — *no logic changes*, just lifting Part-1 cells into Part-2 cells. The reader familiar with Part 1 can skim through §2.1–§2.5; the meat of this notebook starts at §3 where we introduce the setup-to-aero/grip mapping.
# 



# ---------- Part-1 chassis / aero / tyre / powertrain constants ----------
M_CAR     = 830.0
IZZ       = 1350.0
A_LEN     = 1.30
B_LEN     = 1.95
H_COG     = 0.32
L_WB      = A_LEN + B_LEN
RHO_AIR   = 1.225
CLA       = 3.5
CDA       = 1.1
BETA_AERO_F = 0.45
MU_PEAK   = 1.7
PJ_BX, PJ_CX, PJ_EX = 10.0, 1.65, 0.97
PJ_BY, PJ_CY, PJ_EY = 9.0,  1.30, 0.97
ENG_RPM_BRK  = np.array([1500., 3000., 5000., 7000., 9000., 10500., 12000., 13500., 15000.])
ENG_TRQ_BRK  = np.array([300., 410., 470., 510., 540., 560., 555., 510., 410.])
GEAR_RATIOS  = np.array([12.0, 9.0, 7.0, 5.8, 4.9, 4.3, 3.8])
N_GEARS      = len(GEAR_RATIOS)
SHIFT_RPM_UP, SHIFT_RPM_DN, SHIFT_DT = 13800.0, 9500.0, 0.050
ETA_DRIVE = 0.93
T_BRAKE_PEAK_F, T_BRAKE_PEAK_R, BRAKE_BIAS_F = 6_000.0, 6_000.0, 0.58
R_WHEEL = 0.330
G_ACC = 9.81
DELTA_MAX_RAD = np.deg2rad(20.0)
EPS_SPEED = 1.0e-1
I_WHEEL = 1.20




# ---------- Part-1 pure-functional helpers ----------
def pacejka(s, Fz, B, C, D_mu, E):
    Bs = B * s
    inner = Bs - E * (Bs - jnp.arctan(Bs))
    return D_mu * Fz * jnp.sin(C * jnp.arctan(inner))


def friction_ellipse_split(Fx_avail, Fy_avail, Fx_demand, Fy_demand):
    rho2 = (Fx_demand / Fx_avail) ** 2 + (Fy_demand / Fy_avail) ** 2
    rho = jnp.sqrt(jnp.maximum(rho2, 1e-12))
    scale = jnp.where(rho > 1.0, 1.0 / rho, 1.0)
    return Fx_demand * scale, Fy_demand * scale


def _normal_loads(u, m, beta_f, CLA_, a=A_LEN, b=B_LEN, L=L_WB,
                  g=G_ACC, rho=RHO_AIR):
    F_aero = 0.5 * rho * CLA_ * u * u
    Fz_f = m * g * b / L + beta_f * F_aero
    Fz_r = m * g * a / L + (1.0 - beta_f) * F_aero
    return Fz_f, Fz_r


def _drag(u, CDA_):
    return 0.5 * RHO_AIR * CDA_ * u * u * jnp.sign(u)




# ---------- BicycleCar (Part-1 dynamics, extended params for Part 2) ----------
# Part-2 change: m, mu, CLA, CDA, BETA_AERO_F are dynamic parameters so the
# setup vector can write into them via context.with_parameters({...}). Pacejka
# coefficients stay constants (we don't tune them in this notebook).

def car_ode_rhs(state, control, m, mu, CLA_, CDA_, beta_f,
                Izz=IZZ, a=A_LEN, b=B_LEN, rw=R_WHEEL, Iw=I_WHEEL,
                Bx=PJ_BX, Cx=PJ_CX, Ex=PJ_EX,
                By=PJ_BY, Cy=PJ_CY, Ey=PJ_EY):
    u, v, r, psi, X, Y, s_arc, ww = state
    delta, T_drive, T_brake = control
    u_safe = jnp.where(jnp.abs(u) < EPS_SPEED, EPS_SPEED * jnp.sign(u + 1e-12), u)

    af = delta - jnp.arctan((v + a * r) / u_safe)
    ar = -jnp.arctan((v - b * r) / u_safe)
    u_wr = u
    kr = (ww * rw - u_wr) / (jnp.abs(u_wr) + EPS_SPEED)

    Fzf, Fzr = _normal_loads(u, m, beta_f, CLA_)
    Fzf = jnp.maximum(Fzf, 1.0)
    Fzr = jnp.maximum(Fzr, 1.0)

    Fx_avail_f, Fy_avail_f = mu * Fzf, mu * Fzf
    Fx_avail_r, Fy_avail_r = mu * Fzr, mu * Fzr
    Fy_f_raw = pacejka(af, Fzf, By, Cy, mu, Ey)
    Fx_f_raw = 0.0
    Fx_r_raw = pacejka(kr, Fzr, Bx, Cx, mu, Ex)
    Fy_r_raw = pacejka(ar, Fzr, By, Cy, mu, Ey)
    Fx_f, Fy_f = friction_ellipse_split(Fx_avail_f, Fy_avail_f, Fx_f_raw, Fy_f_raw)
    Fx_r, Fy_r = friction_ellipse_split(Fx_avail_r, Fy_avail_r, Fx_r_raw, Fy_r_raw)

    cd, sd = jnp.cos(delta), jnp.sin(delta)
    F_drag_x = _drag(u, CDA_)
    Fx_body = Fx_f * cd - Fy_f * sd + Fx_r - F_drag_x
    Fy_body = Fx_f * sd + Fy_f * cd + Fy_r
    tau_z   = a * (Fx_f * sd + Fy_f * cd) - b * Fy_r

    du   = Fx_body / m + v * r
    dv   = Fy_body / m - u * r
    dr_  = tau_z / Izz
    dpsi = r
    dX   = u * jnp.cos(psi) - v * jnp.sin(psi)
    dY   = u * jnp.sin(psi) + v * jnp.cos(psi)
    ds   = jnp.sqrt(u * u + v * v)
    dww  = (T_drive - T_brake - Fx_r * rw) / Iw
    return jnp.array([du, dv, dr_, dpsi, dX, dY, ds, dww])


class BicycleCar(LeafSystem):
    """Same as Part 1, with (m, mu, CLA, CDA, beta_aero_f) promoted to
    dynamic parameters so the setup vector can write into them."""

    def __init__(self, x0=None, name="car"):
        super().__init__(name=name)
        # Setup-tunable dynamic parameters
        self.declare_dynamic_parameter("m",      float(M_CAR))
        self.declare_dynamic_parameter("mu",     float(MU_PEAK))
        self.declare_dynamic_parameter("CLA",    float(CLA))
        self.declare_dynamic_parameter("CDA",    float(CDA))
        self.declare_dynamic_parameter("beta_f", float(BETA_AERO_F))
        if x0 is None:
            x0 = jnp.zeros(8)
        self.declare_input_port(name="u")
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, *inputs, **params):
        x = state.continuous_state
        (u_ctrl,) = inputs
        return car_ode_rhs(
            x, u_ctrl,
            m=params["m"], mu=params["mu"],
            CLA_=params["CLA"], CDA_=params["CDA"], beta_f=params["beta_f"],
        )




# ---------- Powertrain (Part-1 verbatim) ----------
DT_POWERTRAIN = 0.01


class Powertrain(LeafSystem):
    """Part-1 powertrain, unchanged."""

    def __init__(self, dt=DT_POWERTRAIN, name="powertrain"):
        super().__init__(name=name)
        for nm, val in dict(
            eta_drive=ETA_DRIVE,
            T_brake_peak_f=T_BRAKE_PEAK_F, T_brake_peak_r=T_BRAKE_PEAK_R,
            brake_bias_f=BRAKE_BIAS_F,
            shift_rpm_up=SHIFT_RPM_UP, shift_rpm_dn=SHIFT_RPM_DN,
            shift_dt=SHIFT_DT,
        ).items():
            self.declare_dynamic_parameter(nm, float(val))
        self.dt = float(dt)
        self.declare_input_port(name="u_throttle")
        self.declare_input_port(name="u_brake")
        self.declare_input_port(name="omega_w")
        self.declare_discrete_state(default_value=jnp.array([2.0, 0.0]), as_array=True)
        self.declare_periodic_update(callback=self._gear_update, period=self.dt, offset=0.0)
        self.declare_output_port(self._torques_out, name="torques",
                                  requires_inputs=True,
                                  default_value=jnp.array([0.0, 0.0]))
        self.declare_output_port(self._gear_out, name="gear",
                                  requires_inputs=False,
                                  default_value=jnp.array(2.0),
                                  prerequisites_of_calc=[
                                      jaxonomy.framework.dependency_graph.DependencyTicket.xd])
        self.declare_output_port(self._rpm_out, name="engine_rpm",
                                  requires_inputs=True,
                                  default_value=jnp.array(3000.0))

    def _gear_update(self, time, state, *inputs, **params):
        gear_f, timer = state.discrete_state
        omega_w = inputs[2]
        gear_int = jnp.round(gear_f).astype(jnp.int32)
        ratio = jnp.asarray(GEAR_RATIOS)[gear_int]
        eng_rpm = jnp.abs(omega_w) * ratio * 60.0 / (2 * jnp.pi)
        new_timer = jnp.maximum(0.0, timer - self.dt)
        can_up = (new_timer <= 0.0) & (eng_rpm >= params["shift_rpm_up"]) & (gear_int < N_GEARS - 1)
        can_dn = (new_timer <= 0.0) & (eng_rpm <= params["shift_rpm_dn"]) & (gear_int > 0)
        new_gear = jnp.where(can_up, gear_int + 1,
                              jnp.where(can_dn, gear_int - 1, gear_int))
        new_timer = jnp.where(can_up | can_dn, params["shift_dt"], new_timer)
        return jnp.array([new_gear.astype(jnp.float64), new_timer])

    def _torques_out(self, time, state, *inputs, **params):
        u_thr, u_brk, omega_w = inputs
        gear_f, timer = state.discrete_state
        gear_int = jnp.round(gear_f).astype(jnp.int32)
        ratio = jnp.asarray(GEAR_RATIOS)[gear_int]
        eng_rpm = jnp.abs(omega_w) * ratio * 60.0 / (2 * jnp.pi)
        eng_rpm_q = jnp.clip(eng_rpm, ENG_RPM_BRK[0], ENG_RPM_BRK[-1])
        from jaxonomy.library.lookup_table import interp_1d
        tau_eng = interp_1d(eng_rpm_q,
                            jnp.asarray(ENG_RPM_BRK), jnp.asarray(ENG_TRQ_BRK),
                            method="pchip", extrapolation="clip")
        in_shift = (timer > 0.0).astype(jnp.float64)
        tau_eng = tau_eng * u_thr * (1.0 - in_shift)
        T_drive_wheel = tau_eng * ratio * params["eta_drive"]
        T_brake_total = u_brk * (params["T_brake_peak_r"] * (1.0 - params["brake_bias_f"]))
        T_brake_wheel = T_brake_total * jnp.sign(omega_w + 1e-9)
        return jnp.array([T_drive_wheel, T_brake_wheel])

    def _gear_out(self, time, state, *inputs, **params):
        return state.discrete_state[0]

    def _rpm_out(self, time, state, *inputs, **params):
        _, _, omega_w = inputs
        gear_f, _ = state.discrete_state
        gear_int = jnp.round(gear_f).astype(jnp.int32)
        ratio = jnp.asarray(GEAR_RATIOS)[gear_int]
        return jnp.abs(omega_w) * ratio * 60.0 / (2 * jnp.pi)




# ---------- Track + Driver (Part-1 verbatim) ----------
CORNERS = [
    (350.0,  430.0,  520.0,  600.0,  +150.0),
    (800.0,  840.0,  900.0,  950.0,   +40.0),
    (980.0, 1010.0, 1040.0, 1080.0,   -40.0),
    (1300., 1360.,  1440.,  1500.,    -25.0),
    (1900., 1990.,  2200.,  2300.,   +200.0),
]
S_TRACK = 3100.0


def kappa_track(s):
    s = jnp.asarray(s)
    out = jnp.zeros_like(s, dtype=jnp.float64)
    for s_e, s_as, s_ae, s_x, R in CORNERS:
        k_peak = 1.0 / R
        ramp_in  = jnp.clip((s - s_e) / (s_as - s_e), 0.0, 1.0)
        ramp_out = jnp.clip((s_x - s) / (s_x - s_ae), 0.0, 1.0)
        on_arc = ((s >= s_e) & (s <= s_x)).astype(jnp.float64)
        out = out + on_arc * k_peak * jnp.minimum(ramp_in, ramp_out)
    return out


def centerline_xy(s_grid):
    kappa = np.asarray(kappa_track(jnp.asarray(s_grid)))
    ds = np.diff(s_grid, prepend=s_grid[0])
    psi = np.cumsum(kappa * ds)
    X = np.cumsum(np.cos(psi) * ds)
    Y = np.cumsum(np.sin(psi) * ds)
    return X, Y, psi


def mu_eff_at_speed(V, mu=MU_PEAK, CLA_=CLA, m=M_CAR):
    return mu * (1.0 + 0.5 * RHO_AIR * CLA_ * V * V / (m * G_ACC))


def ax_avail_lat(V, kappa_s, mu=MU_PEAK, CLA_=CLA, m=M_CAR):
    a_y = V * V * np.abs(kappa_s)
    a_max = mu_eff_at_speed(V, mu, CLA_, m) * G_ACC
    return np.sqrt(np.maximum(a_max * a_max - a_y * a_y, 0.0))


def ax_engine(V):
    P_pk = ETA_DRIVE * 820_000.0
    return P_pk / (M_CAR * np.maximum(V, 5.0))


def a_drag(V, CDA_=CDA):
    return 0.5 * RHO_AIR * CDA_ * V * V / M_CAR


def build_speed_profile(s_grid, kappa_arr, mu=MU_PEAK, CLA_=CLA, CDA_=CDA, m=M_CAR):
    R_inv = np.abs(kappa_arr)
    R_safe = np.where(R_inv > 1e-6, 1.0 / R_inv, 1e9)
    V_corner = np.sqrt(mu * G_ACC * R_safe)
    for _ in range(3):
        V_corner = np.sqrt(mu_eff_at_speed(V_corner, mu, CLA_, m) * G_ACC * R_safe)
    V_corner = np.minimum(V_corner, 350.0/3.6)
    V_fwd = np.zeros_like(s_grid); V_fwd[0] = min(40.0, V_corner[0])
    for i in range(len(s_grid) - 1):
        ds = s_grid[i+1] - s_grid[i]
        ax_lat = ax_avail_lat(V_fwd[i], kappa_arr[i], mu, CLA_, m)
        ax_eng = ax_engine(V_fwd[i])
        ax = min(ax_lat, ax_eng) - a_drag(V_fwd[i], CDA_)
        V_next = np.sqrt(np.maximum(V_fwd[i]**2 + 2*ax*ds, 1.0))
        V_fwd[i+1] = min(V_next, V_corner[i+1])
    V_bwd = np.zeros_like(s_grid); V_bwd[-1] = V_fwd[-1]
    for i in range(len(s_grid) - 1, 0, -1):
        ds = s_grid[i] - s_grid[i-1]
        ax_b = ax_avail_lat(V_bwd[i], kappa_arr[i], mu, CLA_, m) + a_drag(V_bwd[i], CDA_)
        V_prev = np.sqrt(np.maximum(V_bwd[i]**2 + 2*ax_b*ds, 1.0))
        V_bwd[i-1] = min(V_prev, V_corner[i-1])
    V_qss = np.minimum(V_fwd, V_bwd)
    kernel = np.ones(7) / 7.0
    return np.convolve(V_qss, kernel, mode="same")


# Build reference profile once at the *baseline* setup so the driver is the
# same controller for every setup vector — fair-comparison invariant.
S_GRID = np.linspace(0., S_TRACK, 3101)
KAPPA_ARR = np.asarray(kappa_track(jnp.asarray(S_GRID)))
V_REF_BASELINE = build_speed_profile(S_GRID, KAPPA_ARR)
V_REF_JNP = jnp.asarray(V_REF_BASELINE)
S_GRID_JNP = jnp.asarray(S_GRID)


def lookup_vref(s):
    return jnp.interp(s, S_GRID_JNP, V_REF_JNP, left=V_REF_JNP[0], right=V_REF_JNP[-1])


def lookup_kappa(s):
    return jnp.interp(s, S_GRID_JNP, jnp.asarray(KAPPA_ARR), left=0.0, right=0.0)


DRIVER_K_THR, DRIVER_K_BRK = 0.10, 0.08
DRIVER_DEAD_BAND, DRIVER_LOOKAHEAD = 0.3, 6.0


class Driver(LeafSystem):
    """Part-1 QSS hot-lap driver, unchanged."""

    def __init__(self, name="driver"):
        super().__init__(name=name)
        for nm, val in dict(k_thr=DRIVER_K_THR, k_brk=DRIVER_K_BRK,
                            dead_band=DRIVER_DEAD_BAND, lookahead=DRIVER_LOOKAHEAD).items():
            self.declare_dynamic_parameter(nm, val)
        self.declare_input_port(name="x_car")
        self.declare_output_port(self._compute_u, name="u_ctrl",
                                  requires_inputs=True,
                                  default_value=jnp.array([0.0, 0.5, 0.0]))

    def _compute_u(self, time, state, *inputs, **params):
        x = inputs[0]
        u_long, v_lat, r_yaw, psi, X, Y, s_arc, ww = x
        V_curr = jnp.sqrt(u_long * u_long + v_lat * v_lat)
        s_look = s_arc + params["lookahead"]
        V_target = jnp.minimum(lookup_vref(s_arc), lookup_vref(s_look))
        err = V_target - V_curr
        thr_err = jnp.maximum(err - params["dead_band"], 0.0)
        u_thr = jnp.clip(params["k_thr"] * thr_err, 0.0, 1.0)
        brk_err = jnp.maximum(-err - params["dead_band"], 0.0)
        u_brk = jnp.clip(params["k_brk"] * brk_err, 0.0, 1.0)
        kappa_here = lookup_kappa(s_arc)
        delta = jnp.clip(jnp.arctan(L_WB * kappa_here), -DELTA_MAX_RAD, DELTA_MAX_RAD)
        return jnp.array([delta, u_thr, u_brk])


class MuxControls(LeafSystem):
    def __init__(self, name="mux"):
        super().__init__(name=name)
        self.declare_input_port(name="delta_from_driver")
        self.declare_input_port(name="torques_from_pt")
        self.declare_output_port(lambda t, s, *i, **p: jnp.array([i[0], i[1][0], i[1][1]]),
                                  name="u_to_car", requires_inputs=True,
                                  default_value=jnp.array([0.0, 0.0, 0.0]))


class CarStateSplit(LeafSystem):
    def __init__(self, name="split"):
        super().__init__(name=name)
        self.declare_input_port(name="x_car")
        self.declare_output_port(lambda t, s, *inp, **p: inp[0][7], name="omega_w",
                                  requires_inputs=True, default_value=jnp.array(0.0))


class DemuxDriver(LeafSystem):
    def __init__(self, name="demux_drv"):
        super().__init__(name=name)
        self.declare_input_port(name="u_drv")
        self.declare_output_port(lambda t, s, *i, **p: i[0][0], name="delta",
                                  requires_inputs=True, default_value=jnp.array(0.0))
        self.declare_output_port(lambda t, s, *i, **p: i[0][1], name="u_thr",
                                  requires_inputs=True, default_value=jnp.array(0.0))
        self.declare_output_port(lambda t, s, *i, **p: i[0][2], name="u_brk",
                                  requires_inputs=True, default_value=jnp.array(0.0))


print("Part-1 stack loaded:",
      "BicycleCar, Powertrain, Driver, MuxControls, CarStateSplit, DemuxDriver.")
print(f"Track: {S_TRACK:.0f} m, baseline V_ref top = {V_REF_BASELINE.max()*3.6:.0f} km/h.")


# ## 3. The 8-D setup vector
# 
# A real-team setup sheet has twenty-plus parameters: spring rates, damper bump/rebound at low and high speed, anti-roll bars, ride heights, rake, camber, toe, differential preload and ramp, brake bias, brake pressure, front/rear wing flap angles, suspension geometry choices, tyre pressures, and on the engine side an array of mappings. In this notebook we slice off eight that have substantial lap-time leverage and live cleanly in our single-track quasi-steady-state model.
# 
# | Index | Symbol | Meaning | Units | Low | Baseline | High |
# |---|---|---|---|---|---|---|
# | 0 | $k_{s,f}$ | Front spring rate | N/m | $1.0\times10^5$ | $1.8\times10^5$ | $3.5\times10^5$ |
# | 1 | $k_{s,r}$ | Rear spring rate | N/m | $1.0\times10^5$ | $1.8\times10^5$ | $3.5\times10^5$ |
# | 2 | $c_f$ | Front damping (bump+rebound avg) | N·s/m | $2.0\times10^3$ | $5.0\times10^3$ | $1.0\times10^4$ |
# | 3 | $c_r$ | Rear damping | N·s/m | $2.0\times10^3$ | $5.0\times10^3$ | $1.0\times10^4$ |
# | 4 | $k_{\text{ARB}}$ | Anti-roll-bar stiffness (front - rear) | N·m/rad | $-3.0\times10^4$ | 0 | $+3.0\times10^4$ |
# | 5 | $h_f$ | Front static ride height | mm | $18$ | $25$ | $40$ |
# | 6 | $h_r$ | Rear static ride height | mm | $30$ | $40$ | $60$ |
# | 7 | $\delta_w$ | Front wing flap angle (relative to nominal) | deg | $-6$ | $0$ | $+6$ |
# 
# The chosen bounds are believable engineering envelopes — spring rates run from a "compliant Spa-style" setup to a "rigid Monaco-style" setup, ride heights span the underbody-stall-risk lower bound to the no-downforce upper bound, and the wing flap is bounded by the regulatory flap angle limits.
# 
# ### From setup parameters to lap-time physics
# 
# The single-track quasi-steady-state model from Part 1 does not see spring stiffness, damping, or anti-roll bar as standalone forces — for that we would need pitch / roll / heave dynamics that Part 1 deferred to MuJoCo as the truth model. What the model *does* see is the five parameters $(m, \mu, C_L A_{\text{ref}}, C_D A_{\text{ref}}, \beta_{\text{aero,f}})$ that govern the in-plane dynamics, with $(C_L A, C_D A, \beta_{\text{aero,f}})$ in turn modulated by the dynamic ride heights and wing angle. The bridge is a **quasi-steady-state setup map** — a closed-form parametric mapping from the eight setup knobs onto perturbations of $(C_L A, C_D A, \beta_{\text{aero,f}}, \mu_{\text{eff}})$ around the Part-1 baseline:
# 
# $$
# \begin{aligned}
# C_L A   &= C_L A^{(0)} \cdot \left[1 - 0.012\,(h_f - 25) - 0.008\,(h_r - 40) + 0.020\,\delta_w \right], \\
# C_D A   &= C_D A^{(0)} \cdot \left[1 + 0.003\,(h_f - 25) + 0.002\,(h_r - 40) + 0.015\,|\delta_w| \right], \\
# \beta_{\text{aero,f}} &= 0.45 - 0.0015\,\bigl[(h_f - h_r) - (25 - 40)\bigr] + 0.010\,\delta_w, \\
# \mu_{\text{eff}} &= \mu^{(0)} \cdot \left[1 + 0.005\,\tanh\!\tfrac{k_{s,f}+k_{s,r}-3.6\times10^5}{2.0\times10^5} - 0.003\,\tanh\!\tfrac{c_f+c_r-10^4}{5\times10^3} - 0.001\,\tanh^{2}\!\tfrac{k_{\text{ARB}}}{2\times10^4} \right].
# \end{aligned}
# \tag{1}
# $$
# 
# Each coefficient is hand-picked to match the *direction and order of magnitude* of the published F1-engineering literature (see references — Milliken Ch. 16 on aero-balance vs ride-height, Pacifico 2019 PhD thesis on suspension-frequency vs tyre-grip coupling). The exact magnitudes are not realistic to the third significant figure — they're a stand-in for the multi-megabyte CFD aero-map and FEA suspension-kinematics tables a real team would feed into the same slot. The *shape* of the dependencies is right: lower front ride height adds downforce (negative coefficient), higher mean spring rate adds dynamic-camber control and therefore a small grip benefit (positive tanh), excessive damping adds friction work and costs grip (negative tanh), and the ARB enters squared because *either* direction of imbalance hurts.
# 
# > **What this is, what this isn't.** This is the *parametric form* every commercial LTS uses behind its setup-optimisation UI, with our coefficients in place of the team's proprietary fit. What this is **not** is a high-fidelity aero map — that's the deliverable of Part 3 (fit from synthetic CFD) and Part 5–6 (the actual CFD adjoint). For Part 2 the point is that the mapping is *differentiable*, so the lap-time gradient flows cleanly from setup space all the way through to the chassis ODE.
# 



# ─────────────────────────────────────────────────────────────────────────────
# 8-D setup vector → physics parameters mapping
# ─────────────────────────────────────────────────────────────────────────────
# Setup index labels and baseline / bounds (lap-time-aware, lifted from §3).
SETUP_NAMES = ["k_sf", "k_sr", "c_f", "c_r", "k_ARB", "h_f", "h_r", "delta_w"]
SETUP_UNITS = ["N/m", "N/m", "N·s/m", "N·s/m", "N·m/rad", "mm", "mm", "deg"]
SETUP_BASELINE = jnp.array([1.8e5, 1.8e5, 5.0e3, 5.0e3, 0.0, 25.0, 40.0, 0.0])
SETUP_LOWER    = jnp.array([1.0e5, 1.0e5, 2.0e3, 2.0e3, -3.0e4, 18.0, 30.0, -6.0])
SETUP_UPPER    = jnp.array([3.5e5, 3.5e5, 1.0e4, 1.0e4, +3.0e4, 40.0, 60.0, +6.0])
N_SETUP = len(SETUP_NAMES)


def setup_to_physics(setup):
    """8-D setup vector -> dict of physics parameters consumed by BicycleCar.

    Implements eq. (1) from the markdown above. Pure JAX so jax.grad flows
    through.
    """
    k_sf, k_sr, c_f, c_r, k_arb, h_f, h_r, dw = setup
    h_f_ref, h_r_ref = 25.0, 40.0
    rake_baseline = h_f_ref - h_r_ref  # i.e. -15 mm
    rake_actual   = h_f - h_r

    cla = CLA * (1.0 - 0.012 * (h_f - h_f_ref) - 0.008 * (h_r - h_r_ref) + 0.020 * dw)
    cda = CDA * (1.0 + 0.003 * (h_f - h_f_ref) + 0.002 * (h_r - h_r_ref)
                  + 0.015 * jnp.abs(dw))
    beta_f = BETA_AERO_F - 0.0015 * (rake_actual - rake_baseline) + 0.010 * dw
    # Mechanical-grip perturbation (small, soft tanh)
    spring_term = jnp.tanh((k_sf + k_sr - 3.6e5) / 2.0e5)
    damp_term   = jnp.tanh((c_f + c_r - 1.0e4)  / 5.0e3)
    arb_term    = jnp.tanh(k_arb / 2.0e4) ** 2
    mu = MU_PEAK * (1.0 + 0.005 * spring_term - 0.003 * damp_term - 0.001 * arb_term)
    # Mass: nominally fixed in the bicycle; we keep it constant here.
    return {
        "m":      jnp.asarray(M_CAR),
        "mu":     mu,
        "CLA":    cla,
        "CDA":    cda,
        "beta_f": beta_f,
    }


# Sanity check the mapping at baseline + a few perturbations.
phys0 = setup_to_physics(SETUP_BASELINE)
print(f"Baseline setup -> physics:")
for k, v in phys0.items():
    print(f"  {k:8s} = {float(v):+.4f}")

# Bias the front wing up by 5 deg -> CLA should rise ~10%, beta_f should shift forward
phys1 = setup_to_physics(SETUP_BASELINE.at[7].set(5.0))
print(f"\nFront wing +5 deg perturbation:")
print(f"  CLA: {float(phys0['CLA']):.3f} -> {float(phys1['CLA']):.3f} "
      f"(+{100*(float(phys1['CLA'])/float(phys0['CLA']) - 1):.1f}%)")
print(f"  beta_f: {float(phys0['beta_f']):.3f} -> {float(phys1['beta_f']):.3f} "
      f"(+{float(phys1['beta_f']) - float(phys0['beta_f']):+.3f})")


# ## 4. Wrapping the lap simulator as `lap_time(setup) → float`
# 
# The canonical pattern (cribbed from [`pid_tuning.ipynb`](./pid_tuning.ipynb)) is to build the closed-loop diagram **once**, then on every call write the current setup into the relevant `LeafContext` via `context.with_parameters({...})`, simulate, and read off a scalar from a `LeafContext.continuous_state` slot. The result is a JAX-traceable closure $f : \mathbb{R}^8 \to \mathbb{R}$ that `jax.grad` can differentiate in one backward pass.
# 
# Three design points are worth dwelling on:
# 
# 1. **Bake the setup as a single `jnp.ndarray`, not eight Python floats.** If you write `context.with_parameter("m", float(setup[0]))` inside an outer loop over setup values, every distinct value triggers a fresh JIT trace — the bug we filed as a follow-up finding after the bouncing-ball event-time tutorial. The remediation is to keep `setup` as a JAX array end-to-end, so the trace cache keys on its abstract `(shape=(8,), dtype=float64)` signature and is reused across calls.
# 
# 2. **Recorded signals are not allowed with autodiff.** `simulate(..., enable_autodiff=True)` refuses any `recorded_signals=` argument — the trajectory storage is `vmap`-unfriendly and would dominate the autodiff trace. The canonical workaround (from `pid_tuning.ipynb`) is to **integrate the cost as part of the diagram itself** and read the final value from the `Integrator`'s continuous-state slot at simulation end. We adopt the same pattern: a tiny `LapTimeAccumulator` `LeafSystem` integrates a smooth indicator of "lap not yet finished" over time. The integral, evaluated at $t = T_{\text{end}}$, is exactly the (smoothed) lap-completion time:
# $$
# \text{lap} \;\approx\; \int_0^{T_{\text{end}}} \tfrac{1}{2}\bigl(1 - \tanh\bigl((s(t) - S_{\text{track}})/\sigma\bigr)\bigr)\,dt.
# \tag{2}
# $$
# With $\sigma = 0.5$ m the indicator transitions from $1$ to $0$ over an arc-length window of $\sim 1$ m, so the lap-time read is accurate to milliseconds while staying everywhere smooth in the setup vector.
# 
# 3. **Final state, not output time-series.** We read `results.context[LAP_BLK.system_id].continuous_state` — a single scalar — and return it from `forward(setup)`. Everything is JIT-able, traceable, and `jax.grad`-able by construction.
# 
# > **Note.** The Casanova thesis (Ch. 5) calls this the "isochrone formulation"; Limebeer & Perantoni (2015) call it the "Mayer terminal cost". Reformulating "time at which $s$ reaches $S$" as "integral of a smoothed indicator until $T_{\text{end}}$" is the standard differentiable racing-line trick.
# 



# ---------- The LapTimeAccumulator block (the differentiable lap-time readout) ----------
SIGMA_LAP = 0.5  # arc-length smoothing window for the indicator [m]


class LapTimeAccumulator(LeafSystem):
    """One continuous state: the integrated 'not-yet-finished' indicator.

    PARAMETERS: S_track (so a future user can sweep track length)
    STATE:      one continuous state (the accumulated lap time)
    INPUTS:     port 0 = scalar arc length s(t) [m]
    OUTPUTS:    port 0 = the current accumulated lap time [s]

    Integrating eq. (2) of the markdown:
      dt_lap/dt = 0.5 * (1 - tanh((s - S_track) / sigma))
    At t = T_end (>> the real lap time), the state holds the smoothed lap-completion
    time. With sigma small (~0.5 m), this is accurate to a few ms.
    """

    def __init__(self, S_track=S_TRACK, sigma=SIGMA_LAP, name="laptime"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("S_track", float(S_track))
        self.declare_dynamic_parameter("sigma",   float(sigma))
        self.declare_input_port(name="s_arc")
        self.declare_continuous_state(default_value=jnp.zeros(1), ode=self._ode)
        self.declare_continuous_state_output(name="lap_time")

    def _ode(self, time, state, *inputs, **params):
        (s_arc,) = inputs
        # 0.5 * (1 - tanh((s - S)/sigma)) is 1 while s < S, 0 after.
        indicator = 0.5 * (1.0 - jnp.tanh((s_arc - params["S_track"]) / params["sigma"]))
        return jnp.array([indicator])


# Adapter to pull s_arc (slot 6) out of the 8-state car output
class CarArcLength(LeafSystem):
    """Pull s_arc = state[6] out of the full car state."""

    def __init__(self, name="s_split"):
        super().__init__(name=name)
        self.declare_input_port(name="x_car")
        self.declare_output_port(lambda t, s, *inp, **p: inp[0][6], name="s_arc",
                                  requires_inputs=True, default_value=jnp.array(0.0))


print("LapTimeAccumulator + CarArcLength blocks declared.")




# ---------- Build the closed-loop lap diagram, once ----------
U0    = 60.0
WW0   = U0 / R_WHEEL
X0_CAR = jnp.array([U0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, WW0])
T_END = 15.0  # NB: shortened from a full lap (60 s) to a single straight+corner so the autodiff-traced simulator compiles + runs tractably in the notebook. The publication-mode results below use T_END=60 (full lap) and are loaded from media/f1_part_2_publication.npz.  # cut from 90 s for autodiff-tractable adjoint trace; covers ~1 lap sector   # horizon: longer than the QSS prediction so the smooth-lap-time term saturates


def build_lap_diagram():
    """Same architecture as Part 1's build_lap_diagram, plus LapTimeAccumulator."""
    b = DiagramBuilder()
    car = b.add(BicycleCar(x0=X0_CAR, name="car"))
    drv = b.add(Driver(name="driver"))
    pt  = b.add(Powertrain(name="powertrain"))
    mux = b.add(MuxControls(name="mux"))
    splt = b.add(CarStateSplit(name="split"))
    demux = b.add(DemuxDriver(name="demux_drv"))
    sarc = b.add(CarArcLength(name="s_split"))
    lap  = b.add(LapTimeAccumulator(name="laptime"))
    # Part-1 wiring
    b.connect(car.output_ports[0], drv.input_ports[0])
    b.connect(drv.output_ports[0], demux.input_ports[0])
    b.connect(demux.output_ports[1], pt.input_ports[0])
    b.connect(demux.output_ports[2], pt.input_ports[1])
    b.connect(car.output_ports[0], splt.input_ports[0])
    b.connect(splt.output_ports[0], pt.input_ports[2])
    b.connect(demux.output_ports[0], mux.input_ports[0])
    b.connect(pt.output_ports[0],    mux.input_ports[1])
    b.connect(mux.output_ports[0],   car.input_ports[0])
    # Part-2 additions: route arc length into the LapTimeAccumulator
    b.connect(car.output_ports[0], sarc.input_ports[0])
    b.connect(sarc.output_ports[0], lap.input_ports[0])
    diag = b.build()
    return diag, car, drv, pt, lap


DIAG, CAR_BLK, DRV_BLK, PT_BLK, LAP_BLK = build_lap_diagram()
CTX0 = DIAG.create_context()
print("Lap diagram built once. Blocks:", [b.name for b in DIAG.nodes])
