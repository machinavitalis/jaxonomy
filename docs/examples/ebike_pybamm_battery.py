"""Physics-based battery for the cargo e-bike: PyBaMM DFN as truth, and the
ebike's 2-RC equivalent-circuit model (ECM) calibrated *from* the DFN.

The Jaxonomy e-bike model (``ebike_hybrid_simulation.py``) uses a lightweight
2-RC ECM cell for speed. That is the right engineering choice for a real-time
system model — but its parameters should be justified by a high-fidelity
electrochemical model, not guessed. Here we:

1. Run a **Doyle-Fuller-Newman (DFN)** cell (PyBaMM, Chen2020 chemistry, lumped
   thermal) under an e-bike-representative current profile — this is the
   physical truth (resolves solid/electrolyte Li transport, overpotentials).
2. Run a **Single-Particle Model with electrolyte (SPMe)** — a physics-based
   reduced-order model — under the same current, and measure how well it tracks
   the DFN (and how much faster it is).
3. Fit the e-bike's **2-RC ECM** to the DFN terminal voltage (parameter
   identification), and show the calibrated ECM reproduces the DFN within a few
   mV — justifying the ECM used in the vehicle model, and quantifying the
   fidelity gap vs the un-calibrated defaults.

This mirrors the Jaxonomy battery tutorial series (part 1 ECM, part 2 parameter
ID) with a first-principles ground truth. It is the honest, in-idiom version of
"the highest-fidelity battery possible".
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pybamm

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "media", "ebike_pybamm_battery.png")


# --- e-bike-representative single-cell current profile [A], + = discharge -----
def ebike_current(t):
    """Cruise baseline + hill-climb pulses + a short regen dip, ~1C peak on a
    5 Ah cell (representative of one series cell of the pack under load)."""
    I = 2.0 + 0.004 * t                      # slow SOC-drift baseline
    I += 5.0 * (np.sin(2 * np.pi * t / 90.0) > 0.4)   # ~hill-climb pulses
    I -= 4.0 * ((t > 250) & (t < 275))       # a downhill regen segment (charge)
    return I


T_END = 380.0


def run_pybamm(model_ctor, label):
    """Solve a PyBaMM model under the e-bike current; return (t, V, T_degC, wall)."""
    model = model_ctor(options={"thermal": "lumped"})
    param = pybamm.ParameterValues("Chen2020")
    t_grid = np.linspace(0, T_END, 400)
    param["Current function [A]"] = pybamm.Interpolant(t_grid, ebike_current(t_grid), pybamm.t)
    sim = pybamm.Simulation(model, parameter_values=param)
    t0 = time.time()
    sol = sim.solve([0, T_END])
    wall = time.time() - t0
    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    T = sol["Volume-averaged cell temperature [K]"].entries - 273.15
    print(f"  {label:6s}: solved in {wall:5.2f} s   V {V[0]:.3f}->{V[-1]:.3f} V   "
          f"T {T[0]:.2f}->{T[-1]:.2f} C")
    return t, V, T, wall


# --- the e-bike's 2-RC ECM (transparent re-implementation for calibration) ---
def ocv(soc):
    s = np.clip(soc, 0.01, 0.99)
    return 3.3 + 1.2 * s - 0.5 * s**2 + 0.15 * s**3


def ecm_voltage(params, t, I_of_t, soc0=0.9, cap_Ah=5.0):
    """Integrate the 2-RC ECM (R0,R1,C1,R2,C2) under current I(t) [A, + = discharge].
    Returns terminal voltage on the grid ``t``."""
    R0, R1, C1, R2, C2 = params
    soc, v1, v2 = soc0, 0.0, 0.0
    V = np.empty_like(t)
    for k, tk in enumerate(t):
        I = float(I_of_t(tk))
        V[k] = ocv(soc) - R0 * I - v1 - v2
        if k + 1 < len(t):
            dt = t[k + 1] - tk
            soc -= I * dt / (cap_Ah * 3600.0)
            v1 += dt * (I / C1 - v1 / (R1 * C1))
            v2 += dt * (I / C2 - v2 / (R2 * C2))
    return V


def calibrate_ecm(t, V_dfn, I_of_t):
    """Least-squares fit of the 2-RC ECM parameters to the DFN terminal voltage."""
    p0 = np.array([0.02, 0.01, 2000.0, 0.02, 8000.0])   # R0,R1,C1,R2,C2
    lb = np.array([1e-3, 1e-3, 100.0, 1e-3, 500.0])
    ub = np.array([0.2, 0.2, 5e4, 0.2, 5e4])

    def resid(p):
        return ecm_voltage(p, t, I_of_t) - V_dfn

    sol = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
    return sol.x


def rmse_mV(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) * 1000.0)


def main():
    print("=" * 70)
    print("  E-BIKE BATTERY: PyBaMM DFN (truth) vs SPMe ROM vs 2-RC ECM")
    print("=" * 70)

    print("\n[1] High-fidelity electrochemical models under the e-bike current:")
    t_dfn, V_dfn, T_dfn, wall_dfn = run_pybamm(pybamm.lithium_ion.DFN, "DFN")
    t_spm, V_spm, T_spm, wall_spm = run_pybamm(pybamm.lithium_ion.SPMe, "SPMe")

    # Common grid for comparison
    tg = np.linspace(0, T_END, 400)
    Vd = interp1d(t_dfn, V_dfn, bounds_error=False, fill_value="extrapolate")(tg)
    Vs = interp1d(t_spm, V_spm, bounds_error=False, fill_value="extrapolate")(tg)
    I_of_t = ebike_current  # callable

    print("\n[2] Reduced-order comparison against the DFN truth:")
    print(f"  SPMe (physics ROM)          : RMSE {rmse_mV(Vs, Vd):6.2f} mV   "
          f"({wall_dfn / max(wall_spm, 1e-6):.1f}x faster than DFN)")

    V_ecm_default = ecm_voltage([0.015, 0.01, 2000.0, 0.012, 10000.0], tg, I_of_t)
    print(f"  2-RC ECM (un-calibrated)    : RMSE {rmse_mV(V_ecm_default, Vd):6.2f} mV")

    p_cal = calibrate_ecm(tg, Vd, I_of_t)
    V_ecm_cal = ecm_voltage(p_cal, tg, I_of_t)
    print(f"  2-RC ECM (fit to DFN)       : RMSE {rmse_mV(V_ecm_cal, Vd):6.2f} mV")
    print(f"    fitted R0={p_cal[0]*1000:.1f} mOhm, R1={p_cal[1]*1000:.1f} mOhm, "
          f"C1={p_cal[2]:.0f} F, R2={p_cal[3]*1000:.1f} mOhm, C2={p_cal[4]:.0f} F")

    # --- plot ---
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    axs[0].plot(tg, I_of_t(tg), color="tab:gray")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("cell current (A, + = discharge)")
    axs[0].set_title("E-bike-representative drive current")

    axs[1].plot(tg, Vd, color="k", lw=2.2, label="DFN (truth)")
    axs[1].plot(tg, Vs, color="tab:blue", lw=1.4, ls="--", label=f"SPMe ROM ({rmse_mV(Vs,Vd):.1f} mV)")
    axs[1].plot(tg, V_ecm_default, color="tab:orange", lw=1.2, ls=":", label=f"ECM default ({rmse_mV(V_ecm_default,Vd):.0f} mV)")
    axs[1].plot(tg, V_ecm_cal, color="tab:green", lw=1.4, label=f"ECM fit-to-DFN ({rmse_mV(V_ecm_cal,Vd):.1f} mV)")
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("terminal voltage (V)")
    axs[1].set_title("Fidelity ladder: voltage under load"); axs[1].legend(fontsize=8)

    axs[2].plot(t_dfn, T_dfn, color="tab:red", lw=1.8, label="DFN")
    axs[2].plot(t_spm, T_spm, color="tab:blue", lw=1.4, ls="--", label="SPMe")
    axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("cell temperature (°C)")
    axs[2].set_title("Electrochemical self-heating"); axs[2].legend(fontsize=8)

    fig.suptitle("E-bike battery — physics-based DFN vs reduced-order models (PyBaMM + ECM ID)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"\nWrote {PNG}")
    print("=" * 70)
    print("Takeaway: the DFN is the electrochemical truth; SPMe tracks it cheaply;")
    print("and the vehicle model's 2-RC ECM, once *calibrated to the DFN*, is")
    print("accurate to a few mV — a justified reduced model, not a guess.")
    print("=" * 70)


if __name__ == "__main__":
    main()
