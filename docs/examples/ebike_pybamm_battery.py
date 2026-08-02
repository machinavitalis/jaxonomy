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
3. Extract the **pseudo-OCV** curve from a slow DFN discharge (the OCV is the
   *dominant* term in any ECM — leaving it hand-written while calling the model
   "calibrated" would be theater), then fit the e-bike's **2-RC ECM** (plus its
   initial SOC) to the DFN terminal voltage, validate on a **held-out** profile
   the fit never saw, and check the fitted values against the group-level
   defaults the vehicle model (Part 1) actually ships — the loop is closed, not
   just gestured at.

**Pack vs cell**: the vehicle pack is 13S3P of ~5 Ah cells; Part 1 lumps each
3P group into one 15 Ah "cell", so per-cell current here = pack current / 3,
and Part 1's group-level parameters = (fitted per-cell R)/3, C*3. All mV
figures below are per cell — multiply by 13 for pack-level error.

This mirrors the Jaxonomy battery tutorial series (part 1 ECM, part 2 parameter
ID) with a first-principles ground truth.
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
#
# Scaled to the Class-1 vehicle: cruise draws ~290 W -> ~6 A pack -> 2 A/cell
# (0.4C on 5 Ah); climb peaks ~700 W -> ~14 A pack -> ~4.8 A/cell (~1C).
# The excitation is designed for *identifiability*, which the vehicle's own
# drive cycle is not: short pulses probe the fast RC branch, long pulses the
# slow one, a charge segment breaks the discharge-only degeneracy (the Part-1
# drivetrain has no regen path -- this is a bench profile, not vehicle
# telemetry), and a full-rest relaxation tail lets both time constants show
# themselves with no forcing.
def ebike_current(t):
    t = np.asarray(t, dtype=float)
    I = 2.0 + 0.002 * t                                # cruise baseline, slow drift
    I = I + 2.8 * (np.sin(2 * np.pi * t / 90.0) > 0.4)  # long hill pulses (~30 s)
    I = I + 1.2 * (np.sin(2 * np.pi * t / 16.0) > 0.7)  # short pulses (~2.4 s)
    I = I - 3.0 * ((t > 230) & (t < 255))               # charge segment
    I = np.where(t >= 300.0, 0.0, I)                    # relaxation tail (rest)
    return I if I.shape else float(I)


def ebike_current_holdout(t):
    """A different profile in the same regime (other periods, phases and
    amplitudes, same SOC window) for held-out validation of the fit."""
    t = np.asarray(t, dtype=float)
    I = 2.4 + 0.001 * t
    I = I + 2.2 * (np.sin(2 * np.pi * (t + 30.0) / 70.0) > 0.3)
    I = I + 0.9 * (np.sin(2 * np.pi * t / 23.0) > 0.6)
    I = I - 2.0 * ((t > 180) & (t < 200))
    I = np.where(t >= 330.0, 0.0, I)
    return I if I.shape else float(I)


T_END = 380.0


def run_pybamm(model_ctor, label, current_fn=None):
    """Solve a PyBaMM model under a current profile; return (t, V, T_degC, wall)."""
    current_fn = current_fn or ebike_current
    model = model_ctor(options={"thermal": "lumped"})
    param = pybamm.ParameterValues("Chen2020")
    t_grid = np.linspace(0, T_END, 400)
    param["Current function [A]"] = pybamm.Interpolant(t_grid, current_fn(t_grid), pybamm.t)
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


# --- pseudo-OCV from the DFN itself ------------------------------------------
def extract_pseudo_ocv(c_rate=0.05):
    """C/20-class constant-current discharge of the DFN -> V(SOC) interpolant.

    The OCV curve is the dominant term of any ECM; extracting it from the same
    first-principles model the dynamics are fit to is what makes "calibrated to
    the DFN" a true sentence. (A hand-written cubic used here previously was
    +221 mV wrong at SOC 0.1 -- invisible in a high-SOC test window, wrong
    everywhere else.) At C/20-class currents the overpotential is a few mV, so
    the discharge curve is a good pseudo-OCV; a bench measurement would use
    GITT or C/50 with rest steps.
    """
    param = pybamm.ParameterValues("Chen2020")
    cap = param["Nominal cell capacity [A.h]"]
    I = c_rate * cap
    t_end = 0.98 / c_rate * 3600.0     # ~98% depth of discharge
    param["Current function [A]"] = I
    model = pybamm.lithium_ion.SPM()   # equilibrium curve: SPM is exact enough
    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve([0, t_end])
    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    soc = 1.0 - I * t / (cap * 3600.0)  # SOC=1 at the fully-charged start
    order = np.argsort(soc)
    return interp1d(soc[order], V[order], bounds_error=False,
                    fill_value=(float(V[np.argmin(soc)]), float(V[np.argmax(soc)])))


# --- the e-bike's 2-RC ECM (transparent re-implementation for calibration) ---
def ecm_voltage(params, t, I_of_t, ocv_fn, cap_Ah=5.0):
    """Integrate the 2-RC ECM (R0,R1,C1,R2,C2,soc0) under current I(t)
    [A, + = discharge]. Returns terminal voltage on the grid ``t``. The initial
    SOC is a *parameter*: aligning it by eye against a hand-written OCV is a
    silent error source, so the fit owns it."""
    R0, R1, C1, R2, C2, soc0 = params
    soc, v1, v2 = soc0, 0.0, 0.0
    V = np.empty_like(t)
    for k, tk in enumerate(t):
        I = float(I_of_t(tk))
        V[k] = float(ocv_fn(soc)) - R0 * I - v1 - v2
        if k + 1 < len(t):
            dt = t[k + 1] - tk
            soc -= I * dt / (cap_Ah * 3600.0)
            v1 += dt * (I / C1 - v1 / (R1 * C1))
            v2 += dt * (I / C2 - v2 / (R2 * C2))
    return V


def calibrate_ecm(t, V_dfn, I_of_t, ocv_fn, soc0_guess=1.0, cap_Ah=5.0):
    """Least-squares fit of the 2-RC ECM parameters (+ initial SOC) to the DFN
    terminal voltage. Returns (params, identifiability report string).

    The sampling grid matters as much as the excitation: on a ~1 s grid the
    fast charge-transfer branch (tau ~ 0.5 s) is invisible and the fit
    silently collapses to 1-RC with both time constants equal -- run the
    notebook's identifiability exercise to see it happen. The 0.25 s grid
    used by ``main`` resolves it.
    """
    p0 = np.array([0.02, 0.01, 2000.0, 0.005, 500.0, soc0_guess])  # R0,R1,C1,R2,C2,soc0
    lb = np.array([1e-3, 1e-3, 100.0, 1e-4, 50.0, 0.5])
    ub = np.array([0.2, 0.2, 5e4, 0.2, 5e4, 1.0])

    def resid(p):
        return ecm_voltage(p, t, I_of_t, ocv_fn, cap_Ah=cap_Ah) - V_dfn

    sol = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
    p = sol.x

    # Identifiability report: a 2-RC fit can silently collapse to 1-RC (equal
    # time constants) or pin a parameter at its bound. Say so, loudly.
    tau1, tau2 = p[1] * p[2], p[3] * p[4]
    notes = []
    for i, nm in ((0, "R0"), (1, "R1"), (2, "C1"), (3, "R2"), (4, "C2")):
        if p[i] <= lb[i] * 1.01 or p[i] >= ub[i] * 0.99:
            notes.append(f"{nm} pinned at its bound -- that branch is not identified")
    if max(tau1, tau2) / max(min(tau1, tau2), 1e-9) < 3.0:
        notes.append(f"tau1 ~ tau2 ({tau1:.1f} vs {tau2:.1f} s) -- branches degenerate,"
                     " effectively a 1-RC model")
    report = "; ".join(notes) if notes else "both RC branches identified (distinct taus, no bound hits)"
    return p, report


def rmse_mV(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) * 1000.0)


def main():
    print("=" * 70)
    print("  E-BIKE BATTERY: PyBaMM DFN (truth) vs SPMe ROM vs 2-RC ECM")
    print("=" * 70)

    print("\n[1] High-fidelity electrochemical models under the e-bike current:")
    t_dfn, V_dfn, T_dfn, wall_dfn = run_pybamm(pybamm.lithium_ion.DFN, "DFN")
    t_spm, V_spm, T_spm, wall_spm = run_pybamm(pybamm.lithium_ion.SPMe, "SPMe")

    # Common grid for comparison: 0.25 s sampling. Coarser grids (~1 s) cannot
    # see the fast charge-transfer branch and the 2-RC fit degenerates.
    tg = np.linspace(0, T_END, 1520)
    Vd = interp1d(t_dfn, V_dfn, bounds_error=False, fill_value="extrapolate")(tg)
    Vs = interp1d(t_spm, V_spm, bounds_error=False, fill_value="extrapolate")(tg)
    I_of_t = ebike_current  # callable

    print("\n[2] Pseudo-OCV from a slow DFN-family discharge (the ECM's dominant term):")
    ocv_fn = extract_pseudo_ocv()
    print(f"  OCV(1.0) = {float(ocv_fn(1.0)):.3f} V, OCV(0.5) = {float(ocv_fn(0.5)):.3f} V, "
          f"OCV(0.1) = {float(ocv_fn(0.1)):.3f} V")

    print("\n[3] Reduced-order comparison against the DFN truth (all mV per cell;")
    print("    x13 for the pack):")
    print(f"  SPMe (physics ROM)          : RMSE {rmse_mV(Vs, Vd):6.2f} mV   "
          f"({wall_dfn / max(wall_spm, 1e-6):.1f}x faster than DFN, this run/machine)")

    # Baseline: what an engineer writes down WITHOUT the DFN -- datasheet-style
    # RC values and a hand-written cubic OCV, initial SOC aligned by eye. This
    # is the "plausible guess" the calibration is measured against.
    def ocv_handwritten(s):
        s = np.clip(s, 0.01, 0.99)
        return 3.3 + 1.2 * s - 0.5 * s**2 + 0.15 * s**3

    p_guess = np.array([0.015, 0.010, 2000.0, 0.012, 10000.0, 0.95])
    V_ecm_guess = ecm_voltage(p_guess, tg, I_of_t, ocv_handwritten)
    print(f"  2-RC ECM (hand-written guess): RMSE {rmse_mV(V_ecm_guess, Vd):6.2f} mV"
          f"   (datasheet-style params + cubic OCV)")

    p_cal, ident = calibrate_ecm(tg, Vd, I_of_t, ocv_fn)
    V_ecm_cal = ecm_voltage(p_cal, tg, I_of_t, ocv_fn)
    print(f"  2-RC ECM (fit to DFN)       : RMSE {rmse_mV(V_ecm_cal, Vd):6.2f} mV  (in-sample)")
    print(f"    fitted R0={p_cal[0]*1000:.2f} mOhm, R1={p_cal[1]*1000:.2f} mOhm, "
          f"C1={p_cal[2]:.0f} F (tau1={p_cal[1]*p_cal[2]:.1f} s),")
    print(f"           R2={p_cal[3]*1000:.2f} mOhm, C2={p_cal[4]:.0f} F "
          f"(tau2={p_cal[3]*p_cal[4]:.1f} s), soc0={p_cal[5]:.4f}")
    print(f"    identifiability: {ident}")

    print("\n[4] Held-out validation (different pulse pattern, same regime):")
    t_h, V_h, _, _ = run_pybamm(pybamm.lithium_ion.DFN, "DFN-h", current_fn=ebike_current_holdout)
    Vh = interp1d(t_h, V_h, bounds_error=False, fill_value="extrapolate")(tg)
    V_ecm_cal_h = ecm_voltage(p_cal, tg, ebike_current_holdout, ocv_fn)
    V_ecm_guess_h = ecm_voltage(p_guess, tg, ebike_current_holdout, ocv_handwritten)
    print(f"  2-RC ECM (fit to DFN)        : RMSE {rmse_mV(V_ecm_cal_h, Vh):6.2f} mV  (held-out)")
    print(f"  2-RC ECM (hand-written guess): RMSE {rmse_mV(V_ecm_guess_h, Vh):6.2f} mV  (held-out)")

    # Part 1's shipped group-level values, for the loop-closure check below
    from ebike_hybrid_simulation import HighFidelityBatteryCellECM as _Cell
    import inspect
    sig = inspect.signature(_Cell.__init__).parameters
    part1_group = {k: sig[k].default for k in ("R00", "R10", "C10", "R20", "C20")}

    print("\n[5] Loop-closure check against the vehicle model (Part 1):")
    print("    Part 1 ships group-level values derived from this fit (group R =")
    print("    cell R / 3, group C = 3 * cell C for the 3P pack): the diffusion")
    print("    branch maps to R10/C10 and the fast charge-transfer branch to")
    print("    R20/C20. Consistency (25% tolerance for fit-to-fit variation):")
    checks = [("R00", part1_group["R00"], p_cal[0] / 3),
              ("R10", part1_group["R10"], p_cal[1] / 3),
              ("C10", part1_group["C10"], 3 * p_cal[2]),
              ("R20", part1_group["R20"], p_cal[3] / 3),
              ("C20", part1_group["C20"], 3 * p_cal[4])]
    for nm, shipped, fitted in checks:
        rel = abs(shipped - fitted) / abs(fitted)
        flag = "ok" if rel < 0.25 else "DRIFTED -- update Part 1's defaults"
        print(f"    {nm}: shipped {shipped:.4g} vs fit {fitted:.4g}  ({rel*100:.0f}% off) [{flag}]")

    # Validity window, stated instead of implied
    cap = 5.0
    dq = np.trapezoid(ebike_current(tg), tg) / 3600.0
    print(f"\n  Validity window of this calibration: SOC {p_cal[5]:.2f} -> "
          f"{p_cal[5] - dq / cap:.2f}, 25 degC, <= ~1C. Outside it (low SOC, cold,")
    print("  high C-rate) the RC parameters change and the fit must be redone.")

    # --- plot ---
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    axs[0].plot(tg, I_of_t(tg), color="tab:gray")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("cell current (A, + = discharge)")
    axs[0].set_title("E-bike-representative drive current")

    axs[1].plot(tg, Vd, color="k", lw=2.2, label="DFN (truth)")
    axs[1].plot(tg, Vs, color="tab:blue", lw=1.4, ls="--", label=f"SPMe ROM ({rmse_mV(Vs,Vd):.1f} mV)")
    axs[1].plot(tg, V_ecm_guess, color="tab:orange", lw=1.2, ls=":", label=f"ECM hand-written guess ({rmse_mV(V_ecm_guess,Vd):.0f} mV)")
    axs[1].plot(tg, V_ecm_cal, color="tab:green", lw=1.4, label=f"ECM fit-to-DFN ({rmse_mV(V_ecm_cal,Vd):.1f} mV)")
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("terminal voltage (V)")
    axs[1].set_title("Fidelity ladder: voltage under load (per cell)"); axs[1].legend(fontsize=8)

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
    print("the 2-RC ECM with a DFN-extracted OCV, fit to the DFN and validated on")
    print("a held-out profile, earns its place in the vehicle model -- whose")
    print("shipped defaults are checked against this very fit above. Valid in the")
    print("stated SOC/temperature/C-rate window; not a general battery model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
