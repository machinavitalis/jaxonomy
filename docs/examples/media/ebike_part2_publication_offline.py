#!/usr/bin/env python3
"""Offline publication run for ``ebike_part2_optimization.ipynb``.

Runs every heavy optimisation / sensitivity study for Part 2 at full fidelity
and writes a compact checkpoint ``media/ebike_part2_publication.npz``. The
notebook loads the NPZ in < 1 s so the reader's experience stays fast; a
single rollout of the true multi-domain DAE is ~50 s of CPU and this script
runs ~90 of them (plus a rebuild per distinct drag area), so it takes 1.5-2 h
on a developer machine.

Set the environment variable ``EBIKE_P2_SMOKE=1`` to run a tiny smoke version
(coarse grid, few optimiser evals, small Sobol design) in a few minutes,
purely to validate the code path before the full run.

Design decisions (each is a lesson the notebook teaches):

* **Per-distance objective.** The optimisation minimises battery energy to
  cover the FIRST 100 m of the route (flat approach + the 6% climb), not
  energy over a fixed time. A fixed-time objective silently rewards riding
  slower (less distance, less climb = less work), which once inflated a
  "37% battery saving" that was ~7% per metre. The route grade is a function
  of position (see Part 1), so every design climbs the same hill.
* **The speed floor genuinely binds.** An exterior penalty converges from the
  infeasible side with equilibrium shortfall s* ~ dJ/dcap / (2*lambda*dv/dcap);
  lambda is sized so s* < 0.05 km/h, and the checkpoint stores an
  optimum-vs-lambda curve (computed from the response grid, no extra sims)
  so the notebook can SHOW the infeasible-side convergence instead of
  hand-waving "lambda large enough".
* **The noise floor is measured, not asserted.** The solver is deterministic,
  so repeat evals are identical; the real "noise" for a derivative-free
  optimiser is discretisation error. We rerun a few points at tightened
  tolerances and use the J-shift as the resolution floor; optimiser
  tolerances are set ABOVE it and the values are stored.
* **Hardware sensitivity is conditioned on the control policy.** Mixing the
  assist cap (a control knob spanning most of the variance) into a "which
  hardware matters" Sobol study buries the hardware indices below the
  surrogate's own error. The Sobol design here fixes the cap at its nominal
  value and ranks cargo mass / drag area / gearing only, with bootstrap
  confidence intervals so a ranking is only claimed when the intervals
  separate.

Run from docs/examples:
    python media/ebike_part2_publication_offline.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxonomy
from jaxonomy.simulation import SimulatorOptions
from jaxonomy.library.rom import fit_pce
from ebike_hybrid_simulation import make_ebike_diagram, EbikeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "ebike_part2_publication.npz")
SMOKE = os.environ.get("EBIKE_P2_SMOKE", "0") == "1"

# --- shared problem definition (matches ebike_trajectory_optimization.py) ----
TF = 45.0                    # rollout horizon; long enough for every design to cover D_REF
D_REF = 100.0                # route segment for the objective [m]
V_TARGET_KMH = 12.0          # mean-speed floor over the D_REF segment
LAMBDA_SPEED = 2.0e4         # J/(km/h)^2; sized so the penalty equilibrium
                             # shortfall is < 0.05 km/h (see lambda sweep)
CAP_BOUNDS = (6.0, 20.0)
CDA_BOUNDS = (0.62, 0.95)    # design range: faired cargo pod ... upright
BASE_CDA = 0.80
NOMINAL_CAP = 12.0

if SMOKE:                    # tiny problem, same code path
    TF, D_REF, V_TARGET_KMH = 20.0, 40.0, 9.0
_OPTS = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6,
                         buffer_length=260000)

# cache of (rollout closures) keyed by rounded CdA so the grid + optimiser reuse
# a compiled diagram whenever the drag area repeats.
_PLANT_CACHE: dict[float, object] = {}


def _metrics_from(res):
    """Per-distance metrics from recorded traces: energy and time at the point
    the vehicle has covered D_REF metres (linear interpolation on the traces)."""
    o = res.outputs
    d = np.asarray(o["d"]).squeeze()
    E = np.asarray(o["E"]).squeeze()
    H = np.asarray(o["H"]).squeeze()
    t = np.asarray(res.time).squeeze()
    if d[-1] < D_REF:      # never covered the segment: infeasible marker
        return dict(E_D=np.inf, t_D=np.inf, v_D=0.0, H_D=np.inf,
                    E_tf=float(E[-1]), d_tf=float(d[-1]))
    E_D = float(np.interp(D_REF, d, E))
    t_D = float(np.interp(D_REF, d, t))
    H_D = float(np.interp(D_REF, d, H))
    return dict(E_D=E_D, t_D=t_D, v_D=D_REF / t_D * 3.6, H_D=H_D,
                E_tf=float(E[-1]), d_tf=float(d[-1]))


def plant_for(CdA, opts=_OPTS):
    """Return ``rollout(cap) -> metrics dict`` for a diagram built at drag
    area ``CdA``. The assist-torque cap is a runtime dynamic parameter, so
    sweeping caps on the returned closure needs no rebuild."""
    key = (round(float(CdA), 4), float(opts.rtol))
    if key in _PLANT_CACHE:
        return _PLANT_CACHE[key]
    diagram, handles = make_ebike_diagram(EbikeConfig(CdA=key[0]), return_handles=True,
                                          enable_speed_event=True)
    ctx = diagram.create_context()
    aid = handles["assist_policy"].system_id
    ports = {p.name: p for p in diagram.output_ports}
    rec = {"E": ports["E_batt_term"], "d": ports["distance"], "H": ports["E_human"]}

    def rollout(cap):
        c = ctx.with_subcontext(
            aid, ctx[aid].with_parameter("max_assist_torque", jnp.array(float(cap))))
        res = jaxonomy.simulate(diagram, c, (0.0, TF), options=opts,
                                recorded_signals=rec)
        return _metrics_from(res)

    _PLANT_CACHE[key] = rollout
    return rollout


def sim_design(m_cargo, CdA, wheel_reduction, cap=NOMINAL_CAP):
    """Full rebuild+rollout for a hardware design vector (Sobol study), at the
    fixed nominal assist cap."""
    cfg = EbikeConfig(max_assist_torque=float(cap), m_cargo=float(m_cargo),
                      CdA=float(CdA), wheel_reduction=float(wheel_reduction))
    diagram = make_ebike_diagram(cfg, enable_speed_event=True)
    ctx = diagram.create_context()
    ports = {p.name: p for p in diagram.output_ports}
    rec = {"E": ports["E_batt_term"], "d": ports["distance"], "H": ports["E_human"]}
    res = jaxonomy.simulate(diagram, ctx, (0.0, TF), options=_OPTS,
                            recorded_signals=rec)
    return _metrics_from(res)


def objective(m, lam=LAMBDA_SPEED):
    if not np.isfinite(m["E_D"]):
        return 1e9
    shortfall = max(0.0, V_TARGET_KMH - m["v_D"])
    return m["E_D"] + lam * shortfall ** 2


def main():
    t_start = time.time()

    # === 1. control-vs-design response grid (cap x CdA) ===================
    print("[1/5] Response grid (assist cap x CdA)...")
    if SMOKE:
        grid_caps = np.array([8.0, 14.0])
        grid_cda = np.array([0.68, 0.90])
    else:
        grid_caps = np.array([6.0, 8.4, 10.8, 13.2, 15.6, 18.0])
        grid_cda = np.array([0.66, 0.80, 0.94])
    G = (len(grid_cda), len(grid_caps))
    grid_ED = np.zeros(G)
    grid_vD = np.zeros(G)
    grid_HD = np.zeros(G)
    grid_tD = np.zeros(G)
    for i, cda in enumerate(grid_cda):
        rollout = plant_for(cda)
        for j, cap in enumerate(grid_caps):
            m = rollout(cap)
            grid_ED[i, j], grid_vD[i, j] = m["E_D"], m["v_D"]
            grid_HD[i, j], grid_tD[i, j] = m["H_D"], m["t_D"]
            print(f"    CdA={cda:4.2f} cap={cap:5.1f} -> E@{D_REF:.0f}m={m['E_D']:8.0f} J "
                  f"v={m['v_D']:5.2f} km/h  H={m['H_D']:6.0f} J")

    base_roll = plant_for(BASE_CDA)
    base_cap = 18.0
    base_m = base_roll(base_cap)

    # --- lambda sweep from grid interpolants (no extra sims): where does the
    # penalty equilibrium sit vs lambda? Exterior penalties converge from the
    # infeasible side; this curve shows the shortfall shrinking ~ 1/lambda.
    i_base = int(np.argmin(np.abs(grid_cda - BASE_CDA)))
    caps_f = np.linspace(CAP_BOUNDS[0], CAP_BOUNDS[1], 400)
    E_f = np.interp(caps_f, grid_caps, grid_ED[i_base])
    v_f = np.interp(caps_f, grid_caps, grid_vD[i_base])
    lam_sweep = np.array([5e2, 2e3, 8e3, 2e4, 8e4, 3e5])
    lam_vopt = []
    for lam in lam_sweep:
        J_f = E_f + lam * np.maximum(0.0, V_TARGET_KMH - v_f) ** 2
        lam_vopt.append(float(v_f[np.argmin(J_f)]))
    lam_vopt = np.array(lam_vopt)
    print("    lambda sweep (from grid): v_opt =",
          np.round(lam_vopt, 3), "vs floor", V_TARGET_KMH)

    # === 2. measured noise floor (tolerance study, not an assertion) =======
    print("[2/5] Noise floor: J at rtol 5e-4 vs 1e-4 (discretisation error)...")
    tight = SimulatorOptions(enable_autodiff=False, rtol=1e-4, atol=1e-6,
                             buffer_length=260000)
    tight_roll = plant_for(BASE_CDA, opts=tight)
    noise_caps = np.array([8.0, 12.0, 16.0]) if not SMOKE else np.array([12.0])
    noise_dJ = []
    for cap in noise_caps:
        J_a = objective(base_roll(cap))
        J_b = objective(tight_roll(cap))
        noise_dJ.append(abs(J_a - J_b))
        print(f"    cap={cap:5.1f}: J(5e-4)={J_a:9.1f}  J(1e-4)={J_b:9.1f}  |dJ|={noise_dJ[-1]:6.1f}")
    noise_floor_J = float(max(noise_dJ))
    # optimiser tolerances must sit ABOVE the measured floor
    fatol_nm = max(4.0 * noise_floor_J, 20.0)
    print(f"    noise floor ~ {noise_floor_J:.1f} J -> NM fatol = {fatol_nm:.1f} J")

    # === 3. 1-D bounded Brent on the assist cap (baseline design) =========
    print("[3/5] 1-D bounded optimisation (Brent) over assist cap...")
    brent_hist = []

    def brent_obj(cap):
        J = objective(base_roll(float(cap)))
        brent_hist.append((float(cap), float(J)))
        print(f"    Brent cap={float(cap):6.3f} -> J={J:11.1f}")
        return J

    maxit = 6 if SMOKE else 20
    r1 = minimize_scalar(brent_obj, bounds=CAP_BOUNDS, method="bounded",
                         options={"xatol": 0.3, "maxiter": maxit})
    brent_cap_opt = float(r1.x)
    brent_m = base_roll(brent_cap_opt)

    # === 4. 2-D Nelder-Mead over (cap, CdA) ===============================
    print("[4/5] 2-D derivative-free optimisation (Nelder-Mead over cap, CdA)...")
    nm_hist = []

    def nm_obj(x):
        cap = float(np.clip(x[0], *CAP_BOUNDS))
        cda = float(np.clip(x[1], *CDA_BOUNDS))
        m = plant_for(cda)(cap)
        pen = 5e3 * (min(0.0, x[0] - CAP_BOUNDS[0]) ** 2
                     + max(0.0, x[0] - CAP_BOUNDS[1]) ** 2
                     + min(0.0, x[1] - CDA_BOUNDS[0]) ** 2
                     + max(0.0, x[1] - CDA_BOUNDS[1]) ** 2)
        J = objective(m) + pen
        nm_hist.append((cap, cda, float(J)))
        print(f"    NM cap={cap:6.3f} CdA={cda:5.3f} -> J={J:11.1f}")
        return J

    maxfev = 8 if SMOKE else 18
    r2 = minimize(nm_obj, x0=np.array([base_cap, BASE_CDA]), method="Nelder-Mead",
                  options={"xatol": 0.3, "fatol": fatol_nm, "maxfev": maxfev,
                           "initial_simplex": np.array([[base_cap, BASE_CDA],
                                                        [11.0, BASE_CDA],
                                                        [base_cap, 0.70]])})
    nm_x_opt = np.array([float(np.clip(r2.x[0], *CAP_BOUNDS)),
                         float(np.clip(r2.x[1], *CDA_BOUNDS))])
    nm_m = plant_for(round(nm_x_opt[1], 4))(nm_x_opt[0])

    # === 5. Hardware Sobol at fixed cap, with bootstrap CIs ================
    print("[5/5] Hardware Sobol (LHS + order-2 PCE + bootstrap CI), cap fixed "
          f"at {NOMINAL_CAP} Nm...")
    sobol_names = np.array(["m_cargo", "CdA", "wheel_gearing"])
    sobol_bounds = np.array([[20.0, 80.0], [0.65, 1.00], [0.42, 0.58]])
    if SMOKE:
        N_TRAIN, N_VAL = 8, 3
    else:
        N_TRAIN, N_VAL = 24, 8
    sampler = qmc.LatinHypercube(d=3, seed=0)
    unit = sampler.random(n=N_TRAIN + N_VAL)
    X_all = qmc.scale(unit, sobol_bounds[:, 0], sobol_bounds[:, 1])

    Y_E = np.zeros(len(X_all))
    Y_v = np.zeros(len(X_all))
    for k, x in enumerate(X_all):
        m = sim_design(*x)
        Y_E[k], Y_v[k] = m["E_D"], m["v_D"]
        print(f"    design {k + 1:2d}/{len(X_all)} m={x[0]:5.1f} CdA={x[1]:4.2f} "
              f"g={x[2]:4.2f} -> E@{D_REF:.0f}m={m['E_D']:8.0f} v={m['v_D']:5.2f}")

    Xtr, Xva = X_all[:N_TRAIN], X_all[N_TRAIN:]
    YE_tr, YE_va = Y_E[:N_TRAIN], Y_E[N_TRAIN:]
    Yv_tr = Y_v[:N_TRAIN]
    dists = [("uniform", lo, hi) for lo, hi in sobol_bounds]

    order = 1 if SMOKE else 2
    pce_E = fit_pce(Xtr, YE_tr, dists, order=order)
    pce_v = fit_pce(Xtr, Yv_tr, dists, order=order)
    s_E = pce_E.sobol_indices()
    s_v = pce_v.sobol_indices()

    yhat_tr = np.asarray(pce_E.predict(Xtr))
    yhat_va = np.asarray(pce_E.predict(Xva))
    R2_train = 1.0 - np.sum((YE_tr - yhat_tr) ** 2) / np.sum((YE_tr - YE_tr.mean()) ** 2)
    R2_val = 1.0 - np.sum((YE_va - yhat_va) ** 2) / np.sum((YE_va - YE_va.mean()) ** 2)

    # bootstrap CIs on the first-order indices (refit on resampled designs)
    B = 200 if not SMOKE else 20
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, N_TRAIN, N_TRAIN)
        try:
            pb = fit_pce(Xtr[idx], YE_tr[idx], dists, order=order)
            boots.append(np.asarray(pb.sobol_indices()["first_order"]))
        except Exception:
            continue
    boots = np.array(boots)
    ci_lo = np.percentile(boots, 2.5, axis=0)
    ci_hi = np.percentile(boots, 97.5, axis=0)

    wall = time.time() - t_start
    print(f"\nTotal offline wall-time: {wall / 60:.1f} min")

    bh = np.array(brent_hist)
    nh = np.array(nm_hist)
    np.savez(
        NPZ,
        placeholder_flag=False, smoke=SMOKE,
        wall_time_s=wall, TF=TF, D_ref=D_REF,
        v_floor=V_TARGET_KMH, lambda_speed=LAMBDA_SPEED,
        noise_caps=noise_caps, noise_dJ=np.array(noise_dJ),
        noise_floor_J=noise_floor_J, nm_fatol=fatol_nm,
        lam_sweep=lam_sweep, lam_vopt=lam_vopt,
        grid_caps=grid_caps, grid_cda=grid_cda,
        grid_ED=grid_ED, grid_vD=grid_vD, grid_HD=grid_HD, grid_tD=grid_tD,
        base_cap=base_cap, base_cda=BASE_CDA,
        base_ED=base_m["E_D"], base_vD=base_m["v_D"], base_HD=base_m["H_D"],
        brent_hist_cap=bh[:, 0], brent_hist_J=bh[:, 1],
        brent_cap_opt=brent_cap_opt, brent_ED=brent_m["E_D"],
        brent_vD=brent_m["v_D"], brent_HD=brent_m["H_D"], brent_nfev=len(bh),
        nm_hist_x=nh[:, :2], nm_hist_J=nh[:, 2],
        nm_x_opt=nm_x_opt, nm_ED=nm_m["E_D"], nm_vD=nm_m["v_D"],
        nm_HD=nm_m["H_D"], nm_nfev=len(nh),
        sobol_names=sobol_names, sobol_bounds=sobol_bounds, sobol_order=order,
        sobol_cap=NOMINAL_CAP,
        sobol_X=Xtr, sobol_ED=YE_tr, sobol_vD=Yv_tr,
        sobol_Xval=Xva, sobol_yval_E=YE_va, sobol_ypred_E=yhat_va,
        sobol_first_E=np.asarray(s_E["first_order"]),
        sobol_total_E=np.asarray(s_E["total"]),
        sobol_first_v=np.asarray(s_v["first_order"]),
        sobol_total_v=np.asarray(s_v["total"]),
        sobol_ci_lo=ci_lo, sobol_ci_hi=ci_hi, sobol_boot_B=len(boots),
        sobol_R2_train=float(R2_train), sobol_R2_val=float(R2_val),
    )
    sz = os.path.getsize(NPZ) / 1024
    print(f"Wrote {NPZ}  ({sz:.1f} KB)")
    print(f"  Brent cap*    = {brent_cap_opt:.2f} Nm  (E@{D_REF:.0f}m={brent_m['E_D']:.0f} J, "
          f"v={brent_m['v_D']:.2f} km/h vs floor {V_TARGET_KMH})")
    print(f"  NM (cap,CdA)* = ({nm_x_opt[0]:.2f}, {nm_x_opt[1]:.3f})  "
          f"(E@{D_REF:.0f}m={nm_m['E_D']:.0f} J, v={nm_m['v_D']:.2f})")
    print("  Hardware Sobol first-order (E@D): "
          + ", ".join(f"{n}={s:.2f} [{lo:.2f},{hi:.2f}]" for n, s, lo, hi in
                      zip(sobol_names, np.asarray(s_E["first_order"]), ci_lo, ci_hi)))
    print(f"  PCE R^2 train={R2_train:.3f} val={R2_val:.3f}")


if __name__ == "__main__":
    main()
