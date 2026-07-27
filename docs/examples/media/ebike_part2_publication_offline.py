#!/usr/bin/env python3
"""Offline publication run for ``ebike_part2_optimization.ipynb``.

Runs every heavy optimisation / sensitivity study for Part 2 at full fidelity
and writes a compact checkpoint ``media/ebike_part2_publication.npz`` (< 200 KB).
The notebook loads the NPZ in < 1 s so the reader's experience stays fast; a
single rollout of the true multi-domain DAE is ~30 s of CPU, and this script
runs ~90 of them, so it takes ~50 min on a developer machine.

Set the environment variable ``EBIKE_P2_SMOKE=1`` to run a tiny smoke version
(coarse grid, few optimiser evals, small Sobol design) in a couple of minutes,
purely to validate the code path before the full run.

What it produces
----------------
1. Control-vs-design response grid  (assist cap x drag area CdA): battery energy,
   mean speed, rider work at every node.  One diagram *build* per CdA, then a
   cheap assist-cap sweep on it (the cap is a runtime-tunable dynamic parameter).
   Feeds the design-space heatmap AND the multi-objective Pareto cloud.
2. 1-D derivative-free optimisation (bounded Brent) of the assist-torque cap
   against the true DAE at the baseline design, with its full eval history.
3. 2-D derivative-free optimisation (Nelder-Mead) over (cap, CdA), history.
4. Global Sobol sensitivity via an order-2 polynomial-chaos surrogate fitted to a
   Latin-hypercube design over four *design* parameters (assist cap, cargo mass,
   drag area CdA, wheel gearing), each sample a full rebuild+rollout, plus a
   held-out validation batch.

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
TF = 30.0
V_TARGET_KMH = 12.0
LAMBDA_SPEED = 800.0
CAP_BOUNDS = (6.0, 20.0)
CDA_BOUNDS = (0.62, 0.95)          # design range: faired cargo pod ... upright
BASE_CDA = 0.80
_OPTS = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6,
                         buffer_length=120000)

# cache of (rollout closures) keyed by rounded CdA so the grid + optimiser reuse
# a compiled diagram whenever the drag area repeats.
_PLANT_CACHE: dict[float, object] = {}


def plant_for(CdA):
    """Return ``rollout(cap) -> (E_batt, v_mean, E_human)`` for a diagram built
    at drag area ``CdA``. The assist-torque cap is a runtime dynamic parameter,
    so sweeping caps on the returned closure needs no rebuild."""
    key = round(float(CdA), 4)
    if key in _PLANT_CACHE:
        return _PLANT_CACHE[key]
    diagram, handles = make_ebike_diagram(EbikeConfig(CdA=key), return_handles=True,
                                          enable_speed_event=True)
    ctx = diagram.create_context()
    aid = handles["assist_policy"].system_id
    ports = {p.name: p for p in diagram.output_ports}
    rec = {"E": ports["E_batt_term"], "d": ports["distance"], "H": ports["E_human"]}

    def rollout(cap):
        c = ctx.with_subcontext(
            aid, ctx[aid].with_parameter("max_assist_torque", jnp.array(float(cap))))
        res = jaxonomy.simulate(diagram, c, (0.0, TF), options=_OPTS,
                                recorded_signals=rec)
        o = res.outputs
        return (float(np.asarray(o["E"])[-1]),
                float(np.asarray(o["d"])[-1]) / TF * 3.6,
                float(np.asarray(o["H"])[-1]))

    _PLANT_CACHE[key] = rollout
    return rollout


def sim_design(cap, m_cargo, CdA, wheel_reduction):
    """Full rebuild+rollout for a 4-D design vector (Sobol study)."""
    cfg = EbikeConfig(max_assist_torque=float(cap), m_cargo=float(m_cargo),
                      CdA=float(CdA), wheel_reduction=float(wheel_reduction))
    diagram = make_ebike_diagram(cfg, enable_speed_event=True)
    ctx = diagram.create_context()
    ports = {p.name: p for p in diagram.output_ports}
    rec = {"E": ports["E_batt_term"], "d": ports["distance"]}
    res = jaxonomy.simulate(diagram, ctx, (0.0, TF), options=_OPTS,
                            recorded_signals=rec)
    o = res.outputs
    return (float(np.asarray(o["E"])[-1]),
            float(np.asarray(o["d"])[-1]) / TF * 3.6)


def objective(E, v):
    shortfall = max(0.0, V_TARGET_KMH - v)
    return E + LAMBDA_SPEED * shortfall ** 2


def main():
    t_start = time.time()

    # === 1. control-vs-design response grid (cap x CdA) ===================
    print("[1/4] Response grid (assist cap x CdA)...")
    if SMOKE:
        grid_caps = np.array([8.0, 14.0])
        grid_cda = np.array([0.68, 0.90])
    else:
        grid_caps = np.array([6.0, 8.4, 10.8, 13.2, 15.6, 18.0])
        grid_cda = np.array([0.66, 0.80, 0.94])
    G = (len(grid_cda), len(grid_caps))
    grid_Ebatt = np.zeros(G)
    grid_vmean = np.zeros(G)
    grid_Ehuman = np.zeros(G)
    for i, cda in enumerate(grid_cda):
        rollout = plant_for(cda)
        for j, cap in enumerate(grid_caps):
            E, v, H = rollout(cap)
            grid_Ebatt[i, j], grid_vmean[i, j], grid_Ehuman[i, j] = E, v, H
            print(f"    CdA={cda:4.2f} cap={cap:5.1f} -> E={E:8.0f} v={v:5.2f} H={H:7.0f}")

    base_roll = plant_for(BASE_CDA)
    base_cap = 18.0
    base_E, base_v, base_H = base_roll(base_cap)

    # === 2. 1-D bounded Brent on the assist cap (baseline design) =========
    print("[2/4] 1-D bounded optimisation (Brent) over assist cap...")
    brent_hist = []

    def brent_obj(cap):
        E, v, H = base_roll(float(cap))
        J = objective(E, v)
        brent_hist.append((float(cap), float(J)))
        print(f"    Brent cap={float(cap):6.3f} -> J={J:11.1f}")
        return J

    maxit = 6 if SMOKE else 20
    r1 = minimize_scalar(brent_obj, bounds=CAP_BOUNDS, method="bounded",
                         options={"xatol": 0.15, "maxiter": maxit})
    brent_cap_opt = float(r1.x)
    bE, bv, bH = base_roll(brent_cap_opt)

    # === 3. 2-D Nelder-Mead over (cap, CdA) ===============================
    print("[3/4] 2-D derivative-free optimisation (Nelder-Mead over cap, CdA)...")
    nm_hist = []

    def nm_obj(x):
        cap = float(np.clip(x[0], *CAP_BOUNDS))
        cda = float(np.clip(x[1], *CDA_BOUNDS))
        E, v, H = plant_for(cda)(cap)
        pen = 5e3 * (min(0.0, x[0] - CAP_BOUNDS[0]) ** 2
                     + max(0.0, x[0] - CAP_BOUNDS[1]) ** 2
                     + min(0.0, x[1] - CDA_BOUNDS[0]) ** 2
                     + max(0.0, x[1] - CDA_BOUNDS[1]) ** 2)
        J = objective(E, v) + pen
        nm_hist.append((cap, cda, float(J)))
        print(f"    NM cap={cap:6.3f} CdA={cda:5.3f} -> J={J:11.1f}")
        return J

    maxfev = 8 if SMOKE else 22
    r2 = minimize(nm_obj, x0=np.array([base_cap, BASE_CDA]), method="Nelder-Mead",
                  options={"xatol": 0.15, "fatol": 5.0, "maxfev": maxfev,
                           "initial_simplex": np.array([[base_cap, BASE_CDA],
                                                        [11.0, BASE_CDA],
                                                        [base_cap, 0.70]])})
    nm_x_opt = np.array([float(np.clip(r2.x[0], *CAP_BOUNDS)),
                         float(np.clip(r2.x[1], *CDA_BOUNDS))])
    nE, nv, nH = plant_for(round(nm_x_opt[1], 4))(nm_x_opt[0])

    # === 4. Global Sobol via order-2 PCE over 4 design parameters =========
    print("[4/4] Sobol sensitivity (LHS design + order-2 PCE)...")
    sobol_names = np.array(["assist_cap", "m_cargo", "CdA", "wheel_gearing"])
    sobol_bounds = np.array([[6.0, 18.0], [20.0, 80.0],
                             [0.65, 1.00], [0.42, 0.58]])
    if SMOKE:
        N_TRAIN, N_VAL = 8, 3
    else:
        N_TRAIN, N_VAL = 26, 6
    sampler = qmc.LatinHypercube(d=4, seed=0)
    unit = sampler.random(n=N_TRAIN + N_VAL)
    X_all = qmc.scale(unit, sobol_bounds[:, 0], sobol_bounds[:, 1])

    Y_E = np.zeros(len(X_all))
    Y_v = np.zeros(len(X_all))
    for k, x in enumerate(X_all):
        E, v = sim_design(*x)
        Y_E[k], Y_v[k] = E, v
        print(f"    design {k + 1:2d}/{len(X_all)} cap={x[0]:5.2f} m={x[1]:5.1f} "
              f"CdA={x[2]:4.2f} g={x[3]:4.2f} -> E={E:8.0f} v={v:5.2f}")

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

    wall = time.time() - t_start
    print(f"\nTotal offline wall-time: {wall / 60:.1f} min")

    bh = np.array(brent_hist)
    nh = np.array(nm_hist)
    np.savez(
        NPZ,
        placeholder_flag=False, smoke=SMOKE,
        wall_time_s=wall, TF=TF, v_floor=V_TARGET_KMH, lambda_speed=LAMBDA_SPEED,
        grid_caps=grid_caps, grid_cda=grid_cda,
        grid_Ebatt=grid_Ebatt, grid_vmean=grid_vmean, grid_Ehuman=grid_Ehuman,
        base_cap=base_cap, base_cda=BASE_CDA,
        base_Ebatt=base_E, base_vmean=base_v, base_Ehuman=base_H,
        brent_hist_cap=bh[:, 0], brent_hist_J=bh[:, 1],
        brent_cap_opt=brent_cap_opt, brent_Ebatt_opt=bE,
        brent_vmean_opt=bv, brent_Ehuman_opt=bH, brent_nfev=len(bh),
        nm_hist_x=nh[:, :2], nm_hist_J=nh[:, 2],
        nm_x_opt=nm_x_opt, nm_Ebatt_opt=nE, nm_vmean_opt=nv, nm_Ehuman_opt=nH,
        nm_nfev=len(nh),
        sobol_names=sobol_names, sobol_bounds=sobol_bounds, sobol_order=order,
        sobol_X=Xtr, sobol_Ebatt=YE_tr, sobol_vmean=Yv_tr,
        sobol_Xval=Xva, sobol_yval_E=YE_va, sobol_ypred_E=yhat_va,
        sobol_first_E=np.asarray(s_E["first_order"]),
        sobol_total_E=np.asarray(s_E["total"]),
        sobol_first_v=np.asarray(s_v["first_order"]),
        sobol_total_v=np.asarray(s_v["total"]),
        sobol_mean_E=float(pce_E.mean()), sobol_var_E=float(pce_E.variance()),
        sobol_R2_train=float(R2_train), sobol_R2_val=float(R2_val),
    )
    sz = os.path.getsize(NPZ) / 1024
    print(f"Wrote {NPZ}  ({sz:.1f} KB)")
    print(f"  Brent cap*   = {brent_cap_opt:.2f} Nm  (E={bE:.0f} J, v={bv:.2f} km/h)")
    print(f"  NM (cap,CdA)* = ({nm_x_opt[0]:.2f}, {nm_x_opt[1]:.3f})  (E={nE:.0f} J, v={nv:.2f})")
    print("  Sobol first-order (E_batt): "
          + ", ".join(f"{n}={s:.2f}" for n, s in
                      zip(sobol_names, np.asarray(s_E["first_order"]))))
    print(f"  PCE R^2 train={R2_train:.3f} val={R2_val:.3f}")


if __name__ == "__main__":
    main()
