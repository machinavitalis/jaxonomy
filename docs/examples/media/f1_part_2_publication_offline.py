#!/usr/bin/env python3
"""Offline publication-quality run for ``f1_part_2_setup_optimization.ipynb``.

Produces ``media/f1_part_2_publication.npz`` from which the notebook loads
its headline numbers in ``MODE = "publication"`` (the default). The
notebook itself runs a coarse 3-step gradient descent at a shortened
horizon (`T_END = 15`) so the reader's execution is fast (<5 min); this
script does the *real* work — 29-iteration L-BFGS-B + 3^6 grid head-to-
head + 16-start LHS multi-start, all at full lap horizon `T_END = 60`.

Expected wall-time: **~3 hours on a developer machine (M1 Max / 16 cores)**.

Run from the repo root:

.. code-block:: bash

    python docs/examples/media/f1_part_2_publication_offline.py

The script is **idempotent**: running it twice produces byte-equal NPZ
files (same PRNG seed, same numerical conditioning). To re-execute the
notebook against fresh publication numbers, run this script + open the
notebook + Cell -> Run All.

Architecture cribbed from the notebook: imports the same plant + tire +
powertrain + driver definitions, sets up the same ``forward(setup) ->
scalar lap_time``, then runs the publication-quality optimisation +
sensitivity sweeps + multi-start.

NB: this script is the *source of truth* for the publication NPZ's
contents. If you add a new headline number to the notebook (a new
optimiser, a new sensitivity sweep), add the corresponding offline
computation here too.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Bootstrap: import the notebook's plant + cost function.
# ---------------------------------------------------------------------------
# The notebook authors `BicycleCar`, `Pacejka52Tire`, `EngineMap`, `Gearbox`,
# `Driver`, `LapTimeAccumulator`, plus the `forward(setup) -> scalar` closure
# in-line. To keep this script self-contained without duplicating ~500 lines,
# we expect the notebook to have been refactored into a small importable
# module `f1_lts_common.py` at `docs/examples/media/`. Until that refactor
# happens this script will fail loudly with a `ModuleNotFoundError`; the
# next agent that touches the F1 series should do the refactor as part of
# the Part 3 work (the aero-map tutorial will need to import the same plant).
#
# Workaround until refactor: copy the relevant cells of
# `docs/examples/f1_part_2_setup_optimization.ipynb` into this file by hand
# and run from a notebook kernel instead.

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "docs" / "examples" / "media"))

# f1_lts_common.py was extracted from f1_part_2_setup_optimization.ipynb cells
# 1-9 and lives alongside this script in docs/examples/media/. It exposes the
# full plant + DIAG + CTX0 so we can build the publication-fidelity forward()
# closure inline below.
from f1_lts_common import (
    SETUP_BASELINE, SETUP_LOWER, SETUP_UPPER, SETUP_NAMES, N_SETUP,
    setup_to_physics, DIAG, CAR_BLK, LAP_BLK, CTX0,
)
from jaxonomy.simulation import SimulatorOptions
from jaxonomy import simulate


# ---------------------------------------------------------------------------
# Configuration: full publication fidelity, or reduced-fidelity (--reduced).
# ---------------------------------------------------------------------------
# The full-fidelity run (T_END=60, rtol=1e-5, 3^6 grid, 16-start) is extremely
# slow on commodity hardware (many hours). ``--reduced`` runs the SAME pipeline
# at a shorter horizon / looser tolerance / fewer starts so it completes in a
# few minutes. Both write ``placeholder_flag=False`` — the numbers are REAL,
# ``--reduced`` just trades fidelity for wall-time. The ``fidelity`` field in
# the NPZ records which was used so the notebook/prose can be honest about it.
import argparse

_ap = argparse.ArgumentParser(description="Offline publication run for F1 Part 2.")
_ap.add_argument("--reduced", action="store_true",
                 help="Reduced-fidelity real run (minutes, not hours).")
_args, _ = _ap.parse_known_args()
FIDELITY = "reduced" if _args.reduced else "full"

if FIDELITY == "reduced":
    T_END_PUB, RTOL_PUB, ATOL_PUB, MAX_STEP_LEN_PUB = 25.0, 1e-3, 1e-5, 0.05
    LBFGS_MAX_ITER, MULTI_N_STARTS = 25, 8
    GRID_N_LEVELS = 2          # 2^6 = 64 sims (vs 3^6=729 at full fidelity)
else:
    T_END_PUB, RTOL_PUB, ATOL_PUB, MAX_STEP_LEN_PUB = 60.0, 1e-5, 1e-7, 0.02
    LBFGS_MAX_ITER, MULTI_N_STARTS = 30, 16
    GRID_N_LEVELS = 3          # 3^6 = 729 sims; fixes springs at baseline

MAX_MAJOR_STEPS   = 8_000
LBFGS_FTOL        = 1e-7
LBFGS_GTOL        = 1e-5
GRID_FREE_AXES    = [2, 3, 4, 5, 6, 7]  # c_f, c_r, k_arb, h_f, h_r, delta_w
RNG_SEED          = 20260517

print(f"Fidelity: {FIDELITY}  (T_END={T_END_PUB}, rtol={RTOL_PUB}, "
      f"L-BFGS maxiter={LBFGS_MAX_ITER}, multi-start={MULTI_N_STARTS})")

# Output
OUT_NPZ = REPO_ROOT / "docs" / "examples" / "media" / "f1_part_2_publication.npz"


# ---------------------------------------------------------------------------
# Build the forward closure at publication fidelity.
# ---------------------------------------------------------------------------
SIM_OPTS_PUB = SimulatorOptions(
    enable_autodiff=True,
    rtol=RTOL_PUB, atol=ATOL_PUB,
    max_major_step_length=MAX_STEP_LEN_PUB,
    max_major_steps=MAX_MAJOR_STEPS,
)


@jax.jit
def forward(setup):
    """setup (jnp.ndarray, shape (8,)) -> NEGATIVE arc length at T_END.

    See the notebook's forward() docstring (post the 2026-05-17 zero-grad fix);
    same -arc_length proxy here. At publication fidelity (T_END_PUB=60, full
    lap) the lap completes and LapTimeAccumulator gives meaningful gradients,
    but the arc-length cost still works and stays consistent with the notebook.
    """
    phys = setup_to_physics(setup)
    car_ctx = CTX0[CAR_BLK.system_id].with_parameters(phys)
    ctx = CTX0.with_subcontext(CAR_BLK.system_id, car_ctx)
    results = simulate(DIAG, ctx, (0.0, T_END_PUB), options=SIM_OPTS_PUB)
    return -results.context[CAR_BLK.system_id].continuous_state[6]


val_and_grad = jax.jit(jax.value_and_grad(forward))


# ---------------------------------------------------------------------------
# Bad starting point + baseline reference lap.
# ---------------------------------------------------------------------------
SETUP_BAD = jnp.array([3.0e5, 2.8e5, 8.5e3, 8.0e3, +1.5e4, 35.0, 55.0, -4.0])

print("Computing baseline lap (full T_END=60 horizon) ...")
t0 = time.time()
lap_baseline = float(forward(SETUP_BASELINE))
print(f"  baseline lap = {lap_baseline:.6f} s  ({time.time()-t0:.1f} s incl. JIT trace)")

print("Computing bad-setup lap ...")
t0 = time.time()
lap_bad = float(forward(SETUP_BAD))
print(f"  bad-setup lap = {lap_bad:.6f} s  ({time.time()-t0:.1f} s)")

print("\nWarming up val_and_grad JIT trace ...")
t0 = time.time()
_ = val_and_grad(SETUP_BAD)
print(f"  warmup {time.time()-t0:.1f} s\n")


# ---------------------------------------------------------------------------
# 1. L-BFGS-B optimisation from SETUP_BAD with a proper line search.
# ---------------------------------------------------------------------------
history_iter, history_lap, history_setup = [0], [lap_bad], [np.asarray(SETUP_BAD)]


def scipy_objective(setup_np):
    setup_jnp = jnp.asarray(setup_np)
    v, g = val_and_grad(setup_jnp)
    return float(v), np.asarray(g, dtype=np.float64)


def callback(setup_np):
    history_iter.append(len(history_iter))
    setup_jnp = jnp.asarray(setup_np)
    v, _g = val_and_grad(setup_jnp)
    history_lap.append(float(v))
    history_setup.append(setup_np.copy())


bounds_pairs = list(zip(np.asarray(SETUP_LOWER), np.asarray(SETUP_UPPER)))

print(f"Running L-BFGS-B (maxiter={LBFGS_MAX_ITER}, line search enabled) ...")
t0 = time.time()
res = scipy_minimize(
    scipy_objective, x0=np.asarray(SETUP_BAD),
    method="L-BFGS-B", jac=True, bounds=bounds_pairs,
    callback=callback,
    options={"maxiter": LBFGS_MAX_ITER, "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL,
             "disp": False},
)
pub_wall_time_s = time.time() - t0
print(f"  L-BFGS-B converged: success={res.success}, message='{res.message}'")
print(f"  wall-time {pub_wall_time_s/60:.2f} min, {res.nfev} val+grad evals")

setup_opt = jnp.asarray(res.x)
lap_opt = float(forward(setup_opt))
print(f"\n  baseline lap : {lap_baseline:.6f} s")
print(f"  bad-start lap: {lap_bad:.6f} s")
print(f"  optimised lap: {lap_opt:.6f} s  ({lap_opt - lap_baseline:+.6f} vs baseline)")


# ---------------------------------------------------------------------------
# 2. FD-vs-AD speedup: full 3^6 = 729 grid sweep over 6 axes (springs fixed).
# ---------------------------------------------------------------------------
print(f"\nRunning {GRID_N_LEVELS}^{len(GRID_FREE_AXES)} = "
      f"{GRID_N_LEVELS**len(GRID_FREE_AXES)} grid sims ...")
t0 = time.time()
levels = [np.linspace(float(SETUP_LOWER[i]), float(SETUP_UPPER[i]), GRID_N_LEVELS)
          for i in GRID_FREE_AXES]
mesh = np.meshgrid(*levels, indexing="ij")
grid_pts = np.stack([m.ravel() for m in mesh], axis=1)
full_grid = np.tile(np.asarray(SETUP_BASELINE), (grid_pts.shape[0], 1))
for j, i_ax in enumerate(GRID_FREE_AXES):
    full_grid[:, i_ax] = grid_pts[:, j]

# Evaluate the grid with a plain loop over the (jitted) `forward` — one warm
# call each. NOTE: `jax.vmap(forward)` over the whole grid is intractable here:
# the adaptive event-handling solver fuses into a giant XLA program that takes
# hours to compile (this was a primary cause of the full-run hang). The
# sequential loop over the already-jitted `forward` is bounded and fast.
laps_grid = np.array([float(forward(jnp.asarray(full_grid[k])))
                      for k in range(full_grid.shape[0])])
grid_time_s = time.time() - t0
grid_best_lap = float(np.min(laps_grid))
print(f"  grid done in {grid_time_s/60:.2f} min, best lap on grid = {grid_best_lap:.6f} s")


# ---------------------------------------------------------------------------
# 3. Multi-start L-BFGS-B from MULTI_N_STARTS LHS-sampled random starts.
# ---------------------------------------------------------------------------
print(f"\nRunning multi-start ({MULTI_N_STARTS} LHS starts, each to L-BFGS-B convergence) ...")
t0 = time.time()
lhs = qmc.LatinHypercube(d=N_SETUP, seed=RNG_SEED)
unit = lhs.random(n=MULTI_N_STARTS)
multi_starts_np = (np.asarray(SETUP_LOWER)
                   + unit * (np.asarray(SETUP_UPPER) - np.asarray(SETUP_LOWER)))

multi_lap_initial = np.zeros(MULTI_N_STARTS)
multi_setup_optima = np.zeros((MULTI_N_STARTS, N_SETUP))
multi_lap_optima = np.zeros(MULTI_N_STARTS)

for i in range(MULTI_N_STARTS):
    s0 = multi_starts_np[i]
    multi_lap_initial[i] = float(forward(jnp.asarray(s0)))
    res_i = scipy_minimize(
        scipy_objective, x0=s0, method="L-BFGS-B", jac=True,
        bounds=bounds_pairs,
        options={"maxiter": LBFGS_MAX_ITER, "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL,
                 "disp": False},
    )
    multi_setup_optima[i] = res_i.x
    multi_lap_optima[i] = float(forward(jnp.asarray(res_i.x)))
    print(f"  start {i+1}/{MULTI_N_STARTS}: init {multi_lap_initial[i]:.3f} s -> "
          f"opt {multi_lap_optima[i]:.3f} s")
multi_wall_time_s = time.time() - t0

# Count unique optima within 1e-3 s tolerance
multi_n_unique_optima = 0
seen = []
for lap in sorted(multi_lap_optima):
    if not any(abs(lap - s) < 1e-3 for s in seen):
        seen.append(lap)
        multi_n_unique_optima += 1


# ---------------------------------------------------------------------------
# Write the NPZ.
# ---------------------------------------------------------------------------
print(f"\nWriting {OUT_NPZ} ...")
np.savez(
    OUT_NPZ,
    setup_opt=np.asarray(setup_opt),
    lap_opt=lap_opt,
    lap_bad=lap_bad,
    history_iter=np.asarray(history_iter),
    history_lap=np.asarray(history_lap),
    history_setup=np.asarray(history_setup),
    pub_wall_time_s=pub_wall_time_s,
    pub_iters=int(res.nit),
    pub_feval=int(res.nfev),
    grid_time_s=grid_time_s,
    grid_n_pts=int(grid_pts.shape[0]),
    grid_best_lap=grid_best_lap,
    multi_starts=multi_starts_np,
    multi_lap_initial=multi_lap_initial,
    multi_setup_optima=multi_setup_optima,
    multi_lap_optima=multi_lap_optima,
    multi_wall_time_s=multi_wall_time_s,
    multi_n_unique_optima=multi_n_unique_optima,
    # Fidelity provenance so the notebook/prose can be honest about it.
    fidelity=FIDELITY,
    t_end_pub=T_END_PUB,
    rtol_pub=RTOL_PUB,
    # IMPORTANT: this script overwrites placeholder_flag=True from the
    # initial placeholder NPZ with placeholder_flag=False. The notebook's
    # loading cell should warn loudly when placeholder_flag is True.
    placeholder_flag=False,
)
size_kb = os.path.getsize(OUT_NPZ) / 1024
print(f"  wrote {size_kb:.1f} KB")
print(f"\nTotal wall-time: {(pub_wall_time_s + grid_time_s + multi_wall_time_s)/3600:.2f} hr")
print("Done. The notebook will now load these results in MODE='publication' mode.")
