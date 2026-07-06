#!/usr/bin/env python3
"""Write a placeholder f1_part_6_publication.npz.

Run via ``f1_part_6_publication_offline.sh --write-placeholder`` (the
canonical entry point) — this script is a helper, not a public CLI.

The numbers are *structurally plausible* but not real: a small bias is
added to the live-mode surrogate's optimum so the publication-mode plot
in the notebook visibly differs from the live-mode plot. The
``placeholder_flag=True`` field triggers the notebook's "this is not
real SU2 output" banner.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "f1_part_6_publication.npz"

# ──────────────────────────────────────────────────────────────────────────
# Design space — must match the notebook §4 table.
# 15 design variables (chord, span, sweep, twist, flap1, flap2, dihedral,
# endplate_camber_1, endplate_camber_2, gurney_height, plus 5 deferred slots
# we keep for the offline-script to populate in the real run).
# ──────────────────────────────────────────────────────────────────────────
DESIGN_NAMES = np.array([
    "chord", "span", "sweep_deg", "twist_deg",
    "flap1_deg", "flap2_deg", "dihedral_deg",
    "endplate_camber_1", "endplate_camber_2", "gurney_mm",
    "wing_AoA_deg", "boundary_layer_trip_x",
    "leading_edge_radius_mult", "trailing_edge_thickness", "tip_washout_deg",
])
N_DESIGN = len(DESIGN_NAMES)

theta_init = np.array([
    0.30, 1.50, 18.0, -2.5,
    +5.0, +12.0, 0.0,
    0.02, 0.02, 8.0,
    +4.0, 0.05,
    1.0, 0.002, 0.0,
])

# Optimised design — pushed in the directions the live surrogate (notebook
# §5) drives the gradient: higher flap angles + more gurney + more wing AoA
# (the lap-time-opt wedge: trade L/D for downforce because corner-exit
# speed wins).
theta_opt = theta_init + np.array([
    +0.04, +0.10, +1.5, -1.2,
    +3.0, +4.0, +0.0,
    +0.01, +0.005, +4.0,
    +1.5, 0.0,
    +0.05, 0.0, +0.5,
])

# ──────────────────────────────────────────────────────────────────────────
# Per-iteration history — 8 L-BFGS-B steps + initial = 9 entries.
# ──────────────────────────────────────────────────────────────────────────
N_ITER = 9
rng = np.random.default_rng(20260517)

# Monotone-ish lap-time descent (publication-fidelity SU2 RANS would give
# numbers within ~5% of these magnitudes; the *shape* is what matters).
lap_init = 89.4   # s — full Part-1 LTS lap at the baseline wing
lap_opt = 87.6   # s — 1.8 s / 2.0% wedge at publication fidelity
lap_history = np.linspace(lap_init, lap_opt, N_ITER) + 0.04 * rng.standard_normal(N_ITER)
lap_history = np.minimum.accumulate(lap_history)
lap_history[0] = lap_init  # pin the start

# Aero coefficients per iteration. The wedge: C_L A goes UP and C_D A goes
# UP (the lap-time-opt direction). The L/D-max optimum would push C_L A
# slightly up and C_D A slightly down — we save both arrays for the
# notebook's head-to-head.
cla_history = np.linspace(3.50, 4.12, N_ITER) + 0.015 * rng.standard_normal(N_ITER)
cda_history = np.linspace(1.10, 1.34, N_ITER) + 0.005 * rng.standard_normal(N_ITER)
cop_history = np.linspace(0.45, 0.51, N_ITER) + 0.003 * rng.standard_normal(N_ITER)

# Design vector per iteration — linear interp between init and opt.
design_history = np.stack([
    theta_init + (theta_opt - theta_init) * (i / (N_ITER - 1))
    for i in range(N_ITER)
])

# Adjoint sensitivities — same shape, mostly decreasing magnitude as the
# optimum is approached.
adjoint_history = (
    0.03 * rng.standard_normal((N_ITER, N_DESIGN))
    * (1.0 - np.linspace(0.0, 0.8, N_ITER))[:, None]
)
# The dominant sensitivities at iter 0 — flap angles and gurney — match
# the live-surrogate direction.
adjoint_history[0, 4] = -0.18   # flap1 -> lap reduction
adjoint_history[0, 5] = -0.22   # flap2 -> lap reduction
adjoint_history[0, 9] = -0.11   # gurney -> lap reduction
adjoint_history[0, 10] = -0.15  # wing_AoA -> lap reduction

# ──────────────────────────────────────────────────────────────────────────
# Pareto head-to-head: L/D-max vs lap-time-opt.
# ──────────────────────────────────────────────────────────────────────────
# L/D-max optimum — minimises -C_L/C_D. Picks a lower-drag, lower-downforce
# wing than the lap-time-opt. Less corner speed, more straight-line speed.
theta_ld_opt = theta_init + np.array([
    +0.02, +0.05, +0.7, -0.5,
    +0.5, +1.0, +0.0,
    +0.002, +0.001, +0.5,
    +0.5, 0.0,
    +0.02, 0.0, +0.2,
])
cla_ld_opt = 3.65
cda_ld_opt = 1.13
lap_ld_opt = 88.2  # s — slower lap than lap-time-opt despite better L/D
ld_ratio_ld_opt = cla_ld_opt / cda_ld_opt
ld_ratio_lt_opt = cla_history[-1] / cda_history[-1]

# ──────────────────────────────────────────────────────────────────────────
# Wall-time accounting (offline machine: 16-core Linux, ~5M-cell mesh).
# ──────────────────────────────────────────────────────────────────────────
wall_time_per_iter_s = 60.0 * 60.0      # 1 hr per primal+adjoint
wall_time_total_s = 8 * wall_time_per_iter_s + 2 * 60 * 60  # +rendering
n_su2_calls = 2 * N_ITER  # primal + adjoint per iter

np.savez(
    OUT,
    placeholder_flag=True,
    design_names=DESIGN_NAMES,
    theta_init=theta_init,
    theta_opt=theta_opt,
    theta_ld_opt=theta_ld_opt,
    design_history=design_history,
    adjoint_history=adjoint_history,
    lap_history=lap_history,
    cla_history=cla_history,
    cda_history=cda_history,
    cop_history=cop_history,
    lap_init=lap_init,
    lap_opt=lap_opt,
    lap_ld_opt=lap_ld_opt,
    cla_ld_opt=cla_ld_opt,
    cda_ld_opt=cda_ld_opt,
    ld_ratio_ld_opt=ld_ratio_ld_opt,
    ld_ratio_lt_opt=ld_ratio_lt_opt,
    wall_time_per_iter_s=wall_time_per_iter_s,
    wall_time_total_s=wall_time_total_s,
    n_su2_calls=n_su2_calls,
    notes=(
        "PLACEHOLDER values produced by media/f1_part_6_publication_offline.sh "
        "--write-placeholder. Replace by running the script without that flag "
        "on a Linux machine with SU2 v8.5.0 + OpenVSP + Blender + ParaView "
        "installed."
    ),
)
print(f"Wrote placeholder NPZ to {OUT}")
print(f"  Per-iter history shape: design={design_history.shape}, lap={lap_history.shape}")
print(f"  lap_init={lap_init:.3f}s  lap_opt={lap_opt:.3f}s  "
      f"wedge={lap_init - lap_opt:+.3f}s ({(lap_init-lap_opt)/lap_init*100:+.2f}%)")
print(f"  L/D-max optimum:    lap={lap_ld_opt:.3f}s  L/D={ld_ratio_ld_opt:.2f}")
print(f"  Lap-time optimum:   lap={lap_opt:.3f}s     L/D={ld_ratio_lt_opt:.2f}")
print(f"  Lap-time-vs-LDmax wedge: {lap_ld_opt - lap_opt:+.3f}s")
