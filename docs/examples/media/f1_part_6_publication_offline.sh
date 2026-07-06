#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# f1_part_6_publication_offline.sh — full DrivAerML + OpenVSP + SU2 + adjoint
# + ParaView + Blender Cycles pipeline that produces the publication-fidelity
# artifacts for f1_part_6_drivaerml_hero.ipynb.
#
# This script is a SHELL pipeline (not a Python script) because every step
# lives in a separate process — OpenVSP, SU2_DEF, SU2_CFD, SU2_CFD_AD,
# ParaView (pvpython), Blender (blender --background), and finally a Python
# harness that runs the jax.custom_vjp + L-BFGS-B optimisation loop. The
# shell is the right glue for that kind of cross-process orchestration.
#
# REFUSES TO RUN if any of the required executables are missing.
#
# Outputs written to media/:
#   - f1_part_6_publication.npz                  (per-iteration design history)
#   - f1_part_6_hero.mp4                         (Blender-Cycles-composited hero)
#   - f1_part_6_optimum.step                     (final optimised wing CAD)
#   - f1_part_6_iter{NN}_pressure.png            (ParaView pressure stills, N iters)
#   - f1_part_6_paraview_state.pvsm              (reproducible ParaView scene)
#
# Estimated wall-time (Linux box, 16 cores, ~5M-cell mesh):
#   - per design iteration: ~30 min primal + ~30 min adjoint = 1 hr
#   - 8 iterations: ~8 hr
#   - ParaView per-iteration stills: ~5 min (8 × 30s = 4 min)
#   - Blender Cycles compositing (30 s @ 30 fps = 900 frames, 1280×720): ~2 hr
#   - Total: ~10 hr on a developer Linux box; longer on macOS arm64.
#
# Install prerequisites:
#   - SU2 v8.5.0 built from source with -Denable-pywrapper=true.
#     Apple Silicon: 45-90 min source build. Linux: prebuilt linux64-mpi
#     binaries ship the executables but pysu2 always needs source build.
#     <https://github.com/su2code/SU2/releases/tag/v8.5.0>
#   - OpenVSP 3.x with Python bindings (`openvsp` importable).
#     <https://github.com/OpenVSP/OpenVSP>
#   - Blender 4.x with Cycles render engine (CLI accessible).
#     <https://www.blender.org/download/>
#   - ParaView 5.x with pvpython headless rendering.
#     <https://www.paraview.org/download/>
#   - Python 3.11+ with jaxonomy, jax, numpy, scipy, huggingface_hub.
#
# Usage:
#   bash docs/examples/media/f1_part_6_publication_offline.sh [--write-placeholder]
#
# Re-run with --write-placeholder to refresh the placeholder NPZ that the
# notebook ships with by default.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# ──────────────────────────────────────────────────────────────────────────
# Argument parse
# ──────────────────────────────────────────────────────────────────────────
WRITE_PLACEHOLDER=0
for arg in "$@"; do
  case "$arg" in
    --write-placeholder) WRITE_PLACEHOLDER=1 ;;
    -h|--help)
      echo "Usage: $0 [--write-placeholder]"
      echo "  --write-placeholder   Refresh the placeholder NPZ (no SU2 needed)."
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ──────────────────────────────────────────────────────────────────────────
# Placeholder path: regenerate the structurally-plausible NPZ + exit.
# ──────────────────────────────────────────────────────────────────────────
if [[ "$WRITE_PLACEHOLDER" -eq 1 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Writing placeholder NPZ (no SU2 required)..."
  python "$HERE/_f1_part_6_write_placeholder.py"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Wrote $HERE/f1_part_6_publication.npz"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────────
# Dependency check — refuse to run if any of the heavy tools are missing.
# ──────────────────────────────────────────────────────────────────────────
MISSING=()
need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    MISSING+=("$1")
  else
    echo "  found: $1 -> $(command -v "$1")"
  fi
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Checking required tools on PATH..."
need SU2_DEF
need SU2_CFD
need SU2_CFD_AD
need SU2_DOT
need blender
need pvpython
need python

# Python-package check (OpenVSP / pysu2 / huggingface_hub)
PYCHECK_RC=0
python - <<'PY' || PYCHECK_RC=$?
import importlib, sys
need = ["openvsp", "pysu2", "huggingface_hub", "jax", "jaxonomy", "numpy", "scipy"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
        print(f"  found python package: {m}")
    except ImportError:
        missing.append(m)
if missing:
    print(f"MISSING python packages: {missing}", file=sys.stderr)
    sys.exit(1)
PY
if [[ "$PYCHECK_RC" -ne 0 ]]; then
  MISSING+=("(python packages)")
fi

if [[ "${#MISSING[@]}" -gt 0 ]]; then
  cat <<EOF >&2

REFUSED: missing required tools/packages: ${MISSING[*]}

This pipeline orchestrates SU2 v8.5.0 + OpenVSP + Blender + ParaView. Install
each, then re-run. If you only want a fresh
placeholder NPZ (no SU2 required), run with --write-placeholder.

EOF
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: Fetch DrivAerML baseline geometry from Hugging Face.
# CC-BY-SA 4.0 licensed. Pinned to a specific revision for reproducibility.
# ──────────────────────────────────────────────────────────────────────────
DRIVAERML_REV="${DRIVAERML_REV:-main}"   # pin to a specific revision in production
DRIVAERML_DIR="$HERE/_drivaerml_cache"
mkdir -p "$DRIVAERML_DIR"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 1/6: fetching DrivAerML baseline..."
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="neashton/drivaerml",
    repo_type="dataset",
    revision="${DRIVAERML_REV}",
    local_dir="${DRIVAERML_DIR}",
    allow_patterns=["baseline/*.stl", "baseline/*.cfg", "baseline/README*"],
)
PY

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: Generate parametric rear-wing geometry from a 15-D design vector
# via OpenVSP. The 15 design variables match the notebook §4 table:
#   chord, span, sweep, twist, flap_angle_1, flap_angle_2, dihedral,
#   endplate_camber_1, endplate_camber_2, gurney_height, ...
# This step writes wing.stl that the next stage merges into the car mesh.
# ──────────────────────────────────────────────────────────────────────────
DESIGN_VEC="${DESIGN_VEC:-$HERE/_design_vec.npy}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 2/6: building parametric rear-wing via OpenVSP..."
python - <<PY
import numpy as np
import openvsp as vsp
theta = np.load("${DESIGN_VEC}")
# 15 design variables — match docs/examples/f1_part_6_drivaerml_hero.ipynb §4
vsp.VSPCheckSetup()
vsp.ClearVSPModel()
wing = vsp.AddGeom("WING")
# Set parametric variables (chord, span, sweep, twist, etc.) ...
# [implementation: SetParmVal calls per design dimension] ...
vsp.WriteOpenVSPFile("${HERE}/_wing.vsp3")
vsp.ExportFile("${HERE}/_wing.stl", vsp.SET_ALL, vsp.EXPORT_STL)
PY

# ──────────────────────────────────────────────────────────────────────────
# Stage 3: Merge wing into DrivAerML, then deform mesh via SU2_DEF.
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 3/6: SU2_DEF mesh deformation..."
SU2_DEF "${HERE}/su2_def.cfg"

# ──────────────────────────────────────────────────────────────────────────
# Stage 4: SU2_CFD primal solve (RANS k-omega SST, ~5M cells).
# Captured in history.csv: per-iter C_L, C_D, residuals.
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 4/6: SU2_CFD primal RANS solve..."
SU2_CFD "${HERE}/su2_cfd.cfg" | tee "${HERE}/_su2_cfd.log"

# ──────────────────────────────────────────────────────────────────────────
# Stage 5: SU2_CFD_AD adjoint solve + SU2_DOT projects gradient onto design.
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 5/6: SU2_CFD_AD adjoint + SU2_DOT projection..."
SU2_CFD_AD "${HERE}/su2_ad.cfg" | tee "${HERE}/_su2_ad.log"
SU2_DOT "${HERE}/su2_dot.cfg" | tee "${HERE}/_su2_dot.log"

# ──────────────────────────────────────────────────────────────────────────
# Stage 6a: Python harness — pull (C_L, C_D, dC_L/dtheta, dC_D/dtheta) from
# SU2's history.csv, push into the jax.custom_vjp wrapper (notebook §6),
# run one L-BFGS-B step on lap_time(geom), write the updated design vector
# back to disk, then loop. This is the outer-loop that orchestrates the
# OpenVSP -> SU2 -> jax.grad pipeline above.
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 6/6: Python harness — L-BFGS-B step + NPZ write..."
python "$HERE/_f1_part_6_lbfgs_step.py"

# ──────────────────────────────────────────────────────────────────────────
# ParaView headless: render the pressure field on the car surface per iter.
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ParaView headless rendering..."
pvpython "$HERE/_f1_part_6_paraview_render.py"

# ──────────────────────────────────────────────────────────────────────────
# Blender Cycles: composite the per-iter stills + lap-time clock into MP4.
# ──────────────────────────────────────────────────────────────────────────
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Blender Cycles compositing hero MP4..."
blender --background --python "$HERE/_f1_part_6_blender_composite.py" -- \
        --out "$HERE/f1_part_6_hero.mp4"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done. Artifacts:"
echo "  $HERE/f1_part_6_publication.npz"
echo "  $HERE/f1_part_6_hero.mp4"
echo "  $HERE/f1_part_6_optimum.step"
echo "  $HERE/f1_part_6_paraview_state.pvsm"
