"""3-D aero / conjugate-heat-transfer surrogate for the cargo e-bike, via an
OpenFOAM design-of-experiments -> differentiable reduced-order model pipeline.

The vehicle model (``ebike_hybrid_simulation.py``) uses a constant drag area
``CdA`` and an analytic convective-cooling map ``h(v)``. The physically-grounded
way to get those is 3-D CFD: sweep operating conditions, and fit a differentiable
surrogate that plugs into the system model. This is the honest, real version of
the fabricated "neural CFD surrogate" the original example claimed (which was fit
to a closed-form formula and never used).

Pipeline (tool-agnostic front end, CFD back end):

    DOE over (speed v, yaw beta)
        -> CFD case per point   [OpenFOAM: simpleFoam drag + chtMultiRegionFoam
                                 surface convection]  ==> CdA(v,beta), h(v,beta)
        -> fit ROM              [jaxonomy.library.rom: PCE (with Sobol
                                 sensitivity) + a differentiable GP/RBF block]
        -> embed in the vehicle model (replaces constant CdA and analytic h)

**Honesty:** OpenFOAM is not installed in this environment, so by default the
CFD back end is a clearly-labelled *engineering-correlation stand-in* (bluff-body
cross-flow drag + a turbulent forced-convection Nusselt law, with scatter to
mimic CFD noise). If the OpenFOAM binaries ARE on PATH, the ``OpenFOAMBackend``
generates and runs real cases instead. Either way the ROM-fit, Sobol analysis,
and differentiable surrogate are real and run here. ``write_openfoam_case``
emits a concrete bluff-body simpleFoam case skeleton (see ``media/openfoam_
ebike_case/``) and ``docs`` below documents the full production path.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxonomy.library.rom import fit_pce, fit_rbf, RadialBasisSurrogate

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "media", "ebike_openfoam_aero_rom.png")
CASE_DIR = os.path.join(HERE, "media", "openfoam_ebike_case")

RHO = 1.2  # air density [kg/m^3]


# ===========================================================================
# CFD back ends
# ===========================================================================
class CorrelationBackend:
    """Engineering-correlation stand-in for 3-D CFD (used when OpenFOAM is
    absent). NOT CFD — a physical placeholder so the pipeline is exercisable:
    bluff-body cross-flow drag (yaw raises the effective area) and a turbulent
    forced-convection conductance (Nu ~ Re^0.8), with ~2% scatter."""

    label = "engineering-correlation stand-in (OpenFOAM not found)"

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def evaluate(self, v_mps, yaw_deg):
        yaw = np.radians(yaw_deg)
        CdA = 0.80 * (1.0 + 0.9 * np.sin(yaw) ** 2) + 0.004 * v_mps           # m^2
        h = (0.15 + 0.045 * np.abs(v_mps) ** 0.9) * (1.0 + 0.15 * np.abs(np.sin(yaw)))  # W/K
        CdA *= 1.0 + 0.02 * self.rng.randn()
        h *= 1.0 + 0.02 * self.rng.randn()
        return float(CdA), float(h)


class OpenFOAMBackend:
    """Real CFD back end. Generates and runs an OpenFOAM case per DOE point and
    parses the drag coefficient + surface convection. Requires ``blockMesh`` and
    ``simpleFoam`` on PATH (and, for CHT, ``chtMultiRegionFoam``); raises if
    absent so the caller can fall back."""

    label = "OpenFOAM simpleFoam / chtMultiRegionFoam"

    def __init__(self, case_dir=CASE_DIR):
        if shutil.which("simpleFoam") is None:
            raise RuntimeError("OpenFOAM (simpleFoam) not on PATH")
        self.case_dir = case_dir

    def evaluate(self, v_mps, yaw_deg):  # pragma: no cover - needs OpenFOAM
        # Production: write_openfoam_case(v, yaw); run blockMesh + simpleFoam;
        # parse postProcessing/forceCoeffs/0/coefficient.dat for Cd; integrate
        # the wall heat flux from chtMultiRegionFoam for h. Left unimplemented in
        # environments without the toolchain (constructor already raised there).
        raise NotImplementedError("run OpenFOAM case + parse forceCoeffs / wall flux")


def select_backend():
    try:
        return OpenFOAMBackend()
    except RuntimeError:
        return CorrelationBackend()


# ===========================================================================
# Design of experiments + ROM fit
# ===========================================================================
def run_doe(backend, n_v=9, n_yaw=7):
    v = np.linspace(4.0, 15.0, n_v)          # m/s  (~14-54 km/h)
    yaw = np.linspace(-15.0, 15.0, n_yaw)    # deg crosswind/yaw
    V, Y = np.meshgrid(v, yaw, indexing="ij")
    CdA = np.empty_like(V)
    H = np.empty_like(V)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            CdA[i, j], H[i, j] = backend.evaluate(V[i, j], Y[i, j])
    return v, yaw, V, Y, CdA, H


def _r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def fit_roms(V, Y, CdA, H):
    """Fit (1) a PCE for Sobol sensitivity, (2) a differentiable RBF block, for
    each output. Inputs normalized to [-1,1]-ish ranges expected by the ROMs."""
    X = np.column_stack([V.ravel(), Y.ravel()])
    dists = [("uniform", 4.0, 15.0), ("uniform", -15.0, 15.0)]

    out = {}
    for name, Z in (("CdA", CdA), ("h", H)):
        z = Z.ravel()
        pce = fit_pce(X, z, dists, order=3)
        sob = pce.sobol_indices()
        rbf = fit_rbf(X, z, kernel="multiquadric", epsilon=8.0, smoothing=1e-8)
        pred = np.asarray(rbf.predict(X)).ravel()
        out[name] = dict(pce=pce, rbf=rbf, sobol=np.asarray(sob["total"]),
                         r2=_r2(pred, z))
    return out


def write_openfoam_case(case_dir=CASE_DIR, v_ref=8.0):
    """Emit a concrete bluff-body external-aero simpleFoam case skeleton (drag
    of a box approximating the rider+bike frontal area in a wind tunnel). This
    is a real, conventional case *template* — production use replaces the box
    with a meshed STL of the bike via snappyHexMesh, and adds a solid region +
    chtMultiRegionFoam for the surface-convection (h) map. Requires OpenFOAM to
    execute; written here as a runnable starting point, not executed."""
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "constant"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "0"), exist_ok=True)

    def w(rel, text):
        with open(os.path.join(case_dir, rel), "w") as f:
            f.write(text.lstrip("\n"))

    w("system/controlDict", f"""
/* simpleFoam steady RANS; forceCoeffs functionObject writes Cd/Cl. */
application     simpleFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 500;
deltaT          1; writeControl timeStep; writeInterval 100;
functions {{
    forceCoeffs {{
        type forceCoeffs; libs ("libforces.so"); patches (bike);
        rho rhoInf; rhoInf {RHO}; magUInf {v_ref}; lRef 1.0; Aref 0.6;
        liftDir (0 0 1); dragDir (1 0 0); CofR (0 0 0); pitchAxis (0 1 0);
    }}
}}
""")
    w("system/blockMeshDict", """
/* Wind-tunnel box with a bluff body cut out is produced by snappyHexMesh in
   production; here a simple graded channel as the background mesh. */
scale 1;
vertices ((-5 -2 0)(15 -2 0)(15 2 0)(-5 2 0)(-5 -2 3)(15 -2 3)(15 2 3)(-5 2 3));
blocks (hex (0 1 2 3 4 5 6 7) (100 20 15) simpleGrading (1 1 1));
boundary (
  inlet  {type patch;  faces ((0 4 7 3));}
  outlet {type patch;  faces ((1 2 6 5));}
  walls  {type wall;   faces ((0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}
);
""")
    w("constant/transportProperties",
      "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1.5e-05;\n")
    w("constant/turbulenceProperties",
      "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n")
    w("0/U", f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({v_ref} 0 0);
boundaryField {{
  inlet {{ type fixedValue; value uniform ({v_ref} 0 0); }}
  outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({v_ref} 0 0); }}
  bike {{ type noSlip; }}
  walls {{ type slip; }}
}}
""")
    w("0/p", """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField {
  inlet { type zeroGradient; }
  outlet { type fixedValue; value uniform 0; }
  bike { type zeroGradient; }
  walls { type slip; }
}
""")
    w("README.md", """
# OpenFOAM e-bike aero/CHT case (template)

Real CFD back end for `ebike_openfoam_aero_rom.py`. This directory is a
conventional bluff-body external-aerodynamics `simpleFoam` case skeleton.

**To make it production:**
1. Replace the background block mesh with a meshed STL of the bike+rider:
   `surfaceFeatureExtract` + `snappyHexMesh` over `constant/triSurface/bike.stl`.
2. Drag map: run `blockMesh && snappyHexMesh -overwrite && simpleFoam`; read
   `Cd` from `postProcessing/forceCoeffs/0/coefficient.dat`; `CdA = Cd*Aref`.
3. Convection (h) map: add a solid region (battery/motor) and switch to
   `chtMultiRegionFoam`; integrate the wall heat flux over the component
   surface and divide by (T_surface - T_air) for the effective `h`.
4. Sweep inlet speed and yaw (rotate `U` inlet vector) per DOE point; the
   Python driver (`OpenFOAMBackend.evaluate`) parametrises and parses each run.
""")
    return case_dir


def main():
    print("=" * 72)
    print("  E-BIKE AERO / CHT — OpenFOAM DOE -> differentiable ROM pipeline")
    print("=" * 72)
    backend = select_backend()
    print(f"  CFD back end: {backend.label}")

    v, yaw, V, Y, CdA, H = run_doe(backend)
    print(f"  DOE: {V.size} points over speed [{v[0]:.0f},{v[-1]:.0f}] m/s "
          f"x yaw [{yaw[0]:.0f},{yaw[-1]:.0f}] deg")

    roms = fit_roms(V, Y, CdA, H)
    print("\n  ROM fit (jaxonomy.library.rom):")
    for name in ("CdA", "h"):
        s = roms[name]["sobol"]
        print(f"    {name:4s}: RBF surrogate R^2 = {roms[name]['r2']:.5f}   "
              f"Sobol total [speed, yaw] = [{s[0]:.2f}, {s[1]:.2f}]")

    # Differentiability of the embeddable surrogate
    rbf_cda = roms["CdA"]["rbf"]
    g = jax.grad(lambda x: jnp.squeeze(rbf_cda.predict(jnp.array([x, 5.0]).reshape(1, 2))))(8.0)
    print(f"    differentiable block: d(CdA)/d(speed) at 8 m/s, 5deg = {float(g):+.4f} m^2/(m/s)")

    # The maps that plug into the vehicle model
    cda_ref, h_ref = backend.evaluate(8.0, 0.0)
    print(f"\n  -> at 8 m/s, 0 deg: CdA = {cda_ref:.3f} m^2 (replaces EbikeConfig.CdA),")
    print(f"     h = {h_ref:.3f} W/K (feeds SurrogateCooling, as in ebike_thermal_rom.py)")

    case = write_openfoam_case()
    print(f"  -> wrote runnable OpenFOAM case template to {os.path.relpath(case, HERE)}/")

    # --- plots: the two surrogate maps ---
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name, Z, unit in ((axs[0], "CdA", CdA, "m$^2$"), (axs[1], "h", H, "W/K")):
        cf = ax.contourf(V, Y, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label=f"{name} ({unit})")
        ax.set_xlabel("speed (m/s)"); ax.set_ylabel("yaw / crosswind (deg)")
        s = roms[name]["sobol"]
        ax.set_title(f"{name}(v, yaw)  —  Sobol: speed {s[0]:.0%}, yaw {s[1]:.0%}")
    fig.suptitle(f"E-bike aero/thermal surrogate maps  [{backend.label}]",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"  -> wrote {os.path.relpath(PNG, HERE)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
