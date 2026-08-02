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

**Honesty:** no CFD runs here, ever. The default back end is a clearly-labelled
*engineering-correlation stand-in* (bluff-body cross-flow drag + a ~v^0.9
forced-convection power law, with scatter to mimic CFD noise), and the
``OpenFOAMBackend`` is a **seam, not an implementation**: its ``evaluate`` is
deliberately unimplemented so the pipeline can never silently present solver
output it does not have. The ROM-fit, Sobol analysis, and differentiable
surrogate are real and run here -- their *numbers* inherit the stand-in's
physics and become meaningful only when the seam is filled with real cases.
``write_openfoam_case`` emits a case *skeleton* (see
``media/openfoam_ebike_case/``): the numerics dictionaries are complete, but
there is no bike geometry, so it is a starting point for snappyHexMesh work,
not a runnable case.

**Units note:** the cooling output "h" is an *effective conductance* h*A in
W/K for the enclosed battery/motor volume (the quantity the vehicle model's
cooling link consumes), not a bare surface coefficient in W/m^2/K. A bare
surface at 8 m/s would give ~40 W/m^2/K (~6 W/K over the 0.15 m^2 case); the
enclosure path is what limits the real number to ~0.5 W/K, consistent with
the vehicle model's analytic cooling map.
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
    """Engineering-correlation stand-in for 3-D CFD. NOT CFD — a physical
    placeholder so the pipeline is exercisable: bluff-body cross-flow drag
    (yaw raises the effective area; CdA itself is Reynolds-independent over
    the bike's 3-12 m/s envelope, so there is deliberately *no* speed term)
    and an effective cooling conductance h*A [W/K] with a ~v^0.9 power law
    (near the turbulent-forced-convection Re^0.8 exponent), with ~1% scatter
    (the repeat-run spread of a well-converged steady-RANS DOE).

    Every number downstream of this backend is a property of these two
    hand-written formulas over the sampled envelope -- the Sobol analysis
    will faithfully recover the coefficients typed below, which is a check
    of the *pipeline*, not a finding about e-bike aerodynamics."""

    label = "engineering-correlation stand-in (NOT CFD)"

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def evaluate_deterministic(self, v_mps, yaw_deg):
        """The noise-free correlation — the stand-in's "truth". Available only
        because the backend is synthetic; with real CFD the analogous check is
        repeat runs at the same operating point."""
        yaw = np.radians(yaw_deg)
        CdA = 0.80 * (1.0 + 0.9 * np.sin(yaw) ** 2)                            # m^2
        h = (0.15 + 0.045 * np.abs(v_mps) ** 0.9) * (1.0 + 0.15 * np.abs(np.sin(yaw)))  # W/K (h*A)
        return float(CdA), float(h)

    def evaluate(self, v_mps, yaw_deg):
        CdA, h = self.evaluate_deterministic(v_mps, yaw_deg)
        CdA *= 1.0 + 0.01 * self.rng.randn()
        h *= 1.0 + 0.01 * self.rng.randn()
        return float(CdA), float(h)


class OpenFOAMBackend:
    """The seam where real CFD plugs in — **not an implementation**. Filling
    it means: write a case per DOE point (geometry via snappyHexMesh over a
    bike+rider STL), run ``blockMesh``/``snappyHexMesh``/``simpleFoam`` (and
    ``chtMultiRegionFoam`` for the cooling map), then parse
    ``postProcessing/forceCoeffs/0/coefficient.dat`` for Cd and integrate the
    wall heat flux for h*A. ``evaluate`` raises until someone does that work,
    so the pipeline can never silently present solver output it does not
    have. Opt in explicitly with ``EBIKE_USE_OPENFOAM=1`` once implemented."""

    label = "OpenFOAM simpleFoam / chtMultiRegionFoam"

    def __init__(self, case_dir=CASE_DIR):
        if shutil.which("simpleFoam") is None:
            raise RuntimeError("OpenFOAM (simpleFoam) not on PATH")
        self.case_dir = case_dir

    def evaluate(self, v_mps, yaw_deg):  # pragma: no cover - needs OpenFOAM
        raise NotImplementedError(
            "OpenFOAMBackend is a seam: implement case generation + parsing "
            "(see class docstring), then set EBIKE_USE_OPENFOAM=1")


def select_backend():
    """Return the CFD backend. The stand-in is the default even when OpenFOAM
    is installed: OpenFOAMBackend.evaluate is unimplemented, so selecting it
    automatically would crash the DOE mid-run. Set ``EBIKE_USE_OPENFOAM=1``
    to opt in once the seam is filled."""
    if os.environ.get("EBIKE_USE_OPENFOAM") == "1":
        return OpenFOAMBackend()
    return CorrelationBackend()


# ===========================================================================
# Design of experiments + ROM fit
# ===========================================================================
# DOE envelope, justified from riding conditions rather than picked freely
# (the Sobol variance split depends entirely on these ranges):
#   speed: the *apparent* wind speed. The bike itself covers ~3-9 m/s
#     (11 km/h climb to 32 km/h descent); ambient wind up to ~3 m/s head-on
#     extends the envelope to ~12 m/s.
#   yaw: apparent-wind angle. At 25 km/h (6.9 m/s), a 2.5 m/s crosswind gives
#     atan(2.5/6.9) ~ 20 deg; larger angles occur only at low riding speed
#     where aero hardly matters.
V_RANGE = (3.0, 12.0)     # m/s apparent wind
YAW_RANGE = (-20.0, 20.0)  # deg


def drag_power(CdA, v_mps, rho=RHO):
    """Aerodynamic power the rider/motor must supply: 0.5*rho*CdA*v^3 [W].

    This -- not CdA -- is the quantity the vehicle actually pays for, and the
    two have very different sensitivities: CdA is a Reynolds-independent
    *coefficient* that here varies only with yaw, while the power it implies
    carries a v^3. A sensitivity study of the coefficient and a sensitivity
    study of the load are different questions with different answers."""
    return 0.5 * rho * CdA * np.asarray(v_mps) ** 3


def run_doe(backend, n_v=9, n_yaw=7):
    v = np.linspace(*V_RANGE, n_v)
    yaw = np.linspace(*YAW_RANGE, n_yaw)
    V, Y = np.meshgrid(v, yaw, indexing="ij")
    CdA = np.empty_like(V)
    H = np.empty_like(V)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            CdA[i, j], H[i, j] = backend.evaluate(V[i, j], Y[i, j])
    return v, yaw, V, Y, CdA, H


def sample_holdout(backend, n=15, seed=123):
    """Random held-out points inside the DOE envelope, for validating the
    surrogates on data they were not fit to. Returns noisy samples plus, when
    the backend exposes it, the noise-free truth at the same points."""
    rng = np.random.RandomState(seed)
    v = rng.uniform(*V_RANGE, size=n)
    yaw = rng.uniform(*YAW_RANGE, size=n)
    cda = np.empty(n)
    h = np.empty(n)
    cda_det = np.full(n, np.nan)
    h_det = np.full(n, np.nan)
    for i in range(n):
        cda[i], h[i] = backend.evaluate(v[i], yaw[i])
        if hasattr(backend, "evaluate_deterministic"):
            cda_det[i], h_det[i] = backend.evaluate_deterministic(v[i], yaw[i])
    return np.column_stack([v, yaw]), cda, h, cda_det, h_det


def _r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def fit_roms(V, Y, CdA, H, holdout=None):
    """Fit (1) a PCE for Sobol sensitivity, (2) a differentiable RBF block, for
    each output. If ``holdout = (X_h, cda_h, h_h)`` is given, also report
    held-out R^2 -- the number that actually measures surrogate quality (the
    RBF is a near-interpolant, so its in-sample R^2 is ~1 by construction and
    says nothing)."""
    X = np.column_stack([V.ravel(), Y.ravel()])
    dists = [("uniform", *V_RANGE), ("uniform", *YAW_RANGE)]

    out = {}
    for k, (name, Z) in enumerate((("CdA", CdA), ("h", H))):
        z = Z.ravel()
        pce = fit_pce(X, z, dists, order=3)
        sob = pce.sobol_indices()
        rbf = fit_rbf(X, z, kernel="multiquadric", epsilon=8.0, smoothing=1e-8)
        pred = np.asarray(rbf.predict(X)).ravel()
        entry = dict(pce=pce, rbf=rbf,
                     sobol_first=np.asarray(sob["first_order"]),
                     sobol=np.asarray(sob["total"]),
                     r2_insample=_r2(pred, z))
        if holdout is not None:
            X_h = holdout[0]
            z_h = holdout[1 + k]           # noisy held-out samples
            z_det = holdout[3 + k]         # noise-free truth (synthetic only)
            entry["r2_holdout_rbf"] = _r2(np.asarray(rbf.predict(X_h)).ravel(), z_h)
            entry["r2_holdout_pce"] = _r2(np.asarray(pce.predict(X_h)).ravel(), z_h)
            if np.isfinite(z_det).all():
                entry["r2_truth_rbf"] = _r2(np.asarray(rbf.predict(X_h)).ravel(), z_det)
                entry["r2_truth_pce"] = _r2(np.asarray(pce.predict(X_h)).ravel(), z_det)
        out[name] = entry
    return out


def write_openfoam_case(case_dir=CASE_DIR, v_ref=8.0):
    """Emit a bluff-body external-aero simpleFoam case **skeleton**. The
    numerics dictionaries (fvSchemes/fvSolution, k-omega-SST fields) are
    complete, but there is deliberately **no bike geometry**: the mesh is an
    empty tunnel, and the ``bike`` patch that ``forceCoeffs`` and ``0/U``
    reference only comes into existence after a snappyHexMesh pass over a
    bike+rider STL (README step 1). Running the skeleton as-is therefore
    fails at mesh time -- by design: an empty tunnel would "converge" to
    Cd ~ 0 and quietly poison the ROM. This is a starting point for real
    case work, not a runnable case."""
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

    w("system/fvSchemes", """
/* Standard steady-RANS schemes: bounded upwind-biased convection. */
ddtSchemes    { default steadyState; }
gradSchemes   { default Gauss linear; }
divSchemes    {
    default                        none;
    div(phi,U)                     bounded Gauss linearUpwind grad(U);
    div(phi,k)                     bounded Gauss upwind;
    div(phi,omega)                 bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))  Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist      { method meshWave; }
""")
    w("system/fvSolution", """
/* SIMPLE with standard relaxation; residual controls stop the run when
   converged instead of blindly hitting endTime. */
solvers {
    p     { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE {
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 1e-4; U 1e-5; "(k|omega)" 1e-5; }
}
relaxationFactors {
    equations { U 0.7; "(k|omega)" 0.7; }
}
""")
    # k-omega-SST turbulence ICs: freestream turbulence intensity ~5%,
    # mixing length ~0.1 m. Wall treatment is via wall functions (the
    # background mesh is far too coarse for y+ ~ 1 wall resolution --
    # snappyHexMesh boundary layers are where that battle is fought).
    k_fs = 1.5 * (0.05 * v_ref) ** 2
    om_fs = k_fs ** 0.5 / (0.09 ** 0.25 * 0.1)
    w("0/k", f"""
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_fs:.4g};
boundaryField {{
  inlet  {{ type fixedValue; value uniform {k_fs:.4g}; }}
  outlet {{ type inletOutlet; inletValue uniform {k_fs:.4g}; value uniform {k_fs:.4g}; }}
  bike   {{ type kqRWallFunction; value uniform {k_fs:.4g}; }}
  walls  {{ type slip; }}
}}
""")
    w("0/omega", f"""
dimensions [0 0 -1 0 0 0 0];
internalField uniform {om_fs:.4g};
boundaryField {{
  inlet  {{ type fixedValue; value uniform {om_fs:.4g}; }}
  outlet {{ type inletOutlet; inletValue uniform {om_fs:.4g}; value uniform {om_fs:.4g}; }}
  bike   {{ type omegaWallFunction; value uniform {om_fs:.4g}; }}
  walls  {{ type slip; }}
}}
""")
    w("0/nut", """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField {
  inlet  { type calculated; value uniform 0; }
  outlet { type calculated; value uniform 0; }
  bike   { type nutkWallFunction; value uniform 0; }
  walls  { type slip; }
}
""")
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
# OpenFOAM e-bike aero/CHT case (SKELETON — not runnable as-is)

Starting point for the real CFD back end of `ebike_openfoam_aero_rom.py`.
The numerics are complete (`fvSchemes`, `fvSolution` with residual controls,
k-omega-SST fields with wall functions), but **there is no bike geometry**:
`blockMeshDict` describes an empty 20 x 4 x 3 m tunnel, and the `bike` patch
referenced by `forceCoeffs`, `0/U` and the turbulence fields only exists
after the snappyHexMesh step below. Do not "fix" that by deleting the patch
references: an empty tunnel converges happily to Cd ~ 0 and would poison the
downstream ROM with a plausible-looking zero.

**To make it a real case:**
1. Geometry: put a watertight bike+rider STL at `constant/triSurface/bike.stl`,
   run `surfaceFeatureExtract`, write a `snappyHexMeshDict` with boundary
   layers on the `bike` surface (this is where the y+ battle is fought; the
   shipped fields assume wall functions, y+ ~ 30-300), then
   `blockMesh && snappyHexMesh -overwrite`.
2. Domain checks before trusting any number: blockage = Aref / tunnel
   cross-section = 0.6 / 12 = 5% — at the guideline ceiling; widen the tunnel
   or correct for blockage. Verify mesh independence (three refinements) and
   that residual controls, not endTime, terminate the run.
3. Drag map: `simpleFoam`; read `Cd` from
   `postProcessing/forceCoeffs/0/coefficient.dat`; `CdA = Cd * Aref`.
4. Convection map: add a solid region (battery/motor) and switch to
   `chtMultiRegionFoam`; integrate wall heat flux over the component surface,
   divide by (T_surface - T_air). That yields h*A in W/K for the *exposed*
   surface — if the component sits inside a frame bag/casing, the enclosure
   dominates and must be part of the solid region, otherwise the CHT number
   will exceed the vehicle model's effective conductance by ~10x.
5. Yaw sweep: rotating only the inlet `U` vector is NOT sufficient in this
   channel (the `slip` side walls would fight a yawed freestream). Either
   rotate the geometry per yaw point, or switch the side patches to
   inlet/outlet pairs consistent with the yawed freestream.
6. Wire the sweep into `OpenFOAMBackend.evaluate` (case-per-point, parse
   results), then run with `EBIKE_USE_OPENFOAM=1`.
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

    holdout = sample_holdout(backend, n=15)
    roms = fit_roms(V, Y, CdA, H, holdout=holdout)
    print("\n  ROM fit (jaxonomy.library.rom); held-out R^2 is the honest one")
    print("  (the RBF near-interpolates its training set, so in-sample ~ 1):")
    for name in ("CdA", "h"):
        r = roms[name]
        print(f"    {name:4s}: R^2 in-sample RBF = {r['r2_insample']:.5f}   "
              f"held-out RBF = {r['r2_holdout_rbf']:.3f}  PCE = {r['r2_holdout_pce']:.3f}")
        if "r2_truth_rbf" in r:
            print(f"          vs noise-free truth: RBF = {r['r2_truth_rbf']:.3f}  "
                  f"PCE = {r['r2_truth_pce']:.3f}  (synthetic backend only)")
        print(f"          Sobol [speed, yaw]: first = [{r['sobol_first'][0]:.2f}, {r['sobol_first'][1]:.2f}]"
              f"  total = [{r['sobol'][0]:.2f}, {r['sobol'][1]:.2f}]")
    print("  CdA's held-out R^2 is noise-limited by construction: over this")
    print("  envelope its deterministic (yaw) variance is only a few times the")
    print("  1% run scatter, so a chunk of what any surrogate sees is noise.")
    print("  More DOE points, not a fancier surrogate, is the remedy.")
    print("  NOTE: with the stand-in backend the Sobol splits are recovered")
    print("  from the hand-written correlation over this envelope -- a pipeline")
    print("  check, not an aerodynamic finding.")

    # Differentiability of the embeddable surrogate: both inputs, FD-verified.
    rbf_cda = roms["CdA"]["rbf"]

    def cda_at(vq, yq):
        return float(jnp.squeeze(rbf_cda.predict(jnp.array([vq, yq]).reshape(1, 2))))

    g_v = float(jax.grad(lambda x: jnp.squeeze(rbf_cda.predict(jnp.array([x, 5.0]).reshape(1, 2))))(8.0))
    g_y = float(jax.grad(lambda y: jnp.squeeze(rbf_cda.predict(jnp.array([8.0, y]).reshape(1, 2))))(5.0))
    eps = 1e-3
    fd_v = (cda_at(8.0 + eps, 5.0) - cda_at(8.0 - eps, 5.0)) / (2 * eps)
    fd_y = (cda_at(8.0, 5.0 + eps) - cda_at(8.0, 5.0 - eps)) / (2 * eps)
    print(f"\n  differentiable block at (8 m/s, 5 deg), AD vs central FD:")
    print(f"    d(CdA)/d(speed) = {g_v:+.5f} vs FD {fd_v:+.5f}  (|diff| = {abs(g_v-fd_v):.1e})")
    print(f"    d(CdA)/d(yaw)   = {g_y:+.5f} vs FD {fd_y:+.5f}  (|diff| = {abs(g_y-fd_y):.1e})")

    # The maps the notebook embeds into the vehicle model (via the *surrogate*,
    # not fresh noisy backend draws)
    cda_ref = cda_at(8.0, 0.0)
    h_ref = float(np.asarray(roms["h"]["rbf"].predict(np.array([[8.0, 0.0]]))).squeeze())
    print(f"\n  -> surrogate at 8 m/s, 0 deg: CdA = {cda_ref:.3f} m^2 (replaces EbikeConfig.CdA),")
    print(f"     h*A = {h_ref:.3f} W/K (effective conductance; feeds SurrogateCooling,")
    print(f"     as in ebike_thermal_rom.py)")

    case = write_openfoam_case()
    print(f"  -> wrote OpenFOAM case SKELETON (needs geometry; see its README) "
          f"to {os.path.relpath(case, HERE)}/")

    # --- plots: the two surrogate maps ---
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name, Z, unit in ((axs[0], "CdA", CdA, "m$^2$"),
                              (axs[1], "h·A", H, "W/K, effective conductance")):
        cf = ax.contourf(V, Y, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label=f"{name} ({unit})")
        ax.set_xlabel("apparent wind speed (m/s)"); ax.set_ylabel("yaw / crosswind (deg)")
        s = roms["CdA" if name == "CdA" else "h"]["sobol"]
        ax.set_title(f"{name}(v, yaw) [STAND-IN, not CFD] — Sobol(total): "
                     f"speed {s[0]:.0%}, yaw {s[1]:.0%}")
    fig.suptitle(f"E-bike aero/thermal surrogate maps  [{backend.label}]",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"  -> wrote {os.path.relpath(PNG, HERE)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
