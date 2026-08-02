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
