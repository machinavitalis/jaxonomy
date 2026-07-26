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
