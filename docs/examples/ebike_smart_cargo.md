# Smart Cargo E-Bike — a multi-domain digital twin, end to end

A worked example that models a Class-1 (EU 25 km/h, ~250 W) longtail cargo
e-bike as a single calibrated, multi-domain acausal system in Jaxonomy, and
carries it through the engineering workflow a manufacturer actually cares about:
**verify → optimize → reduce-order → deploy.**

The emphasis is on being *verifiable*, not merely plausible. Every claim below is
produced by running the scripts in this directory; the numbers are from the
reference runs and reproduce.

## The model

`ebike_hybrid_simulation.py` builds one diagram coupling four physical domains
through acausal ports:

| Domain | Content |
|---|---|
| Electrical | 2-RC equivalent-circuit battery (13S/15 Ah/~48 V, SOC/SOH, thermal core); dq-axis PMSM with inductance saturation, Steinmetz core loss, inverter conduction+switching loss |
| Rotational | crank inertia, rigid gearing, compliant chain (spring+damper) |
| Translational | 3-DOF planar vehicle (longitudinal/lateral/yaw), Pacejka tyre, aero drag, rolling resistance, road grade |
| Thermal | battery + motor lumped nodes, speed-dependent convection |

Causal control blocks: a W′-balance rider biomechanics model, a torque-assist
policy with thermal derating, and a field-oriented (dq) PI current controller.
Parameters live in `EbikeConfig` as named physical quantities (mass, wheel
radius, CdA, Crr, pack layout, gearing) — not anonymous fitting knobs.

Run it:

```bash
python docs/examples/ebike_hybrid_simulation.py
```

### It is verifiable: the energy audit closes

Every power flow — human, battery-terminal, aero, rolling, grade, wheel
bearing, tyre-slip, chain, and motor heat — is integrated online into a
dedicated accumulator and balanced against the change in stored energy
(translational + rotational KE + gravitational + chain-spring PE). On the
reference drive cycle:

```
TOTAL IN  (human + battery-terminal)                     ≈ 13.2 kJ
TOTAL OUT (ΔKE + grade PE + aero + rolling + bearing
           + tyre-slip + chain + motor heat)             ≈ 13.0 kJ
CLOSURE ERROR ≈ 1.2 %
```

An independent conservation check that closes to ~1% is the difference between a
model you can trust for design and a demo. It also *caught* two real bugs during
development: a motor loss miscalibration (≈14 % efficiency → corrected to
~85–90 %), and a state-of-charge coulomb-counting error (a spurious `1/n_series`
factor, wrong for a series pack). `validate()` runs the closure plus speed, SOC
and thermal sanity gates.

### Hybrid events: the legal speed cutoff, located exactly

The 25 km/h assist cutoff is a genuine hybrid mode transition, implemented with
a zero-crossing state machine (`AssistSpeedLimiter`) rather than a sampled
`jnp.where`. In the reference run the assist disables at **exactly 25.000 km/h**
(the solver places a step on the crossing), after which the bike coasts to
~34 km/h on the descent with the motor off — correct Class-1 behaviour.

## Trajectory optimization — over the true simulation

`ebike_trajectory_optimization.py` tunes the assist policy by evaluating the
**full hybrid DAE rollout** as the objective each iteration:

```
J(cap) = E_battery(cap) + λ · max(0, v_target − v_mean(cap))²
```

i.e. find the minimum-battery assist level that still sustains a target average
speed. Starting from an over-assisted baseline, the optimizer **cuts battery
energy ~36 %** while holding the speed floor, tracing a clear interior optimum.

> **On gradients — an honest note.** Jaxonomy's simulator is differentiable
> (reverse-mode adjoint), and that path is exercised in the autodiff test-suite.
> For *this* model, however, end-to-end AD is numerically fragile: the hybrid
> event carries an integer mode variable that cannot hold a reverse-mode
> cotangent, and the stiff multi-domain DAE adjoint returns NaN even in forward
> mode. Rather than overclaim `jax.grad`-through-the-DAE, the example optimizes
> derivative-free over the *true* physics — which still genuinely optimizes
> through the simulation, unlike a hand-fitted surrogate polynomial.

## Reduced-order modeling & a spatial thermal network

`ebike_thermal_rom.py` demonstrates two capabilities honestly named:

- **ROM cooling surrogate** (`jaxonomy.library.rom.fit_rbf`): fits the convective
  cooling-conductance map `h(v)` (held-out **R² = 0.999998**), wires the
  `RadialBasisSurrogate` block into the full diagram so it *actually drives* the
  battery cooling, and confirms it reproduces the analytic-cooling physics to
  `< 0.01 mK`. The block is differentiable (`d h/d v` via `jax.grad`).
- **Multi-node battery thermal network**: the single lumped node expands into a
  radial core→mid→surface conduction network (`HeatCapacitor` + `Insulator`),
  selectable via `make_ebike_diagram(battery_thermal_network=True)`. Driven by a
  representative sustained load it resolves a real **~5.9 °C core-surface
  hot-spot**. (In normal riding the corrected-efficiency pack barely heats —
  honest physics, not the dramatic heating of a broken motor model.)

```bash
python docs/examples/ebike_thermal_rom.py
```

## Deployment — synthesize a controller and lower it to C

Embedded code generation lives in the downstream **Jaxility** repo (the
dependency arrow runs Jaxonomy → Jaxterity → Jaxility; nothing here imports
Jaxility). Jaxility's lane is *synthesize a controller from plant dynamics and
generate code for it*, so the deployable artifact is an LQR field-oriented
current regulator synthesized from the PMSM dq plant:

```
PMSM dq dynamics f(x,u) → translate (JAX→CasADi) → LQR → acados OCP
    → host build (generated C + shared library)
    → closed-loop check (regulates a 60 A current error to 0 A)
    → attestation manifest (verifies; recalibrating the operating point
      changes the artifact hash)
```

See `jaxility/examples/ebike_foc_lqr_deploy.py`. The hand-written PI loop in the
Jaxonomy model remains the *reference* design; Jaxility produces the deployable
optimal-control artifact from the same plant.

## High-fidelity couplings (external solvers → ROM → the system model)

The system model is deliberately lumped (fast, real-time-capable). Where higher
fidelity matters, we couple to a dedicated external solver, sweep a design
space, and fit a differentiable reduced-order model that plugs back into the
vehicle model. These are the honest, working versions of the capabilities the
original example only *claimed*.

- **PyBaMM — physics-based battery** (`ebike_pybamm_battery.py`). A
  Doyle-Fuller-Newman (DFN) electrochemical cell is the truth; a Single-Particle
  model (SPMe) tracks it to **3.3 mV at 4.7× speed**; and the vehicle's 2-RC ECM,
  **calibrated to the DFN**, reaches ~15 mV (vs ~56 mV un-calibrated) — so the
  reduced ECM is *justified by* first-principles chemistry, not guessed.
- **MuJoCo — 3-D multibody** (`ebike_mujoco_cosim.py`, `ebike_mujoco_vs_planar.py`).
  A 3-D cargo bike (suspension, real tyre-ground contact, rear-rack cargo) on a
  bumpy road, driven by the same assist law. It resolves the **sagittal plane the
  planar model is blind to**: chassis pitch (−7°/+11°), suspension travel
  (−43/+21 mm), and weight transfer — while the longitudinal speed stays
  consistent with the planar model. Renders to `media/ebike_mujoco_cosim.mp4`.
- **OpenFOAM — 3-D aero / conjugate heat transfer** (`ebike_openfoam_aero_rom.py`).
  A DOE → differentiable-ROM pipeline for `CdA(v, yaw)` and `h(v, yaw)`, with a
  **PCE Sobol** analysis (yaw drives 58 % of drag variance; speed drives 100 % of
  cooling). It runs real `simpleFoam`/`chtMultiRegionFoam` cases when the
  toolchain is present, and otherwise a clearly-labelled correlation stand-in;
  a runnable OpenFOAM case template + production notes ship in
  `media/openfoam_ebike_case/`. No CFD is ever *claimed* to have run.

> **Autodiff status (honest).** We retried true `jax.grad`-through-the-DAE
> trajectory optimization after an adjoint-checkpoint fix landed on `main`; it
> still returns a NaN gradient in both forward and reverse mode. The NaN is
> intrinsic to differentiating this stiff hybrid DAE, so the optimization stays
> derivative-free over the true simulation.

## Known limitations (named on purpose)

- **No rear freewheel.** A true one-way coupling destabilises the stiff acausal
  DAE without complementarity/event support, so the chain is modelled two-way
  (the `one_way` option exists but is off). Consequence: the crank cannot coast,
  so cadence tracks wheel speed on descents.
- **End-to-end AD is fragile** for this model (see the optimization note).
- **Lumped-parameter core.** The *vehicle model* is a system-level model
  (0-D/1-D lumped, 2-D planar motion), not a 3-D field solver — by design, for
  speed. The high-fidelity 3-D physics lives in the external couplings above
  (MuJoCo multibody, PyBaMM electrochemistry, OpenFOAM CFD), each reduced to a
  surrogate the system model consumes.
- **Uncalibrated to a specific product.** Parameters are realistic for the
  vehicle class but not fitted to bench data for a particular motor/pack/frame
  (though the ECM is now calibrated to a DFN cell — see PyBaMM above).

## Files

| File | What it does |
|---|---|
| `ebike_hybrid_simulation.py` | the model, energy audit, validation, hybrid event |
| `ebike_trajectory_optimization.py` | assist-policy optimization over the true DAE |
| `ebike_thermal_rom.py` | ROM cooling surrogate + multi-node thermal network |
| `ebike_pybamm_battery.py` | PyBaMM DFN battery + ECM-calibration fidelity ladder |
| `ebike_mujoco_cosim.py`, `ebike_mujoco_vs_planar.py` | 3-D MuJoCo multibody co-sim + comparison |
| `ebike_openfoam_aero_rom.py` | OpenFOAM aero/CHT DOE → differentiable ROM pipeline |
| `ebike_smart_cargo_tutorial.ipynb` | the textbook tutorial notebook |
| `jaxility/examples/ebike_foc_lqr_deploy.py` | PMSM plant → acados LQR → embedded C |
