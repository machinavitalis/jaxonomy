# Smart Cargo E-Bike — a multi-domain digital twin, end to end

A five-part worked example that models a Class-1 (EU 25 km/h, ~250 W) longtail
cargo e-bike as a single calibrated, multi-domain acausal system in Jaxonomy,
and carries it through the engineering workflow a manufacturer actually cares
about: **verify → optimize → couple in higher fidelity → deploy.**

The emphasis is on being *verifiable*, not merely plausible. Every number below
is produced by running the scripts in this directory and reproduces. Where a
capability is stood in for rather than run (CFD), the notebooks say so in the
executed output, not only in a preamble.

## The series

| Part | Notebook | What it adds |
|---|---|---|
| 1 | `ebike_part1_smart_cargo.ipynb` | the multi-domain model, built v0→v3, verified by an energy audit |
| 2 | `ebike_part2_optimization.ipynb` | derivative-free optimization over the true DAE; penalty methods, noise floors and Sobol sensitivity done honestly |
| 3 | `ebike_part3_pybamm_battery.ipynb` | PyBaMM DFN → the 2-RC ECM Part 1 ships, calibrated, validated held-out, and written back |
| 4 | `ebike_part4_mujoco_multibody.ipynb` | MuJoCo sagittal-plane multibody: suspension, contact, pitch — quantified against the planar model |
| 5 | `ebike_part5_openfoam_cfd.ipynb` | the DOE → ROM pipeline by which a field solver would feed CdA and cooling maps into a system model |

## The model

`ebike_hybrid_simulation.py` builds one diagram coupling four physical domains
through acausal ports:

| Domain | Content |
|---|---|
| Electrical | 2-RC equivalent-circuit battery (13S3P, ~15 Ah/group, ~48 V, SOC/SOH, thermal core), RC parameters calibrated to a DFN cell in Part 3; dq-axis PMSM with inductance saturation, Steinmetz core loss, inverter conduction+switching loss |
| Rotational | crank inertia, rigid gearing, compliant chain (spring+damper) |
| Translational | 3-DOF planar vehicle (longitudinal/lateral/yaw), Pacejka tyre, aero drag, rolling resistance, road grade |
| Thermal | battery + motor lumped nodes, speed-dependent convection |

Causal control blocks: a W′-balance rider biomechanics model, a torque-assist
policy with thermal derating, and a field-oriented (dq) PI current controller
with back-EMF feedforward, back-calculation anti-windup, and gains scheduled on
the machine's saturation-aware inductance. Parameters live in `EbikeConfig` as
named physical quantities (mass, wheel radius, CdA, Crr, pack layout, gearing)
— not anonymous fitting knobs. The route grade is a function of **position**,
not time, so two designs ridden at different speeds still climb the same hill.

Run it:

```bash
python docs/examples/ebike_hybrid_simulation.py
```

### It is verifiable: the energy audit closes

Every power flow — human, battery-terminal, aero, rolling, grade, wheel
bearing, tyre-slip, chain, motor heat, and motor shaft friction — is integrated
online into a dedicated accumulator and balanced against the change in stored
energy (translational + rotational KE + gravitational + chain-spring PE). On
the reference drive cycle the balance closes to **~0.02 %**, about the level the
5e-4 solver tolerance predicts.

It did not always. An earlier release closed to 1.2 % and attributed the
residual to solver tolerance. It wasn't: the residual was one missing
bookkeeping term — the motor's shaft viscous friction ∫B·ω²dt, which had no
accumulator. In that run it came to 189 J against a 190 J residual, a match to
0.4 % — which is how you know you have found *the* missing term rather than
merely *a* plausible one. The audit had caught a real ~3 W leak and the
narration explained it away. Two lessons the notebook now teaches explicitly:
a conservation audit is sensitive enough to find
single-watt bookkeeping errors, and an unexplained residual is a *finding*, not
a rounding footnote. (It had earlier also caught a motor-loss miscalibration
and a state-of-charge coulomb-counting error.) `validate()` runs the closure
plus speed, SOC, thermal and post-cutoff-torque gates.

### Hybrid events: the legal speed cutoff, located exactly — and obeyed

The 25 km/h assist cutoff is a genuine hybrid mode transition, implemented with
a zero-crossing state machine (`AssistSpeedLimiter`) rather than a sampled
`jnp.where`. In the reference run the solver localizes the crossing to
floating-point resolution: the speed recorded at the event step is 25 km/h to
within ~1e-13.

That is only half the claim, and the half that is easy to get wrong. An earlier
version asserted the exact cutoff, never checked what the *motor* did, and
shipped a descent in which a windup-prone current loop kept pushing ~300 W for
about 12 s after the assist flag cleared — 26 % of the run's battery energy,
plus a peak speed passive coasting could not produce. Three real controller
fixes were needed (back-EMF feedforward and anti-windup; a machine whose base
speed exceeds the maximum descent speed; gains scheduled on the saturated
inductance). The notebook now tabulates the post-cutoff torque decay and
asserts it, and `validate()` gates on it, so a controller regression cannot
ship silently.

## Optimization over the true simulation

`ebike_trajectory_optimization.py` and Part 2 tune the assist policy by
evaluating the **full hybrid DAE rollout** as the objective each iteration. Two
framing decisions carry most of the honesty:

- **Compare per distance, on the same route.** A fixed-time objective silently
  rewards riding slower — less distance means less climb and less drag — and
  can dress "the bike barely moved" up as a large battery saving. Part 1's
  sweep therefore reports the energy to cover a *common* stretch of road, and
  Part 2 optimizes a fixed-distance objective.
- **A penalty term must actually bind.** An exterior penalty converges from the
  infeasible side; sized carelessly, the "optimum" quietly violates the speed
  floor it claims to hold. Part 2 sizes λ from the response grid, shows the
  optimum-vs-λ curve, and reports the residual shortfall.

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
  selectable via `make_ebike_diagram(battery_thermal_network=True)`.

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
vehicle model.

- **PyBaMM — physics-based battery** (`ebike_pybamm_battery.py`, Part 3). A
  Doyle-Fuller-Newman (DFN) electrochemical cell is the truth; a Single-Particle
  model (SPMe) tracks it to a few mV at a fraction of the cost; the **OCV curve
  is extracted from the DFN** rather than hand-written, and the vehicle's 2-RC
  ECM is fitted to the DFN, **validated on a held-out profile**, and its
  parameters are the ones Part 1 actually ships — the script re-runs the fit and
  fails if the shipped defaults drift from it. The fit's identifiability is
  reported (both RC branches distinct, no bound hits) and its validity window
  (SOC, temperature, C-rate) is stated. All voltage errors are per cell;
  multiply by 13 for the pack.
- **MuJoCo — 3-D multibody** (`ebike_mujoco_cosim.py`, `ebike_mujoco_vs_planar.py`,
  Part 4). A sagittal-plane cargo bike (suspension, tyre-ground contact,
  rear-rack cargo) on a 6 %-grade route with a speed hump, driven by a
  quantitative mirror of Part 1's drivetrain at the same 180 kg. Longitudinal
  agreement against the planar model is **measured, not asserted** (RMS ≈ 0.8
  km/h on a like-for-like flat run), and chassis pitch is **decomposed** into
  the road's geometric pitch and the chassis's own response — the latter being
  the genuinely 3-D content, and honestly ~1–2°, not the road-slope-sized angles
  a raw pitch trace suggests. Renders to `media/ebike_mujoco_cosim.mp4`.
- **OpenFOAM — 3-D aero / conjugate heat transfer** (`ebike_openfoam_aero_rom.py`,
  Part 5). A DOE → differentiable-ROM pipeline for `CdA(v, yaw)` and effective
  cooling conductance `h·A(v, yaw)`, with held-out validation and PCE-Sobol
  analysis. **No CFD is run, and none is claimed:** the backend is a
  clearly-labelled engineering-correlation stand-in whose label is printed into
  the committed output, so every downstream number — including the Sobol splits
  — is a check of the *pipeline*, not a finding about e-bike aerodynamics. The
  shipped `media/openfoam_ebike_case/` is a case **skeleton**: complete numerics
  dictionaries and turbulence fields, deliberately no geometry, so it fails at
  mesh time rather than "succeeding" as an empty tunnel that would converge to
  Cd ≈ 0 and poison the ROM. `OpenFOAMBackend` is the seam where real case
  execution plugs in; its `evaluate` raises until implemented.

## Known limitations (named on purpose)

- **No rear freewheel.** A true one-way coupling destabilises the stiff acausal
  DAE without complementarity/event support, so the chain is modelled two-way
  (the `one_way` option exists but is off). Consequence: the crank cannot coast,
  so cadence tracks wheel speed on descents.
- **End-to-end AD is fragile** for this model (see the optimization note).
- **Lumped-parameter core.** The *vehicle model* is a system-level model
  (0-D/1-D lumped, 2-D planar motion), not a 3-D field solver — by design, for
  speed. The high-fidelity 3-D physics lives in the external couplings above,
  each reduced to a surrogate the system model consumes.
- **Calibration is partial and scoped.** The battery's RC parameters are fitted
  to a DFN cell and valid in a stated window; motor, drivetrain, tyre and
  thermal parameters are realistic for the vehicle *class* but not fitted to
  bench data for a particular product.
- **Contact fidelity inverts between parts.** Part 1's planar tyre is a Magic
  Formula model; Part 4's "higher-fidelity" 3-D plant rolls a rigid cylinder in
  a Coulomb friction cone. For longitudinal traction the planar tyre is the
  better one; Part 4's advantage is geometry and suspension, not tyre physics.
- **No CFD.** See Part 5 above.

## Files

| File | What it does |
|---|---|
| `ebike_hybrid_simulation.py` | the model, energy audit, validation, hybrid event |
| `ebike_trajectory_optimization.py` | assist-policy optimization over the true DAE |
| `ebike_thermal_rom.py` | ROM cooling surrogate + multi-node thermal network |
| `ebike_pybamm_battery.py` | PyBaMM DFN battery + ECM-calibration fidelity ladder |
| `ebike_mujoco_cosim.py`, `ebike_mujoco_vs_planar.py` | 3-D MuJoCo multibody + planar comparison |
| `ebike_openfoam_aero_rom.py` | DOE → ROM pipeline + OpenFOAM case skeleton |
| `ebike_part1_smart_cargo.ipynb` … `ebike_part5_openfoam_cfd.ipynb` | the five-part tutorial series |
| `jaxility/examples/ebike_foc_lqr_deploy.py` | PMSM plant → acados LQR → embedded C |
