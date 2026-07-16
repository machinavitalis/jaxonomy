# Known Gaps

This file documents what Jaxonomy does **not** yet do, or does only
partially. It is intentionally public. We'd rather tell you what's
missing than have you discover it during a deployment.

If you hit something not listed here, please open an issue —
undocumented gaps are bugs in this file.

This document is the inverse of `CLAIMS.md`: that file lists what we
claim works (with evidence); this file lists what we don't yet claim.

---

## Format

Each entry has the same shape:

- **Area**: the part of the system affected
- **Status**: `not yet implemented` / `partial` / `experimental` /
  `known limitation`
- **What works**: the part that does work, if any
- **What doesn't**: the specific limitation
- **Workaround**: what to do in the meantime, if anything

---

## Currently known gaps

### Multirate substepping and declared state projection — fixed-step scope

- **Area**: `declare_continuous_state(substeps=N)` and
  `declare_continuous_state(project=fn)` under adaptive solvers
- **Status**: known limitation
- **What works**: under fixed-step `ode_solver_method="rk4"`, `substeps=N`
  advances a stiff block with N inner RK4 steps per outer step and
  `project=fn` retracts the state after every step; both are jit-,
  vmap-, and reverse-AD-compatible and compose
  (`test/simulation/test_t_133_multirate_substepping.py`,
  `test/simulation/test_t_132_state_projection.py`)
- **What doesn't**: adaptive solvers ignore the `substeps=` declaration
  entirely, and apply `project=` only at major-step boundaries, so a
  projected quantity can drift within a major step
- **Workaround**: use `rk4` for blocks that need per-step projection or
  substepping; under adaptive solvers, tighten `max_minor_step_size` to
  bound intra-step drift

### Performance — parameter sweeps re-JIT on every value

- **Area**: `Context.with_parameter(name, float(value))`
- **Status**: known limitation
- **What works**: explicitly traced parameter sweeps using `jax.vmap`
  or `simulate_batch`
- **What doesn't**: the natural Python-loop sweep pattern
  `for v in v_grid: simulate(diag.with_parameter("p", float(v)))`
  triggers a fresh JIT trace per iteration (the trace cache keys on
  the value, not on abstract type/shape). On a bouncing-ball plant
  with `record_event_times=True` this is on the order of 2 minutes per
  iteration.
- **Workaround**: promote sweep parameters to `jnp.asarray` and key
  the loop on traced inputs, or use `simulate_batch`

### Container blocks

- **Area**: control-flow container blocks
- **Status**: partial
- **What works**: `EnabledSubsystem`, `TriggeredSubsystem`, `ForEach`
  in `jaxonomy/framework/containers.py`; `Conditional` (boolean-enabled
  submodel with `reset` / `passthrough` / `hold` disabled-branch
  semantics, T-009) in `jaxonomy/library/`
- **What doesn't**: `ForLoop`, `WhileLoop` are not yet implemented

### FMU support

- **Area**: Functional Mock-up Interface
- **Status**: partial
- **What works**: FMI 2.0 / 3.0 co-simulation import including
  mixed-type and array I/O; pythonfmu-based FMU export with
  auto-exposed `Constant` block inputs via `build_fmu` (see
  `AGENTS/DECISIONS.md` DEC-031, DEC-032). Exported FMUs pass the
  official `fmpy.validate_fmu` checker with zero findings (T-026c —
  `build_fmu` post-processes pythonfmu's XML to add the
  FMI-2.0-required `InitialUnknowns`), and CI additionally runs the
  strict INTO-CPS VDMCheck2 static checker on every generated FMU
  (`test/library/test_t_026c_fmu_official_validation.py`). Exported
  diagram input ports are honored as real FMI inputs, outputs are
  primed during `exitInitializationMode`, declared continuous states
  can be exposed as FMI initialization parameters
  (`EXPOSE_INITIAL_STATES`), and the default-on cached-kernel
  `doStep` path is bit-identical to a fresh `simulate`
  (`test/library/test_fmu_slave.py`,
  `test/library/test_fmu_export_binary.py`).
- **What doesn't**: no model-exchange import; no FMI 3
  scheduledExecution; macOS export needs pythonfmu >= 0.7.0 (older
  releases require a one-time source build documented in `build_fmu`'s
  docstring); validator coverage is FMI 2.0 export only (imports are
  exercised by round-trip tests, not the static checkers)

### State machines

- **Area**: state-machine modelling
- **Status**: partial
- **What works**: flat Mealy-semantics state machines via
  `StateMachineBuilder` with deterministic-by-transition-order
  semantics (DEC-026); guards, resets, transitions
- **What doesn't**: hierarchical state machines

### Backends

- **Area**: `MathDispatcher` backend coverage (DEC-030)
- **Status**: partial
- **What works**: JAX (primary) and NumPy (fallback). `numpy_api as
  npa` dispatches both transparently.
- **What doesn't**: the PyTorch backend is partial — it covers ML
  block wrappers but is not a full simulation backend. CasADi / Numba
  backends are explicitly not planned.

### Determinism across hardware

- **Area**: bit-exact reproducibility across CPU / GPU / TPU
- **Status**: partial
- **What works**: bit-exact reproducibility for a given seed, inputs,
  and tolerance settings on the same hardware
- **What doesn't**: cross-hardware determinism (CPU vs GPU vs TPU) is
  a goal but not guaranteed, and the deviations are not yet
  systematically documented

### Notable absences

- **Area**: legacy "missing capability" list
- **Status**: most items shipped
- **Note**: ONNX (`ONNX` + JAX-native `ONNXJax`, T-023), LQG
  (`LinearQuadraticGaussian`, T-109), distributed ensemble
  (`simulate_distributed`, T-021), lazy/on-demand results
  (`LazyResults`, T-108 + T-015a), and per-signal native-timestamp
  recording (T-013a) have all shipped. A `jaxonnxruntime`
  op-coverage gap on quantised models remains (T-023b).

### Documentation

- **Area**: user-facing tutorials and reference
- **Status**: partial
- **What works**: README quickstart, MkDocs site at `docs/`, ~80
  example notebooks under `docs/examples/`
- **What doesn't**: docs for several recently-shipped surfaces lag
  the code (e.g. `implicit_solver` has no docs page and the
  `pinn_across_stacks` tutorial series is not yet wired into the
  MkDocs nav); the Wave-2 tutorial roadmap is in progress

---

## Out of scope (intentional, not gaps)

These are things we are explicitly **not** building. If you need them,
Jaxonomy may not be the right tool.

- **Robotics-specific abstractions.** Rigid-body kinematics, URDF
  import, actuator-with-friction models, contact-rich simulation —
  these belong in a separate planned layer (Jaxterity). Jaxonomy stays
  general-purpose. See `AGENTS/CONTEXT.md` "What Jaxonomy is NOT" and
  "When modifying Jaxonomy".
- **Cloud-hosted simulation platform.** No web UI, no collaborative
  editing, no project server, no cloud ensemble HPC. Those are
  platform features outside Jaxonomy's scope (DEC-018).
- **Embedded deployment / codegen.** Embedded codegen (C, FPGA, Arm)
  lives in a separate downstream library. Two integration paths are
  documented (MISRA-compliant C from discrete logic blocks; XLA + the
  TensorFlow C API), but Jaxonomy itself does not own the codegen
  pipeline. See `AGENTS/DECISIONS.md` DEC-019.
- **Pure ODE solver.** If you only need to integrate an ODE, use
  Diffrax directly. Jaxonomy adds block-diagram composition, hybrid
  dynamics, event handling, state machines, and acausal modelling on
  top.
- **Fully implicit DAE solver.** Mass-matrix semi-explicit DAEs are
  supported by BDF; fully implicit DAEs are explicitly rejected per
  DEC-027.
- **Real-time collaborative editing, version history, project
  permissions, requirements traceability.** Platform features outside
  scope per DEC-018.
- **Classical-PDE PINNs and neural-operator *field* surrogates**
  (spatial collocation of `u(x, t)`; Burgers / Navier–Stokes /
  heat-equation neural fields). Use DeepXDE, NVIDIA PhysicsNeMo, or
  Neuromancer. Jaxonomy covers the *dynamical-system* side instead:
  physics-informed residual learning on ODEs / DAEs (Neural ODE, SINDy,
  UDE, Neural DAE); reduced-order modeling of ODE/DAE systems (linear
  MOR, POD–Galerkin/DEIM, DMD/DMDc/ERA, Koopman/eDMD — `jaxonomy.library.rom`);
  and statistical surrogates of input→output maps (Gaussian process,
  polynomial chaos, RBF). A spatially *discretized* PDE (method of lines)
  is a large ODE and **can** be reduced with POD–DEIM here. See
  `docs/scope/rom.md` and `docs/scope/pinn.md` for the in/out boundary.
- **Large-scale Krylov/moment-matching linear MOR (IRKA) and
  trajectory-piecewise-linear (TPWL) ROM.** The linear-MOR path is
  SVD/gramian-based (dense Lyapunov solves), which is fine to moderate
  order; Krylov/IRKA for very large sparse LTI systems, and TPWL for
  weakly-nonlinear reduction, are scoped but not yet implemented. Reduce
  in the supported families until a model needs these.

---

## How this file gets updated

- A gap closes only when the corresponding evidence (test or
  benchmark) is committed and passing, and the corresponding row in
  `CLAIMS.md` is updated.
- A gap gets added when someone discovers it. Reporting an
  undocumented gap is a contribution.
- This file is reviewed at every release.
