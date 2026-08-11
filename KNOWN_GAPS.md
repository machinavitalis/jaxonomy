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

### Influence graph — state granularity is per state group, not per component

- **Area**: `jaxonomy.analysis.influence_graph`
- **Status**: known limitation
- **What works**: one node per leaf continuous-state group (`xc`) and
  discrete-state group (`xd`), with exact Jacobian blocks on every edge;
  path products are exact for scalar signals and scalar states
  (`test/analysis/test_influence.py`)
- **What doesn't**: a block with a multi-component state (a 3-state
  `BatteryCell`, a 12-state rigid body) collapses to one node, and the
  scalar edge weight is the induced ∞-norm of the Jacobian block — a
  conservative upper bound. So "which *component* of this state drives
  the output" cannot be answered from the node scores, and a slice
  through a vector state over-approximates. The same applies to vector
  ports.
- **Workaround**: the full Jacobian block is retained on the edge, so
  `graph.graph.edges[src, dst]["relative"]` (or `["jacobian"]` for raw
  partials) gives the per-component answer; split the block into
  scalar-state blocks if the graph itself must resolve components

### Influence graph — trajectory mode costs one simulation per snapshot

- **Area**: `influence_graph(..., at="trajectory", n_snapshots=k)`
- **Status**: known limitation
- **What works**: building at a single operating point is linear in block
  count — measured 0.4 s at 129 blocks, 0.9 s at 629, 3.8 s at 2504 —
  and every query (`slice`, `attribute`, `bottlenecks`,
  `influence_subgraph`) then runs in well under a second at all three
  sizes
- **What doesn't**: recorded signals do not pin down every stateful
  leaf's state, so trajectory mode re-derives the operating points by
  advancing the context, costing `k + 1` `simulate` calls. `simulate`
  has a fixed per-call setup cost that scales with block count and
  dominates the integration — a 1 µs span costs the same as a 4 s one
  (3.38 s vs 3.40 s at 629 blocks) — so the snapshots, not the
  Jacobians, set the price: ~11 minutes for `n_snapshots=6` at 2504
  blocks.
- **Workaround**: use the operating-point mode on large models, or
  lower `n_snapshots`; the per-edge profile is the only thing lost

### Influence graph — path search is bounded, not exhaustive

- **Area**: `InfluenceGraph.slice` / `.attribute` / `.dominant_paths`
- **Status**: known limitation
- **What works**: scores are best *simple*-path products, verified
  against exhaustive enumeration on random graphs, and pruned only by an
  admissible bound — so a contributor whose partial product dips below
  the threshold and recovers is still found. Regions behind an edge with
  no local gradient are resolved by reachability rather than path
  enumeration, so a comparator or quantizer upstream does not make the
  search exponential
  (`test/analysis/test_influence.py::TestTraversalSoundness`)
- **What doesn't**: enumerating simple paths is exponential in the worst
  case, so the search carries a `max_depth` (default 32) and an
  expansion budget. On a densely-connected model both can bite: paths
  longer than `max_depth` are never seen, and if the budget is exhausted
  the scores are lower bounds. Scores *inside* an unmeasurable region
  are placeholders, not rankings — the nodes are retained and flagged,
  but their numbers should not be compared against measured ones.
- **Workaround**: the result says so — `InfluenceSlice.truncated` is set
  and `report()` prints a note. Raise the threshold, lower `max_depth`,
  or focus on a smaller neighbourhood with
  `analysis.influence_subgraph`

### Influence graph — one `tau` per graph on a stiff model

- **Area**: `influence_graph(..., tau=...)`
- **Status**: known limitation
- **What works**: `tau` scales continuous-state-rate edges, and a path
  crossing *k* integrators reads exactly as that path's transfer
  magnitude at ω = 1/`tau`
  (`test/analysis/test_influence.py::TestPathAttribution::test_path_product_is_the_gain_at_one_over_tau`)
- **What doesn't**: a single `tau` applies to every state in the model.
  On a stiff model (a 2 ms electrical loop inside a 4 s mechanical one)
  no single value reads as a percentage for both, so an absolute
  threshold means different things on the fast and slow paths, and node
  scores at different integrator depths from the target are being
  compared through different-order transfers.
- **Workaround**: run the analysis at several `tau` values — it is a
  frequency sweep, and the picture changing *is* the information (see
  `docs/examples/influence_graph_model_slicing.py` section 3); use
  `InfluenceGraph.relative_threshold()` to scale the cutoff to the
  strongest contributor instead of to an absolute number

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
  the value, not on abstract type/shape). Per-iteration cost scales
  with model size: measured (2026-07, jax 0.9.2, arm64 CPU) at
  ~0.15–0.2 s on a bouncing-ball plant with
  `record_event_times=True`, and ~2.3 s on a 160-block diagram.
- **Workaround**: promote sweep parameters to `jnp.asarray` and key
  the loop on traced inputs, or use `simulate_batch`. The persistent
  JIT cache (`enable_persistent_jit_cache`, `docs/jit_cache.md`) does
  **not** mitigate this: each float value is baked into the HLO as a
  constant, so every sweep value is a compulsory cache miss (measured:
  a 3-value sweep against a warm cache added 3 new cache entries and
  every iteration paid the full re-trace + compile cost).

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
  scheduledExecution; validator coverage is FMI 2.0 export only
  (imports are exercised by round-trip tests, not the static checkers)
- **Exported FMUs are tool-coupled**: the slave runs as Python, so the
  importing side needs a Python environment with jaxonomy importable.
  There is no self-contained binary export.
- **The bundled wrapper limits where an export can run.** `build_fmu`
  compiles nothing; it bundles the pre-built wrapper from the installed
  pythonfmu wheel, and that wrapper is x86-64 and links no libpython.
  So a stock install produces FMUs that load only on x86-64, and only
  under a Python FMI master (FMPy, jaxonomy) — a C/C++ master fails at
  `dlopen` with `undefined symbol: _Py_NoneStruct`. Neither shows up as
  a validator finding, because `fmpy.validate_fmu` and VDMCheck read
  `modelDescription.xml` and never load the binary.
  `scripts/build_pythonfmu_wrapper.sh` builds a host wrapper that fixes
  the ISA, links libpython, promotes it to global scope (so numpy's C
  extensions resolve under an `RTLD_LOCAL` master), and skips the
  `Py_Finalize` that otherwise segfaults on `fmi2Terminate` /
  `fmi2FreeInstance`. With it, a pure-C master runs an exported FMU on
  both x86-64 and aarch64 Linux. `wrapper_diagnostics()` reports which
  wrapper is installed, and `build_fmu` warns when it would produce an
  FMU that cannot load.
- **OpenModelica cannot import our FMUs, for its own reasons**:
  `importFMU` in OpenModelica 1.27.0 accepts FMI 2.0 model exchange
  only and rejects every co-simulation FMU, including ones OpenModelica
  itself exported. Our export is co-simulation only, so that pairing
  needs ME export (not implemented) rather than a fix on this side.

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
