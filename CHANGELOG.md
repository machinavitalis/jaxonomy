# Changelog

All notable changes to Jaxonomy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/): the major
version increments on breaking changes, and breaking changes are tracked here
before a new major release.

Entries describe user-visible changes — new capabilities, behavior changes,
fixes that affect users, performance improvements at user-visible scale.
Pure internal refactors live in commits, not here.

---

## [3.1.0] - 2026-07-15

### Added

- **Reduced-order modeling & statistical surrogates** (`jaxonomy.library.rom`, T-143–T-151): a new subpackage with a `reduce(...)` front door returning a `ReducedOrderModel`. Linear MOR — balanced truncation (`balred`), minimal realization (`minreal`), modal truncation and singular-perturbation residualization, with gramians / Hankel singular values and the a priori H∞ error bound. Projection ROM — `pod_basis`, `galerkin_reduce` (Galerkin / Petrov-Galerkin) and `deim` / `deim_galerkin_reduce` (DEIM hyper-reduction, per-step cost independent of full dimension). Data-driven operator ROM — `dmd` / `dmdc` / `era` and the `DMDForecaster` / `KoopmanPredictor` (eDMD) blocks. Statistical surrogates — `GaussianProcess` (kriging, with predictive variance), `PolynomialChaos` (with analytic Sobol indices / moments), and `RadialBasisSurrogate`. Every reduced model is a differentiable, simulatable Jaxonomy block.
- **`implicit_solver` — reverse-mode AD through iterative solvers** (T-131): `jaxonomy.optimization.implicit_solver(solver, residual)` makes a black-box iterative solve (Newton / fixed-point / constraint loop under `lax.while_loop`, which JAX cannot reverse-differentiate) reverse-mode differentiable via the implicit function theorem — the forward pass runs the solver untaped; the backward pass solves the adjoint system from the residual. Unlocks `jax.grad` through `simulate` for dynamics containing iterative solves.
- **Declared state projection** (T-132): `declare_continuous_state(project=fn)` supplies a manifold retraction (e.g. unit-quaternion renormalization) the integrator honors — after every step under fixed-step `rk4`, and at major-step boundaries under adaptive solvers. Keeps attitude quaternions on the unit sphere instead of drifting; differentiable and composable with `substeps=`.
- **Per-block multirate substepping** (T-133): `declare_continuous_state(substeps=N)` advances a stiff block's continuous states with N inner RK4 steps per outer fixed step (zero-order-hold coupling at the interface), honored by `ode_solver_method="rk4"`. Replaces the JIT-safe substep loops stiff blocks previously hand-rolled (motor windings, series-elastic joints); jit-, vmap-, and reverse-AD-compatible (JAX backend; adaptive solvers ignore the declaration).
- **`acausal.electrical.BLDC`** (T-135): the legacy JSON type name now resolves to `IntegratedMotor` (the ported Collimator BLDC), re-enabling models saved with the old block name.
- **FMU official-validator gate** (T-026c): every FMU produced by `build_fmu` passes `fmpy.validate_fmu` with zero findings (CI also runs the INTO-CPS VDMCheck2 static checker). `build_fmu` now adds the FMI-2.0-required `ModelStructure/InitialUnknowns` to the generated XML. New `fmu` extra installs pythonfmu + fmpy.
- **Initial-consistency projection for DAE state resets**: `SimulatorOptions(dae_initial_projection=True)` Newton-projects a caller-supplied context onto the constraint manifold before stepping begins — `with_continuous_state` on a compiled acausal system previously left the algebraic rows inconsistent and the first implicit step returned NaN. Companion API: `AcausalSystem.continuous_state_layout()` maps each state-vector row to its physical variable (differential vs algebraic), so direct state reads/writes no longer require compiler internals. `project_constraints` itself gained IFT gradients (`gradient="implicit"`, via `lax.custom_root`; FD-verified) alongside the default value-only mode, a `lax.while_loop` solver (default budget 20, unused iterations free), and a `UserWarning` on non-convergence — previously it silently returned an unconverged state.
- **FMU export: initial-state parameters and exported-input wiring**: `JaxonomyDiagramSlave` now honors exported diagram input ports as real FMI inputs (previously registered but silently ignored), primes outputs during `exitInitializationMode` (FMI 2.0 conformance — `getReal` before the first `doStep` returned 0.0 placeholders), and can expose declared continuous states as FMI initialization parameters (`EXPOSE_INITIAL_STATES`). A default-on cached-kernel path (`REUSE_SIMULATOR`) drops per-`doStep` cost ~52x (99.6 ms -> 1.9 ms on the reference two-tank FMU) with bit-identical trajectories.
- **BDF non-finite abort diagnostic**: when the BDF corrector/step loop goes non-finite and the terminal bailout poisons the state with NaN (previously silent), the solver now emits a `UserWarning` reporting the failure time, the collapsed step size, and which state rows went non-finite. jit/vmap-safe via `jax.debug.callback` behind a `lax.cond`, so healthy runs pay no host round-trip; a negative reported time indicates the failure happened in the reverse-time adjoint solve.

### Changed

- **Recording-buffer overflow degrades to uniform decimation** (T-138): a full buffer compacts to every-other sample and doubles its keep-stride instead of ring-wrapping — `results.time` always starts at `t0` and spans the whole trajectory at reduced resolution, with memory bounded by `buffer_length`. jit- and vmap-safe; the overflow warning now reads "recorded at reduced resolution (N of M samples)".
- **`simulate_batch(use_vmap=True)` finalize is vectorised** (T-019-followup): CPU sweep at N=1000 improves 1.28 s → 0.41 s; the CPU+small-batch `UserWarning` is gone.
- **Breaking: `jaxonomy.uq` distribution `kind` is keyword-only** (T-130): `Uniform(0, 1, "epistemic")` now raises `TypeError`; write `kind="epistemic"`.

### Fixed

- **Declared acausal initial conditions that the IC search overrides now warn**: a weak (non-fixed) state IC that the consistent-initialization solve replaces by a materially different value (e.g. a declared tank level silently flattened to zero) emits a `UserWarning` naming the symbol, the declared value, and the value actually used, with the `ic_fixed=True` remedy. Previously the override was silent and the simulation simply started from the wrong state.
- **DAE gradients w.r.t. algebraic initial states are no longer garbage**: reverse-mode AD through a BDF/DAE `simulate` leaked the adjoint solve's algebraic variable λ_a(0) one-to-one into `dJ/dx_a(0)` — whose true value is 0, since the implicit solver re-enforces the constraint regardless of the supplied algebraic value. Any reset-then-integrate workflow that seeded the algebraic rows from the differential ones (episodic RL/DPC training on DAE plants) got wrong — not just noisy — chained gradients (sign flips observed on a two-tank test problem). The backward sweep now zeroes the algebraic rows of the initial-state cotangent, the t=0 counterpart of the existing terminal-seed correction; differential-row sensitivities match finite differences unchanged.

- **`LinearDiscreteTimeMPC` works on the pinned OSQP again**: the block passed OSQP 1.x's `warm_starting` setting to `setup()` while the `nmpc` extra pinned `osqp ~= 0.6.5` (whose setting is `warm_start`), so it raised `TypeError` on a clean `jaxonomy[nmpc]` install. The warm-start keyword is now selected by the installed OSQP version, and the pin widened to `osqp >=0.6.5,<2` so 0.6.x and 1.x both work. Adds the block's first `setup`/`solve` regression test.
- **BDF retries transient non-finite steps instead of terminating** (T-134): a Newton blowup on a hard-switching transition (e.g. a diode turning on) now rejects the step and halves `dt`, terminating only once `dt` reaches the floor — the no-hang bound on true divergences is preserved. Long multi-cycle rectifier runs now complete with the documented recipe (`bdf` + `max_minor_step_size` below the switching-transition width: 100 cycles in ~1s wall).
- **Shared top-level parameter aliases propagate again** (T-141): `Diagram.with_parameters({"alias": v})` now updates blocks that reference the alias — previously a silent no-op, both on live diagrams (the alias `Parameter` was swapped out instead of mutated) and after a `model_json` round-trip (deserialized expression parameters lost their dependency links and namespace identity on deepcopy). Affects `fit_parameters` and every MCP/dashboard flow that sets model-level parameters on a loaded model.

## [3.0.0] - 2026-07-05

Everything accumulated since the v2.2.0 open source release, consolidated into a single major release and grouped by area rather than by development order.

### Added

#### Simulation engine, solvers & performance

- **Dynamic Precision Overrides**: Added a `precision` field (`"auto"`, `"float32"`, `"float64"`) to `SimulatorOptions` to locally override JAX precision (`jax_enable_x64` flag and PyTree continuous/discrete state casting) safely within a single simulation context, leaving global configurations unmodified.
- **TPU LU Decomposition Validation**: Integrated structural verification checks inside the implicit BDF solver to raise a clear error when trying to run double-precision implicit DAE solvers on Cloud TPU (which lacks double-precision `LuDecomposition` execution blocks).
- **JIT Warmup / Pre-Compilation**: Exposed a public `compile(tf, context)` API on the `Simulator` class to trace and pre-compile the solver kernel ahead of time, avoiding compile latency on first-solve paths.
- **Lazy Batch Finalization**: Added a `lazy=True` toggle to `simulate_batch()` that bypasses the $O(N)$ CPU element-slicing and interpolation loops, returning raw device-resident JAX arrays of shape `(N, max_steps, ...)` for maximum parallel execution efficiency.
- **`simulate_batch`**: ensemble/parameter-sweep simulation with a selective
  `jax.vmap` code path for parallel runs.
- **`with_parameters()`**: immutable parameter binding on `Diagram` and
  `LeafSystem`; deepcopy-safe callbacks; clean path for `jax.grad` /
  `jax.vmap` sweeps.
- **Diagram validation framework**: `validate_diagram()` and integration into
  `SimulatorOptions` for early detection of mis-wired diagrams.
- **Fixed-step RK4 ODE solver** (T-001a): `SimulatorOptions.ode_solver_method="rk4"` now selects a fixed-step 4th-order Runge-Kutta integrator. Suitable for real-time / MPC control loops, fixed-rate co-simulation, and `simulate_batch` workloads under `vmap` where deterministic per-tick cost beats adaptive accuracy control. Step size comes from `max_minor_step_size` (default 0.01 if unset); `rtol`/`atol` are accepted but ignored. Mass-matrix DAE systems continue to require BDF.
- **Numerical precision policy + enforcement** (T-005): `test/precision/POLICY.md` documents the float64 default, the float32 opt-in path, required tolerance floors per (solver, dtype), and per-solver error bounds. New public API: `jaxonomy.precision_info()` returns the resolved dtype/eps/x64 configuration; `jaxonomy.assert_float64_active()` raises early on silent downgrades.
- **Simulator error remapping** (T-006): `simulate`, `simulate_batch`, and `Simulator.advance_to` now wrap non-`JaxonomyError` exceptions in a new `SimulationError` with the offending block name extracted from the traceback, and a pointer to `JAXONOMY_VERBOSE_TRACEBACK=1` for the raw JAX stack. `JaxonomyError` subclasses propagate unchanged.
- **`LeafSystem.continuous_state_default`**: documented read-only property exposing the declared default continuous-state value (no more reaching into `_default_continuous_state`).
- **Lazy `SimulationResults` with per-signal native sample times** (T-108): `SimulationResults` now carries an optional `per_signal_times` dict and a `time_for(signal)` accessor; `align(time_vector)` resamples to a common grid on demand. The companion `LazyResults` wraps a deferred operation chain that materialises only at terminal calls. `LazyResults.resample(..., method=...)` routes per-channel interpolation through the T-106 backend (`"linear"` default keeps native polars / DuckDB pushdown; `"pchip"` / `"akima"` / `"cubic"` / `"nearest"` / `"flat"` fall back to per-channel `interp_1d` after the upstream chain materialises). Parquet round-trip closed end-to-end via `LazyResults.to_parquet(path)` and `LazyResults.from_parquet(path, backend="polars"|"duckdb")` — the loader re-collapses `name__i` columns into vector-valued numpy arrays.
- **DuckDB-backed lazy results** (T-015a): `LazyResults` can now back its storage with DuckDB so query plans stay lazy *through* execution, enabling results sets that don't fit in memory. Switch via the `backend="duckdb"` keyword on the recorder.
- **Fast restart** (T-112): `FastRestartSimulator` + the new stateful-simulator path skip the compile/recompile cost on parameter sweeps; first call pays the JIT, subsequent calls with the same shapes hit the cache directly. `.reset(diagram=None)` rebinds the simulator to a new diagram (or clears the cached kernel when called with no args); `.run_batch(parameters_batch, initial_states_batch=None)` runs vmap'd parameter sweeps against the warm-cached kernel and returns a `BatchSimulationResults`. A one-time `UserWarning` fires when a `run()` call's context pytree shape/dtype differs from the cached signature, flagging an imminent JIT-cache miss before the recompile.

#### Differentiable simulation & autodiff

- **`SimulatorOptions(diff_mode=...)` — an honest differentiation-mode selector.**
  `enable_autodiff` conflates "make the sim differentiable at all" with "install
  the reverse-mode adjoint," so forward-mode autodiff (`jax.jacfwd`/`jvp`)
  counterintuitively required `enable_autodiff=False` (the reverse-mode
  `custom_vjp` otherwise intercepts forward-mode and silently fails to
  differentiate the solver). The new `diff_mode` resolves into `enable_autodiff`
  at construction: `"reverse"` (→ `jax.grad`/`jacrev`), `"forward"`
  (→ `jax.jacfwd`/`jvp`), `"none"`, or `"auto"`/`None` (legacy default,
  `enable_autodiff` unchanged). Setting `diff_mode="forward"` together with
  `enable_autodiff=True` now raises a clear error instead of silently degrading
  gradients. The resolution is `dataclasses.replace`-safe (the selector clears
  after resolving, so it never re-clobbers an explicit `enable_autodiff`
  override). Surfaced by jaxterity (forward-mode `parameter_jacobian`).
- **`jaxonomy.submodel_function`** (T-008): public API wrapping a `Diagram` or `LeafSystem` as a pure function `f(context, *inputs) -> outputs`. Compatible with `jax.grad` / `jax.jit` / `jax.vmap`. Intended for MPC cost functions, RL environment step functions, and higher-order constructs that need a submodel as an ordinary JAX-traceable callable rather than a simulator run.
- **`jaxonomy.scalar_cost_simulate`**: reverse-mode-differentiable scalar-cost helper — the counterpart to `simulate_jacfwd`. Packages the "accumulate the cost as a diagram state (e.g. an `Integrator` of the running cost) and read the final value off the context" pattern, resolving the `enable_autodiff=True` + `save_time_series=True` rejection (you can't record a trajectory under `jax.grad`). Returns the scalar (compose your own `jax.grad`) or `(value, grad)` with `return_grad=True`.
- **Public event-time gradient API** (T-125): `simulate(..., record_events=True)` populates `results.events`; `jax.grad` flows through event times via the existing custom-VJP infrastructure (single- and multi-event, batched). Enables switched-system MPC, hybrid trajectory optimisation, and contact-implicit work.
- **Differentiable `StateMachine` transition timing** (T-NEW-sm-smooth-guard): `jax.grad` now flows through the *timing* of a `StateMachine` transition. The SM compiles guard strings to boolean predicates (`x > 0.5` → a `±1` trigger with zero gradient), which previously made event-time (saltation) gradients through a transition silently `0`. The SM now also derives a *smooth* guard residual (`x - 0.5`) for any single-comparison guard and attaches it as the new optional `grad_guard` on `ZeroCrossingEvent` / `declare_zero_crossing(grad_guard=…)`; the saltation gradient machinery uses it for `∇g` / the implicit-function denominator, while the boolean guard still drives triggering and zero-crossing localization unchanged. Compound / equality guards keep `grad_guard=None` (no event-time gradient, as before). Validated AD-vs-FD on `Sine(amplitude) → 3-state SM → Integrator`. Together with the T-001c #1d saltation fix this closes differentiable event timing through state machines.
- **`multi_event_time_gradient`** (T-125-followup-multi-event-saltation): computes the saltation gradient `dt_e/dp` for *every* firing along a hybrid trajectory by propagating the forward sensitivity `∂x(t;p)/∂p` through reset maps — the correct path for repeated-event problems (bouncing ball, contact-implicit MPC) where the user cannot write a closed-form `state_at_event_fn` past the first firing. Accepts per-firing or shared `guard_fn` / `reset_map_fn`, scalar or PyTree parameters, and is itself `jax.grad`-traceable. Exported at top level alongside the existing `event_time_gradient` family.

#### Acausal & DAE modeling

- **Neural corrections inside acausal DAEs** (T-044 NeuralDAEBlock, phase 1):
  `jaxonomy.library.add_neural_correction(acausal_system, nn_fn, theta, *,
  state_rows=…)` adds a learned term `f_NN(t, x; θ)` to the **differential**
  rows of a compiled acausal semi-explicit DAE (`M ẋ = f + pad(f_NN)`,
  `0 = g`), declaring `θ` as a dynamic parameter so `jax.grad` / `optax` flow
  into it through `simulate` and the BDF-DAE adjoint. The algebraic constraint
  rows — and the symbolic Pantelides index reduction that produced them — are
  untouched (`f_NN ≡ 0` is byte-equivalent to the bare system; the constraint
  residual is unchanged). This is the differentiable-acausal capability: a neural
  correction *inside* an index-reduced DAE, which Modelica (no autodiff),
  Neuromancer (no acausal), and causal-function-first tools can't express. A
  mass-spring example recovers the velocity-proportional drag an unmodeled
  damper would have produced by fitting `θ` (loss 0.14 → 6e-4). Phase 1 is a
  post-hoc wrapper (no compiler change beyond exposing the compiled RHS).
- **First-class `NeuralDAEBlock`** (T-044 NeuralDAEBlock, phase 2): the same
  learned correction, now authored *in the diagram* —
  `ad.add_neural_correction_block(NeuralDAEBlock(nn_fn, theta, targets=[(comp,
  "v")], name=…))` — instead of patching a compiled system. The compiler
  resolves each `(component, state_name)` target to a differential row of the
  index-reduced DAE and injects `f_NN` at the same post-Pantelides RHS site, so
  the (non-symbolically-differentiable) neural term never enters index
  reduction. The block is registered, not `connect`-ed: it contributes no
  equations. `nn_fn` is written against the targeted physical states
  (gather-in / scatter-out), so authors never touch compiler row indices, and a
  state that index reduction made *algebraic* is rejected with a clear error
  listing the valid differential states. `θ` is exposed as the component-style
  parameter `f"{name}_theta"`; multiple blocks compose. Both `NeuralDAEBlock`
  and `add_neural_correction` are re-exported from `jaxonomy.acausal`.
- **Acausal modeling framework** (`jaxonomy.acausal`): full support for
  electrical, mechanical, thermal, and hydraulic domains with Pantelides
  index-reduction; graduated to a first-class public namespace.
- **DAE constraint-residual detection** (T-003): `jaxonomy.simulation.compute_constraint_residual` and `constraint_residual_norm` return the algebraic-row residual of the semi-explicit mass-matrix system at a given context, letting users detect drift in long DAE simulations. Linear DAEs under BDF (the current acausal library) stay at machine precision; the primitive is the foundation for the projection / Baumgarte layer that a nonlinear DAE test case will motivate.
- **`PlanarPendulum` 2-D acausal primitive** (T-032): `jaxonomy.acausal.component_library.planar.PlanarPendulum` is the first 2-D / Cartesian mechanical primitive — a point mass on a rigid massless link governed by `x² + y² = L²`. The constraint is index-2 in the Lagrange-multiplier formulation, exercising Pantelides index-reduction end-to-end through the existing BDF mass-matrix solver. Self-contained (no acausal ports) by design: a full 2-D port system can layer on top later without breaking this contract. Replaces the V-003 `test_index2_constrained_pendulum` xfail with a passing test that holds `||r|| - L < 1e-6` over a 2-second swing. Exposes scalar `x_out` / `y_out` output ports for sensing.
- **DAE constraint projection / drift detection** (T-113): primitive infrastructure for online algebraic-constraint projection on top of T-003's residual computation. Linear DAEs under BDF stay at machine precision; the projection layer is staged for the first nonlinear DAE that demonstrates drift. Two opt-in stabilization paths wired through `SimulatorOptions`: Baumgarte stabilization (`baumgarte_alpha`, `baumgarte_beta`) adds `-2α·ġ - β²·g` to each algebraic row of the solver RHS, and SSP-style Newton projection (`dae_projection_enabled`, `dae_projection_tol`, `dae_projection_max_iter`) solves the algebraic sub-block per major step. Both default-off paths are byte-equivalent to the prior solver; the IFT-defined projection is `jax.grad`-traceable through the projected state. Validated long-horizon on PlanarPendulum (index-2 DAE) over 60 s with `||f_a||_∞ < 1e-6`.
- **`AcausalSystem.parameter_names_for(component)`** (T-113-followup-acausal-param-name-docs): typed accessor that returns the parameter keys an `AcausalSystem` exposes for one acausal component, removing the grep-or-guess workflow for `jax.grad` / `jax.vmap` over acausal-component parameters. `AcausalCompiler`'s docstring now documents the `{component_name}_{symbol_name}` parameter-naming convention so the bare strings in `ctx.parameters` are discoverable from API docs alone.

#### Control, linearization & estimation

- **`LinearizedSystem`**: `linearize()` now returns a `LinearizedSystem` object
  that carries A/B/C/D matrices and supports frequency-domain analysis directly.
- **Online estimation blocks**: `RecursiveLeastSquares` and
  `AugmentedStateEKF` for real-time parameter estimation.
- **`findop` operating-point trim — `axis_mask`, `residual_fn`, `residual_scaling`** (T-128): the Newton trim now solves robustly on systems the full-state iteration could not handle. `axis_mask=` (boolean array or index list) selects which state components the iteration drives to zero, holding the rest at the initial guess — needed to trim systems with *passive* states whose equilibrium derivative is intrinsically nonzero (a cornering vehicle's heading `ψ̇ = r ≠ 0`, a free integrator), which otherwise dominate the step and block convergence. `residual_scaling=` (`"auto"` → per-component `1/max(|rᵢ(x₀)|, eps)`, or an explicit array; cf. MATLAB `findop`'s `XScaling`/`YScaling`) puts disparate-unit residuals on a common footing for both the Newton step and the convergence test. `residual_fn=` overrides the default `ẋ` residual entirely. The docstring now also points at *integration to steady state* as the robust fallback when Newton stalls. Defaults reproduce the previous behaviour exactly.
- **Linearization workflow** (T-109): `findop` (Newton operating-point trim), `frequency_response`, `bode_data`, `nyquist_data`, `pole_zero_map`, `step_response`, `impulse_response`, plus an empirical-FRE path (`estimate_frequency_response`) for chirp / PRBS / sinestream excitation. Cross-validated against `python-control`'s SLICOT-backed equivalents on the canonical Mass-Spring-Damper, DC-motor, and linearized-pendulum fixtures. All helpers are fully differentiable through `A, B, C, D` and excitation parameters. Higher-order operators: `jaxonomy.discretize(linsys_or_diagram, dt, *, method="zoh"|"euler")` is polymorphic on its first argument — the LTI path returns a discrete-time `LinearizedSystem` (carrying a `dt` field, with `is_stable()` switching to the unit-disk criterion); the diagram path linearizes about a supplied `base_context` then discretizes in one call. `jaxonomy.with_observer(plant, observer, *, plant_u_port, plant_y_port)` wires a Luenberger or Kalman-style observer to a plant and exposes `u` / `x_hat` at the diagram boundary, and `jaxonomy.library.Luenberger(A, B, C, D, L)` ships as the discrete-time observer building block (caller supplies the gain matrix).
- **DPC policy-as-a-block** (T-040-followup): a differentiable predictive-control policy can now be authored as a real jaxonomy `LeafSystem` (`jaxonomy.control.dpc.PolicyBlock`) and run inside a feedback `Diagram` under `simulate` — the same solver/event/recording stack as every other model. The policy's parameters live in the context as a flat `theta` dynamic parameter, so `jax.grad` of a downstream cost flows through `simulate` into them (validated AD-vs-FD). New helpers: `PolicyBlock`, `PlantBlock`, `build_closed_loop`, `simulate_closed_loop`, and `ClosedLoopRunner` — a **build-once / run-many** entry point that constructs the feedback Diagram + Context once and exposes a thin `run(params, x0)` swapping only `theta` / `x0`, so a training loop avoids the per-step diagram rebuild that `simulate_closed_loop` pays; `run` is differentiable and `jax.vmap`-able (`jax.vmap(runner.run)(params_batch)` gives a batched rollout). Worked end-to-end in `docs/examples/dpc_two_tank_reference_tracking.ipynb` (loss 59.97 → 0.038, RMS 1.271 → 0.099 m). The function-level `ClosedLoopRollout` remains for lightweight batched receding-horizon training.
- **`PIDController2DOF` + discrete-filter family extensions** (T-127): 2-DOF PID controller (separate setpoint/feedback paths with weighting), expanded discrete-filter set (`FilterDiscrete`, `LowPassDiscrete`, `IntegratorDiscrete`, `DerivativeDiscrete`, `LeadLag`, `Notch`), plus PID tuning helpers and config round-trip.
- **`jaxonomy.control.dpc` differentiable predictive control scaffolding** (T-040, scaffolding): `ClosedLoopRollout` (fixed-step RK4 + `jax.vmap` over the batch axis), `Penalty` (soft-quadratic / log-barrier), `dpc_loss` (stage + terminal + penalty composition), and `train_policy` (optax-driven gradient descent). Public API matches the T-040 spec so the upgrade to a full Diagram-integrated rollout (T-040-followup) is API-stable. Suitable today for function-level DPC experiments on JAX-traceable plants.

#### Standard library blocks

- **`UnitDelay` docstring now points at `TransportDelay` for multi-step
  latency.** A fixed N-step transport delay is `TransportDelay(dt=outer_dt,
  delay_seconds=N*dt)` — one ring-buffered, signal-differentiable block — not a
  chain of N `UnitDelay`s. (No code change to the blocks; both already existed.)
- **`StateMachineBuilder`**: Python DSL for authoring hybrid state machines
  without writing raw guard/reset callbacks by hand.
- **`with_key` on stochastic blocks**: explicit PRNG key injection for
  `RandomNumber` / `WhiteNoise`, enabling independent noise streams under
  `jax.vmap`.
- **`ShiftRegister` and `MaskedDelayBuffer`** blocks in the standard library.
- **`CustomPythonBlock` isolation**: `finalize_script` now executes in a
  sandboxed environment.
- **`PyTorchPredictor` / `TensorFlowPredictor`** aliases for the `PyTorch` / `TensorFlow` inference blocks — either spelling now imports and constructs the same block.
- **`Conditional` container block** (T-009): wraps a submodel with a boolean enable input; three disabled behaviours (`reset`, `passthrough`, `hold`). Gradients through disabled branches are zero rather than undefined. Suitable for building systems where a subdiagram activates conditionally.
- **Differentiable lookup-table family** (T-106 + T-114): pure-functional 1-D / 2-D / N-D backends with linear, PCHIP, 2-D bicubic, natural cubic spline, and Akima interpolation; `clip` / `linear` / `nan` extrapolation; even-spacing detection. Public blocks `LookupTable1d`, `LookupTable2d`, `LookupTableND`, plus the prelookup family (`Prelookup`, `PrelookupInverse`, `InterpolationUsingPrelookup`) and `TableSearch`. All differentiable through both the table values and the query coordinates.
- **Variable + discrete delay family** (T-107 + T-116): `TransportDelay` (continuous, constant), `VariableTransportDelay` (continuous, signal-driven τ), `UnitDelay` (discrete, 1-step), discrete `Delay(constant_length)`, and `ShiftRegister` / `MaskedDelayBuffer` building blocks. The variable-τ path supports differentiable delay parameters. `VariableTransportDelay(method="linear"|"pchip")` selects per-output interpolation over the ring buffer; PCHIP gives a C¹-smooth gradient w.r.t. `tau` across sample boundaries (linear remains the default and is byte-equivalent to prior behaviour).
- **Smooth + hard rate-limiter / saturation + quantizer modes** (T-115): hard variants (`Saturate`, `RateLimiter`, `Quantizer`) match the bit-exact clipping behaviour HIL deployments depend on; smooth variants (`SoftSaturate`, `SoftRateLimiter`, `soft_dead_zone`, `soft_saturate`) replace hard clips with differentiable approximations so the gradient survives through the saturated region. `Quantizer` gains a `mode={"round","floor","ceil","trunc"}` kwarg with `stop_gradient` wrapping under JAX. `Saturate(limit=L)` keyword-only shorthand expands to symmetric `(upper_limit=+L, lower_limit=-L)` (T-115-followup-saturate-symmetric-kwarg).
- **`Mux` / `Demux` / `BusCreator` / `BusSelector` / `BusUpdate`** (T-117): positional pack/unpack via `Mux`/`Demux`; named-field buses via `BusCreator`/`BusSelector` with full pytree support; `BusUpdate` for functional field overrides on existing buses; per-field unit propagation through buses (cross-link with T-104). `BusSelector` accepts dot-separated nested paths (`BusSelector("chassis.suspension.spring_force")`) to descend into nested NamedTuple buses in one block (T-117-followup-bus-dot-path). `BusUpdate(bus_unit=...)` propagates the schema on `bus_in` / `bus_out` and the per-field unit on `new_value` (T-117-followup-bus-update-units-prop).
- **`Switch` and `MultiPortSwitch`** (T-118): hard and smooth modes; smooth uses sigmoid (`Switch`) or softmax (`MultiPortSwitch`) blends so gradients flow across the transition.
- **`TruthTable` block** (T-119): combinational logic block authored from a row-by-row truth table via `TruthTableBuilder`; branchless under JAX. The constant-output common case (every row outputs a static value) compiles to a single `argmax` + `take` selection over the stacked outputs; callable-output rows fall back to a `where`-reduction loop. `TruthTableBuilder(input_names=...)` labels survive on the built block's `input_ports` so they appear in `print_schedule` / model JSON / error messages (T-119-followup-truth-table-named-ports).
- **Container blocks** (T-120): `EnabledSubsystem`, `TriggeredSubsystem`, `ForEach` (and the supporting `EnabledMode` / `EnabledStateMode` / `TriggerEdge` enums). Subsystems can be enabled/disabled by a boolean control input or fired on rising/falling/either trigger edges; `ForEach` vectorises a child subdiagram across an axis. Live in `jaxonomy/framework/containers.py`.
- **Battery-domain components** (T-121): `BatteryCell` standard-library block plus acausal `jaxonomy.acausal.component_library.battery` components for cell-level electrical modeling.
- **Stochastic source family** (T-122): `RandomSource`, `UniformRandomNumber`, `BandLimitedNoise`, `PRBS`, `PRBSLFSR`, `WhiteNoise` standard-library blocks with explicit `with_key` PRNG injection. Distribution CDFs across the supported families plus multivariate-normal CDF via Genz QMC.
- **`LookupTable2d` exposes fitted arrays as read-only properties** (T-114-followup-lookup-table-fitted-getter): `output_table_array`, `input_x_array`, and `input_y_array` are now plain `@property` accessors on the block, so the fitted Z grid can be inspected / plotted without re-running `fit_table_2d`.
- **`StateMachineBuilder.build(time_mode=, dt=)`** (T-118-followup-state-machine-time-mode): the builder now produces a discrete-time `StateMachine` directly when `time_mode="discrete"` and `dt=` are supplied. The default remains `time_mode="agnostic"` (back-compat); `dt` is required for `"discrete"` and rejected otherwise. Eliminates the previous workaround of building the agnostic version and re-wrapping its `_sm` attribute.

#### Units & multirate

- **Signal/port unit annotations with consistency checking** (T-104): ports and signals carry optional unit metadata (`BusUnit`); the diagram compiler verifies dimensional consistency across connections and surfaces concrete unit-mismatch errors at build time rather than as silent numerical bugs at runtime. Re-exported as `jaxonomy.library.BusUnit` for convenience next to the bus blocks. Extensions: `Unit.physical_quantity` disambiguation tag distinguishes same-dimension units of different meaning (`N·m` as torque vs energy); lossless JSON round-trip via `Unit.to_dict/from_dict/to_json/from_json` and a human-readable `Unit.summary()`; `propagate_diagram_units(diagram, *, overwrite=False)` fixed-point walker stamps inferred units forward through Adder / Gain / Product / Reciprocal / Integrator / Derivative / passthrough blocks via a `register_unit_rule` registry; canonical SI `flow_units` / `pot_units` published on every per-domain acausal `PortBase` subclass (Electrical/Rotational/Translational/Thermal/Hydraulic/Fluid) as module-level constants.
- **Multirate sample times + rate-transition semantics** (T-105): explicit per-block sample-time annotations, a priority-scheduler hook for ties at the same tick, period-jitter modeling, and a Graphviz rate-summary helper that visualises the rate groups in a diagram. `print_schedule` distinguishes feedback-through-discrete from algebraic cycles in its execution-order section (T-105-followup-print-schedule-feedback-cycle), with explanatory messages instead of the bare `<cycle detected>` token. `DiagramBuilder(auto_insert_rate_transitions=True)` opt-in synthesises a `Decimator` (fast→slow) or `ZeroOrderHold` (slow→fast) bridge whenever `connect` joins ports at different inferred rates; default remains the strict-mode warning path.
- **`RateTransition` and `Decimator`** (T-123): explicit rate-conversion blocks for crossing sample-time boundaries with selectable hold / interpolation semantics.

#### Optimization, fitting & UQ

- **`fit_parameters` improvements**: returns `OptimizationResult`; adds
  multi-start optimization, sensitivity analysis, confidence-interval
  estimation via Laplace approximation, and IPOPT support for constrained
  problems.
- **Objective function helpers**: `ise_objective`, `lqr_objective`,
  `tracking_mse`, `weighted_sum` in `jaxonomy.optimization`.
- **Differentiable lookup-table fitting** (T-124): `fit_lookup_table_1d`, `fit_lookup_table_2d`, `fit_lookup_table_nd`, `fit_table_1d_with_grid`, `fit_table_2d`, and `fit_table_nd` fit table values (and optionally grid placement) to data via `fit_parameters`. The N-D fitter (follow-up surfaced by F1 part 3) uses 2^N corner-weighted multilinear design with an N-D Laplacian smoothness option; recovers analytic linear-on-grid data to machine precision. `fit_table_1d_with_grid` defaults `auto_normalize=True` so wide-range data (e.g. `rpm ∈ [80, 650]`) converges with the default `learning_rate=1e-3` instead of NaN-blowing up. Unlocks engine-map / aero-coefficient / battery-OCV-SOC fitting from measurement data, including the 3-D and higher-D surfaces (e.g. F1 aero map with three output channels).
- **Monte-Carlo / Sensitivity / UQ workflow** (T-126): `jaxonomy.uq` module bundles aleatoric Monte Carlo (with parameter distributions, vmap wrapper), Latin-Hypercube sampling, quasi-MC, and Sobol sensitivity decomposition into a first-class API. `sobol_indices(n_bootstrap=N)` returns percentile confidence intervals (`first_order_ci`, `total_order_ci`) alongside the point estimates so users can detect Jansen-estimator negativity at small N; the resampling is JAX-vectorised so a 1000-resample bootstrap on N=1024 runs in a single XLA launch. `quasi_monte_carlo`'s docstring documents the bounded-Hardy-Krause-variation requirement so users understand which QoIs degrade silently to IID convergence.
- **`tune_parameters` API + diagnostics module**: `jaxonomy.optimization.parameter_tuning.tune_parameters` wraps the common single-objective parameter-tuning flow on top of `fit_parameters`; `jaxonomy.diagnostics` surfaces dead-store and empty-inputs warnings at diagram compile time (T-036e). Post-hoc result diagnostics — `analyze_saturation`, `analyze_phase_activity`, `analyze_control_oscillation`, and `analyze_horizon_completion` — catch common silent failure modes (over-saturated actuators, never-fired state-machine phases, bang-bang control, smoothed-indicator integrators that never crossed their event). `analyze_saturation(mode="upper_only" | "lower_only")` and the single-rail auto-promote rule mean one-sided actuators (throttle, brake, PWM) don't trigger spurious warnings at their natural rest state; `analyze_phase_activity(name=...)` matches the labelling convention used by the rest of the family.

#### Provenance, variants & reproducibility

- **Provenance / reproducibility manifest** (T-110): `simulate(diagram, ctx, t_span, options=SimulatorOptions(record_provenance=True))` emits a JSON-serialisable `ProvenanceManifest` (accessible as `results.provenance`) containing JAX/Jaxonomy versions, OS, device, git revision, parameter hash, PRNG seeds, and a complete results bundle pointer. Sufficient to reproduce a simulation byte-exactly on the same device. Persisted via `ProvenanceManifest.save(path)` / round-tripped via `jaxonomy.simulation.load_manifest(path)`; compared via `compare_manifests` / `verify_manifest` (raises `ManifestMismatch` on drift).
- **Variants / configurable diagrams** (T-111): `Variant(name, predicate)` blocks select among alternative subdiagrams at compile time based on parameter values, enabling product-family models from a single source diagram. JSON round-trip via `dump_variant_config_to_json(diagram)` / `load_variant_config_from_json(diagram, json_str)` captures the `{variant_name: active_choice}` binding portably (builder callables stay in Python; only the binding ships). `jaxonomy variants` CLI exposes `list` / `dump` / `apply` subcommands against a `pkg.mod:fn` diagram-builder spec for CI / release-pipeline use. `expand_all_variant_configs(diagram)` enumerates the Cartesian product of every named variant's choices in deterministic order, and `iter_variant_configurations(diagram)` is the generator form yielding `(config_dict, configured_diagram)` for parameter sweeps.
- **`jaxonomy.simulation.simulate_variant_sweep`** (T-126-followup-variant-sweep-vmap): runs `simulate_batch` (or `simulate` when `param_batches=None`) once per `Variant` configuration and returns a dict keyed by configuration. Variant-axis vmap is not possible (pytree shape varies per choice) so this packages the canonical per-variant Python loop and exposes the right `recorded_signals` factory pattern for users sweeping structurally-different sub-diagrams.

#### FMI / co-simulation interop

- **FMI 2.0 / 3.0 mixed-type and array I/O** (T-026a): `jaxonomy.library.ModelicaFMU` now dispatches every co-simulation port to the right typed accessor (`getFloat32/64`, `getInt8/16/32/64`, `getUInt8/16/32/64`, `getBoolean`, `getEnumeration`) and supports FMI 3 array-shaped variables (e.g. a `(3,)` Float64 input/output). One C call per variable type per step instead of one per port. Reference-FMU corpus (BouncingBall, Dahlquist, VanDerPol, Stair, Feedthrough, StateSpace) is exercised in `test/library/test_fmu_reference_corpus.py` across both FMI versions when `JAXONOMY_FMU_CORPUS=` points at a built corpus directory.
- **Binary FMU export** (T-025a): `jaxonomy.library.fmu_export.build_fmu` packages a Python `pythonfmu.Fmi2Slave` subclass (or any user-authored slave script) into a binary `.fmu` zip with `modelDescription.xml` + `binaries/win64/` + `binaries/linux64/` (+ `binaries/darwin64/` after the T-025b one-time wrapper build). `jaxonomy.library.fmu_slave.JaxonomyDiagramSlave` is a base class that wraps a Jaxonomy diagram — every exported input/output port becomes an FMI Real variable, every `Constant` block in the diagram is auto-exposed as a writable FMI input variable named after the block (T-025c), and `do_step` runs `jaxonomy.simulate` over the [t, t + step_size] segment with input writes routed into the simulation context as parameter overrides. Round-trip verification (export → fmpy import → step → compare) runs on every host whose pythonfmu install carries a wrapper for `sys.platform` — the test harness probes for the binary rather than gating on platform string.
- **Darwin pythonfmu wrapper buildable from source** (T-025b): `build_fmu`'s docstring now documents the one-line patch + CMake build that produces `libpythonfmu-export.dylib` for arm64 macOS. With it installed, `binaries/darwin64/` is included in every generated FMU and the round-trip test path runs locally instead of being limited to linux64 / win64.
- **`ModelicaFMU(first_step_at_zero=True)`** (FMU offset asymmetry follow-up): fires the FMU step at `t=0` instead of `t=dt` for FMUs whose exporter expects that semantics, eliminating the one-sample phase lag in round-trip tests. Default `False` preserves the Modelica clocked-block convention.

#### Cloud, serialization & tooling

- Reserved `simulate_cloud()` entry point for future remote batch execution (no execution backend is bundled in this build).
- **Plotting utilities**: extended `DataSource` / `SimulationResultsSource`
  with convenience plotting helpers.
- **Block diagram visualization notebook**: pure-Python/matplotlib renderer
  (replaces the previous CDN/JS approach).
- **Jaxonomy MCP server**: FastMCP-based tool server with test suite for
  IDE and agent integrations.
- **Schema versioning**: JSON model format now carries a schema version with
  round-trip validation.

#### Tests, tutorials & developer docs

- **Marimo tutorial**: hybrid thermostat system notebook.
- **Triple inverted pendulum tutorial** with MuJoCo integration.
- **AGENTS/ directory**: agent-driven development workflow documentation.
- **Gradient-correctness test framework** (T-001): property-based
  finite-difference verification of reverse-mode autodiff across solvers,
  standard-library blocks, and event-handling paths. Documented
  per-(solver, dtype) tolerance policy in `test/autodiff/TOLERANCES.md`.
  Failure messages surface the offending block, solver, dtype, and
  element. CI workflow runs a fast PR-time subset on every pull request
  and a comprehensive nightly sweep on the default schedule.
- **Determinism and reproducibility policy** (T-002): `test/determinism/POLICY.md` documents the bit-exact reproducibility contract — same Jaxonomy version, same device, same inputs, same PRNG seeds → identical output bytes. CI enforces the contract by running every reference simulation twice and comparing `tobytes()`. Negative-control tests guard against trivially-passing bit-exact checks. Cross-device (CPU↔GPU↔TPU) deviation is documented as expected at ULP scale and tested with a skip-guarded scaffolding for accelerator CI.
- **Conservation-law property tests** (T-004): `test/conservation/` verifies energy / angular-momentum conservation on four closed conservative systems (SHO, undamped pendulum, free rigid body, LC circuit) over 10–50 oscillation periods, across every selectable ODE solver. Failures report the conserved quantity, solver, tolerance, and drift magnitude, so a regression in either the solver or a block's math surfaces as an explicit failure rather than a silent numerical skew.
- **Returning-booster tutorial series + cinematic MuJoCo demo**: 6-part notebook series in `docs/examples/` covering modeling, MPC + render, atmosphere & phases, high-fidelity propulsion, sensing & estimation, and GNC validation. Accompanied by a cinematic `booster_landing_cinematic.mp4` rendered through `render_booster.py` (T-019b render demo).

### Changed

#### Simulation engine & options

- **`SimulatorOptions.int_time_scale` now defaults to `"auto"`** (was `None`/picosecond). `"auto"` picks the finest power-of-ten integer-time scale that represents `t_span[1]` with headroom: short simulations keep picosecond resolution, multi-year horizons coarsen transparently so they just run instead of raising a representability error. Pin a float to override; `None` keeps the legacy "leave the global scale untouched" behaviour.
- **`Simulator.advance_to` reuses its compiled kernel across calls** (non-autodiff path): it is now `jax.jit`-wrapped as a stable instance attribute, so a persistent `Simulator` (construct once, `advance_to` many times — interactive stepping, MPC inner loops) hits the JAX cache on the second and later calls instead of re-tracing op-by-op.
- **`SimulatorOptions` accepts `max_major_step_size=`** as an alias for `max_major_step_length=` (symmetric with `max_minor_step_size`). The docstring documents that `max_major_step_length` is JIT-static — it derives the bounded-loop trip count baked into the compiled kernel, so changing it between jitted calls forces a recompile.
- **`declare_output_port(requires_inputs=...)` defaults to `None` (inferred)**: resolves to `True` (collect all inputs) except for the unambiguous `prerequisites_of_calc=[DependencyTicket.nothing]`, which resolves to `False`. A callback that reads `inputs[i]` after inputs were trimmed now raises a clear error naming `requires_inputs` instead of a bare `IndexError`.
- **Buffer-overflow `UserWarning` is now solver/tolerance-aware**: it names the adaptive solver and tolerances, explains that the recorder saves one sample per accepted minor step (so tightening `rtol`/`atol` records more), and recommends a concrete `buffer_length` (or `ode_solver_method="rk4"` for a predictable sample count). The `buffer_length` docstring documents the per-minor-step cadence.
- **`SimulatorOptions.buffer_length` auto-sizes to `max(max_major_steps, 2048)`** (was hardcoded `1000`, then `max_major_steps`). The recorder saves one sample per accepted *minor* step — typically far more than `max_major_steps` (~200 for a continuous system), so the old default silently overflowed the fixed ring buffer and returned a truncated tail (`results.time` starting mid-trajectory). The `2048` floor keeps the common case from overflowing while staying small enough not to balloon memory under `vmap`/batch. Applied at both defaulting sites (`simulate()` finalize and direct `Simulator(...)` construction). Explicit values are honoured byte-equivalently; set `buffer_length` explicitly for very long fine-grained recordings that still exceed the floor.
- **`simulate_batch(use_vmap=True)` emits a one-time `UserWarning` on CPU at `N < 10⁴`**: the per-row finalize after the vmapped XLA kernel is host-side `O(N)`, so the kernel path (the `use_vmap=False` default) typically wins on CPU at moderate batch sizes (concrete: kernel ~0.31 s vs vmap ~1.28 s at N=1000). The docstring's vmap section spells out the trade-off explicitly. GPU/TPU users should ignore — vmap wins there. The proper fix (batching the post-vmap finalize itself) is filed as `T-019-followup-batched-vmap-finalize`.
- Simulator internals refactored in six phases: context-scoped backend
  dispatching, functional output caching, stricter generics, diagram
  flattening, decoupled event/ODE handling, and full simulator decomposition.
- `ParameterCache` is now thread-safe and vmap-safe.

#### Dependencies & packaging

- Migrated away from `jaxopt`; JAX and related dependencies updated for
  compatibility.
- Project metadata, URLs, and documentation updated to reflect the Jaxonomy
  domain.

### Fixed

#### Autodiff & gradient correctness

- **Reverse-mode `jax.grad` through BDF-DAE acausal systems is now correct** (T-113-followup-dae-adjoint-sign-bug): gradients w.r.t. a parameter that enters a semi-explicit-DAE *algebraic* constraint (e.g. `Insulator.R` in the thermal / battery-pack demos) were badly wrong — ~25× too small on a single RC cell and >90% off (sign-flipped on larger packs) — because the reverse-mode adjoint dropped the algebraic terminal-state contribution. The fix applies the correct Cao/Li/Petzold/Serban semi-explicit-DAE terminal handling: a consistent-IC correction to the differential adjoint seed plus the direct terminal boundary term on the parameter gradient. Reverse-mode now matches forward-mode (`simulate_jacfwd`) and central differences to <5%. `jax.grad`-based optimisation/tuning over acausal-DAE parameters (the differentiable-acausal headline) is now trustworthy.
- **`jax.grad` through a `StateMachine` block no longer crashes** (T-001c-followup #1b/#1c). Reverse-mode differentiation of a diagram containing a `StateMachineBuilder`/`StateMachine` block raised a JAX-internal `ValueError: too many values to unpack (expected 1)` (a weak-vs-strong `float0` cotangent disagreement on the state-machine `mode` integer at `cond` transpose time). The fix severs the gradient of integer/boolean leaves of the context in both branches of each event-handling `cond`, so the two branches emit a consistently-typed zero cotangent; differentiable (float) gradients are unaffected. Gradients w.r.t. value-path parameters (e.g. an upstream integrator's initial condition, input-dependent transition actions) are now correct.
- **Event-time (saltation) gradients now propagate when the guard depends on a parameter entering through an upstream input signal** (T-001c-followup #1d). Previously, `jax.grad` reported `0` where finite differences reported a materially nonzero value whenever the zero-crossing *guard* read a signal driven by an upstream block (e.g. a comparator on a `Sine(amplitude=…)` output): the per-block saltation adjoint differentiated the guard against a *frozen* port cache, so the implicit-function denominator (the guard's total time-derivative `dg/dt`) collapsed to zero and the upstream-parameter numerator `∂g/∂p` was discarded. The fix differentiates the guard against a *refreshed* port cache so `∇g` is live in the upstream parameters / signals and the total time-derivative, then applies the rank-1 saltation correction across the full root context. Two paths are covered and validated AD-vs-FD to machine precision: (a) the **self-contained** case where the guard and the dynamics jump live in the same block (`leaf_system._wrap_reset_map`), and (b) the **cross-block** case where the event fires in a block with *no continuous state* and the dynamics jump it induces lives in a downstream block — supplied at the simulator level (`autodiff_rules._cross_block_saltation_correction`) where the full continuous costate is available. Requires a *smooth* guard surface; a boolean threshold guard (`x > 0.5`) has `∇g ≡ 0` and no recoverable event-time gradient on its own (see the `StateMachine` `grad_guard` entry under Added for how the built-in state machine supplies one).
- **Multi-event saltation gradients are now correct** (T-125-followup-multi-event-saltation), via the new `multi_event_time_gradient` helper. The single-shot `event_time_gradient` relied on a user-supplied closed-form `state_at_event_fn` to carry the trajectory sensitivity `∂x_e/∂p`; this is only writable for the *first* firing, so for repeated events (a bouncing ball, contact-implicit MPC, switched-system trajopt) every firing past the first collapsed to ~0 where finite differences reported a materially nonzero value. `multi_event_time_gradient` instead propagates the forward sensitivity `S(t)=∂x(t;p)/∂p` along the recorded trajectory — integrating the variational equation `Ṡ = f_x S + f_p` within each arc and applying the saltation jump `S⁺ = R_x S⁻ + R_p + (R_t + R_x f⁻ − f⁺)(dt_e/dp)` at every event — so the implicit-function-theorem formula sees the correct `S⁻(t_e)` at each firing. Validated AD-vs-FD to machine precision across all bounces of a restitution sweep. The existing single-event helpers are unchanged.
- **`jax.grad` through a `ModelicaFMU` block raises a clear, FMU-specific error** (a co-simulation step is an opaque external call with no derivative) with concrete workarounds, instead of the generic backend message "IO callbacks do not support JVP". The forward simulation is unchanged.
- **`linearize` is now `jax.jit` / `jax.grad` / `jax.vmap` traceable end-to-end** (T-109-followup-linearize-traceable): two host-side diagnostic guards (finite-state precheck, equilibrium-residual warning) are gated behind `isinstance(x, jax.core.Tracer)` so the body composes cleanly under any JAX transformation. Eager-mode diagnostics are unchanged.
- Autodiff correctness for pure discrete-time and hybrid CT+DT+DAE systems.

#### Acausal & DAE modeling

- **`acausal.electrical.Diode` no longer overflows in deep forward bias** (T-134).
  The raw Shockley exponential `exp(V/Vt)` is unbounded and overflows float64 to
  `inf` at V≳28.4 V (default `Vt=0.04`), which under a hard drive produced
  inf/NaN residuals. It is replaced with a SPICE-`pnjlim`-style limited
  exponential — exact below the breakpoint `V = 40·Vt` (1.6 V) and a C¹ tangent
  continuation above — so the diode current stays finite with **no** change in
  the normal operating range (a physical diode sits well below 1.6 V). The
  Shockley `Diode` now has a regression test
  (`test_electrical.py::test_shockley_diode_rectifies`) confirming it rectifies
  an AC source and stays finite. (The docstring's stale `Vknee/Ron/Roff` args
  were also corrected to the real `Ids/Rp/Vt`.) Note: the diode is still a stiff
  device — long multi-cycle AC-rectifier runs need `ode_solver_method="bdf"` and
  remain expensive over hundreds of cycles; see the `Diode` docstring.
- **`AcausalCompiler` printed `add weak IC for x_dot_el=...` to stdout** for every dynamic state during index reduction, polluting tutorial cell output. The bare `print` at `diagram_processing.py:1934` is now gated behind `self.verbose` (matching the rest of the file). Cell output stays clean by default; opt in with `AcausalCompiler(..., verbose=True)` for the diagnostic.
- **Acausal `flow_units` / `pot_units` are now checked by the Pantelides node-merging pass** (T-104-followup-acausal-pantelides-units): `check_port_set_domain` compares both unit metadata across every port at a merged node alongside the existing domain-enum check, and emits a clear `AcausalModelError` naming the conflicting ports + unit values on mismatch. Defensive infrastructure for the per-instance unit-override path (e.g. a temperature sensor reporting in °C connected to a thermal mass in K) — today every same-domain port carries identical class-level units, so the check is a no-op on shipped components.
- **`HeatflowSource` sign convention is now documented** in the constructor docstring: Modelica through-variable convention — `Q > 0` flows *into* this component. With `enable_port_b=True` and `Q_flow > 0`, heat is pumped from `port_a` into `port_b`, so the "heat into the room" intuition needs the right port wiring (or a negative `Q_flow`). Stops the "room goes to −157 °C" debug cycle every new acausal-thermal author would hit.
- **AcausalCompiler determinism** (T-002a): compiling the same `AcausalDiagram` twice now returns byte-identical state vectors. Previously the compiler's reliance on set / sympy-hash iteration produced a different state vector shape and ordering per run (observed: (5,), (6,), (7,) for the same RC circuit). Fixed by replacing four `set()` containers in the diagram and component base with insertion-ordered dicts, replacing `atoms().pop()` with `sorted(...)[0]`, and sorting the final-DAE variable lists in the index-reduction pass.
- **`HeatflowSensor` chained-equation count mismatch** (T-031): two `HeatflowSensor` components in series previously triggered Pantelides "Mismatch between the number of equations N and the number of variables M". The sensor was missing the flow-conservation equation `Q1 + Q2 = 0` — physically required for a passive sensor (heat in = heat out, no storage), and the missing equation per sensor compounded across a chain. Fixed by adding the conservation equation; behaviour for single-sensor topologies is unchanged.
- **Fluid volume components using `IdealGasAir` or `WaterLiquidSimple` media no longer crash on finalize.** Their `gen_eqs` lacked the `name_prefix` keyword that the `fluid.py` volume components (`ClosedVolume`, `Accumulator`, `OpenTank`, `MassflowSource`) now pass, raising `TypeError: gen_eqs() got an unexpected keyword argument 'name_prefix'`. Both media now accept (and ignore) `name_prefix`.
- Acausal compiler: alias elimination, duplicate-equation detection, initial
  condition handling, and domain export bugs.

#### Simulation engine, solvers & scheduling

- **`SimulatorOptions(math_backend="jax", enable_tracing=False)` was silently swapped to numpy** (T-002-followup-tracing-downgrade-warn): the previous `logger.warning` (invisible under default log config) is now a `warnings.warn(UserWarning)` that names the actual culprit and warns about the downstream `ode_solver_method='dopri5'` failure that comes with the numpy backend. The swap behaviour itself is unchanged so existing callers that use `enable_tracing=False` as the "give me eager numpy" knob keep working; they just see the warning.
- **`declare_periodic_update(callback, period=...)` crashed at the first scheduler tick** with `TypeError: minimum(NoneType, float)` when `offset=` was omitted. The default is now `offset=0.0` (the canonical clocked-block convention), so the scheduler sees a numeric offset on every periodic update and the cryptic deep-stack TypeError is gone.
- **`Profiler.stop` masked simulation exceptions** with `KeyError: '_wrapped_simulate'` when `start` had not been called (e.g. an exception in `ScopedProfiler.__exit__` re-raised the bookkeeping error instead of the real one). `stop` now `pop`s the entry and returns silently when no start was recorded, so the underlying exception propagates cleanly.
- **`Diagram.print_schedule()` correctly buckets discrete blocks before context creation** (T-105-followup-print-schedule-pre-context): lazily calls `create_context()` first so blocks whose periodic events register in `initialize()` (`PIDDiscrete`, `Decimator`, `UnitDelay`, `ZeroOrderHold`) appear in the correct rate group. Opt out with `ensure_initialized=False`.
- **`simulate_batch(use_vmap=True)` compatibility** (T-002b): previously raised `NotImplementedError: IO effect not supported in vmap-of-cond` because four IO callbacks sat inside the simulator's inner `lax.cond` (ODE step-size-too-small, end-time-not-reached, end-time-not-representable, and the results-buffer overflow dump). Replaced with pure no-ops so the vmap path compiles. Cost: less descriptive error messages on a narrow set of failure modes, and silent truncation if a recorded simulation exceeds `buffer_length` (users should set `buffer_length` >= expected sample count).
- **`simulate(...)` with `max_results_interval` set but `max_minor_step_size` left at its default no longer crashes.** The minor-step clamp compared `max_results_interval < options.max_minor_step_size` where the latter defaults to `None`, raising `TypeError: '<' not supported between 'float' and 'NoneType'`. The default (unbounded) minor step is now treated as larger than any finite results interval, so it is clamped correctly instead of crashing.
- **A feedthrough sample-and-hold no longer lags an upstream discrete source by one tick.** Phase-1 discrete cache/output updates arrive already sorted in execution order, but the port cache was not refreshed *between* them, so a downstream feedthrough sample-and-hold (e.g. `ZeroOrderHold`) read its upstream's stale pre-tick value: a same-rate `ZeroOrderHold` lagged its source by a step (two in series were *not* a no-op), and a `DiscreteClock → ... → ZeroOrderHold`/`DerivativeDiscrete` chain shifted by one sample. The port cache is now refreshed between Phase-1 updates that feed one another, so a same-rate `ZeroOrderHold` is the identity (Simulink sorted-execution semantics). Only paid when >1 cache update fires on a tick; independent updates are unchanged. This also removed a spurious one-step delay that could destabilise a discrete control loop.
- Four simulation engine bugs: discrete-event atomicity, Dopri5 floor loop,
  `max_major_steps` scoping, and a misleading docstring.

#### Standard library blocks

- **`TransportDelay` (and any event-less output port) used to emit a noisy WARNING** on every construction — `period is None so default_value is not used for port 0 in block delay`. The behaviour is correct (the default really is unused), but the WARNING level read as a user-facing bug. Demoted to `logger.debug`; the diagnostic remains available for anyone wiring up a new event-less port that intended to use `default_value`.
- **`library.Chirp` frequency convention** (T-122-followup-chirp-hz-convention): `Chirp` previously interpreted `f0` / `f1` in rad/s despite the docstring (and the `scipy.signal.chirp` parity claim) calling them Hz. The block now interprets both as Hz by default, matching scipy and the docstring; a user passing `f0=1.0, f1=2.0, stop_time=1.0` now actually gets a sweep from 1 Hz to 2 Hz. Legacy diagrams that depended on the pre-fix semantics can opt in via the new `units="rad/s"` kwarg, which emits a `DeprecationWarning` and will be removed in a future release. Two tutorial notebooks (`linearization_workflow.ipynb`, `actuator_delay_identification.ipynb`) updated to either pin `units="rad/s"` (preserves byte-equivalence) or document the new default.
- **`PIDDiscrete` algebraic-loop false positive** (T-127-followup-pid-discrete-feedthrough): the canonical `plant → err → PIDDiscrete → Saturate → plant` closed-loop pattern previously raised `AlgebraicLoopError` because `PIDDiscrete` declared its sample-and-hold output as feedthrough on the live input. The spurious feedthrough hint is dropped; closed-loop diagrams now build without a hand-inserted `UnitDelay` workaround.
- **Saturate rate-group misclassification** (T-115-followup-saturate-rate-classification): memoryless feedthrough blocks whose only zero-crossing events are solver hints (`Saturate`, `DeadZone`, `IfThenElse`, `Comparator`, `LogicalOperator`, `LogicalReduce`) no longer classify as `event_driven`. `print_schedule` labels them as universal-match; downstream-discrete topologies no longer trip phantom `event_driven ↔ discrete` mismatch warnings. Behavioural-ZC blocks (state machines, `Relay`, user blocks with `reset_map`) keep their `event_driven` classification.
- **State-machine transition priority in agnostic mode** (T-033): the lowest-index transition was documented to win on simultaneous-firing guards, but the zero-crossing path declared one event per transition with no priority gating, so order was implementation-defined. Each guard's effective truth is now AND-ed with the negation of every higher-priority sibling guard from the same source state, so simultaneous truths produce exactly one firing — the highest-priority one. Behaviour for the discrete-time path was already correct and is unchanged.
- **`Saturate` parameters now survive a JSON serialization round-trip.** `Saturate`'s `limit=L` convenience-shorthand wrapper replaced `__init__` without `functools.wraps`, so `inspect.signature` saw `(self, *args, limit, **kwargs)` instead of the real parameters. The JSON block factory filters constructor kwargs by that signature, so it silently dropped `upper_limit` / `lower_limit` on reload, leaving them undeclared — a reloaded `Saturate` then raised `KeyError('upper_limit')` at context creation. The wrapper now preserves the signature, so all parameters round-trip. (Other `@parameters`-decorated blocks were unaffected — only Saturate had the extra un-`wraps`ed wrapper.)

#### Control, linearization & estimation

- **Frequency- and time-domain analysis helpers now respect discrete-time `LinearizedSystem`s.** `frequency_response` / `bode_data` / `nyquist_data` evaluated the transfer function at `s = jω` unconditionally; for a system produced by `discretize(...)` they now evaluate at `z = e^{jωΔt}`, matching `is_stable()`'s discrete branch. `step_response` / `impulse_response`, which used the continuous matrix exponential `expm(A·t)` and returned meaningless numbers for a discrete system, now raise a clear `ValueError` rather than silently producing wrong output (a real discrete recurrence path is tracked as a follow-up finding).
- **`KalmanFilter` with a non-zero feedthrough matrix `D` no longer crashes.** The feedthrough correction referenced `self.K`, which `KalmanFilter` never assigns (the Kalman gain is the per-step local `K`); any filter with direct feedthrough raised `AttributeError`. Now uses the local gain. The two-step correction `x⁺ = x⁻ + K(y − Cx⁻) − K(Du)` is algebraically unchanged.
- **`LTISystem` exposes its `A` / `B` / `C` / `D` matrices at construction.** They were only populated in `initialize()` (at context creation), so reading them off a freshly-built block — e.g. `linearize(...).to_lti().A`, a common idiom — raised `AttributeError`. They are now set in `__init__` (and re-derived from resolved parameters in `initialize()`, unchanged). Construction also accepts plain nested lists for `A/B/C/D`, not only arrays.
- `StateMachineBuilder`, `LinearizedSystem`, and `ParameterCache`
  correctness regressions.

#### Optimization, fitting & sensitivity

- **Sensitivity analysis no longer truncates vector-valued parameters.** `compute_sensitivity` zipped one name per parameter dict key against per-flat-element gradients/sensitivities, so for any array parameter the unidentifiable-parameter detection (and the `summary()` table) silently dropped trailing elements and paired names with the wrong sensitivities. Parameter names are now expanded per element (mirroring `confidence._expand_param_names`).
- IPOPT optimizer: switched to limited-memory Hessian approximation.

#### Units & currency

- **Currency-axis conversions no longer silently treat 1 USD = 1 EUR.** `conversion_factor`, `convert_offset_aware`, and `assert_units_compatible_with_scale` compared only the seven SI base-dimension exponents and ignored the separate currency axis. Two distinct currencies share all-zero SI dims, so `conversion_factor(usd, eur)` returned `1.0` and a USD output port wired to a EUR input port under the default `unit_conversion="auto"` was accepted with factor `1.0` — even though `are_units_compatible(usd, eur)` correctly returns `False`. All three helpers now raise `UnitMismatchError` on a currency mismatch and point the caller at `convert_currency()` / `set_fx_rate()` (a fixed scalar factor cannot represent a live FX rate).
- **`set_fx_rate` no longer leaves a stale reverse rate when a pair is re-set.** Setting `USD→EUR=0.92` auto-populates `EUR→USD=1/0.92`; a later `USD→EUR=0.95` refresh previously updated only the forward rate, leaving `EUR→USD` stale and breaking the docstring's exact-round-trip promise on the daily-snapshot refresh path. Auto-populated reverses are now tracked and refreshed on re-set, while an explicitly user-set asymmetric reverse (bid/ask) is still preserved.

#### Results, serialization & data loading

- **Loading a dashboard model with an init script and an array-valued model parameter no longer crashes.** The "did the init script change this parameter?" check used a bare `!=`, which raises `ValueError: truth value of an array … is ambiguous` for any multi-element array parameter. Now uses `np.array_equal`, which is element-wise and returns a plain bool (scalar/`None` behaviour preserved).
- **`LazyResults.where(...)` on the polars backend no longer corrupts signal names ending in `t`.** A blind `"t " → "time "` substring replace turned a predicate like `"out > 0.5"` into `"outime > 0.5"`, raising `RuntimeError` where the eager and duckdb backends handle the identical predicate. The polars path now uses the same word-boundary substitution (`\bt\b → time`) as the duckdb path.
- **`models.CompactEV` is now simulable.** Its continuous-derivative callback read three non-existent discrete-state fields (`mot_volt_based_enable_state`, and `discrete_stat.veh_speed_zoh` — two typos), so every derivative evaluation raised `AttributeError`. Corrected to the declared field names (`mot_volt_based_enable`, `discrete_state.veh_spd_zoh`).
- Serialization round-trip fidelity for the JSON model format.

#### Provenance / reproducibility

- **`ProvenanceManifest.config_hash` is now stable across processes** (T-110-followup-stable-fingerprint): the hash recurses through child structures via a new `_structural_fingerprint` helper that ignores per-process `system_id`, unlocking the cross-process verification workflow the headline "notarized receipt" claim depends on.

#### FMI / co-simulation interop

- **`ModelicaFMU` co-sim error handling** (found during T-026a): the runtime `except` clause referenced `fmpy.fmi2.FMICallException` and `fmpy.fmi3.FMICallException`, neither of which exists. fmpy defines this exception class once in `fmpy.fmi1`. Every FMU step that errored crashed jaxonomy's wrapper with `AttributeError: module 'fmpy.fmi2' has no attribute 'FMICallException'` instead of producing a `BlockRuntimeError`.
- **`ModelicaFMU` doStep timing** (found during T-026a): the wrapper passed the periodic-update fire time as `currentCommunicationPoint`, but FMI 2.0 §4.2.4 requires it to be the *start* of the step interval — i.e. `time - dt`. Lenient FMUs (e.g. the previous in-tree `thermal_1.fmu`) accepted the misuse; strict FMUs (Reference-FMUs/BouncingBall) rejected the very first step with "Expected currentCommunicationPoint = 0".
- **`ModelicaFMU` FMI 3 alias variables** (found during T-026a): two output variables sharing a `valueReference` (e.g. BouncingBall FMI 3's `h` and `h_ft`, both vr=1) collapsed in a dict-by-vr lookup, producing a namedtuple field-name collision. Outputs are now indexed by position, so each port surfaces independently.

#### Errors & developer experience

- **Actionable errors for common first-contact mistakes**: indexing an empty `block.output_ports[i]` now raises an `IndexError` naming the block and suggesting `declare_continuous_state_output()` / `declare_output_port(...)` (instead of a bare "list index out of range"); indexing a context with an output port (`context[port]`) raises a `TypeError` suggesting `port.eval(context)`; accessing `MLP.mlp` before `create_context()` raises a clear error pointing at `create_context()` instead of `AttributeError`.
- **`Context.with_parameter("k", float(value))` trace-cache miss + `AttributeError` on Python scalars** (T-008-followup-with-parameter-trace-cache + T-008-followup-with-parameters-scalar-coerce): `LeafContext.with_parameters`, `DiagramContext.with_parameters`, and the fast-restart `_pure_patch_context` now coerce Python scalars / numpy generics / 0-d numpy arrays to `jnp.asarray(value, dtype=orig_param.dtype)` on the JAX backend. The naive sweep `for c in cs: kernel(ctx.with_parameter("c", float(c)))` now compiles exactly once instead of recompiling per value, and the canonical `diag.with_parameters({"osc.c": 0.4})` papercut (previously `AttributeError: 'float' object has no attribute 'shape'`) is gone. Validation errors at the callsites name the offending parameter key.
- **`submodel_function` performance envelope is now documented**: the docstring spells out the fine cases (one-shot rollouts, `jax.vmap` batches, `jax.grad` traced once) and the not-fine case (tight Python-side loops invoking it thousands of times per simulated second — typical MPC inner loops, ~100× slowdown vs closing over the underlying primitive). Includes a rule of thumb ("profile if invoked > ~100×/sim") and points at `engine_map_fitting_to_mpc.ipynb` for the hand-rolled-scan workaround.

---

## [2.2.0]

Third open source release.

### Changed

- **License changed from AGPL-3.0 to MIT.**
- Package renamed from `collimator` to `jaxonomy`; all public imports
  updated accordingly.
- Project references, documentation, and branding updated from Collimator,
  Inc. to Jaxonomy.

---

## [2.0.9]

Second open source release.

*(Incremental patch series on the simulation engine and library blocks
shipped from the private repository; detailed per-commit history not
reconstructed here.)*

---

## [2.0.6]

Initial open source release under AGPL-3.0.

The Jaxonomy core simulation engine — `LeafSystem`, `DiagramBuilder`,
`simulate()`, the standard library blocks, ODE solvers, autodiff support,
and the acausal modeling framework — made publicly available for the first
time.
