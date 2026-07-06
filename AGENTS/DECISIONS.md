# Architectural Decisions

This file records significant architectural and design decisions made during
Jaxonomy's development. Each entry uses the ADR (Architectural Decision Record)
format.

## How to use this file

- **Before making a non-trivial architectural choice**, search this file for
  prior decisions on the topic.
- **After making a decision**, add a new entry using the template at the bottom
  of this file.
- **When superseding a decision**, mark the old entry's Status as
  "Superseded by DEC-NNN" and create a new entry. Do not delete old entries.
- Keep entries under ~400 words. Longer analysis belongs in a linked document
  or task spec.
- Numbering: DEC-NNN for accepted decisions, DEC-PNN for proposed but not yet
  accepted. Numbers are never reused, even after supersession.

---

## Decisions

<!-- Newest first. When adding, insert above existing entries. -->

## DEC-032: FMU import dispatches per-port at step time, batched per type

**Status**: Accepted

### Context

T-026 wired version-aware getters/setters for *parameters* but left
`exec_step` assuming every input/output port was a scalar `Real` /
`Float64`. Real FMI 3 FMUs (Reference-FMUs/Feedthrough, StateSpace) have
mixed-type ports (Float32/64, Int8…64, UInt8…64, Boolean, Enumeration)
and array-shaped ports (`(3,)`, `(3, 3)`).

Two viable shapes:

1. **One C call per port per step**: simple, but every step makes O(N)
   FFI calls, dominating cost on FMUs with dozens of ports.
2. **Per-type batched groups**: at init time, bucket ports by type;
   each step issues exactly one C call per occupied bucket. fmpy's
   typed accessors accept lists of value references natively.

### Decision

Adopt shape (2). At init time we build `self._output_groups` and
`self._input_groups`, each a list of `{accessor, dtype, indices, refs,
nvals, shapes}`. `exec_step` walks these groups and reshapes per-port
results from the flat returned tuple. Arrays use FMI 3's
`getXxx(refs, nValues=total)` form; FMI 2 has no array variables.

### Rationale

The per-type cost is what matters in practice — typed-getter dispatch
through fmpy is the same syscall regardless of whether you ask for one
or many references. Building the groupings once at init keeps the
hot path branch-free except for a per-bucket loop. Arrays cost only a
final numpy `reshape`.

### Consequences

- Adding a new FMI type requires one row in the
  `_FMI2_ACCESSORS` / `_FMI3_ACCESSORS` table — the dispatch code
  doesn't need touching.
- String / Binary outputs are skipped from the default port set
  because they don't fit in a JAX-traced state. Users who need them
  must pass `output_names=` explicitly.
- Alias variables (multiple ports sharing a `valueReference`) are now
  surfaced individually because the variable lookup is by position
  rather than by valueReference.

### Alternatives considered

- **Keep per-port C calls** for simplicity. Rejected: a Feedthrough
  FMU with 14 inputs / 14 outputs would issue 28 FFI calls per step,
  measurable overhead vs. 5–6 with grouping.
- **Compile typed access into a per-FMU JAX-traceable graph**.
  Attractive for tracing but defeats the FMU contract — the C side
  has hidden state, so io_callback boundaries are unavoidable.

---

## DEC-031: FMU export uses pythonfmu's wrapper, not a custom C scaffold

**Status**: Accepted

### Context

T-025 shipped the modelDescription.xml generator. T-025a needs to
produce a *binary* `.fmu` zip that an FMI master can load — that
requires a C wrapper exposing the `fmi2*` symbols and routing every
call back into a Python implementation. Three viable paths:

1. **Hand-write a C wrapper** based on the FMI Reference-FMUs scaffold
   and embed CPython.
2. **Adopt `pythonfmu`** — a maintained library that already ships
   pre-built wrappers for win64 / linux64 with the Python-embedding
   plumbing done.
3. **Adopt `unifmu`** — uses gRPC between the wrapper and a backend
   process; supports more languages but requires a heavier runtime
   stack.

### Decision

Wrap `pythonfmu.FmuBuilder` from
`jaxonomy.library.fmu_export.build_fmu`. Provide
`JaxonomyDiagramSlave` as a `Fmi2Slave` subclass that introspects a
Jaxonomy diagram's exported ports and routes `do_step` through
`jaxonomy.simulate`.

### Rationale

pythonfmu's pre-built wrappers eliminate the C toolchain dependency on
the user's machine (the win64 / linux64 cases) and the project is
actively maintained. The Python-facing API (`Fmi2Slave`, typed `Real`
/ `Integer` / `Boolean` variable classes) is clean and matches the
slot-based shape of jaxonomy diagrams. Building our own wrapper would
mean maintaining FMI compliance, CPython embedding semantics, and
cross-platform binaries — all already solved.

### Consequences

- macOS arm64 isn't covered by pythonfmu's wheel, but the upstream
  CMake project builds cleanly against host CPython after a one-line
  patch (`#elif defined(__linux__) || defined(__APPLE__)` for the
  destructor attribute). T-025b documents the steps and ships a
  build script in `build_fmu`'s docstring. Once installed into
  pythonfmu's `resources/binaries/darwin64/`, the existing dylib
  glob picks it up — no jaxonomy-side code change needed.
- `build_fmu`'s contract takes a Python script path containing a
  `Fmi2Slave` subclass, not a `Diagram` object directly. This is
  pythonfmu's contract; users wrap their diagram in a slave script
  using `JaxonomyDiagramSlave`.
- `JaxonomyDiagramSlave` auto-discovers `Constant` blocks and
  exposes each as a writable FMI input (T-025c), removing the
  default-no-op `apply_inputs` hole. Vector-element names with
  bracket / comma syntax are supported via closure-based
  getter/setter pairs targeting an internal dict, since pythonfmu's
  default `setattr`-on-self path won't survive non-identifier names.

### Alternatives considered

- **Custom C wrapper**: rejected for maintenance burden — every FMI
  spec point and every CPython embedding gotcha would land on us.
- **unifmu**: heavier deployment (requires a separate Rust-built
  daemon) and a gRPC transport boundary on every step, which is fine
  for cross-language interop but unnecessary when both sides are
  Python.

---

## DEC-030: MathDispatcher with context-variable scoping for backend switching

**Status**: Accepted

### Context

Backend switching needs to be both global (most code uses one backend per
process) and locally overridable (tests parametrize across backends, occasional
contexts may force a specific backend). Module-level globals don't compose with
parallelism; explicit-passing pollutes every call site.

### Decision

Backend dispatch goes through a `MathDispatcher` class (`backend/backend.py`)
that holds the active backend in a Python context variable. `set_backend()`
updates it. Backend implementations (`JaxBackend`, `NumpyBackend`,
`TorchBackend`) register themselves and provide the dispatched ops. The module
is imported throughout as `numpy_api as npa`.

### Rationale

Context variables compose correctly with asyncio and threading where
module-level globals don't. The single `npa` symbol at call sites keeps
user-facing code clean. Registration of backends as classes makes adding a
fourth backend (e.g., a future CasADi or Numba backend) a closed extension,
not a refactor.

### Alternatives considered

- Module-level global with a setter (rejected: doesn't compose with parallel
  test runs)
- Explicit backend argument on every numerical call (rejected: pollutes every
  call site)
- Pure `if isinstance(...)` dispatch at each site (rejected: verbose,
  error-prone)

### Consequences

PyTorch dispatch is partial — it covers ML wrappers but isn't a full simulation
backend. Adding a new backend means registering it with the dispatcher and
implementing the dispatched op set. See PATTERNS.md for the `npa` vs. `jnp`
vs. `np` rules at call sites.

---

## DEC-029: Custom Python / JAX blocks are stateless-feedthrough by design; persistence via cache only

**Status**: Accepted

### Context

User-authored blocks need a way to compute outputs from inputs and parameters
and to persist data across simulation steps (counters, accumulators, cached
intermediate computations). Two extremes exist: full `LeafSystem` subclassing
with declared continuous and discrete state (powerful but heavyweight), or
stateless lambdas (lightweight but limiting).

### Decision

`CustomPythonBlock` and `CustomJaxBlock` are stateless-feedthrough as their
formal interface — no `declare_continuous_state` or ODE callback is exposed.
Cross-step persistence is provided through `CacheType.persistent_env` in the
cache mechanism, with isolation enforced per block instance.

### Rationale

The vast majority of user custom-block needs are pure or trivially stateful.
Forcing users through full `LeafSystem` subclassing for "I need to remember the
previous output" is excessive. The cache mechanism gives them what they need
without exposing the full state-declaration machinery. Per-instance isolation
prevents the global-state contamination class of bugs (DEC-020).

### Alternatives considered

- Expose continuous-state declaration to custom blocks (rejected: power users
  can subclass `LeafSystem` directly; exposing it through the custom-block API
  would muddy the API contract)
- Module-level globals (rejected: no isolation, breaks vmap)
- Force `LeafSystem` subclassing for any stateful behavior (rejected: too much
  friction for common cases)

### Consequences

Users who need true continuous state in a custom block must subclass
`LeafSystem` directly rather than using `CustomPythonBlock`. The custom-block
path is documented as stateless-feedthrough; persistence is via the cache. The
original RFC's conclusion ("no states, as in: not officially supported") was the
right call and remains in force.

---

## DEC-028: simulate_batch distinguishes static-parameter and dynamic-parameter paths

**Status**: Accepted

### Context

`simulate_batch` has two qualitatively different use cases. (1) Sweeping over
dynamic parameters — the same compiled simulation function is called many times
with different numeric arguments; this is the natural vmap case. (2) Sweeping
over static parameters — the parameter affects control flow, dtype, or shape;
changing it requires recompilation. The earlier (Collimator-era) implementation
silently failed on case (2), producing wrong results without errors (the WC-357
bug class).

### Decision

`simulate_batch` (`simulation/batch.py`) implements two paths: a kernel path
that uses `jax.vmap` (correct only for dynamic parameters) and a loop path that
recompiles per iteration (correct for both, slower for dynamic). The path is
selected based on whether the swept parameters are dynamic or static.
Static-parameter sweeps land on the loop path with per-run recompilation rather
than being silently miscomputed.

### Rationale

Correctness over speed when the user is iterating something that breaks vmap.
Silent miscomputation under iteration was a serious bug in the legacy
implementation; the loop fallback eliminates it. Users who explicitly want vmap
performance can structure their parameters as dynamic and get the kernel path.

### Alternatives considered

- vmap-only (rejected: silently wrong for static parameters, the WC-357 class
  of bug)
- Loop-only (rejected: throws away vmap performance for the common dynamic case)
- Require user to pick the path (rejected: most users don't know which they
  need)

### Consequences

`simulate_batch` is correct on both parameter classes by default. Users sweeping
static parameters pay recompilation cost per iteration; this is documented. The
path selection is automatic but the cost difference is large enough that it
matters for benchmarking and should be visible in profiling.

---

## DEC-027: Mass-matrix ODE form for DAEs; no fully-implicit support

**Status**: Accepted
**Supersedes**: DEC-014

### Context

DAEs come in multiple forms: ODE (no algebraic constraints), semi-explicit DAE
(separable differential and algebraic states), mass-matrix ODE form
(`M(x)·ẋ = f(t,x,p)`, generalizes semi-explicit), and fully implicit
(`F(t, x, ẋ) = 0`, the most general). Each form requires different solver
capabilities.

### Decision

Jaxonomy supports ODE and the mass-matrix DAE form. The BDF solver advertises
`supports_mass_matrix=True`; the mass matrix may be diagonal (recovering
semi-explicit) or general. Fully implicit DAEs (`F(t, x, ẋ) = 0`) are not
supported.

### Rationale

The mass-matrix form is significantly more general than the diagonal
semi-explicit form (it covers FEM discretizations, constrained mechanics,
circuits, and index-2 DAEs in many cases) without the implementation complexity
of full implicit-DAE solvers. The acausal layer's index-reduced output naturally
fits the mass-matrix form. Fully implicit support would require either a
substantially different solver or an external library and is not justified by
current user needs.

### Alternatives considered

- Diagonal-mass-matrix only / semi-explicit (rejected: unnecessarily restrictive
  given the implementation is similar)
- Fully implicit DAE solvers (rejected: solver-implementation complexity,
  marginal additional coverage)

### Consequences

Models that genuinely require the fully-implicit form (rare, mostly exotic
mechanical or chemical systems) cannot be represented. The acausal layer must
produce mass-matrix output. Constraint projection (preventing algebraic-variable
drift) is not currently implemented and is on the gap list.

---

## DEC-026: State machines are flat, Mealy-semantics, deterministic-by-transition-order

**Status**: Accepted

### Context

State-machine semantics is one of the most over-specified domains in modeling
tools — UML statecharts, Harel statecharts, classical FSMs, Mealy vs. Moore,
hierarchical vs. flat, event-driven vs. condition-driven. A choice has to be
made about which subset to implement.

### Decision

Jaxonomy state machines (`library/state_machine.py`) implement: Mealy semantics
(actions on transitions), flat (no hierarchical states), condition-based
transitions (no events/messages), deterministic execution by transition order
(lowest-index successful guard fires), with `initial_actions` run once at init
but no per-state `on_enter` / `during` / `on_exit` hooks.

### Rationale

This is the "v1" subset from the original Statecharts RFC and it covers the
dominant practical use cases for control-systems state machines. Hierarchy,
events, and lifecycle hooks add substantial implementation and semantic
complexity. Determinism by transition order is unambiguous and predictable;
non-deterministic (or event-priority) semantics are a much harder design
question.

### Alternatives considered

- Full UML statecharts (rejected: scope, semantic ambiguities, low ROI for
  control systems)
- Moore semantics (actions on states) (rejected: less compositional with the
  rest of the block-diagram model)
- Hierarchical states from day one (deferred: planned-future, not implemented)

### Consequences

Users wanting hierarchical states must flatten their state machines manually.
Event/message passing between state machines is not supported — communication is
via signal inputs only. The "Possible future features" list in the original RFC
remains the natural extension path; none of it is implemented today.

---

## DEC-025: JSON model addressing uses parent-path + UUID, not bare UUID

**Status**: Accepted

### Context

The original Collimator JSON format used bare node UUIDs as identifiers, which
broke on multiply-instantiated submodels (the same submodel UUID appearing under
different parents). The RFC: Model representation flagged this as a critical
defect.

### Decision

The current serialization (`dashboard/serialization/from_model_json.py`)
addresses nodes by `(parent_path, ui_id)` — an ancestor ID list plus the local
UUID. This makes addressing unique even when the same submodel is instantiated
multiple times in different positions.

### Rationale

Required for correctness in models with reused submodels. The parent-path
approach is what the RFC concluded is right.

### Alternatives considered

- Globally unique UUIDs per instantiation (rejected: breaks reuse semantics)
- Synthetic addressing scheme (rejected: parent-path is the natural one)

### Consequences

Round-trip fidelity is preserved across instantiations. Schema versioning is
read and a guard rejects too-old schemas, but no migration tooling exists — old
files that don't match the current schema can't be auto-upgraded. This is on the
gap list.

---

## DEC-024: ErrorCollector — accumulate diagnostics, don't fail on first error

**Status**: Accepted

### Context

Diagram construction, type checking, and parameter binding can produce multiple
independent errors in a single user model. Failing on the first one and showing
only that to the user produces a frustrating "fix one error, get the next" loop.
The chosen pattern is an `ErrorCollector` that gathers errors through a stack of
operations and surfaces them together.

### Decision

`framework/error.py` provides an `ErrorCollector` that participates as a
context manager and accumulates errors across the operations within its scope.
Used by `library/custom.py`, `library/linear_system.py`, and other consumers.
`check_types` validates port shapes/dtypes at diagram-build time and reports
through this mechanism.

### Rationale

Better UX: surface all diagnosable errors at once. Maps cleanly to functional
patterns (the original RFC considered an Either monad and chose the simpler
`ErrorCollector` approach for Python). `check_types` at build time catches a
class of errors before simulation rather than during it.

### Alternatives considered

- Either-monad pattern (PyMonad) (rejected: less idiomatic in Python, more
  conceptual overhead)
- Fail-on-first (rejected: bad UX)
- Logging-only (rejected: no structured access for callers)

### Consequences

Library code that wants to participate in error collection must call into
`ErrorCollector` rather than raising directly. JAX/XLA tracebacks are not yet
systematically remapped to block/port names — this is a known gap.

---

## DEC-023: Custom VJPs for the simulator, including Cao mass-matrix adjoint correction

**Status**: Accepted

### Context

Reverse-mode autodiff through long simulation trajectories with event handling
is not something Diffrax provides out of the box for the specific shape of
Jaxonomy's simulator. Naive autodiff through `advance_to` either fails (events
break the trace) or produces incorrect gradients (DAE adjoints need an
initial-condition correction not present in standard ODE adjoint methods).

### Decision

`simulation/autodiff_rules.py` implements custom `jax.custom_vjp` rules for
`advance_to` and `guarded_integrate`. The DAE adjoint includes the Cao et al.
(2003) mass-matrix initial-condition correction, ensuring that gradients are
correct for mass-matrix DAE systems and not just plain ODEs.

### Rationale

Required for end-to-end differentiability (DEC-016) on the actual simulator
shape Jaxonomy supports. The Cao correction is the published-correct way to
handle mass-matrix DAE adjoints; not including it produces silently wrong
gradients on DAE problems, which is exactly the failure class DEC-016 exists to
prevent.

### Alternatives considered

- Rely on Diffrax adjoints alone (rejected: doesn't handle the simulator's event
  semantics or DAE adjoint correction)
- Forward-mode only (rejected: inefficient for typical parameter counts)
- Numerical gradients (rejected: noisy, slow, not composable)

### Consequences

Adding new event-handling logic or solver capabilities requires updating the
custom VJPs in lockstep. The autodiff rules are some of the most subtle code in
the repo; changes there warrant extra review even with the lighter PR-review
policy. `jacfwd` is used internally inside the VJP for the IC correction but is
not exposed as a user-facing forward-mode path.

---

## DEC-022: AGENTS/ directory structure for agent-driven development

**Status**: Accepted

### Context

Development increasingly involves AI coding assistants (Claude Code,
Cursor, and similar tools). These agents work best with structured orientation
material: project context, coding patterns, architectural decisions, task
specs, and session handoff state. A single `README.md` plus tribal knowledge
does not scale across many short-lived sessions, each of which starts fresh
without context.

### Decision

Maintain an `AGENTS/` directory at the repository root containing a
versioned set of orientation files: `README.md` (meta-instructions),
`CONTEXT.md` (project state and architecture), `PATTERNS.md` (coding
conventions), `DECISIONS.md` (this file), `RULES.md` (operating
principles), plus topic-specific files (`SCOUTING_*.md`) as needed.
Agent sessions read these at the start and update CHANGELOG as work
progresses. Cross-session state lives in `git status` / `git log`, not
a separate handoff file.

### Rationale

Explicit, structured, version-controlled agent context reduces cycle time on
every new session and prevents re-litigating settled questions. The structure
is compatible with multiple agent tools, not bound to any single one.

### Alternatives considered

- Informal `README.md` only: does not scale, does not capture why decisions
  were made, rapidly goes stale
- Proprietary tool-specific formats (`.cursor/`, etc.): ties the repo to a
  single vendor
- Wiki or external docs: harder to keep synchronized with code changes
- **Gitignored per-worktree `HANDOFF.md` for cross-session state**:
  tried and dropped. Made the bootstrap fragile (no template, no
  guaranteed existence) and broke escalation, because the maintainer
  reviews `git log` and never sees per-worktree-local files. Git state
  + commit messages + PR descriptions are the maintainer-visible
  channel.

### Consequences

Contributors maintain these files alongside code; all of them are
committed and reviewed. Per-worktree session bookkeeping (what's
in-flight) lives in the branch's commits, not in a separate file.

---

## DEC-021: Testing strategy — pytest, backend-parametrized, property-based where applicable

**Status**: Accepted

### Context

The simulation engine must give correct results across multiple backends (JAX
primary, NumPy fallback), across multiple solver choices, and across many
subtle edge cases (stiff systems, near-singular Jacobians, events arbitrarily
close in time, long-horizon drift). Hand-written example-based tests miss
categories of bugs that property-based testing catches routinely.

### Decision

Use pytest as the test runner. Parametrize tests across backends wherever both
backends are supported. Use property-based testing (Hypothesis) for invariants
that should hold over a space of inputs — conservation laws, backend parity,
gradient correctness via finite differences, solver monotonicity properties.
Integration tests run against real JSON models to catch regressions in the
full stack.

### Rationale

Backend parametrization catches regressions where one backend diverges from
the other silently. Property tests catch classes of bugs (off-by-one,
precision, tolerance) that example tests miss. Integration tests catch
composition bugs that unit tests miss.

### Alternatives considered

- JAX-only testing: misses NumPy backend regressions
- Example-based tests only: misses edge cases
- Manual validation scripts: not reproducible, not CI-integrated

### Consequences

Test suite is larger and slower than a simple unit-test-only approach. CI must
be configured to run the full matrix. Some tests are inherently slow (long
simulations, property searches); tag these and run separately in nightly CI.

---

## DEC-020: Custom Python blocks require environment isolation

**Status**: Accepted

### Context

`CustomPythonBlock` lets users write arbitrary Python code inside a block. If
multiple instances of the same custom block share module-level state (globals,
imports, cached objects), state leaks between them and breaks both vmap safety
and reproducibility. Early versions of this capability had exactly these bugs.

### Decision

Each `CustomPythonBlock` instance executes in an isolated persistent
environment — imports and module-level state are scoped per instance, not
shared across the simulation. User code cannot accidentally depend on global
Python state of other blocks.

### Rationale

Isolation is required for vmap safety (each vmapped branch must be
independent), for reproducibility, and for users' intuition about blocks as
self-contained components.

### Alternatives considered

- Shared Python environment (rejected: breaks vmap, reproducibility)
- Forbid custom Python blocks entirely (rejected: kills a key extensibility
  story)
- Sandboxing via subprocess (rejected: too heavy for hot-path code)

### Consequences

Custom Python block construction is slightly more expensive (per-instance
environment setup). Users can rely on blocks being independent. Any shared
state must be explicit, via `Context` parameters.

---

## DEC-019: Embedded codegen lives in a separate downstream library

**Status**: Accepted

### Context

Generating deployable embedded code (MISRA-compliant C, acados integration,
LiteRT/ExecuTorch export paths, hardware-specific compilation) is a natural
extension of Jaxonomy's capabilities. But it has heavyweight dependencies
(acados, cross-compilers, target-specific toolchains), different user
personas (embedded engineers vs. simulation users), and potentially different
licensing considerations. Bundling all of this into Jaxonomy would bloat
the dependency surface and muddle the library's identity.

### Decision

Jaxonomy itself does not implement embedded codegen. A separate downstream
library (to be developed) will consume Jaxonomy models and produce deployable
artifacts. Jaxonomy's responsibility is to preserve the properties required
for downstream codegen: clean model serialization, exposed dynamics
functions, deterministic behavior, and compatibility with JAX's XLA
compilation path.

### Rationale

Separation of concerns. Different release cadences, different dependency
profiles, different users. Keeps Jaxonomy focused on simulation correctness.

### Alternatives considered

- Include codegen in Jaxonomy (rejected: dependency bloat, scope creep)
- External third-party tools only (rejected: no way to guarantee Jaxonomy
  models are codegen-friendly without first-party coordination)

### Consequences

Jaxonomy must not break properties needed by the downstream library: model
serialization stability, dynamics function purity, deterministic numerics.
Any refactor that would break these requires coordination.

---

## DEC-018: GUI and cloud-hosted features are not current scope

**Status**: Accepted

### Context

The Collimator product from which Jaxonomy was extracted had a WebGL-based
graphical editor, real-time collaboration, cloud-hosted HPC, project
permissions, version history, and similar cloud-platform features. These were
substantial engineering investments. The question is whether to attempt to
maintain or rebuild them in the open-source library.

### Decision

GUI, cloud hosting, multi-user collaboration, and HPC orchestration are not
currently in Jaxonomy's scope. The library provides a Python API, a JSON
model format, and local execution. It does not maintain a browser-based
editor, a cloud backend, or real-time collaborative editing.

This decision is about focus, not exclusion. If the project's goals evolve
such that a GUI or cloud capability becomes strategically important, the
decision can be revisited and superseded. The JSON model format and
API-first design deliberately keep the door open to future GUI or
cloud-platform development.

### Rationale

Maintaining a GUI or a cloud platform is a full team-years effort on its own.
The core library has more pressing correctness and capability work.
Concentrating resources on the engine produces a better foundation that any
future UI could sit on top of.

### Alternatives considered

- Rebuild the Collimator GUI as OSS (rejected for scope reasons; revisitable)
- Build a lightweight new GUI (deferred; no current user demand justifying it)
- Integrate with existing visualization tools (Rerun, etc.) — this is a
  lighter-weight alternative that may happen regardless

### Consequences

Users wanting a visual editor currently rely on external tools or code-first
workflows. Documentation should be clear about the Python-first workflow.
The JSON model format is maintained as a stable surface that a future GUI
could target.

---

## DEC-017: `simulate_batch` uses selective vmap, not pmap or manual parallelization

**Status**: Accepted

### Context

Parallel simulation is a primary use case: parameter sweeps, Monte Carlo
studies, ensemble training, optimization over populations. JAX offers
multiple parallelization primitives (`vmap`, `pmap`, `shard_map`, manual
loops), each with different semantics and different performance
characteristics.

### Decision

`simulate_batch` is implemented using `vmap` with selective axis control —
users specify which inputs are batched vs. broadcast. It does not use `pmap`
(which is for multi-device parallelism, a different abstraction) and does
not fall back to Python-level loops (which lose JIT benefits).

### Rationale

`vmap` is the natural JAX primitive for "same computation, many inputs" and
composes cleanly with `jit`, `grad`, and other transformations. Selective
batching avoids the common footgun of accidentally batching things that
should be shared across the ensemble.

### Alternatives considered

- `pmap` (rejected: wrong abstraction for single-device batching)
- Python loops (rejected: loses JIT, 100x slower)
- `shard_map` (deferred: relevant if multi-device support becomes a priority)

### Consequences

Users must understand the batched-vs-broadcast distinction. Multi-device
parallelism is not currently supported through this API; a future addition
may use `shard_map` or `pmap` if needed.

---

## DEC-016: End-to-end differentiability is a non-negotiable invariant

**Status**: Accepted

### Context

Jaxonomy's design thesis (see CONTEXT.md) is that modeling is most valuable
when end-to-end differentiable and optimization-ready. Every time a new
feature is added, there is a temptation to take a shortcut that breaks
differentiability for a local convenience — a Python-level conditional, a
non-traceable external call, a stateful accumulator. Once differentiability
is broken in one layer, all downstream optimization workflows silently fail
or produce wrong gradients.

### Decision

All simulation-path code must preserve reverse-mode autodiff through the full
trajectory, including event handling, state machine transitions, and
acausal constraint solving. Features that cannot be made differentiable
must explicitly document the limitation and provide a non-differentiable
opt-out path, not a silent break.

### Rationale

Silent gradient breakage is one of the most expensive classes of user-facing
bug — users get numbers back, the optimization runs, the result is wrong.
Making differentiability a hard invariant forces us to confront the
tradeoff at design time rather than at user-debugging time.

### Alternatives considered

- Forward-mode only (rejected: inefficient for typical parameter counts)
- Differentiability as opt-in (rejected: creates silently-broken workflows)
- Differentiability best-effort (rejected: same problem as opt-in)

### Consequences

Every new feature must prove it preserves reverse-mode autodiff. Custom
adjoint methods are sometimes required (event handling in particular).
Non-differentiable features (e.g., discrete sampling at simulation time)
must be explicit about their gradient behavior.

---

## DEC-015: Events handled via zero-crossing detection with guard intervals and hysteresis

**Status**: Accepted

### Context

Hybrid dynamical systems have events — moments where discrete logic fires
based on continuous state (e.g., "valve closes when pressure exceeds
threshold"). Naive event detection (check condition at every step) fails in
two ways: numerical noise near the event surface causes spurious
re-triggering (chatter), and fixed-step polling misses events that occur
between steps or localizes them imprecisely.

### Decision

Events are detected via zero-crossing of guard functions between simulation
steps, using bracketing + refinement to localize the crossing time.
Hysteresis bands and guard intervals prevent re-triggering when the system
lingers near the event surface.

### Rationale

Standard approach in hybrid systems simulation. Matches Simulink, Dymola,
Drake behavior. Robust against numerical chatter. Supports variable-step
solvers cleanly.

### Alternatives considered

- Exact event localization only (rejected: fragile without hysteresis)
- Fixed-step polling (rejected: inaccurate, misses events)
- Implicit event handling via DAE constraints (rejected: overkill for most
  events, has performance cost)

### Consequences

Guard functions must be authored carefully — they should be smooth and
sign-changing across the event. Hysteresis parameters are user-tunable and
have correctness implications if set wrong.

---

## DEC-014: Support semi-explicit DAEs, not fully implicit

**Status**: Superseded by DEC-027

### Context

Differential-algebraic equations come in several forms: ODE (pure
differential), semi-explicit DAE (differential + algebraic constraints,
separable), and fully implicit DAE (arbitrary implicit relationships among
states, derivatives, and inputs). Semi-explicit covers the vast majority of
engineering use cases. Fully implicit adds substantial solver complexity
and covers edge cases most users don't need.

### Decision

Jaxonomy's simulation engine supports ODEs and semi-explicit DAEs. Fully
implicit DAEs are not supported.

### Rationale

Semi-explicit covers electrical circuits with Kirchhoff laws, mechanical
systems with constraints, thermal networks, hydraulic systems, and the
common outputs of Pantelides index reduction. Fully implicit adds complexity
that would primarily benefit a narrow set of users.

### Alternatives considered

- Fully implicit DAEs (rejected: solver complexity, user confusion,
  marginal additional coverage)
- ODE-only (rejected: cannot handle acausal modeling outputs)

### Consequences

Some Modelica models with fully implicit formulations won't import directly
— they require reformulation or index reduction upstream. Document the
limitation clearly.

---

## DEC-013: Diffrax as the continuous-time ODE solver

**Status**: Accepted

### Context

The simulation engine needs an ODE solver that is (a) JAX-native so autodiff
flows through it, (b) supports a range of solvers including stiff methods,
(c) actively maintained, (d) supports variable-step integration, and (e)
composes cleanly with JIT and vmap.

### Decision

Use Diffrax as the primary continuous-time ODE/SDE solver library.

### Rationale

Diffrax is JAX-native, differentiable, actively maintained by Patrick Kidger
and the JAX scientific community, includes a broad range of solvers (RK4,
Dopri, Kvaerno, Tsit5, etc.), supports variable-step and adaptive control,
and is the de facto standard for differentiable ODE solving in JAX.

### Alternatives considered

- `scipy.integrate` (rejected: not differentiable in a JAX-native way,
  poor JIT integration)
- Hand-written solvers (rejected: reinventing maintained work, harder to
  support stiff methods)
- `torchdiffeq` (rejected: wrong framework)
- `DifferentialEquations.jl` with Python bindings (rejected: language
  boundary, dependency complexity)

### Consequences

Jaxonomy depends on Diffrax's API stability. Diffrax version pinning
matters for reproducibility. Solver configuration is passed through to
Diffrax with thin adaptation layers.

---

## DEC-012: Pantelides index reduction + SymPy for acausal DAE handling

**Status**: Accepted

### Context

Acausal modeling (Modelica-style) requires reducing a declarative equation
system — potentially with higher-index DAE structure — into a form the
solver can integrate. This requires symbolic manipulation and an index
reduction algorithm.

### Decision

Use the Pantelides algorithm for structural index reduction, with SymPy as
the underlying symbolic math library. Domain-specific component libraries
(electrical, mechanical, thermal, hydraulic) express their equations
symbolically, Pantelides reduces the index, and the reduced system is
handed to the numerical solver.

### Rationale

Pantelides is the industry-standard algorithm, well-understood, handles a
broad class of acausal systems, and composes with symbolic differentiation.
SymPy is the canonical Python symbolic library — imperfect but mature and
widely available.

### Alternatives considered

- Hand-written index reduction (rejected: significant effort to match
  Pantelides generality)
- CasADi-based symbolic layer (rejected: would introduce a second symbolic
  library; integration complexity)
- Numerical index reduction (rejected: less robust, worse diagnostics)

### Consequences

SymPy is a heavy dependency and symbolic manipulation is slow on large
systems. Acausal module is the public `jaxonomy.acausal` surface
(promoted out of `experimental/` once the API stabilised). Some
acausal system classes require user hints to reduce cleanly.

---

## DEC-011: Retain Collimator JSON model format for serialization

**Status**: Accepted

### Context

Models need a serialization format for storage, reproducibility, round-trip
fidelity with external tools, and potential downstream consumers (codegen,
hypothetical future GUI tooling). A format decision affects compatibility
with existing Collimator models and future tool interoperability.

### Decision

Retain the Collimator JSON model format as the canonical serialization
surface. Maintain round-trip fidelity: load → save → load produces
bit-identical state.

### Rationale

Compatibility with existing Collimator models preserves user continuity.
JSON is human-readable, tool-agnostic, and stable. An API-first design
with a serializable model format keeps downstream tools (codegen, future
editors) possible without committing to specific implementations now.

### Alternatives considered

- `pickle` / `dill` (rejected: not portable, Python-version-dependent,
  security concerns)
- Protocol Buffers (rejected: schema management overhead, not human-
  readable)
- A new Jaxonomy-native format (rejected: fragmentation without benefit)

### Consequences

JSON schema evolution must be handled carefully; breaking changes require
migration tooling. Round-trip fidelity is a tested invariant. Schema
validation catches user errors early.

---

## DEC-010: Callback signature uniformity — `(time, state, *inputs, **params)`

**Status**: Accepted

### Context

`LeafSystem` has multiple kinds of callbacks: continuous dynamics (ODE
right-hand side), discrete updates, output functions, guard functions
(event triggers), reset maps (event actions). Each could plausibly have a
different signature tailored to its role, or they could share a single
signature.

### Decision

All simulation callbacks follow a single uniform signature:
`(time, state, *inputs, **params)`. Return value varies by callback type
(derivative, next state, output value, scalar guard, reset state).

### Rationale

Uniformity reduces cognitive load for users writing custom blocks. It
enables generic higher-order transformations (wrappers, instrumentation,
backend dispatch) that operate on callbacks without per-type special
casing. Consistent with JAX's functional style.

### Alternatives considered

- Per-callback-type signatures (rejected: inconsistent, harder to wrap)
- Keyword-only signatures (rejected: awkward for `time` and `state`)
- Positional-only (rejected: too restrictive on parameter passing)

### Consequences

User code has a single pattern to learn. Callbacks that don't need inputs
or params still receive them (or can omit via `*args, **kwargs`).
Documented in PATTERNS.md.

---

## DEC-009: NamedTuple-based discrete state with inner class conventions

**Status**: Accepted

### Context

Discrete state within blocks needs a structured representation that is
JAX-compatible (PyTree), supports static field names (not dict keys), works
with IDE tooling, has static shape/dtype, and composes cleanly with vmap.

### Decision

Discrete state is represented via inner NamedTuple classes following
consistent conventions: `DiscreteStateType`, `RNGState`, etc. Fields are
accessed by name. NamedTuples register as PyTrees automatically.

### Rationale

NamedTuples give static field names (IDE autocomplete, type checking),
are immutable (matches JAX's functional model), PyTree-compatible without
extra registration, and have clear semantics. Inner-class convention keeps
each block's state definition scoped to the block.

### Alternatives considered

- Plain dicts (rejected: no static shape, no IDE support, error-prone)
- `@dataclass` (rejected: PyTree registration overhead, less idiomatic in
  JAX code)
- Arrays with index constants (rejected: error-prone, no field names,
  fragile refactoring)

### Consequences

Users writing blocks follow the inner-class pattern. Documented in
PATTERNS.md with canonical examples.

---

## DEC-008: `@parameters` decorator as the canonical parameter declaration mechanism

**Status**: Accepted

### Context

Blocks need a way to declare parameters that is discoverable, supports both
static and dynamic parameters, integrates with `with_parameters()` for
optimization workflows, and is consistent across the codebase.

### Decision

The `@parameters(dynamic=[...])` decorator on `__init__` is the canonical
mechanism for declaring block parameters. It distinguishes dynamic
parameters (can be varied without recompilation, flow through autodiff)
from static parameters (recompilation on change, baked into the JIT'd
graph).

### Rationale

Declarative decoration is more discoverable than imperative registration.
The static/dynamic distinction matches JAX's JIT semantics and is the
single most important parameter property for users to reason about.

### Alternatives considered

- Imperative `self.declare_parameter(...)` calls (rejected: less
  discoverable, inconsistent with rest of API)
- Convention based on naming (rejected: fragile, no tooling support)
- Separate static/dynamic decorators (rejected: duplicated API surface)

### Consequences

Every block uses this pattern. Documented in PATTERNS.md. Changes to the
decorator semantics must preserve round-trip compatibility with existing
blocks.

---

## DEC-007: Two-phase construction — `__init__` declares, `initialize()` configures

**Status**: Accepted

### Context

Block construction has two concerns: declaring the static shape of the block
(ports, state, parameters) and configuring runtime behavior (connections,
initial conditions, parameter values). Mixing these in a single `__init__`
makes subclassing awkward — subclasses that want to override configuration
end up re-declaring structure.

### Decision

Two-phase construction: `__init__` declares structure (ports, state,
parameters via `@parameters`, callback registrations), and `initialize()`
configures runtime values. `**kwargs` are threaded through `__init__` for
parameter forwarding to parent classes.

### Rationale

Separation of structure from configuration mirrors JAX's separation of trace
time from runtime. Subclasses can override `initialize()` without
re-declaring structure. Consistent with the System/Context pattern where
System is structure and Context is values.

### Alternatives considered

- Single-phase `__init__` (rejected: awkward subclassing)
- Builder pattern for all blocks (rejected: too verbose for simple blocks)
- Configuration methods after construction (rejected: no enforcement of
  "must configure before use")

### Consequences

Developers must understand which work goes in which phase. Documented in
PATTERNS.md with the LeafSystem skeleton.

---

## DEC-006: Backend dispatch via `npa` abstraction

**Status**: Accepted

### Context

Most numerical code in Jaxonomy is backend-agnostic — it works identically
on `jax.numpy` or `numpy` arrays. But some code is backend-specific
(JAX-specific transformations, NumPy-specific behavior). Without an
abstraction, code accumulates either `jnp`/`np` sprinkled throughout
(fragile, per-file inconsistency) or `if backend == 'jax'` dispatch
(verbose, error-prone).

### Decision

Introduce an `npa` ("NumPy abstraction") module that dispatches to `jnp` or
`np` based on the active backend. Most numerical code uses `npa`. Direct
use of `jnp` or `np` is reserved for backend-specific paths with documented
reasons.

### Rationale

Single-symbol dispatch keeps call sites clean. Testing on NumPy backend
catches JAX-specific assumptions (e.g., accidental reliance on tracer
shapes). Debug workflows benefit from NumPy fallback (clearer stack traces,
no XLA compilation).

### Alternatives considered

- JAX-only (rejected: loses debugging fallback, no parity testing)
- Separate JAX and NumPy code paths (rejected: duplication, drift)
- Runtime `if` dispatch (rejected: verbose, breaks JIT in some cases)

### Consequences

Every numerical call site uses `npa` by default. Developers must remember
not to reach for `jnp` directly without reason. Documented in PATTERNS.md
with decision rules.

---

## DEC-005: IntegerTime (picosecond resolution) for event ordering

**Status**: Accepted

### Context

In long or event-dense simulations, floating-point time representation
accumulates error. Two events scheduled at "the same" time can be ordered
inconsistently across runs, breaking determinism. Comparisons like
`t == event_time` are fragile. This is a well-known source of bugs in
hybrid systems simulation.

### Decision

Internal event time is represented as an integer in picoseconds.
Floating-point time is used only where mathematically required (inside
ODE solver steps, in user-facing APIs that accept `float` t). Comparisons,
ordering, and event scheduling all operate on IntegerTime.

### Rationale

Integer representation is exact. Ordering is unambiguous. Picosecond
resolution is fine for all practical simulations (1 ps = 10^-12 s; typical
simulation horizons are seconds to hours). Eliminates a class of bugs
entirely.

### Alternatives considered

- Double-precision float time (rejected: fails in long runs)
- Rational arithmetic (rejected: too slow, complex comparisons)
- Higher integer resolution (femtoseconds etc.) (rejected: picosecond is
  more than sufficient, adds cost)

### Consequences

Conversions between IntegerTime and float are concentrated at specific
layer boundaries (solver step start/end, user-facing APIs). Picosecond
resolution is a documented hard limit.

---

## DEC-004: Pluggable backends with JAX as primary, NumPy as fallback

**Status**: Accepted

### Context

Jaxonomy's primary backend is JAX, but a NumPy fallback is valuable for
debugging (clearer errors, no XLA compilation), for environments where JAX
is awkward to install, and for cross-backend testing that catches subtle
JAX-specific bugs.

### Decision

Support JAX (primary, full-featured) and NumPy (fallback, reduced feature
set) backends. Some features (autodiff, JIT, vmap) are JAX-only by nature.
Partial PyTorch dispatch exists for specific ML block wrappers but is not
a general backend.

### Rationale

JAX gives the headline features (autodiff, JIT, GPU/TPU). NumPy gives
debugging clarity and broader install compatibility. Cross-backend parity
testing catches bugs that single-backend testing misses.

### Alternatives considered

- JAX-only (rejected: no debug fallback, installation friction for some
  users)
- Three-way parity with PyTorch (rejected: enormous maintenance burden,
  JAX/PyTorch semantics differ enough to make full parity impractical)

### Consequences

Backend-neutral code uses `npa` (DEC-006). JAX-only features are clearly
documented as such. NumPy backend is regression-tested but not a first-
class target for new capabilities.

---

## DEC-003: Support both causal (block-diagram) and acausal (equation-based) modeling

**Status**: Accepted

### Context

Engineering models come in two paradigms. Causal (block-diagram,
signal-flow) is natural for controllers, discrete logic, and systems where
inputs-cause-outputs is clear. Acausal (equation-based, Modelica-style) is
natural for physical systems where components connect via shared physical
variables (voltage, force, pressure) without a predetermined causality.
Real engineering models routinely need both — a physical plant is acausal,
its controller is causal.

### Decision

Jaxonomy supports both paradigms as first-class citizens. Causal
block-diagram composition is the core framework (System / LeafSystem /
Diagram / Context, DEC-002). Acausal modeling with Pantelides index
reduction (DEC-012) is an additional layer that produces Systems consumable
by the same simulation engine.

### Rationale

Neither paradigm is sufficient alone. Forcing users to pick one imposes
awkward workarounds — pure-causal users end up hand-deriving physical
models; pure-acausal users end up awkwardly expressing discrete logic as
equations. Both first-class is the only honest answer.

### Alternatives considered

- Causal-only (rejected: forces manual derivation for physical systems)
- Acausal-only (rejected: overkill for controllers and discrete logic,
  performance cost)
- Two separate libraries (rejected: integration is the whole point)

### Consequences

Two parallel subsystems must be maintained. Interop between causal and
acausal is a persistent engineering concern. Users must understand when
to use which. Documented extensively.

---

## DEC-002: Adopt Drake's System / LeafSystem / Diagram / Context abstractions

**Status**: Accepted

### Context

The block-diagram paradigm has many possible API shapes — Simulink-style
flat composition, callable-based, actor model, dataflow graph, and so on.
A specific set of abstractions had to be chosen for Jaxonomy's core.

### Decision

Adopt the abstractions from Drake (MIT's C++/Python robotics modeling
framework): `System` (abstract behavior), `LeafSystem` (atomic blocks),
`Diagram` (composite systems), `Context` (tree-structured state container
mirroring the System tree). Naming is kept aligned with Drake to ease
onboarding for users familiar with it.

### Rationale

Drake's abstractions have desirable properties: clean separation of
structure from state, rigid semantics (close correspondence between
simulated behavior and mathematical representation), natural hierarchical
composition, and a clean mapping onto JAX's functional paradigm (System as
pure function definition, Context as function arguments). Drake is
well-regarded in the robotics and controls community and its abstractions
are battle-tested.

### Alternatives considered

- Simulink-style flat block composition (rejected: hierarchical composition
  is essential for complex models)
- Callable-based functional API (rejected: loses structural information
  that tools need — serialization, inspection, codegen)
- Modelica-style equation-only (rejected: doesn't fit causal or discrete
  systems well)
- Invent new abstractions (rejected: Drake's are good; not-invented-here
  would create onboarding friction)

### Consequences

Hierarchical models require Context tree navigation. Some users find this
unfamiliar initially — documented in CONTEXT.md. Drake users can onboard
quickly. The underlying implementation differs substantially from Drake's
C++ because of the JAX functional paradigm, but the mental model
transfers.

---

## DEC-001: Use JAX as the primary computational backend

**Status**: Accepted

### Context

Jaxonomy's design philosophy (CONTEXT.md) centers on end-to-end
differentiable simulation for optimization workflows. This requires a
computational backend that supports autodiff through complex compositions
of code (including ODE solvers, event handling, custom blocks), is
performant enough for production use, supports hardware acceleration
(GPU/TPU), and is Python-native so users can write code that extends the
engine at the same level as built-in functionality.

### Decision

Use JAX as the primary computational backend. All simulation-path code is
written in JAX-compatible pure functions. JIT compilation, autodiff
(`grad`, `jacfwd`, `jacrev`), and vectorization (`vmap`) are primitive
operations the library assumes throughout.

### Rationale

JAX uniquely combines: (a) autodiff that composes cleanly through solvers
and control flow; (b) JIT compilation via XLA for near-native performance;
(c) GPU/TPU acceleration via the same flag; (d) Python-native so users
can extend the engine without crossing a language boundary; (e) functional
paradigm that maps cleanly onto System + Context. Google DeepMind's
adoption of JAX for robotics and scientific computing (including MuJoCo's
JAX port) validates it for this domain.

### Alternatives considered

- PyTorch (rejected: less functional, autodiff through custom solvers
  harder, eager-mode dominance conflicts with our JIT-first design)
- Pure NumPy (rejected: no autodiff, no GPU, no JIT)
- Julia (rejected: non-Python, contradicts accessibility goal)
- C++ with Python bindings (rejected: extensibility wall — users can't
  extend the engine at the same level as built-in functionality without
  writing C++; Drake's experience confirms this friction)
- TensorFlow (rejected: eager-mode shift made it less suitable for
  scientific computing; JAX has stronger momentum in this space)

### Consequences

Users must learn JAX idioms (pure functions, PyTrees, tracing). Some
Python control flow is restricted in hot paths (see PATTERNS.md).
Installation requires JAX, which has GPU/TPU setup friction on some
platforms. The library's capability ceiling is tied to JAX's feature
evolution.

---

## Proposed / open decisions

Entries below are decisions that have been surfaced but not yet made. They
are recorded here so agents know the question is open, not overlooked.

---

## DEC-P02: Multi-device parallelism strategy (proposed)

**Status**: Proposed

### Context

Current parallelism is single-device via `vmap` (DEC-017). Users with
multi-GPU or TPU-pod setups can't currently distribute ensembles across
devices through a first-class API. `pmap` and `shard_map` are JAX's
primitives for this.

### Open question

Should Jaxonomy expose a first-class multi-device API, and if so, via
`pmap`, `shard_map`, or a higher-level abstraction that selects between
them?

### Current direction

No action until there is concrete user pull. `simulate_batch` users can
`pmap(simulate_batch(...))` manually today for specific workloads.

---

## DEC-P01: TPU-specific code path strategy (proposed)

**Status**: Proposed

### Context

TPU support is on the roadmap. Some JAX operations behave subtly
differently on TPU (precision defaults, memory layout, synchronization).
The question is whether to maintain TPU-specific code paths or rely on
XLA to paper over the differences.

### Open question

Do we need TPU-specific branches, or does XLA's TPU backend handle
everything?

### Current direction

Rely on XLA, validate via TPU test runs. If TPU-specific bugs emerge that
can't be fixed upstream, reconsider.

---

## Template for new decisions

Copy this template when adding a new entry.

```
## DEC-NNN: <Short declarative title>

**Status**: Accepted | Proposed | Superseded by DEC-XXX | Deprecated

### Context
What problem? What constraints? Why is a decision needed now?

### Decision
What did we decide? State declaratively.

### Rationale
Why this over alternatives?

### Alternatives considered
- Option A: rejected because...
- Option B: rejected because...

### Consequences
What does this force? What does it prevent? What becomes easier/harder?

### References
Code, issues, external sources (optional).
```