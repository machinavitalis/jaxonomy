# Jaxonomy — Project Context

Read `AGENTS/README.md` first for navigation. This file is the orientation
document — what Jaxonomy is, its architecture, and its invariants. Read it
before writing code, then PATTERNS.md and DECISIONS.md.

## What Jaxonomy is

Jaxonomy is a Python library for modeling and simulating hybrid dynamical systems
via block-diagram composition. It is built natively on JAX, combining
compositional block-diagram modeling with autodiff, GPU/TPU acceleration, and a
Python-native workflow. The target users are control engineers, roboticists, and
ML researchers who need more than raw ODE solvers but do not want the complexity
or license cost of MATLAB/Simulink or Modelica tooling.

Jaxonomy originated as the open-source simulation engine extracted from
Collimator, Inc.'s commercial product. The rebranding from `collimator` to
`jaxonomy` is complete; no internal references to the old name remain in the
source package.

Jaxonomy is distributed under the MIT license.

## Design philosophy: optimization-first

Jaxonomy is designed around a central thesis: **modeling and simulation is most
useful when end-to-end differentiable and optimization-ready.** A wide range of
engineering problems — parameter calibration against real data, early-stage
design iteration, trajectory optimization, surrogate modeling, controller
tuning, digital-twin closed-loop updates — can be formulated as mathematical
optimizations over simulated dynamics. The library is architected so that
every simulation output, objective, and constraint is differentiable with
respect to model parameters, initial conditions, and neural network weights by
default.

This is a different emphasis than traditional simulation tools, which treat
optimization and machine learning as secondary additions bolted onto a
simulation core. In Jaxonomy, the simulation core itself is a differentiable
function, and optimization is the primary use case it exists to serve.

Concretely, this means:
- Most native blocks are fully differentiable.
- Integration with external libraries preserves differentiability where
  possible.
- Reverse-mode autodiff works over entire simulation trajectories, including
  event handling and state machine transitions.
- The same autodiff infrastructure powers parameter estimation, trajectory
  optimization, state estimation (EKF/RLS), and predictive control.

When in doubt about a design choice, prefer the option that preserves or
improves end-to-end differentiability.

## What Jaxonomy is NOT

- It is not a cloud-hosted simulation platform. There is no web UI, no
  collaborative editing, no project server, no ensemble-on-cloud-HPC. Those
  capabilities belonged to Collimator's commercial product and are outside the
  scope of this library.
- It is not a robotics-specific tool. Robotics-specific abstractions (rigid-body
  kinematics, URDF import, actuator libraries, WBC primitives, etc.) belong in
  a separate layer (Jaxterity) that will be built on top of Jaxonomy. Keep
  Jaxonomy general-purpose.
- It is not a pure ODE solver. If a user only needs to integrate an ODE, they
  should use Diffrax directly. Jaxonomy adds block-diagram composition, hybrid
  (discrete + continuous) dynamics, event handling, state machines, and acausal
  modeling on top of that foundation.
- It is not a PINN / PDE-surrogate library. Classical physics-informed neural
  networks (spatial collocation over PDE residuals — Burgers, Navier–Stokes,
  heat-equation fields) are explicitly out of scope; point users at DeepXDE /
  NVIDIA PhysicsNeMo / Neuromancer. What Jaxonomy *does* own is
  physics-informed **dynamics** learning through the differentiable simulator:
  UDE, Neural DAE, Neural ODE blocks, SINDy. The full argument lives in
  `docs/scope/pinn.md` (T-045).
- It is not a deployment/codegen tool. Embedded codegen is out of
  scope for Jaxonomy itself. That said, two codegen paths are architecturally
  supported and documented as integration points: (1) MISRA-compliant C code
  generation for models built from low-level discrete logic blocks, and (2) JAX
  code compiled via XLA and called from C via the TensorFlow C API. These are
  deployment-layer concerns — the Jaxonomy core does not own the codegen
  pipeline, but it preserves the properties needed for downstream tools to
  consume it.

## Scope — what Jaxonomy owns

Jaxonomy provides a general-purpose hybrid dynamical simulation platform. Its
scope includes:

- **Hybrid dynamics simulation engine**
  - Discrete-time simulation for controllers and sampled systems
  - Continuous-time simulation for physical plants
  - Event handling with zero-crossing detection, guard intervals, hysteresis
  - State machines (programmatic construction via `StateMachineBuilder` DSL)
  - Multiple configurable ODE solvers: RK4 (fixed-step), Dopri5
    (variable-step Dormand-Prince RK45), BDF (stiff/variable-step,
    mass-matrix-capable)
  - Mass-matrix ODE form `M(x)·ẋ = f(t,x,p)` for semi-explicit DAEs;
    BDF supports mass matrices natively
  - Support for semi-explicit differential-algebraic equations (DAEs), not
    only ODEs
  - Data types (float, int, bool) with type inference
  - Vector, matrix, and tensor support
  - Integer-time representation (picosecond resolution) for deterministic event
    ordering, avoiding floating-point drift
- **Block-diagram composition**
  - 150+ foundational built-in blocks across: math/logic/signal
    routing, integrators, PID, filters, sources, sinks, neural-network
    blocks (Equinox MLP), system-identification blocks (SINDy),
    state-space (TransferFunction, StateSpace, LinearizedSystem),
    optimal control (LinearQuadraticRegulator, DiscreteTimeLQR,
    LinearDiscreteTimeMPC, LinearDiscreteTimeMPC_OSQP), state machines,
    hardware/vendor wrappers (Quanser HAL, PyTwin/Ansys, ROS2 via
    rclpy), and FMU co-simulation import
  - Hierarchical submodels for component reuse via nested `Diagram`s
  - Custom Python and Custom JAX blocks (`CustomPythonBlock`,
    `CustomJaxBlock`) for user-defined algorithms, with persistent
    environment isolation per instance via the cache mechanism
- **Acausal (equation-based) modeling**
  - Electrical, mechanical, thermal, and fluid/hydraulic domains
  - Pantelides index reduction with symbolic differentiation (SymPy)
  - Modelica-style flow-direction-dependent enthalpy mixing
    (`h_outflow` / `h_inStream`) for fluid components
  - Component library includes ClosedVolume and SimplePipe; fluid media
    support enthalpy and internal-energy modeling
  - Public API: `jaxonomy.acausal`
- **Differentiable simulation**
  - Reverse-mode autodiff (backpropagation) over entire simulation
    trajectories, including event handling
  - Gradients flow through ODE/DAE solvers (via Diffrax and custom adjoint
    methods)
  - Custom VJP rules (`simulation/autodiff_rules.py`) for `advance_to`
    and `guarded_integrate`, including the Cao et al. (2003)
    mass-matrix adjoint initial-condition correction for DAE backprop
  - Reverse-mode is the wired public interface; `jacfwd` is used
    internally for the mass-matrix adjoint correction but not exposed
    as a user-facing forward-mode path
  - `simulate_batch` for vmap-parallelized ensemble simulations
- **Parameter identification and optimization**
  - `fit_parameters` workflow with a `Result` object and multiple optimizer
    backends
  - Integration with 50+ local and global optimization algorithms (via optax,
    evosax, scipy, IPOPT, and custom wrappers)
  - Built-in multi-start, sensitivity / identifiability analysis, confidence
    intervals / parameter uncertainty, objective-function helpers
  - Online / real-time estimation (EKF, RLS)
  - `with_parameters()` for clean parameter binding, supporting both static
    and dynamic parameter distinctions
  - `ParameterExpr` for arithmetic expressions over `Parameter` objects
    (supports `+`, `-`, `*`, etc. — not arbitrary symbolic expressions
    like `np.sin(A)`)
- **Linearization**
  - `LinearizedSystem` for frequency-domain analysis, LTI extraction, and
    linear control design around operating points
- **JAX-first architecture**
  - Full simulation loop JIT-compiles
  - Autodiff flows through ODE solvers (via Diffrax)
  - Vmap over parallel simulations (`simulate_batch`)
  - Pluggable backends via the `MathDispatcher` mechanism
    (`backend/backend.py`) with `set_backend()` switching between
    registered backends. Currently registered: JAX (primary), NumPy
    (fallback), PyTorch (limited). Imported throughout the codebase as
    `numpy_api as npa` (see PATTERNS.md).
  - ML block wrappers for MLP (native), PyTorch, TensorFlow, Equinox, SINDy
- **Reduced-order modeling & surrogates** (`jaxonomy.library.rom`, DEC-033)
  - Linear MOR (balanced truncation / `minreal` / modal / residualization),
    POD–Galerkin with DEIM hyper-reduction, data-driven operator ROM
    (DMD / DMDc / ERA, Koopman / eDMD), and statistical surrogates
    (Gaussian process, polynomial chaos, RBF). One `reduce(...)` front door;
    every reduced model is a differentiable, simulatable block. Scope
    boundary vs PDE-field surrogates in `docs/scope/rom.md`.
- **Interop**
  - FMU import for FMI 2.0 / 3.0 co-simulation, including mixed-type
    and array/vector I/O; FMU export via `build_fmu` (pythonfmu-based).
    No model-exchange import; no FMI 3 scheduledExecution. See
    DECISIONS.md DEC-031, DEC-032.
  - JSON model serialization (legacy Collimator format, maintained for
    round-trip fidelity)
  - Jupyter notebook support as a first-class workflow

## Scope — what belongs elsewhere

- Robotics-specific: Jaxterity (planned)
- Embedded codegen and deployment: separate downstream library (planned)
- GUI / block-diagram editor: not in scope for the foreseeable future
- Cloud hosting, multi-user collaboration, HPC orchestration: not in scope

## Architectural heritage: Drake and JAX

Jaxonomy's core abstractions are directly inspired by Drake (Russ Tedrake and
the Toyota Research Institute), a C++/Python robotics modeling framework with
an optimization-oriented philosophy. From Drake, Jaxonomy borrows:

- The `System` / `LeafSystem` / `Diagram` hierarchy
- The `Context` concept — a tree-structured container for time, state,
  parameters, and inputs that mirrors the tree structure of the `Diagram`
- The principle of rigid semantics: a close correspondence between simulated
  behavior and the underlying mathematical representation of a hybrid
  dynamical system
- The separation of structure (System) from values (Context)

What Jaxonomy does differently: Drake is implemented in C++ with Python
bindings, which limits extensibility and complicates integration with
Python-native ML libraries. Jaxonomy reimplements the same conceptual structure
in pure Python on top of JAX. The System + Context pattern maps cleanly onto
the JAX functional paradigm: a `System` defines a collection of pure functions
(ODE right-hand sides, discrete-time update rules, output functions), and the
`Context` is the collection of arguments to those functions. Constructing a
model as a `Diagram` is a form of metaprogramming that builds up complex
functions from simple building blocks, which JAX can then trace, JIT-compile,
autodifferentiate, and vectorize.

Naming conventions (`LeafSystem`, `Diagram`, `Context`, etc.) are kept aligned
with Drake so that engineers familiar with Drake can onboard quickly. The
implementation underneath is quite different.

This JAX+Drake hybrid is the core technical bet of the library. It is what
makes Jaxonomy's capability set — end-to-end differentiable block-diagram
simulation with hybrid dynamics and acausal modeling — difficult to replicate
with any single off-the-shelf tool.

## Architecture overview

Top-level package layout (under `jaxonomy/`):

- `framework/` — Core abstractions. Defines `LeafSystem`, `Diagram`,
  `DiagramBuilder`, ports, parameters, contexts. This is the foundation on
  which all block types are built.
- `library/` — Standard block library. Math blocks, logic blocks, signal
  routing, integrators, PID, state machines, ML blocks (MLP, PyTorch wrapper,
  SINDy, TensorFlow wrapper, Equinox support), FMU import, MuJoCo wrapper,
  and so on.
- `simulation/` — The simulation engine. Solvers live in `backend/_jax/`
  (RK4, Dopri5, BDF). This package contains the simulator loop
  (`simulator.py`), event scheduler with zero-crossing detection
  (`zero_crossing_handler.py`), `simulate_batch` (`batch.py` — has both
  a per-iteration recompile loop path for static parameters and a
  vmap-kernel path for dynamic parameters), `SimulationResults` type
  (`types.py`), and custom autodiff rules (`autodiff_rules.py`).
- `acausal/` — Acausal/DAE modeling. Pantelides index reduction, symbolic
  manipulation via SymPy, domain-specific component libraries (electrical,
  mechanical, thermal, hydraulic). Public API: `jaxonomy.acausal`.
- `optimization/` — `fit_parameters`, multi-start, sensitivity, confidence
  intervals, online estimation (EKF/RLS), objective helpers. Integrates with
  optax, evosax, scipy, IPOPT.
- `backend/` — Backend dispatch abstraction (JAX / NumPy / partial PyTorch).
  ODE solver implementations live here under `_jax/`.
- `dashboard/` — JSON model serialization (`dashboard/serialization/`),
  schema definitions, and model IO. Also contains a thin API client layer
  left from the Collimator product era; the serialization subpackage is the
  actively maintained part.
- `mcp/` — FastMCP server exposing Jaxonomy tools to IDE and agent
  integrations.
- `cli/` — Command-line entry points.
- `models/` — Shared data-model types used across packages.
- `testing/` — Test utilities, fixtures, tolerance helpers.
- `utils/` — Internal utilities shared across packages.

## Key abstractions and invariants

Understanding these is required before modifying simulation or framework code.

- **System** — Abstract base. Defines the shape of time, state, parameters,
  inputs, and outputs, plus the pure functions (continuous dynamics, discrete
  update, output) that operate on a `Context`.
- **LeafSystem** — The atomic unit of behavior, subclass of `System`. Any
  custom block subclasses `LeafSystem` and declares input ports, output ports,
  state, parameters, and update functions.
- **Diagram** — A composition of `LeafSystem`s and sub-`Diagram`s connected
  via ports. Diagrams are themselves systems and can be nested to arbitrary
  depth.
- **DiagramBuilder** — Mutable builder for constructing a `Diagram`; produces
  an immutable `Diagram` via `.build()`.
- **Context** — Tree-structured container holding time, continuous state,
  discrete state, parameters, and inputs for a system at a point in
  simulation. The Context tree mirrors the System tree. Contexts are PyTrees
  and vmap-safe.
- **Parameters** — Bound via `with_parameters()`. The static vs. dynamic
  distinction matters for JIT: static parameters trigger recompilation when
  changed, dynamic parameters don't.
- **ParameterCache** — Shared cache for derived parameter values. Must be
  thread-safe and vmap-safe.
- **IntegerTime** — All event timing is expressed in integer picoseconds.
  Floating-point time is only used where mathematically required (inside
  solver steps). This eliminates a class of event-ordering bugs caused by
  floating-point drift in long simulations.
- **StateMachineBuilder** — DSL for constructing finite state machines with
  guards, resets, and transitions that integrate cleanly with the rest of the
  simulation graph.
- **LinearizedSystem** — Extracts an LTI approximation around an operating
  point for frequency-domain analysis and linear controller design.
- **MathDispatcher** — Backend dispatcher (`backend/backend.py`).
  Holds a context-variable-scoped active backend (JaxBackend,
  NumpyBackend, TorchBackend) and dispatches numerical ops. Switched
  via `set_backend()`.
- **ErrorCollector** — Accumulates multiple errors during context
  creation, type checking, and block evaluation rather than failing on
  the first error. Used in `framework/error.py` and consumers in
  `library/custom.py`, `library/linear_system.py`. Surfaces structured
  diagnostics to users instead of bare exceptions.
- **ParameterExpr** — Symbolic arithmetic over `Parameter` objects.
  Supports the arithmetic operators but not arbitrary NumPy/SymPy
  expressions over parameters.
- **Custom Python / JAX blocks** — `CustomPythonBlock` and
  `CustomJaxBlock` are stateless-feedthrough by design; they do not
  declare continuous state or ODE callbacks. They persist data across
  steps via the cache mechanism (`CacheType.persistent_env`), with
  per-instance environment isolation so state does not leak across
  block instances.

### Invariants that must hold

1. **No global mutable state.** All state lives in `Context` or is explicitly
   threaded through function arguments. Global state breaks `vmap` and
   reproducibility.
2. **Pure functions in the simulation path.** Any function on the simulation
   hot path must be a pure JAX-traceable function. Side effects (printing,
   file I/O, mutation) belong outside the simulated step.
3. **Vmap safety.** Any API exposed to `simulate_batch` or downstream vmapping
   must produce identical results when vmapped vs. run in a loop.
4. **PyTree compliance.** All user-facing stateful objects (Systems, Diagrams,
   Contexts, parameter containers) must register as JAX PyTrees so they flow
   correctly through JIT, vmap, grad, and serialization.
5. **Determinism.** Given the same seed, inputs, and tolerance settings,
   simulation outputs are bit-exact reproducible across runs on the same
   hardware. Cross-hardware determinism (CPU vs. GPU vs. TPU) is a goal but
   not guaranteed, and the deviations should be documented.
6. **End-to-end differentiability.** New blocks, solvers, and event-handling
   logic should preserve reverse-mode autodiff through the simulation
   trajectory. If a feature cannot be made differentiable, document why and
   provide a non-differentiable opt-out path rather than silently breaking
   gradients.
7. **Backend neutrality in user APIs.** User code that doesn't need to be
   JAX-specific should work on both JAX and NumPy backends.
8. **No global Python environment leakage from custom blocks.** Every
   `CustomPythonBlock` / `CustomJaxBlock` instance must have its own
   isolated persistent environment. Module-level state from one
   instance must not affect another. This is enforced via the
   per-instance cache.

## Supported versions and dependencies

- Python: 3.10+
- JAX: current stable
- NumPy: 1.26+
- Diffrax for continuous-time integration with autodiff
- SymPy for acausal symbolic manipulation
- Optional: `MuJoCo` (for robotics examples, to move to Jaxterity),
  `FMPy` (FMU import), `optax`, `evosax`, `scipy`, `IPOPT`, `PyTorch`,
  `TensorFlow`, `Equinox` (for ML block wrappers)

Optional dependencies are grouped under extras defined in `pyproject.toml`:
`[safe]` (SciPy / pandas / control / pysindy / pytwin / sympy stack
without NMPC), `[nmpc]` (cyipopt + osqp; needs IPOPT installed),
`[all]` (the union; heavy install surface), `[recommended]`,
`[units]` (pint-backed string parser), `[uq-qmc]` (scipy for QMC
sampling), `[streaming-export]` (h5py + zarr for `LazyResults`
streaming export), `[mcp]` (the FastMCP server bridge), `[test]`,
`[test-full]`. No `[ml]` / `[acausal]` extras — ML wrappers and the
acausal layer ship in the core install.

## Repository layout at root

- `jaxonomy/` — the package itself
- `test/` — pytest test suite (unit, integration, parametrized across backends)
- `docs/` — MkDocs source (published via `mkdocs.yml`)
- `examples/` — standalone example scripts
- `benchmarks/` — benchmark scripts + committed result files; shippable
  surface (results are evidence)
- `validation/` — validation tests against reference results
- `notes/` — transient planning artifacts; `plan-<slug>.md` per RULES.md
  "Plan mode is the default"
- `AGENTS/` — agent-oriented orientation and operating manual
- `CHANGELOG.md` — user-visible delta against this file; shippable surface
- `CLAIMS.md`, `KNOWN_GAPS.md` — claim ledger and its public inverse
- `pyproject.toml`, `requirements.*`, `LICENSE.md`, `README.md` — standard
  Python project files

## Current state

- Baseline test failures (unrelated to your change, not worth chasing
  unless the task is to fix them) are listed in `/CLAUDE.md` under
  "Testing" — the canonical list. Consult it before debugging a red test.
- The library's capabilities are enumerated under "Scope — what
  Jaxonomy owns" above; the published gap list is `/KNOWN_GAPS.md`.
- Not yet shipped: hierarchical state machines, `ForLoop` / `WhileLoop`
  container blocks, fully implicit DAE solver (DEC-027 rejects the last).
- Current priorities emphasize correctness guarantees (determinism,
  precision, gradient correctness, conservation laws) and performance
  characterization over capability expansion.

## Positioning summary

Jaxonomy's one-line description: *JAX-native Simulink/Modelica alternative —
hybrid dynamics, acausal modeling, end-to-end differentiable, Python-first.*

What makes it unique relative to existing tools: no other open-source library
combines compositional block-diagram modeling, hybrid discrete/continuous
dynamics, acausal DAE support via index reduction, and JAX autodiff/GPU
acceleration in one coherent package. Simulink has the composition but is
commercial and MATLAB-locked. Modelica/OpenModelica have acausal modeling but
no JAX/ML story. Diffrax has solvers but no block composition. Python-control
is LTI-only. Drake has the conceptual foundation but is C++-with-bindings and
robotics-opinionated. `bdsim` has the block abstraction but lacks advanced
blocks and differentiability.

## When modifying Jaxonomy

- Check DECISIONS.md before making architectural choices. Many questions have
  already been answered there.
- Follow PATTERNS.md for coding conventions.
- Respect the invariants above. The simulation engine's correctness depends on
  them.
- Prefer pure JAX-native solutions over backend-switching logic where
  possible.
- Preserve end-to-end differentiability in new code; if it can't be preserved,
  document why and provide a non-differentiable opt-out.
- When in doubt about scope ("does this belong in Jaxonomy or Jaxterity?"),
  ask: "is this useful outside of robotics?" If yes, it belongs in Jaxonomy.
  If no, it belongs in Jaxterity.