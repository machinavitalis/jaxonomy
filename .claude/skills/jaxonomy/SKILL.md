# Jaxonomy Skill

You are using Jaxonomy, a JAX-native engine for modeling and simulating hybrid
dynamical systems by block-diagram composition. This file is your operating
manual for *using* the library. Read it before suggesting any Jaxonomy code.
(If you are *modifying* Jaxonomy itself, read `AGENTS.md` and the `AGENTS/`
docs instead.)

## What Jaxonomy does

Jaxonomy composes continuous physics, discrete control, and event-driven logic
into a single model, and runs it on JAX — JIT-compilable, `vmap`-batchable, and
**differentiable end to end**. Its thesis: modeling and simulation is most
useful when it is optimization-ready, so every output, objective, and
constraint is differentiable w.r.t. parameters, initial conditions, and network
weights by default.

It is the engine at the base of a larger stack (robotics and embedded
deployment layer on top of it). For the full architecture, design philosophy,
and invariants, read `AGENTS/CONTEXT.md` — don't restate it here.

## When to use Jaxonomy

Use it when the user is:

- Building a block-diagram model of a dynamical system (continuous, discrete,
  or hybrid) and simulating it.
- Closing a control loop — LQR, MPC, PID, or a Kalman/EKF/UKF estimator around
  a plant.
- Differentiating *through* a simulation: parameter calibration against data,
  trajectory optimization, controller tuning, neural ODE / SINDy, digital-twin
  updates.
- Acausal / multi-physics modeling (electrical, thermal, fluid, mechanical) via
  `jaxonomy.acausal`.
- Uncertainty quantification (Monte Carlo, Sobol, LHS, qMC) via `jaxonomy.uq`.
- Batch/ensemble simulation with `vmap`, or GPU/TPU-accelerated runs.

## When NOT to use Jaxonomy

- **Just integrating an ODE.** Use Diffrax directly. Jaxonomy adds
  block-diagram composition, hybrid dynamics, events, and state machines on top
  of a solver — if none of that is needed, it's overhead.
- **Robotics with joints/actuators/contacts/kinematic chains.** Use Jaxterity,
  the robotics layer built on top of Jaxonomy. Don't re-implement URDF import,
  articulated dynamics, or WBC here.
- **Embedded codegen / cross-compilation to silicon.** That's a downstream
  deployment concern, out of scope for Jaxonomy.
- **A hosted/cloud simulation platform, web UI, or collaborative editor.** Not
  what this library is; see `AGENTS/CONTEXT.md` ("What Jaxonomy is NOT").

## Core API surface, in order of how often agents will use it

### Build a diagram, create a context, simulate

```python
import jaxonomy as jx

builder = jx.DiagramBuilder()
plant      = builder.add(jx.library.LTISystem(A, B, C, D))
controller = builder.add(jx.library.LinearQuadraticRegulator(A, B, Q, R))
builder.connect(plant.output_ports[0], controller.input_ports[0])
builder.connect(controller.output_ports[0], plant.input_ports[0])

diagram = builder.build()
context = diagram.create_context()          # holds state + parameters
results = jx.simulate(diagram, context, stop_time=5.0)
```

- `DiagramBuilder.add(block)` returns a handle whose `.input_ports[i]` /
  `.output_ports[i]` you wire with `builder.connect(src_out, dst_in)`.
- The `Context` is an immutable carrier of state and parameters — thread it
  through; don't mutate it in place.
- `jx.simulate(...)` is the entry point; the whole call is differentiable and
  `jit`/`vmap`-friendly.

### Library blocks

Standard blocks live under `jx.library`, split by category (sources, math_ops,
logic, routing, dynamics, nonlinearities, tables); `jx.library.primitives`
re-exports them for back-compat. Controls/estimation blocks (LQR, MPC, PID,
Kalman/EKF/UKF) are library blocks too — prefer them over hand-rolling.

### Analysis and optimization

- `linearize(...)` → a `LinearizedSystem`; analytical helpers (`bode_data`,
  `nyquist_data`, `step_response`, `frequency_response`, …) return plain dicts
  of arrays (no matplotlib inside Jaxonomy — you plot).
- Because `simulate` is differentiable, parameter estimation, trajectory
  optimization, and controller tuning are just gradient-based optimizations
  over the simulation — use JAX autodiff / the provided tuning helpers.

For exact signatures, see the docs at py.jaxonomy.com and the `examples/`
notebooks — prefer those over guessing an API.

## Key gotchas

- **Backend-neutrality: `npa` vs `jnp` vs `np`.** Inside models, numeric code
  goes through the `npa` abstraction, not `jnp` directly — it's a load-bearing
  invariant. If you write library-style code, follow the pattern in
  `AGENTS/PATTERNS.md`; as a *user* composing existing blocks you rarely touch
  it, but don't assume raw `jnp` everywhere.
- **State is NamedTuple-shaped, not mutable objects.** Read state out of the
  results / context; don't try to assign into it.
- **Differentiability is the default and the point.** If a construct would
  break gradients (Python-side branching on traced values, in-place mutation),
  reach for the block/pattern that preserves them.

## Common pitfalls & idioms

Hard-won usage tips (harvested from prior consumer sessions):

- **Parameter sweeps re-JIT per iteration.** `diagram.with_parameter("p", float(v))`
  in a Python loop keys the trace cache on the *value*, recompiling every step.
  Wrap the scalar as `jnp.asarray(v)`, or use `simulate_batch` / `jax.vmap` from
  the start. `scipy.optimize.minimize_scalar` inherits the same blowup.
- **Update sub-system parameters by dot-path on the *outer* diagram:**
  `outer.with_parameters({"inner.gain.gain": jnp.array(5.0)})` — don't reach into
  `outer["inner"]` (you'll get a stale outer diagram). The dot-path form also
  composes under `vmap`.
- **`declare_periodic_update` needs an explicit `offset=0.0`.** Omitting it
  constructs fine but crashes at the first step deep in the scheduler.
- **Traced-mode guards:** gate host-side checks with
  `isinstance(x, jax.core.Tracer)`, *not* `jax.core.is_concrete` — the latter is
  `True` under `jax.grad`'s tracer and silently breaks gradients.
- **`LookupTable1d.interp_1d` takes `method=`**, not `mode=` (the rest of the
  library uses `mode=` on `Quantizer`/`Saturate`, so the typo is natural).
- **Symmetric saturation shorthand:** `Saturate(limit=L)` instead of
  `upper_limit=+L, lower_limit=-L`.
- **`simulate` specifics:** `context` is a required positional arg (no
  context-less shortcut) and `options` must be a `SimulatorOptions`, not a dict;
  read results via `res.time` and `res.outputs[name]`.
- **Long / stiff / multi-rate runs:** bump `SimulatorOptions(buffer_length=...)`
  (the recorder silently truncates to the *tail* when its ring buffer fills) and
  use `ode_solver_method="bdf"` for stiff fast+slow coupling (the explicit
  default collapses its step size).
- **Stateful feedback controller as a LeafSystem:** declare integrator state
  with `declare_continuous_state(..., ode=cb, requires_inputs=True)` and the
  command with `declare_output_port(..., requires_inputs=True,
  prerequisites_of_calc=[DependencyTicket.xc])`. Wiring controller↔plant is *not*
  an algebraic loop if the plant exposes state via
  `declare_continuous_state_output` (a state output, not feedthrough).

## When to escalate to the human

- A model that won't `jit` or `vmap` cleanly and the cause isn't an obvious
  Python-control-flow-over-traced-values mistake.
- Numerical divergence or stiffness that looks like a solver/tolerance choice
  rather than a modeling bug.
- Any request that implies robotics-specific structure — redirect to Jaxterity
  rather than bolting kinematics into a Jaxonomy model.

## Where to find more

- `AGENTS/CONTEXT.md` — architecture, design philosophy, what it is and is NOT.
- `AGENTS/PATTERNS.md` — coding conventions (`npa`, `LeafSystem`, state).
- `CLAIMS.md` / `KNOWN_GAPS.md` — what Jaxonomy actually does vs. what it does
  not (read both before relying on a surface).
- `README.md` + `examples/` — quick start and end-to-end notebooks.
- Docs: py.jaxonomy.com.
