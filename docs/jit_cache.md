# Persistent JIT compilation cache

Jaxonomy compiles each diagram + solver combination into XLA on first use.
Small single-shot ODE simulations compile in well under a second (roughly
60–250 ms, depending on whether the model has zero-crossings or an acausal
DAE); large diagrams and `simulate_batch` ensembles take longer. JAX
provides a persistent on-disk cache for the XLA-compile share of that
cost — Jaxonomy ships a one-call helper.

```python
import jaxonomy
jaxonomy.enable_persistent_jit_cache()              # ~/.cache/jaxonomy/jit/
# or
jaxonomy.enable_persistent_jit_cache("/scratch/jit")  # custom dir
```

The first run after enabling pays the normal compile cost and writes the
artefact to disk. Subsequent processes (with matching JAX version, JAXPR,
and target device) read the compiled executable from disk instead of
re-running XLA compilation.

## What it does and does not buy you

Two structural limits bound the win, both measured (2026-07, jax 0.9.2,
arm64 CPU):

- **Small models are below the write threshold.** The helper sets
  `jax_persistent_cache_min_compile_time_secs = 1.0`, so kernels that
  compile faster than 1 s are never written. A 4-block PID loop and the
  bouncing-ball model with `record_event_times=True` (first `simulate`
  ≈ 0.21–0.25 s) produced **zero cache entries** — warm-cache startup was
  identical to cold. For these models the cache is a harmless no-op; the
  compile is cheap anyway.
- **Python tracing is not cacheable.** The cache stores compiled XLA
  executables, not traces. On a 160-block diagram (first `simulate`
  ≈ 2.4 s cold), a warm cache cut the first call to ≈ 1.2 s — a genuine
  ~2× — but the remaining ~1.1 s is JAX tracing/lowering, which every
  process pays regardless.

A side benefit on larger models: repeated *bare* `simulate` calls in the
same process re-trace each time (see below), and with the cache enabled
each re-trace's recompile hits the disk cache too (measured ≈ 2.2 s →
≈ 1.1 s per repeat call on the 160-block diagram).

**The cache does not fix Python-loop parameter sweeps.** In
`for v in grid: simulate(..., ctx.with_parameter("p", float(v)))` each
float is baked into the HLO as a constant, so every value is a compulsory
cache miss (measured: a 3-value sweep against a warm cache added 3 fresh
entries and every iteration paid full re-trace + compile). Use
`simulate_batch` or traced parameters instead — see the sweep entry in
`KNOWN_GAPS.md`.

## What is cached

The cache keys on the JAXPR, JAX version, and target device. Bumping JAX,
switching CPU↔GPU, or changing a static parameter that flips a control-flow
branch produces a fresh entry. Stale entries persist; periodically delete
the cache directory on big version bumps to reclaim disk space.

## Tunables

`enable_persistent_jit_cache` configures three JAX options:

| option | value | rationale |
| --- | --- | --- |
| `jax_compilation_cache_dir` | the cache directory | required |
| `jax_persistent_cache_min_compile_time_secs` | `1.0` | trivial computations recompile faster than disk read |
| `jax_persistent_cache_min_entry_size_bytes` | `-1` | size threshold disabled — gating on time alone is enough |

For different thresholds, call `jax.config.update(...)` directly afterwards.

## When to enable

Enable it when your model's compile time is noticeable — large diagrams
(≳100 blocks), big acausal packs, `simulate_batch` ensembles, or jitted
gradient loops (next section), where it roughly halves cold-start and
recompile cost. For small models it is a harmless no-op (compiles under
1 s are never written). See `benchmarks/compile_time.py` for per-case
compile timings.

## Repeated gradients: hoist the `jit`

Every bare call to `jaxonomy.simulate` builds a *fresh* traced closure, so
JAX's in-process jit cache misses on function identity and you pay a full
re-trace + XLA compile **per call** — the numeric solve itself is
milliseconds. This dominates design loops that differentiate through the
simulator repeatedly:

```python
# SLOW — each value_and_grad call re-traces + recompiles the whole
# forward + adjoint (seconds per call; ~30 s on a 24-state acausal pack):
def objective(theta):
    ctx = base_context.with_parameter("g0", theta)
    res = jaxonomy.simulate(model, ctx, (0.0, tf), options=opts)
    return res.context.continuous_state[1]

for step in range(5):
    J, dJ = jax.value_and_grad(objective)(theta)   # re-traces every time
    ...
```

The fix is to define the objective **once** as a pure function of
`(theta, context)` and wrap the *outer* `value_and_grad` in `jax.jit`, so
tracing happens exactly once:

```python
# FAST — one compile, then ~milliseconds per call:
@jax.jit
def value_and_grad_fn(theta, context):
    def objective(theta):
        ctx = context.with_parameter("g0", theta)
        res = jaxonomy.simulate(model, ctx, (0.0, tf), options=opts)
        return res.context.continuous_state[1]
    return jax.value_and_grad(objective)(theta)

for step in range(5):
    J, dJ = value_and_grad_fn(theta, base_context)  # cached after call 1
    ...
```

This works for the implicit **BDF/DAE** path too (with
`SimulatorOptions(enable_autodiff=True)`): the simulation context and BDF
solver state are ordinary pytrees and trace cleanly. Measured on the
index-2 pendulum DAE (9 states, BDF, 2 s horizon, CPU): unjitted
~1.8 s **per call**; jitted 1.8 s once, then **~10 ms per call** (~180×).
Cost envelope: per-call cost of the naive pattern is almost entirely
trace+compile and scales with model size (states, blocks, solver
machinery), not with the horizon; the compiled kernel's runtime scales
with horizon and stiffness. Combine with the persistent cache above to
also amortise the one-time compile across processes.

Requirements for the pattern: pass the context (and any other
non-differentiated inputs) as *arguments* of the jitted function rather
than closing over mutable state, keep `t_span` and options static, and
don't rebuild the diagram inside the traced function (acausal
compilation is not jit-safe — build once, outside).
