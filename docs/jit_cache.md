# Persistent JIT compilation cache

Jaxonomy compiles each diagram + solver combination into XLA on first use.
Single-shot ODE simulations compile in tens of milliseconds; `simulate_batch`
ensembles take >1 s. JAX provides a persistent on-disk cache that recovers
most of this cost across processes — Jaxonomy ships a one-call helper.

```python
import jaxonomy
jaxonomy.enable_persistent_jit_cache()              # ~/.cache/jaxonomy/jit/
# or
jaxonomy.enable_persistent_jit_cache("/scratch/jit")  # custom dir
```

The first run after enabling pays the normal compile cost and writes the
artefact to disk. Subsequent processes (with matching JAX version, JAXPR,
and target device) read from disk in tens of milliseconds rather than
re-running XLA lowering.

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

Always for interactive work and long-running CI jobs; skip for short-lived
single-shot scripts where the cache write itself dominates. See
`benchmarks/compile_time.py` (T-017) for per-case timings.
