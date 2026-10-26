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

Time to build the model and finish the first `simulate` in a fresh
process, measured with `benchmarks/jit_cache.py` (2026-10, JAX 0.9.2,
arm64 CPU, median of 5 processes):

| model | no cache | writing the cache | later process |
| --- | --- | --- | --- |
| PID loop, 6 blocks | 239 ms | 238 ms | 119 ms |
| oscillator chain, 159 blocks | 713 ms | 730 ms | 402 ms |

- **A later process skips XLA compilation.** Every compiled program is
  stored, the small ones a model build produces included, so small models
  gain as well. The cache took 59 kB for the PID loop and 180 kB for the
  chain.
- **Parameter and initial values do not have to match.** The "later
  process" column was run with other parameter values and other initial
  conditions than the process that wrote the cache, and added no entries.
  Values that live in the context are arguments of the compiled program,
  not part of it.
- **Python tracing is not cacheable.** The cache stores compiled XLA
  executables, not traces. What is left in the later process is the model
  build plus JAX tracing and lowering, which every process pays.

**Within one process, repeated runs are fast without the cache.**
`simulate` keeps the compiled kernel of a system and reuses it when it is
called again with the same system object, options and time span — for the
same context, for another initial state, and for a parameter changed
through the context. Measured in the same run: 0.3 ms per call for the PID
loop and 3 ms for the chain, with no compilation, against 182 ms and
605 ms for the first call.

**A copy of the diagram reuses the kernel too.**
`diagram.with_parameters({...})` returns a new diagram, and `simulate`
runs it on the kernel compiled for the diagram it was copied from: nothing
is traced or compiled, and the disk cache gains no entry. Each value still
pays for the copy and its context: 2.6 ms per value for the PID loop and
44 ms for the chain, against 0.3 ms and 3 ms through the context. A copy
gets a kernel of its own when the program would not be the same: after a
parameter value that changes how a block is configured (a PID gain set to
zero), with a block that calls back into Python, or after a static
parameter has changed. For a long sweep in a Python loop, change the
value through the context, or use `simulate_batch`; see the sweep entry in
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
| `jax_persistent_cache_min_compile_time_secs` | `0.0` | store every program: loading one is faster than compiling it, also for programs that compile in a fraction of a second |
| `jax_persistent_cache_min_entry_size_bytes` | `-1` | size threshold disabled |

For different thresholds, call `jax.config.update(...)` directly afterwards.

## When to enable

Enable it when the same models are simulated again in later processes:
scripts that are re-run, test suites, restarted notebooks, batch jobs. It
roughly halves the time to the first result there (table above). It makes
no difference to repeated runs inside one process. See
`benchmarks/compile_time.py` for per-case compile timings.

## Repeated gradients: `jit` the outer function

`jax.grad` or `jax.value_and_grad` through `simulate` can be called bare or
wrapped in `jax.jit`. Both reuse their compiled code on later calls. The
jitted form compiles one program where the bare form compiles about ten
(the forward and adjoint pieces separately), and has less overhead per
call. Gradient of a scalar loss with respect to the initial state, same
run as above:

| model | bare: first call | bare: later calls | jitted: first call | jitted: later calls |
| --- | --- | --- | --- | --- |
| PID loop | 769 ms | 6.2 ms | 648 ms | 3.7 ms |
| oscillator chain | 5.7 s | 75 ms | 4.9 s | 11 ms |

Define the objective once as a pure function of its inputs and wrap the
*outer* `value_and_grad` in `jax.jit`:

```python
@jax.jit
def value_and_grad_fn(theta, context):
    def objective(theta):
        ctx = context.with_parameter("g0", theta)
        res = jaxonomy.simulate(model, ctx, (0.0, tf), options=opts)
        return res.context.continuous_state[1]
    return jax.value_and_grad(objective)(theta)

for step in range(5):
    J, dJ = value_and_grad_fn(theta, base_context)  # compiled on call 1
    ...
```

This works for the implicit **BDF/DAE** path too (with
`SimulatorOptions(enable_autodiff=True)`): the simulation context and BDF
solver state are ordinary pytrees and trace cleanly. Combine with the
persistent cache above to also amortise the one-time compile across
processes.

Requirements for the pattern: pass the context (and any other
non-differentiated inputs) as *arguments* of the jitted function rather
than closing over mutable state, keep `t_span` and options static, and
don't rebuild the diagram inside the traced function (acausal
compilation is not jit-safe — build once, outside).
