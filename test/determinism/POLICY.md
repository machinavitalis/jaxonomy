# Determinism and reproducibility policy (T-002)

This policy is the source of truth for Jaxonomy's simulation-output
reproducibility contract. The contract is enforced by `_framework.py`
(`assert_bitwise_reproducible`) and exercised by the test modules in this
directory. Any change here must be accompanied by updated tests.

## The contract

Given:

- the same `jaxonomy` version,
- the same hardware / device / backend,
- the same process (same Python interpreter, same JAX install, same XLA),
- identical inputs (context, parameters, PRNG seeds, tolerance settings,
  `SimulatorOptions`), and
- no external non-determinism (clocks, file system state, `np.random`
  globals, etc.),

`simulate(...)` returns outputs that are **bit-exact equal across runs**.

"Bit-exact" means `np.array_equal` returns True when applied to the raw
byte representation of the output arrays, not "close to within tolerance".
A single differing bit fails the contract.

## Scope — what the contract covers

The contract applies to every system and option combination that appears in
the reference test matrix:

- Pure continuous-time ODEs under all three selectable solvers (RK4,
  Dopri5, BDF).
- Pure discrete-time systems with periodic events.
- Hybrid CT + DT systems.
- Systems with zero-crossing events and mode transitions.
- Acausal DAE systems solved by BDF with a mass matrix.
- Systems containing stochastic blocks (`RandomNumber`, `WhiteNoise`)
  **when an explicit seed is provided**.
- Ensembles run through `simulate_batch` / `jax.vmap`.
- `StateMachineBuilder`-constructed blocks.

## Scope — what the contract does NOT cover

### Cross-device reproducibility

Running the same simulation on CPU vs. GPU vs. TPU will **generally not**
produce bit-exact outputs. The reason is that XLA fuses and reorders
floating-point operations differently for each backend, and GPU/TPU
reductions may run in parallel with non-deterministic summation order.

- CPU results are reproducible **within a single hardware architecture**
  (x86_64 linux vs. x86_64 linux). Cross-architecture CPU (x86_64 vs.
  aarch64) may diverge at ULP scale due to different vectorised math
  library implementations.
- GPU results are reproducible only when the following environment is set:
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. Even then, results may
  differ from CPU output at ULP scale.
- TPU results are reproducible within a single TPU generation and pod
  topology; across generations they may diverge at ULP scale.

Cross-device tests in this directory (`test_cross_hardware.py`) are
therefore skip-guarded and document the expected divergence envelope
rather than asserting bit-exactness.

### Unseeded stochastic blocks

`RandomNumber(seed=None)` and `WhiteNoise(seed=None)` are **not**
reproducible — they consume system entropy at context creation. The test
suite uses explicit seeds.

### External I/O and wall-clock

`DataSource` blocks that read from disk, `Clock`-driven logic that reads
`time.time()`, and any user-written block that consults external state
fall outside the contract.

### Non-deterministic user code

If a user's `CustomPythonBlock` or ODE right-hand side contains
`np.random` without a seed, uses `time.time()`, or otherwise introduces
external state, the contract does not apply to that simulation.

## Enforcement

Per-commit CI:

- `test/determinism/test_same_run.py`: runs every reference simulation
  twice on the CI runner and asserts byte-exactness. Failure fails the
  quick job.
- `test/determinism/test_negative_controls.py`: asserts that changing a
  seed, tolerance, or input does change the output. Guards against a
  test that would trivially pass by always returning zero.

Nightly CI (extended):

- Same tests on the full solver / solver-option matrix.

Local development:

```bash
pytest -q test/determinism/
```

## Changing a test

If you need to disable a determinism test because the output has become
non-reproducible, **that is a red flag, not a fix**. The underlying
simulation has almost certainly acquired a dependency on external state
(iteration order of a set/dict, a pointer address, an unseeded PRNG, a
`time.time()` call inside a callback). Investigate the root cause;
adjusting the tolerance or skipping the test is the wrong move.

If a change to XLA or JAX legitimately shifts the bit pattern of
otherwise-equivalent outputs (version upgrade), update the expected
outputs in the test file in the same commit and record the upstream
version bump in the commit message.
