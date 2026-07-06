# benchmarks/

Standalone benchmark scripts. Not run by `pytest`; CI invokes them directly.

## `compile_time.py` — T-017 JIT compile-time profile

Measures cold JIT compile time and warm sim time for nine cases:

| case | description | solver |
| --- | --- | --- |
| `scalar_exponential_decay` | dy/dt = -a·y, scalar | dopri5 |
| `harmonic_oscillator` | 2-D ODE | dopri5 |
| `pid_first_order_plant` | PID + 1/s plant, LTI mix | dopri5 |
| `rc_acausal_dae` | acausal RC circuit | BDF (mass matrix) |
| `bouncing_ball_zc` | bouncing ball with restitution | dopri5 + ZC |
| `state_machine_three_state` | 3-state controller, dt=0.05 | dopri5 (discrete) |
| `simulate_batch_decay_n10` | 10-element ensemble | dopri5 (kernel path) |
| `simulate_batch_decay_n100` | 100-element ensemble | dopri5 (kernel path) |
| `simulate_batch_decay_n1000` | 1000-element ensemble | dopri5 (kernel path) |

```bash
python benchmarks/compile_time.py            # write baseline JSON
python benchmarks/compile_time.py --check    # compare vs baseline; exit 1 on regression
```

The script clears JAX's in-memory trace cache between iterations so each
recorded compile is genuinely cold; otherwise the second iteration would
report ~0 s thanks to JAX's automatic in-process cache.

## Baseline (macOS-arm64, JAX 0.9.2)

Median of 3 runs. Hardware fields stored alongside in
`compile_time_baseline.json`.

| case | compile (s) | sim (s) |
| --- | --- | --- |
| scalar_exponential_decay | 0.029 | 0.000 |
| harmonic_oscillator | 0.027 | 0.000 |
| pid_first_order_plant | 0.040 | 0.000 |
| rc_acausal_dae | 0.076 | 0.000 |
| bouncing_ball_zc | 0.042 | 0.000 |
| state_machine_three_state | 0.017 | 0.000 |
| **simulate_batch_decay_n10** | **0.044** | **0.103** |
| **simulate_batch_decay_n100** | **0.043** | **0.115** |
| **simulate_batch_decay_n1000** | **0.044** | **0.214** |

All single-shot cases compile well under 200 ms. After T-017c (see below),
`simulate_batch_*` compile cost is **flat across N** (was 1.28 s at N=10
pre-T-017a, 0.37 s after T-017a, now 0.049 s after T-017c). The kernel
JIT (~50 ms) is the only compile-time cost; resampling and stacking are
now fully host-side numpy with no per-shape recompiles. Sim wall time
still scales linearly with N as expected.

## T-017a — `simulate_batch` finalize hotspot (resolved)

T-017's profile pointed at "vmap-of-jit lowering," but instrumented
profiling found the hotspot was actually inside `JaxResultsData._trim`:
each loop iteration of the scan-kernel path called
`np.array(solution.time[valid_idx])` where `valid_idx` is a
`jnp.isfinite(...)` boolean mask.  The result of JAX boolean indexing has
a dynamically-determined shape, so XLA dispatch fired a fresh
host-blocking call (~80 ms × 10 = ~800 ms).  Fix: materialise the buffer
to numpy first, then apply the boolean mask host-side
(`backend/_jax/results_data.py::_trim`).  Output is bit-identical
(numpy arrays, same values) — `test_v006_*` and the bit-exact
`test_v002_determinism::test_batch_*` suites all pass unchanged.

## T-017c — `simulate_batch` per-shape interp recompile (resolved)

After T-017a addressed the host-side trim cost, profiling at N=100
located a second linear hotspot in `_scan_kernel_path`'s post-finalize
loop: `_interp_on_time` was calling `jnp.interp` on each element's
trimmed numpy outputs.  The adaptive Dopri5 solver yielded ~25 distinct
trim lengths across N=100 elements, and each new shape combination
spawned a fresh `jnp.interp` JIT lowering (~10 ms each) plus a
host→device copy.  Aggregate cost ~1 s wall.  Fix: added a numpy-host
variant `_interp_on_time_np` (uses `numpy.interp`, no compile cost) and
a same-grid fast path that skips resampling when `time_i == time_ref`
element-wise.  Output stacking switched from `jnp.stack` to `np.stack`
to avoid an unnecessary host→device copy.  Bit-equivalent (same linear
interpolation, both at float64).  All `test_v006_*` and bit-exact
`test_v002_determinism` batch suites pass unchanged.  N=10 went from
0.37 s → 0.049 s; N=100 from extrapolated 3.7 s → 0.049 s; N=1000 from
extrapolated 37 s → 0.048 s.

## Identified hotspots

1. **`simulate_batch` ensemble compile (0.049 s, flat across N)** —
   T-017c closed the residual gap.  The remaining ~50 ms is the kernel
   JIT itself; resampling and stacking are pure-numpy host work that
   does not recompile.  No further follow-up filed.

2. **`rc_acausal_dae` (~80 ms)** — second-tier. BDF's implicit-step
   nonlinear solve plus the acausal mass-matrix-aware ODE wrapper
   double the compile cost vs. the equivalent ODE. Filed as **T-017b**;
   the cheap-Python wins (param-key hoist, no-ZC fast path) shipped in
   the original ticket. **T-017b-followup** added a class-level
   ``IndexReduction._sed_cache`` that memoises the Pantelides + dummy-
   derivatives output keyed on the symbolic equations + parameter values:
   it does **not** move this benchmark (which times JAX JIT lowering only,
   not the AcausalCompiler stage), but cuts ~340 ms from repeated
   AcausalCompiler invocations on the same circuit (test re-runs, REPL
   workflows, dashboard reloads). The remaining ~80 ms is dominated by
   BDF Newton-iteration jaxpr tracing inside ``jax.lax.while_loop``;
   hoisting that body to a module-scope JIT is filed as
   **T-017b-followup-newton-blocker** because a naïve hoist breaks the
   custom-VJP autodiff path documented in ``simulation/autodiff_rules.py``.

3. **`pid_first_order_plant` (58 ms)** — third-tier. PID's two-state LTI
   wrapper compiles slightly slower than a bare scalar ODE (extra traced
   parameters: kp/ki/kd/n + the four state-space matrices). Logged for
   context; not separately filed.

## Threshold rationale

`REGRESSION_THRESHOLD = 1.5` (fail at +50 %) is permissive on purpose. CI
runners are noisy and sub-second measurements have high relative jitter.
The benchmark already takes the median of 3 cache-cleared runs, but XLA
lowering still has substantial variance from kernel scheduling and shared-CPU
contention. Tighter bounds (e.g. +20 %) trip on routine noise. The trend
dashboard tracked across nightly runs (T-019's domain) is the long-term
signal; this benchmark is the per-PR backstop.

## Honest gaps

None at the moment — all seven cases run successfully on the macOS-arm64
host used to seed the baseline. If a future case crashes, mark it
`# benchmark for X disabled — TODO: <reason>` rather than silently dropping.

## `memory.py` — T-018 memory footprint profile

Measures peak resident-set size (RSS via `resource.getrusage`) for
three workload families: per-element batch (N=1/10/100), long-horizon
(t_end=10/100/1000 s), and cold-compile peak across the 7 T-017 cases.

```bash
python benchmarks/memory.py            # write baseline JSON
python benchmarks/memory.py --check    # compare vs baseline; exit 1 on regression
```

Each measurement runs in a fresh subprocess so `ru_maxrss` (peak since
process start) is clean. `tracemalloc` is **not** used: JAX/XLA
allocate outside the Python heap, so it captures <5 % of the real
footprint. Module docstring has the full rationale.

### Baseline (macOS-arm64, JAX 0.9.2)

Median of 3 subprocess runs.

| case | peak RSS (MB) |
| --- | ---: |
| batch_n1 / n10 / n100 | 322.9 / 337.0 / 373.0 |
| long_horizon_t10 / 100 / 1000 | 316.9 / 316.3 / 317.7 |
| compile_scalar_exponential_decay | 253.8 |
| compile_harmonic_oscillator | 257.8 |
| compile_pid_first_order_plant | 384.9 |
| compile_rc_acausal_dae | 367.9 |
| compile_bouncing_ball_zc | 266.1 |
| compile_state_machine_three_state | 256.0 |
| compile_simulate_batch_decay_n10 | 336.8 |

Derived: **~0.46 MB/element** batch slope, **~0.001 MB/sim-second**
long-horizon growth (essentially flat — buffer dominates).

### Hotspots

PID-first-order-plant (385 MB) is the highest single-shot compile peak —
same PID-LTI-wrapper root cause as the third-tier compile-time
slowdown noted in T-017. RC-acausal-DAE (368 MB) is second, tracked by
existing **T-017b** (acausal/BDF compile cost). No case crosses
500 MB; **no T-018a hotspot filed**.

### Threshold rationale

Same `REGRESSION_THRESHOLD = 1.5` as `compile_time.py`. Peak-RSS on
shared CI runners has substantial jitter from co-tenant memory
pressure; +50 % is permissive enough that routine noise doesn't trip
the gate. Tighter bounds (e.g. +20 %) flap on cold-CI runs.

### Honest gaps

`ru_maxrss` is peak-since-process-start, not a delta. The subprocess
isolation means each datapoint carries the ~250 MB JAX-startup
baseline; subtract the column minimum to extract *additional* memory.
For finer-grained attribution, run under `mprof`/`psrecord`.

## `public.py` — T-019 public benchmark suite

Public-facing numbers on five standard control/sim problems, designed
to be honestly comparable to MuJoCo Playground / JaxSim where fair and
explicitly *not* where it isn't.  See `docs/benchmarks.md` for the
comparison table and fairness caveats.

| problem | description | headline metric |
| --- | --- | --- |
| `cartpole_throughput` | CartPole, fixed input, t_end=10s | wall-s / sim-s |
| `quadruped_throughput` | 4-leg single-joint, N=1000, t_end=5s | env-s / wall-s |
| `articulated_quadruped_throughput` | 12-DoF MJX body + floor contact, N=1, t_end=5s | wall-s / sim-s |
| `sysid_convergence` | fit (b, k) of m·ẍ+b·ẋ+k·x=u from noisy data | iter / wall / param_mse |
| `linearization_vs_analytic` | nonlinear oscillator linearized at x=0 | wall + max-abs-err vs analytic |

```bash
python benchmarks/public.py            # write baseline JSON
python benchmarks/public.py --check    # compare vs baseline; exit 1 on regression
```

### Baseline (macOS-arm64, JAX 0.9.2, mujoco 3.8.0, CPU only)

| problem | metric | value |
| --- | --- | ---: |
| cartpole_throughput | wall-s / sim-s (median of 3) | **0.0125** |
| quadruped_throughput (N=1000) | env-s / wall-s | **13,822** |
| articulated_quadruped_throughput (N=1, nq=19/nv=18) | wall-s / sim-s | **0.241** |
| sysid_convergence | iters / wall / param-MSE | 12 / 5.1 s / 5.1e-06 |
| linearization_vs_analytic | warm s / err vs analytic | 0.027 / **0.0** |

Cartpole simulates 10 seconds of dynamics in ~125 ms (~80 sim-s per
wall-s, single env). The simplified quadruped at N=1000 envs runs
5,000 env-seconds in ~0.36 s (~1.4M env-steps/s at 100 Hz).
**Articulated quadruped (T-019b)**: real 12-DoF body + free-joint
trunk + floor contact, MJX continuous-time path through dopri5. 5
seconds of multi-body physics in ~1.2 s wall (env-s/wall-s ≈ 4.2 at
N=1 — single-env, batched MJX is filed as
`T-019b-followup-batched`). Linearization matches the analytic
Jacobians to machine precision (the cubic vanishes at x=0).

### Honest gaps and follow-ups

* **GPU comparison data not collected.** All numbers are CPU; filed as
  **T-019a** (run the same problems on T4 / A100).
* **Simplified quadruped is a stand-in** — four independent damped
  pendulums, not a multi-body articulated quadruped. Characterises
  `simulate_batch` throughput, *not* feature parity. See
  `docs/benchmarks.md`.  The articulated quadruped case
  (`articulated_quadruped_throughput`) covers the multi-body story.
* **Articulated quadruped is single-env (N=1)**.  MJX's discrete-mode
  cache currently hits a contact-array dtype mismatch under
  `lax.cond`; the continuous-time path used here is single-env
  serial. Batched MJX is filed as **T-019b-followup-batched**.
* **Sysid fits 2 of 3 parameters** (mass anchored); three-param
  identifiability under noise is a separate study.

Threshold and rationale: same `REGRESSION_THRESHOLD = 1.5` as
`compile_time.py` / `memory.py` — wall-clock on shared CI is jittery,
+50 % is permissive enough to avoid spurious flaps but tight enough to
catch a real architectural regression.
