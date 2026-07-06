# Memory footprint and large parameter sweeps

How much resident memory (RSS) a Jaxonomy simulation will use, so you
can size a parallel sweep without OOMing your machine. All numbers
come from `benchmarks/memory.py` (T-018), median of 3 subprocess runs
on macOS-arm64 / JAX 0.9.2 / 16 GB RAM.

## How much memory does an N-element parameter sweep cost?

For the 5-second exponential-decay benchmark (single integrator + gain,
the canonical `simulate_batch` test case):

| N elements | peak RSS |
| ---: | ---: |
| 1 | ~323 MB |
| 10 | ~337 MB |
| 100 | ~373 MB |

Linear fit gives **~0.46 MB per added batch element** on top of the
~322 MB process baseline (Python interpreter + JAX/XLA runtime + the
compiled simulator kernel). A 1000-element sweep is roughly
**322 + 1000 × 0.46 ≈ 780 MB**.

For larger systems the slope grows with the per-element recorded-signal
buffer size. Multiply the per-element overhead by
`(buffer_length / 200) × (n_signals × dtype_bytes)` for a rough scaling.

## Long-horizon growth

A single harmonic-oscillator simulation at increasing `t_end`:

| t_end | peak RSS |
| ---: | ---: |
| 10 s | ~317 MB |
| 100 s | ~316 MB |
| 1000 s | ~318 MB |

**~0.001 MB per simulated second** — essentially flat. Dopri5 records
into a fixed-size buffer (`SimulatorOptions.max_major_steps`, default
200), so memory does not grow with simulated time, only with the
buffer length you ask for. Plan for ~5 MB per million steps per scalar
recorded signal.

## Compile-time peak memory

Cold-compiling each of T-017's seven benchmark cases (subprocess
isolated):

| case | peak RSS |
| --- | ---: |
| scalar_exponential_decay | ~254 MB |
| state_machine_three_state | ~256 MB |
| harmonic_oscillator | ~258 MB |
| bouncing_ball_zc | ~266 MB |
| simulate_batch_decay (N=10) | ~337 MB |
| rc_acausal_dae | ~368 MB |
| pid_first_order_plant | ~385 MB |

None cross 500 MB. **Allow ~500 MB headroom for compile spikes on
commodity 8 GB machines.** PID's two-state LTI wrapper (kp/ki/kd/n +
state-space matrices) is the current peak — same root cause as its
third-tier compile-time slowdown in `docs/jit_cache.md`.

## `simulate_batch` vs serial loop

For the N=100 sweep above, the per-element overhead (~46 MB) is small
next to the JAX/XLA process baseline (~320 MB). vmapped batch and a
serial Python loop pay the same compile baseline; the trade-off is
**wall-clock, not memory**. vmapped batch is 3–10× faster on CPU at
N=100 because it amortises Python dispatch and lets XLA fuse across
elements. Use `simulate_batch` whenever your sweep fits in memory; the
only reason to fall back to a serial loop is N × per-element-MB
exceeding RAM.

## Reproducing

```bash
python benchmarks/memory.py            # write a fresh baseline JSON
python benchmarks/memory.py --check    # compare a future run vs baseline
```

`--check` tolerates +50 % variance — peak-RSS measurements on shared
CI runners are noisy. See `benchmarks/README.md` for the threshold
rationale.

## Honest limitations

`ru_maxrss` reports peak RSS since process start, not a delta. To get
clean per-workload numbers, the benchmark forks a subprocess per
measurement — each datapoint pays the ~250 MB JAX-startup tax. For
*additional* memory (the slope of N → MB), subtract the smallest value
in the column. We don't attribute memory to specific JAX device
buffers vs the C++ heap vs Python objects; `tracemalloc` only sees the
Python heap (<5 % of the real total) and the XLA debug allocators are
not portably accessible. For finer-grained attribution, run under
`mprof` or `psrecord`.
