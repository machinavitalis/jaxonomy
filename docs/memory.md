# Memory footprint and large parameter sweeps

How much resident memory (RSS) a Jaxonomy simulation will use, so you
can size a parallel sweep without OOMing your machine. All numbers
come from `benchmarks/memory.py`, median of 3 subprocess runs on a
GitHub-hosted Linux x86_64 runner / JAX 0.9.2 / jaxonomy 3.0.0 /
16.8 GB RAM.

## How much memory does an N-element parameter sweep cost?

For the 5-second exponential-decay benchmark (single integrator + gain,
the canonical `simulate_batch` test case):

| N elements | peak RSS |
| ---: | ---: |
| 1 | ~356 MB |
| 10 | ~358 MB |
| 100 | ~358 MB |

Linear fit gives **~0.01 MB per added batch element** on top of the
~356 MB process baseline (Python interpreter + JAX/XLA runtime + the
compiled simulator kernel) — essentially flat for this small case,
because the fixed-size recorded-signal buffer dominates the per-element
state. A 1000-element sweep is roughly **356 + 1000 × 0.01 ≈ 366 MB**.

For larger systems the slope grows with the per-element recorded-signal
buffer size. Multiply the per-element overhead by
`(buffer_length / 200) × (n_signals × dtype_bytes)` for a rough scaling.

## Long-horizon growth

A single harmonic-oscillator simulation at increasing `t_end`:

| t_end | peak RSS |
| ---: | ---: |
| 10 s | ~356 MB |
| 100 s | ~358 MB |
| 1000 s | ~356 MB |

**Essentially flat** (measured slope is ≈0 MB per simulated second).
Dopri5 records into a fixed-size buffer
(`SimulatorOptions.max_major_steps`, default 200), so memory does not
grow with simulated time, only with the buffer length you ask for. Plan
for ~5 MB per million steps per scalar recorded signal.

## Compile-time peak memory

Cold-compiling each benchmark case (subprocess isolated):

| case | peak RSS |
| --- | ---: |
| scalar_exponential_decay | ~298 MB |
| state_machine_three_state | ~299 MB |
| harmonic_oscillator | ~300 MB |
| bouncing_ball_zc | ~311 MB |
| simulate_batch_decay (N=10) | ~358 MB |
| rc_acausal_dae | ~422 MB |
| pid_first_order_plant | _skipped (control not installed)_ |

None cross 500 MB. **Allow ~500 MB headroom for compile spikes on
commodity 8 GB machines.** The acausal DAE case (`rc_acausal_dae`) is
the current peak — same root cause as its top-tier compile-time
slowdown in `jit_cache.md`. The PID case was skipped in the baseline
run because the optional `control` dependency wasn't installed; install
`.[safe]` or `python-control` to measure it.

## `simulate_batch` vs serial loop

For the N=100 sweep above, the per-element overhead (~0.01 MB/element,
≈2 MB at N=100) is negligible next to the JAX/XLA process baseline
(~356 MB). vmapped batch and a serial Python loop pay the same compile
baseline; the trade-off is **wall-clock, not memory**. vmapped batch is
3–10× faster on CPU at N=100 because it amortises Python dispatch and
lets XLA fuse across elements. Use `simulate_batch` whenever your sweep
fits in memory; the only reason to fall back to a serial loop is
N × per-element-MB exceeding RAM.

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
measurement — each datapoint pays the ~300 MB JAX-startup tax. For
*additional* memory (the slope of N → MB), subtract the smallest value
in the column. We don't attribute memory to specific JAX device
buffers vs the C++ heap vs Python objects; `tracemalloc` only sees the
Python heap (<5 % of the real total) and the XLA debug allocators are
not portably accessible. For finer-grained attribution, run under
`mprof` or `psrecord`.
