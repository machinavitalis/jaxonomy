# Public benchmarks

What Jaxonomy looks like compared to MuJoCo Playground and JaxSim on
standard control / simulation problems, with an honest accounting of
where the comparison is fair and where it isn't.

All Jaxonomy CPU numbers come from `benchmarks/public.py` — median of 3
runs on a GitHub-hosted Linux x86_64 runner / JAX 0.9.2 / jaxonomy 3.0.0
/ CPU. Reproduce with:

```bash
python benchmarks/public.py            # write fresh baseline JSON
python benchmarks/public.py --check    # compare vs locked baseline
```

## What we benchmark

Five problems, each exercising a different part of the stack:

1. **Cartpole throughput** — single-env continuous-time ODE,
   t_end = 10 s, fixed input. Stresses cold-compile + sim-loop on a
   small dense-Jacobian system.
2. **Quadruped throughput at N=1000 parallel envs** — `simulate_batch`
   over a simplified 4-leg single-joint pendulum (8 states),
   t_end = 5 s. Stresses the vmapped kernel path.
3. **Articulated quadruped throughput** — single-env MJX
   continuous-time rollout of a hand-authored 12-DoF quadruped (free-
   joint trunk + 4 legs × 3 hinges, nq=19, nv=18) onto a floor with
   contact, t_end = 5 s, no controller. Measures multi-body physics
   throughput on an articulated body — *not* locomotion / RL env-step
   rate.  Two quadruped benchmarks coexist deliberately:
   `quadruped_throughput` (4 independent damped pendulums, a
   `simulate_batch` throughput stand-in) and
   `articulated_quadruped_throughput` (real 12-DoF MJX body, multi-
   body dynamics with contact).
4. **System-ID convergence** — fit `(b, k)` of a spring-damper from
   noisy synthetic data using L-BFGS-B + finite-diff gradients on the
   simulator. Characterises the optimisation stack.
5. **Linearization vs analytic** — `jaxonomy.linearize()` on a
   nonlinear oscillator at the equilibrium x=0 vs the closed-form
   Jacobians. Measures wall time *and* numerical accuracy.

## Headline numbers

CPU numbers are jaxonomy 3.0.0 on the Linux x86_64 baseline runner. The
T4 column was collected separately on an NVIDIA T4 host running jaxonomy
**2.2.0** (see "Hardware notes") — treat it as indicative, not a
version-matched comparison, and note that these small `float64` problems
are often *slower* on a T4 than on CPU.

| problem | metric | Jaxonomy CPU | T4 (2.2.0) | A100 | H100 |
| --- | --- | ---: | ---: | ---: | ---: |
| cartpole_throughput (t_end=10s) | wall-s / sim-s | 0.049 | 0.103 | _pending_ | _pending_ |
| quadruped_throughput (N=1000, t_end=5s) | env-s / wall-s | 3,967 | 101 | _pending_ | _pending_ |
| articulated_quadruped_throughput (N=1, nq=19/nv=18, t_end=5s) | wall-s / sim-s | 0.796 | 0.968 | _pending_ | _pending_ |
| sysid_convergence | iters / wall / param-MSE | 12 / 20.7 s / 5.2e-06 | 12 / 45.9 s / 5.2e-06 | _pending_ | _pending_ |
| linearization_vs_analytic | warm s / err vs analytic | 0.068 / 0.0 | 0.166 / 0.0 | _pending_ | _pending_ |

A100 / H100 columns are populated by re-running `benchmarks/public.py`
on the corresponding NVIDIA runner (runner contract below). Empty cells
mean we have not yet collected the number on that device — they are
*not* zeros and not interpolated from CPU.

Cartpole simulates 10 simulated-seconds in ~490 ms on a single CPU core
(~20 simulated-s per wall-s). The simplified quadruped batch at N=1000
runs 5,000 env-seconds in ~1.26 s wall (~3,967 env-s/wall-s, ≈0.4M
env-steps/s at a 100 Hz step). The articulated quadruped (12-DoF body +
floor contact, MJX continuous-time, no controller) simulates 5 s of
multi-body physics in ~4.0 s wall (env-s/wall-s ≈ 1.26). Linearization
matches the analytic Jacobians to machine precision (the cubic vanishes
at x=0).

## How to think about comparisons

Jaxonomy, MuJoCo Playground, and JaxSim each optimise for different
things. Direct head-to-head numbers are only meaningful on a subset of
workloads, and misleading on the rest.

| library | primary target | fair comparison axis |
| --- | --- | --- |
| **Jaxonomy** | accuracy-first ODE/DAE simulation, controls, autodiff, sysid | small/medium ODE throughput, linearisation correctness, batch parameter sweeps |
| **MuJoCo Playground** | RL training throughput, contact-rich rigid bodies | massively-parallel RL env-steps/s, GPU/TPU only |
| **JaxSim** | robotics multi-body dynamics with contact | articulated rigid bodies, jit-friendly featherstone/RBDL |

**Fair**: single-env continuous ODE (Jaxonomy vs JaxSim on a smooth
contact-free system); batched parameter sweeps with `vmap` on the
same body; linearisation accuracy at an equilibrium; single-env
articulated multi-body throughput on a fixed reference body
(`articulated_quadruped_throughput` is fair vs JaxSim's articulated-
body benchmarks on the same body, modulo solver choice).

**Not fair**: contact-rich RL throughput on GPU (MuJoCo Playground's
~10⁵ env-steps/s for cartpole at 1000 envs is a T4-GPU MJX number —
our CPU number isn't comparable); MuJoCo Playground RL-policy-step
throughput, which includes a learned controller and contact-rich gait
that our zero-control passive drop deliberately excludes;
massively-parallel articulated bodies on GPU (our articulated quadruped
is single-env; a batched MJX ensemble is a filed follow-up); MPC
inner-loop solve at 1 kHz (out of scope for all three).

## Competitor numbers and sourcing

External-reference numbers used here:

* **MuJoCo Playground cartpole-1000-env on T4 GPU**: ~10⁵ env-steps/s
  (source: MuJoCo Playground README).
* **JaxSim multi-body throughput**: comparable order of magnitude on
  GPU for articulated bodies; we have not run a like-for-like
  benchmark of our simplified quadruped against a JaxSim
  pendulum-array equivalent.

If a number isn't here, it's because we don't have a defensible
measurement, not because we cherry-picked. **No competitor numbers
were estimated, interpolated, or derived from secondary sources.**

## Hardware notes

The CPU baseline is produced on a GitHub-hosted Linux x86_64 runner
(Azure-backed `ubuntu-latest`) / 16.8 GB RAM / JAX 0.9.2 / jaxonomy
3.0.0 / CPU only. CI re-runs weekly on `ubuntu-latest` and uploads the
JSON; cross-runner CPU variance is ~10-20 %.

The T4 column was collected on a separate NVIDIA T4 host (Linux
x86_64, 31.5 GB RAM, JAX 0.9.2) running **jaxonomy 2.2.0** — it predates
the current 3.0.0 CPU baseline, so it is indicative rather than
version-matched. A100 / H100 numbers are populated opportunistically
when a matching NVIDIA runner is available — see "GPU runner contract"
below for the recipe. The exact driver / CUDA / cuDNN fingerprint for
each device run is recorded under `hardware.<device_key>` in
`public_baseline.json`.

## GPU runner contract

Recipe for filling in the T4 / A100 / H100 columns when you have
access to an NVIDIA box.

### JAX platform names

JAX accepts both `gpu` and `cuda` for NVIDIA backends. `cuda` is the
canonical name as of JAX 0.4.x+; `gpu` is the documented alias and
what `jax.default_backend()` prints on a CUDA host. Either string is
valid input to `--device`. Use `jax.devices()` to confirm:

```python
import jax
print(jax.default_backend())   # "gpu" on a CUDA host
print(jax.devices())           # [CudaDevice(id=0)] or [GpuDevice(id=0)]
```

The benchmark script forces `JAX_PLATFORMS=cuda,cpu` early at module
import when `--device gpu` (or env var `JAXONOMY_BENCH_DEVICE=gpu`) is
set, so JAX's lazy backend init picks the GPU before any `jax.numpy`
arrays are allocated.

### Populating a device column

```bash
# A T4 host (Colab / GCP n1-standard-4 + nvidia-tesla-t4 / GHA gpu runner)
JAXONOMY_BENCH_DEVICE=gpu python benchmarks/public.py \
    --device gpu --update-baseline gpu_t4

# A100 host (e.g. AWS p4d, Lambda Labs, Modal A100 sandbox)
JAXONOMY_BENCH_DEVICE=gpu python benchmarks/public.py \
    --device gpu --update-baseline gpu_a100

# H100 host (AWS p5, Modal H100, GCP A3)
JAXONOMY_BENCH_DEVICE=gpu python benchmarks/public.py \
    --device gpu --update-baseline gpu_h100
```

`--update-baseline gpu_t4` writes only the `gpu_t4` column under each
`cases.<name>` entry; the existing `cpu` column and other GPU columns
are preserved. The hardware fingerprint for the run lands at
`hardware.gpu_t4` so reviewers can verify driver / CUDA / cuDNN
versions after the fact.

### Verifying without overwriting

```bash
JAXONOMY_BENCH_DEVICE=gpu python benchmarks/public.py \
    --device gpu --check
```

If `gpu` (or the resolved key — `gpu_t4` / `gpu_a100` / `gpu_h100`)
isn't in the baseline yet, every case prints `NEW` and the run exits 0.
This is the backwards-compatibility guarantee: missing device columns
do not fail `--check`.

### Caveats for GPU runs

* **Warmup.** The first `simulate` call pays the XLA AOT compile +
  PTX-to-SASS lowering cost. The script already takes the *warm*
  number (second call) for `cartpole_throughput`. For ensemble cases
  (`quadruped_throughput`) the kernel JIT and the `simulate_batch`
  scan body each cost ~hundreds of ms cold; the recorded `compile_s`
  isolates that.
* **Persistent JIT cache.** Setting
  `JAXONOMY_PERSISTENT_JIT_CACHE=1` (see `jit_cache.md`) makes
  re-runs on the same machine essentially free of compile cost — useful
  for back-to-back `--check` runs but irrelevant to the published
  numbers, which always use the `compile_s_median` /
  `wall_per_simsec_median` fields.
* **Default dtype.** Jaxonomy enforces `float64` by default.
  Some GPUs (especially consumer ones) are massively faster in
  `float32`; do *not* override the precision policy when populating the
  public baseline — that would compare apples to oranges. It is also why
  the T4 numbers above are slower than CPU on these small problems: a T4
  has little `float64` throughput. If you want a separate float32 column,
  add a new device key (e.g. `gpu_a100_f32`).
* **Batch-size scaling.** `quadruped_throughput` is fixed at N=1000 and
  `articulated_quadruped_throughput` at N=1 deliberately, so numbers
  are comparable across devices. Don't bump N to "show off" a bigger
  GPU — that breaks the cross-device comparison. Larger-N is a filed
  follow-up.
* **Persistent NVIDIA driver state.** If the runner is shared (HF
  Spaces, Colab), free GPU memory before invoking via
  `nvidia-smi --gpu-reset` or by restarting the kernel. Compile time
  jitter from leftover allocations is the most common false-regression
  signal.

### Runner specs

Use these as the canonical hosts for each column. Numbers must come
from a host that matches the spec — driver / CUDA / cuDNN versions
should be recorded in `hardware.<device_key>` automatically.

| device_key | runner | accelerator | host CPU | RAM | notes |
| --- | --- | --- | --- | --- | --- |
| `gpu_t4` | NVIDIA T4 host (Colab / GCP n1 + nvidia-tesla-t4) | NVIDIA T4 (16 GB) | 4 vCPU | 31.5 GB | measured on jaxonomy 2.2.0; parity re-run on 3.0.0 pending |
| `gpu_a100` | AWS p4d.24xlarge, Modal a100-40gb sandbox (pending) | NVIDIA A100 40 GB | 8 vCPU | 64 GB+ | mid-tier reference for a fair head-to-head with JaxSim's published numbers |
| `gpu_h100` | AWS p5.48xlarge, Modal h100 sandbox, GCP a3-highgpu (pending) | NVIDIA H100 80 GB | 8 vCPU | 128 GB+ | upper bound; JAX 0.9+ is required for full H100 SM_90 codegen |

When a number lands, record the exact runner identifier (e.g.
`Modal a100-40gb-<run-date>`, `GHA self-hosted [linux, gpu, t4] runner
#3`) so future regressions can be triaged against the same hardware.

### Status

**T4 numbers are measured** (on jaxonomy 2.2.0 — a version-matched 3.0.0
re-run is still pending), and are populated in the headline table above.
The `gpu_a100` / `gpu_h100` columns in `public_baseline.json` are still
explicit `null`s: those runs are pending A100 / H100 access on GitHub
Actions GPU self-hosted runners or a Modal / Replicate notebook (recipe
above).

## Caveats

* **Simplified quadruped is a stand-in.** Four independent damped
  pendulums, not multi-body locomotion — the benchmark characterises
  `simulate_batch` throughput, not feature parity.  The companion
  `articulated_quadruped_throughput` case covers the multi-body story.
* **Articulated quadruped is single-env (N=1)**, no controller, just
  a passive-drop onto a floor under gravity.  Honest physics-
  throughput measurement on an articulated body, *not* a locomotion
  / RL benchmark.  A batched MJX ensemble is a filed follow-up.
* **Sysid fits 2 of 3 parameters** (mass anchored); three-param
  identifiability under noise is a separate study.
* **Cartpole input held at zero**, no controller. Apples-to-apples
  vs other libraries on the same setup; absolute number would shift
  with an LQR in the loop.
* **A100 / H100 columns not yet measured.** The schema, CLI flag, and
  runner contract for GPU runs are in place (see the "GPU runner
  contract" section above). `gpu_a100` / `gpu_h100` are currently
  explicit `null`s in `public_baseline.json`; numbers land when a
  matching NVIDIA runner is provisioned.

## Reproduction checklist

1. `pip install -e ".[test]"` from a clean checkout.
2. `python benchmarks/public.py` writes `public_baseline.json`
   (includes hardware fingerprint).
3. CI runs `--check` weekly (Mon 06:00 UTC) plus on
   `workflow_dispatch`. *Not* on PR/push.

## Filed follow-ups

* **GPU benchmark numbers (A100 / H100).** Schema + CLI flag + runner
  contract shipped (see "GPU runner contract" above); the `gpu_a100` /
  `gpu_h100` rows in `public_baseline.json` are `null` pending an A100 /
  H100 run. The `gpu_t4` row is populated (jaxonomy 2.2.0).
* **Real articulated-quadruped vs JaxSim** on a fixed URDF.
