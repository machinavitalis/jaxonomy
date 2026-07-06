# Distributed ensemble execution

`simulate_batch` runs an N-element parameter sweep on a single JAX device.
`simulate_distributed` shards that same sweep across the host's local
devices using `jax.pmap`.  Cross-host fan-out (multi-machine clusters)
is intentionally out of scope — see "External orchestration" below.

## When to reach for which entry point

| your setup | recommended entry point |
| --- | --- |
| one CPU, one GPU, or one TPU | `simulate_batch` |
| one host, multiple GPUs / TPUs | `simulate_distributed` |
| many hosts (cluster scale) | external orchestration (Modal etc.) |
| diagram contains `CustomPythonBlock` or FMU | `simulate_batch` (loop path); orchestrate externally for fan-out |

`simulate_distributed` is a thin wrapper around the same single-JIT kernel
that `simulate_batch` already builds.  Numerics are bit-equivalent up to
XLA reduction-order rounding (`rtol ~1e-10` in the cases we've measured).

## API

```python
import jax
import jaxonomy

res = jaxonomy.simulate_distributed(
    diagram, t_span=(0.0, 5.0),
    param_batches={"leaf.k": ks},   # leading axis N must divide len(devices)
    options=opts,
    recorded_signals={"x": diagram["leaf"].output_ports[0]},
    devices=None,                    # default: jax.devices()
)
```

Constraints:

- The diagram must be pure-JAX (no `CustomPythonBlock` or FMU blocks).
  `simulate_distributed` raises with a clear error otherwise — wrap
  `simulate_batch` in an external orchestrator (below) for those.
- `N` (leading axis of every entry in `param_batches`) must be divisible by
  `len(devices)`.  The 1-device degenerate case defers to `simulate_batch`'s
  kernel path so numerics match exactly.

## Single-host multi-device recipe

For multi-GPU / multi-TPU, just call `simulate_distributed`.  To exercise
the path on a CPU-only dev machine, fake a multi-device mesh with an
`XLA_FLAGS` environment variable *before* importing JAX:

```python
import os
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)
import jax
import jaxonomy
# jax.devices() now returns 4 CpuDevice's
```

A runnable end-to-end example lives at `examples/distributed_ensemble.py`.
It runs `simulate_distributed` against a 4-device fake mesh and checks
the output against a serial `simulate_batch` call.

## External orchestration (cluster scale)

For workloads that exceed a single host (long-running ensembles, very large
`N`, or a mix of pure-JAX and `CustomPythonBlock` diagrams), wrap
`simulate_batch` in an external orchestrator.  Jaxonomy deliberately does
**not** ship its own job queue / worker fleet (DEC-018, T-206).  The
recommended options:

### Modal

[Modal](https://modal.com) gives a Python-native `@app.function` decorator
that runs the wrapped function in a managed container with optional GPU.
Distribute via `.map()` / `.for_each()`:

```python
import modal
import jax.numpy as jnp
import jaxonomy

app = modal.App("jaxonomy-sweep")
image = modal.Image.debian_slim().pip_install("jaxonomy")

@app.function(image=image, gpu="T4")
def run_one_shard(k_values):
    diagram = build_my_diagram()
    return jaxonomy.simulate_batch(
        diagram, t_span=(0.0, 5.0),
        param_batches={"plant.k": jnp.asarray(k_values)},
        options=opts, recorded_signals=rec,
    )

@app.local_entrypoint()
def main():
    shards = [list(jnp.linspace(0.1 + 0.5 * i, 0.5 + 0.5 * i, 100))
              for i in range(20)]
    results = list(run_one_shard.map(shards))
    # combine N=2000 result across 20 containers
```

### Replicate, SkyPilot, Ray Serve

The shape is identical: define one Python function that takes a parameter
shard and returns a `BatchSimulationResults`, then use the platform's
native fan-out (`replicate.run`, `sky launch`, Ray's `ray.remote` /
`Serve.deployment`) to dispatch shards across worker replicas.

### Why not in-library cluster orchestration?

DEC-018 cuts in-library worker-fleet / job-queue infrastructure.  It is
strictly worse than purpose-built tools (Modal, Replicate, SkyPilot, Ray)
on every axis (autoscaling, retries, cost reporting, multi-cloud).  The
in-library wrapper would only add a thin dispatching layer that those
tools already provide.

## When to consider extending T-021

`simulate_distributed` covers the multi-device-on-one-host case.  Extend
the API only if a real workload demonstrates need; candidate follow-ups:

- **Asynchronous fan-out across hosts** without an external orchestrator
  (would essentially rebuild Modal / Ray — not recommended).
- **`shard_map` over the time axis** to parallelise long-horizon single
  simulations across devices.  Useful only when one simulation exceeds a
  single device's memory; not a current bottleneck.
- **Hybrid pmap + vmap** with a non-trivial mesh (e.g. shard over GPUs and
  vmap within each GPU).  The current recipe already does this — pmap
  outer, vmap inner; if a more complex mesh is needed, switch to
  `shard_map` with an explicit `Mesh`.

File a follow-up `T-021-followup-<name>` if a concrete workload surfaces.
