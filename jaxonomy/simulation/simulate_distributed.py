# SPDX-License-Identifier: MIT

"""Multi-device ensemble execution via :func:`jax.pmap`.

``simulate_distributed`` is a thin wrapper around the :func:`simulate_batch`
kernel that shards the leading batch dimension across the host's available
``jax.Device``s.  The same single-JIT kernel that ``simulate_batch`` uses for
its kernel path is reused; only the dispatch strategy changes.

When to reach for this:

* The diagram is pure-JAX (no ``CustomPythonBlock`` or FMU blocks).
* The host has more than one local device (multi-GPU, multi-TPU, or a fake
  multi-CPU mesh via ``XLA_FLAGS=--xla_force_host_platform_device_count=N``).
* The batch size is divisible by the number of devices used.

For *cluster-scale* fan-out (workloads that don't fit on a single host), wrap
``simulate_batch`` in an external orchestrator such as Modal, Replicate,
SkyPilot, or Ray Serve — see ``docs/distributed.md``.

The legacy managed Ray cluster path is explicitly out of scope (DEC-018).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..framework.diagram import Diagram
from ..framework.port import OutputPort
from .batch import (
    BatchSimulationResults,
    _infer_batch_size,
    _interp_on_time_np,
    _is_vmap_safe,
    _pure_patch_context,
)
from .errors import remap_simulation_errors
from .provenance import ProvenanceManifest, compute_provenance
from .simulator import _check_options
from .types import ResultsOptions, SimulatorOptions

__all__ = ["simulate_distributed"]


@remap_simulation_errors
def simulate_distributed(
    diagram: Diagram,
    t_span: tuple[float, float],
    param_batches: dict[str, Any],
    options: SimulatorOptions | None = None,
    recorded_signals: dict[str, OutputPort] | None = None,
    results_options: ResultsOptions | None = None,
    devices: list[jax.Device] | None = None,
) -> BatchSimulationResults:
    """Run an ensemble simulation pmap'd across ``devices``.

    Args:
        diagram: Pure-JAX template diagram.  Must contain no
            ``CustomPythonBlock`` or FMU blocks.
        t_span: ``(t_start, t_stop)``.
        param_batches: Dot-path → 1-D array of length ``N``, identical
            convention to :func:`simulate_batch`.
        options: :class:`SimulatorOptions` with ``math_backend='jax'`` and
            ``max_major_steps`` set.
        recorded_signals: Same convention as :func:`simulate`.
        results_options: Reserved for parity with :func:`simulate_batch`;
            currently unused on the kernel path.
        devices: List of :class:`jax.Device` objects; defaults to
            ``jax.devices()``.

    Returns:
        :class:`BatchSimulationResults` with ``outputs[name].shape[0] == N``,
        bit-equivalent to :func:`simulate_batch` up to XLA reduction-order
        rounding (~1e-10 relative).

    Raises:
        TypeError: ``diagram`` is not a :class:`Diagram`.
        ValueError: pre-conditions violated (non-pure-JAX diagram, missing
            options, or batch size not divisible by ``len(devices)``).
    """
    del results_options  # parity with simulate_batch; unused on the kernel path

    if not isinstance(diagram, Diagram):
        raise TypeError(
            f"simulate_distributed expects a Diagram, got {type(diagram)}"
        )
    if recorded_signals is None:
        raise ValueError(
            "simulate_distributed requires recorded_signals (same as simulate)."
        )
    if options is None:
        raise ValueError(
            "simulate_distributed requires SimulatorOptions with "
            "math_backend='jax' and max_major_steps set."
        )
    if options.math_backend != "jax":
        raise ValueError(
            "simulate_distributed only supports math_backend='jax', "
            f"got {options.math_backend!r}."
        )
    if options.max_major_steps is None or options.max_major_steps <= 0:
        raise ValueError(
            "simulate_distributed requires options.max_major_steps to be a "
            "positive int."
        )
    if not _is_vmap_safe(diagram):
        raise ValueError(
            "simulate_distributed requires a pure-JAX diagram (no "
            "CustomPythonBlock / FMU blocks). For non-pure diagrams, fall "
            "back to simulate_batch (which uses the safe loop path) and "
            "wrap that call in an external orchestrator (Modal, Replicate, "
            "SkyPilot — see docs/distributed.md)."
        )

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)
    if n_devices < 1:
        raise ValueError("simulate_distributed: no JAX devices available.")

    n = _infer_batch_size(param_batches)
    if n % n_devices != 0:
        raise ValueError(
            f"simulate_distributed: batch size N={n} is not divisible by "
            f"len(devices)={n_devices}. Pad the batch or use a divisor "
            "subset of devices via the `devices` kwarg."
        )
    per_dev = n // n_devices

    # T-110-followup-attach-on-batch: capture provenance once at the start
    # of the distributed batch.  Default-off path skips ``compute_provenance``
    # entirely so wall-clock and numerics are unchanged.  We attach the
    # manifest after the kernel returns so it survives both the 1-device
    # delegation and the multi-device pmap path uniformly.
    provenance_manifest: ProvenanceManifest | None = None
    if getattr(options, "record_provenance", False):
        provenance_manifest = compute_provenance(diagram, options)

    # 1-device degenerate case: defer to simulate_batch's kernel path so the
    # numerics match exactly (no pmap wrapper introduced).
    if n_devices == 1:
        from .batch import simulate_batch
        result = simulate_batch(
            diagram,
            t_span,
            param_batches,
            options=options,
            recorded_signals=recorded_signals,
        )
        if provenance_manifest is not None:
            # ``simulate_batch`` already gathered a manifest of its own when
            # ``record_provenance=True``; overwrite with the one captured at
            # ``simulate_distributed``'s entry so the user sees a single
            # consistent manifest for the distributed call.
            result.provenance = provenance_manifest
        return result

    # ----- Build the kernel once -------------------------------------------------
    from ..backend import ODESolver, set_backend
    from .simulator import Simulator

    set_backend("jax")
    opts = dataclasses.replace(options, enable_autodiff=False)
    opts_k = _check_options(diagram, opts, t_span, recorded_signals)

    base_ctx = diagram.create_context()
    stacked = {path: jnp.asarray(arr) for path, arr in param_batches.items()}

    ode_solver = ODESolver(diagram, options=opts_k.ode_options)
    sim = Simulator(diagram, ode_solver=ode_solver, options=opts_k)

    t0 = float(t_span[0])
    tf = float(t_span[1])

    def _kernel_single(context):
        return sim.advance_to(tf, context.with_time(t0))

    # ----- Stack a (N,) pytree of contexts, then reshape to (n_devices, per_dev)
    ctx_list = [
        _pure_patch_context(
            base_ctx, {path: jnp.asarray(stacked[path][i]) for path in stacked}
        )
        for i in range(n)
    ]
    stacked_ctx = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *ctx_list
    )
    sharded_ctx = jax.tree_util.tree_map(
        lambda x: x.reshape((n_devices, per_dev) + x.shape[1:]), stacked_ctx
    )

    def _per_device(ctx_shard):
        # ``axis_name="batch"`` mirrors the simulate_batch vmap path so that
        # T-122-followup-vmap-fold-in stochastic sources can derive a
        # per-replica independent PRNG stream via
        # ``jax.lax.axis_index("batch")``.
        return jax.vmap(_kernel_single, axis_name="batch")(ctx_shard)

    batch_state = jax.pmap(_per_device, devices=devices)(sharded_ctx)
    results_data_pmap = batch_state.results_data

    # Collapse (n_devices, per_dev, ...) -> (N, ...) for host-side finalize.
    flat_rd = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x).reshape((n,) + x.shape[2:]), results_data_pmap
    )

    # Per-element finalize on the host (variable trim length per element);
    # mirrors the kernel path in simulate_batch.
    time_ref = None
    out_lists: dict[str, list] = {k: [] for k in recorded_signals}
    for i in range(n):
        rd_i = jax.tree_util.tree_map(lambda x: x[i], flat_rd)
        time_i, outputs_i = rd_i.finalize()
        if time_ref is None:
            time_ref = time_i
            for sig_name in recorded_signals:
                out_lists[sig_name].append(np.asarray(outputs_i[sig_name]))
        else:
            same_grid = (
                np.asarray(time_i).shape == np.asarray(time_ref).shape
                and np.array_equal(np.asarray(time_i), np.asarray(time_ref))
            )
            for sig_name in recorded_signals:
                if same_grid:
                    out_lists[sig_name].append(np.asarray(outputs_i[sig_name]))
                else:
                    out_lists[sig_name].append(
                        _interp_on_time_np(
                            outputs_i[sig_name], time_i, time_ref
                        )
                    )

    stacked_out = {k: np.stack(vs, axis=0) for k, vs in out_lists.items()}
    return BatchSimulationResults(
        time=time_ref,
        outputs=stacked_out,
        used_vmap=False,
        provenance=provenance_manifest,
    )
