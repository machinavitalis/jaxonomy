# SPDX-License-Identifier: MIT

"""Batch (ensemble) simulation over parameter sweeps.

``simulate_batch`` runs many simulations that differ only by dynamic parameters.
Two execution paths are provided:

**Loop path (always safe)**:
  Runs N simulations sequentially in a Python loop using ``diagram.with_parameters``
  + ``simulate``. Handles ``CustomPythonBlock``, FMU, and any host-callback based
  block. Each iteration re-JIT-compiles the simulation (N compilations total).

**Kernel path (pure-JAX diagrams only)**:
  When the diagram contains no ``CustomPythonBlock`` or FMU blocks — that is, every
  block is JAX-traceable — a single ``Simulator`` is built once and its
  ``advance_to`` method is compiled into one JIT-cached kernel.  Parameter values
  are injected into a fresh context using a purely-functional patch (no
  ``ParameterCache`` mutations) and the same compiled XLA program is reused for all
  N batch elements.  This eliminates N−1 expensive recompilation calls, giving a
  significant speedup for large sweeps on CPU and GPU alike.

  An optional ``use_vmap=True`` flag additionally vectorises over the batch
  dimension with ``jax.vmap`` so that all N simulations run in a single
  XLA launch (true data-parallelism on GPU/TPU).

The path is selected automatically (kernel path when pure-JAX, loop path otherwise)
unless overridden with the ``_force_loop`` keyword argument.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..framework.diagram import Diagram
from ..framework.port import OutputPort
from .errors import remap_simulation_errors
from .provenance import ProvenanceManifest, compute_provenance
from .results_recorder import maybe_warn_recording_truncation
from .simulator import simulate, _check_options
from .types import ResultsOptions, SimulationResults, SimulatorOptions

__all__ = [
    "attach_provenance_to_batch",
    "BatchSimulationResults",
    "simulate_batch",
]


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BatchSimulationResults:
    """Results from :func:`simulate_batch`.

    Attributes:
        time: Time vector of shape ``(T,)`` taken from the first batch run.  Later runs
            are linearly interpolated onto this grid so all batch rows align.
        outputs: Mapping ``signal_name -> array`` with shape ``(N, T, ...)`` where
            ``N`` is batch size.
        used_vmap: ``True`` if the vectorised vmap path was used.
        provenance: Optional :class:`ProvenanceManifest` capturing library
            versions, options and the system fingerprint at batch-start
            (T-110-followup-attach-on-batch).  ``None`` unless
            ``SimulatorOptions.record_provenance=True``.  One manifest is
            shared across all batch replicas — they all share the same
            system + options, only the parameter values differ (and those
            are already in ``param_batches``).
    """

    time: Any
    outputs: dict[str, Any]
    used_vmap: bool = False
    provenance: ProvenanceManifest | None = None

    def mean(self, signal: str) -> Any:
        """Mean trajectory across the batch (axis 0)."""
        return jnp.mean(self.outputs[signal], axis=0)

    def std(self, signal: str) -> Any:
        """Standard deviation across the batch (axis 0)."""
        return jnp.std(self.outputs[signal], axis=0)

    def percentile(self, signal: str, p: float) -> Any:
        """``p``-th percentile across the batch at each time index; ``p`` in ``[0, 100]``."""
        return jnp.percentile(self.outputs[signal], p, axis=0)

    def to_simulation_results(self, idx: int) -> SimulationResults:
        """Slice one batch index into a :class:`SimulationResults` (no final context)."""
        return SimulationResults(
            None,
            time=self.time,
            outputs={k: v[idx] for k, v in self.outputs.items()},
            parameters=None,
        )


def attach_provenance_to_batch(
    results: BatchSimulationResults,
    system: Diagram | None,
    options: SimulatorOptions | None,
) -> BatchSimulationResults:
    """Attach a :class:`ProvenanceManifest` to ``results`` in place and return it.

    Standalone helper for the rare case where a user has a
    :class:`BatchSimulationResults` produced without
    ``record_provenance=True`` and now wants reproducibility metadata
    attached (for example, after-the-fact archival).  In the normal flow,
    :func:`simulate_batch` and :func:`simulate_distributed` already wire
    up the manifest when ``options.record_provenance=True``; this helper
    is just the explicit, opt-in escape hatch.

    Args:
        results: A :class:`BatchSimulationResults` to mutate.
        system: The diagram that was simulated (passed through to
            :func:`compute_provenance`).
        options: The :class:`SimulatorOptions` used for the batch run.

    Returns:
        The same ``results`` instance, with ``results.provenance``
        populated.
    """
    results.provenance = compute_provenance(system, options)
    return results


# ---------------------------------------------------------------------------
# Helpers shared by both paths
# ---------------------------------------------------------------------------

def _remap_recorded_signals(
    template_diagram: Diagram,
    updated_diagram: Diagram,
    recorded_signals: dict[str, OutputPort],
) -> dict[str, OutputPort]:
    """Map template output ports to the homologous ports on ``updated_diagram``."""
    out: dict[str, OutputPort] = {}
    for name, port in recorded_signals.items():
        if not isinstance(port, OutputPort):
            raise TypeError(
                f"recorded_signals[{name!r}] must be an OutputPort, got {type(port)}"
            )
        sys_t = port.system
        # Diagram-level output on the root diagram (identity, not name match)
        if sys_t is template_diagram:
            out[name] = updated_diagram.output_ports[port.index]
            continue
        path = sys_t.name_path_str
        sub = updated_diagram.find_system_with_path(path)
        if sub is None:
            raise ValueError(
                f"Could not remap recorded signal {name!r}: no subsystem {path!r} "
                f"under diagram {updated_diagram.name!r}"
            )
        out[name] = sub.output_ports[port.index]
    return out


def _interp_on_time(y: Any, t_from: Any, t_to: Any) -> Any:
    """Resample time series ``y`` defined on ``t_from`` onto grid ``t_to``."""
    y = jnp.asarray(y)
    t_from = jnp.asarray(t_from).reshape(-1)
    t_to = jnp.asarray(t_to).reshape(-1)
    if y.shape[0] != t_from.shape[0]:
        raise ValueError(
            f"interp_on_time: length mismatch y.shape[0]={y.shape[0]} vs len(t_from)={t_from.shape[0]}"
        )
    if y.ndim == 1:
        return jnp.interp(t_to, t_from, y)
    flat = y.reshape((t_from.shape[0], -1))

    def interp_col(fp):
        return jnp.interp(t_to, t_from, fp)

    out = jax.vmap(interp_col, in_axes=1, out_axes=1)(flat)
    return out.reshape((t_to.shape[0],) + y.shape[1:])


def _interp_on_time_np(y: Any, t_from: Any, t_to: Any) -> np.ndarray:
    """Numpy-host variant of :func:`_interp_on_time` (T-017c).

    The ``_scan_kernel_path`` and ``_simulate_batch_loop`` paths receive
    per-element outputs that have already been materialised to numpy by
    ``JaxResultsData._trim``.  Calling the JAX variant on those numpy arrays
    re-jits ``jnp.interp`` for every distinct shape combination
    (``len(t_from)`` varies because adaptive ODE solvers terminate on
    different step counts).  At N=100 that re-compile dominates wall clock
    (~10 ms each × ~25 distinct shapes ≈ 250 ms; combined with the
    surrounding host→device copies the loop spends ~1 s on resampling
    alone).  ``numpy.interp`` runs on already-host arrays in ~µs and has
    no compile cost, so this helper completely removes that hotspot.
    Output is bit-equivalent to the JAX path (same linear-interp formula,
    float64 throughout).
    """
    y = np.asarray(y)
    t_from = np.asarray(t_from).reshape(-1)
    t_to = np.asarray(t_to).reshape(-1)
    if y.shape[0] != t_from.shape[0]:
        raise ValueError(
            f"interp_on_time_np: length mismatch y.shape[0]={y.shape[0]} "
            f"vs len(t_from)={t_from.shape[0]}"
        )
    if y.ndim == 1:
        return np.interp(t_to, t_from, y)
    flat = y.reshape((t_from.shape[0], -1))
    # numpy.interp is 1-D only; vectorise over the trailing flattened axis.
    out = np.empty((t_to.shape[0], flat.shape[1]), dtype=y.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_to, t_from, flat[:, j])
    return out.reshape((t_to.shape[0],) + y.shape[1:])


def _batched_searchsorted(t_rows: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """Row-wise ``np.searchsorted(t_rows[i], tq, side="right")``, vectorised.

    T-019-followup: numpy has no batched searchsorted, but each row of the
    recorded time buffer is sorted ascending with an ``inf`` tail (unused
    slots), so a manual binary search vectorised over ``(N, M)`` replaces
    the per-row Python loop. ``O(N·M·log T)`` fully in C.

    Args:
        t_rows: ``(N, T)`` per-row sorted times (``inf``-padded tails).
        tq: ``(M,)`` query grid.

    Returns:
        ``(N, M)`` int64 insertion indices (``side="right"`` convention).
    """
    n, t_len = t_rows.shape
    m = tq.shape[0]
    lo = np.zeros((n, m), dtype=np.int64)
    hi = np.full((n, m), t_len, dtype=np.int64)
    rows = np.arange(n)[:, None]
    for _ in range(int(np.ceil(np.log2(max(t_len, 2)))) + 1):
        mid = (lo + hi) >> 1
        go_right = t_rows[rows, mid] <= tq[None, :]
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(go_right, hi, mid)
    return lo


def _batched_interp_rows(
    t_rows: np.ndarray,
    y_rows: np.ndarray,
    counts: np.ndarray,
    tq: np.ndarray,
) -> np.ndarray:
    """Linearly resample every batch row onto ``tq`` in one vectorised pass.

    Equivalent to ``np.stack([np.interp(tq, t_rows[i, :counts[i]],
    y_rows[i, :counts[i]]) for i in range(N)])`` (with trailing signal
    dims handled by broadcasting), including the endpoint-clamp
    extrapolation semantics of :func:`np.interp`.

    Args:
        t_rows: ``(N, T)`` per-row times, sorted, ``inf``-padded.
        y_rows: ``(N, T, *sig)`` recorded values.
        counts: ``(N,)`` number of valid samples per row (each ``>= 2``).
        tq: ``(M,)`` reference grid.

    Returns:
        ``(N, M, *sig)`` resampled values.
    """
    n = t_rows.shape[0]
    rows = np.arange(n)[:, None]
    # Bracket [j0, j1] fully inside each row's valid region.
    j1 = np.clip(_batched_searchsorted(t_rows, tq), 1, counts[:, None] - 1)
    j0 = j1 - 1
    x0 = t_rows[rows, j0]
    x1 = t_rows[rows, j1]
    denom = x1 - x0
    denom = np.where(denom == 0.0, 1.0, denom)
    # Clamp to [0, 1]: queries left of the first sample take it verbatim,
    # queries right of the last valid sample take that one (np.interp
    # endpoint semantics).
    w = np.clip((tq[None, :] - x0) / denom, 0.0, 1.0)
    y0 = y_rows[rows, j0]
    y1 = y_rows[rows, j1]
    w = w.reshape(w.shape + (1,) * (y_rows.ndim - 2))
    return y0 + w * (y1 - y0)


def _infer_batch_size(param_batches: dict[str, Any]) -> int:
    if not param_batches:
        raise ValueError("param_batches must be non-empty")
    sizes = []
    for k, v in param_batches.items():
        a = jnp.asarray(v)
        if a.ndim < 1:
            raise ValueError(
                f"param_batches[{k!r}] must be at least 1-D with leading batch size, "
                f"got shape {a.shape}"
            )
        sizes.append(int(a.shape[0]))
    n0 = sizes[0]
    for k, s in zip(param_batches.keys(), sizes):
        if s != n0:
            raise ValueError(
                "param_batches must share the same leading batch size; "
                f"got {n0} vs {s} for key {k!r}"
            )
    return n0


# ---------------------------------------------------------------------------
# Pure-JAX detection
# ---------------------------------------------------------------------------

def _is_vmap_safe(diagram: Diagram) -> bool:
    """Return ``True`` iff the diagram tree contains no non-JAX-traceable blocks.

    Specifically, returns ``False`` if any block is a :class:`CustomPythonBlock`
    (uses ``io_callback``) or an FMU block (also uses host callbacks).  All other
    standard library blocks are JAX-traceable and therefore safe for the kernel /
    vmap path.
    """
    from ..framework.validation import _is_custom_python_block, _is_fmu_block

    def _walk(system) -> bool:
        if _is_custom_python_block(system) or _is_fmu_block(system):
            return False
        if hasattr(system, "nodes"):
            return all(_walk(sub) for sub in system.nodes)
        return True

    return _walk(diagram)


# ---------------------------------------------------------------------------
# Pure-functional context parameter injection (no ParameterCache side effects)
# ---------------------------------------------------------------------------

def _pure_patch_context(ctx, updates: dict[str, Any]):
    """Inject parameter values into a context tree without ``ParameterCache`` mutations.

    The context is a JAX pytree; this function performs only structural
    (Python-level) replacements that leave the pytree shape intact.  As a
    result, a ``jax.jit``-compiled function that previously accepted ``ctx``
    will accept the patched context without recompilation, because the
    abstract pytree structure (shapes, dtypes, tree def) is identical.

    Args:
        ctx: Root :class:`DiagramContext` or :class:`LeafContext`.
        updates: Dot-path → value mapping, e.g. ``{"gain.gain": 2.0}``.
            Nested paths like ``"sub.child.param"`` are routed recursively.

    Returns:
        New context with updated parameter values.
    """
    from ..framework.context import DiagramContext, LeafContext

    if not updates:
        return ctx

    if isinstance(ctx, LeafContext):
        # All remaining updates apply to this leaf's own parameters.
        # Coerce Python scalars / 0-d numpy arrays to ``jnp.asarray`` so the
        # patched context's pytree leaves keep the same abstract dtype/shape
        # as the original — otherwise ``jax.jit`` retraces on each new scalar
        # value (T-008-followup-with-parameter-trace-cache).
        from ..framework.context import _coerce_param_for_jit_cache
        coerced = {
            name: _coerce_param_for_jit_cache(ctx.parameters.get(name), val)
            for name, val in updates.items()
        }
        new_params = {**ctx.parameters, **coerced}
        return dataclasses.replace(ctx, parameters=new_params)

    # --- DiagramContext: split updates into "own" vs. "child" ---
    own_updates: dict[str, Any] = {}
    child_updates: dict[str, dict[str, Any]] = {}

    for path, val in updates.items():
        parts = path.split(".", 1)
        if len(parts) == 1:
            own_updates[parts[0]] = val
        else:
            block_name, remainder = parts
            child_updates.setdefault(block_name, {})[remainder] = val

    new_subcontexts = dict(ctx.subcontexts)

    for block_name, block_upd in child_updates.items():
        matched_id = None
        for sys_id, subctx in ctx.subcontexts.items():
            if subctx.owning_system.name == block_name:
                matched_id = sys_id
                break
        if matched_id is None:
            avail = [sc.owning_system.name for sc in ctx.subcontexts.values()]
            raise KeyError(
                f"_pure_patch_context: no subsystem named {block_name!r} in "
                f"{ctx.owning_system.name!r}. Available: {avail}"
            )
        new_subcontexts[matched_id] = _pure_patch_context(
            ctx.subcontexts[matched_id], block_upd
        )

    from ..framework.context import _coerce_param_for_jit_cache
    coerced_own = {
        name: _coerce_param_for_jit_cache(ctx.parameters.get(name), val)
        for name, val in own_updates.items()
    }
    new_own_params = {**ctx.parameters, **coerced_own}
    return dataclasses.replace(ctx, subcontexts=new_subcontexts, parameters=new_own_params)


# ---------------------------------------------------------------------------
# Kernel path (single JIT compilation, optional vmap)
# ---------------------------------------------------------------------------

def _build_kernel(diagram: Diagram, opts: SimulatorOptions, t_span, recorded_signals):
    """Build a JIT-compiled simulation kernel for a pure-JAX diagram.

    Returns a callable ``kernel(context) -> (time, outputs_dict)`` where:

    * The ``Simulator`` and ``ODESolver`` are built **once**.
    * The kernel is compiled once by ``jax.jit``.
    * Calling it with different contexts that have the same pytree structure
      (same shapes/dtypes, only different array values) reuses the compiled
      XLA program — O(1) compilation regardless of batch size.

    ``recorded_signals`` must refer to ports on ``diagram`` (not a copy).
    """
    from ..backend import ODESolver, set_backend
    from .results_recorder import ResultsRecorder
    from .simulator import Simulator

    set_backend("jax")

    ode_solver = ODESolver(diagram, options=opts.ode_options)
    sim = Simulator(diagram, ode_solver=ode_solver, options=opts)

    recorder = ResultsRecorder(recorded_signals, opts)
    t0, tf = float(t_span[0]), float(t_span[1])

    @jax.jit
    def _kernel(context):
        initial_context = context.with_time(t0)
        # Re-initialize results_data inside the JIT so buffer shapes are correct
        results_data = recorder.initialize(initial_context)
        # Manually set the results_data on the initial SimulatorState
        sim_state = sim.initialize(initial_context)
        sim_state = dataclasses.replace(sim_state, results_data=results_data)
        sim_state = sim.advance_to(tf, sim_state.context)
        return sim_state.results_data

    def run(context):
        results_data = _kernel(context)
        time, outputs = results_data.finalize()
        return time, outputs

    return run


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@remap_simulation_errors
def simulate_batch(
    diagram: Diagram,
    t_span: tuple[float, float],
    param_batches: dict[str, Any],
    options: SimulatorOptions | None = None,
    recorded_signals: dict[str, OutputPort] | None = None,
    results_options: ResultsOptions | None = None,
    use_vmap: bool = False,
    _force_loop: bool = False,
    lazy: bool = False,
) -> BatchSimulationResults:
    """Run ``N`` simulations differing only by parameters given in ``param_batches``.

    **Execution paths**:

    * **Kernel path** (default for pure-JAX diagrams): builds the simulator once,
      compiles a single JIT kernel, and injects each batch element's parameters
      directly into a context pytree (no ``ParameterCache`` mutations).  This
      eliminates N−1 recompilations and is substantially faster for moderate to
      large N.

    * **vmap path** (opt-in, ``use_vmap=True``, pure-JAX only): further vectorises
      over the batch dimension with ``jax.vmap`` so all N simulations run as a
      single XLA call.  Requires that all parameter values have compatible shapes
      and that the simulation fits in device memory N-fold.

      **CPU note (updated by T-019-followup, 2026-07-10).** The
      post-vmap finalize is now fully vectorised (batched trim +
      batched binary-search linear resampling instead of a per-row
      host loop), which removed the old CPU penalty: on the CPU
      damped-oscillator sweep at ``N=1000`` the vmap path improved
      from ~1.28 s to ~0.41 s against ~0.33 s for the kernel path
      (naive loop ~130 s, FastRestart ~0.30 s). CPU kernel-path wins
      are now marginal; on GPU / TPU vmap wins decisively. The old
      CPU+small-batch ``UserWarning`` was removed along with the
      penalty it warned about.

    * **Loop path** (forced when ``CustomPythonBlock`` or FMU blocks are present,
      or when ``_force_loop=True``): the safe fallback — N independent calls to
      ``simulate`` + ``with_parameters``.

    Args:
        diagram: Template diagram (unchanged).
        t_span: ``(t_start, t_stop)``.
        param_batches: Dot-path keys mapping to 1-D arrays of length ``N`` (same ``N``
            for every entry), e.g. ``{"gain.gain": jnp.linspace(0.5, 2.0, 16)}``.
        options: :class:`SimulatorOptions` with ``math_backend="jax"`` and
            ``max_major_steps`` set (required).
        recorded_signals: Same convention as :func:`simulate` (ports refer to the
            template ``diagram``; they are remapped per updated diagram for the loop
            path but used directly for the kernel / vmap path).
        results_options: Optional :class:`ResultsOptions` passed through.
        use_vmap: If ``True``, attempt vectorisation via ``jax.vmap`` (pure-JAX
            diagrams only). Raises ``ValueError`` if the diagram is not pure-JAX.
        _force_loop: If ``True``, always use the loop path regardless of diagram
            type (useful for testing / debugging).

    Returns:
        :class:`BatchSimulationResults` with ``outputs[name].shape[0] == N``.

    Raises:
        ValueError: Inconsistent batch sizes, missing options, or invalid backend.
        TypeError: ``diagram`` is not a :class:`~jaxonomy.framework.diagram.Diagram`.
    """
    # Error-message remapping is handled by the @remap_simulation_errors
    # decorator on simulate_batch itself.
    if not isinstance(diagram, Diagram):
        raise TypeError(f"simulate_batch expects a Diagram, got {type(diagram)}")
    if recorded_signals is None:
        raise ValueError("simulate_batch requires recorded_signals (same as simulate).")
    if options is None:
        raise ValueError(
            "simulate_batch requires SimulatorOptions with math_backend='jax' and "
            "max_major_steps set."
        )
    if options.max_major_steps is None or options.max_major_steps <= 0:
        raise ValueError(
            "simulate_batch requires options.max_major_steps to be set to a positive int."
        )
    if options.math_backend != "jax":
        raise ValueError(
            f"simulate_batch only supports math_backend='jax', got {options.math_backend!r}."
        )

    n = _infer_batch_size(param_batches)
    opts = dataclasses.replace(options, enable_autodiff=False)

    vmap_safe = _is_vmap_safe(diagram)

    if use_vmap and not vmap_safe:
        raise ValueError(
            "use_vmap=True requires a pure-JAX diagram (no CustomPythonBlock / FMU). "
            "The diagram contains non-traceable blocks. Use use_vmap=False or the "
            "loop path."
        )

    # T-019-followup (2026-07-10): the CPU+small-batch UserWarning that
    # used to fire here was removed together with the per-row host-loop
    # finalize it warned about — the finalize is now vectorised and the
    # vmap path is within ~25% of the kernel path on CPU at N=1000
    # (0.41 s vs 0.33 s on the reference damped-oscillator sweep).

    use_kernel = vmap_safe and not _force_loop

    # T-110-followup-attach-on-batch: capture provenance ONCE at the start
    # of the batch (shared across all replicas — the only thing that
    # differs per-replica is the parameter dict, which is already in
    # ``param_batches``).  Captured before dispatch so the manifest's
    # timestamp records when the batch began; default-off path stays
    # byte-equivalent because ``compute_provenance`` is never called when
    # ``record_provenance=False``.
    provenance_manifest: ProvenanceManifest | None = None
    if getattr(options, "record_provenance", False):
        provenance_manifest = compute_provenance(diagram, options)

    if use_kernel:
        result = _simulate_batch_kernel(
            diagram, t_span, param_batches, opts, recorded_signals,
            results_options, use_vmap, n, lazy=lazy,
        )
    else:
        result = _simulate_batch_loop(
            diagram, t_span, param_batches, opts, recorded_signals,
            results_options, n,
        )

    if provenance_manifest is not None:
        result.provenance = provenance_manifest
    return result


# ---------------------------------------------------------------------------
# Kernel path implementation
# ---------------------------------------------------------------------------

def _simulate_batch_kernel(
    diagram, t_span, param_batches, opts, recorded_signals,
    results_options, use_vmap, n, lazy=False,
):
    """Execute the batch using a single JIT-compiled kernel (or vmap)."""
    from ..backend import set_backend
    set_backend("jax")

    opts_k = _check_options(diagram, opts, t_span, recorded_signals)

    # Build one base context for the original diagram
    base_ctx = diagram.create_context()

    # Stacked parameter arrays: {path: jnp.array shape (N, ...)}
    stacked = {path: jnp.asarray(arr) for path, arr in param_batches.items()}

    if use_vmap:
        return _vmap_path(
            diagram, base_ctx, t_span, opts_k, recorded_signals,
            stacked, n, lazy=lazy,
        )
    else:
        return _scan_kernel_path(
            diagram, base_ctx, t_span, opts_k, recorded_signals,
            stacked, n, lazy=lazy,
        )


def _scan_kernel_path(
    diagram, base_ctx, t_span, opts, recorded_signals, stacked, n, lazy=False,
):
    """Single JIT compilation; run N times sequentially with different contexts."""
    from .simulator import Simulator
    from ..backend import ODESolver, set_backend

    set_backend("jax")

    # Build simulator ONCE
    ode_solver = ODESolver(diagram, options=opts.ode_options)
    sim = Simulator(diagram, ode_solver=ode_solver, options=opts)

    t0 = float(t_span[0])
    tf = float(t_span[1])

    @jax.jit
    def _kernel(context):
        """JIT-compiled kernel: accepts context, returns SimulatorState."""
        return sim.advance_to(tf, context.with_time(t0))

    # Extract output-port information ONCE
    signal_port_indices = {
        sig_name: (port.system.system_id, port.index)
        for sig_name, port in recorded_signals.items()
    }

    # Overflow (T-138 decimation) is detected host-side after each
    # kernel call; warn at most once per batch to avoid N repeats.
    _overflow_warned = False

    if lazy:
        time_list = []
        out_lists = {k: [] for k in recorded_signals}
        for i in range(n):
            updates = {path: jnp.asarray(stacked[path][i]) for path in stacked}
            ctx_i = _pure_patch_context(base_ctx, updates)
            sim_state = _kernel(ctx_i)
            if not _overflow_warned:
                _overflow_warned = maybe_warn_recording_truncation(
                    sim_state.results_data, opts.buffer_length,
                )
            time_list.append(sim_state.results_data.time)
            for sig_name in recorded_signals:
                out_lists[sig_name].append(sim_state.results_data.outputs[sig_name])

        stacked_time = jnp.stack(time_list, axis=0)
        stacked_out = {k: jnp.stack(vs, axis=0) for k, vs in out_lists.items()}
        return BatchSimulationResults(time=stacked_time, outputs=stacked_out, used_vmap=False)

    time_ref = None
    out_lists: dict[str, list] = {k: [] for k in recorded_signals}

    # T-017c: ``_trim`` returns numpy arrays; resample on the host with
    # ``numpy.interp`` (no recompile per distinct trim length).  At N=100
    # this drops ~1 s off wall clock vs. ``jnp.interp`` which re-jits per
    # shape combination.
    for i in range(n):
        updates = {path: jnp.asarray(stacked[path][i]) for path in stacked}
        ctx_i = _pure_patch_context(base_ctx, updates)

        sim_state = _kernel(ctx_i)
        results_data = sim_state.results_data
        if not _overflow_warned:
            _overflow_warned = maybe_warn_recording_truncation(
                results_data, opts.buffer_length,
            )
        time_i, outputs_i = results_data.finalize()

        if time_ref is None:
            time_ref = time_i
            for sig_name in recorded_signals:
                out_lists[sig_name].append(np.asarray(outputs_i[sig_name]))
        else:
            t_ref_end = float(time_ref[-1])
            t_run_end = float(time_i[-1])
            if abs(t_run_end - t_ref_end) / max(abs(t_ref_end), 1e-10) > 0.01:
                warnings.warn(
                    f"simulate_batch: run {i} ended at t={t_run_end:.4g} but reference "
                    f"run ended at t={t_ref_end:.4g}. Outputs will be interpolated "
                    "(clamped) to fill the time grid.",
                    UserWarning,
                    stacklevel=3,
                )
            same_grid = (
                time_i.shape == time_ref.shape and np.array_equal(time_i, time_ref)
            )
            for sig_name in recorded_signals:
                if same_grid:
                    out_lists[sig_name].append(np.asarray(outputs_i[sig_name]))
                else:
                    out_lists[sig_name].append(
                        _interp_on_time_np(outputs_i[sig_name], time_i, time_ref)
                    )

    stacked_out = {k: np.stack(vs, axis=0) for k, vs in out_lists.items()}
    return BatchSimulationResults(time=time_ref, outputs=stacked_out, used_vmap=False)


def _vmap_path(
    diagram, base_ctx, t_span, opts, recorded_signals, stacked, n, lazy=False,
):
    """True vmap: vectorise the simulation over the batch dimension.

    Creates a batched context (all arrays in the context have a leading batch
    dimension ``N``) then calls ``jax.vmap(kernel)(batched_ctx)`` so that XLA
    executes all N simulations in a single vectorised kernel.
    """
    from .simulator import Simulator
    from ..backend import ODESolver, set_backend

    set_backend("jax")

    ode_solver = ODESolver(diagram, options=opts.ode_options)
    sim = Simulator(diagram, ode_solver=ode_solver, options=opts)

    t0 = float(t_span[0])
    tf = float(t_span[1])

    def _kernel_single(context):
        """Unbatched kernel used as vmap target."""
        return sim.advance_to(tf, context.with_time(t0))

    # Build a batched context by broadcasting all arrays to shape (N, ...) and
    # then overwriting the batch-varied parameters with their true batch values.
    batched_ctx = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (n,) + jnp.asarray(x).shape),
        base_ctx,
    )

    # Inject the batch parameter arrays (they already have shape (N, ...))
    # We cannot use _pure_patch_context directly on the batched ctx because
    # the structure must stay consistent.  Instead we manually place each
    # batched array into the correct slot.
    batched_updates: dict[str, Any] = {}
    for path, arr in stacked.items():
        batched_updates[path] = arr  # shape (N, ...) — already has batch dim

    batched_ctx = _pure_patch_context(batched_ctx, batched_updates)

    # vmap over axis-0 of all arrays in batched_ctx.
    # ``axis_name="batch"`` is required so that stochastic-source blocks
    # opting into ``fold_in_batch_index=True`` (T-122-followup-vmap-fold-in)
    # can call ``jax.lax.axis_index("batch")`` inside their per-step update
    # to derive a per-replica independent PRNG stream.  The name is a no-op
    # for blocks that don't use it.
    batch_sim_states = jax.vmap(_kernel_single, axis_name="batch")(batched_ctx)

    # Finalize: the results_data has shape (N, T, ...) due to vmap
    results_data_batch = batch_sim_states.results_data

    # Post-vmap, host-side overflow check.  ``record_stride`` is a
    # batched (N,) array here; any entry > 1 means that run's recording
    # buffer filled and was decimated (T-138) — warn once for the batch.
    maybe_warn_recording_truncation(results_data_batch, opts.buffer_length)

    if lazy:
        return BatchSimulationResults(
            time=results_data_batch.time,
            outputs=results_data_batch.outputs,
            used_vmap=True,
        )

    # T-019-followup-batched-vmap-finalize: the finalize is fully
    # vectorised — one D2H transfer per array, then batched numpy ops.
    # The previous per-row Python loop (trim + per-column np.interp per
    # element) cost O(N) host time and reversed the vmap advantage on
    # CPU at moderate N.
    np_time = np.asarray(results_data_batch.time)
    np_outputs = {k: np.asarray(results_data_batch.outputs[k]) for k in recorded_signals}

    valid = np.isfinite(np_time)  # (N, T)
    valid0 = valid[0]
    time_ref = np_time[0][valid0]

    # Fast path: every row recorded the identical grid (fixed-step
    # solvers, or adaptive runs that happened to step identically).
    # Pure slicing — bit-equivalent to the old loop's same_grid branch.
    same_grid = bool((valid == valid0).all()) and bool(
        (np_time[:, valid0] == time_ref).all()
    )
    if same_grid:
        stacked_out = {k: v[:, valid0] for k, v in np_outputs.items()}
        return BatchSimulationResults(
            time=time_ref, outputs=stacked_out, used_vmap=True
        )

    counts = valid.sum(axis=1)
    if int(counts.min()) < 2 or time_ref.shape[0] < 1:
        # Degenerate rows (a single recorded sample) can't be linearly
        # resampled by the batched kernel; keep the safe per-row loop.
        out_lists: dict[str, list] = {k: [] for k in recorded_signals}
        for i in range(n):
            valid_i = valid[i]
            time_i = np_time[i][valid_i]
            for sig_name in recorded_signals:
                val_i = np_outputs[sig_name][i][valid_i]
                if time_i.shape == time_ref.shape and np.array_equal(
                    time_i, time_ref
                ):
                    out_lists[sig_name].append(val_i)
                else:
                    out_lists[sig_name].append(
                        _interp_on_time_np(val_i, time_i, time_ref)
                    )
        stacked_out = {k: np.stack(vs, axis=0) for k, vs in out_lists.items()}
        return BatchSimulationResults(
            time=time_ref, outputs=stacked_out, used_vmap=True
        )

    # General (ragged) path: batched binary search + linear interpolation,
    # vectorised over rows, query points, and trailing signal dims.
    # NOTE the recorder guarantees each row's valid samples occupy a
    # sorted prefix with an inf tail, which is exactly the layout
    # _batched_searchsorted requires.
    stacked_out = {
        k: _batched_interp_rows(np_time, v, counts, time_ref)
        for k, v in np_outputs.items()
    }
    return BatchSimulationResults(time=time_ref, outputs=stacked_out, used_vmap=True)


# ---------------------------------------------------------------------------
# Loop path implementation (original, always-safe fallback)
# ---------------------------------------------------------------------------

def _simulate_batch_loop(
    diagram, t_span, param_batches, opts, recorded_signals, results_options, n,
):
    """Execute the batch using the safe Python loop (one simulate() call per element)."""
    time_ref = None
    out_lists: dict[str, list] = {k: [] for k in recorded_signals}

    # T-110-followup-attach-on-batch: per-replica provenance gathering
    # would defeat the "one manifest per batch" contract and add wall-time
    # overhead.  ``simulate_batch`` already captures a shared manifest at
    # batch-start (see top-level dispatch), so we explicitly suppress
    # per-replica capture inside the loop.
    if getattr(opts, "record_provenance", False):
        opts = dataclasses.replace(opts, record_provenance=False)

    for i in range(n):
        updates = {path: jnp.asarray(arr[i]) for path, arr in param_batches.items()}
        d = diagram.with_parameters(updates)
        ctx = d.create_context()
        sig = _remap_recorded_signals(diagram, d, recorded_signals)
        opts_i = _check_options(d, opts, t_span, sig)
        res = simulate(
            d,
            ctx,
            t_span,
            options=opts_i,
            results_options=results_options,
            recorded_signals=sig,
        )
        if res.outputs is None:
            raise RuntimeError("simulate_batch: simulate returned no outputs")
        if time_ref is None:
            time_ref = res.time
            for name in recorded_signals:
                out_lists[name].append(np.asarray(res.outputs[name]))
        else:
            t_ref_end = float(time_ref[-1])
            t_run_end = float(res.time[-1])
            if abs(t_run_end - t_ref_end) / max(abs(t_ref_end), 1e-10) > 0.01:
                warnings.warn(
                    f"simulate_batch: run {i} ended at t={t_run_end:.4g} but reference "
                    f"run ended at t={t_ref_end:.4g}. Outputs will be extrapolated "
                    "(clamped) to fill the time grid.",
                    UserWarning,
                    stacklevel=3,
                )
            same_grid = (
                np.asarray(res.time).shape == np.asarray(time_ref).shape
                and np.array_equal(np.asarray(res.time), np.asarray(time_ref))
            )
            for name in recorded_signals:
                if same_grid:
                    out_lists[name].append(np.asarray(res.outputs[name]))
                else:
                    out_lists[name].append(
                        _interp_on_time_np(res.outputs[name], res.time, time_ref)
                    )

    stacked = {k: np.stack(vs, axis=0) for k, vs in out_lists.items()}
    return BatchSimulationResults(time=time_ref, outputs=stacked, used_vmap=False)
