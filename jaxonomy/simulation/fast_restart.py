# SPDX-License-Identifier: MIT

"""T-112 Phase 1 — Fast Restart (warm-start single-simulation loop).

The classic "Fast Restart" feature in established block-diagram tools lets
a user re-run a model many times with different parameters or initial
conditions while paying the structural / compile cost only once.
Jaxonomy's :func:`simulate_batch` already exhibits
this behaviour for *batched* sweeps (build the :class:`Simulator` once, JIT
:meth:`Simulator.advance_to` once, then re-execute the same compiled XLA
program with different contexts).  T-112 exposes the same kernel-reuse
pattern as a small ergonomic API for the *single-simulation* loop case —
the typical UX for parameter tuning, sensitivity analysis, and design
exploration.

Phase 1 ships :class:`FastRestartSimulator`, a context-manager-friendly
wrapper that holds:

* a built :class:`~jaxonomy.simulation.simulator.Simulator` and its
  :class:`~jaxonomy.backend.ODESolver`;
* a base :class:`~jaxonomy.framework.context.ContextBase` cloned per run;
* the user's recorded-signal port dict (resolved once);
* a reference to the JIT-compiled ``advance_to`` kernel.

Usage::

    with FastRestartSimulator(diagram, options=opts,
                              recorded_signals={"y": port}) as sim:
        for k in jnp.linspace(0.5, 2.0, 16):
            r = sim.run(parameters={"gain.gain": k})
            ...

The first :meth:`run` pays the JIT-compile cost; subsequent calls hit the
JAX persistent cache and are markedly faster.  Numerical results are
identical to :func:`simulate` on the same parameters (the JIT cache key
covers the abstract pytree structure, not the concrete values).

This module is purely additive — :func:`simulate` is unchanged and the
default-off path is byte-equivalent.

T-112-followup-stateful-simulator (this module) layers three opt-in
ergonomics on top of phase 1:

* ``run(initial_state=...)`` overrides the per-run continuous state
  without rebuilding the kernel (shape/dtype must match the diagram's
  default state — otherwise the JIT cache misses and we warn).
* ``reset(diagram=...)`` rebinds to a new diagram and forces a recompile
  on the next :meth:`run`; ``reset()`` with no args just clears the
  cached kernel (e.g. to free memory).
* Structural-change warning: when a :meth:`run` invocation would force
  a JIT recompile (parameter pytree shape changed, ``initial_state``
  shape disagrees with the cached one, etc.), emit a one-time
  ``UserWarning`` so users aren't silently confused about why a "fast
  restart" call is slow.

T-112-followup-multi-system layers a pool-of-diagrams cache on top:

* ``run_with_diagram(diagram, parameters=..., initial_state=...)`` keys
  the compiled kernel by ``id(diagram)`` so users holding a
  ``dict[str, Diagram]`` (e.g. controller-variant pool) can rapidly
  switch between them.  Each distinct Diagram instance compiles once
  and is reused thereafter; the user's :meth:`run` / :meth:`reset` APIs
  are byte-equivalent to before.

  *Limitation:* the cache is keyed on Diagram identity.  Calling
  ``diagram.with_config(...)`` produces a *new* Diagram object, so a
  ``with_config``-rewritten derivative will miss the cache.  A
  ``cache_key=`` user-supplied identifier is a natural future extension
  for that case.

T-112-followup-warm-batch layers a vmap'd batched-parameter-sweep API
on top of the same cached kernel:

* ``run_batch(parameters_batch, initial_states_batch=None)`` accepts
  a ``{path: (N, ...) array}`` mapping (same convention as
  :func:`simulate_batch`'s ``param_batches``) and runs ``N``
  simulations in a single ``jax.vmap`` over the warm-cached kernel.
  This is the "batch" counterpart to :meth:`run` and the
  warm-start counterpart to :func:`simulate_batch` — the kernel is
  built/compiled at most once across both calls.

  Numerical results for a single-row batch
  (``{"K": jnp.array([1.0])}``) match a scalar :meth:`run` call
  (``parameters={"K": 1.0}``) within tolerance.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Iterator, Sequence

import jax
import jax.numpy as jnp

from ..framework.diagram import Diagram
from ..framework.port import OutputPort
from .batch import _pure_patch_context
from .errors import remap_simulation_errors
from .simulator import _check_options
from .types import SimulationResults, SimulatorOptions

__all__ = [
    "FastRestartSimulator",
    "fast_restart",
]


class FastRestartSimulator:
    """Stateful single-simulation runner that reuses one JIT-compiled kernel.

    The simulator is built lazily on the first :meth:`run` so that the
    ``recorded_signals`` set passed to the constructor (which selects the
    set of recorded ports baked into the kernel) is locked in before
    compilation.  Subsequent :meth:`run` calls reuse the same compiled
    XLA program — only the parameter pytree changes.

    Args:
        system: A :class:`~jaxonomy.framework.diagram.Diagram` (or any
            :class:`~jaxonomy.framework.system_base.SystemBase`).  The
            structural shape (block topology, port shapes/dtypes,
            parameter pytree shape) must remain constant across
            :meth:`run` calls — only parameter *values* may change.
        t_span: ``(t_start, t_stop)``.  Locked in on construction since
            it affects the auto-estimated ``max_major_steps`` and the
            recorder buffer length.  A ``run(t_span=...)`` override is
            deferred to a follow-up.
        options: :class:`SimulatorOptions`.  ``math_backend="jax"`` and
            ``enable_tracing=True`` (the defaults) are required to get
            any warm-start benefit; the JIT cache is what makes
            subsequent calls fast.
        recorded_signals: Mapping ``signal_name -> OutputPort`` (same
            convention as :func:`simulate`).  Required so the recorder
            buffer shape is fixed at construction.

    The context-manager protocol is supported but not strictly required;
    use ``with FastRestartSimulator(...) as sim: ...`` for symmetry with
    other resource-holding APIs (the ``__exit__`` clears the JIT cache
    reference, freeing the compiled kernel).
    """

    def __init__(
        self,
        system,
        t_span: tuple[float, float],
        options: SimulatorOptions | None = None,
        recorded_signals: dict[str, OutputPort] | None = None,
    ):
        if recorded_signals is None or not recorded_signals:
            raise ValueError(
                "FastRestartSimulator requires a non-empty recorded_signals "
                "dict (same convention as simulate()).  The set of recorded "
                "signals must be fixed at construction so the kernel buffer "
                "shape can be locked in before JIT compilation."
            )
        if options is None:
            options = SimulatorOptions()
        if options.math_backend != "jax":
            raise ValueError(
                "FastRestartSimulator requires math_backend='jax' to benefit "
                "from JIT cache reuse; got "
                f"math_backend={options.math_backend!r}."
            )
        if not options.enable_tracing:
            raise ValueError(
                "FastRestartSimulator requires enable_tracing=True (the "
                "default) to JIT-compile the kernel; got "
                "enable_tracing=False."
            )

        self.system = system
        self.t_span = (float(t_span[0]), float(t_span[1]))
        self._user_options = options
        self.recorded_signals = dict(recorded_signals)

        # Lazy init — set on first ``run`` so we pay the build cost
        # exactly once, after the user has finalised their setup.
        self._sim = None
        self._base_ctx = None
        self._opts_resolved: SimulatorOptions | None = None
        self._kernel = None

        # Cached abstract-pytree signature of the context the kernel
        # was last driven with — used to emit a one-time warning when
        # a :meth:`run` invocation will force a JIT recompile.  The
        # JIT cache key is the context's treedef + leaf shapes/dtypes;
        # mismatched signature ⇒ recompile.  ``None`` until the
        # kernel has been driven at least once.
        self._cached_ctx_sig: tuple | None = None

        # Per-diagram-identity kernel cache for
        # :meth:`run_with_diagram` (T-112-followup-multi-system).
        # Keyed on ``id(diagram)``; each entry holds the resolved
        # simulator bundle for that diagram.  We also hold strong
        # references to the cached Diagram objects in
        # ``_diagram_pool`` so their ``id``s remain valid for the
        # lifetime of the FastRestartSimulator (Python may otherwise
        # reuse the id of a garbage-collected object).
        self._diagram_kernel_cache: dict[int, dict[str, Any]] = {}
        self._diagram_pool: dict[int, Diagram] = {}

        # Track call counts for diagnostics / tests.
        self.n_runs: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "FastRestartSimulator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Drop references to the compiled kernel and base context.

        Subsequent :meth:`run` calls will rebuild and recompile.  The
        underlying JAX persistent cache (T-017) still holds the compiled
        XLA program, so the second build remains fast.

        The per-diagram-identity kernel cache used by
        :meth:`run_with_diagram` is also cleared.
        """
        self._kernel = None
        self._base_ctx = None
        self._sim = None
        self._opts_resolved = None
        self._cached_ctx_sig = None
        self._diagram_kernel_cache.clear()
        self._diagram_pool.clear()

    def reset(self, diagram: Diagram | None = None) -> None:
        """Clear the cached compiled kernel; optionally rebind to a new diagram.

        When ``diagram`` is ``None`` (default) this is equivalent to
        :meth:`close` — the next :meth:`run` rebuilds the simulator and
        recompiles the kernel.  The JAX persistent cache typically makes
        this a fast operation if the diagram structure is unchanged.

        When ``diagram`` is provided, the simulator rebinds to it.  This
        is the "swap subsystem variant" path: a parameter sweep where the
        *structure* (block topology, port shapes, parameter pytree
        layout) varies between runs.  The next :meth:`run` will perform
        a full recompile against the new diagram.

        Args:
            diagram: Optional new diagram (or any
                :class:`~jaxonomy.framework.system_base.SystemBase`) to
                bind the simulator to.  ``None`` means "keep the current
                one — just drop the cached kernel".
        """
        if diagram is not None:
            self.system = diagram
        self.close()

    # ------------------------------------------------------------------
    # First-run build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Lazily build the Simulator, ODE solver, and JIT kernel."""
        # Local imports to avoid circular module dependencies at import time.
        import jax

        from ..backend import ODESolver, set_backend
        from .simulator import Simulator

        set_backend("jax")

        opts = _check_options(
            self.system,
            self._user_options,
            self.t_span,
            self.recorded_signals,
        )
        self._opts_resolved = opts

        ode_solver = ODESolver(self.system, options=opts.ode_options)
        sim = Simulator(self.system, ode_solver=ode_solver, options=opts)
        self._sim = sim

        self._base_ctx = self.system.create_context()

        t0, tf = self.t_span

        @jax.jit
        def _kernel(context):
            return sim.advance_to(tf, context.with_time(t0))

        self._kernel = _kernel

    # ------------------------------------------------------------------
    # Per-run execution
    # ------------------------------------------------------------------

    def run(
        self,
        parameters: dict[str, Any] | None = None,
        initial_state: Any | None = None,
    ) -> SimulationResults:
        """Run one simulation, optionally patching parameters / initial state first.

        The first call builds the simulator and JIT-compiles the
        kernel.  Subsequent calls reuse the same compiled program;
        only the parameter and initial-state values change.

        Args:
            parameters: Optional dot-path mapping ``{"block.param":
                value, ...}`` — same convention as
                :func:`simulate_batch`'s ``param_batches`` (without the
                leading batch axis).  Values must have the same shape /
                dtype as the parameters they replace; otherwise the JIT
                cache will miss and a recompile will occur (with a
                ``UserWarning``).  Pass ``None`` (the default) to run
                with the base context unchanged.
            initial_state: Optional override for the simulator's
                continuous state at ``t = t_span[0]``.  For a
                :class:`~jaxonomy.framework.context.LeafContext`-rooted
                system, pass a single array.  For a multi-block
                :class:`~jaxonomy.framework.diagram.Diagram` with more
                than one continuous-state block, pass a sequence of
                arrays in the order returned by ``ctx.continuous_state``
                (one per continuous-state subcontext).  As a convenience,
                a single array is auto-wrapped into a single-element
                list when the diagram has exactly one continuous-state
                block.  Shape/dtype must match the diagram's default
                continuous state — otherwise the JIT cache will miss
                and a recompile will occur (with a ``UserWarning``).

        Returns:
            A :class:`SimulationResults` populated with ``time``,
            ``outputs``, and (when
            ``options.return_context=True``) ``context``.
        """
        if self._kernel is None:
            self._build()

        # Patch the base context with the per-run parameter overrides.
        ctx = self._base_ctx
        if parameters:
            updates = {k: jnp.asarray(v) for k, v in parameters.items()}
            ctx = _pure_patch_context(ctx, updates)

        # Apply the initial-state override (if any) on top of the
        # parameter patch.  We do this *after* the parameter patch so
        # that user-supplied initial state always wins regardless of
        # whether the diagram exposed the IC as a parameter as well.
        if initial_state is not None:
            ctx = self._apply_initial_state(ctx, initial_state)

        # Emit a one-time structural-change warning if this run will
        # force a JIT recompile.  Compute the signature *after* all
        # context patches are applied so we compare against what the
        # kernel actually sees.
        self._maybe_warn_structural_change(ctx)

        # Drive the JIT-compiled kernel.
        sim_state = self._kernel(ctx)
        results_data = sim_state.results_data
        time, outputs = results_data.finalize()

        final_context = (
            sim_state.context if self._opts_resolved.return_context else None
        )

        self.n_runs += 1

        return SimulationResults(
            final_context,
            time=time,
            outputs=outputs,
            parameters=dict(parameters) if parameters else None,
        )

    # ------------------------------------------------------------------
    # Batched parameter sweep over the warm-cached kernel
    # ------------------------------------------------------------------

    def run_batch(
        self,
        parameters_batch: dict[str, Any],
        initial_states_batch: Any | None = None,
    ) -> "BatchSimulationResults":
        """Run ``N`` simulations differing only by parameters, vmap'd over the cached kernel.

        Counterpart to :meth:`run` for batched parameter sweeps.  Equivalent
        to :func:`simulate_batch` with ``use_vmap=True`` but reuses the
        warm-cached kernel built by the most recent :meth:`run` call (or
        builds it lazily on first use).  Calling :meth:`run` first to warm
        the cache and then :meth:`run_batch` for the sweep is the typical
        UX pattern; both code paths share one JIT compile.

        Args:
            parameters_batch: ``{path: (N, ...) array}`` — same convention
                as :func:`simulate_batch`'s ``param_batches``.  Every value
                must have the same leading batch size ``N``.
            initial_states_batch: Optional batched initial-state override.
                Either a single array of shape ``(N, ...)`` (auto-wrapped
                for diagrams with a single continuous-state block) or a
                list/tuple of ``(N, ...)`` arrays (one per continuous-state
                block, matching ``ctx.continuous_state`` ordering).
                Default ``None`` reuses the diagram's default IC for every
                batch element.

        Returns:
            A :class:`BatchSimulationResults` with
            ``outputs[name].shape[0] == N``.

        Notes:
            * The kernel is vmap'd over all leaves of the patched context,
              not just the explicitly-batched parameter paths.  Unpatched
              leaves are broadcast to shape ``(N, ...)`` so the vmap'd
              kernel sees a uniformly batched pytree.
            * The cached kernel was JIT'd against a *scalar* context
              signature.  ``jax.vmap`` traces against the batched
              signature, so the very first call to :meth:`run_batch`
              incurs one extra trace (still cheap; XLA caches the inner
              compiled program).  Subsequent :meth:`run_batch` calls with
              the same ``N`` and the same parameter pytree shape reuse
              the vmap-cached kernel.
        """
        from .batch import BatchSimulationResults, _infer_batch_size

        if not parameters_batch:
            raise ValueError(
                "FastRestartSimulator.run_batch: parameters_batch must be "
                "non-empty.  For a single warm-restart simulation pass "
                "parameters=... to .run() instead."
            )

        # Ensure the kernel + base context are built (lazy on first call).
        if self._kernel is None:
            self._build()

        n = _infer_batch_size(parameters_batch)
        stacked = {path: jnp.asarray(arr) for path, arr in parameters_batch.items()}

        # Build a batched context: broadcast every leaf of the base ctx to
        # shape (N, ...), then patch in the per-batch parameter values
        # (which already have the leading N axis).
        batched_ctx = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x)[None], (n,) + jnp.asarray(x).shape
            ),
            self._base_ctx,
        )
        batched_ctx = _pure_patch_context(batched_ctx, stacked)

        # Optional batched initial-state override.  Apply *after* the
        # parameter patch so the user's IC always wins.
        if initial_states_batch is not None:
            batched_ctx = self._apply_batched_initial_state(
                batched_ctx, initial_states_batch, n,
            )

        # vmap the cached kernel.  ``axis_name="batch"`` mirrors
        # :func:`simulate_batch`'s vmap path so any
        # ``fold_in_batch_index=True`` blocks (T-122-followup) work
        # identically here.
        batched_kernel = jax.vmap(self._kernel, axis_name="batch")
        batch_sim_states = batched_kernel(batched_ctx)
        results_data_batch = batch_sim_states.results_data

        # Finalize each batch row (results_data carries variable-length
        # buffers; finalize must run per-element).  Mirror the
        # numpy-host interpolation pattern from
        # :func:`simulate_batch._vmap_path` so output shapes line up
        # across rows with different step counts.
        import warnings as _warnings

        import numpy as _np

        from .batch import _interp_on_time_np

        time_ref = None
        out_lists: dict[str, list] = {k: [] for k in self.recorded_signals}

        for i in range(n):
            rd_i = jax.tree_util.tree_map(lambda x: x[i], results_data_batch)
            time_i, outputs_i = rd_i.finalize()
            if time_ref is None:
                time_ref = time_i
                for sig_name in self.recorded_signals:
                    out_lists[sig_name].append(_np.asarray(outputs_i[sig_name]))
                continue

            t_ref_end = float(time_ref[-1])
            t_run_end = float(time_i[-1])
            if abs(t_run_end - t_ref_end) / max(abs(t_ref_end), 1e-10) > 0.01:
                _warnings.warn(
                    f"FastRestartSimulator.run_batch: row {i} ended at "
                    f"t={t_run_end:.4g} but reference row ended at "
                    f"t={t_ref_end:.4g}.  Outputs will be interpolated "
                    "(clamped) to fill the time grid.",
                    UserWarning,
                    stacklevel=3,
                )
            same_grid = (
                time_i.shape == time_ref.shape
                and _np.array_equal(time_i, time_ref)
            )
            for sig_name in self.recorded_signals:
                if same_grid:
                    out_lists[sig_name].append(_np.asarray(outputs_i[sig_name]))
                else:
                    out_lists[sig_name].append(
                        _interp_on_time_np(outputs_i[sig_name], time_i, time_ref)
                    )

        stacked_out = {k: _np.stack(vs, axis=0) for k, vs in out_lists.items()}
        self.n_runs += n
        return BatchSimulationResults(
            time=time_ref, outputs=stacked_out, used_vmap=True,
        )

    def _apply_batched_initial_state(self, batched_ctx, initial_states_batch, n: int):
        """Apply a batched IC override onto a context already broadcast to ``(N, ...)``.

        Mirrors :meth:`_apply_initial_state` but expects per-element
        arrays to carry the leading batch axis ``N``.  The
        ``with_continuous_state`` API accepts arrays whose shape may be
        ``(N, ...)`` as long as the tree structure matches the context's
        continuous-state slot, so we delegate to it.
        """
        from ..framework.context import DiagramContext, LeafContext

        if isinstance(batched_ctx, LeafContext):
            xc_new = jnp.asarray(initial_states_batch)
            if xc_new.shape[0] != n:
                raise ValueError(
                    "FastRestartSimulator.run_batch(initial_states_batch=...): "
                    f"expected leading batch size {n}, got shape {xc_new.shape}."
                )
            return batched_ctx.with_continuous_state(xc_new)

        if isinstance(batched_ctx, DiagramContext):
            n_xc = len(batched_ctx.continuous_subcontexts)
            if n_xc == 0:
                raise ValueError(
                    "FastRestartSimulator.run_batch(initial_states_batch=...): "
                    "diagram has no continuous-state blocks; nothing to override."
                )
            if isinstance(initial_states_batch, (list, tuple)):
                xs = [jnp.asarray(x) for x in initial_states_batch]
            else:
                if n_xc != 1:
                    raise ValueError(
                        "FastRestartSimulator.run_batch(initial_states_batch=...): "
                        f"diagram has {n_xc} continuous-state blocks but a "
                        "single array was passed; pass a list/tuple of "
                        "arrays matching ctx.continuous_state ordering."
                    )
                xs = [jnp.asarray(initial_states_batch)]
            if len(xs) != n_xc:
                raise ValueError(
                    "FastRestartSimulator.run_batch(initial_states_batch=...): "
                    f"expected {n_xc} arrays (one per continuous-state "
                    f"block); got {len(xs)}."
                )
            for k, x in enumerate(xs):
                if x.shape[0] != n:
                    raise ValueError(
                        "FastRestartSimulator.run_batch(initial_states_batch=...): "
                        f"array {k} has leading dim {x.shape[0]}, expected "
                        f"batch size {n}."
                    )
            return batched_ctx.with_continuous_state(xs)

        return batched_ctx.with_continuous_state(initial_states_batch)

    # ------------------------------------------------------------------
    # Multi-system (per-diagram-identity) kernel cache
    # ------------------------------------------------------------------

    def _build_for_diagram(
        self,
        diagram: Diagram,
        recorded_signals: dict[str, OutputPort],
    ) -> dict[str, Any]:
        """Build a Simulator + JIT kernel bundle for ``diagram``.

        Returns a dict with keys ``sim``, ``base_ctx``, ``opts_resolved``,
        ``kernel``, ``recorded_signals``, ``cached_ctx_sig`` (the latter
        is initially ``None`` and is populated on first kernel drive,
        mirroring the structural-change-warning behaviour of the
        ``self`` kernel).
        """
        import jax

        from ..backend import ODESolver, set_backend
        from .simulator import Simulator

        set_backend("jax")

        opts = _check_options(
            diagram,
            self._user_options,
            self.t_span,
            recorded_signals,
        )

        ode_solver = ODESolver(diagram, options=opts.ode_options)
        sim = Simulator(diagram, ode_solver=ode_solver, options=opts)
        base_ctx = diagram.create_context()

        t0, tf = self.t_span

        @jax.jit
        def _kernel(context):
            return sim.advance_to(tf, context.with_time(t0))

        return {
            "sim": sim,
            "base_ctx": base_ctx,
            "opts_resolved": opts,
            "kernel": _kernel,
            "recorded_signals": dict(recorded_signals),
            "cached_ctx_sig": None,
        }

    def run_with_diagram(
        self,
        diagram: Diagram,
        parameters: dict[str, Any] | None = None,
        initial_state: Any | None = None,
        recorded_signals: dict[str, OutputPort] | None = None,
    ) -> SimulationResults:
        """Run one simulation against ``diagram``, caching its kernel by identity.

        Use this when you hold a *pool* of structurally-different
        diagrams (e.g. controller variants) and want to rapidly switch
        between them.  The compiled kernel for each distinct Diagram
        instance is built on first use and reused thereafter — no
        recompile on cache hit.

        Compared to :meth:`reset` + :meth:`run` (which drops and
        rebuilds the kernel on every swap), :meth:`run_with_diagram`
        keeps one compiled kernel *per Diagram identity* alive, so
        toggling back and forth between N diagrams in a loop costs N
        compiles total, not one per call.

        The user's :meth:`run` and :meth:`reset` APIs are unaffected;
        this is a purely-additive surface.

        Args:
            diagram: Diagram (or any
                :class:`~jaxonomy.framework.system_base.SystemBase`) to
                simulate.  The cache is keyed on ``id(diagram)`` so a
                strong reference to the diagram is held internally for
                the lifetime of this :class:`FastRestartSimulator`.
                **Limitation:** calling ``diagram.with_config(...)``
                produces a new Diagram object — the returned object's
                ``id`` differs from the original, so a
                ``with_config``-rewritten derivative will miss the
                cache.  A ``cache_key=`` user-supplied identifier is a
                natural future extension.
            parameters: Optional dot-path mapping ``{"block.param":
                value, ...}`` — same convention as :meth:`run`.
            initial_state: Optional override for the simulator's
                continuous state at ``t = t_span[0]`` — same convention
                as :meth:`run`.
            recorded_signals: Optional ``{name: OutputPort}`` mapping.
                Required on the *first* call for a given diagram (the
                ports are diagram-specific and locked into the kernel
                buffer shape at compile time).  On warm cache hits the
                originally-cached mapping is reused; passing a
                different ``recorded_signals`` for the same cached
                diagram has no effect.  If omitted on the first call,
                falls back to ``self.recorded_signals`` — which is
                typically only valid for the diagram passed to the
                constructor.

        Returns:
            A :class:`SimulationResults` populated with ``time``,
            ``outputs``, and (when
            ``options.return_context=True``) ``context``.
        """
        cache_key = id(diagram)
        bundle = self._diagram_kernel_cache.get(cache_key)
        if bundle is None:
            # First call for this diagram — build the kernel.
            sigs = recorded_signals if recorded_signals is not None else self.recorded_signals
            if not sigs:
                raise ValueError(
                    "FastRestartSimulator.run_with_diagram(...): no "
                    "recorded_signals available for this diagram.  Pass "
                    "recorded_signals={'name': port, ...} on the first "
                    "call for each diagram in your pool — the ports are "
                    "diagram-specific and must match the diagram passed "
                    "in this call."
                )
            bundle = self._build_for_diagram(diagram, sigs)
            self._diagram_kernel_cache[cache_key] = bundle
            # Hold a strong reference so ``id(diagram)`` stays valid
            # (Python may otherwise reuse the id of a GC'd object).
            self._diagram_pool[cache_key] = diagram

        # Patch the cached base context with per-run overrides.
        ctx = bundle["base_ctx"]
        if parameters:
            updates = {k: jnp.asarray(v) for k, v in parameters.items()}
            ctx = _pure_patch_context(ctx, updates)
        if initial_state is not None:
            ctx = self._apply_initial_state(ctx, initial_state)

        # Per-diagram structural-change warning.
        self._maybe_warn_structural_change_for_bundle(bundle, ctx)

        # Drive the cached JIT kernel.
        sim_state = bundle["kernel"](ctx)
        results_data = sim_state.results_data
        time, outputs = results_data.finalize()

        final_context = (
            sim_state.context if bundle["opts_resolved"].return_context else None
        )

        self.n_runs += 1

        return SimulationResults(
            final_context,
            time=time,
            outputs=outputs,
            parameters=dict(parameters) if parameters else None,
        )

    def _maybe_warn_structural_change_for_bundle(
        self, bundle: dict[str, Any], ctx
    ) -> None:
        """Per-diagram analogue of :meth:`_maybe_warn_structural_change`."""
        ctx_sig = self._signature(ctx)
        cached = bundle["cached_ctx_sig"]
        if cached is None:
            bundle["cached_ctx_sig"] = ctx_sig
            return
        if ctx_sig != cached:
            warnings.warn(
                "FastRestartSimulator: detected a structural change in "
                "the context (parameter pytree or continuous-state "
                "shape/dtype) between runs of the same cached diagram — "
                "the JIT cache will miss and the kernel will be "
                "recompiled, defeating the fast-restart benefit.  "
                "Ensure parameter / initial_state values keep the same "
                "shapes/dtypes across runs.",
                UserWarning,
                stacklevel=3,
            )
            bundle["cached_ctx_sig"] = ctx_sig

    # ------------------------------------------------------------------
    # Initial-state override + structural-change detection
    # ------------------------------------------------------------------

    def _apply_initial_state(self, ctx, initial_state: Any):
        """Return ``ctx`` with its continuous state replaced by ``initial_state``.

        Accepts either a single array (auto-wrapped for the common
        single-block case on a :class:`DiagramContext`) or a sequence of
        arrays (one per continuous-state subcontext).  For a
        :class:`LeafContext`, a single array is passed through directly.
        """
        from ..framework.context import DiagramContext, LeafContext

        if isinstance(ctx, LeafContext):
            xc_new = jnp.asarray(initial_state)
            return ctx.with_continuous_state(xc_new)

        if isinstance(ctx, DiagramContext):
            # ``with_continuous_state`` expects one array per
            # continuous-state subcontext.
            n_xc = len(ctx.continuous_subcontexts)
            if n_xc == 0:
                raise ValueError(
                    "FastRestartSimulator.run(initial_state=...): the "
                    "diagram has no continuous-state blocks; nothing to "
                    "override."
                )
            if isinstance(initial_state, (list, tuple)):
                xs = [jnp.asarray(x) for x in initial_state]
            else:
                # Convenience: a single array for a single-block
                # diagram.
                if n_xc != 1:
                    raise ValueError(
                        "FastRestartSimulator.run(initial_state=...): "
                        "diagram has "
                        f"{n_xc} continuous-state blocks but a single "
                        "array was passed; pass a list/tuple of arrays "
                        "matching ctx.continuous_state ordering."
                    )
                xs = [jnp.asarray(initial_state)]
            if len(xs) != n_xc:
                raise ValueError(
                    "FastRestartSimulator.run(initial_state=...): "
                    f"expected {n_xc} arrays (one per continuous-state "
                    f"block); got {len(xs)}."
                )
            return ctx.with_continuous_state(xs)

        # Fall through: unknown context type — defer to the duck-typed
        # ``with_continuous_state`` and hope for the best.
        return ctx.with_continuous_state(initial_state)

    @staticmethod
    def _signature(value) -> tuple:
        """Return a hashable ``(treedef, leaf-shape/dtype tuple)`` signature.

        Two values share a signature iff a ``jax.jit`` keyed on their
        pytree structure would reuse the same compiled program.  Used to
        detect structural changes that will force a recompile.
        """
        leaves, treedef = jax.tree_util.tree_flatten(value)
        leaf_sig = tuple(
            (
                tuple(getattr(x, "shape", ())),
                str(getattr(x, "dtype", type(x).__name__)),
            )
            for x in leaves
        )
        return (str(treedef), leaf_sig)

    def _maybe_warn_structural_change(self, ctx) -> None:
        """Emit a ``UserWarning`` if this run will force a JIT recompile.

        The kernel is JIT-compiled with ``ctx`` as its single argument,
        so the cache key is the abstract signature (treedef + leaf
        shapes/dtypes) of the patched context.  If that signature
        differs from what the kernel was last driven with, JAX will
        recompile — defeating the fast-restart benefit.

        On the first invocation we just record the baseline signature
        (no warning — the first call is expected to compile).
        """
        ctx_sig = self._signature(ctx)

        if self._cached_ctx_sig is None:
            self._cached_ctx_sig = ctx_sig
            return

        if ctx_sig != self._cached_ctx_sig:
            warnings.warn(
                "FastRestartSimulator: detected a structural change in "
                "the context (parameter pytree or continuous-state "
                "shape/dtype) between runs — the JIT cache will miss "
                "and the kernel will be recompiled, defeating the "
                "fast-restart benefit.  Call sim.reset(diagram=...) "
                "and rebuild with the new structure if this is "
                "intentional, or ensure parameter / initial_state "
                "values keep the same shapes/dtypes across runs.",
                UserWarning,
                stacklevel=3,
            )
            # Update the cached signature so we only warn once per
            # structural change (not on every subsequent run with the
            # new shape).
            self._cached_ctx_sig = ctx_sig


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


@remap_simulation_errors
def fast_restart(
    system,
    t_span: tuple[float, float],
    parameter_grid: Iterable[dict[str, Any]],
    options: SimulatorOptions | None = None,
    recorded_signals: dict[str, OutputPort] | None = None,
) -> Iterator[SimulationResults]:
    """Iterate over a parameter grid, yielding one warm-started result per dict.

    Equivalent to::

        with FastRestartSimulator(system, t_span, options, recorded_signals) as sim:
            for params in parameter_grid:
                yield sim.run(parameters=params)

    Useful for the lazy / streaming case (e.g. a generator-driven optimiser
    that decides the next parameter dict based on the previous result).
    For a fixed grid known up-front, prefer :func:`simulate_batch` —
    it additionally vectorises across the batch dimension (kernel path) or
    can run on multiple devices (vmap path).

    Args:
        system: Diagram to simulate.
        t_span: ``(t_start, t_stop)``.
        parameter_grid: Iterable of ``{block.param: value}`` dicts.  The
            first dict's structure (key set + value shapes/dtypes) locks
            in the JIT cache key; later dicts must match it for warm
            reuse.
        options: :class:`SimulatorOptions` (forwarded to
            :class:`FastRestartSimulator`).
        recorded_signals: ``{name: OutputPort}`` (forwarded).

    Yields:
        :class:`SimulationResults`, one per dict.
    """
    with FastRestartSimulator(
        system,
        t_span,
        options=options,
        recorded_signals=recorded_signals,
    ) as sim:
        for params in parameter_grid:
            yield sim.run(parameters=params)
