# SPDX-License-Identifier: MIT

"""Results recording, decoupled from the Simulator class.

Encapsulates the logic for initializing and updating simulation results data,
previously inlined in ``Simulator.initialize`` and ``Simulator.save_results``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backend import ResultsData
    from ..framework import ContextBase
    from ..framework.port import OutputPort
    from ..backend.typing import Array


def maybe_warn_recording_truncation(
    results_data,
    options_buffer_length: int | None,
    caller: str = "jaxonomy.simulate_batch",
) -> bool:
    """Warn once if the recording buffer overflowed during a run.

    Host-side, post-materialization check (the same idiom as the
    T-138 check in ``simulate``: it runs *after* the JIT'd kernel
    returns, so it never disturbs jit/vmap tracing).  Detection is
    exact: the JAX backend's ``record_stride`` ends > 1 iff at least
    one T-138 buffer compaction ran, i.e. the fixed ``buffer_length``
    ring filled and the trajectory was decimated to keep spanning
    [t0, tf] at reduced resolution.

    Callers that materialize results without going through
    ``simulate`` (the batch kernel and vmap paths) use this to close
    a silent-truncation gap reported by a downstream consumer.

    Args:
        results_data: A (possibly vmapped/batched) backend
            ``ResultsData``; backends without the T-138 decimation
            fields are silently skipped.
        options_buffer_length: The user-configured
            ``SimulatorOptions.buffer_length`` (None means auto-sized).
        caller: Prefix naming the API entry point in the message.

    Returns:
        True iff a ``UserWarning`` was emitted.
    """
    import warnings

    import numpy as np

    if results_data is None:
        return False
    inner = getattr(results_data, "_solution_data", results_data)
    stride = getattr(inner, "record_stride", None)
    if stride is None:
        return False
    try:
        stride_arr = np.asarray(stride)
        max_stride = int(stride_arr.max())
        total_steps = int(np.max(np.asarray(inner.step_count)))
        n_kept = int(np.max(np.asarray(inner.buffer_index)))
    except (TypeError, ValueError):
        # Traced / abstract values (still inside a transform) — skip.
        return False
    if max_stride <= 1:
        return False

    buf_str = (
        f"buffer_length={options_buffer_length}"
        if options_buffer_length is not None
        else "buffer_length=None (auto)"
    )
    run_str = ""
    if stride_arr.ndim > 0 and stride_arr.size > 1:
        n_overflowed = int((stride_arr > 1).sum())
        run_str = f" in {n_overflowed} of {stride_arr.size} runs"
    warnings.warn(
        f"{caller}: the recording buffer ({buf_str}) filled{run_str}; "
        f"the trajectory was recorded at reduced resolution "
        f"(~{n_kept} of {total_steps} samples kept, keeping every "
        f"{max_stride}th at worst). The recorded time-series still "
        f"starts at t0 and covers the whole trajectory. Set "
        f"SimulatorOptions(buffer_length="
        f"{max(total_steps + 1, 4000)}) or larger to capture every "
        f"sample, loosen rtol/atol, or reduce the number of recorded "
        f"signals.",
        UserWarning,
        stacklevel=3,
    )
    return True


class ResultsRecorder:
    """Manages simulation results initialization and recording.

    Args:
        save_time_series: Whether results should be recorded at all.
        recorded_outputs: Dictionary of output ports to record.
        buffer_length: Size of the results buffer for JAX-traced recording.
        per_signal_buffers_classifications: T-013a-followup-mode-a-buffers
            opt-in.  When non-None, the underlying ``ResultsData`` is
            initialized with per-signal ``(times, values, count)`` buffers
            keyed by signal name; periodic signals only consume a slot on
            their cadence ticks.  When None (default), the legacy single-
            shared-time buffer is used and behaviour is byte-equivalent.
    """

    def __init__(
        self,
        save_time_series: bool,
        recorded_outputs: dict[str, "OutputPort"] | None,
        buffer_length: int | None,
        per_signal_buffers_classifications: dict[str, dict] | None = None,
        record_solver_states: bool = False,
    ):
        self.save_time_series = save_time_series
        self.recorded_outputs = recorded_outputs
        self.buffer_length = buffer_length
        self.per_signal_buffers_classifications = per_signal_buffers_classifications
        # T-012a-followup: snapshot the solver's per-step interpolant
        # data alongside ``(time, outputs)`` whenever the user opted in.
        # When ``False`` (default), ``save`` ignores any solver_state
        # passed in — byte-equivalent to the legacy hot path.
        self.record_solver_states = record_solver_states

    def _interpolant_template(self, context: "ContextBase") -> tuple[int, int] | None:
        """T-012a-followup: probe the solver's interp_coeff shape.

        Returns ``(n_coeff, n_y)`` for the JAX backend's Dopri5 path so
        the recorder can pre-allocate the per-step interpolant ring.
        Returns ``None`` for solvers that don't expose a polynomial
        interpolant (e.g. RK4 / BDF in the current implementation —
        their state lacks a fixed-shape ``interp_coeff`` attribute the
        ring can carry verbatim).  In that case the recorder falls back
        to the PCHIP marker so ``query()`` still gets a higher-order
        path.
        """
        try:
            import jax.numpy as _jnp
            xc = getattr(context, "continuous_state", None)
            if xc is None:
                return None
            # Flatten to discover n_y the same way Dopri5State.__post_init__
            # does (interp_coeff defaults to ``[y, y, y, y, y]``).
            import jax
            leaves = jax.tree_util.tree_leaves(xc)
            n_y = int(sum(int(_jnp.asarray(leaf).size) for leaf in leaves))
            if n_y == 0:
                return None
            return (5, n_y)
        except Exception:
            return None

    def initialize(self, context: "ContextBase") -> "ResultsData | None":
        """Create initial ResultsData from context, if recording is enabled.

        Args:
            context: The initial simulation context.

        Returns:
            Initialized ResultsData, or None if recording is disabled.
        """
        if not self.save_time_series:
            return None

        from ..backend import ResultsData

        interp_template = None
        if self.record_solver_states:
            interp_template = self._interpolant_template(context)

        # T-013a-followup-mode-a-buffers: if classifications were
        # supplied, forward them to the JAX backend so it can allocate
        # per-signal buffers.  The numpy backend doesn't support this
        # kwarg yet — fall back to the legacy initializer there.
        if self.per_signal_buffers_classifications is not None:
            try:
                return ResultsData.initialize(
                    context,
                    self.recorded_outputs,
                    self.buffer_length,
                    per_signal_classifications=(
                        self.per_signal_buffers_classifications
                    ),
                    interpolant_template=interp_template,
                )
            except TypeError:
                # Backend doesn't accept the kwarg — silently fall back
                # to the legacy path (caller will get post-finalize
                # Mode A behaviour instead of in-JIT Mode A).
                pass
        if interp_template is not None:
            try:
                return ResultsData.initialize(
                    context,
                    self.recorded_outputs,
                    self.buffer_length,
                    interpolant_template=interp_template,
                )
            except TypeError:
                # Backend doesn't accept ``interpolant_template`` — fall
                # back to the legacy initializer (PCHIP path then).
                pass
        return ResultsData.initialize(
            context, self.recorded_outputs, self.buffer_length
        )

    def save(
        self,
        results_data: "ResultsData | None",
        context: "ContextBase",
        ode_solver_state: object = None,
    ) -> "ResultsData | None":
        """Record a sample if time series recording is enabled.

        Args:
            results_data: The current results buffer (may be None).
            context: The current simulation context.
            ode_solver_state: Optional ODE solver state.  Passed only
                when ``record_solver_states=True``; the recorder forwards
                it to ``ResultsData.update`` so the per-step interpolant
                ring is populated.  Ignored otherwise.

        Returns:
            Updated ResultsData, or the original if recording is disabled.
        """
        if not self.save_time_series:
            return results_data
        if self.record_solver_states and ode_solver_state is not None:
            try:
                return results_data.update(context, ode_solver_state=ode_solver_state)
            except TypeError:
                # Backend doesn't accept the kwarg — drop silently.
                pass
        return results_data.update(context)

    @staticmethod
    def classify_signal_cadence(
        recorded_signals: dict[str, "OutputPort"] | None,
    ) -> dict[str, dict]:
        """T-013a Mode A: classify each recorded signal's natural cadence.

        Inspects the source ``OutputPort`` (and its owning block) to decide
        whether the signal is continuous (sampled every major step), periodic
        (sampled on a fixed period+offset schedule via a cache-update or
        state-update event), or unclassifiable ("default" — fall back to the
        Mode B value-diff dedup).

        The classification is purely structural: it reads the static
        port/event metadata once at simulator init / pre-finalize and never
        introspects runtime values.  An ``OutputPort`` carries a
        ``PeriodicEventData``-shaped ``event`` when it was declared with
        ``period=...`` (e.g. ``ZeroOrderHold``); ports without an event are
        assumed continuous if they read continuous state, default otherwise.

        Args:
            recorded_signals: Mapping of name → OutputPort (or None).

        Returns:
            Mapping of name → cadence info, where each value is one of:
              - ``{"kind": "continuous"}`` — sampled every major step.
              - ``{"kind": "periodic", "period": float, "offset": float}``
                — sampled on a schedule.
              - ``{"kind": "default"}`` — no usable cadence metadata; the
                schedule path will fall back to value-diff dedup.
        """
        classifications: dict[str, dict] = {}
        if not recorded_signals:
            return classifications
        for name, port in recorded_signals.items():
            cls = {"kind": "default"}
            try:
                event = getattr(port, "event", None)
                if event is not None:
                    event_data = getattr(event, "event_data", None)
                    period = getattr(event_data, "period", None)
                    offset = getattr(event_data, "offset", 0.0)
                    if period is not None:
                        try:
                            period_f = float(period)
                            offset_f = float(offset)
                            if period_f > 0:
                                cls = {
                                    "kind": "periodic",
                                    "period": period_f,
                                    "offset": offset_f,
                                }
                        except (TypeError, ValueError):
                            pass
                else:
                    # No event ⇒ either a continuous-state-output port or a
                    # plain feedthrough output.  We tag it as "continuous"
                    # only when the source block reports a non-trivial
                    # continuous state; everything else stays "default" so
                    # Mode B value-diff dedup still cleans constants up.
                    system = getattr(port, "system", None)
                    has_xc = False
                    if system is not None:
                        # ``_default_continuous_state`` is the LeafSystem's
                        # template for the CT state vector; non-None means
                        # the block integrates a CT state.
                        xc_template = getattr(
                            system, "_default_continuous_state", None,
                        )
                        if xc_template is not None:
                            try:
                                # Empty array → no CT state.
                                import numpy as _np
                                has_xc = _np.asarray(xc_template).size > 0
                            except Exception:
                                has_xc = True
                    if has_xc:
                        cls = {"kind": "continuous"}
            except Exception:
                # Defensive: any classification failure falls back to Mode B.
                cls = {"kind": "default"}
            classifications[name] = cls
        return classifications

    @staticmethod
    def compute_per_signal_schedule(
        time: "Array",
        outputs: dict[str, "Array"],
        classifications: dict[str, dict],
        atol: float = 1e-12,
    ) -> tuple[dict[str, "Array"], dict[str, "Array"]]:
        """T-013a Mode A: derive per-signal (times, values) using cadences.

        For each signal:
          - ``continuous`` → keep all samples (full ``time`` / full
            ``outputs[name]``).
          - ``periodic`` → keep only the indices where the recorded ``time``
            entry is an integer multiple of ``period`` (offset by
            ``offset``), within ``atol``.  Both the time vector and the
            output vector are subset, so the output array is genuinely
            shorter — that's the storage saving over Mode B.
          - ``default`` → fall back to ``compute_per_signal_times`` (value-
            diff dedup); the output vector stays at full length.  This
            matches Mode B exactly for unclassifiable signals.

        Returns ``(per_signal_times, per_signal_outputs)``.  The outputs
        dict is identical to the input for ``default``-kind signals (full
        length) and shorter for ``continuous`` (still full length —
        continuous keeps every sample) and ``periodic`` (subset to the
        schedule).
        """
        import numpy as _np

        t = _np.asarray(time)
        per_signal_times: dict[str, "Array"] = {}
        per_signal_outputs: dict[str, "Array"] = {}
        if t.ndim == 0 or t.shape[0] == 0:
            for name, arr in outputs.items():
                per_signal_times[name] = t
                per_signal_outputs[name] = _np.asarray(arr)
            return per_signal_times, per_signal_outputs

        # Pre-compute the value-diff fallback once — only used for
        # "default"-kind signals, but cheap to reuse the helper.
        diff_times = ResultsRecorder.compute_per_signal_times(
            time, outputs, atol=atol,
        )

        for name, arr in outputs.items():
            a = _np.asarray(arr)
            cls = classifications.get(name, {"kind": "default"})
            kind = cls.get("kind", "default")

            if a.shape[0] != t.shape[0]:
                # Pipeline invariant violation — keep originals.
                per_signal_times[name] = t
                per_signal_outputs[name] = a
                continue

            if kind == "continuous":
                per_signal_times[name] = t
                per_signal_outputs[name] = a
                continue

            if kind == "periodic":
                period = cls["period"]
                offset = cls.get("offset", 0.0)
                # Index where ``(t - offset)`` is an integer multiple of
                # ``period`` modulo ``atol``.  Use a tolerance scaled to
                # the period so float drift over long simulations is
                # absorbed.  ``atol`` here is a time tolerance, not a
                # value tolerance.
                shifted = t - offset
                k = _np.rint(shifted / period)
                residual = _np.abs(shifted - k * period)
                tick_tol = max(atol, 1e-9 * max(period, 1.0))
                mask = residual <= tick_tol
                # Always keep the first sample (t=0) so the array is
                # never empty even when offset > 0 and the first tick
                # hasn't fired yet — the recorded value at t=0 is the
                # initial port output.
                if mask.shape[0] > 0:
                    mask = mask.copy()
                    mask[0] = True
                per_signal_times[name] = t[mask]
                per_signal_outputs[name] = a[mask]
                continue

            # default: Mode B value-diff dedup for times; outputs stay
            # at full length so existing align() can reconstruct via
            # searchsorted.  Bit-equivalent to Mode B for this signal.
            per_signal_times[name] = diff_times.get(name, t)
            per_signal_outputs[name] = a

        return per_signal_times, per_signal_outputs

    @staticmethod
    def compute_per_signal_times(
        time: "Array",
        outputs: dict[str, "Array"],
        atol: float = 1e-12,
    ) -> dict[str, "Array"]:
        """T-013a Mode B: derive per-signal timestamp vectors by deduplication.

        The recording pipeline emits one sample per major step for every
        recorded signal — but a zero-order-hold or periodic-update output
        only *advances* on its own clock.  Mode B detects this by looking
        for indices where ``outputs[name]`` differs from its previous
        value (max-abs over leading dimensions for vector signals); the
        first sample is always retained.

        Operates on the post-``finalize`` numpy / jax arrays — i.e.
        outside any JAX trace.  No simulator-side coordination is
        required and the legacy global-vector path is untouched.

        Args:
            time: 1-D global time vector (shape ``(N,)``).
            outputs: Mapping of signal name → recorded array, with
                leading dimension ``N``.
            atol: Absolute tolerance for detecting a change between
                consecutive samples.  Defaults to ``1e-12`` — well below
                any realistic float64 round-off and above the noise floor
                of a typical simulation.

        Returns:
            ``per_signal_times`` mapping ``name -> 1-D time vector`` of
            indices where the signal's value advances.  A constant
            signal collapses to a length-1 vector ``[time[0]]``; a
            continuous signal is identical to ``time``.
        """
        import numpy as _np

        t = _np.asarray(time)
        per_signal_times: dict[str, "Array"] = {}
        if t.ndim == 0 or t.shape[0] == 0:
            # Degenerate: nothing recorded.  Hand back the global time
            # vector for every signal — keeps ``time_for`` well-defined.
            for name in outputs:
                per_signal_times[name] = t
            return per_signal_times

        for name, arr in outputs.items():
            a = _np.asarray(arr)
            if a.shape[0] != t.shape[0]:
                # Pipeline invariant violation; fall back to global time
                # rather than guess at a slicing convention.
                per_signal_times[name] = t
                continue
            if a.shape[0] == 1:
                per_signal_times[name] = t
                continue
            if a.ndim == 1:
                diffs = _np.abs(_np.diff(a))
                changed_tail = diffs > atol
            else:
                d = _np.diff(a, axis=0)
                # Reduce all non-leading axes — any element changing
                # marks the sample as new.
                changed_tail = _np.any(
                    _np.abs(d) > atol,
                    axis=tuple(range(1, d.ndim)),
                )
            mask = _np.concatenate(([True], changed_tail))
            per_signal_times[name] = t[mask]
        return per_signal_times
