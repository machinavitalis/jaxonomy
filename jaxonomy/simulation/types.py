# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from dataclasses_json import dataclass_json

from ..backend import ODESolverOptions, ODESolverState, numpy_api

if TYPE_CHECKING:
    from ..framework import ContextBase, EventCollection, SystemCallback
    from ..backend.typing import Array, Scalar
    from ..backend.results_data import AbstractResultsData


__all__ = [
    "StepEndReason",
    "GuardIsolationData",
    "ContinuousIntervalData",
    "SimulatorOptions",
    "SimulatorState",
    "SimulationResults",
    "ResultsOptions",
    "ResultsMode",
    "NativeInterpolant",
]


class NativeInterpolant(NamedTuple):
    """T-012a-followup: per-major-step native solver-interpolant container.

    Carried on ``SimulationResults.solver_states`` when the simulator
    was run with ``record_solver_states=True`` AND the active solver
    exposes a fixed-shape per-step polynomial (Dopri5 today).

    Stored on the host as numpy arrays — the recording pipeline trims
    the JAX device buffer at finalize time, so ``query()`` can reduce
    to a plain ``np.searchsorted`` + ``np.polyval`` without touching
    XLA again.

    Attributes:
        t_prev: shape ``(N,)`` start times of each segment.
        t_step: shape ``(N,)`` end times.  ``t_prev[i] < t_step[i]``
            (zero-width slots are dropped at finalize).  Both are
            monotonically non-decreasing.
        interp_coeff: shape ``(N, n_coeff, n_y)`` polynomial
            coefficients.  Evaluated as
            ``polyval(coeff[i], (t_eval - t_prev[i]) / (t_step[i] - t_prev[i]))``
            and unraveled back to the continuous-state pytree.
        unravel: callable that restores a flat ``(n_y,)`` array to the
            continuous-state pytree shape.  May be ``None`` for solvers
            with a single 1-D state vector — the result is then handed
            back as a flat array.
        solver: short tag identifying the solver family ("dopri5" for
            now; future backends would extend this).
    """

    t_prev: Any
    t_step: Any
    interp_coeff: Any
    unravel: Any
    solver: str = "dopri5"


# Internal data structure to determine why a major step ended.
class StepEndReason(IntEnum):
    NothingTriggered = 0
    TimeTriggered = 1
    GuardTriggered = 2
    BothTriggered = 3
    TerminalEventTriggered = 4


# Internal data structure for the bisection search for zero-crossing events.
class GuardIsolationData(NamedTuple):
    zc_before_time: int
    zc_after_time: int
    guards: EventCollection
    context: ContextBase


# Internal data structure for the results of advancing continuous time
class ContinuousIntervalData(NamedTuple):
    context: ContextBase
    triggered: bool  # Any zero-crossing events trigger?
    terminate_early: bool  # Terminal event triggered?
    t0: int  # Beginning of the interval (integer time stamp)
    tf: int  # End of the interval (integer time stamp)
    results_data: AbstractResultsData

    # The current state of the ODE solver
    ode_solver_state: ODESolverState = None

    # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
    # per-event mask of shape ``(N_events,)`` (or scalar ``False`` when
    # Zeno protection is disabled or the recovery probe has not just
    # fired) carried into ``_advance_continuous_time`` so the inner ODE
    # step loop can apply a direction-aware ``w0`` nudge for events
    # whose Zeno latch was cleared on the previous major step.  Default
    # ``None`` means "no nudge" — preserves the byte-equivalent default-
    # off path and the autodiff path that does not plumb this field.
    recovery_just_cleared: Any = None

    # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
    # per-event mask of shape ``(N_events,)`` populated by
    # ``_advance_continuous_time`` after the inner ODE step's events
    # collection has been resolved.  This carries per-event triggered
    # info back up to ``_major_step`` so the simulator-level Zeno
    # tracker can update each event's ``tprev[i]`` independently
    # rather than broadcasting a single scalar ``triggered`` to all
    # events (which spuriously latches unrelated events when one
    # leaf's cascade drives sub-tolerance major-step ends).  Default
    # ``None`` keeps the legacy scalar broadcast path.
    per_event_triggered: Any = None

    # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
    # per-event mask of shape ``(N_events,)`` carried through to the
    # inner ODE step's ``check_triggered`` so events whose Zeno latch
    # is engaged at the start of the major step have their
    # ``triggered`` flag suppressed inside the ODE loop.  Without
    # this, a latched-but-still-firing event would terminate the ODE
    # step at the localised trigger time, preventing other unfrozen
    # leaves from advancing.  Default ``None`` keeps the legacy
    # behaviour where every active trigger terminates the ODE step.
    prior_zeno_active: Any = None


# Internal data structure to carry through the main simulation loop
class SimulatorState(NamedTuple):
    context: ContextBase
    timed_events: EventCollection
    step_end_reason: StepEndReason

    # Integer representation of simulation time - used for synchronizing
    # events without floating point drift.
    int_time: int

    results_data: AbstractResultsData

    # The current state of the ODE solver
    ode_solver_state: ODESolverState = None

    # T-027a-followup: simulator-level Zeno tracker fields.  ``zeno_tprev``
    # carries the time of the most recent zero-crossing event; ``zeno_active``
    # is the latch flag (True while the simulator is in a Zeno-hold);
    # ``zeno_frozen_steps`` counts the number of consecutive major steps
    # since the latch engaged so the recovery probe can periodically test
    # for cascade resolution.  All three default to zero/False — the
    # default-off path (`SimulatorOptions.zeno_protection_enabled=False`,
    # the default) never reads or writes these fields, so the carry is
    # byte-equivalent to the pre-T-027a-followup pytree.
    #
    # T-027a-followup-vector-tprev: when ``zeno_protection_enabled=True``,
    # ``Simulator.initialize`` upgrades ``zeno_tprev`` and ``zeno_active``
    # to per-event vectors of shape ``(N_events,)`` (one slot per
    # ``ZeroCrossingEvent`` in ``system.zero_crossing_events.events``).
    # T-027a-followup-per-event-recovery: ``zeno_frozen_steps`` is now
    # also vectorised to ``(N_events,)`` when enabled, so each event's
    # recovery probe fires independently — a still-cascading event no
    # longer gets a free probe when an unrelated event's cascade ends.
    # The defaults below remain scalars so that the default-off path's
    # pytree shape is unchanged.
    zeno_tprev: Any = 0.0
    zeno_active: Any = False
    zeno_frozen_steps: Any = 0
    # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
    # per-event mask flagged True for one major step after the recovery
    # probe clears event ``i``'s latch.  Consumed by the next major step's
    # ``_advance_continuous_time`` to apply a direction-aware nudge to
    # the guard's ``w0`` so a leaf at the post-reset rest condition
    # (e.g. ``h=0`` for a bouncing ball) can still trigger its
    # ``positive_then_non_positive`` reset on the cleared step.  The
    # default-off path (``zeno_protection_enabled=False``) keeps the
    # scalar default so the carry pytree shape is byte-equivalent to
    # the pre-followup state.
    zeno_recovery_just_cleared: Any = False


# Container for options related to the Simulator class.
@dataclasses.dataclass
class SimulatorOptions:
    """Options for the hybrid simulator.

    See documentation for `simulate` for details on these options.
    This also contains all configuration for the ODE solver as a subset of options
    so that multiple options classes don't need to be created separately.
    """

    math_backend: str = dataclasses.field(
        default_factory=lambda: numpy_api.active_backend
    )
    enable_tracing: bool = True
    enable_autodiff: bool = False
    precision: str = "auto"  # "auto", "float32", or "float64"

    # diff-mode follow-up: explicit, self-documenting differentiation-mode
    # selector. ``enable_autodiff`` conflates two things — "make the sim
    # differentiable at all" and "install the reverse-mode adjoint" — which
    # makes forward-mode autodiff counterintuitive: it requires
    # ``enable_autodiff=False`` so JAX traces the real solver ops (the
    # reverse-mode ``custom_vjp`` intercepts ``jax.jacfwd`` / ``jvp`` and does
    # NOT forward-differentiate the solver). Set ``diff_mode`` instead of
    # reasoning about the boolean; it resolves into ``enable_autodiff`` (the
    # canonical flag every downstream site reads):
    #   - "reverse"   → reverse-mode adjoint (``jax.grad`` / ``jacrev``);
    #                   sets ``enable_autodiff=True``.
    #   - "forward"   → forward-mode (``jax.jacfwd`` / ``jvp``); sets
    #                   ``enable_autodiff=False`` so the real ops are traced.
    #   - "none"      → no autodiff; sets ``enable_autodiff=False``.
    #   - "auto"/None → leave ``enable_autodiff`` as given (back-compat
    #                   default).
    # Resolved once in ``__post_init__`` and then cleared back to ``None`` so a
    # later ``dataclasses.replace(enable_autodiff=...)`` is never re-clobbered.
    diff_mode: str | None = None

    # If autodiff is enabled, max_major_steps must be set in order to bound the number
    # of iterations in the while loop.  When running a simulation using the `simulate`
    # function, this can typically be determined automatically based on the number of
    # periodic events in the system.  However, it should be specified manually in the
    # following cases:
    #   - When running a simulation by creating a `Simulator` object and calling the
    #     `advance_to` method directly. In this case the `Simulator` object does not
    #     attempt to automatically determine a bound on the number of major steps.
    #   - When autodiff is used to compute the sensitivity with respect to simulation
    #     end time, for example when computing periodic limit cycles. In this case the
    #     time variables passed to `estimate_max_major_steps` are JAX tracers and cannot
    #     be used to determine a fixed (static) bound on the number of major steps.
    #   - When the system has frequent zero-crossing events.  In this case the "safety
    #     factor" in the heuristic for estimating the number of major steps may be too
    #     small, underestimating the bound on the number of major steps.
    # In any case, `estimate_max_major_steps` can still be called statically ahead
    # of time to determine a reasonable value for `max_major_steps`, using for instance
    # a conservative bound on end time and safety factor.
    max_major_steps: int = None
    # NOTE (T-A3): ``max_major_step_length`` is *JIT-static* — it participates
    # in deriving ``max_major_steps`` (the bounded-loop trip count), which is a
    # Python int baked into the compiled kernel. Changing it between calls to a
    # jitted ``simulate`` / ``advance_to`` therefore forces a recompile. For
    # parameter sweeps, hold it fixed and vary traced quantities (initial state,
    # dynamic parameters) instead.
    max_major_step_length: float = None
    # T-A3 follow-up: ``max_major_step_size`` is an accepted alias for
    # ``max_major_step_length`` — it reads more naturally next to
    # ``max_minor_step_size`` and is the spelling careful users reach for first.
    # Reconciled in ``__post_init__``: if only one is set it populates the
    # other; setting both to conflicting values raises.
    max_major_step_size: float = None

    # Length of the recording ring buffer for the time series.  When ``None``
    # (the default), ``_check_options`` auto-sizes this to ``max_major_steps``.
    #
    # IMPORTANT (T-B3/B8): the recorder saves one sample per *accepted minor
    # (solver) step*, not per major step. Adaptive solvers — Dopri5 and the
    # "auto" default — take many minor steps per major step, and *more* of them
    # as ``rtol`` / ``atol`` tighten. A tight-tolerance Dopri5 run can therefore
    # record far more samples than the major-step-derived auto size, overrunning
    # the ring buffer and silently dropping the *head* of the trajectory
    # (``results.time`` then starts mid-run). The simulator detects this after
    # the fact and emits a loud, solver/tolerance-aware ``UserWarning``
    # recommending a concrete larger ``buffer_length``. To avoid it up front:
    # set ``buffer_length`` explicitly for long fine-grained recordings, loosen
    # the tolerances, or use the fixed-step ``ode_solver_method="rk4"`` (whose
    # sample count is predictable from ``max_minor_step_size``). Set a small
    # fixed N for memory-constrained streaming.
    buffer_length: int | None = None

    # ODE solver options
    ode_solver_method: str = "auto"  # Dopri5 (jax/scipy) or BDF (jax)
    rtol: float = 1e-6  # Relative tolerance for adaptive solvers
    atol: float = 1e-8  # Absolute tolerance for adaptive solvers
    min_minor_step_size: float = None
    max_minor_step_size: float = None

    # This is used to bound the number of "checkpoints" in the adjoint solver and
    # is used only when autodiff is enabled.  Increasing this may improve the
    # accuracy of the adjoint solver (especially over long integration times), but
    # will also increase memory usage.  Whether or not the resulting adjoint solve
    # is faster depends on the details of the problem, for instance on the number of
    # major steps and the ODE solver tolerance.  This can also be set to None to
    # disable checkpointing altogether.
    max_checkpoints: int = 16

    # This option determines whether the simulator saves any data.  If the
    # simulation is initiated from `simulate` this will be set automatically
    # depending on whether `recorded_signals` is provided.  Hence, this
    # should not need to be manually configured.
    # NOTE: remove this and use `recorded_signals` instead. There are usecases
    # where simulate() is not used and we use the Simulator's advance_to function
    # directly. In those cases, recorded_signals can be set while save_time_series
    # is False which is confusing.
    save_time_series: bool = False

    # Dictionary of ports (or other cache sources) for which the time series should
    # be recorded. Note that if the simulation is initiated from `simulate` and
    # `recorded_signals` is provided as a kwarg to `simulate`, anything set here
    # will be overridden.  Hence, this should not need to be manually configured.
    recorded_signals: dict[str, SystemCallback] = None

    # If the context is not needed for anything, opting to not return it can
    # speed up compilation times.  For instance, typical simulation calls from
    # the UI don't use the context for anything, so model_interface.py will
    # set `return_context=False` for performance.
    return_context: bool = True

    # Validate the diagram before simulating to check for common errors and unsupported
    # feature interactions like autodiff through python-only blocks.
    validate: bool = True

    # Zero crossings are localized in time using the ODE solver interpolant,
    # which provides state values for any time value in the previous integration
    # time interval.
    # Bisection is used to search the time interval. Rather than run bisection
    # in a while loop until the time interval is _small_, bisection is run for
    # fixed number of iterations, as this results in localizing zero crossings in
    # time within some small fraction of the integrated time interval.
    # e.g. if the major step length is 1.0 second, and bisection is run for 40
    # loops, the zero crossing time tolerance is approx. 1e-12, a.k.a. picosecond.
    zc_bisection_loop_count: int = 40

    # Scale of integer time used for event synchronization.
    #   - "auto" (default): pick the finest power-of-ten scale that still
    #     represents ``t_span[1]`` with headroom. Short simulations keep
    #     picosecond resolution (1e-12, max ~0.3 years); longer horizons
    #     transparently coarsen (1e-9 ns, 1e-6 µs, ...) so a multi-year
    #     simulation just runs instead of raising a representability error.
    #   - a float (e.g. 1e-9): pin the scale explicitly.
    #   - None: leave the global IntegerTime scale untouched (legacy escape
    #     hatch; not recommended — relies on process-global state).
    int_time_scale: float | str | None = "auto"

    # Called at the end of each major step with the current time as an argument.
    major_step_callback: Callable[[Scalar]] = None

    # T-022a: opt into the lower-triangular discrete-update scheduler.  When
    # True, Phase 2 of `handle_discrete_update` evaluates state updates in
    # the topological order of the discrete dependency graph: a block reading
    # an upstream block's discrete state sees the post-update x⁺ rather than
    # the snapshotted x⁻.  Cycles in the dependency graph raise
    # `DependencyCycleError`.  Default `False` (diagonal Drake-style update,
    # preserving the cross-block-swap atomicity documented on
    # `SystemBase.handle_discrete_update`).
    lower_triangular_discrete_update: bool = False

    # T-105 Phase 1: opt-in multirate consistency check.  When set to
    # ``"warn"`` or ``"error"``, ``simulate`` runs
    # ``rate_groups.detect_rate_mismatches`` over the diagram immediately
    # after ``validate_diagram``.  A "warn" run logs each mismatched
    # connection through ``warnings.warn(RateMismatchWarning, ...)`` but
    # otherwise lets the simulation proceed (back-compat for existing
    # multirate models that work today thanks to per-block periodic
    # events).  An "error" run raises ``RateMismatchError`` on the first
    # offender.  Default ``None`` keeps the path completely off so
    # single-rate diagrams stay byte-equivalent.  Phase 2 (T-123) will
    # add auto-insertion of ``RateTransition`` blocks; until then this
    # is a diagnostic-only switch.
    check_rate_transitions: str | None = None

    # T-003a: opt-in DAE constraint projection (Newton's method on the
    # algebraic states) at the end of each major step.  The differential
    # states are held fixed; only the algebraic component is corrected.
    # Default `False` (no projection — backwards-compatible).  Has no
    # effect on systems without a mass matrix; the simulator skips the
    # projection cleanly in that case.  `dae_projection_tol` is the
    # max-abs threshold above which the corrector iterates;
    # `dae_projection_max_iter` caps the Newton loop at 3 iterations by
    # default — the algebraic constraint is typically linear or
    # mildly-nonlinear in the algebraic unknowns and converges in 1–2
    # steps.  See ``jaxonomy.simulation.dae_projection`` for details.
    dae_projection_enabled: bool = False
    dae_projection_tol: float = 1e-8
    dae_projection_max_iter: int = 3

    # T-113-followup-baumgarte-and-ssp: opt-in Baumgarte stabilization of
    # the algebraic constraint residual.  When ``baumgarte_alpha`` and/or
    # ``baumgarte_beta`` are non-None, the simulator wraps
    # ``ode_solver.ode_rhs`` to add ``-2α·ġ - β²·g`` to each algebraic
    # row of the rhs, where ``g = f_a(x)`` is the algebraic-row residual
    # at the current state.  This drives drift to zero exponentially —
    # critically damped at α = β = 1/τ (τ = relaxation time).
    #
    # Default ``None`` for both → no augmentation, the disabled hot path
    # is byte-equivalent (the wrapper short-circuits and returns the
    # original ``ode_rhs`` unchanged when both gains are None).  Has no
    # effect on systems without a mass matrix.  Composes cleanly with
    # ``dae_projection_enabled`` (projection at major-step boundaries
    # kills accumulated drift; Baumgarte damps drift continuously
    # between projections).
    #
    # See :func:`jaxonomy.simulation.dae_projection.baumgarte_augment_ode_rhs`
    # for the augmentation details and the index-reduction caveat.
    baumgarte_alpha: float | None = None
    baumgarte_beta: float | None = None

    # T-113-followup-event-reprojection: opt-in DAE constraint projection
    # immediately after each discrete event reset *within* a major step.
    # T-003a's ``dae_projection_enabled`` projects only at the end of a
    # major step — after the ODE integration plus any localized ZC reset
    # have already happened.  Discrete updates handled at the *top* of
    # ``_major_step`` (``_handle_discrete_update``) modify state before
    # continuous integration runs; if the reset map drops state off the
    # constraint manifold, the subsequent ODE step integrates on infeasible
    # state until the next major-step boundary projection (T-003a) catches
    # up.  Setting this to ``True`` runs ``project_constraints`` right
    # after the discrete-update reset and again after a triggered ZC reset
    # within ``_advance_continuous_time``, so continuous integration always
    # resumes on feasible state.  Default ``False`` (byte-equivalent to
    # the pre-followup hot path).  Has no effect on systems without a
    # mass matrix; the simulator skips the hook cleanly in that case.
    # Composes with ``dae_projection_enabled`` (both can run — major-step
    # boundary projection still fires) and with ``baumgarte_*`` (continuous
    # damping between projections).  Reuses the same ``dae_projection_tol``
    # / ``dae_projection_max_iter`` knobs.
    dae_reproject_after_events: bool = False

    # T-003b: opt-in DAE constraint-residual drift monitor.  When set,
    # the simulator computes ``||f_a||_∞`` at each major step; values
    # above the threshold emit a ``UserWarning`` naming the step time
    # and the measured residual.  Default ``None`` disables the check
    # (no overhead — the default-off path is byte-equivalent to the
    # pre-T-003b code).  Disable projection
    # (``dae_projection_enabled=False``) and enable just this threshold
    # to monitor drift without correcting it.  Has no effect on systems
    # without a mass matrix; the simulator skips the check cleanly in
    # that case.
    dae_drift_threshold: float | None = None

    # T-113 Phase 1: opt-in per-major-step DAE constraint drift trace.
    # Companion to ``dae_drift_threshold`` — that option emits a
    # ``UserWarning`` per violating step but does not retain the raw
    # samples.  When ``record_dae_drift=True``, the simulator tees the
    # post-projection residual ``||f_a||_∞`` plus the step time to a
    # Python-side accumulator via ``jax.debug.callback`` (mirroring the
    # ``_BDFConditionMonitor`` pattern from
    # T-038a-followup-bdf-condition-check) and surfaces the captured
    # ``(time, residual)`` arrays on
    # ``SimulationResults.dae_drift_trace`` — a small dict
    # ``{"time": np.ndarray, "residual": np.ndarray}`` post-finalize.
    # Default ``False`` disables the trace entirely; the default-off
    # path is byte-equivalent (no extra ops compiled in, no monitor
    # constructed) and ``SimulationResults.dae_drift_trace is None``.
    # Has no effect on systems without a mass matrix; the simulator
    # skips the trace cleanly in that case.
    record_dae_drift: bool = False

    # T-125-followup-record-event-times: opt-in capture of zero-crossing
    # event firing times during ``simulate``.  When ``True``, the simulator
    # tees ``(event_index, t_event)`` from each major step that ends on a
    # guard trigger to a Python-side ``_EventTimeRecorder`` via
    # ``jax.debug.callback`` (mirrors the ``_BDFConditionMonitor`` /
    # ``_DAEDriftMonitor`` pattern from T-038a-followup-bdf-condition-check
    # / T-113 phase 1) and surfaces the captured firing-time arrays on
    # ``SimulationResults.event_times`` — a dict ``{event_index:
    # np.ndarray}`` post-finalize.  Default ``False`` disables the
    # capture entirely (no monitor constructed, no ops compiled in) and
    # ``SimulationResults.event_times is None`` — preserves the byte-
    # equivalent default-off path.  Diagrams without zero-crossing events
    # yield ``None`` even when the option is True.  Pairs with
    # :func:`jaxonomy.event_time_gradient` for the implicit-function
    # gradient: feed an entry from ``results.event_times`` straight in as
    # the recorded ``t_event`` rather than tracking it manually.
    record_event_times: bool = False

    # T-038a-followup-bdf-condition-check: opt-in BDF Newton-iteration
    # condition-number diagnostic.  When non-None, the BDF solver's
    # ``newton_iteration`` computes ``jnp.linalg.cond(M - c*J)`` once
    # per major step (cheap on small Newton matrices — one extra SVD
    # on an ``n_states × n_states`` matrix) and forwards the estimate
    # plus the current time to a Python-side aggregator via
    # ``jax.debug.callback``.  The simulator tracks the *maximum*
    # condition number observed across the whole trajectory along
    # with the time at which it occurred, and, on ``simulate`` exit,
    # emits ONE ``UserWarning`` naming the threshold, the max value,
    # and the time of occurrence.  The aggregated warning surface is
    # deliberately *not* per-step — a per-step warning would be too
    # noisy on stiff systems where every step is poorly conditioned
    # — but the underlying max-tracker is per-step so transient
    # ill-conditioning is still surfaced.  Default ``None`` disables
    # the diagnostic entirely (no extra ops compiled in, the BDF hot
    # path is byte-equivalent to the pre-followup code).  Has no
    # effect on non-BDF solvers; the simulator skips attaching the
    # monitor when ``ode_solver`` is not a BDF solver.
    bdf_condition_warning_threshold: float | None = None

    # T-027a-followup: simulator-level Zeno protection toggles.  When
    # ``zeno_protection_enabled=False`` (the default), the simulator's
    # ``_major_step`` skips the Zeno tracker entirely — the carry is
    # byte-equivalent to the pre-followup hot path, no extra ops compiled
    # in, no test-suite churn.  When True, every major step that ends on
    # a guard trigger consults ``(time - sim_state.zeno_tprev) <
    # zeno_tolerance`` to decide whether to engage a global Zeno-hold; on
    # engagement, the simulator latches ``zeno_active=True`` and pauses
    # continuous-time integration.  ``zeno_recovery_period`` is the
    # number of consecutive frozen major steps after which the simulator
    # briefly probes for recovery: it clears ``zeno_active`` for one step,
    # and the next guard-trigger check will naturally re-engage Zeno if
    # the cascade is still active (because ``(time - tprev) <
    # zeno_tolerance`` will fire again), or stay cleared otherwise — so
    # transient cascades release the latch automatically while persistent
    # ones stay frozen.  The simulator-level path complements (does not
    # replace) the per-leaf ``declare_zero_crossing(zeno_tolerance=...)``
    # protection from T-027/T-027a.
    #
    # T-027a-followup-vector-tprev: ``zeno_tprev`` and ``zeno_active`` are
    # per-event vectors of shape ``(N_events,)``, one slot per
    # ``ZeroCrossingEvent``.  Each event's last-firing time is tracked
    # independently, so an unrelated event's cascade does not poison the
    # tolerance check for another event.
    # T-027a-followup-per-event-recovery: ``zeno_frozen_steps`` is also a
    # ``(N_events,)`` vector, so the recovery probe fires per-event —
    # event ``i`` clears its own latch when ``frozen[i] >= K`` without
    # affecting any other event's counter or latch.
    zeno_protection_enabled: bool = False
    zeno_tolerance: float = 1e-6
    zeno_recovery_period: int = 10

    # T-013a: opt-in per-signal timestamp capture.  The recording pipeline
    # stores every recorded signal at every major step (legacy global-vector
    # behaviour).  When ``per_signal_timestamps=True``, an out-of-JIT post-
    # processor in ``simulate`` populates
    # ``SimulationResults.per_signal_times`` with each signal's native cadence
    # so ``time_for(name)`` switches to the per-signal vector.
    #
    # ``per_signal_timestamps_mode`` selects the strategy:
    #   - ``"auto"`` (default when the option is on): Mode A — classify each
    #     signal from its source ``OutputPort`` (continuous / periodic /
    #     default) and trim BOTH ``per_signal_times[name]`` AND
    #     ``outputs[name]`` to the schedule for periodic signals.  Genuine
    #     storage savings for downstream consumers (a 1 Hz signal in a 10 s
    #     simulation produces ~11 stored samples instead of ~1001).  Per-
    #     signal fallback to Mode B for signals the classifier cannot place.
    #   - ``"schedule"``: alias for ``"auto"`` — same Mode A path.
    #   - ``"diff"``: Mode B — value-diff dedup of times only, outputs stay
    #     at full length.  Bit-equivalent to the previous Mode-B behaviour.
    #   - ``"buffers"``: T-013a-followup-mode-a-buffers — true in-JIT Mode A.
    #     The simulator allocates per-signal ``(times, values, count)`` rings
    #     at init and each major step's recording write only consumes a slot
    #     in a signal's ring when that signal's cadence classification fires
    #     at the current time.  Cuts both peak buffer memory (vs.
    #     ``"auto"``'s post-finalize trim) and finalize-time post-processing.
    #     Falls back transparently to ``"auto"`` semantics for the global
    #     ``outputs`` shape; the storage saving lives in
    #     ``SimulationResults.outputs[name]`` and ``per_signal_times[name]``.
    #
    # ``per_signal_timestamps_atol`` is the absolute tolerance used when
    # detecting "the signal changed since the last sample" (Mode B) or "the
    # current time matches a tick of the period schedule" (Mode A).  The
    # default 1e-12 catches genuine zero-order-hold plateaus / synchronises
    # to integer-time picosecond precision while staying below any realistic
    # float64 round-off.  Mode A scales the time-tolerance by the period so
    # long simulations don't drift off-schedule.
    #
    # Note: Mode A operates on the trimmed numpy arrays out-of-JIT — the
    # in-JIT recording buffer is unchanged from the legacy path, so peak
    # buffer memory is the same.  The savings are in the post-finalize
    # arrays (``outputs[name]`` and ``per_signal_times[name]``) handed to
    # the user.
    per_signal_timestamps: bool = False
    per_signal_timestamps_atol: float = 1e-12
    per_signal_timestamps_mode: str = "auto"

    # T-110 Phase 1: opt-in provenance/reproducibility manifest.  When
    # ``True``, ``simulate`` calls
    # :func:`jaxonomy.simulation.provenance.compute_provenance` (entirely
    # outside the JIT-traced kernel) and attaches the resulting
    # :class:`ProvenanceManifest` to ``SimulationResults.provenance``.
    # Default ``False`` → ``SimulationResults.provenance is None`` and the
    # simulate path stays byte-equivalent to the pre-followup behaviour.
    record_provenance: bool = False

    # T-012a: opt-in higher-order interpolant for ``SimulationResults.query``.
    # When ``False`` (default) ``query`` uses ``jnp.interp`` linear
    # interpolation — preserves legacy behaviour exactly.  When ``True`` the
    # results pipeline marks ``SimulationResults.solver_states = "pchip"`` (a
    # sentinel placeholder for the eventual native solver-state plumbing) and
    # ``query`` falls back to a PCHIP cubic-Hermite interpolant built from the
    # recorded ``(time, outputs)`` samples.  PCHIP is shape-preserving (no
    # spurious overshoot at zero-order-hold plateaus) and ~3 orders of
    # magnitude more accurate than linear on smooth signals.  The native
    # solver-state path (storing ``Dopri5State.interp_coeff`` per major step
    # for sub-ULP accuracy) remains a deferred follow-up.
    record_solver_states: bool = False

    # Internal flag: True when max_major_steps was explicitly set by the caller rather
    # than auto-estimated by _check_simulate_options.  When True, the bounded fori_loop
    # is used even without enable_autodiff=True so that max_major_steps is honored as a
    # hard simulation budget (useful for Zeno-protection and for non-autodiff workflows
    # that still want a step-count cap).
    _explicit_max_major_steps: bool = dataclasses.field(
        default=False, repr=False, compare=False
    )

    def __post_init__(self):
        # T-A3 follow-up: reconcile the ``max_major_step_size`` alias with
        # the canonical ``max_major_step_length``. When only the alias is set it
        # populates the canonical field; the canonical field otherwise wins. The
        # alias is always re-synced so reads of either are consistent — this
        # also keeps ``dataclasses.replace`` (which re-runs ``__post_init__``)
        # well-behaved regardless of which spelling the caller overrode.
        if self.max_major_step_length is None and self.max_major_step_size is not None:
            self.max_major_step_length = self.max_major_step_size
        self.max_major_step_size = self.max_major_step_length

        # diff-mode follow-up: resolve the explicit differentiation-mode
        # selector into the canonical ``enable_autodiff`` flag, then clear it so
        # re-runs of __post_init__ (via ``dataclasses.replace``) do not re-apply
        # it and clobber an explicit ``enable_autodiff`` override.
        if self.diff_mode is not None:
            valid_modes = ("auto", "forward", "reverse", "none")
            if self.diff_mode not in valid_modes:
                raise ValueError(
                    f"diff_mode={self.diff_mode!r} is not valid; expected one "
                    f"of {valid_modes} (or None)."
                )
            if self.diff_mode == "reverse":
                self.enable_autodiff = True
            elif self.diff_mode in ("forward", "none"):
                # enable_autodiff defaults to False, so a True here is an
                # explicit, contradictory override — fail loudly rather than
                # silently picking one (the footgun this option exists to kill).
                if self.enable_autodiff:
                    raise ValueError(
                        f"Conflicting differentiation settings: diff_mode="
                        f"{self.diff_mode!r} requests no reverse-mode adjoint, "
                        "but enable_autodiff=True installs one. Forward-mode "
                        "autodiff (jax.jacfwd / jvp) must not use the reverse "
                        "adjoint. Use diff_mode='forward' on its own (leave "
                        "enable_autodiff unset)."
                    )
                self.enable_autodiff = False
            # "auto" leaves enable_autodiff untouched.
            self.diff_mode = None

    @property
    def ode_options(self) -> ODESolverOptions:
        return ODESolverOptions(
            rtol=self.rtol,
            atol=self.atol,
            min_step_size=self.min_minor_step_size,
            max_step_size=self.max_minor_step_size,
            method=self.ode_solver_method,
            enable_autodiff=self.enable_autodiff,
            max_checkpoints=self.max_checkpoints,
        )

    def __repr__(self) -> str:
        return (
            f"SimulatorOptions("
            f"math_backend={self.math_backend}, "
            f"enable_tracing={self.enable_tracing}, "
            f"max_major_step_length={self.max_major_step_length}, "
            f"max_major_steps={self.max_major_steps}, "
            f"ode_solver_method={self.ode_solver_method}, "
            f"rtol={self.rtol}, "
            f"atol={self.atol}, "
            f"min_minor_step_size={self.min_minor_step_size}, "
            f"max_minor_step_size={self.max_minor_step_size}, "
            f"zc_bisection_loop_count={self.zc_bisection_loop_count}, "
            f"save_time_series={self.save_time_series}, "
            f"recorded_signals={len(self.recorded_signals or [])}, "  # changed
            f"return_context={self.return_context}, "
            f"validate={self.validate}"
            f")"
        )


class ResultsMode(Enum):
    auto = 0
    discrete_steps_only = 1
    fixed_interval = 2


@dataclass_json
@dataclasses.dataclass
class ResultsOptions:
    mode: Optional[ResultsMode] = ResultsMode.auto
    max_results_interval: Optional[float] = None
    fixed_results_interval: Optional[float] = None
    # NOTE: maybe include recorded_signals here?


class SimulationResults(NamedTuple):
    """Data structure for the results of a simulation.

    Attributes:
        context (ContextBase):
            The output context of the simulation, containing final states, times, etc.
            May be None if `return_context=False` was passed to `simulate`.
        outputs (dict[str, Array]):
            A dictionary of the outputs of the simulation, keyed by the name provided
            to `recorded_signals` in `simulate`.  May be None if `recorded_signals` is
            not provided to `simulate`.
        time (Array):
            The time vector of the simulation.
        parameters (dict[str, Any]):
            The parameters used in the simulation, used in ensemble simulations
            to identify different runs.
    """

    context: ContextBase
    time: Array = None
    outputs: dict[str, Array] = None
    parameters: dict[str, Any] = None
    # T-013: optional per-signal timestamp vectors.  When non-None,
    # ``per_signal_times[name]`` is the native time vector for the
    # corresponding recorded signal ``outputs[name]``.  When None, every
    # signal shares ``self.time`` (backwards-compatible default).
    per_signal_times: Optional[dict[str, Array]] = None

    # T-012a: optional marker that ``query`` should use a higher-order
    # interpolant rather than linear.  Today this field carries the literal
    # string ``"pchip"`` when ``SimulatorOptions.record_solver_states=True``
    # (the PCHIP fallback shipped in T-012a partial); a future T-012a-followup
    # will populate it with the per-major-step solver-state pytree needed for
    # the solver's native dense interpolant.  When ``None`` (default and
    # legacy), ``query`` uses ``jnp.interp`` linear interpolation — fully
    # backwards-compatible.
    solver_states: Optional[Any] = None

    # T-110 Phase 1: optional reproducibility manifest populated by
    # ``simulate(...)`` when ``SimulatorOptions.record_provenance=True``.
    # When ``None`` (default), the path is byte-equivalent to the pre-
    # T-110 behaviour.  See :mod:`jaxonomy.simulation.provenance`.
    provenance: Optional[Any] = None

    # T-113 Phase 1: optional per-major-step DAE constraint drift trace
    # populated by ``simulate(...)`` when
    # ``SimulatorOptions.record_dae_drift=True``.  When non-None, a dict
    # of ``{"time": np.ndarray, "residual": np.ndarray}`` recording the
    # post-projection ``||f_a||_∞`` value at each major step (in step
    # order; chronological).  When ``None`` (default), the trace was
    # not recorded — the path is byte-equivalent to the pre-T-113
    # behaviour.  Pure-ODE systems (no mass matrix) yield ``None``
    # even when the option is True.
    dae_drift_trace: Optional[dict] = None

    # T-125-followup-record-event-times: optional dict of zero-crossing
    # event firing times populated by ``simulate(...)`` when
    # ``SimulatorOptions.record_event_times=True``.  When non-None,
    # ``event_times[i]`` is the 1-D ``np.ndarray`` of firing times for the
    # ``i``-th zero-crossing event (matching the order of
    # ``system.zero_crossing_events.events``).  Events that never fired
    # have an empty array; events that fired multiple times have a
    # monotonically-increasing array.  When ``None`` (default), the
    # capture was not requested — the path is byte-equivalent to the
    # pre-followup behaviour.  Diagrams with no zero-crossing events
    # yield ``None`` even when the option is True.  Designed to feed
    # straight into :func:`jaxonomy.event_time_gradient` so callers do
    # not have to track event times manually.
    event_times: Optional[dict] = None

    def time_for(self, signal: str):
        """Return the time vector associated with ``signal``.

        Falls back to ``self.time`` when ``per_signal_times`` is None
        or does not contain ``signal`` — matching the legacy behaviour
        where all recorded signals share one timeline.
        """
        if self.per_signal_times is not None and signal in self.per_signal_times:
            return self.per_signal_times[signal]
        return self.time

    def align(self, time_vector, signals=None):
        """Resample recorded signals onto a common time vector (T-013).

        Useful when per-signal timestamps have been captured at
        different native rates and a rectangular timeline is required
        for plotting or further processing.

        Args:
            time_vector: 1-D array of times to sample at.
            signals: Optional iterable of signal names to include.
                Defaults to all recorded signals.

        Returns:
            A new :class:`SimulationResults` where every requested
            signal has been linearly interpolated onto ``time_vector``.
            ``per_signal_times`` is reset to None because all signals
            now share the same timeline.
        """
        import jax.numpy as _jnp
        import numpy as _np

        if self.outputs is None:
            raise ValueError(
                "SimulationResults.align: no recorded signals to align."
            )
        signals = list(signals) if signals is not None else list(self.outputs)
        time_vector = _jnp.asarray(time_vector)
        t_q = _np.asarray(time_vector)

        new_outputs = {}
        for name in signals:
            if name not in self.outputs:
                raise ValueError(
                    f"SimulationResults.align: unknown signal {name!r}.  "
                    f"Recorded: {list(self.outputs)}"
                )
            t_src = _np.asarray(self.time_for(name))
            y_src = _np.asarray(self.outputs[name])
            # T-013a: when ``per_signal_times`` is populated by Mode B,
            # the outputs array is full-resolution while ``t_src`` is
            # deduplicated.  Reconstruct the matching value vector by
            # picking the leading-axis indices from ``self.time`` that
            # equal the per-signal timestamps, so interp's xp/fp pair
            # is the same length.
            if (
                self.per_signal_times is not None
                and name in self.per_signal_times
                and t_src.shape[0] != y_src.shape[0]
            ):
                global_t = _np.asarray(self.time)
                # Match each t_src entry to its index in the global
                # vector via searchsorted (both are monotonic).
                idx = _np.searchsorted(global_t, t_src)
                # Clamp in case of floating-point drift.
                idx = _np.clip(idx, 0, global_t.shape[0] - 1)
                y_src = y_src[idx]
            t_min, t_max = float(t_src[0]), float(t_src[-1])
            if _np.any(t_q < t_min - 1e-12) or _np.any(t_q > t_max + 1e-12):
                raise ValueError(
                    f"SimulationResults.align: query times out of range "
                    f"for signal {name!r} (covered [{t_min}, {t_max}])."
                )
            if y_src.ndim == 1:
                new_outputs[name] = _jnp.interp(time_vector, t_src, y_src)
            else:
                new_outputs[name] = _jnp.stack(
                    [_jnp.interp(time_vector, t_src, y_src[:, i])
                     for i in range(y_src.shape[1])],
                    axis=-1,
                )

        return SimulationResults(
            context=self.context,
            time=time_vector,
            outputs=new_outputs,
            parameters=self.parameters,
            per_signal_times=None,
            solver_states=self.solver_states,
            provenance=self.provenance,
            dae_drift_trace=self.dae_drift_trace,
            event_times=self.event_times,
        )

    def query(self, t, signal: Optional[str] = None):
        """Interpolate recorded signal(s) at time ``t`` (T-012, T-012a).

        Default path uses a linear interpolant over the recorded
        time/value arrays — fast, consistent across solvers, sufficient
        for the common post-hoc-sampling workflow.

        When the simulation was run with
        ``SimulatorOptions(record_solver_states=True)`` the
        ``solver_states`` field is populated and ``query`` switches to a
        PCHIP cubic-Hermite interpolant built from the same recorded
        samples (T-012a partial).  PCHIP is shape-preserving — no
        overshoot at zero-order-hold plateaus — and gives ~3 orders of
        magnitude better accuracy than linear on smooth (continuous)
        signals.  Discrete (zero-order-hold) signals are detected by
        constant-plateau runs and fall back to step interpolation
        rather than smoothing through the steps.

        The ODE solver's *native* dense interpolant (Dopri5's 5th-order
        polynomial, BDF's polynomial predictor) — which would give
        sub-ULP accuracy — remains a follow-up since it requires plumbing
        per-major-step solver state through the recording pipeline.

        Args:
            t: Scalar time, or 1-D array of times.
            signal: Optional signal name.  If provided, return only
                that signal's interpolated value.  If None, return a
                dict of all recorded signals.

        Returns:
            - If ``signal`` is provided: the interpolated array (scalar
              when ``t`` is scalar, 1-D otherwise).
            - Otherwise: ``dict[str, Array]`` matching ``self.outputs``.

        Raises:
            ValueError: if ``t`` falls outside ``[time[0], time[-1]]``,
                or if ``recorded_signals`` was not supplied to
                ``simulate`` (``self.outputs`` is None), or if
                ``signal`` is not in ``self.outputs``.
        """
        import jax.numpy as _jnp
        import numpy as _np

        if self.outputs is None or self.time is None:
            raise ValueError(
                "SimulationResults.query: no recorded signals.  Pass "
                "recorded_signals= to simulate() first."
            )

        t_vec = _np.asarray(self.time)
        t_arr = _np.asarray(t)

        # Bound check — a single violated endpoint fails the whole call.
        t_min, t_max = float(t_vec[0]), float(t_vec[-1])
        if _np.any(t_arr < t_min - 1e-12) or _np.any(t_arr > t_max + 1e-12):
            raise ValueError(
                f"SimulationResults.query: t out of range.  "
                f"Simulation covered [{t_min}, {t_max}]; got {t_arr!r}."
            )

        # T-012a / T-012a-followup: select interpolant.
        #   ``solver_states is None`` → linear (legacy + load-from-disk).
        #   ``"pchip"`` sentinel → PCHIP cubic-Hermite fallback.
        #   ``NativeInterpolant`` → native solver polynomial (sub-ULP).
        native_interp = (
            self.solver_states
            if isinstance(self.solver_states, NativeInterpolant)
            else None
        )
        use_pchip = self.solver_states == "pchip" and t_vec.shape[0] >= 2
        if native_interp is not None and t_vec.shape[0] >= 2:
            # PCHIP is the per-signal fallback when the native polynomial
            # doesn't match the recorded signal (e.g. a discrete output,
            # not a state passthrough).
            use_pchip = True

        def _is_zoh(col: "_np.ndarray") -> bool:
            """Detect zero-order-hold-style signals: long constant runs.

            PCHIP is shape-preserving but a discrete signal that holds
            a value across many samples and then steps is best served
            by step interpolation — PCHIP would still smooth the corner
            slightly.  Heuristic: if more than half the consecutive
            differences are exactly zero, treat as ZOH.
            """
            if col.shape[0] < 3:
                return False
            d = _np.diff(col)
            return _np.count_nonzero(d == 0) > col.shape[0] / 2

        def _native_eval(col: "_np.ndarray"):
            """T-012a-followup: evaluate the solver's polynomial at t_arr.

            Returns ``(values,)`` matching ``t_arr`` shape if the column
            is a continuous-state passthrough — values match the
            polynomial at every recorded segment endpoint to within
            float64 round-off.  Returns ``None`` otherwise so the caller
            falls back to PCHIP/linear.
            """
            ni = native_interp
            t_prev = _np.asarray(ni.t_prev)
            t_step = _np.asarray(ni.t_step)
            coeffs = _np.asarray(ni.interp_coeff)
            n_seg = t_prev.shape[0]
            if n_seg == 0:
                return None
            n_y = coeffs.shape[2]
            # End-point values per segment via polyval at theta=1.
            end_vals = _np.empty((n_seg, n_y), dtype=coeffs.dtype)
            for i in range(n_seg):
                end_vals[i] = _np.polyval(coeffs[i], 1.0)
            # Match each segment's t_step to the col index.
            seg_end_idx = _np.searchsorted(t_vec, t_step)
            seg_end_idx = _np.clip(seg_end_idx, 0, t_vec.shape[0] - 1)
            # Pick the state-component whose polynomial endpoint best
            # matches the recorded col over all segments.  If no
            # component agrees within 1e-6, the col isn't a state
            # passthrough — abort.
            recorded = col[seg_end_idx]
            best_comp = -1
            best_err = _np.inf
            for c in range(n_y):
                err = _np.max(_np.abs(end_vals[:, c] - recorded))
                if err < best_err:
                    best_err = err
                    best_comp = c
            if best_err > 1e-6 or best_comp < 0:
                return None
            # Locate each query time in the segments.  ``side="left"``
            # plus clip lands t == t_step[i] in segment i (good — the
            # endpoint is the polynomial's right edge).
            t_arr_1d = _np.atleast_1d(t_arr).astype(_np.float64)
            seg_idx = _np.searchsorted(t_step, t_arr_1d, side="left")
            seg_idx = _np.clip(seg_idx, 0, n_seg - 1)
            tp = t_prev[seg_idx]
            ts = t_step[seg_idx]
            dt = ts - tp
            dt = _np.where(dt == 0.0, 1.0, dt)
            theta = (t_arr_1d - tp) / dt
            # Vectorised Horner over the picked component.
            picked = coeffs[seg_idx, :, best_comp]  # (n_q, n_coeff)
            n_coeff = picked.shape[-1]
            result = _np.zeros_like(theta)
            for k in range(n_coeff):
                result = result * theta + picked[..., k]
            # Snap exact-recorded-time queries to recorded values to
            # remove residual round-off (the polynomial is a near-exact
            # interpolant but not bit-exact at the endpoints).
            t_match_idx = _np.searchsorted(t_vec, t_arr_1d)
            t_match_idx = _np.clip(t_match_idx, 0, t_vec.shape[0] - 1)
            on_recorded = _np.isclose(
                t_vec[t_match_idx], t_arr_1d, atol=1e-15, rtol=0.0,
            )
            result = _np.where(on_recorded, col[t_match_idx], result)
            return result.reshape(t_arr.shape) if t_arr.ndim > 0 else result[0]

        def _interp_column(col: "_np.ndarray") -> "_np.ndarray":
            # T-012a-followup: try the native polynomial first.
            if native_interp is not None and not _is_zoh(col):
                native_result = _native_eval(col)
                if native_result is not None:
                    return _np.asarray(native_result)
            if use_pchip and not _is_zoh(col):
                # PCHIP requires strictly-increasing x.  Recorded times
                # are monotonic-non-decreasing (zero-crossing handler
                # may inject a sample at the same instant); collapse
                # any duplicates by keeping the first.
                _, uniq_idx = _np.unique(t_vec, return_index=True)
                uniq_idx = _np.sort(uniq_idx)
                if uniq_idx.shape[0] >= 2:
                    from scipy.interpolate import PchipInterpolator
                    interp = PchipInterpolator(
                        t_vec[uniq_idx], col[uniq_idx], extrapolate=False,
                    )
                    return _np.asarray(interp(t_arr))
            # Linear fallback (legacy and ZOH path).
            return _np.asarray(_jnp.interp(t_arr, t_vec, col))

        def _interp_one(arr):
            arr = _np.asarray(arr)
            if arr.ndim == 1:
                return _jnp.asarray(_interp_column(arr))
            # vector-valued signal: interp each component
            return _jnp.stack(
                [_jnp.asarray(_interp_column(arr[:, i]))
                 for i in range(arr.shape[1])],
                axis=-1,
            )

        if signal is not None:
            if signal not in self.outputs:
                raise ValueError(
                    f"SimulationResults.query: unknown signal {signal!r}.  "
                    f"Recorded: {list(self.outputs)}"
                )
            return _interp_one(self.outputs[signal])
        return {name: _interp_one(arr) for name, arr in self.outputs.items()}

    def lazy(self):
        """Return a :class:`LazyResults` wrapper for fluent / deferred queries.

        See :mod:`jaxonomy.simulation.lazy_results` for the full API.
        """
        from .lazy_results import LazyResults
        return LazyResults.from_results(self)
