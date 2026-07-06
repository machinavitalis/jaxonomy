# SPDX-License-Identifier: MIT

"""Functionality for simulating hybrid dynamical systems.

This module provides the `simulate` function, which is the primary entry point
for running simulations.  It also defines the `Simulator` class used by `simulate`,
which provides more fine-grained control over the simulation process.
"""

from __future__ import annotations
from functools import partial
import dataclasses
import warnings
from typing import TYPE_CHECKING, Callable, Any

import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from ..logging import logger
from ..profiling import Profiler
from ..lazy_loader import LazyLoader

from .types import (
    StepEndReason,
    GuardIsolationData,
    ContinuousIntervalData,
    SimulatorOptions,
    SimulatorState,
    SimulationResults,
    ResultsOptions,
    ResultsMode,
)

from ..backend import (
    ODESolver,
    ResultsData,
    numpy_api as npa,
    set_backend,
    io_callback,
    cond,
)

from ..framework import (
    IntegerTime,
    is_event_data,
    ZeroCrossingEvent,
    flatten_diagram,
)
from ..framework.diagram import Diagram
from .. import backend
from .errors import remap_simulation_errors
from .zero_crossing_handler import (
    ZeroCrossingHandler,
    guard_interval_start,
    guard_interval_end,
    determine_triggered_guards,
)
from .results_recorder import ResultsRecorder

# T-137: modest floor for the auto-sized time-series recording buffer. The old
# default (= max_major_steps, ~200 for a continuous system) far undershot the
# number of recorded MINOR steps a moderate adaptive run produces, so the fixed
# ring buffer silently overflowed and returned a truncated tail. This floor
# keeps the common case from overflowing while staying small enough not to
# balloon memory under vmap/batch (~n_signals * 2048 * dtype per trajectory).
# Users who need more set SimulatorOptions(buffer_length=...) explicitly.
_MIN_AUTO_BUFFER_LENGTH = 2048

if TYPE_CHECKING:
    import equinox as eqx
    from ..backend.ode_solver import ODESolverBase, ODESolverState
    from ..framework import ContextBase, SystemBase
    from ..framework.port import OutputPort
    from ..framework.event import PeriodicEventData, EventCollection
else:
    eqx = LazyLoader("eqx", globals(), "equinox")


__all__ = [
    "estimate_max_major_steps",
    "simulate",
    "Simulator",
]


def _emit_dae_drift_warning(t_val, residual_val, threshold_val):
    """T-003b: ``UserWarning`` emitter for DAE constraint drift.

    Invoked from inside the jit'd ``_major_step`` via
    ``jax.debug.callback`` — that primitive lifts the call out of the
    XLA computation onto the host so ``warnings.warn`` works.  Gates
    the warning host-side so non-violating steps stay silent.
    """
    import warnings
    residual = float(residual_val)
    threshold = float(threshold_val)
    if residual > threshold:
        warnings.warn(
            f"DAE constraint residual {residual:.3e} exceeds "
            f"threshold {threshold:.3e} at t={float(t_val):.4f}. "
            f"Set SimulatorOptions(dae_projection_enabled=True) to correct.",
            UserWarning,
            stacklevel=2,
        )


class _BDFConditionMonitor:
    """T-038a-followup-bdf-condition-check: Python-side aggregator for
    BDF Newton-iteration condition-number estimates.

    The BDF solver forwards a ``cond(M - c*J)`` estimate plus the
    current time at the end of every ``newton_iteration`` call via
    ``jax.debug.callback``.  This class accumulates the running max
    along with the time at which it occurred.  At the end of
    ``simulate``, ``maybe_warn`` emits ONE ``UserWarning`` if the max
    exceeded the threshold — deliberately aggregated rather than
    per-step (a per-step warning would be too noisy on stiff systems).

    Lives on the host side, so the running max is plain Python state
    and does not need to be threaded through the JAX trace.  The
    ``update`` method is invoked from inside the JIT'd simulator via
    ``jax.debug.callback``, which lifts the call onto the host where
    Python mutation is fine.
    """

    __slots__ = ("threshold", "max_cond", "t_at_max", "n_samples")

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        # Seed with -inf so any finite estimate replaces it; t_at_max
        # is NaN until the first sample lands.
        self.max_cond: float = float("-inf")
        self.t_at_max: float = float("nan")
        self.n_samples: int = 0

    def update(self, cond_val, t_val) -> None:
        """Host-side callback target.  Updates the running max."""
        try:
            cval = float(cond_val)
        except (TypeError, ValueError):
            return
        if not (cval == cval):  # NaN
            return
        self.n_samples += 1
        if cval > self.max_cond:
            self.max_cond = cval
            try:
                self.t_at_max = float(t_val)
            except (TypeError, ValueError):
                self.t_at_max = float("nan")

    def maybe_warn(self) -> None:
        """Emit one ``UserWarning`` if the max exceeded the threshold."""
        import warnings
        if self.max_cond > self.threshold and self.max_cond != float("-inf"):
            warnings.warn(
                f"BDF Newton-iteration condition number "
                f"{self.max_cond:.3e} exceeds threshold "
                f"{self.threshold:.3e} (max observed at "
                f"t={self.t_at_max:.4f}).  The Newton system "
                f"M - c*J is ill-conditioned; consider tightening "
                f"rtol/atol, refining the model, or using a finer "
                f"index reduction.",
                UserWarning,
                stacklevel=2,
            )


class _DAEDriftMonitor:
    """T-113 Phase 1: Python-side accumulator for per-major-step DAE
    constraint drift samples.

    Each call to ``update`` appends ``(time, residual)`` to a pair of
    growing lists.  Invoked from inside the JIT'd ``_major_step`` via
    :func:`jax.debug.callback`, which lifts the call onto the host where
    Python list mutation is safe.  At the end of ``simulate``,
    :meth:`finalize` returns the captured arrays as a dict ready to
    attach to ``SimulationResults.dae_drift_trace``.

    Mirrors :class:`_BDFConditionMonitor` — same host-side aggregator
    pattern, different aggregation (full trace vs. running max).  Lives
    on the host side so the trace storage is plain Python state and
    does not need to be threaded through the JAX trace.

    Default-off: the monitor is constructed only when
    ``SimulatorOptions.record_dae_drift=True``; the simulator's
    ``_major_step`` skips the entire trace block at trace time when no
    monitor is attached, so the path is byte-equivalent to the
    pre-T-113 hot path.
    """

    __slots__ = ("times", "residuals")

    def __init__(self):
        self.times: list[float] = []
        self.residuals: list[float] = []

    def update(self, t_val, residual_val) -> None:
        """Host-side callback target.  Appends one ``(time, residual)``."""
        try:
            tval = float(t_val)
            rval = float(residual_val)
        except (TypeError, ValueError):
            return
        self.times.append(tval)
        self.residuals.append(rval)

    def finalize(self) -> dict | None:
        """Return ``{"time": ndarray, "residual": ndarray}`` or None."""
        if not self.times:
            return None
        return {
            "time": np.asarray(self.times, dtype=float),
            "residual": np.asarray(self.residuals, dtype=float),
        }


class _EventTimeRecorder:
    """T-125-followup-record-event-times: Python-side accumulator for
    zero-crossing event firing times.

    Each call to :meth:`update` receives ``(t_event, per_event_mask)``
    from inside the JIT'd ``_advance_continuous_time`` via
    :func:`jax.debug.callback`.  The mask is a length-``N_events`` bool
    array — a ``True`` slot at index ``i`` means event ``i`` fired at
    ``t_event``.  The recorder appends ``t_event`` to its per-event
    bucket for every ``True`` slot, ignoring no-op (no-event) calls so
    non-triggering major steps add nothing.

    Mirrors :class:`_DAEDriftMonitor` — same host-side aggregator
    pattern, just keyed by event index instead of a flat trace.  Lives
    on the host side so the storage is plain Python state and does not
    need to be threaded through the JAX trace.

    Default-off: the recorder is constructed only when
    ``SimulatorOptions.record_event_times=True`` AND the diagram has at
    least one zero-crossing event.  ``_advance_continuous_time`` skips
    the entire callback at trace time when no recorder is attached, so
    the path is byte-equivalent to the pre-followup hot path.
    """

    __slots__ = ("n_events", "times")

    def __init__(self, n_events: int):
        self.n_events = int(n_events)
        # One growing list per event index — keeps each event's
        # firing-time array independent, including events that never
        # fire (yield empty arrays).
        self.times: list[list[float]] = [[] for _ in range(self.n_events)]

    def update(self, t_val, mask) -> None:
        """Host-side callback target.

        ``t_val`` is the event firing time (a JAX scalar, possibly the
        end-of-major-step time when nothing fired — ignored in that
        case via the all-False mask check).  ``mask`` is a length-
        ``n_events`` bool array; each ``True`` slot adds ``t_val`` to
        its bucket.
        """
        try:
            mask_arr = np.asarray(mask).reshape(-1)
        except Exception:
            return
        if mask_arr.shape[0] == 0 or not bool(mask_arr.any()):
            return
        try:
            tval = float(t_val)
        except (TypeError, ValueError):
            return
        for i in range(min(self.n_events, mask_arr.shape[0])):
            if bool(mask_arr[i]):
                self.times[i].append(tval)

    def finalize(self) -> dict | None:
        """Return ``{event_index: ndarray}`` of firing times.

        Returns ``None`` when no events were recorded at all (so a
        diagram that simply never triggers any event still surfaces a
        per-event-index dict of empty arrays — the caller can tell
        "option off" from "option on, no triggers" by checking the
        keys).  When at least one event fired, every event index has
        an entry; events that did not fire have an empty array.
        """
        if self.n_events == 0:
            return None
        return {
            i: np.asarray(self.times[i], dtype=float)
            for i in range(self.n_events)
        }


def error_end_time_not_reached(tf, ctx_time, reason):
    """End-time-not-reached diagnostic.

    Historically raised ``RuntimeError`` via ``jax.debug.callback`` when the
    simulator's major-step budget ran out before reaching ``tf``.  The
    callback was an IO effect inside the simulator's main ``lax.cond`` and
    broke ``simulate_batch(use_vmap=True)`` with ``NotImplementedError: IO
    effect not supported in vmap-of-cond`` (T-002b).

    The callback has been removed.  Users now see the same symptom
    (``ctx_time < tf``) in the returned :class:`SimulationResults` via
    ``results.context.time``; the calling code can compare that to ``tf``
    and raise explicitly if desired.  The vmap path compiles cleanly.

    Signature preserved for call-site compatibility.
    """
    del tf, ctx_time, reason  # unused; kept for signature compatibility


def error_end_time_not_representable(tf, max_tf):
    """Integer-time-overflow diagnostic.

    Plain Python ``raise`` when ``tf`` is a concrete value; silently
    skipped when ``tf`` is a JAX tracer (the caller is running
    ``simulate`` inside a higher-level ``jit`` / ``grad`` / ``vmap``,
    so the bound is unobservable here and any overflow is a runtime
    issue the caller owns).  T-002b's removal of the in-loop
    ``jax.debug.callback`` path remains in effect.
    """
    import jax
    if isinstance(tf, jax.core.Tracer):
        return  # traced context — bound is unknown until runtime
    if float(tf) > float(max_tf):
        from ..framework.event import IntegerTime
        required_scale = float(tf) / float(max_tf)
        current_scale = IntegerTime.time_scale
        raise RuntimeError(
            f"Requested end time {tf} is greater than max representable time "
            f"{max_tf}.  Increase the time scale by setting `int_time_scale` "
            f"in `SimulatorOptions`. Current time scale is {current_scale}, "
            f"but this end time requires at least time_scale="
            f"{current_scale * required_scale}.  The default value of 1e-12 "
            "(picosecond precision) is only capable of representing times up "
            "to ~0.3 years."
        )


def _resolve_int_time_scale(int_time_scale, t_span) -> None:
    """Apply the requested integer-time scale, resolving the ``"auto"`` mode.

    ``"auto"`` (the default) picks the finest power-of-ten scale that
    represents ``t_span[1]`` with ~100x headroom: short simulations keep
    picosecond resolution, long horizons coarsen automatically so a
    multi-year run no longer raises a representability error
    (T-B6-followup-int-time-scale-auto). ``None`` leaves the global scale
    untouched (legacy). A float pins it explicitly.
    """
    from ..framework.event import IntegerTime, DEFAULT_TIME_SCALE

    if int_time_scale is None:
        return
    if isinstance(int_time_scale, str):
        if int_time_scale != "auto":
            raise ValueError(
                f"SimulatorOptions.int_time_scale must be 'auto', a float, or "
                f"None; got {int_time_scale!r}."
            )
        import math
        tf = t_span[1]
        if isinstance(tf, jax.core.Tracer):
            # End time unknown at trace time — keep the default scale; if it
            # overflows at runtime that is the caller's to size explicitly.
            IntegerTime.set_scale(DEFAULT_TIME_SCALE)
            return
        tf_abs = abs(float(tf))
        max_int = float(IntegerTime.max_int_time)
        safety = 100.0
        needed = tf_abs * safety / max_int
        if needed <= DEFAULT_TIME_SCALE or tf_abs == 0.0:
            scale = DEFAULT_TIME_SCALE
        else:
            scale = 10.0 ** math.ceil(math.log10(needed))
        IntegerTime.set_scale(scale)
        return
    # Explicit numeric scale.
    IntegerTime.set_scale(float(int_time_scale))


def estimate_max_major_steps(
    system: SystemBase,
    tspan: tuple[float, float],
    max_major_step_length: float = None,
    safety_factor: int = 2,
) -> int:
    """Heuristic for estimating the required number of major steps.

    This is used to bound the number of iterations in the while loop in the
    `simulate` function when automatic differentiation is enabled.  The number
    of major steps is determined by the smallest discrete period in the system
    and the length of the simulation interval.  The number of major steps is
    bounded by the length of the simulation interval divided by the smallest
    discrete period, with a safety factor applied.  The safety factor accounts
    for unscheduled major steps that may be triggered by zero-crossing events.

    This function assumes static time variables, so cannot be called from within
    traced (JAX-transformed) functions.  This is typically the case when the
    beginning or end time of the simulation is a variable that will be
    differentiated.  In this case `estimate_max_major_steps` should be called
    statically ahead of time to determine a reasonable bound for `max_major_steps`.

    Args:
        system (SystemBase): The system to simulate.
        tspan (tuple[float, float]): The time interval to simulate over.
        max_major_step_length (float, optional): The maximum length of a major
            step. If provided, this will be used to bound the number of major
            steps. Otherwise it will be ignored.
        safety_factor (int, optional): The safety factor to apply to the number of
            major steps.  Defaults to 2.
    """
    # For autodiff of jaxonomy.simulate, this path is not possible, JAX
    # throws an error. To work around this, create:
    #   options = SimulatorOptions(max_major_steps=<my value>)
    # outside jaxonomy.simulate, and pass in like this:
    #   jaxonomy.simulate(my_model, options=options)

    # Find the smallest period amongst the periodic events of the system
    if system.periodic_events.has_events or max_major_step_length is not None:
        # Initialize to infinity - will be overwritten by at least one conditional
        min_discrete_step = np.inf

        # Bound the number of major steps based on the smallest discrete period in
        # the system.
        if system.periodic_events.has_events:
            event_periods = jax.tree_util.tree_map(
                lambda event_data: event_data.period,
                system.periodic_events,
                is_leaf=is_event_data,
            )
            min_discrete_step = jax.tree_util.tree_reduce(min, event_periods)

        # Also bound the number of major steps based on the max major step length
        # in case that is shorter than any of the update periods.
        if max_major_step_length is not None:
            min_discrete_step = min(min_discrete_step, max_major_step_length)

        # in this case, we assume that, on average, major steps triggered by
        # zero crossing event, will be as frequent or less frequent than major steps
        # triggered by the smallest discrete period.
        # anything less than 100 is considered inadequate. user can override if they want this.
        max_major_steps = max(100, safety_factor * int(tspan[1] // min_discrete_step))
        logger.info(
            "max_major_steps=%s based on smallest discrete period=%s",
            max_major_steps,
            min_discrete_step,
        )
    else:
        # in this case we really have no valuable information on which to make an
        # educated guess. who knows how many events might occurr!!!
        # users will have to iterate.
        max_major_steps = 200
        logger.info(
            "max_major_steps=%s by default since no discrete period in system",
            max_major_steps,
        )
    return max_major_steps


def _check_options(
    system: SystemBase,
    options: SimulatorOptions,
    t_span: tuple[float, float],
    recorded_signals: dict[str, OutputPort],
) -> SimulatorOptions:
    """Check consistency of options and adjust settings where necessary."""

    if options is None:
        options = SimulatorOptions()

    # Check based on the options and the system whether JAX tracing is possible.
    math_backend, enable_tracing = _check_backend(options)

    # If we specified JAX but tracing is not enabled, we have fall back to numpy
    # (T-002-followup-tracing-downgrade-warn). Previously this was a quiet
    # ``logger.warning`` that the user never saw under default log config, so
    # they'd hit a misleading downstream error like "Invalid method 'dopri5'
    # for SciPy ODE solver" without realising the backend had been swapped.
    # Routing the message through ``warnings.warn`` makes it visible at
    # default warning levels and gives the user the cause-of-swap up front.
    if (math_backend == "jax") and not enable_tracing:
        warnings.warn(
            "SimulatorOptions(math_backend='jax', enable_tracing=False): JAX "
            "requires tracing, so the backend has been swapped to numpy. If "
            "you set an ode_solver_method like 'dopri5' it will fail "
            "validation downstream (scipy's methods are 'RK45', 'DOP853', "
            "etc.). Set math_backend='numpy' explicitly to silence this "
            "warning, or leave enable_tracing=True to keep the JAX backend.",
            UserWarning,
            stacklevel=2,
        )
        enable_tracing = False
        math_backend = "numpy"

    # Set the global numerical backend as determined by the options and above logic.
    set_backend(math_backend)

    if recorded_signals is None:
        recorded_signals = options.recorded_signals
    save_time_series = recorded_signals is not None

    # The while loop must be bounded in order for reverse-mode autodiff to work.
    # Also need this to set buffer sizes for signal recording in compiled JAX.
    # For the NumPy backend, this will be ignored, since neither bounded while
    # loops nor buffered recording is necessary.
    # Track whether the caller explicitly provided a step budget.
    # When True, the bounded fori_loop is used even without enable_autodiff so that
    # max_major_steps acts as a hard cap (Zeno protection, fixed-budget simulations).
    explicit_max_major_steps = (
        options.max_major_steps is not None and options.max_major_steps > 0
    )
    max_major_steps = options.max_major_steps
    if not explicit_max_major_steps:
        # logger.warning(
        #     "JAX backend requires a bounded number of major steps. This has not "
        #     "been specified in SimulatorOptions. Using a heuristic to estimate "
        #     "the maximum number of steps. If this fails, it may be because the "
        #     "final time is a traced variable.  If it is necessary to "
        #     "differentiate with respect to the end time of the simulation, then "
        #     "max_major_steps must be set manually. A reasonable value can be "
        #     "estimated using estimate_max_major_steps."
        # )
        max_major_steps = estimate_max_major_steps(
            system, t_span, options.max_major_step_length
        )

    buffer_length = options.buffer_length
    if buffer_length is None:
        buffer_length = max(max_major_steps, _MIN_AUTO_BUFFER_LENGTH)

    # Check that the options are configured correctly for autodiff.
    if options.enable_autodiff:
        # JAX tracing is required for automatic differentiation
        if not enable_tracing:
            raise ValueError(
                "Autodiff is only supported with `options.enable_tracing=True`."
            )

        # Cannot record time series during autodiff - only final results can
        # be differentiated
        if save_time_series:
            raise ValueError(
                "Recording output time series is not supported with autodiff."
            )

    # Rescale integer time to fit the simulation horizon.  ``int_time_scale``
    # defaults to "auto", which picks the finest power-of-ten scale that
    # represents ``t_span[1]`` with headroom — short sims keep picosecond
    # resolution, long horizons coarsen automatically (T-B6-followup-int-time-
    # scale-auto).  A float pins the scale; None leaves the global untouched.
    _resolve_int_time_scale(options.int_time_scale, t_span)
    error_end_time_not_representable(t_span[1], IntegerTime.max_float_time)

    return dataclasses.replace(
        options,
        recorded_signals=recorded_signals,
        save_time_series=save_time_series,
        max_major_steps=max_major_steps,
        math_backend=math_backend,
        enable_tracing=enable_tracing,
        buffer_length=buffer_length,
        _explicit_max_major_steps=explicit_max_major_steps,
    )


@remap_simulation_errors
def simulate(
    system: SystemBase,
    context: ContextBase,
    t_span: tuple[float, float] = None,
    options: SimulatorOptions = None,
    results_options: ResultsOptions = None,
    recorded_signals: dict[str, OutputPort] = None,
    postprocess: bool = True,
    flatten: bool = False,
    *,
    tspan: tuple[float, float] = None,
) -> SimulationResults:
    """Simulate the hybrid dynamical system defined by `system`.

    The parameters and initial state are defined by `context`.  The simulation time
    runs from `tspan[0]` to `tspan[1]`.

    The simulation is "hybrid" in the sense that it handles dynamical systems with both
    discrete and continuous components.  The continuous components are integrated using
    an ODE solver, while discrete components are updated periodically as specified by
    the individual system components. The continuous and discrete states can also be
    modified by "zero-crossing" events, which trigger when scalar-valued guard
    functions cross zero in a specified direction.

    The simulation is thus broken into "major" steps, which consist of the following,
    in order:

    (1) Perform any periodic updates to the discrete state.
    (2) Check if the discrete update triggered any zero-crossing events and handle
        associated reset maps if necessary.
    (3) Advance the continuous state using an ODE solver until the next discrete
        update or zero-crossing, localizing the zero-crossing with a bisection search.
    (4) Store the results data.
    (5) If the ODE solver terminated due to a zero-crossing, handle the reset map.

    The steps taken by the ODE solver are "minor" steps in this simulation.  The
    behavior of the ODE solver and the hybrid simulation in general can be controlled
    by configuring `SimulatorOptions`.  Available settings are as follows:

    SimulatorOptions:
        enable_tracing (bool): Allow JAX tracing for JIT compilation
        max_major_step_length (float): Maximum length of a major step
        max_major_steps (int):
            The maximum number of major steps to take in the simulation. This is
            necessary for automatic differentiation - otherwise the "while" loop
            is non-differentiable.  With the default value of None, a heuristic
            is used to determine the maximum number of steps based on the periodic
            update events and time interval.
        rtol (float): Relative tolerance for the ODE solver. Default is 1e-6.
        atol (float): Absolute tolerance for the ODE solver. Default is 1e-8.
        min_minor_step_size (float): Minimum step size for the ODE solver.
        max_minor_step_size (float): Maximum step size for the ODE solver.
        ode_solver_method (str): The DE solver to use.  Default is "auto", which
            will use the Dopri5/Jax if JAX tracing is enabled, otherwise the
            SciPy Dopri5 solver.
        save_time_series (bool):
            This option determines whether the simulator saves any data.  If the
            simulation is initiated from `simulate` this will be set automatically
            depending on whether `recorded_signals` is provided.  Hence, this
            should not need to be manually configured.
        recorded_signals (dict[str, OutputPort]):
            Dictionary of ports or other cache sources for which the time series should
            be recorded. Note that if the simulation is initiated from `simulate` and
            `recorded_signals` is provided as a kwarg to `simulate`, anything set here
            will be overridden.  Hence, this should not need to be manually configured.
        return_context (bool):
            If the context is not needed for anything, opting to not return it can
            speed up compilation times.  For instance, typical simulation calls from
            the UI don't use the context for anything, so model_interface.py will
            set `return_context=False` for performance.
        postprocess (bool):
            If using buffered results recording (i.e. with JAX numerical backend), this
            determines whether to automatically trim the buffer after the simulation is
            complete. This is the default behavior, which will serve unless the full
            call to `simulate` needs to be traced (e.g. with `grad` or `vmap`).

    The return value is a `SimulationResults` object, which is a named tuple containing
    all recorded signals as well as the final context (if `options.return_context` is
    `True`). Signals can be recorded by providing a dict of (name, signal_source) pairs
    Typically the signal sources will be output ports, but they can actually be any
    `SystemCallback` object in the system.

    Args:
        system (SystemBase): The hybrid dynamical system to simulate.
        context (ContextBase): The initial state and parameters of the system.
        tspan (tuple[float, float]): The start and end times of the simulation.
        options (SimulatorOptions): Options for the simulation process and ODE solver.
        results_options (ResultsOptions): Options related to how the outputs are
            stored, interpolated, and returned.
        recorded_signals (dict[str, OutputPort]):
            Dictionary of ports for which the time series should be recorded.

    Returns:
        SimulationResults: A named tuple containing the recorded signals and the final
            context (if `options.return_context` is `True`).

    Notes:
        If `recorded_signals` is provided as a kwarg, it will override any entry in
        `options.recorded_signals`. This will be deprecated in the future in favor of
        only passing via `options`.

        This function is meant to best handle single independent simulations.
        Calling this function repeatedly will always trigger a recompilation of the
        model when using the JAX backend. To avoid this, call advance_to directly.
    """

    # Backward-compatibility shim: accept legacy `tspan` keyword argument.
    if tspan is not None:
        if t_span is not None:
            raise TypeError("Cannot specify both 't_span' and 'tspan'; 'tspan' is deprecated.")
        import warnings
        warnings.warn(
            "The 'tspan' argument is deprecated; use 't_span' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        t_span = tspan

    if t_span is None:
        raise TypeError("simulate() missing required argument: 't_span'")

    options = _check_options(system, options, t_span, recorded_signals)

    import warnings
    from jaxonomy.framework.validation import validate_diagram
    
    if getattr(options, 'validate', True):
        result = validate_diagram(system, options)
        if result.warnings:
            for w in result.warnings:
                warnings.warn(w, UserWarning, stacklevel=2)
        result.raise_if_invalid()

    # T-105 Phase 1: opt-in multirate consistency check.  Default
    # ``None`` keeps the existing single-rate path byte-equivalent;
    # users opt in via ``SimulatorOptions.check_rate_transitions``.
    _rate_check = getattr(options, "check_rate_transitions", None)
    if _rate_check:
        from .rate_groups import detect_rate_mismatches  # local import: avoids cycle
        if isinstance(system, Diagram):
            detect_rate_mismatches(system, on_mismatch=_rate_check)

    # Optionally flatten nested Diagrams to a single depth for reduced overhead.
    if flatten and isinstance(system, Diagram):
        system = flatten_diagram(system)
        context = system.create_context(time=t_span[0])

    if results_options is None:
        results_options = ResultsOptions()

    if results_options.mode != ResultsMode.auto:
        raise NotImplementedError(
            f"Simulation output mode {results_options.mode.name} is not supported. "
            "Only 'auto' is presently supported."
        )

    if system.has_dirty_static_parameters:
        raise ValueError(
            "Some static parameters have been updated. Please create a new context."
        )

    # HACK: Jaxonomy presently does not use interpolant to produce
    # results sample between minor_step end times, so we clamp
    # the minor step size to the max_results_interval instead.
    if (
        results_options.max_results_interval is not None
        and results_options.max_results_interval > 0
        # max_minor_step_size is None by default (unbounded), which is
        # always larger than any finite results interval, so it needs
        # clamping too — guard the comparison against None either way.
        and (
            options.max_minor_step_size is None
            or results_options.max_results_interval < options.max_minor_step_size
        )
    ):
        options = dataclasses.replace(
            options,
            max_minor_step_size=results_options.max_results_interval,
        )
        logger.info(
            "max_minor_step_size reduced to %s to match max_results_interval",
            options.max_minor_step_size,
        )

    orig_x64 = jax.config.read("jax_enable_x64")
    enable_x64 = orig_x64
    if options and options.precision != "auto":
        enable_x64 = (options.precision == "float64")
    jax.config.update("jax_enable_x64", enable_x64)

    if options and options.precision != "auto":
        target_dtype = jnp.float64 if options.precision == "float64" else jnp.float32
        def cast_floats(x):
            if isinstance(x, (jax.Array, np.ndarray)):
                if jnp.issubdtype(x.dtype, jnp.floating):
                    return x.astype(target_dtype)
            return x
        context = jax.tree_util.tree_map(cast_floats, context)

    ode_solver = ODESolver(system, options=options.ode_options)

    sim = Simulator(system, ode_solver=ode_solver, options=options)
    logger.info("Simulator ready to start: %s, %s", options, ode_solver)

    # Define a function to be traced by JAX, if allowed, closing over the
    # arguments to `_simulate`.
    def _wrapped_simulate() -> tuple[ContextBase, ResultsData]:
        t0, tf = t_span
        initial_context = context.with_time(t0)
        sim_state = sim.advance_to(tf, initial_context)
        error_end_time_not_reached(
            tf, sim_state.context.time, sim_state.step_end_reason
        )
        final_context = sim_state.context if options.return_context else None
        return final_context, sim_state.results_data

    # JIT-compile the simulation, if allowed
    if options.enable_tracing:
        _wrapped_simulate = jax.jit(_wrapped_simulate)
        _wrapped_simulate = Profiler.jaxjit_profiledfunc(
            _wrapped_simulate, "_wrapped_simulate"
        )

    # Run the simulation
    try:
        system.cache_enabled = True
        final_context, results_data = _wrapped_simulate()

        if postprocess and results_data is not None:
            time, outputs = results_data.finalize()
            # The backend wrapper exposes ``finalize`` only; reach into
            # the inner ``_solution_data`` for the optional finalize
            # variants (Mode A buffers, native interpolant) when the JAX
            # backend supplied them.  Falls back to None for backends
            # that don't.
            _inner_results = getattr(results_data, "_solution_data", results_data)
            # T-012a-followup: pull the per-step interpolant ring (when
            # the buffer was allocated via ``record_solver_states=True``).
            _interpolant_finalized = None
            finalize_interp = getattr(
                _inner_results, "finalize_interpolant", None,
            )
            if finalize_interp is not None:
                _interpolant_finalized = finalize_interp()
            # T-013a-followup-mode-a-buffers: when per-signal buffers
            # were allocated (mode="buffers"), pull each signal's
            # trimmed (times, values) directly from its own ring rather
            # than reusing the legacy global trim.  This is where the
            # storage saving is realised — periodic signals' arrays are
            # already at the right cadence, no post-trim required.
            per_signal_finalized = None
            finalize_per_signal = getattr(
                _inner_results, "finalize_per_signal", None,
            )
            if finalize_per_signal is not None:
                per_signal_finalized = finalize_per_signal()
            if per_signal_finalized is not None:
                _global_t, per_outputs, per_times = per_signal_finalized
                # Keep ``time`` as the legacy global vector (some
                # downstream code expects it for ``align`` cross-
                # references); use the per-signal trimmed outputs.
                outputs = per_outputs
                # Stash the per-signal times so the post-processor at
                # the bottom of ``simulate`` can promote them to
                # ``SimulationResults.per_signal_times`` without
                # re-running the cadence classifier.
                _per_signal_times_from_buffers = per_times
            else:
                _per_signal_times_from_buffers = None
        else:
            time, outputs = None, None
            _per_signal_times_from_buffers = None
            _interpolant_finalized = None

    finally:
        system.post_simulation_finalize()
        system.cache_enabled = False
        jax.config.update("jax_enable_x64", orig_x64)

    # T-038a-followup-bdf-condition-check: emit ONE aggregated
    # ``UserWarning`` if the BDF Newton-iteration condition number ever
    # exceeded the threshold during the trajectory.  The monitor's
    # running max is updated host-side via ``jax.debug.callback`` from
    # inside the JIT'd BDF ``newton_iteration``.  ``maybe_warn`` is a
    # no-op when the option was unset or when the max stayed below
    # threshold — preserves the byte-equivalent default-off path.
    _bdf_cond_monitor = getattr(sim, "_bdf_cond_monitor", None)
    if _bdf_cond_monitor is not None:
        _bdf_cond_monitor.maybe_warn()

    # T-002b-followup-buffer-overflow-warning: when the recording
    # ring-buffer fills up during a non-vmap simulation, the original
    # IO-effect callback that warned the user was removed in T-002b
    # to make ``simulate_batch(use_vmap=True)`` compilable. That left
    # the single-simulation path silently truncating the recorded
    # time-series to the last ``buffer_length`` samples — surprising
    # to the user, who sees a plot starting mid-trajectory with no
    # diagnostic. This Python-side check runs once after the JIT'd
    # kernel returns (so it doesn't disturb vmap) and emits a clear
    # UserWarning when truncation is detected by signature.
    #
    # Detection signature: ``time[0] > t_span[0]`` by a meaningful
    # amount. The simulator's ring buffer wraps in place, and
    # ``finalize`` then trims to the live region — so an overflow
    # surfaces not as ``len(time) == buffer_length`` (the trimmed
    # tail can be much shorter) but as the *start* of the recorded
    # trajectory being well past the requested ``t_span[0]``. That
    # behaviour is unambiguous: a non-overflowed run always records
    # the first sample at ``t_span[0]``.
    if (
        time is not None
        and len(time) >= 1
        and not getattr(options, "enable_autodiff", False)
        and t_span is not None
    ):
        t0 = float(t_span[0])
        tf = float(t_span[1])
        span = max(tf - t0, 1.0)  # avoid divide-by-zero on degenerate spans
        first_recorded = float(time[0])
        # 0.1% of the integration span is the threshold below which we
        # treat ``time[0]`` as "essentially t_span[0]"; above it the
        # buffer almost certainly wrapped.
        if first_recorded - t0 > span * 1e-3:
            import warnings as _warnings

            resolved_buffer = options.buffer_length
            n_kept = len(time)
            buf_str = (
                f"buffer_length={resolved_buffer}"
                if resolved_buffer is not None
                else "buffer_length=None (auto)"
            )
            # T-B3/B8-followup-buffer-dopri5-sizing: the recorder saves one
            # sample per *minor* (accepted) solver step. Adaptive solvers
            # (Dopri5 / the "auto" default) take far more minor steps than the
            # major-step count the auto-size is derived from — and they take
            # *more* minor steps as rtol/atol tighten. So the auto buffer
            # (sized to the major-step estimate) can be overrun by a tight-
            # tolerance Dopri5 run, silently dropping the trajectory head.
            # Make the recommendation concrete and solver/tolerance-aware.
            solver_method = getattr(options, "ode_solver_method", "auto")
            rtol = getattr(options, "rtol", None)
            atol = getattr(options, "atol", None)
            # The buffer held at least the kept-tail samples; the full run
            # needed more. Recommend a healthy multiple of what we know was
            # already exceeded.
            if resolved_buffer is not None:
                suggested = max(int(resolved_buffer) * 4, n_kept * 4, 4000)
            else:
                suggested = max(n_kept * 4, 4000)
            adaptive_note = ""
            if solver_method in ("auto", "dopri5", "Dopri5", "RK45", "DOP853"):
                adaptive_note = (
                    f" This is an adaptive solver "
                    f"(ode_solver_method={solver_method!r}, rtol={rtol}, "
                    f"atol={atol}); it records one sample per accepted minor "
                    f"step, and tightening rtol/atol increases the step count. "
                    f"A fixed-step solver (ode_solver_method='rk4') records a "
                    f"predictable sample count if you need a tight buffer."
                )
            _warnings.warn(
                f"jaxonomy.simulate: recording buffer overflow detected — "
                f"the returned ``results.time`` starts at "
                f"{first_recorded!r} (requested t_span[0]={t0!r}); samples "
                f"from earlier in the trajectory were overwritten because "
                f"the simulator's recording ring buffer ({buf_str}) "
                f"filled (kept the last {n_kept} samples).{adaptive_note} "
                f"Set SimulatorOptions(buffer_length={suggested}) (or larger) "
                f"to capture the full trajectory, loosen rtol/atol, or reduce "
                f"the number of recorded signals.",
                UserWarning,
                stacklevel=2,
            )

    # Reset the integer time scale to the default value in case we decreased precision
    # to reach the end time of a long simulation.  Typically this won't do anything.
    if options.int_time_scale is not None:
        IntegerTime.set_default_scale()

    # T-013a: opt-in per-signal timestamp capture.  Runs as a post-
    # processor on the trimmed numpy arrays — no schema change to the
    # recording buffer.  Two modes:
    #
    #   - Mode A (``per_signal_timestamps_mode="auto"`` or ``"schedule"``,
    #     the default when the option is enabled): classify each signal's
    #     natural cadence from its source ``OutputPort`` (continuous /
    #     periodic / default) and trim BOTH the times and the outputs of
    #     periodic signals down to the schedule.  Genuine storage savings
    #     for the per-signal arrays.  Falls back to Mode B per signal
    #     when classification is "default".
    #   - Mode B (``per_signal_timestamps_mode="diff"``): legacy value-
    #     diff dedup on times only; outputs stay at full length.
    per_signal_times = None
    if (
        getattr(options, "per_signal_timestamps", False)
        and time is not None
        and outputs is not None
    ):
        atol = getattr(options, "per_signal_timestamps_atol", 1e-12)
        mode = getattr(options, "per_signal_timestamps_mode", "auto")
        if mode == "buffers" and _per_signal_times_from_buffers is not None:
            # T-013a-followup-mode-a-buffers: in-JIT per-signal buffers
            # already produced trimmed arrays during ``finalize_per_signal``
            # — promote them straight to the result.  No post-finalize
            # classification or trim runs.
            per_signal_times = _per_signal_times_from_buffers
        elif mode == "diff":
            per_signal_times = ResultsRecorder.compute_per_signal_times(
                time, outputs, atol=atol,
            )
        else:  # "auto" or "schedule"
            classifications = ResultsRecorder.classify_signal_cadence(
                options.recorded_signals,
            )
            per_signal_times, outputs = (
                ResultsRecorder.compute_per_signal_schedule(
                    time, outputs, classifications, atol=atol,
                )
            )

    # T-012a-followup: when ``record_solver_states=True`` AND the JAX
    # backend produced per-step interpolant data, build a
    # ``NativeInterpolant`` so ``query`` evaluates the solver's own
    # polynomial at sub-ULP accuracy.  Fall back to the PCHIP sentinel
    # otherwise — preserves the T-012a partial behaviour for solvers /
    # backends that don't expose a fixed-shape ``interp_coeff``.
    solver_states_marker = None
    if (
        getattr(options, "record_solver_states", False)
        and time is not None
        and outputs is not None
    ):
        if _interpolant_finalized is not None:
            from .types import NativeInterpolant
            t_prev_arr, t_step_arr, coeff_arr = _interpolant_finalized
            if t_prev_arr.shape[0] >= 1:
                # Build a _StableUnravel from the final context's continuous
                # state so query() can restore the polyval result to the
                # original pytree shape.  When the final_context is None
                # (return_context=False), we still pull the original
                # context the caller passed in via ``context`` — same shape.
                try:
                    from ..backend._jax.ode_solver_impl import _StableUnravel
                    cs_template = (
                        final_context.continuous_state
                        if final_context is not None
                        else context.continuous_state
                    )
                    unravel = _StableUnravel(cs_template)
                except Exception:
                    unravel = None
                solver_states_marker = NativeInterpolant(
                    t_prev=np.asarray(t_prev_arr),
                    t_step=np.asarray(t_step_arr),
                    interp_coeff=np.asarray(coeff_arr),
                    unravel=unravel,
                    solver="dopri5",
                )
            else:
                solver_states_marker = "pchip"
        else:
            solver_states_marker = "pchip"

    # T-110 Phase 1: opt-in provenance/reproducibility manifest.
    # Computed entirely in Python after the JIT-traced kernel returns —
    # never inside ``_wrapped_simulate`` — so the default-off path is
    # byte-equivalent.  See :mod:`jaxonomy.simulation.provenance`.
    provenance_manifest = None
    if getattr(options, "record_provenance", False):
        from .provenance import compute_provenance
        provenance_manifest = compute_provenance(system, options)

    # T-113 Phase 1: opt-in per-major-step DAE drift trace.  Pull the
    # captured (time, residual) lists from the host-side monitor and
    # promote to numpy arrays.  ``finalize`` returns ``None`` when no
    # samples were collected (e.g. zero major steps, or pure-ODE
    # system).  Default-off (no monitor attached) leaves the field
    # ``None`` — byte-equivalent.
    dae_drift_trace = None
    _dae_drift_monitor = getattr(sim, "_dae_drift_monitor", None)
    if _dae_drift_monitor is not None:
        dae_drift_trace = _dae_drift_monitor.finalize()

    # T-125-followup-record-event-times: pull captured ``(event_index,
    # firing_time)`` samples from the host-side recorder.  ``finalize``
    # returns a dict keyed by event index (numpy arrays of times) when
    # the recorder was constructed, or ``None`` when no recorder is
    # attached (default-off path or zero-event diagrams).  Byte-
    # equivalent default-off behaviour: option unset → no recorder →
    # ``event_times is None``.
    event_times = None
    _event_time_recorder = getattr(sim, "_event_time_recorder", None)
    if _event_time_recorder is not None:
        event_times = _event_time_recorder.finalize()

    return SimulationResults(
        final_context,
        time=time,
        outputs=outputs,
        per_signal_times=per_signal_times,
        solver_states=solver_states_marker,
        provenance=provenance_manifest,
        dae_drift_trace=dae_drift_trace,
        event_times=event_times,
    )


def simulate_jacfwd(
    system: SystemBase,
    context_fn: Callable[..., ContextBase],
    t_span: tuple[float, float],
    params: Any,
    output_fn: Callable[[Any], Any] = None,
    *,
    options: SimulatorOptions = None,
    record_provenance: bool = False,
):
    """Forward-mode Jacobian of a simulation w.r.t. parameters (T-100).

    Wraps ``jax.jacfwd`` over a parametrised simulation. Use this when the
    parameter count is small compared to the output count
    (``n_params < n_outputs / 5`` is a useful heuristic) — forward-mode
    AD scales linearly with input dim; reverse-mode (``jax.grad`` /
    ``jax.jacrev``) scales with output dim.

    The implementation uses ``enable_autodiff=False`` to bypass the
    custom-VJP ``simulate`` defines for reverse-mode (custom_vjp blocks
    forward-mode trace with a clear ``TypeError``); the underlying
    simulator's natural JAX trace carries the tangent. Forward-mode
    plumbing is already exercised internally by ``linearize``, the BDF
    Jacobian solve, and the Kalman/EKF blocks — this function exposes
    that plumbing as a stable public surface.

    Args:
        system: a Diagram or LeafSystem to simulate.
        context_fn: a callable ``context_fn(params) -> Context`` that
            constructs an initial context with the given parameter
            pytree applied.
        t_span: ``(t0, tf)`` simulation interval.
        params: parameter pytree (the differentiation argument).
        output_fn: callable applied to the final ``Context`` to produce
            a scalar or array output. Defaults to extracting the final
            continuous state.
        options: ``SimulatorOptions``. ``enable_autodiff`` is forced to
            False for the JVP path; pass ``rtol`` / ``atol`` /
            ``ode_solver_method`` to control accuracy.
        record_provenance: when ``True``, return ``(jacobian, manifest)``
            with a populated :class:`~jaxonomy.simulation.provenance.ProvenanceManifest`
            describing the run.  Default ``False`` keeps the historical
            single-return contract byte-equivalent.  The manifest is
            computed in plain Python around the :func:`jax.jacfwd` call
            — never inside the trace — so the default-off path adds
            zero work.  See T-110-followup-attach-on-jacfwd.

    Returns:
        ``J = ∂output/∂params`` with shape determined by
        ``jax.jacfwd``'s output convention (output × params).  When
        ``record_provenance=True``, returns ``(J, manifest)`` instead.

    Example:
        >>> def make_ctx(a):
        ...     ctx = sys.create_context()
        ...     ctx.parameters['a'] = a
        ...     return ctx
        >>> J = simulate_jacfwd(sys, make_ctx, (0., 2.), jnp.array(1.5))
        >>> J, m = simulate_jacfwd(sys, make_ctx, (0., 2.), jnp.array(1.5),
        ...                         record_provenance=True)
    """
    if options is None:
        options = SimulatorOptions()
    options = dataclasses.replace(options, enable_autodiff=False)

    if output_fn is None:
        def output_fn(ctx):
            return ctx.continuous_state

    def _fwd(p):
        ctx = context_fn(p)
        res = simulate(system, ctx, t_span=t_span, options=options)
        return output_fn(res.context)

    jacobian = jax.jacfwd(_fwd)(params)

    # T-110-followup-attach-on-jacfwd: when opt-in, compute the
    # provenance manifest in plain Python around the jacfwd call and
    # return it as the second element of a tuple.  ``simulate_jacfwd``'s
    # natural return type is a JAX array (or pytree of arrays) — it
    # has no ``.provenance`` field to attach to, so the tuple shape is
    # the cleanest stable surface.  The Python-side ``record_provenance``
    # kwarg never reaches the traced ``_fwd`` body so the default-off
    # path is byte-equivalent to the pre-followup code.
    if record_provenance:
        from .provenance import compute_provenance
        # Force a Python-side option snapshot that reflects what was
        # actually requested for provenance purposes — the inner
        # ``options`` we passed to ``simulate`` has ``enable_autodiff``
        # already flipped to False, which is the value we want to
        # record (it's what produced the result).  We also flip
        # ``record_provenance`` to True on the snapshot so an auditor
        # can tell the manifest was opt-in (mirrors simulate_batch).
        options_for_manifest = dataclasses.replace(
            options, record_provenance=True,
        )
        manifest = compute_provenance(system, options_for_manifest)
        return jacobian, manifest

    return jacobian


def scalar_cost_simulate(
    system: SystemBase,
    context_fn: Callable[[Any], ContextBase],
    t_span: tuple[float, float],
    params: Any,
    cost_fn: Callable[[ContextBase], Any] = None,
    *,
    options: SimulatorOptions = None,
    return_grad: bool = False,
):
    """Reverse-mode differentiable scalar cost from a simulation (T-A1).

    Resolves the most common autodiff friction in jaxonomy: you cannot
    record a trajectory and reduce it to a cost under ``jax.grad``, because
    ``enable_autodiff=True`` forbids ``save_time_series=True`` (recording is
    not ``vmap``/AD-safe). The supported pattern is to **accumulate the cost
    inside the diagram** — e.g. add an ``Integrator`` whose input is the
    running cost ``L(t, x, u)`` — and read the final accumulated value off
    the context at ``t_span[1]``. This helper packages that pattern so the
    canonical ``cost = f(params)`` / ``grad = jax.grad(f)(params)`` workflow
    works out of the box.

    It is the reverse-mode counterpart to :func:`simulate_jacfwd`: use this
    for a *scalar* objective (optimisation / tuning), and ``simulate_jacfwd``
    for a Jacobian when ``n_params`` is small relative to the output size.

    Args:
        system: the Diagram / LeafSystem to simulate.
        context_fn: ``context_fn(params) -> Context`` building the initial
            context with ``params`` applied (e.g. via
            ``diagram.with_parameters(...).create_context()`` or by setting
            ``ctx.parameters``). Differentiation flows through this.
        t_span: ``(t0, tf)`` simulation interval.
        params: parameter pytree — the differentiation argument.
        cost_fn: ``cost_fn(final_context) -> scalar`` reducing the final
            context to the objective (typically reading the accumulated-cost
            state slot, e.g. ``lambda ctx: ctx[acc.system_id].continuous_state[0]``).
            Defaults to ``sum(final continuous_state)`` with a note that you
            almost always want to supply your own.
        options: ``SimulatorOptions``. ``enable_autodiff`` is forced ``True``
            and ``recorded_signals`` is cleared (recording is incompatible
            with AD). Set ``max_major_steps`` for systems with many events or
            when differentiating w.r.t. ``tf``.
        return_grad: when ``True``, return ``(value, grad)`` via
            ``jax.value_and_grad``; otherwise return just the scalar value
            (compose your own ``jax.grad`` / ``jax.value_and_grad`` over a
            ``lambda p: scalar_cost_simulate(...)`` closure).

    Returns:
        The scalar cost, or ``(value, grad)`` when ``return_grad=True``.

    Example:
        >>> # `acc` is an Integrator accumulating the running cost inside the diagram
        >>> def make_ctx(theta):
        ...     return diagram.with_parameters({"ctrl.kp": theta}).create_context()
        >>> cost = lambda ctx: ctx[acc.system_id].continuous_state[0]
        >>> f = lambda th: scalar_cost_simulate(diagram, make_ctx, (0., 5.), th, cost)
        >>> J = jax.grad(f)(jnp.array(1.0))            # doctest: +SKIP
        >>> val, grad = scalar_cost_simulate(diagram, make_ctx, (0., 5.),
        ...                                  jnp.array(1.0), cost, return_grad=True)
    """
    if options is None:
        options = SimulatorOptions()
    # Recording is incompatible with autodiff; force the supported config.
    options = dataclasses.replace(
        options, enable_autodiff=True, recorded_signals=None,
    )

    if cost_fn is None:
        def cost_fn(ctx):
            import jax.numpy as _jnp
            return _jnp.sum(_jnp.asarray(ctx.continuous_state))

    def _cost(p):
        ctx = context_fn(p)
        res = simulate(system, ctx, t_span=t_span, options=options)
        return cost_fn(res.context)

    if return_grad:
        return jax.value_and_grad(_cost)(params)
    return _cost(params)


class Simulator:
    """Class for orchestrating simulations of hybrid dynamical systems.

    See the `simulate` function for more details.
    """

    def __init__(
        self,
        system: SystemBase,
        ode_solver: ODESolverBase = None,
        options: SimulatorOptions = None,
    ):
        """Initialize the simulator.

        Args:
            system (SystemBase): The hybrid dynamical system to simulate.
            ode_solver (ODESolverBase):
                The ODE solver to use for integrating the continuous-time component
                of the system.  If not provided, a default solver will be used.
            options (SimulatorOptions):
                Options for the simulation process.  See `simulate` for details.
        """
        self.system = system

        if options is None:
            options = SimulatorOptions()

        # Determine whether JAX tracing can be used (jit, grad, vmap, etc)
        math_backend, self.enable_tracing = _check_backend(options)

        # Set the math backend
        set_backend(math_backend)

        # Should the simulation be run with autodiff enabled?  This will override
        # the `advance_to` method with a custom autodiff rule.
        self.enable_autodiff = options.enable_autodiff

        if ode_solver is None:
            ode_solver = ODESolver(system, options=options.ode_options)

        # Store configuration options
        self.max_major_steps = options.max_major_steps
        # Honor max_major_steps as a hard cap whenever it was explicitly provided
        # (either via _check_simulate_options or directly by the caller).
        self._explicit_max_major_steps = options._explicit_max_major_steps or (
            options.max_major_steps is not None and options.max_major_steps > 0
        )
        self.max_major_step_length = options.max_major_step_length
        self.zc_bisection_loop_count = options.zc_bisection_loop_count
        self.major_step_callback = options.major_step_callback

        # T-003a: opt-in DAE constraint projection at the end of each major step.
        self.dae_projection_enabled = getattr(
            options, "dae_projection_enabled", False,
        )
        self.dae_projection_tol = getattr(options, "dae_projection_tol", 1e-8)
        self.dae_projection_max_iter = getattr(
            options, "dae_projection_max_iter", 3,
        )

        # T-113-followup-event-reprojection: opt-in projection immediately
        # after discrete-event resets *within* a major step (top-of-step
        # ``_handle_discrete_update`` and triggered ZC resets inside
        # ``_advance_continuous_time``).  Default ``False`` so the hot
        # path is byte-equivalent to the pre-followup code.  Reuses the
        # T-003a tolerance / max-iter knobs to avoid surface-area churn.
        self.dae_reproject_after_events = getattr(
            options, "dae_reproject_after_events", False,
        )

        # T-003b: opt-in DAE drift threshold (post-projection check).
        # ``None`` (default) disables the check entirely — no overhead.
        self.dae_drift_threshold = getattr(
            options, "dae_drift_threshold", None,
        )

        # T-113 Phase 1: opt-in per-major-step DAE drift trace.
        # ``False`` (default) disables the trace entirely — no monitor
        # constructed and the simulator's ``_major_step`` skips the
        # trace block at trace time.  When True AND the system has a
        # mass matrix, attach a host-side ``_DAEDriftMonitor`` so the
        # trace block forwards each major step's ``(time, residual)``
        # via ``jax.debug.callback``.  Non-DAE systems get no monitor
        # (the diagnostic is mass-matrix-specific by definition).
        self.record_dae_drift = getattr(options, "record_dae_drift", False)
        self._dae_drift_monitor: _DAEDriftMonitor | None = None
        if self.record_dae_drift and getattr(
            self.system, "has_mass_matrix", False,
        ):
            self._dae_drift_monitor = _DAEDriftMonitor()

        # T-125-followup-record-event-times: opt-in capture of zero-
        # crossing event firing times.  Construct a host-side recorder
        # only when the option is True AND the diagram has at least one
        # zero-crossing event — diagrams with no events get no recorder
        # so ``_advance_continuous_time`` short-circuits the callback at
        # trace time, preserving the byte-equivalent default-off path.
        # ``n_zero_crossing_events`` is set further down in __init__ but
        # the snapshot below makes the count available without re-
        # importing the system; we settle for re-counting cheaply here
        # to avoid reordering the existing init blocks.
        self.record_event_times = getattr(options, "record_event_times", False)
        self._event_time_recorder: _EventTimeRecorder | None = None
        if self.record_event_times:
            n_zc = len(system.zero_crossing_events.events)
            if n_zc > 0:
                self._event_time_recorder = _EventTimeRecorder(n_zc)

        # T-038a-followup-bdf-condition-check: opt-in BDF Newton
        # condition-number diagnostic.  When the threshold is set AND
        # the active solver is a BDF solver, attach a
        # ``_BDFConditionMonitor`` to it as a side-channel — the BDF
        # solver checks ``getattr(self, "_cond_monitor", None)`` inside
        # ``newton_iteration`` and forwards the cond estimate via
        # ``jax.debug.callback`` only when set.  Default-off path is
        # byte-equivalent (no monitor → no-op in BDF).  Non-BDF
        # solvers silently ignore the option (the diagnostic is BDF-
        # specific by definition).
        self.bdf_condition_warning_threshold = getattr(
            options, "bdf_condition_warning_threshold", None,
        )
        self._bdf_cond_monitor: _BDFConditionMonitor | None = None
        if self.bdf_condition_warning_threshold is not None:
            # Only attach to BDF — non-BDF solvers don't have a Newton
            # iteration to monitor, and we don't want to silently
            # mislead users who set the option on a non-BDF run.
            try:
                from ..backend._jax.bdf import BDFSolver as _BDFSolver
            except Exception:  # pragma: no cover — import-time guard only
                _BDFSolver = None
            if _BDFSolver is not None and isinstance(ode_solver, _BDFSolver):
                self._bdf_cond_monitor = _BDFConditionMonitor(
                    self.bdf_condition_warning_threshold,
                )
                # Attach as an instance attribute so the BDF solver
                # picks it up via ``getattr(self, "_cond_monitor", None)``
                # without needing a constructor change.
                ode_solver._cond_monitor = self._bdf_cond_monitor

        # T-027a-followup: simulator-level Zeno protection options.  All
        # default-off — the recovery probe and the latch are skipped at
        # Python level when ``zeno_protection_enabled=False``, keeping the
        # default hot path byte-equivalent.
        self.zeno_protection_enabled = getattr(
            options, "zeno_protection_enabled", False,
        )
        self.zeno_tolerance = getattr(options, "zeno_tolerance", 1e-6)
        self.zeno_recovery_period = getattr(
            options, "zeno_recovery_period", 10,
        )
        # T-027a-followup-vector-tprev: count zero-crossing events at
        # construction time so ``initialize`` can allocate per-event
        # ``zeno_tprev`` / ``zeno_active`` vectors of the correct shape.
        # ``event_system_ids`` records each event's owning leaf — kept
        # for the per-leaf freeze gate (T-027a-followup-per-leaf-freeze).
        # Both are static (system topology doesn't change at runtime),
        # so this is a one-shot pass at __init__.
        zc_events_static = system.zero_crossing_events.events
        self.n_zero_crossing_events = len(zc_events_static)
        self.event_system_ids = tuple(
            getattr(ev, "system_id", None) for ev in zc_events_static
        )
        # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
        # snapshot each zero-crossing event's static ``direction`` string
        # so the per-event recovery-probe nudge in ``_apply_recovery_w0_nudge``
        # can pick the right side of the threshold to push ``w0`` to.
        # Mirrors ``event_system_ids`` — same order as
        # ``system.zero_crossing_events.events`` and the per-event Zeno
        # carry vectors built in ``initialize``.
        self.event_directions = tuple(
            getattr(ev, "direction", "crosses_zero") for ev in zc_events_static
        )

        # T-027a-followup-per-leaf-freeze: build a static map from each
        # leaf's ``system_id`` to its index in
        # ``DiagramContext.continuous_state`` (which is a list ordered by
        # the iteration of subcontexts that have continuous state).  This
        # lets ``_major_step`` know which slot of the list to roll back
        # when an event owned by that leaf has its Zeno latch engaged.
        # For single-LeafSystem simulations the list collapses to one
        # entry (``LeafContext.continuous_state`` is a single Array, not
        # a list — handled separately at the freeze site).
        if isinstance(system, Diagram):
            _leaves = list(system.leaf_systems)
        else:
            _leaves = [system]
        self._sysid_to_cs_idx = {}
        _cs_idx = 0
        for _leaf in _leaves:
            if getattr(_leaf, "has_continuous_state", False):
                self._sysid_to_cs_idx[_leaf.system_id] = _cs_idx
                _cs_idx += 1
        self._n_continuous_leaves = _cs_idx
        # Reverse map: cs_index -> tuple of event-vector positions whose
        # ``zeno_active[i]`` should freeze that leaf.  Built once so the
        # per-step freeze logic is a flat scatter, no per-step Python
        # iteration over event_system_ids.
        self._cs_idx_to_event_positions: dict[int, tuple[int, ...]] = {}
        for i, sid in enumerate(self.event_system_ids):
            cs_i = self._sysid_to_cs_idx.get(sid)
            if cs_i is None:
                continue
            self._cs_idx_to_event_positions.setdefault(cs_i, []).append(i)
        self._cs_idx_to_event_positions = {
            k: tuple(v) for k, v in self._cs_idx_to_event_positions.items()
        }

        # T-027a-followup-per-leaf-solver-state: compute each continuous
        # leaf's flat slice into the raveled ODE state vector ``y`` so the
        # per-leaf freeze gate can decompose ``Dopri5State.{y, f,
        # interp_coeff}`` (and ``BDFState.{y, f, D}``) along the last
        # axis.  The flat layout matches ``ravel_pytree(context.
        # continuous_state)`` exactly: leaves with continuous state are
        # iterated in ``DiagramContext.continuous_subcontexts`` order
        # (subcontexts.values() filtered by has_continuous_state), which
        # is the same order as ``Diagram.leaf_systems`` filtered by
        # ``has_continuous_state``.  Each leaf's flat size is the sum of
        # its ``_default_continuous_state`` pytree-leaf sizes — usually
        # a single Array, but pytree-valued continuous states are also
        # handled.  ``_leaf_flat_slices`` is a tuple of ``(start, end)``
        # int pairs ordered by ``cs_idx``; total length is the flat
        # ODE state dimension ``_n_y_total``.  Default-off path
        # (``zeno_protection_enabled=False``) never reads these; they
        # are purely metadata.
        self._leaf_flat_slices: tuple[tuple[int, int], ...] = ()
        self._n_y_total: int = 0
        if self._n_continuous_leaves > 0:
            _slices: list[tuple[int, int]] = []
            _offset = 0
            # Re-scan ``_leaves`` in the same order used for ``_sysid_to_cs_idx``.
            for _leaf in _leaves:
                if not getattr(_leaf, "has_continuous_state", False):
                    continue
                _xc0 = getattr(_leaf, "_default_continuous_state", None)
                if _xc0 is None:
                    _size = 0
                else:
                    _size = int(sum(
                        int(np.prod(np.shape(_l))) if np.shape(_l) else 1
                        for _l in jax.tree_util.tree_leaves(_xc0)
                    ))
                _slices.append((_offset, _offset + _size))
                _offset += _size
            self._leaf_flat_slices = tuple(_slices)
            self._n_y_total = _offset

        # T-013a-followup-mode-a-buffers: when the user opts into the
        # "buffers" mode, classify each recorded signal's cadence
        # statically here and pass the result through to the recorder.
        # The classification is reused at the per-step decision in
        # ``JaxResultsData.update`` to skip writes for unfired periodic
        # signals.  Default ``"auto"`` does NOT enable buffers — it
        # remains the post-finalize schedule trim path.
        psts_mode = getattr(options, "per_signal_timestamps_mode", "auto")
        psts_enabled = getattr(options, "per_signal_timestamps", False)
        per_signal_buffers_classifications = None
        if (
            psts_enabled
            and psts_mode == "buffers"
            and options.recorded_signals is not None
        ):
            per_signal_buffers_classifications = (
                ResultsRecorder.classify_signal_cadence(options.recorded_signals)
            )

        # T-012a-followup: thread record_solver_states through to the
        # recorder so the JaxResultsData allocates a per-step interpolant
        # ring and ``save`` snapshots ``Dopri5State.interp_coeff`` per
        # call.  Default-off path is byte-equivalent.
        self.record_solver_states = getattr(
            options, "record_solver_states", False,
        )
        # T-002b-followup-buffer-overflow-auto-size — when the user
        # constructs ``Simulator`` directly (bypassing ``simulate``), the
        # ``_check_options`` auto-sizing path is skipped, so ``options.
        # buffer_length`` may still be ``None``. Fall back to
        # ``max_major_steps`` (the natural cap) or a legacy 1000-sample
        # default when neither is available.
        if options.buffer_length is not None:
            recorder_buffer_length = options.buffer_length
        elif self.max_major_steps is not None and self.max_major_steps > 0:
            recorder_buffer_length = max(
                int(self.max_major_steps), _MIN_AUTO_BUFFER_LENGTH
            )
        else:
            recorder_buffer_length = _MIN_AUTO_BUFFER_LENGTH
        self.results_recorder = ResultsRecorder(
            save_time_series=options.save_time_series,
            recorded_outputs=options.recorded_signals,
            buffer_length=recorder_buffer_length,
            per_signal_buffers_classifications=per_signal_buffers_classifications,
            record_solver_states=self.record_solver_states,
        )

        # Zero-crossing handler encapsulates guard evaluation and bisection logic
        self.zc_handler = ZeroCrossingHandler(
            system,
            self.zc_bisection_loop_count,
            lower_triangular_discrete_update=getattr(
                options, "lower_triangular_discrete_update", False,
            ),
        )

        if self.max_major_step_length is None:
            self.max_major_step_length = np.inf

        logger.debug("Simulator created with enable_tracing=%s", self.enable_tracing)

        self.ode_solver = ode_solver

        # T-113-followup-baumgarte-and-ssp: opt-in Baumgarte stabilization.
        # When ``baumgarte_alpha`` and/or ``baumgarte_beta`` are set, wrap
        # the solver's ``ode_rhs`` to add ``-2α·ġ - β²·g`` to the
        # algebraic rows of the rhs.  ``baumgarte_augment_ode_rhs`` is a
        # no-op (returns the input rhs unchanged) when both gains are
        # ``None`` or when the system has no algebraic constraints — the
        # disabled hot path's JIT trace graph is byte-equivalent to the
        # pre-followup behaviour.  Wraps before any ``ode_solver.initialize``
        # call so ``flat_ode_rhs = ravel_first_arg(self.ode_rhs, ...)`` in
        # the JAX impl picks up the augmented version.
        b_alpha = getattr(options, "baumgarte_alpha", None)
        b_beta = getattr(options, "baumgarte_beta", None)
        if (b_alpha is not None or b_beta is not None) and getattr(
            self.system, "has_mass_matrix", False,
        ):
            from .dae_projection import baumgarte_augment_ode_rhs
            ode_solver.ode_rhs = baumgarte_augment_ode_rhs(
                ode_solver.ode_rhs, self.system, b_alpha, b_beta,
            )

        from .autodiff_rules import make_advance_to_vjp, make_guarded_integrate_vjp
        # Modify the default autodiff rule slightly to correctly capture variations
        # in end time of the simulation interval.
        self.has_terminal_events = system.zero_crossing_events.has_terminal_events
        # T-006: wrap advance_to so direct callers (not going through
        # simulate()) also get JAX-error remapping with block/port context.
        # T-A2-followup-advance-to-jit-cache: jit the inner advance_to so a
        # *persistent* Simulator (construct once, call ``advance_to`` many
        # times — interactive stepping, MPC inner loops) reuses the compiled
        # kernel instead of re-tracing op-by-op on every call. The jit is a
        # stable instance attribute, so JAX's cache hits across calls with the
        # same context aval. Only the non-autodiff path is wrapped: the
        # autodiff path returns a ``custom_vjp`` callable that ``simulate``
        # already jits at the outer ``_wrapped_simulate`` boundary, and we
        # keep ``remap_simulation_errors`` on the *outside* so runtime errors
        # are still remapped at the call boundary (not just at trace time).
        _advance_to_impl = make_advance_to_vjp(self)
        if self.enable_tracing and not self.enable_autodiff:
            _advance_to_impl = jax.jit(_advance_to_impl)
        self.advance_to = remap_simulation_errors(_advance_to_impl)

        # Also override the guarded ODE integration with a custom autodiff rule
        # to capture variations due to zero-crossing time.
        self.guarded_integrate = make_guarded_integrate_vjp(self)

    def compile(self, tf: float, context: ContextBase):
        """Warm up / pre-compile the simulation advance_to method on the device."""
        if self.enable_tracing and not self.enable_autodiff:
            self.advance_to(tf, context)

    def while_loop(self, cond_fun, body_fun, val):
        """Structured control flow primitive for a while loop.

        Dispatches to a bounded while loop when:
          • ``enable_autodiff=True`` (required for reverse-mode AD), or
          • the caller explicitly set ``max_major_steps`` in SimulatorOptions
            (acts as a hard simulation budget, e.g. for Zeno protection).

        Otherwise the standard unbounded ``lax.while_loop`` (JAX backend) or a
        pure-Python loop (NumPy backend) is used.
        """
        use_bounded = self.enable_autodiff or self._explicit_max_major_steps
        if use_bounded:
            return _bounded_while_loop(cond_fun, body_fun, val, self.max_major_steps)
        else:
            return backend.while_loop(cond_fun, body_fun, val)

    def initialize(self, context: ContextBase) -> SimulatorState:
        """Perform initial setup for the simulation."""
        logger.debug("Initializing simulator")
        # context.state.pprint(logger.debug)

        # Initial simulation time as integer (picoseconds)
        initial_int_time = IntegerTime.from_decimal(context.time)

        # Ensure that _next_update_time() can return the current time by perturbing
        # current time as slightly toward negative infinity as possible
        time_of_next_timed_event, timed_events = _next_update_time(
            self.system.periodic_events, initial_int_time - 1
        )

        # timed_events is now marked with the active events at the next update time
        logger.debug("Time of next timed event (int): %s", time_of_next_timed_event)
        logger.debug(
            "Time of next event (sec): %s",
            IntegerTime.as_decimal(time_of_next_timed_event),
        )
        timed_events.pprint(logger.debug)

        end_reason = npa.where(
            time_of_next_timed_event == initial_int_time,
            StepEndReason.TimeTriggered,
            StepEndReason.NothingTriggered,
        )

        # Initialize the results data that will hold recorded time series data.
        results_data = self.results_recorder.initialize(context)

        # T-027a-followup-vector-tprev: when simulator-level Zeno
        # protection is enabled, allocate per-event ``zeno_tprev`` /
        # ``zeno_active`` vectors so each event tracks its own last-
        # firing time independently.  ``zeno_tprev`` is initialised to
        # ``-inf`` so the first firing is never inside tolerance.  When
        # disabled, leave the carry as the scalar defaults from the
        # ``SimulatorState`` declaration so the default-off path's
        # pytree is byte-equivalent.
        if self.zeno_protection_enabled:
            n = max(self.n_zero_crossing_events, 1)
            zeno_tprev = jnp.full((n,), -jnp.inf)
            zeno_active = jnp.zeros((n,), dtype=jnp.bool_)
            # T-027a-followup-per-event-recovery: ``zeno_frozen_steps``
            # vectorises to ``(N_events,)`` so each event independently
            # counts its own consecutive-frozen-step streak.  When event
            # ``i`` hits ``zeno_recovery_period``, only its own latch
            # clears; other events keep cascading.  Default-off path
            # keeps the scalar default in ``SimulatorState`` so the
            # disabled pytree shape is byte-equivalent.
            zeno_frozen_steps = jnp.zeros((n,), dtype=jnp.int32)
            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # per-event mask of "the previous major step's recovery probe
            # just fired for this event".  Initialised to all-False so the
            # first ODE step has no nudge applied.  Shape matches the
            # other per-event carry vectors so the elementwise compare/
            # scatter inside ``_apply_recovery_w0_nudge`` aligns.
            zeno_recovery_just_cleared = jnp.zeros((n,), dtype=jnp.bool_)
            return SimulatorState(
                context=context,
                timed_events=timed_events,
                step_end_reason=end_reason,
                int_time=initial_int_time,
                results_data=results_data,
                ode_solver_state=self.ode_solver.initialize(context),
                zeno_tprev=zeno_tprev,
                zeno_active=zeno_active,
                zeno_frozen_steps=zeno_frozen_steps,
                zeno_recovery_just_cleared=zeno_recovery_just_cleared,
            )

        return SimulatorState(
            context=context,
            timed_events=timed_events,
            step_end_reason=end_reason,
            int_time=initial_int_time,
            results_data=results_data,
            ode_solver_state=self.ode_solver.initialize(context),
        )





    def _guarded_integrate(
        self,
        solver_state: ODESolverState,
        results_data: ResultsData,
        tf: float,
        context: ContextBase,
        zc_events: EventCollection,
        recovery_just_cleared=None,
        prior_zeno_active=None,
    ) -> tuple[bool, ODESolverState, ContextBase, ResultsData, EventCollection]:
        """Guarded ODE integration.

        Advance continuous time using an ODE solver, localizing any zero-crossing events
        that occur during the requested interval.  If any zero-crossing events trigger,
        the dense interpolant is used to localize the events and the associated reset maps
        are handled.  The method then returns, guaranteeing that the major step terminates
        either at the end of the requested interval or at the time of a zero-crossing
        event.

        Args:
            solver_state (ODESolverState): The current state of the ODE solver.
            results_data (ResultsData): The results data that will hold recorded time
                series data.
            tf (float): The end time of the integration interval.
            context (ContextBase): The current state of the system.
            zc_events (EventCollection): The current zero-crossing events.
            recovery_just_cleared: Optional per-event boolean mask flagging
                events whose Zeno latch was cleared by the recovery probe
                on the previous major step.  When non-None, a direction-
                aware nudge is applied to ``w0`` after ``record_interval_start``
                to recover the trigger semantics for events whose host
                leaf is at the post-reset rest condition.  See
                ``_apply_recovery_w0_nudge`` for the rationale.  Default
                ``None`` is the byte-equivalent legacy path (no nudge).

        Returns:
            tuple[bool, ODESolverState, ContextBase, ResultsData, EventCollection]:
                A tuple containing the following:
                - A boolean indicating whether the major step was terminated early due to
                  a zero-crossing event.
                - The updated state of the ODE solver.
                - The updated state of the system.
                - The updated results data.
                - The updated zero-crossing events.
        """
        solver = self.ode_solver
        func = solver.flat_ode_rhs  # Raveled ODE RHS function

        # Close over the additional arguments so that the RHS function has the
        # signature `func(y, t)`.
        def _func(y, t):
            return func(y, t, context)

        def _localize_zc_minor(
            solver_state, context_t0, context_tf, zc_events, results_data
        ):
            # Use the ZeroCrossingHandler to localize via bisection
            int_t1 = IntegerTime.from_decimal(context_tf.time)
            int_t0 = IntegerTime.from_decimal(context_t0.time)
            context_tf, zc_events = self.zc_handler.localize(
                solver_state, context_tf, zc_events, int_t0, int_t1
            )

            # record results sample for the ZC having 'occurred'
            minor_step_end_time = IntegerTime.as_decimal(int_t1)
            minor_step_start_time = IntegerTime.as_decimal(int_t0)
            zc_occur_time = context_tf.time - (
                minor_step_end_time - minor_step_start_time
            ) / (2 ** (self.zc_bisection_loop_count + 1))
            context_zc_time = context_tf.with_time(zc_occur_time)
            context_zc_time = context_zc_time.refresh_port_cache()
            # T-012a-followup: pre-localization solver_state's interp_coeff
            # spans the bracket that contained the ZC time; pass it through
            # so query() can later evaluate the polynomial at any t inside.
            results_data = self.results_recorder.save(
                results_data, context_zc_time, ode_solver_state=solver_state,
            )

            # Handle any triggered zero-crossing events
            context_tf = self.zc_handler.handle_events(zc_events, context_tf)

            # Re-initialize the solver since state may have been reset
            solver_state = solver.initialize(context_tf)
            return solver_state, context_tf, zc_events, results_data

        def _no_events_fun(
            solver_state, context_t0, context_tf, zc_events, results_data
        ):
            return solver_state, context_tf, zc_events, results_data

        # T-017b: when the system has no zero-crossing events, the
        # ``backend.cond(triggered, _localize_zc_minor, _no_events_fun, ...)``
        # below would still trace ``_localize_zc_minor`` (which retraces
        # ``solver.initialize`` — a measurable XLA cost for BDF/DAE
        # systems with mass matrices).  Skip that branch entirely at
        # Python level when the system declares no zero-crossings, and
        # also skip ``zc_handler.check_triggered``.  Numerically
        # bit-equivalent to the original path because ``triggered``
        # would always be False in that case.
        has_zero_crossings = self.system.has_zero_crossing_events

        def _ode_step(carry):
            _, solver_state, context_t0, results_data, zc_events = carry

            # Save results at the top of the loop. This will save data at t=t0,
            # but not at t=tf.  This is okay, since we will save the results at
            # the top of the next major step, as well as at the end of the main
            # simulation loop.
            context_t0 = context_t0.refresh_port_cache()
            # T-012a-followup: pass solver_state so the recorder snapshots
            # the per-step interpolant coefficients alongside (time, outputs).
            # Default-off path: ``record_solver_states=False`` means the
            # recorder ignores the kwarg — byte-equivalent to legacy.
            results_data = self.results_recorder.save(
                results_data, context_t0, ode_solver_state=solver_state,
            )

            zc_events = self.zc_handler.record_interval_start(zc_events, context_t0)

            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # direction-aware ``w0`` nudge for events whose Zeno latch was
            # just cleared by the recovery probe (or whose recorded ``w0``
            # is at the threshold).  Default-off path
            # (``recovery_just_cleared`` is None) skips at Python level
            # so the legacy hot path is byte-equivalent.  When applied,
            # the nudge is gated per-event by ``mask[i]`` so events not
            # in recovery flow through unchanged.  See
            # ``_apply_recovery_w0_nudge`` for the rationale.
            if recovery_just_cleared is not None:
                zc_events = self._apply_recovery_w0_nudge(
                    zc_events, recovery_just_cleared,
                    prior_zeno_active=prior_zeno_active,
                )

            # Advance ODE solver
            solver_state = solver.step(_func, tf, solver_state)
            xc = solver_state.unraveled_state
            context = context_t0.with_time(solver_state.t).with_continuous_state(xc)

            context = context.refresh_port_cache()

            if not has_zero_crossings:
                # No-ZC fast path: skip both ``check_triggered`` and the
                # ``cond(_localize_zc_minor, _no_events_fun)`` branch.
                return (False, solver_state, context, results_data, zc_events)

            # Check for zero-crossing events
            zc_events = self.zc_handler.check_triggered(zc_events, context)

            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # mask out triggers for events whose simulator-level Zeno
            # latch is engaged — the per-leaf freeze rollback at
            # ``_major_step`` already handles those leaves' continuous
            # state, and letting their ``triggered`` flag propagate here
            # would terminate the ODE step at a tiny dt and stall the
            # still-bouncing leaf's natural progression.
            if prior_zeno_active is not None:
                zc_events = self._mask_triggered_for_active_latch(
                    zc_events, prior_zeno_active,
                )

            triggered = zc_events.has_triggered

            args = (solver_state, context_t0, context, zc_events, results_data)
            solver_state, context, zc_events, results_data = backend.cond(
                triggered, _localize_zc_minor, _no_events_fun, *args
            )

            return (triggered, solver_state, context, results_data, zc_events)

        def _cond_fun(carry):
            triggered, solver_state, _, _, _ = carry
            return (solver_state.t < tf) & (~triggered)

        carry = (False, solver_state, context, results_data, zc_events)
        triggered, solver_state, context, results_data, zc_events = backend.while_loop(
            _cond_fun,
            _ode_step,
            carry,
        )

        return triggered, solver_state, context, results_data, zc_events



    def _advance_continuous_time(
        self,
        cdata: ContinuousIntervalData,
    ) -> ContinuousIntervalData:
        """Advance the simulation to the next discrete update or zero-crossing event.

        This stores the values of all active guard functions and advances the
        continuous-time component of the system to the next discrete update or
        zero-crossing event, whichever comes first.  Zero-crossing events are
        localized using a bisection search defined by `_trigger_search`, which will
        also record the final guard function values at the end of the search interval
        and determine which (if any) zero-crossing events were triggered.
        """

        # Unpack inputs
        int_tf = cdata.tf
        context = cdata.context
        results_data = cdata.results_data

        context = context.refresh_port_cache()
        zc_events = self.zc_handler.evaluate_guards(context)

        if self.system.has_continuous_state:
            solver_state = cdata.ode_solver_state
            tf = IntegerTime.as_decimal(int_tf)

            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # plumb the per-event ``recovery_just_cleared`` mask down into
            # ``_guarded_integrate`` so the inner ``_ode_step`` can apply
            # the direction-aware ``w0`` nudge.  When the autodiff custom
            # VJP wrapper is in play, ``self.guarded_integrate`` is a
            # ``custom_vjp`` callable with a fixed 5-arg signature — call
            # the unwrapped ``_guarded_integrate`` directly in that case
            # since autodiff users don't go through the Zeno cascade
            # pathway in practice and the nudge would have no effect on
            # the gradient (it is an idempotent ``where`` on a value
            # that, in normal operation, already satisfies the trigger
            # direction inequality).  Default ``recovery_just_cleared
            # is None`` keeps the legacy 5-arg call so the byte-
            # equivalent default-off path is preserved.
            rjc = cdata.recovery_just_cleared
            pza = cdata.prior_zeno_active
            if rjc is None and pza is None:
                (
                    triggered,
                    solver_state,
                    context,
                    results_data,
                    zc_events,
                ) = self.guarded_integrate(
                    solver_state,
                    results_data,
                    tf,
                    context,
                    zc_events,
                )
            else:
                (
                    triggered,
                    solver_state,
                    context,
                    results_data,
                    zc_events,
                ) = self._guarded_integrate(
                    solver_state,
                    results_data,
                    tf,
                    context,
                    zc_events,
                    recovery_just_cleared=rjc,
                    prior_zeno_active=pza,
                )

            context = context.with_time(solver_state.t)
            context = context.with_continuous_state(solver_state.unraveled_state)

            # Converting from decimal -> integer time incurs a loss of precision.  This is
            # okay for unscheduled zero-crossing events, but problematic for timed events.
            # So only do this conversion if a zero-crossing was triggered.  Otherwise we
            # know we have reached the end of the interval and can keep the requested end
            # time.
            int_tf = npa.where(
                triggered,
                IntegerTime.from_decimal(context.time),
                int_tf,
            )

        else:
            # Skip the ODE solver for systems without continuous state.  We still
            # have to check for triggered events here in case there are any
            # transitions triggered by time that need to be handled before the
            # periodic discrete update at the top of the next major step
            triggered = False
            solver_state = cdata.ode_solver_state

            zc_events = self.zc_handler.record_interval_start(zc_events, context)
            results_data = self.results_recorder.save(results_data, context)

            # Advance time to the end of the interval
            context = context.with_time(IntegerTime.as_decimal(int_tf))
            context = context.refresh_port_cache()

            # Record guard values after the discrete update and check if anything
            # triggered as a result of advancing time
            zc_events = self.zc_handler.record_interval_end(zc_events, context)
            zc_events = self.zc_handler.check_triggered(zc_events, context)

            # Handle any triggered zero-crossing events
            context = self.zc_handler.handle_events(zc_events, context)

        # Even though the zero-crossing events have already been "handled", the
        # information about whether a terminal event has been triggered is still in
        # the events collection (since "triggered" has not been cleared by a call
        # to determine_triggered_guards).
        terminate_early = zc_events.has_active_terminal

        # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
        # extract a per-event triggered mask from the post-step events
        # collection so the simulator-level Zeno tracker can update each
        # event's ``tprev[i]`` independently.  Without this, the scalar
        # ``triggered`` (any event fired) broadcasts to all events,
        # which spuriously latches unrelated events when a single
        # leaf's cascade drives sub-tolerance major-step ends — e.g.
        # ball A's bouncing cascade causing ball B's latch to engage.
        # Default-off path: ``zeno_protection_enabled=False`` ignores
        # this field (cdata.per_event_triggered carries through as
        # ``None``), so the byte-equivalent legacy path is preserved.
        # The terminal-early branch in ``_major_step`` keeps whatever
        # placeholder cdata had at construction time, so when Zeno is
        # enabled we always populate this with a fixed-shape array
        # whether or not events fired (the all-False zeros from the
        # placeholder remain semantically valid in that case).
        if cdata.per_event_triggered is not None and self.n_zero_crossing_events > 0:
            per_event_triggered = self._extract_per_event_triggered(zc_events)
        else:
            per_event_triggered = cdata.per_event_triggered

        # T-125-followup-record-event-times: tee ``(time, per_event_mask)``
        # to the host-side recorder when the option is on AND the diagram
        # has zero-crossing events.  The recorder ignores all-False masks
        # host-side so non-triggering major steps add nothing.  Default-
        # off path: ``self._event_time_recorder is None`` skips the
        # entire block at trace time, preserving byte-equivalence.
        # ``context.time`` is already the localized event firing time
        # when ``triggered`` is True (``_advance_continuous_time`` sets
        # ``context.time = solver_state.t`` after ``guarded_integrate``
        # which clamps to the bisection root); when ``triggered`` is
        # False the mask is all-False and the host-side guard short-
        # circuits without appending anything.
        if (
            self._event_time_recorder is not None
            and self.n_zero_crossing_events > 0
        ):
            event_mask = self._extract_per_event_triggered(zc_events)
            jax.debug.callback(
                self._event_time_recorder.update,
                context.time,
                event_mask,
            )

        return cdata._replace(
            triggered=triggered,
            terminate_early=terminate_early,
            context=context,
            tf=int_tf,
            results_data=results_data,
            ode_solver_state=solver_state,
            per_event_triggered=per_event_triggered,
        )

    def _extract_per_event_triggered(self, zc_events):
        """T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
        flatten ``zc_events`` into a per-event boolean array of shape
        ``(N_events,)`` aligned with ``self.event_directions`` /
        ``self.event_system_ids`` / per-event Zeno carry vectors.

        Walks the events tree in flatten order — the same order as
        ``system.zero_crossing_events.events``.  Each
        ``event.event_data.triggered`` is a JAX scalar that may be a
        tracer or a concrete bool; we collect them into a stack and
        return a ``(N_events,)`` array.

        Used by ``_major_step`` to plumb per-event trigger info up to
        ``_update_zeno_tracking`` so each event's ``tprev[i]`` and
        engagement check is per-event rather than scalar-broadcast.
        """
        triggered_list: list = []

        def _collect(event):
            if isinstance(event, ZeroCrossingEvent):
                triggered_list.append(
                    jnp.asarray(event.event_data.triggered, dtype=jnp.bool_),
                )
                # Returning ``event`` keeps the tree walk happy; the
                # actual collection happens via the side-effect closure.
                return event
            return event

        jax.tree_util.tree_map(
            _collect, zc_events,
            is_leaf=lambda x: isinstance(x, ZeroCrossingEvent),
        )
        if not triggered_list:
            return jnp.zeros((0,), dtype=jnp.bool_)
        return jnp.stack(triggered_list)

    def _handle_discrete_update(
        self, context: ContextBase, timed_events: EventCollection
    ) -> tuple[ContextBase, bool]:
        """Handle discrete updates triggered by time.

        This method is called at the beginning of each major step to handle any
        discrete updates that are triggered by time.  This includes both discrete
        updates that are triggered by time and any zero-crossing events that are
        triggered by the discrete update.

        This will also work when there are no zero-crossing events: the zero-crossing
        collection will be empty and only the periodic discrete update will happen.

        Args:
            context (ContextBase): The current state of the system.
            timed_events (EventCollection):
                The collection of timed events, with the active events marked.

        Returns:
            ContextBase: The updated state of the system.
            bool: Whether the simulation should terminate early as a result of a
                triggered terminal condition.
        """
        return self.zc_handler.check_after_discrete_update(context, timed_events)

    def _update_zeno_tracking(
        self,
        zeno_tprev,
        zeno_active,
        zeno_frozen_steps,
        triggered,
        current_time,
    ):
        """T-027a-followup: simulator-level Zeno tracker with recovery probe.

        Updates the carry triple ``(tprev, active, frozen_steps)`` after a
        major step.  Logic, in order:

        1. If the major step ended on a guard trigger AND the time since
           the last recorded trigger is below ``zeno_tolerance``, latch
           the protection on (``active=True``).  The latch is sticky —
           once on, only the recovery probe (step 3) clears it.
        2. Update ``tprev`` to the current step's end time whenever a
           guard fired, so the tolerance check in step 1 reflects the
           most recent triggering history.
        3. Recovery probe: after ``zeno_recovery_period`` consecutive
           frozen major steps, clear ``active`` for one step.  The next
           guard-trigger check then either re-engages Zeno (cascade
           ongoing — tolerance check fires again) or leaves it cleared
           (cascade resolved).  The frozen-step counter resets on every
           ``active`` flip.

        T-027a-followup-vector-tprev: ``zeno_tprev`` and ``zeno_active``
        are per-event vectors of shape ``(N_events,)``; ``triggered`` may
        be a scalar (broadcast to the full vector) or a same-shape per-
        event mask.  Each event's last-firing time is tracked
        independently, so unrelated events do not contaminate each
        other's tolerance check.

        T-027a-followup-per-event-recovery: ``zeno_frozen_steps`` is a
        per-event vector of shape ``(N_events,)``.  Each event's
        counter increments while THAT event's ``active[i]`` is True and
        resets when it clears.  When a single event ``i`` hits
        ``zeno_recovery_period``, only ``active[i]`` (and its own
        counter) clear — other events' latches and counters are
        untouched, so a still-cascading event A no longer gets a free
        probe just because event B's cascade ended.  When a scalar
        ``zeno_frozen_steps`` is passed in (the default-disabled
        SimulatorState carry), it is broadcast to the per-event shape;
        the byte-equivalent default-off path never enters this branch.

        The actual freeze (zero out ode_rhs while ``active``) is not
        wired in here.  ``zeno_active`` is observational for now;
        callers can inspect ``sim_state.zeno_active`` to diagnose Zeno
        cascades without changing simulation outputs.  The per-leaf
        freeze gate that consumes ``zeno_active[i]`` and the static
        event→leaf map ``self.event_system_ids`` is filed under
        T-027a-followup-per-leaf-freeze.
        """
        # Treat ``triggered`` / ``active`` as JAX-friendly booleans so this
        # works under jit/vmap as well as the Python/numpy backend.  Match
        # the dtypes of the in/out carry fields to the SimulatorState
        # defaults so ``lax.cond`` true/false-branch dtype checks pass.
        tprev = jnp.asarray(zeno_tprev)
        active_b = jnp.asarray(zeno_active, dtype=jnp.bool_)
        # T-027a-followup-per-event-recovery: ``frozen`` is a per-event
        # ``(N_events,)`` vector when Zeno protection is enabled.  A
        # scalar input (legacy default-off carry, or the V-005 helper
        # ``_step_zeno`` which passes ``int(frozen)``) is broadcast up
        # to ``tprev.shape`` so the elementwise compare/where below
        # works for both cases.
        frozen = jnp.broadcast_to(
            jnp.asarray(zeno_frozen_steps, dtype=jnp.int32), tprev.shape,
        )
        # Broadcast a scalar ``triggered`` up to the per-event tprev /
        # active shape so the masks align elementwise.  Callers passing
        # a properly-shaped per-event mask flow through unchanged.
        triggered_b = jnp.broadcast_to(
            jnp.asarray(triggered, dtype=jnp.bool_), tprev.shape,
        )
        t = jnp.asarray(current_time, dtype=tprev.dtype)
        tol = jnp.asarray(self.zeno_tolerance, dtype=tprev.dtype)
        period = jnp.asarray(self.zeno_recovery_period, dtype=frozen.dtype)
        one_step = jnp.asarray(1, dtype=frozen.dtype)
        zero_step = jnp.asarray(0, dtype=frozen.dtype)

        # Engagement condition: a guard fired AND inter-event time was
        # below tolerance.  Per-event: each ``tprev[i]`` is checked
        # against ``t`` independently.
        dt_since_prev = t - tprev
        engage = triggered_b & (dt_since_prev < tol) & (dt_since_prev >= 0)

        # Per-event active value before the recovery probe.
        active_after_engage = active_b | engage

        # T-027a-followup-per-event-recovery: per-event frozen-step
        # counter.  Each event's ``frozen[i]`` increments while its own
        # ``active_after_engage[i]`` is True and resets to zero when
        # cleared.  The recovery probe fires per-event: when
        # ``frozen[i] >= K``, only ``active[i]`` (and ``frozen[i]``)
        # clear — other events keep their latches and counters.  This
        # prevents a still-cascading event from getting a free probe
        # just because an unrelated event's cascade ended.
        new_frozen = jnp.where(active_after_engage, frozen + one_step, zero_step)
        should_probe = active_after_engage & (new_frozen >= period)
        new_active = jnp.where(should_probe, False, active_after_engage)
        new_frozen = jnp.where(should_probe, zero_step, new_frozen)

        # Update ``tprev[i]`` whenever event ``i`` fired so each event's
        # next tolerance check sees its own most-recent firing time.
        new_tprev = jnp.where(triggered_b, t, tprev)

        return new_tprev, new_active, new_frozen

    def _mask_triggered_for_active_latch(self, zc_events, prior_zeno_active):
        """T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
        suppress per-event ``triggered`` when the simulator-level Zeno
        latch is engaged for that event.

        When a leaf is in Zeno hold, the per-leaf freeze rollback at
        the end of ``_major_step`` re-pins its continuous state to the
        snapshot.  But the inner ``_ode_step`` loop checks
        ``zc_events.has_triggered`` and exits early on a trigger, so a
        latched-but-still-firing event would terminate the ODE step at
        a tiny dt — preventing any unfrozen leaf (e.g. ball B in a
        staggered cascade) from advancing.  Masking the latched
        events' triggered flag to False lets the ODE step run to its
        natural termination (next unfrozen-leaf trigger or interval
        end), giving the staggered cascade case a path forward.

        Default-off path (``zeno_protection_enabled=False``) never
        calls this helper, so the legacy semantics are unchanged.
        """
        if self.n_zero_crossing_events == 0 or prior_zeno_active is None:
            return zc_events

        active_mask = jnp.asarray(prior_zeno_active, dtype=jnp.bool_)
        idx_box = [0]

        def _update(event):
            if not isinstance(event, ZeroCrossingEvent):
                return event
            i = idx_box[0]
            idx_box[0] += 1
            old_triggered = jnp.asarray(
                event.event_data.triggered, dtype=jnp.bool_,
            )
            new_triggered = old_triggered & ~active_mask[i]
            return dataclasses.replace(
                event,
                event_data=dataclasses.replace(
                    event.event_data, triggered=new_triggered,
                ),
            )

        return jax.tree_util.tree_map(
            _update, zc_events,
            is_leaf=lambda x: isinstance(x, ZeroCrossingEvent),
        )

    def _apply_recovery_w0_nudge(
        self,
        zc_events,
        recovery_just_cleared,
        prior_zeno_active=None,
    ):
        """T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
        direction-aware ``w0`` nudge so events whose host leaf is at the
        post-reset rest condition can still trigger.

        Background: with reset maps that clamp the continuous state to
        the threshold (e.g. ``h := max(h, 0)`` for a bouncing ball), the
        post-reset state sits exactly on the guard's threshold.  The
        next major step's ``_ode_step`` records ``w0 = guard(t_start) = 0``
        in ``record_interval_start``.  When the post-reset velocity is
        small enough that the ODE step jumps the state to the other
        side of the threshold inside one solver step, the trigger
        condition for ``positive_then_non_positive``,
        ``(w0 > 0) & (w1 <= 0)``, evaluates to ``False & True = False``
        because ``w0 = 0`` is not strictly positive.  No trigger fires,
        the bounce reset is missed, and the leaf is left in free-fall
        below the threshold.  Subsequent steps see ``w0 < 0`` so the
        guard never refires.

        Originally surfaced via the per-event recovery probe (the
        cleared-step has the same ``(h=0, v ≈ 0+)`` initial condition);
        in practice it also fires on the staggered multi-leaf cascade
        BEFORE the latch ever engages, when ball A's natural cascade
        convergence drives the post-reset velocity below the solver's
        step-size resolution.  The nudge therefore fires whenever the
        recorded ``w0`` is on the "wrong side" of the trigger direction's
        active region by at most ``zeno_tolerance``, irrespective of
        whether the recovery probe just fired:

        - ``positive_then_non_positive``: active region is ``w > 0``;
          fire when ``w0 < eps`` (i.e. at or below threshold).  Set
          ``w0 := +eps``.
        - ``negative_then_non_negative``: active region is ``w < 0``;
          fire when ``w0 > -eps``.  Set ``w0 := -eps``.
        - ``crosses_zero``: symmetric trigger; fire when
          ``|w0| < eps`` and pick ``+eps`` (the side the typical
          contact-penetration constraint retreats into).
        - ``none`` / ``edge_detection``: no nudge.

        The ``recovery_just_cleared`` mask is preserved as a "force
        nudge" override — when set, the nudge fires regardless of the
        ``w0``-side check.  This mirrors the original (c) intent and
        gives a known-safe path even if the at-threshold check were
        too conservative for some pathological case.

        Idempotent: when ``w0`` is already strictly on the active side
        with magnitude above ``eps``, the nudge is a no-op.  Default-
        off path: ``zeno_protection_enabled=False`` skips this entirely
        at Python level (``recovery_just_cleared`` is ``None`` and the
        call site short-circuits) — byte-equivalent to legacy.

        Implementation note: a Python list closure counter walks the
        ``tree_map`` in flatten order — the same order as
        ``system.zero_crossing_events.events`` and the per-event
        ``recovery_just_cleared`` mask.  ``LeafEventCollection`` and
        ``DiagramEventCollection`` both register flatten orders that
        match the ``events`` property iteration order.  Under JIT/AD,
        the trace runs once, so the counter sees each event exactly
        once in the right order.
        """
        # Empty-events fast path.  Without this guard, the closure counter
        # below would still execute but produce no work; this avoids the
        # tree_map call entirely.
        if self.n_zero_crossing_events == 0:
            return zc_events

        # Static directions tuple captured at trace time — Python strings
        # so the per-event branch is a Python `if` (no traced control
        # flow), not a `lax.switch`.
        directions = self.event_directions
        mask = jnp.asarray(recovery_just_cleared, dtype=jnp.bool_)
        # When ``prior_zeno_active[i]`` is True, event ``i``'s host leaf
        # is in Zeno hold and the per-leaf freeze rollback is the
        # mechanism keeping its state pinned.  Skipping the nudge for
        # frozen events prevents a tight loop where the nudge fires
        # the trigger, the reset re-pins state at threshold, and the
        # next ODE step would re-fire — burning major steps without
        # advancing time.  Default ``None`` is treated as "no event
        # frozen", which is the correct behaviour pre-engagement and
        # for systems without Zeno protection.
        if prior_zeno_active is None:
            frozen_mask = jnp.zeros_like(mask)
        else:
            frozen_mask = jnp.asarray(prior_zeno_active, dtype=jnp.bool_)
        # Mirror the zeno tolerance for the nudge magnitude.  This is
        # intentionally larger than machine epsilon: it has to be big
        # enough that the next ODE step's ``w1`` lands on the OTHER
        # side of zero so the trigger condition ``w0 > 0 & w1 <= 0``
        # fires, but small enough that it does not perturb the bounce-
        # reset's localised trigger time meaningfully.  ``zeno_tolerance``
        # is the natural scale here — it is the same threshold the
        # latch engagement uses to decide "this firing is sub-tolerance
        # of the previous one".
        eps = jnp.asarray(self.zeno_tolerance)

        idx_box = [0]

        def _update(event):
            if not isinstance(event, ZeroCrossingEvent):
                return event
            i = idx_box[0]
            idx_box[0] += 1
            d = directions[i]
            if d == "positive_then_non_positive":
                # Trigger needs ``w0 > 0`` strictly.  Fire the nudge
                # when the recorded ``w0`` is at or below threshold
                # (``w0 < eps``) OR when the recovery probe just fired
                # for this event.  Skip when the leaf is in Zeno hold
                # (the per-leaf freeze rollback handles it).
                old_w0 = jnp.asarray(event.event_data.w0)
                eps_typed = jnp.asarray(eps, dtype=old_w0.dtype)
                should_nudge = (
                    (mask[i] | (old_w0 < eps_typed)) & ~frozen_mask[i]
                )
                new_w0 = jnp.where(should_nudge, eps_typed, old_w0)
            elif d == "negative_then_non_negative":
                # Trigger needs ``w0 < 0`` strictly.  Fire the nudge
                # when ``w0 > -eps`` OR mask is set.  Skip when frozen.
                old_w0 = jnp.asarray(event.event_data.w0)
                eps_typed = jnp.asarray(eps, dtype=old_w0.dtype)
                neg_eps = -eps_typed
                should_nudge = (
                    (mask[i] | (old_w0 > neg_eps)) & ~frozen_mask[i]
                )
                new_w0 = jnp.where(should_nudge, neg_eps, old_w0)
            elif d == "crosses_zero":
                # Symmetric trigger.  Fire the nudge when ``|w0| < eps``
                # OR mask is set; pick ``+eps`` as the active side.
                # Skip when frozen.
                old_w0 = jnp.asarray(event.event_data.w0)
                eps_typed = jnp.asarray(eps, dtype=old_w0.dtype)
                should_nudge = (
                    (mask[i] | (jnp.abs(old_w0) < eps_typed)) & ~frozen_mask[i]
                )
                new_w0 = jnp.where(should_nudge, eps_typed, old_w0)
            else:
                # ``"none"`` / ``"edge_detection"`` — no nudge applies.
                return event
            return dataclasses.replace(
                event,
                event_data=dataclasses.replace(event.event_data, w0=new_w0),
            )

        return jax.tree_util.tree_map(
            _update, zc_events, is_leaf=lambda x: isinstance(x, ZeroCrossingEvent),
        )

    def _apply_per_leaf_zeno_freeze(
        self,
        pre_xc,
        post_context,
        pre_solver_state,
        post_solver_state,
        zeno_active,
    ):
        """T-027a-followup-per-leaf-freeze: per-leaf rollback of continuous state.

        When one or more events have their Zeno latch engaged, the host
        leaf of each engaged event has its slice of the post-ODE
        continuous state replaced with the pre-ODE snapshot.  Other
        leaves keep their advanced state.  The choice is per-element:
        ``jnp.where(leaf_frozen, pre_leaf, post_leaf)``.

        For a single-LeafSystem simulation, ``pre_xc`` /
        ``post_context.continuous_state`` are scalar Arrays (one leaf,
        the "list" collapses), so the gate becomes a single
        ``jnp.where``.

        T-027a-followup-per-leaf-solver-state: the solver-state
        rollback now decomposes per leaf for the fields whose last
        axis matches the flat ODE state dimension ``n_y``.  An element-
        wise mask of length ``n_y`` (True at indices owned by frozen
        leaves) is built from the per-leaf freeze decisions and the
        static ``_leaf_flat_slices`` map; ``jnp.where(mask, pre, post)``
        rolls the frozen-leaf slice of ``y``, ``f``, and the per-step
        interpolant table back to the pre-ODE snapshot, while non-
        frozen leaves' slices keep their post-ODE values.  Scalar
        integration-step fields (``t``, ``dt``, ``t_prev``, ``t_return``,
        ``n_acc``, ``n_rej``, ``accepted``, ``order``,
        ``n_equal_steps``, ``updated_jacobian``) are global to the
        integrator's adaptive step controller and stay at the post-
        step values.  BDF Jacobian/mass/LU matrices (``J``, ``M``,
        ``LU``, ``U``) are ``(n_y, n_y)``-shaped with cross-leaf
        coupling and are NOT split — they roll back all-or-nothing
        when any leaf is frozen, matching the prior all-or-nothing
        behaviour for those specific fields.  Per-leaf-decomposable:
        Dopri5 ``y``, ``f``, ``interp_coeff``; BDF ``y``, ``f``,
        ``D``.
        """
        active_b = jnp.asarray(zeno_active, dtype=jnp.bool_)
        any_active = jnp.any(active_b)

        # Per-leaf freeze decisions (one Python bool tracer per cs_idx).
        # Built from the static ``_cs_idx_to_event_positions`` map so
        # this loop runs at trace-time and produces a fixed-shape pytree.
        # Leaves with no events trivially stay un-frozen.
        per_leaf_frozen: list = []
        for cs_i in range(self._n_continuous_leaves):
            positions = self._cs_idx_to_event_positions.get(cs_i, ())
            if positions:
                leaf_mask = active_b[jnp.asarray(positions)]
                per_leaf_frozen.append(jnp.any(leaf_mask))
            else:
                per_leaf_frozen.append(jnp.asarray(False))

        # Per-leaf gate on continuous state.
        post_xc = post_context.continuous_state
        if isinstance(post_xc, list):
            # DiagramContext path: per-leaf gate.
            new_xc_list = list(post_xc)
            for cs_i in range(len(post_xc)):
                positions = self._cs_idx_to_event_positions.get(cs_i, ())
                if not positions:
                    continue
                leaf_frozen = per_leaf_frozen[cs_i]
                pre_leaf = pre_xc[cs_i]
                post_leaf = post_xc[cs_i]
                new_xc_list[cs_i] = jax.tree_util.tree_map(
                    lambda p, n, _frozen=leaf_frozen: jnp.where(_frozen, p, n),
                    pre_leaf, post_leaf,
                )
            new_context = post_context.with_continuous_state(new_xc_list)
        else:
            # LeafContext path: single Array (or pytree).  Any event
            # active = freeze.  When the host system IS the leaf, the
            # cs_i=0 freeze decision covers every event.
            if per_leaf_frozen:
                leaf_frozen = per_leaf_frozen[0]
            else:
                leaf_frozen = any_active
            new_xc = jax.tree_util.tree_map(
                lambda p, n: jnp.where(leaf_frozen, p, n),
                pre_xc, post_xc,
            )
            new_context = post_context.with_continuous_state(new_xc)

        # Per-leaf solver-state rollback.  Build a (n_y,) bool mask
        # True at indices belonging to frozen leaves; ``jnp.where``
        # along the last axis splits the per-leaf-decomposable fields.
        new_solver_state = self._per_leaf_solver_gate(
            pre_solver_state, post_solver_state, per_leaf_frozen, any_active,
        )
        return new_context, new_solver_state

    def _per_leaf_solver_gate(
        self,
        pre_solver_state,
        post_solver_state,
        per_leaf_frozen,
        any_active,
    ):
        """T-027a-followup-per-leaf-solver-state: decompose the solver-
        state rollback per leaf along the flat ``n_y`` axis.

        Fields gated per-leaf (last-axis ``n_y``):
          - Dopri5State: ``y``, ``f``, ``interp_coeff``
          - BDFState: ``y``, ``f``, ``D``

        Fields kept post-step (global to the adaptive step controller):
          - ``t``, ``dt``, ``t_prev``, ``t_return``,
            ``n_acc``, ``n_rej``, ``accepted``,
            ``order``, ``n_equal_steps``, ``updated_jacobian``

        Fields rolled back all-or-nothing under ``any_active`` (BDF
        Jacobian/mass/LU matrices are ``(n_y, n_y)``-shaped with
        cross-leaf coupling, not last-axis-decomposable):
          - ``J``, ``M``, ``LU``, ``U``

        When ``self._n_y_total == 0`` (no continuous state) or no leaf
        is frozen, returns ``post_solver_state`` unchanged.  When the
        flat-slice metadata is missing (a defensive fallback), reverts
        to the previous all-or-nothing rollback.
        """
        if self._n_y_total <= 0 or not per_leaf_frozen:
            # Nothing to gate per-leaf — fall back to the
            # all-or-nothing path so behaviour matches the prior
            # T-027a-followup-per-leaf-freeze contract.
            return backend.cond(
                any_active,
                lambda _ss: pre_solver_state,
                lambda _ss: _ss,
                post_solver_state,
            )

        # Build the (n_y,) elementwise mask: True where the leaf
        # owning this index is frozen.  Static ``_leaf_flat_slices``
        # bounds plus per-leaf freeze decisions.
        n_y = self._n_y_total
        leaf_indicators = []
        for cs_i, (start, end) in enumerate(self._leaf_flat_slices):
            seg_len = end - start
            if seg_len <= 0:
                continue
            frozen_i = per_leaf_frozen[cs_i] if cs_i < len(per_leaf_frozen) else jnp.asarray(False)
            seg = jnp.broadcast_to(
                jnp.asarray(frozen_i, dtype=jnp.bool_), (seg_len,),
            )
            leaf_indicators.append(seg)
        if not leaf_indicators:
            return post_solver_state
        mask = jnp.concatenate(leaf_indicators)  # shape (n_y,)
        # Defensive: if any leaf had a flat-size mismatch, keep the
        # all-or-nothing fallback to avoid accidentally mis-aligning
        # the slices.  Static check at trace time.
        if mask.shape[0] != n_y:
            return backend.cond(
                any_active,
                lambda _ss: pre_solver_state,
                lambda _ss: _ss,
                post_solver_state,
            )

        def _gate_last_axis(pre_arr, post_arr):
            # Broadcast the (n_y,) mask to ``post_arr`` shape — the
            # mask aligns with the trailing axis (n_y).
            return jnp.where(mask, pre_arr, post_arr)

        # Detect Dopri5State vs BDFState by attribute presence —
        # avoids importing the backend classes here (the simulator
        # is backend-agnostic).
        post = post_solver_state
        pre = pre_solver_state
        kwargs = {}
        # Always-present per-leaf-decomposable fields.
        kwargs["y"] = _gate_last_axis(pre.y, post.y)
        kwargs["f"] = _gate_last_axis(pre.f, post.f)
        # Dopri5: interp_coeff has shape (5, n_y).
        if hasattr(post, "interp_coeff") and post.interp_coeff is not None:
            kwargs["interp_coeff"] = _gate_last_axis(
                pre.interp_coeff, post.interp_coeff,
            )
        # BDF: D has shape (MAX_ORDER+3, n_y).
        if hasattr(post, "D") and post.D is not None:
            kwargs["D"] = _gate_last_axis(pre.D, post.D)
        # BDF Jacobian/mass/LU matrices are (n_y, n_y) with cross-leaf
        # coupling — roll back all-or-nothing under any_active.
        if hasattr(post, "J") and post.J is not None:
            kwargs["J"] = jnp.where(any_active, pre.J, post.J)
        if hasattr(post, "M") and post.M is not None:
            kwargs["M"] = jnp.where(any_active, pre.M, post.M)
        if hasattr(post, "LU") and post.LU is not None:
            kwargs["LU"] = jnp.where(any_active, pre.LU, post.LU)
        if hasattr(post, "U") and post.U is not None:
            kwargs["U"] = jnp.where(any_active, pre.U, post.U)

        # Replace only the gated fields.  Scalar / integration-step
        # fields stay at their post-step values automatically.
        return dataclasses.replace(post, **kwargs)

    def _major_step(
        self,
        sim_state: SimulatorState,
        int_boundary_time: int,
        int_max_step_length: int,
    ) -> SimulatorState:
        end_reason = sim_state.step_end_reason
        context = sim_state.context
        timed_events = sim_state.timed_events
        int_time = sim_state.int_time

        if not self.enable_tracing:
            logger.debug("Starting a simulation step at t=%s", context.time)
            logger.debug("   merged_events: %s", timed_events)

        # Handle any discrete updates that are triggered by time along with
        # any zero-crossing events that are triggered by the discrete update.
        context, terminate_early = self._handle_discrete_update(
            context, timed_events
        )
        logger.debug("Terminate early after discrete update: %s", terminate_early)

        # T-113-followup-event-reprojection: project the post-reset state
        # back onto the constraint manifold *before* continuous integration
        # resumes.  ``_handle_discrete_update`` may have applied a discrete
        # update or a ZC reset triggered by it — either can drop algebraic
        # states off the manifold.  Default-off path
        # (``dae_reproject_after_events=False``) skips the block at trace
        # time so the disabled hot path is byte-equivalent.  No-op for
        # systems without a mass matrix.  Composes with T-003a's end-of-
        # major-step projection (both can run — they target different
        # within-step instants).
        if self.dae_reproject_after_events and getattr(
            self.system, "has_mass_matrix", False,
        ):
            from .dae_projection import project_constraints
            context = project_constraints(
                self.system,
                context,
                tol=self.dae_projection_tol,
                max_iter=self.dae_projection_max_iter,
            )

        # How far can we go before we have to handle timed events?
        # The time returned here will be the integer time representation.
        time_of_next_timed_event, timed_events = _next_update_time(
            self.system.periodic_events, int_time
        )
        if not self.enable_tracing:
            logger.debug(
                "Next timed event at t=%s",
                IntegerTime.as_decimal(time_of_next_timed_event),
            )
            timed_events.pprint(logger.debug)

        # Determine whether the events include a timed update
        update_time = IntegerTime.max_int_time

        if timed_events.num_events > 0:
            update_time = time_of_next_timed_event

        # Limit the major step end time to the simulation end time, major step limit,
        # or next periodic update time.
        # This is the mechanism used to advance time for systems that have
        # no states and no periodic events.
        # Discrete systems] when there are discrete periodic events, we use those
        # to determine each major step end time.
        # Feedthrough system] when there are just feedthrough blocks (no states or
        # events), use max_major_step_length to determine each major step end time.
        int_tf_limit = int_time + int_max_step_length
        int_tf = npa.min(
            npa.array(
                [
                    int_boundary_time,
                    int_tf_limit,
                    update_time,
                ]
            )
        )
        if not self.enable_tracing:
            logger.debug(
                "Expecting to integrate to t=%s",
                IntegerTime.as_decimal(int_tf),
            )

        # T-027a-followup-per-leaf-freeze: snapshot the pre-ODE
        # continuous state and ODE solver state when simulator-level
        # Zeno protection is enabled.  These are restored *after*
        # ``_advance_continuous_time`` for leaves whose Zeno latch was
        # already engaged at the START of this major step (carried in
        # by ``sim_state.zeno_active`` from step N-1) — mirroring the
        # leaf-level pattern where a latch set at step N-1 prevents
        # continuous-state evolution at step N.  This avoids the
        # pathological "freeze the just-completed reset map" replay.
        # Default-off: no snapshot, no extra ops — byte-equivalent.
        if self.zeno_protection_enabled:
            pre_ode_xc = context.continuous_state
            pre_ode_solver_state = sim_state.ode_solver_state
            prior_zeno_active = sim_state.zeno_active
            prior_any_active = jnp.any(
                jnp.asarray(prior_zeno_active, dtype=jnp.bool_),
            )
            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # carry the prior step's recovery-probe-fired mask through to
            # the inner ODE step's guard-recording site.  When event
            # ``i``'s latch was cleared by the recovery probe at step
            # N-1, this step (N) is the "first post-recovery" step and
            # ``w0[i]`` may need a direction-aware nudge to compensate
            # for the resting-state guard value (e.g. ``h=0`` for a
            # bouncing ball with ``max(h, 0)`` reset clamping).
            prior_recovery_just_cleared = sim_state.zeno_recovery_just_cleared
            # Initial ``per_event_triggered`` placeholder for the cond
            # branches' pytree-shape consistency.  The terminal-early
            # branch passes cdata through unchanged so we need a fixed
            # array shape here; ``_advance_continuous_time`` overrides
            # this with the actual post-step per-event mask.
            n_events = max(self.n_zero_crossing_events, 1)
            per_event_triggered_initial = jnp.zeros((n_events,), dtype=jnp.bool_)
        else:
            pre_ode_xc = None
            pre_ode_solver_state = None
            prior_zeno_active = None
            prior_any_active = None
            prior_recovery_just_cleared = None
            per_event_triggered_initial = None

        # Normally we will advance continuous time to the end of the major step
        # here. However, if a terminal event was triggered as part of the discrete
        # update, we should respect that and skip the continuous update.
        #
        # Construct the container used to hold various data related to advancing
        # continuous time.  This is passed to ODE solvers, zero-crossing
        # localization, and related functions.
        if self.system.has_continuous_state:
            leaves = jax.tree.leaves(context.continuous_state)
            dtype = leaves[0].dtype if leaves else jnp.empty(0).dtype
            context = context.with_time(jnp.asarray(context.time, dtype=dtype))
            if sim_state.ode_solver_state is not None:
                def _cast_leaf(x):
                    if isinstance(x, (float, np.floating)):
                        return jnp.asarray(x, dtype=dtype)
                    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                        return x.astype(dtype)
                    return x
                ode_solver_state = jax.tree.map(_cast_leaf, sim_state.ode_solver_state)
            else:
                ode_solver_state = sim_state.ode_solver_state
        else:
            ode_solver_state = sim_state.ode_solver_state

        cdata = ContinuousIntervalData(
            context=context,
            terminate_early=terminate_early,
            triggered=False,
            t0=int_time,
            tf=int_tf,
            results_data=sim_state.results_data,
            ode_solver_state=ode_solver_state,
            recovery_just_cleared=prior_recovery_just_cleared,
            per_event_triggered=per_event_triggered_initial,
            prior_zeno_active=prior_zeno_active,
        )
        cdata = backend.cond(
            (self.has_terminal_events & cdata.terminate_early),
            lambda cdata: cdata,  # Terminal event triggered - return immediately
            self._advance_continuous_time,  # Advance continuous time normally
            cdata,
        )

        # Unpack the results of the continuous time advance
        context = cdata.context
        terminate_early = cdata.terminate_early
        triggered = cdata.triggered
        int_tf = cdata.tf
        results_data = cdata.results_data
        ode_solver_state = cdata.ode_solver_state

        # Determine the reason why the major step ended.  Did a zero-crossing
        # trigger, did a timed event trigger, neither, or both?
        # terminate_early = terminate_early | zc_events.has_active_terminal
        logger.debug("Terminate early after major step: %s", terminate_early)
        end_reason = _determine_step_end_reason(
            triggered, terminate_early, int_tf, update_time
        )
        logger.debug("Major step end reason: %s", end_reason)

        # Conditionally activate timed events depending on whether the major step
        # ended as a result of a time trigger or zero-crossing event.
        timed_events = activate_timed_events(timed_events, end_reason)

        if self.major_step_callback:
            io_callback(self.major_step_callback, (), context.time)

        # T-113-followup-event-reprojection: project after the
        # continuous-integration phase.  When ``_advance_continuous_time``
        # ends on a localized ZC trigger, ``handle_events`` has just
        # applied the reset map and the algebraic states may have left
        # the manifold; projecting here re-establishes ``f_a = 0`` before
        # the next major step's discrete update sees the post-reset
        # state.  We project unconditionally — projection is a no-op
        # (zero Newton iterations, residual already below tol) on a
        # non-triggering step where state is already feasible, so the
        # extra cost on non-event steps is a single Newton residual
        # evaluation.  Default-off path skips the block at trace time.
        # No-op for systems without a mass matrix.  Runs before T-003a's
        # end-of-major-step projection so both hooks compose cleanly.
        if self.dae_reproject_after_events and getattr(
            self.system, "has_mass_matrix", False,
        ):
            from .dae_projection import project_constraints
            context = project_constraints(
                self.system,
                context,
                tol=self.dae_projection_tol,
                max_iter=self.dae_projection_max_iter,
            )

        # T-003a: opt-in DAE constraint projection.  Re-establishes
        # ``f_a(t, x, p) = 0`` after each major step by Newton-correcting
        # the algebraic component of the continuous state, holding the
        # differential component fixed.  No-op for systems without a mass
        # matrix or when the option is disabled (the default).
        if self.dae_projection_enabled and getattr(
            self.system, "has_mass_matrix", False,
        ):
            from .dae_projection import project_constraints
            context = project_constraints(
                self.system,
                context,
                tol=self.dae_projection_tol,
                max_iter=self.dae_projection_max_iter,
            )

        # T-003b: opt-in DAE drift monitor.  Computes ``||f_a||_∞`` and
        # emits a ``UserWarning`` (via ``jax.debug.callback`` so it works
        # under jit/vmap) when above the threshold.  Default-off path is
        # byte-equivalent — the entire block is skipped at trace time
        # when ``dae_drift_threshold is None``.  Runs *after* projection
        # so the warning reflects the post-correction residual.  The
        # mask-and-max is done inline rather than via
        # ``constraint_residual_norm`` because boolean indexing of a
        # tracer is not jit-safe; ``jnp.where`` is.
        if self.dae_drift_threshold is not None and getattr(
            self.system, "has_mass_matrix", False,
        ):
            from .dae_drift import algebraic_row_mask
            mask_np = algebraic_row_mask(self.system)
            if mask_np is not None and mask_np.any():
                xcdot = self.system.eval_time_derivatives(context)
                xcdot_flat = jnp.concatenate(
                    [jnp.ravel(leaf) for leaf in jax.tree.leaves(xcdot)]
                )
                mask = jnp.asarray(mask_np)
                residual_max = jnp.max(jnp.where(mask, jnp.abs(xcdot_flat), 0.0))
                threshold = jnp.asarray(self.dae_drift_threshold)
                jax.debug.callback(
                    _emit_dae_drift_warning,
                    context.time,
                    residual_max,
                    threshold,
                )

        # T-113 Phase 1: opt-in per-major-step DAE drift trace.
        # Default-off path is byte-equivalent — entire block is skipped
        # at trace time when ``self._dae_drift_monitor is None``
        # (i.e. when ``record_dae_drift=False`` or the system has no
        # mass matrix).  Same residual computation as T-003b above; we
        # do not share the value because T-003b's block is itself
        # gated on ``dae_drift_threshold is not None``, and forcing
        # them to share would couple two independently-opt-in switches.
        # The cost of recomputing ``||f_a||_∞`` is one extra
        # ``eval_time_derivatives`` per major step, only paid when the
        # user opted in to the trace.  Runs *after* projection so the
        # trace reflects the post-correction residual (matching T-003b).
        if self._dae_drift_monitor is not None:
            from .dae_drift import algebraic_row_mask as _alg_mask
            mask_np = _alg_mask(self.system)
            if mask_np is not None and mask_np.any():
                xcdot = self.system.eval_time_derivatives(context)
                xcdot_flat = jnp.concatenate(
                    [jnp.ravel(leaf) for leaf in jax.tree.leaves(xcdot)]
                )
                mask = jnp.asarray(mask_np)
                residual_max = jnp.max(
                    jnp.where(mask, jnp.abs(xcdot_flat), 0.0)
                )
                jax.debug.callback(
                    self._dae_drift_monitor.update,
                    context.time,
                    residual_max,
                )

        # T-027a-followup: simulator-level Zeno tracker + recovery probe.
        # Default-off path: ``zeno_protection_enabled=False`` skips the
        # entire block at Python level — zero ops compiled in, the carry
        # is byte-equivalent to the pre-followup state.  When enabled,
        # ``_update_zeno_tracking`` returns the new ``(tprev, active,
        # frozen_steps)`` triple, including the recovery-probe behaviour:
        # after K=zeno_recovery_period consecutive frozen steps the latch
        # is cleared for one step so the next guard-trigger check
        # naturally re-engages Zeno if the cascade is still active, or
        # stays cleared otherwise.
        if self.zeno_protection_enabled:
            # T-027a-followup-per-leaf-freeze: apply the freeze gate
            # using the PRIOR-step latch (``sim_state.zeno_active``)
            # before updating it.  When a leaf's latch was engaged at
            # step N-1, step N's ODE/ZC advance is rolled back for that
            # leaf — the latch from step N-1 prevents continuous-state
            # evolution at step N, mirroring the leaf-level pattern
            # (``_wrap_ode_for_zeno`` zeros the rhs when the discrete
            # zeno flag is set).  Other leaves' post-ODE state are kept.
            context, ode_solver_state = self._apply_per_leaf_zeno_freeze(
                pre_ode_xc,
                context,
                pre_ode_solver_state,
                ode_solver_state,
                prior_zeno_active,
            )
            # Time-advance fix-up for the frozen-by-prior-step case:
            # ``_advance_continuous_time`` would have clamped ``int_tf``
            # to the just-fired ZC trigger time (essentially equal to
            # ``int_t0`` for a dense cascade), and ``context.time`` to
            # the same.  Spinning on that would never let the recovery
            # probe drain ``zeno_frozen_steps`` to ``K``.
            #
            # T-027a-followup-multi-leaf-cascade-bug (partial fix): the
            # prior gate used ``prior_any_active`` (any leaf frozen),
            # which in a multi-leaf diagram could force-advance global
            # ``int_tf`` past an unfrozen leaf's just-localised ZC
            # trigger time, dropping that leaf's pending event handling.
            # This gate is now ``prior_all_active`` over leaves with
            # continuous state — the only case where there is no
            # genuine ODE progress for any leaf and therefore no harm
            # in jumping global time forward.  When only SOME leaves
            # are frozen, the natural ``int_tf`` from
            # ``_advance_continuous_time`` is kept so the still-
            # bouncing leaf's slice retains the time it actually
            # integrated to; the per-leaf rollback above already
            # restored the frozen leaves' continuous state.  Single-
            # leaf and identical-leaves-cascading-simultaneously
            # cases collapse to ``all == any`` so behaviour is
            # unchanged for them.
            #
            # T-027a-followup-multi-leaf-cascade-architecture (candidate
            # (c), 2026-05-01): the original staggered-cascade failure
            # — leaf at post-reset rest (h=0, v≈0+) integrating under
            # gravity to h<0 with ``positive_then_non_positive`` guards
            # rejecting the trigger because ``w0=0`` is not strictly
            # positive — is now resolved by ``_apply_recovery_w0_nudge``
            # (direction-aware ``w0`` nudge), ``_extract_per_event_triggered``
            # (per-event tracker advance), and ``_mask_triggered_for_active_latch``
            # (suppress latched events' triggers inside the inner ODE
            # step so still-bouncing leaves can advance).
            int_tf_planned = npa.min(
                npa.array([int_boundary_time, int_tf_limit, update_time])
            )
            prior_active_b = jnp.asarray(prior_zeno_active, dtype=jnp.bool_)
            # Per-leaf frozen reduction restricted to leaves with
            # continuous state.  Built from the same static
            # ``_cs_idx_to_event_positions`` map used by the freeze
            # gate so the two stay in sync.  Leaves with NO events
            # (still continuous-state-bearing — e.g. an integrator
            # block in a diagram with a separate bouncing ball)
            # contribute ``False`` to the reduction so their
            # continued ODE progress also blocks the global time-
            # skip; this is the safe default because their state is
            # advancing and any global ``int_tf`` jump would lose it.
            if self._n_continuous_leaves > 0:
                _leaf_frozen_terms = []
                for cs_i in range(self._n_continuous_leaves):
                    positions = self._cs_idx_to_event_positions.get(cs_i, ())
                    if positions:
                        _leaf_frozen_terms.append(
                            jnp.any(prior_active_b[jnp.asarray(positions)]),
                        )
                    else:
                        _leaf_frozen_terms.append(jnp.asarray(False))
                prior_all_active = jnp.all(jnp.stack(_leaf_frozen_terms))
            else:
                # No continuous-state leaves: fall back to the prior
                # ``any_active`` semantics (no rollback to align with).
                prior_all_active = jnp.asarray(prior_any_active, dtype=jnp.bool_)
            should_skip = prior_all_active & (
                jnp.asarray(int_tf, dtype=int_tf_planned.dtype) < int_tf_planned
            )
            int_tf = jnp.where(should_skip, int_tf_planned, int_tf)
            context = context.with_time(IntegerTime.as_decimal(int_tf))
            # When the freeze fully skipped the ODE-advance side
            # effects, the post-ODE ``triggered`` is masked False so
            # the next call to ``_update_zeno_tracking`` sees no firing
            # event.  This lets the recovery probe drain to K and clear
            # the latch so ``int_tf`` resumes ZC-localised stepping.
            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # use the per-event triggered mask plumbed back from
            # ``_advance_continuous_time`` so each event's tracker
            # update fires only for events that actually triggered on
            # this major step.  Falls back to the legacy scalar broadcast
            # when the per-event mask is unavailable (e.g. no
            # continuous-state path or no events in the system).
            per_event_triggered_in = cdata.per_event_triggered
            if per_event_triggered_in is None:
                triggered_b_for_tracker = jnp.asarray(triggered, dtype=jnp.bool_)
            else:
                triggered_b_for_tracker = jnp.asarray(
                    per_event_triggered_in, dtype=jnp.bool_,
                )
            triggered_for_tracker = jnp.where(
                should_skip,
                jnp.zeros_like(triggered_b_for_tracker),
                triggered_b_for_tracker,
            )
            zeno_tprev, zeno_active, zeno_frozen_steps = (
                self._update_zeno_tracking(
                    sim_state.zeno_tprev,
                    sim_state.zeno_active,
                    sim_state.zeno_frozen_steps,
                    triggered_for_tracker,
                    context.time,
                )
            )
            # T-027a-followup-multi-leaf-cascade-architecture (candidate (c)):
            # derive the per-event "recovery probe just fired" mask from
            # the latch transition.  ``should_probe`` is the only path
            # in ``_update_zeno_tracking`` that flips ``active[i]``
            # True->False on a single step; engagement only flips
            # False->True.  So ``prior_active & ~new_active`` is exactly
            # the recovery-probe-fires set.  This mask is consumed by
            # the NEXT major step's ``_advance_continuous_time`` to
            # apply the direction-aware ``w0`` nudge.  The mask self-
            # clears on the step after that because the latch is
            # already cleared (no new ``should_probe`` firing without
            # re-engagement).
            prior_active_b_for_rjc = jnp.asarray(
                sim_state.zeno_active, dtype=jnp.bool_,
            )
            new_active_b_for_rjc = jnp.asarray(zeno_active, dtype=jnp.bool_)
            zeno_recovery_just_cleared = (
                prior_active_b_for_rjc & ~new_active_b_for_rjc
            )
        else:
            zeno_tprev = sim_state.zeno_tprev
            zeno_active = sim_state.zeno_active
            zeno_frozen_steps = sim_state.zeno_frozen_steps
            zeno_recovery_just_cleared = sim_state.zeno_recovery_just_cleared

        return SimulatorState(
            step_end_reason=end_reason,
            context=context,
            timed_events=timed_events,
            int_time=int_tf,
            results_data=results_data,
            ode_solver_state=ode_solver_state,
            zeno_tprev=zeno_tprev,
            zeno_active=zeno_active,
            zeno_frozen_steps=zeno_frozen_steps,
            zeno_recovery_just_cleared=zeno_recovery_just_cleared,
        )

    # This method is marked private because it will be wrapped with a custom autodiff
    # rule to get the correct derivatives with respect to the end time of the
    # simulation interval using `_override_advance_to_vjp`.  This also copies the
    # docstring to the overridden function. Normally the wrapped attribute `advance_to`
    # is what should be called by users.
    def _advance_to(self, boundary_time: float, context: ContextBase) -> SimulatorState:
        """Core control flow logic for running a simulation.

        This is the main loop for advancing the simulation.  It is called by `simulate`
        or can be called directly if more fine-grained control is needed. This method
        essentially loops over "major steps" until the boundary time is reached. See
        the documentation for `simulate` for details on the order of operations in a
        major step.

        Args:
            boundary_time (float): The time to advance to.
            context (ContextBase): The current state of the system.

        Returns:
            SimulatorState:
                A named tuple containing the final state of the simulation, including
                the final context, a collection of pending timed events, and a flag
                indicating the reason that the most recent major step ended.

        Notes:
            API will change slightly as a result of WC-87, which will break out the
            initialization from the main loop so that `advance_to` can be called
            repeatedly.  See:
            https://jaxonomy.atlassian.net/browse/WC-87
        """

        system = self.system
        sim_state = self.initialize(context)
        end_reason = sim_state.step_end_reason
        context = sim_state.context
        timed_events = sim_state.timed_events
        int_boundary_time = IntegerTime.from_decimal(boundary_time)

        # We will be limiting each step by the max_major_step_length.  However, if this
        # is infinite we should just use the end time of the simulation to avoid
        # integer overflow.  This could be problematic if the end time of the
        # simulation is close to the maximum representable integer time, but we can come
        # back to that if it's an issue.
        int_max_step_length = IntegerTime.from_decimal(
            npa.minimum(self.max_major_step_length, boundary_time)
        )

        # Only activate timed events if the major step ended on a time trigger
        timed_events = activate_timed_events(timed_events, end_reason)

        # Called on the "True" branch of the conditional
        def _major_step(sim_state: SimulatorState) -> SimulatorState:
            return self._major_step(sim_state, int_boundary_time, int_max_step_length)

        def _cond_fun(sim_state: SimulatorState):
            return (sim_state.int_time < int_boundary_time) & (
                sim_state.step_end_reason != StepEndReason.TerminalEventTriggered
            )

        # Initialize the "carry" values for the main loop.
        if self.system.has_continuous_state:
            leaves = jax.tree.leaves(context.continuous_state)
            dtype = leaves[0].dtype if leaves else jnp.empty(0).dtype
            context = context.with_time(jnp.asarray(context.time, dtype=dtype))
            if sim_state.ode_solver_state is not None:
                def _cast_leaf(x):
                    if isinstance(x, (float, np.floating)):
                        return jnp.asarray(x, dtype=dtype)
                    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                        return x.astype(dtype)
                    return x
                ode_solver_state = jax.tree.map(_cast_leaf, sim_state.ode_solver_state)
            else:
                ode_solver_state = sim_state.ode_solver_state
        else:
            ode_solver_state = sim_state.ode_solver_state

        sim_state = SimulatorState(
            context=context,
            timed_events=timed_events,
            step_end_reason=end_reason,
            int_time=sim_state.int_time,
            results_data=sim_state.results_data,
            ode_solver_state=ode_solver_state,
            zeno_tprev=sim_state.zeno_tprev,
            zeno_active=sim_state.zeno_active,
            zeno_frozen_steps=sim_state.zeno_frozen_steps,
            zeno_recovery_just_cleared=sim_state.zeno_recovery_just_cleared,
        )

        logger.debug(
            "Running simulation from t=%s to t=%s", context.time, boundary_time
        )

        try:
            # Main loop call
            sim_state = self.while_loop(_cond_fun, _major_step, sim_state)
            logger.debug("Simulation complete at t=%s", sim_state.context.time)
        except KeyboardInterrupt:
            # NOTE: flag simulation as interrupted somewhere in sim_state
            logger.info("Simulation interrupted at t=%s", sim_state.context.time)

        # At the end of the simulation we need to handle any pending discrete updates
        # and store the solution one last time.
        # NOTE (WC-87): The returned simulator state can't be used with advance_to again,
        # since the discrete updates have already been performed. Should be broken out
        # into a `finalize` method as part of WC-87.

        # update discrete state to x+ at the simulation end_time
        if self.results_recorder.save_time_series:
            logger.debug("Finalizing solution...")
            # 1] do discrete update (will skip if the simulation was terminated early)
            context, _terminate_early = self._handle_discrete_update(
                sim_state.context, sim_state.timed_events
            )
            # 2] do update solution
            context = context.refresh_port_cache()
            # T-012a-followup: pass solver_state so the final sample also
            # snapshots the interpolant covering [t_prev, t_end].
            results_data = self.results_recorder.save(
                sim_state.results_data,
                context,
                ode_solver_state=sim_state.ode_solver_state,
            )
            sim_state = sim_state._replace(
                context=context,
                results_data=results_data,
            )
            logger.debug("Done finalizing solution")

        return sim_state


def _bounded_while_loop(
    cond_fun: Callable,
    body_fun: Callable,
    val: Any,
    max_steps: int,
) -> Any:
    """Run a while loop with a bounded number of steps.

    This is a workaround for the fact that JAX's `lax.while_loop` does not support
    reverse-mode autodiff.  The `max_steps` bound can usually be determined
    automatically during calls to `simulate` - see notes on `max_major_steps` in
    `SimulatorOptions` and `estimate_max_major_steps`.
    """

    def _loop_fun(_i, val):
        return backend.cond(
            cond_fun(val),
            body_fun,
            lambda val: val,
            val,
        )

    return backend.fori_loop(0, max_steps, _loop_fun, val)


def _check_backend(options: SimulatorOptions) -> tuple[str, bool]:
    """Check if JAX tracing can be used to simulate this system."""

    math_backend = options.math_backend or "auto"
    if math_backend == "auto":
        math_backend = backend.active_backend

    if math_backend != "jax":
        enable_tracing = False

    else:
        # Otherwise return whatever `options` requested
        enable_tracing = options.enable_tracing

    return math_backend, enable_tracing


def _determine_step_end_reason(
    guard_triggered: bool,
    terminate_early: bool,
    tf: int,
    update_time: int,
) -> StepEndReason:
    """Determine the reason why the major step ended."""
    logger.debug("[_determine_step_end_reason]: tf=%s, update_time=%s", tf, update_time)
    logger.debug("[_determine_step_end_reason]: guard_triggered=%s", guard_triggered)

    # If the integration terminated due to a triggered event, determine whether
    # there are any other events that should be triggered at the same time.
    guard_reason = npa.where(
        tf == update_time,
        StepEndReason.BothTriggered,
        StepEndReason.GuardTriggered,
    )

    # No guard triggered; handle integration as usual.
    no_guard_reason = npa.where(
        tf == update_time,
        StepEndReason.TimeTriggered,
        StepEndReason.NothingTriggered,
    )

    reason = npa.where(guard_triggered, guard_reason, no_guard_reason)

    # No matter why the integration terminated, if a "terminal" event is also
    # active, that will be the overriding reason for the termination.
    return npa.where(terminate_early, StepEndReason.TerminalEventTriggered, reason)


def _next_sample_time(current_time: int, event_data: PeriodicEventData) -> int:
    """Determine when the specified periodic event happens next.

    This is a helper function for `_next_update_time` for a specific event.
    """

    period, offset = event_data.period_int, event_data.offset_int

    # If we shift the current time by the offset, what would the index of the
    # next periodic sample time be?  This tells us how many samples from the
    # offset we are in either direction.  For example, if offset=dt and t=0,
    # the next "k" value will be -1.
    next_k = (current_time - offset) // period

    # What would the next periodic sample time be?  If the period is infinite,
    # the next sample time is also infinite.  This value is shifted back to the
    # original time frame by adding the offset.  If the sample is more than one
    # period away from the offset, this will be negative.
    next_t = npa.where(
        npa.isfinite(event_data.period),
        offset + next_k * period,
        period,
    )

    # If we are in between samples, next_t should be strictly greater than
    # the current time and that should be used as the target major step end time.
    # However, if we are at t = offset + k * period for some k, then the
    # calculation above will give us next_k = k and therefore next_t = t.
    # In this case we should bump to the next time in the series.
    next_sequence_time = npa.where(
        next_t > current_time,
        next_t,
        offset + (next_k + 1) * period,
    )

    return npa.where(
        current_time < offset,
        offset,
        next_sequence_time,
    )


def _next_update_time(periodic_events: EventCollection, current_time: int) -> int:
    """Compute next update time over all events in the periodic_events collection.

    This returns a tuple of the minimum next sample time along with a pytree with
    the same structure as `periodic_events` indicating which events are active at
    the next sample time.
    """
    periodic_events = periodic_events.mark_all_inactive()

    # 0. If no events, return an infinite time and empty event collection
    if not periodic_events.has_events:
        return IntegerTime.max_int_time, periodic_events

    # 1. Compute the next sample time for each event
    def _replace_sample_time(event_data):
        return dataclasses.replace(
            event_data,
            next_sample_time=_next_sample_time(current_time, event_data),
        )

    timed_events = jax.tree_util.tree_map(
        _replace_sample_time,
        periodic_events,
        is_leaf=is_event_data,
    )

    def _get_next_sample_time(event_data: PeriodicEventData) -> int:
        return event_data.next_sample_time

    # 2. Find the minimum next sample time across all events
    min_time = jax.tree_util.tree_reduce(
        npa.minimum,
        jax.tree_util.tree_map(
            _get_next_sample_time,
            timed_events,
            is_leaf=is_event_data,
        ),
    )

    # 3. Find the events corresponding to the minimum time by updating the event data `active` field
    def _replace_active(event_data: PeriodicEventData):
        return dataclasses.replace(
            event_data,
            active=(event_data.next_sample_time == min_time),
        )

    active_events = jax.tree_util.tree_map(
        _replace_active,
        timed_events,
        is_leaf=is_event_data,
    )
    return min_time, active_events


def activate_timed_events(
    timed_events: EventCollection, end_reason: StepEndReason
) -> EventCollection:
    """Conditionally activate timed events.

    Only activate timed events if the major step ended on a time trigger and
    the event was already marked active (by the timing calculation). This will
    deactivate timed events if they were pre-empted by a zero-crossing.
    """

    deactivate = (end_reason != StepEndReason.TimeTriggered) & (
        end_reason != StepEndReason.BothTriggered
    )

    def activation_fn(event_data: PeriodicEventData):
        return event_data.active & ~deactivate

    return timed_events.activate(activation_fn)


# Zero-crossing guard functions are now in zero_crossing_handler.py.
# They are imported at the top of this file for backward compatibility.


#
# Custom VJP for advancing continuous time with an ODE solver
#
#
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 5))
def _odeint(solver: ODESolverBase, ode_rhs, solver_state, tf, context, checkpoint=True):
    """Unguarded ODE integration.

    This function unconditionally advances time to the end of an interval.
    It does not check for any zero-crossing events.  As such, it is not actually
    used directly in the simulation loop.  However, the adjoint (autodiff rule) for
    the unguarded ODE solve is more straightforward than the guarded version, so
    this is wrapped and called by the autodiff rule for the guarded ODE solve.

    Since it is _only_ called in the forward pass of the custom autodiff rule for
    the guarded ODE solve, it does not need to be conditionally wrapped by the simulator
    as the guarded solve does. Hence we can define it as a standalone function.
    """
    max_checkpoints = solver.max_checkpoints

    # Close over the additional arguments so that the RHS function has the
    # signature `func(y, t)`.  We can't do this anywhere upstream because
    # the data in the context has to be differentiable.
    def _func(y, t):
        return ode_rhs(y, t, context)

    def _ode_step(solver_state):
        # Advance the ODE solver one step
        return solver.step(_func, tf, solver_state)

    def cond_fun(solver_state):
        t, dt = solver_state.t, solver_state.dt
        return (t < tf) & (dt > 0)

    # This function does a sort of simplified recursive checkpointing, where we start
    # by filling up the checkpoints array with every minor step. Once the array is
    # full, every other checkpoint gets compressed to the first half of the array, and
    # the "depth" (how frequently we checkpoint) doubles.  This is a bit of a hack to
    # get around the fact that we can't resize the array in JAX and we don't know
    # beforehand how many minor steps we'll need.
    def body_fun(carry):
        # The "depth" is how many ODE solver steps are taken in between
        # checkpoints.  It will double every time the checkpoint array
        # fills up.  The "index" is the current index in the checkpoint
        # array. It will increment every "depth" steps. The loop counter "i"
        # tracks the number of steps taken since the last checkpoint (so it
        # counts from 0 to "depth" before "index" is incremented.
        i, index, depth, solver_state, checkpoints = carry

        # Step with the ODE solver
        solver_state = _ode_step(solver_state)

        # Check if we need to store the current state and time in the checkpoints array
        # (chk_step) and if we've reached the end of the checkpoint array (ext_step).
        chk_step = i + 1 == depth
        ext_step = chk_step & (index + 1 == max_checkpoints)
        i = jnp.where(chk_step, 0, i + 1)

        index = jnp.where(chk_step, index + 1, index)
        index = jnp.where(ext_step, max_checkpoints // 2, index)

        # If the index reaches the end of the array, we have to "extend" the checkpoint
        # array without being able to resize it.  We do this by moving every other
        # checkpoint to the first half of the array and doubling the depth.
        # Set all the second half of the array to NaN to mark unused checkpoints.
        depth = jnp.where(ext_step, 2 * depth, depth)
        checkpoints = jnp.where(
            ext_step,
            checkpoints.at[: max_checkpoints // 2]
            .set(checkpoints[::2])
            .at[max_checkpoints // 2 :]
            .set(jnp.nan),
            checkpoints,
        )

        # Store the current state and time in the checkpoints array
        # if we have reached the next checkpoint.
        yt = jnp.append(solver_state.y, solver_state.t)
        checkpoints = jnp.where(chk_step, checkpoints.at[index].set(yt), checkpoints)

        return i, index, depth, solver_state, checkpoints

    if checkpoint:
        # The loop collects the times and states to "checkpoint" the adjoint simulation
        # at certain time steps.  Note that this is returned as auxiliary data - the
        # _odeint function called by Simulator evaluates _odeint_fwd (below), which
        # only returns the solver state and not the collected time series data.
        checkpoints = jnp.full((max_checkpoints, solver_state.y.size + 1), jnp.nan)

        # Store the initial state and time in the checkpoints array
        checkpoints = checkpoints.at[0].set(jnp.append(solver_state.y, solver_state.t))
        carry = (0, 0, 1, solver_state, checkpoints)

        _i, index, _depth, solver_state, checkpoints = lax.while_loop(
            lambda carry: cond_fun(carry[3]), body_fun, carry
        )

        # Save the last state in the checkpoints array (will be used to
        # initialize adjoint pass)
        index = jnp.maximum(index + 1, max_checkpoints - 1)
        yt = jnp.append(solver_state.y, solver_state.t)
        checkpoints = checkpoints.at[index].set(yt)

        # Fill unused entries with the final state
        checkpoints = jnp.where(jnp.isnan(checkpoints), yt, checkpoints)
        return solver_state, checkpoints

    return lax.while_loop(cond_fun, _ode_step, solver_state)


# The "forward pass" through the ODE solve, but don't save any time-series data
def _odeint_fwd(solver, ode_rhs, solver_state, tf, context, checkpoint=True):
    solver_state, checkpoints = _odeint(solver, ode_rhs, solver_state, tf, context)
    ts = checkpoints[:, -1]
    ys = checkpoints[:, :-1]
    residuals = (solver_state, tf, context, ts, ys)
    return solver_state, residuals


# The "reverse pass" through the ODE solve, using an augmented dynamical
# system with the adjoint variables.
def _odeint_adj(solver, ode_rhs, _checkpoint, residuals, adjoints):
    primals, tf, context, ts, ys = residuals

    # The args may contain bools, ints, or otherwise non-differentiable data.
    # Here we can split the args into dynamic and static components, and only
    # pass the dynamic args through the adjoint system.
    dynamic_args, static_args = eqx.partition(context, eqx.is_inexact_array_like)

    yf = primals.y
    yf_bar = adjoints.y

    # NOTE: The Cao et al. (2003) mass-matrix adjoint IC correction has been moved
    # to _wrapped_advance_to_adj in autodiff_rules.py, where it is applied exactly
    # ONCE at the start of the backward sweep.  Applying it here (inside _odeint_adj)
    # would corrupt the adjoint for hybrid CT+DT systems where _odeint_adj is called
    # multiple times (once per major step), because the correction would be applied at
    # every CT-interval boundary instead of only at the simulation terminal time T.
    #
    # The validation that the mass matrix has canonical semi-explicit form is also done
    # there (with permutation support for non-canonical orderings).

    init_adj_state = (
        yf,
        yf_bar,
        0.0,
        jax.tree_util.tree_map(jnp.zeros_like, dynamic_args),
    )
    solver_state, adj_dynamics = solver.initialize_adjoint(
        ode_rhs, init_adj_state, tf, context
    )

    # Mimic the forward solve but with the adjoint dynamics and using the checkpointed
    # values for restarts.
    ny = len(yf)
    n_steps = len(ts)

    def body_fun(i, solver_state):
        # Update the solver state entries corresponding to the primal values with the
        # values from the forward simulation.
        idx = n_steps - i
        adj_state = solver_state.y.at[:ny].set(ys[idx])
        solver_state = solver_state.with_state_and_time(adj_state, -ts[idx])
        return _odeint(
            solver, adj_dynamics, solver_state, -ts[idx - 1], context, checkpoint=False
        )

    solver_state = lax.fori_loop(0, n_steps, body_fun, solver_state)

    _, y0_bar, t0_bar, ctx_bar = solver_state.unravel(solver_state.y)

    # The Jacobian with respect to the final time is just the time derivative of
    # the state at the final time.
    tf_bar = jnp.dot(ode_rhs(yf, tf, context), yf_bar)

    # Recombine the dynamic and static args
    ctx_bar = eqx.combine(ctx_bar, static_args)
    solver_state_bar = adjoints.with_state_and_time(y0_bar, t0_bar)

    return (solver_state_bar, tf_bar, ctx_bar)


_odeint.defvjp(_odeint_fwd, _odeint_adj)
