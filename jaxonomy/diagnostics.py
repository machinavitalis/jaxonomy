# SPDX-License-Identifier: MIT

"""Post-hoc analysis helpers that catch common silent failure modes.

These utilities operate on the results of a `jaxonomy.simulate` call and
surface conditions that almost always indicate a bug, but which the
simulator itself cannot detect — typically because the integration runs
cleanly even when the system is producing meaningless output.

Each function returns a structured report (dataclass). When called via
``warn=True`` (the default), the function additionally emits a
``UserWarning`` so that the failure mode is impossible to miss in a
notebook or CI log.

The three diagnostics here all came directly from real bugs found while
authoring the returning-booster tutorial series:

- ``analyze_saturation`` would have surfaced the bang-bang gimbal
  oscillation in v3-v7 of Part 3 — the controller hit ±10° saturation
  for ~95% of the boost-back phase, which is a textbook indicator of
  a controller-tuning bug or, as it turned out, a hidden sign error.

- ``analyze_phase_activity`` would have surfaced the dispatcher bug in
  v7-v17 of Part 3 — the boost-back state never fired because its guard
  encoded an implicit assumption about the initial velocity that broke
  when the IC was scoped down. The diagnostic catches "this phase was
  scheduled in the state machine but never selected during simulation".

- ``analyze_control_oscillation`` catches limit-cycle behaviour by
  counting sign changes per unit time. A well-tuned controller produces
  a smooth control signal; a bang-bang one zero-crosses many times per
  second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import warnings

import numpy as np


__all__ = [
    "SaturationReport",
    "PhaseActivityReport",
    "OscillationReport",
    "HorizonCompletionReport",
    "analyze_saturation",
    "analyze_phase_activity",
    "analyze_control_oscillation",
    "analyze_horizon_completion",
]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class SaturationReport:
    """Result of `analyze_saturation`.

    Attributes:
        n_samples: number of samples analyzed.
        fraction_at_upper: fraction of samples within ``atol`` of upper limit.
        fraction_at_lower: fraction of samples within ``atol`` of lower limit.
        fraction_saturated: sum of upper + lower fractions.
        warning_triggered: True if `fraction_saturated >= warn_threshold`.
        message: Human-readable summary.
    """

    n_samples: int
    fraction_at_upper: float
    fraction_at_lower: float
    fraction_saturated: float
    warning_triggered: bool
    message: str

    def __repr__(self) -> str:
        return (
            f"SaturationReport(saturated={self.fraction_saturated:.1%}, "
            f"upper={self.fraction_at_upper:.1%}, lower={self.fraction_at_lower:.1%}, "
            f"warn={self.warning_triggered})"
        )


@dataclass
class PhaseActivityReport:
    """Result of `analyze_phase_activity`.

    Attributes:
        phases_seen: sorted list of unique phase values observed.
        fraction_per_phase: dict mapping phase value → fraction of samples in it.
        never_fired: list of phase values that appear in ``expected_phases``
            but were never active during simulation.
        warning_triggered: True if any expected phase was never active.
        message: Human-readable summary.
    """

    phases_seen: list
    fraction_per_phase: dict
    never_fired: list
    warning_triggered: bool
    message: str

    def __repr__(self) -> str:
        return (
            f"PhaseActivityReport(seen={self.phases_seen}, "
            f"never_fired={self.never_fired}, "
            f"warn={self.warning_triggered})"
        )


@dataclass
class OscillationReport:
    """Result of `analyze_control_oscillation`.

    Attributes:
        zero_crossings: number of sign changes detected.
        duration: time span analyzed.
        crossings_per_second: zero_crossings / duration.
        warning_triggered: True if `crossings_per_second >= warn_threshold`.
        message: Human-readable summary.
    """

    zero_crossings: int
    duration: float
    crossings_per_second: float
    warning_triggered: bool
    message: str

    def __repr__(self) -> str:
        return (
            f"OscillationReport(crossings/s={self.crossings_per_second:.2f}, "
            f"warn={self.warning_triggered})"
        )


@dataclass
class HorizonCompletionReport:
    """Result of :func:`analyze_horizon_completion`.

    Attributes:
        final_value: the accumulator's value at simulation end (the
            ``state.continuous_state`` slot the user reads as the
            differentiable cost).
        t_end: the simulation horizon supplied to the diagnostic.
        completion_ratio: ``final_value / t_end`` (1.0 = never-finished).
        warning_triggered: True if ``completion_ratio >= atol_fraction``.
        message: Human-readable summary.
    """

    final_value: float
    t_end: float
    completion_ratio: float
    warning_triggered: bool
    message: str

    def __repr__(self) -> str:
        return (
            f"HorizonCompletionReport(ratio={self.completion_ratio:.3f}, "
            f"warn={self.warning_triggered})"
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def analyze_saturation(
    values,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    atol: float = 1e-3,
    warn_threshold: float = 0.5,
    name: str = "signal",
    warn: bool = True,
    mode: str = "symmetric",
) -> SaturationReport:
    """Report the fraction of time a signal sits at its saturation limit(s).

    A well-tuned controller should rarely saturate its actuators. A signal
    saturated for >50% of the simulation is almost always a sign of: too-
    aggressive gains, insufficient actuator authority, a sign error, or a
    reference that is dynamically infeasible. Surfacing this number turns
    a silent integration into a loud warning.

    Args:
        values: 1-D or N-D array of signal values. The whole array is
            flattened along the time axis (axis 0) for the fraction count.
        lower: lower saturation limit (e.g., ``-DELTA_MAX``). Pass None to
            skip lower-bound detection.
        upper: upper saturation limit. Pass None to skip upper-bound detection.
        atol: tolerance for "at the limit" — usually 0.1% of the range.
        warn_threshold: fraction at which a UserWarning is emitted.
        name: name to include in the warning message.
        warn: if True, emit a UserWarning when threshold exceeded.
        mode: ``"symmetric"`` (default), ``"upper_only"``, or
            ``"lower_only"``. On one-sided actuators (throttle pedal
            ``[0, 1]``, brake, PWM duty cycle, valve opening), the
            *natural rest state* sits at one rail — counting that rail
            as "saturated" produces misleading warnings on every motor
            / vehicle / PWM tutorial. ``"upper_only"`` treats the
            ``upper`` rail as the saturation; ``"lower_only"`` treats
            the ``lower`` rail. Both still *report* both fractions in
            the returned :class:`SaturationReport` for diagnostic
            visibility; only the warning threshold uses the active
            rail. ``mode`` is also auto-promoted to the right one-sided
            flavour when exactly one of ``lower``/``upper`` is supplied
            (matching the documented workaround), so a single-rail call
            picks up the right semantics without extra plumbing.

    Returns:
        A `SaturationReport`.

    Example:
        >>> u = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 0.8, 1.0])  # mostly at upper
        >>> r = analyze_saturation(u, lower=-1.0, upper=1.0, name="throttle", warn=False)
        >>> r.fraction_saturated
        0.5714285714285714

    Example (one-sided throttle pedal, default ``mode``):
        >>> # Sitting at 0 (idle) is the natural rest state, not a saturation.
        >>> u = np.concatenate([np.ones(40), np.zeros(25), np.linspace(0, 1, 35)])
        >>> r = analyze_saturation(u, lower=0.0, upper=1.0, mode="upper_only",
        ...                        name="throttle", warn=False)
        >>> r.warning_triggered  # only counts the upper rail against the threshold
        False
    """
    if mode not in ("symmetric", "upper_only", "lower_only"):
        raise ValueError(
            f"analyze_saturation: mode must be one of 'symmetric', "
            f"'upper_only', 'lower_only'; got {mode!r}."
        )

    arr = np.asarray(values)
    if arr.size == 0:
        return SaturationReport(
            n_samples=0, fraction_at_upper=0.0, fraction_at_lower=0.0,
            fraction_saturated=0.0, warning_triggered=False,
            message="(empty input)",
        )

    n = arr.size
    frac_upper = 0.0
    if upper is not None:
        frac_upper = float(np.mean(np.abs(arr - upper) <= atol))
    frac_lower = 0.0
    if lower is not None:
        frac_lower = float(np.mean(np.abs(arr - lower) <= atol))
    frac_sat = frac_upper + frac_lower

    # Auto-promote ``mode`` when the user supplied exactly one rail —
    # the documented workaround for one-sided actuators (passing only
    # ``upper=`` for a throttle / valve / PWM signal). Explicit
    # ``mode=`` always wins.
    if mode == "symmetric":
        if upper is not None and lower is None:
            effective_mode = "upper_only"
        elif lower is not None and upper is None:
            effective_mode = "lower_only"
        else:
            effective_mode = "symmetric"
    else:
        effective_mode = mode

    if effective_mode == "upper_only":
        threshold_frac = frac_upper
    elif effective_mode == "lower_only":
        threshold_frac = frac_lower
    else:
        threshold_frac = frac_sat

    triggered = threshold_frac >= warn_threshold
    parts = []
    if upper is not None:
        parts.append(f"{frac_upper:.0%} at upper={upper:.3g}")
    if lower is not None:
        parts.append(f"{frac_lower:.0%} at lower={lower:.3g}")
    mode_note = ""
    if effective_mode != "symmetric":
        rail = "upper" if effective_mode == "upper_only" else "lower"
        mode_note = f" [{rail}_only mode; only {rail} rail counts against threshold]"
    msg = (
        f"{name}: {', '.join(parts)}.{mode_note} "
        f"Total saturated: {frac_sat:.0%} of {n} samples."
    )
    if triggered and warn:
        warnings.warn(
            f"[analyze_saturation] {msg} "
            f"This is above the {warn_threshold:.0%} threshold and usually "
            f"indicates a controller-tuning bug or insufficient actuator "
            f"authority.",
            UserWarning,
            stacklevel=2,
        )

    return SaturationReport(
        n_samples=n,
        fraction_at_upper=frac_upper,
        fraction_at_lower=frac_lower,
        fraction_saturated=frac_sat,
        warning_triggered=triggered,
        message=msg,
    )


def analyze_phase_activity(
    phase_signal,
    *,
    expected_phases: Optional[Sequence] = None,
    name: Optional[str] = None,
    warn: bool = True,
) -> PhaseActivityReport:
    """Report how often each phase / discrete mode was active.

    Use this on the output of a state-machine block (or any signal that
    encodes a discrete mode integer). If you declared K phases but
    simulation shows only K-1 of them were ever active, you almost
    certainly have a guard-condition bug. We hit this bug in Part 3 of
    the returning-booster tutorial: the boost-back state's guard
    encoded an assumption about the IC that broke when the IC was
    scoped down, silently disabling BBB for eleven iterations.

    Args:
        phase_signal: 1-D array of integer (or roundable-to-integer)
            phase values over time.
        expected_phases: iterable of phase values you expected to see.
            If provided, missing phases trigger a warning.
        name: optional label prefixed to the warning message, matching
            the convention used by :func:`analyze_saturation` and
            :func:`analyze_control_oscillation` (e.g. ``name="gear"``).
        warn: if True, emit a UserWarning on missing phases.

    Returns:
        A `PhaseActivityReport`.

    Example:
        >>> phases = np.array([0, 0, 1, 1, 2, 2, 2])  # phase 3 never fires
        >>> r = analyze_phase_activity(phases, expected_phases=[0, 1, 2, 3], warn=False)
        >>> r.never_fired
        [3]
    """
    arr = np.asarray(phase_signal).reshape(-1)
    if arr.size == 0:
        return PhaseActivityReport(
            phases_seen=[], fraction_per_phase={}, never_fired=list(expected_phases or []),
            warning_triggered=bool(expected_phases),
            message="(empty input)",
        )

    # Coerce to integer phases; tolerate floats from a state-machine
    # output port like 0.0, 1.0, 2.0.
    int_phases = np.round(arr).astype(int)
    uniq, counts = np.unique(int_phases, return_counts=True)
    frac_map = {int(p): float(c) / float(arr.size) for p, c in zip(uniq, counts)}
    phases_seen = sorted(frac_map.keys())

    never_fired = []
    if expected_phases is not None:
        expected = [int(p) for p in expected_phases]
        never_fired = sorted([p for p in expected if p not in frac_map])

    triggered = bool(never_fired)
    label = f"{name}: " if name else ""
    msg = (
        f"{label}Phases observed: {phases_seen} "
        f"(fractions: {{{', '.join(f'{p}: {frac_map[p]:.0%}' for p in phases_seen)}}})"
    )
    if never_fired:
        msg += f". NEVER FIRED: {never_fired}"

    if triggered and warn:
        warnings.warn(
            f"[analyze_phase_activity] {msg} "
            f"A state-machine phase that never fires is almost always a "
            f"guard-condition bug. Check the transition guards.",
            UserWarning,
            stacklevel=2,
        )

    return PhaseActivityReport(
        phases_seen=phases_seen,
        fraction_per_phase=frac_map,
        never_fired=never_fired,
        warning_triggered=triggered,
        message=msg,
    )


def analyze_control_oscillation(
    values,
    times,
    *,
    warn_threshold_per_second: float = 5.0,
    name: str = "signal",
    detrend: bool = True,
    warn: bool = True,
) -> OscillationReport:
    """Detect bang-bang / limit-cycle behaviour by counting sign changes.

    A well-tuned controller produces a smooth control signal; bang-bang
    behaviour shows up as many zero crossings per unit time, often near
    actuator saturation. Combined with `analyze_saturation`, this is a
    strong signal of a sign error or grossly mistuned PD gains. The
    Part-3 retrograde-tracking BBB controller crossed zero ~3 times per
    second during boost-back — a value that should set off any alarm.

    Args:
        values: 1-D array of control signal samples.
        times: 1-D array of sample times (same length as values).
        warn_threshold_per_second: zero-crossing rate above which a warning
            is emitted.
        name: name to include in the warning message.
        detrend: if True, subtract the mean before counting crossings (so
            a signal that hovers near a nonzero set point isn't reported
            as silently oscillating around zero).
        warn: if True, emit a UserWarning when threshold exceeded.

    Returns:
        An `OscillationReport`.
    """
    v = np.asarray(values).reshape(-1)
    t = np.asarray(times).reshape(-1)
    if v.size < 3 or t.size != v.size:
        return OscillationReport(
            zero_crossings=0, duration=0.0, crossings_per_second=0.0,
            warning_triggered=False,
            message="(too few samples or time mismatch)",
        )
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return OscillationReport(
            zero_crossings=0, duration=duration, crossings_per_second=0.0,
            warning_triggered=False, message="(non-positive duration)",
        )

    series = v - np.mean(v) if detrend else v
    # Count sign changes
    signs = np.sign(series)
    # Ignore zeros — treat them as not-a-crossing
    signs_nonzero = signs[signs != 0]
    if signs_nonzero.size < 2:
        crossings = 0
    else:
        crossings = int(np.sum(np.diff(signs_nonzero) != 0))
    rate = crossings / duration

    triggered = rate >= warn_threshold_per_second
    msg = (
        f"{name}: {crossings} sign changes over {duration:.2f} s "
        f"= {rate:.2f} crossings/s"
    )
    if triggered and warn:
        warnings.warn(
            f"[analyze_control_oscillation] {msg}. "
            f"This is above the {warn_threshold_per_second:.1f}/s threshold "
            f"and usually indicates bang-bang behaviour (PD-gain or sign bug).",
            UserWarning,
            stacklevel=2,
        )

    return OscillationReport(
        zero_crossings=crossings,
        duration=duration,
        crossings_per_second=rate,
        warning_triggered=triggered,
        message=msg,
    )


def analyze_horizon_completion(
    final_value,
    t_end: float,
    *,
    atol_fraction: float = 0.99,
    name: str = "cost-accumulator",
    warn: bool = True,
) -> HorizonCompletionReport:
    """Detect when a smoothed-indicator integrator never crossed its event.

    The canonical "cost as integrator" pattern under autodiff (Part 2 of
    the F1 tutorial series, ``LapTimeAccumulator``, ``FuelEmptyTime``,
    ``MissionDuration``, etc.) integrates a smooth one-shot indicator
    ``0.5 * (1 - tanh((s - S)/sigma))`` from ``0`` to ``t_end``. If the
    underlying system actually crosses the event (``s >= S``) before
    ``t_end``, the integral equals the event time and its gradient w.r.t.
    the system parameters is meaningful. If the system *never* crosses
    the event (a short ``t_end`` for autodiff tractability, or a setup
    too far off to ever finish), the integral equals ``t_end * 1.0 =
    t_end`` regardless of parameters — and ``jax.grad`` faithfully
    returns zeros that look like real (tiny) sensitivities.

    Call this diagnostic on the accumulator's final continuous-state
    value once the simulation has returned, before reporting any
    gradients downstream. When the integrator pegged at ``~t_end`` the
    diagnostic fires a ``UserWarning`` so the gradients aren't silently
    misinterpreted as a per-parameter sensitivity.

    Args:
        final_value: The accumulator's final value (typically
            ``results.context[acc.system_id].continuous_state[0]``).
        t_end: The simulation horizon used.
        atol_fraction: Trigger the warning when ``final_value / t_end >=
            atol_fraction``. Default ``0.99`` is conservative — a real
            lap-completion run with ``t_end`` ~30% above the lap time
            sits at ``ratio ≈ lap_time / t_end ≈ 0.77`` and won't fire.
        name: Label for the warning text (e.g. ``"laptime"``,
            ``"fuel"``).
        warn: When True (default), emit the ``UserWarning``; pass
            ``False`` to just get the report back.

    Returns:
        :class:`HorizonCompletionReport`.

    Example:
        >>> # A simulation where the lap never finished:
        >>> r = analyze_horizon_completion(final_value=120.0, t_end=120.0,
        ...                                name="laptime", warn=False)
        >>> r.warning_triggered
        True
        >>> # A real lap completion before t_end:
        >>> r = analyze_horizon_completion(final_value=90.0, t_end=120.0,
        ...                                name="laptime", warn=False)
        >>> r.warning_triggered
        False
    """
    if t_end <= 0:
        raise ValueError(
            f"analyze_horizon_completion: t_end must be > 0, got {t_end}"
        )
    if not (0.0 < atol_fraction <= 1.0):
        raise ValueError(
            f"analyze_horizon_completion: atol_fraction must lie in (0, 1], "
            f"got {atol_fraction}"
        )

    final_val = float(np.asarray(final_value).reshape(()))
    ratio = final_val / float(t_end)
    triggered = ratio >= atol_fraction

    msg = (
        f"{name}: final_value = {final_val:.4g}, t_end = {t_end:.4g}, "
        f"completion_ratio = {ratio:.4f}."
    )
    if triggered:
        msg += (
            f" Integrator pegged at the horizon — the underlying event "
            f"(e.g. lap completion, fuel-empty crossing) likely did not "
            f"fire before t_end. jax.grad of this readout is "
            f"mathematically zero; any per-parameter sensitivity you "
            f"report will be meaningless."
        )

    if triggered and warn:
        warnings.warn(
            f"[analyze_horizon_completion] {msg}",
            UserWarning,
            stacklevel=2,
        )

    return HorizonCompletionReport(
        final_value=final_val,
        t_end=float(t_end),
        completion_ratio=ratio,
        warning_triggered=triggered,
        message=msg,
    )
