# SPDX-License-Identifier: MIT

"""Rate-group introspection helpers for T-105 (Multirate Sample Times).

This module is the *Phase 1* deliverable for T-105.  It provides a
read-only inspection layer on top of the periodic-event machinery the
simulator already owns:

* a tiny :class:`SampleTime` value type (``continuous``, ``discrete``,
  ``constant``, ``inherited``);
* :func:`infer_block_sample_time` that introspects a ``LeafSystem``'s
  declared ``periodic_events`` and returns the implied ``SampleTime``;
* :func:`group_blocks_by_rate` that buckets a built ``Diagram``'s leaves
  by sample-time period;
* :func:`detect_rate_mismatches` that walks ``diagram.connection_map``
  and emits :class:`RateMismatchWarning` (or raises
  :class:`RateMismatchError`) wherever an output of one discrete rate
  feeds into a downstream block of a different discrete rate without an
  explicit rate-transition block in between.

Phase 1 is *strictly read-only*: it never mutates the diagram, never
inserts ``RateTransition`` blocks (T-123 / Phase 2), and never alters
the simulator's existing ``_next_update_time`` priority-queue sweep.
The goal is purely to give users a diagnostic that surfaces multirate
modelling errors *before* the simulator runs.

Default-off byte-equivalence
----------------------------

``detect_rate_mismatches`` is opt-in.  ``simulate`` does not call it
unless the user sets ``SimulatorOptions.check_rate_transitions`` to a
truthy value.  Existing single-rate diagrams therefore see no behaviour
change whatsoever.

The :class:`SampleTime` type intentionally has no ``period_ps`` /
``offset_ps`` fields yet — Phase 2 will expand the data model to use
``IntegerTime`` directly, but Phase 1 only needs the float period that
``PeriodicEventData`` already exposes to bucket blocks.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Literal, Mapping

from ..framework.event import (
    PeriodicEventData,
    is_event_data,
)

if TYPE_CHECKING:
    from ..framework.diagram import Diagram
    from ..framework.leaf_system import LeafSystem
    from ..framework.system_base import SystemBase


__all__ = [
    "SampleTime",
    "SampleTimeKind",
    "RateMismatchWarning",
    "RateMismatchError",
    "infer_block_sample_time",
    "infer_block_priority",
    "group_blocks_by_rate",
    "detect_rate_mismatches",
    "format_rate_groups",
    "rate_summary",
    "rate_summary_dot",
    "check_connection_rate_compat",
    "assert_no_rate_mismatches",
    "compute_execution_order",
]


SampleTimeKind = Literal[
    "continuous", "discrete", "constant", "inherited", "event_driven"
]


@dataclasses.dataclass(frozen=True)
class SampleTime:
    """Inferred sample-time descriptor for a leaf block.

    Phase 1 keeps this deliberately minimal: only the ``kind`` plus an
    optional ``period`` / ``offset`` (both seconds, matching
    ``PeriodicEventData``).  Phase 2 will widen this to carry
    ``period_ps`` / ``offset_ps`` ``IntegerTime`` ticks once the
    propagation pass moves the resolution canonically into integer time.

    The five kinds match the standard block-diagram sample-time taxonomy:

    * ``continuous``    — block has continuous state but no periodic events
      (e.g. a plain :class:`Integrator`).
    * ``discrete``      — block declares one or more ``PeriodicEventData``
      with a finite, positive ``period``.
    * ``constant``      — block declares neither continuous state nor
      periodic events (e.g. a :class:`Constant` source).  Constant
      signals connect to any rate without a transition.
    * ``inherited``     — sample time is to be inherited from upstream
      (Phase 2 will resolve this; Phase 1 reports it as-is).
    * ``event_driven``  — block fires only on zero-crossing /
      asynchronous events (e.g. :class:`ZeroCrossingTriggeredSubsystem`).
      Event timing is irregular, so feeding an ``event_driven`` block
      into a periodic-rate block raises a rate-mismatch warning: the
      event may not align with the discrete grid.  Connections between
      two ``event_driven`` blocks pass silently (both irregular — the
      user is expected to manage timing), and continuous → event_driven
      passes silently (events sample continuous state on demand).
    """

    kind: SampleTimeKind
    period: float | None = None
    offset: float | None = None

    @classmethod
    def continuous(cls) -> "SampleTime":
        return cls(kind="continuous")

    @classmethod
    def constant(cls) -> "SampleTime":
        return cls(kind="constant")

    @classmethod
    def inherited(cls) -> "SampleTime":
        return cls(kind="inherited")

    @classmethod
    def event_driven(cls) -> "SampleTime":
        """T-105-followup-event-rates: irregular ZC-triggered rate.

        Distinct from ``continuous`` (which has a smooth derivative)
        and from ``discrete`` (which has a regular grid).  An
        ``event_driven`` block fires only when an upstream condition
        crosses zero — :func:`detect_rate_mismatches` flags
        ``event_driven`` → ``discrete`` connections so users notice
        that the event may not land on a sample-time tick.
        """
        return cls(kind="event_driven")

    @classmethod
    def discrete(cls, period: float, offset: float = 0.0) -> "SampleTime":
        if period is None or not math.isfinite(period) or period <= 0:
            raise ValueError(
                f"discrete sample time requires a finite positive period, got {period!r}"
            )
        return cls(kind="discrete", period=float(period), offset=float(offset))

    # --- Equivalence helpers used by mismatch detection -----------------

    def is_discrete(self) -> bool:
        return self.kind == "discrete"

    def is_event_driven(self) -> bool:
        return self.kind == "event_driven"

    def is_universal(self) -> bool:
        """``True`` for sample times that connect to anything.

        ``constant`` and ``inherited`` are treated as universal at
        connect-time: a constant source feeds any consumer, and an
        inherited sample time is exactly the case Phase 2 will resolve.
        """
        return self.kind in ("constant", "inherited")

    def matches(self, other: "SampleTime", *, period_tolerance: float = 0.0) -> bool:
        """Two sample times match (no rate transition needed) iff:

        * either side is universal (constant / inherited); or
        * both are continuous; or
        * both are discrete with identical ``period`` (or within
          ``period_tolerance`` relative drift if specified — see
          T-105-followup-period-jitter); or
        * both are ``event_driven`` (irregular on both sides — let the
          user manage); or
        * one side is ``continuous`` and the other is ``event_driven``
          (events sample continuous state on demand).

        Phase 1 deliberately compares periods only — *not* offsets.
        Within a single rate group it is normal for one block to have
        an output cache at ``offset=0`` while a downstream consumer's
        state-update fires at ``offset=dt`` (Drake's classic two-phase
        x⁻ atomicity).  Treating those as a rate mismatch would warn on
        every clock-driven UnitDelay.  Phase 2 (T-123) will refine
        this once explicit ``SampleTime`` declarations land on
        non-discrete blocks.

        Continuous-to-continuous is allowed; continuous-to-discrete and
        discrete-to-continuous are flagged because they require an
        explicit ZOH/sampler.  ``event_driven`` ↔ ``discrete`` is
        flagged because the irregular event timing may not align with
        the discrete grid.

        Args:
            other: The peer sample time to compare against.
            period_tolerance: Relative tolerance for the discrete-discrete
                comparison.  ``0.0`` (the default) preserves byte-equivalent
                strict equality; positive values allow two periods to match
                when ``|a - b| <= period_tolerance * max(|a|, |b|)``.
                See :func:`detect_rate_mismatches` for the user-facing
                entry point that plumbs the tolerance through.
        """
        if self.is_universal() or other.is_universal():
            return True
        if self.kind == "continuous" and other.kind == "continuous":
            return True
        if self.is_discrete() and other.is_discrete():
            return _periods_equal(
                self.period, other.period, tolerance=period_tolerance
            )
        # T-105-followup-event-rates: event-driven compatibility table.
        if self.is_event_driven() and other.is_event_driven():
            return True
        # Continuous ↔ event-driven passes silently: events sample
        # continuous state on demand.
        if self.is_event_driven() and other.kind == "continuous":
            return True
        if self.kind == "continuous" and other.is_event_driven():
            return True
        return False


def _periods_equal(
    a: float | None,
    b: float | None,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Compare two periods for rate-group equivalence.

    By default this is strict float equality — user-supplied periods
    flow through dataclasses unchanged, and we do not want to paper
    over genuine drift between, say, 0.01 and 0.010001.

    T-105-followup-period-jitter: when ``tolerance > 0``, the comparison
    becomes relative: two periods are equal iff
    ``|a - b| <= tolerance * max(|a|, |b|)``.  This lets users opt into
    floating-point round-off forgiveness (e.g. tolerance=0.005 accepts
    0.099 ≈ 0.101) without weakening the default strict semantics.
    The ``max(|a|, |b|)`` denominator makes the tolerance behave like
    a relative percentage and stays well-defined for negative periods
    (though those are rejected upstream).
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    fa = float(a)
    fb = float(b)
    if fa == fb:
        return True
    if tolerance <= 0.0:
        return False
    if not math.isfinite(tolerance):
        # A non-finite tolerance would otherwise let any pair match,
        # which is almost certainly a user error — be loud.
        raise ValueError(
            f"period_tolerance must be a finite non-negative float, "
            f"got {tolerance!r}"
        )
    scale = max(abs(fa), abs(fb))
    if scale == 0.0:
        return True
    return abs(fa - fb) <= tolerance * scale


# ---------------------------------------------------------------------
# Inference & grouping
# ---------------------------------------------------------------------


def _collect_periodic_event_data(system: "SystemBase") -> list[PeriodicEventData]:
    """Pull every ``PeriodicEventData`` declared on ``system``.

    We walk both the leaf's discrete-state-update events
    (``_state_update_events``) and any cache-update events attached to
    its output-port callbacks.  This deliberately avoids the
    ``periodic_events`` / ``cache_update_events`` properties on
    ``SystemBase``, which trigger ``sort_trackers`` and therefore
    require an initialised ``dependency_graph`` (only available after
    the diagram is fully built and at the top level).  Phase 1 needs
    to introspect freshly-built leaves before that pass runs, so we
    walk the underlying lists directly.

    Some event slots may be ``None`` (placeholders for
    declare-but-never-configure patterns); we skip those.
    """
    out: list[PeriodicEventData] = []

    # 1. State-update events (set by ``declare_periodic_update`` /
    #    ``configure_periodic_update``).  Skipped on Diagram instances,
    #    which hold no leaf events of their own.
    for event in getattr(system, "_state_update_events", ()) or ():
        if event is None:
            continue
        data = getattr(event, "event_data", None)
        if isinstance(data, PeriodicEventData):
            out.append(data)

    # 2. Cache-update events attached to output-port callbacks (set by
    #    ``declare_output_port(..., period=...)`` in
    #    ``LeafSystem.declare_cache``).  These never go through
    #    ``sorted_callbacks`` for Phase 1's purposes — we just iterate
    #    ``self.callbacks``.
    for cb in getattr(system, "callbacks", ()) or ():
        ev = getattr(cb, "event", None)
        if ev is None:
            continue
        data = getattr(ev, "event_data", None)
        if isinstance(data, PeriodicEventData):
            out.append(data)

    return out


def _has_zero_crossing_events(system: "SystemBase") -> bool:
    """Best-effort detector for blocks that fire on zero-crossing events.

    T-105-followup-event-rates: walks the leaf's declared ZC events to
    classify the block as ``event_driven`` (irregular timing).  The
    ``LeafSystem`` API exposes a ``has_zero_crossing_events`` property,
    but ``SystemBase`` (and ``Diagram``) may not — fall back to a raw
    attribute walk for both robustness on partially-built systems and to
    keep the helper usable on diagram leaves at any build stage.

    T-115-followup-saturate-rate-classification: prefer the new
    ``_n_behavioral_zc_events`` counter when present. It excludes
    *solver-hint* ZC events — guard-only declarations with no
    ``reset_map`` and no mode transition (e.g. :class:`Saturate` /
    :class:`DeadZone` clip-boundary events whose only purpose is to let
    the ODE integrator localise a discontinuity). Those events should
    not flip a memoryless feedthrough block into the ``event_driven``
    rate group — the block has no asynchronous trigger semantics, just
    a piecewise output, and treating it as event-driven was the source
    of the spurious ``discrete → event_driven`` rate-mismatch warning
    in the canonical ``PID → Saturate → plant`` pattern.
    """
    n_behavioral = getattr(system, "_n_behavioral_zc_events", None)
    if n_behavioral is not None:
        return n_behavioral > 0
    prop = getattr(system, "has_zero_crossing_events", None)
    if isinstance(prop, bool):
        return prop
    zce = getattr(system, "_zero_crossing_events", None)
    if zce:
        return len(zce) > 0
    return False


def infer_block_sample_time(system: "SystemBase") -> SampleTime:
    """Infer a :class:`SampleTime` for ``system`` from its declarations.

    Resolution order:

    1. Explicit ``sample_time`` attribute on the block (escape hatch
       for blocks that cannot infer their rate at connect-time).
    2. If the block sets the marker attribute
       ``_jaxonomy_rate_transition = True`` (T-123 ``RateTransition`` /
       ``Decimator``), it falls through to the period-based path so it
       still buckets into its own rate group; the mismatch walker
       special-cases the marker separately.
    3. If the block declares one or more :class:`PeriodicEventData` and
       at least one has a finite positive period, the block is
       ``discrete`` at that period (smallest period wins if there are
       several — that's the block's effective rate group).
    4. T-105-followup-event-rates: if the block exposes
       ``event_driven=True`` *or* declares one or more zero-crossing
       events (via :meth:`LeafSystem.declare_zero_crossing`), it is
       ``event_driven`` — distinct from continuous and from periodic.
       Continuous state alongside ZC events is fine: many event-driven
       blocks (``ZeroCrossingTriggeredSubsystem``) also carry a smooth
       guard, but their *output* is event-paced, so we classify by the
       irregular trigger.
    5. Else, if the block has continuous state, it is ``continuous``.
    6. Else, if the block has neither, it is ``constant``.

    Inherited sample times are not produced by Phase 1 — every existing
    block has a determinate kind.  Phase 2 will introduce blocks that
    can opt into inheritance (e.g. ``Gain`` taking the upstream rate).
    """
    # T-105 Phase 2: honor an explicit ``sample_time`` attribute set on
    # the block.  This is the port-level escape hatch for blocks that
    # cannot infer their rate from periodic events at connect-time
    # (e.g. ``UnitDelay`` declares its event in ``initialize()`` which
    # only runs after ``create_context()``).  Authors of discrete
    # blocks may set ``self.sample_time = SampleTime.discrete(period=dt)``
    # in ``__init__`` to surface their rate immediately at build-time.
    explicit = getattr(system, "sample_time", None)
    if isinstance(explicit, SampleTime):
        return explicit

    # T-123: a block flagged as a rate transition is universal — it
    # exists precisely to bridge two different sample times.  Returning
    # ``constant`` here would technically work (constant is universal in
    # ``SampleTime.matches``), but it would mis-bucket the block in
    # ``group_blocks_by_rate``.  We deliberately fall through to the
    # next-best classification (its declared periodic events) so the
    # block still appears in its own rate group, while the
    # ``detect_rate_mismatches`` walker special-cases the marker.
    finite_events = [
        ev for ev in _collect_periodic_event_data(system)
        if ev.period is not None and math.isfinite(ev.period) and ev.period > 0
    ]
    if finite_events:
        # Pick the smallest period as the block's representative rate
        # group; multi-rate-within-a-leaf is rare today and Phase 2 will
        # generalise this.  We deliberately discard the per-event
        # ``offset`` here: a leaf may legitimately declare an output
        # cache update at offset=0 alongside a state update at offset=dt
        # (Drake's two-phase x⁻ atomicity), and bucketing those two
        # rates separately would falsely report every UnitDelay as a
        # mixed-rate block.  The rate group is identified by period.
        chosen_period = min(ev.period for ev in finite_events)
        return SampleTime.discrete(period=chosen_period, offset=0.0)

    # T-105-followup-event-rates: explicit user marker or implicit
    # ZC-event detection.  Periodic discrete blocks above already won;
    # we only treat a block as ``event_driven`` if it has *no* finite
    # periodic events of its own — otherwise the periodic rate carries
    # the day and the ZC is a side effect (e.g. a hybrid block).
    if getattr(system, "event_driven", False):
        return SampleTime.event_driven()
    if _has_zero_crossing_events(system):
        return SampleTime.event_driven()

    if getattr(system, "has_continuous_state", False):
        return SampleTime.continuous()

    return SampleTime.constant()


def group_blocks_by_rate(
    diagram: "Diagram",
) -> Mapping[SampleTime, list["LeafSystem"]]:
    """Bucket every leaf in ``diagram`` by its inferred :class:`SampleTime`.

    Returns a plain dict keyed by :class:`SampleTime`; iteration order
    matches the order leaves were registered.  Useful both for
    diagnostics (``format_rate_groups``) and for users who want to
    sanity-check a multi-rate model.
    """
    groups: dict[SampleTime, list["LeafSystem"]] = defaultdict(list)
    for leaf in diagram.leaf_systems:
        groups[infer_block_sample_time(leaf)].append(leaf)
    return dict(groups)


def format_rate_groups(diagram: "Diagram") -> str:
    """Render a human-readable summary of the rate groups in ``diagram``.

    Phase 3 of T-105 will turn this into ``Diagram.print_schedule()``;
    Phase 1 ships the underlying string formatter so users can already
    see what the inference pass found.
    """
    groups = group_blocks_by_rate(diagram)
    if not groups:
        return "<no leaves>"

    def _key(st: SampleTime) -> tuple[int, float]:
        # Order: discrete (by period, smallest first), continuous,
        # event_driven (irregular but conceptually adjacent to
        # continuous in the schedule), constant, inherited.
        order = {
            "discrete": 0,
            "continuous": 1,
            "event_driven": 2,
            "constant": 3,
            "inherited": 4,
        }
        return (order[st.kind], st.period or 0.0)

    lines = []
    for st in sorted(groups, key=_key):
        leaves = groups[st]
        if st.is_discrete():
            tag = f"discrete(period={st.period}, offset={st.offset})"
        else:
            tag = st.kind
        names = ", ".join(leaf.name for leaf in leaves)
        lines.append(f"  {tag}: {names}")
    return "rate groups:\n" + "\n".join(lines)


# ---------------------------------------------------------------------
# Mismatch detection
# ---------------------------------------------------------------------


class RateMismatchWarning(UserWarning):
    """Emitted when two connected blocks have incompatible discrete rates.

    Phase 2 (T-123) will replace the warning with auto-insertion of a
    ``RateTransition`` block when ``auto_insert_rate_transitions=True``.
    Phase 1 surfaces the issue but otherwise leaves the diagram alone.
    """


class RateMismatchError(ValueError):
    """Raised by :func:`detect_rate_mismatches` when ``on_mismatch='error'``."""


@dataclasses.dataclass(frozen=True)
class RateMismatch:
    """One detected source/destination rate mismatch."""

    src_system_name: str
    src_port_index: int
    src_sample_time: SampleTime
    dst_system_name: str
    dst_port_index: int
    dst_sample_time: SampleTime

    def describe(self) -> str:
        return (
            f"'{self.src_system_name}.out[{self.src_port_index}]' "
            f"({_describe(self.src_sample_time)}) -> "
            f"'{self.dst_system_name}.in[{self.dst_port_index}]' "
            f"({_describe(self.dst_sample_time)})"
        )


def _describe(st: SampleTime) -> str:
    if st.is_discrete():
        return f"discrete @ {st.period}s (offset={st.offset})"
    return st.kind


def detect_rate_mismatches(
    diagram: "Diagram",
    *,
    on_mismatch: Literal["warn", "error", "collect"] = "warn",
    period_tolerance: float = 0.0,
) -> list[RateMismatch]:
    """Find connections between blocks of incompatible sample times.

    Walks ``diagram.connection_map`` (the leaf-level wiring); for each
    ``(input_locator, output_locator)`` pair it infers the source and
    destination ``SampleTime`` (via :func:`infer_block_sample_time` on
    the owning leaf) and records a :class:`RateMismatch` whenever
    :meth:`SampleTime.matches` is ``False``.

    Args:
        diagram: A built top-level :class:`Diagram`.
        on_mismatch: ``"warn"`` (default) emits a
            :class:`RateMismatchWarning` for each mismatch and returns
            the list; ``"error"`` raises :class:`RateMismatchError` on
            the first; ``"collect"`` returns the list silently.
        period_tolerance: Optional relative tolerance applied when
            comparing two ``discrete`` sample times
            (T-105-followup-period-jitter).  Default ``0.0`` preserves
            strict float equality and therefore byte-equivalent behaviour
            with the legacy detector.  Positive values let users absorb
            floating-point round-off in user-set sample times — e.g.
            ``period_tolerance=0.01`` matches 0.099 ≈ 0.101 (within 1%
            of 0.1).  Non-discrete sample-time compatibility (universal,
            continuous, event-driven) is unaffected.

    Returns:
        The list of detected :class:`RateMismatch` entries.  Empty when
        the diagram is single-rate or only mixes universal sources.
    """
    if period_tolerance < 0.0 or not math.isfinite(period_tolerance):
        raise ValueError(
            f"period_tolerance must be a finite non-negative float, "
            f"got {period_tolerance!r}"
        )

    mismatches: list[RateMismatch] = []

    # Cache inferred sample times by system_id to avoid recomputing on
    # densely-connected fan-out hubs.
    cache: dict[object, SampleTime] = {}

    def _st_for(system: "SystemBase") -> SampleTime:
        sid = system.system_id
        if sid not in cache:
            cache[sid] = infer_block_sample_time(system)
        return cache[sid]

    for input_locator, output_locator in diagram.connection_map.items():
        dst_sys, dst_idx = input_locator
        src_sys, src_idx = output_locator

        # T-123: explicit ``RateTransition`` / ``Decimator`` blocks
        # bridge any two adjacent rates by construction.  Skip the
        # mismatch check on either end of such a connection so the
        # canonical ``Slow → RateTransition → Fast`` pattern stays
        # silent under ``detect_rate_mismatches``.
        if getattr(src_sys, "_jaxonomy_rate_transition", False):
            continue
        if getattr(dst_sys, "_jaxonomy_rate_transition", False):
            continue

        src_st = _st_for(src_sys)
        dst_st = _st_for(dst_sys)

        if src_st.matches(dst_st, period_tolerance=period_tolerance):
            continue

        mismatch = RateMismatch(
            src_system_name=src_sys.name,
            src_port_index=src_idx,
            src_sample_time=src_st,
            dst_system_name=dst_sys.name,
            dst_port_index=dst_idx,
            dst_sample_time=dst_st,
        )
        mismatches.append(mismatch)

        if on_mismatch == "error":
            raise RateMismatchError(
                "Rate mismatch (no RateTransition block inserted): "
                + mismatch.describe()
            )

    if on_mismatch == "warn" and mismatches:
        for mm in mismatches:
            warnings.warn(_format_mismatch_message(mm), RateMismatchWarning,
                          stacklevel=2)

    return mismatches


def _format_mismatch_message(mm: "RateMismatch") -> str:
    """Pick a user-facing message tailored to the mismatch flavour.

    T-105-followup-event-rates: ``event_driven`` connections deserve a
    distinct hint — there is no ``RateTransition`` block that bridges an
    irregular trigger onto a periodic grid, so the canonical
    "auto-insert RateTransition" advice would mislead.  We surface a
    specialised "event timing may not align" message in that case.
    """
    src_ev = mm.src_sample_time.is_event_driven()
    dst_ev = mm.dst_sample_time.is_event_driven()
    if src_ev or dst_ev:
        return (
            "T-105 event-driven rate mismatch: event timing may not "
            "align with the discrete grid (insert a Sample-and-Hold or "
            "rebuild the downstream block to be event-driven too): "
            + mm.describe()
        )
    return (
        "T-105 rate mismatch (Phase 2 will auto-insert a "
        "RateTransition; for now connect blocks of equal sample "
        "time or insert a transition manually): " + mm.describe()
    )


def check_connection_rate_compat(
    src_system: "SystemBase",
    src_port_index: int,
    dst_system: "SystemBase",
    dst_port_index: int,
    *,
    on_mismatch: Literal["warn", "error", "collect"] = "warn",
) -> RateMismatch | None:
    """Connect-time rate-mismatch check for a single source/dest pair.

    T-105 Phase 2: companion to :func:`detect_rate_mismatches` that runs
    on a *single* connection rather than walking the whole built diagram.
    Used by :meth:`DiagramBuilder.connect` when the builder was created
    with ``validate_rates_at_connect=True``.

    Honors the same ``_jaxonomy_rate_transition`` marker that
    :func:`detect_rate_mismatches` uses (T-123 ``RateTransition`` blocks
    bridge any two adjacent rates by construction), and the same
    universal-sample-time rule (constant / inherited connect to
    anything).

    Returns the constructed :class:`RateMismatch` when the rates are
    incompatible (or ``None`` otherwise).  Side-effects depend on
    ``on_mismatch``:

    * ``"warn"``: emits a :class:`RateMismatchWarning` for the mismatch.
    * ``"error"``: raises :class:`RateMismatchError`.
    * ``"collect"``: returns the mismatch without warning or raising.

    Phase 1 fallback: if either side cannot be classified (e.g. a
    ``UnitDelay`` whose periodic update is only configured in
    ``initialize()``), this function still returns ``None`` because
    pre-``initialize`` ``UnitDelay`` infers as ``constant`` (universal),
    which matches anything.  Authors who want immediate connect-time
    enforcement should set ``self.sample_time`` explicitly in
    ``__init__``.
    """
    if getattr(src_system, "_jaxonomy_rate_transition", False):
        return None
    if getattr(dst_system, "_jaxonomy_rate_transition", False):
        return None

    src_st = infer_block_sample_time(src_system)
    dst_st = infer_block_sample_time(dst_system)

    if src_st.matches(dst_st):
        return None

    mismatch = RateMismatch(
        src_system_name=src_system.name,
        src_port_index=src_port_index,
        src_sample_time=src_st,
        dst_system_name=dst_system.name,
        dst_port_index=dst_port_index,
        dst_sample_time=dst_st,
    )

    if on_mismatch == "error":
        raise RateMismatchError(
            "Rate mismatch at connect time (no RateTransition block "
            "between blocks): " + mismatch.describe()
        )
    if on_mismatch == "warn":
        warnings.warn(
            "T-105 connect-time rate mismatch (insert a RateTransition "
            "block between these ports, or wire a same-rate block in "
            "between): " + mismatch.describe(),
            RateMismatchWarning,
            stacklevel=3,
        )

    return mismatch


def assert_no_rate_mismatches(
    diagram: "Diagram",
    *,
    on_mismatch: Literal["warn", "error"] = "error",
) -> list[RateMismatch]:
    """Post-build helper that walks the whole diagram for rate mismatches.

    Thin wrapper around :func:`detect_rate_mismatches` that defaults to
    ``on_mismatch="error"`` — the natural verb for an ``assert_*`` API.

    Useful when the connect-time hook is not enabled (the builder was
    not constructed with ``validate_rates_at_connect=True``) but the
    user wants to validate the whole diagram once it's built.
    """
    return detect_rate_mismatches(diagram, on_mismatch=on_mismatch)


def iter_rate_groups(
    diagram: "Diagram",
) -> Iterable[tuple[SampleTime, list["LeafSystem"]]]:
    """Convenience iterator over rate groups in deterministic order.

    Mirrors :func:`format_rate_groups`'s ordering: discrete (by period,
    smallest first), then continuous, constant, inherited.
    """
    groups = group_blocks_by_rate(diagram)

    def _key(st: SampleTime) -> tuple[int, float]:
        order = {
            "discrete": 0,
            "continuous": 1,
            "event_driven": 2,
            "constant": 3,
            "inherited": 4,
        }
        return (order[st.kind], st.period or 0.0)

    for st in sorted(groups, key=_key):
        yield st, groups[st]


# ---------------------------------------------------------------------
# T-105-followup-tasking-priority — deterministic per-rate task order
# ---------------------------------------------------------------------


def infer_block_priority(system: "SystemBase") -> int | None:
    """Return the user-declared task priority for ``system``, or ``None``.

    Block authors (and end-users post-construction) can assign a
    deterministic execution priority to a block by setting an integer
    ``priority`` attribute on the leaf::

        gain = Gain(2.0)
        gain.priority = 10

    Within a same-rate group, blocks with lower ``priority`` values are
    scheduled to run before blocks with higher values, as a tiebreaker
    that only fires when no explicit topological dependency orders the
    pair.  ``None`` (the default) means *no preference* — fall back to
    the natural ordering, which is alphabetic-by-name for full
    determinism.

    Phase 1 (this followup) is read-only: the helper exists so
    :func:`compute_execution_order` can surface the schedule users
    *would* see if the discrete scheduler honoured priorities.  The
    actual scheduler integration (hooking into
    ``SystemBase.handle_discrete_update``'s Phase-2 topological branch)
    is filed as a deeper followup; default ``priority=None`` preserves
    byte-equivalence with the legacy ordering.
    """
    raw = getattr(system, "priority", None)
    if raw is None:
        return None
    # Be strict: reject non-integers to avoid silently sorting on
    # truthiness or comparing ints to floats with surprising ties.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(
            f"block {system.name!r} has non-integer priority={raw!r}; "
            "expected int or None"
        )
    return raw


def _build_leaf_dependency_graph(
    diagram: "Diagram",
) -> dict[object, set[object]]:
    """Best-effort topological-dependency graph for ALL leaves.

    Unlike :func:`build_discrete_dependency_graph` (which only tracks
    discrete-state dependencies and skips continuous / passthrough
    leaves), this helper walks every leaf's connected input ports and
    records the upstream-leaf system_id as a dependency.  That is
    sufficient for the tasking-priority tiebreaker: an explicit
    input-port wire from B → A always forces A to run after B, even
    when A's user-declared priority is lower than B's.
    """
    leaf_ids = {leaf.system_id for leaf in diagram.leaf_systems}
    graph: dict[object, set[object]] = {
        leaf.system_id: set() for leaf in diagram.leaf_systems
    }
    cmap = getattr(diagram, "connection_map", None) or {}
    for in_locator, out_locator in cmap.items():
        dst_sys, _ = in_locator
        src_sys, _ = out_locator
        if dst_sys.system_id not in leaf_ids:
            continue
        if src_sys.system_id not in leaf_ids:
            continue
        if src_sys.system_id == dst_sys.system_id:
            continue
        graph[dst_sys.system_id].add(src_sys.system_id)
    return graph


def compute_execution_order(diagram: "Diagram") -> list["LeafSystem"]:
    """Return leaves in the order the scheduler *would* process them.

    Read-only inspection helper for T-105-followup-tasking-priority.
    The order is built by Kahn's algorithm over the leaf-dependency
    graph (every connected wire B → A imposes "B before A"), with a
    deterministic tiebreaker among the currently-ready set:

      1. **Rate-group key.**  Discrete leaves first (smallest period
         wins), then continuous, then constant, then inherited.  This
         matches the existing :func:`iter_rate_groups` ordering and
         the conventional base-rate-before-sub-rate task resolution.
      2. **Explicit priority** (lower runs first).  Leaves with
         ``priority=None`` are pushed *after* leaves with any integer
         priority (so an explicit 0 still beats an unset default —
         users opt into ordering globally per same-rate group).
      3. **Alphabetic by name.**  Last-resort tiebreaker for full
         determinism, matching the convention already used by
         :func:`find_cycles` / :func:`topological_sort` on
         discrete-update graphs.

    Topological edges always win.  If B's output feeds A's input,
    A appears after B in the returned list regardless of priorities.

    The function is purely advisory: it does not mutate the diagram
    and the simulator does not (yet) consult it.  Future work will
    plumb this ordering into ``handle_discrete_update``'s Phase 2 so
    that same-rate state updates honour the user's priority.

    Raises:
        DependencyCycleError: if the leaf-connection graph contains a
            cycle (an algebraic loop in the input-port wiring).  The
            existing :func:`detect_rate_mismatches` /
            :func:`assert_no_rate_mismatches` pipeline does not
            normally flag such cycles, so this is the first place a
            user would see them.
    """
    leaves_by_id: dict[object, "LeafSystem"] = {
        leaf.system_id: leaf for leaf in diagram.leaf_systems
    }
    if not leaves_by_id:
        return []

    graph = _build_leaf_dependency_graph(diagram)

    # Reverse adjacency for Kahn.
    reverse: dict[object, set[object]] = {sid: set() for sid in leaves_by_id}
    for sid, deps in graph.items():
        for d in deps:
            reverse.setdefault(d, set()).add(sid)

    in_count = {sid: len(graph.get(sid, ())) for sid in leaves_by_id}

    # Pre-cache rate-group + priority keys so we only resolve each leaf
    # once even when it bounces in and out of the ready set.
    rate_key: dict[object, tuple[int, float]] = {}
    prio_key: dict[object, tuple[int, int]] = {}
    rate_order = {
        "discrete": 0,
        "continuous": 1,
        "event_driven": 2,
        "constant": 3,
        "inherited": 4,
    }
    for sid, leaf in leaves_by_id.items():
        st = infer_block_sample_time(leaf)
        rate_key[sid] = (rate_order[st.kind], st.period or 0.0)
        p = infer_block_priority(leaf)
        # Two-tuple: (0, value) for explicit priorities, (1, 0) for
        # ``None`` so unset blocks sort *after* explicit ones.
        prio_key[sid] = (0, p) if p is not None else (1, 0)

    def _sort_key(sid: object) -> tuple:
        return (rate_key[sid], prio_key[sid], leaves_by_id[sid].name)

    ready = sorted(
        [sid for sid, c in in_count.items() if c == 0], key=_sort_key
    )
    out: list["LeafSystem"] = []
    while ready:
        sid = ready.pop(0)
        out.append(leaves_by_id[sid])
        for nxt in reverse.get(sid, ()):
            in_count[nxt] -= 1
            if in_count[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=_sort_key)

    if len(out) != len(leaves_by_id):
        # Lazy import to keep module-level imports cycle-free.
        from ..framework.discrete_dependencies import DependencyCycleError
        missing = [
            leaves_by_id[sid].name
            for sid in leaves_by_id
            if leaves_by_id[sid] not in out
        ]
        raise DependencyCycleError(
            "compute_execution_order: leaf-connection graph has a cycle; "
            f"unsortable leaves: {sorted(missing)}"
        )
    return out


# ---------------------------------------------------------------------
# T-105-followup-rate-summary — richer diagnostic dump
# ---------------------------------------------------------------------


_RATE_SUMMARY_FORMATS = ("text", "json", "markdown")


def _rate_group_tag(st: "SampleTime") -> str:
    """Compact human label for a rate group, matching ``format_rate_groups``."""
    if st.is_discrete():
        return f"discrete(period={st.period}, offset={st.offset})"
    return st.kind


def _collect_rate_summary_payload(diagram: "Diagram") -> dict:
    """Gather the structured payload used by every ``rate_summary`` format.

    Returns a plain ``dict`` shaped for JSON serialisation:

    * ``rate_groups``: list of ``{kind, period, offset, count, blocks}``
      in the canonical ``iter_rate_groups`` order.
    * ``mismatches``: list of ``{src, src_port, src_sample_time, ...}``
      collected with ``on_mismatch="collect"`` so callers never see a
      warning or exception from a diagnostic helper.
    * ``execution_order``: list of leaf names from
      :func:`compute_execution_order`, or ``None`` if the graph has a
      cycle (we degrade gracefully — a cycle should not poison the rest
      of the report).
    """
    groups_payload: list[dict] = []
    for st, leaves in iter_rate_groups(diagram):
        groups_payload.append(
            {
                "kind": st.kind,
                "period": st.period,
                "offset": st.offset,
                "count": len(leaves),
                "blocks": [leaf.name for leaf in leaves],
            }
        )

    # Collect (never warn / raise) — this is a diagnostic helper, the
    # caller decides whether to escalate.
    mismatches = detect_rate_mismatches(diagram, on_mismatch="collect")
    mismatch_payload = [
        {
            "src": mm.src_system_name,
            "src_port": mm.src_port_index,
            "src_sample_time": _describe(mm.src_sample_time),
            "dst": mm.dst_system_name,
            "dst_port": mm.dst_port_index,
            "dst_sample_time": _describe(mm.dst_sample_time),
        }
        for mm in mismatches
    ]

    # T-105-followup-print-schedule-feedback-cycle: when execution-
    # order computation hits a cycle, distinguish a *real* algebraic
    # loop (which the simulator will reject as ``AlgebraicLoopError``)
    # from a feedback path closed through discrete state (which the
    # simulator runs fine because the discrete sample-and-hold breaks
    # the loop). Report the cycle-flavour in a new ``cycle_kind`` slot
    # consumed by the text / markdown / json renderers.
    order = None
    cycle_kind = None
    cycle_blocks: list[str] | None = None
    try:
        order = [leaf.name for leaf in compute_execution_order(diagram)]
    except Exception:
        # T-105-followup-print-schedule-feedback-cycle: classify the
        # cycle. First make sure every leaf has had its
        # ``initialize()`` hook run so blocks whose periodic events
        # are registered there (PIDDiscrete, Decimator, UnitDelay,
        # ZeroOrderHold) are visible to the periodic-event probe. The
        # call is wrapped in try/except so an unbuildable diagram
        # falls back to the pre-init check; consumers that already
        # call ``print_schedule`` get a no-op on the second init since
        # the first was already done at the print_schedule entry point.
        try:
            diagram.create_context()
        except Exception:  # noqa: BLE001
            pass

        # Do any of the leaves carry a sample-and-hold output (i.e. a
        # periodic event with a finite positive period)? If yes, the
        # cycle is closed through discrete state and the simulator
        # runs fine. If no, the cycle is purely algebraic.
        discrete_leaves = [
            leaf for leaf in diagram.leaf_systems
            if any(
                ev.period is not None
                and math.isfinite(ev.period)
                and ev.period > 0
                for ev in _collect_periodic_event_data(leaf)
            )
        ]
        if discrete_leaves:
            cycle_kind = "feedback-through-discrete"
            cycle_blocks = sorted(leaf.name for leaf in discrete_leaves)
        else:
            cycle_kind = "algebraic"
            cycle_blocks = None

    return {
        "rate_groups": groups_payload,
        "mismatches": mismatch_payload,
        "execution_order": order,
        "cycle_kind": cycle_kind,
        "cycle_discrete_blocks": cycle_blocks,
    }


def _render_rate_summary_text(payload: dict) -> str:
    """Plain-text rendering — the default format."""
    lines: list[str] = ["rate summary:"]

    lines.append("  rate groups:")
    if not payload["rate_groups"]:
        lines.append("    <no leaves>")
    else:
        for grp in payload["rate_groups"]:
            if grp["kind"] == "discrete":
                tag = f"discrete(period={grp['period']}, offset={grp['offset']})"
            else:
                tag = grp["kind"]
            names = ", ".join(grp["blocks"])
            lines.append(f"    {tag} [count={grp['count']}]: {names}")

    lines.append("  mismatches:")
    if not payload["mismatches"]:
        lines.append("    <none>")
    else:
        for mm in payload["mismatches"]:
            lines.append(
                f"    {mm['src']}.out[{mm['src_port']}] "
                f"({mm['src_sample_time']}) -> "
                f"{mm['dst']}.in[{mm['dst_port']}] "
                f"({mm['dst_sample_time']})"
            )

    lines.append("  execution order:")
    order = payload["execution_order"]
    if order is None:
        # T-105-followup-print-schedule-feedback-cycle: render a
        # specific message for feedback-through-discrete vs algebraic.
        ck = payload.get("cycle_kind")
        cb = payload.get("cycle_discrete_blocks") or []
        if ck == "feedback-through-discrete":
            blk_list = ", ".join(cb) if cb else "<none>"
            lines.append(
                f"    <feedback cycle through discrete state in "
                f"[{blk_list}] — the simulator runs this fine because "
                "the sample-and-hold output breaks the loop; "
                "topological execution order is not well-defined>"
            )
        elif ck == "algebraic":
            lines.append(
                "    <algebraic cycle detected — the simulator will "
                "raise AlgebraicLoopError; insert a UnitDelay or other "
                "discrete-state block to break the loop>"
            )
        else:
            lines.append("    <cycle detected>")
    elif not order:
        lines.append("    <no leaves>")
    else:
        lines.append("    " + " -> ".join(order))

    return "\n".join(lines)


def _render_rate_summary_markdown(payload: dict) -> str:
    """Markdown rendering — suitable for generated docs / PR bodies."""
    lines: list[str] = ["## Rate Summary", ""]

    lines.append("## Rate Groups")
    lines.append("")
    if not payload["rate_groups"]:
        lines.append("_no leaves_")
        lines.append("")
    else:
        for grp in payload["rate_groups"]:
            if grp["kind"] == "discrete":
                tag = (
                    f"**discrete** (period={grp['period']}, "
                    f"offset={grp['offset']})"
                )
            else:
                tag = f"**{grp['kind']}**"
            lines.append(f"- {tag} — count={grp['count']}")
            for name in grp["blocks"]:
                lines.append(f"  - `{name}`")
        lines.append("")

    lines.append("## Mismatches")
    lines.append("")
    if not payload["mismatches"]:
        lines.append("_none_")
        lines.append("")
    else:
        for mm in payload["mismatches"]:
            lines.append(
                f"- `{mm['src']}.out[{mm['src_port']}]` "
                f"({mm['src_sample_time']}) → "
                f"`{mm['dst']}.in[{mm['dst_port']}]` "
                f"({mm['dst_sample_time']})"
            )
        lines.append("")

    lines.append("## Execution Order")
    lines.append("")
    order = payload["execution_order"]
    if order is None:
        # T-105-followup-print-schedule-feedback-cycle.
        ck = payload.get("cycle_kind")
        cb = payload.get("cycle_discrete_blocks") or []
        if ck == "feedback-through-discrete":
            blk_list = ", ".join(f"`{b}`" for b in cb) if cb else "_none_"
            lines.append(
                f"_feedback cycle through discrete state in "
                f"{blk_list} — the simulator runs this fine because "
                "the sample-and-hold output breaks the loop; "
                "topological execution order is not well-defined._"
            )
        elif ck == "algebraic":
            lines.append(
                "_algebraic cycle detected — the simulator will raise "
                "`AlgebraicLoopError`; insert a `UnitDelay` or other "
                "discrete-state block to break the loop._"
            )
        else:
            lines.append("_cycle detected_")
    elif not order:
        lines.append("_no leaves_")
    else:
        for idx, name in enumerate(order, start=1):
            lines.append(f"{idx}. `{name}`")

    return "\n".join(lines)


def _render_rate_summary_json(payload: dict) -> str:
    """JSON rendering — round-trippable via ``json.loads``."""
    import json
    return json.dumps(payload, indent=2, sort_keys=False)


def rate_summary(
    diagram: "Diagram",
    *,
    format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """Render a richer diagnostic report of a diagram's rate structure.

    Companion to :func:`format_rate_groups` (which stays byte-identical
    for backwards compatibility) that surfaces rate groups *plus*
    detected mismatches *plus* the deterministic execution order in one
    string.  Designed for:

    * inclusion in provenance manifests / generated docs;
    * dropping into debugging output when a multirate model misbehaves;
    * embedding in PR bodies (via the ``"markdown"`` format).

    Args:
        diagram: A built top-level :class:`Diagram`.
        format: ``"text"`` (default), ``"json"``, or ``"markdown"``.

    Returns:
        A string in the requested format.  The JSON variant is
        round-trippable via :func:`json.loads`; the markdown variant is
        valid CommonMark with ``## `` section headers; the text variant
        is the legacy human-readable shape (an extension of
        :func:`format_rate_groups`).

    Raises:
        ValueError: if ``format`` is not one of the supported values.
    """
    if format not in _RATE_SUMMARY_FORMATS:
        raise ValueError(
            f"rate_summary: unsupported format={format!r}; "
            f"expected one of {_RATE_SUMMARY_FORMATS}"
        )

    payload = _collect_rate_summary_payload(diagram)

    if format == "text":
        return _render_rate_summary_text(payload)
    if format == "markdown":
        return _render_rate_summary_markdown(payload)
    # format == "json"
    return _render_rate_summary_json(payload)


# ---------------------------------------------------------------------
# T-105-followup-rate-summary-graphviz — DOT-format visualization
# ---------------------------------------------------------------------


# Deterministic palette for rate-group cluster fills.  Hand-picked so
# adjacent rate groups stay visually distinct even on greyscale prints.
# Ordering matters: discrete rates exhaust this list in iter_rate_groups
# order; continuous / event_driven / constant / inherited get reserved
# slots at the end so they're predictable across diagrams.
_RATE_GROUP_PALETTE = (
    "#a6cee3",  # light blue
    "#b2df8a",  # light green
    "#fb9a99",  # light red
    "#fdbf6f",  # light orange
    "#cab2d6",  # light purple
    "#ffff99",  # light yellow
    "#1f78b4",  # blue
    "#33a02c",  # green
    "#e31a1c",  # red
    "#ff7f00",  # orange
)
_NON_DISCRETE_FILL = {
    "continuous": "#dddddd",   # neutral grey — always-on
    "event_driven": "#ffd6e7",  # pink — irregular
    "constant": "#f0f0f0",     # near-white — universal
    "inherited": "#ffffff",    # white — to-be-resolved
}
_MISMATCH_EDGE_COLOR = "red"
_COMPATIBLE_EDGE_COLOR = "black"
_RATE_TRANSITION_EDGE_COLOR = "blue"
_RATE_TRANSITION_NODE_FILL = "#ffe680"  # warm yellow — bridges
_RATE_TRANSITION_NODE_SHAPE = "diamond"


def _dot_escape(text: str) -> str:
    """Quote and escape ``text`` for safe inclusion in a DOT string literal."""
    # Per the DOT grammar, only ``"`` and ``\`` need escaping inside a
    # double-quoted ID; we are conservative and also collapse newlines
    # which would otherwise interrupt the ``label="..."`` form.
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "")
    return f'"{escaped}"'


def _dot_node_id(name: str) -> str:
    """Quoted DOT identifier for a leaf, used as both node ID and label seed."""
    return _dot_escape(name)


def _rate_group_cluster_id(idx: int) -> str:
    """Cluster name. The ``cluster_`` prefix is what tells Graphviz to
    draw the surrounding box — bare ``subgraph`` blocks are layout-only.
    """
    return f"cluster_rate_{idx}"


def _rate_group_label(st: "SampleTime") -> str:
    """Human-readable label for a rate-group cluster header."""
    if st.is_discrete():
        return f"discrete\\nperiod={st.period}\\noffset={st.offset}"
    return st.kind


def rate_summary_dot(diagram: "Diagram") -> str:
    """Emit a Graphviz DOT-format visualization of the rate-group structure.

    T-105-followup-rate-summary-graphviz: companion to
    :func:`rate_summary` that turns a diagram's rate-group structure into
    a DOT graph users can render with Graphviz (``dot -Tpng``,
    ``dot -Tsvg``, the ``graphviz`` Python package, or any of the online
    DOT viewers).  The output is a single string — no Graphviz Python
    dependency is required to *produce* it.

    The resulting graph encodes:

    * **One subgraph cluster per rate group**, labelled with the rate
      kind (and ``period``/``offset`` for discrete rates).
    * **One node per leaf**, placed in its rate-group cluster and
      coloured from a deterministic palette so same-rate blocks share
      a fill colour.  Blocks tagged with the
      ``_jaxonomy_rate_transition`` marker (``ZeroOrderHold`` instances
      from :func:`RateTransition`, :class:`Decimator`) get a distinct
      diamond shape and warm-yellow fill so they read as bridges.
    * **One directed edge per connection** in
      ``diagram.connection_map``.  Edges that bridge two incompatible
      sample times (i.e. would be flagged by
      :func:`detect_rate_mismatches`) are coloured red and dashed; edges
      that originate from or terminate at a ``RateTransition`` block are
      coloured blue (these are the *intended* bridges); all other edges
      are the default black.

    The function is purely formatting — it never inserts blocks, never
    mutates the diagram, and never raises a
    :class:`RateMismatchWarning` (mismatches are collected silently).
    The returned string is suitable for piping straight to ``dot``::

        with open("rates.dot", "w") as f:
            f.write(rate_summary_dot(diag))
        # $ dot -Tpng rates.dot -o rates.png

    Or for use with the optional ``graphviz`` Python package::

        import graphviz
        graphviz.Source(rate_summary_dot(diag)).render("rates", view=True)

    Default-float64 policy (T-005) is preserved: this helper only emits
    text and never touches JAX state.

    Args:
        diagram: A built top-level :class:`Diagram`.

    Returns:
        A single ``str`` containing a complete DOT graph beginning with
        ``digraph Jaxonomy_RateGroups {`` and ending with ``}``.  Empty
        diagrams still emit a valid (but body-less) ``digraph`` block.
    """
    leaves_by_id: dict[object, "LeafSystem"] = {
        leaf.system_id: leaf for leaf in diagram.leaf_systems
    }

    # --- Group leaves by rate so we can emit one cluster per group. ---
    # Reuse the canonical iter_rate_groups ordering so the palette
    # assignment is deterministic across runs and across diagrams that
    # share a rate set.
    rate_groups_ordered: list[tuple["SampleTime", list["LeafSystem"]]] = list(
        iter_rate_groups(diagram)
    )

    # Map each leaf system_id to (cluster_idx, fill_colour) for edge
    # rendering and node emission.
    sid_to_group_idx: dict[object, int] = {}
    sid_to_fill: dict[object, str] = {}
    discrete_palette_cursor = 0
    for idx, (st, leaves) in enumerate(rate_groups_ordered):
        if st.is_discrete():
            fill = _RATE_GROUP_PALETTE[
                discrete_palette_cursor % len(_RATE_GROUP_PALETTE)
            ]
            discrete_palette_cursor += 1
        else:
            fill = _NON_DISCRETE_FILL.get(st.kind, "#ffffff")
        for leaf in leaves:
            sid_to_group_idx[leaf.system_id] = idx
            sid_to_fill[leaf.system_id] = fill

    # --- Collect mismatches once so edge rendering is O(edges). -------
    # Use ``"collect"`` so users never get a RateMismatchWarning fired
    # by what is supposed to be a passive visualizer.
    mismatches = detect_rate_mismatches(diagram, on_mismatch="collect")
    mismatch_pairs: set[tuple[object, int, object, int]] = {
        (
            _system_id_by_name(leaves_by_id, mm.src_system_name),
            mm.src_port_index,
            _system_id_by_name(leaves_by_id, mm.dst_system_name),
            mm.dst_port_index,
        )
        for mm in mismatches
    }

    # --- Emit the DOT body. -------------------------------------------
    lines: list[str] = []
    lines.append("digraph Jaxonomy_RateGroups {")
    lines.append("  rankdir=LR;")
    lines.append("  compound=true;")  # lets cluster-to-cluster edges work
    lines.append('  node [style="filled", shape="box", fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=10];')

    # One subgraph cluster per rate group.
    for idx, (st, leaves) in enumerate(rate_groups_ordered):
        cluster_id = _rate_group_cluster_id(idx)
        label = _rate_group_label(st)
        lines.append(f"  subgraph {cluster_id} {{")
        lines.append(f"    label={_dot_escape(label)};")
        lines.append('    style="rounded,filled";')
        # Cluster background: lighten the node fill so nodes still pop.
        lines.append('    color="#888888";')
        lines.append('    fillcolor="#fafafa";')
        for leaf in leaves:
            node_id = _dot_node_id(leaf.name)
            fill = sid_to_fill[leaf.system_id]
            is_bridge = bool(
                getattr(leaf, "_jaxonomy_rate_transition", False)
            )
            if is_bridge:
                shape = _RATE_TRANSITION_NODE_SHAPE
                node_fill = _RATE_TRANSITION_NODE_FILL
            else:
                shape = "box"
                node_fill = fill
            lines.append(
                f"    {node_id} [label={_dot_escape(leaf.name)}, "
                f'fillcolor="{node_fill}", shape="{shape}"];'
            )
        lines.append("  }")

    # One directed edge per connection in the diagram.  We sort by
    # ``(src_name, src_port, dst_name, dst_port)`` so two byte-identical
    # diagrams produce byte-identical DOT output.
    cmap = getattr(diagram, "connection_map", None) or {}
    edge_records: list[tuple[str, int, str, int, object, int, object, int]] = []
    for in_locator, out_locator in cmap.items():
        dst_sys, dst_idx = in_locator
        src_sys, src_idx = out_locator
        # Only render leaf-to-leaf edges; the visualizer would otherwise
        # try to draw arrows to/from a non-rendered diagram-level port.
        if src_sys.system_id not in leaves_by_id:
            continue
        if dst_sys.system_id not in leaves_by_id:
            continue
        edge_records.append(
            (
                src_sys.name,
                src_idx,
                dst_sys.name,
                dst_idx,
                src_sys.system_id,
                src_idx,
                dst_sys.system_id,
                dst_idx,
            )
        )
    edge_records.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    for (src_name, src_idx, dst_name, dst_idx,
         src_sid, _src_pidx, dst_sid, _dst_pidx) in edge_records:
        src_id = _dot_node_id(src_name)
        dst_id = _dot_node_id(dst_name)
        is_mismatch = (src_sid, src_idx, dst_sid, dst_idx) in mismatch_pairs
        # Bridge classification: either endpoint was tagged as a
        # rate-transition block.  Mismatch wins styling-wise (red,
        # dashed) so genuine errors stay loud, but a "blue bridge edge"
        # is still useful when the user has correctly inserted a
        # transition block — readers see that it *is* the bridge.
        src_leaf = leaves_by_id[src_sid]
        dst_leaf = leaves_by_id[dst_sid]
        is_bridge_edge = (
            getattr(src_leaf, "_jaxonomy_rate_transition", False)
            or getattr(dst_leaf, "_jaxonomy_rate_transition", False)
        )
        if is_mismatch:
            color = _MISMATCH_EDGE_COLOR
            style = "dashed"
            label = "rate mismatch"
        elif is_bridge_edge:
            color = _RATE_TRANSITION_EDGE_COLOR
            style = "solid"
            label = "rate transition"
        else:
            color = _COMPATIBLE_EDGE_COLOR
            style = "solid"
            label = ""
        attrs = [f'color="{color}"', f'style="{style}"']
        if label:
            attrs.append(f"label={_dot_escape(label)}")
            attrs.append(f'fontcolor="{color}"')
        lines.append(
            f"  {src_id} -> {dst_id} [{', '.join(attrs)}];"
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _system_id_by_name(
    leaves_by_id: dict[object, "LeafSystem"], name: str
) -> object:
    """Resolve a leaf name back to its system_id.

    :class:`RateMismatch` records carry block names (strings) rather than
    system_ids, so we need a small index when correlating mismatches
    against the leaf table during DOT emission.  Names are unique within
    a built diagram (the builder rejects duplicates), so the first hit
    wins.  Returns ``None`` if no leaf matches — the caller treats that
    pair as "not a mismatch" rather than crashing the visualizer.
    """
    for sid, leaf in leaves_by_id.items():
        if leaf.name == name:
            return sid
    return None
