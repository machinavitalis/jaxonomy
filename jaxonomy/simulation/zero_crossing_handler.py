# SPDX-License-Identifier: MIT

"""Zero-crossing event handling, decoupled from ODE integration.

This module extracts guard-related logic from the monolithic Simulator class,
providing a clean interface for:
  - Evaluating guard functions at interval boundaries
  - Checking if guards have triggered
  - Localizing zero-crossings via bisection
  - Applying reset maps after zero-crossing events
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from .types import GuardIsolationData
from ..framework import IntegerTime
from jaxonomy import backend

if TYPE_CHECKING:
    from ..framework import SystemBase, ContextBase
    from ..framework.event import EventCollection


class ZeroCrossingHandler:
    """Manages zero-crossing event detection, localization, and handling.

    This class encapsulates the guard evaluation and bisection logic that was
    previously embedded inline in the Simulator. It is a stateless service
    that the Simulator delegates to, preserving compatibility with JAX tracing
    and custom VJP rules.

    Args:
        system: The system being simulated.
        zc_bisection_loop_count: Number of bisection iterations for localizing
            zero-crossing events.
    """

    def __init__(
        self,
        system: "SystemBase",
        zc_bisection_loop_count: int,
        lower_triangular_discrete_update: bool = False,
    ):
        self.system = system
        self.zc_bisection_loop_count = zc_bisection_loop_count
        self.lower_triangular_discrete_update = lower_triangular_discrete_update

    def evaluate_guards(self, context: "ContextBase") -> "EventCollection":
        """Determine which zero-crossing guards are active for the current state.

        Delegates to ``system.determine_active_guards``.

        Args:
            context: The current simulation context (should have refreshed port cache).

        Returns:
            EventCollection with guard activity flags set.
        """
        return self.system.determine_active_guards(context)

    def record_interval_start(
        self, events: "EventCollection", context: "ContextBase"
    ) -> "EventCollection":
        """Record guard function values at the start of an integration interval.

        Args:
            events: The current zero-crossing event collection.
            context: The context at the start of the interval.

        Returns:
            Updated EventCollection with ``w0`` values recorded.
        """
        return guard_interval_start(events, context)

    def record_interval_end(
        self, events: "EventCollection", context: "ContextBase"
    ) -> "EventCollection":
        """Record guard function values at the end of an integration interval.

        Args:
            events: The current zero-crossing event collection.
            context: The context at the end of the interval.

        Returns:
            Updated EventCollection with ``w1`` values recorded.
        """
        return guard_interval_end(events, context)

    def check_triggered(
        self, events: "EventCollection", context: "ContextBase"
    ) -> "EventCollection":
        """Determine which guards have triggered by comparing start/end values.

        Records the interval-end guard values and then checks the sign change
        against the direction rule for each event.

        Args:
            events: EventCollection with ``w0`` already recorded.
            context: The context at the end of the interval.

        Returns:
            Updated EventCollection with ``triggered`` flags set.
        """
        return determine_triggered_guards(events, context)

    def handle_events(
        self, events: "EventCollection", context: "ContextBase"
    ) -> "ContextBase":
        """Apply reset maps for triggered zero-crossing events.

        Delegates to ``system.handle_zero_crossings``.

        Args:
            events: EventCollection with triggered flags set.
            context: The current context.

        Returns:
            Updated context after reset maps have been applied.
        """
        return self.system.handle_zero_crossings(events, context)

    def localize(self, solver_state, context_tf, zc_events, int_t0, int_t1):
        """Localize zero-crossing events via bisection search.

        Uses the ODE solver's dense interpolant to narrow down the time interval
        containing the earliest zero-crossing event.

        Args:
            solver_state: Current ODE solver state (provides interpolant).
            context_tf: Context at the end of the ODE step.
            zc_events: EventCollection with triggered flags set.
            int_t0: Integer time at the start of the interval.
            int_t1: Integer time at the end of the interval.

        Returns:
            tuple: (context_at_zc, updated_zc_events) after bisection.
        """
        _body_fun = partial(_bisection_step_fun, solver_state)
        carry = GuardIsolationData(int_t0, int_t1, zc_events, context_tf)
        search_data = backend.fori_loop(
            0, self.zc_bisection_loop_count, _body_fun, carry
        )
        return search_data.context, search_data.guards

    def check_after_discrete_update(
        self, context: "ContextBase", timed_events: "EventCollection"
    ) -> tuple["ContextBase", bool]:
        """Handle zero-crossings that may be triggered by a discrete update.

        Evaluates guards before and after the discrete update, checks for
        triggers, and applies any reset maps.

        Args:
            context: The current context.
            timed_events: The collection of active timed events.

        Returns:
            tuple: (updated_context, terminate_early).
        """
        system = self.system

        context = context.refresh_port_cache()
        zc_events = self.evaluate_guards(context)

        # Record guard values at interval start
        zc_events = self.record_interval_start(zc_events, context)

        # Handle periodic discrete updates
        context = system.handle_discrete_update(
            timed_events, context,
            topological_order=self.lower_triangular_discrete_update,
        )

        # Record guard values after discrete update and check triggers
        context = context.refresh_port_cache()
        zc_events = self.record_interval_end(zc_events, context)
        zc_events = self.check_triggered(zc_events, context)
        terminate_early = zc_events.has_active_terminal

        # Handle any triggered events
        context = self.handle_events(zc_events, context)

        return context, terminate_early


# ---------------------------------------------------------------------------
# Module-level helper functions (unchanged from original simulator.py)
# ---------------------------------------------------------------------------

import dataclasses

import jax

from ..framework import ZeroCrossingEvent
from ..backend import cond


def _is_zc_event(x):
    return isinstance(x, ZeroCrossingEvent)


def _record_guard_values(
    events: "EventCollection", context: "ContextBase", key: str
) -> "EventCollection":
    """Store the current values of guard functions in the event data.

    The "key" can either be ``"w0"`` or ``"w1"`` to indicate whether the recorded
    values correspond to the start or end of the interval.
    """

    def _update(event: ZeroCrossingEvent):
        return dataclasses.replace(
            event,
            event_data=dataclasses.replace(
                event.event_data,
                **{key: event.guard(context)},
            ),
        )

    return jax.tree_util.tree_map(_update, events, is_leaf=_is_zc_event)


# Convenient partial functions for the two valid values of ``key``.
guard_interval_start = partial(_record_guard_values, key="w0")
guard_interval_end = partial(_record_guard_values, key="w1")


def determine_triggered_guards(
    events: "EventCollection", context: "ContextBase"
) -> "EventCollection":
    """Determine which zero-crossing events are triggered.

    Evaluates the guard functions at the end of the interval and compares the
    sign of the values to the sign at the beginning, using the "direction" rule
    for each individual event.
    """
    events = guard_interval_end(events, context)

    def _update(event: ZeroCrossingEvent):
        return dataclasses.replace(
            event,
            event_data=dataclasses.replace(
                event.event_data,
                triggered=event.should_trigger(),
            ),
        )

    return jax.tree_util.tree_map(_update, events, is_leaf=_is_zc_event)


def _bisection_step_fun(step_sol, i, carry: GuardIsolationData):
    """Perform one step of bisection to localize a zero-crossing event.

    See the original implementation in ``simulator.py`` for full algorithmic
    details.
    """
    int_time_mid = (
        carry.zc_before_time + (carry.zc_after_time - carry.zc_before_time) // 2
    )
    time_mid = IntegerTime.as_decimal(int_time_mid)

    import jax.numpy as jnp
    leaves = jax.tree.leaves(carry.context.continuous_state)
    dtype = leaves[0].dtype if leaves else jnp.empty(0).dtype
    time_mid = jnp.asarray(time_mid, dtype=dtype)
    context_mid = carry.context.with_time(time_mid)
    states_mid = step_sol.eval_interpolant(time_mid)
    context_mid = context_mid.with_continuous_state(states_mid)
    context_mid = context_mid.refresh_port_cache()
    guards_mid = determine_triggered_guards(carry.guards, context_mid)

    carry_first_half = GuardIsolationData(
        carry.zc_before_time,
        int_time_mid,
        guards_mid,
        context_mid,
    )

    carry_second_half = GuardIsolationData(
        int_time_mid,
        carry.zc_after_time,
        carry.guards,
        carry.context,
    )

    return cond(
        guards_mid.has_triggered,
        lambda: carry_first_half,
        lambda: carry_second_half,
    )
