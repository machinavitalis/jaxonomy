# SPDX-License-Identifier: MIT

"""Classes for evaluating and storing results of calculations.

The SystemCallback class is a mechanism for associating dependencies with a function
defined for a particular system.  This can include ports, event update functions, the
right-hand-side of an ODE, etc.  Declaring these functions as SystemCallbacks will
automatically create the necessary dependency tracking infrastructure to construct and
sort the execution graphs.

At the moment the caching is barely used (only for determining LeafSystem feedthrough
during automatic loop detection), but are preserved in case it is useful for when
function ordering replaces the current lazy evaluation model.  In this case the results
of each SystemCallback (e.g. output port evaluation) can be stored in the cache and
retrieved by other SystemCallbacks that depend on them.
"""

from __future__ import annotations


from typing import TYPE_CHECKING, NamedTuple, Callable, List
import contextvars
import dataclasses

from .dependency_graph import (
    next_dependency_ticket,
    DependencyTicket,
)

from jaxonomy.logging import logger
from jaxonomy.framework.error import CallbackIsNotDifferentiableError

if TYPE_CHECKING:
    from ..backend.typing import Array
    from .dependency_graph import DependencyTracker
    from .system_base import SystemBase
    from .context import ContextBase
    from .event import Event

__all__ = [
    "SystemCallback",
    "CallbackTracer",
]


# Transient memo for a single top-level SystemCallback.eval() call tree.
#
# Outside of `simulate` (which enables and maintains the context port cache)
# every `eval()` on an initialized context recomputes its full upstream cone.
# For diagrams where one output fans out to several consumers that later
# reconverge — the natural shape of composed reference-submodel models — the
# recompute count grows exponentially with composition depth (a K-level
# diamond chain costs 2^(K+2) calls), which in practice hangs
# `create_context()` / `check_types` / eager `port.eval()` on real composed
# models (e.g. the DemoAeroRocketEngine app model: 13 submodel instances,
# 8 nested groups).
#
# The memo lives only for the duration of ONE outermost eval() call: the
# first eval() on the stack installs a fresh dict and always clears it on
# exit, and nested (recursive) evals within that tree reuse computed values.
# The context is immutable for the duration of one eval tree, so this is
# purely an evaluation-count optimization — no value can differ from the
# recompute path, under eager execution and under JAX tracing alike (within
# one trace, deduplicating a pure subexpression is semantics-preserving; it
# also shrinks the traced jaxpr).  Nothing persists across top-level calls,
# so there is nothing to invalidate.  Keyed on (id(root_context), self) so
# an exotic callback that evaluates against a *different* context mid-tree
# simply gets its own memo entries.
_eval_memo: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "system_callback_eval_memo", default=None
)


@dataclasses.dataclass
class SystemCallback:
    """A function associated with a system that has has specified dependencies.

    This can include port update rules, discrete update functions, the right-hand-side
    of an ODE, etc. Storing these functions as SystemCallbacks allows the system, or a
    Diagram containing the system, to track dependencies across the system or diagram.

    Attributes:
        system (SystemBase):
            The system that owns this callback.
        ticket (int):
            The dependency ticket associated with this callback.  See DependencyTicket
            for built-in tickets. If None, a new ticket will be generated.
        name (str):
            A short description of this callback function.
        prerequisites_of_calc (List[DependencyTicket]):
            Direct prerequisites of the computation, used for dependency tracking.
            These might be built-in tickets or tickets associated with other
            SystemCallbacks.
        default_value (Array):
            A dummy value of the same shape/dtype as the result, if known.  If None,
            any type checking will rely on propagating upstream information via the
            callback.
        callback_index (int):
            The index of this function in the system's list of associated callbacks.
        event (Event):
            Optionally, the callback function may be associated with an event.  If so,
            the associated trackers can be used to sort event execution order in addition
            to the regular callback execution order. For example, if an OutputPort is of
            sample-and-hold type, then this will be the event that periodically updates
            the output value. Default is None.
    """

    callback: dataclasses.InitVar[Callable[[ContextBase], Array]]
    system: SystemBase
    callback_index: int
    ticket: DependencyTicket = None
    name: str = None
    prerequisites_of_calc: List[DependencyTicket] = None
    default_value: Array = None
    event: Event = None

    # If the result is cached (e.g. an output port of "sample-and-hold" type),
    # this will be the index of the cache in the system's cache list.
    cache_index: int = None

    def __post_init__(self, callback):
        self._callback = callback  # Given root context, return calculated value

        if self.ticket is None:
            self.ticket = next_dependency_ticket()
        assert isinstance(self.ticket, int)

        if self.prerequisites_of_calc is None:
            self.prerequisites_of_calc = []

        logger.debug(
            "Initialized callback %s:%s with prereqs %s",
            self.system.name_path_str,
            self.name,
            self.prerequisites_of_calc,
        )


    def __hash__(self) -> int:
        locator = (self.system, self.callback_index)
        return hash(locator)

    def __repr__(self) -> str:
        return f"{self.name}(ticket = {self.ticket})"

    def calc(self, root_context: ContextBase) -> Array:
        """Unconditionally evaluate the callback function.

        This does not check the cache status, but will always recompute the value.
        Typically `eval` should be preferred to `calc` to take advantage of caching
        where possible.

        Args:
            root_context: The root context used for the evaluation.

        Returns:
            The calculated value from the callback, expected to be a Array.
        """
        return self._callback(root_context)

    def eval(self, root_context: ContextBase) -> Array:
        """Evaluate the callback function and return the calculated value.

        Within a single top-level call, repeated evaluations of the same
        callback against the same context are memoized (see ``_eval_memo``
        above) — this keeps eager evaluation of diagrams with fan-out /
        reconvergence linear in graph size instead of exponential in
        composition depth.  Nothing is cached across top-level calls.

        Args:
            root_context: The root context used for the evaluation.

        Returns:
            The calculated value from the callback, expected to be a Array.
        """
        if not root_context.is_initialized:
            if self.default_value is None:
                self.default_value = self.calc(root_context)
            return self.default_value

        memo = _eval_memo.get()
        if memo is None:
            # Outermost eval of this tree: install a fresh memo, always
            # clear it on exit so nothing leaks across top-level calls
            # (or across JAX traces).
            token = _eval_memo.set({})
            try:
                return self._eval_initialized(root_context)
            finally:
                _eval_memo.reset(token)

        key = (id(root_context), self)
        if key in memo:
            return memo[key]
        result = self._eval_initialized(root_context)
        memo[key] = result
        return result

    def _eval_initialized(self, root_context: ContextBase) -> Array:
        try:
            result = self.calc(root_context)
        except ValueError as e:
            # this error is raised if the callback is not differentiable
            if "do not support JVP." in str(e):
                raise CallbackIsNotDifferentiableError(
                    system=self.system,
                    port_name=self.name,
                )
            raise
        return result

    @property
    def tracker(self) -> DependencyTracker:
        return self.system.dependency_graph[self.ticket]


class CallbackTracer(NamedTuple):
    """A stand-in for a value in the computation graph.

    The purpose of this class is to track whether a value is modified by the various
    computations in a system.  This is used for automatically determining feedthrough
    port pairs in a LeafSystem.

    Attributes:
        ticket (int):
            The dependency ticket associated with the result of this callback. Mainly
            useful for debugging.
        is_out_of_date (bool):
            Flag indicating whether the value is out of date as a result of upstream
            prerequisite values becoming out of date.
    """

    ticket: int
    is_out_of_date: bool = True

    def mark_up_to_date(self) -> CallbackTracer:
        """Mark the value as up to date.

        Returns:
            A new CallbackTracer object with `is_out_of_date` set to False.
        """
        return self._replace(is_out_of_date=False)  # pylint: disable=no-member

    def mark_out_of_date(self) -> CallbackTracer:
        """Mark the value as out of date.

        Returns:
            A new CallbackTracer object with `is_out_of_date` set to True.
        """
        return self._replace(is_out_of_date=True)  # pylint: disable=no-member

