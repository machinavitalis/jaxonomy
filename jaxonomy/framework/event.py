# SPDX-License-Identifier: MIT

"""Event classes for hybrid system simulation.

This module defines classes used for event-driven simulation of hybrid systems.
These classes are used internally by the simulation framework and should not
normally need to be used directly by users. Instead, users can declare events
on LeafSystems.  The events will be organized into EventCollections and handled
by the simulation framework.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Callable, List, Hashable
from collections import OrderedDict

import numpy as np
from jax import tree_util

# Import the switchable backend dispatcher as "jaxonomy.numpy" or "npa"
from ..backend import cond, stop_gradient, numpy_api as npa


def _sever_discrete_cotangents(tree: Any) -> Any:
    """Apply ``stop_gradient`` to every integer / boolean leaf of ``tree``.

    T-001c-followup #1b: an event's update is dispatched with a
    ``cond(active, callback, passthrough)``. The two branches read /
    write different subsets of the root ``context``, so during reverse-mode
    transpose the cotangent of a non-differentiable discrete leaf (e.g. a
    state-machine ``mode`` integer, whose tangent dtype is ``float0``)
    comes out *weakly-typed* (``~float0`` ``Zero``) from the branch that
    passes it through and *strongly-typed* (``float0`` ``Zero``) from the
    branch that overwrites or never reads it. JAX's ``_cond_transpose``
    then asserts both branches produce an identical output ``PyTreeDef``
    (``out_tree, = set(out_trees)``) and raises ``ValueError: too many
    values to unpack`` on the weak-vs-strong disagreement.

    Severing the gradient of integer / boolean leaves makes their float0
    cotangents agree (both become a consistently-typed ``Zero`` produced by
    the same ``stop_gradient`` transpose), so the uniqueness assertion
    holds. ``float`` and other differentiable leaves are left untouched, so
    real gradients still flow. The forward pass is unchanged:
    ``stop_gradient`` is the identity in value (and a no-op under the numpy
    backend).
    """

    def _maybe_sever(leaf):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and (
            np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)
        ):
            return stop_gradient(leaf)
        return leaf

    return tree_util.tree_map(_maybe_sever, tree)


def _handle_with_severed_discretes(active, callback, passthrough, context):
    """``cond(active, callback, passthrough, context)`` with the integer /
    boolean leaves severed on both the input ``context`` and each branch's
    result, so the two branches agree on every float0 cotangent's type at
    transpose time. See :func:`_sever_discrete_cotangents` (T-001c #1b).

    Severing the *input* context inside both branches is what fixes the
    mismatch: it guarantees the cotangent of every non-differentiable cond
    input leaf is produced by the *same* ``stop_gradient`` transpose in
    both branches (an identically-typed ``Zero``), regardless of whether a
    branch reads that leaf. Severing the output is belt-and-suspenders for
    discrete leaves a branch newly writes.
    """

    def _active(ctx):
        ctx = _sever_discrete_cotangents(ctx)
        return _sever_discrete_cotangents(callback(ctx))

    def _inactive(ctx):
        ctx = _sever_discrete_cotangents(ctx)
        return _sever_discrete_cotangents(passthrough(ctx))

    return cond(active, _active, _inactive, context)


if TYPE_CHECKING:
    from ..backend.typing import Scalar, Array, DTypeLike
    from .context import ContextBase
    from .state import LeafState

__all__ = [
    "IntegerTime",
    "DiscreteUpdateEvent",
    "ZeroCrossingEvent",
    "PeriodicEventData",
    "ZeroCrossingEventData",
    "EventCollection",
    "LeafEventCollection",
    "FlatEventCollection",
    "DiagramEventCollection",
    "is_event_data",
]


# Default time scale for integer time representation (picosecond resolution)
DEFAULT_TIME_SCALE = 1e-12


class IntegerTime:
    """Class for managing conversion between decimal and integer time."""

    # TODO: Can we use this directly as an int?  Would need to implement __add__,
    # __sub__, etc.  Also, comparisons and floor divide.  Would make the code in
    # Simulator cleaner, but dealing with JAX tracers in `where` and the like
    # might be difficult.  See commit 043c8f757 for a previous attempt.

    #
    # Class variables
    #
    time_scale = DEFAULT_TIME_SCALE  # int -> float conversion factor
    inv_time_scale = 1 / time_scale  # float -> int conversion factor

    # Type of the integer time representation. Defaults to x64 unless explicitly disabled.
    dtype: DTypeLike = npa.intx

    # Largest time value representable by IntegerTime.dtype
    max_int_time = npa.iinfo(dtype).max

    # Floating point representation of max_int_time. Built with concrete
    # numpy (not the backend ``npa``) so it stays a host-side constant: it is
    # read via ``float(...)`` in the representability check and must never
    # become a JAX tracer when ``set_scale`` runs inside an autodiff-through-
    # ``simulate`` trace (T-B6-followup-int-time-scale-trace-safety).
    max_float_time = np.asarray(max_int_time * time_scale, dtype=dtype)

    #
    # Class methods
    #
    @classmethod
    def set_scale(cls, time_scale: float):
        cls.time_scale = time_scale
        cls.inv_time_scale = 1 / time_scale
        # Concrete numpy — keep host-side / tracer-free (see class attr note).
        cls.max_float_time = np.asarray(cls.max_int_time * time_scale, dtype=cls.dtype)

    @classmethod
    def set_default_scale(cls):
        cls.set_scale(DEFAULT_TIME_SCALE)

    @classmethod
    def from_decimal(cls, time: float) -> int:
        """Convert a floating-point time to an integer time."""
        # First limit to the max value to avoid overflow with inf or very large values.
        time = npa.minimum(time, cls.max_float_time)
        return npa.asarray(time * cls.inv_time_scale, dtype=cls.dtype)

    @classmethod
    def as_decimal(cls, time: int) -> float:
        """Convert an integer time to a floating-point time."""
        return time * cls.time_scale


@dataclasses.dataclass(frozen=True)
class EventData:
    active: bool


@dataclasses.dataclass(frozen=True)
class PeriodicEventData(EventData):
    period: float  # Period of the event
    offset: float  # Offset from the start of the simulation for the initial event

    # Time of the next event sample, as determined by the simulation loop.
    # The initial value will be overwritten by the simulation loop.
    next_sample_time: int = 0

    @property
    def period_int(self) -> int:
        return IntegerTime.from_decimal(self.period)

    @property
    def offset_int(self) -> int:
        return IntegerTime.from_decimal(self.offset)


@dataclasses.dataclass(frozen=True)
class ZeroCrossingEventData(EventData):
    w0: Scalar = npa.inf  # Guard value at beginning of interval
    w1: Scalar = npa.inf  # Guard value at end of interval
    triggered: bool = False


#
# Trigger functions for zero-crossing events
#
def _none_trigger(w0: Scalar, w1: Scalar) -> bool:
    return False


def _positive_then_nonpositive_trigger(w0: Scalar, w1: Scalar) -> bool:
    return (w0 > 0) & (w1 <= 0)


def _negative_then_nonnegative_trigger(w0: Scalar, w1: Scalar) -> bool:
    return (w0 < 0) & (w1 >= 0)


def _crosses_zero_trigger(w0: Scalar, w1: Scalar) -> bool:
    return ((w0 > 0) & (w1 <= 0)) | ((w0 < 0) & (w1 >= 0))


def _edge_detection(w0: Scalar, w1: Scalar) -> bool:
    return w0 != w1


_zero_crossing_trigger_functions = {
    "none": _none_trigger,
    "positive_then_non_positive": _positive_then_nonpositive_trigger,
    "negative_then_non_negative": _negative_then_nonnegative_trigger,
    "crosses_zero": _crosses_zero_trigger,
    "edge_detection": _edge_detection,
}


#
# Event classes
#
def is_event_data(x: Any) -> bool:
    return isinstance(x, EventData)


def _activate(self, activation_fn):
    """Map a bool-valued activation function over all events in the tree."""

    def _activate_helper(event_data):
        if is_event_data(event_data):
            return dataclasses.replace(event_data, active=activation_fn(event_data))
        return event_data

    return tree_util.tree_map(
        _activate_helper,
        self,
        is_leaf=is_event_data,
    )


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Event:
    """Class representing a discontinuous update event in a hybrid system.

    Users should not need to interact with these objects directly. They are intended
    to be used internally by the simulation framework for handling events in hybrid
    system simulation. In a normal workflow, events will be declared on LeafSystems
    using `declare_*` methods, and the simulation framework will organize them into
    EventCollections.
    """

    # The ID of the originating system
    system_id: Hashable

    name: str = None

    event_data: EventData = None

    # The callback function is called when the event is triggered. The callback will
    # be passed the root context, but the return value will vary depending on the event
    # type (as defined by subclass implementations).
    callback: Callable[[ContextBase], Any] = None

    # The "passthrough" is a dummy callback that happens if the real callback
    # is not active (False branch of conditional). This is required to have a
    # consistent signature with `callback` for `lax.cond`.
    passthrough: Callable[[ContextBase], Any] = None

    # If true, the update calls will use the structured control flow provided by LAX
    # rather than the standard Python control flow.
    enable_tracing: bool = True

    # If true, this event updates discrete state (x[n+1] = f(x[n], u[n])).
    # If false, this event is an output cache update (y[n] = g(x[n])).
    # Used by handle_discrete_update to implement two-phase x⁻ atomicity:
    # cache updates fire first (so y[n] is correct), then state updates are
    # evaluated against a frozen snapshot (so no block sees another block's x⁺).
    is_state_update: bool = False

    def __post_init__(self):
        if self.passthrough is None:

            def _default_passthrough(context: ContextBase) -> ContextBase:
                return context

            self.passthrough = _default_passthrough

    def __repr__(self) -> str:
        return f"{self.event_data}"

    # Proper typing here is difficult because the return type of the callback will
    # vary depending on the type of callback (determined by the subclass). For example,
    # the callback in a DiscreteUpdateEvent will return an Array, while reset map calls
    # return a LeafState.
    def handle(self, context: ContextBase) -> Any:
        """Conditionally compute the result of the update callback

        If the event is marked "inactive" via its event data attribute, the passthrough
        callback will be called instead of the update callback. Otherwise, the update
        callback will be called. The return types of both callbacks must match, but the
        specific type will depend on the kind of event.
        """
        if self.enable_tracing:
            return _handle_with_severed_discretes(
                self.event_data.active,
                self.callback,
                self.passthrough,
                context,
            )

        # No tracing: use standard control flow
        if not self.event_data.active:
            return self.passthrough(context)
        return self.callback(context)

    def mark_active(self) -> Event:
        """Create a copy of the event with the status marked active"""
        return _activate(self, lambda _: True)

    def mark_inactive(self) -> Event:
        """Create a copy of the event with the status marked inactive"""
        return _activate(self, lambda _: False)

    #
    # PyTree registration
    #
    # Normally it's convenient to move this out of the class definition, but because
    # there are several subclasses that can all be registered in the same way, having
    # it defined here allows code reuse.
    def tree_flatten(self):
        children = (self.event_data,)
        aux_data = (
            self.system_id,
            self.callback,
            self.passthrough,
            self.enable_tracing,
            self.is_state_update,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (event_data,) = children
        system_id, callback, passthrough, enable_tracing, is_state_update = aux_data
        return cls(
            system_id=system_id,
            event_data=event_data,
            callback=callback,
            passthrough=passthrough,
            enable_tracing=enable_tracing,
            is_state_update=is_state_update,
        )


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class DiscreteUpdateEvent(Event):
    """Event representing a discrete update in a hybrid system."""

    # Supersede type hints in Event with the specific signature for discrete updates
    callback: Callable[[ContextBase], Array] = None
    passthrough: Callable[[ContextBase], Array] = None

    # Inherits docstring from Event. This is only needed to specialize type hints.
    def handle(self, context: ContextBase) -> Array:
        return super().handle(context)


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class ZeroCrossingEvent(Event):
    """An event that triggers when a specified "guard" function crosses zero.

    The event is triggered when the guard function crosses zero in the specified
    direction. In addition to the guard callback, the event also has a "reset map"
    which is called when the event is triggered. The reset map may update any state
    component in the system.

    The event can also be defined as "terminal", which means that the simulation will
    terminate when the event is triggered. (TODO: Does the reset map still happen?)

    The "direction" of the zero-crossing is one of the following:
        - "none": Never trigger the event (can be useful for debugging)
        - "positive_then_non_positive": Trigger when the guard goes from positive to
            non-positive
        - "negative_then_non_negative": Trigger when the guard goes from negative to
            non-negative
        - "crosses_zero": Trigger when the guard crosses zero in either direction
        - "edge_detection": Trigger when the guard changes value

    Notes:
        This class should typically not need to be used directly by users. Instead,
        declare the guard function and reset map on a LeafSystem using the
        `declare_zero_crossing` method.  The event will then be auto-generated for
        simulation.
    """

    # Supersede type hints in Event with the specific signature for full-state updates
    callback: Callable[[ContextBase], LeafState] = None
    passthrough: Callable[[ContextBase], LeafState] = None

    guard: Callable[[ContextBase], Scalar] = None
    reset_map: dataclasses.InitVar[Callable[[ContextBase], LeafState]] = None
    direction: str = "crosses_zero"
    is_terminal: bool = False
    event_data: ZeroCrossingEventData = None

    # Optional *smooth* guard residual (``context -> scalar``) used ONLY by the
    # reverse-mode event-time (saltation) gradient machinery — never for
    # triggering or localization, which always use ``guard``.  When the trigger
    # ``guard`` is non-smooth (e.g. a boolean ``where(x>c, 1, -1)`` predicate as
    # a StateMachine emits), its gradient is identically zero and the
    # implicit-function event-time formula ``dt_e/dp = -∇g/D`` is unrecoverable;
    # supplying a smooth residual whose zero coincides with the trigger (e.g.
    # ``x - c``) lets ``∇g`` / ``D`` be taken from it instead.  ``None`` (the
    # default) means "use ``guard``" — byte-equivalent to the legacy path.
    # T-NEW-sm-smooth-guard.
    grad_guard: Callable[[ContextBase], Scalar] = None

    # If not none, only trigger when in this mode. This logic is handled by the owning
    # leaf system.
    active_mode: int = None

    def __post_init__(self, reset_map):  # pylint: disable=arguments-differ
        if self.callback is None:
            self.callback = reset_map

    def _should_trigger(self, w0: Scalar, w1: Scalar) -> bool:
        """Determine if the event should trigger.

        This will use the provided beginning/ending guard value (w0 and w1, resp.),
        as well as the direction of the zero-crossing event. Additionally, the event
        will only trigger if it has been marked as "active", indicating for example
        that the system is in the correct "mode" or "stage" from which the event might
        trigger.
        """
        active = self.event_data.active

        trigger_func = _zero_crossing_trigger_functions[self.direction]
        return active & trigger_func(w0, w1)

    def should_trigger(self) -> bool:
        """Determine if the event should trigger based on the stored guard values."""
        return self._should_trigger(self.event_data.w0, self.event_data.w1)

    def handle(self, context: ContextBase) -> LeafState:
        """Conditionally compute the result of the zero crossing callback

        If the zero crossing is marked "inactive" via its event data attribute, the passthrough
        callback will be called instead of the update callback. Otherwise, the update
        callback will be called. The return types of both callbacks must match, but the
        specific type will depend on the kind of event.
        """
        if self.enable_tracing:  # not driven by simulator.enable_tracing.
            return _handle_with_severed_discretes(
                self.event_data.active & self.event_data.triggered,
                self.callback,
                self.passthrough,
                context,
            )

        # No tracing: use standard control flow
        if self.event_data.active & self.event_data.triggered:
            return self.callback(context)
        return self.passthrough(context)

    #
    # PyTree registration
    #
    def tree_flatten(self):
        children = (self.event_data,)
        aux_data = (
            self.system_id,
            self.guard,
            self.callback,
            self.name,
            self.direction,
            self.is_terminal,
            self.passthrough,
            self.enable_tracing,
            self.active_mode,
            self.grad_guard,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (event_data,) = children
        (
            system_id,
            guard,
            callback,
            name,
            direction,
            is_terminal,
            passthrough,
            enable_tracing,
            active_mode,
            grad_guard,
        ) = aux_data
        return cls(
            system_id=system_id,
            event_data=event_data,
            guard=guard,
            grad_guard=grad_guard,
            callback=callback,
            name=name,
            direction=direction,
            is_terminal=is_terminal,
            passthrough=passthrough,
            enable_tracing=enable_tracing,
            active_mode=active_mode,
        )


#
# Event collections
#
class EventCollection(metaclass=abc.ABCMeta):
    """A collection of events owned by a system.

    Users should not need to interact with these objects directly. They are intended
    to be used internally by the simulation framework for handling events in hybrid
    system simulation.

    These contain callback functions that update the context in various ways
    when the event is triggered. There will be different "collections" for each
    trigger type in simulation (e.g. periodic vs zero-crossing). Within the
    collections, events are broken out by function (e.g. discrete vs unrestricted
    updates).

    There are separate implementations for leaf and diagram systems, where the
    DiagramCEventCollection preserves the tree structure of the underlying
    Diagram. However, the interface in both cases is the same and is identical to
    the interface defined by EventCollection.
    """

    @abc.abstractmethod
    def __getitem__(self, key: Hashable) -> EventCollection:
        pass

    @property
    @abc.abstractmethod
    def events(self) -> List[Event]:
        pass

    @property
    @abc.abstractmethod
    def num_events(self) -> int:
        pass

    @property
    def has_events(self) -> bool:
        return self.num_events > 0

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return self.num_events

    @abc.abstractmethod
    def activate(self, activation_fn) -> EventCollection:
        pass

    def mark_all_active(self) -> EventCollection:
        return self.activate(lambda _: True)

    def mark_all_inactive(self) -> EventCollection:
        return self.activate(lambda _: False)

    @property
    def num_active(self) -> int:
        def _get_active(event_data: EventData) -> bool:
            return event_data.active

        active_tree = tree_util.tree_map(
            _get_active,
            self,
            is_leaf=is_event_data,
        )
        return sum(tree_util.tree_leaves(active_tree))

    @property
    def has_active(self) -> bool:
        return self.num_active > 0

    @property
    def has_triggered(self) -> bool:
        def _get_triggered(event_data: EventData) -> bool:
            return event_data.active & event_data.triggered

        triggered_tree = tree_util.tree_map(
            _get_triggered,
            self,
            is_leaf=is_event_data,
        )
        return sum(tree_util.tree_leaves(triggered_tree)) > 0

    @property
    @abc.abstractmethod
    def terminal_events(self) -> EventCollection:
        pass

    @property
    def has_terminal_events(self):
        return self.terminal_events.has_events

    @property
    def has_active_terminal(self) -> bool:
        return self.terminal_events.has_triggered

    def pprint(self, output=print):
        output(self._pprint_helper().strip())

    def _pprint_helper(self, prefix="") -> str:
        s = f"{prefix}|-- \n"
        if len(self.events) > 0:
            s += f"{prefix}    Events:\n"
            for event in self.events:
                s += f"{prefix}    |  {event}\n"
        return s

    def __repr__(self) -> str:
        s = f"{type(self).__name__}("
        if self.has_events:
            s += f"discrete_update: {self.events} "
        s += ")"

        return s


# This will be registered as a pytree for use in simulation - it should be treated
#  as mutable during system construction, but immutable once the system is finalized.
@dataclasses.dataclass(frozen=True)
class LeafEventCollection(EventCollection):
    _events: tuple[Event, ...] = ()

    def __getitem__(self, _key: Hashable) -> LeafEventCollection:
        # Dummy implementation for compatibility with the EventCollection interface.
        return self

    @property
    def events(self) -> tuple[Event, ...]:
        return self._events

    @property
    def num_events(self) -> int:
        return len(self._events)

    def __add__(self, other: LeafEventCollection) -> LeafEventCollection:
        return type(self)(_events=tuple(list(self.events) + list(other.events)))

    def activate(self, activation_fn) -> LeafEventCollection:
        return _activate(self, activation_fn)

    @property
    def terminal_events(self) -> LeafEventCollection:
        return LeafEventCollection(
            _events=tuple(
                event
                for event in self.events
                if isinstance(event, ZeroCrossingEvent) and event.is_terminal
            )
        )


class FlatEventCollection(LeafEventCollection):
    # NOTE: The use of an extra class that is identical to
    # "Leaf"EventCollection here is obviously unnecessary, but was done as part
    # of WC-185 as a temporary solution until "full event sorting" is possible
    # via WC-188.
    #
    # After WC-188 we will not need `eval_zero_crossing_events` and all event
    # collections can be flattened (what's now called "Leaf"EventCollection).
    # At that point, this class will be renamed `EventCollection` and
    # what is now `DiagramEventCollection` will be fully removed.  In the
    # meantime, all discrete updates are handled using `LeafEventCollection` and
    # zero-crossing events are tracked using either `LeafEventCollection` or
    # `DiagramEventCollection` depending on the system type.  This makes no
    # difference from the point of view of the user or the simulation logic,
    # it's just a temporarily confusing naming scheme until we can sort _all_
    # event types and not just discrete output update events.

    pass


@dataclasses.dataclass(frozen=True)
class DiagramEventCollection(EventCollection):
    subevent_collection: OrderedDict[Hashable, LeafEventCollection] = dataclasses.field(
        default_factory=OrderedDict
    )

    def __getitem__(self, key: Hashable) -> LeafEventCollection:
        return self.subevent_collection[key]

    @property
    def num_subevents(self) -> int:
        return len(self.subevent_collection)

    @property
    def num_events(self) -> int:
        return sum(
            subevent.num_events for subevent in self.subevent_collection.values()
        )

    @property
    def events(self) -> tuple[Event, ...]:
        # Return a flattened list of all events
        events = []
        for subevent in self.subevent_collection.values():
            events.extend(subevent.events)
        return tuple(events)

    def __add__(self, other: DiagramEventCollection) -> DiagramEventCollection:
        assert self.num_subevents == other.num_subevents
        subevent_collection = OrderedDict()
        for sys_id in self.subevent_collection:
            subevent_collection[sys_id] = (
                self.subevent_collection[sys_id] + other.subevent_collection[sys_id]
            )
        return DiagramEventCollection(subevent_collection)

    @property
    def has_events(self) -> bool:
        for subevents in self.subevent_collection.values():
            if subevents.has_events:
                return True
        return False

    def activate(self, activation_fn) -> DiagramEventCollection:
        subevent_collection = OrderedDict(
            {
                sys_id: subevents.activate(activation_fn)
                for sys_id, subevents in self.subevent_collection.items()
            }
        )
        return dataclasses.replace(self, subevent_collection=subevent_collection)

    @property
    def terminal_events(self) -> DiagramEventCollection:
        subevent_collection = OrderedDict(
            {
                sys_id: subevents.terminal_events
                for sys_id, subevents in self.subevent_collection.items()
            }
        )
        return dataclasses.replace(self, subevent_collection=subevent_collection)


#
# PyTree registration
#
def periodic_event_data_flatten(event_data: PeriodicEventData):
    children = (
        event_data.active,
        event_data.next_sample_time,
    )
    aux_data = (event_data.period, event_data.offset)
    return children, aux_data


def periodic_event_data_unflatten(aux_data, children):
    active, next_sample_time = children
    period, offset = aux_data
    return PeriodicEventData(
        active=active,
        period=period,
        offset=offset,
        next_sample_time=next_sample_time,
    )


tree_util.register_pytree_node(
    PeriodicEventData,
    periodic_event_data_flatten,
    periodic_event_data_unflatten,
)


def zero_crossing_data_flatten(event_data: ZeroCrossingEventData):
    children = (event_data.active, event_data.w0, event_data.w1, event_data.triggered)
    aux_data = ()
    return children, aux_data


def zero_crossing_data_unflatten(aux_data, children):
    active, w0, w1, triggered = children
    return ZeroCrossingEventData(
        active=active,
        w0=w0,
        w1=w1,
        triggered=triggered,
    )


tree_util.register_pytree_node(
    ZeroCrossingEventData,
    zero_crossing_data_flatten,
    zero_crossing_data_unflatten,
)


def leaf_collection_flatten(collection: LeafEventCollection):
    children = (collection._events,)
    aux_data = ()
    return children, aux_data


def leaf_collection_unflatten(aux_data, children):
    (events,) = children
    return LeafEventCollection(_events=events)


tree_util.register_pytree_node(
    LeafEventCollection,
    leaf_collection_flatten,
    leaf_collection_unflatten,
)


tree_util.register_pytree_node(
    FlatEventCollection,
    leaf_collection_flatten,
    leaf_collection_unflatten,
)


def diagram_collection_flatten(collection: DiagramEventCollection):
    children = collection.subevent_collection.values()
    aux_data = collection.subevent_collection.keys()
    return children, aux_data


def diagram_collection_unflatten(aux_data, children):
    subevent_collection = OrderedDict(zip(aux_data, children))
    return DiagramEventCollection(subevent_collection)


tree_util.register_pytree_node(
    DiagramEventCollection,
    diagram_collection_flatten,
    diagram_collection_unflatten,
)
