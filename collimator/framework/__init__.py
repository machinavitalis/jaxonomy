# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
from . import build_recorder
from .state import LeafState
from .context import (
    ContextBase,
    LeafContext,
    DiagramContext,
)
from .event import (
    IntegerTime,
    DiscreteUpdateEvent,
    PeriodicEventData,
    ZeroCrossingEvent,
    ZeroCrossingEventData,
    EventCollection,
    LeafEventCollection,
    DiagramEventCollection,
    is_event_data,
)

from .cache import (
    SystemCallback,
)

from .system_base import SystemBase
from .system_decorators import parameters, ports
from .leaf_system import LeafSystem
from .diagram_builder import DiagramBuilder
from .diagram import Diagram
from .parameter import Parameter, ParameterCache
from .error import (
    CollimatorError,
    StaticError,
    ShapeMismatchError,
    DtypeMismatchError,
    BlockInitializationError,
    BlockParameterError,
    BlockRuntimeError,
    ErrorCollector,
)

from .dependency_graph import (
    DependencyTicket,
    next_dependency_ticket,
)

if TYPE_CHECKING:
    from .state import State


__all__ = [
    "SystemCallback",
    "LeafState",
    "State",
    "ContextBase",
    "LeafContext",
    "DiagramContext",
    "IntegerTime",
    "DiscreteUpdateEvent",
    "PeriodicEventData",
    "ZeroCrossingEvent",
    "ZeroCrossingEventData",
    "EventCollection",
    "LeafEventCollection",
    "DiagramEventCollection",
    "is_event_data",
    "SystemBase",
    "LeafSystem",
    "DiagramBuilder",
    "Diagram",
    "StaticError",
    "CollimatorError",
    "ShapeMismatchError",
    "DtypeMismatchError",
    "BlockInitializationError",
    "BlockParameterError",
    "BlockRuntimeError",
    "ErrorCollector",
    "Parameter",
    "ParameterCache",
    "DependencyTicket",
    "next_dependency_ticket",
    "parameters",
    "ports",
    "build_recorder",
]
