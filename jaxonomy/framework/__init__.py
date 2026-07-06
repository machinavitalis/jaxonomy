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
from .flatten import flatten_diagram
from .parameter import Parameter, ParameterCache
from .submodel import submodel_function
from .error import (
    JaxonomyError,
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

from . import units
from . import unit_propagation
from .unit_propagation import propagate_diagram_units
from .units import (
    Unit,
    BusUnit,
    UnitMismatchError,
    assert_unit_compatible,
    are_units_compatible,
    dimensionless,
    dimensionless_unit,
    # T-104-followup-derived-units: helper + extra derived units.
    derived_unit,
    coulomb,
    volt,
    ohm,
    farad,
    weber,
    henry,
    pascal,
    tesla,
    # T-104-followup-currency-units: monetary units + FX conversion.
    usd,
    eur,
    gbp,
    jpy,
    cad,
    set_fx_rate,
    get_fx_rate,
    clear_fx_rates,
    convert_currency,
    CURRENCY_CODES,
)

# T-111 phase 1: build-time Variants / Configurable Diagrams.
# T-111-followup-runtime-switch: RuntimeVariantSubsystem (simulate-time switch).
# T-111-followup-with-config: apply_variant_config / Diagram.with_config
# (post-build configurator).
# T-111-followup-variant-introspection: list_variants / get_variant_choices /
# get_active_variant (discovery helpers for diagrams that contain variants).
from .variants import (
    Variant,
    VariantError,
    select_variant,
    variant_subsystem,
    RuntimeVariantSubsystem,
    apply_variant_config,
    list_variants,
    get_variant_choices,
    get_active_variant,
)

# T-120 phase 1: Container Blocks (EnabledSubsystem, TriggeredSubsystem,
# ForEach). ForEach is a block-diagram-vocabulary alias for ReplicatedFunction.
# T-120-followup-loop-blocks: ForLoop / WhileLoop iteration containers.
# T-120-followup-zc-trigger: ZeroCrossingTriggeredSubsystem (sub-sample
# precision triggered subsystem driven by the framework's continuous
# zero-crossing detector).
from .containers import (
    EnabledSubsystem,
    TriggeredSubsystem,
    ZeroCrossingTriggeredSubsystem,
    ForEach,
    ForLoop,
    WhileLoop,
    EnabledMode,
    EnabledStateMode,
    TriggerEdge,
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
    "JaxonomyError",
    "ShapeMismatchError",
    "DtypeMismatchError",
    "BlockInitializationError",
    "BlockParameterError",
    "BlockRuntimeError",
    "ErrorCollector",
    "Parameter",
    "ParameterCache",
    "submodel_function",
    "DependencyTicket",
    "next_dependency_ticket",
    "parameters",
    "ports",
    "build_recorder",
    "flatten_diagram",
    "units",
    "Unit",
    "BusUnit",
    "UnitMismatchError",
    "assert_unit_compatible",
    "are_units_compatible",
    "dimensionless",
    "dimensionless_unit",
    "derived_unit",
    "coulomb",
    "volt",
    "ohm",
    "farad",
    "weber",
    "henry",
    "pascal",
    "tesla",
    "usd",
    "eur",
    "gbp",
    "jpy",
    "cad",
    "set_fx_rate",
    "get_fx_rate",
    "clear_fx_rates",
    "convert_currency",
    "CURRENCY_CODES",
    "Variant",
    "VariantError",
    "select_variant",
    "variant_subsystem",
    "RuntimeVariantSubsystem",
    "apply_variant_config",
    "list_variants",
    "get_variant_choices",
    "get_active_variant",
    "EnabledSubsystem",
    "TriggeredSubsystem",
    "ZeroCrossingTriggeredSubsystem",
    "ForEach",
    "ForLoop",
    "WhileLoop",
    "EnabledMode",
    "EnabledStateMode",
    "TriggerEdge",
]
