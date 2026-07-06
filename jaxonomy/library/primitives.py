# SPDX-License-Identifier: MIT

"""Re-export hub for primitive blocks (backward-compatibility facade).

The class definitions previously housed in this 14,000+ line module have
been split across seven category files for navigability:

* :mod:`jaxonomy.library.sources` — generative sources (deterministic + stochastic)
* :mod:`jaxonomy.library.math_ops` — arithmetic, matrix, and scalar math
* :mod:`jaxonomy.library.logic` — boolean operators, comparators, switches, truth tables
* :mod:`jaxonomy.library.routing` — multiplexing, slicing, datatype conversion, buses
* :mod:`jaxonomy.library.dynamics` — integrators, discrete state, filters, PID, timing
* :mod:`jaxonomy.library.nonlinearities` — clipping, saturation, dead zones, rate limits
* :mod:`jaxonomy.library.tables` — lookup tables and the prelookup family

This file keeps every public and private symbol that was previously
importable as ``jaxonomy.library.primitives.X`` accessible from the same
path, so external callers (tests, model JSON loaders, downstream code)
do not need to change. New code should import from the category modules
directly.
"""

from __future__ import annotations

# Re-export the framework base classes from .generic that were previously
# re-exported transitively through this module (e.g. wrappers.py imports
# ``FeedthroughBlock`` from here).
from .generic import SourceBlock, FeedthroughBlock, ReduceBlock

# Internal helpers shared across all category modules.
from ._primitives_common import (
    _stop_gradient,
    check_state_type,
    is_discontinuity,
)

# Category modules — wildcard imports bring in every public name listed
# in each module's __all__.
from .sources import *  # noqa: F401,F403
from .math_ops import *  # noqa: F401,F403
from .logic import *  # noqa: F401,F403
from .routing import *  # noqa: F401,F403
from .dynamics import *  # noqa: F401,F403
from .nonlinearities import *  # noqa: F401,F403
from .tables import *  # noqa: F401,F403

# Private helpers that downstream tests / code import by name from
# jaxonomy.library.primitives. The wildcard imports above skip names
# starting with an underscore, so these must be re-imported explicitly.
from .sources import _PRNGState, _LFSRState  # noqa: F401
from .dynamics import _DecimatorMeanState  # noqa: F401
from .tables import _PrelookupResult  # noqa: F401


__all__ = [
    # Base classes re-exported for backward compatibility
    "SourceBlock",
    "FeedthroughBlock",
    "ReduceBlock",
    # Shared helpers
    "check_state_type",
    "is_discontinuity",
    # sources
    "Chirp",
    "Clock",
    "Constant",
    "Counter",
    "DiscreteClock",
    "Pulse",
    "Ramp",
    "Sawtooth",
    "Sine",
    "Step",
    "UniformRandomNumber",
    "RandomSource",
    "BandLimitedNoise",
    "PRBS",
    "PRBSLFSR",
    # math_ops
    "Abs",
    "Adder",
    "Arithmetic",
    "CrossProduct",
    "DotProduct",
    "Exponent",
    "Gain",
    "Logarithm",
    "MatrixConcatenation",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "MinMax",
    "Offset",
    "Power",
    "Product",
    "ProductOfElements",
    "Reciprocal",
    "ScalarBroadcast",
    "SquareRoot",
    "Stack",
    "SumOfElements",
    "Trigonometric",
    # logic
    "Comparator",
    "LogicalOperator",
    "LogicalReduce",
    "IfThenElse",
    "Relay",
    "Switch",
    "MultiPortSwitch",
    "TruthTable",
    "TruthTableBuilder",
    # routing
    "Demultiplexer",
    "Multiplexer",
    "Mux",
    "Demux",
    "IOPort",
    "Slice",
    "SignalDatatypeConversion",
    "BusCreator",
    "BusSelector",
    "BusMerge",
    "BusPassthrough",
    "BusUpdate",
    "bus_fields",
    "flatten_bus",
    "merge_buses",
    "unflatten_bus",
    # dynamics
    "Integrator",
    "TransportDelay",
    "VariableTransportDelay",
    "IntegratorDiscrete",
    "FilterDiscrete",
    "DerivativeDiscrete",
    "DiscreteInitializer",
    "UnitDelay",
    "ZeroOrderHold",
    "PIDDiscrete",
    "LowPassDiscrete",
    "LeadLag",
    "Notch",
    "EdgeDetection",
    "Decimator",
    "PIDController2DOF",
    "RateTransition",
    # nonlinearities
    "DeadZone",
    "DeadZoneInverse",
    "Saturate",
    "SoftSaturate",
    "Quantizer",
    "RateLimiter",
    "SoftRateLimiter",
    "Backlash",
    "Stop",
    "soft_saturate",
    "soft_dead_zone",
    # tables
    "LookupTable1d",
    "LookupTable2d",
    "LookupTableND",
    "Prelookup",
    "PrelookupInverse",
    "InterpolationUsingPrelookup",
    "TableSearch",
]
