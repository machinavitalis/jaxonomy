# SPDX-License-Identifier: MIT

from . import _init  # noqa: F401
from .framework import (
    LeafSystem,
    DiagramBuilder,
    Parameter,
    parameters,
    ports,
    submodel_function,
)
from .framework.validation import validate_diagram
from .framework.state_machine_builder import StateMachineBuilder
from .backend import numpy_api as backend, set_backend

from .library.nmpc import trajopt
from .library.linear_system import linearize, LinearizedSystem
from .library.linearization_workflow import discretize, with_observer

from .simulation import (
    BatchSimulationResults,
    Simulator,
    event_time_gradient,
    event_time_jacobian,
    event_times_gradient,
    multi_event_time_gradient,
    simulate,
    simulate_batch,
    simulate_distributed,
    simulate_cloud,
    Decay,
    simulate_jacfwd,
    scalar_cost_simulate,
    simulate_static_sweep,
    simulate_with_event_time_grad,
    estimate_max_major_steps,
    ODESolver,
    ODESolverOptions,
    SimulatorOptions,
    vmap_event_time_gradient,
    vmap_event_times_gradient,
)
from .cli import load_model, load_model_from_dir
from . import acausal
from . import uq
from .precision import (
    precision_info,
    assert_float64_active,
    PrecisionInfo,
    precision_policy,
    active_precision_policy,
)
from .jit_cache import enable_persistent_jit_cache
from .version import __version__


__all__ = [
    "__version__",
    "load_model",
    "load_model_from_dir",
    "linearize",
    "LinearizedSystem",
    "discretize",
    "with_observer",
    "LeafSystem",
    "DiagramBuilder",
    "StateMachineBuilder",
    "BatchSimulationResults",
    "Simulator",
    "SimulatorOptions",
    "event_time_gradient",
    "event_time_jacobian",
    "event_times_gradient",
    "multi_event_time_gradient",
    "simulate",
    "simulate_batch",
    "simulate_distributed",
    "simulate_cloud",
    "Decay",
    "simulate_jacfwd",
    "scalar_cost_simulate",
    "simulate_static_sweep",
    "simulate_with_event_time_grad",
    "trajopt",
    "estimate_max_major_steps",
    "ODESolver",
    "ODESolverOptions",
    "backend",
    "set_backend",
    "Parameter",
    "parameters",
    "ports",
    "validate_diagram",
    "acausal",
    "uq",
    "precision_info",
    "assert_float64_active",
    "PrecisionInfo",
    "precision_policy",
    "active_precision_policy",
    "submodel_function",
    "enable_persistent_jit_cache",
    "vmap_event_time_gradient",
    "vmap_event_times_gradient",
]
