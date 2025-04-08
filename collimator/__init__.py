# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from . import _init  # noqa: F401
from .framework import (
    LeafSystem,
    DiagramBuilder,
    Parameter,
    parameters,
    ports,
)
from .backend import dispatcher as backend

from .library.nmpc import trajopt
from .library.linear_system import linearize

from .simulation import (
    Simulator,
    simulate,
    estimate_max_major_steps,
    ODESolver,
    ODESolverOptions,
    SimulatorOptions,
)
from .cli import load_model, load_model_from_dir
from .version import __version__

set_backend = backend.set_backend

__all__ = [
    "__version__",
    "load_model",
    "load_model_from_dir",
    "linearize",
    "LeafSystem",
    "DiagramBuilder",
    "Simulator",
    "SimulatorOptions",
    "simulate",
    "trajopt",
    "estimate_max_major_steps",
    "ODESolver",
    "SimulatorOptions",
    "ODESolverOptions",
    "backend",
    "set_backend",
    "Parameter",
    "parameters",
    "ports",
]
