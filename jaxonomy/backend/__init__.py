# SPDX-License-Identifier: MIT

from .backend import (
    DEFAULT_BACKEND,
    REQUESTED_BACKEND,
    IS_JAXLITE,
    get_dispatcher,
    set_backend,
    asarray,
    array,
    zeros,
    zeros_like,
    reshape,
    Rotation,
    cond,
    scan,
    while_loop,
    fori_loop,
    jit,
    io_callback,
    pure_callback,
    ODESolver,
    ResultsData,
    stop_gradient,
    inf,
    nan,
)

from .ode_solver import ODESolverOptions, ODESolverState

# Alternate name for clear imports `from jaxonomy.backend import numpy_api`
class NumpyApiProxy:
    def __getattr__(self, name):
        return getattr(get_dispatcher(), name)
        
numpy_api = NumpyApiProxy()

__all__ = [
    "DEFAULT_BACKEND",
    "REQUESTED_BACKEND",
    "IS_JAXLITE",
    "get_dispatcher",
    "set_backend",
    "asarray",
    "array",
    "zeros",
    "zeros_like",
    "reshape",
    "Rotation",
    "cond",
    "scan",
    "while_loop",
    "fori_loop",
    "jit",
    "io_callback",
    "pure_callback",
    "ODESolver",
    "ODESolverOptions",
    "ODESolverState",
    "ResultsData",
    "stop_gradient",
    "inf",
    "nan",
]
