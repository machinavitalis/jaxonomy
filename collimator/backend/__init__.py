# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from .backend import (
    DEFAULT_BACKEND,
    REQUESTED_BACKEND,
    IS_JAXLITE,
    dispatcher,
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
    inf,
    nan,
)

from .ode_solver import ODESolverOptions, ODESolverState

# Alternate name for clear imports `from collimator.backend import numpy_api`
numpy_api = dispatcher

__all__ = [
    "DEFAULT_BACKEND",
    "REQUESTED_BACKEND",
    "IS_JAXLITE",
    "dispatcher",
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
    "inf",
    "nan",
]
