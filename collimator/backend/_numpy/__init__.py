# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
import numpy as np

from .python_functions import (
    cond,
    scan,
    while_loop,
    fori_loop,
    callback,
    jit,
    astype,
    interp2d,
    switch,
)
from .ode_solver import ODESolver
from .results_data import NumpyResultsData

from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation
else:
    scipy = LazyLoader("scipy", globals(), "scipy")
    Rotation = LazyModuleAccessor(scipy, "spatial.transform.Rotation")

__all__ = ["numpy_functions", "numpy_constants"]

numpy_functions = {
    "astype": astype,
    "cond": cond,
    "scan": scan,
    "while_loop": while_loop,
    "fori_loop": fori_loop,
    "jit": jit,
    "io_callback": callback,
    "pure_callback": callback,
    "ODESolver": ODESolver,
    "interp2d": interp2d,
    "switch": switch,
}

numpy_constants = {
    "lib": np,
    "Rotation": Rotation,
    "ResultsDataImpl": NumpyResultsData,
}
