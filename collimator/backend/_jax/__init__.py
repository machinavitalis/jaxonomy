# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
import jax
from jax import lax
import jax.numpy as jnp
from .python_functions import interp2d

from .ode_solver import ODESolver
from .results_data import JaxResultsData

from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from jax.scipy.spatial.transform import Rotation
else:
    # NOTE: spatial and transform seem to be lazy loaded in jax.scipy too
    transform = LazyLoader("transform", globals(), "jax.scipy.spatial.transform")
    Rotation = LazyModuleAccessor(transform, "Rotation")

__all__ = ["jax_functions", "jax_constants"]


jax_functions = {
    "cond": lax.cond,
    "scan": lax.scan,
    "while_loop": lax.while_loop,
    "fori_loop": lax.fori_loop,
    "jit": jax.jit,
    "io_callback": jax.experimental.io_callback,
    "pure_callback": jax.pure_callback,
    "ODESolver": ODESolver,
    "interp2d": interp2d,
    "switch": lax.switch,
}

jax_constants = {
    "lib": jnp,
    "Rotation": Rotation,
    "ResultsDataImpl": JaxResultsData,
}
