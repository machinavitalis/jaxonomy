# SPDX-License-Identifier: MIT

from __future__ import annotations
import os
from typing import Callable

import numpy as np
from jax.tree_util import register_pytree_node_class
import jax

from ._numpy import numpy_functions, numpy_constants
from .results_data import AbstractResultsData

# IS_JAXLITE is used for the pyodide build where we only have numpy and jaxlite
# FIXME: set JAXONOMY_BACKEND instead of IS_JAXLITE
IS_JAXLITE = os.environ.get("JAXLITE", "0") == "1"
REQUESTED_BACKEND = os.environ.get("JAXONOMY_BACKEND", None)
DEFAULT_BACKEND = REQUESTED_BACKEND or ("jax" if not IS_JAXLITE else "numpy")

if not IS_JAXLITE:
    from ._jax import jax_functions, jax_constants
    from ._torch import torch_functions, torch_constants


def _make_backend(name, functions, constants):
    # Create a new class with the given name and attributes
    static_functions = {
        name: staticmethod(function) for name, function in functions.items()
    }
    attrs = {**static_functions, **constants}
    return type(name, (), attrs)()


import contextvars

class MathDispatcher:
    """Class for calling out to the appropriate backend."""
    
    _backends = {
        "numpy": _make_backend("NumpyBackend", numpy_functions, numpy_constants),
    }

    if not IS_JAXLITE:
        _backends["jax"] = _make_backend("JaxBackend", jax_functions, jax_constants)

    # only load torch backend if requested (it's quite likely broken)
    if REQUESTED_BACKEND == "torch":
        _backends["torch"] = _make_backend(
            "TorchBackend", torch_functions(), torch_constants
        )

    def __init__(self, backend_name=DEFAULT_BACKEND) -> None:
        self._active_backend = backend_name
        self._disable_x64 = False
        
        if IS_JAXLITE:
            return

        # FIXME can't switch to 32 bits after init
        enable_x64 = os.environ.get("JAX_ENABLE_X64", "true").lower() != "false"
        jax.config.update("jax_enable_x64", enable_x64)
        self._disable_x64 = not enable_x64

    @property
    def active_backend(self) -> str:
        return self._active_backend

    @property
    def intx(self):
        """
        Defines native int bit size (32 or 64), by default we want 64 bits.
        """
        return self.int32 if self._disable_x64 else self.int64

    def __getattr__(self, name):
        backend = self._backends[self.active_backend]
        # First look for the attribute in the backend in case the default is overridden
        if hasattr(backend, name):
            return getattr(self._backends[self.active_backend], name)
        # Else try to get it from the underlying lib
        if hasattr(backend.lib, name):
            return getattr(backend.lib, name)
        raise AttributeError(f"Backend {self.active_backend} has no attribute {name}")

    def function(self, name: str) -> Callable:
        # These seem to have to be wrapped in a function to avoid fixing
        # the backend at the time of definition.
        def _call(*args, **kwargs):
            return getattr(self, name)(*args, **kwargs)

        return _call

    @property
    def Rotation(self):
        return self._backends[self.active_backend].Rotation

    @property
    def ResultsDataImpl(self) -> AbstractResultsData:
        return self._backends[self.active_backend].ResultsDataImpl

# Global default backend context var
_active_dispatcher = contextvars.ContextVar(
    "active_dispatcher", default=MathDispatcher()
)

def get_dispatcher() -> MathDispatcher:
    return _active_dispatcher.get()

def set_backend(backend: str):
    """Change the numerical backend (JAX or numpy) for the current thread/context."""
    dispatcher = MathDispatcher(backend)
    _active_dispatcher.set(dispatcher)

# TODO: Do we need to be specific about which version of these constants gets used?
inf = np.inf
nan = np.nan

# Alias some core functions for convenience, retrieving the active dispatcher each time
def _dispatch_proxy(name):
    def proxy(*args, **kwargs):
        dispatcher = get_dispatcher()
        return dispatcher.function(name)(*args, **kwargs)
    return proxy

asarray = _dispatch_proxy("asarray")
array = _dispatch_proxy("array")
zeros = _dispatch_proxy("zeros")
zeros_like = _dispatch_proxy("zeros_like")
reshape = _dispatch_proxy("reshape")
cond = _dispatch_proxy("cond")
scan = _dispatch_proxy("scan")
while_loop = _dispatch_proxy("while_loop")
fori_loop = _dispatch_proxy("fori_loop")
jit = _dispatch_proxy("jit")
io_callback = _dispatch_proxy("io_callback")
pure_callback = _dispatch_proxy("pure_callback")
interp2d = _dispatch_proxy("interp2d")
ODESolver = _dispatch_proxy("ODESolver")
switch = _dispatch_proxy("switch")
stop_gradient = _dispatch_proxy("stop_gradient")

class _RotationProxy:
    def __getattr__(self, name):
        return getattr(get_dispatcher().Rotation, name)

    def __call__(self, *args, **kwargs):
        return get_dispatcher().Rotation(*args, **kwargs)

Rotation = _RotationProxy()


@register_pytree_node_class
class ResultsData:
    def __init__(self, solution_data: AbstractResultsData):
        self._solution_data = solution_data

    @property
    def time(self):
        return self._solution_data.time

    @property
    def outputs(self):
        return self._solution_data.outputs

    @staticmethod
    def initialize(*args, **kwargs) -> ResultsData:
        solution = get_dispatcher().ResultsDataImpl.initialize(*args, **kwargs)
        return ResultsData(solution)

    def update(self, *args, **kwargs) -> ResultsData:
        solution = self._solution_data.update(*args, **kwargs)
        return ResultsData(solution)

    def finalize(self):
        return self._solution_data.finalize()

    @classmethod
    def _scan(cls, *args, **kwargs):
        return scan(*args, **kwargs)

    def tree_flatten(self):
        return (self._solution_data,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (solution_data,) = children
        return ResultsData(solution_data)
