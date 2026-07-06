# SPDX-License-Identifier: MIT

"""Internal helpers shared by all primitive category modules.

These were originally module-level helpers in ``primitives.py``. They are
imported by every split module and re-exported via the ``primitives.py``
re-export hub for backward compatibility.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..logging import logger
from ..framework.error import ErrorCollector
from ..framework import (
    LeafSystem,
    ShapeMismatchError,
    DtypeMismatchError,
    DependencyTicket,
)
from ..backend import numpy_api as npa
from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    from jax import lax as jax_lax
    from ..framework.port import OutputPort
    from ..backend.typing import Array
else:
    jax_lax = LazyLoader("jax_lax", globals(), "jax.lax")




def _stop_gradient(x):
    """Backend-aware ``lax.stop_gradient``.

    Under the JAX backend this delegates to ``jax.lax.stop_gradient`` so
    JAX's autodiff machinery treats ``x`` as a constant. Under the numpy
    backend (or for any non-JAX array) this is the identity, preserving
    the input dtype/shape exactly so byte-equivalence with the legacy
    numpy code-path is maintained.
    """
    # Avoid an eager jax import on the numpy backend by checking the
    # array's module first.  jax tracers and DeviceArrays both live in
    # the ``jax`` package; numpy arrays do not.
    mod = type(x).__module__
    if mod.startswith("jax") or mod.startswith("jaxlib"):
        return jax_lax.stop_gradient(x)
    return x



def check_state_type(
    sys: LeafSystem,
    inp_data: Array,
    state_data: Array,
    error_collector: ErrorCollector = None,
) -> None:
    """Check that the state type of a block matches the type of an input port."""
    inp_data = npa.asarray(inp_data)
    state_data = npa.asarray(state_data)

    with ErrorCollector.context(error_collector):
        if inp_data.shape != state_data.shape:
            logger.debug(
                "System %s shape mismatch, %s != %s",
                sys.system_id,
                inp_data.shape,
                state_data.shape,
            )
            raise ShapeMismatchError(
                system=sys,
                expected_shape=state_data.shape,
                actual_shape=inp_data.shape,
            )
        if inp_data.dtype != state_data.dtype:
            logger.debug(
                "System %s dtype mismatch, %s != %s",
                sys.system_id,
                inp_data.dtype,
                state_data.dtype,
            )
            raise DtypeMismatchError(
                system=sys,
                expected_dtype=state_data.dtype,
                actual_dtype=inp_data.dtype,
            )


def is_discontinuity(port: OutputPort) -> bool:
    """Does this signal represent a discontinuous input to an ODE?"""
    signal_is_continuous = port.tracker.depends_on(
        [DependencyTicket.time, DependencyTicket.xc]
    )
    if not signal_is_continuous:
        return False
    port_is_ode_rhs = port.tracker.is_prerequisite_of([DependencyTicket.xcdot])
    return signal_is_continuous and port_is_ode_rhs
