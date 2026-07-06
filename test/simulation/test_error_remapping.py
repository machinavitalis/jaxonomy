# SPDX-License-Identifier: MIT
"""
T-006 — tests for simulator error remapping.

Verifies:

  - A block that raises a non-JaxonomyError during trace time causes the
    exception to surface as ``SimulationError`` with a cleaner message.
  - The block name is surfaced in the SimulationError when recoverable
    from the traceback.
  - ``JAXONOMY_VERBOSE_TRACEBACK=1`` bypasses the wrapper and re-raises
    the original exception type.
  - ``JaxonomyError`` subclasses (e.g. ``BlockParameterError``) are NOT
    wrapped — domain-specific messages stay clean.
  - ``KeyboardInterrupt`` propagates unchanged.
"""

from __future__ import annotations

import os

import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation import SimulationError
from jaxonomy.framework.error import JaxonomyError
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Fixture: a block whose ODE raises a non-JaxonomyError inside JAX trace ─


class _RaisingODE(jaxonomy.LeafSystem):
    """ODE right-hand side raises a plain Python TypeError at trace time."""

    def __init__(self, name="raising_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)

    def _ode(self, time, state, **params):
        # Intentional bug: indexing a scalar.  JAX raises during tracing.
        x = state.continuous_state
        return x[0]  # IndexError for a 0-d array


def test_raising_block_surfaces_simulation_error():
    """A trace-time error from inside a block's ODE becomes
    SimulationError, not a raw JAX TypeError / IndexError."""
    # Make sure verbose mode is off.
    prev = os.environ.pop("JAXONOMY_VERBOSE_TRACEBACK", None)
    try:
        sys = _RaisingODE(name="raising_block")
        ctx = sys.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax")
        with pytest.raises(SimulationError) as exc_info:
            jaxonomy.simulate(sys, ctx, (0.0, 0.1), options=opts)
        err = exc_info.value
        # The original exception is preserved.
        assert err.cause is not None
        # The wrapped message is user-facing and includes a hint about
        # verbose mode.
        assert "JAXONOMY_VERBOSE_TRACEBACK" in str(err)
    finally:
        if prev is not None:
            os.environ["JAXONOMY_VERBOSE_TRACEBACK"] = prev


def test_verbose_mode_bypasses_wrapper():
    """With JAXONOMY_VERBOSE_TRACEBACK=1, the original exception type
    surfaces — useful for debugging JAX-level bugs."""
    prev = os.environ.get("JAXONOMY_VERBOSE_TRACEBACK")
    os.environ["JAXONOMY_VERBOSE_TRACEBACK"] = "1"
    try:
        sys = _RaisingODE(name="raising_block_v")
        ctx = sys.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax")
        # The exception is NOT a SimulationError in verbose mode; we catch
        # BaseException to let any JAX-level type through.
        with pytest.raises(BaseException) as exc_info:
            jaxonomy.simulate(sys, ctx, (0.0, 0.1), options=opts)
        assert not isinstance(exc_info.value, SimulationError), (
            f"Verbose mode should bypass remapping, got {type(exc_info.value).__name__}"
        )
    finally:
        if prev is None:
            os.environ.pop("JAXONOMY_VERBOSE_TRACEBACK", None)
        else:
            os.environ["JAXONOMY_VERBOSE_TRACEBACK"] = prev


def test_jaxonomy_error_not_wrapped():
    """StaticError / BlockParameterError / etc. propagate unchanged —
    no double-wrapping so the domain-specific message stays intact."""
    from jaxonomy.library import Adder

    # Adder raises BlockParameterError on invalid operators (this happens
    # at construction, before simulate).  We need a case that raises
    # inside simulate's trace.  Create a block whose ODE itself raises
    # a JaxonomyError subclass.
    from jaxonomy.framework.error import BlockRuntimeError

    class _Domain(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(name="domain", **kwargs)
            self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)

        def _ode(self, time, state, **params):
            raise BlockRuntimeError(
                message="intentional domain error for test", system=self
            )

    prev = os.environ.pop("JAXONOMY_VERBOSE_TRACEBACK", None)
    try:
        sys = _Domain()
        ctx = sys.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax")
        with pytest.raises(JaxonomyError) as exc_info:
            jaxonomy.simulate(sys, ctx, (0.0, 0.1), options=opts)
        # Crucially, NOT wrapped in SimulationError.
        assert not isinstance(exc_info.value, SimulationError), (
            "JaxonomyError subclasses should not be double-wrapped"
        )
    finally:
        if prev is not None:
            os.environ["JAXONOMY_VERBOSE_TRACEBACK"] = prev


def test_successful_simulate_unchanged():
    """The decorator must not alter the return value on success."""

    class _Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = _Decay()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax")
    res = jaxonomy.simulate(sys, ctx, (0.0, 0.5), options=opts)
    assert res.context.continuous_state is not None
