# SPDX-License-Identifier: MIT

"""Regression tests for T-008-followup-with-parameters-scalar-coerce and
T-008-followup-with-parameter-trace-cache.

Scalar coerce:
    Before the fix, the canonical sweep pattern
    ``diag.with_parameters({"osc.c": float(c)})`` raised
    ``AttributeError: 'float' object has no attribute 'shape'`` from
    ``_check_values_compatible`` on the JAX backend. Auto-promotion now
    succeeds and the value is stored as a JAX array.

Trace cache:
    Before the fix, ``ctx.with_parameter("k", 0.4)`` stored ``0.4`` as a
    Python float leaf in the context pytree. ``jax.jit`` then treats it as
    a concrete static argument and retraces on every distinct value, which
    makes a naive parameter sweep O(N) compilations. After the fix the
    parameter is coerced to ``jnp.asarray(0.4, dtype=...)`` and the compiled
    function reuses one cached trace across the sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import Gain


class _ScalarOsc(jaxonomy.LeafSystem):
    """Trivial first-order system ``x' = -c x`` with one dynamic parameter ``c``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_continuous_state(1, ode=self._ode)
        self.declare_dynamic_parameter("c", jnp.asarray(0.5))
        self.declare_continuous_state_output()

    def _ode(self, time, state, **params):
        return -params["c"] * state.continuous_state


def _build_diagram():
    builder = jaxonomy.DiagramBuilder()
    osc = builder.add(_ScalarOsc(name="osc"))
    gain = builder.add(Gain(2.0, name="gain"))
    builder.connect(osc.output_ports[0], gain.input_ports[0])
    builder.export_output(gain.output_ports[0])
    return builder.build()


def test_with_parameters_accepts_python_scalar_on_diagram():
    """``diag.with_parameters({"osc.c": 0.4})`` must not crash."""
    diag = _build_diagram()
    new = diag.with_parameters({"osc.c": 0.4})
    # The original is unchanged; the new one carries the new value.
    new_ctx = new.create_context()
    assert float(new_ctx[new.nodes[0].system_id].parameters["c"]) == pytest.approx(0.4)


def test_with_parameter_accepts_python_scalar_on_leaf_system():
    """``leaf.with_parameter("c", 0.4)`` must not crash either."""
    osc = _ScalarOsc(name="osc")
    new = osc.with_parameter("c", 0.4)
    ctx = new.create_context()
    assert float(ctx.parameters["c"]) == pytest.approx(0.4)


def test_with_parameter_on_context_accepts_python_scalar():
    osc = _ScalarOsc(name="osc")
    ctx = osc.create_context()
    new_ctx = ctx.with_parameter("c", 0.4)
    assert float(new_ctx.parameters["c"]) == pytest.approx(0.4)


def test_with_parameter_coerces_scalar_to_jnp_array():
    """The stored parameter must be a JAX array (not a Python float) so the
    context's pytree leaf type stays stable across sweep iterations.
    """
    osc = _ScalarOsc(name="osc")
    ctx = osc.create_context()
    new_ctx = ctx.with_parameter("c", 0.4)
    val = new_ctx.parameters["c"]
    assert isinstance(val, jax.Array), (
        f"expected jax.Array after coercion, got {type(val).__name__}"
    )


def test_with_parameter_coerces_numpy_scalar():
    osc = _ScalarOsc(name="osc")
    ctx = osc.create_context()
    new_ctx = ctx.with_parameter("c", np.float32(0.4))
    val = new_ctx.parameters["c"]
    assert isinstance(val, jax.Array)


def test_with_parameter_preserves_dtype():
    """Coercion must match the original parameter dtype, otherwise we lose
    the very pytree-stability that fixes the trace cache."""
    osc = _ScalarOsc(name="osc")
    ctx = osc.create_context()
    orig_dtype = ctx.parameters["c"].dtype
    new_ctx = ctx.with_parameter("c", 0.4)
    assert new_ctx.parameters["c"].dtype == orig_dtype


def test_with_parameter_sweep_reuses_jit_cache():
    """T-008 trace cache: a Python-float sweep should compile exactly once.

    We track recompiles by counting how often the inner function actually
    re-traces. JAX caches by abstract type/shape; after scalar coercion all
    iterations should share one cache entry.
    """
    osc = _ScalarOsc(name="osc")
    base_ctx = osc.create_context()

    trace_count = {"n": 0}

    @jax.jit
    def kernel(ctx):
        trace_count["n"] += 1
        # Trivial computation that touches the parameter.
        return ctx.parameters["c"] * 2.0

    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        out = kernel(base_ctx.with_parameter("c", float(c)))
        assert float(out) == pytest.approx(2.0 * c)

    # One compile for the whole sweep — not five.
    assert trace_count["n"] == 1, (
        f"expected 1 trace across the sweep, got {trace_count['n']}"
    )


def test_with_parameters_diagram_error_names_offending_key():
    """When validation does fail (shape mismatch), the error must name the
    parameter so the user knows which entry to fix."""
    diag = _build_diagram()
    with pytest.raises(ValueError, match="osc.c|'c'"):
        # Wrong shape: 1-D vector for a scalar parameter.
        diag.with_parameters({"osc.c": jnp.zeros(3)})


def test_with_parameter_leaf_error_names_offending_key():
    osc = _ScalarOsc(name="osc")
    with pytest.raises(ValueError, match="'c'"):
        osc.with_parameter("c", jnp.zeros(3))
