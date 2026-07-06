# SPDX-License-Identifier: MIT
"""
T-012 — SimulationResults.query(t) interpolation tests.

The solver's native dense interpolant would require persisting the
solver state through the recording pipeline; T-012a tracks that
upgrade.  The current implementation uses linear interpolation over
the stored time/value arrays, which is adequate for the common
"sample this signal at my preferred times" post-hoc workflow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


class _Decay(jaxonomy.LeafSystem):
    """dx/dt = -x, x(0) = 1 → x(t) = exp(-t)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        return -state.continuous_state


def _run_decay():
    sys = _Decay()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", save_time_series=True,
    )
    return jaxonomy.simulate(
        sys, ctx, (0.0, 2.0), options=opts,
        recorded_signals={"x": sys.output_ports[0]},
    )


def test_query_scalar_returns_scalar():
    res = _run_decay()
    v = res.query(1.0, signal="x")
    # Linear interp between adjacent recorded points — close to exp(-1).
    assert abs(float(v) - math.exp(-1.0)) < 5e-3


def test_query_array_returns_array():
    res = _run_decay()
    ts = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0])
    vs = res.query(ts, signal="x")
    # Linear interp over Dopri5's adaptive grid gives ~1 % error on a
    # decaying exponential; documented in test/simulation/test_results_query
    # docstring and T-012a follow-up (native-interpolant upgrade).
    np.testing.assert_allclose(
        np.asarray(vs),
        np.exp(-np.asarray(ts)),
        atol=1e-2,
    )


def test_query_returns_all_signals_when_signal_is_none():
    res = _run_decay()
    d = res.query(0.5)
    assert set(d) == {"x"}
    assert abs(float(d["x"]) - math.exp(-0.5)) < 5e-3


def test_query_endpoints_exact():
    """At the recorded endpoints, query returns the stored value exactly."""
    res = _run_decay()
    t0 = float(res.time[0])
    tf = float(res.time[-1])
    v0 = res.query(t0, signal="x")
    vf = res.query(tf, signal="x")
    assert abs(float(v0) - float(res.outputs["x"][0])) < 1e-12
    assert abs(float(vf) - float(res.outputs["x"][-1])) < 1e-12


def test_query_out_of_range_raises():
    res = _run_decay()
    with pytest.raises(ValueError, match="out of range"):
        res.query(5.0, signal="x")
    with pytest.raises(ValueError, match="out of range"):
        res.query(-0.5, signal="x")


def test_query_unknown_signal_raises():
    res = _run_decay()
    with pytest.raises(ValueError, match="unknown signal"):
        res.query(0.5, signal="not_a_signal")


def test_query_without_recorded_signals_raises():
    """If the user didn't pass recorded_signals to simulate, query must fail
    loudly rather than return silently wrong data."""
    sys = _Decay()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax")
    res = jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts)
    with pytest.raises(ValueError, match="no recorded signals"):
        res.query(0.5)
