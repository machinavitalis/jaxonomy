# SPDX-License-Identifier: MIT
"""
T-015 — LazyResults fluent-API tests.

Covers:

  - Construction via SimulationResults.lazy()
  - select / where / resample / with_signal compose deferred
  - explain() shows the operation chain
  - Terminal collect / to_numpy materialises correctly
  - Backwards compatibility: SimulationResults.outputs[name] still works
  - to_pandas / to_polars / to_parquet (skip if optional dep missing)
  - Error paths: unknown signal, mask shape mismatch, out-of-range resample
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation import LazyResults, SimulationResults
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _make_results():
    """Run a tiny decay simulation and return its SimulationResults."""

    class _Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
            self.declare_continuous_state_output(name="x")

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = _Decay()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", save_time_series=True)
    return jaxonomy.simulate(
        sys, ctx, (0.0, 2.0), options=opts,
        recorded_signals={"x": sys.output_ports[0]},
    )


# ── construction & identity ────────────────────────────────────────────────


def test_lazy_factory():
    res = _make_results()
    lazy = res.lazy()
    assert isinstance(lazy, LazyResults)
    out = lazy.collect()
    np.testing.assert_array_equal(out["time"], np.asarray(res.time))
    np.testing.assert_array_equal(out["x"], np.asarray(res.outputs["x"]))


def test_backwards_compat_eager_outputs_unchanged():
    """The legacy results.outputs[name] access path is untouched."""
    res = _make_results()
    arr = res.outputs["x"]
    assert arr is res.outputs["x"]  # same object both times


def test_lazy_from_results_without_outputs_raises():
    res = SimulationResults(context=None, time=None, outputs=None)
    with pytest.raises(ValueError, match="no outputs"):
        LazyResults.from_results(res)


# ── select ────────────────────────────────────────────────────────────────


def test_select_subset():
    res = _make_results()
    out = res.lazy().select("x").collect()
    assert set(out) == {"time", "x"}


def test_select_unknown_signal_raises_at_collect():
    res = _make_results()
    chain = res.lazy().select("nope")
    with pytest.raises(KeyError, match="unknown signal"):
        chain.collect()


# ── where ────────────────────────────────────────────────────────────────


def test_where_array_mask():
    res = _make_results()
    n = len(res.time)
    mask = np.arange(n) >= n // 2
    out = res.lazy().where(mask).collect()
    assert len(out["time"]) == n - n // 2


def test_where_callable_mask():
    res = _make_results()
    out = res.lazy().where(lambda t, signals: t > 1.0).collect()
    assert np.all(out["time"] > 1.0)


def test_where_string_expression():
    res = _make_results()
    out = res.lazy().where("t > 1.0").collect()
    if len(out["time"]) > 0:
        assert np.all(out["time"] > 1.0)


def test_where_string_compound_expression():
    """Compound predicate via string — use np.logical_and to avoid the
    bitwise-and dtype gotcha when ``&`` operands are jnp arrays
    promoted from JAX-traced state."""
    res = _make_results()
    out = res.lazy().where("np.logical_and(t > 1.0, x < 0.5)").collect()
    if len(out["time"]) > 0:
        assert np.all(out["time"] > 1.0)
        assert np.all(out["x"] < 0.5)


def test_where_bad_shape_raises():
    res = _make_results()
    bad_mask = np.array([True, False, True])  # wrong length
    with pytest.raises(ValueError, match="shape"):
        res.lazy().where(bad_mask).collect()


# ── resample ─────────────────────────────────────────────────────────────


def test_resample_uniform_grid():
    res = _make_results()
    t_uniform = np.linspace(0.0, 2.0, 11)
    out = res.lazy().resample(t_uniform).collect()
    np.testing.assert_array_equal(out["time"], t_uniform)
    # Compare against analytic exp(-t).
    np.testing.assert_allclose(
        out["x"], np.exp(-t_uniform), atol=1e-2,
    )


def test_resample_after_where_raises_on_empty():
    res = _make_results()
    chain = res.lazy().where(lambda t, s: t > 100.0).resample(np.array([0.5]))
    with pytest.raises(ValueError, match="empty"):
        chain.collect()


def test_resample_out_of_range_raises():
    res = _make_results()
    with pytest.raises(ValueError, match="outside"):
        res.lazy().resample(np.array([5.0])).collect()


# ── with_signal ──────────────────────────────────────────────────────────


def test_with_signal_derives_new_column():
    res = _make_results()
    out = res.lazy().with_signal("x_squared", lambda t, s: s["x"] ** 2).collect()
    np.testing.assert_allclose(out["x_squared"], np.asarray(res.outputs["x"]) ** 2)


# ── chained operations stay lazy ─────────────────────────────────────────


def test_chain_explain():
    res = _make_results()
    chain = (
        res.lazy()
        .select("x")
        .where("t > 0.5")
        .with_signal("y", lambda t, s: 2 * s["x"])
    )
    explanation = chain.explain()
    assert "select" in explanation
    assert "where" in explanation
    assert "with_signal" in explanation


def test_chained_pipeline_value():
    res = _make_results()
    out = (
        res.lazy()
        .where("t > 0.5")
        .with_signal("two_x", lambda t, s: 2 * s["x"])
        .resample(np.linspace(0.6, 1.5, 5))
        .collect()
    )
    np.testing.assert_allclose(
        out["two_x"], 2 * np.exp(-np.asarray(out["time"])), atol=2e-2,
    )


# ── optional materialisation backends ────────────────────────────────────


def test_to_pandas():
    pd = pytest.importorskip("pandas")
    res = _make_results()
    df = res.lazy().select("x").to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["time", "x"]
    assert len(df) == len(res.time)


def test_to_parquet_via_pandas(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    res = _make_results()
    path = tmp_path / "out.parquet"
    res.lazy().select("x").to_parquet(path)
    assert path.exists() and path.stat().st_size > 0


def test_to_polars_skipped_if_unavailable():
    polars = pytest.importorskip("polars")
    res = _make_results()
    df = res.lazy().select("x").to_polars()
    assert isinstance(df, polars.DataFrame)
    assert df.columns == ["time", "x"]
