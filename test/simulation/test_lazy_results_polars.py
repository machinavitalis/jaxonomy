# SPDX-License-Identifier: MIT
"""
T-015a — polars LazyFrame backend for :class:`LazyResults`.

The polars backend is opt-in via :meth:`LazyResults.with_polars_backend`.
These tests verify:

  1. select+where round-trip matches the eager-numpy path.
  2. The plan stays lazy (LazyFrame) until terminal materialisation.
  3. ``to_parquet`` -> ``from_parquet`` round-trip preserves content.
  4. Callable ``where`` predicates fall back to eager-numpy with a warning.
  5. Default-off path is byte-equivalent to today's eager-numpy path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation import LazyResults
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()
pl = pytest.importorskip("polars")


def _make_results():
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


# ── 1) select + where match eager-numpy path ─────────────────────────────


def test_polars_backend_select_where():
    res = _make_results()
    eager = res.lazy().select("x").where("x > 0.5").collect()
    polars_df = res.lazy().with_polars_backend().select("x").where("x > 0.5").to_polars()
    np.testing.assert_allclose(
        polars_df["time"].to_numpy(), eager["time"], rtol=1e-12,
    )
    np.testing.assert_allclose(
        polars_df["x"].to_numpy(), eager["x"], rtol=1e-12,
    )


# ── 2) plan stays lazy until terminal call ───────────────────────────────


def test_polars_lazy_through_terminal():
    res = _make_results()
    chain = res.lazy().with_polars_backend().select("x").where("x > 0.0")
    # The chain itself stores ops, not a collected DataFrame.
    assert chain._use_polars is True
    # Build the LazyFrame and confirm it's lazy (LazyFrame, not DataFrame).
    lf = chain._build_lazyframe()
    for op in chain._ops:
        expanded = (
            lf.collect_schema().names()
            if hasattr(lf, "collect_schema")
            else lf.columns
        )
        lf = op.polars_apply(lf, expanded)
    assert isinstance(lf, pl.LazyFrame)
    plan = lf.explain(optimized=False)
    # FILTER must appear in the unoptimised plan; the where() pushed
    # it down lazily.
    assert "FILTER" in plan.upper()


# ── 3) parquet round-trip ────────────────────────────────────────────────


def test_polars_to_parquet_round_trip(tmp_path):
    res = _make_results()
    path = tmp_path / "out.parquet"
    res.lazy().with_polars_backend().select("x").to_parquet(path)
    assert path.exists() and path.stat().st_size > 0

    loaded = LazyResults.from_parquet(path)
    out = loaded.collect()
    eager = res.lazy().select("x").collect()
    np.testing.assert_allclose(out["time"], eager["time"], rtol=1e-12)
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)


def test_polars_to_parquet_batched(tmp_path):
    """``batch_size=N`` partitions the output into multiple files."""
    res = _make_results()
    n = len(res.time)
    if n < 4:
        pytest.skip("need at least 4 rows for a batched write")
    path = tmp_path / "chunked.parquet"
    res.lazy().with_polars_backend().to_parquet(path, batch_size=max(2, n // 3))
    chunks = sorted(tmp_path.glob("chunked.*.parquet"))
    assert len(chunks) >= 2
    total = sum(pl.read_parquet(p).height for p in chunks)
    assert total == n


# ── 4) callable predicate falls back ─────────────────────────────────────


def test_polars_callable_predicate_fallback():
    res = _make_results()
    chain = (
        res.lazy()
        .with_polars_backend()
        .where(lambda t, signals: t > 1.0)
    )
    with pytest.warns(RuntimeWarning, match="polars equivalent"):
        out = chain.collect()
    assert np.all(out["time"] > 1.0)


def test_polars_resample_native_correctness():
    """T-015a-followup: ``resample`` is now native polars (asof-join).

    Output must match the eager-numpy ``np.interp`` path within 1e-12,
    and no fallback ``RuntimeWarning`` should fire.
    """
    res = _make_results()
    t_uniform = np.linspace(0.0, 2.0, 13)
    eager = res.lazy().resample(t_uniform).collect()
    with warnings.catch_warnings():
        # Any RuntimeWarning here = regression: we'd be back on the
        # eager-numpy fallback path.
        warnings.simplefilter("error", RuntimeWarning)
        out = res.lazy().with_polars_backend().resample(t_uniform).collect()
    np.testing.assert_allclose(out["time"], eager["time"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["x"], eager["x"], rtol=0, atol=1e-12)


def test_polars_resample_no_fallback_warning():
    """Explicit guard: ``resample`` must not emit the polars-fallback warning."""
    res = _make_results()
    t_uniform = np.linspace(0.0, 2.0, 7)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        res.lazy().with_polars_backend().resample(t_uniform).collect()
    msgs = [str(w.message) for w in captured if issubclass(w.category, RuntimeWarning)]
    assert not any("polars equivalent" in m for m in msgs), (
        f"resample fell back to eager-numpy: {msgs!r}"
    )


def test_polars_resample_unsorted_target_preserves_order():
    """Non-monotonic ``t_new`` must come back in the user-supplied order
    (matches ``np.interp`` semantics)."""
    res = _make_results()
    t_unsorted = np.array([1.5, 0.5, 1.0, 0.1, 1.9, 0.3])
    eager = res.lazy().resample(t_unsorted).collect()
    out = res.lazy().with_polars_backend().resample(t_unsorted).collect()
    np.testing.assert_array_equal(out["time"], eager["time"])
    np.testing.assert_allclose(out["x"], eager["x"], rtol=0, atol=1e-12)


def test_polars_resample_chain_with_where():
    """``where(...).resample(...)`` and ``resample(...).where(...)`` both
    must match the eager path under the polars backend."""
    res = _make_results()
    t_uniform = np.linspace(0.0, 2.0, 31)
    eager_a = res.lazy().resample(t_uniform).where("x > 0.3").collect()
    out_a = (
        res.lazy().with_polars_backend().resample(t_uniform).where("x > 0.3").collect()
    )
    np.testing.assert_allclose(out_a["time"], eager_a["time"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out_a["x"], eager_a["x"], rtol=0, atol=1e-12)


def test_polars_resample_round_trip_via_parquet(tmp_path):
    """Out-of-core: write LazyResults to parquet, reload lazily, resample."""
    res = _make_results()
    path = tmp_path / "rt.parquet"
    res.lazy().with_polars_backend().to_parquet(path)
    loaded = LazyResults.from_parquet(path)
    t_uniform = np.linspace(0.0, 2.0, 9)
    eager = res.lazy().resample(t_uniform).collect()
    out = loaded.resample(t_uniform).collect()
    np.testing.assert_allclose(out["time"], eager["time"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["x"], eager["x"], rtol=0, atol=1e-12)


# ── 5) default-off byte-equivalent ───────────────────────────────────────


def test_polars_default_off_byte_equivalent():
    """Without ``with_polars_backend()``, the path is unchanged."""
    res = _make_results()
    a = res.lazy().select("x").where("x > 0.3").collect()
    b = res.lazy().select("x").where("x > 0.3").collect()
    np.testing.assert_array_equal(a["time"], b["time"])
    np.testing.assert_array_equal(a["x"], b["x"])
    # And the LazyResults instance itself does not have polars enabled.
    assert res.lazy()._use_polars is False


# ── chained ops survive a fallback midway ────────────────────────────────


def test_polars_chain_with_signal_then_select():
    """``with_signal`` collect+re-lazy + ``select`` afterwards."""
    res = _make_results()
    out = (
        res.lazy()
        .with_polars_backend()
        .with_signal("two_x", lambda t, s: 2 * s["x"])
        .select("two_x")
        .collect()
    )
    eager = res.lazy().with_signal("two_x", lambda t, s: 2 * s["x"]).select("two_x").collect()
    np.testing.assert_allclose(out["two_x"], eager["two_x"], rtol=1e-12)
