# SPDX-License-Identifier: MIT
"""
T-015a-followup-resample-pushdown-duckdb — DuckDB SQL backend for
:class:`LazyResults`.

The DuckDB backend is opt-in via :meth:`LazyResults.with_duckdb_backend`.
These tests verify:

  1. ``select`` correctness vs the eager-numpy path.
  2. ``where`` with a string predicate translates to SQL and matches
     the eager-numpy filter.
  3. ``to_parquet`` (DuckDB COPY) round-trips byte-equivalently.
  4. Callable ``where`` predicate falls back with a ``RuntimeWarning``.
  5. ``resample`` falls back with a ``RuntimeWarning`` (DuckDB resample
     was deliberately not implemented).
  6. Default-off path is byte-equivalent — pure numpy backend ignores
     DuckDB.
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
duckdb = pytest.importorskip("duckdb")


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


# ── 1) select correctness ────────────────────────────────────────────────


def test_duckdb_select_correctness():
    """``.with_duckdb_backend().select('x').collect()`` must match eager."""
    res = _make_results()
    eager = res.lazy().select("x").collect()
    out = res.lazy().with_duckdb_backend().select("x").collect()
    np.testing.assert_array_equal(out["time"], eager["time"])
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12, atol=0)
    assert set(out) == {"time", "x"}


def test_duckdb_user_supplied_connection():
    """A user-supplied connection is honoured (lets users configure threads,
    extensions, persistence)."""
    res = _make_results()
    conn = duckdb.connect()
    chain = res.lazy().with_duckdb_backend(connection=conn)
    assert chain._duckdb_conn is conn
    out = chain.select("x").collect()
    eager = res.lazy().select("x").collect()
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)


# ── 2) where string predicate ────────────────────────────────────────────


def test_duckdb_where_string_predicate():
    """``where('time > 1.0')`` translates to SQL and matches eager-numpy."""
    res = _make_results()
    eager = res.lazy().where("t > 1.0").collect()
    out = res.lazy().with_duckdb_backend().where("t > 1.0").collect()
    np.testing.assert_array_equal(out["time"], eager["time"])
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)
    assert np.all(out["time"] > 1.0)


def test_duckdb_where_compound_predicate():
    """Compound predicates with ``&`` map to SQL ``AND``."""
    res = _make_results()
    eager = res.lazy().where("(t > 0.5) & (x < 0.9)").collect()
    out = res.lazy().with_duckdb_backend().where("(t > 0.5) & (x < 0.9)").collect()
    np.testing.assert_array_equal(out["time"], eager["time"])
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)


# ── 3) parquet round-trip ────────────────────────────────────────────────


def test_duckdb_to_parquet_round_trip(tmp_path):
    """``COPY (sql) TO 'path' (FORMAT PARQUET)`` round-trips correctly."""
    res = _make_results()
    path = tmp_path / "out.parquet"
    res.lazy().with_duckdb_backend().select("x").to_parquet(path)
    assert path.exists() and path.stat().st_size > 0

    # Round-trip via the DuckDB ``from_parquet`` entry point.
    loaded = LazyResults.from_parquet(path, backend="duckdb")
    out = loaded.collect()
    eager = res.lazy().select("x").collect()
    np.testing.assert_allclose(out["time"], eager["time"], rtol=1e-12)
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)


def test_duckdb_to_parquet_filtered_round_trip(tmp_path):
    """A filtered DuckDB chain writes only the surviving rows."""
    res = _make_results()
    path = tmp_path / "filtered.parquet"
    res.lazy().with_duckdb_backend().where("t > 1.0").to_parquet(path)
    loaded = LazyResults.from_parquet(path, backend="duckdb")
    out = loaded.collect()
    eager = res.lazy().where("t > 1.0").collect()
    np.testing.assert_array_equal(out["time"], eager["time"])
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)


# ── 4) callable predicate falls back ─────────────────────────────────────


def test_duckdb_callable_predicate_fallback():
    """A callable ``where`` predicate fires ``RuntimeWarning`` and produces
    the same result as eager-numpy."""
    res = _make_results()
    chain = (
        res.lazy()
        .with_duckdb_backend()
        .where(lambda t, signals: t > 1.0)
    )
    with pytest.warns(RuntimeWarning, match="DuckDB SQL equivalent"):
        out = chain.collect()
    eager = res.lazy().where(lambda t, signals: t > 1.0).collect()
    np.testing.assert_array_equal(out["time"], eager["time"])
    np.testing.assert_allclose(out["x"], eager["x"], rtol=1e-12)
    assert np.all(out["time"] > 1.0)


# ── 5) resample falls back ───────────────────────────────────────────────


def test_duckdb_resample_fallback():
    """``resample`` is deliberately not implemented in SQL — falls back
    with a ``RuntimeWarning``; numerical result still matches eager."""
    res = _make_results()
    t_uniform = np.linspace(0.0, 2.0, 11)
    eager = res.lazy().resample(t_uniform).collect()
    chain = res.lazy().with_duckdb_backend().resample(t_uniform)
    with pytest.warns(RuntimeWarning, match="DuckDB SQL equivalent"):
        out = chain.collect()
    np.testing.assert_allclose(out["time"], eager["time"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["x"], eager["x"], rtol=0, atol=1e-12)


# ── 6) default-off byte-equivalent ───────────────────────────────────────


def test_duckdb_default_off_byte_equivalent():
    """Without ``with_duckdb_backend()``, the path is unchanged."""
    res = _make_results()
    a = res.lazy().select("x").where("x > 0.3").collect()
    b = res.lazy().select("x").where("x > 0.3").collect()
    np.testing.assert_array_equal(a["time"], b["time"])
    np.testing.assert_array_equal(a["x"], b["x"])
    assert res.lazy()._use_duckdb is False
    assert res.lazy()._duckdb_conn is None


# ── extras: chain interop ────────────────────────────────────────────────


def test_duckdb_chain_select_then_fallback_then_select():
    """select (SQL) -> resample (fallback) -> select (SQL): the chain
    re-enters SQL after a mid-chain fallback."""
    res = _make_results()
    t_uniform = np.linspace(0.0, 2.0, 7)
    eager = (
        res.lazy()
        .select("x")
        .resample(t_uniform)
        .collect()
    )
    chain = (
        res.lazy()
        .with_duckdb_backend()
        .select("x")
        .resample(t_uniform)
    )
    with pytest.warns(RuntimeWarning, match="DuckDB SQL equivalent"):
        out = chain.collect()
    np.testing.assert_allclose(out["time"], eager["time"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["x"], eager["x"], rtol=0, atol=1e-12)


def test_duckdb_to_pandas_works():
    """``.to_pandas()`` returns a DataFrame from the DuckDB backend."""
    pd = pytest.importorskip("pandas")
    res = _make_results()
    df = res.lazy().with_duckdb_backend().select("x").where("t > 1.0").to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns and "x" in df.columns
    assert (df["time"] > 1.0).all()


def test_duckdb_to_polars_works():
    """``.to_polars()`` returns a polars DataFrame from the DuckDB backend."""
    pl = pytest.importorskip("polars")
    res = _make_results()
    df = res.lazy().with_duckdb_backend().select("x").where("t > 1.0").to_polars()
    assert isinstance(df, pl.DataFrame)
    assert "time" in df.columns and "x" in df.columns
    assert (df["time"].to_numpy() > 1.0).all()


def test_duckdb_explain_tag():
    """The ``explain()`` debug helper marks the chain with ``[duckdb]``."""
    res = _make_results()
    explain = (
        res.lazy()
        .with_duckdb_backend()
        .select("x")
        .where("t > 0.5")
        .explain()
    )
    assert explain.startswith("[duckdb]")
