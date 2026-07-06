# SPDX-License-Identifier: MIT
"""
Lazy results query layer (T-015).

``LazyResults`` wraps a :class:`SimulationResults` (or any duck-typed
``time + outputs`` pair) and exposes a fluent operation API:

  - ``.where(mask)`` / ``.where("t > 5")`` — boolean filter on rows.
  - ``.select(*signals)`` — pick a subset of signals.
  - ``.resample(t)`` — interpolate every signal onto a new time grid
    (uses :meth:`SimulationResults.align`).
  - ``.with_signal(name, fn)`` — derive a new signal from existing ones.

Operations are deferred — they build up a chain of pending steps and
only run when a terminal materialiser is called:

  - ``.collect()`` → ``dict[str, np.ndarray]`` plus a ``time`` array.
  - ``.to_numpy()`` → same as collect, just clearer name.
  - ``.to_pandas()`` → ``pandas.DataFrame`` (optional dep).
  - ``.to_polars()`` → ``polars.DataFrame`` (optional dep).
  - ``.to_parquet(path)`` → write Parquet via pandas (or polars).
  - ``.to_hdf5(path, chunk_size=...)`` → chunk-stream the materialised
    frame to an HDF5 file via h5py (T-108-followup-streaming-export;
    optional dep).
  - ``.to_zarr(path, chunk_size=...)`` → chunk-stream to a zarr store
    (T-108-followup-streaming-export; optional dep).

Backwards compatibility: ``SimulationResults.outputs[name]`` is
untouched.  ``SimulationResults.lazy()`` returns a fresh
``LazyResults``; chaining is only used when the caller asks for it.

Design notes (T-015):

  - The spec mentions DuckDB and Polars as candidate backends.  The
    initial implementation uses NumPy as the eager backbone; Polars /
    pandas are optional materialisation targets via ``importorskip``.
    The fluent API is generic enough that swapping in a Polars-native
    pipeline (``polars.LazyFrame``) is a follow-up if a user needs
    out-of-core operations.
  - Parquet support routes through pandas to keep the default install
    light; users with a polars install get the polars path
    automatically.

Polars LazyFrame backend (T-015a):

  - Opt in via :meth:`LazyResults.with_polars_backend()`.  Default
    OFF: the eager-numpy path is preserved bit-for-bit.
  - When enabled, terminal materialisation builds a ``polars.LazyFrame``
    from the recorded ``(time, outputs)`` arrays and translates each
    pending op into a ``LazyFrame`` operation so polars can apply
    column pruning / predicate pushdown / streaming sinks.
  - Vector-valued signals (shape ``(T, k)`` with ``k > 1``) are
    expanded into ``name__0``, ``name__1`` columns to match the
    ``to_pandas`` / ``to_polars`` convention.
  - Op support:
      * ``select`` and ``with_signal`` translate cleanly.
      * ``where`` with a string expression goes through
        :func:`polars.sql_expr`; array masks become
        ``pl.Series`` filters.
      * ``resample`` is native polars as of
        T-015a-followup-resample-pushdown — see
        :func:`_polars_resample`.  Two ``join_asof`` calls
        (``backward`` + ``forward``) feed a linear-interp expression;
        no Python ``map_batches`` callback is involved, so the
        optimiser can fuse a prior ``where`` filter into the resample
        plan and ``sink_parquet`` can stream the whole chain.
      * ``where`` with a *callable* predicate is the only remaining
        op that cannot be expressed natively; on that op the chain
        emits a :class:`RuntimeWarning` and falls back to the eager-
        numpy path for that step (the rest of the chain still runs
        through polars where it can).
  - ``to_parquet`` uses ``LazyFrame.sink_parquet`` for true streaming
    writes when no ``batch_size`` is given.
  - :meth:`LazyResults.from_parquet` reads a written file back lazily
    via :func:`polars.scan_parquet` — the "results larger than RAM"
    round-trip path.

DuckDB SQL backend (T-015a-followup-resample-pushdown-duckdb):

  - Opt in via :meth:`LazyResults.with_duckdb_backend` (optionally
    passing a configured :class:`duckdb.DuckDBPyConnection`; default
    is a fresh in-memory connection).  Default OFF: the eager-numpy
    path is preserved bit-for-bit; non-DuckDB users do not need
    duckdb installed.
  - When enabled, terminal materialisers translate each pending op
    into SQL fragments and execute one combined query per chain:
      * ``select(*signals)`` → ``SELECT time, sig1, sig2, ...``
        (vector-valued signals become their ``name__i`` columns).
      * ``where(string)`` → ``WHERE <predicate>``; the user-provided
        Python-style expression is lightly munged (``t`` → ``time``,
        ``&``/``|`` → ``AND``/``OR``) before handing to DuckDB.
        Boolean-array masks are joined as a synthetic mask column.
      * ``with_signal`` (arbitrary Python callable), callable
        ``where`` predicates, and ``resample`` are *not* SQL-able
        in general and emit a :class:`RuntimeWarning`, fall back to
        the eager-numpy path for that op, then re-register a fresh
        in-memory table for the rest of the chain.
  - Terminals:
      * ``.collect()`` / ``.to_numpy()`` — `connection.execute(sql)
        .fetchnumpy()`.
      * ``.to_pandas()`` — `.df()`.
      * ``.to_polars()`` — `.pl()` if pyarrow is available, else fall
        back to materialising via numpy then handing to polars.
      * ``.to_parquet(path)`` — `COPY (sql) TO 'path' (FORMAT PARQUET)`,
        which is genuinely streaming on DuckDB's side.
  - ``LazyResults.from_parquet(path, backend="duckdb")`` reads the file
    via ``read_parquet(...)`` against a fresh in-memory connection —
    the out-of-core entry point.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .types import SimulationResults


__all__ = ["LazyResults"]


@dataclass
class _Op:
    name: str
    apply: Callable[[dict, np.ndarray], tuple[dict, np.ndarray]]
    # Optional polars-LazyFrame translator (T-015a).  Receives the
    # current ``polars.LazyFrame`` and the list of "vector-expanded"
    # column names that come from non-time signals; returns the new
    # LazyFrame.  ``None`` means "this op has no polars equivalent;
    # fall back to the eager-numpy path for this step".
    polars_apply: Optional[Callable[[Any, list[str]], Any]] = None
    # Optional DuckDB SQL translator (T-015a-followup-resample-
    # pushdown-duckdb).  Receives the current ``_DuckDBPlan`` (a
    # subquery + column list pair) and returns the new plan.  ``None``
    # means "this op has no SQL equivalent; fall back to eager-numpy
    # for this step, then re-register".
    duckdb_apply: Optional[Callable[[Any, list[str]], Any]] = None


@dataclass
class LazyResults:
    """A deferred-evaluation wrapper around a :class:`SimulationResults`.

    Construct via :meth:`SimulationResults.lazy` rather than directly.
    """

    _outputs: dict[str, np.ndarray]
    _time: np.ndarray
    _ops: list[_Op] = field(default_factory=list)
    _use_polars: bool = False
    _use_duckdb: bool = False
    # Stored as ``Any`` to avoid a hard import of duckdb at module load.
    _duckdb_conn: Optional[Any] = None
    # T-108 phase 1: per-signal native sample-time vectors carried
    # over from :class:`SimulationResults.per_signal_times` (T-013 / T-013a).
    # ``None`` means "every signal shares ``self._time``" — the legacy
    # default that keeps the fluent API byte-equivalent.
    _per_signal_times: Optional[dict[str, np.ndarray]] = None

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_results(cls, results: "SimulationResults") -> "LazyResults":
        if results.outputs is None:
            raise ValueError(
                "LazyResults: SimulationResults has no outputs.  Pass "
                "recorded_signals= to simulate() first."
            )
        outputs = {k: np.asarray(v) for k, v in results.outputs.items()}
        time = np.asarray(results.time) if results.time is not None else np.zeros(0)
        per_signal_times: Optional[dict[str, np.ndarray]] = None
        if getattr(results, "per_signal_times", None) is not None:
            per_signal_times = {
                k: np.asarray(v) for k, v in results.per_signal_times.items()
            }
        return cls(_outputs=outputs, _time=time, _per_signal_times=per_signal_times)

    @classmethod
    def from_parquet(cls, path, backend: str = "polars") -> "LazyResults":
        """Load a parquet file written by :meth:`to_parquet`.

        Parameters
        ----------
        path
            Path to a parquet file produced by :meth:`to_parquet`
            (or any parquet file with a ``time`` column).
        backend
            ``"polars"`` (default; T-015a) returns a :class:`LazyResults`
            with the polars backend pre-enabled.
            ``"duckdb"`` (T-015a-followup-resample-pushdown-duckdb)
            opens the file via DuckDB's ``read_parquet(...)`` against a
            fresh in-memory connection — the out-of-core entry point
            for SQL-style queries.  In both cases vector-valued signals
            stored as ``name__i`` columns are re-collapsed into ``(T, k)``
            numpy arrays for compatibility with the eager-numpy fallback
            path.
        """
        if backend not in {"polars", "duckdb"}:
            raise ValueError(
                f"LazyResults.from_parquet: unknown backend {backend!r}; "
                f"expected 'polars' or 'duckdb'."
            )
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "LazyResults.from_parquet requires polars.  "
                "Install with `pip install polars`."
            ) from e

        df = pl.read_parquet(str(path))
        cols = df.columns
        if "time" not in cols:
            raise ValueError(
                f"LazyResults.from_parquet: file {path!r} has no 'time' column "
                f"(columns={cols})."
            )
        time = np.asarray(df["time"].to_numpy())
        # Re-collapse name__i columns into vector-valued arrays.
        outputs: dict[str, np.ndarray] = {}
        groups: dict[str, dict[int, str]] = {}
        scalars: list[str] = []
        for c in cols:
            if c == "time":
                continue
            if "__" in c:
                base, _, idx_str = c.rpartition("__")
                try:
                    idx = int(idx_str)
                except ValueError:
                    scalars.append(c)
                    continue
                groups.setdefault(base, {})[idx] = c
            else:
                scalars.append(c)
        for c in scalars:
            outputs[c] = np.asarray(df[c].to_numpy())
        for base, idx_map in groups.items():
            ordered = [df[idx_map[i]].to_numpy() for i in sorted(idx_map)]
            outputs[base] = np.stack([np.asarray(a) for a in ordered], axis=-1)
        if backend == "duckdb":
            try:
                import duckdb  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "LazyResults.from_parquet(backend='duckdb') requires "
                    "duckdb.  Install with `pip install duckdb`."
                ) from e
            conn = duckdb.connect()
            return cls(
                _outputs=outputs,
                _time=time,
                _use_duckdb=True,
                _duckdb_conn=conn,
            )
        return cls(_outputs=outputs, _time=time, _use_polars=True)

    # ── backend opt-in (T-015a) ──────────────────────────────────────────

    def with_polars_backend(self) -> "LazyResults":
        """Opt in to the polars LazyFrame execution path (T-015a).

        Returns a copy of this :class:`LazyResults` whose terminal
        materialisers (``to_polars``/``to_pandas``/``to_parquet``/
        ``to_numpy``/``collect``) build a ``polars.LazyFrame`` plan
        rather than evaluating ops eagerly on numpy arrays.

        Falls back to eager-numpy on a per-op basis (with
        :class:`RuntimeWarning`) for ops that polars cannot express
        natively — currently only callable ``where`` predicates.
        ``resample`` is native polars (asof-join + linear-interp
        expression; T-015a-followup-resample-pushdown).
        ``with_signal`` is executed via collect-and-re-lazy.
        """
        return LazyResults(
            _outputs=self._outputs,
            _time=self._time,
            _ops=list(self._ops),
            _use_polars=True,
            _per_signal_times=self._per_signal_times,
        )

    def with_duckdb_backend(self, connection=None) -> "LazyResults":
        """Opt in to the DuckDB SQL execution path (T-015a-followup-...-duckdb).

        Parameters
        ----------
        connection
            An existing :class:`duckdb.DuckDBPyConnection`, or ``None``
            (default) to allocate a fresh in-memory connection.  Pass
            an explicit connection to control persistence, extension
            loading, or thread count.

        Returns a copy of this :class:`LazyResults` whose terminal
        materialisers run a single SQL query against an in-memory
        DuckDB table built from the recorded ``(time, outputs)``
        arrays.  Vector-valued signals are exposed as ``name__i``
        columns (matching the polars backend convention).

        Per-op fallback: ``with_signal``, callable ``where`` predicates,
        and ``resample`` are not generally SQL-able and emit
        :class:`RuntimeWarning` at materialise time, falling back to
        the eager-numpy path for that op (the chain re-enters DuckDB
        afterwards).  ``select`` and ``where`` with a string predicate
        translate cleanly.
        """
        if connection is None:
            try:
                import duckdb
            except ImportError as e:
                raise ImportError(
                    "LazyResults.with_duckdb_backend: duckdb is not "
                    "installed.  Install with `pip install duckdb`."
                ) from e
            connection = duckdb.connect()
        return LazyResults(
            _outputs=self._outputs,
            _time=self._time,
            _ops=list(self._ops),
            _use_duckdb=True,
            _duckdb_conn=connection,
            _per_signal_times=self._per_signal_times,
        )

    # ── lazy operations ──────────────────────────────────────────────────

    def _chain(self, op: _Op) -> "LazyResults":
        return LazyResults(
            _outputs=self._outputs,
            _time=self._time,
            _ops=self._ops + [op],
            _use_polars=self._use_polars,
            _use_duckdb=self._use_duckdb,
            _duckdb_conn=self._duckdb_conn,
            _per_signal_times=self._per_signal_times,
        )

    def select(self, *signals: str) -> "LazyResults":
        """Project to a subset of signals (defers)."""
        names = list(signals)

        def _apply(out: dict, t: np.ndarray):
            missing = [s for s in names if s not in out]
            if missing:
                raise KeyError(
                    f"LazyResults.select: unknown signal(s) {missing!r}.  "
                    f"Available: {list(out)}"
                )
            return {s: out[s] for s in names}, t

        def _polars_apply(lf, expanded_cols):
            # Keep "time" plus every expanded variant of the requested
            # signals (e.g. select("v") keeps both v__0 and v__1 for a
            # vector-valued v).
            keep = ["time"]
            existing = lf.collect_schema().names() if hasattr(lf, "collect_schema") else lf.columns
            for s in names:
                hit = [c for c in existing if c == s or c.startswith(f"{s}__")]
                if not hit:
                    raise KeyError(
                        f"LazyResults.select: unknown signal {s!r}.  "
                        f"Available: {[c for c in existing if c != 'time']}"
                    )
                keep.extend(hit)
            return lf.select(keep)

        def _duckdb_apply(plan, expanded_cols):
            keep = ["time"]
            existing = list(expanded_cols)
            for s in names:
                hit = [c for c in existing if c == s or c.startswith(f"{s}__")]
                if not hit:
                    raise KeyError(
                        f"LazyResults.select: unknown signal {s!r}.  "
                        f"Available: {[c for c in existing if c != 'time']}"
                    )
                keep.extend(hit)
            return plan.with_select(keep)

        return self._chain(
            _Op(
                name=f"select{tuple(names)!r}",
                apply=_apply,
                polars_apply=_polars_apply,
                duckdb_apply=_duckdb_apply,
            )
        )

    def where(self, mask) -> "LazyResults":
        """Boolean-mask filter on rows (defers).

        ``mask`` may be:
          - a boolean numpy array of length ``len(time)``;
          - a callable ``f(t, outputs) -> bool array``;
          - a string expression that uses ``t`` and any signal name as
            free variables (e.g. ``"t > 5"``, ``"x > 0 & t < 1.5"``).
        """

        def _apply(out: dict, t: np.ndarray):
            if callable(mask):
                m = mask(t, out)
            elif isinstance(mask, str):
                # Restrict to a known-safe globals dict.  Each signal is
                # available by name; ``t`` is the time vector.
                env = {"t": t, **out, "np": np}
                m = eval(mask, {"__builtins__": {}}, env)  # noqa: S307
            else:
                m = mask
            m = np.asarray(m, dtype=bool)
            if m.shape != t.shape:
                raise ValueError(
                    f"LazyResults.where: mask shape {m.shape} does not match "
                    f"time shape {t.shape}."
                )
            new_out = {k: v[m] if v.ndim == 1 else v[m, ...] for k, v in out.items()}
            return new_out, t[m]

        # Polars translator — only available for string expressions and
        # boolean-array masks.  Callable predicates fall back to eager-
        # numpy with a RuntimeWarning at materialise time.
        polars_apply: Optional[Callable[[Any, list[str]], Any]]
        duckdb_apply: Optional[Callable[[Any, list[str]], Any]]
        if callable(mask):
            polars_apply = None
            duckdb_apply = None
        elif isinstance(mask, str):
            expr_str = mask

            def polars_apply(lf, expanded_cols, _expr_str=expr_str):  # type: ignore[misc]
                import re

                import polars as pl

                # ``t`` is the time column in polars-land. Use a word-boundary
                # substitution so signal names that merely end in 't' (e.g.
                # ``out``, ``count``) are not mangled — matches the duckdb
                # path's _python_predicate_to_sql.
                sql_expr = re.sub(r"\bt\b", "time", _expr_str)
                # Common case: simple "x > 0.5" — let polars.sql_expr
                # handle it.  Raise a clear error on failure.
                try:
                    return lf.filter(pl.sql_expr(sql_expr))
                except Exception:
                    raise RuntimeError(
                        f"LazyResults.where: polars cannot translate "
                        f"expression {_expr_str!r}; use a numpy mask or "
                        f"omit .with_polars_backend()."
                    )

            def duckdb_apply(plan, expanded_cols, _expr_str=expr_str):  # type: ignore[misc]
                # Translate Python-style operators in the expression to
                # SQL: ``&`` / ``|`` -> ``AND`` / ``OR``; lone ``t`` -> ``time``.
                # We deliberately do not try to be exhaustive — DuckDB's
                # SQL parser already accepts ``>``, ``<``, ``>=``, ``<=``,
                # ``==`` (folded to ``=``), ``!=``, ``+``, ``-``, ``*``,
                # ``/`` directly.
                sql_expr = _python_predicate_to_sql(_expr_str)
                return plan.with_where(sql_expr)
        else:
            mask_arr = np.asarray(mask, dtype=bool)

            def polars_apply(lf, expanded_cols, _mask=mask_arr):  # type: ignore[misc]
                import polars as pl

                return lf.filter(pl.Series("__mask__", _mask))

            def duckdb_apply(plan, expanded_cols, _mask=mask_arr):  # type: ignore[misc]
                # Boolean-array masks: register a row-aligned mask
                # column on the connection and AND it into the WHERE.
                return plan.with_mask_array(_mask)

        return self._chain(
            _Op(
                name=f"where({mask!r})",
                apply=_apply,
                polars_apply=polars_apply,
                duckdb_apply=duckdb_apply,
            )
        )

    def resample(
        self,
        t_new,
        *,
        method: str = "linear",
    ) -> "LazyResults":
        """Interpolate every signal onto ``t_new`` (defers).

        T-108 phase 2 wires the optional ``method=`` kwarg through to
        the T-106 backend (:func:`jaxonomy.library.lookup_table.interp_1d`),
        so callers can pick the smoother interpolation rules without
        leaving the lazy pipeline:

        * ``"linear"`` (default) — uses the existing fast paths
          (``np.interp`` eager, native polars asof-join + linear-interp).
        * ``"pchip"`` — monotone cubic Hermite; smooth gradients,
          no overshoot near monotonic data.
        * ``"akima"`` — Akima 1970 cubic spline; less overshoot than
          the natural cubic on non-monotone data.
        * ``"cubic"`` — natural cubic spline (C^2 continuous, second
          derivative zero at boundaries).
        * ``"nearest"`` / ``"flat"`` — zero-gradient piecewise constant.

        For any non-linear method, the polars / DuckDB lazy paths fall
        back to materialising the upstream chain first and then routing
        each signal through ``interp_1d`` per-channel — non-linear
        interpolation is not expressible as a single polars expression.
        ``method="linear"`` keeps the native-polars / native-DuckDB
        pushdown so large lazy plans stay out-of-core.

        Polars backend (T-015a-followup-resample-pushdown): for
        ``method="linear"`` only, translated natively via two
        ``join_asof`` calls (backward + forward) plus a linear-interp
        expression — no Python ``map_batches`` callback. Target times
        must lie within the source range; non-monotonic ``t_new`` is
        supported (sorted internally, then re-permuted on output).
        """
        from ..library.lookup_table import interp_1d

        t_new_arr = np.asarray(t_new)

        def _interp_channel(t: np.ndarray, v: np.ndarray) -> np.ndarray:
            """Per-channel interpolator. ``method=='linear'`` stays on
            ``np.interp`` for byte-equivalence with phase 1; everything
            else routes through the T-106 backend."""
            if method == "linear":
                return np.interp(t_new_arr, t, v)
            return np.asarray(interp_1d(t_new_arr, t, v, method=method))

        def _apply(out: dict, t: np.ndarray):
            if t.size == 0:
                raise ValueError(
                    "LazyResults.resample: cannot resample an empty result "
                    "(an upstream .where() may have removed all rows)."
                )
            t_min, t_max = float(t[0]), float(t[-1])
            if np.any(t_new_arr < t_min - 1e-12) or np.any(t_new_arr > t_max + 1e-12):
                raise ValueError(
                    f"LazyResults.resample: requested times outside "
                    f"[{t_min}, {t_max}]."
                )
            new_out = {}
            for k, v in out.items():
                if v.ndim == 1:
                    new_out[k] = _interp_channel(t, v)
                else:
                    new_out[k] = np.stack(
                        [_interp_channel(t, v[:, i]) for i in range(v.shape[1])],
                        axis=-1,
                    )
            return new_out, t_new_arr

        def _polars_apply(lf, expanded_cols, _t_new=t_new_arr, _method=method):
            if _method == "linear":
                return _polars_resample(lf, _t_new)
            # Non-linear methods aren't expressible as a single polars
            # expression — collect, route through the eager path, and
            # re-promote. The pushdown on prior ops still ran lazily.
            import polars as pl

            df = lf.collect()
            t = df["time"].to_numpy()
            out = _collapse_vectorized(df)
            new_out, new_t = _apply(out, t)
            return _eager_to_lazyframe(new_t, new_out)

        return self._chain(
            _Op(
                name=f"resample(len={len(t_new_arr)}, method={method!r})",
                apply=_apply,
                polars_apply=_polars_apply,
            )
        )

    def with_signal(self, name: str, fn: Callable) -> "LazyResults":
        """Derive a new signal ``name`` from existing ones (defers).

        ``fn`` receives ``(t, outputs)`` and returns an array shaped
        like ``time``.
        """

        def _apply(out: dict, t: np.ndarray):
            new_out = dict(out)
            new_out[name] = np.asarray(fn(t, out))
            return new_out, t

        def _polars_apply(lf, expanded_cols, _name=name, _fn=fn):
            # Polars cannot express an arbitrary Python user function
            # natively; collect to compute, then re-lazy.  We still
            # benefit from polars's pushdown on prior ops in the chain
            # (they ran lazily before this point).
            import polars as pl

            df = lf.collect()
            t = df["time"].to_numpy()
            # Reconstruct the outputs dict — vector-valued signals are
            # exploded as ``base__i`` so we re-collapse them.
            out = _collapse_vectorized(df)
            new_col = np.asarray(_fn(t, out))
            return df.with_columns(pl.Series(_name, new_col)).lazy()

        return self._chain(
            _Op(
                name=f"with_signal({name!r})",
                apply=_apply,
                polars_apply=_polars_apply,
            )
        )

    # ── per-signal native cadence (T-108 phase 1) ─────────────────────

    def signal(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(time, value)`` for ``name`` at its NATIVE cadence.

        Eager (non-lazy) accessor: bypasses the deferred op chain and
        reads directly from the underlying recorded arrays.  Returns
        the per-signal timestamp vector populated by ``T-013`` /
        ``T-013a`` (Mode A or Mode B) when available, else falls back
        to the global :attr:`_time` vector — matching the semantics of
        :meth:`SimulationResults.time_for`.

        For Mode B "default"-classified signals (per-signal times are
        deduplicated but ``outputs`` stays at full length), the value
        array is back-projected onto the deduplicated times via
        ``searchsorted`` so the returned ``(time, value)`` pair has
        consistent shape — same trick used by
        :meth:`SimulationResults.align`.

        Raises:
            KeyError: if ``name`` is not a recorded signal.
        """
        if name not in self._outputs:
            raise KeyError(
                f"LazyResults.signal: unknown signal {name!r}.  "
                f"Available: {list(self._outputs)}"
            )
        value = np.asarray(self._outputs[name])
        time = self._native_time_for(name)
        if value.shape[0] != time.shape[0]:
            # Mode B "default" signal: outputs is full length, time is
            # deduplicated.  Project value onto the deduplicated times.
            global_t = np.asarray(self._time)
            idx = np.searchsorted(global_t, time)
            if global_t.shape[0] > 0:
                idx = np.clip(idx, 0, global_t.shape[0] - 1)
            value = value[idx]
        return time, value

    def cadence_of(self, name: str) -> str:
        """Classify ``name``'s recording cadence.

        Returns one of:

          - ``"continuous"`` — sampled every major step
            (``time_for(name).shape == self._time.shape`` and the
            value array is full-length).
          - ``"periodic"`` — sampled on a fixed schedule (Mode A path:
            both per-signal times AND outputs are shorter than the
            global vector and have matching length).
          - ``"event-driven"`` — Mode B value-diff dedup populated
            per-signal times but the output array remained at the
            global cadence (the recording pipeline could not pin a
            fixed period to the source ``OutputPort``).
          - ``"default"`` — no per-signal cadence info available; the
            signal shares the global :attr:`_time` vector.

        This is a structural classification derived from the recorded-
        array shapes — it does not re-invoke the static
        ``ResultsRecorder.classify_signal_cadence`` (which requires
        live ``OutputPort`` references that aren't carried on
        :class:`SimulationResults`).  The four buckets nevertheless
        line up 1-to-1 with the four cadence kinds the recording
        pipeline produces (continuous / periodic / event-driven /
        default), so a downstream consumer can plan I/O without
        reaching back into the simulator.

        Raises:
            KeyError: if ``name`` is not a recorded signal.
        """
        if name not in self._outputs:
            raise KeyError(
                f"LazyResults.cadence_of: unknown signal {name!r}.  "
                f"Available: {list(self._outputs)}"
            )
        if (
            self._per_signal_times is None
            or name not in self._per_signal_times
        ):
            return "default"
        global_t = np.asarray(self._time)
        sig_t = np.asarray(self._per_signal_times[name])
        sig_v = np.asarray(self._outputs[name])
        if global_t.shape[0] == 0:
            return "default"
        if sig_t.shape[0] == global_t.shape[0]:
            return "continuous"
        if sig_v.shape[0] == sig_t.shape[0]:
            # Mode A: both arrays were trimmed to the schedule.
            return "periodic"
        # Mode B fallback: times deduplicated, values still full length.
        return "event-driven"

    def align_to(self, name: str) -> "LazyResults":
        """Resample every signal to ``name``'s native cadence (defers).

        Convenience wrapper over :meth:`resample` that targets the
        per-signal time vector for ``name``.  Useful when one signal
        is the natural reference clock (e.g. a 1 Hz sensor) and you
        want every other recorded signal aligned to its ticks before
        materialising.

        The returned chain inherits the active backend (eager / polars
        / duckdb) and routes through the same ``resample`` translator
        — i.e. the polars backend uses the asof-join + linear-interp
        plan from T-015a-followup-resample-pushdown.

        Raises:
            KeyError: if ``name`` is not a recorded signal.
        """
        if name not in self._outputs:
            raise KeyError(
                f"LazyResults.align_to: unknown signal {name!r}.  "
                f"Available: {list(self._outputs)}"
            )
        target_time = self._native_time_for(name)
        return self.resample(target_time)

    def _native_time_for(self, name: str) -> np.ndarray:
        """Return the per-signal native time vector for ``name``.

        Falls back to the global :attr:`_time` vector when the lazy
        results object was constructed from a :class:`SimulationResults`
        without ``per_signal_times`` (the legacy default-off path) or
        when ``name`` is not in the per-signal map.  Mirrors
        :meth:`SimulationResults.time_for`.
        """
        if (
            self._per_signal_times is not None
            and name in self._per_signal_times
        ):
            return np.asarray(self._per_signal_times[name])
        return np.asarray(self._time)

    # ── terminals (eager-numpy path) ─────────────────────────────────────

    def _collect_eager(self) -> dict:
        outputs, time = self._outputs, self._time
        for op in self._ops:
            outputs, time = op.apply(outputs, time)
        return {"time": time, **outputs}

    def collect(self) -> dict:
        """Materialise the chain.  Returns ``{"time": t, **signals}``."""
        if self._use_duckdb:
            return self._collect_duckdb()
        if self._use_polars:
            df = self._to_polars_df()
            time = np.asarray(df["time"].to_numpy())
            outputs = _collapse_vectorized(df)
            return {"time": time, **outputs}
        return self._collect_eager()

    def to_numpy(self) -> dict:
        """Alias for :meth:`collect`."""
        return self.collect()

    def to_pandas(self):
        """Materialise to a ``pandas.DataFrame`` (requires pandas).

        Vector-valued signals are exploded into ``name__0``, ``name__1`` columns.
        """
        if self._use_duckdb:
            return self._duckdb_to_pandas()
        if self._use_polars:
            return self._to_polars_df().to_pandas()
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "LazyResults.to_pandas: pandas is not installed.  "
                "Install with `pip install pandas` or use .to_numpy()."
            ) from e

        materialised = self._collect_eager()
        time = materialised.pop("time")
        cols = {"time": time}
        for k, v in materialised.items():
            if v.ndim == 1:
                cols[k] = v
            else:
                for i in range(v.shape[-1]):
                    cols[f"{k}__{i}"] = v[..., i]
        return pd.DataFrame(cols)

    def to_polars(self):
        """Materialise to a ``polars.DataFrame`` (requires polars)."""
        try:
            import polars as pl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LazyResults.to_polars: polars is not installed.  "
                "Install with `pip install polars` or use .to_pandas()."
            ) from e

        if self._use_duckdb:
            return self._duckdb_to_polars()
        if self._use_polars:
            return self._to_polars_df()

        materialised = self._collect_eager()
        return _eager_dict_to_polars(materialised)

    def to_hdf5(
        self,
        path,
        key: str = "results",
        chunk_size: int = 10_000,
    ) -> None:
        """Stream-write the materialised result to an HDF5 file
        (T-108-followup-streaming-export).

        Layout: a top-level ``time`` dataset and an ``outputs/`` group
        holding one dataset per signal (vector-valued signals are
        exploded into ``outputs/<name>__<i>`` to mirror the parquet
        column convention).  Each dataset is created with
        ``maxshape=(None, ...)`` and extended chunk-by-chunk so the
        file never has to hold the full frame in memory at once.

        Parameters
        ----------
        path
            Destination ``.h5`` file path.  Overwritten if it exists.
        key
            Currently unused — reserved for forward compatibility with
            multi-result HDF5 files; the layout described above is
            relative to the file root and not under ``key``.
        chunk_size
            Rows written per extend.  Tune for memory / I/O trade-off;
            defaults to 10 000 rows.

        Notes
        -----
        Optional dep: requires ``h5py`` (``pip install h5py``).  Raises
        :class:`ImportError` if not available.
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "LazyResults.to_hdf5: h5py is not installed.  "
                "Install with `pip install h5py`."
            ) from e

        del key  # reserved; see docstring
        if chunk_size <= 0:
            raise ValueError(
                f"LazyResults.to_hdf5: chunk_size must be positive (got {chunk_size})."
            )

        with h5py.File(str(path), "w") as f:
            out_grp = f.create_group("outputs")
            time_ds: Optional[Any] = None
            sig_dsets: dict[str, Any] = {}
            written = 0
            for chunk in self._iter_chunks(chunk_size):
                time_chunk = chunk["time"]
                n = time_chunk.shape[0]
                if n == 0:
                    continue
                if time_ds is None:
                    time_ds = f.create_dataset(
                        "time",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=time_chunk.dtype,
                        chunks=(min(chunk_size, max(n, 1)),),
                    )
                time_ds.resize((written + n,))
                time_ds[written : written + n] = time_chunk

                for name, arr in chunk.items():
                    if name == "time":
                        continue
                    if name not in sig_dsets:
                        sig_dsets[name] = out_grp.create_dataset(
                            name,
                            shape=(0,),
                            maxshape=(None,),
                            dtype=arr.dtype,
                            chunks=(min(chunk_size, max(n, 1)),),
                        )
                    ds = sig_dsets[name]
                    ds.resize((written + n,))
                    ds[written : written + n] = arr
                written += n

    def to_zarr(self, path, chunk_size: int = 10_000) -> None:
        """Stream-write the materialised result to a zarr store
        (T-108-followup-streaming-export).

        Layout mirrors :meth:`to_hdf5`: a ``time`` array at the group
        root and one array per signal under ``outputs/`` (vector-valued
        signals exploded as ``outputs/<name>__<i>``).  Each array is
        created with ``shape=(0,)`` and resized in place per chunk.

        Parameters
        ----------
        path
            Destination directory (a zarr v3 store).  Created if absent;
            overwritten otherwise.
        chunk_size
            Rows written per extend.  Also used as the underlying zarr
            chunk dimension so I/O alignment matches the write cadence.

        Notes
        -----
        Optional dep: requires ``zarr`` (``pip install zarr``).  Raises
        :class:`ImportError` if not available.
        """
        try:
            import zarr
        except ImportError as e:
            raise ImportError(
                "LazyResults.to_zarr: zarr is not installed.  "
                "Install with `pip install zarr`."
            ) from e

        if chunk_size <= 0:
            raise ValueError(
                f"LazyResults.to_zarr: chunk_size must be positive (got {chunk_size})."
            )

        root = zarr.open_group(str(path), mode="w")
        out_grp = root.create_group("outputs")
        time_arr: Optional[Any] = None
        sig_arrs: dict[str, Any] = {}
        written = 0
        for chunk in self._iter_chunks(chunk_size):
            time_chunk = chunk["time"]
            n = time_chunk.shape[0]
            if n == 0:
                continue
            if time_arr is None:
                time_arr = root.create_array(
                    "time",
                    shape=(0,),
                    chunks=(min(chunk_size, max(n, 1)),),
                    dtype=time_chunk.dtype,
                )
            time_arr.resize((written + n,))
            time_arr[written : written + n] = time_chunk

            for name, arr in chunk.items():
                if name == "time":
                    continue
                if name not in sig_arrs:
                    sig_arrs[name] = out_grp.create_array(
                        name,
                        shape=(0,),
                        chunks=(min(chunk_size, max(n, 1)),),
                        dtype=arr.dtype,
                    )
                za = sig_arrs[name]
                za.resize((written + n,))
                za[written : written + n] = arr
            written += n

    def _iter_chunks(self, chunk_size: int):
        """Yield ``{column_name: np.ndarray}`` dicts of at most ``chunk_size``
        rows each.

        For the default eager-numpy and the DuckDB / polars backends we
        materialise once and slice the result.  This is "honest
        streaming" only in the sense that the writer never holds more
        than one chunk *as its own copy* — the underlying source frame
        may already be in memory.  True out-of-core streaming would
        require a polars ``sink_batches`` plan, which the writer can
        layer on top of this iterator in a follow-up.

        Vector-valued signals are exploded into ``name__i`` keys so the
        consumer can treat every column as a 1-D array.
        """
        materialised = self.collect()
        time = np.asarray(materialised.pop("time"))
        n = time.shape[0]
        # Pre-explode vector-valued signals so we don't allocate
        # ``name__i`` arrays on every slice.
        exploded: dict[str, np.ndarray] = {}
        for k, v in materialised.items():
            arr = np.asarray(v)
            if arr.ndim == 1:
                exploded[k] = arr
            else:
                for i in range(arr.shape[-1]):
                    exploded[f"{k}__{i}"] = arr[..., i]

        if n == 0:
            yield {"time": time, **exploded}
            return

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            chunk: dict[str, np.ndarray] = {"time": time[start:stop]}
            for k, arr in exploded.items():
                chunk[k] = arr[start:stop]
            yield chunk

    def to_parquet(self, path, batch_size: Optional[int] = None):
        """Write the materialised result to ``path`` as Parquet.

        With the polars backend (T-015a) and ``batch_size=None``, writes
        via ``LazyFrame.sink_parquet`` for true streaming output that
        never materialises the whole frame in memory.  With
        ``batch_size=N``, partitions the output into multiple files
        ``path.0.parquet`` / ``path.1.parquet`` / ... each holding at
        most ``N`` rows.

        With the DuckDB backend (T-015a-followup-resample-pushdown-duckdb)
        and ``batch_size=None``, writes via DuckDB's native
        ``COPY (sql) TO 'path' (FORMAT PARQUET)`` — genuinely streaming
        (DuckDB never materialises the whole result in Python memory).
        ``batch_size=N`` partitions on the Python side just like polars.

        Without an opt-in backend, uses pandas (``pyarrow``) and falls
        back to polars when pandas is unavailable.
        """
        if self._use_duckdb:
            self._sink_parquet_duckdb(path, batch_size=batch_size)
            return
        if self._use_polars:
            self._sink_parquet_polars(path, batch_size=batch_size)
            return

        try:
            df = self.to_pandas()
            df.to_parquet(path)
            return
        except ImportError:
            pass
        df = self.to_polars()
        df.write_parquet(str(path))

    # ── polars-backend internals (T-015a) ────────────────────────────────

    def _build_lazyframe(self):
        """Build the initial polars LazyFrame from numpy outputs."""
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "LazyResults: polars backend requested but polars is not "
                "installed.  Install with `pip install polars`."
            ) from e

        data: dict[str, np.ndarray] = {"time": np.asarray(self._time)}
        for k, v in self._outputs.items():
            arr = np.asarray(v)
            if arr.ndim == 1:
                data[k] = arr
            else:
                for i in range(arr.shape[-1]):
                    data[f"{k}__{i}"] = arr[..., i]
        return pl.DataFrame(data).lazy()

    def _to_polars_df(self):
        """Run the op chain through polars; return a materialised DataFrame.

        Per-op fallback: when an op declares no ``polars_apply``
        translator (or that translator declines), the chain is collected
        to numpy, the eager apply runs, and the chain re-enters the
        polars LazyFrame.  Emits :class:`RuntimeWarning` on fallback.
        """
        lf = self._build_lazyframe()
        for op in self._ops:
            if op.polars_apply is None:
                warnings.warn(
                    f"LazyResults.with_polars_backend: op {op.name!r} has no "
                    f"polars equivalent; falling back to eager-numpy for this "
                    f"step.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                df = lf.collect()
                outputs, time = _df_to_eager(df)
                outputs, time = op.apply(outputs, time)
                lf = _eager_to_lazyframe(time, outputs)
            else:
                expanded = (
                    lf.collect_schema().names()
                    if hasattr(lf, "collect_schema")
                    else lf.columns
                )
                lf = op.polars_apply(lf, expanded)
        return lf.collect()

    def _sink_parquet_polars(self, path, batch_size: Optional[int]) -> None:
        """Stream the LazyFrame to parquet via ``sink_parquet``.

        With ``batch_size=None`` we let polars stream the whole plan to
        a single file.  With ``batch_size=N`` we materialise once,
        partition by row count, and write multiple files.
        """
        # Build the full LazyFrame (running any per-op fallbacks).
        # _to_polars_df runs the chain end-to-end and returns a
        # DataFrame; for the "true streaming" case we want to keep the
        # plan lazy.  When every op has a polars translator, we can
        # build lazily and sink_parquet.  Otherwise we fall back to
        # collect+write.
        from pathlib import Path

        path = str(path)
        if batch_size is None and all(op.polars_apply is not None for op in self._ops):
            lf = self._build_lazyframe()
            for op in self._ops:
                expanded = (
                    lf.collect_schema().names()
                    if hasattr(lf, "collect_schema")
                    else lf.columns
                )
                lf = op.polars_apply(lf, expanded)
            try:
                lf.sink_parquet(path)
                return
            except Exception:
                # Some plans (e.g. those built via map_batches) cannot
                # stream — fall through to the materialise path.
                pass

        df = self._to_polars_df()
        if batch_size is None:
            df.write_parquet(path)
            return

        n = df.height
        base = Path(path)
        stem = base.with_suffix("")
        suffix = base.suffix or ".parquet"
        for i, start in enumerate(range(0, n, batch_size)):
            chunk = df.slice(start, batch_size)
            chunk.write_parquet(f"{stem}.{i}{suffix}")

    # ── debug helpers ────────────────────────────────────────────────────

    def explain(self) -> str:
        """Render the deferred operation chain as a human-readable string."""
        if not self._ops:
            tag = "<identity>"
        else:
            tag = " | ".join(op.name for op in self._ops)
        if self._use_duckdb:
            return f"[duckdb] {tag}"
        if self._use_polars:
            return f"[polars] {tag}"
        return tag

    # ── duckdb-backend internals (T-015a-followup-resample-pushdown-duckdb) ─

    def _build_duckdb_plan(self) -> "_DuckDBPlan":
        """Register the (time, outputs) arrays as a DuckDB table and
        return an initial :class:`_DuckDBPlan` selecting all columns."""
        if self._duckdb_conn is None:  # pragma: no cover — defensive
            raise RuntimeError(
                "LazyResults: DuckDB backend requested but no connection "
                "is attached."
            )

        data: dict[str, np.ndarray] = {"time": np.asarray(self._time)}
        for k, v in self._outputs.items():
            arr = np.asarray(v)
            if arr.ndim == 1:
                data[k] = arr
            else:
                for i in range(arr.shape[-1]):
                    data[f"{k}__{i}"] = arr[..., i]

        # Use a unique table name per build so re-registration after a
        # mid-chain fallback does not collide with prior state on the
        # same connection.
        table = f"jaxonomy_lazy_{id(self)}_{_DuckDBPlan._next_id()}"
        self._duckdb_conn.register(table, data)
        cols = list(data.keys())
        return _DuckDBPlan(
            conn=self._duckdb_conn,
            table=table,
            columns=cols,
            select_cols=list(cols),
            where_clauses=[],
            mask_arrays=[],
        )

    def _materialize_duckdb(self) -> "_DuckDBPlan":
        """Run the op chain through DuckDB; return the final plan.

        Per-op fallback: when an op declares no ``duckdb_apply``
        translator, the chain materialises to numpy via fetchnumpy,
        the eager apply runs, and the chain re-registers a fresh
        DuckDB table.  Emits :class:`RuntimeWarning` on fallback.
        """
        plan = self._build_duckdb_plan()
        for op in self._ops:
            if op.duckdb_apply is None:
                warnings.warn(
                    f"LazyResults.with_duckdb_backend: op {op.name!r} has no "
                    f"DuckDB SQL equivalent; falling back to eager-numpy for "
                    f"this step.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                outputs, time = plan.fetch_eager()
                outputs, time = op.apply(outputs, time)
                plan = _eager_to_duckdb_plan(self._duckdb_conn, time, outputs)
            else:
                plan = op.duckdb_apply(plan, plan.select_cols)
        return plan

    def _collect_duckdb(self) -> dict:
        plan = self._materialize_duckdb()
        outputs, time = plan.fetch_eager()
        return {"time": time, **outputs}

    def _duckdb_to_pandas(self):
        plan = self._materialize_duckdb()
        return plan.fetch_pandas()

    def _duckdb_to_polars(self):
        plan = self._materialize_duckdb()
        return plan.fetch_polars()

    def _sink_parquet_duckdb(self, path, batch_size: Optional[int]) -> None:
        from pathlib import Path

        path_str = str(path)
        plan = self._materialize_duckdb()
        if batch_size is None:
            plan.copy_to_parquet(path_str)
            return

        # Batched: fall back to materialise + slice.  DuckDB supports
        # partitioned writes via PARTITION_BY, but only on a column,
        # not a row-count chunk size; row-chunked output is rare enough
        # that we don't try to optimise it.
        outputs, time = plan.fetch_eager()
        n = len(time)
        base = Path(path_str)
        stem = base.with_suffix("")
        suffix = base.suffix or ".parquet"
        for i, start in enumerate(range(0, n, batch_size)):
            stop = min(start + batch_size, n)
            sub_t = time[start:stop]
            sub_out = {k: v[start:stop] for k, v in outputs.items()}
            sub_plan = _eager_to_duckdb_plan(self._duckdb_conn, sub_t, sub_out)
            sub_plan.copy_to_parquet(f"{stem}.{i}{suffix}")


# ── module-level helpers (T-015a) ────────────────────────────────────────


def _collapse_vectorized(df) -> dict[str, np.ndarray]:
    """Re-collapse ``name__i`` columns of a polars DataFrame into ``(T, k)``."""
    cols = df.columns
    groups: dict[str, dict[int, str]] = {}
    out: dict[str, np.ndarray] = {}
    for c in cols:
        if c == "time":
            continue
        if "__" in c:
            base, _, idx_str = c.rpartition("__")
            try:
                idx = int(idx_str)
            except ValueError:
                out[c] = np.asarray(df[c].to_numpy())
                continue
            groups.setdefault(base, {})[idx] = c
        else:
            out[c] = np.asarray(df[c].to_numpy())
    for base, idx_map in groups.items():
        ordered = [df[idx_map[i]].to_numpy() for i in sorted(idx_map)]
        out[base] = np.stack([np.asarray(a) for a in ordered], axis=-1)
    return out


def _df_to_eager(df) -> tuple[dict[str, np.ndarray], np.ndarray]:
    time = np.asarray(df["time"].to_numpy())
    return _collapse_vectorized(df), time


def _eager_to_lazyframe(time: np.ndarray, outputs: dict[str, np.ndarray]):
    import polars as pl

    data: dict[str, np.ndarray] = {"time": np.asarray(time)}
    for k, v in outputs.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            data[k] = arr
        else:
            for i in range(arr.shape[-1]):
                data[f"{k}__{i}"] = arr[..., i]
    return pl.DataFrame(data).lazy()


def _eager_dict_to_polars(materialised: dict):
    import polars as pl

    time = materialised.pop("time")
    data = {"time": np.asarray(time)}
    for k, v in materialised.items():
        v = np.asarray(v)
        if v.ndim == 1:
            data[k] = v
        else:
            for i in range(v.shape[-1]):
                data[f"{k}__{i}"] = v[..., i]
    return pl.DataFrame(data)


def _polars_resample(lf, t_new: np.ndarray):
    """Native-polars linear-interp resample via dual ``join_asof`` (T-015a-followup).

    For each non-time column ``c`` in the source LazyFrame ``lf`` we
    compute ``c[i] + coef * (c[i+1] - c[i])`` where ``i`` is the index
    of the largest source time ``<= t_new`` and ``coef = (t_new - t_left)
    / (t_right - t_left)``.  Both neighbour lookups happen via
    ``LazyFrame.join_asof`` (``backward`` + ``forward``) so the whole
    plan stays lazy and benefits from polars's columnar / SIMD pipeline.

    Notes:
      * ``join_asof`` requires sorted keys.  We sort the source by
        ``time`` and, if the requested ``t_new`` is not monotonically
        non-decreasing, we sort it internally and re-permute the output
        rows back into the user-supplied order so the eager-numpy path
        and this path stay byte-equivalent.
      * Bounds-checking matches the eager path: ``t_new`` outside
        ``[t.min, t.max]`` (with a 1e-12 tolerance) raises.
      * Endpoints (``t_new == t_left == t_right``) yield ``coef = 0`` —
        the result is exactly the source value at that knot.
    """
    import polars as pl

    t_new = np.asarray(t_new)

    # Materialise the source schema lazily — needed both for column
    # listing and to drive bounds-checking.  ``collect_schema`` keeps
    # this cheap; we don't pull data here.
    schema = (
        lf.collect_schema()
        if hasattr(lf, "collect_schema")
        else {c: None for c in lf.columns}
    )
    cols = list(schema.names()) if hasattr(schema, "names") else list(schema)
    if "time" not in cols:
        raise ValueError(
            "LazyResults.resample (polars): source LazyFrame has no 'time' column."
        )
    signal_cols = [c for c in cols if c != "time"]

    # We need source min/max for bounds-checking and to know whether we
    # have a non-empty source at all.  This is one cheap aggregation
    # over the (already optimised) plan.
    bounds = lf.select(
        pl.col("time").min().alias("__tmin"),
        pl.col("time").max().alias("__tmax"),
        pl.col("time").len().alias("__tlen"),
    ).collect()
    tlen = int(bounds["__tlen"][0])
    if tlen == 0:
        raise ValueError(
            "LazyResults.resample: cannot resample an empty result "
            "(an upstream .where() may have removed all rows)."
        )
    t_min = float(bounds["__tmin"][0])
    t_max = float(bounds["__tmax"][0])
    if np.any(t_new < t_min - 1e-12) or np.any(t_new > t_max + 1e-12):
        raise ValueError(
            f"LazyResults.resample: requested times outside [{t_min}, {t_max}]."
        )

    # Ensure source is sorted on time (asof-join requirement).  Set the
    # sorted flag so polars can skip a re-sort in the optimiser.
    src = lf.sort("time").set_sorted("time")

    # Build the target LazyFrame.  Track original order via a row index
    # so we can restore non-monotonic ``t_new`` on the way out.
    sort_idx = np.argsort(t_new, kind="stable")
    needs_unsort = bool(np.any(sort_idx != np.arange(len(t_new))))
    t_new_sorted = t_new[sort_idx]
    # ``__orig_pos`` is the row's position in the *user-supplied*
    # (possibly unsorted) ``t_new`` array.  Sorting the output by
    # ``__orig_pos`` therefore restores the eager-numpy row order.
    tgt = pl.DataFrame(
        {
            "time": t_new_sorted,
            "__orig_pos": np.asarray(sort_idx, dtype=np.int64),
        }
    ).lazy().set_sorted("time")

    # Left neighbour (backward asof) — the source time <= target time.
    # We carry "_t_left" as a renamed copy of source time so we can
    # compute the interp coefficient without losing it through join.
    src_left = src.with_columns(pl.col("time").alias("_t_left"))
    left = tgt.join_asof(src_left, on="time", strategy="backward")
    # Rename signal columns to "<c>__L" suffix.
    left = left.rename({c: f"{c}__L" for c in signal_cols})

    # Right neighbour (forward asof).
    rename_r = {c: f"{c}__R" for c in signal_cols}
    src_right = src.rename(rename_r).with_columns(pl.col("time").alias("_t_right"))
    combined = left.join_asof(src_right, on="time", strategy="forward")

    # Linear-interp coefficient.  When _t_left == _t_right (exact hit
    # or endpoint), coef must be 0 to avoid 0/0.
    dt = pl.col("_t_right") - pl.col("_t_left")
    coef_expr = (
        pl.when(dt == 0.0)
        .then(pl.lit(0.0))
        .otherwise((pl.col("time") - pl.col("_t_left")) / dt)
    ).alias("__coef")
    combined = combined.with_columns(coef_expr)

    interp_exprs = [
        (pl.col(f"{c}__L") + pl.col("__coef") * (pl.col(f"{c}__R") - pl.col(f"{c}__L"))).alias(c)
        for c in signal_cols
    ]
    out = combined.with_columns(interp_exprs).select(["time", "__orig_pos", *signal_cols])

    if needs_unsort:
        out = out.sort("__orig_pos")
    out = out.drop("__orig_pos")
    return out


# ── module-level helpers (T-015a-followup-resample-pushdown-duckdb) ─────


@dataclass
class _DuckDBPlan:
    """Accumulated SQL plan against a registered DuckDB table.

    Each translatable op pushes a ``SELECT`` projection or a ``WHERE``
    predicate onto the plan; only at the terminal call do we execute
    ``SELECT <select_cols> FROM <table> [WHERE ...]`` against the
    connection.

    Re-registration: when an op cannot be expressed in SQL, the plan
    is materialised to numpy via :meth:`fetch_eager`, the eager apply
    runs, and a new :class:`_DuckDBPlan` is built around a freshly
    registered table (see :func:`_eager_to_duckdb_plan`).  Any column
    drop from a prior ``select`` survives the round-trip because the
    re-registration uses only the surviving columns as input.
    """

    conn: Any
    table: str
    columns: list[str]
    select_cols: list[str]
    where_clauses: list[str]
    mask_arrays: list[np.ndarray]

    _counter: int = 0  # class var; bumped via _next_id

    @classmethod
    def _next_id(cls) -> int:
        cls._counter += 1
        return cls._counter

    # ── plan-building API used by op translators ─────────────────────

    def with_select(self, keep: list[str]) -> "_DuckDBPlan":
        return _DuckDBPlan(
            conn=self.conn,
            table=self.table,
            columns=self.columns,
            select_cols=list(keep),
            where_clauses=list(self.where_clauses),
            mask_arrays=list(self.mask_arrays),
        )

    def with_where(self, sql_expr: str) -> "_DuckDBPlan":
        return _DuckDBPlan(
            conn=self.conn,
            table=self.table,
            columns=self.columns,
            select_cols=list(self.select_cols),
            where_clauses=self.where_clauses + [sql_expr],
            mask_arrays=list(self.mask_arrays),
        )

    def with_mask_array(self, mask: np.ndarray) -> "_DuckDBPlan":
        # Boolean-array masks are aligned to the *source* table's
        # original row order.  Once a prior fallback has rebuilt the
        # table, the mask alignment is invalidated — guard against
        # that case.  In practice the chain enforces alignment at
        # build time; the row count check at materialise catches
        # surviving anomalies.
        return _DuckDBPlan(
            conn=self.conn,
            table=self.table,
            columns=self.columns,
            select_cols=list(self.select_cols),
            where_clauses=list(self.where_clauses),
            mask_arrays=self.mask_arrays + [mask],
        )

    # ── SQL composition / execution ──────────────────────────────────

    def _compose_sql(self) -> str:
        cols = ", ".join(_quote_ident(c) for c in self.select_cols)
        sql = f'SELECT {cols} FROM "{self.table}"'
        clauses = list(self.where_clauses)
        # Mask arrays: register an aux table per mask and AND it via a
        # row-id join.  We use ``rowid`` (DuckDB virtual column) for
        # the alignment.
        mask_joins: list[str] = []
        if self.mask_arrays:
            for i, m in enumerate(self.mask_arrays):
                # Sanity: mask must align with original table length.
                idxs = np.flatnonzero(np.asarray(m, dtype=bool))
                aux_name = f"{self.table}_mask_{i}"
                aux_data = {"__mask_idx": np.asarray(idxs, dtype=np.int64)}
                self.conn.register(aux_name, aux_data)
                mask_joins.append(aux_name)
        if mask_joins:
            # Join on rowid.  Use explicit aliases to keep things tidy.
            base = f'(SELECT *, ROW_NUMBER() OVER () - 1 AS __rn FROM "{self.table}") AS _src'
            join_clause = base
            for i, aux_name in enumerate(mask_joins):
                alias = f"_m{i}"
                join_clause += (
                    f' JOIN "{aux_name}" AS {alias} '
                    f'ON _src.__rn = {alias}.__mask_idx'
                )
            sql = f'SELECT {cols} FROM {join_clause}'
        if clauses:
            sql += " WHERE " + " AND ".join(f"({c})" for c in clauses)
        return sql

    def fetch_eager(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        sql = self._compose_sql()
        result = self.conn.execute(sql).fetchnumpy()
        # ``fetchnumpy`` returns dict[str, masked_array | ndarray]; coerce.
        outputs: dict[str, np.ndarray] = {}
        time = np.asarray(result.get("time", np.zeros(0)))
        scalars: list[str] = []
        groups: dict[str, dict[int, str]] = {}
        for c, arr in result.items():
            if c == "time":
                continue
            if "__" in c:
                base, _, idx_str = c.rpartition("__")
                try:
                    idx = int(idx_str)
                except ValueError:
                    scalars.append(c)
                    continue
                groups.setdefault(base, {})[idx] = c
            else:
                scalars.append(c)
        for c in scalars:
            outputs[c] = np.asarray(result[c])
        for base, idx_map in groups.items():
            ordered = [np.asarray(result[idx_map[i]]) for i in sorted(idx_map)]
            outputs[base] = np.stack(ordered, axis=-1)
        return outputs, time

    def fetch_pandas(self):
        sql = self._compose_sql()
        return self.conn.execute(sql).df()

    def fetch_polars(self):
        sql = self._compose_sql()
        try:
            return self.conn.execute(sql).pl()
        except (ImportError, ModuleNotFoundError):
            # ``.pl()`` requires pyarrow; fall back to building a
            # polars DataFrame from fetchnumpy (no Arrow dependency).
            outputs, time = self.fetch_eager()
            return _eager_dict_to_polars({"time": time, **outputs})

    def copy_to_parquet(self, path: str) -> None:
        sql = self._compose_sql()
        # DuckDB's COPY (subquery) TO 'path' (FORMAT PARQUET) is a
        # genuinely streaming write.
        # Escape single-quotes in path defensively.
        esc = path.replace("'", "''")
        self.conn.execute(f"COPY ({sql}) TO '{esc}' (FORMAT PARQUET)")


def _quote_ident(name: str) -> str:
    # DuckDB identifier quoting: double quotes; double any embedded
    # double-quote.  Our column names are jaxonomy-internal and never
    # contain special chars, but we belt-and-brace this anyway.
    safe = name.replace('"', '""')
    return f'"{safe}"'


def _python_predicate_to_sql(expr: str) -> str:
    """Translate a Python-style :meth:`LazyResults.where` expression to SQL.

    Conservative substitutions:

      * ``&`` / ``|`` (Python bitwise/numpy boolean) -> ``AND`` / ``OR``;
      * ``==`` -> ``=``;
      * lone ``t`` references (free-standing) -> ``time``.

    Comparison operators (``>``, ``<``, ``>=``, ``<=``, ``!=``) and
    arithmetic (``+``, ``-``, ``*``, ``/``) pass through unchanged.
    """
    import re

    s = expr
    # Word-boundary substitution for "t" -> "time".
    s = re.sub(r"\bt\b", "time", s)
    # Operator translation.  Order matters: handle "==" before "=".
    s = s.replace("==", "=")
    s = s.replace("&", " AND ")
    s = s.replace("|", " OR ")
    return s


def _eager_to_duckdb_plan(
    conn, time: np.ndarray, outputs: dict[str, np.ndarray]
) -> "_DuckDBPlan":
    """Re-register a ``(time, outputs)`` pair as a fresh DuckDB table.

    Used to recover from a mid-chain fallback so subsequent ops can
    re-enter the SQL path.  Vector-valued signals are exploded into
    ``name__i`` columns.
    """
    data: dict[str, np.ndarray] = {"time": np.asarray(time)}
    for k, v in outputs.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            data[k] = arr
        else:
            for i in range(arr.shape[-1]):
                data[f"{k}__{i}"] = arr[..., i]
    table = f"jaxonomy_lazy_rebuild_{_DuckDBPlan._next_id()}"
    conn.register(table, data)
    cols = list(data.keys())
    return _DuckDBPlan(
        conn=conn,
        table=table,
        columns=cols,
        select_cols=list(cols),
        where_clauses=[],
        mask_arrays=[],
    )
