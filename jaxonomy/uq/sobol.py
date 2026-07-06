# SPDX-License-Identifier: MIT

"""Variance-based global sensitivity analysis (Sobol indices).

:func:`sobol_indices` runs the standard Saltelli sampling scheme:

1. Draw two independent ``(N, d)`` matrices ``A`` and ``B`` from the user's
   parameter distributions.
2. For each parameter ``i`` build ``AB_i`` by replacing column ``i`` of ``A``
   with column ``i`` of ``B``.
3. Evaluate the QoI on ``A``, ``B``, and each ``AB_i`` (total ``N*(d+2)``
   model evaluations).
4. Estimate first-order ``S_i = V[E[Y|X_i]] / V[Y]`` and total-order
   ``S_T_i = E[V[Y|X_~i]] / V[Y]`` per parameter.

Two execution modes:

* The ``qoi_fn`` callable can take a ``param_batches``-shaped dict and return
  a ``(N,)`` array directly — useful for analytic test functions and when the
  user wants to bypass the simulator (e.g. surrogate models).
* If a ``diagram`` and ``t_span`` are provided, ``simulate_batch`` is called
  for each of the ``d+2`` matrices and the results are passed to
  ``qoi_fn(BatchSimulationResults) -> (N,)``.

Default ``n_samples=1024`` matches the SALib convention.  For ``d > 10``
parameters reduce ``n_samples`` to ~256–512: total evaluations grow linearly
with both ``N`` and ``d`` and the kernel-path JIT compile of
``simulate_batch`` is paid once per matrix regardless.

Example:

    >>> import jax  # doctest: +SKIP
    >>> from jaxonomy.uq import sobol_indices, Uniform  # doctest: +SKIP
    >>> dists = {"x1": Uniform(-3.14, 3.14), "x2": Uniform(-3.14, 3.14)}
    >>> def qoi(p):
    ...     import jax.numpy as jnp
    ...     return jnp.sin(p["x1"]) + 7.0 * jnp.sin(p["x2"]) ** 2
    >>> idx = sobol_indices(None, None, dists, qoi, n_samples=512,
    ...                     key=jax.random.PRNGKey(0))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from .distributions import Distribution

__all__ = [
    "sobol_indices",
    "saltelli_sample",
]


# ---------------------------------------------------------------------------
# Saltelli sampling matrices
# ---------------------------------------------------------------------------

def saltelli_sample(
    distributions: Mapping[str, Distribution],
    n_samples: int,
    key,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray], list[dict[str, jnp.ndarray]]]:
    """Build the Saltelli sample matrices.

    Args:
        distributions: Mapping ``name -> Distribution``.
        n_samples: ``N`` — base samples per matrix; total evaluations are
            ``N * (d + 2)`` where ``d = len(distributions)``.
        key: ``jax.random.PRNGKey``.

    Returns:
        ``(A, B, AB_list)`` where ``A`` and ``B`` are ``param_batches`` dicts
        of shape ``(N,)`` each, and ``AB_list`` is a list (one per parameter)
        of dicts where ``AB_list[i][param_i] = B[param_i]`` and the rest equal
        ``A``.
    """
    names = list(distributions.keys())
    d = len(names)
    keys = jax.random.split(key, 2 * d)

    A = {n: distributions[n].sample(keys[i], (n_samples,)) for i, n in enumerate(names)}
    B = {n: distributions[n].sample(keys[d + i], (n_samples,)) for i, n in enumerate(names)}

    AB_list: list[dict[str, jnp.ndarray]] = []
    for i, name in enumerate(names):
        AB_i = dict(A)
        AB_i[name] = B[name]
        AB_list.append(AB_i)

    return A, B, AB_list


# ---------------------------------------------------------------------------
# Index estimators (Saltelli 2010 / Jansen 1999)
# ---------------------------------------------------------------------------

def _saltelli_indices(yA, yB, yAB) -> tuple[float, float]:
    """Compute first- and total-order Sobol indices for one parameter.

    Uses the Jansen estimators (numerically more stable than the original
    Sobol formulas at moderate ``N``):

    * ``S_i  = (V[Y] - 0.5 * mean((yB - yAB)^2)) / V[Y]``
    * ``S_T_i = 0.5 * mean((yA - yAB)^2) / V[Y]``
    """
    yA = jnp.asarray(yA)
    yB = jnp.asarray(yB)
    yAB = jnp.asarray(yAB)
    var_y = jnp.var(jnp.concatenate([yA, yB]))
    var_y = jnp.where(var_y < 1e-30, 1e-30, var_y)
    s_first = (var_y - 0.5 * jnp.mean((yB - yAB) ** 2)) / var_y
    s_total = 0.5 * jnp.mean((yA - yAB) ** 2) / var_y
    return float(s_first), float(s_total)


def _bootstrap_index_ci(
    yA: jnp.ndarray,
    yB: jnp.ndarray,
    yAB: jnp.ndarray,
    n_bootstrap: int,
    ci_level: float,
    key,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Percentile bootstrap CIs for the first- and total-order Sobol indices.

    Resamples the ``(yA[i], yB[i], yAB[i])`` triplets with replacement
    ``n_bootstrap`` times and computes the Jansen estimators on each
    resample. Returns ``((s1_lo, s1_hi), (st_lo, st_hi))`` at the
    ``ci_level`` confidence level (e.g. 0.95 → 2.5%/97.5% percentiles).

    Vectorised in JAX so a 1000-sample bootstrap on N=1024 runs in a
    single XLA launch with no Python-level iteration.
    """
    yA = jnp.asarray(yA)
    yB = jnp.asarray(yB)
    yAB = jnp.asarray(yAB)
    N = yA.shape[0]
    idx = jax.random.randint(key, (n_bootstrap, N), 0, N)
    yA_b = yA[idx]
    yB_b = yB[idx]
    yAB_b = yAB[idx]
    var_y = jnp.var(jnp.concatenate([yA_b, yB_b], axis=1), axis=1)
    var_y = jnp.where(var_y < 1e-30, 1e-30, var_y)
    s1 = (var_y - 0.5 * jnp.mean((yB_b - yAB_b) ** 2, axis=1)) / var_y
    st = 0.5 * jnp.mean((yA_b - yAB_b) ** 2, axis=1) / var_y
    alpha = (1.0 - ci_level) / 2.0
    quantiles = jnp.asarray([alpha, 1.0 - alpha])
    s1_q = jnp.quantile(s1, quantiles)
    st_q = jnp.quantile(st, quantiles)
    return (
        (float(s1_q[0]), float(s1_q[1])),
        (float(st_q[0]), float(st_q[1])),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sobol_indices(
    diagram,
    t_span,
    distributions: Mapping[str, Distribution],
    qoi_fn: Callable[..., Any],
    n_samples: int = 1024,
    options=None,
    recorded_signals=None,
    key=None,
    fused: bool = True,
    n_bootstrap: int | None = None,
    ci_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Variance-based global sensitivity (first-order + total-order Sobol).

    Two modes:

    * **Analytic / surrogate mode**: ``diagram is None``.  The callable
      ``qoi_fn`` must accept a ``param_batches``-shaped dict and return a
      ``(N,)`` array.  No simulation runs — useful for benchmarking against
      Ishigami / Sobol-G analytic indices and for evaluating fast surrogates.
    * **Simulation mode**: ``diagram`` is a :class:`~jaxonomy.framework.diagram.Diagram`.
      With ``fused=True`` (default) all ``(d + 2) * N`` Saltelli evaluations
      are stacked into a single ``simulate_batch`` call so the kernel JIT
      compile cost is paid once instead of ``d + 2`` times.  With
      ``fused=False`` the legacy path is used (one ``simulate_batch`` call
      per matrix), which trades off compile cost for lower peak memory.

    Args:
        diagram: Optional :class:`Diagram`.  When ``None`` the function runs
            in analytic mode.
        t_span: ``(t_start, t_stop)`` — only used in simulation mode.
        distributions: ``{param_path: Distribution}``.
        qoi_fn: Quantity of interest.  In analytic mode receives a dict of
            shape-``(N,)`` arrays and returns a ``(N,)`` array.  In simulation
            mode receives a :class:`BatchSimulationResults` and returns a
            ``(N,)`` array.
        n_samples: Base sample count ``N``; total evaluations ``= N * (d + 2)``.
            Default 1024.  Reduce to 256–512 for ``d > 10`` if memory is a
            concern.
        options: Forwarded to :func:`simulate_batch` in simulation mode.
        recorded_signals: Forwarded to :func:`simulate_batch` in simulation
            mode.
        key: ``jax.random.PRNGKey`` (defaulting to ``PRNGKey(0)``).
        fused: When ``True`` (default), stack the ``A``, ``B`` and ``AB_i``
            matrices into one ``(d + 2) * N`` batch and run a single
            ``simulate_batch`` / ``qoi_fn`` call — the kernel JIT compile is
            amortised across all matrices and the results are bit-identical
            to the unfused path.  When ``False``, fall back to ``d + 2``
            separate calls (lower peak memory, ``d + 1`` extra compiles).
        n_bootstrap: When set (e.g. 1000), draw ``n_bootstrap`` resamples
            (with replacement) of the ``(yA, yB, yAB_i)`` triplets and
            return percentile confidence intervals on the indices
            alongside the point estimates. The Jansen point estimator
            can come out slightly negative at small ``N`` for near-zero
            true indices; the bootstrap CI is the production-grade way
            to quantify whether such a value is statistically distinct
            from zero. Default ``None`` skips the resampling step (no
            extra compute). (T-126-followup-sobol-bootstrap-cis)
        ci_level: Confidence level for the bootstrap intervals; ignored
            when ``n_bootstrap is None``. Default ``0.95`` (i.e. 2.5%/
            97.5% percentile bands).

    Returns:
        ``{param_path: {"first_order": float, "total_order": float}}`` by
        default. When ``n_bootstrap`` is set, two extra keys per parameter
        are added: ``first_order_ci`` and ``total_order_ci``, each a
        ``(low, high)`` tuple at the requested confidence level.

    Memory characterisation:
        Fused mode allocates one parameter dict of shape ``((d+2)*N,)`` per
        key plus the corresponding stacked ``BatchSimulationResults`` —
        peak memory therefore scales like ``(d + 2) * N``.  At the default
        ``N=1024`` with ``d=10`` this is 12 288 evaluations in one batch;
        on a small dev box (e.g. heavy ODEs at high ``T``) reduce
        ``n_samples`` or set ``fused=False`` to drop back to the legacy
        per-matrix path.

    Tip:
        For very large ``N * (d + 2)``, prefer
        :func:`jaxonomy.simulate_distributed` over the in-process kernel path:
        wrap each Saltelli matrix as a separate distributed call.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    A, B, AB_list = saltelli_sample(distributions, n_samples, key)
    names = list(distributions.keys())
    d = len(names)
    N = n_samples

    if diagram is None:
        if fused:
            stacked = {
                name: jnp.concatenate(
                    [A[name], B[name]] + [AB_list[i][name] for i in range(d)]
                )
                for name in names
            }
            y = jnp.asarray(qoi_fn(stacked))
            yA = y[:N]
            yB = y[N:2 * N]
            yABs = [y[(2 + i) * N:(3 + i) * N] for i in range(d)]
        else:
            yA = jnp.asarray(qoi_fn(A))
            yB = jnp.asarray(qoi_fn(B))
            yABs = [jnp.asarray(qoi_fn(AB_i)) for AB_i in AB_list]
    else:
        from ..simulation import simulate_batch

        def _run(batches: dict[str, jnp.ndarray]) -> jnp.ndarray:
            res = simulate_batch(
                diagram,
                t_span,
                param_batches=batches,
                options=options,
                recorded_signals=recorded_signals,
            )
            return jnp.asarray(qoi_fn(res))

        if fused:
            stacked = {
                name: jnp.concatenate(
                    [A[name], B[name]] + [AB_list[i][name] for i in range(d)]
                )
                for name in names
            }
            y = _run(stacked)
            yA = y[:N]
            yB = y[N:2 * N]
            yABs = [y[(2 + i) * N:(3 + i) * N] for i in range(d)]
        else:
            yA = _run(A)
            yB = _run(B)
            yABs = [_run(AB_i) for AB_i in AB_list]

    out: dict[str, dict[str, float]] = {}
    if n_bootstrap is not None and n_bootstrap > 0:
        if not (0.0 < ci_level < 1.0):
            raise ValueError(
                f"sobol_indices: ci_level must satisfy 0 < ci_level < 1; "
                f"got {ci_level}."
            )
        # Derive a separate bootstrap key so the resampling does not
        # consume bits from the same PRNG stream used for Saltelli sampling.
        bootstrap_keys = jax.random.split(
            jax.random.fold_in(key, 0xB007),
            d,
        )
    for i, (name, yAB) in enumerate(zip(names, yABs)):
        s1, st = _saltelli_indices(yA, yB, yAB)
        entry: dict[str, Any] = {"first_order": s1, "total_order": st}
        if n_bootstrap is not None and n_bootstrap > 0:
            s1_ci, st_ci = _bootstrap_index_ci(
                yA, yB, yAB, n_bootstrap, ci_level, bootstrap_keys[i],
            )
            entry["first_order_ci"] = s1_ci
            entry["total_order_ci"] = st_ci
        out[name] = entry
    return out
