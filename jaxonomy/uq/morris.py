# SPDX-License-Identifier: MIT

"""Morris elementary-effect screening.

:func:`morris_screening` runs the standard Morris OAT (one-at-a-time)
trajectories — the cheap workhorse for first-pass parameter screening before
the more expensive :func:`sobol_indices`:

1. Discretise the unit cube into ``levels`` equally-spaced grid points per
   parameter; choose ``Δ = levels / (2 * (levels - 1))`` (a "balanced"
   step that visits every grid point with equal probability).
2. For each of ``n_trajectories`` random starting points, build a length
   ``d + 1`` trajectory by perturbing one randomly-ordered parameter at a
   time by ``±Δ``.
3. Map each unit-cube point through each parameter's PPF, evaluate the QoI
   at every node, and compute the elementary effect for each step:
   ``EE_ij = (Y(X + Δ) - Y(X)) / Δ`` (in unit-cube space, sign-corrected).
4. Per parameter, summarise the ``n_trajectories`` elementary effects with:

   * ``mu_star = mean(|EE|)`` — overall importance ranking.
   * ``sigma   = std(EE)``    — non-linearity / interaction strength.

Total model evaluations: ``n_trajectories * (d + 1)``.  Default
``n_trajectories=10`` is the SALib convention; reduce to 5 for quick checks
on small ``d``.

Example:

    >>> import jax  # doctest: +SKIP
    >>> from jaxonomy.uq import morris_screening, Uniform  # doctest: +SKIP
    >>> dists = {"x1": Uniform(0, 1), "x2": Uniform(0, 1), "x3": Uniform(0, 1)}
    >>> def qoi(p):
    ...     return p["x1"] + 5.0 * p["x2"]
    >>> morris_screening(None, None, dists, qoi, n_trajectories=10,
    ...                  key=jax.random.PRNGKey(0))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .distributions import Distribution

__all__ = [
    "morris_screening",
    "morris_sample",
]


# ---------------------------------------------------------------------------
# Trajectory construction (Campolongo-style random OAT)
# ---------------------------------------------------------------------------

def _build_trajectory(d: int, levels: int, key) -> np.ndarray:
    """Build one Morris trajectory in unit-cube space.

    Returns an ``(d + 1, d)`` array where row ``k+1`` differs from row ``k``
    in exactly one column, by ``±Δ``, with each parameter perturbed exactly
    once across the trajectory.
    """
    delta = levels / (2.0 * (levels - 1))
    grid_max = 1.0 - delta  # so that grid_max + Δ stays in [0, 1]

    k1, k2, k3 = jax.random.split(key, 3)
    # Random base point on the discrete grid.
    base_idx = jax.random.randint(k1, shape=(d,), minval=0, maxval=levels - 1)
    base = np.asarray(base_idx) / (levels - 1)
    # Clip so the +Δ step lands on the grid as well.
    base = np.minimum(base, grid_max)

    # Random sign per dimension.
    signs = np.where(np.asarray(jax.random.uniform(k2, shape=(d,))) < 0.5, -1.0, 1.0)
    # Flip base to the upper half if sign is -1 so that base + sign*Δ stays in [0,1].
    base = np.where(signs < 0, np.maximum(base, delta), base)

    # Random parameter ordering.
    order = np.asarray(jax.random.permutation(k3, jnp.arange(d)))

    traj = np.zeros((d + 1, d), dtype=np.float64)
    traj[0] = base
    current = base.copy()
    for step, j in enumerate(order):
        current = current.copy()
        current[j] = current[j] + signs[j] * delta
        traj[step + 1] = current
    return traj


def morris_sample(
    distributions: Mapping[str, Distribution],
    n_trajectories: int,
    levels: int,
    key,
) -> tuple[dict[str, jnp.ndarray], np.ndarray, np.ndarray]:
    """Build ``n_trajectories`` Morris trajectories.

    Args:
        distributions: Mapping ``param_path -> Distribution``.
        n_trajectories: Number of independent trajectories.
        levels: Grid resolution (must be even and >= 4).
        key: ``jax.random.PRNGKey``.

    Returns:
        ``(param_batches, deltas, perm_order)`` where:

        * ``param_batches`` has shape ``(n_trajectories * (d + 1),)`` per key —
          ready for :func:`simulate_batch`.
        * ``deltas`` is a ``(n_trajectories, d)`` array of ``±Δ`` values used
          for each parameter in each trajectory.
        * ``perm_order`` is a ``(n_trajectories, d)`` array of integer column
          indices giving the order in which parameters were perturbed.
    """
    if levels < 4 or levels % 2 != 0:
        raise ValueError(
            f"morris_sample: levels must be even and >= 4, got {levels}."
        )
    names = list(distributions.keys())
    d = len(names)
    keys = jax.random.split(key, n_trajectories)

    delta_val = levels / (2.0 * (levels - 1))

    # Build (n_trajectories, d+1, d) unit-cube samples and (n_trajectories, d)
    # signed deltas + perm orders.
    cube = np.zeros((n_trajectories, d + 1, d), dtype=np.float64)
    deltas = np.zeros((n_trajectories, d), dtype=np.float64)
    perm_order = np.zeros((n_trajectories, d), dtype=np.int32)
    for r in range(n_trajectories):
        traj = _build_trajectory(d, levels, keys[r])
        cube[r] = traj
        diffs = traj[1:] - traj[:-1]
        # Each row of diffs has exactly one nonzero entry equal to ±Δ.
        for k in range(d):
            j = int(np.argmax(np.abs(diffs[k])))
            perm_order[r, k] = j
            deltas[r, k] = np.sign(diffs[k, j]) * delta_val

    # Map cube columns through each parameter's PPF.
    flat_cube = cube.reshape(n_trajectories * (d + 1), d)
    param_batches: dict[str, jnp.ndarray] = {}
    for j, name in enumerate(names):
        u_col = jnp.asarray(flat_cube[:, j])
        param_batches[name] = distributions[name].ppf(u_col)

    return param_batches, deltas, perm_order


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def morris_screening(
    diagram,
    t_span,
    distributions: Mapping[str, Distribution],
    qoi_fn: Callable[..., Any],
    n_trajectories: int = 10,
    levels: int = 4,
    options=None,
    recorded_signals=None,
    key=None,
    fused: bool = True,
) -> dict[str, dict[str, float]]:
    """Morris OAT elementary-effect screening.

    Two execution modes match :func:`sobol_indices`:

    * **Analytic / surrogate mode** (``diagram is None``): ``qoi_fn`` accepts
      a ``param_batches`` dict and returns a ``(N,)`` array.
    * **Simulation mode** (``diagram`` is a Diagram): :func:`simulate_batch`
      is called once with all ``n_trajectories * (d + 1)`` evaluations
      stacked, and ``qoi_fn`` extracts the QoI from the
      :class:`BatchSimulationResults`.

    Args:
        diagram: Optional :class:`Diagram`.  When ``None`` runs in analytic
            mode.
        t_span: ``(t_start, t_stop)`` — simulation mode only.
        distributions: ``{param_path: Distribution}``.
        qoi_fn: Quantity of interest, see mode descriptions above.
        n_trajectories: Number of Morris trajectories. Default 10.  Reduce
            to 5 for quick screening on small ``d``.
        levels: Grid resolution (even, >= 4). Default 4.
        options: Forwarded to :func:`simulate_batch`.
        recorded_signals: Forwarded to :func:`simulate_batch`.
        key: ``jax.random.PRNGKey``.  Default ``PRNGKey(0)``.
        fused: When ``True`` (default), the elementary-effect aggregation is
            computed via vectorised numpy (no Python loop over
            ``(n_trajectories, d)``).  When ``False``, the legacy nested
            Python loop is used; bit-identical results either way.  The
            single-call ``simulate_batch`` invocation is the same in both
            paths — the choice only affects the index aggregation.

    Returns:
        ``{param_path: {"mu_star": float, "sigma": float}}``.

    Tip:
        Morris is *qualitative* — its main use is ranking parameters by
        importance to decide which subset to feed into the more expensive
        :func:`sobol_indices`.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    names = list(distributions.keys())
    d = len(names)
    n = n_trajectories

    param_batches, deltas, perm_order = morris_sample(
        distributions, n_trajectories, levels, key
    )

    if diagram is None:
        y = jnp.asarray(qoi_fn(param_batches))
    else:
        from ..simulation import simulate_batch

        res = simulate_batch(
            diagram,
            t_span,
            param_batches=param_batches,
            options=options,
            recorded_signals=recorded_signals,
        )
        y = jnp.asarray(qoi_fn(res))

    y_np = np.asarray(y).reshape(n, d + 1)

    if fused:
        # Vectorised elementary-effect aggregation.  ``diffs[r, k] = (y[r, k+1]
        # - y[r, k]) / deltas[r, k]`` is the EE attributed to the parameter
        # perturbed at step k of trajectory r — i.e. column ``perm_order[r, k]``.
        # Scatter into ``ee[r, perm_order[r, k]] = diffs[r, k]`` via fancy
        # indexing (``perm_order`` is a permutation per row, so the scatter is
        # injective).
        diffs = (y_np[:, 1:] - y_np[:, :-1]) / deltas  # (n, d)
        ee = np.empty((n, d), dtype=np.float64)
        row_idx = np.arange(n)[:, None]
        ee[row_idx, perm_order] = diffs
    else:
        # Legacy nested Python loop (kept for fused=False parity testing).
        # Elementary effects: per trajectory r and step k along
        # perm_order[r, k], EE_{r,perm_order[r,k]} = (y[r, k+1] - y[r, k])
        # / deltas[r, k].
        ee = np.zeros((n, d), dtype=np.float64)
        for r in range(n):
            for k in range(d):
                j = int(perm_order[r, k])
                ee[r, j] = (y_np[r, k + 1] - y_np[r, k]) / deltas[r, k]

    mu_star = np.mean(np.abs(ee), axis=0)
    sigma = np.std(ee, axis=0)
    return {
        name: {"mu_star": float(mu_star[j]), "sigma": float(sigma[j])}
        for j, name in enumerate(names)
    }
