# SPDX-License-Identifier: MIT

"""Lookup-table fitting.

Block-layer wrapper around the pure math in
:func:`jaxonomy.library.lookup_table.fit_table_1d`.

Public API:

- :func:`fit_lookup_table_1d` — fit an ``(x_data, y_data)`` cloud at a
  fixed grid ``xp`` via least-squares and return a fully-built
  ``LookupTable1d`` block ready to drop into a diagram.
- :func:`fit_table_1d_with_grid` — *jointly* optimise the grid
  placement ``xp`` AND the table values ``yp`` to minimise the data
  residual.  Outer loop on a monotonic-by-construction
  parametrisation of ``xp``; inner loop is the closed-form linear
  least-squares solve.
- :func:`fit_table_2d` — pure-functional bilinear LS solver that
  produces ``zp`` of shape ``(len(xp), len(yp))`` from a 3-tuple
  ``(x_data, y_data, z_data)`` measurement cloud.
- :func:`fit_lookup_table_2d` — block wrapper, returns a fully-built
  ``LookupTable2d``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxonomy.backend import numpy_api as npa

from .lookup_table import _build_linear_design, fit_table_1d


def fit_lookup_table_1d(
    xp,
    x_data,
    y_data,
    *,
    interpolation: str = "linear",
    extrapolation: str = "clip",
    weights=None,
    smoothness: float = 0.0,
    name: str | None = None,
    **block_kwargs,
):
    """Fit a 1-D lookup table to data and return a ``LookupTable1d`` block.

    Args:
        xp: Fixed grid of breakpoints (1-D, strictly increasing).
        x_data: Measured input cloud, shape ``(K,)``.
        y_data: Measured output cloud, shape ``(K,)``.
        interpolation: Interpolation rule for the *runtime* block
            (``"linear"`` / ``"pchip"`` / ``"nearest"`` / ``"flat"``).
            The fit itself is always linear-LS — see the module
            docstring of :mod:`jaxonomy.library.lookup_table` for why.
        extrapolation: Out-of-range policy for the runtime block; see
            :class:`jaxonomy.library.LookupTable1d`.
        weights: Optional per-sample weights for weighted least-
            squares.  ``None`` = OLS.
        smoothness: Non-negative discrete first-difference penalty.
            Use small values (1e-3 .. 1.0) on noisy / sparse data.
        name: Optional block name, forwarded to ``LookupTable1d``.
        **block_kwargs: Additional kwargs forwarded to the
            ``LookupTable1d`` constructor (e.g. ``dtype=``).

    Returns:
        A ``LookupTable1d`` instance with ``input_array=xp`` and
        ``output_array`` set to the LS-fit table values.
    """
    yp = fit_table_1d(
        xp,
        x_data,
        y_data,
        weights=weights,
        smoothness=smoothness,
    )
    # Lazy import: the block layer pulls in the rest of the library and
    # we don't want a fitting helper to drag that in at module-load
    # time.  This also keeps the import graph clean — the math module
    # has zero block-layer dependencies.
    from .primitives import LookupTable1d

    kwargs = dict(block_kwargs)
    if name is not None:
        kwargs["name"] = name
    return LookupTable1d(
        input_array=xp,
        output_array=yp,
        interpolation=interpolation,
        extrapolation=extrapolation,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# T-124-followup-grid-optimization — joint xp + yp optimisation.
#
# Phase 1's :func:`fit_table_1d` solves yp at a *fixed* user-supplied xp.
# That's already very useful (closed-form linear LS, exact through
# y_data) but on data with strong local features (e.g. a narrow peak)
# uniform breakpoints waste resolution in flat regions.  This follow-up
# adds a joint optimiser that ALSO moves the xp breakpoints to where
# they reduce the residual the most.
#
# The math: for any candidate xp, the inner problem is still linear in
# yp — solve via lstsq, get the optimal yp(xp).  Substitute back into
# the residual to get a function purely of xp:
#
#     R(xp) = || A(xp) @ lstsq(A(xp), y_data) - y_data ||²
#
# The outer loop minimises R(xp).  Three subtleties:
#
# 1. Monotonicity.  xp must stay strictly increasing.  Hard projection
#    (sort after each step) is non-smooth.  Instead we parametrise
#    xp via a softplus-of-deltas trick:
#
#        xp = x_lo + (x_hi - x_lo) * cumsum(softplus(d)) / sum(softplus(d))
#
#    This is differentiable everywhere, monotonically increasing in the
#    cumulative-sum direction, and respects [x_lo, x_hi] by construction.
#    The endpoints xp[0] = x_lo and xp[-1] = x_hi are pinned (a uniform
#    grid corresponds to all-equal deltas).
#
# 2. Inner LS solve.  The same ``_build_linear_design`` from phase 1 is
#    reused — the design matrix is a function of xp via the bucket
#    indices and alphas.  ``jnp.linalg.lstsq`` flows gradients through
#    both A (so xp) and b (so y_data).
#
# 3. Outer optimiser.  ``jax.scipy.optimize.minimize(method="BFGS")``
#    works for the forward pass but does not yet support
#    differentiation through itself.  For our differentiability story
#    (jax.grad of the loss-at-the-fitted-table w.r.t. y_data) we
#    instead ship a hand-rolled gradient-descent loop using
#    ``jax.lax.scan``: each step computes ``jax.grad(R)(deltas)`` and
#    takes a fixed-step-size move.  This is the "honest fallback"
#    flagged in the task spec — slower than a real L-BFGS but
#    differentiable end-to-end, jit-friendly, and reliable.  An
#    optional ``optimizer="lbfgs"`` path delegates to
#    ``jax.scipy.optimize.minimize`` for the forward fit when speed
#    matters more than backward-grad-through-the-fit.
# ---------------------------------------------------------------------------


def _xp_from_deltas(deltas, x_lo, x_hi):
    """Map an unconstrained ``deltas`` vector to a strictly-increasing
    ``xp`` array spanning ``[x_lo, x_hi]``.

    ``deltas`` has shape ``(n - 1,)``; the resulting ``xp`` has shape
    ``(n,)`` with ``xp[0] = x_lo`` and ``xp[-1] = x_hi``.

    The map is ``xp[i] = x_lo + (x_hi - x_lo) * c[i] / c[-1]`` where
    ``c = cumsum(softplus(deltas))`` and ``c[-1]`` is the total mass.
    Softplus(deltas) is strictly positive, so cumsum is strictly
    increasing → ``xp`` is strictly increasing.  All-equal deltas give
    a uniform grid; positive deltas in a region pull more breakpoints
    there.
    """
    pos = jax.nn.softplus(deltas)  # shape (n - 1,), strictly positive
    cum = jnp.cumsum(pos)
    total = cum[-1]
    interior = x_lo + (x_hi - x_lo) * cum[:-1] / total
    # Pin the endpoints exactly (avoid the cum[-1]/cum[-1] = 1 roundoff).
    return jnp.concatenate(
        [jnp.asarray(x_lo, dtype=interior.dtype)[None], interior, jnp.asarray(x_hi, dtype=interior.dtype)[None]]
    )


def _deltas_from_xp(xp_init, x_lo, x_hi):
    """Inverse of :func:`_xp_from_deltas` — recover ``deltas`` such that
    ``_xp_from_deltas(deltas, x_lo, x_hi) ≈ xp_init``.

    Used to seed the optimiser with a user-supplied initial grid.  Per
    the parametrisation, ``softplus(deltas) = (xp[i+1] - xp[i])`` up to
    a global scale (the sum is normalised away), so we set
    ``deltas = softplus_inv(xp_diffs)`` modulo the scale.

    We pick the scale so the smallest delta lands at softplus_inv(1) = log(e - 1) ≈ 0.541
    (a numerically comfortable value, well clear of softplus's
    near-zero saturation).
    """
    xp_init = jnp.asarray(xp_init)
    diffs = jnp.diff(xp_init)
    # Normalise so the minimum diff = 1.0; this gives a numerically
    # well-conditioned softplus_inv on every component.
    diffs = diffs / jnp.min(diffs)
    # softplus_inv(y) = log(exp(y) - 1) for y > 0.  Use log1p for stability.
    return jnp.log(jnp.expm1(diffs))


def _residual_at_xp(xp, x_data, y_data, smoothness):
    """Inner-loop closed-form LS solve + residual evaluation.

    Returns ``||A(xp) @ yp_opt - y_data||²`` where ``yp_opt`` is the
    least-squares-optimal table values at the given grid.  Differentiable
    through ``xp`` and ``y_data``.
    """
    A = _build_linear_design(xp, x_data)
    b = y_data
    if smoothness > 0:
        n = xp.shape[0]
        D = jnp.eye(n - 1, n, k=1, dtype=A.dtype) - jnp.eye(
            n - 1, n, dtype=A.dtype
        )
        A_full = jnp.concatenate([A, jnp.sqrt(smoothness) * D], axis=0)
        b_full = jnp.concatenate([b, jnp.zeros(n - 1, dtype=b.dtype)], axis=0)
    else:
        A_full = A
        b_full = b
    yp, *_ = jnp.linalg.lstsq(A_full, b_full, rcond=None)
    # Residual on the *data* portion only (smoothness is a regulariser,
    # not part of the data-fit metric).
    pred = A @ yp
    return jnp.sum((pred - y_data) ** 2)


def fit_table_1d_with_grid(
    n_grid_points: int,
    x_data,
    y_data,
    x_lo: float | None = None,
    x_hi: float | None = None,
    init_xp=None,
    *,
    smoothness: float = 0.0,
    optimizer: str = "gd",
    max_iter: int = 200,
    learning_rate: float = 1e-3,
    auto_normalize: bool = True,
):
    """Jointly optimise the grid ``xp`` AND the table values ``yp``.

    This is the T-124-followup-grid-optimization deliverable.  Phase 1's
    :func:`fit_table_1d` fits ``yp`` at a fixed user-supplied ``xp``;
    here we ALSO move the breakpoints to better resolve regions where
    the data has strong features (sharp peaks, kinks).

    The math: for any candidate grid ``xp``, the inner problem is still
    a linear least-squares solve for ``yp`` (closed form).  The outer
    loop minimises the resulting data residual w.r.t. ``xp``, with
    monotonicity enforced via a smooth ``cumsum(softplus(deltas))``
    parametrisation rather than projection.

    Args:
        n_grid_points: Number of breakpoints to place (must be ≥ 2).
        x_data: Measured input cloud, shape ``(K,)``.
        y_data: Measured output cloud, shape ``(K,)``.
        x_lo: Lower endpoint of the grid.  ``None`` (default) = ``min(x_data)``.
            The endpoint is *pinned* — the optimiser only moves the
            interior breakpoints.
        x_hi: Upper endpoint of the grid.  ``None`` (default) = ``max(x_data)``.
        init_xp: Optional initial grid (1-D, strictly increasing,
            spanning ``[x_lo, x_hi]``).  ``None`` (default) starts from
            a uniform grid.
        smoothness: Forwarded to the inner LS solve as a discrete
            first-difference penalty on ``yp``.  ``0.0`` (default) is
            pure data-residual.
        optimizer: ``"gd"`` (default) — hand-rolled fixed-step gradient
            descent on the unconstrained ``deltas``.  Differentiable
            end-to-end, jit-friendly, reliable.  ``"lbfgs"`` —
            delegates the outer loop to
            :func:`jax.scipy.optimize.minimize` (BFGS).  Faster on
            well-conditioned problems but does NOT support
            differentiation through itself (jax.grad of the joint fit
            w.r.t. y_data will fail with this option — use ``"gd"`` if
            you need the gradient).  See
            ``T-124-followup-grid-optimization-lbfgs`` for the proper
            differentiable L-BFGS implementation.
        max_iter: Outer-loop iteration budget.  For ``optimizer="gd"``
            each iter is one gradient step; for ``"lbfgs"`` it is the
            BFGS maxiter.
        learning_rate: Step size for ``optimizer="gd"``.  Default is
            ``1e-3`` — the residual landscape in ``deltas``-space has
            steep cliffs near sharp data features and aggressive step
            sizes overshoot.  Ignored by ``"lbfgs"``.  When
            ``auto_normalize=True`` (the default), the learning rate is
            applied in the normalised ``[-1, +1]`` x-space and ``[-1, +1]``
            y-space rather than in the user's natural units, so a single
            sensible default works across orders-of-magnitude data
            scales.
        auto_normalize: When ``True`` (default,
            T-124-followup-grid-fit-auto-normalize), the optimiser
            internally rescales ``x_data`` and ``y_data`` to roughly
            ``[-1, +1]`` (zero-mean, unit-half-range affine transform)
            so the ``learning_rate=1e-3`` default works on
            wide-but-smooth features (e.g. an engine-map slice with
            ``rpm ∈ [80, 650]`` and ``torque ~250 N·m``) without
            blowing up to NaN.  The optimised grid and table are
            transformed back to the user's natural units on return —
            results are byte-equivalent to the pre-normalisation path
            on data that was already centred near unit scale.  Set to
            ``False`` to disable (e.g. for byte-equivalent
            reproduction of pre-normalisation runs, or when the
            ``learning_rate`` was tuned in natural units).

    Returns:
        ``(xp_opt, yp_opt)`` — the optimised grid (shape
        ``(n_grid_points,)``) and the corresponding optimal table
        values (shape ``(n_grid_points,)``).  Both differentiable
        through ``y_data`` (and through ``x_data`` modulo the discrete
        bucket index in the design matrix) when
        ``optimizer="gd"``.

    Honest fallback note: this ships the gradient-descent path as the
    primary solver (rather than full L-BFGS) per the task spec — it's
    slower but more robust on the inner-outer formulation.  The proper
    differentiable L-BFGS via implicit-function-theorem unrolling is
    filed as ``T-124-followup-grid-optimization-lbfgs``.
    """
    if n_grid_points < 2:
        raise ValueError(
            f"fit_table_1d_with_grid: n_grid_points must be >= 2, got "
            f"{n_grid_points}"
        )
    if optimizer not in ("gd", "lbfgs"):
        raise ValueError(
            f"fit_table_1d_with_grid: unknown optimizer {optimizer!r}; expected "
            f"one of ('gd', 'lbfgs')"
        )
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    if x_data.shape != y_data.shape:
        raise ValueError(
            f"fit_table_1d_with_grid: x_data shape {x_data.shape} must match "
            f"y_data shape {y_data.shape}"
        )
    if x_data.ndim != 1:
        raise ValueError(
            f"fit_table_1d_with_grid: x_data must be 1-D, got shape "
            f"{x_data.shape}"
        )
    if smoothness < 0:
        raise ValueError(
            f"fit_table_1d_with_grid: smoothness must be >= 0, got {smoothness}"
        )

    # Resolve endpoints from data when not supplied.  Cast to the data
    # dtype so npa.float64 propagates through (T-005 default-float64).
    dtype = jnp.result_type(x_data, y_data)
    if x_lo is None:
        x_lo_v = jnp.min(x_data).astype(dtype)
    else:
        x_lo_v = jnp.asarray(x_lo, dtype=dtype)
    if x_hi is None:
        x_hi_v = jnp.max(x_data).astype(dtype)
    else:
        x_hi_v = jnp.asarray(x_hi, dtype=dtype)

    # T-124-followup-grid-fit-auto-normalize — affine-transform x and y
    # to roughly ``[-1, +1]`` so the default ``learning_rate=1e-3``
    # works across orders-of-magnitude data scales. Without this, a
    # wide-but-smooth feature (e.g. ``x ∈ [80, 650]``, ``y ~ 250``)
    # makes the residual landscape so steep in ``deltas``-space that
    # any non-trivial learning rate produces NaN gradients while safe
    # learning rates barely move the breakpoints.
    if auto_normalize:
        x_center = 0.5 * (x_lo_v + x_hi_v)
        x_half_range = 0.5 * (x_hi_v - x_lo_v)
        # Guard against degenerate ranges (all x_data identical); fall
        # back to no scaling rather than dividing by zero.
        x_half_range = jnp.where(x_half_range < 1e-30, jnp.asarray(1.0, dtype=dtype), x_half_range)
        y_center = jnp.mean(y_data).astype(dtype)
        y_half_range = jnp.maximum(
            jnp.max(jnp.abs(y_data - y_center)),
            jnp.asarray(1e-30, dtype=dtype),
        ).astype(dtype)
        x_data_norm = (x_data - x_center) / x_half_range
        y_data_norm = (y_data - y_center) / y_half_range
        x_lo_norm = (x_lo_v - x_center) / x_half_range
        x_hi_norm = (x_hi_v - x_center) / x_half_range
        # The smoothness penalty is on first differences of yp; under
        # the y-rescale by ``y_half_range``, the residual scales as
        # ``y_half_range^2`` and a fair smoothness penalty has to
        # rescale to match (otherwise the regulariser strength shifts
        # with the data scale). The user passes ``smoothness`` in
        # natural units; convert to the normalised space the inner
        # solve sees.
        smoothness_norm = smoothness  # cancels: both data residual
        # and penalty matrix-D scale the same way in normalised space.
    else:
        x_data_norm = x_data
        y_data_norm = y_data
        x_lo_norm = x_lo_v
        x_hi_norm = x_hi_v
        smoothness_norm = smoothness

    # Seed the outer-loop parameters (deltas).  A uniform grid
    # corresponds to all-equal deltas; the precise value is inverted
    # from the desired uniform spacing. ``init_xp`` is supplied in
    # natural units; convert to the normalised space when needed.
    if init_xp is None:
        init_xp_norm = jnp.linspace(x_lo_norm, x_hi_norm, n_grid_points)
    else:
        init_xp_v = jnp.asarray(init_xp, dtype=dtype)
        if init_xp_v.shape != (n_grid_points,):
            raise ValueError(
                f"fit_table_1d_with_grid: init_xp shape {init_xp_v.shape} must "
                f"be ({n_grid_points},)"
            )
        if auto_normalize:
            init_xp_norm = (init_xp_v - x_center) / x_half_range
        else:
            init_xp_norm = init_xp_v

    deltas0 = _deltas_from_xp(init_xp_norm, x_lo_norm, x_hi_norm).astype(dtype)

    def loss_fn(deltas):
        xp = _xp_from_deltas(deltas, x_lo_norm, x_hi_norm)
        return _residual_at_xp(xp, x_data_norm, y_data_norm, smoothness_norm)

    if optimizer == "gd":
        # Fixed-step gradient descent.  ``jax.lax.scan`` unrolls
        # cleanly under jit and supports the implicit-derivatives
        # path through ``jax.grad`` on the outer loss.
        grad_fn = jax.grad(loss_fn)

        def step(carry, _):
            d = carry
            g = grad_fn(d)
            d_new = d - learning_rate * g
            return d_new, None

        deltas_opt, _ = jax.lax.scan(step, deltas0, xs=None, length=max_iter)
    else:
        # optimizer == "lbfgs": forward-only path via jax.scipy BFGS.
        # Note: jax.scipy.optimize.minimize does not support
        # differentiation through itself (per its docstring), so this
        # branch breaks the grad-through-fit story.  Use "gd" if you
        # need that gradient.
        from jax.scipy.optimize import minimize as _jmin

        result = _jmin(
            loss_fn,
            deltas0,
            method="BFGS",
            options={"maxiter": int(max_iter)},
        )
        deltas_opt = result.x

    xp_opt_norm = _xp_from_deltas(deltas_opt, x_lo_norm, x_hi_norm)
    if auto_normalize:
        # Undo the affine x-transform so the returned grid lives in
        # natural units. The matching yp solve below uses the natural-
        # unit data, so the returned table values are in natural units
        # too (no extra y-transform needed on the output).
        xp_opt = xp_opt_norm * x_half_range + x_center
    else:
        xp_opt = xp_opt_norm
    # Final inner solve at the optimised grid — gives the matching yp.
    yp_opt = fit_table_1d(xp_opt, x_data, y_data, smoothness=smoothness)
    return xp_opt, yp_opt


# ---------------------------------------------------------------------------
# T-124-followup-2d-and-nd — bilinear lookup-table fitting in 2-D.
#
# For each measurement ``(x_data[k], y_data[k], z_data[k])`` the bilinear
# interp at the (xp, yp) grid is a sparse linear combination of the four
# z-values at the corners of the cell containing ``(x, y)``.  Stacking
# all measurements gives ``z_predicted = A @ z_flat`` where ``A`` has
# shape ``(K, Nx*Ny)`` with exactly four non-zeros per row and
# ``z_flat = zp.reshape(-1)``.  We solve via dense ``jnp.linalg.lstsq``
# — no native sparse lstsq in JAX today, but with typical grid sizes
# (Nx, Ny ~ 5-50, so Nx*Ny ~ 25-2500) the dense solve is fine.
#
# The smoothness penalty is a 5-point Laplacian: for each interior
# table cell ``(i, j)`` we add the row
# ``z[i,j] - 0.25 * (z[i-1,j] + z[i+1,j] + z[i,j-1] + z[i,j+1])``
# scaled by ``sqrt(λ)`` to the design matrix.  Edge / corner cells use
# the available subset of neighbours (the row penalises departure from
# the average of whatever neighbours exist).  Acts as a soft prior
# toward planar tables; ``λ = 0`` is pure data-fit.
#
# Differentiability: gradient flows through ``z_data`` exactly (the
# linear system is differentiable in its rhs), through ``x_data`` and
# ``y_data`` modulo the discrete bucket index (within a cell the
# bilinear weights are smooth in the queries), and through the table
# corner positions on the same terms.
# ---------------------------------------------------------------------------


def _build_bilinear_design(xp, yp, x_data, y_data):
    """Build the ``(K, Nx*Ny)`` bilinear-interp design matrix.

    Each row ``k`` has exactly four non-zero entries — the four corners
    of the cell containing ``(x_data[k], y_data[k])`` — with the
    standard bilinear weights ``(1-α)(1-β)``, ``α(1-β)``, ``(1-α)β``,
    ``αβ``.  Out-of-range queries clamp to the nearest grid edge
    (matching the ``"clip"`` extrapolation policy of
    :class:`LookupTable2d`).

    Convention: ``zp.reshape(-1)`` is row-major over ``(i, j)`` so
    column ``i*Ny + j`` corresponds to ``zp[i, j]``.
    """
    xp = jnp.asarray(xp)
    yp = jnp.asarray(yp)
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    nx = xp.shape[0]
    ny = yp.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError(
            f"fit_table_2d: need at least 2 grid points per axis, got "
            f"xp shape {xp.shape}, yp shape {yp.shape}"
        )
    if x_data.shape != y_data.shape:
        raise ValueError(
            f"fit_table_2d: x_data shape {x_data.shape} must match y_data "
            f"shape {y_data.shape}"
        )
    if x_data.ndim != 1:
        raise ValueError(
            f"fit_table_2d: x_data / y_data must be 1-D, got x shape "
            f"{x_data.shape}"
        )

    # Per-axis bucket indices, clipped so the "+1" neighbour stays in
    # range; the alpha clip below collapses out-of-range queries onto a
    # single endpoint with weight 1.
    i = jnp.clip(jnp.searchsorted(xp, x_data, side="right") - 1, 0, nx - 2)
    j = jnp.clip(jnp.searchsorted(yp, y_data, side="right") - 1, 0, ny - 2)

    x0 = xp[i]
    x1 = xp[i + 1]
    y0 = yp[j]
    y1 = yp[j + 1]
    alpha = jnp.clip((x_data - x0) / (x1 - x0), 0.0, 1.0)
    beta = jnp.clip((y_data - y0) / (y1 - y0), 0.0, 1.0)

    # Bilinear corner weights.
    w00 = (1.0 - alpha) * (1.0 - beta)
    w10 = alpha * (1.0 - beta)
    w01 = (1.0 - alpha) * beta
    w11 = alpha * beta

    # Flattened column indices for each corner (row-major: col = i*Ny + j).
    c00 = i * ny + j
    c10 = (i + 1) * ny + j
    c01 = i * ny + (j + 1)
    c11 = (i + 1) * ny + (j + 1)

    K = x_data.shape[0]
    k_idx = jnp.arange(K)
    A = jnp.zeros((K, nx * ny), dtype=jnp.result_type(xp, yp, x_data, y_data))
    A = A.at[k_idx, c00].add(w00)
    A = A.at[k_idx, c10].add(w10)
    A = A.at[k_idx, c01].add(w01)
    A = A.at[k_idx, c11].add(w11)
    return A


def _build_laplacian_2d(nx, ny, dtype):
    """Build the 5-point Laplacian smoothness operator on a flattened table.

    Returns an ``((nx*ny), (nx*ny))`` matrix ``L`` such that
    ``L @ z_flat`` gives, for each cell, ``z[i,j] - mean(neighbours)``.
    Edge and corner cells use the available subset of neighbours (the
    row sum is still ``0`` so a constant table has zero penalty).
    """
    n = nx * ny
    L = jnp.zeros((n, n), dtype=dtype)
    # Build via numpy then convert — index arithmetic only, no autograd.
    import numpy as _np

    L_np = _np.zeros((n, n), dtype=_np.float64)
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            neighbours = []
            if i > 0:
                neighbours.append((i - 1) * ny + j)
            if i < nx - 1:
                neighbours.append((i + 1) * ny + j)
            if j > 0:
                neighbours.append(i * ny + (j - 1))
            if j < ny - 1:
                neighbours.append(i * ny + (j + 1))
            L_np[row, row] = 1.0
            if neighbours:
                w = 1.0 / len(neighbours)
                for nb in neighbours:
                    L_np[row, nb] = -w
    return jnp.asarray(L_np, dtype=dtype)


def fit_table_2d(
    xp,
    yp,
    x_data,
    y_data,
    z_data,
    weights=None,
    smoothness: float = 0.0,
    rcond: float | None = None,
):
    """Fit a 2-D lookup table ``zp`` at the fixed grid ``(xp, yp)`` to
    ``(x_data, y_data, z_data)``.

    Solves the bilinear least-squares problem

        min_zp  Σ_k w_k * (z_data[k] - bilinear_interp(x_data[k], y_data[k]; xp, yp, zp))²
                + smoothness * Σ_{i,j} (zp[i,j] - mean(neighbours))²

    via :func:`jnp.linalg.lstsq` on the bilinear design matrix.  Linear
    bilinear only — the 2-D analogue of the ``fit_table_1d`` linear-only
    restriction.  See ``T-124-followup-2d-pchip-fit`` for non-linear
    extensions.

    Args:
        xp: 1-D, strictly increasing grid along the first axis (``Nx``).
        yp: 1-D, strictly increasing grid along the second axis (``Ny``).
        x_data, y_data, z_data: Measurement cloud, all shape ``(K,)``.
        weights: Optional per-sample weights, shape ``(K,)``.  ``None``
            means uniform weighting.
        smoothness: Non-negative coefficient for the 5-point Laplacian
            penalty.  ``0.0`` (default) is pure data-fit; small values
            (1e-3 .. 1.0) regularise on noisy / sparse data.
        rcond: Forwarded to :func:`jnp.linalg.lstsq`.

    Returns:
        ``zp`` of shape ``(len(xp), len(yp))`` — the optimal table
        values.  Differentiable through ``z_data`` (and through
        ``x_data`` / ``y_data`` modulo the discrete bucket indices).
    """
    xp = jnp.asarray(xp)
    yp = jnp.asarray(yp)
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    z_data = jnp.asarray(z_data)
    if z_data.shape != x_data.shape:
        raise ValueError(
            f"fit_table_2d: z_data shape {z_data.shape} must match x_data "
            f"shape {x_data.shape}"
        )
    if smoothness < 0:
        raise ValueError(
            f"fit_table_2d: smoothness must be >= 0, got {smoothness}"
        )

    A = _build_bilinear_design(xp, yp, x_data, y_data)
    b = z_data

    if weights is not None:
        w = jnp.asarray(weights)
        if w.shape != x_data.shape:
            raise ValueError(
                f"fit_table_2d: weights shape {w.shape} must match x_data "
                f"shape {x_data.shape}"
            )
        sqrt_w = jnp.sqrt(w)
        A = A * sqrt_w[:, None]
        b = b * sqrt_w

    nx = xp.shape[0]
    ny = yp.shape[0]
    if smoothness > 0:
        L = _build_laplacian_2d(nx, ny, A.dtype)
        A = jnp.concatenate([A, jnp.sqrt(smoothness) * L], axis=0)
        b = jnp.concatenate([b, jnp.zeros(nx * ny, dtype=b.dtype)], axis=0)

    z_flat, *_ = jnp.linalg.lstsq(A, b, rcond=rcond)
    return z_flat.reshape(nx, ny)


def fit_lookup_table_2d(
    xp,
    yp,
    x_data,
    y_data,
    z_data,
    *,
    interpolation: str = "linear",
    extrapolation: str = "clip",
    weights=None,
    smoothness: float = 0.0,
    name: str | None = None,
    **block_kwargs,
):
    """Fit a 2-D lookup table to data and return a ``LookupTable2d`` block.

    Args:
        xp: Fixed grid of breakpoints along the first axis (1-D,
            strictly increasing).
        yp: Fixed grid of breakpoints along the second axis (1-D,
            strictly increasing).
        x_data, y_data, z_data: Measurement cloud, all shape ``(K,)``.
        interpolation: Interpolation rule for the *runtime* block
            (currently only ``"linear"`` / bilinear).  The fit itself is
            always bilinear-LS.
        extrapolation: Out-of-range policy for the runtime block.
        weights: Optional per-sample weights for weighted least-squares.
        smoothness: Non-negative 5-point-Laplacian smoothness penalty.
        name: Optional block name.
        **block_kwargs: Additional kwargs forwarded to the
            ``LookupTable2d`` constructor (e.g. ``dtype=``).

    Returns:
        A ``LookupTable2d`` instance with ``input_x_array=xp``,
        ``input_y_array=yp``, and ``output_table_array`` set to the
        LS-fit table values of shape ``(len(xp), len(yp))``.
    """
    zp = fit_table_2d(
        xp,
        yp,
        x_data,
        y_data,
        z_data,
        weights=weights,
        smoothness=smoothness,
    )
    # Lazy import — same pattern as the 1-D wrapper.
    from .primitives import LookupTable2d

    kwargs = dict(block_kwargs)
    if name is not None:
        kwargs["name"] = name
    return LookupTable2d(
        input_x_array=xp,
        input_y_array=yp,
        output_table_array=zp,
        interpolation=interpolation,
        extrapolation=extrapolation,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# N-D fitter — generalises the 2-D bilinear LS to an arbitrary number of
# grid axes via the multilinear interpolation kernel that backs
# :class:`LookupTableND`.  Each measurement contributes 2**N corner
# weights to a dense ``(K, prod(B_i))`` design matrix; the smoothness
# penalty is the N-D analogue of the 5-point Laplacian (each cell
# penalised against the mean of its in-bounds neighbours).
# ---------------------------------------------------------------------------


def _build_multilinear_design(grid_axes, x_data):
    """Build the ``(K, prod(B_i))`` multilinear-interp design matrix.

    Args:
        grid_axes: Tuple of ``N`` strictly-increasing 1-D breakpoint
            arrays (each of length ``B_i``).
        x_data: Measurement query points, shape ``(K, N)``.

    Returns:
        Tuple ``(A, B, strides)``: the dense design matrix of shape
        ``(K, prod(B_i))``, the per-axis lengths tuple, and the
        C-order strides used to flatten ``output_array`` into ``A``.
        ``A.reshape(B)`` matches the layout of
        :class:`LookupTableND.output_array`.
    """
    x_data = jnp.asarray(x_data)
    if x_data.ndim != 2:
        raise ValueError(
            f"fit_table_nd: x_data must be 2-D with shape (K, N); got "
            f"{x_data.shape}."
        )
    N = len(grid_axes)
    if x_data.shape[1] != N:
        raise ValueError(
            f"fit_table_nd: x_data has {x_data.shape[1]} columns but "
            f"{N} grid axes were supplied."
        )

    B = tuple(int(jnp.asarray(g).shape[0]) for g in grid_axes)
    for d, b in enumerate(B):
        if b < 2:
            raise ValueError(
                f"fit_table_nd: grid_axes[{d}] needs at least 2 breakpoints, "
                f"got {b}."
            )
    total = int(np.prod(B))

    # C-order strides — last axis is fastest, matching ``reshape(B)``.
    strides = [1] * N
    for d in reversed(range(N - 1)):
        strides[d] = strides[d + 1] * B[d + 1]

    # Per-axis bucket index + interpolation fraction.
    i_per_axis = []
    alpha_per_axis = []
    for d in range(N):
        gp = jnp.asarray(grid_axes[d])
        xd = x_data[:, d]
        idx = jnp.clip(jnp.searchsorted(gp, xd, side="right") - 1, 0, B[d] - 2)
        g0 = gp[idx]
        g1 = gp[idx + 1]
        alpha = jnp.clip((xd - g0) / (g1 - g0), 0.0, 1.0)
        i_per_axis.append(idx)
        alpha_per_axis.append(alpha)

    K = x_data.shape[0]
    dtype = jnp.result_type(*[jnp.asarray(g) for g in grid_axes], x_data)
    A = jnp.zeros((K, total), dtype=dtype)
    k_idx = jnp.arange(K)

    # Enumerate the 2^N corners. Each corner's bitmask selects, per axis,
    # the lower (bit=0) or upper (bit=1) breakpoint of the cell.
    for corner in range(2 ** N):
        col_idx = jnp.zeros(K, dtype=jnp.int32)
        weight = jnp.ones(K, dtype=dtype)
        for d in range(N):
            bit = (corner >> d) & 1
            i_d = i_per_axis[d] + bit
            col_idx = col_idx + i_d * strides[d]
            if bit:
                weight = weight * alpha_per_axis[d]
            else:
                weight = weight * (1.0 - alpha_per_axis[d])
        A = A.at[k_idx, col_idx].add(weight)

    return A, B, tuple(strides)


def _build_laplacian_nd(B, dtype):
    """N-D analogue of the 2-D 5-point Laplacian smoothness operator.

    Each cell row has ``1.0`` on the diagonal and ``-1/k`` on its ``k``
    in-bounds neighbour columns (so the row sum is zero and a constant
    table has zero penalty).
    """
    N = len(B)
    total = int(np.prod(B))
    L_np = np.zeros((total, total), dtype=np.float64)

    strides = [1] * N
    for d in reversed(range(N - 1)):
        strides[d] = strides[d + 1] * B[d + 1]

    for coord in np.ndindex(*B):
        row = 0
        for d in range(N):
            row += coord[d] * strides[d]
        neighbours = []
        for d in range(N):
            if coord[d] > 0:
                nb = list(coord)
                nb[d] -= 1
                neighbours.append(nb)
            if coord[d] < B[d] - 1:
                nb = list(coord)
                nb[d] += 1
                neighbours.append(nb)
        L_np[row, row] = 1.0
        if neighbours:
            w = 1.0 / len(neighbours)
            for nb in neighbours:
                nb_idx = 0
                for d in range(N):
                    nb_idx += nb[d] * strides[d]
                L_np[row, nb_idx] = -w
    return jnp.asarray(L_np, dtype=dtype)


def fit_table_nd(
    grid_axes,
    x_data,
    y_data,
    *,
    weights=None,
    smoothness: float = 0.0,
    rcond: float | None = None,
):
    """Fit an N-D lookup table at fixed grid breakpoints.

    Solves the multilinear least-squares problem

        min_zp  Σ_k w_k * (y_data[k] - multilinear_interp(x_data[k]; grid_axes, zp))²
                + smoothness * Σ_cells (zp[cell] - mean(in-bounds neighbours))²

    via :func:`jnp.linalg.lstsq` on the multilinear design matrix.
    Generalises :func:`fit_table_2d` (and :func:`fit_table_1d` for
    ``N=1``) to an arbitrary number of grid axes.

    Args:
        grid_axes: Tuple of ``N`` strictly-increasing 1-D breakpoint
            arrays. Axis ``d`` has length ``B_d``.
        x_data: Query points, shape ``(K, N)``. Column ``d`` is the
            ``d``-th coordinate.
        y_data: Sample values at the query points, shape ``(K,)``.
        weights: Optional per-sample weights, shape ``(K,)``. ``None``
            means uniform weighting.
        smoothness: Non-negative coefficient on the N-D-Laplacian
            penalty. ``0.0`` (default) is a pure data fit; small values
            (1e-3 .. 1.0) regularise on noisy / sparse data. The
            Laplacian is the canonical smoother on a regular grid — far
            better than diagonal Tikhonov, especially for cells without
            nearby measurements.
        rcond: Forwarded to :func:`jnp.linalg.lstsq`.

    Returns:
        ``zp`` of shape ``(B_1, ..., B_N)`` — the optimal table values.
        Layout matches :class:`LookupTableND.output_array` exactly, so
        the result can be passed straight through.

    Memory note: builds a dense ``(K + prod(B_i), prod(B_i))`` design
    matrix. For ``N=5`` with ``B_i = 10`` that's ``10^5`` columns —
    fine on CPU up to a few thousand measurements. For larger tables
    or higher-D problems, switch to a sparse solver (filed under
    ``T-104-followup-fit-table-nd-sparse``).
    """
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    if y_data.shape != (x_data.shape[0],):
        raise ValueError(
            f"fit_table_nd: y_data shape {y_data.shape} must equal "
            f"(K,) = ({x_data.shape[0]},)."
        )
    if smoothness < 0:
        raise ValueError(
            f"fit_table_nd: smoothness must be >= 0, got {smoothness}"
        )

    A, B, _strides = _build_multilinear_design(grid_axes, x_data)
    b = y_data

    if weights is not None:
        w = jnp.asarray(weights)
        if w.shape != (x_data.shape[0],):
            raise ValueError(
                f"fit_table_nd: weights shape {w.shape} must equal "
                f"(K,) = ({x_data.shape[0]},)."
            )
        sqrt_w = jnp.sqrt(w)
        A = A * sqrt_w[:, None]
        b = b * sqrt_w

    if smoothness > 0:
        L = _build_laplacian_nd(B, A.dtype)
        A = jnp.concatenate([A, jnp.sqrt(smoothness) * L], axis=0)
        b = jnp.concatenate([b, jnp.zeros(L.shape[0], dtype=b.dtype)], axis=0)

    z_flat, *_ = jnp.linalg.lstsq(A, b, rcond=rcond)
    return z_flat.reshape(B)


def fit_lookup_table_nd(
    grid_axes,
    x_data,
    y_data,
    *,
    interpolation: str = "linear",
    extrapolation: str = "clip",
    weights=None,
    smoothness: float = 0.0,
    name: str | None = None,
    **block_kwargs,
):
    """Fit an N-D lookup table to data and return a ``LookupTableND`` block.

    The public N-D counterpart to :func:`fit_lookup_table_1d` and
    :func:`fit_lookup_table_2d`.  Returns a fully-built block whose
    ``output_array`` is the LS-fit table.

    Args:
        grid_axes: Tuple of ``N`` strictly-increasing 1-D breakpoint
            arrays.
        x_data, y_data: Measurement cloud — ``x_data`` shape ``(K, N)``,
            ``y_data`` shape ``(K,)``.
        interpolation: Interpolation rule for the *runtime* block. Only
            ``"linear"`` (multilinear) is supported today; the fit
            itself is always multilinear-LS.
        extrapolation: Out-of-range policy for the runtime block; see
            :class:`jaxonomy.library.LookupTableND`.
        weights: Optional per-sample weights for weighted least-squares.
        smoothness: Non-negative coefficient on the N-D Laplacian
            smoothness penalty.
        name: Optional block name.
        **block_kwargs: Additional kwargs forwarded to the
            ``LookupTableND`` constructor (e.g. ``dtype=``).

    Returns:
        A :class:`LookupTableND` instance with the supplied ``grid_axes``
        and ``output_array`` set to the LS-fit table of shape
        ``(len(grid_axes[0]), ..., len(grid_axes[N-1]))``.
    """
    zp = fit_table_nd(
        grid_axes,
        x_data,
        y_data,
        weights=weights,
        smoothness=smoothness,
    )
    from .primitives import LookupTableND

    kwargs = dict(block_kwargs)
    if name is not None:
        kwargs["name"] = name
    return LookupTableND(
        grid_axes=tuple(grid_axes),
        output_array=zp,
        interpolation=interpolation,
        extrapolation=extrapolation,
        **kwargs,
    )


# Touch ``npa`` so the import is preserved as a forward-looking hook
# for the L-BFGS / numpy-fallback follow-up; pruning it now would just
# force a re-add later.
_ = npa


__all__ = [
    "fit_lookup_table_1d",
    "fit_lookup_table_2d",
    "fit_lookup_table_nd",
    "fit_table_1d_with_grid",
    "fit_table_2d",
    "fit_table_nd",
]
