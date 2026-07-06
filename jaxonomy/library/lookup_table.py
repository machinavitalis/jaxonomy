# SPDX-License-Identifier: MIT

"""T-106 / T-114 — Pure-functional lookup-table backend.

This module is the shared, JAX-friendly backend for the T-114
``LookupTable*`` blocks (and any advanced user who wants direct
access to the interpolation primitives).  Phase 1 ships:

- ``interp_1d(x, xp, fp, method, extrapolation)`` — the only public
  primitive needed by the 1-D block today.  Supports
  ``method in {"linear", "pchip", "nearest", "flat"}`` and
  ``extrapolation in {"clip", "linear", "nan"}``.
- ``even_spacing(xp, atol, rtol)`` — auto-detect uniformly spaced
  breakpoints (the "Evenly spaced" lookup optimization).  Returns
  ``(is_even, dx)`` where ``dx`` is ``None`` if the array is not
  evenly spaced.
- ``pchip_slopes(xp, fp)`` — Hyman/Fritsch-Carlson monotone slope
  computation.  Exposed so callers (e.g. T-124) can precompute
  coefficients ahead of time.

The block layer (``primitives.LookupTable1d``) is the user-facing
surface; this module is the math.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp


_INTERP_METHODS = ("linear", "pchip", "akima", "cubic", "nearest", "flat")
_EXTRAP_MODES = ("clip", "linear", "nan")
_INTERP_2D_METHODS = ("linear", "bicubic")


def even_spacing(xp, atol: float = 1e-10, rtol: float = 1e-7):
    """Return ``(is_even, dx)`` for ``xp``.

    ``is_even`` is a Python ``bool`` (computed eagerly via
    ``numpy.asarray`` — the breakpoints are *static* parameters in
    every block today, so this is safe) and ``dx`` is the common
    spacing if even, else ``None``.
    """
    import numpy as _np

    arr = _np.asarray(xp)
    if arr.ndim != 1 or arr.size < 2:
        return False, None
    diffs = _np.diff(arr)
    dx = float(diffs[0])
    if not _np.allclose(diffs, dx, atol=atol, rtol=rtol):
        return False, None
    return True, dx


def pchip_slopes(xp, fp):
    """Compute Hyman/Fritsch-Carlson monotone PCHIP slopes at ``xp``.

    Mirrors :class:`scipy.interpolate.PchipInterpolator` for the 1-D
    case.  Returned slopes preserve monotonicity on monotone data and
    drive the per-interval Hermite cubic in :func:`interp_1d`.
    """
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    h = jnp.diff(xp)
    delta = jnp.diff(fp) / h

    # Interior slopes via the weighted-harmonic-mean formula.  Where
    # neighbouring secants have opposite sign (or either is zero), the
    # slope is forced to zero — this is what enforces monotonicity.
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    same_sign = jnp.sign(delta[:-1]) * jnp.sign(delta[1:]) > 0
    # Guard against div-by-zero in the weighted mean: replace zero
    # secants with 1.0 in the denominator and rely on ``same_sign``
    # to mask the result back to zero.
    safe_d0 = jnp.where(delta[:-1] == 0, 1.0, delta[:-1])
    safe_d1 = jnp.where(delta[1:] == 0, 1.0, delta[1:])
    interior = (w1 + w2) / (w1 / safe_d0 + w2 / safe_d1)
    interior = jnp.where(same_sign, interior, 0.0)

    # Endpoint slopes via the three-point Hyman one-sided formula
    # (also matches SciPy).
    def _edge(h0, h1, d0, d1):
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        # Monotonicity-preserving sign clip.
        sign_ok = jnp.sign(slope) == jnp.sign(d0)
        slope = jnp.where(sign_ok, slope, 0.0)
        too_big = jnp.abs(slope) > 3.0 * jnp.abs(d0)
        slope = jnp.where(too_big, 3.0 * d0, slope)
        return slope

    left = _edge(h[0], h[1], delta[0], delta[1])
    right = _edge(h[-1], h[-2], delta[-1], delta[-2])
    return jnp.concatenate([left[None], interior, right[None]])


def akima_slopes(xp, fp):
    """Compute Akima 1970 slopes at ``xp`` for cubic Hermite evaluation.

    Mirrors :class:`scipy.interpolate.Akima1DInterpolator` (the original
    1970 variant — SciPy's default, not the 2009 ``"makima"`` modification).
    Returned slopes feed the same Hermite cubic evaluator used for PCHIP;
    only the slope rule differs.

    Boundary slopes use Akima's standard reflection formula: the secant
    sequence is extended by two phantom points on each side via
    ``m_{-1} = 2 m_0 - m_1`` and ``m_{-2} = 2 m_{-1} - m_0`` (and likewise
    on the right).  This matches SciPy.
    """
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    h = jnp.diff(xp)
    m = jnp.diff(fp) / h  # shape (n - 1,)

    # Extend the secants ``m`` with two phantom points on each side per
    # Akima's reflection formula.  Resulting ``mm`` has length n + 3.
    m_lm1 = 2.0 * m[0] - m[1]
    m_lm2 = 2.0 * m_lm1 - m[0]
    m_rp1 = 2.0 * m[-1] - m[-2]
    m_rp2 = 2.0 * m_rp1 - m[-1]
    mm = jnp.concatenate(
        [m_lm2[None], m_lm1[None], m, m_rp1[None], m_rp2[None]]
    )

    # Akima slope at point ``i`` = weighted average of m_{i-1} and m_i,
    # weighted by |m_{i+1} - m_i| and |m_{i-1} - m_{i-2}| respectively.
    # Indices into ``mm``: m_{i-2}=mm[i], m_{i-1}=mm[i+1], m_i=mm[i+2],
    # m_{i+1}=mm[i+3], for i in [0, n-1].
    n = xp.shape[0]
    idx = jnp.arange(n)
    w_left = jnp.abs(mm[idx + 3] - mm[idx + 2])
    w_right = jnp.abs(mm[idx + 1] - mm[idx])
    denom = w_left + w_right
    # When denom == 0 (locally flat), Akima sets slope to the average of
    # the two flanking secants — i.e. (m_{i-1} + m_i) / 2.  This matches
    # SciPy and avoids division by zero.
    safe_denom = jnp.where(denom == 0, 1.0, denom)
    weighted = (w_left * mm[idx + 1] + w_right * mm[idx + 2]) / safe_denom
    average = 0.5 * (mm[idx + 1] + mm[idx + 2])
    return jnp.where(denom == 0, average, weighted)


def natural_cubic_second_derivs(xp, fp):
    """Compute second derivatives ``M`` of the natural cubic spline at ``xp``.

    T-114-followup-natural-cubic-spline.

    The natural cubic spline is the C^2-continuous piecewise cubic that
    interpolates ``(xp, fp)`` and satisfies the *natural* boundary
    condition ``M[0] = M[N-1] = 0`` — i.e. the second derivative vanishes
    at both ends.  Among all C^2 interpolants this minimises the
    integrated square of the second derivative (the "smoothest
    interpolant" property).  Distinct from PCHIP (which prevents
    overshoot at the cost of C^1) and Akima (which damps overshoot
    moderately, also C^1).

    Implementation: the standard tridiagonal system on the interior
    second derivatives ``M[1..N-2]``.  Letting ``h_i = xp[i+1] - xp[i]``,

        h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1}
            = 6 ((fp_{i+1} - fp_i)/h_i - (fp_i - fp_{i-1})/h_{i-1})

    for ``i = 1 .. N-2``, with ``M_0 = M_{N-1} = 0`` (natural BC).  We
    build the dense ``(N-2, N-2)`` tridiagonal matrix and solve via
    :func:`jnp.linalg.solve`; for the typical lookup-table grid sizes
    (N ~ 10 .. 100) this is comfortable and the dense path is trivially
    differentiable through ``fp`` (and through ``xp``, modulo the
    monotonicity precondition).

    Returns:
        ``M`` of shape ``(N,)`` with ``M[0] = M[N-1] = 0``.

    Raises:
        ValueError: if ``len(xp) < 4``.  Natural cubic spline degenerates
            on 3 points (the unique interior equation gives a single
            interior second derivative; below that the system is empty).
            We require at least 4 to keep the spline meaningful.
    """
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    n = int(xp.shape[0])
    if n < 4:
        raise ValueError(
            f"natural_cubic_second_derivs: need at least 4 breakpoints for "
            f"natural cubic spline, got xp of shape {xp.shape}"
        )

    h = jnp.diff(xp)  # shape (n - 1,)
    delta = jnp.diff(fp) / h  # shape (n - 1,) — divided differences

    # Interior tridiagonal system: size (n - 2).
    # diag[i] = 2 * (h[i] + h[i+1]) for i = 0 .. n-3
    # sub[i]  = h[i+1]              for i = 0 .. n-4 (below-diagonal)
    # sup[i]  = h[i+1]              for i = 0 .. n-4 (above-diagonal)
    # rhs[i]  = 6 * (delta[i+1] - delta[i])
    main = 2.0 * (h[:-1] + h[1:])
    off = h[1:-1]
    rhs = 6.0 * (delta[1:] - delta[:-1])

    # Build the dense (n-2, n-2) tridiagonal — JAX has no banded solve in
    # the public API today, so dense ``linalg.solve`` is the
    # differentiable-everywhere fallback.  N <= ~100 in realistic
    # lookup-table use cases keeps this cheap.
    A = jnp.diag(main) + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    M_interior = jnp.linalg.solve(A, rhs)

    # Pad with the natural-BC zeros at both ends.
    zero = jnp.zeros((), dtype=M_interior.dtype)
    return jnp.concatenate([zero[None], M_interior, zero[None]])


def _natural_cubic_eval(x, xp, fp, M):
    """Evaluate the natural cubic spline at ``x`` given knot second derivs ``M``.

    Within the cell ``[xp[i], xp[i+1]]`` the cubic in standard form is

        S(x) = M_i (x_{i+1} - x)^3 / (6 h_i)
             + M_{i+1} (x - x_i)^3 / (6 h_i)
             + (fp_i / h_i - M_i h_i / 6) (x_{i+1} - x)
             + (fp_{i+1} / h_i - M_{i+1} h_i / 6) (x - x_i)

    with ``h_i = xp[i+1] - xp[i]``.  At ``x = xp[i]`` this reduces to
    ``fp_i`` and at ``x = xp[i+1]`` to ``fp_{i+1}``; the second derivative
    interpolates ``M`` linearly across the cell.  C^2 across cell
    boundaries by construction (since ``M`` is shared between adjacent
    cells and the tridiagonal system enforces continuity of S').
    """
    n = xp.shape[0]
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, n - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    y0 = fp[i]
    y1 = fp[i + 1]
    M0 = M[i]
    M1 = M[i + 1]
    h = x1 - x0
    a = x1 - x
    b = x - x0
    return (
        M0 * a * a * a / (6.0 * h)
        + M1 * b * b * b / (6.0 * h)
        + (y0 / h - M0 * h / 6.0) * a
        + (y1 / h - M1 * h / 6.0) * b
    )


def _pchip_eval(x, xp, fp, slopes):
    # Locate the interval [xp[i], xp[i+1]] containing x.  ``i`` is
    # clipped to ``[0, n-2]`` so the cubic evaluates the boundary
    # piece for OOB inputs; the caller (``interp_1d``) overrides the
    # OOB result via the requested extrapolation mode.
    n = xp.shape[0]
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, n - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    y0 = fp[i]
    y1 = fp[i + 1]
    m0 = slopes[i]
    m1 = slopes[i + 1]
    h = x1 - x0
    t = (x - x0) / h
    # Standard Hermite cubic basis.
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t * t * (3.0 - 2.0 * t)
    h11 = t * t * (t - 1.0)
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def _linear_extrap(x, xp, fp, base):
    """Replace ``base`` (which is the clipped jnp.interp result) with
    a slope-extension on either side when ``x`` falls outside ``xp``.
    """
    left_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    right_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    left_val = fp[0] + left_slope * (x - xp[0])
    right_val = fp[-1] + right_slope * (x - xp[-1])
    out = jnp.where(x < xp[0], left_val, base)
    out = jnp.where(x > xp[-1], right_val, out)
    return out


def _apply_extrapolation(x, xp, base, fp, mode: str):
    if mode == "clip":
        return base
    if mode == "linear":
        return _linear_extrap(x, xp, fp, base)
    if mode == "nan":
        nan_val = jnp.asarray(jnp.nan, dtype=base.dtype)
        out = jnp.where(x < xp[0], nan_val, base)
        out = jnp.where(x > xp[-1], nan_val, out)
        return out
    raise ValueError(
        f"unknown extrapolation mode {mode!r}; expected one of {_EXTRAP_MODES}"
    )


def interp_1d(
    x,
    xp,
    fp,
    method: Literal[
        "linear", "pchip", "akima", "cubic", "nearest", "flat"
    ] = "linear",
    extrapolation: Literal["clip", "linear", "nan"] = "clip",
):
    """1-D interpolation backend.

    Args:
        x: Query point(s).  Scalar or array of any shape.
        xp: Strictly increasing 1-D breakpoint array.
        fp: Sample values, shape ``(len(xp),)``.
        method: Interpolation rule. ``"linear"`` is the default
            matching established block-diagram lookup tables; ``"pchip"`` is monotone cubic
            (smooth gradients everywhere — recommended for
            optimization through engine/aero/battery maps); ``"akima"``
            is the Akima 1970 cubic spline (smoother than PCHIP on
            non-monotone data, less prone to overshoot than natural
            cubic splines — matches
            :class:`scipy.interpolate.Akima1DInterpolator`).
            ``"cubic"`` is the natural cubic spline (T-114-followup-
            natural-cubic-spline) — C^2-continuous, second derivative
            zero at the boundaries, the smoothest possible C^2
            interpolant.  Requires at least 4 breakpoints; matches
            :class:`scipy.interpolate.CubicSpline` with ``bc_type='natural'``.
            ``"nearest"`` and ``"flat"`` are zero-gradient
            piecewise-constant rules.
        extrapolation: Out-of-range behaviour. ``"clip"`` holds the
            boundary value (matches :func:`jnp.interp` and the
            standard lookup-table default); ``"linear"`` extends the boundary slope;
            ``"nan"`` returns NaN outside ``[xp[0], xp[-1]]``.

    Returns:
        Interpolated value(s) with the same shape as ``x``.
    """
    if method not in _INTERP_METHODS:
        raise ValueError(
            f"interp_1d: unknown method {method!r}; expected one of "
            f"{_INTERP_METHODS}"
        )
    if extrapolation not in _EXTRAP_MODES:
        raise ValueError(
            f"interp_1d: unknown extrapolation {extrapolation!r}; expected "
            f"one of {_EXTRAP_MODES}"
        )
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    x = jnp.asarray(x)

    if method == "linear":
        base = jnp.interp(x, xp, fp)
    elif method == "pchip":
        slopes = pchip_slopes(xp, fp)
        # Inside the breakpoint range the cubic is the answer; outside
        # it, ``_pchip_eval`` evaluates the boundary piece, which is
        # what ``"clip"`` should *not* return — instead clip uses the
        # exact endpoint value.  Build a clipped base accordingly.
        raw = _pchip_eval(x, xp, fp, slopes)
        in_range = (x >= xp[0]) & (x <= xp[-1])
        boundary = jnp.where(x < xp[0], fp[0], fp[-1])
        base = jnp.where(in_range, raw, boundary)
    elif method == "akima":
        # Akima reuses the same Hermite cubic evaluator as PCHIP — only
        # the slope rule differs.  OOB handling mirrors the PCHIP path:
        # evaluate the boundary piece for the cubic, then mask to the
        # endpoint value so ``"clip"`` extrapolation returns the exact
        # endpoint rather than the cubic continuation.
        slopes = akima_slopes(xp, fp)
        raw = _pchip_eval(x, xp, fp, slopes)
        in_range = (x >= xp[0]) & (x <= xp[-1])
        boundary = jnp.where(x < xp[0], fp[0], fp[-1])
        base = jnp.where(in_range, raw, boundary)
    elif method == "cubic":
        # T-114-followup-natural-cubic-spline.  Solve for second
        # derivatives at the knots (natural BC: M[0] = M[-1] = 0), then
        # evaluate the per-cell cubic.  OOB handling mirrors the PCHIP /
        # Akima paths so ``"clip"`` returns the exact endpoint value.
        if int(xp.shape[0]) < 4:
            raise ValueError(
                f"interp_1d: method='cubic' requires at least 4 breakpoints "
                f"for a natural cubic spline, got xp of shape {xp.shape}"
            )
        M = natural_cubic_second_derivs(xp, fp)
        raw = _natural_cubic_eval(x, xp, fp, M)
        in_range = (x >= xp[0]) & (x <= xp[-1])
        boundary = jnp.where(x < xp[0], fp[0], fp[-1])
        base = jnp.where(in_range, raw, boundary)
    elif method == "nearest":
        # Nearest breakpoint, OOB clamps to the nearest endpoint.
        # Reshape preserves the original ``x`` shape for both scalar
        # and array inputs (the 2-D broadcast below adds a trailing
        # axis we must remove afterwards).
        flat_x = jnp.atleast_1d(x).reshape(-1)
        i = jnp.argmin(jnp.abs(xp[None, :] - flat_x[:, None]), axis=-1)
        i = jnp.clip(i, 0, xp.shape[0] - 1)
        base = fp[i].reshape(jnp.shape(x))
    else:  # method == "flat"
        # Hold-from-the-left: at xp[i] use fp[i], between use fp[i].
        i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 1)
        base = fp[i]

    return _apply_extrapolation(x, xp, base, fp, extrapolation)


# ---------------------------------------------------------------------------
# T-114 phase 2 — 2-D bilinear lookup with extrapolation policy.
#
# Mirrors the 1-D ``interp_1d`` API for the 2-D case so the
# ``LookupTable2d`` block can opt in to ``"linear"`` / ``"nan"``
# extrapolation without breaking the historical ``"clip"`` default
# (which is what the legacy ``npa.interp2d`` already does).
# ---------------------------------------------------------------------------


def _bilinear_eval(x, y, xp, yp, zp):
    """Bilinear interpolation kernel — assumes ``x``, ``y`` already
    clipped into the breakpoint range.  Returns the raw bilinear value
    at every requested ``(x, y)``; the caller decides what to do with
    out-of-range queries (clip / linear-extend / NaN).
    """
    nx = xp.shape[0]
    ny = yp.shape[0]
    # Bucket indices ``[1, n - 1]`` so the ``ix - 1`` / ``iy - 1``
    # neighbours stay in range.  Same convention as ``npa.interp2d``.
    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, nx - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, ny - 1)

    x0 = xp[ix - 1]
    x1 = xp[ix]
    y0 = yp[iy - 1]
    y1 = yp[iy]

    z11 = zp[ix - 1, iy - 1]
    z21 = zp[ix, iy - 1]
    z12 = zp[ix - 1, iy]
    z22 = zp[ix, iy]

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    z_y0 = z11 * (1.0 - tx) + z21 * tx
    z_y1 = z12 * (1.0 - tx) + z22 * tx
    return z_y0 * (1.0 - ty) + z_y1 * ty


def _apply_extrapolation_2d(x, y, xp, yp, base, raw, mode: str):
    """Apply the requested OOB policy to a 2-D bilinear result.

    ``base`` is the *clipped* bilinear value (xs/ys clamped into the
    grid).  ``raw`` is the bilinear value computed *without* clipping —
    i.e. evaluating the boundary piece's linear extrapolation past the
    edge.  This is the bilinear equivalent of ``_linear_extrap`` for the
    1-D case: outside the grid each face's edge slope is preserved.
    """
    if mode == "clip":
        return base
    in_x = (x >= xp[0]) & (x <= xp[-1])
    in_y = (y >= yp[0]) & (y <= yp[-1])
    in_range = in_x & in_y
    if mode == "linear":
        return jnp.where(in_range, base, raw)
    if mode == "nan":
        nan_val = jnp.asarray(jnp.nan, dtype=base.dtype)
        return jnp.where(in_range, base, nan_val)
    raise ValueError(
        f"unknown extrapolation mode {mode!r}; expected one of {_EXTRAP_MODES}"
    )


# ---------------------------------------------------------------------------
# T-114-followup-2d-bicubic — bicubic interpolation (Catmull-Rom).
#
# Hand-rolled JAX-native 4x4 tensor-product Catmull-Rom kernel.  Why not
# scipy's RectBivariateSpline?  Because RBS is a Python callback into
# Fortran, so it can't be jit-compiled or vmapped, and it's not natively
# differentiable through the table values.  The Catmull-Rom path below
# is pure ``jnp`` ops — differentiable end-to-end and traceable.
#
# Catmull-Rom (the cardinal spline with tension 0.5, equivalent to
# Keys' a=-0.5 cubic convolution kernel) is the standard choice for
# smooth interpolation on a uniform grid.  For NON-uniform grids we
# fall back to "uniform-equivalent" parametric Catmull-Rom by
# normalising each cell to t in [0, 1] — this matches what
# ``RectBivariateSpline`` does to leading order and keeps the C^1
# smoothness property at cell boundaries.
#
# The kernel uses 4 surrounding grid points per axis.  At the boundary
# (i = 0 or i = N - 2 cell), the missing "ghost" rows/cols are
# extrapolated via linear reflection so the cubic doesn't blow up.
# This is the standard SciPy approach and matches
# ``RectBivariateSpline`` to within boundary effects.
# ---------------------------------------------------------------------------


def _catmull_rom_weights(t):
    """Catmull-Rom (Keys' a=-0.5) cubic basis weights at parameter t in [0, 1].

    Returns a length-4 array ``(w_{-1}, w_0, w_1, w_2)`` such that the
    interpolated value at parameter ``t`` is
    ``w_{-1} * f_{-1} + w_0 * f_0 + w_1 * f_1 + w_2 * f_2``.  The kernel
    interpolates ``f_0`` at ``t=0`` and ``f_1`` at ``t=1`` and is
    C^1-continuous across cell boundaries.
    """
    t2 = t * t
    t3 = t2 * t
    # Standard Catmull-Rom basis (Keys 1981, a = -1/2).
    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_2 = 0.5 * t3 - 0.5 * t2
    return jnp.stack([w_m1, w_0, w_1, w_2], axis=-1)


def _bicubic_eval(x, y, xp, yp, zp):
    """Bicubic Catmull-Rom interpolation kernel.

    Pulls the 4x4 neighbourhood ``zp[ix-1..ix+2, iy-1..iy+2]`` and
    applies the tensor-product Catmull-Rom basis.  Boundary cells (where
    ``ix - 1 < 0`` or ``ix + 2 >= Nx``) use clamped indexing — pulling
    the boundary row/col in place of the missing ghost row/col.  This
    matches SciPy's ``RectBivariateSpline`` boundary behaviour to
    leading order and avoids spurious oscillations.
    """
    nx = xp.shape[0]
    ny = yp.shape[0]
    # Bucket index ``ix`` in [0, nx - 2] so ``ix`` and ``ix + 1`` are
    # both in range (the cell is [xp[ix], xp[ix+1]]).
    ix = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, nx - 2)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right") - 1, 0, ny - 2)

    x0 = xp[ix]
    x1 = xp[ix + 1]
    y0 = yp[iy]
    y1 = yp[iy + 1]

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    # 4 indices per axis: ix-1, ix, ix+1, ix+2, all clipped to [0, n-1]
    # so the boundary cell pulls the boundary row/col for the missing
    # ghost.  Equivalent to "edge-padding" the table.
    ix_m1 = jnp.clip(ix - 1, 0, nx - 1)
    ix_0 = ix
    ix_p1 = jnp.clip(ix + 1, 0, nx - 1)
    ix_p2 = jnp.clip(ix + 2, 0, nx - 1)

    iy_m1 = jnp.clip(iy - 1, 0, ny - 1)
    iy_0 = iy
    iy_p1 = jnp.clip(iy + 1, 0, ny - 1)
    iy_p2 = jnp.clip(iy + 2, 0, ny - 1)

    # Pull the 4x4 neighbourhood.  ``zp[i, j]`` access via advanced
    # indexing — JAX traces through this fine.
    def _row(ix_i):
        return jnp.stack(
            [
                zp[ix_i, iy_m1],
                zp[ix_i, iy_0],
                zp[ix_i, iy_p1],
                zp[ix_i, iy_p2],
            ],
            axis=-1,
        )

    row_m1 = _row(ix_m1)
    row_0 = _row(ix_0)
    row_p1 = _row(ix_p1)
    row_p2 = _row(ix_p2)

    wy = _catmull_rom_weights(ty)
    # Reduce along y for each of the 4 x-rows.
    f_m1 = jnp.sum(row_m1 * wy, axis=-1)
    f_0 = jnp.sum(row_0 * wy, axis=-1)
    f_p1 = jnp.sum(row_p1 * wy, axis=-1)
    f_p2 = jnp.sum(row_p2 * wy, axis=-1)

    wx = _catmull_rom_weights(tx)
    col = jnp.stack([f_m1, f_0, f_p1, f_p2], axis=-1)
    return jnp.sum(col * wx, axis=-1)


def interp_2d(
    x,
    y,
    xp,
    yp,
    zp,
    method: Literal["linear", "bicubic"] = "linear",
    extrapolation: Literal["clip", "linear", "nan"] = "clip",
):
    """2-D interpolation backend.

    Args:
        x, y: Query coordinates (scalar or array; broadcast together).
        xp: Strictly increasing 1-D breakpoint array along the first
            (``x``) axis.  Length ``Nx``.
        yp: Strictly increasing 1-D breakpoint array along the second
            (``y``) axis.  Length ``Ny``.
        zp: 2-D table values, shape ``(Nx, Ny)``.  ``zp[i, j] = f(xp[i], yp[j])``.
        method: ``"linear"`` (bilinear, default — standard lookup-table behaviour) or
            ``"bicubic"`` (T-114-followup-2d-bicubic — Catmull-Rom cubic
            convolution kernel, C^1-continuous, exact at grid corners,
            smoother than bilinear off-grid).  ``"bicubic"`` requires at
            least 4 breakpoints per axis.
        extrapolation: Out-of-range behaviour. ``"clip"`` holds the
            boundary value (matches :func:`npa.interp2d` and the
            standard 2-D lookup default); ``"linear"`` extends the boundary slope past the
            grid (bilinear continuation); ``"nan"`` returns NaN outside
            the grid.

    Returns:
        Interpolated value(s) with the broadcast shape of ``x`` and ``y``.
    """
    if method not in _INTERP_2D_METHODS:
        raise ValueError(
            f"interp_2d: unknown method {method!r}; expected one of "
            f"{_INTERP_2D_METHODS}"
        )
    if extrapolation not in _EXTRAP_MODES:
        raise ValueError(
            f"interp_2d: unknown extrapolation {extrapolation!r}; expected "
            f"one of {_EXTRAP_MODES}"
        )
    xp = jnp.asarray(xp)
    yp = jnp.asarray(yp)
    zp = jnp.asarray(zp)
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if method == "bicubic":
        # Bicubic Catmull-Rom needs at least 4 breakpoints per axis to
        # form a well-defined 4-point stencil even at the interior.  We
        # technically degrade gracefully to clamped boundaries below 4,
        # but the result loses smoothness and is no better than bilinear
        # in that regime — reject it loudly so callers don't get a
        # surprise.
        if int(xp.shape[0]) < 4 or int(yp.shape[0]) < 4:
            raise ValueError(
                f"interp_2d: method='bicubic' requires at least 4 "
                f"breakpoints per axis, got xp.shape={xp.shape} and "
                f"yp.shape={yp.shape}"
            )
        kernel = _bicubic_eval
    else:
        kernel = _bilinear_eval

    # ``raw`` is the value WITHOUT clipping the inputs (linear / cubic
    # extrapolation past the edge); ``base`` is the same kernel with
    # inputs clipped into the grid.  Cheap to compute both; the policy
    # picks one (or NaN) per element.
    raw = kernel(x, y, xp, yp, zp)
    base = kernel(
        jnp.clip(x, xp[0], xp[-1]),
        jnp.clip(y, yp[0], yp[-1]),
        xp,
        yp,
        zp,
    )
    return _apply_extrapolation_2d(x, y, xp, yp, base, raw, extrapolation)


# ---------------------------------------------------------------------------
# T-114-followup-phase3-2d-cubic — N-D linear interpolation.
#
# JAX has no ``jnp.interpn``, so the implementation does N successive
# 1-D linear interpolations (along axis -1, then -2, ..., 0).  For each
# axis we locate the surrounding bucket via ``searchsorted`` on the
# breakpoints, compute the per-axis blend weight ``alpha``, and convex-
# combine ``table[..., i]`` with ``table[..., i+1]``.  After N steps the
# table has been collapsed to a single value — the multilinear
# interpolant at the query point.
#
# Differentiability: the only non-smooth op is ``searchsorted`` for
# bucket location, which is non-differentiable but piecewise constant —
# within a cell ``jax.grad`` flows through both ``query`` (via the
# linear blend) and ``table`` (via the convex combination of corner
# values).  This matches the 1-D ``interp_1d`` story.
#
# Extrapolation: same three policies as the 1-D / 2-D cases.  ``"clip"``
# clips the query to the grid corners (standard lookup-table default).  ``"linear"``
# evaluates the multilinear kernel without clipping — past an edge each
# axis extends its boundary cell's slope, giving the natural
# multilinear continuation.  ``"nan"`` returns NaN whenever any
# coordinate is outside its axis range.
# ---------------------------------------------------------------------------


_INTERP_ND_METHODS = ("linear",)


def _multilinear_eval(query, grid, values):
    """N-D multilinear interpolation kernel.

    ``query`` is shape ``(N,)`` (single point); ``grid`` is a tuple of N
    1-D breakpoint arrays of lengths ``(B_1, ..., B_N)``; ``values`` is
    shape ``(B_1, ..., B_N)``.

    The implementation interpolates along the LAST axis first, reducing
    a rank-N table to a rank-(N-1) table, and recurses until rank 0.
    Inputs are NOT clipped here — the caller decides between the
    clipped (``base``) and raw (``raw``) evaluations to implement the
    extrapolation policy.
    """
    table = values
    # Walk axes back-to-front so we can keep collapsing the trailing
    # axis at each step (cheap reshape-free reduction).
    for axis in range(len(grid) - 1, -1, -1):
        bp = grid[axis]
        q = query[axis]
        n = bp.shape[0]
        # Bucket index ``i`` in [0, n - 2] so ``i + 1`` is in range.
        i = jnp.clip(jnp.searchsorted(bp, q, side="right") - 1, 0, n - 2)
        x0 = bp[i]
        x1 = bp[i + 1]
        alpha = (q - x0) / (x1 - x0)
        # Pull the two flanking slabs along the (now last) axis.
        left = jnp.take(table, i, axis=-1)
        right = jnp.take(table, i + 1, axis=-1)
        table = left * (1.0 - alpha) + right * alpha
    return table


def _query_in_grid(query, grid):
    """Boolean: is every coordinate of ``query`` inside its axis range?"""
    in_range = jnp.asarray(True)
    for axis in range(len(grid)):
        bp = grid[axis]
        q = query[axis]
        in_range = in_range & (q >= bp[0]) & (q <= bp[-1])
    return in_range


def _clip_query(query, grid):
    """Clip each coordinate of ``query`` to the corresponding axis range."""
    clipped = []
    for axis in range(len(grid)):
        bp = grid[axis]
        clipped.append(jnp.clip(query[axis], bp[0], bp[-1]))
    return jnp.stack(clipped)


def interp_nd(
    grid: tuple,
    values,
    query,
    method: Literal["linear"] = "linear",
    extrapolation: Literal["clip", "linear", "nan"] = "clip",
):
    """N-D multilinear interpolation backend.

    Args:
        grid: Tuple of N strictly-increasing 1-D breakpoint arrays.  The
            i-th array has length ``B_i`` and corresponds to axis ``i``
            of ``values``.
        values: Sample values, shape ``(B_1, B_2, ..., B_N)``.
        query: Query point.  Either shape ``(N,)`` for a single point or
            shape ``(..., N)`` for a batched query (``vmap`` over the
            leading dims).  The trailing axis is the coordinate axis.
        method: Currently only ``"linear"`` (multilinear).  Reserved for
            future N-D smooth methods (filed under
            ``T-114-followup-phase4-nd-cubic``).
        extrapolation: Out-of-range behaviour.  ``"clip"`` clips each
            coordinate to its axis range (standard lookup-table default).
            ``"linear"`` extends the boundary cell's slope past the
            grid (multilinear continuation).  ``"nan"`` returns NaN
            whenever any coordinate is outside its axis range.

    Returns:
        Interpolated value(s).  Scalar for a single ``(N,)`` query, or
        shape ``query.shape[:-1]`` for a batched query.

    Notes:
        Implemented via N successive 1-D linear interpolations rather
        than a single ``interpn`` call (JAX does not ship one today).
        For modest N (up to ~6) and modest grid sizes this is
        comfortable; very high-dim tables should consider tensor
        decomposition or tree-based lookups.  The optimisation is
        filed as ``T-114-followup-phase4-nd-cubic-optimisation``.
    """
    if method not in _INTERP_ND_METHODS:
        raise ValueError(
            f"interp_nd: unknown method {method!r}; expected one of "
            f"{_INTERP_ND_METHODS}"
        )
    if extrapolation not in _EXTRAP_MODES:
        raise ValueError(
            f"interp_nd: unknown extrapolation {extrapolation!r}; expected "
            f"one of {_EXTRAP_MODES}"
        )
    if not isinstance(grid, (tuple, list)) or len(grid) == 0:
        raise ValueError(
            f"interp_nd: grid must be a non-empty tuple/list of 1-D "
            f"breakpoint arrays, got {type(grid).__name__}"
        )
    grid = tuple(jnp.asarray(g) for g in grid)
    values = jnp.asarray(values)
    query = jnp.asarray(query)

    n_axes = len(grid)
    if values.ndim != n_axes:
        raise ValueError(
            f"interp_nd: values.ndim ({values.ndim}) must equal len(grid) "
            f"({n_axes})"
        )
    expected_shape = tuple(int(g.shape[0]) for g in grid)
    if values.shape != expected_shape:
        raise ValueError(
            f"interp_nd: values.shape {values.shape} must match the grid "
            f"shape {expected_shape}"
        )
    if query.shape[-1] != n_axes:
        raise ValueError(
            f"interp_nd: query.shape[-1] ({query.shape[-1]}) must equal the "
            f"number of grid axes ({n_axes})"
        )

    # Vectorised path: when the query has leading batch dims, vmap the
    # scalar kernel over them so callers don't need to wrap.  This keeps
    # the math focused on the single-point case.
    def _scalar(q):
        raw = _multilinear_eval(q, grid, values)
        base = _multilinear_eval(_clip_query(q, grid), grid, values)
        if extrapolation == "clip":
            return base
        if extrapolation == "linear":
            return raw
        # "nan": replace base with NaN whenever the query falls OOB on
        # any axis.  ``base`` keeps the dtype machinery happy.
        in_range = _query_in_grid(q, grid)
        nan_val = jnp.asarray(jnp.nan, dtype=base.dtype)
        return jnp.where(in_range, base, nan_val)

    # Recursively vmap over each leading batch dim.
    if query.ndim == 1:
        return _scalar(query)
    fn = _scalar
    for _ in range(query.ndim - 1):
        from jax import vmap as _vmap

        fn = _vmap(fn)
    return fn(query)


# ---------------------------------------------------------------------------
# T-124 phase 1 — Differentiable lookup-table fitting.
#
# Given a measured ``(x_data, y_data)`` cloud and a fixed grid ``xp``, find
# the table values ``yp`` that minimise the (optionally weighted) least-
# squares residual ``Σ_k w_k * (y_data_k - interp_linear(x_data_k; xp, yp))²``.
#
# For LINEAR interpolation this is a *linear* problem in ``yp``: each
# query at ``x_data[k]`` is a convex combination of the two flanking
# table entries with weights ``(1 - α_k, α_k)`` where
# ``α_k = (x_data[k] - xp[i]) / (xp[i+1] - xp[i])``.  The least-squares
# normal equations are then solved by ``jnp.linalg.lstsq``.
#
# Out-of-range queries are clamped to the nearest endpoint (matching the
# default ``"clip"`` extrapolation policy of :class:`LookupTable1d`).  An
# optional smoothness regulariser ``λ * Σ (yp[i+1] - yp[i])²`` (a discrete
# Laplacian-style penalty on the table) suppresses overfitting on coarse
# grids with sparse data.
#
# The whole pipeline is built on ``jnp`` so ``jax.grad`` flows through
# ``y_data`` (and ``x_data``, modulo the discrete ``searchsorted``
# bucket index — within a bucket the gradient is exact).  Returning a
# fully-built ``LookupTable1d`` would tie this module to the block layer
# (and to the ``primitives.py`` import we explicitly avoid for T-124
# phase 1 collision with T-127), so the *math* is exposed here as
# ``fit_table_1d`` and the convenience wrapper that wraps the result in
# a block lives in ``lookup_table_fitting.py``.
# ---------------------------------------------------------------------------


def _build_linear_design(xp, x_data):
    """Build the ``(len(x_data), len(xp))`` linear-interp design matrix.

    Row ``k`` has weight ``1 - α_k`` at column ``i_k`` and ``α_k`` at
    column ``i_k + 1``, where ``i_k`` is the grid bucket containing
    ``x_data[k]``.  Out-of-range queries clamp to the nearest endpoint
    (left endpoint -> column 0 with weight 1; right endpoint -> column
    ``n - 1`` with weight 1) — matching the ``"clip"`` extrapolation
    policy of the runtime block.

    The matrix is dense (``jnp`` does not have a sparse lstsq today),
    but with a typical grid size of 10-50 entries and ``len(x_data)``
    in the thousands this is comfortable.
    """
    xp = jnp.asarray(xp)
    x_data = jnp.asarray(x_data)
    n = xp.shape[0]
    if n < 2:
        raise ValueError(
            f"fit_table_1d: need at least 2 grid points, got xp of shape "
            f"{xp.shape}"
        )

    # Bucket index ``i`` such that ``xp[i] <= x_data[k] <= xp[i+1]``.
    # Clip to ``[0, n-2]`` so the "+1" neighbour stays in range; the
    # alpha clip below collapses out-of-range queries onto a single
    # endpoint with weight 1.
    i = jnp.clip(jnp.searchsorted(xp, x_data, side="right") - 1, 0, n - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    alpha = (x_data - x0) / (x1 - x0)
    # Clip alpha to [0, 1] so OOB-left -> weight 1 on column i_k = 0
    # (alpha = 0) and OOB-right -> weight 1 on column n - 1 (alpha = 1).
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # Build the design matrix via a one-hot scatter.  Equivalent to the
    # ``A.at[k, i].set(1 - alpha)`` / ``A.at[k, i+1].set(alpha)`` loop
    # in the architecture sketch but vectorised.
    k_idx = jnp.arange(x_data.shape[0])
    A = jnp.zeros((x_data.shape[0], n), dtype=jnp.result_type(xp, x_data))
    A = A.at[k_idx, i].add(1.0 - alpha)
    A = A.at[k_idx, i + 1].add(alpha)
    return A


def fit_table_1d(
    xp,
    x_data,
    y_data,
    weights=None,
    smoothness: float = 0.0,
    rcond: float | None = None,
):
    """Fit table values ``yp`` at the fixed grid ``xp`` to ``(x_data, y_data)``.

    Solves the least-squares problem

        min_yp  Σ_k w_k * (y_data[k] - interp_linear(x_data[k]; xp, yp))²
                + smoothness * Σ_i (yp[i+1] - yp[i])²

    via :func:`jnp.linalg.lstsq` on the linear design matrix.  Linear
    interpolation only — for non-linear (PCHIP / Akima) the relationship
    between ``yp`` and the interpolated curve is no longer linear in
    ``yp`` at the query points (the slopes depend on neighbouring
    entries) and a non-linear solver would be required.  See
    ``T-124-followup-pchip-fit``.

    Args:
        xp: 1-D, strictly increasing grid of breakpoints.  Treated as
            fixed (gradient w.r.t. ``xp`` is *not* exposed; that needs a
            non-linear fit and is filed as
            ``T-124-followup-grid-optimization``).
        x_data: Measured input cloud, shape ``(K,)``.
        y_data: Measured output cloud, shape ``(K,)``.
        weights: Optional per-sample weights, shape ``(K,)``.  ``None``
            means uniform weighting (ordinary least squares).
        smoothness: Non-negative coefficient for the discrete first-
            difference smoothness penalty.  ``0.0`` (default) is pure
            OLS; values around ``1e-3 .. 1.0`` regularise on noisy /
            coarse data.  Acts as a soft prior toward linear table
            shapes.
        rcond: Forwarded to :func:`jnp.linalg.lstsq`.

    Returns:
        ``yp`` of shape ``(len(xp),)`` — the optimal table values.
        Differentiable through ``y_data`` (and through ``x_data`` modulo
        the discrete bucket index).

    Honest fallback (``T-124-followup-regularization``): on very coarse
    grids with regions of zero data coverage the design matrix is
    rank-deficient and ``lstsq`` returns the minimum-norm solution.
    Pass a small ``smoothness`` (or weights) to disambiguate.
    """
    xp = jnp.asarray(xp)
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    if x_data.shape != y_data.shape:
        raise ValueError(
            f"fit_table_1d: x_data shape {x_data.shape} must match y_data "
            f"shape {y_data.shape}"
        )
    if x_data.ndim != 1:
        raise ValueError(
            f"fit_table_1d: x_data must be 1-D, got shape {x_data.shape}"
        )
    if smoothness < 0:
        raise ValueError(
            f"fit_table_1d: smoothness must be >= 0, got {smoothness}"
        )

    A = _build_linear_design(xp, x_data)
    b = y_data

    # Optional per-sample weighting: scale rows of A and entries of b.
    if weights is not None:
        w = jnp.asarray(weights)
        if w.shape != x_data.shape:
            raise ValueError(
                f"fit_table_1d: weights shape {w.shape} must match x_data "
                f"shape {x_data.shape}"
            )
        sqrt_w = jnp.sqrt(w)
        A = A * sqrt_w[:, None]
        b = b * sqrt_w

    # Smoothness penalty as additional rows of the system: rows of the
    # form ``sqrt(λ) * (yp[i+1] - yp[i])`` are stacked on top of the
    # data residuals so ``lstsq`` minimises the joint sum of squares.
    if smoothness > 0:
        n = xp.shape[0]
        # First-difference matrix of shape (n - 1, n).
        D = jnp.eye(n - 1, n, k=1, dtype=A.dtype) - jnp.eye(
            n - 1, n, dtype=A.dtype
        )
        A = jnp.concatenate([A, jnp.sqrt(smoothness) * D], axis=0)
        b = jnp.concatenate([b, jnp.zeros(n - 1, dtype=b.dtype)], axis=0)

    yp, *_ = jnp.linalg.lstsq(A, b, rcond=rcond)
    return yp


__all__ = [
    "interp_1d",
    "interp_2d",
    "interp_nd",
    "pchip_slopes",
    "akima_slopes",
    "natural_cubic_second_derivs",
    "even_spacing",
    "fit_table_1d",
]
