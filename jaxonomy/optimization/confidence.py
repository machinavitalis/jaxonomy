# SPDX-License-Identifier: MIT

"""
Parameter confidence intervals via the Laplace approximation.

After fitting a model, users want to know how certain they are about each
parameter value.  The **Laplace approximation** approximates the posterior
as a Gaussian centred at the optimum θ* with covariance matrix H⁻¹, where
H = ∇²L(θ*) is the Hessian of the loss at the optimum.

For **maximum-likelihood** objectives the covariance is exactly H⁻¹ and the
Laplace approximation is the standard Fisher-information-matrix approach.

For **least-squares** objectives ``L = Σ rᵢ²`` the *residual variance* must
be estimated and used to scale H⁻¹.  Pass ``n_data`` to enable this.

For **nonlinear** models the Laplace approximation is a useful first-order
estimate; it becomes exact for linear-in-parameters models.

Quick start::

    from jaxonomy.optimization import Scipy, compute_confidence_intervals

    scipy_opt = Scipy(my_optimizable, method="L-BFGS-B", use_autodiff_grad=True)
    result = scipy_opt.optimize()

    ci = compute_confidence_intervals(my_optimizable, result)
    print(ci.summary())
    lo, hi = ci.interval("c")

References
----------
* Nocedal & Wright, "Numerical Optimization", §18.5 – parameter covariance
* Press et al., "Numerical Recipes", §15.6 – confidence limits via Hessian
* Raue et al. (2009) – identifiability and profile likelihood
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .framework.base.optimizable import Optimizable
from .framework.base.optimizer import OptimizationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_quantile(confidence_level: float) -> float:
    """Return the standard-normal upper-tail quantile for a two-sided CI.

    For a 95 % CI ``confidence_level=0.95`` → returns ``z = 1.960``.
    Uses scipy.stats if available; falls back to a high-accuracy table.
    """
    try:
        from scipy.stats import norm as _snorm
        return float(_snorm.ppf((1.0 + confidence_level) / 2.0))
    except ImportError:
        pass

    # Hardcoded high-accuracy values for common levels
    _TABLE = {
        0.500: 0.6745,
        0.680: 1.0000,
        0.900: 1.6449,
        0.950: 1.9600,
        0.990: 2.5758,
        0.999: 3.2905,
    }
    if confidence_level in _TABLE:
        return _TABLE[confidence_level]

    # Linear interpolation between bracketing entries
    levels = sorted(_TABLE)
    for i in range(len(levels) - 1):
        lo_l, hi_l = levels[i], levels[i + 1]
        if lo_l <= confidence_level <= hi_l:
            t = (confidence_level - lo_l) / (hi_l - lo_l)
            return _TABLE[lo_l] * (1 - t) + _TABLE[hi_l] * t

    raise ValueError(
        f"confidence_level={confidence_level} is outside [0.5, 0.999]. "
        "Either install scipy or pass a level in [0.5, 0.999]."
    )


def _expand_param_names(params_dict: dict) -> list[str]:
    """Expand a parameter dict into flat names.

    Scalar parameters contribute one name each; array parameters contribute
    one name per element: ``"theta[0,1]"``, ``"theta[2,0]"``, etc.
    """
    names: list[str] = []
    for key, val in params_dict.items():
        val_np = np.asarray(val)
        if val_np.ndim == 0 or val_np.size == 1:
            names.append(key)
        else:
            for idx in np.ndindex(val_np.shape):
                idx_str = ",".join(map(str, idx))
                names.append(f"{key}[{idx_str}]")
    return names


def _to_flat_array(
    opt_params: "OptimizationResult | dict | np.ndarray | jnp.ndarray",
    unflatten_fn,
) -> np.ndarray:
    """Extract a flat 1-D numpy array from various param representations."""
    if isinstance(opt_params, OptimizationResult):
        params_dict = {k: jnp.array(v) for k, v in opt_params.params.items()}
        flat, _ = ravel_pytree(params_dict)
        return np.array(flat, dtype=float)
    if isinstance(opt_params, dict):
        params_dict = {k: jnp.array(v) for k, v in opt_params.items()}
        flat, _ = ravel_pytree(params_dict)
        return np.array(flat, dtype=float)
    # Assume array-like
    return np.asarray(opt_params, dtype=float).ravel()


def _nearest_positive_definite(H: np.ndarray, min_eigval: float = 1e-8) -> tuple[np.ndarray, bool]:
    """
    Return (H_pd, was_pd) where H_pd is the nearest symmetric positive-definite
    matrix to H.

    Strategy: eigendecompose H, clip any eigenvalue below ``min_eigval``,
    reconstruct.  This is the Higham (1988) / Cheng & Higham (1998) flavour
    restricted to the simple eigenvalue-clipping variant.
    """
    H_sym = 0.5 * (H + H.T)  # enforce exact symmetry first
    try:
        eigvals, eigvecs = np.linalg.eigh(H_sym)
    except np.linalg.LinAlgError:
        return H_sym, False

    was_pd = bool(np.all(eigvals >= min_eigval))
    clipped = np.maximum(eigvals, min_eigval)
    H_pd = eigvecs @ np.diag(clipped) @ eigvecs.T
    return H_pd, was_pd


def _safe_invert(H: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Try ``np.linalg.inv`` first; fall back to ``np.linalg.pinv`` with a
    warning note.

    Returns (H_inv, message).
    """
    try:
        H_inv = np.linalg.inv(H)
        msg = ""
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)
        msg = "Hessian was singular; pseudoinverse used — CIs may be unreliable."
    return H_inv, msg


def _compute_hessian(
    obj_fn,
    params_flat: np.ndarray,
    eps: float = 1e-4,
) -> tuple[np.ndarray, str]:
    """
    Compute the Hessian of ``obj_fn`` at ``params_flat``.

    Tries exact JAX second-order AD first; falls back to central finite
    differences if the objective uses ``custom_vjp`` (e.g. ODE solvers that
    only support first-order reverse-mode AD).

    Returns (hessian, method_used_message).
    """
    n = len(params_flat)
    p = jnp.array(params_flat)

    # --- attempt exact AD ---
    try:
        hess_fn = jax.jit(jax.hessian(obj_fn))
        H = np.array(hess_fn(p), dtype=float)
        if not np.any(np.isnan(H)):
            return H, "AD"
    except Exception:
        pass

    # --- finite-difference fallback ---
    H = np.zeros((n, n), dtype=float)
    try:
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = eps
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = eps
                f_pp = float(obj_fn(jnp.array(params_flat + ei + ej)))
                f_pm = float(obj_fn(jnp.array(params_flat + ei - ej)))
                f_mp = float(obj_fn(jnp.array(params_flat - ei + ej)))
                f_mm = float(obj_fn(jnp.array(params_flat - ei - ej)))
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
        return H, "FD"
    except Exception as exc:
        return np.full((n, n), np.nan), f"failed ({exc})"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceIntervalResult:
    """
    Confidence intervals and covariance matrix from the Laplace approximation.

    All matrix/array attributes are plain ``numpy.ndarray`` for easy inspection
    and serialisation.

    Attributes
    ----------
    param_names : list[str]
        Flat parameter names (array params expanded to ``"theta[0]"``, etc.).
    opt_params : dict
        Optimised parameter values in the **original** (un-transformed) space.
    covariance : ndarray, shape (n, n)
        Estimated parameter covariance matrix.
    correlation : ndarray, shape (n, n)
        Correlation matrix (covariance normalised by marginal standard deviations).
    standard_errors : ndarray, shape (n,)
        Marginal standard deviations ``sqrt(diag(covariance))``.
    confidence_intervals : dict[str, tuple[float, float]]
        Per-parameter ``(lower, upper)`` bounds in the **original** space.
        Keys match ``param_names``.
    confidence_level : float
        Nominal confidence level (e.g. ``0.95`` for 95 %).
    z_score : float
        Standard-normal quantile corresponding to ``confidence_level``.
    hessian : ndarray, shape (n, n)
        Hessian of the objective evaluated at the optimum, in the (possibly
        transformed) optimisation space.
    hessian_eigenvalues : ndarray, shape (n,)
        Eigenvalues of the Hessian (ascending).
    hessian_condition_number : float
        Ratio max|λ| / min|λ|.  Large values (> 1 000) signal near-collinear
        parameters or an ill-conditioned problem.
    is_positive_definite : bool
        ``True`` when the Hessian was positive definite at the supplied point
        (necessary condition for a true local minimum).
    residual_variance : float or None
        Residual variance ``σ²`` used to scale the covariance.  ``None`` when
        ``n_data`` was not provided (pure MLE / default).
    n_data : int or None
        Number of observations used (for least-squares scaling).
    objective_value : float
        Loss at the optimum.
    hessian_method : str
        How the Hessian was computed: ``"AD"`` (automatic differentiation),
        ``"FD"`` (finite differences), ``"provided"``, or ``"failed"``.
    message : str
        Any warnings raised during computation (empty when all is well).
    """

    param_names: list[str]
    opt_params: dict[str, Any]
    covariance: np.ndarray
    correlation: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals: dict[str, tuple[float, float]]
    confidence_level: float
    z_score: float
    hessian: np.ndarray
    hessian_eigenvalues: np.ndarray
    hessian_condition_number: float
    is_positive_definite: bool
    residual_variance: float | None
    n_data: int | None
    objective_value: float
    hessian_method: str
    message: str

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def interval(self, param_name: str) -> tuple[float, float]:
        """Return ``(lower, upper)`` for a single parameter by name.

        Raises ``KeyError`` if the name is not found.  For array parameters
        use the expanded name, e.g. ``ci.interval("theta[0]")``.
        """
        return self.confidence_intervals[param_name]

    def contains(self, param_name: str, value: float) -> bool:
        """Return ``True`` when *value* lies within the CI for *param_name*."""
        lo, hi = self.interval(param_name)
        return lo <= value <= hi

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted human-readable summary table."""
        level_pct = self.confidence_level * 100
        lines = [
            f"=== Parameter Confidence Intervals ({level_pct:.1f}%) "
            f"[Laplace approximation] ===",
            f"Objective at optimum       : {self.objective_value:.6g}",
            f"Hessian computation method : {self.hessian_method}",
            f"Hessian positive definite  : {'yes' if self.is_positive_definite else 'NO ← not at a true minimum'}",
            f"Hessian condition number   : {self.hessian_condition_number:.3g}",
        ]
        if self.residual_variance is not None:
            lines.append(f"Residual variance σ²       : {self.residual_variance:.4g}  "
                         f"(n_data={self.n_data})")
        if self.message:
            lines.append(f"⚠  {self.message}")
        lines += [
            "",
            f"{'Parameter':<24} {'Opt. value':>14} {'Std. error':>13} "
            f"  {level_pct:.1f}% CI",
            "-" * 72,
        ]
        for name in self.param_names:
            lo, hi = self.confidence_intervals[name]
            # Retrieve the optimal value (may be multi-element param)
            se_idx = self.param_names.index(name)
            se = self.standard_errors[se_idx]
            # Optimal value in original space
            opt_val = self._opt_val_for(name)
            lines.append(
                f"{name:<24} {opt_val:>14.6g} {se:>13.4e}  "
                f"[{lo:>12.6g}, {hi:>12.6g}]"
            )
        return "\n".join(lines)

    def _opt_val_for(self, flat_name: str) -> float:
        """Extract the scalar optimal value for a flat parameter name."""
        # scalar param
        if flat_name in self.opt_params:
            return float(np.asarray(self.opt_params[flat_name]).ravel()[0])
        # vector param: name looks like "theta[0]" or "theta[0,1]"
        bracket = flat_name.find("[")
        if bracket != -1:
            key = flat_name[:bracket]
            idx_str = flat_name[bracket + 1 : flat_name.find("]")]
            idx = tuple(int(s) for s in idx_str.split(","))
            val = np.asarray(self.opt_params.get(key, np.nan))
            try:
                return float(val[idx])
            except (IndexError, TypeError):
                pass
        return float("nan")

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_confidence_intervals(
    optimizable: Optimizable,
    opt_params: "OptimizationResult | dict | np.ndarray",
    confidence_level: float = 0.95,
    n_data: int | None = None,
    hessian: "np.ndarray | None" = None,
    eps_fd: float = 1e-4,
    regularize: bool = True,
) -> ConfidenceIntervalResult:
    """
    Compute Wald-type confidence intervals for optimised parameters.

    Uses the **Laplace approximation**: the parameter posterior is approximated
    as a Gaussian centred at the optimum θ* with covariance ``H⁻¹``, where
    ``H = ∇²L(θ*)`` is the Hessian of the loss.

    Parameters
    ----------
    optimizable : Optimizable
        The jaxonomy optimizable whose ``objective_flat`` is used.
    opt_params : OptimizationResult | dict | array-like
        Optimised parameters at which to evaluate the Hessian.  Can be:

        * An :class:`~jaxonomy.optimization.OptimizationResult` returned by
          any jaxonomy optimizer — the ``params`` dict is extracted and
          flattened automatically.
        * A plain ``dict`` mapping parameter names to values.
        * A flat 1-D array matching ``optimizable.params_0_flat``.
    confidence_level : float
        Nominal confidence level (default ``0.95`` for 95 % CIs).
    n_data : int or None
        Number of observations.  When provided, the covariance is scaled by
        the residual variance estimate

            ``σ² = 2 · L(θ*) / max(n_data − n_params, 1)``

        This is appropriate for **sum-of-squares objectives**
        ``L = ½ Σ rᵢ²``.  For maximum-likelihood objectives leave ``None``.
    hessian : ndarray or None
        Pre-computed Hessian matrix (e.g. from :func:`compute_sensitivity`).
        When ``None`` (default) the Hessian is computed automatically using
        JAX AD (with a finite-difference fallback for ODE-based objectives).
    eps_fd : float
        Step size used for the finite-difference Hessian fallback (default
        ``1e-4``).  Ignored when ``hessian`` is provided or when AD succeeds.
    regularize : bool
        When ``True`` (default), negative eigenvalues of the Hessian are
        clipped to a small positive value before inversion.  This makes the
        covariance well-defined even when the supplied point is not a true
        local minimum.  A warning is recorded in ``result.message``.

    Returns
    -------
    ConfidenceIntervalResult
        Dataclass containing the covariance matrix, standard errors, and
        per-parameter confidence intervals in the **original** (physical)
        parameter space.

    Notes
    -----
    **Parameter transformations**: if the ``Optimizable`` uses a
    ``transformation`` (e.g. :class:`LogTransform`), the Hessian is computed
    in the *transformed* space and the resulting CI bounds are mapped back to
    the original space via ``transformation.inverse_transform``.

    **Validity**: the Laplace approximation requires the objective to be
    smooth and the optimum to be a true interior local minimum (positive-
    definite Hessian).  If ``is_positive_definite`` is ``False`` in the
    result, the CIs are computed but should be treated with caution.

    **Profile likelihood**: the Laplace approximation is a first-order
    Gaussian approximation.  For strongly nonlinear models or highly
    non-Gaussian posteriors, profile likelihood confidence intervals are
    more accurate but require repeated re-optimisation.

    Examples
    --------
    >>> from jaxonomy.optimization import Scipy, compute_confidence_intervals
    >>> opt = Scipy(my_opt, method="L-BFGS-B", use_autodiff_grad=True)
    >>> result = opt.optimize()
    >>> ci = compute_confidence_intervals(my_opt, result, confidence_level=0.95)
    >>> print(ci.summary())
    >>> lo, hi = ci.interval("c")
    """
    # ------------------------------------------------------------------
    # 1. Resolve flat parameter vector at the optimum
    # ------------------------------------------------------------------
    opt_flat = _to_flat_array(opt_params, optimizable.unflatten_params)
    n_params = len(opt_flat)

    # ------------------------------------------------------------------
    # 2. Objective value at the optimum
    # ------------------------------------------------------------------
    obj_fn = jax.jit(optimizable.objective_flat)
    try:
        obj_val = float(obj_fn(jnp.array(opt_flat)))
    except Exception:
        obj_val = float("nan")

    # ------------------------------------------------------------------
    # 3. Hessian at the optimum
    # ------------------------------------------------------------------
    messages: list[str] = []

    if hessian is not None:
        H_raw = np.asarray(hessian, dtype=float)
        hess_method = "provided"
    else:
        H_raw, hess_method = _compute_hessian(obj_fn, opt_flat, eps=eps_fd)

    if np.any(np.isnan(H_raw)):
        messages.append(
            f"Hessian computation {hess_method!r} produced NaN — "
            "covariance and CIs are unreliable."
        )

    # ------------------------------------------------------------------
    # 4. Eigendecompose and check positive-definiteness
    # ------------------------------------------------------------------
    H_sym = 0.5 * (H_raw + H_raw.T)  # enforce symmetry

    try:
        eigvals_raw = np.linalg.eigvalsh(H_sym)
        is_pd = bool(np.all(eigvals_raw > 0))
        pos = np.abs(eigvals_raw[np.abs(eigvals_raw) > 1e-16])
        hess_cond = float(pos.max() / pos.min()) if len(pos) >= 2 else (
            1.0 if len(pos) == 1 else float("inf")
        )
    except np.linalg.LinAlgError:
        eigvals_raw = np.full(n_params, np.nan)
        is_pd = False
        hess_cond = float("inf")

    if not is_pd and not np.any(np.isnan(H_raw)):
        messages.append(
            "Hessian is not positive definite — supplied point may not be "
            "a true local minimum. "
            + ("CIs computed with regularised Hessian." if regularize else
               "CIs may be unreliable.")
        )

    # ------------------------------------------------------------------
    # 5. Regularise (clip negative eigenvalues) if requested
    # ------------------------------------------------------------------
    if regularize:
        H_for_inv, _ = _nearest_positive_definite(H_sym)
    else:
        H_for_inv = H_sym

    # ------------------------------------------------------------------
    # 6. Invert Hessian → raw covariance
    # ------------------------------------------------------------------
    H_inv, inv_msg = _safe_invert(H_for_inv)
    if inv_msg:
        messages.append(inv_msg)

    # ------------------------------------------------------------------
    # 7. Residual-variance scaling (least-squares mode)
    # ------------------------------------------------------------------
    resid_var: float | None = None
    if n_data is not None:
        dof = max(n_data - n_params, 1)
        if not np.isnan(obj_val):
            resid_var = 2.0 * obj_val / dof
        else:
            resid_var = float("nan")
            messages.append(
                "Could not compute residual variance: objective value is NaN."
            )
        if resid_var is not None and not np.isnan(resid_var):
            H_inv = H_inv * resid_var

    cov = H_inv

    # ------------------------------------------------------------------
    # 8. Correlation matrix and standard errors
    # ------------------------------------------------------------------
    diag = np.diag(cov)
    # Clip negative diagonal entries (can arise from regularisation of badly
    # conditioned problems) to avoid imaginary standard errors.
    diag_safe = np.maximum(diag, 0.0)
    if np.any(diag < 0):
        messages.append(
            "Covariance matrix has negative diagonal entries; "
            "standard errors for affected parameters are set to NaN."
        )
    std_errors = np.where(diag_safe > 0, np.sqrt(diag_safe), np.nan)

    outer_std = np.outer(std_errors, std_errors)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(outer_std > 0, cov / outer_std, np.eye(n_params))

    # ------------------------------------------------------------------
    # 9. z-score and raw (transformed-space) CIs
    # ------------------------------------------------------------------
    z = _z_quantile(confidence_level)
    lo_flat = opt_flat - z * std_errors
    hi_flat = opt_flat + z * std_errors

    # ------------------------------------------------------------------
    # 10. Back-transform CIs to original parameter space
    # ------------------------------------------------------------------
    lo_dict_t = optimizable.unflatten_params(jnp.array(lo_flat))
    hi_dict_t = optimizable.unflatten_params(jnp.array(hi_flat))
    opt_dict_t = optimizable.unflatten_params(jnp.array(opt_flat))

    if getattr(optimizable, "transformation", None) is not None:
        tf = optimizable.transformation
        lo_dict_orig = tf.inverse_transform(lo_dict_t)
        hi_dict_orig = tf.inverse_transform(hi_dict_t)
        opt_dict_orig = tf.inverse_transform(opt_dict_t)
    else:
        lo_dict_orig = lo_dict_t
        hi_dict_orig = hi_dict_t
        opt_dict_orig = opt_dict_t

    # Convert to plain Python dicts (jax arrays → numpy scalars/arrays)
    opt_dict_orig = {k: np.asarray(v) for k, v in opt_dict_orig.items()}
    lo_dict_orig  = {k: np.asarray(v) for k, v in lo_dict_orig.items()}
    hi_dict_orig  = {k: np.asarray(v) for k, v in hi_dict_orig.items()}

    # ------------------------------------------------------------------
    # 11. Build the per-parameter CI dict using expanded names
    # ------------------------------------------------------------------
    flat_names = _expand_param_names(opt_dict_orig)

    def _dict_to_flat(d: dict) -> np.ndarray:
        """Flatten a dict of arrays in the same order as flat_names."""
        vals = []
        for key, val in d.items():
            val_np = np.asarray(val)
            if val_np.ndim == 0 or val_np.size == 1:
                vals.append(float(val_np.ravel()[0]))
            else:
                vals.extend(float(x) for x in val_np.ravel())
        return np.array(vals)

    lo_orig_flat = _dict_to_flat(lo_dict_orig)
    hi_orig_flat = _dict_to_flat(hi_dict_orig)
    opt_orig_flat = _dict_to_flat(opt_dict_orig)

    # When a transform is monotone *decreasing* (unlikely but possible),
    # lo and hi may swap.  Always store as (min, max).
    ci_dict: dict[str, tuple[float, float]] = {}
    for name, lo_val, hi_val in zip(flat_names, lo_orig_flat, hi_orig_flat):
        ci_dict[name] = (min(float(lo_val), float(hi_val)),
                         max(float(lo_val), float(hi_val)))

    # ------------------------------------------------------------------
    # 12. Assemble result
    # ------------------------------------------------------------------
    return ConfidenceIntervalResult(
        param_names=flat_names,
        opt_params=opt_dict_orig,
        covariance=cov,
        correlation=corr,
        standard_errors=std_errors,
        confidence_intervals=ci_dict,
        confidence_level=confidence_level,
        z_score=z,
        hessian=H_raw,
        hessian_eigenvalues=eigvals_raw,
        hessian_condition_number=hess_cond,
        is_positive_definite=is_pd,
        residual_variance=resid_var,
        n_data=n_data,
        objective_value=obj_val,
        hessian_method=hess_method,
        message="  |  ".join(messages),
    )
