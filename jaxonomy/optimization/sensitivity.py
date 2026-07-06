# SPDX-License-Identifier: MIT

"""
Parameter sensitivity and identifiability analysis.

Before spending CPU time fitting parameters, it is worth checking whether the
objective is *sensitive* to each parameter at the initial point.  Near-zero
gradient → the simulation output barely changes with that parameter → the
parameter is unidentifiable from this dataset.

The Fisher Information Matrix (FIM) approximation (Hessian of the objective)
goes further: small eigenvalues reveal unidentifiable *directions* in parameter
space, and a large condition number flags near-collinear parameter pairs.

Usage::

    from jaxonomy.optimization.sensitivity import compute_sensitivity

    result = compute_sensitivity(my_optimizable)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .confidence import _expand_param_names
from .framework.base.optimizable import Optimizable


@dataclass
class SensitivityResult:
    """
    Result of a parameter sensitivity / identifiability analysis.

    All arrays are plain ``numpy.ndarray`` for easy inspection.

    Attributes
    ----------
    param_names : list[str]
        Parameter names, in the same order as the flat parameter vector.
    params_0 : dict[str, Any]
        The parameter values at which the analysis was performed.
    objective_value : float
        Objective value at ``params_0``.
    gradients : ndarray, shape (n_params,)
        Gradient of the objective w.r.t. each parameter.
    normalized_sensitivity : ndarray, shape (n_params,)
        ``|p_i * ∂L/∂p_i|`` — relative sensitivity.  Dimensionless and
        comparable across parameters with different scales.
    hessian : ndarray, shape (n_params, n_params)
        Hessian of the objective (FIM approximation).  ``NaN``-filled when
        ``compute_hessian=False``.
    hessian_diagonal : ndarray, shape (n_params,)
        Diagonal of the Hessian.
    eigenvalues : ndarray, shape (n_params,)
        Eigenvalues of the Hessian (ascending).
    condition_number : float
        Ratio of largest to smallest non-negligible eigenvalue.  Large values
        (> 1e6) indicate near-collinear parameters.
    unidentifiable_params : list[str]
        Parameter names whose normalised sensitivity is below
        ``sensitivity_threshold * max_sensitivity``.
    sensitivity_threshold : float
        Relative threshold used to flag unidentifiable parameters.
    """

    param_names: list[str]
    params_0: dict[str, Any]
    objective_value: float
    gradients: np.ndarray
    normalized_sensitivity: np.ndarray
    hessian: np.ndarray
    hessian_diagonal: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    unidentifiable_params: list[str]
    sensitivity_threshold: float = 1e-3

    def summary(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            "=== Parameter Sensitivity / Identifiability Analysis ===",
            f"Objective at params_0: {self.objective_value:.6g}",
            f"Hessian condition number: {self.condition_number:.3g}",
            "",
            f"{'Parameter':<22} {'Gradient':>14} {'Norm. Sensitivity':>20} "
            f"{'Status':>12}",
            "-" * 70,
        ]
        max_s = float(np.max(self.normalized_sensitivity)) if len(self.normalized_sensitivity) else 1.0
        for name, g, s in zip(
            self.param_names, self.gradients, self.normalized_sensitivity
        ):
            relative = s / max(max_s, 1e-30)
            status = "✓ ok" if name not in self.unidentifiable_params else "✗ LOW"
            lines.append(
                f"{name:<22} {g:>14.4e} {s:>20.4e} {status:>12}"
            )
        if self.unidentifiable_params:
            lines.append(
                f"\n⚠  Low-sensitivity parameters "
                f"(threshold={self.sensitivity_threshold:.1e}): "
                f"{self.unidentifiable_params}"
            )
        if not np.any(np.isnan(self.eigenvalues)):
            lines.append(f"\nHessian eigenvalues: {self.eigenvalues}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def compute_sensitivity(
    optimizable: Optimizable,
    params_0_flat: "jnp.ndarray | None" = None,
    sensitivity_threshold: float = 1e-3,
    compute_hessian: bool = True,
) -> SensitivityResult:
    """
    Compute gradient-based parameter sensitivity at a given operating point.

    Uses JAX automatic differentiation — no finite differences, no extra
    simulations beyond two JIT-compiled evaluations (gradient + optional
    Hessian).

    Parameters
    ----------
    optimizable : Optimizable
        The jaxonomy optimizable whose ``objective_flat`` is differentiated.
    params_0_flat : array-like or None
        Flat parameter vector to evaluate at.  Defaults to
        ``optimizable.params_0_flat``.
    sensitivity_threshold : float
        Relative threshold (0–1) for flagging parameters as low-sensitivity.
        A parameter is flagged when its normalised sensitivity is less than
        ``sensitivity_threshold × max(all normalised sensitivities)``.
        Default ``1e-3``.
    compute_hessian : bool
        Whether to compute the full Hessian / FIM.  Can be expensive for
        many parameters (O(n²) simulations).  Default ``True``.

    Returns
    -------
    SensitivityResult
    """
    if params_0_flat is None:
        params_0_flat = optimizable.params_0_flat
    params_0_flat = jnp.array(params_0_flat, dtype=float)

    obj_fn = jax.jit(optimizable.objective_flat)
    grad_fn = jax.jit(jax.grad(obj_fn))

    # --- objective value and gradient ---
    obj_val = float(obj_fn(params_0_flat))
    grads = np.array(grad_fn(params_0_flat), dtype=float)

    # --- normalised sensitivity: |p_i * ∂L/∂p_i| ---
    p0_np = np.array(params_0_flat, dtype=float)
    # Use max(|p|, 1) so that near-zero parameters still get a meaningful scale
    scale = np.where(np.abs(p0_np) > 1e-10, np.abs(p0_np), 1.0)
    norm_sensitivity = np.abs(grads * scale)

    # --- Hessian (FIM approximation) ---
    n = len(params_0_flat)
    if compute_hessian:
        try:
            hess_fn = jax.jit(jax.hessian(obj_fn))
            hessian = np.array(hess_fn(params_0_flat), dtype=float)
        except Exception:
            # Second-order AD may fail when the objective uses ODE solvers with
            # custom_vjp (which only supports first-order differentiation).
            # Fall back to a finite-difference approximation of the Hessian.
            try:
                eps = 1e-4
                p0_np_h = np.array(params_0_flat, dtype=float)
                hessian = np.zeros((n, n), dtype=float)
                for i in range(n):
                    ei = np.zeros(n, dtype=float)
                    ei[i] = eps
                    for j in range(n):
                        ej = np.zeros(n, dtype=float)
                        ej[j] = eps
                        f_pp = float(obj_fn(jnp.array(p0_np_h + ei + ej)))
                        f_pm = float(obj_fn(jnp.array(p0_np_h + ei - ej)))
                        f_mp = float(obj_fn(jnp.array(p0_np_h - ei + ej)))
                        f_mm = float(obj_fn(jnp.array(p0_np_h - ei - ej)))
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            except Exception:
                hessian = np.full((n, n), np.nan)
    else:
        hessian = np.full((n, n), np.nan)

    hess_diag = np.diag(hessian)

    # --- eigenvalues and condition number ---
    if compute_hessian and not np.any(np.isnan(hessian)):
        try:
            eigvals = np.linalg.eigvalsh(hessian)
            positive = np.abs(eigvals[np.abs(eigvals) > 1e-16])
            if len(positive) >= 2:
                condition_number = float(positive.max() / positive.min())
            elif len(positive) == 1:
                condition_number = 1.0
            else:
                condition_number = float("inf")
        except np.linalg.LinAlgError:
            eigvals = np.full(n, np.nan)
            condition_number = float("inf")
    else:
        eigvals = np.full(n, np.nan)
        condition_number = float("inf")

    # --- unidentifiable parameter detection ---
    max_sensitivity = float(np.max(norm_sensitivity)) if n > 0 else 1.0
    threshold_abs = sensitivity_threshold * max(max_sensitivity, 1e-30)
    param_dict = optimizable.unflatten_params(params_0_flat)
    # Expand vector-valued parameters to one name per flat element so the
    # names line up 1:1 with the per-element gradients / sensitivities
    # (a bare list(keys()) is per-parameter and would silently truncate the
    # zip below for any array param). Mirrors confidence._expand_param_names.
    param_names = _expand_param_names(param_dict)
    unidentifiable = [
        name
        for name, s in zip(param_names, norm_sensitivity)
        if s < threshold_abs
    ]

    return SensitivityResult(
        param_names=param_names,
        params_0=param_dict,
        objective_value=obj_val,
        gradients=grads,
        normalized_sensitivity=norm_sensitivity,
        hessian=hessian,
        hessian_diagonal=hess_diag,
        eigenvalues=eigvals,
        condition_number=condition_number,
        unidentifiable_params=unidentifiable,
        sensitivity_threshold=sensitivity_threshold,
    )
