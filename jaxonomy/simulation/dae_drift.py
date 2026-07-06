# SPDX-License-Identifier: MIT
"""
DAE constraint-residual measurement (T-003).

For a semi-explicit mass-matrix system ``M·ẋ = f(t, x, p)`` where some rows of
``M`` are zero (algebraic constraints), the algebraic rows impose the
implicit relations ``f_a(t, x, p) = 0``. The BDF solver satisfies these
exactly at the solver's rtol/atol each step, but numerical rounding and
integration error in the differential states can cause the algebraic
residual to drift over long simulations — "silent correctness loss" per
T-003.

This module provides the measurement primitive:

  - :func:`compute_constraint_residual` evaluates ``f_a(t, x, p)`` given a
    system and context, returning the per-algebraic-row residual vector.
  - :func:`constraint_residual_norm` returns ``||f_a||_∞`` — the max-abs
    residual, which is the natural quantity to compare against a
    user-configurable threshold.

Full constraint-projection (Newton correction of the algebraic state) and
Baumgarte stabilization are tracked as T-003a follow-ups — the detection
primitive shipped here is sufficient for downstream users to (a) assess
whether their long DAE simulations are drifting, (b) surface a warning
via the new ``SimulatorOptions.dae_drift_threshold``, and (c) build
their own projection / stabilization layer on top if needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..framework import ContextBase, SystemBase


__all__ = [
    "compute_constraint_residual",
    "constraint_residual_norm",
    "algebraic_row_mask",
]


def algebraic_row_mask(system: "SystemBase") -> np.ndarray | None:
    """Boolean mask: True for rows of M that are identically zero.

    Returns ``None`` if the system has no mass matrix (purely ODE form).

    Rows with ``M[i, :] == 0`` correspond to algebraic constraints in the
    semi-explicit form ``M·ẋ = f``.
    """
    if not system.has_mass_matrix:
        return None

    # Diagram.mass_matrix is a list of per-leaf matrices; flatten to a block
    # diagonal to get the combined (n, n) mass matrix.
    from scipy.linalg import block_diag

    mm_tree = system.mass_matrix
    leaves = jax.tree.leaves(mm_tree)
    if not leaves:
        return None
    mm = block_diag(*[np.asarray(leaf) for leaf in leaves])
    # Row is algebraic if all entries are zero (well below eps scale).
    return ~np.any(np.abs(mm) > 1e-12, axis=1)


def compute_constraint_residual(
    system: "SystemBase",
    context: "ContextBase",
) -> jnp.ndarray | None:
    """Return the residual of the algebraic constraints at the given context.

    For a semi-explicit DAE ``M·ẋ = f(t, x, p)``, rows of ``M`` that are
    zero enforce ``f_a(t, x, p) = 0``. This function returns the
    concatenated ``f_a`` vector — ideally near zero on a converged solver
    step, and any growth over simulation time indicates constraint drift.

    Returns ``None`` for systems without a mass matrix (no constraints to
    satisfy; ``M`` is identity).
    """
    mask = algebraic_row_mask(system)
    if mask is None or not mask.any():
        return None

    xcdot = system.eval_time_derivatives(context)
    xcdot_flat = jnp.concatenate(
        [jnp.ravel(leaf) for leaf in jax.tree.leaves(xcdot)]
    )
    return xcdot_flat[jnp.asarray(mask)]


def constraint_residual_norm(
    system: "SystemBase",
    context: "ContextBase",
) -> float | None:
    """Max-abs residual of the algebraic constraints, or ``None`` for pure ODE.

    ``||f_a||_∞`` is the natural comparison quantity for a drift threshold:
    a single violated constraint should trigger the warning even if the
    average residual is tiny.
    """
    residual = compute_constraint_residual(system, context)
    if residual is None:
        return None
    return float(jnp.max(jnp.abs(residual)))
