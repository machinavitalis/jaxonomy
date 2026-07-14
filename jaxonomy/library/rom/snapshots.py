# SPDX-License-Identifier: MIT

"""Snapshot collection and reduced-order-model (ROM) accuracy metrics (T-145).

A *snapshot matrix* stacks state/output samples column-wise,
``X.shape == (n_features, n_samples)``, and is the raw material for
data-driven model reduction (POD/DEIM, see :mod:`jaxonomy.library.rom.pod`).
This module provides a light container plus the standard error/energy metrics
used to size and validate a reduced basis.

References:
    Berkooz, Holmes & Lumley (1993), "The proper orthogonal decomposition in
    the analysis of turbulent flows", Annu. Rev. Fluid Mech. 25:539-575.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "SnapshotData",
    "collect_snapshots",
    "relative_error",
    "retained_energy",
    "projection_error",
]


@dataclass
class SnapshotData:
    """Container for a column-wise snapshot matrix.

    Attributes:
        X: State/output snapshots, shape ``(n_features, n_samples)``.
        time: Optional sample times, shape ``(n_samples,)``.
        inputs: Optional input snapshots ``U``, shape ``(n_inputs, n_samples)``.
        Xdot: Optional time-derivative snapshots, shape ``(n_features, n_samples)``.
    """

    X: np.ndarray
    time: Optional[np.ndarray] = None
    inputs: Optional[np.ndarray] = None
    Xdot: Optional[np.ndarray] = None

    @property
    def n_features(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[1])


def _as_columns(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as ``(dim, n_samples)`` given a ``(n_samples,)`` or
    ``(n_samples, dim)`` recorded signal."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr.T
    raise ValueError(
        f"Recorded signals must be 1-D or 2-D; got shape {arr.shape}."
    )


def collect_snapshots(
    results,
    signals: Optional[Sequence[str]] = None,
) -> SnapshotData:
    """Assemble a snapshot matrix from a jaxonomy ``SimulationResults``.

    Selected recorded signals (``results.outputs``) are stacked column-wise
    into ``X`` of shape ``(n_features, n_samples)`` where ``n_features`` is the
    total width of the selected signals and ``n_samples == len(results.time)``.

    Args:
        results: A ``SimulationResults`` with ``.time`` and an ``.outputs`` dict
            mapping signal name -> array of shape ``(n_samples,)`` or
            ``(n_samples, dim)``.
        signals: Names to include (in order). ``None`` selects every output.

    Returns:
        A :class:`SnapshotData` with ``X`` and ``time`` populated.
    """
    if results.outputs is None:
        raise ValueError(
            "results.outputs is None; run simulate(...) with recorded_signals."
        )
    if signals is None:
        signals = list(results.outputs.keys())

    blocks = []
    for name in signals:
        if name not in results.outputs:
            raise KeyError(
                f"signal {name!r} not in recorded outputs "
                f"{list(results.outputs.keys())}"
            )
        blocks.append(_as_columns(results.outputs[name]))

    X = np.vstack(blocks)
    time = None if results.time is None else np.asarray(results.time)
    return SnapshotData(X=X, time=time)


def relative_error(x_true, x_approx) -> float:
    """Relative L2 (Frobenius) error ``‖x_true − x_approx‖ / ‖x_true‖``.

    Works for a single trajectory column or a full snapshot matrix.
    """
    x_true = np.asarray(x_true)
    x_approx = np.asarray(x_approx)
    denom = np.linalg.norm(x_true)
    if denom == 0.0:
        return float(np.linalg.norm(x_true - x_approx))
    return float(np.linalg.norm(x_true - x_approx) / denom)


def retained_energy(singular_values, r: int) -> float:
    """Fraction of total energy captured by the first ``r`` POD modes.

    Energy is measured in squared singular values,
    ``Σ_{i<r} σ_i² / Σ_i σ_i²`` — monotonically non-decreasing in ``r``.
    """
    s = np.asarray(singular_values, dtype=float)
    total = float(np.sum(s**2))
    if total == 0.0:
        return 0.0
    r = int(max(0, min(r, s.shape[0])))
    return float(np.sum(s[:r] ** 2) / total)


def projection_error(X, basis) -> float:
    """Relative projection error ``‖X − ΦΦᵀX‖ / ‖X‖`` of ``X`` onto ``basis``.

    ``basis`` (``Φ``) is assumed to have orthonormal columns.
    """
    X = np.asarray(X)
    Phi = np.asarray(basis)
    X_proj = Phi @ (Phi.T @ X)
    denom = np.linalg.norm(X)
    if denom == 0.0:
        return float(np.linalg.norm(X - X_proj))
    return float(np.linalg.norm(X - X_proj) / denom)
