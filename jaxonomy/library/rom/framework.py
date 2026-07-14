# SPDX-License-Identifier: MIT

"""Unified reduction entry point (T-143).

Every reducer in this subpackage returns a first-class, simulatable Jaxonomy
object; :func:`reduce` is the one-call front door that wraps whichever method
you name in a :class:`ReducedOrderModel` carrying the reduced ``system`` plus
provenance (method, orders, error bound / spectrum / basis where the method
produces one).

Two families are reachable through :func:`reduce`:

* **Linear MOR** — ``target`` is an LTI model (a
  :class:`~jaxonomy.library.linear_system.LinearizedSystem`, an
  :class:`~jaxonomy.library.linear_system.LTISystem`, or a raw ``(A, B, C, D)``
  tuple) and ``method`` is one of ``"balred"`` / ``"balanced_truncation"``,
  ``"minreal"`` / ``"minimal_realization"``, ``"modal"`` / ``"modal_truncation"``,
  ``"residualize"``. Returns a reduced ``LTISystem`` (or ``LTISystemDiscrete``).
* **Data-driven operator ROM** — ``target`` is snapshot data (a
  :class:`~jaxonomy.library.rom.snapshots.SnapshotData` or a raw snapshot
  matrix) and ``method`` is ``"dmd"``, ``"dmdc"``, or ``"edmd"``. Returns a
  discrete-time predictor block.

Projection ROM (POD–Galerkin / DEIM) needs the full-order RHS *callable*, not
just a system or snapshots, so it is driven directly through
:func:`~jaxonomy.library.rom.pod.galerkin_reduce` /
:func:`~jaxonomy.library.rom.pod.deim_galerkin_reduce` rather than this
dispatcher.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..linear_system import LTISystemDiscrete, LinearizedSystem
from . import linear_mor as _lmor
from .snapshots import SnapshotData
from .dmd import dmd as _dmd, dmdc as _dmdc, DMDForecaster
from .koopman import edmd as _edmd, KoopmanPredictor

__all__ = ["ReducedOrderModel", "reduce"]


_LINEAR_METHODS = {
    "balred": "balred",
    "balanced_truncation": "balred",
    "minreal": "minreal",
    "minimal_realization": "minreal",
    "modal": "modal",
    "modal_truncation": "modal",
    "residualize": "residualize",
}
_DATA_METHODS = {"dmd", "dmdc", "edmd"}


@dataclass
class ReducedOrderModel:
    """A reduced model plus its provenance.

    Attributes:
        system: The reduced, simulatable Jaxonomy block — an ``LTISystem`` /
            ``LTISystemDiscrete`` for linear MOR, or a discrete-time predictor
            ``LeafSystem`` for data-driven methods. Drop it straight into a
            diagram or ``jaxonomy.simulate``.
        method: The reduction method that produced it.
        full_order: State dimension of the source model (when known).
        reduced_order: State dimension of ``system``.
        info: Method-specific extras — e.g. ``error_bound`` and ``hsv`` for
            balanced truncation, ``eigenvalues`` / ``basis`` for DMD, and the
            raw ``result`` object from the underlying routine.
    """

    system: Any
    method: str
    full_order: Optional[int] = None
    reduced_order: Optional[int] = None
    info: dict = field(default_factory=dict)

    def to_block(self):
        """Return the reduced Jaxonomy block (alias for ``.system``)."""
        return self.system

    def __repr__(self):
        return (
            f"ReducedOrderModel(method={self.method!r}, "
            f"full_order={self.full_order}, reduced_order={self.reduced_order})"
        )


def _as_linearized(target) -> LinearizedSystem:
    """Coerce an LTI target to a ``LinearizedSystem`` for the linear-MOR path."""
    if isinstance(target, LinearizedSystem):
        return target
    if isinstance(target, (tuple, list)) and len(target) == 4:
        A, B, C, D = target
        return LinearizedSystem(
            np.asarray(A, float), np.asarray(B, float),
            np.asarray(C, float), np.asarray(D, float), {},
        )
    if all(hasattr(target, k) for k in ("A", "B", "C", "D")):
        dt = getattr(target, "dt", None)
        return LinearizedSystem(
            np.asarray(target.A, float), np.asarray(target.B, float),
            np.asarray(target.C, float), np.asarray(target.D, float), {}, dt,
        )
    raise TypeError(
        "Linear MOR needs a LinearizedSystem, LTISystem, or (A, B, C, D) tuple; "
        f"got {type(target).__name__}."
    )


def _snapshot_matrix(target):
    """Return (X, Xp, U, x0) from a SnapshotData or a raw snapshot matrix."""
    if isinstance(target, SnapshotData):
        X = np.asarray(target.X, float)
        U = None if getattr(target, "inputs", None) is None else np.asarray(target.inputs, float)
        return X, None, U, X[:, 0]
    X = np.asarray(target, float)
    return X, None, None, X[:, 0]


def _reduced_lti(red: LinearizedSystem):
    """Wrap a reduced LinearizedSystem as the matching LTI block."""
    if getattr(red, "dt", None) is None:
        return red.to_lti()
    return LTISystemDiscrete(red.A, red.B, red.C, red.D, red.dt)


def reduce(target, method="balred", *, order=None, tol=None, dt=1.0, **kwargs):
    """Reduce ``target`` by ``method`` and return a :class:`ReducedOrderModel`.

    Args:
        target: An LTI model (linear MOR) or snapshot data (data-driven).
        method: See the module docstring for the supported names.
        order: Target reduced order, where the method takes one.
        tol: Energy/tolerance selector for balanced truncation / ``minreal``.
        dt: Sampling period for the data-driven predictor blocks.
        **kwargs: Forwarded to the underlying routine (e.g. ``keep=`` for modal
            methods, ``dictionary=`` and ``U=`` for eDMD, ``initial_state=``).

    Returns:
        A :class:`ReducedOrderModel` whose ``.system`` is ready to simulate.
    """
    key = method.lower()

    if key in _LINEAR_METHODS:
        canonical = _LINEAR_METHODS[key]
        lin = _as_linearized(target)
        if canonical == "balred":
            red = _lmor.balred(lin, order=order, tol=tol)
        elif canonical == "minreal":
            red = _lmor.minreal(lin, **({"tol": tol} if tol is not None else {}), **kwargs)
        elif canonical == "modal":
            red = _lmor.modal_truncation(lin, order=order, keep=kwargs.get("keep"))
        else:  # residualize
            red = _lmor.residualize(lin, order=order, keep=kwargs.get("keep"))
        info = {"result": red}
        for attr in ("hsv", "error_bound", "reduced_order"):
            if hasattr(red, attr):
                info[attr] = getattr(red, attr)
        return ReducedOrderModel(
            system=_reduced_lti(red),
            method=canonical,
            full_order=int(np.asarray(lin.A).shape[0]),
            reduced_order=int(np.asarray(red.A).shape[0]),
            info=info,
        )

    if key in _DATA_METHODS:
        X, _, U, x0 = _snapshot_matrix(target)
        x0 = kwargs.pop("initial_state", x0)
        if key == "dmd":
            res = _dmd(X, rank=order)
            # Real full one-step operator from the (conjugate-symmetric) DMD
            # spectrum: A = Re(Φ diag(λ) Φ⁺). Tu et al. 2014.
            A_full = np.real(
                res.modes @ np.diag(res.eigenvalues) @ np.linalg.pinv(res.modes)
            )
            system = DMDForecaster(A=A_full, dt=dt, initial_state=np.asarray(x0, float))
            return ReducedOrderModel(
                system=system, method="dmd",
                full_order=A_full.shape[0], reduced_order=len(res.eigenvalues),
                info={"result": res, "eigenvalues": res.eigenvalues},
            )
        if key == "dmdc":
            Xp = kwargs.pop("Xp", None)
            U = kwargs.pop("U", U)
            if U is None:
                raise ValueError("dmdc needs control inputs U (kwarg or SnapshotData.inputs).")
            if Xp is None:
                X, Xp = X[:, :-1], X[:, 1:]
                U = np.asarray(U)[:, : Xp.shape[1]]
            res = _dmdc(X, Xp, U, rank=order)
            system = DMDForecaster(A=res.A, B=res.B, dt=dt, initial_state=np.asarray(x0, float))
            return ReducedOrderModel(
                system=system, method="dmdc",
                full_order=np.asarray(res.A).shape[0], reduced_order=np.asarray(res.A).shape[0],
                info={"result": res, "eigenvalues": res.eigenvalues},
            )
        # edmd
        dictionary = kwargs.pop("dictionary", None)
        if dictionary is None:
            raise ValueError("edmd needs a `dictionary` observable callable.")
        Xp = kwargs.pop("Xp", None)
        U = kwargs.pop("U", U)
        if Xp is None:
            X, Xp = X[:, :-1], X[:, 1:]
            if U is not None:
                U = np.asarray(U)[:, : Xp.shape[1]]
        res = _edmd(X, Xp, dictionary, U=U)
        system = KoopmanPredictor(
            K=res.K, C=res.C, dictionary=res.dictionary, B=res.B, dt=dt,
            initial_state=np.asarray(x0, float),
        )
        return ReducedOrderModel(
            system=system, method="edmd",
            full_order=np.asarray(res.C).shape[0], reduced_order=np.asarray(res.K).shape[0],
            info={"result": res},
        )

    if key in ("pod", "galerkin", "petrov_galerkin", "lspg", "deim"):
        raise ValueError(
            f"Projection ROM ({method!r}) needs the full-order RHS callable — "
            "use galerkin_reduce(rhs_fn, basis, ...) or "
            "deim_galerkin_reduce(...) directly."
        )

    raise ValueError(f"Unknown reduction method {method!r}.")
