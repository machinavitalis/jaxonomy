# SPDX-License-Identifier: MIT

"""Reduced-order modeling and statistical surrogates (T-143..T-151).

Given a full-order model (an LTI system, a nonlinear ODE/DAE RHS, or snapshot
data from either) this subpackage builds a cheaper reduced model that is still
a first-class, differentiable, simulatable Jaxonomy object. See
``docs/scope/rom.md`` for the method-selection guide.

Families:

* **Linear MOR** — :func:`balred`, :func:`minreal`, :func:`modal_truncation`,
  :func:`residualize`, plus the gramian/Hankel primitives.
* **Projection ROM** — :func:`pod_basis`, :func:`galerkin_reduce`,
  :func:`deim`, :func:`deim_galerkin_reduce`.
* **Data-driven operator ROM** — :func:`dmd`, :func:`dmdc`, :func:`era`,
  :func:`edmd`, and the :class:`DMDForecaster` / :class:`KoopmanPredictor`
  blocks.
* **Statistical surrogates** — :func:`fit_gp`, :func:`fit_pce`, :func:`fit_rbf`
  and the :class:`GaussianProcess` / :class:`PolynomialChaos` /
  :class:`RadialBasisSurrogate` blocks.
* **Front door** — :func:`reduce` returns a :class:`ReducedOrderModel`.
"""

from .framework import ReducedOrderModel, reduce

from .snapshots import (
    SnapshotData,
    collect_snapshots,
    relative_error,
    retained_energy,
    projection_error,
)
from .linear_mor import (
    controllability_gramian,
    observability_gramian,
    hankel_singular_values,
    balanced_realization,
    balanced_truncation,
    balred,
    minimal_realization,
    minreal,
    modal_truncation,
    residualize,
)
from .pod import (
    pod_basis,
    galerkin_reduce,
    deim,
    deim_galerkin_reduce,
)
from .dmd import (
    DMDResult,
    DMDcResult,
    ERAResult,
    dmd,
    dmdc,
    era,
    DMDForecaster,
)
from .koopman import (
    identity_dictionary,
    polynomial_dictionary,
    rbf_dictionary,
    EDMDResult,
    edmd,
    KoopmanPredictor,
)
from .surrogates import (
    GPModel,
    fit_gp,
    GaussianProcess,
    PCEModel,
    fit_pce,
    PolynomialChaos,
    RBFModel,
    fit_rbf,
    RadialBasisSurrogate,
)

__all__ = [
    # front door
    "ReducedOrderModel",
    "reduce",
    # snapshots + metrics
    "SnapshotData",
    "collect_snapshots",
    "relative_error",
    "retained_energy",
    "projection_error",
    # linear MOR
    "controllability_gramian",
    "observability_gramian",
    "hankel_singular_values",
    "balanced_realization",
    "balanced_truncation",
    "balred",
    "minimal_realization",
    "minreal",
    "modal_truncation",
    "residualize",
    # projection ROM
    "pod_basis",
    "galerkin_reduce",
    "deim",
    "deim_galerkin_reduce",
    # data-driven operator ROM
    "DMDResult",
    "DMDcResult",
    "ERAResult",
    "dmd",
    "dmdc",
    "era",
    "DMDForecaster",
    "identity_dictionary",
    "polynomial_dictionary",
    "rbf_dictionary",
    "EDMDResult",
    "edmd",
    "KoopmanPredictor",
    # statistical surrogates
    "GPModel",
    "fit_gp",
    "GaussianProcess",
    "PCEModel",
    "fit_pce",
    "PolynomialChaos",
    "RBFModel",
    "fit_rbf",
    "RadialBasisSurrogate",
]
