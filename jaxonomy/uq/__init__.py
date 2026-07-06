# SPDX-License-Identifier: MIT

"""Uncertainty quantification (UQ) workflows on top of :func:`simulate_batch`.

A first-class wrapper for Monte Carlo, variance-based global sensitivity
(Sobol indices), and one-at-a-time screening (Morris elementary effects)
backed by ``simulate_batch``'s pure-JAX kernel path.

Public entry points:

* Distributions: :class:`Uniform`, :class:`Normal`, :class:`LogNormal`,
  :class:`Triangular`, :class:`Exponential`, :class:`Poisson`.
* Sampling: :func:`sample_parameters` (IID), :func:`latin_hypercube_sample`
  (stratified).
* Sobol: :func:`sobol_indices`, :func:`saltelli_sample`.
* Morris: :func:`morris_screening`, :func:`morris_sample`.

Both :func:`sobol_indices` and :func:`morris_screening` accept either a real
:class:`Diagram` (running ``simulate_batch`` under the hood) or
``diagram=None`` for analytic / surrogate-mode evaluation.

For very large ensembles, prefer :func:`jaxonomy.simulate_distributed` over a
single in-process kernel call: wrap each sampling matrix as a separate
distributed call.
"""

from __future__ import annotations

from .aleatoric_epistemic import (
    conditional_monte_carlo,
    conditional_value_at_risk,
    decompose_variance,
    decompose_variance_sobol,
    importance_sample,
    mean_and_variance_by_kind,
    monte_carlo_with_kinds,
    quantile_summary,
    split_distributions_by_kind,
    value_at_risk,
    vmap_qoi,
)
from .distributions import (
    Bernoulli,
    Beta,
    Categorical,
    CorrelatedMarginals,
    Distribution,
    DistributionKind,
    Exponential,
    Gamma,
    LogNormal,
    MultivariateNormal,
    Normal,
    Pareto,
    Poisson,
    Triangular,
    Uniform,
    Weibull,
)
from .morris import morris_sample, morris_screening
from .quasi_mc import halton_sequence, quasi_monte_carlo, sobol_sequence
from .sampling import (
    latin_hypercube_centered_sample,
    latin_hypercube_sample,
    sample_parameters,
)
from .sobol import saltelli_sample, sobol_indices

__all__ = [
    "Bernoulli",
    "Beta",
    "Categorical",
    "CorrelatedMarginals",
    "Distribution",
    "DistributionKind",
    "Exponential",
    "Gamma",
    "LogNormal",
    "MultivariateNormal",
    "Normal",
    "Pareto",
    "Poisson",
    "Triangular",
    "Uniform",
    "Weibull",
    "conditional_monte_carlo",
    "conditional_value_at_risk",
    "decompose_variance",
    "decompose_variance_sobol",
    "halton_sequence",
    "importance_sample",
    "latin_hypercube_centered_sample",
    "latin_hypercube_sample",
    "mean_and_variance_by_kind",
    "monte_carlo_with_kinds",
    "morris_sample",
    "morris_screening",
    "quantile_summary",
    "quasi_monte_carlo",
    "saltelli_sample",
    "sample_parameters",
    "sobol_indices",
    "sobol_sequence",
    "split_distributions_by_kind",
    "value_at_risk",
    "vmap_qoi",
]
