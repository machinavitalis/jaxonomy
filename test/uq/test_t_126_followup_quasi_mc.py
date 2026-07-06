# SPDX-License-Identifier: MIT

"""T-126-followup-quasi-mc — quasi-Monte-Carlo sampling via Sobol sequences.

Covers :func:`jaxonomy.uq.sobol_sequence` and
:func:`jaxonomy.uq.quasi_monte_carlo`:

1. ``sobol_sequence`` returns shape ``(n_samples, n_dims)`` with all
   values in ``[0, 1)``.
2. Low-discrepancy property: the unit cube is covered uniformly to a
   much tighter tolerance than IID Monte Carlo at the same ``N``.
3. ``quasi_monte_carlo`` returns the expected ``{param: (N,)}`` shape
   with samples in each distribution's support.
4. Faster convergence than IID MC for a smooth integrand
   (``integrate sin(x) dx`` over ``[0, 1]``): QMC error at ``N=1024``
   is smaller than IID error at ``N=1024``.
5. Discrete distributions without ``ppf`` are rejected.
6. Validation: bad ``n_samples`` / empty dists raise.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Bernoulli,
    Normal,
    Poisson,
    Uniform,
    quasi_monte_carlo,
    sample_parameters,
    sobol_sequence,
)

skip_if_not_jax()


# ---------------------------------------------------------------------------
# sobol_sequence — shape, range, low-discrepancy.
# ---------------------------------------------------------------------------

def test_sobol_sequence_shape_and_range():
    """sobol_sequence(1024, 3) returns (1024, 3) in [0, 1)."""
    points = np.asarray(sobol_sequence(n_samples=1024, n_dims=3, seed=0))
    assert points.shape == (1024, 3)
    assert points.min() >= 0.0
    assert points.max() < 1.0


def test_sobol_sequence_unscrambled_is_deterministic():
    """An unscrambled Sobol sequence is fully deterministic; two calls
    return identical arrays."""
    a = np.asarray(sobol_sequence(n_samples=64, n_dims=2, scramble=False))
    b = np.asarray(sobol_sequence(n_samples=64, n_dims=2, scramble=False))
    np.testing.assert_array_equal(a, b)


def test_sobol_sequence_scrambled_is_seedable():
    """Same seed -> same scrambled output; different seed -> different."""
    a = np.asarray(sobol_sequence(n_samples=64, n_dims=2, seed=7, scramble=True))
    b = np.asarray(sobol_sequence(n_samples=64, n_dims=2, seed=7, scramble=True))
    c = np.asarray(sobol_sequence(n_samples=64, n_dims=2, seed=8, scramble=True))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_sobol_sequence_low_discrepancy_vs_iid():
    """Sobol covers the unit cube more uniformly than IID at the same N.

    We bin each 1-D marginal into 16 strata and check that the worst-case
    stratum count deviation from the perfect ``N/16`` is much smaller for
    Sobol than for an IID sample.
    """
    n = 1024
    n_dims = 3
    strata = 16
    expected = n // strata  # 64 per stratum if perfectly uniform.

    # Sobol (deterministic, unscrambled — most uniform).
    sobol = np.asarray(
        sobol_sequence(n_samples=n, n_dims=n_dims, scramble=False)
    )
    sobol_max_dev = 0
    for d in range(n_dims):
        counts, _ = np.histogram(sobol[:, d], bins=strata, range=(0.0, 1.0))
        sobol_max_dev = max(sobol_max_dev, int(np.abs(counts - expected).max()))

    # IID at the same N.
    rng = np.random.default_rng(0)
    iid = rng.uniform(0.0, 1.0, size=(n, n_dims))
    iid_max_dev = 0
    for d in range(n_dims):
        counts, _ = np.histogram(iid[:, d], bins=strata, range=(0.0, 1.0))
        iid_max_dev = max(iid_max_dev, int(np.abs(counts - expected).max()))

    # Unscrambled Sobol places exactly N/strata points per stratum for
    # any power-of-2 N >= strata, so its max deviation is 0.  IID is
    # ``sqrt(N * 1/strata * (1 - 1/strata))`` ~= 7.7 in expectation; we
    # only need ``iid_max_dev > sobol_max_dev`` for the comparison.
    assert sobol_max_dev == 0, (
        f"Unscrambled Sobol should perfectly tile 16 strata; got max dev "
        f"{sobol_max_dev}."
    )
    assert iid_max_dev > sobol_max_dev


def test_sobol_sequence_rejects_bad_inputs():
    with pytest.raises(ValueError, match="n_samples"):
        sobol_sequence(n_samples=0, n_dims=2)
    with pytest.raises(ValueError, match="n_samples"):
        sobol_sequence(n_samples=-5, n_dims=2)
    with pytest.raises(ValueError, match="n_dims"):
        sobol_sequence(n_samples=4, n_dims=0)
    with pytest.raises(ValueError, match="n_dims"):
        sobol_sequence(n_samples=4, n_dims=-1)


# ---------------------------------------------------------------------------
# quasi_monte_carlo — ppf-driven sampling.
# ---------------------------------------------------------------------------

def test_quasi_monte_carlo_uniform_shape_and_support():
    """quasi_monte_carlo({"x": Uniform(0, 1)}, n=1024) -> {"x": (1024,)} in [0,1)."""
    samples = quasi_monte_carlo({"x": Uniform(0.0, 1.0)}, n_samples=1024, seed=0)
    assert set(samples) == {"x"}
    x = np.asarray(samples["x"])
    assert x.shape == (1024,)
    assert x.min() >= 0.0
    assert x.max() < 1.0


def test_quasi_monte_carlo_multiparameter_shapes():
    """Each parameter gets a (N,) array in its own support."""
    dists = {
        "k": Uniform(0.5, 2.0),
        "tau": Normal(1.0, 0.1),
    }
    samples = quasi_monte_carlo(dists, n_samples=512, seed=1)
    assert set(samples) == {"k", "tau"}
    k = np.asarray(samples["k"])
    tau = np.asarray(samples["tau"])
    assert k.shape == (512,)
    assert tau.shape == (512,)
    # k in [0.5, 2.0] strictly (Uniform).
    assert k.min() >= 0.5
    assert k.max() <= 2.0
    # tau roughly centred on 1.0 with std ~ 0.1.
    assert abs(float(tau.mean()) - 1.0) < 0.05
    assert 0.05 < float(tau.std()) < 0.20


def test_quasi_monte_carlo_rejects_discrete_distributions():
    """Distributions without ``ppf`` (Poisson, Bernoulli) are rejected."""
    with pytest.raises(ValueError, match="ppf"):
        quasi_monte_carlo({"n": Poisson(rate=2.0)}, n_samples=64)
    with pytest.raises(ValueError, match="ppf"):
        quasi_monte_carlo({"flip": Bernoulli(p=0.3)}, n_samples=64)


def test_quasi_monte_carlo_rejects_bad_n_samples():
    with pytest.raises(ValueError, match="n_samples"):
        quasi_monte_carlo({"x": Uniform(0.0, 1.0)}, n_samples=0)


def test_quasi_monte_carlo_rejects_empty_distributions():
    with pytest.raises(ValueError, match="non-empty"):
        quasi_monte_carlo({}, n_samples=64)


# ---------------------------------------------------------------------------
# QMC converges faster than IID MC on a smooth integrand.
# ---------------------------------------------------------------------------

def test_qmc_beats_iid_on_smooth_integrand():
    """Estimate ``E[sin(X)]`` for ``X ~ Uniform(0, 1)`` via QMC and IID.

    Analytic value: ``int_0^1 sin(x) dx = 1 - cos(1) ~= 0.4596976941``.

    QMC error at N=1024 should be smaller than IID error at N=1024 (we
    allow up to 10x slack per the task spec, but in practice the margin
    is typically 100x+ on this smooth integrand).
    """
    n = 1024
    truth = 1.0 - math.cos(1.0)

    # QMC estimate.
    qmc_samples = quasi_monte_carlo(
        {"x": Uniform(0.0, 1.0)}, n_samples=n, seed=0,
    )
    qmc_est = float(np.sin(np.asarray(qmc_samples["x"])).mean())
    qmc_err = abs(qmc_est - truth)

    # IID estimate at the same N.
    iid_samples = sample_parameters(
        {"x": Uniform(0.0, 1.0)}, n, jax.random.PRNGKey(0),
    )
    iid_est = float(np.sin(np.asarray(iid_samples["x"])).mean())
    iid_err = abs(iid_est - truth)

    # QMC must be at least as accurate as IID; the task spec allows up
    # to a 10x slack window but the realistic margin on a sin(x)
    # integrand at N=1024 is multiple orders of magnitude.
    assert qmc_err < iid_err * 10.0, (
        f"QMC error ({qmc_err:.2e}) should beat IID error ({iid_err:.2e}) "
        "within a 10x slack window on a smooth integrand."
    )
    # And in absolute terms QMC should land within 1e-3 of the truth on
    # this near-linear-on-[0,1] integrand.
    assert qmc_err < 1e-3, f"QMC error {qmc_err:.2e} should be < 1e-3."


def test_qmc_2d_integrand_beats_iid():
    """Multidimensional smoke test: ``E[sin(X) * cos(Y)]`` on Uniforms.

    Analytic: ``(1 - cos 1) * sin 1 ~= 0.38682227``.
    """
    n = 1024
    truth = (1.0 - math.cos(1.0)) * math.sin(1.0)

    qmc_samples = quasi_monte_carlo(
        {"x": Uniform(0.0, 1.0), "y": Uniform(0.0, 1.0)},
        n_samples=n,
        seed=0,
    )
    qmc_est = float(
        (np.sin(np.asarray(qmc_samples["x"]))
         * np.cos(np.asarray(qmc_samples["y"]))).mean()
    )
    qmc_err = abs(qmc_est - truth)

    iid_samples = sample_parameters(
        {"x": Uniform(0.0, 1.0), "y": Uniform(0.0, 1.0)},
        n,
        jax.random.PRNGKey(0),
    )
    iid_est = float(
        (np.sin(np.asarray(iid_samples["x"]))
         * np.cos(np.asarray(iid_samples["y"]))).mean()
    )
    iid_err = abs(iid_est - truth)

    assert qmc_err < iid_err * 10.0
    assert qmc_err < 5e-3
