# SPDX-License-Identifier: MIT
"""T-122-followup-distributions-cdf — forward CDF on every UQ distribution.

Companion to T-122 phase 1 (and the 14 distribution followups): every
``jaxonomy.uq`` distribution now exposes ``cdf(x) -> P(X <= x)``.  The
tests below check three properties for each distribution:

1. Numerical agreement with the corresponding ``scipy.stats`` reference
   to 1e-6 absolute tolerance — this is the canonical correctness bar
   for a CDF implementation.
2. Round-trip ``cdf(ppf(u)) ~ u`` (for distributions that expose both),
   which catches branch / clamping mistakes.
3. Differentiability through ``x`` for continuous distributions
   (:func:`jax.grad` returns finite values at interior points).

``MultivariateNormal`` and ``CorrelatedMarginals`` deliberately raise
:class:`NotImplementedError` — the joint multivariate-normal CDF needs
Genz's quasi-MC algorithm, which is left for a deeper followup.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as sps

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Bernoulli,
    Beta,
    Categorical,
    CorrelatedMarginals,
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

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Uniform                                                                     #
# --------------------------------------------------------------------------- #


def test_uniform_cdf_matches_scipy():
    """``Uniform.cdf`` matches ``scipy.stats.uniform.cdf`` to 1e-6."""
    d = Uniform(2.0, 5.0)
    xs = np.linspace(2.0, 5.0, 11)
    expected = sps.uniform.cdf(xs, loc=2.0, scale=3.0)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_uniform_cdf_boundaries_and_clip():
    """At-low → 0, at-high → 1, below-low → 0 (clipped), above-high → 1."""
    d = Uniform(2.0, 5.0)
    assert float(d.cdf(2.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(5.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(d.cdf(1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(6.0)) == pytest.approx(1.0, abs=1e-12)


def test_uniform_cdf_roundtrips_with_ppf():
    """``cdf(ppf(u)) == u`` for ``u`` in the open unit interval."""
    d = Uniform(-1.0, 4.0)
    for u in (0.05, 0.25, 0.5, 0.75, 0.95):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-10)


def test_uniform_cdf_is_differentiable():
    """``jax.grad(cdf)(x)`` finite inside the support; equals 1/(high-low)."""
    d = Uniform(0.0, 4.0)
    g = float(jax.grad(lambda x: d.cdf(x))(2.0))
    np.testing.assert_allclose(g, 0.25, atol=1e-10)


# --------------------------------------------------------------------------- #
# Normal                                                                      #
# --------------------------------------------------------------------------- #


def test_normal_cdf_matches_scipy():
    d = Normal(loc=1.0, scale=2.0)
    xs = np.linspace(-5.0, 7.0, 13)
    expected = sps.norm.cdf(xs, loc=1.0, scale=2.0)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_normal_cdf_roundtrips_with_ppf():
    d = Normal(loc=0.5, scale=1.5)
    for u in (0.05, 0.3, 0.6, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-7)


def test_normal_cdf_is_differentiable():
    """grad(cdf)(x) = pdf(x); spot-check at the mean."""
    d = Normal(loc=0.0, scale=1.0)
    g = float(jax.grad(lambda x: d.cdf(x))(0.0))
    # Standard-normal pdf at 0 is 1/sqrt(2 pi).
    np.testing.assert_allclose(g, 1.0 / np.sqrt(2.0 * np.pi), atol=1e-7)


# --------------------------------------------------------------------------- #
# LogNormal                                                                   #
# --------------------------------------------------------------------------- #


def test_lognormal_cdf_matches_scipy():
    d = LogNormal(mu=0.0, sigma=1.0)
    xs = np.linspace(0.1, 10.0, 12)
    # scipy.lognorm.cdf(x, s=sigma, scale=exp(mu)).
    expected = sps.lognorm.cdf(xs, s=1.0, scale=np.exp(0.0))
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_lognormal_cdf_zero_below_support():
    d = LogNormal(mu=0.0, sigma=1.0)
    assert float(d.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)


def test_lognormal_cdf_roundtrips_with_ppf():
    d = LogNormal(mu=0.0, sigma=1.0)
    for u in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-7)


def test_lognormal_cdf_is_differentiable():
    d = LogNormal(mu=0.0, sigma=1.0)
    g = float(jax.grad(lambda x: d.cdf(x))(2.0))
    assert np.isfinite(g)
    assert g > 0.0


# --------------------------------------------------------------------------- #
# Triangular                                                                  #
# --------------------------------------------------------------------------- #


def test_triangular_cdf_matches_scipy():
    d = Triangular(low=0.0, mode=0.5, high=1.0)
    xs = np.linspace(0.0, 1.0, 11)
    expected = sps.triang.cdf(xs, c=0.5, loc=0.0, scale=1.0)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_triangular_cdf_boundaries_and_clip():
    d = Triangular(low=0.0, mode=0.5, high=1.0)
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(1.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(d.cdf(2.0)) == pytest.approx(1.0, abs=1e-12)


def test_triangular_cdf_degenerate_modes():
    """``mode == low`` and ``mode == high`` (right-triangular shapes)."""
    d_right = Triangular(low=0.0, mode=0.0, high=1.0)
    d_left = Triangular(low=0.0, mode=1.0, high=1.0)
    np.testing.assert_allclose(
        float(d_right.cdf(0.5)), sps.triang.cdf(0.5, c=0.0), atol=1e-6
    )
    np.testing.assert_allclose(
        float(d_left.cdf(0.5)), sps.triang.cdf(0.5, c=1.0), atol=1e-6
    )


def test_triangular_cdf_roundtrips_with_ppf():
    d = Triangular(low=0.0, mode=0.3, high=2.0)
    for u in (0.05, 0.2, 0.5, 0.8, 0.95):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-7)


def test_triangular_cdf_is_differentiable():
    d = Triangular(low=0.0, mode=0.5, high=1.0)
    g_left = float(jax.grad(lambda x: d.cdf(x))(0.25))
    g_right = float(jax.grad(lambda x: d.cdf(x))(0.75))
    assert np.isfinite(g_left) and np.isfinite(g_right)
    # Slope on left branch at 0.25 = 2*(0.25)/((1)*(0.5)) = 1.0.
    np.testing.assert_allclose(g_left, 1.0, atol=1e-7)


# --------------------------------------------------------------------------- #
# Exponential                                                                 #
# --------------------------------------------------------------------------- #


def test_exponential_cdf_matches_scipy():
    d = Exponential(rate=2.0)
    xs = np.linspace(0.0, 5.0, 11)
    expected = sps.expon.cdf(xs, scale=0.5)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_exponential_cdf_zero_below_zero():
    d = Exponential(rate=2.0)
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)


def test_exponential_cdf_roundtrips_with_ppf():
    d = Exponential(rate=3.0)
    for u in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-7)


def test_exponential_cdf_is_differentiable():
    d = Exponential(rate=2.0)
    # grad(cdf)(x) = rate * exp(-rate * x); at x=1: 2 * exp(-2).
    g = float(jax.grad(lambda x: d.cdf(x))(1.0))
    np.testing.assert_allclose(g, 2.0 * np.exp(-2.0), atol=1e-7)


# --------------------------------------------------------------------------- #
# Poisson                                                                     #
# --------------------------------------------------------------------------- #


def test_poisson_cdf_matches_scipy():
    d = Poisson(rate=3.0)
    ks = np.arange(0, 12)
    expected = sps.poisson.cdf(ks, mu=3.0)
    got = np.asarray(d.cdf(jnp.asarray(ks, dtype=jnp.float64)))
    # gammaincc is implemented in float32 by default in JAX runtime; we
    # request float64 via the input dtype so the precision matches scipy.
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_poisson_cdf_zero_for_negative_k():
    d = Poisson(rate=3.0)
    assert float(d.cdf(-1)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(-5)) == pytest.approx(0.0, abs=1e-12)


def test_poisson_cdf_step_shape():
    """``cdf(k) == cdf(k + 0.5)`` (step-shaped between integers)."""
    d = Poisson(rate=3.0)
    np.testing.assert_allclose(
        float(d.cdf(2.0)), float(d.cdf(2.5)), atol=1e-12
    )


# --------------------------------------------------------------------------- #
# Categorical                                                                 #
# --------------------------------------------------------------------------- #


def test_categorical_cdf_step_function():
    """At-or-above each value, the CDF picks up that category's probability."""
    d = Categorical(values=[0.0, 1.0, 2.0], probs=[0.2, 0.5, 0.3])
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(0.0)) == pytest.approx(0.2, abs=1e-12)
    assert float(d.cdf(0.5)) == pytest.approx(0.2, abs=1e-12)
    assert float(d.cdf(1.0)) == pytest.approx(0.7, abs=1e-12)
    assert float(d.cdf(1.5)) == pytest.approx(0.7, abs=1e-12)
    assert float(d.cdf(2.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(d.cdf(3.0)) == pytest.approx(1.0, abs=1e-12)


def test_categorical_cdf_vector_input():
    d = Categorical(values=[0.0, 1.0, 2.0], probs=[0.2, 0.5, 0.3])
    got = np.asarray(d.cdf(jnp.array([-1.0, 0.0, 1.5, 2.0])))
    np.testing.assert_allclose(got, [0.0, 0.2, 0.7, 1.0], atol=1e-12)


def test_categorical_cdf_rejects_non_sortable():
    """Vector-typed values have no scalar ordering -> TypeError."""
    d = Categorical(values=[[0, 0], [1, 1]], probs=[0.5, 0.5])
    with pytest.raises(TypeError, match="sortable"):
        d.cdf(jnp.array([0.0, 0.0]))


# --------------------------------------------------------------------------- #
# Bernoulli                                                                   #
# --------------------------------------------------------------------------- #


def test_bernoulli_cdf_step_function():
    d = Bernoulli(p=0.3)
    assert float(d.cdf(-0.5)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(0.0)) == pytest.approx(0.7, abs=1e-12)
    assert float(d.cdf(0.5)) == pytest.approx(0.7, abs=1e-12)
    assert float(d.cdf(1.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(d.cdf(2.0)) == pytest.approx(1.0, abs=1e-12)


def test_bernoulli_cdf_edge_p_values():
    """``p == 0`` -> always 0; ``p == 1`` -> always 1."""
    b0 = Bernoulli(p=0.0)
    b1 = Bernoulli(p=1.0)
    # Bernoulli(0): always 0; cdf(0) == 1, cdf(1) == 1.
    assert float(b0.cdf(0.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(b0.cdf(1.0)) == pytest.approx(1.0, abs=1e-12)
    # Bernoulli(1): always 1; cdf(0) == 0, cdf(1) == 1.
    assert float(b1.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(b1.cdf(1.0)) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Beta                                                                        #
# --------------------------------------------------------------------------- #


def test_beta_cdf_matches_scipy():
    d = Beta(alpha=2.0, beta=3.0)
    xs = np.linspace(0.01, 0.99, 11)
    expected = sps.beta.cdf(xs, 2.0, 3.0)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_beta_cdf_boundaries():
    d = Beta(alpha=2.0, beta=3.0)
    assert float(d.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(1.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(d.cdf(-0.5)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(1.5)) == pytest.approx(1.0, abs=1e-12)


def test_beta_cdf_roundtrips_with_ppf():
    d = Beta(alpha=2.0, beta=3.0)
    for u in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-6)


def test_beta_cdf_is_differentiable():
    d = Beta(alpha=2.0, beta=3.0)
    g = float(jax.grad(lambda x: d.cdf(x))(0.5))
    # grad(cdf) = pdf; at x=0.5 with Beta(2,3): pdf = 12 * 0.5 * 0.5^2 = 1.5.
    np.testing.assert_allclose(g, 1.5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Gamma                                                                       #
# --------------------------------------------------------------------------- #


def test_gamma_cdf_matches_scipy():
    d = Gamma(shape_param=2.0, scale=0.5)
    xs = np.linspace(0.01, 5.0, 11)
    expected = sps.gamma.cdf(xs, 2.0, scale=0.5)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_gamma_cdf_zero_below_zero():
    d = Gamma(shape_param=2.0, scale=1.0)
    assert float(d.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)


def test_gamma_cdf_roundtrips_with_ppf():
    d = Gamma(shape_param=2.0, scale=0.5)
    for u in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-6)


def test_gamma_cdf_is_differentiable():
    d = Gamma(shape_param=2.0, scale=1.0)
    g = float(jax.grad(lambda x: d.cdf(x))(1.5))
    assert np.isfinite(g) and g > 0.0


# --------------------------------------------------------------------------- #
# Weibull                                                                     #
# --------------------------------------------------------------------------- #


def test_weibull_cdf_matches_scipy():
    d = Weibull(shape_param=1.5, scale=2.0)
    xs = np.linspace(0.01, 8.0, 11)
    expected = sps.weibull_min.cdf(xs, c=1.5, scale=2.0)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_weibull_cdf_zero_below_zero():
    d = Weibull(shape_param=1.5, scale=2.0)
    assert float(d.cdf(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)


def test_weibull_cdf_roundtrips_with_ppf():
    d = Weibull(shape_param=1.5, scale=2.0)
    for u in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-7)


def test_weibull_cdf_is_differentiable():
    d = Weibull(shape_param=1.5, scale=2.0)
    g = float(jax.grad(lambda x: d.cdf(x))(1.0))
    assert np.isfinite(g) and g > 0.0


# --------------------------------------------------------------------------- #
# Pareto                                                                      #
# --------------------------------------------------------------------------- #


def test_pareto_cdf_matches_scipy():
    d = Pareto(scale=1.0, alpha=2.0)
    xs = np.linspace(1.0, 10.0, 11)
    expected = sps.pareto.cdf(xs, b=2.0, scale=1.0)
    got = np.asarray(d.cdf(jnp.asarray(xs)))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_pareto_cdf_zero_below_scale():
    d = Pareto(scale=1.0, alpha=2.0)
    assert float(d.cdf(0.5)) == pytest.approx(0.0, abs=1e-12)
    assert float(d.cdf(-1.0)) == pytest.approx(0.0, abs=1e-12)
    # At the support boundary cdf == 0.
    assert float(d.cdf(1.0)) == pytest.approx(0.0, abs=1e-12)


def test_pareto_cdf_roundtrips_with_ppf():
    d = Pareto(scale=1.0, alpha=2.0)
    for u in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(float(d.cdf(d.ppf(u))), u, atol=1e-7)


def test_pareto_cdf_is_differentiable():
    d = Pareto(scale=1.0, alpha=2.0)
    g = float(jax.grad(lambda x: d.cdf(x))(2.0))
    # grad(cdf)(x) = pdf(x) = alpha * scale^alpha / x^(alpha+1)
    #              = 2 * 1 / 8 = 0.25.
    np.testing.assert_allclose(g, 0.25, atol=1e-7)


# --------------------------------------------------------------------------- #
# MultivariateNormal / CorrelatedMarginals — deferred                         #
# --------------------------------------------------------------------------- #


def test_multivariate_normal_cdf():
    """Joint multivariate-normal CDF via Genz quasi-MC (now implemented)."""
    d = MultivariateNormal(
        means=jnp.array([0.0, 0.0]),
        cov=jnp.eye(2),
    )
    # Independent standard bivariate at the origin: P(X<=0, Y<=0) = 0.5**2.
    assert jnp.allclose(d.cdf(jnp.array([0.0, 0.0])), 0.25, atol=1e-2)


def test_correlated_marginals_cdf_raises():
    """Joint copula CDF deferred (depends on multivariate-normal CDF)."""
    d = CorrelatedMarginals(
        marginals=(Normal(0.0, 1.0), Normal(0.0, 1.0)),
        corr_matrix=jnp.eye(2),
    )
    with pytest.raises(NotImplementedError, match="multivariate"):
        d.cdf(jnp.array([0.0, 0.0]))
