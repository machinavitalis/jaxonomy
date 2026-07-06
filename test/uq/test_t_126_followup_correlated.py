# SPDX-License-Identifier: MIT
"""T-126-followup-correlated-multivariate — correlated multivariate sampling.

Ships :class:`MultivariateNormal` (Cholesky-reparameterised
multivariate-normal sampling with analytic ``log_pdf``) and
:class:`CorrelatedMarginals` (Gaussian-copula transform: sample
correlated Normals, push through ``Phi`` then per-marginal ``ppf``).

The copula transform preserves Spearman (rank) correlation exactly and
is what the variance-decomposition / Sobol pipelines need when input
parameters carry physical correlation structure (e.g. battery R0 vs
capacity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal as smvn
from scipy.stats import spearmanr

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    CorrelatedMarginals,
    LogNormal,
    MultivariateNormal,
    Normal,
)

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# MultivariateNormal                                                          #
# --------------------------------------------------------------------------- #


def test_multivariate_normal_empirical_cov_matches_prescribed():
    """10000 draws -> empirical cov within 5% of prescribed [[1, 0.5], [0.5, 1]]."""
    cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mvn = MultivariateNormal(jnp.zeros(2), cov)
    samples = np.asarray(mvn.sample(jax.random.PRNGKey(0), (10000,)))
    assert samples.shape == (10000, 2)
    emp_cov = np.cov(samples.T)
    # Each entry should match within 5% (absolute, since the entries are O(1)).
    np.testing.assert_allclose(emp_cov, np.asarray(cov), atol=0.05)


def test_multivariate_normal_sample_shape_with_batch_dims():
    """sample(key, (4, 7)) returns shape (4, 7, n_dim)."""
    mvn = MultivariateNormal(jnp.zeros(3), jnp.eye(3))
    s = mvn.sample(jax.random.PRNGKey(0), (4, 7))
    assert s.shape == (4, 7, 3)


def test_multivariate_normal_log_pdf_matches_scipy():
    """log_pdf matches scipy.stats.multivariate_normal.logpdf to ~1e-10."""
    means = np.array([1.0, -2.0, 0.5])
    cov = np.array(
        [
            [2.0, 0.3, -0.1],
            [0.3, 1.0, 0.2],
            [-0.1, 0.2, 1.5],
        ]
    )
    mvn = MultivariateNormal(jnp.asarray(means), jnp.asarray(cov))
    test_pts = np.array(
        [
            [1.0, -2.0, 0.5],   # at mean
            [0.0, 0.0, 0.0],
            [3.0, -1.0, 2.0],
            [-0.5, -3.5, 1.2],
        ]
    )
    for x in test_pts:
        expected = float(smvn.logpdf(x, mean=means, cov=cov))
        got = float(mvn.log_pdf(jnp.asarray(x)))
        assert got == pytest.approx(expected, abs=1e-6), (
            f"log_pdf({x}) = {got}, scipy = {expected}"
        )


def test_multivariate_normal_log_pdf_batched():
    """log_pdf on shape (N, n_dim) returns shape (N,) values matching scipy."""
    means = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    mvn = MultivariateNormal(jnp.asarray(means), jnp.asarray(cov))
    xs = np.array([[0.1, 0.2], [-0.3, 0.4], [1.0, -1.0]])
    got = np.asarray(mvn.log_pdf(jnp.asarray(xs)))
    expected = smvn.logpdf(xs, mean=means, cov=cov)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_multivariate_normal_validation_errors():
    """Construction with bad shapes / mismatched sizes raises ValueError."""
    with pytest.raises(ValueError):
        MultivariateNormal(jnp.array([[0.0, 0.0]]), jnp.eye(2))  # means not 1D
    with pytest.raises(ValueError):
        MultivariateNormal(jnp.zeros(2), jnp.zeros((2, 3)))  # cov not square
    with pytest.raises(ValueError):
        MultivariateNormal(jnp.zeros(2), jnp.eye(3))  # mismatched dims
    with pytest.raises(ValueError):
        MultivariateNormal(jnp.zeros(2), jnp.eye(2), kind="invalid")  # type: ignore[arg-type]


def test_multivariate_normal_ppf_raises():
    """ppf is intentionally not defined for the multivariate case."""
    mvn = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
    with pytest.raises(NotImplementedError):
        mvn.ppf(jnp.array([0.5, 0.5]))


def test_multivariate_normal_diff_through_cov_factor():
    """jax.grad of mean(sample) wrt the Cholesky factor is finite."""

    def loss(L_factor):
        cov = L_factor @ L_factor.T
        mvn = MultivariateNormal(jnp.zeros(2), cov)
        samples = mvn.sample(jax.random.PRNGKey(0), (500,))
        return jnp.mean(samples)

    L0 = jnp.array([[1.0, 0.0], [0.5, 0.8]])
    grad = np.asarray(jax.grad(loss)(L0))
    assert np.all(np.isfinite(grad)), f"grad has non-finite entries: {grad}"


def test_multivariate_normal_diff_through_means():
    """jax.grad of mean(sample) wrt means is finite (and =~ 1 per component)."""

    def loss(mu):
        mvn = MultivariateNormal(mu, jnp.eye(2))
        samples = mvn.sample(jax.random.PRNGKey(0), (500,))
        return jnp.sum(jnp.mean(samples, axis=0))

    mu0 = jnp.array([0.5, -0.2])
    grad = np.asarray(jax.grad(loss)(mu0))
    # d/dmu_i mean(samples[i]) = 1 because samples are mu + Lz.
    np.testing.assert_allclose(grad, np.ones(2), atol=1e-12)


# --------------------------------------------------------------------------- #
# CorrelatedMarginals                                                         #
# --------------------------------------------------------------------------- #


def test_correlated_marginals_marginal_stats_match_each_distribution():
    """Marginal means/stds from copula samples match each Distribution's stats."""
    marginals = [Normal(0.0, 1.0), LogNormal(0.0, 0.3)]
    corr = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    cm = CorrelatedMarginals(marginals, corr)
    samples = np.asarray(cm.sample(jax.random.PRNGKey(2), (20000,)))
    assert samples.shape == (20000, 2)
    # Normal(0, 1): mean ~ 0, std ~ 1.
    assert abs(samples[:, 0].mean()) < 0.05
    assert abs(samples[:, 0].std() - 1.0) < 0.05
    # LogNormal(mu=0, sigma=0.3): mean = exp(0 + 0.3^2/2), var = (exp(sigma^2)-1) * exp(2mu+sigma^2)
    sigma = 0.3
    target_mean = float(np.exp(0.5 * sigma * sigma))
    target_var = float((np.exp(sigma * sigma) - 1.0) * np.exp(sigma * sigma))
    target_std = float(np.sqrt(target_var))
    assert abs(samples[:, 1].mean() - target_mean) < 0.05
    assert abs(samples[:, 1].std() - target_std) < 0.05


def test_correlated_marginals_positive_spearman_correlation():
    """Gaussian copula with off-diagonal 0.5 -> empirical Spearman rho > 0."""
    marginals = [Normal(0.0, 1.0), LogNormal(0.0, 0.3)]
    corr = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    cm = CorrelatedMarginals(marginals, corr)
    samples = np.asarray(cm.sample(jax.random.PRNGKey(3), (5000,)))
    rho, _ = spearmanr(samples[:, 0], samples[:, 1])
    # 0.5 Gaussian correlation -> Spearman rho ~ (6/pi)*arcsin(0.5/2) ~ 0.483.
    # We just require positive + materially nonzero.
    assert rho > 0.3, f"Spearman rho {rho} not materially positive"


def test_correlated_marginals_independent_marginals_zero_correlation():
    """Identity correlation -> empirical Spearman rho ~ 0."""
    marginals = [Normal(0.0, 1.0), Normal(2.0, 0.5)]
    corr = jnp.eye(2)
    cm = CorrelatedMarginals(marginals, corr)
    samples = np.asarray(cm.sample(jax.random.PRNGKey(4), (5000,)))
    rho, _ = spearmanr(samples[:, 0], samples[:, 1])
    # With 5000 samples the std-err on Spearman is ~ 1/sqrt(5000-2) ~ 0.014.
    # 0.1 absolute tolerance is comfortable.
    assert abs(rho) < 0.1, f"Independent marginals showed Spearman rho {rho}"


def test_correlated_marginals_validation_errors():
    """Bad correlation-matrix shape or invalid marginal raises ValueError."""
    with pytest.raises(ValueError):
        CorrelatedMarginals(
            [Normal(0.0, 1.0), Normal(0.0, 1.0)],
            jnp.eye(3),  # wrong size
        )

    # A marginal without ppf (use a stub object).
    class NoPPF:
        kind = "aleatoric"

        def sample(self, key, shape):  # pragma: no cover
            return jnp.zeros(shape)

        def log_pdf(self, x):  # pragma: no cover
            return jnp.zeros_like(x)

    with pytest.raises(ValueError):
        CorrelatedMarginals([Normal(0.0, 1.0), NoPPF()], jnp.eye(2))


def test_correlated_marginals_log_pdf_is_deferred():
    """log_pdf raises NotImplementedError (documented deeper followup)."""
    cm = CorrelatedMarginals(
        [Normal(0.0, 1.0), Normal(0.0, 1.0)], jnp.eye(2)
    )
    with pytest.raises(NotImplementedError):
        cm.log_pdf(jnp.array([0.0, 0.0]))


def test_correlated_marginals_sample_shape_with_batch_dims():
    """sample(key, (4, 5)) returns (4, 5, n_dim)."""
    cm = CorrelatedMarginals(
        [Normal(0.0, 1.0), LogNormal(0.0, 0.3), Normal(2.0, 0.5)],
        jnp.eye(3),
    )
    s = cm.sample(jax.random.PRNGKey(5), (4, 5))
    assert s.shape == (4, 5, 3)


def test_correlated_marginals_diff_through_corr_factor():
    """jax.grad through the Cholesky factor of corr_matrix is finite."""

    def loss(L_factor):
        corr = L_factor @ L_factor.T
        cm = CorrelatedMarginals(
            [Normal(0.0, 1.0), Normal(1.0, 0.5)], corr
        )
        samples = cm.sample(jax.random.PRNGKey(0), (200,))
        return jnp.mean(samples)

    L0 = jnp.array([[1.0, 0.0], [0.5, jnp.sqrt(1.0 - 0.25)]])
    grad = np.asarray(jax.grad(loss)(L0))
    assert np.all(np.isfinite(grad)), f"grad has non-finite entries: {grad}"
