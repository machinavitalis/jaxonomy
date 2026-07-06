# SPDX-License-Identifier: MIT
"""T-122-followup-mvn-cdf — multivariate-normal CDF via Genz QMC.

Rate-limit fallback: the agent shipped the Genz QMC implementation but
ran out of API budget before producing tests; the orchestrator wrote
these post-hoc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.uq import MultivariateNormal


class TestMultivariateNormalCDF:
    """Genz-QMC multivariate normal CDF — analytic spot checks."""

    def test_2d_independent_at_origin_quarter(self):
        # N(0, I) in 2-D: P(X1 <= 0 AND X2 <= 0) = 0.5 * 0.5 = 0.25
        mvn = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        cdf_val = float(mvn.cdf(jnp.array([0.0, 0.0])))
        np.testing.assert_allclose(cdf_val, 0.25, atol=0.01)

    def test_2d_independent_far_upper_one(self):
        mvn = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        cdf_val = float(mvn.cdf(jnp.array([10.0, 10.0])))
        np.testing.assert_allclose(cdf_val, 1.0, atol=1e-3)

    def test_2d_independent_far_lower_zero(self):
        mvn = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        cdf_val = float(mvn.cdf(jnp.array([-10.0, -10.0])))
        np.testing.assert_allclose(cdf_val, 0.0, atol=1e-3)

    def test_2d_correlated_at_origin(self):
        # ρ = 0.5: P(both <= 0) = 1/4 + arcsin(0.5)/(2π) = 0.25 + 1/12 ≈ 0.3333
        cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        mvn = MultivariateNormal(jnp.zeros(2), cov)
        cdf_val = float(mvn.cdf(jnp.array([0.0, 0.0])))
        np.testing.assert_allclose(cdf_val, 1 / 3, atol=0.01)

    def test_2d_correlated_high_corr_at_origin(self):
        # ρ = 0.9: closer to 0.5 (X1 and X2 nearly comonotone).
        cov = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        mvn = MultivariateNormal(jnp.zeros(2), cov)
        cdf_val = float(mvn.cdf(jnp.array([0.0, 0.0])))
        # Analytic: 0.25 + arcsin(0.9)/(2π) ≈ 0.4282
        np.testing.assert_allclose(cdf_val, 0.4282, atol=0.02)

    def test_3d_independent_at_origin_eighth(self):
        # N(0, I) in 3-D: P(all <= 0) = 0.5^3 = 0.125
        mvn = MultivariateNormal(jnp.zeros(3), jnp.eye(3))
        cdf_val = float(mvn.cdf(jnp.array([0.0, 0.0, 0.0])))
        np.testing.assert_allclose(cdf_val, 0.125, atol=0.015)

    def test_means_shifts_cdf(self):
        # If means = [1, 1], then cdf at [1, 1] should equal cdf at [0, 0]
        # under means = [0, 0] — both are at the joint mean.
        mvn_at_zero = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        mvn_at_one = MultivariateNormal(jnp.ones(2), jnp.eye(2))
        a = float(mvn_at_zero.cdf(jnp.array([0.0, 0.0])))
        b = float(mvn_at_one.cdf(jnp.array([1.0, 1.0])))
        np.testing.assert_allclose(a, b, atol=0.01)


class TestMultivariateNormalCDFDifferentiability:
    """The Genz QMC estimate is smooth in x and means."""

    def test_grad_through_x_finite(self):
        mvn = MultivariateNormal(jnp.zeros(2), jnp.eye(2))

        def fn(x):
            return mvn.cdf(x)

        g = jax.grad(fn)(jnp.array([0.5, 0.5]))
        # ∂P/∂x_i = φ(x_i) * Φ(x_j) for independent (here ~0.352 * 0.692 ≈ 0.244)
        assert jnp.all(jnp.isfinite(g))
        np.testing.assert_array_less(jnp.array(0.0), g)  # both partials positive


class TestMultivariateNormalCDFShape:
    def test_returns_scalar(self):
        mvn = MultivariateNormal(jnp.zeros(2), jnp.eye(2))
        val = mvn.cdf(jnp.array([0.0, 0.0]))
        assert val.shape == ()
