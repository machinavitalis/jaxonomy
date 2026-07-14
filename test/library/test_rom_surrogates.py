# SPDX-License-Identifier: MIT

"""Tests for the statistical surrogate models (T-148..T-150).

Gaussian process (kriging), polynomial chaos expansion (PCE), and radial basis
function (RBF) surrogates, plus their feedthrough ``LeafSystem`` blocks.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxonomy.library.rom.surrogates import (
    fit_gp,
    GPModel,
    GaussianProcess,
    fit_pce,
    PCEModel,
    PolynomialChaos,
    fit_rbf,
    RBFModel,
    RadialBasisSurrogate,
)

pytestmark = pytest.mark.minimal


# ===========================================================================
# Gaussian process / kriging
# ===========================================================================
class TestGaussianProcess:
    def test_interpolates_training_points(self):
        X = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
        y = np.sin(3.0 * X.ravel())
        gp = fit_gp(X, y, kernel="rbf", length_scale=0.25, noise=1e-12)
        assert isinstance(gp, GPModel)

        mean, var = gp.predict(X)
        # Mean interpolates the data and variance collapses at training points
        # (noise -> 0): Rasmussen & Williams 2006, Sec. 2.2.
        assert np.allclose(np.array(mean), y, atol=1e-5)
        assert np.max(np.array(var)) < 1e-5

    def test_variance_grows_away_from_data(self):
        X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y = np.cos(2.0 * X.ravel())
        gp = fit_gp(X, y, kernel="rbf", length_scale=0.2, signal_var=1.0,
                    noise=1e-10)
        _, var_near = gp.predict(np.array([[0.5]]))
        _, var_far = gp.predict(np.array([[10.0]]))
        assert float(var_far[0]) > float(var_near[0])
        # Far from all data the posterior variance returns to the prior.
        assert float(var_far[0]) == pytest.approx(gp.signal_var, rel=1e-3)

    def test_matern_interpolates(self):
        X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = X.ravel() ** 2
        gp = fit_gp(X, y, kernel="matern", length_scale=0.4, noise=1e-12)
        mean, _ = gp.predict(X)
        assert np.allclose(np.array(mean), y, atol=1e-4)

    def test_optimize_improves_marginal_likelihood(self):
        rng = np.random.RandomState(0)
        X = np.sort(rng.uniform(0, 1, 12)).reshape(-1, 1)
        y = np.sin(6.0 * X.ravel())
        gp0 = fit_gp(X, y, length_scale=5.0, signal_var=0.1, noise=1e-3,
                     optimize=False)
        gp1 = fit_gp(X, y, length_scale=5.0, signal_var=0.1, noise=1e-3,
                     optimize=True, n_steps=300, lr=0.05)
        assert float(gp1.log_marginal_likelihood()) > float(
            gp0.log_marginal_likelihood())

    def test_predict_is_jax_traceable(self):
        X = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
        y = np.sin(3.0 * X.ravel())
        gp = fit_gp(X, y, length_scale=0.3, noise=1e-10)
        f = jax.jit(lambda z: gp.predict(z)[0])
        out = f(jnp.array([[0.42]]))
        ref, _ = gp.predict(np.array([[0.42]]))
        assert np.allclose(np.array(out), np.array(ref))

    def test_cross_check_against_sklearn(self):
        skgp = pytest.importorskip("sklearn.gaussian_process")
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        rng = np.random.RandomState(1)
        X = np.sort(rng.uniform(-2, 2, 10)).reshape(-1, 1)
        y = np.sin(X.ravel())

        ls, sv, noise = 0.7, 1.3, 1e-6
        gp = fit_gp(X, y, kernel="rbf", length_scale=ls, signal_var=sv,
                    noise=noise)

        kernel = ConstantKernel(sv, "fixed") * RBF(ls, "fixed")
        sk = skgp.GaussianProcessRegressor(kernel=kernel, alpha=noise,
                                           optimizer=None)
        sk.fit(X, y)

        Xtest = np.linspace(-2, 2, 25).reshape(-1, 1)
        mean, _ = gp.predict(Xtest)
        sk_mean = sk.predict(Xtest)
        assert np.allclose(np.array(mean), sk_mean, atol=1e-4)

    def test_block_matches_model(self):
        rng = np.random.RandomState(2)
        X = rng.uniform(0, 1, (12, 2))
        y = np.sin(X[:, 0]) + X[:, 1] ** 2
        gp = fit_gp(X, y, length_scale=0.5, noise=1e-8)

        block = GaussianProcess(gp)
        u = jnp.array([0.3, 0.6])
        block.input_ports[0].fix_value(u)
        ctx = block.create_context()

        mean = block.output_ports[0].eval(ctx)
        var = block.output_ports[1].eval(ctx)
        ref_mean, ref_var = gp.predict(np.array([[0.3, 0.6]]))
        assert float(mean) == pytest.approx(float(ref_mean[0]), rel=1e-6,
                                            abs=1e-8)
        assert float(var) == pytest.approx(float(ref_var[0]), rel=1e-6,
                                           abs=1e-8)


# ===========================================================================
# Polynomial chaos expansion
# ===========================================================================
class TestPolynomialChaos:
    def test_recovers_low_degree_polynomial_exactly(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(-1, 1, (60, 2))
        # Quadratic cross-term function, exactly representable at order 2.
        y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1]
        pce = fit_pce(X, y, [("uniform", -1, 1), ("uniform", -1, 1)], order=2)
        assert isinstance(pce, PCEModel)
        assert np.allclose(np.array(pce.predict(X)), y, atol=1e-8)

    def test_mean_variance_match_monte_carlo(self):
        # a*x1 + b*x2 with independent normals.
        a, b = 2.0, -1.5
        mu = [0.5, -1.0]
        sigma = [0.8, 1.2]
        rng = np.random.RandomState(3)
        Xfit = np.column_stack([
            rng.normal(mu[0], sigma[0], 200),
            rng.normal(mu[1], sigma[1], 200),
        ])
        yfit = a * Xfit[:, 0] + b * Xfit[:, 1]
        dists = [("normal", mu[0], sigma[0]), ("normal", mu[1], sigma[1])]
        pce = fit_pce(Xfit, yfit, dists, order=2)

        # Monte-Carlo reference over the input distributions.
        Xmc = np.column_stack([
            rng.normal(mu[0], sigma[0], 200000),
            rng.normal(mu[1], sigma[1], 200000),
        ])
        ymc = a * Xmc[:, 0] + b * Xmc[:, 1]
        assert float(pce.mean()) == pytest.approx(ymc.mean(), abs=2e-2)
        assert float(pce.variance()) == pytest.approx(ymc.var(), rel=2e-2)

    def test_sobol_indices_additive_function(self):
        # For y = a*x1 + b*x2, main effects equal total effects and sum to 1.
        a, b = 2.0, 3.0
        rng = np.random.RandomState(4)
        X = rng.uniform(-1, 1, (80, 2))
        y = a * X[:, 0] + b * X[:, 1]
        pce = fit_pce(X, y, [("uniform", -1, 1), ("uniform", -1, 1)], order=1)

        sob = pce.sobol_indices()
        first = np.array(sob["first_order"])
        total = np.array(sob["total"])
        # Variance contributions: a^2 Var(x) and b^2 Var(x) -> a^2 : b^2.
        expected = np.array([a ** 2, b ** 2]) / (a ** 2 + b ** 2)
        assert np.allclose(first, expected, atol=1e-6)
        assert np.allclose(total, expected, atol=1e-6)
        assert np.allclose(first, total, atol=1e-8)  # additive -> no interaction
        assert first.sum() == pytest.approx(1.0, abs=1e-6)

    def test_ishigami_sobol(self):
        # Ishigami function: x_i ~ Uniform(-pi, pi). Known analytic Sobol.
        a, b = 7.0, 0.1
        rng = np.random.RandomState(5)
        n = 4000
        X = rng.uniform(-np.pi, np.pi, (n, 3))
        y = (np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2
             + b * X[:, 2] ** 4 * np.sin(X[:, 0]))
        dists = [("uniform", -np.pi, np.pi)] * 3
        pce = fit_pce(X, y, dists, order=8)

        sob = pce.sobol_indices()
        first = np.array(sob["first_order"])
        total = np.array(sob["total"])

        # Analytic Ishigami indices (Marrel et al. 2009).
        Vtot = a ** 2 / 8 + b * np.pi ** 4 / 5 + b ** 2 * np.pi ** 8 / 18 + 0.5
        V1 = 0.5 + b * np.pi ** 4 / 5 + b ** 2 * np.pi ** 8 / 50
        V2 = a ** 2 / 8
        S1 = np.array([V1, V2, 0.0]) / Vtot
        ST3 = (b ** 2 * np.pi ** 8 / 18 - b ** 2 * np.pi ** 8 / 50) / Vtot
        assert np.allclose(first, S1, atol=5e-2)
        # x2 is purely additive -> no interaction; x3 only interacts.
        assert total[1] == pytest.approx(first[1], abs=5e-2)
        assert float(total[2]) == pytest.approx(ST3, abs=5e-2)
        assert float(first[2]) == pytest.approx(0.0, abs=2e-2)

    def test_block_matches_model(self):
        rng = np.random.RandomState(6)
        X = rng.uniform(-1, 1, (50, 2))
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + X[:, 0] * X[:, 1]
        pce = fit_pce(X, y, [("uniform", -1, 1), ("uniform", -1, 1)], order=2)

        block = PolynomialChaos(pce)
        block.input_ports[0].fix_value(jnp.array([0.2, 0.3]))
        ctx = block.create_context()
        out = block.output_ports[0].eval(ctx)
        ref = pce.predict(np.array([[0.2, 0.3]]))
        assert float(out) == pytest.approx(float(ref[0]), rel=1e-8, abs=1e-8)


# ===========================================================================
# Radial basis function
# ===========================================================================
class TestRadialBasis:
    @pytest.mark.parametrize(
        "kernel",
        ["multiquadric", "inverse_multiquadric", "gaussian",
         "thin_plate_spline"],
    )
    def test_interpolates_training_points(self, kernel):
        X = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
        y = np.cos(4.0 * X.ravel())
        rbf = fit_rbf(X, y, kernel=kernel, epsilon=2.0, smoothing=0.0)
        assert isinstance(rbf, RBFModel)
        pred = rbf.predict(X)
        assert np.allclose(np.array(pred), y, atol=1e-6)

    def test_smooth_between_points(self):
        X = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
        y = np.sin(2.0 * np.pi * X.ravel())
        rbf = fit_rbf(X, y, kernel="gaussian", epsilon=4.0)
        Xtest = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
        pred = np.array(rbf.predict(Xtest))
        # Bounded and continuous: no wild oscillation between the samples.
        assert np.all(np.abs(pred) < 2.0)
        assert np.max(np.abs(np.diff(pred))) < 0.2

    def test_polynomial_tail_interpolates(self):
        rng = np.random.RandomState(7)
        X = rng.uniform(0, 1, (15, 2))
        y = 1.0 + X[:, 0] - 2.0 * X[:, 1] + np.sin(3.0 * X[:, 0])
        rbf = fit_rbf(X, y, kernel="multiquadric", epsilon=1.5, poly_degree=1)
        assert np.allclose(np.array(rbf.predict(X)), y, atol=1e-6)

    def test_predict_is_jax_traceable(self):
        X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = np.cos(4.0 * X.ravel())
        rbf = fit_rbf(X, y, kernel="gaussian", epsilon=3.0)
        f = jax.jit(rbf.predict)
        out = f(jnp.array([[0.31]]))
        assert np.allclose(np.array(out), np.array(rbf.predict(np.array([[0.31]]))))

    def test_block_matches_model(self):
        X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = np.cos(4.0 * X.ravel())
        rbf = fit_rbf(X, y, kernel="gaussian", epsilon=3.0, poly_degree=1)

        block = RadialBasisSurrogate(rbf)
        block.input_ports[0].fix_value(jnp.array([0.55]))
        ctx = block.create_context()
        out = block.output_ports[0].eval(ctx)
        ref = rbf.predict(np.array([[0.55]]))
        assert float(out) == pytest.approx(float(ref[0]), rel=1e-8, abs=1e-8)
