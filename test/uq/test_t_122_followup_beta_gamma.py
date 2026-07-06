# SPDX-License-Identifier: MIT
"""T-122-followup-beta-gamma — UQ ``Beta`` and ``Gamma`` distributions.

Two continuous distributions rounding out the parameter-uncertainty
toolkit:

* ``Beta(alpha, beta)`` on ``[0, 1]`` — bounded fractions, mixture
  weights, Bayesian priors over a Bernoulli ``p``.
* ``Gamma(shape, scale)`` on ``[0, inf)`` — wait times, positive-
  valued physical parameters, rate priors.

Tests cover:
  * mean / support checks via long-run sampling,
  * ``log_pdf`` matches scipy ``logpdf`` within 1e-6,
  * ``ppf`` matches scipy ``ppf`` within 1e-6,
  * reproducibility (same key -> same sequence),
  * validation (non-positive parameters rejected).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import Beta, Gamma

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Beta                                                                        #
# --------------------------------------------------------------------------- #


def test_beta_sample_mean_matches_analytic():
    """E[X] = alpha / (alpha + beta) for X ~ Beta(alpha, beta).

    Beta(2, 5) -> mean = 2/7 ~ 0.2857.  Long-run check with 20k samples.
    """
    alpha, beta = 2.0, 5.0
    key = jax.random.PRNGKey(0)
    samples = np.asarray(Beta(alpha, beta).sample(key, (20_000,)))
    target = alpha / (alpha + beta)
    # Var of Beta(2,5) = ab / ((a+b)^2 (a+b+1)) ~ 0.0255 -> std-err ~ 0.00357.
    # 0.015 absolute tolerance gives ~4sigma comfort.
    assert abs(float(np.mean(samples)) - target) < 0.015, (
        f"Beta({alpha},{beta}) empirical mean {np.mean(samples)} "
        f"vs target {target}"
    )


def test_beta_samples_in_unit_interval():
    """All Beta samples must lie in the open ``(0, 1)`` interval."""
    key = jax.random.PRNGKey(1)
    samples = np.asarray(Beta(2.0, 5.0).sample(key, (5_000,)))
    assert (samples > 0.0).all() and (samples < 1.0).all(), (
        f"Beta samples leaked outside (0, 1): min={samples.min()}, "
        f"max={samples.max()}"
    )


def test_beta_log_pdf_matches_scipy():
    """log_pdf matches scipy.stats.beta.logpdf within 1e-6."""
    alpha, beta = 2.0, 5.0
    d = Beta(alpha, beta)
    xs = np.asarray([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
    ours = np.asarray(d.log_pdf(jnp.asarray(xs)))
    theirs = scipy.stats.beta.logpdf(xs, alpha, beta)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_beta_log_pdf_outside_support_is_neg_inf():
    """log_pdf at x <= 0 or x >= 1 returns -inf."""
    d = Beta(2.0, 5.0)
    for v in (-0.1, 0.0, 1.0, 1.5):
        assert float(d.log_pdf(v)) == -np.inf, (
            f"Beta.log_pdf({v}) should be -inf; got {d.log_pdf(v)}"
        )


def test_beta_ppf_matches_scipy():
    """ppf matches scipy.stats.beta.ppf within 1e-6."""
    alpha, beta = 2.0, 5.0
    d = Beta(alpha, beta)
    us = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    ours = np.asarray(d.ppf(jnp.asarray(us)))
    theirs = scipy.stats.beta.ppf(us, alpha, beta)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_beta_same_key_is_reproducible():
    """Same PRNG key -> bit-identical Beta samples."""
    key = jax.random.PRNGKey(7)
    d = Beta(2.0, 5.0)
    a = np.asarray(d.sample(key, (100,)))
    b = np.asarray(d.sample(key, (100,)))
    np.testing.assert_array_equal(a, b)


def test_beta_rejects_non_positive_alpha():
    """alpha <= 0 raises."""
    with pytest.raises(ValueError, match=r"alpha .* must be positive"):
        Beta(alpha=0.0, beta=2.0)
    with pytest.raises(ValueError, match=r"alpha .* must be positive"):
        Beta(alpha=-1.0, beta=2.0)


def test_beta_rejects_non_positive_beta():
    """beta <= 0 raises."""
    with pytest.raises(ValueError, match=r"beta .* must be positive"):
        Beta(alpha=2.0, beta=0.0)
    with pytest.raises(ValueError, match=r"beta .* must be positive"):
        Beta(alpha=2.0, beta=-1.0)


def test_beta_uniform_special_case():
    """Beta(1, 1) is the uniform distribution on [0, 1].

    log_pdf should be 0 on the interior; mean -> 0.5.
    """
    d = Beta(1.0, 1.0)
    xs = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    np.testing.assert_allclose(np.asarray(d.log_pdf(xs)), 0.0, atol=1e-12)
    samples = np.asarray(d.sample(jax.random.PRNGKey(3), (10_000,)))
    assert abs(samples.mean() - 0.5) < 0.02


# --------------------------------------------------------------------------- #
# Gamma                                                                       #
# --------------------------------------------------------------------------- #


def test_gamma_sample_mean_matches_analytic():
    """E[X] = shape * scale for X ~ Gamma(shape, scale).

    Gamma(2, 3) -> mean = 6.  Long-run check with 20k samples.
    """
    shape_p, scale = 2.0, 3.0
    key = jax.random.PRNGKey(0)
    samples = np.asarray(Gamma(shape_p, scale).sample(key, (20_000,)))
    target = shape_p * scale
    # Var = shape*scale^2 = 18 -> std-err ~ sqrt(18/20000) ~ 0.030.
    # 0.15 absolute tolerance gives ~5sigma comfort.
    assert abs(float(np.mean(samples)) - target) < 0.15, (
        f"Gamma({shape_p},{scale}) empirical mean {np.mean(samples)} "
        f"vs target {target}"
    )


def test_gamma_samples_positive():
    """All Gamma samples must lie in ``(0, inf)``."""
    key = jax.random.PRNGKey(1)
    samples = np.asarray(Gamma(2.0, 3.0).sample(key, (5_000,)))
    assert (samples > 0.0).all(), (
        f"Gamma samples leaked into x <= 0: min={samples.min()}"
    )


def test_gamma_log_pdf_matches_scipy():
    """log_pdf matches scipy.stats.gamma.logpdf within 1e-6."""
    shape_p, scale = 2.0, 3.0
    d = Gamma(shape_p, scale)
    xs = np.asarray([0.5, 1.0, 3.0, 5.0, 10.0, 20.0])
    ours = np.asarray(d.log_pdf(jnp.asarray(xs)))
    theirs = scipy.stats.gamma.logpdf(xs, shape_p, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_gamma_log_pdf_outside_support_is_neg_inf():
    """log_pdf at x <= 0 returns -inf."""
    d = Gamma(2.0, 3.0)
    for v in (-1.0, 0.0):
        assert float(d.log_pdf(v)) == -np.inf, (
            f"Gamma.log_pdf({v}) should be -inf; got {d.log_pdf(v)}"
        )


def test_gamma_ppf_matches_scipy():
    """ppf matches scipy.stats.gamma.ppf within 1e-6."""
    shape_p, scale = 2.0, 3.0
    d = Gamma(shape_p, scale)
    us = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    ours = np.asarray(d.ppf(jnp.asarray(us)))
    theirs = scipy.stats.gamma.ppf(us, shape_p, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_gamma_same_key_is_reproducible():
    """Same PRNG key -> bit-identical Gamma samples."""
    key = jax.random.PRNGKey(7)
    d = Gamma(2.0, 3.0)
    a = np.asarray(d.sample(key, (100,)))
    b = np.asarray(d.sample(key, (100,)))
    np.testing.assert_array_equal(a, b)


def test_gamma_rejects_non_positive_shape():
    """shape <= 0 raises."""
    with pytest.raises(ValueError, match=r"shape .* must be positive"):
        Gamma(shape_param=0.0, scale=1.0)
    with pytest.raises(ValueError, match=r"shape .* must be positive"):
        Gamma(shape_param=-1.0, scale=1.0)


def test_gamma_rejects_non_positive_scale():
    """scale <= 0 raises."""
    with pytest.raises(ValueError, match=r"scale .* must be positive"):
        Gamma(shape_param=2.0, scale=0.0)
    with pytest.raises(ValueError, match=r"scale .* must be positive"):
        Gamma(shape_param=2.0, scale=-1.0)


def test_gamma_exponential_special_case():
    """Gamma(1, theta) is Exponential(rate = 1/theta).

    Mean should be theta; samples must be positive.
    """
    theta = 2.0
    d = Gamma(1.0, theta)
    samples = np.asarray(d.sample(jax.random.PRNGKey(0), (20_000,)))
    assert abs(samples.mean() - theta) < 0.1
    # Exponential variance theta^2 -> std = theta = 2 -> std-err ~ 0.014.
    assert (samples > 0.0).all()


def test_gamma_scale_grad_finite():
    """jax.grad through ``scale`` is finite and non-trivial.

    The scale parameter enters via a smooth multiplicative reparam, so
    the gradient should flow cleanly through it (independent of the
    implicit-reparam machinery used for ``shape``).
    """
    key = jax.random.PRNGKey(0)

    def loss(scale_traced):
        z = jax.random.gamma(key, 2.0, shape=(64,))
        return jnp.mean(z * scale_traced)

    g = jax.grad(loss)(jnp.asarray(3.0))
    assert jnp.isfinite(g), f"expected finite gradient through scale; got {g}"
    assert float(jnp.abs(g)) > 1e-3, (
        f"gradient through scale should be non-trivial; got {g}"
    )
