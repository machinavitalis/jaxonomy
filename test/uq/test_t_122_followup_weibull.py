# SPDX-License-Identifier: MIT
"""T-122-followup-weibull — UQ ``Weibull`` distribution.

Reliability / time-to-failure / wind-speed modelling distribution.
PDF ``f(x) = (k/lambda) * (x/lambda)**(k-1) * exp(-(x/lambda)**k)`` for
``x >= 0`` with shape ``k`` and scale ``lambda``.

Tests cover:
  * long-run mean ``scale * Gamma(1 + 1/shape)`` via sampling,
  * sample support (strictly positive),
  * ``log_pdf`` matches ``scipy.stats.weibull_min.logpdf`` within 1e-6,
  * closed-form ``ppf`` round-trip ``cdf(ppf(u)) ~ u``,
  * reproducibility (same key -> same sequence),
  * ``jax.grad`` finite w.r.t. *both* ``shape`` and ``scale``,
  * validation (non-positive parameters rejected).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import Weibull

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #


def test_weibull_sample_mean_matches_analytic():
    """E[X] = scale * Gamma(1 + 1/shape) for X ~ Weibull(shape, scale).

    Weibull(2, 1) -> mean = Gamma(1.5) ~ 0.8862269.  Long-run check
    with 20k samples.
    """
    shape_p, scale = 2.0, 1.0
    key = jax.random.PRNGKey(0)
    samples = np.asarray(Weibull(shape_p, scale).sample(key, (20_000,)))
    target = scale * math.gamma(1.0 + 1.0 / shape_p)
    # Var = scale^2 * (Gamma(1+2/k) - Gamma(1+1/k)^2)
    #     = 1.0 * (1.0 - 0.8862^2) ~ 0.2146 -> std-err ~ sqrt(0.2146/20000) ~ 0.00328.
    # 0.015 absolute tolerance gives ~4.5sigma comfort.
    assert abs(float(np.mean(samples)) - target) < 0.015, (
        f"Weibull({shape_p},{scale}) empirical mean {np.mean(samples)} "
        f"vs target {target}"
    )


def test_weibull_samples_positive():
    """All Weibull samples must lie in ``(0, inf)``."""
    key = jax.random.PRNGKey(1)
    samples = np.asarray(Weibull(2.0, 1.0).sample(key, (5_000,)))
    assert (samples > 0.0).all(), (
        f"Weibull samples leaked into x <= 0: min={samples.min()}"
    )


def test_weibull_log_pdf_matches_scipy():
    """log_pdf matches scipy.stats.weibull_min.logpdf within 1e-6."""
    shape_p, scale = 2.0, 1.0
    d = Weibull(shape_p, scale)
    xs = np.asarray([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
    ours = np.asarray(d.log_pdf(jnp.asarray(xs)))
    theirs = scipy.stats.weibull_min.logpdf(xs, shape_p, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_weibull_log_pdf_outside_support_is_neg_inf():
    """log_pdf at x <= 0 returns -inf."""
    d = Weibull(2.0, 1.0)
    for v in (-1.0, 0.0):
        assert float(d.log_pdf(v)) == -np.inf, (
            f"Weibull.log_pdf({v}) should be -inf; got {d.log_pdf(v)}"
        )


def test_weibull_ppf_matches_scipy():
    """ppf matches scipy.stats.weibull_min.ppf within 1e-6."""
    shape_p, scale = 2.0, 1.0
    d = Weibull(shape_p, scale)
    us = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    ours = np.asarray(d.ppf(jnp.asarray(us)))
    theirs = scipy.stats.weibull_min.ppf(us, shape_p, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_weibull_ppf_round_trip():
    """``cdf(ppf(u)) ~ u`` for uniforms in (0, 1).

    The closed-form inverse CDF should round-trip exactly through the
    scipy CDF up to floating-point roundoff.
    """
    shape_p, scale = 2.0, 1.0
    d = Weibull(shape_p, scale)
    us = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    xs = np.asarray(d.ppf(jnp.asarray(us)))
    us_round = scipy.stats.weibull_min.cdf(xs, shape_p, scale=scale)
    np.testing.assert_allclose(us_round, us, atol=1e-10, rtol=1e-10)


def test_weibull_same_key_is_reproducible():
    """Same PRNG key -> bit-identical Weibull samples."""
    key = jax.random.PRNGKey(7)
    d = Weibull(2.0, 1.0)
    a = np.asarray(d.sample(key, (100,)))
    b = np.asarray(d.sample(key, (100,)))
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_weibull_rejects_non_positive_shape():
    """shape <= 0 raises."""
    with pytest.raises(ValueError, match=r"shape .* must be positive"):
        Weibull(shape_param=0.0, scale=1.0)
    with pytest.raises(ValueError, match=r"shape .* must be positive"):
        Weibull(shape_param=-1.0, scale=1.0)


def test_weibull_rejects_non_positive_scale():
    """scale <= 0 raises."""
    with pytest.raises(ValueError, match=r"scale .* must be positive"):
        Weibull(shape_param=2.0, scale=0.0)
    with pytest.raises(ValueError, match=r"scale .* must be positive"):
        Weibull(shape_param=2.0, scale=-1.0)


# --------------------------------------------------------------------------- #
# Special cases                                                               #
# --------------------------------------------------------------------------- #


def test_weibull_exponential_special_case():
    """Weibull(1, scale) is Exponential(rate = 1/scale).

    Mean should be ``scale``; samples must be positive.
    """
    scale = 2.0
    d = Weibull(1.0, scale)
    samples = np.asarray(d.sample(jax.random.PRNGKey(0), (20_000,)))
    # Exponential variance scale^2 -> std-err = scale/sqrt(n) ~ 0.014.
    assert abs(samples.mean() - scale) < 0.1
    assert (samples > 0.0).all()


# --------------------------------------------------------------------------- #
# Differentiability                                                           #
# --------------------------------------------------------------------------- #


def test_weibull_shape_grad_finite():
    """jax.grad through ``shape`` is finite and non-trivial.

    The shape parameter enters via the closed-form inverse-CDF reparam
    ``scale * (-log(1-u))**(1/shape)`` which is smooth in ``shape`` --
    no implicit-reparam machinery, so the gradient is analytically clean.
    """
    key = jax.random.PRNGKey(0)
    u = jax.random.uniform(key, shape=(64,))

    def loss(shape_traced):
        d = Weibull(shape_param=shape_traced, scale=1.0)
        # Build the same closed-form transform jax.grad-style.
        return jnp.mean(d.ppf(u))

    g = jax.grad(loss)(jnp.asarray(2.0))
    assert jnp.isfinite(g), f"expected finite gradient through shape; got {g}"
    assert float(jnp.abs(g)) > 1e-3, (
        f"gradient through shape should be non-trivial; got {g}"
    )


def test_weibull_scale_grad_finite():
    """jax.grad through ``scale`` is finite and non-trivial.

    ``scale`` enters as a trivial multiplicative reparam; gradient is
    just the inverse-CDF transform of ``u``.
    """
    key = jax.random.PRNGKey(0)
    u = jax.random.uniform(key, shape=(64,))

    def loss(scale_traced):
        d = Weibull(shape_param=2.0, scale=scale_traced)
        return jnp.mean(d.ppf(u))

    g = jax.grad(loss)(jnp.asarray(1.5))
    assert jnp.isfinite(g), f"expected finite gradient through scale; got {g}"
    assert float(jnp.abs(g)) > 1e-3, (
        f"gradient through scale should be non-trivial; got {g}"
    )


def test_weibull_log_pdf_grad_finite_through_both():
    """``jax.grad`` of ``log_pdf`` is finite through *both* parameters.

    The analytic log-pdf is smooth in ``shape`` and ``scale`` away from
    the boundary; verify the gradient is finite at a typical interior
    point.
    """
    x = jnp.asarray(1.5)

    def loss(params):
        shape_p, scale = params
        d = Weibull(shape_param=shape_p, scale=scale)
        return d.log_pdf(x)

    g_shape, g_scale = jax.grad(loss)((jnp.asarray(2.0), jnp.asarray(1.0)))
    assert jnp.isfinite(g_shape), f"expected finite shape grad; got {g_shape}"
    assert jnp.isfinite(g_scale), f"expected finite scale grad; got {g_scale}"
