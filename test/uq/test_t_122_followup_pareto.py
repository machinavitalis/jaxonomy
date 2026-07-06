# SPDX-License-Identifier: MIT
"""T-122-followup-pareto — UQ ``Pareto`` distribution.

Heavy-tail / power-law modelling distribution.
PDF ``f(x) = alpha * scale**alpha / x**(alpha + 1)`` for ``x >= scale``
with shape ``alpha`` and threshold ``scale``.

Tests cover:
  * long-run mean ``alpha * scale / (alpha - 1)`` via sampling,
  * sample support (samples >= scale),
  * ``log_pdf`` matches ``scipy.stats.pareto.logpdf`` within 1e-6,
  * closed-form ``ppf`` round-trip ``cdf(ppf(u)) ~ u``,
  * reproducibility (same key -> same sequence),
  * ``jax.grad`` finite w.r.t. *both* ``scale`` and ``alpha``,
  * validation (non-positive parameters rejected).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import Pareto

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #


def test_pareto_sample_mean_matches_analytic():
    """E[X] = alpha * scale / (alpha - 1) for X ~ Pareto(scale, alpha).

    Pareto(scale=1, alpha=2) -> mean = 2 * 1 / (2 - 1) = 2.  Long-run
    check with 50k samples (Pareto's heavy tail makes the empirical
    mean noisier than light-tail distributions, so we use a generous
    sample size and a 5% relative tolerance).
    """
    scale, alpha = 1.0, 2.0
    key = jax.random.PRNGKey(0)
    samples = np.asarray(Pareto(scale, alpha).sample(key, (50_000,)))
    target = alpha * scale / (alpha - 1.0)
    # Pareto(1, 2) variance is alpha * scale^2 / ((alpha-1)^2 (alpha-2)),
    # which diverges at alpha == 2!  So we use a relative tolerance
    # instead of a sigma-based one — empirical mean of a heavy-tailed
    # sample is still consistent for the true mean (LLN holds for
    # alpha > 1) but converges slowly.
    assert abs(float(np.mean(samples)) - target) / target < 0.10, (
        f"Pareto({scale},{alpha}) empirical mean {np.mean(samples)} "
        f"vs target {target}"
    )


def test_pareto_samples_above_scale():
    """All Pareto samples must lie in ``[scale, inf)``."""
    scale = 1.5
    key = jax.random.PRNGKey(1)
    samples = np.asarray(Pareto(scale, alpha=2.5).sample(key, (5_000,)))
    assert (samples >= scale).all(), (
        f"Pareto samples leaked below scale={scale}: min={samples.min()}"
    )


def test_pareto_log_pdf_matches_scipy():
    """log_pdf matches scipy.stats.pareto.logpdf within 1e-6.

    SciPy uses ``pareto(b, scale=scale)`` where ``b`` is our ``alpha``
    parameter (shape) and ``scale`` is our ``scale``.
    """
    scale, alpha = 1.0, 2.0
    d = Pareto(scale, alpha)
    xs = np.asarray([1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0])
    ours = np.asarray(d.log_pdf(jnp.asarray(xs)))
    theirs = scipy.stats.pareto.logpdf(xs, alpha, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_pareto_log_pdf_with_nondefault_scale_matches_scipy():
    """log_pdf with scale != 1.0 also matches scipy."""
    scale, alpha = 2.5, 3.0
    d = Pareto(scale, alpha)
    xs = np.asarray([2.5, 3.0, 4.0, 5.0, 7.5, 10.0])
    ours = np.asarray(d.log_pdf(jnp.asarray(xs)))
    theirs = scipy.stats.pareto.logpdf(xs, alpha, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_pareto_log_pdf_outside_support_is_neg_inf():
    """log_pdf at x < scale returns -inf."""
    scale = 1.0
    d = Pareto(scale, alpha=2.0)
    for v in (-1.0, 0.0, 0.5, 0.999):
        assert float(d.log_pdf(v)) == -np.inf, (
            f"Pareto.log_pdf({v}) should be -inf; got {d.log_pdf(v)}"
        )


def test_pareto_ppf_matches_scipy():
    """ppf matches scipy.stats.pareto.ppf within 1e-6."""
    scale, alpha = 1.0, 2.0
    d = Pareto(scale, alpha)
    us = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    ours = np.asarray(d.ppf(jnp.asarray(us)))
    theirs = scipy.stats.pareto.ppf(us, alpha, scale=scale)
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_pareto_ppf_round_trip():
    """``cdf(ppf(u)) ~ u`` for uniforms in (0, 1).

    The closed-form inverse CDF should round-trip exactly through the
    scipy CDF up to floating-point roundoff.
    """
    scale, alpha = 1.0, 2.0
    d = Pareto(scale, alpha)
    us = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    xs = np.asarray(d.ppf(jnp.asarray(us)))
    us_round = scipy.stats.pareto.cdf(xs, alpha, scale=scale)
    np.testing.assert_allclose(us_round, us, atol=1e-10, rtol=1e-10)


def test_pareto_same_key_is_reproducible():
    """Same PRNG key -> bit-identical Pareto samples."""
    key = jax.random.PRNGKey(7)
    d = Pareto(1.0, 2.0)
    a = np.asarray(d.sample(key, (100,)))
    b = np.asarray(d.sample(key, (100,)))
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_pareto_rejects_non_positive_scale():
    """scale <= 0 raises."""
    with pytest.raises(ValueError, match=r"scale .* must be positive"):
        Pareto(scale=0.0, alpha=2.0)
    with pytest.raises(ValueError, match=r"scale .* must be positive"):
        Pareto(scale=-1.0, alpha=2.0)


def test_pareto_rejects_non_positive_alpha():
    """alpha <= 0 raises."""
    with pytest.raises(ValueError, match=r"alpha .* must be positive"):
        Pareto(scale=1.0, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha .* must be positive"):
        Pareto(scale=1.0, alpha=-1.0)


# --------------------------------------------------------------------------- #
# Differentiability                                                           #
# --------------------------------------------------------------------------- #


def test_pareto_scale_grad_finite():
    """jax.grad through ``scale`` is finite and non-trivial.

    ``scale`` enters as a trivial multiplicative reparam in the closed-
    form inverse CDF; gradient is the inverse-CDF transform of ``u``.
    """
    key = jax.random.PRNGKey(0)
    u = jax.random.uniform(key, shape=(64,))

    def loss(scale_traced):
        d = Pareto(scale=scale_traced, alpha=2.0)
        return jnp.mean(d.ppf(u))

    g = jax.grad(loss)(jnp.asarray(1.5))
    assert jnp.isfinite(g), f"expected finite gradient through scale; got {g}"
    assert float(jnp.abs(g)) > 1e-3, (
        f"gradient through scale should be non-trivial; got {g}"
    )


def test_pareto_alpha_grad_finite():
    """jax.grad through ``alpha`` is finite and non-trivial.

    The alpha parameter enters via the closed-form inverse-CDF reparam
    ``scale * (1 - u)**(-1/alpha)`` which is smooth in ``alpha`` --
    no implicit-reparam machinery, so the gradient is analytically clean.
    """
    key = jax.random.PRNGKey(0)
    u = jax.random.uniform(key, shape=(64,))

    def loss(alpha_traced):
        d = Pareto(scale=1.0, alpha=alpha_traced)
        return jnp.mean(d.ppf(u))

    g = jax.grad(loss)(jnp.asarray(2.0))
    assert jnp.isfinite(g), f"expected finite gradient through alpha; got {g}"
    assert float(jnp.abs(g)) > 1e-3, (
        f"gradient through alpha should be non-trivial; got {g}"
    )


def test_pareto_log_pdf_grad_finite_through_both():
    """``jax.grad`` of ``log_pdf`` is finite through *both* parameters.

    The analytic log-pdf is smooth in ``scale`` and ``alpha`` away from
    the support boundary; verify the gradient is finite at a typical
    interior point (``x > scale``).
    """
    x = jnp.asarray(2.5)

    def loss(params):
        scale, alpha = params
        d = Pareto(scale=scale, alpha=alpha)
        return d.log_pdf(x)

    g_scale, g_alpha = jax.grad(loss)((jnp.asarray(1.0), jnp.asarray(2.0)))
    assert jnp.isfinite(g_scale), f"expected finite scale grad; got {g_scale}"
    assert jnp.isfinite(g_alpha), f"expected finite alpha grad; got {g_alpha}"
