# SPDX-License-Identifier: MIT
"""T-122-followup-poisson — UQ ``Exponential`` and ``Poisson`` distributions.

Companion to ``jaxonomy.library.RandomSource`` extensions: the UQ
sampling / Sobol / Morris pipelines also need :class:`Exponential` and
:class:`Poisson` distribution objects.  ``Poisson.ppf`` is deliberately
omitted (the inverse-CDF of a discrete distribution is a step
function, which breaks the smooth quantile-transform contract of the
unit-cube samplers); ``log_pmf`` is provided for likelihood-based
workflows (importance sampling, aleatoric/epistemic decomposition).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import Exponential, Poisson

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Exponential                                                                 #
# --------------------------------------------------------------------------- #


def test_exponential_sample_mean_matches_inverse_rate():
    """E[X] = 1 / rate for Exponential(rate).  1000 draws, rate=2.0."""
    key = jax.random.PRNGKey(0)
    samples = np.asarray(Exponential(rate=2.0).sample(key, (1000,)))
    assert (samples >= 0).all(), "Exponential samples must be non-negative."
    # CLT noise floor at n=1000, rate=2 -> std-err ~ 1/(2*sqrt(1000)) ~ 0.016
    # 0.1 absolute tolerance is comfortable.
    assert abs(float(np.mean(samples)) - 0.5) < 0.1, (
        f"Exponential(2.0) mean {np.mean(samples)} vs target 0.5"
    )


def test_exponential_log_pdf_matches_analytic():
    """log p(x) = log(rate) - rate * x for x >= 0; -inf for x < 0."""
    d = Exponential(rate=2.0)
    # At x=1.0: log(2) - 2 * 1 = log(2) - 2.
    expected = float(np.log(2.0) - 2.0)
    assert float(d.log_pdf(1.0)) == pytest.approx(expected, abs=1e-10)
    # At x=0.0: log(2) - 0 = log(2).
    assert float(d.log_pdf(0.0)) == pytest.approx(float(np.log(2.0)), abs=1e-10)
    # Out of support.
    assert float(d.log_pdf(-1.0)) == -np.inf


def test_exponential_ppf_inverse_cdf_roundtrip():
    """ppf(F(x)) = x for the Exponential CDF F(x) = 1 - exp(-rate*x)."""
    d = Exponential(rate=3.0)
    # Pick a fixed x; F(x) = 1 - exp(-3 * 0.25) = 1 - exp(-0.75).
    x = 0.25
    u = 1.0 - float(np.exp(-3.0 * x))
    np.testing.assert_allclose(float(d.ppf(u)), x, rtol=1e-7)


def test_exponential_rejects_nonpositive_rate():
    """rate must be > 0."""
    with pytest.raises(ValueError, match="rate"):
        Exponential(rate=0.0)
    with pytest.raises(ValueError, match="rate"):
        Exponential(rate=-1.0)


# --------------------------------------------------------------------------- #
# Poisson                                                                     #
# --------------------------------------------------------------------------- #


def test_poisson_sample_mean_matches_rate():
    """E[X] = rate for Poisson(rate).  1000 draws, rate=3.0."""
    key = jax.random.PRNGKey(0)
    samples = np.asarray(Poisson(rate=3.0).sample(key, (1000,)))
    # Integer-typed, non-negative.
    assert np.issubdtype(samples.dtype, np.integer) or np.all(
        np.equal(np.floor(samples), samples)
    ), f"Poisson samples must be integer-valued; got dtype {samples.dtype}"
    assert (samples >= 0).all(), "Poisson samples must be non-negative."
    # CLT std-err ~ sqrt(rate/n) ~ 0.055; 0.25 absolute tolerance is comfortable.
    assert abs(float(np.mean(samples)) - 3.0) < 0.25, (
        f"Poisson(3.0) mean {np.mean(samples)} vs target 3.0"
    )


def test_poisson_log_pmf_matches_analytic():
    """log P(k) = k*log(rate) - rate - log(k!).  Spot-check rate=3, k=5."""
    d = Poisson(rate=3.0)
    # Analytic: 5*log(3) - 3 - log(120).
    expected = 5.0 * float(np.log(3.0)) - 3.0 - float(np.log(120.0))
    assert float(d.log_pmf(5)) == pytest.approx(expected, abs=1e-6)
    # k=0: P(0) = exp(-rate) -> log P(0) = -rate.
    assert float(d.log_pmf(0)) == pytest.approx(-3.0, abs=1e-6)
    # Negative k -> -inf.
    assert float(d.log_pmf(-1)) == -np.inf


def test_poisson_log_pdf_aliases_log_pmf():
    """Poisson.log_pdf == log_pmf (alias so importance-sampling code works)."""
    d = Poisson(rate=2.5)
    for k in (0, 1, 3, 7):
        np.testing.assert_allclose(float(d.log_pdf(k)), float(d.log_pmf(k)))


def test_poisson_rejects_nonpositive_rate():
    """rate must be > 0."""
    with pytest.raises(ValueError, match="rate"):
        Poisson(rate=0.0)
    with pytest.raises(ValueError, match="rate"):
        Poisson(rate=-1.0)


def test_poisson_has_no_ppf():
    """Poisson deliberately omits ``ppf`` (step-function inverse CDF)."""
    d = Poisson(rate=3.0)
    assert not hasattr(d, "ppf"), (
        "Poisson should not expose ppf: a discrete inverse-CDF is a "
        "step function and breaks Saltelli/Morris pipelines."
    )
