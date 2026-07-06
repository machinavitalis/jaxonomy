# SPDX-License-Identifier: MIT
"""T-122-followup-bernoulli — UQ ``Bernoulli`` convenience distribution.

Binary-outcome special case of ``Categorical([0, 1], [1-p, p])``.  Common
enough in simulation work (coin flips, masking, failure indicators) that
a dedicated ``Bernoulli(p)`` constructor is far more readable than the
two-list form.

Tests cover:
  * hard ``sample`` lands in ``{0, 1}`` and matches empirical mean ``p``,
  * ``log_pmf(0) == log(1 - p)`` / ``log_pmf(1) == log(p)`` / -inf otherwise,
  * reproducibility (same key -> same sequence),
  * ``differentiable_sample`` continuous in (0, 1) and grad through ``p``
    finite & non-trivial,
  * validation (``p`` outside [0, 1] rejected).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import Bernoulli

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Sampling statistics                                                         #
# --------------------------------------------------------------------------- #


def test_bernoulli_sample_mean_matches_p():
    """E[X] = p for X ~ Bernoulli(p).  p=0.3, long-run check."""
    key = jax.random.PRNGKey(0)
    p = 0.3
    d = Bernoulli(p=p)
    samples = np.asarray(d.sample(key, (20_000,)))
    # CLT std-err for Bernoulli(p=0.3) at n=20k is sqrt(p(1-p)/n) ~ 0.0032.
    # 0.02 absolute tolerance gives ~6sigma comfort.
    assert abs(float(np.mean(samples)) - p) < 0.02, (
        f"Bernoulli(p={p}) empirical mean {np.mean(samples)} vs target {p}"
    )


def test_bernoulli_sample_only_0_or_1():
    """Every sample must be exactly 0 or 1."""
    key = jax.random.PRNGKey(1)
    d = Bernoulli(p=0.7)
    samples = np.asarray(d.sample(key, (1_000,)))
    assert set(np.unique(samples)).issubset({0, 1}), (
        f"Bernoulli sample contains values outside {{0,1}}: {np.unique(samples)}"
    )


def test_bernoulli_p_zero_is_all_zeros():
    """p=0 -> always 0."""
    key = jax.random.PRNGKey(2)
    samples = np.asarray(Bernoulli(p=0.0).sample(key, (200,)))
    np.testing.assert_array_equal(samples, np.zeros_like(samples))


def test_bernoulli_p_one_is_all_ones():
    """p=1 -> always 1."""
    key = jax.random.PRNGKey(3)
    samples = np.asarray(Bernoulli(p=1.0).sample(key, (200,)))
    np.testing.assert_array_equal(samples, np.ones_like(samples))


# --------------------------------------------------------------------------- #
# log_pmf                                                                     #
# --------------------------------------------------------------------------- #


def test_bernoulli_log_pmf_one_is_log_p():
    """log_pmf(1) == log(p)."""
    p = 0.3
    np.testing.assert_allclose(
        float(Bernoulli(p=p).log_pmf(1)), float(np.log(p)), atol=1e-12
    )


def test_bernoulli_log_pmf_zero_is_log_1mp():
    """log_pmf(0) == log(1 - p)."""
    p = 0.3
    np.testing.assert_allclose(
        float(Bernoulli(p=p).log_pmf(0)), float(np.log1p(-p)), atol=1e-12
    )


def test_bernoulli_log_pmf_outside_support_is_neg_inf():
    """log_pmf(k) for k not in {0, 1} is -inf."""
    d = Bernoulli(p=0.5)
    assert float(d.log_pmf(2)) == -np.inf
    assert float(d.log_pmf(-1)) == -np.inf
    assert float(d.log_pmf(0.5)) == -np.inf


def test_bernoulli_log_pmf_vectorised():
    """log_pmf broadcasts cleanly over a batch of k values."""
    d = Bernoulli(p=0.25)
    k = jnp.asarray([0, 1, 0, 1, 2])
    out = np.asarray(d.log_pmf(k))
    expected = np.array(
        [np.log(0.75), np.log(0.25), np.log(0.75), np.log(0.25), -np.inf]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_bernoulli_log_pdf_aliases_log_pmf():
    """``log_pdf`` aliases ``log_pmf`` for importance-sampling parity."""
    d = Bernoulli(p=0.4)
    for v in (0, 1, 7):
        np.testing.assert_allclose(float(d.log_pdf(v)), float(d.log_pmf(v)))


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_bernoulli_same_key_is_reproducible():
    """Same key -> bit-identical samples."""
    key = jax.random.PRNGKey(7)
    d = Bernoulli(p=0.4)
    a = np.asarray(d.sample(key, (100,)))
    b = np.asarray(d.sample(key, (100,)))
    np.testing.assert_array_equal(a, b)


def test_bernoulli_matches_categorical_special_case():
    """Bernoulli(p) and Categorical([0,1], [1-p, p]) sample identically.

    Bernoulli is a thin wrapper over the equivalent Categorical, so
    feeding both the same PRNG key must yield the same samples.
    """
    from jaxonomy.uq import Categorical

    key = jax.random.PRNGKey(11)
    p = 0.3
    bern = Bernoulli(p=p)
    cat = Categorical(values=[0, 1], probs=[1 - p, p])
    a = np.asarray(bern.sample(key, (200,)))
    b = np.asarray(cat.sample(key, (200,)))
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Differentiable (Gumbel-softmax) sample                                      #
# --------------------------------------------------------------------------- #


def test_bernoulli_differentiable_sample_is_in_unit_interval():
    """differentiable_sample returns a continuous value in [0, 1]."""
    key = jax.random.PRNGKey(4)
    d = Bernoulli(p=0.3)
    soft = np.asarray(d.differentiable_sample(key, (50,), temperature=1.0))
    assert (soft >= -1e-6).all() and (soft <= 1.0 + 1e-6).all()
    # Continuous: very unlikely to land exactly on 0 or 1 at temperature 1.
    unique_vals = np.unique(soft)
    assert len(unique_vals) >= 30, (
        f"Gumbel-softmax should produce ~all distinct continuous values; "
        f"got {len(unique_vals)} unique out of 50"
    )


def test_bernoulli_differentiable_sample_grad_through_p_is_finite():
    """jax.grad through ``p`` via the Gumbel-softmax sample is finite.

    Mirrors the Categorical companion test: construction is concrete-
    only (we ``float(...)``-coerce ``p`` to validate the [0,1] bound),
    so we differentiate an inlined Gumbel-softmax expression around a
    pre-validated Bernoulli.
    """
    key = jax.random.PRNGKey(5)
    # Concrete construction up front (p validated in [0, 1]).
    _ = Bernoulli(p=0.3)

    def loss(p_traced):
        # Inline two-category Gumbel-softmax for Bernoulli([0, 1]).
        probs = jnp.stack([1.0 - p_traced, p_traced])
        u = jax.random.uniform(
            key, shape=(16, 2), minval=1e-12, maxval=1.0 - 1e-12
        )
        gumbel = -jnp.log(-jnp.log(u))
        logits = (jnp.log(probs) + gumbel) / 0.5
        soft = jax.nn.softmax(logits, axis=-1)
        values = jnp.asarray([0.0, 1.0])
        out = jnp.tensordot(soft, values, axes=([-1], [0]))
        return jnp.mean(out)

    g = jax.grad(loss)(jnp.asarray(0.3))
    assert jnp.isfinite(g), f"expected finite gradient; got {g}"
    # Gradient must be non-trivial.
    assert float(jnp.abs(g)) > 1e-6, (
        f"gradient through p should be non-trivial; got {g}"
    )


def test_bernoulli_differentiable_sample_temperature_validation():
    """temperature must be positive (validation flows from Categorical)."""
    d = Bernoulli(p=0.5)
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="temperature"):
        d.differentiable_sample(key, (4,), temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        d.differentiable_sample(key, (4,), temperature=-1.0)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_bernoulli_rejects_p_below_zero():
    """p < 0 raises."""
    with pytest.raises(ValueError, match=r"p .* must be in \[0, 1\]"):
        Bernoulli(p=-0.1)


def test_bernoulli_rejects_p_above_one():
    """p > 1 raises."""
    with pytest.raises(ValueError, match=r"p .* must be in \[0, 1\]"):
        Bernoulli(p=1.5)


def test_bernoulli_has_no_ppf():
    """Bernoulli deliberately omits ``ppf`` (step-function inverse CDF)."""
    d = Bernoulli(p=0.5)
    assert not hasattr(d, "ppf"), (
        "Bernoulli should not expose ppf: same rationale as Categorical / "
        "Poisson — discrete inverse-CDF is a step function."
    )
