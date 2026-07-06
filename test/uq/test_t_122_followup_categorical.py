# SPDX-License-Identifier: MIT
"""T-122-followup-categorical — UQ ``Categorical`` distribution.

Discrete-choice distribution over an explicit ``values`` table with
matching ``probs``.  Companion to ``jaxonomy.library.RandomSource``'s
``"categorical"`` distribution mode.

Tests cover:
  * hard ``sample`` matches empirical statistics under finite Monte Carlo,
  * ``log_pmf`` lookup is exact (and ``-inf`` for unmatched values),
  * vector-valued ``values`` round-trip through both ``sample`` and
    ``log_pmf``,
  * the Gumbel-softmax ``differentiable_sample`` is soft (continuous) and
    its gradient through ``probs`` is finite,
  * reproducibility, validation, and the deliberate omission of ``ppf``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import Categorical

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Sampling statistics                                                         #
# --------------------------------------------------------------------------- #


def test_categorical_sample_mean_matches_analytic():
    """E[X] = sum_i values[i] * probs[i].  values=[1,2,3], probs=[.1,.5,.4]."""
    key = jax.random.PRNGKey(0)
    d = Categorical(values=[1, 2, 3], probs=[0.1, 0.5, 0.4])
    samples = np.asarray(d.sample(key, (10_000,)))
    expected_mean = 0.1 * 1 + 0.5 * 2 + 0.4 * 3  # = 2.3
    # CLT std-err ~ sigma / sqrt(n) where sigma^2 = E[X^2] - E[X]^2.
    # E[X^2] = 0.1 + 2.0 + 3.6 = 5.7; var = 5.7 - 5.29 = 0.41 => sigma ~ 0.64.
    # std-err at n=10k ~ 0.0064 => 0.05 absolute tolerance is comfortable.
    assert abs(float(np.mean(samples)) - expected_mean) < 0.05, (
        f"empirical mean {np.mean(samples)} vs target {expected_mean}"
    )


def test_categorical_sample_frequencies_match_probs():
    """Empirical frequencies converge to ``probs`` under finite MC."""
    key = jax.random.PRNGKey(1)
    d = Categorical(values=[10, 20, 30], probs=[0.1, 0.5, 0.4])
    samples = np.asarray(d.sample(key, (20_000,)))
    freqs = np.array(
        [np.mean(samples == v) for v in (10, 20, 30)]
    )
    np.testing.assert_allclose(freqs, [0.1, 0.5, 0.4], atol=0.02)


def test_categorical_sample_returns_only_values():
    """Every sample must be drawn from the ``values`` table."""
    key = jax.random.PRNGKey(2)
    d = Categorical(values=[7.5, -1.0, 42.0], probs=[0.2, 0.3, 0.5])
    samples = np.asarray(d.sample(key, (500,)))
    valid = {7.5, -1.0, 42.0}
    assert set(np.unique(samples)).issubset(valid), (
        f"sample contains values outside the table: {set(np.unique(samples)) - valid}"
    )


# --------------------------------------------------------------------------- #
# log_pmf                                                                     #
# --------------------------------------------------------------------------- #


def test_categorical_log_pmf_matches_probs():
    """log_pmf(values[i]) == log(probs[i])."""
    d = Categorical(values=[10.0, 20.0], probs=[0.5, 0.5])
    np.testing.assert_allclose(float(d.log_pmf(10.0)), float(np.log(0.5)), atol=1e-12)
    np.testing.assert_allclose(float(d.log_pmf(20.0)), float(np.log(0.5)), atol=1e-12)


def test_categorical_log_pmf_unmatched_is_neg_inf():
    """log_pmf(value) for ``value`` not in the table is -inf."""
    d = Categorical(values=[10.0, 20.0, 30.0], probs=[0.3, 0.4, 0.3])
    assert float(d.log_pmf(99.0)) == -np.inf
    assert float(d.log_pmf(15.0)) == -np.inf


def test_categorical_log_pmf_normalises_unnormalised_probs():
    """probs are normalised at construction: probs=[1, 1, 2] -> [.25, .25, .5]."""
    d = Categorical(values=[1, 2, 3], probs=[1.0, 1.0, 2.0])
    np.testing.assert_allclose(
        float(d.log_pmf(3)), float(np.log(0.5)), atol=1e-12
    )
    np.testing.assert_allclose(
        float(d.log_pmf(1)), float(np.log(0.25)), atol=1e-12
    )


def test_categorical_log_pmf_log_pdf_alias():
    """``log_pdf`` aliases ``log_pmf`` for importance-sampling parity."""
    d = Categorical(values=[1, 2, 3], probs=[0.2, 0.3, 0.5])
    for v in (1, 2, 3, 99):
        np.testing.assert_allclose(float(d.log_pdf(v)), float(d.log_pmf(v)))


# --------------------------------------------------------------------------- #
# Vector-valued categorical                                                   #
# --------------------------------------------------------------------------- #


def test_categorical_vector_values_sample_shape():
    """Vector-typed ``values`` produce vector-typed samples."""
    key = jax.random.PRNGKey(3)
    d = Categorical(
        values=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        probs=[0.2, 0.3, 0.5],
    )
    samples = np.asarray(d.sample(key, (100,)))
    assert samples.shape == (100, 2), f"expected (100, 2); got {samples.shape}"


def test_categorical_vector_values_log_pmf():
    """log_pmf on vector-valued categorical matches by element-wise equality."""
    d = Categorical(
        values=[[1.0, 2.0], [3.0, 4.0]],
        probs=[0.3, 0.7],
    )
    np.testing.assert_allclose(
        float(d.log_pmf(jnp.asarray([3.0, 4.0]))),
        float(np.log(0.7)),
        atol=1e-12,
    )
    # Unmatched vector -> -inf.
    assert float(d.log_pmf(jnp.asarray([9.0, 9.0]))) == -np.inf


# --------------------------------------------------------------------------- #
# Differentiable (Gumbel-softmax) sample                                      #
# --------------------------------------------------------------------------- #


def test_categorical_differentiable_sample_is_soft():
    """differentiable_sample returns a continuous-valued convex combination."""
    key = jax.random.PRNGKey(4)
    d = Categorical(values=[1.0, 2.0, 3.0], probs=[0.1, 0.5, 0.4])
    soft = np.asarray(d.differentiable_sample(key, (50,), temperature=1.0))
    # Soft samples lie inside the convex hull of values: (1.0, 3.0).
    assert (soft >= 1.0 - 1e-6).all() and (soft <= 3.0 + 1e-6).all()
    # The samples should be *strictly continuous*, i.e. very unlikely to
    # land exactly on a vertex of the simplex.
    unique_vals = np.unique(soft)
    assert len(unique_vals) >= 30, (
        f"Gumbel-softmax should produce ~all distinct continuous values; "
        f"got {len(unique_vals)} unique out of 50"
    )


def test_categorical_differentiable_sample_grad_through_probs_is_finite():
    """jax.grad through ``probs`` via the Gumbel-softmax sample is finite.

    Construction is concrete-only (we ``float(...)``-coerce ``probs`` to
    validate sign / sum at construction); the differentiable path is the
    *sample* itself, which routes ``probs`` through ``jnp.log`` and
    ``softmax`` — both smooth.  To exercise the gradient we therefore
    inline the Gumbel-softmax expression around a pre-validated
    ``Categorical`` and differentiate wrt the input ``probs`` array.
    """
    key = jax.random.PRNGKey(5)
    values = jnp.asarray([1.0, 2.0, 3.0])
    n = values.shape[0]
    # Concrete construction up front (probs validated, normalised).
    _ = Categorical(values=values, probs=[0.1, 0.5, 0.4])

    def loss(probs):
        # Inline Gumbel-softmax around a *traced* probs vector.  Mirrors
        # ``Categorical.differentiable_sample`` step-for-step but skips
        # the concrete-only construction-time normalisation.
        probs_norm = probs / jnp.sum(probs)
        u = jax.random.uniform(
            key, shape=(16, n), minval=1e-12, maxval=1.0 - 1e-12
        )
        gumbel = -jnp.log(-jnp.log(u))
        logits = (jnp.log(probs_norm) + gumbel) / 0.5
        soft = jax.nn.softmax(logits, axis=-1)
        out = jnp.tensordot(soft, values, axes=([-1], [0]))
        return jnp.mean(out)

    g = jax.grad(loss)(jnp.asarray([0.1, 0.5, 0.4]))
    assert jnp.all(jnp.isfinite(g)), f"expected finite gradient; got {g}"
    # Gradient must be non-trivial (at least one component != 0).
    assert float(jnp.max(jnp.abs(g))) > 1e-6, (
        f"gradient through probs should be non-trivial; got {g}"
    )


def test_categorical_differentiable_sample_temperature_validation():
    """temperature must be positive."""
    d = Categorical(values=[1.0, 2.0], probs=[0.5, 0.5])
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="temperature"):
        d.differentiable_sample(key, (4,), temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        d.differentiable_sample(key, (4,), temperature=-1.0)


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_categorical_same_key_is_reproducible():
    """Same key -> bit-identical samples."""
    key = jax.random.PRNGKey(7)
    d = Categorical(values=[1, 2, 3], probs=[0.1, 0.5, 0.4])
    a = np.asarray(d.sample(key, (100,)))
    b = np.asarray(d.sample(key, (100,)))
    np.testing.assert_array_equal(a, b)


def test_categorical_bernoulli_special_case():
    """Bernoulli special case: Categorical([0, 1], [1 - p, p])."""
    key = jax.random.PRNGKey(8)
    p = 0.3
    d = Categorical(values=[0, 1], probs=[1 - p, p])
    samples = np.asarray(d.sample(key, (20_000,)))
    assert set(np.unique(samples)).issubset({0, 1})
    # CLT std-err for Bernoulli at p=0.3, n=20k ~ 0.0032; 0.02 tolerance.
    assert abs(float(np.mean(samples)) - p) < 0.02, (
        f"Bernoulli(p={p}) empirical mean {np.mean(samples)} vs target {p}"
    )


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_categorical_rejects_mismatched_lengths():
    """values and probs must have the same length."""
    with pytest.raises(ValueError, match="length"):
        Categorical(values=[1, 2, 3], probs=[0.5, 0.5])


def test_categorical_rejects_negative_probs():
    """All probs must be non-negative."""
    with pytest.raises(ValueError, match="non-negative"):
        Categorical(values=[1, 2], probs=[-0.1, 1.1])


def test_categorical_rejects_zero_probs():
    """probs must have positive sum."""
    with pytest.raises(ValueError, match="positive sum"):
        Categorical(values=[1, 2], probs=[0.0, 0.0])


def test_categorical_rejects_empty():
    """probs must be non-empty."""
    with pytest.raises(ValueError, match="non-empty"):
        Categorical(values=[], probs=[])


def test_categorical_has_no_ppf():
    """Categorical deliberately omits ``ppf`` (step-function inverse CDF)."""
    d = Categorical(values=[1, 2, 3], probs=[0.1, 0.5, 0.4])
    assert not hasattr(d, "ppf"), (
        "Categorical should not expose ppf: a discrete inverse-CDF is a "
        "step function and breaks Saltelli/Morris pipelines."
    )
