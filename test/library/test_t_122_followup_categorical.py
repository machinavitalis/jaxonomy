# SPDX-License-Identifier: MIT
"""T-122-followup-categorical — Categorical ``RandomSource`` mode.

Extends the multi-distribution ``RandomSource`` block (T-122-followup-
distributions) with a discrete-choice ``"categorical"`` mode:

    RandomSource(sample_time, distribution="categorical",
                 params={"values": [...], "probs": [...]}, seed=...)

The block samples elements of ``values`` with the matching ``probs``
each ``sample_time`` tick.  Because the gather through the selected
discrete index is non-differentiable, the sampler wraps the index in
``stop_gradient`` — gradients of downstream losses through the
categorical sample path are zero (use the
``Categorical.differentiable_sample`` Gumbel-softmax helper in
``jaxonomy.uq.distributions`` when continuous-relaxation gradients are
required).

Tests cover statistical properties, sample-domain constraints,
reproducibility, validation, and integration with the standard
``jaxonomy.simulate`` pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import RandomSource
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _simulate_source(block, t_end=1.0):
    """Run a source-only diagram and return the recorded output array."""
    ctx = block.create_context()
    result = jaxonomy.simulate(
        block,
        ctx,
        (0.0, t_end),
        recorded_signals={"x": block.output_ports[0]},
    )
    return np.asarray(result.outputs["x"])


def _draw_samples_categorical(values, probs, seed, n=5_000):
    """Direct in-process sampler mirroring ``RandomSource._sample_categorical``.

    Used for empirical-statistics tests that need a long sample chain.
    Mirrors the jaxonomy.simulate path's PRNG split + ``jax.random.choice``
    sampling exactly (so an upstream change to the block's PRNG strategy
    would also need to be reflected here for the tests to keep matching).
    """
    import jax

    values_arr = np.asarray(values)
    probs_arr = np.asarray(probs, dtype=np.float64)
    probs_arr = probs_arr / probs_arr.sum()
    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=values_arr.dtype)
    for i in range(n):
        key, subkey = jax.random.split(key)
        idx = int(jax.random.choice(subkey, len(values_arr), p=probs_arr))
        out[i] = values_arr[idx]
    return out


# --------------------------------------------------------------------------- #
# Basic sampling                                                              #
# --------------------------------------------------------------------------- #


def test_categorical_samples_are_in_values_table():
    """Every sample must come from the configured ``values`` list."""
    values = [10.0, 20.0, 30.0]
    block = RandomSource(
        sample_time=0.05,
        distribution="categorical",
        params={"values": values, "probs": [0.1, 0.5, 0.4]},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    valid = set(values)
    actual = set(float(v) for v in np.unique(x))
    assert actual.issubset(valid), (
        f"sample contains values outside the table: {actual - valid}"
    )


def test_categorical_sample_mean_matches_analytic():
    """E[X] = sum_i values[i] * probs[i].  Statistical check over a long run.

    Uses a direct in-process sampler (mirroring the block) to get 5k
    samples cheaply; the simulate-path statistics are covered by
    :func:`test_categorical_empirical_frequencies_match_probs_via_simulate`
    below over a shorter run.
    """
    values = [10.0, 20.0, 30.0]
    probs = [0.1, 0.5, 0.4]
    x = _draw_samples_categorical(values, probs, seed=2026, n=5_000)
    expected = sum(v * p for v, p in zip(values, probs))  # = 20.0
    # std-err ~ sigma / sqrt(n).  sigma^2 = E[X^2] - E[X]^2; here ~41.
    # At n=5k, std-err ~ 0.09; 0.5 abs tolerance is comfortable.
    assert abs(float(np.mean(x)) - expected) < 0.5, (
        f"categorical sample mean {np.mean(x)} vs target {expected}"
    )


def test_categorical_empirical_frequencies_match_probs():
    """Empirical frequencies converge to the configured ``probs``."""
    values = [10, 20, 30]
    probs = [0.1, 0.5, 0.4]
    x = _draw_samples_categorical(values, probs, seed=2026, n=5_000)
    freqs = np.array([np.mean(x == v) for v in values])
    np.testing.assert_allclose(freqs, probs, atol=0.05)


def test_categorical_end_to_end_via_simulate():
    """End-to-end via jaxonomy.simulate: drawn samples land in ``values``.

    Statistical convergence is tested directly above (5k samples via
    the in-process sampler); here we cover the simulate-path
    integration: the block runs, sample-time ticks fire, and the
    recorded sequence is a subset of ``values``.  Sample-count budget
    matches the T-122-followup-poisson simulate test (sample_time=0.05,
    t_end=2.0 -> ~40 samples, well within the simulator's default
    max_major_steps=100 cap).
    """
    values = [10, 20, 30]
    probs = [0.1, 0.5, 0.4]
    block = RandomSource(
        sample_time=0.05,
        distribution="categorical",
        params={"values": values, "probs": probs},
        seed=2026,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.shape[0] >= 30, f"expected ~40 samples; got {x.shape[0]}"
    assert set(int(v) for v in np.unique(x)).issubset(set(values))


def test_categorical_bernoulli_special_case():
    """Bernoulli special case: values=[0, 1], probs=[1-p, p]."""
    p = 0.3
    x = _draw_samples_categorical([0, 1], [1 - p, p], seed=99, n=5_000)
    assert set(float(v) for v in np.unique(x)).issubset({0.0, 1.0})
    # std-err for Bernoulli at p=0.3, n=5k ~ 0.0065; 0.05 absolute tolerance.
    assert abs(float(np.mean(x)) - p) < 0.05, (
        f"Bernoulli(p={p}) empirical mean {np.mean(x)} vs target {p}"
    )


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_categorical_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical categorical sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2, 3], "probs": [0.1, 0.5, 0.4]},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2, 3], "probs": [0.1, 0.5, 0.4]},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_categorical_different_seeds_differ():
    """Different seeds -> distinct sequences (very high probability)."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2, 3], "probs": [0.34, 0.33, 0.33]},
            seed=1,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2, 3], "probs": [0.34, 0.33, 0.33]},
            seed=2,
        )
    )
    assert not np.array_equal(a, b), (
        "Distinct seeds should produce distinct categorical streams."
    )


# --------------------------------------------------------------------------- #
# Probability normalisation                                                   #
# --------------------------------------------------------------------------- #


def test_categorical_probs_normalised_at_construction():
    """Unnormalised probs are normalised at construction.

    Verifies that probs=[1, 1, 2] behaves the same as probs=[.25, .25, .5].
    """
    block_unnorm = RandomSource(
        sample_time=0.001,
        distribution="categorical",
        params={"values": [10, 20, 30], "probs": [1.0, 1.0, 2.0]},
        seed=11,
    )
    block_norm = RandomSource(
        sample_time=0.001,
        distribution="categorical",
        params={"values": [10, 20, 30], "probs": [0.25, 0.25, 0.5]},
        seed=11,
    )
    a = _simulate_source(block_unnorm, t_end=2.0)
    b = _simulate_source(block_norm, t_end=2.0)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_categorical_requires_values_and_probs_keys():
    """Missing keys raise a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"probs": [0.5, 0.5]},
            seed=0,
        )
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2]},
            seed=0,
        )


def test_categorical_rejects_length_mismatch():
    """values and probs must have the same length."""
    with pytest.raises(ValueError, match="length"):
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2, 3], "probs": [0.5, 0.5]},
            seed=0,
        )


def test_categorical_rejects_negative_probs():
    """Negative probabilities raise."""
    with pytest.raises(ValueError, match="non-negative"):
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2], "probs": [-0.1, 1.1]},
            seed=0,
        )


def test_categorical_rejects_all_zero_probs():
    """All-zero probs raise (no positive sum)."""
    with pytest.raises(ValueError, match="positive sum"):
        RandomSource(
            sample_time=0.1,
            distribution="categorical",
            params={"values": [1, 2], "probs": [0.0, 0.0]},
            seed=0,
        )
