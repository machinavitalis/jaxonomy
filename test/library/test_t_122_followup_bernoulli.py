# SPDX-License-Identifier: MIT
"""T-122-followup-bernoulli — Bernoulli ``RandomSource`` mode.

Extends the multi-distribution ``RandomSource`` block (T-122-followup-
distributions) with a binary-outcome ``"bernoulli"`` mode:

    RandomSource(sample_time, distribution="bernoulli",
                 params={"p": 0.3}, seed=...)

This is a thin convenience wrapper over the equivalent
``Categorical([0, 1], [1-p, p])`` spelling — the single-scalar ``p`` API
is far more readable for the common Bernoulli-trial / binary-event /
mask-signal use case.

Tests cover statistical properties, sample-domain constraints,
reproducibility, validation, and end-to-end integration via
``jaxonomy.simulate``.
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


def _draw_samples_bernoulli(p, seed, n=5_000):
    """Direct in-process Bernoulli sampler mirroring the block.

    Mirrors the jaxonomy.simulate path's PRNG split + ``jax.random.bernoulli``
    sampling exactly so empirical-statistics tests can use long sample
    chains without paying the per-tick simulator overhead.
    """
    import jax

    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        key, subkey = jax.random.split(key)
        out[i] = int(jax.random.bernoulli(subkey, float(p)))
    return out


# --------------------------------------------------------------------------- #
# Basic sampling                                                              #
# --------------------------------------------------------------------------- #


def test_bernoulli_samples_are_zero_or_one():
    """Every sample must be exactly 0 or 1."""
    block = RandomSource(
        sample_time=0.05,
        distribution="bernoulli",
        params={"p": 0.5},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    actual = set(int(v) for v in np.unique(x))
    assert actual.issubset({0, 1}), (
        f"Bernoulli output contains values outside {{0, 1}}: {actual}"
    )


def test_bernoulli_sample_mean_matches_p():
    """E[X] = p.  Long-run check with the in-process sampler."""
    p = 0.3
    x = _draw_samples_bernoulli(p=p, seed=2026, n=5_000)
    # std-err for Bernoulli at p=0.3, n=5k ~ sqrt(p(1-p)/n) ~ 0.0065.
    # 0.03 absolute tolerance gives ~5sigma comfort.
    assert abs(float(np.mean(x)) - p) < 0.03, (
        f"Bernoulli(p={p}) empirical mean {np.mean(x)} vs target {p}"
    )


def test_bernoulli_end_to_end_via_simulate():
    """End-to-end via jaxonomy.simulate: drawn samples land in {0, 1}.

    Statistical convergence is tested directly above (5k samples via
    the in-process sampler); here we cover the simulate-path
    integration: the block runs, sample-time ticks fire, and the
    recorded sequence is a subset of {0, 1}.
    """
    p = 0.5
    block = RandomSource(
        sample_time=0.05,
        distribution="bernoulli",
        params={"p": p},
        seed=2026,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.shape[0] >= 30, f"expected ~40 samples; got {x.shape[0]}"
    assert set(int(v) for v in np.unique(x)).issubset({0, 1})
    # Sanity-check that p=0.5 produces both outcomes over ~40 ticks
    # (the probability of all-zero or all-one is < 1e-12 here).
    assert len(set(int(v) for v in np.unique(x))) == 2, (
        f"p=0.5 over 40 ticks should produce both 0 and 1; got {np.unique(x)}"
    )


def test_bernoulli_p_zero_is_all_zeros():
    """p=0 -> always 0."""
    block = RandomSource(
        sample_time=0.05,
        distribution="bernoulli",
        params={"p": 0.0},
        seed=1,
    )
    x = _simulate_source(block, t_end=1.0)
    np.testing.assert_array_equal(x, np.zeros_like(x))


def test_bernoulli_p_one_is_all_ones():
    """p=1 -> always 1."""
    block = RandomSource(
        sample_time=0.05,
        distribution="bernoulli",
        params={"p": 1.0},
        seed=1,
    )
    x = _simulate_source(block, t_end=1.0)
    np.testing.assert_array_equal(x, np.ones_like(x))


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_bernoulli_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical Bernoulli sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="bernoulli",
            params={"p": 0.4},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="bernoulli",
            params={"p": 0.4},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_bernoulli_different_seeds_differ():
    """Different seeds -> distinct sequences (very high probability).

    Uses p=0.5 to maximise per-tick entropy so distinct seeds almost
    surely produce distinct bit patterns over ~40 ticks.
    """
    a = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="bernoulli",
            params={"p": 0.5},
            seed=1,
        ),
        t_end=2.0,
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="bernoulli",
            params={"p": 0.5},
            seed=2,
        ),
        t_end=2.0,
    )
    assert not np.array_equal(a, b), (
        "Distinct seeds should produce distinct Bernoulli streams."
    )


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_bernoulli_requires_p_key():
    """Missing ``p`` raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="bernoulli",
            params={},
            seed=0,
        )


def test_bernoulli_rejects_p_below_zero():
    """p < 0 raises."""
    with pytest.raises(ValueError, match=r"p .* must be in \[0, 1\]"):
        RandomSource(
            sample_time=0.1,
            distribution="bernoulli",
            params={"p": -0.1},
            seed=0,
        )


def test_bernoulli_rejects_p_above_one():
    """p > 1 raises."""
    with pytest.raises(ValueError, match=r"p .* must be in \[0, 1\]"):
        RandomSource(
            sample_time=0.1,
            distribution="bernoulli",
            params={"p": 1.1},
            seed=0,
        )
