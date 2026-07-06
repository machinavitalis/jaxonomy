# SPDX-License-Identifier: MIT

"""T-126-followup-latin-hypercube — stratified / Latin Hypercube sampling.

Covers :func:`jaxonomy.uq.latin_hypercube_sample` (already shipped via T-101)
and :func:`jaxonomy.uq.latin_hypercube_centered_sample` (new, deterministic
counterpart):

1. Jittered LHS: each of the ``n_samples`` strata of ``[0, 1]`` contains
   exactly one sample (Latin square property along each parameter).
2. Centered LHS: with ``key=None`` the samples are exactly the stratum
   midpoints ``(i + 0.5) / n`` in increasing order — fully deterministic.
3. Centered LHS with a key: samples are still a permutation of the
   midpoints (each midpoint hit exactly once, just reordered).
4. Multi-parameter independence: each parameter's stratification is
   permuted independently of the others (no diagonal degeneracy).
5. Inverse-CDF mapping: jittered + centered LHS over ``Uniform(low, high)``
   land in ``[low, high]`` and respect the stratum boundaries.
6. Validation: bad ``n_samples`` / empty dists raise.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Normal,
    Uniform,
    latin_hypercube_centered_sample,
    latin_hypercube_sample,
)

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Jittered LHS — each stratum gets exactly one sample                         #
# --------------------------------------------------------------------------- #


def test_lhs_uniform_each_stratum_has_one_sample():
    """Latin square property: each of n_samples strata contains exactly 1 sample."""
    key = jax.random.PRNGKey(0)
    n = 10
    out = latin_hypercube_sample({"x": Uniform(0.0, 1.0)}, n_samples=n, key=key)
    samples = np.asarray(out["x"])
    assert samples.shape == (n,), f"expected ({n},), got {samples.shape}"
    # All samples land in [0, 1].
    assert ((samples >= 0.0) & (samples <= 1.0)).all()
    # Bin into the n equal strata and check each holds exactly one sample.
    bins = np.minimum(np.floor(samples * n).astype(int), n - 1)
    counts = np.bincount(bins, minlength=n)
    assert (counts == 1).all(), (
        f"LHS strata occupancy not exactly 1: {counts.tolist()}"
    )


def test_lhs_uniform_each_stratum_one_sample_n100():
    """Same property at n=100 to rule out small-n flukes."""
    key = jax.random.PRNGKey(1)
    n = 100
    out = latin_hypercube_sample({"x": Uniform(0.0, 1.0)}, n_samples=n, key=key)
    samples = np.asarray(out["x"])
    bins = np.minimum(np.floor(samples * n).astype(int), n - 1)
    counts = np.bincount(bins, minlength=n)
    assert (counts == 1).all(), counts.tolist()


# --------------------------------------------------------------------------- #
# Centered LHS — deterministic stratum midpoints                              #
# --------------------------------------------------------------------------- #


def test_centered_lhs_no_key_returns_exact_midpoints_in_order():
    """key=None: deterministic, samples are exactly (i + 0.5) / n in order."""
    n = 10
    out = latin_hypercube_centered_sample({"x": Uniform(0.0, 1.0)}, n_samples=n)
    samples = np.asarray(out["x"])
    expected = (np.arange(n) + 0.5) / n  # 0.05, 0.15, ..., 0.95
    np.testing.assert_allclose(samples, expected, atol=1e-12)


def test_centered_lhs_no_key_is_deterministic_across_calls():
    """Two key=None calls must return byte-equal samples."""
    a = latin_hypercube_centered_sample({"x": Uniform(0.0, 1.0)}, n_samples=8)
    b = latin_hypercube_centered_sample({"x": Uniform(0.0, 1.0)}, n_samples=8)
    np.testing.assert_array_equal(np.asarray(a["x"]), np.asarray(b["x"]))


def test_centered_lhs_with_key_is_permutation_of_midpoints():
    """With a key, samples are still a permutation of the n midpoints."""
    n = 12
    key = jax.random.PRNGKey(7)
    out = latin_hypercube_centered_sample(
        {"x": Uniform(0.0, 1.0)}, n_samples=n, key=key
    )
    samples = np.asarray(out["x"])
    expected_set = (np.arange(n) + 0.5) / n
    # Same multiset, possibly reordered.
    np.testing.assert_allclose(np.sort(samples), expected_set, atol=1e-12)


def test_centered_lhs_inverse_cdf_maps_into_distribution_support():
    """ppf mapping: Uniform(low, high) midpoints land in [low, high]."""
    n = 20
    low, high = 2.0, 5.0
    out = latin_hypercube_centered_sample(
        {"x": Uniform(low, high)}, n_samples=n
    )
    samples = np.asarray(out["x"])
    assert ((samples >= low) & (samples <= high)).all()
    # Midpoints map to: low + (high - low) * (i + 0.5) / n
    expected = low + (high - low) * (np.arange(n) + 0.5) / n
    np.testing.assert_allclose(samples, expected, atol=1e-12)


def test_centered_lhs_normal_ppf_uses_inverse_cdf():
    """Centered LHS over Normal: samples equal Phi^{-1}((i + 0.5) / n)."""
    import jax.scipy.stats as jss

    n = 5
    out = latin_hypercube_centered_sample(
        {"z": Normal(loc=0.0, scale=1.0)}, n_samples=n
    )
    samples = np.asarray(out["z"])
    u = (np.arange(n) + 0.5) / n
    expected = np.asarray(jss.norm.ppf(u))
    np.testing.assert_allclose(samples, expected, atol=1e-10)


# --------------------------------------------------------------------------- #
# Multi-parameter independence (Latin square across parameters)               #
# --------------------------------------------------------------------------- #


def test_lhs_multiparam_each_param_independently_stratified():
    """For each of d parameters, every stratum holds exactly one sample."""
    key = jax.random.PRNGKey(42)
    n = 16
    dists = {
        "a": Uniform(0.0, 1.0),
        "b": Uniform(0.0, 1.0),
        "c": Uniform(0.0, 1.0),
    }
    out = latin_hypercube_sample(dists, n_samples=n, key=key)
    for name in dists:
        samples = np.asarray(out[name])
        assert samples.shape == (n,)
        bins = np.minimum(np.floor(samples * n).astype(int), n - 1)
        counts = np.bincount(bins, minlength=n)
        assert (counts == 1).all(), (
            f"LHS param {name!r} strata occupancy not exactly 1: {counts.tolist()}"
        )


def test_lhs_multiparam_strata_orderings_are_independent():
    """Stratum permutations across parameters are not all the same.

    If they were identical, LHS would degenerate to a 1-D sweep (perfect
    diagonal correlation).  Independent permutations make this exceedingly
    unlikely at n=20 across 3 parameters.
    """
    key = jax.random.PRNGKey(123)
    n = 20
    dists = {
        "a": Uniform(0.0, 1.0),
        "b": Uniform(0.0, 1.0),
        "c": Uniform(0.0, 1.0),
    }
    out = latin_hypercube_sample(dists, n_samples=n, key=key)
    a = np.asarray(out["a"])
    b = np.asarray(out["b"])
    c = np.asarray(out["c"])
    # Convert each parameter's samples to its stratum index (the "permutation"
    # of strata).  If two parameters' permutations were identical the rank
    # vectors would coincide.
    rank = lambda v: np.argsort(np.argsort(v))
    assert not np.array_equal(rank(a), rank(b)), (
        "params a and b have identical stratum ordering — independence violated"
    )
    assert not np.array_equal(rank(a), rank(c))
    assert not np.array_equal(rank(b), rank(c))


def test_centered_lhs_multiparam_with_key_independent_permutations():
    """Centered + key: each parameter's permutation is independent."""
    key = jax.random.PRNGKey(99)
    n = 24
    dists = {
        "a": Uniform(0.0, 1.0),
        "b": Uniform(0.0, 1.0),
        "c": Uniform(0.0, 1.0),
    }
    out = latin_hypercube_centered_sample(dists, n_samples=n, key=key)
    a = np.asarray(out["a"])
    b = np.asarray(out["b"])
    c = np.asarray(out["c"])
    expected_set = (np.arange(n) + 0.5) / n
    for v in (a, b, c):
        np.testing.assert_allclose(np.sort(v), expected_set, atol=1e-12)
    # Permutations are not identical.
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(b, c)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_centered_lhs_rejects_nonpositive_n_samples():
    with pytest.raises(ValueError, match="n_samples must be > 0"):
        latin_hypercube_centered_sample({"x": Uniform(0.0, 1.0)}, n_samples=0)
    with pytest.raises(ValueError, match="n_samples must be > 0"):
        latin_hypercube_centered_sample({"x": Uniform(0.0, 1.0)}, n_samples=-3)


def test_centered_lhs_rejects_empty_distributions():
    with pytest.raises(ValueError, match="must be non-empty"):
        latin_hypercube_centered_sample({}, n_samples=10)
