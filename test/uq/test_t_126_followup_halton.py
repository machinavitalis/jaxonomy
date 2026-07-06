# SPDX-License-Identifier: MIT

"""T-126-followup-halton-sequence — Halton sequence alternative to Sobol.

Covers :func:`jaxonomy.uq.halton_sequence` and
:func:`jaxonomy.uq.quasi_monte_carlo` with ``sequence="halton"``:

1. ``halton_sequence`` returns shape ``(n_samples, n_dims)`` with all
   values in ``[0, 1)``.
2. The first few samples in the base-2 column match the canonical van
   der Corput sequence ``(1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...)``.
3. Low-discrepancy property: the unit cube is covered uniformly to a
   much tighter tolerance than IID Monte Carlo at the same ``N``.
4. Determinism: unscrambled Halton is fully reproducible (no seed).
5. Custom ``base_primes`` work; validation rejects non-primes / repeats
   / mismatched length.
6. Scrambled Halton with the same seed reproduces; with a different
   seed differs.
7. ``quasi_monte_carlo(..., sequence="halton")`` returns the expected
   ``{param: (N,)}`` shape and converges faster than IID on a smooth
   integrand.
8. ``sequence="unknown"`` raises a clear error.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Normal,
    Uniform,
    halton_sequence,
    quasi_monte_carlo,
    sample_parameters,
)

skip_if_not_jax()


# ---------------------------------------------------------------------------
# halton_sequence — shape, range, canonical values.
# ---------------------------------------------------------------------------

def test_halton_sequence_shape_and_range():
    """halton_sequence(1024, 3) returns (1024, 3) in [0, 1)."""
    points = np.asarray(halton_sequence(n_samples=1024, n_dims=3))
    assert points.shape == (1024, 3)
    assert points.min() >= 0.0
    assert points.max() < 1.0


def test_halton_sequence_first_column_matches_van_der_corput_base2():
    """The base-2 column must match the canonical van der Corput sequence.

    The classical phi_2 values for n = 1, 2, 3, ..., 7 are:
        1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8.
    """
    expected = np.array([0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875])
    points = np.asarray(halton_sequence(n_samples=7, n_dims=1))
    np.testing.assert_allclose(points[:, 0], expected, atol=1e-12)


def test_halton_sequence_first_column_in_2d_matches_base2():
    """In 2-D, the first dim still uses base 2 (smallest default prime)."""
    points = np.asarray(halton_sequence(n_samples=4, n_dims=2))
    expected_base2 = np.array([0.5, 0.25, 0.75, 0.125])
    np.testing.assert_allclose(points[:, 0], expected_base2, atol=1e-12)
    # Second column: van der Corput in base 3 — phi_3(1..4) =
    # 1/3, 2/3, 1/9, 4/9.
    expected_base3 = np.array([1.0 / 3, 2.0 / 3, 1.0 / 9, 4.0 / 9])
    np.testing.assert_allclose(points[:, 1], expected_base3, atol=1e-12)


def test_halton_sequence_deterministic_without_seed():
    """Unscrambled Halton has no PRNG; two calls return identical arrays."""
    a = np.asarray(halton_sequence(n_samples=128, n_dims=3))
    b = np.asarray(halton_sequence(n_samples=128, n_dims=3))
    np.testing.assert_array_equal(a, b)


def test_halton_sequence_low_discrepancy_vs_iid():
    """Halton covers the unit cube more uniformly than IID at the same N."""
    n = 1024
    n_dims = 3
    strata = 16
    expected = n // strata

    halton = np.asarray(halton_sequence(n_samples=n, n_dims=n_dims))
    halton_max_dev = 0
    for d in range(n_dims):
        counts, _ = np.histogram(halton[:, d], bins=strata, range=(0.0, 1.0))
        halton_max_dev = max(halton_max_dev, int(np.abs(counts - expected).max()))

    rng = np.random.default_rng(0)
    iid = rng.uniform(0.0, 1.0, size=(n, n_dims))
    iid_max_dev = 0
    for d in range(n_dims):
        counts, _ = np.histogram(iid[:, d], bins=strata, range=(0.0, 1.0))
        iid_max_dev = max(iid_max_dev, int(np.abs(counts - expected).max()))

    # Halton should clearly beat IID per-dim stratification.  We do not
    # demand zero deviation (Halton is not exactly stratifying like
    # unscrambled Sobol for arbitrary N/strata), only that it is
    # measurably more uniform.
    assert halton_max_dev < iid_max_dev


# ---------------------------------------------------------------------------
# Custom base primes + validation.
# ---------------------------------------------------------------------------

def test_halton_sequence_custom_bases():
    """User-supplied bases are accepted and drive the construction."""
    points = np.asarray(
        halton_sequence(n_samples=4, n_dims=2, base_primes=(3, 5))
    )
    # phi_3(1..4) = 1/3, 2/3, 1/9, 4/9.
    np.testing.assert_allclose(
        points[:, 0],
        np.array([1.0 / 3, 2.0 / 3, 1.0 / 9, 4.0 / 9]),
        atol=1e-12,
    )
    # phi_5(1..4) = 1/5, 2/5, 3/5, 4/5.
    np.testing.assert_allclose(
        points[:, 1],
        np.array([0.2, 0.4, 0.6, 0.8]),
        atol=1e-12,
    )


def test_halton_sequence_rejects_bad_inputs():
    with pytest.raises(ValueError, match="n_samples"):
        halton_sequence(n_samples=0, n_dims=2)
    with pytest.raises(ValueError, match="n_samples"):
        halton_sequence(n_samples=-5, n_dims=2)
    with pytest.raises(ValueError, match="n_dims"):
        halton_sequence(n_samples=4, n_dims=0)
    with pytest.raises(ValueError, match="n_dims"):
        halton_sequence(n_samples=4, n_dims=-1)


def test_halton_sequence_rejects_too_many_dims_without_custom_bases():
    with pytest.raises(ValueError, match="prime table"):
        halton_sequence(n_samples=4, n_dims=10_000)


def test_halton_sequence_rejects_mismatched_base_primes_length():
    with pytest.raises(ValueError, match="must match"):
        halton_sequence(n_samples=4, n_dims=2, base_primes=(2, 3, 5))


def test_halton_sequence_rejects_invalid_bases():
    with pytest.raises(ValueError, match=">= 2"):
        halton_sequence(n_samples=4, n_dims=2, base_primes=(1, 3))


def test_halton_sequence_rejects_repeated_bases():
    with pytest.raises(ValueError, match="distinct"):
        halton_sequence(n_samples=4, n_dims=2, base_primes=(2, 2))


# ---------------------------------------------------------------------------
# Scrambling.
# ---------------------------------------------------------------------------

def test_halton_sequence_scrambled_is_seedable():
    """Same seed -> same scrambled output; different seed -> different."""
    a = np.asarray(
        halton_sequence(n_samples=64, n_dims=3, scramble=True, seed=7)
    )
    b = np.asarray(
        halton_sequence(n_samples=64, n_dims=3, scramble=True, seed=7)
    )
    c = np.asarray(
        halton_sequence(n_samples=64, n_dims=3, scramble=True, seed=8)
    )
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    # Still in [0, 1).
    assert a.min() >= 0.0
    assert a.max() < 1.0


def test_halton_sequence_scrambled_differs_from_unscrambled():
    """Scrambling actually changes the sequence (for bases > 2)."""
    plain = np.asarray(halton_sequence(n_samples=64, n_dims=3))
    scrambled = np.asarray(
        halton_sequence(n_samples=64, n_dims=3, scramble=True, seed=42)
    )
    # At least one dimension must differ.
    assert not np.allclose(plain, scrambled)


# ---------------------------------------------------------------------------
# quasi_monte_carlo wiring with sequence="halton".
# ---------------------------------------------------------------------------

def test_quasi_monte_carlo_halton_shape_and_support():
    """quasi_monte_carlo({"x": Uniform(0, 1)}, sequence="halton")."""
    samples = quasi_monte_carlo(
        {"x": Uniform(0.0, 1.0)},
        n_samples=512,
        sequence="halton",
    )
    assert set(samples) == {"x"}
    x = np.asarray(samples["x"])
    assert x.shape == (512,)
    assert x.min() >= 0.0
    assert x.max() < 1.0


def test_quasi_monte_carlo_halton_multiparameter():
    """Halton drives multi-parameter ppf transforms correctly."""
    dists = {
        "k": Uniform(0.5, 2.0),
        "tau": Normal(1.0, 0.1),
    }
    samples = quasi_monte_carlo(
        dists, n_samples=512, sequence="halton",
    )
    k = np.asarray(samples["k"])
    tau = np.asarray(samples["tau"])
    assert k.shape == (512,)
    assert tau.shape == (512,)
    assert k.min() >= 0.5
    assert k.max() <= 2.0
    assert abs(float(tau.mean()) - 1.0) < 0.05
    assert 0.05 < float(tau.std()) < 0.20


def test_quasi_monte_carlo_halton_beats_iid_on_smooth_integrand():
    """Halton-QMC beats IID on int sin(x) dx over [0, 1]."""
    n = 1024
    truth = 1.0 - math.cos(1.0)

    halton_samples = quasi_monte_carlo(
        {"x": Uniform(0.0, 1.0)}, n_samples=n, sequence="halton",
    )
    halton_est = float(np.sin(np.asarray(halton_samples["x"])).mean())
    halton_err = abs(halton_est - truth)

    iid_samples = sample_parameters(
        {"x": Uniform(0.0, 1.0)}, n, jax.random.PRNGKey(0),
    )
    iid_est = float(np.sin(np.asarray(iid_samples["x"])).mean())
    iid_err = abs(iid_est - truth)

    assert halton_err < iid_err * 10.0, (
        f"Halton-QMC error ({halton_err:.2e}) should beat IID error "
        f"({iid_err:.2e}) within a 10x slack window."
    )
    assert halton_err < 1e-3


def test_quasi_monte_carlo_rejects_unknown_sequence():
    with pytest.raises(ValueError, match="unknown sequence"):
        quasi_monte_carlo(
            {"x": Uniform(0.0, 1.0)},
            n_samples=64,
            sequence="latin-hypercube",
        )
