# SPDX-License-Identifier: MIT
"""T-122-followup-beta-gamma — Beta and Gamma ``RandomSource`` modes.

Extends the multi-distribution ``RandomSource`` block (T-122-followup-
distributions) with two continuous modes:

    RandomSource(sample_time, distribution="beta",
                 params={"alpha": ..., "beta":  ...}, seed=...)
    RandomSource(sample_time, distribution="gamma",
                 params={"shape": ..., "scale": ...}, seed=...)

Beta covers bounded fractions on ``[0, 1]`` (utilisation, mixture weights,
Bayesian priors over a Bernoulli p).  Gamma covers positive-valued
quantities (wait times, physical parameters, rate priors).

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


def _simulate_source(block, t_end=2.0):
    """Run a source-only diagram and return the recorded output array."""
    ctx = block.create_context()
    result = jaxonomy.simulate(
        block,
        ctx,
        (0.0, t_end),
        recorded_signals={"x": block.output_ports[0]},
    )
    return np.asarray(result.outputs["x"])


def _draw_samples_beta(alpha, beta, seed, n=5_000):
    """Direct in-process Beta sampler mirroring the block path.

    Uses ``jax.random.beta`` per-tick (matching the block's split + draw
    pattern) so long-run statistics can be checked without paying the
    per-tick simulator overhead.
    """
    import jax

    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        key, subkey = jax.random.split(key)
        out[i] = float(jax.random.beta(subkey, float(alpha), float(beta)))
    return out


def _draw_samples_gamma(shape_p, scale, seed, n=5_000):
    """Direct in-process Gamma sampler mirroring the block path."""
    import jax

    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        key, subkey = jax.random.split(key)
        out[i] = float(jax.random.gamma(subkey, float(shape_p)) * float(scale))
    return out


# --------------------------------------------------------------------------- #
# Beta — sampling                                                             #
# --------------------------------------------------------------------------- #


def test_beta_samples_in_unit_interval():
    """Every Beta sample must lie in the open ``(0, 1)`` interval."""
    block = RandomSource(
        sample_time=0.05,
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    assert (x > 0.0).all() and (x < 1.0).all(), (
        f"Beta output leaked outside (0, 1): min={x.min()}, max={x.max()}"
    )


def test_beta_sample_mean_matches_analytic():
    """E[X] = alpha / (alpha + beta).  Beta(2, 5) -> 2/7 ~ 0.2857."""
    alpha, beta = 2.0, 5.0
    x = _draw_samples_beta(alpha=alpha, beta=beta, seed=2026, n=5_000)
    target = alpha / (alpha + beta)
    # std-err for Beta(2,5) at n=5k ~ sqrt(0.0255/5000) ~ 0.00226.
    # 0.02 absolute tolerance gives ~9sigma comfort.
    assert abs(float(np.mean(x)) - target) < 0.02, (
        f"Beta({alpha},{beta}) empirical mean {np.mean(x)} vs target {target}"
    )


def test_beta_end_to_end_via_simulate():
    """End-to-end via jaxonomy.simulate: drawn samples land in (0, 1)."""
    block = RandomSource(
        sample_time=0.05,
        distribution="beta",
        params={"alpha": 2.0, "beta": 5.0},
        seed=2026,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.shape[0] >= 30, f"expected ~40 samples; got {x.shape[0]}"
    assert (x > 0.0).all() and (x < 1.0).all()
    # Spread sanity-check: with Beta(2, 5) over ~40 ticks we should see
    # multiple distinct values.
    assert len(np.unique(x)) >= 30


def test_beta_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical Beta sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_beta_different_seeds_differ():
    """Different seeds -> distinct sequences (very high probability)."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            seed=1,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="beta",
            params={"alpha": 2.0, "beta": 5.0},
            seed=2,
        )
    )
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Beta — validation                                                           #
# --------------------------------------------------------------------------- #


def test_beta_requires_alpha_and_beta_keys():
    """Missing ``alpha`` or ``beta`` raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="beta",
            params={"alpha": 2.0},
            seed=0,
        )
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="beta",
            params={"beta": 2.0},
            seed=0,
        )


# --------------------------------------------------------------------------- #
# Gamma — sampling                                                            #
# --------------------------------------------------------------------------- #


def test_gamma_samples_positive():
    """Every Gamma sample must lie in ``(0, inf)``."""
    block = RandomSource(
        sample_time=0.05,
        distribution="gamma",
        params={"shape": 2.0, "scale": 3.0},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    assert (x > 0.0).all(), f"Gamma output leaked into x <= 0: min={x.min()}"


def test_gamma_sample_mean_matches_analytic():
    """E[X] = shape * scale.  Gamma(2, 3) -> mean = 6."""
    shape_p, scale = 2.0, 3.0
    x = _draw_samples_gamma(shape_p=shape_p, scale=scale, seed=2026, n=5_000)
    target = shape_p * scale
    # Var = shape*scale^2 = 18 -> std-err = sqrt(18/5000) ~ 0.060.
    # 0.30 absolute tolerance gives ~5sigma comfort.
    assert abs(float(np.mean(x)) - target) < 0.30, (
        f"Gamma({shape_p},{scale}) empirical mean {np.mean(x)} vs target {target}"
    )


def test_gamma_end_to_end_via_simulate():
    """End-to-end via jaxonomy.simulate: drawn samples are positive."""
    block = RandomSource(
        sample_time=0.05,
        distribution="gamma",
        params={"shape": 2.0, "scale": 3.0},
        seed=2026,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.shape[0] >= 30, f"expected ~40 samples; got {x.shape[0]}"
    assert (x > 0.0).all()
    assert len(np.unique(x)) >= 30


def test_gamma_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical Gamma sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="gamma",
            params={"shape": 2.0, "scale": 3.0},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="gamma",
            params={"shape": 2.0, "scale": 3.0},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_gamma_different_seeds_differ():
    """Different seeds -> distinct sequences (very high probability)."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="gamma",
            params={"shape": 2.0, "scale": 3.0},
            seed=1,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="gamma",
            params={"shape": 2.0, "scale": 3.0},
            seed=2,
        )
    )
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Gamma — validation                                                          #
# --------------------------------------------------------------------------- #


def test_gamma_requires_shape_and_scale_keys():
    """Missing ``shape`` or ``scale`` raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="gamma",
            params={"shape": 2.0},
            seed=0,
        )
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="gamma",
            params={"scale": 2.0},
            seed=0,
        )


def test_gamma_shape_does_not_collide_with_output_shape():
    """``gamma``'s ``shape`` param coexists with the output-shape kwarg.

    ``shape`` is both a static parameter of every ``RandomSource``
    (the output-array shape) and a parameter of the gamma distribution
    (the shape parameter ``k``).  Verify both flow through cleanly: a
    vector-output gamma block constructs and samples without error.
    """
    block = RandomSource(
        sample_time=0.05,
        distribution="gamma",
        params={"shape": 2.0, "scale": 3.0},
        seed=11,
        shape=(3,),
    )
    x = _simulate_source(block, t_end=0.5)
    assert x.ndim == 2 and x.shape[1] == 3
    assert (x > 0.0).all()


# --------------------------------------------------------------------------- #
# Cross-cutting: unknown distribution still rejected                          #
# --------------------------------------------------------------------------- #


def test_unknown_distribution_lists_beta_and_gamma():
    """Error message for an unknown distribution must mention the new names."""
    with pytest.raises(ValueError, match="unknown distribution") as excinfo:
        RandomSource(
            sample_time=0.1,
            distribution="bogus",
            params={},
            seed=0,
        )
    msg = str(excinfo.value)
    assert "beta" in msg and "gamma" in msg, (
        f"Error message should advertise new distributions; got: {msg}"
    )
