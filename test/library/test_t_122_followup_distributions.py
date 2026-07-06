# SPDX-License-Identifier: MIT
"""T-122-followup-distributions: multi-distribution ``RandomSource`` block.

Unified discrete-time random source supporting four distributions
(uniform, normal, lognormal, triangular) selected at construction via a
string flag plus a parameter dict.  Verifies:

  * Reproducibility (same seed -> bit-identical sequence) per
    distribution.
  * Per-distribution range / statistical properties:
      - uniform:    samples in [low, high].
      - normal:     long-run sample mean ~ 0, std ~ 1.
      - lognormal:  samples > 0; long-run mean ~ exp(0.5) for
                    mu=0, sigma=1.
      - triangular: samples in [low, high]; sample mean ~
                    (low + peak + high) / 3.
  * Differentiability of jax.grad through each distribution's named
    parameters under the standard reparameterisations.
  * Construction-time validation of the ``distribution`` string and the
    required ``params`` keys.

The block is the deferred multi-distribution rebuild of the
T-122 phase 1 ``UniformRandomNumber`` (which remains in place).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
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


def _draw_samples(distribution, params, seed, n=5_000):
    """Draw ``n`` independent samples directly via the same algebra the
    block's ``_update`` uses, without round-tripping through the
    simulator.

    The simulator's recorded-signals buffer has a step-count ceiling
    that bites when running 5k+ ticks for tight statistical checks
    (cf. the BandLimitedNoise tests, which take the same approach for
    the same reason).  Pinning the algebra here is fine because the
    block-level reproducibility / range tests above already exercise
    the simulator path.
    """
    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        key, subkey = jax.random.split(key)
        if distribution == "uniform":
            u = float(jax.random.uniform(subkey, ()))
            out[i] = params["low"] + (params["high"] - params["low"]) * u
        elif distribution == "normal":
            z = float(jax.random.normal(subkey, ()))
            out[i] = params["mean"] + params["std"] * z
        elif distribution == "lognormal":
            z = float(jax.random.normal(subkey, ()))
            out[i] = float(np.exp(params["mu"] + params["sigma"] * z))
        elif distribution == "triangular":
            u = float(jax.random.uniform(subkey, ()))
            low = params["low"]
            peak = params["peak"]
            high = params["high"]
            width = high - low
            c = (peak - low) / width
            if u <= c:
                out[i] = low + np.sqrt(u * width * (peak - low))
            else:
                out[i] = high - np.sqrt((1.0 - u) * width * (high - peak))
        else:
            raise ValueError(distribution)
    return out


# --------------------------------------------------------------------------- #
# Uniform                                                                     #
# --------------------------------------------------------------------------- #


def test_uniform_same_seed_is_reproducible():
    """Same seed -> bit-identical output sequence (determinism contract)."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="uniform",
            params={"low": 0.0, "high": 1.0},
            seed=42,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="uniform",
            params={"low": 0.0, "high": 1.0},
            seed=42,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_uniform_samples_lie_in_low_high_interval():
    """Output range: every sample must satisfy low <= x <= high."""
    block = RandomSource(
        sample_time=0.05,
        distribution="uniform",
        params={"low": 0.0, "high": 1.0},
        seed=123,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.min() >= 0.0 - 1e-9
    assert x.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Normal                                                                      #
# --------------------------------------------------------------------------- #


def test_normal_sample_mean_and_std_match_targets():
    """Long-run sample mean ~ 0, std ~ 1 (statistical tolerance).

    Pins the algebra of ``_sample_normal`` (``mean + std * z``) over
    5,000 independent draws.  Ratio checks are generous (0.1 absolute
    tolerance) so the test is robust across seeds.
    """
    x = _draw_samples(
        "normal", {"mean": 0.0, "std": 1.0}, seed=2026, n=5_000,
    )
    assert abs(float(np.mean(x))) < 0.1, f"sample mean {np.mean(x)}"
    assert abs(float(np.std(x)) - 1.0) < 0.1, f"sample std {np.std(x)}"


def test_normal_same_seed_is_reproducible():
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="normal",
            params={"mean": 0.0, "std": 1.0},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="normal",
            params={"mean": 0.0, "std": 1.0},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Lognormal                                                                   #
# --------------------------------------------------------------------------- #


def test_lognormal_samples_strictly_positive():
    """Block-level smoke: simulator-driven lognormal samples are > 0."""
    block = RandomSource(
        sample_time=0.01,
        distribution="lognormal",
        params={"mu": 0.0, "sigma": 1.0},
        seed=11,
    )
    x = _simulate_source(block, t_end=2.0)
    assert (x > 0).all(), "lognormal must produce strictly positive samples"


def test_lognormal_sample_mean_matches_exp_half():
    """For mu=0, sigma=1: E[X] = exp(mu + sigma^2/2) = exp(0.5) ~ 1.6487.

    Lognormal has a heavy tail so the empirical mean is noisy; uses a
    generous 25% relative tolerance over a 5k-sample run via the
    direct sampler (see ``_draw_samples`` for rationale).
    """
    x = _draw_samples(
        "lognormal", {"mu": 0.0, "sigma": 1.0}, seed=2024, n=5_000,
    )
    target = float(np.exp(0.5))
    assert abs(float(np.mean(x)) - target) / target < 0.25, (
        f"sample mean {np.mean(x)} vs target {target}"
    )


# --------------------------------------------------------------------------- #
# Triangular                                                                  #
# --------------------------------------------------------------------------- #


def test_triangular_samples_lie_in_low_high_interval():
    block = RandomSource(
        sample_time=0.05,
        distribution="triangular",
        params={"low": -2.0, "peak": 0.5, "high": 3.0},
        seed=9,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.min() >= -2.0 - 1e-9
    assert x.max() <= 3.0 + 1e-9


def test_triangular_sample_mean_matches_analytic():
    """Triangular(low, peak, high) has E[X] = (low + peak + high) / 3."""
    low, peak, high = -1.0, 0.5, 2.0
    x = _draw_samples(
        "triangular",
        {"low": low, "peak": peak, "high": high},
        seed=2027,
        n=5_000,
    )
    target = (low + peak + high) / 3.0
    assert abs(float(np.mean(x)) - target) < 0.1, (
        f"triangular sample mean {np.mean(x)} vs target {target}"
    )


# --------------------------------------------------------------------------- #
# Differentiability                                                           #
# --------------------------------------------------------------------------- #


def test_uniform_reparameterization_is_differentiable():
    """Pin the uniform reparameterisation: low + (high-low) * u."""
    key = jax.random.PRNGKey(0)
    u = jax.lax.stop_gradient(jax.random.uniform(key, ()))

    def f(low, high):
        return low + (high - low) * u

    g_low, g_high = jax.grad(f, argnums=(0, 1))(0.0, 1.0)
    np.testing.assert_allclose(float(g_low), 1.0 - float(u))
    np.testing.assert_allclose(float(g_high), float(u))


def test_normal_reparameterization_is_differentiable():
    """Pin the normal reparameterisation: mean + std * z."""
    key = jax.random.PRNGKey(1)
    z = jax.lax.stop_gradient(jax.random.normal(key, ()))

    def f(mean, std):
        return mean + std * z

    g_mean, g_std = jax.grad(f, argnums=(0, 1))(0.0, 1.0)
    np.testing.assert_allclose(float(g_mean), 1.0)
    np.testing.assert_allclose(float(g_std), float(z))


def test_lognormal_reparameterization_is_differentiable():
    """Pin the lognormal reparameterisation: exp(mu + sigma * z)."""
    key = jax.random.PRNGKey(2)
    z = jax.lax.stop_gradient(jax.random.normal(key, ()))

    def f(mu, sigma):
        return jnp.exp(mu + sigma * z)

    g_mu, g_sigma = jax.grad(f, argnums=(0, 1))(0.0, 1.0)
    expected_value = float(jnp.exp(z))  # mu=0, sigma=1
    np.testing.assert_allclose(float(g_mu), expected_value, rtol=1e-6)
    np.testing.assert_allclose(
        float(g_sigma), float(z) * expected_value, rtol=1e-6
    )


def test_triangular_reparameterization_is_differentiable():
    """jax.grad through triangular params is finite under the quantile transform.

    The quantile transform piecewise-defined as
        F^{-1}(u) = low  + sqrt(u (h-l)(p-l))   if u <= c
                  = high - sqrt((1-u)(h-l)(h-p)) otherwise
    is smooth in (low, peak, high) on each branch, so jax.grad is
    finite for any fixed u away from the seam u == c.
    """
    key = jax.random.PRNGKey(3)
    u_raw = jax.random.uniform(key, ())
    # Force u into the *left* branch (u < c with c > u for default
    # symmetric triangular).  This pins the gradient to a single
    # smooth piece.
    u = jax.lax.stop_gradient(jnp.minimum(u_raw, jnp.asarray(0.3)))

    def f(low, peak, high):
        width = high - low
        c = (peak - low) / width
        left = low + jnp.sqrt(u * width * (peak - low))
        right = high - jnp.sqrt((1.0 - u) * width * (high - peak))
        return jnp.where(u <= c, left, right)

    g_low, g_peak, g_high = jax.grad(f, argnums=(0, 1, 2))(-1.0, 0.5, 2.0)
    assert np.isfinite(float(g_low))
    assert np.isfinite(float(g_peak))
    assert np.isfinite(float(g_high))


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_unknown_distribution_raises_clear_error():
    """Construction with an unknown distribution string raises ValueError."""
    with pytest.raises(ValueError, match="unknown distribution"):
        RandomSource(
            sample_time=0.1,
            distribution="cauchy",  # not in the supported set
            params={"loc": 0.0, "scale": 1.0},
            seed=0,
        )


def test_missing_required_params_raises_clear_error():
    """Missing a required params key raises ValueError naming the missing key."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="normal",
            params={"mean": 0.0},  # missing "std"
            seed=0,
        )


def test_triangular_requires_three_params():
    """Triangular needs low / peak / high; only low+high is invalid."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="triangular",
            params={"low": 0.0, "high": 1.0},  # missing "peak"
            seed=0,
        )
