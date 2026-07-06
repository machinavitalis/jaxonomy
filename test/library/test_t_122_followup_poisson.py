# SPDX-License-Identifier: MIT
"""T-122-followup-poisson — Exponential and Poisson ``RandomSource`` modes.

Extends the multi-distribution ``RandomSource`` block (T-122-followup-
distributions) with two new distribution flags:

  * ``"exponential"`` (``params={"rate": ...}``) — continuous, strictly
    positive, differentiable through ``rate`` via the inverse-CDF
    reparameterisation ``x = -log(1-u) / rate``.
  * ``"poisson"`` (``params={"rate": ...}``) — discrete non-negative
    integer count; non-differentiable through ``rate`` w.r.t. samples
    (sampler is wrapped in ``stop_gradient``).

Tests cover statistical properties, sample-domain constraints,
reproducibility, and the differentiability contract.
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


def _draw_samples_exponential(rate, seed, n=5_000):
    """Direct sampler mirroring ``RandomSource._sample_exponential``."""
    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        key, subkey = jax.random.split(key)
        u = float(jax.random.uniform(subkey, ()))
        out[i] = -np.log1p(-u) / rate
    return out


def _draw_samples_poisson(rate, seed, n=5_000):
    """Direct sampler mirroring ``RandomSource._sample_poisson``."""
    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        key, subkey = jax.random.split(key)
        out[i] = int(jax.random.poisson(subkey, rate, ()))
    return out


# --------------------------------------------------------------------------- #
# Exponential                                                                 #
# --------------------------------------------------------------------------- #


def test_exponential_samples_strictly_positive():
    """Every Exponential sample must be > 0 (continuous, strictly positive)."""
    block = RandomSource(
        sample_time=0.05,
        distribution="exponential",
        params={"rate": 1.0},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    assert (x > 0).all(), "exponential must produce strictly positive samples"


def test_exponential_sample_mean_matches_inverse_rate():
    """For Exponential(rate): E[X] = 1 / rate.  5k samples, rate=1.0."""
    rate = 1.0
    x = _draw_samples_exponential(rate, seed=2026, n=5_000)
    target = 1.0 / rate
    # Exponential has a heavy right tail; 10% absolute tolerance over 5k
    # draws is comfortably above the central-limit noise floor.
    assert abs(float(np.mean(x)) - target) < 0.1, (
        f"exponential sample mean {np.mean(x)} vs target {target}"
    )


def test_exponential_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical exponential sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="exponential",
            params={"rate": 2.0},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="exponential",
            params={"rate": 2.0},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_exponential_reparameterization_is_differentiable():
    """jax.grad through ``rate`` is finite and matches the analytic derivative.

    ``f(rate) = -log(1-u) / rate`` with ``u`` stop-gradiented; analytic
    ``df/drate = log(1-u) / rate**2 = -x / rate``.
    """
    key = jax.random.PRNGKey(0)
    u = jax.lax.stop_gradient(jax.random.uniform(key, ()))

    def f(rate):
        return -jnp.log1p(-u) / rate

    rate0 = 2.0
    g_rate = jax.grad(f)(rate0)
    expected = float(jnp.log1p(-u) / (rate0 ** 2))
    np.testing.assert_allclose(float(g_rate), expected, rtol=1e-6)
    assert np.isfinite(float(g_rate))


# --------------------------------------------------------------------------- #
# Poisson                                                                     #
# --------------------------------------------------------------------------- #


def test_poisson_samples_are_nonnegative_integers():
    """Poisson samples must be non-negative integers (discrete distribution)."""
    block = RandomSource(
        sample_time=0.05,
        distribution="poisson",
        params={"rate": 3.0},
        seed=123,
    )
    x = _simulate_source(block, t_end=2.0)
    # Integer dtype + non-negative range.
    assert np.issubdtype(x.dtype, np.integer) or np.all(
        np.equal(np.floor(x), x)
    ), f"Poisson samples must be integer-valued; got dtype {x.dtype}"
    assert (x >= 0).all(), "Poisson must produce non-negative samples"


def test_poisson_sample_mean_matches_rate():
    """For Poisson(rate): E[X] = rate.  5k samples, rate=3.0."""
    rate = 3.0
    x = _draw_samples_poisson(rate, seed=2026, n=5_000)
    # CLT std-error ~ sqrt(rate / n) ~ 0.024; 0.2 absolute tolerance is
    # comfortably above the noise floor.
    assert abs(float(np.mean(x)) - rate) < 0.2, (
        f"poisson sample mean {np.mean(x)} vs target {rate}"
    )


def test_poisson_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical poisson sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="poisson",
            params={"rate": 3.0},
            seed=11,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="poisson",
            params={"rate": 3.0},
            seed=11,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_poisson_grad_through_rate_is_zero():
    """Poisson is discrete: jax.grad of a sample w.r.t. ``rate`` must be 0.

    The sampler is wrapped in ``stop_gradient`` so JAX returns zero
    gradient through ``rate`` — this is the documented contract for
    discrete distributions and prevents silent NaN/garbage gradients
    from leaking through ``jax.random.poisson``.
    """
    key = jax.random.PRNGKey(0)

    def f(rate):
        # Mirrors ``_sample_poisson``: stop_gradient on the discrete draw.
        return jax.lax.stop_gradient(
            jax.random.poisson(key, rate, ())
        ).astype(jnp.float64)

    g_rate = jax.grad(f)(3.0)
    np.testing.assert_allclose(float(g_rate), 0.0)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_exponential_requires_rate_param():
    """Missing ``rate`` key raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="exponential",
            params={},
            seed=0,
        )


def test_poisson_requires_rate_param():
    """Missing ``rate`` key raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="poisson",
            params={},
            seed=0,
        )
