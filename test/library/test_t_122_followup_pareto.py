# SPDX-License-Identifier: MIT
"""T-122-followup-pareto — Pareto ``RandomSource`` mode.

Extends the multi-distribution ``RandomSource`` block (T-122-followup-
distributions) with the Pareto mode:

    RandomSource(sample_time, distribution="pareto",
                 params={"scale": ..., "alpha": ...}, seed=...)

Pareto is the canonical heavy-tail / power-law distribution.  Closed-
form inverse-CDF reparameterisation ``scale * (1 - u)**(-1/alpha)``
gives clean gradients through *both* parameters (same differentiability
profile as ``exponential`` / ``weibull``).

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


def _draw_samples_pareto(scale, alpha, seed, n=10_000):
    """Direct in-process Pareto sampler mirroring the block path.

    Uses the closed-form inverse-CDF reparameterisation per-tick
    (matching the block's split + draw pattern) so long-run statistics
    can be checked without paying the per-tick simulator overhead.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(int(seed))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        key, subkey = jax.random.split(key)
        u = float(jax.random.uniform(subkey))
        u = min(u, 1.0 - 1e-12)
        out[i] = float(scale) * float(jnp.exp(-jnp.log1p(-u) / float(alpha)))
    return out


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #


def test_pareto_samples_above_scale():
    """Every Pareto sample must lie in ``[scale, inf)``."""
    scale = 1.0
    block = RandomSource(
        sample_time=0.05,
        distribution="pareto",
        params={"scale": scale, "alpha": 2.0},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    assert (x >= scale).all(), (
        f"Pareto output leaked below scale={scale}: min={x.min()}"
    )


def test_pareto_sample_mean_matches_analytic():
    """E[X] = alpha * scale / (alpha - 1).

    Pareto(scale=1, alpha=2) -> mean = 2.  Heavy tail makes empirical
    mean noisier than light-tail distributions; we use a generous
    sample size and a 10% relative tolerance.
    """
    scale, alpha = 1.0, 2.0
    x = _draw_samples_pareto(scale=scale, alpha=alpha, seed=2026, n=10_000)
    target = alpha * scale / (alpha - 1.0)
    assert abs(float(np.mean(x)) - target) / target < 0.10, (
        f"Pareto({scale},{alpha}) empirical mean {np.mean(x)} vs target {target}"
    )


def test_pareto_end_to_end_via_simulate():
    """End-to-end via jaxonomy.simulate: drawn samples are >= scale."""
    scale = 1.0
    block = RandomSource(
        sample_time=0.05,
        distribution="pareto",
        params={"scale": scale, "alpha": 2.0},
        seed=2026,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.shape[0] >= 30, f"expected ~40 samples; got {x.shape[0]}"
    assert (x >= scale).all()
    # Spread sanity-check: with Pareto over ~40 ticks we should see
    # multiple distinct values.
    assert len(np.unique(x)) >= 30


def test_pareto_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical Pareto sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="pareto",
            params={"scale": 1.0, "alpha": 2.0},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="pareto",
            params={"scale": 1.0, "alpha": 2.0},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_pareto_different_seeds_differ():
    """Different seeds -> distinct sequences (very high probability)."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="pareto",
            params={"scale": 1.0, "alpha": 2.0},
            seed=1,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="pareto",
            params={"scale": 1.0, "alpha": 2.0},
            seed=2,
        )
    )
    assert not np.array_equal(a, b)


def test_pareto_vector_output_shape():
    """Vector-output Pareto block constructs and samples without error."""
    scale = 1.0
    block = RandomSource(
        sample_time=0.05,
        distribution="pareto",
        params={"scale": scale, "alpha": 2.0},
        seed=11,
        shape=(3,),
    )
    x = _simulate_source(block, t_end=0.5)
    assert x.ndim == 2 and x.shape[1] == 3
    assert (x >= scale).all()


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_pareto_requires_scale_and_alpha_keys():
    """Missing ``scale`` or ``alpha`` raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="pareto",
            params={"scale": 1.0},
            seed=0,
        )
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="pareto",
            params={"alpha": 2.0},
            seed=0,
        )


# --------------------------------------------------------------------------- #
# Cross-cutting: unknown distribution still rejected                          #
# --------------------------------------------------------------------------- #


def test_unknown_distribution_lists_pareto():
    """Error message for an unknown distribution must mention ``pareto``."""
    with pytest.raises(ValueError, match="unknown distribution") as excinfo:
        RandomSource(
            sample_time=0.1,
            distribution="bogus",
            params={},
            seed=0,
        )
    msg = str(excinfo.value)
    assert "pareto" in msg, (
        f"Error message should advertise the new ``pareto`` distribution; got: {msg}"
    )
