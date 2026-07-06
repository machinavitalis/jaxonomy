# SPDX-License-Identifier: MIT
"""T-122-followup-weibull — Weibull ``RandomSource`` mode.

Extends the multi-distribution ``RandomSource`` block (T-122-followup-
distributions) with the Weibull mode:

    RandomSource(sample_time, distribution="weibull",
                 params={"shape": ..., "scale": ...}, seed=...)

Weibull is the canonical reliability / time-to-failure / wind-speed
distribution.  Closed-form inverse-CDF reparameterisation
``scale * (-log(1-u))**(1/shape)`` gives clean gradients through *both*
parameters (better than ``gamma``'s implicit reparam).

Tests cover statistical properties, sample-domain constraints,
reproducibility, validation, the ``shape``-vs-output-``shape`` rename
collision (same pattern as ``gamma``), and end-to-end integration via
``jaxonomy.simulate``.
"""

from __future__ import annotations

import math

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


def _draw_samples_weibull(shape_p, scale, seed, n=5_000):
    """Direct in-process Weibull sampler mirroring the block path.

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
        out[i] = float(scale) * float(jnp.power(-jnp.log1p(-u), 1.0 / float(shape_p)))
    return out


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #


def test_weibull_samples_positive():
    """Every Weibull sample must lie in ``(0, inf)``."""
    block = RandomSource(
        sample_time=0.05,
        distribution="weibull",
        params={"shape": 2.0, "scale": 1.0},
        seed=42,
    )
    x = _simulate_source(block, t_end=2.0)
    assert (x > 0.0).all(), f"Weibull output leaked into x <= 0: min={x.min()}"


def test_weibull_sample_mean_matches_analytic():
    """E[X] = scale * Gamma(1 + 1/shape).

    Weibull(2, 1) -> mean = Gamma(1.5) ~ 0.8862.
    """
    shape_p, scale = 2.0, 1.0
    x = _draw_samples_weibull(shape_p=shape_p, scale=scale, seed=2026, n=5_000)
    target = scale * math.gamma(1.0 + 1.0 / shape_p)
    # Var ~ 0.2146 -> std-err = sqrt(0.2146/5000) ~ 0.00655.
    # 0.03 absolute tolerance gives ~4.6sigma comfort.
    assert abs(float(np.mean(x)) - target) < 0.03, (
        f"Weibull({shape_p},{scale}) empirical mean {np.mean(x)} vs target {target}"
    )


def test_weibull_end_to_end_via_simulate():
    """End-to-end via jaxonomy.simulate: drawn samples are positive."""
    block = RandomSource(
        sample_time=0.05,
        distribution="weibull",
        params={"shape": 2.0, "scale": 1.0},
        seed=2026,
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.shape[0] >= 30, f"expected ~40 samples; got {x.shape[0]}"
    assert (x > 0.0).all()
    # Spread sanity-check: with Weibull(2, 1) over ~40 ticks we should
    # see multiple distinct values.
    assert len(np.unique(x)) >= 30


def test_weibull_same_seed_is_reproducible():
    """Determinism contract: same seed -> bit-identical Weibull sequence."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="weibull",
            params={"shape": 2.0, "scale": 1.0},
            seed=7,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.1,
            distribution="weibull",
            params={"shape": 2.0, "scale": 1.0},
            seed=7,
        )
    )
    np.testing.assert_array_equal(a, b)


def test_weibull_different_seeds_differ():
    """Different seeds -> distinct sequences (very high probability)."""
    a = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="weibull",
            params={"shape": 2.0, "scale": 1.0},
            seed=1,
        )
    )
    b = _simulate_source(
        RandomSource(
            sample_time=0.05,
            distribution="weibull",
            params={"shape": 2.0, "scale": 1.0},
            seed=2,
        )
    )
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_weibull_requires_shape_and_scale_keys():
    """Missing ``shape`` or ``scale`` raises a clear ValueError."""
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="weibull",
            params={"shape": 2.0},
            seed=0,
        )
    with pytest.raises(ValueError, match="missing"):
        RandomSource(
            sample_time=0.1,
            distribution="weibull",
            params={"scale": 2.0},
            seed=0,
        )


def test_weibull_shape_does_not_collide_with_output_shape():
    """``weibull``'s ``shape`` param coexists with the output-shape kwarg.

    ``shape`` is both a static parameter of every ``RandomSource``
    (the output-array shape) and a parameter of the Weibull distribution
    (the shape parameter ``k``).  Verify both flow through cleanly: a
    vector-output Weibull block constructs and samples without error.
    Same collision-rename pattern as ``gamma`` (T-122-followup-beta-gamma).
    """
    block = RandomSource(
        sample_time=0.05,
        distribution="weibull",
        params={"shape": 2.0, "scale": 1.0},
        seed=11,
        shape=(3,),
    )
    x = _simulate_source(block, t_end=0.5)
    assert x.ndim == 2 and x.shape[1] == 3
    assert (x > 0.0).all()


# --------------------------------------------------------------------------- #
# Cross-cutting: unknown distribution still rejected                          #
# --------------------------------------------------------------------------- #


def test_unknown_distribution_lists_weibull():
    """Error message for an unknown distribution must mention ``weibull``."""
    with pytest.raises(ValueError, match="unknown distribution") as excinfo:
        RandomSource(
            sample_time=0.1,
            distribution="bogus",
            params={},
            seed=0,
        )
    msg = str(excinfo.value)
    assert "weibull" in msg, (
        f"Error message should advertise the new ``weibull`` distribution; got: {msg}"
    )
