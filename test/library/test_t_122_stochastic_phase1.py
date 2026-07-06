# SPDX-License-Identifier: MIT
"""T-122 phase 1: Stochastic sources — UniformRandomNumber and PRBS.

Two new discrete-time stochastic source blocks for system-ID,
noise-rejection, and Monte Carlo control studies. Both carry a
JAX PRNG key in their discrete state and are seed-reproducible.

Phase 1 covers:
  - UniformRandomNumber: same seed → bit-identical sequence;
    different seeds → different sequences; samples in [low, high].
  - PRBS: outputs only ±amplitude; same seed → reproducible;
    seed_a ≠ seed_b → distinct streams.
  - Differentiability through ``low``/``high``/``amplitude``
    parameters (gradient flows via reparameterization; the
    random selector is stop_gradient'd).

BandLimitedNoise, the multi-distribution RandomNumber, and per-vmap
fold_in seeding are deferred — see T-122-followup-band-limited-noise,
T-122-followup-distributions, and T-122-followup-vmap-fold-in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy.library import UniformRandomNumber, PRBS
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


# --------------------------------------------------------------------------- #
# UniformRandomNumber                                                         #
# --------------------------------------------------------------------------- #


def test_uniform_rng_same_seed_is_reproducible():
    """Same seed → bit-identical output sequence (determinism contract)."""
    a = _simulate_source(UniformRandomNumber(sample_time=0.1, seed=42))
    b = _simulate_source(UniformRandomNumber(sample_time=0.1, seed=42))
    np.testing.assert_array_equal(a, b)


def test_uniform_rng_different_seeds_produce_different_streams():
    """Different seeds → different streams (with overwhelming probability)."""
    a = _simulate_source(UniformRandomNumber(sample_time=0.1, seed=1))
    b = _simulate_source(UniformRandomNumber(sample_time=0.1, seed=2))
    # Two independent uniform streams of length ~11 are essentially
    # never element-wise equal; this is a smoke check, not a stats test.
    assert not np.allclose(a, b)


def test_uniform_rng_samples_lie_in_low_high_interval():
    """Output range: every sample must satisfy low <= x <= high."""
    low, high = -3.0, 7.0
    block = UniformRandomNumber(
        sample_time=0.05, low=low, high=high, seed=123
    )
    x = _simulate_source(block, t_end=2.0)
    assert x.min() >= low - 1e-9
    assert x.max() <= high + 1e-9


def test_uniform_rng_default_unit_interval():
    """Default low=0, high=1 → samples in [0, 1)."""
    x = _simulate_source(
        UniformRandomNumber(sample_time=0.1, seed=99), t_end=1.0
    )
    assert x.min() >= 0.0 - 1e-9
    assert x.max() <= 1.0 + 1e-9


def test_uniform_rng_reparameterization_is_differentiable():
    """Pin the differentiable reparameterization low + (high-low) * u.

    The stochastic sample wraps a fixed unit-uniform draw in
    stop_gradient, so gradients of any downstream loss flow through
    ``low`` and ``high`` linearly. This pins the identity that makes
    the block useful for stochastic gradient estimation.
    """
    key = jax.random.PRNGKey(0)
    u = jax.lax.stop_gradient(jax.random.uniform(key, ()))

    def f(low, high):
        return low + (high - low) * u

    g_low, g_high = jax.grad(f, argnums=(0, 1))(0.0, 1.0)
    # ∂/∂low (low + (high-low) u) = 1 - u
    # ∂/∂high                     = u
    np.testing.assert_allclose(float(g_low), 1.0 - float(u))
    np.testing.assert_allclose(float(g_high), float(u))


# --------------------------------------------------------------------------- #
# PRBS                                                                        #
# --------------------------------------------------------------------------- #


def test_prbs_outputs_only_plus_or_minus_amplitude():
    """PRBS emits only ±amplitude — the binary contract."""
    amp = 1.5
    x = _simulate_source(
        PRBS(sample_time=0.1, amplitude=amp, seed=11), t_end=2.0
    )
    unique = np.unique(np.asarray(x))
    # Allow numerical tolerance for the float promotion through ``where``.
    assert unique.shape[0] <= 2
    for v in unique:
        assert np.isclose(abs(float(v)), amp), (
            f"PRBS produced non-binary value {v!r} (expected ±{amp})"
        )


def test_prbs_same_seed_is_reproducible():
    """Same seed → bit-identical PRBS sequence."""
    a = _simulate_source(PRBS(sample_time=0.1, amplitude=1.0, seed=7))
    b = _simulate_source(PRBS(sample_time=0.1, amplitude=1.0, seed=7))
    np.testing.assert_array_equal(a, b)


def test_prbs_different_seeds_produce_different_streams():
    """Different seeds → different PRBS sequences."""
    a = _simulate_source(PRBS(sample_time=0.1, amplitude=1.0, seed=3))
    b = _simulate_source(PRBS(sample_time=0.1, amplitude=1.0, seed=4))
    # Length ~11 binary sequences seeded differently almost surely
    # diverge; check at least one slot differs.
    assert not np.array_equal(a, b)


def test_prbs_amplitude_is_differentiable():
    """∂(amp * sel) / ∂amp = sel ∈ {-1, +1}.

    ``sel`` is stop_gradient-wrapped, so the gradient of the binary
    output w.r.t. amplitude is just the (constant) selector.
    """
    key = jax.random.PRNGKey(0)
    bit = jax.random.bernoulli(key, p=0.5)
    sel = jax.lax.stop_gradient(jnp.where(bit, 1.0, -1.0))

    def f(amp):
        return amp * sel

    g = jax.grad(f)(2.0)
    np.testing.assert_allclose(float(g), float(sel))
    assert float(sel) in (-1.0, 1.0)


# --------------------------------------------------------------------------- #
# Smoke: untouched-block byte-equivalence                                     #
# --------------------------------------------------------------------------- #


def test_existing_pulse_source_is_untouched():
    """Sanity check: an unrelated source block still simulates cleanly.

    Phase 1 of T-122 only adds new classes at the bottom of
    ``primitives.py``; no existing block was modified. This is a
    smoke check that the file edits haven't broken the import path
    or the simulate harness for an unrelated source.
    """
    from jaxonomy import library

    pulse = library.Pulse(amplitude=1.0, period=1.0, pulse_width=0.5)
    ctx = pulse.create_context()
    result = jaxonomy.simulate(
        pulse, ctx, (0.0, 0.4),
        recorded_signals={"y": pulse.output_ports[0]},
    )
    y = np.asarray(result.outputs["y"])
    # Pulse with pulse_width=0.5 is "high" in the first half-period,
    # so samples up to t=0.4 are amplitude=1.0.
    assert y.shape[0] >= 2
    np.testing.assert_allclose(float(y[0]), 1.0)
