# SPDX-License-Identifier: MIT
"""T-122-followup-lfsr: true LFSR-based maximal-length PRBS-N source.

T-122 phase 1 shipped a Bernoulli-based ``PRBS`` block. This follow-up
ships ``PRBSLFSR`` -- a real Linear-Feedback Shift Register whose
output sequence has period exactly ``2^N - 1`` and the standard
flat-band spectrum used as a system-identification "white" excitation.

This test suite pins:
  - Reproducibility: same seed -> identical sequence.
  - Period: PRBS-7 has period 127; verified via running ``3*127``
    samples and checking the sequence repeats every 127.
  - Output binary: every sample is exactly ``±amplitude``.
  - Validation: ``register_length`` outside the supported set raises
    a clear ``ValueError`` at construction time.
  - Different seeds: produce different (phase-shifted) sequences.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import PRBSLFSR
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
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_prbslfsr_same_seed_is_reproducible():
    """Same seed -> bit-identical PRBS-LFSR sequence (determinism contract)."""
    a = _simulate_source(
        PRBSLFSR(sample_time=0.1, amplitude=1.0, register_length=7, seed=11),
        t_end=2.0,
    )
    b = _simulate_source(
        PRBSLFSR(sample_time=0.1, amplitude=1.0, register_length=7, seed=11),
        t_end=2.0,
    )
    np.testing.assert_array_equal(a, b)


def test_prbslfsr_different_seeds_produce_different_streams():
    """Different non-zero seeds -> different starting phases on the cycle."""
    a = _simulate_source(
        PRBSLFSR(sample_time=0.1, amplitude=1.0, register_length=7, seed=1),
        t_end=2.0,
    )
    b = _simulate_source(
        PRBSLFSR(sample_time=0.1, amplitude=1.0, register_length=7, seed=2),
        t_end=2.0,
    )
    # Two distinct seeds map to two distinct phases on the same length-127
    # orbit; for any reasonable horizon (here ~21 samples) the two streams
    # almost surely diverge in at least one slot.
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Period                                                                      #
# --------------------------------------------------------------------------- #


def test_prbslfsr_period_is_exactly_2N_minus_1_for_n7():
    """PRBS-7 has period 127; verify the sequence repeats every 127 steps.

    We sample ``3 * 127 = 381`` LFSR steps (plus the initial sample at
    ``t=0``) and check that the second and third 127-sample windows
    bit-match the first. This is the load-bearing property that makes a
    "true" LFSR PRBS suitable for system ID.
    """
    period = 127  # 2^7 - 1
    dt = 1.0
    # Run for slightly more than 3 periods so we collect at least three
    # full windows of 127 samples each.
    n_periods = 3
    block = PRBSLFSR(
        sample_time=dt, amplitude=1.0, register_length=7, seed=1
    )
    x = _simulate_source(block, t_end=dt * (n_periods * period + 0.5))

    # Drop the initial t=0 sample and any tail beyond the third period
    # to align cleanly on period boundaries.
    samples = np.asarray(x)
    assert samples.shape[0] >= n_periods * period + 1, (
        f"need >= {n_periods * period + 1} samples, got {samples.shape[0]}"
    )

    # The first sample is the t=0 read of the seeded register; the
    # following samples are post-update outputs. Use the post-update
    # window for the period check (indices 1 .. 1 + 3*period).
    window = samples[1 : 1 + n_periods * period]
    w0 = window[:period]
    w1 = window[period : 2 * period]
    w2 = window[2 * period : 3 * period]
    np.testing.assert_array_equal(w0, w1)
    np.testing.assert_array_equal(w0, w2)


def test_prbslfsr_visits_both_levels_within_one_period():
    """A maximal-length PRBS-7 must emit both -1 and +1 within one period.

    A degenerate (all-zero or constant) LFSR would only emit one value;
    this is a cheap sanity check that the tap polynomial is wired up.
    """
    period = 127
    block = PRBSLFSR(
        sample_time=1.0, amplitude=1.0, register_length=7, seed=1
    )
    x = _simulate_source(block, t_end=float(period) + 0.5)
    unique = np.unique(np.asarray(x))
    assert -1.0 in unique
    assert 1.0 in unique


# --------------------------------------------------------------------------- #
# Binary output contract                                                      #
# --------------------------------------------------------------------------- #


def test_prbslfsr_outputs_only_plus_or_minus_amplitude():
    """Every sample must be exactly ``±amplitude``."""
    amp = 2.5
    block = PRBSLFSR(
        sample_time=0.1, amplitude=amp, register_length=15, seed=7
    )
    x = _simulate_source(block, t_end=2.0)
    unique = np.unique(np.asarray(x))
    assert unique.shape[0] <= 2
    for v in unique:
        assert np.isclose(abs(float(v)), amp), (
            f"PRBSLFSR produced non-binary value {v!r} (expected ±{amp})"
        )


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_prbslfsr_invalid_register_length_raises():
    """``register_length=8`` is not a supported primitive polynomial."""
    with pytest.raises(ValueError, match="register_length"):
        PRBSLFSR(sample_time=0.1, register_length=8, seed=1)


def test_prbslfsr_zero_seed_is_silently_promoted():
    """The all-zero register is an LFSR fixed point; seed=0 -> seed=1.

    Rather than raise, ``initialize`` quietly bumps a zero seed to 1 so
    the block is ergonomic to use with default-zero parameter sweeps.
    The output must still toggle (not collapse to a constant).
    """
    block = PRBSLFSR(
        sample_time=1.0, amplitude=1.0, register_length=7, seed=0
    )
    x = _simulate_source(block, t_end=20.5)
    # Should produce both +1 and -1 within 20+ samples on a length-127
    # cycle (the seeded register state 1 hits both levels well within
    # one period).
    unique = np.unique(np.asarray(x))
    assert unique.shape[0] == 2


# --------------------------------------------------------------------------- #
# Smoke: existing PRBS still works (no regression on T-122 phase 1)           #
# --------------------------------------------------------------------------- #


def test_t_122_phase1_prbs_still_imports_and_runs():
    """The Bernoulli-based ``PRBS`` from T-122 phase 1 is untouched.

    This test imports the original block, builds a one-period diagram,
    and checks the binary-output contract -- a quick guard that
    appending ``PRBSLFSR`` did not perturb the existing module.
    """
    from jaxonomy.library import PRBS

    x = _simulate_source(
        PRBS(sample_time=0.1, amplitude=1.0, seed=42), t_end=1.0
    )
    unique = np.unique(np.asarray(x))
    assert unique.shape[0] <= 2
    for v in unique:
        assert np.isclose(abs(float(v)), 1.0)
