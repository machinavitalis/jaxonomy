# SPDX-License-Identifier: MIT
"""T-122-followup-vmap-fold-in: per-vmap-batch independent PRNG keys.

The T-122 stochastic sources (``UniformRandomNumber``, ``PRBS``,
``BandLimitedNoise``, ``RandomSource``, ``PRBSLFSR``) carry a JAX PRNG
key (or LFSR register) in their discrete state; the same ``seed``
across replicas of a vmap'd ensemble would otherwise produce *identical*
streams in every replica -- not what an ensemble Monte Carlo wants.

This follow-up adds a ``fold_in_batch_index: bool = False`` kwarg to
each source.  When set and the block is run inside
``simulate_batch(use_vmap=True)`` (or ``simulate_distributed``), the
block looks up ``jax.lax.axis_index("batch")`` inside its per-step
update and folds it into the freshly-split subkey (or, for the LFSR,
XOR-perturbs the register on the very first step).  Outside any
``vmap(axis_name="batch")`` context the kwarg is a graceful no-op
(the unbound-axis ``NameError`` is caught at trace time and the
plain seed-derived stream is used).

This test suite pins:
  * Default ``fold_in_batch_index=False`` -> identical streams across
    vmap replicas (current/byte-equivalent behaviour).
  * ``fold_in_batch_index=True`` under ``simulate_batch(use_vmap=True)``
    -> each replica produces a DIFFERENT sequence.
  * Reproducibility: same seed + ``fold_in_batch_index=True`` produces
    the same per-replica sequences across runs.
  * Outside vmap: ``fold_in_batch_index=True`` is a no-op (plain
    ``simulate`` produces the same sequence as ``False``).
  * All five source classes (``UniformRandomNumber``, ``PRBS``,
    ``BandLimitedNoise``, ``RandomSource``, ``PRBSLFSR``) ship and
    honour the same kwarg.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy.library import (
    BandLimitedNoise,
    PRBS,
    PRBSLFSR,
    RandomSource,
    UniformRandomNumber,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _simulate_source(block, t_end=1.0):
    """Run a source-only diagram serially and return the recorded signal."""
    ctx = block.create_context()
    result = jaxonomy.simulate(
        block,
        ctx,
        (0.0, t_end),
        recorded_signals={"x": block.output_ports[0]},
    )
    return np.asarray(result.outputs["x"])


def _vmap_run(make_block, n_replicas, t_end, dummy_param=None):
    """Run ``make_block(0)`` inside ``simulate_batch(use_vmap=True)``.

    ``simulate_batch`` requires a non-empty ``param_batches`` dict; we
    feed it a no-op constant-per-replica entry referencing one of the
    block's dynamic parameters (e.g. ``"src.low"``) so the vmap path
    actually fires without altering the underlying random stream.

    Returns ``outputs["x"]`` of shape ``(N, T)``.
    """
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(make_block(0))
    diagram = builder.build(name="vmap_fold_in_diag")

    if dummy_param is None:
        dummy_param = ("src.low", 0.0)
    path, value = dummy_param
    param_batches = {path: jnp.full((n_replicas,), float(value))}

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", enable_autodiff=False, max_major_steps=200,
    )
    recorded = {"x": src.output_ports[0]}

    result = jaxonomy.simulate_batch(
        diagram, (0.0, t_end), param_batches,
        options=opts, recorded_signals=recorded, use_vmap=True,
    )
    return np.asarray(result.outputs["x"])


def _pairs_distinct(out):
    """Return the count of pairwise-distinct replicas in ``out``."""
    n = out.shape[0]
    distinct = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not np.array_equal(out[i], out[j]):
                distinct += 1
    return distinct


# --------------------------------------------------------------------------- #
# UniformRandomNumber                                                         #
# --------------------------------------------------------------------------- #


def test_uniform_default_replicas_are_identical():
    """``fold_in_batch_index=False`` (default) -> identical replicas under vmap.

    Pins the byte-equivalent fallback contract: with the default kwarg
    off, every vmap replica draws from the same key sequence so the
    output is identical across replicas.
    """
    out = _vmap_run(
        lambda _i: UniformRandomNumber(
            sample_time=0.1, low=0.0, high=1.0, seed=42, name="src"
        ),
        n_replicas=3,
        t_end=0.6,
    )
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])


def test_uniform_fold_in_makes_replicas_distinct():
    """``fold_in_batch_index=True`` under vmap("batch") -> distinct replicas.

    Each replica must differ from every other replica in at least one
    step -- the canonical "Monte Carlo ensemble" guarantee.
    """
    out = _vmap_run(
        lambda _i: UniformRandomNumber(
            sample_time=0.1, low=0.0, high=1.0, seed=42,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=4,
        t_end=0.6,
    )
    n = out.shape[0]
    assert _pairs_distinct(out) == n * (n - 1) // 2, (
        "Expected all N*(N-1)/2 replica pairs to differ under fold_in."
    )


def test_uniform_fold_in_reproducible_across_runs():
    """Same seed + fold_in -> same per-replica sequences across two runs."""
    a = _vmap_run(
        lambda _i: UniformRandomNumber(
            sample_time=0.1, low=0.0, high=1.0, seed=7,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
    )
    b = _vmap_run(
        lambda _i: UniformRandomNumber(
            sample_time=0.1, low=0.0, high=1.0, seed=7,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
    )
    np.testing.assert_array_equal(a, b)


def test_uniform_fold_in_outside_vmap_is_noop():
    """``fold_in_batch_index=True`` outside any vmap -> bit-identical to False.

    The unbound-axis ``NameError`` from ``jax.lax.axis_index("batch")``
    is caught at trace time, so the plain seed-derived key is used and
    we get exactly the same stream as the kwarg-off case.
    """
    a = _simulate_source(
        UniformRandomNumber(sample_time=0.1, low=0.0, high=1.0, seed=99),
    )
    b = _simulate_source(
        UniformRandomNumber(
            sample_time=0.1, low=0.0, high=1.0, seed=99,
            fold_in_batch_index=True,
        ),
    )
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# PRBS (Bernoulli-based)                                                      #
# --------------------------------------------------------------------------- #


def test_prbs_default_replicas_are_identical():
    out = _vmap_run(
        lambda _i: PRBS(
            sample_time=0.05, amplitude=2.0, seed=11, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
        dummy_param=("src.amplitude", 2.0),
    )
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])


def test_prbs_fold_in_makes_replicas_distinct():
    """PRBS with fold_in -> per-replica distinct binary streams."""
    out = _vmap_run(
        lambda _i: PRBS(
            sample_time=0.05, amplitude=2.0, seed=11,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=4,
        t_end=0.8,
        dummy_param=("src.amplitude", 2.0),
    )
    # Most/all pairs should differ (16 binary samples per replica gives
    # vanishingly small collision probability under independent draws).
    assert _pairs_distinct(out) >= 1, (
        "PRBS fold_in produced identical replicas across all pairs."
    )


def test_prbs_fold_in_outputs_still_binary():
    """Fold-in must not break the +/-amplitude invariant."""
    out = _vmap_run(
        lambda _i: PRBS(
            sample_time=0.05, amplitude=2.0, seed=11,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.5,
        dummy_param=("src.amplitude", 2.0),
    )
    abs_vals = np.abs(out)
    np.testing.assert_allclose(abs_vals, 2.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# BandLimitedNoise                                                            #
# --------------------------------------------------------------------------- #


def test_band_limited_noise_default_replicas_are_identical():
    out = _vmap_run(
        lambda _i: BandLimitedNoise(
            sample_time=0.05, tau=0.5, sigma=1.0, mean=0.0, seed=5,
            name="src",
        ),
        n_replicas=3,
        t_end=0.5,
        dummy_param=("src.mean", 0.0),
    )
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])


def test_band_limited_noise_fold_in_makes_replicas_distinct():
    out = _vmap_run(
        lambda _i: BandLimitedNoise(
            sample_time=0.05, tau=0.5, sigma=1.0, mean=0.0, seed=5,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.5,
        dummy_param=("src.mean", 0.0),
    )
    n = out.shape[0]
    assert _pairs_distinct(out) == n * (n - 1) // 2


# --------------------------------------------------------------------------- #
# RandomSource (multi-distribution)                                           #
# --------------------------------------------------------------------------- #


def test_random_source_default_replicas_are_identical():
    out = _vmap_run(
        lambda _i: RandomSource(
            sample_time=0.1, distribution="normal",
            params={"mean": 0.0, "std": 1.0}, seed=21, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
        dummy_param=("src.mean", 0.0),
    )
    np.testing.assert_array_equal(out[0], out[1])


def test_random_source_fold_in_makes_replicas_distinct():
    out = _vmap_run(
        lambda _i: RandomSource(
            sample_time=0.1, distribution="normal",
            params={"mean": 0.0, "std": 1.0}, seed=21,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
        dummy_param=("src.mean", 0.0),
    )
    n = out.shape[0]
    assert _pairs_distinct(out) == n * (n - 1) // 2


def test_random_source_fold_in_reproducible():
    a = _vmap_run(
        lambda _i: RandomSource(
            sample_time=0.1, distribution="uniform",
            params={"low": -1.0, "high": 1.0}, seed=33,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
        dummy_param=("src.low", -1.0),
    )
    b = _vmap_run(
        lambda _i: RandomSource(
            sample_time=0.1, distribution="uniform",
            params={"low": -1.0, "high": 1.0}, seed=33,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.4,
        dummy_param=("src.low", -1.0),
    )
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# PRBSLFSR (LFSR-based)                                                       #
# --------------------------------------------------------------------------- #


def test_prbslfsr_default_replicas_are_identical():
    """LFSR default: every vmap replica traverses the same orbit."""
    out = _vmap_run(
        lambda _i: PRBSLFSR(
            sample_time=0.01, register_length=7, seed=1, amplitude=1.0,
            name="src",
        ),
        n_replicas=3,
        t_end=0.2,
        dummy_param=("src.amplitude", 1.0),
    )
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])


def test_prbslfsr_fold_in_makes_replicas_distinct():
    """LFSR fold_in: each replica starts at a different phase on the cycle."""
    out = _vmap_run(
        lambda _i: PRBSLFSR(
            sample_time=0.01, register_length=7, seed=1, amplitude=1.0,
            fold_in_batch_index=True, name="src",
        ),
        n_replicas=4,
        t_end=0.5,
        dummy_param=("src.amplitude", 1.0),
    )
    # All four replicas must differ pairwise (period-127 LFSR with
    # distinct salt -> distinct sub-sequences in 50 samples).
    n = out.shape[0]
    assert _pairs_distinct(out) == n * (n - 1) // 2


def test_prbslfsr_fold_in_outputs_still_binary():
    """LFSR fold_in must not break the +/-amplitude invariant."""
    out = _vmap_run(
        lambda _i: PRBSLFSR(
            sample_time=0.01, register_length=7, seed=1,
            amplitude=3.0, fold_in_batch_index=True, name="src",
        ),
        n_replicas=3,
        t_end=0.2,
        dummy_param=("src.amplitude", 3.0),
    )
    abs_vals = np.abs(out)
    np.testing.assert_allclose(abs_vals, 3.0, atol=1e-9)


def test_prbslfsr_fold_in_outside_vmap_is_noop():
    """LFSR fold_in outside any vmap -> bit-identical to default."""
    a = _simulate_source(
        PRBSLFSR(sample_time=0.01, register_length=7, seed=1, amplitude=1.0),
        t_end=0.1,
    )
    b = _simulate_source(
        PRBSLFSR(
            sample_time=0.01, register_length=7, seed=1, amplitude=1.0,
            fold_in_batch_index=True,
        ),
        t_end=0.1,
    )
    np.testing.assert_array_equal(a, b)
