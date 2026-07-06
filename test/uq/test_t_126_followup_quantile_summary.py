# SPDX-License-Identifier: MIT

"""T-126 followup — quantile-based summary statistics for Monte Carlo output.

Covers :func:`jaxonomy.uq.quantile_summary`,
:func:`jaxonomy.uq.value_at_risk`, and
:func:`jaxonomy.uq.conditional_value_at_risk`:

1. ``quantile_summary`` of N=10000 N(0, 1) draws recovers q05/q50/q95 within
   statistical tolerance and reports ``mean``/``std``.
2. ``quantile_summary`` accepts a custom quantile tuple.
3. ``quantile_summary`` over a 2-D ``(N, d)`` sample returns ``(d,)`` arrays
   per field (broadcasting along axis 0).
4. ``value_at_risk(N(0, 1), alpha=0.05)`` matches the analytic ``-1.645``.
5. ``conditional_value_at_risk(N(0, 1), alpha=0.05)`` matches the analytic
   ``-2.063`` (mean of the left tail).
6. CVaR equals the mean of the left-tail samples (consistency check).
7. Validation: ``alpha`` outside ``(0, 1)`` raises; empty samples raise;
   non-1-D samples raise for VaR/CVaR; ``quantile_summary`` rejects bad
   levels and key-collisions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Normal,
    conditional_value_at_risk,
    quantile_summary,
    sample_parameters,
    value_at_risk,
)

skip_if_not_jax()
pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_samples(n: int = 10_000, seed: int = 0) -> jnp.ndarray:
    """N=10000 IID N(0, 1) draws via the public sampling API."""
    samples = sample_parameters(
        {"x": Normal(0.0, 1.0)}, n, jax.random.PRNGKey(seed),
    )
    return samples["x"]


# ---------------------------------------------------------------------------
# 1. quantile_summary recovers q05/q50/q95 of N(0, 1).
# ---------------------------------------------------------------------------

def test_quantile_summary_normal_recovers_analytic_quantiles():
    """Quantile summary of N=10000 N(0, 1) matches +/- 1.645 / 0 / +/- 1.645."""
    x = _normal_samples()
    out = quantile_summary(x)

    # Standard normal: q05 = -1.6449, q50 = 0, q95 = 1.6449.
    assert out["q05"] == pytest.approx(-1.6449, abs=0.08)
    assert out["q50"] == pytest.approx(0.0, abs=0.05)
    assert out["q95"] == pytest.approx(1.6449, abs=0.08)
    # Mean / std should be tight at N=10000.
    assert out["mean"] == pytest.approx(0.0, abs=0.05)
    assert out["std"] == pytest.approx(1.0, abs=0.05)


def test_quantile_summary_accepts_custom_quantiles():
    """User-supplied quantile tuple maps to qNN keys."""
    x = _normal_samples()
    out = quantile_summary(x, quantiles=(0.10, 0.25, 0.75, 0.90))

    assert set(out.keys()) == {"q10", "q25", "q75", "q90", "mean", "std"}
    # N(0, 1) deciles: +/-1.2816 at 10/90, +/-0.6745 at 25/75.
    assert out["q10"] == pytest.approx(-1.2816, abs=0.08)
    assert out["q25"] == pytest.approx(-0.6745, abs=0.06)
    assert out["q75"] == pytest.approx(0.6745, abs=0.06)
    assert out["q90"] == pytest.approx(1.2816, abs=0.08)


def test_quantile_summary_multi_output_broadcasts_per_signal():
    """A ``(N, d)`` sample returns ``(d,)`` arrays per field."""
    # Build a (N, 2) tensor: column 0 is N(0,1), column 1 is N(5,2).
    n = 4000
    key = jax.random.PRNGKey(1)
    k0, k1 = jax.random.split(key)
    col0 = jax.random.normal(k0, (n,))
    col1 = 5.0 + 2.0 * jax.random.normal(k1, (n,))
    x = jnp.stack([col0, col1], axis=1)

    out = quantile_summary(x)
    assert np.asarray(out["q50"]).shape == (2,)
    assert np.asarray(out["mean"]).shape == (2,)
    # Per-column medians: ~0 and ~5.
    assert np.asarray(out["q50"])[0] == pytest.approx(0.0, abs=0.1)
    assert np.asarray(out["q50"])[1] == pytest.approx(5.0, abs=0.2)
    # Per-column stds: ~1 and ~2.
    assert np.asarray(out["std"])[0] == pytest.approx(1.0, abs=0.1)
    assert np.asarray(out["std"])[1] == pytest.approx(2.0, abs=0.15)


# ---------------------------------------------------------------------------
# 2. value_at_risk on N(0, 1).
# ---------------------------------------------------------------------------

def test_value_at_risk_matches_normal_quantile():
    """VaR at 5% on N(0, 1) recovers -1.645 within tolerance."""
    x = _normal_samples()
    var05 = value_at_risk(x, alpha=0.05)
    assert var05 == pytest.approx(-1.6449, abs=0.08)


def test_value_at_risk_at_quartile():
    """VaR at 25% on N(0, 1) recovers -0.6745 (lower-quartile sanity check)."""
    x = _normal_samples()
    assert value_at_risk(x, alpha=0.25) == pytest.approx(-0.6745, abs=0.05)


# ---------------------------------------------------------------------------
# 3. conditional_value_at_risk (Expected Shortfall) on N(0, 1).
# ---------------------------------------------------------------------------

def test_conditional_value_at_risk_matches_analytic_left_tail():
    """CVaR at 5% on N(0, 1) recovers the analytic ~ -2.063."""
    # Analytic ES_alpha for N(0, 1) is -phi(Phi^{-1}(alpha)) / alpha:
    #   alpha = 0.05 -> -phi(-1.6449) / 0.05 ~= -0.10314 / 0.05 = -2.0628.
    x = _normal_samples()
    cvar05 = conditional_value_at_risk(x, alpha=0.05)
    assert cvar05 == pytest.approx(-2.0628, abs=0.10)


def test_conditional_value_at_risk_equals_mean_of_left_tail():
    """CVaR is the mean of the samples at-or-below VaR."""
    x = np.asarray(_normal_samples())
    var05 = value_at_risk(x, alpha=0.05)
    tail = x[x <= var05]
    assert tail.size > 0
    expected_cvar = float(tail.mean())
    cvar05 = conditional_value_at_risk(x, alpha=0.05)
    assert cvar05 == pytest.approx(expected_cvar, abs=1e-6)


def test_conditional_value_at_risk_is_more_pessimistic_than_var():
    """CVaR should be <= VaR (deeper into the left tail) for a non-degenerate
    sample."""
    x = _normal_samples()
    var05 = value_at_risk(x, alpha=0.05)
    cvar05 = conditional_value_at_risk(x, alpha=0.05)
    assert cvar05 <= var05 + 1e-9


# ---------------------------------------------------------------------------
# 4. Validation: alpha out of (0, 1) and bad inputs raise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_alpha", [-0.01, 0.0, 1.0, 1.5])
def test_value_at_risk_rejects_bad_alpha(bad_alpha):
    x = _normal_samples()
    with pytest.raises(ValueError, match="alpha"):
        value_at_risk(x, alpha=bad_alpha)


@pytest.mark.parametrize("bad_alpha", [-0.01, 0.0, 1.0, 1.5])
def test_conditional_value_at_risk_rejects_bad_alpha(bad_alpha):
    x = _normal_samples()
    with pytest.raises(ValueError, match="alpha"):
        conditional_value_at_risk(x, alpha=bad_alpha)


def test_value_at_risk_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty"):
        value_at_risk(jnp.array([]), alpha=0.05)


def test_conditional_value_at_risk_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty"):
        conditional_value_at_risk(jnp.array([]), alpha=0.05)


def test_value_at_risk_rejects_non_1d_samples():
    with pytest.raises(ValueError, match="1-D"):
        value_at_risk(jnp.zeros((4, 2)), alpha=0.05)


def test_conditional_value_at_risk_rejects_non_1d_samples():
    with pytest.raises(ValueError, match="1-D"):
        conditional_value_at_risk(jnp.zeros((4, 2)), alpha=0.05)


def test_quantile_summary_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty"):
        quantile_summary(jnp.array([]))


def test_quantile_summary_rejects_scalar_input():
    with pytest.raises(ValueError, match="at least one axis"):
        quantile_summary(jnp.asarray(1.0))


@pytest.mark.parametrize("bad_q", [-0.01, 0.0, 1.0, 1.5])
def test_quantile_summary_rejects_out_of_range_quantiles(bad_q):
    with pytest.raises(ValueError, match="\\(0, 1\\)"):
        quantile_summary(jnp.arange(10.0), quantiles=(bad_q,))


def test_quantile_summary_rejects_colliding_quantile_keys():
    # 0.05 and 0.0549 both round to "q05".
    with pytest.raises(ValueError, match="collide"):
        quantile_summary(jnp.arange(10.0), quantiles=(0.05, 0.0549))
