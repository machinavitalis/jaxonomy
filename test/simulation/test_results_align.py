# SPDX-License-Identifier: MIT
"""
T-013 — SimulationResults.align / time_for / per_signal_times tests.

Covers:

  - Backwards compatibility: if simulate() produces a result with
    ``per_signal_times=None`` (the current recording pipeline), every
    signal's ``time_for`` returns the global ``time`` vector.
  - ``align`` on a legacy result produces a new result sampled at the
    user's target grid.
  - A manually-constructed result with per-signal timestamps has
    ``time_for`` dispatch per signal; ``align`` correctly merges the
    different native rates onto one grid.
  - Out-of-range alignment raises clearly.
  - Unknown signal name raises.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation import SimulationResults
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


class _Decay(jaxonomy.LeafSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        return -state.continuous_state


def _run_decay():
    sys = _Decay()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", save_time_series=True)
    return jaxonomy.simulate(
        sys, ctx, (0.0, 2.0), options=opts,
        recorded_signals={"x": sys.output_ports[0]},
    )


# ── Backwards compatibility ───────────────────────────────────────────────


def test_time_for_falls_back_to_global_vector():
    res = _run_decay()
    # Current recording pipeline produces per_signal_times=None.
    assert res.per_signal_times is None
    np.testing.assert_array_equal(
        np.asarray(res.time_for("x")),
        np.asarray(res.time),
    )


def test_align_on_legacy_result():
    res = _run_decay()
    t_uniform = jnp.linspace(0.0, 2.0, 21)
    aligned = res.align(t_uniform)
    assert aligned.per_signal_times is None
    np.testing.assert_array_equal(np.asarray(aligned.time), np.asarray(t_uniform))
    assert set(aligned.outputs) == {"x"}
    # Sanity-check the resampled values vs exp(-t).
    np.testing.assert_allclose(
        np.asarray(aligned.outputs["x"]),
        np.exp(-np.asarray(t_uniform)),
        atol=1e-2,
    )


# ── Per-signal timestamps (manually constructed result) ───────────────────


def test_time_for_with_per_signal_times():
    """A user constructs a result with two signals captured at different
    native rates; time_for returns each signal's own timestamps."""
    t_fast = jnp.linspace(0.0, 1.0, 101)   # 100 Hz
    t_slow = jnp.linspace(0.0, 1.0, 11)    # 10 Hz
    fast_signal = jnp.sin(2 * jnp.pi * t_fast)
    slow_signal = jnp.cos(2 * jnp.pi * t_slow)

    res = SimulationResults(
        context=None,
        time=t_fast,  # legacy "global" fallback
        outputs={"fast": fast_signal, "slow": slow_signal},
        per_signal_times={"fast": t_fast, "slow": t_slow},
    )

    np.testing.assert_array_equal(
        np.asarray(res.time_for("fast")), np.asarray(t_fast),
    )
    np.testing.assert_array_equal(
        np.asarray(res.time_for("slow")), np.asarray(t_slow),
    )


def test_align_merges_different_rates():
    """align() resamples two signals recorded at different native rates
    onto a single common grid."""
    t_fast = jnp.linspace(0.0, 1.0, 101)
    t_slow = jnp.linspace(0.0, 1.0, 11)
    fast_signal = 2.0 * t_fast      # y = 2 t
    slow_signal = 3.0 * t_slow      # y = 3 t

    res = SimulationResults(
        context=None,
        time=t_fast,
        outputs={"fast": fast_signal, "slow": slow_signal},
        per_signal_times={"fast": t_fast, "slow": t_slow},
    )

    t_common = jnp.linspace(0.1, 0.9, 9)
    aligned = res.align(t_common)

    np.testing.assert_allclose(
        np.asarray(aligned.outputs["fast"]),
        2.0 * np.asarray(t_common),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(aligned.outputs["slow"]),
        3.0 * np.asarray(t_common),
        atol=1e-6,
    )
    assert aligned.per_signal_times is None


# ── Error paths ───────────────────────────────────────────────────────────


def test_align_out_of_range_raises():
    res = _run_decay()
    with pytest.raises(ValueError, match="out of range"):
        res.align(jnp.array([5.0]))


def test_align_unknown_signal_raises():
    res = _run_decay()
    with pytest.raises(ValueError, match="unknown signal"):
        res.align(jnp.array([0.5]), signals=["nope"])


def test_align_no_outputs_raises():
    res = SimulationResults(context=None, time=None, outputs=None)
    with pytest.raises(ValueError, match="no recorded signals"):
        res.align(jnp.array([0.5]))
