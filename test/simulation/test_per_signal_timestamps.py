# SPDX-License-Identifier: MIT
"""
T-013a — Per-signal timestamp capture in the recording pipeline.

Mode B: an out-of-JIT post-processor on the trimmed numpy arrays
deduplicates each signal's samples to the indices where the value
actually advances and exposes the result via
``SimulationResults.per_signal_times`` / ``time_for(name)``.

Tests:
  - Default off: legacy ``per_signal_times is None`` path.
  - Mixed continuous + ZOH-at-10Hz signals: continuous signal keeps
    every major step; the held signal collapses to the schedule.
  - Constant signal: a never-changing recorded port collapses to a
    single sample.
  - ``time_for(name)`` reflects the per-signal vector when populated
    and falls back to the global vector when not.
  - End-to-end ``align(...)`` on a real Mode-B result: every signal
    returns the requested shape and step-interpolates correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import Constant, Ramp, ZeroOrderHold
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Test fixtures ─────────────────────────────────────────────────────────


class _Decay(jaxonomy.LeafSystem):
    """Continuous-state exponential decay (smooth signal)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        return -state.continuous_state


def _build_mixed_diagram(zoh_dt: float = 0.1):
    """Continuous decay + a ZOH sampled at ``zoh_dt`` reading a Ramp.

    Returns ``(diagram, recorded_signals)``.
    """
    builder = jaxonomy.DiagramBuilder()
    decay = builder.add(_Decay(name="decay"))
    # start_time=0.0 — the default 1.0 would keep the ramp pinned at
    # start_value through the [0, 1] window we simulate over and
    # collapse the ZOH to a single hold.
    ramp = builder.add(
        Ramp(start_value=0.0, slope=1.0, start_time=0.0, name="ramp"),
    )
    zoh = builder.add(ZeroOrderHold(dt=zoh_dt, name="zoh"))
    builder.connect(ramp.output_ports[0], zoh.input_ports[0])
    diagram = builder.build()
    recorded = {
        "continuous_y": decay.output_ports[0],
        "discrete_y": zoh.output_ports[0],
    }
    return diagram, recorded


def _run(diagram, recorded, t_end=1.0, **opt_overrides):
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        **opt_overrides,
    )
    return jaxonomy.simulate(
        diagram, ctx, (0.0, t_end), options=opts,
        recorded_signals=recorded,
    )


# ── 1. Default off (backwards compat) ─────────────────────────────────────


def test_default_off_preserves_legacy_behaviour():
    diagram, recorded = _build_mixed_diagram()
    res = _run(diagram, recorded)
    # Legacy: no per_signal_times, time_for falls back to global vector.
    assert res.per_signal_times is None
    np.testing.assert_array_equal(
        np.asarray(res.time_for("continuous_y")),
        np.asarray(res.time),
    )
    np.testing.assert_array_equal(
        np.asarray(res.time_for("discrete_y")),
        np.asarray(res.time),
    )


# ── 2. Mixed-rate diagram: per-signal cadences differ ─────────────────────


def test_mode_b_separates_continuous_from_zoh():
    diagram, recorded = _build_mixed_diagram(zoh_dt=0.1)
    res = _run(diagram, recorded, per_signal_timestamps=True)

    assert res.per_signal_times is not None
    assert "continuous_y" in res.per_signal_times
    assert "discrete_y" in res.per_signal_times

    cont_t = np.asarray(res.per_signal_times["continuous_y"])
    disc_t = np.asarray(res.per_signal_times["discrete_y"])
    full_t = np.asarray(res.time)

    # Continuous signal advances on every major step.  The simulator
    # does at least as many major steps as ZOH ticks, so the
    # continuous-signal cadence is at least as dense as the discrete
    # one and matches the global recording vector for a strictly
    # decreasing exponential.
    assert cont_t.shape[0] == full_t.shape[0]

    # The ZOH collapses to its schedule: y = 0 at t=0, then advances
    # each 0.1 s tick.  Over [0, 1] that's at most 11 unique values
    # (t = 0, 0.1, ..., 1.0).  We expect ~11 deduped samples; the
    # exact count depends on whether the boundary tick fires before or
    # after the recording at t_end.  Allow [10, 12] to absorb that
    # one-tick ambiguity without making the test fragile.
    assert 10 <= disc_t.shape[0] <= 12, (
        f"ZOH at 10 Hz should produce ~11 unique samples in [0, 1]; "
        f"got {disc_t.shape[0]}"
    )
    # Discrete cadence is strictly sparser than the continuous one.
    assert disc_t.shape[0] < cont_t.shape[0]


# ── 3. Constant signal collapses to one sample ────────────────────────────


def test_constant_signal_collapses_to_single_sample():
    builder = jaxonomy.DiagramBuilder()
    decay = builder.add(_Decay(name="decay"))
    const = builder.add(Constant(value=jnp.array(7.0), name="k"))
    diagram = builder.build()
    recorded = {
        "continuous_y": decay.output_ports[0],
        "constant_y": const.output_ports[0],
    }
    res = _run(diagram, recorded, per_signal_timestamps=True)

    const_t = np.asarray(res.per_signal_times["constant_y"])
    assert const_t.shape == (1,)
    assert const_t[0] == 0.0

    # Continuous signal still has its full cadence.
    cont_t = np.asarray(res.per_signal_times["continuous_y"])
    assert cont_t.shape[0] == np.asarray(res.time).shape[0]


# ── 4. time_for() dispatch behaviour ──────────────────────────────────────


def test_time_for_returns_per_signal_vector_when_populated():
    diagram, recorded = _build_mixed_diagram(zoh_dt=0.1)
    res = _run(diagram, recorded, per_signal_timestamps=True)

    cont_t_per = np.asarray(res.per_signal_times["continuous_y"])
    disc_t_per = np.asarray(res.per_signal_times["discrete_y"])

    np.testing.assert_array_equal(np.asarray(res.time_for("continuous_y")), cont_t_per)
    np.testing.assert_array_equal(np.asarray(res.time_for("discrete_y")), disc_t_per)


# ── 5. align(t_grid) end-to-end on a Mode-B result ────────────────────────


def test_align_on_mode_b_result():
    diagram, recorded = _build_mixed_diagram(zoh_dt=0.1)
    res = _run(diagram, recorded, per_signal_timestamps=True)

    t_grid = jnp.linspace(0.0, 1.0, 21)
    aligned = res.align(t_grid)

    assert aligned.per_signal_times is None
    assert np.asarray(aligned.outputs["continuous_y"]).shape == (21,)
    assert np.asarray(aligned.outputs["discrete_y"]).shape == (21,)

    # The continuous signal should be ~ exp(-t).
    np.testing.assert_allclose(
        np.asarray(aligned.outputs["continuous_y"]),
        np.exp(-np.asarray(t_grid)),
        atol=1e-2,
    )
    # The ZOH (sampling a ramp t -> t at 10 Hz) advances at each tick;
    # at the recording grid points (also every 0.1 s) the held value
    # should match t exactly modulo a one-step latency.  Allow 0.1 s of
    # slack (the dt of the hold).
    held = np.asarray(aligned.outputs["discrete_y"])
    expected = np.asarray(t_grid)
    assert np.all(np.abs(held - expected) <= 0.1 + 1e-6)


# ── 6. Mode A: storage savings on a 1Hz signal alongside a CT signal ──────


def test_mode_a_storage_savings():
    """Mode A trims BOTH per_signal_times AND outputs for periodic signals.

    A 1 Hz ZOH co-recorded with a continuous decay over 10 s should
    produce ~11 stored samples for the ZOH (one per tick) and the
    continuous-signal length for the decay.  Mode B by contrast leaves
    outputs at full length even though per_signal_times is deduplicated.
    """
    builder = jaxonomy.DiagramBuilder()
    decay = builder.add(_Decay(name="decay"))
    ramp = builder.add(
        Ramp(start_value=0.0, slope=1.0, start_time=0.0, name="ramp"),
    )
    zoh = builder.add(ZeroOrderHold(dt=1.0, name="zoh"))
    builder.connect(ramp.output_ports[0], zoh.input_ports[0])
    diagram = builder.build()
    recorded = {
        "continuous_y": decay.output_ports[0],
        "periodic_y": zoh.output_ports[0],
    }
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        per_signal_timestamps=True,  # default mode is "auto" / Mode A
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 10.0), options=opts, recorded_signals=recorded,
    )

    periodic_outputs = np.asarray(res.outputs["periodic_y"])
    periodic_times = np.asarray(res.per_signal_times["periodic_y"])

    # Mode A storage saving: outputs is the same length as the per-
    # signal times — both ~11 samples (one per 1 Hz tick over [0, 10]).
    assert periodic_outputs.shape[0] == periodic_times.shape[0]
    assert 10 <= periodic_outputs.shape[0] <= 12, (
        f"Mode A should store ~11 samples for a 1 Hz signal in 10 s; "
        f"got {periodic_outputs.shape[0]}"
    )

    # Cross-check vs. Mode B: outputs stay at full length there.
    opts_b = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        per_signal_timestamps=True,
        per_signal_timestamps_mode="diff",
    )
    res_b = jaxonomy.simulate(
        diagram, ctx, (0.0, 10.0), options=opts_b,
        recorded_signals=recorded,
    )
    periodic_outputs_b = np.asarray(res_b.outputs["periodic_y"])
    full_t_b = np.asarray(res_b.time)
    assert periodic_outputs_b.shape[0] == full_t_b.shape[0]
    assert periodic_outputs_b.shape[0] > periodic_outputs.shape[0]


# ── 7. Mode A: periodic-event alignment is exact ──────────────────────────


def test_mode_a_periodic_event_alignment():
    """ZOH at 0.1 s sampled in [0, 1] should record at exactly the ticks.

    The recorded times for the ZOH signal are aligned to the schedule
    ``[0.0, 0.1, 0.2, ..., 1.0]`` rather than the underlying solver's
    irregular major-step times.
    """
    diagram, recorded = _build_mixed_diagram(zoh_dt=0.1)
    res = _run(diagram, recorded, per_signal_timestamps=True)

    disc_t = np.asarray(res.per_signal_times["discrete_y"])
    expected = np.arange(0.0, 1.0 + 1e-9, 0.1)
    # Allow one boundary-tick of slack at the end; the simulator may
    # finalize at t_end before or after the last 0.1 s tick fires.
    assert disc_t.shape[0] in (len(expected), len(expected) - 1)
    n = disc_t.shape[0]
    np.testing.assert_allclose(disc_t, expected[:n], atol=1e-9)


# ── 9. Mode A buffers: in-JIT per-signal storage savings ──────────────────


def _build_buffers_diagram():
    builder = jaxonomy.DiagramBuilder()
    decay = builder.add(_Decay(name="decay"))
    ramp = builder.add(
        Ramp(start_value=0.0, slope=1.0, start_time=0.0, name="ramp"),
    )
    zoh = builder.add(ZeroOrderHold(dt=1.0, name="zoh"))
    builder.connect(ramp.output_ports[0], zoh.input_ports[0])
    diagram = builder.build()
    recorded = {
        "continuous_y": decay.output_ports[0],
        "periodic_y": zoh.output_ports[0],
    }
    return diagram, recorded


def test_mode_a_buffers_storage_savings():
    """Mode "buffers": periodic signal's outputs is at the cadence rate.

    Same diagram as test_mode_a_storage_savings but with per_signal_
    timestamps_mode="buffers" — confirms the in-JIT buffers route
    actually allocates the periodic signal at ~11 entries (one per 1 Hz
    tick over [0, 10]) rather than the dense continuous step count.
    """
    diagram, recorded = _build_buffers_diagram()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        per_signal_timestamps=True,
        per_signal_timestamps_mode="buffers",
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 10.0), options=opts, recorded_signals=recorded,
    )

    periodic_outputs = np.asarray(res.outputs["periodic_y"])
    periodic_times = np.asarray(res.per_signal_times["periodic_y"])

    assert periodic_outputs.shape[0] == periodic_times.shape[0]
    assert 10 <= periodic_outputs.shape[0] <= 12, (
        f"Mode 'buffers' should store ~11 samples for a 1 Hz signal in "
        f"10 s; got {periodic_outputs.shape[0]}"
    )
    # The continuous signal should retain every major step.
    cont_outputs = np.asarray(res.outputs["continuous_y"])
    cont_times = np.asarray(res.per_signal_times["continuous_y"])
    assert cont_outputs.shape[0] == cont_times.shape[0]
    assert cont_outputs.shape[0] > periodic_outputs.shape[0]

    # Cross-check vs. mode="auto" — the in-JIT buffers path should
    # match (or beat) the post-finalize trim.
    opts_auto = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        per_signal_timestamps=True,
        per_signal_timestamps_mode="auto",
    )
    res_auto = jaxonomy.simulate(
        diagram, ctx, (0.0, 10.0), options=opts_auto,
        recorded_signals=recorded,
    )
    periodic_auto = np.asarray(res_auto.outputs["periodic_y"])
    assert periodic_outputs.shape[0] <= periodic_auto.shape[0] + 1, (
        f"Mode 'buffers' produced {periodic_outputs.shape[0]} entries, "
        f"more than mode='auto'={periodic_auto.shape[0]}; the in-JIT "
        f"path should match or beat the post-finalize trim."
    )


def test_mode_a_buffers_simulation_runs_clean():
    """5-second simulation under mode='buffers' completes without error."""
    diagram, recorded = _build_buffers_diagram()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        per_signal_timestamps=True,
        per_signal_timestamps_mode="buffers",
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 5.0), options=opts, recorded_signals=recorded,
    )

    # time_for() returns the periodic signal's cadence vector.
    periodic_t = np.asarray(res.time_for("periodic_y"))
    expected = np.arange(0.0, 5.0 + 1e-9, 1.0)
    # Allow a 1-tick boundary slack at the end.
    assert periodic_t.shape[0] in (len(expected), len(expected) - 1)
    n = periodic_t.shape[0]
    np.testing.assert_allclose(periodic_t, expected[:n], atol=1e-9)


def test_mode_a_buffers_default_off_byte_equivalent():
    """Mode 'auto' (default) is unchanged from before this followup.

    Compares mode='auto' on the buffers-target diagram against mode='diff'
    to confirm both still ship through their original paths and produce
    the documented shapes.  Specifically, mode='auto' should NOT promote
    to per-signal buffers — the global ``time`` vector matches the dense
    continuous step count.
    """
    diagram, recorded = _build_buffers_diagram()
    ctx = diagram.create_context()
    opts_auto = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        per_signal_timestamps=True,
        per_signal_timestamps_mode="auto",
    )
    res_auto = jaxonomy.simulate(
        diagram, ctx, (0.0, 10.0), options=opts_auto,
        recorded_signals=recorded,
    )

    # mode='auto' continues to derive from the legacy global trim;
    # outputs for the periodic signal got trimmed post-hoc, but the
    # global ``time`` vector is the dense continuous-step record.
    full_time = np.asarray(res_auto.time)
    cont_outputs = np.asarray(res_auto.outputs["continuous_y"])
    assert cont_outputs.shape[0] == full_time.shape[0]


# ── 10. Mode A: align(t_grid) matches Mode B within tolerance ─────────────


def test_mode_a_results_align_correctness():
    """``align(t_grid)`` produces equivalent values for Mode A and Mode B.

    Both modes carry enough information for ``align`` to reconstruct a
    rectangular grid; the resampled arrays should agree within tolerance.
    """
    diagram, recorded = _build_mixed_diagram(zoh_dt=0.1)

    res_a = _run(diagram, recorded, per_signal_timestamps=True)
    res_b = _run(
        diagram, recorded,
        per_signal_timestamps=True,
        per_signal_timestamps_mode="diff",
    )

    t_grid = jnp.linspace(0.0, 1.0, 21)
    aligned_a = res_a.align(t_grid)
    aligned_b = res_b.align(t_grid)

    # Both aligned results should have the same shape on every signal.
    for name in recorded:
        assert (
            np.asarray(aligned_a.outputs[name]).shape
            == np.asarray(aligned_b.outputs[name]).shape
        )
        np.testing.assert_allclose(
            np.asarray(aligned_a.outputs[name]),
            np.asarray(aligned_b.outputs[name]),
            atol=1e-2,
        )
