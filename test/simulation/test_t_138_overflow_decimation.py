# SPDX-License-Identifier: MIT

"""T-138: recording-buffer overflow degrades to uniform decimation.

When the fixed recording buffer fills, the JAX backend compacts the
even-position samples into the lower half and doubles the keep-stride,
so the recorded trajectory always spans from ``t0`` at reduced
resolution — instead of the pre-T-138 ring-wrap that silently kept only
the tail. These tests pin the correctness properties that matter:

* ``results.time[0] == t0`` regardless of horizon,
* strictly monotonic recorded time,
* exact timestamp/value alignment across compaction boundaries,
* bounded memory (``len(time) <= buffer_length``),
* byte-identical behaviour when the buffer never fills,
* vmap-batch safety (the compaction is a pure lax.cond — no IO effects).
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Constant, Gain, Integrator
from jaxonomy.simulation import SimulatorOptions

pytestmark = pytest.mark.minimal


def _ramp_diagram():
    """Constant(1) → Integrator: y(t) = t exactly, so alignment between
    recorded times and values is checkable to solver precision."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    integ = b.add(Integrator(0.0, name="integ"))
    b.connect(src.output_ports[0], integ.input_ports[0])
    return b.build(name="ramp"), integ


def _simulate(buffer_length, t_final=10.0, max_minor=0.01):
    diagram, integ = _ramp_diagram()
    ctx = diagram.create_context()
    options = SimulatorOptions(
        buffer_length=buffer_length,
        max_minor_step_size=max_minor,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, t_final),
            options=options,
            recorded_signals={"y": integ.output_ports[0]},
        )
    reduced = [w for w in caught if "reduced resolution" in str(w.message)]
    return results, reduced


def test_overflow_keeps_head_spans_trajectory():
    results, reduced = _simulate(buffer_length=64)
    t = np.asarray(results.time)
    assert t[0] == 0.0, "decimation must preserve the first sample (t0)"
    assert len(t) <= 64, "memory must stay bounded by buffer_length"
    assert np.all(np.diff(t) > 0), "recorded time must stay strictly monotonic"
    # The tail may stop up to one stride short of t_final, but must be
    # nowhere near the halfway point (which a head-drop would produce).
    assert t[-1] > 9.0
    assert len(reduced) == 1


def test_values_stay_aligned_across_compactions():
    """y(t) = t exactly — any slot misalignment between the compacted
    time and value buffers would show up as |y - t| jumping."""
    results, _ = _simulate(buffer_length=64)
    t = np.asarray(results.time)
    y = np.asarray(results.outputs["y"])
    assert y.shape[0] == t.shape[0]
    np.testing.assert_allclose(y, t, atol=1e-9)


def test_multiple_compactions_still_uniform():
    """A tiny buffer against a dense recording forces several stride
    doublings; coverage must remain uniform (no head/tail crowding)."""
    results, reduced = _simulate(buffer_length=32, t_final=20.0)
    t = np.asarray(results.time)
    assert t[0] == 0.0
    assert len(t) <= 32
    assert np.all(np.diff(t) > 0)
    assert t[-1] > 18.0
    # Uniformity: with ~2000 accepted steps decimated into ≤32 slots the
    # largest gap should stay within a small multiple of the mean gap.
    gaps = np.diff(t)
    assert gaps.max() < 4.0 * gaps.mean()
    assert len(reduced) == 1
    # Values still aligned after 6+ compactions.
    y = np.asarray(results.outputs["y"])
    np.testing.assert_allclose(y, t, atol=1e-9)


def test_no_overflow_is_unaffected():
    """Ample buffer: every accepted sample is recorded and no warning
    fires — the pre-overflow hot path is unchanged."""
    results, reduced = _simulate(buffer_length=8192)
    t = np.asarray(results.time)
    assert t[0] == 0.0
    assert len(t) > 900, "expected a dense (undedecimated) recording"
    assert not reduced


def test_vmap_batch_with_overflow():
    """simulate_batch(use_vmap=True) with an overflowing buffer: the
    compaction is a pure lax.cond, so the vmapped kernel must compile
    and every batch row must produce a well-formed trajectory."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    gain = b.add(Gain(1.0, name="gain"))
    integ = b.add(Integrator(0.0, name="integ"))
    b.connect(src.output_ports[0], gain.input_ports[0])
    b.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = b.build(name="ramp_gain")

    gains = jnp.array([1.0, 2.0, 3.0])
    options = SimulatorOptions(
        math_backend="jax",
        max_major_steps=100,
        buffer_length=64,
        max_minor_step_size=0.01,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # CPU small-batch vmap advisory
        batch = jaxonomy.simulate_batch(
            diagram,
            (0.0, 10.0),
            {"gain.gain": gains},
            options=options,
            recorded_signals={"y": diagram["integ"].output_ports[0]},
            use_vmap=True,
        )

    y = np.asarray(batch.outputs["y"])
    assert y.shape[0] == 3
    assert np.all(np.isfinite(y))
    # y(t) = g * t: check the terminal value of each row against its gain
    # on the row's final recorded time.
    t = np.asarray(batch.time)
    assert t.ndim in (1, 2)
    t_last = t[-1] if t.ndim == 1 else t[:, -1]
    expected = np.asarray(gains) * t_last
    np.testing.assert_allclose(y[:, -1], expected, rtol=1e-6)
