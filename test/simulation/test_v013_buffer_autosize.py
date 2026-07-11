# SPDX-License-Identifier: MIT

"""V-013: recording ring-buffer auto-sizing regression pin.

CHANGELOG (3.0.0): `SimulatorOptions.buffer_length` auto-sizes to
`max(max_major_steps, 2048)` — the old hardcoded default of 1000 silently
overflowed the ring buffer on fine-grained recordings and returned a
truncated tail (`results.time` starting mid-trajectory).

The positive case here records ~1.8k minor steps (between the old 1000
default, which would truncate, and the 2048 floor, which must not) and
asserts the recording is complete from t=0 with no overflow warning. The
negative case pins that an explicit small `buffer_length` is still honoured
byte-for-byte: it truncates and warns.
"""

import warnings

import pytest

import jaxonomy
from jaxonomy.library import Integrator, Sine
from jaxonomy.simulation import SimulatorOptions

pytestmark = pytest.mark.minimal


def _fast_oscillator():
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Sine(frequency=20.0))
    integ = builder.add(Integrator(initial_state=0.0))
    builder.connect(src.output_ports[0], integ.input_ports[0])
    diagram = builder.build()
    return diagram, integ


def _simulate(options):
    diagram, integ = _fast_oscillator()
    ctx = diagram.create_context()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = jaxonomy.simulate(
            diagram,
            ctx,
            (0.0, 10.0),
            options=options,
            recorded_signals={"x": integ.output_ports[0]},
        )
    buffer_warns = [
        w for w in caught
        if "overflow" in str(w.message) or "reduced resolution" in str(w.message)
    ]
    return results, buffer_warns


def test_default_buffer_holds_more_than_old_1000_sample_default():
    """~1.8k accepted minor steps (empirically 1849 on 2026-07-09): the old
    1000-sample default buffer would truncate; the 2048 floor must not."""
    results, overflow = _simulate(SimulatorOptions(rtol=1e-10, atol=1e-12))
    assert len(results.time) > 1000, (
        "expected a recording dense enough to overflow the old 1000-sample "
        f"default (got {len(results.time)} samples) — if solver step-size "
        "heuristics changed, tighten rtol/atol to restore density"
    )
    assert float(results.time[0]) == 0.0, "recording was truncated from the front"
    assert not overflow, "buffer overflow warning fired under the auto-sized default"


def test_explicit_small_buffer_is_honoured_and_warns():
    """T-138 updated this contract: an overflowing buffer now degrades to
    uniform decimation (whole-trajectory coverage at reduced resolution)
    instead of keeping only the tail — so ``time[0]`` stays at t0 and the
    warning says "reduced resolution" rather than "overflow"."""
    results, buffer_warns = _simulate(
        SimulatorOptions(rtol=1e-10, atol=1e-12, buffer_length=64)
    )
    assert len(buffer_warns) == 1
    assert "reduced resolution" in str(buffer_warns[0].message)
    assert float(results.time[0]) == 0.0, (
        "T-138 decimation must preserve the trajectory head on overflow"
    )
    assert len(results.time) <= 64
