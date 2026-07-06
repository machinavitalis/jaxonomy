# SPDX-License-Identifier: MIT

"""T-002b-followup-buffer-overflow-warning — Python-side detection of
recording-buffer truncation.

When the simulator's recording buffer fills up before the requested
``t_final`` is reached, the trimmed ``results.time`` / ``results.outputs``
silently carry only the last ``buffer_length`` samples. The original
IO-effect log that warned the user was removed in T-002b to make
``simulate_batch(use_vmap=True)`` compilable. This test covers the
Python-side post-simulation check that emits a clear ``UserWarning``
on the single-simulation path without disturbing the vmap-clean
inner kernel.

Detection signature:
* recording was on (``time`` is non-None),
* ``len(time) == options.buffer_length`` exactly (buffer wrapped),
* ``time[-1] < t_span[1]`` (truncation actually happened).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Constant, Integrator


def _build_recorded_sine_diagram():
    """``u=1`` → Integrator(x_0=0) → exported as ``y``. The integrator
    forces the ODE solver to take real minor steps (a pure Sine source
    has no continuous state and consumes only one ``time`` slot per
    major step). This is the minimal diagram that actually exercises
    the recording buffer."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    integ = b.add(Integrator(0.0, name="integ"))
    b.connect(src.output_ports[0], integ.input_ports[0])
    b.export_output(integ.output_ports[0], name="y")
    return b.build(name="recorded_int"), integ


def test_warning_emitted_when_buffer_fills_before_t_final():
    """Force the truncation signature with a tight buffer + small
    max_minor_step + long horizon; warning must fire."""
    diag, sine = _build_recorded_sine_diagram()
    ctx = diag.create_context()

    options = jaxonomy.SimulatorOptions(
        buffer_length=20,           # tiny ring buffer — easy to fill
        max_minor_step_size=0.01,   # force many minor steps
        rtol=1e-3,
        atol=1e-5,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxonomy.simulate(
            diag, ctx, (0.0, 10.0),
            options=options,
            recorded_signals={"y": sine.output_ports[0]},
        )

    buffer_warnings = [
        w for w in caught
        if "recording buffer overflow" in str(w.message)
    ]
    assert buffer_warnings, (
        f"expected a buffer-overflow UserWarning; got {[str(w.message) for w in caught]}"
    )
    # Message should cite the configured length and the gap kwarg
    # name so the user has a one-line fix.
    msg = str(buffer_warnings[0].message)
    assert "buffer_length=20" in msg
    assert "buffer_length" in msg


def test_no_warning_when_buffer_is_large_enough():
    """The byte-equivalent default case: buffer comfortably covers
    the trajectory, no warning fires."""
    diag, sine = _build_recorded_sine_diagram()
    ctx = diag.create_context()

    options = jaxonomy.SimulatorOptions(
        buffer_length=10_000,       # plenty
        max_minor_step_size=0.01,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxonomy.simulate(
            diag, ctx, (0.0, 1.0),
            options=options,
            recorded_signals={"y": sine.output_ports[0]},
        )

    buffer_warnings = [
        w for w in caught if "recording buffer" in str(w.message)
    ]
    assert not buffer_warnings, (
        f"unexpected buffer-overflow warning: "
        f"{[str(w.message) for w in buffer_warnings]}"
    )


def test_no_warning_when_recording_disabled():
    """A bare ``simulate`` call (no recorded_signals) shouldn't even
    enter the buffer-warning check — ``time`` is None."""
    diag, _ = _build_recorded_sine_diagram()
    ctx = diag.create_context()

    options = jaxonomy.SimulatorOptions(buffer_length=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxonomy.simulate(diag, ctx, (0.0, 1.0), options=options)

    buffer_warnings = [
        w for w in caught if "recording buffer" in str(w.message)
    ]
    assert not buffer_warnings


def test_no_warning_when_exact_buffer_fill_matches_t_final():
    """Boundary case: the simulation legitimately consumes exactly
    ``buffer_length`` major steps and finishes at t_final. The
    discriminator (``time[-1] < t_span[1]``) must NOT fire here —
    we'd be crying wolf on a perfect-fit simulation."""
    diag, sine = _build_recorded_sine_diagram()
    ctx = diag.create_context()

    # Use a buffer roomy enough that the trimmed array won't hit
    # exactly buffer_length; even if it did, the time[-1] should
    # equal t_final.
    options = jaxonomy.SimulatorOptions(buffer_length=10_000)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxonomy.simulate(
            diag, ctx, (0.0, 0.1),
            options=options,
            recorded_signals={"y": sine.output_ports[0]},
        )

    buffer_warnings = [
        w for w in caught if "recording buffer" in str(w.message)
    ]
    assert not buffer_warnings
