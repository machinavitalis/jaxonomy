# SPDX-License-Identifier: MIT

"""Recording-buffer-full diagnostics (T-002b-followup, updated by T-138).

When the simulator's recording buffer fills before the requested
``t_final``, the JAX backend now degrades to uniform decimation (T-138):
the recorded time-series keeps every Nth sample spanning the whole
trajectory instead of silently keeping only the tail. The Python-side
post-simulation check (running once after the JIT'd kernel returns, so
it does not disturb the vmap-clean inner kernel) emits a ``UserWarning``
saying the results were recorded at reduced resolution.

Detection signature: the backend's ``record_stride`` ends > 1 iff at
least one buffer compaction ran.
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
    """Force an overflow with a tight buffer + small max_minor_step +
    long horizon; the reduced-resolution warning must fire and the
    recording must still start at t0 (T-138)."""
    diag, sine = _build_recorded_sine_diagram()
    ctx = diag.create_context()

    options = jaxonomy.SimulatorOptions(
        buffer_length=20,           # tiny buffer — easy to fill
        max_minor_step_size=0.01,   # force many minor steps
        rtol=1e-3,
        atol=1e-5,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = jaxonomy.simulate(
            diag, ctx, (0.0, 10.0),
            options=options,
            recorded_signals={"y": sine.output_ports[0]},
        )

    buffer_warnings = [
        w for w in caught
        if "reduced resolution" in str(w.message)
    ]
    assert buffer_warnings, (
        f"expected a reduced-resolution UserWarning; got {[str(w.message) for w in caught]}"
    )
    # Message should cite the configured length and the kwarg name so
    # the user has a one-line fix.
    msg = str(buffer_warnings[0].message)
    assert "buffer_length=20" in msg
    assert "buffer_length" in msg
    # T-138 head guarantee: decimation, not tail-keeping.
    assert float(results.time[0]) == 0.0


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


# ---------------------------------------------------------------------------
# Follow-up: ``simulate_batch`` kernel / vmap paths materialize results
# without going through ``simulate``'s post-JIT check, so they need their own
# host-side overflow warning (``maybe_warn_recording_truncation``).
# ---------------------------------------------------------------------------

def _build_gain_integrator_diagram():
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    from jaxonomy.library import Gain
    gain = b.add(Gain(gain=1.0, name="gain"))
    integ = b.add(Integrator(0.0, name="integ"))
    b.connect(src.output_ports[0], gain.input_ports[0])
    b.connect(gain.output_ports[0], integ.input_ports[0])
    return b.build(name="batch_overflow")


def _run_batch(buffer_length, t_final, use_vmap):
    from jaxonomy.simulation.batch import simulate_batch

    diagram = _build_gain_integrator_diagram()
    options = jaxonomy.SimulatorOptions(
        math_backend="jax",
        max_major_steps=2000,
        buffer_length=buffer_length,
        max_minor_step_size=0.01,
        ode_solver_method="rk4",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulate_batch(
            diagram,
            t_span=(0.0, t_final),
            param_batches={"gain.gain": jnp.array([0.5, 1.0, 1.5])},
            options=options,
            recorded_signals={"y": diagram["integ"].output_ports[0]},
            use_vmap=use_vmap,
        )
    return [w for w in caught if "recording buffer" in str(w.message)]


@pytest.mark.parametrize("use_vmap", [False, True])
def test_batch_warns_once_on_buffer_overflow(use_vmap):
    """Tiny buffer + long horizon on the batch kernel/vmap paths: exactly
    one reduced-resolution warning fires for the whole batch, naming the
    configured buffer_length."""
    buffer_warnings = _run_batch(buffer_length=20, t_final=10.0, use_vmap=use_vmap)
    assert len(buffer_warnings) == 1, (
        f"expected exactly one buffer warning, got "
        f"{[str(w.message) for w in buffer_warnings]}"
    )
    msg = str(buffer_warnings[0].message)
    assert "buffer_length=20" in msg
    assert "reduced resolution" in msg


@pytest.mark.parametrize("use_vmap", [False, True])
def test_batch_silent_when_buffer_large_enough(use_vmap):
    """No-overflow batch runs stay silent."""
    buffer_warnings = _run_batch(
        buffer_length=10_000, t_final=1.0, use_vmap=use_vmap
    )
    assert not buffer_warnings, (
        f"unexpected buffer warning: {[str(w.message) for w in buffer_warnings]}"
    )


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
