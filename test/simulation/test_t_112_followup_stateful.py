# SPDX-License-Identifier: MIT

"""Tests for T-112-followup-stateful-simulator.

Layered on top of phase 1, this followup adds three opt-in ergonomics to
:class:`FastRestartSimulator`:

* ``run(initial_state=...)`` overrides the per-run continuous state.
* ``reset(diagram=...)`` rebinds the simulator to a new diagram (forces
  a recompile).
* A ``UserWarning`` is emitted when a :meth:`run` invocation will force
  a JIT recompile (parameter / initial-state pytree shape changed).

Default-off byte-equivalence with phase 1 is verified by re-running a
phase-1-style call sequence and checking it produces the same outputs
as before the followup landed (covered indirectly by phase 1's own
tests still passing).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Gain, Integrator, Sine
from jaxonomy.simulation import FastRestartSimulator, SimulatorOptions

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_diagram(k: float = 1.0, x0: float = 0.0):
    """Sine -> Gain(k) -> Integrator(x0) (single recorded output)."""
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=k, name="gain"))
    integ = builder.add(Integrator(initial_state=x0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="t112fu_diag")


def _opts(max_major_steps: int = 200) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        max_major_steps=max_major_steps,
    )


# ---------------------------------------------------------------------------
# initial_state= kwarg on .run()
# ---------------------------------------------------------------------------


class TestInitialStateOverride:
    def test_default_initial_state_is_zero(self):
        """Without an override, the integrator starts at x=0 (the default)."""
        diag = _build_diagram(k=1.0, x0=0.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),  # very short — sample[0] == initial state
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        r = sim.run()
        # First sample should be the initial state (≈ 0).
        assert abs(float(r.outputs["y"][0])) < 1e-9

    def test_initial_state_overrides_default(self):
        """A run with initial_state=5.0 starts the integrator at x=5."""
        diag = _build_diagram(k=1.0, x0=0.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        # Burn the compile.
        sim.run()
        # Override: pass a single array (auto-wrapped for the
        # single-continuous-state-block case).
        r = sim.run(initial_state=jnp.float64(5.0))
        # First sample should be ≈ 5.0.
        assert abs(float(r.outputs["y"][0]) - 5.0) < 1e-6

    def test_initial_state_accepts_list_form(self):
        """Sequence form (one array per continuous-state block) works."""
        diag = _build_diagram(k=1.0, x0=0.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        sim.run()  # compile
        # The diagram has exactly one continuous-state block (the
        # integrator) — pass a 1-element list.
        r = sim.run(initial_state=[jnp.float64(3.0)])
        assert abs(float(r.outputs["y"][0]) - 3.0) < 1e-6

    def test_initial_state_with_parameters(self):
        """Combining initial_state= and parameters= works."""
        diag = _build_diagram(k=1.0, x0=0.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        sim.run()  # compile
        # The bare-baseline ``run()`` above leaves the params leaf as
        # a Python float; injecting a JAX scalar via ``parameters=``
        # changes the abstract context signature, so suppress the
        # (correct) one-time recompile warning here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = sim.run(
                parameters={"gain.gain": jnp.float64(2.0)},
                initial_state=jnp.float64(7.0),
            )
        # Initial state still wins for the first sample regardless of
        # gain (gain only affects ẋ, not x(0)).
        assert abs(float(r.outputs["y"][0]) - 7.0) < 1e-6

    def test_wrong_count_raises(self):
        """Passing the wrong number of arrays raises a clear error."""
        diag = _build_diagram(k=1.0, x0=0.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        sim.run()  # compile
        with pytest.raises(ValueError, match="expected 1"):
            sim.run(
                initial_state=[jnp.float64(1.0), jnp.float64(2.0)],
            )


# ---------------------------------------------------------------------------
# reset(diagram=...) — swap diagram between runs
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_no_args_drops_kernel(self):
        """``reset()`` without args is equivalent to ``close()``."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        assert sim._kernel is not None
        sim.reset()
        assert sim._kernel is None
        # Re-running rebuilds.
        sim.run(parameters={"gain.gain": jnp.float64(1.5)})
        assert sim._kernel is not None

    def test_reset_with_diagram_rebinds(self):
        """``reset(diagram=new)`` swaps the diagram; next run uses it."""
        diag1 = _build_diagram(k=1.0)
        sim = FastRestartSimulator(
            diag1,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag1["integ"].output_ports[0]},
        )
        r1 = sim.run()
        peak1 = float(jnp.max(jnp.abs(r1.outputs["y"])))

        # Build a structurally-equivalent diagram with a much bigger
        # gain.  We have to rebind the recorded_signals dict to the new
        # diagram's port instance (the OutputPort objects are
        # diagram-specific identity references).
        diag2 = _build_diagram(k=10.0)
        sim.recorded_signals = {"y": diag2["integ"].output_ports[0]}
        sim.reset(diagram=diag2)
        assert sim._kernel is None
        assert sim.system is diag2

        r2 = sim.run()
        peak2 = float(jnp.max(jnp.abs(r2.outputs["y"])))
        assert peak2 > peak1, (
            f"new diagram with k=10 should integrate to a larger peak "
            f"than k=1; got peak1={peak1:.3f}, peak2={peak2:.3f}"
        )


# ---------------------------------------------------------------------------
# Structural-change warning
# ---------------------------------------------------------------------------


class TestStructuralWarning:
    def test_no_warning_on_first_run(self):
        """The very first :meth:`run` call must not warn."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        # No FastRestart-specific warning on the first call.
        relevant = [x for x in w if "FastRestart" in str(x.message)]
        assert relevant == []

    def test_no_warning_on_value_only_changes(self):
        """Same shapes/dtypes across runs ⇒ no warning."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        # Establish baseline.
        sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        # Same dtype & shape — only value differs.  Should not warn.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.run(parameters={"gain.gain": jnp.float64(2.0)})
        relevant = [x for x in w if "FastRestart" in str(x.message)]
        assert relevant == [], (
            "value-only parameter change should not emit a structural-"
            f"change warning; got: {[str(x.message) for x in relevant]}"
        )

    def test_warns_when_param_pytree_changes(self):
        """A run that adds/removes a parameter key relative to the first
        cached call must warn (the patched-context signature differs)."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        # First call: no params override — base context as-is.  This
        # establishes the cached signature.
        sim.run()
        # Second call: introduce a parameter override.  The base
        # context's ``gain.gain`` was a Python float; the patched
        # context's leaf is now a JAX array, which changes the
        # abstract signature ⇒ warn.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.run(parameters={"gain.gain": jnp.float64(2.0)})
        relevant = [x for x in w if "FastRestart" in str(x.message)]
        assert len(relevant) >= 1, (
            "expected a FastRestart structural-change warning when the "
            "patched context's pytree signature changes between runs; "
            f"got: {[str(x.message) for x in w]}"
        )

    def test_no_double_warning_on_same_change(self):
        """After a change is detected and warned about, repeating the same
        new pattern does not warn again."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        sim.run()  # baseline — no override
        # Trigger the warning once.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim.run(parameters={"gain.gain": jnp.float64(2.0)})
        # Second call with the same shape: must NOT re-warn (same
        # signature as the previous call).
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.run(parameters={"gain.gain": jnp.float64(3.0)})
        relevant = [x for x in w if "FastRestart" in str(x.message)]
        assert relevant == [], (
            "should not re-warn for the same structural pattern; got: "
            f"{[str(x.message) for x in relevant]}"
        )
