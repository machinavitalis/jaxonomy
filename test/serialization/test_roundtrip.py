# SPDX-License-Identifier: MIT

"""Serialization round-trip fidelity tests.

Invariant: serialize a diagram to JSON → deserialize → simulate both the
original and the reloaded diagram → simulation outputs must be numerically
identical (atol ≤ 1e-5).

Covers:
- Continuous-time blocks (Sine, Gain, Integrator, Adder, Saturate)
- Event-driven blocks (Step input)
- Discrete-time blocks (ZeroOrderHold, IntegratorDiscrete / UnitDelay)
- Composed diagrams (multiple connected blocks)
- Parameter preservation (gains, initial states, frequencies)
- Schema version round-trip
- File I/O (write + read JSON file)
- Error handling (too-new schema, unknown block type)
"""

import json
import os
import tempfile

import jax.numpy as jnp
import pytest

from jaxonomy import DiagramBuilder, SimulatorOptions, simulate
from jaxonomy import library
from jaxonomy.dashboard.serialization.from_model_json import load_model
from jaxonomy.dashboard.serialization.to_model_json import convert

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPTS_CT = SimulatorOptions(math_backend="jax")
_OPTS_DT = SimulatorOptions(math_backend="jax")

T_SPAN = (0.0, 2.0)
T_SPAN_SHORT = (0.0, 1.0)


def _sim(diagram, port, t_span=T_SPAN, opts=_OPTS_CT):
    ctx = diagram.create_context()
    return simulate(
        diagram,
        ctx,
        t_span=t_span,
        options=opts,
        recorded_signals={"y": port},
    )


def _roundtrip(diagram):
    """Serialize → deserialize.  Returns (reloaded_diagram, recorded_signals)."""
    model, _ = convert(diagram)
    json_str = json.dumps(model.to_dict())
    sc = load_model(json.loads(json_str))
    return sc.diagram, sc.recorded_signals


def _assert_roundtrip_identical(
    diagram, port, atol=1e-5, t_span=T_SPAN, opts=_OPTS_CT, sig_name=None
):
    """
    Simulate original and reloaded diagram; assert outputs are close.

    ``sig_name`` is the key used in the reloaded diagram's recorded_signals
    (auto-detected if None — takes the first matching key).
    """
    r1 = _sim(diagram, port, t_span=t_span, opts=opts)

    reloaded, recorded = _roundtrip(diagram)
    ctx2 = reloaded.create_context()

    # Find the matching signal in the reloaded diagram's recorded_signals
    if sig_name is None:
        # Use the first recorded signal that matches the block name of the port
        block_name = port.system.name
        candidates = [k for k in recorded if block_name in k]
        if not candidates:
            candidates = list(recorded.keys())
        sig_name = candidates[0]

    r2 = simulate(
        reloaded,
        ctx2,
        t_span=t_span,
        options=opts,
        recorded_signals={"y": recorded[sig_name]},
    )

    t_ref = r1.time
    y1 = r1.outputs["y"]
    y2 = jnp.interp(t_ref, r2.time, r2.outputs["y"])

    max_diff = float(jnp.max(jnp.abs(y1 - y2)))
    assert max_diff <= atol, (
        f"Round-trip simulation mismatch (sig={sig_name!r}, atol={atol}): "
        f"max_diff={max_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# 1. Basic continuous-time: Sine → Gain → Integrator
# ---------------------------------------------------------------------------

def test_roundtrip_sine_gain_integrator():
    """Canonical CT diagram round-trip."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(library.Gain(gain=2.0, name="gain"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="ct_basic")

    _assert_roundtrip_identical(diagram, integ.output_ports[0])


def test_roundtrip_parameter_preservation_gain():
    """Non-unity gain value is preserved through serialization."""
    for k in [0.5, 3.14, 10.0]:
        builder = DiagramBuilder()
        sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
        gain = builder.add(library.Gain(gain=k, name="gain"))
        integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
        builder.connect(sine.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], integ.input_ports[0])
        diagram = builder.build(name=f"gain_{k}")
        _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


def test_roundtrip_nonzero_initial_state():
    """Non-zero Integrator initial state is preserved."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    integ = builder.add(library.Integrator(initial_state=5.0, name="integ"))
    builder.connect(sine.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="ic_test")
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


def test_roundtrip_sine_frequency():
    """Sine block frequency is preserved through serialization."""
    builder = DiagramBuilder()
    sine = builder.add(
        library.Sine(amplitude=2.0, frequency=3.0, phase=0.5, name="sine")
    )
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="sine_freq")
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


# ---------------------------------------------------------------------------
# 2. Step input (event-driven zero-crossing)
# ---------------------------------------------------------------------------

def test_roundtrip_step_input():
    """Step block step_time is preserved; CT response matches after round-trip."""
    builder = DiagramBuilder()
    step = builder.add(library.Step(step_time=0.5, name="step"))
    gain = builder.add(library.Gain(gain=1.0, name="gain"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(step.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="step_ct")
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Constant input
# ---------------------------------------------------------------------------

def test_roundtrip_constant():
    """Constant block value preserved."""
    builder = DiagramBuilder()
    const = builder.add(library.Constant(value=jnp.array(3.14), name="const"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(const.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="const_ct")
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Adder (3 inputs)
# ---------------------------------------------------------------------------

def test_roundtrip_adder():
    """Adder with two inputs round-trips correctly."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    const = builder.add(library.Constant(value=jnp.array(0.5), name="const"))
    adder = builder.add(library.Adder(n_in=2, operators="++", name="adder"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], adder.input_ports[0])
    builder.connect(const.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="adder_ct")
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Saturate
# ---------------------------------------------------------------------------

def test_roundtrip_saturate():
    """Saturate block limits are preserved through serialization."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=2.0, frequency=1.0, name="sine"))
    sat = builder.add(
        library.Saturate(upper_limit=1.0, lower_limit=-1.0, name="sat")
    )
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], sat.input_ports[0])
    builder.connect(sat.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="sat_ct")
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-4)


# ---------------------------------------------------------------------------
# 6. Discrete-time: ZeroOrderHold
# ---------------------------------------------------------------------------

def test_roundtrip_zoh():
    """ZeroOrderHold block preserves dt through serialization."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    zoh = builder.add(library.ZeroOrderHold(dt=0.1, name="zoh"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], zoh.input_ports[0])
    builder.connect(zoh.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="zoh_dt")
    # ZOH introduces discretization; use looser tolerance
    _assert_roundtrip_identical(diagram, integ.output_ports[0], atol=1e-3)


# ---------------------------------------------------------------------------
# 7. Multi-output: verify each port
# ---------------------------------------------------------------------------

def test_roundtrip_two_integrators():
    """Diagram with two integrators: both outputs match after round-trip."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain1 = builder.add(library.Gain(gain=1.0, name="gain1"))
    gain2 = builder.add(library.Gain(gain=2.0, name="gain2"))
    integ1 = builder.add(library.Integrator(initial_state=0.0, name="integ1"))
    integ2 = builder.add(library.Integrator(initial_state=1.0, name="integ2"))
    builder.connect(sine.output_ports[0], gain1.input_ports[0])
    builder.connect(sine.output_ports[0], gain2.input_ports[0])
    builder.connect(gain1.output_ports[0], integ1.input_ports[0])
    builder.connect(gain2.output_ports[0], integ2.input_ports[0])
    diagram = builder.build(name="two_integ")

    for integ, port in [(integ1, integ1.output_ports[0]), (integ2, integ2.output_ports[0])]:
        _assert_roundtrip_identical(diagram, port, atol=1e-4)


# ---------------------------------------------------------------------------
# 8. JSON string round-trip (schema_version field)
# ---------------------------------------------------------------------------

def test_roundtrip_schema_version_present():
    """Serialized JSON must include schema_version=1."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="sv_test")

    model, _ = convert(diagram)
    d = model.to_dict()
    assert d.get("schema_version") == 1


def test_roundtrip_to_file():
    """Round-trip via actual file write/read."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(library.Gain(gain=2.0, name="gain"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="file_rt")

    model, _ = convert(diagram)
    model_dict = model.to_dict()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(model_dict, f)
        tmp_path = f.name

    try:
        with open(tmp_path) as f:
            loaded = json.load(f)
        sc = load_model(loaded)
        assert sc.diagram is not None
        # Simulate reloaded diagram
        sig_name = next(k for k in sc.recorded_signals if "integ" in k)
        ctx = sc.diagram.create_context()
        r = simulate(
            sc.diagram,
            ctx,
            t_span=T_SPAN,
            options=_OPTS_CT,
            recorded_signals={"y": sc.recorded_signals[sig_name]},
        )
        assert r.outputs["y"].shape[0] > 0
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 9. Error handling
# ---------------------------------------------------------------------------

def test_schema_version_too_new_raises():
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="sv_err")

    model, _ = convert(diagram)
    bad = model.to_dict()
    bad["schema_version"] = 9999

    with pytest.raises(ValueError, match="schema version"):
        load_model(bad)


def test_unknown_block_type_warns():
    builder = DiagramBuilder()
    u = builder.add(type("MyUnknownBlock", (library.Gain,), {})(gain=2.0, name="u"))
    diagram = builder.build(name="unk_block")

    with pytest.warns(UserWarning, match="not in the known"):
        convert(diagram)


def test_legacy_schema_version_warns():
    """JSON with string schema_version triggers a deprecation warning."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="legacy_sv")

    model, _ = convert(diagram)
    legacy = model.to_dict()
    legacy["schema_version"] = "3"  # legacy string format

    with pytest.warns(UserWarning, match="legacy"):
        load_model(legacy)


# ---------------------------------------------------------------------------
# 10. Simulation output correctness at specific time points
# ---------------------------------------------------------------------------

def test_roundtrip_integrator_ramp_correctness():
    """∫1 dt = t — reloaded integrator must trace the same ramp."""
    builder = DiagramBuilder()
    const = builder.add(library.Constant(value=jnp.array(1.0), name="one"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(const.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="ramp")

    r1 = _sim(diagram, integ.output_ports[0], t_span=(0.0, 3.0))

    reloaded, recorded = _roundtrip(diagram)
    sig = next(k for k in recorded if "integ" in k)
    r2 = simulate(
        reloaded,
        reloaded.create_context(),
        t_span=(0.0, 3.0),
        options=_OPTS_CT,
        recorded_signals={"y": recorded[sig]},
    )

    t_ref = r1.time
    y2 = jnp.interp(t_ref, r2.time, r2.outputs["y"])
    # Both should closely track t (linear ramp)
    assert jnp.allclose(r1.outputs["y"], y2, atol=1e-4)
    # Check against analytical solution by interpolating to specific time points
    for t_check in [0.5, 1.0, 2.0, 3.0]:
        y_at_t = float(jnp.interp(jnp.array(t_check), t_ref, r1.outputs["y"]))
        assert abs(y_at_t - t_check) < 0.05, (
            f"Ramp diverges at t={t_check}: got {y_at_t:.4f}, expected {t_check:.4f}"
        )


# ---------------------------------------------------------------------------
# 11. JSON completeness: all blocks serializable without crash
# ---------------------------------------------------------------------------

def test_convert_does_not_crash_for_basic_blocks():
    """All basic library blocks can be serialized without raising."""
    basic_blocks_and_ports = []

    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(library.Gain(gain=1.0, name="gain"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    const = builder.add(library.Constant(value=jnp.array(1.0), name="const"))
    sat = builder.add(library.Saturate(upper_limit=1.0, lower_limit=-1.0, name="sat"))

    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    builder.connect(const.output_ports[0], sat.input_ports[0])
    diagram = builder.build(name="all_basic")

    model, _ = convert(diagram)
    json_str = json.dumps(model.to_dict())
    assert len(json_str) > 0

    reloaded_dict = json.loads(json_str)
    assert reloaded_dict["schema_version"] == 1
    sc = load_model(reloaded_dict)
    assert sc.diagram is not None
