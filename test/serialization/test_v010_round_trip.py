# SPDX-License-Identifier: MIT
"""V-010: Round-trip serialization fidelity across the block library.

Invariant: every public block type can be serialized to JSON and loaded back
to produce a behaviorally equivalent system. ``load -> save -> load`` must be
deterministic, and simulating the round-tripped diagram must match the original
within numerical tolerance.

Pattern follows ``test/serialization/test_roundtrip.py``:
    convert(diagram) -> json.dumps -> json.loads -> load_model -> simulate
"""

from __future__ import annotations

import copy
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import DiagramBuilder, SimulatorOptions, simulate
from jaxonomy import library
from jaxonomy.dashboard.serialization.from_model_json import load_model
from jaxonomy.dashboard.serialization.to_model_json import convert

pytestmark = pytest.mark.minimal

_OPTS = SimulatorOptions(math_backend="jax")
_T_SPAN = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(diagram) -> dict:
    """Serialize diagram -> JSON dict (round-trip via json.dumps for safety)."""
    model, _ = convert(diagram)
    return json.loads(json.dumps(model.to_dict()))


def _roundtrip(diagram):
    """Serialize -> deserialize. Returns SimulationContext."""
    return load_model(_to_dict(diagram))


def _sim(diagram, port, t_span=_T_SPAN):
    return simulate(
        diagram,
        diagram.create_context(),
        t_span=t_span,
        options=_OPTS,
        recorded_signals={"y": port},
    )


def _recorded_for(recorded, hint):
    for k, v in recorded.items():
        if hint in k:
            return v
    return next(iter(recorded.values()))


def _compare_outputs(diagram, port, hint, atol=1e-3, t_span=_T_SPAN):
    """Round-trip then compare original vs. reloaded simulation outputs."""
    r1 = _sim(diagram, port, t_span=t_span)
    sc = _roundtrip(diagram)
    r2 = simulate(
        sc.diagram,
        sc.diagram.create_context(),
        t_span=t_span,
        options=_OPTS,
        recorded_signals={"y": _recorded_for(sc.recorded_signals, hint)},
    )
    t_ref = r1.time
    y1 = np.asarray(r1.outputs["y"]).astype(float)
    y2 = np.asarray(jnp.interp(t_ref, r2.time, r2.outputs["y"])).astype(float)
    max_diff = float(np.max(np.abs(y1 - y2)))
    assert max_diff <= atol, (
        f"Round-trip mismatch for {hint!r}: max_diff={max_diff:.3e} > {atol}"
    )


def _sine(b, name="src", **kw):
    kw.setdefault("amplitude", 1.0)
    kw.setdefault("frequency", 1.0)
    return b.add(library.Sine(name=name, **kw))


def _const(b, value=1.0, name="src"):
    return b.add(library.Constant(value=jnp.array(value), name=name))


def _make(blk_factory, hint, sources=None):
    """Build (diagram, port, hint) for a block driven by ``sources``."""
    sources = sources or [_sine]
    b = DiagramBuilder()
    srcs = [sf(b, name=f"s{i}") for i, sf in enumerate(sources)]
    blk = b.add(blk_factory(hint))
    for i, s in enumerate(srcs):
        b.connect(s.output_ports[0], blk.input_ports[i])
    return b.build(name=f"d_{hint}"), blk.output_ports[0], hint


def _src_only(blk_factory, hint):
    """Build a diagram with just a single source-block (no inputs)."""
    b = DiagramBuilder()
    blk = b.add(blk_factory(hint))
    return b.build(name=f"d_{hint}"), blk.output_ports[0], hint


def _mux_demux():
    b = DiagramBuilder()
    s1 = _sine(b, name="s1")
    s2 = _const(b, value=2.0, name="s2")
    mux = b.add(library.Multiplexer(n_in=2, name="mux"))
    demux = b.add(library.Demultiplexer(n_out=2, name="demux"))
    b.connect(s1.output_ports[0], mux.input_ports[0])
    b.connect(s2.output_ports[0], mux.input_ports[1])
    b.connect(mux.output_ports[0], demux.input_ports[0])
    return b.build(name="d_mux"), demux.output_ports[0], "demux"


# ---------------------------------------------------------------------------
# Block cases: (id, builder, atol)
# ---------------------------------------------------------------------------


_LTI_ABCD = (
    jnp.array([[-1.0]]),
    jnp.array([[1.0]]),
    jnp.array([[1.0]]),
    jnp.array([[0.0]]),
)


_BLOCK_CASES = [
    ("Integrator", lambda: _make(
        lambda h: library.Integrator(initial_state=0.0, name=h), "integ"), 1e-3),
    ("Gain", lambda: _make(
        lambda h: library.Gain(gain=2.5, name=h), "gain"), 1e-4),
    ("Adder", lambda: _make(
        lambda h: library.Adder(n_in=2, operators="+-", name=h), "adder",
        sources=[_sine, lambda b, name: _const(b, value=0.5, name=name)]), 1e-4),
    ("Sine", lambda: _src_only(
        lambda h: library.Sine(amplitude=2.0, frequency=1.5, phase=0.3, name=h),
        "sine"), 1e-4),
    ("Step", lambda: _src_only(
        lambda h: library.Step(step_time=0.5, name=h), "step"), 1e-4),
    ("Constant", lambda: _src_only(
        lambda h: library.Constant(value=jnp.array(3.14), name=h), "const"), 1e-4),
    ("Comparator", lambda: _make(
        lambda h: library.Comparator(operator=">", name=h), "cmp",
        sources=[_sine, lambda b, name: _const(b, value=0.0, name=name)]), 1e-4),
    ("Product", lambda: _make(
        lambda h: library.Product(n_in=2, operators="**", name=h), "prod",
        sources=[_sine, lambda b, name: _const(b, value=2.0, name=name)]), 1e-4),
    ("Saturate", lambda: _make(
        lambda h: library.Saturate(
            upper_limit=1.0, lower_limit=-1.0, name=h), "sat",
        sources=[lambda b, name: _sine(b, name=name, amplitude=2.0)]), 1e-4),
    ("PID", lambda: _make(
        lambda h: library.PID(kp=1.0, ki=0.5, kd=0.1, n=10.0, name=h),
        "pid"), 1e-3),
    ("TransferFunction", lambda: _make(
        lambda h: library.TransferFunction(num=[1.0], den=[1.0, 1.0], name=h),
        "tf"), 1e-3),
    ("LTISystem", lambda: _make(
        lambda h: library.LTISystem(*_LTI_ABCD, name=h), "lti"), 1e-3),
    ("Derivative", lambda: _make(
        lambda h: library.Derivative(filter_coefficient=50.0, name=h),
        "deriv"), 1e-3),
    # T-037-followup (resolved 2026-04-27): per-block `dt` is now declared
    # via `@parameters(static=...)` on these blocks, so round-tripped JSON
    # preserves the original `dt` rather than falling back to the model-wide
    # `sample_time`. These cases use a non-default `dt` (0.05 vs the 0.1
    # global default) to exercise that path.
    ("FilterDiscrete", lambda: _make(
        lambda h: library.FilterDiscrete(
            dt=0.05, b_coefficients=[0.5, 0.5], name=h), "fir"), 1e-3),
    ("LookupTable1d", lambda: _make(
        lambda h: library.LookupTable1d(
            input_array=jnp.array([-1.0, 0.0, 1.0]),
            output_array=jnp.array([-2.0, 0.0, 2.0]),
            interpolation="linear", name=h), "lut"), 1e-4),
    ("EdgeDetection", lambda: _make(
        lambda h: library.EdgeDetection(
            dt=0.05, edge_detection="rising", initial_state=False, name=h),
        "edge",
        sources=[lambda b, name: b.add(library.Step(step_time=0.3, name=name))]),
        1e-3),
    ("Multiplexer", lambda: _mux_demux(), 1e-4),
    ("Demultiplexer", lambda: _mux_demux(), 1e-4),
    ("ZeroOrderHold", lambda: _make(
        lambda h: library.ZeroOrderHold(dt=0.1, name=h), "zoh"), 1e-3),
    ("UnitDelay", lambda: _make(
        lambda h: library.UnitDelay(dt=0.1, initial_state=0.0, name=h),
        "ud"), 1e-3),
    ("Clock", lambda: _src_only(
        lambda h: library.Clock(name=h), "clk"), 1e-4),
    ("IfThenElse", lambda: _make(
        lambda h: library.IfThenElse(name=h), "ite",
        sources=[
            lambda b, name: b.add(library.Step(step_time=0.5, name=name)),
            lambda b, name: _const(b, value=1.0, name=name),
            lambda b, name: _const(b, value=-1.0, name=name),
        ]), 1e-3),
    ("Abs", lambda: _make(
        lambda h: library.Abs(name=h), "abs"), 1e-4),
    ("SquareRoot", lambda: _make(
        lambda h: library.SquareRoot(name=h), "sqrt",
        sources=[lambda b, name: _const(b, value=4.0, name=name)]), 1e-4),
]


# ---------------------------------------------------------------------------
# 1. Per-block round-trip
# ---------------------------------------------------------------------------


_KNOWN_BLOCK_DTYPE_GAPS: set[str] = set()
# T-037a/b (resolved 2026-04-27): FilterDiscrete and EdgeDetection now enforce
# a block-level dtype contract on their discrete-state elements:
#   * EdgeDetection casts `prev_input` and `output` to `bool_` at every site
#     (declare default, reset_default_values, _update). The block is
#     documented as bool-input/bool-output, so this is the canonical contract.
#   * FilterDiscrete derives a canonical state dtype from `b_coefficients`
#     via `result_type(...)`, casts the zero-init delay line to it, and
#     casts the runtime input on every push. The FIR coefficients are the
#     load-bearing typed parameter, so they own the contract.
# Both fresh-built and round-tripped blocks now agree on dtype, so the
# `lax.cond` branches in the periodic-update reset map no longer diverge.


@pytest.mark.parametrize(
    "name,builder,atol", _BLOCK_CASES, ids=[c[0] for c in _BLOCK_CASES]
)
def test_block_round_trip(name, builder, atol):
    """Each library block survives JSON round-trip with matching simulation."""
    if name in _KNOWN_BLOCK_DTYPE_GAPS:
        pytest.xfail(
            f"{name}: known block-level dtype gap on JSON round-trip; see "
            "the comment block above _KNOWN_BLOCK_DTYPE_GAPS for the active "
            "follow-up tag."
        )
    diagram, port, hint = builder()
    _compare_outputs(diagram, port, hint, atol=atol)


# ---------------------------------------------------------------------------
# 2. Hierarchical: nested DiagramBuilder
# ---------------------------------------------------------------------------


def test_round_trip_hierarchical_diagram():
    """Diagram containing a nested sub-diagram round-trips correctly."""
    inner = DiagramBuilder()
    s = inner.add(library.Sine(amplitude=1.0, frequency=1.0, name="inner_sine"))
    g = inner.add(library.Gain(gain=2.0, name="inner_gain"))
    inner.connect(s.output_ports[0], g.input_ports[0])
    inner.export_output(g.output_ports[0])
    inner_diag = inner.build(name="inner")

    outer = DiagramBuilder()
    sub = outer.add(inner_diag)
    integ = outer.add(library.Integrator(initial_state=0.0, name="outer_integ"))
    outer.connect(sub.output_ports[0], integ.input_ports[0])
    diagram = outer.build(name="outer")
    _compare_outputs(diagram, integ.output_ports[0], "outer_integ", atol=1e-3)


# ---------------------------------------------------------------------------
# 3. State machine — round-trip via JSON file
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "StateMachine round-trip via convert() is not fully supported: "
        "convert() does not currently re-emit the `state_machines` blob "
        "needed by load_model(). V-010 surfaces this as a gap."
    ),
    strict=False,
)
def test_round_trip_state_machine():
    """State-machine-containing diagram should survive a JSON round-trip."""
    sm_json = os.path.join(
        os.path.dirname(__file__), "..", "app", "StateMachine", "model.json"
    )
    with open(sm_json) as f:
        original = json.load(f)
    sc = load_model(original)
    sc2 = load_model(_to_dict(sc.diagram))
    assert sc2.diagram is not None


# ---------------------------------------------------------------------------
# 4. Acausal subsystem (RC circuit)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Acausal subsystems (e.g. RC circuit) are not yet supported by the "
        "JSON serialization path used here. V-010 surfaces this as a gap."
    ),
    strict=False,
)
def test_round_trip_acausal_rc_circuit():
    """RC circuit acausal subsystem should round-trip."""
    from jaxonomy.acausal import EqnEnv, AcausalCompiler  # noqa: F401
    from jaxonomy.acausal.component_library import electrical as elec  # noqa

    ev = EqnEnv()
    v = elec.VoltageSource(ev, name="v", v=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    c = elec.Capacitor(ev, name="c", C=1.0, initial_voltage=0.0)
    g = elec.Ground(ev, name="g")

    builder = DiagramBuilder()
    builder.add(AcausalCompiler(ev, [v, r, c, g])())
    diag = builder.build(name="rc_acausal")
    sc = _roundtrip(diag)
    assert sc.diagram is not None


# ---------------------------------------------------------------------------
# 5. Custom Python block
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "CustomPythonBlock body is a Python callable; convert() cannot embed "
        "and re-import arbitrary code through the JSON schema."
    ),
    strict=False,
)
def test_round_trip_custom_python_block():
    """CustomPythonBlock surface — expected to fail under JSON round-trip."""
    builder = DiagramBuilder()
    s = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="src"))
    blk = builder.add(
        library.CustomPythonBlock(
            dt=0.1,
            init_script="x = 0.0",
            user_statements="x = u_0 * 2.0\ny_0 = x",
            inputs=["u_0"],
            outputs=[("y_0", (), "float64")],
            name="cpb",
        )
    )
    builder.connect(s.output_ports[0], blk.input_ports[0])
    diag = builder.build(name="d_cpb")
    _compare_outputs(diag, blk.output_ports[0], "cpb", atol=1e-3)


# ---------------------------------------------------------------------------
# 6. Idempotence: load -> save -> parse -> save must produce equal dicts
# ---------------------------------------------------------------------------


def _normalize(obj):
    """Sort lists of dicts so dicts can be compared after re-serialization."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        items = [_normalize(v) for v in obj]
        try:
            items = sorted(
                items,
                key=lambda x: x.get("uuid", "") if isinstance(x, dict) else str(x),
            )
        except TypeError:
            pass
        return items
    return obj


def test_round_trip_idempotence():
    """Re-serializing twice must yield equal (after normalization) JSON."""
    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=2.0, name="sine"))
    gain = builder.add(library.Gain(gain=3.0, name="gain"))
    integ = builder.add(library.Integrator(initial_state=0.5, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="idempotence")

    d1 = _to_dict(diagram)
    sc = load_model(copy.deepcopy(d1))
    d2 = _to_dict(sc.diagram)

    assert _normalize(d1) == _normalize(d2), (
        "Round-trip is not idempotent: re-serialized JSON differs from the "
        "first serialization (after normalization)."
    )


# ---------------------------------------------------------------------------
# 7. Per-block dt round-trip (T-037-followup)
# ---------------------------------------------------------------------------


def test_per_block_dt_round_trip():
    """Two UnitDelay blocks with different `dt` values must round-trip.

    Before T-037-followup, both blocks would be reloaded with the global
    `model.configuration.sample_time` (default 0.1), losing the per-block
    override. With per-block `dt` registered as a static parameter, each
    block's own `dt` survives the JSON round-trip.
    """
    builder = DiagramBuilder()
    src_a = _const(builder, value=1.0, name="src_a")
    src_b = _const(builder, value=2.0, name="src_b")
    ud_fast = builder.add(library.UnitDelay(dt=0.05, initial_state=0.0,
                                            name="ud_fast"))
    ud_slow = builder.add(library.UnitDelay(dt=0.2, initial_state=0.0,
                                            name="ud_slow"))
    builder.connect(src_a.output_ports[0], ud_fast.input_ports[0])
    builder.connect(src_b.output_ports[0], ud_slow.input_ports[0])
    diagram = builder.build(name="d_per_block_dt")

    sc = _roundtrip(diagram)
    blocks_by_name = {n.name: n for n in sc.diagram.nodes if hasattr(n, "name")}
    assert "ud_fast" in blocks_by_name and "ud_slow" in blocks_by_name
    # Per-block dt must survive — not collapse to the global sample_time (0.1).
    assert float(blocks_by_name["ud_fast"].dt) == pytest.approx(0.05)
    assert float(blocks_by_name["ud_slow"].dt) == pytest.approx(0.2)
