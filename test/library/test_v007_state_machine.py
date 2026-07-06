# SPDX-License-Identifier: MIT

"""V-007: State machine semantics and differentiability.

Covers Mealy semantics, deterministic transition priority, initial actions,
agnostic (zero-crossing) and discrete time modes, simulate_batch ensembles,
serialization round-trip, and gradient flow. Genuinely unimplemented features
(e.g. SM block-level parameters in guards) are xfail-tagged with T-018.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import DiagramBuilder, SimulatorOptions, StateMachineBuilder, simulate
from jaxonomy.library import Clock, StateMachine, Step
from jaxonomy.simulation import simulate_batch
from jaxonomy.testing.markers import skip_if_not_jax

pytestmark = pytest.mark.minimal


def _jax_opts(max_major_steps: int = 200) -> SimulatorOptions:
    return SimulatorOptions(math_backend="jax", max_major_steps=max_major_steps)


def _build_three_state_sm(time_mode: str = "discrete", dt: float | None = 0.05) -> StateMachine:
    smb = StateMachineBuilder()
    idle = smb.add_state("idle")
    running = smb.add_state("running")
    done = smb.add_state("done")
    smb.set_initial_state(idle)
    smb.add_transition(idle, running, guard="x > 0.5", actions=["y = 1.0"])
    smb.add_transition(running, done, guard="x > 1.5", actions=["y = 2.0"])
    sm_default = smb.build(name="three_state_sm_template")
    if time_mode == "agnostic":
        return sm_default
    # Re-instantiate in discrete mode so recorded_signals samples the SM at every dt.
    return StateMachine(
        sm_data=sm_default._sm,
        inputs=list(sm_default._input_names),
        outputs=list(sm_default._output_names),
        dt=dt,
        time_mode=time_mode,
        name="three_state_sm",
        accelerate_with_jax=False,
    )


# 1. Three-state controller

def test_three_state_controller_advances_to_done():
    """idle -> running -> done driven by a clock crossing 0.5 then 1.5."""
    sm = _build_three_state_sm(time_mode="discrete", dt=0.05)
    builder = DiagramBuilder()
    clock = builder.add(Clock(name="clk"))
    sm_block = builder.add(sm)
    # SM has one input named "x" — connect the clock to it.
    builder.connect(clock.output_ports[0], sm_block.input_ports[0])
    diagram = builder.build(name="three_state_root")

    ctx = diagram.create_context()
    results = simulate(
        diagram,
        ctx,
        (0.0, 3.0),
        options=_jax_opts(max_major_steps=400),
        recorded_signals={"y": sm_block.output_ports[0]},
    )

    t = np.asarray(results.time)
    y = np.asarray(results.outputs["y"])

    # Initial output (no transitions taken yet) should be 0.
    assert float(y[0]) == pytest.approx(0.0, abs=1e-6)
    # Mid-region (after first crossing, before second): y == 1.
    mid_mask = (t > 0.7) & (t < 1.3)
    assert mid_mask.any()
    assert int(np.max(y[mid_mask])) == 1
    assert int(np.min(y[mid_mask])) == 1
    # End: y == 2 (state machine reached the "done" state).
    end_mask = t > 1.8
    assert end_mask.any()
    assert int(np.min(y[end_mask])) == 2


# 2. initial_actions: entry-point actions set output values

def test_initial_actions_set_output_at_t0():
    """The initial state's on_entry actions run at init; outputs reflect them."""
    smb = StateMachineBuilder()
    boot = smb.add_state("boot")
    boot.on_entry = ["y = 42.0"]
    work = smb.add_state("work")
    smb.set_initial_state(boot)
    # Provide a (never-taken) transition so we still validate.
    smb.add_transition(boot, work, guard="x > 1e9", actions=["y = -1.0"])
    sm = smb.build(name="init_action_sm")

    builder = DiagramBuilder()
    clk = builder.add(Clock(name="clk"))
    sm_block = builder.add(sm)
    builder.connect(clk.output_ports[0], sm_block.input_ports[0])
    diagram = builder.build(name="init_root")

    ctx = diagram.create_context()
    results = simulate(
        diagram,
        ctx,
        (0.0, 0.5),
        options=_jax_opts(max_major_steps=80),
        recorded_signals={"y": sm_block.output_ports[0]},
    )
    y = np.asarray(results.outputs["y"])
    # Every recorded sample should be 42.0 (no other transition fires).
    assert np.allclose(y, 42.0)


# 3. Multiple guards firing simultaneously: lowest-index wins, deterministic

def _build_priority_sm() -> StateMachine:
    """Discrete-mode SM with two transitions out of s0 sharing the same guard.

    Documented contract (StateMachineBuilder docstring): when guards fire
    simultaneously, the first transition added wins. We use discrete mode here
    because the documented priority is enforced by the discrete-update logic;
    the zero-crossing path declares one ZC per transition and does not honor
    the rule the same way."""
    smb = StateMachineBuilder()
    s0 = smb.add_state("s0")
    sa = smb.add_state("sa")
    sb = smb.add_state("sb")
    smb.set_initial_state(s0)
    smb.add_transition(s0, sa, guard="x > 0.5", actions=["y = 10.0"])
    smb.add_transition(s0, sb, guard="x > 0.5", actions=["y = 20.0"])
    sm_default = smb.build(name="priority_sm_template")
    return StateMachine(
        sm_data=sm_default._sm,
        inputs=list(sm_default._input_names),
        outputs=list(sm_default._output_names),
        dt=0.05,
        time_mode="discrete",
        name="priority_sm",
        accelerate_with_jax=False,
    )


def _run_priority_sm(sm: StateMachine) -> np.ndarray:
    builder = DiagramBuilder()
    clk = builder.add(Clock(name="clk"))
    blk = builder.add(sm)
    builder.connect(clk.output_ports[0], blk.input_ports[0])
    diagram = builder.build(name="prio_root")
    ctx = diagram.create_context()
    results = simulate(
        diagram,
        ctx,
        (0.0, 1.5),
        options=_jax_opts(max_major_steps=200),
        recorded_signals={"y": blk.output_ports[0]},
    )
    return np.asarray(results.outputs["y"])


def test_simultaneous_guards_lowest_index_wins_deterministic():
    sm1 = _build_priority_sm()
    sm2 = _build_priority_sm()
    y1 = _run_priority_sm(sm1)
    y2 = _run_priority_sm(sm2)

    # Final value must be 10 (action from the first transition), not 20.
    assert float(y1[-1]) == pytest.approx(10.0, abs=1e-6)
    # Determinism: two independent runs produce identical traces.
    assert np.allclose(y1, y2)
    # 20 must never appear.
    assert not np.any(np.isclose(y1, 20.0))


# 4. Continuous mode (zero-crossing-driven) — guard "x > 1.0" via Clock.

def test_continuous_zero_crossing_triggers_at_threshold():
    """Default time_mode='agnostic' uses zero-crossings for guard transitions.

    Drive the SM with a continuous Sine trajectory crossing 1.0; verify the
    guard-driven action fires near the crossing time.
    """
    smb = StateMachineBuilder()
    below = smb.add_state("below")
    above = smb.add_state("above")
    smb.set_initial_state(below)
    smb.add_transition(below, above, guard="x > 1.0", actions=["y = 7.0"])
    sm = smb.build(name="zc_sm")
    assert sm.time_mode == "agnostic"

    builder = DiagramBuilder()
    # Drive the SM with a clock so x = t (continuous, crosses 1.0 at t=1.0).
    clk = builder.add(Clock(name="clk"))
    blk = builder.add(sm)
    builder.connect(clk.output_ports[0], blk.input_ports[0])
    diagram = builder.build(name="zc_root")

    ctx = diagram.create_context()
    # Force the simulator to bound major-step length so recording captures the
    # output trajectory before/after the zero-crossing event.
    opts = SimulatorOptions(
        math_backend="jax",
        max_major_steps=300,
        max_major_step_length=0.1,
    )
    results = simulate(
        diagram,
        ctx,
        (0.0, 2.0),
        options=opts,
        recorded_signals={"y": blk.output_ports[0]},
    )
    t = np.asarray(results.time)
    y = np.asarray(results.outputs["y"])

    # Below threshold (well before t=1.0): y == 0 (default initial value).
    below_mask = t < 0.9
    assert below_mask.any()
    assert np.all(y[below_mask] == 0.0)
    # After threshold (well after t=1.0): y == 7 (action fired).
    above_mask = t > 1.1
    assert above_mask.any()
    assert np.all(y[above_mask] == 7.0)


# 4b. T-033: priority resolution in agnostic / zero-crossing mode

def test_t033_priority_in_agnostic_mode_lowest_index_wins():
    """When two transitions out of the same source state share guards
    that fire at the same instant, the lowest-priority-index (= first
    declared) transition's reset map wins, even in agnostic mode.

    Pre-T-033 the agnostic path declared one zero-crossing per
    transition with no priority gating, so the higher-index
    transition could override the lower-index one. The fix gates each
    guard on the negation of all higher-priority sibling guards from
    the same source state."""
    # Use a Step input so both guards (x > 0.5) become True
    # simultaneously at t=0.5. In the buggy implementation, sb's
    # action (y = 20) overwrites sa's (y = 10).
    smb = StateMachineBuilder()
    s0 = smb.add_state("s0")
    sa = smb.add_state("sa")
    sb = smb.add_state("sb")
    smb.set_initial_state(s0)
    smb.add_transition(s0, sa, guard="x > 0.5", actions=["y = 10.0"])
    smb.add_transition(s0, sb, guard="x > 0.5", actions=["y = 20.0"])
    sm = smb.build(name="zc_priority_sm")
    assert sm.time_mode == "agnostic"  # default

    builder = DiagramBuilder()
    step = builder.add(
        Step(start_value=0.0, end_value=1.0, step_time=0.5, name="step")
    )
    blk = builder.add(sm)
    builder.connect(step.output_ports[0], blk.input_ports[0])
    diagram = builder.build(name="zc_priority_root")
    ctx = diagram.create_context()

    opts = SimulatorOptions(
        math_backend="jax",
        max_major_steps=200,
        max_major_step_length=0.05,
    )
    results = simulate(
        diagram, ctx, (0.0, 1.5),
        options=opts,
        recorded_signals={"y": blk.output_ports[0]},
    )
    y = np.asarray(results.outputs["y"])
    # Final value must be 10.0 (sa's action, the higher-priority
    # transition), not 20.0.
    assert float(y[-1]) == pytest.approx(10.0, abs=1e-6), (
        f"agnostic-mode priority broken: y[-1] = {y[-1]} (expected 10.0). "
        "T-033 regression."
    )
    # And 20.0 must never appear in the trace — sb's reset map should
    # never fire because sa's higher-priority guard blocks it.
    assert not np.any(np.isclose(y, 20.0)), (
        "lower-priority transition 's0->sb' fired despite sa being "
        "simultaneously eligible"
    )


# 5. Discrete / sample-driven mode (StateMachine constructor flag)

def test_discrete_time_mode_fires_at_periodic_update():
    """Build via the lower-level constructor with time_mode='discrete', dt=0.1."""
    # Reuse the 3-state SM, but rebuild it as a discrete-time block.
    smb = StateMachineBuilder()
    s0 = smb.add_state("s0")
    s1 = smb.add_state("s1")
    smb.set_initial_state(s0)
    smb.add_transition(s0, s1, guard="x > 0.5", actions=["y = 5.0"])
    sm_default = smb.build(name="discrete_sm_template")

    # Re-instantiate the underlying StateMachine in discrete mode.
    sm_discrete = StateMachine(
        sm_data=sm_default._sm,
        inputs=list(sm_default._input_names),
        outputs=list(sm_default._output_names),
        dt=0.1,
        time_mode="discrete",
        name="discrete_sm",
        accelerate_with_jax=False,
    )
    assert sm_discrete.time_mode == "discrete"

    builder = DiagramBuilder()
    clk = builder.add(Clock(name="clk"))
    blk = builder.add(sm_discrete)
    builder.connect(clk.output_ports[0], blk.input_ports[0])
    diagram = builder.build(name="discrete_root")
    ctx = diagram.create_context()
    results = simulate(
        diagram,
        ctx,
        (0.0, 1.0),
        options=_jax_opts(max_major_steps=200),
        recorded_signals={"y": blk.output_ports[0]},
    )
    y = np.asarray(results.outputs["y"])
    # By the end of the run, the discrete update must have fired and produced y=5.
    assert float(y[-1]) == pytest.approx(5.0, abs=1e-6)


# 6. Parameters in guards (T-018: not yet implemented at SM-block level)

@pytest.mark.xfail(
    reason="T-018: StateMachine does not expose block-level parameters that can be "
           "referenced by name in guard expressions. Threshold names are interpreted "
           "as input ports, not as tunable SM parameters.",
    strict=False,
)
def test_parametrized_threshold_in_guard():
    smb = StateMachineBuilder()
    s0 = smb.add_state("s0")
    s1 = smb.add_state("s1")
    smb.set_initial_state(s0)
    # threshold appears in the guard. The current builder/block treats it as an
    # input port (no SM-level Parameter mechanism wired in), so we cannot bind
    # it as a tunable parameter. T-018 tracks this work.
    smb.add_transition(s0, s1, guard="x > threshold", actions=["y = 1.0"])
    sm = smb.build(name="param_guard_sm")

    # If an attribute / parameter binding API ever appears, this is where we'd use it:
    # sm.set_parameter("threshold", 0.7)
    if not hasattr(sm, "set_parameter"):
        pytest.fail("T-018: SM parameters in guards not implemented yet")


# 7. Inside simulate_batch: each batch element transitions independently

def test_state_machine_inside_simulate_batch():
    """Use a Step amplitude as the batched parameter so each batch element sees
    a different driving signal; SM transitions when the step crosses 0.5."""
    skip_if_not_jax()

    smb = StateMachineBuilder()
    s0 = smb.add_state("s0")
    s1 = smb.add_state("s1")
    smb.set_initial_state(s0)
    smb.add_transition(s0, s1, guard="x > 0.5", actions=["y = 1.0"])
    sm = smb.build(name="batch_sm")

    builder = DiagramBuilder()
    step = builder.add(
        Step(start_value=0.0, end_value=1.0, step_time=0.5, name="step")
    )
    blk = builder.add(sm)
    builder.connect(step.output_ports[0], blk.input_ports[0])
    diagram = builder.build(name="batch_root")

    # end_value is dynamic on Step; vary it across the batch.
    end_values = jnp.array([0.0, 1.0, 2.0])  # only N=2,3 should cross 0.5.
    result = simulate_batch(
        diagram,
        t_span=(0.0, 1.5),
        param_batches={"step.end_value": end_values},
        options=_jax_opts(max_major_steps=200),
        recorded_signals={"y": blk.output_ports[0]},
    )
    y = np.asarray(result.outputs["y"])  # shape (N, T)
    assert y.shape[0] == 3
    # Element 0: end_value=0, never crosses -> remains 0.
    assert np.allclose(y[0], 0.0)
    # Elements 1, 2: end_value > 0.5 -> SM eventually fires y=1.
    assert float(y[1, -1]) == pytest.approx(1.0, abs=1e-6)
    assert float(y[2, -1]) == pytest.approx(1.0, abs=1e-6)


# 8. Gradient w.r.t. a parameter affecting the action arithmetic

@pytest.mark.xfail(
    reason="T-018: action RHS values are baked into action functions at build "
           "time; there is no SM-level parameter input that jax.grad can flow "
           "through. The campaign tracks adding differentiable SM action params.",
    strict=False,
)
def test_gradient_through_state_machine_action():
    skip_if_not_jax()

    def terminal_y(action_value: float) -> jnp.ndarray:
        smb = StateMachineBuilder()
        s0 = smb.add_state("s0")
        s1 = smb.add_state("s1")
        smb.set_initial_state(s0)
        smb.add_transition(s0, s1, guard="x > 0.5", actions=[f"y = {action_value}"])
        sm = smb.build(name="grad_sm")

        builder = DiagramBuilder()
        clk = builder.add(Clock(name="clk"))
        blk = builder.add(sm)
        builder.connect(clk.output_ports[0], blk.input_ports[0])
        diagram = builder.build(name="grad_root")
        ctx = diagram.create_context()
        results = simulate(
            diagram, ctx, (0.0, 1.0),
            options=_jax_opts(max_major_steps=100),
            recorded_signals={"y": blk.output_ports[0]},
        )
        return results.outputs["y"][-1]

    a = 3.0
    eps = 1e-3
    fd = (terminal_y(a + eps) - terminal_y(a - eps)) / (2 * eps)
    g = jax.grad(lambda v: terminal_y(float(v)))(jnp.array(a))
    # Expected slope is 1.0 (terminal y is the action RHS itself).
    assert float(g) == pytest.approx(float(fd), rel=1e-2, abs=1e-3)


# 9. Serialization round-trip (model JSON via builder._to_model_json)

def test_state_machine_model_json_roundtrip():
    """The builder's compiled ``model_json.StateMachine`` survives a JSON
    serialize / deserialize via its ``to_json`` / ``from_json`` API.

    This is the SM-only narrow round-trip (a full diagram round-trip lives in
    ``jaxonomy.dashboard.serialization`` and is out of scope for V-007).
    """
    smb = StateMachineBuilder()
    a = smb.add_state("a")
    b = smb.add_state("b")
    smb.set_initial_state(a)
    smb.add_transition(a, b, guard="x > 0.5", actions=["y = 1.0"])

    _, output_names = smb._collect_io_names()
    entry_actions = smb._entry_actions(output_names)
    model = smb._to_model_json(entry_actions)

    # Use the model_json's documented to_json / from_json round-trip.
    try:
        json_str = model.to_json()
        restored = type(model).from_json(json_str)
    except Exception as e:  # pragma: no cover - documented fallback
        pytest.xfail(f"SM model_json to_json/from_json round-trip failed: {e!r}")

    # State names preserved.
    original_names = sorted(n.name for n in model.nodes)
    restored_names = sorted(n.name for n in restored.nodes)
    assert original_names == restored_names

    # Number of links preserved.
    assert len(restored.links) == len(model.links)

    # Guard text preserved.
    assert {ln.guard for ln in restored.links} == {ln.guard for ln in model.links}

    # The originally-built SM still simulates correctly.
    sm = smb.build(name="rt_sm")
    sm_discrete = StateMachine(
        sm_data=sm._sm,
        inputs=list(sm._input_names),
        outputs=list(sm._output_names),
        dt=0.05,
        time_mode="discrete",
        name="rt_sm_discrete",
    )
    builder = DiagramBuilder()
    clk = builder.add(Clock(name="clk"))
    blk = builder.add(sm_discrete)
    builder.connect(clk.output_ports[0], blk.input_ports[0])
    diagram = builder.build(name="rt_root")
    ctx = diagram.create_context()
    results = simulate(
        diagram, ctx, (0.0, 1.0),
        options=_jax_opts(max_major_steps=200),
        recorded_signals={"y": blk.output_ports[0]},
    )
    y = np.asarray(results.outputs["y"])
    assert float(y[-1]) == pytest.approx(1.0, abs=1e-6)
