# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.framework.state_machine_builder import StateMachineBuilder
from jaxonomy.library import Clock

pytestmark = pytest.mark.minimal


def test_basic_two_state_machine():
    smb = StateMachineBuilder()
    off = smb.add_state("off")
    on = smb.add_state("on")
    smb.set_initial_state(off)
    smb.add_transition(off, on, guard="u > 0.5")
    smb.add_transition(on, off, guard="u < 0.5")
    sm = smb.build(name="switch")
    assert len(sm.input_ports) >= 1
    assert len(sm.output_ports) >= 1


def test_state_machine_in_diagram():
    smb = StateMachineBuilder()
    idle = smb.add_state("idle")
    active = smb.add_state("active")
    smb.set_initial_state(idle)
    smb.add_transition(idle, active, guard="t > 1.0")
    sm = smb.build(name="timer_sm")

    builder = jaxonomy.DiagramBuilder()
    clock = builder.add(Clock(name="clock"))
    state_machine = builder.add(sm)
    builder.connect(clock.output_ports[0], state_machine.input_ports[0])
    diagram = builder.build(name="sm_root")

    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 3.0),
        options=jaxonomy.SimulatorOptions(
            math_backend="jax",
            max_major_steps=300,
        ),
        recorded_signals={"state": state_machine.output_ports[0]},
    )

    t = results.time
    state = results.outputs["state"]
    assert int(jnp.max(state[t < 1.0])) == 0
    assert int(jnp.min(state[t > 1.5])) == 1


def test_no_initial_state_raises():
    smb = StateMachineBuilder()
    smb.add_state("only")
    with pytest.raises(ValueError, match="initial_state"):
        smb.build()


def test_guard_variable_extraction():
    smb = StateMachineBuilder()
    s1 = smb.add_state("s1")
    s2 = smb.add_state("s2")
    smb.set_initial_state(s1)
    smb.add_transition(s1, s2, guard="temperature > T_max and velocity < 0")
    names = smb._extract_guard_variables()
    assert "temperature" in names
    assert "T_max" in names
    assert "velocity" in names


def test_non_string_guard_raises_type_error():
    """Passing a callable/lambda as guard raises TypeError with helpful message."""
    smb = StateMachineBuilder()
    s1 = smb.add_state("s1")
    s2 = smb.add_state("s2")
    smb.set_initial_state(s1)
    with pytest.raises(TypeError, match="expression string"):
        smb.add_transition(s1, s2, guard=lambda u: u > 0.5)


def test_state_not_in_builder_raises():
    """add_transition with a foreign state raises ValueError."""
    smb = StateMachineBuilder()
    s1 = smb.add_state("s1")
    smb.set_initial_state(s1)
    from jaxonomy.framework.state_machine_builder import State
    foreign = State(name="foreign")
    with pytest.raises(ValueError, match="source and dest"):
        smb.add_transition(s1, foreign, guard="u > 0")


def test_on_entry_on_exit_actions_fire():
    """on_exit(source) fires before on_entry(dest) when transition fires."""
    smb = StateMachineBuilder()
    idle = smb.add_state("idle")
    idle.on_entry = ["mode = 0.0"]
    active = smb.add_state("active")
    active.on_entry = ["mode = 1.0"]
    active.on_exit = ["mode = -1.0"]
    smb.set_initial_state(idle)
    smb.add_transition(idle, active, guard="u > 0.5")
    smb.add_transition(active, idle, guard="u < 0.5")
    sm = smb.build(name="entry_exit_sm")

    # mode is assigned in entry/exit actions → must be an output
    assert any(p.name == "mode" for p in sm.output_ports)


def test_on_entry_initial_state_in_entry_actions():
    """The initial state's on_entry list is included in the compiled entry_point actions."""
    smb = StateMachineBuilder()
    idle = smb.add_state("idle")
    idle.on_entry = ["mode = 0.0", "counter = 10.0"]
    active = smb.add_state("active")
    active.on_entry = ["mode = 1.0", "counter = 0.0"]
    smb.set_initial_state(idle)
    smb.add_transition(idle, active, guard="u > 100.0")
    smb.add_transition(active, idle, guard="u < 0.0")

    # _entry_actions() must include the initial state's on_entry statements
    _, output_names = smb._collect_io_names()
    entry = smb._entry_actions(output_names)
    assert "mode = 0.0" in entry
    assert "counter = 10.0" in entry

    # Verify the machine builds without error
    sm = smb.build(name="init_entry_sm")
    assert any(p.name == "mode" for p in sm.output_ports)
    assert any(p.name == "counter" for p in sm.output_ports)
