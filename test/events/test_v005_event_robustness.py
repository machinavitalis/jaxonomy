# SPDX-License-Identifier: MIT
"""V-005: Event handling robustness.

Test cases for zero-crossing detection, Zeno handling, and near-event behavior in
the Jaxonomy simulation framework. Each test exercises a distinct robustness
property:

- Classical Zeno (bouncing ball with energy loss).
- Multiple independent Zeno systems running in parallel.
- High-frequency switching/chatter near a comparator surface.
- Time-only guards including one that lands exactly on a major-step boundary.
- State-machine zero-crossing transition timing.
- Reset-map sawtooth integrator on threshold crossings.
- Equivalence between Comparator and EdgeDetection rising-edge events.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import (
    DiagramBuilder,
    LeafSystem,
    SimulatorOptions,
    simulate,
)
from jaxonomy.framework.state_machine_builder import StateMachineBuilder
from jaxonomy.library import (
    Adder,
    Clock,
    Comparator,
    Constant,
    EdgeDetection,
    Gain,
    Integrator,
    Sine,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Helper: build a primitives-based bouncing ball using reset-Integrators.
#
# Using `Integrator(enable_reset=True, enable_external_reset=True)` engages the
# framework's built-in Zeno detection: when reset events accumulate faster than
# `zeno_tolerance`, the integrator enters a Zeno-hold state instead of stalling.
# ---------------------------------------------------------------------------
def _build_bouncing_ball_diagram(h0, e, g=9.81, name=""):
    """Return (builder-block dict, position-output port) for a bouncing ball.

    Caller is responsible for `builder.build()`. Returns the position
    integrator so the test can inspect the final state.
    """
    builder = DiagramBuilder()
    accel = builder.add(Constant(-g, name=f"{name}accel"))
    floor = builder.add(Constant(0.0, name=f"{name}floor"))
    vel = builder.add(
        Integrator(
            initial_state=0.0,
            enable_reset=True,
            enable_external_reset=True,
            name=f"{name}vel",
        )
    )
    pos = builder.add(
        Integrator(
            initial_state=h0,
            enable_reset=True,
            enable_external_reset=True,
            name=f"{name}pos",
        )
    )
    impact = builder.add(Comparator(name=f"{name}impact", operator="<"))
    restitution = builder.add(Gain(-e, name=f"{name}rest"))

    builder.connect(accel.output_ports[0], vel.input_ports[0])
    builder.connect(vel.output_ports[0], pos.input_ports[0])
    builder.connect(pos.output_ports[0], impact.input_ports[0])
    builder.connect(floor.output_ports[0], impact.input_ports[1])
    builder.connect(impact.output_ports[0], vel.input_ports[1])
    builder.connect(impact.output_ports[0], pos.input_ports[1])
    builder.connect(vel.output_ports[0], restitution.input_ports[0])
    builder.connect(restitution.output_ports[0], vel.input_ports[2])
    builder.connect(floor.output_ports[0], pos.input_ports[2])
    return builder, pos, vel, impact


# ---------------------------------------------------------------------------
# 1. Bouncing ball with energy loss (classical Zeno).
# ---------------------------------------------------------------------------
def test_bouncing_ball_zeno_comes_to_rest():
    """Bouncing ball with e<1 should reach a quasi-rest state within bounded sim time.

    Expected behavior: total energy decays so the height stays near zero and
    velocity is small.  Without Zeno handling the simulator stalls at the first
    accumulation point.  Built-in Zeno detection (Integrator block) freezes the
    state once events accumulate.
    """
    builder, pos, vel, _ = _build_bouncing_ball_diagram(h0=1.0, e=0.6)
    diagram = builder.build()
    context = diagram.create_context()

    options = SimulatorOptions(rtol=1e-8, atol=1e-10, max_major_steps=2000)
    results = simulate(
        diagram,
        context,
        (0.0, 3.0),
        options=options,
        recorded_signals={
            "pos": pos.output_ports[0],
            "vel": vel.output_ports[0],
        },
    )

    h_final = float(np.asarray(results.outputs["pos"]).reshape(-1)[-1])
    v_final = float(np.asarray(results.outputs["vel"]).reshape(-1)[-1])
    assert h_final >= -1e-3, f"ball sank below floor: {h_final}"
    assert h_final < 5e-2, f"final height {h_final} not near floor"
    assert abs(v_final) < 1.0, f"final speed {v_final} too large"


# ---------------------------------------------------------------------------
# 2. Double bouncing ball (two independent Zeno systems).
# ---------------------------------------------------------------------------
def test_double_bouncing_ball_independent_zeno():
    """Two balls dropped from different heights, each with its own Zeno cascade.

    Both must settle near the floor independently. The simulator's Zeno-hold
    must engage for each ball without interfering with the other.
    """
    # Reuse the helper: build two independent ball sub-diagrams in one diagram.
    builder, pos1, vel1, _ = _build_bouncing_ball_diagram(h0=2.0, e=0.5, name="hi_")
    # Add the second ball into the SAME builder.
    g = 9.81
    accel2 = builder.add(Constant(-g, name="lo_accel"))
    floor2 = builder.add(Constant(0.0, name="lo_floor"))
    vel2 = builder.add(
        Integrator(
            initial_state=0.0,
            enable_reset=True,
            enable_external_reset=True,
            name="lo_vel",
        )
    )
    pos2 = builder.add(
        Integrator(
            initial_state=0.5,
            enable_reset=True,
            enable_external_reset=True,
            name="lo_pos",
        )
    )
    impact2 = builder.add(Comparator(name="lo_impact", operator="<"))
    rest2 = builder.add(Gain(-0.6, name="lo_rest"))
    builder.connect(accel2.output_ports[0], vel2.input_ports[0])
    builder.connect(vel2.output_ports[0], pos2.input_ports[0])
    builder.connect(pos2.output_ports[0], impact2.input_ports[0])
    builder.connect(floor2.output_ports[0], impact2.input_ports[1])
    builder.connect(impact2.output_ports[0], vel2.input_ports[1])
    builder.connect(impact2.output_ports[0], pos2.input_ports[1])
    builder.connect(vel2.output_ports[0], rest2.input_ports[0])
    builder.connect(rest2.output_ports[0], vel2.input_ports[2])
    builder.connect(floor2.output_ports[0], pos2.input_ports[2])

    diagram = builder.build()
    context = diagram.create_context()

    options = SimulatorOptions(rtol=1e-8, atol=1e-10, max_major_steps=4000)
    results = simulate(
        diagram,
        context,
        (0.0, 3.0),
        options=options,
        recorded_signals={
            "h1": pos1.output_ports[0],
            "h2": pos2.output_ports[0],
        },
    )

    h1 = float(np.asarray(results.outputs["h1"]).reshape(-1)[-1])
    h2 = float(np.asarray(results.outputs["h2"]).reshape(-1)[-1])
    assert h1 < 5e-2 and h1 >= -1e-3, f"ball_hi final height {h1}"
    assert h2 < 5e-2 and h2 >= -1e-3, f"ball_lo final height {h2}"


# ---------------------------------------------------------------------------
# 3. Switched system at high switching frequency (chatter).
# ---------------------------------------------------------------------------
def test_high_frequency_switching_completes_in_bounded_steps():
    """A comparator-driven gain switch with a fast sine input crosses zero
    many times.  Verify the simulator finishes the run with the major-step
    budget honored, and that the comparator output flips.
    """
    builder = DiagramBuilder()
    sine = builder.add(Sine(name="sine", amplitude=1.0, frequency=20.0))
    zero = builder.add(Constant(0.0, name="zero"))
    cmp = builder.add(Comparator(name="cmp", operator=">"))
    builder.connect(sine.output_ports[0], cmp.input_ports[0])
    builder.connect(zero.output_ports[0], cmp.input_ports[1])
    # Drive an integrator so the comparator's zero-crossing event is exposed.
    gain = builder.add(Gain(1.0, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(cmp.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()

    # Cap major steps so a runaway / infinite-event loop would fail this test.
    options = SimulatorOptions(
        rtol=1e-6,
        atol=1e-8,
        max_major_step_length=0.05,
        max_major_steps=5000,
    )
    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        options=options,
        recorded_signals={"cmp": cmp.output_ports[0]},
    )

    # Sim must reach (or essentially reach) the requested t_final.
    assert results.context.time >= 1.0 - 1e-6
    # The comparator output must register at least one True (rising) sample.
    assert np.any(np.asarray(results.outputs["cmp"]))
    # And at least one False (falling) sample - i.e., it really chatters.
    assert not np.all(np.asarray(results.outputs["cmp"]))


# ---------------------------------------------------------------------------
# 4. Time-only guard (and exact major-step boundary).
# ---------------------------------------------------------------------------
class TimeGuardLeaf(LeafSystem):
    """LeafSystem with a single zero-crossing on (t - T)."""

    def __init__(self, T, name="tg"):
        super().__init__(name=name)
        self.T = float(T)
        # Boolean state used to record event firing.
        self.declare_continuous_state(default_value=jnp.array([0.0]), ode=self._ode)
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            direction="negative_then_non_negative",
        )

    def _ode(self, time, state, **params):
        return jnp.array([0.0])

    def _guard(self, time, state, **params):
        return time - self.T

    def _reset(self, time, state, **params):
        x = state.continuous_state
        return state.with_continuous_state(x + jnp.array([1.0]))


@pytest.mark.parametrize("T", [0.123, 0.5, 1.0])
def test_time_only_guard_fires_once_at_correct_time(T):
    """Guard `t - T` must fire exactly once at t=T.

    Includes T=1.0 which exact-matches the major-step grid (max_major_step_length
    =0.1).  Counter must increment from 0 to 1, no double-fires, and final time
    >= T.
    """
    leaf = TimeGuardLeaf(T=T)
    context = leaf.create_context()

    options = SimulatorOptions(
        rtol=1e-6, atol=1e-8, max_major_step_length=0.1
    )
    results = simulate(leaf, context, (0.0, 2.0), options=options)
    counter = float(np.asarray(results.context.continuous_state)[0])
    assert counter == pytest.approx(1.0), (
        f"event counter {counter} != 1 for T={T}"
    )


# ---------------------------------------------------------------------------
# 5. State-machine zero-crossing transition.
# ---------------------------------------------------------------------------
def test_state_machine_zero_crossing_transition():
    """A 2-state SM with guard `x > 1.0` driven by a unit-slope ramp must
    transition state -> 1 at t=1.0.  Verify by inspecting recorded mode signal.
    """
    smb = StateMachineBuilder()
    s0 = smb.add_state("s0")
    s1 = smb.add_state("s1")
    smb.set_initial_state(s0)
    smb.add_transition(s0, s1, guard="x > 1.0")
    sm = smb.build(name="threshold_sm")

    builder = DiagramBuilder()
    clock = builder.add(Clock(name="clock"))  # produces t directly == ramp slope 1.
    sm_block = builder.add(sm)
    builder.connect(clock.output_ports[0], sm_block.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 2.0),
        options=SimulatorOptions(rtol=1e-8, atol=1e-10, max_major_steps=400),
        recorded_signals={"state": sm_block.output_ports[0]},
    )

    t = np.asarray(results.time)
    state = np.asarray(results.outputs["state"]).astype(int)
    # Before threshold (well clear of t=1.0) state==0; after, state==1.
    assert int(state[t < 0.9].max()) == 0
    assert int(state[t > 1.1].min()) == 1


# ---------------------------------------------------------------------------
# 6. Reset map sets state to zero on threshold crossing (sawtooth).
# ---------------------------------------------------------------------------
class SawtoothIntegrator(LeafSystem):
    """Integrator that resets x->0 each time x crosses 1.0 from below.

    Driven externally with a unit ramp (du/dt=1), x grows from 0 and resets
    every 1.0 seconds, producing a sawtooth.
    """

    def __init__(self, name="saw"):
        super().__init__(name=name)
        self.declare_continuous_state(default_value=jnp.array([0.0]), ode=self._ode)
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(1),
        )
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            direction="negative_then_non_negative",
        )

    def _ode(self, time, state, **params):
        return jnp.array([1.0])

    def _guard(self, time, state, **params):
        return state.continuous_state[0] - 1.0

    def _reset(self, time, state, **params):
        return state.with_continuous_state(jnp.array([0.0]))


def test_reset_map_sawtooth_on_threshold():
    """Sawtooth integrator: x must reset to 0 at t=1.0, 2.0, 3.0 (each within
    rtol).  Verify by sampling the recorded state immediately after each
    expected reset and checking that x is small."""
    leaf = SawtoothIntegrator()
    context = leaf.create_context()

    options = SimulatorOptions(
        rtol=1e-7, atol=1e-9, max_major_step_length=0.05
    )
    results = simulate(
        leaf,
        context,
        (0.0, 3.5),
        options=options,
        recorded_signals={"x": leaf.output_ports[0]},
    )
    t = np.asarray(results.time)
    x = np.asarray(results.outputs["x"]).reshape(-1)

    # Final time and state.
    assert results.context.time >= 3.5 - 1e-6
    # At t = 3.0 + epsilon the state should have just reset to ~0.
    # Because the trajectory is sawtooth the maximum should be ~ 1.0 each
    # period.  Check by counting resets via the difference signal.
    drops = np.where(np.diff(x) < -0.5)[0]
    assert len(drops) == 3, f"expected 3 resets, got {len(drops)}"
    # Times of the drops should be near 1, 2, 3.
    drop_times = t[drops + 1]
    assert np.allclose(drop_times, [1.0, 2.0, 3.0], atol=5e-3), drop_times


# ---------------------------------------------------------------------------
# 7. EdgeDetection vs Comparator equivalence on rising edges.
# ---------------------------------------------------------------------------
def test_edge_detection_matches_comparator_rising_edges():
    """A sine wave compared to 0 produces a square wave at the comparator output.
    Feeding that through a (rising) EdgeDetection block must yield True samples
    coincident with the comparator's rising transitions.

    We simulate both in one diagram, record both signals, and check that every
    EdgeDetection True coincides with a comparator False->True transition in
    the recorded trace.
    """
    dt = 1e-3  # EdgeDetection sample period
    builder = DiagramBuilder()
    sine = builder.add(Sine(name="sine", amplitude=1.0, frequency=2.0))
    zero = builder.add(Constant(0.0, name="zero"))
    cmp = builder.add(Comparator(name="cmp", operator=">"))
    builder.connect(sine.output_ports[0], cmp.input_ports[0])
    builder.connect(zero.output_ports[0], cmp.input_ports[1])

    edge = builder.add(
        EdgeDetection(name="edge", dt=dt, edge_detection="rising")
    )
    builder.connect(cmp.output_ports[0], edge.input_ports[0])

    # An integrator to expose the zero-crossing events to the simulator.
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    gain = builder.add(Gain(1.0, name="gain"))
    builder.connect(cmp.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    options = SimulatorOptions(rtol=1e-7, atol=1e-9, max_major_steps=4000)
    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        options=options,
        recorded_signals={
            "cmp": cmp.output_ports[0],
            "edge": edge.output_ports[0],
        },
    )

    cmp_out = np.asarray(results.outputs["cmp"]).astype(bool)
    edge_out = np.asarray(results.outputs["edge"]).astype(bool)
    t = np.asarray(results.time)

    # Both signals should record at least one rising event over [0, 1] for f=2 Hz.
    assert np.any(edge_out), "EdgeDetection produced no rising edges"
    # Number of rising edges in cmp should equal number of edge_out True bursts.
    cmp_rises = np.where(np.diff(cmp_out.astype(int)) == 1)[0]
    edge_rises = np.where(edge_out)[0]
    # Allow off-by-one due to discretization of EdgeDetection's dt versus the
    # event-localized comparator output.
    assert abs(len(cmp_rises) - len(edge_rises)) <= 1, (
        f"cmp rises={len(cmp_rises)} edge rises={len(edge_rises)}"
    )

    # Check time alignment: each EdgeDetection True must be within `dt` of a
    # comparator rising transition.
    if len(cmp_rises) and len(edge_rises):
        cmp_times = t[cmp_rises + 1]
        edge_times = t[edge_rises]
        for et in edge_times:
            assert np.min(np.abs(cmp_times - et)) <= 5 * dt, (
                f"edge time {et} far from any cmp rise {cmp_times}"
            )


# ---------------------------------------------------------------------------
# 8. T-027: zeno_tolerance on a custom LeafSystem (bouncing ball).
# ---------------------------------------------------------------------------
class _LeafBouncingBall(LeafSystem):
    """Bouncing ball as a single LeafSystem with a custom zero-crossing.

    State `x = [h, v]` with `dh/dt = v`, `dv/dt = -g`. On impact (`h <= 0`
    while falling), the reset map flips and damps the velocity. Without
    Zeno protection the simulator stalls at the accumulation point near
    t~1.35s for the parameters used here. With `zeno_tolerance=1e-6` on
    `declare_zero_crossing`, the framework engages a per-event Zeno-hold
    (mimicking ``Integrator(enable_reset=True)``).
    """

    def __init__(self, h0=1.0, e=0.6, g=9.81, name="leaf_ball"):
        super().__init__(name=name)
        self.e = float(e)
        self.g = float(g)
        self.declare_continuous_state(
            default_value=jnp.array([float(h0), 0.0]),
            ode=self._ode,
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(2),
            name="state",
        )
        self.declare_zero_crossing(
            guard=self._impact_guard,
            reset_map=self._impact_reset,
            direction="positive_then_non_positive",
            name="impact",
            zeno_tolerance=1e-6,
        )

    def _ode(self, time, state, **params):
        v = state.continuous_state[1]
        return jnp.array([v, -self.g])

    def _impact_guard(self, time, state, **params):
        return state.continuous_state[0]  # h

    def _impact_reset(self, time, state, **params):
        x = state.continuous_state
        h, v = x[0], x[1]
        # Clamp height to floor and reflect velocity with restitution.
        new_x = jnp.array([jnp.maximum(h, 0.0), -self.e * v])
        return state.with_continuous_state(new_x)


def test_leaf_system_zeno_tolerance():
    """T-027: a LeafSystem-defined bouncing ball must complete the run when
    `declare_zero_crossing(zeno_tolerance=1e-6)` is set, and settle near the
    floor.

    Without T-027 this raises ``RuntimeError: Simulator failed to reach
    specified end time`` around t~1.35s.
    """
    leaf = _LeafBouncingBall(h0=1.0, e=0.6)
    context = leaf.create_context()

    options = SimulatorOptions(rtol=1e-8, atol=1e-10, max_major_steps=2000)
    results = simulate(
        leaf,
        context,
        (0.0, 3.0),
        options=options,
        recorded_signals={"x": leaf.output_ports[0]},
    )

    # Simulator must have reached the requested end time.
    assert results.context.time >= 3.0 - 1e-6, (
        f"sim only reached t={float(results.context.time)} (Zeno stall)"
    )

    x_final = np.asarray(results.outputs["x"]).reshape(-1, 2)[-1]
    h_final, v_final = float(x_final[0]), float(x_final[1])
    # Ball settled at/near the floor with small residual velocity.
    assert h_final >= -1e-3, f"ball sank below floor: {h_final}"
    assert h_final < 5e-2, f"final height {h_final} not near floor"
    assert abs(v_final) < 1.0, f"final speed {v_final} too large"


# ---------------------------------------------------------------------------
# 9. T-027a: Zeno protection composes with user discrete state.
# ---------------------------------------------------------------------------
class _BouncingBallWithCounter(LeafSystem):
    """Bouncing ball that ALSO declares its own discrete state (a bounce
    counter). T-027a: the framework must pack the Zeno tracker alongside the
    user's discrete state rather than refusing the declaration.
    """

    def __init__(self, h0=1.0, e=0.6, g=9.81, name="leaf_ball_counter"):
        super().__init__(name=name)
        self.e = float(e)
        self.g = float(g)
        self.declare_continuous_state(
            default_value=jnp.array([float(h0), 0.0]),
            ode=self._ode,
        )
        self.declare_discrete_state(default_value=jnp.int32(0))  # bounce counter
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            direction="positive_then_non_positive",
            name="impact",
            zeno_tolerance=1e-6,
        )

    def _ode(self, time, state, **params):
        v = state.continuous_state[1]
        return jnp.array([v, -self.g])

    def _guard(self, time, state, **params):
        return state.continuous_state[0]

    def _reset(self, time, state, **params):
        h, v = state.continuous_state[0], state.continuous_state[1]
        new_count = state.discrete_state + jnp.int32(1)
        new_xc = jnp.array([jnp.maximum(h, 0.0), -self.e * v])
        return state.with_continuous_state(new_xc).with_discrete_state(new_count)


def test_zeno_composes_with_user_discrete_state():
    """T-027a: a LeafSystem can declare both user discrete state AND a
    Zeno-protected zero-crossing on the same block.

    Confirms: (1) no error at construction time, (2) the simulator runs to
    t_final without stalling, (3) the user's discrete state is preserved and
    advanced by the user's reset map (bounce count > 0), (4) the ball settles
    near the floor.
    """
    leaf = _BouncingBallWithCounter(h0=1.0, e=0.6)
    ctx = leaf.create_context()

    options = SimulatorOptions(rtol=1e-8, atol=1e-10, max_major_steps=2000)
    res = simulate(
        leaf,
        ctx,
        (0.0, 3.0),
        options=options,
        recorded_signals={"x": leaf.output_ports[0]} if leaf.output_ports else None,
    )

    # Sim must reach the requested end time (no Zeno stall).
    assert float(res.context.time) >= 3.0 - 1e-6, (
        f"sim only reached t={float(res.context.time)}"
    )

    # The user's discrete state survives the framework's Zeno packing: the
    # bounce counter must reflect at least one reset firing.
    leaf_state = res.context[leaf.system_id].state
    xd = leaf_state.discrete_state
    # `xd` is the framework's combined wrapper; the user's slot is `xd.user`.
    user_xd = xd.user if hasattr(xd, "user") else xd
    final_count = int(np.asarray(user_xd))
    assert final_count > 0, f"expected at least one bounce, got count={final_count}"

    final_h = float(np.asarray(leaf_state.continuous_state)[0])
    assert final_h >= -1e-3, f"ball below floor: {final_h}"
    assert final_h < 5e-2, f"final height {final_h} not near floor"


# ---------------------------------------------------------------------------
# 10. T-027a-followup: simulator-level Zeno tracker recovery probe.
#
# These tests exercise the SimulatorOptions / SimulatorState plumbing and
# the recovery probe inside ``Simulator._update_zeno_tracking``.  The
# simulator-level Zeno *freeze* gate (i.e. zeroing out ode_rhs while the
# latch is engaged) is not yet implemented at simulator level — that
# remains scope of T-027a (b) cross-leaf Zeno detection.  What ships here
# is the tracker + recovery probe infrastructure; the leaf-level Zeno
# protection from T-027/T-027a continues to handle the actual cascade
# freezes for a single LeafSystem.  The default-off path
# (``zeno_protection_enabled=False``) is byte-equivalent to the
# pre-followup behaviour.
# ---------------------------------------------------------------------------
def test_zeno_recovery_period_default():
    """T-027a-followup: ``SimulatorOptions().zeno_recovery_period`` defaults to 10.

    Also exercises the related defaults for the followup options: the
    simulator-level Zeno tracker is off by default, and the default
    tolerance matches the per-leaf default (1e-6).
    """
    options = SimulatorOptions()
    assert options.zeno_recovery_period == 10
    assert options.zeno_protection_enabled is False
    assert options.zeno_tolerance == 1e-6


def _step_zeno(sim, tprev, active, frozen, triggered, t):
    """Helper: convert tracker outputs to plain Python scalars after each step."""
    tprev, active, frozen = sim._update_zeno_tracking(
        float(tprev), bool(active), int(frozen), triggered, t,
    )
    return float(tprev), bool(active), int(frozen)


def test_zeno_recovery_probe_logic_clears_after_K_frozen_steps():
    """T-027a-followup: ``_update_zeno_tracking`` clears the latch after
    ``zeno_recovery_period`` frozen steps.  Verifies engagement, counter
    progression, the probe firing at K=4, and post-recovery quiescence.
    """
    from jaxonomy.simulation.simulator import Simulator
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-3, zeno_recovery_period=4,
    ))
    # Step 0: dt = 1.0 >> tol — no engagement, tprev advances.
    tprev, active, frozen = _step_zeno(sim, 0.0, False, 0, True, 1.0)
    assert (active, frozen, tprev) == (False, 0, pytest.approx(1.0))
    # Step 1: sub-tolerance retrigger — latch engages.
    tprev, active, frozen = _step_zeno(sim, 1.0, False, 0, True, 1.0 + 1e-7)
    assert (active, frozen) == (True, 1)
    # Steps 2..3: still latched, no triggers — counter increments.
    for expected in (2, 3):
        tprev, active, frozen = _step_zeno(sim, tprev, active, frozen, False, tprev + 1e-7)
        assert (active, frozen) == (True, expected)
    # Step 4: hits K=zeno_recovery_period — probe clears the latch.
    tprev, active, frozen = _step_zeno(sim, tprev, active, frozen, False, tprev + 1e-7)
    assert (active, frozen) == (False, 0), "probe should clear latch at K"
    # Step 5: post-recovery quiescence.
    tprev, active, frozen = _step_zeno(sim, tprev, active, frozen, False, tprev + 1e-7)
    assert (active, frozen) == (False, 0)


def test_zeno_recovery_during_persistent_cascade():
    """T-027a-followup: when the cascade keeps firing inside tolerance,
    the probe clears the latch but the next sub-tolerance trigger re-arms
    it.  Verifies the frozen-step counter never runs away and at least
    one re-engagement is observed.
    """
    from jaxonomy.simulation.simulator import Simulator
    K = 5
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1.0, zeno_recovery_period=K,
    ))
    # Bootstrap a latch: two close triggers.
    tprev, active, frozen = _step_zeno(sim, 0.0, False, 0, True, 0.0)
    tprev, active, frozen = _step_zeno(sim, tprev, active, frozen, True, 0.01)
    assert active is True
    # Persistent cascade: retrigger every K-1 steps so the probe always
    # has a pending guard to re-engage off of.
    re_engagements, max_frozen = 0, frozen
    for i in range(40):
        prev_active = active
        tprev, active, frozen = _step_zeno(
            sim, tprev, active, frozen, (i % (K - 1) == 0), 0.02 + i * 0.01,
        )
        max_frozen = max(max_frozen, frozen)
        re_engagements += int((not prev_active) and active)
    assert max_frozen <= K, f"frozen counter ran away to {max_frozen} > K={K}"
    assert re_engagements >= 1, f"persistent cascade never re-engaged; {re_engagements=}"


def test_zeno_recovery_default_off_byte_equivalent():
    """T-027a-followup: ``zeno_protection_enabled=False`` (the default)
    leaves the bouncing-ball baseline behaviour unchanged.  Same
    invariants as ``test_bouncing_ball_zeno_comes_to_rest`` but with the
    new option explicitly disabled.
    """
    builder, pos, vel, _ = _build_bouncing_ball_diagram(h0=1.0, e=0.6)
    diagram = builder.build()
    options = SimulatorOptions(
        rtol=1e-8, atol=1e-10, max_major_steps=2000,
        zeno_protection_enabled=False,
    )
    results = simulate(
        diagram, diagram.create_context(), (0.0, 3.0), options=options,
        recorded_signals={"pos": pos.output_ports[0], "vel": vel.output_ports[0]},
    )
    h = float(np.asarray(results.outputs["pos"]).reshape(-1)[-1])
    v = float(np.asarray(results.outputs["vel"]).reshape(-1)[-1])
    assert -1e-3 <= h < 5e-2 and abs(v) < 1.0, f"baseline drifted: h={h} v={v}"


# ---------------------------------------------------------------------------
# 11. T-027a-followup-vector-tprev: per-event tprev / per-event active.
#
# These tests exercise the per-event vectorisation of the simulator-level
# Zeno tracker.  The freeze gate that consumes ``zeno_active[i]`` per
# leaf is filed under T-027a-followup-per-leaf-freeze and is NOT shipped
# here — the assertions below are about the carry shape and the per-
# event tolerance check, not about which leaves get frozen.
# ---------------------------------------------------------------------------
def test_per_event_zeno_tprev_state_shape():
    """T-027a-followup-vector-tprev: ``Simulator.initialize`` allocates
    ``zeno_tprev`` and ``zeno_active`` of shape ``(N_events,)`` when
    simulator-level Zeno protection is enabled.

    Builds a diagram with two independent bouncing balls (each with one
    impact zero-crossing) and verifies the carry vectors have shape
    ``(2,)``.  The default-off case must keep the scalar defaults.
    """
    from jaxonomy.simulation.simulator import Simulator
    builder, _, _, _ = _build_bouncing_ball_diagram(h0=1.0, e=0.6, name="hi_")
    g = 9.81
    accel2 = builder.add(Constant(-g, name="lo_accel"))
    floor2 = builder.add(Constant(0.0, name="lo_floor"))
    vel2 = builder.add(Integrator(initial_state=0.0, enable_reset=True,
                                  enable_external_reset=True, name="lo_vel"))
    pos2 = builder.add(Integrator(initial_state=0.5, enable_reset=True,
                                  enable_external_reset=True, name="lo_pos"))
    impact2 = builder.add(Comparator(name="lo_impact", operator="<"))
    rest2 = builder.add(Gain(-0.6, name="lo_rest"))
    builder.connect(accel2.output_ports[0], vel2.input_ports[0])
    builder.connect(vel2.output_ports[0], pos2.input_ports[0])
    builder.connect(pos2.output_ports[0], impact2.input_ports[0])
    builder.connect(floor2.output_ports[0], impact2.input_ports[1])
    builder.connect(impact2.output_ports[0], vel2.input_ports[1])
    builder.connect(impact2.output_ports[0], pos2.input_ports[1])
    builder.connect(vel2.output_ports[0], rest2.input_ports[0])
    builder.connect(rest2.output_ports[0], vel2.input_ports[2])
    builder.connect(floor2.output_ports[0], pos2.input_ports[2])
    diagram = builder.build()

    n_events = diagram.zero_crossing_events.num_events
    assert n_events >= 2, f"expected >=2 zero-crossings, got {n_events}"

    sim_on = Simulator(diagram, options=SimulatorOptions(
        zeno_protection_enabled=True, max_major_steps=100,
    ))
    sim_state_on = sim_on.initialize(diagram.create_context())
    assert np.asarray(sim_state_on.zeno_tprev).shape == (n_events,), (
        f"on: zeno_tprev shape {np.asarray(sim_state_on.zeno_tprev).shape}, "
        f"expected ({n_events},)"
    )
    assert np.asarray(sim_state_on.zeno_active).shape == (n_events,)
    # Initialised to -inf so the first firing is never inside tolerance.
    assert np.all(np.isneginf(np.asarray(sim_state_on.zeno_tprev)))
    assert not np.any(np.asarray(sim_state_on.zeno_active))

    sim_off = Simulator(diagram, options=SimulatorOptions(
        zeno_protection_enabled=False, max_major_steps=100,
    ))
    sim_state_off = sim_off.initialize(diagram.create_context())
    # Default-off path: scalar defaults preserved.
    assert np.asarray(sim_state_off.zeno_tprev).shape == ()
    assert np.asarray(sim_state_off.zeno_active).shape == ()


def test_per_event_zeno_tprev_independent_tolerance_check():
    """T-027a-followup-vector-tprev: ``_update_zeno_tracking`` performs
    the tolerance check on a per-event basis.  Synthetic carry vectors
    with two events: event 0 fires within tolerance of its own tprev;
    event 1 fires outside its own tolerance.  Only event 0's
    ``zeno_active[0]`` should latch.
    """
    from jaxonomy.simulation.simulator import Simulator
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-3, zeno_recovery_period=10,
    ))
    # Two events: event 0 last fired at t=1.0, event 1 last fired at t=0.0.
    tprev_in = jnp.array([1.0, 0.0])
    active_in = jnp.array([False, False])
    triggered_per_event = jnp.array([True, True])  # both fire now
    new_tprev, new_active, new_frozen = sim._update_zeno_tracking(
        tprev_in, active_in, jnp.int32(0), triggered_per_event, 1.0 + 1e-7,
    )
    new_tprev = np.asarray(new_tprev)
    new_active = np.asarray(new_active)
    # Event 0 fired within tolerance (dt = 1e-7 < 1e-3) → latch.
    # Event 1 fired but dt = 1.0 + 1e-7 >> 1e-3 → no latch.
    assert bool(new_active[0]) is True, (
        f"event 0 should latch, got active={new_active}"
    )
    assert bool(new_active[1]) is False, (
        f"event 1 should NOT latch (dt >> tol), got active={new_active}"
    )
    # Both tprev[i] updated to the current time because both fired.
    assert np.allclose(new_tprev, [1.0 + 1e-7, 1.0 + 1e-7])
    # T-027a-followup-per-event-recovery: per-event frozen counter —
    # event 0 latched so its counter increments to 1; event 1 did not
    # latch so its counter stays at 0.
    new_frozen = np.asarray(new_frozen)
    assert int(new_frozen[0]) == 1
    assert int(new_frozen[1]) == 0


def test_per_event_zeno_active_only_for_triggered_events():
    """T-027a-followup-vector-tprev: only events whose ``triggered[i]``
    is True update their own ``tprev[i]``.  Synthetic case: event 0
    fires inside tolerance, event 1 does NOT fire — event 1's
    ``tprev[1]`` must NOT advance.
    """
    from jaxonomy.simulation.simulator import Simulator
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-3, zeno_recovery_period=10,
    ))
    tprev_in = jnp.array([1.0, 0.5])
    active_in = jnp.array([False, False])
    triggered_per_event = jnp.array([True, False])
    new_tprev, new_active, _ = sim._update_zeno_tracking(
        tprev_in, active_in, jnp.int32(0), triggered_per_event, 1.0 + 1e-7,
    )
    new_tprev = np.asarray(new_tprev)
    new_active = np.asarray(new_active)
    # Only event 0 updates its tprev; event 1's tprev[1] is unchanged.
    assert np.isclose(new_tprev[0], 1.0 + 1e-7)
    assert np.isclose(new_tprev[1], 0.5), (
        f"event 1 did not fire, tprev[1] should be unchanged: got {new_tprev}"
    )
    # Only event 0 latches.
    assert bool(new_active[0]) is True
    assert bool(new_active[1]) is False


def test_per_event_zeno_global_recovery_probe_clears_all_events():
    """T-027a-followup-per-event-recovery: the per-event recovery probe
    clears each event's ``active[i]`` independently when its own
    ``frozen[i]`` hits K.  In this test only event 0 is latched, so
    only event 0's counter advances and the probe clears only it —
    event 1 was never latched and stays cleared throughout.
    """
    from jaxonomy.simulation.simulator import Simulator
    K = 3
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-3, zeno_recovery_period=K,
    ))
    # Engage event 0 with a sub-tolerance retrigger.
    tprev = jnp.array([1.0, 0.0])
    active = jnp.array([False, False])
    frozen = jnp.zeros((2,), dtype=jnp.int32)
    tprev, active, frozen = sim._update_zeno_tracking(
        tprev, active, frozen, jnp.array([True, False]), 1.0 + 1e-7,
    )
    assert bool(np.asarray(active)[0]) is True
    assert bool(np.asarray(active)[1]) is False
    # Step forward without triggers — only event 0's counter advances.
    for _ in range(K - 1):
        tprev, active, frozen = sim._update_zeno_tracking(
            tprev, active, frozen, jnp.array([False, False]),
            float(np.asarray(tprev)[0]) + 1e-7,
        )
    # Hit the probe: frozen[0] >= K, only event 0's latch clears.
    frozen = np.asarray(frozen)
    active = np.asarray(active)
    assert int(frozen[0]) == 0  # cleared by the probe
    assert int(frozen[1]) == 0  # never advanced (event 1 never latched)
    assert not np.any(active)


# ---------------------------------------------------------------------------
# 12. T-027a-followup-per-leaf-freeze: simulator-level freeze gate.
#
# These tests exercise the per-leaf freeze gate that consumes
# ``zeno_active[i]`` and the static event->leaf map ``event_system_ids``
# to roll back ONLY the host leaf's continuous state when its events
# enter a Zeno cascade.  The default-off path
# (``zeno_protection_enabled=False``) is byte-equivalent to the
# pre-freeze-gate behaviour.  The previous T-027a (b) commit message
# claimed the freeze gate landed; the actual code was missing —
# ``zeno_active`` was observational on the carry until this followup.
# ---------------------------------------------------------------------------
class _LeafBallNoZenoOptIn(LeafSystem):
    """Bouncing ball as a single LeafSystem WITHOUT ``zeno_tolerance=``
    on its zero-crossing — leaf-level Zeno is OFF so the simulator-level
    freeze gate is the only protection in play.
    """

    def __init__(self, h0=1.0, e=0.6, g=9.81, name="leaf_ball_no_optin"):
        super().__init__(name=name)
        self.e = float(e)
        self.g = float(g)
        self.declare_continuous_state(
            default_value=jnp.array([float(h0), 0.0]),
            ode=self._ode,
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(2),
            name="state",
        )
        self.declare_zero_crossing(
            guard=lambda t, s, **p: s.continuous_state[0],
            reset_map=self._reset,
            direction="positive_then_non_positive",
            name="impact",
            # NOTE: no zeno_tolerance= -- leaf-level Zeno OFF.
        )

    def _ode(self, time, state, **params):
        v = state.continuous_state[1]
        return jnp.array([v, -self.g])

    def _reset(self, time, state, **params):
        h, v = state.continuous_state[0], state.continuous_state[1]
        new_x = jnp.array([jnp.maximum(h, 0.0), -self.e * v])
        return state.with_continuous_state(new_x)


def test_simulator_level_zeno_actually_freezes_continuous_state():
    """T-027a-followup-per-leaf-freeze: the simulator-level gate alone
    (no per-leaf opt-in) must roll back the continuous state when
    ``zeno_active`` is True.

    Drives ``Simulator._apply_per_leaf_zeno_freeze`` directly with a
    synthetic ``zeno_active=True`` carry and verifies that the post-ODE
    state is replaced with the pre-ODE snapshot.  The ``zeno_active=
    False`` case must let the post-ODE state through unchanged.
    """
    from jaxonomy.simulation.simulator import Simulator
    leaf = _LeafBallNoZenoOptIn(h0=1.0, e=0.6)
    ctx = leaf.create_context()
    sim = Simulator(leaf, options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-6, max_major_steps=10,
    ))
    pre_xc = jnp.array([0.5, -1.0])
    post_xc = jnp.array([0.49, -1.0981])  # what an ODE step would advance to
    pre_ctx = ctx.with_continuous_state(pre_xc)
    post_ctx = ctx.with_continuous_state(post_xc)
    pre_solver = sim.ode_solver.initialize(pre_ctx)
    post_solver = sim.ode_solver.initialize(post_ctx)

    # Frozen path: zeno_active=True -> rollback.
    active_on = jnp.ones((sim.n_zero_crossing_events,), dtype=jnp.bool_)
    new_ctx, _ = sim._apply_per_leaf_zeno_freeze(
        pre_xc, post_ctx, pre_solver, post_solver, active_on,
    )
    assert np.allclose(np.asarray(new_ctx.continuous_state), np.asarray(pre_xc)), (
        "freeze gate did not roll back continuous state: "
        f"got {new_ctx.continuous_state}, want {pre_xc}"
    )

    # Unfrozen path: zeno_active=False -> identity (post-ODE state preserved).
    active_off = jnp.zeros((sim.n_zero_crossing_events,), dtype=jnp.bool_)
    new_ctx_off, _ = sim._apply_per_leaf_zeno_freeze(
        pre_xc, post_ctx, pre_solver, post_solver, active_off,
    )
    assert np.allclose(
        np.asarray(new_ctx_off.continuous_state), np.asarray(post_xc)
    ), "freeze gate spuriously rolled back when zeno_active was all False"


def test_simulator_level_zeno_per_leaf_independence():
    """T-027a-followup-per-leaf-freeze: when one leaf's events latch
    Zeno but another leaf's events do NOT, only the latched leaf's
    continuous state is rolled back.  The unlatched leaf's post-ODE
    state advances normally.

    Builds a Diagram of TWO independent LeafSystem bouncing balls
    (each with its own zero-crossing, no per-leaf opt-in).  Constructs
    a synthetic ``zeno_active`` vector with ball A's event True and
    ball B's event False, then drives the freeze gate directly and
    checks that ball A's slice is rolled back while ball B's advances.
    """
    from jaxonomy.simulation.simulator import Simulator
    builder = DiagramBuilder()
    ball_a = builder.add(_LeafBallNoZenoOptIn(h0=1.0, e=0.6, name="ball_a"))
    ball_b = builder.add(_LeafBallNoZenoOptIn(h0=2.0, e=0.5, name="ball_b"))
    diagram = builder.build()
    ctx = diagram.create_context()
    sim = Simulator(diagram, options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-6, max_major_steps=10,
    ))
    # Map system_id -> cs index.  ``_sysid_to_cs_idx`` was built at init.
    a_idx = sim._sysid_to_cs_idx[ball_a.system_id]
    b_idx = sim._sysid_to_cs_idx[ball_b.system_id]
    # Find each ball's event position(s) in the (N_events,) carry.
    a_pos = sim._cs_idx_to_event_positions[a_idx]
    b_pos = sim._cs_idx_to_event_positions[b_idx]
    assert a_pos and b_pos, "expected each ball to own at least one event"

    pre_xc_list = list(ctx.continuous_state)
    pre_a = pre_xc_list[a_idx]
    pre_b = pre_xc_list[b_idx]
    post_a = pre_a + jnp.array([-0.01, -0.0981])  # ODE-advanced ball A
    post_b = pre_b + jnp.array([-0.02, -0.1962])  # ODE-advanced ball B
    post_xc_list = list(pre_xc_list)
    post_xc_list[a_idx] = post_a
    post_xc_list[b_idx] = post_b
    post_ctx = ctx.with_continuous_state(post_xc_list)
    solver_pre = sim.ode_solver.initialize(ctx)
    solver_post = sim.ode_solver.initialize(post_ctx)

    # Latch ball A's events only.
    active = np.zeros(sim.n_zero_crossing_events, dtype=bool)
    for i in a_pos:
        active[i] = True
    active_jax = jnp.asarray(active)
    new_ctx, _ = sim._apply_per_leaf_zeno_freeze(
        pre_xc_list, post_ctx, solver_pre, solver_post, active_jax,
    )
    new_xc = new_ctx.continuous_state
    # Ball A: rolled back to pre.
    assert np.allclose(np.asarray(new_xc[a_idx]), np.asarray(pre_a)), (
        f"ball A not rolled back: got {new_xc[a_idx]}, want {pre_a}"
    )
    # Ball B: post-ODE state preserved (NOT rolled back).
    assert np.allclose(np.asarray(new_xc[b_idx]), np.asarray(post_b)), (
        f"ball B spuriously rolled back: got {new_xc[b_idx]}, want {post_b}"
    )


def test_simulator_level_zeno_freezes_bouncing_ball_without_per_leaf():
    """T-027a-followup-per-leaf-freeze: end-to-end, the simulator-level
    freeze gate alone is sufficient to drive a leaf-Zeno-free bouncing
    ball to t=3.0 without ``RuntimeError: Simulator failed to reach
    specified end time``.

    Uses ``_LeafBallNoZenoOptIn`` (NO ``zeno_tolerance=`` on
    ``declare_zero_crossing``) so the leaf-level path is OFF.  Without
    the simulator-level freeze gate this raises a Zeno-stall around
    t~1.35s; with it engaged the ball settles near the floor.
    """
    leaf = _LeafBallNoZenoOptIn(h0=1.0, e=0.6)
    ctx = leaf.create_context()
    options = SimulatorOptions(
        rtol=1e-8, atol=1e-10, max_major_steps=2000,
        zeno_protection_enabled=True, zeno_tolerance=1e-6,
    )
    results = simulate(
        leaf, ctx, (0.0, 3.0), options=options,
        recorded_signals={"x": leaf.output_ports[0]},
    )
    assert float(results.context.time) >= 3.0 - 1e-6, (
        f"sim only reached t={float(results.context.time)} (Zeno stall)"
    )
    x_final = np.asarray(results.outputs["x"]).reshape(-1, 2)[-1]
    h_final, v_final = float(x_final[0]), float(x_final[1])
    assert h_final >= -1e-3, f"ball sank below floor: {h_final}"
    assert h_final < 5e-2, f"final height {h_final} not near floor"
    assert abs(v_final) < 1.5, f"final speed {v_final} too large"


# ---------------------------------------------------------------------------
# T-027a-followup-per-leaf-solver-state: per-leaf solver-state rollback.
#
# When leaf A is Zeno-frozen and leaf B is not, leaf B's slice of the
# Dopri5 solver state (``y``, ``f``, ``interp_coeff``) must keep its
# advanced post-ODE values; only leaf A's slice rolls back to the pre-
# ODE snapshot.  Global scalar fields (``t``, ``dt``, etc.) advance
# normally — those are integration-step-level, not per-leaf.
# ---------------------------------------------------------------------------
def _two_ball_sim_for_solver_state():
    """Helper: build a two-ball Diagram + Simulator + reusable contexts."""
    from jaxonomy.simulation.simulator import Simulator
    builder = DiagramBuilder()
    ball_a = builder.add(_LeafBallNoZenoOptIn(h0=1.0, e=0.6, name="ball_a"))
    ball_b = builder.add(_LeafBallNoZenoOptIn(h0=2.0, e=0.5, name="ball_b"))
    diagram = builder.build()
    ctx = diagram.create_context()
    sim = Simulator(diagram, options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1e-6, max_major_steps=10,
    ))
    return diagram, ctx, sim, ball_a, ball_b


def test_per_leaf_solver_state_independence():
    """When leaf A is frozen and leaf B is not, leaf B's slice of the
    Dopri5 ``y`` / ``f`` / ``interp_coeff`` advances to its post-ODE
    values exactly (no rollback contamination from A's freeze).
    """
    diagram, ctx, sim, ball_a, ball_b = _two_ball_sim_for_solver_state()
    a_idx = sim._sysid_to_cs_idx[ball_a.system_id]
    b_idx = sim._sysid_to_cs_idx[ball_b.system_id]
    a_pos = sim._cs_idx_to_event_positions[a_idx]
    a_slice = sim._leaf_flat_slices[a_idx]
    b_slice = sim._leaf_flat_slices[b_idx]

    # Build pre / post solver states by perturbing the unraveled state.
    pre_xc_list = list(ctx.continuous_state)
    post_xc_list = list(pre_xc_list)
    post_xc_list[a_idx] = pre_xc_list[a_idx] + jnp.array([-0.01, -0.0981])
    post_xc_list[b_idx] = pre_xc_list[b_idx] + jnp.array([-0.02, -0.1962])
    pre_ctx = ctx
    post_ctx = ctx.with_continuous_state(post_xc_list)
    pre_solver = sim.ode_solver.initialize(pre_ctx)
    post_solver = sim.ode_solver.initialize(post_ctx)

    # Latch ball A's events only.
    active = np.zeros(sim.n_zero_crossing_events, dtype=bool)
    for i in a_pos:
        active[i] = True
    active_jax = jnp.asarray(active)
    _, new_solver = sim._apply_per_leaf_zeno_freeze(
        pre_xc_list, post_ctx, pre_solver, post_solver, active_jax,
    )

    # Ball B's slice of ``y`` must equal ``post_solver.y`` (advanced).
    new_y = np.asarray(new_solver.y)
    post_y = np.asarray(post_solver.y)
    pre_y = np.asarray(pre_solver.y)
    bs_lo, bs_hi = b_slice
    np.testing.assert_allclose(
        new_y[bs_lo:bs_hi], post_y[bs_lo:bs_hi], rtol=1e-12, atol=0.0,
        err_msg="ball B's y slice was contaminated by ball A's freeze",
    )
    # And ball B's slice of ``interp_coeff`` (5, n_y) must equal post.
    new_ic = np.asarray(new_solver.interp_coeff)
    post_ic = np.asarray(post_solver.interp_coeff)
    np.testing.assert_allclose(
        new_ic[..., bs_lo:bs_hi], post_ic[..., bs_lo:bs_hi],
        rtol=1e-12, atol=0.0,
        err_msg="ball B's interp_coeff slice was contaminated by A's freeze",
    )
    # And ball B's slice of ``f`` (the rhs evaluation) must equal post.
    new_f = np.asarray(new_solver.f)
    post_f = np.asarray(post_solver.f)
    np.testing.assert_allclose(
        new_f[bs_lo:bs_hi], post_f[bs_lo:bs_hi], rtol=1e-12, atol=0.0,
        err_msg="ball B's f slice was contaminated by A's freeze",
    )
    # Sanity: ball A's slices must NOT match post (rollback should differ).
    a_lo, a_hi = a_slice
    assert not np.allclose(new_y[a_lo:a_hi], post_y[a_lo:a_hi]), (
        "ball A's y slice unexpectedly matches post-ODE — freeze did not engage"
    )


def test_per_leaf_solver_state_rollback_isolation():
    """Frozen leaf A's slice of ``y`` / ``f`` / ``interp_coeff`` must
    match the pre-ODE snapshot exactly — the freeze gate rolled it
    back, while leaf B advanced.
    """
    diagram, ctx, sim, ball_a, ball_b = _two_ball_sim_for_solver_state()
    a_idx = sim._sysid_to_cs_idx[ball_a.system_id]
    a_pos = sim._cs_idx_to_event_positions[a_idx]
    a_slice = sim._leaf_flat_slices[a_idx]

    pre_xc_list = list(ctx.continuous_state)
    post_xc_list = list(pre_xc_list)
    post_xc_list[a_idx] = pre_xc_list[a_idx] + jnp.array([-0.01, -0.0981])
    post_xc_list[1 - a_idx] = (
        pre_xc_list[1 - a_idx] + jnp.array([-0.02, -0.1962])
    )
    pre_ctx = ctx
    post_ctx = ctx.with_continuous_state(post_xc_list)
    pre_solver = sim.ode_solver.initialize(pre_ctx)
    post_solver = sim.ode_solver.initialize(post_ctx)

    active = np.zeros(sim.n_zero_crossing_events, dtype=bool)
    for i in a_pos:
        active[i] = True
    _, new_solver = sim._apply_per_leaf_zeno_freeze(
        pre_xc_list, post_ctx, pre_solver, post_solver, jnp.asarray(active),
    )

    new_y = np.asarray(new_solver.y)
    pre_y = np.asarray(pre_solver.y)
    a_lo, a_hi = a_slice
    np.testing.assert_allclose(
        new_y[a_lo:a_hi], pre_y[a_lo:a_hi], rtol=1e-12, atol=0.0,
        err_msg="frozen ball A's y slice did not roll back to pre-ODE",
    )
    new_ic = np.asarray(new_solver.interp_coeff)
    pre_ic = np.asarray(pre_solver.interp_coeff)
    np.testing.assert_allclose(
        new_ic[..., a_lo:a_hi], pre_ic[..., a_lo:a_hi],
        rtol=1e-12, atol=0.0,
        err_msg="frozen A's interp_coeff slice did not roll back",
    )


def test_global_scalar_fields_advance_under_per_leaf_freeze():
    """Even when leaf A is frozen, the global integration-step scalar
    fields (``t``, ``dt``, ``n_acc`` etc.) on the solver state must
    advance to their post-step values.  Those are global to the
    adaptive step controller — freezing them would break the main
    loop's progress detection.
    """
    diagram, ctx, sim, ball_a, ball_b = _two_ball_sim_for_solver_state()
    a_idx = sim._sysid_to_cs_idx[ball_a.system_id]
    a_pos = sim._cs_idx_to_event_positions[a_idx]

    pre_xc_list = list(ctx.continuous_state)
    post_xc_list = list(pre_xc_list)
    post_xc_list[a_idx] = pre_xc_list[a_idx] + jnp.array([-0.01, -0.0981])
    post_xc_list[1 - a_idx] = (
        pre_xc_list[1 - a_idx] + jnp.array([-0.02, -0.1962])
    )
    pre_ctx = ctx
    post_ctx = ctx.with_continuous_state(post_xc_list)
    pre_solver = sim.ode_solver.initialize(pre_ctx)
    post_solver = sim.ode_solver.initialize(post_ctx)
    # Synthetically advance scalar step state on post_solver to a
    # distinct value vs pre_solver so we can detect rollback if it
    # incorrectly happens.
    post_solver = dataclasses.replace(
        post_solver,
        t=jnp.asarray(0.123),
        dt=jnp.asarray(0.07),
        n_acc=jnp.asarray(post_solver.n_acc) + 1,
    )

    active = np.zeros(sim.n_zero_crossing_events, dtype=bool)
    for i in a_pos:
        active[i] = True
    _, new_solver = sim._apply_per_leaf_zeno_freeze(
        pre_xc_list, post_ctx, pre_solver, post_solver, jnp.asarray(active),
    )

    # Scalar / step fields stay at the post-step value.
    np.testing.assert_allclose(float(new_solver.t), 0.123, rtol=1e-12)
    np.testing.assert_allclose(float(new_solver.dt), 0.07, rtol=1e-12)
    assert int(new_solver.n_acc) == int(post_solver.n_acc), (
        "n_acc was rolled back — should advance globally"
    )


# ---------------------------------------------------------------------------
# 14. T-027a-followup-per-event-recovery: per-event recovery probe.
#
# The recovery counter ``zeno_frozen_steps`` is now per-event.  Each
# event's counter advances only while THAT event's latch is engaged
# and clears only its own latch on probe.  A still-cascading event
# A no longer gets a free probe just because event B's cascade ended.
# ---------------------------------------------------------------------------
def test_per_event_recovery_independent_clears():
    """T-027a-followup-per-event-recovery: with two events, A latched
    and never re-triggered (its counter increments each step toward K)
    while B is re-triggered every step inside tolerance (stays latched
    with counter pinned), event A's probe fires at K and clears ONLY
    ``active[A]``.  Event B stays latched and its counter keeps tracking
    its own life.

    Synthetic carry seeded so A is closer to its probe than B; this
    isolates the per-event independence from any global synchronisation.
    """
    from jaxonomy.simulation.simulator import Simulator
    K = 4
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1.0, zeno_recovery_period=K,
    ))
    # Seed: both events latched, but A's counter is ahead of B's so A
    # hits K first.  tprev seeded so further "no-trigger" steps don't
    # touch tprev (engage gate stays False without a trigger).
    tprev = jnp.array([10.0, 10.0])
    active = jnp.array([True, True])
    frozen = jnp.array([K - 2, 0], dtype=jnp.int32)

    # B is re-triggered every step inside tolerance (B's counter pins
    # against the per-step probe by clearing+re-engaging in lockstep);
    # A is never re-triggered (its counter walks freely toward K).
    a_seen_clear = False
    b_latched_when_a_cleared = None
    t = 11.0
    for step in range(K + 2):
        tprev, active, frozen = sim._update_zeno_tracking(
            tprev, active, frozen,
            jnp.array([False, True]),  # only event B re-triggers
            t,
        )
        active = np.asarray(active)
        frozen = np.asarray(frozen)
        if not bool(active[0]):
            a_seen_clear = True
            b_latched_when_a_cleared = bool(active[1])
            # When A clears, its counter must also reset.
            assert int(frozen[0]) == 0, (
                f"event A counter not reset on probe: {frozen}"
            )
            break
        t += 1e-7
    assert a_seen_clear, (
        f"event A's per-event probe never fired in {K + 2} steps; frozen={frozen}"
    )
    # B's latch must NOT have been cleared by A's probe — independent.
    assert b_latched_when_a_cleared is True, (
        f"event B's latch leaked-cleared by A's probe (active={active})"
    )


def test_per_event_recovery_does_not_leak_to_other_events():
    """T-027a-followup-per-event-recovery: when event A's recovery probe
    fires, event B's ``frozen[B]`` must not be touched.  Constructs a
    state where A is on the brink of probing (counter at K-1) and B is
    latched with a small counter; one step that does NOT advance A or
    B's latches lets the probe fire — only A's counter resets.
    """
    from jaxonomy.simulation.simulator import Simulator
    K = 5
    sim = Simulator(_LeafBouncingBall(h0=1.0, e=0.6), options=SimulatorOptions(
        zeno_protection_enabled=True, zeno_tolerance=1.0, zeno_recovery_period=K,
    ))
    # Synthetic carry: A latched with frozen[A]=K-1, B latched with frozen[B]=2.
    tprev = jnp.array([10.0, 10.0])
    active = jnp.array([True, True])
    frozen = jnp.array([K - 1, 2], dtype=jnp.int32)
    # No triggers — both still latched, A's counter advances to K → probe.
    tprev, active, frozen = sim._update_zeno_tracking(
        tprev, active, frozen, jnp.array([False, False]), 11.0,
    )
    active = np.asarray(active)
    frozen = np.asarray(frozen)
    # A's probe fired: latch cleared, counter reset.
    assert bool(active[0]) is False
    assert int(frozen[0]) == 0
    # B is unaffected: latch stays, counter advances by exactly one.
    assert bool(active[1]) is True
    assert int(frozen[1]) == 3, (
        f"event B's counter leaked from A's probe: {frozen}"
    )


def test_simulator_level_zeno_per_event_recovery_e2e():
    """T-027a-followup-per-event-recovery: end-to-end on a two-ball
    Diagram.  Each ball has its own zero-crossing event; with simulator-
    level Zeno enabled, the simulation must reach t_end without stalling
    on either ball's cascade — the per-event recovery probe lets each
    leaf clear its latch independently.
    """
    builder = DiagramBuilder()
    builder.add(_LeafBallNoZenoOptIn(h0=1.0, e=0.6, name="ball_a"))
    builder.add(_LeafBallNoZenoOptIn(h0=0.5, e=0.4, name="ball_b"))
    diagram = builder.build()
    ctx = diagram.create_context()
    options = SimulatorOptions(
        rtol=1e-8, atol=1e-10, max_major_steps=4000,
        zeno_protection_enabled=True, zeno_tolerance=1e-6,
    )
    results = simulate(diagram, ctx, (0.0, 3.0), options=options)
    # Both balls must have advanced past their cascade region.
    assert float(results.context.time) >= 3.0 - 1e-6, (
        f"sim only reached t={float(results.context.time)} — per-event "
        f"recovery did not let one of the leaves drain"
    )


# ---------------------------------------------------------------------------
# 15. T-027a-followup-multi-leaf-cascade: two genuinely Zeno-prone leaves
# cascading at DIFFERENT times under simulator-level Zeno protection only.
#
# The honest-scope deferred under T-027a-followup-per-leaf-freeze flagged
# concurrent multi-leaf cascades as fragile because ``int_tf`` is shared
# across the diagram.  T-027a-followup-multi-leaf-cascade-architecture
# (2026-05-01) resolved the underlying staggered-cascade interaction
# (per-event ``triggered`` plumbing + direction-aware ``w0`` nudge +
# active-latch trigger mask).  T-027a-followup-multi-leaf-cascade-test-
# physics then retuned ball B's parameters in the e2e test below so its
# natural cascade accumulation falls inside the 5s integration window
# (the previous ``e=0.8, g=1.62`` choice put the accumulation point at
# t~10s and was only "passing" before the architecture fix because of an
# unrelated scalar-broadcast bug that spuriously latched ball B).  Both
# tests in this section now run as strict-pass.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_simulator_level_zeno_multi_leaf_cascade_e2e():
    """Two independent bouncing balls (different ``e``, different ``g``)
    in one Diagram, both cascading into Zeno at different times without
    per-leaf opt-in.  Verifies the simulator-level freeze gate handles
    per-leaf cascades correctly end-to-end over a 5-second window.

    Parameters chosen so each ball's natural Zeno-cascade accumulation
    time ``t_total = (1+e)/(1-e) * sqrt(2*h0/g)`` lies inside the
    integration window:

      - Ball A (``h0=1, e=0.5, g=9.81``):  ``t_total ~ 1.35s``
      - Ball B (``h0=1, e=0.7, g=4.0``):   ``t_total ~ 4.0s``

    The cascades therefore onset roughly 2.7s apart, exercising the
    time-staggered per-leaf freeze path: when ball A's latch engages
    ball B is still bouncing, and ball B's latch only engages once its
    own bounces collapse to sub-``zeno_tolerance`` intervals.  Both
    balls must reach ``t=5`` settled near the floor.

    Without the per-leaf freeze gate this would raise ``RuntimeError:
    Simulator failed to reach specified end time`` near the first
    cascade onset.  With the gate engaged, both balls settle near the
    floor.
    """
    import time as _time

    builder = DiagramBuilder()
    # Ball A: low restitution, terrestrial gravity -> cascades earlier
    # (analytical accumulation t ~ 3 * sqrt(2/9.81) ~ 1.35s).
    ball_a = builder.add(_LeafBallNoZenoOptIn(h0=1.0, e=0.5, g=9.81, name="ball_a"))
    # Ball B: higher restitution, reduced gravity -> cascades later but
    # still within the 5s window (analytical accumulation t ~
    # (1+0.7)/(1-0.7) * sqrt(2/4.0) ~ 4.0s).  Both `e` and `g` differ
    # from ball A so the two leaves cascade at distinctly different
    # times.
    ball_b = builder.add(_LeafBallNoZenoOptIn(h0=1.0, e=0.7, g=4.0, name="ball_b"))
    diagram = builder.build()

    options = SimulatorOptions(
        rtol=1e-7, atol=1e-9,
        max_major_steps=4000,
        zeno_protection_enabled=True,  # simulator-level only
        zeno_tolerance=1e-6,
    )
    ctx = diagram.create_context()

    t0 = _time.perf_counter()
    res = simulate(diagram, ctx, (0.0, 5.0), options=options)
    wall = _time.perf_counter() - t0

    # 1. Both balls reach the requested end time (no Zeno stall).
    assert float(res.context.time) >= 5.0 - 1e-6, (
        f"sim ended at t={float(res.context.time)} — multi-leaf cascade "
        f"did not let both leaves drain"
    )

    # 2. Both balls settled near the floor.
    final_xc_a = np.asarray(res.context[ball_a.system_id].continuous_state)
    final_xc_b = np.asarray(res.context[ball_b.system_id].continuous_state)
    final_h_a = float(final_xc_a[0])
    final_h_b = float(final_xc_b[0])
    assert -1e-3 <= final_h_a < 0.05, f"ball_a not settled: h={final_h_a}"
    assert -1e-3 <= final_h_b < 0.05, f"ball_b not settled: h={final_h_b}"

    # 3. Wall-time sanity bound.  First-call JIT compile dominates so
    # the bound is generous — what we want to rule out is unbounded
    # growth, not a tight perf regression.
    assert wall < 60.0, f"multi-leaf cascade sim took {wall:.2f}s (expected < 60s)"


@pytest.mark.slow
def test_simulator_level_zeno_staggered_multi_leaf_cascade():
    """Two balls released from different initial heights so ball A
    enters Zeno well before ball B.  Records the trajectory of each
    ball's height; verifies that when ball A first stops decreasing
    (Zeno onset), ball B is still in flight (height significantly
    above the floor).  After ball A's recovery, both eventually
    reach rest.

    This exercises per-leaf isolation under TIME-STAGGERED cascades:
    ball A's freeze must not synchronise / drag ball B with it.
    """
    builder = DiagramBuilder()
    # Ball A: drops from low height under terrestrial gravity -> hits
    # floor and cascades around t ~ 0.45s with e=0.5.
    ball_a = builder.add(_LeafBallNoZenoOptIn(h0=1.0, e=0.5, g=9.81, name="ball_a"))
    # Ball B: drops from much higher under terrestrial gravity -> first
    # impact much later, cascade onset shifted to roughly t ~ 1.4s.
    ball_b = builder.add(_LeafBallNoZenoOptIn(h0=10.0, e=0.5, g=9.81, name="ball_b"))
    diagram = builder.build()

    options = SimulatorOptions(
        rtol=1e-7, atol=1e-9,
        max_major_steps=4000,
        zeno_protection_enabled=True,
        zeno_tolerance=1e-6,
    )
    ctx = diagram.create_context()
    res = simulate(
        diagram, ctx, (0.0, 5.0), options=options,
        recorded_signals={
            "xa": ball_a.output_ports[0],
            "xb": ball_b.output_ports[0],
        },
    )

    # Both balls reach the end time.
    assert float(res.context.time) >= 5.0 - 1e-6, (
        f"sim only reached t={float(res.context.time)}"
    )

    t = np.asarray(res.time)
    xa = np.asarray(res.outputs["xa"]).reshape(-1, 2)
    xb = np.asarray(res.outputs["xb"]).reshape(-1, 2)
    ha = xa[:, 0]
    hb = xb[:, 0]

    # Find ball A's Zeno-onset proxy: the first index where ball A's
    # height is at the floor (within tol) and stops decreasing.  Look
    # for the first time ha drops below 1e-2 — that's near the cascade
    # accumulation point for h0=1.0, e=0.5, g=9.81.
    near_floor_a = np.where(ha < 1e-2)[0]
    assert near_floor_a.size > 0, "ball A never reached near-floor"
    a_onset_idx = int(near_floor_a[0])
    a_onset_t = float(t[a_onset_idx])

    # Ball A's cascade should onset well before t=1.0 with these params
    # (analytical accumulation time for h0=1, e=0.5, g=9.81 is < 1.4s).
    assert a_onset_t < 1.5, (
        f"ball A cascade onset unexpectedly late at t={a_onset_t}"
    )

    # At ball A's onset, ball B must still be in flight (significantly
    # above the floor) — h0_B=10 means ball B's first impact is around
    # t ~ sqrt(2 * 10 / 9.81) ~ 1.43s, so at ball A's onset (< 1.5s)
    # ball B is either still falling or just after first bounce.
    hb_at_a_onset = float(hb[a_onset_idx])
    assert hb_at_a_onset > 1.0, (
        f"ball B should still be in flight at ball A's Zeno onset "
        f"(t={a_onset_t}), but h_B={hb_at_a_onset}"
    )

    # By end of sim, both balls are settled near the floor.
    assert -1e-3 <= float(ha[-1]) < 0.05, (
        f"ball A not settled at t_end: h={float(ha[-1])}"
    )
    assert -1e-3 <= float(hb[-1]) < 0.05, (
        f"ball B not settled at t_end: h={float(hb[-1])}"
    )
