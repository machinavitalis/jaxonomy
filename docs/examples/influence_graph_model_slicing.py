"""Quantitative model slicing and path attribution on a DC-motor speed loop.

A boolean dependency graph answers *whether* a block can affect an output. On a
closed loop that makes "what matters to the shaft speed?" answerable only with
"everything" — true, and useless. ``jaxonomy.analysis.influence_graph`` weights
every edge of that same graph with the exact local Jacobian, so the question
becomes quantitative.

The model is a brushed DC motor under PI speed control with a voltage-limited
driver, built entirely from library primitives so every physical term is its own
block and every signal its own node:

    ref --(+)--> [kp, ki/s] --> driver limit --> back-EMF summing junction
                                                            |
      w <-- 1/J integrator <-- torque balance <-- Kt <-- di/dt integrator
      |                              ^
      +-- speed feedback             +-- load torque, viscous drag, and a
                                         token stiction term

What this script demonstrates and verifies as it runs:

1. The edge weights are the real derivatives — checked against central
   differences taken through the whole diagram, not block by block.
2. Path attribution reproduces the analytic end-to-end sensitivity (-1/J for
   the load-torque path into the shaft acceleration).
3. A quantitative slice is strictly smaller than the boolean one: the stiction
   term is real structure carrying no influence. Because ``tau`` sets the
   frequency the question is asked at, the slice is also swept over ``tau``.
4. A saturated driver makes the *local* derivative zero, which the honest
   answer labels rather than hides: ``dead_edges`` finds it, ``probe=`` shows
   the connection is alive over a finite step, and trajectory mode shows the
   edge switching on once the loop leaves the rail.
5. The same graph serializes into a token-budgeted, citable context block.

Run: ``python docs/examples/influence_graph_model_slicing.py``
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy.analysis import influence_graph, influence_subgraph
from jaxonomy.library import Adder, Constant, Gain, Integrator, Saturate
from jaxonomy.simulation import SimulatorOptions


# --- Motor and controller parameters (a small hobby-scale brushed DC motor) --
R_ARM = 1.2       # armature resistance [ohm]
L_ARM = 2.4e-3    # armature inductance [H]
K_T = 0.042       # torque constant [N*m/A]
K_E = 0.042       # back-EMF constant [V*s/rad]
J_ROT = 1.1e-4    # rotor + load inertia [kg*m^2]
B_VISC = 3.0e-5   # viscous damping [N*m*s/rad]
B_STICTION = 1e-8 # a token stiction-compensation gain: real structure, no influence
V_LIMIT = 12.0    # driver voltage rail [V]
T_LOAD = 0.010    # constant load torque [N*m]
W_REF = 220.0     # speed setpoint [rad/s]
KP, KI = 0.02, 0.6

# tau is the time scale a state-rate edge represents (see the module docstring of
# jaxonomy.analysis.influence): a weight through an integrator means "influence
# accumulated over tau seconds", and a path crossing k integrators scales as
# tau**k — exactly that path's transfer magnitude at omega = 1/tau. So tau
# chooses the time scale the question is asked on, and it is a real choice on a
# stiff model like this one: the electrical loop's L/R is ~2 ms while the
# mechanical J/b is ~3.7 s. The closed loop settles in a few hundred
# milliseconds, so that is the scale the interesting question lives on.
TAU_ELECTRICAL = L_ARM / R_ARM
TAU_MECHANICAL = J_ROT / B_VISC
TAU = 0.1

# Relative step for the secant cross-check in section 5. It has to be large
# enough to leave the saturated region — the 44 V command must come back under
# the 12 V rail, so 0.5x the signal is not enough and 0.9x is. There is no
# universal probe size; it depends on how far into saturation the model sits.
PROBE = 0.9


def make_motor_loop(v_limit=V_LIMIT, kp=KP, w0=0.0, i0=0.0):
    """PI-controlled DC motor speed loop, one block per physical term."""
    builder = jaxonomy.DiagramBuilder()

    reference = builder.add(Constant(W_REF, name="w_ref"))
    speed_error = builder.add(Adder(2, operators="+-", name="speed_error"))

    # PI from primitives rather than the PID block: the block's derivative
    # filter carries a state that is identically zero when kd = 0, and a
    # relative weight normalized by a signal that is always zero is governed by
    # scale_floor rather than by the model (graph.nodes_at_scale_floor() reports
    # exactly this condition).
    proportional = builder.add(Gain(kp, name="kp"))
    integral_gain = builder.add(Gain(KI, name="ki"))
    integral = builder.add(Integrator(0.0, name="integral"))
    command = builder.add(Adder(2, operators="++", name="v_cmd"))
    driver = builder.add(
        Saturate(upper_limit=v_limit, lower_limit=-v_limit, name="driver")
    )

    # Electrical: L·di/dt = V - R·i - Ke·w
    resistive_drop = builder.add(Gain(R_ARM, name="R_drop"))
    back_emf = builder.add(Gain(K_E, name="back_emf"))
    voltage_balance = builder.add(Adder(3, operators="+--", name="v_balance"))
    inductance = builder.add(Gain(1.0 / L_ARM, name="inv_L"))
    current = builder.add(Integrator(i0, name="current"))

    # Mechanical: J·dw/dt = Kt·i - b·w - T_load - stiction_term
    motor_torque = builder.add(Gain(K_T, name="Kt"))
    drag = builder.add(Gain(B_VISC, name="drag"))
    stiction = builder.add(Gain(B_STICTION, name="stiction"))
    load = builder.add(Constant(T_LOAD, name="load_torque"))
    torque_balance = builder.add(Adder(4, operators="+---", name="torque_balance"))
    inertia = builder.add(Gain(1.0 / J_ROT, name="inv_J"))
    speed = builder.add(Integrator(w0, name="speed"))

    # Control path
    builder.connect(reference.output_ports[0], speed_error.input_ports[0])
    builder.connect(speed.output_ports[0], speed_error.input_ports[1])
    builder.connect(speed_error.output_ports[0], proportional.input_ports[0])
    builder.connect(speed_error.output_ports[0], integral_gain.input_ports[0])
    builder.connect(integral_gain.output_ports[0], integral.input_ports[0])
    builder.connect(proportional.output_ports[0], command.input_ports[0])
    builder.connect(integral.output_ports[0], command.input_ports[1])
    builder.connect(command.output_ports[0], driver.input_ports[0])

    # Electrical loop
    builder.connect(driver.output_ports[0], voltage_balance.input_ports[0])
    builder.connect(current.output_ports[0], resistive_drop.input_ports[0])
    builder.connect(resistive_drop.output_ports[0], voltage_balance.input_ports[1])
    builder.connect(speed.output_ports[0], back_emf.input_ports[0])
    builder.connect(back_emf.output_ports[0], voltage_balance.input_ports[2])
    builder.connect(voltage_balance.output_ports[0], inductance.input_ports[0])
    builder.connect(inductance.output_ports[0], current.input_ports[0])

    # Mechanical loop
    builder.connect(current.output_ports[0], motor_torque.input_ports[0])
    builder.connect(motor_torque.output_ports[0], torque_balance.input_ports[0])
    builder.connect(speed.output_ports[0], drag.input_ports[0])
    builder.connect(drag.output_ports[0], torque_balance.input_ports[1])
    builder.connect(load.output_ports[0], torque_balance.input_ports[2])
    builder.connect(speed.output_ports[0], stiction.input_ports[0])
    builder.connect(stiction.output_ports[0], torque_balance.input_ports[3])
    builder.connect(torque_balance.output_ports[0], inertia.input_ports[0])
    builder.connect(inertia.output_ports[0], speed.input_ports[0])

    return builder.build(name="motor_loop")


def step_response(diagram, context, t_end=1.5):
    return jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_end),
        options=SimulatorOptions(rtol=1e-8, atol=1e-10),
        recorded_signals={
            "w": diagram["speed"].output_ports[0],
            "v_cmd": diagram["v_cmd"].output_ports[0],
        },
    )


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# 1. The graph
# ---------------------------------------------------------------------------

banner("1. The influence graph")

diagram = make_motor_loop()
context = diagram.create_context()
results = step_response(diagram, context)

# Trajectory mode is the default here for a reason worth stating: relative
# weights divide by each signal's magnitude, and at any single instant of a
# control loop some signal is passing through zero — the error at steady state,
# every derivative once settled. Trajectory mode normalizes by the largest
# magnitude each signal reaches over the run instead, which is both stable and
# the right denominator for "how much does this matter over the manoeuvre".
graph = influence_graph(
    diagram, context, at="trajectory", results=results, n_snapshots=8, tau=TAU
)
print(graph.summary())


banner("2. Edge weights vs central differences through the whole diagram")

# Raw (un-normalized) weights are literal partial derivatives, so they can be
# compared directly against a finite difference of the assembled model.
raw = influence_graph(diagram, context, normalize="none", tau=1.0)


def shaft_acceleration(load_value):
    """dw/dt for a perturbed load torque, evaluated through the full diagram.

    The perturbation goes in where the load signal arrives, so the whole
    downstream chain (torque balance, 1/J, the integrator's ODE) is re-evaluated
    for real rather than block by block.
    """
    load_input = diagram["torque_balance"].input_ports[2]
    with load_input.fixed(jnp.asarray(load_value)):
        rate = diagram["speed"].eval_time_derivatives(context)
    return float(np.asarray(rate).reshape(()))


step = 1e-7
reference = (
    shaft_acceleration(T_LOAD + step) - shaft_acceleration(T_LOAD - step)
) / (2 * step)
attribution = raw.attribute("speed:xc", "load_torque:out:out_0")
print(f"analytic  d(dw/dt)/d(T_load) = -1/J        = {-1.0 / J_ROT:+.6f}")
print(f"finite difference through the diagram      = {reference:+.6f}")
print(f"influence-graph path attribution           = {attribution.total:+.6f}")
print(f"paths found: {len(attribution.paths)}")
print(
    "relative agreement with finite differences = "
    f"{abs(attribution.total - reference) / abs(reference):.2e}"
)


# ---------------------------------------------------------------------------
# 3. Quantitative slicing vs boolean slicing
# ---------------------------------------------------------------------------

banner("3. What actually matters to the shaft speed")

structural = graph.structural_slice("speed:xc")
quantitative = graph.slice("speed:xc", threshold=0.01)
dropped = sorted(set(structural) - set(quantitative.blocks))
print(f"boolean slice      ({len(structural):2d} blocks): {', '.join(structural)}")
print(
    f"1% influence slice ({len(quantitative.blocks):2d} blocks): "
    f"{', '.join(quantitative.blocks)}"
)
print(f"dropped: {', '.join(dropped) if dropped else '(none)'}")
print(
    "\nEverything in a closed loop is structurally upstream of everything else, so\n"
    "the boolean answer is the whole model — correct and useless. The weighted\n"
    "answer drops the stiction-compensation term: at 1e-8 against 3e-5 of viscous\n"
    "drag it is real structure carrying no influence, which is the quantitative\n"
    "form of a dead-store warning."
)

print(
    "\nThe time scale is part of the question, not a nuisance parameter. Because a\n"
    "path crossing k integrators scales as tau**k, shrinking tau is asking about\n"
    "higher frequencies, where the slow mechanical terms stop mattering:"
)
for tau in (TAU_MECHANICAL, 0.1, 0.01, TAU_ELECTRICAL):
    scaled = influence_graph(
        diagram, context, at="trajectory", results=results, n_snapshots=8, tau=tau
    )
    kept = scaled.slice("speed:xc", threshold=0.01)
    lost = sorted(set(structural) - set(kept.blocks))
    print(
        f"  tau = {tau:<7.4g} s  (omega = {1.0 / tau:>8.4g} rad/s): "
        f"{len(kept.blocks):2d} of {scaled.n_blocks} blocks carry >=1%"
        f"   dropped: {', '.join(lost) if lost else '(none)'}"
    )
print(
    "An absolute threshold reads as a percentage only when tau is comparable to\n"
    "the time constants on the path. InfluenceGraph.relative_threshold() scales\n"
    "the cutoff to the strongest contributor when that is the comparison you want."
)


banner("4. Dominant paths and bottlenecks")


def blocks_along(path):
    """Collapse a signal-level path to the block sequence it visits."""
    visited = []
    for node in path["nodes"]:
        block = graph.graph.nodes[node]["block"]
        if not visited or visited[-1] != block:
            visited.append(block)
    return " -> ".join(visited)


for entry in graph.dominant_paths("speed:xc", k=4):
    print(f"  {entry['product']:+.4g}  {blocks_along(entry)}")
print(
    "\nThe strongest routes differ in whether they pass through the integral term\n"
    "or the proportional one — a distinction a boolean graph cannot draw at all."
)
print("\nbottleneck blocks (every influential path passes through these):")
bottleneck_blocks = sorted(
    {
        graph.graph.nodes[node]["block"]
        for node in graph.bottlenecks("speed:xc", threshold=0.01)
    }
)
print(f"  {', '.join(bottleneck_blocks) if bottleneck_blocks else '(none)'}")


# ---------------------------------------------------------------------------
# 5. The saturated case: where a local derivative is honestly not enough
# ---------------------------------------------------------------------------

banner("5. A saturated driver, and two honest answers")

# A more aggressive proportional gain: from rest the command is kp*220 = 44 V
# against a 12 V rail, so the driver starts hard against its limit and its local
# derivative there is exactly zero.
saturated = make_motor_loop(kp=0.2)
saturated_context = saturated.create_context()
saturated_results = step_response(saturated, saturated_context)

local = influence_graph(saturated, saturated_context, tau=TAU)
probed = influence_graph(saturated, saturated_context, tau=TAU, probe=PROBE)

local_slice = local.slice("speed:xc", 0.01)
probed_slice = probed.slice("speed:xc", 0.01)
print(
    f"exact local derivative : {len(local_slice.blocks):2d} of {local.n_blocks} "
    f"blocks influence the shaft speed"
)
print(f"                         {', '.join(local_slice.blocks)}")
print(
    f"with probe={PROBE}         : {len(probed_slice.blocks):2d} of "
    f"{probed.n_blocks} blocks"
)
recovered = sorted(set(probed_slice.blocks) - set(local_slice.blocks))
print(f"                         recovered: {', '.join(recovered)}")
print(f"boolean slice          : {len(local.structural_slice('speed:xc')):2d} blocks")
print("\ndead edges the local derivative reports:")
for entry in local.dead_edges():
    print(f"  {entry['src']} -> {entry['dst']}  ({entry['kind']})")
print(
    "\nBoth slices answer a real question. The driver is hard against its rail, so\n"
    "the *derivative* of everything upstream of it truly is zero, and the boolean\n"
    "slice's claim that the setpoint matters is equally true over a finite step —\n"
    "which is what the secant probe recovers. The block is flagged hybrid either\n"
    "way: its Jacobian describes the mode it is in and nothing about the others."
)
print(f"  driver hybrid = {local.graph.nodes['driver:out:out_0']['hybrid']}")


banner("6. Trajectory-resolved weights")

speeds = np.asarray(saturated_results.outputs["w"])
commands = np.asarray(saturated_results.outputs["v_cmd"])
print(f"shaft speed  : {speeds[0]:.1f} -> {speeds[-1]:.1f} rad/s over 1.5 s")
print(
    f"PI command   : {commands[0]:.1f} -> {commands[-1]:.1f} V "
    f"against a {V_LIMIT:.0f} V rail"
)

over_time = influence_graph(
    saturated,
    saturated_context,
    at="trajectory",
    results=saturated_results,
    n_snapshots=7,
    tau=TAU,
)
driver_edge = over_time.graph.edges["driver:in:in_0", "driver:out:out_0"]
print(f"snapshot times   : {np.array2string(over_time.times, precision=3)}")
print(f"driver edge      : {np.array2string(driver_edge['profile'], precision=3)}")
print(
    "\nThe edge is zero while the driver is saturated and non-zero once the loop\n"
    "comes off the rail. One number per edge cannot express that; a profile can.\n"
    "reduce='max' is the default because it never hides an influence that appears\n"
    "somewhere on the trajectory:"
)
print(
    f"  1% slice from the trajectory: "
    f"{len(over_time.slice('speed:xc', 0.01).blocks)} of {over_time.n_blocks} blocks"
)


# ---------------------------------------------------------------------------
# 7. Serializing a bounded neighbourhood
# ---------------------------------------------------------------------------

banner("7. A budgeted, citable context block")

serialized = influence_subgraph(graph, "speed:xc", budget_tokens=400, hops=6)
print(serialized["text"])
print(
    f"\nestimated tokens: {serialized['estimated_tokens']} "
    f"(budget 400), edges dropped to fit: {len(serialized['dropped_edges'])}"
)
print(
    "Edges are spent strongest-first, so what the budget drops is what mattered\n"
    "least. Every line is keyed by a block-path + port-name id that survives a\n"
    "rebuild of the model, so an answer citing 'speed:xc' can be checked."
)


banner("Caveats worth stating plainly")
print(
    f"- A weight through an integrator is scaled by tau (here {TAU:.4g} s), and a\n"
    "  path crossing k integrators by tau**k — that path's gain at omega = 1/tau.\n"
    "  Change tau and you change the frequency the question is asked at; the\n"
    "  algebraic weights do not move.\n"
    "- Relative weights divide by a signal's magnitude, so a signal that sits at\n"
    "  zero at the chosen operating point is governed by scale_floor rather than\n"
    "  by the model. graph.nodes_at_scale_floor() names them; trajectory mode\n"
    "  (used above) avoids the problem by normalizing over the whole run.\n"
    "- A multi-component state or a vector port collapses to one node, and its\n"
    "  edge weight is the induced infinity-norm of the Jacobian block — an upper\n"
    "  bound. The full block stays available as edge['relative'].\n"
    "- A hybrid block's weights describe its current mode only. Use trajectory\n"
    "  mode, or probe=, when that matters."
)
