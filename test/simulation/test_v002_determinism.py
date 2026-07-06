# SPDX-License-Identifier: MIT
"""V-002: Determinism and reproducibility.

Verifies ``jaxonomy.simulate`` produces bit-exact outputs across repeated runs
on the same hardware (same CPU, same backend), and that ``simulate_batch``
with batch size 1 reproduces a serial ``simulate`` within tight numerical
tolerance.

Coverage: scalar decay ODE, harmonic oscillator (2-D), van der Pol, hybrid
integrator with periodic discrete reset, bouncing-ball zero-crossing, RC
circuit (acausal, optional), 3-state machine, 100s long-horizon harmonic,
PID controller around a 1st-order plant.

Out of scope: GPU/TPU cross-hardware bit-exactness — JAX gives bit-exact
results only on a fixed device + XLA build; cross-hardware reproducibility
is not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

import jaxonomy
from jaxonomy import (
    DiagramBuilder,
    SimulatorOptions,
    simulate,
    simulate_batch,
)
from jaxonomy.framework.state_machine_builder import StateMachineBuilder
from jaxonomy.library import (
    Adder,
    Clock,
    Comparator,
    Constant,
    DiscreteClock,
    Gain,
    Integrator,
    PID,
    Step,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.slow


try:
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    HAS_ACAUSAL = True
except Exception:  # pragma: no cover - depends on optional install
    HAS_ACAUSAL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opts(max_major_steps=200):
    return SimulatorOptions(math_backend="jax", max_major_steps=max_major_steps)


def _assert_serial_bit_exact(r1, r2):
    np.testing.assert_array_equal(np.asarray(r1.time), np.asarray(r2.time))
    assert set(r1.outputs.keys()) == set(r2.outputs.keys())
    for name in r1.outputs:
        np.testing.assert_array_equal(
            np.asarray(r1.outputs[name]), np.asarray(r2.outputs[name]),
            err_msg=f"output {name!r} differs across identical serial runs",
        )


def _run_twice(diagram, t_span, recorded, options):
    r1 = simulate(diagram, diagram.create_context(), t_span,
                  options=options, recorded_signals=recorded)
    r2 = simulate(diagram, diagram.create_context(), t_span,
                  options=options, recorded_signals=recorded)
    return r1, r2


def _assert_batch1_matches_serial(diagram, t_span, recorded, options,
                                  batch_param, serial):
    br = simulate_batch(diagram, t_span=t_span, param_batches=batch_param,
                        options=options, recorded_signals=recorded)
    assert br.outputs[next(iter(recorded))].shape[0] == 1
    n = min(len(br.time), len(serial.time))
    np.testing.assert_allclose(np.asarray(br.time)[:n],
                               np.asarray(serial.time)[:n],
                               rtol=1e-12, atol=0.0)
    for name in recorded:
        y_b = np.asarray(br.outputs[name][0])[:n]
        y_s = np.asarray(serial.outputs[name])[:n]
        np.testing.assert_allclose(
            y_b, y_s, rtol=1e-12, atol=1e-12,
            err_msg=f"batch(N=1) ≠ serial for {name!r}",
        )


def _assert_batch2_self_consistent(diagram, t_span, recorded, options, bp):
    br = simulate_batch(diagram, t_span=t_span, param_batches=bp,
                        options=options, recorded_signals=recorded)
    for name in recorded:
        np.testing.assert_array_equal(
            np.asarray(br.outputs[name][0]),
            np.asarray(br.outputs[name][1]),
            err_msg=f"batch[0] ≠ batch[1] for {name!r}",
        )


# ---------------------------------------------------------------------------
# Diagram builders
# ---------------------------------------------------------------------------

def _build_scalar_decay(tau=0.5):
    b = DiagramBuilder()
    integ = b.add(Integrator(initial_state=1.0, name="x"))
    g = b.add(Gain(gain=-1.0 / tau, name="kdecay"))
    b.connect(integ.output_ports[0], g.input_ports[0])
    b.connect(g.output_ports[0], integ.input_ports[0])
    return b.build(name="scalar_decay"), {"x": integ.output_ports[0]}


def _build_harmonic(omega=2.0):
    b = DiagramBuilder()
    pos = b.add(Integrator(initial_state=1.0, name="pos"))
    vel = b.add(Integrator(initial_state=0.0, name="vel"))
    k = b.add(Gain(gain=-omega * omega, name="k"))
    b.connect(vel.output_ports[0], pos.input_ports[0])
    b.connect(pos.output_ports[0], k.input_ports[0])
    b.connect(k.output_ports[0], vel.input_ports[0])
    return b.build(name="harmonic"), {
        "pos": pos.output_ports[0], "vel": vel.output_ports[0],
    }


def _build_van_der_pol(mu=1.0):
    from jaxonomy.models.van_der_pol import VanDerPol
    b = DiagramBuilder()
    vdp = b.add(VanDerPol(x0=[1.0, 0.0], mu=mu, name="vdp"))
    return b.build(name="vdp_root"), {"x": vdp.output_ports[0]}


def _build_periodic_reset_integrator(dt=0.1):
    b = DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    integ = b.add(Integrator(initial_state=0.0, enable_reset=True, name="integ"))
    clk = b.add(DiscreteClock(dt=dt, name="clk"))
    thr = b.add(Constant(0.5, name="thr"))
    cmp_ = b.add(Comparator(operator=">", name="cmp"))
    b.connect(src.output_ports[0], integ.input_ports[0])
    b.connect(clk.output_ports[0], cmp_.input_ports[0])
    b.connect(thr.output_ports[0], cmp_.input_ports[1])
    b.connect(cmp_.output_ports[0], integ.input_ports[1])
    return b.build(name="reset_integ"), {"x": integ.output_ports[0]}


def _build_bouncing_ball():
    b = DiagramBuilder()
    accel = b.add(Constant(-9.81, name="accel"))
    floor = b.add(Constant(0.0, name="floor"))
    vel = b.add(Integrator(initial_state=0.0, enable_reset=True,
                           enable_external_reset=True, name="vel"))
    pos = b.add(Integrator(initial_state=1.0, enable_reset=True,
                           enable_external_reset=True, name="pos"))
    impact = b.add(Comparator(operator="<", name="impact"))
    rest = b.add(Gain(-0.6, name="rest"))
    b.connect(accel.output_ports[0], vel.input_ports[0])
    b.connect(vel.output_ports[0], pos.input_ports[0])
    b.connect(pos.output_ports[0], impact.input_ports[0])
    b.connect(floor.output_ports[0], impact.input_ports[1])
    b.connect(impact.output_ports[0], vel.input_ports[1])
    b.connect(impact.output_ports[0], pos.input_ports[1])
    b.connect(vel.output_ports[0], rest.input_ports[0])
    b.connect(rest.output_ports[0], vel.input_ports[2])
    b.connect(floor.output_ports[0], pos.input_ports[2])
    return b.build(name="bouncer"), {"pos": pos.output_ports[0]}


def _build_state_machine_3state():
    smb = StateMachineBuilder()
    s0, s1, s2 = smb.add_state("s0"), smb.add_state("s1"), smb.add_state("s2")
    smb.set_initial_state(s0)
    smb.add_transition(s0, s1, guard="t > 1.0")
    smb.add_transition(s1, s2, guard="t > 2.0")
    smb.add_transition(s2, s0, guard="t > 3.0")
    sm = smb.build(name="ctrl3")
    b = DiagramBuilder()
    clk = b.add(Clock(name="clk"))
    sm_blk = b.add(sm)
    b.connect(clk.output_ports[0], sm_blk.input_ports[0])
    return b.build(name="sm_diag"), {"state": sm_blk.output_ports[0]}


def _build_pid_plant():
    b = DiagramBuilder()
    ref = b.add(Step(start_value=0.0, end_value=1.0, step_time=0.1, name="ref"))
    plant = b.add(Integrator(initial_state=0.0, name="plant"))
    decay = b.add(Gain(gain=-1.0, name="decay"))
    psum = b.add(Adder(2, operators="++", name="psum"))
    esum = b.add(Adder(2, operators="+-", name="esum"))
    pid = b.add(PID(kp=2.0, ki=1.0, kd=0.1, n=10.0, name="pid"))
    b.connect(ref.output_ports[0], esum.input_ports[0])
    b.connect(plant.output_ports[0], esum.input_ports[1])
    b.connect(esum.output_ports[0], pid.input_ports[0])
    b.connect(plant.output_ports[0], decay.input_ports[0])
    b.connect(decay.output_ports[0], psum.input_ports[0])
    b.connect(pid.output_ports[0], psum.input_ports[1])
    b.connect(psum.output_ports[0], plant.input_ports[0])
    return b.build(name="pid_plant"), {"y": plant.output_ports[0]}


# ---------------------------------------------------------------------------
# Parametrized core: serial bit-exactness + batch(N=1) match
# ---------------------------------------------------------------------------

# (id, builder_fn, t_span, max_major_steps, batch_param_key, batch_param_value)
# bp_key/bp_val form a 1-element batch whose value matches the diagram default
# (so simulate_batch is semantically a no-op apart from the batch wrapping).
_CASES = [
    pytest.param(_build_scalar_decay, (0.0, 5.0), 200,
                 "kdecay.gain", jnp.array([-2.0]), id="scalar_decay"),
    pytest.param(_build_harmonic, (0.0, 5.0), 300,
                 "k.gain", jnp.array([-4.0]), id="harmonic"),
    pytest.param(_build_van_der_pol, (0.0, 5.0), 400,
                 "vdp.mu", jnp.array([1.0]), id="van_der_pol"),
    pytest.param(_build_periodic_reset_integrator, (0.0, 1.0), 200,
                 "src.value", jnp.array([1.0]), id="periodic_reset"),
    pytest.param(_build_bouncing_ball, (0.0, 1.0), 400,
                 "rest.gain", jnp.array([-0.6]), id="bouncing_ball"),
    pytest.param(_build_state_machine_3state, (0.0, 4.0), 400,
                 None, None, id="state_machine_3state"),
    pytest.param(_build_pid_plant, (0.0, 2.0), 400,
                 "pid.kp", jnp.array([2.0]), id="pid_first_order_plant"),
]


@pytest.mark.parametrize("builder_fn,t_span,max_major_steps,bp_key,bp_val", _CASES)
def test_serial_bit_exact(builder_fn, t_span, max_major_steps, bp_key, bp_val):
    """Two identical serial runs → bit-exact arrays for time and outputs."""
    diagram, recorded = builder_fn()
    options = _opts(max_major_steps)
    r1, r2 = _run_twice(diagram, t_span, recorded, options)
    _assert_serial_bit_exact(r1, r2)


@pytest.mark.parametrize("builder_fn,t_span,max_major_steps,bp_key,bp_val", _CASES)
def test_batch_size_one_matches_serial(builder_fn, t_span, max_major_steps,
                                       bp_key, bp_val):
    """``simulate_batch`` (N=1) ≡ a serial ``simulate`` within rtol=1e-12."""
    if bp_key is None:
        pytest.skip("no scalar batchable parameter for this case")
    diagram, recorded = builder_fn()
    options = _opts(max_major_steps)
    serial = simulate(diagram, diagram.create_context(), t_span,
                      options=options, recorded_signals=recorded)
    _assert_batch1_matches_serial(diagram, t_span, recorded, options,
                                  {bp_key: bp_val}, serial)


@pytest.mark.parametrize(
    "builder_fn,t_span,max_major_steps,bp_key,bp_val",
    [c for c in _CASES if c.id in
     {"scalar_decay", "harmonic", "pid_first_order_plant"}],
)
def test_batch_size_two_self_consistent(builder_fn, t_span, max_major_steps,
                                        bp_key, bp_val):
    """``simulate_batch([ctx, ctx])`` → both elements bit-equal."""
    if bp_key is None:
        pytest.skip("no scalar batchable parameter for this case")
    diagram, recorded = builder_fn()
    options = _opts(max_major_steps)
    twice = jnp.stack([bp_val[0], bp_val[0]])
    _assert_batch2_self_consistent(diagram, t_span, recorded, options,
                                   {bp_key: twice})


# ---------------------------------------------------------------------------
# Long-horizon
# ---------------------------------------------------------------------------

def test_long_horizon_harmonic_serial_bit_exact():
    """100 s harmonic oscillator: two identical runs are bit-exact."""
    diagram, recorded = _build_harmonic(omega=2.0)
    options = _opts(max_major_steps=5000)
    r1, r2 = _run_twice(diagram, (0.0, 100.0), recorded, options)
    _assert_serial_bit_exact(r1, r2)


def test_long_horizon_harmonic_batch1_matches_serial():
    """100 s harmonic oscillator: batch(N=1) matches serial within rtol=1e-12."""
    diagram, recorded = _build_harmonic(omega=2.0)
    options = _opts(max_major_steps=5000)
    serial = simulate(diagram, diagram.create_context(), (0.0, 100.0),
                      options=options, recorded_signals=recorded)
    _assert_batch1_matches_serial(diagram, (0.0, 100.0), recorded, options,
                                  {"k.gain": jnp.array([-4.0])}, serial)


# ---------------------------------------------------------------------------
# Acausal: simple RC circuit (skipped if optional dep missing)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ACAUSAL, reason="acausal layer unavailable")
def test_rc_circuit_determinism():
    """Acausal RC low-pass: two serial runs produce bit-exact outputs."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(ev, name="c1", C=1.0,
                        initial_voltage=0.0, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", gnd, "p")

    sys = AcausalCompiler(ev, ad, verbose=False)()
    b = DiagramBuilder()
    sys = b.add(sys)
    diagram = b.build(name="rc_root")

    recorded = {"x": sys.output_ports[0]}
    r1 = simulate(diagram, diagram.create_context(), (0.0, 5.0),
                  recorded_signals=recorded)
    r2 = simulate(diagram, diagram.create_context(), (0.0, 5.0),
                  recorded_signals=recorded)
    _assert_serial_bit_exact(r1, r2)
