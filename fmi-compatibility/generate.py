# SPDX-License-Identifier: MIT
"""Regenerate the FMI compatibility artifacts in this directory.

Produces, per model, the file set the FMI project's CONTRIBUTING asks tool
vendors to publish: the ``.fmu`` itself, a reference solution computed by
the exporting tool, the options used to compute it, an input signal where
the model has one, and a Readme.

Run from the repository root::

    python fmi-compatibility/generate.py

Requires ``pip install jaxonomy[fmu]`` and a PythonFMU wrapper that can be
loaded by whatever will import the result — see ``../scripts/
build_pythonfmu_wrapper.sh`` and the note in README.md. The script refuses
to write FMUs a non-Python master could not open, so the published set
always matches what the page claims.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from textwrap import dedent, fill

import numpy as np

HERE = Path(__file__).resolve().parent
STOP_TIME = 10.0
STEP = 0.01
RTOL = 1e-6

# Each entry: the slave module source, plus how to compute the reference
# solution with the same equations in-process.
MODELS: dict[str, dict] = {
    "SpringDamper": {
        "description": "Damped mass-spring driven by an external force. "
                       "Continuous states with a real input and output.",
        "inputs": {"F": lambda t: np.sin(t)},
        "outputs": ["x"],
        "slave": '''
            import numpy as np
            import jaxonomy
            from jaxonomy.library import LTISystem
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            M, C, K = 1.0, 0.5, 1.0

            def _build():
                bld = jaxonomy.DiagramBuilder()
                plant = bld.add(LTISystem(
                    A=np.array([[0.0, 1.0], [-K / M, -C / M]]),
                    B=np.array([[0.0], [1.0 / M]]),
                    C=np.array([[1.0, 0.0]]),
                    D=np.array([[0.0]]),
                    name="plant",
                ))
                bld.export_input(plant.input_ports[0], name="F")
                bld.export_output(plant.output_ports[0], name="x")
                return bld.build()

            class SpringDamper(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "PIController": {
        "description": "Discrete-time PI controller. Sampled dynamics with "
                       "two real inputs and one output.",
        "inputs": {"setpoint": lambda t: np.ones_like(t),
                   "measurement": lambda t: 1.0 - np.exp(-t)},
        "outputs": ["u"],
        "slave": '''
            import numpy as np
            import jaxonomy
            from jaxonomy.library import LTISystemDiscrete, Adder, Gain
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            KP, KI, DT_ = 2.0, 0.5, 0.01

            def _build():
                bld = jaxonomy.DiagramBuilder()
                err = bld.add(Adder(2, operators="+-", name="error"))
                # Integral state advanced once per sample: x[k+1] = x[k] + dt*e.
                integ = bld.add(LTISystemDiscrete(
                    A=np.array([[1.0]]), B=np.array([[DT_]]),
                    C=np.array([[KI]]), D=np.array([[0.0]]),
                    dt=DT_, name="integral",
                ))
                prop = bld.add(Gain(KP, name="proportional"))
                total = bld.add(Adder(2, name="u"))
                bld.connect(err.output_ports[0], integ.input_ports[0])
                bld.connect(err.output_ports[0], prop.input_ports[0])
                bld.connect(prop.output_ports[0], total.input_ports[0])
                bld.connect(integ.output_ports[0], total.input_ports[1])
                bld.export_input(err.input_ports[0], name="setpoint")
                bld.export_input(err.input_ports[1], name="measurement")
                bld.export_output(total.output_ports[0], name="u")
                return bld.build()

            class PIController(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "BouncingBall": {
        "description": "Ball under gravity bouncing on a floor. Continuous "
                       "states with a zero-crossing event and a state reset, "
                       "no inputs.",
        "inputs": {},
        "outputs": ["h", "v"],
        "slave": '''
            import jax.numpy as jnp
            import jaxonomy
            from jaxonomy import LeafSystem
            from jaxonomy.framework import DependencyTicket
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            G, E = 9.81, 0.7

            class Ball(LeafSystem):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.declare_continuous_state(
                        default_value=jnp.array([1.0, 0.0]), ode=self._ode,
                        requires_inputs=False,
                    )
                    for index, port in enumerate(("h", "v")):
                        self.declare_output_port(
                            self._reader(index), name=port,
                            prerequisites_of_calc=[DependencyTicket.xc],
                            requires_inputs=False,
                        )
                    self.declare_zero_crossing(
                        guard=lambda t, s, *a, **k: s.continuous_state[0],
                        reset_map=self._bounce,
                        direction="positive_then_non_positive",
                        name="impact",
                    )

                @staticmethod
                def _reader(index):
                    def _read(time, state, *inputs, **params):
                        return state.continuous_state[index]
                    return _read

                def _ode(self, time, state, *inputs, **params):
                    return jnp.array([state.continuous_state[1], -G])

                def _bounce(self, time, state, *inputs, **params):
                    h, v = state.continuous_state
                    return state.with_continuous_state(
                        jnp.array([jnp.maximum(h, 0.0), -E * v])
                    )

            def _build():
                bld = jaxonomy.DiagramBuilder()
                ball = bld.add(Ball(name="ball"))
                bld.export_output(ball.output_ports[0], name="h")
                bld.export_output(ball.output_ports[1], name="v")
                return bld.build()

            class BouncingBall(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "Feedthrough": {
        "description": "Scalar and vector feedthrough. Exercises array-valued "
                       "FMI variables alongside plain scalars.",
        "inputs": {"u_scalar": lambda t: np.sin(t)},
        "outputs": ["y_scalar", "y_gain"],
        "slave": '''
            import jaxonomy
            from jaxonomy.library import Gain
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            def _build():
                bld = jaxonomy.DiagramBuilder()
                passthrough = bld.add(Gain(1.0, name="passthrough"))
                doubler = bld.add(Gain(2.0, name="doubler"))
                bld.connect(passthrough.output_ports[0], doubler.input_ports[0])
                bld.export_input(passthrough.input_ports[0], name="u_scalar")
                bld.export_output(passthrough.output_ports[0], name="y_scalar")
                bld.export_output(doubler.output_ports[0], name="y_gain")
                return bld.build()

            class Feedthrough(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "RCNetwork": {
        "description": "Series RC network built from acausal electrical "
                       "components and compiled through index reduction.",
        "inputs": {},
        "outputs": ["v_c"],
        "slave": '''
            import jaxonomy
            from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
            from jaxonomy.acausal import electrical as elec
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            def _build():
                ev = EqnEnv()
                ad = AcausalDiagram()
                # Lowercase v: VoltageSource takes **kwargs, so a
                # capitalised V is silently swallowed and the source
                # quietly keeps its 1.0 V default.
                src = elec.VoltageSource(ev, name="src", v=1.0,
                                         enable_voltage_port=False)
                res = elec.Resistor(ev, name="res", R=100.0)
                # initial_voltage_fixed=True is load-bearing: left free, the
                # consistent-initialization solve is entitled to start the
                # network at steady state, and the capacitor reports a
                # constant 1 V instead of charging.
                cap = elec.Capacitor(ev, name="cap", C=1e-3,
                                     initial_voltage=0.0,
                                     initial_voltage_fixed=True)
                gnd = elec.Ground(ev, name="gnd")
                sensor = elec.VoltageSensor(ev, name="v_c")
                ad.connect(src, "p", res, "p")
                ad.connect(res, "n", cap, "p")
                ad.connect(cap, "n", gnd, "p")
                ad.connect(src, "n", gnd, "p")
                ad.connect(sensor, "p", cap, "p")
                ad.connect(sensor, "n", gnd, "p")
                plant = AcausalCompiler(ev, ad)()

                bld = jaxonomy.DiagramBuilder()
                block = bld.add(plant)
                bld.export_output(block.output_ports[0], name="v_c")
                return bld.build()

            class RCNetwork(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "MixedTypes": {
        "description": "Real, Integer, Boolean and String FMI variables "
                       "side by side. Mirrors the Reference-FMUs "
                       "Feedthrough's type coverage.",
        "inputs": {"u_real": lambda t: np.sin(t),
                   "u_int": lambda t: np.floor(t).astype(int),
                   "u_bool": lambda t: (np.sin(t) > 0.0)},
        "outputs": ["y_real", "y_int", "y_bool", "y_over"],
        "slave": '''
            import numpy as np
            import jaxonomy
            from jaxonomy.library import (
                Comparator, Constant, Gain, LogicalOperator,
            )
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            THRESHOLD = 0.5

            def _build():
                bld = jaxonomy.DiagramBuilder()
                # The FMI type of each variable follows its signal dtype,
                # so the Constant's dtype is what makes u_int an Integer
                # and u_bool a Boolean rather than a Real.
                u_real = bld.add(Constant(0.0, name="u_real"))
                u_int = bld.add(Constant(np.int64(0), name="u_int"))
                u_bool = bld.add(Constant(np.bool_(False), name="u_bool"))
                threshold = bld.add(Constant(THRESHOLD, name="threshold"))
                doubler = bld.add(Gain(2.0, name="doubler"))
                over = bld.add(Comparator(operator=">", name="over"))
                invert = bld.add(LogicalOperator("not", name="invert"))
                bld.connect(u_real.output_ports[0], doubler.input_ports[0])
                bld.connect(u_real.output_ports[0], over.input_ports[0])
                bld.connect(threshold.output_ports[0], over.input_ports[1])
                bld.connect(u_bool.output_ports[0], invert.input_ports[0])
                bld.export_output(doubler.output_ports[0], name="y_real")
                bld.export_output(u_int.output_ports[0], name="y_int")
                bld.export_output(invert.output_ports[0], name="y_bool")
                bld.export_output(over.output_ports[0], name="y_over")
                return bld.build()

            class MixedTypes(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
                # Jaxonomy has no string-valued signal, so a String
                # variable carries metadata rather than feeding the
                # diagram.
                EXPOSE_STRINGS = {"units_note": "u_real in volts"}
        ''',
    },
    "VectorIO": {
        "description": "Array-valued variables. FMI 2.0 has no array "
                       "type, so a vector port flattens to one scalar "
                       "variable per element, named name[i].",
        "inputs": {},
        "outputs": ["y_vec[0]", "y_vec[1]", "y_vec[2]", "y_sum"],
        "notes": """
Array variables
---------------
FMI 2.0 has no array type, so each element is its own ScalarVariable:
u_vec[0..2] as inputs and y_vec[0..2] as outputs, plus the scalar
y_sum. Vector Constants and vector output ports both flatten this way.

An *exported* vector input port (bld.export_input) does not: the port
carries no shape until something feeds it, so the slave registers a
single scalar. Drive array inputs from a vector Constant, as here.
""",
        "slave": '''
            import numpy as np
            import jaxonomy
            from jaxonomy.library import Constant, Gain, SumOfElements
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            def _build():
                bld = jaxonomy.DiagramBuilder()
                # A vector Constant is auto-discovered as three FMI
                # inputs u_vec[0..2]; the vector output flattens the
                # same way. An *exported* vector input port would not
                # work here — see the note in Readme.txt.
                u_vec = bld.add(Constant(np.array([1.0, 2.0, 3.0]),
                                         name="u_vec"))
                scale = bld.add(Gain(2.0, name="scale"))
                total = bld.add(SumOfElements(name="total"))
                bld.connect(u_vec.output_ports[0], scale.input_ports[0])
                bld.connect(scale.output_ports[0], total.input_ports[0])
                bld.export_output(scale.output_ports[0], name="y_vec")
                bld.export_output(total.output_ports[0], name="y_sum")
                return bld.build()

            class VectorIO(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "Parameterized": {
        "description": "FMI parameters applied at initialization: an "
                       "EXPOSE_INITIAL_STATES initial state and a "
                       "Constant-backed rate, both set before "
                       "exitInitializationMode.",
        "inputs": {},
        "outputs": ["x"],
        # Set during initialization mode, so the reference exercises the
        # parameter path rather than the model's built-in defaults.
        "start_values": {"x0": 2.5, "decay_rate": 0.8},
        "slave": '''
            import jaxonomy
            from jaxonomy.library import Constant, Gain, Integrator, Product
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            def _build():
                bld = jaxonomy.DiagramBuilder()
                # dx/dt = -(decay_rate * x), so x(t) = x0*exp(-k*t).
                rate = bld.add(Constant(0.5, name="decay_rate"))
                state = bld.add(Integrator(1.0, name="state"))
                scaled = bld.add(Product(2, name="rate_times_x"))
                negate = bld.add(Gain(-1.0, name="negate"))
                bld.connect(state.output_ports[0], scaled.input_ports[0])
                bld.connect(rate.output_ports[0], scaled.input_ports[1])
                bld.connect(scaled.output_ports[0], negate.input_ports[0])
                bld.connect(negate.output_ports[0], state.input_ports[0])
                bld.export_output(state.output_ports[0], name="x")
                return bld.build()

            class Parameterized(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
                # x0 becomes a fixed-variability FMI parameter applied
                # to the Integrator's continuous state at init.
                EXPOSE_INITIAL_STATES = {"x0": "state"}
        ''',
    },
    "Thermostat": {
        "description": "Two-sided hysteresis. A relay with distinct on "
                       "and off thresholds switches a heater, so the "
                       "model carries mode state across the boundary "
                       "alongside its continuous state.",
        "inputs": {},
        "outputs": ["T", "heater"],
        "slave": '''
            import jax.numpy as jnp
            import jaxonomy
            from jaxonomy import LeafSystem
            from jaxonomy.framework import DependencyTicket
            from jaxonomy.library import Adder, Constant, Relay
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            C_TH, R_TH, T_AMB, P_HEAT = 2.0, 1.0, 20.0, 40.0
            T_SET, BAND = 50.0, 1.0

            class Room(LeafSystem):
                """C dT/dt = P*heater - (T - T_amb)/R."""

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.declare_input_port(name="heater")
                    self.declare_continuous_state(
                        default_value=jnp.array(T_AMB), ode=self._ode,
                    )
                    self.declare_output_port(
                        lambda t, s, *u, **p: s.continuous_state,
                        name="T",
                        prerequisites_of_calc=[DependencyTicket.xc],
                        requires_inputs=False,
                    )

                def _ode(self, time, state, *inputs, **params):
                    T = state.continuous_state
                    return (P_HEAT * inputs[0] - (T - T_AMB) / R_TH) / C_TH

            def _build():
                bld = jaxonomy.DiagramBuilder()
                room = bld.add(Room(name="room"))
                setpoint = bld.add(Constant(T_SET, name="setpoint"))
                error = bld.add(Adder(2, operators="+-", name="error"))
                # error = T_set - T, so the heater turns ON once the
                # room is BAND below setpoint and OFF once it is BAND
                # above it: the switching points differ, which is what
                # makes this hysteresis rather than a comparator.
                relay = bld.add(Relay(
                    on_threshold=BAND, off_threshold=-BAND,
                    on_value=1.0, off_value=0.0, initial_state=1.0,
                    name="thermostat",
                ))
                bld.connect(setpoint.output_ports[0], error.input_ports[0])
                bld.connect(room.output_ports[0], error.input_ports[1])
                bld.connect(error.output_ports[0], relay.input_ports[0])
                bld.connect(relay.output_ports[0], room.input_ports[0])
                bld.export_output(room.output_ports[0], name="T")
                bld.export_output(relay.output_ports[0], name="heater")
                return bld.build()

            class Thermostat(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "StiffChemical": {
        "description": "Robertson kinetics — a classic stiff problem. "
                       "The implicit BDF solver is re-entered on every "
                       "doStep, and the three species conserve mass.",
        "inputs": {},
        "outputs": ["y1", "y2", "y3"],
        "slave": '''
            import jax.numpy as jnp
            import jaxonomy
            from jaxonomy import LeafSystem
            from jaxonomy.framework import DependencyTicket
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            K1, K2, K3 = 0.04, 1.0e4, 3.0e7

            class Robertson(LeafSystem):
                """y1' = -k1 y1 + k2 y2 y3
                   y2' =  k1 y1 - k2 y2 y3 - k3 y2^2
                   y3' =                     k3 y2^2
                Mass is conserved exactly: y1 + y2 + y3 = 1."""

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.declare_continuous_state(
                        default_value=jnp.array([1.0, 0.0, 0.0]),
                        ode=self._ode, requires_inputs=False,
                    )
                    for index, port in enumerate(("y1", "y2", "y3")):
                        self.declare_output_port(
                            self._reader(index), name=port,
                            prerequisites_of_calc=[DependencyTicket.xc],
                            requires_inputs=False,
                        )

                @staticmethod
                def _reader(index):
                    def _read(time, state, *inputs, **params):
                        return state.continuous_state[index]
                    return _read

                def _ode(self, time, state, *inputs, **params):
                    y1, y2, y3 = state.continuous_state
                    return jnp.array([
                        -K1 * y1 + K2 * y2 * y3,
                        K1 * y1 - K2 * y2 * y3 - K3 * y2**2,
                        K3 * y2**2,
                    ])

            def _build():
                bld = jaxonomy.DiagramBuilder()
                rob = bld.add(Robertson(name="robertson"))
                for index, port in enumerate(("y1", "y2", "y3")):
                    bld.export_output(rob.output_ports[index], name=port)
                return bld.build()

            class StiffChemical(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
                # Without this the slave rebuilds default (non-stiff)
                # options on every communication step and the stiff
                # solver is never reached.
                SIMULATOR_OPTIONS = {
                    "ode_solver_method": "bdf",
                    "rtol": 1e-8,
                    "atol": 1e-10,
                }
        ''',
    },
    "DCMotor": {
        "description": "Cross-domain acausal model — electrical and "
                       "rotational — driven through a causal voltage "
                       "input, so the FMI boundary sits on a "
                       "causal/acausal seam.",
        "inputs": {},
        "outputs": ["amp_i", "speed_w_rel"],
        "slave": '''
            import jaxonomy
            from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
            from jaxonomy.acausal import electrical as elec
            from jaxonomy.acausal.component_library import rotational as rot
            from jaxonomy.library import Constant
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            R_A, K_M, L_A, J_M, B_L, V_IN = 1.0, 0.05, 0.5, 0.01, 1e-4, 12.0

            def _build():
                ev = EqnEnv()
                ad = AcausalDiagram()
                # enable_voltage_port=True is the causal seam: the
                # acausal network takes its supply from a signal.
                src = elec.VoltageSource(ev, name="src",
                                         enable_voltage_port=True)
                # The ICs are pinned; left free, consistent
                # initialization is entitled to start the machine at
                # its steady state and the whole spin-up disappears.
                motor = elec.IdealMotor(
                    ev, name="motor", R=R_A, K=K_M, L=L_A, J=J_M,
                    initial_current=0.0, initial_current_fixed=True,
                    initial_velocity=0.0, initial_velocity_fixed=True,
                )
                gnd = elec.Ground(ev, name="gnd")
                load = rot.Damper(ev, name="load", D=B_L)
                mount = rot.FixedAngle(ev, name="mount")
                speed = rot.MotionSensor(ev, name="speed",
                                         enable_flange_b=False)
                amp = elec.CurrentSensor(ev, name="amp")
                ad.connect(src, "p", amp, "p")
                ad.connect(amp, "n", motor, "pos")
                ad.connect(motor, "neg", gnd, "p")
                ad.connect(src, "n", gnd, "p")
                ad.connect(motor, "shaft", load, "flange_a")
                ad.connect(load, "flange_b", mount, "flange")
                ad.connect(speed, "flange_a", motor, "shaft")
                plant = AcausalCompiler(ev, ad)()

                bld = jaxonomy.DiagramBuilder()
                block = bld.add(plant)
                supply = bld.add(Constant(V_IN, name="v_supply"))
                bld.connect(supply.output_ports[0], block.input_ports[0])
                for port in block.output_ports:
                    bld.export_output(port, name=port.name)
                return bld.build()

            class DCMotor(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
    "ONNXPolicy": {
        "description": "A neural policy exported to ONNX, evaluated "
                       "inside the FMU by jaxonomy's ONNXJax block and "
                       "held at the sample rate it was trained for.",
        "inputs": {},
        "outputs": ["x", "u"],
        "project_files": ["policy.onnx"],
        "slave": '''
            import os

            import jax.numpy as jnp
            import jaxonomy
            from jaxonomy import LeafSystem
            from jaxonomy.framework import DependencyTicket
            from jaxonomy.library import ZeroOrderHold
            from jaxonomy.library.onnx_jax_block import ONNXJax
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            TS, X0 = 0.1, 1.0
            # The .onnx rides in the FMU's resources/, next to this file.
            POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "policy.onnx")

            class Lag(LeafSystem):
                """dx/dt = -x + u."""

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.declare_input_port(name="u")
                    self.declare_continuous_state(
                        default_value=jnp.array(X0), ode=self._ode,
                    )
                    self.declare_output_port(
                        lambda t, s, *u, **p: s.continuous_state,
                        name="x",
                        prerequisites_of_calc=[DependencyTicket.xc],
                        requires_inputs=False,
                    )

                def _ode(self, time, state, *inputs, **params):
                    return -state.continuous_state + inputs[0]

            class Reshape(LeafSystem):
                """Scalar <-> the (1, 1) float32 batch the graph wants."""

                def __init__(self, to_batch, **kwargs):
                    super().__init__(**kwargs)
                    self.declare_input_port(name="in")
                    if to_batch:
                        fn = lambda t, s, *u, **p: jnp.reshape(  # noqa: E731
                            u[0], (1, 1)).astype(jnp.float32)
                    else:
                        fn = lambda t, s, *u, **p: jnp.reshape(  # noqa: E731
                            u[0], ()).astype(jnp.float64)
                    self.declare_output_port(fn, name="out",
                                             requires_inputs=True)

            def _build():
                bld = jaxonomy.DiagramBuilder()
                plant = bld.add(Lag(name="plant"))
                pre = bld.add(Reshape(True, name="to_policy"))
                policy = bld.add(ONNXJax(
                    file_name=POLICY, num_inputs=1, num_outputs=1,
                    cast_outputs_to_dtype="float32", name="policy",
                ))
                post = bld.add(Reshape(False, name="from_policy"))
                # A policy trained as sample-and-hold must be held, or
                # it gets re-evaluated at every solver stage.
                hold = bld.add(ZeroOrderHold(dt=TS, name="hold"))
                bld.connect(plant.output_ports[0], pre.input_ports[0])
                bld.connect(pre.output_ports[0], policy.input_ports[0])
                bld.connect(policy.output_ports[0], post.input_ports[0])
                bld.connect(post.output_ports[0], hold.input_ports[0])
                bld.connect(hold.output_ports[0], plant.input_ports[0])
                bld.export_output(plant.output_ports[0], name="x")
                bld.export_output(hold.output_ports[0], name="u")
                return bld.build()

            class ONNXPolicy(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.1
                SIMULATOR_OPTIONS = {
                    "max_major_step_length": TS,
                    "max_minor_step_size": TS,
                    "rtol": 1e-12,
                    "atol": 1e-14,
                }
        ''',
    },
    "UnitsAnnotated": {
        "description": "Ports carrying jaxonomy Unit annotations. The "
                       "model is unit-consistent internally; the FMU "
                       "does not carry the units (see Readme.txt).",
        "inputs": {},
        "outputs": ["level"],
        "notes": """
Units do not cross the boundary
-------------------------------
The source annotates its ports with jaxonomy Units (m for the level,
m3/s for the inflow) and jaxonomy checks them at connect time. The FMU
does not carry them: modelDescription.xml has no unit attribute on
`level` and no UnitDefinitions block, because the PythonFMU wrapper
this export is built on emits neither.

So this model is the negative result in the set. Its trajectory is
correct -- it matches the analytic h(t) = q_in*R*(1 - exp(-t/(A*R)))
exactly -- but an importer learns nothing about what the number means.
""",
        "slave": '''
            import jax.numpy as jnp
            import jaxonomy
            from jaxonomy import LeafSystem
            from jaxonomy.framework import DependencyTicket
            from jaxonomy.framework.units import Unit
            from jaxonomy.library import Constant
            from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

            AREA, R_OUT, Q_IN, H0 = 2.0, 4.0, 0.5, 0.0

            METER = Unit(dims=(0, 1, 0, 0, 0, 0, 0), name="m")
            CUBIC_METER_PER_SECOND = Unit(dims=(0, 3, -1, 0, 0, 0, 0),
                                          name="m3/s")

            class Tank(LeafSystem):
                """A dh/dt = q_in - h/R_out, with annotated ports."""

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.declare_input_port(
                        name="q_in", units=CUBIC_METER_PER_SECOND,
                    )
                    self.declare_continuous_state(
                        default_value=jnp.array(H0), ode=self._ode,
                    )
                    self.declare_output_port(
                        lambda t, s, *u, **p: s.continuous_state,
                        name="level", units=METER,
                        prerequisites_of_calc=[DependencyTicket.xc],
                        requires_inputs=False,
                    )

                def _ode(self, time, state, *inputs, **params):
                    h = state.continuous_state
                    return (inputs[0] - h / R_OUT) / AREA

            def _build():
                bld = jaxonomy.DiagramBuilder()
                tank = bld.add(Tank(name="tank"))
                inflow = bld.add(Constant(Q_IN, name="q_in"))
                bld.connect(inflow.output_ports[0], tank.input_ports[0])
                bld.export_output(tank.output_ports[0], name="level")
                return bld.build()

            class UnitsAnnotated(JaxonomyDiagramSlave):
                DIAGRAM_FACTORY = staticmethod(_build)
                DT = 0.01
        ''',
    },
}


def _check_wrapper() -> None:
    from jaxonomy.library.fmu_export import wrapper_diagnostics

    info = wrapper_diagnostics()
    if not (info["present"] and info["arch_matches_host"]):
        sys.exit(
            f"the installed PythonFMU wrapper is {info['machine']} on a "
            f"{info['host_machine']} host; run "
            f"scripts/build_pythonfmu_wrapper.sh first"
        )
    if not info["embeds_python"]:
        sys.exit(
            "the installed PythonFMU wrapper links no libpython, so the FMUs "
            "would not load in a non-Python master; run "
            "scripts/build_pythonfmu_wrapper.sh first"
        )


def _write_options(path: Path, stop_time: float, step: float = STEP) -> None:
    path.write_text(
        f"StartTime, 0.0\n"
        f"StopTime, {stop_time}\n"
        f"StepSize, {step}\n"
        f"RelTol, {RTOL}\n"
        f"SolverType, FixedStep\n"
        f"OutputIntervalLength, {step}\n"
    )


def _start_values_note(spec: dict) -> str:
    """Readme paragraph recording the parameter values the reference was
    computed with, when they differ from the model's own defaults."""
    start_values = spec.get("start_values")
    if not start_values:
        return ""
    settings = ", ".join(f"{k} = {v}" for k, v in start_values.items())
    return "Parameters\n----------\n" + fill(
        f"The reference was computed with {settings}, set during "
        f"initialization mode. _ref.opt records the remaining options.",
        width=72,
    )


def _write_csv(path: Path, time: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    header = "time," + ",".join(columns)
    data = np.column_stack([time, *columns.values()])
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")


def _write_policy_onnx(out: Path) -> None:
    """Emit the small MLP policy ONNXPolicy runs.

    Built here rather than committed as an opaque binary: the weights
    come from a seeded generator, so the artifact is reproducible from
    this source alone.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    hidden = 8
    weights = {
        "W1": rng.normal(0.0, 0.8, (1, hidden)).astype(np.float32),
        "b1": rng.normal(0.0, 0.1, (hidden,)).astype(np.float32),
        "W2": rng.normal(0.0, 0.8, (hidden, 1)).astype(np.float32),
        "b2": np.zeros((1,), np.float32),
    }
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h0"]),
        helper.make_node("Add", ["h0", "b1"], ["h1"]),
        helper.make_node("Tanh", ["h1"], ["h2"]),
        helper.make_node("MatMul", ["h2", "W2"], ["h3"]),
        helper.make_node("Add", ["h3", "b2"], ["u"]),
    ]
    graph = helper.make_graph(
        nodes, "policy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("u", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(a, n) for n, a in weights.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(out / "policy.onnx"))


ASSETS = {"ONNXPolicy": _write_policy_onnx}


def main() -> None:
    _check_wrapper()
    from jaxonomy.library.fmu_export import build_fmu

    for name, spec in MODELS.items():
        stop_time = float(spec.get("stop_time", STOP_TIME))
        step = float(spec.get("step", STEP))
        time = np.arange(0.0, stop_time + step / 2, step)

        out = HERE / name
        out.mkdir(exist_ok=True)
        script = out / f"{name.lower()}_slave.py"
        script.write_text(dedent(spec["slave"]).strip() + "\n")

        if name in ASSETS:
            ASSETS[name](out)

        build_fmu(script, out / f"{name}.fmu", project_files=[
            out / f for f in spec.get("project_files", ())
        ] or None)

        if spec["inputs"]:
            _write_csv(out / f"{name}_in.csv", time,
                       {k: np.asarray(f(time), dtype=float)
                        for k, f in spec["inputs"].items()})

        # Reference solution as computed by the exporting tool: drive the very
        # FMU we ship, through fmpy, so the published CSV is what an importer
        # should reproduce rather than a separately-derived idealization.
        import fmpy

        result = fmpy.simulate_fmu(
            str(out / f"{name}.fmu"),
            stop_time=stop_time,
            output_interval=step,
            input=(np.genfromtxt(out / f"{name}_in.csv", delimiter=",", names=True)
                   if spec["inputs"] else None),
            start_values=spec.get("start_values", {}),
            output=spec["outputs"],
        )
        _write_csv(out / f"{name}_ref.csv", np.asarray(result["time"]),
                   {c: np.asarray(result[c]) for c in spec["outputs"]})
        _write_options(out / f"{name}_ref.opt",
                       float(np.asarray(result["time"])[-1]), step)

        # Assembled from already-dedented pieces rather than one
        # f-string: a zero-indent line anywhere in the interpolated
        # notes would make dedent() a no-op and indent the whole file.
        files = [f"{name}.fmu          the FMU",
                 f"{name}_ref.csv      reference solution computed by Jaxonomy",
                 f"{name}_ref.opt      options used to compute it"]
        if spec["inputs"]:
            files.append(f"{name}_in.csv       input signals")
        files.append(f"{name.lower()}_slave.py  the model source, for reference")
        files += [f"{f}       bundled into the FMU resources"
                  for f in spec.get("project_files", ())]

        sections = [
            f"{name}\n{'=' * len(name)}",
            fill(spec["description"], width=72),
            dedent("""\
                Exported from Jaxonomy (https://py.jaxonomy.com/) as an FMI 2.0
                co-simulation FMU via jaxonomy.library.fmu_export.build_fmu.

                IMPORTANT — this FMU is tool-coupled. The slave runs as Python,
                so the importing side needs a Python environment with jaxonomy
                installed:

                    pip install "jaxonomy[fmu]"

                It is not a self-contained binary. Platform binaries come from
                the PythonFMU wrapper and are x86-64."""),
            "Files\n-----\n" + "\n".join(files),
        ]
        sections += [s for s in (_start_values_note(spec),
                                 (spec.get("notes") or "").strip()) if s]
        sections.append(dedent("""\
            Checked with fmpy.validate_fmu, INTO-CPS VDMCheck 1.1.3 and
            fmusim validate. See ../README.md for the full compatibility
            matrix."""))
        (out / "Readme.txt").write_text("\n\n".join(sections) + "\n")
        # pythonfmu imports the slave script to find its class, which
        # leaves a __pycache__ next to it. Not part of the published set.
        shutil.rmtree(out / "__pycache__", ignore_errors=True)
        print(f"  {name}: fmu + reference written to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    if shutil.which("python") is None:  # pragma: no cover - sanity only
        sys.exit("python not found")
    main()
