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
