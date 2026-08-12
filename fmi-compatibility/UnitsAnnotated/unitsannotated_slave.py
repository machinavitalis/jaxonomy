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
