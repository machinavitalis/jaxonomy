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
