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
