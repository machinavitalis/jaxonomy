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
