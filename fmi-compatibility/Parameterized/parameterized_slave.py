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
