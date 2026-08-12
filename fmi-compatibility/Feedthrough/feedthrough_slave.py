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
