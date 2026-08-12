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
