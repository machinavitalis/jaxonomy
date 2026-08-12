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
