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
