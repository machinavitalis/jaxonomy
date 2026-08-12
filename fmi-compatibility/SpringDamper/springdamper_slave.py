import numpy as np
import jaxonomy
from jaxonomy.library import LTISystem
from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

M, C, K = 1.0, 0.5, 1.0

def _build():
    bld = jaxonomy.DiagramBuilder()
    plant = bld.add(LTISystem(
        A=np.array([[0.0, 1.0], [-K / M, -C / M]]),
        B=np.array([[0.0], [1.0 / M]]),
        C=np.array([[1.0, 0.0]]),
        D=np.array([[0.0]]),
        name="plant",
    ))
    bld.export_input(plant.input_ports[0], name="F")
    bld.export_output(plant.output_ports[0], name="x")
    return bld.build()

class SpringDamper(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build)
    DT = 0.01
