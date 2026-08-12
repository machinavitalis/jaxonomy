import jaxonomy
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

def _build():
    ev = EqnEnv()
    ad = AcausalDiagram()
    # Lowercase v: VoltageSource takes **kwargs, so a
    # capitalised V is silently swallowed and the source
    # quietly keeps its 1.0 V default.
    src = elec.VoltageSource(ev, name="src", v=1.0,
                             enable_voltage_port=False)
    res = elec.Resistor(ev, name="res", R=100.0)
    # initial_voltage_fixed=True is load-bearing: left free, the
    # consistent-initialization solve is entitled to start the
    # network at steady state, and the capacitor reports a
    # constant 1 V instead of charging.
    cap = elec.Capacitor(ev, name="cap", C=1e-3,
                         initial_voltage=0.0,
                         initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    sensor = elec.VoltageSensor(ev, name="v_c")
    ad.connect(src, "p", res, "p")
    ad.connect(res, "n", cap, "p")
    ad.connect(cap, "n", gnd, "p")
    ad.connect(src, "n", gnd, "p")
    ad.connect(sensor, "p", cap, "p")
    ad.connect(sensor, "n", gnd, "p")
    plant = AcausalCompiler(ev, ad)()

    bld = jaxonomy.DiagramBuilder()
    block = bld.add(plant)
    bld.export_output(block.output_ports[0], name="v_c")
    return bld.build()

class RCNetwork(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build)
    DT = 0.01
