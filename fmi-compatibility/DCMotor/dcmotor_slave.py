import jaxonomy
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.acausal.component_library import rotational as rot
from jaxonomy.library import Constant
from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

R_A, K_M, L_A, J_M, B_L, V_IN = 1.0, 0.05, 0.5, 0.01, 1e-4, 12.0

def _build():
    ev = EqnEnv()
    ad = AcausalDiagram()
    # enable_voltage_port=True is the causal seam: the
    # acausal network takes its supply from a signal.
    src = elec.VoltageSource(ev, name="src",
                             enable_voltage_port=True)
    # The ICs are pinned; left free, consistent
    # initialization is entitled to start the machine at
    # its steady state and the whole spin-up disappears.
    motor = elec.IdealMotor(
        ev, name="motor", R=R_A, K=K_M, L=L_A, J=J_M,
        initial_current=0.0, initial_current_fixed=True,
        initial_velocity=0.0, initial_velocity_fixed=True,
    )
    gnd = elec.Ground(ev, name="gnd")
    load = rot.Damper(ev, name="load", D=B_L)
    mount = rot.FixedAngle(ev, name="mount")
    speed = rot.MotionSensor(ev, name="speed",
                             enable_flange_b=False)
    amp = elec.CurrentSensor(ev, name="amp")
    ad.connect(src, "p", amp, "p")
    ad.connect(amp, "n", motor, "pos")
    ad.connect(motor, "neg", gnd, "p")
    ad.connect(src, "n", gnd, "p")
    ad.connect(motor, "shaft", load, "flange_a")
    ad.connect(load, "flange_b", mount, "flange")
    ad.connect(speed, "flange_a", motor, "shaft")
    plant = AcausalCompiler(ev, ad)()

    bld = jaxonomy.DiagramBuilder()
    block = bld.add(plant)
    supply = bld.add(Constant(V_IN, name="v_supply"))
    bld.connect(supply.output_ports[0], block.input_ports[0])
    for port in block.output_ports:
        bld.export_output(port, name=port.name)
    return bld.build()

class DCMotor(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build)
    DT = 0.01
