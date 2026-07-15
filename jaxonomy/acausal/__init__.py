# SPDX-License-Identifier: MIT
"""
Acausal (equation-based) modeling and simulation framework.

This module provides Modelica-inspired, equation-based modeling for multi-domain
physical systems (electrical, mechanical, thermal, fluid, hydraulic). Models are
described as networks of components connected at ports, and the compiler
automatically:

1. **Diagram processing**: assembles node-flow balance equations and eliminates
   aliases, producing a raw DAE system.
2. **Index reduction**: applies the Pantelides algorithm to reduce the DAE to
   index ≤ 1, followed by dummy-derivative substitution (Mattsson & Söderlind,
   1993) and BLT ordering to obtain a semi-explicit ODE/DAE form.
3. **Code generation**: lambdifies the symbolic RHS into a :class:`~jaxonomy.framework.LeafSystem`
   (``AcausalSystem``) that integrates directly with the Jaxonomy simulation
   framework.

Quick-start example::

    import jaxonomy
    from jaxonomy.acausal import (
        AcausalCompiler, AcausalDiagram, EqnEnv,
        electrical as elec,
    )

    ev = EqnEnv()
    ad = AcausalDiagram()
    v1  = elec.VoltageSource(ev, name="v1", V=1.0)
    r1  = elec.Resistor(ev, name="r1", R=1.0)
    c1  = elec.Capacitor(ev, name="c1", C=1.0,
                         initial_voltage=0.0, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", gnd, "p")

    system = AcausalCompiler(ev, ad)()   # diagram_processing → index_reduction → generate

    builder = jaxonomy.DiagramBuilder()
    system  = builder.add(system)
    diagram = builder.build()
    results = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 10.0))

Supported physical domains (component libraries):

- :mod:`~jaxonomy.acausal.component_library.electrical` —
  voltage/current; Resistor, Capacitor, Inductor, VoltageSource, CurrentSource,
  Ground, IdealDiode, IdealMotor, CurrentSensor, VoltageSensor.
- :mod:`~jaxonomy.acausal.component_library.rotational` —
  angle/torque; Inertia, Spring, Damper, TorqueSource, FixedAngle, sensors.
- :mod:`~jaxonomy.acausal.component_library.translational` —
  position/force; Mass, Spring, Damper, ForceSource, FixedPosition, sensors.
- :mod:`~jaxonomy.acausal.component_library.thermal` —
  temperature/heat-flow; HeatCapacitor, Insulator, TemperatureSource,
  HeatFlowSource, RadiativeHeatTransfer, sensors.
- :mod:`~jaxonomy.acausal.component_library.fluid` —
  pressure/mass-flow with thermal effects; ClosedVolume, Accumulator, OpenTank,
  StaticPipe, SimplePipe, ThermalPipe, MassflowSource, Boundary_pT, sensors
  (experimental).
- :mod:`~jaxonomy.acausal.component_library.hydraulic` —
  incompressible hydraulic oil; Pump, Accumulator, Pipe, PressureSource,
  MassflowSource, HydraulicActuatorLinear, sensors.

There is no valve component yet in either fluid domain: model orifices/valves
with a resistive ``Pipe(R=...)`` (optionally with ``enable_resistance_port``)
or a small custom component.

Numerical tips:

- Pass ``scale=True`` to :class:`AcausalCompiler` when variables span several
  orders of magnitude (e.g. mixing pressure in Pa with mass in kg); this
  enables automatic variable scaling to improve Jacobian conditioning.
- Provide explicit ``initial_*_fixed=True`` conditions on dynamic components
  whenever possible to achieve consistent, deterministic initialization.
- For stiff systems, prefer the JAX backend (``leaf_backend="jax"``) which
  uses an implicit BDF integrator internally.

References:

- Pantelides, C. C. (1988). The consistent initialization of differential-algebraic
  systems. *SIAM J. Sci. Stat. Comput.*, 9(2), 213–231.
- Mattsson, S. E. & Söderlind, G. (1993). Index reduction in differential-algebraic
  equations using dummy derivatives. *SIAM J. Sci. Comput.*, 14(3), 677–692.
"""

from .acausal_compiler import AcausalCompiler, AcausalSystem
from .acausal_diagram import AcausalDiagram
from .component_library.base import EqnEnv
from .component_library import (
    electrical,
    rotational,
    translational,
    thermal,
    fluid,
    hydraulic,
    battery,
)
from .component_library import fluid_media
from .error import AcausalCompilerError, AcausalModelError
from ..library.neural_dae import NeuralDAEBlock, add_neural_correction

__all__ = [
    "AcausalSystem",
    "AcausalCompiler",
    "AcausalDiagram",
    "EqnEnv",
    "NeuralDAEBlock",
    "add_neural_correction",
    "electrical",
    "rotational",
    "translational",
    "thermal",
    "fluid",
    "hydraulic",
    "battery",
    "AcausalCompilerError",
    "AcausalModelError",
    "fluid_media",
]
