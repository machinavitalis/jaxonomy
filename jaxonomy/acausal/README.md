# Acausal Framework (Experimental)

This package provides an experimental acausal (physical) modeling layer for jaxonomy.
It compiles connected physical components into an `AcausalSystem` (`LeafSystem`) that
can be simulated inside regular jaxonomy diagrams.

## Design intent

The implementation follows the Modelica-style acausal modelling tradition but starts
directly from symbolic Python component definitions (SymPy expressions) rather than a
separate text language/parser pipeline.

High-level flow:

1. Build an `AcausalDiagram` from component instances and connections.
2. `AcausalCompiler` transforms diagram equations into index-reduced DAEs.
3. The resulting `AcausalSystem` is inserted into a normal jaxonomy diagram and simulated.

## Minimal working example (electrical RC)

```python
import jaxonomy
from jaxonomy.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.experimental import electrical as elec

ev = EqnEnv()
ad = AcausalDiagram()
v1 = elec.VoltageSource(ev, name="v1", v=1.0)
r1 = elec.Resistor(ev, name="r1", R=1.0)
c1 = elec.Capacitor(ev, name="c1", C=1.0, initial_voltage=0.0, initial_voltage_fixed=True)
gnd = elec.Ground(ev, name="gnd")

ad.connect(v1, "p", r1, "n")
ad.connect(r1, "p", c1, "p")
ad.connect(c1, "n", v1, "n")
ad.connect(v1, "n", gnd, "p")

ac = AcausalCompiler(ev, ad, verbose=False)
asys = ac()

builder = jaxonomy.DiagramBuilder()
asys = builder.add(asys)
diagram = builder.build()
context = diagram.create_context(check_types=True)

results = jaxonomy.simulate(
    diagram,
    context,
    (0.0, 5.0),
    recorded_signals={"x": asys.output_ports[0]},
)
```

## Useful imports

For common blocks:

```python
from jaxonomy.experimental.acausal.component_library import (
    Resistor,
    Capacitor,
    VoltageSource,
    ACVoltageSource,
    IdealTransformer,
    IdealSwitch,
    DCMotorSimple,
    Mass,
    TranslationalSpring,
    RotationalSpring,
    HardStop,
    Clutch,
    LeadScrew,
    HeatCapacitor,
    TemperatureSource,
)
```

Domain modules are also available:

```python
from jaxonomy.experimental import electrical, rotational, translational, thermal, fluid, hydraulic
```

## Notes

- This package is still experimental; APIs may evolve.
- Some advanced fluid/thermal paths are known to be sensitive to initialization quality.
- Enable `verbose=True` in `AcausalCompiler` when debugging model construction.

## Tests

- Core acausal tests: `test/acausal/`
- JSON/app-level acausal tests: `test/app/Acausal/`