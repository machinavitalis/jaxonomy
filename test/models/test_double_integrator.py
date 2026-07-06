# SPDX-License-Identifier: MIT

import pytest


import jaxonomy
from jaxonomy.library import (
    Integrator,
    Sine,
)
from jaxonomy.backend import numpy_api as npa
from jaxonomy import logging


logging.set_file_handler("test.log")

pytestmark = pytest.mark.minimal


def test_double_integrator(dtype=npa.float64):
    builder = jaxonomy.DiagramBuilder()
    Sin_0 = builder.add(Sine(name="Sin_0"))

    x0 = dtype(0.0)
    v0 = dtype(-1.0)

    Integrator_0 = builder.add(Integrator(v0))  # v
    Integrator_1 = builder.add(Integrator(x0))  # x

    builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])

    diagram = builder.build()
    ctx = diagram.create_context()

    print([(p.name, p.system) for p in Sin_0.output_ports])
    print([(p.name, p.system) for p in Integrator_0.input_ports])
    print([(p.name, p.system) for p in Integrator_0.output_ports])
    print([(p.name, p.system) for p in Integrator_1.input_ports])

    t = npa.linspace(0.0, 10.0, 100, dtype=dtype)
    options = jaxonomy.SimulatorOptions(atol=1e-8, rtol=1e-6)
    recorded_signals = {
        "x": Integrator_1.output_ports[0],
        "v": Integrator_0.output_ports[0],
    }
    sol = jaxonomy.simulate(
        diagram,
        ctx,
        (t[0], t[-1]),
        options=options,
        recorded_signals=recorded_signals,
    )
    x, v = sol.outputs["x"], sol.outputs["v"]
    t = sol.time

    print(x)
    print(v)
    print(npa.sin(t))
    print(npa.cos(t))
    print(npa.std(x - npa.sin(t)))
    print(npa.std(x + npa.sin(t)))
    assert npa.allclose(x, -npa.sin(t), rtol=1e-4, atol=1e-6)
    assert npa.allclose(v, -npa.cos(t), rtol=1e-4, atol=1e-6)

    assert x.dtype == dtype
    assert v.dtype == dtype
