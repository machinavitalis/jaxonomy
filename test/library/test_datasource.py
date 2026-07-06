# SPDX-License-Identifier: MIT

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import DataSource, Gain, Sine, SimulationResultsSource

pytestmark = pytest.mark.minimal


def _sine_diagram():
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=1.0, name="gain"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    return builder.build(name="ds_src_test"), gain


def test_simulation_results_source():
    diagram, gain = _sine_diagram()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 1.0),
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=120),
        recorded_signals={"y": gain.output_ports[0]},
    )

    source = SimulationResultsSource(results=results, signal_name="y", name="replay")

    builder = jaxonomy.DiagramBuilder()
    src = builder.add(source)
    d2 = builder.build(name="replay_root")
    ctx2 = d2.create_context()
    r2 = jaxonomy.simulate(
        d2,
        ctx2,
        (0.0, 1.0),
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=120),
        recorded_signals={"replay": src.output_ports[0]},
    )

    assert jnp.allclose(
        jnp.asarray(r2.outputs["replay"]),
        jnp.asarray(results.outputs["y"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_datasource_csv_with_header():
    content = "t,y\n0.0,0.0\n0.5,1.0\n1.0,2.0\n"
    fd, path = tempfile.mkstemp(suffix=".csv", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        ds = DataSource(
            file_name=path,
            header_as_first_row=True,
            time_samples_as_column=True,
            time_column="t",
            column="y",
            interpolation="linear",
            name="csv_ds",
        )
        builder = jaxonomy.DiagramBuilder()
        block = builder.add(ds)
        diagram = builder.build(name="csv_test")
        ctx = diagram.create_context()
        t0 = jnp.array(0.25)
        y0 = block.output_ports[0].eval(ctx.with_time(t0))
        assert float(y0) == pytest.approx(0.5, rel=0, abs=1e-6)
    finally:
        os.unlink(path)
