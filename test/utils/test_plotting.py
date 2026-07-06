# SPDX-License-Identifier: MIT

import sys
import unittest.mock as mock

import pytest

import jaxonomy
from jaxonomy.library import Gain, Integrator, Sine
from jaxonomy.utils.plotting import plot_batch_results, plot_results

pytestmark = pytest.mark.minimal


def _simple_diagram():
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=1.0, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="plot_test")


def test_plot_results_runs_without_error():
    diagram = _simple_diagram()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.5),
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=80),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    fig = plot_results(results, show=False)
    assert fig is not None


def test_plot_results_returns_figure():
    import matplotlib.figure

    diagram = _simple_diagram()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.5),
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=80),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    fig = plot_results(results, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_batch_results():
    from jaxonomy.simulation.batch import simulate_batch
    import jax.numpy as jnp

    diagram = _simple_diagram()
    k_values = jnp.linspace(0.5, 1.5, 4)
    results = simulate_batch(
        diagram,
        t_span=(0.0, 0.5),
        param_batches={"gain.gain": k_values},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=80),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    fig = plot_batch_results(results, show=False)
    assert fig is not None


def test_plot_no_matplotlib_raises():
    diagram = _simple_diagram()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.2),
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=40),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    mpl_keys = [k for k in list(sys.modules) if k.startswith("matplotlib")]
    saved = {k: sys.modules.pop(k) for k in mpl_keys}
    try:
        with mock.patch.dict("sys.modules", {"matplotlib": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                plot_results(results, show=False)
    finally:
        sys.modules.pop("matplotlib", None)
        for k, mod in saved.items():
            sys.modules[k] = mod
