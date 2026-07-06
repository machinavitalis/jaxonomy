# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch

import jaxonomy
from jaxonomy.library import Gain, Integrator, Sine
from jaxonomy.simulation.batch import simulate_batch
from jaxonomy.simulation.simulator import Simulator, simulate
from jaxonomy.backend.ode_solver import ODESolverError

pytestmark = pytest.mark.minimal


def build_test_diagram():
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=2.0, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="improvements_test")


def test_simulator_compile():
    """Verify that Simulator.compile() executes without error."""
    diagram = build_test_diagram()
    context = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
    
    sim = Simulator(diagram, options=opts)
    
    # Verify compile method exists and runs successfully
    assert hasattr(sim, "compile")
    sim.compile(0.5, context)


def test_precision_options():
    """Verify that SimulatorOptions.precision overrides global config locally."""
    diagram = build_test_diagram()
    context = diagram.create_context()
    
    # Assert default state of x64
    orig_x64 = jax.config.read("jax_enable_x64")
    
    try:
        # Force float32 precision
        opts_f32 = jaxonomy.SimulatorOptions(
            math_backend="jax",
            precision="float32",
            max_major_steps=10
        )
        res_f32 = simulate(
            diagram,
            context,
            t_span=(0.0, 0.2),
            options=opts_f32,
            recorded_signals={"y": diagram["integ"].output_ports[0]}
        )
        # Results should be in float32
        assert res_f32.outputs["y"].dtype == jnp.float32
        
        # Force float64 precision
        opts_f64 = jaxonomy.SimulatorOptions(
            math_backend="jax",
            precision="float64",
            max_major_steps=10
        )
        res_f64 = simulate(
            diagram,
            context,
            t_span=(0.0, 0.2),
            options=opts_f64,
            recorded_signals={"y": diagram["integ"].output_ports[0]}
        )
        # Results should be in float64
        assert res_f64.outputs["y"].dtype == jnp.float64
        
        # Verify global config state is preserved
        assert jax.config.read("jax_enable_x64") == orig_x64
        
    finally:
        jax.config.update("jax_enable_x64", orig_x64)


def test_lazy_batch_simulation():
    """Verify simulate_batch with lazy=True returns raw unfinalized JAX arrays."""
    diagram = build_test_diagram()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
    recorded = {"y": diagram["integ"].output_ports[0]}
    
    # 1. Test use_vmap=False (scan path)
    res_scan = simulate_batch(
        diagram,
        t_span=(0.0, 0.5),
        param_batches={"gain.gain": jnp.array([1.0, 2.0])},
        options=opts,
        recorded_signals=recorded,
        use_vmap=False,
        lazy=True
    )
    
    # Check that outputs are raw unfinalized JAX arrays
    assert isinstance(res_scan.outputs["y"], jax.Array)
    assert res_scan.outputs["y"].ndim == 2  # (N, max_steps)
    assert res_scan.outputs["y"].shape[0] == 2
    
    # 2. Test use_vmap=True (vmap path)
    res_vmap = simulate_batch(
        diagram,
        t_span=(0.0, 0.5),
        param_batches={"gain.gain": jnp.array([1.0, 2.0])},
        options=opts,
        recorded_signals=recorded,
        use_vmap=True,
        lazy=True
    )
    
    assert isinstance(res_vmap.outputs["y"], jax.Array)
    assert res_vmap.outputs["y"].ndim == 2
    assert res_vmap.outputs["y"].shape[0] == 2


def test_tpu_float64_implicit_validation():
    """Verify BDFSolver raises ODESolverError when running float64 on TPU."""
    diagram = build_test_diagram()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        precision="float64",
        ode_solver_method="bdf",
        max_major_steps=10
    )
    
    # Patch default_backend to return 'tpu'
    with patch("jax.default_backend", return_value="tpu"):
        with pytest.raises(ODESolverError, match="TPU backend does not support double-precision"):
            simulate(
                diagram,
                diagram.create_context(),
                t_span=(0.0, 0.2),
                options=opts
            )
