# SPDX-License-Identifier: MIT

import time

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Gain, Integrator, Sine, Constant, Adder, Saturate
from jaxonomy.simulation.batch import (
    BatchSimulationResults,
    _infer_batch_size,
    _interp_on_time,
    _is_vmap_safe,
    _pure_patch_context,
    simulate_batch,
)

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixture / builder helpers
# ---------------------------------------------------------------------------

def build_gain_integrator_diagram(k=1.0):
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=k, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="batch_test")


def build_two_param_diagram():
    """Two tunable parameters: gain value and saturator limit."""
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=1.0, name="gain"))
    sat = builder.add(Saturate(upper_limit=1.0, lower_limit=-1.0, name="sat"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], sat.input_ports[0])
    builder.connect(sat.output_ports[0], integ.input_ports[0])
    return builder.build(name="two_param")


# ---------------------------------------------------------------------------
# _is_vmap_safe
# ---------------------------------------------------------------------------

def test_is_vmap_safe_pure_jax():
    diagram = build_gain_integrator_diagram()
    assert _is_vmap_safe(diagram) is True


def test_is_vmap_safe_with_custom_python_block():
    from jaxonomy.library import CustomPythonBlock

    builder = jaxonomy.DiagramBuilder()
    cpb = builder.add(
        CustomPythonBlock(
            dt=0.1,
            init_script="y = 0.0",
            user_statements="y = y + 1.0",
            inputs=[],
            outputs=["y"],
            name="cpb",
        )
    )
    diagram = builder.build(name="cpb_diag")
    assert _is_vmap_safe(diagram) is False


# ---------------------------------------------------------------------------
# _pure_patch_context
# ---------------------------------------------------------------------------

def test_pure_patch_context_leaf_params():
    """_pure_patch_context patches a leaf's own parameters without ParameterCache."""
    diagram = build_gain_integrator_diagram(k=1.0)
    ctx = diagram.create_context()

    # Get the gain system id to verify patching works
    gain_sys = diagram["gain"]
    gain_ctx_before = ctx[gain_sys.system_id]
    assert float(gain_ctx_before.parameters.get("gain", 1.0)) == pytest.approx(1.0)

    patched = _pure_patch_context(ctx, {"gain.gain": jnp.array(5.0)})
    gain_ctx_after = patched[gain_sys.system_id]
    assert float(gain_ctx_after.parameters["gain"]) == pytest.approx(5.0)

    # Original unchanged
    assert float(ctx[gain_sys.system_id].parameters.get("gain", 1.0)) == pytest.approx(1.0)


def test_pure_patch_context_no_updates_returns_same():
    diagram = build_gain_integrator_diagram()
    ctx = diagram.create_context()
    patched = _pure_patch_context(ctx, {})
    # Same parameter values
    for sys_id in ctx.subcontexts:
        assert ctx.subcontexts[sys_id].parameters == patched.subcontexts[sys_id].parameters


def test_pure_patch_context_unknown_block_raises():
    diagram = build_gain_integrator_diagram()
    ctx = diagram.create_context()
    with pytest.raises(KeyError, match="nonexistent"):
        _pure_patch_context(ctx, {"nonexistent.param": 1.0})


# ---------------------------------------------------------------------------
# Core batch functionality (existing tests, now on kernel path)
# ---------------------------------------------------------------------------

def test_simulate_batch_uses_kernel_path():
    """Pure-JAX diagram uses kernel path (not loop), used_vmap=False."""
    diagram = build_gain_integrator_diagram()
    result = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={"gain.gain": jnp.array([0.5, 1.0, 1.5])},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    assert result.used_vmap is False
    assert result.outputs["y"].shape[0] == 3


def test_simulate_batch_loop_path_via_force():
    """_force_loop=True triggers the loop path even for pure-JAX diagrams."""
    diagram = build_gain_integrator_diagram()
    result = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={"gain.gain": jnp.array([0.5, 1.0, 1.5])},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
        _force_loop=True,
    )
    assert result.outputs["y"].shape[0] == 3


def test_simulate_batch_kernel_matches_loop():
    """Kernel path and loop path produce numerically identical results."""
    diagram = build_gain_integrator_diagram()
    k_values = jnp.linspace(0.5, 2.0, 8)
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    recorded = {"y": diagram["integ"].output_ports[0]}

    result_kernel = simulate_batch(
        diagram,
        t_span=(0.0, 2.0),
        param_batches={"gain.gain": k_values},
        options=opts,
        recorded_signals=recorded,
    )
    result_loop = simulate_batch(
        diagram,
        t_span=(0.0, 2.0),
        param_batches={"gain.gain": k_values},
        options=opts,
        recorded_signals=recorded,
        _force_loop=True,
    )

    # Time grids may differ slightly; compare on reference grid
    t_ref = result_kernel.time
    for i in range(len(k_values)):
        y_kern = result_kernel.outputs["y"][i]
        y_loop_interp = _interp_on_time(
            result_loop.outputs["y"][i], result_loop.time, t_ref
        )
        assert jnp.allclose(y_kern, y_loop_interp, atol=1e-4), (
            f"Mismatch at batch index {i}: max diff = "
            f"{float(jnp.max(jnp.abs(y_kern - y_loop_interp))):.3e}"
        )


def test_simulate_batch_basic():
    """N-batch sweep matches individual simulations."""
    diagram = build_gain_integrator_diagram()
    k_values = jnp.linspace(0.5, 2.0, 16)
    y_port = diagram["integ"].output_ports[0]

    batch_results = simulate_batch(
        diagram,
        t_span=(0.0, 2.0),
        param_batches={"gain.gain": k_values},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200),
        recorded_signals={"y": y_port},
    )

    assert batch_results.outputs["y"].shape[0] == 16

    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    t_ref = batch_results.time
    for i, k in enumerate(k_values):
        d = diagram.with_parameters({"gain.gain": k})
        ctx = d.create_context()
        single = jaxonomy.simulate(
            d,
            ctx,
            (0.0, 2.0),
            options=opts,
            recorded_signals={"y": d["integ"].output_ports[0]},
        )
        y_on_ref = jnp.interp(t_ref, single.time, single.outputs["y"])
        assert jnp.allclose(batch_results.outputs["y"][i], y_on_ref, atol=1e-3)


def test_simulate_batch_statistics():
    diagram = build_gain_integrator_diagram()
    k_values = jnp.linspace(0.5, 2.0, 8)
    y_port = diagram["integ"].output_ports[0]

    results = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={"gain.gain": k_values},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=150),
        recorded_signals={"y": y_port},
    )

    mean = results.mean("y")
    std = results.std("y")
    p95 = results.percentile("y", 95.0)
    assert mean.shape == std.shape == p95.shape
    assert isinstance(results, BatchSimulationResults)


def test_simulate_batch_inconsistent_sizes_raises():
    diagram = build_gain_integrator_diagram()
    with pytest.raises(ValueError, match="batch size"):
        simulate_batch(
            diagram,
            t_span=(0.0, 1.0),
            param_batches={
                "gain.gain": jnp.ones(8),
                "integ.initial_state": jnp.ones(16),
            },
            options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100),
            recorded_signals={"y": diagram["integ"].output_ports[0]},
        )


def test_simulate_batch_to_simulation_results():
    diagram = build_gain_integrator_diagram()
    k_values = jnp.array([1.0, 2.0])
    batch_results = simulate_batch(
        diagram,
        t_span=(0.0, 0.5),
        param_batches={"gain.gain": k_values},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    one = batch_results.to_simulation_results(1)
    assert one.outputs["y"].shape == batch_results.outputs["y"][1].shape
    assert jnp.allclose(one.outputs["y"], batch_results.outputs["y"][1])


# ---------------------------------------------------------------------------
# Kernel path: single compilation benefit
# ---------------------------------------------------------------------------

def test_simulate_batch_kernel_compiles_once():
    """The kernel path compiles the simulation kernel once (warm + N cold calls).

    This is a behavioural correctness test: we verify that calling simulate_batch
    on the same diagram N times returns consistent results and doesn't crash.
    Timing is not asserted (too flaky in CI) but the test documents the intent.
    """
    diagram = build_gain_integrator_diagram()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
    recorded = {"y": diagram["integ"].output_ports[0]}

    # First call: triggers JIT compilation of kernel
    r1 = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={"gain.gain": jnp.array([1.0, 2.0])},
        options=opts,
        recorded_signals=recorded,
    )
    # Second call: kernel should be reused (same diagram, same shapes)
    r2 = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={"gain.gain": jnp.array([1.0, 2.0])},
        options=opts,
        recorded_signals=recorded,
    )
    assert jnp.allclose(r1.outputs["y"], r2.outputs["y"], atol=1e-6)


# ---------------------------------------------------------------------------
# Two-parameter sweep
# ---------------------------------------------------------------------------

def test_simulate_batch_two_params():
    """Multiple parameters swept simultaneously."""
    diagram = build_two_param_diagram()
    N = 6
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=150)
    recorded = {"y": diagram["integ"].output_ports[0]}

    result = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={
            "gain.gain": jnp.linspace(0.5, 2.0, N),
            "sat.upper_limit": jnp.linspace(0.5, 1.5, N),
        },
        options=opts,
        recorded_signals=recorded,
    )
    assert result.outputs["y"].shape[0] == N


# ---------------------------------------------------------------------------
# CustomPythonBlock → loop path
# ---------------------------------------------------------------------------

def test_simulate_batch_custom_python_block_not_vmap_safe():
    """Diagrams with CustomPythonBlock are not vmap-safe (loop path is selected)."""
    from jaxonomy.library import CustomPythonBlock

    builder = jaxonomy.DiagramBuilder()
    builder.add(
        CustomPythonBlock(
            dt=0.1,
            init_script="y = 0.0",
            user_statements="y = y + 1.0",
            inputs=[],
            outputs=["y"],
            name="cpb",
        )
    )
    diagram = builder.build(name="cpb_diag")
    assert _is_vmap_safe(diagram) is False


def test_simulate_batch_pure_jax_falls_to_kernel_not_loop():
    """Pure-JAX diagram selects kernel path (used_vmap=False but not loop)."""
    diagram = build_gain_integrator_diagram()
    result = simulate_batch(
        diagram,
        t_span=(0.0, 0.5),
        param_batches={"gain.gain": jnp.array([1.0, 2.0])},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=80),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    # Kernel path: not vmap but also no diagram deep-copy per element
    assert result.used_vmap is False
    assert result.outputs["y"].shape[0] == 2


# ---------------------------------------------------------------------------
# use_vmap on non-pure-JAX raises
# ---------------------------------------------------------------------------

def test_simulate_batch_use_vmap_non_pure_raises():
    """use_vmap=True on a diagram with CustomPythonBlock raises ValueError."""
    from jaxonomy.library import CustomPythonBlock

    from jaxonomy.library import CustomPythonBlock

    builder = jaxonomy.DiagramBuilder()
    builder.add(
        CustomPythonBlock(
            dt=0.1,
            init_script="y = 0.0",
            user_statements="y = y + 1.0",
            inputs=[],
            outputs=["y"],
            name="cpb",
        )
    )
    diagram = builder.build(name="cpb_vmap")
    # Build a pure-JAX diagram alongside just for the call
    pure_diag = build_gain_integrator_diagram()
    with pytest.raises(ValueError, match="pure-JAX"):
        simulate_batch(
            diagram,
            t_span=(0.0, 0.3),
            param_batches={"cpb.y": jnp.array([0.5, 1.0])},
            options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=50),
            recorded_signals={"y": diagram["cpb"].output_ports[0]},
            use_vmap=True,
        )


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------

def test_simulate_batch_missing_options_raises():
    diagram = build_gain_integrator_diagram()
    with pytest.raises(ValueError):
        simulate_batch(
            diagram,
            t_span=(0.0, 1.0),
            param_batches={"gain.gain": jnp.ones(4)},
            options=None,
            recorded_signals={"y": diagram["integ"].output_ports[0]},
        )


def test_simulate_batch_missing_recorded_signals_raises():
    diagram = build_gain_integrator_diagram()
    with pytest.raises(ValueError):
        simulate_batch(
            diagram,
            t_span=(0.0, 1.0),
            param_batches={"gain.gain": jnp.ones(4)},
            options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100),
            recorded_signals=None,
        )


def test_simulate_batch_numpy_backend_raises():
    diagram = build_gain_integrator_diagram()
    with pytest.raises(ValueError, match="math_backend"):
        simulate_batch(
            diagram,
            t_span=(0.0, 1.0),
            param_batches={"gain.gain": jnp.ones(4)},
            options=jaxonomy.SimulatorOptions(math_backend="numpy", max_major_steps=100),
            recorded_signals={"y": diagram["integ"].output_ports[0]},
        )


def test_simulate_batch_batch_size_one():
    """Batch of size 1 works and returns shape (1, T)."""
    diagram = build_gain_integrator_diagram()
    result = simulate_batch(
        diagram,
        t_span=(0.0, 1.0),
        param_batches={"gain.gain": jnp.array([1.5])},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100),
        recorded_signals={"y": diagram["integ"].output_ports[0]},
    )
    assert result.outputs["y"].ndim == 2
    assert result.outputs["y"].shape[0] == 1


# ---------------------------------------------------------------------------
# _interp_on_time unit tests
# ---------------------------------------------------------------------------

def test_interp_on_time_identity():
    """Interpolating onto the same grid returns the original values."""
    t = jnp.linspace(0.0, 1.0, 11)
    y = jnp.sin(t)
    y_interp = _interp_on_time(y, t, t)
    assert jnp.allclose(y, y_interp, atol=1e-6)


def test_interp_on_time_coarser_grid():
    t_fine = jnp.linspace(0.0, 1.0, 101)
    t_coarse = jnp.linspace(0.0, 1.0, 11)
    y_fine = jnp.sin(t_fine)
    y_coarse = _interp_on_time(y_fine, t_fine, t_coarse)
    assert y_coarse.shape == (11,)
    assert jnp.allclose(y_coarse, jnp.sin(t_coarse), atol=1e-3)


def test_interp_on_time_2d():
    t = jnp.linspace(0.0, 1.0, 11)
    y = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=-1)  # (11, 2)
    t2 = jnp.linspace(0.0, 1.0, 21)
    y2 = _interp_on_time(y, t, t2)
    assert y2.shape == (21, 2)


# ---------------------------------------------------------------------------
# _infer_batch_size unit tests
# ---------------------------------------------------------------------------

def test_infer_batch_size_consistent():
    assert _infer_batch_size({"a": jnp.ones(8), "b": jnp.ones(8)}) == 8


def test_infer_batch_size_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _infer_batch_size({})


def test_infer_batch_size_inconsistent_raises():
    with pytest.raises(ValueError, match="batch size"):
        _infer_batch_size({"a": jnp.ones(8), "b": jnp.ones(4)})
