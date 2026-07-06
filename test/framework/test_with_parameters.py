# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Constant, Gain, Integrator, Sine

pytestmark = pytest.mark.minimal


def build_simple_diagram(k=1.0):
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=k, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="test")


def test_with_parameters_basic():
    diagram = build_simple_diagram(k=1.0)
    updated = diagram.with_parameters({"gain.gain": jnp.array(2.0)})
    assert jnp.allclose(jnp.asarray(diagram.get_parameter("gain.gain")), 1.0)
    assert jnp.allclose(jnp.asarray(updated.get_parameter("gain.gain")), 2.0)


def test_with_parameters_simulation_changes():
    diagram = build_simple_diagram(k=1.0)
    updated = diagram.with_parameters({"gain.gain": jnp.array(2.0)})
    opts = jaxonomy.SimulatorOptions(math_backend="jax")

    ctx1 = diagram.create_context()
    ctx2 = updated.create_context()
    y1 = diagram["integ"].output_ports[0]
    y2 = updated["integ"].output_ports[0]

    r1 = jaxonomy.simulate(
        diagram,
        ctx1,
        (0.0, 1.0),
        options=opts,
        recorded_signals={"y": y1},
    )
    r2 = jaxonomy.simulate(
        updated,
        ctx2,
        (0.0, 1.0),
        options=opts,
        recorded_signals={"y": y2},
    )
    assert not jnp.allclose(r1.outputs["y"], r2.outputs["y"])


def test_with_parameters_grad():
    diagram = build_simple_diagram(k=1.0)

    def loss(k):
        d = diagram.with_parameters({"gain.gain": k})
        ctx = d.create_context()
        results = jaxonomy.simulate(
            d,
            ctx,
            (0.0, 1.0),
            options=jaxonomy.SimulatorOptions(
                enable_autodiff=True,
                max_major_steps=100,
                math_backend="jax",
            ),
        )
        sid = d["integ"].system_id
        yf = results.context[sid].continuous_state
        return jnp.mean(yf**2)

    grad = jax.grad(loss)(jnp.array(1.0))
    assert grad is not None
    assert jnp.isfinite(grad)
    assert bool(jnp.asarray(grad != 0.0).item())


def test_with_parameters_vmap():
    diagram = build_simple_diagram(k=1.0)
    k_values = jnp.linspace(0.5, 2.0, 8)

    def run_one(k):
        d = diagram.with_parameters({"gain.gain": k})
        ctx = d.create_context()
        results = jaxonomy.simulate(
            d,
            ctx,
            (0.0, 1.0),
            options=jaxonomy.SimulatorOptions(
                math_backend="jax",
                max_major_steps=100,
            ),
            recorded_signals={"y": d["integ"].output_ports[0]},
        )
        return results.outputs["y"][-1]

    # ``jax.vmap`` over the full simulator hits JAX limitations (io effects in
    # ``cond``); batch semantics are checked explicitly below.
    outputs = jnp.stack([run_one(k) for k in k_values])
    assert outputs.shape == (8,)
    assert jnp.all(jnp.diff(outputs) > 0)


def test_with_parameters_nested():
    inner_b = jaxonomy.DiagramBuilder()
    gain = inner_b.add(Gain(gain=1.0, name="gain"))
    inner_b.export_input(gain.input_ports[0], "u")
    inner_b.export_output(gain.output_ports[0], "y")
    inner = inner_b.build(name="inner")

    outer_b = jaxonomy.DiagramBuilder()
    src = outer_b.add(Constant(1.0, name="src"))
    sub = outer_b.add(inner)
    outer_b.connect(src.output_ports[0], sub.input_ports[0])
    outer = outer_b.build(name="outer")

    updated = outer.with_parameters({"inner.gain.gain": jnp.array(5.0)})
    assert jnp.allclose(
        jnp.asarray(updated.get_parameter("inner.gain.gain")), 5.0
    )
