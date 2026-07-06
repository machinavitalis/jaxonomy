import jax
import jax.numpy as jnp

from jaxonomy import DiagramBuilder, simulate, SimulatorOptions
from jaxonomy.library.random import RandomNumber


def build_and_run(seed=0, n_steps=2):
    builder = DiagramBuilder()
    noise = builder.add(RandomNumber(dt=0.1, seed=seed, name="noise"))
    builder.export_output(noise.output_ports[0], "y")
    diagram = builder.build()
    
    context = diagram.create_context()
    result = simulate(
        diagram,
        context,
        tspan=(0.0, n_steps * 0.1),
        options=SimulatorOptions(
            enable_autodiff=True, 
            max_major_steps=100
        )
    )
    return diagram.get_output_port("y").eval(result.context)


def test_different_seeds_produce_different_output():
    out1 = build_and_run(seed=1)
    out2 = build_and_run(seed=2)
    assert not jnp.allclose(out1, out2)


def test_same_seed_produces_same_output():
    out1 = build_and_run(seed=42)
    out2 = build_and_run(seed=42)
    assert jnp.allclose(out1, out2)


def test_with_key_produces_independent_streams():
    key1, key2 = jax.random.split(jax.random.PRNGKey(10), 2)
    
    builder1 = DiagramBuilder()
    noise1 = builder1.add(RandomNumber.with_key(key1, dt=0.1, name="noise"))
    builder1.export_output(noise1.output_ports[0], "y")
    diagram1 = builder1.build()
    
    builder2 = DiagramBuilder()
    noise2 = builder2.add(RandomNumber.with_key(key2, dt=0.1, name="noise"))
    builder2.export_output(noise2.output_ports[0], "y")
    diagram2 = builder2.build()
    
    opt = SimulatorOptions(enable_autodiff=True, max_major_steps=100)
    res1 = simulate(diagram1, diagram1.create_context(), tspan=(0.0, 0.2), options=opt)
    res2 = simulate(diagram2, diagram2.create_context(), tspan=(0.0, 0.2), options=opt)
    
    out1 = diagram1.get_output_port("y").eval(res1.context)
    out2 = diagram2.get_output_port("y").eval(res2.context)
    
    assert not jnp.allclose(out1, out2)


def test_vmap_with_different_keys():
    base_diagram = DiagramBuilder()
    noise = base_diagram.add(RandomNumber(dt=0.1, name="noise"))
    base_diagram.export_output(noise.output_ports[0], "y")
    base_diagram = base_diagram.build()
    
    def run_one(key):
        # Substitute the state containing the key instead of rebuilding the diagram
        # But `with_parameters` in collymator can set the state or parameter.
        # Actually, let's just use the with_key explicitly inside vmap to avoid 
        # API mismatch or we can rebuild since JAX tracks it.
        builder = DiagramBuilder()
        noise = builder.add(RandomNumber.with_key(key, dt=0.1, name="noise"))
        builder.export_output(noise.output_ports[0], "y")
        diagram = builder.build()
        result = simulate(diagram, diagram.create_context(), tspan=(0.0, 0.2), 
                          options=SimulatorOptions(enable_autodiff=True, max_major_steps=100))
        return diagram.get_output_port("y").eval(result.context)

    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    # vmap over diagrams built with keys!
    outputs = jax.vmap(run_one)(keys)
    
    for i in range(4):
        for j in range(i + 1, 4):
            assert not jnp.allclose(outputs[i], outputs[j])

