# SPDX-License-Identifier: MIT
"""
T-008 — public submodel-as-pure-function API tests.

Covers:

  - single-input single-output via `export_input` / `export_output`
  - multi-input
  - `jax.grad` through the closure matches analytic / FD gradient
  - `jax.vmap` over inputs with context held constant
  - `jax.jit` wrapping
  - auto-seed placeholders → `create_context` succeeds on dangling
    exported inputs
  - referential transparency: port is unfixed after call returns
  - validation: wrong input count, empty outputs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Adder, Gain
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _gain_diagram(k=2.0):
    bld = jaxonomy.DiagramBuilder()
    g = bld.add(Gain(k, name="g"))
    bld.export_input(g.input_ports[0], name="u")
    bld.export_output(g.output_ports[0], name="y")
    return bld.build()


def _adder_diagram(n=3, operators="+-+"):
    bld = jaxonomy.DiagramBuilder()
    a = bld.add(Adder(n, operators=operators, name="a"))
    for i in range(n):
        bld.export_input(a.input_ports[i], name=f"u{i}")
    bld.export_output(a.output_ports[0], name="y")
    return bld.build()


# ── Single input / output ──────────────────────────────────────────────────


def test_single_input_single_output():
    d = _gain_diagram(2.0)
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()
    y = f(ctx, jnp.array(3.0))
    assert float(y) == 6.0


def test_repeated_calls_independent():
    """Referential transparency: two calls with different inputs produce
    independent, correct outputs.  Guards against tracer leaking or
    fix-state bleed between calls."""
    d = _gain_diagram(2.0)
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()
    y1 = f(ctx, jnp.array(1.5))
    y2 = f(ctx, jnp.array(3.5))
    assert float(y1) == 3.0
    assert float(y2) == 7.0


# ── Multi input ────────────────────────────────────────────────────────────


def test_multi_input_adder():
    """Adder(3, '+-+'): y = a - b + c."""
    d = _adder_diagram(3, "+-+")
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()
    y = f(ctx, jnp.array(1.0), jnp.array(2.0), jnp.array(4.0))
    assert float(y) == 3.0


# ── jax.grad ───────────────────────────────────────────────────────────────


def test_grad_single_input():
    d = _gain_diagram(2.0)
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()

    def loss(u):
        return f(ctx, u)

    dy_du = jax.grad(loss)(jnp.array(5.0))
    assert float(dy_du) == 2.0


def test_grad_multi_input():
    d = _adder_diagram(3, "+-+")
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()

    def loss(a, b, c):
        return f(ctx, a, b, c)

    ga, gb, gc = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.array(1.0), jnp.array(2.0), jnp.array(4.0)
    )
    assert float(ga) == 1.0
    assert float(gb) == -1.0
    assert float(gc) == 1.0


# ── jax.vmap ───────────────────────────────────────────────────────────────


def test_vmap_over_inputs():
    """in_axes=(None, 0): context constant, input batched."""
    d = _gain_diagram(2.0)
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()
    xs = jnp.arange(5, dtype=jnp.float64)
    ys = jax.vmap(f, in_axes=(None, 0))(ctx, xs)
    np.testing.assert_allclose(np.asarray(ys), 2 * np.asarray(xs))


# ── jax.jit ────────────────────────────────────────────────────────────────


def test_jit_wrapping():
    d = _gain_diagram(3.0)
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()

    @jax.jit
    def jit_f(u):
        return f(ctx, u)

    assert float(jit_f(jnp.array(2.0))) == 6.0


# ── Composition: Gain → Gain through export ────────────────────────────────


def test_composed_diagram():
    bld = jaxonomy.DiagramBuilder()
    g1 = bld.add(Gain(2.0, name="g1"))
    g2 = bld.add(Gain(5.0, name="g2"))
    bld.connect(g1.output_ports[0], g2.input_ports[0])
    bld.export_input(g1.input_ports[0], name="u")
    bld.export_output(g2.output_ports[0], name="y")
    d = bld.build()
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()
    assert float(f(ctx, jnp.array(1.0))) == 10.0


def test_composed_diagram_grad_matches_fd():
    bld = jaxonomy.DiagramBuilder()
    g1 = bld.add(Gain(2.0, name="g1"))
    g2 = bld.add(Gain(5.0, name="g2"))
    bld.connect(g1.output_ports[0], g2.input_ports[0])
    bld.export_input(g1.input_ports[0], name="u")
    bld.export_output(g2.output_ports[0], name="y")
    d = bld.build()
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()

    def loss(u):
        return f(ctx, u)

    g = jax.grad(loss)(jnp.array(1.0))
    eps = 1e-6
    fd = (loss(jnp.array(1.0 + eps)) - loss(jnp.array(1.0 - eps))) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-4


# ── Validation ─────────────────────────────────────────────────────────────


def test_raises_on_wrong_input_count():
    d = _gain_diagram(2.0)
    f = jaxonomy.submodel_function(d)
    ctx = d.create_context()
    with pytest.raises(TypeError, match="expected 1 input"):
        f(ctx)
    with pytest.raises(TypeError, match="expected 1 input"):
        f(ctx, jnp.array(1.0), jnp.array(2.0))


def test_raises_on_empty_output_ports():
    d = _gain_diagram(2.0)
    with pytest.raises(ValueError, match="no output ports"):
        jaxonomy.submodel_function(d, output_ports=[])
