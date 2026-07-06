# SPDX-License-Identifier: MIT
"""
T-010 — ReplicatedFunction container block tests.

Covers:

  - Broadcast input (in_axes=None): N replicas each see the same scalar,
    all produce the same output.
  - Batched input (in_axes=0): N replicas each see a different row.
  - Per-instance "parameters" via batched input axis.
  - Two-input combination: one broadcast + one batched.
  - Gradient flows through vmap correctly.
  - jit compiles the block.
  - Validation: bad n, bad in_axes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Constant, ReplicatedFunction
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Broadcast input ────────────────────────────────────────────────────────


def test_broadcast_input():
    """in_axes=(None,) broadcasts a scalar input to 8 replicas."""
    rep = ReplicatedFunction(
        submodel=lambda u: 3.0 * u,
        n=8,
        n_inputs=1,
        in_axes=(None,),
    )
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(2.0), name="u"))
    r = bld.add(rep)
    bld.connect(src.output_ports[0], r.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = r.output_ports[0].eval(ctx)
    assert y.shape == (8,)
    np.testing.assert_allclose(np.asarray(y), 6.0 * np.ones(8))


# ── Batched input ──────────────────────────────────────────────────────────


def test_batched_input():
    """in_axes=(0,) processes each row of a batched input separately."""
    rep = ReplicatedFunction(
        submodel=lambda u: 2.0 * u + 1.0,
        n=4,
        n_inputs=1,
        in_axes=(0,),
    )
    bld = jaxonomy.DiagramBuilder()
    us = jnp.array([1.0, 2.0, 3.0, 4.0])
    src = bld.add(Constant(us, name="u"))
    r = bld.add(rep)
    bld.connect(src.output_ports[0], r.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = r.output_ports[0].eval(ctx)
    assert y.shape == (4,)
    np.testing.assert_allclose(np.asarray(y), 2.0 * np.asarray(us) + 1.0)


# ── Broadcast + batched combination ────────────────────────────────────────


def test_mixed_axes():
    """One input broadcast, one batched — classic per-instance parameter."""
    # submodel: f(shared, per_instance) = shared * per_instance
    rep = ReplicatedFunction(
        submodel=lambda shared, pi: shared * pi,
        n=3,
        n_inputs=2,
        in_axes=(None, 0),
    )
    bld = jaxonomy.DiagramBuilder()
    shared = bld.add(Constant(jnp.array(5.0), name="shared"))
    per_inst = bld.add(Constant(jnp.array([1.0, 2.0, 3.0]), name="pi"))
    r = bld.add(rep)
    bld.connect(shared.output_ports[0], r.input_ports[0])
    bld.connect(per_inst.output_ports[0], r.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = r.output_ports[0].eval(ctx)
    np.testing.assert_allclose(np.asarray(y), np.array([5.0, 10.0, 15.0]))


# ── Gradient flows through vmap ────────────────────────────────────────────


def test_grad_through_vmap():
    """Gradient of sum(y) where y = 2·u over all N replicas = 2 per replica."""
    rep = ReplicatedFunction(
        submodel=lambda u: 2.0 * u,
        n=5,
        n_inputs=1,
        in_axes=(0,),
    )
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.arange(5, dtype=jnp.float64), name="u"))
    r = bld.add(rep)
    bld.connect(src.output_ports[0], r.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def loss(us):
        ctx = ctx0.with_subcontext(
            src.system_id, ctx0[src.system_id].with_parameter("value", us),
        )
        y = r.output_ports[0].eval(ctx)
        return jnp.sum(y)

    g = jax.grad(loss)(jnp.arange(5, dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(g), 2.0 * np.ones(5))


# ── jit wraps ──────────────────────────────────────────────────────────────


def test_jit_compatibility():
    rep = ReplicatedFunction(
        submodel=lambda u: u**2,
        n=6,
        n_inputs=1,
        in_axes=(0,),
    )
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.arange(6, dtype=jnp.float64), name="u"))
    r = bld.add(rep)
    bld.connect(src.output_ports[0], r.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()

    @jax.jit
    def run(ctx):
        return r.output_ports[0].eval(ctx)

    y = run(ctx)
    np.testing.assert_allclose(np.asarray(y), np.arange(6) ** 2)


# ── Validation ────────────────────────────────────────────────────────────


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="n must be"):
        ReplicatedFunction(submodel=lambda u: u, n=0, n_inputs=1)


def test_n_inputs_must_match_in_axes():
    with pytest.raises(ValueError, match="len\\(in_axes\\)"):
        ReplicatedFunction(
            submodel=lambda a, b: a + b, n=3, n_inputs=2, in_axes=(0,),
        )


def test_invalid_in_axes_value():
    with pytest.raises(ValueError, match="in_axes entries must be 0 or None"):
        ReplicatedFunction(
            submodel=lambda u: u, n=3, n_inputs=1, in_axes=(1,),
        )
