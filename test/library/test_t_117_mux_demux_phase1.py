# SPDX-License-Identifier: MIT
"""T-117 phase 1: Mux / Demux signal-routing primitives.

Mux(n) packs n homogeneous inputs into a single output via npa.stack
(adds one leading axis). Demux(n) unpacks a vector input of length n
into n scalar outputs via index-based slicing.

These pin the basic forward/backward pass + Mux↔Demux round-trip
identity. BusCreator/BusSelector (named-field bus signals) are deferred
as T-117-followup-bus-namedtuple.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy import library
from jaxonomy.library import Mux, Demux
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def test_mux_packs_three_scalars_into_vector():
    """Mux(3) of three Constant scalars produces a length-3 vector."""
    a = library.Constant(1.0)
    b = library.Constant(2.0)
    c = library.Constant(3.0)
    mux = Mux(3)

    builder = jaxonomy.DiagramBuilder()
    builder.add(a, b, c, mux)
    builder.connect(a.output_ports[0], mux.input_ports[0])
    builder.connect(b.output_ports[0], mux.input_ports[1])
    builder.connect(c.output_ports[0], mux.input_ports[2])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"mux_out": mux.output_ports[0]},
    )
    out = np.asarray(results.outputs["mux_out"])
    np.testing.assert_array_equal(out[-1], np.array([1.0, 2.0, 3.0]))


def test_mux_packs_two_vectors_into_matrix():
    """Mux(2) of two shape-(2,) vectors → shape (2, 2)."""
    a = library.Constant(np.array([1.0, 2.0]))
    b = library.Constant(np.array([3.0, 4.0]))
    mux = Mux(2)

    builder = jaxonomy.DiagramBuilder()
    builder.add(a, b, mux)
    builder.connect(a.output_ports[0], mux.input_ports[0])
    builder.connect(b.output_ports[0], mux.input_ports[1])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"mux_out": mux.output_ports[0]},
    )
    out = np.asarray(results.outputs["mux_out"])
    np.testing.assert_array_equal(out[-1], np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_demux_splits_vector_into_three_scalars():
    """Demux(3) on Constant([1, 2, 3]) → (1, 2, 3)."""
    src = library.Constant(np.array([1.0, 2.0, 3.0]))
    demux = Demux(3)

    builder = jaxonomy.DiagramBuilder()
    builder.add(src, demux)
    builder.connect(src.output_ports[0], demux.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={
            "out0": demux.output_ports[0],
            "out1": demux.output_ports[1],
            "out2": demux.output_ports[2],
        },
    )
    np.testing.assert_allclose(float(np.asarray(results.outputs["out0"])[-1]), 1.0)
    np.testing.assert_allclose(float(np.asarray(results.outputs["out1"])[-1]), 2.0)
    np.testing.assert_allclose(float(np.asarray(results.outputs["out2"])[-1]), 3.0)


def test_mux_then_demux_roundtrip():
    """Mux ↔ Demux is an identity in a small Diagram."""
    a = library.Constant(1.0)
    b = library.Constant(2.0)
    c = library.Constant(3.0)
    mux = Mux(3, name="mux")
    demux = Demux(3, name="demux")

    builder = jaxonomy.DiagramBuilder()
    builder.add(a, b, c, mux, demux)
    builder.connect(a.output_ports[0], mux.input_ports[0])
    builder.connect(b.output_ports[0], mux.input_ports[1])
    builder.connect(c.output_ports[0], mux.input_ports[2])
    builder.connect(mux.output_ports[0], demux.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={
            "out0": demux.output_ports[0],
            "out1": demux.output_ports[1],
            "out2": demux.output_ports[2],
        },
    )
    np.testing.assert_allclose(float(np.asarray(results.outputs["out0"])[-1]), 1.0)
    np.testing.assert_allclose(float(np.asarray(results.outputs["out1"])[-1]), 2.0)
    np.testing.assert_allclose(float(np.asarray(results.outputs["out2"])[-1]), 3.0)


def test_demux_n_one_is_identity_extraction():
    """Edge case: Demux(1) extracts the single element of a length-1 vector."""
    src = library.Constant(np.array([42.0]))
    demux = Demux(1)

    builder = jaxonomy.DiagramBuilder()
    builder.add(src, demux)
    builder.connect(src.output_ports[0], demux.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": demux.output_ports[0]},
    )
    np.testing.assert_allclose(float(np.asarray(results.outputs["out"])[-1]), 42.0)


def test_mux_underlying_op_is_differentiable():
    """The Mux op (npa.stack-based ReduceBlock) is JAX-differentiable.

    The Mux block is a thin ReduceBlock(npa.stack) wrapper, so end-to-end
    differentiability comes for free as long as the underlying op gradient
    flows. We test the op directly here; the block's wiring through
    simulate is exercised by the other tests in this module.
    """
    def f(a, b, c):
        return jnp.sum(jnp.stack([a, b, c]))

    g_a, g_b, g_c = jax.grad(f, argnums=(0, 1, 2))(1.0, 2.0, 3.0)
    np.testing.assert_allclose(float(g_a), 1.0)
    np.testing.assert_allclose(float(g_b), 1.0)
    np.testing.assert_allclose(float(g_c), 1.0)


def test_demux_underlying_op_is_differentiable():
    """∂(sum(input[i] for i in range(n))) / ∂input = 1s.

    Demux uses index-based slicing per output port. Each output picks one
    slot of the input vector, so the gradient of the sum is all-ones.
    """
    def f(x):
        return sum(x[i] for i in range(3))

    g = jax.grad(f)(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(g), np.array([1.0, 1.0, 1.0]))
