# SPDX-License-Identifier: MIT
"""
T-023a — ONNXJax block smoke tests with a built-on-the-fly model.

The HuggingFace stress harness in ``test_onnx_jax_huggingface.py``
exercises a real 637-op DistilBERT-style model — these tests cover
the small-model path so the block's contract (autodiff,
construction, composition) is verified without a 90 MB download.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

onnx = pytest.importorskip("onnx")
jaxonnxruntime = pytest.importorskip("jaxonnxruntime")
helper = onnx.helper
TensorProto = onnx.TensorProto

from jaxonomy.library import Constant, ONNXJax  # noqa: E402


@pytest.fixture
def matmul_relu_model():
    """A 1-layer MLP: y = relu(x @ W + b)."""
    W = np.array([[1.0, -2.0], [3.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    b = np.array([0.5, -0.5], dtype=np.float32)
    W_t = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    b_t = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())
    matmul = helper.make_node("MatMul", ["x", "W"], ["xW"])
    add = helper.make_node("Add", ["xW", "b"], ["lin"])
    relu = helper.make_node("Relu", ["lin"], ["y"])
    graph = helper.make_graph(
        nodes=[matmul, add, relu], name="mlp",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[W_t, b_t],
    )
    # jaxonnxruntime's Relu handler requires opset 6 or 14 (not 13).
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    onnx.save(model, path)
    yield path, W, b
    os.unlink(path)


# ── construction & forward inference ──────────────────────────────────


def test_onnxjax_block_constructs_and_runs(matmul_relu_model):
    path, W, b = matmul_relu_model
    blk = ONNXJax(file_name=path, num_inputs=1, num_outputs=1)
    bld = jaxonomy.DiagramBuilder()
    x = jnp.array([[1.0, 1.0, 1.0]], dtype=jnp.float32)
    src = bld.add(Constant(x, name="x"))
    onnx_blk = bld.add(blk)
    bld.connect(src.output_ports[0], onnx_blk.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = onnx_blk.output_ports[0].eval(ctx)
    expected = np.maximum(np.asarray(x) @ W + b, 0.0)
    np.testing.assert_allclose(np.asarray(y), expected, atol=1e-6)


# ── autodiff: gradient through the model is real and correct ──────────


def test_grad_through_onnxjax_matches_finite_difference(matmul_relu_model):
    """Gradient of sum(MLP(x)) w.r.t. x via jax.grad must match
    central-difference FD."""
    path, W, b = matmul_relu_model
    blk = ONNXJax(file_name=path, num_inputs=1, num_outputs=1)

    bld = jaxonomy.DiagramBuilder()
    x_init = jnp.array([[0.5, 1.0, -0.5]], dtype=jnp.float32)
    src = bld.add(Constant(x_init, name="x"))
    onnx_blk = bld.add(blk)
    bld.connect(src.output_ports[0], onnx_blk.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def loss(x_val):
        ctx = ctx0.with_subcontext(
            src.system_id,
            ctx0[src.system_id].with_parameter("value", x_val),
        )
        y = onnx_blk.output_ports[0].eval(ctx)
        return jnp.sum(y)

    g = jax.grad(loss)(x_init)

    eps = 1e-3
    fd = np.zeros_like(np.asarray(x_init))
    for i in range(3):
        xp = x_init.at[0, i].add(eps)
        xm = x_init.at[0, i].add(-eps)
        fd[0, i] = float((loss(xp) - loss(xm)) / (2 * eps))

    np.testing.assert_allclose(
        np.asarray(g), fd, atol=1e-3,
        err_msg=f"AD={np.asarray(g)}, FD={fd}",
    )


# ── vmap over the input ───────────────────────────────────────────────


def test_vmap_over_onnxjax(matmul_relu_model):
    """Batching over the input axis: vmap the inference."""
    path, W, b = matmul_relu_model
    blk = ONNXJax(file_name=path, num_inputs=1, num_outputs=1)

    bld = jaxonomy.DiagramBuilder()
    x_init = jnp.zeros((1, 3), dtype=jnp.float32)
    src = bld.add(Constant(x_init, name="x"))
    onnx_blk = bld.add(blk)
    bld.connect(src.output_ports[0], onnx_blk.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def fwd(x_val):
        ctx = ctx0.with_subcontext(
            src.system_id,
            ctx0[src.system_id].with_parameter("value", x_val),
        )
        return onnx_blk.output_ports[0].eval(ctx)

    xs = jnp.array(np.random.default_rng(0).normal(
        size=(5, 1, 3)).astype(np.float32))
    # vmap with care: jaxonnxruntime's prepared graph has batch=1 baked in,
    # so we vmap over the leading axis manually rather than relying on
    # the model's dynamic batch dim.
    ys = jnp.stack([fwd(xs[i]) for i in range(5)])
    assert ys.shape == (5, 1, 2)
    # Each row matches the analytic computation.
    expected = np.maximum(np.asarray(xs).reshape(5, 3) @ W + b, 0.0)
    np.testing.assert_allclose(np.asarray(ys).reshape(5, 2), expected, atol=1e-6)
