# SPDX-License-Identifier: MIT
"""
T-023 — ONNX block tests.

Builds a tiny ONNX model on the fly (a 3×3 matmul) and verifies the
block loads it, runs inference via onnxruntime, and integrates with a
DiagramBuilder.  Skips if onnxruntime / onnx are not installed.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
helper = onnx.helper
TensorProto = onnx.TensorProto

from jaxonomy.library import Constant, ONNX  # noqa: E402


@pytest.fixture
def matmul_model():
    """Build an ONNX MatMul model: y[1x3] = x[1x3] @ W[3x3]."""
    W = np.array(
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    W_tensor = helper.make_tensor(
        name="W", data_type=TensorProto.FLOAT,
        dims=W.shape, vals=W.flatten().tolist(),
    )
    node = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"])
    graph = helper.make_graph(
        nodes=[node],
        name="tiny_matmul",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[W_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    onnx.save(model, path)
    yield path, W
    os.unlink(path)


# ── construction & inference ──────────────────────────────────────────────


def test_loads_and_runs_matmul(matmul_model):
    path, W = matmul_model
    blk = ONNX(file_name=path, num_inputs=1, num_outputs=1)
    bld = jaxonomy.DiagramBuilder()
    x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    src = bld.add(Constant(x, name="x"))
    onnx_blk = bld.add(blk)
    bld.connect(src.output_ports[0], onnx_blk.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = onnx_blk.output_ports[0].eval(ctx)
    expected = np.asarray(x) @ W
    np.testing.assert_allclose(np.asarray(y), expected, atol=1e-6)


def test_input_count_mismatch_raises(matmul_model):
    """Declaring num_inputs=2 for a 1-input model should fail at
    create_context with a clear error."""
    path, _ = matmul_model
    blk = ONNX(file_name=path, num_inputs=2, num_outputs=1)
    bld = jaxonomy.DiagramBuilder()
    x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    bld.add(Constant(x, name="x0"))
    bld.add(Constant(x, name="x1"))
    bld.add(blk)
    diagram = bld.build()
    with pytest.raises(Exception, match="num_inputs|num_outputs|inputs"):
        diagram.create_context()


def test_output_count_mismatch_raises(matmul_model):
    path, _ = matmul_model
    blk = ONNX(file_name=path, num_inputs=1, num_outputs=2)
    bld = jaxonomy.DiagramBuilder()
    x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    src = bld.add(Constant(x, name="x"))
    onnx_blk = bld.add(blk)
    bld.connect(src.output_ports[0], onnx_blk.input_ports[0])
    diagram = bld.build()
    with pytest.raises(Exception, match="num_inputs|num_outputs|outputs"):
        diagram.create_context()


def test_cast_outputs_to_dtype(matmul_model):
    path, W = matmul_model
    blk = ONNX(
        file_name=path, num_inputs=1, num_outputs=1,
        cast_outputs_to_dtype="float64",
    )
    bld = jaxonomy.DiagramBuilder()
    x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    src = bld.add(Constant(x, name="x"))
    onnx_blk = bld.add(blk)
    bld.connect(src.output_ports[0], onnx_blk.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = onnx_blk.output_ports[0].eval(ctx)
    assert y.dtype == jnp.float64
