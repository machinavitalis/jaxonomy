# SPDX-License-Identifier: MIT

"""
Tests for Predictor (Inference) blocks

Contains tests for:
- PyTorch
- TensorFlow
"""

import importlib.util
import os
import sys

import jax.numpy as jnp
import pytest

from jaxonomy.library import PyTorch, TensorFlow

# These suites exercise real Torch / TensorFlow interop; skip cleanly when the
# optional dependency isn't installed rather than failing on ImportError.
_requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch not installed"
)
_requires_tensorflow = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed"
)

# Prevent tests from running indefinitely. It should not happen.
pytestmark = pytest.mark.timeout(20)


@pytest.fixture(scope="class")
def manage_models():
    # Directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute file paths relative to the current script
    filename_torch_model_1 = os.path.join(current_script_dir, "assets", "adder_1.pt")
    filename_torch_model_2 = os.path.join(current_script_dir, "assets", "adder_2.pt")

    filename_tf_model_1 = os.path.join(current_script_dir, "assets", "adder_1.zip")
    filename_tf_model_2 = os.path.join(current_script_dir, "assets", "adder_2.zip")

    resource_names = [
        filename_torch_model_1,
        filename_torch_model_2,
        filename_tf_model_1,
        filename_tf_model_2,
    ]
    return resource_names


@pytest.mark.usefixtures("manage_models")
@_requires_torch
class TestPyTorch:
    @pytest.mark.parametrize(
        "x, y, dtype, expected_result",
        [
            (1, 10, "int32", jnp.array(11, dtype=jnp.int32)),
            (1, 10, "float32", jnp.array(11, dtype=jnp.float32)),
            (1, 10, "float64", jnp.array(11, dtype=jnp.float64)),
            (
                jnp.array([1.0, 2.0], dtype=jnp.int16),
                jnp.array([3.0, 4.0], dtype=jnp.int16),
                "int16",
                jnp.array([4.0, 6.0], dtype=jnp.int16),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float32),
                jnp.array([3.3, 4.4], dtype=jnp.float32),
                "float32",
                jnp.array([4.4, 6.6], dtype=jnp.float32),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float32),
                jnp.array([3.3, 4.4], dtype=jnp.float32),
                "float64",
                jnp.array([4.4, 6.6], dtype=jnp.float64),
            ),
        ],
    )
    def test_torch_model_1_cast(self, manage_models, x, y, dtype, expected_result):
        torch_model_1_filename = manage_models[0]

        predictor = PyTorch(
            torch_model_1_filename,
            num_inputs=2,
            num_outputs=1,
            cast_outputs_to_dtype=dtype,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result = predictor.output_ports[0].eval(context)

        assert jnp.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    @pytest.mark.parametrize(
        "x, y, expected_result",
        [
            (1, 10, jnp.array(11, dtype=jnp.int64)),
            (1.0, 10.0, jnp.array(11, dtype=jnp.float64)),
            (
                jnp.array([1.0, 2.0], dtype=jnp.int16),
                jnp.array([3.0, 4.0], dtype=jnp.int16),
                jnp.array([4.0, 6.0], dtype=jnp.int16),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float32),
                jnp.array([3.3, 4.4], dtype=jnp.float32),
                jnp.array([4.4, 6.6], dtype=jnp.float32),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float64),
                jnp.array([3.3, 4.4], dtype=jnp.float64),
                jnp.array([4.4, 6.6], dtype=jnp.float64),
            ),
        ],
    )
    def test_torch_model_1_no_cast(self, manage_models, x, y, expected_result):
        # NOTE: not sure what would be the right behavior here, even?
        if sys.platform == "win32":
            pytest.xfail(reason="On windows, pytorch defaults to int32")

        torch_model_1_filename = manage_models[0]

        predictor = PyTorch(
            torch_model_1_filename,
            num_inputs=2,
            num_outputs=1,
            cast_outputs_to_dtype=None,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result = predictor.output_ports[0].eval(context)

        assert jnp.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    @pytest.mark.parametrize(
        "x, y, dtype, expected_result",
        [
            (1, 10, "int32", jnp.array(11, dtype=jnp.int32)),
            (1, 10, "float32", jnp.array(11, dtype=jnp.float32)),
            (1, 10, "float64", jnp.array(11, dtype=jnp.float64)),
            (
                jnp.array([1.0, 2.0], dtype=jnp.int16),
                jnp.array([3.0, 4.0], dtype=jnp.int16),
                "int16",
                jnp.array([4.0, 6.0], dtype=jnp.int16),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float32),
                jnp.array([3.3, 4.4], dtype=jnp.float32),
                "float32",
                jnp.array([4.4, 6.6], dtype=jnp.float32),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float32),
                jnp.array([3.3, 4.4], dtype=jnp.float32),
                "float64",
                jnp.array([4.4, 6.6], dtype=jnp.float64),
            ),
        ],
    )
    def test_torch_model_2_cast(self, manage_models, x, y, dtype, expected_result):
        torch_model_2_filename = manage_models[1]

        predictor = PyTorch(
            torch_model_2_filename,
            num_inputs=2,
            num_outputs=2,
            cast_outputs_to_dtype=dtype,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result_0 = predictor.output_ports[0].eval(context)
        result_1 = predictor.output_ports[1].eval(context)

        assert jnp.allclose(result_0, expected_result)
        assert result_0.dtype == expected_result.dtype
        assert jnp.allclose(result_1, x)
        assert result_1.dtype == expected_result.dtype

    @pytest.mark.parametrize(
        "x, y, expected_result",
        [
            (1, 10, jnp.array(11, dtype=jnp.int64)),
            (1.0, 10.0, jnp.array(11, dtype=jnp.float64)),
            (
                jnp.array([1.0, 2.0], dtype=jnp.int16),
                jnp.array([3.0, 4.0], dtype=jnp.int16),
                jnp.array([4.0, 6.0], dtype=jnp.int16),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float32),
                jnp.array([3.3, 4.4], dtype=jnp.float32),
                jnp.array([4.4, 6.6], dtype=jnp.float32),
            ),
            (
                jnp.array([1.1, 2.2], dtype=jnp.float64),
                jnp.array([3.3, 4.4], dtype=jnp.float64),
                jnp.array([4.4, 6.6], dtype=jnp.float64),
            ),
        ],
    )
    def test_torch_model_2_no_cast(self, manage_models, x, y, expected_result):
        # NOTE: not sure what would be the right behavior here, even?
        if sys.platform == "win32":
            pytest.xfail(reason="On windows, pytorch defaults to int32")

        torch_model_2_filename = manage_models[1]

        predictor = PyTorch(
            torch_model_2_filename,
            num_inputs=2,
            num_outputs=2,
            cast_outputs_to_dtype=None,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result_0 = predictor.output_ports[0].eval(context)
        result_1 = predictor.output_ports[1].eval(context)

        assert jnp.allclose(result_0, expected_result)
        assert result_0.dtype == expected_result.dtype
        assert jnp.allclose(result_1, x)
        assert result_1.dtype == expected_result.dtype


@_requires_tensorflow
class TestTensorFlow:
    @pytest.mark.parametrize(
        "x, y, dtype, expected_result",
        [
            (
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(11.0, dtype=jnp.float32),
                "int16",
                jnp.array(12, dtype=jnp.int16),
            ),
            (
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(11.0, dtype=jnp.float32),
                "float64",
                jnp.array(12.0, dtype=jnp.float64),
            ),
        ],
    )
    def test_tf_model_1_cast(self, manage_models, x, y, dtype, expected_result):
        tf_model_1_zip = manage_models[2]

        predictor = TensorFlow(
            tf_model_1_zip,
            cast_outputs_to_dtype=dtype,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result = predictor.output_ports[0].eval(context)

        assert jnp.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    def test_tf_model_1_no_cast(self, manage_models):
        tf_model_1_zip = manage_models[2]

        x = jnp.array(1.0, dtype=jnp.float32)
        y = jnp.array(11.0, dtype=jnp.float32)
        expected_result = jnp.array(12, dtype=jnp.float32)

        predictor = TensorFlow(
            tf_model_1_zip,
            cast_outputs_to_dtype=None,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result = predictor.output_ports[0].eval(context)

        assert jnp.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    def test_tf_wrong_dtype(self, manage_models):
        tf_model_1_zip = manage_models[2]

        x = jnp.array(1.0, dtype=jnp.float64)
        y = jnp.array(11.0, dtype=jnp.float64)

        predictor = TensorFlow(
            tf_model_1_zip,
            cast_outputs_to_dtype=None,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        # should autocast inputs and not raise an exception
        predictor.create_context()

    @pytest.mark.parametrize(
        "x, y, dtype, expected_result",
        [
            (
                jnp.ones(10, dtype=jnp.float64),
                11 * jnp.ones(10, dtype=jnp.float64),
                "int16",
                12 * jnp.ones(10, dtype=jnp.int16),
            ),
            (
                jnp.ones(10, dtype=jnp.float64),
                11 * jnp.ones(10, dtype=jnp.float64),
                "float32",
                12 * jnp.ones(10, dtype=jnp.float32),
            ),
        ],
    )
    def test_tf_model_2_cast(self, manage_models, x, y, dtype, expected_result):
        tf_model_2_zip = manage_models[3]

        predictor = TensorFlow(
            tf_model_2_zip,
            cast_outputs_to_dtype=dtype,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result_0 = predictor.output_ports[0].eval(context)
        result_1 = predictor.output_ports[1].eval(context)

        assert jnp.allclose(result_0, expected_result)
        assert result_0.dtype == expected_result.dtype

        assert jnp.allclose(result_1, x)
        assert result_1.dtype == expected_result.dtype

    def test_tf_model_2_no_cast(self, manage_models):
        tf_model_2_zip = manage_models[3]

        x = 11 * jnp.ones(10, dtype=jnp.float64)
        y = jnp.ones(10, dtype=jnp.float64)
        expected_result = 12 * jnp.ones(10, dtype=jnp.float64)

        predictor = TensorFlow(
            tf_model_2_zip,
            cast_outputs_to_dtype=None,
        )

        predictor.input_ports[0].fix_value(x)
        predictor.input_ports[1].fix_value(y)

        context = predictor.create_context()

        result_0 = predictor.output_ports[0].eval(context)
        result_1 = predictor.output_ports[1].eval(context)

        assert jnp.allclose(result_0, expected_result)
        assert result_0.dtype == expected_result.dtype

        assert jnp.allclose(result_1, x)
        assert result_1.dtype == expected_result.dtype
