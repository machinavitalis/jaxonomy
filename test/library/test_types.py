# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""Test for types

Contains tests for:
- SignalDatatypeConversion
"""

import numpy as np
import jax.numpy as jnp
import pytest

from collimator.library import SignalDatatypeConversion


@pytest.fixture(scope="class")
def eval_conversion():
    def convert(x, dtype_str):
        converter = SignalDatatypeConversion(dtype_str)
        converter.input_ports[0].fix_value(x)
        context = converter.create_context()
        output = converter.output_ports[0].eval(context)
        return output

    return convert


@pytest.mark.usefixtures("eval_conversion")
class TestEvalConversion:
    @pytest.mark.parametrize(
        "input_val, dtype_str, expected_output_type",
        [
            (3.14, "float32", np.float32),
            (1, "int64", np.int64),
            (np.array([1, 2, 3]), "float32", np.float32),
            (jnp.array([1, 2, 3]), "float64", np.float64),
            (np.array([1.0, 2.0, 3.0]), "int32", np.int32),
            (jnp.array([1.0, 2.0, 3.0]), "int32", np.int32),
            (np.array([1, 0, 1]), "bool", np.bool_),
            (jnp.array([1, 0, 1]), "bool", np.bool_),
            (jnp.array([1.0, 0.0, 2.0]), "bool", np.bool_),
            (jnp.array([True, False, False]), "bool", np.bool_),
        ],
    )
    def test_dtype_conversion(
        self, eval_conversion, input_val, dtype_str, expected_output_type
    ):
        result = eval_conversion(input_val, dtype_str)
        assert isinstance(result, np.ndarray) or isinstance(
            result, jnp.ndarray
        ), "Result should be a NumPy or JAX array"
        assert (
            result.dtype == expected_output_type
        ), f"Expected dtype {expected_output_type}, got {result.dtype}"

    @pytest.mark.parametrize(
        "input_val, dtype_str, expected_value",
        [
            (3, "float32", 3.0),
            (3, "int32", 3),
            (3.4, "int16", 3.0),
            (jnp.array([1, 2, 3]), "int64", jnp.array([1, 2, 3], dtype=np.int64)),
            (np.array([1, 2, 3]), "int32", np.array([1, 2, 3], dtype=np.int32)),
            (jnp.array([1.1, 2.2, 3.3]), "int16", jnp.array([1, 2, 3], dtype=np.int16)),
            (
                jnp.array([1.1, 2.2, 3.3]),
                "float32",
                jnp.array([1.1, 2.2, 3.3], dtype=np.float32),
            ),
            (
                jnp.array([1.1, 2.2, 3.3], dtype=np.float64),
                "float32",
                jnp.array([1.1, 2.2, 3.3], dtype=np.float32),
            ),
            (
                jnp.array([1.0, 0.0, 2], dtype=jnp.float16),
                "bool",
                jnp.array([True, False, True], dtype=np.bool_),
            ),
            (
                jnp.array([True, False, False]),
                "bool",
                jnp.array([True, False, False], dtype=np.bool_),
            ),
            (
                jnp.array([True, False, False]),
                "float64",
                jnp.array([1.0, 0.0, 0], np.float64),
            ),
        ],
    )
    def test_value_preservation(
        self, eval_conversion, input_val, dtype_str, expected_value
    ):
        result = eval_conversion(input_val, dtype_str)
        np.testing.assert_array_equal(
            result,
            expected_value,
            err_msg="Converted values do not match expected values",
        )

    @pytest.mark.parametrize(
        "input_val, dtype_str",
        [
            ("a_string", "float64"),
            ([True, False, True], "bool"),  # JAX can't determine dtype of lists
            ([1.0, 2.0, 3.0], "float32"),  # JAX can't determine dtype of lists
        ],
    )
    def test_invalid_dtype_conversion(self, eval_conversion, input_val, dtype_str):
        with pytest.raises((TypeError, ValueError)):
            eval_conversion(input_val, dtype_str)
