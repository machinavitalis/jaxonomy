# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from . import numpy_api as cnp
from .typing import (
    Array,
    DTypeLike,
    ShapeLike,
)


def make_array(
    default_value: Array, dtype: DTypeLike = None, shape: ShapeLike = None
) -> Array:
    assert not (
        shape is None and default_value is None
    ), "Must provide either shape or default_value"

    if default_value is not None:
        default_value = cnp.array(default_value, dtype=dtype)
    else:
        default_value = cnp.zeros(shape, dtype=dtype)

    # JAX doesn't support non-numeric arrays, so for consistency we will
    # ensure that no backend does.  This will mimic the error that JAX raises
    # when trying to convert a non-numeric value to a JAX array.
    if not cnp.issubdtype(default_value.dtype, cnp.number) and not (
        default_value.dtype == bool
    ):
        msg = (
            f"Parameter values must be numeric.  Got: {default_value} with "
            f"dtype {default_value.dtype}"
        )
        raise TypeError(msg)

    return default_value
