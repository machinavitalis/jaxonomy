# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from ..backend import numpy_api as cnp
from ..framework import LeafSystem
from .primitives import FeedthroughBlock

__all__ = ["ode_block", "feedthrough_block"]


def ode_block(state_dim, dtype=cnp.float64, num_inputs=0, name=None):
    template_vector = cnp.zeros(state_dim, dtype=dtype)

    def _wrapper(func):
        block_name = name if name is not None else func.__name__
        block = LeafSystem(name=block_name)

        for i in range(num_inputs):
            block.declare_input_port(name=f"{block.name}:input[{i}]")

        block.declare_continuous_state(default_value=template_vector, ode=func)
        block.declare_continuous_state_output(
            name=f"{block.name}:output"
        )  # One vector-valued output
        return block

    return _wrapper


def feedthrough_block(func):
    return FeedthroughBlock(func, name=func.__name__)
