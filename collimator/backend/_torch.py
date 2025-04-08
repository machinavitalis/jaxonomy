# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from ..lazy_loader import LazyLoader

torch = LazyLoader("torch", globals(), "torch")

__all__ = ["torch_functions", "torch_constants"]


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def zeros_like(x):
    return torch.zeros(*x.shape, dtype=x.dtype)


def torch_functions():
    return (
        {
            "asarray": torch.as_tensor,
            "array": torch.tensor,
            "zeros_like": zeros_like,
            "cond": cond,
        }
        if torch is not None
        else {}
    )


torch_constants = (
    {
        "lib": torch,
    }
    if torch is not None
    else {}
)
