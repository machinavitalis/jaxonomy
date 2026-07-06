# SPDX-License-Identifier: MIT

"""Arithmetic, matrix, and scalar math primitive blocks."""

from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING, NamedTuple
from functools import partial, wraps
from collections import namedtuple
from enum import IntEnum

import numpy as np

from ..logging import logger
from ..framework.error import BlockParameterError, ErrorCollector
from ..framework.event import LeafEventCollection, ZeroCrossingEvent
from ..framework.system_base import UpstreamEvalError
from ..framework import (
    LeafSystem,
    ShapeMismatchError,
    DtypeMismatchError,
    DependencyTicket,
    Parameter,
    parameters,
)
from ..backend import cond, numpy_api as npa
from ..lazy_loader import LazyLoader
from .generic import SourceBlock, FeedthroughBlock, ReduceBlock
from .linear_system import derivative_filter

if TYPE_CHECKING:
    import equinox as eqx
    from jax import lax as jax_lax
    from ..framework.port import OutputPort
    from ..backend.typing import Array
else:
    eqx = LazyLoader("eqx", globals(), "equinox")
    jax_lax = LazyLoader("jax_lax", globals(), "jax.lax")


from ._primitives_common import _stop_gradient, check_state_type, is_discontinuity


__all__ = [
    "Abs",
    "Adder",
    "Arithmetic",
    "CrossProduct",
    "DotProduct",
    "Exponent",
    "Gain",
    "Logarithm",
    "MatrixConcatenation",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "MinMax",
    "Offset",
    "Power",
    "Product",
    "ProductOfElements",
    "Reciprocal",
    "ScalarBroadcast",
    "SquareRoot",
    "Stack",
    "SumOfElements",
    "Trigonometric",
]



class Abs(FeedthroughBlock):
    """Output the absolute value of the input signal.

    Input ports:
        None

    Output ports:
        (0) The absolute value of the input signal.

    Events:
        An event is triggered when the output changes from positive to negative
        or vice versa.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(npa.abs, *args, **kwargs)

    def _zero_crossing(self, _time, _state, u):
        return u

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output is
        # fed to an ODE.
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._zero_crossing, direction="crosses_zero")

        return super().initialize_static_data(context)


class Adder(ReduceBlock):
    """Computes the sum/difference of the input.

    The add/subtract operation can be switched by setting the `operators` parameter.
    For example, a 3-input block specified as `Adder(3, operators="+-+")` would add
    the first and third inputs and subtract the second input.

    Input ports:
        (0..n_in-1) The input signals to add/subtract.

    Output ports:
        (0) The sum/difference of the input signals.
    """

    @parameters(static=["operators"])
    def __init__(self, n_in, *args, operators=None, dtype=None, **kwargs):
        # T-038a-followup-other-blocks: per-block dtype override; stored
        # outside the @parameters list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.  Lazy
        # import avoids a circular dep — ``precision.py`` does not
        # import block code.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(n_in, None, *args, **kwargs)

    def initialize(self, operators):
        if operators is not None and any(char not in {"+", "-"} for char in operators):
            raise BlockParameterError(
                message=f"Adder block {self.name} has invalid operators {operators}. Can only contain '+' and '-'",
                system=self,
                parameter_name="operators",
            )

        if operators is None:
            _func = sum
        else:
            signs = [1 if op == "+" else -1 for op in operators]

            def _func(inputs):
                signed_inputs = [s * u for (s, u) in zip(signs, inputs)]
                return sum(signed_inputs)

        if self._dtype is not None:
            # T-038a-followup-other-blocks: wrap the reducer so the final
            # sum is cast to the per-block dtype.
            _inner = _func
            _dtype = self._dtype

            def _func(inputs):
                return npa.asarray(_inner(inputs)).astype(_dtype)

        self.replace_op(_func)


class Arithmetic(ReduceBlock):
    """Performs addition, subtraction, multiplication, and division on the input.

    The arithmetic operation is determined by setting the `operators` parameter.
    For example, a 4-input block specified as `Arithmetic(4, operators="+-*/")` would:
        - Add the first input,
        - Subtract the second input,
        - Multiply the third input,
        - Divide by the fourth input.

    Input ports:
        (0..n_in-1) The input signals for the specified arithmetic operations.

    Output ports:
        (0) The result of the specified arithmetic operations on the input signals.

    """

    @parameters(static=["operators"])
    def __init__(self, n_in, *args, operators=None, **kwargs):
        super().__init__(n_in, None, *args, **kwargs)

    def initialize(self, operators):
        if operators is not None and any(
            char not in {"+", "-", "*", "/"} for char in operators
        ):
            raise BlockParameterError(
                message=f"Arithmetic block {self.name} has invalid operators {operators}. Can only contain '+', '-', '*', '/'.",
                system=self,
                parameter_name="operators",
            )

        ops = {
            "+": npa.add,
            "-": npa.subtract,
            "/": npa.divide,
            "*": npa.multiply,
        }

        def evaluate_expression(operands, operators):
            operands = operands[:]
            operators = operators[:]

            # Handle multiplication and division
            while "*" in operators or "/" in operators:
                for op in ("*", "/"):
                    if op in operators:
                        index = operators.index(op)
                        result = ops[op](operands[index], operands[index + 1])
                        operands = operands[:index] + [result] + operands[index + 2 :]
                        operators = operators[:index] + operators[index + 1 :]

            # Handle addition and subtraction
            while "+" in operators or "-" in operators:
                for op in ("-", "+"):
                    if op in operators:
                        index = operators.index(op)
                        result = ops[op](operands[index], operands[index + 1])
                        operands = operands[:index] + [result] + operands[index + 2 :]
                        operators = operators[:index] + operators[index + 1 :]

            return operands[0]

        def _func(inputs):
            inputs = list(inputs)
            if operators[0] == "/":
                inputs[0] = 1.0 / inputs[0]
            if operators[0] == "-":
                inputs[0] = -inputs[0]
            ops = operators[1:]
            return evaluate_expression(inputs, ops)

        self.replace_op(_func)



class CrossProduct(ReduceBlock):
    """Compute the cross product between the inputs.

    See NumPy docs for details:
    https://numpy.org/doc/stable/reference/generated/numpy.cross.html

    Input ports:
        (0) The first input vector.
        (1) The second input vector.

    Output ports:
        (0) The cross product of the inputs.
    """

    def __init__(self, *args, **kwargs):
        def _cross(inputs):
            return npa.cross(*inputs)

        super().__init__(2, _cross, *args, **kwargs)



class DotProduct(ReduceBlock):
    """Compute the dot product between the inputs.

    This block dispatches to `jax.numpy.dot`, so the semantics, broadcasting rules,
    etc. are the same.  See the JAX docs for details:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html

    Input ports:
        (0) The first input vector.
        (1) The second input vector.

    Output ports:
        (0) The dot product of the inputs.
    """

    def __init__(self, **kwargs):
        super().__init__(2, self._compute_output, **kwargs)

    def _compute_output(self, inputs):
        return npa.dot(inputs[0], inputs[1])



class Exponent(FeedthroughBlock):
    """Compute the exponential of the input signal.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The exponential of the input signal.

    Parameters:
        base:
            One of "exp" or "2". Determines the base of the exponential function.
    """

    @parameters(static=["base"])
    def __init__(self, base, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, base):
        func_lookup = {"exp": npa.exp, "2": npa.exp2}
        if base not in func_lookup:
            raise BlockParameterError(
                message=f"Exponent block {self.name} has invalid selection {base} for 'base'. Valid selections: "
                + ", ".join([k for k in func_lookup.keys()]),
                parameter_name="base",
            )
        self.replace_op(func_lookup[base])



class Gain(FeedthroughBlock):
    """Multiply the input signal by a constant value.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal multiplied by the gain: `y = gain * u`.

    Parameters:
        gain:
            The value to scale the input signal by.
        dtype (optional, T-038a-followup-other-blocks):
            If set, the block's output is cast to this dtype.  See
            ``LookupTable1d`` for the per-block dtype contract.
    """

    @parameters(dynamic=["gain"])
    def __init__(self, gain, *args, dtype=None, **kwargs):
        # T-038a-followup-other-blocks: dtype is stored outside the
        # @parameters dynamic list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        if dtype is None:
            super().__init__(lambda x, gain: gain * x, *args, **kwargs)
        else:
            _dtype = dtype

            def _gain_op(x, gain):
                return npa.asarray(gain * x).astype(_dtype)

            super().__init__(_gain_op, *args, **kwargs)

    def initialize(self, gain):
        pass



class Logarithm(FeedthroughBlock):
    """Compute the logarithm of the input signal.

    This block dispatches to `jax.numpy.log`, `jax.numpy.log2`, or `jax.numpy.log10`,
    so the semantics, broadcasting rules, etc. are the same.  See the JAX docs for
    details:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log.html
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log2.html
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log10.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The logarithm of the input signal.

    Parameters:
        base:
            One of "natural", "2", or "10". Determines the base of the logarithm.
            The default is "natural".
    """

    @parameters(static=["base"])
    def __init__(self, base="natural", **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, base="natural"):
        func_lookup = {
            "10": npa.log10,
            "2": npa.log2,
            "natural": npa.log,
        }
        if base not in func_lookup:
            # cannot pass system=self because this error must be raised BEFORE calling super.__init__()
            # in the case of inheritting from FeedthroughBlock.
            # if we call super.__init__() first, we get missing key error for func_lookup[base].
            raise BlockParameterError(
                message=f"Logarithm block {self.name} has invalid selection {base} for 'base'. Valid selections: "
                + ", ".join([k for k in func_lookup.keys()]),
                parameter_name="base",
            )
        self.replace_op(func_lookup[base])



class MatrixConcatenation(ReduceBlock):
    """Concatenate two matrices along a given axis.

    Dispatches to `jax.numpy.concatenate`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.concatenate.html

    Args:
        axis: The axis along which the matrices are concatenated. 0 for vertical
            and 1 for horizontal. Default is 0.

    Input ports:
        (0, 1) The input matrices `A` and `B`

    Output ports:
        (0) The concatenation input matrices: e.g. `[A,B]`.
    """

    @parameters(static=["axis"])
    def __init__(self, n_in=2, axis=0, **kwargs):
        if n_in != 2:
            raise ValueError(
                "MatrixConcatenation block only supports two input matrices."
            )
        super().__init__(2, None, **kwargs)

    def initialize(self, axis):
        def _func(inputs):
            return npa.concatenate((inputs[0], inputs[1]), axis=int(axis))

        self.replace_op(_func)


class MatrixInversion(FeedthroughBlock):
    """Compute the matrix inverse of the input signal.

    Dispatches to `jax.numpy.inv`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.inv.html

    Input ports:
        (0) The input matrix.

    Output ports:
        (0) The inverse of the input matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(npa.linalg.inv, *args, **kwargs)


class MatrixMultiplication(ReduceBlock):
    """Compute the matrix product of the input signals.

    Dispatches to `jax.numpy.matmul`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html

    Input ports:
        (0, 1) The input matrices `A` and `B`

    Output ports:
        (0) The matrix product of the input matrices: `A @ B`.
    """

    def __init__(
        self,
        n_in=2,
        **kwargs,
    ):
        if n_in != 2:
            raise ValueError(
                "MatrixMultiplication block only supports two input signals."
            )

        def _func(inputs):
            return npa.matmul(inputs[0], inputs[1])

        super().__init__(2, _func, **kwargs)


class MatrixTransposition(FeedthroughBlock):
    """Compute the matrix transpose of the input signal.

    Dispatches to `jax.numpy.transpose`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html

    Input ports:
        (0) The input matrix.

    Output ports:
        (0) The transpose of the input matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(npa.transpose, *args, **kwargs)


class MinMax(ReduceBlock):
    """Return the extremum of the input signals.

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The minimum or maximum of the input signals.

    Parameters:
        operator:
            One of "min" or "max". Determines whether the block returns the minimum
            or maximum of the input signals.

    Events:
        An event is triggered when the extreme input signal changes.  For example,
        if the block is configured as a "max" block with two inputs and the second
        signal becomes greater than the first, a zero-crossing event will be triggered.
    """

    @parameters(static=["operator"])
    def __init__(self, n_in, operator, **kwargs):
        super().__init__(n_in, None, **kwargs)

    def initialize(self, operator):
        func_lookup = {
            "max": self._max,
            "min": self._min,
        }
        if operator not in func_lookup:
            # cannot pass system=self because this error must be raised BEFORE calling super.__init__()
            # in the case of inheritting from FeedthroughBlock.
            # if we call super.__init__() first, we get missing key error for func_lookup[base].
            raise BlockParameterError(
                message=f"MinMax block {self.name} has invalid selection {operator} for 'operator'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()]),
                parameter_name="operator",
            )

        self.operator = operator

        self.replace_op(func_lookup[operator])

        guard_lookup = {
            "max": self._max_guard,
            "min": self._min_guard,
        }

        self._guard = guard_lookup[operator]

    def _min(self, inputs):
        return npa.min(npa.array(inputs))

    def _max(self, inputs):
        return npa.max(npa.array(inputs))

    def _min_guard(self, _time, _state, *inputs, **_params):
        return npa.argmin(npa.array(inputs)).astype(float)

    def _max_guard(self, _time, _state, *inputs, **_params):
        return npa.argmax(npa.array(inputs)).astype(float)

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and (self.output_ports[0]):
            self.declare_zero_crossing(self._guard, direction="edge_detection")

        return super().initialize_static_data(context)



class Offset(FeedthroughBlock):
    """Add a constant offset or bias to the input signal.

    Given an input signal `u` and offset value `b`, this will return `y = u + b`.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal plus the offset.

    Parameters:
        offset:
            The constant offset to add to the input signal.
    """

    @parameters(dynamic=["offset"])
    def __init__(self, offset, *args, **kwargs):
        super().__init__(lambda x, offset: x + offset, *args, **kwargs)

    def initialize(self, offset):
        pass



class Power(FeedthroughBlock):
    """Raise the input signal to a constant power.

    Dispatches to `jax.numpy.power`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.power.html

    For input signal `u` with exponent `p`, the output will be `y = u ** p`.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal raised to the power of the exponent.

    Parameters:
        exponent:
            The exponent to which the input signal is raised.
    """

    @parameters(static=["exponent"])
    def __init__(self, exponent, **kwargs):
        super().__init__(self._func, **kwargs)

        # Note that the exponent here is declared as a configuration
        # parameter and not a context parameter, making it non-differentiable.
        # This is because the derivative rule for the exponent includes a log
        # of the primal input signal, which can cause NaN values during backprop
        # if the input signal is non-positive. Specifically, for `y = u ** p`, the
        # linearization with respect to `p` is `dy = y * log(u) * dp`. If we
        # eventually want to support backprop through this block, we will need
        # to handle the log of the input signal in a way that avoids NaN values.
        # (e.g. with gradient clipping). Tracked in WC-306
        self.exponent = exponent

    def initialize(self, exponent):
        self.exponent = exponent

    def _func(self, *inputs, **parameters):
        (u,) = inputs
        return u**self.exponent


class Product(ReduceBlock):
    """Compute the product and/or quotient of the input signals.

    The block will multiply or divide the input signals, depending on the specified
    operators.  For example, if the block has three inputs `u1`, `u2`, and `u3` and
    is configured with operators="**/", then the output signal will be
    `y = u1 * u2 / u3`.  By default, the block will multiply all of the input signals.

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The product and/or quotient of the input signals.

    Parameters:
        n_in:
            The number of input ports.
        operators:
            A string of length `n_in` specifying the operators to apply to each of
            the input signals.  Each character in the string must be either "*" or "/".
            The default is "*".
        denominator_limit:
            Currently unsupported
        divide_by_zero_behavior:
            Currently unsupported
    """

    @parameters(static=["operators", "denominator_limit", "divide_by_zero_behavior"])
    def __init__(
        self,
        n_in,
        operators=None,  # Expect "**/*", etc
        denominator_limit=None,
        divide_by_zero_behavior=None,
        **kwargs,
    ):
        super().__init__(n_in, None, **kwargs)

    def initialize(
        self,
        operators=None,  # Expect "**/*", etc
        denominator_limit=None,
        divide_by_zero_behavior=None,
    ):
        if operators is not None and any(char not in {"*", "/"} for char in operators):
            raise BlockParameterError(
                message=f"Product block {self.name} has invalid operators {operators}. Can only contain '*' and '/'",
                system=self,
                parameter_name="operators",
            )

        if operators is not None and "/" in operators:
            num_indices = npa.array(
                [idx for idx, op in enumerate(operators) if op == "*"]
            )
            den_indices = npa.array(
                [idx for idx, op in enumerate(operators) if op == "/"]
            )

            def _func(inputs):
                ain = npa.array(inputs)
                num = npa.take(ain, num_indices, axis=0)
                den = npa.take(ain, den_indices, axis=0)
                return npa.prod(num, axis=0) / npa.prod(den, axis=0)

        else:

            def _func(inputs):
                return npa.prod(npa.array(inputs), axis=0)

        self.replace_op(_func)


class ProductOfElements(FeedthroughBlock):
    """Compute the product of the elements of the input signal.

    Dispatches to `jax.numpy.prod`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.prod.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The product of the elements of the input signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(npa.prod, *args, **kwargs)



class Reciprocal(FeedthroughBlock):
    """Compute the reciprocal of the input signal.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The reciprocal of the input signal: `y = 1 / u`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: 1 / x, *args, **kwargs)



class ScalarBroadcast(FeedthroughBlock):
    """Broadcast a scalar to a vector or matrix.

    Given a scalar input `u` and dimensions `m` and `n`, this block will return
    a vector or matrix of shape `(m, n)` with all elements equal to `u`.

    Input ports:
        (0) The scalar input signal.

    Output ports:
        (0) The broadcasted output signal.

    Parameters:
        m:
            The number of rows in the output matrix.  If `m` is None, then the output
            will be a vector with shape `(n,)`. To get a row vector of size `(1,n)`,
            set `m=1` expliclty.
        n:
            The number of columns in the output matrix.  If `n` is None, then the
            output will be a vector with shape `(m,)`. To get a column vector of size
            `(m,1)`, set `n=1` expliclty.
    """

    @parameters(static=["m", "n"])
    def __init__(self, m, n, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, m, n):
        if m is not None:
            m = int(m)
        else:
            m = 0
        if n is not None:
            n = int(n)
        else:
            n = 0

        if m > 0 and n > 0:
            ones_ = npa.ones((m, n))
        elif m > 0:
            ones_ = npa.ones((m,))
        elif n > 0:
            ones_ = npa.ones((n,))
        else:
            raise BlockParameterError(
                message=f"ScalarBroadcast block {self.name} at least m or n must not be None or Zero"
            )
        self.replace_op(lambda x: ones_ * x)



class SquareRoot(FeedthroughBlock):
    """Compute the square root of the input signal.

    Dispatches to `jax.numpy.sqrt`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sqrt.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The square root of the input signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(npa.sqrt, *args, **kwargs)


class Stack(ReduceBlock):
    """Stack the input signals into a single output signal along a new axis.

    Dispatches to `jax.numpy.stack`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.stack.html

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The stacked output signal.

    Parameters:
        axis:
            The axis along which the input signals are stacked.  Default is 0.
    """

    @parameters(static=["axis"])
    def __init__(self, n_in, axis=0, **kwargs):
        super().__init__(n_in, None, **kwargs)

    def initialize(self, axis):
        self.replace_op(partial(npa.stack, axis=int(axis)))



class SumOfElements(FeedthroughBlock):
    """Compute the sum of the elements of the input signal.

    Dispatches to `jax.numpy.sum`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sum.html

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The sum of the elements of the input signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(npa.sum, *args, **kwargs)


class Trigonometric(FeedthroughBlock):
    """Apply a trigonometric function to the input signal.

    Available functions are:
        sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh

    Dispatches to `jax.numpy.sin`, `jax.numpy.cos`, etc, so see the JAX docs for details.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The trigonometric function applied to the input signal.

    Parameters:
        function:
            The trigonometric function to apply to the input signal.  Must be one of
            "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh".
    """

    @parameters(static=["function"])
    def __init__(self, function, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, function):
        func_lookup = {
            "sin": npa.sin,
            "cos": npa.cos,
            "tan": npa.tan,
            "asin": npa.arcsin,
            "acos": npa.arccos,
            "atan": npa.arctan,
            "sinh": npa.sinh,
            "cosh": npa.cosh,
            "tanh": npa.tanh,
            "asinh": npa.arcsinh,
            "acosh": npa.arccosh,
            "atanh": npa.arctanh,
        }
        if function not in func_lookup:
            raise BlockParameterError(
                message=f"Trigonometric block {self.name} has invalid selection {function} for 'function'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()]),
                parameter_name="function",
            )
        self.replace_op(func_lookup[function])
