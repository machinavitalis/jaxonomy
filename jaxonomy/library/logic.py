# SPDX-License-Identifier: MIT

"""Boolean operators, comparators, switches, and truth tables."""

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
    "Comparator",
    "LogicalOperator",
    "LogicalReduce",
    "IfThenElse",
    "Relay",
    "Switch",
    "MultiPortSwitch",
    "TruthTable",
    "TruthTableBuilder",
]



class Comparator(LeafSystem):
    """Compare two signals using typical relational operators.

    When using == and != operators, the block uses tolerances to determine if the
    expression is true or false.

    Parameters:
        operator: one of ("==", "!=", ">=", ">", ">=", "<")
        atol: the absolute tolerance value used with "==" or "!="
        rtol: the relative tolerance value used with "==" or "!="

    Input Ports:
        (0) The left side operand
        (1) The right side operand

    Output Ports:
        (0) The result of the comparison (boolean signal)

    Events:
        An event is triggered when the output changes from true to false or vice versa.
    """

    @parameters(static=["operator", "atol", "rtol"])
    def __init__(self, atol=1e-5, rtol=1e-8, operator=None, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, atol, rtol, operator):
        func_lookup = {
            ">": npa.greater,
            ">=": npa.greater_equal,
            "<": npa.less,
            "<=": npa.less_equal,
            "==": self._equal,
            "!=": self._ne,
        }

        if operator not in func_lookup:
            message = (
                f"Comparator block '{self.name}' has invalid selection "
                + f"'{operator}' for parameter 'operator'. Valid options: "
                + ",".join([k for k in func_lookup.keys()])
            )
            raise BlockParameterError(
                message=message, system=self, parameter_name="operator"
            )

        self.rtol = rtol
        self.atol = atol

        compare = func_lookup[operator]

        def _compute_output(_time, _state, *inputs, **_params):
            return compare(*inputs)

        self.configure_output_port(
            self._output_port_idx,
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )
        self.evt_direction = self._process_operator(operator)

    def _equal(self, x, y):
        if npa.issubdtype(x.dtype, npa.floating):
            return npa.isclose(x, y, self.rtol, self.atol)
        return x == y

    def _ne(self, x, y):
        if npa.issubdtype(x.dtype, npa.floating):
            return npa.logical_not(npa.isclose(x, y, self.rtol, self.atol))
        return x != y

    def _zero_crossing(self, _time, _state, *inputs, **_params):
        return inputs[0] - inputs[1]

    def _process_operator(self, operator):
        if operator in ["<", "<="]:
            return "positive_then_non_positive"
        if operator in [">", ">="]:
            return "negative_then_non_negative"
        return "crosses_zero"

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output is
        # fed to an ODE.
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(
                self._zero_crossing, direction=self.evt_direction
            )

        return super().initialize_static_data(context)



class IfThenElse(LeafSystem):
    """Applies a conditional expression to the input signals.

    Given inputs `pred`, `true_val`, and `false_val`, the block computes:
    ```
    y = true_val if pred else false_val
    ```

    The true and false values may be any arrays, but must have the same
    shape and dtype.

    Input ports:
        (0) The boolean predicate.
        (1) The true value.
        (2) The false value.

    Output ports:
        (0) The result of the conditional expression. Shape and dtype will match
            the true and false values.

    Events:
        An event is triggered when the output changes from true to false or vice versa.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()  # pred
        self.declare_input_port()  # true_val
        self.declare_input_port()  # false_val

        def _compute_output(_time, _state, *inputs, **_params):
            return npa.where(inputs[0], inputs[1], inputs[2])

        self.declare_output_port(
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    def _edge_detection(self, _time, _state, *inputs, **_params):
        return npa.where(inputs[0], 1.0, -1.0)

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output is
        # fed to an ODE.
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

        return super().initialize_static_data(context)



class LogicalOperator(LeafSystem):
    """Apply a boolean function elementwise to the input signals.

    This block implements the following boolean functions:
        - "or": same as np.logical_or
        - "and": same as np.logical_and
        - "not": same as np.logical_not
        - "nor": equivalent to np.logical_not(np.logical_or(in_0,in_1))
        - "nand": equivalent to np.logical_not(np.logical_and(in_0,in_1))
        - "xor": same as np.logical_xor

    Input ports:
        (0,1) The input signals.  If numeric, they are interpreted as boolean
            types (so 0 is False and any other value is True).

    Output ports:
        (0) The result of the logical operation, a boolean-valued signal.

    Parameters:
        function:
            The boolean function to apply. One of "or", "and", "not", "nor", "nand",
            or "xor".

    Events:
        An event is triggered when the output changes from True to False or vice versa.
    """

    @parameters(static=["function"])
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        if not function == "not":
            self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def initialize(self, function):
        self.function = function
        func_lookup = {
            "or": self._or,
            "and": self._and,
            "not": self._not,
            "xor": self._xor,
            "nor": self._nor,
            "nand": self._nand,
        }
        if function not in func_lookup:
            raise BlockParameterError(
                message=f"LogicalOperator block {self.name} has invalid selection {function} for 'function'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()]),
                system=self,
            )

        if function != "not" and len(self.input_ports) < 2:
            raise BlockParameterError(
                message=f"Can't change logical operator from 'not' to {function} for block {self.name}",
                system=self,
            )

        if function == "not" and len(self.input_ports) > 1:
            raise BlockParameterError(
                message=f"Can't change logical operator from {function} to 'not' for block {self.name}",
                system=self,
            )

        self._func = func_lookup[function]

        self.configure_output_port(
            self._output_port_idx,
            self._func,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def _edge_detection(self, time, state, *inputs, **params):
        outp = self._func(time, state, *inputs, **params)
        return npa.where(outp, 1.0, -1.0)

    def _or(self, time, state, *inputs, **parameters):
        return npa.logical_or(npa.array(inputs[0]), npa.array(inputs[1]))

    def _and(self, time, state, *inputs, **parameters):
        return npa.logical_and(npa.array(inputs[0]), npa.array(inputs[1]))

    def _not(self, time, state, *inputs, **parameters):
        (x,) = inputs
        return npa.logical_not(npa.array(x))

    def _xor(self, time, state, *inputs, **parameters):
        return npa.logical_xor(npa.array(inputs[0]), npa.array(inputs[1]))

    def _nor(self, time, state, *inputs, **parameters):
        return npa.logical_not(
            npa.logical_or(npa.array(inputs[0]), npa.array(inputs[1]))
        )

    def _nand(self, time, state, *inputs, **parameters):
        return npa.logical_not(
            npa.logical_and(npa.array(inputs[0]), npa.array(inputs[1]))
        )

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity.  For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

        return super().initialize_static_data(context)


class LogicalReduce(FeedthroughBlock):
    """Apply a boolean reduce function to the elements of the input signal.

    This block implements the following boolean functions:
        - "any": Output is True if any input element is True.
        - "all": Output is True if all input elements are True.

    Input ports:
        (0) The input signal.  If numeric, they are interpreted as boolean
            types (so 0 is False and any other value is True).

    Output ports:
        (0) The result of the logical operation, a boolean-valued signal.

    Parameters:
        function:
            The boolean function to apply. One of "any", "all".
        axis:
            Axis or axes along which a logical OR/AND reduction is performed.

    Events:
        An event is triggered when the output changes from True to False or vice versa.
    """

    @parameters(static=["function", "axis"])
    def __init__(self, function, axis=None, **kwargs):
        super().__init__(None, **kwargs)

    def initialize(self, function, axis=None):
        self.function = function
        self.axis = int(axis) if axis is not None else None
        func_lookup = {
            "any": self._any,
            "all": self._all,
        }
        if function not in func_lookup:
            raise BlockParameterError(
                message=f"LogicalReduce block {self.name} has invalid selection {function} for 'function'. Valid options: "
                + ", ".join([f for f in func_lookup.keys()])
            )

        self._func = func_lookup[function]
        self.replace_op(self._func)

    def _edge_detection(self, _time, _state, *inputs, **_params):
        outp = self._func(inputs)
        return npa.where(outp, 1.0, -1.0)

    def _any(self, inputs):
        return npa.any(npa.array(inputs), axis=self.axis)

    def _all(self, inputs):
        return npa.all(npa.array(inputs), axis=self.axis)

    def initialize_static_data(self, context):
        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity.  For efficiency, only do this if the output
        # is fed to an ODE block
        if not self.has_zero_crossing_events and is_discontinuity(self.output_ports[0]):
            self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

        return super().initialize_static_data(context)



class Relay(LeafSystem):
    """Simple state machine implementing hysteresis behavior.

    The input-output map is as follows:

    ```
            output
              |
    on_value  |          -------<------<---------------------
              |          |                    |
              |          ⌄                    ^
              |          |                    |
    off_value |----------|-------->----->-----|
              |
              |---------------------------------------------- input
                         | off_threshold      | on_threshold
    ```

    Note that the "time mode" behavior of this block will follow the input
    signal.  That is, if the input signal varies continuously in time, then
    the zero-crossing event from OFF->ON or vice versa will be localized in
    time.  On the other hand, if the input signal varies only as a result
    of periodic updates to the discrete state, the relay will only change state
    at those instants.  If the input signal is continuous, the block can
    be "forced" to this discrete-time periodic behavior by adding a ZeroOrderHold
    block before the input.

    The exception to this is the case where there are no blocks in the system
    containing either discrete or continuous state.  In this case the state changes
    will only be localized to the resolution of the major step.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The relay output signal, which is equal to either the on_value or
            the off_value, depending on the internal state of the relay.

    Parameters:
        on_threshold:
            When input rises above this value, the internal state transitions to ON.
        off_threshold:
            When input falls below this value, the internal state transitions to OFF.
        on_value:
            Value of the output signal when state is ON.
        off_value:
            Value of the output signal when state is OFF
        initial_state:
            If equal to on_value, the block will be initialized in the ON state.
            Otherwise, it will be initialized to the OFF state.

    Events:
        There are two zero-crossing events: one to transition from OFF->ON and one
        for the opposite transition from ON->OFF.
    """

    class State(IntEnum):
        OFF = 0
        ON = 1

    @parameters(
        dynamic=[
            "on_threshold",
            "off_threshold",
            "initial_state",
            "on_value",
            "off_value",
        ],
    )
    def __init__(
        self, on_threshold, off_threshold, on_value, off_value, initial_state, **kwargs
    ):
        super().__init__(**kwargs)

        self.declare_default_mode(
            self.State.ON if initial_state == on_value else self.State.OFF
        )

        self.declare_input_port()
        self.declare_output_port(
            self._output,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.mode],
        )

        # transition to ON event
        def _on_guard(_time, _state, u, **parameters):
            return u - parameters["on_threshold"]

        self.declare_zero_crossing(
            guard=_on_guard,
            direction="negative_then_non_negative",
            start_mode=self.State.OFF,
            end_mode=self.State.ON,
        )

        # transition to OFF event
        def _off_guard(_time, _state, u, **parameters):
            return u - parameters["off_threshold"]

        self.declare_zero_crossing(
            guard=_off_guard,
            direction="positive_then_non_positive",
            start_mode=self.State.ON,
            end_mode=self.State.OFF,
        )

    def initialize(
        self, on_threshold, off_threshold, on_value, off_value, initial_state
    ):
        self.configure_default_mode(
            self.State.ON if initial_state == on_value else self.State.OFF
        )

    def reset_default_values(self, **dynamic_parameters):
        self.configure_default_mode(
            self.State.ON
            if dynamic_parameters["initial_state"] == dynamic_parameters["on_value"]
            else self.State.OFF
        )

    def _output(self, _time, state, **parameters):
        return npa.where(
            state.mode == self.State.ON,
            parameters["on_value"],
            parameters["off_value"],
        )



# ---------------------------------------------------------------------------
# T-118 phase 1 — Switch / MultiPortSwitch (block-diagram-style).
# (Original task ID T-MW-205, renumbered to T-118 in 124c178.)
#
# ``Switch`` is a 3-input data-routing block: data_a, control, data_b →
# data_a if ``criteria(control, threshold)`` else data_b. Implemented via
# ``npa.where`` so gradients flow through both branches (selector branch
# is non-differentiable, expected). This is the differentiable phase-1
# subset of the spec.
#
# T-118-followup-modes (2026-05-09) added ``mode="smooth"``: a sigmoid
# blend ``alpha * data_a + (1 - alpha) * data_b`` with
# ``alpha = sigmoid(sharpness * sign * (control - threshold))`` (the
# sign is +1 for ``>=``/``>`` and -1 for ``<=``/``<`` so the smooth
# output approaches the hard answer in the strict-active region; the
# equality criteria ``==``/``!=`` cannot be sigmoid-approximated and
# raise on construction in smooth mode). The marketing wedge: gradient
# flows through ``threshold`` itself, which the hard ``where`` zeroes
# out. T-118-followup-cond-mode (2026-05-10) shipped ``mode="hard"``
# via ``jax.lax.cond``: only the active branch is evaluated, which
# matters when one branch is much more expensive than the other
# (e.g. conditional MJX simulation) or when branches have incompatible
# side effects. The trade-off: ``lax.cond`` requires a scalar predicate
# and is incompatible with ``vmap`` — using ``mode="hard"`` under
# ``simulate_batch`` raises a ``TracerBoolConversionError`` from JAX.
# Users who explicitly opt into hard-mode and aren't batching get the
# only-one-branch-computed property; everyone else should stick with
# the where/smooth defaults. On older JAX (<0.4) a batched predicate
# raises ``TracerBoolConversionError`` outright; on modern JAX the
# cond is silently rewritten to a select over both branches —
# numerically correct, but the perf benefit is gone. ``mode="where"``
# (the default) is byte-equivalent to phase 1.
#
# Sigmoid is computed via ``0.5 * (1 + tanh(x / 2))`` to stay inside
# ``npa`` (the backend doesn't expose ``sigmoid``/``expit`` directly)
# and because the tanh form is numerically a touch better behaved than
# ``1 / (1 + exp(-x))`` for very negative arguments.
#
# ``MultiPortSwitch(n_data_inputs)`` takes one selector input plus
# n data inputs and outputs ``data[clip(round(selector), 0, n-1)]``.
# Implemented via ``npa.stack`` + integer indexing rather than
# ``jax.lax.switch`` because the latter requires a static branch list
# whose closures we'd have to build at __init__; index-based selection
# is simpler, fully differentiable through the selected branch's
# upstream signal, and follows the standard ``zero-based`` indexing
# convention. ``one-based`` indexing and ``mode="smooth"`` (softmax-blend)
# are deferred — see T-118-followup-modes.
# ---------------------------------------------------------------------------


# Sign convention for smooth-mode sigmoid: maps each criterion to the
# sign of ``(control - threshold)`` that should send ``alpha -> 1``
# (i.e. pick data_a) deep in the strict-active region. ``==`` and ``!=``
# are intentionally absent — neither has a meaningful sigmoid
# approximation (a smooth bump or its negation would need a different
# formula) and we'd rather fail loudly than silently route through one
# of the inequalities.
_SWITCH_SMOOTH_SIGN = {
    ">=": 1.0,
    ">": 1.0,
    "<=": -1.0,
    "<": -1.0,
}


def _switch_sigmoid(x):
    # Stay in npa.* — the backend doesn't expose sigmoid/expit directly,
    # and 0.5*(1 + tanh(x/2)) is bit-exact equal to 1/(1+exp(-x)) up to
    # FP rounding while being slightly better behaved at large |x|.
    return 0.5 * (1.0 + npa.tanh(x / 2.0))


_SWITCH_VALID_MODES = ("where", "smooth", "hard")


_SWITCH_CRITERIA = {
    ">=": npa.greater_equal,
    ">": npa.greater,
    "<=": npa.less_equal,
    "<": npa.less,
    "!=": npa.not_equal,
    "==": npa.equal,
}


class Switch(LeafSystem):
    """Route one of two data signals based on a thresholded control signal.

    Three inputs ``(data_a, control, data_b)`` and one output:

    .. code-block:: python

        y = data_a if criteria(control, threshold) else data_b

    Default ``mode="where"`` is implemented via ``npa.where``, so JAX
    gradients flow through *both* data branches simultaneously (the
    selector branch is treated as non-differentiable, which is the
    only well-defined choice for a hard threshold).

    ``mode="smooth"`` replaces the hard ``where`` with a sigmoid blend

    .. code-block:: python

        alpha  = sigmoid(sharpness * sign * (control - threshold))
        y      = alpha * data_a + (1 - alpha) * data_b

    where ``sign`` is +1 for the ``>=``/``>`` criteria and -1 for the
    ``<=``/``<`` criteria, so the smooth output approaches the hard
    answer in the strict-active region as ``sharpness -> inf``. This
    mode lets gradients flow through the *threshold itself*, which the
    hard ``where`` zeroes out — the killer feature for trajectory
    optimization where the threshold is a tunable parameter.

    ``data_a`` and ``data_b`` must be broadcast-compatible (same as
    ``npa.where``'s requirements). The block does not enforce that
    ``control`` is a scalar — element-wise selection is supported when
    ``control`` and the data inputs broadcast together.

    Parameters:
        threshold: scalar threshold against which ``control`` is compared.
        criteria: one of ``">="``, ``">"``, ``"<="``, ``"<"``, ``"=="``,
            ``"!="``. Default ``">="``. The equality criteria
            (``"=="``/``"!="``) are not supported in ``mode="smooth"``
            (no sigmoid approximation makes sense for them).
        mode: one of ``"where"`` (default), ``"smooth"``, or ``"hard"``.
            ``"where"`` is byte-equivalent to T-118 phase 1 and the
            right pick for simulation. ``"smooth"`` is the right pick
            for gradient-based optimization through the threshold.
            ``"hard"`` dispatches to ``jax.lax.cond`` so only the
            active branch is evaluated — useful when one branch is
            much more expensive than the other or when branches have
            incompatible side effects. **``Switch(mode='hard')`` is
            incompatible with vmap; use mode='where' or mode='smooth'
            for batched use** (e.g. under ``simulate_batch``). On
            older JAX (<0.4) a batched predicate raises
            ``TracerBoolConversionError``; on modern JAX the cond is
            silently rewritten to evaluate both branches with a
            select, which is numerically correct but defeats the
            entire point of picking ``mode='hard'`` over ``mode='where'``.
        sharpness: positive scalar controlling sigmoid steepness in
            ``mode="smooth"``. Default ``10.0``. Larger values give a
            tighter approximation to the hard switch but smaller (and
            faster vanishing) gradients in the strict-active region.
            Ignored when ``mode="where"``.

    Input ports:
        (0) data_a — output when criteria(control, threshold) is True.
        (1) control — the selector signal compared to ``threshold``.
        (2) data_b — output when criteria(control, threshold) is False.

    Output ports:
        (0) The selected data signal, with shape determined by
            broadcasting between data_a and data_b.
    """

    @parameters(static=["threshold", "criteria", "mode"], dynamic=["sharpness"])
    def __init__(
        self,
        threshold=0.0,
        criteria=">=",
        mode="where",
        sharpness=10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if criteria not in _SWITCH_CRITERIA:
            raise BlockParameterError(
                message=(
                    f"Switch block '{self.name}' has invalid selection "
                    f"'{criteria}' for parameter 'criteria'. Valid options: "
                    + ",".join(_SWITCH_CRITERIA.keys())
                ),
                system=self,
                parameter_name="criteria",
            )

        if mode not in _SWITCH_VALID_MODES:
            raise BlockParameterError(
                message=(
                    f"Switch block '{self.name}' has invalid selection "
                    f"'{mode}' for parameter 'mode'. Valid options: "
                    + ",".join(_SWITCH_VALID_MODES)
                    + "."
                ),
                system=self,
                parameter_name="mode",
            )

        if mode == "smooth":
            if criteria not in _SWITCH_SMOOTH_SIGN:
                raise BlockParameterError(
                    message=(
                        f"Switch block '{self.name}': mode='smooth' is not "
                        f"compatible with criteria='{criteria}' (no sigmoid "
                        "approximation defined for equality). Use one of: "
                        + ",".join(_SWITCH_SMOOTH_SIGN.keys())
                        + "."
                    ),
                    system=self,
                    parameter_name="criteria",
                )
            if not np.isfinite(sharpness) or sharpness <= 0:
                raise BlockParameterError(
                    message=(
                        f"Switch block '{self.name}': sharpness must be "
                        f"finite and > 0, got {sharpness}."
                    ),
                    system=self,
                    parameter_name="sharpness",
                )

        self.declare_input_port()  # data_a
        self.declare_input_port()  # control
        self.declare_input_port()  # data_b
        self._output_port_idx = self.declare_output_port()

    def initialize(self, threshold, criteria, mode, sharpness=10.0):
        if mode == "where":
            compare = _SWITCH_CRITERIA[criteria]

            def _compute_output(_time, _state, *inputs, **_params):
                data_a, control, data_b = inputs
                return npa.where(compare(control, threshold), data_a, data_b)
        elif mode == "hard":
            # Local import keeps the JAX dependency lazy for non-JAX
            # backends that load primitives.py for class definitions.
            # ``lax.cond`` evaluates only the active branch — the whole
            # point of hard mode — but requires a scalar predicate, so
            # this path is incompatible with vmap (and therefore with
            # ``simulate_batch``). Documented at the class level.
            from jax import lax as _jlax

            compare = _SWITCH_CRITERIA[criteria]

            def _compute_output(_time, _state, *inputs, **_params):
                data_a, control, data_b = inputs
                pred = compare(control, threshold)
                # operand-passing form keeps the closure pure and lets
                # XLA elide unused captures; both branches must return
                # the same shape/dtype, which is broadcast-compatible
                # by the same rule as npa.where in the where path.
                return _jlax.cond(
                    pred,
                    lambda ops: ops[0],
                    lambda ops: ops[1],
                    (data_a, data_b),
                )
        else:
            # mode == "smooth"; validated in __init__ so the lookup is safe.
            sign = _SWITCH_SMOOTH_SIGN[criteria]

            def _compute_output(_time, _state, *inputs, **params):
                data_a, control, data_b = inputs
                # sharpness flows through dynamic params so optimizers can
                # anneal it; threshold is static so it appears in the
                # closure as a Python value (still differentiable via
                # jax.grad on the underlying op — the sigmoid is smooth
                # in `threshold`).
                k = params.get("sharpness", sharpness)
                alpha = _switch_sigmoid(k * sign * (control - threshold))
                return alpha * data_a + (1.0 - alpha) * data_b

        self.configure_output_port(
            self._output_port_idx,
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )


class MultiPortSwitch(LeafSystem):
    """Route one of N data signals based on an integer selector input.

    Inputs are ``(selector, data_0, data_1, ..., data_{n-1})`` and the
    output is ``data_{clip(round(selector), 0, n-1)}``.

    Implementation strategy: stack the data inputs along a new leading
    axis and pick out the selected slice with integer indexing. This is
    fully differentiable through the *selected* data input (zero
    gradient on the others), which is the standard documented
    semantics for this block. The ``selector`` is rounded and clipped to ``[0, n-1]``
    so floating-point inputs are tolerated; the selector itself is
    non-differentiable (``round``/``clip`` zero out the gradient).

    All data inputs must share the same shape and dtype (the stack
    requires it). Mixed shapes are rejected by ``npa.stack`` at trace
    time.

    Parameters:
        n_data_inputs: number of data input ports. Must be ``>= 1``.
        choice_names: optional tuple of unique non-empty string labels,
            one per data input, used for self-documenting diagrams.
            When supplied its length MUST equal ``n_data_inputs`` and
            entries must be distinct. Resolve a friendly name to its
            integer selector via ``index_of(name)`` when wiring the
            diagram — see "Named choices" below. Default ``None``
            (integer-only, byte-equivalent to phase 1).

    Input ports:
        (0) selector — scalar integer-valued signal in ``[0, n-1]``.
            Floating values are rounded and clipped.
        (1..n_data_inputs) data inputs.

    Output ports:
        (0) The data input at index ``selector``.

    Notes:
        The original T-118 spec includes ``indexing="one-based"`` and
        ``mode="smooth"`` (softmax-blend across data inputs). Both are
        deferred — see T-118-followup-modes. Zero-based indexing is the
        only mode supported in phase 1, matching Python conventions.

    Named choices (T-118-followup-multi-port-string-keys, 2026-05-13):
        ``choice_names=("low", "medium", "high")`` lets a diagram
        document which port means what. The selector port itself still
        expects an integer at runtime — JAX cannot trace strings, so
        this is the build-time-only interpretation the followup spec
        calls out. Look up the integer for a name on the Python side:

        .. code-block:: python

            mps = MultiPortSwitch(3, choice_names=("low", "med", "high"))
            sel = library.Constant(mps.index_of("med"))   # → 1

        Passing a string to ``index_of`` returns the matching integer;
        passing an int returns it unchanged after a range check, so
        callers can mix the two without branching. Unknown strings and
        out-of-range ints raise ``BlockParameterError`` at construction
        (build) time, not at trace time. The runtime _compute_output
        path is unchanged when ``choice_names`` is ``None`` — the
        default-off byte-equivalence guarantee.
    """

    def __init__(self, n_data_inputs, choice_names=None, **kwargs):
        super().__init__(**kwargs)

        n = int(n_data_inputs)
        if n < 1:
            raise BlockParameterError(
                message=(
                    f"MultiPortSwitch block '{self.name}' requires "
                    f"n_data_inputs >= 1; got {n_data_inputs}."
                ),
                system=self,
                parameter_name="n_data_inputs",
            )
        self._n_data_inputs = n

        # Validate + store choice_names. ``None`` preserves the phase-1
        # path byte-for-byte: no extra branches in _compute_output, no
        # extra state visible to traces. Build-time-only by design —
        # see class docstring.
        if choice_names is not None:
            names = tuple(choice_names)
            if len(names) != n:
                raise BlockParameterError(
                    message=(
                        f"MultiPortSwitch block '{self.name}': "
                        f"choice_names has {len(names)} entries but "
                        f"n_data_inputs={n}."
                    ),
                    system=self,
                    parameter_name="choice_names",
                )
            for nm in names:
                if not isinstance(nm, str) or not nm:
                    raise BlockParameterError(
                        message=(
                            f"MultiPortSwitch block '{self.name}': "
                            f"choice_names entries must be non-empty "
                            f"strings; got {nm!r}."
                        ),
                        system=self,
                        parameter_name="choice_names",
                    )
            if len(set(names)) != len(names):
                raise BlockParameterError(
                    message=(
                        f"MultiPortSwitch block '{self.name}': "
                        f"choice_names entries must be unique; got "
                        f"{names!r}."
                    ),
                    system=self,
                    parameter_name="choice_names",
                )
            self._choice_names = names
            self._choice_index = {name: i for i, name in enumerate(names)}
        else:
            self._choice_names = None
            self._choice_index = None

        self.declare_input_port()  # selector
        for _ in range(n):
            self.declare_input_port()  # data_i

        def _compute_output(_time, _state, *inputs, **_params):
            selector = inputs[0]
            data = inputs[1:]
            stacked = npa.stack(data, axis=0)
            idx = npa.clip(npa.round(selector).astype(npa.int32), 0, n - 1)
            return stacked[idx]

        self.declare_output_port(
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    @property
    def choice_names(self):
        """Tuple of channel labels, or ``None`` if unlabeled."""
        return self._choice_names

    def index_of(self, selector):
        """Resolve a string or int selector to its integer index.

        ``selector`` may be a string (looked up in ``choice_names``)
        or any object convertible via ``int()``. Strings only resolve
        when ``choice_names`` was supplied at construction.
        Out-of-range ints and unknown strings raise
        ``BlockParameterError`` at *build time*; runtime selectors
        flowing through the input port are still clipped silently by
        ``_compute_output`` (no change to the runtime path).
        """
        if isinstance(selector, str):
            if self._choice_index is None:
                raise BlockParameterError(
                    message=(
                        f"MultiPortSwitch block '{self.name}': string "
                        f"selector {selector!r} requires choice_names "
                        f"at construction."
                    ),
                    system=self,
                    parameter_name="choice_names",
                )
            if selector not in self._choice_index:
                raise BlockParameterError(
                    message=(
                        f"MultiPortSwitch block '{self.name}': unknown "
                        f"choice {selector!r}; valid options: "
                        + ",".join(self._choice_names)
                        + "."
                    ),
                    system=self,
                    parameter_name="choice_names",
                )
            return self._choice_index[selector]
        idx = int(selector)
        if idx < 0 or idx >= self._n_data_inputs:
            raise BlockParameterError(
                message=(
                    f"MultiPortSwitch block '{self.name}': integer "
                    f"selector {idx} out of range [0, "
                    f"{self._n_data_inputs - 1}]."
                ),
                system=self,
                parameter_name="choice_names",
            )
        return idx


# ---------------------------------------------------------------------------
# T-119 phase 1 — TruthTable block (hybrid state-machine-modelling style).
# (Original task ID T-MW-206, renumbered to T-119 in 124c178.)
#
# A ``TruthTable`` evaluates a fixed list of ``(input_pattern, output)`` rows
# over a tuple of boolean-castable inputs and returns the output of the first
# matching pattern (or ``default_output`` if none match). Patterns may use
# the string ``"X"`` as a wildcard for an input position.
#
# Implementation: a chained ``npa.where`` over the rows, evaluated in
# *reverse* order so that earlier rows take precedence (since each
# ``where`` overwrites the previous result when its match is True). This
# is fully traceable under JAX — no Python control flow on input values
# — and preserves T-005 default-float64: outputs are ``npa.asarray``'d
# without explicit casting, so float defaults stay float64.
#
# Differentiability: constant outputs are not differentiable w.r.t. inputs
# (no dependence on the boolean inputs). The block IS differentiable
# w.r.t. its OUTPUT VALUES if those happen to be parameters in a wider
# diagram (each output is just an array constant captured by closure).
#
# T-119-followup-numeric-output extends row outputs to accept CALLABLES:
# ``output = lambda *inputs: <expression>`` is invoked with the RAW input
# values (not the bool-cast pattern-matching versions) at each step, so
# row outputs can depend on numerical input values — the classic
# "action" cell. Pattern matching still uses the bool cast, so every
# input is mixed-use: bool for matching, raw value for callable outputs.
# Callables trace through JAX cleanly (they are invoked at trace time
# like any other computation) and grads flow through the active row's
# callable via the same ``npa.where`` machinery — branches are
# *evaluated* (not switched) so the callable must be defined for ALL
# inputs, not just those matching its pattern.
#
# T-119-followup-default-callable extends ``default_output`` to also
# accept a callable with the same semantics: the fallback value (used
# when no row matches) may be computed from the raw inputs via
# ``default_output=lambda *inputs: <expression>``. The callable is
# evaluated at every step (under the same branchless ``where``
# semantics as row callables) and its result becomes the seed for the
# row chain. ``to_dict`` / ``to_csv`` reject callable defaults with
# the same error path as callable rows — Python closures aren't
# JSON/CSV-serialisable.
#
# Phase 1 stores ``rows`` and ``default_output`` as plain Python
# attributes (not @parameters) because list-of-tuples-with-strings does
# not round-trip cleanly through the Equinox parameter machinery; the
# T-119-followup-serialization shipment provides explicit
# ``to_dict()`` / ``from_dict()`` round-trip helpers (see method docs
# below) so callers can persist a TruthTable to JSON without going
# through the dashboard model serializer. Full @parameters wiring would
# require ``declare_static_parameters`` to accept arbitrary nested
# Python (it currently coerces lists to ``np.array`` which would
# clobber the ``[(pattern, output)]`` structure); plumbing that into
# the dashboard's ``to_model_json``/``from_model_json`` pair is left as
# a deeper followup.
#
# The static-completeness checker described in the spec
# is also deferred — see T-119-followup-completeness-checker.
# ---------------------------------------------------------------------------


class TruthTable(LeafSystem):
    """Evaluate a fixed truth table over boolean-castable inputs.

    Given a list of ``(input_pattern, output)`` rows, this block compares
    its inputs against each pattern and emits the output of the first
    matching row (or ``default_output`` if none match). Patterns are tuples
    of ``bool`` values or the string ``"X"`` as a wildcard.

    Example — a 2-input AND gate:

    .. code-block:: python

        tt = TruthTable(
            rows=[
                ((True, True), 1.0),
                ((True, False), 0.0),
                ((False, True), 0.0),
                ((False, False), 0.0),
            ],
            n_inputs=2,
            default_output=0.0,
        )

    Wildcard example — ignore the first input:

    .. code-block:: python

        tt = TruthTable(
            rows=[(("X", True), 1.0), (("X", False), 0.0)],
            n_inputs=2,
            default_output=0.0,
        )

    Callable output example — row output depends on raw input values
    (T-119-followup-numeric-output):

    .. code-block:: python

        tt = TruthTable(
            rows=[
                ((True, True), lambda a, b: a + b),
                ((True, False), lambda a, b: a - b),
                ((False, "X"), 0.0),  # constant fallback row
            ],
            n_inputs=2,
            default_output=0.0,
        )

    Parameters:
        rows: list of ``(pattern, output)`` tuples. ``pattern`` is a
            length-``n_inputs`` tuple whose entries are ``bool`` (matched
            literally) or the string ``"X"`` (wildcard, matches anything).
            ``output`` may be a scalar/array (constant for that row) or
            a callable ``f(*inputs) -> scalar_or_vector`` invoked with
            the RAW inputs (every row callable is evaluated at every
            step under JAX's branchless ``where`` semantics; the result
            is *selected* only when the row matches). All row outputs
            (and the callable return values) must broadcast against
            ``default_output``.
        n_inputs: number of input ports.
        default_output: value emitted when no row matches. May be a
            scalar/array (constant fallback) or a callable
            ``f(*inputs) -> scalar_or_vector`` invoked with the RAW
            inputs (T-119-followup-default-callable). For a callable
            default, the output shape/dtype is determined at trace
            time by the callable's return value; for a constant
            default, it is determined statically from the value.

    Input ports:
        (0..n_inputs-1) Boolean-castable scalars. Non-boolean inputs are
        coerced to bool before pattern matching (any non-zero is True).

    Output ports:
        (0) The output of the first matching row, or ``default_output``.

    Notes:
        Earlier rows take precedence: if multiple patterns would match
        the same input combination, the one listed first in ``rows`` wins.
        The static-completeness/ambiguity checker is deferred
        (see ``T-119-followup-completeness-checker``); JSON serialization
        of the rows table is deferred (see
        ``T-119-followup-serialization``).
    """

    def __init__(self, rows, n_inputs, default_output, input_names=None, **kwargs):
        super().__init__(**kwargs)

        n = int(n_inputs)
        if n < 1:
            raise BlockParameterError(
                message=(
                    f"TruthTable block '{self.name}' requires n_inputs >= 1; "
                    f"got {n_inputs}."
                ),
                system=self,
                parameter_name="n_inputs",
            )

        # T-119-followup-truth-table-named-ports — labels forwarded from
        # ``TruthTableBuilder(input_names=...)`` (or supplied directly) become
        # the names of the declared input ports. Without this, the labels
        # survive only as ``.row(...)`` keyword targets and the block's
        # ``input_ports`` show up as anonymous ``in_0`` / ``in_1`` slots in
        # ``print_schedule`` and model JSON.
        if input_names is None:
            resolved_input_names = None
        else:
            resolved_input_names = tuple(input_names)
            if len(resolved_input_names) != n:
                raise BlockParameterError(
                    message=(
                        f"TruthTable block '{self.name}' input_names length "
                        f"({len(resolved_input_names)}) does not match "
                        f"n_inputs={n}."
                    ),
                    system=self,
                    parameter_name="input_names",
                )

        # Validate row shapes up front so misconfiguration fails at __init__,
        # not deep inside a JAX trace.
        # Each entry is ``(pattern, output_or_callable, is_callable)`` —
        # constant outputs are pre-coerced via ``npa.asarray``; callables
        # are stored as-is and invoked inside ``_compute_output``.
        # (T-119-followup-numeric-output)
        validated_rows: list[tuple[tuple, object, bool]] = []
        for row_idx, row in enumerate(rows):
            if not (isinstance(row, tuple) and len(row) == 2):
                raise BlockParameterError(
                    message=(
                        f"TruthTable block '{self.name}' row {row_idx} must "
                        f"be a (pattern, output) tuple; got {row!r}."
                    ),
                    system=self,
                    parameter_name="rows",
                )
            pattern, output = row
            if not (isinstance(pattern, tuple) and len(pattern) == n):
                raise BlockParameterError(
                    message=(
                        f"TruthTable block '{self.name}' row {row_idx} pattern "
                        f"must be a length-{n} tuple; got {pattern!r}."
                    ),
                    system=self,
                    parameter_name="rows",
                )
            for p in pattern:
                if p == "X":
                    continue
                if not isinstance(p, (bool, np.bool_)):
                    raise BlockParameterError(
                        message=(
                            f"TruthTable block '{self.name}' row {row_idx} "
                            f"pattern entries must be bool or 'X'; got {p!r}."
                        ),
                        system=self,
                        parameter_name="rows",
                    )
            if callable(output):
                # Defer evaluation to runtime; stored as-is so the
                # closure can call ``output(*inputs)`` with raw values.
                validated_rows.append((pattern, output, True))
            else:
                validated_rows.append((pattern, npa.asarray(output), False))

        # Stored as plain Python attributes — see module-header comment.
        # ``_default_output_is_callable`` mirrors the per-row
        # ``is_callable`` flag and selects the runtime branch in
        # ``_compute_output``. (T-119-followup-default-callable)
        self._rows = validated_rows
        self._n_inputs = n
        self._default_output_is_callable = callable(default_output)
        if self._default_output_is_callable:
            # Defer evaluation to runtime — the callable is invoked with
            # the raw inputs inside ``_compute_output``. We keep the
            # callable as-is so ``to_dict`` / ``to_csv`` can detect and
            # reject it.
            self._default_output = default_output
        else:
            self._default_output = npa.asarray(default_output)

        for i in range(n):
            if resolved_input_names is None:
                self.declare_input_port()
            else:
                self.declare_input_port(name=resolved_input_names[i])

        # Capture by closure to keep the compute function pure for JAX trace.
        rows_local = validated_rows
        default_local = self._default_output
        default_is_callable = self._default_output_is_callable

        # T-119-followup-truth-table-true-vectorise — when every row's
        # output is constant AND the default is constant (the common case
        # for combinational logic tables), pre-stack the outputs and the
        # pattern matrix so ``_compute_output`` compiles to one
        # ``argmax`` + ``take`` selection instead of N sequential
        # ``where`` operations. Earlier rows win on conflict, matching the
        # row-by-row loop semantics.
        all_constant = (
            not default_is_callable
            and all(not is_callable for _, _, is_callable in rows_local)
        )
        if all_constant and rows_local:
            # ``pattern_codes[r, i]`` is -1 for "don't care" (X), 0 for
            # False, 1 for True. Used at runtime to build a row-match
            # mask without re-walking the Python pattern tuples.
            pattern_codes = np.full((len(rows_local), n), -1, dtype=np.int8)
            for r, (pattern, _, _) in enumerate(rows_local):
                for i, p in enumerate(pattern):
                    if p == "X":
                        continue
                    pattern_codes[r, i] = int(bool(p))
            pattern_codes_local = pattern_codes
            outputs_stack_local = npa.stack(
                [output for _, output, _ in rows_local], axis=0
            )

            def _compute_output(_time, _state, *inputs, **_params):
                bool_inputs = npa.stack(
                    [npa.asarray(x).astype(npa.int8) for x in inputs], axis=0
                )
                # ``per_input_ok[r, i] = (pattern_codes[r, i] == -1) | (pattern_codes[r, i] == bool_inputs[i])``
                pc = npa.asarray(pattern_codes_local)
                per_input_ok = (pc == -1) | (pc == bool_inputs)
                match_vec = npa.all(per_input_ok, axis=1)
                any_match = npa.any(match_vec)
                # ``argmax`` returns the FIRST True, matching the
                # row-precedence semantics.
                first_idx = npa.argmax(match_vec)
                selected = outputs_stack_local[first_idx]
                return npa.where(any_match, selected, default_local)
        else:
            def _compute_output(_time, _state, *inputs, **_params):
                bool_inputs = tuple(npa.asarray(x).astype(bool) for x in inputs)
                if default_is_callable:
                    # Same semantics as a callable row output: evaluated
                    # unconditionally with raw inputs, selected by the
                    # ``where`` chain when no row matches.
                    # (T-119-followup-default-callable)
                    result = npa.asarray(default_local(*inputs))
                else:
                    result = default_local
                # Evaluate rows in reverse so that earlier rows take precedence
                # (each ``where`` overwrites the running result on a match).
                for pattern, output, is_callable in reversed(rows_local):
                    match = npa.array(True)
                    for i, p in enumerate(pattern):
                        if p == "X":
                            continue
                        match = match & (bool_inputs[i] == bool(p))
                    if is_callable:
                        # Pass RAW inputs (not bool-cast) so callable outputs
                        # see the actual numerical values. The callable is
                        # evaluated unconditionally — ``npa.where`` selects
                        # the active row, but both branches are traced by
                        # JAX. (T-119-followup-numeric-output)
                        row_value = npa.asarray(output(*inputs))
                    else:
                        row_value = output
                    result = npa.where(match, row_value, result)
                return result

        # ``default_value`` is only known statically when the default is
        # a constant array — for a callable default the shape/dtype is
        # discovered at trace time, so pass ``None`` and let the port
        # framework infer it from the first call.
        port_default = None if self._default_output_is_callable else self._default_output
        self.declare_output_port(
            _compute_output,
            default_value=port_default,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    @classmethod
    def builder(cls, n_inputs, default_output, input_names=None, **block_kwargs):
        """Construct a fluent builder for this truth table.

        See :class:`TruthTableBuilder` for usage. Equivalent to
        ``TruthTableBuilder(n_inputs, default_output, input_names, **block_kwargs)``.
        """
        return TruthTableBuilder(
            n_inputs=n_inputs,
            default_output=default_output,
            input_names=input_names,
            **block_kwargs,
        )

    # -----------------------------------------------------------------
    # T-119-followup-completeness-checker — static analysis on the truth table.
    #
    # The conventional TruthTable static analysis performs two construction-time checks:
    #   1. Completeness — every one of the 2^N input combinations is
    #      matched by at least one row's pattern (treating ``"X"`` as a
    #      wildcard). If not, those combinations would silently fall
    #      through to ``default_output``, which is a frequent source of
    #      logic bugs.
    #   2. Disjointness — no two rows match the same input combination.
    #      Jaxonomy's runtime semantics resolve overlaps by earlier-row-
    #      wins (see ``_compute_output``), so overlap is *not* a runtime
    #      error here, but flagging it surfaces unintended row shadowing.
    #
    # Default-off: ``validate(strict_completeness=False, strict_disjointness=
    # False)`` returns the report dict and never raises. Pass either flag
    # as ``True`` to escalate to ``BlockParameterError`` on the relevant
    # finding. Enumeration is pure-Python over ``itertools.product`` —
    # this runs at construction time, never inside a JAX trace.
    # -----------------------------------------------------------------

    def validate(self, strict_completeness=False, strict_disjointness=False):
        """Static analysis of the truth-table rows.

        Enumerates all ``2**n_inputs`` boolean input vectors and checks:

        * **Completeness** — each vector is matched by at least one row's
          pattern (with ``"X"`` as wildcard). Vectors that no row matches
          are reported as ``missing_patterns``; without coverage they
          silently hit ``default_output`` at runtime.
        * **Disjointness** — no two rows match the same input vector.
          Overlaps are reported as ``(earlier_idx, later_idx)`` pairs.
          Jaxonomy's runtime resolves overlaps by earlier-row-wins, so
          this is informational unless ``strict_disjointness=True``.

        Args:
            strict_completeness: if True, raise :class:`BlockParameterError`
                when any input combination is uncovered. Default False.
            strict_disjointness: if True, raise :class:`BlockParameterError`
                when any two rows match the same input combination.
                Default False.

        Returns:
            dict with keys:

            * ``covered_combinations`` (int) — number of distinct input
              vectors matched by at least one row.
            * ``total_combinations`` (int) — ``2 ** n_inputs``.
            * ``missing_patterns`` (list[tuple[bool, ...]]) — input
              vectors not matched by any row.
            * ``overlapping_pairs`` (list[tuple[int, int]]) — sorted
              ``(i, j)`` row-index pairs (``i < j``) where row ``i`` and
              row ``j`` both match at least one common input vector.

        Notes:
            For ``n_inputs > 10`` (i.e. > 1024 enumerated combinations)
            the check emits a :class:`UserWarning` since cost grows as
            ``2 ** n_inputs * len(rows)``.
        """
        import itertools

        n = self._n_inputs
        rows = self._rows

        if n > 10:
            warnings.warn(
                f"TruthTable.validate(): enumerating 2**{n} = {2 ** n} "
                f"input combinations across {len(rows)} rows may be slow. "
                f"Consider whether the static check is worth the cost for "
                f"this many inputs.",
                UserWarning,
                stacklevel=2,
            )

        # Pre-extract patterns once; we iterate them per combination.
        # Rows are 3-tuples ``(pattern, output, is_callable)`` after
        # T-119-followup-numeric-output; only the pattern matters here.
        patterns = [row[0] for row in rows]

        def _row_matches(pattern, vec):
            for p, v in zip(pattern, vec):
                if p == "X":
                    continue
                if bool(p) != bool(v):
                    return False
            return True

        total = 1 << n  # 2 ** n
        covered = 0
        missing: list[tuple[bool, ...]] = []
        # Track row-pairs that overlap on at least one vector. Use a set
        # to dedupe across enumerated vectors, then sort for stability.
        overlapping: set[tuple[int, int]] = set()

        for vec in itertools.product((False, True), repeat=n):
            matching_indices = [
                idx for idx, pat in enumerate(patterns)
                if _row_matches(pat, vec)
            ]
            if matching_indices:
                covered += 1
                if len(matching_indices) > 1:
                    for a in range(len(matching_indices)):
                        for b in range(a + 1, len(matching_indices)):
                            overlapping.add(
                                (matching_indices[a], matching_indices[b])
                            )
            else:
                missing.append(vec)

        overlapping_pairs = sorted(overlapping)

        report = {
            "covered_combinations": covered,
            "total_combinations": total,
            "missing_patterns": missing,
            "overlapping_pairs": overlapping_pairs,
        }

        if strict_completeness and missing:
            raise BlockParameterError(
                message=(
                    f"TruthTable block '{self.name}' is incomplete: "
                    f"{len(missing)} of {total} input combination(s) "
                    f"have no matching row and would fall through to "
                    f"default_output. Missing: {missing!r}."
                ),
                system=self,
                parameter_name="rows",
            )
        if strict_disjointness and overlapping_pairs:
            raise BlockParameterError(
                message=(
                    f"TruthTable block '{self.name}' has overlapping rows: "
                    f"row pair(s) {overlapping_pairs!r} match a common "
                    f"input combination. Earlier-row-wins resolves this at "
                    f"runtime, but the overlap is likely unintentional."
                ),
                system=self,
                parameter_name="rows",
            )

        return report

    # -----------------------------------------------------------------
    # T-119-followup-serialization — explicit dict/JSON round-trip.
    #
    # The runtime ``rows`` form — ``[(tuple of bool|"X", scalar|ndarray)]``
    # — is not directly JSON-friendly: tuples become lists, ``"X"``
    # mixes with bools, and ndarrays must be flattened with their
    # shape preserved. The pair below normalizes that shape into a
    # dict-of-primitives that survives ``json.dumps`` / ``json.loads``
    # untouched, then rebuilds the runtime form on the way back in
    # via the existing ``TruthTable(rows=..., n_inputs=..., ...)``
    # constructor (so the validation path is reused, not duplicated).
    #
    # Pattern encoding: bools become ``"1"``/``"0"`` and the wildcard
    # stays as ``"X"`` so the per-row pattern is a length-``n_inputs``
    # string. This keeps JSON output compact and human-readable while
    # avoiding any ambiguity between ``False`` and ``"X"``.
    #
    # Output encoding: scalars survive as Python ``float``; arrays are
    # encoded as ``{"shape": [...], "data": [...]}`` (flattened to a
    # plain list) so dtype is reconstructed via ``np.asarray`` —
    # consistent with the constructor's existing ``npa.asarray(output)``
    # contract and with T-005 default-float64 (no explicit cast).
    #
    # The ``@parameters(static=...)`` route was rejected because
    # ``declare_static_parameters`` coerces list-typed values to
    # ``np.array`` (see system_base.py), which would clobber the
    # ``[(pattern, output)]`` nested structure on first declaration.
    # Wiring this into the dashboard model serializer is a deeper
    # followup; the dict round-trip here lets callers persist /
    # reload TruthTable rows on their own.
    # -----------------------------------------------------------------

    @staticmethod
    def _encode_pattern(pattern):
        """Encode a runtime pattern tuple to a compact string.

        ``True`` -> ``"1"``, ``False`` -> ``"0"``, ``"X"`` stays ``"X"``.
        """
        chars = []
        for p in pattern:
            if p == "X":
                chars.append("X")
            else:
                chars.append("1" if bool(p) else "0")
        return "".join(chars)

    @staticmethod
    def _decode_pattern(pattern_str, n_inputs):
        """Inverse of :meth:`_encode_pattern`."""
        if not isinstance(pattern_str, str) or len(pattern_str) != n_inputs:
            raise ValueError(
                f"TruthTable pattern string must be length {n_inputs}; "
                f"got {pattern_str!r}."
            )
        decoded = []
        for ch in pattern_str:
            if ch == "X":
                decoded.append("X")
            elif ch == "1":
                decoded.append(True)
            elif ch == "0":
                decoded.append(False)
            else:
                raise ValueError(
                    f"TruthTable pattern character must be '0', '1' or "
                    f"'X'; got {ch!r}."
                )
        return tuple(decoded)

    @staticmethod
    def _encode_output(output):
        """Encode a scalar/ndarray output as a JSON-friendly value.

        Scalars (0-D arrays or plain floats) become ``float``.
        Higher-rank arrays become ``{"shape": [...], "data": [...]}``.
        """
        arr = np.asarray(output)
        if arr.shape == ():
            return float(arr)
        return {
            "shape": list(arr.shape),
            "data": arr.flatten().tolist(),
        }

    @staticmethod
    def _decode_output(encoded):
        """Inverse of :meth:`_encode_output`.

        Returns either a Python ``float`` (for scalars) or a numpy
        ``ndarray`` (for vector/matrix outputs). The constructor will
        ``npa.asarray`` either form, preserving T-005 default-float64.
        """
        if isinstance(encoded, dict):
            data = encoded["data"]
            shape = tuple(encoded["shape"])
            return np.asarray(data).reshape(shape)
        # Scalar fallback — accept int/float/bool/numpy scalar.
        return float(encoded)

    def to_dict(self):
        """Return a JSON-serializable dict describing this TruthTable.

        The dict round-trips through :meth:`from_dict` to a TruthTable
        with identical behaviour for every input combination. Pattern
        wildcards (``"X"``) and vector outputs are preserved.

        Returns:
            dict with keys:

            * ``n_inputs`` (int)
            * ``default_output`` (float | ``{"shape", "data"}``)
            * ``rows`` (list of ``{"pattern": str, "output": ...}``)
        """
        # Callable row outputs (T-119-followup-numeric-output) and
        # callable default_output (T-119-followup-default-callable)
        # cannot be round-tripped through JSON; surface a clear error
        # rather than silently dropping the arithmetic on serialize/load.
        if self._default_output_is_callable:
            raise ValueError(
                f"TruthTable.to_dict(): default_output is a callable "
                f"(got {self._default_output!r}); callable default "
                f"outputs are not JSON-serializable. Replace with a "
                f"constant scalar/array before persisting, or "
                f"reconstruct the TruthTable in code."
            )
        encoded_rows = []
        for row_idx, (pattern, output, is_callable) in enumerate(self._rows):
            if is_callable:
                raise ValueError(
                    f"TruthTable.to_dict(): row {row_idx} has a callable "
                    f"output (got {output!r}); callable row outputs are "
                    f"not JSON-serializable. Replace callable rows with "
                    f"constant outputs before persisting, or reconstruct "
                    f"the TruthTable in code."
                )
            encoded_rows.append({
                "pattern": self._encode_pattern(pattern),
                "output": self._encode_output(output),
            })
        return {
            "n_inputs": int(self._n_inputs),
            "default_output": self._encode_output(self._default_output),
            "rows": encoded_rows,
        }

    @classmethod
    def from_dict(cls, data, **block_kwargs):
        """Reconstruct a TruthTable from the dict produced by :meth:`to_dict`.

        Extra keyword arguments (``name=``, ``system_id=``, ...) are
        forwarded to the underlying ``TruthTable`` constructor, so a
        deserialized block can pick up a fresh name in its target diagram.

        Args:
            data: dict with the keys documented on :meth:`to_dict`.
            **block_kwargs: forwarded to ``TruthTable.__init__`` (e.g.
                ``name``, ``system_id``).

        Returns:
            A new :class:`TruthTable` whose ``rows`` and
            ``default_output`` match the serialized form.
        """
        n_inputs = int(data["n_inputs"])
        default_output = cls._decode_output(data["default_output"])
        rows = [
            (
                cls._decode_pattern(entry["pattern"], n_inputs),
                cls._decode_output(entry["output"]),
            )
            for entry in data["rows"]
        ]
        return cls(
            rows=rows,
            n_inputs=n_inputs,
            default_output=default_output,
            **block_kwargs,
        )

    # -----------------------------------------------------------------
    # T-119-followup-import-from-csv — load a TruthTable from a CSV file.
    #
    # CSV layout (header row required):
    #     in1,in2,in3,output
    #     T,T,T,1.0
    #     T,T,F,0.5
    #     T,F,X,0.25
    #     F,X,X,0.0
    #
    # Input cells accept ``T``/``True``/``1`` for True,
    # ``F``/``False``/``0`` for False, and ``X``/``-``/``*`` (or empty)
    # for the wildcard. Comparison is case-insensitive and surrounding
    # whitespace is stripped. The OUTPUT column (literally named
    # ``output``, case-insensitive) carries a single float per row;
    # if multiple ``output*`` columns are present (e.g. ``output_x``,
    # ``output_y``) they are stacked into a 1-D vector output per row,
    # in the column order they appear in the header. ``default_output``
    # defaults to a zero matching the row-output shape; pass it
    # explicitly via ``**block_kwargs`` to override.
    #
    # Pure-stdlib ``csv`` parsing — no pandas / numpy.loadtxt dep. The
    # existing constructor handles validation (length / dtype) so any
    # malformed row still surfaces a clear ``BlockParameterError`` once
    # the parsed rows reach ``TruthTable.__init__``; the parser itself
    # raises ``ValueError`` for header-level / cell-level mistakes.
    #
    # T-005 default-float64 is preserved: outputs are parsed with
    # ``float(...)`` and forwarded through the constructor's
    # ``npa.asarray`` (no explicit dtype cast).
    # -----------------------------------------------------------------

    # Accepted tokens for input cells (case-insensitive, whitespace-stripped).
    _CSV_TRUE_TOKENS = frozenset({"t", "true", "1"})
    _CSV_FALSE_TOKENS = frozenset({"f", "false", "0"})
    _CSV_WILDCARD_TOKENS = frozenset({"x", "-", "*", ""})

    @staticmethod
    def _parse_csv_input_cell(cell, row_idx, col_name):
        """Parse one input-column cell into ``True``, ``False`` or ``"X"``.

        Raises ``ValueError`` on unrecognised tokens, with row + column
        context so the caller can pinpoint the bad cell.
        """
        token = str(cell).strip().lower()
        if token in TruthTable._CSV_TRUE_TOKENS:
            return True
        if token in TruthTable._CSV_FALSE_TOKENS:
            return False
        if token in TruthTable._CSV_WILDCARD_TOKENS:
            return "X"
        raise ValueError(
            f"TruthTable.from_csv: row {row_idx} column {col_name!r} has "
            f"unrecognised input token {cell!r}; expected one of "
            f"T/True/1, F/False/0, X/-/* (case-insensitive)."
        )

    @classmethod
    def from_csv(cls, path, **block_kwargs):
        """Load a TruthTable from a CSV file.

        The CSV must have a header row whose last column(s) are named
        ``output`` (single scalar output) or any sequence of columns
        whose names start with ``output`` (e.g. ``output_x, output_y``)
        which are stacked into a 1-D vector output per row. All columns
        preceding the first ``output*`` column are treated as input
        columns, in order.

        Input cells accept ``T``/``True``/``1`` (True), ``F``/``False``/
        ``0`` (False), and ``X``/``-``/``*`` or an empty cell
        (wildcard). Matching is case-insensitive and whitespace is
        stripped. Output cells must parse as ``float``.

        Example CSV::

            in1,in2,in3,output
            T,T,T,1.0
            T,T,F,0.5
            T,F,X,0.25
            F,X,X,0.0

        Args:
            path: filesystem path (``str`` or ``os.PathLike``) to a
                readable CSV file.
            **block_kwargs: forwarded to ``TruthTable.__init__`` —
                typically ``name`` / ``system_id``. May also include
                ``default_output`` to override the zero-default that
                this loader picks (a scalar ``0.0`` for single-output
                CSVs, a zeros vector of the right shape for multi-output
                CSVs).

        Returns:
            A :class:`TruthTable` whose ``rows`` mirror the CSV.

        Raises:
            ValueError: if the file is empty, has no header, has no
                ``output`` column, or any row has the wrong number of
                cells / an unparseable input or output cell.
        """
        import csv

        with open(path, "r", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(
                    f"TruthTable.from_csv: file {path!r} is empty; expected "
                    f"a header row followed by data rows."
                )
            # Strip surrounding whitespace from each header cell so callers
            # can author the CSV with comfortable spacing.
            header = [h.strip() for h in header]
            if not header:
                raise ValueError(
                    f"TruthTable.from_csv: file {path!r} has an empty "
                    f"header row."
                )

            # Identify the (contiguous) trailing output column(s). A column
            # is an "output column" if its name (lowercased) starts with
            # ``output``. All input columns must precede the first output
            # column; interleaving is rejected to keep the CSV layout
            # unambiguous.
            output_indices = [
                i for i, name in enumerate(header)
                if name.lower().startswith("output")
            ]
            if not output_indices:
                raise ValueError(
                    f"TruthTable.from_csv: file {path!r} has no 'output' "
                    f"column in header {header!r}; expected at least one "
                    f"column whose name starts with 'output'."
                )
            # Outputs must be contiguous and trail the inputs.
            first_out = output_indices[0]
            expected = list(range(first_out, first_out + len(output_indices)))
            if output_indices != expected:
                raise ValueError(
                    f"TruthTable.from_csv: file {path!r} has non-contiguous "
                    f"output columns at indices {output_indices!r}; all "
                    f"'output*' columns must be the trailing columns."
                )

            input_names = header[:first_out]
            output_names = header[first_out:]
            if not input_names:
                raise ValueError(
                    f"TruthTable.from_csv: file {path!r} has no input "
                    f"columns; at least one input column is required "
                    f"before the 'output' column(s)."
                )
            n_inputs = len(input_names)
            is_vector_output = len(output_names) > 1

            rows: list[tuple[tuple, object]] = []
            for row_idx, raw_row in enumerate(reader):
                # csv.reader yields empty lists for blank lines; skip them
                # so trailing newlines in the file don't blow up parsing.
                if not raw_row or all(c.strip() == "" for c in raw_row):
                    continue
                if len(raw_row) != len(header):
                    raise ValueError(
                        f"TruthTable.from_csv: row {row_idx} has "
                        f"{len(raw_row)} cell(s); expected {len(header)} "
                        f"(header: {header!r}, row: {raw_row!r})."
                    )
                pattern = tuple(
                    cls._parse_csv_input_cell(
                        raw_row[i], row_idx, input_names[i]
                    )
                    for i in range(n_inputs)
                )
                try:
                    output_cells = [
                        float(raw_row[i].strip()) for i in output_indices
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"TruthTable.from_csv: row {row_idx} output cell(s) "
                        f"{[raw_row[i] for i in output_indices]!r} did not "
                        f"parse as float: {exc}."
                    ) from None
                if is_vector_output:
                    output = np.asarray(output_cells)
                else:
                    output = output_cells[0]
                rows.append((pattern, output))

        if not rows:
            raise ValueError(
                f"TruthTable.from_csv: file {path!r} has a header but no "
                f"data rows."
            )

        # Pick a default_output matching the row-output shape unless the
        # caller passed one explicitly via block_kwargs.
        if "default_output" not in block_kwargs:
            if is_vector_output:
                block_kwargs["default_output"] = np.zeros(len(output_names))
            else:
                block_kwargs["default_output"] = 0.0

        return cls(rows=rows, n_inputs=n_inputs, **block_kwargs)

    # -----------------------------------------------------------------
    # T-119-followup-export-to-csv — write a TruthTable to a CSV file.
    #
    # Inverse of ``from_csv``: emits the same header layout
    # ``in1,in2,...,output`` (or ``output_0,output_1,...`` for vector
    # outputs), with input cells written as ``T`` / ``F`` / ``X`` and
    # output cells written as ``float(...)``. Round-tripping
    # ``TruthTable.from_csv(t.to_csv(path))`` reproduces the same rows,
    # ``n_inputs`` and ``default_output``.
    #
    # Callable row outputs (T-119-followup-numeric-output) cannot be
    # serialised — there is no portable representation of a Python
    # closure. ``to_csv`` raises ``ValueError`` in that case.
    #
    # Pure-stdlib ``csv`` writer; T-005 default-float64 is preserved by
    # writing values via ``float(...)``.
    # -----------------------------------------------------------------

    def to_csv(self, path, **csv_kwargs):
        """Write this TruthTable to a CSV file (inverse of ``from_csv``).

        Emits a header row of ``in1,in2,...,output`` (single-output) or
        ``in1,...,output_0,output_1,...`` (vector output), followed by
        one data row per ``rows`` entry. Input cells are written as
        ``T`` / ``F`` / ``X``; output cells are written as
        ``float(...)``.

        Args:
            path: filesystem path (``str`` or ``os.PathLike``) for the
                CSV file to (over)write.
            **csv_kwargs: forwarded to ``csv.writer`` (e.g. ``delimiter``,
                ``quoting``).

        Returns:
            ``path`` (so the caller can chain
            ``TruthTable.from_csv(t.to_csv(p))``).

        Raises:
            ValueError: if any row's output is a callable
                (T-119-followup-numeric-output) or ``default_output``
                is a callable (T-119-followup-default-callable) —
                callables are not representable in CSV.
        """
        import csv

        # Reject callable outputs up front so the file is never created
        # in a half-written state. (Both callable rows and a callable
        # default_output — T-119-followup-default-callable.)
        if self._default_output_is_callable:
            raise ValueError(
                f"TruthTable.to_csv: default_output is a callable; "
                f"callable default outputs cannot be serialised to CSV. "
                f"Replace with a constant scalar/array, or serialise "
                f"via a different format."
            )
        for row_idx, (_pattern, _output, is_callable) in enumerate(self._rows):
            if is_callable:
                raise ValueError(
                    f"TruthTable.to_csv: row {row_idx} has a callable "
                    f"output; callable row outputs cannot be serialised "
                    f"to CSV. Replace with a constant scalar/array, or "
                    f"serialise via a different format."
                )

        # Determine output column layout from the default_output shape
        # (which matches all row outputs by construction — the constructor
        # broadcasts row values against default_output). A 0-d / scalar
        # default produces a single ``output`` column; a 1-D vector
        # produces ``output_0, output_1, ...`` columns. This matches the
        # ``from_csv`` round-trip: a single ``output`` column yields a
        # scalar default, while ``output_*`` columns yield a vector default.
        default = np.asarray(self._default_output)
        if default.ndim == 0:
            output_names = ["output"]
            is_vector_output = False
        elif default.ndim == 1:
            output_names = [f"output_{i}" for i in range(default.shape[0])]
            is_vector_output = True
        else:
            raise ValueError(
                f"TruthTable.to_csv: default_output has shape "
                f"{default.shape!r}; only scalar (0-d) and 1-D vector "
                f"outputs are supported for CSV export."
            )

        input_names = [f"in{i + 1}" for i in range(self._n_inputs)]
        header = input_names + output_names

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh, **csv_kwargs)
            writer.writerow(header)
            for pattern, output, _is_callable in self._rows:
                pattern_cells = []
                for p in pattern:
                    if p == "X":
                        pattern_cells.append("X")
                    elif bool(p):
                        pattern_cells.append("T")
                    else:
                        pattern_cells.append("F")
                out_arr = np.asarray(output)
                if is_vector_output:
                    # Broadcast scalars (rare — constructor stores
                    # ``npa.asarray(output)``) up to the vector width.
                    out_vec = np.broadcast_to(out_arr, (len(output_names),))
                    output_cells = [float(v) for v in out_vec]
                else:
                    # Scalar output column — ``out_arr`` should be 0-d.
                    output_cells = [float(out_arr)]
                writer.writerow(pattern_cells + output_cells)

        return path


# ---------------------------------------------------------------------------
# T-119-followup-builder-api — fluent ``TruthTable.builder()`` API.
#
# Constructing a :class:`TruthTable` from a positional list-of-tuples gets
# unreadable past 3 inputs: ``((True, False, True, False, True), ...)`` is
# a wall of bools with no per-input labels. The builder lets the caller
# name each input slot once (or fall back to ``in1, in2, ...``) and then
# assemble rows with ``.row(in1=True, in2="X", output=1.0)`` — a missing
# keyword defaults to the wildcard ``"X"``. ``.build()`` calls back into
# the existing ``TruthTable(rows=...)`` constructor with no behavioural
# changes, so the non-builder path stays byte-equivalent.
# ---------------------------------------------------------------------------


class TruthTableBuilder:
    """Fluent builder for :class:`TruthTable` rows by named-input keywords.

    Example — a 2-input AND gate:

    .. code-block:: python

        tt = (
            TruthTable.builder(n_inputs=2, default_output=0.0)
            .row(in1=True, in2=True, output=1.0)
            .row(in1=True, in2=False, output=0.0)
            .row(in1=False, in2="X", output=0.0)
            .build()
        )

    Inputs omitted from a ``.row(...)`` call default to the wildcard
    ``"X"``, so partial decision tables are concise. Custom input names
    may be supplied via the ``input_names=`` constructor argument; the
    default names are ``in1, in2, ..., inN``.
    """

    def __init__(
        self,
        n_inputs,
        default_output,
        input_names=None,
        **block_kwargs,
    ):
        n = int(n_inputs)
        if n < 1:
            raise ValueError(
                f"TruthTableBuilder requires n_inputs >= 1; got {n_inputs}."
            )
        if input_names is None:
            self._input_names = tuple(f"in{i + 1}" for i in range(n))
        else:
            names = tuple(input_names)
            if len(names) != n:
                raise ValueError(
                    f"TruthTableBuilder input_names must have length n_inputs="
                    f"{n}; got {len(names)} ({names!r})."
                )
            if len(set(names)) != len(names):
                raise ValueError(
                    f"TruthTableBuilder input_names must be unique; got {names!r}."
                )
            self._input_names = names
        self._n_inputs = n
        self._default_output = default_output
        self._block_kwargs = block_kwargs
        self._rows: list[tuple[tuple, object]] = []

    def row(self, output, **input_assignments):
        """Append a row, named by input keyword.

        ``output`` is the value emitted when the row matches. Each
        keyword in ``input_assignments`` must be one of the configured
        input names; omitted inputs default to the wildcard ``"X"``.
        Returns ``self`` for fluent chaining.
        """
        unknown = set(input_assignments) - set(self._input_names)
        if unknown:
            raise ValueError(
                f"TruthTableBuilder.row(...) got unknown input name(s) "
                f"{sorted(unknown)!r}; expected one of {list(self._input_names)!r}."
            )
        pattern = tuple(
            input_assignments.get(name, "X") for name in self._input_names
        )
        self._rows.append((pattern, output))
        return self

    def build(self):
        """Materialize the accumulated rows into a :class:`TruthTable`."""
        # Forward ``input_names`` so the labels survive on the built block's
        # ``input_ports`` (visible in print_schedule / model JSON / error
        # messages) — T-119-followup-truth-table-named-ports. Skip the
        # default placeholder names (``in1``/``in2``/...) so a user who
        # never set ``input_names=`` keeps the existing anonymous-port
        # behaviour.
        default_input_names = tuple(f"in{i + 1}" for i in range(self._n_inputs))
        if self._input_names == default_input_names:
            input_names = None
        else:
            input_names = self._input_names
        return TruthTable(
            rows=list(self._rows),
            n_inputs=self._n_inputs,
            default_output=self._default_output,
            input_names=input_names,
            **self._block_kwargs,
        )
