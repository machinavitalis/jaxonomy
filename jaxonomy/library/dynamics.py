# SPDX-License-Identifier: MIT

"""Integrators, discrete state, filters, PID, and timing blocks."""

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

# Cross-module reference: PIDController2DOF and PIDDiscrete use
# ``soft_dead_zone`` (defined in :mod:`jaxonomy.library.nonlinearities`)
# inside their smooth-mode error-deadband kernels.
from .nonlinearities import soft_dead_zone


__all__ = [
    "Integrator",
    "TransportDelay",
    "VariableTransportDelay",
    "IntegratorDiscrete",
    "FilterDiscrete",
    "DerivativeDiscrete",
    "DiscreteInitializer",
    "UnitDelay",
    "ZeroOrderHold",
    "PIDDiscrete",
    "LowPassDiscrete",
    "LeadLag",
    "Notch",
    "EdgeDetection",
    "Decimator",
    "PIDController2DOF",
    "RateTransition",
]



class DerivativeDiscrete(LeafSystem):
    """Discrete approximation to the derivative of the input signal w.r.t. time.'

    By default the block uses a simple backward difference approximation:
    ```
    y[k] = (u[k] - u[k-1]) / dt
    ```
    However, the block can also be configured to use a recursive filter for a
    better approximation. In this case the filter coefficients are determined
    by the `filter_type` and `filter_coefficient` parameters. The filter is
    a pair of two-element arrays `a` and `b` and the filter equation is:
    ```
    a0*y[k] + a1*y[k-1] = b0*u[k] + b1*u[k-1]
    ```

    Denoting the `filter_coefficient` parameter by `N`, the following filters are
    available:
    - "none": The default, a simple finite difference approximation.
    - "forward": A filtered forward Euler discretization. The filter is:
        `a = [1, (N*dt - 1)]` and `b = [N, -N]`.
    - "backward": A filtered backward Euler discretization. The filter is:
        `a = [(1 + N*dt), -1]` and `b = [N, -N]`.
    - "bilinear": A filtered bilinear transform discretization. The filter is:
        `a = [(2 + N*dt), (-2 + N*dt)]` and `b = [2*N, -2*N]`.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The approximate derivative of the input signal.

    Parameters:
        dt:
            The time step of the discrete approximation.
        filter_type:
            One of "none", "forward", "backward", or "bilinear". This determines the
            type of filter used to approximate the derivative. The default is "none",
            corresponding to a simple backward difference approximation.
        filter_coefficient:
            The coefficient in the filter (`N` in the equations above). This is only
            used if `filter_type` is not "none". The default is 1.0.
    """

    @parameters(static=["dt", "filter_type", "filter_coefficient"])
    def __init__(self, dt, filter_type="none", filter_coefficient=1.0, dtype=None, **kwargs):
        # T-038a-followup-other-blocks: per-block dtype override; stored
        # outside the @parameters list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self.deriv_output = self.declare_output_port(
            period=dt,
            offset=0.0,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def initialize(self, filter_type="none", filter_coefficient=1.0, dt=None):
        # Determine the coefficients of the filter, if applicable
        # The filter is a pair of two-element array and the filter
        # equation is:
        # a0*y[k] + a1*y[k-1] = b0*u[k] + b1*u[k-1]
        b, a = derivative_filter(
            N=filter_coefficient, dt=self.dt, filter_type=filter_type
        )
        if self._dtype is not None:
            # T-038a-followup-other-blocks: cast filter coefficients to
            # the per-block dtype so the output arithmetic runs at this
            # precision regardless of upstream/global default.
            b = npa.asarray(b).astype(self._dtype)
            a = npa.asarray(a).astype(self._dtype)
        self.filter = (b, a)

        self.declare_discrete_state(default_value=None, as_array=False)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # At t=0 we have no prior information, so the output will
        # be held from its initial value (zero). At t=dt, we have
        # a previous sample, so there is enough information to estimate
        # the derivative.
        self.configure_output_port(
            self.deriv_output,
            self._output,
            period=self.dt,
            offset=self.dt,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _output(self, _time, state, *inputs, **_params):
        # Compute the filtered derivative estimate
        (u,) = inputs
        b, a = self.filter
        y_prev = state.cache[self.deriv_output]
        u_prev = state.discrete_state
        y = (b[0] * u + b[1] * u_prev - a[1] * y_prev) / a[0]
        # T-038a-followup-other-blocks: cast the output to the per-block
        # dtype so cross-dtype upstream connections promote down to the
        # requested precision (best-effort; see ``LookupTable1d`` doc).
        if self._dtype is not None:
            y = npa.asarray(y).astype(self._dtype)
        return y

    def _update(self, time, state, u, **params):
        # Every dt seconds, update the state to the current values
        # T-038a-followup-other-blocks: cast u so the saved state lands
        # the same dtype on every step.
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        return u

    def initialize_static_data(self, context):
        """Infer the size and dtype of the internal states"""
        # If building as part of a subsystem, this may not be fully connected yet.
        # That's fine, as long as it is connected by root context creation time.
        # This probably isn't a good long-term solution:
        #   see https://jaxonomy.atlassian.net/browse/WC-51
        try:
            u = self.eval_input(context)
            self._default_discrete_state = u
            local_context = context[self.system_id].with_discrete_state(u)
            self._default_cache[self.deriv_output] = 0 * u
            local_context = local_context.with_cached_value(self.deriv_output, 0 * u)
            context = context.with_subcontext(self.system_id, local_context)

        except UpstreamEvalError:
            logger.debug(
                "DerivativeDiscrete.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
        return super().initialize_static_data(context)



class DiscreteInitializer(LeafSystem):
    """Discrete Initializer.

    Outputs True for first discrete step, then outputs False there after.
    Or, outputs False for first discrete step, then outputs True there after.
    Practical for cases where it is necessary to have some signal fed initially
    by some initialization, but then after from else in the model.

    Input ports:
        None

    Output ports:
        (0) The dot product of the inputs.
    """

    @parameters(static=["dt"], dynamic=["initial_state"])
    def __init__(self, dt, initial_state=True, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_output_port(self._output)
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, initial_state, dt=None):
        self.declare_discrete_state(default_value=initial_state, dtype=npa.bool_)
        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=npa.inf,
            offset=self.dt,
        )

    def reset_default_values(self, initial_state, dt=None):
        self.configure_discrete_state_default_value(default_value=initial_state)

    def _update(self, time, state, *_inputs, **_params):
        return npa.logical_not(state.discrete_state)

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state



class EdgeDetection(LeafSystem):
    """Output is true only when the input signal changes in a specified way.

    The block updates at a discrete rate, checking the boolean- or binary-valued input
    signal for changes.  Available edge detection modes are:
        - "rising": Output is true when the input changes from False (0) to True (1).
        - "falling": Output is true when the input changes from True (1) to False (0).
        - "either": Output is true when the input changes in either direction

    Input ports:
        (0) The input signal. Must be boolean or binary-valued.

    Output ports:
        (0) The edge detection output signal. Boolean-valued.

    Parameters:
        dt:
            The sampling period of the block.
        edge_detection:
            One of "rising", "falling", or "either". Determines the type of edge
            detection performed by the block.
        initial_state:
            The initial value of the output signal.
    """

    class DiscreteStateType(NamedTuple):
        prev_input: Array
        output: bool

    @parameters(dynamic=["initial_state"], static=["dt", "edge_detection"])
    def __init__(self, dt, edge_detection, initial_state=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.declare_input_port()

        # Declare the periodic update
        self._periodic_update_idx = self.declare_periodic_update()

        # Declare the output port
        self._output_port_idx = self.declare_output_port(
            self._output,
            prerequisites_of_calc=[DependencyTicket.xd, self.input_ports[0].ticket],
            requires_inputs=False,
        )

    def initialize(self, edge_detection, initial_state, dt=None):
        # Determine the type of edge detection
        _detection_funcs = {
            "rising": self._detect_rising,
            "falling": self._detect_falling,
            "either": self._detect_either,
        }
        if edge_detection not in _detection_funcs:
            raise ValueError(
                f"EdgeDetection block {self.name} has invalid selection "
                f"{edge_detection} for 'edge_detection'"
            )
        self._detect_edge = _detection_funcs[edge_detection]

        # The discrete state will contain the previous input value and the output.
        # T-037b: cast `prev_input` to `bool_` so the JSON round-trip can't change
        # its dtype — EdgeDetection is documented as bool-input/bool-output, and
        # without the cast a parameter-typed `initial_state` (e.g. Python `False`)
        # may downcast to float64 across JSON, while runtime updates carry the
        # input port dtype, breaking lax.cond branches in the reset map.
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                prev_input=npa.asarray(initial_state, dtype=npa.bool_),
                output=npa.asarray(False, dtype=npa.bool_),
            ),
            as_array=False,
        )
        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Declare the output port
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xd, self.input_ports[0].ticket],
            requires_inputs=False,
        )

    def reset_default_values(self, initial_state, dt=None):
        # The discrete state will contain the previous input value and the output
        self.configure_discrete_state_default_value(
            default_value=self.DiscreteStateType(
                prev_input=npa.asarray(initial_state, dtype=npa.bool_),
                output=npa.asarray(False, dtype=npa.bool_),
            ),
            as_array=False,
        )

    def _update(self, time, state, *inputs, **params):
        # Update the stored previous state
        # and the output as the result of the edge detection function.
        # T-037b: enforce the bool_ contract on every update so the reset-map
        # NamedTuple has a stable dtype regardless of how the upstream port
        # types its signal (e.g. Step emits 0/1 floats by default).
        (e,) = inputs
        return self.DiscreteStateType(
            prev_input=npa.asarray(e, dtype=npa.bool_),
            output=npa.asarray(
                self._detect_edge(time, state, e, **params), dtype=npa.bool_
            ),
        )

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state.output

    def _detect_rising(self, _time, state, *inputs, **_params):
        (e,) = inputs
        e_prev = state.discrete_state.prev_input
        e_prev = npa.array(e_prev)
        e = npa.array(e)
        not_e_prev = npa.logical_not(e_prev)
        return npa.logical_and(not_e_prev, e)

    def _detect_falling(self, _time, state, *inputs, **_params):
        (e,) = inputs
        e_prev = state.discrete_state.prev_input
        e_prev = npa.array(e_prev)
        e = npa.array(e)
        not_e = npa.logical_not(e)
        return npa.logical_and(e_prev, not_e)

    def _detect_either(self, _time, state, *inputs, **_params):
        (e,) = inputs
        e_prev = state.discrete_state.prev_input
        e_prev = npa.array(e_prev)
        e = npa.array(e)
        not_e_prev = npa.logical_not(e_prev)
        not_e = npa.logical_not(e)
        rising = npa.logical_and(not_e_prev, e)
        falling = npa.logical_and(e_prev, not_e)
        return npa.logical_or(rising, falling)



class FilterDiscrete(LeafSystem):
    """Finite Impulse Response (FIR) filter.

    Similar to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    Note: does not implement the IIR filter.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The filtered signal.

    Parameters:
        b_coefficients:
            Array of filter coefficients.
    """

    @parameters(static=["dt", "b_coefficients"])
    def __init__(
        self,
        dt,
        b_coefficients,
        *args,
        dtype=None,
        **kwargs,
    ):
        # T-038a-followup-other-blocks: per-block dtype override; stored
        # outside the @parameters list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, b_coefficients, dt=None):
        # T-037a: establish a block-level dtype contract. The discrete-state
        # delay-line and the runtime input share their dtype with
        # `b_coefficients`, so JSON round-trip (which can downcast list-typed
        # coefficients) cannot make the reset-map and the declared default
        # disagree. Without this, fresh-built blocks could end up with a
        # float32 default and a float64 update (or vice versa), tripping
        # JAX's lax.cond dtype check on the reloaded diagram.
        # T-038a-followup-other-blocks: an explicit per-block ``dtype=``
        # overrides the inferred coefficient dtype so the entire delay
        # line and feed-forward sum run at the requested precision.
        if self._dtype is not None:
            b_arr = npa.asarray(b_coefficients).astype(self._dtype)
            self._state_dtype = self._dtype
        else:
            b_arr = npa.asarray(b_coefficients)
            self._state_dtype = npa.result_type(b_arr)
        initial_state = npa.zeros(len(b_coefficients) - 1, dtype=self._state_dtype)
        self.declare_discrete_state(default_value=initial_state)

        self.is_feedthrough = bool(b_coefficients[0] != 0)
        self.b_coefficients = b_arr
        prerequisites_of_calc = []
        if self.is_feedthrough:
            prerequisites_of_calc.append(self.input_ports[0].ticket)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=self.dt,
        )

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=self.dt,
            requires_inputs=self.is_feedthrough,
            prerequisites_of_calc=prerequisites_of_calc,
        )

    def _update(self, _time, state, u, **_parameters):
        xd = state.discrete_state
        # T-037a: cast u to the canonical state dtype so the FIFO push lands
        # the same dtype on every step, fresh-built or round-tripped.
        u = npa.asarray(u, dtype=self._state_dtype)
        return npa.concatenate([npa.atleast_1d(u), xd[:-1]])

    def _output(self, time, state, *inputs, **parameters):
        xd = state.discrete_state

        y = npa.sum(npa.dot(self.b_coefficients[1:], xd))

        if self.is_feedthrough:
            (u,) = inputs
            y += u * self.b_coefficients[0]

        return y



class Integrator(LeafSystem):
    """Integrate the input signal in time.

    The Integrator block is the main primitive for building continuous-time
    models.  It is a first-order integrator, implementing the following linear
    time-invariant ordinary differential equation for input values `u` and output
    values `y`:
    ```
        ẋ = u
        y = x
    ```
    where `x` is the state of the integrator.  The integrator is initialized
    with the value of the `initial_state` parameter.

    Options:
        Reset: the integrator can be configured to reset its state on an input
            trigger.  The reset value can be either the initial state of the
            integrator or an external value provided by an input port.
        Limits: the integrator can be configured such that the output and state
            are constrained by upper and lower limits.
        Hold: the integrator can be configured to hold integration based on an
            input trigger.

    The Integrator block is also designed to detect "Zeno" behavior, where the
    reset events happen asymptotically closer together.  This is a pathological
    case that can cause numerical issues in the simulation and should typically be
    avoided by introducing some physically realistic hysteresis into the model.
    However, in the event that Zeno behavior is unavoidable, the integrator will
    enter a "Zeno" state where the output is held constant until the trigger
    changes value to False.  See the "bouncing ball" demo for a Zeno example.

    Input ports:
        (0) The input signal.  Must match the shape and dtype of the initial
            continuous state.
        (1) The reset trigger.  Optional, only if `enable_reset` is True.
        (2) The reset value.  Optional, only if `enable_external_reset` is True.
        (3) The hold trigger. Optional, only if 'enable_hold' is True.

    Output ports:
        (0) The continuous state of the integrator.

    Parameters:
        initial_state:
            The initial value of the integrator state.  Can be any array, or even
            a nested structure of arrays, but the data type should be floating-point.
        enable_reset:
            If True, the integrator will reset its state to the initial value
            when the reset trigger is True.  Adds an additional input port for
            the reset trigger.  This signal should be boolean- or binary-valued.
        enable_external_reset:
            If True, the integrator will reset its state to the value provided
            by the reset value input port when the reset trigger is True. Otherwise,
            the integrator will reset to the initial value.  Adds an additional
            input port for the reset value.  This signal should match the shape
            and dtype of the initial continuous state.
        enable_limits:
            If True, the integrator will constrain its state and output to within
            the upper and lower limits. Either limit may be disbale by setting its
            value to None.
        enable_hold:
            If True, the integrator will hold integration when the hold trigger is
            True.
        reset_on_enter_zeno:
            If True, the integrator will reset its state to the initial value
            when the integrator enters the Zeno state.  This option is ignored unless
            `enable_reset` is True.
        zeno_tolerance:
            The tolerance used to determine if the integrator is in the Zeno state.
            If the time between events is less than this tolerance, then the
            integrator is in the Zeno state.  This option is ignored unless
            `enable_reset` is True.


    Events:
        An event is triggered when the "reset" port changes.

        An event is triggered when the state hit one of the limits.

        An event is triggered when the "hold" port changes.

        Another guard is conditionally active when the integrator is in the Zeno
        state, and is triggered when the "reset" port changes from True to False.
        This event is used to exit the Zeno state and resume normal integration.
    """

    @parameters(
        static=[
            "enable_reset",
            "enable_external_reset",
            "enable_limits",
            "enable_hold",
            "reset_on_enter_zeno",
        ],
        dynamic=["zeno_tolerance", "lower_limit", "upper_limit", "initial_state"],
    )
    def __init__(
        self,
        initial_state,
        enable_reset=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_hold=False,
        enable_external_reset=False,
        zeno_tolerance=1e-6,
        reset_on_enter_zeno=False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.enable_reset = enable_reset
        self.enable_external_reset = enable_external_reset
        self.enable_hold = enable_hold
        self.discrete_state_type = namedtuple(
            "IntegratorDiscreteState", ["zeno", "counter", "tprev"]
        )

        self.xdot_index = self.declare_input_port(name="in_0")

        x0 = npa.array(initial_state, dtype=self.dtype)
        self.dtype = self.dtype if self.dtype is not None else x0.dtype
        self._continuous_state_idx = self.declare_continuous_state(
            default_value=x0,
            ode=self._ode,
            prerequisites_of_calc=[self.input_ports[self.xdot_index].ticket],
        )

        if enable_reset:
            # Boolean input for triggering reset
            self.reset_trigger_index = self.declare_input_port(name="reset_trigger")
            # prerequisites_of_calc.append(
            #     self.input_ports[self.reset_trigger_index].ticket
            # )

            # Declare a custom discrete state to track Zeno behavior
            self.declare_discrete_state(
                default_value=self.discrete_state_type(
                    zeno=False, counter=0, tprev=0.0
                ),
                as_array=False,
            )

            #
            # Declare reset event
            #
            # when reset is triggered, execute the reset map.
            self.declare_zero_crossing(
                guard=self._reset_guard,
                reset_map=self._reset,
                name="reset_on",
                direction="negative_then_non_negative",
            )
            # when reset is deasserted, do not change the state.
            self.declare_zero_crossing(
                guard=self._reset_guard,
                name="reset_off",
                direction="positive_then_non_positive",
            )

            self.declare_zero_crossing(
                guard=self._exit_zeno_guard,
                reset_map=self._exit_zeno,
                name="exit_zeno",
                direction="positive_then_non_positive",
            )

            # Optional: reset value defined by external signal
            if enable_external_reset:
                self.reset_value_index = self.declare_input_port(name="reset_value")
                # prerequisites_of_calc.append(
                #     self.input_ports[self.reset_value_index].ticket
                # )

        if enable_hold:
            # Boolean input for triggering hold assert/deassert
            self.hold_trigger_index = self.declare_input_port(name="hold_trigger")

            def _hold_guard(_time, _state, *inputs, **_params):
                trigger = inputs[self.hold_trigger_index]
                return npa.where(trigger, 1.0, -1.0)

            self.declare_zero_crossing(
                guard=_hold_guard,
                name="hold",
                direction="crosses_zero",
            )

        self._output_port_idx = self.declare_output_port(name="out_0")

    def initialize(
        self,
        initial_state,
        enable_reset=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_hold=False,
        enable_external_reset=False,
        zeno_tolerance=1e-6,
        reset_on_enter_zeno=False,
    ):
        if self.enable_reset != enable_reset:
            raise ValueError("enable_reset cannot be changed after initialization")
        if self.enable_external_reset != enable_external_reset:
            raise ValueError(
                "enable_external_reset cannot be changed after initialization"
            )
        if self.enable_hold != enable_hold:
            raise ValueError("enable_hold cannot be changed after initialization")

        # Default initial condition unless modified in context
        x0 = npa.array(initial_state, dtype=self.dtype)
        self.dtype = self.dtype if self.dtype is not None else x0.dtype

        self.configure_continuous_state(
            self._continuous_state_idx,
            default_value=x0,
            ode=self._ode,
            prerequisites_of_calc=[self.input_ports[self.xdot_index].ticket],
        )

        self.reset_on_enter_zeno = reset_on_enter_zeno

        self.enable_limits = enable_limits
        self.has_lower_limit = lower_limit is not None
        self.has_upper_limit = upper_limit is not None

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
        )

        if enable_limits:
            if lower_limit is not None:

                def _lower_limit_guard(_time, state, *_inputs, **params):
                    return state.continuous_state - params["lower_limit"]

                self.declare_zero_crossing(
                    guard=_lower_limit_guard,
                    name="lower_limit",
                    direction="positive_then_non_positive",
                )

            if upper_limit is not None:

                def _upper_limit_guard(_time, state, *_inputs, **params):
                    return state.continuous_state - params["upper_limit"]

                self.declare_zero_crossing(
                    guard=_upper_limit_guard,
                    name="upper_limit",
                    direction="negative_then_non_negative",
                )

    def reset_default_values(self, **dynamic_parameters):
        x0 = npa.array(dynamic_parameters["initial_state"], dtype=self.dtype)
        self.configure_continuous_state_default_value(
            self._continuous_state_idx,
            default_value=x0,
        )

    def _ode(self, _time, state, *inputs, **params):
        # Normally, just integrate the input signal
        xdot = inputs[self.xdot_index]

        # However, if the reset trigger is high or the integrator is in the Zeno state,
        # then the integrator should hold
        if self.enable_reset:
            trigger = inputs[self.reset_trigger_index]
            in_zeno_state = state.discrete_state.zeno
            xdot = npa.where((trigger | in_zeno_state), npa.zeros_like(xdot), xdot)

        # Additionally, if the limits are enabled, the derivative is set to zero if
        # either limit is presnetly violated.
        if self.enable_limits:
            xc = state.continuous_state

            if self.has_lower_limit:
                llim_violation = npa.logical_and(
                    xdot < 0.0, xc <= params["lower_limit"]
                )
            else:
                llim_violation = False

            if self.has_upper_limit:
                ulim_violation = npa.logical_and(
                    xdot > 0.0, xc >= params["upper_limit"]
                )
            else:
                ulim_violation = False

            xdot = npa.where(
                (llim_violation | ulim_violation), npa.zeros_like(xdot), xdot
            )

        if self.enable_hold:
            hold = inputs[self.hold_trigger_index]
            xdot = npa.where(hold, npa.zeros_like(xdot), xdot)

        return xdot

    def _output(self, _time, state, *_inputs, **params):
        xc = state.continuous_state
        if self.enable_limits:
            lower_limit = params["lower_limit"] if self.has_lower_limit else -np.inf
            upper_limit = params["upper_limit"] if self.has_upper_limit else np.inf
            return npa.clip(xc, lower_limit, upper_limit)

        return xc

    def _reset_guard(self, _time, _state, *inputs, **_params):
        trigger = inputs[self.reset_trigger_index]
        return npa.where(trigger, 1.0, -1.0)

    def _reset(self, time, state, *inputs, **params):
        # If the distance between events is less than the tolerance, then enter the Zeno state.
        dt = time - state.discrete_state.tprev
        zeno = (dt - params["zeno_tolerance"]) <= 0
        tprev = time

        # Handle the reset event as usual
        if self.enable_external_reset:
            xc = inputs[self.reset_value_index]
        else:
            xc = npa.array(params["initial_state"], dtype=self.dtype)

        # Don't reset if entering Zeno state
        new_continuous_state = npa.where(
            zeno & (not self.reset_on_enter_zeno),
            state.continuous_state,
            xc,
        )
        state = state.with_continuous_state(new_continuous_state)

        # Count number of resets (for debugging)
        counter = state.discrete_state.counter + 1

        # Update the discrete state
        xd_plus = self.discrete_state_type(zeno=zeno, counter=counter, tprev=tprev)
        state = state.with_discrete_state(xd_plus)

        logger.debug("Resetting to %s", state)
        return state

    def _exit_zeno_guard(self, _time, _state, *inputs, **_params):
        # This will only be active when in the Zeno state.  It monitors the boolean trigger input
        # and will go from 1.0 (when trigger=True) to 0.0 (when trigger=False)
        trigger = inputs[self.reset_trigger_index]
        return npa.array(trigger, dtype=self.dtype)

    def _exit_zeno(self, _time, state, *_inputs, **_params):
        xd = state.discrete_state._replace(zeno=False)
        return state.with_discrete_state(xd)

    def determine_active_guards(self, root_context):
        # TODO: Update this to use the new zero crossing event system
        # defined in LeafSystem.
        zero_crossing_events = self.zero_crossing_events.mark_all_active()

        if not self.enable_reset:
            return zero_crossing_events

        def _get_reset(events: LeafEventCollection):
            return events.events[0]

        context = root_context[self.system_id]
        in_zeno_state = context.discrete_state.zeno

        reset = cond(
            in_zeno_state,
            lambda e: e.mark_inactive(),
            lambda e: e.mark_active(),
            _get_reset(zero_crossing_events),
        )

        def _get_exit_zeno(events: LeafEventCollection):
            return events.events[1]

        exit_zeno: ZeroCrossingEvent = cond(
            in_zeno_state,
            lambda e: e.mark_active(),
            lambda e: e.mark_inactive(),
            _get_exit_zeno(zero_crossing_events),
        )

        zero_crossing_events = eqx.tree_at(_get_reset, zero_crossing_events, reset)
        zero_crossing_events = eqx.tree_at(
            _get_exit_zeno, zero_crossing_events, exit_zeno
        )

        return zero_crossing_events

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xc = context[self.system_id].continuous_state
        check_state_type(
            self,
            inp_data=u,
            state_data=xc,
            error_collector=error_collector,
        )


class IntegratorDiscrete(LeafSystem):
    """Discrete first-order integrator.

    This block is a discrete-time approximation to the behavior of the Integrator
    block.  It implements the following linear time-invariant difference equation
    for input values `u` and output values `y`:
    ```
        x[k+1] = x[k] + dt * u[k]
        y[k] = x[k]
    ```
    where `x` is the state of the integrator.  The integrator is initialized with
    the value of the `initial_state` parameter.

    Options:
        Reset: the integrator can be configured to reset its state on an input
            trigger.  The reset value can be either the initial state of the
            integrator or an external value provided by an input port.
        Limits: the integrator can be configured such that the output and state
            are constrained by upper and lower limits.
        Hold: the integrator can be configured to hold integration based on an
            input trigger.

    Unlike the continuous-time integrator, the discrete integrator does not detect
    Zeno behavior, since this is not a concern in discrete-time systems.

    Input ports:
        (0) The input signal.  Must match the shape and dtype of the initial
            state.
        (1) The reset trigger.  Optional, only if `enable_reset` is True.
        (2) The reset value.  Optional, only if `enable_external_reset` is True.
        (3) The hold trigger. Optional, only if 'enable_hold' is True.

    Output ports:
        (0) The current state of the integrator.

    Parameters:
        initial_state:
            The initial value of the integrator state.  Can be any array, or even
            a nested structure of arrays, but the data type should be floating-point.
        enable_reset:
            If True, the integrator will reset its state to the initial value
            when the reset trigger is True.  Adds an additional input port for
            the reset trigger.  This signal should be boolean- or binary-valued.
        enable_external_reset:
            If True, the integrator will reset its state to the value provided
            by the reset value input port when the reset trigger is True. Otherwise,
            the integrator will reset to the initial value.  Adds an additional
            input port for the reset value.  This signal should match the shape
            and dtype of the initial continuous state.
        enable_limits:
            If True, the integrator will constrain its state and output to within
            the upper and lower limits. Either limit may be disbale by setting its
            value to None.
        enable_hold:
            If True, the integrator will hold integration when the hold trigger is
            True.
    """

    @parameters(
        static=[
            "dt",
            "enable_reset",
            "enable_external_reset",
            "enable_limits",
            "enable_hold",
        ],
        dynamic=["lower_limit", "upper_limit", "initial_state"],
    )
    def __init__(
        self,
        dt,
        initial_state,
        enable_reset=False,
        enable_hold=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_external_reset=False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self.dtype = dtype

        self.enable_reset = enable_reset
        self.enable_external_reset = enable_external_reset

        self.xdot_index = self.declare_input_port(
            name="in_0"
        )  # One vector-valued input

        self._periodic_update_idx = self.declare_periodic_update()

        if enable_reset:
            self.reset_trigger_index = self.declare_input_port(
                name="reset_trigger"
            )  # Boolean input for triggering reset

            if enable_external_reset:
                self.reset_value_index = self.declare_input_port(
                    name="reset_value"
                )  # Optional reset value

        self.enable_hold = enable_hold
        if enable_hold:
            self.hold_trigger_index = self.declare_input_port(
                name="hold_trigger"
            )  # Boolean input for triggering hold

        self.state_output_index = self.declare_output_port(name="out_0")

    def initialize(
        self,
        initial_state,
        enable_reset=False,
        enable_hold=False,
        enable_limits=False,
        lower_limit=None,
        upper_limit=None,
        enable_external_reset=False,
        dt=None,
    ):
        if self.enable_reset != enable_reset:
            raise ValueError("enable_reset cannot be changed after initialization")
        if self.enable_external_reset != enable_external_reset:
            raise ValueError(
                "enable_external_reset cannot be changed after initialization"
            )
        if self.enable_hold != enable_hold:
            raise ValueError("enable_hold cannot be changed after initialization")

        # Default initial condition unless modified in context
        x0 = npa.array(initial_state, dtype=self.dtype)
        self.dtype = self.dtype if self.dtype is not None else x0.dtype
        self.declare_discrete_state(default_value=x0)
        self.configure_periodic_update(
            self._periodic_update_idx, self._update, period=self.dt, offset=0.0
        )

        # Since the reset is applied to the output port, having this
        # active makes the block feedthrough with respect to related
        # input ports.
        self.is_feedthrough = enable_reset

        self.enable_limits = enable_limits
        self.has_lower_limit = lower_limit is not None
        self.has_upper_limit = upper_limit is not None

        prereqs = [DependencyTicket.xd]
        if enable_reset:
            prereqs.append(self.input_ports[self.reset_trigger_index].ticket)
            if enable_external_reset:
                prereqs.append(self.input_ports[self.reset_value_index].ticket)

        self.configure_output_port(
            self.state_output_index,
            self._output,
            period=self.dt,
            offset=0.0,
            default_value=x0,
            prerequisites_of_calc=prereqs,
        )

    def reset_default_values(self, **dynamic_parameters):
        x0 = npa.array(dynamic_parameters["initial_state"], dtype=self.dtype)
        self.configure_discrete_state_default_value(default_value=x0)
        self.configure_output_port_default_value(self.state_output_index, x0)

    def _reset(self, *inputs, **params):
        if self.enable_external_reset:
            return inputs[self.reset_value_index]
        return npa.array(params["initial_state"], dtype=self.dtype)

    def _apply_reset_and_limits(self, x_new, *inputs, **params):
        # Reset and limits are applied to both the update and outputs
        # so that they respond to the discontinuities simultaneously.

        if self.enable_reset:
            # If the reset is high, then return the reset value
            trigger = inputs[self.reset_trigger_index]
            x_new = npa.where(trigger, self._reset(*inputs, **params), x_new)

        if self.enable_limits:
            lower_limit = params["lower_limit"] if self.has_lower_limit else -npa.inf
            upper_limit = params["upper_limit"] if self.has_upper_limit else npa.inf
            x_new = npa.clip(x_new, lower_limit, upper_limit)

        return x_new

    def _apply_hold(self, x, x_new, *inputs, **_params):
        # Hold is only applied to the update, but not the output

        if self.enable_hold:
            # If the reset is high, then return the reset value
            trigger = inputs[self.hold_trigger_index]
            x_new = npa.where(trigger, x, x_new)

        return x_new

    def _update(self, _time, state, *inputs, **params):
        x = state.discrete_state
        xdot = inputs[self.xdot_index]
        x_new = x + self.dt * xdot
        x_new = self._apply_hold(x, x_new, *inputs, **params)
        x_new = self._apply_reset_and_limits(x_new, *inputs, **params)
        return x_new.astype(x.dtype)

    def _output(self, _time, state, *inputs, **params):
        x = state.discrete_state
        # To ensure that the discontinuities happen simultaneously with
        # the input signal, also apply the reset and limits to the outputs.
        # this makes the block feedthrough.
        y = self._apply_reset_and_limits(x, *inputs, **params)
        return y

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )



class PIDDiscrete(LeafSystem):
    """Discrete-time PID controller.

    This block implements a discrete-time PID controller with a first-order
    approximation to the integrated error and an optional derivative filter.
    The integrated error term is computed as:
    ```
        e_int[k+1] = e_int[k] + e[k] * dt
    ```
    where `e` is the error signal and `dt` is the sampling period.  The derivative
    term is computed in the same way as for the DerivativeDiscrete block, including
    filter options described there.  With the running error integral `e_int` and
    current estimate of the time derivative of the error `e_dot`, the output is:
    ```
        u[k] = kp * e[k] + ki * e_int[k] + kd * e_dot[k]
    ```

    Input ports:
        (0) The error signal.

    Output ports:
        (0) The control signal computed by the PID algorithm.

    Parameters:
        kp:
            The proportional gain (scalar)
        ki:
            The integral gain (scalar)
        kd:
            The derivative gain (scalar)
        dt:
            The sampling period of the block.
        initial_state:
            The initial value of the running error integral.  Default is 0.
        enable_external_initial_state:
            Source for the value used for the integrator initial state. True=from inport,
            False=from the initial_state parameter.
        filter_type:
            One of "none", "forward", "backward", or "bilinear".  Determines the type of
            filter used to estimate the derivative of the error signal.  Default is
            "none".  See DerivativeDiscrete documentation for details.
        filter_coefficient:
            The filter coefficient for the derivative filter.  Default is 1.0.  See
            DerivativeDiscrete documentation for details.
    """

    class DiscreteStateType(NamedTuple):
        integral: Array
        # Recursive filter memory for the derivative estimate
        e_prev: Array
        e_dot_prev: Array

    @parameters(
        static=["dt", "filter_type", "filter_coefficient"],
        dynamic=["kp", "ki", "kd", "initial_state"],
    )
    def __init__(
        self,
        dt,
        kp=1.0,
        ki=1.0,
        kd=1.0,
        initial_state=0.0,
        enable_external_initial_state=False,
        filter_type="none",
        filter_coefficient=1.0,
        dtype=None,
        **kwargs,
    ):
        # T-038a-followup-other-blocks: per-block dtype override; stored
        # outside the @parameters list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(**kwargs)
        self.dt = dt
        self.input_index = self.declare_input_port()

        self.enable_external_initial_state = enable_external_initial_state
        self.initial_state_index = None
        if enable_external_initial_state:
            self.initial_state_index = self.declare_input_port()

        # Declare the periodic update
        self._periodic_update_idx = self.declare_periodic_update()

        # Declare an output port for the control signal
        self.control_output = self.declare_output_port()

        # NOTE:
        # An extra output port for the derivative value is not strictly necessary,
        # but the filtered estimate could be resused elsewhere.  Also, having the
        # previous value saved in the discrete output component of state would allows
        # it to be reused in the recursive filter without recomputing it as part of
        # the update step, a minor efficiency gain.  The tradeoff is an extra event
        # that has to be handled.  This implementation uses one output event and
        # re-does the derivative calculation when a recursive filter is used, but
        # we could always do it the other way in the future.

    def initialize(
        self,
        kp,
        ki,
        kd,
        initial_state,
        filter_type,
        filter_coefficient,
        dt=None,
    ):
        # T-038a-followup-other-blocks: when an explicit per-block dtype
        # is set, cast the discrete-state seed values (integral / e_prev /
        # e_dot_prev) to that dtype so the strict ``check_types`` pass
        # does not see a mismatch between the f32 input signal and an
        # implicit-f64 ``0.0`` default.
        _zero = 0.0
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
            _zero = npa.asarray(0.0).astype(self._dtype)

        # Declare an internal discrete state
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                integral=initial_state,
                e_prev=_zero,
                e_dot_prev=_zero,
            ),
            as_array=False,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Determine the coefficients of the filter, if applicable
        # The filter is a pair of two-element array and the filter
        # equation is:
        # a0*y[k] + a1*y[k-1] = b0*u[k] + b1*u[k-1]
        self.filter_type = filter_type
        b, a = derivative_filter(
            N=filter_coefficient, dt=self.dt, filter_type=filter_type
        )
        if self._dtype is not None:
            # T-038a-followup-other-blocks: cast filter coefficients to
            # the per-block dtype so the derivative arithmetic runs at
            # this precision regardless of upstream/global default.
            b = npa.asarray(b).astype(self._dtype)
            a = npa.asarray(a).astype(self._dtype)
        self.filter = (b, a)

        # T-127-followup-pid-discrete-feedthrough: the output port is
        # already sample-and-hold (``period=self.dt`` causes
        # ``configure_output_port`` to register a periodic update event
        # that writes the cache, and the actual output callback just
        # reads ``state.cache[cache_index]``). Listing the input ticket
        # in ``prerequisites_of_calc`` only serves to flag the output as
        # feedthrough to the algebraic-loop detector — a spurious
        # designation, since between sample boundaries the cache is
        # constant. Dropping the input prereq matches ``UnitDelay`` and
        # un-breaks the canonical
        # ``plant → err → PIDDiscrete → Saturate → plant`` closed-loop
        # pattern (it no longer needs a hand-inserted ``UnitDelay`` to
        # silence ``AlgebraicLoopError``). Same-tick reads still work:
        # the discrete update event collects inputs because
        # ``requires_inputs`` is the default ``True``.
        self.configure_output_port(
            self.control_output,
            self._output,
            period=self.dt,
            offset=0.0,
            default_value=initial_state,
            prerequisites_of_calc=[DependencyTicket.xd],
        )

    def reset_default_values(self, **dynamic_parameters):
        # T-038a-followup-other-blocks: keep the per-block dtype contract
        # consistent across reset_default_values; otherwise the discrete
        # state and output port end up with mixed f32/f64 components.
        initial_state = dynamic_parameters["initial_state"]
        _zero = 0.0
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
            _zero = npa.asarray(0.0).astype(self._dtype)
        self.configure_discrete_state_default_value(
            self.DiscreteStateType(
                integral=initial_state,
                e_prev=_zero,
                e_dot_prev=_zero,
            ),
            as_array=False,
        )
        self.configure_output_port_default_value(
            self.control_output, initial_state
        )

    def _eval_derivative(self, _time, state, *inputs, **_params):
        # Filtered derivative estimate

        e = inputs[self.input_index]  # Error signal from upstream
        e_prev = state.discrete_state.e_prev
        b, a = self.filter  # IIR filter coefficients

        # If the filter is recursive we need to reuse the previous derivative
        # estimate.
        if self.filter_type != "none":
            # Filtered estimate of the time derivative
            e_dot_prev = state.discrete_state.e_dot_prev

            # New estimate of the time derivative of the error signal
            e_dot = (b[0] * e + b[1] * e_prev - a[1] * e_dot_prev) / a[0]

        else:
            # Standard finite difference approximation - no recursion
            e_dot = (b[0] * e + b[1] * e_prev) / a[0]

        return e_dot

    def _update(self, time, state, *inputs, **params):
        e = inputs[self.input_index]  # Error signal from upstream

        # Integrated error signal
        e_int = state.discrete_state.integral

        # Update the derivative estimate if needed for a recursive filter.
        if self.filter_type != "none":
            e_dot = self._eval_derivative(time, state, *inputs, **params)
        else:
            # This state entry isn't used for the finite difference estimator.
            # Can just keep the original value as a placeholder.
            e_dot = state.discrete_state.e_dot_prev

        # Update the internal state
        return self.DiscreteStateType(
            integral=e_int + e * self.dt, e_prev=e, e_dot_prev=e_dot
        )

    def _eval_control(self, e, e_int, e_dot, **params):
        # Calculate the control signal for the PID control law
        kp, ki, kd = params["kp"], params["ki"], params["kd"]
        u = kp * e + ki * e_int + kd * e_dot
        return u

    def _output(self, time, state, *inputs, **params):
        e = inputs[self.input_index]  # Error signal from upstream
        e_int = state.discrete_state.integral
        e_dot = self._eval_derivative(time, state, *inputs, **params)
        u = self._eval_control(e, e_int, e_dot, **params)
        # T-038a-followup-other-blocks: cast the control signal to the
        # per-block dtype so cross-dtype upstream connections promote
        # down to the requested precision (best-effort).
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        return u

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state.integral
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )

    def initialize_static_data(self, context):
        """Set the initial state from the input port, if specified via config"""
        if self.initial_state_index is not None:
            try:
                initial_state = self.eval_input(context, self.initial_state_index)
                default_value = self.DiscreteStateType(
                    integral=initial_state,
                    e_prev=0.0,
                    e_dot_prev=0.0,
                )
                self._default_discrete_state = default_value
                local_context = context[self.system_id].with_discrete_state(
                    default_value
                )
                context = context.with_subcontext(self.system_id, local_context)

            except UpstreamEvalError:
                # The diagram has only been partially created.  Defer the
                # inference of the initial state until the upstream block has been
                # connected.
                logger.debug(
                    "PID_Discrete.initialize_static_data: UpstreamEvalError. "
                    "Continuing without default value initialization."
                )
        return super().initialize_static_data(context)



class UnitDelay(LeafSystem):
    """Hold and delay the input signal by one time step.

    This block implements a "unit delay" with the following difference equation
    for internal state `x`, input signal `u`, and output signal `y`:
    ```
        x[k+1] = u[k]
        y[k] = x[k]
    ```
    Or, in a hybrid context, the discrete update advances the internal state from
    the "pre" or "minus" value x⁻ to the "post" or "plus" value x⁺ at time
    `tₖ = t0 + k * dt`.  According to the discrete update rules, this calculation
    happens using the input values computed during the update step (i.e. by computing
    upstream outputs before evaluating the inputs to this block). That is, the update
    rule can be written `x⁺(tₖ) = f(tₖ, x⁻(tₖ), u(tₖ))`.  The values of `u` are not
    distinguished as "pre" or "post" because there is only one value at the update
    time.  In the difference equation notation, x⁺(tₖ) ≡ x[k+1]`, `x⁻(tₖ) ≡ x[k],
    and u(tₖ) ≡ u[k].  The hybrid update rule is then:
    ```
        x⁺(tₖ) = u(tₖ)
        y(t) = x⁻(tₖ),       between tₖ⁺ and (tₖ+dt)⁻
    ```

    The output signal "seen" by all other blocks on the time interval (tₖ, tₖ+dt)
    is then the value of the input signal u(tₖ) at the previous update. Therefore, all
    downstream discrete-time blocks updating at the same time tₖ will still see the
    value of x⁻(tₖ), the value of the internal state prior to the update.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal delayed by one time step

    Parameters:
        dt:
            The time step of the discrete update.
        initial_state:
            The initial state of the block.  Default is 0.0.

    Note:
        For a *multi-step* / fixed transport latency, do not chain N
        ``UnitDelay`` blocks — use a single :class:`TransportDelay`
        (``delay_seconds = N * dt``), which buffers the history in one block
        and is differentiable through the signal. ``UnitDelay`` is the exact
        one-sample ``z⁻¹`` primitive; :class:`TransportDelay` is the
        parameterized N-sample delay line.
    """

    @parameters(static=["dt"], dynamic=["initial_state"])
    def __init__(self, dt, initial_state, *args, dtype=None, **kwargs):
        # T-038a-followup-other-blocks: per-block dtype override; stored
        # outside the @parameters list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, initial_state, dt=None):
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
        self.configure_periodic_update(
            self._periodic_update_idx, self._update, period=self.dt, offset=self.dt
        )

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=0.0,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
            default_value=initial_state,
        )

    def reset_default_values(self, initial_state, dt=None):
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
        self.declare_discrete_state(default_value=initial_state)
        self.configure_output_port_default_value(self._output_port_idx, initial_state)

    def _update(self, _time, _state, u, **_params):
        # Every dt seconds, update the state to the current input value
        # T-038a-followup-other-blocks: when a per-block dtype is set,
        # cast u so the stored discrete state lands the same dtype on
        # every step, regardless of upstream promotion.
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        return u

    def _output(self, _time, state, **parameters):
        return state.discrete_state

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        inp_data = self.eval_input(context)
        xd = context[self.system_id].discrete_state
        check_state_type(
            self,
            inp_data=inp_data,
            state_data=xd,
            error_collector=error_collector,
        )


class ZeroOrderHold(LeafSystem):
    """Implements a "zero-order hold" A/D conversion.

    https://en.wikipedia.org/wiki/Zero-order_hold

    The block implements a "zero-order hold" with the following difference equation
    for input signal `u` and output signal `y`:
    ```
        y[k] = u[k]
    ```

    The block does not maintain an internal state, but simply holds the value of the
    input signal at the previous update time.  As a result, the block is "feedthrough"
    from its inputs to outputs and cannot be used to break an algebraic loop. The data
    type of this hold value is inferred from upstream blocks.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The "hold" value of the input signal.  If the input signal is continuous,
            then the output will be the value of the input signal at the previous
            update time.  If the input signal is discrete and synchonous with the
            block, the output will be the value of the input signal at the current
            time (i.e. identical to the input signal).

    Parameters:
        dt:
            The time step of the discrete update.
    """

    @parameters(static=["dt"])
    def __init__(self, dt, *args, dtype=None, **kwargs):
        # T-038a-followup-other-blocks: per-block dtype override; stored
        # outside the @parameters list so it does not round-trip through
        # model JSON or get JAX-traced.
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(*args, **kwargs)
        self.dt = dt

        self.declare_input_port()
        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
            prerequisites_of_calc=[self.input_ports[0].ticket, DependencyTicket.xd],
        )

    def _output(self, _time, _state, u, **_params):
        # Every dt seconds, update the state to the current input value
        if self._dtype is not None:
            # T-038a-followup-other-blocks: cast the held value to the
            # per-block dtype.
            u = npa.asarray(u).astype(self._dtype)
        return u


class TransportDelay(LeafSystem):
    """Continuous-time fixed transport (pure) delay.

    Implements ``y(t) = u(t - delay_seconds)`` for ``t >= delay_seconds``;
    for ``t < delay_seconds`` the output is ``initial_output`` (the
    standard "Initial output" semantics).

    The block samples its input on a periodic clock with period ``dt`` and
    stores the most recent ``history_length`` ``(time, value)`` pairs in a
    discrete-state ring buffer. Output evaluation at any continuous time
    ``t`` is a linear interpolation over the buffered ``(time, value)``
    samples at ``t - delay_seconds``.

    The delay is differentiable via the input signal (gradient flows
    through ``npa.interp`` over the values buffer). Differentiability
    w.r.t. the delay value itself is well-defined wherever the buffer
    interpolant is differentiable; the linear interpolant has a kink at
    sample boundaries — pass ``method="pchip"`` on
    :class:`VariableTransportDelay` for a C¹-smooth alternative.

    The buffer is sized statically as ``history_length`` samples. To cover
    a delay of ``delay_seconds`` at sample period ``dt``, you need at
    least ``ceil(delay_seconds / dt) + 1`` slots; we recommend a small
    safety margin. ``history_length`` defaults to
    ``max(8, ceil(delay_seconds / dt) + 4)`` which is sufficient for the
    default constant delay.

    Input ports:
        (0) The input signal ``u(t)``. Scalar or array.

    Output ports:
        (0) The delayed signal ``y(t) = u(t - delay_seconds)`` (or
            ``initial_output`` while ``t < delay_seconds``).

    Parameters:
        dt: Sampling period for the history buffer. Smaller ``dt`` ⇒
            finer interpolation but a larger ring buffer to cover the
            same physical delay.
        delay_seconds: Fixed delay τ in seconds. Dynamic parameter (may
            be tuned via ``with_parameters``); see notes on
            differentiability above.
        initial_output: Output value while ``t < delay_seconds``. Default
            is 0.0.
        history_length: Number of ``(time, value)`` pairs stored. Static
            (compile-time) — required for vmap/JIT-safe buffer sizing.
            If None, defaults to ``max(8, ceil(delay_seconds / dt) + 4)``.

    Notes:
        - For arbitrary array-shaped signals the interpolation is applied
          elementwise via ``jax.vmap`` over the trailing axes.
        - Buffer overflow (delay larger than ``history_length * dt``) is
          not raised; ``npa.interp`` clamps to the boundary, which means
          the oldest stored sample is repeated. This is a documented
          T-107 follow-up; for now, size ``history_length`` generously.
        - ``VariableTransportDelay`` (signal-driven τ) is the natural
          phase-2 extension; it reuses the same ring-buffer machinery
          with the delay sourced from an input port.
    """

    class _BufferState(NamedTuple):
        # Newest sample at index 0; oldest at index -1. Reversed buffers
        # form a monotonically increasing time axis for ``npa.interp``.
        times: "Array"
        values: "Array"

    @parameters(
        static=["dt", "history_length"],
        dynamic=["delay_seconds", "initial_output"],
    )
    def __init__(
        self,
        dt,
        delay_seconds,
        initial_output=0.0,
        history_length=None,
        *args,
        dtype=None,
        **kwargs,
    ):
        # T-038a-followup-mixed-precision-cascade: when no explicit
        # ``dtype=`` kwarg was passed, fall back to the active
        # ``precision_policy`` context manager's dtype, if any.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(*args, **kwargs)

        if dt is None or float(dt) <= 0.0:
            raise BlockParameterError(
                message=(
                    f"TransportDelay block {self.name!r} requires a positive "
                    f"sample period dt; got {dt!r}."
                ),
                parameter_name="dt",
            )
        try:
            delay_hint = float(delay_seconds)
        except (TypeError, ValueError):
            delay_hint = 0.0
        if history_length is None:
            history_length = max(8, int(np.ceil(max(delay_hint, 0.0) / dt)) + 4)
        if int(history_length) < 2:
            raise BlockParameterError(
                message=(
                    f"TransportDelay block {self.name!r} requires "
                    f"history_length >= 2; got {history_length!r}."
                ),
                parameter_name="history_length",
            )

        self.dt = float(dt)
        self.history_length = int(history_length)

        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, dt, delay_seconds, initial_output, history_length=None):
        # ``history_length`` is a static parameter resolved at __init__
        # time; the framework still passes it here for symmetry, but we
        # rely on ``self.history_length`` to size buffers.
        del history_length  # noqa: F841 — silence unused-arg lint

        initial_value = npa.asarray(initial_output)
        if self._dtype is not None:
            initial_value = initial_value.astype(self._dtype)
        self._signal_shape = tuple(initial_value.shape)

        # Pre-fill the times buffer with strictly increasing sentinels
        # below t=0 so that ``npa.interp(t - delay, times[::-1], ...)``
        # clamps to the oldest sample (== ``initial_output``) for any
        # query time before the first real sample has been written. The
        # spacing matches ``dt`` so the reversed time axis stays
        # monotonically increasing.
        sentinel_t0 = -self.dt * (self.history_length + 1) - 1.0
        times = sentinel_t0 + self.dt * np.arange(self.history_length, dtype=np.float64)
        # Newest first: reverse so position 0 is the largest sentinel.
        times = times[::-1].copy()
        if self._dtype is not None:
            times = times.astype(self._dtype)

        values = npa.broadcast_to(
            initial_value, (self.history_length, *self._signal_shape)
        )

        default_state = self._BufferState(
            times=npa.asarray(times), values=npa.asarray(values)
        )
        self.declare_discrete_state(default_value=default_state, as_array=False)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Output is continuous-time (depends on ``time`` and on the
        # discrete-state buffer) — no period; ``requires_inputs=False``
        # because the lookup reads only state + time.
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xd, DependencyTicket.time],
            requires_inputs=False,
            default_value=initial_value,
        )

    def reset_default_values(
        self, dt=None, delay_seconds=None, initial_output=None, history_length=None
    ):
        # Mirror UnitDelay's pattern: rebuild defaults if the dynamic
        # ``initial_output`` changes between calls.
        del dt, delay_seconds, history_length  # noqa: F841

        if initial_output is None:
            return
        initial_value = npa.asarray(initial_output)
        if self._dtype is not None:
            initial_value = initial_value.astype(self._dtype)
        self._signal_shape = tuple(initial_value.shape)

        sentinel_t0 = -self.dt * (self.history_length + 1) - 1.0
        times = sentinel_t0 + self.dt * np.arange(self.history_length, dtype=np.float64)
        times = times[::-1].copy()
        if self._dtype is not None:
            times = times.astype(self._dtype)

        values = npa.broadcast_to(
            initial_value, (self.history_length, *self._signal_shape)
        )
        default_state = self._BufferState(
            times=npa.asarray(times), values=npa.asarray(values)
        )
        self.configure_discrete_state_default_value(
            default_value=default_state, as_array=False
        )
        self.configure_output_port_default_value(
            self._output_port_idx, initial_value
        )

    def _update(self, time, state, *inputs, **_params):
        u = inputs[0]
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        buf = state.discrete_state
        new_times = npa.roll(buf.times, shift=1, axis=0).at[0].set(time)
        new_values = npa.roll(buf.values, shift=1, axis=0).at[0].set(u)
        return self._BufferState(times=new_times, values=new_values)

    def _output(self, time, state, *_inputs, **params):
        buf = state.discrete_state
        # Reverse so the time axis is monotonically increasing for
        # ``npa.interp``: index 0 is oldest, index -1 is newest.
        xp = buf.times[::-1]
        fp = buf.values[::-1]
        delay = params["delay_seconds"]
        initial_output = params["initial_output"]

        query_t = time - delay
        if len(self._signal_shape) == 0:
            y = npa.interp(query_t, xp, fp)
        else:
            # ``npa.interp`` only handles 1-D ``fp``; vmap over the
            # trailing axes of ``values``. ``fp`` has shape
            # ``(history_length, *signal_shape)``; reshape/iterate via
            # ``jax.numpy.apply_along_axis``-style by flattening the
            # trailing dims.
            flat_fp = fp.reshape((self.history_length, -1))
            # Loop over trailing dim count statically (it's a static
            # shape) — JIT-friendly because the loop unrolls.
            ys = [npa.interp(query_t, xp, flat_fp[:, i]) for i in range(flat_fp.shape[1])]
            y = npa.stack(ys).reshape(self._signal_shape)

        # Hold the initial output before the first physical sample is
        # available; ``npa.interp`` would otherwise return the boundary
        # of the (sentinel-filled) buffer, which already equals
        # ``initial_output`` — but explicitly gating on ``time`` keeps
        # the semantics robust to dtype/shape pre-fill quirks.
        y = npa.where(time < delay, npa.asarray(initial_output), y)
        if self._dtype is not None:
            y = npa.asarray(y).astype(self._dtype)
        return y



# T-122 phase 1 end-of-file marker (intentionally distinct from T-123's marker).


# ===========================================================================
# T-123 phase 1 — RateTransition block (companion to T-105 Multirate).
# (Original task ID T-MW-210, renumbered to T-123 in 124c178.)
#
# Explicit user-placed block(s) that bridge two discrete sample rates,
# silencing T-105's ``detect_rate_mismatches`` warning at the connection.
# Phase 1 ships:
#
# * ``Decimator(input_dt, output_dt, ...)``  — fast-to-slow subsampler.
#   Output is sampled-and-held at ``output_dt`` (the slow rate); the
#   input runs at ``input_dt`` (the fast rate).  The block fires its
#   periodic update at ``output_dt`` and latches the most recent input.
# * ``RateTransition(input_dt, output_dt, ...)`` — factory dispatching
#   on ``input_dt`` vs ``output_dt`` to ``ZeroOrderHold`` (slow → fast),
#   ``Decimator`` (fast → slow), or ``UnitDelay`` (same rate).
#
# Both blocks set the marker attribute ``_jaxonomy_rate_transition =
# True`` so :func:`jaxonomy.simulation.rate_groups.detect_rate_mismatches`
# skips the rate-equality check on either end of a connection that
# crosses a rate transition.  This is how a properly-bridged
# ``Slow → RateTransition → Fast`` chain stays silent under the T-105
# Phase 1 detector.
#
# Differentiability: the block is a deterministic resample (sample-and-
# hold or subsample); gradients flow through the input signal exactly
# the same way they flow through a ``UnitDelay`` / ``ZeroOrderHold``.
#
# Deferred follow-ups:
# * Linear-interpolation mode (``mode="linear"``) for downsampling
#   smooth signals.
# * Double-buffer mode (``mode="double_buffer"``) for thread-safety
#   modelling.
# * ``initial_condition`` kwarg currently surfaced for API symmetry with
#   the spec but only honoured in the same-rate (UnitDelay) branch and
#   in ``Decimator``; ``ZeroOrderHold`` does not currently take an
#   initial value.
# ===========================================================================


_DecimatorMeanState = namedtuple(
    "_DecimatorMeanState", ["output", "accumulator", "count"]
)
_DecimatorPeakState = namedtuple(
    "_DecimatorPeakState", ["output", "peak_abs", "peak_value"]
)

_DECIMATOR_MODES = ("pick_last", "mean", "peak")


class Decimator(LeafSystem):
    """Fast-to-slow rate transition: subsample-and-hold at ``output_dt``.

    Implements a discrete-time decimator that samples its input on a
    periodic clock at ``output_dt`` (the slow rate) and holds the value
    until the next slow tick.  The input is assumed to be running at
    ``input_dt`` (the fast rate); the block is agnostic to the actual
    upstream sampling, but the rate-mismatch detector uses this declared
    pair to recognise the bridge.

    Difference equation, with input ``u`` and output ``y``::

        x[k+1] = u[k * (output_dt / input_dt)]
        y(t)   = x[k],   t in (t_k, t_k + output_dt)

    Equivalent to a "Rate Transition (fast to slow)" block in
    its default ZOH-with-decimation mode.

    Input ports:
        (0) The fast-rate input signal.

    Output ports:
        (0) The slow-rate held signal.

    Parameters:
        input_dt: Sample period of the upstream (fast) source. Used for
            documentation / the rate-mismatch detector; the block does
            not actually read the upstream clock.
        output_dt: Sample period of this block's output (slow rate).
            Must satisfy ``output_dt > input_dt`` for "fast to slow"
            semantics; the constructor warns if not.
        initial_state: Initial output value held until the first slow
            tick fires.  Default 0.0.
        mode: How to combine the input samples within each ``output_dt``
            window before emitting at the slow tick.  One of:

            * ``"pick_last"`` (default) — emit the most recent input
              sample at the slow tick (the standard "Rate Transition fast →
              slow" default).  Byte-equivalent to T-123 phase 1.
            * ``"mean"`` — emit the arithmetic mean of every input
              sample observed during the window.  Standard
              anti-aliasing decimation for continuous signals;
              differentiable through the input (linear).
            * ``"peak"`` — emit the input sample with the largest
              absolute value over the window.  Preserves peak
              excursions for envelope tracking / detection.  The
              selector itself is non-differentiable but gradients flow
              through the selected sample's value (a ``np.where``
              branch picks the max-|u| candidate).

    Notes (``mode="mean"`` / ``mode="peak"``):
        The block declares a second periodic update at ``input_dt`` that
        accumulates samples into a running buffer.  At each slow tick
        the emit-and-reset update fires first (declared before the
        accumulator in ``__init__``), reads the buffer, computes the
        window result, and zeroes the buffer for the next window.  At
        simultaneous slow+fast ticks the emit therefore sees the full
        window from the previous interval; the fast tick at the same
        ``t`` then starts the next window with the current input as its
        first sample.
    """

    # T-123: marker attribute used by
    # ``jaxonomy.simulation.rate_groups.detect_rate_mismatches`` to
    # recognise this block as a rate-bridge and skip the mismatch check
    # on adjacent connections.
    _jaxonomy_rate_transition = True

    @parameters(static=["input_dt", "output_dt", "mode"], dynamic=["initial_state"])
    def __init__(
        self,
        input_dt,
        output_dt,
        initial_state=0.0,
        *args,
        mode="pick_last",
        dtype=None,
        **kwargs,
    ):
        # Mirror the per-block dtype / precision-policy plumbing used by
        # ``UnitDelay`` and ``ZeroOrderHold``.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        if mode not in _DECIMATOR_MODES:
            raise ValueError(
                f"Decimator: mode={mode!r} is not one of "
                f"{_DECIMATOR_MODES!r}."
            )
        self._mode = mode
        super().__init__(*args, **kwargs)
        self.input_dt = input_dt
        self.output_dt = output_dt

        if not (output_dt > input_dt):
            warnings.warn(
                f"Decimator block '{self.name}' got output_dt={output_dt} "
                f"<= input_dt={input_dt}; expected output_dt > input_dt "
                f"for fast-to-slow rate transitions. "
                f"Consider RateTransition(input_dt, output_dt) which "
                f"auto-picks the right block.",
                UserWarning,
                stacklevel=3,
            )

        self.declare_input_port()
        # Declaration order matters for the two-phase event scheduler:
        # at simultaneous slow+fast ticks the events fire sequentially
        # in declaration order against the *accumulated* context, so
        # registering the slow emit-and-reset first lets it see the
        # accumulated window before the fast tick begins the next one.
        self._periodic_update_idx = self.declare_periodic_update()
        if self._mode != "pick_last":
            self._fast_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    # ------------------------------------------------------------------
    # initialize / reset for pick_last (legacy single-state path) and
    # for the windowed modes (NamedTuple state + dual periodic update).
    # ------------------------------------------------------------------

    def initialize(self, initial_state, input_dt=None, output_dt=None, mode=None):
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
        if self._mode == "pick_last":
            # Legacy phase-1 path: single periodic update at output_dt,
            # scalar discrete state.  Byte-equivalent to the pre-followup
            # block.
            self.configure_periodic_update(
                self._periodic_update_idx,
                self._update_pick_last,
                period=self.output_dt,
                offset=self.output_dt,
            )
            self.configure_output_port(
                self._output_port_idx,
                self._output_pick_last,
                period=self.output_dt,
                offset=0.0,
                requires_inputs=False,
                prerequisites_of_calc=[DependencyTicket.xd],
                default_value=initial_state,
            )
            return

        # Windowed modes need ``initial_state.dtype`` below to keep the
        # NamedTuple state's accumulator / count / peak fields all on
        # the same dtype.  Outside any ``precision_policy`` context the
        # pick_last branch above leaves ``initial_state`` as the user's
        # raw value (often a Python float); normalise it here so the
        # ``.dtype`` access is safe under T-005 default-float64.
        initial_state = npa.asarray(initial_state)
        # Windowed modes: two periodic updates + NamedTuple state.
        if self._mode == "mean":
            init_state = _DecimatorMeanState(
                output=initial_state,
                accumulator=npa.zeros_like(initial_state),
                count=npa.asarray(0.0, dtype=initial_state.dtype),
            )
            slow_cb, fast_cb = (
                self._update_mean_emit,
                self._update_mean_accumulate,
            )
        else:  # "peak"
            init_state = _DecimatorPeakState(
                output=initial_state,
                peak_abs=npa.full_like(
                    initial_state, npa.asarray(-npa.inf, dtype=initial_state.dtype)
                ),
                peak_value=npa.zeros_like(initial_state),
            )
            slow_cb, fast_cb = (
                self._update_peak_emit,
                self._update_peak_accumulate,
            )
        self.configure_periodic_update(
            self._periodic_update_idx,
            slow_cb,
            period=self.output_dt,
            offset=self.output_dt,
        )
        self.configure_periodic_update(
            self._fast_update_idx,
            fast_cb,
            period=self.input_dt,
            offset=0.0,
        )
        self.configure_output_port(
            self._output_port_idx,
            self._output_windowed,
            period=self.output_dt,
            offset=0.0,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
            default_value=initial_state,
        )

    def reset_default_values(
        self, initial_state, input_dt=None, output_dt=None, mode=None,
    ):
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
        if self._mode == "pick_last":
            self.declare_discrete_state(default_value=initial_state)
            self.configure_output_port_default_value(
                self._output_port_idx, initial_state
            )
            return

        initial_state = npa.asarray(initial_state)
        if self._mode == "mean":
            new_state = _DecimatorMeanState(
                output=initial_state,
                accumulator=npa.zeros_like(initial_state),
                count=npa.asarray(0.0, dtype=initial_state.dtype),
            )
        else:  # "peak"
            new_state = _DecimatorPeakState(
                output=initial_state,
                peak_abs=npa.full_like(
                    initial_state, npa.asarray(-npa.inf, dtype=initial_state.dtype)
                ),
                peak_value=npa.zeros_like(initial_state),
            )
        self.declare_discrete_state(default_value=new_state, as_array=False)
        self.configure_output_port_default_value(
            self._output_port_idx, initial_state
        )

    # ------------------------------------------------------------------
    # pick_last callbacks (T-123 phase 1, unchanged).
    # ------------------------------------------------------------------

    def _update_pick_last(self, _time, _state, u, **_params):
        # Subsample: at every slow tick, latch the current fast input.
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        return u

    def _output_pick_last(self, _time, state, **_parameters):
        return state.discrete_state

    # Legacy aliases — preserved for any external caller that imported
    # the original ``Decimator._update`` / ``Decimator._output`` names
    # (e.g. the phase-1 differentiability test that calls them directly).
    _update = _update_pick_last
    _output = _output_pick_last

    # ------------------------------------------------------------------
    # mean-mode callbacks.
    # ------------------------------------------------------------------

    def _update_mean_emit(self, _time, state, _u, **_params):
        # Slow tick: read the accumulated window, emit its mean, reset.
        xd = state.discrete_state
        # Guard against the degenerate ``count == 0`` case — the
        # ``npa.where`` keeps gradients finite.  In practice the slow
        # tick has offset=output_dt, so count is always ratio>=1 when
        # the slow update fires.
        safe_count = npa.where(
            xd.count > 0, xd.count, npa.asarray(1.0, dtype=xd.count.dtype)
        )
        mean = xd.accumulator / safe_count
        return _DecimatorMeanState(
            output=mean,
            accumulator=npa.zeros_like(xd.accumulator),
            count=npa.asarray(0.0, dtype=xd.count.dtype),
        )

    def _update_mean_accumulate(self, _time, state, u, **_params):
        # Fast tick: add the current input sample to the running sum.
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        xd = state.discrete_state
        return _DecimatorMeanState(
            output=xd.output,
            accumulator=xd.accumulator + u,
            count=xd.count + npa.asarray(1.0, dtype=xd.count.dtype),
        )

    # ------------------------------------------------------------------
    # peak-mode callbacks (max-absolute-value sample within window).
    # ------------------------------------------------------------------

    def _update_peak_emit(self, _time, state, _u, **_params):
        xd = state.discrete_state
        neg_inf = npa.full_like(
            xd.peak_abs, npa.asarray(-npa.inf, dtype=xd.peak_abs.dtype)
        )
        return _DecimatorPeakState(
            output=xd.peak_value,
            peak_abs=neg_inf,
            peak_value=npa.zeros_like(xd.peak_value),
        )

    def _update_peak_accumulate(self, _time, state, u, **_params):
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        xd = state.discrete_state
        abs_u = npa.abs(u)
        is_new_peak = abs_u > xd.peak_abs
        new_peak_abs = npa.where(is_new_peak, abs_u, xd.peak_abs)
        new_peak_value = npa.where(is_new_peak, u, xd.peak_value)
        return _DecimatorPeakState(
            output=xd.output,
            peak_abs=new_peak_abs,
            peak_value=new_peak_value,
        )

    # ------------------------------------------------------------------
    # Shared output for the windowed modes — both store the held value
    # on ``state.discrete_state.output``.
    # ------------------------------------------------------------------

    def _output_windowed(self, _time, state, **_parameters):
        return state.discrete_state.output

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        inp_data = self.eval_input(context)
        xd = context[self.system_id].discrete_state
        if self._mode != "pick_last":
            # Type-check against the ``output`` field — that's the
            # signal flowing out of the block.  The accumulator/count
            # (or peak_abs/peak_value) fields are private bookkeeping.
            xd = xd.output
        check_state_type(
            self,
            inp_data=inp_data,
            state_data=xd,
            error_collector=error_collector,
        )


def RateTransition(
    input_dt,
    output_dt,
    initial_state=0.0,
    *,
    name=None,
    dtype=None,
    **kwargs,
):
    """Auto-pick the right rate-bridging block based on ``input_dt`` vs ``output_dt``.

    * ``input_dt > output_dt`` (slow source → fast destination):
      :class:`ZeroOrderHold` at ``output_dt`` (the fast rate).  The held
      value is whatever the upstream slow block last produced; the ZOH
      re-samples on every fast tick.
    * ``input_dt < output_dt`` (fast source → slow destination):
      :class:`Decimator` at ``output_dt`` (the slow rate).
    * ``input_dt == output_dt`` (same rate): :class:`UnitDelay` at
      ``input_dt`` — a one-step delay so adjacent same-rate blocks can
      still break feedthrough loops.

    Both ZOH and Decimator paths are tagged with the
    ``_jaxonomy_rate_transition`` marker so
    :func:`jaxonomy.simulation.rate_groups.detect_rate_mismatches`
    silences the rate-mismatch warning across the connection.  The
    same-rate (``UnitDelay``) path does not need the marker because it
    cannot itself be a rate mismatch.

    Args:
        input_dt: Sample period of the upstream block.
        output_dt: Sample period of the downstream block.
        initial_state: Initial output value (only meaningful for the
            same-rate ``UnitDelay`` and the fast→slow ``Decimator``
            branches; ``ZeroOrderHold`` ignores it in Phase 1).
        name: Optional block name.
        dtype: Optional per-block dtype (forwarded to the underlying
            block).  See ``T-038a-followup-other-blocks``.
        **kwargs: Forwarded to the underlying block constructor.

    Returns:
        A :class:`LeafSystem` instance: ``ZeroOrderHold``,
        :class:`Decimator`, or :class:`UnitDelay``.
    """
    if input_dt > output_dt:
        # Slow → fast: ZOH at the fast rate.  Tag the instance with the
        # rate-transition marker so ``detect_rate_mismatches`` recognises
        # it as a bridge.
        block = ZeroOrderHold(
            dt=output_dt, name=name, dtype=dtype, **kwargs
        )
        # Instance-level attribute override: the base ``ZeroOrderHold``
        # class is *not* always a rate transition (most users place a
        # ZOH at a single rate, not as a bridge), so we tag the
        # individual instance returned by this factory.
        block._jaxonomy_rate_transition = True
        return block
    if input_dt < output_dt:
        # Fast → slow: explicit Decimator (which sets the marker on the
        # class).
        return Decimator(
            input_dt=input_dt,
            output_dt=output_dt,
            initial_state=initial_state,
            name=name,
            dtype=dtype,
            **kwargs,
        )
    # Same rate: a one-step UnitDelay so users can still break a
    # feedthrough loop at a same-rate boundary.  No bridge marker
    # needed — the rate is identical on both sides.
    return UnitDelay(
        dt=input_dt,
        initial_state=initial_state,
        name=name,
        dtype=dtype,
        **kwargs,
    )


# T-123 phase 1 end-of-file marker.


# ===========================================================================
# T-127 phase 1 — PIDController2DOF (two-degree-of-freedom discrete PID).
# (Original task ID T-MW-304, renumbered to T-127 in 124c178.)
#
# A 2-DOF PID separates the response to setpoint changes from the response
# to disturbances by weighting the setpoint differently in the proportional
# and derivative paths::
#
#     u = Kp * (b*r - y) + Ki * integral(r - y) + Kd * d/dt(c*r - y)
#
# where ``r`` is the setpoint, ``y`` is the measurement, and ``b``, ``c`` in
# ``[0, 1]`` are the proportional and derivative setpoint weights.  With
# ``b = c = 1`` the block reduces to the standard 1-DOF PID (matching the
# existing ``PIDDiscrete`` block); with ``b = c = 0`` it becomes an "I-PD"
# controller, where step changes in the setpoint produce no proportional or
# derivative kick (only the integral term reacts to setpoint changes).
#
# Phase 1 ships a single block that mirrors the ``PIDDiscrete`` discrete-
# state layout (``integral``, ``e_prev``, ``e_dot_prev``) but tracks the
# weighted error signals ``e_p = b*r - y`` (proportional), ``e_i = r - y``
# (integral), ``e_d = c*r - y`` (derivative) separately.  Differentiability
# through ``Kp``, ``Ki``, ``Kd``, ``b``, ``c``, and the derivative-filter
# coefficient is preserved by routing the gains through ``@parameters
# (dynamic=...)``.
#
# Anti-windup (back-calculation, clamping) shipped via
# ``output_min`` / ``output_max`` + ``anti_windup_method`` kwargs.
# External setpoint-weight inputs (gain scheduling) shipped via
# ``b_dynamic`` / ``c_dynamic`` kwargs adding optional runtime input
# ports for ``b`` / ``c``.
# * ``DiscreteIntegrator`` / ``DiscreteDerivative`` standalone blocks (the
#   internals are already factored inside ``IntegratorDiscrete`` /
#   ``DerivativeDiscrete``) — T-127-followup-discrete-integrator-derivative
#   (replaceable kernel kwargs shipped 2026-05-10; ``integrator_method``
#   ∈ {forward_euler, backward_euler, trapezoidal} and
#   ``derivative_method`` ∈ {forward_diff, backward_diff, centered_diff}).
# * Feedforward term — T-127-followup-feedforward (shipped 2026-05-10;
#   ``kff`` scalar adds ``kff * r`` to the PID output, BEFORE saturation.
#   With ``Kff = 1/G(0)`` the controller tracks step changes in ``r``
#   without integrator action — fastest possible step response.
#   Default 0.0 → byte-equivalent to phase 1 / no feedforward).
# * Gain scheduling — T-127-followup-gain-scheduling (shipped 2026-05-11;
#   ``kp_dynamic`` / ``ki_dynamic`` / ``kd_dynamic`` / ``kff_dynamic``
#   kwargs add optional runtime input ports for the four scalar gains,
#   mirroring the ``b_dynamic`` / ``c_dynamic`` pattern from
#   T-127-followup-external-weights.  Combined with the T-114 lookup
#   blocks (``LookupTable1d`` / ``LookupTable2d``), this exposes the
#   "advance from constant-gain PID to operating-point-aware PID"
#   recipe: build one ``LookupTable1d(scheduling_variable, K_table)``
#   per scheduled gain and wire it to the matching PID port.  No lookup
#   tables are baked into the PID class; the wrapper pattern keeps the
#   surface area minimal and the scheduling policy fully user-owned.
#   Default ``*_dynamic=False`` is byte-equivalent to phase 1.
# * Error deadband — T-127-followup-deadband-error (shipped 2026-05-11;
#   ``error_deadband`` / ``error_deadband_mode`` kwargs apply a deadband
#   to the raw error signal before it feeds the P / I / D terms.  In
#   hard mode the controller is silent for ``|e_raw| <= error_deadband``;
#   smooth mode uses :func:`soft_dead_zone` for a differentiable
#   sigmoid-blended gate.  Default ``error_deadband=0.0`` is
#   byte-equivalent to phase 1.
# * Derivative-on-measurement-only — T-127-followup-derivative-on-
#   measurement (shipped 2026-05-13; ``derivative_on_measurement_only``
#   kwarg is sugar for ``c=0`` + ``c_dynamic=False`` and rejects any
#   conflicting explicit ``c`` / ``c_dynamic`` override).  Two factory
#   classmethods, ``PIDController2DOF.standard`` (``b=c=1``) and
#   ``PIDController2DOF.with_derivative_on_measurement`` (``b=1, c=0``),
#   make the two common configurations self-documenting at call sites.
#   Default ``derivative_on_measurement_only=False`` keeps phase 1
#   byte-equivalence.
# * Tracking mode — T-127-followup-tracking-mode (shipped 2026-05-13;
#   ``tracking_enabled`` / ``tracking_gain`` kwargs add bumpless-
#   transfer support.  When ``tracking_enabled=True`` a new input port
#   ``u_ext`` is declared and the integrator is nudged each tick toward
#   the value that would have produced ``u_ext`` (back-calculation with
#   tracking time constant ``Tt = tracking_gain``).  Composes with anti-
#   windup — both effects sum into the integrator update.  Default
#   ``tracking_enabled=False`` is byte-equivalent to phase 1.
# * Integrate-on-regulation-error-only — T-127-followup-i-on-error-only
#   (shipped 2026-05-13; ``integrate_tracking_error`` kwarg gates the
#   back-calculation contribution into the integrator.  ``True``
#   (default) keeps the T-127-followup-tracking-mode behavior:
#   integrator update folds in ``Ki*(r-y)*dt`` AND
#   ``(u_ext - u_pid)/Tt * dt``.  ``False`` restricts the integrator
#   update to ``Ki*(r-y)*dt`` only — the standard "tracking only via a
#   feedforward path" architecture, where ``u_ext`` does NOT pull the
#   integrator.  Both modes remain differentiable through ``jax.grad``.
#   Default ``True`` is byte-equivalent to T-127-followup-tracking-mode.
# * Bumpless mode switch — T-127-followup-bumpless-mode-switch
#   (shipped 2026-05-13; ``tracking_enabled_dynamic=True`` promotes
#   ``tracking_enabled`` from a static flag to a runtime SCALAR INPUT
#   port whose value is interpreted as a boolean (``0`` = OFF / AUTO,
#   non-zero = ON / MANUAL/TRACKING).  When the runtime flag is OFF the
#   tracking-pull branch is suppressed (the integrator behaves like
#   ``tracking_enabled=False`` for that tick); when ON the branch is
#   applied exactly as in T-127-followup-tracking-mode.  Composes with
#   ``tracking_gain`` and ``integrate_tracking_error``.  When this kwarg
#   is True the static ``tracking_enabled`` MUST also be True (the port
#   must exist for the runtime gate to mean anything); the runtime port
#   layout is ``(u_ext, mode_flag)`` — ``mode_flag`` is the LAST input
#   port appended.  Default ``False`` is byte-equivalent to T-127-
#   followup-tracking-mode.
# ===========================================================================


class PIDController2DOF(LeafSystem):
    """Two-degree-of-freedom discrete-time PID controller.

    Implements the standard 2-DOF PID control law::

        u = Kp * (b*r - y) + Ki * integral(r - y) + Kd * d/dt(c*r - y)

    where ``r`` is the setpoint, ``y`` is the measurement, and ``b`` and
    ``c`` are setpoint weights in ``[0, 1]`` for the proportional and
    derivative paths respectively.  With ``b = c = 1`` the block is
    numerically equivalent to the existing :class:`PIDDiscrete` block on
    the error signal ``e = r - y``; with ``b = c = 0`` it becomes an
    "I-PD" controller (only the integral term reacts to setpoint
    changes).

    The integral term uses a forward-Euler approximation::

        e_int[k+1] = e_int[k] + (r[k] - y[k]) * dt

    and the derivative term is computed exactly as for
    :class:`DerivativeDiscrete` / :class:`PIDDiscrete`, including the
    optional first-order filter (``filter_type``, ``filter_coefficient``).

    Input ports:
        (0) Setpoint signal ``r``.
        (1) Measurement signal ``y``.
        Dynamic-port appendices, declared in this deterministic order
        when the corresponding ``*_dynamic`` flag is True:

        (a) ``b`` if ``b_dynamic=True`` (T-127-followup-external-weights)
        (b) ``c`` if ``c_dynamic=True``
        (c) ``kp`` if ``kp_dynamic=True``
            (T-127-followup-gain-scheduling)
        (d) ``ki`` if ``ki_dynamic=True``
        (e) ``kd`` if ``kd_dynamic=True``
        (f) ``kff`` if ``kff_dynamic=True``
        (g) ``u_ext`` if ``tracking_enabled=True``
            (T-127-followup-tracking-mode)
        (h) ``mode_flag`` if ``tracking_enabled_dynamic=True``
            (T-127-followup-bumpless-mode-switch).  Scalar input cast to
            a {0, 1} gate that selects whether the tracking-pull branch
            runs this tick (0 = OFF / AUTO, non-zero = ON / MANUAL /
            TRACKING).  Requires ``tracking_enabled=True`` so the
            ``u_ext`` port exists.

        Port indices skip any flag set to False, so e.g. with only
        ``kp_dynamic=True`` the ``kp`` port is at index 2; with
        ``b_dynamic=True`` + ``kp_dynamic=True`` the ``b`` port is at
        index 2 and ``kp`` is at index 3.  The instance attributes
        ``self.b_index`` / ``self.c_index`` / ``self.kp_index`` /
        ``self.ki_index`` / ``self.kd_index`` / ``self.kff_index`` /
        ``self.u_ext_index`` / ``self.mode_flag_index`` expose the
        resolved positions.

    Output ports:
        (0) The control signal ``u`` computed by the 2-DOF PID law.

    Parameters:
        kp:
            Proportional gain (scalar).
        ki:
            Integral gain (scalar).
        kd:
            Derivative gain (scalar).
        b:
            Setpoint weight for the proportional term, in ``[0, 1]``.
            Default 1.0 (matches 1-DOF PID).  Ignored when
            ``b_dynamic=True`` (the runtime port value is used instead).
        c:
            Setpoint weight for the derivative term, in ``[0, 1]``.
            Default 1.0 (matches 1-DOF PID).  Ignored when
            ``c_dynamic=True``.

            **Recommendation for real-world controllers:** prefer
            ``c = 0`` (a.k.a. "derivative on measurement only").  With
            ``c = 1`` (textbook 2-DOF / 1-DOF PID), a step change in the
            setpoint produces a one-tick spike of magnitude ``Kd / dt``
            in ``d/dt(c*r - y)`` — "derivative kick" — that propagates
            into a brief, often saturation-inducing control transient.
            Setting ``c = 0`` routes the derivative through the
            measurement only (``-d/dt(y)``), so setpoint steps no
            longer kick the derivative term and integral / proportional
            action alone drive the response.  See the
            :meth:`with_derivative_on_measurement` factory and the
            ``derivative_on_measurement_only=True`` convenience kwarg
            below for the standard recipe.
        derivative_on_measurement_only:
            T-127-followup-derivative-on-measurement.  Convenience flag
            equivalent to ``c=0`` + ``c_dynamic=False``.  When True the
            derivative term sees only the measurement (``-d/dt(y)``)
            and is immune to setpoint-step kick; the user-supplied
            ``c`` / ``c_dynamic`` kwargs are rejected with a
            ``ValueError`` to keep the contract unambiguous.  Default
            False (``c=1``) preserves byte-equivalence with phase 1.
            See :meth:`with_derivative_on_measurement` for the matching
            factory function.
        b_dynamic:
            If True, the proportional setpoint weight ``b`` is read from
            an additional input port (index 2) rather than from the
            static ``b`` parameter.  The static ``b`` value is then
            ignored at runtime; the user MUST connect a signal to the
            new port.  Mirrors the ``enable_dynamic_*`` pattern used by
            :class:`Saturate` and :class:`RateLimiter`.  Default False
            (byte-equivalent to phase 1).
        c_dynamic:
            If True, the derivative setpoint weight ``c`` is read from
            an additional input port instead of the static ``c``
            parameter.  The new port lives at index 2 (when only
            ``c_dynamic`` is set) or index 3 (when both ``b_dynamic``
            and ``c_dynamic`` are set).  Default False.
        dt:
            Sampling period of the block.
        initial_state:
            Initial value of the integral.  Default 0.0.
        filter_type:
            One of ``"none"``, ``"forward"``, ``"backward"``, or
            ``"bilinear"`` — derivative-filter mode.  Default ``"none"``.
        filter_coefficient:
            Filter coefficient ``N`` for the derivative filter (the
            conventional "filter coefficient" PID-tuning parameter).  Default 1.0.
        output_min:
            Lower saturation limit on the control output (T-127-followup-
            anti-windup).  ``None`` (default) disables the lower clip.
            When either ``output_min`` or ``output_max`` is set the
            *saturated* control value is published on port (0); the
            unsaturated value also feeds the anti-windup correction.
        output_max:
            Upper saturation limit on the control output.  ``None``
            (default) disables the upper clip.
        anti_windup_method:
            One of ``"none"``, ``"back_calc"``, or ``"clamping"``
            (T-127-followup-anti-windup).  Default ``"none"``.

            * ``"none"`` — no anti-windup; integrator update unchanged.
            * ``"back_calc"`` — back-calculation: subtract
              ``(u_unsat - u_sat) / anti_windup_gain * dt`` from the
              integrator each tick.  Smooth, fully differentiable.
            * ``"clamping"`` — integrator-tracking: only update the
              integral when the controller is *not* pushing further into
              saturation (``u_unsat == u_sat`` OR the error sign points
              away from the saturated direction).

            Anti-windup is a no-op unless ``output_min`` or ``output_max``
            is also set.
        anti_windup_gain:
            Tracking time constant ``Tt`` used by ``"back_calc"``.  Smaller
            values pull the integrator back faster.  Default 1.0.
            Differentiable through ``jax.grad``.
        integrator_method:
            One of ``"forward_euler"`` (default), ``"backward_euler"``, or
            ``"trapezoidal"`` — selects the discretisation used to advance
            the integral term (T-127-followup-discrete-integrator-
            derivative).  Backward-Euler is more stable for stiff loops;
            trapezoidal is more accurate.  Defaults to byte-equivalence
            with phase 1.  Non-default values add an extra ``e_i_prev``
            delay cell to the discrete state.
        derivative_method:
            One of ``"forward_diff"`` (default), ``"backward_diff"``, or
            ``"centered_diff"`` — selects the unfiltered derivative
            kernel.  ``"backward_diff"`` uses past samples only (typical
            for real-time control, introduces a one-tick delay relative
            to ``"forward_diff"``).  ``"centered_diff"`` is less noisy
            but adds one extra delay cell (``e_d_prev_prev``).  Only
            applies when ``filter_type='none'`` — combining a non-default
            ``derivative_method`` with a recursive filter raises a
            ``ValueError`` at construction.
        kff:
            Feedforward gain on the setpoint (T-127-followup-feedforward).
            Adds ``kff * r`` to the PID output *before* saturation /
            anti-windup, so the feedforward term participates in the
            ``output_min`` / ``output_max`` clip and the anti-windup
            comparison between unsaturated and saturated control values.
            For a plant whose steady-state transfer function from ``u``
            to ``y`` is ``G(0)``, choosing ``kff = 1/G(0)`` makes the
            controller track step changes in ``r`` without integrator
            action — fastest possible step response (pair with PID for
            disturbance rejection).  Differentiable through ``jax.grad``.
            Default ``0.0`` → byte-equivalent to phase 1 (no
            feedforward).
        kp_dynamic:
            T-127-followup-gain-scheduling.  If True, the proportional
            gain ``kp`` is read from a runtime input port instead of the
            static ``kp`` parameter.  The new port is appended after
            ``r``, ``y``, and any active ``b`` / ``c`` ports (see "Input
            ports" above).  The static ``kp`` value is then ignored at
            runtime; the user MUST connect a signal to the new port.
            Default False (byte-equivalent to phase 1).
        ki_dynamic:
            Same as ``kp_dynamic`` but for the integral gain ``ki``.
            Default False.
        kd_dynamic:
            Same as ``kp_dynamic`` but for the derivative gain ``kd``.
            Default False.
        kff_dynamic:
            Same as ``kp_dynamic`` but for the feedforward gain ``kff``.
            Default False.
        error_deadband:
            T-127-followup-deadband-error.  Non-negative scalar; when
            positive, the raw error signal ``e_raw`` (the unweighted
            ``r - y`` for the integral path and the weighted ``b*r - y``
            / ``c*r - y`` for the P / D paths) is gated through a
            deadband before it feeds each PID term.  In hard mode the
            gate is ``e = e_raw`` for ``|e_raw| > error_deadband`` and
            ``e = 0`` otherwise; in smooth mode the gate is
            :func:`soft_dead_zone(e_raw, error_deadband,
            error_deadband_sharpness)`.  Default ``0.0`` disables the
            deadband entirely (byte-equivalent to phase 1).
            Differentiable through ``error_deadband`` in smooth mode;
            hard mode is kinked at the boundary.
        error_deadband_mode:
            ``"hard"`` (default) or ``"smooth"``.  Hard mode uses an
            ``npa.where`` gate; smooth mode uses :func:`soft_dead_zone`
            for a sigmoid-blended kernel with finite gradient through
            the band.  Only relevant when ``error_deadband > 0``.
        error_deadband_sharpness:
            Positive scalar controlling the steepness of the smooth
            deadband transition (passed straight to
            :func:`soft_dead_zone`).  Default ``10.0``.  Ignored when
            ``error_deadband_mode='hard'``.
        tracking_enabled:
            T-127-followup-tracking-mode.  If True, declares an extra
            input port ``u_ext`` (appended after every other dynamic
            port) and folds a *tracking-error* term into the integrator
            update::

                e_track = u_ext - u_unsat
                I[k+1] += (e_track / tracking_gain) * dt

            This is the standard "tracking mode" / "manual mode"
            mechanism: while another controller (or an operator) drives
            ``u_ext``, the PID's integrator is pulled toward the value
            that would produce ``u_ext`` so the handoff back to PID-
            driven control is bumpless.  Implementation-wise it is back-
            calculation on the *external* signal — the same kernel as
            ``anti_windup_method="back_calc"`` but using ``u_ext``
            instead of ``u_sat`` as the target.  Both mechanisms compose:
            their corrections sum into the integrator each tick.
            Default False (byte-equivalent to phase 1).
        tracking_gain:
            Tracking time constant ``Tt`` for the tracking-mode back-
            calculation kernel (T-127-followup-tracking-mode).  Smaller
            values pull the integrator toward ``u_ext`` faster.
            Differentiable through ``jax.grad``.  Default 1.0.
        integrate_tracking_error:
            T-127-followup-i-on-error-only.  Selects whether the
            tracking-error term ``(u_ext - u_unsat)/Tt * dt`` is folded
            into the integrator each tick.  When ``True`` (default) the
            integrator update is::

                I[k+1] = I[k] + Ki*(r-y)*dt + (u_ext - u_unsat)/Tt * dt

            preserving the T-127-followup-tracking-mode kernel exactly.
            When ``False`` the integrator only accumulates the
            regulation error::

                I[k+1] = I[k] + Ki*(r-y)*dt

            and ``u_ext`` does NOT pull the integrator at all (the
            tracking signal still flows through any parallel path the
            user has wired up, e.g. a feedforward addition outside the
            block).  Only meaningful when ``tracking_enabled=True``; the
            flag is silently irrelevant otherwise but still round-trips
            through :meth:`to_dict` / :meth:`from_dict`.  Default
            ``True`` is byte-equivalent to T-127-followup-tracking-mode.
        tracking_enabled_dynamic:
            T-127-followup-bumpless-mode-switch.  When ``True``,
            promotes the tracking-mode flag from a static construction-
            time choice to a runtime SCALAR INPUT port appended after
            ``u_ext``.  The port value is treated as a boolean (``0`` =
            OFF / AUTO, non-zero = ON / MANUAL/TRACKING) — multiplying
            the per-tick tracking-pull correction by that gate.  This
            lets a model toggle between PID-driven control and external
            override mid-simulation while keeping the integrator loaded
            with the value that would produce ``u_ext`` (so handoffs in
            either direction remain bumpless).  Requires
            ``tracking_enabled=True``; ``tracking_enabled=False`` plus
            ``tracking_enabled_dynamic=True`` raises ``ValueError`` at
            construction (without the ``u_ext`` port the runtime gate
            has nothing to multiply).  Composes with
            ``integrate_tracking_error`` (the gate multiplies the
            correction term that flag exposes).  Default ``False`` is
            byte-equivalent to T-127-followup-tracking-mode.

    Gain-scheduling recipe:
        Each of ``kp_dynamic``, ``ki_dynamic``, ``kd_dynamic``, and
        ``kff_dynamic`` is the natural plug for a lookup-table-driven
        gain.  The standard wiring uses one :class:`LookupTable1d` (or
        :class:`LookupTable2d` for two scheduling variables) per
        scheduled gain and the same scheduling-variable signal source
        for all of them::

            import jaxonomy
            from jaxonomy.library import (
                Constant, LookupTable1d, PIDController2DOF,
            )

            builder = jaxonomy.DiagramBuilder()
            r = builder.add(Constant(1.0, name="r"))
            y = builder.add(Constant(0.0, name="y"))
            # Scheduling variable -- e.g. engine speed, Mach number,
            # tank level.  Replace with whatever source you have.
            sched = builder.add(Constant(0.5, name="sched"))
            # Schedule kp as a function of the scheduling variable.
            kp_tbl = builder.add(
                LookupTable1d(
                    input_array=[0.0, 0.5, 1.0],
                    output_array=[1.0, 2.0, 4.0],
                    interpolation="linear",
                    name="kp_schedule",
                )
            )
            pid = builder.add(
                PIDController2DOF(
                    dt=0.01, kp_dynamic=True, name="pid"
                )
            )
            builder.connect(r.output_ports[0], pid.input_ports[0])
            builder.connect(y.output_ports[0], pid.input_ports[1])
            builder.connect(sched.output_ports[0], kp_tbl.input_ports[0])
            # kp port is at index 2 when only kp_dynamic is set.
            builder.connect(kp_tbl.output_ports[0], pid.input_ports[2])

        Because every dynamic port is a regular signal port, gradients
        flow through the lookup-table parameters (breakpoints / values)
        AND through the scheduling-variable signal — the standard T-114
        guarantee.  Multiple gains can be scheduled simultaneously;
        flagging ``ki_dynamic`` and ``kd_dynamic`` simply adds two more
        ports for the integral / derivative tables.
    """

    class DiscreteStateType(NamedTuple):
        integral: Array
        # Recursive filter memory for the derivative estimate.  We keep the
        # *weighted* derivative error ``e_d = c*r - y`` (and the previous
        # filtered derivative) so the same recursive-filter formulas as
        # ``PIDDiscrete`` apply.
        e_d_prev: Array
        e_dot_prev: Array

    # T-127-followup-anti-windup — supported anti-windup methods.
    _ANTI_WINDUP_METHODS = ("none", "back_calc", "clamping")

    # T-127-followup-discrete-integrator-derivative — pluggable kernels.
    _INTEGRATOR_METHODS = ("forward_euler", "backward_euler", "trapezoidal")
    _DERIVATIVE_METHODS = ("forward_diff", "backward_diff", "centered_diff")

    # T-127-followup-deadband-error — supported deadband gate modes.
    _ERROR_DEADBAND_MODES = ("hard", "smooth")

    @parameters(
        static=[
            "dt",
            "filter_type",
            "filter_coefficient",
            "anti_windup_method",
            "b_dynamic",
            "c_dynamic",
            "integrator_method",
            "derivative_method",
            "kp_dynamic",
            "ki_dynamic",
            "kd_dynamic",
            "kff_dynamic",
            "error_deadband_mode",
            "tracking_enabled",
            "integrate_tracking_error",
            "tracking_enabled_dynamic",
        ],
        dynamic=[
            "kp",
            "ki",
            "kd",
            "b",
            "c",
            "initial_state",
            "output_min",
            "output_max",
            "anti_windup_gain",
            "kff",
            "error_deadband",
            "error_deadband_sharpness",
            "tracking_gain",
        ],
    )
    def __init__(
        self,
        dt,
        kp=1.0,
        ki=1.0,
        kd=1.0,
        b=1.0,
        c=1.0,
        initial_state=0.0,
        filter_type="none",
        filter_coefficient=1.0,
        output_min=None,
        output_max=None,
        anti_windup_method="none",
        anti_windup_gain=1.0,
        b_dynamic=False,
        c_dynamic=False,
        integrator_method="forward_euler",
        derivative_method="forward_diff",
        kff=0.0,
        kp_dynamic=False,
        ki_dynamic=False,
        kd_dynamic=False,
        kff_dynamic=False,
        error_deadband=0.0,
        error_deadband_mode="hard",
        error_deadband_sharpness=10.0,
        tracking_enabled=False,
        tracking_gain=1.0,
        integrate_tracking_error=True,
        tracking_enabled_dynamic=False,
        dtype=None,
        **kwargs,
    ):
        # T-127-followup-derivative-on-measurement note: the
        # ``derivative_on_measurement_only`` convenience kwarg is
        # intercepted by an outer wrapper installed after the class
        # body (see ``_pid2dof_derivative_on_measurement_wrapper``)
        # so that the rewrite to ``c=0`` / ``c_dynamic=False`` lands
        # in the original ``kwargs`` BEFORE the ``@parameters``
        # decorator captures them.  The wrapper validates conflicting
        # ``c`` / ``c_dynamic`` overrides; by the time this body runs
        # the values are already canonical.
        # Per-block dtype override — same plumbing as PIDDiscrete (T-038a-
        # followup-other-blocks / T-038a-followup-mixed-precision-cascade).
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype

        # T-127-followup-anti-windup — validate and cache static config.
        if anti_windup_method not in self._ANTI_WINDUP_METHODS:
            raise ValueError(
                f"anti_windup_method must be one of "
                f"{self._ANTI_WINDUP_METHODS!r}; got {anti_windup_method!r}"
            )
        self._anti_windup_method = anti_windup_method
        # Anti-windup is "active" only when at least one saturation limit
        # is set.  When both are None the block is byte-equivalent to
        # phase 1 (no clip on the output, no integrator correction),
        # regardless of method string — matching the documented default-
        # off contract.
        self._anti_windup_active = (
            output_min is not None or output_max is not None
        )

        # T-127-followup-discrete-integrator-derivative — validate and
        # cache the integrator/derivative kernel selection.  Defaults
        # ("forward_euler"/"forward_diff") leave the phase 1 update path
        # bit-identical (no extra state cells, same numerical formulas).
        if integrator_method not in self._INTEGRATOR_METHODS:
            raise ValueError(
                f"integrator_method must be one of "
                f"{self._INTEGRATOR_METHODS!r}; got {integrator_method!r}"
            )
        if derivative_method not in self._DERIVATIVE_METHODS:
            raise ValueError(
                f"derivative_method must be one of "
                f"{self._DERIVATIVE_METHODS!r}; got {derivative_method!r}"
            )
        # The derivative-method kwarg only governs the unfiltered finite-
        # difference path; when ``filter_type != "none"`` the recursive
        # filter coefficients (forward/backward/bilinear Euler) own the
        # discretisation.  Reject ambiguous combinations early.
        if derivative_method != "forward_diff" and filter_type != "none":
            raise ValueError(
                "derivative_method only applies when filter_type='none'; "
                f"got derivative_method={derivative_method!r} with "
                f"filter_type={filter_type!r}"
            )
        self._integrator_method = integrator_method
        self._derivative_method = derivative_method

        # T-127-followup-deadband-error — validate the deadband mode at
        # construction so we can dispatch cheaply inside ``_apply_deadband``
        # without re-parsing the string each tick.  Mirrors the
        # ``DeadZone`` block's mode-flag pattern (see T-115-followup-
        # deadzone-backlash).  The half-width and sharpness flow through
        # dynamic parameters so ``jax.grad`` w.r.t. them is finite in
        # smooth mode.
        if error_deadband_mode not in self._ERROR_DEADBAND_MODES:
            raise ValueError(
                f"error_deadband_mode must be one of "
                f"{self._ERROR_DEADBAND_MODES!r}; got {error_deadband_mode!r}"
            )
        self._error_deadband_mode = error_deadband_mode
        # Track whether the deadband is "active" purely for the
        # byte-equivalence fast path: when ``error_deadband == 0.0`` we
        # bypass the gate entirely so phase 1 / earlier-followup tests
        # remain bit-identical.  The flag mirrors ``_anti_windup_active``.
        try:
            _eb = float(error_deadband)
        except (TypeError, ValueError):
            # Tracer / array-like deadband — assume the gate runs.
            _eb = 1.0
        self._error_deadband_active = _eb != 0.0
        # Build a per-instance state tuple: extend the base 3-field layout
        # with optional delay cells when the chosen kernels need them.
        # Defaults keep the state shape identical to phase 1.
        state_fields = ["integral", "e_d_prev", "e_dot_prev"]
        if integrator_method != "forward_euler":
            # backward_euler / trapezoidal both consume the previous
            # integral-error sample, so we have to remember it.
            state_fields.append("e_i_prev")
        if derivative_method == "centered_diff":
            # centered_diff = (e[k+1] - e[k-1]) / (2*dt) needs one extra
            # delay beyond the phase 1 ``e_d_prev``.
            state_fields.append("e_d_prev_prev")
        self._state_fields = tuple(state_fields)
        # Per-instance NamedTuple (overrides the class-level default for
        # this instance).  ``self.DiscreteStateType`` is what
        # initialize() / _update() / reset_default_values() construct.
        from collections import namedtuple as _namedtuple

        self.DiscreteStateType = _namedtuple(
            "PIDController2DOFState", self._state_fields
        )

        super().__init__(**kwargs)
        self.dt = dt
        self.setpoint_index = self.declare_input_port()  # r
        self.measurement_index = self.declare_input_port()  # y

        # T-127-followup-external-weights — optional runtime input ports
        # for the setpoint weights ``b`` (proportional) and ``c``
        # (derivative).  Order matters: when both flags are True the
        # ports are appended in the (b, c) order so the indices are
        # deterministic and can be documented up front.
        self.b_dynamic = bool(b_dynamic)
        self.c_dynamic = bool(c_dynamic)
        if self.b_dynamic:
            self.b_index = self.declare_input_port()
        if self.c_dynamic:
            self.c_index = self.declare_input_port()

        # T-127-followup-gain-scheduling — optional runtime input ports
        # for the four scalar gains (Kp, Ki, Kd, Kff).  Appended after
        # the b/c ports so the indexing remains backward-compatible with
        # T-127-followup-external-weights (existing models that only set
        # b_dynamic / c_dynamic don't see their port indices shift).  The
        # documented order (kp, ki, kd, kff) is the same order in which
        # users typically schedule them.
        self.kp_dynamic = bool(kp_dynamic)
        self.ki_dynamic = bool(ki_dynamic)
        self.kd_dynamic = bool(kd_dynamic)
        self.kff_dynamic = bool(kff_dynamic)
        if self.kp_dynamic:
            self.kp_index = self.declare_input_port()
        if self.ki_dynamic:
            self.ki_index = self.declare_input_port()
        if self.kd_dynamic:
            self.kd_index = self.declare_input_port()
        if self.kff_dynamic:
            self.kff_index = self.declare_input_port()

        # T-127-followup-tracking-mode — optional ``u_ext`` input port for
        # bumpless-transfer / manual-override.  Appended last so older
        # consumers keep their port indices; the user MUST wire the
        # port when ``tracking_enabled=True``.
        self.tracking_enabled = bool(tracking_enabled)
        if self.tracking_enabled:
            self.u_ext_index = self.declare_input_port()

        # T-127-followup-bumpless-mode-switch — optional runtime
        # ``mode_flag`` input port that gates the tracking-pull branch on
        # / off each tick.  Requires ``tracking_enabled=True`` so the
        # ``u_ext`` port exists; otherwise the runtime gate has nothing
        # to multiply.  Appended LAST (after every other dynamic port
        # including ``u_ext``) so older consumers keep their indices.
        self.tracking_enabled_dynamic = bool(tracking_enabled_dynamic)
        if self.tracking_enabled_dynamic and not self.tracking_enabled:
            raise ValueError(
                "PIDController2DOF: tracking_enabled_dynamic=True requires "
                "tracking_enabled=True (the u_ext port must exist for the "
                "runtime mode flag to gate)."
            )
        if self.tracking_enabled_dynamic:
            self.mode_flag_index = self.declare_input_port()

        # T-127-followup-i-on-error-only — controls whether the
        # tracking-error term ``(u_ext - u_unsat)/Tt * dt`` is folded
        # into the integrator update.  When False, ``u_ext`` reaches the
        # integrator ONLY via the regulation error ``r - y`` (i.e. the
        # tracking signal pulls through the parallel back-calculation
        # path is suppressed).  Default True preserves the
        # T-127-followup-tracking-mode behavior, hence byte-equivalence
        # with that followup.  Stored unconditionally so the flag round-
        # trips even when ``tracking_enabled=False`` (in which case it
        # is moot but still serializable).
        self.integrate_tracking_error = bool(integrate_tracking_error)

        # Declare the periodic update.
        self._periodic_update_idx = self.declare_periodic_update()

        # Declare an output port for the control signal.
        self.control_output = self.declare_output_port()

    # T-127-followup-discrete-integrator-derivative -----------------------
    def _make_state(self, *, integral, e_d_prev, e_dot_prev,
                    e_i_prev=None, e_d_prev_prev=None):
        """Build a ``DiscreteStateType`` tuple, supplying optional fields
        only when the configured kernels need them.

        Defaults (forward_euler / forward_diff) skip the optional
        fields, so the returned tuple is shape-identical to phase 1.
        """
        kw = dict(
            integral=integral,
            e_d_prev=e_d_prev,
            e_dot_prev=e_dot_prev,
        )
        if "e_i_prev" in self._state_fields:
            kw["e_i_prev"] = (
                e_i_prev if e_i_prev is not None else npa.zeros_like(integral)
            )
        if "e_d_prev_prev" in self._state_fields:
            kw["e_d_prev_prev"] = (
                e_d_prev_prev if e_d_prev_prev is not None
                else npa.zeros_like(integral)
            )
        return self.DiscreteStateType(**kw)

    def initialize(
        self,
        kp,
        ki,
        kd,
        b,
        c,
        initial_state,
        filter_type,
        filter_coefficient,
        dt=None,
        output_min=None,
        output_max=None,
        anti_windup_method="none",
        anti_windup_gain=1.0,
        b_dynamic=False,
        c_dynamic=False,
        integrator_method="forward_euler",
        derivative_method="forward_diff",
        kff=0.0,
        kp_dynamic=False,
        ki_dynamic=False,
        kd_dynamic=False,
        kff_dynamic=False,
        error_deadband=0.0,
        error_deadband_mode="hard",
        error_deadband_sharpness=10.0,
        tracking_enabled=False,
        tracking_gain=1.0,
        integrate_tracking_error=True,
        tracking_enabled_dynamic=False,
    ):
        # T-127-followup-external-weights — port topology is decided at
        # construction time (mirrors the RateLimiter/SoftRateLimiter
        # contract); reject mid-life flips that would silently shift
        # port indices.
        if bool(b_dynamic) != self.b_dynamic:
            raise ValueError(
                "PIDController2DOF: b_dynamic cannot be changed after "
                "initialization"
            )
        if bool(c_dynamic) != self.c_dynamic:
            raise ValueError(
                "PIDController2DOF: c_dynamic cannot be changed after "
                "initialization"
            )
        # T-127-followup-gain-scheduling — same port-topology lock for
        # the runtime-gain flags.
        if bool(kp_dynamic) != self.kp_dynamic:
            raise ValueError(
                "PIDController2DOF: kp_dynamic cannot be changed after "
                "initialization"
            )
        if bool(ki_dynamic) != self.ki_dynamic:
            raise ValueError(
                "PIDController2DOF: ki_dynamic cannot be changed after "
                "initialization"
            )
        if bool(kd_dynamic) != self.kd_dynamic:
            raise ValueError(
                "PIDController2DOF: kd_dynamic cannot be changed after "
                "initialization"
            )
        if bool(kff_dynamic) != self.kff_dynamic:
            raise ValueError(
                "PIDController2DOF: kff_dynamic cannot be changed after "
                "initialization"
            )
        # T-127-followup-discrete-integrator-derivative — kernel choice
        # is part of the static port topology; reject mid-life flips that
        # would silently change the state-tuple shape.
        if integrator_method != self._integrator_method:
            raise ValueError(
                "PIDController2DOF: integrator_method cannot be changed "
                "after initialization"
            )
        if derivative_method != self._derivative_method:
            raise ValueError(
                "PIDController2DOF: derivative_method cannot be changed "
                "after initialization"
            )
        # T-127-followup-deadband-error — mode is static so the
        # dispatch inside ``_apply_deadband`` can specialise without
        # re-parsing each tick.
        if error_deadband_mode != self._error_deadband_mode:
            raise ValueError(
                "PIDController2DOF: error_deadband_mode cannot be "
                "changed after initialization"
            )
        # T-127-followup-tracking-mode — port-topology lock for the
        # tracking-mode flag (same contract as the other ``*_dynamic`` /
        # ``*_enabled`` flags above).
        if bool(tracking_enabled) != self.tracking_enabled:
            raise ValueError(
                "PIDController2DOF: tracking_enabled cannot be changed "
                "after initialization"
            )
        # T-127-followup-i-on-error-only — gate the tracking-integrator
        # contribution at construction time so the update path can
        # specialise without re-parsing each tick.
        if bool(integrate_tracking_error) != self.integrate_tracking_error:
            raise ValueError(
                "PIDController2DOF: integrate_tracking_error cannot be "
                "changed after initialization"
            )
        # T-127-followup-bumpless-mode-switch — port-topology lock for
        # the runtime ``mode_flag`` port; same contract as the other
        # ``*_dynamic`` / ``*_enabled`` flags above.
        if bool(tracking_enabled_dynamic) != self.tracking_enabled_dynamic:
            raise ValueError(
                "PIDController2DOF: tracking_enabled_dynamic cannot be "
                "changed after initialization"
            )
        # Cast initial state and zero seeds to the per-block dtype if set.
        _zero = 0.0
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
            _zero = npa.asarray(0.0).astype(self._dtype)

        self.declare_discrete_state(
            default_value=self._make_state(
                integral=initial_state,
                e_d_prev=_zero,
                e_dot_prev=_zero,
                e_i_prev=_zero,
                e_d_prev_prev=_zero,
            ),
            as_array=False,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Derivative-filter coefficients (b0,b1) / (a0,a1) — same helper as
        # PIDDiscrete / DerivativeDiscrete.
        self.filter_type = filter_type
        b_coef, a_coef = derivative_filter(
            N=filter_coefficient, dt=self.dt, filter_type=filter_type
        )
        if self._dtype is not None:
            b_coef = npa.asarray(b_coef).astype(self._dtype)
            a_coef = npa.asarray(a_coef).astype(self._dtype)
        self.filter = (b_coef, a_coef)

        # T-127-followup-external-weights / T-127-followup-gain-
        # scheduling — include the optional dynamic-input ports in the
        # output-prerequisite set so the scheduler eagerly evaluates
        # them before ``_output``.  When a flag is False the matching
        # port simply does not exist.
        prereqs = [
            DependencyTicket.xd,
            self.input_ports[0].ticket,
            self.input_ports[1].ticket,
        ]
        if self.b_dynamic:
            prereqs.append(self.input_ports[self.b_index].ticket)
        if self.c_dynamic:
            prereqs.append(self.input_ports[self.c_index].ticket)
        if self.kp_dynamic:
            prereqs.append(self.input_ports[self.kp_index].ticket)
        if self.ki_dynamic:
            prereqs.append(self.input_ports[self.ki_index].ticket)
        if self.kd_dynamic:
            prereqs.append(self.input_ports[self.kd_index].ticket)
        if self.kff_dynamic:
            prereqs.append(self.input_ports[self.kff_index].ticket)
        # T-127-followup-tracking-mode — ``u_ext`` only feeds the
        # integrator update (``_update``), not the output value, so we
        # only need it in the update path's prerequisite set.  The
        # scheduler resolves prerequisites of the periodic update via
        # its own input-port tickets — we list it here for parity with
        # the other dynamic ports so a tracer always sees the same
        # signature.
        if self.tracking_enabled:
            prereqs.append(self.input_ports[self.u_ext_index].ticket)
        # T-127-followup-bumpless-mode-switch — runtime mode-flag port
        # only feeds ``_update`` (gates the tracking-pull correction),
        # but list it in the prereq set for tracer parity with the
        # other dynamic ports.
        if self.tracking_enabled_dynamic:
            prereqs.append(self.input_ports[self.mode_flag_index].ticket)

        self.configure_output_port(
            self.control_output,
            self._output,
            period=self.dt,
            offset=0.0,
            default_value=initial_state,
            prerequisites_of_calc=prereqs,
        )

    def reset_default_values(self, **dynamic_parameters):
        initial_state = dynamic_parameters["initial_state"]
        _zero = 0.0
        if self._dtype is not None:
            initial_state = npa.asarray(initial_state).astype(self._dtype)
            _zero = npa.asarray(0.0).astype(self._dtype)
        self.configure_discrete_state_default_value(
            self._make_state(
                integral=initial_state,
                e_d_prev=_zero,
                e_dot_prev=_zero,
                e_i_prev=_zero,
                e_d_prev_prev=_zero,
            ),
            as_array=False,
        )
        self.configure_output_port_default_value(
            self.control_output, initial_state
        )

    # T-127-followup-external-weights ------------------------------------
    def _resolve_b(self, inputs, params):
        """Return ``b`` from the runtime port when ``b_dynamic`` is set,
        otherwise from the static dynamic-parameter ``b``."""
        if self.b_dynamic:
            return inputs[self.b_index]
        return params["b"]

    def _resolve_c(self, inputs, params):
        """Return ``c`` from the runtime port when ``c_dynamic`` is set,
        otherwise from the static dynamic-parameter ``c``."""
        if self.c_dynamic:
            return inputs[self.c_index]
        return params["c"]

    # T-127-followup-gain-scheduling -------------------------------------
    def _resolve_kp(self, inputs, params):
        """Return ``kp`` from the runtime port when ``kp_dynamic`` is
        set, otherwise from the static dynamic-parameter ``kp``."""
        if self.kp_dynamic:
            return inputs[self.kp_index]
        return params["kp"]

    def _resolve_ki(self, inputs, params):
        """Return ``ki`` from the runtime port when ``ki_dynamic`` is
        set, otherwise from the static dynamic-parameter ``ki``."""
        if self.ki_dynamic:
            return inputs[self.ki_index]
        return params["ki"]

    def _resolve_kd(self, inputs, params):
        """Return ``kd`` from the runtime port when ``kd_dynamic`` is
        set, otherwise from the static dynamic-parameter ``kd``."""
        if self.kd_dynamic:
            return inputs[self.kd_index]
        return params["kd"]

    def _resolve_kff(self, inputs, params):
        """Return ``kff`` from the runtime port when ``kff_dynamic`` is
        set, otherwise from the static dynamic-parameter ``kff`` (which
        defaults to 0.0 when feedforward is disabled)."""
        if self.kff_dynamic:
            return inputs[self.kff_index]
        return params.get("kff", 0.0)

    # T-127-followup-tracking-mode --------------------------------------
    def _resolve_u_ext(self, inputs):
        """Return the external tracking signal ``u_ext`` (the value the
        integrator is nudged toward when ``tracking_enabled=True``).

        Only callable when ``self.tracking_enabled`` is True; callers
        gate on the flag before invoking.
        """
        return inputs[self.u_ext_index]

    # T-127-followup-deadband-error -------------------------------------
    def _apply_deadband(self, e_raw, params):
        """Gate ``e_raw`` through the configured deadband.

        Returns ``e_raw`` unchanged when the deadband is inactive
        (``error_deadband == 0.0`` at construction time → byte-equivalent
        to phase 1).  In ``"hard"`` mode the gate is the standard
        ``where(|e_raw| > half_range, e_raw, 0)``; in ``"smooth"`` mode
        the gate uses :func:`soft_dead_zone`, which is differentiable
        through the band (this is the only place the deadband-followup
        differs from the inline phase 1 arithmetic).
        """
        if not self._error_deadband_active:
            return e_raw
        half_range = params.get("error_deadband", 0.0)
        if self._error_deadband_mode == "smooth":
            sharpness = params.get("error_deadband_sharpness", 10.0)
            return soft_dead_zone(e_raw, half_range, sharpness)
        # Hard gate — npa.where is kinked but the per-branch gradient is
        # finite, matching the DeadZone(mode="hard") semantics.
        return npa.where(npa.abs(e_raw) > half_range, e_raw, 0.0)

    def _eval_derivative(self, _time, state, *inputs, **params):
        # Filtered estimate of the time derivative of the *weighted*
        # derivative error e_d = c*r - y.
        c = self._resolve_c(inputs, params)
        r = inputs[self.setpoint_index]
        y = inputs[self.measurement_index]
        # T-127-followup-deadband-error — gate the raw derivative error
        # before it feeds the recursive filter / finite-difference
        # kernel.  ``e_d_prev`` was written by the previous tick's
        # ``_update``, which already applied the same gate, so the
        # filter state is consistent across ticks.
        e_d = self._apply_deadband(c * r - y, params)
        e_d_prev = state.discrete_state.e_d_prev
        b_coef, a_coef = self.filter

        if self.filter_type != "none":
            e_dot_prev = state.discrete_state.e_dot_prev
            e_dot = (
                b_coef[0] * e_d + b_coef[1] * e_d_prev - a_coef[1] * e_dot_prev
            ) / a_coef[0]
        else:
            # T-127-followup-discrete-integrator-derivative — kernel
            # dispatch.  ``forward_diff`` reproduces phase 1 exactly via
            # the (b=[1,-1], a=[dt,0]) coefficients.  The other kernels
            # ignore the filter coefficients and use explicit finite-
            # difference formulas over the (e_d, e_d_prev, e_d_prev_prev)
            # delay line.
            if self._derivative_method == "forward_diff":
                # Phase 1 path — uses the precomputed filter coefficients.
                e_dot = (b_coef[0] * e_d + b_coef[1] * e_d_prev) / a_coef[0]
            elif self._derivative_method == "backward_diff":
                # D[k] = (e[k] - e[k-1]) / dt — uses past data only, so
                # the *output* lags by one tick: read the previous tick's
                # finite difference from ``e_dot_prev`` (set during the
                # last _update).  This is what gives the documented
                # one-tick transient delay relative to forward_diff.
                e_dot = state.discrete_state.e_dot_prev
            else:  # "centered_diff"
                # D[k] = (e[k+1] - e[k-1]) / (2*dt) — uses the extra
                # delay cell stored in ``e_d_prev_prev``.
                e_d_prev_prev = state.discrete_state.e_d_prev_prev
                e_dot = (e_d - e_d_prev_prev) / (2.0 * self.dt)

        return e_dot

    # T-127-followup-anti-windup ------------------------------------------
    def _saturate(self, u, **params):
        """Clip ``u`` to ``[output_min, output_max]`` when configured.

        Returns ``u`` unchanged when no saturation limits were declared
        (preserves byte-equivalence with the phase 1 default path).
        """
        if not self._anti_windup_active:
            return u
        u_sat = u
        # Dynamic params are only present when declared at construction
        # time (None-valued ones get skipped by the @parameters decorator).
        u_max = params.get("output_max", None)
        u_min = params.get("output_min", None)
        if u_max is not None:
            u_sat = npa.minimum(u_sat, u_max)
        if u_min is not None:
            u_sat = npa.maximum(u_sat, u_min)
        return u_sat

    def _update(self, time, state, *inputs, **params):
        b = self._resolve_b(inputs, params)
        c = self._resolve_c(inputs, params)
        r = inputs[self.setpoint_index]
        y = inputs[self.measurement_index]
        # T-127-followup-deadband-error — gate every error signal that
        # feeds a downstream PID term.  Default error_deadband=0.0
        # leaves all three identities unchanged (byte-equivalent to
        # phase 1).  Each branch (P / I / D) goes through its own gate
        # so the weighted-setpoint semantics of ``b`` / ``c`` are
        # preserved (the gate is applied AFTER the weighting, on the
        # same composite signal the PID terms actually see).
        e_p = self._apply_deadband(b * r - y, params)
        e_i = self._apply_deadband(r - y, params)
        e_d = self._apply_deadband(c * r - y, params)

        e_int = state.discrete_state.integral

        # T-127-followup-discrete-integrator-derivative — when using the
        # ``backward_diff`` derivative kernel, the *next* stored
        # ``e_dot_prev`` is the finite difference computed from the
        # current samples (so the next tick's _output reads it as the
        # one-tick-delayed derivative).  For forward_diff/centered_diff
        # the field is left at zero (forward_diff doesn't read it,
        # centered_diff reads ``e_d_prev_prev`` instead).
        if self.filter_type != "none":
            # Recursive filters need e_dot_prev for the IIR update;
            # compute it via the standard filtered-derivative path.
            e_dot_next = self._eval_derivative(
                time, state, *inputs, **params
            )
        elif self._derivative_method == "backward_diff":
            # Store the just-computed (e_d - e_d_prev)/dt so the next
            # tick's _output sees it as the lagged derivative.
            e_d_prev = state.discrete_state.e_d_prev
            e_dot_next = (e_d - e_d_prev) / self.dt
        else:
            # forward_diff / centered_diff: e_dot_prev is unused on the
            # output path, keep it at the previous value (matches phase 1
            # placeholder semantics).
            e_dot_next = state.discrete_state.e_dot_prev

        # Integrator kernel dispatch.
        # forward_euler (phase 1): I[k+1] = I[k] + e[k] * dt — uses the
        # most-recent error sample.  backward_euler / trapezoidal pull
        # in the previously stored sample to differentiate them by the
        # documented one-step shift on a ramp.
        if self._integrator_method == "forward_euler":
            integral_next = e_int + e_i * self.dt
        else:
            e_i_prev = state.discrete_state.e_i_prev
            if self._integrator_method == "backward_euler":
                # I[k+1] = I[k] + e[k+1] * dt — labelled per the spec; the
                # available "next" sample at tick k is the *next-tick*
                # update's input.  We approximate by integrating the
                # previous sample, which produces the documented one-tick
                # lag relative to forward_euler.
                integral_next = e_int + e_i_prev * self.dt
            else:  # "trapezoidal"
                # I[k+1] = I[k] + (e[k] + e[k+1]) / 2 * dt — average of
                # current and previous error samples.
                integral_next = e_int + (e_i + e_i_prev) * 0.5 * self.dt

        # The integral consumed by ``_eval_control`` for the anti-windup
        # path below must match what the *next* output tick sees.  That
        # is the freshly computed ``integral_next`` (phase 1 used
        # ``e_int`` here; for byte-equivalence in the default config we
        # keep that behaviour by gating on _anti_windup_active).
        # T-127-followup-anti-windup / T-127-followup-tracking-mode -----
        # Both corrections need ``u_unsat`` — the value the controller
        # *would* publish before saturation.  Compute it once when either
        # mechanism is active and feed both branches.  When neither is
        # active this whole block is skipped → byte-equivalent to phase 1.
        aw_on = (
            self._anti_windup_active and self._anti_windup_method != "none"
        )
        tr_on = self.tracking_enabled
        if aw_on or tr_on:
            # Use the e_dot the next _output would read, for consistency
            # with the saturated-output path above.
            if self.filter_type != "none":
                e_dot_for_aw = e_dot_next
            elif self._derivative_method == "backward_diff":
                # _output reads state.e_dot_prev (the *previous* tick's
                # value); preserve that semantic here too.
                e_dot_for_aw = state.discrete_state.e_dot_prev
            elif self._derivative_method == "centered_diff":
                e_d_prev_prev = state.discrete_state.e_d_prev_prev
                e_dot_for_aw = (e_d - e_d_prev_prev) / (2.0 * self.dt)
            else:
                # forward_diff: re-use the phase-1 finite-difference
                # formula via the precomputed filter coefficients.
                b_coef, a_coef = self.filter
                e_d_prev = state.discrete_state.e_d_prev
                e_dot_for_aw = (
                    b_coef[0] * e_d + b_coef[1] * e_d_prev
                ) / a_coef[0]
            # T-127-followup-feedforward — feedforward is part of the
            # unsaturated control value the anti-windup logic compares
            # against ``u_sat``; pass ``r`` so ``_eval_control`` folds it
            # in.  Default kff=0.0 leaves the comparison unchanged.
            # T-127-followup-gain-scheduling — resolve runtime-port gains
            # here so the anti-windup comparison uses the same scheduled
            # values as the main ``_output`` path.
            kp_aw = self._resolve_kp(inputs, params)
            ki_aw = self._resolve_ki(inputs, params)
            kd_aw = self._resolve_kd(inputs, params)
            kff_aw = self._resolve_kff(inputs, params)
            u_unsat = self._eval_control(
                e_p, e_int, e_dot_for_aw,
                kp_aw, ki_aw, kd_aw, kff_aw, r=r,
            )
            if aw_on:
                u_sat = self._saturate(u_unsat, **params)
                if self._anti_windup_method == "back_calc":
                    # Back-calculation: pull the integrator toward the
                    # value that would have produced u_sat, with time
                    # constant Tt = anti_windup_gain.
                    tt = params["anti_windup_gain"]
                    integral_next = integral_next - (u_unsat - u_sat) / tt * self.dt
                elif self._anti_windup_method == "clamping":
                    # Integrator-tracking: only update when the controller
                    # is not pushing further into saturation.  ``saturating``
                    # is True when we are clamped AND the error sign matches
                    # the direction of saturation (positive sat → positive
                    # e_i pushes harder; negative sat → negative e_i pushes
                    # harder).  Use ``where`` for differentiability: gradient
                    # is zero on the clamped branch.
                    sat_excess = u_unsat - u_sat  # >0 if hit upper, <0 if lower
                    pushing_further = npa.logical_and(
                        sat_excess != 0,
                        npa.sign(e_i) == npa.sign(sat_excess),
                    )
                    integral_next = npa.where(
                        pushing_further, e_int, integral_next
                    )
            # T-127-followup-tracking-mode -----------------------------
            # Bumpless transfer: nudge the integrator toward the value
            # that would have produced ``u_ext``.  Implementation is
            # back-calculation with the *external* tracking signal in
            # place of ``u_sat``.  When anti-windup is also active the
            # two corrections sum (each is a small per-tick perturbation
            # of the integrator, so superposition is correct to first
            # order).
            # T-127-followup-i-on-error-only — when
            # ``integrate_tracking_error=False`` the tracking signal
            # ``u_ext`` MUST NOT touch the integrator.  We still declare
            # the ``u_ext`` port (callers may use it for downstream
            # pull-through paths) but the per-tick correction term is
            # suppressed entirely.  The default (``True``) preserves the
            # T-127-followup-tracking-mode kernel byte-for-byte.
            if tr_on and self.integrate_tracking_error:
                u_ext = self._resolve_u_ext(inputs)
                tt_tr = params.get("tracking_gain", 1.0)
                e_track = u_ext - u_unsat
                correction = (e_track / tt_tr) * self.dt
                # T-127-followup-bumpless-mode-switch — when the runtime
                # mode-flag port is declared, multiply the correction by
                # the gate so the user can flip auto/manual mid-sim.
                # The flag is cast to the integrator dtype so it remains
                # differentiable through ``tracking_gain`` (the gate
                # itself is a non-differentiable boolean — gradient
                # through the flag is zero).  Default
                # ``tracking_enabled_dynamic=False`` skips this
                # branch entirely, preserving the
                # T-127-followup-tracking-mode kernel byte-for-byte.
                if self.tracking_enabled_dynamic:
                    mode_flag = inputs[self.mode_flag_index]
                    # Explicit boolean coercion: any non-zero value
                    # turns tracking ON.  Multiplying by an array gate
                    # keeps the operation jit/grad-friendly.
                    gate = npa.where(
                        npa.asarray(mode_flag) != 0,
                        npa.asarray(1.0, dtype=correction.dtype),
                        npa.asarray(0.0, dtype=correction.dtype),
                    )
                    correction = correction * gate
                integral_next = integral_next + correction

        # Build the new state tuple, populating optional delay cells
        # only when the configured kernels need them.
        return self._make_state(
            integral=integral_next,
            e_d_prev=e_d,
            e_dot_prev=e_dot_next,
            e_i_prev=e_i,  # only stored when integrator_method != forward_euler
            e_d_prev_prev=state.discrete_state.e_d_prev,  # shift for centered_diff
        )

    def _eval_control(self, e_p, e_int, e_dot, kp, ki, kd, kff=0.0, r=None):
        # T-127-followup-feedforward — fold ``kff * r`` into the
        # unsaturated control sum.  ``kff`` defaults to 0.0 so the phase
        # 1 path (kff=0.0) reduces to the original
        # ``kp*e_p + ki*e_int + kd*e_dot`` arithmetic identity and
        # remains byte-equivalent.  ``r`` is only passed explicitly by
        # the saturation/anti-windup branches that need the unsaturated
        # value; older call sites without ``r`` get no feedforward
        # contribution.
        #
        # T-127-followup-gain-scheduling — the four gains are always
        # supplied by the caller (resolved from ``_resolve_k{p,i,d,ff}``),
        # so this helper neither consults ``params`` nor takes ``**params``.
        # That keeps the signature collision-free with the `**params`
        # unpacking used by ``_output`` / ``_update``.
        u = kp * e_p + ki * e_int + kd * e_dot
        if r is not None:
            u = u + kff * r
        return u

    def _output(self, time, state, *inputs, **params):
        b = self._resolve_b(inputs, params)
        r = inputs[self.setpoint_index]
        y = inputs[self.measurement_index]
        # T-127-followup-deadband-error — gate the proportional error
        # the same way ``_update`` does so the published ``u`` reflects
        # the deadband on every tick.  ``e_int`` already reflects the
        # gated integrator-error trajectory because ``_update`` writes
        # the gated ``e_i`` into the integral; ``e_dot`` is gated inside
        # ``_eval_derivative``.  Default error_deadband=0.0 is
        # byte-equivalent to phase 1.
        e_p = self._apply_deadband(b * r - y, params)
        e_int = state.discrete_state.integral
        e_dot = self._eval_derivative(time, state, *inputs, **params)
        # T-127-followup-feedforward — ``kff * r`` is added inside
        # ``_eval_control`` so it participates in saturation and the
        # anti-windup u_unsat/u_sat comparison.
        # T-127-followup-gain-scheduling — resolve each scalar gain from
        # its runtime port (if the matching ``*_dynamic`` flag is set)
        # or from the static parameter.  Default ``*_dynamic=False`` is
        # byte-equivalent to the phase 1 path.
        kp = self._resolve_kp(inputs, params)
        ki = self._resolve_ki(inputs, params)
        kd = self._resolve_kd(inputs, params)
        kff = self._resolve_kff(inputs, params)
        u = self._eval_control(e_p, e_int, e_dot, kp, ki, kd, kff, r=r)
        # T-127-followup-anti-windup — publish the saturated control
        # value when limits are configured.  Default-off path returns
        # ``u`` unchanged, matching phase 1 byte-for-byte.
        u = self._saturate(u, **params)
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        return u

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        # Use the setpoint port as the canonical input shape/dtype, matching
        # PIDDiscrete's single-input check_types pattern.
        u = self.eval_input(context, self.setpoint_index)
        xd = context[self.system_id].discrete_state.integral
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )

    # -----------------------------------------------------------------
    # T-127-followup-config-roundtrip — JSON-friendly config serialization.
    #
    # ``to_dict()`` captures every construction-time field that affects
    # behavior (the @parameters static + dynamic lists, plus the four
    # mode strings — anti_windup_method / integrator_method /
    # derivative_method / error_deadband_mode — and the ``*_dynamic``
    # port-topology flags).  ``from_dict()`` reconstructs an equivalent
    # block; round-tripping through ``json.dumps`` / ``json.loads`` is
    # supported because every encoded value is a Python primitive
    # (float / int / bool / str) or ``None``.
    #
    # Caveat (honest fallback documented in T-127-followup-config-
    # roundtrip): when any ``*_dynamic`` flag is True the corresponding
    # static scalar is still captured (it is the declared default that
    # would be used if the user later switched the flag off and
    # reconstructed without re-wiring); the dangling input-port
    # connection has to be re-built by the caller after ``from_dict``
    # since topology / wiring lives in the diagram, not the block.
    # -----------------------------------------------------------------

    # Static (non-dynamic-parameter) construction-time fields owned by
    # this block.  These are stored as attributes / static parameters
    # and never come from a runtime port.
    _CONFIG_STATIC_FIELDS = (
        "dt",
        "filter_type",
        "filter_coefficient",
        "anti_windup_method",
        "b_dynamic",
        "c_dynamic",
        "integrator_method",
        "derivative_method",
        "kp_dynamic",
        "ki_dynamic",
        "kd_dynamic",
        "kff_dynamic",
        "error_deadband_mode",
        "tracking_enabled",
        "integrate_tracking_error",
        "tracking_enabled_dynamic",
    )

    # Dynamic-parameter scalars.  These are declared via
    # ``declare_dynamic_parameter`` (possibly skipped when ``None`` —
    # see ``output_min`` / ``output_max``).
    _CONFIG_DYNAMIC_FIELDS = (
        "kp",
        "ki",
        "kd",
        "b",
        "c",
        "initial_state",
        "output_min",
        "output_max",
        "anti_windup_gain",
        "kff",
        "error_deadband",
        "error_deadband_sharpness",
        "tracking_gain",
    )

    @staticmethod
    def _encode_scalar(value):
        """Convert a parameter value (Python scalar, numpy / jax array)
        into a JSON-friendly primitive.

        - ``None`` stays ``None`` (used for unset saturation limits).
        - Scalar numbers / 0-D arrays become Python ``float``.
        - Booleans stay ``bool``.
        - Strings stay ``str``.
        Other types raise ``TypeError`` — config round-trip is only
        intended for the scalar-config subset.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return value
        # numpy / jax array fallback: only 0-D scalars are supported.
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        raise TypeError(
            f"PIDController2DOF.to_dict(): cannot serialize value "
            f"{value!r} of type {type(value).__name__}; only scalar "
            f"configuration values are supported."
        )

    def to_dict(self):
        """Return a JSON-serializable dict describing this controller.

        The dict captures every construction-time field that controls
        the block's behavior — all ``@parameters``-registered fields
        plus the mode strings and ``*_dynamic`` port-topology flags
        that live outside ``@parameters``.  Round-tripping through
        :meth:`from_dict` (optionally via ``json.dumps`` /
        ``json.loads``) produces a block with identical step-response
        behavior on a fixed input.

        Note:
            Diagram wiring (which signal feeds which input port) is
            not part of the block config and must be re-established by
            the caller after :meth:`from_dict`.  This is especially
            relevant when any ``*_dynamic`` flag is True — the
            reconstructed block still declares the runtime port, but
            it has no upstream connection until the caller wires it
            up.

        Returns:
            dict mapping each of :attr:`_CONFIG_STATIC_FIELDS` and
            :attr:`_CONFIG_DYNAMIC_FIELDS` to a JSON primitive.
        """
        data = {}
        # Static fields: read directly off the static-parameter dict
        # (where ``@parameters`` stashed them) or fall back to the
        # cached instance attribute when the kwarg lives there too.
        for name in self._CONFIG_STATIC_FIELDS:
            if name in self._static_parameters:
                value = Parameter.unwrap(self._static_parameters[name])
            else:
                value = getattr(self, name, None)
            data[name] = self._encode_scalar(value)
        # Dynamic fields: only declared when non-None (see
        # ``parameters`` decorator).  Encode missing entries as None
        # — that lets the constructor's own default reactivate on
        # ``from_dict`` and keeps ``output_min`` / ``output_max``
        # round-trippable.
        for name in self._CONFIG_DYNAMIC_FIELDS:
            if name in self._dynamic_parameters:
                value = Parameter.unwrap(self._dynamic_parameters[name])
            else:
                value = None
            data[name] = self._encode_scalar(value)
        return data

    @classmethod
    def from_dict(cls, data, **block_kwargs):
        """Reconstruct a :class:`PIDController2DOF` from a config dict.

        Extra keyword arguments (``name=``, ``system_id=``, ...) are
        forwarded to the constructor so a deserialized block can pick
        up a fresh name in its target diagram.

        Args:
            data: dict produced by :meth:`to_dict` (or any equivalent
                mapping with the same key set).  ``dt`` is the only
                required field; every other field falls back to the
                constructor default when absent.
            **block_kwargs: forwarded to ``PIDController2DOF.__init__``
                (typically ``name=...``).

        Raises:
            ValueError: if ``data`` lacks the required ``dt`` field.

        Returns:
            A new :class:`PIDController2DOF` whose configuration
            matches ``data``.
        """
        if "dt" not in data or data["dt"] is None:
            raise ValueError(
                "PIDController2DOF.from_dict(): missing required field "
                "'dt' (the controller sample period)."
            )
        # Build the constructor kwargs.  Drop missing / None-valued
        # entries for fields whose constructor default is *not* None
        # so the constructor picks them up; preserve None for
        # ``output_min`` / ``output_max`` (their default IS None and
        # round-trip needs them to stay None).
        ctor_kwargs = {}
        for name in cls._CONFIG_STATIC_FIELDS + cls._CONFIG_DYNAMIC_FIELDS:
            if name not in data:
                continue
            value = data[name]
            # output_min / output_max naturally accept None.
            if value is None and name not in ("output_min", "output_max"):
                continue
            ctor_kwargs[name] = value
        ctor_kwargs.update(block_kwargs)
        return cls(**ctor_kwargs)

    # ------------------------------------------------------------------
    # T-127-followup-derivative-on-measurement — convenience factories
    # for the two common 2-DOF PID configurations.  ``standard`` keeps
    # the textbook ``b = c = 1`` defaults (equivalent to a 1-DOF PID on
    # the error signal), while ``with_derivative_on_measurement`` ships
    # ``b = 1, c = 0`` — the standard "no derivative kick" recipe
    # recommended for real-world controllers (a step change in the
    # setpoint no longer produces a ``Kd / dt`` spike through the
    # derivative term).
    # ------------------------------------------------------------------
    @classmethod
    def standard(cls, kp, ki, kd, dt, **kwargs):
        """Construct a textbook PID with ``b = c = 1``.

        Convenience factory equivalent to ``PIDController2DOF(dt, kp,
        ki, kd)``: the proportional and derivative paths both see the
        setpoint with weight 1, so the block reduces to a 1-DOF PID on
        the error signal ``e = r - y``.  Useful as the explicit
        counterpart to :meth:`with_derivative_on_measurement` —
        callers self-document which 2-DOF configuration they want.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            dt: Sampling period.
            **kwargs: Forwarded to :class:`PIDController2DOF`.  Setting
                ``b`` or ``c`` here is allowed but discouraged (use the
                main constructor for non-standard weights).

        Returns:
            A :class:`PIDController2DOF` with ``b = c = 1``.
        """
        kwargs.setdefault("b", 1.0)
        kwargs.setdefault("c", 1.0)
        return cls(dt=dt, kp=kp, ki=ki, kd=kd, **kwargs)

    @classmethod
    def with_derivative_on_measurement(cls, kp, ki, kd, dt, **kwargs):
        """Construct a PID with derivative-on-measurement-only (``b=1, c=0``).

        Convenience factory for the standard "no derivative kick"
        recipe used by most real-world controllers.  With ``c = 0`` the
        derivative term sees only the measurement (``-d/dt(y)``), so a
        step change in the setpoint does NOT inject a ``Kd / dt`` spike
        through the derivative path — only integral and proportional
        action drive the transient.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            dt: Sampling period.
            **kwargs: Forwarded to :class:`PIDController2DOF`.  Passing
                ``c`` or ``c_dynamic`` here raises ``ValueError`` via
                the constructor's ``derivative_on_measurement_only``
                contract — by construction this factory pins ``c=0``.

        Returns:
            A :class:`PIDController2DOF` with ``b = 1`` and ``c = 0``.
        """
        kwargs.setdefault("b", 1.0)
        return cls(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            derivative_on_measurement_only=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # T-127-followup-pid-tuning-helpers — classical tuning-rule
    # classmethods.  Each helper computes ``(Kp, Ki, Kd)`` from the
    # plant-characterisation inputs and returns a configured
    # :class:`PIDController2DOF`.  The helpers are pure factories: they
    # add no behavioural surface to the existing PID class and the gains
    # they emit are forwarded to the standard constructor unchanged.
    #
    # Three rules are shipped (the classical control-engineering set):
    #
    # 1. ``ziegler_nichols`` — closed-loop ultimate-cycle method.  Given
    #    the ultimate gain ``Ku`` (the proportional gain at which the
    #    closed loop just oscillates with sustained period ``Tu``), the
    #    Z-N table maps to PID gains for P / PI / PID modes.  The
    #    coefficients are the canonical Ziegler & Nichols (1942) values
    #    reproduced in every textbook (e.g. Astrom & Hagglund).
    #
    # 2. ``cohen_coon`` — open-loop process-reaction-curve method for a
    #    first-order-plus-dead-time (FOPDT) plant
    #    ``G(s) = K * exp(-theta*s) / (tau*s + 1)``.  Slightly more
    #    aggressive than Z-N on dead-time-dominated plants; widely used
    #    for chemical-process control.
    #
    # 3. ``tyreus_luyben`` — Z-N alternative with a much longer integral
    #    time (PI: ``Kp = Ku/3.2, Ti = 2.2*Tu``).  Trades response speed
    #    for robustness; commonly used when the Z-N gains produce too
    #    much overshoot or sensitivity to model error.
    # ------------------------------------------------------------------
    _ZIEGLER_NICHOLS_MODES = ("P", "PI", "PID")
    _COHEN_COON_MODES = ("P", "PI", "PID")

    @classmethod
    def ziegler_nichols(cls, Ku, Tu, dt, mode="PID", **kwargs):
        """Construct a PID tuned by the Ziegler-Nichols ultimate-cycle rule.

        Given the ultimate gain ``Ku`` (the proportional-only gain at
        which the closed loop just sustains oscillation) and the
        corresponding ultimate period ``Tu``, the Z-N table maps to
        controller gains as::

            P:    Kp = 0.5  * Ku,                 Ki = 0,            Kd = 0
            PI:   Kp = 0.45 * Ku, Ti = Tu / 1.2 → Ki = 0.54*Ku/Tu,   Kd = 0
            PID:  Kp = 0.6  * Ku, Ti = Tu / 2.0 → Ki = 1.2 *Ku/Tu,
                  Td = Tu / 8.0                 → Kd = 0.075*Ku*Tu

        The coefficients are the canonical Ziegler & Nichols (1942)
        values; see e.g. Astrom & Hagglund, *PID Controllers: Theory,
        Design, and Tuning* (1995), Table 4.1.

        Args:
            Ku: Ultimate gain (proportional-only gain at sustained
                oscillation). Must be positive.
            Tu: Ultimate period (period of the sustained oscillation
                in seconds). Must be positive.
            dt: Sampling period for the discrete PID.
            mode: One of ``"P"``, ``"PI"``, ``"PID"`` (default
                ``"PID"``). Selects which gains are non-zero.
            **kwargs: Forwarded to :class:`PIDController2DOF`.

        Returns:
            A :class:`PIDController2DOF` whose ``(Kp, Ki, Kd)`` match
            the Z-N table for the requested mode.

        Raises:
            ValueError: If ``mode`` is not one of ``"P"``, ``"PI"``,
                ``"PID"``, or if ``Ku`` / ``Tu`` are non-positive.
        """
        if mode not in cls._ZIEGLER_NICHOLS_MODES:
            raise ValueError(
                f"ziegler_nichols: mode must be one of "
                f"{cls._ZIEGLER_NICHOLS_MODES!r}; got {mode!r}"
            )
        Ku_f = float(Ku)
        Tu_f = float(Tu)
        if Ku_f <= 0.0:
            raise ValueError(
                f"ziegler_nichols: Ku must be positive; got {Ku_f}"
            )
        if Tu_f <= 0.0:
            raise ValueError(
                f"ziegler_nichols: Tu must be positive; got {Tu_f}"
            )
        if mode == "P":
            kp = 0.5 * Ku_f
            ki = 0.0
            kd = 0.0
        elif mode == "PI":
            kp = 0.45 * Ku_f
            ki = 0.54 * Ku_f / Tu_f
            kd = 0.0
        else:  # PID
            kp = 0.6 * Ku_f
            ki = 1.2 * Ku_f / Tu_f
            kd = 0.075 * Ku_f * Tu_f
        return cls(dt=dt, kp=kp, ki=ki, kd=kd, **kwargs)

    @classmethod
    def cohen_coon(cls, K, tau, theta, dt, mode="PID", **kwargs):
        """Construct a PID tuned by the Cohen-Coon rule for a FOPDT plant.

        For a first-order-plus-dead-time plant
        ``G(s) = K * exp(-theta*s) / (tau*s + 1)`` (process gain ``K``,
        time constant ``tau``, dead time ``theta``), the Cohen-Coon
        (1953) formulas are::

            r = theta / tau

            P:    Kp = (1/K) * (1/r) * (1 + r/3)

            PI:   Kp = (1/K) * (1/r) * (9/10 + r/12)
                  Ti = theta * (30 + 3*r) / (9 + 20*r)

            PID:  Kp = (1/K) * (1/r) * (4/3 + r/4)
                  Ti = theta * (32 + 6*r) / (13 + 8*r)
                  Td = theta * 4 / (11 + 2*r)

        with ``Ki = Kp / Ti`` and ``Kd = Kp * Td``.

        Cohen-Coon is more aggressive than Ziegler-Nichols on plants
        where ``theta / tau`` is large (dead-time-dominated processes);
        it is widely used in chemical-process control.

        Args:
            K: Process (steady-state) gain. Must be non-zero.
            tau: First-order time constant in seconds. Must be positive.
            theta: Dead time in seconds. Must be positive.
            dt: Sampling period for the discrete PID.
            mode: One of ``"P"``, ``"PI"``, ``"PID"`` (default
                ``"PID"``). Selects which gains are non-zero.
            **kwargs: Forwarded to :class:`PIDController2DOF`.

        Returns:
            A :class:`PIDController2DOF` whose ``(Kp, Ki, Kd)`` match
            the Cohen-Coon formulas for the requested mode.

        Raises:
            ValueError: If ``mode`` is not one of ``"P"``, ``"PI"``,
                ``"PID"``, or if ``K`` is zero, or if ``tau`` /
                ``theta`` are non-positive.
        """
        if mode not in cls._COHEN_COON_MODES:
            raise ValueError(
                f"cohen_coon: mode must be one of "
                f"{cls._COHEN_COON_MODES!r}; got {mode!r}"
            )
        K_f = float(K)
        tau_f = float(tau)
        theta_f = float(theta)
        if K_f == 0.0:
            raise ValueError("cohen_coon: K must be non-zero")
        if tau_f <= 0.0:
            raise ValueError(
                f"cohen_coon: tau must be positive; got {tau_f}"
            )
        if theta_f <= 0.0:
            raise ValueError(
                f"cohen_coon: theta must be positive; got {theta_f}"
            )
        # Use npa for the arithmetic so the helper composes with the
        # backend selector even when callers pass JAX scalars.  npa is
        # already imported at module scope.
        r = npa.divide(theta_f, tau_f)
        inv_K = npa.divide(1.0, K_f)
        inv_r = npa.divide(1.0, r)
        if mode == "P":
            kp = float(inv_K * inv_r * (1.0 + r / 3.0))
            ki = 0.0
            kd = 0.0
        elif mode == "PI":
            kp = float(inv_K * inv_r * (9.0 / 10.0 + r / 12.0))
            Ti = float(theta_f * (30.0 + 3.0 * r) / (9.0 + 20.0 * r))
            ki = kp / Ti
            kd = 0.0
        else:  # PID
            kp = float(inv_K * inv_r * (4.0 / 3.0 + r / 4.0))
            Ti = float(theta_f * (32.0 + 6.0 * r) / (13.0 + 8.0 * r))
            Td = float(theta_f * 4.0 / (11.0 + 2.0 * r))
            ki = kp / Ti
            kd = kp * Td
        return cls(dt=dt, kp=kp, ki=ki, kd=kd, **kwargs)

    @classmethod
    def tyreus_luyben(cls, Ku, Tu, dt, **kwargs):
        """Construct a PI controller tuned by the Tyreus-Luyben rule.

        A gentler Ziegler-Nichols alternative that trades response
        speed for robustness.  Tyreus & Luyben (1992) recommend the
        PI form for most chemical-process applications because the
        derivative term tends to amplify measurement noise::

            Kp = Ku / 3.2,  Ti = 2.2 * Tu  →  Ki = Kp / Ti

        and ``Kd = 0`` (no derivative action).  Compared with Z-N,
        Tyreus-Luyben gives roughly 1/3 the proportional gain and a
        ~4x longer integral time, producing a much less aggressive
        loop with substantially better robustness to model error.

        Args:
            Ku: Ultimate gain (proportional-only gain at sustained
                oscillation). Must be positive.
            Tu: Ultimate period (period of the sustained oscillation
                in seconds). Must be positive.
            dt: Sampling period for the discrete PID.
            **kwargs: Forwarded to :class:`PIDController2DOF`.

        Returns:
            A :class:`PIDController2DOF` configured as a PI
            controller (``Kd = 0``) with the Tyreus-Luyben gains.

        Raises:
            ValueError: If ``Ku`` / ``Tu`` are non-positive.
        """
        Ku_f = float(Ku)
        Tu_f = float(Tu)
        if Ku_f <= 0.0:
            raise ValueError(
                f"tyreus_luyben: Ku must be positive; got {Ku_f}"
            )
        if Tu_f <= 0.0:
            raise ValueError(
                f"tyreus_luyben: Tu must be positive; got {Tu_f}"
            )
        kp = Ku_f / 3.2
        Ti = 2.2 * Tu_f
        ki = kp / Ti
        kd = 0.0
        return cls(dt=dt, kp=kp, ki=ki, kd=kd, **kwargs)


# T-127-followup-derivative-on-measurement: outer wrapper that rewrites
# ``c=0`` BEFORE the ``@parameters`` decorator captures the call's
# original kwargs.  The decorator reads ``kwargs`` / ``args`` directly
# (see ``_get_params`` in framework/system_decorators.py) and ignores
# any in-body mutation of the ``c`` local, so to propagate the
# convenience flag's intent into the dynamic-parameter dispatch we have
# to mutate the kwargs dict at this outer-most layer.
def _pid2dof_derivative_on_measurement_wrapper(decorated_init):
    @wraps(decorated_init)
    def wrapper(
        self,
        *args,
        derivative_on_measurement_only=False,
        **kwargs,
    ):
        if derivative_on_measurement_only:
            # Reject explicit ``c`` (anything other than the default
            # 1.0 sentinel) — the contract is "use one or the other".
            c_explicit = kwargs.get("c", 1.0)
            if c_explicit != 1.0:
                raise ValueError(
                    "PIDController2DOF: derivative_on_measurement_only="
                    "True implies c=0; cannot combine with an explicit "
                    f"c={c_explicit!r}"
                )
            if kwargs.get("c_dynamic", False):
                raise ValueError(
                    "PIDController2DOF: derivative_on_measurement_only="
                    "True implies a static c=0; cannot combine with "
                    "c_dynamic=True"
                )
            kwargs["c"] = 0.0
            kwargs["c_dynamic"] = False
        return decorated_init(self, *args, **kwargs)

    return wrapper


PIDController2DOF.__init__ = _pid2dof_derivative_on_measurement_wrapper(
    PIDController2DOF.__init__
)



# T-122-followup-band-limited-noise end-of-file marker.


# ===========================================================================
# T-127-followup-discrete-filter-family — discrete-filter primitives
# commonly paired with the T-127 ``PIDController2DOF`` (and the existing
# ``PIDDiscrete``/``DerivativeDiscrete``/``IntegratorDiscrete`` family).
#
# Phase 1 of this follow-up ships two of the three blocks listed in the
# parent T-127 deferral (LowPassDiscrete, LeadLag).  ``Notch`` is left as
# a deeper follow-up — its biquad formula is straightforward but the
# bandwidth-vs-Q parameterisation deserves a separate review pass.
#
# Both blocks below:
# * Are differentiable through their tunable parameters (``cutoff_hz``
#   for LowPassDiscrete; ``K``, ``T_lead``, ``T_lag`` for LeadLag) under
#   ``jax.grad`` because the recursive update is composed of arithmetic
#   on the dynamic parameters.
# * Preserve the T-005 default-float64 policy — the discrete-state seed
#   is a plain Python ``0.0`` (no explicit dtype), so the JAX/NumPy
#   default float (float64 with x64 enabled) is honoured.
# * Use ``jaxonomy.backend.numpy_api as npa`` for any array math that
#   needs to live inside the traced update / output functions.
# * Mirror the ``FilterDiscrete`` / ``PIDController2DOF`` pattern:
#   ``@parameters`` decorator, ``declare_periodic_update`` +
#   ``configure_periodic_update`` in ``initialize`` for the recursive
#   state update, plus a ``configure_output_port`` for the readout.
#
# Math:
#
# ``LowPassDiscrete`` (single-pole RC, bilinear-equivalent at low f):
#     tau     = 1 / (2*pi*cutoff_hz)
#     alpha   = dt / (dt + tau)
#     y[k]    = alpha * x[k] + (1 - alpha) * y[k-1]
#   The -3 dB point of this update is at ``f = 1/(2*pi*tau) = cutoff_hz``
#   in the limit ``dt -> 0``; for finite ``dt`` the discrete cutoff drifts
#   slightly below the design value, but the simple-RC form is the most
#   widely-used discrete LPF in control loops.
#
# ``LeadLag`` (continuous compensator G(s) = K * (1 + T_lead*s) /
# (1 + T_lag*s) discretised by Tustin / bilinear transform with
# ``s = (2/dt) * (z-1)/(z+1)``):
#     c   = 2 / dt
#     den = 1 + T_lag * c
#     b0  = K * (1 + T_lead * c) / den
#     b1  = K * (1 - T_lead * c) / den
#     a1  = (1 - T_lag * c) / den
#     y[k] = b0 * x[k] + b1 * x[k-1] - a1 * y[k-1]
#   With ``T_lead == T_lag`` the compensator collapses to a pure gain
#   ``K`` (the s-domain pole and zero cancel), which is used as an
#   identity / sanity check in the test corpus.
# ===========================================================================


class LowPassDiscrete(LeafSystem):
    """Discrete first-order (single-pole RC) low-pass filter.

    Implements the difference equation::

        tau   = 1 / (2*pi*cutoff_hz)
        alpha = dt / (dt + tau)
        y[k]  = alpha * x[k] + (1 - alpha) * y[k-1]

    The continuous-time analogue is ``H(s) = 1 / (1 + tau*s)``, with -3 dB
    crossover at ``f = cutoff_hz`` in the small-``dt`` limit.

    Input ports:
        (0) The input signal ``x``.

    Output ports:
        (0) The filtered signal ``y``.

    Parameters:
        dt:
            Sampling period of the block (s).
        cutoff_hz:
            Design cutoff frequency (Hz).  Differentiable.
        initial_state:
            Initial value of ``y[-1]``.  Default 0.0.

    Notes:
        Differentiability: ``cutoff_hz`` enters the recursive update via
        smooth arithmetic (``alpha = dt/(dt + 1/(2*pi*cutoff_hz))``), so
        ``jax.grad`` is finite through it.
    """

    @parameters(
        static=["dt"],
        dynamic=["cutoff_hz", "initial_state"],
    )
    def __init__(
        self,
        dt,
        cutoff_hz=1.0,
        initial_state=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, cutoff_hz, initial_state, dt=None):
        # Discrete-state seed: previous output ``y[k-1]``.  Use a plain
        # Python float so the T-005 default (float64 when x64 is on) is
        # honoured at trace time.
        y0 = npa.asarray(initial_state)
        self.declare_discrete_state(default_value=y0)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=self.dt,
        )

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=self.dt,
            default_value=y0,
            prerequisites_of_calc=[DependencyTicket.xd],
        )

    def reset_default_values(self, **dynamic_parameters):
        y0 = npa.asarray(dynamic_parameters["initial_state"])
        self.configure_discrete_state_default_value(default_value=y0)
        self.configure_output_port_default_value(self._output_port_idx, y0)

    @staticmethod
    def _alpha(dt, cutoff_hz):
        # ``tau = 1/(2*pi*fc)``, ``alpha = dt / (dt + tau)``.  Smooth in
        # ``cutoff_hz`` so jax.grad is finite.
        two_pi = 2.0 * npa.pi
        tau = 1.0 / (two_pi * cutoff_hz)
        return dt / (dt + tau)

    def _update(self, _time, state, *inputs, **params):
        x = inputs[0]
        y_prev = state.discrete_state
        alpha = self._alpha(self.dt, params["cutoff_hz"])
        return alpha * x + (1.0 - alpha) * y_prev

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )


class LeadLag(LeafSystem):
    """Discrete first-order lead-lag compensator.

    Discretisation of the continuous compensator
    ``G(s) = K * (1 + T_lead * s) / (1 + T_lag * s)``
    via the Tustin (bilinear) transform with ``s = (2/dt)*(z-1)/(z+1)``.

    The resulting difference equation is::

        c   = 2 / dt
        den = 1 + T_lag * c
        b0  = K * (1 + T_lead * c) / den
        b1  = K * (1 - T_lead * c) / den
        a1  = (1 - T_lag * c) / den
        y[k] = b0 * x[k] + b1 * x[k-1] - a1 * y[k-1]

    With ``T_lead = T_lag`` the s-domain pole and zero cancel and the
    block reduces to a pure gain ``K`` (used as an identity check in
    the corpus).

    Input ports:
        (0) The input signal ``x``.

    Output ports:
        (0) The compensated signal ``y``.

    Parameters:
        dt:
            Sampling period of the block (s).
        K:
            Compensator gain.  Differentiable.
        T_lead:
            Lead time constant (s).  Differentiable.
        T_lag:
            Lag time constant (s).  Must be > 0.  Differentiable.
        initial_state:
            Initial value of ``y[-1]``.  Default 0.0.

    Notes:
        Differentiability: ``K``, ``T_lead`` and ``T_lag`` flow into the
        biquad coefficients via smooth arithmetic, so ``jax.grad`` is
        finite through them.

        The block is feedthrough on its input port (``b0 != 0`` whenever
        ``K != 0``), so ``y[k]`` depends on ``x[k]`` directly.
    """

    class DiscreteStateType(NamedTuple):
        x_prev: Array
        y_prev: Array

    @parameters(
        static=["dt"],
        dynamic=["K", "T_lead", "T_lag", "initial_state"],
    )
    def __init__(
        self,
        dt,
        K=1.0,
        T_lead=1.0,
        T_lag=1.0,
        initial_state=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, K, T_lead, T_lag, initial_state, dt=None):
        y0 = npa.asarray(initial_state)
        x0 = npa.zeros_like(y0)
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(x_prev=x0, y_prev=y0),
            as_array=False,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=self.dt,
        )

        # Feedthrough: y[k] depends on x[k] through b0.
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=self.dt,
            default_value=y0,
            requires_inputs=True,
            prerequisites_of_calc=[
                DependencyTicket.xd,
                self.input_ports[0].ticket,
            ],
        )

    def reset_default_values(self, **dynamic_parameters):
        y0 = npa.asarray(dynamic_parameters["initial_state"])
        x0 = npa.zeros_like(y0)
        self.configure_discrete_state_default_value(
            self.DiscreteStateType(x_prev=x0, y_prev=y0),
            as_array=False,
        )
        self.configure_output_port_default_value(self._output_port_idx, y0)

    def _coeffs(self, K, T_lead, T_lag):
        # Bilinear-transform biquad coefficients.  Pure arithmetic on
        # the dynamic parameters → JAX-traceable and differentiable.
        c = 2.0 / self.dt
        den = 1.0 + T_lag * c
        b0 = K * (1.0 + T_lead * c) / den
        b1 = K * (1.0 - T_lead * c) / den
        a1 = (1.0 - T_lag * c) / den
        return b0, b1, a1

    def _update(self, _time, state, *inputs, **params):
        x = inputs[0]
        b0, b1, a1 = self._coeffs(params["K"], params["T_lead"], params["T_lag"])
        y_prev = state.discrete_state.y_prev
        x_prev = state.discrete_state.x_prev
        y_new = b0 * x + b1 * x_prev - a1 * y_prev
        return self.DiscreteStateType(x_prev=x, y_prev=y_new)

    def _output(self, _time, state, *inputs, **params):
        # Feedthrough output: recompute y[k] from x[k] and the stored
        # x[k-1], y[k-1] so the readout is consistent with the update.
        x = inputs[0]
        b0, b1, a1 = self._coeffs(params["K"], params["T_lead"], params["T_lag"])
        y_prev = state.discrete_state.y_prev
        x_prev = state.discrete_state.x_prev
        return b0 * x + b1 * x_prev - a1 * y_prev

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state.y_prev
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )


# T-127-followup-discrete-filter-family end-of-file marker.


# ===========================================================================
# T-127-fu-notch — Biquad band-stop ("notch") filter.
#
# Completes the discrete-filter family started by
# ``T-127-followup-discrete-filter-family`` (which shipped
# ``LowPassDiscrete`` + ``LeadLag``) by adding the deferred ``Notch`` block.
#
# Transfer function (standard biquad notch / band-stop, direct-form-I):
#
#     omega0 = 2*pi * frequency_hz * dt
#     r      = 1 - pi * bandwidth_hz * dt          (pole radius)
#     H(z)   = (1 - 2*cos(omega0)*z^-1 + z^-2)
#              / (1 - 2*r*cos(omega0)*z^-1 + r^2 * z^-2)
#
# Difference equation::
#
#     b0_raw = 1
#     b1_raw = -2 * rho * cos(omega0)
#     b2_raw = rho * rho                            (see ``depth`` below)
#     a1     = -2 * r * cos(omega0)
#     a2     = r * r
#     dc_gain = (b0_raw + b1_raw + b2_raw) / (1 + a1 + a2)
#     b0, b1, b2 = b0_raw/dc_gain, b1_raw/dc_gain, b2_raw/dc_gain
#     y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
#            - a1*y[k-1] - a2*y[k-2]
#
# The DC-gain normalisation is a scalar on the numerator, so it does
# *not* change the location of the zeros — on-notch attenuation is
# preserved while ``H(1) = 1`` exactly.
#
# The ``depth`` argument trims the *numerator* zeros between the
# denominator's pole radius (``depth = 0`` ⇒ zero/pole cancellation, the
# block is a pure pass-through) and the unit circle (``depth = 1`` ⇒
# infinitely deep notch).  In closed form::
#
#     rho2 = r*r + depth * (1 - r*r)      (squared zero radius)
#     b0   = 1
#     b1   = -2 * sqrt(rho2) * cos(omega0)
#     b2   = rho2
#
# So ``depth = 0`` ⇒ ``rho2 = r*r`` and the numerator equals the
# denominator (H(z) = 1), while ``depth = 1`` ⇒ ``rho2 = 1`` and the
# zeros sit on the unit circle, recovering the textbook biquad notch
# whose attenuation at ``omega0`` is exactly zero.
#
# Discrete state: previous two inputs ``(x[k-1], x[k-2])`` and previous
# two outputs ``(y[k-1], y[k-2])`` — 4 scalars stored as a NamedTuple
# pytree (matches the LeadLag pattern so JAX traces the recursive update
# cleanly).
#
# Differentiability: ``frequency_hz`` and ``bandwidth_hz`` (and
# ``depth``) flow into ``omega0`` / ``r`` / ``rho2`` via smooth
# arithmetic + ``cos`` + ``sqrt`` so ``jax.grad`` is finite through them.
#
# T-005: discrete-state seeds are plain Python ``0.0``s wrapped via
# ``npa.asarray`` so the default float (float64 with x64 on) is honoured.
# ===========================================================================


class Notch(LeafSystem):
    """Discrete biquad band-stop ("notch") filter.

    Implements the textbook biquad notch::

        omega0 = 2*pi * frequency_hz * dt
        r      = 1 - pi * bandwidth_hz * dt          (pole radius)
        rho2   = r*r + depth * (1 - r*r)             (zero radius**2)
        b0, b1, b2 = 1, -2*sqrt(rho2)*cos(omega0), rho2
        a1, a2     = -2*r*cos(omega0), r*r
        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]

    With ``depth = 1`` the zeros sit on the unit circle and the notch is
    infinitely deep.  With ``depth = 0`` the numerator collapses to the
    denominator and the block becomes a unit-gain pass-through (useful
    as an identity check).

    Input ports:
        (0) The input signal ``x``.

    Output ports:
        (0) The filtered signal ``y``.

    Parameters:
        dt:
            Sampling period of the block (s).
        frequency_hz:
            Notch centre frequency (Hz).  Differentiable.
        bandwidth_hz:
            Approximate -3 dB bandwidth of the notch (Hz).  Differentiable.
            Must satisfy ``pi * bandwidth_hz * dt < 1`` so the pole
            radius ``r`` remains in ``(0, 1)``.
        depth:
            Notch depth in ``[0, 1)``.  Default 0.99.  Larger ⇒ deeper
            notch; ``depth = 1`` puts the zeros on the unit circle for an
            infinitely deep notch (allowed but numerically borderline).
        initial_state:
            Initial value of ``y[-1]`` (and, implicitly, ``y[-2]``,
            ``x[-1]``, ``x[-2]``).  Default 0.0.

    Notes:
        Differentiability: ``frequency_hz``, ``bandwidth_hz``, and
        ``depth`` enter the biquad coefficients via smooth arithmetic
        (``cos``, ``sqrt``), so ``jax.grad`` is finite through them.

        The block is feedthrough on its input port (``b0 = 1``), so
        ``y[k]`` depends on ``x[k]`` directly.
    """

    class DiscreteStateType(NamedTuple):
        x_prev1: Array  # x[k-1]
        x_prev2: Array  # x[k-2]
        y_prev1: Array  # y[k-1]
        y_prev2: Array  # y[k-2]

    @parameters(
        static=["dt"],
        dynamic=["frequency_hz", "bandwidth_hz", "depth", "initial_state"],
    )
    def __init__(
        self,
        dt,
        frequency_hz=1.0,
        bandwidth_hz=0.1,
        depth=0.99,
        initial_state=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(
        self,
        frequency_hz,
        bandwidth_hz,
        depth,
        initial_state,
        dt=None,
    ):
        y0 = npa.asarray(initial_state)
        x0 = npa.zeros_like(y0)
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                x_prev1=x0, x_prev2=x0, y_prev1=y0, y_prev2=y0
            ),
            as_array=False,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=self.dt,
        )

        # Feedthrough: y[k] depends on x[k] through b0 = 1.
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=self.dt,
            default_value=y0,
            requires_inputs=True,
            prerequisites_of_calc=[
                DependencyTicket.xd,
                self.input_ports[0].ticket,
            ],
        )

    def reset_default_values(self, **dynamic_parameters):
        y0 = npa.asarray(dynamic_parameters["initial_state"])
        x0 = npa.zeros_like(y0)
        self.configure_discrete_state_default_value(
            self.DiscreteStateType(
                x_prev1=x0, x_prev2=x0, y_prev1=y0, y_prev2=y0
            ),
            as_array=False,
        )
        self.configure_output_port_default_value(self._output_port_idx, y0)

    def _coeffs(self, frequency_hz, bandwidth_hz, depth):
        # ``omega0 = 2*pi*f*dt`` (digital angular frequency at the notch).
        # ``r = 1 - pi*bw*dt`` is the pole radius; the standard
        # approximation that gives an *approximate* -3 dB bandwidth of
        # ``bandwidth_hz`` (cf. Steiglitz, "A Digital Signal Processing
        # Primer", §9.4).
        two_pi = 2.0 * npa.pi
        omega0 = two_pi * frequency_hz * self.dt
        r = 1.0 - npa.pi * bandwidth_hz * self.dt

        cos_w0 = npa.cos(omega0)
        r2 = r * r

        # depth in [0, 1]: 0 ⇒ rho2 = r2 (numerator = denominator ⇒
        # pass-through), 1 ⇒ rho2 = 1 (zeros on the unit circle ⇒
        # infinitely deep notch).
        rho2 = r2 + depth * (1.0 - r2)
        rho = npa.sqrt(rho2)

        b0 = 1.0
        b1 = -2.0 * rho * cos_w0
        b2 = rho2
        a1 = -2.0 * r * cos_w0
        a2 = r2

        # Normalise so DC gain ``H(1) = (b0+b1+b2)/(1+a1+a2)`` equals 1.
        # Without this, the textbook biquad has a slight (~1 %) DC bump.
        # Scaling the numerator zeros uniformly preserves the on-notch
        # attenuation.
        dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2)
        b0 = b0 / dc_gain
        b1 = b1 / dc_gain
        b2 = b2 / dc_gain
        return b0, b1, b2, a1, a2

    def _update(self, _time, state, *inputs, **params):
        x = inputs[0]
        b0, b1, b2, a1, a2 = self._coeffs(
            params["frequency_hz"],
            params["bandwidth_hz"],
            params["depth"],
        )
        xd = state.discrete_state
        y_new = (
            b0 * x
            + b1 * xd.x_prev1
            + b2 * xd.x_prev2
            - a1 * xd.y_prev1
            - a2 * xd.y_prev2
        )
        return self.DiscreteStateType(
            x_prev1=x,
            x_prev2=xd.x_prev1,
            y_prev1=y_new,
            y_prev2=xd.y_prev1,
        )

    def _output(self, _time, state, *inputs, **params):
        # Feedthrough output: recompute y[k] from x[k] and the stored
        # delay-line so the readout matches the update.
        x = inputs[0]
        b0, b1, b2, a1, a2 = self._coeffs(
            params["frequency_hz"],
            params["bandwidth_hz"],
            params["depth"],
        )
        xd = state.discrete_state
        return (
            b0 * x
            + b1 * xd.x_prev1
            + b2 * xd.x_prev2
            - a1 * xd.y_prev1
            - a2 * xd.y_prev2
        )

    def check_types(
        self,
        context,
        error_collector: ErrorCollector = None,
    ):
        u = self.eval_input(context)
        xd = context[self.system_id].discrete_state.y_prev1
        check_state_type(
            self,
            inp_data=u,
            state_data=xd,
            error_collector=error_collector,
        )



# T-114-followup-phase3-2d-cubic end-of-block marker.


# ---------------------------------------------------------------------------
# T-107-followup-variable-tau — Variable Transport Delay block.
#
# Generalises ``TransportDelay`` (T-107 phase 1) to a signal-driven delay:
# the delay value ``tau`` is read from a second input port at every output
# evaluation rather than supplied as a static parameter. Reuses the same
# ring-buffer machinery (a ``_BufferState(times, values)`` NamedTuple held
# in discrete state, populated by a periodic-update tick at sample period
# ``dt``) and the same ``npa.interp(t - tau, times[::-1], values[::-1])``
# lookup. The buffer is sized statically from ``max_delay_seconds`` so the
# block remains JIT/vmap-safe; ``tau`` is clipped to
# ``[0, max_delay_seconds]`` inside the output computation to guard
# against transient out-of-range values from upstream.
#
# Differentiability: gradient flows through both the data input
# (``npa.interp`` over ``values``) AND the delay input (``npa.interp``
# also differentiates w.r.t. its query coordinate via the standard
# linear-interp Jacobian — which is the marketing-wedge feature called
# out in T-107's "fit actuator delay from data" use case).
#
# Lives at the bottom of primitives.py after the LookupTableND section to
# keep the diff disjoint from concurrent T-120 / T-125 work in
# containers.py / event_gradient.py.
# ---------------------------------------------------------------------------


class VariableTransportDelay(LeafSystem):
    """Continuous-time variable transport (pure) delay.

    Implements ``y(t) = u(t - tau(t))`` where the delay ``tau`` is supplied
    as a runtime input signal (second input port) rather than as a static
    parameter. This is the T-107-followup-variable-tau extension to the
    fixed-delay :class:`TransportDelay` block (T-107 phase 1).

    Mechanism: identical to :class:`TransportDelay` — a periodic clock at
    period ``dt`` writes the most recent ``history_length`` ``(time, u)``
    pairs into a discrete-state ring buffer, and the (continuous-time)
    output port performs a linear interpolation over the buffer at
    ``t - clip(tau, 0, max_delay_seconds)``. The clip guards the
    interpolation against transient out-of-range delay values from
    upstream blocks; out-of-band ``tau`` is clamped (not raised) so that
    the block remains differentiable everywhere.

    Differentiability:

    * w.r.t. the data input ``u``: via ``npa.interp`` over ``values``,
      same as :class:`TransportDelay`.
    * w.r.t. the delay input ``tau``: under ``method="linear"`` (default,
      phase 3) via ``npa.interp``'s gradient w.r.t. its query
      coordinate — the standard linear-interp Jacobian, well defined
      except at sample boundaries where the gradient has a jump
      discontinuity. Pass ``method="pchip"`` (T-107 phase 4) to route
      through the T-106 backend's monotone cubic Hermite interpolant
      instead: smooth (C^1) gradient w.r.t. tau across every sample
      boundary, at the cost of one extra slope-array compute per
      output evaluation.

    The buffer is sized statically from ``max_delay_seconds``: at sample
    period ``dt`` you need at least ``ceil(max_delay_seconds / dt) + 1``
    slots; ``history_length`` defaults to
    ``max(8, ceil(max_delay_seconds / dt) + 4)``.

    Input ports:
        (0) The input signal ``u(t)``. Scalar or array.
        (1) The delay signal ``tau(t)`` in seconds. Runtime scalar in
            ``[0, max_delay_seconds]``; values outside that range are
            silently clamped.

    Output ports:
        (0) The delayed signal ``y(t) = u(t - tau(t))`` (or
            ``initial_output`` while ``t < tau(t)``).

    Parameters:
        dt: Sampling period for the history buffer. Smaller ``dt`` ⇒
            finer interpolation but a larger ring buffer to cover the
            same physical delay.
        max_delay_seconds: Upper bound on the runtime delay value. Used
            to size the ring buffer and to clip out-of-range ``tau``
            inputs. Static (compile-time).
        initial_output: Output value while ``t < tau(t)``. Default is
            0.0.
        history_length: Number of ``(time, value)`` pairs stored. Static
            (compile-time) — required for vmap/JIT-safe buffer sizing.
            If None, defaults to
            ``max(8, ceil(max_delay_seconds / dt) + 4)``.

    Notes:
        - Default-off / non-touched-block path is byte-equivalent: the
          existing :class:`TransportDelay` is untouched.
        - Buffer overflow (``tau > max_delay_seconds``) is clamped to
          ``max_delay_seconds`` rather than raised; this keeps the block
          differentiable but means the user is responsible for choosing
          a sufficiently large ``max_delay_seconds``.
        - The variable-tau interpolation runs once per output evaluation
          (continuous-time semantics). For workloads where the delay
          changes only at major-step granularity, sampling ``tau`` at the
          periodic update would be cheaper — deferred until profiling
          demands it.
        - For arbitrary array-shaped data signals the interpolation is
          applied elementwise via a static loop over the trailing axes
          (mirrors :class:`TransportDelay`).
    """

    class _BufferState(NamedTuple):
        # Newest sample at index 0; oldest at index -1. Reversing the
        # buffer gives a monotonically increasing time axis suitable for
        # ``npa.interp``.
        times: "Array"
        values: "Array"

    @parameters(
        static=["dt", "max_delay_seconds", "history_length", "method"],
        dynamic=["initial_output"],
    )
    def __init__(
        self,
        dt,
        max_delay_seconds,
        initial_output=0.0,
        history_length=None,
        method="linear",
        *args,
        dtype=None,
        **kwargs,
    ):
        # T-005 default-float64 / T-038a-followup-mixed-precision-cascade:
        # honor the active precision policy when no explicit dtype was
        # supplied.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype
        super().__init__(*args, **kwargs)

        if dt is None or float(dt) <= 0.0:
            raise BlockParameterError(
                message=(
                    f"VariableTransportDelay block {self.name!r} requires a "
                    f"positive sample period dt; got {dt!r}."
                ),
                parameter_name="dt",
            )
        # T-107 phase 4: interpolation method over the ring buffer.
        # ``"linear"`` is the phase-1 / phase-3 default (byte-equivalent);
        # ``"pchip"`` routes through the T-106 backend (monotone cubic
        # Hermite) for smooth gradients w.r.t. tau across sample
        # boundaries.
        if method not in ("linear", "pchip"):
            raise BlockParameterError(
                message=(
                    f"VariableTransportDelay block {self.name!r}: method "
                    f"must be 'linear' or 'pchip'; got {method!r}."
                ),
                parameter_name="method",
            )
        self.method = method
        try:
            max_delay_hint = float(max_delay_seconds)
        except (TypeError, ValueError):
            max_delay_hint = 0.0
        if max_delay_hint < 0.0:
            raise BlockParameterError(
                message=(
                    f"VariableTransportDelay block {self.name!r} requires "
                    f"max_delay_seconds >= 0; got {max_delay_seconds!r}."
                ),
                parameter_name="max_delay_seconds",
            )
        if history_length is None:
            history_length = max(8, int(np.ceil(max_delay_hint / dt)) + 4)
        if int(history_length) < 2:
            raise BlockParameterError(
                message=(
                    f"VariableTransportDelay block {self.name!r} requires "
                    f"history_length >= 2; got {history_length!r}."
                ),
                parameter_name="history_length",
            )

        self.dt = float(dt)
        self.max_delay_seconds = float(max_delay_hint)
        self.history_length = int(history_length)

        # Two input ports: (0) data ``u``, (1) delay ``tau``.
        self.declare_input_port()  # u
        self.declare_input_port()  # tau (variable delay)
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(
        self,
        dt,
        max_delay_seconds,
        initial_output,
        history_length=None,
        method=None,
    ):
        # ``history_length`` and ``method`` are static parameters resolved
        # at __init__ time; the framework still passes them for symmetry —
        # drop them here.
        del history_length, method  # noqa: F841

        initial_value = npa.asarray(initial_output)
        if self._dtype is not None:
            initial_value = initial_value.astype(self._dtype)
        self._signal_shape = tuple(initial_value.shape)

        # Pre-fill the times buffer with strictly increasing sentinels
        # below t=0 so that ``npa.interp(t - tau, times[::-1], ...)``
        # clamps to the oldest sample (== ``initial_output``) for any
        # query time before the first real sample has been written.
        sentinel_t0 = -self.dt * (self.history_length + 1) - 1.0
        times = sentinel_t0 + self.dt * np.arange(self.history_length, dtype=np.float64)
        # Newest first: reverse so position 0 is the largest sentinel.
        times = times[::-1].copy()
        if self._dtype is not None:
            times = times.astype(self._dtype)

        values = npa.broadcast_to(
            initial_value, (self.history_length, *self._signal_shape)
        )

        default_state = self._BufferState(
            times=npa.asarray(times), values=npa.asarray(values)
        )
        self.declare_discrete_state(default_value=default_state, as_array=False)

        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )

        # Output reads time + discrete-state buffer + the ``tau`` input
        # port; mark ``requires_inputs=True`` so the framework wires up
        # both input ports for the lookup.
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[
                DependencyTicket.xd,
                DependencyTicket.time,
                self.input_ports[1].ticket,
            ],
            requires_inputs=True,
            default_value=initial_value,
        )

    def reset_default_values(
        self,
        dt=None,
        max_delay_seconds=None,
        initial_output=None,
        history_length=None,
        method=None,
    ):
        # Mirror TransportDelay's pattern: rebuild defaults if the
        # dynamic ``initial_output`` changes between calls.
        del dt, max_delay_seconds, history_length, method  # noqa: F841

        if initial_output is None:
            return
        initial_value = npa.asarray(initial_output)
        if self._dtype is not None:
            initial_value = initial_value.astype(self._dtype)
        self._signal_shape = tuple(initial_value.shape)

        sentinel_t0 = -self.dt * (self.history_length + 1) - 1.0
        times = sentinel_t0 + self.dt * np.arange(self.history_length, dtype=np.float64)
        times = times[::-1].copy()
        if self._dtype is not None:
            times = times.astype(self._dtype)

        values = npa.broadcast_to(
            initial_value, (self.history_length, *self._signal_shape)
        )
        default_state = self._BufferState(
            times=npa.asarray(times), values=npa.asarray(values)
        )
        self.configure_discrete_state_default_value(
            default_value=default_state, as_array=False
        )
        self.configure_output_port_default_value(
            self._output_port_idx, initial_value
        )

    def _update(self, time, state, *inputs, **_params):
        # Only the data input (port 0) is written into the ring buffer;
        # the delay input (port 1) is consumed at output evaluation time.
        u = inputs[0]
        if self._dtype is not None:
            u = npa.asarray(u).astype(self._dtype)
        buf = state.discrete_state
        new_times = npa.roll(buf.times, shift=1, axis=0).at[0].set(time)
        new_values = npa.roll(buf.values, shift=1, axis=0).at[0].set(u)
        return self._BufferState(times=new_times, values=new_values)

    def _output(self, time, state, *inputs, **params):
        buf = state.discrete_state
        # Reverse so the time axis is monotonically increasing for
        # ``npa.interp``: index 0 is oldest, index -1 is newest.
        xp = buf.times[::-1]
        fp = buf.values[::-1]
        # Variable-tau: read the delay from the second input port and
        # clamp it into ``[0, max_delay_seconds]``. Out-of-band values
        # are silently clipped (rather than raised) so the block stays
        # differentiable everywhere.
        tau_raw = inputs[1]
        tau = npa.clip(tau_raw, 0.0, self.max_delay_seconds)
        initial_output = params["initial_output"]

        query_t = time - tau

        # T-107 phase 4: dispatch on interpolation method. Linear stays
        # on ``npa.interp`` for byte-equivalence with phase 3 (the
        # established default); PCHIP routes through the T-106 backend
        # for smooth gradients w.r.t. tau across sample boundaries.
        if self.method == "pchip":
            from .lookup_table import interp_1d as _interp_1d

            def _scalar_interp(values_1d):
                # ``interp_1d`` returns a JAX array; the surrounding
                # npa context handles eager-numpy callers.
                return _interp_1d(query_t, xp, values_1d, method="pchip")
        else:
            def _scalar_interp(values_1d):
                return npa.interp(query_t, xp, values_1d)

        if len(self._signal_shape) == 0:
            y = _scalar_interp(fp)
        else:
            # The 1-D interpolator only handles 1-D ``fp``; statically
            # loop over the trailing axes (shape known at trace time).
            flat_fp = fp.reshape((self.history_length, -1))
            ys = [_scalar_interp(flat_fp[:, i]) for i in range(flat_fp.shape[1])]
            y = npa.stack(ys).reshape(self._signal_shape)

        # Hold the initial output before the first physical sample is
        # available; explicitly gate on ``time < tau`` to keep semantics
        # robust to dtype/shape pre-fill quirks.
        y = npa.where(time < tau, npa.asarray(initial_output), y)
        if self._dtype is not None:
            y = npa.asarray(y).astype(self._dtype)
        return y
