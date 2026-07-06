# SPDX-License-Identifier: MIT

"""Clipping, saturation, dead zones, and rate-limit blocks."""

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
    "DeadZone",
    "DeadZoneInverse",
    "Saturate",
    "SoftSaturate",
    "Quantizer",
    "RateLimiter",
    "SoftRateLimiter",
    "Backlash",
    "Stop",
    "soft_saturate",
    "soft_dead_zone",
]



class DeadZone(FeedthroughBlock):
    """Generates zero output within a specified range.

    Applies the following function:
    ```
             [ input,       input < -half_range
    output = | 0,           -half_range <= input <= half_range
             [ input        input > half_range
    ```

    Parameters:
        half_range: The range of the dead zone.  Must be > 0.
        mode: ``"hard"`` (default, byte-equivalent to legacy behavior) or
            ``"smooth"``. Smooth mode replaces the discontinuous gate with
            a sigmoid-blended kernel so gradients flow through the dead
            zone region; the output then has no discontinuity at
            ``|x| = half_range`` and no zero-crossing events are declared.
        sharpness: Positive scalar; default ``10.0``. Only used in smooth
            mode. Larger values give a tighter approximation to the hard
            dead zone (with smaller gradients inside the band).
        output_shifted: ``False`` (default, byte-equivalent to legacy
            behavior) or ``True``. When ``True``, the hard-mode output
            outside the band is shifted by ``half_range * sign(input)`` so
            the output is continuous across the band boundary (slope 1
            outside, value ``0`` at the boundary). The ``False`` form keeps
            the legacy "Coulomb friction" semantics where the output jumps
            at the band boundary. The ``smooth`` mode already produces a
            continuous output and is unaffected by this flag.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The input signal modified by the dead zone.

    Events:
        An event is triggered when the signal enters or exits the dead zone
        in either direction (hard mode only).

    T-115-followup-deadzone-backlash:
        The ``mode`` kwarg unifies a smooth (differentiable) variant.
        ``mode="hard"`` (default) is byte-equivalent to the legacy
        behavior, including zero-crossing event declaration.
        ``mode="smooth"`` dispatches to a sigmoid-blended formula
        (see :func:`soft_dead_zone`) and does *not* declare zero-crossing
        events.

    T-115-followup-deadzone-bilinear:
        The ``output_shifted`` kwarg toggles between the legacy
        Coulomb-style hard dead-zone (default, output jumps at the band
        boundary) and the shifted-output variant
        (continuous across the band boundary). Default ``False`` keeps
        the block byte-equivalent to phase 1.
    """

    @parameters(dynamic=["half_range", "sharpness"], static=["mode", "output_shifted"])
    def __init__(
        self,
        half_range=1.0,
        mode="hard",
        sharpness=10.0,
        output_shifted=False,
        **kwargs,
    ):
        if mode not in ("hard", "smooth"):
            raise BlockParameterError(
                message=(
                    f"DeadZone block: mode must be 'hard' or 'smooth', "
                    f"got {mode!r}."
                ),
                parameter_name="mode",
            )
        super().__init__(self._dead_zone, **kwargs)
        if half_range <= 0:
            raise BlockParameterError(
                message=f"DeadZone block {self.name} has invalid half_range {half_range}. Must be > 0.",
                system=self,
                parameter_name="half_range",
            )
        if mode == "smooth" and sharpness <= 0:
            raise BlockParameterError(
                message=(
                    f"DeadZone block {self.name}: mode='smooth' requires "
                    f"sharpness > 0, got {sharpness}."
                ),
                system=self,
                parameter_name="sharpness",
            )
        if not isinstance(output_shifted, bool):
            raise BlockParameterError(
                message=(
                    f"DeadZone block {self.name}: output_shifted must be a "
                    f"bool, got {output_shifted!r}."
                ),
                system=self,
                parameter_name="output_shifted",
            )
        self.mode = mode
        self.output_shifted = output_shifted

    def initialize(
        self, half_range, mode="hard", sharpness=10.0, output_shifted=False
    ):
        if mode != self.mode:
            raise ValueError(
                "DeadZone: mode cannot be changed after initialization"
            )
        if output_shifted != self.output_shifted:
            raise ValueError(
                "DeadZone: output_shifted cannot be changed after initialization"
            )

    def _dead_zone(self, x, **params):
        if self.mode == "smooth":
            # T-115-followup-deadzone-backlash: differentiable variant.
            return soft_dead_zone(x, params["half_range"], params["sharpness"])
        hr = params["half_range"]
        if self.output_shifted:
            # T-115-followup-deadzone-bilinear: shifted-output variant.
            # Outside the band the output is ``x - hr*sign(x)``;
            # this yields slope 1 with value 0 at ``|x| = hr`` so the
            # output is continuous across the band boundary.
            return npa.where(abs(x) < hr, x * 0, x - hr * npa.sign(x))
        # Legacy Coulomb-style: output jumps at the band boundary.
        return npa.where(abs(x) < hr, x * 0, x)

    def _lower_limit_event_value(self, _time, _state, *inputs, **params):
        (u,) = inputs
        return u + params["half_range"]

    def _upper_limit_event_value(self, _time, _state, *inputs, **params):
        (u,) = inputs
        return u - params["half_range"]

    def initialize_static_data(self, context):
        # Add zero-crossing events so ODE solvers can't try to integrate
        # through a discontinuity.
        #
        # T-115-followup-deadzone-backlash: smooth mode has no
        # discontinuity, so we never declare zero-crossing events for it.
        if (
            self.mode == "hard"
            and not self.has_zero_crossing_events
            and (self.output_ports[0])
        ):
            self.declare_zero_crossing(
                self._lower_limit_event_value, direction="crosses_zero"
            )
            self.declare_zero_crossing(
                self._upper_limit_event_value, direction="crosses_zero"
            )

        return super().initialize_static_data(context)



_QUANTIZER_MODES = ("round", "floor", "ceil", "trunc")


class Quantizer(FeedthroughBlock):
    """Discritize the input signal into a set of discrete values.

    Given an input signal ``u`` and a resolution ``interval``, this block
    quantizes the input signal onto the integer multiples of ``interval``.
    The output signal is ``y = interval * f(u / interval)`` where ``f`` is
    selected by ``mode``:

    - ``"round"`` (default): round-half-to-even (banker's rounding,
      IEEE-754 default). Byte-equivalent with the phase-1 implementation
      (which used ``npa.round`` unconditionally).
    - ``"floor"``: round toward -inf (truncation in many DSP impls).
    - ``"ceil"``: round toward +inf.
    - ``"trunc"``: round toward zero (chops the fractional part regardless
      of sign).

    Quantization is non-differentiable: the output is piecewise-constant
    with measure-zero jumps. The block wraps the rounded result in
    :func:`jax.lax.stop_gradient` (JAX backend only) so JAX always sees a
    zero gradient through the block. This both matches the underlying
    mathematical reality and prevents spurious gradient leakage if any
    backend ever provides a smoothed surrogate for ``round``/``floor``/
    ``ceil``/``trunc``. Under the numpy backend the helper is the
    identity, preserving dtype/value byte-equivalence.

    Input ports:
        (0) The continuous input signal. In most cases, should be scaled to the range
            ``[0, interval]``.

    Output ports:
        (0) The quantized output signal, on the same scale as the input signal.

    Parameters:
        interval:
            The quantization step size — output values are integer
            multiples of ``interval``.
        mode:
            One of ``"round"``, ``"floor"``, ``"ceil"``, ``"trunc"``.
            Default ``"round"``.
    """

    @parameters(dynamic=["interval"])
    def __init__(self, interval, mode="round", *args, **kwargs):
        if mode not in _QUANTIZER_MODES:
            raise BlockParameterError(
                message=(
                    f"Quantizer mode must be one of {_QUANTIZER_MODES}, "
                    f"got {mode!r}."
                ),
            )
        self._mode = mode
        if mode == "round":
            _round_fn = npa.round
        elif mode == "floor":
            _round_fn = npa.floor
        elif mode == "ceil":
            _round_fn = npa.ceil
        else:  # mode == "trunc"
            _round_fn = npa.trunc

        def _op(x, interval):
            return _stop_gradient(interval * _round_fn(x / interval))

        super().__init__(_op, *args, **kwargs)

    def initialize(self, interval):
        pass



class RateLimiter(LeafSystem):
    """Limit the time derivative of the block output.

    Given an input signal `u` computes the derivative of the output signal as:
    ```
        y_rate = (u(t) - y(Tprev))/(t - Tprev)
    ```
    Where Tprev is the last time the block was called for output update.

    When y_rate is greater than the upper_limit, the output is:
    ```
        y(t) = (t - Tprev)*upper_limit + y(Tprev)
    ```

    When y_rate is less than the lower_limit, the output is:
    ```
        y(t) = (t - Tprev)*lower_limit + y(Tprev)
    ```

    If the lower_limit is greater than the upper_limit, and both
    are being violated, the upper_limit takes precedence.

    Optionally, the block can also be configured with "dynamic" limits, which will
    add input ports for time-varying upper and lower limits.

    Presently, the block is constrainted to periodic updates.

    Input ports:
        (0) The input signal.
        (1) The upper limit, if dynamic limits are enabled.
        (2) The lower limit, if dynamic limits are enabled. (Will be indexed as 1 if
            dynamic upper limits are not enabled.)

    Output ports:
        (0) The rate limited output signal.

    Parameters:
        upper_limit:
            The upper limit of the input signal.  Default is `np.inf`.
        enable_dynamic_upper_limit:
            If True, then the upper limit can be set by an external signal. Default
            is False.
        lower_limit:
            The lower limit of the input signal.  Default is `-np.inf`.
        enable_dynamic_lower_limit:
            If True, then the lower limit can be set by an external signal. Default
            is False.

    T-115-followup-mode-flag:
        The ``mode`` kwarg unifies the smooth (differentiable) variant
        previously exposed as :class:`SoftRateLimiter`. ``mode="hard"``
        (default) is byte-equivalent to the legacy behavior.
        ``mode="smooth"`` replaces the inner per-step delta clip with
        :func:`soft_saturate` so gradients flow through active rate
        limiting. Smooth mode requires finite (static) ``upper_limit`` /
        ``lower_limit`` and ``sharpness > 0`` (defaults to ``10.0``).
    """

    class DiscreteStateType(NamedTuple):
        y_prev: Array
        t_prev: float

    @parameters(
        static=[
            "dt",
            "enable_dynamic_upper_limit",
            "enable_dynamic_lower_limit",
            "mode",
        ],
        dynamic=["upper_limit", "lower_limit", "sharpness"],
    )
    def __init__(
        self,
        dt,
        upper_limit=np.inf,
        enable_dynamic_upper_limit=False,
        lower_limit=-np.inf,
        enable_dynamic_lower_limit=False,
        mode="hard",
        sharpness=10.0,
        **kwargs,
    ):
        if mode not in ("hard", "smooth"):
            raise BlockParameterError(
                message=(
                    f"RateLimiter block: mode must be 'hard' or 'smooth', "
                    f"got {mode!r}."
                ),
                parameter_name="mode",
            )
        super().__init__(**kwargs)
        self.primary_input_index = self.declare_input_port()
        self.enable_dynamic_upper_limit = enable_dynamic_upper_limit
        self.enable_dynamic_lower_limit = enable_dynamic_lower_limit
        self.dt = dt
        self.mode = mode

        if enable_dynamic_upper_limit:
            # If dynamic limit, simply ignore the static limit
            self.upper_limit_index = self.declare_input_port()

        if enable_dynamic_lower_limit:
            # If dynamic limit, simply ignore the static limit
            self.lower_limit_index = self.declare_input_port()

        # Smooth-mode validation: needs finite static limits and positive
        # sharpness (matches SoftRateLimiter contract).
        if mode == "smooth":
            if (
                not enable_dynamic_upper_limit
                and not np.isfinite(upper_limit)
            ):
                raise BlockParameterError(
                    message=(
                        f"RateLimiter block {self.name}: mode='smooth' requires "
                        f"finite upper_limit, got {upper_limit}."
                    ),
                    system=self,
                    parameter_name="upper_limit",
                )
            if (
                not enable_dynamic_lower_limit
                and not np.isfinite(lower_limit)
            ):
                raise BlockParameterError(
                    message=(
                        f"RateLimiter block {self.name}: mode='smooth' requires "
                        f"finite lower_limit, got {lower_limit}."
                    ),
                    system=self,
                    parameter_name="lower_limit",
                )
            if sharpness <= 0:
                raise BlockParameterError(
                    message=(
                        f"RateLimiter block {self.name}: mode='smooth' requires "
                        f"sharpness > 0, got {sharpness}."
                    ),
                    system=self,
                    parameter_name="sharpness",
                )

        self.output_index = self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
        )

    def initialize(
        self,
        upper_limit=np.inf,
        enable_dynamic_upper_limit=False,
        lower_limit=-np.inf,
        enable_dynamic_lower_limit=False,
        mode="hard",
        sharpness=10.0,
        dt=None,
    ):
        if enable_dynamic_upper_limit != self.enable_dynamic_upper_limit:
            raise ValueError(
                "RateLimiter: enable_dynamic_upper_limit cannot be changed after initialization"
            )
        if enable_dynamic_lower_limit != self.enable_dynamic_lower_limit:
            raise ValueError(
                "RateLimiter: enable_dynamic_lower_limit cannot be changed after initialization"
            )
        if mode != self.mode:
            raise ValueError(
                "RateLimiter: mode cannot be changed after initialization"
            )

    def _output(self, time, state, *inputs, **params):
        y_prev = state.cache[self.output_index]

        u = inputs[self.primary_input_index]

        t_diff = self.dt

        ulim = (
            inputs[self.upper_limit_index]
            if self.enable_dynamic_upper_limit
            else params["upper_limit"]
        )
        llim = (
            inputs[self.lower_limit_index]
            if self.enable_dynamic_lower_limit
            else params["lower_limit"]
        )

        if self.mode == "smooth":
            # T-115-followup-mode-flag: smooth per-step delta clip via
            # soft_saturate (matches SoftRateLimiter behavior).
            delta = u - y_prev
            delta_lo = t_diff * llim
            delta_hi = t_diff * ulim
            return y_prev + soft_saturate(
                delta, delta_lo, delta_hi, params["sharpness"]
            )

        y_rate = (u - y_prev) / t_diff

        y_ulim = t_diff * ulim + y_prev
        y_llim = t_diff * llim + y_prev
        y_tmp = npa.where(y_rate < llim, y_llim, u)
        y = npa.where(y_rate > ulim, y_ulim, y_tmp)

        return y

    def initialize_static_data(self, context):
        """Infer the size and dtype of the internal states"""
        # If building as part of a subsystem, this may not be fully connected yet.
        # That's fine, as long as it is connected by root context creation time.
        # This probably isn't a good long-term solution:
        #   see https://jaxonomy.atlassian.net/browse/WC-51
        try:
            u = self.eval_input(context)
            self._default_cache[self.output_index] = u
            local_context = context[self.system_id].with_discrete_state(u)
            local_context = local_context.with_cached_value(self.output_index, u)
            context = context.with_subcontext(self.system_id, local_context)

        except UpstreamEvalError:
            logger.debug(
                "RateLimiter.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
        return context


class SoftRateLimiter(LeafSystem):
    """Smooth (differentiable) rate limiter.

    Drop-in differentiable variant of :class:`RateLimiter`. Identical
    discrete update semantics, except the inner hard ``clip`` on the
    desired step ``(u - y_prev)`` is replaced by a smooth saturation so
    gradients flow through the limiter even when it is actively limiting.

    The smooth clip is implemented via :func:`soft_saturate` and recovers
    the hard rate limiter as ``sharpness -> inf``.

    Parameters mirror :class:`RateLimiter` plus:
        sharpness:
            Scalar > 0 controlling how sharply the smooth saturation
            transitions at the rate limits. Larger ``sharpness`` -> closer
            to the hard rate limiter. Default ``10.0``.

    See :class:`RateLimiter` for the (non-smoothed) reference behavior.
    """

    class DiscreteStateType(NamedTuple):
        y_prev: Array
        t_prev: float

    @parameters(
        static=["dt", "enable_dynamic_upper_limit", "enable_dynamic_lower_limit"],
        dynamic=["upper_limit", "lower_limit", "sharpness"],
    )
    def __init__(
        self,
        dt,
        upper_limit=np.inf,
        enable_dynamic_upper_limit=False,
        lower_limit=-np.inf,
        enable_dynamic_lower_limit=False,
        sharpness=10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.primary_input_index = self.declare_input_port()
        self.enable_dynamic_upper_limit = enable_dynamic_upper_limit
        self.enable_dynamic_lower_limit = enable_dynamic_lower_limit
        self.dt = dt

        if enable_dynamic_upper_limit:
            self.upper_limit_index = self.declare_input_port()

        if enable_dynamic_lower_limit:
            self.lower_limit_index = self.declare_input_port()

        self.output_index = self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
        )

    def initialize(
        self,
        upper_limit=np.inf,
        enable_dynamic_upper_limit=False,
        lower_limit=-np.inf,
        enable_dynamic_lower_limit=False,
        sharpness=10.0,
        dt=None,
    ):
        if enable_dynamic_upper_limit != self.enable_dynamic_upper_limit:
            raise ValueError(
                "SoftRateLimiter: enable_dynamic_upper_limit cannot be changed after initialization"
            )
        if enable_dynamic_lower_limit != self.enable_dynamic_lower_limit:
            raise ValueError(
                "SoftRateLimiter: enable_dynamic_lower_limit cannot be changed after initialization"
            )

    def _output(self, time, state, *inputs, **params):
        y_prev = state.cache[self.output_index]
        u = inputs[self.primary_input_index]
        t_diff = self.dt

        ulim = (
            inputs[self.upper_limit_index]
            if self.enable_dynamic_upper_limit
            else params["upper_limit"]
        )
        llim = (
            inputs[self.lower_limit_index]
            if self.enable_dynamic_lower_limit
            else params["lower_limit"]
        )
        k = params["sharpness"]

        # Smoothly clip the per-step delta in y.
        delta = u - y_prev
        delta_lo = t_diff * llim
        delta_hi = t_diff * ulim
        delta_clipped = soft_saturate(delta, delta_lo, delta_hi, k)
        return y_prev + delta_clipped

    def initialize_static_data(self, context):
        try:
            u = self.eval_input(context)
            self._default_cache[self.output_index] = u
            local_context = context[self.system_id].with_discrete_state(u)
            local_context = local_context.with_cached_value(self.output_index, u)
            context = context.with_subcontext(self.system_id, local_context)
        except UpstreamEvalError:
            logger.debug(
                "SoftRateLimiter.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
        return context



class Saturate(LeafSystem):
    """Clip the input signal to a specified range.

    Given an input signal `u` and upper and lower limits `ulim` and `llim`,
    the output signal is:
    ```
        y = max(llim, min(ulim, u))
    ```
    where `max` and `min` are the element-wise maximum and minimum functions.
    This is equivalent to `y = clip(u, llim, ulim)`.

    Optionally, the block can also be configured with "dynamic" limits, which will
    add input ports for time-varying upper and lower limits.

    Input ports:
        (0) The input signal.
        (1) The upper limit, if dynamic limits are enabled.
        (2) The lower limit, if dynamic limits are enabled. (Will be indexed as 1 if
            dynamic upper limits are not enabled.)

    Output ports:
        (0) The clipped output signal.

    Parameters:
        upper_limit:
            The upper limit of the input signal.  Default is `np.inf`.
        enable_dynamic_upper_limit:
            If True, then the upper limit can be set by an external signal. Default
            is False.
        lower_limit:
            The lower limit of the input signal.  Default is `-np.inf`.
        enable_dynamic_lower_limit:
            If True, then the lower limit can be set by an external signal. Default
            is False.
        limit:
            T-115-followup-saturate-symmetric-kwarg: shorthand for the
            symmetric case. ``Saturate(limit=L)`` expands to
            ``Saturate(upper_limit=+L, lower_limit=-L)``. Mutually
            exclusive with explicit ``upper_limit`` / ``lower_limit``
            and with the dynamic-limit flags. ``L`` must be a positive
            finite scalar.

    Events:
        The block will trigger an event when the input signal crosses either the upper
        or lower limit.  For example, if the block is configured with static upper and
        lower limits and the input signal crosses the upper limit, then a zero-crossing
        event will be triggered.

    T-115-followup-mode-flag:
        The ``mode`` kwarg unifies the smooth (differentiable) variant
        previously exposed as :class:`SoftSaturate`. ``mode="hard"``
        (default) is byte-equivalent to the legacy behavior, including
        zero-crossing event declaration. ``mode="smooth"`` dispatches to
        :func:`soft_saturate` and does *not* declare zero-crossing events
        (the smooth output has no discontinuity for the solver to catch).
        The smooth path requires finite ``upper_limit`` / ``lower_limit``
        and a ``sharpness > 0`` (defaults to ``10.0``).
    """

    @parameters(
        static=["enable_dynamic_upper_limit", "enable_dynamic_lower_limit", "mode"],
        dynamic=["upper_limit", "lower_limit", "sharpness"],
    )
    def __init__(
        self,
        upper_limit=None,
        enable_dynamic_upper_limit=False,
        lower_limit=None,
        enable_dynamic_lower_limit=False,
        mode="hard",
        sharpness=10.0,
        **kwargs,
    ):
        # Note: T-115-followup-saturate-symmetric-kwarg's ``limit=L``
        # convenience shorthand is handled by the post-class
        # ``_inject_limit_kwarg`` wrapper below, which runs BEFORE
        # the ``@parameters`` decorator sees the call so that
        # ``upper_limit`` / ``lower_limit`` arrive populated.
        if mode not in ("hard", "smooth"):
            raise BlockParameterError(
                message=(
                    f"Saturate block: mode must be 'hard' or 'smooth', "
                    f"got {mode!r}."
                ),
                parameter_name="mode",
            )
        super().__init__(**kwargs)
        self.primary_input_index = self.declare_input_port()
        self.enable_dynamic_upper_limit = enable_dynamic_upper_limit
        self.enable_dynamic_lower_limit = enable_dynamic_lower_limit
        self.mode = mode

        prerequisites_of_calc = [self.input_ports[self.primary_input_index].ticket]

        if enable_dynamic_upper_limit:
            # If dynamic limit, simply ignore the static limit
            self.upper_limit_index = self.declare_input_port()
            prerequisites_of_calc.append(
                self.input_ports[self.upper_limit_index].ticket
            )
        else:
            if upper_limit is None:
                upper_limit = np.inf

        if enable_dynamic_lower_limit:
            # If dynamic limit, simply ignore the static limit
            self.lower_limit_index = self.declare_input_port()
            prerequisites_of_calc.append(
                self.input_ports[self.lower_limit_index].ticket
            )
        else:
            if lower_limit is None:
                lower_limit = -np.inf

        # Smooth-mode validation: needs finite limits and positive
        # sharpness. We can only validate static (non-dynamic) limits at
        # construction time.
        if mode == "smooth":
            if (
                not enable_dynamic_upper_limit
                and not np.isfinite(upper_limit)
            ):
                raise BlockParameterError(
                    message=(
                        f"Saturate block {self.name}: mode='smooth' requires "
                        f"finite upper_limit, got {upper_limit}. Use "
                        "mode='hard' for unbounded sides."
                    ),
                    system=self,
                    parameter_name="upper_limit",
                )
            if (
                not enable_dynamic_lower_limit
                and not np.isfinite(lower_limit)
            ):
                raise BlockParameterError(
                    message=(
                        f"Saturate block {self.name}: mode='smooth' requires "
                        f"finite lower_limit, got {lower_limit}. Use "
                        "mode='hard' for unbounded sides."
                    ),
                    system=self,
                    parameter_name="lower_limit",
                )
            if sharpness <= 0:
                raise BlockParameterError(
                    message=(
                        f"Saturate block {self.name}: mode='smooth' requires "
                        f"sharpness > 0, got {sharpness}."
                    ),
                    system=self,
                    parameter_name="sharpness",
                )

        self.declare_output_port(
            self._compute_output, prerequisites_of_calc=prerequisites_of_calc
        )

    def initialize(
        self,
        upper_limit=None,
        enable_dynamic_upper_limit=False,
        lower_limit=None,
        enable_dynamic_lower_limit=False,
        mode="hard",
        sharpness=10.0,
    ):
        if enable_dynamic_lower_limit != self.enable_dynamic_lower_limit:
            raise ValueError(
                "enable_dynamic_lower_limit must be the same as the value passed to the constructor"
            )
        if enable_dynamic_upper_limit != self.enable_dynamic_upper_limit:
            raise ValueError(
                "enable_dynamic_upper_limit must be the same as the value passed to the constructor"
            )
        if mode != self.mode:
            raise ValueError(
                "Saturate: mode cannot be changed after initialization"
            )

    def _lower_limit_event_value(self, _time, _state, *inputs, **params):
        u = inputs[self.primary_input_index]
        if self.enable_dynamic_lower_limit:
            lim = inputs[self.lower_limit_index]
        else:
            lim = params["lower_limit"]
        return u - lim

    def _upper_limit_event_value(self, _time, _state, *inputs, **params):
        u = inputs[self.primary_input_index]
        if self.enable_dynamic_upper_limit:
            lim = inputs[self.upper_limit_index]
        else:
            lim = params["upper_limit"]
        return u - lim

    def _compute_output(self, _time, _state, *inputs, **params):
        u = inputs[self.primary_input_index]

        ulim = (
            inputs[self.upper_limit_index]
            if self.enable_dynamic_upper_limit
            else params["upper_limit"]
        )
        llim = (
            inputs[self.lower_limit_index]
            if self.enable_dynamic_lower_limit
            else params["lower_limit"]
        )

        if self.mode == "smooth":
            # T-115-followup-mode-flag: dispatch to differentiable variant.
            return soft_saturate(u, llim, ulim, params["sharpness"])
        return npa.clip(u, llim, ulim)

    def initialize_static_data(self, context):
        # Add zero-crossing events so ODE solvers can't try to integrate
        # through a discontinuity. For efficiency, only do this if the output
        # is fed to an ODE block.
        #
        # T-115-followup-mode-flag: smooth mode has no discontinuity, so we
        # never declare zero-crossing events for it.
        if (
            self.mode == "hard"
            and not self.has_zero_crossing_events
            and is_discontinuity(self.output_ports[0])
        ):
            self.declare_zero_crossing(
                self._lower_limit_event_value,
                direction="positive_then_non_positive",
                name="llim",
            )
            self.declare_zero_crossing(
                self._upper_limit_event_value,
                direction="negative_then_non_negative",
                name="ulim",
            )

        return super().initialize_static_data(context)


# ---------------------------------------------------------------------
# T-115-followup-saturate-symmetric-kwarg
#
# The ``Saturate(limit=L)`` shorthand is implemented as a *post-class*
# wrapper around the ``@parameters``-decorated ``__init__`` because the
# decorator reads parameter values directly from ``kwargs`` *before*
# the user's body runs (see :func:`_get_params` in
# ``framework/system_decorators.py``). Translating ``limit`` to
# ``upper_limit`` / ``lower_limit`` inside the body would leave the
# decorator's parameter snapshot pointing at ``None``, which then
# trips ``KeyError: 'upper_limit'`` in ``_compute_output``. Wrapping
# the already-decorated ``__init__`` lets us mutate kwargs *first*,
# so the decorator sees the expanded form.
# ---------------------------------------------------------------------


def _inject_saturate_limit_kwarg(cls):
    """Wrap ``cls.__init__`` to expand ``limit=L`` into
    ``upper_limit=+L, lower_limit=-L`` before the decorated init runs."""
    _orig_init = cls.__init__

    @wraps(_orig_init)
    def _init_with_limit(self, *args, limit=None, **kwargs):
        if limit is not None:
            if "upper_limit" in kwargs or "lower_limit" in kwargs:
                raise BlockParameterError(
                    message=(
                        f"Saturate: ``limit={limit!r}`` is the symmetric "
                        f"shorthand and cannot be combined with explicit "
                        f"``upper_limit`` / ``lower_limit``. Pass either "
                        f"``limit=L`` (symmetric) or ``upper_limit=U, "
                        f"lower_limit=L`` (asymmetric)."
                    ),
                    parameter_name="limit",
                )
            if (
                kwargs.get("enable_dynamic_upper_limit", False)
                or kwargs.get("enable_dynamic_lower_limit", False)
            ):
                raise BlockParameterError(
                    message=(
                        "Saturate: ``limit=`` cannot be combined with "
                        "``enable_dynamic_upper_limit=True`` / "
                        "``enable_dynamic_lower_limit=True`` — pass the "
                        "limits through input ports instead, or use "
                        "static ``upper_limit`` / ``lower_limit``."
                    ),
                    parameter_name="limit",
                )
            if not np.isfinite(limit) or limit <= 0:
                raise BlockParameterError(
                    message=(
                        f"Saturate: ``limit`` must be a positive finite "
                        f"scalar (got {limit!r}). Pass an asymmetric "
                        f"``upper_limit`` / ``lower_limit`` for unbounded "
                        f"or negative-spanning ranges."
                    ),
                    parameter_name="limit",
                )
            kwargs["upper_limit"] = float(limit)
            kwargs["lower_limit"] = -float(limit)
        _orig_init(self, *args, **kwargs)

    # @wraps(_orig_init) above sets __wrapped__ so inspect.signature() follows
    # through to the real parameter names (upper_limit / lower_limit / ...),
    # which the JSON block factory relies on to pass them to the constructor.
    cls.__init__ = _init_with_limit
    return cls


Saturate = _inject_saturate_limit_kwarg(Saturate)


def soft_saturate(u, lower, upper, sharpness=10.0):
    """Smooth (differentiable) saturation between ``lower`` and ``upper``.

    Approximates ``npa.clip(u, lower, upper)`` with a tanh-based smooth
    clamp (per the T-115 spec):

    ::

        mid  = (lower + upper) / 2
        span = upper - lower
        y    = mid + (span / 2) * tanh(sharpness * (u - mid) / span)

    With ``sharpness = 10.0`` and a unit span this gives ~99% saturation
    by ``|u - mid| = span``, which is the design default.

    Properties:
      * As ``sharpness -> inf`` the function converges to a hard ``clip``.
      * Strictly monotonically increasing in ``u`` (analytically; in
        finite precision the slope underflows to zero very far from
        ``mid`` because tanh saturates exponentially).
      * Has a positive derivative across and inside the bound region --
        gradients flow through saturation, which is the whole reason
        this exists.
      * Requires *finite* ``lower`` / ``upper``; for unbounded sides use
        the hard :class:`Saturate` block.

    Args:
        u: Input array.
        lower: Lower limit (scalar or broadcastable array).
        upper: Upper limit (scalar or broadcastable array).
        sharpness: Positive scalar; default ``10.0``. Larger values give
            a tighter approximation to ``clip`` but smaller (and faster
            vanishing) gradients outside the bounds.

    Returns:
        Smoothly saturated array, same shape as ``u``.
    """
    mid = (lower + upper) / 2.0
    span = upper - lower
    return mid + (span / 2.0) * npa.tanh(sharpness * (u - mid) / span)


class SoftSaturate(FeedthroughBlock):
    """Smooth (differentiable) saturation block.

    Drop-in differentiable variant of :class:`Saturate` that uses
    :func:`soft_saturate` instead of ``npa.clip``. The original hard
    :class:`Saturate` block is unchanged.

    Why a separate block: the standard :class:`Saturate` block returns
    ``npa.clip(u, lo, hi)``, whose gradient is exactly zero outside the
    bounds. That kills gradient signal in any optimization that drives
    the input past the limits. ``SoftSaturate`` keeps gradient flow alive
    so e.g. trajectory optimization through actuator limits actually
    converges.

    See :func:`soft_saturate` for the formula. Unlike :class:`Saturate`,
    this block does *not* declare zero-crossing events (it has no
    discontinuity to catch).

    Parameters:
        upper_limit: Upper limit; default ``1.0``. Must be finite.
        lower_limit: Lower limit; default ``0.0``. Must be finite and
            strictly less than ``upper_limit``.
        sharpness: Smoothing knob, > 0; default ``10.0``. As
            ``sharpness -> inf`` this approaches the hard
            :class:`Saturate` block.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The smoothly-saturated output signal.
    """

    @parameters(dynamic=["upper_limit", "lower_limit", "sharpness"])
    def __init__(
        self,
        lower_limit=0.0,
        upper_limit=1.0,
        sharpness=10.0,
        **kwargs,
    ):
        super().__init__(self._soft_saturate, **kwargs)
        if not np.isfinite(lower_limit) or not np.isfinite(upper_limit):
            raise BlockParameterError(
                message=(
                    f"SoftSaturate block {self.name} requires finite "
                    f"lower_limit/upper_limit, got lower={lower_limit}, "
                    f"upper={upper_limit}. Use the hard Saturate block "
                    "for unbounded sides."
                ),
                system=self,
                parameter_name="lower_limit",
            )
        if upper_limit <= lower_limit:
            raise BlockParameterError(
                message=(
                    f"SoftSaturate block {self.name}: upper_limit "
                    f"({upper_limit}) must be > lower_limit ({lower_limit})."
                ),
                system=self,
                parameter_name="upper_limit",
            )
        if sharpness <= 0:
            raise BlockParameterError(
                message=(
                    f"SoftSaturate block {self.name}: sharpness must be "
                    f"> 0, got {sharpness}."
                ),
                system=self,
                parameter_name="sharpness",
            )

    def initialize(self, lower_limit=0.0, upper_limit=1.0, sharpness=10.0):
        pass

    def _soft_saturate(self, u, **params):
        return soft_saturate(
            u,
            params["lower_limit"],
            params["upper_limit"],
            params["sharpness"],
        )



class Stop(LeafSystem):
    """Stop the simulation early as soon as the input signal becomes True.

    If the input signal changes as a result of a discrete update, the simulation
    will terminate the major step early (before advancing continuous time).

    Input ports:
        (0): the boolean- or binary-valued termination signal

    Output ports:
        None
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()

        self.declare_zero_crossing(
            guard=self._guard,
            direction="negative_then_non_negative",
            terminal=True,
        )

    def _guard(self, time, state, u, **p):
        return npa.where(u, 1.0, -1.0)



# T-114-followup-prelookup end-of-file marker.


# ===========================================================================
# T-115-followup-deadzone-backlash
# ---------------------------------------------------------------------------
# This section adds two pieces on top of the T-115 / T-115-followup-mode-flag
# work:
#
#   * ``soft_dead_zone(u, half_range, sharpness)`` -- a differentiable
#     approximation to the hard dead-zone gate used by
#     :class:`DeadZone(mode="smooth")`.
#   * :class:`Backlash` -- a discrete-state hysteresis block modelling
#     mechanical gearbox slack / actuator hysteresis. The output lags the
#     input within a ``[last_output - width/2, last_output + width/2]``
#     band; outside the band the output sticks to the active edge.
#
# Default ``DeadZone(mode="hard")`` remains byte-equivalent to its phase 1
# behaviour (zero-crossing events still declared) -- the dispatch happens
# inside ``_dead_zone`` and ``initialize_static_data``.
# ===========================================================================


def soft_dead_zone(u, half_range, sharpness=10.0):
    """Smooth (differentiable) dead-zone gate.

    Approximates the hard dead-zone ``where(|u| < half_range, 0, u)``
    used by :class:`DeadZone(mode="hard")` with a sigmoid-blended kernel
    so gradients flow through the band.  The blend factor is
    ``sigmoid(sharpness * (|u| - half_range))``:

    ::

        gate = 0.5 * (1.0 + tanh(sharpness * (|u| - half_range) / half_range))
        y    = u * gate

    Properties:
      * ``y(0) = 0`` exactly.
      * ``gate -> 1`` outside the band, so ``y -> u`` for ``|u| >> half_range``.
      * ``gate -> 0`` inside the band, so ``y -> 0`` for ``|u| << half_range``.
      * Continuous everywhere; finite gradient even inside the band
        (this is the whole reason it exists).
      * As ``sharpness -> inf`` the function converges to the hard gate.

    Args:
        u: Input array.
        half_range: Positive scalar; band half-width.
        sharpness: Positive scalar; default ``10.0``. Larger values give
            a tighter approximation to the hard dead zone.

    Returns:
        Smoothly gated array, same shape as ``u``.
    """
    # Normalise by ``half_range`` so the blend transition lives near
    # ``|u| = half_range`` regardless of band width; this matches the
    # convention used by :func:`soft_saturate`.
    gate = 0.5 * (1.0 + npa.tanh(sharpness * (npa.abs(u) - half_range) / half_range))
    return u * gate


class Backlash(LeafSystem):
    """Hysteretic nonlinearity modelling mechanical slack / backlash.

    A ``Backlash(width)`` block has a single discrete state ``last_output``
    that tracks the most recent output value. At each sample tick the
    update rule is::

        delta = u - last_output
        if   delta >  width/2:  new_output = u - width/2
        elif delta < -width/2:  new_output = u + width/2
        else:                   new_output = last_output

    Equivalently, the output "follows" the input only after the input has
    moved by more than ``width/2`` from the last output; within that band
    the output sticks. This is the standard model for gearbox slack /
    actuator hysteresis: the input must take up the slack before the
    output moves.

    Differentiability: the per-step update is expressed via ``npa.where``
    on ``delta``; the output is a continuous (piecewise-linear) function
    of both ``u`` and ``width``, so ``jax.grad`` w.r.t. ``width`` is
    finite. The non-smooth "knee" at ``|delta| = width/2`` has subgradient
    ``0`` (inside the band) or ``-sign(delta)/2`` (outside) w.r.t.
    ``width`` -- both finite, as required.

    Input ports:
        (0) The driving input signal.

    Output ports:
        (0) The hysteretic output signal.

    Parameters:
        width: Positive scalar; total hysteresis band width. ``width=0``
            recovers ``y = u`` (no hysteresis).
        dt: Periodic update sample time. Required: this is a discrete
            block.
        initial_output: Initial value of the output / discrete state.
            Default ``0.0``.

    Notes:
        For an exact match against a canonical discrete-time "Backlash" block
        the discrete sample time must match. For a *continuous* hysteresis
        approximation, choose ``dt`` much smaller than the dominant input
        timescale; the block then tracks the input modulo the
        ``width/2`` slack with single-step latency.
    """

    @parameters(static=["dt"], dynamic=["width", "initial_output"])
    def __init__(self, width=1.0, dt=0.01, initial_output=0.0, **kwargs):
        super().__init__(**kwargs)
        if width < 0:
            raise BlockParameterError(
                message=(
                    f"Backlash block {self.name}: width must be >= 0, "
                    f"got {width}."
                ),
                system=self,
                parameter_name="width",
            )
        if dt <= 0:
            raise BlockParameterError(
                message=(
                    f"Backlash block {self.name}: dt must be > 0, got {dt}."
                ),
                system=self,
                parameter_name="dt",
            )
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, width=1.0, initial_output=0.0, dt=None):
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
            offset=0.0,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
            default_value=initial_output,
        )

    def reset_default_values(self, width=1.0, initial_output=0.0):
        self.declare_discrete_state(default_value=initial_output)
        self.configure_output_port_default_value(
            self._output_port_idx, initial_output
        )

    @staticmethod
    def _apply(last_output, u, width):
        """Pure backlash kernel.

        Splitting this out makes it directly testable / gradient-friendly
        outside the LeafSystem update path (see the T-115-followup tests).
        """
        half = width / 2.0
        delta = u - last_output
        # Output snaps to the edge of the band the input has crossed, or
        # stays put if the input is still inside the band.
        upper_edge = u - half  # active when delta > +half
        lower_edge = u + half  # active when delta < -half
        # First branch: above band -> follow at upper edge.
        # Second branch: below band -> follow at lower edge.
        # Else: still inside band -> hold previous output.
        return npa.where(
            delta > half,
            upper_edge,
            npa.where(delta < -half, lower_edge, last_output),
        )

    def _update(self, _time, state, u, **params):
        return self._apply(state.discrete_state, u, params["width"])

    def _output(self, _time, state, **_params):
        return state.discrete_state


# T-115-followup-deadzone-backlash end-of-file marker.


# ===========================================================================
# T-115-followup-deadzone-bilinear
# ---------------------------------------------------------------------------
# Adds :class:`DeadZoneInverse` -- the "complement" of the hard
# :class:`DeadZone` block. Where ``DeadZone(hr)`` zeroes the input INSIDE
# ``[-hr, hr]`` and passes it through outside, ``DeadZoneInverse(hr)``
# zeroes the input OUTSIDE the band and passes it through inside. The
# bilinear ``output_shifted`` kwarg on ``DeadZone`` lives next to the
# class itself (see the ``T-115-followup-deadzone-bilinear`` docstring
# block in :class:`DeadZone`).
# ===========================================================================


class DeadZoneInverse(FeedthroughBlock):
    """Pass through the input only inside the dead band; zero outside.

    Dual of :class:`DeadZone(mode="hard")`. The hard dead-zone block
    suppresses signals inside ``[-half_range, +half_range]`` and passes
    them through outside; this block does the opposite -- it passes the
    input through inside the band and clips it to zero outside.

    Applies the following function::

                 [ 0,           input < -half_range
        output = | input,       -half_range <= input <= half_range
                 [ 0            input > half_range

    Parameters:
        half_range: Positive scalar; the half-width of the pass-through
            band.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The gated input signal.

    Events:
        Zero-crossing events are declared at the band edges so ODE solvers
        do not integrate through the discontinuity (mirrors
        :class:`DeadZone` hard mode).
    """

    @parameters(dynamic=["half_range"])
    def __init__(self, half_range=1.0, **kwargs):
        super().__init__(self._dead_zone_inverse, **kwargs)
        if half_range <= 0:
            raise BlockParameterError(
                message=(
                    f"DeadZoneInverse block {self.name} has invalid "
                    f"half_range {half_range}. Must be > 0."
                ),
                system=self,
                parameter_name="half_range",
            )

    def _dead_zone_inverse(self, x, **params):
        hr = params["half_range"]
        return npa.where(abs(x) < hr, x, x * 0)

    def _lower_limit_event_value(self, _time, _state, *inputs, **params):
        (u,) = inputs
        return u + params["half_range"]

    def _upper_limit_event_value(self, _time, _state, *inputs, **params):
        (u,) = inputs
        return u - params["half_range"]

    def initialize_static_data(self, context):
        if not self.has_zero_crossing_events and (self.output_ports[0]):
            self.declare_zero_crossing(
                self._lower_limit_event_value, direction="crosses_zero"
            )
            self.declare_zero_crossing(
                self._upper_limit_event_value, direction="crosses_zero"
            )
        return super().initialize_static_data(context)
