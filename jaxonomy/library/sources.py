# SPDX-License-Identifier: MIT

"""Generative source blocks (deterministic + stochastic)."""

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
    "Chirp",
    "Clock",
    "Constant",
    "Counter",
    "DiscreteClock",
    "Pulse",
    "Ramp",
    "Sawtooth",
    "Sine",
    "Step",
    "UniformRandomNumber",
    "RandomSource",
    "BandLimitedNoise",
    "PRBS",
    "PRBSLFSR",
]



class Chirp(SourceBlock):
    """Produces a linear chirp signal — matches :func:`scipy.signal.chirp`.

    The output signal is ``cos(2π·f(t)·t + phi)`` with the linearly
    swept frequency ``f(t) = f0 + (f1 − f0)·t/(2·stop_time)``. At
    ``t=0`` the instantaneous frequency is ``f0`` Hz; at
    ``t=stop_time`` it is ``f1`` Hz.

    See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html

    Parameters:
        f0 (float): Frequency (Hz) at time t=0.
        f1 (float): Frequency (Hz) at time t=stop_time.
        stop_time (float): Time to end the signal (seconds).
        phi (float): Phase offset (radians).
        units (str | None, optional): T-122-followup-chirp-hz-convention
            — frequency-unit convention for ``f0`` / ``f1``. Defaults to
            ``"hz"`` (matches the docstring and :func:`scipy.signal.chirp`).
            Legacy diagrams that depended on the pre-2026-05 behaviour
            (where ``f0`` / ``f1`` were silently interpreted in rad/s)
            can opt into the old semantics by passing ``units="rad/s"``;
            doing so emits a :class:`DeprecationWarning` because the
            legacy path will be removed in a future release.

    Input ports:
        None

    Output ports:
        (0) The chirp signal.
    """

    @parameters(static=["units"], dynamic=["f0", "f1", "stop_time", "phi"])
    def __init__(self, f0, f1, stop_time, phi=0.0, units="hz", **kwargs):
        # T-122-followup-chirp-hz-convention: ``units`` is a static
        # opt-in to the legacy ``rad/s`` semantics; not JAX-traced.
        if units not in ("hz", "rad/s"):
            raise BlockParameterError(
                message=(
                    f"Chirp: units must be 'hz' (default) or 'rad/s' "
                    f"(legacy), got {units!r}."
                ),
                parameter_name="units",
            )
        if units == "rad/s":
            warnings.warn(
                "Chirp(..., units='rad/s') is the pre-2026-05 legacy "
                "convention that contradicts the documented Hz "
                "semantics and the scipy.signal.chirp parity claim. It "
                "is preserved here only for backwards compatibility "
                "and will be removed in a future release. Migrate to "
                "the default (units='hz') and divide your existing "
                "f0/f1 by 2π if you actually meant angular frequency.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._chirp_units = units
        super().__init__(None, **kwargs)

    def initialize(self, f0, f1, stop_time, phi, units="hz"):
        # T-122-followup-chirp-hz-convention: the Hz path multiplies by
        # 2π so the instantaneous frequency at time ``t`` matches the
        # docstring's stated f0 + (f1−f0)·t/stop_time Hz schedule. The
        # legacy ``rad/s`` path preserves the pre-fix expression
        # exactly for byte-equivalent reproducibility on existing
        # diagrams. Reads ``self._chirp_units`` so the JSON round-trip
        # path (which goes through ``@parameters(static=...)`` and
        # injects ``units`` as a re-init kwarg) and the manual path
        # both wind up with the same effective setting.
        del units  # parity arg for the @parameters decorator
        if self._chirp_units == "hz":
            two_pi = 2 * npa.pi

            def _func(time, stop_time, f0, f1, phi):
                f = f0 + (f1 - f0) * time / (2 * stop_time)
                return npa.cos(two_pi * f * time + phi)

        else:  # "rad/s" — legacy path

            def _func(time, stop_time, f0, f1, phi):
                f = f0 + (f1 - f0) * time / (2 * stop_time)
                return npa.cos(f * time + phi)

        self.replace_op(_func)


class Clock(SourceBlock):
    """Source block returning simulation time.

    Input ports:
        None

    Output ports:
        (0) The simulation time.

    Parameters:
        dtype:
            The data type of the output signal.  The default is "None", which will
            default to the current default floating point precision
    """

    def __init__(self, dtype=None, **kwargs):
        super().__init__(lambda t: npa.array(t, dtype=dtype), **kwargs)



class Constant(LeafSystem):
    """A source block that emits a constant value.

    Parameters:
        value: The constant value of the block.
        dtype (optional, T-038a-followup-other-blocks):
            If set, the constant value is cast to this dtype on output.
            See ``LookupTable1d`` for the per-block dtype contract.
        units (optional, T-104-followup-units-on-source-blocks):
            If set, the output port advertises this :class:`Unit`. The
            connect-time consistency check (T-104) then enforces
            downstream ports declare a compatible unit. Default
            ``None`` keeps the legacy "no-units" behaviour
            (byte-equivalent to pre-T-104 diagrams).

    Input ports:
        None

    Output ports:
        (0) The constant value.
    """

    @parameters(dynamic=["value"])
    def __init__(self, value, *args, dtype=None, units=None, **kwargs):
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
        super().__init__(**kwargs)
        # T-104-followup-units-on-source-blocks: forward the optional
        # ``units=`` kwarg to the output-port declaration so the source
        # block can advertise its own output unit (rather than relying
        # on the downstream port to do so).
        self._output_port_idx = self.declare_output_port(
            name="out_0", units=units
        )

    def initialize(self, value):
        if self._dtype is None:

            def _func(time, state, *inputs, **parameters):
                return parameters["value"]

        else:
            _dtype = self._dtype

            def _func(time, state, *inputs, **parameters):
                return npa.asarray(parameters["value"]).astype(_dtype)

        self.configure_output_port(
            self._output_port_idx,
            _func,
            prerequisites_of_calc=[DependencyTicket.nothing],
            requires_inputs=False,
        )



class DiscreteClock(LeafSystem):
    """Source block that produces the time sampled at a fixed rate.

    The block maintains the most recently sampled time as a discrete state, provided
    to the output port during the following interval. Graphically, a discrete clock
    sampled at 100 Hz would have the following time series:

    ```
      x(t)                  ●━
        |                   ┆
    .03 |              ●━━━━○
        |              ┆
    .02 |         ●━━━━○
        |         ┆
    .01 |    ●━━━━○
        |    ┆
      0 ●━━━━○----+----+----+-- t
        0   .01  .02  .03  .04
    ```

    The recorded states are the closed circles, which should be interpreted at index
    `n` as the value seen by all other blocks on the interval `(t[n], t[n+1])`.

    Input ports:
        None

    Output ports:
        (0) The sampled time.

    Parameters:
        dt:
            The sampling period of the clock.
        start_time:
            The simulation time at which the clock starts. Defaults to 0.
    """

    @parameters(static=["dt"])
    def __init__(self, dt, dtype=None, start_time=0, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype or float
        start_time = npa.array(start_time, dtype=self.dtype)

        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
            requires_inputs=False,
            default_value=start_time,
            prerequisites_of_calc=[DependencyTicket.time],
        )

    def _output(self, time, _state, *_inputs, **_params):
        return npa.array(time, dtype=self.dtype)



class Pulse(SourceBlock):
    """A periodic pulse signal.

    Given amplitude `a`, pulse width `w`, and period `p`, the output signal is:
    ```
        y(t) = a if t % p < w else 0
    ```
    where `%` is the modulo operator.

    Input ports:
        None

    Output ports:
        (0) The pulse signal.

    Parameters:
        amplitude:
            The amplitude of the pulse signal.
        pulse_width:
            The fraction of the period during which the pulse is "high".
        period:
            The period of the pulse signal.
        phase_delay:
            Currently unsupported.
    """

    @parameters(dynamic=["amplitude", "pulse_width", "period", "phase_delay"])
    def __init__(
        self, amplitude=1.0, pulse_width=0.5, period=1.0, phase_delay=0.0, **kwargs
    ):
        super().__init__(self._func, **kwargs)

        # Initialize the floating-point tolerance.  This will be machine epsilon
        # for the floating point type of the time variable (determined in the
        # static initialization step).
        self.eps = 0.0

        if abs(phase_delay) > 1e-9:
            warnings.warn("Warning. Pulse block phase_delay not implemented.")

        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        # ad 2 events, one for the up jump, and one the down jump
        self.declare_discrete_state(default_value=False)
        self._dummy_periodic_update_idx = self.declare_periodic_update()
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, amplitude, pulse_width, period, phase_delay):
        if abs(phase_delay) > 1e-9:
            warnings.warn("Warning. Pulse block phase_delay not implemented.")

        self.configure_periodic_update(
            self._dummy_periodic_update_idx,
            lambda *args, **kwargs: True,
            period=period,
            offset=period,
        )

        self.configure_periodic_update(
            self._periodic_update_idx,
            lambda *args, **kwargs: True,
            period=period,
            offset=period + period * pulse_width,
        )

    def _func(self, time, **parameters):
        # Add a floating-point tolerance to the modulo operation to avoid
        # accuracy issues when the time is an "exact" multiple of the period.
        period_fraction = (
            npa.remainder(time + self.eps, parameters["period"]) / parameters["period"]
        )
        return npa.where(
            period_fraction >= parameters["pulse_width"],
            0.0,
            parameters["amplitude"],
        )

    def initialize_static_data(self, context):
        # Determine machine epsilon for the type of the time variable
        self.eps = 2 * npa.finfo(npa.result_type(context.time)).eps
        return super().initialize_static_data(context)



class Ramp(SourceBlock):
    """Output a linear ramp signal in time.

    Given a slope `m`, a start value `y0`, and a start time `t0`, the output signal is:
    ```
        y(t) = m * (t - t0) + y0 if t >= t0 else y0
    ```
    where `t` is the current simulation time.

    Input ports:
        None

    Output ports:
        (0) The ramp signal.

    Parameters:
        start_value:
            The value of the output signal at the start time.
        slope:
            The slope of the ramp signal.
        start_time:
            The time at which the ramp signal begins.
        units (optional, T-104-followup-units-on-source-blocks):
            If set, the output port advertises this :class:`Unit`. The
            connect-time consistency check (T-104) then enforces
            downstream ports declare a compatible unit. Default
            ``None`` keeps the legacy "no-units" behaviour
            (byte-equivalent to pre-T-104 diagrams).
    """

    @parameters(dynamic=["start_value", "slope", "start_time"])
    def __init__(
        self,
        start_value=0.0,
        slope=1.0,
        start_time=1.0,
        units=None,
        **kwargs,
    ):
        super().__init__(self._func, **kwargs)
        # T-104-followup-units-on-source-blocks: see Sine for rationale.
        self.output_ports[self._output_port_idx].units = units

    def initialize(self, start_value, slope, start_time):
        pass

    def _func(self, time, **parameters):
        m = parameters["slope"]
        t0 = parameters["start_time"]
        y0 = parameters["start_value"]
        return npa.where(time >= t0, m * (time - t0) + y0, y0)



class Sawtooth(SourceBlock):
    """Produces a modulated linear sawtooth signal.

    The signal is similar to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html

    Given amplitude `a`, period `p`, and phase delay `phi`, the output signal is:
    ```
        y(t) = a * ((t - phi) % p)
    ```
    where `%` is the modulo operator.

    Input ports:
        None

    Output ports:
        (0) The sawtooth signal.
    """

    # `frequency` is set as a static parameter because it reconfigures the periodic
    # update when initialize() is called which would break optimization and
    # ensemble because they don't re-create the context and therefore won't call
    # initialize() if `frequency` is updated.
    @parameters(dynamic=["amplitude", "phase_delay"], static=["frequency"])
    def __init__(self, amplitude=1.0, frequency=0.5, phase_delay=1.0, **kwargs):
        super().__init__(self._func, **kwargs)

        # Initialize the floating-point tolerance.  This will be machine epsilon
        # for the floating point type of the time variable (determined in the
        # static initialization step).
        self.eps = 0.0
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, amplitude, frequency, phase_delay):
        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        self.declare_discrete_state(default_value=False)

        self.period = 1 / frequency
        self.configure_periodic_update(
            self._periodic_update_idx,
            lambda *args, **kwargs: True,
            period=self.period,
            offset=phase_delay,
        )

    def _func(self, time, **parameters):
        # np.mod((t - phase_delay), (1.0 / frequency)) * amplitude
        period_fraction = npa.mod(
            time - parameters["phase_delay"] + self.eps, self.period
        )
        return period_fraction * parameters["amplitude"]

    def initialize_static_data(self, context):
        # Determine machine epsilon for the type of the time variable
        self.eps = 2 * npa.finfo(npa.result_type(context.time)).eps
        return super().initialize_static_data(context)



class Sine(SourceBlock):
    """Generates a sinusoidal signal.

    Given amplitude `a`, frequency `f`, phase `phi`, and bias `b`, the output signal is:
    ```
        y(t) = a * sin(f * t + phi) + b
    ```

    Input ports:
        None

    Output ports:
        (0) The sinusoidal signal.

    Parameters:
        amplitude:
            The amplitude of the sinusoidal signal.
        frequency:
            The frequency of the sinusoidal signal.
        phase:
            The phase of the sinusoidal signal.
        bias:
            The bias of the sinusoidal signal.
        units (optional, T-104-followup-units-on-source-blocks):
            If set, the output port advertises this :class:`Unit`. The
            connect-time consistency check (T-104) then enforces
            downstream ports declare a compatible unit. Default
            ``None`` keeps the legacy "no-units" behaviour
            (byte-equivalent to pre-T-104 diagrams).
    """

    @parameters(dynamic=["amplitude", "frequency", "phase", "bias"])
    def __init__(
        self,
        amplitude=1.0,
        frequency=1.0,
        phase=0.0,
        bias=0.0,
        units=None,
        **kwargs,
    ):
        super().__init__(self._eval, **kwargs)
        # T-104-followup-units-on-source-blocks: the parent
        # ``SourceBlock.__init__`` already declared the output port; we
        # tag it with the requested unit here so the source advertises
        # its own output unit (rather than relying on the downstream
        # port to do so). Stored as a plain attribute on the OutputPort
        # to match the convention established by T-104 phase 1 in
        # ``framework/system_base.py``.
        self.output_ports[self._output_port_idx].units = units

    def initialize(self, amplitude=1.0, frequency=1.0, phase=0.0, bias=0.0):
        pass

    def _eval(self, t, **parameters):
        a = parameters["amplitude"]
        f = parameters["frequency"]
        phi = parameters["phase"]
        b = parameters["bias"]
        return a * npa.sin(f * t + phi) + b



class Step(SourceBlock):
    """A step signal.

    Given start value `y0`, end value `y1`, and step time `t0`, the
    output signal is:
    ```
        y(t) = y0 if t < t0 else y1
    ```

    Input ports:
        None

    Output ports:
        (0) The step signal.

    Parameters:
        start_value:
            The value of the output signal before the step time.
        end_value:
            The value of the output signal after the step time.
        step_time:
            The time at which the step occurs.
        units (optional, T-104-followup-units-on-source-blocks):
            If set, the output port advertises this :class:`Unit`. The
            connect-time consistency check (T-104) then enforces
            downstream ports declare a compatible unit. Default
            ``None`` keeps the legacy "no-units" behaviour
            (byte-equivalent to pre-T-104 diagrams).
    """

    @parameters(dynamic=["start_value", "end_value"], static=["step_time"])
    def __init__(
        self,
        start_value=0.0,
        end_value=1.0,
        step_time=1.0,
        units=None,
        **kwargs,
    ):
        super().__init__(self._func, **kwargs)
        # T-104-followup-units-on-source-blocks: see Sine for rationale.
        self.output_ports[self._output_port_idx].units = units
        self._periodic_update_idx = self.declare_periodic_update()

    def initialize(self, start_value, end_value, step_time):
        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        self._step_time = step_time
        self.declare_discrete_state(default_value=False)
        self.configure_periodic_update(
            self._periodic_update_idx,
            lambda *args, **kwargs: True,
            period=np.inf,
            offset=step_time,
        )

    def _func(self, time, **parameters):
        return npa.where(
            time >= self._step_time,
            parameters["end_value"],
            parameters["start_value"],
        )



# ---------------------------------------------------------------------------
# T-122 phase 1 — Stochastic sources: UniformRandomNumber and PRBS.
# (Original task ID T-MW-209, renumbered to T-122 in 124c178.)
#
# Two new sources for system-identification, noise-rejection, and Monte
# Carlo control studies. Both are discrete-time samplers driven by a
# JAX PRNG key carried in the block's discrete state, fully reproducible
# from an integer ``seed``.
#
# ``UniformRandomNumber(low, high, sample_time, seed)``: emits a fresh
# Uniform[low, high] sample every ``sample_time`` seconds. Implemented
# as ``low + (high - low) * uniform(0, 1)`` so gradients flow cleanly
# through ``low`` / ``high``; the unit-uniform draw itself is wrapped
# in ``lax.stop_gradient`` so JAX never tries to differentiate the
# random sequence with respect to the key.
#
# ``PRBS(sample_time, amplitude, seed)``: pseudo-random binary sequence
# emitting ``+amplitude`` or ``-amplitude``. Useful for system-ID input
# excitation. Implemented via ``jax.random.bernoulli`` then mapped
# ``2*b - 1``; ``amplitude`` is differentiable.
#
# ``BandLimitedNoise``, the full multi-distribution ``RandomNumber``
# spec, and per-vmap ``fold_in`` seeding are deferred — see
# T-122-followup-band-limited-noise, T-122-followup-distributions,
# and T-122-followup-vmap-fold-in.
# ---------------------------------------------------------------------------


class _PRNGState(NamedTuple):
    """Discrete state for stochastic source blocks: (key, current sample)."""
    key: "Array"
    val: "Array"


# T-122-followup-vmap-fold-in helper.
#
# When a stochastic-source block is run inside ``simulate_batch(use_vmap=True)``
# (or ``simulate_distributed``), the underlying ``jax.vmap`` is wrapped with
# ``axis_name="batch"`` (see jaxonomy/simulation/batch.py).  Blocks that opt
# into ``fold_in_batch_index=True`` call ``jax.lax.axis_index("batch")``
# inside their per-step ``_update`` to derive a per-replica unique salt and
# fold it into the freshly-split subkey, so every vmap'd replica draws an
# *independent* random stream from the same master ``seed``.
#
# Outside any vmap context (plain ``simulate``, or ``simulate_batch`` without
# ``use_vmap=True``), ``axis_index("batch")`` raises ``NameError`` at trace
# time -- we catch it and fall back to the plain subkey.  The kwarg therefore
# defaults to ``False`` and is byte-equivalent to T-122 phase 1 when unset.
#
# Honest-fallback note: a more explicit ``seed_per_replica`` kwarg could
# achieve the same effect without the vmap dependence; we ship both -- the
# auto fold-in is the convenient default for ensemble Monte Carlo, while
# users wanting full manual control can pass distinct ``seed`` integers per
# replica through ``param_batches`` (still supported as before).
def _maybe_fold_in_batch_axis(jrandom, subkey, fold_in_batch_index):
    """Fold the vmap batch index into ``subkey`` when requested.

    No-op when ``fold_in_batch_index`` is ``False`` or when the calling
    function is not running inside a ``jax.vmap(axis_name="batch")``
    (the unbound-axis ``NameError`` is caught at trace time).
    """
    if not fold_in_batch_index:
        return subkey
    import jax as _jax
    try:
        idx = _jax.lax.axis_index("batch")
    except NameError:
        # No vmap("batch") context -- e.g. plain simulate or
        # simulate_batch without use_vmap=True.  Fall back gracefully.
        return subkey
    return jrandom.fold_in(subkey, idx)


class UniformRandomNumber(LeafSystem):
    """Discrete-time uniform random number generator.

    Emits a fresh Uniform[low, high] sample every ``sample_time``
    seconds, using ``jax.random.uniform`` with a key carried in the
    block's discrete state. Reproducible: same ``seed`` and same
    diagram → bit-identical sequence.

    The sample is computed as ``low + (high - low) * u`` where
    ``u ~ Uniform[0, 1)``, so gradients of downstream losses flow
    cleanly through ``low`` and ``high`` via the reparameterization
    trick. The ``u`` draw is wrapped in ``lax.stop_gradient`` so JAX
    never tries to differentiate the random sequence w.r.t. the key.

    Input ports:
        None.

    Output ports:
        (0) The most recent uniform sample.

    Parameters:
        sample_time: Period (s) at which a fresh sample is drawn.
        low: Lower bound of the uniform interval (differentiable).
        high: Upper bound of the uniform interval (differentiable).
        seed: Integer seed for the PRNG key. If ``None``, a 32-bit
            random seed is drawn from ``numpy.random``.
        shape: Output shape. Default ``()`` (scalar).

    Notes:
        Per-vmap-batch independence: pass ``fold_in_batch_index=True``
        (T-122-followup-vmap-fold-in) to derive a per-replica
        independent PRNG stream via ``jax.lax.axis_index("batch")``
        inside ``simulate_batch(use_vmap=True)`` / ``simulate_distributed``.
        Outside any vmap context the kwarg is a no-op (the unbound-axis
        ``NameError`` is caught gracefully and the plain seed-derived
        key is used).  The default ``False`` preserves bit-identical
        behaviour with T-122 phase 1.
    """

    @parameters(
        static=["seed", "shape", "fold_in_batch_index"],
        dynamic=["low", "high"],
    )
    def __init__(
        self,
        sample_time: float,
        low: float = 0.0,
        high: float = 1.0,
        seed: int = None,
        shape=(),
        fold_in_batch_index: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._sample_time = float(sample_time)

        self.declare_output_port(
            self._output,
            period=sample_time,
            offset=0.0,
        )
        self.declare_periodic_update(
            self._update,
            period=sample_time,
            offset=0.0,
        )

    def initialize(
        self,
        low: float = 0.0,
        high: float = 1.0,
        seed: int = None,
        shape=(),
        fold_in_batch_index: bool = False,
    ):
        # Lazy JAX import — the framework supports a numpy-only backend,
        # but the stochastic sources require jax.random just like
        # ``RandomNumber`` and ``WhiteNoise`` already do (see
        # ``library/random.py`` module header).
        from jax import random as _jrandom
        from jax import lax as _jlax

        self._jrandom = _jrandom
        self._jlax = _jlax
        self._shape = tuple(int(s) for s in shape) if shape else ()
        self._fold_in_batch_index = bool(fold_in_batch_index)

        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
        key = _jrandom.PRNGKey(int(seed))
        key, subkey = _jrandom.split(key)
        # Build initial sample with the same shape/dtype the update
        # function will produce, so the discrete-state pytree is
        # stable across periodic updates.
        u0 = _jrandom.uniform(subkey, self._shape)
        val0 = float(low) + (float(high) - float(low)) * u0
        default_state = _PRNGState(key=key, val=val0)
        self.declare_discrete_state(default_value=default_state, as_array=False)

    def _output(self, _time, state, *_inputs, **_parameters):
        return state.discrete_state.val

    def _update(self, _time, state, *_inputs, **parameters):
        key, subkey = self._jrandom.split(state.discrete_state.key)
        # T-122-followup-vmap-fold-in: when running under
        # vmap(axis_name="batch") and ``fold_in_batch_index=True``, fold
        # the per-replica batch index into the freshly-split subkey so
        # each replica draws an independent stream from the same seed.
        subkey = _maybe_fold_in_batch_axis(
            self._jrandom, subkey, self._fold_in_batch_index
        )
        # ``stop_gradient`` makes the non-differentiability of the
        # random draw explicit — gradients still flow through ``low``
        # / ``high`` via the reparameterization below.
        u = self._jlax.stop_gradient(
            self._jrandom.uniform(subkey, self._shape)
        )
        low = parameters["low"]
        high = parameters["high"]
        val = low + (high - low) * u
        return _PRNGState(key=key, val=val)


class PRBS(LeafSystem):
    """Pseudo-Random Binary Sequence (PRBS) source.

    Emits ``+amplitude`` or ``-amplitude`` at each ``sample_time``
    tick, drawn from a fair Bernoulli(0.5) under
    ``jax.random.bernoulli`` and remapped via ``2*b - 1``. Useful as
    a broad-band excitation signal for system identification.

    The ``amplitude`` parameter is differentiable (scaling the binary
    selector); the ``+1 / -1`` selector itself is wrapped in
    ``lax.stop_gradient``.

    Input ports:
        None.

    Output ports:
        (0) The most recent ``±amplitude`` sample.

    Parameters:
        sample_time: Period (s) at which a fresh bit is drawn.
        amplitude: Magnitude of the binary output (differentiable).
        seed: Integer seed for the PRNG key. If ``None``, a 32-bit
            random seed is drawn from ``numpy.random``.

    Notes:
        Phase 1 uses Bernoulli(0.5) sampling rather than a true
        maximal-length LFSR. Period-faithful PRBS-N (n_bits register
        size) is deferred — see ``T-122-followup-lfsr``.

        Per-vmap-batch independence: pass ``fold_in_batch_index=True``
        (T-122-followup-vmap-fold-in) to derive a per-replica
        independent PRNG stream via ``jax.lax.axis_index("batch")``
        inside ``simulate_batch(use_vmap=True)`` / ``simulate_distributed``.
        Default ``False`` preserves bit-identical phase 1 behaviour.
    """

    @parameters(static=["seed", "fold_in_batch_index"], dynamic=["amplitude"])
    def __init__(
        self,
        sample_time: float,
        amplitude: float = 1.0,
        seed: int = None,
        fold_in_batch_index: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._sample_time = float(sample_time)

        self.declare_output_port(
            self._output,
            period=sample_time,
            offset=0.0,
        )
        self.declare_periodic_update(
            self._update,
            period=sample_time,
            offset=0.0,
        )

    def initialize(
        self,
        amplitude: float = 1.0,
        seed: int = None,
        fold_in_batch_index: bool = False,
    ):
        from jax import random as _jrandom
        from jax import lax as _jlax
        import jax.numpy as _jnp

        self._jrandom = _jrandom
        self._jlax = _jlax
        self._jnp = _jnp
        self._fold_in_batch_index = bool(fold_in_batch_index)

        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
        key = _jrandom.PRNGKey(int(seed))
        key, subkey = _jrandom.split(key)
        bit0 = _jrandom.bernoulli(subkey, p=0.5)
        # Map {False, True} -> {-1.0, +1.0} as float.
        sel0 = _jnp.where(bit0, 1.0, -1.0)
        val0 = float(amplitude) * sel0
        default_state = _PRNGState(key=key, val=val0)
        self.declare_discrete_state(default_value=default_state, as_array=False)

    def _output(self, _time, state, *_inputs, **_parameters):
        return state.discrete_state.val

    def _update(self, _time, state, *_inputs, **parameters):
        key, subkey = self._jrandom.split(state.discrete_state.key)
        # T-122-followup-vmap-fold-in: fold per-replica batch index into
        # subkey when running under vmap(axis_name="batch") and opted-in.
        subkey = _maybe_fold_in_batch_axis(
            self._jrandom, subkey, self._fold_in_batch_index
        )
        bit = self._jrandom.bernoulli(subkey, p=0.5)
        sel = self._jlax.stop_gradient(self._jnp.where(bit, 1.0, -1.0))
        amplitude = parameters["amplitude"]
        val = amplitude * sel
        return _PRNGState(key=key, val=val)



# T-127 phase 1 end-of-file marker.


# ===========================================================================
# T-122-followup-band-limited-noise — Continuous-time band-limited noise.
#
# ``BandLimitedNoise`` is the continuous-time analogue of ``RandomNumber``:
# a noise source whose power spectral density rolls off above a
# user-specified bandwidth.  The standard implementation is a discretised
# Ornstein-Uhlenbeck process,
#
#     dx/dt = -x / tau + sqrt(2 * sigma^2 / tau) * dW/dt
#
# integrated by the *exact-discrete* update at the block's sample period
# ``dt`` (no Euler-Maruyama integration error):
#
#     a       = exp(-dt / tau)
#     x[k+1]  = a * x[k] + sqrt(sigma^2 * (1 - a^2)) * randn()
#
# This preserves the OU steady-state variance ``sigma^2`` and the lag-tau
# autocorrelation ``e^{-1}`` exactly at the sample times, regardless of how
# coarse ``dt`` is relative to ``tau``.
#
# The block follows the same ``_PRNGState(key, val)`` discrete-state pattern
# as T-122 phase 1's ``UniformRandomNumber`` / ``PRBS``.  Gradients flow
# through ``tau``, ``sigma`` and ``mean`` via the standard reparameterisation
# (``mean + a * x + sqrt(...) * z``); the ``z ~ N(0, 1)`` draw itself is
# wrapped in ``lax.stop_gradient``.
# ===========================================================================


class BandLimitedNoise(LeafSystem):
    """Continuous-time band-limited noise (Ornstein-Uhlenbeck) source.

    Generates a zero-mean (plus a constant ``mean`` offset) Gaussian
    process with steady-state standard deviation ``sigma`` and
    correlation time ``tau``.  The power spectral density rolls off
    above the bandwidth ``1 / tau``.  Implemented as the
    *exact-discrete* update of an Ornstein-Uhlenbeck SDE at the block's
    sample period ``sample_time``::

        a      = exp(-sample_time / tau)
        x[k+1] = a * x[k] + sqrt(sigma^2 * (1 - a^2)) * z,   z ~ N(0, 1)

    The output is ``mean + x[k]``.  No integration error: the discrete
    samples have the same mean, variance and autocorrelation as the
    continuous OU process at those instants, irrespective of how
    coarse ``sample_time`` is relative to ``tau``.

    Input ports:
        None.

    Output ports:
        (0) The most recent ``mean + x`` sample.

    Parameters:
        sample_time: Period (s) at which a fresh OU step is taken.
        tau: Correlation time (s).  ``1 / tau`` is the roll-off
            bandwidth (rad/s).  Must be > 0.  Differentiable.
        sigma: Steady-state standard deviation of the OU process.
            Differentiable.
        mean: Constant offset added to the OU sample.  Differentiable.
        seed: Integer seed for the PRNG key.  If ``None``, a 32-bit
            random seed is drawn from ``numpy.random``.
        shape: Output shape.  Default ``()`` (scalar).

    Notes:
        Differentiability: ``tau``, ``sigma`` and ``mean`` flow into the
        OU update via smooth ``exp`` / ``sqrt`` so ``jax.grad`` is finite
        through them under the standard reparameterisation; the
        ``z ~ N(0, 1)`` draw is ``stop_gradient``-wrapped.

        Per-vmap-batch independence: pass ``fold_in_batch_index=True``
        (T-122-followup-vmap-fold-in) to derive a per-replica
        independent OU sample stream via ``jax.lax.axis_index("batch")``
        inside ``simulate_batch(use_vmap=True)`` / ``simulate_distributed``.
        Default ``False`` preserves bit-identical behaviour.
    """

    @parameters(
        static=["seed", "shape", "fold_in_batch_index"],
        dynamic=["tau", "sigma", "mean"],
    )
    def __init__(
        self,
        sample_time: float,
        tau: float = 1.0,
        sigma: float = 1.0,
        mean: float = 0.0,
        seed: int = None,
        shape=(),
        fold_in_batch_index: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._sample_time = float(sample_time)

        self.declare_output_port(
            self._output,
            period=sample_time,
            offset=0.0,
        )
        self.declare_periodic_update(
            self._update,
            period=sample_time,
            offset=0.0,
        )

    def initialize(
        self,
        tau: float = 1.0,
        sigma: float = 1.0,
        mean: float = 0.0,
        seed: int = None,
        shape=(),
        fold_in_batch_index: bool = False,
    ):
        # Lazy JAX import — mirrors the UniformRandomNumber / PRBS
        # convention so the framework's numpy-only backend stays
        # importable.
        from jax import random as _jrandom
        from jax import lax as _jlax
        import jax.numpy as _jnp

        self._jrandom = _jrandom
        self._jlax = _jlax
        self._jnp = _jnp
        self._shape = tuple(int(s) for s in shape) if shape else ()
        self._fold_in_batch_index = bool(fold_in_batch_index)

        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
        key = _jrandom.PRNGKey(int(seed))
        # The *initial* OU sample is drawn from the steady-state
        # distribution N(0, sigma^2): this avoids a transient spin-up
        # artefact at t=0 and makes the steady-state variance test pass
        # immediately.  Equivalent to the ``a = 0`` limit of the update.
        key, subkey = _jrandom.split(key)
        z0 = _jrandom.normal(subkey, self._shape)
        x0 = float(sigma) * z0
        default_state = _PRNGState(key=key, val=x0)
        self.declare_discrete_state(default_value=default_state, as_array=False)

    def _output(self, _time, state, *_inputs, **parameters):
        # Output is mean + x; gradients flow through ``mean`` directly
        # and through ``x`` via the recursive OU update in ``_update``.
        return parameters["mean"] + state.discrete_state.val

    def _update(self, _time, state, *_inputs, **parameters):
        key, subkey = self._jrandom.split(state.discrete_state.key)
        # T-122-followup-vmap-fold-in: fold per-replica batch index into
        # subkey when running under vmap(axis_name="batch") and opted-in.
        subkey = _maybe_fold_in_batch_axis(
            self._jrandom, subkey, self._fold_in_batch_index
        )
        # Standard normal draw; stop_gradient mirrors the
        # UniformRandomNumber / PRBS convention so JAX never tries to
        # differentiate the random sequence w.r.t. the key.
        z = self._jlax.stop_gradient(
            self._jrandom.normal(subkey, self._shape)
        )
        tau = parameters["tau"]
        sigma = parameters["sigma"]
        # Exact-discrete OU update: preserves variance and lag-tau
        # autocorrelation regardless of dt / tau ratio.
        a = self._jnp.exp(-self._sample_time / tau)
        std = self._jnp.sqrt(sigma * sigma * (1.0 - a * a))
        x_next = a * state.discrete_state.val + std * z
        return _PRNGState(key=key, val=x_next)



# T-127-fu-notch end-of-file marker.


# ===========================================================================
# T-122-followup-distributions — Multi-distribution ``RandomSource`` block.
#
# A unified discrete-time stochastic source that selects its distribution
# at construction via a string flag and a parameter dict::
#
#     RandomSource(sample_time, distribution="normal",
#                  params={"mean": 0.0, "std": 1.0}, seed=42)
#
# Supported distributions and their ``params`` keys:
#   * ``"uniform"``     — ``low``, ``high``
#   * ``"normal"``      — ``mean``, ``std``
#   * ``"lognormal"``   — ``mu``, ``sigma``    (log-space mean / std)
#   * ``"triangular"``  — ``low``, ``peak``, ``high``
#   * ``"exponential"`` — ``rate``             (T-122-followup-poisson)
#   * ``"poisson"``     — ``rate``             (T-122-followup-poisson;
#                                               discrete integer output)
#   * ``"categorical"`` — ``values``, ``probs``  (T-122-followup-categorical;
#                                                 discrete choice; ``values``
#                                                 is captured as a static
#                                                 table, ``probs`` is a
#                                                 dynamic vector parameter.)
#   * ``"bernoulli"``   — ``p``                  (T-122-followup-bernoulli;
#                                                 binary 0/1 outcome; thin
#                                                 convenience over
#                                                 ``categorical([0, 1], [1-p, p])``.
#                                                 Differentiable through ``p``
#                                                 only via the Gumbel-softmax
#                                                 path in ``jaxonomy.uq``.)
#
# Architecture: Python-time dispatch — the ``distribution`` string is
# *static* (declared in the @parameters static list), so different
# distribution choices produce different traced compute graphs. The
# concrete sampler is selected once in ``initialize`` and stored on
# ``self``; no ``lax.switch`` over distributions is needed at runtime
# (which would be clunky given the per-distribution param-shape mismatch).
#
# Reparameterisation for differentiability mirrors T-122 phase 1:
#   * uniform     → ``low + (high - low) * u``,            u ~ U[0,1)
#   * normal      → ``mean + std * z``,                    z ~ N(0,1)
#   * lognormal   → ``exp(mu + sigma * z)``,               z ~ N(0,1)
#   * triangular  → quantile transform of ``u`` (piecewise sqrt),
#                   smooth in (low, peak) and (peak, high).
#
# In every case, the underlying ``jax.random`` draw is wrapped in
# ``lax.stop_gradient`` so JAX never tries to differentiate the random
# sequence w.r.t. the key — gradients flow only through the named
# distribution parameters via the smooth transforms above. Output dtype
# follows the T-005 default-float64 policy (we never override dtype).
#
# Per-vmap-batch independence via ``jax.random.fold_in`` is *not*
# addressed here; same as T-122 phase 1, batched independence is via
# distinct ``seed`` integers per replica. See T-122-followup-vmap-fold-in.
# ===========================================================================


# Mapping ``distribution`` -> required ``params`` keys.  Pinned at module
# scope so that ``_validate_distribution`` can be reused by
# ``RandomSource.__init__`` and ``RandomSource.initialize`` without
# duplication, and so that the test file can in principle import it for
# round-tripping (kept underscored for now since the multi-distribution
# API is fresh).
_RANDOM_SOURCE_DISTRIBUTIONS = {
    "uniform": ("low", "high"),
    "normal": ("mean", "std"),
    "lognormal": ("mu", "sigma"),
    "triangular": ("low", "peak", "high"),
    # T-122-followup-poisson — Exponential / Poisson extensions.
    "exponential": ("rate",),
    "poisson": ("rate",),
    # T-122-followup-categorical — Categorical / discrete-choice extension.
    # ``values`` is captured as a static (Python-time) table; ``probs`` is a
    # dynamic vector parameter that flows through the simulation context
    # so gradients / vmap reach the per-category probabilities.
    "categorical": ("values", "probs"),
    # T-122-followup-bernoulli — Binary 0/1 outcome convenience.
    # ``p`` is a dynamic scalar parameter flowing through the context so
    # gradients / vmap reach the success probability.  Internally this
    # is the ``categorical([0, 1], [1-p, p])`` special case, but the
    # single-scalar API is far more readable for the very common
    # Bernoulli-trial / binary-event use case.
    "bernoulli": ("p",),
    # T-122-followup-beta-gamma — Bounded-fraction / positive-valued
    # extensions.  ``beta`` supports bounded fractions on [0, 1]
    # (utilisation, mixture weights, probability-of-probability priors);
    # ``gamma`` supports positive-valued quantities (wait times,
    # physical parameters, rate priors).  Both expose differentiable
    # sampling: ``gamma`` cleanly via ``jax.random.gamma`` scaled by
    # ``scale``; ``beta`` via ``jax.random.beta`` (implicit reparam,
    # higher gradient variance — see ``Beta`` docstring in
    # ``jaxonomy.uq.distributions``).
    "beta": ("alpha", "beta"),
    "gamma": ("shape", "scale"),
    # T-122-followup-weibull — Reliability / time-to-failure / wind-speed
    # distribution.  Closed-form inverse-CDF reparameterisation
    # ``x = scale * (-log(1-u))**(1/shape)`` is smooth in *both* ``shape``
    # and ``scale``, so gradients flow cleanly through both parameters
    # (strictly better differentiability profile than ``gamma`` which
    # relies on JAX's implicit-reparam machinery).  ``shape``
    # collides with the RandomSource output-shape static kwarg — handled
    # via ``_RANDOM_SOURCE_PARAM_INTERNAL_NAME`` (renamed to
    # ``weibull_shape`` for the dynamic-parameter slot).
    "weibull": ("shape", "scale"),
    # T-122-followup-pareto — Heavy-tail / power-law distribution.
    # PDF ``f(x) = alpha * scale**alpha / x**(alpha + 1)`` for x >= scale.
    # Closed-form inverse-CDF reparameterisation
    # ``x = scale * (1 - u)**(-1/alpha)`` is smooth in *both* ``scale``
    # and ``alpha``, so gradients flow cleanly through both parameters
    # (same differentiability profile as ``exponential`` and ``weibull``).
    # Neither param key collides with RandomSource's static parameters
    # so no rename is needed.
    "pareto": ("scale", "alpha"),
}

# Distributions whose ``params`` are *not* simple scalar floats but
# instead arrays / lists: they need bespoke registration so the
# ``float(params[k])`` scalar-coercion path does not fire on them.
# Keys here are the (distribution_name, param_key) pairs that must be
# treated as arrays rather than scalars.
_RANDOM_SOURCE_ARRAY_PARAMS = {
    ("categorical", "values"),
    ("categorical", "probs"),
}

# Distributions where some ``params`` are *static* (Python-time
# constants baked into the graph) rather than dynamic (live in the
# simulation context, flow through gradients / vmap).  For
# ``categorical``, ``values`` is a static table — gradients through a
# categorical sample's value choice are non-differentiable by
# construction, so there is no benefit to making it dynamic, and
# heterogeneous-dtype ``values`` (ints / strings / vectors) cannot
# safely flow as a single dynamic-parameter array.
_RANDOM_SOURCE_STATIC_PARAMS = {
    ("categorical", "values"),
}

# T-122-followup-beta-gamma — Some distribution param keys collide with
# RandomSource's own static parameters (notably ``shape``: the output-
# array shape is a static parameter of every RandomSource, and
# ``gamma``'s shape parameter would clash).  This table maps the user-
# facing ``params`` key to the *internal* dynamic-parameter name that
# we register on the block.  When unset, the user-facing key is reused
# verbatim.
_RANDOM_SOURCE_PARAM_INTERNAL_NAME = {
    ("gamma", "shape"): "gamma_shape",
    # T-122-followup-weibull: same shape-vs-output-shape collision as gamma.
    ("weibull", "shape"): "weibull_shape",
}


def _internal_param_name(distribution: str, key: str) -> str:
    """Internal dynamic-parameter name for a (distribution, key) pair.

    Defaults to the user-facing ``key``; overridden for collisions
    (e.g. ``gamma``'s ``shape`` -> ``gamma_shape``).
    """
    return _RANDOM_SOURCE_PARAM_INTERNAL_NAME.get((distribution, key), key)


def _validate_random_source_distribution(distribution: str, params: dict):
    """Validate ``distribution`` and the ``params`` dict shape.

    Raises ``ValueError`` with a clear human-readable message if the
    distribution name is unknown or if a required parameter is missing.
    """
    if distribution not in _RANDOM_SOURCE_DISTRIBUTIONS:
        valid = ", ".join(sorted(_RANDOM_SOURCE_DISTRIBUTIONS.keys()))
        raise ValueError(
            f"RandomSource: unknown distribution {distribution!r}. "
            f"Supported distributions: {valid}."
        )
    required = _RANDOM_SOURCE_DISTRIBUTIONS[distribution]
    if params is None:
        params = {}
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(
            f"RandomSource: distribution {distribution!r} requires "
            f"params keys {list(required)}; missing: {missing}."
        )


class RandomSource(LeafSystem):
    """Multi-distribution discrete-time random source.

    Unified rebuild of the single-distribution ``UniformRandomNumber``
    pattern from T-122 phase 1: one block, four distributions, selected
    at construction by a string flag plus a ``params`` dict.

    Supported distributions::

        distribution="uniform"     params={"low":  ..., "high": ...}
        distribution="normal"      params={"mean": ..., "std":  ...}
        distribution="lognormal"   params={"mu":   ..., "sigma": ...}
        distribution="triangular"  params={"low":  ..., "peak": ...,
                                           "high": ...}
        distribution="exponential" params={"rate": ...}
        distribution="poisson"     params={"rate": ...}   # integer-typed output
        distribution="bernoulli"   params={"p":    ...}   # integer-typed 0/1 output
        distribution="beta"        params={"alpha": ..., "beta":  ...}
        distribution="gamma"       params={"shape": ..., "scale": ...}
        distribution="weibull"     params={"shape": ..., "scale": ...}
        distribution="pareto"      params={"scale": ..., "alpha": ...}

    ``"exponential"`` is differentiable through ``rate`` via the
    standard inverse-CDF reparameterisation ``x = -log(1-u) / rate``;
    ``"poisson"`` is the discrete count distribution and is *not*
    differentiable through ``rate`` w.r.t. its samples (the per-sample
    grad is zero by construction — the sampler is wrapped in
    ``stop_gradient``).  See T-122-followup-poisson.

    Same seed -> bit-identical sequence (determinism contract).  Under
    ``simulate_batch(use_vmap=True)`` / ``simulate_distributed``, pass
    ``fold_in_batch_index=True`` (T-122-followup-vmap-fold-in) to derive
    a per-replica independent stream from the same master seed via
    ``jax.lax.axis_index("batch")``.  Default ``False`` preserves
    bit-identical behaviour with the original distributions follow-up.

    All named ``params`` flow through smooth, differentiable
    reparameterisations of an underlying ``Uniform[0,1)`` or ``N(0,1)``
    draw; the random draw itself is wrapped in ``lax.stop_gradient`` so
    gradients of downstream losses flow cleanly through the
    distribution parameters but never attempt to differentiate the PRNG
    key.

    Input ports:
        None.

    Output ports:
        (0) The most recent sample.

    Parameters:
        sample_time: Period (s) at which a fresh sample is drawn.
        distribution: One of ``"uniform"``, ``"normal"``, ``"lognormal"``,
            ``"triangular"``, ``"exponential"``, ``"poisson"``.
        params: Dict of distribution parameters (see above for keys).
            Each value is registered as a *dynamic* parameter and is
            differentiable / vmap-mappable.
        seed: Integer seed for the PRNG key.  If ``None``, a 32-bit
            random seed is drawn from ``numpy.random``.
        shape: Output shape.  Default ``()`` (scalar).

    Notes:
        Honest-fallback note: the spec contemplated a ``lax.switch``
        over distributions to share one block class.  Because each
        distribution has different ``params`` keys (and ``triangular``
        has three), a runtime switch would require padding/aligning
        the param tuples per distribution, which is clunky and
        defeats the whole point of named params.  Instead we dispatch
        at *Python time* on the static ``distribution`` flag --
        different distributions trace into different compute graphs,
        which is exactly the JAX-idiomatic path for static-flag
        polymorphism.
    """

    @parameters(static=["distribution", "seed", "shape", "fold_in_batch_index"])
    def __init__(
        self,
        sample_time: float,
        distribution: str = "uniform",
        params: dict = None,
        seed: int = None,
        shape=(),
        fold_in_batch_index: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if params is None:
            params = {}
        # Fail fast at construction so the user gets a clear error
        # before the simulator tries to compile a broken graph.
        _validate_random_source_distribution(distribution, params)

        self._sample_time = float(sample_time)
        self._distribution = distribution
        # Capture for ``initialize`` (which receives the static
        # parameters but not the dynamic ``params`` dict directly).
        self._param_keys = tuple(_RANDOM_SOURCE_DISTRIBUTIONS[distribution])

        # Build the per-key defaults.  Scalar-typed params get coerced
        # to float for parity with the T-122 phase 1 single-distribution
        # blocks; array-typed params (categorical ``values`` / ``probs``)
        # are converted to numpy arrays so they survive the ``initialize``
        # round-trip with stable shape / dtype.
        defaults = {}
        for key in self._param_keys:
            if (distribution, key) in _RANDOM_SOURCE_ARRAY_PARAMS:
                defaults[key] = np.asarray(params[key])
            else:
                defaults[key] = float(params[key])
        # T-122-followup-categorical: normalise ``probs`` to sum to 1 so
        # the static-table / dynamic-param pair agrees with the
        # ``Categorical`` distribution semantics in jaxonomy.uq.
        if distribution == "categorical":
            probs = np.asarray(defaults["probs"], dtype=np.float64)
            if probs.ndim != 1:
                raise ValueError(
                    f"RandomSource(categorical): probs must be 1D; got shape {probs.shape}."
                )
            values_arr = np.asarray(defaults["values"])
            if values_arr.shape[0] != probs.shape[0]:
                raise ValueError(
                    f"RandomSource(categorical): values length {values_arr.shape[0]} "
                    f"must match probs length {probs.shape[0]}."
                )
            if np.any(probs < 0.0) or probs.sum() <= 0.0:
                raise ValueError(
                    f"RandomSource(categorical): probs must be non-negative with "
                    f"positive sum; got {probs}."
                )
            defaults["probs"] = probs / probs.sum()
            defaults["values"] = values_arr
        # T-122-followup-bernoulli: validate ``p`` is a probability in
        # [0, 1].  The downstream sampler is the categorical sampler with
        # values=[0, 1], so a value of ``p`` outside [0, 1] would silently
        # produce a malformed distribution.
        if distribution == "bernoulli":
            p_val = defaults["p"]
            if not (0.0 <= p_val <= 1.0):
                raise ValueError(
                    f"RandomSource(bernoulli): p ({p_val}) must be in [0, 1]."
                )
        self._param_defaults = defaults

        # Register each distribution parameter as a *dynamic* parameter
        # so that gradients/vmap flow through them (cf. the T-122 phase 1
        # ``UniformRandomNumber.low/high`` pattern), and so they live in
        # the simulation context rather than baked into the closure.
        # Exception: keys flagged in ``_RANDOM_SOURCE_STATIC_PARAMS`` are
        # captured at Python time (e.g. categorical ``values``) — the
        # gather-by-index into a heterogeneous-dtype table cannot
        # safely flow as a single dynamic-parameter array.
        for key in self._param_keys:
            if (distribution, key) in _RANDOM_SOURCE_STATIC_PARAMS:
                continue
            internal_name = _internal_param_name(distribution, key)
            self.declare_dynamic_parameter(internal_name, self._param_defaults[key])

        self.declare_output_port(
            self._output,
            period=sample_time,
            offset=0.0,
        )
        self.declare_periodic_update(
            self._update,
            period=sample_time,
            offset=0.0,
        )

    def initialize(
        self,
        distribution: str = "uniform",
        seed: int = None,
        shape=(),
        fold_in_batch_index: bool = False,
        **_dynamic_params,
    ):
        # ``_dynamic_params`` swallows the per-distribution keys (low,
        # high, mean, std, mu, sigma, peak) that the framework passes
        # to ``initialize`` because they were declared via
        # ``declare_dynamic_parameter``.  The defaults captured at
        # construction (``self._param_defaults``) drive the initial
        # discrete-state sample, so we don't need to read them here.

        # Lazy JAX import — mirrors the UniformRandomNumber / PRBS /
        # BandLimitedNoise convention so the framework's numpy-only
        # backend stays importable.
        from jax import random as _jrandom
        from jax import lax as _jlax
        import jax.numpy as _jnp

        self._jrandom = _jrandom
        self._jlax = _jlax
        self._jnp = _jnp
        self._shape = tuple(int(s) for s in shape) if shape else ()
        self._fold_in_batch_index = bool(fold_in_batch_index)

        # T-122-followup-categorical: convert the static ``values`` table
        # to a JAX array once here so per-update gathers are zero-cost.
        # ``values`` is captured at construction (Python-time table); it
        # is not a dynamic context parameter.
        if self._distribution == "categorical":
            self._values_table = _jnp.asarray(self._param_defaults["values"])
        else:
            self._values_table = None

        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
        key = _jrandom.PRNGKey(int(seed))

        # Build the initial sample using the *default* parameter values
        # captured at construction.  This keeps the discrete-state
        # pytree shape stable across periodic updates regardless of
        # later context-time parameter overrides.
        key, subkey = _jrandom.split(key)
        val0 = self._sample_initial(subkey)
        default_state = _PRNGState(key=key, val=val0)
        self.declare_discrete_state(default_value=default_state, as_array=False)

    # ------------------------------------------------------------------ #
    # Per-distribution sampling                                          #
    # ------------------------------------------------------------------ #

    def _draw_uniform(self, subkey):
        """Stop-gradient unit-uniform draw of the configured shape."""
        return self._jlax.stop_gradient(
            self._jrandom.uniform(subkey, self._shape)
        )

    def _draw_normal(self, subkey):
        """Stop-gradient standard-normal draw of the configured shape."""
        return self._jlax.stop_gradient(
            self._jrandom.normal(subkey, self._shape)
        )

    def _sample_uniform(self, subkey, params):
        u = self._draw_uniform(subkey)
        return params["low"] + (params["high"] - params["low"]) * u

    def _sample_normal(self, subkey, params):
        z = self._draw_normal(subkey)
        return params["mean"] + params["std"] * z

    def _sample_lognormal(self, subkey, params):
        z = self._draw_normal(subkey)
        return self._jnp.exp(params["mu"] + params["sigma"] * z)

    def _sample_triangular(self, subkey, params):
        # Inverse-CDF ("quantile") transform of u ~ U[0,1) for the
        # triangular distribution on [low, high] with mode ``peak``:
        #     F^{-1}(u) =
        #       low  + sqrt(u  * (high-low) * (peak-low))     if u <= c
        #       high - sqrt((1-u) * (high-low) * (high-peak)) otherwise
        # where c = (peak - low) / (high - low) is the CDF at peak.
        # Smooth in low/peak/high (away from the degenerate
        # peak == low or peak == high boundaries), so jax.grad flows
        # cleanly through all three.
        u = self._draw_uniform(subkey)
        low = params["low"]
        peak = params["peak"]
        high = params["high"]
        width = high - low
        # ``c`` is the cumulative probability at the peak; the where
        # branches on a constant-shape boolean so jit/vmap are fine.
        c = (peak - low) / width
        left = low + self._jnp.sqrt(u * width * (peak - low))
        right = high - self._jnp.sqrt((1.0 - u) * width * (high - peak))
        return self._jnp.where(u <= c, left, right)

    # T-122-followup-poisson — Exponential / Poisson sampling.

    def _sample_exponential(self, subkey, params):
        """Reparameterised exponential draw: ``-log(1 - u) / rate``.

        Smooth in ``rate``, so jax.grad flows cleanly through the
        ``rate`` parameter when ``u`` is drawn under stop_gradient.
        Equivalently ``jax.random.exponential(key) / rate`` — we use
        the inverse-CDF form so the reparameterisation is the same one
        documented in the T-122 phase 1 architecture comment.
        """
        u = self._draw_uniform(subkey)
        # ``log1p(-u)`` is numerically stable near ``u -> 0``.
        return -self._jnp.log1p(-u) / params["rate"]

    def _sample_poisson(self, subkey, params):
        """Discrete Poisson count sampler.

        Output is integer-typed (``jax.random.poisson`` returns int32/
        int64), and the sample is wrapped in ``stop_gradient`` so JAX
        never tries to differentiate the discrete count through
        ``rate``.  This makes ``jax.grad(loss, rate)`` return zero from
        this block — which is correct given Poisson is non-
        differentiable w.r.t. its rate via the sample path.
        """
        # ``jax.random.poisson(key, lam, shape)`` is the canonical entry
        # point.  Cast the (potentially traced) ``rate`` to a JAX scalar
        # so the call works under jit/grad even when ``rate`` arrives
        # as a Python float.
        return self._jlax.stop_gradient(
            self._jrandom.poisson(subkey, params["rate"], shape=self._shape)
        )

    # T-122-followup-categorical — Categorical / discrete-choice sampling.

    def _sample_categorical(self, subkey, params):
        """Discrete categorical draw from a static ``values`` table.

        Picks an index ``i`` with probability ``probs[i]`` (normalised
        at construction) and returns ``values[i]`` (possibly a vector
        for vector-typed ``values``).  The selected index is wrapped in
        ``stop_gradient`` to make the non-differentiability of the
        hard categorical sample explicit — gradients through ``probs``
        and ``values`` from this sample path are zero.  Use the
        ``Categorical.differentiable_sample`` helper in
        :mod:`jaxonomy.uq.distributions` for a Gumbel-softmax relaxation
        that *is* differentiable through ``probs``.

        ``params["probs"]`` may be a tracer (it flows through the
        simulation context as a dynamic-vector parameter); we re-
        normalise here so the sampler is robust to upstream parameter
        edits that would otherwise break the ``sum == 1`` invariant.
        """
        probs = params["probs"]
        probs = probs / self._jnp.sum(probs)
        n_cat = self._values_table.shape[0]
        idx = self._jrandom.choice(subkey, n_cat, shape=self._shape, p=probs)
        idx = self._jlax.stop_gradient(idx)
        return self._values_table[idx]

    # T-122-followup-bernoulli — Binary 0/1 sampling.

    def _sample_bernoulli(self, subkey, params):
        """Bernoulli(p) draw — returns 0 with prob ``1 - p``, 1 with prob ``p``.

        Implemented via ``jax.random.bernoulli`` (which under the hood
        compares a unit-uniform draw to ``p``) cast to int32, then
        wrapped in ``stop_gradient`` so the discrete sample never tries
        to backpropagate through ``p``.  This makes
        ``jax.grad(loss, p)`` return zero from this block via the
        sample path — the corresponding differentiable channel is the
        Gumbel-softmax helper on ``Bernoulli`` /
        ``Categorical`` in :mod:`jaxonomy.uq.distributions`.

        ``p`` may arrive as a tracer (it flows through the simulation
        context as a dynamic scalar parameter); we clip to ``[0, 1]`` to
        be robust to small numerical drift, mirroring the categorical
        re-normalisation guard.
        """
        p = self._jnp.clip(params["p"], 0.0, 1.0)
        # ``jax.random.bernoulli`` returns a boolean array; cast to
        # int32 so downstream consumers see 0/1 integers (matching the
        # ``Categorical([0, 1], [1-p, p])`` semantics this delegates to).
        sample = self._jrandom.bernoulli(subkey, p, shape=self._shape)
        return self._jlax.stop_gradient(sample.astype(self._jnp.int32))

    # T-122-followup-beta-gamma — Beta and Gamma sampling.

    def _sample_beta(self, subkey, params):
        """Beta(alpha, beta) draw on the open ``(0, 1)`` interval.

        Routes through ``jax.random.beta``, which uses an implicit-reparam
        sampler — gradients flow through ``alpha`` / ``beta`` but with
        higher variance than the inverse-CDF reparams used by the other
        continuous distributions.  See the ``Beta`` docstring in
        :mod:`jaxonomy.uq.distributions` for the gradient-variance caveat.
        """
        return self._jrandom.beta(
            subkey, params["alpha"], params["beta"], shape=self._shape
        )

    def _sample_gamma(self, subkey, params):
        """Gamma(shape, scale) draw on ``[0, inf)``.

        Routes through ``jax.random.gamma`` (Marsaglia–Tsang reparam for
        shape >= 1; boost trick for shape < 1) scaled by ``scale``.
        Gradients flow cleanly through ``scale`` via the multiplicative
        rescaling and through ``shape`` via JAX's implicit-reparam
        machinery inside ``jax.random.gamma``.
        """
        z = self._jrandom.gamma(subkey, params["shape"], shape=self._shape)
        return z * params["scale"]

    # T-122-followup-weibull — Weibull sampling via closed-form inverse CDF.

    def _sample_weibull(self, subkey, params):
        """Weibull(shape, scale) draw on ``[0, inf)``.

        Closed-form inverse-CDF reparameterisation::

            x = scale * (-log(1 - u))**(1/shape),  u ~ U[0, 1)

        Smooth in *both* ``shape`` and ``scale`` -- gradients flow
        cleanly through both parameters analytically (no implicit-
        reparam machinery needed).  ``u`` is drawn under
        ``stop_gradient`` so JAX never tries to differentiate the
        random sequence w.r.t. the key.
        """
        u = self._draw_uniform(subkey)
        # ``log1p(-u)`` is numerically stable near ``u -> 0``; clamp the
        # open right boundary at ``1 - eps`` so the log stays finite.
        one_minus_eps = 1.0 - 1e-12
        u_safe = self._jnp.minimum(u, one_minus_eps)
        return params["scale"] * self._jnp.power(
            -self._jnp.log1p(-u_safe), 1.0 / params["shape"]
        )

    # T-122-followup-pareto — Pareto sampling via closed-form inverse CDF.

    def _sample_pareto(self, subkey, params):
        """Pareto(scale, alpha) draw on ``[scale, inf)``.

        Closed-form inverse-CDF reparameterisation::

            x = scale * (1 - u)**(-1 / alpha),  u ~ U[0, 1)

        Smooth in *both* ``scale`` and ``alpha`` -- gradients flow
        cleanly through both parameters analytically (no implicit-
        reparam machinery needed).  ``u`` is drawn under
        ``stop_gradient`` so JAX never tries to differentiate the
        random sequence w.r.t. the key.

        Computed via ``exp(-log1p(-u) / alpha)`` for numerical
        stability near ``u -> 0`` (where ``1 - u`` is close to 1 and
        the naive ``(1 - u)**(-1/alpha)`` form loses precision in
        the log).
        """
        u = self._draw_uniform(subkey)
        # Clamp the open right boundary so the ``-1/alpha`` exponent
        # stays finite at ``u -> 1``.
        one_minus_eps = 1.0 - 1e-12
        u_safe = self._jnp.minimum(u, one_minus_eps)
        return params["scale"] * self._jnp.exp(
            -self._jnp.log1p(-u_safe) / params["alpha"]
        )

    def _sample(self, subkey, params):
        """Dispatch on the static distribution flag."""
        if self._distribution == "uniform":
            return self._sample_uniform(subkey, params)
        if self._distribution == "normal":
            return self._sample_normal(subkey, params)
        if self._distribution == "lognormal":
            return self._sample_lognormal(subkey, params)
        if self._distribution == "triangular":
            return self._sample_triangular(subkey, params)
        if self._distribution == "exponential":
            return self._sample_exponential(subkey, params)
        if self._distribution == "poisson":
            return self._sample_poisson(subkey, params)
        if self._distribution == "categorical":
            return self._sample_categorical(subkey, params)
        if self._distribution == "bernoulli":
            return self._sample_bernoulli(subkey, params)
        if self._distribution == "beta":
            return self._sample_beta(subkey, params)
        if self._distribution == "gamma":
            return self._sample_gamma(subkey, params)
        if self._distribution == "weibull":
            return self._sample_weibull(subkey, params)
        if self._distribution == "pareto":
            return self._sample_pareto(subkey, params)
        # Unreachable: validated at __init__.  Defensive.
        raise ValueError(
            f"RandomSource: unknown distribution {self._distribution!r}"
        )

    def _sample_initial(self, subkey):
        """Draw the initial discrete-state sample using captured defaults."""
        return self._sample(subkey, self._param_defaults)

    # ------------------------------------------------------------------ #
    # LeafSystem callbacks                                               #
    # ------------------------------------------------------------------ #

    def _output(self, _time, state, *_inputs, **_parameters):
        return state.discrete_state.val

    def _update(self, _time, state, *_inputs, **parameters):
        key, subkey = self._jrandom.split(state.discrete_state.key)
        # T-122-followup-vmap-fold-in: fold per-replica batch index into
        # subkey when running under vmap(axis_name="batch") and opted-in.
        subkey = _maybe_fold_in_batch_axis(
            self._jrandom, subkey, self._fold_in_batch_index
        )
        # ``parameters`` is the *dynamic* parameter dict at simulation
        # time; pull just the *dynamic* keys this distribution uses so
        # we don't accidentally depend on stale context entries.  Static
        # keys (e.g. categorical ``values``) come from
        # ``self._param_defaults`` because they were captured at
        # Python time rather than registered as dynamic parameters.
        params = {}
        for k in self._param_keys:
            if (self._distribution, k) in _RANDOM_SOURCE_STATIC_PARAMS:
                params[k] = self._param_defaults[k]
            else:
                # T-122-followup-beta-gamma: dynamic parameters may be
                # registered under a renamed internal name (see
                # ``_internal_param_name``) when the user-facing key
                # collides with a RandomSource static parameter
                # (notably ``gamma``'s ``shape`` vs the output-shape
                # parameter).  Look up by the internal name so the
                # rename is transparent to ``_sample``.
                internal = _internal_param_name(self._distribution, k)
                params[k] = parameters[internal]
        val = self._sample(subkey, params)
        return _PRNGState(key=key, val=val)



# T-107-followup-variable-tau end-of-block marker.


# ===========================================================================
# T-122-followup-lfsr — true LFSR-based maximal-length PRBS-N source.
#
# T-122 phase 1's ``PRBS`` block uses ``jax.random.bernoulli(0.5)`` and
# remaps via ``2*b - 1``. That produces a binary {-1, +1} stream but its
# spectrum is only "white" in the IID-Bernoulli sense; it has no
# guaranteed period and no exact flat-band property. For system
# identification, the gold standard is a true LFSR-N maximal-length PRBS,
# whose period is exactly ``2^N - 1`` (every non-zero N-bit register
# state is visited once) and whose autocorrelation is a perfect
# Kronecker delta minus a tiny DC offset of ``-1/(2^N - 1)``.
#
# Architecture: a Galois/Fibonacci LFSR carried in the discrete state as
# a single ``uint32``. The feedback bit is the XOR of the bits at the
# tap positions for the chosen register length. The output is the LSB of
# the register, mapped {0, 1} -> {-amp, +amp}. ``amplitude`` flows
# through as a dynamic, differentiable parameter; the binary selector is
# wrapped in ``lax.stop_gradient`` for the same reason as PRBS phase 1.
#
# Tap polynomials are the standard primitive polynomials for each N
# (e.g. PRBS-7 ``x^7 + x^6 + 1``, PRBS-15 ``x^15 + x^14 + 1``, etc.).
# We unroll the XOR over taps in Python at construction time --
# ``functools.reduce`` over the static tap tuple -- so the per-step
# update traces into a flat XOR chain with no Python-level loop in the
# JAX trace.
# ===========================================================================


class _LFSRState(NamedTuple):
    """Discrete state for the LFSR-based PRBS-N source.

    Carries the current register value (``reg``) and the current output
    sample (``val``). Splitting the output from the register lets the
    output port read a precomputed value without re-deriving the LSB
    inside the read-only output callback.

    ``phase_advanced`` (added by T-122-followup-vmap-fold-in) is a
    one-shot uint32 flag (``0`` -> not yet, ``1`` -> already done) used
    only when ``fold_in_batch_index=True``: on the very first
    ``_update`` call the LFSR register is XOR-perturbed by a per-replica
    salt derived from ``jax.lax.axis_index("batch")``, then the flag
    flips to ``1`` so subsequent steps evolve the *unperturbed* LFSR
    from the new starting phase.  When ``fold_in_batch_index=False``
    (the default) the flag is unread and the behaviour is byte-equivalent
    to the original LFSR follow-up.
    """
    reg: "Array"
    val: "Array"
    phase_advanced: "Array"


class PRBSLFSR(LeafSystem):
    """True maximal-length PRBS-N source built on a binary LFSR.

    Emits ``+amplitude`` or ``-amplitude`` at each ``sample_time`` tick,
    drawn from a Linear-Feedback Shift Register (LFSR) of length
    ``register_length`` configured with the standard primitive
    feedback polynomial. The output sequence has period exactly
    ``2^N - 1`` and a flat power spectrum below ``1/(2N)`` of the
    sample rate, making it the canonical "white" excitation for system
    identification.

    Reproducibility: same ``seed`` -> bit-identical sequence. Distinct
    seeds traverse the same cyclic orbit at different starting phases.

    Differentiability: ``amplitude`` is a dynamic parameter and flows
    through gradients linearly; the binary selector itself is wrapped
    in ``lax.stop_gradient`` (the LSB extraction is non-differentiable).

    Input ports:
        None.

    Output ports:
        (0) The most recent ``±amplitude`` sample.

    Parameters:
        sample_time: Period (s) at which the LFSR advances by one step.
        amplitude: Magnitude of the binary output (differentiable).
        register_length: One of ``{7, 9, 11, 15, 17, 23, 31}``.
            Determines the period (``2^N - 1``) and the tap polynomial.
        seed: Non-zero integer seeding the LFSR register state. ``0``
            is silently promoted to ``1`` since the all-zero register
            is a fixed point of any LFSR (would emit a constant zero).

    Notes:
        Per-vmap-batch independence: pass ``fold_in_batch_index=True``
        (T-122-followup-vmap-fold-in) to derive a per-replica
        independent starting phase on the same maximal-length cycle.
        Inside ``simulate_batch(use_vmap=True)`` /
        ``simulate_distributed`` (which wrap their vmap with
        ``axis_name="batch"``), the LFSR register is XOR-perturbed by a
        per-replica non-zero salt derived from
        ``jax.lax.axis_index("batch")`` on the very first update step
        (tracked via a ``phase_advanced`` flag in the discrete state).
        The salt is masked to the low N bits of the register and
        promoted from 0 to 1 to avoid the all-zero fixed point.  Default
        ``False`` preserves bit-identical behaviour with the original
        LFSR follow-up.
    """

    # Standard primitive feedback polynomials (1-indexed bit positions
    # from the LSB end). Each tuple is the set of tap positions whose
    # bits are XORed to form the feedback bit.
    _TAPS = {
        7: (7, 6),
        9: (9, 5),
        11: (11, 9),
        15: (15, 14),
        17: (17, 14),
        23: (23, 18),
        31: (31, 28),
    }

    @parameters(
        static=["seed", "register_length", "fold_in_batch_index"],
        dynamic=["amplitude"],
    )
    def __init__(
        self,
        sample_time: float,
        amplitude: float = 1.0,
        register_length: int = 15,
        seed: int = 1,
        fold_in_batch_index: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if register_length not in self._TAPS:
            valid = ", ".join(str(k) for k in sorted(self._TAPS))
            raise ValueError(
                f"PRBSLFSR: unsupported register_length={register_length!r}. "
                f"Supported lengths: {valid}."
            )

        self._sample_time = float(sample_time)
        self._register_length = int(register_length)
        self._taps = self._TAPS[register_length]
        # Mask of the low N bits, used to keep the register inside
        # ``[0, 2^N)`` after every shift.
        self._reg_mask = (1 << self._register_length) - 1

        self.declare_output_port(
            self._output,
            period=sample_time,
            offset=0.0,
        )
        self.declare_periodic_update(
            self._update,
            period=sample_time,
            offset=0.0,
        )

    def initialize(
        self,
        amplitude: float = 1.0,
        register_length: int = 15,
        seed: int = 1,
        fold_in_batch_index: bool = False,
    ):
        # Lazy JAX import (consistent with the T-122 phase 1 sources).
        from jax import lax as _jlax
        from jax import random as _jrandom
        import jax.numpy as _jnp

        self._jlax = _jlax
        self._jrandom = _jrandom
        self._jnp = _jnp
        self._fold_in_batch_index = bool(fold_in_batch_index)

        # All-zero register is a fixed point of any LFSR; promote 0 -> 1
        # rather than silently emit a constant zero output.
        seed_int = int(seed) if seed is not None else 1
        seed_int = seed_int & self._reg_mask
        if seed_int == 0:
            seed_int = 1
        self._seed_int = seed_int

        reg0 = _jnp.asarray(seed_int, dtype=_jnp.uint32)
        bit0 = reg0 & _jnp.asarray(1, dtype=_jnp.uint32)
        sel0 = _jnp.where(bit0 == 1, 1.0, -1.0)
        val0 = float(amplitude) * sel0
        # ``phase_advanced=0`` means the per-replica LFSR phase shift
        # (T-122-followup-vmap-fold-in) has not yet been applied; it
        # gets flipped to ``1`` on the first ``_update`` call when
        # ``fold_in_batch_index=True``.  Always present in the state
        # tuple so the discrete-state pytree shape is independent of
        # the kwarg (keeps JIT cache keys stable).
        flag0 = _jnp.asarray(0, dtype=_jnp.uint32)
        default_state = _LFSRState(reg=reg0, val=val0, phase_advanced=flag0)
        self.declare_discrete_state(default_value=default_state, as_array=False)

    def _maybe_perturb_lfsr_register(self, reg, phase_advanced):
        """One-shot per-replica derivation of a fresh LFSR register.

        Returns ``(new_reg, new_phase_advanced)``.  No-op when
        ``fold_in_batch_index`` is ``False``, when ``phase_advanced``
        is already ``1`` (perturbation already applied on a previous
        step), or when not running under ``vmap(axis_name="batch")``
        (the unbound-axis ``NameError`` is caught at trace time).

        Applying the perturbation on the FIRST step only (rather than
        every step) preserves the LFSR's maximal-length property: each
        replica simply starts at a distinct register state on the same
        ``2^N - 1`` cycle and then evolves under the unperturbed
        feedback polynomial.

        The fresh per-replica register is derived by folding the batch
        index into a ``PRNGKey(seed)`` then taking the low N bits of a
        ``jax.random.bits`` draw -- this gives well-distributed
        register states across replicas and avoids the pathological
        degeneracies of a naive ``seed XOR axis_index`` (which can hit
        the all-zero fixed point or collide across replicas when
        ``seed`` and ``idx`` are small).  The all-zero candidate is
        promoted to ``1`` for the same reason as in ``initialize``.
        """
        if not self._fold_in_batch_index:
            return reg, phase_advanced
        import jax as _jax
        try:
            idx = _jax.lax.axis_index("batch")
        except NameError:
            return reg, phase_advanced
        # Derive a fresh per-replica register from a PRNG seeded with
        # ``seed`` and folded with the batch index.  This avoids the
        # naive-XOR pathologies (replica-0 collisions, all-zero fixed
        # point) and gives well-distributed starting states across
        # replicas.
        base_key = self._jrandom.PRNGKey(int(self._seed_int))
        per_rep_key = self._jrandom.fold_in(base_key, idx)
        # Draw 32 random bits and mask to N bits.
        rand_u32 = self._jrandom.bits(per_rep_key, shape=(), dtype=self._jnp.uint32)
        candidate = rand_u32 & self._jnp.asarray(
            self._reg_mask, dtype=self._jnp.uint32
        )
        # Promote the all-zero candidate to ``1`` -- same pattern as
        # ``initialize`` -- to avoid the LFSR's only fixed point.
        zero = self._jnp.asarray(0, dtype=self._jnp.uint32)
        one = self._jnp.asarray(1, dtype=self._jnp.uint32)
        candidate = self._jnp.where(candidate == zero, one, candidate)
        # Apply only on the first call (phase_advanced == 0).
        already = phase_advanced != zero
        new_reg = self._jnp.where(already, reg, candidate)
        new_flag = one
        return new_reg, new_flag

    def _output(self, _time, state, *_inputs, **_parameters):
        return state.discrete_state.val

    def _update(self, _time, state, *_inputs, **parameters):
        reg = state.discrete_state.reg
        phase_advanced = state.discrete_state.phase_advanced
        # T-122-followup-vmap-fold-in: one-shot per-replica phase shift
        # on the very first update step.  Bit-identical to the original
        # LFSR follow-up when ``fold_in_batch_index=False`` or when not
        # running under ``vmap(axis_name="batch")``.
        reg, phase_advanced = self._maybe_perturb_lfsr_register(
            reg, phase_advanced
        )
        one = self._jnp.asarray(1, dtype=self._jnp.uint32)
        # Unroll the XOR-over-taps chain in Python at trace time so the
        # JAX graph is a flat sequence of bit ops with no Python loop.
        feedback = (reg >> (self._taps[0] - 1)) & one
        for tap in self._taps[1:]:
            feedback = feedback ^ ((reg >> (tap - 1)) & one)
        mask = self._jnp.asarray(self._reg_mask, dtype=self._jnp.uint32)
        new_reg = ((reg << 1) | feedback) & mask
        new_bit = new_reg & one
        sel = self._jlax.stop_gradient(
            self._jnp.where(new_bit == 1, 1.0, -1.0)
        )
        amplitude = parameters["amplitude"]
        new_val = amplitude * sel
        return _LFSRState(
            reg=new_reg, val=new_val, phase_advanced=phase_advanced
        )



# ---------------------------------------------------------------------------
# T-120-followup-counter-block — discrete ``Counter`` block.
#
# Common block-diagram counter pattern: count rising edges on a trigger
# signal, optionally saturating or wrapping at a maximum count. The count
# itself is an integer state and therefore non-differentiable; downstream
# paths that use the count as a scaling factor are differentiable through
# the float-cast output, but gradients do not flow back into the count.
# ---------------------------------------------------------------------------


class Counter(LeafSystem):
    """Discrete counter that increments on rising edges of its trigger input.

    The block samples a boolean/binary trigger signal every ``dt`` seconds.
    On each rising edge (``prev_trigger == False`` and ``current_trigger ==
    True``) the internal count advances by ``increment``. When ``max_count``
    is set, the counter either *saturates* (clamps at ``max_count``) or
    *wraps* to ``0`` after the increment that hits / exceeds ``max_count``,
    according to ``reset_on_max``.

    Input ports:
        (0) Trigger signal — boolean / binary-valued.

    Output ports:
        (0) Current count (integer, stored as ``int32``).

    Parameters:
        initial_count:
            Starting count value at ``t = 0``. Default ``0``.
        dt:
            Sample period (seconds) of the discrete update.
        increment:
            Amount to add to the count on each rising edge. Default ``1``.
        max_count:
            Optional cap on the count. If ``None`` the counter is
            unbounded. Default ``None``.
        reset_on_max:
            When ``max_count`` is set, controls behaviour at saturation.
            ``True`` wraps the count back to ``0`` once it reaches /
            exceeds ``max_count``. ``False`` clamps the count at
            ``max_count``. Default ``False``.

    Notes:
        Edge detection is the same simple "previous-sample-was-False,
        current-sample-is-True" rule used by :class:`EdgeDetection` —
        adequate for boolean triggers driven at the block's own sample
        rate. For sub-sample-period precision use
        ``ZeroCrossingTriggeredSubsystem`` together with this block.

        The output is integer-typed and therefore non-differentiable in
        the strict sense; gradient-flow tests should not expect
        gradients with respect to the count itself.
    """

    class _DiscreteStateType(NamedTuple):
        prev_trigger: Array
        count: Array

    @parameters(
        dynamic=["initial_count"],
        static=["dt", "increment", "max_count", "reset_on_max"],
    )
    def __init__(
        self,
        dt,
        initial_count=0,
        increment=1,
        max_count=None,
        reset_on_max=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(
        self,
        initial_count,
        dt=None,
        increment=1,
        max_count=None,
        reset_on_max=False,
    ):
        if increment is None:
            raise BlockParameterError(
                message=f"Counter block {self.name} requires non-None increment.",
            )
        if max_count is not None and int(max_count) <= 0:
            raise BlockParameterError(
                message=(
                    f"Counter block {self.name} requires max_count > 0 "
                    f"(got {max_count})."
                ),
            )

        # Store static config as plain Python ints — they're declared as
        # static @parameters so they will not be JAX-traced.
        self._increment = int(increment)
        self._max_count = None if max_count is None else int(max_count)
        self._reset_on_max = bool(reset_on_max)

        count0 = npa.asarray(int(initial_count), dtype=npa.int32)
        prev0 = npa.asarray(False, dtype=npa.bool_)
        self.declare_discrete_state(
            default_value=self._DiscreteStateType(
                prev_trigger=prev0,
                count=count0,
            ),
            as_array=False,
        )
        self.configure_periodic_update(
            self._periodic_update_idx,
            self._update,
            period=self.dt,
            offset=0.0,
        )
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xd],
            requires_inputs=False,
            default_value=count0,
        )

    def reset_default_values(
        self,
        initial_count,
        dt=None,
        increment=1,
        max_count=None,
        reset_on_max=False,
    ):
        count0 = npa.asarray(int(initial_count), dtype=npa.int32)
        prev0 = npa.asarray(False, dtype=npa.bool_)
        self.configure_discrete_state_default_value(
            default_value=self._DiscreteStateType(
                prev_trigger=prev0,
                count=count0,
            ),
            as_array=False,
        )
        self.configure_output_port_default_value(self._output_port_idx, count0)

    def _update(self, _time, state, *inputs, **_params):
        (trigger,) = inputs
        # Cast trigger to bool so float/int/bool sources all yield the
        # same rising-edge semantics; mirrors EdgeDetection.
        trig = npa.asarray(trigger, dtype=npa.bool_)
        prev = state.discrete_state.prev_trigger
        count = state.discrete_state.count
        # Rising edge: previous sample was False, current sample is True.
        rising = npa.logical_and(npa.logical_not(prev), trig)

        # Tentative incremented count if a rising edge fired.
        inc = npa.asarray(self._increment, dtype=count.dtype)
        next_count = count + inc

        if self._max_count is not None:
            cap = npa.asarray(self._max_count, dtype=count.dtype)
            zero = npa.asarray(0, dtype=count.dtype)
            if self._reset_on_max:
                # Wrap: once the post-increment value would reach or
                # exceed the cap, wrap back to 0.
                next_count = npa.where(next_count >= cap, zero, next_count)
            else:
                # Saturate: clamp the post-increment value at the cap.
                next_count = npa.where(next_count > cap, cap, next_count)

        new_count = npa.where(rising, next_count, count)
        return self._DiscreteStateType(
            prev_trigger=trig,
            count=new_count,
        )

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state.count
