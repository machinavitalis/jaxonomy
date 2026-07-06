# SPDX-License-Identifier: MIT

"""Lookup tables and the prelookup/interpolation family."""

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
    "LookupTable1d",
    "LookupTable2d",
    "LookupTableND",
    "Prelookup",
    "PrelookupInverse",
    "InterpolationUsingPrelookup",
    "TableSearch",
]



class LookupTable1d(FeedthroughBlock):
    """Interpolate the input signal into a static lookup table.

    If a function `y = f(x)` is sampled at a set of points `(x_i, y_i)`, then this
    block will interpolate the input signal `x` to compute the output signal `y`.
    The behavior is modeled after `scipy.interpolate.interp1d` but is implemented
    in JAX.  Available interpolation modes are:
        - "linear": Linear interpolation using `jax.interp`.
        - "pchip": Monotone cubic Hermite (Hyman/Fritsch-Carlson). T-106
          phase 1 — smooth gradients everywhere, monotone on monotone data.
        - "akima": Akima 1970 cubic spline (T-114 phase 2). Smoother than
          PCHIP on non-monotone data, less prone to overshoot than
          natural cubic splines. Matches
          ``scipy.interpolate.Akima1DInterpolator``.
        - "cubic": Natural cubic spline (T-114-followup-natural-cubic-
          spline). C^2-continuous, second derivative zero at the
          boundaries — the smoothest possible C^2 interpolant. Requires
          at least 4 breakpoints. Matches
          ``scipy.interpolate.CubicSpline(bc_type='natural')``.
        - "nearest": Nearest-neighbor interpolation.
        - "flat": Flat interpolation.

    Input ports:
        (0) The input signal, which is used as the interpolation coordinate.

    Output ports:
        (0) The interpolated output signal.

    Parameters:
        input_array:
            The array of input values at which the output values are provided.
        output_array:
            The array of output values.
        interpolation:
            One of "linear", "pchip", "nearest", or "flat". Determines the type
            of interpolation performed by the block.
        extrapolation (optional, T-114 phase 1):
            One of "clip" (default — standard lookup-table behaviour, holds
            the boundary value), "linear" (extends the boundary slope past the breakpoints),
            or "nan" (returns NaN outside ``[input_array[0], input_array[-1]]``).
            Stored outside ``@parameters``, so this kwarg is *not* round-tripped
            through model JSON — pre-existing models reload byte-equivalently.
        dtype (optional, T-038a):
            If set (e.g. ``jnp.float32``), the block's ``input_array`` and
            ``output_array`` are cast to this dtype on construction, so the
            interpolation arithmetic — and therefore the output signal — runs at
            this precision regardless of the global x64 setting.  The default
            (``None``) preserves the pre-T-038a behavior: the arrays are stored
            verbatim and float64 is used under the default x64-enabled install.

            T-038a-followup-mixed-precision-cascade: when ``dtype is None``
            and a :func:`jaxonomy.precision_policy` context manager is
            active, the block falls back to the context's dtype.  Explicit
            ``dtype=`` always wins (explicit-over-implicit).

            Best-effort: this enforces dtype on the block's internal arrays
            and on the output of ``jnp.interp`` / ``jnp.argmin`` lookups, but
            downstream operations (e.g. connecting to a default-dtype block)
            are subject to JAX's standard promotion rules — the result of the
            wider arithmetic may be promoted to ``float64``.

    Notes:
        Currently restricted to 1D input and output data.  This may be expanded to
        support multi-dimensional output arrays in the future.
    """

    @parameters(static=["input_array", "output_array", "interpolation"])
    def __init__(
        self,
        input_array,
        output_array,
        interpolation,
        dtype=None,
        extrapolation="clip",
        **kwargs,
    ):
        # T-038a-followup-mixed-precision-cascade — when no explicit
        # ``dtype=`` kwarg was passed (``dtype is None``), fall back to
        # the active ``precision_policy`` context manager's dtype, if
        # any.  Explicit per-block dtype always wins (explicit-over-
        # implicit).  Outside any active context the policy resolver
        # returns ``None`` and the block's default-dtype path runs
        # unchanged — byte-equivalent to pre-follow-up behavior.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        # T-038a — remember the per-block dtype override so ``initialize``
        # (called by InitializeParameterResolver after parameter resolution)
        # can apply it to the resolved array values.  ``dtype`` is *not* a
        # @parameters-tracked field: it is a build-time block-shape decision
        # that doesn't round-trip through model JSON or get JAX-traced.
        self._dtype = dtype
        # T-114 phase 1 — extrapolation policy.  Stored outside the
        # @parameters list so the kwarg does not round-trip through model
        # JSON.  The default ``"clip"`` matches the historical behavior of
        # ``jnp.interp`` so existing pipelines stay byte-equivalent.
        if extrapolation not in ("clip", "linear", "nan"):
            raise ValueError(
                f"LookupTable1d: extrapolation must be one of "
                f"('clip','linear','nan'), got {extrapolation!r}"
            )
        self._extrapolation = extrapolation
        super().__init__(None, **kwargs)
        # T-002: reject non-monotonic input_array up front so silent
        # interpolation garbage doesn't propagate downstream.
        _input_np = np.asarray(input_array)
        if _input_np.ndim == 1 and _input_np.size >= 2 and not np.all(
            np.diff(_input_np) > 0
        ):
            raise ValueError(
                f"LookupTable1d '{self.name}': input_array must be strictly "
                f"monotonically increasing, got {list(_input_np)}"
            )

    def initialize(self, input_array, output_array, interpolation):
        if self._dtype is not None:
            # T-038a — cast both lookup arrays to the per-block dtype.  Done
            # after npa.array() so the dtype override survives the backend's
            # default-float promotion.
            self.input_array = npa.asarray(input_array).astype(self._dtype)
            self.output_array = npa.asarray(output_array).astype(self._dtype)
        else:
            self.input_array = npa.array(input_array)
            self.output_array = npa.array(output_array)
        if len(self.input_array.shape) != 1:
            raise ValueError(
                f"LookupTable1d block {self.name} input_array must be 1D, got shape "
                f"{self.input_array.shape}"
            )
        if len(self.output_array.shape) != 1:
            raise ValueError(
                f"LookupTable1d block {self.name} output_array must be 1D, got shape "
                f"{self.output_array.shape}"
            )
        self.max_i = len(self.input_array) - 1

        # T-114 phase 1 — fast path: when the user requested the
        # historical default (``extrapolation="clip"`` and one of the
        # original three methods), keep the original implementations
        # untouched so this default path stays byte-equivalent with the
        # pre-T-114 code, including for the numpy backend.  Only when
        # the user opts in to ``"pchip"`` or non-clip extrapolation do we
        # route through the JAX-only ``interp_1d`` backend.
        legacy_methods = {
            "linear": self._lookup_linear,
            "nearest": self._lookup_nearest,
            "flat": self._lookup_flat,
        }
        if self._extrapolation == "clip" and interpolation in legacy_methods:
            self.replace_op(legacy_methods[interpolation])
            return

        if interpolation not in (
            "linear", "pchip", "akima", "cubic", "nearest", "flat"
        ):
            raise ValueError(
                f"LookupTable1d block {self.name} has invalid selection {interpolation} "
                "for 'interpolation'"
            )

        from .lookup_table import interp_1d as _interp_1d

        extrapolation = self._extrapolation

        def _op(x):
            return _interp_1d(
                x,
                self.input_array,
                self.output_array,
                method=interpolation,
                extrapolation=extrapolation,
            )

        self.replace_op(_op)

    def _lookup_linear(self, x):
        return npa.interp(x, self.input_array, self.output_array)

    def _lookup_nearest(self, x):
        i = npa.argmin(npa.abs(self.input_array - x))
        i = npa.clip(i, 0, self.max_i)
        return self.output_array[i]

    def _lookup_flat(self, x):
        i = npa.where(
            x < self.input_array[1],
            0,
            npa.argmin(x >= self.input_array) - 1,
        )
        return self.output_array[i]

    @classmethod
    def fit_from_data(
        cls,
        xp,
        x_data,
        y_data,
        *,
        weights=None,
        smoothness: float = 0.0,
        **block_kwargs,
    ):
        """Build a ``LookupTable1d`` whose output values are fitted by
        least squares to ``(x_data, y_data)`` at the fixed grid ``xp``.

        Ergonomic wrapper around :func:`jaxonomy.library.fit_lookup_table_1d`
        so the fitting entry point is discoverable from the block class
        itself.

        Args:
            xp: Fixed grid of breakpoints (1-D, strictly increasing).
            x_data: Measured input cloud, shape ``(K,)``.
            y_data: Measured output cloud, shape ``(K,)``.
            weights: Optional per-sample weights for weighted least
                squares.  ``None`` = OLS.
            smoothness: Non-negative discrete first-difference penalty.
                Use small values (1e-3 .. 1.0) on noisy / sparse data.
            **block_kwargs: Forwarded to
                :func:`jaxonomy.library.fit_lookup_table_1d` (e.g.
                ``interpolation=``, ``extrapolation=``, ``name=``,
                ``dtype=``).

        Returns:
            A ``LookupTable1d`` instance with ``input_array=xp`` and
            ``output_array`` set to the LS-fit table values.
        """
        # Lazy import — ``lookup_table_fitting`` imports ``LookupTable1d``
        # from this module, so doing the import at function-call time
        # breaks any circular import risk while keeping the public
        # ``LookupTable1d(...)`` constructor path untouched (existing
        # call sites stay byte-equivalent).
        from .lookup_table_fitting import fit_lookup_table_1d

        return fit_lookup_table_1d(
            xp,
            x_data,
            y_data,
            weights=weights,
            smoothness=smoothness,
            **block_kwargs,
        )


class LookupTable2d(LeafSystem):
    """Interpolate the input signals into a static lookup table.

    The behavior is modeled on `scipy.interpolate.interp2d` but is implemented
    in JAX.  ``"linear"`` (bilinear, default) and ``"bicubic"`` (Catmull-Rom)
    interpolation are supported.  The input arrays must be 1D and the output
    array must be 2D.

    Input ports:
        (0) The first input signal, used as the first interpolation coordinate.
        (1) The second input signal, used as the second interpolation coordinate.

    Output ports:
        (0) The interpolated output signal.

    Parameters:
        input_x_array:
            The array of input values at which the output values are provided,
            corresponding to the first input signal. Must be 1D
        input_y_array:
            The array of input values at which the output values are provided,
            corresponding to the second input signal. Must be 1D
        output_table_array:
            The array of output values. Must be 2D with shape `(m, n)`, where
            `m = len(input_x_array)` and `n = len(input_y_array)`.
        interpolation:
            ``"linear"`` (default, standard bilinear behaviour) or
            ``"bicubic"`` (T-114-followup-2d-bicubic — Catmull-Rom
            cubic-convolution kernel; C^1-continuous, exact at grid
            corners, smoother than bilinear off-grid).  ``"bicubic"``
            requires at least 4 breakpoints per axis.
        extrapolation (optional, T-114 phase 2):
            One of "clip" (default — standard lookup-table behaviour, holds
            the boundary value), "linear" (bilinear extension past the grid
            via edge-slope continuation), or "nan" (returns NaN outside
            the grid).  The default ``"clip"`` matches the legacy
            ``npa.interp2d`` behaviour byte-equivalently.  Stored
            outside ``@parameters`` so this kwarg is not round-tripped
            through model JSON; pre-existing models reload byte-
            equivalently.
    """

    @parameters(
        static=["input_x_array", "input_y_array", "output_table_array", "interpolation"]
    )
    def __init__(
        self,
        input_x_array,
        input_y_array,
        output_table_array,
        interpolation="linear",
        dtype=None,
        extrapolation="clip",
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
        # T-114 phase 2 — extrapolation policy.  Stored outside the
        # @parameters list so the kwarg does not round-trip through model
        # JSON.  The default ``"clip"`` matches the historical behavior of
        # ``npa.interp2d`` so existing pipelines stay byte-equivalent.
        if extrapolation not in ("clip", "linear", "nan"):
            raise ValueError(
                f"LookupTable2d: extrapolation must be one of "
                f"('clip','linear','nan'), got {extrapolation!r}"
            )
        self._extrapolation = extrapolation
        super().__init__(**kwargs)
        self.declare_input_port()
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    # T-114-followup-lookup-table-fitted-getter — expose the stored static
    # parameters as plain read-only attributes so the fitted table can be
    # inspected (e.g. plotted) without re-running ``fit_table_2d``.
    @property
    def output_table_array(self):
        return self._static_parameters["output_table_array"].get()

    @property
    def input_x_array(self):
        return self._static_parameters["input_x_array"].get()

    @property
    def input_y_array(self):
        return self._static_parameters["input_y_array"].get()

    def initialize(
        self, input_x_array, input_y_array, output_table_array, interpolation
    ):
        if self._dtype is not None:
            # T-038a-followup-other-blocks: cast the lookup arrays to the
            # per-block dtype so the interp arithmetic runs at this
            # precision regardless of the global x64 setting.
            xp = npa.asarray(input_x_array).astype(self._dtype)
            yp = npa.asarray(input_y_array).astype(self._dtype)
            zp = npa.asarray(output_table_array).astype(self._dtype)
        else:
            xp = npa.array(input_x_array)
            yp = npa.array(input_y_array)
            zp = npa.array(output_table_array)

        if len(xp.shape) != 1:
            raise ValueError(
                f"LookupTable2d block {self.name} input_x_array must be 1D, got "
                f"shape {xp.shape}"
            )

        if len(yp.shape) != 1:
            raise ValueError(
                f"LookupTable2d block {self.name} input_y_array must be 1D, got "
                f"shape {yp.shape}"
            )

        if len(zp.shape) != 2:
            raise ValueError(
                f"LookupTable2d block {self.name} output_table_array must be 2D, "
                f"got shape {zp.shape}"
            )

        if zp.shape != (len(xp), len(yp)):
            raise ValueError(
                f"LookupTable2d block {self.name} output_table_array must have "
                f"shape (len(input_x_array), len(input_y_array)), got shape {zp.shape}"
            )

        if interpolation not in ("linear", "bicubic"):
            raise NotImplementedError(
                f"LookupTable2d block {self.name} only supports "
                f"'linear' or 'bicubic' interpolation, got {interpolation!r}."
            )

        if interpolation == "bicubic" and (len(xp) < 4 or len(yp) < 4):
            # Catmull-Rom needs 4 breakpoints per axis for a well-defined
            # stencil — fail loud rather than silently degrade.
            raise ValueError(
                f"LookupTable2d block {self.name}: interpolation='bicubic' "
                f"requires at least 4 breakpoints per axis, got "
                f"len(input_x_array)={len(xp)} and len(input_y_array)={len(yp)}"
            )

        # T-114 phase 2 — fast path: when the user requested the
        # historical default (``interpolation="linear"`` +
        # ``extrapolation="clip"``), keep the legacy ``npa.interp2d``
        # call so this default path stays byte-equivalent with the
        # pre-T-114-phase-2 code, including for the numpy backend.
        # Anything else (non-clip extrapolation OR bicubic interp)
        # routes through the JAX-only ``interp_2d`` backend.
        if interpolation == "linear" and self._extrapolation == "clip":
            self._compute_output = partial(npa.interp2d, xp, yp, zp)
        else:
            from .lookup_table import interp_2d as _interp_2d

            extrapolation = self._extrapolation
            method = interpolation

            def _op(x, y):
                return _interp_2d(
                    x, y, xp, yp, zp,
                    method=method,
                    extrapolation=extrapolation,
                )

            self._compute_output = _op

        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

    def _output(self, _time, _state, *inputs, **params):
        (x, y) = inputs
        return self._compute_output(x, y)

    @classmethod
    def fit_from_data(
        cls,
        xp,
        yp,
        x_data,
        y_data,
        z_data,
        *,
        weights=None,
        smoothness: float = 0.0,
        **block_kwargs,
    ):
        """Build a ``LookupTable2d`` whose table values are fitted by
        bilinear least squares to ``(x_data, y_data, z_data)`` at the
        fixed grid ``(xp, yp)``.

        Ergonomic classmethod mirror of
        :func:`jaxonomy.library.fit_lookup_table_2d` so the 2-D fitting
        entry point is discoverable from the block class itself.

        Args:
            xp: Fixed grid along the first axis (1-D, strictly
                increasing).
            yp: Fixed grid along the second axis (1-D, strictly
                increasing).
            x_data, y_data, z_data: Measurement cloud, all shape
                ``(K,)``.
            weights: Optional per-sample weights for weighted least
                squares.  ``None`` = OLS.
            smoothness: Non-negative 5-point Laplacian penalty on the
                fitted table.  ``0.0`` (default) is pure data-fit.
            **block_kwargs: Forwarded to
                :func:`jaxonomy.library.fit_lookup_table_2d` (e.g.
                ``interpolation=``, ``extrapolation=``, ``name=``,
                ``dtype=``).

        Returns:
            A ``LookupTable2d`` instance with ``input_x_array=xp``,
            ``input_y_array=yp``, and ``output_table_array`` set to the
            LS-fit table of shape ``(len(xp), len(yp))``.
        """
        # Lazy import — ``lookup_table_fitting`` imports ``LookupTable2d``
        # from this module, so doing the import at function-call time
        # avoids the circular import while keeping the public
        # ``LookupTable2d(...)`` constructor path untouched (existing
        # call sites stay byte-equivalent).
        from .lookup_table_fitting import fit_lookup_table_2d

        return fit_lookup_table_2d(
            xp,
            yp,
            x_data,
            y_data,
            z_data,
            weights=weights,
            smoothness=smoothness,
            **block_kwargs,
        )



# T-117-followup-bus-namedtuple end-of-file marker.


# ---------------------------------------------------------------------------
# T-114-followup-phase3-2d-cubic — N-D lookup-table block.
#
# Mirrors the ``LookupTable1d`` / ``LookupTable2d`` API for the N-D case.
# Delegates to ``jaxonomy.library.lookup_table.interp_nd`` (multilinear,
# implemented as N successive 1-D linear interpolations under the hood
# because JAX has no ``jnp.interpn`` today).
#
# Lives at the bottom of primitives.py inside a clearly marked section
# to keep the diff disjoint from concurrent work in T-105 / T-122.
# ---------------------------------------------------------------------------


class LookupTableND(LeafSystem):
    """Interpolate the input signal into a static N-D lookup table.

    Generalises :class:`LookupTable1d` and :class:`LookupTable2d` to an
    arbitrary number of axes.  The block takes a single input port whose
    value is a length-``N`` query vector ``[q_1, ..., q_N]`` and returns
    the multilinearly interpolated table value at that point.

    Implementation: delegates to
    :func:`jaxonomy.library.lookup_table.interp_nd`, which performs
    ``N`` successive 1-D linear interpolations along each axis (no
    ``jnp.interpn`` exists today — see the deeper-followup note in
    that function's docstring).

    Input ports:
        ``(0)`` — the query vector, shape ``(N,)`` where ``N`` is the
        number of grid axes.

    Output ports:
        ``(0)`` — the multilinearly interpolated table value.

    Parameters:
        grid_axes:
            Tuple of ``N`` strictly-increasing 1-D breakpoint arrays.
            The ``i``-th array has length ``B_i`` and corresponds to
            axis ``i`` of ``output_array``.
        output_array:
            Sample values, shape ``(B_1, B_2, ..., B_N)``.
            ``output_array[i_1, ..., i_N] = f(grid_axes[0][i_1], ...,
            grid_axes[N-1][i_N])``.
        interpolation:
            Currently only ``"linear"`` (multilinear).  Reserved for
            future N-D smooth methods (filed under
            ``T-114-followup-phase4-nd-cubic``).
        extrapolation (optional):
            One of ``"clip"`` (default — clips each coordinate to its
            axis range, the standard lookup-table default), ``"linear"`` (multilinear
            continuation past the grid), or ``"nan"`` (returns NaN
            whenever any coordinate is outside its axis range).
        dtype (optional):
            If set (e.g. ``jnp.float32``), the block's grid arrays and
            output array are cast to this dtype on construction.
            Mirrors the per-block dtype contract of ``LookupTable1d``.

    Notes:
        Differentiable through both the query vector and the table
        values (modulo the discrete bucket-index ``searchsorted``,
        whose gradient is piecewise constant — within a cell the
        gradient is exact).
    """

    def __init__(
        self,
        grid_axes,
        output_array,
        interpolation="linear",
        dtype=None,
        extrapolation="clip",
        **kwargs,
    ):
        # Per-block dtype + active precision policy fallback, matching
        # the LookupTable1d / LookupTable2d contract.  Stored outside
        # the @parameters list so the kwarg does not round-trip through
        # model JSON or get JAX-traced.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype

        if extrapolation not in ("clip", "linear", "nan"):
            raise ValueError(
                f"LookupTableND: extrapolation must be one of "
                f"('clip','linear','nan'), got {extrapolation!r}"
            )
        self._extrapolation = extrapolation

        if interpolation != "linear":
            raise NotImplementedError(
                f"LookupTableND only supports interpolation='linear' "
                f"(multilinear) today; got {interpolation!r}.  N-D smooth "
                f"methods are filed as T-114-followup-phase4-nd-cubic."
            )
        self._interpolation = interpolation

        # Eagerly validate grid + table shapes so misconfigured blocks
        # fail at construction rather than during context build.
        if not isinstance(grid_axes, (tuple, list)) or len(grid_axes) == 0:
            raise ValueError(
                f"LookupTableND: grid_axes must be a non-empty tuple/list "
                f"of 1-D breakpoint arrays, got {type(grid_axes).__name__}"
            )
        n_axes = len(grid_axes)
        for i, axis in enumerate(grid_axes):
            arr = np.asarray(axis)
            if arr.ndim != 1:
                raise ValueError(
                    f"LookupTableND: grid_axes[{i}] must be 1-D, got shape "
                    f"{arr.shape}"
                )
            if arr.size >= 2 and not np.all(np.diff(arr) > 0):
                raise ValueError(
                    f"LookupTableND: grid_axes[{i}] must be strictly "
                    f"monotonically increasing"
                )
        out_arr = np.asarray(output_array)
        if out_arr.ndim != n_axes:
            raise ValueError(
                f"LookupTableND: output_array.ndim ({out_arr.ndim}) must "
                f"equal len(grid_axes) ({n_axes})"
            )
        expected_shape = tuple(int(np.asarray(g).shape[0]) for g in grid_axes)
        if out_arr.shape != expected_shape:
            raise ValueError(
                f"LookupTableND: output_array.shape {out_arr.shape} must "
                f"match the grid shape {expected_shape}"
            )

        # Cast (or copy) into the backend's array type with the requested
        # dtype (or the default).  Stored on ``self`` for use in the
        # output computation closure.
        if self._dtype is not None:
            self._grid_axes = tuple(
                npa.asarray(np.asarray(g)).astype(self._dtype)
                for g in grid_axes
            )
            self._output_array = npa.asarray(out_arr).astype(self._dtype)
        else:
            self._grid_axes = tuple(npa.array(np.asarray(g)) for g in grid_axes)
            self._output_array = npa.array(out_arr)

        super().__init__(**kwargs)
        self.declare_input_port()

        from .lookup_table import interp_nd as _interp_nd

        extrapolation_local = self._extrapolation
        grid_local = self._grid_axes
        values_local = self._output_array

        def _compute(_time, _state, *inputs, **_params):
            (query,) = inputs
            return _interp_nd(
                grid_local,
                values_local,
                query,
                method="linear",
                extrapolation=extrapolation_local,
            )

        self.declare_output_port(
            _compute,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )



# T-122-followup-lfsr end-of-file marker.


# ---------------------------------------------------------------------------
# T-114-followup-prelookup — Prelookup + InterpolationUsingPrelookup pair.
#
# The standard ``Prelookup``/``InterpolationUsingPrelookup`` block pair is
# an optimisation for the case where the SAME query coordinate is used to
# look up MANY tables (think 10 maps sharing one engine-RPM axis).
# Without ``Prelookup`` each downstream :class:`LookupTable1d` re-runs the
# binary search; with it the search runs ONCE in ``Prelookup`` and the
# (index, fraction) pair fans out to every consumer.
#
# Architecture: ``Prelookup`` outputs a 2-tuple ``(index, fraction)``
# packaged as a ``collections.namedtuple`` so the signal flows through
# ``jit``/``vmap``/``grad`` cleanly (NamedTuples are first-class JAX
# pytrees -- same trick as ``BusCreator``).  ``InterpolationUsingPrelookup``
# unpacks the tuple and applies the standard linear blend
# ``(1 - alpha) * yp[i] + alpha * yp[i+1]``.
#
# Differentiability: the discrete bucket index ``i`` is non-differentiable
# (gradient is piecewise constant -- same as every other ``searchsorted``-
# based block in the library); the gradient flows through ``alpha`` (via
# the smooth fraction computation) and through ``output_array`` (via the
# convex combination of the two flanking entries).  This matches the
# differentiability story of :class:`LookupTable1d` itself.
#
# OOB handling: ``Prelookup`` clips the bucket index to ``[0, n-2]`` so
# the ``i + 1`` neighbour stays in range.  The ``extrapolation`` kwarg on
# ``Prelookup`` (mirrored on ``InterpolationUsingPrelookup`` for API
# symmetry and so the pair agrees on intent) picks the OOB policy:
#
#   * ``"clip"`` (default, byte-equivalent to T-114-fu-prelookup):
#     ``alpha`` is clipped to ``[0, 1]`` so OOB queries collapse to the
#     nearest endpoint -- matches :class:`LookupTable1d` ``"clip"``.
#   * ``"linear"``: ``alpha`` is NOT clipped; since the bucket index is
#     clamped to ``[0, n-2]``, an OOB query gives ``alpha < 0`` (left
#     side) or ``alpha > 1`` (right side), and the downstream blend
#     ``(1 - alpha) * yp[i] + alpha * yp[i+1]`` extends the boundary
#     slope linearly -- matches :class:`LookupTable1d` ``"linear"``.
#   * ``"nan"``: ``alpha`` is set to NaN when the query is OOB; the
#     downstream blend then propagates NaN unconditionally -- matches
#     :class:`LookupTable1d` ``"nan"``.
#
# ``InterpolationUsingPrelookup`` validates that its own ``extrapolation``
# kwarg matches the producer ``Prelookup`` only at construction (we can't
# bind the producer cheaply), but operationally it just consumes
# ``alpha`` -- the OOB math is all done in the upstream alpha
# computation, by design.  The kwarg exists on both sides for API
# symmetry / explicit user intent / IDE discoverability.
#
# Honest fallback: if NamedTuple-typed tuple signals turn out to break
# any downstream framework path we have not yet exercised, the fix is to
# split into two output ports on ``Prelookup`` (one int port for the
# index, one float port for the fraction) and have
# ``InterpolationUsingPrelookup`` declare two input ports.  We took the
# tuple path first because :class:`BusCreator` already demonstrates the
# NamedTuple-signal pattern works end-to-end.
#
# Lives at the bottom of primitives.py inside a clearly marked section
# to keep the diff disjoint from concurrent work in T-127-fu (which owns
# the mid-file ``PIDController2DOF`` class) and T-126-fu (which lives
# under jaxonomy/uq/).
# ---------------------------------------------------------------------------


class _PrelookupResult(NamedTuple):
    """Output of :class:`Prelookup`: a (bucket_index, alpha) pair.

    ``index`` is the lower-bracket grid index ``i`` such that the query
    falls between ``xp[i]`` and ``xp[i + 1]`` (both clipped to the grid
    range so ``i + 1`` is in bounds).  ``fraction`` is the linear-interp
    blend weight ``alpha = (x - xp[i]) / (xp[i+1] - xp[i])``.  The
    extrapolation policy chosen on the upstream :class:`Prelookup`
    determines whether ``alpha`` is clipped to ``[0, 1]`` (``"clip"``),
    left raw to let the downstream blend extend boundary slopes
    (``"linear"``), or set to NaN to propagate NaN downstream
    (``"nan"``).

    Packaged as a ``NamedTuple`` so the (index, fraction) signal is a
    JAX pytree -- survives ``jit``/``vmap``/``grad`` without extra
    ``register_pytree_node`` boilerplate.
    """

    index: "Array"
    fraction: "Array"


class Prelookup(LeafSystem):
    """Compute the (bucket_index, fraction) pair for a query against a
    precomputed grid.

    This is the upstream half of the standard
    ``Prelookup``/``InterpolationUsingPrelookup`` pair.  Pair with one or
    more :class:`InterpolationUsingPrelookup` blocks downstream -- each
    can interpolate a DIFFERENT output table that shares the same grid
    axis without re-running the bucket search.

    The marketing wedge: when N downstream tables share one query axis
    (e.g. 10 lookup maps on a common engine-RPM input), this saves
    ``N - 1`` binary searches per evaluation.

    Input ports:
        ``(0)`` -- the query coordinate (scalar).

    Output ports:
        ``(0)`` -- a NamedTuple with fields ``(index, fraction)`` ready
        to plug into one or more :class:`InterpolationUsingPrelookup`
        blocks.

    Parameters:
        input_array:
            1-D, strictly-increasing grid of breakpoints.  Stored
            verbatim for the bucket search.
        dtype (optional):
            If set (e.g. ``jnp.float32``), the grid array is cast to this
            dtype on construction.  Mirrors the per-block dtype contract
            of :class:`LookupTable1d`.
        extrapolation (optional, T-114-fu-prelookup-extrap):
            Out-of-range policy for the ``alpha`` blend weight.  One of
            ``"clip"`` (default; alpha clamped to ``[0, 1]`` so OOB
            queries map to the nearest endpoint), ``"linear"`` (alpha
            left raw -- the downstream blend extends the boundary slope
            linearly), or ``"nan"`` (alpha set to NaN on OOB queries --
            the downstream blend propagates NaN).  All three modes
            match the corresponding :class:`LookupTable1d`
            extrapolation policies byte-for-byte.  Any paired
            :class:`InterpolationUsingPrelookup` should declare the
            SAME mode for API agreement (the math is owned by
            ``Prelookup``'s alpha computation).

    Notes:
        Differentiable through the query coordinate via ``fraction``
        (the discrete ``index`` is piecewise-constant).
    """

    def __init__(self, input_array, dtype=None, extrapolation="clip", **kwargs):
        # Per-block dtype + active precision policy fallback, matching
        # the LookupTable1d contract.  Stored outside the @parameters
        # list so the kwarg does not round-trip through model JSON or
        # get JAX-traced.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype

        # T-114-fu-prelookup-extrap -- validate the extrapolation kwarg
        # eagerly so a bad value raises ValueError on construction
        # rather than at trace time.  Stored outside the @parameters
        # list (Python-string kwarg, never JAX-traced).
        if extrapolation not in ("clip", "linear", "nan"):
            raise ValueError(
                f"Prelookup: extrapolation must be one of "
                f"('clip','linear','nan'), got {extrapolation!r}"
            )
        self._extrapolation = extrapolation

        # Eagerly validate the grid up front so the failure mode is a
        # clear ValueError on construction, not a cryptic shape error
        # at trace time.
        _input_np = np.asarray(input_array)
        if _input_np.ndim != 1:
            raise ValueError(
                f"Prelookup: input_array must be 1-D, got shape "
                f"{_input_np.shape}"
            )
        if _input_np.size < 2:
            raise ValueError(
                f"Prelookup: input_array must have at least 2 entries, "
                f"got shape {_input_np.shape}"
            )
        if not np.all(np.diff(_input_np) > 0):
            raise ValueError(
                f"Prelookup: input_array must be strictly monotonically "
                f"increasing, got {list(_input_np)}"
            )

        if self._dtype is not None:
            self._input_array = npa.asarray(_input_np).astype(self._dtype)
        else:
            self._input_array = npa.array(_input_np)

        super().__init__(**kwargs)
        self.declare_input_port()

        # Capture the grid in a local so the closure does not pull
        # ``self`` into the JAX trace.
        xp_local = self._input_array
        n_local = int(self._input_array.shape[0])
        extrap_local = self._extrapolation

        def _compute(_time, _state, *inputs, **_params):
            (x_query,) = inputs
            # Clip bucket index to [0, n - 2] so the i + 1 neighbour
            # downstream stays in range.  Same convention as the
            # ``interp_1d`` / ``interp_nd`` backends.
            i = npa.clip(
                npa.searchsorted(xp_local, x_query, side="right") - 1,
                0,
                n_local - 2,
            )
            x0 = xp_local[i]
            x1 = xp_local[i + 1]
            alpha = (x_query - x0) / (x1 - x0)
            # T-114-fu-prelookup-extrap -- apply the OOB policy to
            # alpha.  The downstream blend (1 - alpha) * yp[i] + alpha
            # * yp[i+1] then naturally produces:
            #   * "clip"   -- nearest endpoint (alpha in [0, 1]);
            #   * "linear" -- boundary-slope extension (alpha raw);
            #   * "nan"    -- NaN propagation (alpha is NaN past edges).
            if extrap_local == "clip":
                alpha = npa.clip(alpha, 0.0, 1.0)
            elif extrap_local == "nan":
                # Out-of-range mask uses the ORIGINAL grid endpoints
                # ``xp[0]``/``xp[-1]`` (not the clipped bucket), so a
                # query exactly on a breakpoint is in-range.  ``npa``
                # falls back to ``jnp`` for ``where``/``isnan`` under a
                # JAX trace, so this is differentiable around the
                # finite (non-NaN) branch.
                oob = (x_query < xp_local[0]) | (x_query > xp_local[-1])
                nan_val = npa.asarray(npa.nan, dtype=alpha.dtype)
                alpha = npa.where(oob, nan_val, alpha)
            # else "linear": alpha left raw -- with i clamped to
            # [0, n-2], an OOB query gets alpha < 0 (left) or alpha > 1
            # (right), and the downstream blend extends the boundary
            # slope linearly.  No-op on this branch.
            return _PrelookupResult(index=i, fraction=alpha)

        # NOTE: deliberately NOT passing ``default_value=`` here -- the
        # framework's ``declare_output_port`` would call
        # ``npa.array(default_value)`` and flatten our NamedTuple into
        # a plain 1-D array, losing the tuple type.  Letting the
        # framework lazily compute the default by invoking ``_compute``
        # on a dummy context yields the correct NamedTuple-typed
        # default.  Same trick as :class:`BusCreator`.
        self.declare_output_port(
            _compute,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    @property
    def input_array(self):
        """The breakpoint array used for the bucket search."""
        return self._input_array

    @property
    def extrapolation(self):
        """The OOB policy applied to ``alpha`` (``"clip"``/``"linear"``/``"nan"``)."""
        return self._extrapolation


class InterpolationUsingPrelookup(LeafSystem):
    """Interpolate a static ``output_array`` using a precomputed
    (index, fraction) tuple from :class:`Prelookup`.

    This is the downstream half of the standard
    ``Prelookup``/``InterpolationUsingPrelookup`` pair.  Plug as many of
    these as you like into a single :class:`Prelookup`'s output port --
    each one interpolates its OWN ``output_array`` against the shared
    (index, fraction) signal, avoiding the redundant binary searches you
    would do with N independent :class:`LookupTable1d` blocks.

    Input ports:
        ``(0)`` -- the (index, fraction) NamedTuple produced by an
        upstream :class:`Prelookup` block.

    Output ports:
        ``(0)`` -- the linearly-interpolated value
        ``(1 - alpha) * output_array[i] + alpha * output_array[i + 1]``.

    Parameters:
        output_array:
            1-D table values, shape ``(N,)`` where ``N == len(prelookup
            input_array)``.  ``output_array[i]`` corresponds to the
            i-th breakpoint in the upstream :class:`Prelookup`'s grid.
        dtype (optional):
            If set (e.g. ``jnp.float32``), the table values are cast to
            this dtype on construction.  Mirrors the per-block dtype
            contract of :class:`LookupTable1d`.
        extrapolation (optional, T-114-fu-prelookup-extrap):
            Out-of-range policy.  Must match the upstream
            :class:`Prelookup`'s ``extrapolation`` kwarg (the math is
            all done in the producer's ``alpha`` computation; this
            block just consumes ``alpha``).  Exists on this side of the
            API for symmetry, IDE discoverability and so model JSON /
            user code make intent explicit.  One of ``"clip"`` (default,
            byte-equivalent to T-114-fu-prelookup), ``"linear"``, or
            ``"nan"``.

    Notes:
        Differentiable through ``output_array`` (every time-step the
        interpolation is a convex combination of two of its entries; the
        gradient w.r.t. the picked entries is exact).  The fraction-side
        gradient flows through the upstream :class:`Prelookup`'s
        ``alpha`` computation; the discrete ``index`` is piecewise
        constant (expected -- same as every ``searchsorted``-based block
        in the library).

        Today only linear interpolation is
        supported -- PCHIP/Akima downstream interpolation is a deeper
        followup (``T-114-followup-prelookup-cubic``).
    """

    def __init__(self, output_array, dtype=None, extrapolation="clip", **kwargs):
        # Per-block dtype + active precision policy fallback, matching
        # the LookupTable1d / Prelookup contract.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype

        # T-114-fu-prelookup-extrap -- validate the kwarg eagerly.  The
        # producer ``Prelookup`` does all the alpha math; this block
        # just records the user's declared intent for API symmetry.
        if extrapolation not in ("clip", "linear", "nan"):
            raise ValueError(
                f"InterpolationUsingPrelookup: extrapolation must be one "
                f"of ('clip','linear','nan'), got {extrapolation!r}"
            )
        self._extrapolation = extrapolation

        _out_np = np.asarray(output_array)
        if _out_np.ndim != 1:
            raise ValueError(
                f"InterpolationUsingPrelookup: output_array must be 1-D, "
                f"got shape {_out_np.shape}"
            )
        if _out_np.size < 2:
            raise ValueError(
                f"InterpolationUsingPrelookup: output_array must have at "
                f"least 2 entries, got shape {_out_np.shape}"
            )

        if self._dtype is not None:
            self._output_array = npa.asarray(_out_np).astype(self._dtype)
        else:
            self._output_array = npa.array(_out_np)

        super().__init__(**kwargs)
        self.declare_input_port()

        # Capture the table in a local so the closure does not pull
        # ``self`` into the JAX trace.
        yp_local = self._output_array

        def _compute(_time, _state, *inputs, **_params):
            (prelookup_result,) = inputs
            # Unpack the NamedTuple.  ``index`` and ``fraction`` are
            # plain JAX arrays produced by the upstream Prelookup
            # closure; both are pytree-friendly via NamedTuple
            # registration.
            i = prelookup_result.index
            alpha = prelookup_result.fraction
            return (1.0 - alpha) * yp_local[i] + alpha * yp_local[i + 1]

        self.declare_output_port(
            _compute,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    @property
    def output_array(self):
        """The 1-D table being interpolated."""
        return self._output_array

    @property
    def extrapolation(self):
        """The declared OOB policy (must match the upstream Prelookup)."""
        return self._extrapolation


# ---------------------------------------------------------------------------
# T-114-followup-prelookup-inverse — PrelookupInverse block.
#
# Use case: an implicit equation ``y = f(x)`` where ``f`` is represented by
# a 1-D lookup table.  Given a known ``y`` we want to recover ``x``.  When
# ``f`` is monotonic on the grid this is well-defined: find ``(i, alpha)``
# such that
#
#     output_array[i] + alpha * (output_array[i+1] - output_array[i]) ≈ y
#
# i.e. invert the table value-axis.  Emit the same NamedTuple-typed
# ``(index, fraction)`` signal that :class:`Prelookup` emits, so the result
# plugs straight into an existing :class:`InterpolationUsingPrelookup`
# block connected to a DIFFERENT output table (typically the inverse table
# or a related table sharing the breakpoint axis).
#
# Headline use case: gain scheduling where ``y = a*x + b`` (or any
# monotonic curve); given the scheduled gain ``y``, recover the operating
# point ``x`` for a downstream controller, all without re-running a
# Newton iteration at every step.
#
# Differentiability: same story as :class:`Prelookup`.  The discrete
# bucket index is piecewise constant; the gradient flows through
# ``fraction`` (w.r.t. the query ``y``) and through the table values
# (``output_array``).
#
# Monotonicity: validated at construction.  Both strictly-increasing and
# strictly-decreasing tables are supported; for a decreasing table we
# flip ``searchsorted`` semantics by reversing the array before the
# bucket lookup and then re-mapping the index back.  Non-monotonic
# tables raise ``ValueError``.
#
# Honest fallback: only the ``"clip"`` extrapolation policy is shipped in
# this followup -- queries outside ``[min(output_array), max(output_array)]``
# collapse to the nearest endpoint, matching the default
# :class:`Prelookup` behaviour.  ``"linear"``/``"nan"`` are deeper
# followups (``T-114-followup-prelookup-inverse-extrap``) because the
# OOB definition on a value-axis depends on extrapolation outside the
# monotone range, which interacts non-trivially with the
# direction-flipping logic.
# ---------------------------------------------------------------------------


class PrelookupInverse(LeafSystem):
    """Compute the (bucket_index, fraction) pair for an INVERSE-direction
    lookup against a strictly-monotonic value array.

    Forward :class:`Prelookup` answers: given ``x``, find ``(i, alpha)``
    s.t. ``xp[i] + alpha * (xp[i+1] - xp[i]) ≈ x``.

    Inverse :class:`PrelookupInverse` answers: given ``y``, find
    ``(i, alpha)`` s.t. ``yp[i] + alpha * (yp[i+1] - yp[i]) ≈ y``.

    The output is the same NamedTuple-typed
    :class:`_PrelookupResult` produced by :class:`Prelookup`, so it plugs
    straight into one or more :class:`InterpolationUsingPrelookup` blocks
    connected to OTHER tables -- typically the inverse table that maps
    ``i`` back to the recovered ``x`` (for example the breakpoints of the
    forward table).

    Marketing wedge: implicit equations ``y = f(x)`` where ``f`` is a
    monotonic 1-D table -- gain scheduling, sensor calibration, etc.

    Input ports:
        ``(0)`` -- the query coordinate in the OUTPUT space (``y``).

    Output ports:
        ``(0)`` -- a :class:`_PrelookupResult` NamedTuple
        ``(index, fraction)`` ready to plug into an
        :class:`InterpolationUsingPrelookup` block.

    Parameters:
        output_array:
            1-D, strictly-monotonic (increasing OR decreasing) array of
            values to invert.  Stored verbatim for the bucket search;
            decreasing tables are handled by reversing the search
            direction.
        dtype (optional):
            If set, the value array is cast to this dtype on
            construction.  Mirrors the :class:`Prelookup` /
            :class:`LookupTable1d` dtype contract.
        extrapolation (optional):
            Only ``"clip"`` is supported in this followup -- queries
            outside the monotone range collapse to the nearest endpoint.
            ``"linear"``/``"nan"`` are filed as
            ``T-114-followup-prelookup-inverse-extrap`` because the OOB
            definition on a value-axis interacts with the
            direction-flipping logic non-trivially.

    Notes:
        Differentiable through the query coordinate and through
        ``output_array``.  Non-monotonic ``output_array`` raises
        ``ValueError`` at construction time.
    """

    def __init__(
        self, output_array, dtype=None, extrapolation="clip", **kwargs
    ):
        # Per-block dtype + active precision policy fallback, mirroring
        # Prelookup / LookupTable1d.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype

        # Honest fallback: only "clip" ships in this followup.  Reject
        # "linear"/"nan" with a clear error pointing at the deeper
        # followup ticket so callers know it is on the roadmap.
        if extrapolation != "clip":
            if extrapolation in ("linear", "nan"):
                raise NotImplementedError(
                    f"PrelookupInverse: extrapolation={extrapolation!r} is "
                    f"a deeper followup (T-114-followup-prelookup-inverse-"
                    f"extrap); only 'clip' is shipped today."
                )
            raise ValueError(
                f"PrelookupInverse: extrapolation must be 'clip' "
                f"(other modes are deferred), got {extrapolation!r}"
            )
        self._extrapolation = extrapolation

        _out_np = np.asarray(output_array)
        if _out_np.ndim != 1:
            raise ValueError(
                f"PrelookupInverse: output_array must be 1-D, got shape "
                f"{_out_np.shape}"
            )
        if _out_np.size < 2:
            raise ValueError(
                f"PrelookupInverse: output_array must have at least 2 "
                f"entries, got shape {_out_np.shape}"
            )

        # Monotonicity check.  Strictly increasing OR strictly decreasing
        # is fine; anything else is ambiguous to invert.
        diffs = np.diff(_out_np)
        if np.all(diffs > 0):
            self._direction = "increasing"
        elif np.all(diffs < 0):
            self._direction = "decreasing"
        else:
            raise ValueError(
                f"PrelookupInverse: output_array must be strictly "
                f"monotonic (increasing or decreasing) to invert; got "
                f"{list(_out_np)}"
            )

        if self._dtype is not None:
            self._output_array = npa.asarray(_out_np).astype(self._dtype)
        else:
            self._output_array = npa.array(_out_np)

        super().__init__(**kwargs)
        self.declare_input_port()

        # Capture the table + direction in locals so the closure does
        # not pull ``self`` into the JAX trace.
        yp_local = self._output_array
        n_local = int(self._output_array.shape[0])
        direction_local = self._direction

        def _compute(_time, _state, *inputs, **_params):
            (y_query,) = inputs
            if direction_local == "increasing":
                # Standard searchsorted on the value-axis.
                i = npa.clip(
                    npa.searchsorted(yp_local, y_query, side="right") - 1,
                    0,
                    n_local - 2,
                )
                y0 = yp_local[i]
                y1 = yp_local[i + 1]
            else:
                # Decreasing case: searchsorted needs an ascending array
                # to work, so search the reversed array and re-map the
                # index back into the original orientation.  After
                # remap, ``i`` still indexes into yp_local with the
                # invariant that yp_local[i] >= y_query >= yp_local[i+1]
                # (modulo the boundary clip).
                yp_rev = yp_local[::-1]
                j = npa.clip(
                    npa.searchsorted(yp_rev, y_query, side="right") - 1,
                    0,
                    n_local - 2,
                )
                # Reversed bucket j corresponds to original bucket
                # (n-2) - j.
                i = (n_local - 2) - j
                y0 = yp_local[i]
                y1 = yp_local[i + 1]
            alpha = (y_query - y0) / (y1 - y0)
            # Only "clip" ships in this followup.  Collapses OOB queries
            # to the nearest endpoint.  ``alpha`` outside [0, 1] can
            # still arise when the query is exactly at a boundary
            # due to floating-point; the clip pins it.
            alpha = npa.clip(alpha, 0.0, 1.0)
            return _PrelookupResult(index=i, fraction=alpha)

        # Same NamedTuple-output-port trick as Prelookup: do NOT pass
        # ``default_value=`` (the framework would flatten the
        # NamedTuple); let it lazily compute via ``_compute``.
        self.declare_output_port(
            _compute,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    @property
    def output_array(self):
        """The 1-D monotonic value array being inverted."""
        return self._output_array

    @property
    def direction(self):
        """``"increasing"`` or ``"decreasing"`` -- monotonicity sense."""
        return self._direction

    @property
    def extrapolation(self):
        """The OOB policy (always ``"clip"`` in this followup)."""
        return self._extrapolation


# T-117-followup-nested-buses end-of-file marker.


# ===========================================================================
# T-114-followup-table-search — TableSearch block.
# ---------------------------------------------------------------------------
# The standard "Direct Lookup" / "Search" pattern: given a strictly-
# monotonic 1-D table ``xp`` and a scalar query ``x``, return the bucket
# index ``i`` such that ``xp[i] <= x < xp[i+1]``.  Out-of-range queries
# clamp to the left endpoint (``i = 0``) or the right endpoint
# (``i = n - 1``) -- the standard "clip" policy.
#
# Different from :class:`Prelookup` in that it returns ONLY the discrete
# index -- no fractional alpha.  Useful for binning, threshold detection,
# inverse-table indexing, and any pattern where the downstream block only
# needs to know WHICH bucket the query lands in.
#
# Two ``mode`` options:
#   * ``"binary"`` -- ``jnp.searchsorted`` (O(log n)); the default for
#     production use.
#   * ``"linear"`` -- a linear scan via ``npa.sum`` over a comparison mask
#     (O(n)); simpler / easier to reason about for very small grids.
# Both modes return byte-identical results on monotonic grids.
#
# Differentiability: the bucket index is a step function of the query
# coordinate, so the gradient is zero almost everywhere.  We wrap the
# returned index in ``jax.lax.stop_gradient`` to make the
# non-differentiability EXPLICIT -- no spurious zero-gradient surprises
# under ``grad``, and the downstream consumer cannot accidentally rely on
# a piecewise-constant gradient signal.
#
# Output dtype: the index is returned as a float (so it composes with
# the rest of the library's float-defaulting numeric pipeline -- T-005
# default-float64 policy).  Callers who need integer indexing can cast
# explicitly via ``jnp.int32``.
# ===========================================================================


class TableSearch(LeafSystem):
    """Search a monotonic table for the bucket containing a query value.

    Given a strictly-increasing 1-D grid ``xp`` of length ``n`` and a
    scalar query ``x``, returns the bucket index ``i`` (as a float) such
    that ``xp[i] <= x < xp[i+1]``.  Out-of-range queries clamp to the
    nearest endpoint: ``x < xp[0]`` returns ``0``; ``x >= xp[-1]``
    returns ``n - 1``.

    The standard "Direct Lookup" pattern.  Different from
    :class:`Prelookup` in that the output is just the bucket index --
    no fractional ``alpha`` is computed.  Useful for binning,
    threshold detection, and inverse-table indexing.

    Input ports:
        ``(0)`` -- scalar query coordinate ``x``.

    Output ports:
        ``(0)`` -- scalar bucket index, returned as a float (so it
        composes with the float-defaulting numeric pipeline).  The
        output is wrapped in ``jax.lax.stop_gradient`` -- gradient is
        zero almost everywhere by construction (step function), so we
        make that non-differentiability explicit to avoid spurious
        grad-flow surprises.

    Parameters:
        xp:
            1-D, strictly-monotonically-increasing grid of breakpoints
            (length >= 2).  Stored verbatim for the bucket search.
        mode:
            ``"binary"`` (default) uses ``jnp.searchsorted`` (O(log n));
            ``"linear"`` uses a linear scan via a sum over a comparison
            mask (O(n)).  Both modes return byte-identical results on
            valid (strictly-monotonic) grids.
        dtype (optional):
            If set (e.g. ``jnp.float32``), the grid array is cast to
            this dtype on construction.  Mirrors the per-block dtype
            contract of :class:`LookupTable1d` / :class:`Prelookup`.

    Notes:
        Index is wrapped in ``jax.lax.stop_gradient`` -- the gradient
        through the query coordinate is zero, by construction.  Callers
        who need a differentiable index-like quantity should use
        :class:`Prelookup` (which exposes the fractional ``alpha``) or
        :class:`LookupTable1d` directly.
    """

    def __init__(self, xp, mode="binary", dtype=None, **kwargs):
        # Per-block dtype + active precision policy fallback, matching
        # the LookupTable1d / Prelookup contract.
        if dtype is None:
            from ..precision import active_precision_policy

            dtype = active_precision_policy()
        self._dtype = dtype

        if mode not in ("binary", "linear"):
            raise ValueError(
                f"TableSearch: mode must be one of ('binary','linear'), "
                f"got {mode!r}"
            )
        self._mode = mode

        # Eagerly validate the grid up front so the failure mode is a
        # clear ValueError on construction, not a cryptic shape error
        # at trace time.
        _xp_np = np.asarray(xp)
        if _xp_np.ndim != 1:
            raise ValueError(
                f"TableSearch: xp must be 1-D, got shape {_xp_np.shape}"
            )
        if _xp_np.size < 2:
            raise ValueError(
                f"TableSearch: xp must have at least 2 entries, got "
                f"shape {_xp_np.shape}"
            )
        if not np.all(np.diff(_xp_np) > 0):
            raise ValueError(
                f"TableSearch: xp must be strictly monotonically "
                f"increasing, got {list(_xp_np)}"
            )

        if self._dtype is not None:
            self._xp = npa.asarray(_xp_np).astype(self._dtype)
        else:
            self._xp = npa.array(_xp_np)

        super().__init__(**kwargs)
        self.declare_input_port()

        # Capture locals so the closure does not pull ``self`` into the
        # JAX trace.
        xp_local = self._xp
        n_local = int(self._xp.shape[0])
        mode_local = self._mode

        # Lazy-import jax.lax for the stop_gradient wrap.  Keeping this
        # at module-import time would defeat the lazy-loader pattern
        # used by the rest of this file for ``equinox``.
        from jax import lax as _jlax

        def _compute(_time, _state, *inputs, **_params):
            (x_query,) = inputs
            if mode_local == "binary":
                # ``side="right"`` then subtract 1 gives the bucket index
                # ``i`` s.t. ``xp[i] <= x < xp[i+1]``.  Clip to
                # ``[0, n - 1]`` for OOB clamping (left -> 0, right ->
                # n - 1; matches the "clip" policy used elsewhere in
                # T-114).
                i = npa.clip(
                    npa.searchsorted(xp_local, x_query, side="right") - 1,
                    0,
                    n_local - 1,
                )
            else:
                # ``"linear"`` mode -- count the breakpoints strictly
                # less-or-equal to ``x_query``, then subtract 1 to land
                # on the bucket index.  Same OOB clamp semantics as the
                # binary path.  Implemented as a sum over a comparison
                # mask so it stays jit/vmap-safe.
                mask = xp_local <= x_query
                count = npa.sum(mask.astype(xp_local.dtype))
                i = npa.clip(count - 1, 0, n_local - 1)
            # Return as a float so the output composes with the
            # library's float-defaulting numeric pipeline (T-005).
            # Wrap in ``stop_gradient`` to make the
            # piecewise-constant-gradient non-differentiability
            # explicit.
            i_float = npa.asarray(i).astype(xp_local.dtype)
            return _jlax.stop_gradient(i_float)

        self.declare_output_port(
            _compute,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
            default_value=npa.asarray(0.0, dtype=xp_local.dtype),
        )

    @property
    def xp(self):
        """The 1-D strictly-increasing breakpoint array."""
        return self._xp

    @property
    def mode(self):
        """Search mode (``"binary"`` or ``"linear"``)."""
        return self._mode
