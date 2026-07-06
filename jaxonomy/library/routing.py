# SPDX-License-Identifier: MIT

"""Multiplexing, slicing, datatype conversion, and bus utilities."""

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
    "Demultiplexer",
    "Multiplexer",
    "Mux",
    "Demux",
    "IOPort",
    "Slice",
    "SignalDatatypeConversion",
    "BusCreator",
    "BusSelector",
    "BusMerge",
    "BusPassthrough",
    "BusUpdate",
    "bus_fields",
    "flatten_bus",
    "merge_buses",
    "unflatten_bus",
]



class Demultiplexer(LeafSystem):
    """Split a vector signal into its components.

    Input ports:
        (0) The vector signal to split.

    Output ports:
        (0..n_out-1) The components of the input signal.
    """

    def __init__(self, n_out, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()

        # Need a helper function so that the lambda captures the correct value of i
        # and doesn't use something that ends up fixed in scope.
        def _declare_output(i):
            def _compute_output(_time, _state, *inputs, **_params):
                (input_vec,) = inputs
                return input_vec[i]

            self.declare_output_port(
                _compute_output,
                prerequisites_of_calc=[self.input_ports[0].ticket],
            )

        for i in npa.arange(n_out):
            _declare_output(i)



class IOPort(FeedthroughBlock):
    """Simple class for organizing input/output ports for groups/submodels.

    Since these are treated as standalone blocks in the UI rather than specific
    input/output ports exported to the parent model, it is more straightforward
    to represent them that way here as well.

    This class represents a simple one-input, one-output feedthrough block where
    the feedthrough function is an identity.  The input (resp. output) port can then
    be exported to the parent model to create an Inport (resp. Outport).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: x, *args, **kwargs)



class Multiplexer(ReduceBlock):
    """Stack the input signals into a single output signal.

    Dispatches to `jax.numpy.hstack`, so see the JAX docs for details:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.hstack.html

    Input ports:
        (0..n_in-1) The input signals.

    Output ports:
        (0) The stacked output signal.
    """

    def __init__(self, n_in, *args, **kwargs):
        super().__init__(n_in, npa.hstack, *args, **kwargs)



class Slice(FeedthroughBlock):
    """Slice the input signal using Python indexing rules.

    Input ports:
        (0) The input signal.

    Output ports:
        (0) The sliced output signal.

    Parameters:
        slice_:
            The slice operator to apply to the input signal.  Must be specified as a
            string input, e.g. the output `u[1:3]` would be created with the block
            `Slice("1:3")`.

    Notes:
        Currently only up to 3-dimensional slices are supported.
    """

    @parameters(static=["slice_"])
    def __init__(self, slice_, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

    def initialize(self, slice_):
        # if slice was provided as numpy slice object, remove this before validating.
        if slice_.startswith("np.s_"):
            slice_ = slice_[len("np.s_") :]
        # if slice is wrapped in [], remove them temporarily.
        if slice_[0] == "[":
            slice_ = slice_[1:]
        if slice_[-1] == "]":
            slice_ = slice_[:-1]

        # validate slice_ and ensure no nefarious code.
        pattern = re.compile(r"^[0-9,:]+$")
        if not pattern.match(slice_):
            raise BlockParameterError(
                message=f"Slice block {self.name} detected invalid slice operator {slice_}. [] are optional. Valid examples: '1:3,4', '[:,4:10]'",
                parameter_name="slice_",
            )

        # replace the [] and eval to numpy slcie object
        slice_ = "np.s_[" + slice_ + "]"
        np_slice = eval(slice_)

        def _func(inp):
            return npa.array(inp)[np_slice]

        self.replace_op(_func)



class SignalDatatypeConversion(FeedthroughBlock):
    """Convert the input signal to a different data type.
    Input ports:
        (0) The input signal.
    Output ports:
        (0) The input signal converted to the specified data type.
    Parameters:
        dtype:
            The data type to which the input signal is converted.  Must be a valid
            NumPy data type, e.g. "float32", "int64", etc.
    """

    def _op(self, dtype, x):
        # This check makes the numpy backend strict like jax
        if npa.active_backend == "numpy" and isinstance(x, (list, tuple)):
            raise ValueError(
                "SignalDatatypeConversion block does not support list or tuple inputs."
            )

        return cond(
            isinstance(x, npa.ndarray),
            lambda x: npa.astype(x, dtype),
            lambda x: npa.array(x, dtype),
            x,
        )

    @parameters(static=["convert_to_type"])
    def __init__(self, convert_to_type, *args, **kwargs):
        super().__init__(partial(self._op, np.dtype(convert_to_type)), *args, **kwargs)

    def initialize(self, convert_to_type):
        self.dtype = np.dtype(convert_to_type)
        self.replace_op(partial(self._op, np.dtype(convert_to_type)))


# ---------------------------------------------------------------------------
# T-117 phase 1 — Mux / Demux signal-routing primitives.
#
# Positional pack/unpack of n homogeneous signals into a single
# array signal (and back). ``Mux`` is a thin wrapper around ``npa.stack``
# (so scalar inputs become a 1-D vector and same-shape vector inputs
# become a 2-D array along a new leading axis); ``Demux`` is its inverse,
# unstacking along axis 0 into ``n_outputs`` ports. Both blocks are
# differentiable through every input — that is the whole reason these
# exist as separate primitives instead of relying on the older
# ``Multiplexer`` block (which uses ``hstack`` and therefore flattens
# vector inputs into one long 1-D output).
#
# ``BusCreator`` / ``BusSelector`` are deferred: they require a
# NamedTuple-typed signal and a ``BusType`` registry on the framework
# side, which is a much larger lift. See T-117-followup-bus-namedtuple.
# ---------------------------------------------------------------------------


class Mux(ReduceBlock):
    """Stack ``n_inputs`` homogeneous signals into a single output signal.

    This is the standard ``Mux`` block. It dispatches to ``npa.stack``
    along axis 0, so:

    * ``Mux(3)([1.0, 2.0, 3.0]) -> array([1.0, 2.0, 3.0])`` (shape ``(3,)``).
    * ``Mux(2)([(1.0, 2.0), (3.0, 4.0)]) -> array([[1.0, 2.0],
      [3.0, 4.0]])`` (shape ``(2, 2)``).

    All inputs must be the same shape and dtype; this matches the
    conventional ``Mux`` semantics and ``npa.stack``'s broadcasting rules.

    For the older flatten-by-concatenation behavior (``hstack``), use
    :class:`Multiplexer` instead.

    Input ports:
        (0..n_inputs-1) The input signals (must share shape and dtype).

    Output ports:
        (0) The stacked output signal, with one extra leading axis.
    """

    def __init__(self, n_inputs, *args, **kwargs):
        super().__init__(n_inputs, npa.stack, *args, **kwargs)


class Demux(LeafSystem):
    """Unstack a single array input into ``n_outputs`` separate signals.

    This is the standard ``Demux`` block and the inverse of :class:`Mux`:
    given a 1-D input ``[a, b, c]`` it produces three scalar outputs
    ``a``, ``b``, ``c``; given a 2-D input of shape ``(n_outputs, k)``
    it produces ``n_outputs`` outputs of shape ``(k,)``.

    Internally this is index-based slicing along axis 0, which is fully
    differentiable through every output port (each output picks one
    slice of the input vector).

    Input ports:
        (0) The vector or array signal to split. Its leading axis must
            have length ``n_outputs``.

    Output ports:
        (0..n_outputs-1) The components of the input signal.
    """

    def __init__(self, n_outputs, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()

        # Helper closure so each output port captures its own ``i``.
        def _declare_output(i):
            def _compute_output(_time, _state, *inputs, **_params):
                (input_vec,) = inputs
                return input_vec[i]

            self.declare_output_port(
                _compute_output,
                prerequisites_of_calc=[self.input_ports[0].ticket],
            )

        for i in range(int(n_outputs)):
            _declare_output(i)



# T-122-followup-distributions end-of-file marker.


# ---------------------------------------------------------------------------
# T-117-followup-bus-namedtuple — BusCreator / BusSelector.
#
# Mux/Demux (T-117 phase 1, above) pack signals positionally — useful, but
# semantically poor for large signal groups (e.g. a "vehicle state" bus
# with position/velocity/acceleration components). BusCreator packs n
# inputs into a single output that is a ``collections.namedtuple``-typed
# pytree, with one named field per input port; BusSelector picks one
# named field back out.
#
# Why a NamedTuple? It is the cleanest JAX-pytree-compatible "named
# struct": registered automatically with ``jax.tree_util`` (no
# ``register_pytree_node`` boilerplate), survives ``jit``/``vmap``/
# ``grad`` cleanly, and behaves like a frozen dataclass at the Python
# level. ``dict`` would also work, but a NamedTuple gives us attribute
# access (``bus.position`` vs ``bus["position"]``) and a stable field
# order — important because our port-ordering API is also positional.
#
# A full ``BusType``/data-dictionary surface
# (per-field units, JSON-serializable schema registry, nested buses,
# heterogeneous-dtype enforcement) is not in scope here; the spec
# explicitly carved that out as a deeper followup. What ships:
#
#   * ``BusCreator(field_names)`` — n input ports → 1 NamedTuple output.
#   * ``BusSelector(field_name)`` — 1 NamedTuple input → 1 scalar output.
#
# Both blocks are differentiable through every field (the underlying ops
# are tuple construction + ``getattr``, both transparent to autodiff)
# and JIT-traceable (NamedTuples are pytrees).
#
# Honest fallback: if the NamedTuple-typed output ever turns out to
# break some part of simulator pytree handling that we haven't yet
# exercised, the fix is to swap the inner type for a plain ``dict``;
# the public API does not need to change. We took the typed-bus path
# first because the test suite exercises ``simulate`` end-to-end and
# confirms the NamedTuple flows through cleanly.
# ---------------------------------------------------------------------------


class BusCreator(LeafSystem):
    """Pack ``n = len(field_names)`` signals into a single named-bus output.

    The output is a ``collections.namedtuple`` (named ``"Bus"``) whose
    fields are exactly ``field_names`` in declaration order. NamedTuples
    are first-class JAX pytrees, so the bus signal flows through
    ``jax.jit``, ``vmap``, and ``grad`` without any extra registration.

    Pair with :class:`BusSelector` to pull individual fields back out
    downstream. Use this when signals share a logical group identity
    (e.g. ``("position", "velocity", "acceleration")`` for a vehicle
    state bus) and you would rather refer to them by name than by
    positional ``Mux``/``Demux`` index.

    Parameters:
        field_names: Tuple/list of strings — one name per input port,
            in declaration order. Must be unique, valid Python
            identifiers (NamedTuple constraint).
        field_units: Optional mapping from field name to :class:`Unit`
            (T-117-followup-bus-units). When supplied, each input port
            is declared with the corresponding ``units=`` and the
            output port carries a :class:`BusUnit` so downstream
            :class:`BusSelector` blocks can recover per-field units at
            connect time. When ``None`` (the default), input/output
            ports carry no unit metadata — byte-equivalent to the
            T-117-fu-bus-namedtuple shipping behaviour.
        field_shapes: Optional mapping from field name to a JAX-style
            shape tuple (T-117-followup-bus-array). When supplied, the
            named field carries an array of the declared shape rather
            than a scalar — useful for grouping e.g. an 8-element
            thermocouple readout under a single ``"sensors"`` slot
            without manually muxing. Fields not listed default to
            scalar shape ``()``. Default ``None`` is byte-equivalent
            to the T-117-fu-bus-namedtuple all-scalar behaviour.

    Input ports:
        ``(0..n-1)`` — one port per field name; values are packed into
        the corresponding slot of the output NamedTuple.

    Output ports:
        ``(0)`` — the bus, a NamedTuple with fields ``field_names``.
    """

    def __init__(
        self,
        field_names,
        *args,
        field_units=None,
        field_shapes=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Validate up front so the failure mode is a clear ValueError on
        # construction, not a cryptic NamedTuple "Type names and field
        # names must be valid identifiers" error from deep inside the
        # collections module.
        field_names = tuple(field_names)
        if len(field_names) == 0:
            raise ValueError(
                "BusCreator requires at least one field name; "
                "got an empty tuple."
            )
        if len(set(field_names)) != len(field_names):
            raise ValueError(
                f"BusCreator field_names must be unique; got {field_names!r}."
            )
        for fname in field_names:
            if not isinstance(fname, str) or not fname.isidentifier():
                raise ValueError(
                    f"BusCreator field name {fname!r} is not a valid "
                    "Python identifier (required for NamedTuple fields)."
                )

        # T-117-followup-bus-units: optional per-field unit propagation.
        # ``field_units=None`` keeps the historic default-off path
        # byte-equivalent (no BusUnit on the output port, no units on
        # input ports). When provided, we validate the keys match
        # ``field_names`` exactly so the user gets a clear error before
        # any port is declared.
        self._bus_unit = None
        per_input_units: dict[str, object] = {fname: None for fname in field_names}
        if field_units is not None:
            from ..framework.units import BusUnit as _BusUnit, Unit as _Unit

            field_units_dict = dict(field_units)
            extra = set(field_units_dict) - set(field_names)
            missing = set(field_names) - set(field_units_dict)
            if extra or missing:
                raise ValueError(
                    "BusCreator field_units keys must match field_names "
                    f"exactly; got extra={sorted(extra)!r}, "
                    f"missing={sorted(missing)!r}."
                )
            for fname, u in field_units_dict.items():
                if not isinstance(u, _Unit):
                    raise TypeError(
                        f"BusCreator field_units[{fname!r}] must be a Unit "
                        f"instance, got {type(u).__name__}: {u!r}."
                    )
                per_input_units[fname] = u
            self._bus_unit = _BusUnit(fields=field_units_dict)

        # T-117-followup-bus-array: optional per-field array shapes.
        # ``field_shapes=None`` keeps every field scalar (``()``), which
        # is byte-equivalent to the T-117-fu-bus-namedtuple all-scalar
        # behaviour. When supplied, fields not listed in the mapping
        # default to scalar so users only spell out the array fields.
        per_field_shapes: dict[str, tuple] = {fname: () for fname in field_names}
        if field_shapes is not None:
            field_shapes_dict = dict(field_shapes)
            extra = set(field_shapes_dict) - set(field_names)
            if extra:
                raise ValueError(
                    "BusCreator field_shapes keys must be a subset of "
                    f"field_names; got unknown keys={sorted(extra)!r}."
                )
            for fname, shape in field_shapes_dict.items():
                shape_t = tuple(shape)
                for dim in shape_t:
                    if not isinstance(dim, int) or dim < 0:
                        raise ValueError(
                            f"BusCreator field_shapes[{fname!r}] must be a "
                            f"tuple of non-negative ints, got {shape!r}."
                        )
                per_field_shapes[fname] = shape_t

        self._field_names = field_names
        self._field_shapes = per_field_shapes
        self._bus_type = namedtuple("Bus", field_names)

        input_tickets = []
        for fname in field_names:
            idx = self.declare_input_port(
                name=fname, units=per_input_units[fname]
            )
            input_tickets.append(self.input_ports[idx].ticket)

        def _compute_bus(_time, _state, *inputs, **_params):
            # ``inputs`` is exactly the tuple of upstream values in port
            # order, matching ``field_names`` by construction.
            return self._bus_type(*inputs)

        # NOTE: we deliberately do NOT pass ``default_value=`` to
        # ``declare_output_port`` here — see the leaf_system.py code path
        # at line 931, which calls ``npa.array(default_value)`` and
        # would flatten our NamedTuple into a plain 1-D array (losing
        # the bus type). Letting the framework lazily compute the
        # default by calling ``_compute_bus`` on a dummy context yields
        # the correct NamedTuple-typed default. T-005 default-float64
        # is preserved transitively via the upstream ``Constant``
        # blocks' float64 zeros.
        self.declare_output_port(
            _compute_bus,
            name="bus",
            prerequisites_of_calc=input_tickets,
            requires_inputs=True,
            units=self._bus_unit,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """The tuple of field names in declaration / port order."""
        return self._field_names

    @property
    def bus_type(self) -> type:
        """The underlying NamedTuple class for this bus."""
        return self._bus_type

    @property
    def bus_unit(self):
        """The compound :class:`BusUnit` for this bus, or ``None`` if
        ``field_units`` was not supplied at construction time."""
        return self._bus_unit

    @property
    def field_shapes(self) -> dict:
        """Mapping from field name to declared array shape tuple
        (T-117-followup-bus-array). Fields default to scalar ``()``
        when ``field_shapes`` is omitted at construction time."""
        return dict(self._field_shapes)


class BusSelector(LeafSystem):
    """Pull one named field out of a bus signal.

    The bus input is expected to be a NamedTuple-shaped value (typically
    produced by :class:`BusCreator`). The selected field is read with
    plain ``getattr``, so this block is the inverse of ``BusCreator``
    when wired correctly: ``BusSelector("a")(BusCreator(["a","b","c"])
    (a, b, c)) == a``.

    Both ``getattr`` and the NamedTuple constructor are transparent to
    JAX autodiff, so gradients flow cleanly from the selector output
    back to the upstream input that filled the corresponding bus slot.

    T-117-followup-bus-dot-path: ``field_name`` may contain dots to
    descend into nested bus signals — e.g.
    ``BusSelector("chassis.suspension.spring_force")`` extracts the
    leaf in one block instead of three cascaded selectors. The path is
    resolved via :func:`operator.attrgetter`, so each segment must
    name a valid NamedTuple field at its level. Each segment is
    validated as a Python identifier at construction time.

    Parameters:
        field_name: Name (or dot-separated path) of the bus field to
            extract. Raises ``AttributeError`` at execution time if
            the upstream bus does not have a field at any segment of
            the path.
        bus_unit: Optional :class:`BusUnit` describing the per-field
            units of the upstream bus (T-117-followup-bus-units). When
            supplied, the selector's *input* port is tagged with this
            ``BusUnit`` (so the connect-time check verifies that the
            upstream :class:`BusCreator` produced a compatible bus)
            and its *output* port is tagged with
            ``bus_unit.fields[field_name]`` so further downstream
            blocks see the right scalar unit. Default ``None``
            preserves the unit-less behaviour byte-for-byte.

            When ``field_name`` contains dots, the leaf unit cannot be
            propagated because :class:`BusUnit` is flat (one Unit per
            top-level field); the input bus is still tagged but the
            output unit is ``None``. Pass a single-segment
            ``field_name`` if you need leaf-unit propagation.

    Input ports:
        ``(0)`` — the bus signal (NamedTuple-shaped).

    Output ports:
        ``(0)`` — the value of ``bus.<field_name>``, optionally sliced
        at ``slice_idx``.
    """

    def __init__(
        self, field_name, *args, bus_unit=None, slice_idx=None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(field_name, str):
            raise ValueError(
                f"BusSelector field_name must be a str, got "
                f"{type(field_name).__name__}: {field_name!r}."
            )
        # T-117-followup-bus-dot-path: validate each segment of the
        # (possibly dotted) path individually; an all-empty / pure-dot
        # path is rejected up front.
        path_segments = field_name.split(".") if field_name else []
        if not path_segments:
            raise ValueError(
                "BusSelector field_name must be non-empty."
            )
        for seg in path_segments:
            if not seg.isidentifier():
                # Phrase the error so it matches the legacy
                # "not a valid Python identifier" wording for the
                # single-segment case, plus an extra hint for the
                # dot-path case.
                detail = (
                    f"BusSelector field_name {field_name!r} is not a "
                    f"valid Python identifier (NamedTuple field "
                    f"constraint)."
                )
                if len(path_segments) > 1:
                    detail = (
                        f"BusSelector field_name {field_name!r} "
                        f"contains segment {seg!r} that is not a valid "
                        f"Python identifier (each dot-separated segment "
                        f"must be a valid Python identifier — "
                        f"NamedTuple field constraint)."
                    )
                raise ValueError(detail)
        is_dotted = len(path_segments) > 1

        # T-117-followup-bus-array: ``slice_idx`` must be a plain
        # non-negative int when supplied (bools are technically ints so
        # we filter them out explicitly). Validating up front means an
        # obvious mistake fires at construction time rather than
        # halfway through a traced JAX path.
        if slice_idx is not None:
            if isinstance(slice_idx, bool) or not isinstance(slice_idx, int):
                raise TypeError(
                    f"BusSelector slice_idx must be an int, got "
                    f"{type(slice_idx).__name__}: {slice_idx!r}."
                )
            if slice_idx < 0:
                raise ValueError(
                    f"BusSelector slice_idx must be non-negative, "
                    f"got {slice_idx!r}."
                )

        # T-117-followup-bus-units: when a BusUnit is supplied, the
        # selector's output unit is the per-field unit from the bus.
        # We validate the field exists in the supplied schema so the
        # error fires here rather than at connect time.
        #
        # T-117-followup-bus-dot-path: dotted paths defeat the
        # flat-BusUnit lookup, so the leaf unit is silently dropped
        # (the input bus is still tagged for the connect-time check).
        output_unit = None
        if bus_unit is not None:
            from ..framework.units import BusUnit as _BusUnit

            if not isinstance(bus_unit, _BusUnit):
                raise TypeError(
                    f"BusSelector bus_unit must be a BusUnit instance, "
                    f"got {type(bus_unit).__name__}: {bus_unit!r}."
                )
            if not is_dotted:
                if field_name not in bus_unit.fields:
                    raise ValueError(
                        f"BusSelector field_name {field_name!r} is not "
                        f"present in bus_unit fields "
                        f"{sorted(bus_unit.fields)!r}."
                    )
                output_unit = bus_unit.fields[field_name]
            else:
                # Top-segment must still be present so the connect-time
                # check can verify the upstream bus has the right
                # outer shape. We don't recurse — BusUnit is flat — but
                # at least pin the outer field name.
                top = path_segments[0]
                if top not in bus_unit.fields:
                    raise ValueError(
                        f"BusSelector field_name {field_name!r}: top-"
                        f"level segment {top!r} is not present in "
                        f"bus_unit fields {sorted(bus_unit.fields)!r}."
                    )
                # output_unit stays None for nested paths.

        self._field_name = field_name
        self._bus_unit = bus_unit
        self._slice_idx = slice_idx
        self.declare_input_port(name="bus", units=bus_unit)

        # T-117-followup-bus-dot-path: ``operator.attrgetter`` resolves
        # dot-paths in a single call; for the single-segment case it
        # is exactly equivalent to ``getattr(bus, name)``. Bind it in
        # the closure so the traced computation has a fixed callable.
        # T-117-followup-bus-array: when ``slice_idx`` is supplied we
        # bind it inside the closure so the traced computation indexes
        # the field array at the static slot. ``attrgetter`` + integer
        # indexing are both transparent to ``jax.grad``/``jit``.
        import operator as _operator
        _getter = _operator.attrgetter(field_name)
        if slice_idx is None:

            def _compute_field(_time, _state, *inputs, **_params):
                (bus,) = inputs
                return _getter(bus)
        else:
            _idx = slice_idx

            def _compute_field(_time, _state, *inputs, **_params):
                (bus,) = inputs
                return _getter(bus)[_idx]

        # Use only the leaf segment as the output-port name to keep
        # port names valid Python identifiers / NamedTuple-friendly.
        leaf_name = path_segments[-1]
        self.declare_output_port(
            _compute_field,
            name=leaf_name,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
            units=output_unit,
        )

    @property
    def field_name(self) -> str:
        """The name of the bus field this block selects."""
        return self._field_name

    @property
    def bus_unit(self):
        """The :class:`BusUnit` describing the upstream bus, or
        ``None`` if no unit metadata was supplied."""
        return self._bus_unit

    @property
    def slice_idx(self):
        """The integer index into an array-valued bus field, or
        ``None`` if the selector returns the field value as-is
        (T-117-followup-bus-array)."""
        return self._slice_idx



# T-115-followup-deadzone-bilinear end-of-file marker.


# ---------------------------------------------------------------------------
# T-117-followup-nested-buses — nested-bus helpers (bus_fields,
# flatten_bus, unflatten_bus).
#
# The T-117-fu-bus-namedtuple :class:`BusCreator` produces a
# ``collections.namedtuple``-typed pytree. NamedTuples nest natively
# in JAX: a ``BusCreator`` whose inputs include other bus signals
# simply yields a NamedTuple whose values are themselves NamedTuples,
# and ``jax.tree_util`` walks the whole nested structure as a single
# pytree — so ``jit`` / ``vmap`` / ``grad`` already work through any
# depth of nesting without further plumbing.
#
# This section ships explicit verification of that fact plus three
# helpers used by generic block code that processes bus signals
# without hard-coding their schema:
#
#   * :func:`bus_fields` — yields ``(name, value)`` pairs over a bus
#     signal's NamedTuple fields in declaration order.
#   * :func:`flatten_bus` — concatenates all scalar/vector fields of a
#     bus into one flat 1-D array (recursive: nested buses are
#     flattened in order of declaration).
#   * :func:`unflatten_bus` — inverse: given a flat array and a
#     "spec" (the NamedTuple class, possibly with nested classes), it
#     reconstructs the original NamedTuple-typed bus.
#
# Lives at the BOTTOM of primitives.py in a clearly marked section so
# the diff is disjoint from T-114-fu-prelookup-extrap (Prelookup /
# InterpolationUsingPrelookup, mid-file) and the original
# T-117-fu-bus-namedtuple block (~line 8628).
# ---------------------------------------------------------------------------


def _is_bus_signal(value) -> bool:
    """Return True iff ``value`` is a BusCreator-style NamedTuple.

    The check is structural: NamedTuples are tuples with a ``_fields``
    attribute. We do *not* require the class to be the literal type
    produced by :class:`BusCreator`, so user-defined NamedTuples used
    as bus signals (e.g. for testing) also pass.
    """
    return isinstance(value, tuple) and hasattr(value, "_fields")


def bus_fields(bus_signal):
    """Yield ``(name, value)`` pairs over a bus signal's fields.

    The order matches the NamedTuple's declaration order — which, for
    a bus produced by :class:`BusCreator`, is the input-port order.

    Useful when a generic block needs to walk over a bus signal
    without hard-coding the schema: e.g. summing all fields, applying
    a per-field transform, or recursing into nested sub-buses.

    Parameters:
        bus_signal: A NamedTuple-typed value (typically produced by
            :class:`BusCreator`). Any object with a ``_fields`` tuple
            attribute is accepted.

    Returns:
        A list of ``(field_name, field_value)`` pairs in declaration
        order. Returns a list (not a generator) so callers can index
        or re-iterate without surprises.

    Raises:
        TypeError: If ``bus_signal`` is not a NamedTuple-shaped value.
    """
    if not _is_bus_signal(bus_signal):
        raise TypeError(
            f"bus_fields expects a NamedTuple-shaped bus signal "
            f"(must have a ``_fields`` attribute); got "
            f"{type(bus_signal).__name__}: {bus_signal!r}."
        )
    return [(name, getattr(bus_signal, name)) for name in bus_signal._fields]


def flatten_bus(bus_signal):
    """Concatenate all scalar/vector fields of a bus into one 1-D array.

    Recurses into nested bus signals (NamedTuple-typed values): the
    flat output is the in-order concatenation of every leaf, where
    "leaf" is anything that is not itself a NamedTuple-shaped value.

    The traversal order is the NamedTuple's declaration order at each
    level, which matches the input-port order for a :class:`BusCreator`.

    Parameters:
        bus_signal: A NamedTuple-typed value (possibly nested).

    Returns:
        A 1-D array whose dtype is whatever ``npa.concatenate`` yields
        for the leaf values. T-005 default-float64 is preserved when
        every leaf is float64 (the upstream default).

    Raises:
        TypeError: If ``bus_signal`` is not a NamedTuple-shaped value.

    See also:
        :func:`unflatten_bus` for the inverse operation. The pair
        round-trips byte-equivalently for any bus whose leaves are
        scalars or 1-D vectors.
    """
    if not _is_bus_signal(bus_signal):
        raise TypeError(
            f"flatten_bus expects a NamedTuple-shaped bus signal "
            f"(must have a ``_fields`` attribute); got "
            f"{type(bus_signal).__name__}: {bus_signal!r}."
        )
    parts = []
    for name in bus_signal._fields:
        v = getattr(bus_signal, name)
        if _is_bus_signal(v):
            parts.append(flatten_bus(v))
        else:
            # Use ``npa.atleast_1d`` so scalars contribute one element
            # each and 1-D vectors contribute their length. Higher-rank
            # tensors fall through ``reshape(-1)`` for a defined
            # behaviour; the typical bus carries scalars or short
            # vectors.
            parts.append(npa.atleast_1d(npa.asarray(v)).reshape(-1))
    return npa.concatenate(parts)


def unflatten_bus(flat_array, bus_spec, *, leaf_sizes=None):
    """Reconstruct a NamedTuple-typed bus from a flat 1-D array.

    Inverse of :func:`flatten_bus` for buses whose leaves are scalars
    or 1-D vectors. The ``bus_spec`` argument tells us the NamedTuple
    class (and, for nested buses, the nested NamedTuple classes) so we
    can rebuild the typed structure.

    For nested buses, ``bus_spec`` is expected to carry a
    ``_field_specs`` class-level mapping from each nested-bus field
    name to its NamedTuple class (recursive). Helper-built specs
    populate this automatically. Bare NamedTuple classes without
    ``_field_specs`` are treated as fully flat (every field is a
    scalar leaf).

    For non-scalar leaves, pass ``leaf_sizes`` — a nested tuple
    matching the bus shape with one integer per scalar/vector leaf
    (and a nested tuple at each nested-bus position). The total of
    all leaf sizes must equal ``flat_array.shape[0]``.

    Parameters:
        flat_array: 1-D array, length equal to the total number of
            scalar elements across all leaves.
        bus_spec: NamedTuple class describing the bus shape. May
            carry a ``_field_specs`` dict for nested buses.
        leaf_sizes: Optional nested tuple of ints — one per scalar /
            vector leaf, in declaration order. Required when leaves
            are non-scalar.

    Returns:
        A ``bus_spec``-typed NamedTuple matching the original bus.

    Raises:
        TypeError: If ``bus_spec`` is not a NamedTuple class.
        ValueError: If the supplied ``leaf_sizes`` does not match the
            bus shape, or if the flat array length disagrees with the
            sum of leaf sizes.
    """
    if not (
        isinstance(bus_spec, type)
        and issubclass(bus_spec, tuple)
        and hasattr(bus_spec, "_fields")
    ):
        raise TypeError(
            f"unflatten_bus expects ``bus_spec`` to be a NamedTuple "
            f"class; got {bus_spec!r}."
        )
    flat = npa.asarray(flat_array).reshape(-1)

    if leaf_sizes is None:
        # Default: every leaf is a scalar. For nested buses we discover
        # the structure from ``_field_specs`` (set when the caller built
        # the spec via this module's helpers); otherwise assume all
        # fields at the top level are scalar leaves.
        def _build_scalar(spec, cursor):
            vals = []
            for fname in spec._fields:
                sub_spec = getattr(spec, "_field_specs", {}).get(fname)
                if (
                    sub_spec is not None
                    and isinstance(sub_spec, type)
                    and issubclass(sub_spec, tuple)
                    and hasattr(sub_spec, "_fields")
                ):
                    sub_val, cursor = _build_scalar(sub_spec, cursor)
                    vals.append(sub_val)
                else:
                    vals.append(flat[cursor])
                    cursor += 1
            return spec(*vals), cursor

        out, cursor = _build_scalar(bus_spec, 0)
        if cursor != int(flat.shape[0]):
            raise ValueError(
                f"unflatten_bus: bus_spec has {cursor} scalar leaves "
                f"but flat_array has length {int(flat.shape[0])}."
            )
        return out

    # Explicit leaf-size path: walk spec + sizes in lockstep.
    def _build_sized(spec, sizes, cursor):
        if len(sizes) != len(spec._fields):
            raise ValueError(
                f"unflatten_bus: leaf_sizes shape mismatch — spec "
                f"{spec.__name__} has {len(spec._fields)} fields but "
                f"got {len(sizes)} size entries."
            )
        vals = []
        for fname, sz in zip(spec._fields, sizes):
            sub_spec = getattr(spec, "_field_specs", {}).get(fname)
            if (
                sub_spec is not None
                and isinstance(sub_spec, type)
                and issubclass(sub_spec, tuple)
                and hasattr(sub_spec, "_fields")
            ):
                if not isinstance(sz, tuple):
                    raise ValueError(
                        f"unflatten_bus: nested-bus field {fname!r} "
                        f"expects a nested tuple of sizes, got {sz!r}."
                    )
                sub_val, cursor = _build_sized(sub_spec, sz, cursor)
                vals.append(sub_val)
            else:
                n = int(sz)
                vals.append(flat[cursor:cursor + n])
                cursor += n
        return spec(*vals), cursor

    out, cursor = _build_sized(bus_spec, leaf_sizes, 0)
    if cursor != int(flat.shape[0]):
        raise ValueError(
            f"unflatten_bus: leaf_sizes sum to {cursor} but flat_array "
            f"has length {int(flat.shape[0])}."
        )
    return out



# T-114-followup-table-search end-of-file marker.


# ---------------------------------------------------------------------------
# T-117-followup-bus-merge — union-merge two NamedTuple-shaped bus signals.
#
# Bus signals from different subsystems often need to be combined into a
# single wider bus (e.g. one block produces a "vehicle_state" bus with
# {position, velocity} and another produces a "vehicle_cmd" bus with
# {throttle, brake}; downstream we want a single "vehicle" bus with all
# four fields). ``merge_buses`` (pure helper) and ``BusMerge`` (the
# LeafSystem wrapper) implement this union operation over the
# T-117-fu-bus-namedtuple representation.
#
# Design notes:
#   * The merged bus is a fresh NamedTuple class built at the point of
#     merge. Its field order is ``fields(a) ++ fields(b)`` for disjoint
#     inputs, mirroring the input-port order on ``BusMerge``.
#   * Collisions are user policy: ``on_collision`` selects between
#     raising (default — safest), preferring ``a``'s value, or preferring
#     ``b``'s value. The merged-bus schema (field order) is unchanged by
#     the collision policy; the policy only governs which leaf value
#     lands in the colliding slot.
#   * Both the helper and the block are differentiable: the underlying
#     op is NamedTuple construction over inputs, which is autodiff- and
#     jit-transparent (same as ``BusCreator``).
#   * Default-off byte-equivalence: this section only adds new public
#     symbols (``merge_buses``, ``BusMerge``); the existing T-117 code
#     paths are not touched.
#
# Sits at the bottom of primitives.py in a clearly marked section to
# keep the diff disjoint from concurrent work on Decimator (T-123-fu)
# and ``framework/units.py`` (T-104-fu-currency-units).
# ---------------------------------------------------------------------------


def _bus_field_names(bus_spec) -> tuple[str, ...]:
    """Normalize ``bus_spec`` to a tuple of field-name strings.

    Accepts three forms so the API composes cleanly with the rest of
    T-117:

      * A :class:`BusCreator` instance — uses its ``field_names`` property.
      * A NamedTuple *class* (has a ``_fields`` class attribute) — uses
        ``_fields`` directly.
      * A bare iterable of strings (tuple/list) — treated as the
        ordered field-name sequence.

    Returns:
        A tuple of strings, in declaration / port order.

    Raises:
        TypeError: If ``bus_spec`` does not match any of the supported
            forms.
        ValueError: If the resulting names are not all valid Python
            identifiers (NamedTuple constraint).
    """
    if isinstance(bus_spec, BusCreator):
        names = tuple(bus_spec.field_names)
    elif isinstance(bus_spec, type) and hasattr(bus_spec, "_fields"):
        names = tuple(bus_spec._fields)
    elif isinstance(bus_spec, (tuple, list)):
        names = tuple(bus_spec)
    else:
        raise TypeError(
            "merge_buses/BusMerge bus_spec must be a BusCreator instance, "
            "a NamedTuple class, or a tuple/list of field-name strings; "
            f"got {type(bus_spec).__name__}: {bus_spec!r}."
        )
    for fname in names:
        if not isinstance(fname, str) or not fname.isidentifier():
            raise ValueError(
                f"merge_buses/BusMerge field name {fname!r} is not a valid "
                "Python identifier (required for NamedTuple fields)."
            )
    return names


def _merged_field_order(
    fields_a: tuple[str, ...],
    fields_b: tuple[str, ...],
    on_collision: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compute the merged field order and the set of colliding fields.

    The merged order is ``fields_a`` followed by the fields of
    ``fields_b`` that are not already in ``fields_a``. This keeps the
    schema independent of the ``on_collision`` policy: only the *value*
    in a colliding slot depends on the policy, never the slot order.

    Returns:
        ``(merged_order, collisions)`` — ``merged_order`` is the tuple
        of field names in the merged bus, and ``collisions`` is the
        tuple of names that appeared in both inputs.

    Raises:
        ValueError: If ``on_collision == "error"`` and any names
            collide.
        ValueError: If ``on_collision`` itself is not one of the
            supported policies.
    """
    if on_collision not in ("error", "prefer_a", "prefer_b"):
        raise ValueError(
            "merge_buses/BusMerge on_collision must be one of "
            "('error', 'prefer_a', 'prefer_b'); got "
            f"{on_collision!r}."
        )
    set_a = set(fields_a)
    collisions = tuple(name for name in fields_b if name in set_a)
    if collisions and on_collision == "error":
        raise ValueError(
            "merge_buses/BusMerge: field name collision(s) between "
            f"bus_a {fields_a!r} and bus_b {fields_b!r}: "
            f"{collisions!r}. Pass on_collision='prefer_a' or "
            "'prefer_b' to resolve, or rename the colliding fields."
        )
    merged_order = fields_a + tuple(
        name for name in fields_b if name not in set_a
    )
    return merged_order, collisions


def merge_buses(bus_a, bus_b, *, on_collision: str = "error"):
    """Merge two NamedTuple-shaped bus signals by union of fields.

    The merged bus is a fresh NamedTuple (named ``"MergedBus"``) whose
    fields are ``bus_a._fields`` followed by the fields of
    ``bus_b._fields`` not already in ``bus_a`` (de-duplicated while
    preserving declaration order). The result is a JAX-pytree-friendly
    value identical in shape to what :class:`BusCreator` would produce
    for the merged schema.

    Differentiability: gradients flow from each merged-bus leaf back to
    whichever input bus contributed the leaf — the underlying op is
    NamedTuple construction over ``getattr`` lookups, both of which are
    transparent to ``jax.grad`` / ``jax.jit``.

    Parameters:
        bus_a: First bus signal. Must be a NamedTuple-shaped value
            (``isinstance(bus_a, tuple) and hasattr(bus_a, "_fields")``).
        bus_b: Second bus signal. Same contract as ``bus_a``.
        on_collision: Policy for fields that appear in both inputs.

            * ``"error"`` (default) — raise :class:`ValueError`.
            * ``"prefer_a"`` — keep the value from ``bus_a``.
            * ``"prefer_b"`` — keep the value from ``bus_b``.

            The merged-bus *schema* (field order) is independent of the
            policy: collisions only change which leaf value lands in
            the colliding slot.

    Returns:
        A NamedTuple instance whose fields are the union of
        ``bus_a._fields`` and ``bus_b._fields``.

    Raises:
        TypeError: If either input is not a NamedTuple-shaped value.
        ValueError: If ``on_collision`` is not one of the supported
            policies, or if collisions exist and ``on_collision ==
            "error"``.
    """
    if not _is_bus_signal(bus_a):
        raise TypeError(
            "merge_buses expects bus_a to be a NamedTuple-shaped bus "
            f"signal; got {type(bus_a).__name__}: {bus_a!r}."
        )
    if not _is_bus_signal(bus_b):
        raise TypeError(
            "merge_buses expects bus_b to be a NamedTuple-shaped bus "
            f"signal; got {type(bus_b).__name__}: {bus_b!r}."
        )

    fields_a = tuple(bus_a._fields)
    fields_b = tuple(bus_b._fields)
    merged_order, _ = _merged_field_order(fields_a, fields_b, on_collision)

    # Build the leaf values in merged-order. For each name, look in the
    # input dictated by the collision policy (or the unique input if no
    # collision). Using ``getattr`` keeps the path transparent to JAX.
    set_a = set(fields_a)
    set_b = set(fields_b)
    values = []
    for name in merged_order:
        in_a = name in set_a
        in_b = name in set_b
        if in_a and in_b:
            # Collision — policy dictates the source (error case is
            # already raised by _merged_field_order above).
            if on_collision == "prefer_a":
                values.append(getattr(bus_a, name))
            else:  # on_collision == "prefer_b"
                values.append(getattr(bus_b, name))
        elif in_a:
            values.append(getattr(bus_a, name))
        else:
            values.append(getattr(bus_b, name))

    MergedBus = namedtuple("MergedBus", merged_order)
    return MergedBus(*values)


class BusMerge(LeafSystem):
    """Merge two bus signals by union of fields (LeafSystem wrapper).

    Wraps :func:`merge_buses` as a block for use inside a Diagram. The
    two upstream bus signals are read from input ports 0 and 1; the
    merged bus is produced on output port 0.

    The merged-bus schema is fixed at construction time from
    ``bus_spec_a`` and ``bus_spec_b`` so the output port can be declared
    with a known NamedTuple type (and so the framework's pytree handling
    sees a consistent type across context-build and trace time, matching
    the T-117-fu-bus-namedtuple design for :class:`BusCreator`).

    Parameters:
        bus_spec_a: Schema of the first bus. Accepts a
            :class:`BusCreator` instance, a NamedTuple class, or a
            tuple/list of field-name strings.
        bus_spec_b: Schema of the second bus. Same forms as
            ``bus_spec_a``.
        on_collision: Policy for fields that appear in both schemas:

            * ``"error"`` (default) — raise :class:`ValueError` at
              construction time.
            * ``"prefer_a"`` — read the colliding leaf from input
              port 0.
            * ``"prefer_b"`` — read the colliding leaf from input
              port 1.

    Input ports:
        ``(0)`` — bus_a (NamedTuple-shaped).
        ``(1)`` — bus_b (NamedTuple-shaped).

    Output ports:
        ``(0)`` — the merged bus, a NamedTuple whose fields are the
        union of the two input schemas.
    """

    def __init__(
        self,
        bus_spec_a,
        bus_spec_b,
        *args,
        on_collision: str = "error",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        fields_a = _bus_field_names(bus_spec_a)
        fields_b = _bus_field_names(bus_spec_b)
        if len(fields_a) == 0:
            raise ValueError(
                "BusMerge bus_spec_a must have at least one field; "
                "got an empty schema."
            )
        if len(fields_b) == 0:
            raise ValueError(
                "BusMerge bus_spec_b must have at least one field; "
                "got an empty schema."
            )
        merged_order, collisions = _merged_field_order(
            fields_a, fields_b, on_collision
        )

        self._fields_a = fields_a
        self._fields_b = fields_b
        self._on_collision = on_collision
        self._collisions = collisions
        self._merged_field_names = merged_order
        self._bus_type = namedtuple("MergedBus", merged_order)

        # Declare the two bus input ports. We do not pass units here —
        # the merged-bus unit/schema surface is a deeper followup (the
        # T-117-fu-bus-units machinery is per-BusCreator). Default-off
        # parity: callers who do not opt into units see no behavioural
        # change.
        idx_a = self.declare_input_port(name="bus_a")
        idx_b = self.declare_input_port(name="bus_b")
        ticket_a = self.input_ports[idx_a].ticket
        ticket_b = self.input_ports[idx_b].ticket

        # Pre-compute the per-output-field source map so the runtime
        # closure stays tight: a tuple of ``(source_index, field_name)``
        # pairs where ``source_index`` is 0 for bus_a and 1 for bus_b.
        set_a = set(fields_a)
        set_b = set(fields_b)
        source_map = []
        for name in merged_order:
            in_a = name in set_a
            in_b = name in set_b
            if in_a and in_b:
                # Collision — policy dictates the source. ``error`` is
                # rejected above in ``_merged_field_order`` so only the
                # two prefer-* cases survive here.
                source_map.append((0 if on_collision == "prefer_a" else 1, name))
            elif in_a:
                source_map.append((0, name))
            else:
                source_map.append((1, name))
        source_map_local = tuple(source_map)
        bus_type_local = self._bus_type

        def _compute_merged(_time, _state, *inputs, **_params):
            # ``inputs`` is ``(bus_a, bus_b)`` in declaration order.
            bus_a_val, bus_b_val = inputs
            buses = (bus_a_val, bus_b_val)
            leaves = tuple(
                getattr(buses[src], name) for (src, name) in source_map_local
            )
            return bus_type_local(*leaves)

        # As with BusCreator: do NOT pass ``default_value=`` — the
        # framework would call ``npa.array`` on the NamedTuple and lose
        # the typed-bus shape. The lazy default-value path computes the
        # correct NamedTuple-typed default from upstream defaults.
        self.declare_output_port(
            _compute_merged,
            name="bus",
            prerequisites_of_calc=[ticket_a, ticket_b],
            requires_inputs=True,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """The tuple of merged field names, in output / declaration order."""
        return self._merged_field_names

    @property
    def bus_type(self) -> type:
        """The underlying NamedTuple class for the merged bus."""
        return self._bus_type

    @property
    def on_collision(self) -> str:
        """The collision-resolution policy in effect for this block."""
        return self._on_collision

    @property
    def collisions(self) -> tuple[str, ...]:
        """The tuple of field names that collided between the two
        input schemas. Empty unless ``on_collision`` is ``"prefer_a"``
        or ``"prefer_b"``."""
        return self._collisions


# T-117-followup-bus-merge end-of-file marker.


# ---------------------------------------------------------------------------
# T-117-fu-passthrough — BusPassthrough block (copy-bus identity).
#
# A trivial single-input / single-output LeafSystem whose sole job is to
# forward its input bus signal to its output port unchanged. Useful for:
#
#   * Rewiring diagrams: route a bus through a "junction" point so a
#     downstream consumer can be added/removed without touching the
#     producer's port.
#   * Debugging signal flow: insert a Passthrough at a suspect edge,
#     then attach a recorder to its output port to inspect the bus
#     value mid-diagram.
#   * Breaking long dependency chains for the scheduler: an explicit
#     LeafSystem boundary gives the topological sort an extra node it
#     can pivot around when laying out execution order.
#
# The implementation is intentionally minimal: a single ``declare_input_port``
# / ``declare_output_port`` pair, with the output computed by simply
# returning the upstream value. NamedTuple-shaped buses (the
# T-117-fu-bus-namedtuple representation) flow through without flattening
# because we never call ``npa.array`` on the value -- the framework's
# default-value computation lazily evaluates the closure on a dummy
# context, mirroring the BusCreator/BusMerge approach.
#
# Differentiability: identity has gradient = identity matrix, and
# autodiff sees this as a pass-through closure; gradients propagate
# back to the upstream input unchanged. T-005 default-float64 is
# preserved transitively via whatever upstream block sources the bus.
# ---------------------------------------------------------------------------


class BusPassthrough(LeafSystem):
    """Identity copy of a bus signal — single input port, single output port.

    The output is the input value, returned as-is. Pass-through semantics:
    NamedTuple-shaped buses (as produced by :class:`BusCreator` /
    :class:`BusMerge`) flow through unchanged, scalar/array signals
    likewise. This block exists to give Diagrams an explicit "junction"
    node for rewiring, debugging, or scheduler-boundary purposes; it
    has no parameters and does no computation beyond forwarding.

    Differentiable: ``jax.grad`` flows from the output back to the input
    leaf-by-leaf (identity has Jacobian = identity), and the block is
    JIT-traceable since the underlying op is a no-op closure return.

    Parameters:
        bus_unit: Optional :class:`BusUnit` describing the per-field
            units of the bus signal. When supplied, both input and
            output ports are tagged with this ``BusUnit`` so connect-
            time unit checks see a matched pair. ``None`` (the default)
            preserves the unit-less behaviour byte-for-byte.

    Input ports:
        ``(0)`` — the bus (or any) signal to forward.

    Output ports:
        ``(0)`` — the same value, returned as-is.
    """

    def __init__(self, *args, bus_unit=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Optional bus_unit propagation mirrors BusCreator/BusSelector:
        # when supplied, we tag both ports so the connect-time check
        # sees compatible BusUnit metadata on both sides. When None,
        # the default-off path is byte-equivalent to a no-op LeafSystem
        # with one input/one output and no unit metadata.
        if bus_unit is not None:
            from ..framework.units import BusUnit as _BusUnit

            if not isinstance(bus_unit, _BusUnit):
                raise TypeError(
                    f"BusPassthrough bus_unit must be a BusUnit instance, "
                    f"got {type(bus_unit).__name__}: {bus_unit!r}."
                )

        self._bus_unit = bus_unit
        self.declare_input_port(name="in", units=bus_unit)
        in_ticket = self.input_ports[0].ticket

        def _passthrough(_time, _state, *inputs, **_params):
            # ``inputs`` is a 1-tuple containing the upstream value;
            # we return it unchanged so NamedTuple-typed buses survive
            # without being flattened by ``npa.array``.
            (value,) = inputs
            return value

        # As with BusCreator / BusMerge: do NOT pass ``default_value=`` —
        # the leaf-system path would call ``npa.array`` on a NamedTuple-
        # shaped default and flatten it into a 1-D array, losing the
        # bus type. The lazy default-value computation re-runs the
        # closure on a dummy context and yields the correct typed value.
        self.declare_output_port(
            _passthrough,
            name="out",
            prerequisites_of_calc=[in_ticket],
            requires_inputs=True,
            units=bus_unit,
        )

    @property
    def bus_unit(self):
        """The :class:`BusUnit` propagated through this passthrough, or
        ``None`` if no unit metadata was supplied at construction."""
        return self._bus_unit


# T-117-followup-bus-passthrough end-of-file marker.


# ---------------------------------------------------------------------------
# T-117-fu-update — BusUpdate block (replace one field of a bus signal).
#
# Today buses are immutable NamedTuples (the T-117-fu-bus-namedtuple
# representation). To "edit" one field of a bus a user has to BusSelector
# every field, modify the one they care about, then BusCreator them all
# back together. That is verbose and error-prone -- especially for wide
# buses where most fields are passthroughs.
#
# ``BusUpdate(bus_spec, field_name)`` collapses that pattern to a single
# block: input port 0 is the upstream bus, input port 1 is the new value
# for ``field_name``; the output port is a fresh bus identical to the
# input except in the ``field_name`` slot, which now holds the new value.
#
# Design notes:
#   * The output schema is fixed at construction time from ``bus_spec``
#     (a BusCreator instance, NamedTuple class, or tuple/list of names),
#     mirroring BusMerge. Field order is preserved exactly.
#   * The runtime closure uses ``NamedTuple._replace(**{field: value})``
#     under the hood -- a single-line standard-library call that JAX
#     traces transparently because it is just NamedTuple construction
#     with a getattr loop.
#   * Differentiability: the op is NamedTuple construction over upstream
#     values, so ``jax.grad`` flows back to both input ports leaf-by-leaf
#     (gradient = 1 on the new_value path for the targeted field;
#     gradient = 1 on the bus_in path for every untouched field; zero on
#     the bus_in path for the targeted field, since its value is replaced).
#   * Default-off byte-equivalence: this section adds a single new public
#     class ``BusUpdate``; the existing T-117 code paths and other blocks
#     in this file are not touched.
#   * As with BusCreator/BusMerge/BusPassthrough, we deliberately do NOT
#     pass ``default_value=`` to ``declare_output_port`` -- the framework
#     would call ``npa.array`` on the NamedTuple and flatten it. Lazy
#     evaluation yields the correct NamedTuple-typed default.
# ---------------------------------------------------------------------------


class BusUpdate(LeafSystem):
    """Replace one field of a bus signal with a new value.

    Two-input / one-output LeafSystem. Input port 0 carries the upstream
    bus (a NamedTuple-shaped value, typically produced by
    :class:`BusCreator`). Input port 1 carries the new value for the
    field named ``field_name``. The output port is a fresh bus identical
    to the input except that the ``field_name`` slot is replaced by the
    new value. Field order is preserved exactly; all other fields are
    forwarded unchanged.

    Use this instead of the BusSelector-modify-BusCreator triplet when
    you only need to edit one field of a wide bus -- the block makes the
    intent explicit and avoids manually wiring N-1 passthrough edges.

    Differentiable: the underlying op is NamedTuple construction over
    ``getattr`` lookups on the input bus plus the ``new_value`` leaf, all
    of which are transparent to ``jax.grad`` and ``jax.jit`` (same as
    :class:`BusCreator` and :func:`merge_buses`).

    Parameters:
        bus_spec: Schema of the bus. Accepts a :class:`BusCreator`
            instance, a NamedTuple class, or a tuple/list of field-name
            strings (same forms as :class:`BusMerge`). Used both to
            validate ``field_name`` at construction time and to type the
            output port's NamedTuple class so the framework's pytree
            handling sees a consistent type across context-build and
            trace time.
        field_name: The name of the field to replace. Must be one of the
            names declared in ``bus_spec``; otherwise a clear
            :class:`ValueError` is raised at construction time.
        bus_unit: Optional :class:`BusUnit` describing the per-field
            units of the upstream bus (T-117-followup-bus-update-units-
            prop). When supplied, the block's ``bus_in`` and ``bus_out``
            ports advertise this ``BusUnit`` so the connect-time check
            verifies that the upstream :class:`BusCreator` produced a
            compatible bus and that downstream consumers see the right
            schema. The ``new_value`` input port is independently tagged
            with the per-field unit ``bus_unit.fields[field_name]`` so
            the connect-time check enforces unit compatibility on the
            replacement value. Composes with T-104 Phase 2 behaviour on
            ``Sum`` / ``Product`` / ``Integrator``. Default ``None``
            preserves the unit-less behaviour byte-for-byte.

    Input ports:
        ``(0)`` — ``bus_in``, the upstream bus signal (NamedTuple-shaped).
        ``(1)`` — ``new_value``, the value to put into the ``field_name``
        slot of the output bus.

    Output ports:
        ``(0)`` — ``bus_out``, a NamedTuple of the same shape as
        ``bus_in`` with ``field_name`` replaced by ``new_value``.
    """

    def __init__(
        self,
        bus_spec,
        field_name,
        *args,
        bus_unit=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Reuse the bus_spec normalisation helper from the BusMerge
        # section: accepts a BusCreator instance, a NamedTuple class, or
        # a tuple/list of strings, with valid-identifier validation.
        field_names = _bus_field_names(bus_spec)
        if len(field_names) == 0:
            raise ValueError(
                "BusUpdate bus_spec must have at least one field; "
                "got an empty schema."
            )
        if not isinstance(field_name, str):
            raise TypeError(
                "BusUpdate field_name must be a string; got "
                f"{type(field_name).__name__}: {field_name!r}."
            )
        if field_name not in field_names:
            raise ValueError(
                f"BusUpdate field_name {field_name!r} is not in the "
                f"bus schema {field_names!r}."
            )

        # T-117-followup-bus-update-units-prop: validate bus_unit
        # against the schema at construction time so an obvious mistake
        # fires here rather than at the connect-time check downstream.
        new_value_unit = None
        if bus_unit is not None:
            from ..framework.units import BusUnit as _BusUnit

            if not isinstance(bus_unit, _BusUnit):
                raise TypeError(
                    f"BusUpdate bus_unit must be a BusUnit instance, "
                    f"got {type(bus_unit).__name__}: {bus_unit!r}."
                )
            # The BusUnit's field set must match the bus schema exactly;
            # a missing or stray name signals a mismatched declaration.
            if set(bus_unit.fields.keys()) != set(field_names):
                raise ValueError(
                    f"BusUpdate bus_unit fields "
                    f"{sorted(bus_unit.fields.keys())!r} do not match "
                    f"the bus schema {sorted(field_names)!r}."
                )
            new_value_unit = bus_unit.fields[field_name]

        self._field_names = field_names
        self._field_name = field_name
        self._bus_unit = bus_unit
        # The output bus is the same shape as the input -- we name the
        # NamedTuple class "Bus" to match the T-117-fu-bus-namedtuple
        # convention used by BusCreator. The class identity differs from
        # the upstream BusCreator's class (one fresh class per BusUpdate
        # instance) but the field tuple matches exactly, so JAX's pytree
        # handling treats them as structurally equivalent.
        self._bus_type = namedtuple("Bus", field_names)

        # Declare input ports: bus_in then new_value, in that order.
        # T-117-followup-bus-update-units-prop: when ``bus_unit`` is
        # supplied, the ``bus_in`` and ``bus_out`` ports advertise the
        # full BusUnit (so downstream consumers see the schema and the
        # upstream connect-time check verifies compatibility), and the
        # ``new_value`` port advertises the per-field leaf unit.
        idx_bus = self.declare_input_port(name="bus_in", units=bus_unit)
        idx_new = self.declare_input_port(
            name="new_value", units=new_value_unit,
        )
        ticket_bus = self.input_ports[idx_bus].ticket
        ticket_new = self.input_ports[idx_new].ticket

        # Pre-compute the per-output-slot source map so the runtime
        # closure stays tight: tuple of ``(use_new_value, name)`` pairs
        # in declaration order. ``use_new_value`` is True for exactly the
        # ``field_name`` slot; False for every other field (which is
        # forwarded unchanged from ``bus_in``).
        source_map_local = tuple(
            (name == field_name, name) for name in field_names
        )
        bus_type_local = self._bus_type

        def _compute_updated(_time, _state, *inputs, **_params):
            # ``inputs`` is ``(bus_in, new_value)`` in declaration order.
            bus_in_val, new_value = inputs
            leaves = tuple(
                new_value if use_new else getattr(bus_in_val, name)
                for (use_new, name) in source_map_local
            )
            return bus_type_local(*leaves)

        # As with BusCreator / BusMerge / BusPassthrough: do NOT pass
        # ``default_value=`` -- the framework would call ``npa.array``
        # on the NamedTuple and flatten it. The lazy default-value
        # path computes the correct NamedTuple-typed default by running
        # ``_compute_updated`` on a dummy context.
        # T-117-followup-bus-update-units-prop: tag the output port with
        # the same BusUnit as the input bus so downstream blocks see
        # the schema preserved through the update.
        self.declare_output_port(
            _compute_updated,
            name="bus_out",
            prerequisites_of_calc=[ticket_bus, ticket_new],
            requires_inputs=True,
            units=bus_unit,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """The tuple of bus field names, in declaration / output order."""
        return self._field_names

    @property
    def field_name(self) -> str:
        """The name of the field this block replaces on each tick."""
        return self._field_name

    @property
    def bus_type(self) -> type:
        """The underlying NamedTuple class for the output bus."""
        return self._bus_type

    @property
    def bus_unit(self):
        """The :class:`BusUnit` propagated through this update, or
        ``None`` if no unit metadata was supplied at construction
        (T-117-followup-bus-update-units-prop)."""
        return self._bus_unit
