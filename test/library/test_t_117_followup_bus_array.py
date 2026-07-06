# SPDX-License-Identifier: MIT
"""T-117-followup-bus-array: variable-length array fields in buses.

The T-117-followup-bus-namedtuple :class:`BusCreator` packs one scalar
per input port. For some use cases users want a "list" semantically —
e.g. a ``"sensors"`` bus that bundles 8 thermocouple readings alongside
a separate scalar ``"control"`` field. This followup extends:

  * :class:`BusCreator` accepts an optional ``field_shapes`` mapping;
    listed fields are array-valued at the declared JAX-style shape,
    fields not listed remain scalar.
  * :class:`BusSelector` accepts an optional ``slice_idx`` integer;
    when supplied, the selector returns ``bus.<field_name>[slice_idx]``
    rather than the full array, so callers can pull a single element of
    an array-valued field without a separate ``Demux``.

When both kwargs are omitted, behaviour is byte-equivalent to
T-117-fu-bus-namedtuple — re-verified by ``test_t_117_followup_bus.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import BusCreator, BusSelector
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Construction-time validation for ``field_shapes`` / ``slice_idx``.
# These paths are pure Python so they do not need a Diagram.
# ---------------------------------------------------------------------------


def test_bus_creator_field_shapes_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown keys"):
        BusCreator(("a", "b"), field_shapes={"a": (4,), "c": (2,)})


def test_bus_creator_field_shapes_bad_dim_type_raises():
    with pytest.raises(ValueError, match="non-negative ints"):
        BusCreator(("sensors",), field_shapes={"sensors": ("4",)})


def test_bus_creator_field_shapes_negative_dim_raises():
    with pytest.raises(ValueError, match="non-negative ints"):
        BusCreator(("sensors",), field_shapes={"sensors": (-1,)})


def test_bus_creator_field_shapes_property_defaults_to_scalar():
    """Fields not listed in ``field_shapes`` default to scalar ``()``."""
    creator = BusCreator(("a", "b"), field_shapes={"a": (4,)})
    assert creator.field_shapes == {"a": (4,), "b": ()}


def test_bus_creator_field_shapes_none_property_all_scalar():
    """The default ``field_shapes=None`` yields all-scalar shapes."""
    creator = BusCreator(("x", "y", "z"))
    assert creator.field_shapes == {"x": (), "y": (), "z": ()}


def test_bus_selector_slice_idx_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        BusSelector("sensors", slice_idx=-1)


def test_bus_selector_slice_idx_bad_type_raises():
    with pytest.raises(TypeError, match="must be an int"):
        BusSelector("sensors", slice_idx=1.0)


def test_bus_selector_slice_idx_bool_raises():
    """Bool is an int subclass; we reject it explicitly."""
    with pytest.raises(TypeError, match="must be an int"):
        BusSelector("sensors", slice_idx=True)


def test_bus_selector_slice_idx_property_defaults_to_none():
    """Default ``slice_idx=None`` is exposed via the property."""
    sel = BusSelector("velocity")
    assert sel.slice_idx is None


def test_bus_selector_slice_idx_property_round_trips():
    sel = BusSelector("sensors", slice_idx=3)
    assert sel.slice_idx == 3
    assert sel.field_name == "sensors"


# ---------------------------------------------------------------------------
# End-to-end ``simulate`` tests. These pin that array-valued fields flow
# through the simulator and that ``slice_idx`` extracts the right element.
# ---------------------------------------------------------------------------


def _build_array_bus_diagram(field_names, values, field_shapes, select, *, slice_idx=None):
    """Construct one Constant per field, BusCreator with field_shapes,
    BusSelector with optional slice_idx, and return ``(diagram, selector)``.
    """
    sources = [library.Constant(np.asarray(v, dtype=np.float64)) for v in values]
    creator = BusCreator(field_names, field_shapes=field_shapes, name="creator")
    selector = BusSelector(select, slice_idx=slice_idx, name="selector")

    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, creator, selector)
    for src, port in zip(sources, creator.input_ports):
        builder.connect(src.output_ports[0], port)
    builder.connect(creator.output_ports[0], selector.input_ports[0])
    diagram = builder.build()
    return diagram, selector


def test_bus_array_field_round_trip_full_array():
    """BusSelector with no slice_idx returns the full sensors array."""
    sensors_value = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    scalar_value = 7.0
    diagram, selector = _build_array_bus_diagram(
        ("sensors", "scalar"),
        (sensors_value, scalar_value),
        field_shapes={"sensors": (4,), "scalar": ()},
        select="sensors",
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = np.asarray(results.outputs["out"])
    # Final row should equal the constant sensors array.
    np.testing.assert_allclose(out[-1], sensors_value)


def test_bus_array_field_round_trip_slice_idx():
    """BusSelector(slice_idx=2) extracts element 2 of the sensors array."""
    sensors_value = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    scalar_value = 7.0
    diagram, selector = _build_array_bus_diagram(
        ("sensors", "scalar"),
        (sensors_value, scalar_value),
        field_shapes={"sensors": (4,), "scalar": ()},
        select="sensors",
        slice_idx=2,
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, sensors_value[2])


def test_bus_array_scalar_field_unaffected_by_field_shapes():
    """A scalar field declared in field_shapes still round-trips cleanly."""
    sensors_value = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    scalar_value = 7.5
    diagram, selector = _build_array_bus_diagram(
        ("sensors", "scalar"),
        (sensors_value, scalar_value),
        field_shapes={"sensors": (4,), "scalar": ()},
        select="scalar",
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, scalar_value)


# ---------------------------------------------------------------------------
# JAX trace-level tests: confirm array-valued fields differentiate through
# both the full-array selector and the slice_idx selector. We exercise the
# underlying ops directly (BusCreator's NamedTuple + getattr / getitem) so
# the test isolates "array fields flow through JAX" from any simulator
# machinery.
# ---------------------------------------------------------------------------


def test_bus_array_field_differentiates_through_full_array():
    """Gradient flows through every element of an array-valued bus field."""
    creator = BusCreator(
        ("sensors", "scalar"), field_shapes={"sensors": (4,), "scalar": ()}
    )
    Bus = creator.bus_type

    def f(sensors, scalar):
        bus = Bus(sensors, scalar)
        # Pull the full array out, weight by [1,2,3,4], add the scalar.
        v = getattr(bus, "sensors")
        return jnp.sum(v * jnp.asarray([1.0, 2.0, 3.0, 4.0])) + getattr(bus, "scalar")

    sensors0 = jnp.asarray([0.0, 0.0, 0.0, 0.0])
    g_sensors, g_scalar = jax.grad(f, argnums=(0, 1))(sensors0, 0.0)
    np.testing.assert_allclose(np.asarray(g_sensors), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(float(g_scalar), 1.0)


def test_bus_array_field_differentiates_through_slice_idx():
    """Gradient flows through a single sliced element of an array field."""
    creator = BusCreator(("sensors",), field_shapes={"sensors": (4,)})
    Bus = creator.bus_type

    def f(sensors):
        bus = Bus(sensors)
        # Slice idx=2 — only element 2 contributes to the gradient.
        return getattr(bus, "sensors")[2] * 5.0

    sensors0 = jnp.asarray([0.0, 0.0, 0.0, 0.0])
    g = jax.grad(f)(sensors0)
    np.testing.assert_allclose(np.asarray(g), [0.0, 0.0, 5.0, 0.0])


def test_bus_array_field_traces_through_jit():
    """A bus with an array-valued field survives jax.jit."""
    creator = BusCreator(
        ("sensors", "scalar"), field_shapes={"sensors": (4,), "scalar": ()}
    )
    Bus = creator.bus_type

    @jax.jit
    def make_and_slice(sensors, scalar, idx):
        bus = Bus(sensors, scalar)
        return getattr(bus, "sensors")[idx]

    sensors = jnp.asarray([10.0, 11.0, 12.0, 13.0])
    out = make_and_slice(sensors, jnp.asarray(7.0), 2)
    np.testing.assert_allclose(float(out), 12.0)


# ---------------------------------------------------------------------------
# Default-off byte-equivalence: when ``field_shapes`` / ``slice_idx`` are
# both omitted, the runtime behaviour matches T-117-fu-bus-namedtuple
# exactly. We pin the property values + a round-trip selection here, and
# rely on ``test_t_117_followup_bus.py`` continuing to pass for the rest.
# ---------------------------------------------------------------------------


def test_bus_array_default_off_byte_equivalent_scalar_round_trip():
    """Default-off path: all-scalar fields, no slice, identical to T-117-fu."""
    sources = [library.Constant(1.0), library.Constant(2.0), library.Constant(3.0)]
    creator = BusCreator(("a", "b", "c"))
    selector = BusSelector("b")

    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, creator, selector)
    for src, port in zip(sources, creator.input_ports):
        builder.connect(src.output_ports[0], port)
    builder.connect(creator.output_ports[0], selector.input_ports[0])
    diagram = builder.build()

    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, 2.0)
    # Property values reflect the default-off path.
    assert creator.field_shapes == {"a": (), "b": (), "c": ()}
    assert selector.slice_idx is None
