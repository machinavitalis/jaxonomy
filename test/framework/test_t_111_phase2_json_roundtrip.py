# SPDX-License-Identifier: MIT

"""T-111 phase 2 — variant configuration JSON round-trip.

Adds and exercises:

* ``dump_variant_config(diagram) -> dict[str, str]`` — captures the
  active choice for every *named* variant.
* ``load_variant_config(diagram, config) -> Diagram`` — re-binds variant
  choices on a freshly-built diagram.
* ``dump_variant_config_to_json`` / ``load_variant_config_from_json`` —
  JSON-string convenience wrappers.

The Variant choices themselves (zero-arg builder callables) are not
serialised — only the *binding* of variant name to active choice name
is, which is exactly what model JSON / CI reproducibility manifests
need.
"""

from __future__ import annotations

import json

import pytest

import jaxonomy
from jaxonomy.framework import (
    Diagram,
    Variant,
    VariantError,
    apply_variant_config,
    select_variant,
)
from jaxonomy.framework.variants import (
    apply_variant_config_from_dict,
    dump_variant_config,
    dump_variant_config_to_json,
    load_variant_config,
    load_variant_config_from_json,
)
from jaxonomy.library import Constant, Gain, Integrator


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures — same shape as test_t_111_followup_with_config.py.
# ---------------------------------------------------------------------------


def _build_p_controller() -> jaxonomy.Diagram:
    b = jaxonomy.DiagramBuilder()
    p = b.add(Gain(2.0, name="P_gain"))
    b.export_input(p.input_ports[0], name="u")
    b.export_output(p.output_ports[0], name="y")
    return b.build(name="p_controller")


def _build_pi_controller() -> jaxonomy.Diagram:
    b = jaxonomy.DiagramBuilder()
    p = b.add(Gain(2.0, name="P_gain"))
    i_gain = b.add(Gain(0.5, name="I_gain"))
    integ = b.add(Integrator(0.0, name="I_state"))
    b.connect(p.output_ports[0], i_gain.input_ports[0])
    b.connect(i_gain.output_ports[0], integ.input_ports[0])
    b.export_input(p.input_ports[0], name="u")
    b.export_output(p.output_ports[0], name="y")
    return b.build(name="pi_controller")


def _build_passthrough_plant() -> jaxonomy.Diagram:
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(1.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="passthrough_plant")


def _build_double_plant() -> jaxonomy.Diagram:
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="double_plant")


def _make_controller_variant() -> Variant:
    return Variant(
        choices={"p": _build_p_controller, "pi": _build_pi_controller},
        default="p",
        name="controller",
    )


def _make_plant_variant() -> Variant:
    return Variant(
        choices={"lti": _build_passthrough_plant, "doubler": _build_double_plant},
        default="lti",
        name="plant",
    )


def _build_root_diagram() -> jaxonomy.Diagram:
    controller = select_variant(_make_controller_variant())
    plant = select_variant(_make_plant_variant())
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(3.0, name="src"))
    b.add(controller)
    b.add(plant)
    b.connect(src.output_ports[0], controller.input_ports[0])
    b.connect(controller.output_ports[0], plant.input_ports[0])
    b.export_output(plant.output_ports[0], name="y")
    return b.build(name="root")


def _eval_output(diagram) -> float:
    ctx = diagram.create_context()
    return float(diagram.output_ports[0].eval(ctx))


# ---------------------------------------------------------------------------
# dump_variant_config — captures the active binding.
# ---------------------------------------------------------------------------


def test_dump_captures_active_choice_for_every_named_variant():
    diagram = _build_root_diagram()
    config = dump_variant_config(diagram)
    assert config == {"controller": "p", "plant": "lti"}


def test_dump_reflects_apply_variant_config_changes():
    diagram = _build_root_diagram()
    swapped = apply_variant_config(diagram, controller="pi", plant="doubler")
    assert dump_variant_config(swapped) == {
        "controller": "pi",
        "plant": "doubler",
    }


def test_dump_of_diagram_with_no_named_variants_is_empty():
    """A vanilla diagram (no select_variant calls) yields {}."""
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(1.0, name="g"))
    b.export_output(g.output_ports[0], name="y")
    diagram = b.build(name="vanilla")
    assert dump_variant_config(diagram) == {}


def test_dump_skips_anonymous_variants():
    """Variant built without name= is skipped (cannot be addressed by
    name across builds)."""
    anonymous = Variant(
        choices={"a": _build_p_controller, "b": _build_pi_controller},
        default="a",
        # name=None — intentionally anonymous
    )
    named = Variant(
        choices={"lti": _build_passthrough_plant, "doubler": _build_double_plant},
        default="lti",
        name="plant",
    )
    anon_subsys = select_variant(anonymous)
    named_subsys = select_variant(named)
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(3.0, name="src"))
    b.add(anon_subsys)
    b.add(named_subsys)
    b.connect(src.output_ports[0], anon_subsys.input_ports[0])
    b.connect(anon_subsys.output_ports[0], named_subsys.input_ports[0])
    b.export_output(named_subsys.output_ports[0], name="y")
    diagram = b.build(name="mixed")

    config = dump_variant_config(diagram)
    # Only the named variant survives.
    assert config == {"plant": "lti"}


# ---------------------------------------------------------------------------
# load_variant_config — applies a dump back to a freshly-built diagram.
# ---------------------------------------------------------------------------


def test_dump_then_load_round_trips_identically():
    """Build, swap, dump, then re-load on a freshly-built diagram —
    the resulting active choices match the dumped config."""
    src = _build_root_diagram()
    swapped = apply_variant_config(src, controller="pi", plant="doubler")
    config = dump_variant_config(swapped)

    fresh = _build_root_diagram()  # back to defaults
    restored = load_variant_config(fresh, config)
    assert dump_variant_config(restored) == config


def test_load_empty_config_returns_diagram_unchanged():
    diagram = _build_root_diagram()
    out = load_variant_config(diagram, {})
    assert out is diagram  # no allocation, no swap


def test_load_unknown_variant_name_raises_variant_error():
    diagram = _build_root_diagram()
    with pytest.raises(VariantError, match="no Variant with name"):
        load_variant_config(diagram, {"nonexistent": "p"})


def test_load_unknown_choice_raises_variant_error():
    diagram = _build_root_diagram()
    with pytest.raises(VariantError):
        load_variant_config(diagram, {"controller": "not_a_choice"})


def test_apply_variant_config_from_dict_alias_matches_load():
    diagram = _build_root_diagram()
    config = {"controller": "pi"}
    a = load_variant_config(diagram, config)
    b = apply_variant_config_from_dict(diagram, config)
    assert dump_variant_config(a) == dump_variant_config(b)


# ---------------------------------------------------------------------------
# JSON string wrappers.
# ---------------------------------------------------------------------------


def test_dump_variant_config_to_json_is_valid_json_object():
    diagram = _build_root_diagram()
    s = dump_variant_config_to_json(diagram)
    parsed = json.loads(s)
    assert parsed == {"controller": "p", "plant": "lti"}


def test_dump_variant_config_to_json_pretty_by_default():
    """Default indent=2 produces multi-line, diff-friendly JSON."""
    diagram = _build_root_diagram()
    s = dump_variant_config_to_json(diagram)
    assert "\n" in s


def test_dump_variant_config_to_json_compact_when_indent_none():
    diagram = _build_root_diagram()
    s = dump_variant_config_to_json(diagram, indent=None)
    assert "\n" not in s


def test_load_variant_config_from_json_round_trips_swapped_state():
    src = _build_root_diagram()
    swapped = apply_variant_config(src, controller="pi", plant="doubler")
    json_str = dump_variant_config_to_json(swapped)

    fresh = _build_root_diagram()
    restored = load_variant_config_from_json(fresh, json_str)
    assert dump_variant_config(restored) == dump_variant_config(swapped)


def test_load_variant_config_from_json_rejects_non_object():
    """A JSON array, string, or number at the top level isn't a valid
    config — fail loud rather than silently mis-applying."""
    diagram = _build_root_diagram()
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_variant_config_from_json(diagram, "[1, 2, 3]")


# ---------------------------------------------------------------------------
# End-to-end reproducibility: dump → swap others → reload → simulate.
# ---------------------------------------------------------------------------


def test_round_trip_reproduces_simulation_output():
    """Persist the controller=pi/plant=doubler config, re-apply it
    against a fresh build, and confirm the simulation output matches
    the originally-swapped diagram."""
    src = _build_root_diagram()
    swapped = apply_variant_config(src, controller="pi", plant="doubler")
    expected = _eval_output(swapped)

    json_str = dump_variant_config_to_json(swapped)
    fresh = _build_root_diagram()
    restored = load_variant_config_from_json(fresh, json_str)
    got = _eval_output(restored)

    assert got == pytest.approx(expected)
