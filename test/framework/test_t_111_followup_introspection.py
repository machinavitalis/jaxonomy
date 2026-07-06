# SPDX-License-Identifier: MIT

"""T-111-followup-variant-introspection — discovery helpers.

Covers ``list_variants``, ``get_variant_choices``, and
``get_active_variant``: the trio of read-only walkers that let user
code (e.g. a CLI ``variants list`` helper) introspect the variant
structure of a built or partially-built diagram WITHOUT rebuilding it.

Tests verify:

    - Build a diagram with two named variants (``controller``,
      ``plant``); ``list_variants`` returns metadata for both.
    - ``get_variant_choices`` returns the choice names of a named
      variant.
    - ``get_active_variant`` returns the currently-selected choice,
      and tracks ``select_variant`` / ``with_config`` re-selections.
    - Default-off: a diagram with no variants → empty list.
    - Anonymous variants: ``list_variants`` still surfaces them (with
      ``name=None``) but the name-keyed helpers refuse / miss them.
    - Validation: ``get_variant_choices`` on an unknown name raises;
      ``get_active_variant`` on an unknown name returns ``None``.
"""

from __future__ import annotations

import pytest

import jaxonomy
from jaxonomy.framework import (
    Variant,
    VariantError,
    get_active_variant,
    get_variant_choices,
    list_variants,
    select_variant,
)
from jaxonomy.library import (
    Constant,
    Gain,
    Integrator,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures: two named variants ("controller" and "plant"), mirroring the
# with_config test file so the introspection helpers are exercised on the
# same shape of diagram users will inspect in practice.
# ---------------------------------------------------------------------------


def _build_p_controller() -> jaxonomy.Diagram:
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(2.0, name="P_gain"))
    builder.export_input(p.input_ports[0], name="u")
    builder.export_output(p.output_ports[0], name="y")
    return builder.build(name="p_controller")


def _build_pi_controller() -> jaxonomy.Diagram:
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(2.0, name="P_gain"))
    i_gain = builder.add(Gain(0.5, name="I_gain"))
    integ = builder.add(Integrator(0.0, name="I_state"))
    builder.connect(p.output_ports[0], i_gain.input_ports[0])
    builder.connect(i_gain.output_ports[0], integ.input_ports[0])
    builder.export_input(p.input_ports[0], name="u")
    builder.export_output(p.output_ports[0], name="y")
    return builder.build(name="pi_controller")


def _build_passthrough_plant() -> jaxonomy.Diagram:
    builder = jaxonomy.DiagramBuilder()
    g = builder.add(Gain(1.0, name="plant_gain"))
    builder.export_input(g.input_ports[0], name="u")
    builder.export_output(g.output_ports[0], name="y")
    return builder.build(name="passthrough_plant")


def _build_double_plant() -> jaxonomy.Diagram:
    builder = jaxonomy.DiagramBuilder()
    g = builder.add(Gain(2.0, name="plant_gain"))
    builder.export_input(g.input_ports[0], name="u")
    builder.export_output(g.output_ports[0], name="y")
    return builder.build(name="double_plant")


def _make_controller_variant() -> Variant:
    return Variant(
        choices={
            "p": _build_p_controller,
            "pi": _build_pi_controller,
        },
        default="p",
        name="controller",
    )


def _make_plant_variant() -> Variant:
    return Variant(
        choices={
            "lti": _build_passthrough_plant,
            "doubler": _build_double_plant,
        },
        default="lti",
        name="plant",
    )


def _build_root_diagram(
    controller_choice=None, plant_choice=None
) -> jaxonomy.Diagram:
    """Build a small root diagram containing two named variant points.

    ``controller_choice`` / ``plant_choice`` default to ``None`` (use
    each variant's own default), letting the same fixture build either
    a default-state diagram or one bound to a non-default choice up
    front (useful for cross-checking ``get_active_variant``).
    """
    controller = select_variant(
        _make_controller_variant(), name=controller_choice
    )
    plant = select_variant(_make_plant_variant(), name=plant_choice)
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(3.0, name="src"))
    builder.add(controller)
    builder.add(plant)
    builder.connect(src.output_ports[0], controller.input_ports[0])
    builder.connect(controller.output_ports[0], plant.input_ports[0])
    builder.export_output(plant.output_ports[0], name="y")
    return builder.build(name="root")


def _build_no_variants_diagram() -> jaxonomy.Diagram:
    """A plain diagram with no Variant points (for default-off coverage)."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(3.0, name="src"))
    g = builder.add(Gain(2.0, name="P_gain"))
    builder.connect(src.output_ports[0], g.input_ports[0])
    builder.export_output(g.output_ports[0], name="y")
    return builder.build(name="plain_root")


# ---------------------------------------------------------------------------
# list_variants
# ---------------------------------------------------------------------------


class TestListVariants:
    def test_lists_both_named_variants(self):
        diagram = _build_root_diagram()
        results = list_variants(diagram)
        names = {entry[0] for entry in results}
        assert names == {"controller", "plant"}
        assert len(results) == 2

    def test_entry_shape(self):
        """Each entry is (name, choice_names, active_choice)."""
        diagram = _build_root_diagram()
        results = list_variants(diagram)
        by_name = {entry[0]: entry for entry in results}

        ctrl_name, ctrl_choices, ctrl_active = by_name["controller"]
        assert ctrl_name == "controller"
        assert ctrl_choices == ("p", "pi")
        assert ctrl_active == "p"  # default

        plant_name, plant_choices, plant_active = by_name["plant"]
        assert plant_name == "plant"
        assert plant_choices == ("lti", "doubler")
        assert plant_active == "lti"  # default

    def test_no_variants_returns_empty(self):
        """Default-off: a diagram with no variants returns []."""
        diagram = _build_no_variants_diagram()
        assert list_variants(diagram) == []

    def test_non_diagram_returns_empty(self):
        """Passing a non-Diagram argument degrades gracefully."""
        assert list_variants(None) == []
        assert list_variants(42) == []
        assert list_variants("hello") == []

    def test_tracks_with_config_swaps(self):
        """After ``with_config(controller='pi')`` the active_choice for
        the controller variant must update accordingly."""
        diagram = _build_root_diagram()
        configured = diagram.with_config(controller="pi", plant="doubler")
        by_name = {entry[0]: entry for entry in list_variants(configured)}
        assert by_name["controller"][2] == "pi"
        assert by_name["plant"][2] == "doubler"
        # And the source diagram is untouched.
        by_name_src = {entry[0]: entry for entry in list_variants(diagram)}
        assert by_name_src["controller"][2] == "p"
        assert by_name_src["plant"][2] == "lti"

    def test_anonymous_variant_surfaced_with_none_name(self):
        """``list_variants`` includes anonymous variants but reports
        their name slot as ``None``."""
        anon = Variant(
            choices={"p": _build_p_controller, "pi": _build_pi_controller},
            default="p",
            # name= deliberately omitted
        )
        ctrl = select_variant(anon)
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(3.0, name="src"))
        builder.add(ctrl)
        builder.connect(src.output_ports[0], ctrl.input_ports[0])
        builder.export_output(ctrl.output_ports[0], name="y")
        diagram = builder.build(name="anon_root")

        results = list_variants(diagram)
        assert len(results) == 1
        name, choices, active = results[0]
        assert name is None
        assert choices == ("p", "pi")
        assert active == "p"


# ---------------------------------------------------------------------------
# get_variant_choices
# ---------------------------------------------------------------------------


class TestGetVariantChoices:
    def test_returns_choice_names(self):
        diagram = _build_root_diagram()
        assert get_variant_choices(diagram, "controller") == ("p", "pi")
        assert get_variant_choices(diagram, "plant") == ("lti", "doubler")

    def test_choices_unchanged_after_swap(self):
        """The choice list is a property of the Variant, not the active
        binding: swapping which choice is active doesn't change the
        choice menu."""
        diagram = _build_root_diagram()
        configured = diagram.with_config(controller="pi")
        assert get_variant_choices(configured, "controller") == ("p", "pi")

    def test_unknown_name_raises(self):
        diagram = _build_root_diagram()
        with pytest.raises(VariantError, match=r"no Variant with name"):
            get_variant_choices(diagram, "nonexistent")

    def test_unknown_name_lists_available(self):
        """Error message should help the user find the right name."""
        diagram = _build_root_diagram()
        with pytest.raises(VariantError, match=r"controller"):
            get_variant_choices(diagram, "nonexistent")

    def test_no_variants_raises_with_empty_available(self):
        diagram = _build_no_variants_diagram()
        with pytest.raises(VariantError, match=r"\[\]"):
            get_variant_choices(diagram, "controller")


# ---------------------------------------------------------------------------
# get_active_variant
# ---------------------------------------------------------------------------


class TestGetActiveVariant:
    def test_returns_default_active(self):
        diagram = _build_root_diagram()
        assert get_active_variant(diagram, "controller") == "p"
        assert get_active_variant(diagram, "plant") == "lti"

    def test_tracks_select_variant_choice(self):
        """Building the root with a non-default choice should be
        reflected in ``get_active_variant``."""
        diagram = _build_root_diagram(
            controller_choice="pi", plant_choice="doubler"
        )
        assert get_active_variant(diagram, "controller") == "pi"
        assert get_active_variant(diagram, "plant") == "doubler"

    def test_tracks_with_config_swap(self):
        diagram = _build_root_diagram()
        configured = diagram.with_config(controller="pi")
        assert get_active_variant(configured, "controller") == "pi"
        # plant was not overridden → still default
        assert get_active_variant(configured, "plant") == "lti"

    def test_unknown_name_returns_none(self):
        """Soft-miss semantics for the unknown-name path."""
        diagram = _build_root_diagram()
        assert get_active_variant(diagram, "nonexistent") is None

    def test_no_variants_returns_none(self):
        diagram = _build_no_variants_diagram()
        assert get_active_variant(diagram, "controller") is None
