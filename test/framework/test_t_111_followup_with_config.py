# SPDX-License-Identifier: MIT

"""T-111-followup-with-config — post-build configurator.

Covers ``apply_variant_config`` (and its alias ``Diagram.with_config``):
the post-build configurator that lets the user build a
diagram once with variants in their default state, then bind a new
variant configuration without rebuilding from scratch.

Tests verify:

    - Build a diagram with two named variants (``controller``, ``plant``)
      and reconfigure both via ``with_config(controller=..., plant=...)``.
      The resulting diagram simulates correctly.
    - Different config → structurally different diagram (different
      variant choices produce different node lists / different outputs).
    - Re-configurable: the SAME source diagram can be reconfigured
      multiple times (each call returns an independent diagram).
    - Default-off: ``with_config()`` with no overrides returns a
      structurally-equivalent copy of the source diagram (no swaps).
    - Diagram.with_config and apply_variant_config are equivalent
      (the method is a thin wrapper around the free function).
    - Validation: unknown variant name and unknown choice both raise
      ``VariantError`` with helpful messages.
    - Anonymous Variants (built with no ``name=`` kwarg) are correctly
      ignored by the configurator (they cannot be addressed by name).
"""

from __future__ import annotations

import pytest

import jax.numpy as jnp

import jaxonomy
from jaxonomy.framework import (
    Diagram,
    Variant,
    VariantError,
    apply_variant_config,
    select_variant,
)
from jaxonomy.library import (
    Constant,
    Gain,
    Integrator,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures: two named variants ("controller" and "plant").
# ---------------------------------------------------------------------------


def _build_p_controller() -> jaxonomy.Diagram:
    """Trivial proportional controller: y = 2.0 * u."""
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(2.0, name="P_gain"))
    builder.export_input(p.input_ports[0], name="u")
    builder.export_output(p.output_ports[0], name="y")
    return builder.build(name="p_controller")


def _build_pi_controller() -> jaxonomy.Diagram:
    """A "PI"-style controller: structurally distinct from P (more nodes).

    Exposes a single input port (matching the P controller's port count
    so the same parent wiring works for both choices). The integrator
    branch is included to make the diagram structurally distinct from
    the P-only controller (different node count, different leaf systems);
    its dynamics are immaterial for the with_config tests.
    """
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(2.0, name="P_gain"))
    i_gain = builder.add(Gain(0.5, name="I_gain"))
    integ = builder.add(Integrator(0.0, name="I_state"))
    # Wire the integrator path off the SAME exported input as the
    # P branch so a single parent connection drives both.
    builder.connect(p.output_ports[0], i_gain.input_ports[0])
    builder.connect(i_gain.output_ports[0], integ.input_ports[0])
    builder.export_input(p.input_ports[0], name="u")
    builder.export_output(p.output_ports[0], name="y")
    return builder.build(name="pi_controller")


def _build_passthrough_plant() -> jaxonomy.Diagram:
    """Trivial passthrough plant: y = 1.0 * u."""
    builder = jaxonomy.DiagramBuilder()
    g = builder.add(Gain(1.0, name="plant_gain"))
    builder.export_input(g.input_ports[0], name="u")
    builder.export_output(g.output_ports[0], name="y")
    return builder.build(name="passthrough_plant")


def _build_double_plant() -> jaxonomy.Diagram:
    """Doubler plant: y = 2.0 * u."""
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


def _build_root_diagram() -> jaxonomy.Diagram:
    """Build a small root diagram containing two named variant points.

    Layout: a constant source feeds the controller; the controller's
    output feeds the plant; the plant's output is exported. Both
    variants start in their default state.
    """
    controller = select_variant(_make_controller_variant())  # default "p"
    plant = select_variant(_make_plant_variant())  # default "lti"
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(3.0, name="src"))
    builder.add(controller)
    builder.add(plant)
    builder.connect(src.output_ports[0], controller.input_ports[0])
    builder.connect(controller.output_ports[0], plant.input_ports[0])
    builder.export_output(plant.output_ports[0], name="y")
    return builder.build(name="root")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _node_names(diagram: Diagram) -> set[str]:
    """Return the set of all leaf node names in the (possibly nested) diagram."""
    out: set[str] = set()

    def _walk(d):
        for child in d.nodes:
            out.add(child.name)
            if isinstance(child, Diagram):
                _walk(child)

    _walk(diagram)
    return out


def _eval_root_output(diagram: Diagram) -> float:
    """Create a context, simulate briefly, and return the diagram's
    exported output port's value."""
    ctx = diagram.create_context()
    return float(diagram.output_ports[0].eval(ctx))


# ---------------------------------------------------------------------------
# Reconfiguration.
# ---------------------------------------------------------------------------


class TestReconfiguration:
    def test_with_config_returns_working_diagram(self):
        """A reconfigured diagram should simulate without error and
        produce an output consistent with the chosen branches."""
        diagram = _build_root_diagram()
        configured = diagram.with_config(controller="p", plant="lti")
        # P controller: y = 2*u with u=3 → 6. LTI plant passes through → 6.
        assert _eval_root_output(configured) == pytest.approx(6.0)

    def test_different_config_different_output(self):
        """Different variant choices must produce different diagrams."""
        diagram = _build_root_diagram()
        # P + LTI: 2 * 3 * 1 = 6.0
        a = diagram.with_config(controller="p", plant="lti")
        # P + doubler: 2 * 3 * 2 = 12.0
        b = diagram.with_config(controller="p", plant="doubler")
        assert _eval_root_output(a) == pytest.approx(6.0)
        assert _eval_root_output(b) == pytest.approx(12.0)
        assert _eval_root_output(a) != _eval_root_output(b)

    def test_different_config_different_structure(self):
        """Switching to PI adds extra blocks (I_gain, I_state) that the
        P-only diagram does not contain."""
        diagram = _build_root_diagram()
        with_p = diagram.with_config(controller="p")
        with_pi = diagram.with_config(controller="pi")
        names_p = _node_names(with_p)
        names_pi = _node_names(with_pi)
        # PI-only blocks must appear only in the "pi" config.
        assert "I_gain" in names_pi
        assert "I_state" in names_pi
        assert "I_gain" not in names_p
        assert "I_state" not in names_p
        # P_gain is shared by both controllers' fixtures.
        assert "P_gain" in names_p
        assert "P_gain" in names_pi

    def test_repeated_reconfiguration_is_independent(self):
        """The same source diagram can be reconfigured multiple times;
        each call returns an independent diagram and the source is unchanged."""
        diagram = _build_root_diagram()
        a = diagram.with_config(controller="p", plant="lti")
        b = diagram.with_config(controller="pi", plant="doubler")
        c = diagram.with_config(controller="p", plant="doubler")
        # Independent objects.
        assert a is not b
        assert b is not c
        assert a is not diagram
        # Independent outputs.
        out_a = _eval_root_output(a)         # 2 * 3 * 1 = 6.0
        out_c = _eval_root_output(c)         # 2 * 3 * 2 = 12.0
        assert out_a == pytest.approx(6.0)
        assert out_c == pytest.approx(12.0)
        # Source diagram is not mutated.
        assert _eval_root_output(diagram) == pytest.approx(6.0)

    def test_reconfigure_a_reconfigured_diagram(self):
        """A diagram returned by ``with_config`` should itself be
        reconfigurable (the metadata tag survives the swap)."""
        diagram = _build_root_diagram()
        first = diagram.with_config(controller="p", plant="doubler")
        # Now flip the controller again on the already-configured diagram.
        second = first.with_config(controller="pi")
        names_pi = _node_names(second)
        assert "I_state" in names_pi  # PI branch is now active
        # The plant should still be the doubler (we didn't override it).
        assert "plant_gain" in _node_names(second)


# ---------------------------------------------------------------------------
# Default-off: with_config() with no overrides.
# ---------------------------------------------------------------------------


class TestDefaultOff:
    def test_no_overrides_returns_equivalent_diagram(self):
        """Calling ``.with_config()`` with no kwargs should produce a
        structurally-identical copy (same node names, same output)."""
        diagram = _build_root_diagram()
        copy = diagram.with_config()
        assert copy is not diagram
        assert _node_names(copy) == _node_names(diagram)
        assert _eval_root_output(copy) == pytest.approx(_eval_root_output(diagram))

    def test_apply_variant_config_with_no_overrides(self):
        """Same as above, but via the free-function API."""
        diagram = _build_root_diagram()
        copy = apply_variant_config(diagram)
        assert copy is not diagram
        assert _node_names(copy) == _node_names(diagram)
        assert _eval_root_output(copy) == pytest.approx(_eval_root_output(diagram))


# ---------------------------------------------------------------------------
# Equivalence: free function vs method.
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_with_config_method_matches_free_function(self):
        """``Diagram.with_config(...)`` must be a no-op wrapper around
        ``apply_variant_config``: same overrides → same output."""
        diagram = _build_root_diagram()
        via_method = diagram.with_config(controller="pi", plant="doubler")
        via_func = apply_variant_config(diagram, controller="pi", plant="doubler")
        assert _eval_root_output(via_method) == pytest.approx(
            _eval_root_output(via_func)
        )
        assert _node_names(via_method) == _node_names(via_func)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_variant_name_raises(self):
        diagram = _build_root_diagram()
        with pytest.raises(VariantError, match=r"no Variant with name"):
            diagram.with_config(nonexistent="whatever")

    def test_unknown_choice_raises(self):
        diagram = _build_root_diagram()
        with pytest.raises(VariantError, match=r"unknown choice 'mpc'"):
            diagram.with_config(controller="mpc")

    def test_apply_to_non_diagram_raises(self):
        with pytest.raises(VariantError, match=r"expected a Diagram"):
            apply_variant_config(42, controller="p")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Anonymous variants are not addressable by with_config.
# ---------------------------------------------------------------------------


class TestAnonymousVariants:
    def test_anonymous_variants_not_addressable(self):
        """A Variant built with no ``name=`` kwarg is anonymous; it gets
        skipped by the configurator's name index, so an override naming
        an unrelated variant should report 'available: []'."""
        anonymous_variant = Variant(
            choices={"p": _build_p_controller, "pi": _build_pi_controller},
            default="p",
            # name= deliberately omitted → anonymous
        )
        ctrl = select_variant(anonymous_variant)  # tagged with anonymous metadata
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(3.0, name="src"))
        builder.add(ctrl)
        builder.connect(src.output_ports[0], ctrl.input_ports[0])
        builder.export_output(ctrl.output_ports[0], name="y")
        diagram = builder.build(name="anon_root")

        # The diagram still simulates fine in its default state.
        assert _eval_root_output(diagram) == pytest.approx(6.0)

        # No-overrides reconfig works (default-off path).
        copy = diagram.with_config()
        assert _eval_root_output(copy) == pytest.approx(6.0)

        # But the anonymous variant cannot be addressed by name.
        with pytest.raises(VariantError, match=r"no Variant with name"):
            diagram.with_config(controller="pi")
