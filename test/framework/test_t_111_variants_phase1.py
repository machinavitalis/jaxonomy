# SPDX-License-Identifier: MIT

"""T-111 phase 1 — build-time Variants / Configurable Diagrams.

Covers the smallest useful slice of the variant DSL: a build-time
selector that picks one of N alternative sub-diagrams. Verifies:

    - ``select_variant(name="...")`` returns different concrete diagrams
      depending on the chosen name.
    - Omitting the name falls back to a documented default.
    - The active variant integrates correctly via ``simulate``.
    - Inactive variants are *never built*: their builders are not invoked
      and no system they would have created appears in the resolved
      diagram's registered-systems list.
    - Misuse (unknown choice, missing default, non-callable choice) raises
      a clear ``VariantError``.
"""

from __future__ import annotations

import pytest

import jax.numpy as jnp

import jaxonomy
from jaxonomy.framework import (
    Variant,
    VariantError,
    select_variant,
    variant_subsystem,
)
from jaxonomy.library import (
    Constant,
    Gain,
    Integrator,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures: a "controller" variant point with two pre-built sub-diagrams.
# ---------------------------------------------------------------------------


def _build_p_controller(gain: float = 2.0) -> jaxonomy.Diagram:
    """Build a trivial proportional controller: y = gain * u."""
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(gain, name="P_gain"))
    builder.export_input(p.input_ports[0], name="u")
    builder.export_output(p.output_ports[0], name="y")
    return builder.build(name="p_controller")


def _build_pi_controller(p_gain: float = 2.0, i_gain: float = 0.5) -> jaxonomy.Diagram:
    """Build a P+I-style controller (sums proportional and integrator paths).

    NOTE: structurally distinct from the P controller — extra blocks, extra
    ports — so the two diagrams are easy to tell apart.
    """
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(p_gain, name="P_gain"))
    i_gain_block = builder.add(Gain(i_gain, name="I_gain"))
    integ = builder.add(Integrator(0.0, name="I_state"))
    # Wire the integrator path: u -> I_gain -> Integrator
    builder.connect(i_gain_block.output_ports[0], integ.input_ports[0])
    # Export the same input twice (P path + I path) by using one and
    # treating the I path as a separate input. For phase-1 tests we keep
    # the API surface tiny: expose only the P path's input as the diagram
    # input, and expose the integrator's output as the diagram output. The
    # I_gain block stays disconnected from the diagram input but still
    # contributes to the registered-systems count, which is what the
    # "different shape" assertions key off of.
    builder.export_input(p.input_ports[0], name="u")
    builder.export_input(i_gain_block.input_ports[0], name="u_i")
    builder.export_output(integ.output_ports[0], name="y")
    return builder.build(name="pi_controller")


def _make_controller_variant() -> Variant:
    return Variant(
        choices={
            "p": lambda: _build_p_controller(),
            "pi": lambda: _build_pi_controller(),
        },
        default="p",
        name="controller",
    )


# ---------------------------------------------------------------------------
# Selection: different names produce different diagrams.
# ---------------------------------------------------------------------------


class TestSelection:
    def test_select_named_choice_returns_that_diagram(self):
        v = _make_controller_variant()

        p = select_variant(v, name="p")
        pi = select_variant(v, name="pi")

        assert p.name == "p_controller"
        assert pi.name == "pi_controller"
        # Different sub-system counts: P has 1 block, PI has 3.
        assert len(p.nodes) == 1
        assert len(pi.nodes) == 3

    def test_default_when_name_is_none(self):
        v = _make_controller_variant()
        d = select_variant(v)  # no name -> default
        assert d.name == "p_controller"

    def test_unknown_choice_raises(self):
        v = _make_controller_variant()
        with pytest.raises(VariantError, match=r"unknown choice 'lqr'"):
            select_variant(v, name="lqr")

    def test_choice_names_introspection(self):
        v = _make_controller_variant()
        assert v.choice_names == ("p", "pi")


# ---------------------------------------------------------------------------
# Inactive choices are never built.
# ---------------------------------------------------------------------------


class TestInactiveNotBuilt:
    def test_unselected_builder_is_not_invoked(self):
        # The unselected builder raises if invoked — proves it never runs.
        invocations = {"p": 0, "explode": 0}

        def _build_p():
            invocations["p"] += 1
            return _build_p_controller()

        def _explode():
            invocations["explode"] += 1
            raise AssertionError("Inactive variant builder was invoked!")

        v = Variant(
            choices={"p": _build_p, "explode": _explode},
            default="p",
            name="inactive_check",
        )

        result = select_variant(v, name="p")
        assert result.name == "p_controller"
        assert invocations["p"] == 1
        assert invocations["explode"] == 0

    def test_inactive_systems_not_in_registered_list(self):
        # When we select "p", no block created by the "pi" builder should
        # show up in the resulting diagram's node list.
        v = _make_controller_variant()
        active = select_variant(v, name="p")
        node_names = {node.name for node in active.nodes}
        # PI-only systems must not be present.
        assert "I_gain" not in node_names
        assert "I_state" not in node_names
        # P-only system must be present.
        assert "P_gain" in node_names


# ---------------------------------------------------------------------------
# Active variant integrates correctly via simulate.
# ---------------------------------------------------------------------------


class TestSimulateActiveVariant:
    """Wrap a selected variant inside a parent diagram and simulate it."""

    @staticmethod
    def _wrap_and_simulate(controller: jaxonomy.Diagram, u_value: float) -> jnp.ndarray:
        """Drive ``controller`` with a constant ``u_value`` for 0.5s.

        Returns the diagram-level output at the end of simulation. Works
        for both the P (1 input) and PI (2 inputs) sub-diagrams by feeding
        every exported input with the same constant.
        """
        builder = jaxonomy.DiagramBuilder()
        u_src = builder.add(Constant(u_value, name="u_src"))
        builder.add(controller)
        for port in controller.input_ports:
            builder.connect(u_src.output_ports[0], port)
        diagram = builder.build(name="root")

        ctx = diagram.create_context()
        results = jaxonomy.simulate(diagram, ctx, (0.0, 0.5))
        # Output port 0 of the wrapped controller, evaluated in the final
        # context, gives us the active variant's output.
        return controller.output_ports[0].eval(results.context)

    def test_p_variant_output(self):
        v = _make_controller_variant()
        active = select_variant(v, name="p")
        # P controller: y = 2.0 * u. With u=3.0 -> y=6.0 (constant; no state).
        y = self._wrap_and_simulate(active, u_value=3.0)
        assert jnp.allclose(y, 6.0)

    def test_pi_variant_output(self):
        v = _make_controller_variant()
        active = select_variant(v, name="pi")
        # PI controller's exported output is the integrator state, which
        # integrates I_gain(=0.5) * u_i over 0.5s. With u=4.0 -> 0.5*4*0.5=1.0.
        y = self._wrap_and_simulate(active, u_value=4.0)
        assert jnp.allclose(y, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# variant_subsystem one-liner.
# ---------------------------------------------------------------------------


class TestVariantSubsystem:
    def test_one_liner_picks_default_when_called_without_name(self):
        select = variant_subsystem(
            choices={
                "p": lambda: _build_p_controller(),
                "pi": lambda: _build_pi_controller(),
            },
            default="pi",
            name="ctrl",
        )
        d = select()
        assert d.name == "pi_controller"

    def test_one_liner_named_call(self):
        select = variant_subsystem(
            choices={
                "p": lambda: _build_p_controller(),
                "pi": lambda: _build_pi_controller(),
            },
            default="pi",
            name="ctrl",
        )
        d = select(name="p")
        assert d.name == "p_controller"

    def test_implicit_default_is_first_key(self):
        # default omitted -> first insertion-order key wins
        select = variant_subsystem(
            choices={
                "p": lambda: _build_p_controller(),
                "pi": lambda: _build_pi_controller(),
            },
            name="ctrl",
        )
        assert select.variant.default == "p"
        assert select().name == "p_controller"

    def test_resolver_returns_fresh_instance_each_call(self):
        # Each call invokes the builder, producing a *new* Diagram object.
        # This matters because re-binding a Diagram into a parent builder
        # twice would normally trip the "already registered" check.
        select = variant_subsystem(
            choices={"p": lambda: _build_p_controller()},
            default="p",
            name="ctrl",
        )
        a = select()
        b = select()
        assert a is not b


# ---------------------------------------------------------------------------
# Validation: misuse of the Variant container itself.
# ---------------------------------------------------------------------------


class TestVariantValidation:
    def test_empty_choices_rejected(self):
        with pytest.raises(VariantError, match=r"at least one entry"):
            Variant(choices={}, default="anything", name="bad")

    def test_bad_default_rejected(self):
        with pytest.raises(VariantError, match=r"default 'nope' is not in choices"):
            Variant(
                choices={"p": lambda: _build_p_controller()},
                default="nope",
                name="bad",
            )

    def test_non_callable_choice_rejected(self):
        with pytest.raises(VariantError, match=r"is not callable"):
            Variant(
                choices={"p": "not a callable"},  # type: ignore[dict-item]
                default="p",
                name="bad",
            )

    def test_builder_returning_non_systembase_rejected(self):
        v = Variant(
            choices={"p": lambda: 42},  # type: ignore[dict-item,return-value]
            default="p",
            name="bad",
        )
        with pytest.raises(VariantError, match=r"expected a SystemBase"):
            select_variant(v)
