# SPDX-License-Identifier: MIT

"""T-111 phase 4 — multi-variant resolution policies.

Exercises:

* ``expand_all_variant_configs(diagram)`` — Cartesian product of every
  named variant's choices, returned as a list of binding dicts.
* ``iter_variant_configurations(diagram)`` — generator yielding
  ``(config, configured_diagram)`` pairs, with each diagram already
  reconfigured via ``load_variant_config``.

Anonymous Variants are skipped (consistent with phase 2's
``dump_variant_config`` policy).
"""

from __future__ import annotations

from itertools import product

import pytest

import jaxonomy
from jaxonomy.framework import Variant, select_variant
from jaxonomy.framework.variants import (
    dump_variant_config,
    expand_all_variant_configs,
    iter_variant_configurations,
)
from jaxonomy.library import Constant, Gain, Integrator


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixture builders (same shape as the phase-2 / phase-3 tests).
# ---------------------------------------------------------------------------


def _p():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="P_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="p")


def _pi():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="P_gain"))
    i_gain = b.add(Gain(0.5, name="I_gain"))
    integ = b.add(Integrator(0.0, name="I_state"))
    b.connect(g.output_ports[0], i_gain.input_ports[0])
    b.connect(i_gain.output_ports[0], integ.input_ports[0])
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="pi")


def _lti():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(1.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="lti")


def _doubler():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="doubler")


def _triple():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(3.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="triple")


def _build_root_2x2():
    controller = select_variant(
        Variant(choices={"p": _p, "pi": _pi}, default="p", name="controller")
    )
    plant = select_variant(
        Variant(choices={"lti": _lti, "doubler": _doubler}, default="lti", name="plant")
    )
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(3.0, name="src"))
    b.add(controller)
    b.add(plant)
    b.connect(src.output_ports[0], controller.input_ports[0])
    b.connect(controller.output_ports[0], plant.input_ports[0])
    b.export_output(plant.output_ports[0], name="y")
    return b.build(name="root")


def _build_root_2x3():
    """Same as 2x2 fixture but with a 3-choice plant variant."""
    controller = select_variant(
        Variant(choices={"p": _p, "pi": _pi}, default="p", name="controller")
    )
    plant = select_variant(
        Variant(
            choices={"lti": _lti, "doubler": _doubler, "triple": _triple},
            default="lti",
            name="plant",
        )
    )
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(3.0, name="src"))
    b.add(controller)
    b.add(plant)
    b.connect(src.output_ports[0], controller.input_ports[0])
    b.connect(controller.output_ports[0], plant.input_ports[0])
    b.export_output(plant.output_ports[0], name="y")
    return b.build(name="root")


# ---------------------------------------------------------------------------
# expand_all_variant_configs — Cartesian product semantics.
# ---------------------------------------------------------------------------


def test_expand_returns_full_cartesian_product():
    diagram = _build_root_2x2()
    configs = expand_all_variant_configs(diagram)
    # 2 controllers × 2 plants = 4 configurations.
    assert len(configs) == 4
    expected = [
        {"controller": c, "plant": p}
        for c, p in product(("p", "pi"), ("lti", "doubler"))
    ]
    assert configs == expected


def test_expand_scales_with_variant_size():
    diagram = _build_root_2x3()
    configs = expand_all_variant_configs(diagram)
    assert len(configs) == 6  # 2 × 3


def test_expand_on_vanilla_diagram_returns_single_empty_config():
    """No named variants → single trivial configuration (empty dict)
    — matches the empty-Cartesian-product convention used by itertools.product."""
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(1.0, name="g"))
    b.export_output(g.output_ports[0], name="y")
    diagram = b.build(name="vanilla")
    assert expand_all_variant_configs(diagram) == [{}]


def test_expand_skips_anonymous_variants():
    """Anonymous Variants (no name=) don't contribute axes to the
    expansion — they cannot be addressed across builds."""
    anonymous = Variant(choices={"a": _p, "b": _pi}, default="a")
    named = Variant(
        choices={"lti": _lti, "doubler": _doubler}, default="lti", name="plant"
    )
    a_sub = select_variant(anonymous)
    p_sub = select_variant(named)
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(3.0, name="src"))
    b.add(a_sub)
    b.add(p_sub)
    b.connect(src.output_ports[0], a_sub.input_ports[0])
    b.connect(a_sub.output_ports[0], p_sub.input_ports[0])
    b.export_output(p_sub.output_ports[0], name="y")
    diagram = b.build(name="mixed")

    configs = expand_all_variant_configs(diagram)
    # Only the named "plant" variant should appear; the anonymous one
    # contributes no axis (so the product collapses to its 2 choices).
    assert configs == [{"plant": "lti"}, {"plant": "doubler"}]


def test_expand_is_deterministic_order():
    """List ordering: outer variant in tree-traversal order, choices
    in insertion order. Run twice and compare."""
    diagram = _build_root_2x2()
    a = expand_all_variant_configs(diagram)
    b = expand_all_variant_configs(diagram)
    assert a == b


# ---------------------------------------------------------------------------
# iter_variant_configurations — generator that applies each config.
# ---------------------------------------------------------------------------


def test_iter_yields_pair_per_configuration():
    diagram = _build_root_2x2()
    pairs = list(iter_variant_configurations(diagram))
    assert len(pairs) == 4
    # Each yielded diagram has the expected binding.
    for cfg, configured in pairs:
        assert dump_variant_config(configured) == cfg


def test_iter_configurations_match_expand_list():
    """``iter_variant_configurations`` and ``expand_all_variant_configs``
    must agree on the binding set."""
    diagram = _build_root_2x2()
    expanded = expand_all_variant_configs(diagram)
    iter_configs = [cfg for cfg, _ in iter_variant_configurations(diagram)]
    assert expanded == iter_configs


def test_iter_yields_independent_diagrams_per_config():
    """Each yielded diagram is a freshly-applied config; mutating /
    selecting in one binding should not affect the rest."""
    diagram = _build_root_2x2()
    diagrams = [d for _, d in iter_variant_configurations(diagram)]
    # All four diagrams should be distinct objects (no aliasing).
    ids = {id(d) for d in diagrams}
    assert len(ids) == 4
    # And each one carries the dump of its own binding (sanity).
    for d, cfg in zip(diagrams, expand_all_variant_configs(diagram)):
        assert dump_variant_config(d) == cfg


def test_iter_on_vanilla_diagram_yields_single_pair():
    """No named variants → single ({}, diagram-untouched) pair."""
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(1.0, name="g"))
    b.export_output(g.output_ports[0], name="y")
    diagram = b.build(name="vanilla")

    pairs = list(iter_variant_configurations(diagram))
    assert len(pairs) == 1
    cfg, configured = pairs[0]
    assert cfg == {}
    # load_variant_config with empty dict returns the input untouched.
    assert configured is diagram


# ---------------------------------------------------------------------------
# Smoke test: every expanded configuration round-trips through dump+load.
# ---------------------------------------------------------------------------


def test_every_expanded_config_dumps_back_to_itself():
    """Sanity-check the closure of the expansion → load → dump cycle."""
    diagram = _build_root_2x2()
    for cfg, configured in iter_variant_configurations(diagram):
        assert dump_variant_config(configured) == cfg
