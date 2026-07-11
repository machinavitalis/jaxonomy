# SPDX-License-Identifier: MIT

"""T-141: a shared top-level Parameter alias must keep driving its
referencing blocks through ``with_parameters`` — both on a live diagram and
after a ``model_json`` round-trip.

Historically both paths were silent no-ops: ``Diagram.with_parameters``
swapped the alias for a fresh ``Parameter`` (leaving the dependents of the
old one stale), and a deserialized block parameter (a string expression
``"k"`` evaluated against ``py_namespace``) lost its dependency link on
deepcopy because the namespace was shared unmapped and expression
dependencies were never re-registered.
"""

import json

import numpy as np
import pytest

import jaxonomy
from jaxonomy import DiagramBuilder, Parameter
from jaxonomy.library import Constant, Gain
from jaxonomy.dashboard.serialization.from_model_json import load_model
from jaxonomy.dashboard.serialization.to_model_json import convert

pytestmark = pytest.mark.minimal


def _build_aliased_diagram():
    """const(3.0) → gain(k), with k declared as a top-level diagram parameter."""
    builder = DiagramBuilder()
    k_param = Parameter(name="k", value=np.array(2.0))
    gain = builder.add(Gain(k_param, name="gain"))
    const = builder.add(Constant(3.0, name="const"))
    builder.connect(const.output_ports[0], gain.input_ports[0])
    return builder.build(name="aliased", parameters={"k": k_param})


def _roundtrip(diagram):
    model, _ = convert(diagram)
    sc = load_model(json.loads(json.dumps(model.to_dict())))
    return sc.diagram


def _gain_output(diagram):
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.1),
        recorded_signals={"y": diagram["gain"].output_ports[0]},
    )
    return float(np.asarray(results.outputs["y"])[-1])


def test_alias_listed_on_live_and_loaded():
    diagram = _build_aliased_diagram()
    assert "k" in diagram.list_parameters()
    loaded = _roundtrip(diagram)
    assert "k" in loaded.list_parameters()


def test_live_alias_propagates_to_block():
    diagram = _build_aliased_diagram()
    assert _gain_output(diagram) == pytest.approx(6.0)

    updated = diagram.with_parameters({"k": np.array(5.0)})
    assert _gain_output(updated) == pytest.approx(15.0)


def test_live_alias_update_leaves_original_unchanged():
    diagram = _build_aliased_diagram()
    diagram.with_parameters({"k": np.array(5.0)})
    assert _gain_output(diagram) == pytest.approx(6.0)


def test_roundtrip_alias_propagates_to_block():
    loaded = _roundtrip(_build_aliased_diagram())
    assert _gain_output(loaded) == pytest.approx(6.0)

    updated = loaded.with_parameters({"k": np.array(5.0)})
    assert _gain_output(updated) == pytest.approx(15.0)


def test_roundtrip_alias_update_leaves_source_unchanged():
    loaded = _roundtrip(_build_aliased_diagram())
    updated = loaded.with_parameters({"k": np.array(5.0)})
    assert _gain_output(loaded) == pytest.approx(6.0)
    assert _gain_output(updated) == pytest.approx(15.0)


def test_roundtrip_repeated_updates_are_independent():
    """Each with_parameters copy owns its alias: updates don't leak between copies."""
    loaded = _roundtrip(_build_aliased_diagram())
    up_a = loaded.with_parameters({"k": np.array(5.0)})
    up_b = loaded.with_parameters({"k": np.array(10.0)})
    assert _gain_output(up_a) == pytest.approx(15.0)
    assert _gain_output(up_b) == pytest.approx(30.0)


def test_roundtrip_block_path_still_works():
    """The functional per-block path (T-142's workaround) must keep working."""
    loaded = _roundtrip(_build_aliased_diagram())
    updated = loaded.with_parameters({"gain.gain": np.array(4.0)})
    assert _gain_output(updated) == pytest.approx(12.0)
