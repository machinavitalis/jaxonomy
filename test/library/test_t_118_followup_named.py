# SPDX-License-Identifier: MIT
"""T-118-followup-multi-port-string-keys: ``MultiPortSwitch.choice_names``.

Pin the build-time named-choice helper:
  - ``choice_names`` is stored and exposed as a property.
  - ``index_of("low" / "medium" / "high")`` maps to the right integer.
  - ``index_of`` also accepts already-integer selectors and range-checks them.
  - Default ``choice_names=None`` keeps the runtime path byte-equivalent to
    T-118 phase 1 (same output for both int selector paths, identical
    forward result whether ``choice_names`` is provided or not).
  - Validation: unknown strings, length mismatch, duplicate entries, and
    non-string entries all raise ``BlockParameterError`` at construction.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library import MultiPortSwitch
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _eval_multiport(mps, selector_value, data_values):
    """Build a tiny MultiPortSwitch diagram and return the final output."""
    sel = library.Constant(float(selector_value))
    data_blocks = [library.Constant(float(v)) for v in data_values]

    builder = jaxonomy.DiagramBuilder()
    builder.add(sel, *data_blocks, mps)
    builder.connect(sel.output_ports[0], mps.input_ports[0])
    for i, blk in enumerate(data_blocks):
        builder.connect(blk.output_ports[0], mps.input_ports[i + 1])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"y": mps.output_ports[0]},
    )
    return float(np.asarray(results.outputs["y"])[-1])


# --- choice_names attribute -------------------------------------------------


def test_choice_names_default_is_none():
    mps = MultiPortSwitch(n_data_inputs=3)
    assert mps.choice_names is None


def test_choice_names_stored_as_tuple():
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    assert mps.choice_names == ("low", "medium", "high")


# --- index_of: string lookup ------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("low", 0), ("medium", 1), ("high", 2),
])
def test_index_of_resolves_string_selector(name, expected):
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    assert mps.index_of(name) == expected


# --- index_of: integer pass-through ----------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2])
def test_index_of_accepts_in_range_int(k):
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    assert mps.index_of(k) == k


def test_index_of_int_works_without_choice_names():
    """Integers still resolve even when choice_names is None."""
    mps = MultiPortSwitch(n_data_inputs=3)
    assert mps.index_of(0) == 0
    assert mps.index_of(2) == 2


# --- index_of: error paths --------------------------------------------------


def test_index_of_unknown_string_raises():
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    with pytest.raises(BlockParameterError):
        mps.index_of("ludicrous")


def test_index_of_string_without_choice_names_raises():
    mps = MultiPortSwitch(n_data_inputs=3)
    with pytest.raises(BlockParameterError):
        mps.index_of("low")


def test_index_of_out_of_range_int_raises():
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    with pytest.raises(BlockParameterError):
        mps.index_of(3)
    with pytest.raises(BlockParameterError):
        mps.index_of(-1)


# --- choice_names construction validation -----------------------------------


def test_choice_names_length_mismatch_raises():
    with pytest.raises(BlockParameterError):
        MultiPortSwitch(n_data_inputs=3, choice_names=("a", "b"))


def test_choice_names_duplicate_entries_raise():
    with pytest.raises(BlockParameterError):
        MultiPortSwitch(n_data_inputs=3, choice_names=("a", "a", "b"))


def test_choice_names_empty_string_entry_raises():
    with pytest.raises(BlockParameterError):
        MultiPortSwitch(n_data_inputs=3, choice_names=("a", "", "b"))


def test_choice_names_non_string_entry_raises():
    with pytest.raises(BlockParameterError):
        MultiPortSwitch(n_data_inputs=3, choice_names=("a", 1, "b"))


# --- end-to-end: string selector resolved at build time routes correctly ----


@pytest.mark.parametrize("name,expected", [
    ("low", 10.0), ("medium", 20.0), ("high", 30.0),
])
def test_named_selector_routes_to_correct_branch(name, expected):
    """index_of(name) feeds a Constant block which selects the right port."""
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    y = _eval_multiport(mps, mps.index_of(name), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(y, expected)


@pytest.mark.parametrize("k,expected", [(0, 10.0), (1, 20.0), (2, 30.0)])
def test_int_selector_still_works_with_choice_names(k, expected):
    """Integer selectors keep working when choice_names is supplied."""
    mps = MultiPortSwitch(n_data_inputs=3,
                          choice_names=("low", "medium", "high"))
    y = _eval_multiport(mps, k, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(y, expected)


# --- default-off byte equivalence ------------------------------------------


@pytest.mark.parametrize("k,expected", [(0, 10.0), (1, 20.0), (2, 30.0)])
def test_default_choice_names_none_matches_phase1(k, expected):
    """With choice_names=None the runtime output is identical to phase 1."""
    mps = MultiPortSwitch(n_data_inputs=3)
    y = _eval_multiport(mps, k, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(y, expected)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_named_and_unnamed_produce_byte_equivalent_runtime(k):
    """Named and unnamed switches produce identical outputs for the same int."""
    mps_named = MultiPortSwitch(
        n_data_inputs=3, choice_names=("low", "medium", "high"))
    mps_plain = MultiPortSwitch(n_data_inputs=3)
    data = [10.0, 20.0, 30.0]
    y_named = _eval_multiport(mps_named, k, data)
    y_plain = _eval_multiport(mps_plain, k, data)
    assert y_named == y_plain
