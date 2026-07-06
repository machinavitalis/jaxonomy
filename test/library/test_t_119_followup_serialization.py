# SPDX-License-Identifier: MIT
"""T-119-followup-serialization: TruthTable dict/JSON round-trip.

The TruthTable runtime form for ``rows`` is a list of
``(pattern_tuple, output_array)`` pairs whose pattern entries mix
``bool`` and the wildcard string ``"X"``. Phase 1 stored that as a
plain Python attribute (not part of ``@parameters``) since
``declare_static_parameters`` coerces lists to ``np.array``, which
clobbers nested structure.

This shipment adds explicit ``TruthTable.to_dict()`` / ``from_dict()``
helpers that normalize ``rows`` into a JSON-friendly shape: pattern
becomes a length-``n_inputs`` string of ``"0"``/``"1"``/``"X"``,
output becomes either a Python ``float`` or
``{"shape": [...], "data": [...]}`` for arrays. The pair survives
``json.dumps`` / ``json.loads`` and rebuilds an equivalent block.

The default-off path (no ``to_dict`` call) is byte-equivalent to the
phase-1 ``TruthTable(rows=...)`` constructor.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import TruthTable
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Helpers — reuse the phase-1 simulation harness so equivalence is checked
# end-to-end (constructor + JAX trace + simulator), not just on the dict.
# ---------------------------------------------------------------------------


def _run_truth_table(tt, input_values):
    """Wire ``len(input_values)`` Constants into ``tt`` and simulate.

    Returns the recorded output at the final timestep as a ``np.ndarray``.
    """
    sources = [library.Constant(v) for v in input_values]
    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, tt)
    for i, src in enumerate(sources):
        builder.connect(src.output_ports[0], tt.input_ports[i])
    diagram = builder.build()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": tt.output_ports[0]},
    )
    return np.asarray(results.outputs["out"])[-1]


def _all_input_combinations(n):
    """Enumerate all 2**n boolean input vectors as plain Python tuples."""
    return list(itertools.product((False, True), repeat=n))


# ---------------------------------------------------------------------------
# 4-row scalar table — full round-trip via dict and via JSON string.
# ---------------------------------------------------------------------------


AND_ROWS = [
    ((True, True), 1.0),
    ((True, False), 0.0),
    ((False, True), 0.0),
    ((False, False), 0.0),
]


def test_to_dict_then_from_dict_preserves_outputs_for_all_input_combos():
    """Round-trip through dict; verify every input combo agrees."""
    original = TruthTable(rows=AND_ROWS, n_inputs=2, default_output=0.0)
    payload = original.to_dict()
    restored = TruthTable.from_dict(payload)

    for combo in _all_input_combinations(2):
        out_orig = _run_truth_table(original, list(combo))
        out_rest = _run_truth_table(restored, list(combo))
        np.testing.assert_allclose(np.asarray(out_orig), np.asarray(out_rest))


def test_to_dict_payload_is_json_serializable_and_recoverable():
    """The to_dict() output must survive a real json.dumps / json.loads cycle."""
    original = TruthTable(rows=AND_ROWS, n_inputs=2, default_output=0.0)
    payload = original.to_dict()

    # Must round-trip through json (no numpy types, no tuples, no "X" mixed
    # with bool — all primitives).
    serialized = json.dumps(payload)
    reloaded = json.loads(serialized)

    restored = TruthTable.from_dict(reloaded)
    for combo in _all_input_combinations(2):
        out_orig = _run_truth_table(original, list(combo))
        out_rest = _run_truth_table(restored, list(combo))
        np.testing.assert_allclose(np.asarray(out_orig), np.asarray(out_rest))


# ---------------------------------------------------------------------------
# Wildcard "X" patterns survive the round-trip.
# ---------------------------------------------------------------------------


WILDCARD_ROWS = [
    (("X", True), 1.0),
    (("X", False), 0.0),
]


def test_wildcard_patterns_survive_round_trip():
    original = TruthTable(rows=WILDCARD_ROWS, n_inputs=2, default_output=-1.0)
    payload = original.to_dict()

    # Patterns should be encoded as strings containing the literal "X".
    pattern_strings = [row["pattern"] for row in payload["rows"]]
    assert pattern_strings == ["X1", "X0"]

    restored = TruthTable.from_dict(json.loads(json.dumps(payload)))
    for combo in _all_input_combinations(2):
        out_orig = _run_truth_table(original, list(combo))
        out_rest = _run_truth_table(restored, list(combo))
        np.testing.assert_allclose(np.asarray(out_orig), np.asarray(out_rest))


# ---------------------------------------------------------------------------
# Vector outputs survive the round-trip with shape and values intact.
# ---------------------------------------------------------------------------


def test_vector_outputs_round_trip():
    rows = [
        ((True,), np.array([1.0, 2.0, 3.0])),
        ((False,), np.array([4.0, 5.0, 6.0])),
    ]
    original = TruthTable(
        rows=rows, n_inputs=1, default_output=np.array([0.0, 0.0, 0.0])
    )
    payload = original.to_dict()

    # Vector outputs encoded as {"shape": [...], "data": [...]} dicts.
    for entry in payload["rows"]:
        assert isinstance(entry["output"], dict)
        assert entry["output"]["shape"] == [3]
        assert len(entry["output"]["data"]) == 3
    assert payload["default_output"]["shape"] == [3]

    # JSON-string round-trip and end-to-end equivalence.
    restored = TruthTable.from_dict(json.loads(json.dumps(payload)))
    out_orig_t = _run_truth_table(original, [True])
    out_rest_t = _run_truth_table(restored, [True])
    np.testing.assert_allclose(np.asarray(out_orig_t), np.asarray(out_rest_t))
    np.testing.assert_allclose(np.asarray(out_rest_t), [1.0, 2.0, 3.0])

    out_orig_f = _run_truth_table(original, [False])
    out_rest_f = _run_truth_table(restored, [False])
    np.testing.assert_allclose(np.asarray(out_orig_f), np.asarray(out_rest_f))
    np.testing.assert_allclose(np.asarray(out_rest_f), [4.0, 5.0, 6.0])


def test_2d_vector_output_round_trip_preserves_shape():
    """A non-1D ndarray output must reconstruct with the original shape."""
    rows = [((True,), np.array([[1.0, 2.0], [3.0, 4.0]]))]
    original = TruthTable(
        rows=rows,
        n_inputs=1,
        default_output=np.array([[0.0, 0.0], [0.0, 0.0]]),
    )
    payload = original.to_dict()
    assert payload["rows"][0]["output"]["shape"] == [2, 2]

    restored = TruthTable.from_dict(json.loads(json.dumps(payload)))
    # Verify the rebuilt block holds the same row data, by reading back via
    # to_dict — avoids requiring 2D-output simulation infra.
    re_payload = restored.to_dict()
    assert re_payload == payload


# ---------------------------------------------------------------------------
# Default-off API — the existing TruthTable(rows=...) constructor is
# byte-equivalent to phase 1 (no behavioural change).
# ---------------------------------------------------------------------------


def test_existing_constructor_unchanged_byte_equivalent():
    """Phase-1 callers that never call to_dict() see identical behaviour.

    The shipment must not alter ``__init__`` semantics: the same
    ``rows`` / ``n_inputs`` / ``default_output`` keyword set still
    produces a working block whose recorded outputs match the phase-1
    reference values across every input combination.
    """
    tt = TruthTable(rows=AND_ROWS, n_inputs=2, default_output=0.0)
    expected = {
        (True, True): 1.0,
        (True, False): 0.0,
        (False, True): 0.0,
        (False, False): 0.0,
    }
    for combo, expected_value in expected.items():
        out = _run_truth_table(tt, list(combo))
        np.testing.assert_allclose(float(out), expected_value)


# ---------------------------------------------------------------------------
# Encode/decode helpers behave on edge cases.
# ---------------------------------------------------------------------------


def test_encode_pattern_emits_compact_strings():
    assert TruthTable._encode_pattern((True, False, "X")) == "10X"
    assert TruthTable._encode_pattern(("X", "X", "X")) == "XXX"
    assert TruthTable._encode_pattern((False, False)) == "00"


def test_decode_pattern_inverse_of_encode():
    for pattern in [
        (True, False, "X"),
        ("X", True, True, False),
        (False,),
    ]:
        encoded = TruthTable._encode_pattern(pattern)
        decoded = TruthTable._decode_pattern(encoded, len(pattern))
        assert decoded == pattern


def test_decode_pattern_rejects_wrong_length_and_bad_chars():
    with pytest.raises(ValueError):
        TruthTable._decode_pattern("10", n_inputs=3)
    with pytest.raises(ValueError):
        TruthTable._decode_pattern("1Y0", n_inputs=3)


# ---------------------------------------------------------------------------
# from_dict accepts forwarded block kwargs (e.g. name).
# ---------------------------------------------------------------------------


def test_from_dict_forwards_block_kwargs_like_name():
    payload = TruthTable(
        rows=AND_ROWS, n_inputs=2, default_output=0.0
    ).to_dict()
    restored = TruthTable.from_dict(payload, name="renamed_table")
    assert restored.name == "renamed_table"
