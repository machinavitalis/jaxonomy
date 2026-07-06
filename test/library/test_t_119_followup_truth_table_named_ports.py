# SPDX-License-Identifier: MIT

"""Regression test for T-119-followup-truth-table-named-ports.

Before the followup, ``TruthTableBuilder(input_names=...)`` silently dropped
the labels on the way to ``TruthTable.__init__`` — the names survived only as
``.row(...)`` keyword targets and the built block's ``input_ports`` came out
as anonymous ``in_0`` / ``in_1`` slots, invisible in ``print_schedule`` /
model JSON / error messages. Wide-table positional wiring became a footgun.

After the followup the names propagate through to ``declare_input_port(name=...)``.
"""

from __future__ import annotations

from jaxonomy.library import TruthTableBuilder, TruthTable


def test_builder_input_names_propagate_to_input_ports():
    b = TruthTableBuilder(
        n_inputs=3,
        default_output=False,
        input_names=("brake", "throttle", "ignition"),
    )
    b.row(brake=True, throttle=False, output=False)
    b.row(brake=False, throttle=True, ignition=True, output=True)
    block = b.build()

    port_names = [p.name for p in block.input_ports]
    assert port_names == ["brake", "throttle", "ignition"]


def test_builder_without_input_names_keeps_anonymous_ports():
    """When ``input_names`` is the default placeholder (``in1``, ``in2``...),
    don't force them onto the built block — preserves the pre-followup
    byte-equivalence for callers that never set the kwarg."""
    b = TruthTableBuilder(n_inputs=2, default_output=False)
    b.row(in1=True, in2=False, output=True)
    block = b.build()

    port_names = [p.name for p in block.input_ports]
    # Default port names should remain whatever declare_input_port() picks,
    # *not* the in1/in2 builder labels.
    assert port_names != ["in1", "in2"]


def test_truth_table_constructor_accepts_input_names():
    """The direct ``TruthTable`` constructor also accepts ``input_names``
    so users who skip the builder still get labelled ports."""
    block = TruthTable(
        rows=[((True, False), True)],
        n_inputs=2,
        default_output=False,
        input_names=("enable", "armed"),
    )
    assert [p.name for p in block.input_ports] == ["enable", "armed"]


def test_truth_table_rejects_mismatched_input_names_length():
    import pytest

    with pytest.raises(Exception, match="input_names"):
        TruthTable(
            rows=[((True, False), True)],
            n_inputs=2,
            default_output=False,
            input_names=("only_one",),  # too short
        )
