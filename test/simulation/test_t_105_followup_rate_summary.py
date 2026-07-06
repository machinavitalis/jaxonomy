# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-rate-summary — richer rate-group diagnostic.

Verifies the new :func:`jaxonomy.simulation.rate_groups.rate_summary`
helper:

* ``format="text"`` includes every rate period and block name.
* ``format="markdown"`` produces valid markdown with ``## `` section
  headers and lists block names.
* ``format="json"`` produces a round-trippable JSON payload that
  exposes ``rate_groups``, ``mismatches`` and ``execution_order``.
* Invalid ``format`` raises a clear ``ValueError``.

The legacy :func:`format_rate_groups` remains byte-identical to its
T-105 phase-1 shape — covered indirectly by the existing
``test_t_105_multirate_phase1.py`` tests.
"""

from __future__ import annotations

import json

import pytest

import jaxonomy
from jaxonomy.simulation.rate_groups import (
    format_rate_groups,
    rate_summary,
)


pytestmark = pytest.mark.minimal


def _build_two_rate_three_block_diagram():
    """clk_fast(0.01s) -> ud_fast(0.01s); ud_slow(0.10s) standalone.

    Three leaves total at two distinct discrete rates.  The
    ``clk_fast -> ud_fast`` wiring is intentionally matched-rate so the
    diagram builds cleanly; ``ud_slow`` is fed by a ``Constant`` source
    (universal, so no rate mismatch) just so the builder accepts the
    diagram without a disconnected-input gripe.
    """
    from jaxonomy.library import Constant, DiscreteClock, UnitDelay

    builder = jaxonomy.DiagramBuilder()
    clk_fast = builder.add(DiscreteClock(dt=0.01, name="clk_fast"))
    ud_fast = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud_fast"))
    src_slow = builder.add(Constant(0.0, name="src_slow"))
    ud_slow = builder.add(UnitDelay(dt=0.10, initial_state=0.0, name="ud_slow"))
    builder.connect(clk_fast.output_ports[0], ud_fast.input_ports[0])
    builder.connect(src_slow.output_ports[0], ud_slow.input_ports[0])
    diag = builder.build()
    diag.create_context()  # triggers UnitDelay.initialize()
    return diag


def _build_mismatch_diagram():
    """clk_fast(0.01s) -> ud_slow(0.10s) — a single rate mismatch."""
    from jaxonomy.library import DiscreteClock, UnitDelay

    builder = jaxonomy.DiagramBuilder()
    clk_fast = builder.add(DiscreteClock(dt=0.01, name="clk_fast"))
    ud_slow = builder.add(UnitDelay(dt=0.10, initial_state=0.0, name="ud_slow"))
    builder.connect(clk_fast.output_ports[0], ud_slow.input_ports[0])
    diag = builder.build()
    diag.create_context()
    return diag


# =====================================================================
# format="text"
# =====================================================================


class TestRateSummaryText:
    def test_text_includes_both_periods_and_block_names(self):
        diag = _build_two_rate_three_block_diagram()
        out = rate_summary(diag, format="text")

        # Periods of both rate groups appear in the dump.
        assert "0.01" in out
        assert "0.1" in out  # 0.10 may render as 0.1
        # All three discrete blocks are named.
        assert "clk_fast" in out
        assert "ud_fast" in out
        assert "ud_slow" in out
        # Section headers ship.
        assert "rate groups" in out
        assert "mismatches" in out
        assert "execution order" in out

    def test_text_default_format_is_text(self):
        diag = _build_two_rate_three_block_diagram()
        assert rate_summary(diag) == rate_summary(diag, format="text")

    def test_text_lists_mismatches(self):
        diag = _build_mismatch_diagram()
        out = rate_summary(diag, format="text")
        # The mismatch line should mention both block names with their
        # incompatible rates.
        assert "clk_fast" in out
        assert "ud_slow" in out
        # No "<none>" placeholder because we have a mismatch.
        assert "mismatches:\n    <none>" not in out

    def test_text_no_mismatch_renders_none(self):
        from jaxonomy.library import Constant, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        k = builder.add(Constant(1.0, name="K"))
        ud = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud"))
        builder.connect(k.output_ports[0], ud.input_ports[0])
        diag = builder.build()
        diag.create_context()

        out = rate_summary(diag, format="text")
        assert "<none>" in out


# =====================================================================
# format="markdown"
# =====================================================================


class TestRateSummaryMarkdown:
    def test_markdown_has_section_headers(self):
        diag = _build_two_rate_three_block_diagram()
        out = rate_summary(diag, format="markdown")
        # CommonMark H2 headers — the load-bearing assertion from the task.
        assert "## " in out
        assert "## Rate Groups" in out
        assert "## Mismatches" in out
        assert "## Execution Order" in out

    def test_markdown_lists_block_names(self):
        diag = _build_two_rate_three_block_diagram()
        out = rate_summary(diag, format="markdown")
        for name in ("clk_fast", "ud_fast", "ud_slow"):
            assert f"`{name}`" in out

    def test_markdown_mentions_periods(self):
        diag = _build_two_rate_three_block_diagram()
        out = rate_summary(diag, format="markdown")
        assert "0.01" in out
        assert "0.1" in out


# =====================================================================
# format="json"
# =====================================================================


class TestRateSummaryJSON:
    def test_json_round_trips_via_json_loads(self):
        diag = _build_two_rate_three_block_diagram()
        out = rate_summary(diag, format="json")
        payload = json.loads(out)
        assert isinstance(payload, dict)
        assert set(payload) >= {"rate_groups", "mismatches", "execution_order"}

    def test_json_rate_groups_payload_shape(self):
        diag = _build_two_rate_three_block_diagram()
        payload = json.loads(rate_summary(diag, format="json"))
        groups = payload["rate_groups"]
        assert isinstance(groups, list)
        assert len(groups) >= 2  # at least the two discrete rates

        # Every group has the documented shape.
        for grp in groups:
            assert set(grp) == {"kind", "period", "offset", "count", "blocks"}
            assert isinstance(grp["blocks"], list)
            assert grp["count"] == len(grp["blocks"])

        # Both discrete periods are present.
        discrete_periods = sorted(
            g["period"] for g in groups if g["kind"] == "discrete"
        )
        assert 0.01 in discrete_periods
        assert 0.10 in discrete_periods

    def test_json_execution_order_lists_all_leaves(self):
        diag = _build_two_rate_three_block_diagram()
        payload = json.loads(rate_summary(diag, format="json"))
        order = payload["execution_order"]
        assert isinstance(order, list)
        assert set(order) == {"clk_fast", "ud_fast", "src_slow", "ud_slow"}

    def test_json_mismatches_populated_on_mismatch_diagram(self):
        diag = _build_mismatch_diagram()
        payload = json.loads(rate_summary(diag, format="json"))
        mismatches = payload["mismatches"]
        assert len(mismatches) == 1
        mm = mismatches[0]
        assert mm["src"] == "clk_fast"
        assert mm["dst"] == "ud_slow"
        assert "0.01" in mm["src_sample_time"]
        assert "0.1" in mm["dst_sample_time"]


# =====================================================================
# Validation
# =====================================================================


class TestRateSummaryValidation:
    def test_invalid_format_raises_value_error(self):
        diag = _build_two_rate_three_block_diagram()
        with pytest.raises(ValueError, match="unsupported format"):
            rate_summary(diag, format="yaml")  # type: ignore[arg-type]

    def test_invalid_format_message_mentions_supported_set(self):
        diag = _build_two_rate_three_block_diagram()
        with pytest.raises(ValueError) as exc:
            rate_summary(diag, format="xml")  # type: ignore[arg-type]
        msg = str(exc.value)
        assert "text" in msg
        assert "json" in msg
        assert "markdown" in msg


# =====================================================================
# Byte-equivalence: legacy format_rate_groups stays unchanged
# =====================================================================


class TestLegacyFormatRateGroupsUnchanged:
    def test_format_rate_groups_signature_and_shape_preserved(self):
        diag = _build_two_rate_three_block_diagram()
        out = format_rate_groups(diag)
        # Same legacy shape the phase-1 tests assert on.
        assert out.startswith("rate groups:\n")
        assert "clk_fast" in out
        assert "ud_slow" in out
