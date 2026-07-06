# SPDX-License-Identifier: MIT

"""T-105 phase 3 — ``Diagram.print_schedule()`` inspection helper.

Phase-3 covers the visualization affordance promised in the original
T-105 phasing list: a ``Diagram.print_schedule()`` method that prints
which blocks fire at which periods. Phase-1 shipped the underlying
``format_rate_groups`` and ``rate_summary`` string formatters; phase-3
exposes them as a convenience method on the built diagram so users
don't need to know the helper module path.
"""

from __future__ import annotations

import io
import json

import pytest

import jaxonomy


pytestmark = pytest.mark.minimal


def _build_two_rate_diagram():
    """clk_fast(0.01s) → ud_fast; src_slow(Constant) → ud_slow(0.10s).

    Three discrete leaves at two distinct periods plus a Constant
    source. Same fixture as the rate_summary tests so the assertions
    line up.
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
    diag.create_context()
    return diag


# ---------------------------------------------------------------------------
# Default behavior — prints to stdout, format="text".
# ---------------------------------------------------------------------------


def test_print_schedule_writes_text_to_stdout(capsys):
    diag = _build_two_rate_diagram()
    result = diag.print_schedule()

    # Convenience method returns None — no side-channel value to confuse
    # callers; the printed output is the API.
    assert result is None

    captured = capsys.readouterr()
    # Periods of both rate groups appear.
    assert "0.01" in captured.out
    assert "0.1" in captured.out
    # Every block name appears.
    assert "clk_fast" in captured.out
    assert "ud_fast" in captured.out
    assert "ud_slow" in captured.out
    # The text-format section headers ship.
    assert "rate groups" in captured.out
    assert "execution order" in captured.out


def test_print_schedule_default_matches_rate_summary_text(capsys):
    """`print_schedule()` and `rate_summary(diag, format="text")` agree."""
    from jaxonomy.simulation.rate_groups import rate_summary

    diag = _build_two_rate_diagram()
    expected = rate_summary(diag, format="text")

    diag.print_schedule()
    captured = capsys.readouterr()
    # `print` appends a trailing newline; the formatter output ends without one.
    assert captured.out.rstrip("\n") == expected.rstrip("\n")


# ---------------------------------------------------------------------------
# Format kwarg — markdown and json variants.
# ---------------------------------------------------------------------------


def test_print_schedule_markdown_format(capsys):
    diag = _build_two_rate_diagram()
    diag.print_schedule(format="markdown")
    out = capsys.readouterr().out
    # Markdown variant ships ``## `` section headers.
    assert "## " in out
    assert "clk_fast" in out


def test_print_schedule_json_format_round_trips(capsys):
    diag = _build_two_rate_diagram()
    diag.print_schedule(format="json")
    out = capsys.readouterr().out
    # JSON variant is parseable.
    payload = json.loads(out)
    assert "rate_groups" in payload
    assert "execution_order" in payload


# ---------------------------------------------------------------------------
# `file=` redirection.
# ---------------------------------------------------------------------------


def test_print_schedule_writes_to_supplied_file_handle():
    diag = _build_two_rate_diagram()
    buf = io.StringIO()
    diag.print_schedule(file=buf)
    contents = buf.getvalue()
    assert "rate groups" in contents
    assert "clk_fast" in contents


def test_print_schedule_does_not_touch_stdout_when_file_given(capsys):
    diag = _build_two_rate_diagram()
    buf = io.StringIO()
    diag.print_schedule(file=buf)

    captured = capsys.readouterr()
    assert captured.out == ""  # all output went to `buf`, not stdout
    assert "clk_fast" in buf.getvalue()


# ---------------------------------------------------------------------------
# Invalid format propagates the underlying ValueError verbatim.
# ---------------------------------------------------------------------------


def test_print_schedule_invalid_format_raises_value_error():
    diag = _build_two_rate_diagram()
    with pytest.raises(ValueError, match="unsupported format"):
        diag.print_schedule(format="yaml")
