# SPDX-License-Identifier: MIT

"""Tests for T-115-followup-saturate-symmetric-kwarg.

Pre-fix, the canonical symmetric :class:`Saturate(3.0)` style call was
unavailable — users had to write
``Saturate(lower_limit=-3.0, upper_limit=3.0)`` for what should have
been a single-scalar declaration. Every closed-loop tutorial wrote out
the full asymmetric form.

Post-fix, an opt-in ``limit=L`` kwarg expands to the symmetric
``upper_limit=+L, lower_limit=-L`` form. Mutually exclusive with
explicit ``upper_limit`` / ``lower_limit`` and with the dynamic-limit
flags. Must be a positive finite scalar.

Tests:
* ``Saturate(limit=L)`` clips symmetrically at ``[-L, +L]``.
* Asymmetric construction (the legacy form) is unchanged.
* ``limit`` combined with explicit ``upper_limit`` raises.
* ``limit`` combined with explicit ``lower_limit`` raises.
* ``limit`` combined with ``enable_dynamic_upper_limit`` raises.
* ``limit=0`` / negative / non-finite raise.
* ``limit`` interacts correctly with the smooth-mode validation.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------


class TestSymmetricExpansion:
    def test_limit_expands_to_symmetric_bounds(self):
        builder = jaxonomy.DiagramBuilder()
        # Drive the saturator with a sine ramping past the bounds.
        clk = builder.add(library.Clock())
        gain = builder.add(library.Gain(gain=4.0))  # ramps to ±4.
        sat = builder.add(library.Saturate(limit=2.0))
        builder.connect(clk.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], sat.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 1.0),
            recorded_signals={"y": sat.output_ports[0]},
        )
        y = np.asarray(results.outputs["y"])
        assert float(np.max(y)) <= 2.0 + 1e-9
        assert float(np.min(y)) >= -2.0 - 1e-9
        # And the output actually reaches the clip value (gain ramps to
        # 4 over 1 s, so we expect y to be saturated for at least the
        # upper half of the sweep).
        assert float(np.max(y)) >= 2.0 - 1e-9

    def test_asymmetric_construction_unchanged(self):
        """The pre-fix ``upper_limit`` / ``lower_limit`` style still works."""
        builder = jaxonomy.DiagramBuilder()
        clk = builder.add(library.Clock())
        gain = builder.add(library.Gain(gain=4.0))
        sat = builder.add(library.Saturate(
            upper_limit=3.0, lower_limit=-1.0,
        ))
        builder.connect(clk.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], sat.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 1.0),
            recorded_signals={"y": sat.output_ports[0]},
        )
        y = np.asarray(results.outputs["y"])
        assert float(np.max(y)) <= 3.0 + 1e-9
        assert float(np.min(y)) >= -1.0 - 1e-9


# ---------------------------------------------------------------------
# Mutual-exclusion validation
# ---------------------------------------------------------------------


class TestMutualExclusion:
    def test_limit_with_upper_limit_raises(self):
        with pytest.raises(BlockParameterError,
                           match="cannot be combined with explicit"):
            library.Saturate(limit=2.0, upper_limit=3.0)

    def test_limit_with_lower_limit_raises(self):
        with pytest.raises(BlockParameterError,
                           match="cannot be combined with explicit"):
            library.Saturate(limit=2.0, lower_limit=-3.0)

    def test_limit_with_dynamic_upper_raises(self):
        with pytest.raises(BlockParameterError,
                           match="cannot be combined with"):
            library.Saturate(limit=2.0, enable_dynamic_upper_limit=True)

    def test_limit_with_dynamic_lower_raises(self):
        with pytest.raises(BlockParameterError,
                           match="cannot be combined with"):
            library.Saturate(limit=2.0, enable_dynamic_lower_limit=True)


# ---------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------


class TestLimitRangeValidation:
    def test_zero_limit_raises(self):
        with pytest.raises(BlockParameterError, match="positive finite"):
            library.Saturate(limit=0.0)

    def test_negative_limit_raises(self):
        with pytest.raises(BlockParameterError, match="positive finite"):
            library.Saturate(limit=-1.0)

    def test_infinite_limit_raises(self):
        with pytest.raises(BlockParameterError, match="positive finite"):
            library.Saturate(limit=float("inf"))


# ---------------------------------------------------------------------
# Composition with mode="smooth"
# ---------------------------------------------------------------------


class TestSmoothCompose:
    def test_limit_with_smooth_mode(self):
        """``Saturate(limit=L, mode='smooth')`` works and smooth-clamps
        symmetrically at ``[-L, +L]``."""
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(value=5.0, name="src"))
        sat = builder.add(library.Saturate(limit=1.5, mode="smooth"))
        builder.connect(src.output_ports[0], sat.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        # Smooth mode honoured.
        assert sat.mode == "smooth"
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.05),
            recorded_signals={"y": sat.output_ports[0]},
        )
        y_pos = float(np.asarray(results.outputs["y"])[0])
        # Now flip the source to -5.0 and re-run.
        builder2 = jaxonomy.DiagramBuilder()
        src2 = builder2.add(library.Constant(value=-5.0, name="src"))
        sat2 = builder2.add(library.Saturate(limit=1.5, mode="smooth"))
        builder2.connect(src2.output_ports[0], sat2.input_ports[0])
        diagram2 = builder2.build()
        ctx2 = diagram2.create_context()
        results2 = jaxonomy.simulate(
            diagram2, ctx2, (0.0, 0.05),
            recorded_signals={"y": sat2.output_ports[0]},
        )
        y_neg = float(np.asarray(results2.outputs["y"])[0])
        # Symmetric soft-saturation: |y_pos| ≈ |y_neg|.
        assert abs(y_pos + y_neg) < 1e-6, (
            f"Soft-saturation should be symmetric around 0: "
            f"y_pos={y_pos}, y_neg={y_neg}."
        )
        # And within bounds.
        assert abs(y_pos) <= 1.5 + 1e-6
