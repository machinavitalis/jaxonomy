# SPDX-License-Identifier: MIT

"""V-013: degenerate and unusual `t_span` edge cases.

The CHANGELOG-era feature work (int_time_scale="auto", buffer auto-sizing,
long-horizon support) is tested elsewhere; what has no coverage is the
*degenerate* end of `t_span`: zero-length spans, spans that do not start at
zero, reversed spans, and very short spans. These tests pin the engine's
current behaviour so a refactor of the major-step loop cannot silently
change it.

Behaviour verified empirically on 2026-07-09:
- zero-length span: returns a single sample at t0, state untouched.
- reversed span (t0 > tf): silently no-ops — the simulation ends immediately
  at t0. No error is raised. (A validation error would arguably be better;
  if that is ever added, update `test_reversed_span_is_silent_noop` to
  assert the raise instead.)
- nonzero / negative t0: integrates the correct duration, `results.time`
  starts at t0.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Constant, Integrator

pytestmark = pytest.mark.minimal


def _unit_ramp_diagram():
    """Constant(1.0) -> Integrator: x(t) = x0 + (t - t0)."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(1.0))
    integ = builder.add(Integrator(initial_state=0.0))
    builder.connect(src.output_ports[0], integ.input_ports[0])
    diagram = builder.build()
    return diagram, integ


class TestZeroLengthSpan:
    def test_returns_single_sample_at_t0(self):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.0), recorded_signals={"x": integ.output_ports[0]}
        )
        assert results.time.shape == (1,)
        assert float(results.time[0]) == 0.0

    def test_state_is_untouched(self):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(diagram, ctx, (0.0, 0.0))
        xf = results.context[integ.system_id].continuous_state
        assert float(xf) == 0.0

    def test_zero_length_at_nonzero_t0(self):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (3.0, 3.0), recorded_signals={"x": integ.output_ports[0]}
        )
        assert float(results.time[0]) == 3.0
        assert float(results.context[integ.system_id].continuous_state) == 0.0


class TestNonzeroStart:
    def test_integrates_correct_duration_from_positive_t0(self):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram,
            ctx,
            (5.0, 6.0),
            recorded_signals={"x": integ.output_ports[0]},
        )
        assert float(results.time[0]) == pytest.approx(5.0)
        assert float(results.time[-1]) == pytest.approx(6.0)
        # unit ramp over a 1s window integrates to exactly 1.0
        assert float(results.outputs["x"][-1]) == pytest.approx(1.0, abs=1e-8)


class TestNegativeTime:
    """Negative *start* times work; a span that ENDS below zero silently
    no-ops (verified 2026-07-09: t_span=(-2,-1) returns a single sample at
    t0 with the state untouched — same immediate-stop signature as the
    reversed span, presumably the internal integer-time stop condition).
    These tests pin the boundary; if tf<0 support (or a validation error)
    is ever added, update `test_fully_negative_span_is_silent_noop`."""

    @pytest.mark.parametrize("t_span", [(-2.0, 1.0), (-0.5, 0.5)])
    def test_span_crossing_zero_integrates_full_duration(self, t_span):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, t_span, recorded_signals={"x": integ.output_ports[0]}
        )
        duration = t_span[1] - t_span[0]
        assert float(results.time[-1]) == pytest.approx(t_span[1])
        assert float(results.outputs["x"][-1]) == pytest.approx(duration, rel=1e-8)

    def test_fully_negative_span_is_silent_noop(self):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (-2.0, -1.0), recorded_signals={"x": integ.output_ports[0]}
        )
        assert len(results.time) == 1
        assert float(results.time[-1]) == pytest.approx(-2.0)
        assert float(results.context[integ.system_id].continuous_state) == 0.0


class TestReversedSpan:
    def test_reversed_span_is_silent_noop(self):
        """t0 > tf currently ends the simulation immediately at t0 without
        raising. This test pins that behaviour; if input validation is ever
        added (the friendlier outcome), flip this to pytest.raises."""
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(diagram, ctx, (1.0, 0.0))
        assert float(results.context.time) == pytest.approx(1.0)
        # no backwards integration happened
        assert float(results.context[integ.system_id].continuous_state) == 0.0


class TestTinySpan:
    def test_nanosecond_span_completes_and_is_accurate(self):
        diagram, integ = _unit_ramp_diagram()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 1e-9), recorded_signals={"x": integ.output_ports[0]}
        )
        assert float(results.time[-1]) == pytest.approx(1e-9, rel=1e-6)
        assert float(results.outputs["x"][-1]) == pytest.approx(1e-9, rel=1e-6)
