# SPDX-License-Identifier: MIT
"""T-120-followup-counter-block — discrete ``Counter`` block.

Covers the rising-edge counter block deferred from T-120. The block
counts rising edges on a boolean / binary trigger input, with optional
``max_count`` saturation or wrap-around. Tests assert:

- Five rising edges on a Pulse trigger drive the count from 0 to 5.
- ``max_count=3, reset_on_max=True`` wraps the count back to 0 after
  the increment that hits the cap.
- ``max_count=3, reset_on_max=False`` saturates the count at 3.
- ``initial_count`` is honoured at ``t=0`` and propagates through the
  output port before any update fires.
- Construction-time validation: bad ``max_count`` raises.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library import Counter
from jaxonomy.testing.markers import skip_if_not_jax


skip_if_not_jax()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SquareWave(jaxonomy.LeafSystem):
    """Boolean square wave driven by an internal discrete state.

    Toggles between False and True every ``dt`` seconds. Gives a clean,
    sample-aligned alternation pattern (False, True, False, True, ...)
    that produces exactly one rising edge per ``2 * dt`` window when
    consumed by a discrete block sampling at the same ``dt``.

    We use this rather than :class:`Pulse` because Pulse's modulo-based
    output is sampled from a continuous time variable, and at exact
    sample-boundary times the floating-point ``time % period`` does not
    map cleanly to a single rising edge per cycle.
    """

    def __init__(self, dt, name=None):
        super().__init__(name=name)
        self._dt = dt
        self._upd_idx = self.declare_periodic_update()
        # Start LOW so the first transition is a clean rising edge.
        self.declare_discrete_state(default_value=jnp.array(False))
        self.configure_periodic_update(
            self._upd_idx,
            self._update,
            period=dt,
            offset=0.0,
        )
        self.declare_output_port(
            self._output,
            prerequisites_of_calc=[],
            requires_inputs=False,
            default_value=jnp.array(False),
        )

    def _update(self, _time, state, **_params):
        return jnp.logical_not(state.discrete_state)

    def _output(self, _time, state, *_inputs, **_params):
        return state.discrete_state


def _build_counter_with_square_trigger(
    *,
    initial_count=0,
    increment=1,
    max_count=None,
    reset_on_max=False,
    dt=0.1,
):
    """Diagram: ``_SquareWave`` -> ``Counter``. The square wave produces
    one rising edge per ``2 * dt`` so simulating for ``10 * dt`` yields
    exactly 5 rising edges."""
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_SquareWave(dt=dt, name="trig"))
    counter = bld.add(
        Counter(
            dt=dt,
            initial_count=initial_count,
            increment=increment,
            max_count=max_count,
            reset_on_max=reset_on_max,
            name="counter",
        )
    )
    bld.connect(trig.output_ports[0], counter.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    return diagram, ctx, counter


# Backwards-compatibility alias used by tests below.
_build_pulse_driven_counter = _build_counter_with_square_trigger


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_counter_construction_smoke():
    """Bare construction with defaults must not raise."""
    Counter(dt=0.1)


def test_counter_invalid_max_count_raises():
    """``max_count <= 0`` is rejected at ``initialize`` time."""
    with pytest.raises(BlockParameterError):
        blk = Counter(dt=0.1, max_count=0)
        blk.initialize(initial_count=0, dt=0.1, increment=1, max_count=0,
                       reset_on_max=False)


# ---------------------------------------------------------------------------
# Initial count
# ---------------------------------------------------------------------------


def test_counter_initial_count_propagates_to_output():
    """Before any rising edge, the output equals ``initial_count``."""
    diagram, ctx, c = _build_pulse_driven_counter(initial_count=7)
    y = c.output_ports[0].eval(ctx)
    assert int(np.asarray(y)) == 7


def test_counter_initial_count_default_is_zero():
    diagram, ctx, c = _build_pulse_driven_counter()
    y = c.output_ports[0].eval(ctx)
    assert int(np.asarray(y)) == 0


# ---------------------------------------------------------------------------
# Five rising edges -> count = 5
# ---------------------------------------------------------------------------


def test_counter_five_rising_edges():
    """Square wave at dt=0.1 produces one rising edge per 0.2s window.
    Simulate for 1.05s -> 5 rising edges -> count = 5."""
    diagram, ctx, c = _build_pulse_driven_counter(dt=0.1)
    recorded = {"count": c.output_ports[0]}
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 1.05), recorded_signals=recorded,
    )
    final_count = int(np.asarray(res.outputs["count"][-1]))
    assert final_count == 5, (
        f"expected 5 rising edges -> count=5, got {final_count} "
        f"(full trace: {np.asarray(res.outputs['count']).tolist()})"
    )


# ---------------------------------------------------------------------------
# max_count behaviour: wrap vs saturate
# ---------------------------------------------------------------------------


def test_counter_wraps_when_reset_on_max_true():
    """With ``max_count=3, reset_on_max=True``: after 3 increments the
    count wraps to 0. A 4th increment then takes it to 1, a 5th to 2."""
    diagram, ctx, c = _build_pulse_driven_counter(
        max_count=3, reset_on_max=True,
    )
    recorded = {"count": c.output_ports[0]}
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 1.05), recorded_signals=recorded,
    )
    counts = np.asarray(res.outputs["count"])
    # After 5 rising edges with wrap-at-3 semantics: 1, 2, 0 (wrap),
    # 1, 2. Final count must equal 2.
    assert int(counts[-1]) == 2, (
        f"expected wrap to land on 2 after 5 edges with cap=3, "
        f"got {int(counts[-1])} (full trace: {counts.tolist()})"
    )
    # Sanity: the recorded trace must hit 0 *after* having gone above
    # zero — i.e. the wrap actually fired.
    assert (counts > 0).any() and (counts == 0).sum() >= 1


def test_counter_saturates_when_reset_on_max_false():
    """With ``max_count=3, reset_on_max=False``: the count climbs to 3
    and then clamps. After 5 rising edges it must still be 3."""
    diagram, ctx, c = _build_pulse_driven_counter(
        max_count=3, reset_on_max=False,
    )
    recorded = {"count": c.output_ports[0]}
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 1.05), recorded_signals=recorded,
    )
    counts = np.asarray(res.outputs["count"])
    assert int(counts[-1]) == 3, (
        f"expected saturation at 3, got {int(counts[-1])} "
        f"(full trace: {counts.tolist()})"
    )
    # The count never exceeds the cap.
    assert int(counts.max()) == 3


# ---------------------------------------------------------------------------
# increment != 1
# ---------------------------------------------------------------------------


def test_counter_custom_increment():
    """``increment=2`` adds 2 per rising edge. 5 edges -> count = 10."""
    diagram, ctx, c = _build_pulse_driven_counter(increment=2)
    recorded = {"count": c.output_ports[0]}
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 1.05), recorded_signals=recorded,
    )
    final_count = int(np.asarray(res.outputs["count"][-1]))
    assert final_count == 10, (
        f"expected 5 edges * increment=2 = 10, got {final_count}"
    )


# ---------------------------------------------------------------------------
# Output dtype is integer (counter is non-differentiable)
# ---------------------------------------------------------------------------


def test_counter_output_dtype_is_int32():
    """The count is integer-typed; downstream gradient paths through
    the count itself are not expected to carry gradients."""
    diagram, ctx, c = _build_pulse_driven_counter()
    y = c.output_ports[0].eval(ctx)
    arr = np.asarray(y)
    assert arr.dtype == np.int32, f"expected int32, got {arr.dtype}"
