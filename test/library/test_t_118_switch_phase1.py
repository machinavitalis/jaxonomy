# SPDX-License-Identifier: MIT
"""T-118 phase 1: Switch / MultiPortSwitch (block-diagram-style).

Switch(threshold, criteria) takes (data_a, control, data_b) and outputs
``data_a if criteria(control, threshold) else data_b`` via ``npa.where``
so JAX gradients flow through both data branches.

MultiPortSwitch(n_data_inputs) takes one selector + n data inputs and
selects ``data[clip(round(selector), 0, n-1)]``.

These pin the basic forward path + boundary semantics + differentiability
of the data branches. ``mode="smooth"`` (sigmoid/softmax blend) and
one-based indexing are deferred — see T-118-followup-modes.
(Originally tracked as T-MW-205, renumbered to T-118.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library import Switch, MultiPortSwitch
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _eval_switch(threshold, criteria, data_a, control, data_b):
    """Build a tiny Switch diagram and return the final-step output."""
    a = library.Constant(float(data_a))
    c = library.Constant(float(control))
    b = library.Constant(float(data_b))
    sw = Switch(threshold=threshold, criteria=criteria)

    builder = jaxonomy.DiagramBuilder()
    builder.add(a, c, b, sw)
    builder.connect(a.output_ports[0], sw.input_ports[0])
    builder.connect(c.output_ports[0], sw.input_ports[1])
    builder.connect(b.output_ports[0], sw.input_ports[2])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"y": sw.output_ports[0]},
    )
    return float(np.asarray(results.outputs["y"])[-1])


def test_switch_geq_default_threshold_picks_a_when_control_positive():
    """control=1.0 >= threshold=0.0 → data_a."""
    y = _eval_switch(threshold=0.0, criteria=">=",
                     data_a=10.0, control=1.0, data_b=-7.0)
    np.testing.assert_allclose(y, 10.0)


def test_switch_geq_default_threshold_picks_b_when_control_negative():
    """control=-1.0 not >= threshold=0.0 → data_b."""
    y = _eval_switch(threshold=0.0, criteria=">=",
                     data_a=10.0, control=-1.0, data_b=-7.0)
    np.testing.assert_allclose(y, -7.0)


def test_switch_boundary_geq_inclusive_picks_a():
    """control == threshold with criteria=">=" should pick data_a."""
    y = _eval_switch(threshold=0.0, criteria=">=",
                     data_a=10.0, control=0.0, data_b=-7.0)
    np.testing.assert_allclose(y, 10.0)


def test_switch_boundary_strict_gt_picks_b():
    """control == threshold with criteria=">" should pick data_b."""
    y = _eval_switch(threshold=0.0, criteria=">",
                     data_a=10.0, control=0.0, data_b=-7.0)
    np.testing.assert_allclose(y, -7.0)


def test_switch_nondefault_threshold():
    """Threshold can be nonzero; criteria fires relative to it."""
    y_a = _eval_switch(threshold=5.0, criteria=">=",
                       data_a=1.0, control=5.5, data_b=2.0)
    y_b = _eval_switch(threshold=5.0, criteria=">=",
                       data_a=1.0, control=4.5, data_b=2.0)
    np.testing.assert_allclose(y_a, 1.0)
    np.testing.assert_allclose(y_b, 2.0)


def test_switch_invalid_criteria_raises():
    """Bad criteria string fails fast at construction."""
    with pytest.raises(BlockParameterError):
        Switch(threshold=0.0, criteria="<>")


def test_switch_underlying_op_is_differentiable_through_both_branches():
    """jax.grad through ``where`` flows to both data branches.

    The Switch block is a thin npa.where wrapper, so end-to-end
    differentiability comes for free as long as the underlying op
    gradient flows. The selector's gradient is zero (boolean cast),
    which matches the standard hard-threshold semantics.
    """
    def f(a, control, b):
        # Mirror Switch's _compute_output exactly.
        return jnp.where(control >= 0.0, a, b)

    # control > 0 → output = a, so d/da=1, d/db=0
    g_a, g_ctrl, g_b = jax.grad(f, argnums=(0, 1, 2))(3.0, 1.0, 7.0)
    np.testing.assert_allclose(float(g_a), 1.0)
    np.testing.assert_allclose(float(g_b), 0.0)
    np.testing.assert_allclose(float(g_ctrl), 0.0)

    # control < 0 → output = b, so d/da=0, d/db=1
    g_a2, g_ctrl2, g_b2 = jax.grad(f, argnums=(0, 1, 2))(3.0, -1.0, 7.0)
    np.testing.assert_allclose(float(g_a2), 0.0)
    np.testing.assert_allclose(float(g_b2), 1.0)
    np.testing.assert_allclose(float(g_ctrl2), 0.0)


def _eval_multiport(n, selector, data_values):
    """Build a tiny MultiPortSwitch diagram and return the final output."""
    sel = library.Constant(float(selector))
    data_blocks = [library.Constant(float(v)) for v in data_values]
    mps = MultiPortSwitch(n)

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


@pytest.mark.parametrize("selector,expected", [(0, 10.0), (1, 20.0), (2, 30.0)])
def test_multiport_switch_picks_correct_branch(selector, expected):
    """MultiPortSwitch(3): selector=k picks data_k (zero-based)."""
    y = _eval_multiport(3, selector, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(y, expected)


def test_multiport_switch_clips_out_of_range_selector():
    """selector >= n is clipped to n-1; selector < 0 is clipped to 0."""
    y_high = _eval_multiport(3, selector=99, data_values=[10.0, 20.0, 30.0])
    y_low = _eval_multiport(3, selector=-5, data_values=[10.0, 20.0, 30.0])
    np.testing.assert_allclose(y_high, 30.0)
    np.testing.assert_allclose(y_low, 10.0)


def test_multiport_switch_rounds_float_selector():
    """selector=1.4 rounds to 1; selector=1.6 rounds to 2."""
    y_1 = _eval_multiport(3, selector=1.4, data_values=[10.0, 20.0, 30.0])
    y_2 = _eval_multiport(3, selector=1.6, data_values=[10.0, 20.0, 30.0])
    np.testing.assert_allclose(y_1, 20.0)
    np.testing.assert_allclose(y_2, 30.0)


def test_multiport_switch_invalid_n_raises():
    """n_data_inputs < 1 fails fast."""
    with pytest.raises(BlockParameterError):
        MultiPortSwitch(0)


def test_multiport_switch_underlying_op_is_differentiable_through_selected():
    """Gradient through MultiPortSwitch's stack+index op flows only to the selected input."""
    def f(selector, *data):
        stacked = jnp.stack(data, axis=0)
        idx = jnp.clip(jnp.round(selector).astype(jnp.int32), 0, len(data) - 1)
        return stacked[idx]

    # selector=1 → output = data[1], so d/d(data[1]) = 1, others = 0.
    grads = jax.grad(f, argnums=(1, 2, 3))(1.0, 10.0, 20.0, 30.0)
    np.testing.assert_allclose(float(grads[0]), 0.0)
    np.testing.assert_allclose(float(grads[1]), 1.0)
    np.testing.assert_allclose(float(grads[2]), 0.0)
