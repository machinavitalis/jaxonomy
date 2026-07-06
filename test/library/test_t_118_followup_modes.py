# SPDX-License-Identifier: MIT
"""T-118-followup-modes: Switch ``mode="where"`` (default) and ``mode="smooth"``.

Phase 1 shipped only the npa.where-based path. This follow-up adds
``mode="smooth"``: a sigmoid-blended Switch that lets gradients flow
through the *threshold itself* (the killer feature the hard ``where``
zeroes out).

These tests pin:
  * Default (``mode="where"``) is byte-equivalent to phase 1.
  * Smooth mode tracks the hard answer in the strict-active region as
    sharpness grows.
  * Smooth mode produces a non-zero gradient w.r.t. ``threshold`` —
    the whole point of the mode.
  * Construction errors fire for invalid mode / unsupported criteria.

``mode="hard"`` (lax.cond) was deferred at the time this file was
written; it was shipped in T-118-followup-cond-mode (2026-05-10) with
its tests in ``test_t_118_followup_cond_mode.py``. The vmap
incompatibility caveat still holds — that's why it's not the default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library import Switch
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Diagram helpers (mirror test_t_118_switch_phase1._eval_switch so the
# byte-equivalence test below is a true apples-to-apples comparison).
# ---------------------------------------------------------------------------


def _eval_switch(
    threshold,
    criteria,
    data_a,
    control,
    data_b,
    *,
    mode="where",
    sharpness=10.0,
):
    """Build a tiny Switch diagram and return the final-step output."""
    a = library.Constant(float(data_a))
    c = library.Constant(float(control))
    b = library.Constant(float(data_b))
    sw = Switch(
        threshold=threshold,
        criteria=criteria,
        mode=mode,
        sharpness=sharpness,
    )

    builder = jaxonomy.DiagramBuilder()
    builder.add(a, c, b, sw)
    builder.connect(a.output_ports[0], sw.input_ports[0])
    builder.connect(c.output_ports[0], sw.input_ports[1])
    builder.connect(b.output_ports[0], sw.input_ports[2])
    diagram = builder.build()
    ctx = diagram.create_context()

    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.1),
        recorded_signals={"y": sw.output_ports[0]},
    )
    return float(np.asarray(results.outputs["y"])[-1])


# ---------------------------------------------------------------------------
# Default mode preserves phase-1 byte-equivalence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "threshold,criteria,a,c,b,expected",
    [
        # Same battery as test_t_118_switch_phase1's parametrized cases —
        # phase-1 byte-equivalence on the default path.
        (0.0, ">=", 10.0, 1.0, -7.0, 10.0),
        (0.0, ">=", 10.0, -1.0, -7.0, -7.0),
        (0.0, ">=", 10.0, 0.0, -7.0, 10.0),  # boundary inclusive
        (0.0, ">", 10.0, 0.0, -7.0, -7.0),  # boundary strict
        (5.0, ">=", 1.0, 5.5, 2.0, 1.0),
        (5.0, ">=", 1.0, 4.5, 2.0, 2.0),
    ],
)
def test_default_mode_where_matches_phase1(
    threshold, criteria, a, c, b, expected
):
    """``mode="where"`` (default) returns identical values to phase 1."""
    y_default = _eval_switch(threshold, criteria, a, c, b)
    y_explicit = _eval_switch(threshold, criteria, a, c, b, mode="where")
    np.testing.assert_allclose(y_default, expected)
    np.testing.assert_allclose(y_explicit, expected)


# ---------------------------------------------------------------------------
# Smooth mode tracks the hard answer in the strict-active region.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "criteria,a,c,b",
    [
        # control well above threshold: smooth ~ data_a for >=, ~ data_b for <=
        (">=", 10.0, 5.0, -7.0),
        (">", 10.0, 5.0, -7.0),
        # control well below threshold: smooth ~ data_b for >=, ~ data_a for <=
        (">=", 10.0, -5.0, -7.0),
        ("<=", 10.0, 5.0, -7.0),
        ("<", 10.0, 5.0, -7.0),
        ("<=", 10.0, -5.0, -7.0),
    ],
)
def test_smooth_mode_high_sharpness_matches_where_in_strict_active_region(
    criteria, a, c, b
):
    """At |control - threshold| >> 1/sharpness, smooth ~ where within 1e-3."""
    y_hard = _eval_switch(0.0, criteria, a, c, b, mode="where")
    # Sharpness=200 with |control - threshold|=5 puts us deep in the
    # saturated tail of the sigmoid: alpha differs from {0,1} by
    # exp(-1000) ~ 0, so the blend is pinned to one of the two data
    # branches up to FP rounding.
    y_smooth = _eval_switch(
        0.0, criteria, a, c, b, mode="smooth", sharpness=200.0
    )
    np.testing.assert_allclose(y_smooth, y_hard, atol=1e-3)


# ---------------------------------------------------------------------------
# The marketing wedge: gradient w.r.t. threshold is non-zero in smooth mode.
# ---------------------------------------------------------------------------


def _smooth_op(threshold, control, data_a, data_b, *, sharpness, sign):
    # Mirror Switch.initialize's mode='smooth' _compute_output exactly.
    x = sharpness * sign * (control - threshold)
    alpha = 0.5 * (1.0 + jnp.tanh(x / 2.0))
    return alpha * data_a + (1.0 - alpha) * data_b


def _hard_op(threshold, control, data_a, data_b):
    return jnp.where(control >= threshold, data_a, data_b)


def test_smooth_mode_has_nonzero_gradient_wrt_threshold():
    """The whole point of smooth mode: d output / d threshold != 0.

    The hard ``where`` path has zero gradient w.r.t. threshold (the
    comparison is non-differentiable). The smooth path's sigmoid is
    smooth in threshold, so jax.grad returns a finite non-zero value.
    """
    # Hard path: gradient is exactly zero (boolean cast).
    g_hard = jax.grad(_hard_op, argnums=0)(0.0, 0.5, 10.0, -7.0)
    np.testing.assert_allclose(float(g_hard), 0.0)

    # Smooth path with finite sharpness: gradient is non-zero and finite.
    g_smooth = jax.grad(_smooth_op, argnums=0)(
        0.0, 0.5, 10.0, -7.0, sharpness=2.0, sign=1.0
    )
    g_smooth_f = float(g_smooth)
    assert np.isfinite(g_smooth_f), f"gradient not finite: {g_smooth_f}"
    assert abs(g_smooth_f) > 1e-3, (
        f"expected non-trivial gradient w.r.t. threshold, got {g_smooth_f}"
    )

    # Sanity: gradient sign matches expectation. For criteria='>=' with
    # sign=+1, raising threshold pushes alpha toward 0 (favoring data_b),
    # so output moves from data_a toward data_b. data_a > data_b here, so
    # d output / d threshold should be NEGATIVE.
    assert g_smooth_f < 0.0, (
        f"expected negative gradient (raising threshold reduces alpha), "
        f"got {g_smooth_f}"
    )


def test_smooth_mode_has_finite_gradient_at_threshold_boundary():
    """Gradient at control == threshold is exactly the sigmoid's peak slope.

    At x=0, d/dx sigmoid(k*x) = k/4. This is a sharper sanity check
    than the previous test because we know the closed-form value.
    """
    sharpness = 4.0
    # control == threshold, so x=0 in the sigmoid.
    g = jax.grad(_smooth_op, argnums=0)(
        0.0, 0.0, 10.0, -7.0, sharpness=sharpness, sign=1.0
    )
    # d alpha / d threshold at x=0 is -sharpness/4 (sign flip from chain
    # rule on (control - threshold)). Then d output / d threshold =
    # (data_a - data_b) * d alpha / d threshold.
    expected = (10.0 - (-7.0)) * (-sharpness / 4.0)
    np.testing.assert_allclose(float(g), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Smooth-mode end-to-end through the diagram (not just the underlying op).
# ---------------------------------------------------------------------------


def test_smooth_mode_blends_at_low_sharpness():
    """At low sharpness, output is a true blend, not a hard pick."""
    # control == threshold, sharpness=1.0 → alpha = sigmoid(0) = 0.5,
    # output = 0.5*a + 0.5*b.
    y = _eval_switch(0.0, ">=", 10.0, 0.0, -2.0, mode="smooth", sharpness=1.0)
    np.testing.assert_allclose(y, 0.5 * 10.0 + 0.5 * -2.0, atol=1e-9)


def test_smooth_mode_respects_lt_criteria_sign_flip():
    """For criteria='<', control above threshold should push toward data_b."""
    # criteria='<' with control >> threshold: hard answer is data_b.
    y_hard = _eval_switch(0.0, "<", 10.0, 5.0, -7.0, mode="where")
    np.testing.assert_allclose(y_hard, -7.0)
    # Smooth with high sharpness should agree.
    y_smooth = _eval_switch(
        0.0, "<", 10.0, 5.0, -7.0, mode="smooth", sharpness=50.0
    )
    np.testing.assert_allclose(y_smooth, -7.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Construction-time validation.
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    """Unknown mode strings fail fast.

    Note: ``mode="hard"`` was deferred in T-118-followup-modes but
    shipped in T-118-followup-cond-mode (2026-05-10), so it is now a
    valid mode and no longer surfaces in this error path.
    """
    with pytest.raises(BlockParameterError):
        Switch(threshold=0.0, criteria=">=", mode="bogus")

    # Sanity: mode='hard' now constructs without raising.
    sw = Switch(threshold=0.0, criteria=">=", mode="hard")
    assert sw is not None


@pytest.mark.parametrize("bad_criteria", ["==", "!="])
def test_smooth_mode_rejects_equality_criteria(bad_criteria):
    """No sigmoid approximation for == / != — fail loudly at construction."""
    with pytest.raises(BlockParameterError):
        Switch(threshold=0.0, criteria=bad_criteria, mode="smooth")


@pytest.mark.parametrize("bad_sharpness", [0.0, -1.0, float("inf"), float("nan")])
def test_smooth_mode_rejects_nonpositive_or_nonfinite_sharpness(bad_sharpness):
    """sharpness must be finite and > 0 in smooth mode."""
    with pytest.raises(BlockParameterError):
        Switch(threshold=0.0, criteria=">=", mode="smooth", sharpness=bad_sharpness)


def test_where_mode_accepts_equality_and_ignores_sharpness():
    """Default mode never touches sharpness, so any value is fine."""
    # Construction succeeds despite sharpness=0 because mode='where'
    # never validates it (and the closure never reads it).
    sw = Switch(
        threshold=0.0, criteria="==", mode="where", sharpness=0.0
    )
    assert sw is not None
