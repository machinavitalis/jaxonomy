# SPDX-License-Identifier: MIT

"""T-126 followup — formal Sobol-style aleatoric vs epistemic decomposition.

Covers :func:`jaxonomy.uq.decompose_variance_sobol`:

1. Linear model ``y = a*x + b`` (a epistemic, x aleatoric, b constant) —
   Sobol-decomposed variances match analytic ANOVA values, sum equals
   ``var_total`` within MC tolerance, interaction ~ 0 (additively
   separable).
2. Pure aleatoric: epistemic component ~ 0 and ~ all variance flows to
   the aleatoric bucket.
3. Pure epistemic: symmetric to case 2.
4. Multiplicative-coupling nonlinear model where the linearised T-126
   phase 1 :func:`decompose_variance` is biased: the formal Sobol path
   correctly attributes a non-trivial chunk of the variance to the
   ``interaction`` term while the phase 1 residual silently absorbs it.
5. Input-validation errors (overlapping group names, both groups empty,
   non-positive ``n_samples``, wrong qoi shape).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Normal,
    Uniform,
    decompose_variance,
    decompose_variance_sobol,
)

skip_if_not_jax()
pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# 1. Linear model — Sobol decomposition matches analytic ANOVA.
# ---------------------------------------------------------------------------

def test_sobol_decomposition_linear_model_matches_analytic():
    """``y = a*x + b`` with ``a`` epistemic, ``x`` aleatoric, ``b`` aleatoric.

    Analytical first-order grouped variances (independent, additively
    separable in (a) and (x, b)):

        var_aleatoric  = mu_a^2 * Var(x) + Var(b)
        var_epistemic  = mu_x^2 * Var(a)
        var_total      = var_aleatoric + var_epistemic        (no interaction beyond
                                                                the a*x cross term,
                                                                which adds a small
                                                                Var(a)*Var(x) to the
                                                                interaction bucket)
    """
    a_mean, a_std = 2.0, 0.05
    x_mean, x_std = 5.0, 0.10
    b_mean, b_std = 0.0, 0.02

    a_dist = Normal(a_mean, a_std, kind="epistemic")
    x_dist = Normal(x_mean, x_std, kind="aleatoric")
    b_dist = Normal(b_mean, b_std, kind="aleatoric")

    def qoi(p):
        return p["a"] * p["x"] + p["b"]

    out = decompose_variance_sobol(
        qoi,
        aleatoric_dists={"x": x_dist, "b": b_dist},
        epistemic_dists={"a": a_dist},
        n_samples=4096,
        key=jax.random.PRNGKey(0),
    )

    expected_alea = (a_mean ** 2) * (x_std ** 2) + b_std ** 2
    expected_epi = (x_mean ** 2) * (a_std ** 2)
    # Cross-term Var(a)*Var(x) ~ 0.05^2 * 0.10^2 = 2.5e-5 — tiny vs ~0.06 total.
    expected_total = expected_alea + expected_epi

    assert out["var_total"] == pytest.approx(expected_total, rel=0.10)
    assert out["var_aleatoric"] == pytest.approx(expected_alea, rel=0.15)
    assert out["var_epistemic"] == pytest.approx(expected_epi, rel=0.15)
    # Sum should equal total within MC noise (interaction near zero).
    assert (out["var_aleatoric"] + out["var_epistemic"] + out["interaction"]) == pytest.approx(
        out["var_total"], rel=1e-6
    )
    # Interaction is small relative to total for this near-additive QoI.
    assert abs(out["interaction"]) < 0.05 * out["var_total"]


def test_sobol_decomposition_agrees_with_phase1_on_linear_model():
    """On a linear QoI both estimators should agree to within MC noise."""
    a_mean, a_std = 2.0, 0.05
    x_mean, x_std = 5.0, 0.10
    b_mean, b_std = 0.0, 0.02

    a_dist = Normal(a_mean, a_std, kind="epistemic")
    x_dist = Normal(x_mean, x_std, kind="aleatoric")
    b_dist = Normal(b_mean, b_std, kind="aleatoric")

    def qoi(p):
        return p["a"] * p["x"] + p["b"]

    sobol_out = decompose_variance_sobol(
        qoi,
        aleatoric_dists={"x": x_dist, "b": b_dist},
        epistemic_dists={"a": a_dist},
        n_samples=4096,
        key=jax.random.PRNGKey(1),
    )

    # Phase 1 reference — IID sample then call the Taylor-style estimator.
    n = 16384
    key = jax.random.PRNGKey(2)
    ka, kx, kb = jax.random.split(key, 3)
    a = a_dist.sample(ka, (n,))
    x = x_dist.sample(kx, (n,))
    b = b_dist.sample(kb, (n,))
    y = a * x + b
    phase1_out = decompose_variance(
        y,
        {"a": a, "x": x, "b": b},
        {"a": "epistemic", "x": "aleatoric", "b": "aleatoric"},
    )

    assert sobol_out["var_total"] == pytest.approx(phase1_out["var_total"], rel=0.10)
    assert sobol_out["var_aleatoric"] == pytest.approx(
        phase1_out["var_aleatoric"], rel=0.20
    )
    assert sobol_out["var_epistemic"] == pytest.approx(
        phase1_out["var_epistemic"], rel=0.20
    )


# ---------------------------------------------------------------------------
# 2 + 3. Pure aleatoric / pure epistemic edge cases.
# ---------------------------------------------------------------------------

def test_sobol_decomposition_pure_aleatoric():
    """No epistemic group — epistemic bucket reports 0, aleatoric ~ total."""
    x_dist = Normal(0.0, 1.0, kind="aleatoric")
    b_dist = Normal(0.0, 0.5, kind="aleatoric")

    def qoi(p):
        return 2.0 * p["x"] + p["b"]

    out = decompose_variance_sobol(
        qoi,
        aleatoric_dists={"x": x_dist, "b": b_dist},
        epistemic_dists={},
        n_samples=4096,
        key=jax.random.PRNGKey(3),
    )
    assert out["var_epistemic"] == 0.0
    assert out["var_aleatoric"] == pytest.approx(out["var_total"], rel=0.05)
    # Interaction = total - aleatoric - 0 ≈ 0 (within MC).
    assert abs(out["interaction"]) < 0.05 * out["var_total"]


def test_sobol_decomposition_pure_epistemic():
    """Symmetric to the pure-aleatoric case."""
    a_dist = Normal(1.0, 0.2, kind="epistemic")
    c_dist = Normal(3.0, 0.1, kind="epistemic")

    def qoi(p):
        return p["a"] + p["c"]

    out = decompose_variance_sobol(
        qoi,
        aleatoric_dists={},
        epistemic_dists={"a": a_dist, "c": c_dist},
        n_samples=4096,
        key=jax.random.PRNGKey(4),
    )
    assert out["var_aleatoric"] == 0.0
    assert out["var_epistemic"] == pytest.approx(out["var_total"], rel=0.05)
    assert abs(out["interaction"]) < 0.05 * out["var_total"]


# ---------------------------------------------------------------------------
# 4. Strongly nonlinear / interaction-rich QoI where Sobol > phase-1 Taylor.
# ---------------------------------------------------------------------------

def test_sobol_decomposition_captures_interaction_phase1_misses():
    """``y = a * x`` with broad uniform priors on both sides.

    Closed form (independent uniforms on [-1, 1]): the QoI is symmetric so
    ``E[Y] = 0``, ``E[Y^2] = E[a^2]*E[x^2] = 1/9``, hence ``Var(Y) = 1/9``.

    The first-order grouped Sobol indices vanish (``E_a[a*x] = 0`` for any
    fixed ``x``, and vice versa) so the ANOVA puts *all* of the variance
    into the interaction term. The phase-1 Taylor-style estimator assumes
    smooth small-noise expansion and silently parks the missed variance in
    its ``residual`` field instead.
    """
    a_dist = Uniform(-1.0, 1.0, kind="epistemic")
    x_dist = Uniform(-1.0, 1.0, kind="aleatoric")

    def qoi(p):
        return p["a"] * p["x"]

    out = decompose_variance_sobol(
        qoi,
        aleatoric_dists={"x": x_dist},
        epistemic_dists={"a": a_dist},
        n_samples=8192,
        key=jax.random.PRNGKey(5),
    )

    # Total variance ≈ 1/9.
    assert out["var_total"] == pytest.approx(1.0 / 9.0, rel=0.10)
    # Both first-order grouped indices ≈ 0 by symmetry.
    assert out["var_aleatoric"] < 0.05 * out["var_total"]
    assert out["var_epistemic"] < 0.05 * out["var_total"]
    # Interaction explains essentially all the variance.
    assert out["interaction"] == pytest.approx(out["var_total"], rel=0.10)

    # Phase-1 reference on the same problem: it should mis-estimate by a
    # large margin (residual carries most of the variance).
    n = 16384
    key = jax.random.PRNGKey(6)
    ka, kx = jax.random.split(key, 2)
    a_samples = a_dist.sample(ka, (n,))
    x_samples = x_dist.sample(kx, (n,))
    y_samples = a_samples * x_samples
    phase1 = decompose_variance(
        y_samples,
        {"a": a_samples, "x": x_samples},
        {"a": "epistemic", "x": "aleatoric"},
    )
    # Phase-1 var_aleatoric and var_epistemic both ~ 0; residual carries it.
    assert abs(phase1["residual"]) > 0.5 * phase1["var_total"]


# ---------------------------------------------------------------------------
# 5. Input validation.
# ---------------------------------------------------------------------------

def test_sobol_decomposition_rejects_overlapping_groups():
    a = Normal(0.0, 1.0, kind="epistemic")
    x = Normal(0.0, 1.0, kind="aleatoric")
    with pytest.raises(ValueError, match="appear in both"):
        decompose_variance_sobol(
            lambda p: p["a"] + p["x"],
            aleatoric_dists={"a": a, "x": x},
            epistemic_dists={"a": a},  # overlap
            n_samples=64,
        )


def test_sobol_decomposition_rejects_both_groups_empty():
    with pytest.raises(ValueError, match="non-empty"):
        decompose_variance_sobol(
            lambda p: jnp.zeros(64),
            aleatoric_dists={},
            epistemic_dists={},
            n_samples=64,
        )


def test_sobol_decomposition_rejects_bad_n_samples():
    with pytest.raises(ValueError, match="n_samples"):
        decompose_variance_sobol(
            lambda p: p["a"],
            aleatoric_dists={"a": Normal(0.0, 1.0)},
            epistemic_dists={},
            n_samples=0,
        )


def test_sobol_decomposition_rejects_wrong_qoi_shape():
    a = Normal(0.0, 1.0, kind="epistemic")
    x = Normal(0.0, 1.0, kind="aleatoric")
    # qoi returns wrong length.
    with pytest.raises(ValueError, match="expected"):
        decompose_variance_sobol(
            lambda p: jnp.zeros(7),
            aleatoric_dists={"x": x},
            epistemic_dists={"a": a},
            n_samples=64,
        )
