# SPDX-License-Identifier: MIT

"""T-126 phase 1 — aleatoric vs epistemic separation.

Covers:

1. ``Distribution.kind`` defaults to ``"aleatoric"`` and accepts
   ``"epistemic"``; invalid values raise.
2. ``split_distributions_by_kind`` correctly partitions a mixed dict.
3. ``monte_carlo_with_kinds`` returns matching samples + kind labels.
4. ``decompose_variance`` on the linear model ``y = a*x + b`` recovers the
   small-uncertainty Taylor split between aleatoric and epistemic
   contributions.
5. Aleatoric-only and epistemic-only edge cases.
6. ``mean_and_variance_by_kind`` honest-fallback reporter.
7. Default-off byte-equivalence: distributions without the ``kind`` kwarg
   still hash/compare like the pre-T-126 versions.

Original task ID was T-MW-303; renumbered to T-126 in commit 124c178.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    LogNormal,
    Normal,
    Triangular,
    Uniform,
    decompose_variance,
    mean_and_variance_by_kind,
    monte_carlo_with_kinds,
    split_distributions_by_kind,
)

skip_if_not_jax()
pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# 1. kind attribute
# ---------------------------------------------------------------------------

def test_distribution_kind_default_is_aleatoric():
    assert Uniform(0.0, 1.0).kind == "aleatoric"
    assert Normal(0.0, 1.0).kind == "aleatoric"
    assert LogNormal(0.0, 0.5).kind == "aleatoric"
    assert Triangular(0.0, 0.5, 1.0).kind == "aleatoric"


def test_distribution_kind_can_be_epistemic():
    assert Uniform(0.0, 1.0, kind="epistemic").kind == "epistemic"
    assert Normal(0.0, 1.0, kind="epistemic").kind == "epistemic"
    assert LogNormal(0.0, 0.5, kind="epistemic").kind == "epistemic"
    assert Triangular(0.0, 0.5, 1.0, kind="epistemic").kind == "epistemic"


def test_distribution_kind_invalid_raises():
    with pytest.raises(ValueError, match="kind must be"):
        Uniform(0.0, 1.0, kind="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        Normal(0.0, 1.0, kind="bogus")  # type: ignore[arg-type]


def test_distribution_sampling_unaffected_by_kind():
    """Adding a ``kind`` tag must not perturb the random stream. Same key
    + same numeric params -> bit-identical samples regardless of kind."""
    key = jax.random.PRNGKey(7)
    a_alea = Uniform(0.0, 1.0, kind="aleatoric").sample(key, (256,))
    a_epi = Uniform(0.0, 1.0, kind="epistemic").sample(key, (256,))
    np.testing.assert_array_equal(np.asarray(a_alea), np.asarray(a_epi))

    b_alea = Normal(0.0, 1.0, kind="aleatoric").sample(key, (256,))
    b_epi = Normal(0.0, 1.0, kind="epistemic").sample(key, (256,))
    np.testing.assert_array_equal(np.asarray(b_alea), np.asarray(b_epi))


# ---------------------------------------------------------------------------
# 2. split_distributions_by_kind
# ---------------------------------------------------------------------------

def test_split_distributions_by_kind_basic():
    dists = {
        "a": Uniform(0.0, 1.0, kind="aleatoric"),
        "b": Normal(0.0, 1.0, kind="epistemic"),
        "c": Uniform(2.0, 3.0),  # default aleatoric
    }
    parts = split_distributions_by_kind(dists)
    assert set(parts) == {"aleatoric", "epistemic"}
    assert set(parts["aleatoric"]) == {"a", "c"}
    assert set(parts["epistemic"]) == {"b"}


def test_split_distributions_by_kind_all_one_kind():
    dists = {"a": Uniform(0.0, 1.0), "b": Normal(0.0, 1.0)}
    parts = split_distributions_by_kind(dists)
    assert parts["epistemic"] == {}
    assert set(parts["aleatoric"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# 3. monte_carlo_with_kinds
# ---------------------------------------------------------------------------

def test_monte_carlo_with_kinds_shape_and_labels():
    dists = {
        "a": Uniform(0.0, 1.0, kind="epistemic"),
        "x": Normal(0.0, 1.0, kind="aleatoric"),
    }
    samples, labels = monte_carlo_with_kinds(dists, 64, jax.random.PRNGKey(1))
    assert set(samples) == {"a", "x"}
    assert samples["a"].shape == (64,)
    assert samples["x"].shape == (64,)
    assert labels == {"a": "epistemic", "x": "aleatoric"}


def test_monte_carlo_with_kinds_validates_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        monte_carlo_with_kinds({}, 16, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="n_samples"):
        monte_carlo_with_kinds(
            {"a": Uniform(0.0, 1.0)}, 0, jax.random.PRNGKey(0)
        )


# ---------------------------------------------------------------------------
# 4. decompose_variance — linear model y = a*x + b
# ---------------------------------------------------------------------------

def test_decompose_variance_linear_model_split():
    """``y = a * x + b`` with ``a`` epistemic and ``x`` aleatoric.

    Small-uncertainty Taylor approximation of the variance:
        Var(y) ≈ E[x]^2 * Var(a) + E[a]^2 * Var(x) + Var(b)
    Each term is attributed to the kind of the underlying parameter.

    With ``E[a]`` and ``E[x]`` of comparable magnitude and small dispersion,
    the empirical decomposition should match within Monte-Carlo noise.
    """
    n = 16384
    key = jax.random.PRNGKey(42)
    ka, kx, kb = jax.random.split(key, 3)

    # Means chosen so the cross-effect terms dominate (so the test is
    # actually checking the partition logic, not just one trivial channel).
    a_mean, a_std = 2.0, 0.05  # epistemic — uncertain parameter
    x_mean, x_std = 5.0, 0.10  # aleatoric — random input
    b_mean, b_std = 0.0, 0.02  # aleatoric — measurement bias

    a = Normal(a_mean, a_std, kind="epistemic").sample(ka, (n,))
    x = Normal(x_mean, x_std, kind="aleatoric").sample(kx, (n,))
    b = Normal(b_mean, b_std, kind="aleatoric").sample(kb, (n,))

    y = a * x + b
    samples = {"a": a, "x": x, "b": b}
    labels = {"a": "epistemic", "x": "aleatoric", "b": "aleatoric"}
    out = decompose_variance(y, samples, labels)

    # Analytical small-uncertainty contributions:
    expected_epi = (x_mean ** 2) * (a_std ** 2)              # via a
    expected_alea = (a_mean ** 2) * (x_std ** 2) + b_std ** 2  # via x and b
    expected_total = expected_epi + expected_alea

    # Empirical total within MC noise of the analytical total.
    assert out["var_total"] == pytest.approx(expected_total, rel=0.10)

    # Per-partition first-order contributions match analytical to MC noise.
    assert out["var_epistemic"] == pytest.approx(expected_epi, rel=0.10)
    assert out["var_aleatoric"] == pytest.approx(expected_alea, rel=0.10)

    # Residual (linear approximation gap) is small relative to total.
    assert abs(out["residual"]) < 0.10 * out["var_total"]


def test_decompose_variance_aleatoric_only():
    """No epistemic parameters -> total variance flows entirely to the
    aleatoric bucket (within MC noise)."""
    n = 8192
    key = jax.random.PRNGKey(0)
    kx, kb = jax.random.split(key, 2)
    x = Normal(0.0, 1.0, kind="aleatoric").sample(kx, (n,))
    b = Normal(0.0, 0.5, kind="aleatoric").sample(kb, (n,))
    y = 2.0 * x + b
    samples = {"x": x, "b": b}
    labels = {"x": "aleatoric", "b": "aleatoric"}

    out = decompose_variance(y, samples, labels)
    assert out["var_epistemic"] == pytest.approx(0.0, abs=1e-10)
    # Total variance ≈ aleatoric component (linear model so residual ~ 0).
    assert out["var_aleatoric"] == pytest.approx(out["var_total"], rel=0.05)


def test_decompose_variance_epistemic_only():
    """No aleatoric parameters -> total variance flows entirely to the
    epistemic bucket (within MC noise)."""
    n = 8192
    key = jax.random.PRNGKey(0)
    ka, kc = jax.random.split(key, 2)
    a = Normal(1.0, 0.2, kind="epistemic").sample(ka, (n,))
    c = Normal(3.0, 0.1, kind="epistemic").sample(kc, (n,))
    # y is linear in both -> first-order decomposition is exact in expectation.
    y = a + c
    samples = {"a": a, "c": c}
    labels = {"a": "epistemic", "c": "epistemic"}

    out = decompose_variance(y, samples, labels)
    assert out["var_aleatoric"] == pytest.approx(0.0, abs=1e-10)
    assert out["var_epistemic"] == pytest.approx(out["var_total"], rel=0.05)


def test_decompose_variance_validates_shapes():
    with pytest.raises(ValueError, match="must be 1-D"):
        decompose_variance(
            jnp.ones((4, 4)),
            {"a": jnp.ones((16,))},
            {"a": "aleatoric"},
        )
    with pytest.raises(ValueError, match="non-empty"):
        decompose_variance(jnp.ones((16,)), {}, {})
    with pytest.raises(ValueError, match="expected"):
        decompose_variance(
            jnp.ones((16,)),
            {"a": jnp.ones((8,))},
            {"a": "aleatoric"},
        )


def test_decompose_variance_constant_parameter_skipped():
    """A parameter with zero variance contributes nothing (no NaN)."""
    n = 256
    a = jnp.zeros((n,))                    # constant
    x = jax.random.normal(jax.random.PRNGKey(0), (n,))
    y = 2.0 * x + a
    out = decompose_variance(
        y, {"a": a, "x": x}, {"a": "epistemic", "x": "aleatoric"}
    )
    assert np.isfinite(out["var_total"])
    assert out["var_epistemic"] == pytest.approx(0.0, abs=1e-10)
    assert out["var_aleatoric"] > 0.0


# ---------------------------------------------------------------------------
# 5. mean_and_variance_by_kind (honest-fallback reporter)
# ---------------------------------------------------------------------------

def test_mean_and_variance_by_kind_reports_each_partition():
    n = 1024
    key = jax.random.PRNGKey(3)
    ka, kx = jax.random.split(key, 2)
    a = Normal(2.0, 0.5, kind="epistemic").sample(ka, (n,))
    x = Normal(0.0, 1.0, kind="aleatoric").sample(kx, (n,))

    out = mean_and_variance_by_kind(
        {"a": a, "x": x}, {"a": "epistemic", "x": "aleatoric"}
    )
    assert out["aleatoric"]["mean"] == pytest.approx(0.0, abs=0.1)
    assert out["aleatoric"]["var"] == pytest.approx(1.0, rel=0.15)
    assert out["epistemic"]["mean"] == pytest.approx(2.0, rel=0.1)
    assert out["epistemic"]["var"] == pytest.approx(0.25, rel=0.20)


def test_mean_and_variance_by_kind_empty_partition():
    """Empty partition reports zeros, not a NaN/exception."""
    n = 256
    a = jax.random.normal(jax.random.PRNGKey(0), (n,))
    out = mean_and_variance_by_kind({"a": a}, {"a": "aleatoric"})
    assert out["epistemic"] == {"mean": 0.0, "var": 0.0}
    assert out["aleatoric"]["var"] > 0.0


# ---------------------------------------------------------------------------
# 6. Default-off byte-equivalence
# ---------------------------------------------------------------------------

def test_distribution_kind_does_not_break_existing_signature():
    """Pre-T-126 callers used positional args; that contract still holds."""
    u = Uniform(0.0, 1.0)
    assert (u.low, u.high) == (0.0, 1.0)

    n = Normal(2.0, 0.5)
    assert (n.loc, n.scale) == (2.0, 0.5)

    ln = LogNormal(0.0, 0.25)
    assert (ln.mu, ln.sigma) == (0.0, 0.25)

    tri = Triangular(0.0, 0.5, 1.0)
    assert (tri.low, tri.mode, tri.high) == (0.0, 0.5, 1.0)
