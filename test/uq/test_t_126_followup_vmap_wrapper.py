# SPDX-License-Identifier: MIT

"""T-126 followup — vmap-friendly Sobol/UQ pipeline.

Covers :func:`jaxonomy.uq.vmap_qoi`:

1. A scalar per-sample ``qoi_fn(params: dict[str, float]) -> float`` wrapped
   with :func:`vmap_qoi` and passed to :func:`decompose_variance_sobol` /
   :func:`sobol_indices` / :func:`morris_screening` reproduces the result
   from a hand-written batched callable bit-for-bit.
2. The pre-existing batched form continues to work (no API regression).
3. Validation: malformed per-sample ``qoi_fn`` (wrong shape, ragged inputs,
   empty dict, non-callable) raises clear errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    Normal,
    Uniform,
    decompose_variance_sobol,
    morris_screening,
    sobol_indices,
    vmap_qoi,
)

skip_if_not_jax()
pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# 1. vmap_qoi vs hand-written batched callable — bit-equivalence in Sobol path.
# ---------------------------------------------------------------------------

def test_vmap_qoi_matches_handwritten_batched_in_decompose_variance_sobol():
    """Per-sample callable wrapped with ``vmap_qoi`` matches batched form."""
    a_dist = Normal(2.0, 0.05, kind="epistemic")
    x_dist = Normal(5.0, 0.10, kind="aleatoric")
    b_dist = Normal(0.0, 0.02, kind="aleatoric")

    # Hand-written batched form: user broadcasts per-key arrays themselves.
    def qoi_batched(p):
        return p["a"] * p["x"] + p["b"]

    # Per-sample form: user writes the natural scalar math.
    def qoi_scalar(p):
        return p["a"] * p["x"] + p["b"]

    qoi_wrapped = vmap_qoi(qoi_scalar)

    key = jax.random.PRNGKey(0)
    out_batched = decompose_variance_sobol(
        qoi_batched,
        aleatoric_dists={"x": x_dist, "b": b_dist},
        epistemic_dists={"a": a_dist},
        n_samples=512,
        key=key,
    )
    out_wrapped = decompose_variance_sobol(
        qoi_wrapped,
        aleatoric_dists={"x": x_dist, "b": b_dist},
        epistemic_dists={"a": a_dist},
        n_samples=512,
        key=key,
    )

    # Same key + Saltelli draw + Jansen estimator => bit equivalence within
    # a tight 1e-12 tolerance (T-005 default-float64 path).
    for field in ("var_total", "var_aleatoric", "var_epistemic", "interaction"):
        assert out_wrapped[field] == pytest.approx(out_batched[field], abs=1e-12), (
            f"field {field!r}: wrapped={out_wrapped[field]} "
            f"batched={out_batched[field]}"
        )


def test_vmap_qoi_matches_handwritten_batched_in_sobol_indices():
    """Bit-equivalence also for ``sobol_indices`` analytic mode."""
    dists = {
        "x1": Uniform(-3.14, 3.14),
        "x2": Uniform(-3.14, 3.14),
    }

    def qoi_batched(p):
        return jnp.sin(p["x1"]) + 7.0 * jnp.sin(p["x2"]) ** 2

    def qoi_scalar(p):
        return jnp.sin(p["x1"]) + 7.0 * jnp.sin(p["x2"]) ** 2

    key = jax.random.PRNGKey(1)
    out_batched = sobol_indices(
        None, None, dists, qoi_batched, n_samples=256, key=key,
    )
    out_wrapped = sobol_indices(
        None, None, dists, vmap_qoi(qoi_scalar), n_samples=256, key=key,
    )

    for name in dists:
        for field in ("first_order", "total_order"):
            assert out_wrapped[name][field] == pytest.approx(
                out_batched[name][field], abs=1e-12
            )


def test_vmap_qoi_matches_handwritten_batched_in_morris():
    """Bit-equivalence also for ``morris_screening`` analytic mode."""
    dists = {
        "x1": Uniform(0.0, 1.0),
        "x2": Uniform(0.0, 1.0),
        "x3": Uniform(0.0, 1.0),
    }

    def qoi_batched(p):
        return p["x1"] + 5.0 * p["x2"] + 0.1 * p["x3"]

    def qoi_scalar(p):
        return p["x1"] + 5.0 * p["x2"] + 0.1 * p["x3"]

    key = jax.random.PRNGKey(2)
    out_batched = morris_screening(
        None, None, dists, qoi_batched, n_trajectories=8, levels=4, key=key,
    )
    out_wrapped = morris_screening(
        None, None, dists, vmap_qoi(qoi_scalar),
        n_trajectories=8, levels=4, key=key,
    )

    for name in dists:
        for field in ("mu_star", "sigma"):
            assert out_wrapped[name][field] == pytest.approx(
                out_batched[name][field], abs=1e-12
            )


# ---------------------------------------------------------------------------
# 2. Pre-existing batched form still works (no regression in the default API).
# ---------------------------------------------------------------------------

def test_batched_form_still_works_decompose_variance_sobol():
    """User-supplied batched callable continues to work unchanged."""
    a_dist = Normal(1.0, 0.1, kind="epistemic")
    x_dist = Normal(0.0, 1.0, kind="aleatoric")

    def qoi(p):
        return p["a"] * p["x"]

    out = decompose_variance_sobol(
        qoi,
        aleatoric_dists={"x": x_dist},
        epistemic_dists={"a": a_dist},
        n_samples=256,
        key=jax.random.PRNGKey(42),
    )
    assert "var_total" in out
    assert out["var_total"] > 0.0


# ---------------------------------------------------------------------------
# 3. Validation — malformed inputs raise clear errors.
# ---------------------------------------------------------------------------

def test_vmap_qoi_rejects_non_callable():
    with pytest.raises(TypeError, match="must be callable"):
        vmap_qoi(42)  # type: ignore[arg-type]


def test_vmap_qoi_rejects_empty_params():
    wrapped = vmap_qoi(lambda p: p["a"])
    with pytest.raises(ValueError, match="non-empty"):
        wrapped({})


def test_vmap_qoi_rejects_ragged_sample_axis():
    """Per-key sample axes must agree."""
    wrapped = vmap_qoi(lambda p: p["a"] + p["b"])
    with pytest.raises(ValueError, match="leading dim"):
        wrapped({"a": jnp.zeros(8), "b": jnp.zeros(7)})


def test_vmap_qoi_rejects_zero_dim_input():
    """Each value must have at least one (sample) axis."""
    wrapped = vmap_qoi(lambda p: p["a"])
    with pytest.raises(ValueError, match="0-D scalar"):
        wrapped({"a": jnp.asarray(3.14)})


def test_vmap_qoi_rejects_non_scalar_per_sample_output():
    """If user's qoi_fn returns a vector per sample, surface a clear error."""
    # Per-sample callable that returns an array of shape (3,) instead of ().
    def bad_qoi(p):
        return jnp.array([p["a"], p["a"], p["a"]])

    wrapped = vmap_qoi(bad_qoi)
    with pytest.raises(ValueError, match="expected"):
        wrapped({"a": jnp.arange(5.0)})
