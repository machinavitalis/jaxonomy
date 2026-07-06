# SPDX-License-Identifier: MIT

"""Regression tests for T-126-followup-sobol-bootstrap-cis.

Before the followup, ``sobol_indices`` returned only point estimates of the
first- and total-order Sobol indices. The Jansen estimator can come out
slightly negative at small ``N`` for near-zero true indices, and the CHANGELOG
called the UQ surface "production-grade" without a way to quantify estimator
uncertainty. The followup adds an opt-in ``n_bootstrap=`` kwarg that returns
percentile confidence intervals alongside the point estimates.

The CIs are computed via JAX-vectorised resampling (one XLA launch for all
``n_bootstrap`` resamples) so a 1000-resample bootstrap on N=1024 is cheap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.uq import sobol_indices, Uniform


# Ishigami function fixture — same as test_uq.py's analytic benchmark.
ISH_A = 7.0
ISH_B = 0.1


def _ishigami(p):
    x1, x2, x3 = p["x1"], p["x2"], p["x3"]
    return jnp.sin(x1) + ISH_A * jnp.sin(x2) ** 2 + ISH_B * (x3 ** 4) * jnp.sin(x1)


@pytest.fixture
def ishigami_dists():
    pi = float(np.pi)
    return {
        "x1": Uniform(-pi, pi),
        "x2": Uniform(-pi, pi),
        "x3": Uniform(-pi, pi),
    }


def test_no_bootstrap_keeps_legacy_output_shape(ishigami_dists):
    """Default (n_bootstrap=None) must not add CI keys — backwards compat."""
    res = sobol_indices(
        diagram=None,
        t_span=None,
        distributions=ishigami_dists,
        qoi_fn=_ishigami,
        n_samples=512,
        key=jax.random.PRNGKey(0),
    )
    for name in ishigami_dists:
        assert set(res[name].keys()) == {"first_order", "total_order"}, (
            f"unexpected keys for {name}: {sorted(res[name].keys())}"
        )


def test_bootstrap_adds_ci_tuples(ishigami_dists):
    res = sobol_indices(
        diagram=None,
        t_span=None,
        distributions=ishigami_dists,
        qoi_fn=_ishigami,
        n_samples=512,
        n_bootstrap=200,
        key=jax.random.PRNGKey(0),
    )
    for name in ishigami_dists:
        entry = res[name]
        assert "first_order_ci" in entry
        assert "total_order_ci" in entry
        s1_lo, s1_hi = entry["first_order_ci"]
        st_lo, st_hi = entry["total_order_ci"]
        assert s1_lo <= s1_hi
        assert st_lo <= st_hi


def test_point_estimate_inside_ci_band(ishigami_dists):
    """The percentile bootstrap should bracket the point estimate by
    construction (the original sample is one of the possible resamples).
    Allow a small numerical slack for finite n_bootstrap."""
    res = sobol_indices(
        diagram=None,
        t_span=None,
        distributions=ishigami_dists,
        qoi_fn=_ishigami,
        n_samples=512,
        n_bootstrap=400,
        key=jax.random.PRNGKey(0),
    )
    for name in ishigami_dists:
        entry = res[name]
        s1 = entry["first_order"]
        s1_lo, s1_hi = entry["first_order_ci"]
        st = entry["total_order"]
        st_lo, st_hi = entry["total_order_ci"]
        # Use a slack of 1.5x the half-width to absorb bootstrap noise.
        s1_slack = max((s1_hi - s1_lo) * 0.5, 0.05)
        st_slack = max((st_hi - st_lo) * 0.5, 0.05)
        assert s1_lo - s1_slack <= s1 <= s1_hi + s1_slack
        assert st_lo - st_slack <= st <= st_hi + st_slack


def test_ci_narrows_with_more_samples(ishigami_dists):
    """Confidence intervals should shrink as the base sample size grows —
    a basic sanity check that the bootstrap is informative."""
    res_small = sobol_indices(
        diagram=None,
        t_span=None,
        distributions=ishigami_dists,
        qoi_fn=_ishigami,
        n_samples=256,
        n_bootstrap=300,
        key=jax.random.PRNGKey(0),
    )
    res_large = sobol_indices(
        diagram=None,
        t_span=None,
        distributions=ishigami_dists,
        qoi_fn=_ishigami,
        n_samples=2048,
        n_bootstrap=300,
        key=jax.random.PRNGKey(0),
    )
    # Check the first-order CI width on x1 (the strongest signal) — should
    # be roughly 1/sqrt(N) narrower at the larger sample size.
    w_small = res_small["x1"]["first_order_ci"][1] - res_small["x1"]["first_order_ci"][0]
    w_large = res_large["x1"]["first_order_ci"][1] - res_large["x1"]["first_order_ci"][0]
    assert w_large < w_small, (
        f"expected CI width to shrink with more samples; got {w_small=} {w_large=}"
    )


def test_invalid_ci_level_raises(ishigami_dists):
    with pytest.raises(ValueError, match="ci_level"):
        sobol_indices(
            diagram=None,
            t_span=None,
            distributions=ishigami_dists,
            qoi_fn=_ishigami,
            n_samples=128,
            n_bootstrap=10,
            ci_level=1.5,  # invalid
            key=jax.random.PRNGKey(0),
        )
