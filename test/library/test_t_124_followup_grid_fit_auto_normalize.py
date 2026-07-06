# SPDX-License-Identifier: MIT

"""Regression test for T-124-followup-grid-fit-auto-normalize.

Before the followup, ``fit_table_1d_with_grid`` was scale-sensitive — the
default ``learning_rate=1e-3`` blew up to NaN on wide-but-smooth features
(e.g. engine-map slice over ``rpm ∈ [80, 650]``) while safe learning rates
barely moved the breakpoints. The function only worked on already-
normalised data with sharp features (e.g. the existing library test
fixture: narrow Gaussian on ``[-5, 5]``).

After the followup, the optimiser internally rescales ``x_data`` and
``y_data`` to roughly ``[-1, +1]`` and undoes the transform on return.
The default ``learning_rate=1e-3`` now works across data scales.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library.lookup_table_fitting import fit_table_1d_with_grid
from jaxonomy.library.lookup_table import fit_table_1d


def _wide_smooth_data():
    """A sigmoid-with-linear-trend feature on ``x ∈ [80, 650]`` with
    ``y ~ 250 N·m`` — the engine-map slice that motivated the followup."""
    rng = np.random.default_rng(0)
    x = np.linspace(80.0, 650.0, 200)
    y_smooth = 250.0 + 60.0 / (1.0 + np.exp(-(x - 350.0) / 40.0)) + 0.1 * x
    noise = rng.normal(scale=0.5, size=x.shape)
    return jnp.asarray(x), jnp.asarray(y_smooth + noise)


def test_grid_fit_handles_wide_range_with_default_lr():
    """The fitted xp/yp must contain no NaNs and the data residual must
    improve over the uniform-grid baseline."""
    x_data, y_data = _wide_smooth_data()
    n = 7

    xp_opt, yp_opt = fit_table_1d_with_grid(
        n_grid_points=n,
        x_data=x_data,
        y_data=y_data,
        max_iter=400,
        # default learning_rate=1e-3, default auto_normalize=True
    )
    assert jnp.all(jnp.isfinite(xp_opt)), f"xp_opt contains NaN/inf: {xp_opt}"
    assert jnp.all(jnp.isfinite(yp_opt)), f"yp_opt contains NaN/inf: {yp_opt}"

    # The optimised grid should outperform a uniform-grid fit on the
    # same data (the whole point of moving the breakpoints).
    xp_uniform = jnp.linspace(jnp.min(x_data), jnp.max(x_data), n)
    yp_uniform = fit_table_1d(xp_uniform, x_data, y_data)

    def rmse(xp, yp):
        # Bilinear interpolation onto x_data, then RMSE vs y_data.
        pred = jnp.interp(x_data, xp, yp)
        return float(jnp.sqrt(jnp.mean((pred - y_data) ** 2)))

    rmse_uniform = rmse(xp_uniform, yp_uniform)
    rmse_opt = rmse(xp_opt, yp_opt)
    assert rmse_opt <= rmse_uniform + 1e-6, (
        f"grid-optimised fit ({rmse_opt:.4f}) should be no worse than the "
        f"uniform-grid baseline ({rmse_uniform:.4f}); the difference signals "
        f"the optimiser is overshooting on the wide-range data"
    )


def test_grid_fit_grid_in_natural_units():
    """The returned ``xp_opt`` must be in the user's natural units, not
    the internal normalised space."""
    x_data, y_data = _wide_smooth_data()

    xp_opt, _ = fit_table_1d_with_grid(
        n_grid_points=5,
        x_data=x_data,
        y_data=y_data,
        max_iter=200,
    )
    # Endpoints pinned to data range.
    assert float(xp_opt[0]) == pytest.approx(float(jnp.min(x_data)), abs=1e-3)
    assert float(xp_opt[-1]) == pytest.approx(float(jnp.max(x_data)), abs=1e-3)
    # Interior breakpoints should land somewhere inside.
    assert all(80.0 <= float(b) <= 650.0 for b in xp_opt[1:-1])


def test_grid_fit_explicit_disable_normalize_still_works_on_unit_data():
    """``auto_normalize=False`` should reproduce pre-followup behaviour
    on already-unit-scale data — the existing library test fixture."""
    rng = np.random.default_rng(1)
    x_data = jnp.asarray(rng.uniform(-5.0, 5.0, 500))
    y_data = jnp.exp(-x_data ** 2)

    xp_norm_off, yp_norm_off = fit_table_1d_with_grid(
        n_grid_points=7,
        x_data=x_data,
        y_data=y_data,
        max_iter=200,
        auto_normalize=False,
    )
    xp_norm_on, yp_norm_on = fit_table_1d_with_grid(
        n_grid_points=7,
        x_data=x_data,
        y_data=y_data,
        max_iter=200,
        auto_normalize=True,
    )
    # On data already centred near unit scale, the two paths should
    # produce close (not identical, since the learning rate's effective
    # step changes) results.
    assert jnp.all(jnp.isfinite(xp_norm_off))
    assert jnp.all(jnp.isfinite(yp_norm_off))
    # Both should outperform a uniform-grid baseline.
    xp_unif = jnp.linspace(jnp.min(x_data), jnp.max(x_data), 7)
    yp_unif = fit_table_1d(xp_unif, x_data, y_data)
    pred_unif = jnp.interp(x_data, xp_unif, yp_unif)
    rmse_unif = float(jnp.sqrt(jnp.mean((pred_unif - y_data) ** 2)))
    for xp, yp, label in [
        (xp_norm_off, yp_norm_off, "auto_normalize=False"),
        (xp_norm_on, yp_norm_on, "auto_normalize=True"),
    ]:
        pred = jnp.interp(x_data, xp, yp)
        rmse = float(jnp.sqrt(jnp.mean((pred - y_data) ** 2)))
        assert rmse <= rmse_unif, (
            f"{label}: optimiser RMSE ({rmse:.4f}) should beat uniform "
            f"({rmse_unif:.4f})"
        )
