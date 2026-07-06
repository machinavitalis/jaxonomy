# SPDX-License-Identifier: MIT

"""Regression test for ``fit_table_nd`` / ``fit_lookup_table_nd`` (F1 part 3 follow-up).

Before this followup, ``fit_lookup_table_1d`` and ``fit_lookup_table_2d``
shipped but the N-D analogue did not — every tutorial author hitting a
multi-dimensional surrogate (3-D aero map, 4-D engine torque table, ...)
re-implemented the same ~30 LOC multilinear design matrix.
``fit_table_nd`` exposes the math; ``fit_lookup_table_nd`` wraps a fitted
``LookupTableND`` block.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import (
    LookupTableND,
    fit_lookup_table_nd,
    fit_table_nd,
)


def _seeded_rng(seed=0):
    return np.random.default_rng(seed)


def test_fit_table_nd_3d_linear_recovers_analytic():
    """f(x, y, z) = x + 2y + 3z is exactly representable on a multilinear
    grid; the LS fit should recover machine-precision table values."""
    rng = _seeded_rng()
    xs = jnp.linspace(0.0, 1.0, 4)
    ys = jnp.linspace(0.0, 1.0, 5)
    zs = jnp.linspace(0.0, 1.0, 6)
    x_data = jnp.asarray(rng.uniform(0, 1, (500, 3)))
    y_data = x_data[:, 0] + 2 * x_data[:, 1] + 3 * x_data[:, 2]

    table = fit_table_nd((xs, ys, zs), x_data, y_data)
    expected = (
        xs[:, None, None] + 2 * ys[None, :, None] + 3 * zs[None, None, :]
    )
    assert table.shape == expected.shape
    err = float(jnp.max(jnp.abs(table - expected)))
    assert err < 1e-9, f"expected machine-precision recovery on linear data, got {err}"


def test_fit_table_nd_4d_shape_and_finite():
    rng = _seeded_rng(1)
    grids = (
        jnp.linspace(0.0, 1.0, 4),
        jnp.linspace(0.0, 1.0, 5),
        jnp.linspace(0.0, 1.0, 3),
        jnp.linspace(0.0, 1.0, 3),
    )
    K = 800
    x_data = jnp.asarray(rng.uniform(0, 1, (K, 4)))
    y_data = (
        x_data[:, 0] - x_data[:, 1] + 0.5 * x_data[:, 2] + 2.0 * x_data[:, 3]
    )

    table = fit_table_nd(grids, x_data, y_data)
    assert table.shape == (4, 5, 3, 3)
    assert jnp.all(jnp.isfinite(table))


def test_fit_table_nd_smoothness_reduces_noise():
    """With sparse + noisy data, the N-D Laplacian penalty should pull the
    fitted table toward something smoother than the unregularised LS."""
    rng = _seeded_rng(2)
    xs = jnp.linspace(0.0, 1.0, 6)
    ys = jnp.linspace(0.0, 1.0, 6)
    zs = jnp.linspace(0.0, 1.0, 6)

    K = 80  # sparse relative to 6^3 = 216 cells
    x_data = jnp.asarray(rng.uniform(0, 1, (K, 3)))
    y_data_clean = x_data[:, 0] + x_data[:, 1] + x_data[:, 2]
    noise = jnp.asarray(rng.normal(scale=0.3, size=K))
    y_data = y_data_clean + noise

    table_unreg = fit_table_nd((xs, ys, zs), x_data, y_data, smoothness=0.0)
    table_smooth = fit_table_nd((xs, ys, zs), x_data, y_data, smoothness=1.0)

    # Smoothness penalty reduces cell-to-cell variation. Measure first-
    # difference RMS along axis 0 — the smoothed fit should be quieter.
    diff_unreg = jnp.diff(table_unreg, axis=0)
    diff_smooth = jnp.diff(table_smooth, axis=0)
    rms_unreg = float(jnp.sqrt(jnp.mean(diff_unreg ** 2)))
    rms_smooth = float(jnp.sqrt(jnp.mean(diff_smooth ** 2)))
    assert rms_smooth < rms_unreg, (
        f"smoothness=1.0 did not reduce variation: "
        f"rms_unreg={rms_unreg:.4f}, rms_smooth={rms_smooth:.4f}"
    )


def test_fit_lookup_table_nd_returns_usable_block():
    import jaxonomy
    from jaxonomy.library import Constant

    rng = _seeded_rng(3)
    xs = jnp.linspace(0.0, 1.0, 4)
    ys = jnp.linspace(0.0, 1.0, 4)
    zs = jnp.linspace(0.0, 1.0, 4)
    K = 300
    x_data = jnp.asarray(rng.uniform(0, 1, (K, 3)))
    y_data = jnp.sin(2 * x_data[:, 0]) + x_data[:, 1] * x_data[:, 2]

    block = fit_lookup_table_nd((xs, ys, zs), x_data, y_data)
    assert isinstance(block, LookupTableND)

    # Wire into a Diagram with a Constant source so the input port has an
    # upstream — direct ``fixed(...)`` works only when the block is built
    # standalone via ``submodel_function`` machinery.
    builder = jaxonomy.DiagramBuilder()
    blk = builder.add(block)
    src = builder.add(Constant(jnp.array([0.3, 0.4, 0.5])))
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    val = float(blk.output_ports[0].eval(ctx))
    assert jnp.isfinite(val)


def test_fit_table_nd_validates_inputs():
    xs = jnp.linspace(0.0, 1.0, 4)
    ys = jnp.linspace(0.0, 1.0, 5)
    # Wrong x_data dimensionality.
    with pytest.raises(ValueError, match="x_data"):
        fit_table_nd((xs, ys), jnp.zeros(10), jnp.zeros(10))
    # Wrong number of columns vs grid axes.
    with pytest.raises(ValueError, match="grid axes"):
        fit_table_nd((xs, ys), jnp.zeros((10, 3)), jnp.zeros(10))
    # y_data shape mismatch.
    with pytest.raises(ValueError, match="y_data shape"):
        fit_table_nd((xs, ys), jnp.zeros((10, 2)), jnp.zeros(7))
    # Negative smoothness.
    with pytest.raises(ValueError, match="smoothness"):
        fit_table_nd((xs, ys), jnp.zeros((10, 2)), jnp.zeros(10), smoothness=-1.0)
    # Grid axis too short.
    too_short = jnp.array([0.0])
    with pytest.raises(ValueError, match="at least 2 breakpoints"):
        fit_table_nd((too_short, ys), jnp.zeros((10, 2)), jnp.zeros(10))
