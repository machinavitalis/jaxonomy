# SPDX-License-Identifier: MIT

"""Regression test for T-114-followup-lookup-table-fitted-getter.

Before the followup, ``LookupTable2d.output_table_array`` raised
``AttributeError`` — the fitted Z array (stored via ``@parameters(static=[...])``)
was only retrievable by re-running ``fit_table_2d`` separately, which appears
in the engine-map fitting tutorial.

After the followup the static-parameter values are exposed as plain read-only
properties on the block instance.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxonomy.library import LookupTable2d


def test_lookup_table_2d_exposes_fitted_arrays():
    xs = jnp.array([0.0, 1.0, 2.0])
    ys = jnp.array([0.0, 0.5, 1.0])
    zs = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    lut = LookupTable2d(xs, ys, zs)

    np.testing.assert_array_equal(np.asarray(lut.input_x_array), np.asarray(xs))
    np.testing.assert_array_equal(np.asarray(lut.input_y_array), np.asarray(ys))
    np.testing.assert_array_equal(np.asarray(lut.output_table_array), np.asarray(zs))


def test_lookup_table_2d_properties_are_read_only():
    """The properties should be read-only — assigning to them must fail
    rather than silently shadow the static parameter."""
    import pytest

    xs = jnp.array([0.0, 1.0])
    ys = jnp.array([0.0, 1.0])
    zs = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    lut = LookupTable2d(xs, ys, zs)

    with pytest.raises(AttributeError):
        lut.output_table_array = jnp.zeros((2, 2))  # type: ignore[misc]
