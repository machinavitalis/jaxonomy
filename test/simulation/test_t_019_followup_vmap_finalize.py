# SPDX-License-Identifier: MIT

"""T-019-followup: the vectorised vmap-path finalize must be numerically
equivalent to the per-row loop it replaced, on both of its branches:

* identical-grid fast path (fixed-step rk4 → all rows share one grid,
  finalize is pure slicing);
* ragged path (adaptive dopri5 → per-row grids differ, finalize is the
  batched binary-search linear resampler).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Adder, Gain, Integrator

pytestmark = pytest.mark.minimal


def _oscillator():
    """Damped oscillator with the stiffness gain as the swept parameter."""
    b = jaxonomy.DiagramBuilder()
    xd = b.add(Integrator(1.0, name="xd"))
    x = b.add(Integrator(0.0, name="x"))
    kx = b.add(Gain(-4.0, name="kx"))
    cxd = b.add(Gain(-0.5, name="cxd"))
    acc = b.add(Adder(2, name="acc"))
    b.connect(xd.output_ports[0], x.input_ports[0])
    b.connect(x.output_ports[0], kx.input_ports[0])
    b.connect(xd.output_ports[0], cxd.input_ports[0])
    b.connect(kx.output_ports[0], acc.input_ports[0])
    b.connect(cxd.output_ports[0], acc.input_ports[1])
    b.connect(acc.output_ports[0], xd.input_ports[0])
    return b.build()


def _run(diagram, options, n, use_vmap, force_loop=False):
    param_batches = {"kx.gain": jnp.linspace(-5.0, -3.0, n)}
    signals = {"x": diagram["x"].output_ports[0]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jaxonomy.simulate_batch(
            diagram,
            (0.0, 5.0),
            param_batches,
            options=options,
            recorded_signals=signals,
            use_vmap=use_vmap,
            _force_loop=force_loop,
        )


def test_ragged_grids_match_loop_path():
    """Adaptive solver → per-row grids diverge across the parameter sweep;
    the batched resampler must agree with per-row np.interp regridding of
    the loop-path reference."""
    n = 8
    diagram = _oscillator()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
    res_vmap = _run(diagram, opts, n, use_vmap=True)
    res_loop = _run(diagram, opts, n, use_vmap=False, force_loop=True)

    t_v = np.asarray(res_vmap.time)
    x_v = np.asarray(res_vmap.outputs["x"])
    t_l = np.asarray(res_loop.time)
    x_l = np.asarray(res_loop.outputs["x"])
    assert x_v.shape[0] == n

    for i in range(n):
        x_regrid = np.interp(t_v, t_l, x_l[i])
        np.testing.assert_allclose(x_v[i], x_regrid, atol=1e-4)


def test_identical_grids_use_exact_slicing():
    """Fixed-step rk4 → every row records the same grid; the fast path
    must return the samples verbatim (no interpolation error at all)."""
    n = 6
    diagram = _oscillator()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        max_major_steps=100,
        ode_solver_method="rk4",
        max_minor_step_size=0.05,
    )
    res_vmap = _run(diagram, opts, n, use_vmap=True)
    res_loop = _run(diagram, opts, n, use_vmap=False, force_loop=True)

    t_v = np.asarray(res_vmap.time)
    t_l = np.asarray(res_loop.time)
    np.testing.assert_allclose(t_v, t_l, atol=0)
    np.testing.assert_allclose(
        np.asarray(res_vmap.outputs["x"]),
        np.asarray(res_loop.outputs["x"]),
        atol=1e-12,
    )
