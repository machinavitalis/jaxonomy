# SPDX-License-Identifier: MIT

"""Tests for :func:`jaxonomy.simulate_distributed` (T-021).

The multi-device path is gated on ``len(jax.devices()) >= 2`` and is
exercised in CI via the ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
recipe; locally without that flag, the multi-device tests skip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import (
    DiagramBuilder,
    LeafSystem,
    SimulatorOptions,
    simulate_batch,
    simulate_distributed,
)

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Diagram + helpers
# ---------------------------------------------------------------------------

class _Decay(LeafSystem):
    """Scalar exponential decay xdot = -k * x."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("k", 1.0)
        self.declare_continuous_state(
            default_value=jnp.array(1.0), ode=self._ode
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, t, state, **p):
        return -p["k"] * state.continuous_state


def _build_decay_diagram():
    db = DiagramBuilder()
    db.add(_Decay(name="leaf"))
    return db.build(name="root")


def _opts():
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method="dopri5",
        max_major_steps=200,
        return_context=False,
    )


# ---------------------------------------------------------------------------
# 1-device degenerate path
# ---------------------------------------------------------------------------

def test_single_device_matches_simulate_batch():
    """With 1 device, simulate_distributed defers to simulate_batch's kernel
    path and returns identical numerics."""
    sys = _build_decay_diagram()
    rec = {"x": sys["leaf"].output_ports[0]}

    n = 4
    ks = jnp.linspace(0.5, 2.0, n)
    res_ref = simulate_batch(
        sys, t_span=(0.0, 1.0),
        param_batches={"leaf.k": ks},
        options=_opts(), recorded_signals=rec,
    )
    res_dist = simulate_distributed(
        sys, t_span=(0.0, 1.0),
        param_batches={"leaf.k": ks},
        options=_opts(), recorded_signals=rec,
        devices=[jax.devices()[0]],
    )
    assert res_dist.outputs["x"].shape == res_ref.outputs["x"].shape
    np.testing.assert_array_equal(
        np.asarray(res_dist.outputs["x"]),
        np.asarray(res_ref.outputs["x"]),
    )


# ---------------------------------------------------------------------------
# Multi-device path (gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="multi-device test needs >=2 jax devices "
           "(re-run with XLA_FLAGS=--xla_force_host_platform_device_count=N)",
)
def test_multi_device_matches_simulate_batch():
    """With multiple devices, simulate_distributed produces the same numerics
    as simulate_batch up to XLA reduction-order rounding."""
    sys = _build_decay_diagram()
    rec = {"x": sys["leaf"].output_ports[0]}

    n_dev = len(jax.devices())
    n = n_dev * 2
    ks = jnp.linspace(0.5, 2.0, n)

    res_ref = simulate_batch(
        sys, t_span=(0.0, 1.0),
        param_batches={"leaf.k": ks},
        options=_opts(), recorded_signals=rec,
    )
    res_dist = simulate_distributed(
        sys, t_span=(0.0, 1.0),
        param_batches={"leaf.k": ks},
        options=_opts(), recorded_signals=rec,
    )
    np.testing.assert_allclose(
        np.asarray(res_dist.outputs["x"]),
        np.asarray(res_ref.outputs["x"]),
        rtol=1e-9, atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Error paths (run without multi-device)
# ---------------------------------------------------------------------------

def test_indivisible_batch_size_raises():
    """N not divisible by len(devices) raises a clear error."""
    if len(jax.devices()) < 2:
        pytest.skip("needs >=2 devices to test divisibility")
    sys = _build_decay_diagram()
    rec = {"x": sys["leaf"].output_ports[0]}
    n_dev = len(jax.devices())
    n = n_dev * 2 + 1  # not divisible
    ks = jnp.linspace(0.5, 2.0, n)

    with pytest.raises(ValueError, match="not divisible"):
        simulate_distributed(
            sys, t_span=(0.0, 1.0),
            param_batches={"leaf.k": ks},
            options=_opts(), recorded_signals=rec,
        )


def test_non_pure_jax_diagram_raises():
    """A diagram with a CustomPythonBlock raises at simulate_distributed."""
    from jaxonomy.library import CustomPythonBlock

    db = DiagramBuilder()
    db.add(CustomPythonBlock(
        dt=0.1, init_script="y = 0.0",
        user_statements="y = y + 1.0",
        inputs=[], outputs=["y"], name="cpb",
    ))
    sys = db.build(name="cpb_diag")
    rec = {"y": sys["cpb"].output_ports[0]}

    with pytest.raises(ValueError, match="pure-JAX"):
        simulate_distributed(
            sys, t_span=(0.0, 1.0),
            param_batches={"cpb.dt": jnp.array([0.1, 0.2])},
            options=_opts(), recorded_signals=rec,
            devices=[jax.devices()[0]],
        )


def test_missing_options_raises():
    sys = _build_decay_diagram()
    rec = {"x": sys["leaf"].output_ports[0]}
    with pytest.raises(ValueError, match="SimulatorOptions"):
        simulate_distributed(
            sys, t_span=(0.0, 1.0),
            param_batches={"leaf.k": jnp.array([1.0, 2.0])},
            options=None, recorded_signals=rec,
        )


def test_distributed_exported_from_top_level():
    """``jaxonomy.simulate_distributed`` resolves and is callable."""
    assert callable(jaxonomy.simulate_distributed)
