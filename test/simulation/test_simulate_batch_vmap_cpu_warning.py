# SPDX-License-Identifier: MIT

"""Regression test for the ``simulate_batch(use_vmap=True)`` CPU caveat.

Before this followup, users running ``use_vmap=True`` on a CPU laptop hit
a counter-intuitive perf regression: the vmap path is 4-5× slower than
the kernel path because the per-row finalize is host-side ``O(N)``. The
docstring claimed vmap was always faster. The followup ships:

1. A docstring section explaining the CPU caveat.
2. A runtime ``UserWarning`` when ``use_vmap=True`` is requested on a
   CPU device with ``N < 10^4`` — the regime where the kernel path
   typically wins.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Gain
from jaxonomy.simulation import simulate_batch


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="CPU-specific warning; ignored on GPU/TPU runners.",
)


def _build_diagram():
    builder = jaxonomy.DiagramBuilder()
    osc = builder.add(_TrivialPlant(name="osc"))
    gain = builder.add(Gain(2.0, name="gain"))
    builder.connect(osc.output_ports[0], gain.input_ports[0])
    builder.export_output(gain.output_ports[0])
    return builder.build(), osc


class _TrivialPlant(jaxonomy.LeafSystem):
    """Linear decay with one dynamic parameter ``k``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_continuous_state(1, ode=self._ode)
        self.declare_dynamic_parameter("k", jnp.asarray(1.0))
        self.declare_continuous_state_output()

    def _ode(self, time, state, **params):
        return -params["k"] * state.continuous_state


def test_use_vmap_true_cpu_small_batch_emits_warning():
    diag, osc = _build_diagram()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
    n = 8  # small batch (< 10^4)
    param_batches = {"osc.k": jnp.linspace(0.5, 2.0, n)}
    recorded_signals = {"y": diag.output_ports[0]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulate_batch(
            diag,
            (0.0, 0.1),
            param_batches=param_batches,
            options=opts,
            recorded_signals=recorded_signals,
            use_vmap=True,
        )
    msgs = [str(w.message) for w in caught]
    assert any("use_vmap=True" in m and "CPU" in m for m in msgs), msgs


def test_use_vmap_false_cpu_silent():
    """The default ``use_vmap=False`` path must not emit the CPU warning."""
    diag, osc = _build_diagram()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
    n = 8
    param_batches = {"osc.k": jnp.linspace(0.5, 2.0, n)}
    recorded_signals = {"y": diag.output_ports[0]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulate_batch(
            diag,
            (0.0, 0.1),
            param_batches=param_batches,
            options=opts,
            recorded_signals=recorded_signals,
            # use_vmap defaults to False
        )
    cpu_warnings = [w for w in caught if "use_vmap=True" in str(w.message)]
    assert not cpu_warnings
