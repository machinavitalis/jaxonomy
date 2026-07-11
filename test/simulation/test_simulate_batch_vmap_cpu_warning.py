# SPDX-License-Identifier: MIT

"""T-019-followup: the vmap path emits no CPU advisory anymore.

Historical context: an earlier follow-up added a ``UserWarning`` when
``use_vmap=True`` ran on a CPU device with ``N < 10^4``, because the
post-vmap per-row host-side finalize made the vmap path 4-5× slower than
the kernel path in that regime. T-019-followup vectorised the finalize
(batched trim + batched binary-search resampling), removing the penalty
(0.41 s vs 0.33 s at N=1000 on the reference sweep) — and the warning
with it. This file pins the new contract: **no** CPU advisory fires on
either path, and the vmap path returns well-formed results at small N
on CPU.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Gain
from jaxonomy.simulation import simulate_batch


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="CPU-specific contract; ignored on GPU/TPU runners.",
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


def _run(use_vmap: bool):
    diag, _osc = _build_diagram()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
    n = 8
    param_batches = {"osc.k": jnp.linspace(0.5, 2.0, n)}
    recorded_signals = {"y": diag.output_ports[0]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = simulate_batch(
            diag,
            (0.0, 0.1),
            param_batches=param_batches,
            options=opts,
            recorded_signals=recorded_signals,
            use_vmap=use_vmap,
        )
    return results, [str(w.message) for w in caught]


def test_use_vmap_true_cpu_small_batch_is_silent_and_correct():
    results, msgs = _run(use_vmap=True)
    assert not any("use_vmap=True" in m and "CPU" in m for m in msgs), msgs
    y = np.asarray(results.outputs["y"])
    assert y.shape[0] == 8
    assert np.all(np.isfinite(y))


def test_use_vmap_false_cpu_silent():
    _results, msgs = _run(use_vmap=False)
    assert not any("use_vmap=True" in m for m in msgs), msgs
