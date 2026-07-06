# SPDX-License-Identifier: MIT

"""Regression test: ``declare_periodic_update(period=...)`` must default
``offset`` to ``0.0`` rather than leave it as ``None``.

Before the fix, omitting ``offset=`` made construction succeed but the
first simulation step raised an opaque
``TypeError: unsupported operand type(s) for ...: 'NoneType' and 'float'``
deep inside the scheduler when it tried
``npa.minimum(None, next_periodic_event_time)``.

The scheduler error never named the offending block. Users would chase the
simulator core; the right place to fail (or just not fail) is at
``declare_periodic_update``.
"""

from __future__ import annotations

import pytest

import jax.numpy as jnp

import jaxonomy


class _DTNoOffset(jaxonomy.LeafSystem):
    """Discrete-time block declaring a periodic update without an explicit
    ``offset`` kwarg."""

    def __init__(self):
        super().__init__()
        self.declare_discrete_state(default_value=jnp.array(1.0))
        # NOTE: deliberately omitting ``offset=`` — this is the user pattern
        # that used to crash at the first scheduler tick.
        self.declare_periodic_update(self._upd, period=0.1)

    def _upd(self, time, state, **params):
        return state.discrete_state * 0.9


def test_simulate_without_explicit_offset():
    sys = _DTNoOffset()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=30)
    results = jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts)
    # 10 updates over 1 s at period 0.1 starting at t=0 → x[n] = 0.9^n.
    final = float(results.context.discrete_state)
    assert final == pytest.approx(0.9 ** 10, abs=1e-6)
