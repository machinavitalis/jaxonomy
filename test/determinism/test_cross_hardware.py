# SPDX-License-Identifier: MIT
"""
T-002 Phase D — cross-hardware consistency scaffolding.

POLICY.md explicitly narrows the bit-exact contract to a single device.
Cross-device reproducibility is not guaranteed because XLA fuses and
orders float operations differently on CPU / GPU / TPU, and accelerator
reductions may run in parallel with non-deterministic summation order.

These tests encode the *weakened* contract that is expected to hold
cross-device, and skip cleanly when no accelerator is present. On a
CPU-only runner the file passes with "skipped". On an accelerator CI
runner the same file runs its substance.

Scope of the weakened contract:

  - Same-device repeatability **is** required (covered by the bit-exact
    tests on every runner).
  - CPU ↔ GPU / CPU ↔ TPU equality is **not** required; results may
    differ at ULP scale (~1e-6 relative for f32, ~1e-14 for f64).
  - Integer-time event ordering **is** required identically across
    devices — the integer-time representation exists precisely to
    make this deterministic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy

from ._framework import assert_bitwise_reproducible
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _accelerator_devices():
    """Return the list of non-CPU JAX devices, or empty if none available."""
    accels = []
    for kind in ("gpu", "tpu"):
        try:
            devices = jax.devices(kind)
        except RuntimeError:
            continue
        accels.extend(devices)
    return accels


_ACCELS = _accelerator_devices()
_HAS_ACCEL = len(_ACCELS) > 0
_skip_no_accel = pytest.mark.skipif(
    not _HAS_ACCEL,
    reason="no GPU/TPU available; cross-device tests only meaningful on accelerators",
)


# ── bit-exact across repeated runs on each available accelerator ────────────


@_skip_no_accel
@pytest.mark.parametrize("device_kind", [d.platform for d in _ACCELS])
def test_bit_exact_on_each_accelerator(device_kind):
    """Same-device repeatability must hold on every available accelerator,
    not only CPU. This checks that nothing in the simulator loop is pinned
    to CPU semantics."""

    class Decay(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", 1.5)
            self.declare_continuous_state(default_value=jnp.array(2.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return -params["a"] * state.continuous_state

    device = jax.devices(device_kind)[0]

    def run():
        with jax.default_device(device):
            sys = Decay()
            ctx = sys.create_context()
            opts = jaxonomy.SimulatorOptions(math_backend="jax")
            return jaxonomy.simulate(sys, ctx, (0.0, 0.5), options=opts).context.continuous_state

    assert_bitwise_reproducible(run, label=f"CT/Decay/{device_kind}")


# ── CPU vs accelerator: close but not byte-exact ────────────────────────────


@_skip_no_accel
def test_cpu_vs_accelerator_float64_close():
    """CPU and accelerator results must agree to float-ULP-scale at f64.

    This is the *allowed* cross-device deviation. A looser comparison than
    byte-exact but tight enough to fail if a numerical bug creeps in on the
    accelerator path."""

    class Decay(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", 1.5)
            self.declare_continuous_state(default_value=jnp.array(2.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return -params["a"] * state.continuous_state

    def run_on(device):
        with jax.default_device(device):
            sys = Decay()
            ctx = sys.create_context()
            opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=1e-10, atol=1e-12)
            return np.asarray(
                jaxonomy.simulate(sys, ctx, (0.0, 0.5), options=opts).context.continuous_state
            )

    cpu_result = run_on(jax.devices("cpu")[0])
    accel_result = run_on(_ACCELS[0])

    rel_err = abs(accel_result - cpu_result) / (abs(cpu_result) + 1e-30)
    assert rel_err < 1e-10, (
        f"CPU vs {_ACCELS[0].platform}: rel_err={rel_err:.3e}, "
        f"cpu={cpu_result}, accel={accel_result}"
    )


# ── integer-time event ordering must be identical cross-device ──────────────


@_skip_no_accel
def test_integer_time_event_ordering_cross_device():
    """The integer-time representation (picosecond resolution) is explicitly
    designed for deterministic event ordering. A discrete-only system with
    multiple periodic updates must fire events in the same order on every
    device, and the resulting discrete-state trajectory must be bit-exact
    across devices (no float arithmetic → no ULP noise)."""
    from jaxonomy.library import Adder

    class Counter(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_discrete_state(default_value=jnp.array(0, dtype=jnp.int32))
            self.declare_periodic_update(
                lambda t, s, **p: s.discrete_state + jnp.int32(1),
                period=0.1, offset=0.0,
            )

    def run_on(device):
        with jax.default_device(device):
            sys = Counter()
            ctx = sys.create_context()
            opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=30)
            return np.asarray(
                jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts).context.discrete_state
            )

    cpu = run_on(jax.devices("cpu")[0])
    accel = run_on(_ACCELS[0])
    assert cpu.tobytes() == accel.tobytes(), (
        f"integer-state simulation diverged CPU vs {_ACCELS[0].platform}: "
        f"cpu={cpu}, accel={accel}"
    )
