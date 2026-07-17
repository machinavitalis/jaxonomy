# SPDX-License-Identifier: MIT

"""Multi-device ensemble execution recipe (T-021).

Runs an N=8 parameter sweep over a scalar exponential-decay system across a
fake 4-device CPU mesh, using ``jaxonomy.simulate_distributed``, and compares
the result against a serial ``simulate_batch`` call.

This example uses ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` so it
runs on any machine with one CPU.  On real multi-GPU / multi-TPU hosts, drop
the flag (``jax.devices()`` will return the real devices) and otherwise keep
the recipe identical.

Run:

    python docs/examples/distributed_ensemble.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before jax is imported.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

# Allow `python docs/examples/distributed_ensemble.py` from the repo root
# without an editable install — Python only adds the script's directory to
# sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jaxonomy  # noqa: E402
from jaxonomy import (  # noqa: E402
    DiagramBuilder,
    LeafSystem,
    SimulatorOptions,
    simulate_batch,
    simulate_distributed,
)


class Decay(LeafSystem):
    """xdot = -k * x; output = x."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("k", 1.0)
        self.declare_continuous_state(
            default_value=jnp.array(1.0), ode=self._ode,
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, t, state, **p):
        return -p["k"] * state.continuous_state


def main():
    print(f"jax.devices() = {jax.devices()}")
    print(f"jax.device_count() = {jax.device_count()}")

    db = DiagramBuilder()
    db.add(Decay(name="leaf"))
    diagram = db.build(name="root")

    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        max_major_steps=200, return_context=False,
    )
    rec = {"x": diagram["leaf"].output_ports[0]}

    n = 8
    ks = jnp.linspace(0.5, 2.0, n)
    t_span = (0.0, 1.0)

    res_serial = simulate_batch(
        diagram, t_span=t_span,
        param_batches={"leaf.k": ks},
        options=opts, recorded_signals=rec,
    )
    res_dist = simulate_distributed(
        diagram, t_span=t_span,
        param_batches={"leaf.k": ks},
        options=opts, recorded_signals=rec,
    )

    diff = float(np.max(np.abs(
        np.asarray(res_serial.outputs["x"]) - np.asarray(res_dist.outputs["x"])
    )))
    print(f"simulate_batch       output shape = {res_serial.outputs['x'].shape}")
    print(f"simulate_distributed output shape = {res_dist.outputs['x'].shape}")
    print(f"max |serial - distributed| = {diff:.3e}")
    print(f"first run final value = {float(res_dist.outputs['x'][0, -1]):.6f}")
    print(f"last  run final value = {float(res_dist.outputs['x'][-1, -1]):.6f}")
    # exp(-0.5) ~ 0.6065, exp(-2.0) ~ 0.1353
    expected_first = float(np.exp(-float(ks[0])))
    expected_last = float(np.exp(-float(ks[-1])))
    print(f"expected first / last = {expected_first:.6f} / {expected_last:.6f}")
    assert diff < 1e-6, f"distributed result diverged from serial: {diff}"
    print("OK")


if __name__ == "__main__":
    main()
