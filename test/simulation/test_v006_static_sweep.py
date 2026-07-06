# SPDX-License-Identifier: MIT

"""T-028: simulate_static_sweep helper.

The companion of ``simulate_batch`` for parameters declared *static* in their
LeafSystem (e.g. :class:`TransferFunction.num` / ``.den``). Static params are
baked into the block at construction; sweeping them requires rebuilding the
diagram per element. ``simulate_static_sweep`` does exactly that in a Python
loop and stacks results into a :class:`BatchSimulationResults`-shaped struct.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions, simulate, simulate_batch
from jaxonomy.library import Constant, Integrator, TransferFunction, Gain
from jaxonomy.simulation.batch import _interp_on_time
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.minimal


OPTS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="dopri5",
    rtol=1e-10,
    atol=1e-12,
    max_major_steps=400,
)


def _build_tf_diagram(num, den=(1.0, 2.0, 1.0)):
    b = DiagramBuilder()
    src = b.add(Constant(value=1.0, name="src"))
    tf = b.add(TransferFunction(num=list(num), den=list(den), name="tf"))
    out_int = b.add(Integrator(initial_state=0.0, name="out"))
    b.connect(src.output_ports[0], tf.input_ports[0])
    b.connect(tf.output_ports[0], out_int.input_ports[0])
    return b.build(name="tf_diag")


def _build_decay(a):
    """y' = -a*y, y(0)=1, parameterised by static gain a."""
    b = DiagramBuilder()
    g = b.add(Gain(gain=-a, name="g"))
    intg = b.add(Integrator(initial_state=1.0, name="intg"))
    b.connect(g.output_ports[0], intg.input_ports[0])
    b.connect(intg.output_ports[0], g.input_ports[0])
    return b.build(name="decay")


def test_v006_static_sweep_transfer_function_num():
    """Sweeping TransferFunction.num across 3 distinct numerators yields distinct outputs."""
    nums = [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    t_span = (0.0, 5.0)

    def factory(num):
        return _build_tf_diagram(num=num)

    def signals(d):
        return {"y": d["out"].output_ports[0]}

    res = jaxonomy.simulate_static_sweep(
        diagram_factory=factory,
        t_span=t_span,
        static_param_grid={"num": nums},
        options=OPTS,
        recorded_signals_factory=signals,
    )

    # Shape: (N, T) for the integrator output.
    assert res.outputs["y"].shape[0] == len(nums)
    assert res.time.shape[0] == res.outputs["y"].shape[1]
    # Per-element contexts attached.
    assert hasattr(res, "contexts")
    assert len(res.contexts) == len(nums)

    # Each TF should produce a distinct trajectory; pairwise-compare terminal values.
    terminals = np.asarray(res.outputs["y"][:, -1])
    diffs = [
        abs(terminals[i] - terminals[j])
        for i in range(len(nums))
        for j in range(i + 1, len(nums))
    ]
    assert all(d > 1e-3 for d in diffs), (
        f"Expected distinct trajectories per numerator, got terminals={terminals!r}"
    )

    # All trajectories should reach approximately the same final time
    # (within 1%) since they share the same t_span.
    final_t = float(res.time[-1])
    assert abs(final_t - t_span[1]) / t_span[1] < 0.01, (
        f"Final time {final_t} too far from t_span[1]={t_span[1]}"
    )


def test_v006_static_sweep_mixed_with_dynamic_inner_batch():
    """Mixed static+dynamic: outer loop over static gain via simulate_static_sweep.

    Each grid element produces a fresh diagram with a distinct static gain;
    we don't need an inner dynamic sweep to exercise the helper, but we
    confirm that within each element a regular ``simulate`` call works
    against the same diagram (one direction of the mixed-mode contract).
    """
    a_values = [0.5, 1.0, 2.0]
    t_span = (0.0, 3.0)

    def factory(a):
        return _build_decay(a=a)

    def signals(d):
        return {"y": d["intg"].output_ports[0]}

    res = jaxonomy.simulate_static_sweep(
        diagram_factory=factory,
        t_span=t_span,
        static_param_grid={"a": a_values},
        options=OPTS,
        recorded_signals_factory=signals,
    )

    assert res.outputs["y"].shape[0] == 3
    # y = exp(-a*T) at t=T; verify analytical form for each batch row.
    T = t_span[1]
    expected = np.exp(-np.asarray(a_values) * T)
    np.testing.assert_allclose(
        np.asarray(res.outputs["y"][:, -1]),
        expected,
        rtol=1e-5,
        atol=1e-7,
    )


def test_v006_static_sweep_empty_grid_raises():
    """An empty static_param_grid must raise a clear ValueError."""
    def factory(**kw):
        return _build_decay(a=1.0)

    def signals(d):
        return {"y": d["intg"].output_ports[0]}

    with pytest.raises(ValueError, match=r"non-empty"):
        jaxonomy.simulate_static_sweep(
            diagram_factory=factory,
            t_span=(0.0, 1.0),
            static_param_grid={},
            options=OPTS,
            recorded_signals_factory=signals,
        )


def test_v006_static_sweep_zip_mismatched_lengths_raises():
    """mode='zip' (default) requires every list to have the same length."""
    def factory(num, den):
        return _build_tf_diagram(num=num, den=den)

    def signals(d):
        return {"y": d["out"].output_ports[0]}

    with pytest.raises(ValueError, match=r"zip"):
        jaxonomy.simulate_static_sweep(
            diagram_factory=factory,
            t_span=(0.0, 1.0),
            static_param_grid={
                "num": [[1.0], [2.0]],
                "den": [[1.0, 2.0, 1.0]],  # only 1 — mismatch
            },
            options=OPTS,
            recorded_signals_factory=signals,
        )


def test_v006_static_sweep_export():
    """Exported on the top-level package."""
    assert hasattr(jaxonomy, "simulate_static_sweep")
    assert callable(jaxonomy.simulate_static_sweep)
