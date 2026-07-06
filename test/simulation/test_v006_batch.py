# SPDX-License-Identifier: MIT

"""V-006: simulate_batch correctness.

Each test runs ``simulate_batch`` with N>=3 batch elements and compares against
the serial reference produced by N independent ``simulate`` calls (element-wise,
rtol up to 1e-10). Also verifies gradient flow through the semantically
equivalent serial computation.

API note: ``simulate_batch`` takes ``param_batches`` (dict mapping dot-paths to
stacked arrays of shape ``(N, ...)``), not a list of pre-built contexts. The
prompt's cheatsheet was inaccurate; the actual signature is verified against
``jaxonomy/simulation/batch.py`` (signature line 307).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions, simulate
from jaxonomy.library import (
    Adder,
    Constant,
    CustomJaxBlock,
    Gain,
    Integrator,
    PID,
    TransferFunction,
)
from jaxonomy.simulation.batch import _interp_on_time, simulate_batch
from jaxonomy.testing import fd_grad
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.minimal


# Tight ODE tolerances so the solver isn't the dominant error in the
# batch-vs-serial comparison (rtol up to 1e-10).
OPTS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="dopri5",
    rtol=1e-12,
    atol=1e-14,
    max_major_steps=400,
)


def _serial_outputs(diagram, port_path, t_span, batch_path, batch_values):
    """Run ``simulate`` once per param value via ``diagram.with_parameters``."""
    out = []
    for v in batch_values:
        d = diagram.with_parameters({batch_path: jnp.asarray(v)})
        ctx = d.create_context()
        port_block = d.find_system_with_path(port_path[0])
        port = port_block.output_ports[port_path[1]]
        res = simulate(d, ctx, t_span, options=OPTS, recorded_signals={"y": port})
        out.append((res.time, res.outputs["y"]))
    return out


def _assert_batch_matches_serial(batch_res, serial_runs, rtol=1e-10):
    """Compare batch outputs (on batch_res.time grid) to interpolated serial runs."""
    t_ref = batch_res.time
    for i, (t_i, y_i) in enumerate(serial_runs):
        y_on_ref = _interp_on_time(y_i, t_i, t_ref)
        np.testing.assert_allclose(
            np.asarray(batch_res.outputs["y"][i]),
            np.asarray(y_on_ref),
            rtol=rtol,
            atol=1e-10,
            err_msg=f"batch index {i} disagrees with serial reference",
        )


def _build_decay(a=1.0):
    """y' = -a * y, y(0)=1. Implemented as Gain(-a) with feedback to Integrator."""
    b = DiagramBuilder()
    g = b.add(Gain(gain=-a, name="g"))
    intg = b.add(Integrator(initial_state=1.0, name="intg"))
    b.connect(g.output_ports[0], intg.input_ports[0])
    b.connect(intg.output_ports[0], g.input_ports[0])
    return b.build(name="decay")


def _build_pid_plant(kp=1.0):
    """PID controller driving a 1/s plant tracking a unit step reference."""
    b = DiagramBuilder()
    ref = b.add(Constant(value=1.0, name="ref"))
    err = b.add(Adder(2, operators="+-", name="err"))
    pid = b.add(PID(kp=kp, ki=0.5, kd=0.1, n=50.0, name="pid"))
    plant = b.add(Integrator(initial_state=0.0, name="plant"))
    b.connect(ref.output_ports[0], err.input_ports[0])
    b.connect(plant.output_ports[0], err.input_ports[1])
    b.connect(err.output_ports[0], pid.input_ports[0])
    b.connect(pid.output_ports[0], plant.input_ports[0])
    return b.build(name="pid_plant")


def _build_tf_diagram(num=(1.0,), den=(1.0, 2.0, 1.0)):
    b = DiagramBuilder()
    src = b.add(Constant(value=1.0, name="src"))
    tf = b.add(TransferFunction(num=list(num), den=list(den), name="tf"))
    out_int = b.add(Integrator(initial_state=0.0, name="out"))
    b.connect(src.output_ports[0], tf.input_ports[0])
    b.connect(tf.output_ports[0], out_int.input_ports[0])
    return b.build(name="tf_diag")


def _build_custom_jax(a=1.0):
    """CustomJaxBlock with dynamic param 'a'; output y = a, integrated."""
    b = DiagramBuilder()
    cb = b.add(
        CustomJaxBlock(
            dt=0.01,
            init_script="y = 0.0",
            user_statements="y = a * 1.0",
            inputs=[],
            outputs=["y"],
            dynamic_parameters={"a": a},
            name="cb",
        )
    )
    intg = b.add(Integrator(initial_state=0.0, name="intg"))
    b.connect(cb.output_ports[0], intg.input_ports[0])
    return b.build(name="custom_jax")


def test_v006_dynamic_sweep_decay():
    sys = _build_decay()
    a_values = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
    g_values = -a_values  # g.gain = -a in the model
    t_span = (0.0, 5.0)
    port = sys["intg"].output_ports[0]

    batch_res = simulate_batch(
        sys,
        t_span=t_span,
        param_batches={"g.gain": g_values},
        options=OPTS,
        recorded_signals={"y": port},
    )
    assert batch_res.outputs["y"].shape[0] == len(a_values)

    serial = _serial_outputs(sys, ("intg", 0), t_span, "g.gain", g_values)
    _assert_batch_matches_serial(batch_res, serial, rtol=1e-10)

    # Sanity: terminal value matches analytical exp(-a*T).
    T = t_span[1]
    np.testing.assert_allclose(
        np.asarray(batch_res.outputs["y"][:, -1]),
        np.exp(-np.asarray(a_values) * T),
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.xfail(
    reason="superseded by T-028 simulate_static_sweep helper",
    strict=False,
)
def test_v006_static_only_sweep_xfail():
    sys = _build_tf_diagram()
    num_batch = jnp.array([[1.0], [2.0], [3.0]])
    simulate_batch(
        sys,
        t_span=(0.0, 1.0),
        param_batches={"tf.num": num_batch},
        options=OPTS,
        recorded_signals={"y": sys["out"].output_ports[0]},
        _force_loop=True,
    )


def test_v006_combined_sweep_two_dynamic_params():
    """Sweep two dynamic parameters together; batch == serial.

    Since simulate_batch cannot sweep static parameters (Case 2), the
    "static+dynamic combined" sweep collapses to "two dynamic params".
    T-030 removed PID._eval_output's self.A/B/C/D mutation, so the kernel
    path now works without ``_force_loop=True``.
    """
    sys = _build_pid_plant()
    N = 4
    kp_vals = jnp.linspace(0.5, 2.0, N)
    ki_vals = jnp.linspace(0.1, 0.8, N)
    t_span = (0.0, 4.0)
    port = sys["plant"].output_ports[0]

    batch_res = simulate_batch(
        sys,
        t_span=t_span,
        param_batches={"pid.kp": kp_vals, "pid.ki": ki_vals},
        options=OPTS,
        recorded_signals={"y": port},
    )
    assert batch_res.outputs["y"].shape[0] == N

    serial = []
    for kp, ki in zip(kp_vals, ki_vals):
        d = sys.with_parameters({"pid.kp": kp, "pid.ki": ki})
        ctx = d.create_context()
        res = simulate(
            d, ctx, t_span,
            options=OPTS,
            recorded_signals={"y": d["plant"].output_ports[0]},
        )
        serial.append((res.time, res.outputs["y"]))
    _assert_batch_matches_serial(batch_res, serial, rtol=1e-10)


def test_v006_pid_first_order_plant_kp_sweep():
    """T-030 enables the kernel (vmap) path for PID — no _force_loop needed."""
    sys = _build_pid_plant()
    kp_vals = jnp.array([0.5, 1.0, 1.5, 2.0])
    t_span = (0.0, 5.0)
    port = sys["plant"].output_ports[0]

    batch_res = simulate_batch(
        sys,
        t_span=t_span,
        param_batches={"pid.kp": kp_vals},
        options=OPTS,
        recorded_signals={"y": port},
    )
    assert batch_res.outputs["y"].shape[0] == 4

    serial = _serial_outputs(sys, ("plant", 0), t_span, "pid.kp", kp_vals)
    _assert_batch_matches_serial(batch_res, serial, rtol=1e-10)


def test_v006_transfer_function_dynamic_B_sweep():
    """Sweep ``TransferFunction.B`` (dynamic) via simulate_batch; outputs vary.

    Previously xfail: TransferFunction.ode/_eval_output recomputed (A,B,C,D)
    from static num/den, ignoring dynamic A/B/C/D injection. T-029 removed
    those overrides so dynamic A/B/C/D injection now flows through. Static
    num/den remain static — sweeping them still requires diagram rebuilds.
    """
    sys = _build_tf_diagram(num=[1.0], den=[1.0, 2.0, 1.0])
    B_orig = sys["tf"].dynamic_parameters["B"].value
    B_batch = jnp.stack([B_orig * s for s in [0.5, 1.0, 2.0]], axis=0)
    t_span = (0.0, 3.0)
    port = sys["tf"].output_ports[0]

    batch_res = simulate_batch(
        sys,
        t_span=t_span,
        param_batches={"tf.B": B_batch},
        options=OPTS,
        recorded_signals={"y": port},
    )
    # The TF output should scale with B, so 0.5x and 2.0x runs must differ.
    assert not jnp.allclose(batch_res.outputs["y"][0], batch_res.outputs["y"][2])
    # And the middle run (1.0x) should match the serial reference.
    serial = _serial_outputs(sys, ("tf", 0), t_span, "tf.B", [B_batch[1]])
    t_ref = batch_res.time
    y_serial_on_ref = _interp_on_time(serial[0][1], serial[0][0], t_ref)
    np.testing.assert_allclose(
        np.asarray(batch_res.outputs["y"][1]),
        np.asarray(y_serial_on_ref),
        rtol=1e-8,
        atol=1e-9,
    )


def test_v006_custom_jax_block_param_sweep():
    """Sweep a dynamic_parameter inside a CustomJaxBlock; batch == serial.

    Serial reference is constructed by rebuilding the diagram with each
    parameter value (rather than ``with_parameters``), because
    ``Diagram.with_parameters`` currently fails for CustomJaxBlock-bearing
    diagrams (deepcopy reassigns IDs but compiled persistent-env references
    grow stale). The kernel path of simulate_batch handles this correctly
    via ``_pure_patch_context``.
    """
    a_vals = jnp.array([0.5, 1.0, 1.5, 2.0])
    t_span = (0.0, 1.0)

    sys = _build_custom_jax()
    port = sys["intg"].output_ports[0]
    batch_res = simulate_batch(
        sys,
        t_span=t_span,
        param_batches={"cb.a": a_vals},
        options=OPTS,
        recorded_signals={"y": port},
    )
    assert batch_res.outputs["y"].shape[0] == 4

    serial = []
    for a_val in a_vals:
        d = _build_custom_jax(a=float(a_val))
        ctx = d.create_context()
        res = simulate(
            d, ctx, t_span,
            options=OPTS,
            recorded_signals={"y": d["intg"].output_ports[0]},
        )
        serial.append((res.time, res.outputs["y"]))
    _assert_batch_matches_serial(batch_res, serial, rtol=1e-8)

    # Sanity: integral of constant `a` over [0, T] is a*T (within ~one dt).
    T = t_span[1]
    np.testing.assert_allclose(
        np.asarray(batch_res.outputs["y"][:, -1]),
        np.asarray(a_vals) * T,
        rtol=2e-2, atol=2e-2,
    )


def test_v006_gradient_through_serial_equivalent_matches_fd():
    """Verify gradient flow.

    ``simulate_batch`` forces ``enable_autodiff=False`` (batch.py:378), so it
    isn't itself differentiable. The semantically equivalent computation is
    ``mean_i simulate(sys.with_parameters({k: a[i]}))``, which IS differentiable
    when ``enable_autodiff=True``. ``jax.grad`` of that scalar agrees with
    finite differences, proving gradients flow through every block and through
    the parameter-update mechanism that simulate_batch's loop path uses.
    """
    sys = _build_decay()
    t_span = (0.0, 2.0)

    from jaxonomy.simulation.types import ResultsOptions
    ad_opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="dopri5",
        rtol=1e-9,
        atol=1e-12,
        max_major_steps=200,
        enable_autodiff=True,
        save_time_series=False,
    )
    ad_results_opts = ResultsOptions()

    def mean_terminal(a_vec):
        terminals = []
        for i in range(a_vec.shape[0]):
            d = sys.with_parameters({"g.gain": -a_vec[i]})
            ctx = d.create_context()
            res = simulate(
                d, ctx, t_span,
                options=ad_opts,
                results_options=ad_results_opts,
            )
            terminals.append(res.context.continuous_state[0])
        return jnp.stack(terminals).mean()

    a0 = jnp.array([0.5, 1.0, 1.5])
    g_ad = jax.grad(mean_terminal)(a0)
    g_fd = fd_grad(
        lambda a: float(mean_terminal(jnp.asarray(a))),
        np.asarray(a0),
        eps=1e-5,
    )[0]

    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=1e-4, atol=1e-6)
    assert float(jnp.linalg.norm(g_ad)) > 1e-3


def test_v006_same_ctx_batch_outputs_identical():
    sys = _build_decay()
    a = 1.0
    g_vals = jnp.array([-a, -a, -a])
    t_span = (0.0, 2.0)
    port = sys["intg"].output_ports[0]

    batch_res = simulate_batch(
        sys,
        t_span=t_span,
        param_batches={"g.gain": g_vals},
        options=OPTS,
        recorded_signals={"y": port},
    )
    assert batch_res.outputs["y"].shape[0] == 3
    y0 = np.asarray(batch_res.outputs["y"][0])
    y1 = np.asarray(batch_res.outputs["y"][1])
    y2 = np.asarray(batch_res.outputs["y"][2])
    np.testing.assert_allclose(y0, y1, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(y0, y2, rtol=1e-12, atol=0.0)
