# SPDX-License-Identifier: MIT

"""T-133: per-block multirate substepping under the fixed-step RK4 solver.

``declare_continuous_state(substeps=N)`` advances a stiff block's states
with N inner RK4 steps of ``h/N`` while the rest of the diagram takes one
step of ``h``, with zero-order-hold coupling at the interface. This is
the framework version of the JIT-safe substep loop stiff blocks
previously hand-rolled (BLDC windings, series-elastic joints, cables).

The canonical stiffness fixture: a fast first-order tracker
``dx/dt = -λ(x - u)`` with λh far beyond RK4's stability limit (λh ≲
2.78), driven by a slow decay ``du/dt = -u``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.simulation import SimulatorOptions

pytestmark = pytest.mark.minimal

LAM = 2000.0
H = 0.005  # λh = 10: hopelessly unstable for single-rate RK4


class SlowDecay(jaxonomy.LeafSystem):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self.ode)
        self.declare_continuous_state_output(name="u")

    def ode(self, t, state, *inputs, **params):
        return -state.continuous_state


class FastTracker(jaxonomy.LeafSystem):
    def __init__(self, substeps=1, lam=LAM, name=None):
        super().__init__(name=name)
        self.declare_dynamic_parameter("lam", jnp.asarray(lam))
        self.declare_input_port(name="u")
        self.declare_continuous_state(
            default_value=jnp.array(0.0), ode=self.ode, substeps=substeps
        )
        self.declare_continuous_state_output(name="x")

    def ode(self, t, state, u, **params):
        return -params["lam"] * (state.continuous_state - u)


def _build(substeps):
    b = jaxonomy.DiagramBuilder()
    slow = b.add(SlowDecay(name="slow"))
    fast = b.add(FastTracker(substeps=substeps, name="fast"))
    b.connect(slow.output_ports[0], fast.input_ports[0])
    return b.build()


def _run(diagram, t_final=1.0, **opt_kwargs):
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        max_minor_step_size=H,
        **opt_kwargs,
    )
    ctx = diagram.create_context()
    return jaxonomy.simulate(
        diagram, ctx, (0.0, t_final), options=opts,
        recorded_signals={
            "x": diagram["fast"].output_ports[0],
            "u": diagram["slow"].output_ports[0],
        },
    )


def test_single_rate_blows_up_substepped_is_stable():
    """The motivating contrast: λh = 10 diverges at substeps=1 and is
    stable and accurate at substeps=8 (λh/N = 1.25)."""
    res1 = _run(_build(substeps=1))
    x1 = np.asarray(res1.outputs["x"])
    assert not np.all(np.isfinite(x1)), (
        "expected the single-rate run to diverge — if RK4 got more stable, "
        "raise LAM to keep this fixture meaningful"
    )

    res8 = _run(_build(substeps=8))
    x8 = np.asarray(res8.outputs["x"])
    u8 = np.asarray(res8.outputs["u"])
    assert np.all(np.isfinite(x8))
    # After the fast transient, x tracks u with the O(h) ZOH coupling lag.
    lag = np.max(np.abs(x8[20:] - u8[20:]))
    assert lag < 3.0 * H, f"coupling lag {lag:.3e} exceeds the O(h) bound"
    # The slow state is integrated at full RK4 accuracy, unpolluted by
    # the (frozen) fast entries.
    assert float(u8[-1]) == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_substeps_default_is_single_rate():
    """substeps=1 (the default) takes the legacy single-rate code path;
    a benign diagram integrates identically either way."""
    b1 = jaxonomy.DiagramBuilder()
    s1 = b1.add(SlowDecay(name="slow"))
    d1 = b1.build()
    res = jaxonomy.simulate(
        d1, d1.create_context(), (0.0, 1.0),
        options=SimulatorOptions(
            math_backend="jax", ode_solver_method="rk4",
            max_minor_step_size=0.01,
        ),
        recorded_signals={"u": s1.output_ports[0]},
    )
    assert float(np.asarray(res.outputs["u"])[-1]) == pytest.approx(
        np.exp(-1.0), rel=1e-8
    )


def test_substeps_validation():
    with pytest.raises(ValueError, match="static Python int"):
        FastTracker(substeps=0)
    with pytest.raises(ValueError, match="static Python int"):
        FastTracker(substeps=2.5)


def test_substep_vector_alignment():
    """The flat factor vector aligns with the flattened continuous state
    (slow scalar first, fast scalar second, in leaf_systems order)."""
    d = _build(substeps=4)
    assert d.has_multirate_substeps
    leaves = jax.tree_util.tree_leaves(d.continuous_substep_vector)
    flat = np.concatenate([np.ravel(v) for v in leaves])
    assert flat.tolist() == [1, 4]


def test_multiple_fast_groups():
    """Two stiff blocks with different factors substep independently."""
    b = jaxonomy.DiagramBuilder()
    slow = b.add(SlowDecay(name="slow"))
    fast_a = b.add(FastTracker(substeps=8, lam=2000.0, name="fast"))
    fast_b = b.add(FastTracker(substeps=4, lam=1000.0, name="fast_b"))
    b.connect(slow.output_ports[0], fast_a.input_ports[0])
    b.connect(slow.output_ports[0], fast_b.input_ports[0])
    d = b.build()
    res = jaxonomy.simulate(
        d, d.create_context(), (0.0, 1.0),
        options=SimulatorOptions(
            math_backend="jax", ode_solver_method="rk4",
            max_minor_step_size=H,
        ),
        recorded_signals={
            "xa": fast_a.output_ports[0],
            "xb": fast_b.output_ports[0],
            "u": slow.output_ports[0],
        },
    )
    u = np.asarray(res.outputs["u"])
    for key in ("xa", "xb"):
        x = np.asarray(res.outputs[key])
        assert np.all(np.isfinite(x))
        assert np.max(np.abs(x[20:] - u[20:])) < 3.0 * H


def test_vmap_batch_with_substeps():
    """The substep loop is a static-bound fori_loop — the vmapped batch
    kernel must compile and every row stay finite."""
    import warnings

    d = _build(substeps=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = jaxonomy.simulate_batch(
            d,
            (0.0, 0.5),
            {"fast.lam": jnp.array([1000.0, 1500.0, 2000.0])},
            options=SimulatorOptions(
                math_backend="jax", ode_solver_method="rk4",
                max_minor_step_size=H, max_major_steps=200,
            ),
            recorded_signals={"x": d["fast"].output_ports[0]},
            use_vmap=True,
        )
    x = np.asarray(batch.outputs["x"])
    assert x.shape[0] == 3
    assert np.all(np.isfinite(x))


def _grad_fixture(substeps, lam, h=H):
    b = jaxonomy.DiagramBuilder()
    slow = b.add(SlowDecay(name="slow"))
    fast = b.add(FastTracker(substeps=substeps, lam=lam, name="fast"))
    b.connect(slow.output_ports[0], fast.input_ports[0])
    d = b.build()
    fast_id = d["fast"].system_id
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        max_minor_step_size=h,
        enable_autodiff=True,
    )
    base_ctx = d.create_context()

    def fwd(lam_val, context):
        sub = context[fast_id].with_parameter("lam", lam_val)
        context = context.with_subcontext(fast_id, sub)
        res = jaxonomy.simulate(d, context, (0.0, 0.1), options=opts)
        return res.context[fast_id].continuous_state

    return fwd, base_ctx


def _adjoint_vs_fd_ratio(substeps, lam, h):
    fwd, base_ctx = _grad_fixture(substeps=substeps, lam=lam, h=h)
    vg = jax.jit(jax.value_and_grad(fwd))
    _, grad = vg(jnp.float64(lam), base_ctx)
    fwd_jit = jax.jit(fwd)
    dh = 0.1
    fd = (
        float(fwd_jit(jnp.float64(lam + dh), base_ctx))
        - float(fwd_jit(jnp.float64(lam - dh), base_ctx))
    ) / (2 * dh)
    return float(grad) / fd


def test_grad_single_rate_matches_fd():
    """substeps=1 keeps the exact monolithic adjoint: FD match to <0.1%."""
    ratio = _adjoint_vs_fd_ratio(substeps=1, lam=100.0, h=H)
    assert ratio == pytest.approx(1.0, abs=1e-3)


def test_grad_multirate_coupling_error_is_first_order():
    """The multirate adjoint is *consistent*: its deviation from FD is the
    O(h) ZOH coupling error (same order as the forward scheme's), so it
    must shrink linearly as the outer step is refined. Measured on this
    fixture: |ratio-1| = 0.50 → 0.27 → 0.14 for h = 5e-3 → 2.5e-3 →
    1.25e-3 (2026-07-11)."""
    err_coarse = abs(_adjoint_vs_fd_ratio(substeps=4, lam=100.0, h=0.005) - 1.0)
    err_fine = abs(_adjoint_vs_fd_ratio(substeps=4, lam=100.0, h=0.00125) - 1.0)
    assert np.isfinite(err_coarse) and np.isfinite(err_fine)
    assert err_fine < 0.6 * err_coarse, (
        f"adjoint coupling error not converging: |ratio-1| {err_coarse:.3f} "
        f"at h=5e-3 vs {err_fine:.3f} at h=1.25e-3"
    )
    assert err_fine < 0.2


def test_grad_through_stiff_substepped_rk4_is_finite():
    """For dynamics genuinely unstable at the outer step (λh = 10), the
    checkpointed continuous adjoint re-integrates the primal in reverse
    time, where the fast mode grows ~e^{λh} within each (checkpoint-reset)
    step — an inherent stiff-adjoint accuracy limit, not a multirate bug.
    The structural contract pinned here: the substepped adjoint pass
    compiles and returns finite values (no crash, no NaN). For accurate
    stiff gradients, reduce the outer step or use forward-mode."""
    fwd, base_ctx = _grad_fixture(substeps=8, lam=LAM)
    vg = jax.jit(jax.value_and_grad(fwd))
    value, grad = vg(jnp.float64(LAM), base_ctx)
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
