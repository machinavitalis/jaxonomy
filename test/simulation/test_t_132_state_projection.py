# SPDX-License-Identifier: MIT

"""T-132: declared per-block state projection (manifold retraction).

``declare_continuous_state(project=fn)`` supplies the retraction back
onto the state's manifold, applied by the simulator at the end of every
major step. The canonical case: unit-quaternion attitude kinematics
``q̇ = ½ q ⊗ [0, ω]`` — integrated componentwise, ``‖q‖`` drifts off 1
under any one-step integrator; with the declared renormalization the
recorded trajectory stays on the unit sphere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.simulation import SimulatorOptions
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.minimal

OMEGA = 4.0  # rad/s about the z axis
T_FINAL = 10.0


def _normalize(q):
    return q / jnp.linalg.norm(q)


class QuaternionKinematics(jaxonomy.LeafSystem):
    """Rigid-body attitude: q̇ = ½ q ⊗ [0, ω] with constant body rate."""

    def __init__(self, project=None, name=None):
        super().__init__(name=name)
        self.declare_dynamic_parameter("wz", jnp.float64(OMEGA))
        q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        self.declare_continuous_state(default_value=q0, ode=self.ode, project=project)
        self.declare_continuous_state_output(name="q")

    def ode(self, t, state, **params):
        w, x, y, z = state.continuous_state
        wz = params["wz"]
        # q̇ = ½ q ⊗ (0, 0, 0, wz)
        return 0.5 * jnp.array([-z * wz, -y * wz, x * wz, w * wz])


def _run(project, method="rk4", t_final=T_FINAL, **opt_kwargs):
    model = QuaternionKinematics(project=project)
    defaults = dict(math_backend="jax", ode_solver_method=method)
    if method == "rk4":
        defaults["max_minor_step_size"] = 0.05  # fat step → visible drift
    defaults.update(opt_kwargs)
    opts = SimulatorOptions(**defaults)
    ctx = model.create_context()
    res = jaxonomy.simulate(
        model, ctx, (0.0, t_final), options=opts,
        recorded_signals={"q": model.output_ports[0]},
    )
    return res


def test_unprojected_quaternion_drifts():
    """The motivating defect: componentwise integration leaves the unit
    sphere. (If this stops drifting, fatten the step to keep the fixture
    meaningful.)"""
    res = _run(project=None)
    q = np.asarray(res.outputs["q"])
    norms = np.linalg.norm(q, axis=1)
    assert np.max(np.abs(norms - 1.0)) > 1e-9


def test_projected_quaternion_stays_unit_norm():
    res = _run(project=_normalize)
    q = np.asarray(res.outputs["q"])
    norms = np.linalg.norm(q, axis=1)
    # Recorded samples land on major-step boundaries, where the declared
    # retraction has just been applied.
    assert np.max(np.abs(norms - 1.0)) < 1e-12

    # And the attitude is *right*, not just normalized: rotation about z
    # by OMEGA*t → q = [cos(θ/2), 0, 0, sin(θ/2)].
    t = np.asarray(res.time)
    expected_w = np.cos(OMEGA * t / 2)
    expected_z = np.sin(OMEGA * t / 2)
    np.testing.assert_allclose(q[:, 0], expected_w, atol=5e-4)
    np.testing.assert_allclose(q[:, 3], expected_z, atol=5e-4)


@pytest.mark.parametrize("method", ["auto", "bdf"])
def test_projection_applies_under_adaptive_solvers(method):
    """Adaptive solvers apply the retraction at major-step boundaries
    (per-solver-step retraction is exact only for the one-step rk4; BDF's
    multistep history cannot be retracted mid-stride). Contract: the
    final state is exactly on the manifold, and within-step recorded
    drift stays at the solver's own error level."""
    res = _run(project=_normalize, method=method, t_final=2.0)
    q = np.asarray(res.outputs["q"])
    norms = np.linalg.norm(q, axis=1)
    assert abs(norms[-1] - 1.0) < 1e-12, "boundary retraction missing"
    assert np.max(np.abs(norms - 1.0)) < 1e-4, "drift beyond solver error"


def test_projection_composes_with_substeps():
    """A projected state can also declare multirate substeps."""

    class FastQuat(QuaternionKinematics):
        def __init__(self):
            jaxonomy.LeafSystem.__init__(self, name="fastq")
            self.declare_dynamic_parameter("wz", jnp.float64(40.0))
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 0.0, 0.0, 0.0]),
                ode=self.ode,
                project=_normalize,
                substeps=4,
            )
            self.declare_continuous_state_output(name="q")

    model = FastQuat()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.05
    )
    res = jaxonomy.simulate(
        model, model.create_context(), (0.0, 2.0), options=opts,
        recorded_signals={"q": model.output_ports[0]},
    )
    q = np.asarray(res.outputs["q"])
    assert np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)) < 1e-12


def test_projection_in_diagram_with_unprojected_sibling():
    """Only the declaring block is projected; siblings are untouched."""
    b = jaxonomy.DiagramBuilder()
    quat = b.add(QuaternionKinematics(project=_normalize, name="quat"))
    integ = b.add(jaxonomy.library.Integrator(0.0, name="integ"))
    src = b.add(jaxonomy.library.Constant(1.0, name="src"))
    b.connect(src.output_ports[0], integ.input_ports[0])
    d = b.build()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="rk4", max_minor_step_size=0.05
    )
    res = jaxonomy.simulate(
        d, d.create_context(), (0.0, 5.0), options=opts,
        recorded_signals={
            "q": d["quat"].output_ports[0],
            "y": d["integ"].output_ports[0],
        },
    )
    q = np.asarray(res.outputs["q"])
    assert np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)) < 1e-12
    # The ramp integrates exactly — untouched by the sibling's projection.
    t = np.asarray(res.time)
    np.testing.assert_allclose(np.asarray(res.outputs["y"]), t, atol=1e-9)


def test_grad_through_projected_state():
    """Reverse-mode AD flows through the projection (ordinary traced
    ops): d(q_w(T))/d(wz) matches FD."""
    model = QuaternionKinematics(project=_normalize)
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        max_minor_step_size=0.05,
        enable_autodiff=True,
    )
    base_ctx = model.create_context()

    def fwd(wz, context):
        context = context.with_parameter("wz", wz)
        res = jaxonomy.simulate(model, context, (0.0, 1.0), options=opts)
        return res.context.continuous_state[0]

    vg = jax.jit(jax.value_and_grad(fwd))
    value, grad = vg(jnp.float64(OMEGA), base_ctx)
    # Analytic: q_w(T) = cos(wz*T/2) → d/dwz = -T/2 sin(wz*T/2).
    assert float(value) == pytest.approx(np.cos(OMEGA / 2), abs=1e-4)

    f = jax.jit(fwd)
    eps = 1e-5
    fd = (
        float(f(jnp.float64(OMEGA + eps), base_ctx))
        - float(f(jnp.float64(OMEGA - eps), base_ctx))
    ) / (2 * eps)
    assert float(grad) == pytest.approx(fd, rel=1e-4)
    assert float(grad) == pytest.approx(-0.5 * np.sin(OMEGA / 2), abs=1e-3)


def test_project_must_be_callable():
    with pytest.raises(ValueError, match="callable"):
        QuaternionKinematics(project="normalize")