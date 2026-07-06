# SPDX-License-Identifier: MIT

"""Tests for `jaxonomy.optimization.tune_parameters`.

We use small known-answer systems so the tests are fast and the optimum is
easy to verify.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power
from jaxonomy.optimization import tune_parameters, TuningResult


pytestmark = pytest.mark.slow


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _build_spring_mass_damper(c0=0.5, k0=1.0):
    """A second-order plant with one tunable damping coefficient `c` and
    fixed spring `k`. The objective is the integrated squared position +
    velocity over a short window, so the optimum chooses a `c` close to
    critical damping. We expose `c` as a `Parameter` so it can be tuned.
    """
    params = {"c": Parameter(np.array(c0)), "k": Parameter(np.array(k0))}
    b = DiagramBuilder()
    k_x = b.add(Gain(params["k"], name="k_x"))
    c_v = b.add(Gain(params["c"], name="c_v"))
    add = b.add(Adder(2, operators="--", name="adder"))
    inv = b.add(Gain(1.0, name="inv_m"))
    v = b.add(Integrator(0.1, name="v"))   # initial vel 0.1
    x = b.add(Integrator(1.0, name="x"))   # initial pos 1.0
    b.connect(k_x.output_ports[0], add.input_ports[0])
    b.connect(c_v.output_ports[0], add.input_ports[1])
    b.connect(add.output_ports[0], inv.input_ports[0])
    b.connect(inv.output_ports[0], v.input_ports[0])
    b.connect(v.output_ports[0], x.input_ports[0])
    b.connect(v.output_ports[0], c_v.input_ports[0])
    b.connect(x.output_ports[0], k_x.input_ports[0])

    # ISE-style objective integrator
    sq_v = b.add(Power(2.0, name="sq_v"))
    sq_x = b.add(Power(2.0, name="sq_x"))
    cv = b.add(Integrator(0.0, name="cv"))
    cx = b.add(Integrator(0.0, name="cx"))
    obj = b.add(Adder(2, operators="++", name="obj"))
    b.connect(v.output_ports[0], sq_v.input_ports[0])
    b.connect(x.output_ports[0], sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0], obj.input_ports[0])
    b.connect(cx.output_ports[0], obj.input_ports[1])

    diagram = b.build(parameters=params)
    return diagram


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_tune_parameters_returns_correct_type():
    """Smoke test: the API returns a TuningResult with the requested keys."""
    diagram = _build_spring_mass_damper(c0=0.1, k0=1.0)
    obj_port = diagram["obj"].output_ports[0]

    result = tune_parameters(
        diagram=diagram,
        base_context=diagram.create_context(),
        sim_t_span=(0.0, 2.0),
        params_0={"c": 0.1},
        set_params=lambda ctx, p: ctx.with_parameters({"c": p["c"]}),
        objective_fn=lambda ctx: obj_port.eval(ctx),
        bounds={"c": (0.01, 5.0)},
        optimizer="scipy-lbfgs",
        n_iter=20,
        sim_options=SimulatorOptions(max_major_steps=1, enable_autodiff=True),
        verbose=False,
    )

    assert isinstance(result, TuningResult)
    assert "c" in result.params
    assert np.isfinite(result.objective)


def test_tune_parameters_improves_objective():
    """Optimizing `c` from a poor starting point should strictly reduce the
    integrated-squared-state objective."""
    diagram = _build_spring_mass_damper(c0=0.01, k0=1.0)
    obj_port = diagram["obj"].output_ports[0]

    # Evaluate the objective at the initial poor `c` for a baseline.
    ctx0 = diagram.create_context().with_parameters({"c": np.array(0.01)})
    from jaxonomy.simulation import Simulator
    sim = Simulator(diagram, options=SimulatorOptions(max_major_steps=1))
    baseline = float(obj_port.eval(sim.advance_to(2.0, ctx0).context))

    result = tune_parameters(
        diagram=diagram,
        base_context=diagram.create_context(),
        sim_t_span=(0.0, 2.0),
        params_0={"c": 0.01},
        set_params=lambda ctx, p: ctx.with_parameters({"c": p["c"]}),
        objective_fn=lambda ctx: obj_port.eval(ctx),
        bounds={"c": (0.001, 5.0)},
        optimizer="scipy-lbfgs",
        n_iter=30,
        sim_options=SimulatorOptions(max_major_steps=1, enable_autodiff=True),
        verbose=False,
    )

    # Tuned `c` should lie strictly between bounds and reduce the objective.
    c_opt = float(result.params["c"])
    assert 0.001 <= c_opt <= 5.0, f"c_opt {c_opt} out of bounds"
    assert result.objective < baseline, (
        f"objective did not improve: baseline={baseline}, final={result.objective}"
    )


def test_tune_parameters_respects_bounds():
    """The optimizer should not return a value outside the requested bounds."""
    diagram = _build_spring_mass_damper(c0=0.1, k0=1.0)
    obj_port = diagram["obj"].output_ports[0]

    # Bounds chosen tight enough that optimum without bounds would lie outside.
    lo, hi = 0.05, 0.15
    result = tune_parameters(
        diagram=diagram,
        base_context=diagram.create_context(),
        sim_t_span=(0.0, 2.0),
        params_0={"c": 0.1},
        set_params=lambda ctx, p: ctx.with_parameters({"c": p["c"]}),
        objective_fn=lambda ctx: obj_port.eval(ctx),
        bounds={"c": (lo, hi)},
        optimizer="scipy-lbfgs",
        n_iter=30,
        sim_options=SimulatorOptions(max_major_steps=1, enable_autodiff=True),
        verbose=False,
    )

    c_opt = float(result.params["c"])
    assert lo - 1e-6 <= c_opt <= hi + 1e-6, (
        f"c_opt {c_opt} outside [{lo}, {hi}]"
    )


def test_tune_parameters_multiple_params():
    """Two-parameter tuning: simultaneously optimize damping and spring."""
    diagram = _build_spring_mass_damper(c0=0.1, k0=0.5)
    obj_port = diagram["obj"].output_ports[0]

    result = tune_parameters(
        diagram=diagram,
        base_context=diagram.create_context(),
        sim_t_span=(0.0, 2.0),
        params_0={"c": 0.1, "k": 0.5},
        set_params=lambda ctx, p: ctx.with_parameters({"c": p["c"], "k": p["k"]}),
        objective_fn=lambda ctx: obj_port.eval(ctx),
        bounds={"c": (0.001, 5.0), "k": (0.1, 5.0)},
        optimizer="scipy-lbfgs",
        n_iter=30,
        sim_options=SimulatorOptions(max_major_steps=1, enable_autodiff=True),
        verbose=False,
    )

    assert set(result.params.keys()) == {"c", "k"}
    assert all(np.isfinite(v) for v in result.params.values())


def test_tune_parameters_unknown_optimizer_raises():
    """Unknown optimizer names should fail loudly, not silently."""
    diagram = _build_spring_mass_damper()
    obj_port = diagram["obj"].output_ports[0]
    with pytest.raises(ValueError, match="Unknown optimizer"):
        tune_parameters(
            diagram=diagram,
            base_context=diagram.create_context(),
            sim_t_span=(0.0, 1.0),
            params_0={"c": 0.1},
            set_params=lambda ctx, p: ctx.with_parameters({"c": p["c"]}),
            objective_fn=lambda ctx: obj_port.eval(ctx),
            optimizer="not-a-real-optimizer",
            verbose=False,
        )


if __name__ == "__main__":
    test_tune_parameters_returns_correct_type()
    test_tune_parameters_improves_objective()
    test_tune_parameters_respects_bounds()
    test_tune_parameters_multiple_params()
    test_tune_parameters_unknown_optimizer_raises()
    print("All tune_parameters tests passed.")
