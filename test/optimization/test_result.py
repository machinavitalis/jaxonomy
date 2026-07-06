# SPDX-License-Identifier: MIT

"""Tests for OptimizationResult — dict interface, fields, and optimizer outputs."""

import numpy as np
import pytest

import jax.numpy as jnp

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Constant, Gain, Integrator, Power
from jaxonomy.optimization import (
    Optimizable, Optax, Scipy, OptimizationResult,
)

pytestmark = pytest.mark.slow


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_simple_optimizable(c0=0.5):
    """1-param spring-mass damped oscillator, same as in test_scenarios.py."""
    params = {"c": Parameter(np.array(1.0))}
    b = DiagramBuilder()
    k_x   = b.add(Gain(1.0,          name="k_x"))
    c_v   = b.add(Gain(params["c"],  name="c_v"))
    add   = b.add(Adder(2, operators="--", name="adder"))
    inv   = b.add(Gain(1.0,          name="inv_m"))
    v     = b.add(Integrator(0.1,    name="v"))
    x     = b.add(Integrator(1.0,    name="x"))
    b.connect(k_x.output_ports[0], add.input_ports[0])
    b.connect(c_v.output_ports[0], add.input_ports[1])
    b.connect(add.output_ports[0], inv.input_ports[0])
    b.connect(inv.output_ports[0], v.input_ports[0])
    b.connect(v.output_ports[0],   x.input_ports[0])
    b.connect(v.output_ports[0],   c_v.input_ports[0])
    b.connect(x.output_ports[0],   k_x.input_ports[0])
    sq_v = b.add(Power(2.0, name="sq_v"))
    sq_x = b.add(Power(2.0, name="sq_x"))
    cv   = b.add(Integrator(0.0, name="cost_v"))
    cx   = b.add(Integrator(0.0, name="cost_x"))
    obj  = b.add(Adder(2, operators="++", name="obj"))
    b.connect(v.output_ports[0], sq_v.input_ports[0])
    b.connect(x.output_ports[0], sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0], obj.input_ports[0])
    b.connect(cx.output_ports[0], obj.input_ports[1])
    diagram = b.build(parameters=params)

    class _Opt(Optimizable):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._obj_port = diagram["obj"].output_ports[0]
        def optimizable_params(self, ctx):
            return {"c": ctx.parameters["c"]}
        def objective_from_context(self, ctx):
            return self._obj_port.eval(ctx)
        def prepare_context(self, ctx, params):
            return ctx.with_parameters(params)

    return _Opt(
        diagram, diagram.create_context(),
        params_0={"c": c0},
        sim_t_span=(0.0, 2.0),
        sim_options=SimulatorOptions(max_major_steps=1),
    )


# ── unit tests for OptimizationResult ────────────────────────────────────────

class TestOptimizationResultDictInterface:
    """Verify the backward-compatible dict interface."""

    def _make_result(self):
        return OptimizationResult(
            params={"a": 1.0, "b": jnp.array(2.0)},
            success=True,
            nit=10,
            nfev=50,
            message="converged",
            final_loss=0.01,
            loss_history=[1.0, 0.5, 0.1, 0.01],
        )

    def test_getitem(self):
        r = self._make_result()
        assert r["a"] == 1.0
        assert float(r["b"]) == 2.0

    def test_setitem(self):
        r = self._make_result()
        r["c"] = 3.0
        assert r["c"] == 3.0

    def test_contains(self):
        r = self._make_result()
        assert "a" in r
        assert "z" not in r

    def test_iter(self):
        r = self._make_result()
        keys = list(r)
        assert "a" in keys and "b" in keys

    def test_len(self):
        r = self._make_result()
        assert len(r) == 2

    def test_items(self):
        r = self._make_result()
        d = dict(r.items())
        assert d["a"] == 1.0

    def test_keys(self):
        r = self._make_result()
        assert set(r.keys()) == {"a", "b"}

    def test_values(self):
        r = self._make_result()
        vals = list(r.values())
        assert 1.0 in vals

    def test_get_existing(self):
        r = self._make_result()
        assert r.get("a") == 1.0

    def test_get_default(self):
        r = self._make_result()
        assert r.get("z", 99) == 99

    def test_repr_contains_key_fields(self):
        r = self._make_result()
        s = repr(r)
        assert "success=True" in s
        assert "nit=10" in s
        assert "final_loss=0.01" in s

    def test_fields_accessible(self):
        r = self._make_result()
        assert r.success is True
        assert r.nit == 10
        assert r.nfev == 50
        assert r.message == "converged"
        assert r.final_loss == pytest.approx(0.01)
        assert r.loss_history == [1.0, 0.5, 0.1, 0.01]


class TestOptimizationResultFromOptimizers:
    """Verify that real optimizers return OptimizationResult with correct fields."""

    def test_scipy_returns_optimization_result(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20},
                      use_autodiff_grad=True)
        r = optim.optimize()
        assert isinstance(r, OptimizationResult)

    def test_scipy_has_success_field(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20},
                      use_autodiff_grad=True)
        r = optim.optimize()
        assert isinstance(r.success, bool)

    def test_scipy_has_nit_and_nfev(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20},
                      use_autodiff_grad=True)
        r = optim.optimize()
        assert r.nit >= 0
        assert r.nfev >= 0

    def test_scipy_has_final_loss(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20},
                      use_autodiff_grad=True)
        r = optim.optimize()
        assert r.final_loss is not None
        assert np.isfinite(r.final_loss)
        assert r.final_loss > 0.0

    def test_scipy_has_message(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20})
        r = optim.optimize()
        assert isinstance(r.message, str)
        assert len(r.message) > 0

    def test_scipy_dict_access_still_works(self):
        """Backward-compat: result['c'] must still work."""
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20},
                      use_autodiff_grad=True)
        r = optim.optimize()
        # This is how all legacy tests access results:
        assert float(r["c"]) == pytest.approx(1.65, abs=0.15)

    def test_scipy_final_loss_less_than_initial(self):
        """Optimizer must reduce the objective."""
        model = _make_simple_optimizable(c0=0.5)
        initial_loss = float(model.objective_flat(model.params_0_flat))
        optim = Scipy(model, "L-BFGS-B", opt_method_config={"maxiter": 20},
                      use_autodiff_grad=True)
        r = optim.optimize()
        assert r.final_loss < initial_loss

    def test_optax_returns_optimization_result(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Optax(model, "adam", learning_rate=0.005,
                      opt_method_config={}, num_epochs=200)
        r = optim.optimize()
        assert isinstance(r, OptimizationResult)

    def test_optax_has_loss_history(self):
        model = _make_simple_optimizable(c0=0.5)
        n_epochs = 100
        optim = Optax(model, "adam", learning_rate=0.005,
                      opt_method_config={}, num_epochs=n_epochs)
        r = optim.optimize()
        assert len(r.loss_history) == n_epochs
        assert all(isinstance(v, float) for v in r.loss_history)

    def test_optax_nit_equals_epochs(self):
        model = _make_simple_optimizable(c0=0.5)
        n_epochs = 77
        optim = Optax(model, "adam", learning_rate=0.005,
                      opt_method_config={}, num_epochs=n_epochs)
        r = optim.optimize()
        assert r.nit == n_epochs

    def test_optax_loss_history_decreasing_trend(self):
        """Loss history should decrease overall (not necessarily monotonically)."""
        model = _make_simple_optimizable(c0=0.5)
        optim = Optax(model, "adam", learning_rate=0.005,
                      opt_method_config={}, num_epochs=500)
        r = optim.optimize()
        # First quarter average should be > last quarter average
        n = len(r.loss_history)
        avg_start = np.mean(r.loss_history[: n // 4])
        avg_end = np.mean(r.loss_history[3 * n // 4 :])
        assert avg_end < avg_start, (
            f"Loss did not decrease: start avg={avg_start:.4f}, "
            f"end avg={avg_end:.4f}"
        )

    def test_optax_dict_access_still_works(self):
        model = _make_simple_optimizable(c0=0.5)
        optim = Optax(model, "adam", learning_rate=0.005,
                      opt_method_config={}, num_epochs=500)
        r = optim.optimize()
        _ = float(r["c"])  # must not raise

    def test_scipy_nelder_mead_has_fields(self):
        """Gradient-free Scipy also fills result fields."""
        model = _make_simple_optimizable(c0=0.5)
        optim = Scipy(model, "Nelder-Mead", opt_method_config={"maxiter": 50})
        r = optim.optimize()
        assert isinstance(r, OptimizationResult)
        assert isinstance(r.success, bool)
        assert r.nfev > 0
