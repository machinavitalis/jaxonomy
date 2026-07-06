# SPDX-License-Identifier: MIT

"""Tests for the MultiStart wrapper."""

import numpy as np
import pytest
import jax.numpy as jnp

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power
from jaxonomy.optimization import (
    Optimizable, Scipy, Optax, OptimizationResult,
    MultiStart, MultiStartResult,
)

pytestmark = pytest.mark.slow


# ── fixtures ─────────────────────────────────────────────────────────────────

def _make_spring_optimizable(c0=0.5):
    """Damped oscillator: minimise ISE, optimal c ≈ 1.65."""
    params = {"c": Parameter(np.array(1.0))}
    b = DiagramBuilder()
    k_x  = b.add(Gain(1.0,         name="k_x"))
    c_v  = b.add(Gain(params["c"], name="c_v"))
    add  = b.add(Adder(2, operators="--", name="adder"))
    inv  = b.add(Gain(1.0,         name="inv_m"))
    v    = b.add(Integrator(0.1,   name="v"))
    x    = b.add(Integrator(1.0,   name="x"))
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
    b.connect(v.output_ports[0],   sq_v.input_ports[0])
    b.connect(x.output_ports[0],   sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0],  obj.input_ports[0])
    b.connect(cx.output_ports[0],  obj.input_ports[1])
    diagram = b.build(parameters=params)

    class _Opt(Optimizable):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._obj = diagram["obj"].output_ports[0]
        def optimizable_params(self, ctx): return {"c": ctx.parameters["c"]}
        def objective_from_context(self, ctx): return self._obj.eval(ctx)
        def prepare_context(self, ctx, p): return ctx.with_parameters(p)

    return _Opt(
        diagram, diagram.create_context(),
        params_0={"c": c0},
        sim_t_span=(0.0, 2.0),
        sim_options=SimulatorOptions(max_major_steps=1),
    )


def _lbfgsb_factory(optimizable):
    return Scipy(optimizable, "L-BFGS-B",
                 opt_method_config={"maxiter": 30},
                 use_autodiff_grad=True)


# ── unit tests ────────────────────────────────────────────────────────────────

class TestMultiStartBasics:

    def test_returns_multi_start_result(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=3, seed=0)
        result = ms.run()
        assert isinstance(result, MultiStartResult)

    def test_result_has_n_starts_entries(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=4, seed=1)
        result = ms.run()
        assert len(result.results) == 4
        assert result.n_starts == 4

    def test_best_result_is_optimization_result(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=3, seed=2)
        result = ms.run()
        assert isinstance(result.best_result, OptimizationResult)

    def test_best_index_is_valid(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=3, seed=3)
        result = ms.run()
        assert 0 <= result.best_start_index < 3

    def test_best_result_has_lowest_loss(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=4, seed=4)
        result = ms.run()
        losses = [
            r.final_loss for r in result.results
            if r.success and r.final_loss is not None
        ]
        assert result.best_result.final_loss == pytest.approx(min(losses))

    def test_n_successful_count(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=3, seed=5)
        result = ms.run()
        assert result.n_successful == sum(1 for r in result.results if r.success)

    def test_include_initial_true(self):
        """First start must use the original params_0 when include_initial=True."""
        c0 = 0.9
        model = _make_spring_optimizable(c0=c0)
        starts_seen = []

        def factory(opt):
            starts_seen.append(float(opt.params_0_flat[0]))
            return _lbfgsb_factory(opt)

        ms = MultiStart(model, factory, n_starts=3, seed=6, include_initial=True)
        ms.run()
        assert starts_seen[0] == pytest.approx(c0, abs=1e-6), (
            f"Expected first start c0={c0}, got {starts_seen[0]}"
        )

    def test_summary_method(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=2, seed=7)
        result = ms.run()
        s = result.summary()
        assert "best start" in s.lower() or "best" in s.lower()
        assert "0" in s  # start index

    def test_repr(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=2, seed=8)
        result = ms.run()
        assert repr(result)  # must not raise or return empty


class TestMultiStartConvergence:

    def test_best_converges_to_global_minimum(self):
        """All starts should converge to c ≈ 1.65 (unimodal landscape)."""
        model = _make_spring_optimizable(c0=0.5)
        ms = MultiStart(model, _lbfgsb_factory, n_starts=5, seed=42,
                        sample_scale=0.5)
        result = ms.run()
        best_c = float(result.best_result["c"])
        assert np.isclose(best_c, 1.65, atol=0.15), (
            f"Best result c={best_c} does not match expected ≈1.65"
        )

    def test_best_is_better_than_initial(self):
        """The best result must be at least as good as starting from params_0."""
        model = _make_spring_optimizable(c0=0.5)
        # Single start at params_0
        single = _lbfgsb_factory(model).optimize()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=4, seed=99,
                        include_initial=True)
        result = ms.run()
        assert result.best_result.final_loss <= single.final_loss + 1e-6

    def test_custom_sampler(self):
        """User-supplied sampler is used for start generation."""
        model = _make_spring_optimizable()
        recorded_starts = []

        def my_sampler(n, p0):
            # Always return c values in [0.3, 0.7]
            starts = np.linspace(0.3, 0.7, n).reshape(n, 1)
            recorded_starts.extend(starts[:, 0].tolist())
            return starts

        ms = MultiStart(model, _lbfgsb_factory, n_starts=4, seed=0,
                        init_sampler=my_sampler, include_initial=False)
        ms.run()
        # Sampler must have been called
        assert len(recorded_starts) == 4
        for s in recorded_starts:
            assert 0.3 - 1e-6 <= s <= 0.7 + 1e-6

    def test_bounds_respected_in_default_sampler(self):
        """Default sampler must clip generated starts to bounds."""
        model = _make_spring_optimizable()
        model.bounds_flat = [(0.3, 0.8)]  # override bounds
        ms = MultiStart(model, _lbfgsb_factory, n_starts=20, seed=0,
                        sample_scale=2.0)
        starts = ms._generate_starts()
        assert np.all(starts >= 0.3 - 1e-6)
        assert np.all(starts <= 0.8 + 1e-6)

    def test_results_property_after_run(self):
        model = _make_spring_optimizable()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=3, seed=1)
        assert ms.results == []  # empty before run
        ms.run()
        assert len(ms.results) == 3

    def test_multistart_vs_single_start_same_params_0(self):
        """With n_starts=1 and include_initial=True, result == single run."""
        model = _make_spring_optimizable(c0=1.0)
        single = _lbfgsb_factory(model).optimize()
        ms = MultiStart(model, _lbfgsb_factory, n_starts=1,
                        include_initial=True)
        ms_result = ms.run()
        assert float(ms_result.best_result["c"]) == pytest.approx(
            float(single["c"]), abs=1e-4
        )


class TestMultiStartWithOptax:

    def test_multistart_with_optax(self):
        """MultiStart works with gradient-based Optax optimizer too."""
        model = _make_spring_optimizable(c0=0.5)

        def adam_factory(opt):
            return Optax(opt, "adam", learning_rate=0.01,
                         opt_method_config={}, num_epochs=300)

        ms = MultiStart(model, adam_factory, n_starts=3, seed=0)
        result = ms.run()
        assert isinstance(result.best_result, OptimizationResult)
        assert result.n_successful >= 1
