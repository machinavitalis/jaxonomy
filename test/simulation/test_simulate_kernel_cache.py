# SPDX-License-Identifier: MIT

"""``simulate`` reuses its compiled kernel when a call is genuinely repeated.

``simulate`` used to hand ``jax.jit`` a freshly-created closure on every call.
``jax.jit`` keys its cache on the function object, so that was a guaranteed
100% cache miss: every call re-traced and re-compiled the whole diagram even
when nothing had changed. The ``Simulator`` and its jitted kernel are now
memoized.

Scope: ``t_span`` and ``context`` are traced kernel *arguments*, so varying the
span, the initial state or dynamic parameters all reuse one compile. What keys
the cache is what is genuinely baked into the traced program — the system, the
resolved options, the recorded signals, and static parameter values.

Both historical blockers to hoisting turned out to be bugs elsewhere, not
tracing constraints: a tabulated BDF/DAE cell went NaN under a traced context
(a Newton-convergence bug — see ``test_bdf_newton_convergence.py``), and a video
source reported the wrong frame under a traced span (it read a host-side
counter instead of deriving the frame from time — see ``test_video.py``).

These tests are about *correctness under reuse* — a cached kernel must never
change an answer, and must never be reused when something baked into it moved.
Timing is not asserted (machine-dependent); the cache counters stand in for
"did it actually reuse".
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import LeafSystem
from jaxonomy.framework import parameters
from jaxonomy.framework.parameter import Parameter
from jaxonomy.library import Constant, Gain, Integrator
from jaxonomy.simulation.simulator import clear_simulate_cache, simulate_cache_info
from jaxonomy.simulation.types import SimulatorOptions

pytestmark = pytest.mark.minimal


class _GainWithStaticParam(LeafSystem):
    """Gain whose factor is a *static* parameter, i.e. baked into the trace."""

    @parameters(static=["p"])
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    def initialize(self, p):
        self.configure_output_port(
            self._output_port_idx,
            lambda time, state, u: p * u,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_simulate_cache()
    yield
    clear_simulate_cache()


def _chain(n_blocks=5, gain=0.99):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(1.0, name="src"))
    prev = src.output_ports[0]
    for k in range(n_blocks):
        g = builder.add(Gain(gain, name=f"g{k}"))
        builder.connect(prev, g.input_ports[0])
        prev = g.output_ports[0]
    integ = builder.add(Integrator(initial_state=0.0, name="out"))
    builder.connect(prev, integ.input_ports[0])
    return builder.build(name="chain"), integ


def _run(diagram, context, integ, tf, **kwargs):
    return jaxonomy.simulate(
        diagram, context, (0.0, tf),
        recorded_signals={"y": integ.output_ports[0]}, **kwargs,
    )


class TestReuse:
    def test_repeated_identical_calls_hit_the_cache(self):
        diagram, integ = _chain()
        context = diagram.create_context()

        first = _run(diagram, context, integ, 1.0)
        for _ in range(2):
            again = _run(diagram, context, integ, 1.0)
            assert np.array_equal(again.time, first.time)
            assert np.array_equal(again.outputs["y"], first.outputs["y"])

        info = simulate_cache_info()
        assert info["hits"] == 2, info
        assert info["misses"] == 1, info

    def test_opt_out_matches_reuse(self):
        diagram, integ = _chain()
        context = diagram.create_context()
        options = SimulatorOptions(reuse_compiled_kernel=False)

        fresh = float(_run(diagram, context, integ, 1.3, options=options).outputs["y"][-1])
        reused = float(_run(diagram, context, integ, 1.3).outputs["y"][-1])

        assert reused == fresh

    def test_opt_out_does_not_populate_the_cache(self):
        diagram, integ = _chain()
        context = diagram.create_context()
        options = SimulatorOptions(reuse_compiled_kernel=False)

        _run(diagram, context, integ, 1.0, options=options)
        _run(diagram, context, integ, 1.0, options=options)

        assert simulate_cache_info()["hits"] == 0

    def test_clear_cache_forces_a_recompile(self):
        diagram, integ = _chain()
        context = diagram.create_context()

        _run(diagram, context, integ, 1.0)
        clear_simulate_cache()
        _run(diagram, context, integ, 1.0)

        info = simulate_cache_info()
        assert info["hits"] == 0 and info["misses"] == 1, info


class TestInvalidation:
    """Everything baked into the kernel must key the cache."""

    def test_a_different_span_reuses_the_kernel_and_stays_exact(self):
        """The span is a kernel argument too, so a snapshot walk reuses it."""
        diagram, integ = _chain()
        context = diagram.create_context()

        _run(diagram, context, integ, 1.0)
        hits_before = simulate_cache_info()["hits"]
        warm = float(_run(diagram, context, integ, 2.0).outputs["y"][-1])

        assert simulate_cache_info()["hits"] == hits_before + 1, "expected reuse"

        clear_simulate_cache()
        cold = float(_run(diagram, context, integ, 2.0).outputs["y"][-1])
        assert warm == cold, "a reused kernel changed the answer across spans"

    def test_snapshot_walk_compiles_once(self):
        """The reported case: k+1 calls with a growing end time."""
        diagram, integ = _chain()
        context = diagram.create_context()

        values = [
            float(_run(diagram, context, integ, 0.3 * (k + 1)).outputs["y"][-1])
            for k in range(5)
        ]

        info = simulate_cache_info()
        assert info["misses"] == 1, f"expected one compile, got {info}"
        assert info["hits"] == 4, info
        # The recorded value must still grow with the span.
        assert values == sorted(values)

    def test_a_different_context_reuses_the_kernel_but_not_the_answer(self):
        """The context is a kernel *argument*, so varying it is a cache hit.

        The property that matters is that reuse must not make two different
        initial states produce the same trajectory — the kernel is shared, the
        result is not.
        """
        diagram, integ = _chain()

        base = diagram.create_context()
        first = float(_run(diagram, base, integ, 1.0).outputs["y"][-1])
        hits_before = simulate_cache_info()["hits"]

        # Same structure, different initial state.
        shifted = base.with_continuous_state(
            jax.tree_util.tree_map(lambda x: x + 5.0, base.continuous_state)
        )
        second = float(_run(diagram, shifted, integ, 1.0).outputs["y"][-1])

        assert simulate_cache_info()["hits"] == hits_before + 1, "expected reuse"
        assert second != first, "a shared kernel must not share the answer"
        # Integrator started 5.0 higher stays exactly 5.0 higher.
        assert second == pytest.approx(first + 5.0, rel=1e-9)

    def test_reused_kernel_matches_a_cold_compile(self):
        """A reused kernel must agree with one compiled fresh for that state."""
        diagram, integ = _chain()
        base = diagram.create_context()
        shifted = base.with_continuous_state(
            jax.tree_util.tree_map(lambda x: x + 2.5, base.continuous_state)
        )

        clear_simulate_cache()
        cold = float(_run(diagram, shifted, integ, 1.0).outputs["y"][-1])

        clear_simulate_cache()
        _run(diagram, base, integ, 1.0)          # warm on a different state
        warm = float(_run(diagram, shifted, integ, 1.0).outputs["y"][-1])

        assert warm == cold

    def test_changed_static_parameter_is_not_reused(self):
        """The regression that motivated the parameter epoch.

        A *static* parameter's value is baked into the trace, and
        ``Parameter.set`` leaves the system object untouched, so without an
        epoch in the key the second run would silently return the first run's
        answer. Uses a genuinely static parameter — a dynamic one travels in
        the context and would pass for the wrong reason.
        """
        p = Parameter(1.0)
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(1.0, name="src"))
        gain = builder.add(_GainWithStaticParam(p, name="g"))
        builder.connect(src.output_ports[0], gain.input_ports[0])
        diagram = builder.build(name="param")

        first = jaxonomy.simulate(
            diagram, diagram.create_context(), (0.0, 1.0),
            recorded_signals={"y": gain.output_ports[0]},
        )
        assert float(first.outputs["y"][0]) == pytest.approx(1.0)

        p.set(2.0)
        hits_before = simulate_cache_info()["hits"]
        second = jaxonomy.simulate(
            diagram, diagram.create_context(), (0.0, 1.0),
            recorded_signals={"y": gain.output_ports[0]},
        )
        assert simulate_cache_info()["hits"] == hits_before, (
            "a changed static parameter must not reuse the kernel"
        )
        assert float(second.outputs["y"][0]) == pytest.approx(2.0)

    def test_changed_dynamic_parameter_may_reuse_the_kernel(self):
        """A dynamic parameter rides in the context, so reuse is correct here.

        The epoch is deliberately gated on ``is_static``: bumping for dynamic
        parameters too would be correct but would defeat the cache for the
        common sweep-via-context workload. What must hold is that the shared
        kernel still yields the swept values.
        """
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(1.0, name="src"))
        gain = builder.add(Gain(1.0, name="g"))
        integ = builder.add(Integrator(initial_state=0.0, name="out"))
        builder.connect(src.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], integ.input_ports[0])
        diagram = builder.build(name="sweep")
        base = diagram.create_context()

        results = []
        for value in (1.0, 3.0, 5.0):
            sub = base[gain.system_id].with_parameter("gain", value)
            ctx = base.with_subcontext(gain.system_id, sub)
            results.append(float(_run(diagram, ctx, integ, 1.0).outputs["y"][-1]))

        # Unit input through Gain(v) integrated over 1s is exactly v.
        assert results == pytest.approx([1.0, 3.0, 5.0], rel=1e-9)
        assert simulate_cache_info()["hits"] >= 1, "sweep should reuse the kernel"

    def test_different_recorded_signals_get_different_kernels(self):
        diagram, integ = _chain()
        context = diagram.create_context()
        first_gain = next(s for s in diagram.nodes if s.name == "g0")

        integ_value = float(_run(diagram, context, integ, 1.0).outputs["y"][-1])
        gain_value = float(
            jaxonomy.simulate(
                diagram, context, (0.0, 1.0),
                recorded_signals={"y": first_gain.output_ports[0]},
            ).outputs["y"][-1]
        )

        assert gain_value == pytest.approx(0.99, rel=1e-9)
        assert integ_value != gain_value

    def test_with_parameters_copies_get_their_own_values(self):
        diagram, _ = _chain(n_blocks=5, gain=1.0)
        diagram.create_context()

        for value in (0.5, 2.0):
            swept = diagram.with_parameters({"g0.gain": value})
            integ = next(s for s in swept.nodes if s.name == "out")
            got = float(_run(swept, swept.create_context(), integ, 1.0).outputs["y"][-1])
            assert got == pytest.approx(value, rel=1e-6)


def _out(diagram):
    return next(s for s in diagram.nodes if s.name == "out")


def _final(diagram, tf=1.0, **kwargs):
    result = _run(diagram, diagram.create_context(), _out(diagram), tf, **kwargs)
    return result, float(result.outputs["y"][-1])


def _final_state(diagram):
    result = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, 1.0))
    return float(result.context[_out(diagram).system_id].continuous_state)


class _HostGain(LeafSystem):
    """Gain evaluated on the host: the traced kernel calls back into Python."""

    def __init__(self, gain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_dynamic_parameter("gain", gain)
        self.declare_input_port()

        def host(u, gain):
            return np.asarray(u * gain, dtype=np.float64)

        def output(time, state, u, gain):
            return jax.pure_callback(host, jax.ShapeDtypeStruct((), jnp.float64), u, gain)

        self.declare_output_port(output, requires_inputs=True)


def _pid_plant(kp):
    from jaxonomy.library import PID

    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(1.0, name="src"))
    pid = builder.add(PID(kp=kp, ki=0, kd=0, n=100, name="pid"))
    integ = builder.add(Integrator(initial_state=0.0, name="out"))
    builder.connect(src.output_ports[0], pid.input_ports[0])
    builder.connect(pid.output_ports[0], integ.input_ports[0])
    return builder.build(name="pid_plant", parameters={"kp": kp})


class TestKernelSharing:
    """A ``with_parameters`` copy runs on the kernel compiled for its original
    (or the other way round) whenever that is the same program."""

    def test_copies_run_on_the_original_kernel(self):
        diagram, _ = _chain(n_blocks=5, gain=1.0)
        _, base = _final(diagram)
        assert base == pytest.approx(1.0, rel=1e-6)
        built = simulate_cache_info()["misses"]

        for value in (0.5, 2.0, 3.0):
            swept = diagram.with_parameters({"g0.gain": value})
            result, got = _final(swept)
            assert got == pytest.approx(value, rel=1e-6)
            # The context that comes back belongs to the copy, not the original.
            assert result.context.owning_system is swept
            for sub, leaf in zip(result.context.subcontexts.values(), swept.leaf_systems):
                assert sub.owning_system is leaf
            g0 = next(s for s in swept.nodes if s.name == "g0")
            assert float(result.context[g0.system_id].parameters["gain"]) == value

        assert simulate_cache_info()["misses"] == built

    def test_shared_kernel_matches_a_kernel_of_its_own(self):
        diagram, _ = _chain(n_blocks=5, gain=1.0)
        _final(diagram)
        shared, _ = _final(diagram.with_parameters({"g0.gain": 1.7}), tf=2.0)

        clear_simulate_cache()
        own, _ = _final(diagram.with_parameters({"g0.gain": 1.7}), tf=2.0)

        for a, b in zip(
            jax.tree.leaves((shared.context.continuous_state, shared.time, shared.outputs)),
            jax.tree.leaves((own.context.continuous_state, own.time, own.outputs)),
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_the_original_runs_on_a_copy_kernel(self):
        diagram, _ = _chain(n_blocks=5, gain=1.0)
        diagram.create_context()
        _, got = _final(diagram.with_parameters({"g0.gain": 2.0}))
        assert got == pytest.approx(2.0, rel=1e-6)
        built = simulate_cache_info()["misses"]

        _, got = _final(diagram)
        assert got == pytest.approx(1.0, rel=1e-6)
        _, got = _final(diagram.with_parameters({"g0.gain": 4.0}))
        assert got == pytest.approx(4.0, rel=1e-6)
        assert simulate_cache_info()["misses"] == built

    def test_copies_record_their_own_ports(self):
        diagram, _ = _chain(n_blocks=3, gain=2.0)
        _final(diagram)
        built = simulate_cache_info()["misses"]

        swept = diagram.with_parameters({"g0.gain": 5.0})
        g1 = next(s for s in swept.nodes if s.name == "g1")
        recorded = jaxonomy.simulate(
            swept, swept.create_context(), (0.0, 1.0),
            recorded_signals={"y": g1.output_ports[0]},
        )
        # Another recorded port is another program: 5 * 2, not the integral.
        assert float(recorded.outputs["y"][-1]) == pytest.approx(10.0, rel=1e-9)
        assert simulate_cache_info()["misses"] == built + 1

    def test_a_copy_configured_differently_gets_its_own_kernel(self):
        # A zero gain makes the PID non-feedthrough: another traced program,
        # which never reads the block's input.
        final_state = _final_state
        diagram = _pid_plant(Parameter(jnp.asarray(0.0), name="kp"))
        assert final_state(diagram) == pytest.approx(0.0, abs=1e-12)
        built = simulate_cache_info()["misses"]

        assert final_state(diagram.with_parameters({"kp": 2.0})) == pytest.approx(
            2.0, rel=1e-6
        )
        assert simulate_cache_info()["misses"] == built + 1

        # Copies configured like one of the two share with it.
        assert final_state(diagram.with_parameters({"kp": 3.0})) == pytest.approx(
            3.0, rel=1e-6
        )
        assert final_state(diagram.with_parameters({"kp": 0.0})) == pytest.approx(
            0.0, abs=1e-12
        )
        assert simulate_cache_info()["misses"] == built + 1

    def test_a_reconfigured_original_stops_lending_its_kernel(self):
        kp = Parameter(jnp.asarray(1.0), name="kp")
        diagram = _pid_plant(kp)
        assert _final_state(diagram) == pytest.approx(1.0, rel=1e-6)
        built = simulate_cache_info()["misses"]
        swept = diagram.with_parameters({"kp": 2.0})

        # Now non-feedthrough: a new trace of its kernel would be another program.
        kp.set(0.0)
        diagram.create_context()

        assert _final_state(swept) == pytest.approx(2.0, rel=1e-6)
        assert simulate_cache_info()["misses"] == built + 1

    def test_a_static_change_ends_the_sharing(self):
        p = Parameter(2.0, name="p")
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(1.0, name="src"))
        static = builder.add(_GainWithStaticParam(p, name="static"))
        gain = builder.add(Gain(1.0, name="g0"))
        integ = builder.add(Integrator(initial_state=0.0, name="out"))
        builder.connect(src.output_ports[0], static.input_ports[0])
        builder.connect(static.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], integ.input_ports[0])
        diagram = builder.build(name="static_chain")

        _, got = _final(diagram)
        assert got == pytest.approx(2.0, rel=1e-6)
        before = diagram.with_parameters({"g0.gain": 3.0})  # has p = 2 baked in

        p.set(5.0)
        _, got = _final(diagram)
        assert got == pytest.approx(5.0, rel=1e-6)
        _, got = _final(before)
        assert got == pytest.approx(6.0, rel=1e-6)
        _, got = _final(diagram.with_parameters({"g0.gain": 3.0}))
        assert got == pytest.approx(15.0, rel=1e-6)

    def test_kernels_with_host_callbacks_are_not_shared(self):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(1.0, name="src"))
        host = builder.add(_HostGain(2.0, name="host"))
        integ = builder.add(Integrator(initial_state=0.0, name="out"))
        builder.connect(src.output_ports[0], host.input_ports[0])
        builder.connect(host.output_ports[0], integ.input_ports[0])
        diagram = builder.build(name="host_chain")

        _, got = _final(diagram)
        assert got == pytest.approx(2.0, rel=1e-6)
        built = simulate_cache_info()["misses"]

        # The program runs code of the blocks it was traced with.
        _, got = _final(diagram.with_parameters({"host.gain": 3.0}))
        assert got == pytest.approx(3.0, rel=1e-6)
        assert simulate_cache_info()["misses"] == built + 1

    def test_opt_out_does_not_share(self):
        diagram, _ = _chain(n_blocks=5, gain=1.0)
        _final(diagram)
        info = simulate_cache_info()

        _, got = _final(
            diagram.with_parameters({"g0.gain": 2.0}),
            options=SimulatorOptions(reuse_compiled_kernel=False),
        )
        assert got == pytest.approx(2.0, rel=1e-6)
        assert simulate_cache_info()["hits"] == info["hits"]


class TestPerRunState:
    """Host-side accumulators on a reused Simulator must not leak between runs."""

    def test_drift_monitor_reset_clears_the_trace(self):
        """Without this the second run appends to the first run's trace."""
        from jaxonomy.simulation.simulator import _DAEDriftMonitor

        monitor = _DAEDriftMonitor()
        monitor.update(0.0, 1e-9)
        monitor.update(0.1, 2e-9)
        assert monitor.finalize()["residual"].shape == (2,)

        monitor.reset()
        assert monitor.finalize() is None

        monitor.update(0.0, 3e-9)
        assert monitor.finalize()["residual"].shape == (1,)

    def test_bdf_condition_monitor_reset_clears_the_running_max(self):
        """Without this a later run could inherit an earlier run's warning."""
        from jaxonomy.simulation.simulator import _BDFConditionMonitor

        monitor = _BDFConditionMonitor(threshold=1e6)
        monitor.update(1e12, 0.5)
        assert monitor.max_cond == 1e12
        assert monitor.n_samples == 1

        monitor.reset()

        assert monitor.max_cond == float("-inf")
        assert monitor.n_samples == 0

        # A reset monitor has nothing to report.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            monitor.maybe_warn()
        assert caught == []


class TestCacheHousekeeping:
    def test_cache_is_bounded(self):
        maxsize = simulate_cache_info()["maxsize"]

        for _ in range(maxsize + 5):
            diagram, integ = _chain(n_blocks=2)
            _run(diagram, diagram.create_context(), integ, 0.1)

        assert simulate_cache_info()["size"] <= maxsize

    def test_clearing_drops_every_entry(self):
        for _ in range(3):
            diagram, integ = _chain(n_blocks=2)
            _run(diagram, diagram.create_context(), integ, 0.1)
        assert simulate_cache_info()["size"] > 0

        clear_simulate_cache()

        info = simulate_cache_info()
        assert info["size"] == 0
        assert info["hits"] == 0 and info["misses"] == 0

    def test_evicted_kernel_is_recompiled_not_reused(self):
        diagram, integ = _chain(n_blocks=2)
        context = diagram.create_context()
        first = float(_run(diagram, context, integ, 0.4).outputs["y"][-1])

        for _ in range(simulate_cache_info()["maxsize"]):
            other, other_integ = _chain(n_blocks=2)
            _run(other, other.create_context(), other_integ, 0.1)

        hits_before = simulate_cache_info()["hits"]
        again = float(_run(diagram, context, integ, 0.4).outputs["y"][-1])

        assert simulate_cache_info()["hits"] == hits_before, "expected a miss"
        assert again == first, "recompiled kernel changed the answer"


class TestUnderJaxTransforms:
    """The context is a traced argument now, so transforms must still be exact."""

    def test_vmap_over_initial_conditions_shares_one_kernel(self):
        diagram, integ = _chain(n_blocks=1, gain=2.0)
        base = diagram.create_context()

        def final_state(x0):
            ctx = base.with_continuous_state(
                jax.tree_util.tree_map(lambda _: x0, base.continuous_state)
            )
            results = jaxonomy.simulate(diagram, ctx, (0.0, 1.0))
            return jax.tree_util.tree_leaves(results.context.continuous_state)[0]

        batched = np.asarray(jax.vmap(final_state)(jnp.array([0.0, 1.0, 2.0])))

        # Constant(1) -> Gain(2) -> Integrator over 1s adds exactly 2.0.
        assert batched.ravel() == pytest.approx([2.0, 3.0, 4.0], rel=1e-9)

    def test_reverse_mode_grad_reuses_the_kernel_and_is_correct(self):
        diagram, integ = _chain(n_blocks=1, gain=2.0)
        base = diagram.create_context()
        options = SimulatorOptions(enable_autodiff=True, max_major_steps=20)

        def loss(x0):
            ctx = base.with_continuous_state(
                jax.tree_util.tree_map(lambda _: x0, base.continuous_state)
            )
            results = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=options)
            return jnp.sum(
                jax.tree_util.tree_leaves(results.context.continuous_state)[0]
            )

        first = float(jax.grad(loss)(jnp.asarray(0.0)))
        hits_before = simulate_cache_info()["hits"]
        second = float(jax.grad(loss)(jnp.asarray(1.0)))

        # d(final)/d(x0) == 1: an integrator carries an offset through unchanged.
        assert first == pytest.approx(1.0, rel=1e-9)
        assert second == pytest.approx(1.0, rel=1e-9)
        assert simulate_cache_info()["hits"] > hits_before, "expected reuse"

    def test_autodiff_and_non_autodiff_do_not_share_a_kernel(self):
        """``enable_autodiff`` changes the traced program, so it must key the cache."""
        diagram, integ = _chain()
        context = diagram.create_context()

        _run(diagram, context, integ, 1.0)
        hits_before = simulate_cache_info()["hits"]
        jaxonomy.simulate(
            diagram, context, (0.0, 1.0),
            options=SimulatorOptions(enable_autodiff=True, max_major_steps=20),
        )
        assert simulate_cache_info()["hits"] == hits_before
