# SPDX-License-Identifier: MIT

"""Tests for the sensitivity-weighted influence graph.

The load-bearing properties, in the order they are checked below:

1. Edge weights are the real Jacobians (finite-difference cross-check).
2. Path products telescope, so ``attribute`` returns the true end-to-end
   sensitivity — including the sign, so cancellation between paths shows up.
3. A quantitative slice is materially smaller than the boolean one on a model
   where a block is locally inert, and the ``probe`` cross-check recovers the
   blocks that a purely local derivative writes off.
4. Anything that cannot be differentiated is labelled, not zeroed.
5. Building the graph has no side effect on the model.
"""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

import jaxonomy
from jaxonomy.analysis import (
    influence_graph,
    influence_subgraph,
    leaf_jacobians,
)
from jaxonomy.analysis.influence_context import CHARS_PER_TOKEN
from jaxonomy.library import (
    Adder,
    Comparator,
    Constant,
    Gain,
    Integrator,
    Quantizer,
    Saturate,
)
from jaxonomy.library.battery_cell import BatteryCell
from jaxonomy.library.dynamics import DiscreteInitializer

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def gain_chain(gains=(3.0, -2.0, 5.0), x0=0.5):
    """source -> gain -> gain -> gain -> integrator."""
    builder = jaxonomy.DiagramBuilder()
    source = builder.add(Constant(1.0, name="src"))
    blocks = [
        builder.add(Gain(value, name=f"g{index}")) for index, value in enumerate(gains)
    ]
    plant = builder.add(Integrator(x0, name="plant"))
    builder.connect(source.output_ports[0], blocks[0].input_ports[0])
    for upstream, downstream in zip(blocks, blocks[1:]):
        builder.connect(upstream.output_ports[0], downstream.input_ports[0])
    builder.connect(blocks[-1].output_ports[0], plant.input_ports[0])
    return builder.build(name="chain")


def saturating_loop(upper=0.5, kp=4.0, x0=0.2):
    """Unity-feedback loop whose actuator is saturated at the initial state."""
    builder = jaxonomy.DiagramBuilder()
    reference = builder.add(Constant(1.0, name="ref"))
    error = builder.add(Adder(2, operators="+-", name="err"))
    gain = builder.add(Gain(kp, name="kp"))
    actuator = builder.add(
        Saturate(upper_limit=upper, lower_limit=-upper, name="sat")
    )
    plant = builder.add(Integrator(x0, name="plant"))
    builder.connect(reference.output_ports[0], error.input_ports[0])
    builder.connect(plant.output_ports[0], error.input_ports[1])
    builder.connect(error.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], actuator.input_ports[0])
    builder.connect(actuator.output_ports[0], plant.input_ports[0])
    return builder.build(name="loop")


def cancelling_paths(a=2.0, b=2.0, drive=1.5):
    """One source reaching one adder by a ``+`` path and a ``-`` path.

    The source is a unity gain with a fixed input rather than a ``Constant`` so
    a test can differentiate the whole diagram with respect to it and compare
    against the graph's path products.
    """
    builder = jaxonomy.DiagramBuilder()
    source = builder.add(Gain(1.0, name="src"))
    source.input_ports[0].fix_value(jnp.asarray(drive))
    left = builder.add(Gain(a, name="left"))
    right = builder.add(Gain(b, name="right"))
    total = builder.add(Adder(2, operators="+-", name="sum"))
    builder.connect(source.output_ports[0], left.input_ports[0])
    builder.connect(source.output_ports[0], right.input_ports[0])
    builder.connect(left.output_ports[0], total.input_ports[0])
    builder.connect(right.output_ports[0], total.input_ports[1])
    return builder.build(name="cancel")


# ---------------------------------------------------------------------------
# 1. The weights are the real derivatives
# ---------------------------------------------------------------------------


class TestWeightsAreRealJacobians:
    def test_feedthrough_weight_matches_finite_difference(self):
        diagram = gain_chain(gains=(3.0, -2.0, 5.0))
        graph = influence_graph(diagram, normalize="none")

        gain_block = diagram["g1"]
        context = diagram.create_context()
        u0 = float(gain_block.input_ports[0].eval(context))
        step = 1e-4

        def output_at(value):
            with gain_block.input_ports[0].fixed(jnp.asarray(value)):
                return float(gain_block.output_ports[0].eval(context))

        secant = (output_at(u0 + step) - output_at(u0 - step)) / (2 * step)
        edge = graph.graph.edges["g1:in:in_0", "g1:out:out_0"]
        assert edge["weight"] == pytest.approx(secant, rel=1e-6)
        assert edge["weight"] == pytest.approx(-2.0, rel=1e-9)

    def test_state_edges_are_the_integrator_b_and_c_matrices(self):
        diagram = gain_chain(x0=0.5)
        graph = influence_graph(diagram, normalize="none", tau=1.0)
        # ẋ = u  and  y = x, exactly.
        assert graph.graph.edges["plant:in:in_0", "plant:xc"]["weight"] == pytest.approx(
            1.0
        )
        assert graph.graph.edges["plant:xc", "plant:out:out_0"][
            "weight"
        ] == pytest.approx(1.0)
        # A = 0 for a plain integrator.
        assert graph.graph.edges["plant:xc", "plant:xc"]["weight"] == pytest.approx(0.0)

    def test_tau_scales_only_continuous_state_rate_edges(self):
        diagram = gain_chain()
        one = influence_graph(diagram, normalize="none", tau=1.0)
        ten = influence_graph(diagram, normalize="none", tau=10.0)

        to_state = ("plant:in:in_0", "plant:xc")
        from_state = ("plant:xc", "plant:out:out_0")
        assert ten.graph.edges[to_state]["weight"] == pytest.approx(
            10.0 * one.graph.edges[to_state]["weight"]
        )
        assert ten.graph.edges[from_state]["weight"] == pytest.approx(
            one.graph.edges[from_state]["weight"]
        )
        assert ten.graph.edges[to_state]["tau_applied"] is True
        assert ten.graph.edges[from_state]["tau_applied"] is False

    def test_relative_weights_are_elasticities(self):
        # A pure gain has elasticity 1 whatever the gain: y = k·u so
        # (∂y/∂u)·(u/y) = k·u/(k·u) = 1.
        graph = influence_graph(gain_chain(gains=(3.0, -2.0, 5.0)))
        for name in ("g0", "g1", "g2"):
            weight = graph.graph.edges[f"{name}:in:in_0", f"{name}:out:out_0"]["weight"]
            assert abs(weight) == pytest.approx(1.0)

    def test_wire_edges_are_exactly_identity(self):
        graph = influence_graph(gain_chain())
        edge = graph.graph.edges["g0:out:out_0", "g1:in:in_0"]
        assert edge["kind"] == "wire"
        assert edge["weight"] == pytest.approx(1.0)
        assert np.allclose(edge["jacobian"], np.eye(1))

    def test_leaf_jacobians_handles_namedtuple_state(self):
        # BatteryCell's continuous state is a NamedTuple, not an array.
        builder = jaxonomy.DiagramBuilder()
        load = builder.add(Constant(-2.0, name="load"))
        cell = builder.add(BatteryCell(name="cell"))
        builder.connect(load.output_ports[0], cell.input_ports[0])
        diagram = builder.build(name="pack")
        jacs = leaf_jacobians(diagram["cell"], diagram.create_context())
        assert jacs.notes == {}
        assert ("xc", 0) in jacs.b
        assert jacs.a[("xc", "xc")].ndim == 2


# ---------------------------------------------------------------------------
# 2. Path products telescope
# ---------------------------------------------------------------------------


class TestPathAttribution:
    def test_raw_path_product_is_the_end_to_end_derivative(self):
        gains = (3.0, -2.0, 5.0)
        graph = influence_graph(gain_chain(gains=gains), normalize="none", tau=1.0)
        result = graph.attribute("plant:xc", "src:out:out_0")
        assert len(result.paths) == 1
        # ẋ = g0·g1·g2·src, scaled by tau=1.
        assert result.total == pytest.approx(float(np.prod(gains)))

    def test_relative_path_product_telescopes(self):
        gains = (3.0, -2.0, 5.0)
        x0 = 0.5
        graph = influence_graph(gain_chain(gains=gains, x0=x0), tau=1.0)
        result = graph.attribute("plant:xc", "src:out:out_0")
        # Relative sensitivity = tau · (∂ẋ/∂src) · src/x.
        expected = 1.0 * float(np.prod(gains)) * 1.0 / x0
        assert result.total == pytest.approx(expected)

    def test_path_product_is_the_gain_at_one_over_tau(self):
        # The claim the module docstring makes about tau: for a path of
        # algebraic gains and k integrators, G(s) = (prod gains)/s**k, so
        # |G(jw)| = (prod gains)/w**k. The tau-scaled path product should equal
        # that exactly with w = 1/tau -- which is what makes tau readable as a
        # frequency choice rather than a fudge factor.
        first, second = 3.0, 0.5
        builder = jaxonomy.DiagramBuilder()
        drive = builder.add(Gain(1.0, name="u"))
        drive.input_ports[0].fix_value(jnp.asarray(1.0))
        gain_a = builder.add(Gain(first, name="ga"))
        state_a = builder.add(Integrator(1.0, name="A"))
        gain_b = builder.add(Gain(second, name="gb"))
        state_b = builder.add(Integrator(1.0, name="B"))
        builder.connect(drive.output_ports[0], gain_a.input_ports[0])
        builder.connect(gain_a.output_ports[0], state_a.input_ports[0])
        builder.connect(state_a.output_ports[0], gain_b.input_ports[0])
        builder.connect(gain_b.output_ports[0], state_b.input_ports[0])
        diagram = builder.build(name="double_integrator")

        for tau in (0.01, 0.1, 1.0, 10.0):
            graph = influence_graph(diagram, normalize="none", tau=tau)
            product = graph.attribute("B:xc", "u:in:in_0").total
            omega = 1.0 / tau
            assert product == pytest.approx(first * second / omega**2, rel=1e-9)

    def test_cancellation_is_visible_in_the_signed_total(self):
        graph = influence_graph(cancelling_paths(a=2.0, b=2.0), normalize="none")
        result = graph.attribute("sum:out:out_0", "src:out:out_0")
        assert len(result.paths) == 2
        assert result.total == pytest.approx(0.0, abs=1e-12)
        assert result.total_magnitude == pytest.approx(4.0)

    def test_uncancelled_total_matches_autodiff(self):
        diagram = cancelling_paths(a=2.0, b=0.5, drive=1.5)
        graph = influence_graph(diagram, normalize="none")
        result = graph.attribute("sum:out:out_0", "src:in:in_0")

        # Ground truth: differentiate the whole diagram, not block by block.
        context = diagram.create_context()
        driver = diagram["src"].input_ports[0]
        output = diagram["sum"].output_ports[0]

        def readout(value):
            with driver.fixed(value):
                return jnp.reshape(output.eval(context), ())

        reference = float(jax.grad(readout)(jnp.asarray(1.5)))
        assert result.total == pytest.approx(reference, rel=1e-9)
        assert result.total == pytest.approx(1.5, rel=1e-9)

    def test_dominant_paths_ranks_by_magnitude(self):
        graph = influence_graph(cancelling_paths(a=2.0, b=0.5), normalize="none")
        paths = graph.dominant_paths("sum:out:out_0", k=2)
        assert len(paths) == 2
        assert abs(paths[0]["product"]) >= abs(paths[1]["product"])
        assert paths[0]["nodes"][0] == "src:in:in_0"

    def test_attribution_is_truncated_not_infinite_on_a_loop(self):
        graph = influence_graph(saturating_loop(), probe=0.9, tau=0.1)
        result = graph.attribute("plant:xc", "ref:out:out_0", max_depth=8)
        assert result.paths  # the loop does not prevent finding the path
        assert all(len(entry["nodes"]) <= 9 for entry in result.paths)


# ---------------------------------------------------------------------------
# 3. Quantitative slicing beats boolean slicing
# ---------------------------------------------------------------------------


class TestSlicing:
    def test_quantitative_slice_is_smaller_than_the_structural_slice(self):
        # The actuator is saturated at the operating point, so nothing upstream
        # of it has any local influence on the plant state — while the boolean
        # slice has to return the whole loop.
        graph = influence_graph(saturating_loop(), tau=0.1)
        quantitative = graph.slice("plant:xc", threshold=0.01)
        structural = graph.structural_slice("plant:xc")
        assert set(quantitative.blocks) == {"plant", "sat"}
        assert set(structural) == {"ref", "err", "kp", "sat", "plant"}
        assert len(structural) >= 2 * len(quantitative.blocks)

    def test_probe_recovers_blocks_a_local_derivative_writes_off(self):
        diagram = saturating_loop()
        without = influence_graph(diagram, tau=0.1)
        with_probe = influence_graph(diagram, tau=0.1, probe=0.9)
        assert set(without.slice("plant:xc").blocks) == {"plant", "sat"}
        assert set(with_probe.slice("plant:xc").blocks) == set(
            with_probe.structural_slice("plant:xc")
        )

    def test_threshold_prunes_weak_contributors(self):
        graph = influence_graph(gain_chain(gains=(1e-4,), x0=1.0), tau=1.0)
        # Relative influence of src on x is 1e-4, below a 1% threshold.
        assert set(graph.slice("plant:xc", threshold=0.01).blocks) == {"plant"}
        assert set(graph.slice("plant:xc", threshold=1e-6).blocks) == {
            "plant",
            "g0",
            "src",
        }

    def test_relative_threshold_tracks_the_strongest_contributor(self):
        # An absolute threshold only reads as a percentage when tau is
        # comparable to the path's time constants; relative_threshold restores
        # "1% of the dominant contributor" at any tau.
        diagram = gain_chain(gains=(3.0,), x0=1.0)
        for tau in (0.01, 1.0, 100.0):
            graph = influence_graph(diagram, tau=tau)
            cutoff = graph.relative_threshold("plant:xc", 0.5)
            scores = graph.slice("plant:xc", threshold=1e-12).scores
            strongest = max(v for k, v in scores.items() if k != "plant:xc")
            assert cutoff == pytest.approx(0.5 * strongest)
            assert set(graph.slice("plant:xc", threshold=cutoff).blocks) <= {
                "plant",
                "g0",
                "src",
            }

    def test_relative_threshold_on_an_isolated_target(self):
        block = Integrator(1.0, name="alone")
        block.input_ports[0].fix_value(jnp.asarray(0.0))
        graph = influence_graph(block)
        # "alone:xc" has an upstream input port, but a target with nothing
        # upstream at all must not divide by an empty maximum.
        assert graph.relative_threshold("alone:in:in_0", 0.01) > 0.0

    def test_forward_slice_reports_what_a_signal_reaches(self):
        graph = influence_graph(gain_chain(), tau=1.0)
        downstream = graph.slice("g0:out:out_0", threshold=1e-9, direction="forward")
        assert set(downstream.blocks) == {"g0", "g1", "g2", "plant"}
        assert "src:out:out_0" not in downstream.scores

    def test_slice_rejects_an_unknown_direction(self):
        graph = influence_graph(gain_chain())
        with pytest.raises(ValueError, match="backward"):
            graph.slice("plant:xc", direction="sideways")

    def test_bottleneck_is_the_single_series_connection(self):
        graph = influence_graph(gain_chain(gains=(3.0, -2.0, 5.0)), tau=1.0)
        bottlenecks = graph.bottlenecks("plant:xc", threshold=1e-9)
        # Every path from the source to the state runs through the whole chain.
        for node in ("g1:in:in_0", "g1:out:out_0", "plant:in:in_0"):
            assert node in bottlenecks
        assert "plant:xc" not in bottlenecks

    def test_slice_report_is_human_readable(self):
        graph = influence_graph(gain_chain(), tau=1.0)
        report = graph.slice("plant:xc", threshold=1e-9).report()
        assert "Influence slice (backward) for plant:xc" in report
        assert "src:out:out_0" in report
        assert "tau=1" in report

    def test_docstring_example_behaves_as_written(self):
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(1.0, name="src"))
        gain = builder.add(Gain(3.0, name="gain"))
        plant = builder.add(Integrator(1.0, name="plant"))
        builder.connect(source.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], plant.input_ports[0])
        diagram = builder.build(name="root")
        graph = influence_graph(diagram)
        assert graph.slice("plant:xc", threshold=0.01).blocks == [
            "gain",
            "plant",
            "src",
        ]


# ---------------------------------------------------------------------------
# 4. Honesty labels
# ---------------------------------------------------------------------------


class TestHonestyLabels:
    def logic_model(self):
        builder = jaxonomy.DiagramBuilder()
        signal = builder.add(Constant(0.3, name="s"))
        limit = builder.add(Constant(0.5, name="t"))
        compare = builder.add(Comparator(operator=">", name="cmp"))
        builder.connect(signal.output_ports[0], compare.input_ports[0])
        builder.connect(limit.output_ports[0], compare.input_ports[1])
        return builder.build(name="logic")

    def test_boolean_output_is_labelled_not_zeroed(self):
        graph = influence_graph(self.logic_model())
        edge = graph.graph.edges["cmp:in:in_0", "cmp:out:out_0"]
        assert edge["local_gradient"] is False
        assert np.isnan(edge["magnitude"])
        assert "bool" in edge["note"]
        # A zeroed weight would have silently dropped these from the slice.
        model_slice = graph.slice("cmp:out:out_0", threshold=0.5)
        assert set(model_slice.blocks) == {"cmp", "s", "t"}
        assert model_slice.unknown_paths
        assert "s:out:out_0" in model_slice.unknown_nodes

    def test_unknown_edges_are_not_reported_as_dead(self):
        graph = influence_graph(self.logic_model())
        assert graph.dead_edges() == []

    def test_summary_surfaces_unknown_gradients(self):
        summary = influence_graph(self.logic_model()).summary()
        assert "no local gradient" in summary

    def test_zero_crossing_blocks_are_flagged_hybrid(self):
        graph = influence_graph(saturating_loop())
        assert graph.graph.nodes["sat:out:out_0"]["hybrid"] is True
        assert graph.graph.nodes["plant:xc"]["hybrid"] is False
        assert "hybrid blocks" in graph.summary()

    def test_dead_edge_found_and_self_loops_excluded(self):
        graph = influence_graph(saturating_loop(), tau=0.1)
        dead = graph.dead_edges()
        assert ("sat:in:in_0", "sat:out:out_0") in {
            (entry["src"], entry["dst"]) for entry in dead
        }
        # The integrator's own A = 0 block is not a defect.
        assert all(entry["src"] != entry["dst"] for entry in dead)

    def test_probe_relabels_a_locally_flat_edge(self):
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(0.3, name="s"))
        quantizer = builder.add(Quantizer(0.1, name="q"))
        builder.connect(source.output_ports[0], quantizer.input_ports[0])
        diagram = builder.build(name="quantized")

        exact = influence_graph(diagram)
        assert exact.graph.edges["q:in:in_0", "q:out:out_0"]["weight"] == 0.0
        assert exact.dead_edges()

        probed = influence_graph(diagram, probe=0.5)
        edge = probed.graph.edges["q:in:in_0", "q:out:out_0"]
        assert edge["weight"] != 0.0
        assert "secant" in edge["note"]
        assert probed.dead_edges() == []


# ---------------------------------------------------------------------------
# 5. No side effects on the model
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    def test_simulation_is_unchanged_after_building_the_graph(self):
        diagram = gain_chain()
        context = diagram.create_context()
        plant = diagram["plant"]
        before = jaxonomy.simulate(
            diagram, context, (0.0, 1.0), recorded_signals={"x": plant.output_ports[0]}
        )
        influence_graph(diagram, context, probe=0.1)
        after = jaxonomy.simulate(
            diagram, context, (0.0, 1.0), recorded_signals={"x": plant.output_ports[0]}
        )
        assert np.array_equal(np.asarray(before.outputs["x"]), np.asarray(after.outputs["x"]))

    def test_a_user_fixed_input_port_survives(self):
        block = Gain(2.0, name="lonely")
        block.input_ports[0].fix_value(jnp.asarray(7.0))
        influence_graph(block, block.create_context())
        assert block.input_ports[0].is_fixed
        assert float(block.input_ports[0].eval(block.create_context())) == 7.0

    def test_input_ports_are_left_unfixed_when_they_started_unfixed(self):
        diagram = gain_chain()
        influence_graph(diagram)
        assert not diagram["g1"].input_ports[0].is_fixed

    def test_works_on_a_bare_leaf_system(self):
        block = Integrator(0.25, name="alone")
        block.input_ports[0].fix_value(jnp.asarray(3.0))
        graph = influence_graph(block, normalize="none", tau=1.0)
        assert graph.n_blocks == 1
        assert graph.graph.edges["alone:in:in_0", "alone:xc"][
            "weight"
        ] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Discrete state
# ---------------------------------------------------------------------------


class TestDiscreteState:
    def test_boolean_discrete_state_is_labelled_not_differentiated(self):
        block = DiscreteInitializer(dt=0.1, initial_state=True, name="hold")
        graph = influence_graph(block, normalize="none")
        # A boolean discrete state has no derivative; it must be reported as
        # such rather than raveled into a float and differentiated. The edge out
        # of it still exists, labelled unknown, so a slice cannot conclude the
        # state is irrelevant to the output.
        assert "bool" in graph.block_notes["hold"]["state:xd"]
        edge = graph.graph.edges["hold:xd", "hold:out:out_0"]
        assert edge["local_gradient"] is False
        assert graph.dead_edges() == []

    def test_discrete_filter_state_edges_carry_no_tau(self):
        from jaxonomy.library.linear_system import LTISystemDiscrete

        block = LTISystemDiscrete(
            A=jnp.array([[0.5]]),
            B=jnp.array([[1.0]]),
            C=jnp.array([[2.0]]),
            D=jnp.array([[0.0]]),
            dt=0.1,
            name="filt",
        )
        block.input_ports[0].fix_value(jnp.array([1.0]))
        graph = influence_graph(block, normalize="none", tau=100.0)
        to_state = graph.graph.edges["filt:in:in_0", "filt:xd"]
        assert to_state["tau_applied"] is False
        assert to_state["weight"] == pytest.approx(1.0)
        assert graph.graph.edges["filt:xd", "filt:xd"]["weight"] == pytest.approx(0.5)
        # The output port is sample-and-hold, so its own callback just reads the
        # port cache and would differentiate to zero; the block's real C matrix
        # lives in the port's periodic update.
        assert graph.graph.nodes["filt:out:out_0"]["sampled"] is True
        assert graph.graph.edges["filt:xd", "filt:out:out_0"][
            "weight"
        ] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Trajectory mode
# ---------------------------------------------------------------------------


class TestTrajectoryMode:
    def test_profile_captures_an_edge_that_only_conducts_later(self):
        diagram = saturating_loop()
        context = diagram.create_context()
        plant = diagram["plant"]
        results = jaxonomy.simulate(
            diagram,
            context,
            (0.0, 4.0),
            recorded_signals={"x": plant.output_ports[0]},
        )
        graph = influence_graph(
            diagram,
            context,
            at="trajectory",
            results=results,
            n_snapshots=6,
            tau=0.1,
        )
        edge = graph.graph.edges["sat:in:in_0", "sat:out:out_0"]
        assert graph.at == "trajectory"
        assert len(graph.times) == 6
        assert edge["profile"][0] == pytest.approx(0.0)  # saturated at the start
        assert edge["profile"][-1] > 0.0  # in the linear region later
        # reduce="max" is conservative, so the whole loop survives the slice.
        assert set(graph.slice("plant:xc", threshold=0.01).blocks) == set(
            graph.structural_slice("plant:xc")
        )

    def test_trajectory_weights_stay_telescoping(self):
        # A per-snapshot normalizer would let a late snapshot's tiny error
        # signal inflate the product without bound; trajectory-wide scales keep
        # the end-to-end relative sensitivity finite and interpretable.
        diagram = saturating_loop()
        context = diagram.create_context()
        results = jaxonomy.simulate(diagram, context, (0.0, 4.0))
        graph = influence_graph(
            diagram,
            context,
            at="trajectory",
            times=np.linspace(0.0, 4.0, 6),
            tau=0.1,
        )
        best = graph.slice("plant:xc", threshold=1e-9).scores
        assert max(best.values()) < 10.0

    def test_reduce_final_uses_the_last_snapshot(self):
        diagram = saturating_loop()
        context = diagram.create_context()
        common = dict(at="trajectory", times=[0.0, 4.0], tau=0.1)
        highest = influence_graph(diagram, context, reduce="max", **common)
        last = influence_graph(diagram, context, reduce="final", **common)
        edge = ("sat:in:in_0", "sat:out:out_0")
        assert last.graph.edges[edge]["magnitude"] == pytest.approx(
            last.graph.edges[edge]["profile"][-1]
        )
        assert highest.graph.edges[edge]["magnitude"] >= last.graph.edges[edge][
            "magnitude"
        ]

    def test_trajectory_mode_requires_times_or_results(self):
        diagram = gain_chain()
        with pytest.raises(ValueError, match="results="):
            influence_graph(diagram, at="trajectory")

    def test_snapshot_before_the_context_time_is_rejected(self):
        diagram = gain_chain()
        context = diagram.create_context().with_time(1.0)
        with pytest.raises(ValueError, match="at or after the context time"):
            influence_graph(diagram, context, at="trajectory", times=[0.0, 0.5])


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------


class TestTraversalSoundness:
    """Regressions for defects found in adversarial review of this module."""

    def diluting_chain(self):
        """A path whose middle nearly cancels, then is amplified back.

        ``src`` carries 100% of the influence on ``n2:out``, but the partial
        product dips to ~1e-4 at the cancelling junction before the next
        junction's elasticity lifts it back. Any traversal that prunes on the
        running product drops the dominant contributor entirely.
        """
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(1.0, name="src"))
        gain = builder.add(Gain(1.0001, name="k"))
        unit = builder.add(Constant(1.0, name="one"))
        other = builder.add(Constant(1.0, name="big"))
        cancel = builder.add(Adder(2, operators="+-", name="n1"))
        total = builder.add(Adder(2, operators="++", name="n2"))
        builder.connect(source.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], cancel.input_ports[0])
        builder.connect(unit.output_ports[0], cancel.input_ports[1])
        builder.connect(cancel.output_ports[0], total.input_ports[0])
        builder.connect(other.output_ports[0], total.input_ports[1])
        return builder.build(name="dilute")

    def test_slice_keeps_a_contributor_whose_partial_product_dips(self):
        graph = influence_graph(self.diluting_chain())
        # Ground truth first: the end-to-end relative sensitivity is ~1.
        assert graph.attribute("n2:out:out_0", "src:out:out_0").total == pytest.approx(
            1.0, rel=1e-6
        )
        assert "src" in graph.slice("n2:out:out_0", threshold=0.01).blocks

    def test_attribute_finds_the_path_through_the_dip(self):
        graph = influence_graph(self.diluting_chain())
        result = graph.attribute("n2:out:out_0", "src:out:out_0")
        assert len(result.paths) == 1
        assert result.total_magnitude == pytest.approx(1.0, rel=1e-6)
        assert not result.truncated

    def test_scores_do_not_circulate_a_feedback_loop(self):
        # Allowing a path to revisit a node lets it go round the loop, so the
        # target ends up "influencing itself" by the loop gain to the power of
        # the depth bound. Paths must be simple.
        graph = influence_graph(saturating_loop(upper=1e6), tau=1.0)
        model_slice = graph.slice("plant:xc", threshold=1e-9)
        assert model_slice.scores["plant:xc"] == pytest.approx(1.0)
        assert all(np.isfinite(score) for score in model_slice.scores.values())

    def test_reach_matches_brute_force_on_random_graphs(self):
        """The traversal's scores are exactly the best simple-path products.

        The pruning bound has to be admissible for this to hold. Random graphs
        with weights well above 1 (the elasticity case that breaks a
        running-product cutoff) are checked against exhaustive enumeration of
        every simple path, at thresholds that exercise the pruning.
        """
        import itertools
        import random

        from jaxonomy.analysis.influence import InfluenceGraph

        def brute_force(graph, target, max_depth):
            nodes = list(graph)
            best = {target: 1.0}
            others = [n for n in nodes if n != target]
            for source in others:
                middle = [n for n in nodes if n not in (source, target)]
                top = None
                for length in range(1, max_depth + 1):
                    for interior in itertools.permutations(middle, length - 1):
                        path = (source,) + interior + (target,)
                        if not all(
                            graph.has_edge(a, b) for a, b in zip(path, path[1:])
                        ):
                            continue
                        product = 1.0
                        for a, b in zip(path, path[1:]):
                            product *= abs(graph.edges[a, b]["magnitude"])
                        top = product if top is None else max(top, product)
                if top is not None:
                    best[source] = top
            return best

        rng = random.Random(20260730)
        for _ in range(25):
            size = rng.randint(3, 6)
            graph = nx.DiGraph()
            for index in range(size):
                graph.add_node(
                    f"n{index}",
                    block=f"n{index}",
                    value=np.ones(1),
                    scale=np.ones(1),
                    kind="output",
                    hybrid=False,
                    size=1,
                )
            for a in range(size):
                for b in range(size):
                    if a == b or rng.random() >= 0.45:
                        continue
                    weight = rng.choice([0.01, 0.3, 1.0, 3.0, 50.0]) * rng.choice(
                        [1, -1]
                    )
                    graph.add_edge(
                        f"n{a}",
                        f"n{b}",
                        kind="feedthrough",
                        jacobian=None,
                        relative=np.array([[weight]]),
                        weight=weight,
                        magnitude=abs(weight),
                        local_gradient=True,
                        note="",
                        tau_applied=False,
                    )
            influence = InfluenceGraph(
                system=type("Stub", (), {"name": "stub"})(),
                graph=graph,
                tau=1.0,
                normalize="none",
                scale_floor=1e-6,
                at="operating_point",
                times=None,
                reduce="max",
            )
            expected = brute_force(graph, "n0", 4)
            for threshold in (0.0, 0.5, 5.0):
                found, _, routes, truncated = influence._reach(
                    "n0", 4, "backward", threshold=threshold
                )
                assert not truncated
                for node, exact in expected.items():
                    if exact >= threshold:
                        assert node in found
                        assert found[node] == pytest.approx(exact, rel=1e-9)
                    if node in found:
                        assert found[node] <= exact * (1 + 1e-9)
                        # The recorded route must realize the reported score.
                        route = routes[node]
                        assert route[0] == node and route[-1] == "n0"
                        assert len(set(route)) == len(route)
                        realized = 1.0
                        for a, b in zip(route, route[1:]):
                            realized *= abs(graph.edges[a, b]["magnitude"])
                        assert realized == pytest.approx(found[node], rel=1e-9)

    def test_slice_subgraph_is_connected_to_the_target(self):
        """Every retained node must have a route to the target in the slice.

        Retaining nodes purely by their own score leaves the intermediate hops
        of a dominant route out, and the resulting disconnected subgraph makes
        ``bottlenecks()`` report every parallel branch as a single point of
        failure.
        """
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(1.0, name="src"))
        gain = builder.add(Gain(1.0001, name="k"))
        unit = builder.add(Constant(1.0, name="one"))
        other = builder.add(Constant(1.0, name="big"))
        cancel = builder.add(Adder(2, operators="+-", name="n1"))
        middle = builder.add(Gain(1.0, name="middleman"))
        total = builder.add(Adder(2, operators="++", name="n2"))
        builder.connect(source.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], cancel.input_ports[0])
        builder.connect(unit.output_ports[0], cancel.input_ports[1])
        builder.connect(cancel.output_ports[0], middle.input_ports[0])
        builder.connect(middle.output_ports[0], total.input_ports[0])
        builder.connect(other.output_ports[0], total.input_ports[1])
        graph = influence_graph(builder.build(name="dilute_with_middleman"))

        model_slice = graph.slice("n2:out:out_0", threshold=0.01)
        assert "middleman" in model_slice.blocks
        subgraph = model_slice.subgraph
        for node in model_slice.scores:
            assert node in subgraph
            assert nx.has_path(subgraph, node, model_slice.target)

        # With the route intact, only the genuinely shared hops are bottlenecks.
        bottleneck_blocks = {
            graph.graph.nodes[node]["block"]
            for node in graph.bottlenecks("n2:out:out_0", threshold=0.01)
        }
        assert "middleman" in bottleneck_blocks
        assert "big" not in bottleneck_blocks  # a parallel input, never a cut

    def test_attribute_is_complete_on_a_target_fed_by_a_dense_mesh(self):
        # attribute() used to bound itself with an unpruned reachability sweep,
        # which hit its expansion budget on a dense model and then reported
        # "no path" with truncated=False.
        width, depth = 5, 5
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(1.0, name="src"))
        previous = [source.output_ports[0]] * width
        for layer_index in range(depth):
            layer = []
            for node_index in range(width):
                node = builder.add(
                    Adder(width, operators="+" * width, name=f"L{layer_index}_{node_index}")
                )
                for slot, port in enumerate(previous):
                    builder.connect(port, node.input_ports[slot])
                layer.append(node.output_ports[0])
            previous = layer
        probe = builder.add(Constant(2.0, name="probe_src"))
        relay = builder.add(Gain(0.5, name="relay"))
        target = builder.add(Adder(2, operators="++", name="tgt"))
        builder.connect(probe.output_ports[0], relay.input_ports[0])
        builder.connect(previous[0], target.input_ports[0])
        builder.connect(relay.output_ports[0], target.input_ports[1])
        graph = influence_graph(builder.build(name="mesh_plus_chain"), tau=1.0)

        result = graph.attribute("tgt:out:out_0", "probe_src:out:out_0")
        assert len(result.paths) == 1
        assert result.total_magnitude > 0.0
        assert not result.truncated
        assert graph.dominant_paths("tgt:out:out_0", 3)

    def test_matrix_weights_compose_as_an_upper_bound(self):
        # max|AB| <= k*max|A|*max|B|, so chaining largest-entries would
        # UNDER-state a two-hop product. The induced norm must dominate.
        from jaxonomy.analysis.influence import _scalarize

        left = np.ones((3, 3))
        right = np.ones((3, 3))
        assert _scalarize(left) * _scalarize(right) >= float(
            np.max(np.abs(right @ left))
        )

    def test_an_unknown_edge_behind_a_tiny_gain_is_still_surfaced(self):
        # The pruning bound must treat "no local gradient" as unbounded, not as
        # unit gain: otherwise a route is discarded before it ever reaches the
        # unmeasurable edge, and the slice reports no uncertainty at all.
        builder = jaxonomy.DiagramBuilder()
        signal = builder.add(Constant(0.3, name="s"))
        limit = builder.add(Constant(0.5, name="t"))
        compare = builder.add(Comparator(operator=">", name="cmp"))
        attenuate = builder.add(Gain(1e-9, name="tiny"))
        other = builder.add(Constant(1.0, name="big"))
        total = builder.add(Adder(2, operators="++", name="tgt"))
        builder.connect(signal.output_ports[0], compare.input_ports[0])
        builder.connect(limit.output_ports[0], compare.input_ports[1])
        builder.connect(compare.output_ports[0], attenuate.input_ports[0])
        builder.connect(attenuate.output_ports[0], total.input_ports[0])
        builder.connect(other.output_ports[0], total.input_ports[1])
        graph = influence_graph(builder.build(name="unknown_behind_tiny"))

        model_slice = graph.slice("tgt:out:out_0", threshold=0.01)
        assert {"cmp", "s", "t"} <= set(model_slice.blocks)
        assert model_slice.unknown_paths

    def test_a_zero_weight_upstream_of_an_unknown_edge_is_handled(self):
        # The pruning bound is `running * amplification`, and amplification is
        # infinite past an unknown edge — so a zero-weight edge produces NaN.
        # It must survive (unmeasurable, not absent) and leave finite scores.
        import math

        builder = jaxonomy.DiagramBuilder()
        zero = builder.add(Constant(0.0, name="z"))
        blocked = builder.add(Gain(0.0, name="zero_gain"))
        limit = builder.add(Constant(0.5, name="lim"))
        compare = builder.add(Comparator(operator=">", name="cmp"))
        other = builder.add(Constant(1.0, name="big"))
        total = builder.add(Adder(2, operators="++", name="tgt"))
        builder.connect(zero.output_ports[0], blocked.input_ports[0])
        builder.connect(blocked.output_ports[0], compare.input_ports[0])
        builder.connect(limit.output_ports[0], compare.input_ports[1])
        builder.connect(compare.output_ports[0], total.input_ports[0])
        builder.connect(other.output_ports[0], total.input_ports[1])
        graph = influence_graph(builder.build(name="zero_then_unknown"))

        model_slice = graph.slice("tgt:out:out_0", threshold=0.01)
        assert "cmp" in model_slice.blocks
        assert model_slice.unknown_paths
        assert all(math.isfinite(score) for score in model_slice.scores.values())
        # The zero gain is still correctly reported as a dead edge.
        assert ("zero_gain:in:in_0", "zero_gain:out:out_0") in {
            (entry["src"], entry["dst"]) for entry in graph.dead_edges()
        }

    def test_one_comparator_does_not_collapse_the_search(self):
        """An unmeasurable block upstream must not cost the rest of the answer.

        Comparators, quantizers and mode signals are the norm in hybrid models,
        so "there is an unknown edge somewhere upstream" cannot be allowed to
        switch off pruning across the model and truncate the search — nor to
        drop the region behind it. The unknown region is resolved by
        reachability, which is linear, rather than by path enumeration.
        """

        def mesh(width, depth, with_comparator):
            builder = jaxonomy.DiagramBuilder()
            source = builder.add(Constant(1.0, name="src"))
            feed = source.output_ports[0]
            if with_comparator:
                limit = builder.add(Constant(0.5, name="lim"))
                compare = builder.add(Comparator(operator=">", name="cmp"))
                builder.connect(source.output_ports[0], compare.input_ports[0])
                builder.connect(limit.output_ports[0], compare.input_ports[1])
                feed = compare.output_ports[0]
            previous = [feed] * width
            for layer_index in range(depth):
                layer = []
                for node_index in range(width):
                    node = builder.add(
                        Adder(
                            width,
                            operators="+" * width,
                            name=f"a{layer_index}_{node_index}",
                        )
                    )
                    for slot, port in enumerate(previous):
                        builder.connect(port, node.input_ports[slot])
                    layer.append(node.output_ports[0])
                previous = layer
            return builder.build(name="mesh"), f"a{depth - 1}_0:out:out_0"

        plain, target = mesh(6, 6, with_comparator=False)
        plain_slice = influence_graph(plain, tau=1.0).slice(target, 0.01)
        assert not plain_slice.truncated
        assert not plain_slice.unknown_paths

        gated, target = mesh(6, 6, with_comparator=True)
        gated_graph = influence_graph(gated, tau=1.0)
        gated_slice = gated_graph.slice(target, 0.01)
        assert not gated_slice.truncated
        assert gated_slice.unknown_paths
        # Adding a block cannot remove influence: everything the measurable
        # model retained is still there, plus the unmeasurable region.
        assert set(plain_slice.blocks) <= set(gated_slice.blocks)
        assert "cmp" in gated_slice.blocks

        # And the budget must not be what produced that answer: with the same
        # model and an effectively unbounded budget, nothing more is found.
        best, _, _, truncated = gated_graph._reach(
            gated_graph.resolve(target),
            32,
            "backward",
            threshold=0.01,
            max_expansions=10_000_000,
        )
        assert not truncated
        assert {gated_graph.graph.nodes[n]["block"] for n in best} <= set(
            gated_slice.blocks
        )

        # A denser instance stays bounded too — the failure was a scale cliff.
        big, big_target = mesh(10, 12, with_comparator=True)
        big_slice = influence_graph(big, tau=1.0).slice(big_target, 0.01)
        assert not big_slice.truncated

    def test_unknown_regions_stay_complete_and_connected_on_random_models(self):
        """Randomized check of the two invariants the unknown path must hold.

        Behind an unmeasurable edge the search switches from path enumeration
        to reachability, so both properties need checking on shapes that were
        not hand-picked: nothing structurally upstream of the target through an
        unknown edge may be dropped, and every retained node must still have a
        route to the target inside the slice.
        """
        import random

        rng = random.Random(20260730)
        for trial in range(20):
            builder = jaxonomy.DiagramBuilder()
            sources = [
                builder.add(Constant(0.3 + 0.1 * index, name=f"s{index}"))
                for index in range(3)
            ]
            limit = builder.add(Constant(0.5, name="lim"))
            compare = builder.add(Comparator(operator=">", name="cmp"))
            builder.connect(sources[0].output_ports[0], compare.input_ports[0])
            builder.connect(limit.output_ports[0], compare.input_ports[1])
            feeds = [
                compare.output_ports[0],
                sources[1].output_ports[0],
                sources[2].output_ports[0],
            ]
            last = None
            for index in range(rng.randint(3, 7)):
                junction = builder.add(
                    Adder(2, operators=rng.choice(["++", "+-"]), name=f"n{index}")
                )
                picks = rng.sample(feeds, 2)
                builder.connect(picks[0], junction.input_ports[0])
                builder.connect(picks[1], junction.input_ports[1])
                scaled = builder.add(
                    Gain(rng.choice([1e-9, 0.5, 3.0]), name=f"g{index}")
                )
                builder.connect(junction.output_ports[0], scaled.input_ports[0])
                feeds.append(scaled.output_ports[0])
                last = scaled
            graph = influence_graph(builder.build(name=f"trial{trial}"), tau=1.0)

            target = f"{last.name}:out:out_0"
            model_slice = graph.slice(target, threshold=0.01)
            structural = set(graph.structural_slice(target))
            if "cmp" in structural:
                assert "cmp" in model_slice.blocks
                assert model_slice.unknown_paths
            subgraph = model_slice.subgraph
            for retained in model_slice.scores:
                assert retained in subgraph
                assert nx.has_path(subgraph, retained, model_slice.target)
            # The retained set and the subgraph must describe the same slice:
            # keeping an edge whose endpoint is not in `scores` would drop that
            # block from `blocks` while still routing influence through it.
            assert set(subgraph.nodes) == set(model_slice.scores)

    def test_unknown_edges_do_not_truncate_other_topologies(self):
        """The frontier sweep must hold up outside the layered-mesh shape.

        Both halves of the unmeasurable question — reaching an unknown edge and
        going past it — are reachability, so no arrangement of comparators
        should push the path search into its budget.
        """
        cases = []

        # A long chain with unknown edges part-way along.
        builder = jaxonomy.DiagramBuilder()
        feed = builder.add(Constant(1.0, name="src")).output_ports[0]
        for index in range(24):
            if index in (6, 12, 18):
                limit = builder.add(Constant(0.5, name=f"lim{index}"))
                compare = builder.add(Comparator(operator=">", name=f"cmp{index}"))
                builder.connect(feed, compare.input_ports[0])
                builder.connect(limit.output_ports[0], compare.input_ports[1])
                feed = compare.output_ports[0]
            else:
                scaled = builder.add(Gain(1.1, name=f"g{index}"))
                builder.connect(feed, scaled.input_ports[0])
                feed = scaled.output_ports[0]
        cases.append((builder.build(name="chain"), "g23:out:out_0", 1.0))

        # Many comparators fanning into one junction.
        builder = jaxonomy.DiagramBuilder()
        fan = 24
        limit = builder.add(Constant(0.5, name="lim"))
        junction = builder.add(Adder(fan, operators="+" * fan, name="tgt"))
        for index in range(fan):
            source = builder.add(Constant(0.3 + 0.001 * index, name=f"s{index}"))
            compare = builder.add(Comparator(operator=">", name=f"c{index}"))
            builder.connect(source.output_ports[0], compare.input_ports[0])
            builder.connect(limit.output_ports[0], compare.input_ports[1])
            builder.connect(compare.output_ports[0], junction.input_ports[index])
        cases.append((builder.build(name="fan"), "tgt:out:out_0", 1.0))

        for diagram, target, tau in cases:
            graph = influence_graph(diagram, tau=tau)
            model_slice = graph.slice(target, threshold=0.01)
            assert not model_slice.truncated
            # Same computation with an effectively unbounded budget finds no
            # more, so the answer is not an artefact of the budget.
            best, _, _, truncated = graph._reach(
                graph.resolve(target),
                32,
                "backward",
                threshold=0.01,
                max_expansions=5_000_000,
            )
            assert not truncated
            assert {graph.graph.nodes[n]["block"] for n in best} <= set(
                model_slice.blocks
            )

    def test_scores_and_subgraph_agree_on_what_is_retained(self):
        """`blocks`/`scores` must not under-report what `edges` connects.

        A route from an unmeasurable edge to the target can run through blocks
        the numeric search pruned. Keeping their edges but dropping them from
        the retained set makes the slice omit the very blocks the influence
        travels through — here, the heater chain between a thermostat relay and
        the zone it heats.
        """
        builder = jaxonomy.DiagramBuilder()
        setpoint = builder.add(Constant(21.0, name="setpoint"))
        room = builder.add(Constant(19.5, name="room_temp"))
        thermostat = builder.add(Comparator(operator=">", name="thermostat"))
        builder.connect(setpoint.output_ports[0], thermostat.input_ports[0])
        builder.connect(room.output_ports[0], thermostat.input_ports[1])
        duty = builder.add(Gain(1.0, name="ssr_duty"))
        builder.connect(thermostat.output_ports[0], duty.input_ports[0])
        element = builder.add(Gain(2e-4, name="element_kW"))
        builder.connect(duty.output_ports[0], element.input_ports[0])
        flux = builder.add(Gain(1e-3, name="heat_flux"))
        builder.connect(element.output_ports[0], flux.input_ports[0])
        load = builder.add(Constant(4.0, name="ambient_load"))
        zone = builder.add(Adder(2, operators="+-", name="zone_balance"))
        builder.connect(flux.output_ports[0], zone.input_ports[0])
        builder.connect(load.output_ports[0], zone.input_ports[1])
        graph = influence_graph(builder.build(name="thermostat_zone"))

        model_slice = graph.slice("zone_balance:out:out_0", threshold=0.01)
        assert {"element_kW", "heat_flux"} <= set(model_slice.blocks)
        subgraph = model_slice.subgraph
        assert set(subgraph.nodes) == set(model_slice.scores)
        assert {
            graph.graph.nodes[n]["block"] for n in subgraph.nodes
        } == set(model_slice.blocks)

    def test_a_slice_with_no_edges_still_contains_its_target(self):
        # Slicing to a source, or to a target everything upstream falls below
        # the threshold for, leaves no edges. An edge-induced subgraph would be
        # empty and drop the target itself.
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(1.0, name="src"))
        attenuate = builder.add(Gain(1e-12, name="tiny"))
        builder.connect(source.output_ports[0], attenuate.input_ports[0])
        graph = influence_graph(builder.build(name="attenuated"))

        for target in ("src:out:out_0", "tiny:out:out_0"):
            model_slice = graph.slice(target, threshold=0.01)
            assert model_slice.edges == []
            assert set(model_slice.subgraph.nodes) == set(model_slice.scores)
            assert model_slice.target in model_slice.subgraph
            assert graph.bottlenecks(target) == []

    def test_truncation_is_reported_everywhere_it_leaks(self):
        """A truncated slice must not be consumed silently.

        ``bottlenecks`` and ``dominant_paths`` return bare lists, so they have
        nowhere to carry the flag — and a missing path is precisely what turns a
        non-bottleneck into an apparent single point of failure.
        """
        graph = influence_graph(gain_chain(), tau=1.0)
        node = graph.resolve("plant:xc")
        # Force the budget to bite on a model that would otherwise complete.
        _, _, _, truncated = graph._reach(
            node, 32, "backward", threshold=0.01, max_expansions=1
        )
        assert truncated

        starved = graph.slice("plant:xc", threshold=0.01, max_depth=32)
        object.__setattr__(starved, "truncated", True)
        assert "TRUNCATED" in repr(starved)
        assert "budget" in starved.report()

        import unittest.mock

        with unittest.mock.patch.object(
            type(graph), "slice", return_value=starved
        ):
            with pytest.warns(UserWarning, match="search budget"):
                graph.bottlenecks("plant:xc")
            with pytest.warns(UserWarning, match="search budget"):
                graph.dominant_paths("plant:xc", 2)

    def test_structural_slice_does_not_depend_on_the_jacobians(self):
        # A boolean over-approximation computed from the weighted graph would
        # be missing exactly the edges a Jacobian could not produce.
        builder = jaxonomy.DiagramBuilder()
        signal = builder.add(Constant(0.3, name="s"))
        limit = builder.add(Constant(0.5, name="t"))
        compare = builder.add(Comparator(operator=">", name="cmp"))
        builder.connect(signal.output_ports[0], compare.input_ports[0])
        builder.connect(limit.output_ports[0], compare.input_ports[1])
        diagram = builder.build(name="logic")
        graph = influence_graph(diagram)
        assert set(graph.structural_slice("cmp:out:out_0")) == {"cmp", "s", "t"}

    def test_a_non_differentiable_input_to_a_state_is_labelled(self):
        # The block's output reads only its state, so there is no feedthrough
        # pair to label — without an explicit input -> state edge the gate would
        # be reported as having no influence at all.
        class GatedIntegrator(jaxonomy.LeafSystem):
            def __init__(self, name=None, **kwargs):
                super().__init__(name=name, **kwargs)
                self.declare_input_port(name="gate")
                self.declare_input_port(name="val")
                self.declare_continuous_state(
                    shape=(), ode=self._ode, dtype=jnp.float64
                )
                self.declare_continuous_state_output(name="x")

            def _ode(self, time, state, *inputs, **params):
                gate, value = inputs
                return jnp.where(gate, value, 0.0)

        builder = jaxonomy.DiagramBuilder()
        signal = builder.add(Constant(0.3, name="s"))
        limit = builder.add(Constant(0.5, name="t"))
        compare = builder.add(Comparator(operator=">", name="cmp"))
        value = builder.add(Constant(3.0, name="val"))
        gated = builder.add(GatedIntegrator(name="gated"))
        builder.connect(signal.output_ports[0], compare.input_ports[0])
        builder.connect(limit.output_ports[0], compare.input_ports[1])
        builder.connect(compare.output_ports[0], gated.input_ports[0])
        builder.connect(value.output_ports[0], gated.input_ports[1])
        diagram = builder.build(name="gated_model")

        graph = influence_graph(diagram)
        edge = graph.graph.edges["gated:in:gate", "gated:xc"]
        assert edge["local_gradient"] is False
        model_slice = graph.slice("gated:xc", threshold=0.01)
        assert "cmp" in model_slice.blocks
        assert "gated:in:gate" in model_slice.unknown_nodes

    def test_wire_weights_carry_the_scale_ratio(self):
        # A sample-and-hold output node reports the post-tick value while its
        # consumer's input node holds the pre-tick one, so an unscaled identity
        # would break telescoping across the connection.
        from jaxonomy.library.linear_system import LTISystemDiscrete

        builder = jaxonomy.DiagramBuilder()
        block = builder.add(
            LTISystemDiscrete(
                A=jnp.array([[10.0]]),
                B=jnp.array([[0.0]]),
                C=jnp.array([[1.0]]),
                D=jnp.array([[0.0]]),
                dt=0.1,
                initialize_states=jnp.array([1.0]),
                name="filt",
            )
        )
        block.input_ports[0].fix_value(jnp.array([0.0]))
        sink = builder.add(Gain(3.0, name="g"))
        builder.connect(block.output_ports[0], sink.input_ports[0])
        diagram = builder.build(name="sampled")
        graph = influence_graph(diagram)

        source_scale = graph.graph.nodes["filt:out:out_0"]["scale"]
        sink_scale = graph.graph.nodes["g:in:in_0"]["scale"]
        wire = graph.graph.edges["filt:out:out_0", "g:in:in_0"]["weight"]
        assert wire == pytest.approx(float(source_scale[0] / sink_scale[0]))

    def test_trajectory_merge_keeps_a_partly_unknown_edge_unknown(self):
        # Reducing over the snapshots that happened to be differentiable would
        # let a "dead edge" verdict rest on incomplete evidence.
        from jaxonomy.analysis.influence import _merge_profiles

        first = nx.DiGraph()
        first.add_node("a", block="a", value=np.ones(1))
        first.add_node("b", block="b", value=np.ones(1))
        first.add_edge(
            "a",
            "b",
            kind="feedthrough",
            jacobian=None,
            relative=None,
            weight=0.0,
            magnitude=0.0,
            local_gradient=True,
            note="",
            tau_applied=False,
        )
        second = first.copy()
        second.edges["a", "b"].update(
            local_gradient=False, weight=float("nan"), magnitude=float("nan")
        )
        merged = _merge_profiles([first, second], "max")
        edge = merged.edges["a", "b"]
        assert edge["local_gradient"] is False
        assert "snapshot" in edge["note"]


class TestResolve:
    def test_resolves_port_objects_and_fragments(self):
        diagram = gain_chain()
        graph = influence_graph(diagram)
        assert graph.resolve(diagram["g1"].output_ports[0]) == "g1:out:out_0"
        assert graph.resolve("g1:out:out_0") == "g1:out:out_0"
        assert graph.resolve("plant:xc") == "plant:xc"

    def test_missing_node_names_the_convention(self):
        graph = influence_graph(gain_chain())
        with pytest.raises(KeyError, match="No influence-graph node matches"):
            graph.resolve("nonexistent_block:out:y")

    def test_ambiguous_fragment_lists_the_candidates(self):
        graph = influence_graph(gain_chain())
        with pytest.raises(KeyError, match="ambiguous"):
            graph.resolve("out_0")

    def test_invalid_options_are_rejected(self):
        diagram = gain_chain()
        with pytest.raises(ValueError, match="operating_point"):
            influence_graph(diagram, at="nowhere")
        with pytest.raises(ValueError, match="normalize"):
            influence_graph(diagram, normalize="loudly")
        with pytest.raises(ValueError, match="tau"):
            influence_graph(diagram, tau=0.0)
        with pytest.raises(ValueError, match="reduce"):
            influence_graph(diagram, reduce="median")


# ---------------------------------------------------------------------------
# Budgeted serialization
# ---------------------------------------------------------------------------


class TestInfluenceSubgraph:
    def test_focus_by_block_name_expands_to_its_signals(self):
        graph = influence_graph(gain_chain(), tau=1.0)
        result = influence_subgraph(graph, "plant", hops=1)
        assert set(result["focus"]) == {
            "plant:in:in_0",
            "plant:out:out_0",
            "plant:xc",
        }

    def test_budget_is_respected_and_drops_are_reported(self):
        graph = influence_graph(gain_chain(), tau=1.0)
        generous = influence_subgraph(graph, "plant:xc", budget_tokens=4000, hops=8)
        tight = influence_subgraph(graph, "plant:xc", budget_tokens=100, hops=8)
        assert len(tight["text"]) <= len(generous["text"])
        assert len(tight["edges"]) < len(generous["edges"])
        assert tight["dropped_edges"]
        assert not generous["dropped_edges"]
        assert generous["estimated_tokens"] == -(
            -len(generous["text"]) // CHARS_PER_TOKEN
        )

    def test_strongest_edges_survive_a_tight_budget(self):
        graph = influence_graph(gain_chain(), tau=1.0)
        tight = influence_subgraph(graph, "plant:xc", budget_tokens=110, hops=8)
        kept = [abs(float(entry["weight"])) for entry in tight["edges"]]
        dropped_edges = {
            (entry["src"], entry["dst"]) for entry in tight["dropped_edges"]
        }
        dropped = [
            abs(graph.graph.edges[edge]["magnitude"])
            for edge in dropped_edges
            if graph.graph.edges[edge]["local_gradient"]
        ]
        assert kept
        assert min(kept) >= max(dropped, default=0.0)

    def test_output_is_json_serializable_and_cites_stable_ids(self):
        import json

        graph = influence_graph(gain_chain(), tau=1.0)
        result = influence_subgraph(graph, "plant:xc", hops=4)
        json.dumps(result)
        for entry in result["nodes"]:
            assert entry["id"] in graph.graph
            assert entry["id"] in result["text"]
        assert result["conventions"]["tau_seconds"] == 1.0
        assert "relative" in result["conventions"]["weight_meaning"]

    def test_enrichment_carries_units_and_rates(self):
        from jaxonomy.framework.units import parse_unit

        builder = jaxonomy.DiagramBuilder()
        source = builder.add(Constant(1.0, name="src"))
        plant = builder.add(Integrator(0.5, name="plant"))
        builder.connect(source.output_ports[0], plant.input_ports[0])
        diagram = builder.build(name="united")
        plant.output_ports[0].units = parse_unit("m")

        graph = influence_graph(diagram, tau=1.0)
        result = influence_subgraph(graph, "plant:out:out_0", hops=2)
        node = next(e for e in result["nodes"] if e["id"] == "plant:out:out_0")
        assert "m" in node["units"] and node["units"] != "-"
        assert node["sample_time"] == "continuous"
        assert node["block_type"] == "Integrator"

    def test_budget_goes_to_what_reaches_the_focus_not_local_weight(self):
        """A near-cancelling junction must not eat the budget.

        Its two inputs nearly cancel, so its *local* elasticity is enormous —
        the output is a small difference of large numbers — while it passes
        almost nothing onward for exactly that reason. Ranking by local weight
        fills the serialization with such junctions and never reaches the block
        that drives the target; ranking by influence carried to the focus does.
        """
        builder = jaxonomy.DiagramBuilder()
        # A live path: source -> gain -> junction -> target.
        driver = builder.add(Constant(2.0, name="driver"))
        relay = builder.add(Gain(1.0, name="relay"))
        # A near-cancelling junction whose tiny output also reaches the target.
        left = builder.add(Constant(1.0, name="bridge_hi"))
        right = builder.add(Constant(0.999, name="bridge_lo"))
        bridge = builder.add(Adder(2, operators="+-", name="bridge"))
        total = builder.add(Adder(2, operators="++", name="total"))
        builder.connect(driver.output_ports[0], relay.input_ports[0])
        builder.connect(left.output_ports[0], bridge.input_ports[0])
        builder.connect(right.output_ports[0], bridge.input_ports[1])
        builder.connect(relay.output_ports[0], total.input_ports[0])
        builder.connect(bridge.output_ports[0], total.input_ports[1])
        graph = influence_graph(builder.build(name="bridge_model"))

        # The bridge's own edges are by far the strongest in the graph...
        bridge_weight = abs(
            graph.graph.edges["bridge:in:in_0", "bridge:out:out_0"]["weight"]
        )
        relay_weight = abs(
            graph.graph.edges["relay:in:in_0", "relay:out:out_0"]["weight"]
        )
        assert bridge_weight > 100 * relay_weight

        # ...yet the tightest budget spends itself on the live path, and the
        # bridge only appears once there is room to spare.
        tight = influence_subgraph(
            graph, "total:out:out_0", budget_tokens=230, hops=6
        )
        assert "relay" in tight["blocks"]
        assert "bridge" not in tight["blocks"]
        kept = {(e["src"], e["dst"]) for e in tight["edges"]}
        assert ("relay:in:in_0", "relay:out:out_0") in kept

        # With a little more room the whole live path is in before the bridge's
        # own inputs are.
        roomier = influence_subgraph(
            graph, "total:out:out_0", budget_tokens=250, hops=6
        )
        assert "driver" in roomier["blocks"]
        assert "bridge_hi" not in roomier["blocks"]

    def test_footer_separates_negligible_from_unknown(self):
        """Absent-because-small and absent-because-truncated must not look alike.

        To a reader they mean opposite things: the first is an answer, the
        second is a reason to abstain. Without the footer both are just missing,
        and treating a truncation as negligible is exactly the fabrication this
        serialization exists to avoid.
        """
        graph = influence_graph(gain_chain(), tau=1.0)

        generous = influence_subgraph(graph, "plant:xc", budget_tokens=4000, hops=8)
        assert "coverage:" in generous["text"]
        assert "negligible, NOT unknown" in generous["text"]
        assert "WARNING" not in generous["text"]
        assert not generous["dropped_edges"]

        tight = influence_subgraph(graph, "plant:xc", budget_tokens=120, hops=8)
        assert tight["dropped_edges"]
        assert "WARNING" in tight["text"]
        assert "UNKNOWN, not negligible" in tight["text"]
        # The count in the warning is the count actually dropped.
        assert str(len(tight["dropped_edges"])) in tight["text"]

    def test_invalid_arguments_are_rejected(self):
        graph = influence_graph(gain_chain())
        with pytest.raises(ValueError, match="direction"):
            influence_subgraph(graph, "plant:xc", direction="upward")
        with pytest.raises(ValueError, match="budget_tokens"):
            influence_subgraph(graph, "plant:xc", budget_tokens=0)
