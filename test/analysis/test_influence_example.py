# SPDX-License-Identifier: MIT

"""Gate on the shippable claims in ``docs/examples/influence_graph_model_slicing.py``.

The script is a shippable surface, and its narrative asserts specific numbers:
that the path attribution matches finite differences, that the quantitative
slice drops exactly the negligible block, that the secant probe recovers the
blocks a saturated derivative writes off, and that the trajectory profile shows
the driver edge switching on. Each of those is re-derived here from the script's
own model, so the prose cannot drift from the behaviour.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import jaxonomy
from jaxonomy.analysis import influence_graph

pytestmark = pytest.mark.minimal

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "influence_graph_model_slicing.py"
)


@pytest.fixture(scope="module")
def example():
    """The example's model builder, imported without running its narrative."""
    spec = importlib.util.spec_from_file_location(
        "_influence_example", EXAMPLE, submodule_search_locations=[]
    )
    module = importlib.util.module_from_spec(spec)
    source = EXAMPLE.read_text(encoding="utf-8")
    # Execute only up to the first banner() call: everything above is the model
    # and its parameters, everything below is the printed walk-through.
    header = source.split("# ------", 1)[0]
    exec(compile(header, str(EXAMPLE), "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture(scope="module")
def trajectory(example):
    """The graph the narrative's headline sections are built on."""
    diagram = example.make_motor_loop()
    context = diagram.create_context()
    results = example.step_response(diagram, context)
    graph = influence_graph(
        diagram,
        context,
        at="trajectory",
        results=results,
        n_snapshots=8,
        tau=example.TAU,
    )
    return diagram, context, results, graph


def test_example_file_exists_and_defines_the_model(example):
    assert callable(example.make_motor_loop)
    assert len(list(example.make_motor_loop().leaf_systems)) == 19


def test_attribution_matches_finite_differences(example):
    import jax.numpy as jnp

    diagram = example.make_motor_loop()
    context = diagram.create_context()
    graph = influence_graph(diagram, context, normalize="none", tau=1.0)

    def shaft_acceleration(load_value):
        load_input = diagram["torque_balance"].input_ports[2]
        with load_input.fixed(jnp.asarray(load_value)):
            return float(
                np.asarray(diagram["speed"].eval_time_derivatives(context)).reshape(())
            )

    step = 1e-7
    secant = (
        shaft_acceleration(example.T_LOAD + step)
        - shaft_acceleration(example.T_LOAD - step)
    ) / (2 * step)
    total = graph.attribute("speed:xc", "load_torque:out:out_0").total
    assert total == pytest.approx(-1.0 / example.J_ROT, rel=1e-9)
    assert total == pytest.approx(secant, rel=1e-6)


def test_quantitative_slice_drops_exactly_the_stiction_term(trajectory):
    _, _, _, graph = trajectory
    structural = set(graph.structural_slice("speed:xc"))
    quantitative = set(graph.slice("speed:xc", threshold=0.01).blocks)
    assert structural - quantitative == {"stiction"}


def test_the_model_has_no_degenerate_normalizers(trajectory):
    # The narrative claims trajectory mode sidesteps the zero-signal problem;
    # a node pinned to scale_floor would make its elasticities meaningless.
    _, _, _, graph = trajectory
    assert graph.nodes_at_scale_floor() == []


def test_slice_narrows_as_tau_shrinks(example, trajectory):
    diagram, context, results, _ = trajectory
    sizes = []
    for tau in (example.TAU_MECHANICAL, 0.1, 0.01, example.TAU_ELECTRICAL):
        graph = influence_graph(
            diagram,
            context,
            at="trajectory",
            results=results,
            n_snapshots=8,
            tau=tau,
        )
        sizes.append(len(graph.slice("speed:xc", threshold=0.01).blocks))
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 18 and sizes[-1] < sizes[0]


def test_dominant_paths_separate_the_integral_and_proportional_routes(trajectory):
    _, _, _, graph = trajectory
    paths = graph.dominant_paths("speed:xc", k=2)
    routes = [{graph.graph.nodes[n]["block"] for n in entry["nodes"]} for entry in paths]
    assert any("integral" in route for route in routes)
    assert any("kp" in route and "integral" not in route for route in routes)


def test_probe_recovers_the_blocks_a_saturated_derivative_writes_off(example):
    saturated = example.make_motor_loop(kp=0.2)
    context = saturated.create_context()

    local = influence_graph(saturated, context, tau=example.TAU)
    probed = influence_graph(saturated, context, tau=example.TAU, probe=example.PROBE)

    local_blocks = set(local.slice("speed:xc", 0.01).blocks)
    probed_blocks = set(probed.slice("speed:xc", 0.01).blocks)
    assert probed_blocks - local_blocks == {
        "integral",
        "ki",
        "kp",
        "speed_error",
        "v_cmd",
        "w_ref",
    }
    assert len(probed_blocks) == local.n_blocks

    dead = {(entry["src"], entry["dst"]) for entry in local.dead_edges()}
    assert ("driver:in:in_0", "driver:out:out_0") in dead
    assert probed.dead_edges() == []


def test_trajectory_profile_shows_the_driver_leaving_the_rail(example):
    saturated = example.make_motor_loop(kp=0.2)
    context = saturated.create_context()
    results = example.step_response(saturated, context)
    graph = influence_graph(
        saturated,
        context,
        at="trajectory",
        results=results,
        n_snapshots=7,
        tau=example.TAU,
    )
    profile = graph.graph.edges["driver:in:in_0", "driver:out:out_0"]["profile"]
    assert profile[0] == pytest.approx(0.0)
    assert profile[-1] > 0.0
    # reduce="max" keeps the edge, so the control path survives the slice.
    kept = set(graph.slice("speed:xc", 0.01).blocks)
    assert {"w_ref", "speed_error", "kp", "ki", "integral", "driver"} <= kept


@pytest.mark.slow
def test_example_script_runs_end_to_end():
    """The narrative itself executes — no stale API call in the printed walk-through."""
    import os
    import subprocess
    import sys

    # Point the child at the same jaxonomy this process imported. A script's
    # sys.path[0] is its own directory, so without this the subprocess would
    # resolve `jaxonomy` through whatever is installed rather than the checkout
    # under test — which silently passes or fails for the wrong reason.
    root = str(Path(jaxonomy.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "relative agreement with finite differences" in completed.stdout
