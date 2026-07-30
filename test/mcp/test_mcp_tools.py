# SPDX-License-Identifier: MIT

import json

import pytest

pytest.importorskip("mcp")

from jaxonomy.mcp.tools.model_tools import explain_model, list_blocks, validate_model
from jaxonomy.mcp.tools.simulate_tools import run_simulation

pytestmark = pytest.mark.minimal


def build_and_serialize_simple_model() -> str:
    from jaxonomy import DiagramBuilder, library
    from jaxonomy.dashboard.serialization.to_model_json import convert

    builder = DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(library.Gain(gain=1.0, name="gain"))
    integ = builder.add(library.Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="mcp_test")
    model, _ = convert(diagram)
    return json.dumps(model.to_dict())


def test_list_blocks_returns_valid_json():
    result = list_blocks()
    data = json.loads(result)
    assert "blocks" in data
    assert len(data["blocks"]) > 0


def test_validate_valid_model():
    model_json = build_and_serialize_simple_model()
    result = validate_model(model_json)
    data = json.loads(result)
    assert data["valid"] is True


def test_run_simulation_returns_results():
    model_json = build_and_serialize_simple_model()
    result = run_simulation(
        model_json=model_json,
        t_start=0.0,
        t_stop=2.0,
        recorded_signals=["integ.out_0"],
    )
    data = json.loads(result)
    assert "error" not in data
    assert "time" in data
    assert "signals" in data
    assert "integ.out_0" in data["signals"]


def test_tools_handle_invalid_json_gracefully():
    result = validate_model("not valid json{{{")
    data = json.loads(result)
    assert "error" in data
    assert data.get("valid") is False


def test_explain_model_returns_description():
    model_json = build_and_serialize_simple_model()
    out = explain_model(model_json)
    data = json.loads(out)
    assert "description" in data
    assert "integ" in data["description"]


def test_influence_subgraph_returns_weighted_context():
    from jaxonomy.mcp.tools.analysis_tools import influence_subgraph

    model_json = build_and_serialize_simple_model()
    result = influence_subgraph(
        model_json=model_json,
        focus="integ",
        budget_tokens=800,
        hops=4,
        tau=1.0,
    )
    data = json.loads(result)
    assert "error" not in data, data.get("traceback", "")
    assert set(data["focus"]) >= {"integ:xc", "integ:out:out_0"}
    assert "gain" in data["blocks"]
    # The weights are real: a unity gain has elasticity 1, and the connection
    # into the integrator's state carries the tau=1 s step.
    edges = {(e["src"], e["dst"]): e for e in data["edges"]}
    assert edges[("gain:in:in_0", "gain:out:out_0")]["weight"].startswith("+1")
    assert data["conventions"]["tau_seconds"] == 1.0
    assert data["estimated_tokens"] <= 800
    assert data["text"].startswith("# influence subgraph of model")


def test_influence_subgraph_reports_a_bad_focus():
    from jaxonomy.mcp.tools.analysis_tools import influence_subgraph

    result = influence_subgraph(
        model_json=build_and_serialize_simple_model(), focus="no_such_block"
    )
    data = json.loads(result)
    assert "error" in data
    assert "No influence-graph node matches" in data["error"]
