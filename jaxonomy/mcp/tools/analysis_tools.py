# SPDX-License-Identifier: MIT

import json
import traceback

import jax.numpy as jnp
import numpy as np

from jaxonomy.mcp._helpers import apply_input_values
from jaxonomy.mcp.server import mcp


@mcp.tool()
def linearize_model(
    model_json: str,
    state_values: str,
    input_values: str,
) -> str:
    """
    Linearize a model around an operating point.

    Args:
        model_json: JSON string of the model
        state_values: JSON string mapping state names
                      to values at operating point, or a JSON array
                      of continuous state components for a single
                      continuous-state subsystem (in diagram order).
                      Example: '{"x": 0.0, "v": 1.0}' or '[0.0, 1.0]'
        input_values: JSON string mapping input names
                      to values.
                      Example: '{"u": 0.5}' as 'block.in_0': value

    Returns JSON with:
        A: state matrix (list of lists)
        B: input matrix (list of lists)
        C: output matrix (list of lists)
        D: feedthrough matrix (list of lists)
        eigenvalues: list of {real, imag} dicts
        is_stable: bool
    """
    try:
        from jaxonomy.dashboard.serialization.from_model_json import load_model
        from jaxonomy.library.linear_system import linearize

        model_dict = json.loads(model_json)
        sim_context = load_model(model_dict)
        diagram = sim_context.diagram

        iv = json.loads(input_values) if input_values.strip() else {}
        if iv:
            apply_input_values(
                diagram, {str(k): float(v) for k, v in iv.items()}
            )

        ctx = diagram.create_context()
        sv_raw = json.loads(state_values) if state_values.strip() else {}
        if sv_raw:
            if isinstance(sv_raw, list):
                sub_states = [jnp.array(sv_raw, dtype=jnp.float64)]
            elif isinstance(sv_raw, dict):
                sub_states = [
                    jnp.array(
                        [float(sv_raw[k]) for k in sorted(sv_raw, key=str)],
                        dtype=jnp.float64,
                    )
                ]
            else:
                raise ValueError("state_values must be a JSON array or object")
            subs = ctx.continuous_subcontexts
            if len(subs) != 1:
                raise ValueError(
                    "Automatic state_values application requires exactly one "
                    "continuous-state subsystem; leave state_values as '{}' to "
                    "use defaults."
                )
            n = subs[0].num_continuous_states
            if int(sub_states[0].size) != n:
                raise ValueError(
                    f"state vector length {sub_states[0].size} != "
                    f"expected {n} continuous states"
                )
            ctx = ctx.with_continuous_state(sub_states)

        lin = linearize(diagram, ctx)
        A = np.asarray(jnp.asarray(lin.A))
        B = np.asarray(jnp.asarray(lin.B))
        C = np.asarray(jnp.asarray(lin.C))
        D = np.asarray(jnp.asarray(lin.D))
        eigs = np.asarray(lin.eigenvalues())
        ev_json = [
            {"real": float(np.real(z)), "imag": float(np.imag(z))} for z in eigs
        ]
        return json.dumps(
            {
                "A": A.tolist(),
                "B": B.tolist(),
                "C": C.tolist(),
                "D": D.tolist(),
                "eigenvalues": ev_json,
                "is_stable": bool(lin.is_stable),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def influence_subgraph(
    model_json: str,
    focus: str,
    budget_tokens: int = 1500,
    hops: int = 4,
    threshold: float = 0.0,
    tau: float = 1.0,
    probe: float = 0.0,
    input_values: str = "",
) -> str:
    """
    Serialize the sensitivity-weighted neighbourhood of one or more signals.

    Answers "what actually drives this signal, and by how much" on a model too
    large to serialize whole. The model's dependency structure supplies the
    edges; autodiff supplies the weights; expansion takes the strongest edges
    first and stops at a token budget, so what gets dropped is what mattered
    least.

    Weights are relative (dimensionless) sensitivities: a path's product is the
    relative end-to-end sensitivity, so 0.4 reads as "a 1% change here is a
    0.4% change there". Edges into a continuous state are scaled by ``tau``
    seconds, since a state derivative is a rate rather than a gain.

    Args:
        model_json: JSON string of the model
        focus: Comma-separated focus points. Each may be a block name
               ("plant", which expands to all of that block's signals), a node
               id ("plant:out:out_0", "plant:xc"), or any unambiguous fragment.
        budget_tokens: Approximate ceiling on the returned text (~4 chars/token)
        hops: Graph edges to expand from the focus. Nodes are signals, so
              crossing one block costs two hops.
        threshold: Minimum edge |weight| to include
        tau: Seconds represented by an edge into a continuous state
        probe: Relative step for the secant cross-check on zero-derivative
               edges (0 disables). Use a non-zero value on models with
               saturation, quantization, or dead zones, whose exact local
               derivative is zero where the block is flat even though the
               connection carries information.
        input_values: JSON object fixing input ports, as 'block.in_0': value

    Returns JSON with:
        text: the rendered context, one line per signal and per edge
        nodes, edges: the same content structured
        blocks: block name paths included
        estimated_tokens, dropped_edges: what the budget cost
        conventions: how to read a weight
    """
    try:
        from jaxonomy.analysis import influence_graph
        from jaxonomy.analysis.influence_context import influence_subgraph as build
        from jaxonomy.dashboard.serialization.from_model_json import load_model

        model_dict = json.loads(model_json)
        sim_context = load_model(model_dict)
        diagram = sim_context.diagram

        iv = json.loads(input_values) if input_values.strip() else {}
        if iv:
            apply_input_values(diagram, {str(k): float(v) for k, v in iv.items()})

        focus_specs = [part.strip() for part in focus.split(",") if part.strip()]
        if not focus_specs:
            raise ValueError(
                "focus must name at least one block, node id, or fragment"
            )

        graph = influence_graph(
            diagram,
            diagram.create_context(),
            tau=tau,
            probe=probe if probe > 0 else None,
        )
        result = build(
            graph,
            focus_specs,
            budget_tokens=budget_tokens,
            hops=hops,
            threshold=threshold,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
