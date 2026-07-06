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
                "is_stable": bool(lin.is_stable()),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
