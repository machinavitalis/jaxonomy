# SPDX-License-Identifier: MIT

import json
import traceback

from jaxonomy.framework import LeafSystem
from jaxonomy.mcp._helpers import collect_blocks
from jaxonomy.mcp.server import mcp


@mcp.tool()
def list_blocks() -> str:
    """
    List all available block types in the Jaxonomy library.

    Returns JSON with block names, descriptions, and
    key parameters.

    Use this before create_diagram to know what blocks
    are available.
    """
    try:
        from jaxonomy import library

        blocks = []
        for name in sorted(dir(library)):
            if name.startswith("_"):
                continue
            obj = getattr(library, name, None)
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, LeafSystem):
                continue
            if obj is LeafSystem:
                continue
            doc = (obj.__doc__ or "").strip().split("\n")[0] if obj.__doc__ else ""
            blocks.append({"name": name, "doc": doc})
        return json.dumps({"blocks": blocks}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def validate_model(model_json: str) -> str:
    """
    Validate a Jaxonomy model JSON string.

    Args:
        model_json: JSON string of the model
                    (Jaxonomy model format)

    Returns JSON with:
        valid: bool
        errors: list of error strings
        warnings: list of warning strings
    """
    try:
        model_dict = json.loads(model_json)
        from jaxonomy.dashboard.serialization.from_model_json import load_model
        from jaxonomy.framework.validation import validate_diagram

        sim_context = load_model(model_dict)
        result = validate_diagram(sim_context.diagram)
        return json.dumps(
            {
                "valid": result.valid,
                "errors": [str(e) for e in result.errors],
                "warnings": [str(w) for w in result.warnings],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "error": str(e),
                "errors": [str(e)],
                "warnings": [],
                "traceback": traceback.format_exc(),
            }
        )


@mcp.tool()
def explain_model(model_json: str) -> str:
    """
    Generate a plain English description of what a
    Jaxonomy model does.

    Args:
        model_json: JSON string of the model

    Returns a human-readable description of the model
    structure, blocks, and signal flow.
    """
    try:
        model_dict = json.loads(model_json)
        from jaxonomy.dashboard.serialization.from_model_json import load_model

        sim_context = load_model(model_dict)
        diagram = sim_context.diagram

        params = diagram.list_parameters()
        blocks = collect_blocks(diagram)

        lines = [
            f"Model: {diagram.name}",
            f"Blocks: {len(blocks)}",
            "",
            "Block summary:",
        ]
        for block_name in sorted(blocks.keys()):
            block = blocks[block_name]
            lines.append(f"  {block_name}: {type(block).__name__}")

        if params:
            lines.append("")
            lines.append("Parameters:")
            items = list(params.items())[:20]
            for k, v in items:
                lines.append(f"  {k} = {v!r}")
            if len(params) > 20:
                lines.append(f"  ... and {len(params) - 20} more")

        return json.dumps({"description": "\n".join(lines)})
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
