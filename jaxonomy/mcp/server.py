# SPDX-License-Identifier: MIT

"""Jaxonomy MCP server entrypoint."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="jaxonomy",
    instructions=(
        "Jaxonomy simulation engine. Build block-diagram "
        "models of dynamical systems, run simulations, "
        "fit parameters to data, and analyze results."
    ),
)


def main() -> None:
    """Run the MCP server (stdio transport by default)."""
    mcp.run()


# Register tools (imports depend on ``mcp`` above)
from jaxonomy.mcp.tools import simulate_tools  # noqa: E402, F401
from jaxonomy.mcp.tools import model_tools  # noqa: E402, F401
from jaxonomy.mcp.tools import analysis_tools  # noqa: E402, F401


if __name__ == "__main__":
    main()
