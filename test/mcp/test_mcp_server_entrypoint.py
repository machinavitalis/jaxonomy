# SPDX-License-Identifier: MIT

"""Coverage for the MCP server's entry point.

`test_mcp_tools.py` calls the tool functions directly, so nothing else in the
suite imports `jaxonomy/mcp/server.py` — the only module that touches
`FastMCP`. The quick-test CI job installs the `[mcp]` extra so these run.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest

# The extra stays optional for local development.
pytest.importorskip("mcp")

pytestmark = pytest.mark.minimal


def test_installed_mcp_provides_fastmcp():
    """An `mcp` that imports is not enough; it must still provide FastMCP.

    An assertion rather than `importorskip` on purpose: a wrong-major `mcp`
    should fail here, not skip.
    """
    assert importlib.util.find_spec("mcp.server.fastmcp") is not None, (
        f"mcp {importlib.metadata.version('mcp')} does not provide "
        "`mcp.server.fastmcp`, which jaxonomy/mcp/server.py imports. "
        "The [mcp] extra must resolve to >=1.2,<2."
    )


def test_server_module_imports_and_exposes_main():
    """`jaxonomy-mcp` is declared as `jaxonomy.mcp.server:main` — resolve it."""
    from jaxonomy.mcp import server

    assert callable(server.main), "console entry point `main` is not callable"
    assert server.mcp is not None, "FastMCP instance was not constructed"
