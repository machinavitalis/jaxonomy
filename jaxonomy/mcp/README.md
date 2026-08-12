# Jaxonomy MCP Server

Exposes Jaxonomy simulation as tools for AI agents over stdio.

**User documentation lives at [py.jaxonomy.com/mcp](https://py.jaxonomy.com/mcp/)**
— install, client configuration, the tool reference, and limitations. Keep that
page canonical; this file covers only what a contributor needs.

## Layout

- `server.py` — the `FastMCP` instance and the `main()` entry point. Tool
  modules are imported at the bottom because their `@mcp.tool()` decorators
  need `mcp` to exist first.
- `tools/model_tools.py` — `list_blocks`, `validate_model`, `explain_model`
- `tools/simulate_tools.py` — `run_simulation`, `fit_parameters`
- `tools/analysis_tools.py` — `linearize_model`, `influence_subgraph`
- `_helpers.py` — shared model-JSON deserialization and result formatting

Tests are in `test/mcp/test_mcp_tools.py`.

## Running it locally

```bash
pip install -e .[mcp]
jaxonomy-mcp          # or: python -m jaxonomy.mcp.server
```

## Registry entry

The server is published to the [MCP Registry](https://registry.modelcontextprotocol.io)
as `io.github.machinavitalis/jaxonomy`, described by `server.json` at the repo
root. Two things must stay in step when the package is released:

- the three version fields in `server.json` (`version`, and the package's
  `version` and pinned `--from` extra) must match the released PyPI version;
- the `mcp-name:` comment in the root `README.md` must match `server.json`'s
  `name` — the registry reads it from the README as published to PyPI to verify
  package ownership, so a release without it will fail validation.
