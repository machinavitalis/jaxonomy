# Jaxonomy MCP Server

Exposes Jaxonomy simulation as tools for Claude and other AI agents.

## Installation

```bash
pip install jaxonomy[mcp]
```

## Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jaxonomy": {
      "command": "python",
      "args": ["-m", "jaxonomy.mcp.server"]
    }
  }
}
```

Use the same Python interpreter where `jaxonomy` and `jaxonomy[mcp]` are installed (or use `jaxonomy-mcp` if installed via pip entry point).

## Available Tools

- **list_blocks**: catalog of library block types (`LeafSystem` subclasses)
- **validate_model**: check model JSON for structural / validation issues
- **explain_model**: plain-text summary of blocks and parameters
- **run_simulation**: execute a simulation from model JSON
- **fit_parameters**: fit selected parameters to CSV data (finite-difference gradients + Adam)
- **linearize_model**: compute linearized A, B, C, D and eigenvalues

## Usage with Claude

Once configured, Claude can run simulations directly, for example:

> Build a mass-spring-damper model and show me how damping ratio affects settling time

The agent can call `list_blocks`, construct model JSON, run `run_simulation`, and interpret results.

## Console entry point

After install:

```bash
jaxonomy-mcp
```

This runs the same MCP server as `python -m jaxonomy.mcp.server`.
