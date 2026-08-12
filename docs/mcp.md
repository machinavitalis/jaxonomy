# MCP server

Jaxonomy ships a [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes the engine as tools an AI agent can call directly. Instead
of writing Jaxonomy code and running it, the agent enumerates the block library,
builds and validates a model, runs the simulation, and reads the actual numbers
back.

This page is the reference for that server. If you are writing Python by hand,
you do not need any of it — `pip install jaxonomy` is enough. If you want an
agent to *write* Jaxonomy for you rather than *drive* it, point it at
[Using Jaxonomy from an AI agent](agents.md) instead; the two are complementary.

The server is registered in the
[MCP Registry](https://registry.modelcontextprotocol.io) as
`io.github.machinavitalis/jaxonomy`.

## Install

The server lives behind an optional extra, so it is not installed by default:

```bash
pip install jaxonomy[mcp]
```

## Configure a client

The server speaks stdio. Point your client at the `jaxonomy-mcp` entry point, or
equivalently at `python -m jaxonomy.mcp.server`.

**Claude Code:**

```bash
claude mcp add jaxonomy -- jaxonomy-mcp
```

**Claude Desktop** — in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jaxonomy": {
      "command": "jaxonomy-mcp"
    }
  }
}
```

**Without installing first**, `uvx` can fetch the package and the extra in one
step:

```bash
uvx --from 'jaxonomy[mcp]' jaxonomy-mcp
```

That is convenient for a one-off trial, but `uvx` builds a throwaway
environment, so it pulls JAX and its dependencies on every cold start. For
regular use, install into a real environment and point the client at that
interpreter.

Whichever form you use, the interpreter running the server must be the one where
`jaxonomy[mcp]` is installed. A client that launches a bare `python` may pick up
a different environment; give it an absolute path to the interpreter if the
server fails to start.

## Tools

The server exposes seven tools. Models are passed as JSON strings in Jaxonomy's
model format; `list_blocks` is the usual starting point because it tells the
agent what it has to work with.

| Tool | What it does |
|---|---|
| `list_blocks` | Catalogue of available library block types, with descriptions and key parameters. |
| `validate_model` | Checks a model JSON for structural and validation problems; returns `valid`, `errors`, `warnings`. |
| `explain_model` | Plain-English description of a model's blocks, parameters, and signal flow. |
| `run_simulation` | Runs a simulation over `[t_start, t_stop]`, recording named signals (e.g. `integrator.out_0`). Selectable `jax` or `numpy` backend. |
| `fit_parameters` | Fits chosen parameters to measured data supplied as CSV, via finite-difference gradients and Adam, with optional bounds. |
| `linearize_model` | Linearizes around an operating point; returns `A`, `B`, `C`, `D` and the eigenvalues. |
| `influence_subgraph` | Serializes what actually drives a chosen signal — the dependency structure weighted by autodiff Jacobians, expanded strongest-edge-first under a token budget. |

`influence_subgraph` exists for models too large to hand to an agent whole: it
answers "what drives this signal, and by how much" while keeping the response
inside a token budget, so what gets dropped is what mattered least.

## Limitations

- `fit_parameters` uses **finite-difference** gradients, not Jaxonomy's
  end-to-end autodiff. It is a convenience path for an agent holding a CSV, not
  the recommended way to calibrate a model — for that, write the `jax.grad` loop
  directly (see [Using Jaxonomy from an AI agent](agents.md)).
- Models cross the boundary as JSON, so anything requiring a custom Python
  `LeafSystem` cannot be expressed through these tools. Custom blocks are a
  code-writing task.
- The server is stdio-only; there is no hosted or HTTP transport.
