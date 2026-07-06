# Jaxonomy documentation

**Jaxonomy** is a Python package for simulating **hybrid dynamical systems** described as **block diagrams**: wired blocks (integrators, gains, custom subsystems, acausal networks, and more), continuous and discrete states, and event/zero-crossing logic. The runtime is built around **JAX**, so you get JIT-friendly execution and **automatic differentiation** where the model allows it, while keeping a NumPy-style API for numerics.

The library runs **entirely locally** and emits the Collimator-format JSON for serialised models. There is no hosted cloud service — see the [About](about.md) page for the project's scope.

---

## Install

```bash
pip install jaxonomy
```

Use a virtual environment when possible. Platform notes, optional extras (`[safe]`, `[nmpc]`, `[all]`), and **development installs from a git clone** are covered in the **[installation guide](installation.md)**.

---

## Where to go next

| Goal | Link |
|------|------|
| First simulation walkthrough | [Tutorials → Getting started](tutorials/01-getting-started.ipynb) |
| Shorter topical guides | [Tutorials index](tutorials/index.md) |
| Applied notebooks (control, MPC, ML, …) | [Examples](examples/index.md) |
| `DiagramBuilder`, `LeafSystem`, ports | [Framework](framework.md) |
| Built-in blocks | [Block library](library.md) |
| `simulate`, solvers, options | [Simulation](simulation.md) |
| Training / optimization helpers | [Optimization](optimization.md) |

---

## Minimal pattern

1. Add blocks with `DiagramBuilder`, `connect` outputs to inputs, then `build()`.
2. Call `jaxonomy.simulate(diagram, start_time, end_time, ...)` (see [Simulation](simulation.md) for `SimulatorOptions` and results handling).

The [Getting started](tutorials/01-getting-started.ipynb) tutorial builds a simple mass–spring–damper-style diagram step by step.

---

## Build this site locally

```bash
pip install -r requirements.docs.txt
mkdocs serve
```

Source for this page: `docs/index.md` (repository root).

---

## License and attribution

This project is released under the [MIT License](https://mit-license.org/). See the `LICENSE.md` file in the repository for the full text.

**Provenance:** This library is derived from the MIT-licensed open-source Python package **pycollimator**, developed by **Collimator, Inc.**
