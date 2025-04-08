## Getting started

### Prerequisites

Python 3.10 or later is required.

### Installation steps

```bash
pip install pycollimator
```

### Optional dependencies

Nonlinear MPC blocks require `IPOPT` to be preinstalled.

- On Ubuntu: `sudo apt install coinor-libipopt-dev`.
- On macOS: `brew install ipopt`.

On macOS with Apple Silicon (M series), `cmake` is also required to build and
install `qdldl` and `osqp` dependencies. Install it with `brew install cmake`.

Install all optional dependencies with:

```bash
pip install pycollimator[all]
```

### Tutorials

Read the [Getting Started Tutorial](https://py.collimator.ai/tutorials/01-getting-started/)
for a more complete example.

## Documentation

Head over to [https://py.collimator.ai](https://py.collimator.ai) for
the API reference documentation as well as examples and tutorials.

## MIT Licensed

This package is released and licensed under the [MIT](https://mit-license.org/)
license starting from version 2.2.0.
