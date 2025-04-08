# Installation

## Prerequisites

Python 3.10 or later is required.

## Installation steps

It is highly recommended to use a virtual environment to install `pycollimator`.

```bash
pip install pycollimator
```

- On Windows: `set JAX_ENABLE_X64=True`.
- On macOS you may need to install `cmake` first.

## Optional dependencies

More advanced features require additional dependencies that can be installed with:

```bash
pip install pycollimator[safe]
```

This will not include support for NMPC blocks.

### Nonlinear MPC

Nonlinear MPC blocks require `IPOPT` to be preinstalled.

- On Ubuntu: `sudo apt install coinor-libipopt-dev`.
- On macOS: `brew install ipopt` and `brew install cmake`.

Install all optional dependencies with `pip install pycollimator[all]` or
just the NMPC dependencies with `pip install pycollimator[nmpc]`.

<details>
<summary>Licensed under MIT</summary>
This `pycollimator` package is released and licensed under the
<a href="https://mit-license.org/">MIT</a> license.
</details>
