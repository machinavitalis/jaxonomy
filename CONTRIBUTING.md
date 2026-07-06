# Contributing to Jaxonomy

Thanks for your interest in contributing. Jaxonomy is a JAX-native simulation
engine for hybrid dynamical systems, released under the [MIT License](LICENSE.md).
By contributing, you agree that your contributions are licensed under the same
terms.

## Development setup

Jaxonomy requires **Python 3.10+**. Install in editable mode with the test
extras:

```bash
git clone https://github.com/machinavitalis/jaxonomy.git
cd jaxonomy
python -m pip install --upgrade pip
pip install -e ".[test]"
```

`jax`/`jaxlib` are pinned to a minor version (breaking changes occur across JAX
minor releases) — let the editable install resolve them rather than installing
JAX separately.

## Running the tests

The default suite (fast, what PR CI runs):

```bash
pytest -q -n auto
```

`pytest.ini` deselects the `slow`, `dashboard`, and `autodiff_full` markers by
default. To run more:

```bash
pip install -e ".[test-full]"        # adds optional ML/extra backends
pytest -q -m "slow or autodiff_full" # the heavier sweeps
```

Some suites are **opt-in via environment variables** (they need external
corpora the repo doesn't bundle):

- `JAXONOMY_FMU_CORPUS` — path to a locally built Modelica Reference-FMUs corpus
  (FMI import conformance tests).
- `JAXONOMY_HF_MODEL_PATH` — a local ONNX/HF model (ONNX→JAX tests).
- `JAXONOMY_PALLASCAT_PROJDIR`, `JAXONOMY_BENCH_DEVICE` — app/benchmark suites.

## What we value in a change

Jaxonomy is correctness-first. Contributions that touch the engine are expected
to come with evidence:

- **Gradient correctness** — new solvers/blocks/event paths should pass the
  property-based finite-difference checks.
- **Determinism** — reference simulations are byte-reproducible on a given
  device; don't introduce nondeterminism into that path.
- **Tests + docstrings** — a fix ships with a regression test; a feature ships
  with a runnable example or test and a docstring explaining the public surface.
- **User-visible changes** get a `CHANGELOG.md` entry under `[Unreleased]`.

Match the style and conventions of the surrounding code. The `AGENTS/` directory
holds the deeper architecture, conventions, and decision records
(`CONTEXT.md`, `PATTERNS.md`, `DECISIONS.md`) — skim the relevant ones before a
non-trivial change.

## Pull request process

1. Branch from `main`.
2. Keep the change focused; one logical change per PR.
3. Make sure `pytest -q -n auto` passes locally and the CI checks are green.
4. Describe the change and its motivation in the PR; link any related issue.

## Reporting bugs & security issues

Open a GitHub issue for bugs and feature requests. For anything with security
impact, **do not open a public issue** — follow [SECURITY.md](SECURITY.md).
