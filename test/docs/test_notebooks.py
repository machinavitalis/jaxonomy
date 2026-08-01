# SPDX-License-Identifier: MIT

"""Executable-documentation gate: every shipped notebook must still run.

The sibling of ``test_readme_snippets.py``, one level up. The README's fenced
blocks are executed there for the same reason the notebooks are executed here —
a sample that no longer runs is worse than no sample, and a notebook hides the
rot particularly well because its committed outputs keep rendering correctly on
the docs site long after the code beneath them stopped working.

What counts as passing is *execution without an uncaught exception*. Outputs are
deliberately not compared against the committed ones: notebook results are JAX
floats, and pinning them would produce a gate that fails on every harmless
last-digit change while catching nothing a plain execution does not. Notebooks
that need to assert something specific do it with an ``assert`` cell in the
notebook itself, which this gate then enforces for free.

The executed notebook is never written back — this only ever reads from
``docs/``.

Notebooks run with their own directory as the working directory, which means
the repository root is *not* on the kernel's path and ``import jaxonomy``
resolves through the installed distribution instead. In a git worktree that
silently points at whichever checkout the editable install was made from —
so the gate would validate some other copy of the library while appearing to
test this branch. ``repo_under_test`` below prevents that, and
``test_notebook_kernel_imports_repo_under_test`` proves it holds in a real
kernel rather than merely in this process.

Tiers come from ``notebook_manifest``: the smoke tier is unmarked and runs on
every pull request, everything else carries the ``notebook`` marker that
``pytest.ini`` deselects by default. To run the full set locally::

    pytest -m notebook test/docs/
"""

from __future__ import annotations

import os
import shutil

import pytest

from .notebook_manifest import MANIFEST, REPO_ROOT, SMOKE, Notebook

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")  # provides the python3 kernel spec


def _params() -> list:
    """One pytest param per manifest entry, carrying its tier + timeout marks."""
    params = []
    for nb in MANIFEST:
        marks = [pytest.mark.timeout(nb.timeout)]
        if nb.tier != SMOKE:
            marks.append(pytest.mark.notebook)
        params.append(pytest.param(nb, marks=marks, id=nb.path))
    return params


def _execute(nb: Notebook) -> None:
    from nbclient import NotebookClient

    document = nbformat.read(nb.full_path, as_version=4)
    client = NotebookClient(
        document,
        timeout=nb.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb.run_dir)}},
        allow_errors=False,
    )
    client.execute()


@pytest.fixture
def repo_under_test(monkeypatch):
    """Make the kernel import *this* checkout's jaxonomy, not the installed one.

    On CI this is a no-op: there is one checkout and ``pip install -e .`` points
    at it. It matters locally, where a worktree shares the editable install of
    the checkout it was created from — so without this, a notebook run from
    ``docs/examples`` imports the other tree's library and the gate reports on
    code that is not the code under test.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)] + ([existing] if existing else [])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts))


@pytest.fixture
def headless_matplotlib(monkeypatch):
    """Force a non-interactive backend in the notebook's kernel subprocess.

    Nearly every notebook plots, so on a headless runner a figure must not try
    to open a window. This uses the inline backend rather than plain ``Agg``
    deliberately: it is what Jupyter and nbconvert actually render these
    notebooks with, and under ``Agg`` every ``plt.show()`` emits a
    ``FigureCanvasAgg is non-interactive`` warning whose text embeds the
    ipykernel temp path — which ``scripts/check_portable_paths.py`` then
    rejects the moment anyone re-executes a notebook and commits the outputs.
    """
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")


def test_notebook_kernel_imports_repo_under_test(repo_under_test):
    """A real kernel, started where notebooks start, must import this checkout.

    Asserted in a kernel rather than in-process because the discrepancy only
    appears in the subprocess: this process runs with the repository root as its
    working directory and resolves jaxonomy correctly regardless.
    """
    from nbclient import NotebookClient

    probe = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("import jaxonomy; print(jaxonomy.__file__)")]
    )
    run_dir = REPO_ROOT / "docs" / "examples"
    NotebookClient(
        probe,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(run_dir)}},
        allow_errors=False,
    ).execute()

    printed = "".join(
        output.get("text", "")
        for output in probe.cells[0].outputs
        if output.output_type == "stream"
    ).strip()

    assert printed, "probe kernel produced no output"
    assert printed.startswith(str(REPO_ROOT)), (
        "notebooks would execute against a different checkout of jaxonomy:\n"
        f"  kernel imported: {printed}\n"
        f"  repo under test: {REPO_ROOT}"
    )


@pytest.mark.parametrize("nb", _params())
def test_notebook_executes(nb: Notebook, repo_under_test, headless_matplotlib):
    for module in nb.requires:
        pytest.importorskip(module)

    for program in nb.binaries:
        if shutil.which(program) is None:
            pytest.skip(f"{program} not on PATH")

    for artifact in nb.artifacts:
        if not (nb.run_dir / artifact).exists():
            pytest.skip(f"missing input artifact {artifact} (not tracked in the repo)")

    _execute(nb)


def test_manifest_covers_every_discovered_notebook():
    # Mirrors the check in test_notebook_manifest.py, kept here so that running
    # this module alone cannot pass while notebooks go unexecuted.
    from .notebook_manifest import discovered_notebooks, manifest_paths

    assert set(discovered_notebooks()) == set(manifest_paths())
