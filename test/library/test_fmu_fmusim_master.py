# SPDX-License-Identifier: MIT
"""Export gate against ``fmusim``, an independent non-Python FMI master.

``fmpy.validate_fmu`` and VDMCheck both read ``modelDescription.xml`` and
never load the platform binary, so an FMU that cannot be instantiated at
all still passes them with zero findings. That gap is not hypothetical:
the wrapper PythonFMU ships links no libpython, and every C/C++ master
fails on it at ``dlopen`` with ``undefined symbol: _Py_NoneStruct`` while
the validators stay green.

``fmusim`` is the Modelica Association's reference simulator (Rust, no
Python in the process), so running an exported FMU through it is the
check the static validators cannot provide. Install it with:

    curl -sSL https://raw.githubusercontent.com/modelica/fmusim/main/install.sh | sh

and build a loadable wrapper with ``scripts/build_pythonfmu_wrapper.sh``.
The module skips cleanly when either is missing.

.. note:: ``fmusim simulate`` exits 0 even when the binary fails to load,
    so these assertions read its output rather than its return code.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest


pytestmark = pytest.mark.slow

pytest.importorskip("pythonfmu")
from jaxonomy.library.fmu_export import build_fmu, wrapper_diagnostics  # noqa: E402


FMUSIM = shutil.which("fmusim")
_WRAPPER = wrapper_diagnostics()

requires_fmusim = pytest.mark.skipif(
    FMUSIM is None, reason="fmusim not on PATH; see this module's docstring"
)
requires_loadable_wrapper = pytest.mark.skipif(
    not (_WRAPPER["present"] and _WRAPPER["arch_matches_host"]
         and _WRAPPER["embeds_python"]),
    reason=(
        "the installed pythonfmu wrapper cannot be loaded by a non-Python "
        "master (wrong ISA or no libpython); run "
        "scripts/build_pythonfmu_wrapper.sh"
    ),
)


_SPRING_DAMPER_SLAVE = dedent(
    """
    import numpy as np
    import jaxonomy
    from jaxonomy.library import LTISystem
    from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

    m, c, k = 1.0, 0.5, 1.0

    def _build():
        bld = jaxonomy.DiagramBuilder()
        plant = bld.add(LTISystem(
            A=np.array([[0.0, 1.0], [-k / m, -c / m]]),
            B=np.array([[0.0], [1.0 / m]]),
            C=np.array([[1.0, 0.0]]),
            D=np.array([[0.0]]),
            name="plant",
        ))
        bld.export_input(plant.input_ports[0], name="F")
        bld.export_output(plant.output_ports[0], name="x")
        return bld.build()

    class SpringDamperSlave(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build)
        DT = 0.01
    """
).strip()


def _run_fmusim(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [FMUSIM, *args], capture_output=True, text=True, timeout=600
    )


@pytest.fixture(scope="module")
def exported_fmu(tmp_path_factory) -> Path:
    workdir = tmp_path_factory.mktemp("fmusim")
    script = workdir / "spring_damper.py"
    script.write_text(_SPRING_DAMPER_SLAVE)
    return Path(build_fmu(script, workdir / "SpringDamper.fmu"))


@requires_fmusim
def test_fmusim_validates_the_exported_fmu(exported_fmu: Path):
    """A third validator, independent of fmpy and VDMCheck."""
    result = _run_fmusim("validate", str(exported_fmu))
    combined = result.stdout + result.stderr
    assert "error" not in combined.lower(), combined


@requires_fmusim
@requires_loadable_wrapper
def test_fmusim_simulates_the_exported_fmu(exported_fmu: Path, tmp_path: Path):
    """The load-bearing check: a master with no Python interpreter of its
    own instantiates the FMU and steps it.

    With the wrapper pythonfmu ships this fails at dlopen while every
    static validator still reports the FMU as clean.
    """
    output = tmp_path / "out.csv"
    result = _run_fmusim(
        "simulate",
        "--stop-time", "1.0",
        "--output-interval", "0.25",
        "--output-file", str(output),
        str(exported_fmu),
    )
    combined = result.stdout + result.stderr
    assert "dlopen failed" not in combined, combined
    assert "error" not in combined.lower(), combined
    assert output.is_file(), f"fmusim wrote no output: {combined}"

    with open(output) as handle:
        rows = list(csv.DictReader(handle))
    assert "x" in rows[0], f"expected the exported output port; got {rows[0]}"
    # start plus one row per output interval over [0, 1]
    assert len(rows) == 5, f"expected 5 samples, got {len(rows)}"
    assert all(float(row["x"]) == pytest.approx(0.0, abs=1e-9) for row in rows), (
        "undriven spring-damper must stay at rest"
    )


@requires_fmusim
@requires_loadable_wrapper
def test_fmusim_tracks_an_input_signal(exported_fmu: Path, tmp_path: Path):
    """Driven through the FMI boundary by an external master.

    m*x'' + c*x' + k*x = sin(t) with these parameters is a lightly damped
    resonant system driven at its own natural frequency, so the amplitude
    grows to roughly Q = 1/(2*zeta) = 2. A boundary that silently drops
    inputs leaves x at zero instead.
    """
    import numpy as np

    time = np.arange(0.0, 20.0001, 0.01)
    input_file = tmp_path / "F.csv"
    np.savetxt(
        input_file, np.column_stack([time, np.sin(time)]),
        delimiter=",", header="time,F", comments="", fmt="%.10g",
    )
    output = tmp_path / "driven.csv"
    result = _run_fmusim(
        "simulate",
        "--stop-time", "20",
        "--output-interval", "0.01",
        "--input-file", str(input_file),
        "--output-file", str(output),
        str(exported_fmu),
    )
    combined = result.stdout + result.stderr
    assert "error" not in combined.lower(), combined

    x = np.array([float(row["x"]) for row in csv.DictReader(open(output))])
    assert np.abs(x).max() == pytest.approx(2.0, rel=0.15), (
        f"resonant amplitude off: peak |x| = {np.abs(x).max():.3f}"
    )
