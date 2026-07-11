# SPDX-License-Identifier: MIT

"""T-026c — validate ``build_fmu`` output with the *official* checkers.

Before this task the only export validation was an fmpy re-import
round-trip, which tolerates spec-non-conformant ``modelDescription.xml``.
Two independent validators close that hole:

1. **fmpy** ``validate_fmu()`` — the de-facto reference validator. Hard
   gate: every FMU produced by ``build_fmu`` must report **zero**
   problems. (This immediately caught a real defect: pythonfmu omits
   ``ModelStructure/InitialUnknowns``; ``build_fmu`` now post-processes
   the XML — see ``_ensure_initial_unknowns``.)
2. **VDMCheck2** (INTO-CPS FMI-VDM-Model) — the strict static VDM-SL
   checker of ``modelDescription.xml``. Optional: requires a JVM plus
   the distribution zip; set the ``VDMCHECK2`` env var to the extracted
   ``VDMCheck2.sh`` (or the ``vdmcheck2-*.jar``) to enable. CI's
   fmu-validators job wires this; locally it skips cleanly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

pythonfmu = pytest.importorskip("pythonfmu")
fmpy = pytest.importorskip("fmpy")

from fmpy.validation import validate_fmu  # noqa: E402

from jaxonomy.library.fmu_export import build_fmu  # noqa: E402

# Reuse the canonical slave fixtures so the validated FMUs are exactly
# the ones the rest of the export suite exercises.
from test.library.test_fmu_export_binary import (  # noqa: E402
    _ADDER_SLAVE,
    _DIAGRAM_SLAVE,
)


def _build(tmp_path: Path, name: str, source: str) -> Path:
    script = tmp_path / f"{name.lower()}.py"
    script.write_text(source)
    return Path(build_fmu(script, tmp_path / f"{name}.fmu"))


@pytest.fixture(params=["Adder", "ConstSlave"])
def exported_fmu(request, tmp_path: Path) -> Path:
    source = {"Adder": _ADDER_SLAVE, "ConstSlave": _DIAGRAM_SLAVE}[request.param]
    return _build(tmp_path, request.param, source)


# ── 1. fmpy validate_fmu: the hard gate ──────────────────────────────


def test_fmpy_validate_reports_zero_problems(exported_fmu: Path):
    problems = validate_fmu(str(exported_fmu))
    assert problems == [], (
        f"fmpy.validate_fmu found problems in {exported_fmu.name}: {problems}"
    )


def test_initial_unknowns_mirror_calculated_outputs(exported_fmu: Path):
    """The T-026c conformance fix: ModelStructure/InitialUnknowns lists
    every calculated output (pythonfmu omits the element entirely)."""
    with zipfile.ZipFile(exported_fmu) as z:
        root = ET.fromstring(z.read("modelDescription.xml"))
    structure = root.find("ModelStructure")
    outputs = {
        u.attrib["index"] for u in structure.find("Outputs").findall("Unknown")
    }
    initial = structure.find("InitialUnknowns")
    assert initial is not None, "InitialUnknowns missing from ModelStructure"
    initial_idx = [int(u.attrib["index"]) for u in initial.findall("Unknown")]
    assert set(map(str, initial_idx)) == outputs
    assert initial_idx == sorted(initial_idx), (
        "FMI 2.0 requires InitialUnknowns in ascending index order"
    )


# ── 2. VDMCheck2: optional strict static checker ─────────────────────

_VDMCHECK2 = os.environ.get("VDMCHECK2", "")
_HAVE_JAVA = shutil.which("java") is not None


def _vdmcheck_available() -> bool:
    return bool(_VDMCHECK2) and os.path.exists(_VDMCHECK2) and _HAVE_JAVA


@pytest.mark.skipif(
    not _vdmcheck_available(),
    reason=(
        "VDMCheck2 not configured: set VDMCHECK2=/path/to/VDMCheck2.sh "
        "(or the vdmcheck2-*.jar) from the INTO-CPS FMI-VDM-Model "
        "distribution and ensure a JVM is on PATH."
    ),
)
def test_vdmcheck2_passes(exported_fmu: Path):
    if _VDMCHECK2.endswith(".jar"):
        cmd = ["java", "-jar", _VDMCHECK2, str(exported_fmu)]
    else:
        cmd = ["bash", _VDMCHECK2, str(exported_fmu)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
        cwd=os.path.dirname(_VDMCHECK2) or None,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "No errors found" in output, (
        f"VDMCheck2 failed on {exported_fmu.name} "
        f"(rc={proc.returncode}):\n{output[-3000:]}"
    )
