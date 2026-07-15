# SPDX-License-Identifier: MIT
"""
T-025a — binary FMU export tests.

Two layers of testing are useful here, because the host platform
matters:

1. **Structural** (runs everywhere): ``build_fmu`` produces a
   well-formed ``.fmu`` zip with the right ``modelDescription.xml``
   and platform binaries inside.

2. **Round-trip** (any platform whose wrapper binary pythonfmu ships —
   win64 + linux64 always, darwin64 since pythonfmu 0.7.0): re-import
   the generated FMU via fmpy and confirm its ``do_step`` matches the
   in-process Python slave's logic.

The tests use a pythonfmu-style slave embedded as a temp file so they
can run without depending on any external FMU corpus, and they
exercise both a hand-rolled arithmetic slave and a
``JaxonomyDiagramSlave`` wrapping a one-block jaxonomy diagram.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from textwrap import dedent
from xml.etree import ElementTree as ET

import pytest


pythonfmu = pytest.importorskip("pythonfmu")
from jaxonomy.library.fmu_export import build_fmu, FmuBuildError  # noqa: E402


# When the master process is itself Python (this test run), the FMU
# wrapper's Py_Initialize is a no-op and the slave shares our
# interpreter — it sees the same jaxonomy these tests import. A
# non-Python master initializes a fresh embedded interpreter whose
# sys.path comes from the environment instead, so also prepend this
# checkout's root to PYTHONPATH: the round-trip tests then exercise
# this checkout either way (not whatever an editable install happens
# to point at).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)


# Round-trip tests need the host platform's wrapper binary inside
# pythonfmu's resources/binaries/ tree. pythonfmu's wheel ships
# win64 + linux64 always and darwin64 since 0.7.0; on older pythonfmu
# darwin needs a one-time source build (T-025b). We probe at import
# time so whatever wrapper is installed enables the round-trip path
# automatically.
def _runtime_host_ok() -> bool:
    import os
    import pythonfmu
    pf_dir = os.path.dirname(pythonfmu.__file__)
    if sys.platform == "win32":
        platform = "win64"
        ext = "dll"
    elif sys.platform == "darwin":
        platform = "darwin64"
        ext = "dylib"
    elif sys.platform.startswith("linux"):
        platform = "linux64"
        ext = "so"
    else:
        return False
    wrapper_dir = os.path.join(pf_dir, "resources", "binaries", platform)
    if not os.path.isdir(wrapper_dir):
        return False
    return any(f.endswith("." + ext) for f in os.listdir(wrapper_dir))


_RUNTIME_HOST_OK = _runtime_host_ok()
_RUNTIME_REASON = (
    f"no pythonfmu wrapper binary installed for sys.platform={sys.platform!r}; "
    f"upgrade to pythonfmu >= 0.7.0 (ships darwin64) or run T-025b's "
    f"source build to install one. Structural checks still apply."
)


# ── 1. Hand-rolled arithmetic slave: structural + (Linux) round-trip ──


_ADDER_SLAVE = dedent(
    """
    from pythonfmu import (
        Fmi2Slave, Fmi2Causality, Fmi2Variability, Real,
    )

    class Adder(Fmi2Slave):
        author = "jaxonomy-test"
        description = "Adder: y = a + b + offset"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.a = 0.0
            self.b = 0.0
            self.offset = 0.0
            self.y = 0.0
            self.register_variable(Real("a", causality=Fmi2Causality.input))
            self.register_variable(Real("b", causality=Fmi2Causality.input))
            # FMI 2 requires parameters to have variability=fixed/tunable.
            self.register_variable(Real(
                "offset",
                causality=Fmi2Causality.parameter,
                variability=Fmi2Variability.tunable,
            ))
            self.register_variable(Real("y", causality=Fmi2Causality.output))

        def do_step(self, current_time, step_size):
            self.y = self.a + self.b + self.offset
            return True
    """
).strip()


@pytest.fixture
def adder_fmu(tmp_path: Path):
    script = tmp_path / "adder.py"
    script.write_text(_ADDER_SLAVE)
    fmu = tmp_path / "Adder.fmu"
    build_fmu(script, fmu)
    return fmu


def test_build_fmu_produces_valid_zip(adder_fmu: Path):
    assert adder_fmu.is_file() and adder_fmu.stat().st_size > 1024
    with zipfile.ZipFile(adder_fmu) as z:
        names = z.namelist()
    assert "modelDescription.xml" in names
    # pythonfmu always bundles win64 and linux64 wrappers
    assert any(n.startswith("binaries/win64/") for n in names)
    assert any(n.startswith("binaries/linux64/") for n in names)
    # the user's slave script must be in resources/
    assert any(n.endswith("adder.py") for n in names)


def test_model_description_xml_round_trips_variable_names(adder_fmu: Path):
    with zipfile.ZipFile(adder_fmu) as z:
        xml = z.read("modelDescription.xml").decode("utf-8")
    root = ET.fromstring(xml)
    assert root.tag == "fmiModelDescription"
    assert root.attrib["fmiVersion"] == "2.0"
    names = {sv.attrib["name"]
             for sv in root.findall(".//ModelVariables/ScalarVariable")}
    assert {"a", "b", "offset", "y"}.issubset(names)
    cs = root.find("CoSimulation")
    assert cs is not None
    # Co-Simulation must declare a model identifier matching the slave
    # class name; that's what the C wrapper looks up at runtime.
    assert cs.attrib["modelIdentifier"] == "Adder"


def test_build_fmu_to_directory_destination(tmp_path: Path):
    """When fmu_path is a directory, pythonfmu picks the filename
    from the slave class name."""
    script = tmp_path / "adder.py"
    script.write_text(_ADDER_SLAVE)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = build_fmu(script, out_dir)
    assert result == str((out_dir / "Adder.fmu").resolve())
    assert (out_dir / "Adder.fmu").is_file()


def test_build_fmu_overwrites_existing_by_default(adder_fmu: Path):
    size_before = adder_fmu.stat().st_size
    # Build again to the same path; should replace, not error.
    build_fmu(adder_fmu.parent / "adder.py", adder_fmu)
    assert adder_fmu.stat().st_size == size_before


def test_build_fmu_refuses_overwrite_when_disabled(adder_fmu: Path):
    with pytest.raises(FmuBuildError, match="already exists"):
        build_fmu(adder_fmu.parent / "adder.py", adder_fmu, overwrite=False)


def test_build_fmu_missing_script_raises(tmp_path: Path):
    with pytest.raises(FmuBuildError, match="not found"):
        build_fmu(tmp_path / "nope.py", tmp_path / "out.fmu")


# ── 2. Round-trip (linux64 / win64): export -> fmpy import -> compare ─


@pytest.mark.skipif(not _RUNTIME_HOST_OK, reason=_RUNTIME_REASON)
def test_round_trip_adder_matches_in_process(adder_fmu: Path, tmp_path: Path):
    """Re-import the generated FMU via fmpy and step it side-by-side
    with the in-process slave class."""
    fmpy = pytest.importorskip("fmpy")

    # In-process reference: compute y = a + b + offset directly.
    a, b, offset = 1.5, -0.25, 0.5
    expected_y = a + b + offset

    md = fmpy.read_model_description(str(adder_fmu))
    unzipdir = fmpy.extract(str(adder_fmu))
    try:
        fmu = fmpy.fmi2.FMU2Slave(
            guid=md.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=md.coSimulation.modelIdentifier,
            instanceName="adder",
        )
        fmu.instantiate()
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        # Resolve refs by name so we don't hard-code numbers.
        refs = {v.name: v.valueReference for v in md.modelVariables}
        fmu.setReal([refs["offset"]], [offset])
        fmu.exitInitializationMode()
        fmu.setReal([refs["a"]], [a])
        fmu.setReal([refs["b"]], [b])
        fmu.doStep(currentCommunicationPoint=0.0, communicationStepSize=0.01)
        (got_y,) = fmu.getReal([refs["y"]])
        fmu.terminate()
        fmu.freeInstance()
    finally:
        import shutil
        shutil.rmtree(unzipdir, ignore_errors=True)

    assert got_y == pytest.approx(expected_y, abs=1e-12), (
        f"FMU step produced y={got_y}, expected {expected_y}"
    )


# ── 3. JaxonomyDiagramSlave: end-to-end through the diagram kernel ────

_DIAGRAM_SLAVE = dedent(
    """
    from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave
    import jaxonomy
    from jaxonomy.library import Constant

    def _build():
        bld = jaxonomy.DiagramBuilder()
        c = bld.add(Constant(7.5, name="seven_pt_five"))
        # Expose the Constant's output as a top-level diagram output;
        # JaxonomyDiagramSlave registers FMI variables from
        # ``diagram.output_ports``, so the port has to live there.
        bld.export_output(c.output_ports[0], name="seven_pt_five")
        return bld.build()

    class ConstSlave(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build)
        DT = 0.01
    """
).strip()


def test_diagram_slave_export_structural(tmp_path: Path):
    """A diagram with one Constant block exports cleanly: the FMU has
    a single output named after the constant and no inputs. This test
    only checks structure (round-trip runs are gated separately on the
    host wrapper binary)."""
    script = tmp_path / "diag.py"
    script.write_text(_DIAGRAM_SLAVE)
    fmu = build_fmu(script, tmp_path / "ConstSlave.fmu")
    with zipfile.ZipFile(fmu) as z:
        xml = z.read("modelDescription.xml").decode("utf-8")
    root = ET.fromstring(xml)
    outs = [
        sv.attrib["name"]
        for sv in root.findall(".//ModelVariables/ScalarVariable")
        if sv.attrib.get("causality") == "output"
    ]
    # The diagram has one output port named "seven_pt_five" (the
    # Constant block). JaxonomyDiagramSlave registers it as an FMI
    # Real output.
    assert any("seven_pt_five" in o for o in outs), (
        f"expected an output named after the Constant block; got {outs}"
    )


@pytest.mark.skipif(not _RUNTIME_HOST_OK, reason=_RUNTIME_REASON)
def test_diagram_slave_round_trip_const_value(tmp_path: Path):
    """On linux64, run the diagram slave inside the FMU wrapper and
    confirm the constant value comes back across the FMI boundary."""
    fmpy = pytest.importorskip("fmpy")
    script = tmp_path / "diag.py"
    script.write_text(_DIAGRAM_SLAVE)
    fmu_path = build_fmu(script, tmp_path / "ConstSlave.fmu")

    md = fmpy.read_model_description(fmu_path)
    unzipdir = fmpy.extract(fmu_path)
    try:
        fmu = fmpy.fmi2.FMU2Slave(
            guid=md.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=md.coSimulation.modelIdentifier,
            instanceName="cs",
        )
        fmu.instantiate()
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()
        fmu.doStep(currentCommunicationPoint=0.0, communicationStepSize=0.01)
        ref = next(v.valueReference for v in md.modelVariables
                   if "seven_pt_five" in v.name)
        (got,) = fmu.getReal([ref])
        fmu.terminate()
        fmu.freeInstance()
    finally:
        import shutil
        shutil.rmtree(unzipdir, ignore_errors=True)

    assert got == pytest.approx(7.5, abs=1e-9)


# ── 4. T-025c: auto-discovered Constant-block FMI inputs ──────────────


_T025C_SLAVE = dedent(
    """
    import jaxonomy
    from jaxonomy.library import Constant, Gain
    from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

    def _build():
        bld = jaxonomy.DiagramBuilder()
        sp = bld.add(Constant(0.0, name="setpoint"))
        g = bld.add(Gain(2.0, name="g"))
        bld.connect(sp.output_ports[0], g.input_ports[0])
        bld.export_output(g.output_ports[0], name="out")
        return bld.build()

    class T025c(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build)
        DT = 0.01
    """
).strip()


def test_t025c_constant_input_auto_discovery_in_process(tmp_path: Path):
    """Without going through the full FMU build, instantiate the
    slave class directly and verify auto-discovered inputs +
    apply_inputs route to Constant.value as designed."""
    script = tmp_path / "t025c.py"
    script.write_text(_T025C_SLAVE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("t025c_mod", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    slave = mod.T025c(instance_name="x", resources=str(tmp_path))

    # Auto-discovery exposed "setpoint" as an FMI input.
    var_names = {v.name for v in slave.vars.values()}
    assert "setpoint" in var_names, (
        f"expected 'setpoint' auto-discovered as an FMI input; got {var_names}"
    )
    assert slave._constant_inputs.get("setpoint") == (
        next(c.system_id
             for c in slave._diagram.leaf_systems
             if c.name == "setpoint"),
        "value",
    )

    # Drive the input and step three times with different setpoints.
    # Values flow through self._values, the dict the registered
    # getter/setter pair targets.
    cases = [(5.0, 10.0), (-3.0, -6.0), (1.5, 3.0)]
    t = 0.0
    for sp, expected_out in cases:
        slave._values["setpoint"] = sp
        ok = slave.do_step(t, 0.01)
        assert ok is True
        assert slave._values["out"] == pytest.approx(expected_out, abs=1e-9), (
            f"setpoint={sp}: expected out={expected_out}, "
            f"got {slave._values['out']}"
        )
        t += 0.01


def test_t025c_skips_constants_already_exported_as_outputs(tmp_path: Path):
    """A Constant block whose name collides with an exported output
    must NOT be re-registered as an input variable; the FMI standard
    forbids same-name distinct variables."""
    script = tmp_path / "collision.py"
    script.write_text(dedent("""
        import jaxonomy
        from jaxonomy.library import Constant
        from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

        def _build():
            bld = jaxonomy.DiagramBuilder()
            c = bld.add(Constant(7.5, name="duplicate"))
            bld.export_output(c.output_ports[0], name="duplicate")
            return bld.build()

        class Coll(JaxonomyDiagramSlave):
            DIAGRAM_FACTORY = staticmethod(_build)
    """).strip())
    import importlib.util
    spec = importlib.util.spec_from_file_location("coll_mod", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    slave = mod.Coll(instance_name="x", resources=str(tmp_path))
    # Name appears once (as output), not twice.
    names = [v.name for v in slave.vars.values()]
    assert names.count("duplicate") == 1, names
    # Auto-discovery skipped it — not in the constant_inputs map.
    assert "duplicate" not in slave._constant_inputs


@pytest.mark.skipif(not _RUNTIME_HOST_OK, reason=_RUNTIME_REASON)
def test_t025c_round_trip_through_fmu(tmp_path: Path):
    """On linux64, build the full FMU and verify the auto-discovered
    Constant input drives the diagram from across the FMI boundary."""
    fmpy = pytest.importorskip("fmpy")
    script = tmp_path / "t025c.py"
    script.write_text(_T025C_SLAVE)
    fmu_path = build_fmu(script, tmp_path / "T025c.fmu")
    md = fmpy.read_model_description(fmu_path)
    unzipdir = fmpy.extract(fmu_path)
    try:
        fmu = fmpy.fmi2.FMU2Slave(
            guid=md.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=md.coSimulation.modelIdentifier,
            instanceName="t025c",
        )
        refs = {v.name: v.valueReference for v in md.modelVariables}
        fmu.instantiate()
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()
        fmu.setReal([refs["setpoint"]], [4.0])
        fmu.doStep(currentCommunicationPoint=0.0, communicationStepSize=0.01)
        (out,) = fmu.getReal([refs["out"]])
        fmu.terminate()
        fmu.freeInstance()
    finally:
        import shutil
        shutil.rmtree(unzipdir, ignore_errors=True)
    assert out == pytest.approx(8.0, abs=1e-9)


# ── 5. Initialization priming, exported input ports, x0 parameters ────

_FULL_SURFACE_SLAVE = dedent(
    """
    import jaxonomy
    from jaxonomy.library import Gain, Integrator
    from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

    def _build():
        bld = jaxonomy.DiagramBuilder()
        g = bld.add(Gain(2.0, name="g"))
        integ = bld.add(Integrator(1.5, name="integ"))
        bld.connect(g.output_ports[0], integ.input_ports[0])
        bld.export_input(g.input_ports[0], name="u")
        bld.export_output(integ.output_ports[0], name="x")
        return bld.build()

    class FullSurface(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build)
        DT = 0.1
        EXPOSE_INITIAL_STATES = {"x0": "integ"}
    """
).strip()


@pytest.mark.skipif(not _RUNTIME_HOST_OK, reason=_RUNTIME_REASON)
def test_round_trip_init_priming_exported_input_and_x0(tmp_path: Path):
    """End-to-end conformance across the FMI boundary:

    - an exported diagram input port ("u") is honored (setReal on it
      actually drives the diagram);
    - the EXPOSE_INITIAL_STATES parameter ("x0") is applied at
      exitInitializationMode;
    - outputs are primed at initialization — getReal on "x" right
      after exitInitializationMode returns the true t=0 value, before
      any doStep (FMI 2.0 masters read initial outputs this way).
    """
    fmpy = pytest.importorskip("fmpy")
    script = tmp_path / "full_surface.py"
    script.write_text(_FULL_SURFACE_SLAVE)
    fmu_path = build_fmu(script, tmp_path / "FullSurface.fmu")

    md = fmpy.read_model_description(fmu_path)
    by_name = {v.name: v for v in md.modelVariables}
    assert by_name["u"].causality == "input"
    assert by_name["x0"].causality == "parameter"
    assert by_name["x"].causality == "output"

    unzipdir = fmpy.extract(fmu_path)
    try:
        fmu = fmpy.fmi2.FMU2Slave(
            guid=md.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=md.coSimulation.modelIdentifier,
            instanceName="full_surface",
        )
        refs = {v.name: v.valueReference for v in md.modelVariables}
        fmu.instantiate()
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        fmu.setReal([refs["x0"]], [2.0])
        fmu.setReal([refs["u"]], [1.0])
        fmu.exitInitializationMode()

        # Priming: the initial state must be readable before doStep.
        (x_init,) = fmu.getReal([refs["x"]])
        assert x_init == pytest.approx(2.0, abs=1e-9), (
            f"output not primed at initialization: got {x_init}"
        )

        # One step: dx/dt = 2 * u = 2, so x(0.1) = 2.0 + 0.2.
        fmu.doStep(currentCommunicationPoint=0.0, communicationStepSize=0.1)
        (x1,) = fmu.getReal([refs["x"]])
        fmu.terminate()
        fmu.freeInstance()
    finally:
        import shutil
        shutil.rmtree(unzipdir, ignore_errors=True)

    assert x1 == pytest.approx(2.2, abs=1e-6), (
        f"exported input port ignored or x0 not applied: x(0.1)={x1}"
    )
