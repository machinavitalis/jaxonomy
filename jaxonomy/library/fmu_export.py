# SPDX-License-Identifier: MIT
"""
FMU export — modelDescription.xml generator (T-025).

Full FMU export packages an FMI 2.0 ``modelDescription.xml`` together
with a compiled C shared library and a manifest into a ``.fmu`` zip.
This module ships the **metadata half** — a generator that produces a
spec-compliant ``modelDescription.xml`` from a Jaxonomy diagram.  The
compiled-binary half (a co-simulation C wrapper that calls back into
jaxonomy's simulation kernel) is filed as T-025a; it requires a C
toolchain and the FMI 2.0 reference implementation, neither of which
this module assumes.

Usage::

    from jaxonomy.library.fmu_export import write_model_description

    diagram = build_my_diagram()
    write_model_description(
        diagram,
        path="my_model/modelDescription.xml",
        model_name="MyModel",
        guid="auto",  # or a fixed UUID string
    )

The generated XML includes:
  - ``fmiModelDescription`` root with FMI version 2.0
  - ``CoSimulation`` element with ``modelIdentifier``
  - ``ModelVariables`` containing one ``ScalarVariable`` per exported
    diagram input (causality=input) and exported output (causality=output)
  - ``ModelStructure`` with ``Outputs`` index list

Limitations:
  - Only scalar inputs/outputs (vector ports → one ScalarVariable per
    element with name ``portname[i]``).  Array variables are an FMI 3
    feature anyway.
  - Real-valued only; integer / boolean / string variables aren't
    auto-detected from port dtypes yet.  Add explicit type
    annotations on ports if you need them.
  - No discrete-state surfacing; the FMU presents a pure
    input → output map.  Internal state is hidden behind doStep.
"""

from __future__ import annotations

import os
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..framework.diagram import Diagram


__all__ = [
    "write_model_description",
    "model_description_xml",
    "build_fmu",
    "FmuBuildError",
]


class FmuBuildError(RuntimeError):
    """Raised when an FMU export fails (toolchain or input issue)."""


def _gen_guid() -> str:
    return "{" + str(uuid.uuid4()) + "}"


def _flatten_port_name_shape(port):
    """Return ``[(scalar_name, ()), ...]`` for vector ports, or
    ``[(name, ())]`` for scalar ports."""
    name = port.name or f"port_{port.index}"
    default = getattr(port, "default_value", None)
    if default is None:
        return [(name, ())]
    arr = np.asarray(default)
    if arr.ndim == 0:
        return [(name, ())]
    if arr.ndim == 1:
        return [(f"{name}[{i}]", ()) for i in range(arr.shape[0])]
    # Higher dims: flatten with multi-dim index.
    flat = []
    for i in range(arr.size):
        idx = np.unravel_index(i, arr.shape)
        flat.append((f"{name}[{','.join(str(k) for k in idx)}]", ()))
    return flat


def model_description_xml(
    diagram: "Diagram",
    *,
    model_name: str,
    guid: str | None = None,
    description: str = "Exported by jaxonomy.library.fmu_export",
    generation_tool: str = "jaxonomy",
) -> str:
    """Build the FMI 2.0 modelDescription XML as a string.

    Args:
        diagram: A :class:`~jaxonomy.framework.diagram.Diagram` whose
            input and output ports define the FMU's I/O surface.
        model_name: Human-readable model name.  Also used as the
            modelIdentifier (with non-identifier characters stripped).
        guid: Optional FMU GUID; auto-generated if None.
        description: Free-form description string.
        generation_tool: Stored in the FMU metadata.

    Returns:
        UTF-8 XML string ending with a trailing newline.
    """
    if guid is None:
        guid = _gen_guid()

    model_identifier = "".join(
        c if c.isalnum() or c == "_" else "_" for c in model_name
    )
    if not model_identifier or not model_identifier[0].isalpha():
        model_identifier = "M" + model_identifier

    root = ET.Element("fmiModelDescription", attrib={
        "fmiVersion": "2.0",
        "modelName": model_name,
        "guid": guid,
        "description": description,
        "generationTool": generation_tool,
        "generationDateAndTime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variableNamingConvention": "structured",
    })

    # CoSimulation element.
    ET.SubElement(root, "CoSimulation", attrib={
        "modelIdentifier": model_identifier,
        "canHandleVariableCommunicationStepSize": "true",
        "canInterpolateInputs": "false",
        "maxOutputDerivativeOrder": "0",
    })

    variables = ET.SubElement(root, "ModelVariables")

    # Number variables sequentially starting at 1 (FMI convention).
    next_value_ref = 1
    output_indices: list[int] = []

    # Inputs first.
    for port in diagram.input_ports:
        for varname, _shape in _flatten_port_name_shape(port):
            sv = ET.SubElement(variables, "ScalarVariable", attrib={
                "name": varname,
                "valueReference": str(next_value_ref),
                "causality": "input",
                "variability": "continuous",
                "initial": "exact",
            })
            ET.SubElement(sv, "Real", attrib={"start": "0.0"})
            next_value_ref += 1

    # Outputs.
    for port in diagram.output_ports:
        for varname, _shape in _flatten_port_name_shape(port):
            output_indices.append(next_value_ref)
            sv = ET.SubElement(variables, "ScalarVariable", attrib={
                "name": varname,
                "valueReference": str(next_value_ref),
                "causality": "output",
                "variability": "continuous",
                "initial": "calculated",
            })
            ET.SubElement(sv, "Real", attrib={})
            next_value_ref += 1

    # ModelStructure / Outputs.
    structure = ET.SubElement(root, "ModelStructure")
    if output_indices:
        outputs_el = ET.SubElement(structure, "Outputs")
        for i, _ref in enumerate(output_indices, start=len(diagram.input_ports) + 1):
            ET.SubElement(outputs_el, "Unknown", attrib={"index": str(i)})

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8") + "\n"


def build_fmu(
    slave_script: str | os.PathLike,
    fmu_path: str | os.PathLike,
    *,
    project_files: "list[str | os.PathLike] | None" = None,
    documentation: str | os.PathLike | None = None,
    options: dict | None = None,
    overwrite: bool = True,
) -> str:
    """T-025a — package a Python ``Fmi2Slave`` subclass into a binary
    FMI 2.0 co-simulation .fmu file.

    This delegates the binary half of the FMU (the C wrapper that
    embeds Python and dispatches FMI calls into the slave) to the
    `pythonfmu` library. pythonfmu's wheel ships pre-built wrappers
    for ``binaries/win64/`` and ``binaries/linux64/``; **darwin
    requires a one-time source build** (T-025b) of
    ``libpythonfmu-export.dylib`` into pythonfmu's
    ``resources/binaries/darwin64/`` folder::

        git clone https://github.com/NTNU-IHB/PythonFMU.git
        cd PythonFMU && cmake -B build -DCMAKE_BUILD_TYPE=Release \\
            && cmake --build build --config Release
        # Patch needed before configure on macOS: in
        # src/pythonfmu/PySlaveInstance.cpp, change the
        # ``#elif defined(__linux__)`` block guarding the
        # destructor attribute to also accept ``__APPLE__``.
        # Then copy the resulting dylib into the installed package:
        cp build-output/libpythonfmu-export.dylib \\
           "$(python -c 'import os, pythonfmu;
                          print(os.path.dirname(pythonfmu.__file__))')\\
           /resources/binaries/darwin64/"

    Once the wrapper is in place, ``build_fmu()`` automatically
    bundles all three platforms into the FMU and round-trip via
    ``fmpy`` works end-to-end on the same host. Without it, the
    generated ``.fmu`` is still structurally valid (XML + win64 +
    linux64 binaries) but cannot be re-imported on darwin.

    .. warning:: FMUs produced this way inherit pythonfmu's
        **one-instance-per-process** limitation: the embedded-Python
        wrapper holds a process-wide ``Py_Initialize`` singleton, so
        the same ``.fmu`` cannot be instantiated twice in one Python
        process (multi-start / batched co-simulation must
        subprocess-isolate each instance). See the matching warning on
        :class:`jaxonomy.library.ModelicaFMU` for the workaround.

    Args:
        slave_script: Path to a Python file that defines exactly one
            :class:`pythonfmu.Fmi2Slave` subclass. Variable
            registration happens in ``__init__``; ``do_step`` performs
            one cosim step.
        fmu_path: Output path. May be a directory (a ``<ClassName>.fmu``
            is created inside) or a full ``.fmu`` filename.
        project_files: Extra source files / directories to bundle into
            the FMU's ``resources/``. Useful for shipping helper
            modules the slave imports.
        documentation: Optional folder bundled into ``documentation/``.
        options: Forwarded to :meth:`pythonfmu.FmuBuilder.build_FMU`
            (``needsExecutionTool``, ``canHandleVariableCommunicationStepSize``,
            …). ``None`` keeps pythonfmu's defaults.
        overwrite: If False and ``fmu_path`` already exists, raises.

    Returns:
        Absolute path to the generated ``.fmu``.

    Raises:
        FmuBuildError: pythonfmu missing, slave script invalid, or the
            build step itself failed.
    """
    try:
        from pythonfmu import FmuBuilder
    except ImportError as exc:
        raise FmuBuildError(
            "build_fmu requires the 'pythonfmu' package "
            "(pip install pythonfmu)"
        ) from exc

    slave_script = os.fspath(slave_script)
    fmu_path = os.fspath(fmu_path)
    if not os.path.isfile(slave_script):
        raise FmuBuildError(f"slave script not found: {slave_script}")

    # FmuBuilder.build_FMU writes to a *directory* — it picks the
    # filename from the slave class name. To honour an explicit
    # ``foo.fmu`` target, build into the parent dir then move.
    target_is_file = fmu_path.endswith(".fmu")
    if target_is_file:
        dest_dir = os.path.dirname(os.path.abspath(fmu_path)) or "."
    else:
        dest_dir = fmu_path
    os.makedirs(dest_dir, exist_ok=True)

    if target_is_file and os.path.exists(fmu_path) and not overwrite:
        raise FmuBuildError(f"{fmu_path} already exists (overwrite=False)")

    project_files = list(project_files) if project_files else []
    project_files = [os.fspath(p) for p in project_files]

    try:
        produced = FmuBuilder.build_FMU(
            slave_script,
            dest=dest_dir,
            project_files=set(project_files),
            documentation_folder=(os.fspath(documentation)
                                  if documentation else None),
            **(options or {}),
        )
    except Exception as exc:
        raise FmuBuildError(f"FMU build failed: {exc}") from exc

    produced = str(produced)
    if target_is_file and produced != fmu_path:
        if os.path.abspath(produced) != os.path.abspath(fmu_path):
            if os.path.exists(fmu_path) and overwrite:
                os.remove(fmu_path)
            os.replace(produced, fmu_path)
            produced = fmu_path
    _ensure_initial_unknowns(produced)
    return os.path.abspath(produced)


def _ensure_initial_unknowns(fmu_path: str) -> None:
    """T-026c — make pythonfmu's modelDescription.xml pass the official
    validators.

    FMI 2.0 requires every output whose ``initial`` is ``calculated`` /
    ``approx`` (the default for outputs) to be listed under
    ``ModelStructure/InitialUnknowns``. pythonfmu emits the ``Outputs``
    unknowns but omits ``InitialUnknowns`` entirely, which
    ``fmpy.validation.validate_fmu`` flags on every generated FMU.
    Post-process the archive: mirror the ``Outputs`` ``Unknown`` entries
    (skipping any whose ScalarVariable declares ``initial="exact"``)
    into an ``InitialUnknowns`` element, in ascending index order as the
    schema requires. No-op when ``InitialUnknowns`` already exists or
    there are no outputs.
    """
    import io
    import zipfile
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(fmu_path, "r") as zf:
        names = zf.namelist()
        if "modelDescription.xml" not in names:
            return
        xml_bytes = zf.read("modelDescription.xml")
        root = ET.fromstring(xml_bytes)
        structure = root.find("ModelStructure")
        if structure is None or structure.find("InitialUnknowns") is not None:
            return
        outputs = structure.find("Outputs")
        if outputs is None:
            return
        variables = root.findall(".//ModelVariables/ScalarVariable")
        unknown_indices = []
        for unk in outputs.findall("Unknown"):
            idx = int(unk.attrib["index"])
            sv = variables[idx - 1]  # FMI variable indices are 1-based
            if sv.attrib.get("initial") == "exact":
                continue
            unknown_indices.append(idx)
        if not unknown_indices:
            return
        initial_unknowns = ET.SubElement(structure, "InitialUnknowns")
        for idx in sorted(unknown_indices):
            ET.SubElement(initial_unknowns, "Unknown", {"index": str(idx)})
        new_xml = ET.tostring(root, encoding="UTF-8", xml_declaration=True)

        # Rewrite the archive with the patched XML (zipfile cannot
        # replace a member in place).
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for item in zf.infolist():
                data = (
                    new_xml
                    if item.filename == "modelDescription.xml"
                    else zf.read(item.filename)
                )
                out.writestr(item, data)

    with open(fmu_path, "wb") as f:
        f.write(buf.getvalue())


def write_model_description(
    diagram: "Diagram",
    path: str,
    *,
    model_name: str | None = None,
    guid: str | None = None,
    description: str | None = None,
) -> str:
    """Write a Jaxonomy diagram's FMI 2.0 modelDescription.xml to disk.

    Args:
        diagram: Diagram to export.
        path: Output file path.
        model_name: Defaults to ``diagram.name``.
        guid: Optional GUID.
        description: Optional free-form description.

    Returns:
        The same ``path`` argument (for chaining convenience).
    """
    if model_name is None:
        model_name = getattr(diagram, "name", "JaxonomyModel")
    xml = model_description_xml(
        diagram,
        model_name=model_name,
        guid=guid,
        description=description or f"FMI 2.0 export of {model_name}",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path
