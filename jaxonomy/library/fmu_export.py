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
import platform
import sys
import uuid
import warnings
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
    "wrapper_diagnostics",
    "FmuBuildError",
]


class FmuBuildError(RuntimeError):
    """Raised when an FMU export fails (toolchain or input issue)."""


# ── PythonFMU wrapper diagnostics ────────────────────────────────────
#
# ``build_fmu`` compiles nothing itself: it bundles the pre-built FMI
# wrapper shipped inside the installed ``pythonfmu`` wheel. Two
# properties of that binary decide whether the resulting ``.fmu`` can
# be instantiated at all, and neither is visible to the FMI validators
# — ``fmpy.validate_fmu`` and VDMCheck read ``modelDescription.xml``
# and never dlopen the binary, so a wrapper that cannot load still
# reports zero findings.
#
# 1. **Architecture.** FMI 2.0's platform folders (``linux64`` /
#    ``darwin64`` / ``win64``) name a word size, not an instruction
#    set, and pythonfmu's wheels carry x86-64 builds. On an aarch64
#    host the bundled wrapper is simply the wrong ISA, and the FMU
#    fails to load with a bare "cannot open shared object file".
# 2. **Python linkage.** Upstream links the wrapper against CMake's
#    ``Python3::Module``, which deliberately leaves ``libpython``
#    unresolved — extension-module semantics, where the hosting
#    interpreter supplies the symbols. A Python master (FMPy,
#    jaxonomy) works; a C/C++ master (OpenModelica, fmusim, most
#    commercial tools) cannot resolve ``Py_Initialize``.
#
# ``scripts/build_pythonfmu_wrapper.sh`` builds a wrapper for the host
# ISA linked against ``Python3::Python``, and fixes two further defects
# that only surface outside a Python master: numpy's C extensions
# failing to resolve when the master dlopened the wrapper
# ``RTLD_LOCAL``, and a ``Py_Finalize`` on teardown that segfaults once
# numpy/jax are loaded, taking ``fmi2Terminate`` with it.

_ELF_MACHINES = {0x03: "x86", 0x28: "arm", 0x3E: "x86-64", 0xB7: "aarch64"}
_PE_MACHINES = {0x014C: "x86", 0x8664: "x86-64", 0xAA64: "aarch64"}
_MACHO_CPUS = {0x01000007: "x86-64", 0x0100000C: "aarch64"}

_WRAPPER_REMEDIATION = (
    "Build a wrapper for this host with "
    "scripts/build_pythonfmu_wrapper.sh (it patches PythonFMU to link "
    "Python3::Python and installs the result into the active pythonfmu)."
)


def _normalize_machine(name: str) -> str:
    """Map a ``platform.machine()`` spelling onto our canonical names."""
    lowered = name.lower()
    if lowered in ("x86_64", "amd64", "x64"):
        return "x86-64"
    if lowered in ("aarch64", "arm64"):
        return "aarch64"
    if lowered in ("i386", "i686", "x86"):
        return "x86"
    return lowered


def _binary_machine(path: str) -> str | None:
    """Instruction set of an ELF / Mach-O / PE binary, or None.

    Deliberately header-only: no pyelftools dependency, and an
    unrecognized format returns None rather than raising, so a
    diagnostic can never break a build that would otherwise succeed.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        return None
    if len(head) < 20:
        return None
    if head[:4] == b"\x7fELF":
        little = head[5] == 1
        machine = int.from_bytes(head[18:20], "little" if little else "big")
        return _ELF_MACHINES.get(machine)
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):  # Mach-O LE
        return _MACHO_CPUS.get(int.from_bytes(head[4:8], "little"))
    if head[:2] == b"MZ":
        try:
            with open(path, "rb") as handle:
                handle.seek(0x3C)
                pe_offset = int.from_bytes(handle.read(4), "little")
                handle.seek(pe_offset)
                if handle.read(4) != b"PE\0\0":
                    return None
                return _PE_MACHINES.get(int.from_bytes(handle.read(2), "little"))
        except OSError:
            return None
    return None


def _links_libpython(path: str) -> bool:
    """Whether a wrapper binary carries a libpython dependency.

    The dependency name lives in the binary's string table for every
    format we care about, so scanning for the soname answers the
    question without a per-format parser. The pattern requires a digit
    right after ``libpython`` — the wrapper's own filename is
    ``libpythonfmu-export``, which would otherwise match itself.

    False means the wrapper expects its Python symbols from the hosting
    process, i.e. it only works under a Python-based FMI master.
    """
    import re

    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError:
        return False
    # libpython3.12.so.1.0 (ELF), libpython3.11.dylib (Mach-O),
    # python312.dll / python3.dll (PE).
    return bool(
        re.search(rb"libpython\d", blob)
        or re.search(rb"python\d+\.dll", blob, re.IGNORECASE)
    )


def _host_wrapper_slot() -> tuple[str, str]:
    """``(platform_folder, extension)`` pythonfmu uses for this host."""
    if sys.platform == "win32":
        return "win64", "dll"
    if sys.platform == "darwin":
        return "darwin64", "dylib"
    if sys.platform.startswith("linux"):
        return "linux64", "so"
    raise FmuBuildError(f"unsupported host platform for FMU export: {sys.platform}")


def wrapper_diagnostics() -> dict:
    """Report on the PythonFMU wrapper ``build_fmu`` would bundle here.

    Returns a dict with ``platform`` (the FMI platform folder),
    ``path``, ``present``, ``machine`` (instruction set of the bundled
    binary, or ``None`` if unreadable), ``host_machine``,
    ``arch_matches_host``, and ``embeds_python`` (whether the wrapper
    links libpython, which a non-Python FMI master requires).

    Useful before shipping an FMU to someone else: the FMI validators
    check ``modelDescription.xml`` only, so neither an ISA mismatch nor
    a missing libpython shows up as a finding.

    Example::

        >>> from jaxonomy.library.fmu_export import wrapper_diagnostics
        >>> info = wrapper_diagnostics()
        >>> info["arch_matches_host"], info["embeds_python"]
        (True, True)
    """
    slot, ext = _host_wrapper_slot()
    host_machine = _normalize_machine(platform.machine())
    info = {
        "platform": slot,
        "path": None,
        "present": False,
        "machine": None,
        "host_machine": host_machine,
        "arch_matches_host": False,
        "embeds_python": False,
    }
    try:
        import pythonfmu
    except ImportError:
        return info

    path = os.path.join(
        os.path.dirname(pythonfmu.__file__),
        "resources",
        "binaries",
        slot,
        f"libpythonfmu-export.{ext}" if slot != "win64" else "pythonfmu-export.dll",
    )
    info["path"] = path
    if not os.path.isfile(path):
        return info
    info["present"] = True
    info["machine"] = _binary_machine(path)
    info["arch_matches_host"] = info["machine"] == host_machine
    info["embeds_python"] = _links_libpython(path)
    return info


_wrapper_warned: set[str] = set()


def _warn_on_wrapper_limits() -> None:
    """Surface the silent failure modes of a bundled wrapper.

    Warns once per process per distinct problem: the limitation is a
    property of the installation, so repeating it for every exported
    FMU is noise (same policy as the recording-buffer warning).
    """
    try:
        info = wrapper_diagnostics()
    except FmuBuildError:
        return
    if not info["present"] and "missing" not in _wrapper_warned:
        _wrapper_warned.add("missing")
        warnings.warn(
            f"pythonfmu ships no {info['platform']} wrapper, so this FMU "
            f"cannot be instantiated on this host. {_WRAPPER_REMEDIATION}",
            UserWarning,
            stacklevel=3,
        )
        return
    if (
        info["machine"]
        and not info["arch_matches_host"]
        and "arch" not in _wrapper_warned
    ):
        _wrapper_warned.add("arch")
        warnings.warn(
            f"the bundled {info['platform']} wrapper is {info['machine']} but "
            f"this host is {info['host_machine']}, so the generated FMU cannot "
            f"be loaded here (the FMI validators will still pass — they read "
            f"modelDescription.xml, not the binary). {_WRAPPER_REMEDIATION}",
            UserWarning,
            stacklevel=3,
        )
    if not info["embeds_python"] and "libpython" not in _wrapper_warned:
        _wrapper_warned.add("libpython")
        warnings.warn(
            "the bundled wrapper does not link libpython, so the generated FMU "
            "only loads under a Python-based FMI master (FMPy, jaxonomy). A "
            "C/C++ master such as OpenModelica or fmusim cannot resolve "
            f"Py_Initialize. {_WRAPPER_REMEDIATION}",
            UserWarning,
            stacklevel=3,
        )


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
    `pythonfmu` library. Nothing is compiled here: the wrappers that
    ship inside the installed pythonfmu wheel are bundled as-is, which
    puts two limits on the result. Neither shows up as a validator
    finding — ``fmpy.validate_fmu`` and VDMCheck read
    ``modelDescription.xml`` and never load the binary.

    **Architecture.** The FMI 2.0 platform folders (``linux64`` /
    ``darwin64`` / ``win64``) name a word size, not an instruction
    set, and pythonfmu's wheels carry x86-64 builds. A wheel installed
    on aarch64 therefore yields an FMU that cannot be loaded on the
    machine that built it. Observed with pythonfmu 0.7.0 on Linux:
    ``win64`` and ``linux64`` are present, both x86-64, and no
    ``darwin64`` at all.

    **Python linkage.** Upstream links the wrapper against CMake's
    ``Python3::Module``, which deliberately leaves ``libpython``
    unresolved — extension-module semantics, where the hosting
    interpreter supplies the symbols. Under a Python master (FMPy,
    jaxonomy) that works. A C/C++ master (OpenModelica, fmusim, most
    commercial tools) fails at ``dlopen`` with ``undefined symbol:
    _Py_NoneStruct``.

    ``scripts/build_pythonfmu_wrapper.sh`` builds a wrapper for the
    host that fixes both, plus two further defects that only surface
    outside a Python master: numpy's C extensions failing to resolve
    when the master used ``RTLD_LOCAL``, and a ``Py_Finalize`` on
    teardown that segfaults once numpy/jax are loaded. Call
    :func:`wrapper_diagnostics` to see which wrapper is installed;
    ``build_fmu`` warns when it would produce an unloadable FMU.

    Whatever wrapper is installed, an FMU produced here is
    *tool-coupled*: the importing side needs a Python environment with
    jaxonomy on its path, since the slave is executed as Python.

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
    _normalize_model_description(produced)
    _warn_on_wrapper_limits()
    return os.path.abspath(produced)


def _normalize_fmi_datetime(value: str) -> str:
    """Normalize a timestamp to the FMI-required ``YYYY-MM-DDThh:mm:ssZ``.

    pythonfmu stamps ``generationDateAndTime`` as ISO-8601 with a numeric
    offset (e.g. ``2026-07-11T02:22:07+00:00``), which the INTO-CPS
    VDMCheck validator rejects — the FMI 2.0 spec mandates the ``Z``
    suffix form with whole seconds. Unparseable values are returned
    unchanged (the validator will then say so, loudly).
    """
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_model_description(fmu_path: str) -> None:
    """T-026c — make pythonfmu's modelDescription.xml pass the official
    validators (``fmpy.validate_fmu`` + INTO-CPS VDMCheck2).

    Two conformance gaps in pythonfmu's generator are patched in place:

    1. FMI 2.0 requires every output whose ``initial`` is ``calculated``
       / ``approx`` (the default for outputs) to be listed under
       ``ModelStructure/InitialUnknowns``; pythonfmu omits the element
       entirely (flagged by fmpy). The ``Outputs`` ``Unknown`` entries
       are mirrored there (skipping ``initial="exact"`` variables), in
       the ascending index order the schema requires.
    2. ``generationDateAndTime`` must be ``YYYY-MM-DDThh:mm:ssZ``;
       pythonfmu stamps a ``+00:00``-offset form (flagged by VDMCheck).

    No-op when the XML already conforms.
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
        changed = False

        gdt = root.attrib.get("generationDateAndTime")
        if gdt:
            normalized = _normalize_fmi_datetime(gdt)
            if normalized != gdt:
                root.set("generationDateAndTime", normalized)
                changed = True

        structure = root.find("ModelStructure")
        if structure is not None and structure.find("InitialUnknowns") is None:
            outputs = structure.find("Outputs")
            if outputs is not None:
                variables = root.findall(".//ModelVariables/ScalarVariable")
                unknown_indices = []
                for unk in outputs.findall("Unknown"):
                    idx = int(unk.attrib["index"])
                    sv = variables[idx - 1]  # FMI variable indices are 1-based
                    if sv.attrib.get("initial") == "exact":
                        continue
                    unknown_indices.append(idx)
                if unknown_indices:
                    initial_unknowns = ET.SubElement(
                        structure, "InitialUnknowns"
                    )
                    for idx in sorted(unknown_indices):
                        ET.SubElement(
                            initial_unknowns, "Unknown", {"index": str(idx)}
                        )
                    changed = True

        if not changed:
            return
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
