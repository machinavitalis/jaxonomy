# SPDX-License-Identifier: MIT
"""
T-025 — modelDescription.xml generator tests.

Verifies the XML half of FMU export.  Compiled-binary packaging is
filed as T-025a.

We use a simple Gain → Gain diagram with one exported input (``u``)
and one exported output (``y``) and check:

  - The XML parses as XML.
  - The root element is fmiModelDescription with fmiVersion=2.0.
  - There's a CoSimulation child.
  - ModelVariables contains one ScalarVariable for the input
    (causality=input) and one for the output (causality=output).
  - ModelStructure/Outputs lists the output(s) by index.
  - write_model_description writes the file at the given path.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Gain, model_description_xml, write_model_description
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _gain_diagram():
    bld = jaxonomy.DiagramBuilder()
    g = bld.add(Gain(2.0, name="g"))
    bld.export_input(g.input_ports[0], name="u")
    bld.export_output(g.output_ports[0], name="y")
    return bld.build()


# ── XML structure ──────────────────────────────────────────────────────


def test_xml_parses_and_has_correct_root():
    d = _gain_diagram()
    xml = model_description_xml(d, model_name="GainModel")
    root = ET.fromstring(xml)
    assert root.tag == "fmiModelDescription"
    assert root.attrib["fmiVersion"] == "2.0"
    assert root.attrib["modelName"] == "GainModel"
    assert "guid" in root.attrib
    assert root.attrib["guid"].startswith("{")


def test_cosimulation_section_present():
    d = _gain_diagram()
    xml = model_description_xml(d, model_name="GainModel")
    root = ET.fromstring(xml)
    cs = root.find("CoSimulation")
    assert cs is not None
    assert "modelIdentifier" in cs.attrib


def test_model_variables_includes_input_and_output():
    d = _gain_diagram()
    xml = model_description_xml(d, model_name="GainModel")
    root = ET.fromstring(xml)
    mvars = root.find("ModelVariables")
    assert mvars is not None
    svs = mvars.findall("ScalarVariable")
    by_causality = {sv.attrib["causality"] for sv in svs}
    assert "input" in by_causality
    assert "output" in by_causality
    # One u, one y.
    names = {sv.attrib["name"] for sv in svs}
    assert "u" in names
    assert "y" in names


def test_model_structure_outputs_indexed():
    d = _gain_diagram()
    xml = model_description_xml(d, model_name="GainModel")
    root = ET.fromstring(xml)
    structure = root.find("ModelStructure")
    assert structure is not None
    outputs = structure.find("Outputs")
    assert outputs is not None
    unknowns = outputs.findall("Unknown")
    assert len(unknowns) == 1
    # The "y" port is index 2 (after the "u" input at index 1).
    assert unknowns[0].attrib["index"] == "2"


def test_user_guid_is_preserved():
    d = _gain_diagram()
    xml = model_description_xml(
        d, model_name="GainModel", guid="{abc-123}",
    )
    root = ET.fromstring(xml)
    assert root.attrib["guid"] == "{abc-123}"


def test_value_references_unique():
    d = _gain_diagram()
    xml = model_description_xml(d, model_name="GainModel")
    root = ET.fromstring(xml)
    refs = [
        sv.attrib["valueReference"]
        for sv in root.findall("ModelVariables/ScalarVariable")
    ]
    assert len(refs) == len(set(refs)), f"duplicate valueReferences: {refs}"


# ── write_model_description on disk ───────────────────────────────────


def test_write_model_description_to_file(tmp_path):
    d = _gain_diagram()
    path = tmp_path / "out.xml"
    write_model_description(d, str(path), model_name="GainModel")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert content.startswith("<?xml")
    assert "fmiModelDescription" in content


def test_write_creates_parent_directories(tmp_path):
    d = _gain_diagram()
    path = tmp_path / "deep" / "nested" / "out.xml"
    write_model_description(d, str(path), model_name="GainModel")
    assert path.exists()


# ── vector input handling ────────────────────────────────────────────


def test_vector_port_helper_expands():
    """Direct test of the _flatten_port_name_shape helper: a port whose
    default_value is shape (3,) expands into 3 named scalar variables.
    The exported-port path may strip default_value depending on diagram
    flatten behaviour; the helper guarantees the expansion logic itself."""
    from jaxonomy.library.fmu_export import _flatten_port_name_shape

    class _Stub:
        index = 0
        name = "u"

        def __init__(self, default):
            self.default_value = default

    p = _Stub(jnp.array([0.0, 0.0, 0.0]))
    expanded = _flatten_port_name_shape(p)
    assert [e[0] for e in expanded] == ["u[0]", "u[1]", "u[2]"]


def test_scalar_port_helper_does_not_expand():
    from jaxonomy.library.fmu_export import _flatten_port_name_shape

    class _Stub:
        index = 0
        name = "y"
        default_value = jnp.array(0.0)

    expanded = _flatten_port_name_shape(_Stub())
    assert expanded == [("y", ())]
