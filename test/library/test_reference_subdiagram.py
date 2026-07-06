# SPDX-License-Identifier: MIT

import pytest

from jaxonomy.framework import Diagram, Parameter
from jaxonomy.library.reference_subdiagram import ReferenceSubdiagram


def dummy_diagram_constructor(*args, instance_name, parameters, **kwargs):
    diagram = Diagram(name=instance_name)
    # Just store parameters for verification
    diagram.stored_parameters = parameters
    return diagram

@pytest.fixture(autouse=True)
def clear_registry():
    # Clear registry before each test to ensure isolation
    ReferenceSubdiagram._registry.clear()
    ReferenceSubdiagram._default_parameters.clear()
    yield

def test_register_reference_subdiagram():
    param_def = [Parameter(name="gain", value=1.0)]
    ref_id = ReferenceSubdiagram.register(
        constructor=dummy_diagram_constructor,
        parameter_definitions=param_def,
        ref_id="my_dummy_subdiagram"
    )
    
    assert ref_id == "my_dummy_subdiagram"
    assert "my_dummy_subdiagram" in ReferenceSubdiagram._registry
    assert ReferenceSubdiagram._registry["my_dummy_subdiagram"] == dummy_diagram_constructor
    
    # Check parameters
    retrieved_params = ReferenceSubdiagram.get_parameter_definitions("my_dummy_subdiagram")
    assert len(retrieved_params) == 1
    assert retrieved_params[0].name == "gain"
    assert retrieved_params[0].value == 1.0

def test_create_diagram_with_default_parameters():
    param_def = [Parameter(name="gain", value=1.0)]
    ref_id = ReferenceSubdiagram.register(
        constructor=dummy_diagram_constructor,
        parameter_definitions=param_def,
        ref_id="my_dummy_subdiagram"
    )
    
    diagram = ReferenceSubdiagram.create_diagram(
        ref_id="my_dummy_subdiagram",
        instance_name="test_instance"
    )
    
    assert diagram.name == "test_instance"
    assert diagram.ref_id == "my_dummy_subdiagram"
    # Should contain the default parameter
    assert "gain" in diagram.stored_parameters
    assert diagram.stored_parameters["gain"].value == 1.0
    
    # Check dynamic parameters declared on diagram
    assert "gain" in diagram.parameters

def test_create_diagram_with_instance_parameters_override():
    param_def = [
        Parameter(name="gain", value=1.0),
        Parameter(name="offset", value=0.0)
    ]
    ref_id = ReferenceSubdiagram.register(
        constructor=dummy_diagram_constructor,
        parameter_definitions=param_def,
        ref_id="my_dummy_subdiagram"
    )
    
    diagram = ReferenceSubdiagram.create_diagram(
        ref_id="my_dummy_subdiagram",
        instance_name="test_instance",
        instance_parameters={"gain": 5.0} # Override gain, leave offset
    )
    
    assert diagram.name == "test_instance"
    assert diagram.ref_id == "my_dummy_subdiagram"
    
    assert diagram.stored_parameters["gain"].value == 5.0
    assert diagram.stored_parameters["offset"].value == 0.0
    
    assert "gain" in diagram.instance_parameters
    assert "offset" not in diagram.instance_parameters

def test_create_diagram_invalid_ref_id():
    with pytest.raises(ValueError, match="not found"):
        ReferenceSubdiagram.create_diagram(
            ref_id="invalid_id",
            instance_name="test"
        )

def test_create_diagram_invalid_parameter():
    param_def = [Parameter(name="gain", value=1.0)]
    ref_id = ReferenceSubdiagram.register(
        constructor=dummy_diagram_constructor,
        parameter_definitions=param_def,
        ref_id="my_dummy_subdiagram"
    )
    
    with pytest.raises(ValueError, match="not found in parameter definitions"):
        ReferenceSubdiagram.create_diagram(
            ref_id="my_dummy_subdiagram",
            instance_name="test",
            instance_parameters={"unknown_param": 10.0}
        )
