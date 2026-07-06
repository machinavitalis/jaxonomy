from jaxonomy.framework.diagram_builder import DiagramBuilder
from jaxonomy import library
from jaxonomy.simulation.types import SimulatorOptions
from jaxonomy.framework.validation import validate_diagram


def test_custom_python_block_autodiff_warning():
    builder = DiagramBuilder()
    
    # Add a CustomPythonBlock
    custom_block = builder.add(
        library.CustomPythonBlock(
            name="custom_python_block",
            time_mode="agnostic",
            init_script="x = 1",
            user_statements="y = x * 2",
            inputs=[],
            outputs=["y"]
        )
    )
    diagram = builder.build()
    
    result = validate_diagram(
        diagram,
        SimulatorOptions(enable_autodiff=True, max_major_steps=10)
    )
    assert result.valid  # should be a warning, not an error
    assert any("CustomPythonBlock" in w for w in result.warnings)


def test_valid_diagram_no_issues():
    builder = DiagramBuilder()
    gain = builder.add(library.Gain(gain=1.0, name="g"))
    diagram = builder.build()
    
    result = validate_diagram(diagram)
    assert result.valid
    assert len(result.errors) == 0


def test_max_major_steps_warning():
    builder = DiagramBuilder()
    gain = builder.add(library.Gain(gain=1.0, name="g"))
    diagram = builder.build()
    
    result = validate_diagram(
        diagram,
        SimulatorOptions(
            enable_autodiff=True, 
            max_major_steps=None
        )
    )
    # Warning due to max_major_steps missing
    assert any("max_major_steps" in w for w in result.warnings)


def test_custom_python_block_finalize_script_no_warning():
    """finalize_script is now supported — validation should not warn about it."""
    builder = DiagramBuilder()

    builder.add(
        library.CustomPythonBlock(
            name="custom_python_block",
            time_mode="agnostic",
            init_script="x = 1",
            user_statements="y = x * 2",
            finalize_script="x = 0",
            inputs=[],
            outputs=["y"],
        )
    )
    diagram = builder.build()

    result = validate_diagram(diagram, SimulatorOptions(enable_autodiff=False))
    assert result.valid
    assert not any("finalize_script" in w for w in result.warnings)

