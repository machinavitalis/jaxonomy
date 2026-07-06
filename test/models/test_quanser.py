# SPDX-License-Identifier: MIT

import numpy as np

import jaxonomy
from jaxonomy.library.quanser import QubeServoModel

from jaxonomy.testing.markers import requires_jax


def test_qube_simulation(show_plots=False):
    system = QubeServoModel(x0=[0.0, 0.5, 0.0, 0.0])
    system.input_ports[0].fix_value(0.0)
    context = system.create_context()

    recorded_signals = {"y": system.output_ports[0]}
    results = jaxonomy.simulate(
        system,
        context,
        (0.0, 5.0),
        recorded_signals=recorded_signals,
    )

    if show_plots:
        import matplotlib.pyplot as plt

        t = results.time
        y = results.outputs["y"]

        plt.figure(figsize=(7, 2))
        plt.plot(t, y)
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.title("Qube Simulation")
        plt.show()


@requires_jax(xfail=True)  # why no numpy?
def test_qube_linearization():
    xd = np.array([0.0, 0.0, 0.0, 0.0])
    xu = np.array([0.0, np.pi, 0.0, 0.0])
    u0 = np.array([0.0])

    # Check that the "down" fixed point is stable
    system = QubeServoModel()
    system.input_ports[0].fix_value(u0)
    context = system.create_context()

    context = context.with_continuous_state(xd)
    xdot = system.eval_time_derivatives(context)
    assert np.allclose(xdot, 0.0)

    lin_sys = jaxonomy.library.linearize(system, context).to_lti()
    evals = np.linalg.eigvals(lin_sys.A)

    # One zero eigenvalue for the rotor degree of freedom
    assert sum(e == 0 for e in evals) == 1

    # Three stable modes
    assert sum(e.real < 0 for e in evals) == 3

    # Check that the "up" fixed point is unstable
    context = context.with_continuous_state(xu)
    xdot = system.eval_time_derivatives(context)
    assert np.allclose(xdot, 0.0)

    lin_sys = jaxonomy.library.linearize(system, context).to_lti()
    evals = np.linalg.eigvals(lin_sys.A)

    # Expect one positive eigenvalue for the unstable "falling" mode
    assert sum(e.real > 0 for e in evals) == 1

    # One zero eigenvalue for the rotor degree of freedom
    assert sum(e == 0 for e in evals) == 1

    # Two stable modes
    assert sum(e.real < 0 for e in evals) == 2


import sys
from unittest.mock import MagicMock

def test_quanser_hal():
    # Setup the mock for pal.products.qube to bypass missing hardware constraints
    mock_pal = MagicMock()
    mock_qube_module = MagicMock()
    mock_qube2_class = MagicMock()
    mock_qube_instance = MagicMock()
    
    # Setup standard attributes expected by the runtime
    mock_qube_instance.card = "mocked_hardware_card"
    mock_qube_instance.motorPosition = 1.2
    mock_qube_instance.pendulumPosition = 3.4
    mock_qube2_class.return_value = mock_qube_instance
    mock_qube_module.QubeServo2 = mock_qube2_class
    
    mock_pal.products.qube = mock_qube_module
    
    # Inject mocks into sys.modules
    sys.modules["pal"] = mock_pal
    sys.modules["pal.products"] = mock_pal.products
    sys.modules["pal.products.qube"] = mock_qube_module
    
    from jaxonomy.library.quanser import QuanserHAL
    
    # Initialize HAL safely
    hal = QuanserHAL(dt=0.01)
    
    # Check that it instantiated `QubeServo2` correctly
    mock_qube2_class.assert_called_once_with(hardware=False, pendulum=1, frequency=100.0)
    mock_qube_instance.write_led.assert_called_with(color=[0, 1, 0])
    
    # Test step (writes voltage)
    hal._impure_step(3.5)
    mock_qube_instance.write_voltage.assert_called_once_with(3.5)
    
    # Test output (reads sensors)
    outputs = hal._impure_output()
    assert np.allclose(outputs, [1.2, 3.4])
    mock_qube_instance.read_outputs.assert_called_once()
    
    # Test finalization and cleanup triggers
    hal.terminate()
    mock_qube_instance.write_led.assert_called_with(color=[1, 1, 0])
    mock_qube_instance.terminate.assert_called_once()


if __name__ == "__main__":
    test_qube_simulation(show_plots=True)
