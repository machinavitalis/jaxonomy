import pytest
import numpy as np
import jax.numpy as jnp
import jax

import jaxonomy
from jaxonomy import library


class TestDelayBlocks:
    
    def test_shift_register_scalar(self):
        """Output matches input from n_steps ago."""
        dt = 0.1
        n_steps = 3
        builder = jaxonomy.DiagramBuilder()
        
        step = builder.add(library.Step(start_value=0.0, end_value=1.0, step_time=0.05))
        delay = builder.add(library.ShiftRegister(n_steps=n_steps, signal_shape=(), dt=dt))
        
        builder.connect(step.output_ports[0], delay.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        
        res = jaxonomy.simulate(
            diagram, context, (0.0, 0.5), 
            recorded_signals={"delay": delay.output_ports[0]}
        )
        
        # Step happens at 0.1. delay is 3 steps, meaning output steps at 0.1 + 0.3 = 0.4
        ts = np.array(res.time)
        ys = np.array(res.outputs["delay"])
        
        expected = np.where(ts >= 0.4 - 1e-4, 1.0, 0.0)
        assert np.allclose(ys, expected)

    def test_shift_register_matches_unit_delay(self):
        """ShiftRegister(n_steps=1) == UnitDelay behavior."""
        dt = 0.1
        builder = jaxonomy.DiagramBuilder()
        
        step = builder.add(library.Step(start_value=0.0, end_value=1.0, step_time=0.2))
        shift_reg = builder.add(library.ShiftRegister(n_steps=1, signal_shape=(), dt=dt))
        unit_delay = builder.add(library.UnitDelay(dt=dt, initial_state=0.0))
        
        builder.connect(step.output_ports[0], shift_reg.input_ports[0])
        builder.connect(step.output_ports[0], unit_delay.input_ports[0])
        
        diagram = builder.build()
        context = diagram.create_context()
        
        res = jaxonomy.simulate(
            diagram, context, (0.0, 0.6), 
            recorded_signals={
                "shift": shift_reg.output_ports[0],
                "unit": unit_delay.output_ports[0]
            }
        )
        
        assert np.allclose(res.outputs["shift"], res.outputs["unit"])

    def test_shift_register_vector(self):
        """Delay a 3D vector signal."""
        dt = 0.01
        n_steps = 2
        builder = jaxonomy.DiagramBuilder()
        
        const = builder.add(library.Constant(jnp.array([1.0, 2.0, 3.0])))
        delay = builder.add(library.ShiftRegister(n_steps=n_steps, signal_shape=(3,), dt=dt))
        
        builder.connect(const.output_ports[0], delay.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        
        res = jaxonomy.simulate(
            diagram, context, (0.0, 0.05), 
            recorded_signals={"delay": delay.output_ports[0]}
        )
        
        ys = res.outputs["delay"]
        # Output should be [0,0,0] up to 0.02 (first update at 0.01 pushing it through), 
        # then [1,2,3] starting at 0.03
        assert np.allclose(ys[0], [0., 0., 0.])
        assert np.allclose(ys[1], [0., 0., 0.]) # t=0.01, output uses pre-update state [0,0]
        assert np.allclose(ys[2], [0., 0., 0.]) # t=0.02, output uses pre-update state [vec, 0]
        assert np.allclose(ys[3], [1., 2., 3.]) # t=0.03, output uses pre-update state [vec, vec]

    def test_shift_register_gradient(self):
        """Gradient flows through the buffer."""
        dt = 0.1
        n_steps = 2
        delay = library.ShiftRegister(n_steps=n_steps, signal_shape=(), dt=dt)
        delay.system_id = "test_sys"
        
        def simulate_delay(u_val):
            import collections
            State = collections.namedtuple("State", ["discrete_state"])
            state = State(discrete_state=jnp.zeros((n_steps,)))
            for _ in range(n_steps + 1): # update enough times to push it to output
                buffer = delay._update(0.0, state, u_val)
                state = State(discrete_state=buffer)
                
            return delay._output(0.0, state)
            
        grad_fn = jax.grad(simulate_delay)
        
        assert jnp.isclose(grad_fn(5.0), 1.0) # Derivative of x w.r.t x is 1

    def test_masked_delay_buffer_matches_shift_register(self):
        """MaskedDelayBuffer with fixed delay == ShiftRegister."""
        dt = 0.1
        n_steps = 3
        builder = jaxonomy.DiagramBuilder()
        
        step = builder.add(library.Step(start_value=0.0, end_value=2.0, step_time=0.1))
        delay_steps_const = builder.add(library.Constant(3))
        
        shift_reg = builder.add(library.ShiftRegister(n_steps=n_steps, signal_shape=(), dt=dt))
        masked_delay = builder.add(library.MaskedDelayBuffer(max_steps=5, signal_shape=(), dt=dt))
        
        builder.connect(step.output_ports[0], shift_reg.input_ports[0])
        builder.connect(step.output_ports[0], masked_delay.input_ports[0])
        builder.connect(delay_steps_const.output_ports[0], masked_delay.input_ports[1])
        
        diagram = builder.build()
        context = diagram.create_context()
        
        res = jaxonomy.simulate(
            diagram, context, (0.0, 0.5), 
            recorded_signals={
                "shift": shift_reg.output_ports[0],
                "masked": masked_delay.output_ports[0]
            }
        )
        
        assert np.allclose(res.outputs["shift"], res.outputs["masked"])

    def test_masked_delay_buffer_variable(self):
        """delay_steps input actually changes delay."""
        dt = 0.1
        builder = jaxonomy.DiagramBuilder()
        
        step1 = builder.add(library.Step(start_value=0.0, end_value=1.0, step_time=0.05))
        delay_signal = builder.add(library.Step(start_value=2, end_value=4, step_time=0.35))
        
        masked_delay = builder.add(library.MaskedDelayBuffer(max_steps=5, signal_shape=(), dt=dt))
        
        builder.connect(step1.output_ports[0], masked_delay.input_ports[0])
        builder.connect(delay_signal.output_ports[0], masked_delay.input_ports[1])
        
        diagram = builder.build()
        context = diagram.create_context()
        
        res = jaxonomy.simulate(
            diagram, context, (0.0, 0.8), 
            recorded_signals={"masked": masked_delay.output_ports[0]}
        )
        
        ts = np.array(res.time)
        ys = np.array(res.outputs["masked"])
        
        # t < 0.4: delay is 2
        # input is 1 at t=0.1, so output (delay=2) goes up at t=0.3
        # t >= 0.4: delay becomes 4 immediately!
        # at t=0.4, pre-update buffer has inputs from t=0.3. Oldest=4 is t=0.0 (value 0).
        # So output will DROP back to 0 at t=0.4!
        # It goes up again at t=0.5 (since step was at 0.1, +4 steps)
        
        y_sol = np.zeros_like(ts)
        y_sol[(ts >= 0.3 - 1e-4) & (ts < 0.4 - 1e-4)] = 1.0  # High at 0.3 due to delay=2
        y_sol[ts >= 0.5 - 1e-4] = 1.0  # High at 0.5 due to delay=4
        
        assert np.allclose(ys, y_sol)

