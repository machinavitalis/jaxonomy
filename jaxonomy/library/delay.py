# SPDX-License-Identifier: MIT

from __future__ import annotations

from ..framework import LeafSystem, DependencyTicket, parameters
from ..backend import numpy_api as npa

class ShiftRegister(LeafSystem):
    """
    Fixed-length shift register delay line.
    
    Delays an input signal by exactly n_steps discrete 
    timesteps. Output at time t is the input value from 
    n_steps timesteps ago.
    
    Parameters:
        n_steps (int): Number of steps to delay. 
            STATIC — set at construction, cannot be changed 
            at runtime. Must be >= 1.
        signal_shape (tuple): Shape of each signal frame.
            Use () for scalar, (3,) for 3-vector, etc.
        initial_value (array-like): Value to fill the buffer 
            with before any input has been received.
            Default: zeros.
        dt (float): Discrete update interval in seconds.
    
    Ports:
        Input[0] "u": signal to delay, shape=signal_shape
        Output[0] "y": delayed signal, shape=signal_shape
    """
    
    @parameters(static=["n_steps", "signal_shape"])
    def __init__(
        self, 
        n_steps: int, 
        signal_shape: tuple = (), 
        initial_value=None, 
        dt: float = 0.01, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.n_steps = n_steps
        self.signal_shape = signal_shape
        
        if initial_value is None:
            self.initial_value = npa.zeros(signal_shape)
        else:
            self.initial_value = npa.array(initial_value)

        self.input_idx = self.declare_input_port()
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, n_steps, signal_shape):
        buffer = npa.broadcast_to(self.initial_value, (n_steps, *signal_shape))
        self.declare_discrete_state(default_value=buffer)
        
        self.configure_periodic_update(
            self._periodic_update_idx, 
            self._update, 
            period=self.dt, 
            offset=self.dt
        )
        
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=0.0,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
            default_value=buffer[-1],
        )

    def _update(self, _time, state, *inputs, **_params):
        u = inputs[self.input_idx]
        buffer = npa.roll(state.discrete_state, shift=1, axis=0)
        buffer = buffer.at[0].set(u)
        return buffer

    def _output(self, _time, state, **_params):
        return state.discrete_state[-1]


class MaskedDelayBuffer(LeafSystem):
    """
    Delay buffer where the delay length can be set at 
    runtime (up to max_steps).
    
    Like ShiftRegister, but allows the delay to be specified 
    as an input signal. Uses masking (not dynamic indexing) 
    for JAX compatibility.
    
    Parameters:
        max_steps (int): Maximum possible delay. STATIC.
        signal_shape (tuple): Shape of each signal frame.
        dt (float): Discrete update interval.
    
    Ports:
        Input[0] "u": signal to delay
        Input[1] "delay_steps": integer scalar, 
            0 < delay_steps <= max_steps
        Output[0] "y": delayed signal
    """
    
    @parameters(static=["max_steps", "signal_shape"])
    def __init__(
        self, 
        max_steps: int, 
        signal_shape: tuple = (), 
        dt: float = 0.01, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.max_steps = max_steps
        self.signal_shape = signal_shape

        self.input_u_idx = self.declare_input_port()
        self.input_delay_idx = self.declare_input_port()
        
        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port()

    def initialize(self, max_steps, signal_shape):
        initial_value = npa.zeros(signal_shape)
        buffer = npa.broadcast_to(initial_value, (max_steps, *signal_shape))
        self.declare_discrete_state(default_value=buffer)
        
        self.configure_periodic_update(
            self._periodic_update_idx, 
            self._update, 
            period=self.dt, 
            offset=self.dt
        )
        
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=0.0,
            requires_inputs=True,
            prerequisites_of_calc=[
                DependencyTicket.xd, 
                self.input_ports[self.input_delay_idx].ticket
            ],
        )

    def _update(self, _time, state, *inputs, **_params):
        u = inputs[self.input_u_idx]
        buffer = npa.roll(state.discrete_state, shift=1, axis=0)
        buffer = buffer.at[0].set(u)
        return buffer

    def _output(self, _time, state, *inputs, **_params):
        delay_steps = npa.clip(inputs[self.input_delay_idx], 1, self.max_steps)
        buffer = state.discrete_state
        
        mask = npa.arange(self.max_steps) == (delay_steps - 1)
        axes = tuple(range(1, 1 + len(self.signal_shape)))
        if axes:
            mask = npa.expand_dims(mask, axis=axes)
            
        return npa.sum(buffer * mask, axis=0)
