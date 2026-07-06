# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import jax.numpy as jnp
from jaxonomy import SimulatorOptions, simulate
from benchmarks.brusselator import Brusselator2D

class TestBrusselator2D(unittest.TestCase):
    def test_brusselator_sim(self):
        # Initialize a small N=4 grid Brusselator (2 * 16 = 32 states)
        N = 4
        plant = Brusselator2D(N=N, A=3.4, B=1.0, alpha=10.0)
        
        ctx = plant.create_context()
        self.assertEqual(ctx.continuous_state.shape, (2 * N * N,))
        
        # Test simulating using stiff solver (bdf)
        opts = SimulatorOptions(
            math_backend="jax",
            ode_solver_method="bdf",
            max_major_steps=20,
            return_context=False,
        )
        
        t_span = (0.0, 0.5)
        res = simulate(
            plant,
            ctx,
            t_span=t_span,
            options=opts,
            recorded_signals={"state": plant.output_ports[0]}
        )
        
        # Check shapes
        states = np.array(res.outputs["state"])
        self.assertEqual(states.shape[1], 2 * N * N)
        self.assertGreater(states.shape[0], 0)
        
        # Verify values are reasonable (e.g. positive chemical concentrations)
        self.assertTrue(np.all(states >= 0.0))
        
if __name__ == "__main__":
    unittest.main()
