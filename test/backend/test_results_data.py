# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest

import collimator
from collimator import DiagramBuilder, Simulator, SimulatorOptions
from collimator.library import Sine

import jax


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_jax_dump_buffer(backend):
    collimator.set_backend(backend)

    builder = DiagramBuilder()
    sine = builder.add(Sine(name="SineWave_0"))
    diagram = builder.build()

    options = SimulatorOptions(
        save_time_series=True,
        recorded_signals={"SineWave_0.out_0": sine.output_ports[0]},
        max_major_step_length=0.01,
    )
    simulator = Simulator(diagram, options=options)

    def _run_sim():
        context = diagram.create_context()
        results = simulator.advance_to(10, context)
        return results.results_data

    if backend == "jax":
        _run_sim = jax.jit(_run_sim)

    results1 = _run_sim()
    t1, _ = results1.finalize()
    assert len(t1) == 1001

    results2 = _run_sim()
    t2, _ = results2.finalize()
    assert len(t2) == 1001
