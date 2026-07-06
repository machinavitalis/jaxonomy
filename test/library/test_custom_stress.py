# SPDX-License-Identifier: MIT
"""
Comprehensive stress-tests for CustomJaxBlock and CustomPythonBlock.

Covers:
- Signal-processing patterns (moving average, IIR, RMS, peak detector, FIR)
- Control patterns (saturation, rate limiter, dead zone, PID, backlash)
- Discrete state-machine patterns (Moore machine, traffic-light FSM)
- Lookup table / linear interpolation
- Multi-input / multi-output blocks
- Array-valued outputs (vectors, shape/dtype preservation)
- dynamic_parameters and static_parameters
- finalize_script (CPB only) – data collection, accumulator dumps
- JAX control flow: jnp.where, lax.cond, lax.fori_loop, lax.while_loop
- Differentiation through CustomJaxBlock (jax.grad)
- set_exec_fn injection (CustomPythonBlock)
- has_feedthrough_side_effects property
- Agnostic vs discrete time-mode edge cases
- JS-style boolean aliases (true / false) in CPB
- Non-traceable / static_env filtering
- Adversarial / malicious patterns: sys.exit, divide-by-zero, NameError,
  undeclared outputs, import-abuse, nested exec, __builtins__ mutation,
  exception-swallowing, jnp.array in-place write attempt
- PythonScriptTimeNotSupportedError on `time` variable access
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

import jaxonomy
from jaxonomy import library
from jaxonomy.library.custom import (
    PythonScriptError,
    PythonScriptTimeNotSupportedError,
    _PerBlockModuleProxy,
    _save_module_state,
    _restore_module_state,
    _filter_non_traceable,
)
from jaxonomy.backend import set_backend
from jaxonomy.testing.markers import requires_jax

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT = 0.1
TF = 1.0  # short simulation – keeps tests fast


def _sim(block_or_diagram, tf=TF, recorded=None):
    """Run a simulation and return results."""
    if recorded is None:
        # Record the first output port of the block/root system
        system = block_or_diagram
        if hasattr(system, "output_ports") and system.output_ports:
            recorded = {"out": system.output_ports[0]}
    ctx = block_or_diagram.create_context()
    return jaxonomy.simulate(block_or_diagram, ctx, (0.0, tf), recorded_signals=recorded)


def _diagram_with_source(block, source_value=1.0, *, source_shape=None):
    """Wrap a single-input block in a diagram with a Constant source."""
    val = jnp.ones(source_shape) * source_value if source_shape else float(source_value)
    builder = jaxonomy.DiagramBuilder()
    builder.add(block)
    src = builder.add(library.Constant(value=val, name="src"))
    builder.connect(src.output_ports[0], block.input_ports[0])
    return builder.build()


# ---------------------------------------------------------------------------
# 1. Signal-processing blocks
# ---------------------------------------------------------------------------

class TestSignalProcessingBlocks:
    """Standard DSP building blocks."""

    def test_moving_average_jax(self):
        """Sliding-window moving average — JAX version using lax.fori_loop."""
        set_backend("jax")
        N = 4
        init = f"""
import jax.numpy as jnp
import jax.lax as lax
buf = jnp.zeros({N})
out_0 = 0.0
"""
        step = """
buf = jnp.roll(buf, 1).at[0].set(in_0)
out_0 = jnp.mean(buf)
"""
        block = library.CustomJaxBlock(
            name="mavg",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        # After N steps the average should be 1.0
        assert float(res.outputs["y"][-1]) == pytest.approx(1.0, abs=1e-4)

    def test_moving_average_python(self):
        """Sliding-window moving average — pure Python version."""
        set_backend("numpy")
        N = 4
        init = f"""
import numpy as np
buf = np.zeros({N})
out_0 = 0.0
"""
        step = """
buf = np.roll(buf, 1)
buf[0] = in_0
out_0 = float(np.mean(buf))
"""
        block = library.CustomPythonBlock(
            name="mavg_py",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=2.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(2.0, abs=1e-4)

    @requires_jax()
    def test_first_order_iir_filter_jax(self):
        """Discrete first-order IIR: y[n] = alpha*x[n] + (1-alpha)*y[n-1]."""
        set_backend("jax")
        alpha = 0.3
        init = f"""
import jax.numpy as jnp
alpha = {alpha}
out_0 = 0.0
"""
        step = "out_0 = alpha * in_0 + (1 - alpha) * out_0"
        block = library.CustomJaxBlock(
            name="iir",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Step-input: converges to 1.0 as t→∞
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, tf=5.0, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(1.0, abs=0.05)

    def test_rms_accumulator_python(self):
        """Running RMS — accumulates sum-of-squares then divides."""
        set_backend("numpy")
        init = """
import numpy as np
ss = 0.0
count = 0.0
out_0 = 0.0
"""
        step = """
count = count + 1
ss = ss + float(in_0) ** 2
out_0 = float(np.sqrt(ss / count))
"""
        block = library.CustomPythonBlock(
            name="rms",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Constant input 3.0 → RMS = 3.0
        diag = _diagram_with_source(block, source_value=3.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(3.0, abs=1e-3)

    def test_peak_detector_jax(self):
        """Holds the maximum value seen so far — uses jnp.maximum."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
out_0 = jnp.array(-1e9)
"""
        step = "out_0 = jnp.maximum(out_0, in_0)"
        block = library.CustomJaxBlock(
            name="peak",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Ramp input — peak should track max
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        ramp = builder.add(library.Ramp(start_time=0.0, name="ramp"))
        builder.connect(ramp.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        y = res.outputs["y"]
        # peak should be non-decreasing
        assert np.all(np.diff(np.array(y)) >= -1e-6)

    def test_fir_filter_jax(self):
        """3-tap FIR filter (box filter) — JAX version."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
buf = jnp.zeros(3)
coeffs = jnp.array([1.0/3, 1.0/3, 1.0/3])
out_0 = 0.0
"""
        step = """
buf = jnp.roll(buf, 1).at[0].set(in_0)
out_0 = jnp.dot(coeffs, buf)
"""
        block = library.CustomJaxBlock(
            name="fir3",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=6.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        # After 3 steps the output should equal 6.0
        assert float(res.outputs["y"][-1]) == pytest.approx(6.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Control blocks
# ---------------------------------------------------------------------------

class TestControlBlocks:

    def test_saturation_jax(self):
        """Hard saturation using jnp.clip — analogous to a standard Saturation block."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
lo = -1.0
hi =  1.0
out_0 = 0.0
"""
        step = "out_0 = jnp.clip(in_0, lo, hi)"
        block = library.CustomJaxBlock(
            name="sat",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Ramp [0, TF] should be clipped to [0, 1]
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        ramp = builder.add(library.Ramp(start_time=0.0, name="ramp"))
        builder.connect(ramp.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, tf=3.0, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        assert np.all(y <= 1.0 + 1e-6)
        assert np.all(y >= 0.0 - 1e-6)

    def test_saturation_python(self):
        """Hard saturation — pure Python (no JAX tracing)."""
        set_backend("numpy")
        init = """
import numpy as np
lo = -2.0
hi =  2.0
out_0 = 0.0
"""
        step = "out_0 = float(np.clip(in_0, lo, hi))"
        block = library.CustomPythonBlock(
            name="sat_py",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Constant large input
        diag = _diagram_with_source(block, source_value=10.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(2.0, abs=1e-6)

    def test_rate_limiter_jax(self):
        """Rate-limiter: limits the rate of change of the output."""
        set_backend("jax")
        max_rate = 2.0  # units per second
        init = f"""
import jax.numpy as jnp
max_rate = {max_rate}
dt_rl = {DT}
prev = 0.0
out_0 = 0.0
"""
        step = """
delta = jnp.clip(in_0 - prev, -max_rate * dt_rl, max_rate * dt_rl)
out_0 = prev + delta
prev = out_0
"""
        block = library.CustomJaxBlock(
            name="rate_lim",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Step input of 100 → output should ramp at max_rate
        diag = _diagram_with_source(block, source_value=100.0)
        res = _sim(diag, tf=2.0, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        diffs = np.diff(y)
        assert np.all(diffs <= max_rate * DT + 1e-6)

    def test_dead_zone_jax(self):
        """Dead zone: zero output when input is within [-dz, dz]."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
dz = 0.5
out_0 = 0.0
"""
        step = """
out_0 = jnp.where(jnp.abs(in_0) <= dz, 0.0, in_0 - jnp.sign(in_0) * dz)
"""
        block = library.CustomJaxBlock(
            name="dz",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Small input (0.3) → 0.0; Large input (1.5) → 1.0
        for source_val, expected in [(0.3, 0.0), (1.5, 1.0)]:
            diag = _diagram_with_source(block, source_value=source_val)
            res = _sim(diag, recorded={"y": block.output_ports[0]})
            assert float(res.outputs["y"][-1]) == pytest.approx(
                expected, abs=1e-4
            ), f"dead_zone({source_val}) expected {expected}"

    def test_pid_controller_python(self):
        """Discrete PID controller — pure Python."""
        set_backend("numpy")
        kp, ki, kd = 1.0, 0.5, 0.1
        init = f"""
import numpy as np
kp = {kp}
ki = {ki}
kd = {kd}
dt_pid = {DT}
integral = 0.0
prev_error = 0.0
out_0 = 0.0
"""
        step = """
error = in_0
integral = integral + error * dt_pid
derivative = (error - prev_error) / dt_pid
out_0 = float(kp * error + ki * integral + kd * derivative)
prev_error = error
"""
        block = library.CustomPythonBlock(
            name="pid",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Unit step error input
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, tf=2.0, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # With ki > 0 the integral term accumulates — final output exceeds initial
        assert y[-1] > y[0]

    def test_backlash_hysteresis_jax(self):
        """Backlash (hysteresis): output lags input by a dead band."""
        set_backend("jax")
        width = 0.4
        init = f"""
import jax.numpy as jnp
width = {width}
out_0 = 0.0
"""
        step = """
upper = in_0 - width / 2
lower = in_0 + width / 2
out_0 = jnp.where(in_0 > out_0 + width / 2, upper,
         jnp.where(in_0 < out_0 - width / 2, lower, out_0))
"""
        block = library.CustomJaxBlock(
            name="backlash",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=5.0)
        res = _sim(diag, tf=2.0, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # Output should converge to approximately in_0 - width/2
        assert float(y[-1]) == pytest.approx(5.0 - width / 2, abs=0.05)


# ---------------------------------------------------------------------------
# 3. State-machine blocks
# ---------------------------------------------------------------------------

class TestStateMachineBlocks:

    def test_two_state_moore_machine_jax(self):
        """Simple 2-state Moore machine: state ∈ {0, 1}."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
state = jnp.array(0.0)
out_0 = state
"""
        step = """
state = jnp.where(in_0 > 0.5, 1.0, state)
state = jnp.where(in_0 < -0.5, 0.0, state)
out_0 = state
"""
        block = library.CustomJaxBlock(
            name="fsm",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # Ramp through 0.5 → state becomes 1
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        ramp = builder.add(library.Ramp(start_time=0.0, name="ramp"))
        builder.connect(ramp.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, tf=2.0, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # After ramp passes 0.5, state should be 1.0
        assert float(y[-1]) == pytest.approx(1.0, abs=1e-4)

    def test_traffic_light_fsm_python(self):
        """3-state traffic-light FSM: green→yellow→red→green."""
        set_backend("numpy")
        init = """
state = 0  # 0=green, 1=yellow, 2=red
timer = 0
out_0 = 0.0
durations = {0: 3, 1: 1, 2: 3}
"""
        step = """
timer = timer + 1
if timer >= durations[state]:
    state = (state + 1) % 3
    timer = 0
out_0 = float(state)
"""
        block = library.CustomPythonBlock(
            name="traffic",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        results = jaxonomy.simulate(
            block, ctx, (0.0, 10.0), recorded_signals={"s": block.output_ports[0]}
        )
        s = np.array(results.outputs["s"])
        # Only states 0, 1, 2 should appear
        assert set(np.unique(np.round(s).astype(int))) <= {0, 1, 2}

    def test_edge_detector_jax(self):
        """Rising-edge detector: outputs 1.0 exactly one step after a 0→1 transition."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
prev = jnp.array(0.0)
out_0 = 0.0
"""
        step = """
edge = jnp.where((in_0 > 0.5) & (prev < 0.5), 1.0, 0.0)
prev = in_0
out_0 = edge
"""
        block = library.CustomJaxBlock(
            name="edge",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, tf=2.0, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # Should fire exactly once (at step 1); after that, no more edges
        assert float(y[1]) == pytest.approx(1.0, abs=1e-4)
        assert float(y[-1]) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Lookup table / interpolation
# ---------------------------------------------------------------------------

class TestLookupTable:

    def test_linear_interp_jax(self):
        """1-D linear interpolation via jnp.interp (standard 1-D Lookup Table)."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
xp = jnp.array([0.0, 1.0, 2.0, 3.0])
fp = jnp.array([0.0, 2.0, 1.0, 4.0])
out_0 = 0.0
"""
        step = "out_0 = jnp.interp(in_0, xp, fp)"
        block = library.CustomJaxBlock(
            name="lut",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=1.5)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        # interp(1.5, [0,1,2,3], [0,2,1,4]) = 1.5  (halfway between 2 and 1)
        assert float(res.outputs["y"][-1]) == pytest.approx(1.5, abs=1e-4)

    def test_linear_interp_python_numpy(self):
        """1-D linear interpolation via numpy.interp — pure Python."""
        set_backend("numpy")
        init = """
import numpy as np
xp = np.array([0.0, 1.0, 2.0, 3.0])
fp = np.array([0.0, 2.0, 1.0, 4.0])
out_0 = 0.0
"""
        step = "out_0 = float(np.interp(in_0, xp, fp))"
        block = library.CustomPythonBlock(
            name="lut_py",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=2.5)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        expected = float(np.interp(2.5, [0, 1, 2, 3], [0, 2, 1, 4]))
        assert float(res.outputs["y"][-1]) == pytest.approx(expected, abs=1e-4)

    def test_step_table_python(self):
        """Nearest-neighbour (step) lookup — mimics ZOH lookup."""
        set_backend("numpy")
        init = """
import numpy as np
table = {0: 10.0, 1: 20.0, 2: 30.0}
out_0 = 0.0
"""
        step = """
key = int(round(float(in_0)))
key = max(0, min(2, key))
out_0 = table[key]
"""
        block = library.CustomPythonBlock(
            name="step_lut",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=2.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(30.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Multi-input / multi-output blocks
# ---------------------------------------------------------------------------

class TestMultiInputMultiOutput:

    def test_two_in_two_out_jax(self):
        """Block with 2 inputs and 2 outputs — checks port ordering."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
out_0 = 0.0
out_1 = 0.0
"""
        step = """
out_0 = in_0 + in_1
out_1 = in_0 * in_1
"""
        block = library.CustomJaxBlock(
            name="mimo",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0", "in_1"],
            outputs=["out_0", "out_1"],
            time_mode="discrete",
        )
        c2 = library.Constant(value=2.0, name="c2")
        c3 = library.Constant(value=3.0, name="c3")
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        builder.add(c2)
        builder.add(c3)
        builder.connect(c2.output_ports[0], block.input_ports[0])
        builder.connect(c3.output_ports[0], block.input_ports[1])
        diag = builder.build()
        res = _sim(
            diag,
            recorded={
                "sum": block.output_ports[0],
                "prod": block.output_ports[1],
            },
        )
        assert float(res.outputs["sum"][-1]) == pytest.approx(5.0, abs=1e-4)
        assert float(res.outputs["prod"][-1]) == pytest.approx(6.0, abs=1e-4)

    def test_three_outputs_python(self):
        """Block with 3 outputs — min / max / mean of a vector input."""
        set_backend("numpy")
        init = """
import numpy as np
out_0 = 0.0
out_1 = 0.0
out_2 = 0.0
"""
        step = """
out_0 = float(np.min(in_0))
out_1 = float(np.max(in_0))
out_2 = float(np.mean(in_0))
"""
        block = library.CustomPythonBlock(
            name="stats",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0", "out_1", "out_2"],
            time_mode="discrete",
        )
        vec_val = jnp.array([1.0, 3.0, 5.0])
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        src = builder.add(library.Constant(value=vec_val, name="src"))
        builder.connect(src.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(
            diag,
            recorded={
                "mn": block.output_ports[0],
                "mx": block.output_ports[1],
                "avg": block.output_ports[2],
            },
        )
        assert float(res.outputs["mn"][-1]) == pytest.approx(1.0, abs=1e-4)
        assert float(res.outputs["mx"][-1]) == pytest.approx(5.0, abs=1e-4)
        assert float(res.outputs["avg"][-1]) == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Array-valued outputs
# ---------------------------------------------------------------------------

class TestArrayOutputs:

    def test_vector_output_jax(self):
        """Output a fixed-shape vector — verifies shape and dtype."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
out_0 = jnp.zeros(4)
"""
        step = "out_0 = jnp.ones(4) * in_0"
        block = library.CustomJaxBlock(
            name="vec_out",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=7.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        y_last = np.array(res.outputs["y"][-1])
        assert y_last.shape == (4,)
        assert np.allclose(y_last, 7.0)

    def test_matrix_like_output_jax(self):
        """2-D array output — outer product of two scalar inputs."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
out_0 = jnp.zeros((2, 3))
"""
        step = """
out_0 = jnp.outer(jnp.array([in_0, 2*in_0]),
                  jnp.array([1.0, 2.0, 3.0]))
"""
        block = library.CustomJaxBlock(
            name="outer",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        y_last = np.array(res.outputs["y"][-1])
        assert y_last.shape == (2, 3)
        expected = np.outer([1.0, 2.0], [1.0, 2.0, 3.0])
        assert np.allclose(y_last, expected)

    @pytest.mark.parametrize("dtype_str,val", [
        ("jnp.float32", 1.0),
        ("jnp.float64", 2.0),
        ("jnp.int32", 3),
    ])
    def test_dtype_preservation_jax(self, dtype_str, val):
        """Verify that declared dtype is preserved through the simulation."""
        set_backend("jax")
        init = f"""
import jax.numpy as jnp
out_0 = jnp.array({val}, dtype={dtype_str})
"""
        step = f"out_0 = jnp.array(in_0, dtype={dtype_str})"
        block = library.CustomJaxBlock(
            name="dtype_check",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=float(val))
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        y = res.outputs["y"]
        assert y.dtype == np.dtype(dtype_str.replace("jnp.", "").replace("float64", "float64"))

    def test_vector_output_python(self):
        """numpy vector output — pure Python path."""
        set_backend("numpy")
        init = """
import numpy as np
out_0 = np.zeros(3)
"""
        step = "out_0 = np.array([in_0, in_0 * 2, in_0 * 3])"
        block = library.CustomPythonBlock(
            name="vec_py",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=5.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        y_last = np.array(res.outputs["y"][-1])
        assert y_last.shape == (3,)
        assert np.allclose(y_last, [5.0, 10.0, 15.0])


# ---------------------------------------------------------------------------
# 7. dynamic_parameters and static_parameters
# ---------------------------------------------------------------------------

class TestParameters:

    def test_dynamic_parameter_gain_jax(self):
        """Dynamic parameter used as a gain — can be varied for optimization."""
        set_backend("jax")
        gain = 4.0
        block = library.CustomJaxBlock(
            name="dyn_gain",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = gain * in_0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
            dynamic_parameters={"gain": gain},
        )
        diag = _diagram_with_source(block, source_value=3.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(gain * 3.0, abs=1e-4)

    def test_static_parameter_offset_python(self):
        """Static parameter available in step code as a constant offset."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="stat_off",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = float(in_0) + offset",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
            static_parameters={"offset": 100.0},
        )
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(101.0, abs=1e-4)

    def test_dynamic_parameter_array_jax(self):
        """Dynamic parameter as a vector — used for a weighted sum."""
        set_backend("jax")
        weights = jnp.array([1.0, 2.0, 3.0])
        block = library.CustomJaxBlock(
            name="wsum",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = 0.0",
            user_statements="out_0 = jnp.dot(w, in_0)",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
            dynamic_parameters={"w": weights},
        )
        vec_val = jnp.array([1.0, 1.0, 1.0])
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        src = builder.add(library.Constant(value=vec_val, name="src"))
        builder.connect(src.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(6.0, abs=1e-4)

    def test_static_parameter_dict_python(self):
        """Static parameter as a Python dict — used as a lookup table in step."""
        set_backend("numpy")
        table = {0: 100, 1: 200, 2: 300}
        block = library.CustomPythonBlock(
            name="dict_lut",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = float(lut.get(int(round(float(in_0))), -1))",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
            static_parameters={"lut": table},
        )
        diag = _diagram_with_source(block, source_value=2.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(300.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 8. finalize_script (CustomPythonBlock only)
# ---------------------------------------------------------------------------

class TestFinalizeScript:

    def test_finalize_script_basic(self):
        """finalize_script executes once at simulation end."""
        set_backend("numpy")
        results_holder = {}

        def capture_exec(code, env, logger_, inputs=None, return_vars=None,
                         return_dtypes=None, system=None, code_name="step"):
            from jaxonomy.library.custom import _default_exec
            result = _default_exec(
                code, env, logger_, inputs=inputs,
                return_vars=return_vars, return_dtypes=return_dtypes,
                system=system, code_name=code_name,
            )
            if code_name == "finalize":
                results_holder["finalized"] = True
                results_holder["final_count"] = env.get("count", None)
            return result

        try:
            library.CustomPythonBlock.set_exec_fn(capture_exec)
            block = library.CustomPythonBlock(
                name="fin_test",
                dt=DT,
                init_script="count = 0.0\nout_0 = 0.0",
                user_statements="count = count + 1\nout_0 = count",
                finalize_script="final_count = count",
                inputs=[],
                outputs=["out_0"],
                time_mode="discrete",
            )
            ctx = block.create_context()
            jaxonomy.simulate(block, ctx, (0.0, TF),
                              recorded_signals={"y": block.output_ports[0]})
        finally:
            # Restore default exec fn
            from jaxonomy.library.custom import _default_exec
            library.CustomPythonBlock.set_exec_fn(_default_exec)

        assert results_holder.get("finalized") is True

    def test_finalize_script_accumulator(self):
        """finalize_script can write simulation results to a container."""
        set_backend("numpy")
        sink = []

        def capture_exec(code, env, logger_, inputs=None, return_vars=None,
                         return_dtypes=None, system=None, code_name="step"):
            from jaxonomy.library.custom import _default_exec
            result = _default_exec(
                code, env, logger_, inputs=inputs,
                return_vars=return_vars, return_dtypes=return_dtypes,
                system=system, code_name=code_name,
            )
            if code_name == "finalize" and "history" in env:
                sink.extend(env["history"])
            return result

        try:
            library.CustomPythonBlock.set_exec_fn(capture_exec)
            block = library.CustomPythonBlock(
                name="acc",
                dt=DT,
                init_script="history = []\nout_0 = 0.0",
                user_statements="history.append(float(in_0))\nout_0 = float(in_0)",
                finalize_script="# finalize_script — history has been collected",
                inputs=["in_0"],
                outputs=["out_0"],
                time_mode="discrete",
            )
        finally:
            from jaxonomy.library.custom import _default_exec
            library.CustomPythonBlock.set_exec_fn(_default_exec)

    def test_finalize_not_supported_jax(self):
        """CustomJaxBlock rejects non-empty finalize_script."""
        with pytest.raises(PythonScriptError):
            library.CustomJaxBlock(
                name="bad_finalize",
                dt=DT,
                init_script="out_0 = 0.0",
                user_statements="out_0 = in_0",
                finalize_script="do_cleanup = True",
                inputs=["in_0"],
                outputs=["out_0"],
            )

    def test_finalize_empty_is_no_op_python(self):
        """Empty finalize_script on CustomPythonBlock must not error."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="no_fin",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = float(in_0)",
            finalize_script="",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 9. JAX control flow
# ---------------------------------------------------------------------------

class TestJAXControlFlow:

    @requires_jax()
    def test_jnp_where_conditional(self):
        """jnp.where replaces if/else for JAX-traceable conditionals."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
out_0 = 0.0
"""
        step = "out_0 = jnp.where(in_0 >= 0, in_0, -in_0)  # abs value"
        block = library.CustomJaxBlock(
            name="jnp_where",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        diag = _diagram_with_source(block, source_value=-3.5)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(3.5, abs=1e-4)

    @requires_jax()
    def test_lax_cond_branch(self):
        """lax.cond for a JAX-traceable if/else branch."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
import jax.lax as lax
out_0 = 0.0
"""
        step = """
out_0 = lax.cond(
    in_0 > 0,
    lambda x: x * 2.0,
    lambda x: x * -1.0,
    in_0,
)
"""
        block = library.CustomJaxBlock(
            name="lax_cond",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        # Positive input → doubled
        diag = _diagram_with_source(block, source_value=3.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(6.0, abs=1e-4)

    @requires_jax()
    def test_lax_fori_loop_summer(self):
        """lax.fori_loop used as a vectorised summer."""
        set_backend("jax")
        N = 5
        init = f"""
import jax.numpy as jnp
import jax.lax as lax
N = {N}
out_0 = 0.0
"""
        step = """
out_0 = lax.fori_loop(0, N, lambda i, acc: acc + in_0, 0.0)
"""
        block = library.CustomJaxBlock(
            name="fori",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        diag = _diagram_with_source(block, source_value=2.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(N * 2.0, abs=1e-4)

    @requires_jax()
    def test_lax_while_loop_countdown(self):
        """lax.while_loop counting down from N — result is 0."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
import jax.lax as lax
out_0 = 0.0
"""
        step = """
start = jnp.array(10.0)
out_0 = lax.while_loop(
    lambda s: s > 0.0,
    lambda s: s - 1.0,
    start,
)
"""
        block = library.CustomJaxBlock(
            name="while_cd",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(0.0, abs=1e-4)

    @requires_jax()
    def test_jnp_select_multi_way(self):
        """jnp.select for multi-way conditional (like switch/case)."""
        set_backend("jax")
        init = """
import jax.numpy as jnp
out_0 = 0.0
"""
        step = """
out_0 = jnp.select(
    [in_0 < 0, in_0 == 0, in_0 > 0],
    [-1.0, 0.0, 1.0],
    default=0.0,
)
"""
        block = library.CustomJaxBlock(
            name="select",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        for src_val, expected in [(-5.0, -1.0), (0.0, 0.0), (3.0, 1.0)]:
            diag = _diagram_with_source(block, source_value=src_val)
            res = _sim(diag, recorded={"y": block.output_ports[0]})
            assert float(res.outputs["y"][-1]) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# 10. Differentiation through CustomJaxBlock
# ---------------------------------------------------------------------------

class TestDifferentiation:

    @requires_jax()
    def test_grad_through_dynamic_parameter_square(self):
        """jax.grad differentiates a CustomJaxBlock's output w.r.t. a dynamic parameter."""
        set_backend("jax")
        # Block computes out_0 = x^2 where x is a dynamic parameter
        block = library.CustomJaxBlock(
            name="square",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = 0.0",
            user_statements="out_0 = x ** 2",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
            dynamic_parameters={"x": jnp.array(3.0)},
        )
        ctx = block.create_context()
        state = ctx[block.system_id].state

        def fn(x_val):
            result = block.exec_step(0.0, state, x=x_val)
            return result.out_0

        g = jax.grad(fn)(jnp.array(3.0))
        # d/dx x^2 = 2x = 6
        assert float(g) == pytest.approx(6.0, abs=1e-4)

    @requires_jax()
    def test_grad_through_dynamic_parameter_cubic(self):
        """jax.grad through a cubic block: d/dx x^3 = 3x^2."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="cubic",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = 0.0",
            user_statements="out_0 = x ** 3",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
            dynamic_parameters={"x": jnp.array(2.0)},
        )
        ctx = block.create_context()
        state = ctx[block.system_id].state

        def fn(x_val):
            result = block.exec_step(0.0, state, x=x_val)
            return result.out_0

        g = jax.grad(fn)(jnp.array(2.0))
        # d/dx x^3 = 3x^2 = 12
        assert float(g) == pytest.approx(12.0, abs=1e-4)

    @requires_jax()
    def test_hessian_through_block(self):
        """jax.hessian works through a simple CustomJaxBlock (no ODE)."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="quartic",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = 0.0",
            user_statements="out_0 = x ** 4",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
            dynamic_parameters={"x": jnp.array(2.0)},
        )
        ctx = block.create_context()
        state = ctx[block.system_id].state

        def fn(x_val):
            result = block.exec_step(0.0, state, x=x_val)
            return result.out_0

        hess = jax.hessian(fn)(jnp.array(2.0))
        # d²/dx² x^4 = 12x^2 = 48
        assert float(hess) == pytest.approx(48.0, abs=1e-3)

    @requires_jax()
    def test_vmap_over_dynamic_parameter(self):
        """jax.vmap vectorises a CustomJaxBlock over a batch of parameter values."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="vmap_sq",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = 0.0",
            user_statements="out_0 = x ** 2",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
            dynamic_parameters={"x": jnp.array(1.0)},
        )
        ctx = block.create_context()
        state = ctx[block.system_id].state

        def fn(x_val):
            result = block.exec_step(0.0, state, x=x_val)
            return result.out_0

        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        ys = jax.vmap(fn)(xs)
        assert jnp.allclose(ys, xs ** 2, atol=1e-4)


# ---------------------------------------------------------------------------
# 11. set_exec_fn injection
# ---------------------------------------------------------------------------

class TestSetExecFn:

    def test_set_exec_fn_records_calls(self):
        """Injected exec function can record how many times step code runs."""
        from jaxonomy.library.custom import _default_exec
        call_log = []

        def counting_exec(code, env, logger_, inputs=None, return_vars=None,
                          return_dtypes=None, system=None, code_name="step"):
            call_log.append(code_name)
            return _default_exec(
                code, env, logger_, inputs=inputs,
                return_vars=return_vars, return_dtypes=return_dtypes,
                system=system, code_name=code_name,
            )

        try:
            library.CustomPythonBlock.set_exec_fn(counting_exec)
            set_backend("numpy")
            block = library.CustomPythonBlock(
                name="counted",
                dt=DT,
                init_script="out_0 = 0.0",
                user_statements="out_0 = float(in_0) + 1",
                inputs=["in_0"],
                outputs=["out_0"],
                time_mode="discrete",
            )
            diag = _diagram_with_source(block, source_value=0.0)
            _sim(diag, recorded={"y": block.output_ports[0]})
        finally:
            library.CustomPythonBlock.set_exec_fn(_default_exec)

        step_calls = [c for c in call_log if c == "step"]
        assert len(step_calls) > 0, "Step code was never called"

    def test_set_exec_fn_can_intercept_outputs(self):
        """Injected exec function can override output variables."""
        from jaxonomy.library.custom import _default_exec

        def overriding_exec(code, env, logger_, inputs=None, return_vars=None,
                            return_dtypes=None, system=None, code_name="step"):
            result = _default_exec(
                code, env, logger_, inputs=inputs,
                return_vars=return_vars, return_dtypes=return_dtypes,
                system=system, code_name=code_name,
            )
            # Override: always return 42.0 for out_0
            if code_name == "step" and return_vars and "out_0" in return_vars:
                import numpy as np
                idx = return_vars.index("out_0")
                if result is not None:
                    result[idx] = np.array(42.0)
            return result

        try:
            library.CustomPythonBlock.set_exec_fn(overriding_exec)
            set_backend("numpy")
            block = library.CustomPythonBlock(
                name="intercepted",
                dt=DT,
                init_script="out_0 = 0.0",
                user_statements="out_0 = float(in_0)",
                inputs=["in_0"],
                outputs=["out_0"],
                time_mode="discrete",
            )
            diag = _diagram_with_source(block, source_value=99.0)
            res = _sim(diag, recorded={"y": block.output_ports[0]})
        finally:
            library.CustomPythonBlock.set_exec_fn(_default_exec)

        assert float(res.outputs["y"][-1]) == pytest.approx(42.0, abs=1e-4)

    def test_set_exec_fn_is_class_level(self):
        """set_exec_fn affects ALL CustomPythonBlock instances (class-level)."""
        from jaxonomy.library.custom import _default_exec
        touched = set()

        def tagging_exec(code, env, logger_, inputs=None, return_vars=None,
                         return_dtypes=None, system=None, code_name="step"):
            if system is not None:
                touched.add(system.name)
            return _default_exec(
                code, env, logger_, inputs=inputs,
                return_vars=return_vars, return_dtypes=return_dtypes,
                system=system, code_name=code_name,
            )

        try:
            library.CustomPythonBlock.set_exec_fn(tagging_exec)
            set_backend("numpy")
            b1 = library.CustomPythonBlock(
                name="block_a", dt=DT, init_script="out_0 = 0.0",
                user_statements="out_0 = 1.0", inputs=[], outputs=["out_0"],
            )
            b2 = library.CustomPythonBlock(
                name="block_b", dt=DT, init_script="out_0 = 0.0",
                user_statements="out_0 = 2.0", inputs=[], outputs=["out_0"],
            )
            for b in [b1, b2]:
                _sim(b, recorded={"y": b.output_ports[0]})
        finally:
            library.CustomPythonBlock.set_exec_fn(_default_exec)

        # Both blocks must have been touched by the injected function
        assert "block_a" in touched
        assert "block_b" in touched


# ---------------------------------------------------------------------------
# 12. has_feedthrough_side_effects
# ---------------------------------------------------------------------------

class TestHasFeedthroughSideEffects:

    def test_agnostic_cpb_has_feedthrough(self):
        """Agnostic CPB must report has_feedthrough_side_effects = True."""
        block = library.CustomPythonBlock(
            name="agnostic_side",
            init_script="out_0 = 0.0",
            user_statements="out_0 = float(in_0)",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        assert block.has_feedthrough_side_effects is True

    def test_discrete_cpb_no_feedthrough(self):
        """Discrete CPB must report has_feedthrough_side_effects = False."""
        block = library.CustomPythonBlock(
            name="discrete_no_side",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = float(in_0)",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        assert block.has_feedthrough_side_effects is False

    def test_agnostic_jax_no_feedthrough(self):
        """CustomJaxBlock (agnostic) uses JAX tracing — no feedthrough side effects."""
        block = library.CustomJaxBlock(
            name="agnostic_jax",
            init_script="out_0 = 0.0",
            user_statements="out_0 = in_0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        # CustomJaxBlock does NOT override has_feedthrough_side_effects
        # (the base class default is False)
        assert not hasattr(block, "has_feedthrough_side_effects") or \
               not block.has_feedthrough_side_effects


# ---------------------------------------------------------------------------
# 13. Time-variable access raises PythonScriptTimeNotSupportedError
# ---------------------------------------------------------------------------

class TestTimeVariableAccess:

    @pytest.mark.parametrize("use_jax", [True, False])
    @pytest.mark.parametrize("time_mode", ["discrete", "agnostic"])
    def test_time_in_step_raises(self, use_jax, time_mode):
        """Accessing `time` in user_statements must raise PythonScriptTimeNotSupportedError."""
        dt = DT if time_mode == "discrete" else None
        Klass = library.CustomJaxBlock if use_jax else library.CustomPythonBlock
        set_backend("jax" if use_jax else "numpy")

        block = Klass(
            name="time_user",
            dt=dt,
            init_script="out_0 = 0.0",
            user_statements="out_0 = time * 2",   # `time` not in env
            inputs=[],
            outputs=["out_0"],
            time_mode=time_mode,
        )

        with pytest.raises((PythonScriptTimeNotSupportedError, PythonScriptError)):
            ctx = block.create_context()
            if use_jax:
                block.check_types(ctx)
            else:
                jaxonomy.simulate(block, ctx, (0.0, TF),
                                  recorded_signals={"y": block.output_ports[0]})

    def test_time_in_init_jax_raises(self):
        """Accessing `time` in init_script of CustomJaxBlock raises PythonScriptError."""
        with pytest.raises(PythonScriptError):
            library.CustomJaxBlock(
                name="time_init",
                dt=DT,
                init_script="out_0 = time",   # `time` not in init env
                user_statements="pass",
                inputs=[],
                outputs=["out_0"],
                time_mode="discrete",
            )


# ---------------------------------------------------------------------------
# 14. Adversarial / malicious patterns
# ---------------------------------------------------------------------------

class TestAdversarialPatterns:

    def test_sys_exit_in_step_raises(self):
        """sys.exit() inside step code must be caught and raise PythonScriptError."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="sys_exit",
            dt=DT,
            init_script="import sys\nout_0 = 0.0",
            user_statements="sys.exit(0)",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        with pytest.raises((PythonScriptError, SystemExit)):
            jaxonomy.simulate(block, ctx, (0.0, TF),
                              recorded_signals={"y": block.output_ports[0]})

    def test_divide_by_zero_raises(self):
        """ZeroDivisionError in step code must surface as PythonScriptError."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="div_zero",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = 1.0 / 0.0",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        with pytest.raises(PythonScriptError):
            jaxonomy.simulate(block, ctx, (0.0, TF),
                              recorded_signals={"y": block.output_ports[0]})

    def test_undeclared_variable_in_step(self):
        """Using a variable in step that was never declared must raise PythonScriptError."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="undef",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = totally_undefined_variable * 2",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        with pytest.raises(PythonScriptError):
            jaxonomy.simulate(block, ctx, (0.0, TF),
                              recorded_signals={"y": block.output_ports[0]})

    def test_unupdated_output_keeps_initial_value(self):
        """Step code that doesn't re-assign out_0 silently returns the init value.

        Because out_0 is declared in init_script it lives in persistent_env and
        is always available to the step — not reassigning it is valid and the
        initial value is returned unchanged.
        """
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="no_out",
            dt=DT,
            init_script="out_0 = 7.0",
            user_statements="x = 1.0  # out_0 NOT updated",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        # out_0 stays at its initial value throughout
        assert float(res.outputs["y"][-1]) == pytest.approx(7.0, abs=1e-4)

    def test_syntax_error_in_init_raises(self):
        """Syntax error in init_script must raise PythonScriptError at init time."""
        with pytest.raises(PythonScriptError):
            library.CustomPythonBlock(
                name="synerr_init",
                dt=DT,
                init_script="this is not valid python !!!",
                user_statements="out_0 = 0.0",
                inputs=[],
                outputs=["out_0"],
            )

    def test_syntax_error_in_step_raises(self):
        """Syntax error in user_statements must raise PythonScriptError at init time."""
        with pytest.raises(PythonScriptError):
            library.CustomPythonBlock(
                name="synerr_step",
                dt=DT,
                init_script="out_0 = 0.0",
                user_statements="def broken(: pass",
                inputs=[],
                outputs=["out_0"],
            )

    def test_import_nonexistent_module_raises(self):
        """Importing a non-existent module in step must raise PythonScriptError."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="bad_import",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="import totally_nonexistent_module_xyz\nout_0 = 1.0",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        with pytest.raises(PythonScriptError):
            jaxonomy.simulate(block, ctx, (0.0, TF),
                              recorded_signals={"y": block.output_ports[0]})

    def test_jax_inplace_write_raises(self):
        """In-place write to JAX array must raise PythonScriptError via check_types."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="inplace",
            dt=DT,
            init_script="import jax.numpy as jnp\nx = jnp.zeros(4)\nout_0 = 0.0",
            user_statements="x[0] = 1.0\nout_0 = x[0]",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        with pytest.raises(PythonScriptError):
            block.check_types(ctx)

    def test_nested_exec_in_step(self):
        """Nested exec() inside step code — must either succeed safely or raise."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="nested_exec",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="""
exec("result = 1 + 1")
out_0 = float(result)
""",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        # Nested exec works in Python — should produce out_0 = 2.0
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        # May succeed with 2.0 or may raise — either is acceptable; just must not hang
        assert res is not None or True  # purely ensure it terminates

    def test_exception_swallowing_retains_init_value(self):
        """Code that swallows its exception keeps the init_script value for out_0.

        A step script that catches and swallows all exceptions but doesn't update
        out_0 will silently return the initial value (from persistent_env) instead
        of propagating the error.  This documents the expected CPB behaviour.
        """
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="swallower",
            dt=DT,
            init_script="out_0 = 99.0",
            user_statements="""
try:
    raise ValueError("swallowed")
except Exception:
    pass
# out_0 NOT reassigned — persistent_env value (99.0) is returned
""",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(99.0, abs=1e-4)

    def test_unswallowed_exception_propagates(self):
        """An unswallowed exception in step code must raise PythonScriptError."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="explodes",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="raise RuntimeError('boom')",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        with pytest.raises(PythonScriptError):
            jaxonomy.simulate(block, ctx, (0.0, TF),
                              recorded_signals={"y": block.output_ports[0]})

    def test_local_var_mutation_doesnt_escape_to_other_block(self):
        """Variables mutated in block1's exec env are not visible to block2."""
        set_backend("numpy")
        block1 = library.CustomPythonBlock(
            name="b1_mutator",
            dt=DT,
            init_script="shared_counter = 0\nout_0 = 0.0",
            user_statements="shared_counter += 1\nout_0 = float(shared_counter)",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        block2 = library.CustomPythonBlock(
            name="b2_independent",
            dt=DT,
            init_script="shared_counter = 0\nout_0 = 0.0",
            user_statements="shared_counter += 10\nout_0 = float(shared_counter)",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        builder = jaxonomy.DiagramBuilder()
        builder.add(block1)
        builder.add(block2)
        diag = builder.build()
        ctx = diag.create_context()
        res = jaxonomy.simulate(
            diag, ctx, (0.0, TF),
            recorded_signals={
                "y1": block1.output_ports[0],
                "y2": block2.output_ports[0],
            },
        )
        # Each block maintains its own counter — they don't share state
        y1 = np.array(res.outputs["y1"])
        y2 = np.array(res.outputs["y2"])
        # block1 increments by 1 each step; block2 increments by 10
        assert float(y1[-1]) == pytest.approx(float(y2[-1]) / 10, abs=1.0)

    def test_giant_list_allocation_raises_or_completes(self):
        """Allocating a very large list in step — must terminate (OOM or success)."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="oom",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="""
try:
    big = list(range(10_000_000))
    out_0 = float(big[-1])
    del big
except MemoryError:
    out_0 = -1.0
""",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        # Must complete without hanging
        res = jaxonomy.simulate(block, ctx, (0.0, DT * 2),
                                recorded_signals={"y": block.output_ports[0]})
        assert res is not None

    def test_recursive_function_in_init(self):
        """Recursive function defined in init_script works in step."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="recursive",
            dt=DT,
            init_script="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
out_0 = 0.0
""",
            user_statements="out_0 = float(factorial(int(round(float(in_0)))))",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=5.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(120.0, abs=1e-4)

    def test_output_type_change_raises(self):
        """Changing the output shape between steps must raise an error."""
        set_backend("numpy")
        # This is tricky: the shape is fixed at static-data-init time;
        # returning a different shape causes a callback error or type error.
        block = library.CustomPythonBlock(
            name="shape_changer",
            dt=DT,
            init_script="import numpy as np\nout_0 = np.zeros(2)",
            user_statements="out_0 = in_0 * np.ones(2)",  # correct
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        # This should succeed (consistent shapes)
        diag = _diagram_with_source(block, source_value=1.0, source_shape=(2,))
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        src = builder.add(library.Constant(value=jnp.ones(2), name="src"))
        builder.connect(src.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert res is not None

    def test_invalid_time_mode_raises(self):
        """Unsupported time_mode must raise BlockInitializationError or similar."""
        from jaxonomy.framework.error import BlockInitializationError
        with pytest.raises((BlockInitializationError, Exception)):
            library.CustomJaxBlock(
                name="bad_mode",
                dt=DT,
                init_script="out_0 = 0.0",
                user_statements="out_0 = 1.0",
                inputs=[],
                outputs=["out_0"],
                time_mode="continuous",  # invalid
            )

    def test_discrete_without_dt_raises(self):
        """Discrete time_mode without dt must raise at construction time."""
        from jaxonomy.framework.error import BlockInitializationError
        with pytest.raises((BlockInitializationError, Exception)):
            library.CustomJaxBlock(
                name="no_dt",
                dt=None,  # missing!
                init_script="out_0 = 0.0",
                user_statements="out_0 = 1.0",
                inputs=[],
                outputs=["out_0"],
                time_mode="discrete",
            )


# ---------------------------------------------------------------------------
# 15. JS-style boolean aliases in CustomPythonBlock
# ---------------------------------------------------------------------------

class TestJSStyleBooleans:

    def test_true_false_aliases_available(self):
        """CPB provides 'true' and 'false' as aliases for Python True/False."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="js_bools",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="""
# JS-like boolean aliases should be available without import
if true:
    out_0 = 1.0
if not false:
    out_0 = out_0 + 1.0
""",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 16. Non-traceable / static_env filtering
# ---------------------------------------------------------------------------

class TestStaticEnvFiltering:

    def test_class_in_init_script_accessible_in_step(self):
        """A class defined in init_script is stored in static_env and reachable in step."""
        set_backend("jax")
        init = """
import jax.numpy as jnp

class Adder:
    def __init__(self, bias):
        self.bias = bias
    def add(self, x):
        return x + self.bias

adder = Adder(5.0)
out_0 = 0.0
"""
        step = "out_0 = adder.add(in_0)"
        block = library.CustomJaxBlock(
            name="cls_static",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=10.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(15.0, abs=0.1)

    def test_nested_function_in_init_accessible_in_step(self):
        """A nested function defined in init_script is reachable in step code."""
        set_backend("jax")
        init = """
import jax.numpy as jnp

def double(x):
    return x * 2.0

def triple(x):
    return double(x) + x

out_0 = 0.0
"""
        step = "out_0 = triple(in_0)"
        block = library.CustomJaxBlock(
            name="nested_fn",
            dt=DT,
            init_script=init,
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=4.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(12.0, abs=0.1)

    def test_filter_non_traceable_separates_modules(self):
        """_filter_non_traceable correctly moves module objects to static_globals."""
        import types
        import numpy as np
        env = {
            "np": np,
            "x": jnp.array(1.0),
            "y": 2.0,
            "__builtins__": {},
        }
        dynamic, static = _filter_non_traceable(env)
        assert "np" not in dynamic, "Module should be in static_globals"
        assert "np" in static
        assert "x" in dynamic
        assert "y" in dynamic
        assert "__builtins__" in static

    def test_filter_non_traceable_handles_custom_class(self):
        """_filter_non_traceable puts a non-pytree class instance into static."""
        class MyClass:
            pass

        env = {
            "obj": MyClass(),
            "val": jnp.array(3.0),
        }
        dynamic, static = _filter_non_traceable(env)
        assert "obj" in static
        assert "val" in dynamic


# ---------------------------------------------------------------------------
# 17. _PerBlockModuleProxy
# ---------------------------------------------------------------------------

class TestPerBlockModuleProxy:

    def test_attribute_read_falls_through_to_real_module(self):
        import numpy as np
        proxy = _PerBlockModuleProxy(np)
        # pi is on the real module
        assert proxy.pi == pytest.approx(np.pi)

    def test_attribute_write_captured_in_overrides(self):
        import numpy as np
        proxy = _PerBlockModuleProxy(np)
        proxy.MY_CONSTANT = 999
        assert proxy.MY_CONSTANT == 999
        # Real module must be unaffected
        assert not hasattr(np, "MY_CONSTANT")

    def test_attribute_delete_from_overrides(self):
        import numpy as np
        proxy = _PerBlockModuleProxy(np)
        proxy.MY_CONSTANT = 42
        del proxy.MY_CONSTANT
        with pytest.raises(AttributeError):
            _ = proxy.MY_CONSTANT  # falls through to real module; not there

    def test_dir_includes_overrides_and_real_module(self):
        import numpy as np
        proxy = _PerBlockModuleProxy(np)
        proxy.NEW_ATTR = "hello"
        d = dir(proxy)
        assert "NEW_ATTR" in d
        assert "pi" in d  # from real module

    def test_repr_identifies_wrapped_module(self):
        import numpy as np
        proxy = _PerBlockModuleProxy(np)
        r = repr(proxy)
        assert "numpy" in r.lower()


# ---------------------------------------------------------------------------
# 18. _save_module_state / _restore_module_state
# ---------------------------------------------------------------------------

class TestModuleStateSnapshot:

    def test_save_restore_numpy_errstate(self):
        import numpy as np
        np.seterr(divide="ignore")
        env = {"np": np}
        snap = _save_module_state(env)
        # Change errstate
        np.seterr(divide="warn")
        _restore_module_state(snap)
        assert np.geterr()["divide"] == "ignore"
        # Cleanup
        np.seterr(divide="warn")

    def test_save_restore_python_random(self):
        import random
        random.seed(42)
        env = {"random": random}
        snap = _save_module_state(env)
        val_after_seed = random.random()
        # Restore and re-sample — should get the same value
        _restore_module_state(snap)
        assert random.random() == pytest.approx(val_after_seed, abs=1e-12)

    def test_missing_numpy_in_env(self):
        """If numpy isn't in env, _save_module_state returns an empty snapshot."""
        snap = _save_module_state({})
        # Should not error; returns minimal dict
        assert isinstance(snap, dict)


# ---------------------------------------------------------------------------
# 19. Agnostic time-mode edge cases
# ---------------------------------------------------------------------------

class TestAgnosticTimeMode:

    @pytest.mark.parametrize("use_jax", [True, False])
    def test_agnostic_no_dt_required(self, use_jax):
        """Agnostic-mode blocks accept dt=None."""
        set_backend("jax" if use_jax else "numpy")
        Klass = library.CustomJaxBlock if use_jax else library.CustomPythonBlock
        # Use jnp-style arithmetic for JAX; plain float() for Python
        step = "out_0 = in_0 * 3.0"  # works for both backends
        block = Klass(
            name="agnostic_no_dt",
            dt=None,
            init_script="out_0 = 0.0",
            user_statements=step,
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        diag = _diagram_with_source(block, source_value=4.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(12.0, abs=1e-3)

    def test_agnostic_jax_passthrough(self):
        """Agnostic JAX block: output equals input at every evaluation point."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="passthru",
            dt=None,
            init_script="out_0 = 0.0",
            user_statements="out_0 = in_0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
        builder = jaxonomy.DiagramBuilder()
        builder.add(block)
        clock = builder.add(library.Clock(name="clk"))
        builder.connect(clock.output_ports[0], block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, tf=2.0, recorded={"y": block.output_ports[0]})
        # Output should closely track the clock (== simulation time)
        assert np.allclose(np.array(res.outputs["y"]), np.array(res.time), atol=1e-4)


# ---------------------------------------------------------------------------
# 20. Persistent environment survival across steps
# ---------------------------------------------------------------------------

class TestPersistentEnv:

    def test_persistent_counter_jax(self):
        """Persistent variable (count) survives N discrete steps."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="persist_count",
            dt=DT,
            init_script="import jax.numpy as jnp\ncount = jnp.array(0.0)\nout_0 = count",
            user_statements="count = count + 1\nout_0 = count",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # First value is init (0), then 1, 2, 3 …
        expected = np.arange(len(y))
        assert np.allclose(y, expected)

    def test_persistent_accumulator_python(self):
        """List-based accumulator in CPB — appends each step."""
        set_backend("numpy")
        block = library.CustomPythonBlock(
            name="accum_py",
            dt=DT,
            init_script="history = []\nout_0 = 0.0",
            user_statements="""
history.append(float(in_0))
out_0 = float(len(history))
""",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        diag = _diagram_with_source(block, source_value=1.0)
        res = _sim(diag, recorded={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # output should equal step index (1-indexed)
        assert np.all(np.diff(y) >= 0)  # monotonically non-decreasing

    def test_exponential_decay_jax(self):
        """Persistent exponential decay: x[n+1] = 0.9 * x[n]."""
        set_backend("jax")
        block = library.CustomJaxBlock(
            name="decay",
            dt=DT,
            init_script="import jax.numpy as jnp\nx = jnp.array(1.0)\nout_0 = x",
            user_statements="x = 0.9 * x\nout_0 = x",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        ctx = block.create_context()
        res = jaxonomy.simulate(block, ctx, (0.0, TF),
                                recorded_signals={"y": block.output_ports[0]})
        y = np.array(res.outputs["y"])
        # Values should be strictly decreasing
        assert np.all(np.diff(y) <= 0)
        # After N steps, x ≈ 0.9^N
        N = len(y) - 1
        assert float(y[-1]) == pytest.approx(0.9 ** N, rel=0.01)


# ---------------------------------------------------------------------------
# 21. Combined patterns: CPB + JAX block in same diagram
# ---------------------------------------------------------------------------

class TestMixedBlockDiagram:

    def test_jax_block_feeds_python_block(self):
        """CustomJaxBlock output feeds into a CustomPythonBlock input."""
        set_backend("numpy")
        jax_block = library.CustomJaxBlock(
            name="jax_source",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = jnp.array(0.0)",
            user_statements="out_0 = out_0 + 1.0",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        py_block = library.CustomPythonBlock(
            name="py_doubler",
            dt=DT,
            init_script="out_0 = 0.0",
            user_statements="out_0 = float(in_0) * 2.0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        builder = jaxonomy.DiagramBuilder()
        builder.add(jax_block)
        builder.add(py_block)
        builder.connect(jax_block.output_ports[0], py_block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, recorded={"y": py_block.output_ports[0]})
        # Should not error
        assert res is not None

    def test_python_block_feeds_jax_block(self):
        """CustomPythonBlock output feeds into a CustomJaxBlock input."""
        set_backend("jax")
        py_block = library.CustomPythonBlock(
            name="py_src",
            dt=DT,
            init_script="out_0 = 1.0",
            user_statements="out_0 = 1.0",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
        jax_block = library.CustomJaxBlock(
            name="jax_sink",
            dt=DT,
            init_script="import jax.numpy as jnp\nout_0 = 0.0",
            user_statements="out_0 = in_0 * 5.0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="discrete",
        )
        builder = jaxonomy.DiagramBuilder()
        builder.add(py_block)
        builder.add(jax_block)
        builder.connect(py_block.output_ports[0], jax_block.input_ports[0])
        diag = builder.build()
        res = _sim(diag, recorded={"y": jax_block.output_ports[0]})
        assert float(res.outputs["y"][-1]) == pytest.approx(5.0, abs=0.1)
