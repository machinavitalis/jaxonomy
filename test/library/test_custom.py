# SPDX-License-Identifier: MIT

import logging
import pytest

try:
    from jaxlib.xla_extension import XlaRuntimeError
except ImportError:
    from jax.errors import JaxRuntimeError as XlaRuntimeError

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxonomy
from jaxonomy import library
from jaxonomy.library.custom import PythonScriptError
from jaxonomy.backend import numpy_api, set_backend
from jaxonomy.testing.markers import requires_jax

# from jaxonomy import logging
# logging.set_file_handler("test.log")

sin_init_code_np = "import numpy as np"
sin_step_code_np = "out_0 = np.sin(10 * in_0)"

sin_init_code_jnp = "import jax.numpy as jnp"
sin_step_code_jnp = "out_0 = jnp.sin(10 * in_0)"

relay_init_code_implicit = """
import jax.numpy as jnp
state = 0.0
"""

relay_init_code_explicit = """
import jax.numpy as jnp
state = 0.0
out_0 = state
"""

relay_step_code = """
state = jnp.where(in_0 > 0.5, 1.0, state)
state = jnp.where(in_0 < -0.5, 0.0, state)
out_0 = state
"""

# Legacy code for PythonScript relay definition
untraceable_step_code = """
if in_0 > 0.5:
    state = 1.0
elif in_0 < -0.5:
    state = 0.0
out_0 = state
"""

xfail_init_code = """
import jax.numpy as jnp
x = jnp.zeros(4)
out_0 = 0.0
"""

xfail_step_code = """
x[0] = 1.0
out_0 = x[0]
"""

counter_init_code_implicit = "count = 0.0"

counter_init_code_explicit = """
count = 0.0
out_0 = count
"""

counter_step_code = """
count = count + 1
out_0 = count
"""


def _test_relay(diagram, tf, dt):
    context = diagram.create_context()

    recorded_signals = {
        "sin.out_0": diagram["sin_psb"].output_ports[0],
        "relay.out_0": diagram["relay_psb"].output_ports[0],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )

    sin_sim = results.outputs["sin.out_0"]
    sin_ex = jnp.sin(10 * results.time)
    assert jnp.allclose(sin_sim, sin_ex)

    relay_sim = results.outputs["relay.out_0"]
    relay_ex = np.zeros(np.shape(relay_sim))
    for idx in range(sin_ex.size):
        if sin_ex[idx] > 0.5:
            relay_ex[idx] = 1.0
        elif sin_ex[idx] < -0.5:
            relay_ex[idx] = 0.0
        elif idx > 0:
            relay_ex[idx] = relay_ex[idx - 1]
    print("time")
    time = results.time
    print(f"{time=}")
    print("sin block")
    print(f"{sin_ex=}")
    print(f"{sin_sim=}")

    print("relay block")
    print(f"{relay_ex=}")
    print(f"{relay_sim=}")

    assert jnp.allclose(relay_sim, relay_ex)
    return results.time, results.outputs


def _make_jax_relay_diagram(dt, reverse_psb_order=False, implicit=False):
    ramp = library.Ramp(start_time=0.0, name="ramp")

    sin_psb = library.CustomJaxBlock(
        name="sin_psb",
        init_script=sin_init_code_jnp,
        user_statements=sin_step_code_jnp,
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    if implicit:
        # Test the case where the output value is not explicitly initialized
        # in the init_script.  This should produce a PythonScriptError
        relay_init_code = relay_init_code_implicit
    else:
        # Test the case where the output value is explicitly initialized
        # in the init_script. This should work properly.
        relay_init_code = relay_init_code_explicit

    relay_psb = library.CustomJaxBlock(
        name="relay_psb",
        dt=dt,
        time_mode="discrete",
        init_script=relay_init_code,
        user_statements=relay_step_code,
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder = jaxonomy.DiagramBuilder()
    builder.add(ramp)

    if reverse_psb_order:
        builder.add(relay_psb)
        builder.add(sin_psb)

    else:
        builder.add(sin_psb)
        builder.add(relay_psb)

    builder.connect(ramp.output_ports[0], sin_psb.input_ports[0])
    builder.connect(sin_psb.output_ports[0], relay_psb.input_ports[0])
    return builder.build()


@requires_jax()
@pytest.mark.minimal
def test_custom_relay_traceable():
    set_backend("jax")
    dt = 0.1

    diagram = _make_jax_relay_diagram(dt=dt, reverse_psb_order=False)
    t1, sol1 = _test_relay(diagram, tf=2.0, dt=dt)

    # Test adding the blocks in the reverse order
    # see https://jaxonomy.atlassian.net/browse/WC-66 for a description
    # of the bug this tests

    diagram = _make_jax_relay_diagram(dt=dt, reverse_psb_order=True)
    t2, sol2 = _test_relay(diagram, tf=2.0, dt=dt)

    assert jnp.allclose(t1, t2)
    assert jnp.allclose(sol1["sin.out_0"], sol2["sin.out_0"])
    assert jnp.allclose(sol1["relay.out_0"], sol2["relay.out_0"])

    # Check that the implicit initialization fails
    with pytest.raises(PythonScriptError):
        diagram = _make_jax_relay_diagram(dt=dt, implicit=True)


@pytest.mark.minimal
@pytest.mark.parametrize("use_jax", [True, False])
def test_custom_agnostic(use_jax):
    # Connecting a discrete clock to a discrete block will lead to a "data
    # flow" delay of one step in the output of the block. This will not
    # be the case for a block in "agnostic" time mode.
    dt = 0.1

    backend = "jax" if use_jax else "numpy"
    set_backend(backend)
    CustomBlock = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    agnostic_psb = CustomBlock(
        name="agnostic_psb",
        init_script="x = 2.0",
        user_statements="out_0 = x * in_0",
        time_mode="agnostic",
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder = jaxonomy.DiagramBuilder()
    builder.add(agnostic_psb)
    source = builder.add(library.DiscreteClock(dt=dt))
    builder.connect(source.output_ports[0], agnostic_psb.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()
    assert not diagram.has_ode_side_effects

    recorded_signals = {
        "psb.out_0": agnostic_psb.output_ports[0],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 2.0),
        recorded_signals=recorded_signals,
    )
    assert jnp.allclose(results.outputs["psb.out_0"], 2.0 * results.time)


@pytest.mark.minimal
def test_custom_fail_compile():
    dt = 0.1
    block = library.CustomJaxBlock(
        name="xfail",
        dt=dt,
        time_mode="discrete",
        init_script=xfail_init_code,
        user_statements=xfail_step_code,
        inputs=[],
        outputs=["out_0"],
    )

    with pytest.raises(PythonScriptError):
        ctx = block.create_context()
        block.check_types(ctx)


@pytest.mark.minimal
@pytest.mark.parametrize("use_jax", [True, False])
@pytest.mark.parametrize("ui_id", [None, "e7465c47-15ef-4c0a-8bab-e0446b22f98d"])
@pytest.mark.parametrize("time_mode", ["discrete", "agnostic"])
def test_custom_fail_init(caplog, use_jax: bool, ui_id: str | None, time_mode: str):
    # This check validates that errors happening inside a PythonScript block's
    # init_script are properly bubbled up for both notebooks and the UI.
    # For the UI, we want to explicitly print the original exception with a clean
    # backtrace in the logs.

    caplog.set_level(logging.ERROR)

    init_code = """
def fun():
    def crash():
        raise ValueError("This is a crash")
    crash()
fun()
"""

    dt = 0.1 if time_mode == "discrete" else None
    klass = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    with pytest.raises(PythonScriptError) as e:
        block = klass(
            name="BlockThatWillFailAtInit",
            dt=dt,
            time_mode=time_mode,
            init_script=init_code,
            user_statements="raise RuntimeError('Invalid error occurred')",
            inputs=[],
            outputs=["out_0"],
            ui_id=ui_id,
        )
        ctx = block.create_context()
        block.check_types(ctx)

    assert "This is a crash" in str(e.value)
    assert "BlockThatWillFailAtInit" in str(e.value)

    if ui_id is not None:
        assert "ValueError: This is a crash" in caplog.text
        assert (
            """  File "<init>", line 5, in fun
  File "<init>", line 4, in crash"""
            in caplog.text
        )
        # assert "BlockThatWillFailAtInit" in caplog.text


@pytest.mark.minimal
@pytest.mark.parametrize("use_jax", [True, False])
@pytest.mark.parametrize("ui_id", [None, "e7465c47-15ef-4c0a-8bab-e0446b22f98d"])
@pytest.mark.parametrize("time_mode", ["discrete", "agnostic"])
def test_custom_fail_step(caplog, use_jax: bool, ui_id: str | None, time_mode: str):
    # This check validates that errors happening inside a PythonScript block's
    # user_statements are properly bubbled up for both notebooks and the UI.
    # For the UI, we want to explicitly print the original exception with a clean
    # backtrace in the logs.

    caplog.set_level(logging.ERROR)

    init_code = """
def fun():
    def crash():
        raise ValueError("This is a crash")
    crash()

out_0 = 0.0
"""

    step_code = "fun()"

    dt = 0.1 if time_mode == "discrete" else None
    klass = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    with pytest.raises((PythonScriptError, XlaRuntimeError)) as e:
        block = klass(
            name="BlockThatWillFailAtStep",
            dt=dt,
            time_mode=time_mode,
            init_script=init_code,
            user_statements=step_code,
            inputs=[],
            outputs=["out_0"],
            ui_id=ui_id,
        )
        ctx = block.create_context()
        jaxonomy.simulate(
            block, ctx, (0.0, 1.0), recorded_signals={"x": block.output_ports[0]}
        )

    assert "This is a crash" in str(e.value)
    assert "BlockThatWillFailAtStep" in str(e.value)

    if ui_id is not None:
        assert "ValueError: This is a crash" in caplog.text
        assert (
            """  File "<step>", line 1, in <module>
  File "<init>", line 5, in fun
  File "<init>", line 4, in crash"""
            in caplog.text
        )
        # assert "BlockThatWillFailAtStep" in caplog.text


def _make_python_relay_diagram(dt, reverse_psb_order=False, implicit=False):
    ramp = library.Ramp(start_time=0.0, name="ramp")

    sin_psb = library.CustomPythonBlock(
        name="sin_psb",
        dt=dt,
        init_script=sin_init_code_np,
        user_statements=sin_step_code_np,
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    if implicit:
        # Test the case where the output value is not explicitly initialized
        # in the init_script.  This should produce a PythonScriptError
        relay_init_code = relay_init_code_implicit
    else:
        # Test the case where the output value is explicitly initialized
        # in the init_script. This should work properly.
        relay_init_code = relay_init_code_explicit

    relay_psb = library.CustomPythonBlock(
        name="relay_psb",
        dt=dt,
        time_mode="discrete",
        init_script=relay_init_code,
        user_statements=untraceable_step_code,
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder = jaxonomy.DiagramBuilder()

    builder.add(ramp)

    if reverse_psb_order:
        builder.add(relay_psb)
        builder.add(sin_psb)

    else:
        builder.add(sin_psb)
        builder.add(relay_psb)

    builder.connect(ramp.output_ports[0], sin_psb.input_ports[0])
    builder.connect(sin_psb.output_ports[0], relay_psb.input_ports[0])

    return builder.build()


# Repeat using untraceable Python (standard control flow)
@pytest.mark.slow
def test_custom_relay_untraceable():
    set_backend("numpy")
    dt = 0.1

    diagram = _make_python_relay_diagram(dt=dt, reverse_psb_order=False)
    diagram.create_context()
    assert not diagram.has_ode_side_effects
    t1, sol1 = _test_relay(diagram, tf=2.0, dt=dt)

    # Test adding the blocks in the reverse order
    # see https://jaxonomy.atlassian.net/browse/WC-66 for a description
    # of the bug this tests
    diagram = _make_python_relay_diagram(dt=dt, reverse_psb_order=True)
    diagram.create_context()
    assert not diagram.has_ode_side_effects
    t2, sol2 = _test_relay(diagram, tf=2.0, dt=dt)

    assert jnp.allclose(t1, t2)
    assert jnp.allclose(sol1["sin.out_0"], sol2["sin.out_0"])
    assert jnp.allclose(sol1["relay.out_0"], sol2["relay.out_0"])

    # Check that the implicit initialization fails
    with pytest.raises(PythonScriptError):
        diagram = _make_python_relay_diagram(dt=dt, implicit=True)


def _make_jax_counter_diagram(dt, implicit=False):
    if implicit:
        # Test the case where the output value is not explicitly initialized
        # in the init_script.  This should produce a PythonScriptError
        counter_init_code = counter_init_code_implicit
    else:
        # Test the case where the output value is explicitly initialized
        # in the init_script. This should work properly.
        counter_init_code = counter_init_code_explicit

    counter_psb = library.CustomJaxBlock(
        name="counter_psb",
        dt=dt,
        init_script=counter_init_code,
        user_statements=counter_step_code,
        outputs=["out_0"],
        time_mode="discrete",
    )

    gain_by_2 = library.CustomJaxBlock(
        name="gain_by_2",
        dt=dt,
        user_statements="out_0 = in_0 * 2",
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    builder = jaxonomy.DiagramBuilder()
    builder.add(counter_psb)
    builder.add(gain_by_2)

    builder.connect(counter_psb.output_ports[0], gain_by_2.input_ports[0])
    return builder.build()


def _test_counter(diagram, tf, show_plot=False):
    context = diagram.create_context()

    recorded_signals = {
        "counter": diagram["counter_psb"].output_ports[0],
        "gain_by_2": diagram["gain_by_2"].output_ports[0],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )

    time = results.time
    counter_res = results.outputs["counter"]
    gain_res = results.outputs["gain_by_2"]

    print(f"time=\n{time}")
    print(f"counter_res=\n{counter_res}")
    print(f"counter_sol=\n{np.arange(len(counter_res))}")
    print(f"gain_res=\n{gain_res}")

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))

        ax1.plot(time, counter_res, label="counter", marker="x")
        ax1.plot(time, gain_res, label="gain_by_2", marker="o")
        ax1.grid(True)
        ax1.legend()

        plt.show()

    assert np.allclose(counter_res, np.arange(len(counter_res)))
    assert np.allclose(gain_res, 2 * counter_res)


@requires_jax()
@pytest.mark.minimal
def test_custom_counter(show_plot=False):
    set_backend("jax")

    dt = 0.1

    diagram = _make_jax_counter_diagram(dt)
    _test_counter(diagram, tf=2.0, show_plot=show_plot)

    # Check that the implicit initialization fails
    with pytest.raises(PythonScriptError):
        diagram = _make_jax_counter_diagram(dt, implicit=True)


user_statements = """
out_0 = in_0 * 3.0
out_1 = in_1 * 6.0
"""

init_script = """
import jax.numpy as jnp
out_0 = 0.0
out_1 = jnp.zeros(2)
"""


@pytest.mark.minimal
def test_custom_output_shape(show_plot=False):
    set_backend("jax")
    dt = 0.1

    sclr = library.Constant(value=1.0)
    vec = library.Constant(value=jnp.ones(2))

    my_blk = library.CustomJaxBlock(
        name="my_blk",
        dt=dt,
        init_script=init_script,
        user_statements=user_statements,
        inputs=["in_0", "in_1"],
        outputs=["out_0", "out_1"],
    )

    builder = jaxonomy.DiagramBuilder()
    builder.add(sclr)
    builder.add(vec)
    builder.add(my_blk)

    builder.connect(sclr.output_ports[0], my_blk.input_ports[0])
    builder.connect(vec.output_ports[0], my_blk.input_ports[1])
    diagram = builder.build()

    context = diagram.create_context()

    recorded_signals = {
        "vec": vec.output_ports[0],
        "my_blk.out_0": my_blk.output_ports[0],
        "my_blk.out_1": my_blk.output_ports[1],
    }
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 0.3),
        recorded_signals=recorded_signals,
    )

    my_blk_out_1 = results.outputs["my_blk.out_1"]
    vec_out = results.outputs["vec"]
    out_1_sol = vec_out * 6.0

    print(f"vec_out={vec_out}\n")
    print(f"my_blk_out_1={my_blk_out_1}\n")
    print(f"out_1_sol={out_1_sol}\n")

    assert np.allclose(my_blk_out_1[1:, :], out_1_sol[1:, :])


@pytest.mark.parametrize("use_jax", [True, False])
def test_wc159(use_jax):
    # Tests a bug where initializing the values in discrete mode fails if the outputs
    # are not scalars.  Also tests that outputs written as lists will get properly
    # type-converted.
    dt = 0.1
    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    out_0_init = [3, 4]
    out_0_step = [5, 6]
    system = PythonScript(
        name="my_blk",
        dt=dt,
        time_mode="discrete",
        user_statements=f"out_0 = {out_0_step}",
        init_script=f"out_0 = {out_0_init}",
        inputs=[],
        outputs=["out_0"],
    )

    context = system.create_context()
    recorded_signals = {
        "system.out_0": system.output_ports[0],
    }
    results = jaxonomy.simulate(
        system,
        context,
        (0.0, 0.3),
        recorded_signals=recorded_signals,
    )

    out_0 = results.outputs["system.out_0"]

    y_init = np.array(out_0_init)
    y_step = np.array(out_0_step)

    assert np.allclose(out_0[0], y_init)
    assert np.allclose(out_0[1:], y_step)
    # Make sure we get a int64 on all platforms
    # assert out_0.dtype == int  # on windows we could get int64 != int
    assert np.iinfo(out_0.dtype).bits == 64


def test_wc230():
    # Test a bug where PyTree continuous state did not get properly unraveled
    # when passing `result_shape_dtype` to the jax pure callback. To trigger
    # this, need to use an agnostic-mode PythonScript block fed to an Integrator,
    # with another continuous state elsewhere in the system.
    # See https://jaxonomy.atlassian.net/browse/WC-230 for details.
    builder = jaxonomy.DiagramBuilder()

    Clock_0 = builder.add(library.Clock(name="Clock_0"))
    PythonScript_0 = builder.add(
        library.CustomPythonBlock(
            name="PythonScript_0",
            init_script="",
            user_statements="out_0 = in_0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
    )
    Integrator_0 = builder.add(library.Integrator(0.0, name="Integrator_0"))
    Integrator_1 = builder.add(library.Integrator(0.0, name="Integrator_1"))

    builder.connect(Clock_0.output_ports[0], PythonScript_0.input_ports[0])
    builder.connect(PythonScript_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Clock_0.output_ports[0], Integrator_1.input_ports[0])

    system = builder.build()
    context = system.create_context()

    recorded_signals = {
        "Integrator_0.out_0": Integrator_0.output_ports[0],
        "Integrator_1.out_0": Integrator_1.output_ports[0],
        "PythonScript_0.out_0": PythonScript_0.output_ports[0],
    }

    options = jaxonomy.SimulatorOptions(enable_tracing=True)

    tf = 10.0
    results = jaxonomy.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
        options=options,
    )

    assert results.time[-1] == tf
    assert jnp.allclose(results.outputs["PythonScript_0.out_0"], results.time)
    assert jnp.allclose(results.outputs["Integrator_0.out_0"], 0.5 * results.time**2)
    assert jnp.allclose(results.outputs["Integrator_1.out_0"], 0.5 * results.time**2)


def test_wc235():
    # Test a bug where the type inference calls were not sorted according to callback
    # ordering.  See https://jaxonomy.atlassian.net/browse/WC-235

    builder = jaxonomy.DiagramBuilder()

    # First add the downstream block (Agnostic mode)
    psb_0 = builder.add(
        library.CustomPythonBlock(
            name="psb_0",
            init_script="",
            user_statements="out_0 = in_0 * 2",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
        )
    )

    # Then add the upstream block (Discrete mode)
    psb_1 = builder.add(
        library.CustomPythonBlock(
            name="psb_1",
            dt=0.1,
            init_script="out_0 = 0.0",
            user_statements="out_0 = out_0 + 1",
            inputs=[],
            outputs=["out_0"],
            time_mode="discrete",
        )
    )

    builder.connect(psb_1.output_ports[0], psb_0.input_ports[0])

    system = builder.build()

    # Previously this raised an error because the upstream discrete block did not
    # do type inference before the downstream agnostic block.
    system.create_context()


@pytest.mark.parametrize("use_jax", [True, False])
def test_custom_reuse_outport_state(use_jax):
    # Tests initializing an output port value and then reusing the output
    # port name in the step code.
    dt = 0.1
    backend = "jax" if use_jax else "numpy"
    set_backend(backend)

    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    system = PythonScript(
        name="my_blk",
        dt=dt,
        time_mode="discrete",
        user_statements="out_0 = out_0 + 1",
        init_script="out_0 = 0",
        inputs=[],
        outputs=["out_0"],
    )

    context = system.create_context()
    assert not system.has_ode_side_effects

    recorded_signals = {
        "system.out_0": system.output_ports[0],
    }
    results = jaxonomy.simulate(
        system,
        context,
        (0.0, 10.0),
        recorded_signals=recorded_signals,
    )

    out_0 = results.outputs["system.out_0"]
    assert np.allclose(out_0, np.arange(101))


function_def_init = """
def my_func(x):
    return x * 2
"""

function_def_step = "out_0 = my_func(in_0)"


@pytest.mark.parametrize("use_jax", [True, False])
def test_custom_function_def(use_jax):
    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock
    builder = jaxonomy.DiagramBuilder()
    clock = library.Clock(name="clock")
    psb = PythonScript(
        name="my_blk",
        time_mode="agnostic",
        init_script=function_def_init,
        user_statements=function_def_step,
        inputs=["in_0"],
        outputs=["out_0"],
    )

    builder.add(clock, psb)
    builder.connect(clock.output_ports[0], psb.input_ports[0])

    system = builder.build()
    context = system.create_context()

    t = 1.5
    context = context.with_time(t)

    psb_out = psb.output_ports[0].eval(context)
    assert psb_out == 2 * t


van_der_pol_init = """
import numpy as np
mu = 1.0
"""

van_der_pol_step = """
x, y = in_0
x_dot = y
y_dot = mu * (1 - x**2) * y - x
out_0 = np.array([x_dot, y_dot])
"""


@pytest.fixture
def van_der_pol():
    builder = jaxonomy.DiagramBuilder()

    rhs_fun = library.CustomPythonBlock(
        name="rhs",
        init_script=van_der_pol_init,
        user_statements=van_der_pol_step,
        inputs=["in_0"],
        outputs=["out_0"],
        time_mode="agnostic",
    )

    integrator = library.Integrator(
        name="integrator",
        initial_state=[0.0, 1.0],
    )

    builder.add(rhs_fun, integrator)
    builder.connect(rhs_fun.output_ports[0], integrator.input_ports[0])
    builder.connect(integrator.output_ports[0], rhs_fun.input_ports[0])

    builder.export_output(integrator.output_ports[0])

    return builder.build(name="van_der_pol")


class TestFeedthroughSideEffects:
    def test_scipy_fallback(self, van_der_pol):
        # See WC-209 bug report
        # Test that an untraceable feedthrough block can be used as the RHS of an ODE
        integrator = van_der_pol["integrator"]
        context = van_der_pol.create_context()
        assert van_der_pol.has_ode_side_effects

        recorded_signals = {
            "integrator.out_0": integrator.output_ports[0],
        }
        tf = 10.0
        result = jaxonomy.simulate(
            van_der_pol,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        # Check that the sim made it to the final time
        assert result.time[-1] == tf

    def test_with_zc(self, van_der_pol):
        # If we have both a zero-crossing and ODE side effects, the
        # solver should still work by falling back to untraced execution.

        builder = jaxonomy.DiagramBuilder()
        builder.add(van_der_pol)

        demux = builder.add(library.Demultiplexer(2, name="demux"))

        # Add something with a zero-crossing event
        relay = builder.add(
            library.Relay(
                on_threshold=0.5,
                off_threshold=-0.5,
                on_value=1.0,
                off_value=0.0,
                initial_state=0.0,
                name="relay",
            )
        )

        builder.connect(van_der_pol.output_ports[0], demux.input_ports[0])
        builder.connect(demux.output_ports[0], relay.input_ports[0])

        system = builder.build()
        context = system.create_context()

        assert system.has_ode_side_effects
        assert system.zero_crossing_events.has_events

        recorded_signals = {
            "integrator.out_0": van_der_pol.output_ports[0],
        }
        tf = 10.0
        result = jaxonomy.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        assert result.time[-1] == tf


if __name__ == "__main__":
    # test_custom_counter(show_plot=True)
    # test_custom_relay_traceable()
    test_custom_output_shape()


# ============================================================================
# Environment isolation tests for CustomPythonBlock
# ============================================================================

class TestCustomPythonBlockIsolation:
    """Verify that multiple CustomPythonBlock instances do not contaminate each
    other's numpy error state or module-level mutable globals."""

    def _make_cpb(self, name, init_code, step_code, outputs):
        return library.CustomPythonBlock(
            dt=0.1,
            init_script=init_code,
            user_statements=step_code,
            inputs=[],
            outputs=outputs,
            name=name,
        )

    def _simulate_two_blocks(self, cpb_a, cpb_b, t_span=(0.0, 0.3)):
        builder = jaxonomy.DiagramBuilder()
        builder.add(cpb_a)
        builder.add(cpb_b)
        diag = builder.build()
        ctx = diag.create_context()
        res = jaxonomy.simulate(
            diag,
            ctx,
            t_span,
            options=jaxonomy.SimulatorOptions(
                math_backend="jax", max_major_steps=10
            ),
            recorded_signals={
                "a": diag["a"].output_ports[0],
                "b": diag["b"].output_ports[0],
            },
        )
        return res

    def test_numpy_errstate_isolation(self):
        """Block A's np.seterr() must not contaminate block B's numeric policy."""
        cpb_a = self._make_cpb(
            "a",
            "import numpy as np\ny = 1.0",
            'np.seterr(divide="ignore"); y = 1.0',
            ["y"],
        )
        cpb_b = self._make_cpb(
            "b",
            "import numpy as np\ny = 1.0",
            'state = np.geterr(); y = 1.0 if state["divide"] == "warn" else 0.0',
            ["y"],
        )
        res = self._simulate_two_blocks(cpb_a, cpb_b)
        # If isolated: b.y == 1.0 (divide is still "warn" for block B)
        # If contaminated: b.y == 0.0
        final_b = float(res.outputs["b"][-1])
        assert final_b == pytest.approx(1.0), (
            f"Block B was contaminated by block A's np.seterr(). "
            f"Expected 1.0 (isolated), got {final_b}"
        )

    def test_numpy_errstate_independent_per_block(self):
        """Each block can maintain its own independent numpy error policy."""
        cpb_a = self._make_cpb(
            "a",
            "import numpy as np\ny = 0.0",
            # A sets divide=ignore and verifies it persists
            'np.seterr(divide="ignore"); state = np.geterr(); y = 1.0 if state["divide"] == "ignore" else 0.0',
            ["y"],
        )
        cpb_b = self._make_cpb(
            "b",
            "import numpy as np\ny = 0.0",
            # B keeps default (divide=warn) and verifies it
            'state = np.geterr(); y = 1.0 if state["divide"] == "warn" else 0.0',
            ["y"],
        )
        res = self._simulate_two_blocks(cpb_a, cpb_b)
        # Both should be 1.0: A maintains its "ignore" setting, B maintains "warn"
        assert float(res.outputs["a"][-1]) == pytest.approx(1.0), \
            "Block A did not maintain its own divide='ignore' setting"
        assert float(res.outputs["b"][-1]) == pytest.approx(1.0), \
            "Block B should see divide='warn' (not contaminated by A)"

    def test_global_numpy_errstate_restored_after_sim(self):
        """The global numpy error state is restored after simulation."""
        import numpy as np

        before = np.geterr()
        cpb_a = self._make_cpb(
            "a",
            "import numpy as np\ny = 1.0",
            'np.seterr(divide="ignore", over="ignore"); y = 1.0',
            ["y"],
        )
        cpb_b = self._make_cpb(
            "b",
            "import numpy as np\ny = 1.0",
            "y = 1.0",
            ["y"],
        )
        self._simulate_two_blocks(cpb_a, cpb_b)
        after = np.geterr()
        assert after == before, (
            f"Global numpy errstate leaked after simulation. "
            f"Before: {before}, After: {after}"
        )

    def test_numpy_random_state_isolation(self):
        """Block A seeding numpy.random must not advance block B's random stream."""
        import numpy as np

        # Block A: seeds numpy.random with 42 every step
        cpb_a = self._make_cpb(
            "a",
            "import numpy as np\ny = 0.0",
            "np.random.seed(42); y = np.random.rand()",
            ["y"],
        )
        # Block B: draws a random number WITHOUT seeding — should NOT be affected by A
        # We record 3 consecutive values to verify the stream is not reset each step
        # (if A's seed contaminated B, all draws in B would always be the same)
        cpb_b = self._make_cpb(
            "b",
            "import numpy as np\ny = 0.0",
            "y = np.random.rand()",
            ["y"],
        )

        builder = jaxonomy.DiagramBuilder()
        builder.add(cpb_a)
        builder.add(cpb_b)
        diag = builder.build()
        ctx = diag.create_context()
        res = jaxonomy.simulate(
            diag,
            ctx,
            (0.0, 0.5),
            options=jaxonomy.SimulatorOptions(
                math_backend="jax", max_major_steps=20
            ),
            recorded_signals={"b": diag["b"].output_ports[0]},
        )

        b_values = res.outputs["b"]
        # If A's seed contaminated B, consecutive b values would all be the same
        # (reset to the value right after seed(42)). They should vary.
        assert b_values.shape[0] > 1
        # Not all values are identical
        assert not all(
            abs(float(b_values[i]) - float(b_values[0])) < 1e-6
            for i in range(1, len(b_values))
        ), "Block B's random stream appears to be reset each step (contamination from A)"

    def test_module_proxy_attribute_isolation(self):
        """_PerBlockModuleProxy captures attribute writes per-instance."""
        from jaxonomy.library.custom import _PerBlockModuleProxy
        import types

        # Create a simple module with a mutable attribute
        mod = types.ModuleType("test_mod")
        mod.value = 10

        proxy_a = _PerBlockModuleProxy(mod)
        proxy_b = _PerBlockModuleProxy(mod)

        # Write to proxy_a
        proxy_a.value = 99

        # proxy_b and real module should be unaffected
        assert proxy_b.value == 10, "proxy_a write leaked to proxy_b"
        assert mod.value == 10, "proxy_a write mutated real module"

        # proxy_a sees its own write
        assert proxy_a.value == 99

    def test_module_proxy_read_fallthrough(self):
        """_PerBlockModuleProxy reads fall through to the real module."""
        from jaxonomy.library.custom import _PerBlockModuleProxy
        import types

        mod = types.ModuleType("test_mod")
        mod.foo = "bar"
        mod.num = 42

        proxy = _PerBlockModuleProxy(mod)
        assert proxy.foo == "bar"
        assert proxy.num == 42

    def test_save_restore_module_state(self):
        """_save_module_state / _restore_module_state correctly checkpoint numpy."""
        from jaxonomy.library.custom import _save_module_state, _restore_module_state
        import numpy as np

        original = np.geterr()
        env = {}  # save uses sys.modules["numpy"]

        snapshot = _save_module_state(env)
        assert "numpy_errstate" in snapshot

        # Mutate numpy state
        np.seterr(divide="ignore", over="ignore")
        assert np.geterr()["divide"] == "ignore"

        # Restore
        _restore_module_state(snapshot)
        restored = np.geterr()
        assert restored == original, f"Restore failed: {restored} != {original}"

    def test_single_cpb_errstate_persistence(self):
        """A single CPB's self-set errstate persists across time steps."""
        # Block sets divide='ignore' at t=0.1, reads it at t=0.2 — must still be 'ignore'
        cpb = library.CustomPythonBlock(
            dt=0.1,
            init_script="import numpy as np\nstep_count = 0\ny = 1.0",
            user_statements="""
step_count = step_count + 1
if step_count == 1:
    np.seterr(divide='ignore')
state = np.geterr()
# 1.0 if state correctly persists, 0.0 if reset each step
y = 1.0 if state['divide'] == 'ignore' else 0.0
""",
            inputs=[],
            outputs=["y"],
            name="a",
        )
        builder = jaxonomy.DiagramBuilder()
        builder.add(cpb)
        diag = builder.build()
        ctx = diag.create_context()
        res = jaxonomy.simulate(
            diag,
            ctx,
            (0.0, 0.5),
            options=jaxonomy.SimulatorOptions(
                math_backend="jax", max_major_steps=20
            ),
            recorded_signals={"y": diag["a"].output_ports[0]},
        )
        # After the first step sets divide='ignore', all subsequent steps should see it
        assert float(res.outputs["y"][-1]) == pytest.approx(1.0), \
            "CPB's own errstate was reset between steps"
