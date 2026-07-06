# SPDX-License-Identifier: MIT
"""
Phase B (T-001): gradient-correctness coverage for stateless standard-library
blocks (feedthrough / reduce / source). Each block is evaluated through its
pytree-exposed callback with JAX tracers and checked against FD.

Runs on every PR. See ``test_block_gradients_full.py`` for the simulation-level
sweep (nightly, marked ``autodiff_full``).

Non-differentiable blocks (Quantizer, LogicalOperator, Comparator, Stop, etc.)
are excluded here — their outputs are integer/boolean and ``jax.grad`` is not
meaningful. The intent of T-001 is correctness of *differentiable* gradients.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

import jaxonomy
from jaxonomy.library import (
    Abs,
    Adder,
    Arithmetic,
    Chirp,
    Clock,
    Comparator,
    Constant,
    CrossProduct,
    DeadZone,
    DotProduct,
    Exponent,
    Gain,
    IfThenElse,
    Logarithm,
    LogicalOperator,
    LookupTable1d,
    LookupTable2d,
    MatrixConcatenation,
    MatrixInversion,
    MatrixMultiplication,
    MatrixTransposition,
    MinMax,
    Multiplexer,
    Offset,
    Power,
    Product,
    ProductOfElements,
    Pulse,
    Quantizer,
    Ramp,
    Reciprocal,
    Saturate,
    Sawtooth,
    ScalarBroadcast,
    SignalDatatypeConversion,
    Sine,
    Slice,
    SquareRoot,
    Stack,
    Step,
    Stop,
    SumOfElements,
    Trigonometric,
)

from ._framework import assert_grad_matches_fd, sim_options
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# Reasonable hypothesis strategies shared across tests.  Kept narrow so FD is
# stable (no enormous dynamic range, no values near singularities).
_finite_float = st.floats(
    min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False
)
_positive_float = st.floats(
    min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False
)
_non_small_float = st.floats(min_value=0.5, max_value=4.0, allow_nan=False)
SETTINGS = settings(deadline=None, max_examples=15)


# ── helpers ──────────────────────────────────────────────────────────────────


def _block_output(block, input_values: list, param_overrides: dict | None = None):
    """Evaluate a stateless block given tracer input/param values.

    Uses ``InputPort.fix_value`` at construction time with a concrete *shape*
    placeholder, then re-fixes to the tracer inside the traced function via
    ``with_parameters`` on the block's internal params.  For simplicity we go
    through a small Constant→block diagram so input ports accept traced
    Constant values without hitting the "tracer leak" path documented on
    ``fix_value``.
    """
    bld = jaxonomy.DiagramBuilder()
    consts = []
    for iv in input_values:
        c = bld.add(Constant(jnp.asarray(iv)))
        consts.append(c)
    b = bld.add(block)
    for c, p in zip(consts, b.input_ports):
        bld.connect(c.output_ports[0], p)
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def eval_with(*traced_values_and_params):
        # traced_values_and_params = (*inputs, *params) — we pack them here
        n_in = len(input_values)
        traced_inputs = traced_values_and_params[:n_in]
        traced_params = traced_values_and_params[n_in:]

        ctx = ctx0
        for c, iv in zip(consts, traced_inputs):
            ctx = ctx.with_subcontext(
                c.system_id,
                ctx[c.system_id].with_parameter("value", iv),
            )
        if param_overrides:
            for (pname, _init), pval in zip(param_overrides.items(), traced_params):
                ctx = ctx.with_subcontext(
                    b.system_id,
                    ctx[b.system_id].with_parameter(pname, pval),
                )
        return b.output_ports[0].eval(ctx)

    return eval_with


def _loss(block, input_values, param_overrides=None, reducer=jnp.sum):
    """Return a scalar-loss forward function differentiable over inputs + params."""
    evaluator = _block_output(block, input_values, param_overrides)

    def fwd(*traced):
        y = evaluator(*traced)
        return reducer(y)

    return fwd


# ── tests: simple pointwise feedthrough ─────────────────────────────────────


@given(x=_finite_float)
@SETTINGS
def test_grad_abs(x):
    # |x| is non-differentiable at 0; hypothesis filter keeps |x| > 0.05.
    if abs(x) < 0.05:
        return
    fwd = _loss(Abs(), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Abs")


@given(x=_finite_float)
@SETTINGS
def test_grad_gain(x):
    fwd = _loss(Gain(2.5), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Gain")


@given(x=_finite_float)
@SETTINGS
def test_grad_offset(x):
    fwd = _loss(Offset(1.5), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Offset")


@given(x=_finite_float)
@SETTINGS
def test_grad_exponent(x):
    # Exponent(e^x) is defined everywhere.
    fwd = _loss(Exponent(base="exp"), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Exponent")


@given(x=_positive_float)
@SETTINGS
def test_grad_logarithm(x):
    fwd = _loss(Logarithm(base="natural"), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Logarithm")


@given(x=_positive_float)
@SETTINGS
def test_grad_sqrt(x):
    fwd = _loss(SquareRoot(), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="SquareRoot")


@given(x=st.floats(min_value=0.2, max_value=3.0))
@SETTINGS
def test_grad_reciprocal(x):
    fwd = _loss(Reciprocal(), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Reciprocal")


@given(x=_finite_float, exp=st.floats(min_value=1.0, max_value=3.0))
@SETTINGS
def test_grad_power(x, exp):
    # Require x > 0 for non-integer exponent differentiability.
    if x <= 0.05:
        return
    fwd = _loss(Power(exponent=exp), [x])
    assert_grad_matches_fd(fwd, jnp.asarray(x), solver=None, dtype="float64", block="Power")


@pytest.mark.parametrize("fn", ["sin", "cos", "tan", "sinh", "cosh", "tanh"])
@given(x=st.floats(min_value=-1.2, max_value=1.2))  # stay well away from tan singularities
@SETTINGS
def test_grad_trig(fn, x):
    fwd = _loss(Trigonometric(function=fn), [x])
    assert_grad_matches_fd(
        fwd, jnp.asarray(x), solver=None, dtype="float64", block=f"Trigonometric[{fn}]"
    )


# ── reduce-block arithmetic ──────────────────────────────────────────────────


@given(a=_finite_float, b=_finite_float, c=_finite_float)
@SETTINGS
def test_grad_adder_3in(a, b, c):
    fwd = _loss(Adder(3, operators="+-+"), [a, b, c])
    assert_grad_matches_fd(
        fwd, jnp.asarray(a), jnp.asarray(b), jnp.asarray(c),
        solver=None, dtype="float64", block="Adder(+-+)",
    )


@given(a=_finite_float, b=_non_small_float, c=_non_small_float)
@SETTINGS
def test_grad_arithmetic_mul_div(a, b, c):
    # operators="+*/" for 3 inputs → sign='+', then a * b / c.  b,c > 0.5 keeps
    # FD stable.
    fwd = _loss(Arithmetic(3, operators="+*/"), [a, b, c])
    assert_grad_matches_fd(
        fwd, jnp.asarray(a), jnp.asarray(b), jnp.asarray(c),
        solver=None, dtype="float64", block="Arithmetic(+*/)",
    )


@given(a=_finite_float, b=_finite_float)
@SETTINGS
def test_grad_product(a, b):
    fwd = _loss(Product(2), [a, b])
    assert_grad_matches_fd(
        fwd, jnp.asarray(a), jnp.asarray(b),
        solver=None, dtype="float64", block="Product",
    )


# ── vector / matrix blocks (fixed shapes, seeded) ────────────────────────────


def _seeded_vec(n, seed=0, lo=-1.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(lo, hi, size=n))


def _seeded_mat(shape, seed=0, lo=-1.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(lo, hi, size=shape))


def test_grad_crossproduct_3d():
    a = _seeded_vec(3, seed=1)
    b = _seeded_vec(3, seed=2)
    fwd = _loss(CrossProduct(), [a, b])
    assert_grad_matches_fd(
        fwd, a, b, solver=None, dtype="float64", block="CrossProduct",
    )


def test_grad_dotproduct():
    a = _seeded_vec(4, seed=3)
    b = _seeded_vec(4, seed=4)
    fwd = _loss(DotProduct(), [a, b])
    assert_grad_matches_fd(
        fwd, a, b, solver=None, dtype="float64", block="DotProduct",
    )


def test_grad_sum_of_elements():
    a = _seeded_vec(5, seed=5)
    fwd = _loss(SumOfElements(), [a])
    assert_grad_matches_fd(fwd, a, solver=None, dtype="float64", block="SumOfElements")


def test_grad_product_of_elements():
    # Keep values away from zero so log(|x|) derivative is well-conditioned.
    a = _seeded_vec(4, seed=6, lo=0.5, hi=1.5)
    fwd = _loss(ProductOfElements(), [a])
    assert_grad_matches_fd(fwd, a, solver=None, dtype="float64", block="ProductOfElements")


def test_grad_matmul_2x2():
    A = _seeded_mat((2, 3), seed=7)
    B = _seeded_mat((3, 2), seed=8)
    fwd = _loss(MatrixMultiplication(2), [A, B])
    assert_grad_matches_fd(
        fwd, A, B, solver=None, dtype="float64", block="MatrixMultiplication",
    )


def test_grad_matrix_inversion():
    # 2×2 with non-tiny determinant.
    rng = np.random.default_rng(9)
    A = jnp.asarray(rng.uniform(0.5, 1.5, size=(2, 2))) + 0.5 * jnp.eye(2)
    fwd = _loss(MatrixInversion(), [A])
    assert_grad_matches_fd(
        fwd, A, solver=None, dtype="float64", block="MatrixInversion",
    )


def test_grad_matrix_transposition():
    A = _seeded_mat((2, 3), seed=10)
    fwd = _loss(MatrixTransposition(), [A])
    assert_grad_matches_fd(fwd, A, solver=None, dtype="float64", block="MatrixTransposition")


def test_grad_matrix_concat_vertical():
    A = _seeded_mat((2, 2), seed=11)
    B = _seeded_mat((2, 2), seed=12)
    fwd = _loss(MatrixConcatenation(2, axis=0), [A, B])
    assert_grad_matches_fd(
        fwd, A, B, solver=None, dtype="float64", block="MatrixConcat",
    )


def test_grad_stack():
    a = _seeded_vec(3, seed=13)
    b = _seeded_vec(3, seed=14)
    fwd = _loss(Stack(2, axis=0), [a, b])
    assert_grad_matches_fd(fwd, a, b, solver=None, dtype="float64", block="Stack")


def test_grad_multiplexer():
    fwd = _loss(Multiplexer(3), [1.0, 2.0, 3.0])
    assert_grad_matches_fd(
        fwd, jnp.array(1.0), jnp.array(2.0), jnp.array(3.0),
        solver=None, dtype="float64", block="Multiplexer",
    )


# ── saturating / piecewise blocks ────────────────────────────────────────────


@given(x=st.floats(min_value=-2.5, max_value=2.5))
@SETTINGS
def test_grad_saturate_interior(x):
    # Inside the [−1, 1] band: gradient = 1.  FD near ±1 is noisy so exclude a
    # small boundary layer.
    if abs(abs(x) - 1.0) < 0.1:
        return
    fwd = _loss(Saturate(lower_limit=-1.0, upper_limit=1.0), [x])
    assert_grad_matches_fd(
        fwd, jnp.asarray(x), solver=None, dtype="float64", block="Saturate",
    )


@given(x=st.floats(min_value=-3.0, max_value=3.0))
@SETTINGS
def test_grad_deadzone_outside(x):
    # Outside the [−0.5, 0.5] dead band the gradient is 1.  Boundary excluded.
    if abs(x) < 0.65:
        return
    fwd = _loss(DeadZone(half_range=0.5), [x])
    assert_grad_matches_fd(
        fwd, jnp.asarray(x), solver=None, dtype="float64", block="DeadZone",
    )


def test_grad_minmax_smooth():
    # MinMax's ``max`` path is differentiable away from ties.  Pick values with
    # a clear winner so the argmax is stable.
    fwd = _loss(MinMax(2, operator="max"), [1.5, 0.5])
    assert_grad_matches_fd(
        fwd, jnp.array(1.5), jnp.array(0.5),
        solver=None, dtype="float64", block="MinMax(max)",
    )


# ── lookup tables ────────────────────────────────────────────────────────────


def test_grad_lookup1d_linear():
    table_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    table_y = np.array([0.0, 2.0, 1.0, 3.0, 0.5])
    blk = LookupTable1d(input_array=table_x, output_array=table_y, interpolation="linear")
    # Pick an interior point well away from breakpoints.
    fwd = _loss(blk, [1.5])
    assert_grad_matches_fd(
        fwd, jnp.array(1.5), solver=None, dtype="float64", block="LookupTable1d",
    )


# ── selection / routing ──────────────────────────────────────────────────────


def test_grad_slice():
    a = _seeded_vec(5, seed=20)
    fwd = _loss(Slice("[1:4]"), [a])
    assert_grad_matches_fd(fwd, a, solver=None, dtype="float64", block="Slice")


def test_grad_scalar_broadcast():
    # ``m=None, n=3`` produces a (3,) output; gradient of sum(y) w.r.t. scalar
    # input is 3.
    fwd = _loss(ScalarBroadcast(m=None, n=3), [2.5])
    assert_grad_matches_fd(
        fwd, jnp.array(2.5), solver=None, dtype="float64", block="ScalarBroadcast",
    )


# ── T-001b additions: LookupTable2d, IfThenElse, Saturate(dyn) ───────────────


def test_grad_lookup2d_linear():
    # 1-D x-coords, 1-D y-coords, 2-D table. Gradient w.r.t. an interior (x,y)
    # query point should match FD on the bilinear-interp surface.
    table_x = np.array([0.0, 1.0, 2.0, 3.0])
    table_y = np.array([0.0, 1.0, 2.0])
    # Output table is f(xi, yj) = xi + 0.5 * yj^2 (smooth, bilinear-stable).
    z = np.array([[xi + 0.5 * yj**2 for yj in table_y] for xi in table_x])
    blk = LookupTable2d(
        input_x_array=table_x,
        input_y_array=table_y,
        output_table_array=z,
        interpolation="linear",
    )
    fwd = _loss(blk, [1.5, 1.25])
    assert_grad_matches_fd(
        fwd, jnp.array(1.5), jnp.array(1.25),
        solver=None, dtype="float64", block="LookupTable2d",
    )


def test_grad_ifthenelse_true_branch():
    # pred is a fixed boolean; gradient of jnp.where(True, t, f) w.r.t. t is 1
    # and w.r.t. f is 0. Verify both branches match FD when the predicate is
    # statically true and statically false.
    blk = IfThenElse()
    fwd_true = _loss(blk, [True, 2.0, -1.0])
    assert_grad_matches_fd(
        fwd_true,
        jnp.array(True), jnp.array(2.0), jnp.array(-1.0),
        solver=None, dtype="float64", block="IfThenElse(true)",
        argnums=(1, 2),
    )

    blk2 = IfThenElse()
    fwd_false = _loss(blk2, [False, 2.0, -1.0])
    assert_grad_matches_fd(
        fwd_false,
        jnp.array(False), jnp.array(2.0), jnp.array(-1.0),
        solver=None, dtype="float64", block="IfThenElse(false)",
        argnums=(1, 2),
    )


def test_grad_saturate_dynamic_limits_interior():
    # Dynamic upper / lower limits via input ports. Interior x → grad w.r.t.
    # x is 1, w.r.t. limits is 0 (saturation does not bind).
    blk = Saturate(
        enable_dynamic_upper_limit=True,
        enable_dynamic_lower_limit=True,
    )
    fwd = _loss(blk, [0.3, 1.0, -1.0])  # x, ulim, llim
    assert_grad_matches_fd(
        fwd, jnp.array(0.3), jnp.array(1.0), jnp.array(-1.0),
        solver=None, dtype="float64", block="Saturate(dynamic-limits,interior)",
    )


def test_grad_saturate_dynamic_upper_active():
    # Saturate active on the upper limit: x=1.5 > ulim=1.0 → y=ulim. Gradient
    # of sum(y) w.r.t. x is 0, w.r.t. ulim is 1, w.r.t. llim is 0.
    blk = Saturate(
        enable_dynamic_upper_limit=True,
        enable_dynamic_lower_limit=True,
    )
    fwd = _loss(blk, [1.5, 1.0, -1.0])
    assert_grad_matches_fd(
        fwd, jnp.array(1.5), jnp.array(1.0), jnp.array(-1.0),
        solver=None, dtype="float64", block="Saturate(dynamic-limits,upper-active)",
    )


# ── T-001b additions: Source-block dynamic-parameter gradients ──────────────
#
# Source blocks have no inputs; their gradient is taken w.r.t. their dynamic
# parameters by running a 1-step simulation and reading the output. This
# exercises the same trace/ad path used in production.


def _source_grad_via_eval(block, t_eval, param_setters, param_values, block_name):
    """Differentiate ``block.output(t_eval)`` w.r.t. its dynamic parameters.

    Source blocks have no inputs and no state; the output is a pure function
    of (time, params). Bypass the simulator and directly call
    ``output_ports[0].eval(ctx.with_time(t_eval))`` so the test surfaces
    *block* gradients, not solver gradients (those are exercised in
    test_block_gradients_full.py).

    ``param_setters`` is a tuple of parameter-name strings used with
    ``with_parameter``; ``param_values`` is the matching tuple of values to
    differentiate at.
    """
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(block)
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def fwd(*pvals):
        c = ctx0
        sub = c[src.system_id]
        for name, v in zip(param_setters, pvals):
            sub = sub.with_parameter(name, v)
        c = c.with_subcontext(src.system_id, sub).with_time(t_eval)
        return jnp.sum(jnp.asarray(src.output_ports[0].eval(c)))

    assert_grad_matches_fd(
        fwd, *(jnp.asarray(v) for v in param_values),
        solver=None, dtype="float64", block=block_name,
    )


def test_grad_source_sine_amplitude_frequency():
    # y = a * sin(f * t + phi) + b → grad through simulate.
    blk = Sine(amplitude=1.0, frequency=2.0, phase=0.1, bias=0.0)
    _source_grad_via_eval(
        blk, t_eval=0.4,
        param_setters=("amplitude", "frequency"),
        param_values=(1.2, 2.3),
        block_name="Sine(amp,freq)",
    )


def test_grad_source_chirp():
    # Chirp has dynamic f0, f1, stop_time, phi. Use a sub-stop-time evaluation
    # so the time integral stays in the linear-chirp regime.
    blk = Chirp(f0=0.5, f1=2.0, stop_time=1.0, phi=0.1)
    _source_grad_via_eval(
        blk, t_eval=0.3,
        param_setters=("f0", "f1", "phi"),
        param_values=(0.7, 1.5, 0.2),
        block_name="Chirp(f0,f1,phi)",
    )


def test_grad_source_ramp():
    # y(t) = m*(t - t0) + y0 for t >= t0. Choose t_eval well past t0 so we are
    # firmly inside the linear region (FD over the kink would corrupt the test).
    blk = Ramp(start_value=0.5, slope=1.5, start_time=0.0)
    _source_grad_via_eval(
        blk, t_eval=0.4,
        param_setters=("start_value", "slope"),
        param_values=(0.7, 1.3),
        block_name="Ramp(y0,m)",
    )


def test_grad_source_step_amplitudes():
    # Step has dynamic start_value/end_value and static step_time. After
    # step_time, output is end_value; before, start_value. Evaluate strictly
    # after the step so grad d(sum y)/d(end_value) ~ N (number of recorded
    # samples) and d/d(start_value) is 0.
    blk = Step(start_value=0.0, end_value=1.0, step_time=0.1)
    _source_grad_via_eval(
        blk, t_eval=0.4,
        param_setters=("start_value", "end_value"),
        param_values=(0.2, 1.5),
        block_name="Step(start,end)",
    )


def test_grad_source_pulse_amplitude():
    # Pulse: dynamic amplitude/period/pulse_width. Differentiating through
    # `period` and `pulse_width` is not stable across the discontinuity (the
    # block uses jnp.where with an integer-quantised threshold) so we
    # restrict to ``amplitude`` which is smooth wherever the simulator
    # samples.
    blk = Pulse(amplitude=1.0, pulse_width=0.5, period=0.4, phase_delay=0.0)
    _source_grad_via_eval(
        blk, t_eval=0.1,  # sample inside the first "high" half-cycle
        param_setters=("amplitude",),
        param_values=(1.3,),
        block_name="Pulse(amplitude)",
    )


def test_grad_source_sawtooth_amplitude():
    # Sawtooth has dynamic amplitude (and phase_delay), static frequency.
    # Within one period the signal is linear in (t - phi), so amplitude
    # gradients are smooth and FD-stable.
    blk = Sawtooth(amplitude=1.0, frequency=0.5, phase_delay=0.0)
    _source_grad_via_eval(
        blk, t_eval=0.3,  # well within the first period of length 2.0
        param_setters=("amplitude",),
        param_values=(1.5,),
        block_name="Sawtooth(amplitude)",
    )


# ── T-001b additions: explicit zero-gradient / non-differentiable blocks ─────
#
# These blocks emit integer/boolean outputs or are flagged as terminal events.
# The point of these tests is to make any future regression — e.g. a JAX-side
# change that quietly returns a non-zero gradient through what should be a
# step function — surface as a test failure rather than silent correctness
# loss. We do NOT assert "raises NotImplementedError"; today, JAX returns
# zero gradients through `jnp.round`, `jnp.where`-on-bool, etc.


def test_zero_gradient_quantizer():
    # y = interval * round(x / interval) — gradient of `round` is zero almost
    # everywhere (and undefined at exact half-integer boundaries). Pick an
    # interior point and assert AD returns 0.
    blk = Quantizer(interval=0.5)
    fwd = _loss(blk, [0.3])
    g = jax.grad(fwd)(jnp.array(0.3))
    assert float(g) == 0.0, f"Quantizer gradient should be 0, got {float(g)}"


def test_zero_gradient_logical_operator_and():
    # AND of two boolean-ish floats: the elementwise operation produces a bool
    # output. We cast the bool result to float and sum-reduce; the gradient
    # w.r.t. inputs must be exactly 0 (jnp.logical_* is non-differentiable).
    blk = LogicalOperator(function="and")
    inner = _block_output(blk, [True, True])

    def fwd(a, b):
        y = inner(a, b)
        return jnp.sum(jnp.asarray(y, dtype=jnp.float64))

    g = jax.grad(fwd, argnums=(0, 1))(jnp.array(1.0), jnp.array(1.0))
    assert all(float(gi) == 0.0 for gi in g), (
        f"LogicalOperator(and) gradient should be (0, 0), "
        f"got {tuple(float(gi) for gi in g)}"
    )


def test_zero_gradient_comparator():
    # Comparator output is a bool. Cast to float, sum-reduce → expect 0 grad
    # because the comparison result is discrete-valued.
    blk = Comparator(operator=">")
    inner = _block_output(blk, [1.0, 0.5])

    def fwd(a, b):
        y = inner(a, b)
        return jnp.sum(jnp.asarray(y, dtype=jnp.float64))

    g = jax.grad(fwd, argnums=(0, 1))(jnp.array(1.0), jnp.array(0.5))
    assert all(float(gi) == 0.0 for gi in g), (
        f"Comparator gradient should be (0, 0), "
        f"got {tuple(float(gi) for gi in g)}"
    )


def test_zero_gradient_signal_datatype_conversion_int():
    # Float → int conversion is non-differentiable. JAX returns a zero
    # gradient because the integer cast is treated as a no-op-on-tangent.
    blk = SignalDatatypeConversion(convert_to_type="int32")
    inner = _block_output(blk, [2.7])

    def fwd(x):
        y = inner(x)
        return jnp.sum(jnp.asarray(y, dtype=jnp.float64))

    g = jax.grad(fwd)(jnp.array(2.7))
    assert float(g) == 0.0, (
        f"SignalDatatypeConversion(int) gradient should be 0, got {float(g)}"
    )


def test_stop_block_has_no_output_to_differentiate():
    # `Stop` has only an input port and no output ports, so there is no
    # scalar to take a gradient of through this block — by construction it
    # cannot appear in a differentiable forward pass. Document that here
    # with an explicit skip so a future regression that adds an output port
    # will trip on this test.
    pytest.skip(
        "Stop block has no output port; not part of any differentiable forward "
        "path. T-001b explicit non-differentiability marker."
    )
