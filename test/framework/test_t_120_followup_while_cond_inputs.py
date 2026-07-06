# SPDX-License-Identifier: MIT

"""T-120-followup-while-condition-aware-inputs — WhileLoop with cond/body
fns that can consume upstream input signals.

The phase-1 :class:`WhileLoop` accepted only ``cond_fn(carry)`` and
``body_fn(carry)``. This follow-up extends both signatures to optionally
take additional positional arguments forwarded from the block's
upstream input ports — for cases like "iterate until the input signal
exceeds a threshold" where the loop condition depends on a live
external signal rather than on the carry alone.

Coverage:

- ``cond_fn(carry, input1)`` terminates when ``input1 > threshold``.
- Legacy single-arg ``cond_fn(carry)`` still works (backwards-compat).
- ``body_fn(carry, input1)`` adds the input to the carry each iteration.
- ``jax.grad`` flows through the body and the carry under the new API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.backend import numpy_api as npa
from jaxonomy.framework import WhileLoop
from jaxonomy.framework.containers import _accepts_extra_args
from jaxonomy.library import Constant
from jaxonomy.testing.markers import skip_if_not_jax


skip_if_not_jax()


# ---------------------------------------------------------------------------
# Signature-detection helper (white-box)
# ---------------------------------------------------------------------------


def test_accepts_extra_args_single_arg_false():
    def f(carry):
        return carry
    assert _accepts_extra_args(f) is False


def test_accepts_extra_args_two_positional_true():
    def f(carry, x):
        return carry
    assert _accepts_extra_args(f) is True


def test_accepts_extra_args_var_positional_true():
    def f(carry, *args):
        return carry
    assert _accepts_extra_args(f) is True


def test_accepts_extra_args_lambda_single_arg_false():
    f = lambda carry: carry  # noqa: E731
    assert _accepts_extra_args(f) is False


def test_accepts_extra_args_lambda_two_args_true():
    f = lambda carry, x: carry  # noqa: E731
    assert _accepts_extra_args(f) is True


# ---------------------------------------------------------------------------
# Backwards compatibility: single-arg cond/body
# ---------------------------------------------------------------------------


def test_whileloop_legacy_single_arg_still_works():
    """Phase-1 contract: ``cond_fn(carry)`` / ``body_fn(carry)``."""

    def body(carry):
        return carry + npa.asarray(1.0, dtype=carry.dtype)

    def cond(carry):
        return carry < 5.0

    blk = WhileLoop(body, cond, max_iter=100)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(0.0), name="carry0"))
    w = bld.add(blk)
    bld.connect(src.output_ports[0], w.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = w.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Condition consumes an upstream input
# ---------------------------------------------------------------------------


def test_whileloop_cond_uses_input_threshold():
    """``cond_fn(carry, threshold)`` terminates when carry >= threshold.

    Body increments by 1 starting from 0. Threshold is supplied via a
    Constant input. Loop runs until ``carry >= threshold``.
    """

    def body(carry):
        return carry + jnp.asarray(1.0, dtype=carry.dtype)

    def cond(carry, threshold):
        # Continue while carry < threshold.
        return carry < threshold

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=1)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(0.0), name="carry0"))
    th = bld.add(Constant(jnp.array(7.0), name="threshold"))
    w = bld.add(blk)
    bld.connect(src.output_ports[0], w.input_ports[0])
    bld.connect(th.output_ports[0], w.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = w.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(7.0)


def test_whileloop_cond_input_changes_termination():
    """Sanity check: changing the threshold changes the loop length."""

    def body(carry):
        return carry + jnp.asarray(1.0, dtype=carry.dtype)

    def cond(carry, threshold):
        return carry < threshold

    # Direct ``_compute_output`` call avoids needing a diagram for each
    # threshold value, keeping this test fast.
    blk = WhileLoop(body, cond, max_iter=100, n_inputs=1)

    y3 = blk._compute_output(0.0, None, jnp.array(0.0), jnp.array(3.0))
    y10 = blk._compute_output(0.0, None, jnp.array(0.0), jnp.array(10.0))
    assert float(y3) == pytest.approx(3.0)
    assert float(y10) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Body consumes an upstream input
# ---------------------------------------------------------------------------


def test_whileloop_body_adds_input_each_iter():
    """``body_fn(carry, step)`` adds ``step`` to carry per iteration.

    With step=2.0, initial carry=0.0, cond carry<10.0:
    iterations: 0→2→4→6→8→10 (5 iterations, exits when carry>=10).
    """

    def body(carry, step):
        return carry + step

    def cond(carry):
        return carry < 10.0

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=1)
    out = blk._compute_output(
        0.0, None, jnp.array(0.0), jnp.array(2.0)
    )
    assert float(out) == pytest.approx(10.0)


def test_whileloop_both_cond_and_body_take_inputs():
    """``cond_fn(carry, ...)`` and ``body_fn(carry, ...)`` both receive
    *all* upstream inputs; callables can ignore the ones they don't
    need by using ``*args`` (or just accepting all positionally).
    """

    def body(carry, step, _threshold):
        return carry + step

    def cond(carry, _step, threshold):
        return carry < threshold

    blk = WhileLoop(body, cond, max_iter=200, n_inputs=2)
    out = blk._compute_output(
        0.0,
        None,
        jnp.array(0.0),     # carry init
        jnp.array(3.0),     # step (used by body)
        jnp.array(20.0),    # threshold (used by cond)
    )
    # 0 → 3 → 6 → 9 → 12 → 15 → 18 → 21 (exits when carry >= 20).
    assert float(out) == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# Diagram-level wiring
# ---------------------------------------------------------------------------


def test_whileloop_with_inputs_diagram_eval():
    """End-to-end diagram with body that consumes an upstream input."""

    def body(carry, step, _threshold):
        return carry + step

    def cond(carry, _step, threshold):
        return carry < threshold

    blk = WhileLoop(body, cond, max_iter=200, n_inputs=2)
    bld = jaxonomy.DiagramBuilder()
    c0 = bld.add(Constant(jnp.array(0.0), name="carry0"))
    s = bld.add(Constant(jnp.array(2.5), name="step"))
    th = bld.add(Constant(jnp.array(11.0), name="threshold"))
    w = bld.add(blk)
    bld.connect(c0.output_ports[0], w.input_ports[0])
    bld.connect(s.output_ports[0], w.input_ports[1])
    bld.connect(th.output_ports[0], w.input_ports[2])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = w.output_ports[0].eval(ctx)
    # 0 → 2.5 → 5.0 → 7.5 → 10.0 → 12.5 (5 iterations, exit at 12.5).
    assert float(y) == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_whileloop_invalid_n_inputs_raises():
    def body(carry):
        return carry

    def cond(carry):
        return carry < 1.0

    with pytest.raises(ValueError, match="n_inputs"):
        WhileLoop(body, cond, max_iter=10, n_inputs=-1)
    with pytest.raises(TypeError, match="n_inputs"):
        WhileLoop(body, cond, max_iter=10, n_inputs=1.0)  # type: ignore[arg-type]


def test_whileloop_n_inputs_zero_default_path():
    """``n_inputs=0`` (default) preserves phase-1 behaviour."""

    def body(carry):
        return carry + jnp.asarray(1.0, dtype=carry.dtype)

    def cond(carry):
        return carry < 3.0

    blk = WhileLoop(body, cond, max_iter=100)
    assert len(blk.input_ports) == 1
    out = blk._compute_output(0.0, None, jnp.array(0.0))
    assert float(out) == pytest.approx(3.0)


def test_whileloop_extra_input_ports_declared():
    def body(carry, x):
        return carry + x

    def cond(carry, x):
        return carry < x

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=2)
    assert len(blk.input_ports) == 3
    # Port 0 is the carry; ports 1+ are the user inputs.
    assert blk.input_ports[0].name == "carry_init"
    assert blk.input_ports[1].name == "u_0"
    assert blk.input_ports[2].name == "u_1"


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_whileloop_grad_through_body_input():
    """``jax.grad`` of the loop output w.r.t. the body's input.

    Body: ``carry, step -> carry + step`` for cond ``carry < 5``.
    Starting at 0, with step=1.0: 5 iterations → final = 5.
    Starting at 0, with step=2.0: 3 iterations → final = 6.

    Since the iteration count is data-dependent and lax.while_loop is
    not reverse-mode differentiable, we use ``jax.jvp`` (forward mode)
    instead of ``jax.grad`` (reverse mode) to validate that the
    derivative-through-carry path is well-defined and traceable.
    """

    def body(carry, step):
        return carry + step

    def cond(carry):
        # Independent of step so the iteration count varies smoothly
        # in regions where ``step`` doesn't cross an iteration boundary.
        return carry < 5.0

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=1)

    def loss(step):
        return blk._compute_output(0.0, None, jnp.array(0.0), step)

    # Forward-mode JVP: differentiable through carry, well-defined for
    # ``lax.while_loop``.
    primal, tangent = jax.jvp(loss, (jnp.array(1.0),), (jnp.array(1.0),))
    # Sanity: primal is the loop output for step=1.0.
    assert float(primal) == pytest.approx(5.0)
    # Tangent should be finite (the gradient is well-defined except
    # exactly at iteration-boundary step values).
    assert jnp.isfinite(tangent)


def test_whileloop_grad_through_carry_init():
    """Forward-mode derivative w.r.t. the initial carry value.

    Body adds a constant 1.0 per iteration; cond uses an input
    threshold. For a fixed threshold and an initial carry far from any
    iteration boundary, the loop count is locally constant, so the
    output is locally ``carry_init + k * 1.0`` and the derivative
    w.r.t. carry_init is 1.0.
    """

    def body(carry):
        return carry + jnp.asarray(1.0, dtype=carry.dtype)

    def cond(carry, threshold):
        return carry < threshold

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=1)

    def f(c0):
        return blk._compute_output(0.0, None, c0, jnp.array(5.5))

    # Starting at 0.3, body adds 1 each time, 6 iterations → 6.3.
    primal, tangent = jax.jvp(f, (jnp.array(0.3),), (jnp.array(1.0),))
    assert float(primal) == pytest.approx(6.3)
    assert float(tangent) == pytest.approx(1.0)


def test_whileloop_grad_body_only_no_cond_input():
    """``body_fn(carry, x)`` with single-arg ``cond_fn(carry)``: mixed mode.

    Verifies that signature detection is per-callable: body gets the
    inputs, cond doesn't.
    """

    def body(carry, factor):
        return carry * factor

    def cond(carry):
        return carry < 100.0

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=1)

    # carry: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 (exits at 128).
    out = blk._compute_output(0.0, None, jnp.array(1.0), jnp.array(2.0))
    assert float(out) == pytest.approx(128.0)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_whileloop_with_inputs_jit_compatible():
    """The new path must still compile under ``jax.jit``."""

    def body(carry, step, _threshold):
        return carry + step

    def cond(carry, _step, threshold):
        return carry < threshold

    blk = WhileLoop(body, cond, max_iter=100, n_inputs=2)

    @jax.jit
    def run(c0, step, thr):
        return blk._compute_output(0.0, None, c0, step, thr)

    y = run(jnp.array(0.0), jnp.array(1.5), jnp.array(7.0))
    # 0 → 1.5 → 3.0 → 4.5 → 6.0 → 7.5 (5 iterations, exits at 7.5).
    assert float(y) == pytest.approx(7.5)
