# SPDX-License-Identifier: MIT

"""T-120-followup-loop-blocks — ``ForLoop`` and ``WhileLoop`` containers.

Covers the iteration-container slice deferred from T-120 phase 1:

- ``ForLoop``: runs a body ``(i, carry) -> carry`` for ``n_iter``
  iterations via ``jax.lax.fori_loop``.
- ``WhileLoop``: runs a body ``carry -> carry`` while
  ``cond_fn(carry)`` is True, capped at ``max_iter`` via
  ``jax.lax.while_loop``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import ForLoop, WhileLoop
from jaxonomy.library import Constant
from jaxonomy.testing.markers import skip_if_not_jax


skip_if_not_jax()


# ---------------------------------------------------------------------------
# ForLoop
# ---------------------------------------------------------------------------


def _accumulate_index(i, carry):
    """body_fn for ForLoop: carry += i."""
    return carry + i.astype(carry.dtype)


def test_forloop_sums_indices():
    """body adds i to carry, run for n_iter=10 → 0+1+...+9 = 45."""
    blk = ForLoop(_accumulate_index, n_iter=10)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(0.0), name="carry0"))
    f = bld.add(blk)
    bld.connect(src.output_ports[0], f.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = f.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(45.0)


def test_forloop_zero_iter_is_identity():
    """n_iter=0 → output equals the initial carry."""
    blk = ForLoop(_accumulate_index, n_iter=0)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(7.5), name="carry0"))
    f = bld.add(blk)
    bld.connect(src.output_ports[0], f.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = f.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(7.5)


def test_forloop_invalid_n_iter_raises():
    with pytest.raises(ValueError, match="n_iter"):
        ForLoop(_accumulate_index, n_iter=-1)
    with pytest.raises(TypeError, match="n_iter"):
        ForLoop(_accumulate_index, n_iter=3.0)  # type: ignore[arg-type]


def test_forloop_grad_through_body_parameters():
    """Gradient flows through body parameters (initial carry & body
    closure constants).

    Body: ``carry -> carry * a`` for n_iter=4 → carry * a**4.
    d/dcarry = a**4, d/da = 4 * carry * a**3.
    """
    a = jnp.array(2.0)
    n = 4

    def loss(carry0):
        def body(i, c):
            del i
            return c * a
        blk = ForLoop(body, n_iter=n)
        # Run the underlying compute_output directly (no diagram needed
        # for a pure-grad test of the body parameter path).
        return blk._compute_output(0.0, None, carry0)

    g = jax.grad(loss)(jnp.array(3.0))
    expected = float(a) ** n  # 2**4 = 16
    assert float(g) == pytest.approx(expected)


def test_forloop_grad_through_diagram_initial_carry():
    """Grad of diagram output w.r.t. the Constant carry feeding ForLoop."""

    def body(i, c):
        return c + i.astype(c.dtype)

    blk = ForLoop(body, n_iter=5)  # carry + (0+1+2+3+4) = carry + 10
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(1.0), name="carry0"))
    f = bld.add(blk)
    bld.connect(src.output_ports[0], f.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def loss(c0):
        ctx = ctx0.with_subcontext(
            src.system_id,
            ctx0[src.system_id].with_parameter("value", c0),
        )
        return f.output_ports[0].eval(ctx)

    val = float(loss(jnp.array(1.0)))
    assert val == pytest.approx(11.0)
    g = float(jax.grad(loss)(jnp.array(1.0)))
    assert g == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# WhileLoop
# ---------------------------------------------------------------------------


def _inc(carry):
    return carry + jnp.asarray(1.0, dtype=carry.dtype)


def _below_five(carry):
    return carry < 5.0


def test_whileloop_runs_until_condition_false():
    """Body increments by 1; condition carry<5; start at 0 → final = 5."""
    blk = WhileLoop(_inc, _below_five, max_iter=100)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(0.0), name="carry0"))
    w = bld.add(blk)
    bld.connect(src.output_ports[0], w.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = w.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(5.0)


def test_whileloop_max_iter_cap():
    """Condition always True; max_iter=10; start at 0 → exits at 10."""

    def always_true(carry):
        del carry
        return jnp.asarray(True)

    blk = WhileLoop(_inc, always_true, max_iter=10)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(0.0), name="carry0"))
    w = bld.add(blk)
    bld.connect(src.output_ports[0], w.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = w.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(10.0)


def test_whileloop_initial_condition_false_is_identity():
    """If the condition is False at entry, carry passes through."""
    blk = WhileLoop(_inc, _below_five, max_iter=100)
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(7.0), name="carry0"))
    w = bld.add(blk)
    bld.connect(src.output_ports[0], w.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = w.output_ports[0].eval(ctx)
    assert float(y) == pytest.approx(7.0)


def test_whileloop_invalid_max_iter_raises():
    with pytest.raises(ValueError, match="max_iter"):
        WhileLoop(_inc, _below_five, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        WhileLoop(_inc, _below_five, max_iter=-3)
    with pytest.raises(TypeError, match="max_iter"):
        WhileLoop(_inc, _below_five, max_iter=10.0)  # type: ignore[arg-type]


def test_whileloop_jit_compatible():
    """The ``while_loop`` plumbing must compile under jit."""
    blk = WhileLoop(_inc, _below_five, max_iter=100)

    @jax.jit
    def run(c0):
        return blk._compute_output(0.0, None, c0)

    y = run(jnp.array(0.0))
    assert float(y) == pytest.approx(5.0)


def test_whileloop_pytree_carry():
    """Carry can be a pytree (e.g. a dict)."""

    def body(carry):
        return {"x": carry["x"] + 1.0, "y": carry["y"] * 2.0}

    def cond(carry):
        return carry["x"] < 3.0

    blk = WhileLoop(body, cond, max_iter=50)
    out = blk._compute_output(
        0.0, None, {"x": jnp.array(0.0), "y": jnp.array(1.0)}
    )
    # Iterations: x: 0→1→2→3 (3 steps); y: 1→2→4→8.
    assert float(out["x"]) == pytest.approx(3.0)
    assert float(out["y"]) == pytest.approx(8.0)
